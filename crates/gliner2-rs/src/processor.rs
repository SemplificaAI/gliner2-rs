// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Prompt construction and tokenization.
//!
//! Replicates `gliner2/processor.py::SchemaTransformer` from the gliner2 2.0.0
//! package. The token layout of one schema group is exactly:
//!
//! ```text
//! ["(", "[P]", prompt_str, "("] + [child_prefix, field_name] * N + [")", ")"]
//! ```
//!
//! with `child_prefix` being `[E]` for entities, `[R]` for relations and `[L]`
//! for classifications. Groups are separated by `[SEP_STRUCT]`, and
//! `[SEP_TEXT]` separates the schema from the text.
//!
//! ## Three details that are easy to get wrong
//!
//! 1. **No `[CLS]`/`[SEP]`.** The sequence starts directly at `(` and ends with
//!    the last word of the text. Wrapping the input feeds the model a format it
//!    was never trained on.
//!
//! 2. **Field indices point at the marker, not at the label.** Python collects
//!    `schema_special_positions` at the positions of `[P]` and of the
//!    `child_prefix` markers, not at the field *name* that follows.
//!
//! 3. **Text words must be lower-cased.** `_tokenize_text` calls
//!    `word_splitter(text, lower=True)`; schema tokens are left alone.
//!    Character offsets keep indexing the original text, so extracted spans
//!    preserve their original casing.
//!
//! Verified against ground truth from gliner2 2.0.0: for
//! `entities(person, organization, location) + classification(sentiment, …)`
//! the expected positions are `[[1, 6, 8, 10], [18, 21, 23]]`.
//! See the `ground_truth_layout` test at the bottom of this file.

use anyhow::{Result, anyhow};
use regex::Regex;
use tokenizers::Tokenizer;

pub const SEP_STRUCT: &str = "[SEP_STRUCT]";
pub const SEP_TEXT: &str = "[SEP_TEXT]";
pub const P_TOKEN: &str = "[P]";
pub const E_TOKEN: &str = "[E]";
pub const R_TOKEN: &str = "[R]";
pub const L_TOKEN: &str = "[L]";
pub const DESC_TOKEN: &str = "[DESCRIPTION]";

/// One schema task.
#[derive(Debug, Clone)]
pub enum SchemaTask {
    /// Entity extraction. Labels become `[E]` markers.
    Entities(Vec<String>),
    /// Relation extraction: relation name plus roles (`head`, `tail`).
    /// Roles become `[R]` markers.
    Relations(String, Vec<String>),
    /// Classification: task name plus choices. Choices become `[L]` markers.
    ///
    /// `multi_label` decides how the logits are normalised: independent
    /// sigmoids when true, a softmax over the choices when false. gliner2
    /// carries this per task, not per request — `prompt_safety` is
    /// single-label while `prompt_toxicity` and `jailbreak_detection` are
    /// multi-label, and they are routinely passed in the same call.
    Classifications {
        task: String,
        labels: Vec<String>,
        multi_label: bool,
    },
}

impl SchemaTask {
    /// Single-label classification (softmax over the choices).
    pub fn classification(task: impl Into<String>, labels: Vec<String>) -> Self {
        Self::Classifications { task: task.into(), labels, multi_label: false }
    }

    /// Multi-label classification (independent sigmoids).
    pub fn multi_label_classification(task: impl Into<String>, labels: Vec<String>) -> Self {
        Self::Classifications { task: task.into(), labels, multi_label: true }
    }
}

impl SchemaTask {
    fn task_type(&self) -> TaskType {
        match self {
            Self::Entities(_) => TaskType::Entities,
            Self::Relations(..) => TaskType::Relations,
            Self::Classifications { .. } => TaskType::Classifications,
        }
    }

    fn child_prefix(&self) -> &'static str {
        match self {
            Self::Entities(_) => E_TOKEN,
            Self::Relations(..) => R_TOKEN,
            Self::Classifications { .. } => L_TOKEN,
        }
    }

    /// The `prompt_str` that follows `[P]`.
    fn prompt_str(&self) -> String {
        match self {
            // Python uses the literal "entities" as the group name.
            Self::Entities(_) => "entities".to_string(),
            Self::Relations(name, _) => name.clone(),
            Self::Classifications { task, .. } => task.clone(),
        }
    }

    fn fields(&self) -> &[String] {
        match self {
            Self::Entities(v) => v,
            Self::Relations(_, v) => v,
            Self::Classifications { labels, .. } => labels,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Entities,
    Relations,
    Classifications,
}

/// One schema group, with the sub-token positions of its markers.
#[derive(Debug, Clone)]
pub struct TaskMapping {
    pub task_name: String,
    pub task_type: TaskType,
    pub labels: Vec<String>,
    /// Sub-token position of the group's `[P]` marker.
    pub prompt_tok_idx: usize,
    /// Sub-token positions of the child markers (`[E]`/`[R]`/`[L]`), one per label.
    pub field_tok_indices: Vec<usize>,
    /// For classification groups, whether the choices are scored independently.
    pub multi_label: bool,
}

/// Tokenization result: everything the engine needs.
#[derive(Debug, Clone)]
pub struct ProcessedRecord {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub tasks: Vec<TaskMapping>,

    /// For each word of the text, its sub-token range `[start, end)`.
    pub word_to_token_maps: Vec<(usize, usize)>,
    /// For each word of the text, its byte range `[start, end)` in the original text.
    pub word_to_char_maps: Vec<(usize, usize)>,
}

impl ProcessedRecord {
    pub fn num_words(&self) -> usize {
        self.word_to_token_maps.len()
    }

    /// Index of each word's first sub-token (`token_pooling = "first"`).
    pub fn word_first_positions(&self) -> Vec<i64> {
        self.word_to_token_maps.iter().map(|(s, _)| *s as i64).collect()
    }

    /// Boundary query markers: the children of the extractive groups
    /// (`entities` and `relations`), in schema order.
    ///
    /// Returns the positions and, for each, `(group index, role index)` so a
    /// result can be traced back to the field that produced it.
    pub fn query_markers(&self) -> (Vec<i64>, Vec<(usize, usize)>) {
        let mut positions = Vec::new();
        let mut specs = Vec::new();
        for (g, task) in self.tasks.iter().enumerate() {
            if task.task_type == TaskType::Classifications {
                continue;
            }
            for (r, idx) in task.field_tok_indices.iter().enumerate() {
                positions.push(*idx as i64);
                specs.push((g, r));
            }
        }
        (positions, specs)
    }

    /// Classification choice markers (`[L]`), in schema order.
    pub fn cls_markers(&self) -> (Vec<i64>, Vec<(usize, usize)>) {
        let mut positions = Vec::new();
        let mut specs = Vec::new();
        for (g, task) in self.tasks.iter().enumerate() {
            if task.task_type != TaskType::Classifications {
                continue;
            }
            for (c, idx) in task.field_tok_indices.iter().enumerate() {
                positions.push(*idx as i64);
                specs.push((g, c));
            }
        }
        (positions, specs)
    }
}

/// Word splitter equivalent to gliner2's `WhitespaceTokenSplitter`.
#[derive(Clone, Debug)]
pub struct WhitespaceTokenSplitter {
    re: Regex,
}

impl WhitespaceTokenSplitter {
    pub fn new() -> Result<Self> {
        let re = Regex::new(
            r"(?xi)
            (?:https?://[^\s]+|www\.[^\s]+)
            |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
            |@[a-z0-9_]+
            |\w+(?:[-_]\w+)*
            |\S
        ",
        )?;
        Ok(Self { re })
    }

    /// Mirrors gliner2's `WhitespaceTokenSplitter.__call__(text, lower=True)`:
    /// matching happens on the **original** text, so offsets index the
    /// caller's string, and only the token value is lower-cased.
    /// Lower-casing the source first would be unsafe: Unicode case folding
    /// can change its length (e.g. "İ" -> "i\u{0307}") and corrupt the
    /// offsets.
    pub fn split_with_offsets(&self, text: &str) -> Vec<(String, usize, usize)> {
        self.re
            .find_iter(text)
            .map(|m| (m.as_str().to_lowercase(), m.start(), m.end()))
            .collect()
    }
}

pub struct SchemaTransformer {
    tokenizer: Tokenizer,
    splitter: WhitespaceTokenSplitter,
}

/// One prompt token, tagged with whether its position must be tracked.
enum Slot {
    /// Untracked structural token: `(`, `)`, prompt_str, a field name…
    Plain(String),
    /// `[P]` marker of the given group.
    Prompt(String, usize),
    /// Child marker of the given group and role.
    Field(String, usize, usize),
    /// A word of the text, with its byte offsets in the original string.
    Word(String, usize, usize),
}

impl SchemaTransformer {
    pub fn new(tokenizer: Tokenizer) -> Result<Self> {
        Ok(Self { tokenizer, splitter: WhitespaceTokenSplitter::new()? })
    }

    pub fn from_tokenizer_file(path: &std::path::Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("could not read {}: {e}", path.display()))?;
        Self::new(tokenizer)
    }

    /// Builds the prompt and tokenizes it.
    ///
    /// `descriptions` maps, per group, `label -> description`; the ones present
    /// are appended to `prompt_str` as `[DESCRIPTION] label: desc`, exactly as
    /// `_transform_schema` does.
    pub fn transform(&self, text: &str, tasks: &[SchemaTask]) -> Result<ProcessedRecord> {
        self.transform_with_descriptions(text, tasks, &[])
    }

    pub fn transform_with_descriptions(
        &self,
        text: &str,
        tasks: &[SchemaTask],
        descriptions: &[Vec<(String, String)>],
    ) -> Result<ProcessedRecord> {
        let mut slots: Vec<Slot> = Vec::new();

        for (g, task) in tasks.iter().enumerate() {
            let mut prompt_str = task.prompt_str();
            if let Some(descs) = descriptions.get(g) {
                let fields = task.fields();
                for (label, desc) in descs {
                    if fields.iter().any(|f| f == label) {
                        prompt_str.push_str(&format!(" {DESC_TOKEN} {label}: {desc}"));
                    }
                }
            }

            slots.push(Slot::Plain("(".into()));
            slots.push(Slot::Prompt(P_TOKEN.into(), g));
            slots.push(Slot::Plain(prompt_str));
            slots.push(Slot::Plain("(".into()));
            for (r, field) in task.fields().iter().enumerate() {
                slots.push(Slot::Field(task.child_prefix().into(), g, r));
                slots.push(Slot::Plain(field.clone()));
            }
            slots.push(Slot::Plain(")".into()));
            slots.push(Slot::Plain(")".into()));

            if g + 1 < tasks.len() {
                slots.push(Slot::Plain(SEP_STRUCT.into()));
            }
        }

        slots.push(Slot::Plain(SEP_TEXT.into()));
        for (w, start, end) in self.splitter.split_with_offsets(text) {
            slots.push(Slot::Word(w, start, end));
        }

        // ── tokenize, tracking marker and word positions ───────────────────
        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_to_token_maps = Vec::new();
        let mut word_to_char_maps = Vec::new();

        let mut prompt_positions: Vec<Option<usize>> = vec![None; tasks.len()];
        let mut field_positions: Vec<Vec<usize>> =
            tasks.iter().map(|t| vec![0usize; t.fields().len()]).collect();

        for slot in &slots {
            let piece = match slot {
                Slot::Plain(s) => s.as_str(),
                Slot::Prompt(s, _) => s.as_str(),
                Slot::Field(s, ..) => s.as_str(),
                Slot::Word(s, ..) => s.as_str(),
            };
            let start = input_ids.len();

            // `add_special_tokens = false`: the prompt is not wrapped in
            // [CLS]/[SEP], matching gliner2.
            let enc = self
                .tokenizer
                .encode(piece, false)
                .map_err(|e| anyhow!("tokenization failed for {piece:?}: {e}"))?;
            input_ids.extend(enc.get_ids().iter().map(|&id| id as i64));
            let end = input_ids.len();

            match slot {
                Slot::Prompt(_, g) => prompt_positions[*g] = Some(start),
                Slot::Field(_, g, r) => field_positions[*g][*r] = start,
                Slot::Word(_, cs, ce) => {
                    // A word producing no sub-tokens still keeps a row, so that
                    // words and embeddings stay aligned one to one.
                    word_to_token_maps.push((start, end.max(start + 1)));
                    word_to_char_maps.push((*cs, *ce));
                }
                Slot::Plain(_) => {}
            }
        }

        let attention_mask = vec![1i64; input_ids.len()];

        let mapped_tasks = tasks
            .iter()
            .enumerate()
            .map(|(g, task)| {
                Ok(TaskMapping {
                    task_name: task.prompt_str(),
                    task_type: task.task_type(),
                    labels: task.fields().to_vec(),
                    prompt_tok_idx: prompt_positions[g]
                        .ok_or_else(|| anyhow!("missing [P] marker for group {g}"))?,
                    field_tok_indices: field_positions[g].clone(),
                    multi_label: matches!(
                        task,
                        SchemaTask::Classifications { multi_label: true, .. }
                    ),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(ProcessedRecord {
            input_ids,
            attention_mask,
            tasks: mapped_tasks,
            word_to_token_maps,
            word_to_char_maps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locates a gliner2 `tokenizer.json`. Model directories are gitignored, so
    /// the test skips instead of failing when none is available.
    fn tokenizer_path() -> Option<std::path::PathBuf> {
        // Walk up to the workspace root: model directories are gitignored and
        // shared by every crate, so they live there rather than per crate.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .to_path_buf();
        [
            "models/tokenizer.json",
            "models/pii-onnx/tokenizer.json",
            "models/pii-legacy/fp16_v2/tokenizer.json",
            "models/guardrails-pii-multi-onnx/tokenizer.json",
            "models/gliner2.5-multi-v1-onnx/tokenizer.json",
        ]
        .iter()
        .map(|p| root.join(p))
        .find(|p| p.exists())
    }

    /// Ground truth produced by gliner2 2.0.0:
    ///
    /// ```text
    /// schema_tokens_list = [
    ///   ['(', '[P]', 'entities', '(', '[E]', 'person', '[E]', 'organization', '[E]', 'location', ')', ')'],
    ///   ['(', '[P]', 'sentiment', '(', '[L]', 'positive', '[L]', 'negative', ')', ')'],
    /// ]
    /// schema_special_positions  = [[1, 6, 8, 10], [18, 21, 23]]
    /// text_word_first_positions = [31, 33, 35, 36, 37, 38, 40, 43]
    /// ```
    #[test]
    fn ground_truth_layout() {
        let Some(path) = tokenizer_path() else {
            eprintln!("no tokenizer.json available, test skipped");
            return;
        };
        let tf = SchemaTransformer::from_tokenizer_file(&path).unwrap();
        let tasks = vec![
            SchemaTask::Entities(vec![
                "person".into(),
                "organization".into(),
                "location".into(),
            ]),
            SchemaTask::classification(
                "sentiment",
                vec!["positive".into(), "negative".into()],
            ),
        ];
        let rec = tf
            .transform("Mario Rossi lavora ad Apple a Cupertino.", &tasks)
            .unwrap();

        assert_eq!(rec.tasks[0].prompt_tok_idx, 1);
        assert_eq!(rec.tasks[0].field_tok_indices, vec![6, 8, 10]);
        assert_eq!(rec.tasks[1].prompt_tok_idx, 18);
        assert_eq!(rec.tasks[1].field_tok_indices, vec![21, 23]);
        assert_eq!(
            rec.word_first_positions(),
            vec![31, 33, 35, 36, 37, 38, 40, 43]
        );
        assert_eq!(rec.num_words(), 8);

        // boundary queries are the extractive markers only
        let (qpos, qspec) = rec.query_markers();
        assert_eq!(qpos, vec![6, 8, 10]);
        assert_eq!(qspec, vec![(0, 0), (0, 1), (0, 2)]);
        let (cpos, _) = rec.cls_markers();
        assert_eq!(cpos, vec![21, 23]);
    }
}
