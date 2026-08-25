// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Engine for the GLiNER2 **span** architecture, including
//! `fastino/gliner2-privacy-filter-PII-multi`.
//!
//! Chain of the eight fragments produced by `export_span_v3.py`:
//!
//! ```text
//! encoder(input_ids, attention_mask) -> last_hidden_state [1,S,H]
//!   |
//!   +- token_gather(last_hidden_state, word_indices) -> text_embs [1,W,H]
//!   |     +- span_rep(text_embs, span_idx) -> span_embeddings [1,W,mw,H]
//!   |
//!   +- schema_gather(last_hidden_state, schema_indices) -> pc_emb [1,H], field_embs [M,H]
//!         +- count_pred_argmax(pc_emb)    -> pred_count  i64
//!         +- count_lstm_fixed(field_embs) -> struct_proj [MAX_COUNT,M,H]
//!               +- scorer(span_embeddings, struct_proj) -> entity_scores [MAX_COUNT,W,mw,M]
//!
//! classifier(field_embs) -> logits [M]
//! ```
//!
//! Span `[w][k]` covers words `w` through `w+k` **inclusive**, and is valid only
//! while `w + k < W`. Invalid spans are zeroed to `(0,0)` before `span_rep`, as
//! `SpanExtractorModel.compute_span_rep` does, and discarded afterwards.
//!
//! Note the difference from the boundary architecture of GLiNER2.5, whose spans
//! are half-open `[start, end)`.

use anyhow::{Result, anyhow};
use std::path::PathBuf;

use ort::session::Session;

use gliner_core::error::GlinerError;
use gliner_core::overlap::{OverlapPolicy, Spanned, resolve_overlaps};
use gliner_core::processor::{ProcessedRecord, SchemaTask, SchemaTransformer, TaskType};
use gliner_core::runtime::{
    IoDType, Precision, build_session, float_tensor, i64_tensor, resolve_fragment,
    resolve_tokenizer, sigmoid, softmax, take_float, take_i64,
};

/// Inference parameters, adjustable on every call.
#[derive(Debug, Clone)]
pub struct InferenceParams {
    /// Confidence threshold on entities.
    pub threshold: f32,
    /// When `true`, greedy NMS removes every overlap regardless of label.
    /// When `false`, overlapping spans with different labels coexist
    /// (e.g. "Mario Rossi" as `person` and "Mario" as `first_name`).
    pub flat_ner: bool,
    /// Temperature applied to the classification logits.
    pub classification_temperature: f32,
    /// Overrides the per-task `multi_label` flag carried by
    /// [`SchemaTask::Classifications`]. Leave it `None` — the schema is the
    /// right place for that decision, since a single request routinely mixes
    /// single-label and multi-label tasks.
    pub multi_label_override: Option<bool>,
    /// Shared overlap policy. `None` (the default) selects the historical
    /// greedy decoder of the span architecture, governed by `flat_ner`; that is
    /// the same choice gliner2 2.0.0 makes when `overlap_policy` is left
    /// unspecified on a span checkpoint. Naming one switches to the resolver
    /// shared with the boundary architecture, and `flat_ner` is then ignored.
    pub overlap_policy: Option<OverlapPolicy>,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            flat_ner: false,
            classification_temperature: 1.0,
            multi_label_override: None,
            overlap_policy: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub score: f32,
    /// Byte range `[start, end)` in the original text.
    pub char_start: usize,
    pub char_end: usize,
    /// Inclusive word range `[start, end]`.
    pub word_start: usize,
    pub word_end: usize,
    /// Occurrence slot produced by `count_lstm` (0 = first occurrence).
    pub slot: usize,
    /// Schema group that produced the entity.
    pub task: String,
}

impl Spanned for Entity {
    fn start(&self) -> usize {
        self.word_start
    }
    /// Span-architecture spans are **inclusive**; the shared resolver works on
    /// half-open intervals, so the conversion happens here.
    fn end(&self) -> usize {
        self.word_end + 1
    }
    fn score(&self) -> f32 {
        self.score
    }
}

#[derive(Debug, Clone)]
pub struct Classification {
    pub task: String,
    pub label: String,
    pub score: f32,
}

/// Result of one span extraction.
#[derive(Debug, Clone, Default)]
pub struct SpanOutput {
    pub entities: Vec<Entity>,
    /// Every label of every classification task, with its probability. Use
    /// [`SpanOutput::verdict`] to turn one task into the answer gliner2 would
    /// give.
    pub classifications: Vec<Classification>,
}

impl SpanOutput {
    /// The labels gliner2 would report for a classification task.
    ///
    /// Reproduces `_extract_classification_result`, including the detail that
    /// is easy to miss: in multi-label mode, **when no label clears the
    /// threshold the top-scoring one is returned anyway**. The list is never
    /// empty. Thresholding the scores yourself and keeping the empty result
    /// will silently disagree with the reference implementation — for a
    /// jailbreak schema, a genuine attack scoring 0.39 against a 0.4 threshold
    /// is reported by gliner2, not dropped.
    ///
    /// In single-label mode the argmax is returned, which is the same thing.
    pub fn verdict(&self, task: &str, threshold: f32) -> Vec<&Classification> {
        let mut rows: Vec<&Classification> =
            self.classifications.iter().filter(|c| c.task == task).collect();
        if rows.is_empty() {
            return rows;
        }
        rows.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        let over: Vec<&Classification> =
            rows.iter().copied().filter(|c| c.score >= threshold).collect();
        if over.is_empty() { vec![rows[0]] } else { over }
    }
}

#[derive(Debug, Clone)]
pub struct SpanConfig {
    pub models_dir: PathBuf,
    pub precision: Precision,
    pub intra_threads: usize,
}

impl SpanConfig {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        let models_dir = models_dir.into();
        let precision = Precision::autodetect(&models_dir, "encoder");
        Self { models_dir, precision, intra_threads: 4 }
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_intra_threads(mut self, n: usize) -> Self {
        self.intra_threads = n;
        self
    }
}

pub struct SpanEngine {
    encoder: Session,
    token_gather: Session,
    span_rep: Session,
    schema_gather: Session,
    count_pred: Session,
    count_lstm: Session,
    scorer: Session,
    classifier: Session,

    transformer: SchemaTransformer,
    dtype: IoDType,
    max_width: usize,
    max_count: usize,
    hidden: usize,
    classifier_layout: ClassifierLayout,
}

/// How the exported classifier expects its input.
///
/// The model applies the classifier to the choice-marker embeddings,
/// `[M, H] -> [M]`, and `export_span_v3.py` exports exactly that. Exports made
/// by the earlier script instead carry `[batch, num_labels, max_width, H] ->
/// [batch, num_labels, max_width, 1]`, with `max_width` frozen at export time:
/// correct only because the MLP acts on the last dimension, but it forces the
/// caller to replicate the same vector `max_width` times and discard all but
/// one slice. Both are supported, detected from the input rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierLayout {
    /// `[num_labels, H] -> [num_labels]`
    Flat,
    /// `[batch, num_labels, max_width, H] -> [batch, num_labels, max_width, 1]`
    Padded,
}

impl SpanEngine {
    pub fn new(config: SpanConfig) -> Result<Self> {
        let dir = &config.models_dir;
        let precision = config.precision;
        // Both layouts are accepted: flat, as produced by export_span_v3.py, and
        // the legacy `fp32_v2/` + `fp16_v2/` subfolders the earlier exporter
        // published on the Hub.
        let load = |stem: &str| -> Result<Session> {
            let path = resolve_fragment(dir, stem, precision).ok_or_else(|| {
                GlinerError::IncompleteModelDir(format!(
                    "fragment '{stem}{}' not found in {} (looked in the directory itself and in {}/)",
                    precision.suffix(),
                    dir.display(),
                    precision.legacy_subdir(),
                ))
            })?;
            build_session(&path, config.intra_threads)
        };

        let tok_path = resolve_tokenizer(dir, precision).ok_or_else(|| {
            GlinerError::IncompleteModelDir(format!(
                "tokenizer.json not found in {} nor in its variant subfolders",
                dir.display()
            ))
        })?;
        let transformer = SchemaTransformer::from_tokenizer_file(&tok_path)?;

        let encoder = load("encoder")?;
        let count_lstm = load("count_lstm_fixed")?;
        let scorer = load("scorer")?;

        // MAX_COUNT, max_width and hidden_size are read from the fragments'
        // static shapes rather than hard-coded, so a checkpoint exported with
        // different parameters stays usable.
        let (max_count, hidden) = static_dims_count_lstm(&count_lstm)?;
        let max_width = static_max_width(&scorer)?;

        let classifier = load("classifier")?;
        let classifier_layout = detect_classifier_layout(&classifier)?;

        Ok(Self {
            encoder,
            token_gather: load("token_gather")?,
            span_rep: load("span_rep")?,
            schema_gather: load("schema_gather")?,
            count_pred: load("count_pred_argmax")?,
            count_lstm,
            scorer,
            classifier,
            transformer,
            dtype: config.precision.io_dtype(),
            max_width,
            max_count,
            hidden,
            classifier_layout,
        })
    }

    pub fn max_width(&self) -> usize {
        self.max_width
    }
    pub fn max_count(&self) -> usize {
        self.max_count
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }
    /// Which classifier signature the loaded export carries.
    pub fn classifier_layout(&self) -> ClassifierLayout {
        self.classifier_layout
    }

    pub fn extract(&mut self, text: &str, tasks: &[SchemaTask]) -> Result<SpanOutput> {
        self.extract_with(text, tasks, &InferenceParams::default())
    }

    pub fn extract_with(
        &mut self,
        text: &str,
        tasks: &[SchemaTask],
        params: &InferenceParams,
    ) -> Result<SpanOutput> {
        let record = self.transformer.transform(text, tasks)?;
        let num_words = record.num_words();
        if num_words == 0 {
            return Ok(SpanOutput::default());
        }

        // ── 1. encoder ────────────────────────────────────────────────────
        let seq = record.input_ids.len() as i64;
        let hidden = {
            let ids = i64_tensor(vec![1, seq], record.input_ids.clone())?;
            let mask = i64_tensor(vec![1, seq], record.attention_mask.clone())?;
            let out = self.encoder.run(ort::inputs![ids, mask])?;
            take_float(&out["last_hidden_state"], self.dtype)?.1
        };

        // ── 2. token_gather -> text_embs ──────────────────────────────────
        let text_embs = {
            let h = float_tensor(self.dtype, vec![1, seq, self.hidden as i64], hidden.clone())?;
            let idx = i64_tensor(vec![num_words as i64], record.word_first_positions())?;
            let out = self.token_gather.run(ort::inputs![h, idx])?;
            take_float(&out["text_embs"], self.dtype)?.1
        };

        // ── 3. span_rep -> span_embeddings ────────────────────────────────
        let span_embeddings = {
            let spans = build_span_idx(num_words, self.max_width);
            let n_spans = (num_words * self.max_width) as i64;
            let te = float_tensor(
                self.dtype,
                vec![1, num_words as i64, self.hidden as i64],
                text_embs,
            )?;
            let si = i64_tensor(vec![1, n_spans, 2], spans)?;
            let out = self.span_rep.run(ort::inputs![te, si])?;
            take_float(&out["span_embeddings"], self.dtype)?.1
        };

        // ── 4. one pass per schema group ──────────────────────────────────
        let mut result = SpanOutput::default();
        for task in &record.tasks {
            let m = task.labels.len();
            if m == 0 {
                continue;
            }

            let mut schema_indices = Vec::with_capacity(m + 1);
            schema_indices.push(task.prompt_tok_idx as i64);
            schema_indices.extend(task.field_tok_indices.iter().map(|i| *i as i64));

            let (pc_emb, field_embs) = {
                let h = float_tensor(self.dtype, vec![1, seq, self.hidden as i64], hidden.clone())?;
                let si = i64_tensor(vec![schema_indices.len() as i64], schema_indices)?;
                let out = self.schema_gather.run(ort::inputs![h, si])?;
                (
                    take_float(&out["pc_emb"], self.dtype)?.1,
                    take_float(&out["field_embs"], self.dtype)?.1,
                )
            };

            match task.task_type {
                TaskType::Classifications => {
                    let logits = match self.classifier_layout {
                        ClassifierLayout::Flat => {
                            let fe = float_tensor(
                                self.dtype,
                                vec![m as i64, self.hidden as i64],
                                field_embs,
                            )?;
                            let out = self.classifier.run(ort::inputs![fe])?;
                            take_float(&out["logits"], self.dtype)?.1
                        }
                        ClassifierLayout::Padded => {
                            // The legacy graph wants [1, M, max_width, H]. The MLP
                            // acts on the last dimension, so replicating each row
                            // across max_width and reading back slice 0 is exact —
                            // just max_width times the arithmetic for one result.
                            let mw = self.max_width;
                            let mut padded = vec![0.0f32; m * mw * self.hidden];
                            for label in 0..m {
                                let src = &field_embs[label * self.hidden..(label + 1) * self.hidden];
                                for w in 0..mw {
                                    let base = (label * mw + w) * self.hidden;
                                    padded[base..base + self.hidden].copy_from_slice(src);
                                }
                            }
                            let fe = float_tensor(
                                self.dtype,
                                vec![1, m as i64, mw as i64, self.hidden as i64],
                                padded,
                            )?;
                            let out = self.classifier.run(ort::inputs![fe])?;
                            let flat = take_float(&out["logits"], self.dtype)?.1;
                            // [1, M, mw, 1] -> one logit per label
                            (0..m).map(|label| flat[label * mw]).collect()
                        }
                    };
                    let scaled: Vec<f32> = logits
                        .iter()
                        .map(|l| l / params.classification_temperature)
                        .collect();
                    let multi_label = params.multi_label_override.unwrap_or(task.multi_label);
                    let probs = if multi_label {
                        scaled.iter().copied().map(sigmoid).collect::<Vec<_>>()
                    } else {
                        softmax(&scaled)
                    };
                    for (label, score) in task.labels.iter().zip(probs) {
                        result.classifications.push(Classification {
                            task: task.task_name.clone(),
                            label: label.clone(),
                            score,
                        });
                    }
                }
                TaskType::Entities | TaskType::Relations => {
                    let pred_count = {
                        let pc = float_tensor(self.dtype, vec![1, self.hidden as i64], pc_emb)?;
                        let out = self.count_pred.run(ort::inputs![pc])?;
                        take_i64(&out["pred_count"])?.1[0].max(0) as usize
                    };
                    let pred_count = pred_count.min(self.max_count);
                    if pred_count == 0 {
                        continue;
                    }

                    let struct_proj = {
                        let fe = float_tensor(
                            self.dtype,
                            vec![m as i64, self.hidden as i64],
                            field_embs,
                        )?;
                        let out = self.count_lstm.run(ort::inputs![fe])?;
                        take_float(&out["struct_proj"], self.dtype)?.1
                    };

                    let scores = {
                        let se = float_tensor(
                            self.dtype,
                            vec![1, num_words as i64, self.max_width as i64, self.hidden as i64],
                            span_embeddings.clone(),
                        )?;
                        let sp = float_tensor(
                            self.dtype,
                            vec![self.max_count as i64, m as i64, self.hidden as i64],
                            struct_proj,
                        )?;
                        let out = self.scorer.run(ort::inputs![se, sp])?;
                        take_float(&out["entity_scores"], self.dtype)?.1
                    };

                    self.collect_entities(
                        &scores,
                        &record,
                        task,
                        m,
                        pred_count,
                        num_words,
                        text,
                        params,
                        &mut result.entities,
                    );
                }
            }
        }

        Ok(result)
    }

    /// `entity_scores` is laid out `[MAX_COUNT, W, max_width, M]` and has
    /// already gone through the sigmoid inside the scorer graph.
    #[allow(clippy::too_many_arguments)]
    fn collect_entities(
        &self,
        scores: &[f32],
        record: &ProcessedRecord,
        task: &gliner_core::processor::TaskMapping,
        m: usize,
        pred_count: usize,
        num_words: usize,
        text: &str,
        params: &InferenceParams,
        out: &mut Vec<Entity>,
    ) {
        let mw = self.max_width;
        let stride_c = num_words * mw * m;
        let stride_w = mw * m;

        let mut found: Vec<Entity> = Vec::new();
        for c in 0..pred_count {
            for w in 0..num_words {
                for k in 0..mw {
                    // the span covers w..=w+k, valid only while inside the text
                    if w + k >= num_words {
                        break;
                    }
                    for label_idx in 0..m {
                        let score = scores[c * stride_c + w * stride_w + k * m + label_idx];
                        if score < params.threshold {
                            continue;
                        }
                        let (cs, _) = record.word_to_char_maps[w];
                        let (_, ce) = record.word_to_char_maps[w + k];
                        found.push(Entity {
                            text: text[cs..ce].to_string(),
                            label: task.labels[label_idx].clone(),
                            score,
                            char_start: cs,
                            char_end: ce,
                            word_start: w,
                            word_end: w + k,
                            slot: c,
                            task: task.task_name.clone(),
                        });
                    }
                }
            }
        }

        match params.overlap_policy {
            None => out.extend(greedy_nms(found, params.flat_ner)),
            Some(policy) => out.extend(resolve_overlaps(&found, policy)),
        }
    }
}

/// Greedy suppression by descending score, reproducing gliner2's default.
///
/// The reference implementation decodes **one label at a time**: for each entity
/// name it thresholds the scores, sorts by confidence and drops any candidate
/// overlapping one already kept *for that same name*. Labels never interact, so
/// the same span can legitimately come back under several of them — a doctor's
/// name is both `medical_professional` and `person`, and a redaction pipeline
/// wants to know both.
///
/// `flat_ner = true` is a deliberate extension, not the reference behaviour: it
/// suppresses across labels too, leaving one label per stretch of text.
fn greedy_nms(mut candidates: Vec<Entity>, flat_ner: bool) -> Vec<Entity> {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.word_start.cmp(&b.word_start))
            .then(a.word_end.cmp(&b.word_end))
            .then(a.label.cmp(&b.label))
    });

    let mut kept: Vec<Entity> = Vec::new();
    for cand in candidates {
        let suppressed = kept.iter().any(|k| {
            if !flat_ner && k.label != cand.label {
                return false;
            }
            // inclusive word ranges overlap when neither ends before the other starts
            cand.word_start <= k.word_end && k.word_start <= cand.word_end
        });
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// `span_idx[w * max_width + k] = (w, w + k)`, with `(0,0)` on invalid spans.
fn build_span_idx(num_words: usize, max_width: usize) -> Vec<i64> {
    let mut spans = Vec::with_capacity(num_words * max_width * 2);
    for w in 0..num_words {
        for k in 0..max_width {
            let end = w + k;
            if end < num_words {
                spans.push(w as i64);
                spans.push(end as i64);
            } else {
                spans.push(0);
                spans.push(0);
            }
        }
    }
    spans
}

/// Detects which classifier signature the export carries, from the input rank.
fn detect_classifier_layout(session: &Session) -> Result<ClassifierLayout> {
    let inp = session
        .inputs()
        .first()
        .ok_or_else(|| anyhow!("the classifier fragment declares no input"))?;
    let rank = tensor_dims(inp.dtype())?.len();
    match rank {
        2 => Ok(ClassifierLayout::Flat),
        4 => Ok(ClassifierLayout::Padded),
        other => Err(anyhow!(
            "unexpected classifier input rank {other} (expected 2 for [num_labels, H] \
             or 4 for [batch, num_labels, max_width, H])"
        )),
    }
}

/// Reads `MAX_COUNT` and `hidden_size` from `struct_proj`'s static shape.
fn static_dims_count_lstm(session: &Session) -> Result<(usize, usize)> {
    let out = session
        .outputs()
        .iter()
        .find(|o| o.name() == "struct_proj")
        .ok_or_else(|| anyhow!("count_lstm_fixed does not expose struct_proj"))?;
    let dims = tensor_dims(out.dtype())?;
    let max_count = dims
        .first()
        .and_then(|d| *d)
        .ok_or_else(|| anyhow!("struct_proj: MAX_COUNT dimension is not static"))?;
    let hidden = dims
        .get(2)
        .and_then(|d| *d)
        .ok_or_else(|| anyhow!("struct_proj: hidden_size is not static"))?;
    Ok((max_count as usize, hidden as usize))
}

/// Reads `max_width` from the scorer's `span_embeddings` static shape.
fn static_max_width(session: &Session) -> Result<usize> {
    let inp = session
        .inputs()
        .iter()
        .find(|i| i.name() == "span_embeddings")
        .ok_or_else(|| anyhow!("scorer does not accept span_embeddings"))?;
    let dims = tensor_dims(inp.dtype())?;
    dims.get(2)
        .and_then(|d| *d)
        .map(|d| d as usize)
        .ok_or_else(|| anyhow!("span_embeddings: max_width is not static"))
}

fn tensor_dims(vt: &ort::value::ValueType) -> Result<Vec<Option<i64>>> {
    match vt {
        ort::value::ValueType::Tensor { shape, .. } => Ok(shape
            .iter()
            .map(|d| if *d < 0 { None } else { Some(*d) })
            .collect()),
        other => Err(anyhow!("expected a tensor, found {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_idx_layout() {
        // 3 words, max_width 2: (0,0) (0,1) (1,1) (1,2) (2,2) invalid -> (0,0)
        assert_eq!(build_span_idx(3, 2), vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0]);
    }

    #[test]
    fn nms_suppresses_within_label_only() {
        let e = |ws, we, label: &str, score| Entity {
            text: String::new(),
            label: label.into(),
            score,
            char_start: 0,
            char_end: 0,
            word_start: ws,
            word_end: we,
            slot: 0,
            task: "entities".into(),
        };
        // flat_ner suppresses across labels
        let kept = greedy_nms(vec![e(0, 1, "person", 0.9), e(0, 0, "first_name", 0.8)], true);
        assert_eq!(kept.len(), 1);
        // by default labels never interact, so a nested span of another label stays
        let kept = greedy_nms(vec![e(0, 1, "person", 0.9), e(0, 0, "first_name", 0.8)], false);
        assert_eq!(kept.len(), 2);
        // the same span under two labels is legitimate: a doctor is both
        // `medical_professional` and `person`
        let kept = greedy_nms(vec![e(0, 1, "medical_professional", 0.9), e(0, 1, "person", 0.8)], false);
        assert_eq!(kept.len(), 2);
        // within one label, the weaker overlapping span is dropped
        let kept = greedy_nms(vec![e(0, 1, "person", 0.9), e(1, 2, "person", 0.8)], false);
        assert_eq!(kept.len(), 1);
    }
}
