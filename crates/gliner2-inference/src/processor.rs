// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Module for preprocessing and tokenization of text and schemas.

use tokenizers::Tokenizer;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use regex::Regex;

// GLiNER2 special constants extracted from processor.py
pub const SEP_STRUCT: &str = "[SEP_STRUCT]";
pub const SEP_TEXT: &str = "[SEP_TEXT]";
pub const P_TOKEN: &str = "[P]";
pub const C_TOKEN: &str = "[C]";
pub const E_TOKEN: &str = "[E]";
pub const R_TOKEN: &str = "[R]";
pub const L_TOKEN: &str = "[L]";
pub const EXAMPLE_TOKEN: &str = "[EXAMPLE]";
pub const OUTPUT_TOKEN: &str = "[OUTPUT]";
pub const DESC_TOKEN: &str = "[DESCRIPTION]";

#[derive(Debug, Clone)]
pub enum SchemaTask {
    /// Entity extraction task. Contains the list of label names (e.g., "person", "city").
    Entities(Vec<String>),
    /// Relation extraction task. Contains the relation name and fields (e.g., "works_at", ["head", "tail"]).
    Relations(String, Vec<String>),
    /// Text classification task. Contains the task name and classes (e.g., "sentiment", ["positive", "negative"]).
    Classifications(String, Vec<String>),
}

#[derive(Debug, Clone)]
pub struct TaskMapping {
    pub task_name: String,
    pub task_type: String,
    pub labels: Vec<String>,
    pub prompt_tok_idx: usize,
    pub field_tok_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ProcessedRecord {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    
    pub tasks: Vec<TaskMapping>,
    
    pub text_start: usize,
    pub text_end: usize,
    
    pub word_to_token_maps: Vec<(usize, usize)>,
    pub word_to_char_maps: Vec<(usize, usize)>,
}

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

    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.re
            .find_iter(text)
            .map(|m| m.as_str())
            .collect()
    }

    /// Mirrors `WhitespaceTokenSplitter.__call__(text, lower=True)`: matching
    /// happens on the **original** text so offsets index the caller's string,
    /// and only the token value is lower-cased. Lower-casing the source first
    /// would be unsafe, because Unicode case folding can change its length and
    /// corrupt the offsets.
    pub fn split_with_offsets(&self, text: &str) -> Vec<(String, usize, usize)> {
        self.re
            .find_iter(text)
            .map(|m| (m.as_str().to_lowercase(), m.start(), m.end()))
            .collect()
    }
}

pub struct SchemaTransformer {
    tokenizer: Tokenizer,
    word_splitter: WhitespaceTokenSplitter,
}

impl SchemaTransformer {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { 
            tokenizer,
            word_splitter: WhitespaceTokenSplitter::new().unwrap(),
        }
    }

    pub fn transform(&self, text: &str, schema_tasks: &[SchemaTask]) -> Result<ProcessedRecord> {
        let words_with_offsets = self.word_splitter.split_with_offsets(text);
        
        let mut combined_tokens = Vec::new();
        let mut task_mappings_temp = Vec::new();

        for (i, task) in schema_tasks.iter().enumerate() {
            let mut field_indices = Vec::new();
            let mut labels = Vec::new();

            match task {
                SchemaTask::Entities(entity_labels) => {
                    combined_tokens.push("(");
                    let prompt_idx = combined_tokens.len();
                    combined_tokens.push(P_TOKEN);
                    combined_tokens.push("entities");
                    combined_tokens.push("(");
                    
                    for label in entity_labels {
                        // position of the marker itself, matching Python's
                        // `schema_marker_orig_indices`
                        field_indices.push(combined_tokens.len());
                        combined_tokens.push(E_TOKEN);
                        combined_tokens.push(label.as_str());
                        labels.push(label.clone());
                    }
                    combined_tokens.push(")");
                    combined_tokens.push(")");
                    
                    task_mappings_temp.push((
                        "entities".to_string(),
                        "entities".to_string(), 
                        labels, 
                        prompt_idx, 
                        field_indices
                    ));
                }
                SchemaTask::Relations(rel_name, fields) => {
                    combined_tokens.push("(");
                    let prompt_idx = combined_tokens.len();
                    combined_tokens.push(P_TOKEN);
                    combined_tokens.push(rel_name.as_str());
                    combined_tokens.push("(");
                    
                    for field in fields {
                        // position of the marker itself, matching Python's
                        // `schema_marker_orig_indices`
                        field_indices.push(combined_tokens.len());
                        combined_tokens.push(R_TOKEN);
                        combined_tokens.push(field.as_str());
                        labels.push(field.clone());
                    }
                    combined_tokens.push(")");
                    combined_tokens.push(")");
                    
                    task_mappings_temp.push((
                        rel_name.clone(),
                        "relations".to_string(), 
                        labels, 
                        prompt_idx, 
                        field_indices
                    ));
                }
                SchemaTask::Classifications(task_name, cls_labels) => {
                    combined_tokens.push("(");
                    let prompt_idx = combined_tokens.len();
                    combined_tokens.push(P_TOKEN);
                    combined_tokens.push(task_name.as_str());
                    combined_tokens.push("(");
                    
                    for label in cls_labels {
                        // position of the marker itself, matching Python's
                        // `schema_marker_orig_indices`
                        field_indices.push(combined_tokens.len());
                        combined_tokens.push(L_TOKEN);
                        combined_tokens.push(label.as_str());
                        labels.push(label.clone());
                    }
                    combined_tokens.push(")");
                    combined_tokens.push(")");
                    
                    task_mappings_temp.push((
                        task_name.clone(),
                        "classifications".to_string(), 
                        labels, 
                        prompt_idx, 
                        field_indices
                    ));
                }
            }
            
            if i < schema_tasks.len() - 1 {
                combined_tokens.push(SEP_STRUCT);
            }
        }
        
        combined_tokens.push(SEP_TEXT);
        let text_start_idx = combined_tokens.len();
        
        let mut word_to_char_maps = Vec::new();
        for (w, start_char, end_char) in &words_with_offsets {
            combined_tokens.push(w.as_str());
            word_to_char_maps.push((*start_char, *end_char));
        }
        let text_end_idx = combined_tokens.len();

        let mut final_input_ids = Vec::new();
        let mut final_attention_mask = Vec::new();
        let mut word_to_token_maps = Vec::new();
        
        let mut combined_to_final_map = HashMap::new();
        
        // gliner2 non incapsula il prompt fra [CLS] e [SEP]: la sequenza inizia
        // sulla prima `(` del primo gruppo di schema e finisce sull'ultima
        // parola del testo. Incapsularlo dava al modello un formato su cui non
        // e' stato addestrato.
        let mut current_subword_idx = 0;

        for (i, token) in combined_tokens.iter().enumerate() {
            combined_to_final_map.insert(i, current_subword_idx);
            
            let encoding = self.tokenizer.encode(*token, false)
                .map_err(|e| anyhow!("Tokenization failed for {token}: {e}"))?;
                
            let ids = encoding.get_ids();
            let start_sub = current_subword_idx;
            let end_sub = current_subword_idx + ids.len();
            
            for &id in ids {
                final_input_ids.push(id as i64);
                final_attention_mask.push(1);
                current_subword_idx += 1;
            }
            
            if i >= text_start_idx && i < text_end_idx {
                word_to_token_maps.push((start_sub, end_sub));
            }
        }
        
        let text_real_start = word_to_token_maps.first().map(|v| v.0).unwrap_or(0);
        let text_real_end = word_to_token_maps.last().map(|v| v.1).unwrap_or(0);

        let mut tasks = Vec::new();
        for (task_name, task_type, labels, prompt_idx, field_indices) in task_mappings_temp {
            let real_prompt_idx = *combined_to_final_map.get(&prompt_idx)
                .ok_or_else(|| anyhow!("prompt_idx {prompt_idx} fuori mappa (invariante rotta)"))?;
            let real_field_indices: Vec<usize> = field_indices.iter()
                .map(|idx| combined_to_final_map.get(idx).copied()
                    .ok_or_else(|| anyhow!("field_idx {idx} fuori mappa (invariante rotta)")))
                .collect::<Result<_>>()?;
                
            tasks.push(TaskMapping {
                task_name,
                task_type,
                labels,
                prompt_tok_idx: real_prompt_idx,
                field_tok_indices: real_field_indices,
            });
        }

        Ok(ProcessedRecord {
            input_ids: final_input_ids,
            attention_mask: final_attention_mask,
            tasks,
            text_start: text_real_start,
            text_end: text_real_end,
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
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        [
            "models/gliner2-multi-v1-onnx-v2/tokenizer.json",
            "models/gliner2-multi-finetuned-v2/tokenizer.json",
            "models/iobinding_test/tokenizer.json",
            "../models/fastino_gliner2_multi_v1_fp32/tokenizer.json",
            "../models/jugaad_gliner2_multi_v1_fp32/tokenizer.json",
        ]
        .iter()
        .map(|p| root.join(p))
        .find(|p| p.exists())
    }

    /// Ground truth produced by running gliner2 2.0.0. The prompt layout is
    ///
    /// ```text
    /// ["(", "[P]", prompt_str, "("] + [child_prefix, field_name] * N + [")", ")"]
    /// ```
    ///
    /// and for `entities(person, organization, location)` plus
    /// `classification(sentiment, ...)` on
    /// `"Mario Rossi lavora ad Apple a Cupertino."` it yields
    ///
    /// ```text
    /// schema_special_positions  = [[1, 6, 8, 10], [18, 21, 23]]
    /// text_word_first_positions = [31, 33, 35, 36, 37, 38, 40, 43]
    /// ```
    ///
    /// which pins all three properties at once: no `[CLS]`/`[SEP]` wrapping,
    /// field indices on the markers rather than the labels, and lower-cased
    /// text words.
    #[test]
    fn ground_truth_prompt_layout() {
        let Some(path) = tokenizer_path() else {
            eprintln!("no tokenizer.json available, test skipped");
            return;
        };
        let tokenizer = Tokenizer::from_file(&path).expect("tokenizer");
        let transformer = SchemaTransformer::new(tokenizer);

        let tasks = vec![
            SchemaTask::Entities(vec![
                "person".to_string(),
                "organization".to_string(),
                "location".to_string(),
            ]),
            SchemaTask::Classifications(
                "sentiment".to_string(),
                vec!["positive".to_string(), "negative".to_string()],
            ),
        ];
        let record = transformer
            .transform("Mario Rossi lavora ad Apple a Cupertino.", &tasks)
            .expect("transform");

        assert_eq!(record.tasks[0].prompt_tok_idx, 1);
        assert_eq!(record.tasks[0].field_tok_indices, vec![6, 8, 10]);
        assert_eq!(record.tasks[1].prompt_tok_idx, 18);
        assert_eq!(record.tasks[1].field_tok_indices, vec![21, 23]);

        let word_starts: Vec<usize> =
            record.word_to_token_maps.iter().map(|(s, _)| *s).collect();
        assert_eq!(word_starts, vec![31, 33, 35, 36, 37, 38, 40, 43]);
        assert_eq!(record.word_to_token_maps.len(), 8);
        assert_eq!(record.input_ids.len(), 45);
        assert_eq!(record.input_ids.len(), record.attention_mask.len());

        // offsets index the original text, so casing survives
        let text = "Mario Rossi lavora ad Apple a Cupertino.";
        let (s, e) = record.word_to_char_maps[0];
        assert_eq!(&text[s..e], "Mario");
    }
}
