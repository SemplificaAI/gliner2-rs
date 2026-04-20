// Copyright 2026 Dario Finardi, Semplifica s.r.l.
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

//! `gliner2-rs` is a high-performance native Rust inference engine for GLiNER2 models.
//!
//! It enables Zero-Python execution of complex Natural Language Processing tasks 
//! (Entities, Relations, Classifications) using ONNX Runtime (with CPU and CUDA GPU support).

pub mod processor;

use anyhow::Result;
use ndarray::{Array0, Array2, Array3, s};
use ort::{
    execution_providers::{
        CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
        OpenVINOExecutionProvider, QNNExecutionProvider, XNNPACKExecutionProvider,
    },
    session::{builder::GraphOptimizationLevel, Session},
    value::{Tensor, Value, DynValueTypeMarker},
};
use tokenizers::Tokenizer;
use std::path::Path;
use serde::Serialize;

use processor::SchemaTransformer;
pub use processor::SchemaTask;

/// Configurazione base per l'inizializzazione del motore.
#[derive(Debug, Clone)]
pub struct Gliner2Config {
    pub models_dir: String,
    pub max_width: usize,
    pub model_type: ModelType,
}

impl Default for Gliner2Config {
    fn default() -> Self {
        Self {
            models_dir: "models/fragments_fp16".to_string(),
            max_width: 8,
            model_type: ModelType::PyTorch,
        }
    }
}

/// Tipo di modello GLiNER2 per gestire diverse architetture ONNX.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    /// Modello PyTorch convertito (nostro server) - ha last_hidden_state
    PyTorch,
    /// Modello HuggingFace (download pubblico) - architettura diversa
    HuggingFace,
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::PyTorch
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::PyTorch => write!(f, "PyTorch"),
            ModelType::HuggingFace => write!(f, "HuggingFace"),
        }
    }
}

/// Dati di un'entità estratta dal testo.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractedEntity {
    pub text: String,
    pub label: String,
    pub score: f32,
    pub start_tok: usize,
    pub end_tok: usize,
}

/// Dati di una relazione tra due entità.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractedRelation {
    pub head: ExtractedEntity,
    pub tail: ExtractedEntity,
    pub relation_type: String,
}

/// Dati di classificazione globale sul testo in esame.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractedClassification {
    pub task_name: String,
    pub label: String,
    pub score: f32,
}

/// Motore di inferenza principale.
pub struct Gliner2Engine {
    encoder: Session,
    span_rep: Session,
    count_lstm: Session,
    count_pred: Session,
    classifier: Session,
    tokenizer: Tokenizer,
    config: Gliner2Config,
}

impl Gliner2Engine {
    /// Inizializza le reti neurali caricando i file ONNX e il Tokenizer.
    pub fn new(config: Gliner2Config) -> Result<Self> {
        let dir = Path::new(&config.models_dir);
        
        let load_session = |base_name: &str| -> Result<Session> {
            let path_fp16 = dir.join(format!("{}_fp16.onnx", base_name));
            let path_fp32 = dir.join(format!("{}_fp32.onnx", base_name));
            
            let path = if path_fp16.exists() {
                path_fp16
            } else if path_fp32.exists() {
                path_fp32
            } else {
                return Err(anyhow::anyhow!("Neither {}_fp16.onnx nor {}_fp32.onnx exist", base_name, base_name));
            };

            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_memory_pattern(false)?
                .with_execution_providers([
                    QNNExecutionProvider::default().build(),
                    OpenVINOExecutionProvider::default().build(),
                    CoreMLExecutionProvider::default().build(),
                    CUDAExecutionProvider::default().build(),
                    XNNPACKExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build()
                ])?
                .commit_from_file(&path)
                .map_err(|e| anyhow::anyhow!("Error loading {:?}: {}", path, e))
        };

        // Caricamento modelli basato sul tipo di modello
        let (count_pred, count_lstm) = match config.model_type {
            ModelType::PyTorch => {
                // Modello PyTorch convertito
                let count_lstm = load_session("count_lstm")?;
                let count_pred = load_session("count_pred")?;
                (count_pred, count_lstm)
            }
            ModelType::HuggingFace => {
                // Modello HuggingFace
                let count_pred = load_session("count_pred")?;
                let count_lstm = load_session("count_lstm")
                    .or_else(|_| load_session("count_pred"))?; // Fallback
                (count_pred, count_lstm)
            }
        };

        // Caricamento modelli comuni
        let encoder = load_session("encoder")?;
        let span_rep = load_session("span_rep")?;
        let classifier = load_session("classifier")?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Error loading Tokenizer: {}", e))?;

        Ok(Self { 
            encoder, 
            span_rep, 
            count_lstm, 
            count_pred, 
            classifier, 
            tokenizer, 
            config 
        })
    }

    /// Esegue il flusso end-to-end su una stringa in input
    /// in base agli Schema Tasks forniti.
    pub fn extract(
        &self, 
        text: &str, 
        tasks: &[SchemaTask]
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>, Vec<ExtractedClassification>)> {
        
        // 1. Processo prompt + testo (creazione vettori di token)
        let transformer = SchemaTransformer::new(self.tokenizer.clone());
        let record = transformer.transform(text, tasks)?;
        let seq_len = record.input_ids.len();

        let input_ids = Array2::from_shape_vec((1, seq_len), record.input_ids.clone())?;
        let attention_mask = Array2::from_shape_vec((1, seq_len), record.attention_mask.clone())?;

        // 2. Passaggio Encoder (DeBERTa) -> Contextual Embeddings
        let mut has_attention_mask = false;
        for input in &self.encoder.inputs {
            if input.name == "attention_mask" {
                has_attention_mask = true;
            }
        }
        
        let enc_inputs = if has_attention_mask {
            ort::inputs![
                "input_ids" => Tensor::from_array(input_ids)?,
                "attention_mask" => Tensor::from_array(attention_mask)?
            ]?
        } else {
            ort::inputs![
                "input_ids" => Tensor::from_array(input_ids)?
            ]?
        };
        
        let enc_outputs = self.encoder.run(enc_inputs)?;
        
        // Gestione output diversi basati sul tipo di modello
        let lhs_tensor = {
            if let Some(val) = enc_outputs.get("hidden_states") {
                val.try_extract_tensor::<f32>()?.into_owned()
            } else if let Some(val) = enc_outputs.get("last_hidden_state") {
                val.try_extract_tensor::<f32>()?.into_owned()
            } else if let Some(val) = enc_outputs.get("output") {
                val.try_extract_tensor::<f32>()?.into_owned()
            } else {
                return Err(anyhow::anyhow!("No valid encoder output found (tried hidden_states, last_hidden_state, output)"));
            }
        };
        
        let num_words = record.word_to_token_maps.len();
        if num_words == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let hidden_size = lhs_tensor.shape()[2];
        let mut word_embs_data = Vec::with_capacity(num_words * hidden_size);
        for &(start_sub, _) in &record.word_to_token_maps {
            let word_emb = lhs_tensor.slice(s![0, start_sub, ..]);
            for &val in word_emb {
                word_embs_data.push(val);
            }
        }
        let text_embs = Array3::from_shape_vec((1, num_words, hidden_size), word_embs_data)?;
        let text_len = num_words;

        // Generazione iterativa degli Span Index (alberi di combinazione)
        let num_spans = text_len * self.config.max_width;
        let mut span_idx_data: Vec<i64> = Vec::with_capacity(num_spans * 2);
        for start in 0..text_len {
            for width in 0..self.config.max_width {
                let end = start + width;
                if end >= text_len {
                    // Out-of-bounds pad per sicurezza ONNX gather node
                    span_idx_data.push(0);
                    span_idx_data.push(0);
                } else {
                    span_idx_data.push(start as i64);
                    span_idx_data.push(end as i64);
                }
            }
        }
        let span_idx_arr = Array3::from_shape_vec((1, num_spans, 2), span_idx_data)?;

        // 3. Span Representation Layer
        let mut has_span_idx = false;
        let mut text_embs_name = "hidden_states";
        for i in &self.span_rep.inputs {
            if i.name == "span_idx" { has_span_idx = true; }
            if i.name == "last_hidden_state" { text_embs_name = "last_hidden_state"; }
            if i.name == "hidden_states" { text_embs_name = "hidden_states"; }
            if i.name == "output" { text_embs_name = "output"; }
        }

        let span_inputs = if has_span_idx {
            // PyTorch model style: uses text_embs and span_idx
            ort::inputs![
                text_embs_name => Tensor::from_array(text_embs)?,
                "span_idx" => Tensor::from_array(span_idx_arr)?
            ]?
        } else {
            // HuggingFace model style: uses text_embs, span_start_idx, span_end_idx
            let mut start_idx_data = Vec::with_capacity(num_spans);
            let mut end_idx_data = Vec::with_capacity(num_spans);
            
            for start in 0..text_len {
                for width in 0..self.config.max_width {
                    let end = start + width;
                    if end >= text_len {
                        start_idx_data.push(0i64);
                        end_idx_data.push(0i64);
                    } else {
                        start_idx_data.push(start as i64);
                        end_idx_data.push(end as i64);
                    }
                }
            }
            
            let start_arr = Array2::from_shape_vec((1, num_spans), start_idx_data)?;
            let end_arr = Array2::from_shape_vec((1, num_spans), end_idx_data)?;
            
            ort::inputs![
                text_embs_name => Tensor::from_array(text_embs)?,
                "span_start_idx" => Tensor::from_array(start_arr)?,
                "span_end_idx" => Tensor::from_array(end_arr)?
            ]?
        };
        
        let span_outputs = self.span_rep.run(span_inputs)?;
        let span_embeddings = {
            if let Some(val) = span_outputs.get("span_embeddings") {
                val.try_extract_tensor::<f32>()?.into_owned()
            } else if let Some(val) = span_outputs.get("span_representations") {
                val.try_extract_tensor::<f32>()?.into_owned()
            } else {
                return Err(anyhow::anyhow!("No valid span_rep output found (tried span_embeddings, span_representations)"));
            }
        };

        let hidden_size = lhs_tensor.shape()[2];
        let span_emb_shape = span_embeddings.shape();

        let mut final_entities = Vec::new();
        let mut final_relations = Vec::new();
        let mut final_classifications = Vec::new();

        // 4. Esecuzione Task Paralleli
        for task_map in &record.tasks {
            let labels = &task_map.labels;
            let num_labels = labels.len();

            let mut schema_embs_data = Vec::with_capacity(num_labels * hidden_size);
            for &idx in &task_map.field_tok_indices {
                let label_emb = lhs_tensor.slice(s![0, idx, ..]);
                for &val in label_emb {
                    schema_embs_data.push(val);
                }
            }
            if schema_embs_data.len() != num_labels * hidden_size {
                schema_embs_data.resize(num_labels * hidden_size, 0.0);
            }
            let schema_embs = Array2::from_shape_vec((num_labels, hidden_size), schema_embs_data)?;

            // 4a. Ramo Classificazione (Softmax Testo-intero)
            if task_map.task_type == "classifications" {
                let mut padded_embs = ndarray::Array4::<f32>::zeros((1, num_labels, self.config.max_width, hidden_size));
                for m in 0..num_labels {
                    for d in 0..hidden_size {
                        padded_embs[[0, m, 0, d]] = schema_embs[[m, d]];
                    }
                }
                
                let cls_inputs = ort::inputs![
                    "span_embeddings" => Tensor::from_array(padded_embs)?
                ]?;
                let cls_outputs = self.classifier.run(cls_inputs)?;
                let logits_tensor = {
                    if let Some(val) = cls_outputs.get("logits") {
                        val.try_extract_tensor::<f32>()?.into_owned()
                    } else if let Some(val) = cls_outputs.get("output") {
                        val.try_extract_tensor::<f32>()?.into_owned()
                    } else {
                        return Err(anyhow::anyhow!("No valid classifier output found"));
                    }
                };
                
                let mut exp_sum = 0.0;
                let mut exps = Vec::with_capacity(num_labels);
                
                for m in 0..num_labels {
                    let logit = logits_tensor[[0, m, 0, 0]];
                    let e = logit.exp();
                    exps.push(e);
                    exp_sum += e;
                }
                
                let mut best_score = 0.0;
                let mut best_idx = 0;
                
                for m in 0..num_labels {
                    let prob = exps[m] / exp_sum;
                    if prob > best_score {
                        best_score = prob;
                        best_idx = m;
                    }
                }
                
                final_classifications.push(ExtractedClassification {
                    task_name: task_map.task_name.clone(),
                    label: labels[best_idx].clone(),
                    score: best_score,
                });
                continue;
            }

            // 4b. Ramo Count LSTM (Entità e Relazioni)
            let pc_emb_first = lhs_tensor.slice(s![0..1, task_map.prompt_tok_idx, ..]).to_owned();
            let cpred_input_name = self.count_pred.inputs[0].name.as_str();
            let cpred_inputs = ort::inputs![
                cpred_input_name => Tensor::from_array(pc_emb_first)?
            ]?;
            let cpred_outputs = self.count_pred.run(cpred_inputs)?;
            
            let count_logits = {
                if let Some(val) = cpred_outputs.get("count_logits") {
                    val.try_extract_tensor::<f32>()?.into_owned()
                } else if let Some(val) = cpred_outputs.get("output") {
                    val.try_extract_tensor::<f32>()?.into_owned()
                } else {
                    return Err(anyhow::anyhow!("No valid count_pred output found"));
                }
            };
            
            let max_count = count_logits.shape()[1];
            let mut pred_count = 0;
            let mut max_val = f32::MIN;
            for c in 0..max_count {
                let val = count_logits[[0, c]];
                if val > max_val {
                    max_val = val;
                    pred_count = c;
                }
            }

            if pred_count <= 0 {
                continue; // Nessuna estrazione per questo task
            }

            let mut schema_embs_data = Vec::with_capacity(num_labels * hidden_size);
            for &idx in &task_map.field_tok_indices {
                let label_emb = lhs_tensor.slice(s![0, idx, ..]);
                for &val in label_emb {
                    schema_embs_data.push(val);
                }
            }
            if schema_embs_data.len() != num_labels * hidden_size {
                schema_embs_data.resize(num_labels * hidden_size, 0.0);
            }
            let schema_embs = Array2::from_shape_vec((num_labels, hidden_size), schema_embs_data)?;

            let mut count_inputs_vec: Vec<(&str, Value<DynValueTypeMarker>)> = Vec::new();
            count_inputs_vec.push(("pc_emb", Tensor::from_array(schema_embs)?.into_dyn()));
            
            // Pass the required integer to any remaining input parameter.
            // In flawed PyTorch exports this is often named "onnx::Cast_1".
            // In corrected exports it's "gold_count_val" or similar.
            for input in &self.count_lstm.inputs {
                if input.name != "pc_emb" {
                    let gold_val = Array0::from_elem((), pred_count as i64);
                    count_inputs_vec.push((
                        input.name.as_str(), 
                        Tensor::from_array(gold_val)?.into_dyn()
                    ));
                }
            }
            let count_outputs = self.count_lstm.run(count_inputs_vec)?;
            let struct_proj = {
                if let Some(val) = count_outputs.get("count_embeddings") {
                    val.try_extract_tensor::<f32>()?.into_owned()
                } else if let Some(val) = count_outputs.get("output") {
                    val.try_extract_tensor::<f32>()?.into_owned()
                } else {
                    return Err(anyhow::anyhow!("No valid count_lstm output found"));
                }
            };
            let struct_proj_shape = struct_proj.shape();
            
            let mut proj_hidden = 0;
            let mut count_val_max = 1;
            let mut label_max = 0;

            if struct_proj_shape.len() >= 2 {
                proj_hidden = struct_proj_shape[1]; 
                label_max = struct_proj_shape[0];

                if struct_proj_shape.len() == 3 {
                    count_val_max = struct_proj_shape[0];
                    label_max = struct_proj_shape[1];
                    proj_hidden = struct_proj_shape[2];
                }
            }

            let span_hidden = span_emb_shape[3];
            
            // 5. Einsum Finale (Similarità e Probability)
            if proj_hidden == span_hidden && label_max >= num_labels && count_val_max > 0 {
                
                for c_idx in 0..count_val_max {
                    let mut c_matches = Vec::new();
                    
                    for start in 0..text_len {
                        for width_idx in 0..self.config.max_width {
                            let end = std::cmp::min(start + width_idx + 1, text_len);
                            
                            for m in 0..num_labels {
                                let mut logit = 0.0;
                                for d in 0..hidden_size {
                                    let span_val = span_embeddings[[0, start, width_idx, d]];
                                    let schema_val = if struct_proj_shape.len() == 3 {
                                        struct_proj[[c_idx, m, d]]
                                    } else {
                                        struct_proj[[m, d]]
                                    };
                                    logit += span_val * schema_val;
                                }
                                
                                let prob = 1.0 / (1.0 + (-logit).exp());
                                
                                if prob > 0.15 { // Lowered threshold for greater recall
                                    let original_start = record.word_to_token_maps[start].0;
                                    let original_end = record.word_to_token_maps[end - 1].1;
                                    
                                    if original_end <= record.input_ids.len() && original_start < original_end {
                                        let token_slice = &record.input_ids[original_start..original_end];
                                        let u32_tokens: Vec<u32> = token_slice.iter().map(|&x| x as u32).collect();
                                        let entity_text = self.tokenizer.decode(&u32_tokens, true).unwrap_or_default();
                                            
                                        if !entity_text.trim().is_empty() {
                                            c_matches.push(ExtractedEntity {
                                                score: prob,
                                                label: labels[m].to_string(),
                                                text: entity_text.trim().to_string(),
                                                start_tok: original_start,
                                                end_tok: original_end,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // 6. Non-Maximum Suppression (NMS) per rimuovere overlap di span fittizi
                    c_matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    let mut selected: Vec<ExtractedEntity> = Vec::new();
                    for m in c_matches {
                        let overlap = selected.iter().any(|s| !(m.end_tok <= s.start_tok || m.start_tok >= s.end_tok));
                        if !overlap {
                            selected.push(m);
                        }
                    }

                    if task_map.task_type == "entities" {
                        final_entities.extend(selected);
                    } else if task_map.task_type == "relations" {
                        let head = selected.iter().find(|x| x.label == "head");
                        let tail = selected.iter().find(|x| x.label == "tail");
                        if let (Some(h), Some(t)) = (head, tail) {
                            final_relations.push(ExtractedRelation {
                                head: h.clone(),
                                tail: t.clone(),
                                relation_type: task_map.labels[0].clone()
                            });
                        }
                    }
                }
            } else {
                 eprintln!("Errore di dimensionality Shape Mismatch su Einsum.");
            }
        }
        
        Ok((final_entities, final_relations, final_classifications))
    }
}

