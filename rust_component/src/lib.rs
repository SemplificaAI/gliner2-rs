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
//! (Entities, Relations, Classifications) using ONNX Runtime.
//!
//! Supported execution providers (in priority order, first available wins):
//!   - QNN  – Qualcomm AI Engine / Hexagon NPU (X Elite, Snapdragon 8cx, ecc.)
//!   - OpenVINO – Intel CPU/GPU/VPU
//!   - CoreML  – Apple Neural Engine + GPU (macOS, iOS)
//!   - CUDA    – NVIDIA GPU
//!   - ROCm    – AMD GPU (Linux + ROCm stack)
//!   - XNNPACK – CPU accelerato (ARM NEON, x86 AVX2)
//!   - CPU     – fallback generico

pub mod processor;
pub mod lib_v2;
pub use lib_v2::Gliner2EngineV2;

use anyhow::Result;
pub mod error;
pub use error::GlinerError;
use ndarray::{Array0, Array2, Array3, s};
use ort::{
    execution_providers::{
        CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
        OpenVINOExecutionProvider, QNNExecutionProvider, ROCmExecutionProvider,
        XNNPACKExecutionProvider,
    },
    session::{builder::GraphOptimizationLevel, Session},
    value::{Tensor, Value, DynValueTypeMarker},
};
use tokenizers::Tokenizer;
use std::path::Path;
use std::sync::RwLock;
use serde::Serialize;

use processor::SchemaTransformer;
pub use processor::SchemaTask;

/// Base configuration for initializing the engine.
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

/// GLiNER2 model type to handle different ONNX architectures.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    /// Converted PyTorch model (our server) - has last_hidden_state
    PyTorch,
    /// HuggingFace model (public download) - different architecture
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

/// Data for an entity extracted from the text.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractedEntity {
    pub text: String,
    pub label: String,
    pub score: f32,
    pub start_tok: usize,
    pub end_tok: usize,
}

/// Data of a relation between two entities.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractedRelation {
    pub head: ExtractedEntity,
    pub tail: ExtractedEntity,
    pub relation_type: String,
}

/// Global classification data on the examined text.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractedClassification {
    pub task_name: String,
    pub label: String,
    pub score: f32,
}

/// Advanced inference parameters.
#[derive(Debug, Clone, Copy)]
pub struct InferenceParams {
    pub threshold: f32,
    pub flat_ner: bool,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            flat_ner: false,
        }
    }
}


/// Main inference engine.
pub struct Gliner2EngineV1 {
    encoder: Session,
    span_rep: Session,
    count_lstm: Session,
    count_pred: Session,
    classifier: Session,
    tokenizer: Tokenizer,
    config: Gliner2Config,
    pub execution_mode: RwLock<ExecutionMode>,
}

impl Gliner2EngineV1 {
    /// Downloads the models and initializes the engine directly from HuggingFace Hub.
    /// The download includes the `User-Agent` header as specified in:
    /// https://huggingface.co/docs/hub/models-download-stats
    pub fn from_pretrained(
        repo_id: &str,
        subfolder: Option<&str>,
        model_type: ModelType,
    ) -> Result<Self> {
        // Header as per HF specs: <library_name>/<library_version>; <language_name>/<language_version>; <os_name>/<os_version>
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_user_agent(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
            .with_user_agent("rust", "unknown")
            .with_user_agent(std::env::consts::OS, "unknown")
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to initialize HF API: {}", e))?;

        let repo = api.model(repo_id.to_string());

        let is_fp16 = subfolder.unwrap_or("").contains("16");
        let suffix = if is_fp16 { "_fp16.onnx" } else { "_fp32.onnx" };

        let mut files_to_download = vec![
            format!("encoder{}", suffix),
            format!("span_rep{}", suffix),
            format!("count_pred{}", suffix),
            format!("classifier{}", suffix),
            "tokenizer.json".to_string(),
        ];

        if model_type == ModelType::PyTorch {
            files_to_download.push(format!("count_lstm{}", suffix));
        } else {
            files_to_download.push(format!("count_lstm{}", suffix));
        }

        let mut last_path = None;
        for file in &files_to_download {
            let repo_path = if let Some(sub) = subfolder {
                format!("{}/{}", sub, file)
            } else {
                file.clone()
            };

            println!("Downloading/verifying {}...", repo_path);
            match repo.get(&repo_path) {
                Ok(p) => last_path = Some(p),
                Err(e) => {
                    if file.starts_with("count_lstm") && model_type == ModelType::HuggingFace {
                        println!("Note: {} not found, using fallback.", repo_path);
                    } else {
                        return Err(anyhow::anyhow!("Failed to download {}: {}", repo_path, e));
                    }
                }
            }
        }

        let models_dir = if let Some(p) = last_path {
            p.parent()
                .ok_or_else(|| anyhow::anyhow!("Invalid file path"))?
                .to_string_lossy()
                .into_owned()
        } else {
            return Err(anyhow::anyhow!("No files downloaded"));
        };

        let config = Gliner2Config {
            models_dir,
            max_width: 8,
            model_type,
        };

        Self::new(config)
    }

    /// Initializes the neural networks by loading the ONNX files and the Tokenizer.
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

            let mut builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_memory_pattern(false)?;

            let force_cpu = std::env::var("FORCE_CPU").is_ok();
            
            if force_cpu {
                builder = builder.with_execution_providers([
                    QNNExecutionProvider::default().build(),
                    OpenVINOExecutionProvider::default().build(),
                    CoreMLExecutionProvider::default().build(),
                    XNNPACKExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build(),
                ])?;
            } else {
                builder = builder.with_execution_providers([
                    QNNExecutionProvider::default().build(),
                    OpenVINOExecutionProvider::default().build(),
                    CoreMLExecutionProvider::default().build(),
                    CUDAExecutionProvider::default().build(),
                    ROCmExecutionProvider::default().build(),
                    XNNPACKExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build(),
                ])?;
            }

            builder.commit_from_file(&path)
                .map_err(|e| anyhow::anyhow!("Error loading {:?}: {}", path, e))
        };

        // Load models based on model type
        let (count_pred, count_lstm) = match config.model_type {
            ModelType::PyTorch => {
                // Converted PyTorch model
                let count_lstm = load_session("count_lstm")?;
                let count_pred = load_session("count_pred")?;
                (count_pred, count_lstm)
            }
            ModelType::HuggingFace => {
                // Direct HuggingFace export (requires distinct layers)
                let count_lstm = load_session("count_lstm")?;
                let count_pred = load_session("count_pred")?;
                (count_pred, count_lstm)
            }
        };

        // Load common models
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
            config,
            execution_mode: RwLock::new(ExecutionMode::IoBinding),
        })
    }

    /// Executes the end-to-end flow on an input string
    /// based on the provided Schema Tasks. It tries IoBinding first,
    /// then falls back to Standard execution on OOM.
    pub fn extract(
        &self, 
        text: &str, 
        tasks: &[SchemaTask],
        params: Option<InferenceParams>
    ) -> anyhow::Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>, Vec<ExtractedClassification>)> {
        let current_mode = *self.execution_mode.read().unwrap();
        
        match current_mode {
            ExecutionMode::IoBinding => {
                match self.extract_iobinding(text, tasks, params.clone()) {
                    Ok(res) => Ok(res),
                    Err(GlinerError::OomDeviceBinding(msg)) => {
                        eprintln!("[GLiNER2] OOM in IoBinding detected. Falling back to Standard Mode. Details: {}", msg);
                        *self.execution_mode.write().unwrap() = ExecutionMode::Standard;
                        self.extract_standard(text, tasks, params)
                    },
                    Err(other) => Err(anyhow::anyhow!(other)),
                }
            },
            ExecutionMode::Standard => {
                self.extract_standard(text, tasks, params)
            }
        }
    }

    /// Fast path: IOBinding logic directly in VRAM.
    pub fn extract_iobinding(
        &self, 
        text: &str, 
        tasks: &[SchemaTask],
        _params: Option<InferenceParams>
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>, Vec<ExtractedClassification>), GlinerError> {
        // TODO: Full IOBinding implementation goes here.
        // For now, simulate failure to trigger fallback.
        Err(GlinerError::OomDeviceBinding("IOBinding non ancora implementato, fallback alla modalità Standard.".to_string()))
    }

    /// Safe path: Standard execution logic (data transferred to CPU between steps).
    pub fn extract_standard(
        &self, 
        text: &str, 
        tasks: &[SchemaTask],
        params: Option<InferenceParams>
    ) -> anyhow::Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>, Vec<ExtractedClassification>)> {
        let p = params.unwrap_or_default();
        let threshold = p.threshold;
        let flat_ner = p.flat_ner;
        
        // 1. Process prompt + text (token vector creation)
        let transformer = SchemaTransformer::new(self.tokenizer.clone());
        let record = transformer.transform(text, tasks)?;
        let seq_len = record.input_ids.len();

        let input_ids = Array2::from_shape_vec((1, seq_len), record.input_ids.clone())?;
        let attention_mask = Array2::from_shape_vec((1, seq_len), record.attention_mask.clone())?;

        // 2. Encoder pass (DeBERTa) -> Contextual Embeddings
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
        
        // Handle different outputs based on model type
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

        // Iterative generation of Span Index (combination trees)
        let num_spans = text_len * self.config.max_width;
        let mut span_idx_data: Vec<i64> = Vec::with_capacity(num_spans * 2);
        for start in 0..text_len {
            for width in 0..self.config.max_width {
                let end = start + width;
                if end >= text_len {
                    // Out-of-bounds pad for ONNX gather node safety
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

        // 4. Parallel Task Execution
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

            // 4a. Classification Branch (Full-text Softmax)
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

            // 4b. Count LSTM Branch (Entities and Relations)
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
                continue; // No extraction needed for this task
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
            
            // 5. Final Einsum (Similarity and Probability)
            if proj_hidden == span_hidden && label_max >= num_labels && count_val_max > 0 {
                let mut all_entity_matches = Vec::new();
                
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
                                
                                if prob > threshold {
                                    let original_start = record.word_to_token_maps[start].0;
                                    let original_end = record.word_to_token_maps[end - 1].1;
                                    
                                    let char_start = record.word_to_char_maps[start].0;
                                    let char_end = record.word_to_char_maps[end - 1].1;

                                    if original_end <= record.input_ids.len() && original_start < original_end {
                                        let entity_text = if char_start <= char_end && char_end <= text.len() {
                                            text[char_start..char_end].to_string()
                                        } else {
                                            String::new()
                                        };
                                        
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

                    if task_map.task_type == "entities" {
                        all_entity_matches.extend(c_matches);
                    } else if task_map.task_type == "relations" {
                        c_matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                        let mut selected: Vec<ExtractedEntity> = Vec::new();
                        for m in c_matches {
                            let overlap = selected.iter().any(|s| {
                            let spans_overlap = !(m.end_tok <= s.start_tok || m.start_tok >= s.end_tok);
                            spans_overlap && (flat_ner || s.label == m.label)
                        });
                            if !overlap {
                                selected.push(m);
                            }
                        }
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
                
                if task_map.task_type == "entities" {
                    all_entity_matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    let mut selected: Vec<ExtractedEntity> = Vec::new();
                    for m in all_entity_matches {
                        let overlap = selected.iter().any(|s| {
                        let spans_overlap = !(m.end_tok <= s.start_tok || m.start_tok >= s.end_tok);
                        spans_overlap && (flat_ner || s.label == m.label)
                    });
                        if !overlap {
                            selected.push(m);
                        }
                    }
                    final_entities.extend(selected);
                }
            } else {
                 eprintln!("Dimensionality Shape Mismatch Error on Einsum.");
            }
        }
        
        Ok((final_entities, final_relations, final_classifications))
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    IoBinding, // Modalita' 2
    Standard,  // Modalita' 1 (Fallback)
}

/// Universal Inference Engine for GLiNER2
/// Automatically detects whether the models are V1 (Standard) or V2 (IOBinding)
/// based on the contents of the model directory, and routes inference accordingly.
pub enum Gliner2Engine {
    V1(Gliner2EngineV1),
    V2(crate::lib_v2::Gliner2EngineV2),
}

impl Gliner2Engine {
    /// Downloads models from HuggingFace. Currently defaults to V1 models.
    pub fn from_pretrained(
        repo_id: &str,
        subfolder: Option<&str>,
        model_type: ModelType,
    ) -> Result<Self> {
        let is_v2 = subfolder.unwrap_or("").contains("v2");
        if is_v2 {
            let engine = crate::lib_v2::Gliner2EngineV2::from_pretrained(repo_id, subfolder, model_type)?;
            Ok(Gliner2Engine::V2(engine))
        } else {
            let engine = Gliner2EngineV1::from_pretrained(repo_id, subfolder, model_type)?;
            Ok(Gliner2Engine::V1(engine))
        }
    }

    /// Initializes the neural networks by loading the ONNX files and the Tokenizer.
    /// Automatically selects V1 or V2 engine based on the presence of V2 specific files.
    pub fn new(config: Gliner2Config) -> Result<Self> {
        let dir = Path::new(&config.models_dir);
        
        let is_v2 = dir.join("token_gather_fp16.onnx").exists() 
                 || dir.join("token_gather_fp32.onnx").exists()
                 || dir.join("token_gather_fp16_iobinding.onnx").exists();

        if is_v2 {
            println!("[GLiNER2 Autodetect] Modello V2 (Fuso) trovato. Avvio motore IOBinding...");
            Ok(Gliner2Engine::V2(crate::lib_v2::Gliner2EngineV2::new(config)?))
        } else {
            println!("[GLiNER2 Autodetect] Modello V1 (Standard) trovato. Avvio motore CPU-slicing...");
            Ok(Gliner2Engine::V1(Gliner2EngineV1::new(config)?))
        }
    }

    /// Executes the end-to-end flow on an input string
    /// based on the provided Schema Tasks.
    pub fn extract(
        &self, 
        text: &str, 
        tasks: &[SchemaTask],
        params: Option<InferenceParams>
    ) -> anyhow::Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>, Vec<ExtractedClassification>)> {
        match self {
            Gliner2Engine::V1(engine) => engine.extract(text, tasks, params),
            Gliner2Engine::V2(engine) => engine.extract(text, tasks, params),
        }
    }
}
