// Copyright 2026 Dario Finardi, Semplifica s.r.l.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! `Gliner2EngineV2` — engine IOBinding-ready che usa i modelli ONNX v2
//! generati da `export_gliner2_onnx_fragments_v2.py`.
//!
//! # Differenze rispetto a `Gliner2Engine` (v1)
//!
//! | v1 (`Gliner2Engine`)         | v2 (`Gliner2EngineV2`)                  |
//! |------------------------------|-----------------------------------------|
//! | 5 sessioni ONNX              | 8 sessioni ONNX                         |
//! | Gather/ArgMax/Einsum in Rust | Gather/ArgMax/Einsum fusi nell'ONNX     |
//! | `extract_iobinding` → stub   | `extract_iobinding` → implementata      |
//! | `extract_standard` completa  | `extract_standard` identica al v1       |
//!
//! # File ONNX attesi
//!
//! La directory dei modelli deve contenere i file generati dallo script v2:
//! - `encoder_fp16_iobinding.onnx`
//! - `token_gather_fp16_iobinding.onnx`
//! - `span_rep_fp16_iobinding.onnx`
//! - `schema_gather_fp16_iobinding.onnx`
//! - `count_pred_argmax_fp16_iobinding.onnx`
//! - `count_lstm_fixed_fp16_iobinding.onnx`
//! - `scorer_fp16_iobinding.onnx`
//! - `classifier_fp16_iobinding.onnx`
//! - `tokenizer.json`
//!
//! Oppure le varianti `_fp32.onnx` / `_fp16.onnx` per Standard mode / CoreML.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4};

/// Estrae un `ndarray::ArrayD<f32>` da un output ONNX, provando prima `f32` e
/// poi `f16` con conversione. Necessario perché i modelli v2 in formato
/// `fp16_iobinding` emettono output `Tensor<f16>` mentre senza questo fallback
/// `try_extract_array::<f32>()` fallirebbe con
/// `Cannot extract Tensor<f32> from Tensor<f16>` rompendo l'inferenza.
///
/// PATCH LOCALE — upstream crate non ha questo fallback su tutti i path.
macro_rules! extract_f32 {
    ($val:expr) => {{
        let __v = $val;
        if let Ok(t) = __v.try_extract_array::<f32>() {
            t.into_owned()
        } else if let Ok(t) = __v.try_extract_array::<half::f16>() {
            t.into_owned().mapv(|x| x.to_f32())
        } else {
            anyhow::bail!("tensor output: né f32 né f16");
        }
    }};
}
use ort::{
    execution_providers::{
        CPU, CUDA, CoreML,
        OpenVINO, ROCm,
        XNNPACK,
    },
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use tokenizers::Tokenizer;
use std::path::Path;
use std::sync::{Mutex, RwLock};

/// Converte un errore `ort` in `anyhow::Error`.
///
/// Da ort rc.13 `Error<R>` porta con sé un valore di *recovery* (il
/// SessionBuilder, la MemoryInfo, …) che non è Send+Sync: `?` verso
/// anyhow non compila più. Qui codice e messaggio vengono estratti e
/// l'errore diventa un anyhow piatto.
#[inline]
fn oa<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("ort [{:?}]: {}", e.code(), e.message())
}


use crate::{
    error::GlinerError,
    processor::{SchemaTask, SchemaTransformer},
    ExecutionMode, ExtractedClassification, ExtractedEntity, ExtractedRelation, Gliner2Config,
    InferenceParams,
};

// ─────────────────────────────────────────────────────────────────────────────
// Struct principale
// ─────────────────────────────────────────────────────────────────────────────

/// Engine IOBinding-ready per modelli ONNX v2.
pub struct Gliner2EngineV2 {
    encoder: Mutex<Session>,
    token_gather: Mutex<Session>,
    span_rep: Mutex<Session>,
    schema_gather: Mutex<Session>,
    count_pred_argmax: Mutex<Session>,
    count_lstm_fixed: Mutex<Session>,
    scorer: Mutex<Session>,
    classifier: Mutex<Session>,
    tokenizer:          Tokenizer,
    config:             Gliner2Config,
    pub execution_mode: RwLock<ExecutionMode>,
}


impl Gliner2EngineV2 {
    /// Scarica i modelli V2 da Hugging Face
    pub fn from_pretrained(
        repo_id: &str,
        subfolder: Option<&str>,
        model_type: crate::ModelType,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_user_agent(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
            .with_user_agent("rust", "unknown")
            .with_user_agent(std::env::consts::OS, "unknown")
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to initialize HF API: {}", e))?;

        let repo = api.model(repo_id.to_string());

        let is_fp16 = subfolder.unwrap_or("").contains("16");
        
        // Optimize download size: download only IOBinding files for CUDA/ROCm platforms, 
        // and standard FP16 files for CoreML (Apple) platforms or if explicitly requested.
        let is_apple = std::env::consts::OS == "macos" || std::env::consts::OS == "ios";
        let no_iobinding = std::env::var("GLINER2_NO_IOBINDING").is_ok();
        let prefer_iobinding = is_fp16 && !is_apple && !no_iobinding;

        let suffix = if is_fp16 { 
            if prefer_iobinding { "_fp16_iobinding.onnx" } else { "_fp16.onnx" }
        } else { 
            "_fp32.onnx" 
        };

        let mut files_to_download = vec![
            "tokenizer.json".to_string(),
        ];
        
        let bases = [
            "encoder",
            "token_gather",
            "span_rep",
            "schema_gather",
            "count_pred_argmax",
            "count_lstm_fixed",
            "scorer",
            "classifier"
        ];
        
        for base in bases.iter() {
            files_to_download.push(format!("{}{}", base, suffix));
        }

        let mut models_dir = std::path::PathBuf::new();

        for file in files_to_download {
            let path_in_repo = match subfolder {
                Some(sub) => format!("{}/{}", sub, file),
                None => file.clone(),
            };

            println!("Downloading/verifying {}...", path_in_repo);
            let local_path = repo.get(&path_in_repo).map_err(|e| {
                anyhow::anyhow!("Failed to download {}: {}", path_in_repo, e)
            })?;

            if models_dir.as_os_str().is_empty() {
                if let Some(parent) = local_path.parent() {
                    models_dir = parent.to_path_buf();
                }
            }
        }

        let config = Gliner2Config {
            models_dir: models_dir.to_string_lossy().to_string(),
            max_width: 8,
            model_type,
        };

        Self::new(config)
    }

    // ── Costruzione ──────────────────────────────────────────────────────────

    /// Carica l'engine dalla directory dei modelli v2.
    ///
    /// Cerca automaticamente nella priorità: `_fp16_iobinding.onnx`,
    /// poi `_fp16.onnx`, poi `_fp32.onnx`.
    pub fn new(config: Gliner2Config) -> Result<Self> {
        let dir = Path::new(&config.models_dir);

        let load = |base: &str| -> Result<Session> {
            // Priorità: iobinding FP16 > FP16 > FP32
            let candidates = [
                dir.join(format!("{}_fp16_iobinding.onnx", base)),
                dir.join(format!("{}_fp16.onnx", base)),
                dir.join(format!("{}_fp32.onnx", base)),
            ];

            let path = candidates.iter()
                .find(|p| p.exists())
                .ok_or_else(|| anyhow::anyhow!(
                    "Nessun modello trovato per '{}' in {:?}", base, dir
                ))?;

            let force_cpu = std::env::var("FORCE_CPU").is_ok();

            // QNN/Hexagon NPU **NON usato** per GLiNER2: i modelli sono
            // FP32 e non possono essere quantizzati INT8 senza perdere
            // precisione (verifica utente 2026-04-27). HTP ottimizza
            // principalmente INT8, su FP32 il guadagno è negativo.
            // Bench Snapdragon X Elite: CPU vince ~10% vs QNN/HTP.
            // Vedi memory `reference_qnn_npu_benchmark_2026_04_27`.
            let mut builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3).map_err(oa)?
                .with_memory_pattern(false).map_err(oa)?;

            if force_cpu {
                builder = builder.with_execution_providers([
                    OpenVINO::default().build(),
                    CoreML::default().build(),
                    XNNPACK::default().build(),
                    CPU::default().build(),
                ]).map_err(oa)?;
            } else {
                // Multi-backend cross-platform per future build:
                // - OpenVINO (Intel CPU+iGPU)
                // - CoreML (Apple Silicon ANE+GPU)
                // - CUDA (NVIDIA discrete)
                // - ROCm (AMD)
                // - XNNPACK (CPU SIMD ottimizzato)
                // - CPU (fallback finale)
                builder = builder.with_execution_providers([
                    OpenVINO::default().build(),
                    CoreML::default().build(),
                    CUDA::default().build(),
                    ROCm::default().build(),
                    XNNPACK::default().build(),
                    CPU::default().build(),
                ]).map_err(oa)?;
            }

            println!("  Caricamento {:?}", path.file_name().unwrap());
            builder.commit_from_file(path)
                .map_err(|e| anyhow::anyhow!("Errore caricamento {:?}: {}", path, e))
        };

        let encoder           = load("encoder")?;
        let token_gather      = load("token_gather")?;
        let span_rep          = load("span_rep")?;
        let schema_gather     = load("schema_gather")?;
        let count_pred_argmax = load("count_pred_argmax")?;
        let count_lstm_fixed  = load("count_lstm_fixed")?;
        let scorer            = load("scorer")?;
        let classifier        = load("classifier")?;

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Errore tokenizer: {}", e))?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            token_gather: Mutex::new(token_gather),
            span_rep: Mutex::new(span_rep),
            schema_gather: Mutex::new(schema_gather),
            count_pred_argmax: Mutex::new(count_pred_argmax),
            count_lstm_fixed: Mutex::new(count_lstm_fixed),
            scorer: Mutex::new(scorer),
            classifier: Mutex::new(classifier),
            tokenizer,
            config,
            execution_mode: RwLock::new(ExecutionMode::IoBinding),
        })
    }

    // ── API pubblica ─────────────────────────────────────────────────────────

    /// Smart wrapper: tenta IOBinding, fallback silenzioso a Standard su OOM.
    pub fn extract(
        &self,
        text: &str,
        tasks: &[SchemaTask],
        params: Option<InferenceParams>,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>, Vec<ExtractedClassification>)> {
        let mode = *self.execution_mode.read().unwrap();
        match mode {
            ExecutionMode::IoBinding => {
                match self.extract_iobinding(text, tasks, params) {
                    Ok(res) => Ok(res),
                    Err(GlinerError::OomDeviceBinding(msg)) => {
                        eprintln!(
                            "[GLiNER2-v2] OOM IOBinding, fallback Standard. Dettagli: {}",
                            msg
                        );
                        *self.execution_mode.write().unwrap() = ExecutionMode::Standard;
                        self.extract_standard(text, tasks, params)
                    }
                    Err(other) => Err(anyhow::anyhow!(other)),
                }
            }
            ExecutionMode::Standard => self.extract_standard(text, tasks, params),
        }
    }

    // ── IOBinding ────────────────────────────────────────────────────────────

    /// Esegue la pipeline v2 con IOBinding Zero-Copy.
    ///
    /// I tensori float intermedi restano in memoria device (VRAM/NPU) tra un
    /// layer e il successivo. Gli unici trasferimenti device→host sono:
    ///   - `pred_count` (8 byte int64) per ogni task
    ///   - `entity_scores` (per NMS + soglia su CPU)
    ///
    /// Se qualsiasi passo di binding o esecuzione fallisce (dispositivo
    /// non disponibile, OOM, EP non supporta IOBinding), l'errore viene
    /// convertito in `GlinerError::OomDeviceBinding` e `extract()` effettua
    /// automaticamente il fallback a `extract_standard()`.
    ///
    /// # Nota ROCm
    /// Attualmente prova prima CUDA (`AllocationDevice::CUDA`).
    /// Su macchine ROCm-only il primo `run()` fallisce → fallback Standard.
    /// TODO: rilevare l'EP attivo e usare `AllocationDevice::HIP` per ROCm.
    fn extract_iobinding(
        &self,
        text: &str,
        tasks: &[SchemaTask],
        params: Option<InferenceParams>,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>, Vec<ExtractedClassification>), GlinerError> {
        // Un solo lock per sessione, preso qui: Mutex non rientrante.
        let mut g_encoder = self.encoder.lock().unwrap();
        let mut g_token_gather = self.token_gather.lock().unwrap();
        let mut g_span_rep = self.span_rep.lock().unwrap();
        let mut g_schema_gather = self.schema_gather.lock().unwrap();
        let mut g_count_pred_argmax = self.count_pred_argmax.lock().unwrap();
        let mut g_count_lstm_fixed = self.count_lstm_fixed.lock().unwrap();
        let mut g_scorer = self.scorer.lock().unwrap();
        let mut g_classifier = self.classifier.lock().unwrap();
        let p = params.unwrap_or_default();
        let threshold = p.threshold;
        let flat_ner = p.flat_ner;
        // Mappa errori ORT → OomDeviceBinding (qualsiasi fallimento qui → fallback Standard)
        macro_rules! oe {
            ($expr:expr, $ctx:literal) => {
                $expr.map_err(|e| GlinerError::OomDeviceBinding(
                    format!("{}: {}", $ctx, e)
                ))?
            };
        }

        // ── MemoryInfo ────────────────────────────────────────────────────────
        // device_mem: tensori intermedi restano su GPU/NPU
        // cpu_out_mem: tensori che devono rientrare su CPU (pred_count, scores)
        let device_mem = oe!(
            MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default),
            "device MemoryInfo"
        );
        let cpu_out_mem = oe!(
            MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput),
            "cpu_out MemoryInfo"
        );

        // ── Tokenizzazione ────────────────────────────────────────────────────
        let transformer = SchemaTransformer::new(self.tokenizer.clone());
        let record = transformer.transform(text, tasks)
            .map_err(|e| GlinerError::Other(e))?;
        let seq_len = record.input_ids.len();

        let num_words = record.word_to_token_maps.len();
        if num_words == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        // ── Step 1: Encoder ───────────────────────────────────────────────────
        let input_ids_arr = Array2::from_shape_vec((1, seq_len), record.input_ids.clone())
            .map_err(|e| GlinerError::Other(anyhow::anyhow!(e)))?;
        let attn_mask_arr = Array2::from_shape_vec((1, seq_len), record.attention_mask.clone())
            .map_err(|e| GlinerError::Other(anyhow::anyhow!(e)))?;

        let input_ids_t = oe!(Tensor::from_array(input_ids_arr), "encoder input_ids tensor");
        let attn_mask_t = oe!(Tensor::from_array(attn_mask_arr), "encoder attn_mask tensor");

        let enc_out_name = g_encoder.outputs().get(0)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("encoder: nessun output registrato".into()))?;

        let mut b_enc = oe!(g_encoder.create_binding(), "encoder create_binding");
        oe!(b_enc.bind_input("input_ids", &input_ids_t), "encoder bind input_ids");
        oe!(b_enc.bind_input("attention_mask", &attn_mask_t), "encoder bind attention_mask");
        oe!(b_enc.bind_output_to_device(&enc_out_name, &device_mem), "encoder bind output");

        let hs_val = {
            let mut out = oe!(g_encoder.run_binding(&b_enc), "encoder run");
            out.remove(enc_out_name.as_str())
                .ok_or_else(|| GlinerError::OomDeviceBinding(
                    format!("encoder: output '{}' non trovato", enc_out_name)
                ))?
        };

        // ── Step 2: TokenGather ───────────────────────────────────────────────
        let word_starts: Vec<i64> = record.word_to_token_maps.iter()
            .map(|&(s, _)| s as i64).collect();
        let word_idx_t = oe!(
            Tensor::from_array(Array1::from_vec(word_starts)),
            "token_gather word_idx tensor"
        );

        let tg_out_name = g_token_gather.outputs().get(0)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("token_gather: nessun output".into()))?;

        let mut b_tg = oe!(g_token_gather.create_binding(), "token_gather create_binding");
        oe!(b_tg.bind_input("last_hidden_state", &hs_val), "token_gather bind last_hidden_state");
        oe!(b_tg.bind_input("word_indices", &word_idx_t), "token_gather bind word_indices");
        oe!(b_tg.bind_output_to_device(&tg_out_name, &device_mem), "token_gather bind output");

        let text_embs_val = {
            let mut out = oe!(g_token_gather.run_binding(&b_tg), "token_gather run");
            out.remove(tg_out_name.as_str())
                .ok_or_else(|| GlinerError::OomDeviceBinding(
                    format!("token_gather: output '{}' non trovato", tg_out_name)
                ))?
        };

        // ── Step 3: SpanRep ───────────────────────────────────────────────────
        let num_spans = num_words * self.config.max_width;
        let mut span_idx_data = Vec::with_capacity(num_spans * 2);
        for start in 0..num_words {
            for width in 0..self.config.max_width {
                let end = start + width;
                if end >= num_words {
                    span_idx_data.extend_from_slice(&[0i64, 0i64]);
                } else {
                    span_idx_data.push(start as i64);
                    span_idx_data.push(end as i64);
                }
            }
        }
        let span_idx_arr = Array3::from_shape_vec((1, num_spans, 2), span_idx_data)
            .map_err(|e| GlinerError::Other(anyhow::anyhow!(e)))?;
        let span_idx_t = oe!(Tensor::from_array(span_idx_arr), "span_rep span_idx tensor");

        let sr_out_name = g_span_rep.outputs().get(0)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("span_rep: nessun output".into()))?;

        let mut b_sr = oe!(g_span_rep.create_binding(), "span_rep create_binding");
        oe!(b_sr.bind_input("hidden_states", &text_embs_val), "span_rep bind hidden_states");
        oe!(b_sr.bind_input("span_idx", &span_idx_t), "span_rep bind span_idx");
        oe!(b_sr.bind_output_to_device(&sr_out_name, &device_mem), "span_rep bind output");

        let span_embs_val = {
            let mut out = oe!(g_span_rep.run_binding(&b_sr), "span_rep run");
            out.remove(sr_out_name.as_str())
                .ok_or_else(|| GlinerError::OomDeviceBinding(
                    format!("span_rep: output '{}' non trovato", sr_out_name)
                ))?
        };

        // ── Nomi output sessioni per-task (raccolti una sola volta) ───────────
        let sg_pc_name = g_schema_gather.outputs().get(0)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("schema_gather: nessun output 0".into()))?;
        let sg_fe_name = g_schema_gather.outputs().get(1)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("schema_gather: nessun output 1".into()))?;
        let cp_out_name = g_count_pred_argmax.outputs().get(0)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("count_pred_argmax: nessun output".into()))?;
        let cl_out_name = g_count_lstm_fixed.outputs().get(0)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("count_lstm_fixed: nessun output".into()))?;
        let sc_out_name = g_scorer.outputs().get(0)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("scorer: nessun output".into()))?;
        let cls_out_name = g_classifier.outputs().get(0)
            .map(|o| o.name().to_string())
            .ok_or_else(|| GlinerError::OomDeviceBinding("classifier: nessun output".into()))?;

        // ── Step 4: Loop per-task ─────────────────────────────────────────────
        let mut final_entities       = Vec::new();
        let mut final_relations      = Vec::new();
        let mut final_classifications = Vec::new();

        for task_map in &record.tasks {
            let num_labels = task_map.labels.len();
            let is_cls = task_map.task_type == "classifications";

            // 4a. SchemaGather
            let schema_indices: Vec<i64> =
                std::iter::once(task_map.prompt_tok_idx as i64)
                .chain(task_map.field_tok_indices.iter().map(|&i| i as i64))
                .collect();
            let schema_idx_t = oe!(
                Tensor::from_array(Array1::from_vec(schema_indices)),
                "schema_gather schema_indices tensor"
            );

            // field_embs: device per entity, CPU per classification (serve sul host per padding)
            let field_mem: &MemoryInfo = if is_cls { &cpu_out_mem } else { &device_mem };

            let mut b_sg = oe!(g_schema_gather.create_binding(), "schema_gather create_binding");
            oe!(b_sg.bind_input("last_hidden_state", &hs_val), "schema_gather bind last_hidden_state");
            oe!(b_sg.bind_input("schema_indices", &schema_idx_t), "schema_gather bind schema_indices");
            oe!(b_sg.bind_output_to_device(&sg_pc_name, &device_mem), "schema_gather bind pc_emb");
            oe!(b_sg.bind_output_to_device(&sg_fe_name, field_mem), "schema_gather bind field_embs");

            let (pc_emb_val, field_embs_val) = {
                let mut out = oe!(g_schema_gather.run_binding(&b_sg), "schema_gather run");
                let pc = out.remove(sg_pc_name.as_str())
                    .ok_or_else(|| GlinerError::OomDeviceBinding(
                        format!("schema_gather: output '{}' non trovato", sg_pc_name)
                    ))?;
                let fe = out.remove(sg_fe_name.as_str())
                    .ok_or_else(|| GlinerError::OomDeviceBinding(
                        format!("schema_gather: output '{}' non trovato", sg_fe_name)
                    ))?;
                (pc, fe)
            };

            // 4b. CountPredArgmax → output su CPU (int64 scalare, 8 byte)
            let mut b_cp = oe!(g_count_pred_argmax.create_binding(), "count_pred_argmax create_binding");
            oe!(b_cp.bind_input("pc_emb", &pc_emb_val), "count_pred_argmax bind pc_emb");
            oe!(b_cp.bind_output_to_device(&cp_out_name, &cpu_out_mem), "count_pred_argmax bind output");

            let pred_count: usize = {
                let mut out = oe!(g_count_pred_argmax.run_binding(&b_cp), "count_pred_argmax run");
                let val = out.remove(cp_out_name.as_str())
                    .ok_or_else(|| GlinerError::OomDeviceBinding(
                        format!("count_pred_argmax: output '{}' non trovato", cp_out_name)
                    ))?;
                oe!(
                    val.try_extract_array::<i64>().map_err(|e| anyhow::anyhow!(e)),
                    "count_pred_argmax extract i64"
                ).into_owned()[[0]] as usize
            };

            if pred_count == 0 {
                continue;
            }

            // 4c. Task classificazione — field_embs è già su CPU (cpu_out_mem sopra)
            if is_cls {
                let fe_arr = if let Ok(t) = field_embs_val.try_extract_array::<f32>() {
                    t.into_owned()
                } else if let Ok(t) = field_embs_val.try_extract_array::<half::f16>() {
                    t.into_owned().mapv(|x| x.to_f32())
                } else {
                    return Err(GlinerError::OomDeviceBinding("field_embs format error".into()));
                };
                let hidden_size = fe_arr.shape()[1];

                let mut padded = Array4::<half::f16>::from_elem((1, num_labels, self.config.max_width, hidden_size), half::f16::from_f32(0.0));
                for m in 0..num_labels {
                    for d in 0..hidden_size {
                        padded[[0, m, 0, d]] = half::f16::from_f32(fe_arr[[m, d]]);
                    }
                }
                let padded_t = oe!(Tensor::from_array(padded), "cls padded tensor");

                let mut b_cls = oe!(g_classifier.create_binding(), "classifier create_binding");
                oe!(b_cls.bind_input("span_embeddings", &padded_t), "classifier bind span_embeddings");
                oe!(b_cls.bind_output_to_device(&cls_out_name, &cpu_out_mem), "classifier bind output");

                let logits = {
                    let mut out = oe!(g_classifier.run_binding(&b_cls), "classifier run");
                    let val = out.remove(cls_out_name.as_str())
                        .ok_or_else(|| GlinerError::OomDeviceBinding(
                            format!("classifier: output '{}' non trovato", cls_out_name)
                        ))?;
                    if let Ok(t) = val.try_extract_array::<f32>() {
                        t.into_owned()
                    } else if let Ok(t) = val.try_extract_array::<half::f16>() {
                        t.into_owned().mapv(|x| x.to_f32())
                    } else {
                        return Err(GlinerError::OomDeviceBinding("classifier extract: type error".into()));
                    }
                };

                let mut exp_sum = 0.0f32;
                let exps: Vec<f32> = (0..num_labels).map(|m| {
                    let e = logits[[0, m, 0, 0]].exp();
                    exp_sum += e;
                    e
                }).collect();

                let (best_idx, best_score) = exps.iter().enumerate()
                    .map(|(i, &e)| (i, e / exp_sum))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap_or((0, 0.0));

                final_classifications.push(ExtractedClassification {
                    task_name: task_map.task_name.clone(),
                    label:     task_map.labels[best_idx].clone(),
                    score:     best_score,
                });
                continue;
            }

            // 4d. CountLSTMFixed → struct_proj su device
            let mut b_cl = oe!(g_count_lstm_fixed.create_binding(), "count_lstm_fixed create_binding");
            oe!(b_cl.bind_input("field_embs", &field_embs_val), "count_lstm_fixed bind field_embs");
            oe!(b_cl.bind_output_to_device(&cl_out_name, &device_mem), "count_lstm_fixed bind output");

            let struct_proj_val = {
                let mut out = oe!(g_count_lstm_fixed.run_binding(&b_cl), "count_lstm_fixed run");
                out.remove(cl_out_name.as_str())
                    .ok_or_else(|| GlinerError::OomDeviceBinding(
                        format!("count_lstm_fixed: output '{}' non trovato", cl_out_name)
                    ))?
            };

            // 4e. Scorer → entity_scores su CPU (per NMS)
            let mut b_sc = oe!(g_scorer.create_binding(), "scorer create_binding");
            oe!(b_sc.bind_input("span_embeddings", &span_embs_val), "scorer bind span_embeddings");
            oe!(b_sc.bind_input("struct_proj", &struct_proj_val), "scorer bind struct_proj");
            oe!(b_sc.bind_output_to_device(&sc_out_name, &cpu_out_mem), "scorer bind output");

            let scores = {
                let mut out = oe!(g_scorer.run_binding(&b_sc), "scorer run");
                let val = out.remove(sc_out_name.as_str())
                    .ok_or_else(|| GlinerError::OomDeviceBinding(
                        format!("scorer: output '{}' non trovato", sc_out_name)
                    ))?;
                if let Ok(t) = val.try_extract_array::<f32>() {
                    t.into_owned()
                } else if let Ok(t) = val.try_extract_array::<half::f16>() {
                    t.into_owned().mapv(|x| x.to_f32())
                } else {
                    return Err(GlinerError::OomDeviceBinding("scorer extract: type error".into()));
                }
            };
            // scores: [MAX_COUNT, num_words, max_width, M]  (sigmoid già applicato nell'ONNX)

            // 4f. NMS greedy + soglia (identico a extract_standard)
            let mut all_matches: Vec<ExtractedEntity> = Vec::new();

            for c_idx in 0..pred_count {
                for start in 0..num_words {
                    for width_idx in 0..self.config.max_width {
                        let end = std::cmp::min(start + width_idx + 1, num_words);
                        for m in 0..num_labels {
                            let prob = scores[[c_idx, start, width_idx, m]];
                            if prob > threshold {
                                let char_start = record.word_to_char_maps[start].0;
                                let char_end   = record.word_to_char_maps[end - 1].1;
                                let orig_start = record.word_to_token_maps[start].0;
                                let orig_end   = record.word_to_token_maps[end - 1].1;

                                if char_start <= char_end && char_end <= text.len() {
                                    let entity_text = text[char_start..char_end].trim().to_string();
                                    if !entity_text.is_empty() {
                                        all_matches.push(ExtractedEntity {
                                            score:     prob,
                                            label:     task_map.labels[m].clone(),
                                            text:      entity_text,
                                            start_tok: orig_start,
                                            end_tok:   orig_end,
                                            start_char: char_start,
                                            end_char: char_end,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            all_matches.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut selected: Vec<ExtractedEntity> = Vec::new();
            for candidate in all_matches {
                let overlap = selected.iter().any(|s| {
                    let spans_overlap = !(candidate.end_tok <= s.start_tok || candidate.start_tok >= s.end_tok);
                    spans_overlap && (flat_ner || s.label == candidate.label)
                });
                if !overlap {
                    selected.push(candidate);
                }
            }

            if task_map.task_type == "entities" {
                final_entities.extend(selected);
            } else if task_map.task_type == "relations" {
                let head = selected.iter().find(|x| x.label == "head").cloned();
                let tail = selected.iter().find(|x| x.label == "tail").cloned();
                if let (Some(h), Some(t)) = (head, tail) {
                    final_relations.push(ExtractedRelation {
                        head:          h,
                        tail:          t,
                        relation_type: task_map.labels[0].clone(),
                    });
                }
            }
        }

        Ok((final_entities, final_relations, final_classifications))
    }

    // ── Standard mode (identico al v1) ───────────────────────────────────────

    /// Esegue la pipeline con trasferimenti CPU↔device espliciti tra ogni layer.
    /// Usa le sessioni v2 (8 modelli) ma con il pattern Standard del v1:
    /// ogni output viene estratto su CPU e passato come input al layer successivo.
    ///
    /// Questa implementazione garantisce correttezza e retrocompatibilità
    /// indipendentemente dall'hardware disponibile.
    pub fn extract_standard(
        &self,
        text: &str,
        tasks: &[SchemaTask],
        params: Option<InferenceParams>,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>, Vec<ExtractedClassification>)> {
        // Un solo lock per sessione, preso qui: Mutex non rientrante.
        let mut g_encoder = self.encoder.lock().unwrap();
        let mut g_token_gather = self.token_gather.lock().unwrap();
        let mut g_span_rep = self.span_rep.lock().unwrap();
        let mut g_schema_gather = self.schema_gather.lock().unwrap();
        let mut g_count_pred_argmax = self.count_pred_argmax.lock().unwrap();
        let mut g_count_lstm_fixed = self.count_lstm_fixed.lock().unwrap();
        let mut g_scorer = self.scorer.lock().unwrap();
        let mut g_classifier = self.classifier.lock().unwrap();
        let p = params.unwrap_or_default();
        let threshold = p.threshold;
        let flat_ner = p.flat_ner;
        let transformer = SchemaTransformer::new(self.tokenizer.clone());
        let record = transformer.transform(text, tasks)?;
        let seq_len = record.input_ids.len();

        // 1. Encoder
        let input_ids   = Array2::from_shape_vec((1, seq_len), record.input_ids.clone())?;
        let attn_mask   = Array2::from_shape_vec((1, seq_len), record.attention_mask.clone())?;
        let enc_out     = g_encoder.run(ort::inputs![
            "input_ids"      => Tensor::from_array(input_ids)?,
            "attention_mask" => Tensor::from_array(attn_mask)?
        ])?;
        let hs = {
            let mut found = None;
            for name in ["hidden_states", "last_hidden_state", "output"] {
                if let Some(v) = enc_out.get(name) {
                    found = Some(extract_f32!(v));
                    break;
                }
            }
            match found {
                Some(a) => a,
                None => {
                    let v_fallback = enc_out.values().next()
                        .ok_or_else(|| anyhow::anyhow!("encoder: nessun output trovato"))?;
                    extract_f32!(v_fallback)
                }
            }
        };

        let num_words = record.word_to_token_maps.len();
        if num_words == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let hidden_size = hs.shape()[2];

        // 2. TokenGather — estrae embedding word-level da hs
        let word_starts: Vec<i64> = record.word_to_token_maps.iter()
            .map(|&(s, _)| s as i64).collect();
        let word_idx_arr = Array1::from_vec(word_starts);
        let tg_out = g_token_gather.run(ort::inputs![
            "last_hidden_state" => Tensor::from_array(hs.clone())?,
            "word_indices"      => Tensor::from_array(word_idx_arr)?
        ])?;
        let text_embs = {
            let v = tg_out.values().next()
                .ok_or_else(|| anyhow::anyhow!("token_gather: nessun output"))?;
            extract_f32!(v)
        };

        // 3. SpanRep
        let num_spans = num_words * self.config.max_width;
        let mut span_idx_data = Vec::with_capacity(num_spans * 2);
        for start in 0..num_words {
            for width in 0..self.config.max_width {
                let end = start + width;
                if end >= num_words {
                    span_idx_data.extend_from_slice(&[0i64, 0i64]);
                } else {
                    span_idx_data.push(start as i64);
                    span_idx_data.push(end as i64);
                }
            }
        }
        let span_idx_arr = Array3::from_shape_vec((1, num_spans, 2), span_idx_data)?;
        let sr_out = g_span_rep.run(ort::inputs![
            "hidden_states" => Tensor::from_array(text_embs)?,
            "span_idx"      => Tensor::from_array(span_idx_arr)?
        ])?;
        let span_embs = {
            let v = sr_out.values().next()
                .ok_or_else(|| anyhow::anyhow!("span_rep: nessun output"))?;
            extract_f32!(v)
        };

        // 4. Loop per-task
        let mut final_entities       = Vec::new();
        let mut final_relations      = Vec::new();
        let mut final_classifications = Vec::new();

        for task_map in &record.tasks {
            let num_labels = task_map.labels.len();

            // 4a. SchemaGather — un solo Gather per prompt + campi
            let schema_indices: Vec<i64> =
                std::iter::once(task_map.prompt_tok_idx as i64)
                .chain(task_map.field_tok_indices.iter().map(|&i| i as i64))
                .collect();
            let schema_idx_arr = Array1::from_vec(schema_indices);
            let sg_out = g_schema_gather.run(ort::inputs![
                "last_hidden_state" => Tensor::from_array(hs.clone())?,
                "schema_indices"    => Tensor::from_array(schema_idx_arr)?
            ])?;
            let mut sg_iter = sg_out.values();
            let pc_emb = {
                let v = sg_iter.next()
                    .ok_or_else(|| anyhow::anyhow!("schema_gather: pc_emb mancante"))?;
                extract_f32!(v)   // [1, H]
            };
            let field_embs = {
                let v = sg_iter.next()
                    .ok_or_else(|| anyhow::anyhow!("schema_gather: field_embs mancante"))?;
                extract_f32!(v)   // [M, H]
            };

            // 4b. CountPredArgmax — restituisce int64
            let cp_out = g_count_pred_argmax.run(ort::inputs![
                "pc_emb" => Tensor::from_array(pc_emb)?
            ])?;
            let pred_count = cp_out.values().next()
                .ok_or_else(|| anyhow::anyhow!("count_pred_argmax: nessun output"))?
                .try_extract_array::<i64>()?.into_owned()[[0]] as usize;

            if pred_count == 0 {
                continue;
            }

            // 4c. Task classificazione: usa classifier invece di scorer
            if task_map.task_type == "classifications" {
                let _span_emb_shape = span_embs.shape();
                let mut padded = ndarray::Array4::<half::f16>::from_elem(
                    (1, num_labels, self.config.max_width, hidden_size), half::f16::from_f32(0.0)
                );
                for m in 0..num_labels {
                    for d in 0..hidden_size {
                        padded[[0, m, 0, d]] = half::f16::from_f32(field_embs[[m, d]]);
                    }
                }
                let cls_out = g_classifier.run(ort::inputs![
                    "span_embeddings" => Tensor::from_array(padded)?
                ])?;
                let logits = {
                    let v = cls_out.values().next()
                        .ok_or_else(|| anyhow::anyhow!("classifier: nessun output"))?;
                    extract_f32!(v)
                };

                let mut exp_sum = 0.0f32;
                let exps: Vec<f32> = (0..num_labels).map(|m| {
                    let e = logits[[0, m, 0, 0]].exp();
                    exp_sum += e;
                    e
                }).collect();

                let (best_idx, best_score) = exps.iter().enumerate()
                    .map(|(i, &e)| (i, e / exp_sum))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap_or((0, 0.0));

                final_classifications.push(ExtractedClassification {
                    task_name: task_map.task_name.clone(),
                    label: task_map.labels[best_idx].clone(),
                    score: best_score,
                });
                continue;
            }

            // 4d. CountLSTMFixed — output sempre [MAX_COUNT, M, H]
            let cl_out = g_count_lstm_fixed.run(ort::inputs![
                "field_embs" => Tensor::from_array(field_embs)?
            ])?;
            let struct_proj = {
                let v = cl_out.values().next()
                    .ok_or_else(|| anyhow::anyhow!("count_lstm_fixed: nessun output"))?;
                extract_f32!(v)
            };
            // struct_proj: [MAX_COUNT, M, H] — usiamo solo [:pred_count]

            // 4e. Scorer — restituisce probabilità sigmoid già calcolate
            //     entity_scores: [MAX_COUNT, num_words, max_width, M]
            let sc_out = g_scorer.run(ort::inputs![
                "span_embeddings" => Tensor::from_array(span_embs.clone())?,
                "struct_proj"     => Tensor::from_array(struct_proj)?
            ])?;
            let scores = {
                let v = sc_out.values().next()
                    .ok_or_else(|| anyhow::anyhow!("scorer: nessun output"))?;
                extract_f32!(v)
            };
            // scores: [MAX_COUNT, num_words, max_width, M]

            // 4f. NMS + soglia (identico al v1, ma ora scores sono già sigmoid)
            let mut all_matches: Vec<ExtractedEntity> = Vec::new();

            for c_idx in 0..pred_count {
                for start in 0..num_words {
                    for width_idx in 0..self.config.max_width {
                        let end = std::cmp::min(start + width_idx + 1, num_words);
                        for m in 0..num_labels {
                            let prob = scores[[c_idx, start, width_idx, m]];
                            if prob > threshold {
                                let char_start = record.word_to_char_maps[start].0;
                                let char_end   = record.word_to_char_maps[end - 1].1;
                                let orig_start = record.word_to_token_maps[start].0;
                                let orig_end   = record.word_to_token_maps[end - 1].1;

                                if char_start <= char_end && char_end <= text.len() {
                                    let entity_text = text[char_start..char_end]
                                        .trim().to_string();
                                    if !entity_text.is_empty() {
                                        all_matches.push(ExtractedEntity {
                                            score: prob,
                                            label: task_map.labels[m].clone(),
                                            text:  entity_text,
                                            start_tok: orig_start,
                                            end_tok:   orig_end,
                                            start_char: char_start,
                                            end_char: char_end,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // NMS greedy per score decrescente
            all_matches.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut selected: Vec<ExtractedEntity> = Vec::new();
            for m in all_matches {
                let overlap = selected.iter().any(|s| {
                    let spans_overlap = !(m.end_tok <= s.start_tok || m.start_tok >= s.end_tok);
                    spans_overlap && (flat_ner || s.label == m.label)
                });
                if !overlap {
                    selected.push(m);
                }
            }

            if task_map.task_type == "entities" {
                final_entities.extend(selected);
            } else if task_map.task_type == "relations" {
                let head = selected.iter().find(|x| x.label == "head").cloned();
                let tail = selected.iter().find(|x| x.label == "tail").cloned();
                if let (Some(h), Some(t)) = (head, tail) {
                    final_relations.push(ExtractedRelation {
                        head: h,
                        tail: t,
                        relation_type: task_map.labels[0].clone(),
                    });
                }
            }
        }

        Ok((final_entities, final_relations, final_classifications))
    }
}

// MAX_COUNT deve coincidere con il parametro max_count usato durante l'export.
// Baked nei modelli count_lstm_fixed (output shape fisso [MAX_COUNT, M, H]).
#[allow(dead_code)]
const MAX_COUNT: usize = 20;
