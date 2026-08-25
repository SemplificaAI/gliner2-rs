// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! GLiNER2 **span**-architecture inference on ONNX Runtime.
//!
//! Runs any GLiNER2 span export — `gliner2-multi-v1`, `gliner2-base-v1`,
//! `gliner2-privacy-filter-PII-multi`, `GLiNER2-Guardrails-PII-Multi`, or a
//! local fine-tune. Model-specific schemas and helpers live in the extension
//! crates on top: `gliner2-guardrails` and `gliner2-privacy`.
//!
//! For GLiNER2.5 checkpoints use `gliner25-core`: the boundary architecture
//! shares nothing with this one beyond the encoder and the prompt format, and
//! its spans are half-open rather than inclusive.
//!
//! ## Both export layouts are read
//!
//! Fragments are resolved flat, as `export_span_v3.py` produces them, and in
//! the legacy `fp32_v2/` + `fp16_v2/` subfolders published earlier on the Hub.
//! The classifier signature is detected from the graph, so exports carrying
//! either shape work unchanged — see [`ClassifierLayout`].

pub mod span;

pub use gliner_core::{
    GlinerError, OverlapPolicy, ProcessedRecord, SchemaTask, SchemaTransformer, TaskType, init,
};
pub use span::{
    Classification, ClassifierLayout, Entity, InferenceParams, SpanConfig, SpanEngine, SpanOutput,
};
