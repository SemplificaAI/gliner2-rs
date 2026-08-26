// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! GLiNER2 **span**-architecture inference on ONNX Runtime.
//!
//! Runs any GLiNER2 span export — `gliner2-multi-v1`, `gliner2-base-v1`,
//! `gliner2-privacy-filter-PII-multi`, `GLiNER2-Guardrails-PII-Multi`, or a
//! local fine-tune. Model-specific schemas and helpers live in the [`privacy`]
//! and [`guardrails`] modules, behind default-on features of the same names.
//!
//! For GLiNER2.5 checkpoints use `gliner25-rs`: the boundary architecture
//! shares nothing with this one beyond the encoder and the prompt format, and
//! its spans are half-open rather than inclusive.
//!
//! ## Both export layouts are read
//!
//! Fragments are resolved flat, as `export_span_v3.py` produces them, and in
//! the legacy `fp32_v2/` + `fp16_v2/` subfolders published earlier on the Hub.
//! The classifier signature is detected from the graph, so exports carrying
//! either shape work unchanged — see [`ClassifierLayout`].

//! ## Layout
//!
//! | module | what it holds |
//! |---|---|
//! | [`processor`] | prompt construction and tokenization |
//! | [`runtime`] | `ort` helpers, precision selection, export-layout resolution |
//! | [`overlap`] | span overlap policies |
//! | [`span`] | the inference engine |
//! | [`error`] | diagnosable engine errors |
//!
//! The first four are architecture-agnostic — identical to what the boundary
//! engine in [gliner25-rs](https://github.com/dariofinardi/gliner25-rs) needs,
//! because gliner2 itself shares them. They live here rather than in a crate of
//! their own: one shared crate would have to be published under a single name
//! and would tie the two repositories' release cadences together, for four
//! modules that change rarely. The cost is that a fix to them lands in two
//! places; the alternative was worse.

pub mod chain;
pub mod chunker;
pub mod error;
/// Fetching a published export from the Hub when it is not on disk.
#[cfg(feature = "hub")]
pub mod hub;
pub mod overlap;
pub mod processor;
pub mod runtime;
pub mod span;

/// Moderation schemas for `GLiNER2-Guardrails-PII-Multi`: the label sets with
/// the per-task thresholds and single/multi-label settings the model expects.
#[cfg(feature = "guardrails")]
pub mod guardrails;

/// PII schemas and redaction: the 42 labels in their semantic groups, plus
/// masking and the anonymisation gate.
#[cfg(feature = "privacy")]
pub mod privacy;

pub use error::GlinerError;
pub use overlap::{OverlapPolicy, Spanned, resolve_overlaps};
pub use processor::{ProcessedRecord, SchemaTask, SchemaTransformer, TaskMapping, TaskType};
pub use runtime::Precision;
pub use chain::ExecutionMode;
pub use chunker::Chunker;
pub use span::{
    Classification, ClassifierLayout, Entity, InferenceParams, SpanConfig, SpanEngine, SpanOutput,
};

/// Initialises the ONNX Runtime environment. Call once per process; later calls
/// are ignored.
///
/// In `ort` rc.13 `commit()` returns `bool` rather than `Result`: `false` means
/// an environment already existed, not that anything failed.
pub fn init(name: &str) -> bool {
    ort::init().with_name(name.to_string()).commit()
}

#[cfg(feature = "test-support")]
pub mod test_support {
    use std::path::PathBuf;

    /// Locates a gliner2 `tokenizer.json` for tests.
    ///
    /// Model directories are gitignored, so a test that cannot find one should
    /// skip rather than fail. All GLiNER2 checkpoints share the mDeBERTa-v3
    /// vocabulary plus the same added markers, so any of them pins the prompt
    /// layout equally well.
    pub fn find_tokenizer() -> Option<PathBuf> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
}

