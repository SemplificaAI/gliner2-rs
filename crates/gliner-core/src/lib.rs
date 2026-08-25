// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Shared foundation for the GLiNER Rust engines.
//!
//! Everything here is architecture-agnostic: it is identical between the
//! **span** models (GLiNER2) and the **boundary** models (GLiNER2.5), because
//! gliner2 itself shares it. The engines live above, in `gliner2-core` and
//! `gliner25-core`.
//!
//! | module | what it holds |
//! |---|---|
//! | [`processor`] | prompt construction and tokenization |
//! | [`runtime`] | `ort` helpers, precision selection, export-layout resolution |
//! | [`overlap`] | span overlap policies |
//! | [`error`] | diagnosable engine errors |
//!
//! ## Span conventions differ above this layer
//!
//! [`overlap`] works on **half-open** `[start, end)` intervals, which is what
//! the boundary architecture produces natively. The span architecture emits
//! **inclusive** `[w, w + k]` ranges and converts on the way in. Getting this
//! wrong is a silent off-by-one, so the conversion is done once, in the span
//! engine's `Spanned` implementation, rather than at each call site.

pub mod error;
pub mod overlap;
pub mod processor;
pub mod runtime;

pub use error::GlinerError;
pub use overlap::{OverlapPolicy, Spanned, resolve_overlaps};
pub use processor::{ProcessedRecord, SchemaTask, SchemaTransformer, TaskMapping, TaskType};
pub use runtime::Precision;

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
