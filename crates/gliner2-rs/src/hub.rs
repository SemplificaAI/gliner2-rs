//! Fetching a published export from the Hub when it is not on disk.
//!
//! The engine loads ONNX fragments from a directory. If that directory is not
//! there — first run, fresh container, a machine where nobody has fetched the
//! weights yet — [`SpanConfig::or_download`](crate::SpanConfig::or_download)
//! names the repository to pull it from, and the fetch happens inside
//! [`SpanEngine::new`](crate::SpanEngine::new) rather than as a separate step
//! the caller has to remember.
//!
//! ```no_run
//! use gliner2_rs::{SpanConfig, SpanEngine, hub};
//!
//! // Uses ./models/pii if it holds an export, downloads it if it does not.
//! let cfg = SpanConfig::new("models/pii").or_download(hub::PRIVACY_PII_MULTI);
//! let mut engine = SpanEngine::new(cfg)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Files land in the Hub cache (`HF_HOME`, else `~/.cache/huggingface`), shared
//! with every other tool on the machine, so a model already fetched by the
//! Python library is not fetched again. A model already present is verified,
//! not re-downloaded.
//!
//! ## Transport
//!
//! `hf-hub` is pulled with `default-features = false, features = ["ureq"]`,
//! which resolves TLS through `rustls` rather than `native-tls`. There is no
//! `openssl` in the tree, and so no OpenSSL C library to keep patched: the
//! whole transport is Rust.
//!
//! Turn the feature off (`default-features = false`) and the crate goes back to
//! having no network stack whatsoever — no `ureq`, no `rustls`, no TLS at all.

use crate::error::GlinerError;
use crate::runtime::Precision;
use anyhow::{Context, Result};
use std::path::PathBuf;

/// How an export arranges its fragments in the repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Fragments at the repository root, precision in the file name:
    /// `encoder_fp32.onnx`.
    Flat,
    /// Fragments grouped by precision: `fp32_v2/encoder_fp32.onnx`. The exports
    /// published before the flat layout are laid out this way.
    Legacy,
}

/// A published ONNX export.
///
/// The constants below name the exports Jugaad publishes. Any other repository
/// works too — build a `Model` with [`Model::new`] and the layout its files use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub repo_id: &'static str,
    pub layout: Layout,
}

impl Model {
    pub const fn new(repo_id: &'static str, layout: Layout) -> Self {
        Self { repo_id, layout }
    }
}

/// `fastino/gliner2-multi-v1`, the general-purpose span checkpoint.
pub const GLINER2_MULTI_V1: Model =
    Model::new("jugaadsrl/gliner2-multi-v1-onnx", Layout::Legacy);

/// `fastino/gliner2-privacy-filter-PII-multi`, tuned for PII.
pub const PRIVACY_PII_MULTI: Model =
    Model::new("jugaadsrl/gliner2-privacy-filter-PII-multi-onnx", Layout::Legacy);

/// `fastino/GLiNER2-Guardrails-PII-Multi`, PII plus LLM-guardrail classification.
pub const GUARDRAILS_PII_MULTI: Model =
    Model::new("jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx", Layout::Flat);

/// The fragments a span engine needs, plus the tokenizer.
const FRAGMENTS: [&str; 8] = [
    "encoder",
    "token_gather",
    "span_rep",
    "schema_gather",
    "count_pred_argmax",
    "count_lstm_fixed",
    "scorer",
    "classifier",
];

/// Downloads `model` and returns the directory to load from, with the variant
/// actually obtained.
///
/// Only one variant is fetched — the one the caller asked for, or the first of
/// its fallbacks the repository publishes. An export carries up to three copies
/// of every fragment and the encoder alone is half a gigabyte, so fetching all
/// of them to use one is most of a download wasted.
///
/// The returned path is the snapshot root, so a legacy export resolves through
/// its `fp32_v2/` subfolder exactly as a local checkout would.
pub fn download(model: Model, precision: Precision) -> Result<(PathBuf, Precision)> {
    let mut last_err = None;
    for candidate in precision.fallback_chain() {
        match download_exact(model, *candidate) {
            Ok(dir) => {
                if *candidate != precision {
                    eprintln!(
                        "[gliner2] {} does not publish {}{}; using {} instead",
                        model.repo_id,
                        "encoder",
                        precision.suffix(),
                        candidate.suffix(),
                    );
                }
                return Ok((dir, *candidate));
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no precision variant could be fetched")))
}

/// Fetches exactly one variant, failing if the repository does not carry it.
fn download_exact(model: Model, precision: Precision) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_user_agent(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
        .build()
        .map_err(|e| GlinerError::Hub(format!("could not initialise the Hub client: {e}")))?;
    let repo = api.model(model.repo_id.to_string());

    let subfolder = match model.layout {
        Layout::Flat => None,
        Layout::Legacy => Some(precision.legacy_subdir()),
    };
    let at = |name: &str| match subfolder {
        Some(sub) => format!("{sub}/{name}"),
        None => name.to_string(),
    };

    let mut last = None;
    for stem in FRAGMENTS {
        let file = at(&format!("{stem}{}.onnx", precision.suffix()));
        let path = repo.get(&file).map_err(|e| {
            GlinerError::Hub(format!(
                "{}: could not fetch {file} ({e}). The repository may not publish \
                 this precision, or the layout may differ from the one declared.",
                model.repo_id
            ))
        })?;
        // A fragment past the 2 GB protobuf limit keeps its weights in a sidecar
        // `.onnx.data`, which ONNX Runtime opens by relative name at session
        // build time. Most fragments have none, so a miss here is not an error -
        // but a fragment that has one and does not get it fails at load, not at
        // download, with a filesystem error naming a file nobody asked for.
        let _ = repo.get(&format!("{file}.data"));
        last = Some(path);
    }

    // The tokenizer sits beside the fragments in some exports and at the root in
    // others, so accept either.
    let tok = at("tokenizer.json");
    if repo.get(&tok).is_err() && subfolder.is_some() {
        repo.get("tokenizer.json").map_err(|e| {
            GlinerError::Hub(format!("{}: could not fetch tokenizer.json ({e})", model.repo_id))
        })?;
    }

    let leaf = last
        .context("no fragment was downloaded")?
        .parent()
        .context("downloaded fragment has no parent directory")?
        .to_path_buf();

    // Hand back the snapshot root: `resolve_fragment` looks in the directory and
    // then in the precision subfolder, so the root works for both layouts.
    Ok(match subfolder {
        Some(_) => leaf.parent().context("legacy snapshot has no root")?.to_path_buf(),
        None => leaf,
    })
}
