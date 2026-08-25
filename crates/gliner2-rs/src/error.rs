// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Diagnosable engine errors.

use std::fmt;

#[derive(Debug)]
pub enum GlinerError {
    /// E_GLI_001: device OOM while pre-allocating IOBinding buffers.
    OomDeviceBinding(String),
    /// E_GLI_002: device OOM during standard execution.
    OomDeviceStandard(String),
    /// E_GLI_003: host RAM exhausted; the models could not be loaded.
    OomHostRam(String),
    /// E_GLI_004: the execution provider does not support IOBinding; falling back.
    BindingNotSupported(String),
    /// E_GLI_005: shape mismatch between one fragment's output and the next one's input.
    TensorShapeMismatch(String),
    /// E_GLI_006: the model directory does not hold the expected fragments.
    IncompleteModelDir(String),
    /// E_GLI_007: no exported length bucket is large enough for the text.
    NoLengthBucket { words: usize, max_bucket: usize },
    /// E_GLI_008: the export could not be fetched from the Hub.
    #[cfg(feature = "hub")]
    Hub(String),
    /// Anything else (tokenizer, IO, ONNX Runtime...).
    Other(anyhow::Error),
}

impl fmt::Display for GlinerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "hub")]
            Self::Hub(m) => write!(f, "[E_GLI_008] HUB_FETCH: {m}"),
            Self::OomDeviceBinding(m) => write!(f, "[E_GLI_001] OOM_DEVICE_BINDING: {m}"),
            Self::OomDeviceStandard(m) => write!(f, "[E_GLI_002] OOM_DEVICE_STANDARD: {m}"),
            Self::OomHostRam(m) => write!(f, "[E_GLI_003] OOM_HOST_RAM: {m}"),
            Self::BindingNotSupported(m) => write!(f, "[E_GLI_004] BINDING_NOT_SUPPORTED: {m}"),
            Self::TensorShapeMismatch(m) => write!(f, "[E_GLI_005] TENSOR_SHAPE_MISMATCH: {m}"),
            Self::IncompleteModelDir(m) => write!(f, "[E_GLI_006] INCOMPLETE_MODEL_DIR: {m}"),
            Self::NoLengthBucket { words, max_bucket } => write!(
                f,
                "[E_GLI_007] NO_LENGTH_BUCKET: {words} words exceed the largest exported \
                 bucket ({max_bucket}); re-export with --buckets, or split the text"
            ),
            Self::Other(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for GlinerError {}

impl From<anyhow::Error> for GlinerError {
    fn from(err: anyhow::Error) -> Self {
        GlinerError::Other(err)
    }
}
