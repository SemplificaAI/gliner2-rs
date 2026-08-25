use std::fmt;

#[derive(Debug)]
pub enum GlinerError {
    /// E_GLI_001: Out of Memory sul Device (GPU/NPU) durante la pre-allocazione IOBinding.
    OomDeviceBinding(String),
    /// E_GLI_002: Out of Memory sul Device durante l'esecuzione standard (Modalità 1).
    OomDeviceStandard(String),
    /// E_GLI_003: Memoria RAM di sistema (Host) esaurita. Impossibile caricare i modelli iniziali.
    OomHostRam(String),
    /// E_GLI_004: L'Execution Provider corrente non supporta IOBinding nativo. Fallback automatico.
    BindingNotSupported(String),
    /// E_GLI_005: Dimensioni dei tensori errate tra un output e l'input successivo durante il binding.
    TensorShapeMismatch(String),
    /// Altri errori (tokenizer, IO, ecc.)
    Other(anyhow::Error),
}

impl fmt::Display for GlinerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OomDeviceBinding(m) => write!(f, "[E_GLI_001] OOM_DEVICE_BINDING: {}", m),
            Self::OomDeviceStandard(m) => write!(f, "[E_GLI_002] OOM_DEVICE_STANDARD: {}", m),
            Self::OomHostRam(m) => write!(f, "[E_GLI_003] OOM_HOST_RAM: {}", m),
            Self::BindingNotSupported(m) => write!(f, "[E_GLI_004] BINDING_NOT_SUPPORTED: {}", m),
            Self::TensorShapeMismatch(m) => write!(f, "[E_GLI_005] TENSOR_SHAPE_MISMATCH: {}", m),
            Self::Other(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for GlinerError {}

impl From<anyhow::Error> for GlinerError {
    fn from(err: anyhow::Error) -> Self {
        GlinerError::Other(err)
    }
}
