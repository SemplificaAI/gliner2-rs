// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! One pipeline, two ways of moving tensors through it.
//!
//! The span engine is a chain of eight ONNX fragments. Between any two of them
//! the intermediate tensor can either be copied back to host memory and rebuilt
//! for the next fragment, or left where the provider produced it and bound
//! straight into the next fragment's input.
//!
//! Both are the *same* pipeline: same fragments, same order, same maths. Only
//! the transport differs. So rather than two engines, there is one chain and a
//! [`Chain`] that knows how to run a single step either way — which is what
//! keeps the two paths from drifting apart, the way they did when they lived in
//! separate crates.
//!
//! ```text
//!   Standard      encoder ──▶ host ──▶ token_gather ──▶ host ──▶ span_rep ─▶ …
//!   IoBinding     encoder ─────────────▶ token_gather ─────────▶ span_rep ─▶ …
//!                         (device memory throughout)
//! ```
//!
//! On CPU the round trip costs nothing — "host memory" is where the tensor
//! already is. On a discrete GPU it is PCIe traffic per fragment per call,
//! which is what `IoBinding` exists to avoid.

use crate::error::GlinerError;
use crate::runtime::{IoDType, float_tensor, take_float};
use anyhow::{Result, anyhow};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::DynValue;

/// How intermediate tensors travel between fragments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    /// Every fragment's output is copied to host memory before the next runs.
    /// Works on every execution provider.
    Standard,
    /// Intermediates stay in device memory across the chain, via ORT's
    /// `IoBinding`. Meaningful only on a provider with its own memory.
    IoBinding,
    /// `IoBinding` on a device provider, `Standard` on CPU. On a device OOM the
    /// engine drops to `Standard` for the rest of its life rather than failing.
    #[default]
    Auto,
}

impl ExecutionMode {
    /// Resolves `Auto` against the provider actually configured.
    ///
    /// Separate from [`Chain::new`] because the answer is needed *before* a
    /// chain can be built: it decides which precision to fetch, and the chain
    /// needs the precision to know its element type.
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto if crate::runtime::provider_has_device_memory() => Self::IoBinding,
            Self::Auto => Self::Standard,
            other => other,
        }
    }

    /// The export variant this transport is meant to run.
    ///
    /// `_fp16_iobinding` leaves the graph boundaries in FP16, which is what a
    /// bound chain needs to pass one fragment's output to the next untouched.
    /// The standard path gains nothing from that and would pay a cast at every
    /// boundary, so it asks for FP32.
    pub fn preferred_precision(self) -> crate::runtime::Precision {
        match self.resolve() {
            Self::IoBinding => crate::runtime::Precision::Fp16IoBinding,
            _ => crate::runtime::Precision::Fp32,
        }
    }

    /// Reads `GLINER2_EXECUTION=standard|binding|auto`.
    ///
    /// `GLINER2_NO_IOBINDING=1` is also honoured — it is what the older engine
    /// used, and forcing the standard path is exactly what it meant.
    pub fn from_env() -> Self {
        if std::env::var("GLINER2_NO_IOBINDING").is_ok_and(|v| v != "0") {
            return Self::Standard;
        }
        match std::env::var("GLINER2_EXECUTION").as_deref() {
            Ok("standard") => Self::Standard,
            Ok("binding") | Ok("iobinding") => Self::IoBinding,
            _ => Self::Auto,
        }
    }
}

/// Where a fragment's output should be left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sink {
    /// Keep it on the device when binding; the next fragment consumes it there.
    Device,
    /// Bring it to host memory: the engine itself needs to read the numbers.
    Host,
    /// Host memory, integer output. `count_pred_argmax` returns an `i64`.
    HostI64,
}

/// A tensor travelling between two fragments.
pub enum Carrier {
    /// Values in host memory, with the shape the producing fragment gave them.
    Host { shape: Vec<i64>, data: Vec<f32> },
    /// Integer values in host memory.
    HostI64 { shape: Vec<i64>, data: Vec<i64> },
    /// A value left wherever the provider put it.
    Device(DynValue),
}

impl Carrier {
    /// The numbers, copying from the device if that is where they are.
    pub fn host(&self, dtype: IoDType) -> Result<Vec<f32>> {
        match self {
            Self::Host { data, .. } => Ok(data.clone()),
            Self::HostI64 { data, .. } => Ok(data.iter().map(|v| *v as f32).collect()),
            Self::Device(v) => Ok(take_float(v, dtype)?.1),
        }
    }

    /// The integers, for a fragment that produced them.
    pub fn host_i64(&self) -> Result<Vec<i64>> {
        match self {
            Self::HostI64 { data, .. } => Ok(data.clone()),
            Self::Device(v) => Ok(crate::runtime::take_i64(v)?.1),
            Self::Host { .. } => Err(anyhow!("carrier holds floats, not integers")),
        }
    }
}

/// An input to a fragment.
pub enum Feed<'a> {
    /// A tensor built for this call — token ids, span indices.
    Owned(DynValue),
    /// An intermediate from an earlier fragment.
    ///
    /// The shape is the one the *standard* path gives it when rebuilding the
    /// tensor on the host. It is carried explicitly rather than taken from the
    /// producing fragment because the two need not agree: several graphs here
    /// emit a tensor that the next one reads at a different rank, and the
    /// standard path has always reshaped on the way through. Binding ignores it
    /// and binds the device value as it stands, which is what keeps a bound run
    /// free of host copies.
    Carried(&'a Carrier, Vec<i64>),
}

/// Runs one fragment, either way.
pub struct Chain {
    mode: ExecutionMode,
    dtype: IoDType,
    device: Option<MemoryInfo<'static>>,
    host: Option<MemoryInfo<'static>>,
}

impl Chain {
    /// Resolves `Auto` against the provider actually in use.
    ///
    /// `IoBinding` is requested explicitly even when no device provider is
    /// configured: ORT will simply bind to CPU memory, which is harmless and
    /// keeps an explicit request honest rather than silently ignored.
    pub fn new(mode: ExecutionMode, dtype: IoDType) -> Result<Self> {
        let device_id = crate::runtime::device_id();
        let wants_binding = mode.resolve() == ExecutionMode::IoBinding;

        if !wants_binding {
            return Ok(Self { mode: ExecutionMode::Standard, dtype, device: None, host: None });
        }

        let device = MemoryInfo::new(
            crate::runtime::allocation_device(),
            device_id,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| GlinerError::BindingNotSupported(format!("device MemoryInfo: {e}")))?;
        let host = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .map_err(|e| GlinerError::BindingNotSupported(format!("host MemoryInfo: {e}")))?;

        Ok(Self { mode: ExecutionMode::IoBinding, dtype, device: Some(device), host: Some(host) })
    }

    /// The mode actually in force, after `Auto` was resolved.
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Permanently drops to the standard path.
    ///
    /// Called after a device OOM: one slow run beats a failed one, and retrying
    /// the binding on every call would just pay the same allocation failure
    /// again. Binding holds every intermediate on the device at once, so it is
    /// the first thing to give way on a long input — the standard path releases
    /// each tensor as soon as the next fragment has consumed it.
    pub fn fall_back(&mut self) {
        self.mode = ExecutionMode::Standard;
        self.device = None;
        self.host = None;
    }

    /// Runs `session`, returning one carrier per output, in the order the graph
    /// declares them.
    ///
    /// Inputs are positional, matched against the graph's declared order, and
    /// the names needed for binding are read from the session rather than
    /// written here. That is not incidental: the same role carries different
    /// names across exports — the legacy `classifier` declares its input as
    /// `span_embeddings` even though what it receives is field embeddings — so
    /// a hard-coded name works on one checkpoint and fails on the next.
    pub fn run(
        &self,
        session: &mut Session,
        inputs: &[Feed<'_>],
        outputs: &[Sink],
    ) -> Result<Vec<Carrier>> {
        let in_names: Vec<String> =
            session.inputs().iter().map(|i| i.name().to_string()).collect();
        let out_names: Vec<String> =
            session.outputs().iter().map(|o| o.name().to_string()).collect();
        if inputs.len() != in_names.len() {
            return Err(anyhow!(
                "fragment declares {} inputs, {} supplied",
                in_names.len(),
                inputs.len()
            ));
        }
        if outputs.len() != out_names.len() {
            return Err(anyhow!(
                "fragment declares {} outputs, {} requested",
                out_names.len(),
                outputs.len()
            ));
        }
        let inputs: Vec<(&str, &Feed<'_>)> = in_names
            .iter()
            .map(|s| s.as_str())
            .zip(inputs.iter())
            .collect();
        let outputs: Vec<(&str, Sink)> = out_names
            .iter()
            .map(|s| s.as_str())
            .zip(outputs.iter().copied())
            .collect();
        self.dispatch(session, &inputs, &outputs)
    }

    fn dispatch(
        &self,
        session: &mut Session,
        inputs: &[(&str, &Feed<'_>)],
        outputs: &[(&str, Sink)],
    ) -> Result<Vec<Carrier>> {
        match self.mode {
            ExecutionMode::Standard => self.run_standard(session, inputs, outputs),
            _ => self.run_bound(session, inputs, outputs),
        }
    }

    fn run_standard(
        &self,
        session: &mut Session,
        inputs: &[(&str, &Feed<'_>)],
        outputs: &[(&str, Sink)],
    ) -> Result<Vec<Carrier>> {
        // A carrier reaching here is already on the host unless a previous run
        // left it on a device — which only happens if the mode changed
        // mid-flight, so rebuild from its numbers either way.
        let mut owned: Vec<(&str, DynValue)> = Vec::with_capacity(inputs.len());
        for (name, feed) in inputs {
            let value = match feed {
                Feed::Owned(v) => clone_value(v, self.dtype)?,
                Feed::Carried(c, shape) => {
                    float_tensor(self.dtype, shape.clone(), c.host(self.dtype)?)?
                }
            };
            owned.push((name, value));
        }

        let out = session
            .run(owned)
            .map_err(|e| classify(e, "standard run"))?;

        outputs
            .iter()
            .map(|(name, sink)| {
                let v = out
                    .get(*name)
                    .ok_or_else(|| anyhow!("fragment produced no output named '{name}'"))?;
                Ok(materialise(v, *sink, self.dtype)?)
            })
            .collect()
    }

    fn run_bound(
        &self,
        session: &mut Session,
        inputs: &[(&str, &Feed<'_>)],
        outputs: &[(&str, Sink)],
    ) -> Result<Vec<Carrier>> {
        let device = self.device.as_ref().expect("binding mode without device memory");
        let host = self.host.as_ref().expect("binding mode without host memory");

        let mut binding = session
            .create_binding()
            .map_err(|e| classify(e, "create_binding"))?;

        // Tensors rebuilt from host carriers must outlive the binding.
        let mut lifted: Vec<DynValue> = Vec::new();

        for (name, feed) in inputs {
            match feed {
                Feed::Owned(v) => binding.bind_input(*name, v),
                Feed::Carried(Carrier::Device(v), _) => binding.bind_input(*name, v),
                Feed::Carried(c, shape) => {
                    // Produced on the host — by an earlier fragment whose output
                    // the engine had to read. Lift it back so the chain carries
                    // on bound instead of dropping to the standard path.
                    lifted.push(float_tensor(self.dtype, shape.clone(), c.host(self.dtype)?)?);
                    binding.bind_input(*name, lifted.last().expect("just pushed"))
                }
            }
            .map_err(|e| classify(e, &format!("bind input '{name}'")))?;
        }

        for (name, sink) in outputs {
            let mem = match sink {
                Sink::Device => device,
                Sink::Host | Sink::HostI64 => host,
            };
            binding
                .bind_output_to_device(*name, mem)
                .map_err(|e| classify(e, &format!("bind output '{name}'")))?;
        }

        let mut out = session
            .run_binding(&binding)
            .map_err(|e| classify(e, "run_binding"))?;

        outputs
            .iter()
            .map(|(name, sink)| {
                let v = out
                    .remove(*name)
                    .ok_or_else(|| anyhow!("fragment produced no output named '{name}'"))?;
                Ok(match sink {
                    Sink::Device => Carrier::Device(v),
                    _ => materialise(&v, *sink, self.dtype)?,
                })
            })
            .collect()
    }
}

/// Copies a fragment output into host memory in the right element type.
fn materialise(v: &DynValue, sink: Sink, dtype: IoDType) -> Result<Carrier> {
    Ok(match sink {
        Sink::HostI64 => {
            let (shape, data) = crate::runtime::take_i64(v)?;
            Carrier::HostI64 { shape, data }
        }
        // A device sink reaching here means the standard path ran, where every
        // output is on the host to begin with.
        Sink::Host | Sink::Device => {
            let (shape, data) = take_float(v, dtype)?;
            Carrier::Host { shape, data }
        }
    })
}

/// Rebuilds an owned tensor. `DynValue` is not `Clone`, and the standard path
/// needs to hand `run` an owned list.
fn clone_value(v: &DynValue, dtype: IoDType) -> Result<DynValue> {
    if let Ok((shape, data)) = crate::runtime::take_i64(v) {
        return crate::runtime::i64_tensor(shape, data);
    }
    let (shape, data) = take_float(v, dtype)?;
    float_tensor(dtype, shape, data)
}

/// Turns an ORT failure into something the engine can act on.
///
/// A device allocation failure is recoverable — drop to the standard path — so
/// it must be distinguishable from a genuine error. ORT reports it in the
/// message rather than in a code, so the message is what we read.
fn classify(err: impl std::fmt::Display, what: &str) -> anyhow::Error {
    let msg = err.to_string();
    let low = msg.to_lowercase();
    // ORT reports an exhausted arena as "Failed to allocate memory for requested
    // buffer of size N" and not as anything containing "out of memory", so
    // matching the obvious phrase alone silently misses the case this exists
    // for. All three spellings observed in the wild are listed.
    if low.contains("out of memory")
        || low.contains("failed to allocate memory")
        || low.contains("cudaerrormemoryallocation")
    {
        return GlinerError::OomDeviceBinding(format!("{what}: {msg}")).into();
    }
    if low.contains("not supported") || low.contains("invalid allocator") {
        return GlinerError::BindingNotSupported(format!("{what}: {msg}")).into();
    }
    anyhow::Error::msg(msg).context(what.to_string())
}
