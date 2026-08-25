# gliner2-rs

[![GitHub](https://img.shields.io/badge/GitHub-dariofinardi/gliner2--rs-blue?style=flat-square&logo=github)](https://github.com/dariofinardi/gliner2-rs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Beta-blue.svg)](https://github.com/dariofinardi/gliner2-rs)

**Native Rust inference for GLiNER2 on ONNX Runtime**

Extract entities, relations and classifications from text with no Python at
inference time. This repository is a Cargo workspace: a shared foundation, the
span engine, and thin model-specific extensions on top.

Written by **Dario Finardi**. Published by **Jugaad s.r.l.**, which uses it in
production inside **Edito** and **Omissis** —
[edito-pdf.com](https://edito-pdf.com).

For **GLiNER2.5** use [gliner25-rs](https://github.com/dariofinardi/gliner25-rs)
instead. Its `boundary` architecture shares only the foundation with this one,
and its spans are half-open rather than inclusive — mixing the conventions is a
silent off-by-one.

---

## The crates

| crate | what it is | docs |
|---|---|---|
| [`gliner-core`](crates/gliner-core) | prompt construction, ONNX Runtime helpers, overlap policies | [README](crates/gliner-core/README.md) |
| [`gliner2-core`](crates/gliner2-core) | the span inference engine | [README](crates/gliner2-core/README.md) |
| [`gliner2-guardrails`](crates/gliner2-guardrails) | LLM safety moderation schemas | [README](crates/gliner2-guardrails/README.md) |
| [`gliner2-privacy`](crates/gliner2-privacy) | PII schemas and redaction | [README](crates/gliner2-privacy/README.md) |
| [`gliner2-inference`](crates/gliner2-inference) | the original engine: V1 pipeline, HuggingFace downloader | [README](crates/gliner2-inference/README.md) |

`gliner2-inference` predates the split and stays as it is: edition 2021, its own
V1 fallback and `from_pretrained`. Use it if you want models pulled from the Hub
automatically. Use `gliner2-core` and the extensions for everything else.

Start with the extension that matches your model — they carry the label
vocabularies and helpers, and pull the engine in for you.

---

## Model compatibility

Weights are **not** ours. GLiNER2 is developed by [Fastino](https://fastino.ai)
(arXiv:2507.18546); the GLiNER line it descends from is the work of Urchade
Zaratiana et al. Converting a model changes neither its licence nor its
ownership — see [`NOTICE`](NOTICE).

| model | ONNX export | export layout | crate to use |
|---|---|---|---|
| [`fastino/gliner2-multi-v1`](https://huggingface.co/fastino/gliner2-multi-v1) | [`jugaadsrl/gliner2-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2-multi-v1-onnx) | legacy | `gliner2-core` |
| [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | [`jugaadsrl/gliner2-privacy-filter-PII-multi-onnx`](https://huggingface.co/jugaadsrl/gliner2-privacy-filter-PII-multi-onnx) | legacy | `gliner2-privacy` |
| [`fastino/GLiNER2-Guardrails-PII-Multi`](https://huggingface.co/fastino/GLiNER2-Guardrails-PII-Multi) | [`jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx`](https://huggingface.co/jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx) | flat | `gliner2-guardrails` |
| `fastino/gliner2-base-v1` and local fine-tunes | export it yourself | flat | `gliner2-core` |

Any GLiNER2 **span** checkpoint works: the engine reads `max_width` and
`MAX_COUNT` from the exported graphs rather than assuming them, so a checkpoint
exported with different parameters stays usable.

### The two export layouts

Both are read, and the difference is invisible to callers. Point the config at
the directory and it works.

```text
flat, from export_span_v3.py          legacy, published earlier on the Hub
  encoder_fp32.onnx                     fp32_v2/encoder_fp32.onnx
  encoder_fp16.onnx                     fp16_v2/encoder_fp16.onnx
  encoder_fp16_iobinding.onnx           fp16_v2/encoder_fp16_iobinding.onnx
  …                                     …
  tokenizer.json                        fp32_v2/tokenizer.json
                                        fp16_v2/tokenizer.json
```

Legacy exports also carry an older `classifier` signature,
`[batch, num_labels, max_width, H]` instead of `[num_labels, H]`. The engine
detects it from the graph and shapes the input accordingly; check which one it
found with `engine.classifier_layout()`.

### Precision

| suffix | I/O | use for |
|---|---|---|
| `_fp32` | FP32 | universal fallback, OpenVINO, CPU |
| `_fp16` | FP32 (`keep_io_types=True`) | CoreML, which demands FP32 I/O |
| `_fp16_iobinding` | FP16 | CUDA, ROCm, QNN — see the note below |

Selected automatically per platform — `_fp16_iobinding` on Linux and Windows,
`_fp16` on macOS — and overridable with
`GLINER2_PRECISION=fp32|fp16|fp16_iobinding`.

### A note on `_fp16_iobinding`

The suffix names what the variant was *exported for*, not what this engine does
with it. `keep_io_types=False` leaves the graph inputs and outputs in FP16 as
well as the weights, which is what ORT's zero-copy `IoBinding` needs to keep
tensors in device memory across the fragment chain.

**This engine does not implement `IoBinding`.** It loads those graphs and runs
them normally, so the variant still saves the FP32↔FP16 conversions at each
boundary, but intermediate tensors round-trip through host memory between
fragments. On CPU that costs nothing; on a discrete GPU it is the PCIe traffic
the variant exists to avoid.

If you need real zero-copy binding today, use
[`gliner2-inference`](crates/gliner2-inference), which implements it in its V2
pipeline. Implementing it in `gliner2-core` is tracked work, not a claim.


---

## Quick start

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
cargo run --release --example extract -p gliner2-privacy -- models/pii-onnx
```

```rust
use gliner2_core::{SchemaTask, SpanConfig, SpanEngine};

gliner2_core::init("my-app");
let mut engine = SpanEngine::new(SpanConfig::new("models/my-onnx-export"))?;

let tasks = vec![SchemaTask::Entities(vec![
    "person".into(), "organization".into(), "location".into(),
])];

for e in engine.extract("Mario Rossi works at Apple.", &tasks)?.entities {
    println!("{} -> {} ({:.1}%)  bytes [{}..{})",
             e.text, e.label, e.score * 100.0, e.char_start, e.char_end);
}
```

Byte offsets index the original text, so extracted spans keep their original
casing — which matters when redacting a document rather than labelling it.

Each crate README has worked examples for its own vocabulary.

---

## Requirements

- Rust **edition 2024**, MSRV **1.88** for the new crates;
  `gliner2-inference` is edition 2021.
- ONNX Runtime shared library, resolved at run time from `ORT_DYLIB_PATH`. The
  workspace pins `ort = 2.0.0-rc.13` with `default-features = false`, so nothing
  is downloaded at build time and no EP libraries are copied next to your
  binary. Verified against ONNX Runtime 1.25.1 at API level 17.
- Enable `ort`'s `download-binaries` feature instead if you would rather it
  fetch the runtime for you.

---

## Performance

Measured on an RTX 3090 and a Ryzen 9 5900XT, with the caveats that matter — the
host was under load 18 throughout, so the CPU figures are an upper bound and
some GPU rows are noise. See [`BENCHMARKS.md`](BENCHMARKS.md) for the full table
and what can and cannot be concluded from it.

The short version: use `fp32` on GPU. `_fp16_iobinding` is the slowest variant
everywhere until `IoBinding` is implemented, because FP16 graph I/O moves the
conversion into a scalar host-side loop at every fragment boundary.

## Verification

Per-fragment parity proves the ONNX graphs are faithful. It is not enough:
prompt construction, word routing, span decoding and suppression all live in
Rust, outside the graphs. The test that matters is the end-to-end comparison
against the PyTorch reference.

Run on both devices and every precision, because a CUDA kernel producing
different numbers from its CPU counterpart is exactly the kind of thing that
goes unnoticed otherwise:

| crate | device | `fp32` | `fp16` | `fp16_iobinding` |
|---|---|---|---|---|
| guardrails, 13 cases / 6 languages | CPU | 61/61 (0.0001) | 61/61 (0.0034) | 61/61 (0.0035) |
| guardrails | RTX 3090 | 61/61 (0.0001) | 61/61 (0.0036) | 61/61 (0.0035) |
| privacy, 13 cases / 7 languages | CPU | 58/58 (**0.0000**) | 58/58 (0.0023) | 58/58 (0.0021) |
| privacy | RTX 3090 | 58/58 (**0.0000**) | 58/58 (0.0022) | 58/58 (0.0021) |

Spans identical to the PyTorch reference in all twelve configurations; the figure
in brackets is the largest score delta.

Two things worth reading off that table. In `fp32` the privacy export matches
PyTorch **exactly** — not to within a tolerance, to the fourth decimal the
harness records — on both devices. And the CUDA path agrees with the CPU one to
within FP16 rounding, 0.0034 against 0.0036 on the same case. Whatever else
changes when you move to a GPU, the answers do not.

```sh
python onnx_conversion_scripts/compare_with_pytorch.py reference \
    --model_path fastino/<checkpoint> --cases tests/cases_pii.json --out /tmp/pytorch.json

ORT_DYLIB_PATH=… cargo run --release --example dump_json -p gliner2-privacy -- \
    models/pii-onnx tests/cases_pii.json > /tmp/rust.json

python onnx_conversion_scripts/compare_with_pytorch.py diff \
    --reference /tmp/pytorch.json --candidate /tmp/rust.json
```

Every decoding bug this project has had was caught this way and by nothing else:
the prompt layout drift, the cross-label suppression, the multi-label argmax
fallback. Tolerances are **relative**, scaled to each tensor's magnitude —
`span_rep` emits activations up to ~9e3 while `scorer` is a probability in
`[0,1]`, so an absolute threshold is meaningless across that range.

---

## 📊 Benchmark & Performance

> These figures were measured with **`gliner2-inference`**, whose V2 pipeline
> uses `IoBinding`. `gliner2-core` does not implement binding yet, so its GPU
> numbers will differ; the CPU ones are comparable.

Tested on complex text extraction tasks spanning up to 62 classes. Total Inference Time per Sentence is the primary metric used for fair cross-framework comparison, allowing precise cross-device and cross-language comparisons.

### 🖥️ Rust ONNX vs Python PyTorch (Desktop & Discrete GPUs)
Comparison of a 50-run continuous benchmark on x86_64 architecture with NVIDIA GPUs.

| Language | Engine (Hardware) | Total Time (50 runs) | Avg Time / Sentence | Avg Time / Entity (15-17) |
| :--- | :--- | :--- | :--- | :--- |
| **Python 3.10** | PyTorch (RTX 4090) | **~0.88 s** 🚀 | **4.40 ms** | 1.17 ms |
| **Python 3.10** | PyTorch (RTX 3090) | **~0.90 s** 🚀 | **4.52 ms** | 1.20 ms |
| **Rust (V1)** | ONNX Runtime CUDA (RTX 4090) | **~8.18 s** | **40.90 ms** | 10.90 ms |
| **Rust (V2)* ** | ONNX Runtime CUDA (RTX 4090) | **~5.91 s** ⚡ | **29.59 ms** | 6.96 ms |
| **Rust (V1)** | ONNX Runtime CUDA (RTX 3090) | **~8.59 s** | **42.97 ms** | 11.45 ms |
| **Rust (V2)* ** | ONNX Runtime CUDA (RTX 3090) | **~6.13 s** ⚡ | **30.68 ms** | 7.21 ms |
| **Python 3.10** | PyTorch (Ryzen 5900XT CPU) | **~7.26 s** | **36.33 ms** | 9.68 ms |
| **Rust (V1)** | ONNX Runtime (Ryzen 5900XT CPU) | **~13.75 s** | **68.76 ms** | 18.33 ms |

> *( * ) **V2 IOBinding Engine:** The new V2 implementation eliminates the PCIe bottleneck by fusing operations (`Gather`, `ArgMax`, `MatMul`) inside the ONNX graph and keeping tensors entirely in VRAM (Zero-Copy) using ORT's `IoBinding`. This drastically drops the execution time.


**Understanding the GPU Gap: Why is PyTorch still faster than V2?**
While V2 IOBinding successfully eliminates the PCIe data transfer bottleneck (tensors now stay in VRAM), Python/PyTorch remains ~6x faster on discrete GPUs. This is due to the **Fragmentation Penalty**:
1. **Kernel Launch & Orchestration Overhead:** Because GLiNER2's architecture relies on dynamic loops (e.g. iterating over an unknown number of schema tasks and varying predicted entity counts), it cannot be exported as a single monolithic ONNX graph. It must be split into 8 separate ONNX sessions. The Rust host CPU must orchestrate the execution of these 8 fragments sequentially. Even though the *data* stays in VRAM, the *control flow* (calling `.run()` multiple times per sentence) incurs severe CUDA kernel launch overhead and forces continuous CPU-GPU synchronization.
2. **Lack of Global Graph Fusion:** PyTorch executes the entire model inside a single unified context, allowing its backend to fuse kernels across the entire architecture. ONNX Runtime can only optimize and fuse operations within the hard boundaries of each individual fragment.
3. **Dynamic Shapes:** ONNX Runtime achieves peak performance (e.g., via TensorRT) with static shapes. GLiNER2 is highly dynamic (varying sequence lengths, changing number of entities), which prevents ORT from locking in optimal execution paths—a scenario where PyTorch's native dynamic execution naturally excels.

*Conclusion:* Rust ONNX V2 represents the upper limit of optimization for a fragmented pipeline. While PyTorch wins on raw continuous throughput on discrete GPUs, Rust ONNX completely dominates PyTorch in **Cold Start** scenarios (loading in ~2s vs ~10s) and is the absolute winner for **Unified Memory Architectures** (Apple Silicon / ARM Snapdragon NPU) and edge deployments.

### 🐍 Rust vs Python on ARM (Snapdragon X Elite)
Comparison between native Rust ONNX execution and standard Python PyTorch inference on the same ARM hardware.
Note: Benchmarks executed plugged in (Max Performance profile). Testing 51 target entities extraction.

| Environment | Hardware (Backend) | Precision (Model) | Startup Time | Total Inference Time (Sentence) | Time / Entity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rust (V1)** | CPU ARM64 (Oryon) | `fp32` | **~3.64 s** | **0.43 s** 🚀 | ~8.53 ms |
| **Rust (V2)* ** | NPU (QNN) | `fp16_v2` | **~2.28 s** | **0.65 s** ✨ | ~12.88 ms |
| **Rust (V2)* ** | CPU ARM64 (Oryon) | `fp16_v2` | **~1.96 s** ⚡ | **0.66 s** | ~13.10 ms |
| **Rust (V1)** | CPU ARM64 (Oryon) | `fp16` | **~1.82 s** | **0.68 s** | ~13.43 ms |
| **Rust (V1)** | NPU (QNN) | `fp16` | **~2.12 s** | **0.71 s** | ~14.11 ms |
| **Python 3.12** | CPU ARM64 (PyTorch) | `jugaadsrl/gliner2-multi-v1` | **~12.74 s** 🐢 | **0.31 s** | ~15.03 ms |
| **Python 3.12** | CPU ARM64 (PyTorch) | `fastino/gliner2-multi-v1` | **~8.76 s** 🐢 | **0.36 s** | ~24.51 ms |

**Takeaways:**
- **The FP32 Surprise:** Instructing the Rust ONNX runtime to load full FP32 precision models allows the Snapdragon ARM64 Oryon CPU to skip expensive hardware/software downcasting. It slashes inference time to **0.43s per sentence**, completely crushing FP16 times and heavily outperforming the limited NPU drivers.
- **V2 IOBinding is Consistent:** At matched FP16 precision, the fused V2 consistently beats the standard V1 architecture, both on CPU and NPU.
- **Rust = Cold Start Speed & Reproducibility:** Rust boots in ~1.8-3.6s (depending on precision) and flawlessly extracts the exact overlapping entities without implicit filtering. Python struggles for ~9-12s just to load tensors and forces unrequested NMS flat_ner filtering which artificially alters the output count.

---


---

## Exporting a model

```sh
python onnx_conversion_scripts/export_span_v3.py \
    --model_path fastino/gliner2-privacy-filter-PII-multi \
    --out_dir models/pii-onnx
```

Needs `torch`, `transformers`, `gliner2>=2.0.0`, `onnx`, `onnxruntime`. It
refuses `boundary` checkpoints rather than producing a silently wrong export.

The span architecture cannot be traced into a single ONNX graph — it loops over
a variable number of schema tasks and a predicted, variable number of entity
occurrences — so it is exported as eight fragments. See
[`crates/gliner2-core/README.md`](crates/gliner2-core/README.md) for the chain.

---

## License and attribution

Licensed under the [Apache License, Version 2.0](LICENSE).

The models are the work of the [Fastino](https://fastino.ai) team and are not
distributed from this repository. See [`NOTICE`](NOTICE) for the full
attribution, and [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for the history.

Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
