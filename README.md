# gliner2-rs

[![GitHub](https://img.shields.io/badge/GitHub-SemplificaAI/gliner2--rs-blue?style=flat-square&logo=github)](https://github.com/SemplificaAI/gliner2-rs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/Version-0.5.0-brightgreen.svg)](https://github.com/SemplificaAI/gliner2-rs)
[![Status](https://img.shields.io/badge/Status-Beta-blue.svg)](https://github.com/SemplificaAI/gliner2-rs)

**Native Rust Inference Engine for GLiNER2**

`gliner2-rs` is a high-performance, Zero-Python inference engine designed to execute **GLiNER2** models using **ONNX Runtime**. It allows for extracting Named Entities (NER), Relations, and Global Classifications natively in Rust with maximum speed, supporting both CPU and NVIDIA GPU (CUDA) via hardware-accelerated Tensor operations.

This crate completely replicates the advanced sub-word tokenization and prompt-generation logic of GLiNER2's `processor.py` internally, using the official `tokenizers` crate for zero-overhead BPE tokenization.

*Copyright 2026 Dario Finardi, [https://semplifica.ai](Semplifica s.r.l.)*  
*Licensed under Apache License 2.0*

## 🚀 Features

### ⚡ What's New in 0.5.0 (Dynamic Inference Parameters)
- **Zero-Copy PCIe bypass**: Replaces CPU manipulations with `Gather`, `ArgMax`, and `MatMul` operations fused directly into the ONNX graphs. Data now stays inside GPU/NPU VRAM, speeding up performance by ~30% (currently tested on NVIDIA RTX GPUs and AMD Ryzen CPUs).
- **Automatic Engine Facade**: `Gliner2Engine` acts as an intelligent wrapper. It detects whether the model folder contains V1 or V2 files, automatically switching to the optimal execution pipeline. **No code changes are required** to use V2!
- **Smart HF Downloader**: `Gliner2Engine::from_pretrained` now detects your OS. On CUDA/ROCm platforms it downloads the `_iobinding` variants, while on macOS (Apple Silicon/CoreML) it safely downloads the standard `_fp16` fallback. This **halves bandwidth and disk usage**!
- **New V2 ONNX Exporter**: We provide `export_gliner2_onnx_fragments_v2.py` which automatically generates `fp32`, `fp16`, and `fp16_iobinding` (Full IO Types) variants of the fusions.

### Key Features

- **End-to-End Execution**: Full recreation of the GLiNER2 inference loop natively in Rust.
- **Multi-Task Extraction**: Supports Entity Extraction, Relation Extraction, and Text Classifications in a single forward pass.
- **Hardware Accelerated**: Dynamically uses QNN (Qualcomm NPU), CoreML (Apple Silicon), OpenVINO (Intel/AMD), CUDA Execution Provider if an NVIDIA GPU is available, falling back to optimized XNNPACK/CPU execution.
- **FP16 & FP32 Support**: Fully compatible with Half-Precision (Float16) ONNX exports to cut memory footprints in half.
- **Zero-Copy Tensor Flow**: Direct injection of raw hidden states across multiple neural network slices without CPU-GPU memory swaps.
- **Built-in NMS**: Automatic Non-Maximum Suppression (NMS) to elegantly remove overlapping fictions entities based on their probabilities.

---

## 📊 Benchmark & Performance

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
| **Python 3.12** | CPU ARM64 (PyTorch) | `SemplificaAI/gliner2-multi-v1` | **~12.74 s** 🐢 | **0.31 s** | ~15.03 ms |
| **Python 3.12** | CPU ARM64 (PyTorch) | `fastino/gliner2-multi-v1` | **~8.76 s** 🐢 | **0.36 s** | ~24.51 ms |

**Takeaways:**
- **The FP32 Surprise:** Instructing the Rust ONNX runtime to load full FP32 precision models allows the Snapdragon ARM64 Oryon CPU to skip expensive hardware/software downcasting. It slashes inference time to **0.43s per sentence**, completely crushing FP16 times and heavily outperforming the limited NPU drivers.
- **V2 IOBinding is Consistent:** At matched FP16 precision, the fused V2 consistently beats the standard V1 architecture, both on CPU and NPU.
- **Rust = Cold Start Speed & Reproducibility:** Rust boots in ~1.8-3.6s (depending on precision) and flawlessly extracts the exact overlapping entities without implicit filtering. Python struggles for ~9-12s just to load tensors and forces unrequested NMS flat_ner filtering which artificially alters the output count.

---

## Installation & Setup

### Installation Steps

1. Clone this repository or add `gliner2-rs` to your `Cargo.toml`.
2. Ensure you have the `onnxruntime` C/C++ libraries available on your system path.
3. Export the GLiNER2 models to ONNX fragmented versions. 

### Model Export & Smart Downloader

Because of GLiNER2's dynamic architecture (which cycles dynamically over a sequence of JSON prompts rather than acting as a static FeedForward layer), the PyTorch model must be exported into a fragmented pipeline. We provide two architectures:

#### 1. V2 Architecture (Zero-Copy IOBinding ⚡ - Recommended)
Fuses data manipulation operations directly into the ONNX graph. Tensors stay inside the GPU/NPU VRAM, yielding a ~30% performance boost. Generates 8 files:
*   `encoder...`
*   `token_gather...`
*   `span_rep...`
*   `schema_gather...`
*   `count_pred_argmax...`
*   `count_lstm_fixed...`
*   `scorer...`
*   `classifier...`
*   `tokenizer.json`

*(Export script: `onnx_conversion_scripts/export_gliner2_onnx_fragments_v2.py`)*

#### 2. V1 Architecture (Standard CPU Slicing - Legacy)
Standard PyTorch export into 5 files. Slower on discrete GPUs due to PCIe transfers, but completely stable on older hardware.
*   `encoder...`
*   `span_rep...`
*   `count_pred...`
*   `count_lstm...`
*   `classifier...`
*   `tokenizer.json`

*(Export script: `onnx_conversion_scripts/export_gliner2_onnx.py`)*

#### 🌍 Smart HF Downloader
When downloading a model via `Gliner2Engine::from_pretrained("SemplificaAI/gliner2-multi-v1-onnx", Some("fp16_v2"), ...)`, the Rust engine uses an **OS-Aware Smart Downloader** to fetch only the optimal variant:
- **Windows/Linux**: Downloads the `_fp16_iobinding.onnx` variants to maximize CUDA/ROCm/TensorRT performance.
- **macOS/iOS**: Automatically falls back to standard `_fp16.onnx` to ensure compatibility with Apple CoreML.

This mechanism cuts bandwidth and disk usage by ~50% while delivering the best possible performance out of the box!

## 💻 Usage

```rust
use gliner2_inference::{Gliner2Engine, Gliner2Config, SchemaTask, ModelType};

fn main() -> anyhow::Result<()> {
    // Initialize ONNX Runtime environment (automatically binds to available NPUs/GPUs)
    ort::init().with_name("GLiNER2_Engine").commit()?;

    // Configure engine
    let config = Gliner2Config {
        models_dir: "./models/fastino_gliner2_multi_v1_fp16".to_string(),
        max_width: 8, // Maximum tokens per span
        model_type: ModelType::HuggingFace, // Automatically routes tensors correctly
    };
    
    // Load and build session
    let engine = Gliner2Engine::new(config)?;

    let text = "Mario Rossi works at Apple in Cupertino.";
    
    // Create schema tasks dynamically
    let tasks = vec![
        SchemaTask::Entities(vec![
            "person".to_string(), 
            "organization".to_string(), 
            "location".to_string()
        ]),
        SchemaTask::Relations("works_at".to_string(), vec![
            "head".to_string(), 
            "tail".to_string()
        ]),
        SchemaTask::Classifications("sentiment".to_string(), vec![
            "positive".to_string(), 
            "negative".to_string()
        ])
    ];

    // Extract features
    let (entities, relations, classifications) = engine.extract(text, &tasks)?;

    for entity in entities {
        println!("Found: {} (Label: {} - Score: {:.2}%)", entity.text, entity.label, entity.score * 100.0);
    }
    
    Ok(())
}
```

## Model Types

### HuggingFace Base Model
- **Type**: `ModelType::HuggingFace`
- **Source**: `fastino/gliner2-multi-v1` from HuggingFace
- **Usage**: Free for testing and development
- **Performance**: Good baseline, trained on general data

### Premium Fine-Tuned Model (PyTorch Local Export)
- **Type**: `ModelType::PyTorch`
- **Access**: Proprietary fine-tuned weights
- **Performance**: Superior accuracy on domain-specific entities

## ⚖️ License

Licensed under the [Apache License, Version 2.0](LICENSE).  
This project was developed by Dario Finardi at Semplifica s.r.l.
# Release Notes

## [v0.4.1] - 2026-04-21
### 🎉 Improvements
- **Advanced Multitask Extraction**: Expanded `test_hf_download.rs` to demonstrate concurrent extraction of Entities, Relations, and Classifications (Sentiment/Topic).
- **Relations Schema Fix**: Corrected the relations schema mapping to properly use `head` and `tail` node identifiers.
- **Internationalization**: Translated remaining Italian logs and comments to English for broader accessibility.

## [v0.3.0] - 2026-04-21
### 🎉 New Features
- **HuggingFace Hub Auto-Download**: Added `Gliner2Engine::from_pretrained()` to dynamically download ONNX models (FP16/FP32) directly from HuggingFace via the official `hf-hub` crate.
- **Download Stats Tracking**: Native API calls inject the required `User-Agent` HTTP headers (`<library_name>/<version>; rust/unknown; <os_name>/unknown`) directly respecting HuggingFace's model download statistics policies.
- **Dynamic Execution Lengths (CountLSTM)**: Replaced `CompileSafeGRU` loop unrolling in PyTorch with a fully dynamic native `nn.GRU` during ONNX export. The `Gather` out-of-bounds error on variable-length texts is now permanently resolved!

### 🧹 Fixes & Cleanups
- Removed obsolete dependencies and hardcoded references to the old `lmo3` checkpoints.
- Removed arbitrary length caps and fixed Python export logic for sequence counts, avoiding invalid loop unrolling.
- Optimized and refactored standard examples (`test_simple.rs`, `run_inference.rs`) and added `test_hf_download.rs`.

---

## [v0.2.3]
- Initial functional release supporting basic Pytorch-converted fragments with local paths.


## [v0.5.0] - 2026-04-23
### ⚙️ Dynamic Inference Parameters (`InferenceParams`)
Introduced the `InferenceParams` struct to the `extract()` function, allowing per-request control over inference behavior without rebuilding the engine:
- **`threshold`**: Controls the confidence score threshold (default `0.5`).
- **`flat_ner`**: When `false` (default), overlapping entities with different labels are allowed (e.g. "Apple Inc." as `organization` and "Apple" as `company`). When `true`, strict greedy NMS removes any overlap, regardless of label.

### 📐 Note on `max_width`
You may notice that `max_width` (the maximum length of an entity in tokens) is not part of `InferenceParams` but remains in `Gliner2Config` at engine initialization. 
**Why isn't it dynamic?** In the high-performance V2 IOBinding architecture, the span representation layer is fused directly into the ONNX computational graph. During export, the dimension for `max_width` is hard-baked into the model tensors (e.g., `[batch, num_words, 8, hidden_size]`). Changing `max_width` at runtime in V2 would cause an immediate ONNX shape mismatch error. Thus, it remains a structural configuration parameter.

## [v0.4.2] - 2026-04-22
### 🚀 Smart Downloads & HF Ecosystem
- **OS-Aware Model Downloader**: `from_pretrained` logic has been heavily optimized. It now parses `std::env::consts::OS` to selectively download only the `_fp16_iobinding` variants for Linux/Windows (CUDA/ROCm) and standard `_fp16` for macOS (CoreML). This drops the V2 download size from 1.2GB to ~600MB.
- **Manual IOBinding Override**: Introduced `GLINER2_NO_IOBINDING=1` environment variable to force fallback to standard FP16 execution even on supported hardware.
- **Hugging Face Model Card**: Generated the optimal `README_HF.md` to properly showcase the V2 capabilities on the Hub.
- **Automated V2 Uploads**: Included `upload_v2_to_hf.py` inside `onnx_conversion_scripts` to streamline uploading the double V2 variants (`fp16_v2` and `fp32_v2`) to the Hugging Face ecosystem.

## [v0.4.1] - 2026-04-22
### ⚡ V2 Zero-Copy IOBinding Architecture
- **Performance**: Up to 30% reduction in inference latency (currently tested and verified on NVIDIA RTX GPUs and AMD Ryzen CPUs).
- **ONNX Graph Fusion**: Ported previously CPU-bound operations (`Gather` for Token/Schema representations, `ArgMax` for prediction counts, and `MatMul` replacing Einsum for the Scorer) directly into the ONNX session.
- **IOBinding Bypass**: Data now remains fully encapsulated within the VRAM buffer avoiding expensive PCIe bus transactions.
- **Facade Auto-detect**: Built an intelligent `Gliner2Engine` wrapper to automatically detect whether to use V1 CPU-slicing logic or V2 IOBinding without breaking changes to the consumer code.
