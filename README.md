# gliner2-rs

[![GitHub](https://img.shields.io/badge/GitHub-SemplificaAI/gliner2--rs-blue?style=flat-square&logo=github)](https://github.com/SemplificaAI/gliner2-rs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/Version-0.3.2-brightgreen.svg)](https://github.com/SemplificaAI/gliner2-rs)
[![Status](https://img.shields.io/badge/Status-Beta-blue.svg)](https://github.com/SemplificaAI/gliner2-rs)

**Native Rust Inference Engine for GLiNER2**

`gliner2-rs` is a high-performance, Zero-Python inference engine designed to execute **GLiNER2** models using **ONNX Runtime**. It allows for extracting Named Entities (NER), Relations, and Global Classifications natively in Rust with maximum speed, supporting both CPU and NVIDIA GPU (CUDA) via hardware-accelerated Tensor operations.

This crate completely replicates the advanced sub-word tokenization and prompt-generation logic of GLiNER2's `processor.py` internally, using the official `tokenizers` crate for zero-overhead BPE tokenization.

*Copyright 2026 Dario Finardi, Semplifica s.r.l.*  
*Licensed under Apache License 2.0*

## 🚀 Features

### Key Features

- **End-to-End Execution**: Full recreation of the GLiNER2 inference loop natively in Rust.
- **Multi-Task Extraction**: Supports Entity Extraction, Relation Extraction, and Text Classifications in a single forward pass.
- **Hardware Accelerated**: Dynamically uses QNN (Qualcomm NPU), CoreML (Apple Silicon), OpenVINO (Intel/AMD), CUDA Execution Provider if an NVIDIA GPU is available, falling back to optimized XNNPACK/CPU execution.
- **FP16 & FP32 Support**: Fully compatible with Half-Precision (Float16) ONNX exports to cut memory footprints in half.
- **Zero-Copy Tensor Flow**: Direct injection of raw hidden states across multiple neural network slices without CPU-GPU memory swaps.
- **Built-in NMS**: Automatic Non-Maximum Suppression (NMS) to elegantly remove overlapping fictions entities based on their probabilities.

---

## 📊 Benchmark & Performance

Tested on complex text extraction tasks spanning up to 62 classes. Metrics are normalized per extracted entity (15 entities) to allow precise cross-device and cross-language comparisons.

### 🖥️ Rust ONNX vs Python PyTorch (Desktop & Discrete GPUs)
Comparison of a 50-run continuous benchmark on x86_64 architecture with NVIDIA GPUs.

| Language | Engine (Hardware) | Total Time (50 runs) | Avg Time / Sentence | Avg Time / Entity (15) |
| :--- | :--- | :--- | :--- | :--- |
| **Python 3.10** | PyTorch (RTX 4090) | **~0.88 s** 🚀 | 4.40 ms | **1.17 ms** |
| **Python 3.10** | PyTorch (RTX 3090) | **~0.90 s** 🚀 | 4.52 ms | **1.20 ms** |
| **Rust** | ONNX Runtime CUDA (RTX 4090) | **~8.18 s** | 40.90 ms | **10.90 ms** |
| **Rust** | ONNX Runtime CUDA (RTX 3090) | **~8.59 s** | 42.97 ms | **11.45 ms** |
| **Python 3.10** | PyTorch (Ryzen 5900XT CPU) | **~7.26 s** | 36.33 ms | **9.68 ms** |
| **Rust** | ONNX Runtime (Ryzen 5900XT CPU) | **~13.75 s** | 68.76 ms | **18.33 ms** |

**Understanding the GPU Gap (The Fragmented ONNX Pipeline Overhead):**
While PyTorch is astonishingly fast on discrete GPUs, the gap is not due to pure mathematical compute speed, but rather **memory bandwidth bottlenecks**. 
Because of GLiNER2's dynamic architecture, the ONNX model had to be split into a pipeline of 5 fragmented files (`encoder`, `span_rep`, `count_pred`, `count_lstm`, `classifier`). 
During Rust ONNX inference, intermediate tensors (often tens of Megabytes) must be constantly copied back and forth between the CPU system RAM and the GPU VRAM across the **PCIe bus** for *each* of the 5 fragments. PyTorch, on the other hand, utilizes a CUDA Caching Allocator and keeps the entire computational graph and memory strictly inside the GPU VRAM without ever returning to the CPU until the final logits are ready.

However, Rust ONNX becomes highly competitive or superior on **Unified Memory Architectures** (like Apple Silicon or ARM Snapdragon) where the CPU-GPU transfer cost is zero, and it completely dominates PyTorch in **Cold Start** scenarios.

### 🐍 Rust vs Python on ARM (Snapdragon X Elite)
Comparison between native Rust ONNX execution and standard Python PyTorch inference on the same ARM hardware.

| Environment | Backend (Hardware) | Model | Startup Time | Entities Extracted | Avg Time (Total) | Avg Time / Entity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rust** | ONNX Runtime (NPU - QNN) | `gliner2-multi-v1-onnx` | **~2.02 s** ⚡ | 41 | 0.65 s | **~16.07 ms** |
| **Rust** | ONNX Runtime (CPU ARM64) | `gliner2-multi-v1-onnx` | **~1.89 s** | 41 | 0.68 s | **~16.58 ms** |
| **Python 3.12** | PyTorch (CPU ARM64) | `fastino/gliner2-multi-v1` | **~9.21 s** 🐢 | 15 | 0.33 s | **~22.02 ms** |
| **Python 3.12** | PyTorch (GLiNER2 - CPU ARM64) | `SemplificaAI/gliner2-multi-v1` | **~10.89 s** 🐢 | 18 | 0.35 s | **~19.41 ms** |

**Takeaways:**
- **Cold Start (Startup Time):** Rust completely skips the massive Python/PyTorch loading overhead, initializing the engine and weights **>4.5x faster** (~2s vs ~9s). This makes it vastly superior for edge devices, serverless functions, or quick on-demand extractions.
- **Inference Speed:** Rust ONNX natively leverages the NPU (which PyTorch currently struggles to target effectively on Windows on ARM), gaining a solid speed advantage even when extracting 4x more entities.

---

## Installation & Setup

### Installation Steps

1. Clone this repository or add `gliner2-rs` to your `Cargo.toml`.
2. Ensure you have the `onnxruntime` C/C++ libraries available on your system path.
3. Export the GLiNER2 models to ONNX fragmented versions. 

### Model Export

Because of GLiNER2's dynamic architecture (which cycles dynamically over a sequence of JSON prompts rather than acting as a static FeedForward layer), the PyTorch model must be exported into 5 fragmented files using a tracing script:

*   `encoder_fp16.onnx` (or `encoder_fp32.onnx`)
*   `span_rep_fp16.onnx` (or `span_rep_fp32.onnx`)
*   `count_pred_fp16.onnx` (or `count_pred_fp32.onnx`)
*   `count_lstm_fp16.onnx` (or `count_lstm_fp32.onnx`)
*   `classifier_fp16.onnx` (or `classifier_fp32.onnx`)
*   `tokenizer.json`

Use the provided `onnx_conversion_scripts/06_export_base_multi_v1.py` script to trace and export models cleanly. Place these files in a specific directory (e.g. `./models/`).

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
