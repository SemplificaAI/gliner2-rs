# gliner2-rs

[![GitHub](https://img.shields.io/badge/GitHub-SemplificaAI/gliner2--rs-blue?style=flat-square&logo=github)](https://github.com/SemplificaAI/gliner2-rs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/Version-0.2.3-brightgreen.svg)](https://github.com/SemplificaAI/gliner2-rs)
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
