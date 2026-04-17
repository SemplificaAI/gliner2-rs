# gliner2-rs

[![GitHub](https://img.shields.io/badge/GitHub-SemplificaAI/gliner2--rs-blue?style=flat-square&logo=github)](https://github.com/SemplificaAI/gliner2-rs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/Version-0.1.0--alpha-red.svg)](https://github.com/SemplificaAI/gliner2-rs)
[![Status](https://img.shields.io/badge/Status-Bugged-orange.svg)](https://github.com/SemplificaAI/gliner2-rs)

**Native Rust Inference Engine for GLiNER2**

> **WARNING: This is the first alpha release (v0.1.0-alpha) and contains known bugs.**
> 
> **Current Issues:**

- HuggingFace model compatibility problems with `onnx::Cast_1` input
- Entity extraction may produce fragmented results  
- Some tensor shape mismatches with certain model types

**Status:** Under active development - expect breaking changes

`gliner2-rs` is a high-performance, Zero-Python inference engine designed to execute **GLiNER2** models using **ONNX Runtime**. It allows for extracting Named Entities (NER), Relations, and Global Classifications natively in Rust with maximum speed, supporting both CPU and NVIDIA GPU (CUDA) via hardware-accelerated Tensor operations.

This crate completely replicates the advanced sub-word tokenization and prompt-generation logic of GLiNER2's `processor.py` internally, using the official `tokenizers` crate for zero-overhead BPE tokenization.

*Copyright 2026 Dario Finardi, Semplifica s.r.l.*  
*Licensed under Apache License 2.0*

## 🚀 Features

### Key Features

- **End-to-End Execution**: Full recreation of the GLiNER2 inference loop natively in Rust.
- **Multi-Task Extraction**: Supports Entity Extraction, Relation Extraction, and Text Classifications in a single forward pass.
- **Hardware Accelerated**: Dynamically uses the CUDA Execution Provider if an NVIDIA GPU is available, falling back to optimized CPU execution.
- **FP16 & FP32 Support**: Fully compatible with Half-Precision (Float16) ONNX exports to cut memory footprints in half.
- **Zero-Copy Tensor Flow**: Direct injection of raw hidden states across multiple neural network slices without CPU-GPU memory swaps.
- **Built-in NMS**: Automatic Non-Maximum Suppression (NMS) to elegantly remove overlapping fictions entities based on their probabilities.

## Installation & Setup

### Installation Steps

1. Add `gliner2-rs` to your `Cargo.toml`.
2. Ensure you have the `onnxruntime` C/C++ libraries available on your system path.
3. Export the GLiNER2 models to ONNX fragmented versions. 

### Model Export

Because of GLiNER2's dynamic architecture (which cycles dynamically over a sequence of JSON prompts rather than acting as a static FeedForward layer), the PyTorch model must be exported into 5 fragmented files using a tracing script:

*   `encoder_fp16.onnx`
*   `span_rep_fp16.onnx`
*   `count_pred_fp16.onnx`
*   `count_lstm_fp16.onnx`
*   `classifier_fp16.onnx`
*   `tokenizer.json`

Place these files in a specific directory (e.g. `./models/`).

## 💻 Usage

```rust
use gliner2_inference::{Gliner2Engine, Gliner2Config, SchemaTask, ModelType};

fn main() -> anyhow::Result<()> {
    // Initialize ONNX Runtime environment
    ort::init().with_name("GLiNER2_Engine").commit()?;

    // Configure engine
    let config = Gliner2Config {
        models_dir: "./models/fragments_fp16".to_string(),
        max_width: 8, // Maximum tokens per span
        model_type: ModelType::Lmo3, // Use public lmo3 model
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

### Public Model (Lmo3)
- **Type**: `ModelType::Lmo3`
- **Source**: `lmo3/gliner2-multi-v1-onnx` from HuggingFace
- **Usage**: Free for testing and development
- **Performance**: Good baseline, trained on general data

### Premium Model (Semplifica) 
- **Type**: `ModelType::Semplifica`
- **Access**: Restricted to Semplifica s.r.l. customers only
- **Performance**: Superior accuracy on domain-specific entities
- **Contact**: sales@semplifica.ai for commercial licensing

**Note**: The Semplifica model is protected and will return an authorization error if accessed without proper licensing.

## ⚖️ License

Licensed under the [Apache License, Version 2.0](LICENSE).  
This project was developed by Dario Finardi at Semplifica s.r.l.
