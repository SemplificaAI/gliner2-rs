---
pipeline_tag: token-classification
tags:
- gliner
- gliner2
- onnx
- rust
- iobinding
language:
- it
- en
- fr
- de
- es
- pt
---

# GLiNER2-multi-v1 (ONNX & Rust IOBinding)

This repository contains the ONNX exported weights for **[GLiNER2-multi-v1](https://huggingface.co/SemplificaAI/gliner2-multi-v1)**, fully optimized for inference using the official `gliner2-rs` Rust engine.

## 🚀 Native Rust Inference
This model is designed to be run using our high-performance **Zero-Python** inference engine:
[https://github.com/SemplificaAI/gliner2-rs](https://github.com/SemplificaAI/gliner2-rs)

```rust
use gliner2_inference::*;

// For maximum performance on RTX GPUs and Unified Memory architectures,
// use the new "fp16_v2" IOBinding subfolder!
let engine = Gliner2Engine::from_pretrained(
    "SemplificaAI/gliner2-multi-v1-onnx",
    Some("fp16_v2"), 
    ModelType::HuggingFace
).unwrap();
```

## 📂 Repository Structure & Available Variants

This repository provides different sets of ONNX files inside subfolders to support different hardware architectures and `gliner2-rs` versions:

### V2 Models (Zero-Copy IOBinding ⚡ - Requires gliner2-rs >= 0.4.1)
The new V2 models fuse critical operations (`Gather`, `ArgMax`, `MatMul`) directly inside the ONNX graph, ensuring tensors never leave the GPU/NPU VRAM. This yields a ~30% performance boost.
*   **`fp16_v2`**: Contains both `_fp16.onnx` and `_fp16_iobinding.onnx` files. Best for NVIDIA CUDA, AMD ROCm, Apple CoreML, and Qualcomm QNN NPUs.
*   **`fp32_v2`**: High precision V2 fusions for CPU execution (AVX, XNNPACK).

### V1 Models (Standard / Legacy - Compatible with all versions)
Fragmented standard ONNX models. Slower on discrete GPUs due to PCIe transfers, but completely stable and supported everywhere.
*   **`fp16`**: Float16 ONNX models.
*   **`fp32`**: Float32 ONNX models.

## How to update your models
To generate your own V2 fusions, you can use the `export_gliner2_onnx_fragments_v2.py` script provided in the `gliner2-rs` GitHub repository.
