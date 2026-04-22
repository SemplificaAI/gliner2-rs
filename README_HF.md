---
language: 
  - it
  - en
  - fr
  - es
  - de
tags:
  - gliner
  - named-entity-recognition
  - onnx
  - rust
pipeline_tag: token-classification
---

# GLiNER2 Multi-v1 (ONNX Fragmented & IOBinding)

This repository contains the ONNX-exported weights for **GLiNER2-Multi-v1**.
The model is specifically exported in a fragmented format (encoder, span_rep, count_pred, count_lstm, classifier) to be directly compatible with [gliner2-rs](https://github.com/SemplificaAI/gliner2-rs), the official Zero-Python Native Rust Inference Engine for GLiNER2.

### 🆕 Update: V2 Zero-Copy IOBinding Models Available!
We have introduced **V2 fused models** (`fp16_v2` and `fp32_v2`) that fuse `Gather`, `ArgMax`, and `MatMul` operations directly into the ONNX graph. By using ORT's `IoBinding`, these models ensure that tensors **never leave the GPU/NPU VRAM**, completely bypassing the PCIe bus and reducing inference latency by ~30% on discrete GPUs.

## 📂 Available Variants

*   **`fp16_v2`** *(Recommended)*: Zero-Copy VRAM optimized models. Fused operations with Full IO Types (native FP16). Drastically reduces inference time on NVIDIA CUDA, AMD ROCm, and Apple CoreML. Requires `gliner2-rs >= 0.4.1`.
*   **`fp32_v2`**: High precision V2 fusions for CPU execution (AVX, XNNPACK). Requires `gliner2-rs >= 0.4.1`.
*   **`fp16`** *(Standard)*: Legacy Float16 ONNX models. Slower on discrete GPUs due to PCIe transfers, but completely stable and supported everywhere.
*   **`fp32`** *(Standard)*: Legacy Float32 ONNX models.

## 🚀 Performance & Benchmarks

The ONNX conversion, combined with the Rust native engine (`ort` binding), allows this model to run extremely fast on both GPUs and edge devices like NPUs.

**Benchmark Task:** Tested on complex text extraction tasks spanning up to 62 classes (metrics normalized per extracted entity to allow cross-device comparison).

| Hardware | Execution Provider | Model Variant | Avg Time / Entity |
| :--- | :--- | :--- | :--- |
| **NVIDIA RTX 4090** | CUDA (V2 IOBinding) | `fp16_v2` | **~7.0 ms** ⚡ |
| **NVIDIA RTX 3090** | CUDA (V2 IOBinding) | `fp16_v2` | **~7.2 ms** ⚡ |
| **NVIDIA RTX 4090** | CUDA (V1 Standard) | `fp16` | **~12.0 ms** 🚀 |
| **NVIDIA RTX 3090** | CUDA (V1 Standard) | `fp16` | **~11.6 ms** 🚀 |
| **Qualcomm Snapdragon X Elite** | QNN (V2 NPU Native) | `fp16_v2` | **~19.36 ms** ✨ |
| **Qualcomm Snapdragon X Elite** | QNN (V1 NPU Native) | `fp16` | **~22.78 ms** |
| **AMD Ryzen 9 5900XT** (16-Core) | CPU (x86 AVX2) | `fp32_v2` | **~20.6 ms** 💻 |
| **Qualcomm Snapdragon X Elite** | CPU (V2 ARM NEON) | `fp32_v2` | **~24.32 ms** |
| **Qualcomm Snapdragon X Elite** | CPU (V1 ARM NEON) | `fp32` | **~28.62 ms** |

## 📦 Usage in Rust

You can dynamically download and execute these ONNX weights from Rust in 3 lines of code. With `gliner2-rs >= 0.4.1`, the engine automatically detects if you are using V1 or V2 models and routes the execution perfectly.

```rust
use gliner2_inference::{Gliner2Engine, ModelType, SchemaTask};

// Auto-downloads the V2 FP16 models from this HuggingFace repo
// and automatically switches to the high-performance IOBinding Engine!
let engine = Gliner2Engine::from_pretrained(
    "SemplificaAI/gliner2-multi-v1-onnx",
    Some("fp16_v2"),
    ModelType::HuggingFace
)?;

let text = "Mario Rossi works at Apple in Cupertino.";
let tasks = vec![
    SchemaTask::Entities(vec![
        "person".to_string(), "organization".to_string(), "location".to_string()
    ])
];

let (entities, _, _) = engine.extract(text, &tasks)?;
```

## 🔧 Model Fixes
- `count_lstm` has been successfully exported with dynamic sequence unrolling by replacing the training `CompileSafeGRU` with native `nn.GRU`, resolving Out-of-Bounds `Gather` ONNX errors for variable length texts.
- **(V2)** `Scorer` now uses a heavily optimized fused combination of `Reshape` + `MatMul` + `Transpose` instead of `Einsum`, ensuring compatibility with execution providers that don't support `Einsum` in FP16 (e.g. QNN, CoreML).
