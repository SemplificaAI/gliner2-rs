## [v0.5.0] - 2026-04-23
### ✨ New Features & API Improvements
- **Dynamic Inference Parameters (`InferenceParams`)**: Introduced `InferenceParams` to allow changing parameters at runtime for each `extract()` call without rebuilding the engine.
  - **`threshold`**: Now a dynamic parameter (default `0.5`). Replaces the previous hardcoded constant.
  - **`flat_ner`**: Added support for non-flat NER. If set to `false`, overlapping entities with different labels are allowed (e.g. "Mario Rossi" as `person` and "Mario" as `first_name`). If `true`, strict greedy NMS removes any overlap regardless of label.
- **Architectural Clarity on `max_width`**: Clarified that `max_width` (default `8`) remains in `Gliner2Config` rather than `InferenceParams`. In the V2 IOBinding architecture, the span representation layer is fused directly into the ONNX graph with a hard-baked dimension (`[batch, num_words, 8, hidden_size]`). Therefore, `max_width` is a structural constraint of the V2 ONNX model and cannot be modified at runtime without causing a shape mismatch error.

# Release Notes

## [v0.3.1] - 2026-04-21
### 🧹 Fixes & Cleanups
- **Accurate Span Extraction Boundaries**: Fixed a token decoding formatting bug. Entity text extraction is now mathematically sliced directly from the original character byte offsets of the input string, rather than being lossy-reconstructed by the tokenizer decoder. This completely preserves the native spacing, punctuation, and capitalization of entities like "s.r.l." instead of producing "s . r . l .".

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
