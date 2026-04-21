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
