"""
PyTorch vs ONNX parity check for the GLiNER2 span export.

Note: this expects the flat layout produced by export_span_v3.py. Exports using
the legacy fp32_v2/ + fp16_v2/ subfolders should be pointed at the subfolder
itself, or re-exported.

Compares every ONNX fragment against the matching PyTorch module on random
inputs, across all three precision variants.

Usage:
    python verify_parity.py \
        --model_path fastino/gliner2-privacy-filter-PII-multi \
        --onnx_dir models/pii-onnx

Exits with status 1 if any check exceeds its tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# **Relative** tolerances, scaled by the reference's largest magnitude. An
# absolute threshold is unusable here because the fragments operate at very
# different scales: `span_rep` emits activations up to ~9e3, while `scorer` has
# already gone through the sigmoid and lives in [0,1].
TOL_FP32 = 1e-5
TOL_FP16 = 5e-3

# `scorer` is already a probability in [0,1], so relative and absolute error
# coincide. With random inputs the dot products over 768 dimensions have
# magnitude ~sqrt(768), the logits saturate, and an FP16 perturbation around the
# sigmoid transition is worth a few thousandths of a probability. That is the
# cost of half precision on a probability, not a defect of the export; on real
# encoder activations the effect is smaller.
TOL_FP16_PROB = 2e-2


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(sess, feeds: dict) -> list[np.ndarray]:
    typed = {}
    for i in sess.get_inputs():
        v = feeds[i.name]
        want = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }[i.type]
        typed[i.name] = np.asarray(v).astype(want)
    return sess.run(None, typed)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, float, float, bool]] = []

    def add(self, fragment: str, variant: str, delta: float, tol: float) -> None:
        self.rows.append((fragment, variant, delta, tol, delta <= tol))

    @staticmethod
    def relative(ref: np.ndarray, got: np.ndarray) -> float:
        """Largest error, scaled by the reference's magnitude."""
        ref = ref.astype(np.float64)
        got = got.astype(np.float64)
        scale = max(float(np.abs(ref).max()), 1e-12)
        return float(np.abs(ref - got).max() / scale)

    def failed(self) -> bool:
        return any(not ok for *_, ok in self.rows)

    def show(self) -> None:
        print()
        print(f"{'fragment':<26} {'variant':<18} {'rel. error':>12} {'tolerance':>12}  result")
        print("-" * 82)
        for frag, var, d, tol, ok in self.rows:
            print(f"{frag:<26} {var:<18} {d:>12.3e} {tol:>12.3e}  {'OK' if ok else 'FAILED'}")
        print()
        print("FAILED" if self.failed() else "all checks passed")


def _variants(onnx_dir: Path, stem: str) -> list[tuple[str, Path, float]]:
    out = []
    for suffix, tol in (("_fp32", TOL_FP32), ("_fp16", TOL_FP16), ("_fp16_iobinding", TOL_FP16)):
        p = onnx_dir / f"{stem}{suffix}.onnx"
        if p.exists():
            out.append((suffix.lstrip("_"), p, tol))
    return out


def _compare(
    report: Report,
    stem: str,
    onnx_dir: Path,
    feeds: dict,
    ref: np.ndarray,
    out_index: int = 0,
    tol_fp16: float = TOL_FP16,
) -> None:
    for name, path, tol in _variants(onnx_dir, stem):
        if name != "fp32":
            tol = tol_fp16
        got = _run(_session(path), feeds)[out_index]
        report.add(stem, name, Report.relative(ref, got), tol)


# ─────────────────────────────────────────────────────────────────────────────
# span
# ─────────────────────────────────────────────────────────────────────────────
def verify_span(model_path: str, onnx_dir: Path) -> Report:
    from gliner2 import Extractor

    model = Extractor.from_pretrained(model_path)
    model.eval()
    H = model.encoder.config.hidden_size
    mw = model.max_width
    max_count = getattr(model.count_embed, "max_count", 20)

    torch.manual_seed(0)
    report = Report()
    SEQ, W, M = 40, 24, 5

    ids = torch.randint(5, 1000, (1, SEQ))
    mask = torch.ones(1, SEQ, dtype=torch.long)
    with torch.no_grad():
        hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    _compare(report, "encoder", onnx_dir,
             {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}, hidden.numpy())

    word_idx = torch.arange(W)
    _compare(report, "token_gather", onnx_dir,
             {"last_hidden_state": hidden.numpy(), "word_indices": word_idx.numpy()},
             hidden[:, word_idx, :].numpy())

    text_embs = torch.randn(1, W, H)
    spans = torch.zeros(1, W * mw, 2, dtype=torch.long)
    for w in range(W):
        for k in range(mw):
            if w + k < W:
                spans[0, w * mw + k] = torch.tensor([w, w + k])
    with torch.no_grad():
        ref = model.span_rep(text_embs, spans)
    _compare(report, "span_rep", onnx_dir,
             {"hidden_states": text_embs.numpy(), "span_idx": spans.numpy()}, ref.numpy())

    schema_idx = torch.arange(M + 1)
    gathered = hidden[0, schema_idx, :]
    _compare(report, "schema_gather", onnx_dir,
             {"last_hidden_state": hidden.numpy(), "schema_indices": schema_idx.numpy()},
             gathered[0:1, :].numpy(), out_index=0)
    _compare(report, "schema_gather[field]", onnx_dir,
             {"last_hidden_state": hidden.numpy(), "schema_indices": schema_idx.numpy()},
             gathered[1:, :].numpy(), out_index=1)

    field_embs = torch.randn(M, H)
    with torch.no_grad():
        ref = model.count_embed(field_embs, max_count)
    _compare(report, "count_lstm_fixed", onnx_dir,
             {"field_embs": field_embs.numpy()}, ref.numpy())

    with torch.no_grad():
        ref = model.classifier(field_embs).squeeze(-1)
    _compare(report, "classifier", onnx_dir, {"field_embs": field_embs.numpy()}, ref.numpy())

    # count_pred_argmax emits int64: compare for exact equality
    pc = torch.randn(1, H)
    with torch.no_grad():
        ref_count = torch.argmax(model.count_pred(pc), dim=-1).numpy()
    for name, path, _ in _variants(onnx_dir, "count_pred_argmax"):
        got = _run(_session(path), {"pc_emb": pc.numpy()})[0]
        report.add("count_pred_argmax", name, 0.0 if np.array_equal(ref_count, got) else 1.0, 0.0)

    # scorer: sigmoid(span . struct)
    span_emb = torch.randn(1, W, mw, H)
    struct = torch.randn(max_count, M, H)
    ref = torch.sigmoid(torch.einsum("bwkh,cmh->cwkm", span_emb, struct))
    _compare(report, "scorer", onnx_dir,
             {"span_embeddings": span_emb.numpy(), "struct_proj": struct.numpy()}, ref.numpy(),
             tol_fp16=TOL_FP16_PROB)

    return report


def main() -> int:
    p = argparse.ArgumentParser(description="PyTorch vs ONNX parity (span)")
    p.add_argument("--model_path", required=True)
    p.add_argument("--onnx_dir", required=True)
    args = p.parse_args()

    onnx_dir = Path(args.onnx_dir)
    print(f"checkpoint   : {args.model_path}")
    print(f"onnx         : {onnx_dir}")

    report = verify_span(args.model_path, onnx_dir)
    report.show()
    return 1 if report.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
