"""
GLiNER2 *span* ONNX Fragment Exporter v3  -  gliner2 2.0.0 / IOBinding-ready
============================================================================
Exports any checkpoint using the **span** architecture (`config.json` with no
`"architecture"` field, or `"architecture": "span"`) into 8 ONNX fragments:

    fastino/gliner2-privacy-filter-PII-multi <- primary target
    fastino/gliner2-multi-v1
    fastino/gliner2-base-v1
    any local fine-tune of the same family

Multi-EP compatibility choices:

    +------------------+-----------------------+--------------------+
    | Backend          | File variant          | Notes              |
    +------------------+-----------------------+--------------------+
    | Qualcomm QNN NPU | _fp16_iobinding.onnx  | Gather/ArgMax/MM   |
    |                  |                       | offloaded to HTP   |
    | Apple CoreML     | _fp16.onnx            | keep_io_types=True |
    | NVIDIA CUDA      | _fp16_iobinding.onnx  | IOBinding in VRAM  |
    | AMD ROCm         | _fp16_iobinding.onnx  |                    |
    | Intel OpenVINO   | _fp32.onnx            |                    |
    | CPU (XNNPACK)    | _fp32.onnx            | universal fallback |
    +------------------+-----------------------+--------------------+

    - opset 17, supported by every modern EP
    - MatMul+Reshape+Transpose instead of Einsum in the scorer: Einsum is not
      supported across all QNN/CoreML backends
    - Gather (axis=1) for token_gather / schema_gather: a base op, available
      everywhere including Qualcomm HTP
    - ArgMax fused into count_pred_argmax, natively offloadable
    - count_lstm_fixed: the GRU is unrolled to MAX_COUNT fixed steps, so there
      are no dynamic loops - CoreML and QNN do not support them

Pipeline (IOBinding chain):

    encoder(input_ids, attention_mask)
        -> last_hidden_state [VRAM]
        |
        +- token_gather(last_hidden_state, word_indices)
        |       -> text_embs
        |       +- span_rep(text_embs, span_idx)
        |               -> span_embeddings
        |
        +- schema_gather(last_hidden_state, schema_indices)
                -> pc_emb, field_embs
                +- count_pred_argmax(pc_emb)      -> pred_count (int64)
                +- count_lstm_fixed(field_embs)   -> struct_proj [MAX_COUNT,M,H]
                        +- scorer(span_embeddings, struct_proj)
                                -> entity_scores

    classifier(field_embs) -> logits    (classification tasks only)

MAX_COUNT is read from `model.count_embed.max_count` rather than hard-coded,
and a `boundary` checkpoint (GLiNER2.5) is rejected outright instead of being
exported into something silently wrong.

Usage:
    python export_span_v3.py \
        --model_path fastino/gliner2-privacy-filter-PII-multi \
        --out_dir models/pii-onnx

    python export_span_v3.py \
        --model_path /path/to/checkpoint/best \
        --out_dir models/my-finetune
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from gliner2 import Extractor

# Historical default; overridden at run time from model.count_embed.max_count.
MAX_COUNT: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 1 - Encoder
# ─────────────────────────────────────────────────────────────────────────────
class EncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor,      # [1, seq_len]  int64
        attention_mask: torch.Tensor,  # [1, seq_len]  int64
    ) -> torch.Tensor:                 # [1, seq_len, H]  float32
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 2 - TokenGather
# Takes the first sub-token embedding of each word (word-level pooling).
# ─────────────────────────────────────────────────────────────────────────────
class TokenGatherWrapper(nn.Module):
    """
    Input:
        hidden_state  [1, seq_len, H]  - encoder output
        word_indices  [num_words]      - index of each word's first sub-token
                                         (word_to_token_maps[:,0] on the Rust side)
    Output:
        text_embs  [1, num_words, H]
    """

    def forward(
        self,
        hidden_state: torch.Tensor,  # [1, seq_len, H]
        word_indices: torch.Tensor,  # [num_words]  int64
    ) -> torch.Tensor:               # [1, num_words, H]
        # ONNX: Gather(hidden_state, word_indices, axis=1)
        return hidden_state[:, word_indices, :]


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 3 - SpanRep
# ─────────────────────────────────────────────────────────────────────────────
class SpanRepWrapper(nn.Module):
    def __init__(self, span_rep: nn.Module):
        super().__init__()
        self.span_rep = span_rep

    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, num_words, H]
        span_idx: torch.Tensor,        # [1, num_spans, 2]  int64
    ) -> torch.Tensor:                 # [1, num_words, max_width, H]
        return self.span_rep(hidden_states, span_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 4 - SchemaGather
# One Gather for prompt and field embeddings from a single index tensor.
# ─────────────────────────────────────────────────────────────────────────────
class SchemaGatherWrapper(nn.Module):
    """
    Input:
        hidden_state    [1, seq_len, H]
        schema_indices  [M+1]  int64
                        schema_indices[0]  = prompt_tok_idx
                        schema_indices[1:] = field_tok_indices

    Output:
        pc_emb      [1, H]  - embedding of the [P] (prompt) token
        field_embs  [M, H]  - embeddings of the fields/labels
    """

    def forward(
        self,
        hidden_state: torch.Tensor,    # [1, seq_len, H]
        schema_indices: torch.Tensor,  # [M+1]  int64
    ):  # -> (pc_emb [1,H], field_embs [M,H])
        # Gather: hidden_state[0, schema_indices, :]  → [M+1, H]
        gathered = hidden_state[0, schema_indices, :]  # [M+1, H]
        # Split: Slice ops – entrambi diventano output VRAM separati in IOBinding
        pc_emb = gathered[0:1, :]   # [1, H]  – staticSlice (start=0, end=1)
        field_embs = gathered[1:, :]  # [M, H]  – dinamico ma stabile in ONNX
        return pc_emb, field_embs


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 5 - CountPredArgmax (fuses ArgMax)
# ─────────────────────────────────────────────────────────────────────────────
class CountPredArgmaxWrapper(nn.Module):
    """
    Input:   pc_emb  [1, H]
    Output:  pred_count  [1]  int64  (argmax of the count distribution)

    The int64 output stays int64 through FP16 conversion. Shape [1] is kept
    instead of a 0-D scalar to keep binding on the Rust side uniform: always a
    Tensor, never a scalar Value.
    """

    def __init__(self, count_pred: nn.Module):
        super().__init__()
        self.count_pred = count_pred

    def forward(self, pc_emb: torch.Tensor) -> torch.Tensor:  # [1]  int64
        logits = self.count_pred(pc_emb)               # [1, max_count]
        return torch.argmax(logits, dim=-1)             # [1]  int64


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 6 - CountLSTMFixed (output always [MAX_COUNT, M, H])
# ─────────────────────────────────────────────────────────────────────────────
class CountLSTMFixedWrapper(nn.Module):
    """
    Always runs MAX_COUNT GRU steps instead of `pred_count` steps, so the
    output has the FIXED shape [MAX_COUNT, M, H] - which is what makes
    IOBinding possible.

    Why this is correct: the GRU is causal, so output[i] depends only on
    output[0..i-1]. Slicing [:pred_count] is therefore identical to running
    with gold_count_val=pred_count. The Rust side uses pred_count to ignore
    the extra rows.

    Works with CountLSTM, CountLSTMv2 and CountLSTMoE unmodified: it is enough
    to call forward with gold_count_val=MAX_COUNT. During tracing that is a
    Python constant, so torch.arange(MAX_COUNT) becomes a Constant node.
    """

    def __init__(self, count_embed: nn.Module, max_count: int = MAX_COUNT):
        super().__init__()
        self.count_embed = count_embed
        self._max_count = max_count

    def forward(self, field_embs: torch.Tensor) -> torch.Tensor:
        # field_embs : [M, H]  (field_embs output di SchemaGather)
        # output     : [MAX_COUNT, M, H]
        return self.count_embed(field_embs, self._max_count)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 7 - Scorer (fuses Einsum + Sigmoid)
# ─────────────────────────────────────────────────────────────────────────────
class ScorerWrapper(nn.Module):
    """
    Computes the sigmoid probability of every span for every schema field,
    across all MAX_COUNT entity slots.

    scores[c, s, w, m] = sigmoid( Σ_d span[s,w,d] * struct[c,m,d] )

    Uses Reshape + MatMul + Transpose instead of Einsum, for maximum
    compatibility across EPs (CoreML, QNN, XNNPACK).

    Input:
        span_embeddings  [1, num_words, max_width, H]
        struct_proj      [MAX_COUNT, M, H]

    Output:
        entity_scores  [MAX_COUNT, num_words, max_width, M]  float32  ∈ [0,1]

    The Rust side then:
        1. reads pred_count (int64) from the count_pred_argmax session
        2. uses entity_scores[:pred_count] as the effective scores
        3. applies the threshold and NMS
    """

    def forward(
        self,
        span_embeddings: torch.Tensor,  # [1, num_words, max_width, H]
        struct_proj: torch.Tensor,       # [MAX_COUNT, M, H]
    ) -> torch.Tensor:                   # [MAX_COUNT, num_words, max_width, M]
        span = span_embeddings[0]        # [num_words, max_width, H]  – rimuovi batch dim
        nw, mw, H = span.shape
        C, M, _   = struct_proj.shape

        # Flatten dimensioni spaziali per un singolo MatMul
        span_flat = span.reshape(nw * mw, H)              # [NW*MW, H]
        struct_T  = struct_proj.reshape(C * M, H).transpose(0, 1)  # [H, C*M]

        # Dot product: span × struct^T → [NW*MW, C*M]
        scores_flat = torch.matmul(span_flat, struct_T)

        # Reshape a [nw, mw, C, M] poi permuta a [C, nw, mw, M]
        scores = scores_flat.reshape(nw, mw, C, M).permute(2, 0, 1, 3)

        return torch.sigmoid(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 8 - Classifier
# ─────────────────────────────────────────────────────────────────────────────
class ClassifierWrapper(nn.Module):
    """
    Firma allineata all'uso reale del modello span.

    `SpanExtractorModel` applica il classificatore agli embedding dei marker
    di scelta, non agli span::

        cls_embeds = schema_emb[1:]                  # [M, H]
        logits = self.classifier(cls_embeds).squeeze(-1)   # [M]

    Il v1/v2 esportava invece la firma `[1, num_labels, max_width, H]`, con
    `max_width` congelato a 8: corretta solo perche' `create_mlp` agisce
    sull'ultima dimensione, ma costringeva il runtime a replicare 8 volte lo
    stesso vettore e a scartarne 7 ottavi.
    """

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, field_embs: torch.Tensor) -> torch.Tensor:
        # [M, H] -> [M]
        return self.classifier(field_embs).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: export FP32, then convert to FP16
# ─────────────────────────────────────────────────────────────────────────────
def _export_fp32(
    module: nn.Module,
    dummy_inputs: tuple,
    out_path: Path,
    input_names: list,
    output_names: list,
    dynamic_axes: dict,
    opset: int = 17,
) -> None:
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    FP32 → {out_path.name}  ({size_mb:.1f} MB)")


def _convert_fp16(
    fp32_path: Path,
    keep_io_types: bool,
    out_path: Path | None = None,
) -> Path:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    if out_path is None:
        out_path = Path(str(fp32_path).replace("_fp32.onnx", "_fp16.onnx"))
    model = onnx.load(str(fp32_path))
    model_fp16 = convert_float_to_float16(model, keep_io_types=keep_io_types)
    onnx.save(model_fp16, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    label = "fp16 (keep_io=FP32)" if keep_io_types else "fp16 (full FP16 IO)"
    print(f"    {label} → {out_path.name}  ({size_mb:.1f} MB)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main export
# ─────────────────────────────────────────────────────────────────────────────
def _assert_span_architecture(model_path: str) -> dict:
    """
    Reads the checkpoint's config.json and rejects non-span architectures.

    gliner2 2.0.0 introduces `architecture: "boundary"` (GLiNER2.5). Those
    checkpoints have neither `span_rep` nor `count_embed`; they belong to the
    gliner25-rs project.
    """
    cfg: dict = {}
    local = Path(model_path) / "config.json"
    if local.exists():
        cfg = json.loads(local.read_text())
    else:
        try:
            from huggingface_hub import hf_hub_download

            cfg = json.loads(Path(hf_hub_download(model_path, "config.json")).read_text())
        except Exception as e:  # pragma: no cover - solo diagnostica
            print(f"WARN: config.json unreadable ({e}); proceeding without the guard")
            return cfg

    arch = str(cfg.get("architecture", "span")).lower()
    if arch not in ("span", ""):
        raise SystemExit(
            f"\nERROR: architecture '{arch}' is not supported by this exporter.\n"
            f"  {model_path} is not a span model.\n"
            f"  For GLiNER2.5 / BoundaryExtractor see github.com/dariofinardi/gliner25-rs\n"
        )
    return cfg


def export_span(
    model_path: str,
    out_dir: Path,
    max_count_override: int | None = None,
) -> None:
    global MAX_COUNT

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GLiNER2 span ONNX Fragment Exporter v3  -  IOBinding Ready")
    print("=" * 60)
    print(f"Model     : {model_path}")
    print(f"Output    : {out_dir}")
    print()

    cfg = _assert_span_architecture(model_path)
    if cfg:
        print(f"config    : counting_layer={cfg.get('counting_layer')}, "
              f"max_width={cfg.get('max_width')}, "
              f"token_pooling={cfg.get('token_pooling')}")

    print("Loading Extractor (gliner2 2.0.0)...")
    model = Extractor.from_pretrained(model_path)
    model.eval()

    H = model.encoder.config.hidden_size
    max_width = model.max_width

    # MAX_COUNT deve coincidere con il modulo di counting del checkpoint,
    # altrimenti struct_proj esce con una dim 0 incompatibile con il
    # pred_count prodotto da count_pred_argmax.
    resolved = getattr(model.count_embed, "max_count", MAX_COUNT)
    MAX_COUNT = int(max_count_override or resolved)
    if max_count_override and int(max_count_override) != int(resolved):
        print(f"WARN: --max_count={max_count_override} diverge da "
              f"count_embed.max_count={resolved}")

    print(f"hidden_size = {H},  max_width = {max_width},  MAX_COUNT = {MAX_COUNT}")
    print(f"count_embed = {type(model.count_embed).__name__}")
    print()

    # ── dummy shapes for tracing ────────────────────────────────────
    SEQ    = 32    # sub-token seq_len
    NWORDS = 20    # words in the text
    NSPANS = NWORDS * max_width
    M      = 5     # schema fields

    # ═════════════════════════════════════════════════════════════════════════
    # 1. ENCODER
    # ═════════════════════════════════════════════════════════════════════════
    print("─── 1. encoder ───")
    enc_fp32 = out_dir / "encoder_fp32.onnx"
    _export_fp32(
        EncoderWrapper(model.encoder),
        (
            torch.randint(0, 1000, (1, SEQ)),
            torch.ones((1, SEQ), dtype=torch.long),
        ),
        enc_fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids":         {0: "batch", 1: "seq_len"},
            "attention_mask":    {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
    )
    _convert_fp16(enc_fp32, keep_io_types=True,
                  out_path=out_dir / "encoder_fp16.onnx")
    _convert_fp16(enc_fp32, keep_io_types=False,
                  out_path=out_dir / "encoder_fp16_iobinding.onnx")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 2. TOKEN GATHER  [NUOVO]
    # ═════════════════════════════════════════════════════════════════════════
    print("─── 2. token_gather  [NUOVO] ───")
    tg_fp32 = out_dir / "token_gather_fp32.onnx"
    _export_fp32(
        TokenGatherWrapper(),
        (
            torch.randn(1, SEQ, H),
            torch.randint(0, SEQ, (NWORDS,)),
        ),
        tg_fp32,
        input_names=["last_hidden_state", "word_indices"],
        output_names=["text_embs"],
        dynamic_axes={
            "last_hidden_state": {0: "batch", 1: "seq_len"},
            "word_indices":      {0: "num_words"},
            "text_embs":         {0: "batch", 1: "num_words"},
        },
    )
    _convert_fp16(tg_fp32, keep_io_types=True,
                  out_path=out_dir / "token_gather_fp16.onnx")
    _convert_fp16(tg_fp32, keep_io_types=False,
                  out_path=out_dir / "token_gather_fp16_iobinding.onnx")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 3. SPAN REP
    # ═════════════════════════════════════════════════════════════════════════
    print("─── 3. span_rep ───")
    sr_fp32 = out_dir / "span_rep_fp32.onnx"
    dummy_spans = torch.zeros((1, NSPANS, 2), dtype=torch.long)
    _export_fp32(
        SpanRepWrapper(model.span_rep),
        (torch.randn(1, NWORDS, H), dummy_spans),
        sr_fp32,
        input_names=["hidden_states", "span_idx"],
        output_names=["span_embeddings"],
        dynamic_axes={
            "hidden_states":   {0: "batch", 1: "num_words"},
            "span_idx":        {0: "batch", 1: "num_spans"},
            "span_embeddings": {0: "batch", 1: "num_words"},
        },
    )
    _convert_fp16(sr_fp32, keep_io_types=True,
                  out_path=out_dir / "span_rep_fp16.onnx")
    _convert_fp16(sr_fp32, keep_io_types=False,
                  out_path=out_dir / "span_rep_fp16_iobinding.onnx")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 4. SCHEMA GATHER  [NUOVO]
    # ═════════════════════════════════════════════════════════════════════════
    print("─── 4. schema_gather  [NUOVO] ───")
    sg_fp32 = out_dir / "schema_gather_fp32.onnx"
    _export_fp32(
        SchemaGatherWrapper(),
        (
            torch.randn(1, SEQ, H),
            torch.randint(0, SEQ, (M + 1,)),  # M field_indices + 1 prompt_idx
        ),
        sg_fp32,
        input_names=["last_hidden_state", "schema_indices"],
        output_names=["pc_emb", "field_embs"],
        dynamic_axes={
            "last_hidden_state": {0: "batch", 1: "seq_len"},
            "schema_indices":    {0: "M_plus_1"},
            "field_embs":        {0: "num_fields"},
        },
    )
    _convert_fp16(sg_fp32, keep_io_types=True,
                  out_path=out_dir / "schema_gather_fp16.onnx")
    _convert_fp16(sg_fp32, keep_io_types=False,
                  out_path=out_dir / "schema_gather_fp16_iobinding.onnx")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 5. COUNT PRED ARGMAX  [MODIFICATO: fonde ArgMax]
    # ═════════════════════════════════════════════════════════════════════════
    print("─── 5. count_pred_argmax  [MODIFICATO] ───")
    cp_fp32 = out_dir / "count_pred_argmax_fp32.onnx"
    _export_fp32(
        CountPredArgmaxWrapper(model.count_pred),
        (torch.randn(1, H),),
        cp_fp32,
        input_names=["pc_emb"],
        output_names=["pred_count"],
        dynamic_axes={
            "pc_emb":     {0: "batch"},
            "pred_count": {0: "batch"},
        },
    )
    # pred_count is int64 in both variants; keep_io_types only affects pc_emb
    _convert_fp16(cp_fp32, keep_io_types=True,
                  out_path=out_dir / "count_pred_argmax_fp16.onnx")
    _convert_fp16(cp_fp32, keep_io_types=False,
                  out_path=out_dir / "count_pred_argmax_fp16_iobinding.onnx")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 6. COUNT LSTM FIXED  [MODIFICATO: output fisso MAX_COUNT, senza gold_count input]
    # ═════════════════════════════════════════════════════════════════════════
    print("─── 6. count_lstm_fixed  [MODIFICATO] ───")
    cl_fp32 = out_dir / "count_lstm_fixed_fp32.onnx"
    try:
        _export_fp32(
            CountLSTMFixedWrapper(model.count_embed, MAX_COUNT),
            (torch.randn(M, H),),
            cl_fp32,
            input_names=["field_embs"],
            output_names=["struct_proj"],
            dynamic_axes={
                "field_embs":  {0: "num_fields"},
                "struct_proj": {1: "num_fields"},  # dim 0 = MAX_COUNT (fisso)
            },
        )
        _convert_fp16(cl_fp32, keep_io_types=True,
                      out_path=out_dir / "count_lstm_fixed_fp16.onnx")
        _convert_fp16(cl_fp32, keep_io_types=False,
                      out_path=out_dir / "count_lstm_fixed_fp16_iobinding.onnx")
    except Exception as e:
        print(f"    WARN count_lstm_fixed export failed: {e}")
        print("    Falling back to count_lstm with an explicit gold_count_val (v1-compat)")
        _export_count_lstm_v1_compat(model, out_dir, H, M)
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 7. SCORER  [NUOVO: fonde Einsum + Sigmoid]
    # ═════════════════════════════════════════════════════════════════════════
    print("─── 7. scorer  [NUOVO] ───")
    sc_fp32 = out_dir / "scorer_fp32.onnx"
    _export_fp32(
        ScorerWrapper(),
        (
            torch.randn(1, NWORDS, max_width, H),
            torch.randn(MAX_COUNT, M, H),
        ),
        sc_fp32,
        input_names=["span_embeddings", "struct_proj"],
        output_names=["entity_scores"],
        dynamic_axes={
            "span_embeddings": {0: "batch", 1: "num_words"},
            "struct_proj":     {1: "num_fields"},  # dim 0 = MAX_COUNT (fisso)
            "entity_scores":   {1: "num_words", 3: "num_fields"},
        },
    )
    _convert_fp16(sc_fp32, keep_io_types=True,
                  out_path=out_dir / "scorer_fp16.onnx")
    _convert_fp16(sc_fp32, keep_io_types=False,
                  out_path=out_dir / "scorer_fp16_iobinding.onnx")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # 8. CLASSIFIER
    # ═════════════════════════════════════════════════════════════════════════
    print("--- 8. classifier ---")
    cls_fp32 = out_dir / "classifier_fp32.onnx"
    _export_fp32(
        ClassifierWrapper(model.classifier),
        (torch.randn(M, H),),
        cls_fp32,
        input_names=["field_embs"],
        output_names=["logits"],
        dynamic_axes={
            "field_embs": {0: "num_labels"},
            "logits":     {0: "num_labels"},
        },
    )
    _convert_fp16(cls_fp32, keep_io_types=True,
                  out_path=out_dir / "classifier_fp16.onnx")
    _convert_fp16(cls_fp32, keep_io_types=False,
                  out_path=out_dir / "classifier_fp16_iobinding.onnx")
    print()

    # ── copia tokenizer ─────────────────────────────────────────────
    _copy_tokenizer(model_path, out_dir)

    # ── summary ─────────────────────────────────────────────────────
    _print_summary(out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: v1-compatible count_lstm for GRUs that will not trace at MAX_COUNT
# ─────────────────────────────────────────────────────────────────────────────
def _export_count_lstm_v1_compat(
    model: Extractor, out_dir: Path, H: int, M: int
) -> None:
    """
    Exports count_lstm with gold_count_val as an explicit input (v1-compatible).
    Used as a fallback when CountLSTMFixedWrapper fails to trace.
    """

    class CountLSTMV1Compat(nn.Module):
        def __init__(self, count_embed):
            super().__init__()
            self.count_embed = count_embed

        def forward(self, field_embs, gold_count_val):
            return self.count_embed(field_embs, gold_count_val)

    out_path = out_dir / "count_lstm_fixed_fp32.onnx"
    dummy_count = torch.tensor(3, dtype=torch.int64)
    with torch.no_grad():
        torch.onnx.export(
            CountLSTMV1Compat(model.count_embed),
            (torch.randn(M, H), dummy_count),
            str(out_path),
            input_names=["field_embs", "gold_count_val"],
            output_names=["struct_proj"],
            dynamic_axes={
                "field_embs":  {0: "num_fields"},
                "struct_proj": {0: "count_val", 1: "num_fields"},
            },
            opset_version=17,
            dynamo=False,
        )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    FP32 fallback -> {out_path.name}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Copy the tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def _copy_tokenizer(model_path: str, out_dir: Path) -> None:
    import os

    if os.path.isdir(model_path):
        src = Path(model_path) / "tokenizer.json"
        if src.exists():
            shutil.copy(src, out_dir / "tokenizer.json")
            print(f"tokenizer.json copied from {model_path}")
            return

    # HuggingFace Hub: scarica il tokenizer
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_path, "tokenizer.json")
        shutil.copy(path, out_dir / "tokenizer.json")
        print(f"tokenizer.json downloaded from the HuggingFace Hub: {model_path}")
    except Exception as e:
        print(f"WARN: could not copy tokenizer.json: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def _print_summary(out_dir: Path) -> None:
    print("=" * 60)
    print("Span export v3 complete")
    print()
    print("Standard mode (fallback, every backend):")
    for f in sorted(out_dir.glob("*_fp32.onnx")):
        print(f"  {f.name}")
    print()
    print("Standard mode FP16 (keep_io_types=True, CoreML):")
    for f in sorted(out_dir.glob("*_fp16.onnx")):
        print(f"  {f.name}")
    print()
    print("IOBinding mode FP16 (full FP16 IO, CUDA / QNN):")
    for f in sorted(out_dir.glob("*_fp16_iobinding.onnx")):
        print(f"  {f.name}")
    print()
    print("Rust IOBinding pipeline:")
    print("  encoder_fp16_iobinding.onnx")
    print("  ├─ token_gather_fp16_iobinding.onnx")
    print("  │    └─ span_rep_fp16_iobinding.onnx")
    print("  └─ schema_gather_fp16_iobinding.onnx")
    print("       ├─ count_pred_argmax_fp16_iobinding.onnx  (→ pred_count int64)")
    print("       └─ count_lstm_fixed_fp16_iobinding.onnx")
    print("            └─ scorer_fp16_iobinding.onnx  (→ entity_scores)")
    print("  classifier_fp16_iobinding.onnx  (classification tasks only)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GLiNER2 span ONNX Fragment Exporter v3 - IOBinding Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model_path",
        required=True,
        help="Local path or HuggingFace repo id "
             "(e.g. fastino/gliner2-privacy-filter-PII-multi)",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for the ONNX fragments",
    )
    p.add_argument(
        "--max_count",
        type=int,
        default=None,
        help="Override MAX_COUNT (default: model.count_embed.max_count)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_span(
        model_path=args.model_path,
        out_dir=Path(args.out_dir),
        max_count_override=args.max_count,
    )
