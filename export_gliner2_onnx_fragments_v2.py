"""
GLiNER2 ONNX Fragment Exporter v2  –  IOBinding-Ready
======================================================
Esporta 8 frammenti ONNX dove le trasformazioni inter-layer
(Gather, ArgMax, Einsum+Sigmoid) sono fuse direttamente nei grafi,
abilitando il flusso Zero-Copy IOBinding su tutti i backend hardware.

I modelli v1 (export_gliner2_onnx_fragments.py) restano inalterati.
Questo script crea una directory parallela con suffisso _iobinding.

Compatibilità target:
    ┌──────────────────────────────────────────────────────────────┐
    │ Backend          │ Variante file         │ Note               │
    ├──────────────────┼───────────────────────┼────────────────────┤
    │ Qualcomm QNN NPU │ _fp16_iobinding.onnx  │ X Elite HTP/HTA;   │
    │ (X Elite, 8cx…)  │                       │ Gather/ArgMax/MM   │
    │                  │                       │ offloadati su HTP  │
    ├──────────────────┼───────────────────────┼────────────────────┤
    │ Apple CoreML     │ _fp16.onnx            │ Neural Engine + GPU│
    │ (macOS, M-series)│ (keep_io_types=True)  │ tramite CoreML EP; │
    │                  │                       │ FP32 IO richiesto  │
    ├──────────────────┼───────────────────────┼────────────────────┤
    │ NVIDIA CUDA      │ _fp16_iobinding.onnx  │ IOBinding VRAM     │
    ├──────────────────┼───────────────────────┼────────────────────┤
    │ AMD ROCm         │ _fp16_iobinding.onnx  │ Come CUDA; ROCm    │
    │                  │                       │ stack su Linux     │
    ├──────────────────┼───────────────────────┼────────────────────┤
    │ Intel OpenVINO   │ _fp32.onnx            │ CPU/GPU/VPU Intel  │
    ├──────────────────┼───────────────────────┼────────────────────┤
    │ CPU (XNNPACK/    │ _fp32.onnx            │ fallback universale│
    │ generic)         │                       │                    │
    └──────────────────┴───────────────────────┴────────────────────┘

    Scelte tecniche per la compatibilità multi-EP:
    - opset 17  (ONNX 1.14+, supportato da tutti gli EP moderni)
    - MatMul+Reshape+Transpose invece di Einsum nel Scorer
      (Einsum ONNX non è supportato da tutti i backend QNN/CoreML)
    - Gather (axis=1) per token_gather/schema_gather: op base,
      supportato ovunque incluso Qualcomm HTP
    - ArgMax fuso in count_pred_argmax: op nativa ONNX, offloadabile
    - count_lstm_fixed: GRU unrollato a 20 step fissi durante il
      tracing → diventa sequenza di Linear + GRU step, compatibile
      con CoreML e QNN che non supportano cicli dinamici
    - FP16 keep_io_types=True per CoreML (richiede FP32 agli IO)
    - FP16 keep_io_types=False per QNN/CUDA/ROCm IOBinding

Pipeline v2 (IOBinding chain):

    encoder(input_ids, attention_mask)
        → last_hidden_state [VRAM]
        │
        ├─ token_gather(last_hidden_state, word_indices)
        │       → text_embs [VRAM]
        │       └─ span_rep(text_embs, span_idx)
        │               → span_embeddings [VRAM]
        │
        └─ schema_gather(last_hidden_state, schema_indices)
                → pc_emb [VRAM], field_embs [VRAM]
                ├─ count_pred_argmax(pc_emb)
                │       → pred_count  [int64 scalar, 8 byte GPU→CPU]
                └─ count_lstm_fixed(field_embs)
                        → struct_proj [VRAM, shape sempre (MAX_COUNT, M, H)]
                        └─ scorer(span_embeddings, struct_proj)
                                → entity_scores [VRAM→CPU per soglia]

    classifier(span_embeddings_cls)  ← solo per task di classificazione
        → logits [VRAM→CPU per softmax]

Fallback Standard (v1) rimane invariato nel codice Rust.

Uso:
    python export_gliner2_onnx_fragments_v2.py \\
        --model_path /path/to/checkpoint_or_hf_id \\
        --out_dir models/mio_modello_iobinding

    # Modello pubblico ufficiale (scarica da HF):
    python export_gliner2_onnx_fragments_v2.py \\
        --model_path SemplificaAI/gliner2-multi-v1 \\
        --out_dir models/fastino_gliner2_multi_v1_fp16_iobinding
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

import torch
import torch.nn as nn

# ─── path setup: usa il modulo GLiNER2 del progetto senza modificarlo ────────
_SCRIPT_DIR = Path(__file__).parent
_GLINER2_REF = (
    _SCRIPT_DIR
    / "rust_inference"
    / "reference_implementations"
    / "GLiNER2"
)
if str(_GLINER2_REF) not in sys.path:
    sys.path.insert(0, str(_GLINER2_REF))

from gliner2 import Extractor  # noqa: E402  (import dopo sys.path)

MAX_COUNT: int = 20  # deve coincidere con CountLSTM.max_count


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 1 – Encoder  (identico al v1, output rinominato per chiarezza)
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
# Wrapper 2 – TokenGather  [NUOVO]
# Estrae l'embedding del primo sub-token per ogni parola (word-level pooling).
# ─────────────────────────────────────────────────────────────────────────────
class TokenGatherWrapper(nn.Module):
    """
    Input:
        hidden_state  [1, seq_len, H]  – output dell'encoder
        word_indices  [num_words]      – indice del primo sub-token per parola
                                         (word_to_token_maps[:,0] dalla pipeline Rust)
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
# Wrapper 3 – SpanRep  (identico al v1)
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
# Wrapper 4 – SchemaGather  [NUOVO]
# Gather unificato per prompt + field embeddings da un unico indice.
# ─────────────────────────────────────────────────────────────────────────────
class SchemaGatherWrapper(nn.Module):
    """
    Input:
        hidden_state    [1, seq_len, H]
        schema_indices  [M+1]  int64
                        schema_indices[0]  = prompt_tok_idx
                        schema_indices[1:] = field_tok_indices

    Output:
        pc_emb      [1, H]  – embedding del token [P] (prompt)
        field_embs  [M, H]  – embedding dei campi/label
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
# Wrapper 5 – CountPredArgmax  [MODIFICATO: fonde ArgMax]
# ─────────────────────────────────────────────────────────────────────────────
class CountPredArgmaxWrapper(nn.Module):
    """
    Input:   pc_emb  [1, H]
    Output:  pred_count  [1]  int64  (argmax della distribuzione di count)

    L'output int64 resta int64 anche dopo la conversione FP16.
    La dimensione [1] è mantenuta invece dello scalare 0-D per
    semplificare il binding in Rust (sempre un Tensor, mai un Value scalare).
    """

    def __init__(self, count_pred: nn.Module):
        super().__init__()
        self.count_pred = count_pred

    def forward(self, pc_emb: torch.Tensor) -> torch.Tensor:  # [1]  int64
        logits = self.count_pred(pc_emb)               # [1, max_count]
        return torch.argmax(logits, dim=-1)             # [1]  int64


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 6 – CountLSTMFixed  [MODIFICATO: output sempre (MAX_COUNT, M, H)]
# ─────────────────────────────────────────────────────────────────────────────
class CountLSTMFixedWrapper(nn.Module):
    """
    Esegue sempre MAX_COUNT step GRU invece di 'pred_count' step.
    L'output ha shape FISSA [MAX_COUNT, M, H] – abilitante per IOBinding.

    Correttezza: il GRU è causale → output[i] dipende solo da output[0..i-1].
    Quindi slice [:pred_count] dell'output è identica a eseguire con
    gold_count_val=pred_count. Il codice Rust usa pred_count per ignorare
    le righe extra.

    Funziona con CountLSTM, CountLSTMv2 e CountLSTMoE senza modificare
    nessuno di questi moduli: basta chiamare forward con gold_count_val=MAX_COUNT.
    Durante il tracing ONNX, gold_count_val=MAX_COUNT è una costante Python
    → torch.arange(MAX_COUNT) diventa un nodo Constant nel grafo ONNX.
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
# Wrapper 7 – Scorer  [NUOVO: fonde Einsum + Sigmoid]
# ─────────────────────────────────────────────────────────────────────────────
class ScorerWrapper(nn.Module):
    """
    Calcola la probabilità sigmoid di ogni span per ogni schema-field,
    iterando su tutti i MAX_COUNT slot di entity.

    scores[c, s, w, m] = sigmoid( Σ_d span[s,w,d] * struct[c,m,d] )

    Usa Reshape + MatMul + Transpose invece di Einsum per
    massima compatibilità con tutti gli EP (CoreML, QNN, XNNPACK).

    Input:
        span_embeddings  [1, num_words, max_width, H]
        struct_proj      [MAX_COUNT, M, H]

    Output:
        entity_scores  [MAX_COUNT, num_words, max_width, M]  float32  ∈ [0,1]

    Il codice Rust:
        1. Legge pred_count (int64) dalla count_pred_argmax session
        2. Usa entity_scores[:pred_count] come scores effettivi
        3. Applica soglia (default 0.15) e NMS
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
# Wrapper 8 – Classifier  (identico al v1)
# ─────────────────────────────────────────────────────────────────────────────
class ClassifierWrapper(nn.Module):
    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(
        self,
        span_embeddings: torch.Tensor,  # [1, num_labels, max_width, H]
    ) -> torch.Tensor:                   # [1, num_labels, max_width, 1]
        return self.classifier(span_embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: export FP32 + converti in FP16
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
# Export principale
# ─────────────────────────────────────────────────────────────────────────────
def export_v2(model_path: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GLiNER2 ONNX Fragment Exporter v2  –  IOBinding Ready")
    print("=" * 60)
    print(f"Modello   : {model_path}")
    print(f"Output    : {out_dir}")
    print()

    # ── carica modello ────────────────────────────────────────────────────────
    print("Caricamento Extractor...")
    model = Extractor.from_pretrained(model_path)
    model.eval()

    H = model.encoder.config.hidden_size
    max_width = model.max_width
    print(f"hidden_size = {H},  max_width = {max_width},  MAX_COUNT = {MAX_COUNT}")
    print()

    # ── dimensioni dummy per il tracing ───────────────────────────────────────
    SEQ    = 32    # seq_len sub-token
    NWORDS = 20    # parole nel testo
    NSPANS = NWORDS * max_width
    M      = 5     # campi schema

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
    # pred_count è int64 in entrambe le varianti; keep_io_types influisce solo su pc_emb
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
        print(f"    WARN count_lstm_fixed export fallito: {e}")
        print("    Fallback: esporto count_lstm con gold_count_val esplicito (v1-compat)")
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
    print("─── 8. classifier ───")
    cls_fp32 = out_dir / "classifier_fp32.onnx"
    _export_fp32(
        ClassifierWrapper(model.classifier),
        (torch.randn(1, M, max_width, H),),
        cls_fp32,
        input_names=["span_embeddings"],
        output_names=["logits"],
        dynamic_axes={
            "span_embeddings": {0: "batch", 1: "num_labels"},
            "logits":          {0: "batch", 1: "num_labels"},
        },
    )
    _convert_fp16(cls_fp32, keep_io_types=True,
                  out_path=out_dir / "classifier_fp16.onnx")
    _convert_fp16(cls_fp32, keep_io_types=False,
                  out_path=out_dir / "classifier_fp16_iobinding.onnx")
    print()

    # ── copia tokenizer ──────────────────────────────────────────────────────
    _copy_tokenizer(model_path, out_dir)

    # ── summary ──────────────────────────────────────────────────────────────
    _print_summary(out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: count_lstm v1-compat per modelli con GRU non tracciabile a MAX_COUNT
# ─────────────────────────────────────────────────────────────────────────────
def _export_count_lstm_v1_compat(
    model: Extractor, out_dir: Path, H: int, M: int
) -> None:
    """
    Esporta count_lstm con gold_count_val come input esplicito (compatibile v1).
    Usato come fallback se CountLSTMFixedWrapper non riesce a tracciare.
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
    print(f"    FP32 fallback → {out_path.name}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Copia tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def _copy_tokenizer(model_path: str, out_dir: Path) -> None:
    import os

    if os.path.isdir(model_path):
        src = Path(model_path) / "tokenizer.json"
        if src.exists():
            shutil.copy(src, out_dir / "tokenizer.json")
            print(f"tokenizer.json copiato da {model_path}")
            return

    # HuggingFace Hub: scarica il tokenizer
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_path, "tokenizer.json")
        shutil.copy(path, out_dir / "tokenizer.json")
        print(f"tokenizer.json scaricato da HuggingFace Hub: {model_path}")
    except Exception as e:
        print(f"WARN: impossibile copiare tokenizer.json: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def _print_summary(out_dir: Path) -> None:
    print("=" * 60)
    print("✅ Export v2 completato")
    print()
    print("File per Standard mode (fallback, tutti i backend):")
    for f in sorted(out_dir.glob("*_fp32.onnx")):
        print(f"  {f.name}")
    print()
    print("File per Standard mode FP16 (keep_io_types=True, CoreML):")
    for f in sorted(out_dir.glob("*_fp16.onnx")):
        print(f"  {f.name}")
    print()
    print("File per IOBinding mode FP16 (full FP16 IO, CUDA / QNN):")
    for f in sorted(out_dir.glob("*_fp16_iobinding.onnx")):
        print(f"  {f.name}")
    print()
    print("Pipeline IOBinding Rust:")
    print("  encoder_fp16_iobinding.onnx")
    print("  ├─ token_gather_fp16_iobinding.onnx")
    print("  │    └─ span_rep_fp16_iobinding.onnx")
    print("  └─ schema_gather_fp16_iobinding.onnx")
    print("       ├─ count_pred_argmax_fp16_iobinding.onnx  (→ pred_count int64)")
    print("       └─ count_lstm_fixed_fp16_iobinding.onnx")
    print("            └─ scorer_fp16_iobinding.onnx  (→ entity_scores)")
    print("  classifier_fp16_iobinding.onnx  (solo task 'classifications')")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GLiNER2 ONNX Fragment Exporter v2 – IOBinding Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model_path",
        default=(
            "/mnt/crucial/jugaad/experiments/edito-gliner2"
            "/finetuning_local/scripts/training/experiments"
            "/run_20260414_191732/best"
        ),
        help="Path locale al checkpoint oppure HuggingFace repo ID",
    )
    p.add_argument(
        "--out_dir",
        default="models/semplifica_gliner2_multi_v1_fp16_iobinding",
        help="Directory di output per i modelli v2",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_v2(
        model_path=args.model_path,
        out_dir=Path(args.out_dir),
    )
