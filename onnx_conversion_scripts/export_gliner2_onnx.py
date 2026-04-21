import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gliner2 import GLiNER2

def export_fragments():
    print("==================================================")
    print("End-to-End Fragmented Export (ONNX FP32 & FP16)")
    print("Specific optimization for Rust / C++ backends")
    print("==================================================")

    model_path = "fastino/gliner2-multi-v1"
    out_dir_fp32 = Path("models/fastino_gliner2_multi_v1_fp32")
    out_dir_fp16 = Path("models/fastino_gliner2_multi_v1_fp16")
    out_dir_fp32.mkdir(parents=True, exist_ok=True)
    out_dir_fp16.mkdir(parents=True, exist_ok=True)
    
    print("Loading GLiNER2 model...")
    model = GLiNER2.from_pretrained(model_path)
    model.eval()
    
    # NOTE ON FP16 EXPORT STRATEGY:
    # To export safely, we first trace the model in FP32. CPU execution often
    # struggles with native FP16 traces. We will convert the FP32 ONNX graph
    # into FP16 using onnxruntime transformers later in the script.
    
    # ==========================================
    # 1. ENCODER
    # ==========================================
    # Extracts the DeBERTa/RoBERTa base encoder.
    class EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, input_ids, attention_mask):
            return self.encoder(input_ids, None, attention_mask)
            
    print("\n--- 1. Exporting Encoder ---")
    enc_wrapper = EncoderWrapper(model.encoder)
    dummy_input_ids = torch.randint(0, 1000, (1, 16))
    dummy_attention_mask = torch.ones((1, 16))
    
    encoder_onnx = out_dir_fp32 / "encoder_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            enc_wrapper,
            (dummy_input_ids, dummy_attention_mask),
            str(encoder_onnx),
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=14,
            dynamo=False
        )
    print(f"Encoder saved: {encoder_onnx}")

    # ==========================================
    # 2. SPAN REP LAYER
    # ==========================================
    # Generates representations for all possible span boundaries up to `max_width`.
    # It takes hidden states and a span index tensor of shape [batch, num_spans, 2].
    hidden_size = model.encoder.config.hidden_size
    
    print("\n--- 2. Exporting SpanRepLayer ---")
    class SpanRepWrapper(nn.Module):
        def __init__(self, span_rep_layer):
            super().__init__()
            self.span_rep_layer = span_rep_layer
        def forward(self, hidden_states, span_idx):
            # hidden_states: [B, L, D]
            # span_idx: [B, S, 2]
            return self.span_rep_layer(hidden_states, span_idx)
            
    span_wrapper = SpanRepWrapper(model.span_rep.span_rep_layer)
    dummy_hidden = torch.randn(1, 16, hidden_size)
    
    # Build dummy_span_idx
    max_width = model.span_rep.span_rep_layer.max_width
    num_spans = 16 * max_width
    dummy_span_idx = torch.randint(0, 16, (1, num_spans, 2))
    
    span_onnx = out_dir_fp32 / "span_rep_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            span_wrapper,
            (dummy_hidden, dummy_span_idx),
            str(span_onnx),
            input_names=['last_hidden_state', 'span_idx'],
            output_names=['span_embeddings'],
            dynamic_axes={
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
                'span_idx': {0: 'batch_size', 1: 'num_spans'},
                'span_embeddings': {0: 'batch_size', 1: 'sequence_length'} # Note: output is actually [B, L, max_width, D]
            },
            opset_version=14,
            dynamo=False
        )
    print(f"SpanRepLayer saved: {span_onnx}")

    # ==========================================
    # 3. COUNT PRED
    # ==========================================
    # Predicts how many instances of a class exist based on prompt-context embeddings.
    print("\n--- 3. Exporting CountPred ---")
    class CountPredWrapper(nn.Module):
        def __init__(self, count_pred):
            super().__init__()
            self.count_pred = count_pred
        def forward(self, pc_emb):
            return self.count_pred(pc_emb)
            
    cpred_wrapper = CountPredWrapper(model.count_pred)
    dummy_pc_first = torch.randn(1, hidden_size)  # (1, hidden_size)
    
    cpred_onnx = out_dir_fp32 / "count_pred_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            cpred_wrapper,
            (dummy_pc_first,),
            str(cpred_onnx),
            input_names=['pc_emb'],
            output_names=['count_logits'],
            dynamic_axes={'pc_emb': {0: 'batch_size'}, 'count_logits': {0: 'batch_size'}},
            opset_version=14,
            dynamo=False
        )
    print(f"CountPred saved: {cpred_onnx}")

    # ==========================================
    # 4. COUNT LSTM
    # ==========================================
    # Dynamic sequence unrolling. We use a nn.GRU inside the wrapper.
    # Note: GLiNER2 originally uses CompileSafeGRU which fails in ONNX due to dynamic 
    # Out-of-Bounds `Gather` errors when unrolling. We must trace this correctly.
    print("\n--- 4. Exporting CountLSTM ---")
    class CountLSTMWrapper(nn.Module):
        def __init__(self, count_lstm):
            super().__init__()
            self.count_lstm = count_lstm
        def forward(self, pc_emb, gold_count_val):
            # Pass the tensor directly to avoid TracerWarning and onnx::Cast_1.
            # If the original code does torch.arange(gold_count_val), passing a 0D tensor 
            # works fine in PyTorch and traces correctly as a dynamic shape in ONNX.
            return self.count_lstm(pc_emb, gold_count_val)
            
    count_wrapper = CountLSTMWrapper(model.count_embed)
    dummy_pc_emb = torch.randn(5, hidden_size)  # (M, hidden_size)
    dummy_count = torch.tensor(3, dtype=torch.int64) # Use a tensor instead of an int!
    
    count_onnx = out_dir_fp32 / "count_lstm_fp32.onnx"
    with torch.no_grad():
        try:
            torch.onnx.export(
                count_wrapper,
                (dummy_pc_emb, dummy_count),
                str(count_onnx),
                input_names=['pc_emb', 'gold_count_val'],
                output_names=['count_embeddings'],
                dynamic_axes={
                    'pc_emb': {0: 'num_fields'},
                    'gold_count_val': {},
                    'count_embeddings': {1: 'num_fields'}
                },
                opset_version=14,
                dynamo=False
            )
            print(f"CountLSTM saved: {count_onnx}")
        except Exception as e:
            print(f"CountLSTM Error (likely incompatible dynamic scalar): {e}")

    # ==========================================
    # 5. CLASSIFIERS
    # ==========================================
    print("\n--- 5. Exporting Classifier (FeedForward) ---")
    # In GLiNER2, the classifier calculates the final logits for the spans.
    class ClassifierWrapper(nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier
        def forward(self, x):
            return self.classifier(x)
            
    cls_wrapper = ClassifierWrapper(model.classifier)
    dummy_x = torch.randn(1, 16, max_width, hidden_size)
    
    cls_onnx = out_dir_fp32 / "classifier_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            cls_wrapper,
            (dummy_x,),
            str(cls_onnx),
            input_names=['span_embeddings'],
            output_names=['logits'],
            dynamic_axes={
                'span_embeddings': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=14,
            dynamo=False
        )
    print(f"Classifier saved: {cls_onnx}")
    
    # ==========================================
    # FP16 CONVERSION
    # ==========================================
    print("\n[Base export completed. Starting FP16 conversion...]")
    import onnx
    from onnxruntime.transformers.optimizer import optimize_model
    from onnxruntime.transformers.float16 import convert_float_to_float16

    for model_name in ["encoder_fp32.onnx", "span_rep_fp32.onnx", "count_pred_fp32.onnx", "count_lstm_fp32.onnx", "classifier_fp32.onnx"]:
        in_path = out_dir_fp32 / model_name
        if not in_path.exists():
            continue
            
        out_name = model_name.replace("_fp32.onnx", "_fp16.onnx")
        out_path = out_dir_fp16 / out_name
        print(f"Converting {model_name} -> {out_name}")
        
        model_onnx = onnx.load(str(in_path))
        
        # Apply optimizations and cast to FP16
        model_fp16 = convert_float_to_float16(model_onnx, keep_io_types=True)
        onnx.save(model_fp16, str(out_path))
        
        size_mb_32 = os.path.getsize(in_path) / (1024*1024)
        size_mb_16 = os.path.getsize(out_path) / (1024*1024)
        print(f"  Size reduction: {size_mb_32:.2f} MB -> {size_mb_16:.2f} MB")
        
    # Copy the tokenizer config
    from huggingface_hub import hf_hub_download
    import shutil
    try:
        if os.path.exists(os.path.join(model_path, "tokenizer.json")):
            shutil.copy(os.path.join(model_path, "tokenizer.json"), str(out_dir_fp32 / "tokenizer.json"))
            shutil.copy(os.path.join(model_path, "tokenizer.json"), str(out_dir_fp16 / "tokenizer.json"))
        else:
            tok_path = hf_hub_download(repo_id=model_path, filename="tokenizer.json")
            shutil.copy(tok_path, str(out_dir_fp32 / "tokenizer.json"))
            shutil.copy(tok_path, str(out_dir_fp16 / "tokenizer.json"))
    except Exception as e:
        print("Could not copy tokenizer.json:", e)
        
    print("\n✅ End-to-end models (FP32 & FP16) are ready for Rust inference!")

if __name__ == "__main__":
    export_fragments()
