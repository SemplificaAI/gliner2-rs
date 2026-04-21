#!/usr/bin/env python3
"""
Debug span embedding access issues in Rust implementation
"""

def analyze_span_embedding_access():
    """Analyze how span embeddings should be accessed"""
    print("SPAN EMBEDDING ACCESS ANALYSIS")
    print("=" * 50)
    
    print("PROBLEM:")
    print("Word-level spans work but produce no entities")
    print("Only French produces 2 incorrect entities")
    print()
    
    print("ISSUE ANALYSIS:")
    print("1. Span generation: Working (word-level)")
    print("2. Span embedding access: BROKEN")
    print("3. Tensor indexing: MISMATCH")
    print()
    
    print("RUST CURRENT CODE:")
    print("let span_val = if span_emb_shape.len() == 4 {")
    print("    span_embeddings[[0, span_idx, 0, d]]")
    print("} else {")
    print("    span_embeddings[[0, span_idx, d]]")
    print("};")
    print()
    
    print("PROBLEM:")
    print("- span_idx is word span index (0, 1, 2, ...)")
    print("- But span_embeddings expects character-level indexing")
    print("- Shape mismatch causes wrong embeddings")
    print()
    
    print("SOLUTION:")
    print("Need to map word spans to correct embedding indices")
    print("OR use character-level spans with proper word boundary detection")
    
    print()
    print("ALTERNATIVE APPROACH:")
    print("Keep character-level spans but fix word boundary mapping")
    print("This might be more compatible with the ONNX model")

def analyze_span_shapes():
    """Analyze expected span tensor shapes"""
    print("\n" + "=" * 50)
    print("SPAN TENSOR SHAPE ANALYSIS")
    print("=" * 50)
    
    print("ONNX MODEL EXPECTS:")
    print("- Character-level span indices")
    print("- Shape: [batch, num_char_spans, hidden]")
    print("- Max spans: text_len * max_width")
    print()
    
    print("RUST WORD-LEVEL APPROACH:")
    print("- Word-level span indices")
    print("- Shape: [batch, num_word_spans, hidden]")
    print("- Max spans: num_words * max_span_words")
    print()
    
    print("MISMATCH:")
    print("ONNX model trained for character-level spans")
    print("Rust using word-level spans")
    print("Embedding lookup fails")

def main():
    analyze_span_embedding_access()
    analyze_span_shapes()
    
    print("\n" + "=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)
    print("1. Revert to character-level spans")
    print("2. Fix word boundary detection logic")
    print("3. Improve span scoring to prefer longer spans")
    print("4. Add post-processing to merge adjacent spans")

if __name__ == "__main__":
    main()
