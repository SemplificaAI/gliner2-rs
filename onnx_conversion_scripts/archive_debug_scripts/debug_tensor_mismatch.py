#!/usr/bin/env python3
"""
Debug tensor shape mismatch between word-level spans and ONNX model expectations
"""

def analyze_tensor_mismatch():
    """Analyze the tensor shape mismatch issue"""
    print("TENSOR SHAPE MISMATCH ANALYSIS")
    print("=" * 50)
    
    print("PROBLEM:")
    print("- Word-level spans generate different number of spans")
    print("- ONNX model expects specific tensor shape")
    print("- Span embedding access fails due to shape mismatch")
    print()
    
    text = "Apple Inc. announced its quarterly earnings report"
    words = text.split()
    
    print("CHARACTER-LEVEL SPANS (Original):")
    text_len = len(text)
    max_width = 12
    char_spans = text_len * max_width
    print(f"  Text length: {text_len}")
    print(f"  Max width: {max_width}")
    print(f"  Character spans: {char_spans}")
    print(f"  Expected tensor shape: [1, {char_spans}, hidden]")
    print()
    
    print("WORD-LEVEL SPANS (Current Rust):")
    max_span_words = 3
    word_spans = 0
    for i in range(len(words)):
        for width in range(1, max_span_words + 1):
            if i + width <= len(words):
                word_spans += 1
    print(f"  Word count: {len(words)}")
    print(f"  Max span words: {max_span_words}")
    print(f"  Word spans: {word_spans}")
    print(f"  Actual tensor shape: [1, {word_spans}, hidden]")
    print()
    
    print("MISMATCH:")
    print(f"  Expected: {char_spans} spans")
    print(f"  Actual: {word_spans} spans")
    print(f"  Difference: {char_spans - word_spans} spans")
    print()
    
    print("CONSEQUENCE:")
    print("- Span embeddings tensor has wrong shape")
    print("- Indexing fails or returns wrong embeddings")
    print("- Model gets incorrect or missing span representations")

def analyze_solution_approaches():
    """Analyze possible solutions"""
    print("\n" + "=" * 50)
    print("SOLUTION APPROACHES")
    print("=" * 50)
    
    print("OPTION 1: Pad word-level spans to expected size")
    print("- Generate word-level spans")
    print("- Pad with dummy spans to match expected count")
    print("- Risk: Dummy spans might confuse the model")
    print()
    
    print("OPTION 2: Use character-level spans with word boundary filtering")
    print("- Generate all character-level spans")
    print("- Filter to only word boundary aligned spans")
    print("- Keep original tensor structure")
    print()
    
    print("OPTION 3: Modify model architecture")
    print("- Convert ONNX model to accept variable span count")
    print("- Complex and requires model retraining")
    print()
    
    print("OPTION 4: Hybrid approach")
    print("- Use character-level generation for compatibility")
    print("- Apply word boundary logic in post-processing")
    print("- Merge adjacent character spans")

def recommend_solution():
    """Recommend the best solution"""
    print("\n" + "=" * 50)
    print("RECOMMENDED SOLUTION")
    print("=" * 50)
    
    print("OPTION 2: Character-level spans with word boundary filtering")
    print()
    print("REASONS:")
    print("1. Maintains ONNX model compatibility")
    print("2. Preserves expected tensor shapes")
    print("3. Allows word boundary logic implementation")
    print("4. Minimal changes to existing code")
    print()
    
    print("IMPLEMENTATION:")
    print("1. Generate all character-level spans (original)")
    print("2. Filter spans to word boundaries during processing")
    print("3. Keep only spans that align with word boundaries")
    print("4. Extract entities using word-aware logic")

def main():
    analyze_tensor_mismatch()
    analyze_solution_approaches()
    recommend_solution()

if __name__ == "__main__":
    main()
