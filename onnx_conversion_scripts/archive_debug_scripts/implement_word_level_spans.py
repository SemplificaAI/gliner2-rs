#!/usr/bin/env python3
"""
Implement word-level spans like PyTorch GLiNER2 markerV0 mode
"""

def analyze_markerV0_span_generation():
    """Analyze how markerV0 span generation should work"""
    print("MARKERV0 SPAN GENERATION ANALYSIS")
    print("=" * 50)
    
    print("PYTORCH GLiNER2 CONFIG:")
    print("- span_mode: markerV0")
    print("- max_width: 12")
    print("- Word-level spans")
    print()
    
    print("MARKERV0 MODE:")
    print("- Uses word boundaries")
    print("- Generates spans from word combinations")
    print("- Up to max_width words per span")
    print()
    
    text = "Apple Inc. announced its quarterly earnings report"
    words = text.split()
    
    print("EXPECTED WORD-LEVEL SPANS:")
    for start_word in range(len(words)):
        for width in range(1, min(4, len(words) - start_word + 1)):  # Up to 3 words
            end_word = start_word + width
            span_words = words[start_word:end_word]
            span_text = ' '.join(span_words)
            print(f"  Words {start_word}-{end_word} (width {width}): '{span_text}'")
    
    print("\nCHARACTER POSITIONS FOR WORD SPANS:")
    char_pos = 0
    word_positions = []
    for i, word in enumerate(words):
        word_start = text.find(word, char_pos)
        word_end = word_start + len(word)
        word_positions.append((word_start, word_end))
        print(f"  Word {i} '{word}': {word_start}-{word_end}")
        char_pos = word_end + 1
    
    print("\nSPAN TO CHARACTER MAPPING:")
    for start_word in range(len(words)):
        for width in range(1, min(4, len(words) - start_word + 1)):
            end_word = start_word + width
            start_char = word_positions[start_word][0]
            end_char = word_positions[end_word - 1][1]
            span_words = words[start_word:end_word]
            span_text = ' '.join(span_words)
            print(f"  Words {start_word}-{end_word} -> Chars {start_char}:{end_char} -> '{span_text}'")

def design_rust_word_level_spans():
    """Design the Rust implementation for word-level spans"""
    print("\n" + "=" * 50)
    print("RUST WORD-LEVEL SPAN DESIGN")
    print("=" * 50)
    
    print("RUST IMPLEMENTATION PLAN:")
    print("1. Split text into words")
    print("2. Find character positions for each word")
    print("3. Generate word-level spans (up to max_width words)")
    print("4. Map word spans to character positions")
    print("5. Use character positions for ONNX model")
    print("6. Extract entity text using character boundaries")
    print()
    
    print("PSEUDOCODE:")
    print("""
    words = text.split_whitespace()
    word_positions = []
    char_pos = 0
    for word in words:
        start = text.find(word, char_pos)
        end = start + word.len()
        word_positions.push((start, end))
        char_pos = end + 1
    
    spans = []
    for start_word in 0..words.len():
        for width in 1..=max_width:
            end_word = min(start_word + width, words.len())
            if end_word > start_word {
                start_char = word_positions[start_word].0
                end_char = word_positions[end_word - 1].1
                spans.push((start_char, end_char))
    """)

def main():
    analyze_markerV0_span_generation()
    design_rust_word_level_spans()

if __name__ == "__main__":
    main()
