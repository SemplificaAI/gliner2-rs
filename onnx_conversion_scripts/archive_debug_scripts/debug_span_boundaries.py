#!/usr/bin/env python3
"""
Debug exact span boundaries for 'Apple Inc' case to fix Rust partial extraction
"""

import json

def debug_span_boundaries():
    """Debug why Rust extracts 'Apple' instead of 'Apple Inc'"""
    print("SPAN BOUNDARIES DEBUG - Apple Inc Case")
    print("=" * 50)
    
    # Load test data
    with open('test_sentences.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    english_text = test_data[0]['text']
    
    print(f"TEXT: '{english_text}'")
    print(f"LENGTH: {len(english_text)} characters")
    
    # Python results (complete entities)
    python_entities = [
        {'text': 'Apple Inc', 'label': 'organization_name', 'position': (0, 9)},
        {'text': 'Tim Cook', 'label': 'person_name', 'position': (113, 121)},
        {'text': 'investor.relations@apple.com', 'label': 'email', 'position': (262, 290)}
    ]
    
    print(f"\nPYTHON COMPLETE ENTITIES:")
    for entity in python_entities:
        start, end = entity['position']
        print(f"  '{entity['text']}' -> {start}-{end}")
        print(f"    Text slice: '{english_text[start:end]}'")
        print(f"    Context: '{english_text[max(0,start-5):min(end+5,len(english_text))]}...'")
    
    # Rust debug output we saw
    print(f"\nRUST DEBUG OUTPUT (PARTIAL):")
    print("  DEBUG: Span mapping - char 0:1 -> word 0:1")
    print("  DEBUG: Token boundaries - 0:1 -> tokens 46:47")
    print("  DEBUG: Decoded entity: 'Apple' (len: 5)")
    print("  DEBUG: Added entity: 'Apple' with score 0.957")
    
    print(f"\nPROBLEM ANALYSIS:")
    print("Python: 'Apple Inc' -> character position 0-9 (length 9)")
    print("Rust:   'Apple' -> character position 0:1 (length 1)")
    print("")
    print("ISSUE: Rust is only getting span (0,1) instead of (0,9)!")
    print("")
    print("ROOT CAUSE: The span generation in Rust is producing")
    print("           incorrect character boundaries.")
    
    # Analyze word boundaries
    print(f"\nWORD BOUNDARY ANALYSIS:")
    words = english_text.split()
    char_pos = 0
    for i, word in enumerate(words[:5]):
        word_start = english_text.find(word, char_pos)
        word_end = word_start + len(word)
        print(f"  Word {i}: '{word}' -> {word_start}-{word_end}")
        char_pos = word_end + 1
    
    print(f"\nAPPLE INC ANALYSIS:")
    apple_inc_start = 0
    apple_inc_end = 9  # "Apple Inc" = 9 characters
    
    print(f"  'Apple Inc' should be: {apple_inc_start}-{apple_inc_end}")
    print(f"  Contains words: 'Apple' (0-5) + 'Inc.' (6-9)")
    print(f"  Word indices: 0, 1")
    print(f"  Expected span: word 0 to word 2 (exclusive)")
    
    print(f"\nRUST CURRENT BEHAVIOR:")
    print(f"  Getting span: char 0:1")
    print(f"  This maps to: word 0:1 (only 'Apple')")
    print(f"  Missing: word 1 ('Inc.')")
    
    print(f"\nFIX NEEDED:")
    print(f"  The span generation should produce (0,9) for 'Apple Inc'")
    print(f"  This should map to word indices (0,2)")
    print(f"  Current mapping is (0,1) -> word (0,1)")

def main():
    debug_span_boundaries()
    
    print("\n" + "="*50)
    print("SPAN BOUNDARIES DEBUG COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
