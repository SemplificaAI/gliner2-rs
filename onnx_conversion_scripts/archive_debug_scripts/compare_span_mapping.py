#!/usr/bin/env python3
"""
Compare Python GLiNER2 span mapping with Rust implementation to identify the exact issues
"""

import json

def analyze_span_differences():
    """Analyze the differences between Python and Rust span mapping"""
    print("SPAN MAPPING COMPARISON - Python vs Rust")
    print("=" * 60)
    
    # Load test data
    with open('test_sentences.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    english_text = test_data[0]['text']
    
    print("ENGLISH SENTENCE ANALYSIS:")
    print(f"TEXT: '{english_text}'")
    print(f"LENGTH: {len(english_text)} characters")
    
    # Python results (from our analysis)
    python_entities = [
        {'text': 'Apple Inc', 'label': 'organization_name', 'position': (0, 9)},
        {'text': 'January 15, 2024', 'label': 'date', 'position': (54, 70)},
        {'text': '$119.6 billion', 'label': 'currency', 'position': (93, 107)},
        {'text': 'Tim Cook', 'label': 'person_name', 'position': (113, 121)},
        {'text': 'Europe and', 'label': 'location', 'position': (172, 182)},
        {'text': 'Asia.', 'label': 'location', 'position': (183, 188)},
        {'text': 'AAPL)', 'label': 'company_id', 'position': (216, 221)},
        {'text': 'Nasdaq trading', 'label': 'location', 'position': (238, 252)},
        {'text': 'investor.relations@apple.com', 'label': 'email', 'position': (262, 290)}
    ]
    
    # Rust results (from our previous test)
    rust_entities = [
        {'text': 'Apple', 'label': 'unknown', 'position': 'fragment'},
        {'text': 'revenue of $119.6', 'label': 'unknown', 'position': 'fragment'},
        {'text': 'Europe and', 'label': 'unknown', 'position': 'fragment'},
        {'text': 'stock', 'label': 'unknown', 'position': 'fragment'}
    ]
    
    print("\n" + "="*40)
    print("PYTHON RESULTS (CORRECT)")
    print("="*40)
    
    for i, entity in enumerate(python_entities):
        start, end = entity['position']
        print(f"{i+1}. [{entity['label']}] '{entity['text']}' -> {start}-{end}")
        print(f"   Context: '{english_text[max(0,start-10):min(end+10,len(english_text))]}...'")
    
    print("\n" + "="*40)
    print("RUST RESULTS (INCORRECT)")
    print("="*40)
    
    for i, entity in enumerate(rust_entities):
        print(f"{i+1}. [{entity['label']}] '{entity['text']}' -> {entity['position']}")
    
    print("\n" + "="*60)
    print("ROOT CAUSE ANALYSIS")
    print("="*60)
    
    print("\n1. SPAN BOUNDARY ISSUES:")
    print("   Python: Uses exact character positions from text")
    print("   Rust: Uses word_to_token_maps incorrectly")
    
    print("\n2. ENTITY COMPLETENESS:")
    print("   Python: 'Apple Inc' (complete)")
    print("   Rust: 'Apple' (truncated)")
    
    print("\n3. EMAIL/PHONE DETECTION:")
    print("   Python: 'investor.relations@apple.com' (complete)")
    print("   Rust: NOT DETECTED")
    
    print("\n4. SPAN RECONSTRUCTION LOGIC:")
    print("   Python: Direct text-to-position mapping")
    print("   Rust: Complex word-to-token mapping with errors")
    
    print("\n" + "="*60)
    print("RUST CODE ISSUES IDENTIFIED")
    print("="*60)
    
    print("\nProblem 1 - Span Index Mapping:")
    print("   Rust code:")
    print("   let span_start_word = start;  // This is WRONG!")
    print("   let span_end_word = end;")
    print("   ")
    print("   Issue: 'start' and 'end' are character positions,")
    print("         but Rust treats them as word indices!")
    
    print("\nProblem 2 - Token Boundary Lookup:")
    print("   Rust code:")
    print("   let token_start = record.word_to_token_maps[span_start_word].0;")
    print("   let token_end = record.word_to_token_maps[span_end_word - 1].1;")
    print("   ")
    print("   Issue: Using character positions as word indices causes")
    print("         incorrect token boundary lookups!")
    
    print("\nProblem 3 - Threshold:")
    print("   Rust code: if prob > 0.5")
    print("   Python: threshold 0.3 for best results")
    print("   ")
    print("   Issue: Too high threshold filters out valid entities")
    
    print("\n" + "="*60)
    print("FIX STRATEGY")
    print("="*60)
    
    print("\n1. FIX SPAN MAPPING:")
    print("   - Convert character positions to word indices correctly")
    print("   - Use proper text-to-word mapping")
    
    print("\n2. FIX TOKEN BOUNDARIES:")
    print("   - Validate word-to-token mapping logic")
    print("   - Ensure span boundaries are within valid ranges")
    
    print("\n3. FIX THRESHOLD:")
    print("   - Lower threshold from 0.5 to 0.3")
    print("   - Match Python GLiNER2 behavior")
    
    print("\n4. DEBUG LOGGING:")
    print("   - Add detailed span boundary logging")
    print("   - Compare with Python results during development")

def main():
    analyze_span_differences()
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
