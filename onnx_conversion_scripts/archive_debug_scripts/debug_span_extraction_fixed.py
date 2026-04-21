#!/usr/bin/env python3
"""
Debug script to show how Python GLiNER2 extracts spans
Forces loading of a working GLiNER model to show span extraction details
"""

import json
import time
import os
import sys
import re

def load_test_data(file_path):
    """Load test sentences from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def debug_span_extraction(test_data):
    """Debug span extraction with any available GLiNER model"""
    print("DEBUG: Python GLiNER2 Span Extraction Analysis")
    print("=" * 60)
    
    # Try to load ANY GLiNER model for debugging
    model = None
    model_name = None
    
    # Try different model loading strategies
    models_to_try = [
        "urchade/gliner_base",
        "urchade/gliner_multi-v2.1", 
        "urchade/gliner_small"
    ]
    
    for model_attempt in models_to_try:
        try:
            print(f"Attempting to load: {model_attempt}")
            from gliner import GLiNER
            model = GLiNER.from_pretrained(model_attempt)
            model_name = model_attempt
            print(f"SUCCESS: Loaded {model_attempt}")
            break
        except Exception as e:
            print(f"Failed to load {model_attempt}: {e}")
            continue
    
    if model is None:
        print("Could not load any GLiNER model. Creating manual tokenization debug...")
        return manual_tokenization_debug(test_data)
    
    # Entity labels for testing
    entity_labels = [
        "person_name", "organization_name", "location", "date", 
        "email", "phone_number", "address", "company_id", 
        "currency", "amount"
    ]
    
    # Test only first 2 sentences for detailed debugging
    debug_sentences = test_data[:2]
    
    for i, record in enumerate(debug_sentences):
        lang = record.get('language', 'unknown').upper()
        text = record['text']
        
        print(f"\n" + "="*60)
        print(f"DEBUGGING SENTENCE {i+1}: {lang}")
        print("="*60)
        print(f"FULL TEXT: '{text}'")
        print(f"TEXT LENGTH: {len(text)} characters")
        
        # 1. TOKENIZATION ANALYSIS
        print(f"\n1. TOKENIZATION ANALYSIS:")
        print("-" * 30)
        
        try:
            # Get basic tokenization
            tokens = model.tokenizer.tokenize(text)
            print(f"Total tokens: {len(tokens)}")
            print(f"Tokens: {tokens}")
            
            # Get token IDs
            token_ids = model.tokenizer.encode(text, return_tensors="pt").input_ids[0]
            print(f"Token IDs: {token_ids.tolist()}")
            
            # Decode back to verify
            decoded_text = model.tokenizer.decode(token_ids)
            print(f"Decoded text: '{decoded_text}'")
            
            # Show token-to-text mapping
            print(f"\nToken-to-Text Mapping:")
            for idx, token in enumerate(tokens[:20]):  # First 20 tokens
                # Find position of this token in original text
                token_start = text.find(token)
                if token_start != -1:
                    token_end = token_start + len(token)
                    context_start = max(0, token_start - 5)
                    context_end = min(token_end + 5, len(text))
                    context = f"...{text[context_start:token_start]}[{token}]{text[token_end:context_end]}..."
                    print(f"  [{idx:2d}] '{token}' -> pos {token_start}-{token_end} {context}")
                else:
                    print(f"  [{idx:2d}] '{token}' -> NOT FOUND")
            
        except Exception as e:
            print(f"Tokenization error: {e}")
        
        # 2. ENTITY PREDICTION ANALYSIS
        print(f"\n2. ENTITY PREDICTION ANALYSIS:")
        print("-" * 35)
        
        try:
            start_time = time.time()
            entities = model.predict_entities(text, entity_labels, threshold=0.3)  # Lower threshold for more entities
            inference_time = time.time() - start_time
            
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Entities found: {len(entities)}")
            
            for j, entity in enumerate(entities):
                print(f"\n  Entity {j+1}:")
                print(f"    Text: '{entity['text']}'")
                print(f"    Label: {entity['label']}")
                print(f"    Score: {entity['score']:.3f}")
                
                # Find entity position in original text
                entity_text = entity['text']
                start_pos = text.find(entity_text)
                if start_pos != -1:
                    end_pos = start_pos + len(entity_text)
                    print(f"    Position: {start_pos}-{end_pos}")
                    
                    # Extract context around entity
                    context_start = max(0, start_pos - 15)
                    context_end = min(len(text), end_pos + 15)
                    context = text[context_start:context_end]
                    print(f"    Context: '{context}'")
                    
                    # Show character-level analysis
                    print(f"    Character analysis:")
                    for k, char in enumerate(entity_text):
                        char_pos = start_pos + k
                        print(f"      [{char_pos:2d}] '{char}'")
                else:
                    print(f"    Position: NOT FOUND in original text!")
                    
                    # Try to find partial matches
                    print(f"    Partial match analysis:")
                    for k in range(1, min(len(entity_text), 10)):
                        partial = entity_text[:k]
                        pos = text.find(partial)
                        if pos != -1:
                            print(f"      '{partial}' found at position {pos}")
                            break
                
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        
        # 3. SPAN BOUNDARY ANALYSIS
        print(f"\n3. SPAN BOUNDARY ANALYSIS:")
        print("-" * 30)
        
        # Manual span detection for comparison
        print("Manual pattern detection:")
        
        # Email detection
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.finditer(email_pattern, text)
        for match in emails:
            start, end = match.span()
            print(f"  Email: '{match.group()}' at {start}-{end}")
            context_start = max(0, start - 10)
            context_end = min(end + 10, len(text))
            print(f"    Context: '{text[context_start:context_end]}'")
        
        # Phone detection
        phone_pattern = r'\+?[0-9][0-9\s\-\(\)]{7,}[0-9]'
        phones = re.finditer(phone_pattern, text)
        for match in phones:
            start, end = match.span()
            print(f"  Phone: '{match.group()}' at {start}-{end}")
            context_start = max(0, start - 10)
            context_end = min(end + 10, len(text))
            print(f"    Context: '{text[context_start:context_end]}'")
        
        # Name detection (simple)
        name_patterns = {
            'EN': r'\b(Tim Cook|Apple Inc)\b',
            'IT': r'\b(Mario Rossi|Tech Italia)\b',
            'FR': r'\b(Jean Dupont|Tech France)\b',
            'ES': r'\b(Juan García|Tech España)\b',
            'DE': r'\b(Hans Müller|Tech Deutschland)\b',
            'PT': r'\b(João Silva|Tech Portugal)\b'
        }
        
        if lang in name_patterns:
            names = re.finditer(name_patterns[lang], text)
            for match in names:
                start, end = match.span()
                print(f"  Name: '{match.group()}' at {start}-{end}")
                context_start = max(0, start - 10)
                context_end = min(end + 10, len(text))
                print(f"    Context: '{text[context_start:context_end]}'")

def manual_tokenization_debug(test_data):
    """Manual tokenization debug when no model is available"""
    print("MANUAL TOKENIZATION DEBUG")
    print("=" * 40)
    
    # Test only first sentence
    record = test_data[0]
    text = record['text']
    
    print(f"Text: '{text}'")
    print(f"Length: {len(text)}")
    
    # Character-by-character analysis
    print(f"\nCharacter analysis:")
    for i, char in enumerate(text[:100]):  # First 100 chars
        if char in [' ', '.', '@', '+', '-', '(', ')']:
            print(f"[{i:2d}] '{char}' <- SPECIAL")
        else:
            print(f"[{i:2d}] '{char}'")
    
    # Word boundaries
    print(f"\nWord boundaries:")
    words = text.split()
    pos = 0
    for word in words[:10]:  # First 10 words
        word_start = text.find(word, pos)
        word_end = word_start + len(word)
        print(f"'{word}' -> {word_start}-{word_end}")
        pos = word_end

def main():
    print("Python GLiNER2 Span Extraction Debug")
    print("=" * 50)
    
    # Load test data
    test_file = "test_sentences.jsonl"
    try:
        test_data = load_test_data(test_file)
        print(f"Loaded {len(test_data)} test sentences")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Debug span extraction
    debug_span_extraction(test_data)
    
    print("\n" + "=" * 50)
    print("DEBUG COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
