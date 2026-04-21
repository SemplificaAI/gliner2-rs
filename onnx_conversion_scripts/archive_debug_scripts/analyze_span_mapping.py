#!/usr/bin/env python3
"""
Detailed analysis of Python GLiNER2 span mapping to understand correct entity boundaries
This will help us fix the Rust span reconstruction issues
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

def analyze_span_mapping():
    """Analyze Python GLiNER2 span mapping in detail"""
    print("SPAN MAPPING ANALYSIS - Python GLiNER2")
    print("=" * 60)
    
    # Load the fine-tuned model
    try:
        from gliner import GLiNER
        model_path = "/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad/GLiNER_Jugaad-PII-Multi"
        print(f"Loading model from: {model_path}")
        model = GLiNER.from_pretrained(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    test_data = load_test_data("test_sentences.jsonl")
    
    # Entity labels for testing
    entity_labels = [
        "person_name", "organization_name", "location", "date", 
        "email", "phone_number", "address", "company_id", 
        "currency", "amount"
    ]
    
    # Analyze first sentence (English) in detail
    print(f"\n" + "="*60)
    print("DETAILED SPAN ANALYSIS - ENGLISH SENTENCE")
    print("="*60)
    
    english_sentence = test_data[0]
    text = english_sentence['text']
    
    print(f"TEXT: '{text}'")
    print(f"LENGTH: {len(text)} characters")
    
    # Show character positions for key entities
    print(f"\nCHARACTER POSITIONS:")
    print("-" * 30)
    
    # Find key positions manually
    key_entities = {
        'Apple Inc': text.find('Apple Inc'),
        'Tim Cook': text.find('Tim Cook'),
        'investor.relations@apple.com': text.find('investor.relations@apple.com'),
        'January 15, 2024': text.find('January 15, 2024'),
        '$119.6 billion': text.find('$119.6 billion')
    }
    
    for entity, pos in key_entities.items():
        if pos != -1:
            end_pos = pos + len(entity)
            print(f"  '{entity}' -> {pos}-{end_pos}")
            # Show context
            context_start = max(0, pos - 10)
            context_end = min(len(text), end_pos + 10)
            context = text[context_start:context_end]
            print(f"    Context: '{context}'")
    
    # Get Python predictions with detailed analysis
    print(f"\n" + "="*40)
    print("PYTHON GLiNER2 PREDICTIONS")
    print("="*40)
    
    try:
        entities = model.predict_entities(text, entity_labels, threshold=0.3)
        
        print(f"Entities found: {len(entities)}")
        
        for i, entity in enumerate(entities):
            print(f"\nEntity {i+1}:")
            print(f"  Text: '{entity['text']}'")
            print(f"  Label: {entity['label']}")
            print(f"  Score: {entity['score']:.3f}")
            
            # Find position in original text
            entity_text = entity['text']
            start_pos = text.find(entity_text)
            if start_pos != -1:
                end_pos = start_pos + len(entity_text)
                print(f"  Position: {start_pos}-{end_pos}")
                
                # Character-level analysis
                print(f"  Character analysis:")
                for j, char in enumerate(entity_text):
                    char_pos = start_pos + j
                    print(f"    [{char_pos:2d}] '{char}'")
                
                # Context analysis
                context_start = max(0, start_pos - 15)
                context_end = min(len(text), end_pos + 15)
                context = text[context_start:context_end]
                print(f"  Context: '{context}'")
                
                # Word boundary analysis
                words_before = text[:start_pos].split()
                words_after = text[end_pos:].split()
                print(f"  Words before: {words_before[-3:] if len(words_before) >= 3 else words_before}")
                print(f"  Words after: {words_after[:3] if len(words_after) >= 3 else words_after}")
                
            else:
                print(f"  Position: NOT FOUND!")
    
    except Exception as e:
        print(f"Error getting predictions: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    # Compare with Italian sentence
    print(f"\n" + "="*60)
    print("COMPARISON - ITALIAN SENTENCE")
    print("="*60)
    
    italian_sentence = test_data[1]
    italian_text = italian_sentence['text']
    
    print(f"TEXT: '{italian_text[:100]}...'")
    
    try:
        entities = model.predict_entities(italian_text, entity_labels, threshold=0.3)
        
        print(f"Entities found: {len(entities)}")
        
        for i, entity in enumerate(entities[:3]):  # First 3 entities
            print(f"\nEntity {i+1}:")
            print(f"  Text: '{entity['text']}'")
            print(f"  Label: {entity['label']}")
            print(f"  Score: {entity['score']:.3f}")
            
            # Find position
            entity_text = entity['text']
            start_pos = italian_text.find(entity_text)
            if start_pos != -1:
                end_pos = start_pos + len(entity_text)
                print(f"  Position: {start_pos}-{end_pos}")
                
                # Context
                context_start = max(0, start_pos - 10)
                context_end = min(len(italian_text), end_pos + 10)
                context = italian_text[context_start:context_end]
                print(f"  Context: '{context}'")
    
    except Exception as e:
        print(f"Error with Italian: {e}")
    
    # Try to access internal tokenizer for word-to-token mapping
    print(f"\n" + "="*60)
    print("INTERNAL TOKENIZER ANALYSIS")
    print("="*60)
    
    try:
        # Try different methods to access tokenizer
        if hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
            tokenizer = model.model.tokenizer
            print("Found tokenizer via model.model.tokenizer")
        elif hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
            print("Found tokenizer via model.tokenizer")
        else:
            print("Tokenizer not accessible")
            return
        
        # Test tokenization
        print(f"\nTokenizing English sentence:")
        tokens = tokenizer.tokenize(text)
        print(f"Total tokens: {len(tokens)}")
        print(f"First 20 tokens: {tokens[:20]}")
        
        # Get token IDs
        token_ids = tokenizer.encode(text, return_tensors="pt").input_ids[0]
        print(f"Token IDs length: {len(token_ids)}")
        
        # Decode tokens back to text
        print(f"\nToken-to-text mapping (first 15):")
        for i in range(min(15, len(tokens))):
            token_text = tokens[i]
            # Try to find this token in original text
            token_pos = text.find(token_text)
            if token_pos != -1:
                print(f"  [{i:2d}] '{token_text}' -> pos {token_pos}")
            else:
                print(f"  [{i:2d}] '{token_text}' -> NOT FOUND")
        
        # Show word boundaries
        print(f"\nWord boundaries in text:")
        words = text.split()
        pos = 0
        word_to_tokens = []
        
        for word_idx, word in enumerate(words[:10]):
            word_start = text.find(word, pos)
            if word_start != -1:
                word_end = word_start + len(word)
                print(f"  Word {word_idx}: '{word}' -> {word_start}-{word_end}")
                
                # Find tokens that map to this word
                word_tokens = []
                for token_idx, token in enumerate(tokens[:30]):
                    if token in word:
                        word_tokens.append((token_idx, token))
                
                if word_tokens:
                    print(f"    Tokens: {word_tokens}")
                
                pos = word_end
    
    except Exception as e:
        print(f"Error in tokenizer analysis: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def main():
    analyze_span_mapping()
    
    print("\n" + "="*60)
    print("SPAN MAPPING ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
