#!/usr/bin/env python3
"""
Analyze Python GLiNER2 original code logic to understand entity extraction
"""

import json
import sys
import os

# Add the path to access GLiNER
sys.path.append('/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad')

def analyze_python_gliner_logic():
    """Analyze how Python GLiNER2 extracts entities correctly"""
    print("ANALYZING PYTHON GLiNER2 LOGIC")
    print("=" * 50)
    
    try:
        from gliner import GLiNER
        
        # Load the fine-tuned model
        model_path = "/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad/GLiNER_Jugaad-PII-Multi"
        model = GLiNER.from_pretrained(model_path)
        
        # Test sentence
        text = "Apple Inc. announced its quarterly earnings report on January 15, 2024, showing a revenue of $119.6 billion. CEO Tim Cook stated that iPhone sales exceeded expectations in Europe and Asia. The company's stock price (AAPL) rose by 5.2% in Nasdaq trading. Contact investor.relations@apple.com for more information."
        
        entity_labels = [
            "person_name", "organization_name", "location", "date", 
            "email", "phone_number", "address", "company_id", 
            "currency", "amount"
        ]
        
        print(f"TEXT: {text}")
        print(f"LABELS: {entity_labels}")
        print()
        
        # Get predictions
        entities = model.predict_entities(text, entity_labels, threshold=0.3)
        
        print("PYTHON GLiNER2 RESULTS:")
        for i, entity in enumerate(entities):
            print(f"  {i+1}. [{entity['score']:.1%}] {entity['label']} | '{entity['text']}'")
        
        print()
        
        # Try to access internal model structure
        print("MODEL STRUCTURE ANALYSIS:")
        print(f"Model type: {type(model)}")
        
        # Check if model has tokenizer
        if hasattr(model, 'tokenizer'):
            print(f"Tokenizer: {type(model.tokenizer)}")
        else:
            print("No direct tokenizer access")
        
        # Check model config
        if hasattr(model, 'config'):
            print(f"Config: {type(model.config)}")
            if hasattr(model.config, 'max_width'):
                print(f"Max width: {model.config.max_width}")
        
        # Try to understand the prediction process
        print()
        print("PREDICTION PROCESS ANALYSIS:")
        
        # Check if we can access internal prediction methods
        if hasattr(model, 'predict'):
            print("Has predict method")
        
        if hasattr(model, 'predict_entities'):
            print("Has predict_entities method")
        
        # Try to get tokenization info
        try:
            # This might fail but let's try
            if hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
                tokenizer = model.model.tokenizer
                tokens = tokenizer.tokenize(text)
                print(f"Token count: {len(tokens)}")
                print(f"First 10 tokens: {tokens[:10]}")
        except Exception as e:
            print(f"Cannot access tokenizer: {e}")
        
    except Exception as e:
        print(f"Error analyzing Python GLiNER2: {e}")
        import traceback
        traceback.print_exc()

def analyze_span_generation():
    """Analyze how spans are generated in Python GLiNER2"""
    print("\n" + "=" * 50)
    print("SPAN GENERATION ANALYSIS")
    print("=" * 50)
    
    text = "Apple Inc. announced its quarterly earnings report on January 15, 2024"
    
    # Manual span generation like GLiNER2 does
    words = text.split()
    print(f"Words: {words}")
    
    # Character positions
    char_pos = 0
    word_positions = []
    for word in words:
        word_start = text.find(word, char_pos)
        word_end = word_start + len(word)
        word_positions.append((word_start, word_end, word))
        char_pos = word_end + 1
        print(f"  '{word}' -> {word_start}-{word_end}")
    
    # Generate all possible spans
    print("\nPossible spans:")
    for start_idx, (start_char, _, start_word) in enumerate(word_positions):
        for end_idx, (_, end_char, end_word) in enumerate(word_positions[start_idx:], start_idx):
            if end_idx > start_idx:  # At least 2 words
                span_text = text[start_char:end_char]
                print(f"  Words {start_idx}-{end_idx}: '{span_text}' -> {start_char}-{end_char}")

def main():
    analyze_python_gliner_logic()
    analyze_span_generation()

if __name__ == "__main__":
    main()
