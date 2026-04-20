#!/usr/bin/env python3
"""
Investigate PyTorch GLiNER2 span generation to understand the correct approach
"""

import sys
import os

# Add the path to access GLiNER
sys.path.append('/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad')

def investigate_pytorch_span_generation():
    """Investigate how PyTorch GLiNER2 generates spans"""
    print("PYTORCH SPAN GENERATION INVESTIGATION")
    print("=" * 50)
    
    try:
        from gliner import GLiNER
        
        # Load the fine-tuned model
        model_path = "/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad/GLiNER_Jugaad-PII-Multi"
        model = GLiNER.from_pretrained(model_path)
        
        # Test sentence
        text = "Apple Inc. announced its quarterly earnings report"
        
        print(f"TEXT: {text}")
        print(f"MODEL: {type(model)}")
        
        # Try to access internal span generation
        if hasattr(model, 'model'):
            inner_model = model.model
            print(f"Inner model type: {type(inner_model)}")
            
            # Check for span generation methods
            if hasattr(inner_model, 'get_span_idx'):
                print("Found get_span_idx method")
                try:
                    # Try to call it to see what it generates
                    tokens = model.tokenizer(text, return_tensors="pt")
                    print(f"Token count: {len(tokens['input_ids'][0])}")
                    
                    # Try to get span indices
                    span_idx = inner_model.get_span_idx(tokens['input_ids'], tokens['attention_mask'])
                    print(f"Span indices shape: {span_idx.shape}")
                    print(f"First few spans: {span_idx[0][:10]}")
                    
                except Exception as e:
                    print(f"Error calling get_span_idx: {e}")
            else:
                print("No get_span_idx method found")
            
            # Check for other relevant methods
            methods = [attr for attr in dir(inner_model) if 'span' in attr.lower()]
            print(f"Span-related methods: {methods}")
            
        # Check model config
        if hasattr(model, 'config'):
            config = model.config
            print(f"Config type: {type(config)}")
            
            # Look for span-related config
            span_config = [attr for attr in dir(config) if 'span' in attr.lower() or 'width' in attr.lower()]
            print(f"Span/Wdith config: {span_config}")
            
            for attr in span_config:
                try:
                    value = getattr(config, attr)
                    print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: <unable to access>")
        
        # Try to understand tokenization
        print("\nTOKENIZATION ANALYSIS:")
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
            tokens = tokenizer.tokenize(text)
            print(f"Tokens: {tokens}")
            print(f"Token count: {len(tokens)}")
            
            # Get token IDs
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded['input_ids'][0]
            print(f"Input IDs: {input_ids.tolist()}")
            
            # Try to decode back
            decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Decoded: '{decoded}'")
    
    except Exception as e:
        print(f"Error investigating PyTorch model: {e}")
        import traceback
        traceback.print_exc()

def analyze_character_vs_word_spans():
    """Analyze the difference between character and word spans"""
    print("\n" + "=" * 50)
    print("CHARACTER vs WORD SPANS ANALYSIS")
    print("=" * 50)
    
    text = "Apple Inc. announced"
    
    print("CHARACTER-LEVEL SPANS (Current Rust):")
    for i in range(min(10, len(text))):
        for j in range(i+1, min(i+4, len(text)+1)):
            span = text[i:j]
            print(f"  {i}:{j} -> '{span}'")
    
    print("\nWORD-LEVEL SPANS (Expected):")
    words = text.split()
    for i in range(len(words)):
        for j in range(i+1, min(i+3, len(words)+1)):
            span_words = words[i:j]
            span = ' '.join(span_words)
            print(f"  Words {i}-{j} -> '{span}'")
    
    print("\nISSUE:")
    print("Character-level spans create many small, meaningless fragments")
    print("Word-level spans create meaningful entity candidates")
    print("The model was likely trained on word-level spans")

def main():
    investigate_pytorch_span_generation()
    analyze_character_vs_word_spans()

if __name__ == "__main__":
    main()
