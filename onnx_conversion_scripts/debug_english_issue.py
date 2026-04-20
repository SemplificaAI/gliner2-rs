#!/usr/bin/env python3
"""
Debug script to investigate why the fine-tuned GLiNER2 model fails on English sentence
"""

import json
import time
import os
import sys

def load_test_data(file_path):
    """Load test sentences from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def debug_english_sentence():
    """Debug the English sentence issue with fine-tuned GLiNER2"""
    print("DEBUG: English Sentence Issue Investigation")
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
    english_sentence = test_data[0]  # First sentence is English
    
    print(f"\nEnglish Sentence:")
    print(f"TEXT: '{english_sentence['text']}'")
    print(f"LANGUAGE: {english_sentence['language']}")
    print(f"LENGTH: {len(english_sentence['text'])} characters")
    
    # Entity labels for testing
    entity_labels = [
        "person_name", "organization_name", "location", "date", 
        "email", "phone_number", "address", "company_id", 
        "currency", "amount"
    ]
    
    print(f"\nEntity labels being tested:")
    for i, label in enumerate(entity_labels, 1):
        print(f"  {i}. {label}")
    
    # Test with different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"\n" + "="*60)
    print("TESTING WITH DIFFERENT THRESHOLDS")
    print("="*60)
    
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        
        try:
            start_time = time.time()
            entities = model.predict_entities(
                english_sentence['text'], 
                entity_labels, 
                threshold=threshold
            )
            inference_time = time.time() - start_time
            
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Entities found: {len(entities)}")
            
            if entities:
                for i, entity in enumerate(entities):
                    print(f"  [{i+1}] [{entity['score']:.1%}] {entity['label']} | '{entity['text']}'")
            else:
                print("  No entities found")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Compare with a working language (Italian)
    print(f"\n" + "="*60)
    print("COMPARISON WITH ITALIAN (WORKING)")
    print("="*60)
    
    italian_sentence = test_data[1]  # Second sentence is Italian
    
    print(f"\nItalian Sentence:")
    print(f"TEXT: '{italian_sentence['text'][:100]}...'")
    print(f"LANGUAGE: {italian_sentence['language']}")
    
    for threshold in [0.5, 0.7]:
        print(f"\n--- Italian - Threshold: {threshold} ---")
        
        try:
            start_time = time.time()
            entities = model.predict_entities(
                italian_sentence['text'], 
                entity_labels, 
                threshold=threshold
            )
            inference_time = time.time() - start_time
            
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Entities found: {len(entities)}")
            
            if entities:
                for i, entity in enumerate(entities):
                    print(f"  [{i+1}] [{entity['score']:.1%}] {entity['label']} | '{entity['text']}'")
            else:
                print("  No entities found")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Test with simplified entity labels
    print(f"\n" + "="*60)
    print("TESTING WITH SIMPLIFIED ENTITY LABELS")
    print("="*60)
    
    simplified_labels = ["person", "organization", "email", "phone", "date"]
    
    print(f"\nSimplified labels: {simplified_labels}")
    
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n--- English - Simplified Labels - Threshold: {threshold} ---")
        
        try:
            start_time = time.time()
            entities = model.predict_entities(
                english_sentence['text'], 
                simplified_labels, 
                threshold=threshold
            )
            inference_time = time.time() - start_time
            
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Entities found: {len(entities)}")
            
            if entities:
                for i, entity in enumerate(entities):
                    print(f"  [{i+1}] [{entity['score']:.1%}] {entity['label']} | '{entity['text']}'")
            else:
                print("  No entities found")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Test model prediction method details
    print(f"\n" + "="*60)
    print("MODEL PREDICTION METHOD ANALYSIS")
    print("="*60)
    
    try:
        # Test if the model has different prediction methods
        print(f"\nAvailable model methods:")
        methods = [method for method in dir(model) if not method.startswith('_')]
        for method in methods:
            print(f"  - {method}")
        
        # Test if there's a different predict method
        if hasattr(model, 'predict'):
            print(f"\nTrying model.predict() method:")
            try:
                result = model.predict(
                    english_sentence['text'],
                    entity_labels,
                    threshold=0.5
                )
                print(f"Result type: {type(result)}")
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error with predict(): {e}")
        
        # Check model config
        if hasattr(model, 'config'):
            print(f"\nModel config:")
            print(f"  Type: {type(model.config)}")
            if hasattr(model.config, '__dict__'):
                for key, value in model.config.__dict__.items():
                    if not key.startswith('_') and len(str(value)) < 100:
                        print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error analyzing model: {e}")

def main():
    debug_english_sentence()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
