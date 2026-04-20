#!/usr/bin/env python3
"""
Identify the exact issue with Rust model vs Python model
"""

import json

def analyze_model_difference():
    """Analyze why Rust and Python produce different results"""
    print("MODEL DIFFERENCE ANALYSIS")
    print("=" * 50)
    
    print("PYTHON GLiNER2 (Fine-Tuned Model):")
    print("Model: GLiNER_Jugaad-PII-Multi")
    print("Results:")
    print("  [67.2%] organization_name | 'Apple Inc'")
    print("  [97.6%] date | 'January 15, 2024'")
    print("  [93.3%] person_name | 'Tim Cook'")
    print("  [98.9%] email | 'investor.relations@apple.com'")
    
    print("\nRUST GLiNER2 (ONNX Model):")
    print("Model: lmo3-gliner2-multi-v1-onnx")
    print("Results:")
    print("  [95.7%] organization_name | 'Apple'")
    print("  [41.3%] date | 'announced'")
    print("  [92.0%] person_name | 'quarterly'")
    print("  [98.2%] email | '2024,'")
    
    print("\nPROBLEM IDENTIFICATION:")
    print("1. Different Models:")
    print("   Python: Fine-tuned GLiNER_Jugaad-PII-Multi")
    print("   Rust:   Base lmo3-gliner2-multi-v1-onnx")
    
    print("\n2. Different Entity Extraction:")
    print("   Python: 'Apple Inc' (complete)")
    print("   Rust:   'Apple' (partial)")
    
    print("\n3. Different Label Accuracy:")
    print("   Python: 'January 15, 2024' -> date")
    print("   Rust:   'announced' -> date (WRONG)")
    
    print("\n4. Different Span Quality:")
    print("   Python: Complete, accurate entities")
    print("   Rust:   Partial, incorrect entities")
    
    print("\nROOT CAUSE:")
    print("The Rust implementation is using a different model!")
    print("It needs the same fine-tuned model as Python.")

def analyze_span_generation_difference():
    """Analyze span generation differences"""
    print("\n" + "=" * 50)
    print("SPAN GENERATION DIFFERENCE")
    print("=" * 50)
    
    text = "Apple Inc. announced its quarterly earnings report"
    
    # Word-level spans (what Python does)
    words = text.split()
    print("Word-level spans (Python GLiNER2):")
    for i in range(len(words)):
        for j in range(i+1, min(i+4, len(words)+1)):  # Up to 3 words
            span_words = words[i:j]
            span_text = ' '.join(span_words)
            print(f"  Words {i}-{j}: '{span_text}'")
    
    print("\nCharacter-level spans (What Rust might be doing):")
    for start in range(0, min(20, len(text))):
        for width in range(1, min(8, len(text)-start)):
            end = start + width
            span_text = text[start:end]
            if len(span_text.strip()) > 0:
                print(f"  Char {start}:{end}: '{span_text}'")

def main():
    analyze_model_difference()
    analyze_span_generation_difference()
    
    print("\n" + "=" * 50)
    print("CONCLUSION")
    print("=" * 50)
    print("The Rust implementation needs:")
    print("1. The same fine-tuned model as Python")
    print("2. Word-level span generation, not character-level")
    print("3. Proper entity boundary detection")
    print("4. Correct label assignment logic")

if __name__ == "__main__":
    main()
