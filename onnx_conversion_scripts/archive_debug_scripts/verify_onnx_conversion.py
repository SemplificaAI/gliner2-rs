#!/usr/bin/env python3
"""
Verify ONNX model conversion and compare with PyTorch behavior
"""

import sys
import os

# Add the path to access GLiNER
sys.path.append('/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad')

def verify_onnx_conversion():
    """Verify that ONNX model should behave like PyTorch model"""
    print("ONNX CONVERSION VERIFICATION")
    print("=" * 50)
    
    print("ASSUMPTION:")
    print("- ONNX model converted from PyTorch (fp32/fp16)")
    print("- Should produce same results as PyTorch")
    print("- Rust implementation issue, not model issue")
    print()
    
    print("CURRENT RUST BEHAVIOR:")
    print("- Extracts single characters: 'A', 'l', '2'")
    print("- Very short spans (1-2 characters)")
    print("- Wrong labels for single characters")
    print()
    
    print("EXPECTED BEHAVIOR:")
    print("- Should extract complete entities: 'Apple Inc', 'Tim Cook'")
    print("- Longer spans (multiple words)")
    print("- Correct labels")
    print()
    
    print("POSSIBLE RUST ISSUES:")
    print("1. Span generation logic mismatch")
    print("2. Token indexing problems")
    print("3. Input preprocessing differences")
    print("4. Tensor shape mismatches")
    print("5. Threshold differences")
    print("6. Post-processing missing")

def analyze_span_generation_differences():
    """Analyze potential span generation issues"""
    print("\n" + "=" * 50)
    print("SPAN GENERATION ANALYSIS")
    print("=" * 50)
    
    print("PYTHON GLiNER2 SPAN GENERATION:")
    print("- Word-level spans")
    print("- Multi-word combinations")
    print("- Proper entity boundaries")
    print()
    
    print("RUST CURRENT SPAN GENERATION:")
    print("- Character-level spans")
    print("- All possible character combinations")
    print("- No word boundary awareness")
    print()
    
    print("ISSUE:")
    print("Character-level spans create too many small spans")
    print("Model might be picking the highest scoring small spans")
    print("Instead of preferring longer, meaningful spans")

def investigate_model_expectations():
    """Investigate what the model actually expects"""
    print("\n" + "=" * 50)
    print("MODEL EXPECTATIONS INVESTIGATION")
    print("=" * 50)
    
    print("QUESTIONS:")
    print("1. Was the PyTorch model trained on character-level or word-level spans?")
    print("2. Does the model expect specific span preprocessing?")
    print("3. Are there missing normalization steps?")
    print("4. Is the tokenizer producing different tokens?")
    print("5. Are input tensor shapes correct?")
    print()
    
    print("INVESTIGATION NEEDED:")
    print("- Check PyTorch model span generation")
    print("- Verify tokenizer behavior")
    print("- Compare input preprocessing")
    print("- Examine model architecture")

def main():
    verify_onnx_conversion()
    analyze_span_generation_differences()
    investigate_model_expectations()
    
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print("1. Check PyTorch GLiNER2 span generation code")
    print("2. Compare with Rust span generation")
    print("3. Identify the exact mismatch")
    print("4. Fix Rust implementation")

if __name__ == "__main__":
    main()
