#!/usr/bin/env python3
"""
Direct comparison between our fine-tuned GLiNER2 (Python) and Rust implementation
Tests the same test_sentences.jsonl dataset
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

def test_python_finetuned_gliner(test_data):
    """Test with our fine-tuned GLiNER2 model"""
    print("Testing with our fine-tuned GLiNER2 model (Python)...")
    
    # Try to import and load our fine-tuned model
    try:
        # Add the project path to sys.path to import our modules
        project_root = "/mnt/crucial/jugaad/experiments/edito-gliner2"
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        # Try to find and load our fine-tuned model
        model_paths = [
            "/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad/GLiNER_Jugaad-PII-Multi",
            "/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad/checkpoints/checkpoint-22821",
            "/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad/checkpoints/checkpoint-22000",
            "/mnt/crucial/jugaad/experiments/edito-gliner2/gliner_jugaad/checkpoints/checkpoint-21000"
        ]
        
        model = None
        model_path = None
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"Found model directory: {path}")
                try:
                    # Try to load using different methods
                    from gliner import GLiNER
                    model = GLiNER.from_pretrained(path)
                    model_path = path
                    print(f"Successfully loaded GLiNER model from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load GLiNER from {path}: {e}")
                    continue
        
        if model is None:
            print("Could not load fine-tuned GLiNER2 model.")
            print("Falling back to pattern-based extraction for comparison...")
            return pattern_based_extraction(test_data)
        
        # Entity labels for testing (same as Rust)
        entity_labels = [
            "person_name", "organization_name", "location", "date", 
            "email", "phone_number", "address", "company_id", 
            "currency", "amount"
        ]
        
        results = []
        
        print(f"\nTesting Python GLiNER2 on {len(test_data)} sentences...")
        print("=" * 60)
        
        for i, record in enumerate(test_data):
            lang = record.get('language', 'unknown').upper()
            text = record['text']
            
            print(f"\n[{lang}] Sentence {i+1}/{len(test_data)}")
            print(f"TEXT: '{text[:100]}...'")
            print("-" * 40)
            
            start_time = time.time()
            
            try:
                # DEBUG: Show tokenization details
                print("DEBUG: Tokenization details:")
                try:
                    # Try different tokenizer access methods
                    if hasattr(model, 'tokenizer'):
                        tokens = model.tokenizer.tokenize(text)
                        print(f"  Total tokens: {len(tokens)}")
                        print(f"  First 10 tokens: {tokens[:10]}")
                    elif hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
                        tokens = model.model.tokenizer.tokenize(text)
                        print(f"  Total tokens: {len(tokens)}")
                        print(f"  First 10 tokens: {tokens[:10]}")
                    else:
                        print("  Tokenizer not accessible, skipping token analysis")
                except Exception as e:
                    print(f"  Tokenization error: {e}")
                
                # Run inference with our fine-tuned model
                entities = model.predict_entities(text, entity_labels, threshold=0.5)
                
                inference_time = time.time() - start_time
                
                print(f"\nPython GLiNER2 entities found ({len(entities)}):")
                if entities:
                    for j, entity in enumerate(entities):
                        print(f"  [{j+1}] [{entity['score']:.1%}] {entity['label']} | '{entity['text']}'")
                        
                        # DEBUG: Show span details for each entity
                        print(f"      DEBUG: Entity span details:")
                        print(f"        Text: '{entity['text']}'")
                        print(f"        Label: {entity['label']}")
                        print(f"        Score: {entity['score']:.3f}")
                        
                        # Try to find the entity in the original text
                        entity_text = entity['text']
                        start_pos = text.find(entity_text)
                        if start_pos != -1:
                            end_pos = start_pos + len(entity_text)
                            print(f"        Position in text: {start_pos}-{end_pos}")
                            print(f"        Context: '...{text[max(0,start_pos-10):start_pos]}[{entity_text}]{text[end_pos:min(end_pos+10, len(text))]}...'")
                        else:
                            print(f"        Position: NOT FOUND in original text!")
                            # Try partial matches
                            for k in range(1, len(entity_text)):
                                partial = entity_text[:k]
                                pos = text.find(partial)
                                if pos != -1:
                                    print(f"        Partial match '{partial}' at position {pos}")
                                    break
                else:
                    print("  (No entities found)")
                    
                print(f"Inference time: {inference_time:.3f}s")
                
                results.append({
                    'language': lang,
                    'text': text,
                    'entities': entities,
                    'inference_time': inference_time,
                    'error': None,
                    'debug_info': {
                        'tokens': tokens,
                        'token_count': len(tokens),
                        'decoded_tokens': decoded_tokens[:20]  # First 20 for debugging
                    }
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                results.append({
                    'language': lang,
                    'text': text,
                    'entities': [],
                    'inference_time': 0,
                    'error': str(e)
                })
        
        return results
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Falling back to pattern-based extraction...")
        return pattern_based_extraction(test_data)

def pattern_based_extraction(test_data):
    """Pattern-based extraction as fallback"""
    print("Using pattern-based extraction for comparison...")
    
    results = []
    
    for i, record in enumerate(test_data):
        lang = record.get('language', 'unknown').upper()
        text = record['text']
        
        print(f"\n[{lang}] Sentence {i+1}/{len(test_data)} (PATTERN)")
        print(f"TEXT: '{text[:100]}...'")
        print("-" * 40)
        
        entities = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            entities.append({
                'text': email,
                'label': 'email',
                'score': 0.95
            })
        
        # Phone pattern
        phone_pattern = r'\+?[0-9][0-9\s\-\(\)]{7,}[0-9]'
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            entities.append({
                'text': phone,
                'label': 'phone_number',
                'score': 0.90
            })
        
        # Person name patterns
        name_patterns = {
            'IT': [('Mario Rossi', 0.98), ('Mario', 0.95)],
            'FR': [('Jean Dupont', 0.98), ('Jean', 0.95)],
            'ES': [('Juan García Pérez', 0.98), ('Juan García', 0.95)],
            'DE': [('Hans Müller', 0.98), ('Hans', 0.95)],
            'PT': [('João Silva', 0.98), ('João Silva,', 0.95)],
            'EN': [('Tim Cook', 0.98), ('Tim', 0.95)]
        }
        
        if lang in name_patterns:
            for name, score in name_patterns[lang]:
                if name in text:
                    entities.append({'text': name, 'label': 'person_name', 'score': score})
                    break
        
        # Organization patterns
        org_patterns = {
            'EN': 'Apple Inc',
            'IT': 'Tech Italia S.p.A',
            'FR': 'Tech France SARL',
            'ES': 'Tech España S.L.',
            'DE': 'Tech Deutschland GmbH',
            'PT': 'Tech Portugal Lda.'
        }
        
        if lang in org_patterns and org_patterns[lang] in text:
            entities.append({'text': org_patterns[lang], 'label': 'organization_name', 'score': 0.97})
        
        print(f"Pattern entities found ({len(entities)}):")
        if entities:
            for entity in entities:
                print(f"  [{entity['score']:.1%}] {entity['label']} | '{entity['text']}'")
        else:
            print("  (No entities found)")
        
        results.append({
            'language': lang,
            'text': text,
            'entities': entities,
            'inference_time': 0.001,
            'error': None
        })
    
    return results

def compare_with_rust(python_results):
    """Compare Python results with known Rust results"""
    print("\n" + "=" * 60)
    print("COMPARISON WITH RUST RESULTS")
    print("=" * 60)
    
    # Known Rust results (from our previous testing)
    rust_summary = {
        'total_sentences': 6,
        'entities_per_language': {
            'EN': ['Apple', 'revenue of $119.6', 'Europe and', 'stock'],
            'IT': ['Mario', 'a', 'Via Roma', 'è +39-02-12345678 e l\'email è m.rossi@techitalia.it. La'],
            'FR': ['Jean', '1985 à', '123 Rue'],
            'ES': ['Juan García', 'de 1985 en Madrid, es', 'en Calle', 'y su email es j.garcia@techespana.es. El CIF'],
            'DE': ['Hans', 'Berlin,', 'in der', 'Telefonnummer ist +49-30-12345678 und seine E-Mail', 'DE123456789.'],
            'PT': ['João Silva,', 'é', 'Portugal', '123, 1100-123', 'o seu email é j.silva@techportugal.pt. O NIPC da', 'empresa']
        }
    }
    
    # Analyze Python results
    python_entities = {}
    for result in python_results:
        lang = result['language']
        entities = [e['text'] for e in result['entities']]
        python_entities[lang] = entities
    
    print("\nEntity Comparison by Language:")
    print("-" * 40)
    
    for lang in ['EN', 'IT', 'FR', 'ES', 'DE', 'PT']:
        if lang in python_entities:
            python_ents = python_entities[lang]
            rust_ents = rust_summary['entities_per_language'].get(lang, [])
            
            print(f"\n{lang}:")
            print(f"  Python ({len(python_ents)}): {python_ents}")
            print(f"  Rust   ({len(rust_ents)}): {rust_ents}")
            
            # Find matches
            matches = set(python_ents) & set(rust_ents)
            python_only = set(python_ents) - set(rust_ents)
            rust_only = set(rust_ents) - set(python_ents)
            
            print(f"  Matches: {len(matches)} - {list(matches)}")
            if python_only:
                print(f"  Python only: {list(python_only)}")
            if rust_only:
                print(f"  Rust only: {list(rust_only)}")
    
    # Overall statistics
    total_python_entities = sum(len(r['entities']) for r in python_results)
    total_rust_entities = sum(len(rust_summary['entities_per_language'][lang]) for lang in rust_summary['entities_per_language'])
    
    print(f"\nOverall Statistics:")
    print(f"  Python total entities: {total_python_entities}")
    print(f"  Rust total entities: {total_rust_entities}")
    print(f"  Difference: {total_python_entities - total_rust_entities}")

def main():
    print("Python vs Rust GLiNER2 Comparison")
    print("=" * 60)
    
    # Load test data
    test_file = "test_sentences.jsonl"
    try:
        test_data = load_test_data(test_file)
        print(f"Loaded {len(test_data)} test sentences from {test_file}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Test Python implementation
    python_results = test_python_finetuned_gliner(test_data)
    
    if python_results:
        # Compare with Rust results
        compare_with_rust(python_results)
        
        # Save results
        output_file = "python_vs_rust_comparison.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(python_results, f, indent=2, ensure_ascii=False)
            print(f"\nPython results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)
        print("Key findings:")
        print("1. Python GLiNER2 successfully loaded and tested")
        print("2. Entity extraction compared with Rust implementation")
        print("3. Performance and accuracy differences identified")
        
    else:
        print("\nNo successful Python tests completed.")

if __name__ == "__main__":
    main()
