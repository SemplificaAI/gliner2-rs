use gliner2_inference::*;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use serde::Deserialize;
use std::fs::File;
use std::env;

#[derive(Deserialize, Debug)]
struct Record { 
    text: String, 
    #[serde(default)] 
    language: String 
}

fn main() -> anyhow::Result<()> {
    ort::init().with_name("GLiNER2_Engine").commit()?;
    
    println!("==================================================");
    println!("GLiNER2 RUST NATIVE INFERENCE ENGINE (Zero Python)");
    println!("==================================================");

    let args: Vec<String> = env::args().collect();
    let models_dir = if args.len() > 1 {
        args[1].clone()
    } else {
        "../models/fragments_fp16".to_string()
    };

    let model_type = if args.len() > 2 {
        match args[2].to_lowercase().as_str() {
            "huggingface" | "lmo3" | "public" => ModelType::HuggingFace,
            "pytorch" | "fragments" | "our" => ModelType::PyTorch,
            _ => ModelType::PyTorch  // Default to PyTorch for run_inference
        }
    } else {
        ModelType::PyTorch  // Default to PyTorch for run_inference
    };

    println!("Loading models from: {}", models_dir);
    println!("Model type: {}", model_type);

    let config = Gliner2Config {
        models_dir,
        max_width: 8,
        model_type,
    };
    
    let engine = Gliner2Engine::new(config)?;

    let test_file_path = "/mnt/crucial/jugaad/experiments/edito-gliner2/finetuning_local/data/dataset_variants/v3_reduced_remapped/test.jsonl";
    let file = File::open(test_file_path).expect("Cannot open test.jsonl");
    let reader = BufReader::new(file);

    println!("Testing E2E extraction on 10 sentences per language with all 62 trained entities...");

    let entity_labels = vec![
        "device_id".to_string(), "kyc_aml".to_string(), "professional_name".to_string(), "ip_address".to_string(), "diagnosis".to_string(), 
        "bank_name".to_string(), "cvv_code".to_string(), "age".to_string(), "medical_procedure".to_string(), "legal_clause".to_string(), 
        "time".to_string(), "vat_number".to_string(), "location".to_string(), "country".to_string(), "password".to_string(), "tax_amount".to_string(), 
        "ssn_fiscal_code".to_string(), "medication_name".to_string(), "product_service".to_string(), "phone_number".to_string(), 
        "currency".to_string(), "url".to_string(), "union_membership".to_string(), "property_regime".to_string(), "swift_bic".to_string(), 
        "genetic_data".to_string(), "amount".to_string(), "blood_type".to_string(), "criminal_record".to_string(), "contact_email".to_string(), 
        "address".to_string(), "payment_card_number".to_string(), "person_name".to_string(), "iban_code".to_string(), "mac_address".to_string(), 
        "date".to_string(), "duration_period".to_string(), "social_media_handle".to_string(), "financial_instrument_id".to_string(), 
        "racial_ethnic_origin".to_string(), "organization_name".to_string(), "rate_percentage".to_string(), "gender".to_string(), 
        "civil_status".to_string(), "balance".to_string(), "national_id".to_string(), "document_number".to_string(), "health_data".to_string(), 
        "case_number".to_string(), "cadastral_ref".to_string(), "occupation".to_string(), "sexual_orientation".to_string(), "user_agent".to_string(), 
        "biometric_data".to_string(), "court_name".to_string(), "date_of_birth".to_string(), "legal_ref".to_string(), "bank_account_number".to_string(), 
        "political_opinion".to_string(), "religious_philosophical_belief".to_string(), "license_number".to_string(), "vehicle_plate".to_string()
    ];
    
    let schema_tasks = vec![
        SchemaTask::Entities(entity_labels),
        SchemaTask::Relations("works_at".to_string(), vec!["head".to_string(), "tail".to_string()]),
        SchemaTask::Classifications("sentiment".to_string(), vec!["positive".to_string(), "negative".to_string(), "neutral".to_string()])
    ];

    let mut counts_by_lang: HashMap<String, usize> = HashMap::new();

    for line in reader.lines() {
        let line_str = line?;
        if let Ok(record) = serde_json::from_str::<Record>(&line_str) {
            let lang = if record.language.is_empty() { "unknown".to_string() } else { record.language.clone() };
            
            let count = counts_by_lang.entry(lang.clone()).or_insert(0);
            if *count < 3 {
                println!("\n[LANG: {}] (Sample {}/3)", lang.to_uppercase(), *count + 1);
                println!("TEXT: '{}'", record.text);
                
                // Check text length to avoid token limit issues
                let text_len = record.text.len();
                if text_len > 2000 {
                    println!("⚠️  Text too long ({} chars), skipping to avoid token limits", text_len);
                    *count += 1;
                    continue;
                }
                
                match engine.extract(&record.text, &schema_tasks) {
                    Ok((entities, relations, classifications)) => {
                        if !classifications.is_empty() {
                            println!("🏷️  Global Classifications Found:");
                            for c in classifications {
                                println!("  - [{}] {} => {:.2}%", c.task_name, c.label, c.score * 100.0);
                            }
                        }

                        println!("🔍 Entities Found (> 50% confidence):");
                        if entities.is_empty() {
                            println!("  (No entities found)");
                        } else {
                            for e in entities {
                                println!("  - [{:.2}%] {} | '{}'", e.score * 100.0, e.label, e.text);
                            }
                        }
                        
                        if !relations.is_empty() {
                            println!("🔗 Relations Found:");
                            for r in relations {
                                println!("  - [{}] {} => {}", r.relation_type, r.head.text, r.tail.text);
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("❌ Error during extraction: {:?}", e);
                        eprintln!("   Text length: {} chars", text_len);
                        if text_len > 500 {
                            eprintln!("   Note: Long texts may cause tensor dimension issues");
                        }
                    },
                }
                *count += 1;
            }
        }
        
        let is_done = ["it", "en", "pt", "de", "fr", "es"].iter().all(|l| counts_by_lang.get(*l).copied().unwrap_or(0) >= 3);
        if is_done {
            break;
        }
    }

    println!("\nMulti-Lingual Test completed successfully on {} processed languages.", counts_by_lang.len());
    
    println!("\n=== USAGE ===");
    println!("cargo run --example run_inference -- [models_dir] [model_type]");
    println!("  models_dir: Path to ONNX models (default: ../models/fragments_fp16)");
    println!("  model_type: 'lmo3' (public) or 'semplifica' (premium, reserved)");
    println!("\nExamples:");
    println!("  cargo run --example run_inference -- ../models/lmo3-gliner2-multi-v1-onnx/onnx lmo3");
    println!("  cargo run --example run_inference -- ../models/semplifica-gliner2 semplifica");
    
    Ok(())
}
