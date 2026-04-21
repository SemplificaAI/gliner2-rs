use gliner2_inference::*;
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
    println!("GLiNER2 RUST NATIVE INFERENCE ENGINE - Simple Test");
    println!("==================================================");

    let args: Vec<String> = env::args().collect();
    let models_dir = if args.len() > 1 {
        args[1].clone()
    } else {
        "../models/fastino_gliner2_multi_v1_fp16".to_string()
    };

    let model_type = if args.len() > 2 {
        match args[2].to_lowercase().as_str() {
            "huggingface" | "public" => ModelType::HuggingFace,
            "pytorch" | "fragments" | "our" => ModelType::PyTorch,
            _ => ModelType::HuggingFace  // Default to HuggingFace
        }
    } else {
        ModelType::HuggingFace  // Default to HuggingFace
    };

    println!("Loading models from: {}", models_dir);
    println!("Model type: {}", model_type);

    let config = Gliner2Config {
        models_dir,
        max_width: 8,
        model_type,
    };
    
    let engine = Gliner2Engine::new(config)?;

    let test_file_path = "test_sentences.jsonl";
    let file = File::open(test_file_path).expect("Cannot open test_sentences.jsonl");
    let reader = BufReader::new(file);

    println!("\nTesting GLiNER2 on classic PII/PII sentences...");

    // Core PII entities for testing
    let entity_labels = vec![
        "person_name".to_string(), "organization_name".to_string(), "location".to_string(),
        "date".to_string(), "email".to_string(), "phone_number".to_string(), "address".to_string(),
        "company_id".to_string(), "currency".to_string(), "amount".to_string()
    ];
    
    let schema_tasks = vec![
        SchemaTask::Entities(entity_labels),
        SchemaTask::Classifications("sentiment".to_string(), vec!["positive".to_string(), "negative".to_string(), "neutral".to_string()])
    ];

    for line in reader.lines() {
        let line_str = line?;
        if let Ok(record) = serde_json::from_str::<Record>(&line_str) {
            let lang = record.language.to_uppercase();
            
            println!("\n[{}] {}", lang, record.text);
            println!("{}", "-".repeat(60));
            
            match engine.extract(&record.text, &schema_tasks) {
                Ok((entities, relations, classifications)) => {
                    // Classifications
                    if !classifications.is_empty() {
                        println!("Sentiment:");
                        for c in classifications {
                            println!("  {} => {:.1}%", c.label, c.score * 100.0);
                        }
                    }

                    // Entities
                    if !entities.is_empty() {
                        println!("Entities:");
                        for e in entities {
                            println!("  [{:.1}%] {} | '{}'", e.score * 100.0, e.label, e.text);
                        }
                    } else {
                        println!("No entities found");
                    }
                    
                    // Relations (usually empty in this test)
                    if !relations.is_empty() {
                        println!("Relations:");
                        for r in relations {
                            println!("  [{}] {} => {}", r.relation_type, r.head.text, r.tail.text);
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Error: {:?}", e);
                }
            }
        }
    }
    
    println!("\n=== USAGE ===");
    println!("cargo run --example test_simple -- [models_dir] [model_type]");
    println!("  models_dir: Path to ONNX models (default: ../models/fastino_gliner2_multi_v1_fp16)");
    println!("  model_type: 'fastino' (public) or 'semplifica' (premium, reserved)");
    
    Ok(())
}
