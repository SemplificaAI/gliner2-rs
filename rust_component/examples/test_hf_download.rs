use gliner2_inference::*;

fn main() -> anyhow::Result<()> {
    ort::init().with_name("GLiNER2_Engine_HF").commit()?;
    
    println!("==================================================");
    println!("GLiNER2 RUST NATIVE - HuggingFace Auto-Download");
    println!("==================================================");

    let repo_id = "SemplificaAI/gliner2-multi-v1-onnx";
    let subfolder = Some("fp16");
    let model_type = ModelType::HuggingFace;

    println!("Downloading models from: {}/{}", repo_id, subfolder.unwrap_or(""));
    
    let engine = Gliner2Engine::from_pretrained(repo_id, subfolder, model_type)?;

    println!("\nModels downloaded and engine initialized successfully!");

    let text = "La dottoressa Giulia Bianchi, nata a Milano il 15/05/1985, è stata recentemente assunta come Chief Technology Officer presso InnovaTech S.p.A., un'azienda leader con sede a Torino. Il suo indirizzo email è giulia.bianchi@innovatech.it. Giulia è estremamente felice e soddisfatta del suo nuovo ambiente lavorativo e non vede l'ora di iniziare!";
    println!("\nTesto di prova: '{}'", text);
    
    let schema_tasks = vec![
        SchemaTask::Entities(vec!["person".to_string(), "organization".to_string(), "location".to_string(), "date".to_string(), "role".to_string(), "email".to_string()]),
        SchemaTask::Classifications("sentiment".to_string(), vec!["positive".to_string(), "negative".to_string(), "neutral".to_string()]),
        SchemaTask::Classifications("topic".to_string(), vec!["business".to_string(), "health".to_string(), "politics".to_string()]),
        SchemaTask::Relations("works_for".to_string(), vec!["head".to_string(), "tail".to_string()]),
        SchemaTask::Relations("based_in".to_string(), vec!["head".to_string(), "tail".to_string()]),
    ];

    match engine.extract(text, &schema_tasks) {
        Ok((entities, relations, classifications)) => {
            println!("\n--- CLASSIFICATIONS ---");
            if !classifications.is_empty() {
                for c in classifications {
                    println!("  [Task: {}] {} => {:.1}%", c.task_name, c.label, c.score * 100.0);
                }
            } else {
                println!("Nessuna classificazione trovata.");
            }

            println!("\n--- ENTITIES ---");
            if !entities.is_empty() {
                for e in entities {
                    println!("  [{:.1}%] {} | '{}'", e.score * 100.0, e.label, e.text);
                }
            } else {
                println!("Nessuna entità trovata.");
            }

            println!("\n--- RELATIONS ---");
            if !relations.is_empty() {
                for r in relations {
                    println!("  [{}] '{}' => '{}'", r.relation_type, r.head.text, r.tail.text);
                }
            } else {
                println!("Nessuna relazione trovata.");
            }
        },
        Err(e) => {
            eprintln!("Error: {:?}", e);
        }
    }
    
    Ok(())
}
