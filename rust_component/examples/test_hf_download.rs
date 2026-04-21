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

    println!("\nModelli scaricati e motore inizializzato con successo!");

    let text = "Il signor Mario Rossi vive a Roma e lavora per Semplifica s.r.l. dal 2020.";
    println!("\nTesto di prova: '{}'", text);
    
    let schema_tasks = vec![
        SchemaTask::Entities(vec!["person".to_string(), "organization".to_string(), "location".to_string(), "date".to_string()])
    ];

    match engine.extract(text, &schema_tasks) {
        Ok((entities, _, _)) => {
            if !entities.is_empty() {
                println!("\nEntità trovate:");
                for e in entities {
                    println!("  [{:.1}%] {} | '{}'", e.score * 100.0, e.label, e.text);
                }
            } else {
                println!("Nessuna entità trovata.");
            }
        },
        Err(e) => {
            eprintln!("Error: {:?}", e);
        }
    }
    
    Ok(())
}
