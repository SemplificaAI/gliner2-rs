use gliner2_inference::*;
use std::time::Instant;
use std::env;

fn main() -> anyhow::Result<()> {
    ort::init().with_name("GLiNER2_Bench").commit()?;
    
    let force_cpu = env::var("FORCE_CPU").is_ok();
    println!("GLiNER2 RUST NATIVE - Benchmark & Test (Force CPU: {})", force_cpu);
    
    let engine = Gliner2Engine::from_pretrained(
        "SemplificaAI/gliner2-multi-v1-onnx",
        Some("fp32"),
        ModelType::HuggingFace
    )?;

    let text = "Il signor Mario Rossi vive a Roma e lavora per Semplifica s.r.l. dal 2020. \
    L'azienda, fondata da Giuseppe Verdi, ha recentemente aperto una nuova sede a Milano, vicino al Duomo. \
    Nel 2023, il fatturato è cresciuto del 45%, spinto dalle nuove tecnologie di intelligenza artificiale. \
    La dottoressa Francesca Bianchi, CEO della divisione europea, ha tenuto una conferenza a Parigi \
    il 15 Maggio 2024, annunciando partnership strategiche con Microsoft e Google.";

    // Estimated number of sentences in the text (4 logical sentences)
    let num_sentences = 4;

    let tasks = vec![
        SchemaTask::Entities(vec![
            "person".to_string(), 
            "organization".to_string(), 
            "location".to_string(), 
            "date".to_string(),
            "event".to_string()
        ])
    ];

    println!("Warm-up (1 run)...");
    let (entities, _, _) = engine.extract(text, &tasks)?;
    let num_entities = entities.len() as u32;
    
    println!("\n=== Correct Extraction ===");
    for e in &entities {
        println!("  [{:.1}%] {} | '{}'", e.score * 100.0, e.label, e.text);
    }
    
    println!("\n=== Benchmark (50 runs) ===");
    let num_runs = 50;
    let mut total_duration = std::time::Duration::new(0, 0);
    
    for i in 1..=num_runs {
        let start = std::time::Instant::now();
        let _ = engine.extract(text, &tasks)?;
        let duration = start.elapsed();
        total_duration += duration;
        if i % 10 == 0 || i == 1 {
            println!("  [Run {}/{}] completed in {:?}", i, num_runs, duration);
        }
    }
    
    let avg_duration = total_duration / num_runs as u32;
    let time_per_sentence = avg_duration / num_sentences;
    let time_per_entity = avg_duration / num_entities;

    println!("⏱️ Total Avg Time: {:?}", avg_duration);
    println!("⏱️ Avg Time per Sentence: {:?}", time_per_sentence);
    println!("⏱️ Avg Time per Entity ({} extracted): {:?}", num_entities, time_per_entity);

    Ok(())
}
