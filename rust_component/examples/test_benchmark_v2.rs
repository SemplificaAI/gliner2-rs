use gliner2_inference::{Gliner2EngineV2, Gliner2Config, SchemaTask, ModelType};

fn main() -> anyhow::Result<()> {
    ort::init().with_name("GLiNER2_Engine_V2").commit()?;

    let models_dir = "models/gliner2-multi-v1-onnx-v2";
    
    let config = Gliner2Config {
        models_dir: models_dir.to_string(),
        max_width: 8,
        model_type: ModelType::PyTorch,
    };

    println!("==================================================");
    println!("GLiNER2 RUST NATIVE - Benchmark V2 (IOBinding)");
    println!("==================================================");

    let engine = Gliner2EngineV2::new(config)?;

    let text = "La dottoressa Giulia Bianchi, nata a Milano il 15/05/1985, è stata recentemente assunta come Chief Technology Officer presso InnovaTech S.p.A., un'azienda leader con sede a Torino. Il suo indirizzo email è giulia.bianchi@innovatech.it. Giulia è estremamente felice e soddisfatta del suo nuovo ambiente lavorativo e non vede l'ora di iniziare!";

    let tasks = vec![
        SchemaTask::Entities(vec![
            "person".to_string(),
            "location".to_string(),
            "organization".to_string(),
            "date".to_string(),
            "role".to_string(),
            "email".to_string(),
        ]),
        SchemaTask::Classifications("sentiment".to_string(), vec!["positive".to_string(), "negative".to_string()]),
        SchemaTask::Relations("works_for".to_string(), vec!["head".to_string(), "tail".to_string()]),
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
    let time_per_entity = if num_entities > 0 { avg_duration / num_entities } else { std::time::Duration::new(0, 0) };

    println!("⏱️ Total Avg Time: {:?}", avg_duration);
    println!("⏱️ Avg Time per Entity ({} extracted): {:?}", num_entities, time_per_entity);

    Ok(())
}
