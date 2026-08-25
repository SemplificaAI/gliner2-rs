//! Smoke check against a local export, used to validate the 0.5.1 processor fix.
use gliner2_inference::*;

fn main() -> anyhow::Result<()> {
    ort::init().with_name("local_check").commit()?;
    let dir = std::env::args().nth(1).expect("usage: local_check <models_dir>");

    let engine = Gliner2Engine::new(Gliner2Config {
        models_dir: dir,
        max_width: 8,
        model_type: ModelType::HuggingFace,
    })?;

    let text = "Mario Rossi lavora ad Apple a Cupertino e la sua email e' mario.rossi@example.com.";
    let tasks = vec![SchemaTask::Entities(vec![
        "person".to_string(),
        "organization".to_string(),
        "location".to_string(),
        "email".to_string(),
    ])];

    let (entities, _, _) = engine.extract(
        text,
        &tasks,
        Some(InferenceParams { threshold: 0.5, flat_ner: false }),
    )?;
    println!("{} entita'", entities.len());
    for e in &entities {
        println!("  {:<28} {:<14} {:>6.2}%", format!("{:?}", e.text), e.label, e.score * 100.0);
    }
    Ok(())
}
