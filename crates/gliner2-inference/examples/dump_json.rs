//! Runs a fixed suite of cases and prints JSON, so the engine's output can be
//! diffed against a PyTorch reference.
//!
//! ```sh
//! ORT_DYLIB_PATH=… cargo run --release --example dump_json -- <models_dir> <cases.json>
//! ```

use gliner2_inference::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    text: String,
    entities: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: dump_json <models_dir> <cases.json>");
    let cases_path = args.next().expect("usage: dump_json <models_dir> <cases.json>");

    ort::init().with_name("dump_json").commit();
    let engine = Gliner2Engine::new(Gliner2Config {
        models_dir: dir,
        max_width: 8,
        model_type: ModelType::HuggingFace,
    })?;

    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&cases_path)?)?;
    let params = InferenceParams { threshold: 0.5, flat_ner: false };

    let mut out = Vec::new();
    for case in &cases {
        let tasks = vec![SchemaTask::Entities(case.entities.clone())];
        let (entities, _, _) = engine.extract(&case.text, &tasks, Some(params.clone()))?;
        let mut rows: Vec<serde_json::Value> = entities
            .iter()
            .map(|e| {
                serde_json::json!({
                    "text": e.text,
                    "label": e.label,
                    "start": e.start_char,
                    "end": e.end_char,
                    "score": (e.score * 1e4).round() / 1e4,
                })
            })
            .collect();
        rows.sort_by(|a, b| {
            (a["start"].as_u64(), a["end"].as_u64(), a["label"].as_str())
                .cmp(&(b["start"].as_u64(), b["end"].as_u64(), b["label"].as_str()))
        });
        out.push(serde_json::json!({ "name": case.name, "entities": rows }));
    }
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
