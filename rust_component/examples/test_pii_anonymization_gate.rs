use gliner2_inference::*;
use serde_json::json;
use std::env;

fn build_engine() -> anyhow::Result<Gliner2Engine> {
    let repo_id = "SemplificaAI/gliner2-privacy-filter-PII-multi";
    let subfolder = Some("fp16_v2");
    let model_type = ModelType::HuggingFace;

    if let Ok(models_dir) = env::var("PII_MODELS_DIR") {
        println!("Using local exported ONNX fragments from: {}", models_dir);
        Gliner2Engine::new(Gliner2Config {
            models_dir,
            max_width: 8,
            model_type,
        })
    } else {
        println!("Downloading models from: {}/{}", repo_id, subfolder.unwrap_or(""));
        Gliner2Engine::from_pretrained(repo_id, subfolder, model_type)
    }
}

fn main() -> anyhow::Result<()> {
    ort::init().with_name("GLiNER2_PII_Anonymization_Gate").commit()?;

    let engine = build_engine()?;

    let threshold: f32 = env::var("PII_SCORE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.50);

    let labels = vec![
        "person".to_string(),
        "organization".to_string(),
        "email".to_string(),
        "phone".to_string(),
        "phone_number".to_string(),
        "address".to_string(),
        "date".to_string(),
        "fiscal_code".to_string(),
        "vat_number".to_string(),
        "iban".to_string(),
    ];

    let schema_tasks = vec![SchemaTask::Entities(labels)];

    let samples = vec![
        "Please contact Maria Jensen at maria.jensen@example.dk or +45 20 12 34 56.",
        "The package has shipped and should arrive tomorrow afternoon.",
    ];

    for text in samples {
        let (entities, _, _) = engine.extract(text, &schema_tasks, None)?;

        let hits: Vec<ExtractedEntity> = entities
            .into_iter()
            .filter(|e| e.score >= threshold)
            .collect();

        let needs_anonymization = !hits.is_empty();
        let redacted_text = if needs_anonymization {
            mask_pii_text(text, &hits)
        } else {
            text.to_string()
        };

        let payload = json!({
            "text": text,
            "threshold": threshold,
            "needs_anonymization": needs_anonymization,
            "matches": hits.iter().map(|e| json!({
                "label": e.label,
                "score": e.score,
                "text": e.text,
                "start_char": e.start_char,
                "end_char": e.end_char
            })).collect::<Vec<_>>(),
            "redacted_text": redacted_text
        });

        println!("{}", serde_json::to_string_pretty(&payload)?);
        println!("--------------------------------------------------");
    }

    Ok(())
}
