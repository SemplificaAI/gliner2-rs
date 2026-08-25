//! Compiles every code sample in the crate README, so a signature that drifts
//! breaks the build instead of misleading a reader.
#![allow(unused_variables, dead_code, unreachable_code)]

use gliner2_rs::chain::ExecutionMode;
use gliner2_rs::chunker::Chunker;
use gliner2_rs::guardrails::{prompt_moderation_schema, response_moderation_schema};
use gliner2_rs::privacy::{Group, all_labels_task, needs_anonymization, redact, redact_with};
use gliner2_rs::{SchemaTask, SpanConfig, SpanEngine, hub};

fn main() -> anyhow::Result<()> {
    return Ok(()); // compile-only

    gliner2_rs::init("my-app");
    let mut engine = SpanEngine::new(SpanConfig::new("models/pii-onnx"))?;

    let tasks = vec![SchemaTask::Entities(vec!["person".into(), "email".into()])];
    let out = engine.extract("Mario Rossi — m.rossi@example.it", &tasks)?;
    for e in &out.entities {
        println!("{:?} {} {:.1}%", e.text, e.label, e.score * 100.0);
    }

    let tasks = vec![
        SchemaTask::Entities(vec!["person".into(), "organization".into()]),
        SchemaTask::Relations("works_for".into(), vec!["head".into(), "tail".into()]),
        SchemaTask::classification("tone", vec!["formal".into(), "casual".into()]),
        SchemaTask::multi_label_classification("topics", vec!["legal".into(), "medical".into()]),
    ];
    for c in out.verdict("tone", 0.5) {
        println!("{} {:.1}%", c.label, c.score * 100.0);
    }

    let document = String::new();
    let out = engine.extract_long(&document, &tasks)?;
    let params = gliner2_rs::InferenceParams::default();
    let out = engine.extract_long_with(&document, &tasks, &params, Chunker::new(256, 48)?)?;

    let cfg = SpanConfig::new("models/pii-onnx").with_execution(ExecutionMode::IoBinding);
    let _ = engine.execution();

    let cfg = SpanConfig::new("models/pii-onnx").or_download(hub::PRIVACY_PII_MULTI);
    let _ = SpanConfig::from_hub(hub::GUARDRAILS_PII_MULTI);
    let _ = hub::Model::new("acme/my-export", hub::Layout::Flat);

    let text = "…";
    let ptasks = [Group::Person, Group::Contact, Group::Banking].map(Group::task);
    let out = engine.extract(text, &ptasks)?;
    if needs_anonymization(&out.entities, 0.5) {
        println!("{}", redact(text, &out.entities));
    }
    let _ = redact_with(text, &out.entities, |label| format!("<{label}>"));
    let _ = all_labels_task();

    let user_prompt = "…";
    let out = engine.extract(user_prompt, &prompt_moderation_schema())?;
    for c in out.verdict("prompt_injection", 0.5) {
        let _ = c;
    }
    let _ = response_moderation_schema();
    Ok(())
}
