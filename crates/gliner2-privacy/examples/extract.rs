//! PII extraction and redaction.
//!
//! ```sh
//! ORT_DYLIB_PATH=/path/libonnxruntime.so \
//! cargo run --release --example extract -p gliner2-privacy -- models/pii-onnx
//! ```

use gliner2_core::{SpanConfig, SpanEngine};
use gliner2_privacy::{Group, needs_anonymization, redact};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/pii-onnx".to_string());

    gliner2_core::init("gliner2-privacy-example");

    let t0 = Instant::now();
    let mut engine = SpanEngine::new(SpanConfig::new(&dir))?;
    println!(
        "loaded in {:.2}s — max_width {}, max_count {}, classifier {:?}",
        t0.elapsed().as_secs_f32(),
        engine.max_width(),
        engine.max_count(),
        engine.classifier_layout()
    );

    let text = "Il paziente Giuseppe Verdi, seguito dalla dottoressa Francesca Neri, \
                e' contattabile a g.verdi@example.it o al +39 340 1234567. \
                IBAN IT60X0542811101000000123456, residente a Milano.";

    // One task per group: a narrow schema lets the labels stop competing.
    let tasks = [Group::Person, Group::Contact, Group::Banking].map(Group::task);

    let t1 = Instant::now();
    let out = engine.extract(text, &tasks)?;
    println!(
        "\n{} entities in {:.1} ms  (needs anonymization: {})\n",
        out.entities.len(),
        t1.elapsed().as_secs_f32() * 1000.0,
        needs_anonymization(&out.entities, 0.5)
    );

    for e in &out.entities {
        println!(
            "  {:<32} {:<16} {:>6.2}%   bytes [{}..{})",
            format!("{:?}", e.text),
            e.label,
            e.score * 100.0,
            e.char_start,
            e.char_end
        );
    }

    println!("\n--- redacted ---\n{}", redact(text, &out.entities));
    Ok(())
}
