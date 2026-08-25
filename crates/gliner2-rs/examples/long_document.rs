//! Extraction over a document longer than the model's largest length bucket.
//!
//! ```sh
//! ORT_DYLIB_PATH=… cargo run --release --example long_document -p gliner2-rs -- <models_dir> [words]
//! ```

use gliner2_rs::{SchemaTask, SpanConfig, SpanEngine};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).expect("models dir");
    let target: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1200);
    gliner2_rs::init("gliner2-long-document");

    // A document with known entities scattered through it, padded to length so
    // the boundary head's largest bucket is genuinely exceeded.
    let filler = "Il presente accordo disciplina i rapporti tra le parti e resta valido \
                  fino a revoca scritta di una di esse. ";
    let mut text = String::new();
    let people = ["Mario Rossi", "Giulia Bianchi", "Ahmed Hassan", "Sofia Ricci"];
    let places = ["Milano", "Napoli", "Berlino", "Reggio Emilia"];
    let mut i = 0usize;
    while text.split_whitespace().count() < target {
        text.push_str(filler);
        if i % 3 == 0 {
            text.push_str(&format!(
                "Il signor {} risiede a {} ed e' reperibile all'indirizzo p{}@example.com. ",
                people[i % people.len()],
                places[i % places.len()],
                i
            ));
        }
        i += 1;
    }
    let words = text.split_whitespace().count();
    println!("documento: {words} parole, {} byte", text.len());

    let mut engine = SpanEngine::new(SpanConfig::new(&dir))?;
    println!("max_width {}, max_count {}", engine.max_width(), engine.max_count());

    let tasks = vec![SchemaTask::Entities(vec![
        "person".into(),
        "location".into(),
        "email".into(),
    ])];

    // The single-call probe is skipped on request: a device OOM here leaves the
    // ORT arena fragmented, and the chunked run that follows would then fail
    // for a reason that has nothing to do with chunking.
    if std::env::var("SKIP_PROBE").is_err() {
    print!("\nextract()      -> ");
    match engine.extract(&text, &tasks) {
        Ok(o) => println!("{} entita'", o.entities.len()),
        Err(e) => println!("errore: {e}"),
    }
    }

    let t0 = Instant::now();
    let out = engine.extract_long(&text, &tasks)?;
    println!(
        "extract_long() -> {} entita' in {:.0} ms",
        out.entities.len(),
        t0.elapsed().as_secs_f32() * 1000.0
    );

    let mut by_field: std::collections::BTreeMap<&str, usize> = Default::default();
    for mm in &out.entities {
        *by_field.entry(mm.label.as_str()).or_default() += 1;
    }
    for (f, n) in &by_field {
        println!("   {f:<10} {n}");
    }
    println!("\nprime cinque:");
    for mm in out.entities.iter().take(5) {
        println!(
            "   {:<26} {:<10} {:>6.2}%  bytes [{}..{})",
            format!("{:?}", mm.text),
            mm.label,
            mm.score * 100.0,
            mm.char_start,
            mm.char_end
        );
    }
    // Offsets must index the original document, not a window.
    let bad = out
        .entities
        .iter()
        .filter(|mm| text.get(mm.char_start..mm.char_end) != Some(mm.text.as_str()))
        .count();
    println!("\noffset che non combaciano col documento: {bad}");
    std::process::exit(0);
}
