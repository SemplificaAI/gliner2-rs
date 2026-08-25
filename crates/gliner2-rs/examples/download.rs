//! Loading a model without having downloaded it first.
//!
//! ```sh
//! ORT_DYLIB_PATH=/path/libonnxruntime.so \
//! cargo run --release --example download -p gliner2-rs -- models/pii-onnx
//! ```
//!
//! Pass the directory you want the export in. If it already holds one it is
//! used untouched and nothing is fetched; if it does not, the export is pulled
//! from the Hub before the engine starts.

use gliner2_rs::privacy::Group;
use gliner2_rs::{SpanConfig, SpanEngine, hub};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "models/pii-onnx".to_string());
    gliner2_rs::init("gliner2-download-example");

    let present = std::path::Path::new(&dir).exists();
    println!("{dir}: {}", if present { "present, no fetch" } else { "absent, fetching" });

    let t0 = Instant::now();
    let mut engine = SpanEngine::new(SpanConfig::new(&dir).or_download(hub::PRIVACY_PII_MULTI))?;
    println!("ready in {:.1}s", t0.elapsed().as_secs_f32());

    let text = "Giuseppe Verdi, g.verdi@example.it, Milano.";
    let tasks = [Group::Person, Group::Contact].map(Group::task);
    let out = engine.extract(text, &tasks)?;
    for e in &out.entities {
        println!("  {:<20} {:<12} {:>6.2}%", e.text, e.label, e.score * 100.0);
    }
    Ok(())
}
