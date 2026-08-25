//! Prints every run individually, so the warm-up curve is visible.
//!
//! The CUDA provider pays for cuDNN algorithm search, arena growth and kernel
//! module loading on first use, and this pipeline has eight sessions, each
//! warming separately. A single warm-up run before timing may not be enough —
//! this shows how many it actually takes.
//!
//! ```sh
//! ORT_DYLIB_PATH=… GLINER2_DEVICE=cuda:1 \
//! cargo run --release --example warmup -p gliner2-core -- <models_dir> [runs]
//! ```

use gliner2_rs::{SchemaTask, SpanConfig, SpanEngine};
use std::time::Instant;

const TEXT: &str = "Il signor Mario Rossi vive a Roma e lavora per Jugaad s.r.l. dal 2020. \
L'azienda, fondata da Giuseppe Verdi, ha recentemente aperto una nuova sede a Milano, vicino al Duomo. \
Nel 2023, il fatturato e' cresciuto del 45%, spinto dalle nuove tecnologie di intelligenza artificiale. \
La dottoressa Francesca Bianchi, CEO della divisione europea, ha tenuto una conferenza a Parigi \
il 15 Maggio 2024, annunciando partnership strategiche con Microsoft e Google.";

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: warmup <models_dir> [runs]");
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);

    gliner2_rs::init("warmup");
    let t0 = Instant::now();
    let mut engine = SpanEngine::new(SpanConfig::new(&dir))?;
    println!("session build: {:.2}s", t0.elapsed().as_secs_f64());

    let tasks = vec![SchemaTask::Entities(vec![
        "person".into(), "organization".into(), "location".into(),
        "date".into(), "event".into(),
    ])];

    let mut times = Vec::with_capacity(runs);
    for i in 0..runs {
        let t = Instant::now();
        let _ = engine.extract(TEXT, &tasks)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        println!("  run {:>3}  {:>9.2} ms", i + 1, ms);
    }

    // where does it settle? compare each run against the median of the tail
    let tail: Vec<f64> = times[times.len() / 2..].to_vec();
    let mut sorted = tail.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let steady = sorted[sorted.len() / 2];
    let settled = times
        .iter()
        .position(|t| *t <= steady * 1.25)
        .map(|i| i + 1)
        .unwrap_or(0);

    println!();
    println!("steady-state median (2nd half): {steady:.2} ms");
    println!("first run within 25% of it    : #{settled}");
    println!("cost of run 1 over steady     : {:.2} ms ({:.1}x)",
             times[0] - steady, times[0] / steady);
    Ok(())
}
