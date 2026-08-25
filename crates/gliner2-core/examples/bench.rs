//! Measures load time and per-sentence inference time.
//!
//! ```sh
//! ORT_DYLIB_PATH=… GLINER2_DEVICE=cuda GLINER2_PRECISION=fp16_iobinding \
//! cargo run --release --example bench -p gliner2-core -- <models_dir> [runs]
//! ```
//!
//! Reports the median rather than the mean: a single scheduling hiccup skews a
//! mean over a few dozen runs, and the median is what a caller actually feels.

use gliner2_core::{SchemaTask, SpanConfig, SpanEngine};
use std::time::Instant;

const TEXT: &str = "Il signor Mario Rossi vive a Roma e lavora per Jugaad s.r.l. dal 2020. \
L'azienda, fondata da Giuseppe Verdi, ha recentemente aperto una nuova sede a Milano, vicino al Duomo. \
Nel 2023, il fatturato e' cresciuto del 45%, spinto dalle nuove tecnologie di intelligenza artificiale. \
La dottoressa Francesca Bianchi, CEO della divisione europea, ha tenuto una conferenza a Parigi \
il 15 Maggio 2024, annunciando partnership strategiche con Microsoft e Google.";

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: bench <models_dir> [runs]");
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);

    gliner2_core::init("bench");

    let t0 = Instant::now();
    let mut engine = SpanEngine::new(SpanConfig::new(&dir))?;
    let load = t0.elapsed();

    let tasks = vec![SchemaTask::Entities(vec![
        "person".into(), "organization".into(), "location".into(),
        "date".into(), "event".into(),
    ])];

    // warm-up: the first run pays for lazy allocator and kernel setup
    let warm = engine.extract(TEXT, &tasks)?;
    let entities = warm.entities.len();

    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let _ = engine.extract(TEXT, &tasks)?;
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = samples[samples.len() / 2];
    let p95 = samples[((samples.len() as f64 * 0.95) as usize).min(samples.len() - 1)];
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;

    println!(
        "device={:<8} precision={:<16} load={:>6.2}s  median={:>8.2}ms  mean={:>8.2}ms  p95={:>8.2}ms  min={:>8.2}ms  entities={:>2}  per-entity={:>6.2}ms",
        std::env::var("GLINER2_DEVICE").unwrap_or_else(|_| "auto".into()),
        std::env::var("GLINER2_PRECISION").unwrap_or_else(|_| "auto".into()),
        load.as_secs_f64(), median, mean, p95, samples[0], entities,
        median / entities.max(1) as f64,
    );
    Ok(())
}
