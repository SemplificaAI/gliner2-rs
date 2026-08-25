//! How much does the host-side f32<->f16 conversion cost?
//!
//! `float_tensor` and `take_float` convert element by element on the CPU when a
//! fragment declares FP16 I/O. This measures that loop against the tensor sizes
//! the span pipeline actually moves.

use half::f16;
use std::time::Instant;

fn main() {
    for (label, n) in [
        ("encoder hidden [1,150,768]", 150 * 768),
        ("span_embeddings [1,70,8,768]", 70 * 8 * 768),
        ("entity_scores [20,70,8,5]", 20 * 70 * 8 * 5),
    ] {
        let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();

        let t = Instant::now();
        let half: Vec<f16> = src.iter().copied().map(f16::from_f32).collect();
        let to_half = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let back: Vec<f32> = half.iter().map(|v| v.to_f32()).collect();
        let to_float = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{label:<32} {n:>9} elems   f32->f16 {to_half:>7.3}ms   f16->f32 {to_float:>7.3}ms   round trip {:>7.3}ms",
            to_half + to_float
        );
        std::hint::black_box(back);
    }
}
