# Benchmarks

Measured 2026-08-25 with `cargo run --release --example bench`. Read the caveats
before quoting any of it.

## Hardware

| | |
|---|---|
| CPU | AMD Ryzen 9 5900XT, 16 cores / 32 threads |
| GPU | NVIDIA RTX 3090, 24 GB (device 1) |
| ONNX Runtime | 1.25.1 CPU-only wheel; 1.23.2 GPU wheel for CUDA |
| CUDA | 12.8, cuDNN 9 |
| crate | `gliner2-core`, `ort` 2.0.0-rc.13, `load-dynamic` |

Device 0, an RTX 4090, was running an unrelated multi-day training job and was
deliberately left alone. `GLINER2_DEVICE=cuda:1` pins the benchmark to the idle
card.

## Method

One paragraph of Italian text, ~90 words, five entity labels, 13 entities found.
One warm-up run — the first pays for lazy allocator and kernel setup — then 20
timed runs on GPU and 12 on CPU. The **median** is reported because a single
scheduling hiccup skews a mean over a few dozen runs, alongside the minimum,
which is the cleanest estimate of uncontended time.

## Results

`gliner2-guardrails`, flat export:

| device | precision | median | min | p95 | per entity |
|---|---|---|---|---|---|
| RTX 3090 | `fp32` | **26.4 ms** | 23.0 ms | 35.6 ms | 2.03 ms |
| RTX 3090 | `fp16` | 28.2 ms | 24.2 ms | 60.3 ms | 2.17 ms |
| RTX 3090 | `fp16_iobinding` | 29.2 ms | 24.8 ms | 36.7 ms | 2.24 ms |
| Ryzen 5900XT | `fp32` | 1679 ms | 564 ms | 3163 ms | 129 ms |
| Ryzen 5900XT | `fp16` | 1717 ms | 921 ms | 3882 ms | 132 ms |
| Ryzen 5900XT | `fp16_iobinding` | 3200 ms | 1271 ms | 3556 ms | 246 ms |

`gliner2-privacy`, legacy export. The `fp32_v2/` folder was not downloaded
locally, so those rows are **absent, not zero**:

| device | precision | median | min | p95 | per entity |
|---|---|---|---|---|---|
| RTX 3090 | `fp16` | **24.9 ms** | 22.2 ms | 34.1 ms | 1.66 ms |
| RTX 3090 | `fp16_iobinding` | 25.1 ms | 22.6 ms | 38.4 ms | 1.67 ms |
| Ryzen 5900XT | `fp16` | 2676 ms | 959 ms | 3993 ms | 178 ms |
| Ryzen 5900XT | `fp16_iobinding` | 3443 ms | 1409 ms | 4060 ms | 230 ms |

Load time is 8–16 s in every configuration, dominated by reading the 530 MB–1 GB
encoder from disk.

## Caveats, which are not small

**The machine was contended throughout.** Load average held at 18 on 32 threads
for the whole run — dozens of PyTorch dataloader workers from the training job
on device 0. The CPU figures are therefore an upper bound on a busy machine, not
a clean measurement: median 1679 ms against a minimum of 564 ms is a factor of
three, and that spread is contention, not the engine. **Re-run on an idle
machine before quoting CPU numbers anywhere.** GPU figures are much less
affected — the card itself was idle — but their p95 still carries host-side
scheduling noise.

**One text, one schema.** Timings scale with sequence length and with the number
of schema tasks, since each task costs a `schema_gather` plus a `count_lstm` and
a `scorer` pass. Nothing here says how the engine behaves on a long document or
a wide schema.

## What the numbers actually show

**GPU is worth it.** Comparing minima, the least contaminated statistic:
23 ms against 564 ms, roughly 24×. Comparing medians it looks like 64×, but that
number is mostly measuring how busy the CPU was.

**`fp16_iobinding` does not pay, and on CPU it hurts.** It is the slowest
variant on GPU by ~10% and 1.9× slower than `fp32` on CPU. The mechanism is at
least partly ours: with FP16 graph I/O, `float_tensor` and `take_float` convert
element by element in a scalar Rust loop at every fragment boundary, where
`keep_io_types=True` leaves that to ONNX Runtime. Measured with
`cargo run --release --example cast_cost -p gliner-core`:

| tensor | elements | f32→f16 | f16→f32 | round trip |
|---|---|---|---|---|
| encoder hidden `[1,150,768]` | 115k | 0.475 ms | 0.660 ms | 1.14 ms |
| `span_embeddings [1,70,8,768]` | 430k | 1.450 ms | 1.931 ms | **3.38 ms** |
| `entity_scores [20,70,8,5]` | 56k | 0.127 ms | 0.176 ms | 0.30 ms |

That accounts for the ~3 ms GPU gap. It does **not** account for the ~1500 ms
CPU gap, and the honest position is that the rest is unexplained: the plausible
cause is FP16 kernel coverage in ONNX Runtime's CPU provider, but that is a
hypothesis and would need profiling to confirm.

Until `IoBinding` is implemented, `_fp16_iobinding` buys halved weight memory and
nothing else. Pick `fp32` on GPU and `fp32` or `fp16` on CPU.

**FP16 is not the problem; the pipeline shape is.** On Ampere and Ada, FP16
tensor throughput is roughly double FP32 — the hardware prefers it. It does not
show here because the span architecture runs as eight separate ONNX sessions
with a host round-trip between each, so wall time is dominated by kernel launch
and synchronisation rather than by matrix multiplication. Tensor Cores never
become the bottleneck, so their advantage never appears, and only the conversion
cost remains. Fixing that means implementing `IoBinding`, not changing precision.

## Reproducing

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
GLINER2_DEVICE=cuda:1 GLINER2_PRECISION=fp32 \
cargo run --release --example bench -p gliner2-core -- models/guardrails-pii-multi-onnx 20
```

The GPU run needs a runtime that ships the CUDA provider — the plain
`onnxruntime` wheel does not. `pip install onnxruntime-gpu` provides
`libonnxruntime_providers_cuda.so`; point `LD_LIBRARY_PATH` at its directory
along with the CUDA 12 libraries.
