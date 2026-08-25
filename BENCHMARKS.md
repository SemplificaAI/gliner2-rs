# Benchmarks

Measured 2026-08-25 on a **shared development machine** — other users on some
cores, a training job on the other GPU. The figures below are an indication of
magnitude, not a clean measurement; see the caveats.

Measured with `cargo run --release --example bench`. Read the caveats
before quoting any of it.

## Hardware

| | |
|---|---|
| CPU | AMD Ryzen 9 5900XT, 16 cores / 32 threads |
| GPU | NVIDIA RTX 3090, 24 GB (device 1) |
| ONNX Runtime | 1.25.1 CPU-only wheel; 1.23.2 GPU wheel for CUDA |
| CUDA | 12.8, cuDNN 9 |
| crate | `gliner2-core`, `ort` 2.0.0-rc.13, `load-dynamic` |

Device 0, an RTX 4090, was carrying an unrelated multi-day training job and was
deliberately left alone; `GLINER2_DEVICE=cuda:1` pins the benchmark to the 3090,
which was idle. The host itself is shared, so the CPU cores were not exclusively
available — which is why the CPU figures are the unreliable ones here, and the
GPU figures the more usable.

### About the card

An RTX 3090 is a 2020 consumer part, not a datacenter accelerator, and it is the
slower end of what you would deploy on. Its performance class sits close to an
**NVIDIA L4**, which is what a good deal of cloud inference actually runs on
today — GCP G2, AWS G6 and similar. The two are within the same range on FP32
throughput; the 3090 has substantially more memory bandwidth, the L4 draws a
fraction of the power.

For this workload the distinction matters less than it looks. The span pipeline
is bound by kernel launches and host round-trips across eight ONNX sessions, not
by arithmetic or bandwidth, so a faster card moves these numbers less than
implementing `IoBinding` would. Read the GPU figures as *what a realistic
inference instance gives you*, not as a ceiling — and not as a best case either.

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

`gliner2-privacy`, legacy export, 15 entities:

| device | precision | median | min | p95 | per entity |
|---|---|---|---|---|---|
| RTX 3090 | `fp32` | 27.1 ms | 23.7 ms | 30.4 ms | 1.81 ms |
| RTX 3090 | `fp16` | **24.9 ms** | 22.2 ms | 34.1 ms | 1.66 ms |
| RTX 3090 | `fp16_iobinding` | 25.1 ms | 22.6 ms | 38.4 ms | 1.67 ms |
| Ryzen 5900XT | `fp32` | 2697 ms | 1652 ms | 3337 ms | 180 ms |
| Ryzen 5900XT | `fp16` | 2676 ms | 959 ms | 3993 ms | 178 ms |
| Ryzen 5900XT | `fp16_iobinding` | 3443 ms | 1409 ms | 4060 ms | 230 ms |

Note that `fp32` and `fp16` swap places between the two models on GPU — `fp32`
wins on guardrails, `fp16` on privacy — and both gaps are around 10%, inside the
spread of a single configuration on this host. **That ordering is not a
finding.** What survives across both models is only the `fp16_iobinding` penalty
on CPU.

Load time is 8–16 s in every configuration, dominated by reading the 530 MB–1 GB
encoder from disk.

## Caveats, which are not small

**This is a shared development machine, not a benchmark rig.** Other users had
work on some of the cores, and the other GPU was carrying a training job for the
whole run. Load average held at 18 on 32 threads. Treat every figure here as an
**order-of-magnitude indication**, not a measurement.

The contention is visible in the numbers themselves: median 1679 ms against a
minimum of 564 ms is a factor of three on the same configuration, and that
spread is other people's work, not the engine. The GPU card itself was idle, so those figures are the
more usable of the two — but their p95 still carries host-side scheduling noise,
since feeding the GPU is CPU work. **Re-run on a quiet machine before quoting
any of this.**

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

## Correctness across devices

Speed is not the only thing that changes when you move to a GPU. The end-to-end
suites were re-run on both devices and every precision against the same PyTorch
reference:

| crate | device | `fp32` | `fp16` | `fp16_iobinding` |
|---|---|---|---|---|
| guardrails | CPU | 61/61 (0.0001) | 61/61 (0.0034) | 61/61 (0.0035) |
| guardrails | RTX 3090 | 61/61 (0.0001) | 61/61 (0.0036) | 61/61 (0.0035) |
| privacy | CPU | 58/58 (**0.0000**) | 58/58 (0.0023) | 58/58 (0.0021) |
| privacy | RTX 3090 | 58/58 (**0.0000**) | 58/58 (0.0022) | 58/58 (0.0021) |

Identical spans everywhere; brackets give the largest score delta. The CUDA
kernels agree with the CPU ones to within FP16 rounding — 0.0034 against 0.0036
on the same case — so the device is a performance decision, not an accuracy one.

The `fp32` privacy rows are exact: zero deviation from PyTorch at the fourth
decimal the harness records, on both devices. That is the cleanest evidence in
this document that the export itself is faithful, and it puts a floor under
every other row — whatever deviation the FP16 variants show is quantisation,
not a bug in the graphs.

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
