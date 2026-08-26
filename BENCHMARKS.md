# Benchmarks

Measured 2026-08-25 on a **shared development machine** — other users on some
cores, a training job on the other GPU, load average 17–19 throughout. Treat
every figure as an indication of magnitude, not a measurement.

Each model is reported on its own. They are different checkpoints with different
label vocabularies finding different numbers of entities; nothing here is a
comparison between them.

## Setup

| | |
|---|---|
| CPU | AMD Ryzen 9 5900XT, 16 cores / 32 threads |
| GPU | NVIDIA RTX 3090, 24 GB, device 1, idle |
| ONNX Runtime | 1.25.1 CPU wheel; 1.23.2 GPU wheel for CUDA |
| CUDA | 12.8, cuDNN 9 |
| crate | `gliner2-rs`, `ort` 2.0.0-rc.13, `load-dynamic` |

An RTX 3090 is a 2020 consumer part whose performance class sits close to an
**NVIDIA L4** — what a good deal of cloud inference runs on today. Comparable
FP32 throughput; the 3090 has substantially more memory bandwidth and draws
several times the power. Read the GPU figures as a realistic inference instance,
not a ceiling.

**Method.** One Italian paragraph, ~90 words, five entity labels. Five warm-up
runs, then 25 timed on GPU and 15 on CPU, with a 2 ms yield between iterations —
see *The harness was wrong* below for why that is not cosmetic. Median reported
alongside the minimum, which is the cleanest estimate of uncontended time.

## GLiNER2-Guardrails-PII-Multi

Flat export, 13 entities found.

| device | precision | median | min | p95 |
|---|---|---|---|---|
| RTX 3090 | `fp32` | 26.5 ms | 24.5 ms | 28.3 ms |
| RTX 3090 | `fp16` | 55.4 ms | 26.1 ms | 179 ms |
| RTX 3090 | `fp16_iobinding` | **20.6 ms** | 18.6 ms | 25.4 ms |
| Ryzen 5900XT | `fp32` | **2608 ms** | 884 ms | 3596 ms |
| Ryzen 5900XT | `fp16` | 2641 ms | 1035 ms | 4868 ms |
| Ryzen 5900XT | `fp16_iobinding` | 3165 ms | 1559 ms | 4529 ms |

## gliner2-privacy-filter-PII-multi

Legacy export, 15 entities found.

| device | precision | median | min | p95 |
|---|---|---|---|---|
| RTX 3090 | `fp32` | **24.4 ms** | 22.2 ms | 26.5 ms |
| RTX 3090 | `fp16` | 25.7 ms | 21.3 ms | 29.2 ms |
| RTX 3090 | `fp16_iobinding` | 30.9 ms | 26.4 ms | 35.4 ms |
| Ryzen 5900XT | `fp32` | 3017 ms | 1455 ms | 3752 ms |
| Ryzen 5900XT | `fp16` | **2456 ms** | 1110 ms | 3839 ms |
| Ryzen 5900XT | `fp16_iobinding` | 3212 ms | 1161 ms | 4509 ms |

## What holds and what does not

**The GPU gap holds.** Two orders of magnitude on both models, on medians and on
minima alike. That is the one conclusion this host is capable of supporting.

**The precision ordering does not.** The two models disagree: `fp16_iobinding` is
the fastest GPU variant on guardrails and the slowest on privacy, while `fp32`
does the opposite. The spread between precisions is 20–30%, and this host has
been shown to move the same configuration by more than that. **Do not choose a
precision from this table** — measure on your own hardware, with your own text
lengths and schema width.

Earlier revisions of this file claimed `fp16_iobinding` was consistently the
slowest variant. That was an artefact of the broken harness described below, and
the claim is withdrawn.

**CPU figures are the least reliable.** The GPU card was idle; the CPU cores were
not. Medians sit at two to three times their own minima, and that spread is other
people's work.

## The harness was wrong

Worth recording, because the failure was invisible and cost two retracted
conclusions.

The first harness ran one warm-up and then timed back-to-back iterations in a
tight loop. On this contended host that produced numbers up to **100× too
large**, and reproducibly so: the same model, in the same minute, measured 13 ms
with a variant that printed each run and 1700 ms with the one that did not. The
only difference was a `println!` inside the loop.

The explanation that fits: the CUDA synchronisation spins, and a process spinning
in a tight loop on an oversubscribed machine gets descheduled while it waits. A
syscall per iteration yields the CPU and breaks that pattern. Pinning to eight
dedicated cores with `taskset` halved the damage but did not remove it — 915 ms
against 1696 ms — which is consistent with scheduling rather than with the
engine.

The harness now does five warm-up runs and sleeps 2 ms between iterations. Use
`--example warmup` to see the curve directly: it prints every run, and on this
pipeline the first costs ~450 ms against a 25 ms steady state, settling on the
second run.

## Correctness across devices

Unlike the timings, this part is solid.

| model | device | `fp32` | `fp16` | `fp16_iobinding` |
|---|---|---|---|---|
| guardrails, 13 cases / 6 languages | CPU | 61/61 (0.0001) | 61/61 (0.0034) | 61/61 (0.0035) |
| guardrails | RTX 3090 | 61/61 (0.0001) | 61/61 (0.0036) | 61/61 (0.0035) |
| privacy, 13 cases / 7 languages | CPU | 58/58 (**0.0000**) | 58/58 (0.0023) | 58/58 (0.0021) |
| privacy | RTX 3090 | 58/58 (**0.0000**) | 58/58 (0.0022) | 58/58 (0.0021) |

Spans identical to the PyTorch reference in all twelve configurations; brackets
give the largest score delta. In `fp32` the privacy export matches PyTorch
exactly at the precision the harness records, on both devices — which puts a
floor under the FP16 rows: their deviation is quantisation, not a defect in the
graphs.

The CUDA kernels agree with the CPU ones to within FP16 rounding. Choosing a
device is a performance decision, not an accuracy one.

## Two hypotheses that turned out to be wrong

Recorded so nobody spends the afternoon on them again.

**The GPU is not falling back to CPU.** Profiling a fragment under the CUDA
provider puts **97.1% of nodes on CUDA** and 2.9% on CPU. The graph is not being
partitioned with a device transfer between each piece.

**The pipeline is not unusually large.** 5815 nodes per extraction, of which
4588 are the mDeBERTa encoder — it dominates, and the eight fragments around it
are a small addition.

## Not measured

- `fastino/gliner2-multi-v1`, the base checkpoint. Never fetched locally.
- `gliner2-inference`, the legacy engine (removed 2026-08-26; in git history). Its README quoted
  benchmarks from April, taken on different hardware with a different method;
  they are not comparable with anything here.
- Long documents and wide schemas. Every figure comes from one ~90-word
  paragraph with five labels. Timings scale with sequence length and with the
  number of schema tasks, since each task costs its own `schema_gather`,
  `count_lstm` and `scorer` pass.

## Reproducing

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
GLINER2_DEVICE=cuda:1 GLINER2_PRECISION=fp32 \
cargo run --release --example bench -p gliner2-rs -- models/guardrails-pii-multi-onnx 25
```

The GPU run needs a runtime shipping the CUDA provider — the plain `onnxruntime`
wheel does not. `pip install onnxruntime-gpu` provides
`libonnxruntime_providers_cuda.so`; point `LD_LIBRARY_PATH` at its directory
along with the CUDA 12 libraries.
