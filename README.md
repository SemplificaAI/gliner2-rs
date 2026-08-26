# gliner2-rs

[![GitHub](https://img.shields.io/badge/GitHub-dariofinardi/gliner2--rs-blue?style=flat-square&logo=github)](https://github.com/dariofinardi/gliner2-rs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Beta-blue.svg)](https://github.com/dariofinardi/gliner2-rs)

**Native Rust inference for GLiNER2 on ONNX Runtime**

Extract entities, relations and classifications from text with no Python at
inference time. One crate: the span engine, with the PII and guardrail
vocabularies behind default-on features.

Written by **Dario Finardi**. Published by **Jugaad s.r.l.**, which uses it in
production inside **Edito** and **Omissis** —
[edito-pdf.com](https://edito-pdf.com).

For **GLiNER2.5** use [gliner25-rs](https://github.com/dariofinardi/gliner25-rs)
instead. Its `boundary` architecture shares only the foundation with this one,
and its spans are half-open rather than inclusive — mixing the conventions is a
silent off-by-one.

---

## The crates

| crate | what it is | docs |
|---|---|---|
| [`gliner2-rs`](crates/gliner2-rs) | the engine, plus the PII and guardrail vocabularies behind default-on features | [README](crates/gliner2-rs/README.md) |
| `gliner2_inference` | the original engine: V1 pipeline, HuggingFace downloader. In-repo only, **not published** | [README](crates/gliner2-inference/README.md) |

```toml
[dependencies]
gliner2-rs = "0.9"
```

`gliner2-rs` loads a model from a local directory, and fetches it from the Hub
if that directory is empty — see [Getting the weights](#getting-the-weights).
Switch the `hub` feature off and the crate has no HTTP client, no TLS stack and
no Hub client in its dependency tree at all.

`gliner2_inference` is the pre-split engine, kept in the repository for the
V1 fallback and its own `from_pretrained`. It is **not published to crates.io**:
depend on it by path or git if you want it. It is also the only thing here that
pulls `openssl`, through `hf-hub`'s default `native-tls`; the published crate
takes `hf-hub` with `rustls` instead and has no OpenSSL anywhere.

0.1 split this into five crates — engine, two vocabularies, and the same again
for GLiNER2.5. They were only ever installed as a set, so they are one crate and
two features now. `--no-default-features` still drops either vocabulary if you
want the engine bare.

---

## Getting the weights

Point the engine at a directory. If it holds an export, it is used untouched.
If it does not, the export is fetched from the Hub before the engine starts:

```rust
use gliner2_rs::{SpanConfig, SpanEngine, hub};

let cfg = SpanConfig::new("models/pii-onnx").or_download(hub::PRIVACY_PII_MULTI);
let mut engine = SpanEngine::new(cfg)?;   // downloads only if the directory is empty
```

Skip the local path entirely and work straight out of the cache:

```rust
let mut engine = SpanEngine::new(SpanConfig::from_hub(hub::GUARDRAILS_PII_MULTI))?;
```

**The local directory always wins.** A checkout already on disk is never
re-fetched, and the network is reached only on a miss. Files land in the shared
Hub cache (`HF_HOME`, else `~/.cache/huggingface`), so a model already pulled by
the Python library is not pulled again.

**Only the variant you will run is fetched.** An export carries up to three
copies of every fragment and the encoder alone is half a gigabyte, so the
execution mode picks: a bound engine takes the FP16-I/O graphs, a standard one
the FP32. Measured on the PII export with a cold cache:

| `GLINER2_EXECUTION` | files | variant | downloaded |
|---|---|---|---|
| `standard` | 8 | `_fp32.onnx` | 1 245 MB |
| `binding` | 8 | `_fp16_iobinding.onnx` | **632 MB** |

`with_precision` pins the variant and overrides that choice. If the repository
does not publish the preferred one the engine falls back rather than failing.

| constant | repository |
|---|---|
| `hub::GLINER2_MULTI_V1` | [`jugaadsrl/gliner2-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2-multi-v1-onnx) |
| `hub::PRIVACY_PII_MULTI` | [`jugaadsrl/gliner2-privacy-filter-PII-multi-onnx`](https://huggingface.co/jugaadsrl/gliner2-privacy-filter-PII-multi-onnx) |
| `hub::GUARDRAILS_PII_MULTI` | [`jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx`](https://huggingface.co/jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx) |

Any other repository works — `hub::Model::new(repo_id, layout)` takes a private
fine-tune as readily as a published one. `layout` says whether the fragments sit
at the repository root or under `fp32_v2/` and `fp16_v2/`, as the earlier
exports do.

Try it:

```sh
ORT_DYLIB_PATH=… cargo run --release --example download -p gliner2-rs -- models/pii-onnx
```

### If you would rather it never touched the network

```toml
gliner2-rs = { version = "0.7", default-features = false, features = ["guardrails", "privacy"] }
```

That removes `hf-hub`, `ureq` and `rustls` and leaves the crate with no network
stack whatsoever. Fetch the export however you like — `hf download`, `git
clone`, a build step — and hand `SpanConfig::new` the path.

### On the TLS backend

`hf-hub` arrives with `default-features = false, features = ["ureq"]`, which
resolves TLS through **`rustls`**. Its default feature set pulls `native-tls`
and with it `openssl` — a C library and the CVE stream that comes with it — for
no benefit here. Downloading weights over HTTPS does not need OpenSSL, so this
crate does not carry it.

---

## Long documents

`extract` sees one window, and one call builds `num_words × max_width` span
representations — so the memory it needs grows with the text until the device
refuses. Measured on an RTX 3090 (24 GB, idle) with the PII model:

| words | `extract` |
|---|---|
| up to 4 000 | works, both transports |
| 5 000 and above | **fails** — `Expand` alone asks for 10.8 GB at 6 600 words |

Binding gives way first and the engine drops to the standard path by itself;
past the threshold neither fits, and the error now says so and names the way out
rather than relaying a byte count.

```rust
let out = engine.extract_long(&document, &tasks)?;   // 384-word windows, 64 overlap
```

Measured on that same 5 000-word document, `cuda:1`: `extract` fails,
`extract_long` returns **211 entities in 1 953 ms**, every offset indexing the
original text.

Text that fits in one window takes the single-call path, so this is safe to use
unconditionally. `extract_long_with(.., Chunker::new(256, 48)?)` sets the
geometry.

**Why the windows overlap.** A mention straddling a window edge is seen whole by
neither. The overlap is its second chance: with 64 words of margin, anything
shorter appears intact in at least one window. What no merge can recover is an
entity longer than the overlap, or a relation whose ends fall in different
windows — inherent to chunking, not to this implementation.

Duplicates are collapsed by span keeping the highest score. Classifications are
collapsed per label, also by highest score — for a guardrail that is exactly
right, since a prompt injection buried on page nine is still an injection; for a
descriptive label it is optimistic, so read a document-level classification as
"somewhere in here", not "overall".

**One caveat worth knowing.** A device OOM can leave the ORT arena in a state
where later calls fail for reasons of their own. In practice `extract_long`
recovers — the engine has dropped to the standard path by then and the windows
are small — but calling it directly rather than as a fallback is the cleaner
shape.

---

## Execution modes

The engine is a chain of eight ONNX fragments. Between any two of them the
intermediate tensor either goes back to host memory and is rebuilt for the next
fragment, or stays where the provider produced it and is bound straight into the
next fragment's input.

Same fragments, same order, same arithmetic — only the transport differs. So
there is one pipeline and a switch, not two engines:

```rust
use gliner2_rs::{SpanConfig, SpanEngine, chain::ExecutionMode};

let cfg = SpanConfig::new("models/pii-onnx")
    .with_execution(ExecutionMode::IoBinding);
```

| mode | what it does |
|---|---|
| `Auto` *(default)* | bound on a device provider, standard on CPU |
| `IoBinding` | intermediates stay in device memory across the chain |
| `Standard` | every output returns to host memory first — works everywhere |

`GLINER2_EXECUTION=standard\|binding\|auto` sets it from the environment, and
`GLINER2_NO_IOBINDING=1` still forces the standard path — it is what the older
engine used and it means the same thing here.

`SpanEngine::execution()` reports the mode actually in force, after `Auto` has
resolved and after any fallback.

### What it is worth

RTX 3090, GLiNER2 PII, 25 runs, median. A shared development machine, so read
these as ratios rather than absolutes:

| precision | `Standard` | `IoBinding` | |
|---|---|---|---|
| `fp32` | 25.9 ms | **10.8 ms** | 2.4× |
| `fp16_iobinding` | 28.1 ms | **11.3 ms** | 2.5× |

No CPU figures, deliberately: this machine sits at load average 17–19 and the
same mode varied by more than 10× between consecutive runs, so any number
quoted from it would be invented. What can be said is what the design implies —
on CPU "device memory" *is* host memory, so binding saves no copy and only its
bookkeeping remains. That is why `Auto` does not use it there.

### Falling back

A device allocation failure during binding is not fatal. The engine drops to the
standard path for the rest of its life and carries on — one slow run beats a
failed one, and retrying the binding on every call would just pay the same failure
again. `execution()` will report `Standard` afterwards.

### Same numbers either way

Verified on GPU with both modes over the same input: identical entities,
identical scores. The standard path is also byte-identical to the engine before
binding existed — confirmed against the same ONNX Runtime build, since a
different runtime version moves the last decimal on its own.

---

## Model compatibility

Weights are **not** ours. GLiNER2 is developed by [Fastino](https://fastino.ai)
(arXiv:2507.18546); the GLiNER line it descends from is the work of Urchade
Zaratiana et al. Converting a model changes neither its licence nor its
ownership — see [`NOTICE`](NOTICE).

| model | ONNX export | export layout | feature |
|---|---|---|---|
| [`fastino/gliner2-multi-v1`](https://huggingface.co/fastino/gliner2-multi-v1) | [`jugaadsrl/gliner2-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2-multi-v1-onnx) | legacy | none — engine core |
| [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | [`jugaadsrl/gliner2-privacy-filter-PII-multi-onnx`](https://huggingface.co/jugaadsrl/gliner2-privacy-filter-PII-multi-onnx) | legacy | `privacy` |
| [`fastino/GLiNER2-Guardrails-PII-Multi`](https://huggingface.co/fastino/GLiNER2-Guardrails-PII-Multi) | [`jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx`](https://huggingface.co/jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx) | flat | `guardrails` |
| `fastino/gliner2-base-v1` and local fine-tunes | export it yourself | flat | none — engine core |

Any GLiNER2 **span** checkpoint works: the engine reads `max_width` and
`MAX_COUNT` from the exported graphs rather than assuming them, so a checkpoint
exported with different parameters stays usable.

### The two export layouts

Both are read, and the difference is invisible to callers. Point the config at
the directory and it works.

```text
flat, from export_span_v3.py          legacy, published earlier on the Hub
  encoder_fp32.onnx                     fp32_v2/encoder_fp32.onnx
  encoder_fp16.onnx                     fp16_v2/encoder_fp16.onnx
  encoder_fp16_iobinding.onnx           fp16_v2/encoder_fp16_iobinding.onnx
  …                                     …
  tokenizer.json                        fp32_v2/tokenizer.json
                                        fp16_v2/tokenizer.json
```

Legacy exports also carry an older `classifier` signature,
`[batch, num_labels, max_width, H]` instead of `[num_labels, H]`. The engine
detects it from the graph and shapes the input accordingly; check which one it
found with `engine.classifier_layout()`.

### Precision

| suffix | I/O | use for |
|---|---|---|
| `_fp32` | FP32 | universal fallback, OpenVINO, CPU |
| `_fp16` | FP32 (`keep_io_types=True`) | CoreML, which demands FP32 I/O |
| `_fp16_iobinding` | FP16 | CUDA, ROCm, QNN — see the note below |

Selected automatically per platform — `_fp16_iobinding` on Linux and Windows,
`_fp16` on macOS — and overridable with
`GLINER2_PRECISION=fp32|fp16|fp16_iobinding`.

### A note on `_fp16_iobinding`

The suffix names what the variant was exported for: `keep_io_types=False` leaves
the graph inputs and outputs in FP16 as well as the weights, which is what
zero-copy binding needs to keep tensors in device memory across the chain.

Binding is implemented — see [Execution modes](#execution-modes) — and the
variant is worth using with it. Without binding it still saves the FP32↔FP16
conversions at each boundary, so it is not wasted either way.


---

## Quick start

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
cargo run --release --example extract_pii -p gliner2-rs -- models/pii-onnx
```

```rust
use gliner2_rs::{SchemaTask, SpanConfig, SpanEngine};

gliner2_rs::init("my-app");
let mut engine = SpanEngine::new(SpanConfig::new("models/my-onnx-export"))?;

let tasks = vec![SchemaTask::Entities(vec![
    "person".into(), "organization".into(), "location".into(),
])];

for e in engine.extract("Mario Rossi works at Apple.", &tasks)?.entities {
    println!("{} -> {} ({:.1}%)  bytes [{}..{})",
             e.text, e.label, e.score * 100.0, e.char_start, e.char_end);
}
```

Byte offsets index the original text, so extracted spans keep their original
casing — which matters when redacting a document rather than labelling it.

Each crate README has worked examples for its own vocabulary.

---

## Requirements

- Rust **edition 2024**, MSRV **1.88** for the new crates;
  `gliner2-inference` is edition 2021.
- `ort` **≥ 2.0.0-rc.13, < 3.0**, with `default-features = false` — nothing is
  downloaded at build time and no execution-provider libraries are copied next
  to your binary.
- ONNX Runtime shared library **1.23 or newer**, resolved at run time from
  `ORT_DYLIB_PATH`. Verified against ONNX Runtime 1.25.1 at API level 17, and
  against the `onnxruntime-gpu` 1.23.2 build for CUDA.

  **Older runtimes segfault when the process exits.** The API level `ort`
  requires is 17, which runtimes well below 1.23 satisfy — so an old shared
  library loads, runs correctly, and then crashes on the way out. One binary,
  three runtimes:

  | ONNX Runtime | inference | exit code |
  |---|---|---|
  | 1.20.0 | correct | **139 — SIGSEGV** |
  | 1.22.0 | correct | **139 — SIGSEGV** |
  | 1.23.2 | correct | 0 |

  The scores are identical in all three; only the exit differs. The crash lands
  after `main` returns, so output is complete and nothing is corrupted — but the
  exit code breaks CI, shell `&&` chains and process supervisors, and in a
  long-running server it surfaces at shutdown.

  It is not this crate, and it is not the CUDA EP either — this is CPU-only.
  The same fault reproduces in twenty lines of `ort` with no GLiNER code
  involved: create a session, drop it, return from `main`. It goes away if the
  session is leaked instead of dropped, and it reproduces with one intra-op
  thread as readily as with four, so it is not the thread pool.

  The root cause is `ort`'s global `Environment` being released at process exit
  after the session state it refers to is gone. It is fixed upstream by
  [pykeio/ort#610](https://github.com/pykeio/ort/pull/610) (details and
  measurements in [pykeio/ort#614](https://github.com/pykeio/ort/issues/614)),
  which makes the
  environment manual instead of global — verified here: `ort` from git exits
  cleanly even against ONNX Runtime 1.20.0. That change is **not in rc.13**, the
  newest release, so until it ships the runtime version is what decides it.

  If you are pinned to an older runtime, `std::process::exit(0)` at the end of
  `main` sidesteps the crash, at the cost of skipping every other destructor
  too.

  **The rc.13 floor is not arbitrary.** Release candidates 10 through 12 were
  tried and rejected: on **ARM CPU** some models hung during session
  initialisation or inference, reproducibly enough that this project stayed on
  rc.9 for months rather than move to them. rc.13 is the first candidate since
  rc.9 that runs those models on ARM, which is why the migration skipped three
  releases. Do not lower the floor.

  Upwards, the requirement is a caret rather than an exact pin, so these crates
  can be combined with anything else depending on `ort`. That is a calculated
  risk while `ort` is still in release candidates: between rc.9 and rc.13,
  `commit()` changed its return type, `Session::run` started taking `&mut self`,
  `try_extract_tensor` began returning a `Shape`, and `Outlet`'s fields went
  private. A later rc can break the build the same way. **Pin exactly in your
  own application** if you need that guarantee — a library should not impose it
  on its dependents.
- Enable `ort`'s `download-binaries` feature instead if you would rather it
  fetch the runtime for you.

---

## Performance

On an RTX 3090, one ~90-word paragraph with five entity labels: **20–27 ms**
depending on model and precision, against **2.5–3.2 s** on a contended CPU.
Load time is 7–14 s, dominated by reading the encoder from disk.

Each model is reported separately in [`BENCHMARKS.md`](BENCHMARKS.md) — they are
different checkpoints finding different entities, and comparing them to each
other is not meaningful. The one conclusion that survives this host is the GPU
gap; the precision ordering does not, since the two models disagree about it and
the spread is smaller than the machine's own variance.

That file also records a harness bug worth knowing about before you benchmark
this yourself: timing back-to-back iterations in a tight loop on a contended
machine inflated the numbers by up to 100×.

## Verification

Per-fragment parity proves the ONNX graphs are faithful. It is not enough:
prompt construction, word routing, span decoding and suppression all live in
Rust, outside the graphs. The test that matters is the end-to-end comparison
against the PyTorch reference.

Run on both devices and every precision, because a CUDA kernel producing
different numbers from its CPU counterpart is exactly the kind of thing that
goes unnoticed otherwise:

| crate | device | `fp32` | `fp16` | `fp16_iobinding` |
|---|---|---|---|---|
| guardrails, 13 cases / 6 languages | CPU | 61/61 (0.0001) | 61/61 (0.0034) | 61/61 (0.0035) |
| guardrails | RTX 3090 | 61/61 (0.0001) | 61/61 (0.0036) | 61/61 (0.0035) |
| privacy, 13 cases / 7 languages | CPU | 58/58 (**0.0000**) | 58/58 (0.0023) | 58/58 (0.0021) |
| privacy | RTX 3090 | 58/58 (**0.0000**) | 58/58 (0.0022) | 58/58 (0.0021) |

Spans identical to the PyTorch reference in all twelve configurations; the figure
in brackets is the largest score delta.

Two things worth reading off that table. In `fp32` the privacy export matches
PyTorch **exactly** — not to within a tolerance, to the fourth decimal the
harness records — on both devices. And the CUDA path agrees with the CPU one to
within FP16 rounding, 0.0034 against 0.0036 on the same case. Whatever else
changes when you move to a GPU, the answers do not.

```sh
python onnx_conversion_scripts/compare_with_pytorch.py reference \
    --model_path fastino/<checkpoint> --cases tests/cases_pii.json --out /tmp/pytorch.json

ORT_DYLIB_PATH=… cargo run --release --example dump_json -p gliner2-rs -- \
    models/pii-onnx tests/cases_pii.json > /tmp/rust.json

python onnx_conversion_scripts/compare_with_pytorch.py diff \
    --reference /tmp/pytorch.json --candidate /tmp/rust.json
```

Every decoding bug this project has had was caught this way and by nothing else:
the prompt layout drift, the cross-label suppression, the multi-label argmax
fallback. Tolerances are **relative**, scaled to each tensor's magnitude —
`span_rep` emits activations up to ~9e3 while `scorer` is a probability in
`[0,1]`, so an absolute threshold is meaningless across that range.

---

## 📊 Benchmark & Performance

> These figures were measured with **`gliner2-inference`**, whose V2 pipeline
> uses `IoBinding`. `gliner2-rs` does not implement binding yet, so its GPU
> numbers will differ; the CPU ones are comparable.

Tested on complex text extraction tasks spanning up to 62 classes. Total Inference Time per Sentence is the primary metric used for fair cross-framework comparison, allowing precise cross-device and cross-language comparisons.

### 🖥️ Rust ONNX vs Python PyTorch (Desktop & Discrete GPUs)
Comparison of a 50-run continuous benchmark on x86_64 architecture with NVIDIA GPUs.

| Language | Engine (Hardware) | Total Time (50 runs) | Avg Time / Sentence | Avg Time / Entity (15-17) |
| :--- | :--- | :--- | :--- | :--- |
| **Python 3.10** | PyTorch (RTX 4090) | **~0.88 s** 🚀 | **4.40 ms** | 1.17 ms |
| **Python 3.10** | PyTorch (RTX 3090) | **~0.90 s** 🚀 | **4.52 ms** | 1.20 ms |
| **Rust (V1)** | ONNX Runtime CUDA (RTX 4090) | **~8.18 s** | **40.90 ms** | 10.90 ms |
| **Rust (V2)* ** | ONNX Runtime CUDA (RTX 4090) | **~5.91 s** ⚡ | **29.59 ms** | 6.96 ms |
| **Rust (V1)** | ONNX Runtime CUDA (RTX 3090) | **~8.59 s** | **42.97 ms** | 11.45 ms |
| **Rust (V2)* ** | ONNX Runtime CUDA (RTX 3090) | **~6.13 s** ⚡ | **30.68 ms** | 7.21 ms |
| **Python 3.10** | PyTorch (Ryzen 5900XT CPU) | **~7.26 s** | **36.33 ms** | 9.68 ms |
| **Rust (V1)** | ONNX Runtime (Ryzen 5900XT CPU) | **~13.75 s** | **68.76 ms** | 18.33 ms |

> *( * ) **V2 IOBinding Engine:** The new V2 implementation eliminates the PCIe bottleneck by fusing operations (`Gather`, `ArgMax`, `MatMul`) inside the ONNX graph and keeping tensors entirely in VRAM (Zero-Copy) using ORT's `IoBinding`. This drastically drops the execution time.


**Understanding the GPU Gap: Why is PyTorch still faster than V2?**
While V2 IOBinding successfully eliminates the PCIe data transfer bottleneck (tensors now stay in VRAM), Python/PyTorch remains ~6x faster on discrete GPUs. This is due to the **Fragmentation Penalty**:
1. **Kernel Launch & Orchestration Overhead:** Because GLiNER2's architecture relies on dynamic loops (e.g. iterating over an unknown number of schema tasks and varying predicted entity counts), it cannot be exported as a single monolithic ONNX graph. It must be split into 8 separate ONNX sessions. The Rust host CPU must orchestrate the execution of these 8 fragments sequentially. Even though the *data* stays in VRAM, the *control flow* (calling `.run()` multiple times per sentence) incurs severe CUDA kernel launch overhead and forces continuous CPU-GPU synchronization.
2. **Lack of Global Graph Fusion:** PyTorch executes the entire model inside a single unified context, allowing its backend to fuse kernels across the entire architecture. ONNX Runtime can only optimize and fuse operations within the hard boundaries of each individual fragment.
3. **Dynamic Shapes:** ONNX Runtime achieves peak performance (e.g., via TensorRT) with static shapes. GLiNER2 is highly dynamic (varying sequence lengths, changing number of entities), which prevents ORT from locking in optimal execution paths—a scenario where PyTorch's native dynamic execution naturally excels.

*Conclusion:* Rust ONNX V2 represents the upper limit of optimization for a fragmented pipeline. While PyTorch wins on raw continuous throughput on discrete GPUs, Rust ONNX completely dominates PyTorch in **Cold Start** scenarios (loading in ~2s vs ~10s) and is the absolute winner for **Unified Memory Architectures** (Apple Silicon / ARM Snapdragon NPU) and edge deployments.

### 🐍 Rust vs Python on ARM (Snapdragon X Elite)
Comparison between native Rust ONNX execution and standard Python PyTorch inference on the same ARM hardware.
Note: Benchmarks executed plugged in (Max Performance profile). Testing 51 target entities extraction.

| Environment | Hardware (Backend) | Precision (Model) | Startup Time | Total Inference Time (Sentence) | Time / Entity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rust (V1)** | CPU ARM64 (Oryon) | `fp32` | **~3.64 s** | **0.43 s** 🚀 | ~8.53 ms |
| **Rust (V2)* ** | NPU (QNN) | `fp16_v2` | **~2.28 s** | **0.65 s** ✨ | ~12.88 ms |
| **Rust (V2)* ** | CPU ARM64 (Oryon) | `fp16_v2` | **~1.96 s** ⚡ | **0.66 s** | ~13.10 ms |
| **Rust (V1)** | CPU ARM64 (Oryon) | `fp16` | **~1.82 s** | **0.68 s** | ~13.43 ms |
| **Rust (V1)** | NPU (QNN) | `fp16` | **~2.12 s** | **0.71 s** | ~14.11 ms |
| **Python 3.12** | CPU ARM64 (PyTorch) | `jugaadsrl/gliner2-multi-v1` | **~12.74 s** 🐢 | **0.31 s** | ~15.03 ms |
| **Python 3.12** | CPU ARM64 (PyTorch) | `fastino/gliner2-multi-v1` | **~8.76 s** 🐢 | **0.36 s** | ~24.51 ms |

**Takeaways:**
- **The FP32 Surprise:** Instructing the Rust ONNX runtime to load full FP32 precision models allows the Snapdragon ARM64 Oryon CPU to skip expensive hardware/software downcasting. It slashes inference time to **0.43s per sentence**, completely crushing FP16 times and heavily outperforming the limited NPU drivers.
- **V2 IOBinding is Consistent:** At matched FP16 precision, the fused V2 consistently beats the standard V1 architecture, both on CPU and NPU.
- **Rust = Cold Start Speed & Reproducibility:** Rust boots in ~1.8-3.6s (depending on precision) and flawlessly extracts the exact overlapping entities without implicit filtering. Python struggles for ~9-12s just to load tensors and forces unrequested NMS flat_ner filtering which artificially alters the output count.

---


---

## Exporting a model

```sh
python onnx_conversion_scripts/export_span_v3.py \
    --model_path fastino/gliner2-privacy-filter-PII-multi \
    --out_dir models/pii-onnx
```

Needs `torch`, `transformers`, `gliner2>=2.0.0`, `onnx`, `onnxruntime`. It
refuses `boundary` checkpoints rather than producing a silently wrong export.

The span architecture cannot be traced into a single ONNX graph — it loops over
a variable number of schema tasks and a predicted, variable number of entity
occurrences — so it is exported as eight fragments. See
[`crates/gliner2-rs/README.md`](crates/gliner2-rs/README.md) for the chain.

---

## License and attribution

Licensed under the [Apache License, Version 2.0](LICENSE).

The models are the work of the [Fastino](https://fastino.ai) team and are not
distributed from this repository. See [`NOTICE`](NOTICE) for the full
attribution, and [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for the history.

Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
