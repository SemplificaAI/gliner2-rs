# gliner2-rs

Native Rust inference for GLiNER2 **span** checkpoints on ONNX Runtime. No
Python at inference time, and no Python at build time either.

One crate: the engine, plus the PII and LLM-guardrail vocabularies behind
default-on features.

```toml
[dependencies]
gliner2-rs = "0.9"
```

For GLiNER2.5 `boundary` checkpoints use
[gliner25-rs](https://crates.io/crates/gliner25-rs) instead. The two share a
lineage but not a graph, and mixing them silently produces nonsense rather than
an error.

---

## The five-minute version

```rust
use gliner2_rs::{SchemaTask, SpanConfig, SpanEngine};

gliner2_rs::init("my-app");                       // ort::init(), once per process

let mut engine = SpanEngine::new(SpanConfig::new("models/pii-onnx"))?;

let tasks = vec![SchemaTask::Entities(vec!["person".into(), "email".into()])];
let out = engine.extract("Mario Rossi — m.rossi@example.it", &tasks)?;

for e in &out.entities {
    println!("{:?} {} {:.1}%", e.text, e.label, e.score * 100.0);
}
```

`ORT_DYLIB_PATH` must point at an ONNX Runtime **1.23 or newer** shared library.
Older runtimes run correctly and then segfault at process exit — see the root
README for the measurements.

---

## What the engine gives you

| type | what it is |
|---|---|
| `SpanEngine` | the loaded model. Owns eight ONNX sessions; `extract` takes `&mut self` |
| `SpanConfig` | where the model is, which variant, how it runs. Builder-style |
| `SchemaTask` | one group of things to look for: entities, relations, or a classification |
| `SpanOutput` | `entities` and `classifications` |
| `Entity` | text, label, score, byte range, word range, occurrence slot, source task |
| `Classification` | task, label, probability |
| `InferenceParams` | `threshold`, `flat_ner`, `classification_temperature`, `multi_label_override`, `overlap_policy` |
| `GlinerError` | eight diagnosable failures, `E_GLI_001`…`E_GLI_008` |

### Asking for three kinds of thing

```rust
let tasks = vec![
    SchemaTask::Entities(vec!["person".into(), "organization".into()]),
    SchemaTask::Relations("works_for".into(), vec!["head".into(), "tail".into()]),
    SchemaTask::classification("tone", vec!["formal".into(), "casual".into()]),
    SchemaTask::multi_label_classification("topics", vec!["legal".into(), "medical".into()]),
];
```

`classification` is a softmax over the choices, `multi_label_classification`
independent sigmoids. The flag lives on the task and not on the request,
because one call routinely mixes both — `prompt_safety` is single-label while
`prompt_toxicity` and `jailbreak_detection` are not.

Each group is a separate pass over the same encoder output, and **labels compete
only inside their group**. That is not an implementation detail: putting twelve
unrelated labels in one group measurably fragments spans, and splitting them
into families fixes it.

### Reading a classification

```rust
for c in out.verdict("tone", 0.5) { println!("{} {:.1}%", c.label, c.score * 100.0); }
```

`verdict` returns every label over the threshold, and if none clears it, the
single highest — which is what `gliner2` does. Reading `out.classifications`
directly gives you every label with its probability instead.

---

## Long documents

`extract` sees one window. mDeBERTa-v3 is trained to 512 positions, and past
that the span buffers grow until the device refuses them — on a 5 000-word
document `span_rep` asks for half a gigabyte and the call fails.

```rust
let out = engine.extract_long(&document, &tasks)?;      // 384-word windows, 64 overlap
```

Text that fits in one window takes the single-call path, so this is safe to use
unconditionally. Offsets always index the original document.

`extract_long_with(text, tasks, params, Chunker::new(256, 48)?)` sets the
geometry. What no merge can recover is an entity longer than the overlap, or a
relation whose ends fall in different windows — see the [`chunker`] module docs.

---

## Execution modes

Between any two fragments an intermediate tensor either returns to host memory
or stays where the provider produced it. Same maths, different transport.

```rust
use gliner2_rs::chain::ExecutionMode;
let cfg = SpanConfig::new("models/pii-onnx").with_execution(ExecutionMode::IoBinding);
```

| mode | |
|---|---|
| `Auto` *(default)* | bound on a device provider, standard on CPU |
| `IoBinding` | intermediates stay in device memory — **2.4× on an RTX 3090** |
| `Standard` | every output returns to the host first; works everywhere |

A device allocation failure drops the engine to the standard path for the rest
of its life rather than failing the call. `engine.execution()` reports what is
actually in force.

---

## Getting the weights

Point at a directory; if it holds no export, one is fetched:

```rust
use gliner2_rs::hub;
let cfg = SpanConfig::new("models/pii-onnx").or_download(hub::PRIVACY_PII_MULTI);
```

**Only the variant you will run is downloaded** — the execution mode picks it,
so a bound engine fetches the FP16-I/O graphs and a standard one the FP32,
rather than pulling every copy of every fragment. Measured on the PII export:
632 MB against 1 245 MB.

`hub::GLINER2_MULTI_V1`, `hub::PRIVACY_PII_MULTI` and `hub::GUARDRAILS_PII_MULTI`
name the published exports; `hub::Model::new` takes any other repository.
`with_precision` pins the variant and overrides the mode's preference.

---

## The two vocabularies

Both are default-on features. They are tables of labels and a few helpers — no
extra dependencies — and exist because the checkpoints they belong to are
useless without them.

### `privacy` — `fastino/gliner2-privacy-filter-PII-multi`

```rust
use gliner2_rs::privacy::{Group, needs_anonymization, redact};

let tasks = [Group::Person, Group::Contact, Group::Banking].map(Group::task);
let out = engine.extract(text, &tasks)?;

if needs_anonymization(&out.entities, 0.5) {
    println!("{}", redact(text, &out.entities));
}
```

`Group` splits sixty-odd PII labels into families that do not compete —
`Person`, `Contact`, `Banking`, and the rest. `Group::task()` turns one into a
`SchemaTask`; `all_labels_task()` builds the single flat task with every label,
which is what you want only if you have measured that it beats the families on
your data. It usually does not. `redact` resolves overlapping spans **by score**, so
`Giuseppe Verdi` at 99.8 % as `full_name` becomes `[FULL_NAME]` rather than
`[FIRST_NAME] [LAST_NAME]`. `redact_with` takes your own placeholder format.

### `guardrails` — `fastino/GLiNER2-Guardrails-PII-Multi`

```rust
use gliner2_rs::guardrails::{prompt_moderation_schema, response_moderation_schema};

let out = engine.extract(user_prompt, &prompt_moderation_schema())?;
for c in out.verdict("prompt_injection", 0.5) { /* … */ }
```

The same checkpoint carries PII extraction *and* moderation heads, so one pass
gives both. `prompt_moderation_schema` and `response_moderation_schema` are the
two ready-made schemas; `Task` enumerates the individual heads, each with `labels()` and the
`threshold()` the model was calibrated for, if you want to assemble your own.

Turn either feature off and its module disappears:

```toml
gliner2-rs = { version = "0.8", default-features = false, features = ["hub"] }
```

---

## Model compatibility

Any GLiNER2 **span** checkpoint. `max_width`, `MAX_COUNT` and the hidden size
are read from the exported graphs rather than assumed, and both export layouts
are accepted — flat, and the `fp32_v2/` + `fp16_v2/` subfolders the earlier
exporter published.

| checkpoint | ONNX export | needs |
|---|---|---|
| `fastino/gliner2-multi-v1` | [`jugaadsrl/gliner2-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2-multi-v1-onnx) | — |
| `fastino/gliner2-privacy-filter-PII-multi` | [`jugaadsrl/gliner2-privacy-filter-PII-multi-onnx`](https://huggingface.co/jugaadsrl/gliner2-privacy-filter-PII-multi-onnx) | `privacy` |
| `fastino/GLiNER2-Guardrails-PII-Multi` | [`jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx`](https://huggingface.co/jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx) | `guardrails` |
| your own fine-tune | export it yourself | — |

---

## Two things that will surprise you

**Spans are inclusive.** `Entity::word_start` and `word_end` are `[start, end]`,
both ends included — a one-word entity has `word_start == word_end`. The
`boundary` architecture in `gliner25-rs` uses half-open ranges. Byte offsets
(`char_start`, `char_end`) are half-open in both.

**`flat_ner` and `overlap_policy` are alternatives, not companions.** Leaving
`overlap_policy` at `None` selects the span architecture's historical greedy
decoder, governed by `flat_ner` — which is what `gliner2` 2.0.0 does on a span
checkpoint. Naming a policy switches to the resolver shared with the boundary
architecture, and `flat_ner` is then ignored.

**The prompt is not wrapped.** `gliner2` emits no `[CLS]`/`[SEP]` around the
schema, field indices point at the `[E]`/`[R]`/`[L]` marker rather than the
label name, and text words are lower-cased. Getting any of the three wrong
misaligns the gathered embeddings against what the model was trained on, and
the output stays plausible while being wrong — one codice fiscale scored 0.5016
against a 0.5 threshold before the fix.

---

## Weights are not ours

GLiNER2 is developed by [Fastino](https://fastino.ai) (arXiv:2507.18546); the
GLiNER line it descends from is the work of Urchade Zaratiana et al. Converting
a model to ONNX changes neither its licence nor its ownership. See
[`NOTICE`](../../NOTICE).

Engine code: Apache-2.0. Written by **Dario Finardi**, published by **Jugaad
s.r.l.**, used in production in **Edito** and **Omissis** —
[edito-pdf.com](https://edito-pdf.com).
