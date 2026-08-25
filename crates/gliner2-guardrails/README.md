# gliner2-guardrails

LLM safety moderation schemas for
[`fastino/GLiNER2-Guardrails-PII-Multi`](https://huggingface.co/fastino/GLiNER2-Guardrails-PII-Multi).
Thin layer over [`gliner2-core`](../gliner2-core): the engine is there, this
crate carries the moderation vocabulary with the per-task thresholds and
single/multi-label settings the model expects.

Model: [`jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx`](https://huggingface.co/jugaadsrl/GLiNER2-Guardrails-PII-Multi-onnx)
— PII detection and guardrails in one checkpoint, flat export layout.

## Moderate a prompt

```rust
use gliner2_core::{SpanConfig, SpanEngine};
use gliner2_guardrails::{Task, prompt_moderation_schema, verdict};

gliner2_core::init("my-app");
let mut engine = SpanEngine::new(SpanConfig::new("models/guardrails-onnx"))?;

let out = engine.extract(prompt, &prompt_moderation_schema())?;

println!("{:?}", verdict(&out, Task::PromptSafety));
println!("{:?}", verdict(&out, Task::JailbreakDetection));
```

## Moderate a response

The card is specific about the input format:

```rust
use gliner2_guardrails::{response_input, response_moderation_schema};

let text = response_input(Some(prompt), response);   // "Prompt: …\nResponse: …"
let out = engine.extract(&text, &response_moderation_schema())?;
```

## Why the task builders exist

`multi_label` and the threshold are properties of the *task*, not of the
request: `prompt_safety` is single-label at 0.5, `prompt_toxicity` and
`jailbreak_detection` are multi-label at 0.4, and all three travel in the same
call. Getting one wrong changes the verdict silently, so they are encoded here.

| task | labels | multi-label | threshold |
|---|---|---|---|
| `PromptSafety` / `ResponseSafety` | 2 | no | 0.5 |
| `PromptToxicity` / `ResponseToxicity` | 15 | yes | 0.4 |
| `JailbreakDetection` | 12 | yes | 0.4 |
| `ResponseRefusal` | 2 | no | 0.5 |

Use `verdict()` rather than thresholding the scores yourself: gliner2's
multi-label decoding **never returns an empty list** — when nothing clears the
threshold, the top-scoring label comes back anyway. A jailbreak scoring 0.39
against a 0.4 threshold is reported, not dropped.

## Prompt injection hidden in a document

The scenario this model exists for. An attacker embeds instructions in a
contract as white-on-white text: invisible on screen, but any PDF extractor
returns it as ordinary text.

Measured on a contract carrying an injection that orders the model to ignore its
rules, to not flag any personal data, and to exfiltrate the document:

| input | `prompt_safety` | `jailbreak_detection` |
|---|---|---|
| contract, clean | `safe` | `benign` |
| injection alone | `unsafe` | `instruction_override` |
| contract + hidden injection | `unsafe` | `data_exfiltration` |

It is flagged even when diluted in a full contract, where it is about a third of
the text — and it does **not** suppress extraction. All 14 entities of the clean
contract are still found in the injected one, and the attacker's own drop
address is extracted along with them.

That is architectural, not luck: GLiNER2 is a discriminative encoder, not an
instruction-following model. Injected text is *data* to it, never *commands*.
Running extraction through a model like this instead of an LLM removes the
injection surface rather than trying to defend it.

## Verified

13 cases across 6 languages against the PyTorch reference: **61/61 spans
identical**, max score delta 0.0001 in fp32 and 0.0035 in fp16. Moderation
verdicts identical across 3 cases × 3 tasks × 3 precisions.

Apache-2.0. Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
