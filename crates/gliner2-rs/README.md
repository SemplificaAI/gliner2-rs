# gliner2-rs

The GLiNER2 **span**-architecture inference engine, on ONNX Runtime via `ort`
2.0.0-rc.13.

Runs any GLiNER2 span export. The model-specific vocabularies ship in the same
crate, behind default-on features:

| feature | module | what it carries |
|---|---|---|
| `privacy` | [`privacy`](src/privacy.rs) | the 42 PII labels in their seven groups, redaction, the anonymisation gate |
| `guardrails` | [`guardrails`](src/guardrails.rs) | the moderation label sets with the per-task thresholds the model expects |

Both are on by default. They are tables of labels plus a few helpers — a few KB,
no extra dependencies — and anyone using one of these checkpoints wants them.
They were separate crates in 0.1; that produced five packages that were only
ever installed together, so they are features now.

## Usage

```rust
use gliner2_rs::{InferenceParams, SchemaTask, SpanConfig, SpanEngine};

gliner2_rs::init("my-app");

let mut engine = SpanEngine::new(
    SpanConfig::new("models/my-onnx-export").with_intra_threads(8),
)?;

let tasks = vec![
    SchemaTask::Entities(vec!["person".into(), "organization".into()]),
    SchemaTask::classification("sentiment", vec!["positive".into(), "negative".into()]),
];

let out = engine.extract_with(
    text,
    &tasks,
    &InferenceParams { threshold: 0.5, ..Default::default() },
)?;

for e in &out.entities {
    println!("{} -> {} ({:.1}%)", e.text, e.label, e.score * 100.0);
}
```

`SpanConfig::new` detects the export layout and the best precision for the
platform. Override precision with `GLINER2_PRECISION=fp32|fp16|fp16_iobinding`.

## Two things that will surprise you

**Labels never suppress each other.** The reference decodes one entity name at a
time — threshold, sort by confidence, drop candidates overlapping one already
kept *for that same name*. Labels never interact, so the same span legitimately
returns under several of them:

```text
"Francesca Neri"   medical_professional   91.99%
"Francesca Neri"   person                 91.46%
```

A doctor's name is both, and a redaction pipeline usually wants both — they may
carry different retention rules. `InferenceParams::flat_ner = true` opts into
cross-label suppression as a deliberate extension, not the default.

**Multi-label classification never returns an empty list.** When no label clears
the threshold, the top-scoring one is returned anyway. Thresholding the raw
scores yourself silently disagrees with the reference — a jailbreak scoring 0.39
against a 0.4 threshold is reported, not dropped. Use `SpanOutput::verdict`,
which implements the rule.

Relatedly, `multi_label` belongs to the task rather than the request, since a
single call routinely mixes both kinds. Hence `SchemaTask::classification` and
`SchemaTask::multi_label_classification`.

## The fragment chain

The span architecture cannot be traced into a single ONNX graph: it loops over a
variable number of schema tasks and a predicted, variable number of entity
occurrences. Eight fragments, orchestrated here:

```text
encoder(input_ids, attention_mask) -> last_hidden_state [1,S,H]
  |
  +- token_gather(lhs, word_indices) -> text_embs [1,W,H]
  |     +- span_rep(text_embs, span_idx) -> span_embeddings [1,W,mw,H]
  |
  +- schema_gather(lhs, schema_indices) -> pc_emb [1,H], field_embs [M,H]
        +- count_pred_argmax(pc_emb)    -> pred_count  i64
        +- count_lstm_fixed(field_embs) -> struct_proj [MAX_COUNT,M,H]
              +- scorer(span_embeddings, struct_proj) -> entity_scores, sigmoid-ed

classifier(field_embs) -> logits [M]
```

Span `[w][k]` covers words `w` through `w + k` **inclusive**, valid only while
`w + k < W`. Invalid spans are zeroed to `(0,0)` before `span_rep` and discarded
after, as the reference does.

`max_width`, `MAX_COUNT` and `hidden_size` are read from the graphs' static
shapes rather than hard-coded, so a checkpoint exported with different
parameters stays usable.



Apache-2.0. Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
