# gliner2-inference

The original engine, predating the workspace split. Kept as it is because it has
users and two capabilities the newer crates do not:

- **`from_pretrained`** — pulls ONNX exports from the HuggingFace Hub, choosing
  the variant that suits the platform, instead of requiring a local directory.
- **the V1 pipeline** — the pre-IOBinding fragment layout, with CPU-side
  slicing, for exports that predate the fused V2 graphs.

Edition 2021, `ort = 2.0.0-rc.13`. For new work prefer
[`gliner2-rs`](../gliner2-rs) and its extensions: same engine lineage, but
they read both export layouts and detect the classifier signature.

## Usage

```rust
use gliner2_inference::*;

ort::init().with_name("my-app").commit();

let engine = Gliner2Engine::from_pretrained(
    "jugaadsrl/gliner2-privacy-filter-PII-multi-onnx",
    Some("fp16_v2"),
    ModelType::HuggingFace,
)?;

let tasks = vec![SchemaTask::Entities(vec!["person".into(), "email".into()])];
let (entities, relations, classifications) = engine.extract(
    text,
    &tasks,
    Some(InferenceParams { threshold: 0.5, flat_ner: false }),
)?;
```

A local export works too:

```rust
let engine = Gliner2Engine::new(Gliner2Config {
    models_dir: "models/my-export".into(),
    max_width: 8,
    model_type: ModelType::HuggingFace,
})?;
```

`Gliner2Engine` detects whether the directory holds V1 or V2 fragments and picks
the pipeline. `GLINER2_NO_IOBINDING=1` forces the standard path on hardware
where IOBinding misbehaves.

## Known issue

The V1 pipeline still fails on some `count_lstm_fp32` exports with
`Gather node '/count_lstm/gru/Gather_3': indices element out of data bounds`.
Pre-existing and unrelated to the 0.5.2 prompt fix; the V2 IOBinding pipeline is
unaffected.

See [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md) for the history, including the
0.5.2 prompt-layout fix that changed model output.

Apache-2.0. Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
