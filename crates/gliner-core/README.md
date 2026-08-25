# gliner-core

Shared foundation for the GLiNER Rust engines. Everything here is
architecture-agnostic: identical between the **span** models (GLiNER2) and the
**boundary** models (GLiNER2.5), because gliner2 itself shares it.

You rarely depend on this directly — [`gliner2-core`](../gliner2-core) re-exports
what callers need. Depend on it when you are building a new engine on top.

| module | what it holds |
|---|---|
| `processor` | prompt construction and tokenization |
| `runtime` | `ort` helpers, precision selection, export-layout resolution |
| `overlap` | span overlap policies |
| `error` | diagnosable engine errors |

## Prompt construction

Reproduces `gliner2/processor.py::SchemaTransformer`. One schema group is
exactly:

```text
["(", "[P]", prompt_str, "("] + [child_prefix, field_name] * N + [")", ")"]
```

with `child_prefix` being `[E]` for entities, `[R]` for relations and `[L]` for
classifications. Groups are joined by `[SEP_STRUCT]`, then `[SEP_TEXT]`, then
the text.

Three details are easy to get wrong, and each one misaligns the gathered
embeddings against the format the model was trained on:

- **No `[CLS]`/`[SEP]` wrapping.** The sequence starts at the first `(`.
- **Field indices point at the marker**, not at the label name after it.
- **Text words are lower-cased** before sub-word tokenization, while character
  offsets keep indexing the original string — case folding can change length,
  so lower-casing the source first would corrupt the offsets.

`processor::tests::ground_truth_layout` pins all three against output generated
by gliner2 2.0.0. It skips when no `tokenizer.json` is reachable, since model
directories are gitignored.

## Overlap policies

```rust
use gliner_core::{OverlapPolicy, resolve_overlaps};

let kept = resolve_overlaps(&candidates, OverlapPolicy::Flat);
```

Intervals are **half-open** `[start, end)` — what the boundary architecture
produces natively. The span architecture emits inclusive `[w, w + k]` and
converts on the way in, once, in its `Spanned` implementation rather than at
each call site.

`Flat` is **not** a greedy pass: it is weighted interval scheduling, the
non-overlapping subset of maximum total score. With `A=[0,4) 0.6`,
`B=[0,2) 0.5`, `C=[2,4) 0.5`, greedy keeps `A` while the correct answer is
`B+C`. `Allow`, `Nested` and `Longest` are also implemented, matching
`gliner2/inference/overlap.py`.

## Export layouts

`runtime::resolve_fragment` accepts both directory shapes — flat with
`_fp32`/`_fp16`/`_fp16_iobinding` suffixes, and the legacy `fp32_v2/` +
`fp16_v2/` subfolders — so engines above do not each reimplement the lookup.
`runtime::resolve_tokenizer` does the same for `tokenizer.json`, which the
legacy layout keeps inside each variant subfolder.

## ort 2.0.0-rc.13

`runtime` absorbs the API changes since rc.9, so engines above do not repeat
them:

| rc.9 | rc.13 |
|---|---|
| `ort::init().commit()?` | returns `bool`, not `Result` |
| `Session::run(&self, …)` | `run(&mut self, …)`; outputs borrow the session |
| `builder.with_x()?` in `anyhow` | `Error<SessionBuilder>` is not `Send`: `.map_err(ort::Error::<()>::from)?` |
| `try_extract_tensor() -> (Vec<i64>, &[T])` | `-> (&Shape, &[T])`; ndarray view is `try_extract_array` |
| public fields on `Outlet` | `name()` / `dtype()` accessors |

Because `run` takes `&mut self`, engines own their sessions and their extraction
methods take `&mut self`.

Apache-2.0. Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
