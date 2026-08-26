// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Engine for the GLiNER2 **span** architecture, including
//! `fastino/gliner2-privacy-filter-PII-multi`.
//!
//! Chain of the eight fragments produced by `export_span_v3.py`:
//!
//! ```text
//! encoder(input_ids, attention_mask) -> last_hidden_state [1,S,H]
//!   |
//!   +- token_gather(last_hidden_state, word_indices) -> text_embs [1,W,H]
//!   |     +- span_rep(text_embs, span_idx) -> span_embeddings [1,W,mw,H]
//!   |
//!   +- schema_gather(last_hidden_state, schema_indices) -> pc_emb [1,H], field_embs [M,H]
//!         +- count_pred_argmax(pc_emb)    -> pred_count  i64
//!         +- count_lstm_fixed(field_embs) -> struct_proj [MAX_COUNT,M,H]
//!               +- scorer(span_embeddings, struct_proj) -> entity_scores [MAX_COUNT,W,mw,M]
//!
//! classifier(field_embs) -> logits [M]
//! ```
//!
//! Span `[w][k]` covers words `w` through `w+k` **inclusive**, and is valid only
//! while `w + k < W`. Invalid spans are zeroed to `(0,0)` before `span_rep`, as
//! `SpanExtractorModel.compute_span_rep` does, and discarded afterwards.
//!
//! Note the difference from the boundary architecture of GLiNER2.5, whose spans
//! are half-open `[start, end)`.

use anyhow::{Result, anyhow};
use std::path::PathBuf;

use ort::session::Session;

use crate::error::GlinerError;
use crate::overlap::{OverlapPolicy, Spanned, resolve_overlaps};
use crate::processor::{ProcessedRecord, SchemaTask, SchemaTransformer, TaskType};
use crate::chunker::Chunker;
use crate::chain::{Chain, ExecutionMode, Feed, Sink};
use crate::runtime::{
    IoDType, Precision, build_session, float_tensor, i64_tensor, resolve_fragment,
    resolve_tokenizer, sigmoid, softmax,
};

/// Inference parameters, adjustable on every call.
#[derive(Debug, Clone)]
pub struct InferenceParams {
    /// Confidence threshold on entities.
    pub threshold: f32,
    /// When `true`, greedy NMS removes every overlap regardless of label.
    /// When `false`, overlapping spans with different labels coexist
    /// (e.g. "Mario Rossi" as `person` and "Mario" as `first_name`).
    pub flat_ner: bool,
    /// Temperature applied to the classification logits.
    pub classification_temperature: f32,
    /// Overrides the per-task `multi_label` flag carried by
    /// [`SchemaTask::Classifications`]. Leave it `None` — the schema is the
    /// right place for that decision, since a single request routinely mixes
    /// single-label and multi-label tasks.
    pub multi_label_override: Option<bool>,
    /// Shared overlap policy. `None` (the default) selects the historical
    /// greedy decoder of the span architecture, governed by `flat_ner`; that is
    /// the same choice gliner2 2.0.0 makes when `overlap_policy` is left
    /// unspecified on a span checkpoint. Naming one switches to the resolver
    /// shared with the boundary architecture, and `flat_ner` is then ignored.
    pub overlap_policy: Option<OverlapPolicy>,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            flat_ner: false,
            classification_temperature: 1.0,
            multi_label_override: None,
            overlap_policy: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub score: f32,
    /// Byte range `[start, end)` in the original text.
    pub char_start: usize,
    pub char_end: usize,
    /// Inclusive word range `[start, end]`.
    pub word_start: usize,
    pub word_end: usize,
    /// Occurrence slot produced by `count_lstm` (0 = first occurrence).
    pub slot: usize,
    /// Schema group that produced the entity.
    pub task: String,
}

impl Spanned for Entity {
    fn start(&self) -> usize {
        self.word_start
    }
    /// Span-architecture spans are **inclusive**; the shared resolver works on
    /// half-open intervals, so the conversion happens here.
    fn end(&self) -> usize {
        self.word_end + 1
    }
    fn score(&self) -> f32 {
        self.score
    }
}

#[derive(Debug, Clone)]
pub struct Classification {
    pub task: String,
    pub label: String,
    pub score: f32,
}

/// Result of one span extraction.
#[derive(Debug, Clone, Default)]
pub struct SpanOutput {
    pub entities: Vec<Entity>,
    /// Every label of every classification task, with its probability. Use
    /// [`SpanOutput::verdict`] to turn one task into the answer gliner2 would
    /// give.
    pub classifications: Vec<Classification>,
}

impl SpanOutput {
    /// The labels gliner2 would report for a classification task.
    ///
    /// Reproduces `_extract_classification_result`, including the detail that
    /// is easy to miss: in multi-label mode, **when no label clears the
    /// threshold the top-scoring one is returned anyway**. The list is never
    /// empty. Thresholding the scores yourself and keeping the empty result
    /// will silently disagree with the reference implementation — for a
    /// jailbreak schema, a genuine attack scoring 0.39 against a 0.4 threshold
    /// is reported by gliner2, not dropped.
    ///
    /// In single-label mode the argmax is returned, which is the same thing.
    pub fn verdict(&self, task: &str, threshold: f32) -> Vec<&Classification> {
        let mut rows: Vec<&Classification> =
            self.classifications.iter().filter(|c| c.task == task).collect();
        if rows.is_empty() {
            return rows;
        }
        rows.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        let over: Vec<&Classification> =
            rows.iter().copied().filter(|c| c.score >= threshold).collect();
        if over.is_empty() { vec![rows[0]] } else { over }
    }
}

#[derive(Debug, Clone)]
pub struct SpanConfig {
    pub models_dir: PathBuf,
    pub precision: Precision,
    pub intra_threads: usize,
    /// How intermediate tensors travel between fragments.
    pub execution: ExecutionMode,
    /// Set when the caller named a precision, so the engine does not override
    /// it with the one the execution mode would prefer.
    precision_pinned: bool,
    /// Where to fetch the export from if `models_dir` does not hold one.
    #[cfg(feature = "hub")]
    pub hub: Option<crate::hub::Model>,
}

impl SpanConfig {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        let models_dir = models_dir.into();
        let precision = Precision::autodetect(&models_dir, "encoder");
        Self {
            models_dir,
            precision,
            intra_threads: 4,
            execution: ExecutionMode::from_env(),
            // GLINER2_PRECISION is as explicit as calling with_precision.
            precision_pinned: std::env::var("GLINER2_PRECISION").is_ok(),
            #[cfg(feature = "hub")]
            hub: None,
        }
    }

    /// Fetches the export straight from the Hub, into the shared cache.
    ///
    /// Equivalent to `SpanConfig::new(<cache>).or_download(model)`: nothing is
    /// downloaded until [`SpanEngine::new`] runs, so `with_precision` still
    /// applies to what gets fetched.
    #[cfg(feature = "hub")]
    pub fn from_hub(model: crate::hub::Model) -> Self {
        Self::new(PathBuf::new()).or_download(model)
    }

    /// Names the repository to fall back to when `models_dir` holds no export.
    ///
    /// The local directory always wins: a checkout already on disk is used as
    /// it is, and the network is touched only when the fragments are missing.
    #[cfg(feature = "hub")]
    pub fn or_download(mut self, model: crate::hub::Model) -> Self {
        self.hub = Some(model);
        self
    }

    /// Pins the export variant.
    ///
    /// Also stops the engine choosing one from the execution mode when it has
    /// to download: an explicit choice is an instruction, not a hint.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self.precision_pinned = true;
        self
    }

    pub fn with_intra_threads(mut self, n: usize) -> Self {
        self.intra_threads = n;
        self
    }

    /// Chooses the transport between fragments.
    ///
    /// Overrides `GLINER2_EXECUTION`. The default is
    /// [`ExecutionMode::Auto`]: bound on a device provider, standard on CPU.
    pub fn with_execution(mut self, mode: ExecutionMode) -> Self {
        self.execution = mode;
        self
    }
}

pub struct SpanEngine {
    encoder: Session,
    token_gather: Session,
    span_rep: Session,
    schema_gather: Session,
    count_pred: Session,
    count_lstm: Session,
    scorer: Session,
    classifier: Session,

    transformer: SchemaTransformer,
    chain: Chain,
    dtype: IoDType,
    max_width: usize,
    max_count: usize,
    hidden: usize,
    classifier_layout: ClassifierLayout,
}

/// How the exported classifier expects its input.
///
/// The model applies the classifier to the choice-marker embeddings,
/// `[M, H] -> [M]`, and `export_span_v3.py` exports exactly that. Exports made
/// by the earlier script instead carry `[batch, num_labels, max_width, H] ->
/// [batch, num_labels, max_width, 1]`, with `max_width` frozen at export time:
/// correct only because the MLP acts on the last dimension, but it forces the
/// caller to replicate the same vector `max_width` times and discard all but
/// one slice. Both are supported, detected from the input rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierLayout {
    /// `[num_labels, H] -> [num_labels]`
    Flat,
    /// `[batch, num_labels, max_width, H] -> [batch, num_labels, max_width, 1]`
    Padded,
}

impl SpanEngine {
    pub fn new(config: SpanConfig) -> Result<Self> {
        #[allow(unused_mut)]
        let mut config = config;

        // A directory that already holds the fragments is used untouched; only a
        // missing one reaches the network.
        #[cfg(feature = "hub")]
        if let Some(model) = config.hub {
            if resolve_fragment(&config.models_dir, "encoder", config.precision).is_none() {
                // Nothing on disk, so `autodetect` had no files to inspect and
                // fell back to FP32. Let the transport pick instead: a bound
                // chain wants the FP16 I/O graphs, the standard path does not.
                if !config.precision_pinned {
                    config.precision = config.execution.preferred_precision();
                }
                let (dir, got) = crate::hub::download(model, config.precision)?;
                config.models_dir = dir;
                config.precision = got;
            }
        }

        let dir = &config.models_dir;
        let precision = config.precision;

        // A GLiNER2.5 boundary export also carries an `encoder`, so without
        // this check it fails several fragments in with "count_lstm_fixed not
        // found" — true, and pointing at entirely the wrong problem. The
        // manifest is the boundary architecture's signature; a span export
        // never has one.
        for probe in [
            dir.join("boundary_manifest.json"),
            dir.join("fp32_25").join("boundary_manifest.json"),
            dir.join("fp16_25").join("boundary_manifest.json"),
        ] {
            if probe.exists() {
                return Err(GlinerError::IncompleteModelDir(format!(
                    "{} holds a GLiNER2.5 **boundary** export (boundary_manifest.json \
                     is present). This engine runs the span architecture only — \
                     use the gliner25-rs crate for this model.",
                    dir.display()
                ))
                .into());
            }
        }

        // Both layouts are accepted: flat, as produced by export_span_v3.py, and
        // the legacy `fp32_v2/` + `fp16_v2/` subfolders the earlier exporter
        // published on the Hub.
        let load = |stem: &str| -> Result<Session> {
            let path = resolve_fragment(dir, stem, precision).ok_or_else(|| {
                GlinerError::IncompleteModelDir(format!(
                    "fragment '{stem}{}' not found in {} (looked in the directory itself and in {}/)",
                    precision.suffix(),
                    dir.display(),
                    precision.legacy_subdir(),
                ))
            })?;
            build_session(&path, config.intra_threads)
        };

        let tok_path = resolve_tokenizer(dir, precision).ok_or_else(|| {
            GlinerError::IncompleteModelDir(format!(
                "tokenizer.json not found in {} nor in its variant subfolders",
                dir.display()
            ))
        })?;
        let transformer = SchemaTransformer::from_tokenizer_file(&tok_path)?;

        let encoder = load("encoder")?;
        let count_lstm = load("count_lstm_fixed")?;
        let scorer = load("scorer")?;

        // MAX_COUNT, max_width and hidden_size are read from the fragments'
        // static shapes rather than hard-coded, so a checkpoint exported with
        // different parameters stays usable.
        let (max_count, hidden) = static_dims_count_lstm(&count_lstm)?;
        let max_width = static_max_width(&scorer)?;

        let classifier = load("classifier")?;
        let classifier_layout = detect_classifier_layout(&classifier)?;

        Ok(Self {
            chain: Chain::new(config.execution, precision.io_dtype())?,
            encoder,
            token_gather: load("token_gather")?,
            span_rep: load("span_rep")?,
            schema_gather: load("schema_gather")?,
            count_pred: load("count_pred_argmax")?,
            count_lstm,
            scorer,
            classifier,
            transformer,
            dtype: config.precision.io_dtype(),
            max_width,
            max_count,
            hidden,
            classifier_layout,
        })
    }

    pub fn max_width(&self) -> usize {
        self.max_width
    }
    pub fn max_count(&self) -> usize {
        self.max_count
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }
    /// The transport actually in force, after `Auto` was resolved — and after
    /// any fallback a device OOM forced.
    pub fn execution(&self) -> ExecutionMode {
        self.chain.mode()
    }

    /// Which classifier signature the loaded export carries.
    pub fn classifier_layout(&self) -> ClassifierLayout {
        self.classifier_layout
    }

    /// Extracts over a document longer than the encoder can attend to at once.
    ///
    /// mDeBERTa-v3 is trained to 512 positions and the schema markers share
    /// that budget with the text, so past it quality degrades quietly — you get
    /// an answer, with no signal that the tail was never really read. This
    /// splits the text into overlapping word windows, runs each, shifts the
    /// offsets back onto the original, and merges the duplicates.
    ///
    /// Text that fits in one window takes the single-call path, so this is safe
    /// to use unconditionally.
    ///
    /// See [`chunker`](crate::chunker) for what merging can and cannot recover.
    pub fn extract_long(&mut self, text: &str, tasks: &[SchemaTask]) -> Result<SpanOutput> {
        self.extract_long_with(text, tasks, &InferenceParams::default(), Chunker::default())
    }

    /// [`extract_long`](Self::extract_long) with the window geometry spelled out.
    pub fn extract_long_with(
        &mut self,
        text: &str,
        tasks: &[SchemaTask],
        params: &InferenceParams,
        chunker: Chunker,
    ) -> Result<SpanOutput> {
        let chunks = chunker.split(text)?;
        if chunks.len() <= 1 {
            return self.extract_with(text, tasks, params);
        }
        let mut parts = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let mut part = self.extract_with(chunk.slice(text), tasks, params)?;
            crate::chunker::remap(&mut part, chunk, text);
            parts.push(part);
        }
        Ok(crate::chunker::merge(parts))
    }

    pub fn extract(&mut self, text: &str, tasks: &[SchemaTask]) -> Result<SpanOutput> {
        self.extract_with(text, tasks, &InferenceParams::default())
    }

    /// Runs the chain, and on a device allocation failure runs it again on the
    /// standard path.
    ///
    /// Binding holds every intermediate on the device at once, so it is the
    /// first thing to give way on a long input — `span_rep` alone asks for
    /// hundreds of megabytes once the word count climbs. Failing the call there
    /// would be the wrong answer: the standard path releases each tensor as
    /// soon as the next fragment has consumed it and will very often succeed on
    /// exactly the input that broke binding. The engine stays on the standard
    /// path afterwards rather than paying the same failure on every call.
    pub fn extract_with(
        &mut self,
        text: &str,
        tasks: &[SchemaTask],
        params: &InferenceParams,
    ) -> Result<SpanOutput> {
        match self.extract_once(text, tasks, params) {
            Err(e)
                if self.chain.mode() == ExecutionMode::IoBinding
                    && matches!(
                        e.downcast_ref::<GlinerError>(),
                        Some(GlinerError::OomDeviceBinding(_) | GlinerError::BindingNotSupported(_))
                    ) =>
            {
                eprintln!("[gliner2] {e}; continuing on the standard path");
                self.chain.fall_back();
                self.extract_once(text, tasks, params)
                    .map_err(|again| self.explain_oom(text, again))
            }
            Err(e) => Err(self.explain_oom(text, e)),
            other => other,
        }
    }

    /// Adds the one thing the runtime's own message never says: what to do.
    ///
    /// `span_rep` enumerates `num_words × max_width` spans, so the memory a call
    /// needs grows with the text and there is a length past which no transport
    /// helps — on an RTX 3090 with this model that is somewhere between four and
    /// five thousand words. ORT reports it as a failed allocation of N bytes,
    /// which is true and useless.
    fn explain_oom(&self, text: &str, err: anyhow::Error) -> anyhow::Error {
        let is_oom = matches!(
            err.downcast_ref::<GlinerError>(),
            Some(GlinerError::OomDeviceBinding(_) | GlinerError::OomDeviceStandard(_))
        );
        if !is_oom {
            return err;
        }
        let words = crate::processor::WhitespaceTokenSplitter::new()
            .map(|s| s.split_with_offsets(text).len())
            .unwrap_or(0);
        anyhow::anyhow!(
            "{err}\n\n\
             The text is {words} words. One call builds num_words x max_width \
             ({mw}) span representations, so past a few thousand words no execution \
             mode fits in device memory. Use extract_long(), which windows the text \
             and merges the results.",
            mw = self.max_width,
        )
    }

    fn extract_once(
        &mut self,
        text: &str,
        tasks: &[SchemaTask],
        params: &InferenceParams,
    ) -> Result<SpanOutput> {
        let record = self.transformer.transform(text, tasks)?;
        let num_words = record.num_words();
        if num_words == 0 {
            return Ok(SpanOutput::default());
        }

        // ── the chain, written once ───────────────────────────────────────
        //
        // Each step hands `self.chain` its inputs in the order the graph
        // declares them and says where each output should land. The chain
        // decides whether that means a host round trip or a device binding, so
        // the two transports cannot drift apart the way they did when they
        // lived in separate crates.
        let seq = record.input_ids.len() as i64;
        let hidden_shape = vec![1, seq, self.hidden as i64];

        // 1. encoder — consumed by token_gather and, per group, schema_gather.
        let hidden = self
            .chain
            .run(
                &mut self.encoder,
                &[
                    Feed::Owned(i64_tensor(vec![1, seq], record.input_ids.clone())?),
                    Feed::Owned(i64_tensor(vec![1, seq], record.attention_mask.clone())?),
                ],
                &[Sink::Device],
            )?
            .remove(0);

        // 2. token_gather -> text_embs
        let text_embs = self
            .chain
            .run(
                &mut self.token_gather,
                &[
                    Feed::Carried(&hidden, hidden_shape.clone()),
                    Feed::Owned(i64_tensor(vec![num_words as i64], record.word_first_positions())?),
                ],
                &[Sink::Device],
            )?
            .remove(0);

        // 3. span_rep -> span_embeddings
        let n_spans = (num_words * self.max_width) as i64;
        let span_embeddings = self
            .chain
            .run(
                &mut self.span_rep,
                &[
                    Feed::Carried(&text_embs, vec![1, num_words as i64, self.hidden as i64]),
                    Feed::Owned(i64_tensor(
                        vec![1, n_spans, 2],
                        build_span_idx(num_words, self.max_width),
                    )?),
                ],
                &[Sink::Device],
            )?
            .remove(0);
        let span_shape = vec![1, num_words as i64, self.max_width as i64, self.hidden as i64];

        // 4. one pass per schema group
        let mut result = SpanOutput::default();
        for task in &record.tasks {
            let m = task.labels.len();
            if m == 0 {
                continue;
            }

            let mut schema_indices = Vec::with_capacity(m + 1);
            schema_indices.push(task.prompt_tok_idx as i64);
            schema_indices.extend(task.field_tok_indices.iter().map(|i| *i as i64));

            // A classification group reads `field_embs` on the host to pad it,
            // so ask for it there rather than copying it back afterwards.
            let field_sink = match task.task_type {
                TaskType::Classifications => Sink::Host,
                _ => Sink::Device,
            };
            let mut gathered = self.chain.run(
                &mut self.schema_gather,
                &[
                    Feed::Carried(&hidden, hidden_shape.clone()),
                    Feed::Owned(i64_tensor(vec![schema_indices.len() as i64], schema_indices)?),
                ],
                // declared order: pc_emb, field_embs
                &[Sink::Device, field_sink],
            )?;
            let field_embs = gathered.remove(1);
            let pc_emb = gathered.remove(0);
            let field_shape = vec![m as i64, self.hidden as i64];

            match task.task_type {
                TaskType::Classifications => {
                    let logits = match self.classifier_layout {
                        ClassifierLayout::Flat => self
                            .chain
                            .run(
                                &mut self.classifier,
                                &[Feed::Carried(&field_embs, field_shape.clone())],
                                &[Sink::Host],
                            )?
                            .remove(0)
                            .host(self.dtype)?,
                        ClassifierLayout::Padded => {
                            // The legacy graph wants [1, M, max_width, H]. The MLP
                            // acts on the last dimension, so replicating each row
                            // across max_width and reading back slice 0 is exact —
                            // just max_width times the arithmetic for one result.
                            let mw = self.max_width;
                            let flat_embs = field_embs.host(self.dtype)?;
                            let mut padded = vec![0.0f32; m * mw * self.hidden];
                            for label in 0..m {
                                let src = &flat_embs[label * self.hidden..(label + 1) * self.hidden];
                                for w in 0..mw {
                                    let base = (label * mw + w) * self.hidden;
                                    padded[base..base + self.hidden].copy_from_slice(src);
                                }
                            }
                            let fe = float_tensor(
                                self.dtype,
                                vec![1, m as i64, mw as i64, self.hidden as i64],
                                padded,
                            )?;
                            let flat = self
                                .chain
                                .run(&mut self.classifier, &[Feed::Owned(fe)], &[Sink::Host])?
                                .remove(0)
                                .host(self.dtype)?;
                            // [1, M, mw, 1] -> one logit per label
                            (0..m).map(|label| flat[label * mw]).collect()
                        }
                    };
                    let scaled: Vec<f32> = logits
                        .iter()
                        .map(|l| l / params.classification_temperature)
                        .collect();
                    let multi_label = params.multi_label_override.unwrap_or(task.multi_label);
                    let probs = if multi_label {
                        scaled.iter().copied().map(sigmoid).collect::<Vec<_>>()
                    } else {
                        softmax(&scaled)
                    };
                    for (label, score) in task.labels.iter().zip(probs) {
                        result.classifications.push(Classification {
                            task: task.task_name.clone(),
                            label: label.clone(),
                            score,
                        });
                    }
                }
                TaskType::Entities | TaskType::Relations => {
                    let pred_count = self
                        .chain
                        .run(
                            &mut self.count_pred,
                            &[Feed::Carried(&pc_emb, vec![1, self.hidden as i64])],
                            &[Sink::HostI64],
                        )?
                        .remove(0)
                        .host_i64()?[0]
                        .max(0) as usize;
                    let pred_count = pred_count.min(self.max_count);
                    if pred_count == 0 {
                        continue;
                    }

                    let struct_proj = self
                        .chain
                        .run(
                            &mut self.count_lstm,
                            &[Feed::Carried(&field_embs, field_shape.clone())],
                            &[Sink::Device],
                        )?
                        .remove(0);

                    let scores = self
                        .chain
                        .run(
                            &mut self.scorer,
                            &[
                                Feed::Carried(&span_embeddings, span_shape.clone()),
                                Feed::Carried(
                                    &struct_proj,
                                    vec![self.max_count as i64, m as i64, self.hidden as i64],
                                ),
                            ],
                            &[Sink::Host],
                        )?
                        .remove(0)
                        .host(self.dtype)?;

                    self.collect_entities(
                        &scores,
                        &record,
                        task,
                        m,
                        pred_count,
                        num_words,
                        text,
                        params,
                        &mut result.entities,
                    );
                }
            }
        }

        Ok(result)
    }

    /// `entity_scores` is laid out `[MAX_COUNT, W, max_width, M]` and has
    /// already gone through the sigmoid inside the scorer graph.
    #[allow(clippy::too_many_arguments)]
    fn collect_entities(
        &self,
        scores: &[f32],
        record: &ProcessedRecord,
        task: &crate::processor::TaskMapping,
        m: usize,
        pred_count: usize,
        num_words: usize,
        text: &str,
        params: &InferenceParams,
        out: &mut Vec<Entity>,
    ) {
        let mw = self.max_width;
        let stride_c = num_words * mw * m;
        let stride_w = mw * m;

        let mut found: Vec<Entity> = Vec::new();
        for c in 0..pred_count {
            for w in 0..num_words {
                for k in 0..mw {
                    // the span covers w..=w+k, valid only while inside the text
                    if w + k >= num_words {
                        break;
                    }
                    for label_idx in 0..m {
                        let score = scores[c * stride_c + w * stride_w + k * m + label_idx];
                        if score < params.threshold {
                            continue;
                        }
                        let (cs, _) = record.word_to_char_maps[w];
                        let (_, ce) = record.word_to_char_maps[w + k];
                        found.push(Entity {
                            text: text[cs..ce].to_string(),
                            label: task.labels[label_idx].clone(),
                            score,
                            char_start: cs,
                            char_end: ce,
                            word_start: w,
                            word_end: w + k,
                            slot: c,
                            task: task.task_name.clone(),
                        });
                    }
                }
            }
        }

        match params.overlap_policy {
            None => out.extend(greedy_nms(found, params.flat_ner)),
            // Python scopes `finalize_spans` per entity name — the resolver
            // runs inside `for name in entity_order`, one label's spans at a
            // time — so labels never interact under an explicit policy either
            // (gliner2 2.0.0, inference/runtime.py). Resolving across the
            // whole task at once would collapse the same stretch under two
            // labels, which the reference deliberately keeps.
            Some(policy) => {
                let mut by_label: std::collections::BTreeMap<String, Vec<Entity>> =
                    Default::default();
                for e in found {
                    by_label.entry(e.label.clone()).or_default().push(e);
                }
                for (_, group) in by_label {
                    out.extend(resolve_overlaps(&group, policy));
                }
            }
        }
    }
}

/// Greedy suppression by descending score, reproducing gliner2's default.
///
/// The reference implementation decodes **one label at a time**: for each entity
/// name it thresholds the scores, sorts by confidence and drops any candidate
/// overlapping one already kept *for that same name*. Labels never interact, so
/// the same span can legitimately come back under several of them — a doctor's
/// name is both `medical_professional` and `person`, and a redaction pipeline
/// wants to know both.
///
/// `flat_ner = true` is a deliberate extension, not the reference behaviour: it
/// suppresses across labels too, leaving one label per stretch of text.
fn greedy_nms(mut candidates: Vec<Entity>, flat_ner: bool) -> Vec<Entity> {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.word_start.cmp(&b.word_start))
            .then(a.word_end.cmp(&b.word_end))
            .then(a.label.cmp(&b.label))
    });

    let mut kept: Vec<Entity> = Vec::new();
    for cand in candidates {
        let suppressed = kept.iter().any(|k| {
            if !flat_ner && k.label != cand.label {
                return false;
            }
            // inclusive word ranges overlap when neither ends before the other starts
            cand.word_start <= k.word_end && k.word_start <= cand.word_end
        });
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// `span_idx[w * max_width + k] = (w, w + k)`, with `(0,0)` on invalid spans.
fn build_span_idx(num_words: usize, max_width: usize) -> Vec<i64> {
    let mut spans = Vec::with_capacity(num_words * max_width * 2);
    for w in 0..num_words {
        for k in 0..max_width {
            let end = w + k;
            if end < num_words {
                spans.push(w as i64);
                spans.push(end as i64);
            } else {
                spans.push(0);
                spans.push(0);
            }
        }
    }
    spans
}

/// Detects which classifier signature the export carries, from the input rank.
fn detect_classifier_layout(session: &Session) -> Result<ClassifierLayout> {
    let inp = session
        .inputs()
        .first()
        .ok_or_else(|| anyhow!("the classifier fragment declares no input"))?;
    let rank = tensor_dims(inp.dtype())?.len();
    match rank {
        2 => Ok(ClassifierLayout::Flat),
        4 => Ok(ClassifierLayout::Padded),
        other => Err(anyhow!(
            "unexpected classifier input rank {other} (expected 2 for [num_labels, H] \
             or 4 for [batch, num_labels, max_width, H])"
        )),
    }
}

/// Reads `MAX_COUNT` and `hidden_size` from `struct_proj`'s static shape.
fn static_dims_count_lstm(session: &Session) -> Result<(usize, usize)> {
    let out = session
        .outputs()
        .iter()
        .find(|o| o.name() == "struct_proj")
        .ok_or_else(|| anyhow!("count_lstm_fixed does not expose struct_proj"))?;
    let dims = tensor_dims(out.dtype())?;
    let max_count = dims
        .first()
        .and_then(|d| *d)
        .ok_or_else(|| anyhow!("struct_proj: MAX_COUNT dimension is not static"))?;
    let hidden = dims
        .get(2)
        .and_then(|d| *d)
        .ok_or_else(|| anyhow!("struct_proj: hidden_size is not static"))?;
    Ok((max_count as usize, hidden as usize))
}

/// Reads `max_width` from the scorer's `span_embeddings` static shape.
fn static_max_width(session: &Session) -> Result<usize> {
    let inp = session
        .inputs()
        .iter()
        .find(|i| i.name() == "span_embeddings")
        .ok_or_else(|| anyhow!("scorer does not accept span_embeddings"))?;
    let dims = tensor_dims(inp.dtype())?;
    dims.get(2)
        .and_then(|d| *d)
        .map(|d| d as usize)
        .ok_or_else(|| anyhow!("span_embeddings: max_width is not static"))
}

fn tensor_dims(vt: &ort::value::ValueType) -> Result<Vec<Option<i64>>> {
    match vt {
        ort::value::ValueType::Tensor { shape, .. } => Ok(shape
            .iter()
            .map(|d| if *d < 0 { None } else { Some(*d) })
            .collect()),
        other => Err(anyhow!("expected a tensor, found {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_idx_layout() {
        // 3 words, max_width 2: (0,0) (0,1) (1,1) (1,2) (2,2) invalid -> (0,0)
        assert_eq!(build_span_idx(3, 2), vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0]);
    }

    #[test]
    fn explicit_policy_is_scoped_per_label_like_python() {
        // gliner2's runtime applies finalize_spans inside `for name in
        // entity_order`, so the same stretch under two labels survives any
        // policy — Flat included, which would collapse them if it ran across
        // the task.
        let e = |ws: usize, we: usize, label: &str, score: f32| Entity {
            text: String::new(),
            label: label.into(),
            score,
            char_start: 0,
            char_end: 0,
            word_start: ws,
            word_end: we,
            slot: 0,
            task: "entities".into(),
        };
        let found = vec![
            e(0, 1, "person", 0.9),
            e(0, 1, "medical_professional", 0.8),
            // and inside one label, Flat still resolves: the pair beats the whole
            e(3, 6, "location", 0.6),
            e(3, 4, "location", 0.5),
            e(5, 6, "location", 0.5),
        ];
        let mut by_label: std::collections::BTreeMap<String, Vec<Entity>> = Default::default();
        for x in found {
            by_label.entry(x.label.clone()).or_default().push(x);
        }
        let mut out: Vec<Entity> = Vec::new();
        for (_, group) in by_label {
            out.extend(resolve_overlaps(&group, OverlapPolicy::Flat));
        }
        let labels: Vec<&str> = out.iter().map(|x| x.label.as_str()).collect();
        assert!(labels.contains(&"person"));
        assert!(labels.contains(&"medical_professional"));
        // inclusive spans [3..=4] and [5..=6] are disjoint; half-open resolver
        // sees [3,5) and [5,7): total 1.0 beats the whole span's 0.6
        assert_eq!(out.iter().filter(|x| x.label == "location").count(), 2);
    }

    #[test]
    fn nms_suppresses_within_label_only() {
        let e = |ws, we, label: &str, score| Entity {
            text: String::new(),
            label: label.into(),
            score,
            char_start: 0,
            char_end: 0,
            word_start: ws,
            word_end: we,
            slot: 0,
            task: "entities".into(),
        };
        // flat_ner suppresses across labels
        let kept = greedy_nms(vec![e(0, 1, "person", 0.9), e(0, 0, "first_name", 0.8)], true);
        assert_eq!(kept.len(), 1);
        // by default labels never interact, so a nested span of another label stays
        let kept = greedy_nms(vec![e(0, 1, "person", 0.9), e(0, 0, "first_name", 0.8)], false);
        assert_eq!(kept.len(), 2);
        // the same span under two labels is legitimate: a doctor is both
        // `medical_professional` and `person`
        let kept = greedy_nms(vec![e(0, 1, "medical_professional", 0.9), e(0, 1, "person", 0.8)], false);
        assert_eq!(kept.len(), 2);
        // within one label, the weaker overlapping span is dropped
        let kept = greedy_nms(vec![e(0, 1, "person", 0.9), e(1, 2, "person", 0.8)], false);
        assert_eq!(kept.len(), 1);
    }
}
