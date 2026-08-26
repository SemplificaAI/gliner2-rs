// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Extraction over documents longer than the model can see at once.
//!
//! The span engine has no length buckets, but its encoder does have a position
//! limit — mDeBERTa-v3 is trained to 512 positions — and the schema markers
//! share that budget with the text. Past it, quality degrades rather than
//! failing loudly, which is worse: you get an answer, and no signal that the
//! tail of the document was never really read.
//!
//! This module does what `gliner2.inference.chunking` does on the Python side:
//! splits the text into overlapping word windows, runs each one, shifts the
//! offsets back onto the original document, and merges what the windows have in
//! common.
//!
//! ```no_run
//! use gliner2_rs::{SpanConfig, SpanEngine, SchemaTask};
//!
//! let document = std::fs::read_to_string("contract.txt")?;
//! let mut engine = SpanEngine::new(SpanConfig::new("models/g25"))?;
//! let tasks = vec![SchemaTask::Entities(vec!["person".into(), "location".into()])];
//! let out = engine.extract_long(&document, &tasks)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Why the windows overlap
//!
//! A mention straddling a window edge is seen by neither window whole. The
//! overlap is what gives it a second chance: with 64 words of margin, anything
//! shorter than that appears intact in at least one window. Widening the
//! overlap costs inference time — it is the fraction of the document processed
//! twice — and narrowing it starts losing mentions at the seams.
//!
//! ## What merging can and cannot fix
//!
//! Duplicate mentions from overlapping windows are collapsed by span, keeping
//! the highest score. Classifications are collapsed per label, also by highest
//! score: for a guardrail that is the answer you want — one flagged window
//! means a flagged document — and for a descriptive label it is optimistic, so
//! read a document-level classification as "somewhere in here", not "overall".
//!
//! What no merge can recover is an entity longer than the overlap, or a
//! relation whose two ends fall in different windows. Both are inherent to
//! chunking rather than to this implementation.

use crate::span::{Entity, SpanOutput};
use crate::processor::WhitespaceTokenSplitter;
use anyhow::{Result, anyhow};
use std::collections::HashMap;

/// One window over the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Byte range `[start, end)` of this window in the original text.
    pub byte_start: usize,
    pub byte_end: usize,
    /// Half-open word range `[start, end)` in the original text.
    pub word_start: usize,
    pub word_end: usize,
}

impl Chunk {
    /// The window's own text.
    pub fn slice<'a>(&self, text: &'a str) -> &'a str {
        &text[self.byte_start..self.byte_end]
    }
}

/// How a document is cut into windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunker {
    size: usize,
    overlap: usize,
}

impl Default for Chunker {
    /// 384 words with 64 of overlap — the defaults `gliner2` uses.
    ///
    /// 384 leaves room under the encoder's 512 positions for the schema markers
    /// the prompt adds, which are counted against the same budget.
    fn default() -> Self {
        Self { size: 384, overlap: 64 }
    }
}

impl Chunker {
    pub fn new(size: usize, overlap: usize) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!("chunk size must be greater than 0"));
        }
        if overlap >= size {
            return Err(anyhow!(
                "chunk overlap ({overlap}) must be smaller than the size ({size}); \
                 equal or larger and the window never advances"
            ));
        }
        Ok(Self { size, overlap })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn overlap(&self) -> usize {
        self.overlap
    }

    /// Cuts `text` into overlapping word windows.
    ///
    /// Words are counted with the same splitter the engine tokenises with, so a
    /// window of `size` words is a window of `size` words as the model will
    /// count them — not as whitespace would.
    pub fn split(&self, text: &str) -> Result<Vec<Chunk>> {
        let splitter = WhitespaceTokenSplitter::new()?;
        let words = splitter.split_with_offsets(text);
        if words.is_empty() {
            return Ok(vec![Chunk {
                byte_start: 0,
                byte_end: text.len(),
                word_start: 0,
                word_end: 0,
            }]);
        }

        let step = self.size - self.overlap;
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < words.len() {
            let end = (start + self.size).min(words.len());
            chunks.push(Chunk {
                byte_start: words[start].1,
                byte_end: words[end - 1].2,
                word_start: start,
                word_end: end,
            });
            if end == words.len() {
                break;
            }
            start += step;
        }
        Ok(chunks)
    }
}

/// Shifts a window's output onto the original document.
pub fn remap(output: &mut SpanOutput, chunk: &Chunk, text: &str) {
    for m in &mut output.entities {
        m.char_start += chunk.byte_start;
        m.char_end += chunk.byte_start;
        m.word_start += chunk.word_start;
        m.word_end += chunk.word_start;
        // Re-slice rather than trust the window's copy: identical in practice,
        // but it keeps `text` and the offsets from ever disagreeing.
        if let Some(s) = text.get(m.char_start..m.char_end) {
            m.text = s.to_string();
        }
    }
}

/// Collapses what overlapping windows saw twice.
///
/// Two passes. Identical spans are keyed by `(range, task, label)` and the
/// highest score wins. Then, within each `(task, label)`, *overlapping* spans
/// are resolved greedily by score — the seam case, where one window saw
/// `Mario` at its edge and the neighbouring window saw `Mario Rossi` whole,
/// and both survived the first pass because their ranges differ. A single
/// window never produces such a pair (the engine's own NMS removed it), so
/// this pass only ever removes seam artefacts. `gliner2`'s
/// `merge_chunk_results` resolves overlaps at merge for the same reason.
///
/// Labels never interact, exactly as in single-window decoding: the same
/// stretch under two labels is information, not a duplicate.
///
/// Classifications are collapsed per `(task, label)` by highest score. Order is
/// by score descending, so the result reads the way a single-call result does.
pub fn merge(parts: Vec<SpanOutput>) -> SpanOutput {
    let mut entities: HashMap<(usize, usize, String, String), Entity> = HashMap::new();
    let mut classes: HashMap<(String, String), crate::span::Classification> = HashMap::new();

    for part in parts {
        for m in part.entities {
            let key = (m.char_start, m.char_end, m.task.clone(), m.label.clone());
            match entities.get(&key) {
                Some(seen) if seen.score >= m.score => {}
                _ => {
                    entities.insert(key, m);
                }
            }
        }
        for c in part.classifications {
            let key = (c.task.clone(), c.label.clone());
            match classes.get(&key) {
                Some(seen) if seen.score >= c.score => {}
                _ => {
                    classes.insert(key, c);
                }
            }
        }
    }

    // Seam pass: greedy by score within each (task, label), spans inclusive.
    let mut entities: Vec<Entity> = entities.into_values().collect();
    entities.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.word_start.cmp(&b.word_start))
            .then(a.word_end.cmp(&b.word_end))
    });
    let mut kept: Vec<Entity> = Vec::new();
    for cand in entities {
        let clashes = kept.iter().any(|k| {
            k.task == cand.task
                && k.label == cand.label
                && cand.word_start <= k.word_end
                && k.word_start <= cand.word_end
        });
        if !clashes {
            kept.push(cand);
        }
    }
    let mut entities = kept;
    entities.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.char_start.cmp(&b.char_start))
    });
    let mut classifications: Vec<crate::span::Classification> =
        classes.into_values().collect();
    classifications.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    SpanOutput { entities, classifications }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_advance_by_size_minus_overlap() {
        let text = (0..10).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let chunks = Chunker::new(4, 1).unwrap().split(&text).unwrap();
        let spans: Vec<(usize, usize)> =
            chunks.iter().map(|c| (c.word_start, c.word_end)).collect();
        assert_eq!(spans, vec![(0, 4), (3, 7), (6, 10)]);
    }

    #[test]
    fn every_word_is_covered() {
        let text = (0..97).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let chunks = Chunker::new(16, 4).unwrap().split(&text).unwrap();
        let mut covered = [false; 97];
        for c in &chunks {
            covered[c.word_start..c.word_end].fill(true);
        }
        assert!(covered.iter().all(|c| *c), "a window boundary dropped a word");
    }

    #[test]
    fn short_text_is_one_window() {
        let chunks = Chunker::default().split("Mario Rossi lavora a Roma.").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].word_start, 0);
    }

    #[test]
    fn empty_text_still_yields_a_window() {
        assert_eq!(Chunker::default().split("").unwrap().len(), 1);
    }

    fn ent(label: &str, cs: usize, ce: usize, ws: usize, we: usize, score: f32) -> Entity {
        Entity {
            text: String::new(),
            label: label.into(),
            score,
            char_start: cs,
            char_end: ce,
            word_start: ws,
            word_end: we,
            slot: 0,
            task: "entities".into(),
        }
    }

    #[test]
    fn merge_collapses_seam_truncations_within_a_label() {
        // window A saw "Mario" at its edge (word 5, inclusive range 5..=5);
        // window B saw "Mario Rossi" whole (words 5..=6). Different ranges, so
        // key dedup keeps both — the seam pass must drop the truncation.
        let a = SpanOutput {
            entities: vec![ent("person", 30, 35, 5, 5, 0.71)],
            classifications: vec![],
        };
        let b = SpanOutput {
            entities: vec![ent("person", 30, 41, 5, 6, 0.97)],
            classifications: vec![],
        };
        let merged = merge(vec![a, b]);
        assert_eq!(merged.entities.len(), 1);
        assert_eq!(merged.entities[0].word_end, 6, "the whole mention wins");
    }

    #[test]
    fn merge_keeps_the_same_stretch_under_two_labels() {
        // labels never interact: a doctor is both medical_professional and
        // person, in one window or across two.
        let a = SpanOutput {
            entities: vec![ent("person", 30, 41, 5, 6, 0.91)],
            classifications: vec![],
        };
        let b = SpanOutput {
            entities: vec![ent("medical_professional", 30, 41, 5, 6, 0.95)],
            classifications: vec![],
        };
        assert_eq!(merge(vec![a, b]).entities.len(), 2);
    }

    #[test]
    fn merge_keeps_distinct_occurrences_of_one_label() {
        // two genuinely different mentions of the same label do not overlap
        // and must both survive.
        let a = SpanOutput {
            entities: vec![ent("person", 0, 5, 0, 0, 0.9)],
            classifications: vec![],
        };
        let b = SpanOutput {
            entities: vec![ent("person", 50, 55, 9, 9, 0.9)],
            classifications: vec![],
        };
        assert_eq!(merge(vec![a, b]).entities.len(), 2);
    }

    #[test]
    fn overlap_must_be_smaller_than_size() {
        assert!(Chunker::new(64, 64).is_err());
        assert!(Chunker::new(64, 65).is_err());
        assert!(Chunker::new(0, 0).is_err());
    }
}
