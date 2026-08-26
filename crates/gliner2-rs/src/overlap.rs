// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Span overlap policies, equivalent to
//! `gliner2/inference/overlap.py::resolve_overlaps`.
//!
//! Spans are **half-open**: `[start, end)`.
//!
//! Note that `flat` is not a greedy NMS. It is weighted interval scheduling —
//! the non-overlapping subset of **maximum total score**. A greedy pass by
//! descending score gives different answers: with `A=[0,4) score 0.6`,
//! `B=[0,2) score 0.5` and `C=[2,4) score 0.5`, greedy keeps only `A` (0.6)
//! while the DP keeps `B+C` (1.0).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapPolicy {
    /// Keeps every distinct span.
    Allow,
    /// Allows disjoint and nested spans, rejects crossings.
    Nested,
    /// Non-overlapping subset of maximum total score.
    Flat,
    /// Drops spans strictly contained in another candidate.
    Longest,
}

impl OverlapPolicy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().replace('-', "_").as_str() {
            "none" | "allow" | "all" => Some(Self::Allow),
            "nested" | "allow_nested" => Some(Self::Nested),
            "flat" | "disallow" | "no_overlap" | "non_overlapping" => Some(Self::Flat),
            "longest" | "keep_longest" => Some(Self::Longest),
            _ => None,
        }
    }
}

/// A candidate span, carrying just enough to resolve overlaps.
pub trait Spanned {
    fn start(&self) -> usize;
    /// **Exclusive** end.
    fn end(&self) -> usize;
    fn score(&self) -> f32;
}

/// Stable ordering key: descending score, then start, end, index.
///
/// Python compares the float directly; this quantises to 1e-9 to get an `Ord`
/// key. The two can only disagree when two *distinct* scores fall in the same
/// 1e-9 bucket — impossible for f32 probabilities at or above any practical
/// threshold, whose spacing near 0.4 is ~3e-8, thirty times coarser than the
/// bucket. Scores small enough to collide are filtered out before they reach
/// the resolver.
fn rank_key<T: Spanned>(index: usize, item: &T) -> (i64, usize, usize, usize) {
    let neg = -((item.score() as f64 * 1e9) as i64);
    (neg, item.start(), item.end(), index)
}

pub fn resolve_overlaps<T: Spanned + Clone>(items: &[T], policy: OverlapPolicy) -> Vec<T> {
    if items.is_empty() {
        return Vec::new();
    }

    // ── 1. rank, then collapse exact-boundary duplicates ───────────────────
    let mut ranked: Vec<usize> = (0..items.len()).collect();
    ranked.sort_by_key(|&i| rank_key(i, &items[i]));

    let mut distinct: Vec<usize> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &i in &ranked {
        let key = (items[i].start(), items[i].end());
        if seen.insert(key) {
            distinct.push(i);
        }
    }

    match policy {
        OverlapPolicy::Allow => distinct.into_iter().map(|i| items[i].clone()).collect(),

        OverlapPolicy::Nested => {
            let mut kept: Vec<usize> = Vec::new();
            for &i in &distinct {
                let (cs, ce) = (items[i].start(), items[i].end());
                let crossing = kept.iter().any(|&j| {
                    let (es, ee) = (items[j].start(), items[j].end());
                    let overlaps = cs < ee && es < ce;
                    let contains = (cs <= es && ee <= ce) || (es <= cs && ce <= ee);
                    overlaps && !contains
                });
                if !crossing {
                    kept.push(i);
                }
            }
            kept.into_iter().map(|i| items[i].clone()).collect()
        }

        OverlapPolicy::Longest => distinct
            .iter()
            .filter(|&&i| {
                let (cs, ce) = (items[i].start(), items[i].end());
                !distinct.iter().any(|&j| {
                    let (os, oe) = (items[j].start(), items[j].end());
                    os <= cs && ce <= oe && (os < cs || ce < oe)
                })
            })
            .map(|&i| items[i].clone())
            .collect(),

        OverlapPolicy::Flat => flat_schedule(items, &distinct),
    }
}

/// Weighted interval scheduling, with ties broken exactly as Python does.
fn flat_schedule<T: Spanned + Clone>(items: &[T], distinct: &[usize]) -> Vec<T> {
    // order by ascending right endpoint
    let mut by_end: Vec<usize> = distinct.to_vec();
    by_end.sort_by_key(|&i| {
        let it = &items[i];
        (it.end(), it.start(), -((it.score() as f64 * 1e9) as i64), i)
    });

    let ends: Vec<usize> = by_end.iter().map(|&i| items[i].end()).collect();

    // predecessors[k] = last interval ending at or before k's start
    let predecessors: Vec<isize> = by_end
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let start = items[i].start();
            // bisect_right(ends, start, 0, k) - 1
            let mut lo = 0usize;
            let mut hi = k;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if ends[mid] <= start { lo = mid + 1 } else { hi = mid }
            }
            lo as isize - 1
        })
        .collect();

    // best[k] = best solution over the first k intervals
    let mut best: Vec<(f64, Vec<usize>)> = Vec::with_capacity(by_end.len() + 1);
    best.push((0.0, Vec::new()));

    let selection_key = |sel: &[usize]| -> Vec<(i64, usize, usize, usize)> {
        let mut rows: Vec<(i64, usize, usize, usize)> = sel
            .iter()
            .map(|&k| {
                let i = by_end[k];
                rank_key(i, &items[i])
            })
            .collect();
        rows.sort();
        rows
    };

    for (k, &i) in by_end.iter().enumerate() {
        let prev_idx = (predecessors[k] + 1) as usize;
        let (prev_score, prev_sel) = best[prev_idx].clone();
        let mut with_sel = prev_sel;
        with_sel.push(k);
        let with = (prev_score + items[i].score() as f64, with_sel);
        let without = best[k].clone();

        let chosen = if with.0 > without.0 {
            with
        } else if with.0 < without.0 {
            without
        } else if with.1.len() > without.1.len() {
            // missing confidences are represented as zero when merging chunks:
            // prefer the largest compatible set over selecting nothing
            with
        } else if with.1.len() < without.1.len() {
            without
        } else if selection_key(&with.1) < selection_key(&without.1) {
            with
        } else {
            without
        };
        best.push(chosen);
    }

    let mut selected: Vec<usize> = best.last().unwrap().1.iter().map(|&k| by_end[k]).collect();
    selected.sort_by_key(|&i| rank_key(i, &items[i]));
    selected.into_iter().map(|i| items[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct S(usize, usize, f32);
    impl Spanned for S {
        fn start(&self) -> usize { self.0 }
        fn end(&self) -> usize { self.1 }
        fn score(&self) -> f32 { self.2 }
    }

    #[test]
    fn flat_maximises_total_score_rather_than_being_greedy() {
        // greedy would keep only A (0.6); the DP keeps B+C (1.0)
        let items = vec![S(0, 4, 0.6), S(0, 2, 0.5), S(2, 4, 0.5)];
        let kept = resolve_overlaps(&items, OverlapPolicy::Flat);
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&S(0, 2, 0.5)));
        assert!(kept.contains(&S(2, 4, 0.5)));
    }

    #[test]
    fn flat_keeps_the_best_when_incompatible() {
        let items = vec![S(0, 4, 0.9), S(1, 3, 0.5)];
        assert_eq!(resolve_overlaps(&items, OverlapPolicy::Flat), vec![S(0, 4, 0.9)]);
    }

    #[test]
    fn nested_allows_containment_rejects_crossing() {
        let nested = vec![S(0, 4, 0.9), S(1, 3, 0.5)];
        assert_eq!(resolve_overlaps(&nested, OverlapPolicy::Nested).len(), 2);
        let crossing = vec![S(0, 3, 0.9), S(2, 5, 0.5)];
        assert_eq!(resolve_overlaps(&crossing, OverlapPolicy::Nested).len(), 1);
    }

    #[test]
    fn longest_drops_strictly_contained() {
        let items = vec![S(0, 4, 0.5), S(1, 3, 0.9)];
        assert_eq!(resolve_overlaps(&items, OverlapPolicy::Longest), vec![S(0, 4, 0.5)]);
    }

    #[test]
    fn boundary_duplicates_collapse_onto_the_best() {
        let items = vec![S(0, 2, 0.4), S(0, 2, 0.9)];
        let kept = resolve_overlaps(&items, OverlapPolicy::Allow);
        assert_eq!(kept, vec![S(0, 2, 0.9)]);
    }
}
