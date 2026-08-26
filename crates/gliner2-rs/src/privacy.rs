// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! PII schemas and redaction for GLiNER2 privacy filtering.
//!
//! The label vocabulary of
//! [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
//! and the redaction helpers a document pipeline needs.
//!
//! ```no_run
//! use gliner2_rs::privacy::{Group, redact};
//! use gliner2_rs::{SpanConfig, SpanEngine};
//!
//! # fn main() -> anyhow::Result<()> {
//! gliner2_rs::init("my-app");
//! let mut engine = SpanEngine::new(SpanConfig::new("models/pii-onnx"))?;
//!
//! let text = "Contact Mario Rossi at mario.rossi@example.com.";
//! let out = engine.extract(text, &[Group::Contact.task()])?;
//! println!("{}", redact(text, &out.entities));
//! # Ok(())
//! # }
//! ```

use crate::{Entity, SchemaTask};

/// The 42 labels the model was trained on, in the seven groups its card uses.
///
/// The model conditions on whatever labels you pass at inference time, so any
/// subset works — but only these are in-distribution. Passing many at once has
/// a cost: labels in one schema interfere with each other, so prefer the
/// smallest group that covers your policy.
pub mod labels {
    pub const PERSON: &[&str] = &[
        "person", "full_name", "first_name", "middle_name", "last_name", "date_of_birth",
    ];
    pub const CONTACT: &[&str] = &[
        "email", "phone_number", "address", "street_address", "city", "state_or_region",
        "postal_code", "country",
    ];
    pub const GOVERNMENT_ID: &[&str] = &[
        "government_id", "national_id_number", "passport_number", "drivers_license_number",
        "license_number", "tax_id", "tax_number",
    ];
    pub const BANKING: &[&str] = &[
        "bank_account", "account_number", "routing_number", "iban", "payment_card",
        "card_number", "card_expiry", "card_cvv",
    ];
    pub const DIGITAL_IDENTITY: &[&str] =
        &["username", "ip_address", "account_id", "sensitive_account_id"];
    pub const SECRETS: &[&str] =
        &["password", "secret", "api_key", "access_token", "recovery_code"];
    pub const SENSITIVE_DATES: &[&str] = &[
        "sensitive_date", "document_date", "expiration_date", "transaction_date",
    ];

    /// All 42, in group order.
    pub fn all() -> Vec<&'static str> {
        [PERSON, CONTACT, GOVERNMENT_ID, BANKING, DIGITAL_IDENTITY, SECRETS, SENSITIVE_DATES]
            .concat()
    }
}

/// One of the model's seven semantic groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Person,
    Contact,
    GovernmentId,
    Banking,
    DigitalIdentity,
    Secrets,
    SensitiveDates,
}

impl Group {
    pub const ALL: [Group; 7] = [
        Group::Person,
        Group::Contact,
        Group::GovernmentId,
        Group::Banking,
        Group::DigitalIdentity,
        Group::Secrets,
        Group::SensitiveDates,
    ];

    pub fn labels(self) -> &'static [&'static str] {
        match self {
            Group::Person => labels::PERSON,
            Group::Contact => labels::CONTACT,
            Group::GovernmentId => labels::GOVERNMENT_ID,
            Group::Banking => labels::BANKING,
            Group::DigitalIdentity => labels::DIGITAL_IDENTITY,
            Group::Secrets => labels::SECRETS,
            Group::SensitiveDates => labels::SENSITIVE_DATES,
        }
    }

    /// An entity task covering this group.
    pub fn task(self) -> SchemaTask {
        SchemaTask::Entities(self.labels().iter().map(|s| s.to_string()).collect())
    }
}

/// An entity task covering every label the model knows.
///
/// Convenient, but not always the best answer: a wide schema makes the labels
/// compete, and precision on any one of them drops. Prefer [`Group::task`] when
/// your policy only needs part of the vocabulary.
pub fn all_labels_task() -> SchemaTask {
    SchemaTask::Entities(labels::all().into_iter().map(|s| s.to_string()).collect())
}

/// Replaces every detected span with `[LABEL]`.
///
/// See [`redact_with`] for the overlap rule and for custom placeholders.
pub fn redact(text: &str, entities: &[Entity]) -> String {
    redact_with(text, entities, |label| format!("[{}]", label.to_uppercase()))
}

/// [`redact`] with a caller-supplied placeholder, for pseudonymisation schemes
/// that need stable identifiers rather than a bare label.
///
/// ## How overlaps are resolved
///
/// Labels are decoded independently, so one stretch of text routinely arrives
/// under several of them. `Giuseppe Verdi` comes back as `full_name` at 99.8%,
/// as `person` at 91.6%, and separately as `first_name` + `last_name`. Only one
/// rewrite can happen, so the **highest-scoring** span wins and anything
/// overlapping it is dropped — here, `[FULL_NAME]` rather than
/// `[FIRST_NAME] [LAST_NAME]`.
///
/// Resolving by score rather than by position matters: doing it positionally
/// lets a short, later, lower-scoring span pre-empt the long high-scoring one
/// that contains it, so the output depends on where the entity sits in the
/// sentence. Either way nothing leaks, but only one of the two is predictable.
///
/// The rewrite itself then runs from the end backwards, so earlier byte offsets
/// stay valid.
pub fn redact_with<F>(text: &str, entities: &[Entity], placeholder: F) -> String
where
    F: Fn(&str) -> String,
{
    // 1. keep a non-overlapping set, best score first.
    //
    // The boundary guard matters: `replace_range` panics on a byte offset that
    // is not a UTF-8 character boundary, and entities do not have to come from
    // this engine — a caller merging offsets from elsewhere can hand in a
    // malformed one. Dropping it turns a would-be panic in the middle of a
    // redaction pipeline into a span that is simply not rewritten.
    let mut ranked: Vec<&Entity> = entities
        .iter()
        .filter(|e| {
            e.char_start < e.char_end
                && e.char_end <= text.len()
                && text.is_char_boundary(e.char_start)
                && text.is_char_boundary(e.char_end)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then((b.char_end - b.char_start).cmp(&(a.char_end - a.char_start)))
            .then(a.char_start.cmp(&b.char_start))
            .then(a.label.cmp(&b.label))
    });

    let mut kept: Vec<&Entity> = Vec::new();
    for cand in ranked {
        let overlaps = kept
            .iter()
            .any(|k| cand.char_start < k.char_end && k.char_start < cand.char_end);
        if !overlaps {
            kept.push(cand);
        }
    }

    // 2. rewrite from the end, so the offsets ahead of each edit stay valid
    kept.sort_by(|a, b| b.char_start.cmp(&a.char_start));
    let mut out = text.to_string();
    for e in kept {
        out.replace_range(e.char_start..e.char_end, &placeholder(&e.label));
    }
    out
}

/// Whether anything was detected at or above `threshold`.
///
/// The gate a document pipeline needs before deciding to pseudonymise: cheaper
/// than redacting, and it answers the question the caller actually has.
pub fn needs_anonymization(entities: &[Entity], threshold: f32) -> bool {
    entities.iter().any(|e| e.score >= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(label: &str, start: usize, end: usize) -> Entity {
        Entity {
            text: String::new(),
            label: label.into(),
            score: 0.99,
            char_start: start,
            char_end: end,
            word_start: 0,
            word_end: 0,
            slot: 0,
            task: "entities".into(),
        }
    }

    #[test]
    fn the_card_lists_forty_two_labels() {
        assert_eq!(labels::all().len(), 42);
        let total: usize = Group::ALL.iter().map(|g| g.labels().len()).sum();
        assert_eq!(total, 42);
    }

    #[test]
    fn redaction_rewrites_from_the_end() {
        let text = "Contact Mario Rossi at m.rossi@example.com.";
        let out = redact(
            text,
            &[entity("person", 8, 19), entity("email", 23, 42)],
        );
        assert_eq!(out, "Contact [PERSON] at [EMAIL].");
    }

    #[test]
    fn overlapping_labels_are_rewritten_once() {
        // labels are decoded independently, so the same span can arrive twice
        let text = "Dr. Francesca Neri";
        let mut a = entity("medical_professional", 4, 18);
        a.score = 0.95;
        let mut b = entity("person", 4, 18);
        b.score = 0.91;
        assert_eq!(redact(text, &[a, b]), "Dr. [MEDICAL_PROFESSIONAL]");
    }

    #[test]
    fn the_best_scoring_span_wins_regardless_of_position() {
        // "Giuseppe Verdi" as full_name contains "Giuseppe" and "Verdi".
        // Resolving positionally would let the later, shorter `last_name`
        // pre-empt the containing span; resolving by score does not.
        let text = "Il paziente Giuseppe Verdi.";
        let mut full = entity("full_name", 12, 26);
        full.score = 0.998;
        let mut first = entity("first_name", 12, 20);
        first.score = 0.99;
        let mut last = entity("last_name", 21, 26);
        last.score = 0.99;
        assert_eq!(
            redact(text, &[first, last, full]),
            "Il paziente [FULL_NAME]."
        );
    }

    #[test]
    fn malformed_offsets_do_not_panic() {
        // "è" is two bytes; offset 9 falls inside it. replace_range would
        // panic — the guard drops the span instead.
        let text = "Il caffè di Mario.";
        assert!(!text.is_char_boundary(8));
        let out = redact(text, &[entity("person", 7, 8)]);
        assert_eq!(out, text, "a malformed span is skipped, not applied");
        // a well-formed span beside it still works ("Mario" = bytes 13..18)
        let out = redact(text, &[entity("person", 13, 18)]);
        assert_eq!(out, "Il caffè di [PERSON].");
    }

    #[test]
    fn gate_respects_the_threshold() {
        let mut low = entity("person", 0, 4);
        low.score = 0.3;
        assert!(!needs_anonymization(&[low], 0.5));
        assert!(needs_anonymization(&[entity("person", 0, 4)], 0.5));
    }
}
