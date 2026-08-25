// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! PII schemas and redaction for GLiNER2 privacy filtering.
//!
//! Thin layer over [`gliner2_core`]: the engine is there, this crate carries the
//! label vocabulary of
//! [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
//! and the redaction helpers a document pipeline needs.
//!
//! ```no_run
//! use gliner2_privacy::{Group, redact};
//! use gliner2_core::{SpanConfig, SpanEngine};
//!
//! # fn main() -> anyhow::Result<()> {
//! gliner2_core::init("my-app");
//! let mut engine = SpanEngine::new(SpanConfig::new("models/pii-onnx"))?;
//!
//! let text = "Contact Mario Rossi at mario.rossi@example.com.";
//! let out = engine.extract(text, &[Group::Contact.task()])?;
//! println!("{}", redact(text, &out.entities));
//! # Ok(())
//! # }
//! ```

use gliner2_core::{Entity, SchemaTask};

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
/// Rewrites from the end backwards so earlier byte offsets stay valid, and skips
/// entities whose spans overlap an already-rewritten stretch — the engine can
/// return the same text under two labels, since labels are decoded
/// independently, and rewriting both would corrupt the output.
pub fn redact(text: &str, entities: &[Entity]) -> String {
    redact_with(text, entities, |label| format!("[{}]", label.to_uppercase()))
}

/// [`redact`] with a caller-supplied placeholder, for pseudonymisation schemes
/// that need stable identifiers rather than a bare label.
pub fn redact_with<F>(text: &str, entities: &[Entity], placeholder: F) -> String
where
    F: Fn(&str) -> String,
{
    let mut ordered: Vec<&Entity> = entities.iter().collect();
    // Descending by start, so each rewrite leaves the offsets before it intact.
    // Ties break on the longer span, which then swallows the shorter one.
    ordered.sort_by(|a, b| {
        b.char_start
            .cmp(&a.char_start)
            .then((b.char_end - b.char_start).cmp(&(a.char_end - a.char_start)))
    });

    let mut out = text.to_string();
    let mut last_start = usize::MAX;
    for e in ordered {
        if e.char_end > last_start {
            continue; // overlaps a stretch already rewritten
        }
        if e.char_start >= e.char_end || e.char_end > out.len() {
            continue;
        }
        out.replace_range(e.char_start..e.char_end, &placeholder(&e.label));
        last_start = e.char_start;
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
        let out = redact(
            text,
            &[entity("medical_professional", 4, 18), entity("person", 4, 18)],
        );
        assert_eq!(out, "Dr. [MEDICAL_PROFESSIONAL]");
    }

    #[test]
    fn gate_respects_the_threshold() {
        let mut low = entity("person", 0, 4);
        low.score = 0.3;
        assert!(!needs_anonymization(&[low], 0.5));
        assert!(needs_anonymization(&[entity("person", 0, 4)], 0.5));
    }
}
