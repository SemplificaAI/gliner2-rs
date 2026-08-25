# gliner2-privacy

PII schemas and redaction for GLiNER2 privacy filtering. Thin layer over
[`gliner2-core`](../gliner2-core): the engine is there, this crate carries the
label vocabulary of
[`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
and the helpers a document pipeline needs.

Model: [`jugaadsrl/gliner2-privacy-filter-PII-multi-onnx`](https://huggingface.co/jugaadsrl/gliner2-privacy-filter-PII-multi-onnx)
— 42 PII types, 7 languages, legacy export layout.

## Extract and redact

```rust
use gliner2_core::{SpanConfig, SpanEngine};
use gliner2_privacy::{Group, redact};

gliner2_core::init("my-app");
let mut engine = SpanEngine::new(SpanConfig::new("models/pii-onnx"))?;

let text = "Contact Mario Rossi at m.rossi@example.com.";
let out = engine.extract(text, &[Group::Person.task(), Group::Contact.task()])?;

println!("{}", redact(text, &out.entities));
// Contact [PERSON] at [EMAIL].
```

## Pseudonymise instead of masking

`redact_with` takes the placeholder, so you can emit stable identifiers rather
than a bare label:

```rust
use std::collections::HashMap;
use std::cell::RefCell;

let counters: RefCell<HashMap<String, usize>> = RefCell::new(HashMap::new());
let pseudonymised = gliner2_privacy::redact_with(text, &out.entities, |label| {
    let mut c = counters.borrow_mut();
    let n = c.entry(label.to_string()).or_insert(0);
    *n += 1;
    format!("[[{}_{}]]", label.to_uppercase(), n)
});
```

Both rewrite from the end backwards, so earlier byte offsets stay valid, and
skip entities overlapping an already-rewritten stretch. That last part matters:
labels are decoded independently, so the same text can arrive under two labels,
and rewriting both would corrupt the output.

## Decide before you redact

```rust
if gliner2_privacy::needs_anonymization(&out.entities, 0.5) {
    // route to the pseudonymisation pipeline
}
```

Cheaper than redacting, and it answers the question a document pipeline actually
has.

## The 42 labels

Seven semantic groups, as the model card defines them. Each is a
[`Group`](src/lib.rs) with a `task()` constructor.

| group | labels |
|---|---|
| `Person` | `person`, `full_name`, `first_name`, `middle_name`, `last_name`, `date_of_birth` |
| `Contact` | `email`, `phone_number`, `address`, `street_address`, `city`, `state_or_region`, `postal_code`, `country` |
| `GovernmentId` | `government_id`, `national_id_number`, `passport_number`, `drivers_license_number`, `license_number`, `tax_id`, `tax_number` |
| `Banking` | `bank_account`, `account_number`, `routing_number`, `iban`, `payment_card`, `card_number`, `card_expiry`, `card_cvv` |
| `DigitalIdentity` | `username`, `ip_address`, `account_id`, `sensitive_account_id` |
| `Secrets` | `password`, `secret`, `api_key`, `access_token`, `recovery_code` |
| `SensitiveDates` | `sensitive_date`, `document_date`, `expiration_date`, `transaction_date` |

`all_labels_task()` covers every one of them, but a wide schema makes the labels
compete and precision on any one drops. Prefer the smallest group your policy
needs. The model conditions on whatever you pass, so any subset works — these
are simply the ones in distribution.

## Verified

13 cases across 7 languages against the PyTorch reference, using the export
published on the Hub rather than a locally tweaked one: **58/58 spans identical**
in both `fp16` and `fp16_iobinding`, max score delta 0.0023.

Apache-2.0. Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
