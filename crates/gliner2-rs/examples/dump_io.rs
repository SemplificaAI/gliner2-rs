//! Prints each fragment's input and output names.
use gliner2_rs::runtime::{Precision, build_session, resolve_fragment};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap();
    gliner2_rs::init("dump-io");
    let dir = Path::new(&dir);
    let p = Precision::autodetect(dir, "encoder");
    println!("precision {p:?}");
    for stem in [
        "encoder", "token_gather", "span_rep", "schema_gather",
        "count_pred_argmax", "count_lstm_fixed", "scorer", "classifier",
    ] {
        let Some(path) = resolve_fragment(dir, stem, p) else {
            println!("  {stem:20} MANCANTE");
            continue;
        };
        let s = build_session(&path, 1)?;
        let ins: Vec<String> = s.inputs().iter().map(|i| i.name().to_string()).collect();
        let outs: Vec<String> = s.outputs().iter().map(|o| o.name().to_string()).collect();
        println!("  {stem:20} in={ins:?}  out={outs:?}");
    }
    std::process::exit(0);
}
