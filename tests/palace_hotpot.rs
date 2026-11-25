//! Background MiniLM embed of the existing HotpotQA 1M passage file.
//! Set PALACE_HOTPOT=1. Leaves two logical processors free when workers is 12 on a 16-thread machine.

use loci_search::palace::embed_passage_jsonl;
use std::path::PathBuf;

#[test]
fn embed_hotpotqa_1m() {
    if std::env::var("PALACE_HOTPOT").ok().as_deref() != Some("1") {
        eprintln!("set PALACE_HOTPOT=1 to MiniLM-embed HotpotQA 1M");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let passages = root.join("runs/hotpotqa-bge/passages.jsonl");
    let output = root.join("runs/hotpotqa-minilm/vectors.f32");
    let written = embed_passage_jsonl(&passages, &output, 12).unwrap();
    assert_eq!(written, 1_000_000);
}
