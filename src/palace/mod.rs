//! Palace MiniLM retrieval: tokenizer, BM25, fusion, and chunking.
//!
//! The score follows `hybrid_search_with` in palace-rs. It is separate from
//! Loci's coverage ranker and from the k1=1.2 rescue BM25 used to pick candidates.

mod chunk;
mod embed;
mod intent;
mod preference;
mod score;
mod time;

pub use embed::{DIMENSIONS, MiniLm, embed_passage_jsonl, pinned_minilm};

pub use chunk::{chunk_exchanges, chunk_text};
pub use score::{
    Corpus, Passage, Scored, Weights, build_corpus, development_half, distinct_sources, score,
    sessions_match,
};
pub use time::{parse_dataset_timestamp, parse_timestamp};
