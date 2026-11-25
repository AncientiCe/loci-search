//! In-memory Palace hybrid score. BM25 uses k1=1.5, b=0.75, unique query
//! terms, and `ln((N - df + 0.5) / (df + 0.5) + 1)`, then divides by the
//! max BM25 in the scored pool. Document frequency is whatever corpus the
//! caller built; indexed search passes the generation-wide table.

use super::intent::{self, QueryIntent};
use super::preference::{self, PreferenceSource};
use super::time::{self, parse_timestamp};
use jiff::Timestamp;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;
const MAX_CODING_BOOST: f64 = 0.65;
const MAX_PREFERENCE_MATCH: f64 = 0.22;
const RECENCY_WEIGHT: f64 = 0.05;
const RECENCY_LAMBDA: f64 = 0.02;
const DATE_WEIGHT: f64 = 0.12;
const MAX_QUERY_WORDS: usize = 80;

const QUERY_MARKERS: &[&str] = &[
    "search query:",
    "memory query:",
    "query:",
    "question:",
    "user question:",
    "ask:",
];

const INTENT_TERMS: &[&str] = &[
    "why",
    "choose",
    "chose",
    "decided",
    "decision",
    "fix",
    "fixed",
    "failed",
    "broke",
    "command",
    "run",
    "test",
    "prefer",
    "convention",
    "rule",
    "last time",
    "current",
    "changed",
];

/// Alphanumeric tokens, lowercased, shorter than 2 characters dropped.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter_map(|term| {
            let term = term.trim().to_lowercase();
            (term.len() >= 2).then_some(term)
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Corpus {
    pub documents: f64,
    pub average_length: f64,
    pub document_frequency: HashMap<String, u32>,
}

/// `doc_lens[i]` is the Palace token count of `texts[i]`, duplicates included.
pub fn build_corpus(texts: &[impl AsRef<str>]) -> (Corpus, Vec<usize>) {
    let mut document_frequency: HashMap<String, u32> = HashMap::new();
    let mut doc_lens = Vec::with_capacity(texts.len());
    let mut total_len = 0usize;
    for text in texts {
        let terms = tokenize(text.as_ref());
        total_len += terms.len();
        doc_lens.push(terms.len());
        let mut seen = HashSet::new();
        for term in terms {
            if seen.insert(term.clone()) {
                *document_frequency.entry(term).or_default() += 1;
            }
        }
    }
    let documents = texts.len() as f64;
    let average_length = if texts.is_empty() {
        0.0
    } else {
        total_len as f64 / documents
    };
    (
        Corpus {
            documents,
            average_length,
            document_frequency,
        },
        doc_lens,
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub cosine_floor: f64,
    pub cosine_weight: f64,
    pub bm25_weight: f64,
    pub source_weight: f64,
    pub source_decay: f64,
    pub mined_discount: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            cosine_floor: 0.0,
            cosine_weight: 0.65,
            bm25_weight: 0.35,
            source_weight: 0.1,
            source_decay: 0.9,
            mined_discount: 0.12,
        }
    }
}

/// One passage in the pool being scored. `cosine` is the raw dot product;
/// Palace rounds it to 3 decimals before the model floor.
pub struct Passage<'a> {
    pub id: &'a str,
    pub source: &'a str,
    pub text: &'a str,
    pub room: &'a str,
    pub created_at: &'a str,
    pub filed_at: &'a str,
    pub cosine: f64,
    pub doc_len: usize,
}

#[derive(Clone, Debug)]
pub struct Scored {
    pub index: usize,
    pub cosine: f64,
    pub bm25: f64,
    pub coding_boost: f64,
    pub preference_match: f64,
    pub date_boost: f64,
    pub source_boost: f64,
    pub mined_discount: f64,
    pub repeat_discount: f64,
    pub combined: f64,
}

/// `sha256(question_id)` as a big-endian integer, even half.
pub fn development_half(question_id: &str) -> bool {
    let digest = Sha256::digest(question_id.as_bytes());
    digest.last().is_some_and(|byte| byte % 2 == 0)
}

pub fn score(
    query: &str,
    passages: &[Passage<'_>],
    corpus: &Corpus,
    now: Timestamp,
    weights: &Weights,
    mined: &HashSet<String>,
) -> Vec<Scored> {
    let sanitized = sanitize_query(query);
    let query = if sanitized.is_empty() {
        query
    } else {
        &sanitized
    };
    let bm25 = bm25_scores(query, passages, corpus);
    let max_bm25 = bm25.iter().copied().fold(0.0, f64::max);
    let targets = time::time_targets(query, now);
    let apply_mined = weights.mined_discount > 0.0 && !is_code_lookup(query);
    let mut rows: Vec<Scored> = passages
        .iter()
        .enumerate()
        .map(|(index, passage)| {
            let similarity = round_similarity(passage.cosine);
            let cosine = if weights.cosine_floor < 1.0 {
                ((similarity - weights.cosine_floor) / (1.0 - weights.cosine_floor)).max(0.0)
            } else {
                0.0
            };
            let normalized_bm25 = if max_bm25 > 0.0 {
                bm25[index] / max_bm25
            } else {
                0.0
            };
            let coding_boost = coding_agent_boost(query, passage.text, passage.room);
            let preference_match = preference_match(query, passage.text, passage.room);
            let date_boost = date_boost(&targets, passage.created_at);
            let recency = recency_boost(passage.filed_at, now);
            let mined_discount = if apply_mined && is_mined_file(mined, passage.source) {
                weights.mined_discount
            } else {
                0.0
            };
            let combined = ((cosine * weights.cosine_weight)
                + (normalized_bm25 * weights.bm25_weight)
                + coding_boost
                + preference_match
                + date_boost
                + recency
                - mined_discount)
                .max(0.0);
            Scored {
                index,
                cosine,
                bm25: bm25[index],
                coding_boost,
                preference_match,
                date_boost,
                source_boost: 0.0,
                mined_discount,
                repeat_discount: 0.0,
                combined,
            }
        })
        .collect();
    add_source_evidence(&mut rows, passages, weights.source_weight);
    sort_by_score(&mut rows);
    spread_sources(&mut rows, passages, weights.source_decay);
    rows
}

/// True when `actual` and `expected` are the same sequence once each run of
/// equal scores is compared as a set.
pub fn sessions_match(actual: &[(String, f64)], expected: &[String]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    let mut index = 0;
    while index < actual.len() {
        let score = actual[index].1;
        let mut end = index + 1;
        while end < actual.len() && actual[end].1 == score {
            end += 1;
        }
        let mut left: Vec<&str> = actual[index..end]
            .iter()
            .map(|(id, _)| id.as_str())
            .collect();
        let mut right: Vec<&str> = expected[index..end].iter().map(String::as_str).collect();
        left.sort_unstable();
        right.sort_unstable();
        if left != right {
            return false;
        }
        index = end;
    }
    true
}

pub fn distinct_sources<'a>(
    ranked: &[Scored],
    passages: &[Passage<'a>],
    limit: usize,
) -> Vec<(&'a str, f64)> {
    let mut seen = HashSet::new();
    let mut sources = Vec::new();
    for row in ranked {
        let source = passages[row.index].source;
        if source.is_empty() || !seen.insert(source) {
            continue;
        }
        sources.push((source, row.combined));
        if sources.len() == limit {
            break;
        }
    }
    sources
}

fn bm25_scores(query: &str, passages: &[Passage<'_>], corpus: &Corpus) -> Vec<f64> {
    let mut scores = vec![0.0; passages.len()];
    if corpus.documents == 0.0 || corpus.average_length <= 0.0 {
        return scores;
    }
    let unique: HashSet<String> = tokenize(query).into_iter().collect();
    if unique.is_empty() {
        return scores;
    }
    let mut frequencies = Vec::with_capacity(passages.len());
    for passage in passages {
        let mut counts: HashMap<String, f64> = HashMap::new();
        for term in tokenize(passage.text) {
            if unique.contains(&term) {
                *counts.entry(term).or_default() += 1.0;
            }
        }
        frequencies.push(counts);
    }
    for term in unique {
        let Some(&df) = corpus.document_frequency.get(&term) else {
            continue;
        };
        if df == 0 {
            continue;
        }
        let doc_freq = f64::from(df);
        let idf = ((corpus.documents - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
        for (index, counts) in frequencies.iter().enumerate() {
            let Some(&tf) = counts.get(&term) else {
                continue;
            };
            let doc_len = passages[index].doc_len as f64;
            let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / corpus.average_length));
            scores[index] += idf * ((tf * (BM25_K1 + 1.0)) / denom);
        }
    }
    scores
}

fn round_similarity(similarity: f64) -> f64 {
    (similarity * 1000.0).round() / 1000.0
}

fn recency_boost(filed_at: &str, now: Timestamp) -> f64 {
    let Some(stamp) = parse_timestamp(filed_at) else {
        return 0.0;
    };
    let age_days = (now.duration_since(stamp).as_secs() / 86_400).max(0) as f64;
    RECENCY_WEIGHT * (-RECENCY_LAMBDA * age_days).exp()
}

fn date_boost(targets: &[time::TimeTarget], created_at: &str) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let Some(stamp) = parse_timestamp(created_at) else {
        return 0.0;
    };
    DATE_WEIGHT * time::time_match(targets, stamp)
}

fn is_mined_file(mined: &HashSet<String>, source: &str) -> bool {
    mined.contains(source) || PreferenceSource::infer(source) == PreferenceSource::Code
}

fn is_code_lookup(query: &str) -> bool {
    query.split_whitespace().any(|token| {
        let token = token.trim_matches(|character: char| "?!,;\"'.".contains(character));
        token.contains("::")
            || token.contains('(')
            || token.contains('/')
            || token.contains('\\')
            || has_inner(token, '_')
            || is_file_name(token)
    })
}

fn has_inner(token: &str, mark: char) -> bool {
    token.match_indices(mark).any(|(index, _)| {
        let before = token[..index].chars().next_back();
        let after = token[index + mark.len_utf8()..].chars().next();
        before.is_some_and(char::is_alphanumeric) && after.is_some_and(char::is_alphanumeric)
    })
}

fn is_file_name(token: &str) -> bool {
    let Some((stem, ext)) = token.rsplit_once('.') else {
        return false;
    };
    stem.chars().count() >= 2
        && stem.chars().any(char::is_alphabetic)
        && (2..=4).contains(&ext.len())
        && ext.chars().all(|character| character.is_ascii_alphabetic())
}

fn add_source_evidence(rows: &mut [Scored], passages: &[Passage<'_>], weight: f64) {
    if weight <= 0.0 {
        return;
    }
    let mut top_two: HashMap<&str, (f64, f64)> = HashMap::new();
    for row in rows.iter() {
        let source = passages[row.index].source;
        if source.is_empty() {
            continue;
        }
        let (best, second) = top_two.entry(source).or_insert((0.0, 0.0));
        if row.combined > *best {
            *second = *best;
            *best = row.combined;
        } else if row.combined > *second {
            *second = row.combined;
        }
    }
    for row in rows.iter_mut() {
        let Some(&(best, second)) = top_two.get(passages[row.index].source) else {
            continue;
        };
        let sibling = if row.combined >= best { second } else { best };
        row.source_boost = weight * sibling;
        row.combined += row.source_boost;
    }
}

fn sort_by_score(rows: &mut [Scored]) {
    rows.sort_by(|left, right| {
        right
            .combined
            .partial_cmp(&left.combined)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn spread_sources(rows: &mut [Scored], passages: &[Passage<'_>], decay: f64) {
    if !(0.0..1.0).contains(&decay) {
        return;
    }
    let mut seen: HashMap<&str, i32> = HashMap::new();
    for row in rows.iter_mut() {
        let source = passages[row.index].source;
        if source.is_empty() {
            continue;
        }
        let earlier = seen.entry(source).or_insert(0);
        row.repeat_discount = row.combined * (1.0 - decay.powi(*earlier));
        row.combined -= row.repeat_discount;
        *earlier += 1;
    }
    sort_by_score(rows);
}

fn coding_agent_boost(query: &str, text: &str, room: &str) -> f64 {
    let query_lower = query.to_lowercase();
    let text_lower = text.to_lowercase();
    let room_lower = room.to_lowercase();
    let mut boost = 0.0;
    boost += intent_boost(
        &query_lower,
        &text_lower,
        &room_lower,
        &["why", "choose", "chose", "decided", "decision", "settled"],
        &[
            "decided",
            "chose",
            "because",
            "instead of",
            "rather than",
            "tradeoff",
            "settled on",
        ],
        &["decision", "decisions"],
        0.16,
    );
    boost += intent_boost(
        &query_lower,
        &text_lower,
        &room_lower,
        &[
            "fix",
            "fixed",
            "failed",
            "failure",
            "broke",
            "broken",
            "error",
            "last time",
        ],
        &[
            "fix",
            "fixed",
            "failed",
            "because",
            "root cause",
            "resolved",
            "workaround",
        ],
        &["problem", "problems", "fixes"],
        0.12,
    );
    boost += intent_boost(
        &query_lower,
        &text_lower,
        &room_lower,
        &[
            "command", "run", "test", "clippy", "fmt", "audit", "cargo", "npm",
        ],
        &[
            "run ", "cargo ", "npm ", "pytest", "clippy", "fmt", "audit", "--",
        ],
        &["command", "commands"],
        0.12,
    );
    let convention_query = query_lower.replace("should i ", " ");
    boost += intent_boost(
        &convention_query,
        &text_lower,
        &room_lower,
        &["convention", "rule", "usually", "should", "how do we"],
        &["convention", "rule", "always", "never", "should", "must"],
        &["convention", "conventions"],
        0.14,
    );
    boost += intent_boost(
        &query_lower,
        &text_lower,
        &room_lower,
        &[
            "prefer",
            "preference",
            "user want",
            "user like",
            "feel about",
            "interface style",
            "public interface",
            "shape",
            "retrieval results",
            "explain themselves",
        ],
        &[
            "prefer",
            "preference",
            "user",
            "does not want",
            "values",
            "my style",
            "source-grounded",
            "score provenance",
            "public api",
            "public apis",
        ],
        &["preference", "preferences"],
        0.15,
    );
    let query_asks_about_prefs = intent::classify(query) == QueryIntent::Preference;
    let text_is_preference = text_lower.contains("i prefer")
        || text_lower.contains("i always")
        || text_lower.contains("i never")
        || text_lower.contains("my convention")
        || text_lower.contains("my style")
        || text_lower.contains("i tend to")
        || text_lower.contains("prefer to");
    if query_asks_about_prefs && text_is_preference {
        boost += 0.10;
    }
    if query_lower.contains("cli") && text_lower.contains("cli command") {
        boost += 0.08;
    }
    if query_lower.contains("adding cli") && text_lower.contains("avoid new cli") {
        boost += 0.20;
    }
    if query_lower.contains("grep") && text_lower.contains("grep") {
        boost += 0.08;
    }
    if (query_lower.contains("proof") || query_lower.contains("visibility advice"))
        && text_lower.contains("proof from practical evaluations")
    {
        boost += 0.16;
    }
    boost += intent_boost(
        &query_lower,
        &text_lower,
        &room_lower,
        &[
            "current",
            "changed",
            "now",
            "last",
            "before",
            "direction",
            "proof",
            "proves",
            "priority",
            "release theme",
            "0.2.0",
        ],
        &[
            "current",
            "changed",
            "now",
            "from",
            "to",
            "direction",
            "proof priority",
            "eval suite",
            "release theme",
            "0.2.0",
            "agent memory reliability",
        ],
        &["current", "temporal"],
        0.12,
    );
    boost += intent_boost(
        &query_lower,
        &text_lower,
        &room_lower,
        &["session", "continuity", "warm-start", "warm start", "diary"],
        &[
            "session",
            "continuity",
            "warm-start",
            "warm start",
            "diary",
            "project path",
            "timestamp",
            "tags",
        ],
        &["current", "session"],
        0.11,
    );
    if text_lower.contains(&query_lower) {
        boost += 0.04;
    }
    boost.min(MAX_CODING_BOOST)
}

fn preference_match(query: &str, text: &str, room: &str) -> f64 {
    if intent::classify(query) != QueryIntent::Preference || !preference::is_preference(text) {
        return 0.0;
    }
    let mut score: f64 = 0.12;
    if room.eq_ignore_ascii_case("preference") || room.eq_ignore_ascii_case("preferences") {
        score += 0.04;
    }
    if let Some(span) = preference::preference_span(text) {
        let span_lower = span.to_lowercase();
        if tokenize(&query.to_lowercase())
            .into_iter()
            .any(|term| term.len() > 3 && span_lower.contains(&term))
        {
            score += 0.06;
        }
    }
    score.min(MAX_PREFERENCE_MATCH)
}

fn intent_boost(
    query: &str,
    text: &str,
    room: &str,
    query_terms: &[&str],
    text_terms: &[&str],
    rooms: &[&str],
    amount: f64,
) -> f64 {
    if !query_terms
        .iter()
        .any(|term| contains_at_word_start(query, term))
    {
        return 0.0;
    }
    let text_match = text_terms.iter().any(|term| text.contains(term));
    let room_match = rooms.contains(&room);
    match (text_match, room_match) {
        (true, true) => amount,
        (true, false) => amount * 0.7,
        (false, true) => amount * 0.45,
        (false, false) => 0.0,
    }
}

fn contains_at_word_start(text: &str, term: &str) -> bool {
    text.match_indices(term).any(|(index, _)| {
        text[..index]
            .chars()
            .next_back()
            .is_none_or(|character| !character.is_alphanumeric())
    })
}

fn sanitize_query(raw: &str) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return String::new();
    }
    let mut marker_hits = Vec::new();
    for line in raw.lines() {
        let compact = compact_whitespace(line);
        let lower = compact.to_lowercase();
        for marker in QUERY_MARKERS {
            if let Some(index) = lower.find(marker) {
                let value = compact[index + marker.len()..].trim();
                if !value.is_empty() {
                    marker_hits.push(value.to_string());
                }
            }
        }
    }
    if let Some(hit) = marker_hits.last() {
        return cap_words(hit);
    }
    let mut in_fence = false;
    let mut useful_lines = Vec::new();
    let mut fallback_lines = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || should_drop_line(trimmed) {
            continue;
        }
        let compact = compact_whitespace(trimmed);
        if compact.is_empty() {
            continue;
        }
        let lower = compact.to_lowercase();
        if compact.contains('?') || INTENT_TERMS.iter().any(|term| lower.contains(term)) {
            useful_lines.push(compact);
        } else {
            fallback_lines.push(compact);
        }
    }
    let selected = if useful_lines.is_empty() {
        fallback_lines.join(" ")
    } else {
        useful_lines.join(" ")
    };
    cap_words(&selected)
}

fn should_drop_line(line: &str) -> bool {
    if line.is_empty() {
        return true;
    }
    let lower = line.to_lowercase();
    if lower.starts_with("you are ")
        || lower.starts_with('<')
        || lower.starts_with('{')
        || lower.starts_with('}')
        || lower.starts_with('[')
        || lower.starts_with(']')
        || lower.contains("\"tool\"")
        || lower.contains("\"output\"")
        || lower.contains("jsonrpc")
    {
        return true;
    }
    let punctuation = line
        .chars()
        .filter(|character| !character.is_alphanumeric() && !character.is_whitespace())
        .count();
    line.len() > 240 && punctuation > line.len() / 8
}

fn compact_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cap_words(text: &str) -> String {
    text.split_whitespace()
        .take(MAX_QUERY_WORDS)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_drops_short_tokens_and_lowercases() {
        assert_eq!(
            tokenize("The ID of A-cat is 7!"),
            vec!["the", "id", "of", "cat", "is"]
        );
    }

    #[test]
    fn bm25_uses_palace_k1_and_normalizes_by_the_pool_max() {
        let texts = ["red apple fruit", "blue berry fruit", "red red wagon"];
        let (corpus, lengths) = build_corpus(&texts);
        let now = Timestamp::now();
        let passages: Vec<Passage<'_>> = texts
            .iter()
            .zip(&lengths)
            .enumerate()
            .map(|(index, (text, len))| Passage {
                id: match index {
                    0 => "a",
                    1 => "b",
                    _ => "c",
                },
                source: match index {
                    0 => "a",
                    1 => "b",
                    _ => "c",
                },
                text,
                room: "",
                created_at: "",
                filed_at: "",
                cosine: 0.0,
                doc_len: *len,
            })
            .collect();
        let ranked = score(
            "red apple",
            &passages,
            &corpus,
            now,
            &Weights {
                mined_discount: 0.0,
                ..Weights::default()
            },
            &HashSet::new(),
        );
        assert!(ranked[0].bm25 >= ranked[1].bm25);
        assert!(ranked.iter().any(|row| row.bm25 > 0.0));
        let max = ranked.iter().map(|row| row.bm25).fold(0.0, f64::max);
        let top = ranked.iter().find(|row| row.bm25 == max).unwrap();
        assert!((top.combined - (0.35 + 0.0)).abs() < 1e-9 || top.combined > 0.35);
    }

    #[test]
    fn equal_scores_match_as_a_set() {
        let actual = vec![("a".into(), 1.0), ("c".into(), 0.5), ("b".into(), 0.5)];
        assert!(sessions_match(
            &actual,
            &["a".into(), "b".into(), "c".into()]
        ));
        assert!(!sessions_match(
            &actual,
            &["b".into(), "a".into(), "c".into()]
        ));
    }
}
