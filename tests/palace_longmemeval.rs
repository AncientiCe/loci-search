//! Replay the archived LongMemEval-S exchange run with stored cosines.
//! Skips when the dump is not on this machine.

use loci_search::palace::{
    MiniLm, Passage, Weights, build_corpus, distinct_sources, parse_dataset_timestamp,
    parse_timestamp, pinned_minilm, score, sessions_match,
};
use serde::Deserialize;
use serde::de::{SeqAccess, Visitor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const QUESTIONS: &str = r"D:\Dev\Projects\mempalace-rs\data\longmemeval\longmemeval_s_cleaned.json";
const CHUNKS: &str = r"D:\Dev\Projects\mempalace-rs\data\out\longmemeval\recall_all_research\minilm-l6-v2-q8-exchange\longmemeval_cases.jsonl";
const ORDER: &str = r"D:\Dev\Projects\mempalace-rs\data\out\longmemeval\source_decay\s\minilm-l6-v2-q8-exchange\longmemeval_cases.jsonl";

#[derive(Deserialize)]
struct Question {
    question_id: String,
    question_type: String,
    question: String,
    question_date: String,
    #[serde(default)]
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_dates: Vec<String>,
}

#[derive(Deserialize)]
struct DumpCase {
    question_id: String,
    #[serde(default)]
    answer_session_ids: Vec<String>,
    #[serde(default)]
    ranked_session_ids: Vec<String>,
    #[serde(default)]
    chunks: Vec<DumpChunk>,
}

#[derive(Deserialize)]
struct DumpChunk {
    session_id: String,
    text: String,
    cosine: f64,
    bm25: f64,
}

#[test]
fn archived_exchange_dump_matches_the_palace_gate() {
    if !Path::new(CHUNKS).exists() || !Path::new(QUESTIONS).exists() || !Path::new(ORDER).exists() {
        eprintln!("LongMemEval exchange dump is absent; skipping the Palace gate");
        return;
    }
    let questions = load_questions(Path::new(QUESTIONS));
    let expected_order = load_order(Path::new(ORDER));
    let file = File::open(CHUNKS).unwrap();
    let mut any_at_5 = 0usize;
    let mut all_at_5 = 0usize;
    let mut neutral_any = 0usize;
    let mut neutral_all = 0usize;
    let mut questions_scored = 0usize;
    let mut bm25_error = 0.0f64;
    let mut mismatches = Vec::new();

    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let case: DumpCase = serde_json::from_str(&line.unwrap()).unwrap();
        let question = questions.get(&case.question_id).unwrap_or_else(|| {
            panic!("question {} missing from the cleaned set", case.question_id)
        });
        let expected = expected_order.get(&case.question_id).unwrap_or_else(|| {
            panic!(
                "question {} missing from the source-decay order",
                case.question_id
            )
        });
        let now = parse_dataset_timestamp(&question.question_date)
            .unwrap_or_else(|| panic!("unparsed question date {}", question.question_date));
        let filed_at = now.to_string();
        assert_eq!(parse_timestamp(&filed_at), Some(now));
        let dates: HashMap<&str, String> = question
            .haystack_session_ids
            .iter()
            .zip(&question.haystack_dates)
            .filter_map(|(id, date)| {
                parse_dataset_timestamp(date).map(|stamp| (id.as_str(), stamp.to_string()))
            })
            .collect();
        let (corpus, lengths) = build_corpus(
            &case
                .chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<Vec<_>>(),
        );
        let make = |with_dates: bool| {
            case.chunks
                .iter()
                .zip(&lengths)
                .map(|(chunk, len)| {
                    let created = if with_dates {
                        dates.get(chunk.session_id.as_str()).map(String::as_str)
                    } else {
                        None
                    };
                    OwnedPassage {
                        id: chunk.session_id.clone(),
                        source: chunk.session_id.clone(),
                        text: chunk.text.clone(),
                        room: question.question_type.clone(),
                        created_at: created.unwrap_or("").to_string(),
                        filed_at: if with_dates {
                            filed_at.clone()
                        } else {
                            String::new()
                        },
                        cosine: chunk.cosine,
                        doc_len: *len,
                        bm25: chunk.bm25,
                    }
                })
                .collect::<Vec<_>>()
        };
        let owned = make(true);
        let ranked = score_owned(&question.question, &owned, &corpus, now);
        for (row, passage) in ranked.iter().zip(&owned) {
            // `score` reorders. Compare by index.
            let _ = (row, passage);
        }
        for row in &ranked {
            let error = (row.bm25 - owned[row.index].bm25).abs();
            bm25_error = bm25_error.max(error);
        }
        let sessions = distinct_sources_owned(&ranked, &owned, 10);
        let actual: Vec<(String, f64)> = sessions
            .iter()
            .map(|(source, score)| ((*source).to_string(), *score))
            .collect();
        if !sessions_match(&actual, expected) && mismatches.len() < 8 {
            mismatches.push(format!(
                "{} expected {expected:?} got {:?}",
                case.question_id,
                actual.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>()
            ));
        }
        if !case.answer_session_ids.is_empty() {
            questions_scored += 1;
            let top: Vec<&str> = actual.iter().take(5).map(|(id, _)| id.as_str()).collect();
            if case
                .answer_session_ids
                .iter()
                .any(|id| top.contains(&id.as_str()))
            {
                any_at_5 += 1;
            }
            if case
                .answer_session_ids
                .iter()
                .all(|id| top.contains(&id.as_str()))
            {
                all_at_5 += 1;
            }
            let neutral_owned = make(false);
            let neutral = score_owned(&question.question, &neutral_owned, &corpus, now);
            let neutral_sessions = distinct_sources_owned(&neutral, &neutral_owned, 5);
            let neutral_top: Vec<&str> = neutral_sessions.iter().map(|(id, _)| *id).collect();
            if case
                .answer_session_ids
                .iter()
                .any(|id| neutral_top.contains(&id.as_str()))
            {
                neutral_any += 1;
            }
            if case
                .answer_session_ids
                .iter()
                .all(|id| neutral_top.contains(&id.as_str()))
            {
                neutral_all += 1;
            }
        }
        if line_number == 0 && bm25_error > 1e-4 {
            panic!(
                "first question bm25 diverges by {bm25_error} (got {} dump {})",
                ranked[0].bm25, owned[ranked[0].index].bm25
            );
        }
    }

    eprintln!(
        "palace dump replay any-support@5 {any_at_5}/{questions_scored} all-support@5 {all_at_5}/{questions_scored} bm25 max abs {bm25_error} neutral {neutral_any}/{neutral_all} order mismatches {}",
        mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "session order differs outside ties: {mismatches:?}"
    );
    assert!(
        bm25_error < 1e-6,
        "palace BM25 diverges from the dump by {bm25_error}"
    );
    assert_eq!(questions_scored, 470);
    assert_eq!(any_at_5, 464, "any-support@5");
    assert_eq!(all_at_5, 423, "all-support@5");
}

struct OwnedPassage {
    id: String,
    source: String,
    text: String,
    room: String,
    created_at: String,
    filed_at: String,
    cosine: f64,
    doc_len: usize,
    bm25: f64,
}

fn score_owned(
    query: &str,
    owned: &[OwnedPassage],
    corpus: &loci_search::palace::Corpus,
    now: jiff::Timestamp,
) -> Vec<loci_search::palace::Scored> {
    let views: Vec<Passage<'_>> = owned
        .iter()
        .map(|passage| Passage {
            id: &passage.id,
            source: &passage.source,
            text: &passage.text,
            room: &passage.room,
            created_at: &passage.created_at,
            filed_at: &passage.filed_at,
            cosine: passage.cosine,
            doc_len: passage.doc_len,
        })
        .collect();
    score(
        query,
        &views,
        corpus,
        now,
        &Weights::default(),
        &std::collections::HashSet::new(),
    )
}

fn distinct_sources_owned<'a>(
    ranked: &[loci_search::palace::Scored],
    owned: &'a [OwnedPassage],
    limit: usize,
) -> Vec<(&'a str, f64)> {
    let views: Vec<Passage<'_>> = owned
        .iter()
        .map(|passage| Passage {
            id: &passage.id,
            source: &passage.source,
            text: &passage.text,
            room: &passage.room,
            created_at: &passage.created_at,
            filed_at: &passage.filed_at,
            cosine: passage.cosine,
            doc_len: passage.doc_len,
        })
        .collect();
    distinct_sources(ranked, &views, limit)
}

fn load_order(path: &Path) -> HashMap<String, Vec<String>> {
    let file = File::open(path).unwrap();
    let mut order = HashMap::new();
    for line in BufReader::new(file).lines() {
        let case: DumpCase = serde_json::from_str(&line.unwrap()).unwrap();
        order.insert(case.question_id, case.ranked_session_ids);
    }
    order
}

fn load_questions(path: &Path) -> HashMap<String, Question> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut deser = serde_json::Deserializer::from_reader(reader);
    let mut map = HashMap::new();
    serde::Deserializer::deserialize_seq(&mut deser, Collect(&mut map)).unwrap();
    map
}

struct Collect<'a>(&'a mut HashMap<String, Question>);

impl<'de> Visitor<'de> for Collect<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a LongMemEval question array")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while let Some(question) = seq.next_element::<Question>()? {
            self.0.insert(question.question_id.clone(), question);
        }
        Ok(())
    }
}

#[test]
fn reembedded_exchange_dump_matches_the_palace_gate() {
    if std::env::var("PALACE_REEMBED").ok().as_deref() != Some("1") {
        eprintln!("set PALACE_REEMBED=1 to re-embed all 470 haystacks");
        return;
    }
    if !Path::new(CHUNKS).exists() || !Path::new(QUESTIONS).exists() || !Path::new(ORDER).exists() {
        eprintln!("LongMemEval exchange dump is absent; skipping the re-embed gate");
        return;
    }
    let (model, vocab) = pinned_minilm();
    let encoder = MiniLm::open(&model, &vocab, 2).unwrap();
    let questions = load_questions(Path::new(QUESTIONS));
    let expected_order = load_order(Path::new(ORDER));
    let mut any_at_5 = 0usize;
    let mut all_at_5 = 0usize;
    let mut neutral_any = 0usize;
    let mut neutral_all = 0usize;
    let mut questions_scored = 0usize;
    let mut mismatches = Vec::new();
    for (line_number, line) in BufReader::new(File::open(CHUNKS).unwrap())
        .lines()
        .enumerate()
    {
        let mut case: DumpCase = serde_json::from_str(&line.unwrap()).unwrap();
        let question = &questions[&case.question_id];
        let mut texts = vec![question.question.clone()];
        texts.extend(case.chunks.iter().map(|chunk| chunk.text.clone()));
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vectors = encoder.embed(&refs).unwrap();
        let query_vector = &vectors[0];
        for (chunk, vector) in case.chunks.iter_mut().zip(vectors.iter().skip(1)) {
            let dot = query_vector
                .iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            chunk.cosine = (f64::from(dot) * 1000.0).round() / 1000.0;
        }
        let now = parse_dataset_timestamp(&question.question_date).unwrap();
        let filed_at = now.to_string();
        let dates: HashMap<&str, String> = question
            .haystack_session_ids
            .iter()
            .zip(&question.haystack_dates)
            .filter_map(|(id, date)| {
                parse_dataset_timestamp(date).map(|stamp| (id.as_str(), stamp.to_string()))
            })
            .collect();
        let (corpus, lengths) = build_corpus(
            &case
                .chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<Vec<_>>(),
        );
        let owned = |with_dates: bool| {
            case.chunks
                .iter()
                .zip(&lengths)
                .map(|(chunk, len)| OwnedPassage {
                    id: chunk.session_id.clone(),
                    source: chunk.session_id.clone(),
                    text: chunk.text.clone(),
                    room: question.question_type.clone(),
                    created_at: if with_dates {
                        dates
                            .get(chunk.session_id.as_str())
                            .cloned()
                            .unwrap_or_default()
                    } else {
                        String::new()
                    },
                    filed_at: if with_dates {
                        filed_at.clone()
                    } else {
                        String::new()
                    },
                    cosine: chunk.cosine,
                    doc_len: *len,
                    bm25: 0.0,
                })
                .collect::<Vec<_>>()
        };
        let dated = owned(true);
        let ranked = score_owned(&question.question, &dated, &corpus, now);
        let sessions = distinct_sources_owned(&ranked, &dated, 10);
        let actual: Vec<(String, f64)> = sessions
            .iter()
            .map(|(source, score)| ((*source).to_string(), *score))
            .collect();
        let expected = &expected_order[&case.question_id];
        if !case.answer_session_ids.is_empty() {
            questions_scored += 1;
            let top: Vec<&str> = actual.iter().take(5).map(|(id, _)| id.as_str()).collect();
            if case
                .answer_session_ids
                .iter()
                .any(|id| top.contains(&id.as_str()))
            {
                any_at_5 += 1;
            }
            if case
                .answer_session_ids
                .iter()
                .all(|id| top.contains(&id.as_str()))
            {
                all_at_5 += 1;
            }
            let neutral_owned = owned(false);
            let neutral = score_owned(&question.question, &neutral_owned, &corpus, now);
            let neutral_top: Vec<&str> = distinct_sources_owned(&neutral, &neutral_owned, 5)
                .iter()
                .map(|(id, _)| *id)
                .collect();
            if case
                .answer_session_ids
                .iter()
                .any(|id| neutral_top.contains(&id.as_str()))
            {
                neutral_any += 1;
            }
            if case
                .answer_session_ids
                .iter()
                .all(|id| neutral_top.contains(&id.as_str()))
            {
                neutral_all += 1;
            }
        }
        if !sessions_match(&actual, expected) && mismatches.len() < 8 {
            mismatches.push(case.question_id.clone());
        }
        if line_number.is_multiple_of(20) {
            eprintln!(
                "re-embed {line_number} any {any_at_5} all {all_at_5} mismatches {}",
                mismatches.len()
            );
        }
    }
    eprintln!(
        "re-embed gate any-support@5 {any_at_5}/{questions_scored} all-support@5 {all_at_5}/{questions_scored} neutral {neutral_any}/{neutral_all} order mismatches {}",
        mismatches.len()
    );
    assert_eq!(questions_scored, 470);
    assert_eq!(any_at_5, 464, "re-embedded any-support@5");
    assert_eq!(all_at_5, 423, "re-embedded all-support@5");
}
