#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryIntent {
    Preference,
    Decision,
    HowTo,
    Definition,
    Temporal,
    Unknown,
}

pub(crate) fn classify(query: &str) -> QueryIntent {
    let q = query.to_lowercase();
    let q = q.split_whitespace().collect::<Vec<_>>().join(" ");

    if contains_any(
        &q,
        &[
            "what do i prefer",
            "what do i like",
            "what is my favorite",
            "my go-to",
            "my preferred",
            "my preference",
            "do i prefer",
            "how do i like",
            "what should i use",
        ],
    ) {
        return QueryIntent::Preference;
    }

    if contains_any(
        &q,
        &[
            "what changed",
            "last session",
            "last time",
            "recent",
            "currently",
            "now",
            "timeline",
            "as of",
            "before",
            "after",
        ],
    ) {
        return QueryIntent::Temporal;
    }

    if contains_any(
        &q,
        &[
            "why did we",
            "why was",
            "why were",
            "why choose",
            "why chose",
            "decision",
            "decided",
            "tradeoff",
            "settled on",
        ],
    ) {
        return QueryIntent::Decision;
    }

    if q.starts_with("how do ")
        || q.starts_with("how should ")
        || q.starts_with("how can ")
        || contains_any(&q, &["command", "run ", "steps", "setup", "install"])
    {
        return QueryIntent::HowTo;
    }

    if q.starts_with("what is ")
        || q.starts_with("what are ")
        || q.starts_with("define ")
        || q.starts_with("where is ")
    {
        return QueryIntent::Definition;
    }

    QueryIntent::Unknown
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}
