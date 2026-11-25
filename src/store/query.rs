use super::{ACCESS, Document, Snapshot};
use anyhow::{Result, ensure};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Filter {
    #[serde(alias = "term")]
    Eq {
        field: String,
        value: Value,
    },
    Exists {
        field: String,
    },
    Prefix {
        field: String,
        value: String,
    },
    Range {
        field: String,
        gt: Option<Value>,
        gte: Option<Value>,
        lt: Option<Value>,
        lte: Option<Value>,
    },
    And {
        filters: Vec<Filter>,
    },
    Or {
        filters: Vec<Filter>,
    },
    Not {
        filter: Box<Filter>,
    },
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    #[default]
    Asc,
    Desc,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sort {
    pub field: String,
    #[serde(default, alias = "order")]
    pub direction: Direction,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    Count,
    Min,
    Max,
    Sum,
    Avg,
    DateHistogram,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    pub field: String,
    #[serde(alias = "op")]
    pub kind: MetricKind,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub width: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchRequest {
    pub text: String,
    pub filter: Option<Filter>,
    pub sort: Vec<Sort>,
    pub offset: usize,
    pub limit: usize,
    pub facets: Vec<String>,
    pub aggregations: BTreeMap<String, Metric>,
    pub highlight: bool,
    pub fuzzy: bool,
    pub synonyms: BTreeMap<String, Vec<String>>,
    pub boosts: BTreeMap<String, f64>,
    pub cursor: Option<String>,
    /// Stop scanning once this many authorized matches are in hand.
    pub match_limit: usize,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            filter: None,
            sort: Vec::new(),
            offset: 0,
            limit: 10,
            facets: Vec::new(),
            aggregations: BTreeMap::new(),
            highlight: false,
            fuzzy: false,
            synonyms: BTreeMap::new(),
            boosts: BTreeMap::new(),
            cursor: None,
            match_limit: 100_000,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Bucket {
    pub value: Value,
    pub count: usize,
}

/// UTF-8 byte offsets into the original, unescaped source text.
#[derive(Debug, Serialize, Deserialize)]
pub struct Highlight {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchPage {
    pub generation: u64,
    pub total: usize,
    pub hits: Vec<Document>,
    pub facets: BTreeMap<String, Vec<Bucket>>,
    pub aggregations: BTreeMap<String, Value>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub highlights: BTreeMap<String, Vec<Highlight>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// False when `match_limit` stopped the scan before every authorized match was read.
    #[serde(default = "exact_answer", skip_serializing_if = "is_exact")]
    pub exact: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_reason: Option<String>,
}

fn exact_answer() -> bool {
    true
}

fn is_exact(value: &bool) -> bool {
    *value
}

fn field<'a>(document: &'a Document, key: &str) -> Option<Cow<'a, Value>> {
    match key {
        "id" => Some(Cow::Owned(Value::String(document.id.clone()))),
        "revision" => Some(Cow::Owned(Value::String(document.revision.clone()))),
        "text" => Some(Cow::Owned(Value::String(document.text.clone()))),
        key => document
            .fields
            .get(key.strip_prefix("fields.").unwrap_or(key))
            .map(Cow::Borrowed),
    }
}

fn unsigned_float_cmp(integer: u64, float: f64) -> Ordering {
    if float < 0.0 {
        return Ordering::Greater;
    }
    if float >= 2.0f64.powi(64) {
        return Ordering::Less;
    }
    integer.cmp(&(float as u64)).then_with(|| {
        if float.fract() > 0.0 {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    })
}

fn numeric_cmp(a: &Number, b: &Number) -> Ordering {
    if let Some(x) = a.as_u64() {
        if let Some(y) = b.as_u64() {
            return x.cmp(&y);
        }
        if b.as_i64().is_some() {
            return Ordering::Greater;
        }
        return unsigned_float_cmp(x, b.as_f64().expect("JSON number"));
    }
    if let Some(x) = a.as_i64() {
        if let Some(y) = b.as_i64() {
            return x.cmp(&y);
        }
        if b.as_u64().is_some() {
            return Ordering::Less;
        }
        let y = b.as_f64().expect("JSON number");
        if y >= 0.0 {
            return Ordering::Less;
        }
        return unsigned_float_cmp(x.unsigned_abs(), -y).reverse();
    }
    if b.is_i64() || b.is_u64() {
        return numeric_cmp(b, a).reverse();
    }
    a.as_f64()
        .partial_cmp(&b.as_f64())
        .unwrap_or(Ordering::Equal)
}

fn compare(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => Some(numeric_cmp(a, b)),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Null, Value::Null) => Some(Ordering::Equal),
        _ => None,
    }
}

fn mapped_date(settings: &Value, key: &str) -> bool {
    let key = key.strip_prefix("fields.").unwrap_or(key);
    let Some(def) = settings.get("mappings").and_then(|m| m.get(key)) else {
        return false;
    };
    let kind = def
        .as_str()
        .or_else(|| def.get("type").and_then(Value::as_str));
    matches!(kind, Some("date" | "date[]"))
}

fn date_value(v: &Value) -> Option<jiff::Timestamp> {
    if let Some(s) = v.as_str() {
        return s.parse().ok();
    }
    jiff::Timestamp::from_second(v.as_i64()?).ok()
}

fn compare_typed(a: &Value, b: &Value, date: bool) -> Option<Ordering> {
    if date {
        return Some(date_value(a)?.cmp(&date_value(b)?));
    }
    compare(a, b)
}

fn any_value(value: &Value, predicate: impl Fn(&Value) -> bool) -> bool {
    match value {
        Value::Array(items) => items.iter().any(predicate),
        value => predicate(value),
    }
}

impl Filter {
    fn matches(&self, doc: &Document, settings: &Value) -> bool {
        match self {
            Self::Eq { field: key, value } => field(doc, key).is_some_and(|v| {
                any_value(&v, |x| {
                    x == value
                        || compare_typed(x, value, mapped_date(settings, key))
                            == Some(Ordering::Equal)
                })
            }),
            Self::Exists { field: key } => field(doc, key).is_some_and(|v| !v.is_null()),
            Self::Prefix { field: key, value } => field(doc, key).is_some_and(|v| {
                any_value(&v, |x| x.as_str().is_some_and(|s| s.starts_with(value)))
            }),
            Self::Range {
                field: key,
                gt,
                gte,
                lt,
                lte,
            } => field(doc, key).is_some_and(|v| {
                any_value(&v, |x| {
                    (x.is_number() || x.is_string())
                        && gt.as_ref().is_none_or(|b| {
                            compare_typed(x, b, mapped_date(settings, key))
                                == Some(Ordering::Greater)
                        })
                        && gte.as_ref().is_none_or(|b| {
                            compare_typed(x, b, mapped_date(settings, key))
                                .is_some_and(|c| c != Ordering::Less)
                        })
                        && lt.as_ref().is_none_or(|b| {
                            compare_typed(x, b, mapped_date(settings, key)) == Some(Ordering::Less)
                        })
                        && lte.as_ref().is_none_or(|b| {
                            compare_typed(x, b, mapped_date(settings, key))
                                .is_some_and(|c| c != Ordering::Greater)
                        })
                })
            }),
            Self::And { filters } => filters.iter().all(|f| f.matches(doc, settings)),
            Self::Or { filters } => filters.iter().any(|f| f.matches(doc, settings)),
            Self::Not { filter } => !filter.matches(doc, settings),
        }
    }
}

fn order_key(value: &Value) -> u8 {
    match value {
        Value::Bool(_) => 0,
        Value::Number(_) => 1,
        Value::String(_) => 2,
        Value::Array(_) => 3,
        Value::Object(_) => 4,
        Value::Null => 5,
    }
}

fn compare_field(left: &Document, right: &Document, sort: &Sort, settings: &Value) -> Ordering {
    let a = field(left, &sort.field);
    let b = field(right, &sort.field);
    match (
        a.as_deref().filter(|x| !x.is_null()),
        b.as_deref().filter(|x| !x.is_null()),
    ) {
        (Some(a), Some(b)) => {
            let order =
                compare_typed(a, b, mapped_date(settings, &sort.field)).unwrap_or_else(|| {
                    order_key(a)
                        .cmp(&order_key(b))
                        .then_with(|| a.to_string().cmp(&b.to_string()))
                });
            match sort.direction {
                Direction::Asc => order,
                Direction::Desc => order.reverse(),
            }
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn exact_identifier(doc: &Document, text: &str) -> bool {
    doc.id.eq_ignore_ascii_case(text)
        || doc
            .fields
            .get("sku")
            .and_then(Value::as_str)
            .is_some_and(|sku| sku.eq_ignore_ascii_case(text))
}

// Levenshtein with a threshold. Stop early once no edit path fits the budget.
fn edit_distance(query: &[char], word: &str, budget: usize) -> Option<usize> {
    let n = word.chars().count();
    if query.len().abs_diff(n) > budget {
        return None;
    }
    let mut previous: Vec<usize> = (0..=query.len()).collect();
    let mut current = vec![0; query.len() + 1];
    for (j, b) in word.chars().enumerate() {
        current[0] = j + 1;
        let mut minimum = current[0];
        for (i, a) in query.iter().enumerate() {
            current[i + 1] = (previous[i + 1] + 1)
                .min(current[i] + 1)
                .min(previous[i] + usize::from(*a != b));
            minimum = minimum.min(current[i + 1]);
        }
        if minimum > budget {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[query.len()] <= budget).then_some(previous[query.len()])
}

fn field_values(value: &Value) -> &[Value] {
    match value {
        Value::Array(values) => values,
        scalar => std::slice::from_ref(scalar),
    }
}

fn facets(matches: &[(Document, f64)], fields: &[String]) -> BTreeMap<String, Vec<Bucket>> {
    let mut result = BTreeMap::new();
    for key in fields {
        let mut counts = BTreeMap::<String, Bucket>::new();
        for (doc, _) in matches {
            if let Some(value) = field(doc, key) {
                let mut seen = BTreeSet::new();
                for item in field_values(&value)
                    .iter()
                    .filter(|x| !x.is_null() && !x.is_object() && !x.is_array())
                {
                    let identity = item.to_string();
                    if seen.insert(identity.clone()) {
                        counts
                            .entry(identity)
                            .or_insert_with(|| Bucket {
                                value: item.clone(),
                                count: 0,
                            })
                            .count += 1;
                    }
                }
            }
        }
        let mut buckets: Vec<_> = counts.into_values().collect();
        buckets.sort_by(|a, b| {
            b.count.cmp(&a.count).then_with(|| {
                compare(&a.value, &b.value)
                    .unwrap_or_else(|| order_key(&a.value).cmp(&order_key(&b.value)))
            })
        });
        result.insert(key.clone(), buckets);
    }
    result
}

fn metric(matches: &[(Document, f64)], spec: &Metric) -> Result<Value> {
    if matches!(spec.kind, MetricKind::DateHistogram) {
        let (Some(start), Some(end), Some(width)) = (spec.start, spec.end, spec.width) else {
            anyhow::bail!("date_histogram requires start, end, width in seconds");
        };
        ensure!(width > 0 && end > start, "invalid histogram range or width");
        let n = (i128::from(end) - i128::from(start) + i128::from(width) - 1) / i128::from(width);
        ensure!(n <= 10_000, "histogram exceeds 10000 buckets");
        let mut buckets = vec![0u64; n as usize];
        for (doc, _) in matches {
            let Some(v) = field(doc, &spec.field) else {
                continue;
            };
            let mut seen = BTreeSet::new();
            for value in field_values(&v) {
                let t = super::stream::timestamp(value)?;
                if t >= start && t < end {
                    seen.insert(((i128::from(t) - i128::from(start)) / i128::from(width)) as usize);
                }
            }
            for bucket in seen {
                buckets[bucket] += 1;
            }
        }
        return Ok(Value::Array(
            buckets
                .into_iter()
                .enumerate()
                .map(|(i, count)| {
                    let time = (i128::from(start) + i as i128 * i128::from(width)) as i64;
                    serde_json::json!([time, count])
                })
                .collect(),
        ));
    }
    let mut present = 0u64;
    let mut numbers = 0u64;
    let mut sum = 0.0;
    let mut integer_sum = Some(0i128);
    let mut extreme: Option<Value> = None;
    for (doc, _) in matches {
        let Some(value) = field(doc, &spec.field) else {
            continue;
        };
        for v in field_values(&value).iter().filter(|v| !v.is_null()) {
            present += 1;
            if let Some(x) = v.as_f64() {
                numbers += 1;
                match spec.kind {
                    MetricKind::Min | MetricKind::Max => {
                        let desired = if matches!(spec.kind, MetricKind::Min) {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        };
                        if extreme
                            .as_ref()
                            .is_none_or(|old| compare(v, old) == Some(desired))
                        {
                            extreme = Some(v.clone());
                        }
                    }
                    MetricKind::Sum | MetricKind::Avg => {
                        sum += x;
                        ensure!(sum.is_finite(), "aggregation overflow");
                        let integer = v
                            .as_i64()
                            .map(i128::from)
                            .or_else(|| v.as_u64().map(i128::from));
                        integer_sum = integer_sum.zip(integer).map(|(a, b)| a + b);
                    }
                    MetricKind::Count | MetricKind::DateHistogram => {}
                }
            }
        }
    }
    Ok(match spec.kind {
        MetricKind::Count => Value::from(present),
        MetricKind::Min | MetricKind::Max => extreme.unwrap_or(Value::Null),
        MetricKind::Sum if integer_sum.is_some() => serde_json::to_value(integer_sum)?,
        MetricKind::Sum => Value::from(sum),
        MetricKind::Avg if numbers > 0 => {
            Value::from(integer_sum.map_or(sum, |n| n as f64) / numbers as f64)
        }
        MetricKind::Avg => Value::Null,
        MetricKind::DateHistogram => unreachable!(),
    })
}

fn offsets(text: &str, terms: &BTreeSet<String>) -> Vec<Highlight> {
    let mut result = Vec::new();
    let mut start = None;
    for (end, ch) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        if ch.is_alphanumeric() {
            start.get_or_insert(end);
        } else if let Some(start) = start.take()
            && terms.contains(&text[start..end].to_lowercase())
        {
            result.push(Highlight { start, end });
        }
    }
    result
}

// Score within the authorized query scope. SQLite's shared FTS table supplies
// candidate IDs, but its corpus statistics also cover other indexes and readers.
// BM25 uses k1=1.2 and b=0 here; field weights are applied to term frequencies.
/// Terms from the best hit that can pull a second document into a long question.
/// Short queries stay exact: a SKU or an identifier must not grow a new candidate set.
fn evidence_terms(top: &Document, query: &BTreeSet<String>) -> Vec<String> {
    let mut terms: Vec<String> = crate::terms(&top.text)
        .filter(|term| term.chars().count() >= 5 && !query.contains(term))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    terms.sort_by(|left, right| right.len().cmp(&left.len()).then(left.cmp(right)));
    terms.truncate(4);
    terms
}

fn rank_candidates(
    matches: &mut [(Document, f64)],
    words: &BTreeSet<String>,
    boosts: &BTreeMap<String, f64>,
    population: usize,
) {
    let mut frequencies = Vec::with_capacity(matches.len());
    let mut documents = BTreeMap::<String, usize>::new();
    for (doc, _) in matches.iter() {
        let mut frequency = BTreeMap::<String, f64>::new();
        let mut add = |name: &str, text: &str, default: f64| {
            let weight = boosts.get(name).copied().unwrap_or(default);
            for term in crate::terms(text).filter(|t| words.contains(t)) {
                *frequency.entry(term).or_default() += weight;
            }
        };
        add("id", &doc.id, 4.0);
        add("text", &doc.text, 1.0);
        for (key, value) in &doc.fields {
            let default = match key.as_str() {
                "sku" => 4.0,
                "title" => 2.0,
                _ => 1.0,
            };
            for v in field_values(value).iter().filter_map(Value::as_str) {
                add(key, v, default);
            }
        }
        for term in frequency.keys() {
            *documents.entry(term.clone()).or_default() += 1;
        }
        frequencies.push(frequency);
    }
    for ((_, score), frequency) in matches.iter_mut().zip(frequencies) {
        for (term, weight) in frequency {
            let df = documents[&term] as f64;
            let idf = ((population as f64 - df + 0.5) / (df + 0.5)).ln_1p();
            *score += idf * (weight * 2.2) / (weight + 1.2);
        }
    }
}

impl Snapshot {
    pub fn eligible(
        &self,
        filter: Option<&Filter>,
        principals: &[String],
    ) -> Result<Vec<Document>> {
        let mut documents = Vec::new();
        let settings = self.settings()?;
        self.visit_documents(principals, |document| {
            if filter.is_none_or(|f| f.matches(&document, &settings)) {
                documents.push(document);
            }
            Ok(())
        })?;
        Ok(documents)
    }

    pub fn search(&self, request: &SearchRequest, principals: &[String]) -> Result<SearchPage> {
        ensure!(
            request.cursor.is_none(),
            "use Catalog::search to resume a cursor"
        );
        let text = request.text.trim();
        let mut words: BTreeSet<_> = crate::terms(text).collect();
        let question = words.len() >= 8;
        ensure!(words.len() <= 128, "at most 128 query terms");
        let settings = self.settings()?;
        let mut synonyms: BTreeMap<String, Vec<String>> = settings
            .get("synonyms")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?
            .unwrap_or_default();
        synonyms.extend(request.synonyms.clone());
        let mut boosts: BTreeMap<String, f64> = settings
            .get("boosts")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?
            .unwrap_or_default();
        boosts.extend(request.boosts.clone());
        ensure!(
            boosts
                .values()
                .all(|x| x.is_finite() && *x > 0.0 && *x <= 100.0),
            "invalid field boost"
        );
        for origin in words.clone() {
            if let Some(alternatives) = synonyms.get(&origin) {
                for alternative in alternatives {
                    words.extend(crate::terms(alternative));
                }
            }
        }
        if !words.is_empty()
            && (request.fuzzy || settings.get("fuzzy").and_then(Value::as_bool) == Some(true))
        {
            let mut vocabulary = BTreeSet::new();
            self.visit_documents(principals, |doc| {
                if request
                    .filter
                    .as_ref()
                    .is_none_or(|f| f.matches(&doc, &settings))
                {
                    vocabulary.extend(crate::terms(&doc.text));
                    for name in ["title", "sku", "name"] {
                        if let Some(s) = doc.fields.get(name).and_then(Value::as_str) {
                            vocabulary.extend(crate::terms(s));
                        }
                    }
                }
                Ok(())
            })?;
            for origin in words.clone() {
                let chars: Vec<_> = origin.chars().collect();
                if !(4..=64).contains(&chars.len()) {
                    continue;
                }
                let limit = if chars.len() >= 8 { 2 } else { 1 };
                let mut candidates = Vec::new();
                for v in &vocabulary {
                    if let Some(distance) = edit_distance(&chars, v, limit)
                        && distance > 0
                    {
                        candidates.push((distance, v));
                    }
                }
                candidates.sort_unstable();
                words.extend(candidates.into_iter().take(8).map(|(_, v)| v.clone()));
            }
        }
        ensure!(words.len() <= 512, "at most 512 expanded query terms");
        let mut matches: Vec<(Document, f64)> = Vec::new();
        if text.is_empty() {
            matches = self
                .eligible(request.filter.as_ref(), principals)?
                .into_iter()
                .map(|d| (d, 0.0))
                .collect();
        } else {
            // Quote individual tokens so user text is always literal FTS input.
            let expression = words
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(" OR ");
            let full_text = if expression.is_empty() {
                "SELECT rowid FROM loci_fts WHERE 0 AND ?3 IS NULL"
            } else {
                "SELECT rowid FROM loci_fts WHERE loci_fts MATCH ?3 AND idx=?1"
            };
            let sql = format!("WITH matched AS MATERIALIZED ({full_text})
                SELECT d.body FROM matched m JOIN loci_documents d ON d.rowid=m.rowid
                WHERE d.idx=?1{ACCESS}
                UNION ALL SELECT d.body FROM loci_documents d WHERE d.idx=?1{ACCESS}
                AND (d.id=?4 COLLATE NOCASE OR json_extract(d.body,'$.fields.sku')=?4 COLLATE NOCASE)
                AND NOT EXISTS(SELECT 1 FROM matched m WHERE m.rowid=d.rowid)");
            let mut stmt = self.db.prepare(&sql)?;
            let mut rows = stmt.query(params![
                self.index,
                serde_json::to_string(principals)?,
                expression,
                text
            ])?;
            while let Some(row) = rows.next()? {
                let body: String = row.get(0)?;
                let doc = serde_json::from_str(&body)?;
                if request
                    .filter
                    .as_ref()
                    .is_none_or(|f| f.matches(&doc, &settings))
                {
                    matches.push((doc, 0.0));
                }
            }
            if request.sort.is_empty() && matches.len() > 1 {
                let population = if let Some(filter) = &request.filter {
                    let mut count = 0;
                    self.visit_documents(principals, |doc| {
                        count += usize::from(filter.matches(&doc, &settings));
                        Ok(())
                    })?;
                    count
                } else {
                    let sql =
                        format!("SELECT count(*) FROM loci_documents d WHERE d.idx=?1{ACCESS}");
                    self.db.query_row(
                        &sql,
                        params![self.index, serde_json::to_string(principals)?],
                        |r| r.get(0),
                    )?
                };
                rank_candidates(&mut matches, &words, &boosts, population);
            }
            if question && request.sort.is_empty() && !matches.is_empty() {
                let top = matches
                    .iter()
                    .max_by(|left, right| {
                        left.1
                            .total_cmp(&right.1)
                            .then_with(|| right.0.id.cmp(&left.0.id))
                    })
                    .map(|(doc, _)| doc.clone());
                if let Some(top) = top {
                    let extra = evidence_terms(&top, &words);
                    if !extra.is_empty() {
                        let expression = extra
                            .iter()
                            .map(|term| format!("\"{term}\""))
                            .collect::<Vec<_>>()
                            .join(" OR ");
                        let sql = format!(
                            "SELECT d.body FROM loci_fts
                             JOIN loci_documents d ON d.rowid = loci_fts.rowid
                             WHERE loci_fts MATCH ?3 AND d.idx=?1{ACCESS}
                             LIMIT 24"
                        );
                        let mut seen: BTreeSet<_> =
                            matches.iter().map(|(doc, _)| doc.id.clone()).collect();
                        let mut stmt = self.db.prepare(&sql)?;
                        let mut rows = stmt.query(params![
                            self.index,
                            serde_json::to_string(principals)?,
                            expression
                        ])?;
                        while let Some(row) = rows.next()? {
                            let doc: Document = serde_json::from_str(&row.get::<_, String>(0)?)?;
                            if seen.insert(doc.id.clone())
                                && request
                                    .filter
                                    .as_ref()
                                    .is_none_or(|filter| filter.matches(&doc, &settings))
                            {
                                matches.push((doc, 0.0));
                            }
                        }
                        words.extend(extra);
                        let population = matches.len().max(1);
                        for (_, score) in &mut matches {
                            *score = 0.0;
                        }
                        rank_candidates(&mut matches, &words, &boosts, population);
                    }
                }
            }
        }
        matches.sort_by(|(a, sa), (b, sb)| {
            for sort in &request.sort {
                let order = compare_field(a, b, sort, &settings);
                if order != Ordering::Equal {
                    return order;
                }
            }
            if !text.is_empty() && request.sort.is_empty() {
                exact_identifier(b, text)
                    .cmp(&exact_identifier(a, text))
                    .then_with(|| sb.total_cmp(sa))
                    .then_with(|| a.id.cmp(&b.id))
            } else {
                a.id.cmp(&b.id)
            }
        });
        let cap = request.match_limit.clamp(1, 100_000);
        let (exact, limit_reason) = if matches.len() > cap {
            matches.truncate(cap);
            (false, Some("match_limit".to_string()))
        } else {
            (true, None)
        };
        let facets = facets(&matches, &request.facets);
        let aggregations = request
            .aggregations
            .iter()
            .map(|(key, spec)| Ok((key.clone(), metric(&matches, spec)?)))
            .collect::<Result<_>>()?;
        let total = matches.len();
        let hits: Vec<_> = matches
            .into_iter()
            .skip(request.offset)
            .take(request.limit)
            .map(|(d, _)| d)
            .collect();
        let highlights = if request.highlight {
            hits.iter()
                .map(|doc| (doc.id.clone(), offsets(&doc.text, &words)))
                .collect()
        } else {
            BTreeMap::new()
        };
        Ok(SearchPage {
            generation: self.generation,
            total,
            hits,
            facets,
            aggregations,
            highlights,
            next_cursor: None,
            exact,
            limit_reason,
        })
    }

    pub fn suggest(
        &self,
        prefix: &str,
        limit: usize,
        principals: &[String],
    ) -> Result<Vec<String>> {
        let prefix = prefix.trim().to_lowercase();
        if prefix.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let mut counts = BTreeMap::<String, usize>::new();
        self.visit_documents(principals, |doc| {
            let mut values: BTreeSet<_> = crate::terms(&doc.text).collect();
            values.insert(doc.id);
            for name in ["title", "sku"] {
                if let Some(s) = doc.fields.get(name).and_then(Value::as_str) {
                    values.insert(s.to_owned());
                }
            }
            for value in values
                .into_iter()
                .filter(|s| s.to_lowercase().starts_with(&prefix))
            {
                *counts.entry(value).or_default() += 1;
            }
            Ok(())
        })?;
        let mut ranked: Vec<_> = counts.into_iter().collect();
        ranked.sort_by(|(a, x), (b, y)| y.cmp(x).then_with(|| a.cmp(b)));
        Ok(ranked.into_iter().take(limit).map(|(s, _)| s).collect())
    }
}
