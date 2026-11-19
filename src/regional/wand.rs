use super::{LexicalRescue, Posting, RescueScores, TermPostings, rescue_impact};
use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

struct Cursor<'a> {
    remaining: &'a [Posting],
    idf: f64,
    upper_bound: f64,
}

impl Cursor<'_> {
    fn seek(&mut self, target: u32) {
        // Most skips are short, so avoid a binary search in the common case.
        for _ in 0..4 {
            if !self
                .remaining
                .first()
                .is_some_and(|posting| posting.id < target)
            {
                return;
            }
            self.remaining = &self.remaining[1..];
        }
        let skipped = self
            .remaining
            .partition_point(|posting| posting.id < target);
        self.remaining = &self.remaining[skipped..];
    }
}

struct Ranked {
    id: usize,
    score: f64,
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ranked {}

pub(super) fn rescue(
    terms: &[(&TermPostings, f64)],
    norms: &[f64],
    limit: usize,
    budget: usize,
) -> LexicalRescue {
    let mut cursors = terms
        .iter()
        .map(|&(postings, idf)| Cursor {
            remaining: &postings.documents,
            idf,
            upper_bound: (idf * postings.max_rescue_impact).next_up(),
        })
        .collect::<Vec<_>>();
    let mut ordered = (0..cursors.len()).collect::<Vec<_>>();
    let mut best = BinaryHeap::<Reverse<Ranked>>::new();
    let mut scores = vec![0.0; norms.len()];
    let mut count = 0;

    let budget_hit = loop {
        ordered.retain(|&i| !cursors[i].remaining.is_empty());
        ordered.sort_unstable_by_key(|&i| cursors[i].remaining[0].id);
        let threshold = if best.len() == limit {
            // Allow for accumulated rounding in a different summation order.
            best.peek().expect("full heap").0.score
                * (1.0 - 4.0 * f64::EPSILON * cursors.len() as f64)
        } else {
            0.0
        };
        let mut bound = 0.0;
        let pivot = ordered.iter().find_map(|&i| {
            // Round the bound outwards because the real score sums terms in query order.
            bound = (bound + cursors[i].upper_bound).next_up();
            (bound >= threshold).then_some(cursors[i].remaining[0].id)
        });
        let Some(pivot) = pivot else { break false };
        if cursors[ordered[0]].remaining[0].id != pivot {
            for &i in &ordered {
                if cursors[i].remaining[0].id >= pivot {
                    break;
                }
                cursors[i].seek(pivot);
            }
            continue;
        }
        if count == budget {
            break true;
        }
        let id = pivot as usize;
        let mut score = 0.0;
        for cursor in &mut cursors {
            if let Some(posting) = cursor
                .remaining
                .first()
                .filter(|posting| posting.id == pivot)
            {
                score += cursor.idf * rescue_impact(posting.frequency, norms[id]);
                cursor.remaining = &cursor.remaining[1..];
            }
        }
        count += 1;
        scores[id] = score;
        let ranked = Ranked { id, score };
        if best.len() < limit {
            best.push(Reverse(ranked));
        } else if ranked > best.peek().expect("full heap").0 {
            *best.peek_mut().expect("full heap") = Reverse(ranked);
        }
    };
    let ranked = best
        .into_sorted_vec()
        .into_iter()
        .map(|entry| entry.0.id)
        .collect();
    LexicalRescue {
        ranked,
        scores: Some(RescueScores { scores, count }),
        budget_hit,
    }
}
