//! Exact top-k global lexical rescue with MaxScore dynamic pruning.
//!
//! Terms are ordered by their score upper bound. The low-bound prefix whose
//! summed bounds cannot reach the current top-k threshold is "non-essential":
//! those lists are never iterated, only probed with galloping seeks for
//! candidates produced by the essential lists. Final scores are summed in query
//! term order so they match the dense scorer bit for bit.
use super::{LexicalRescue, Posting, RescueScores, Scored, TermPostings, rescue_impact};
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
    fn head(&self) -> Option<u32> {
        self.remaining.first().map(|posting| posting.id)
    }

    /// Advance to the first posting with `id >= target` by galloping, so short
    /// skips stay cheap and long skips cost a logarithm of the distance.
    fn seek(&mut self, target: u32) {
        let postings = self.remaining;
        if postings.first().is_none_or(|posting| posting.id >= target) {
            return;
        }
        let mut high = 1;
        while high < postings.len() && postings[high].id < target {
            high *= 2;
        }
        let low = high / 2;
        let high = high.min(postings.len());
        let skipped = low + postings[low..high].partition_point(|posting| posting.id < target);
        self.remaining = &postings[skipped..];
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
    allow: Option<&[bool]>,
) -> LexicalRescue {
    let mut cursors = terms
        .iter()
        .map(|&(postings, idf)| Cursor {
            remaining: &postings.documents,
            idf,
            upper_bound: (idf * postings.max_rescue_impact).next_up(),
        })
        .collect::<Vec<_>>();
    // Cursor indices by ascending upper bound, with outward-rounded prefix sums.
    let mut by_bound = (0..cursors.len()).collect::<Vec<_>>();
    by_bound.sort_by(|&a, &b| cursors[a].upper_bound.total_cmp(&cursors[b].upper_bound));
    let mut prefix = vec![0.0; cursors.len() + 1];
    for (k, &i) in by_bound.iter().enumerate() {
        prefix[k + 1] = (prefix[k] + cursors[i].upper_bound).next_up();
    }
    let mut essential = 0;
    let mut contributions = vec![0.0; cursors.len()];
    let mut best = BinaryHeap::<Reverse<Ranked>>::new();
    let mut scored = Vec::new();
    let mut count = 0;

    let budget_hit = 'candidates: loop {
        let threshold = if best.len() == limit {
            // Allow for accumulated rounding in a different summation order.
            best.peek().expect("full heap").0.score
                * (1.0 - 4.0 * f64::EPSILON * cursors.len() as f64)
        } else {
            0.0
        };
        while essential < cursors.len() && prefix[essential + 1] < threshold {
            essential += 1;
        }
        let Some(candidate) = by_bound[essential..]
            .iter()
            .filter_map(|&i| cursors[i].head())
            .min()
        else {
            break false;
        };
        let id = candidate as usize;
        if allow.is_some_and(|allow| !allow[id]) {
            // Excluded passages consume no budget; step every essential list past them.
            for &i in &by_bound[essential..] {
                let cursor = &mut cursors[i];
                if cursor.head() == Some(candidate) {
                    cursor.remaining = &cursor.remaining[1..];
                }
            }
            continue;
        }
        if count == budget {
            break true;
        }
        contributions.fill(0.0);
        let mut partial = 0.0;
        for &i in &by_bound[essential..] {
            let cursor = &mut cursors[i];
            if cursor.head() == Some(candidate) {
                let value = cursor.idf * rescue_impact(cursor.remaining[0].frequency, norms[id]);
                contributions[i] = value;
                partial += value;
                cursor.remaining = &cursor.remaining[1..];
            }
        }
        for k in (0..essential).rev() {
            // `prefix[k + 1]` bounds every non-essential term not yet probed.
            if (partial + prefix[k + 1]).next_up() < threshold {
                continue 'candidates;
            }
            let cursor = &mut cursors[by_bound[k]];
            cursor.seek(candidate);
            if cursor.head() == Some(candidate) {
                let value = cursor.idf * rescue_impact(cursor.remaining[0].frequency, norms[id]);
                contributions[by_bound[k]] = value;
                partial += value;
                cursor.remaining = &cursor.remaining[1..];
            }
        }
        // Sum in query term order, as the dense scorer does.
        let score = contributions.iter().sum::<f64>();
        count += 1;
        // Candidates arrive in ascending id order, so `scored` stays sorted.
        scored.push(candidate);
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
        scores: Some(RescueScores {
            scored: Scored::Sorted(scored),
            count,
        }),
        budget_hit,
    }
}
