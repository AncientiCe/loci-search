//! Int8 copy of the passage vectors for the regional scan.
//!
//! Each vector keeps one scale (its largest absolute component / 127) and 8-bit
//! codes, cutting scan memory traffic by 4x against float32. The query stays in
//! float32, so the only approximation is the per-component rounding of passages.
//! Callers re-score the best approximate candidates with exact cosine.

pub(crate) struct Quantized {
    dimensions: usize,
    codes: Vec<i8>,
    scales: Vec<f32>,
}

impl Quantized {
    pub(crate) fn build(vectors: &[Vec<f32>]) -> Self {
        let dimensions = vectors.first().map_or(0, Vec::len);
        let mut codes = Vec::with_capacity(vectors.len() * dimensions);
        let mut scales = Vec::with_capacity(vectors.len());
        for vector in vectors {
            let largest = vector
                .iter()
                .fold(0.0_f32, |max, value| max.max(value.abs()));
            let scale = largest / 127.0;
            let inverse = if scale > 0.0 { scale.recip() } else { 0.0 };
            codes.extend(
                vector
                    .iter()
                    .map(|&value| (value * inverse).round().clamp(-127.0, 127.0) as i8),
            );
            scales.push(scale);
        }
        Self {
            dimensions,
            codes,
            scales,
        }
    }

    pub(crate) fn similarity(&self, query: &[f32], id: usize) -> f32 {
        let start = id * self.dimensions;
        let codes = &self.codes[start..start + self.dimensions];
        let (query, query_tail) = query.as_chunks::<16>();
        let (codes, codes_tail) = codes.as_chunks::<16>();
        let mut lanes = [0.0_f32; 16];
        for (q, c) in query.iter().zip(codes) {
            for lane in 0..lanes.len() {
                lanes[lane] += q[lane] * f32::from(c[lane]);
            }
        }
        let tail = query_tail
            .iter()
            .zip(codes_tail)
            .map(|(&q, &c)| q * f32::from(c))
            .sum::<f32>();
        self.scales[id] * (lanes.into_iter().sum::<f32>() + tail)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine;

    #[test]
    fn int8_similarity_tracks_exact_cosine() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let normalize = |v: Vec<f32>| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect::<Vec<_>>()
        };
        let vectors = (0..64)
            .map(|_| normalize((0..1027).map(|_| next()).collect()))
            .collect::<Vec<_>>();
        let query = normalize((0..1027).map(|_| next()).collect());
        let quantized = Quantized::build(&vectors);
        assert_eq!(quantized.bytes(), 64 * 1027 + 64 * 4);
        for (id, vector) in vectors.iter().enumerate() {
            let error = (quantized.similarity(&query, id) - cosine(&query, vector)).abs();
            assert!(error < 2e-3, "id {id}: error {error}");
        }
    }
}
