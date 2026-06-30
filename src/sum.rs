//! NumPy's pairwise summation, reproduced bit-for-bit.
//!
//! `silhouette_score` is `float(np.mean(silhouette_samples(...)))`, and
//! `np.mean` reduces with numpy's `pairwise_sum`: blocks of up to 128 elements
//! are summed with eight interleaved accumulators, larger spans recurse on a
//! half split rounded down to a multiple of eight. Matching this order keeps
//! the mean bit-identical to numpy rather than off by a few ULP.

const BLOCK: usize = 128;

pub fn pairwise_sum(data: &[f64]) -> f64 {
    sum_range(data, 0, data.len())
}

fn sum_range(data: &[f64], lo: usize, n: usize) -> f64 {
    if n < 8 {
        let mut s = 0.0;
        for &v in &data[lo..lo + n] {
            s += v;
        }
        s
    } else if n <= BLOCK {
        let mut acc = [
            data[lo],
            data[lo + 1],
            data[lo + 2],
            data[lo + 3],
            data[lo + 4],
            data[lo + 5],
            data[lo + 6],
            data[lo + 7],
        ];
        let mut i = 8;
        while i + 8 <= n {
            for j in 0..8 {
                acc[j] += data[lo + i + j];
            }
            i += 8;
        }
        let mut res =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        while i < n {
            res += data[lo + i];
            i += 1;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        sum_range(data, lo, half) + sum_range(data, lo + half, n - half)
    }
}

#[cfg(test)]
mod tests {
    use super::pairwise_sum;

    #[test]
    fn small_naive_sum() {
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn block_boundary() {
        let v: Vec<f64> = (1..=200).map(f64::from).collect();
        assert_eq!(pairwise_sum(&v), (200.0 * 201.0) / 2.0);
    }
}
