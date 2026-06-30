//! Silhouette coefficient, value-exact with scikit-learn's `silhouette_samples`
//! / `silhouette_score`.
//!
//! scikit-learn's `_silhouette_reduce` forms, per sample `i`, the per-cluster
//! sum of distances (`np.bincount(labels, D[i])`); the self distance, 0, lands
//! in `i`'s own cluster. The own-cluster sum becomes `a(i)` after dividing by
//! `freq − 1`; the other clusters' sums divided by their frequencies give the
//! mean distances whose minimum is `b(i)`. Then `s(i) = (b − a) / max(a, b)`,
//! with singleton clusters (and `a == b`) yielding 0 via the same `nan_to_num`.
//!
//! Distances are symmetric, so we visit each unordered pair once and scatter
//! `d(i, j)` into both samples' per-cluster sums — half the distance work. The
//! resulting per-cell sum is a reordering of scikit-learn's strict left-fold,
//! so the exact bits can shift by a few ULP, staying well inside the
//! value-exactness tolerance. `silhouette_score` is the numpy-pairwise mean.

use rayon::prelude::*;

use crate::distance::{Metric, Norms, row_upper};
use crate::io::{Dataset, Matrix};
use crate::sum::pairwise_sum;

/// Per-sample silhouette coefficients, in input order.
pub fn silhouette_samples(d: &Dataset, metric: Metric) -> Vec<f64> {
    check_n_labels(d.n_clusters, d.x.rows);

    let n = d.x.rows;
    let k = d.n_clusters;
    let freq = label_freqs(&d.labels, k);
    let cluster = cluster_distance_sums(&d.x, &d.labels, k, metric);

    (0..n)
        .map(|i| coefficient(&cluster[i * k..(i + 1) * k], &freq, d.labels[i] as usize))
        .collect()
}

/// Mean silhouette coefficient over all samples.
pub fn silhouette_score(d: &Dataset, metric: Metric) -> f64 {
    let s = silhouette_samples(d, metric);
    pairwise_sum(&s) / s.len() as f64
}

/// `cluster[i*k + c]` = sum of distances from sample `i` to every sample in
/// cluster `c`. Distances are symmetric, so each row `i` only computes its
/// distances to samples `j > i` and scatters `d(i, j)` into both
/// `cluster[i][label[j]]` and `cluster[j][label[i]]` — halving the distance
/// kernel. Rows split into contiguous bands across rayon workers, each with its
/// own `n × k` accumulator summed at the end. A single band reproduces
/// scikit-learn's left-fold (`np.bincount`) order exactly; more bands reorder
/// the per-cell sum within floating-point noise.
fn cluster_distance_sums(x: &Matrix, labels: &[u32], k: usize, metric: Metric) -> Vec<f64> {
    let n = x.rows;
    let norms = Norms::build(metric, x);
    let n_threads = rayon::current_num_threads().max(1);
    let band = n.div_ceil(n_threads).max(1);

    let partials: Vec<Vec<f64>> = (0..n)
        .step_by(band)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let end = (start + band).min(n);
            let mut acc = vec![0.0f64; n * k];
            let mut upper = vec![0.0f64; n];
            for i in start..end {
                let dst = &mut upper[..n - i - 1];
                row_upper(metric, x, &norms, i, dst);
                let li = labels[i] as usize;
                for (off, &dij) in dst.iter().enumerate() {
                    let j = i + 1 + off;
                    acc[i * k + labels[j] as usize] += dij;
                    acc[j * k + li] += dij;
                }
            }
            acc
        })
        .collect();

    let mut cluster = vec![0.0f64; n * k];
    for part in &partials {
        for (c, p) in cluster.iter_mut().zip(part) {
            *c += *p;
        }
    }
    cluster
}

/// One sample's silhouette from its per-cluster distance sums.
fn coefficient(cluster: &[f64], freq: &[f64], own: usize) -> f64 {
    let intra = cluster[own] / (freq[own] - 1.0);

    let mut inter = f64::INFINITY;
    for (c, (&sum, &f)) in cluster.iter().zip(freq).enumerate() {
        if c == own {
            continue;
        }
        let mean = sum / f;
        if mean < inter {
            inter = mean;
        }
    }

    nan_to_num((inter - intra) / intra.max(inter))
}

fn label_freqs(labels: &[u32], k: usize) -> Vec<f64> {
    let mut f = vec![0.0f64; k];
    for &l in labels {
        f[l as usize] += 1.0;
    }
    f
}

/// scikit-learn's `nan_to_num`: NaN → 0, +inf → f64::MAX, −inf → f64::MIN. Only
/// the NaN case arises here (singleton clusters), but the inf clamps match.
fn nan_to_num(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else if x == f64::INFINITY {
        f64::MAX
    } else if x == f64::NEG_INFINITY {
        f64::MIN
    } else {
        x
    }
}

fn check_n_labels(n_labels: usize, n_samples: usize) {
    assert!(
        1 < n_labels && n_labels < n_samples,
        "Number of labels is {n_labels}. Valid values are 2 to n_samples - 1 (inclusive)"
    );
}

/// Whether the label set satisfies scikit-learn's `2 ≤ n_labels ≤ n_samples − 1`.
pub fn valid_label_count(n_labels: usize, n_samples: usize) -> bool {
    1 < n_labels && n_labels < n_samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::parse_dataset;

    #[test]
    fn singleton_cluster_is_zero() {
        // cluster 2 has a single member -> its silhouette is 0
        let d = parse_dataset("0.0 0\n0.1 0\n0.2 0\n5.0 1\n5.1 1\n9.0 2\n", None).unwrap();
        let s = silhouette_samples(&d, Metric::Euclidean);
        assert_eq!(s[5], 0.0);
    }

    #[test]
    fn well_separated_near_one() {
        let d = parse_dataset(
            "0.0 0.0 0\n0.1 0.0 0\n0.0 0.1 0\n10.0 10.0 1\n10.1 10.0 1\n10.0 10.1 1\n",
            None,
        )
        .unwrap();
        let score = silhouette_score(&d, Metric::Euclidean);
        assert!(score > 0.9, "well-separated score {score} should be near 1");
    }
}
