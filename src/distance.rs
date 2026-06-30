//! Pairwise distance rows, matching scikit-learn's `pairwise_distances` to
//! within the BLAS-boundary tolerance per metric.
//!
//! `cityblock` and `sqeuclidean` follow scipy `cdist`'s direct per-pair fold
//! over features — a plain left-to-right accumulation that reproduces
//! bit-for-bit. `euclidean` is computed the same direct way
//! (`sqrt(Σ(xᵢₖ − xⱼₖ)²)`): scikit-learn instead expands it as
//! `sqrt(‖x‖² − 2x·y + ‖y‖²)` through a BLAS matrix product, and the two agree
//! to a few ULP (observed ≤ ~2e-15) — the direct fold is far cheaper than
//! re-implementing an OpenBLAS GEMM and stays well inside value-exactness.
//! `cosine` keeps scikit-learn's normalized-dot form (`1 − x̂·ŷ`), also a
//! BLAS-boundary match.

use crate::io::Matrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Cosine,
    Cityblock,
    Sqeuclidean,
}

/// Distances from sample `i` to the samples after it; `out[t]` is `d(i, i+1+t)`.
/// Only the upper triangle is needed because every metric here is symmetric.
pub fn row_upper(metric: Metric, x: &Matrix, norms: &Norms, i: usize, out: &mut [f64]) {
    match metric {
        Metric::Euclidean => {
            let xi = x.row(i);
            for (t, o) in out.iter_mut().enumerate() {
                *o = sq_dist(xi, x.row(i + 1 + t)).sqrt();
            }
        }
        Metric::Sqeuclidean => {
            let xi = x.row(i);
            for (t, o) in out.iter_mut().enumerate() {
                *o = sq_dist(xi, x.row(i + 1 + t));
            }
        }
        Metric::Cityblock => {
            let xi = x.row(i);
            for (t, o) in out.iter_mut().enumerate() {
                *o = l1_dist(xi, x.row(i + 1 + t));
            }
        }
        Metric::Cosine => {
            let xn = norms.normalized.as_ref().expect("cosine normalized matrix");
            let xi = xn.row(i);
            for (t, o) in out.iter_mut().enumerate() {
                *o = (1.0 - dot(xi, xn.row(i + 1 + t))).clamp(0.0, 2.0);
            }
        }
    }
}

/// The one precomputation cosine needs: the L2-normalized matrix, so each
/// pairwise distance is a plain dot product. Empty for the other metrics.
pub struct Norms {
    /// `normalize(X)` — each row divided by its L2 norm, only for cosine.
    pub normalized: Option<Matrix>,
}

impl Norms {
    pub fn build(metric: Metric, x: &Matrix) -> Self {
        match metric {
            Metric::Cosine => Norms {
                normalized: Some(normalize(x)),
            },
            _ => Norms { normalized: None },
        }
    }
}

/// `Σ_k (aₖ − bₖ)²` — scipy `cdist`'s squared-euclidean kernel, a left-fold.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for k in 0..a.len() {
        let diff = a[k] - b[k];
        s += diff * diff;
    }
    s
}

/// `Σ_k |aₖ − bₖ|` — scipy `cdist`'s cityblock kernel, a left-fold.
#[inline]
fn l1_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for k in 0..a.len() {
        s += (a[k] - b[k]).abs();
    }
    s
}

/// L2-normalize each row, dividing by its own euclidean norm; a zero row stays
/// zero — matching scikit-learn's `normalize` (norm 0 left untouched).
fn normalize(x: &Matrix) -> Matrix {
    let mut data = x.data.clone();
    for i in 0..x.rows {
        let r = &mut data[i * x.cols..(i + 1) * x.cols];
        let n = dot(r, r).sqrt();
        if n != 0.0 {
            for v in r.iter_mut() {
                *v /= n;
            }
        }
    }
    Matrix {
        rows: x.rows,
        cols: x.cols,
        data,
    }
}

/// Left-fold inner product over features.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for k in 0..a.len() {
        s += a[k] * b[k];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        Matrix { rows, cols, data }
    }

    fn upper(metric: Metric, x: &Matrix, i: usize) -> Vec<f64> {
        let n = Norms::build(metric, x);
        let mut out = vec![0.0; x.rows - i - 1];
        row_upper(metric, x, &n, i, &mut out);
        out
    }

    #[test]
    fn euclidean_matches_hand() {
        let x = m(2, 2, vec![0.0, 0.0, 3.0, 4.0]);
        assert_eq!(upper(Metric::Euclidean, &x, 0)[0], 5.0);
    }

    #[test]
    fn cityblock_matches_hand() {
        let x = m(2, 3, vec![1.0, 2.0, 3.0, 4.0, 0.0, 1.0]);
        assert_eq!(upper(Metric::Cityblock, &x, 0)[0], 3.0 + 2.0 + 2.0);
    }

    #[test]
    fn sqeuclidean_matches_hand() {
        let x = m(2, 2, vec![0.0, 0.0, 3.0, 4.0]);
        assert_eq!(upper(Metric::Sqeuclidean, &x, 0)[0], 25.0);
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let x = m(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        assert!((upper(Metric::Cosine, &x, 0)[0] - 1.0).abs() < 1e-15);
    }
}
