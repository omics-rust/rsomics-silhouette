//! Silhouette clustering-quality coefficient — a value-exact, faster port of
//! scikit-learn's `sklearn.metrics.silhouette_score` and `silhouette_samples`.
//!
//! Given a feature matrix and a cluster labeling, the silhouette of a sample is
//! `(b − a) / max(a, b)` where `a` is the mean distance to the rest of its own
//! cluster and `b` the smallest mean distance to any other cluster. The score
//! is the mean over all samples. Distances follow scikit-learn's per-metric
//! float arithmetic (`euclidean`, `cosine`, `cityblock`/`manhattan`,
//! `sqeuclidean`); the per-cluster accumulation and the score mean reproduce
//! numpy's reduction order.

mod distance;
mod io;
mod silhouette;
mod sum;

pub use distance::Metric;
pub use io::{Dataset, Matrix, parse_dataset, read_dataset};
pub use silhouette::{silhouette_samples, silhouette_score, valid_label_count};
pub use sum::pairwise_sum;
