# rsomics-silhouette

Silhouette clustering-quality coefficient — `silhouette_score` (the mean over
all samples) and `silhouette_samples` (the per-sample coefficient). A
value-exact, faster port of `sklearn.metrics.silhouette_score` /
`silhouette_samples`.

The silhouette is an *internal* validation metric: it scores a clustering from
the feature matrix itself, without a reference labeling. For each sample it
compares the mean distance to its own cluster (`a`) against the smallest mean
distance to any other cluster (`b`), giving `s = (b − a) / max(a, b)` in
`[−1, 1]`; the score is the mean. Useful for picking `k`, comparing embeddings,
or flagging samples assigned to the wrong cluster.

```sh
cargo install rsomics-silhouette
```

## Usage

Input is a feature matrix: `n` rows, `d` whitespace-separated numeric columns,
one sample per line (a file argument or `-` for stdin). Cluster labels are the
matrix's last column, or a separate one-per-line `--labels FILE`. Labels may be
integers or strings and are encoded as scikit-learn's `LabelEncoder` orders the
tokens read as text — an all-integer column numerically, any other column
(including float-formatted strings) lexicographically.

```sh
# labels in the matrix's last column
rsomics-silhouette features.tsv
rsomics-silhouette --metric cosine features.tsv
rsomics-silhouette --samples features.tsv          # per-sample s(i)

# labels in a separate file
rsomics-silhouette features.tsv --labels clusters.txt

rsomics-silhouette --metric manhattan features.tsv --json
```

`--metric` selects the distance, matching scikit-learn's `pairwise_distances`:

- `euclidean` (default)
- `cosine`
- `cityblock` / `manhattan`
- `sqeuclidean`

By default the mean `silhouette_score` is printed. `--samples` prints the
per-sample coefficient `s(i)`, one per line in input order. `--json` wraps the
result (scalar, or array with `--samples`) in the standard envelope.

The silhouette is only defined for `2 ≤ n_clusters ≤ n_samples − 1`; outside
that range the tool fails loud, exactly as scikit-learn raises.

## Accuracy

Verified against `scikit-learn` 1.9.0 over a differential sweep of six feature
matrices (n = 100..3000, d = 2..50, k = 2..8, well-separated and overlapping
blobs) across all four metrics, plus a separate string-label dataset and the
single-cluster / `k = n` degenerate cases — `silhouette_score` and every
`silhouette_samples` value:

- Run **single-threaded**, where the per-cluster accumulation visits samples in
  exactly scikit-learn's left-fold (`np.bincount`) order:
  - **cityblock / manhattan / sqeuclidean** are **bit-identical (0 ULP)** — the
    whole chain (scipy `cdist`'s direct per-pair fold, the per-cluster sums, and
    NumPy's pairwise mean) reproduces scikit-learn to the bit.
  - **euclidean / cosine** are within **~3e-15** absolute. scikit-learn computes
    these through the BLAS distance expansion (`sqrt(‖x‖² − 2x·y + ‖y‖²)` and
    `1 − x̂·ŷ`); matching an OpenBLAS matrix product to the last bit is not
    feasible, so these are at machine precision — far inside the 1e-12 bound.
  - The **silhouette_score** itself is bit-identical for cityblock / manhattan /
    sqeuclidean and within 1 ULP for euclidean / cosine.
- With more threads the accumulation reorders; values are no longer bit-exact
  but stay within **≤ 1e-12** of the goldens across every metric.

The compatibility test checks committed golden expectations (stored as IEEE-754
hex bits) and needs no Python at test time.

## Performance

`O(n² · d)` distance work, single-threaded, versus scikit-learn on the same
machine with `OPENBLAS/OMP/MKL_NUM_THREADS=1`. On a fair compute-only axis
(both sides with the data already in memory) ours is faster across every metric
because the distance matrix is symmetric: we compute each pair once and scatter
it into both samples' per-cluster sums, halving the kernel, whereas
scikit-learn forms the full chunked matrix. scikit-learn additionally pays
NumPy `bincount` per row and Python orchestration. The win grows further with
threads, and end-to-end the gap is larger still since scikit-learn pays Python
startup and import. See the crate's perf record for the measured numbers and
provenance.

## Origin

This crate is an independent Rust reimplementation of `scikit-learn` 1.9.0's
`sklearn.metrics.silhouette_score` and `silhouette_samples`
(`sklearn/metrics/cluster/_unsupervised.py`). The algorithm was read from and
follows that BSD-3-Clause source: the `_silhouette_reduce` per-cluster
distance accumulation (`np.bincount(labels, D[i])`), the own-cluster `a` over
`freq − 1` with `clip`-mode indexing, the `inf`-then-`min` for the
nearest-cluster `b`, the `(b − a) / max(a, b)` form, the `nan_to_num` of
singleton clusters, and `check_number_of_labels`. The per-metric distance float
arithmetic follows scikit-learn's `pairwise_distances` (`euclidean` /
`cosine` via the BLAS expansion, `cityblock` / `sqeuclidean` via scipy
`cdist`); euclidean is computed by the equivalent direct fold, which agrees
with scikit-learn's expansion to a few ULP. Golden expectations in
`tests/golden/` were generated once from scikit-learn and are checked into the
repo as hex bits.

License: MIT OR Apache-2.0.
Upstream credit: scikit-learn (https://scikit-learn.org, BSD-3-Clause).
