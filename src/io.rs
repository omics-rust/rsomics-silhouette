//! Feature-matrix and label input.
//!
//! The feature matrix is a rectangular whitespace-separated TSV (`n` rows ×
//! `d` columns). Labels are either a separate one-per-line column (`--labels`)
//! or the last column of the matrix. Labels are encoded to dense `{0, …, k−1}`
//! indices over their sorted-unique set, matching scikit-learn's `LabelEncoder`
//! (`np.unique`: integer-looking labels sort numerically, otherwise lexically).

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use rsomics_common::{Result, RsomicsError};

/// A dense row-major `rows × cols` feature matrix.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

/// Parsed input: the feature matrix, per-sample cluster indices, and the
/// number of distinct clusters.
pub struct Dataset {
    pub x: Matrix,
    pub labels: Vec<u32>,
    pub n_clusters: usize,
}

fn read_to_string(path: Option<&Path>) -> Result<String> {
    let mut buf = String::new();
    match path {
        Some(p) if p.as_os_str() != "-" => {
            File::open(p)
                .map_err(RsomicsError::Io)?
                .read_to_string(&mut buf)
                .map_err(RsomicsError::Io)?;
        }
        _ => {
            std::io::stdin()
                .lock()
                .read_to_string(&mut buf)
                .map_err(RsomicsError::Io)?;
        }
    }
    Ok(buf)
}

/// Read the feature matrix from `matrix` (`-`/`None` = stdin) and the cluster
/// labels either from `labels_path` or the matrix's last column.
pub fn read_dataset(matrix: Option<&Path>, labels_path: Option<&Path>) -> Result<Dataset> {
    let mtext = read_to_string(matrix)?;
    let ltext = labels_path.map(|p| read_to_string(Some(p))).transpose()?;
    parse_dataset(&mtext, ltext.as_deref())
}

/// Parse a feature-matrix text block and optional separate label column.
pub fn parse_dataset(matrix: &str, labels: Option<&str>) -> Result<Dataset> {
    let rows: Vec<Vec<&str>> = matrix
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().collect())
        .collect();
    if rows.is_empty() {
        return Err(RsomicsError::InvalidInput("empty feature matrix".into()));
    }

    let label_tokens: Vec<&str> = match labels {
        Some(text) => text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.split_whitespace()
                    .next()
                    .expect("non-empty line has a token")
            })
            .collect(),
        None => rows
            .iter()
            .map(|r| *r.last().expect("non-empty row"))
            .collect(),
    };

    let n = rows.len();
    if label_tokens.len() != n {
        return Err(RsomicsError::InvalidInput(format!(
            "{} feature rows but {} labels",
            n,
            label_tokens.len()
        )));
    }

    let feat_cols = match labels {
        Some(_) => rows[0].len(),
        None => rows[0]
            .len()
            .checked_sub(1)
            .filter(|&c| c > 0)
            .ok_or_else(|| {
                RsomicsError::InvalidInput(
                    "matrix needs ≥1 feature column plus a trailing label column".into(),
                )
            })?,
    };

    let mut data = Vec::with_capacity(n * feat_cols);
    for (i, r) in rows.iter().enumerate() {
        let need = match labels {
            Some(_) => feat_cols,
            None => feat_cols + 1,
        };
        if r.len() != need {
            return Err(RsomicsError::InvalidInput(format!(
                "row {} has {} columns, expected {}",
                i + 1,
                r.len(),
                need
            )));
        }
        for tok in &r[..feat_cols] {
            let v: f64 = fast_float2::parse(tok)
                .map_err(|_| RsomicsError::InvalidInput(format!("'{tok}' is not a number")))?;
            data.push(v);
        }
    }

    let (labels, n_clusters) = encode_labels(&label_tokens)?;
    Ok(Dataset {
        x: Matrix {
            rows: n,
            cols: feat_cols,
            data,
        },
        labels,
        n_clusters,
    })
}

/// Encode labels to dense indices over their sorted-unique set, ordered as
/// scikit-learn's `LabelEncoder` orders the tokens read as text: an all-integer
/// column sorts numerically (an int-dtype `np.unique`), any other column —
/// including float-formatted strings like `1.0` — sorts lexicographically (a
/// string-dtype `np.unique`).
fn encode_labels(tokens: &[&str]) -> Result<(Vec<u32>, usize)> {
    let numeric = tokens.iter().all(|s| s.parse::<i64>().is_ok());
    let mut set: BTreeSet<Key> = BTreeSet::new();
    for s in tokens {
        set.insert(Key::new(s, numeric));
    }
    let names: Vec<&str> = set.iter().map(|k| k.raw).collect();
    let index: std::collections::HashMap<&str, u32> = names
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i as u32))
        .collect();
    let idx = tokens.iter().map(|s| index[s]).collect();
    Ok((idx, names.len()))
}

/// Sort key for one label set: integer-looking labels sort by numeric value,
/// any non-integer token forces lexicographic order over the whole set.
struct Key<'a> {
    sort_int: Option<i64>,
    raw: &'a str,
}

impl<'a> Key<'a> {
    fn new(s: &'a str, numeric: bool) -> Self {
        let sort_int = if numeric { s.parse::<i64>().ok() } else { None };
        Key { sort_int, raw: s }
    }
}

impl PartialEq for Key<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl Eq for Key<'_> {}
impl PartialOrd for Key<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.sort_int, other.sort_int) {
            (Some(a), Some(b)) => a.cmp(&b),
            _ => self.raw.cmp(other.raw),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_column_is_label() {
        let d = parse_dataset("1.0 2.0 0\n3.0 4.0 1\n5.0 6.0 0\n", None).unwrap();
        assert_eq!(d.x.rows, 3);
        assert_eq!(d.x.cols, 2);
        assert_eq!(d.labels, [0, 1, 0]);
        assert_eq!(d.n_clusters, 2);
        assert_eq!(d.x.row(1), [3.0, 4.0]);
    }

    #[test]
    fn separate_label_file() {
        let d = parse_dataset("1.0 2.0\n3.0 4.0\n", Some("a\nb\n")).unwrap();
        assert_eq!(d.x.cols, 2);
        assert_eq!(d.labels, [0, 1]);
    }

    #[test]
    fn integer_labels_sort_numerically() {
        let d = parse_dataset("0 10\n0 2\n0 3\n", None).unwrap();
        assert_eq!(d.labels, [2, 0, 1]);
    }

    #[test]
    fn label_count_mismatch_fails() {
        assert!(parse_dataset("1.0\n2.0\n", Some("a\n")).is_err());
    }

    #[test]
    fn ragged_matrix_fails() {
        assert!(parse_dataset("1.0 2.0 0\n3.0 1\n", None).is_err());
    }
}
