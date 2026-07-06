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
            if !v.is_finite() {
                return Err(RsomicsError::InvalidInput(format!(
                    "Input X contains NaN or infinity: '{tok}' at row {}",
                    i + 1
                )));
            }
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

/// Encode labels to dense indices over their sorted-unique set, matching
/// scikit-learn's `LabelEncoder` on the tokens read as text: an all-integer
/// column groups and sorts by numeric value (an int-dtype `np.unique`, so
/// `-0`/`0`/`01`/`+1` collapse to the value they denote), any other column —
/// including float-formatted strings like `1.0` — groups and sorts
/// lexicographically over the raw tokens (a string-dtype `np.unique`).
fn encode_labels(tokens: &[&str]) -> Result<(Vec<u32>, usize)> {
    let numeric = tokens.iter().all(|s| s.parse::<i64>().is_ok());
    if numeric {
        let values: Vec<i64> = tokens.iter().map(|s| s.parse::<i64>().unwrap()).collect();
        let uniq: BTreeSet<i64> = values.iter().copied().collect();
        let index: std::collections::HashMap<i64, u32> = uniq
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
        let idx = values.iter().map(|v| index[v]).collect();
        Ok((idx, uniq.len()))
    } else {
        let uniq: BTreeSet<&str> = tokens.iter().copied().collect();
        let index: std::collections::HashMap<&str, u32> = uniq
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as u32))
            .collect();
        let idx = tokens.iter().map(|s| index[*s]).collect();
        Ok((idx, uniq.len()))
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

    #[test]
    fn int_labels_dedup_by_value_not_text() {
        // '-0' and '0' both denote 0; sklearn's integer LabelEncoder collapses
        // them to one cluster. Earlier this panicked keying the index by raw text.
        let d = parse_dataset("1.0 0\n2.0 -0\n8.0 2\n9.0 2\n", None).unwrap();
        assert_eq!(d.labels, [0, 0, 1, 1]);
        assert_eq!(d.n_clusters, 2);
    }

    #[test]
    fn int_labels_dedup_leading_zero_and_plus() {
        let d = parse_dataset("1.0 01\n2.0 1\n8.0 +2\n9.0 2\n", None).unwrap();
        assert_eq!(d.labels, [0, 0, 1, 1]);
        assert_eq!(d.n_clusters, 2);
    }

    #[test]
    fn nan_feature_fails_loud() {
        assert!(parse_dataset("1.0 0\nnan 0\n8.0 1\n9.0 1\n", None).is_err());
    }

    #[test]
    fn inf_feature_fails_loud() {
        assert!(parse_dataset("1.0 0\ninf 0\n8.0 1\n9.0 1\n", None).is_err());
        assert!(parse_dataset("1.0 0\n-Infinity 0\n8.0 1\n9.0 1\n", None).is_err());
    }
}
