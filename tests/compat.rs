//! Compat against frozen scikit-learn 1.9.0 goldens — no scikit-learn at test
//! time. `tests/golden/expected_score.tsv` holds the `silhouette_score` each
//! (metric, dataset) produced; `expected_samples.tsv` points at a per-sample
//! hex file with every `silhouette_samples` value. All expected floats are
//! stored as little-endian IEEE-754 hex bits, so the comparison is exact.
//!
//! The golden comparison runs single-threaded (`-t1`), where the symmetric
//! per-cluster accumulation visits samples in exactly scikit-learn's left-fold
//! order. There `cityblock` / `manhattan` / `sqeuclidean` reproduce scipy
//! `cdist` bit-for-bit (0 ULP); `euclidean` / `cosine` differ by a few ULP
//! (≤ ~3e-15) from scikit-learn's BLAS distance expansion. With more threads
//! the accumulation reorders and all metrics land within ≤ 1e-12 of the
//! goldens (asserted separately).

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rsomics-silhouette"))
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn from_hexbits(h: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(h.trim(), 16).expect("hex bits"))
}

fn ulp_diff(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    let ia = a.to_bits() as i64;
    let ib = b.to_bits() as i64;
    let ma = if ia < 0 { i64::MIN - ia } else { ia };
    let mb = if ib < 0 { i64::MIN - ib } else { ib };
    ma.abs_diff(mb)
}

fn run(args: &[&str]) -> String {
    let out = Command::new(bin()).args(args).output().expect("run binary");
    assert!(
        out.status.success(),
        "binary failed for {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// Run single-threaded, the deterministic-reduction-order configuration the
/// goldens are compared against.
fn run_st(args: &[&str]) -> String {
    let mut a = args.to_vec();
    a.push("-t");
    a.push("1");
    run(&a)
}

/// Metrics whose distance path is scipy `cdist`'s exact per-pair fold; with the
/// single-threaded left-fold order these reproduce scipy bit-for-bit.
const BIT_EXACT_METRICS: [&str; 3] = ["cityblock", "manhattan", "sqeuclidean"];

#[test]
fn score_matches_sklearn() {
    let gd = golden_dir();
    let expected = std::fs::read_to_string(gd.join("expected_score.tsv")).unwrap();
    let mut checked = 0usize;
    let mut max_blas_diff = 0.0f64;
    let mut max_blas_ulp = 0u64;

    for line in expected.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (metric, target, want) = (c[0], c[1], from_hexbits(c[2]));

        // target is "matrix.tsv" or "matrix.tsv|labels.txt"
        let got: f64 = if let Some((mfile, lfile)) = target.split_once('|') {
            let m = gd.join(mfile);
            let l = gd.join(lfile);
            run_st(&[
                "--metric",
                metric,
                m.to_str().unwrap(),
                "--labels",
                l.to_str().unwrap(),
            ])
            .trim()
            .parse()
            .unwrap()
        } else {
            let m = gd.join(target);
            run_st(&["--metric", metric, m.to_str().unwrap()])
                .trim()
                .parse()
                .unwrap()
        };

        if BIT_EXACT_METRICS.contains(&metric) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{metric} {target}: got {got}, want {want} (cdist path must be bit-exact)"
            );
        } else {
            let diff = (got - want).abs();
            max_blas_diff = max_blas_diff.max(diff);
            max_blas_ulp = max_blas_ulp.max(ulp_diff(got, want));
            assert!(
                diff <= 1e-12,
                "{metric} {target}: got {got}, want {want}, |diff|={diff:e}"
            );
        }
        checked += 1;
    }

    assert!(checked >= 30, "expected >= 30 score rows, ran {checked}");
    eprintln!("score BLAS-path max |diff| = {max_blas_diff:e}, max ULP = {max_blas_ulp}");
}

#[test]
fn samples_match_sklearn() {
    let gd = golden_dir();
    let expected = std::fs::read_to_string(gd.join("expected_samples.tsv")).unwrap();
    let mut arrays = 0usize;
    let mut values = 0usize;
    let mut bit_exact = 0usize;
    let mut max_diff = 0.0f64;
    let mut max_ulp = 0u64;

    for line in expected.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (metric, mfile, sfile) = (c[0], c[1], c[2]);
        let m = gd.join(mfile);
        let out = run_st(&["--metric", metric, m.to_str().unwrap(), "--samples"]);
        let got: Vec<f64> = out.split_whitespace().map(|s| s.parse().unwrap()).collect();
        let want: Vec<f64> = std::fs::read_to_string(gd.join(sfile))
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(from_hexbits)
            .collect();
        assert_eq!(
            got.len(),
            want.len(),
            "{metric} {mfile}: sample count {} != {}",
            got.len(),
            want.len()
        );

        let bit = BIT_EXACT_METRICS.contains(&metric);
        for (&g, &w) in got.iter().zip(&want) {
            let diff = (g - w).abs();
            max_diff = max_diff.max(diff);
            max_ulp = max_ulp.max(ulp_diff(g, w));
            if g.to_bits() == w.to_bits() {
                bit_exact += 1;
            } else if bit {
                panic!("{metric} {mfile}: cdist sample not bit-exact: got {g}, want {w}");
            }
            assert!(
                diff <= 1e-12,
                "{metric} {mfile}: sample got {g}, want {w}, |diff|={diff:e}"
            );
            values += 1;
        }
        arrays += 1;
    }

    assert!(arrays >= 30, "expected >= 30 sample arrays, ran {arrays}");
    eprintln!(
        "samples: {arrays} arrays, {values} values, bit-exact {bit_exact}/{values}, max |diff| = {max_diff:e}, max ULP = {max_ulp}"
    );
}

/// With the default (multi-thread) reduction, the per-cluster accumulation
/// reorders, so values are no longer bit-exact but stay within ≤ 1e-12 of the
/// goldens across every metric.
#[test]
fn samples_within_tolerance_multithread() {
    let gd = golden_dir();
    let expected = std::fs::read_to_string(gd.join("expected_samples.tsv")).unwrap();
    let mut max_diff = 0.0f64;
    for line in expected.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (metric, mfile, sfile) = (c[0], c[1], c[2]);
        let m = gd.join(mfile);
        let out = run(&["--metric", metric, m.to_str().unwrap(), "--samples"]);
        let got: Vec<f64> = out.split_whitespace().map(|s| s.parse().unwrap()).collect();
        let want: Vec<f64> = std::fs::read_to_string(gd.join(sfile))
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(from_hexbits)
            .collect();
        for (&g, &w) in got.iter().zip(&want) {
            let diff = (g - w).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff <= 1e-12,
                "{metric} {mfile}: multithread sample got {g}, want {w}, |diff|={diff:e}"
            );
        }
    }
    eprintln!("multithread samples max |diff| = {max_diff:e}");
}

#[test]
fn last_column_matches_separate_labels() {
    let gd = golden_dir();
    let m = gd.join("sep_n100_d2_k3.tsv");
    let from_last = run(&["--metric", "euclidean", m.to_str().unwrap()]);

    let text = std::fs::read_to_string(&m).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mat = dir.path().join("mat.tsv");
    let lab = dir.path().join("lab.txt");
    let mut mbuf = String::new();
    let mut lbuf = String::new();
    for line in text.lines() {
        let mut cols: Vec<&str> = line.split_whitespace().collect();
        let l = cols.pop().unwrap();
        mbuf.push_str(&cols.join(" "));
        mbuf.push('\n');
        lbuf.push_str(l);
        lbuf.push('\n');
    }
    std::fs::write(&mat, mbuf).unwrap();
    std::fs::write(&lab, lbuf).unwrap();

    let from_sep = run(&[
        "--metric",
        "euclidean",
        mat.to_str().unwrap(),
        "--labels",
        lab.to_str().unwrap(),
    ]);
    assert_eq!(from_last.trim(), from_sep.trim(), "label layouts disagree");
}

#[test]
fn stdin_matches_file() {
    let gd = golden_dir();
    let m = gd.join("mid_n300_d8_k5.tsv");
    let from_file = run(&["--metric", "euclidean", m.to_str().unwrap()]);
    let text = std::fs::read_to_string(&m).unwrap();
    let out = Command::new(bin())
        .args(["--metric", "euclidean", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(text.as_bytes())
                .unwrap();
            child.wait_with_output()
        })
        .expect("run stdin");
    let from_stdin = String::from_utf8(out.stdout).unwrap();
    assert_eq!(from_file.trim(), from_stdin.trim(), "stdin != file");
}

#[test]
fn json_envelope_scalar() {
    let gd = golden_dir();
    let m = gd.join("mid_n300_d8_k5.tsv");
    let out = run(&["--metric", "euclidean", m.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(out.trim()).expect("one json envelope");
    assert_eq!(v["status"], "ok");
    assert!(v["result"].is_number(), "score result not a number: {out}");
}

#[test]
fn json_envelope_samples_array() {
    let gd = golden_dir();
    let m = gd.join("mid_n300_d8_k5.tsv");
    let out = run(&[
        "--metric",
        "euclidean",
        m.to_str().unwrap(),
        "--samples",
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(out.trim()).expect("one json envelope");
    assert_eq!(v["status"], "ok");
    assert!(v["result"].is_array(), "samples result not an array: {out}");
}

#[test]
fn single_cluster_fails_loud() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.tsv");
    std::fs::write(&path, "1 2 0\n3 4 0\n5 6 0\n").unwrap();
    let out = Command::new(bin())
        .args(["--metric", "euclidean", path.to_str().unwrap()])
        .output()
        .expect("run");
    assert!(!out.status.success(), "single-cluster input must fail loud");
}

#[test]
fn k_equals_n_fails_loud() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kn.tsv");
    std::fs::write(&path, "1 2 0\n3 4 1\n5 6 2\n").unwrap();
    let out = Command::new(bin())
        .args(["--metric", "euclidean", path.to_str().unwrap()])
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "n_labels == n_samples must fail loud"
    );
}

/// Textually-varied integer labels (`-0`/`0`/`00`, `+2`/`2`) collapse by numeric
/// value the way scikit-learn's integer `LabelEncoder` does. Compared against a
/// committed golden generated from sklearn 1.9.0 (no live oracle).
#[test]
fn int_label_dedup_matches_sklearn() {
    let gd = golden_dir();
    let m = gd.join("intdedup_X.tsv");
    let score: f64 = run_st(&["--metric", "euclidean", m.to_str().unwrap()])
        .trim()
        .parse()
        .unwrap();
    let want_score = from_hexbits("3fef963508dcf81b");
    assert!(
        (score - want_score).abs() <= 1e-12,
        "int-dedup score got {score}, want {want_score}"
    );

    let out = run_st(&["--metric", "euclidean", m.to_str().unwrap(), "--samples"]);
    let got: Vec<f64> = out.split_whitespace().map(|s| s.parse().unwrap()).collect();
    let want: Vec<f64> = std::fs::read_to_string(gd.join("intdedup__euclidean.samples.hex"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(from_hexbits)
        .collect();
    assert_eq!(got.len(), want.len(), "int-dedup sample count mismatch");
    assert_eq!(
        got.len(),
        8,
        "int-dedup expects 8 samples (2 clusters, no panic)"
    );
    for (&g, &w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-12, "int-dedup sample got {g}, want {w}");
    }
}

/// A non-finite feature cell (NaN / infinity) must fail loud, matching
/// scikit-learn's `check_X_y(force_all_finite=True)` `ValueError` — never a
/// silent score-0.
#[test]
fn nonfinite_feature_fails_loud() {
    for cell in ["nan", "inf", "-inf", "Infinity"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.tsv");
        std::fs::write(&path, format!("1.0 0\n{cell} 0\n8.0 1\n9.0 1\n")).unwrap();
        let out = Command::new(bin())
            .args(["--metric", "euclidean", path.to_str().unwrap()])
            .output()
            .expect("run");
        assert!(
            !out.status.success(),
            "non-finite feature '{cell}' must fail loud, got exit {:?}",
            out.status.code()
        );
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            !err.trim().is_empty(),
            "non-finite feature '{cell}' must print a diagnostic to stderr"
        );
    }
}

#[test]
fn help_exits_zero() {
    let out = Command::new(bin())
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(out.status.success(), "--help did not exit 0");
}
