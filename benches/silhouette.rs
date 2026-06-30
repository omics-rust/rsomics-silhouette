//! Criterion bench: silhouette_score on a moderate feature matrix, the same
//! O(n²·d) work scikit-learn's `silhouette_score` performs. The fair upstream
//! comparison (this binary single-threaded vs scikit-learn in-process) lives in
//! the perf record; this bench tracks our own regression.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_silhouette::{Metric, parse_dataset, silhouette_score};

fn make_blobs(n: usize, d: usize, k: usize) -> String {
    let mut s = String::new();
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..n {
        let c = i % k;
        for _ in 0..d {
            let v = c as f64 * 10.0 + next();
            s.push_str(&format!("{v} "));
        }
        s.push_str(&format!("{c}\n"));
    }
    s
}

fn bench(c: &mut Criterion) {
    let text = make_blobs(4000, 20, 6);
    let d = parse_dataset(&text, None).unwrap();
    c.bench_function("silhouette_score n4000 d20 k6 euclidean", |b| {
        b.iter(|| black_box(silhouette_score(black_box(&d), Metric::Euclidean)));
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
