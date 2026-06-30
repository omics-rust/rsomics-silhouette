use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use serde::Serialize;

use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};

use rsomics_silhouette::{
    Metric, read_dataset, silhouette_samples, silhouette_score, valid_label_count,
};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MetricArg {
    Euclidean,
    Cosine,
    /// Alias for `manhattan`.
    Cityblock,
    Manhattan,
    Sqeuclidean,
}

impl From<MetricArg> for Metric {
    fn from(m: MetricArg) -> Self {
        match m {
            MetricArg::Euclidean => Metric::Euclidean,
            MetricArg::Cosine => Metric::Cosine,
            MetricArg::Cityblock | MetricArg::Manhattan => Metric::Cityblock,
            MetricArg::Sqeuclidean => Metric::Sqeuclidean,
        }
    }
}

/// Silhouette clustering-quality coefficient — value-exact `scikit-learn`
/// equivalent.
///
/// Reads a feature matrix (`n` rows × `d` whitespace-separated columns) and a
/// cluster labeling. Labels are the matrix's last column, or a separate
/// one-per-line `--labels FILE`. Prints `silhouette_score` (the mean over all
/// samples); `--samples` prints the per-sample coefficient `s(i)` instead, one
/// per line in input order.
///
/// `--metric` selects the distance (euclidean default; also cosine,
/// cityblock/manhattan, sqeuclidean), matching scikit-learn's
/// `pairwise_distances`. Requires `2 ≤ n_clusters ≤ n_samples − 1`.
#[derive(Parser, Debug)]
#[command(name = "rsomics-silhouette", version, about, long_about = None)]
pub struct Cli {
    /// Feature matrix TSV (`-` or omitted reads stdin).
    #[arg(value_name = "MATRIX")]
    pub matrix: Option<PathBuf>,

    /// Cluster labels, one per line; if omitted the matrix's last column is used.
    #[arg(long = "labels", value_name = "FILE")]
    pub labels: Option<PathBuf>,

    /// Distance metric.
    #[arg(long = "metric", value_enum, default_value_t = MetricArg::Euclidean)]
    pub metric: MetricArg,

    /// Emit per-sample silhouette values instead of the mean score.
    #[arg(long = "samples")]
    pub samples: bool,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Output {
    Score(f64),
    Samples(Vec<f64>),
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        let metric: Metric = self.metric.into();
        run(&common, META, || {
            let d = read_dataset(self.matrix.as_deref(), self.labels.as_deref())?;
            if !valid_label_count(d.n_clusters, d.x.rows) {
                return Err(RsomicsError::InvalidInput(format!(
                    "Number of labels is {}. Valid values are 2 to n_samples - 1 (inclusive)",
                    d.n_clusters
                )));
            }

            let out = if self.samples {
                Output::Samples(silhouette_samples(&d, metric))
            } else {
                Output::Score(silhouette_score(&d, metric))
            };

            if !common.json {
                let stdout = std::io::stdout().lock();
                let mut w = BufWriter::new(stdout);
                match &out {
                    Output::Score(v) => writeln!(w, "{}", fmt(*v)).map_err(RsomicsError::Io)?,
                    Output::Samples(vs) => {
                        for v in vs {
                            writeln!(w, "{}", fmt(*v)).map_err(RsomicsError::Io)?;
                        }
                    }
                }
                w.flush().map_err(RsomicsError::Io)?;
            }
            Ok(out)
        })
    }
}

/// Shortest round-trip decimal, scientific for tiny or huge magnitudes.
fn fmt(x: f64) -> String {
    if x != 0.0 && (x.abs() < 1e-4 || x.abs() >= 1e16) {
        format!("{x:e}")
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
