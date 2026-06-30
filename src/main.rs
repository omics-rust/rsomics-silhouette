mod cli;

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    cli::Cli::parse().run()
}
