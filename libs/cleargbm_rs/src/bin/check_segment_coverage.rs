//! Fails the build when this crate's segment coverage falls below a threshold.
//!
//! Usage:
//!
//! ```text
//! cargo llvm-cov --all-features --json --output-path coverage.json
//! cargo run --bin check_segment_coverage -- --threshold 100
//! ```
//!
//! All logic lives in [`cleargbm_rs::coverage_check`]; this entry point only
//! wires the real filesystem, writes the report, and maps the outcome to an
//! exit code.

use std::process::ExitCode;

use cleargbm_rs::coverage_check::cli::{parse_options, run, EXIT_FAIL, EXIT_PASS};
use cleargbm_rs::coverage_check::fs::RealFileSystem;

/// Runs the checker over the process arguments.
fn main() -> ExitCode {
    let tokens: Vec<String> = std::env::args().skip(1_usize).collect();
    let options = match parse_options(&tokens) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(EXIT_FAIL);
        }
    };
    let files = RealFileSystem;
    match run(&files, &options) {
        Ok(report) => {
            for line in &report.lines {
                println!("{line}");
            }
            if report.passed {
                ExitCode::from(EXIT_PASS)
            } else {
                ExitCode::from(EXIT_FAIL)
            }
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(EXIT_FAIL)
        }
    }
}
