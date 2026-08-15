//! Command-line parsing and orchestration for the coverage checker.

use std::path::{Path, PathBuf};

use crate::coverage_check::analysis::{meets_threshold, summarize};
use crate::coverage_check::decode::decode_coverage_export;
use crate::coverage_check::fs::FileSystem;
use crate::coverage_check::render::{render_summary, render_verdict};
use crate::coverage_check::types::{CoverageCheckError, Options};

/// Flag naming the coverage export to read.
const FLAG_JSON: &str = "--json";

/// Flag naming the minimum coverage percentage.
const FLAG_THRESHOLD: &str = "--threshold";

/// Export path used when `--json` is absent.
const DEFAULT_JSON_PATH: &str = "coverage.json";

/// Threshold used when `--threshold` is absent.
const DEFAULT_THRESHOLD: f64 = 100.0_f64;

/// Lowest accepted threshold.
const MINIMUM_THRESHOLD: f64 = 0.0_f64;

/// Highest accepted threshold.
const MAXIMUM_THRESHOLD: f64 = 100.0_f64;

/// Exit code reported when coverage meets the threshold.
pub const EXIT_PASS: u8 = 0_u8;

/// Exit code reported when coverage falls short.
pub const EXIT_FAIL: u8 = 1_u8;

/// Parses a token written in plain decimal notation.
///
/// Validation and parsing are one step so there is no second, unreachable
/// failure path: exponent forms, `inf` and `NaN` -- all of which Rust's float
/// parser would otherwise accept -- are rejected by the same pass that decides
/// the token is a number at all.
///
/// Returns `None` when the token is not plain decimal notation.
fn parse_plain_decimal(token: &str) -> Option<f64> {
    let body = match token.strip_prefix(['+', '-']) {
        Some(rest) => rest,
        None => token,
    };
    let mut seen_dot = false;
    let mut seen_digit = false;
    for character in body.chars() {
        if character == '.' {
            if seen_dot {
                return None;
            }
            seen_dot = true;
        } else if character.is_ascii_digit() {
            seen_digit = true;
        } else {
            return None;
        }
    }
    if !seen_digit {
        return None;
    }
    token.parse::<f64>().ok()
}

/// Parses and range-checks a threshold argument.
///
/// # Errors
///
/// Returns [`CoverageCheckError::ThresholdNotANumber`] when the token is not
/// plain decimal notation, or [`CoverageCheckError::ThresholdOutOfRange`] when
/// it falls outside 0-100.
pub fn parse_threshold(token: &str) -> Result<f64, CoverageCheckError> {
    let value = match parse_plain_decimal(token) {
        Some(parsed) => parsed,
        None => {
            return Err(CoverageCheckError::ThresholdNotANumber {
                token: token.to_owned(),
            })
        }
    };
    if !(MINIMUM_THRESHOLD..=MAXIMUM_THRESHOLD).contains(&value) {
        return Err(CoverageCheckError::ThresholdOutOfRange { value });
    }
    Ok(value)
}

/// Reads the value that follows a flag.
///
/// # Errors
///
/// Returns [`CoverageCheckError::CliFlagMissingValue`] when the flag ends the
/// argument list.
fn require_value<'a>(
    tokens: &'a [String],
    index: usize,
    flag: &str,
) -> Result<&'a str, CoverageCheckError> {
    match tokens.get(index) {
        Some(value) => Ok(value.as_str()),
        None => Err(CoverageCheckError::CliFlagMissingValue {
            flag: flag.to_owned(),
        }),
    }
}

/// Parses command-line tokens into typed options.
///
/// # Errors
///
/// Returns [`CoverageCheckError::CliFlagUnknown`] for an unrecognised token,
/// [`CoverageCheckError::CliFlagMissingValue`] for a flag with no value, or a
/// threshold error when `--threshold` is malformed.
pub fn parse_options(tokens: &[String]) -> Result<Options, CoverageCheckError> {
    let mut json_path = PathBuf::from(DEFAULT_JSON_PATH);
    let mut threshold = DEFAULT_THRESHOLD;
    let mut index = 0_usize;

    while index < tokens.len() {
        let token = tokens[index].as_str();
        if token == FLAG_JSON {
            let value = propagate!(require_value(
                tokens,
                index.saturating_add(1_usize),
                FLAG_JSON
            ));
            json_path = PathBuf::from(value);
            index = index.saturating_add(2_usize);
        } else if token == FLAG_THRESHOLD {
            let value = propagate!(require_value(
                tokens,
                index.saturating_add(1_usize),
                FLAG_THRESHOLD
            ));
            threshold = propagate!(parse_threshold(value));
            index = index.saturating_add(2_usize);
        } else {
            return Err(CoverageCheckError::CliFlagUnknown {
                token: token.to_owned(),
            });
        }
    }

    Ok(Options {
        json_path,
        threshold,
    })
}

/// Outcome of a checker run: the report lines and whether coverage passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunReport {
    /// Report lines, in the order they should be written.
    pub lines: Vec<String>,
    /// Whether coverage met the threshold.
    pub passed: bool,
}

/// Checks segment coverage and builds the report.
///
/// # Errors
///
/// Returns [`CoverageCheckError::ExportNotFound`] when the export is absent,
/// [`CoverageCheckError::ExportUnreadable`] when it cannot be read, or a
/// decode error when it is malformed.
pub fn run(files: &dyn FileSystem, options: &Options) -> Result<RunReport, CoverageCheckError> {
    let path: &Path = options.json_path.as_path();
    let display = path.display().to_string();
    if !files.exists(path) {
        return Err(CoverageCheckError::ExportNotFound { path: display });
    }
    let raw = match files.read_to_string(path) {
        Ok(contents) => contents,
        Err(reason) => {
            return Err(CoverageCheckError::ExportUnreadable {
                path: display,
                reason,
            })
        }
    };
    let export = propagate!(decode_coverage_export(&raw, &display));
    let summary = propagate!(summarize(&export));
    let mut lines = propagate!(render_summary(files, &summary, options.threshold));
    let passed = meets_threshold(&summary, options.threshold);
    lines.push(render_verdict(&summary, options.threshold, passed));
    Ok(RunReport { lines, passed })
}
