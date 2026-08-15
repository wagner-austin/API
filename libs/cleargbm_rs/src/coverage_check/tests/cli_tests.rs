//! Tests for command-line parsing and orchestration.

use std::path::{Path, PathBuf};

use crate::coverage_check::cli::{parse_options, parse_threshold, run, EXIT_FAIL, EXIT_PASS};
use crate::coverage_check::fs::MemoryFileSystem;
use crate::coverage_check::types::{CoverageCheckError, Options};

/// Builds an export document covering one file at the given counts.
fn export_document(filename: &str, counts: &[u64]) -> String {
    let segments: Vec<String> = counts
        .iter()
        .enumerate()
        .map(|(index, count)| {
            let line = index.saturating_add(1_usize);
            format!("[{line}, 1, {count}, true, true, false]")
        })
        .collect();
    let joined = segments.join(", ");
    format!(r#"{{"data": [{{"files": [{{"filename": "{filename}", "segments": [{joined}]}}]}}]}}"#)
}

/// Builds options pointing at the given export path with a 100% threshold.
fn options_for(path: &str) -> Options {
    Options {
        json_path: PathBuf::from(path),
        threshold: 100.0_f64,
    }
}

// ── parse_threshold ────────────────────────────────────────────────

#[test]
fn test_parse_threshold_accepts_decimal_forms() -> Result<(), CoverageCheckError> {
    for token in ["100", "99.5", "0", ".5", "+80", "12."] {
        match parse_threshold(token) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[test]
fn test_parse_threshold_rejects_non_numeric() -> Result<(), CoverageCheckError> {
    for token in ["abc", "", "1.2.3", "1e5", "inf", "NaN", "+", "-", "."] {
        match parse_threshold(token) {
            Ok(_) => return Err(CoverageCheckError::ExportDataEmpty),
            Err(e) => assert_eq!(
                e,
                CoverageCheckError::ThresholdNotANumber {
                    token: token.to_owned(),
                }
            ),
        }
    }
    Ok(())
}

#[test]
fn test_parse_threshold_rejects_out_of_range() -> Result<(), CoverageCheckError> {
    match parse_threshold("101") {
        Ok(_) => return Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => assert_eq!(
            e,
            CoverageCheckError::ThresholdOutOfRange { value: 101.0_f64 }
        ),
    }
    match parse_threshold("-1") {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::ThresholdOutOfRange { value: -1.0_f64 }
            );
            Ok(())
        }
    }
}

// ── parse_options ──────────────────────────────────────────────────

#[test]
fn test_parse_options_applies_defaults() -> Result<(), CoverageCheckError> {
    let parsed = match parse_options(&[]) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(parsed.json_path, PathBuf::from("coverage.json"));
    assert!((parsed.threshold - 100.0_f64).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn test_parse_options_reads_both_flags() -> Result<(), CoverageCheckError> {
    let tokens = vec![
        "--json".to_owned(),
        "other.json".to_owned(),
        "--threshold".to_owned(),
        "90".to_owned(),
    ];
    let parsed = match parse_options(&tokens) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(parsed.json_path, PathBuf::from("other.json"));
    assert!((parsed.threshold - 90.0_f64).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn test_parse_options_rejects_unknown_token() -> Result<(), CoverageCheckError> {
    let tokens = vec!["--verbose".to_owned()];
    match parse_options(&tokens) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::CliFlagUnknown {
                    token: "--verbose".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_parse_options_rejects_json_flag_without_value() -> Result<(), CoverageCheckError> {
    let tokens = vec!["--json".to_owned()];
    match parse_options(&tokens) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::CliFlagMissingValue {
                    flag: "--json".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_parse_options_rejects_threshold_flag_without_value() -> Result<(), CoverageCheckError> {
    let tokens = vec!["--threshold".to_owned()];
    match parse_options(&tokens) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::CliFlagMissingValue {
                    flag: "--threshold".to_owned(),
                }
            );
            Ok(())
        }
    }
}

#[test]
fn test_parse_options_propagates_threshold_error() -> Result<(), CoverageCheckError> {
    let tokens = vec!["--threshold".to_owned(), "abc".to_owned()];
    match parse_options(&tokens) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(
                e,
                CoverageCheckError::ThresholdNotANumber {
                    token: "abc".to_owned(),
                }
            );
            Ok(())
        }
    }
}

// ── run ────────────────────────────────────────────────────────────

#[test]
fn test_run_reports_missing_export() -> Result<(), CoverageCheckError> {
    let files = MemoryFileSystem::new();
    match run(&files, &options_for("coverage.json")) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            match e {
                CoverageCheckError::ExportNotFound { path } => {
                    assert!(path.contains("coverage.json"));
                }
                other => return Err(other),
            }
            Ok(())
        }
    }
}

#[test]
fn test_run_reports_unreadable_export() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("coverage.json");
    let files = MemoryFileSystem::new().with_unreadable(&path, "permission denied");
    match run(&files, &options_for("coverage.json")) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            match e {
                CoverageCheckError::ExportUnreadable { reason, .. } => {
                    assert_eq!(reason, "permission denied");
                }
                other => return Err(other),
            }
            Ok(())
        }
    }
}

#[test]
fn test_run_propagates_decode_failure() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("coverage.json");
    let files = MemoryFileSystem::new().with_file(&path, "{not json");
    match run(&files, &options_for("coverage.json")) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            match e {
                CoverageCheckError::ExportNotJson { .. } => {}
                other => return Err(other),
            }
            Ok(())
        }
    }
}

#[test]
fn test_run_passes_on_full_coverage() -> Result<(), CoverageCheckError> {
    let export_path = PathBuf::from("coverage.json");
    let files = MemoryFileSystem::new()
        .with_file(&export_path, &export_document("src/a.rs", &[1_u64, 2_u64]));
    let report = match run(&files, &options_for("coverage.json")) {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    assert!(report.passed);
    assert!(report
        .lines
        .iter()
        .any(|line| line.contains("All source lines are covered!")));
    assert!(report.lines.iter().any(|line| line.starts_with("PASS:")));
    Ok(())
}

#[test]
fn test_run_fails_and_lists_uncovered_lines() -> Result<(), CoverageCheckError> {
    let export_path = PathBuf::from("coverage.json");
    let source_path = Path::new("src/a.rs");
    let files = MemoryFileSystem::new()
        .with_file(&export_path, &export_document("src/a.rs", &[1_u64, 0_u64]))
        .with_file(source_path, "fn covered() {}\nfn missed() {}\n");
    let report = match run(&files, &options_for("coverage.json")) {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    assert!(!report.passed);
    assert!(report
        .lines
        .iter()
        .any(|line| line.contains("fn missed() {}")));
    assert!(report.lines.iter().any(|line| line.starts_with("FAIL:")));
    Ok(())
}

#[test]
fn test_exit_codes_are_distinct() -> Result<(), CoverageCheckError> {
    assert_eq!(EXIT_PASS, 0_u8);
    assert_eq!(EXIT_FAIL, 1_u8);
    Ok(())
}
