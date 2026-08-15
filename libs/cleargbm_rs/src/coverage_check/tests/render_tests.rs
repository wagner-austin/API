//! Tests for report rendering.

use std::path::PathBuf;

use crate::coverage_check::fs::MemoryFileSystem;
use crate::coverage_check::render::{render_gap, render_summary, render_verdict};
use crate::coverage_check::types::{CoverageCheckError, CoverageSummary, FileGap};

/// Builds a gap record for a file with the given uncovered lines.
fn gap(filename: &str, covered: usize, total: usize, uncovered: Vec<usize>) -> FileGap {
    FileGap {
        filename: filename.to_owned(),
        covered,
        total,
        uncovered,
    }
}

#[test]
fn test_render_gap_shows_source_text() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("src/a.rs");
    let files = MemoryFileSystem::new().with_file(&path, "line one\n    line two\nline three\n");
    let record = gap("src/a.rs", 2_usize, 3_usize, vec![2_usize]);
    let lines = match render_gap(&files, &record) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };
    assert!(lines.iter().any(|line| line.contains("line two")));
    assert!(lines.iter().any(|line| line.contains("a.rs")));
    Ok(())
}

#[test]
fn test_render_gap_reports_missing_source_file() -> Result<(), CoverageCheckError> {
    let files = MemoryFileSystem::new();
    let record = gap("src/gone.rs", 0_usize, 1_usize, vec![1_usize]);
    match render_gap(&files, &record) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            match e {
                CoverageCheckError::SourceFileMissing { path } => {
                    assert!(path.contains("gone.rs"));
                }
                other => return Err(other),
            }
            Ok(())
        }
    }
}

#[test]
fn test_render_gap_reports_unreadable_source_file() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("src/locked.rs");
    let files = MemoryFileSystem::new().with_unreadable(&path, "permission denied");
    let record = gap("src/locked.rs", 0_usize, 1_usize, vec![1_usize]);
    match render_gap(&files, &record) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            match e {
                CoverageCheckError::SourceFileUnreadable { path, reason } => {
                    assert!(path.contains("locked.rs"));
                    assert_eq!(reason, "permission denied");
                }
                other => return Err(other),
            }
            Ok(())
        }
    }
}

#[test]
fn test_render_gap_reports_line_beyond_end_of_file() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("src/short.rs");
    let files = MemoryFileSystem::new().with_file(&path, "only line\n");
    let record = gap("src/short.rs", 0_usize, 1_usize, vec![99_usize]);
    match render_gap(&files, &record) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            match e {
                CoverageCheckError::SourceLineOutOfRange {
                    path,
                    line_no,
                    line_count,
                } => {
                    assert!(path.contains("short.rs"));
                    assert_eq!(line_no, 99_usize);
                    assert_eq!(line_count, 1_usize);
                }
                other => return Err(other),
            }
            Ok(())
        }
    }
}

#[test]
fn test_render_gap_reports_zero_line_number() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("src/zero.rs");
    let files = MemoryFileSystem::new().with_file(&path, "only line\n");
    let record = gap("src/zero.rs", 0_usize, 1_usize, vec![0_usize]);
    match render_gap(&files, &record) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            match e {
                CoverageCheckError::SourceLineOutOfRange { line_no, .. } => {
                    assert_eq!(line_no, 0_usize);
                }
                other => return Err(other),
            }
            Ok(())
        }
    }
}

#[test]
fn test_render_gap_uses_full_path_when_there_is_no_file_name() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("..");
    let files = MemoryFileSystem::new().with_file(&path, "content\n");
    let record = gap("..", 0_usize, 1_usize, vec![1_usize]);
    let lines = match render_gap(&files, &record) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };
    assert!(lines.iter().any(|line| line.contains("..")));
    Ok(())
}

#[test]
fn test_render_gap_of_empty_file_reports_zero_percent() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("src/empty.rs");
    let files = MemoryFileSystem::new().with_file(&path, "a line\n");
    let record = gap("src/empty.rs", 0_usize, 0_usize, vec![1_usize]);
    let lines = match render_gap(&files, &record) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };
    assert!(lines.iter().any(|line| line.contains("0.0%")));
    Ok(())
}

#[test]
fn test_render_summary_reports_full_coverage() -> Result<(), CoverageCheckError> {
    let files = MemoryFileSystem::new();
    let summary = CoverageSummary {
        covered: 4_usize,
        total: 4_usize,
        percent: 100.0_f64,
        gaps: Vec::new(),
    };
    let lines = match render_summary(&files, &summary, 100.0_f64) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };
    assert!(lines
        .iter()
        .any(|line| line.contains("All source lines are covered!")));
    assert!(lines.iter().any(|line| line.contains("4/4 lines")));
    Ok(())
}

#[test]
fn test_render_summary_lists_files_with_gaps() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("src/a.rs");
    let files = MemoryFileSystem::new().with_file(&path, "one\ntwo\n");
    let summary = CoverageSummary {
        covered: 1_usize,
        total: 2_usize,
        percent: 50.0_f64,
        gaps: vec![gap("src/a.rs", 1_usize, 2_usize, vec![2_usize])],
    };
    let lines = match render_summary(&files, &summary, 100.0_f64) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };
    assert!(lines
        .iter()
        .any(|line| line.contains("FILES WITH UNCOVERED LINES (1):")));
    Ok(())
}

#[test]
fn test_render_verdict_reports_pass_and_fail() -> Result<(), CoverageCheckError> {
    let summary = CoverageSummary {
        covered: 1_usize,
        total: 1_usize,
        percent: 100.0_f64,
        gaps: Vec::new(),
    };
    let passed = render_verdict(&summary, 100.0_f64, true);
    assert!(passed.starts_with("PASS: Coverage 100.00% >= 100.00%"));

    let failed = render_verdict(&summary, 100.0_f64, false);
    assert!(failed.starts_with("FAIL: Coverage 100.00% < 100.00%"));
    Ok(())
}
