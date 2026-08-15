//! Tests for the pure coverage computation.

use crate::coverage_check::analysis::{
    is_crate_source, meets_threshold, merge_segment_counts, percentage, summarize, summarize_file,
};
use crate::coverage_check::types::{
    CoverageCheckError, CoverageExport, CoverageSummary, FileCoverage, Segment,
};

/// Builds a segment with the given line, count and count-validity.
fn segment(line: usize, count: u64, has_count: bool) -> Segment {
    Segment {
        line,
        column: 1_usize,
        count,
        has_count,
        is_region_entry: true,
        is_gap_region: false,
    }
}

/// Builds a file record from a list of segments.
fn file(filename: &str, segments: Vec<Segment>) -> FileCoverage {
    FileCoverage {
        filename: filename.to_owned(),
        segments,
    }
}

// ── is_crate_source ────────────────────────────────────────────────

#[test]
fn test_is_crate_source_accepts_both_separators() -> Result<(), CoverageCheckError> {
    assert!(is_crate_source("C:\\repo\\src\\tree\\mod.rs"));
    assert!(is_crate_source("/repo/src/tree/mod.rs"));
    Ok(())
}

#[test]
fn test_is_crate_source_rejects_dependency_paths() -> Result<(), CoverageCheckError> {
    assert!(!is_crate_source(
        "/home/u/.cargo/registry/rayon-1.10/lib.rs"
    ));
    Ok(())
}

// ── merge_segment_counts ───────────────────────────────────────────

#[test]
fn test_merge_skips_segments_without_count() -> Result<(), CoverageCheckError> {
    let record = file("src/a.rs", vec![segment(10_usize, 7_u64, false)]);
    let merged = merge_segment_counts(&record);
    assert!(merged.is_empty());
    Ok(())
}

#[test]
fn test_merge_keeps_highest_count_per_line() -> Result<(), CoverageCheckError> {
    let record = file(
        "src/a.rs",
        vec![
            segment(10_usize, 0_u64, true),
            segment(10_usize, 5_u64, true),
            segment(10_usize, 2_u64, true),
        ],
    );
    let merged = merge_segment_counts(&record);
    assert_eq!(merged.get(&10_usize), Some(&5_u64));
    Ok(())
}

#[test]
fn test_merge_records_each_line_once() -> Result<(), CoverageCheckError> {
    let record = file(
        "src/a.rs",
        vec![segment(1_usize, 1_u64, true), segment(2_usize, 0_u64, true)],
    );
    let merged = merge_segment_counts(&record);
    assert_eq!(merged.len(), 2_usize);
    Ok(())
}

// ── summarize_file ─────────────────────────────────────────────────

#[test]
fn test_summarize_file_splits_covered_and_uncovered() -> Result<(), CoverageCheckError> {
    let record = file(
        "src/a.rs",
        vec![
            segment(1_usize, 3_u64, true),
            segment(2_usize, 0_u64, true),
            segment(3_usize, 0_u64, true),
        ],
    );
    let gap = summarize_file(&record);
    assert_eq!(gap.covered, 1_usize);
    assert_eq!(gap.total, 3_usize);
    assert_eq!(gap.uncovered, vec![2_usize, 3_usize]);
    Ok(())
}

// ── percentage ─────────────────────────────────────────────────────

#[test]
fn test_percentage_of_empty_run_is_zero() -> Result<(), CoverageCheckError> {
    let value = match percentage(0_usize, 0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((value - 0.0_f64).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn test_percentage_computes_ratio() -> Result<(), CoverageCheckError> {
    let value = match percentage(1_usize, 4_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((value - 25.0_f64).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn test_percentage_rejects_counts_beyond_exact_conversion() -> Result<(), CoverageCheckError> {
    let huge = 5_000_000_000_usize;
    match percentage(0_usize, huge) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(e, CoverageCheckError::CountTooLarge { value: huge });
            Ok(())
        }
    }
}

#[test]
fn test_percentage_rejects_large_covered_count() -> Result<(), CoverageCheckError> {
    let huge = 5_000_000_000_usize;
    match percentage(huge, huge) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(e) => {
            assert_eq!(e, CoverageCheckError::CountTooLarge { value: huge });
            Ok(())
        }
    }
}

// ── summarize ──────────────────────────────────────────────────────

#[test]
fn test_summarize_ignores_non_crate_files() -> Result<(), CoverageCheckError> {
    let export = CoverageExport {
        files: vec![file(
            "/home/u/.cargo/registry/dep/lib.rs",
            vec![segment(1_usize, 0_u64, true)],
        )],
    };
    let summary = match summarize(&export) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };
    assert_eq!(summary.total, 0_usize);
    assert!(summary.gaps.is_empty());
    Ok(())
}

#[test]
fn test_summarize_collects_gaps_and_totals() -> Result<(), CoverageCheckError> {
    let export = CoverageExport {
        files: vec![
            file(
                "src/a.rs",
                vec![segment(1_usize, 1_u64, true), segment(2_usize, 0_u64, true)],
            ),
            file("src/b.rs", vec![segment(1_usize, 4_u64, true)]),
        ],
    };
    let summary = match summarize(&export) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };
    assert_eq!(summary.covered, 2_usize);
    assert_eq!(summary.total, 3_usize);
    assert_eq!(summary.gaps.len(), 1_usize);
    assert_eq!(summary.gaps[0].filename, "src/a.rs");
    Ok(())
}

#[test]
fn test_summarize_reports_full_coverage_without_gaps() -> Result<(), CoverageCheckError> {
    let export = CoverageExport {
        files: vec![file("src/a.rs", vec![segment(1_usize, 1_u64, true)])],
    };
    let summary = match summarize(&export) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };
    assert!((summary.percent - 100.0_f64).abs() < f64::EPSILON);
    assert!(summary.gaps.is_empty());
    Ok(())
}

// ── meets_threshold ────────────────────────────────────────────────

#[test]
fn test_meets_threshold_boundary_and_shortfall() -> Result<(), CoverageCheckError> {
    let summary = CoverageSummary {
        covered: 1_usize,
        total: 1_usize,
        percent: 100.0_f64,
        gaps: Vec::new(),
    };
    assert!(meets_threshold(&summary, 100.0_f64));

    let short = CoverageSummary {
        covered: 1_usize,
        total: 2_usize,
        percent: 50.0_f64,
        gaps: Vec::new(),
    };
    assert!(!meets_threshold(&short, 100.0_f64));
    Ok(())
}
