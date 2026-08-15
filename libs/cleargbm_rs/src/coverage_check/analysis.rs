//! Pure coverage computation over a decoded export.
//!
//! Segments merge coverage across every instantiation of generic code: a line
//! counts as covered when *any* instantiation executed it. Counting merged
//! segments instead of per-instantiation regions is what stops monomorphised
//! generics from being reported as phantom uncovered lines.
//!
//! Nothing here performs I/O, so every function is testable against a literal
//! export.

use std::collections::BTreeMap;

use crate::coverage_check::types::{
    CoverageCheckError, CoverageExport, CoverageSummary, FileCoverage, FileGap,
};

/// Path fragments marking a file as first-party crate source. cargo-llvm-cov
/// records absolute paths using the host separator.
const CRATE_SOURCE_MARKERS: [&str; 2] = ["src\\", "src/"];

/// Reports whether a coverage record belongs to this crate's own source.
#[must_use]
pub fn is_crate_source(filename: &str) -> bool {
    CRATE_SOURCE_MARKERS
        .iter()
        .any(|marker| filename.contains(marker))
}

/// Converts a line count to `f64` for percentage arithmetic.
///
/// # Errors
///
/// Returns [`CoverageCheckError::CountTooLarge`] when the count exceeds the
/// range that converts to `f64` without precision loss.
fn count_to_f64(value: usize) -> Result<f64, CoverageCheckError> {
    match u32::try_from(value) {
        Ok(converted) => Ok(f64::from(converted)),
        Err(_) => Err(CoverageCheckError::CountTooLarge { value }),
    }
}

/// Merges segment execution counts down to one count per source line.
///
/// Segments without a count carry no execution information and are skipped.
/// Where several segments report the same line, the highest count wins, which
/// is what makes a line covered if any generic instantiation executed it.
#[must_use]
pub fn merge_segment_counts(file_coverage: &FileCoverage) -> BTreeMap<usize, u64> {
    let mut line_counts: BTreeMap<usize, u64> = BTreeMap::new();
    for segment in &file_coverage.segments {
        if !segment.has_count {
            continue;
        }
        let entry = line_counts.entry(segment.line).or_insert(segment.count);
        if segment.count > *entry {
            *entry = segment.count;
        }
    }
    line_counts
}

/// Computes the coverage outcome for one source file.
#[must_use]
pub fn summarize_file(file_coverage: &FileCoverage) -> FileGap {
    let line_counts = merge_segment_counts(file_coverage);
    let mut covered = 0_usize;
    let mut uncovered: Vec<usize> = Vec::new();
    for (line, count) in &line_counts {
        if *count > 0_u64 {
            covered = covered.saturating_add(1_usize);
        } else {
            uncovered.push(*line);
        }
    }
    FileGap {
        filename: file_coverage.filename.clone(),
        covered,
        total: line_counts.len(),
        uncovered,
    }
}

/// Expresses covered lines as a percentage of total lines.
///
/// An empty run reports 0.0 rather than 100.0, so a coverage job that produced
/// no data fails the threshold instead of silently passing it.
///
/// # Errors
///
/// Returns [`CoverageCheckError::CountTooLarge`] when a count cannot be
/// converted for the division.
pub fn percentage(covered: usize, total: usize) -> Result<f64, CoverageCheckError> {
    if total == 0_usize {
        return Ok(0.0_f64);
    }
    let covered_f = propagate!(count_to_f64(covered));
    let total_f = propagate!(count_to_f64(total));
    Ok(100.0_f64 * covered_f / total_f)
}

/// Computes the whole-run coverage outcome.
///
/// Only this crate's own source files are counted; dependency and generated
/// files present in the export are ignored.
///
/// # Errors
///
/// Returns [`CoverageCheckError::CountTooLarge`] when a count cannot be
/// converted for percentage arithmetic.
pub fn summarize(export: &CoverageExport) -> Result<CoverageSummary, CoverageCheckError> {
    let mut total_covered = 0_usize;
    let mut total_lines = 0_usize;
    let mut gaps: Vec<FileGap> = Vec::new();
    for file_coverage in &export.files {
        if !is_crate_source(&file_coverage.filename) {
            continue;
        }
        let gap = summarize_file(file_coverage);
        total_covered = total_covered.saturating_add(gap.covered);
        total_lines = total_lines.saturating_add(gap.total);
        if !gap.uncovered.is_empty() {
            gaps.push(gap);
        }
    }
    let percent = propagate!(percentage(total_covered, total_lines));
    Ok(CoverageSummary {
        covered: total_covered,
        total: total_lines,
        percent,
        gaps,
    })
}

/// Reports whether a summary satisfies the required coverage percentage.
#[must_use]
pub fn meets_threshold(summary: &CoverageSummary, threshold: f64) -> bool {
    summary.percent >= threshold
}
