//! Render a coverage summary into report lines.
//!
//! Rendering returns owned lines rather than writing to stdout, so the report
//! is asserted directly in tests and the binary stays the only place that
//! performs output.

use std::path::Path;

use crate::coverage_check::fs::FileSystem;
use crate::coverage_check::types::{CoverageCheckError, CoverageSummary, FileGap};

/// Width of the banner framing the report header.
const BANNER_WIDTH: usize = 70;

/// Reads one line of source for display alongside a coverage gap.
///
/// A coverage export naming a file or line that no longer exists means the
/// export is stale relative to the tree, which is a real defect: it is
/// reported rather than papered over with placeholder text.
///
/// # Errors
///
/// Returns [`CoverageCheckError::SourceFileMissing`] when the file is absent,
/// [`CoverageCheckError::SourceFileUnreadable`] when it cannot be read, or
/// [`CoverageCheckError::SourceLineOutOfRange`] when the line does not exist.
fn source_line(
    files: &dyn FileSystem,
    path: &Path,
    line_no: usize,
) -> Result<String, CoverageCheckError> {
    let display = path.display().to_string();
    if !files.exists(path) {
        return Err(CoverageCheckError::SourceFileMissing { path: display });
    }
    let text = match files.read_to_string(path) {
        Ok(contents) => contents,
        Err(reason) => {
            return Err(CoverageCheckError::SourceFileUnreadable {
                path: display,
                reason,
            })
        }
    };
    let lines: Vec<&str> = text.lines().collect();
    let index = match line_no.checked_sub(1_usize) {
        Some(value) => value,
        None => {
            return Err(CoverageCheckError::SourceLineOutOfRange {
                path: display,
                line_no,
                line_count: lines.len(),
            })
        }
    };
    match lines.get(index) {
        Some(line) => Ok(line.trim().to_owned()),
        None => Err(CoverageCheckError::SourceLineOutOfRange {
            path: display,
            line_no,
            line_count: lines.len(),
        }),
    }
}

/// Renders one file's uncovered lines with their source text.
///
/// # Errors
///
/// Returns [`CoverageCheckError::CountTooLarge`] when the file's line counts
/// cannot be converted for percentage arithmetic, or a source-file error when
/// the export is stale relative to the tree.
pub fn render_gap(
    files: &dyn FileSystem,
    gap: &FileGap,
) -> Result<Vec<String>, CoverageCheckError> {
    let percent = propagate!(crate::coverage_check::analysis::percentage(
        gap.covered,
        gap.total
    ));
    let path = Path::new(&gap.filename);
    let short_name = match path.file_name() {
        Some(name) => name.to_string_lossy().into_owned(),
        None => gap.filename.clone(),
    };
    let mut lines = vec![String::new(), format!("  {short_name} ({percent:.1}%):")];
    for line_no in &gap.uncovered {
        let text = propagate!(source_line(files, path, *line_no));
        lines.push(format!("    {line_no:4}: {text}"));
    }
    Ok(lines)
}

/// Renders the whole-run coverage report.
///
/// # Errors
///
/// Returns [`CoverageCheckError::CountTooLarge`] when a count cannot be
/// converted for percentage arithmetic.
pub fn render_summary(
    files: &dyn FileSystem,
    summary: &CoverageSummary,
    threshold: f64,
) -> Result<Vec<String>, CoverageCheckError> {
    let banner = "=".repeat(BANNER_WIDTH);
    let covered = summary.covered;
    let total = summary.total;
    let percent = summary.percent;
    let mut lines = vec![
        banner.clone(),
        "SEGMENT-BASED COVERAGE CHECK".to_owned(),
        banner,
        format!("Total: {covered}/{total} lines ({percent:.2}%)"),
        format!("Threshold: {threshold:.2}%"),
        String::new(),
    ];

    if summary.gaps.is_empty() {
        lines.push("All source lines are covered!".to_owned());
    } else {
        let gap_count = summary.gaps.len();
        lines.push(format!("FILES WITH UNCOVERED LINES ({gap_count}):"));
        for gap in &summary.gaps {
            lines.extend(propagate!(render_gap(files, gap)));
        }
    }

    lines.push(String::new());
    Ok(lines)
}

/// Renders the final pass or fail line.
#[must_use]
pub fn render_verdict(summary: &CoverageSummary, threshold: f64, passed: bool) -> String {
    let comparison = if passed { ">=" } else { "<" };
    let outcome = if passed { "PASS" } else { "FAIL" };
    let percent = summary.percent;
    format!("{outcome}: Coverage {percent:.2}% {comparison} {threshold:.2}%")
}
