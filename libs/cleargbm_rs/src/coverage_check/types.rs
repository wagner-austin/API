//! Typed views over the cargo-llvm-cov JSON export and the report derived from it.
//!
//! The decoder produces these types at the process boundary, so nothing
//! downstream ever indexes into a dynamic JSON value.

use std::path::PathBuf;

use thiserror::Error;

/// One LLVM coverage segment.
///
/// cargo-llvm-cov emits each segment as a fixed six-element JSON array; this
/// struct names those positions so downstream code never indexes by number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// 1-indexed source line the segment begins on.
    pub line: usize,
    /// 1-indexed source column the segment begins on.
    pub column: usize,
    /// Number of times the segment executed.
    pub count: u64,
    /// Whether `count` carries meaningful execution information.
    pub has_count: bool,
    /// Whether the segment begins a coverage region.
    pub is_region_entry: bool,
    /// Whether the segment spans a gap between regions.
    pub is_gap_region: bool,
}

/// Coverage segments recorded for a single source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileCoverage {
    /// Absolute path of the source file, as recorded in the export.
    pub filename: String,
    /// Every segment reported for the file, in export order.
    pub segments: Vec<Segment>,
}

/// The subset of the cargo-llvm-cov export this checker consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageExport {
    /// One record per source file present in the export.
    pub files: Vec<FileCoverage>,
}

/// Per-file coverage outcome for a file that has uncovered lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileGap {
    /// Absolute path of the source file.
    pub filename: String,
    /// Number of lines executed at least once.
    pub covered: usize,
    /// Number of lines carrying coverage information.
    pub total: usize,
    /// Ascending line numbers that were never executed.
    pub uncovered: Vec<usize>,
}

/// Whole-run coverage outcome across every analysed source file.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageSummary {
    /// Total lines executed at least once.
    pub covered: usize,
    /// Total lines carrying coverage information.
    pub total: usize,
    /// `covered` as a percentage of `total`.
    pub percent: f64,
    /// One entry per file that has uncovered lines, in export order.
    pub gaps: Vec<FileGap>,
}

/// Command-line options accepted by the checker.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Path to the cargo-llvm-cov JSON export.
    pub json_path: PathBuf,
    /// Minimum percentage required to pass.
    pub threshold: f64,
}

/// Every way the segment-coverage checker can fail.
///
/// Each variant identifies one specific defect so a CI log names the actual
/// problem rather than a single generic parse failure.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CoverageCheckError {
    /// The coverage export file does not exist.
    #[error("coverage export not found at {path}; generate it with: cargo llvm-cov --all-features --json --output-path {path}")]
    ExportNotFound {
        /// Path that was checked.
        path: String,
    },

    /// The coverage export file could not be read.
    #[error("coverage export at {path} could not be read: {reason}")]
    ExportUnreadable {
        /// Path that was read.
        path: String,
        /// Underlying I/O failure.
        reason: String,
    },

    /// The coverage export is not valid JSON.
    #[error("coverage export at {path} is not valid JSON: {reason}")]
    ExportNotJson {
        /// Path that was parsed.
        path: String,
        /// Underlying parse failure.
        reason: String,
    },

    /// A JSON value had the wrong type.
    #[error("field '{field}' must be {expected}, got {got}")]
    FieldWrongType {
        /// Name or position of the offending field.
        field: String,
        /// JSON type that was required.
        expected: String,
        /// JSON type that was present.
        got: String,
    },

    /// A required JSON field was absent.
    #[error("missing required field '{field}'")]
    FieldMissing {
        /// Name of the absent field.
        field: String,
    },

    /// A segment array did not hold exactly six elements.
    #[error("coverage segment must hold {expected} elements, got {got}")]
    SegmentArityInvalid {
        /// Number of elements required.
        expected: usize,
        /// Number of elements present.
        got: usize,
    },

    /// A numeric JSON value did not fit the target integer type.
    #[error("field '{field}' value {value} is out of range for {target}")]
    NumberOutOfRange {
        /// Name or position of the offending field.
        field: String,
        /// Value that was rejected.
        value: i64,
        /// Target type it had to fit.
        target: String,
    },

    /// The export carried no `data` entries.
    #[error("coverage export carries no 'data' entries")]
    ExportDataEmpty,

    /// A source file named by the export is no longer on disk.
    #[error("source file {path} named by the coverage export does not exist; the export is stale")]
    SourceFileMissing {
        /// Path that was checked.
        path: String,
    },

    /// A source file named by the export could not be read.
    #[error("source file {path} could not be read: {reason}")]
    SourceFileUnreadable {
        /// Path that was read.
        path: String,
        /// Underlying I/O failure.
        reason: String,
    },

    /// The export named a line that does not exist in the source file.
    #[error("source file {path} has {line_count} lines but the export names line {line_no}; the export is stale")]
    SourceLineOutOfRange {
        /// Path of the source file.
        path: String,
        /// Line number named by the export.
        line_no: usize,
        /// Number of lines the file actually has.
        line_count: usize,
    },

    /// A line count exceeded what percentage arithmetic can represent exactly.
    #[error("line count {value} is too large to convert for percentage arithmetic")]
    CountTooLarge {
        /// Count that could not be converted.
        value: usize,
    },

    /// A command-line flag was not recognised.
    #[error("unrecognised argument '{token}'")]
    CliFlagUnknown {
        /// Token that was rejected.
        token: String,
    },

    /// A command-line flag was given without its value.
    #[error("flag {flag} requires a value")]
    CliFlagMissingValue {
        /// Flag that was missing a value.
        flag: String,
    },

    /// A threshold argument was not plain decimal notation.
    #[error("threshold must be a decimal number, got '{token}'")]
    ThresholdNotANumber {
        /// Token that was rejected.
        token: String,
    },

    /// A threshold argument fell outside 0-100.
    #[error("threshold must lie between 0 and 100, got {value}")]
    ThresholdOutOfRange {
        /// Value that was rejected.
        value: f64,
    },
}
