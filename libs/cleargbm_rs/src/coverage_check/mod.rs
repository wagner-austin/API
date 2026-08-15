//! Segment-based coverage checking for this crate.
//!
//! `cargo llvm-cov` reports coverage per generic instantiation, which makes
//! monomorphised code look uncovered even when some instantiation executed it.
//! This module merges the LLVM *segments* down to one count per source line,
//! so a line counts as covered when any instantiation ran it, and fails the
//! build when the merged result falls below the required percentage.
//!
//! The `check_segment_coverage` binary is a thin entry point over
//! [`cli::run`]; all logic lives here so it is unit-testable.

/// Pure computation over a decoded export.
pub mod analysis;

/// Command-line parsing and orchestration.
pub mod cli;

/// Decoding the cargo-llvm-cov JSON export into typed structures.
pub mod decode;

/// Filesystem seam used by the checker.
pub mod fs;

/// Rendering a summary into report lines.
pub mod render;

/// Typed views over the export and the derived report.
pub mod types;

#[cfg(test)]
mod tests;
