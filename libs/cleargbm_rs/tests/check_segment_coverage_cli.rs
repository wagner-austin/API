//! End-to-end tests for the `check_segment_coverage` binary.
//!
//! These run the compiled binary as a subprocess, which is what exercises its
//! `main` entry point. cargo-llvm-cov collects coverage from spawned
//! processes, so the entry point is measured like any other code rather than
//! being excluded from the coverage requirement.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Builds an export document covering one file at the given execution counts.
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

/// Writes a fixture into a temporary `src` directory, returning its path.
///
/// The directory is named `src` because the checker only counts files whose
/// path lies under a `src` directory, matching how it filters dependency code
/// out of a real export.
fn write_fixture(name: &str, contents: &str) -> Result<PathBuf, String> {
    let mut dir = std::env::temp_dir();
    dir.push("cleargbm_rs_cov_fixtures");
    dir.push("src");
    if let Err(err) = std::fs::create_dir_all(&dir) {
        return Err(format!("failed to create fixture dir: {err}"));
    }
    let path = dir.join(name);
    match std::fs::write(&path, contents) {
        Ok(()) => Ok(path),
        Err(err) => Err(format!("failed to write fixture {name}: {err}")),
    }
}

/// Renders a path for embedding in a JSON string literal.
fn json_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

/// Runs the checker binary with the given arguments.
fn run_checker(args: &[String]) -> Result<Output, String> {
    let exe = env!("CARGO_BIN_EXE_check_segment_coverage");
    match Command::new(exe).args(args).output() {
        Ok(output) => Ok(output),
        Err(err) => Err(format!("failed to spawn checker: {err}")),
    }
}

#[test]
fn test_binary_passes_on_full_coverage() -> Result<(), String> {
    let export = export_document("src/covered.rs", &[1_u64, 2_u64]);
    let path = match write_fixture("pass.json", &export) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let args = vec![
        "--json".to_owned(),
        path.display().to_string(),
        "--threshold".to_owned(),
        "100".to_owned(),
    ];
    // A fully covered export has no gaps, so no source file is read.
    let output = match run_checker(&args) {
        Ok(o) => o,
        Err(e) => return Err(e),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("PASS:") {
        return Err(format!("expected PASS in output, got: {stdout}"));
    }
    if output.status.code() != Some(0_i32) {
        return Err(format!("expected exit 0, got {:?}", output.status.code()));
    }
    Ok(())
}

#[test]
fn test_binary_fails_when_coverage_falls_short() -> Result<(), String> {
    // The report reads the source file named by the export, so it must exist.
    let source_path = match write_fixture("partial.rs", "fn a() {}\nfn b() {}\n") {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let export = export_document(&json_path(&source_path), &[1_u64, 0_u64]);
    let export_path = match write_fixture("fail.json", &export) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let args = vec!["--json".to_owned(), export_path.display().to_string()];
    let output = match run_checker(&args) {
        Ok(o) => o,
        Err(e) => return Err(e),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("FAIL:") {
        return Err(format!("expected FAIL in output, got: {stdout}"));
    }
    if output.status.code() != Some(1_i32) {
        return Err(format!("expected exit 1, got {:?}", output.status.code()));
    }
    Ok(())
}

#[test]
fn test_binary_reports_unknown_flag() -> Result<(), String> {
    let args = vec!["--nope".to_owned()];
    let output = match run_checker(&args) {
        Ok(o) => o,
        Err(e) => return Err(e),
    };
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains("unrecognised argument") {
        return Err(format!("expected flag error, got: {stderr}"));
    }
    if output.status.code() != Some(1_i32) {
        return Err(format!("expected exit 1, got {:?}", output.status.code()));
    }
    Ok(())
}

#[test]
fn test_binary_reports_missing_export() -> Result<(), String> {
    let args = vec![
        "--json".to_owned(),
        "definitely-not-a-real-export-xyz.json".to_owned(),
    ];
    let output = match run_checker(&args) {
        Ok(o) => o,
        Err(e) => return Err(e),
    };
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains("coverage export not found") {
        return Err(format!("expected missing-export error, got: {stderr}"));
    }
    if output.status.code() != Some(1_i32) {
        return Err(format!("expected exit 1, got {:?}", output.status.code()));
    }
    Ok(())
}
