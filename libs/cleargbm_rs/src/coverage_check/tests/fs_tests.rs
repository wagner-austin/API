//! Tests for the filesystem seam.

use std::path::{Path, PathBuf};

use crate::coverage_check::fs::{FileSystem, MemoryFileSystem, RealFileSystem};
use crate::coverage_check::types::CoverageCheckError;

#[test]
fn test_memory_filesystem_reports_missing_paths() -> Result<(), CoverageCheckError> {
    let files = MemoryFileSystem::new();
    assert!(!files.exists(Path::new("absent.json")));
    Ok(())
}

#[test]
fn test_memory_filesystem_reads_stored_file() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("stored.json");
    let files = MemoryFileSystem::new().with_file(&path, "contents");
    assert!(files.exists(&path));
    match files.read_to_string(&path) {
        Ok(text) => assert_eq!(text, "contents"),
        Err(reason) => {
            return Err(CoverageCheckError::ExportUnreadable {
                path: "stored.json".to_owned(),
                reason,
            })
        }
    }
    Ok(())
}

#[test]
fn test_memory_filesystem_reports_read_failure() -> Result<(), CoverageCheckError> {
    let path = PathBuf::from("broken.json");
    let files = MemoryFileSystem::new().with_unreadable(&path, "permission denied");
    assert!(files.exists(&path));
    match files.read_to_string(&path) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(reason) => {
            assert_eq!(reason, "permission denied");
            Ok(())
        }
    }
}

#[test]
fn test_memory_filesystem_read_of_absent_file_fails() -> Result<(), CoverageCheckError> {
    let files = MemoryFileSystem::new();
    match files.read_to_string(Path::new("absent.json")) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(reason) => {
            assert!(reason.contains("no such file"));
            Ok(())
        }
    }
}

#[test]
fn test_real_filesystem_reports_missing_path() -> Result<(), CoverageCheckError> {
    let files = RealFileSystem;
    assert!(!files.exists(Path::new("definitely-not-a-real-file-xyz.json")));
    Ok(())
}

#[test]
fn test_real_filesystem_reads_manifest() -> Result<(), CoverageCheckError> {
    let files = RealFileSystem;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(files.exists(&manifest));
    match files.read_to_string(&manifest) {
        Ok(text) => assert!(text.contains("cleargbm_rs")),
        Err(reason) => {
            return Err(CoverageCheckError::ExportUnreadable {
                path: "Cargo.toml".to_owned(),
                reason,
            })
        }
    }
    Ok(())
}

#[test]
fn test_real_filesystem_read_failure_is_reported() -> Result<(), CoverageCheckError> {
    let files = RealFileSystem;
    match files.read_to_string(Path::new("definitely-not-a-real-file-xyz.json")) {
        Ok(_) => Err(CoverageCheckError::ExportDataEmpty),
        Err(reason) => {
            assert!(!reason.is_empty());
            Ok(())
        }
    }
}
