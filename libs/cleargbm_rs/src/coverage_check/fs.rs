//! Filesystem seam for the coverage checker.
//!
//! Every filesystem read the checker performs goes through [`FileSystem`], so
//! tests drive the checker against an in-memory tree without touching disk and
//! without patching anything. Production wires [`RealFileSystem`] at the entry
//! point; there is no `if testing` branch in the logic.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Read-only filesystem operations the checker depends on.
pub trait FileSystem {
    /// Reports whether a path exists.
    fn exists(&self, path: &Path) -> bool;

    /// Reads a UTF-8 text file in full.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when the file cannot be read or is not
    /// valid UTF-8.
    fn read_to_string(&self, path: &Path) -> Result<String, String>;
}

/// [`FileSystem`] backed by the real filesystem.
#[derive(Debug, Clone, Copy, Default)]
pub struct RealFileSystem;

impl FileSystem for RealFileSystem {
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn read_to_string(&self, path: &Path) -> Result<String, String> {
        match std::fs::read_to_string(path) {
            Ok(contents) => Ok(contents),
            Err(err) => Err(err.to_string()),
        }
    }
}

/// [`FileSystem`] backed by an in-memory map, for tests.
#[derive(Debug, Clone, Default)]
pub struct MemoryFileSystem {
    /// Files present in the tree, keyed by path.
    entries: BTreeMap<PathBuf, String>,
    /// Paths that exist but fail to read, keyed to their failure reason.
    unreadable: BTreeMap<PathBuf, String>,
}

impl MemoryFileSystem {
    /// Creates an empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            unreadable: BTreeMap::new(),
        }
    }

    /// Adds a readable file to the tree.
    #[must_use]
    pub fn with_file(mut self, path: &Path, contents: &str) -> Self {
        self.entries.insert(path.to_path_buf(), contents.to_owned());
        self
    }

    /// Adds a file that exists but fails to read.
    #[must_use]
    pub fn with_unreadable(mut self, path: &Path, reason: &str) -> Self {
        self.unreadable
            .insert(path.to_path_buf(), reason.to_owned());
        self
    }
}

impl FileSystem for MemoryFileSystem {
    fn exists(&self, path: &Path) -> bool {
        self.entries.contains_key(path) || self.unreadable.contains_key(path)
    }

    fn read_to_string(&self, path: &Path) -> Result<String, String> {
        match self.unreadable.get(path) {
            Some(reason) => Err(reason.clone()),
            None => match self.entries.get(path) {
                Some(contents) => Ok(contents.clone()),
                None => Err(format!("no such file: {}", path.display())),
            },
        }
    }
}
