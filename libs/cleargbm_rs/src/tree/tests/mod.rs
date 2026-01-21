//! Tests for the tree module.
//!
//! Tests are organized into the following sub-modules:
//! - `builder_tests`: Tests for tree builder functions (compute_leaf_value, split_samples, etc.)
//! - `config_tests`: Tests for TreeBuildConfig and Tree struct creation and accessors
//! - `error_tests`: Tests for error conditions in finalize_nodes and tree building
//! - `proptest_tests`: Property-based tests using proptest
//! - `serde_tests`: Serialization/deserialization tests and error path coverage

mod builder_tests;
mod config_tests;
mod error_tests;
mod proptest_tests;
mod serde_tests;
