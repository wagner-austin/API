//! Tests for the tree module.
//!
//! Tests are organized into the following sub-modules:
//! - `builder_tests`: Tests for tree builder functions (compute_leaf_value, split_samples, etc.)
//! - `builder_build_tests`: Tests for build_tree behavior on well-formed inputs
//! - `builder_build_edge_tests`: build_tree malformed inputs and hook-injected failures
//! - `config_tests`: Tests for TreeBuildConfig and Tree struct creation and accessors
//! - `error_tests`: Tests for error conditions in finalize_nodes and the
//!   internal histogram/split functions
//! - `error_hook_tests`: Hook-based error injection through build_tree
//! - `feature_subsample_tests`: Tests for the per-tree and per-node feature
//!   mask derivations and their composition
//! - `leafwise_helpers`: Shared fixtures for the leaf-wise tests
//! - `leafwise_tests`: Tests for best-first growth, including its equivalence
//!   with depth-wise growth when the leaf budget never binds
//! - `leafwise_error_tests`: Validation and error propagation for leaf-wise growth
//! - `proptest_tests`: Property-based tests using proptest
//! - `serde_tests`: Serialization/deserialization tests and error path coverage

mod builder_build_edge_tests;
mod builder_build_tests;
mod builder_tests;
mod config_tests;
mod error_hook_tests;
mod error_tests;
mod feature_subsample_tests;
mod leafwise_error_tests;
mod leafwise_helpers;
mod leafwise_tests;
mod proptest_tests;
mod serde_tests;
