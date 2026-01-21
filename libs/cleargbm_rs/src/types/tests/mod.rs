//! Tests for the types module.
//!
//! Tests are organized into the following sub-modules:
//! - `tree_node_tests`: Tests for TreeNode type
//! - `histogram_buffer_tests`: Tests for HistogramBuffer type
//! - `split_config_tests`: Tests for SplitConfig type
//! - `*_serde_tests`: Serde error path tests for each type
//! - `type_mismatch_tests`: Type mismatch and deserializer tests
//! - `serializer_tests`: Serialization error path tests

mod histogram_buffer_serde_tests;
mod histogram_buffer_tests;
mod serializer_tests;
mod split_config_serde_tests;
mod split_config_tests;
mod tree_node_config_serde_tests;
mod tree_node_serde_tests;
mod tree_node_tests;
mod type_mismatch_tests;
