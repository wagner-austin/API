//! Tests for the PyO3 bindings module.
//!
//! Tests error conversion mappings, array helper functions, and Python-callable
//! binding functions through the PyO3 runtime.

mod array_helpers_tests;
mod error_conversion_tests;
mod helpers;
mod model_persistence_tests;
mod module_init_tests;
mod predict_fns_tests;
mod training_config_key_tests;
mod training_fns_tests;
mod training_options_tests;
mod training_regression_entry_tests;
