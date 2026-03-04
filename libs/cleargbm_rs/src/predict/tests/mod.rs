//! Tests for the predict module.
//!
//! Tests are organized into the following sub-modules:
//! - `sigmoid_tests`: Tests for the sigmoid function
//! - `single_tests`: Tests for single-sample tree prediction
//! - `batch_tests`: Tests for batch tree prediction
//! - `ensemble_tests`: Tests for ensemble prediction and probability conversion
//! - `error_tests`: Tests for error conditions and malformed trees

mod batch_tests;
mod ensemble_tests;
mod error_tests;
mod sigmoid_tests;
mod single_tests;
