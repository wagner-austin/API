//! Integration tests for ClearGBM Rust core.
//!
//! Each submodule tests a distinct concern:
//! - `math_tests` — leaf value, split gain, histogram, best-split formulas
//! - `tree_building_tests` — binary classification, squared error, monotonic, regularization
//! - `edge_case_tests` — degenerate inputs, accessor coverage, histogram subtraction
//! - `serde_tests` — serialization round-trips
//! - `prediction_tests` — single/batch/ensemble predict, probability bounds, sigmoid stability
//! - `loss_tests` — gradient-hessian consistency, initial prediction, loss→tree integration
//! - `training_tests` — full train→predict pipeline, convergence, early stopping, determinism

#[path = "integration_tests/edge_case_tests.rs"]
mod edge_case_tests;
#[path = "integration_tests/loss_tests.rs"]
mod loss_tests;
#[path = "integration_tests/math_tests.rs"]
mod math_tests;
#[path = "integration_tests/prediction_tests.rs"]
mod prediction_tests;
#[path = "integration_tests/serde_tests.rs"]
mod serde_tests;
#[path = "integration_tests/training_tests.rs"]
mod training_tests;
#[path = "integration_tests/tree_building_tests.rs"]
mod tree_building_tests;

/// Tolerance for floating-point comparisons across all integration tests.
const EPSILON: f64 = 1e-10_f64;
