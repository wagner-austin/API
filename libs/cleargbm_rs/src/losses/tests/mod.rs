//! Tests for the losses module.
//!
//! Tests are organized into the following sub-modules:
//! - `validation_tests`: Tests for label/length validation and usize-to-f64 conversion
//! - `loss_tests`: Tests for binary cross-entropy (log loss) computation
//! - `gradient_tests`: Tests for gradient (first derivative) computation
//! - `hessian_tests`: Tests for hessian (second derivative) computation
//! - `initial_prediction_tests`: Tests for log-odds initial prediction
//! - `sigmoid_array_tests`: Tests for vectorized sigmoid

mod gradient_tests;
mod hessian_tests;
mod initial_prediction_tests;
mod loss_tests;
mod sigmoid_array_tests;
mod validation_tests;
