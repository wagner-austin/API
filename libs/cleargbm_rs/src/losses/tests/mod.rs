//! Tests for the losses module.
//!
//! Tests are organized into the following sub-modules:
//! - `validation_tests`: Tests for label/length validation and usize-to-f64 conversion
//! - `loss_tests`: Tests for binary cross-entropy (log loss) computation
//! - `initial_prediction_tests`: Tests for log-odds initial prediction
//! - `squared_error_tests`: Tests for the regression base score and MSE
//! - `sigmoid_array_tests`: Tests for vectorized sigmoid

mod initial_prediction_tests;
mod lambdarank_tests;
mod loss_tests;
mod multiclass_tests;
mod sigmoid_array_tests;
mod squared_error_tests;
mod validation_tests;
