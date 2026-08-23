//! Loss functions for gradient boosting.
//!
//! One module per objective, each owning its base score and evaluation loss;
//! per-round gradients and hessians are computed inline in the training loop
//! (see `crate::training::train`), which dispatches on the configured
//! [`crate::training::Objective`].
//!
//! # Functions
//!
//! Binary log loss (classification):
//!
//! - [`binary_log_loss`] computes weighted mean binary cross-entropy
//! - [`binary_log_loss_initial_prediction`] computes weighted log-odds from labels
//! - [`sigmoid_array`] applies sigmoid to a slice of raw scores
//!
//! Squared error (regression):
//!
//! - [`squared_error_loss`] computes mean squared error over raw scores
//! - [`squared_error_initial_prediction`] computes the label mean

mod initial_prediction;
mod loss;
mod sigmoid_arr;
mod squared_error;
pub(crate) mod validation;

#[cfg(test)]
mod tests;

pub mod multiclass;

pub use initial_prediction::binary_log_loss_initial_prediction;
pub use loss::binary_log_loss;
pub use multiclass::{multiclass_initial_predictions, multiclass_log_loss};
pub use sigmoid_arr::sigmoid_array;
pub use squared_error::{squared_error_initial_prediction, squared_error_loss};
