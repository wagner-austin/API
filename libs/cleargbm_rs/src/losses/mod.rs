//! Loss functions for gradient boosting.
//!
//! Provides binary log loss (binary cross-entropy) with gradient and hessian
//! computation, plus initial prediction (log-odds). These are the core functions
//! that feed the tree building loop at each boosting round.
//!
//! # Functions
//!
//! - [`binary_log_loss`] computes mean binary cross-entropy
//! - [`binary_log_loss_gradients`] computes first derivatives (p - y)
//! - [`binary_log_loss_hessians`] computes second derivatives (p * (1-p))
//! - [`binary_log_loss_initial_prediction`] computes log-odds from labels
//! - [`sigmoid_array`] applies sigmoid to a slice of values

mod derivatives;
mod initial_prediction;
mod loss;
mod sigmoid_arr;
pub(crate) mod validation;

#[cfg(test)]
mod tests;

pub use derivatives::{binary_log_loss_gradients, binary_log_loss_hessians};
pub use initial_prediction::binary_log_loss_initial_prediction;
pub use loss::binary_log_loss;
pub use sigmoid_arr::sigmoid_array;
