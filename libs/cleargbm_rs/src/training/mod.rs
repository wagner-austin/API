//! Gradient boosting training pipeline.
//!
//! Orchestrates the full training loop: binning, boosting iterations,
//! early stopping, and model construction.
//!
//! # Overview
//!
//! - [`GradientBoostingConfig`] holds validated training hyperparameters
//! - [`train_gradient_boosting`] runs the full training loop
//! - [`GradientBoostingModel`] wraps the trained ensemble for prediction

pub(crate) mod config;
pub(crate) mod early_stopping;
pub(crate) mod importance;
pub(crate) mod model;
pub(crate) mod rng;
pub(crate) mod serde_impl;
pub(crate) mod subsampling;
pub(crate) mod train;
pub(crate) mod validation;

#[cfg(test)]
mod tests;

pub use config::{GradientBoostingConfig, GradientBoostingConfigParams};
pub use importance::feature_importances;
pub use model::GradientBoostingModel;
pub use train::train_gradient_boosting;
