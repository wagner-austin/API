//! Gradient boosting training pipeline.
//!
//! Orchestrates the full training loop: binning, boosting iterations,
//! early stopping, and model construction.
//!
//! # Overview
//!
//! - [`GradientBoostingConfig`] holds validated training hyperparameters
//! - [`Parallelism`] selects the worker-thread count for a run
//! - [`train_gradient_boosting`] runs the full training loop
//! - [`GradientBoostingModel`] wraps the trained ensemble for prediction

pub(crate) mod config;
mod config_rules;
mod continue_training;
pub(crate) mod early_stopping;
mod goss;
pub(crate) mod importance;
pub(crate) mod labels;
pub(crate) mod model;
pub(crate) mod objective_enums;
pub(crate) mod parallelism;
mod quantize;
pub(crate) mod rng;
pub(crate) mod serde_impl;
pub(crate) mod setup;
mod single_score_rounds;
pub(crate) mod subsampling;
mod train;
mod train_multiclass;
mod train_ranking;
pub(crate) mod validation;

#[cfg(test)]
mod tests;

pub use config::{GradientBoostingConfig, GradientBoostingConfigParams, GrowthStrategy, Objective};
pub use continue_training::continue_gradient_boosting;
pub use importance::feature_importances;
pub use labels::{TrainingLabels, ValidationData};
pub use model::GradientBoostingModel;
pub use parallelism::Parallelism;
pub use train::{train_gradient_boosting, TrainingRuntime};
pub use train_ranking::{
    train_gradient_boosting_ranking, RankingTrainingData, RankingValidationData,
};
