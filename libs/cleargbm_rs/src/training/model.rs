//! Trained gradient boosting model for inference.
//!
//! Wraps the trained ensemble of trees with the training configuration
//! for making predictions on new data.

use crate::error::ClearGbmError;
use crate::predict::{predict_ensemble, predict_proba, PredictEnsembleConfig};
use crate::tree::Tree;

use super::config::{GradientBoostingConfig, Objective};

/// A trained gradient boosting model.
///
/// Contains the ensemble of trees, training metadata, and configuration
/// needed for inference. Constructed by [`super::train_gradient_boosting`].
/// The embedded config's `objective` decides how raw scores read:
/// log-odds under `binary_log_loss` (probabilities via
/// [`Self::predict_proba`]), predictions directly under `squared_error`.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientBoostingModel {
    /// Trained decision trees.
    trees: Vec<Tree>,
    /// Initial prediction before tree contributions (the objective's base
    /// score: weighted log-odds for binary, label mean for regression).
    base_prediction: f64,
    /// Shrinkage factor applied to each tree's predictions.
    learning_rate: f64,
    /// Feature names from training data.
    feature_names: Vec<String>,
    /// Training configuration used to build this model.
    config: GradientBoostingConfig,
}

impl GradientBoostingModel {
    /// Creates a new trained model.
    ///
    /// # Args
    ///
    /// * `trees` - Trained decision trees.
    /// * `base_prediction` - The objective's base score.
    /// * `learning_rate` - Shrinkage factor in (0.0, 1.0].
    /// * `feature_names` - Feature names from training.
    /// * `config` - Training configuration.
    pub(crate) fn new(
        trees: Vec<Tree>,
        base_prediction: f64,
        learning_rate: f64,
        feature_names: Vec<String>,
        config: GradientBoostingConfig,
    ) -> Self {
        Self {
            trees,
            base_prediction,
            learning_rate,
            feature_names,
            config,
        }
    }

    /// Returns the trained trees.
    #[must_use]
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Returns the base prediction (the objective's base score).
    #[must_use]
    pub fn base_prediction(&self) -> f64 {
        self.base_prediction
    }

    /// Returns the learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the feature names.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Returns the training configuration.
    #[must_use]
    pub fn config(&self) -> &GradientBoostingConfig {
        &self.config
    }

    /// Returns the number of trained trees.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Predicts raw scores for a batch of samples.
    ///
    /// Computes: `raw[i] = base_prediction + learning_rate * sum(tree_j.predict(x[i]))`
    ///
    /// Under `binary_log_loss` the raw score is a log-odds; under
    /// `squared_error` it IS the prediction — regression inference is this
    /// function.
    ///
    /// # Args
    ///
    /// * `x` - Feature matrix `[n_samples][n_features]`.
    ///
    /// # Returns
    ///
    /// Raw predictions, one per sample.
    ///
    /// # Errors
    ///
    /// * `ClearGbmError::EmptyInput` if `x` is empty or trees are empty.
    /// * `ClearGbmError::InvalidParameter` on invalid learning rate.
    /// * Any prediction error from tree traversal.
    pub fn predict_raw(&self, x: &[&[f64]]) -> Result<Vec<f64>, ClearGbmError> {
        let ensemble_config =
            match PredictEnsembleConfig::new(self.base_prediction, self.learning_rate) {
                Ok(c) => c,
                Err(e) => return Err(e),
            };
        match predict_ensemble(&self.trees, x, &ensemble_config) {
            Ok(preds) => Ok(preds),
            Err(e) => Err(e),
        }
    }

    /// Predicts class probabilities for a batch of samples.
    ///
    /// Computes raw log-odds via [`predict_raw`](Self::predict_raw), then applies
    /// sigmoid to produce `(probability_class_0, probability_class_1)` pairs.
    /// Only meaningful when the model was trained under `binary_log_loss`;
    /// a squared-error model's raw scores are predictions, not log-odds, so
    /// asking for probabilities is rejected rather than silently squashed
    /// through a sigmoid.
    ///
    /// # Args
    ///
    /// * `x` - Feature matrix `[n_samples][n_features]`.
    ///
    /// # Returns
    ///
    /// Vector of `(prob_class_0, prob_class_1)` tuples, one per sample.
    ///
    /// # Errors
    ///
    /// * `ClearGbmError::InvalidParameter` if the model's objective is
    ///   `squared_error`.
    /// * Any error from [`predict_raw`](Self::predict_raw).
    pub fn predict_proba(&self, x: &[&[f64]]) -> Result<Vec<(f64, f64)>, ClearGbmError> {
        match self.config.objective() {
            Objective::BinaryLogLoss => {}
            Objective::SquaredError => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "objective".to_string(),
                    reason: "predict_proba requires objective \"binary_log_loss\"; a \
                             \"squared_error\" model's predictions are its raw scores, so use \
                             predict_raw"
                        .to_string(),
                })
            }
        }
        let raw_preds = match self.predict_raw(x) {
            Ok(p) => p,
            Err(e) => return Err(e),
        };
        Ok(predict_proba(&raw_preds))
    }
}
