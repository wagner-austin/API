//! Trained gradient boosting model for inference.
//!
//! Wraps the trained ensemble of trees with the training configuration
//! for making predictions on new data.

use crate::error::ClearGbmError;
use crate::predict::{predict_ensemble, predict_proba, PredictEnsembleConfig};
use crate::tree::Tree;

use super::config::GradientBoostingConfig;

/// A trained gradient boosting model.
///
/// Contains the ensemble of trees, training metadata, and configuration
/// needed for inference. Constructed by [`super::train_gradient_boosting`].
#[derive(Debug, Clone, PartialEq)]
pub struct GradientBoostingModel {
    /// Trained decision trees.
    trees: Vec<Tree>,
    /// Initial prediction before tree contributions (log-odds of class prevalence).
    base_prediction: f64,
    /// Shrinkage factor applied to each tree's predictions.
    learning_rate: f64,
    /// Feature names from training data.
    feature_names: Vec<String>,
    /// Number of classes (always 2 for binary classification).
    n_classes: usize,
    /// Training configuration used to build this model.
    config: GradientBoostingConfig,
}

impl GradientBoostingModel {
    /// Creates a new trained model.
    ///
    /// # Args
    ///
    /// * `trees` - Trained decision trees.
    /// * `base_prediction` - Initial prediction (log-odds).
    /// * `learning_rate` - Shrinkage factor in (0.0, 1.0].
    /// * `feature_names` - Feature names from training.
    /// * `n_classes` - Number of classes (always 2).
    /// * `config` - Training configuration.
    pub(crate) fn new(
        trees: Vec<Tree>,
        base_prediction: f64,
        learning_rate: f64,
        feature_names: Vec<String>,
        n_classes: usize,
        config: GradientBoostingConfig,
    ) -> Self {
        Self {
            trees,
            base_prediction,
            learning_rate,
            feature_names,
            n_classes,
            config,
        }
    }

    /// Returns the trained trees.
    #[must_use]
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Returns the base prediction (log-odds of class prevalence).
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

    /// Returns the number of classes (always 2 for binary classification).
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
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

    /// Predicts raw log-odds scores for a batch of samples.
    ///
    /// Computes: `raw[i] = base_prediction + learning_rate * sum(tree_j.predict(x[i]))`
    ///
    /// # Args
    ///
    /// * `x` - Feature matrix `[n_samples][n_features]`.
    ///
    /// # Returns
    ///
    /// Raw predictions (log-odds), one per sample.
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
    /// Any error from [`predict_raw`](Self::predict_raw).
    pub fn predict_proba(&self, x: &[&[f64]]) -> Result<Vec<(f64, f64)>, ClearGbmError> {
        let raw_preds = match self.predict_raw(x) {
            Ok(p) => p,
            Err(e) => return Err(e),
        };
        Ok(predict_proba(&raw_preds))
    }
}
