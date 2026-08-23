//! Prediction and inference for gradient boosted trees.
//!
//! Provides functions for traversing decision trees to make predictions
//! on new data. Supports single-sample, batch, and ensemble prediction,
//! as well as probability conversion via sigmoid.
//!
//! # Overview
//!
//! - [`predict_single`] traverses a single tree for one sample
//! - [`predict_tree`] batches single-tree prediction across samples
//! - [`predict_ensemble`] aggregates predictions from multiple trees
//! - [`predict_proba`] converts raw log-odds to class probabilities
//! - [`sigmoid`] applies the logistic function with numerical stability

use crate::error::ClearGbmError;
use crate::tree::Tree;

#[cfg(test)]
mod tests;

/// Minimum value for sigmoid input clipping.
const SIGMOID_CLIP_MIN: f64 = -500.0_f64;

/// Maximum value for sigmoid input clipping.
const SIGMOID_CLIP_MAX: f64 = 500.0_f64;

/// Configuration for ensemble prediction.
///
/// Groups the parameters needed for [`predict_ensemble`] to avoid
/// too many function arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictEnsembleConfig {
    /// Initial prediction before any tree contributions (e.g., log-odds of class prevalence).
    base_prediction: f64,

    /// Shrinkage factor applied to each tree's predictions.
    learning_rate: f64,
}

impl PredictEnsembleConfig {
    /// Creates a new ensemble prediction configuration.
    ///
    /// # Args
    ///
    /// * `base_prediction` - Initial prediction before any tree contributions.
    /// * `learning_rate` - Shrinkage factor in `(0.0, 1.0]` applied to each tree's predictions.
    ///
    /// # Returns
    ///
    /// A validated `PredictEnsembleConfig`.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if `learning_rate` is not in `(0.0, 1.0]`.
    pub fn new(base_prediction: f64, learning_rate: f64) -> Result<Self, ClearGbmError> {
        if learning_rate <= 0.0_f64 || learning_rate > 1.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be in (0.0, 1.0]".to_string(),
            });
        }
        Ok(Self {
            base_prediction,
            learning_rate,
        })
    }

    /// Returns the base prediction.
    #[must_use]
    pub const fn base_prediction(&self) -> f64 {
        self.base_prediction
    }

    /// Returns the learning rate.
    #[must_use]
    pub const fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// Computes the sigmoid (logistic) function with numerical stability.
///
/// Clips the input to `[-500.0, 500.0]` to prevent overflow in the
/// exponential computation, matching the Python `cleargbm` implementation.
///
/// # Args
///
/// * `x` - Input value (typically log-odds).
///
/// # Returns
///
/// A probability in the range `(0.0, 1.0)`.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    let x_clipped = x.clamp(SIGMOID_CLIP_MIN, SIGMOID_CLIP_MAX);
    1.0_f64 / (1.0_f64 + (-x_clipped).exp())
}

/// Predicts the leaf value for a single sample by traversing a decision tree.
///
/// Starts at the root node and follows the tree structure until reaching a leaf:
/// - If the feature value is NaN, follows the node's `nan_goes_left` direction.
/// - If the feature value is less than or equal to the threshold, goes left.
/// - Otherwise, goes right.
///
/// Includes a cycle guard that limits traversal to `tree.n_nodes()` iterations
/// to protect against malformed trees.
///
/// # Args
///
/// * `tree` - The decision tree to traverse.
/// * `features` - Feature values for a single sample.
///
/// # Returns
///
/// The leaf prediction value for this sample.
///
/// # Errors
///
/// * `ClearGbmError::NodeNotFound` - If the tree is empty or a child node does not exist.
/// * `ClearGbmError::FeatureIndexOutOfBounds` - If an internal node references a feature
///   index beyond the length of `features`.
/// * `ClearGbmError::TreeConstructionFailed` - If an internal node is missing its threshold
///   or child pointers, or if traversal exceeds the maximum iteration count.
pub fn predict_single(tree: &Tree, features: &[f64]) -> Result<f64, ClearGbmError> {
    let n_features = features.len();
    let max_iterations = tree.n_nodes();

    let mut current = match tree.root() {
        Ok(node) => node,
        Err(e) => return Err(e),
    };

    let mut iteration = 0_usize;

    loop {
        if current.is_leaf() {
            return Ok(current.value());
        }

        if iteration >= max_iterations {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "traversal exceeded maximum iterations (possible cycle)".to_string(),
            });
        }
        iteration += 1_usize;

        let feature_index = match current.feature_index() {
            Some(idx) => idx,
            None => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("internal node {} missing feature_index", current.node_id()),
                })
            }
        };

        if feature_index >= n_features {
            return Err(ClearGbmError::FeatureIndexOutOfBounds {
                index: feature_index,
                n_features,
            });
        }

        let left_child_id = match current.left_child() {
            Some(id) => id,
            None => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("internal node {} missing left_child", current.node_id()),
                })
            }
        };

        let right_child_id = match current.right_child() {
            Some(id) => id,
            None => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("internal node {} missing right_child", current.node_id()),
                })
            }
        };

        let feature_value = features[feature_index];
        let next_id = if feature_value.is_nan() {
            if current.nan_goes_left() {
                left_child_id
            } else {
                right_child_id
            }
        } else if let Some(categories) = current.categories_goes_left() {
            // Categorical set-split: left exactly when the value matches a
            // left-routed code. `+ 0.0` normalizes -0.0 to 0.0, matching the
            // normalization binning applied when the codes were recorded;
            // any value that is not a recorded code — including unseen
            // categories and non-integer values — routes right.
            let normalized = feature_value + 0.0_f64;
            let is_member = categories
                .binary_search_by(|code| code.total_cmp(&normalized))
                .is_ok();
            if is_member {
                left_child_id
            } else {
                right_child_id
            }
        } else {
            let threshold = match current.threshold() {
                Some(t) => t,
                None => {
                    return Err(ClearGbmError::TreeConstructionFailed {
                        reason: format!("internal node {} missing threshold", current.node_id()),
                    })
                }
            };
            if feature_value <= threshold {
                left_child_id
            } else {
                right_child_id
            }
        };

        current = match tree.node(next_id) {
            Ok(node) => node,
            Err(e) => return Err(e),
        };
    }
}

/// Predicts leaf values for a batch of samples by traversing a decision tree.
///
/// Calls [`predict_single`] for each row in the feature matrix.
///
/// # Args
///
/// * `tree` - The decision tree to traverse.
/// * `features` - Feature matrix as a slice of slices. Each inner slice is one
///   sample's feature values.
///
/// # Returns
///
/// A vector of predictions, one per sample.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` - If `features` is empty.
/// * Any error from [`predict_single`] for individual samples.
pub fn predict_tree(tree: &Tree, features: &[&[f64]]) -> Result<Vec<f64>, ClearGbmError> {
    if features.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "features matrix for predict_tree".to_string(),
        });
    }

    let mut predictions = Vec::with_capacity(features.len());
    for row in features {
        let pred = match predict_single(tree, row) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        predictions.push(pred);
    }
    Ok(predictions)
}

/// Predicts raw scores for a batch of samples using an ensemble of trees.
///
/// Computes: `raw_pred[i] = base_prediction + learning_rate * sum(tree_j.predict(features[i]))`
/// for all trees `j`.
///
/// # Args
///
/// * `trees` - Slice of decision trees in the ensemble.
/// * `features` - Feature matrix as a slice of slices. Each inner slice is one
///   sample's feature values.
/// * `config` - Ensemble prediction configuration (base prediction and learning rate).
///
/// # Returns
///
/// A vector of raw predictions (log-odds for classification), one per sample.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` - If `features` or `trees` is empty.
/// * Any error from [`predict_single`] for individual samples.
pub fn predict_ensemble(
    trees: &[Tree],
    features: &[&[f64]],
    config: &PredictEnsembleConfig,
) -> Result<Vec<f64>, ClearGbmError> {
    if features.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "features matrix for predict_ensemble".to_string(),
        });
    }
    if trees.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "trees for predict_ensemble".to_string(),
        });
    }

    let n_samples = features.len();
    let mut raw_preds = vec![config.base_prediction(); n_samples];

    for tree in trees {
        for (i, row) in features.iter().enumerate() {
            let tree_pred = match predict_single(tree, row) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            raw_preds[i] += config.learning_rate() * tree_pred;
        }
    }

    Ok(raw_preds)
}

/// Converts raw predictions to binary class probabilities using sigmoid.
///
/// For binary classification, applies [`sigmoid`] to each raw prediction
/// (log-odds) to produce `(probability_class_0, probability_class_1)` pairs.
///
/// # Args
///
/// * `raw_predictions` - Raw predictions (log-odds) from [`predict_ensemble`].
///
/// # Returns
///
/// A vector of `(prob_class_0, prob_class_1)` tuples, one per sample.
/// Returns an empty vector if input is empty.
#[must_use]
pub fn predict_proba(raw_predictions: &[f64]) -> Vec<(f64, f64)> {
    let mut result = Vec::with_capacity(raw_predictions.len());
    for &raw in raw_predictions {
        let prob_1 = sigmoid(raw);
        let prob_0 = 1.0_f64 - prob_1;
        result.push((prob_0, prob_1));
    }
    result
}
