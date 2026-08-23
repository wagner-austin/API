//! Trained gradient boosting model for inference.
//!
//! Wraps the trained ensemble of trees with the training configuration
//! for making predictions on new data.

use crate::error::ClearGbmError;
use crate::losses::multiclass::softmax_row_into;
use crate::predict::{predict_ensemble, predict_proba, predict_tree, PredictEnsembleConfig};
use crate::tree::Tree;

use super::config::{GradientBoostingConfig, Objective};

/// The objective's base score, shaped by task: one scalar for the
/// single-score objectives, one score per class for multiclass. An enum so
/// a trainer can never hand the model a fabricated scalar beside real
/// per-class scores or vice versa.
pub(crate) enum BaseScore {
    /// One base score (binary log-odds or the regression label mean).
    Single(f64),
    /// One base score per class (the per-class log priors).
    PerClass(Vec<f64>),
}

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
    /// Initial prediction before tree contributions for the single-score
    /// objectives (weighted log-odds for binary, label mean for
    /// regression). `None` exactly when the model is multiclass.
    base_prediction: Option<f64>,
    /// Per-class base scores (log priors). `Some` exactly when the model
    /// is multiclass; its length is the config's `n_classes`.
    class_base_predictions: Option<Vec<f64>>,
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
    /// * `base_score` - The objective's base score, shaped by task.
    /// * `learning_rate` - Shrinkage factor in (0.0, 1.0].
    /// * `feature_names` - Feature names from training.
    /// * `config` - Training configuration.
    pub(crate) fn new(
        trees: Vec<Tree>,
        base_score: BaseScore,
        learning_rate: f64,
        feature_names: Vec<String>,
        config: GradientBoostingConfig,
    ) -> Self {
        let (base_prediction, class_base_predictions) = match base_score {
            BaseScore::Single(b) => (Some(b), None),
            BaseScore::PerClass(v) => (None, Some(v)),
        };
        Self {
            trees,
            base_prediction,
            class_base_predictions,
            learning_rate,
            feature_names,
            config,
        }
    }

    /// Reassembles a model from its wire parts, enforcing the base-score
    /// pairing the constructor guarantees by shape.
    ///
    /// # Args
    ///
    /// * `trees` - The trees.
    /// * `base_prediction` - The scalar base score, or null.
    /// * `class_base_predictions` - The per-class base scores, or null.
    /// * `learning_rate` - Shrinkage factor.
    /// * `feature_names` - Feature names.
    /// * `config` - Training configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ClearGbmError::InvalidParameter`] unless exactly one base
    /// form is present and it matches the config's objective (per-class
    /// with length `n_classes` for multiclass, scalar otherwise).
    pub(crate) fn from_parts(
        trees: Vec<Tree>,
        base_prediction: Option<f64>,
        class_base_predictions: Option<Vec<f64>>,
        learning_rate: f64,
        feature_names: Vec<String>,
        config: GradientBoostingConfig,
    ) -> Result<Self, ClearGbmError> {
        let base_score = match (config.objective(), base_prediction, class_base_predictions) {
            (Objective::MulticlassSoftmax, None, Some(v)) => {
                let expected = config.n_classes().unwrap_or_default();
                if v.len() != expected {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "class_base_predictions".to_string(),
                        reason: format!("must hold n_classes ({expected}) scores, got {}", v.len()),
                    });
                }
                BaseScore::PerClass(v)
            }
            (Objective::BinaryLogLoss | Objective::SquaredError, Some(b), None) => {
                BaseScore::Single(b)
            }
            (objective, bp, cbp) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "base_prediction".to_string(),
                    reason: format!(
                        "objective \"{}\" pairs with {}; got base_prediction {} and \
                         class_base_predictions {}",
                        objective.as_str(),
                        match objective {
                            Objective::MulticlassSoftmax =>
                                "null base_prediction plus per-class scores",
                            Objective::BinaryLogLoss | Objective::SquaredError =>
                                "a scalar base_prediction and null per-class scores",
                        },
                        match bp {
                            Some(_) => "set",
                            None => "null",
                        },
                        match cbp {
                            Some(_) => "set",
                            None => "null",
                        },
                    ),
                })
            }
        };
        Ok(Self::new(
            trees,
            base_score,
            learning_rate,
            feature_names,
            config,
        ))
    }

    /// Returns the trained trees.
    #[must_use]
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Returns the scalar base score (`None` for multiclass models).
    #[must_use]
    pub fn base_prediction(&self) -> Option<f64> {
        self.base_prediction
    }

    /// Returns the per-class base scores (`None` for single-score models).
    #[must_use]
    pub fn class_base_predictions(&self) -> Option<&[f64]> {
        self.class_base_predictions.as_deref()
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
        let base = match self.base_prediction {
            Some(b) => b,
            None => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "objective".to_string(),
                    reason: "predict_raw is single-score; a \"multiclass_softmax\" model \
                             scores one column per class, so use predict_raw_multiclass"
                        .to_string(),
                })
            }
        };
        let ensemble_config = match PredictEnsembleConfig::new(base, self.learning_rate) {
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
            Objective::MulticlassSoftmax => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "objective".to_string(),
                    reason: "predict_proba is binary; a \"multiclass_softmax\" model has one \
                             probability per class, so use predict_proba_multiclass"
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

impl GradientBoostingModel {
    /// Predicts raw per-class scores for a batch of samples.
    ///
    /// For each row: `raw[k] = class_base[k] + learning_rate *
    /// sum(tree contributions for class k)`. Trees are stored round-major
    /// (round 0's K trees, then round 1's, ...), so tree `t` belongs to
    /// class `t % n_classes`.
    ///
    /// # Args
    ///
    /// * `x` - Feature matrix `[n_samples][n_features]`.
    ///
    /// # Returns
    ///
    /// One `n_classes`-long score vector per sample.
    ///
    /// # Errors
    ///
    /// * `ClearGbmError::InvalidParameter` if the model is not multiclass,
    ///   or its tree count is not a multiple of `n_classes`.
    /// * `ClearGbmError::EmptyInput` if `x` is empty.
    /// * Any prediction error from tree traversal.
    pub fn predict_raw_multiclass(&self, x: &[&[f64]]) -> Result<Vec<Vec<f64>>, ClearGbmError> {
        let bases = match &self.class_base_predictions {
            Some(b) => b,
            None => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "objective".to_string(),
                    reason: format!(
                        "predict_raw_multiclass requires objective \"multiclass_softmax\", \
                         got \"{}\"; use predict_raw",
                        self.config.objective().as_str()
                    ),
                })
            }
        };
        let n_classes = bases.len();
        if x.is_empty() {
            return Err(ClearGbmError::EmptyInput {
                context: "features matrix for predict_raw_multiclass".to_string(),
            });
        }
        if !self.trees.is_empty() && !self.trees.len().is_multiple_of(n_classes) {
            return Err(ClearGbmError::InvalidParameter {
                name: "trees".to_string(),
                reason: format!(
                    "a multiclass model stores whole rounds of {n_classes} trees, got {}",
                    self.trees.len()
                ),
            });
        }

        let mut scores: Vec<Vec<f64>> = x.iter().map(|_| bases.clone()).collect();
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            let class = tree_idx % n_classes;
            let tree_preds = match predict_tree(tree, x) {
                Ok(p) => p,
                Err(e) => return Err(e),
            };
            for (row, &pred) in tree_preds.iter().enumerate() {
                scores[row][class] += self.learning_rate * pred;
            }
        }
        Ok(scores)
    }

    /// Predicts per-class probabilities for a batch of samples.
    ///
    /// Softmaxes each row's raw scores from
    /// [`predict_raw_multiclass`](Self::predict_raw_multiclass).
    ///
    /// # Args
    ///
    /// * `x` - Feature matrix `[n_samples][n_features]`.
    ///
    /// # Returns
    ///
    /// One `n_classes`-long probability vector per sample, summing to 1.
    ///
    /// # Errors
    ///
    /// Same as [`predict_raw_multiclass`](Self::predict_raw_multiclass).
    pub fn predict_proba_multiclass(&self, x: &[&[f64]]) -> Result<Vec<Vec<f64>>, ClearGbmError> {
        let raw = match self.predict_raw_multiclass(x) {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        let n_classes = raw.first().map(Vec::len).unwrap_or_default();
        let mut out: Vec<Vec<f64>> = Vec::with_capacity(raw.len());
        let mut probas = vec![0.0_f64; n_classes];
        for row_scores in &raw {
            // The class-major gather expects a flat buffer; one row is the
            // n_samples = 1 special case of that layout.
            softmax_row_into(row_scores, 1_usize, 0_usize, &mut probas);
            out.push(probas.clone());
        }
        Ok(out)
    }

    /// Predicts the class label for a batch of samples.
    ///
    /// The argmax over each row's raw scores; ties resolve to the lowest
    /// class index, deterministically.
    ///
    /// # Args
    ///
    /// * `x` - Feature matrix `[n_samples][n_features]`.
    ///
    /// # Returns
    ///
    /// One class index per sample.
    ///
    /// # Errors
    ///
    /// Same as [`predict_raw_multiclass`](Self::predict_raw_multiclass).
    pub fn predict_class(&self, x: &[&[f64]]) -> Result<Vec<usize>, ClearGbmError> {
        let raw = match self.predict_raw_multiclass(x) {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        Ok(raw
            .iter()
            .map(|row_scores| {
                let mut best = 0_usize;
                let mut best_score = f64::NEG_INFINITY;
                for (k, &s) in row_scores.iter().enumerate() {
                    if s > best_score {
                        best_score = s;
                        best = k;
                    }
                }
                best
            })
            .collect())
    }
}
