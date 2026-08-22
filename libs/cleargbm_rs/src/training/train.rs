//! Core training loop for gradient boosting.
//!
//! Orchestrates the full training pipeline: validation, binning,
//! iterative tree construction with gradient/hessian updates,
//! optional early stopping, and model assembly.

use crate::binning::precompute_feature_bins;
use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::losses::{binary_log_loss, binary_log_loss_initial_prediction, sigmoid_array};
use crate::predict::predict_tree;
use crate::tree::{
    build_tree_leaf_wise_with_leaf_assignment, build_tree_with_leaf_assignment, BuildTreeInput,
    Tree, TreeBuildConfig,
};
use crate::types::SplitConfig;

use super::config::{GradientBoostingConfig, GrowthStrategy};
use super::early_stopping::EarlyStoppingState;
use super::model::GradientBoostingModel;
use super::parallelism::Parallelism;
use super::rng::SimpleRng;
use super::subsampling::get_sample_indices;
use super::validation::{validate_training_inputs, validate_validation_inputs};

/// How a training run executes, as opposed to what it learns.
///
/// Groups the two knobs that describe execution rather than the model: the
/// worker-thread policy and the injection hooks. Neither is persisted with the
/// fitted model, so they travel together and separately from the
/// hyperparameters in [`GradientBoostingConfig`].
pub struct TrainingRuntime<'a> {
    /// Worker-thread policy for this run.
    pub parallelism: Parallelism,

    /// Dependency injection hooks for histogram building and pool construction.
    pub hooks: &'a Hooks,
}

/// Trains a gradient boosting model on binary classification data.
///
/// Orchestrates the full training pipeline:
/// 1. Validates inputs (shapes, labels, feature names)
/// 2. Computes initial prediction (log-odds of class prevalence)
/// 3. Pre-bins features into histogram indices
/// 4. Iterates boosting rounds: compute gradients/hessians, subsample,
///    build tree, update predictions
/// 5. Optionally applies early stopping based on validation loss
/// 6. Returns the trained model
///
/// # Args
///
/// * `x_train` - Training feature matrix `[n_samples][n_features]`.
/// * `y_train` - Training labels (binary: 0 or 1).
/// * `x_val` - Optional validation feature matrix.
/// * `y_val` - Optional validation labels.
/// * `config` - Training hyperparameters.
/// * `feature_names` - Feature names (one per feature).
/// * `runtime` - Worker-thread policy and injection hooks. Does not affect the
///   fitted model, only how it is built.
///
/// # Returns
///
/// A trained [`GradientBoostingModel`] ready for inference.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if training data is empty.
/// * `ClearGbmError::ShapeMismatch` if dimensions are inconsistent.
/// * `ClearGbmError::InvalidLabel` if labels are not 0/1.
/// * `ClearGbmError::InvalidParameter` if only one of x_val/y_val is provided,
///   or on configuration errors.
/// * Any tree construction or prediction error.
pub fn train_gradient_boosting(
    x_train: &[&[f64]],
    y_train: &[u8],
    x_val: Option<&[&[f64]]>,
    y_val: Option<&[u8]>,
    config: &GradientBoostingConfig,
    feature_names: &[String],
    runtime: &TrainingRuntime<'_>,
) -> Result<GradientBoostingModel, ClearGbmError> {
    let hooks = runtime.hooks;
    // 1. Validate training inputs
    let n_features = match validate_training_inputs(x_train, y_train, feature_names) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };

    // 2. Validate optional validation inputs (both or neither required)
    let validation_data: Option<(&[&[f64]], &[u8])> = match (x_val, y_val) {
        (Some(xv), Some(yv)) => {
            match validate_validation_inputs(xv, yv, n_features) {
                Ok(()) => {}
                Err(e) => return Err(e),
            };
            Some((xv, yv))
        }
        (None, None) => None,
        (Some(_), None) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "y_val".to_string(),
                reason: "y_val must be provided when x_val is provided".to_string(),
            });
        }
        (None, Some(_)) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "x_val".to_string(),
                reason: "x_val must be provided when y_val is provided".to_string(),
            });
        }
    };

    // 3. Compute initial prediction (log-odds of class prevalence)
    let base_prediction = match binary_log_loss_initial_prediction(y_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };

    let n_train = x_train.len();
    let mut raw_preds_train = vec![base_prediction; n_train];

    // 4. Initialize validation predictions if needed
    let mut raw_preds_val: Vec<f64> = match validation_data {
        Some((xv, _)) => vec![base_prediction; xv.len()],
        None => Vec::new(),
    };

    // 5. Precompute feature bins
    let feature_bins = propagate!(precompute_feature_bins(x_train, config.max_bins()));
    let bin_thresholds = feature_bins.bin_thresholds();

    // 6. Initialize RNG for subsampling
    let mut rng = SimpleRng::new(config.random_state());

    // 7. Initialize early stopping state
    let mut es_state: Option<EarlyStoppingState> =
        config.early_stopping_rounds().map(EarlyStoppingState::new);

    // 8. Build tree configuration
    let split_config = propagate!(SplitConfig::new(
        config.min_samples_split(),
        config.min_samples_leaf(),
        config.max_bins(),
        config.reg_lambda(),
        0.0_f64,
    ));

    // Under depth-wise growth the leaf count is left unbounded (0) and
    // `max_depth` does the bounding, which is what every manifest recorded
    // before the growth axis existed. Under leaf-wise there is no depth to
    // bound the shape, so the validated `num_leaves` becomes the budget.
    let max_leaves = config.num_leaves().unwrap_or_default();

    let tree_build_config = propagate!(TreeBuildConfig::new(
        config.max_depth(),
        max_leaves,
        config.reg_alpha(),
        config.reg_lambda(),
        split_config,
    ));

    // Build the worker pool once for the whole run rather than per tree.
    let pool = match (hooks.build_pool)(runtime.parallelism.thread_count()) {
        Ok(built) => built,
        Err(e) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "n_jobs".to_string(),
                reason: format!("could not build a worker pool: {e}"),
            })
        }
    };

    // 9. Validate monotonic constraints length if provided
    if let Some(mc) = config.monotonic_constraints() {
        if mc.len() != n_features {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("{n_features} monotonic constraints"),
                got: format!("{} monotonic constraints", mc.len()),
            });
        }
    }

    // 10. Boosting loop, run inside the run-scoped pool so the caller's
    // `n_jobs` actually bounds the worker count. Tree building owns every
    // rayon dispatch in the crate (per-feature histogram construction), so
    // installing here covers all of it. Without the install, `into_par_iter`
    // would fall back to rayon's global pool — one worker per core regardless
    // of `n_jobs` — which both ignores the caller and degrades badly on a
    // contended machine, since every worker stays runnable and competes.
    //
    // Installed once for the whole run rather than once per tree: `install`
    // hands the closure to the pool and blocks the calling thread until it
    // finishes, so installing per boosting round adds one handoff per round.
    let build_trees = || -> Result<Vec<Tree>, ClearGbmError> {
        let mut trees: Vec<Tree> = Vec::with_capacity(config.n_estimators());

        for round in 0_usize..config.n_estimators() {
            // a. Compute probabilities from current raw predictions
            let probas = sigmoid_array(&raw_preds_train);

            // b. Compute gradients and hessians (inline — lengths match by
            // construction). Kept in f64 end to end. Narrowing these two streams
            // to f32 for the histogram hot loop was measured 8% SLOWER on this
            // workload: at the node sizes reached here both widths already fit in
            // L2, so there is no bandwidth to save, and every element then pays a
            // widening conversion before its accumulate. See the wiki page
            // `cleargbm-f32-score-narrowing-reverted`.
            let gradients: Vec<f64> = probas
                .iter()
                .zip(y_train.iter())
                .map(|(&p, &y)| p - f64::from(y))
                .collect();
            let hessians: Vec<f64> = probas.iter().map(|&p| p * (1.0_f64 - p)).collect();

            // c. Get sample indices (subsampling)
            let sample_indices =
                propagate!(get_sample_indices(n_train, config.subsample(), &mut rng));

            // d. Build tree input
            let input = BuildTreeInput {
                sample_indices: &sample_indices,
                gradients: &gradients,
                hessians: &hessians,
                bins: feature_bins.bins(),
                n_samples: feature_bins.n_samples(),
                n_features: feature_bins.n_features(),
                n_regular_bins: feature_bins.n_regular_bins(),
                bin_thresholds: &bin_thresholds,
                config: &tree_build_config,
                monotonic_constraints: config.monotonic_constraints(),
            };

            // e. Build tree (and capture per-sample leaf assignments as a
            // side effect — see build_tree_with_leaf_assignment docs).
            // The growth policy is dispatched here, the one place that holds
            // the full `GradientBoostingConfig`; the tree module exposes the
            // two growers and stays free of the policy vocabulary.
            let (tree, leaf_value_per_sample) = match config.growth_strategy() {
                GrowthStrategy::DepthWise => {
                    propagate!(build_tree_with_leaf_assignment(&input, hooks))
                }
                GrowthStrategy::LeafWise => {
                    propagate!(build_tree_leaf_wise_with_leaf_assignment(&input, hooks))
                }
            };

            // f. Update training predictions.
            // Fast path: for samples covered by sample_indices (leaf_value not
            // NaN), the leaf-value is known from tree building — direct O(N)
            // lookup + add, no tree walk. Fallback path: NaN samples
            // (subsampled-out this round) need predict_tree. When subsample=1.0
            // every sample is covered by the fast path.
            let lr = config.learning_rate();
            let mut needs_fallback: Vec<usize> = Vec::new();
            for i in 0_usize..n_train {
                let lv = leaf_value_per_sample[i];
                if lv.is_nan() {
                    needs_fallback.push(i);
                } else {
                    raw_preds_train[i] += lr * lv;
                }
            }
            if !needs_fallback.is_empty() {
                // Only walk the tree for samples the tree wasn't built on.
                let fallback_features: Vec<&[f64]> =
                    needs_fallback.iter().map(|&i| x_train[i]).collect();
                let fallback_preds = propagate!(predict_tree(&tree, &fallback_features));
                for (j, &i) in needs_fallback.iter().enumerate() {
                    raw_preds_train[i] += lr * fallback_preds[j];
                }
            }

            // g. Early stopping check on validation set (before push, so we can borrow tree)
            let stop_at_round: Option<usize> = match validation_data {
                Some((xv, yv)) => {
                    let val_preds = propagate!(predict_tree(&tree, xv));
                    for i in 0_usize..raw_preds_val.len() {
                        raw_preds_val[i] += config.learning_rate() * val_preds[i];
                    }
                    let val_probas = sigmoid_array(&raw_preds_val);
                    let val_loss = propagate!(binary_log_loss(yv, &val_probas));
                    match es_state {
                        Some(ref mut es) => {
                            if es.update(val_loss, round) {
                                Some(es.best_round())
                            } else {
                                None
                            }
                        }
                        None => None,
                    }
                }
                None => None,
            };

            // h. Store tree
            trees.push(tree);

            // i. If early stopping triggered, truncate and break
            if let Some(best_round) = stop_at_round {
                trees.truncate(best_round + 1_usize);
                break;
            }
        }

        Ok(trees)
    };
    let trees = propagate!(pool.install(build_trees));

    Ok(GradientBoostingModel::new(
        trees,
        base_prediction,
        config.learning_rate(),
        feature_names.to_vec(),
        2_usize, // n_classes = 2 for binary classification
        config.clone(),
    ))
}
