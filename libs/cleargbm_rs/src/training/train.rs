//! Core training loop for gradient boosting.
//!
//! Orchestrates the full training pipeline: validation, objective
//! resolution, binning, iterative tree construction with gradient/hessian
//! updates, optional early stopping, and model assembly.

use crate::binning::precompute_feature_bins;
use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::losses::{
    binary_log_loss, binary_log_loss_initial_prediction, sigmoid_array,
    squared_error_initial_prediction, squared_error_loss,
};
use crate::predict::predict_tree;
use crate::tree::{
    build_tree_leaf_wise_with_leaf_assignment, build_tree_with_leaf_assignment,
    select_tree_features, BuildTreeInput, FeatureSubsample, Tree, TreeBuildConfig,
};
use crate::types::SplitConfig;

use super::config::{GradientBoostingConfig, GrowthStrategy};
use super::early_stopping::EarlyStoppingState;
use super::labels::{resolve_objective, ResolvedObjective, TrainingLabels, ValidationData};
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

/// Trains a gradient boosting model under the configured objective.
///
/// Orchestrates the full training pipeline:
/// 1. Validates input shapes (rows, labels, feature names)
/// 2. Resolves the objective against the label kind (binary `u8` labels for
///    `binary_log_loss`, continuous `f64` targets for `squared_error`) and
///    validates label contents
/// 3. Computes the objective's base score (weighted log-odds / label mean)
/// 4. Pre-bins features into histogram indices
/// 5. Iterates boosting rounds: compute the objective's gradients/hessians,
///    subsample, build tree, update predictions
/// 6. Optionally applies early stopping based on the objective's validation
///    loss
/// 7. Returns the trained model
///
/// # Args
///
/// * `x_train` - Training feature matrix `[n_samples][n_features]`.
/// * `y_train` - Training labels, typed by kind ([`TrainingLabels`]).
/// * `sample_weight` - Optional per-row training weights (finite, > 0);
///   `None` weighs every row 1 and is bit-identical to weightless history.
/// * `validation` - Validation features paired with labels of the same kind
///   and optional evaluation weights, or `None`.
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
/// * `ClearGbmError::InvalidLabel` if binary labels are not 0/1.
/// * `ClearGbmError::InvalidParameter` if the label kind does not match the
///   objective, a continuous label is not finite, or on configuration errors.
/// * Any tree construction or prediction error.
pub fn train_gradient_boosting(
    x_train: &[&[f64]],
    y_train: TrainingLabels<'_>,
    sample_weight: Option<&[f64]>,
    validation: Option<ValidationData<'_>>,
    config: &GradientBoostingConfig,
    feature_names: &[String],
    runtime: &TrainingRuntime<'_>,
) -> Result<GradientBoostingModel, ClearGbmError> {
    let hooks = runtime.hooks;
    // 1. Validate training input shapes
    let n_features = match validate_training_inputs(x_train, y_train.len(), feature_names) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };

    // 2. Validate validation input shapes if provided
    if let Some(v) = validation {
        match validate_validation_inputs(v.x, v.y.len(), n_features) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    // 2b. Resolve the objective against the label kinds. Past this point an
    // objective/label mismatch is unrepresentable, so the boosting loop
    // dispatches with total matches.
    let resolved = propagate!(resolve_objective(
        config.objective(),
        config.scale_pos_weight(),
        y_train,
        sample_weight,
        validation,
    ));

    // 3. Compute the objective's base score
    let base_prediction = match &resolved {
        ResolvedObjective::Binary {
            y_train: yt,
            weights,
            scale_pos_weight,
            ..
        } => propagate!(binary_log_loss_initial_prediction(
            yt,
            *scale_pos_weight,
            *weights
        )),
        ResolvedObjective::SquaredError {
            y_train: yt,
            weights,
            ..
        } => {
            propagate!(squared_error_initial_prediction(yt, *weights))
        }
    };

    let n_train = x_train.len();
    let mut raw_preds_train = vec![base_prediction; n_train];

    // 4. Initialize validation predictions if needed
    let mut raw_preds_val: Vec<f64> = match resolved.val_features() {
        Some(xv) => vec![base_prediction; xv.len()],
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

    // 9b. Validate the per-split feature budget against the feature count
    // (the config layer cannot: it does not know n_features).
    if let Some(k) = config.max_features() {
        if k > n_features {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_features".to_string(),
                reason: format!("must be <= n_features ({n_features}), got {k}"),
            });
        }
    }

    // 9c. Resolve the per-tree column budget: k_tree = max(1,
    // floor(colsample_bytree * n_features)), the row-subsampling convention.
    // The count lives on [1, n_features] by construction (the fraction is
    // validated in (0, 1) exclusive), so no further pairing check is needed.
    let tree_column_budget: Option<usize> = match config.colsample_bytree() {
        Some(fraction) => Some(propagate!(crate::tree::tree_column_budget(
            fraction, n_features
        ))),
        None => None,
    };

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
            // a/b. Compute gradients and hessians under the objective
            // (inline — lengths match by construction). Kept in f64 end to
            // end. Narrowing these two streams to f32 for the histogram hot
            // loop was measured 8% SLOWER on this workload: at the node sizes
            // reached here both widths already fit in L2, so there is no
            // bandwidth to save, and every element then pays a widening
            // conversion before its accumulate. See the wiki page
            // `cleargbm-f32-score-narrowing-reverted`.
            let (gradients, hessians): (Vec<f64>, Vec<f64>) = match &resolved {
                ResolvedObjective::Binary {
                    y_train: yt,
                    weights,
                    scale_pos_weight,
                    ..
                } => {
                    // Probabilities come from the sigmoid of the running raw
                    // scores. The effective row weight is the product of the
                    // class term (`scale_pos_weight` for positives, 1 for
                    // negatives) and the optional per-row sample weight; the
                    // weighted log loss's first and second derivatives scale
                    // by it together. The weightless arm keeps the exact
                    // historical expressions - no synthesized `* 1.0` - so
                    // every recorded manifest stays bit-valid, and at weight
                    // 1.0 the class multiply is an IEEE identity: each
                    // specialization is the general path's special case,
                    // bit for bit.
                    let scale_pos_weight = *scale_pos_weight;
                    let probas = sigmoid_array(&raw_preds_train);
                    match weights {
                        Some(ws) => {
                            let gradients: Vec<f64> = probas
                                .iter()
                                .zip(yt.iter())
                                .zip(ws.iter())
                                .map(|((&p, &y), &w)| {
                                    if y == 1_u8 {
                                        (scale_pos_weight * w) * (p - 1.0_f64)
                                    } else {
                                        w * p
                                    }
                                })
                                .collect();
                            let hessians: Vec<f64> = probas
                                .iter()
                                .zip(yt.iter())
                                .zip(ws.iter())
                                .map(|((&p, &y), &w)| {
                                    if y == 1_u8 {
                                        (scale_pos_weight * w) * (p * (1.0_f64 - p))
                                    } else {
                                        w * (p * (1.0_f64 - p))
                                    }
                                })
                                .collect();
                            (gradients, hessians)
                        }
                        None => {
                            let gradients: Vec<f64> = probas
                                .iter()
                                .zip(yt.iter())
                                .map(|(&p, &y)| {
                                    if y == 1_u8 {
                                        scale_pos_weight * (p - 1.0_f64)
                                    } else {
                                        p
                                    }
                                })
                                .collect();
                            let hessians: Vec<f64> = probas
                                .iter()
                                .zip(yt.iter())
                                .map(|(&p, &y)| {
                                    if y == 1_u8 {
                                        scale_pos_weight * (p * (1.0_f64 - p))
                                    } else {
                                        p * (1.0_f64 - p)
                                    }
                                })
                                .collect();
                            (gradients, hessians)
                        }
                    }
                }
                ResolvedObjective::SquaredError {
                    y_train: yt,
                    weights,
                    ..
                } => {
                    // Squared error differentiates in raw-score space
                    // directly: gradient = w * (prediction - y), hessian = w,
                    // with the weightless arm keeping the bare historical
                    // expressions (gradient = pred - y, hessian = 1).
                    match weights {
                        Some(ws) => {
                            let gradients: Vec<f64> = raw_preds_train
                                .iter()
                                .zip(yt.iter())
                                .zip(ws.iter())
                                .map(|((&pred, &y), &w)| w * (pred - y))
                                .collect();
                            let hessians: Vec<f64> = ws.to_vec();
                            (gradients, hessians)
                        }
                        None => {
                            let gradients: Vec<f64> = raw_preds_train
                                .iter()
                                .zip(yt.iter())
                                .map(|(&pred, &y)| pred - y)
                                .collect();
                            let hessians: Vec<f64> = vec![1.0_f64; n_train];
                            (gradients, hessians)
                        }
                    }
                }
            };

            // c. Get sample indices (subsampling)
            let sample_indices =
                propagate!(get_sample_indices(n_train, config.subsample(), &mut rng));

            // d. Build tree input. The feature-subsample seed mixes the
            // boosting round into the run seed so the same node id draws a
            // different subset in every tree; the derivation is stream-free
            // (see `FeatureSubsample`), so the row-subsampling RNG above is
            // untouched and unsubsampled runs stay bit-identical.
            let round_u64 = u64::try_from(round).unwrap_or(u64::MAX);
            let feature_subsample = config.max_features().map(|k| FeatureSubsample {
                k,
                seed: config
                    .random_state()
                    .wrapping_add(round_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15_u64)),
            });
            // The per-tree column mask, when `colsample_bytree` is set: a
            // pure function of (random_state, round) on its own stream (see
            // TREE_MIX), so the other RNG consumers are untouched and the
            // colsample-off path stays bit-identical.
            let tree_mask: Option<Vec<bool>> = tree_column_budget
                .map(|k| select_tree_features(config.random_state(), round, k, n_features));
            let input = BuildTreeInput {
                sample_indices: &sample_indices,
                gradients: &gradients,
                hessians: &hessians,
                bins_rows: feature_bins.bins(),
                n_samples: feature_bins.n_samples(),
                n_features: feature_bins.n_features(),
                n_regular_bins: feature_bins.n_regular_bins(),
                bin_thresholds: &bin_thresholds,
                config: &tree_build_config,
                monotonic_constraints: config.monotonic_constraints(),
                feature_subsample,
                tree_feature_mask: tree_mask.as_deref(),
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

            // g. Early stopping check on the objective's validation loss
            // (before push, so we can borrow tree)
            let val_loss: Option<f64> = match &resolved {
                ResolvedObjective::Binary {
                    val: Some(v),
                    scale_pos_weight,
                    ..
                } => {
                    let val_preds = propagate!(predict_tree(&tree, v.x));
                    for i in 0_usize..raw_preds_val.len() {
                        raw_preds_val[i] += config.learning_rate() * val_preds[i];
                    }
                    let val_probas = sigmoid_array(&raw_preds_val);
                    Some(propagate!(binary_log_loss(
                        v.y,
                        &val_probas,
                        *scale_pos_weight,
                        v.weight
                    )))
                }
                ResolvedObjective::SquaredError { val: Some(v), .. } => {
                    let val_preds = propagate!(predict_tree(&tree, v.x));
                    for i in 0_usize..raw_preds_val.len() {
                        raw_preds_val[i] += config.learning_rate() * val_preds[i];
                    }
                    Some(propagate!(squared_error_loss(
                        v.y,
                        &raw_preds_val,
                        v.weight
                    )))
                }
                ResolvedObjective::Binary { val: None, .. }
                | ResolvedObjective::SquaredError { val: None, .. } => None,
            };
            let stop_at_round: Option<usize> = match val_loss {
                Some(loss) => match es_state {
                    Some(ref mut es) => {
                        if es.update(loss, round) {
                            Some(es.best_round())
                        } else {
                            None
                        }
                    }
                    None => None,
                },
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
        config.clone(),
    ))
}
