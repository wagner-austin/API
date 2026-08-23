//! The shared single-score boosting rounds: one tree per round over
//! full-dataset gradients, for both fresh training and continuation.
//!
//! Extracted from the fresh trainer so continued training reuses the SAME
//! loop instead of a copy: the fresh path calls it with a zero round
//! offset and scores initialized at the base prediction; continuation
//! calls it with scores initialized from the existing model's predictions
//! and the offset set to the existing tree count, so per-tree feature-mask
//! seeds keep advancing instead of repeating the original run's draws.
//! With offset 0 the loop is the fresh trainer's, operation for operation
//! — the identity benchmark holds it to that, byte for byte.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::losses::{binary_log_loss, sigmoid_array, squared_error_loss};
use crate::predict::predict_tree;
use crate::tree::{
    build_tree_leaf_wise_with_leaf_assignment, build_tree_with_leaf_assignment,
    select_tree_features, BuildTreeInput, FeatureSubsample, Tree,
};

use super::config::{GradientBoostingConfig, GrowthStrategy};
use super::early_stopping::EarlyStoppingState;
use super::labels::ResolvedObjective;
use super::rng::SimpleRng;
use super::setup::PreparedTraining;
use super::subsampling::get_sample_indices;

/// Everything one single-score boosting run needs, borrowed from its
/// caller: the data, the resolved objective, the running score buffers,
/// the shared preparation, and the loop's shape (round count and the
/// global tree-index offset).
pub(super) struct SingleScoreRounds<'a, 'b> {
    /// Training feature matrix `[n_samples][n_features]`.
    pub x_train: &'a [&'a [f64]],
    /// The resolved objective with its labels and validation split.
    pub resolved: &'a ResolvedObjective<'a>,
    /// Running raw training scores; already initialized by the caller
    /// (base prediction for fresh training, the existing model's
    /// predictions for continuation). Updated in place.
    pub raw_preds_train: &'b mut Vec<f64>,
    /// Running raw validation scores, same initialization contract;
    /// empty when there is no validation split.
    pub raw_preds_val: &'b mut Vec<f64>,
    /// The shared one-time preparation (bins, thresholds, tree config).
    pub prepared: &'a PreparedTraining,
    /// Training hyperparameters.
    pub config: &'a GradientBoostingConfig,
    /// Rounds to run (the fresh path passes `config.n_estimators()`).
    pub n_rounds: usize,
    /// Global tree-index offset for the feature-subsample and per-tree
    /// mask derivations: 0 for fresh training, the existing tree count
    /// for continuation.
    pub round_offset: usize,
    /// Row-subsampling RNG, sequenced by the caller.
    pub rng: &'b mut SimpleRng,
    /// Early-stopping state, present when the config enables it.
    pub es_state: &'b mut Option<EarlyStoppingState>,
    /// Injection hooks for histogram building and pool construction.
    pub hooks: &'a Hooks,
}

/// Runs the boosting rounds and returns the trees they built, truncated
/// by early stopping when it fires.
///
/// # Args
///
/// * `run` - The borrowed run state; see [`SingleScoreRounds`].
///
/// # Errors
///
/// Any tree construction or prediction error, and the loss functions'
/// shape errors.
pub(super) fn run_single_score_rounds(
    run: &mut SingleScoreRounds<'_, '_>,
) -> Result<Vec<Tree>, ClearGbmError> {
    let config = run.config;
    let n_train = run.x_train.len();
    let n_features = run.prepared.feature_bins.n_features();
    let feature_bins = &run.prepared.feature_bins;
    let bin_thresholds = &run.prepared.bin_thresholds;
    let categorical_layout = run.prepared.categorical_layout.as_ref();
    let tree_build_config = &run.prepared.tree_build_config;
    let tree_column_budget = run.prepared.tree_column_budget;

    let mut trees: Vec<Tree> = Vec::with_capacity(run.n_rounds);

    for round in 0_usize..run.n_rounds {
        // a/b. Compute gradients and hessians under the objective
        // (inline — lengths match by construction). Kept in f64 end to
        // end. Narrowing these two streams to f32 for the histogram hot
        // loop was measured 8% SLOWER on this workload: at the node sizes
        // reached here both widths already fit in L2, so there is no
        // bandwidth to save, and every element then pays a widening
        // conversion before its accumulate. See the wiki page
        // `cleargbm-f32-score-narrowing-reverted`.
        let (gradients, hessians): (Vec<f64>, Vec<f64>) = match run.resolved {
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
                let probas = sigmoid_array(run.raw_preds_train);
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
                        let gradients: Vec<f64> = run
                            .raw_preds_train
                            .iter()
                            .zip(yt.iter())
                            .zip(ws.iter())
                            .map(|((&pred, &y), &w)| w * (pred - y))
                            .collect();
                        let hessians: Vec<f64> = ws.to_vec();
                        (gradients, hessians)
                    }
                    None => {
                        let gradients: Vec<f64> = run
                            .raw_preds_train
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
        let sample_indices = propagate!(get_sample_indices(n_train, config.subsample(), run.rng));

        // d. Build tree input. The feature-subsample seed mixes the
        // GLOBAL tree index (the boosting round plus the continuation
        // offset) into the run seed so the same node id draws a different
        // subset in every tree — including trees added by continuation;
        // the derivation is stream-free (see `FeatureSubsample`), so the
        // row-subsampling RNG above is untouched and unsubsampled runs
        // stay bit-identical.
        let global_round = round + run.round_offset;
        let round_u64 = u64::try_from(global_round).unwrap_or(u64::MAX);
        let feature_subsample = config.max_features().map(|k| FeatureSubsample {
            k,
            seed: config
                .random_state()
                .wrapping_add(round_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15_u64)),
        });
        // The per-tree column mask, when `colsample_bytree` is set: a
        // pure function of (random_state, global round) on its own stream
        // (see TREE_MIX), so the other RNG consumers are untouched and
        // the colsample-off path stays bit-identical.
        let tree_mask: Option<Vec<bool>> = tree_column_budget
            .map(|k| select_tree_features(config.random_state(), global_round, k, n_features));
        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins_rows: feature_bins.bins(),
            n_samples: feature_bins.n_samples(),
            n_features: feature_bins.n_features(),
            n_regular_bins: feature_bins.n_regular_bins(),
            bin_thresholds,
            config: tree_build_config,
            monotonic_constraints: config.monotonic_constraints(),
            feature_subsample,
            tree_feature_mask: tree_mask.as_deref(),
            categorical: categorical_layout,
        };

        // e. Build tree (and capture per-sample leaf assignments as a
        // side effect — see build_tree_with_leaf_assignment docs).
        // The growth policy is dispatched here, the one place that holds
        // the full `GradientBoostingConfig`; the tree module exposes the
        // two growers and stays free of the policy vocabulary.
        let (tree, leaf_value_per_sample) = match config.growth_strategy() {
            GrowthStrategy::DepthWise => {
                propagate!(build_tree_with_leaf_assignment(&input, run.hooks))
            }
            GrowthStrategy::LeafWise => {
                propagate!(build_tree_leaf_wise_with_leaf_assignment(&input, run.hooks))
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
        for (i, &lv) in leaf_value_per_sample.iter().enumerate() {
            if lv.is_nan() {
                needs_fallback.push(i);
            } else {
                run.raw_preds_train[i] += lr * lv;
            }
        }
        if !needs_fallback.is_empty() {
            // Only walk the tree for samples the tree wasn't built on.
            let fallback_features: Vec<&[f64]> =
                needs_fallback.iter().map(|&i| run.x_train[i]).collect();
            let fallback_preds = propagate!(predict_tree(&tree, &fallback_features));
            for (j, &i) in needs_fallback.iter().enumerate() {
                run.raw_preds_train[i] += lr * fallback_preds[j];
            }
        }

        // g. Early stopping check on the objective's validation loss
        // (before push, so we can borrow tree)
        let val_loss: Option<f64> = match run.resolved {
            ResolvedObjective::Binary {
                val: Some(v),
                scale_pos_weight,
                ..
            } => {
                let val_preds = propagate!(predict_tree(&tree, v.x));
                for (dst, &pred) in run.raw_preds_val.iter_mut().zip(val_preds.iter()) {
                    *dst += config.learning_rate() * pred;
                }
                let val_probas = sigmoid_array(run.raw_preds_val);
                Some(propagate!(binary_log_loss(
                    v.y,
                    &val_probas,
                    *scale_pos_weight,
                    v.weight
                )))
            }
            ResolvedObjective::SquaredError { val: Some(v), .. } => {
                let val_preds = propagate!(predict_tree(&tree, v.x));
                for (dst, &pred) in run.raw_preds_val.iter_mut().zip(val_preds.iter()) {
                    *dst += config.learning_rate() * pred;
                }
                Some(propagate!(squared_error_loss(
                    v.y,
                    run.raw_preds_val,
                    v.weight
                )))
            }
            ResolvedObjective::Binary { val: None, .. }
            | ResolvedObjective::SquaredError { val: None, .. } => None,
        };
        let stop_at_round: Option<usize> = match val_loss {
            Some(loss) => match run.es_state {
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
}
