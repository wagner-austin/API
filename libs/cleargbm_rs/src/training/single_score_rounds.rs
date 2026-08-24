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
    select_tree_features, BuildTreeInput, FeatureSubsample, QuantizedTreeData, Tree,
};

use super::config::{GradientBoostingConfig, GrowthStrategy};
use super::early_stopping::EarlyStoppingState;
use super::goss::{goss_sample_indices, GossRates};
use super::labels::ResolvedObjective;
use super::quantize::{
    discretize_gradients, generate_rounding_randoms, rotation_offset, DiscretizeRequest,
    QuantRoundingRandoms,
};
use super::rng::SimpleRng;
use super::setup::PreparedTraining;
use super::subsampling::get_sample_indices;

/// The run-scoped quantization state: the knob's bin count and the
/// pre-generated rounding randoms, both pure functions of the config
/// (and the row count), created once before the rounds.
struct QuantRunState {
    /// The `quantized_gradient_bins` value.
    n_quant_bins: usize,
    /// The pre-generated per-row rounding randoms.
    randoms: QuantRoundingRandoms,
}

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

    // Quantized training's run state: generated once, before any round.
    // The randoms are a pure function of (random_state, n_train), so a
    // continuation regenerates exactly the vectors the fresh run had —
    // one of the two properties (with the pure per-round rotation
    // offset) that keep split training exact under quantization.
    let quant_state: Option<QuantRunState> =
        config.quantized_gradient_bins().map(|bins| QuantRunState {
            n_quant_bins: bins,
            randoms: generate_rounding_randoms(config.random_state(), n_train),
        });

    for round in 0_usize..run.n_rounds {
        // a/b. Compute gradients and hessians under the objective
        // (inline — lengths match by construction). Kept in f64 end to
        // end. Narrowing these two streams to f32 for the histogram hot
        // loop was measured 8% SLOWER on this workload: at the node sizes
        // reached here both widths already fit in L2, so there is no
        // bandwidth to save, and every element then pays a widening
        // conversion before its accumulate. See the wiki page
        // `cleargbm-f32-score-narrowing-reverted`.
        let (mut gradients, mut hessians): (Vec<f64>, Vec<f64>) = match run.resolved {
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

        // c. Row selection: the GOSS pass when both rates are set and the
        // warmup is over (the shipped semantics skip sampling while
        // `round < 1 / learning_rate`, integer-truncated), otherwise the
        // subsample draw. GOSS excludes `subsample < 1` at the config
        // boundary, so exactly one sampler ever consumes the RNG.
        let global_round_for_warmup = round + run.round_offset;
        let sample_indices = match (config.goss_top_rate(), config.goss_other_rate()) {
            (Some(top_rate), Some(other_rate)) => {
                let warmup = (1.0_f64 / config.learning_rate()).trunc();
                let round_u32 = u32::try_from(global_round_for_warmup).unwrap_or(u32::MAX);
                if f64::from(round_u32) < warmup {
                    propagate!(get_sample_indices(n_train, config.subsample(), run.rng))
                } else {
                    propagate!(goss_sample_indices(
                        &mut gradients,
                        &mut hessians,
                        GossRates {
                            top_rate,
                            other_rate,
                        },
                        run.rng,
                    ))
                }
            }
            (None, None) | (Some(_), None) | (None, Some(_)) => {
                propagate!(get_sample_indices(n_train, config.subsample(), run.rng))
            }
        };

        // c2. Quantized training: discretize this round's (post-GOSS)
        // gradients and hessians into the interleaved int8 stream. The
        // rotation offset is a pure function of (random_state, global
        // round), so continuation rounds discretize exactly as the
        // fresh run's same-numbered rounds do. The scan and the stream
        // cover ALL rows (LightGBM's shape); the tree reads only the
        // sampled ones.
        let quantized_round = match quant_state.as_ref() {
            Some(qs) => {
                let round_for_offset = u64::try_from(round + run.round_offset).unwrap_or(u64::MAX);
                let offset = propagate!(rotation_offset(
                    config.random_state(),
                    round_for_offset,
                    n_train,
                ));
                Some(discretize_gradients(DiscretizeRequest {
                    gradients: &gradients,
                    hessians: &hessians,
                    n_quant_bins: qs.n_quant_bins,
                    randoms: &qs.randoms,
                    offset,
                }))
            }
            None => None,
        };
        let quantized_data: Option<QuantizedTreeData<'_>> =
            match (quantized_round.as_ref(), quant_state.as_ref()) {
                (Some(q), Some(qs)) => Some(QuantizedTreeData {
                    packed_int8: &q.packed_int8,
                    grad_scale: q.scales.grad_scale,
                    hess_scale: q.scales.hess_scale,
                    n_quant_bins: qs.n_quant_bins,
                }),
                _ => None,
            };

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
            quantized: quantized_data,
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
