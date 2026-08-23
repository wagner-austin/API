//! The multiclass boosting loop: K score columns, K trees per round.
//!
//! Scores, gradients and hessians live in flat CLASS-MAJOR buffers
//! (`buf[class * n_samples + row]`, LightGBM's layout), so each class's
//! gradient/hessian block is a contiguous slice the tree builder consumes
//! directly. Every round softmaxes each row once, fills all K blocks, then
//! trains one tree per class on that class's block; trees are stored
//! round-major, so tree `t` belongs to class `t % n_classes`.
//!
//! Rows are subsampled once per ROUND and shared by the round's K trees.
//! The per-split and per-tree feature-mask derivations mix the GLOBAL tree
//! index (`round * n_classes + class`) instead of the round, so every tree
//! in a round draws distinct masks while the derivations stay pure
//! functions of the run seed.

use crate::error::ClearGbmError;
use crate::losses::multiclass::{
    fill_row_grad_hess, multiclass_log_loss, softmax_row_into, MulticlassGradBuffers,
};
use crate::losses::multiclass_initial_predictions;
use crate::predict::predict_tree;
use crate::tree::{
    build_tree_leaf_wise_with_leaf_assignment, build_tree_with_leaf_assignment,
    select_tree_features, BuildTreeInput, FeatureSubsample, Tree,
};

use super::config::{GradientBoostingConfig, GrowthStrategy};
use super::early_stopping::EarlyStoppingState;
use super::labels::ResolvedMulticlass;
use super::model::{BaseScore, GradientBoostingModel};
use super::rng::SimpleRng;
use super::setup::prepare_training;
use super::subsampling::get_sample_indices;
use super::train::TrainingRuntime;

/// Trains the multiclass softmax task.
///
/// # Args
///
/// * `x_train` - Training feature matrix, already shape-validated.
/// * `n_features` - The validated feature count.
/// * `mc` - The resolved multiclass task (labels, class count, weights,
///   optional validation split).
/// * `config` - Training hyperparameters.
/// * `feature_names` - Feature names (one per feature).
/// * `runtime` - Worker-thread policy and injection hooks.
///
/// # Errors
///
/// Everything [`prepare_training`] and tree construction can raise, plus
/// the loss functions' shape errors.
pub(super) fn train_multiclass(
    x_train: &[&[f64]],
    n_features: usize,
    mc: &ResolvedMulticlass<'_>,
    config: &GradientBoostingConfig,
    feature_names: &[String],
    runtime: &TrainingRuntime<'_>,
) -> Result<GradientBoostingModel, ClearGbmError> {
    let hooks = runtime.hooks;
    let n_classes = mc.n_classes;
    let n_train = x_train.len();

    // The u32 ceiling, checked before any K-sized allocation: it bounds the
    // Friedman rescale conversion below and matches the crate's other count
    // ceilings.
    let n_classes_u32 = match u32::try_from(n_classes) {
        Ok(v) => v,
        Err(_) => {
            return Err(ClearGbmError::IntegerConversion {
                context: format!("n_classes = {n_classes} exceeds u32::MAX"),
            })
        }
    };

    let prepared = propagate!(prepare_training(x_train, n_features, config));
    let feature_bins = &prepared.feature_bins;
    let bin_thresholds = &prepared.bin_thresholds;
    let categorical_layout = prepared.categorical_layout.as_ref();
    let tree_build_config = &prepared.tree_build_config;
    let tree_column_budget = prepared.tree_column_budget;

    // Per-class base scores: log of the (weighted) class priors,
    // uncentered (the LightGBM comparison arm).
    let class_bases = propagate!(multiclass_initial_predictions(
        mc.y_train, n_classes, mc.weights
    ));

    // Class-major running scores, each class's block filled with its base.
    let mut scores = vec![0.0_f64; n_train * n_classes];
    for (class, &base) in class_bases.iter().enumerate() {
        scores[class * n_train..(class + 1_usize) * n_train].fill(base);
    }
    let mut val_scores: Vec<f64> = match &mc.val {
        Some(v) => {
            let n_val = v.x.len();
            let mut buf = vec![0.0_f64; n_val * n_classes];
            for (class, &base) in class_bases.iter().enumerate() {
                buf[class * n_val..(class + 1_usize) * n_val].fill(base);
            }
            buf
        }
        None => Vec::new(),
    };

    // The Friedman rescale K / (K - 1) for the hessian.
    let k_f64 = f64::from(n_classes_u32);
    let factor = k_f64 / (k_f64 - 1.0_f64);

    let mut rng = SimpleRng::new(config.random_state());
    let mut es_state: Option<EarlyStoppingState> =
        config.early_stopping_rounds().map(EarlyStoppingState::new);

    let pool = match (hooks.build_pool)(runtime.parallelism.thread_count()) {
        Ok(built) => built,
        Err(e) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "n_jobs".to_string(),
                reason: format!("could not build a worker pool: {e}"),
            })
        }
    };

    let build_trees = || -> Result<Vec<Tree>, ClearGbmError> {
        let mut trees: Vec<Tree> = Vec::with_capacity(config.n_estimators() * n_classes);
        let mut gradients = vec![0.0_f64; n_train * n_classes];
        let mut hessians = vec![0.0_f64; n_train * n_classes];
        let mut probas = vec![0.0_f64; n_classes];

        for round in 0_usize..config.n_estimators() {
            // a. One softmax per row fills every class's gradient/hessian
            // entry; the optional per-row weight multiplies both, matching
            // the P2 weighting rule.
            {
                let mut bufs = MulticlassGradBuffers {
                    gradients: &mut gradients,
                    hessians: &mut hessians,
                    n_samples: n_train,
                    factor,
                };
                for (row, &label) in mc.y_train.iter().enumerate() {
                    softmax_row_into(&scores, n_train, row, &mut probas);
                    let weight = match mc.weights {
                        Some(ws) => ws[row],
                        None => 1.0_f64,
                    };
                    fill_row_grad_hess(&mut bufs, &probas, label, weight, row);
                }
            }

            // b. One row subsample per round, shared by the round's K trees.
            let sample_indices =
                propagate!(get_sample_indices(n_train, config.subsample(), &mut rng));

            // c. One tree per class over that class's contiguous block.
            for class in 0_usize..n_classes {
                let tree_index = round * n_classes + class;
                let tree_index_u64 = u64::try_from(tree_index).unwrap_or(u64::MAX);
                let feature_subsample = config.max_features().map(|k| FeatureSubsample {
                    k,
                    seed: config
                        .random_state()
                        .wrapping_add(tree_index_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15_u64)),
                });
                let tree_mask: Option<Vec<bool>> = tree_column_budget.map(|k| {
                    select_tree_features(config.random_state(), tree_index, k, n_features)
                });

                let grad_block = &gradients[class * n_train..(class + 1_usize) * n_train];
                let hess_block = &hessians[class * n_train..(class + 1_usize) * n_train];
                let input = BuildTreeInput {
                    sample_indices: &sample_indices,
                    gradients: grad_block,
                    hessians: hess_block,
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

                let (tree, leaf_value_per_sample) = match config.growth_strategy() {
                    GrowthStrategy::DepthWise => {
                        propagate!(build_tree_with_leaf_assignment(&input, hooks))
                    }
                    GrowthStrategy::LeafWise => {
                        propagate!(build_tree_leaf_wise_with_leaf_assignment(&input, hooks))
                    }
                };

                // Update this class's running scores: fast path from the
                // leaf assignment, tree walk only for subsampled-out rows.
                let lr = config.learning_rate();
                let class_offset = class * n_train;
                let mut needs_fallback: Vec<usize> = Vec::new();
                for i in 0_usize..n_train {
                    let lv = leaf_value_per_sample[i];
                    if lv.is_nan() {
                        needs_fallback.push(i);
                    } else {
                        scores[class_offset + i] += lr * lv;
                    }
                }
                if !needs_fallback.is_empty() {
                    let fallback_features: Vec<&[f64]> =
                        needs_fallback.iter().map(|&i| x_train[i]).collect();
                    let fallback_preds = propagate!(predict_tree(&tree, &fallback_features));
                    for (j, &i) in needs_fallback.iter().enumerate() {
                        scores[class_offset + i] += lr * fallback_preds[j];
                    }
                }

                // Keep the validation scores in step, class by class.
                if let Some(v) = &mc.val {
                    let n_val = v.x.len();
                    let val_preds = propagate!(predict_tree(&tree, v.x));
                    let val_offset = class * n_val;
                    for (i, &pred) in val_preds.iter().enumerate() {
                        val_scores[val_offset + i] += lr * pred;
                    }
                }

                trees.push(tree);
            }

            // d. Early stopping on the round's validation loss.
            let stop_at_round: Option<usize> = match &mc.val {
                Some(v) => {
                    let loss =
                        propagate!(multiclass_log_loss(v.y, &val_scores, n_classes, v.weight));
                    match es_state {
                        Some(ref mut es) => {
                            if es.update(loss, round) {
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
            if let Some(best_round) = stop_at_round {
                trees.truncate((best_round + 1_usize) * n_classes);
                break;
            }
        }

        Ok(trees)
    };
    let trees = propagate!(pool.install(build_trees));

    Ok(GradientBoostingModel::new(
        trees,
        BaseScore::PerClass(class_bases),
        config.learning_rate(),
        feature_names.to_vec(),
        config.clone(),
    ))
}
