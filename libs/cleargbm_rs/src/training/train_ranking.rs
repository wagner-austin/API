//! The LambdaMART boosting loop: per-query pair lambdas, one tree per round.
//!
//! Ranking is a single-score task — each row carries one raw score, the
//! ranking key — so the loop mirrors the binary/regression shape: one tree
//! per round over full-dataset gradients. What differs is the gradient
//! source (the truncation-bounded pair scan of
//! [`crate::losses::lambdarank`], computed query by query over the running
//! scores) and the evaluation metric (mean NDCG at the truncation level;
//! early stopping minimizes `1 - NDCG`).
//!
//! Query groups are DATA, not configuration: they describe the rows, travel
//! with them, and never enter the persisted config — the same rule P2
//! established for sample weights. The base score is 0.0: ranking scores
//! are relative within a query, so there is no meaningful global offset to
//! start from.

use crate::error::ClearGbmError;
use crate::losses::lambdarank::{
    fill_query_lambdas, inverse_max_dcg_at_k, label_gains, mean_ndcg_at_k, validate_query_groups,
    validate_ranking_labels,
};
use crate::losses::validation::validate_weight_pairing;
use crate::predict::predict_tree;
use crate::tree::{
    build_tree_leaf_wise_with_leaf_assignment, build_tree_with_leaf_assignment,
    select_tree_features, BuildTreeInput, FeatureSubsample, Tree,
};

use super::config::{GradientBoostingConfig, GrowthStrategy, Objective};
use super::early_stopping::EarlyStoppingState;
use super::model::{BaseScore, GradientBoostingModel};
use super::rng::SimpleRng;
use super::setup::prepare_training;
use super::subsampling::get_sample_indices;
use super::train::TrainingRuntime;
use super::validation::{validate_training_inputs, validate_validation_inputs};

/// A ranking validation split: features, relevance labels, and the query
/// groups that partition them.
///
/// There is no evaluation-weight field on purpose: NDCG is a per-query
/// metric, and a per-document evaluation weight has no defined meaning for
/// it. Training weights (which multiply lambdas) remain per-row data on the
/// training side.
#[derive(Debug, Clone, Copy)]
pub struct RankingValidationData<'a> {
    /// Validation feature matrix `[n_val_samples][n_features]`.
    pub x: &'a [&'a [f64]],
    /// Validation relevance labels, each `<= 31`.
    pub y: &'a [u32],
    /// Documents per validation query, partitioning the rows exactly.
    pub groups: &'a [usize],
}

/// The ranking training data that travels with the feature matrix: labels,
/// the query groups that partition them, and optional per-row weights.
#[derive(Debug, Clone, Copy)]
pub struct RankingTrainingData<'a> {
    /// Relevance labels, each an integer grade `<= 31`
    /// (gain = `2^label - 1`).
    pub y: &'a [u32],
    /// Documents per training query, in row order, partitioning the rows
    /// exactly; every group holds 1..=10000 documents.
    pub groups: &'a [usize],
    /// Optional per-row training weights (finite, > 0); they multiply each
    /// row's lambda and hessian after the query scan. `None` weighs every
    /// row 1.
    pub weight: Option<&'a [f64]>,
}

/// Trains a LambdaMART ranking model.
///
/// # Args
///
/// * `x_train` - Training feature matrix `[n_samples][n_features]`.
/// * `train` - Relevance labels, query groups and optional weights.
/// * `validation` - Optional validation split with its own query groups.
/// * `config` - Training hyperparameters; `config.objective()` must be
///   `"lambdarank"`, which pairs with `lambdarank_truncation_level`.
/// * `feature_names` - Feature names (one per feature).
/// * `runtime` - Worker-thread policy and injection hooks.
///
/// # Returns
///
/// A trained [`GradientBoostingModel`]; its raw scores are ranking keys
/// (documents sort by them, descending).
///
/// # Errors
///
/// * `ClearGbmError::InvalidParameter` if the objective is not
///   `"lambdarank"`, a label is over 31, groups do not partition the rows,
///   a query is empty or over 10000 documents, or a weight is invalid.
/// * `ClearGbmError::EmptyInput` / `ClearGbmError::ShapeMismatch` on shape
///   violations.
/// * Any tree construction or prediction error.
pub fn train_gradient_boosting_ranking(
    x_train: &[&[f64]],
    train: &RankingTrainingData<'_>,
    validation: Option<RankingValidationData<'_>>,
    config: &GradientBoostingConfig,
    feature_names: &[String],
    runtime: &TrainingRuntime<'_>,
) -> Result<GradientBoostingModel, ClearGbmError> {
    let hooks = runtime.hooks;
    let y_train = train.y;
    let groups = train.groups;
    let sample_weight = train.weight;

    if config.objective() != Objective::LambdaRank {
        return Err(ClearGbmError::InvalidParameter {
            name: "objective".to_string(),
            reason: format!(
                "the ranking entry trains \"lambdarank\", got \"{}\"; use the entry that \
                 matches the objective",
                config.objective().as_str()
            ),
        });
    }
    // Present by the config's pairing rule (enforced at construction), so
    // the None arm is statically dead — the crate's dead-arm idiom.
    let truncation_level = config.lambdarank_truncation_level().unwrap_or(1_usize);

    let n_features = propagate!(validate_training_inputs(
        x_train,
        y_train.len(),
        feature_names
    ));
    let n_train = x_train.len();
    propagate!(validate_ranking_labels(y_train, "y_train"));
    propagate!(validate_query_groups(groups, n_train, "group"));
    if let Some(w) = sample_weight {
        propagate!(validate_weight_pairing(n_train, w, "sample_weight"));
    }
    if let Some(v) = &validation {
        propagate!(validate_validation_inputs(v.x, v.y.len(), n_features));
        propagate!(validate_ranking_labels(v.y, "y_val"));
        propagate!(validate_query_groups(v.groups, v.x.len(), "val_group"));
    }

    // GOSS is implemented for the single-score loop; a ranking run
    // stating it would carry a knob training does not honour.
    if config.goss_top_rate().is_some() {
        return Err(ClearGbmError::InvalidParameter {
            name: "goss_top_rate".to_string(),
            reason: "GOSS is implemented for the single-score objectives; \
                     \"lambdarank\" GOSS is not implemented"
                .to_string(),
        });
    }

    let prepared = propagate!(prepare_training(x_train, n_features, config));
    let feature_bins = &prepared.feature_bins;
    let bin_thresholds = &prepared.bin_thresholds;
    let categorical_layout = prepared.categorical_layout.as_ref();
    let tree_build_config = &prepared.tree_build_config;
    let tree_column_budget = prepared.tree_column_budget;

    // Per-query 1/maxDCG, precomputed once: labels never change, so the
    // normalizer is a constant of the query (LightGBM precomputes the same).
    let gains = label_gains();
    let mut inverse_max_dcgs: Vec<f64> = Vec::with_capacity(groups.len());
    {
        let mut start = 0_usize;
        for &cnt in groups {
            let end = start + cnt;
            inverse_max_dcgs.push(inverse_max_dcg_at_k(
                &y_train[start..end],
                truncation_level,
                &gains,
            ));
            start = end;
        }
    }

    // Ranking has no global base score: scores are relative within a query.
    let base_prediction = 0.0_f64;
    let mut raw_preds_train = vec![base_prediction; n_train];
    let mut raw_preds_val: Vec<f64> = match &validation {
        Some(v) => vec![base_prediction; v.x.len()],
        None => Vec::new(),
    };

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
        let mut trees: Vec<Tree> = Vec::with_capacity(config.n_estimators());
        let mut gradients = vec![0.0_f64; n_train];
        let mut hessians = vec![0.0_f64; n_train];

        for round in 0_usize..config.n_estimators() {
            // a. Fill lambdas query by query over the running scores, then
            // apply the optional per-row weights to both streams (the P2
            // rule: weights multiply lambda AND hessian, after the scan).
            {
                let mut start = 0_usize;
                for (query, &cnt) in groups.iter().enumerate() {
                    let end = start + cnt;
                    fill_query_lambdas(
                        &raw_preds_train[start..end],
                        &y_train[start..end],
                        inverse_max_dcgs[query],
                        truncation_level,
                        &gains,
                        &mut gradients[start..end],
                        &mut hessians[start..end],
                    );
                    start = end;
                }
                if let Some(ws) = sample_weight {
                    for i in 0_usize..n_train {
                        gradients[i] *= ws[i];
                        hessians[i] *= ws[i];
                    }
                }
            }

            // b. Row subsample for tree building (gradients stay full).
            let sample_indices =
                propagate!(get_sample_indices(n_train, config.subsample(), &mut rng));

            // c. Build the round's tree — identical wiring to the
            // single-score loop.
            let round_u64 = u64::try_from(round).unwrap_or(u64::MAX);
            let feature_subsample = config.max_features().map(|k| FeatureSubsample {
                k,
                seed: config
                    .random_state()
                    .wrapping_add(round_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15_u64)),
            });
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
                bin_thresholds,
                config: tree_build_config,
                monotonic_constraints: config.monotonic_constraints(),
                feature_subsample,
                tree_feature_mask: tree_mask.as_deref(),
                categorical: categorical_layout,
                quantized: None,
            };
            let (tree, leaf_value_per_sample) = match config.growth_strategy() {
                GrowthStrategy::DepthWise => {
                    propagate!(build_tree_with_leaf_assignment(&input, hooks))
                }
                GrowthStrategy::LeafWise => {
                    propagate!(build_tree_leaf_wise_with_leaf_assignment(&input, hooks))
                }
            };

            // d. Update training scores: fast path from the leaf
            // assignment, tree walk only for subsampled-out rows.
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
                let fallback_features: Vec<&[f64]> =
                    needs_fallback.iter().map(|&i| x_train[i]).collect();
                let fallback_preds = propagate!(predict_tree(&tree, &fallback_features));
                for (j, &i) in needs_fallback.iter().enumerate() {
                    raw_preds_train[i] += lr * fallback_preds[j];
                }
            }

            // e. Early stopping on 1 - mean validation NDCG at the
            // truncation level (the state minimizes, NDCG maximizes).
            let stop_at_round: Option<usize> = match &validation {
                Some(v) => {
                    let val_preds = propagate!(predict_tree(&tree, v.x));
                    for i in 0_usize..raw_preds_val.len() {
                        raw_preds_val[i] += lr * val_preds[i];
                    }
                    let ndcg =
                        mean_ndcg_at_k(&raw_preds_val, v.y, v.groups, truncation_level, &gains);
                    let loss = 1.0_f64 - ndcg;
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

            trees.push(tree);
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
        BaseScore::Single(base_prediction),
        config.learning_rate(),
        feature_names.to_vec(),
        config.clone(),
    ))
}
