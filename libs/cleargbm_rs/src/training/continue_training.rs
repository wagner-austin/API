//! Continued training: more boosting rounds on top of an existing model.
//!
//! The design inverts LightGBM's `init_model` shape on purpose. LightGBM
//! bakes the old model's raw predictions into the dataset as an init score
//! and returns a booster holding ONLY the new trees — a delta model that
//! excludes its own baseline. Here the running scores are initialized the
//! same way (the existing model's raw predictions over the continuation
//! data), but the new trees are APPENDED to the existing model's trees, so
//! the returned artifact is self-contained: one model, one embedded
//! config, every question answerable from the file.
//!
//! What continuation states, and what it cannot:
//!
//! * The model trains under its OWN embedded config — the caller states
//!   only the additional round budget. The continued config's
//!   `n_estimators` becomes `existing trees + additional_rounds`, so the
//!   stated budget matches the artifact's actual capacity.
//! * Histogram bin edges are recomputed from the CONTINUATION data — the
//!   original bin edges are not stored in the model, and stating otherwise
//!   would be a fabrication. Same data in, same edges out.
//! * Per-tree feature-mask seeds continue from the existing tree count, so
//!   continuation trees draw new masks instead of repeating the original
//!   run's; the row-subsampling RNG starts a fresh stream from the config
//!   seed. Deterministic per (model, data, additional_rounds).

use crate::error::ClearGbmError;
use crate::tree::Tree;

use super::config::Objective;
use super::early_stopping::EarlyStoppingState;
use super::labels::{resolve_objective, TrainingLabels, ValidationData};
use super::model::{BaseScore, GradientBoostingModel};
use super::rng::SimpleRng;
use super::setup::prepare_training;
use super::single_score_rounds::{run_single_score_rounds, SingleScoreRounds};
use super::train::TrainingRuntime;
use super::validation::{validate_training_inputs, validate_validation_inputs};

/// Trains `additional_rounds` more trees on top of an existing model.
///
/// Supported for the single-score objectives (`binary_log_loss`,
/// `squared_error`) this landing; a multiclass or ranking model is
/// refused with the scope named rather than half-continued.
///
/// # Args
///
/// * `model` - The existing trained model; its embedded config drives the
///   continuation (objective, learning rate, tree shape, sampling knobs,
///   early stopping).
/// * `x_train` - Continuation feature matrix `[n_samples][n_features]`;
///   the column count must match the model's feature names.
/// * `y_train` - Continuation labels, typed to match the model's
///   objective.
/// * `sample_weight` - Optional per-row training weights.
/// * `validation` - Optional validation split for the config's early
///   stopping.
/// * `additional_rounds` - New boosting rounds to run (>= 1).
/// * `runtime` - Worker-thread policy and injection hooks.
///
/// # Returns
///
/// A new self-contained [`GradientBoostingModel`]: the existing trees
/// plus the continuation trees, the same base score and feature names,
/// and the embedded config updated so `n_estimators` states the combined
/// tree budget.
///
/// # Errors
///
/// * `ClearGbmError::InvalidParameter` if the model's objective is
///   multiclass or ranking, `additional_rounds` is 0, the feature count
///   differs from the model's, or labels/weights fail their validation.
/// * Any tree construction or prediction error.
pub fn continue_gradient_boosting(
    model: &GradientBoostingModel,
    x_train: &[&[f64]],
    y_train: TrainingLabels<'_>,
    sample_weight: Option<&[f64]>,
    validation: Option<ValidationData<'_>>,
    additional_rounds: usize,
    runtime: &TrainingRuntime<'_>,
) -> Result<GradientBoostingModel, ClearGbmError> {
    let hooks = runtime.hooks;
    let config = model.config();

    match config.objective() {
        Objective::BinaryLogLoss | Objective::SquaredError => {}
        Objective::MulticlassSoftmax | Objective::LambdaRank => {
            return Err(ClearGbmError::InvalidParameter {
                name: "objective".to_string(),
                reason: format!(
                    "continued training supports the single-score objectives; \
                     \"{}\" continuation is not implemented",
                    config.objective().as_str()
                ),
            })
        }
    }
    if additional_rounds < 1_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "additional_rounds".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }

    let n_features = propagate!(validate_training_inputs(
        x_train,
        y_train.len(),
        model.feature_names()
    ));
    if let Some(v) = validation {
        propagate!(validate_validation_inputs(v.x, v.y.len(), n_features));
    }

    // Objective/label pairing and content validation, shared with the
    // fresh trainer. The multiclass arm is unreachable here (the
    // objective gate above), so only the single-score resolution can
    // come back.
    let resolved = propagate!(propagate!(resolve_objective(
        config.objective(),
        config.scale_pos_weight(),
        config.n_classes(),
        y_train,
        sample_weight,
        validation,
    ))
    .into_single_score());

    // The continuation's starting scores ARE the existing model's
    // predictions — the same mathematical object LightGBM bakes into the
    // dataset as an init score, kept model-side here.
    let mut raw_preds_train = propagate!(model.predict_raw(x_train));
    let mut raw_preds_val: Vec<f64> = match resolved.val_features() {
        Some(xv) => propagate!(model.predict_raw(xv)),
        None => Vec::new(),
    };

    let prepared = propagate!(prepare_training(x_train, n_features, config));

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

    let existing_trees = model.n_trees();
    let mut run = SingleScoreRounds {
        x_train,
        resolved: &resolved,
        raw_preds_train: &mut raw_preds_train,
        raw_preds_val: &mut raw_preds_val,
        prepared: &prepared,
        config,
        n_rounds: additional_rounds,
        round_offset: existing_trees,
        rng: &mut rng,
        es_state: &mut es_state,
        hooks,
    };
    let new_trees: Vec<Tree> = propagate!(pool.install(|| run_single_score_rounds(&mut run)));

    // Append: the continued artifact carries every tree it predicts with.
    let mut trees: Vec<Tree> = model.trees().to_vec();
    trees.extend(new_trees);

    // Present for every single-score model by construction; the None arm
    // is statically dead (the crate's dead-arm idiom).
    let base = model.base_prediction().unwrap_or(0.0_f64);
    Ok(GradientBoostingModel::new(
        trees,
        BaseScore::Single(base),
        config.learning_rate(),
        model.feature_names().to_vec(),
        config.with_n_estimators(existing_trees + additional_rounds),
    ))
}
