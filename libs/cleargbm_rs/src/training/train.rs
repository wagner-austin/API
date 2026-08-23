//! Core training loop for gradient boosting.
//!
//! Orchestrates the full training pipeline: validation, objective
//! resolution, binning, iterative tree construction with gradient/hessian
//! updates, optional early stopping, and model assembly.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::losses::{binary_log_loss_initial_prediction, squared_error_initial_prediction};
use crate::tree::Tree;

use super::config::GradientBoostingConfig;
use super::early_stopping::EarlyStoppingState;
use super::labels::{
    resolve_objective, ResolvedObjective, ResolvedTraining, TrainingLabels, ValidationData,
};
use super::model::{BaseScore, GradientBoostingModel};
use super::parallelism::Parallelism;
use super::rng::SimpleRng;
use super::setup::prepare_training;
use super::single_score_rounds::{run_single_score_rounds, SingleScoreRounds};
use super::train_multiclass::train_multiclass;
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
    // objective/label mismatch is unrepresentable, so each boosting loop
    // dispatches with total matches. The multiclass task gets its own
    // trainer (K score columns, K trees per round); everything else runs
    // the single-score loop below.
    let resolved = match propagate!(resolve_objective(
        config.objective(),
        config.scale_pos_weight(),
        config.n_classes(),
        y_train,
        sample_weight,
        validation,
    )) {
        ResolvedTraining::Multiclass(mc) => {
            return train_multiclass(x_train, n_features, &mc, config, feature_names, runtime)
        }
        ResolvedTraining::SingleScore(resolved) => resolved,
    };

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

    // 5-9. Shared one-time preparation: feature-count-dependent config
    // validation, categorical resolution, binning, tree configuration.
    let prepared = propagate!(prepare_training(x_train, n_features, config));

    // Initialize RNG for subsampling and the early-stopping state.
    let mut rng = SimpleRng::new(config.random_state());
    let mut es_state: Option<EarlyStoppingState> =
        config.early_stopping_rounds().map(EarlyStoppingState::new);

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
    // The rounds themselves live in `single_score_rounds`, shared with the
    // continuation trainer.
    let mut run = SingleScoreRounds {
        x_train,
        resolved: &resolved,
        raw_preds_train: &mut raw_preds_train,
        raw_preds_val: &mut raw_preds_val,
        prepared: &prepared,
        config,
        n_rounds: config.n_estimators(),
        round_offset: 0_usize,
        rng: &mut rng,
        es_state: &mut es_state,
        hooks,
    };
    let trees: Vec<Tree> = propagate!(pool.install(|| run_single_score_rounds(&mut run)));

    Ok(GradientBoostingModel::new(
        trees,
        BaseScore::Single(base_prediction),
        config.learning_rate(),
        feature_names.to_vec(),
        config.clone(),
    ))
}
