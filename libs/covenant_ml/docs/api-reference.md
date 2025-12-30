# API Reference

Detailed API documentation for covenant-ml.

## Configuration Types

### TrainConfig (XGBoost)

| Field | Type | Description |
|-------|------|-------------|
| `device` | str | `"cpu"`, `"cuda"`, or `"auto"` |
| `learning_rate` | float | Learning rate (alias: eta) |
| `max_depth` | int | Maximum tree depth |
| `n_estimators` | int | Number of boosting rounds |
| `subsample` | float | Row sampling ratio |
| `colsample_bytree` | float | Column sampling ratio |
| `reg_alpha` | float | L1 regularization |
| `reg_lambda` | float | L2 regularization |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_rounds` | int | Rounds without improvement before stopping |
| `scale_pos_weight` | float | Optional: positive class weight for imbalanced data |

### LightGBMConfig

| Field | Type | Description |
|-------|------|-------------|
| `device` | str | `"cpu"`, `"cuda"`, or `"auto"` |
| `learning_rate` | float | Learning rate (alias: eta) |
| `max_depth` | int | Maximum tree depth |
| `n_estimators` | int | Number of boosting rounds |
| `num_leaves` | int | Maximum leaves per tree (LightGBM-specific) |
| `min_child_samples` | int | Minimum samples per leaf (LightGBM-specific) |
| `subsample` | float | Row sampling ratio |
| `colsample_bytree` | float | Column sampling ratio |
| `reg_alpha` | float | L1 regularization |
| `reg_lambda` | float | L2 regularization |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_rounds` | int | Rounds without improvement before stopping |

### ClearGBMConfig

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | int | Number of boosting rounds |
| `max_depth` | int | Maximum tree depth |
| `learning_rate` | float | Shrinkage factor for updates |
| `min_samples_split` | int | Minimum samples to split a node |
| `min_samples_leaf` | int | Minimum samples in a leaf |
| `max_bins` | int | Histogram bins for O(K) split finding (default: 64) |
| `subsample` | float | Row subsampling ratio (1.0 = no subsampling) |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_rounds` | int | Rounds without improvement before stopping |

### MLPConfig

| Field | Type | Description |
|-------|------|-------------|
| `device` | str | `"cpu"`, `"cuda"`, or `"auto"` |
| `precision` | str | `"fp32"`, `"fp16"`, `"bf16"`, or `"auto"` |
| `optimizer` | str | `"adamw"`, `"adam"`, or `"sgd"` |
| `hidden_sizes` | tuple[int, ...] | Hidden layer sizes (e.g., `(64, 32)`) |
| `learning_rate` | float | Learning rate |
| `batch_size` | int | Training batch size |
| `n_epochs` | int | Maximum training epochs |
| `dropout` | float | Dropout rate (0.0-1.0) |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_patience` | int | Epochs without improvement before stopping |

### LSTMConfig

| Field | Type | Description |
|-------|------|-------------|
| `device` | str | `"cpu"`, `"cuda"`, or `"auto"` |
| `precision` | str | `"fp32"`, `"fp16"`, `"bf16"`, or `"auto"` |
| `hidden_size` | int | LSTM hidden state size |
| `num_layers` | int | Number of stacked LSTM layers |
| `dropout` | float | Dropout rate between layers (0.0-1.0) |
| `bidirectional` | bool | Use bidirectional LSTM |
| `sequence_length` | int | Number of time periods per sequence |
| `learning_rate` | float | Learning rate |
| `batch_size` | int | Training batch size |
| `n_epochs` | int | Maximum training epochs |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_patience` | int | Epochs without improvement before stopping |

### LogRegConfig

| Field | Type | Description |
|-------|------|-------------|
| `solver` | str | `"lbfgs"`, `"liblinear"`, `"newton-cg"`, `"newton-cholesky"`, `"sag"`, `"saga"` |
| `penalty` | str | `"l1"`, `"l2"`, `"elasticnet"`, `"none"` |
| `C` | float | Inverse regularization strength (smaller = stronger reg) |
| `max_iter` | int | Maximum solver iterations |
| `tol` | float | Tolerance for stopping criteria |
| `class_weight_balanced` | bool | If True, weights inversely proportional to class frequencies |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `l1_ratio` | float | ElasticNet mixing (0=L2, 1=L1). Only with `penalty="elasticnet"` |

### RandomForestConfig

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | int | Number of trees in the forest |
| `max_depth` | int \| None | Maximum tree depth (None = unlimited) |
| `min_samples_split` | int | Minimum samples to split an internal node |
| `min_samples_leaf` | int | Minimum samples required in a leaf node |
| `max_features` | str \| float \| int \| None | Features per split: `"sqrt"`, `"log2"`, fraction, count, or None |
| `bootstrap` | bool | Whether to use bootstrap samples |
| `class_weight_balanced` | bool | If True, weights inversely proportional to class frequencies |
| `n_jobs` | int | Number of parallel workers (-1 = all cores) |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `oob_score` | bool | Whether to compute out-of-bag score (requires bootstrap=True) |

## Result Types

### TrainOutcome

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | str | Unique model identifier |
| `model_path` | str | Path to saved model file |
| `samples_total` | int | Total samples |
| `samples_train` | int | Training samples |
| `samples_val` | int | Validation samples |
| `samples_test` | int | Test samples |
| `best_val_auc` | float | Best validation AUC |
| `best_round` | int | Round with best AUC |
| `total_rounds` | int | Total training rounds |
| `early_stopped` | bool | Whether training stopped early |
| `train_metrics` | EvalMetrics | Training set metrics |
| `val_metrics` | EvalMetrics | Validation set metrics |
| `test_metrics` | EvalMetrics | Test set metrics |
| `feature_importances` | list[FeatureImportance] | Ranked feature importances |
| `config` | ClassifierTrainConfig | Training configuration |
| `scale_pos_weight_computed` | float | Auto-calculated class weight |

### EvalMetrics

| Field | Type | Description |
|-------|------|-------------|
| `loss` | float | Log loss (cross-entropy) |
| `ppl` | float | Perplexity (exp(loss)) |
| `auc` | float | Area under ROC curve |
| `accuracy` | float | Classification accuracy |
| `precision` | float | Precision for breach class |
| `recall` | float | Recall for breach class |
| `f1_score` | float | F1 score |

### FeatureImportance

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Feature name |
| `importance` | float | Importance score (gain-based) |
| `rank` | int | Rank (1 = most important) |

### TrainProgress

| Field | Type | Description |
|-------|------|-------------|
| `round` | int | Current training round |
| `total_rounds` | int | Total training rounds |
| `train_loss` | float | Training loss |
| `train_auc` | float | Training AUC |
| `val_loss` | float \| None | Validation loss |
| `val_auc` | float \| None | Validation AUC |

## Preprocessing

### Pipeline Steps

| Step | Description | Details |
|------|-------------|---------|
| 1. Special Code Detection | Replace sentinel values with NaN | 96, 98, 99, 999, -1, -9, -999 |
| 2. Outlier Capping | Cap extreme values | 1st/99th percentile bounds |
| 3. Missing Imputation | Fill NaN with median | Per-feature median from training data |
| 4. Z-Score Normalization | Standardize features | Mean=0, std=1 using training stats |

### PreprocessingState

| Field | Type | Description |
|-------|------|-------------|
| `n_features` | int | Number of features |
| `outlier_bounds` | tuple[OutlierBounds, ...] | Per-feature lower/upper bounds |
| `special_codes` | tuple[SpecialCodeSpec, ...] | Per-feature detected special codes |
| `imputation_values` | tuple[ImputationSpec, ...] | Per-feature imputation values |
| `feature_means` | NDArray[np.float64] | Per-feature means for z-score |
| `feature_stds` | NDArray[np.float64] | Per-feature stds for z-score |

### PreprocessedDataSplits

| Attribute | Type | Description |
|-----------|------|-------------|
| `x_train` | NDArray[np.float64] | Preprocessed training features |
| `y_train` | NDArray[np.int64] | Training labels |
| `x_val` | NDArray[np.float64] | Preprocessed validation features |
| `y_val` | NDArray[np.int64] | Validation labels |
| `x_test` | NDArray[np.float64] | Preprocessed test features |
| `y_test` | NDArray[np.int64] | Test labels |
| `state` | PreprocessingState | Fitted preprocessing state |

## Dataset Loading

### DatasetConfig

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Dataset identifier |
| `path` | str | Relative path within data directory |
| `format` | FileFormat | `"csv"` or `"arff"` |
| `target_column` | TargetColumnSpec | Target column config |
| `encoding` | FileEncoding | `"utf-8"`, `"latin-1"`, etc. |
| `description` | str | Human-readable description |

### LoadedDataset

| Field | Type | Description |
|-------|------|-------------|
| `meta` | DatasetMeta | Dataset metadata with statistics |
| `x` | NDArray[np.float64] | Feature matrix (n_samples, n_features) |
| `y` | NDArray[np.int64] | Labels (n_samples,) - 0=healthy, 1=breach |

### DatasetMeta

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Dataset identifier |
| `n_samples` | int | Total number of samples |
| `n_features` | int | Number of feature columns |
| `n_positive` | int | Number of positive class samples |
| `n_negative` | int | Number of negative class samples |
| `positive_ratio` | float | Fraction of positive samples |
| `feature_names` | tuple[str, ...] | Ordered tuple of feature column names |

## Time-Series Dataset Loading

### TimeSeriesSpec

| Field | Type | Description |
|-------|------|-------------|
| `entity_column` | str | Column identifying unique entities |
| `time_column` | str | Column for temporal ordering |
| `aggregation` | AggregationStrategy | `"last"`, `"first"`, `"mean"`, or `"statistics"` |
| `labels_file` | str | Separate CSV file containing entity labels |
| `labels_entity_column` | str | Entity column name in labels file |
| `include_rank_features` | bool | Add per-entity percentile rank features |
| `include_diff_features` | bool | Add row-to-row difference features |
| `include_window_features` | bool | Add window aggregation features |
| `window_sizes` | tuple[int, ...] | Window sizes for window features |

### Aggregation Strategies

| Strategy | Description | Output per Feature |
|----------|-------------|-------------------|
| `"last"` | Take most recent observation | 1 |
| `"first"` | Take oldest observation | 1 |
| `"mean"` | Average all observations | 1 |
| `"statistics"` | Compute mean, std, min, max | 4 |

### Feature Count Formula

For N base features:
- Base aggregation (`"statistics"`): N * 4
- With `include_rank_features=True`: + N
- With `include_diff_features=True`: + N * 5
- With `include_window_features=True, window_sizes=(3, 6)`: + N * 4 * len(window_sizes)

## Cross-Validation

### Types

| Type | Description |
|------|-------------|
| `CVSplit` | Single fold with train/val indices |
| `CVSplitInfo` | All folds with metadata |
| `CVResult` | Complete CV results with OOF predictions |
| `FoldResult` | Single fold training result |
| `FoldTrainer` | Protocol for fold training function |
| `OOFMetrics` | Out-of-fold evaluation metrics |

### Functions

| Function | Description |
|----------|-------------|
| `stratified_kfold_split` | Create stratified train/val splits |
| `group_stratified_kfold_split` | Create group-aware splits (no entity leakage) |
| `run_cross_validation` | Execute k-fold CV with trainer |
| `run_group_cross_validation` | Execute group-aware CV |
| `compute_oof_metrics` | Compute metrics from OOF predictions |
| `get_fold_data` | Extract train/val data for a fold |

## Feature Engineering

### FeatureEngineeringConfig

| Field | Type | Description |
|-------|------|-------------|
| `use_ratios` | bool | Include pairwise ratio features |
| `use_products` | bool | Include pairwise product features |
| `use_log_transforms` | bool | Include log-transformed features |
| `max_ratio_features` | int | Limit ratio features (0 = no limit) |
| `max_product_features` | int | Limit product features (0 = no limit) |

### Presets

| Preset | Ratios | Products | Log | Description |
|--------|--------|----------|-----|-------------|
| `"minimal"` | No | No | Yes | Original + log transforms only |
| `"standard"` | Yes | No | Yes | Default: ratios but no products |
| `"full"` | Yes | Yes | Yes | All transforms enabled |

### Transforms

| Transform | Description | Example |
|-----------|-------------|---------|
| Pairwise Ratios | Xi/Xj for relative relationships | debt_ratio/interest_cover |
| Pairwise Products | Xi*Xj for interaction effects | debt_ratio*current_ratio |
| Log Transforms | log(1 + \|x\|) * sign(x) for skewed data | log_debt_ratio |

## Explainers

### Types

| Type | Description |
|------|-------------|
| `ExplainerRegistry` | Registry of available explainers |
| `ExplainerRegistration` | Registration entry with factory and metadata |
| `FeatureImportanceScore` | TypedDict with name, importance, rank |
| `SupportedExplainer` | Literal type of explainer names |
| `ExplainerCapabilities` | TypedDict with requirements and cost |
| `PermutationConfig` | Config for permutation explainer |
| `GradientConfig` | Config for gradient explainer |
| `IntegratedGradientsConfig` | Config for integrated gradients |

## Hyperparameter Optimization

### OptimizationConfig

| Field | Type | Description |
|-------|------|-------------|
| `n_trials` | int | Number of optimization trials |
| `timeout_seconds` | int \| None | Optional timeout |
| `n_jobs` | int | Parallel jobs (-1 = all cores) |
| `direction` | str | `"maximize"` or `"minimize"` |
| `sampler_seed` | int | Random seed for reproducibility |

### OptimizationSummary

| Field | Type | Description |
|-------|------|-------------|
| `best_value` | float | Best objective value (e.g., AUC) |
| `best_params` | dict | Best hyperparameters found |
| `best_trial_number` | int | Trial number of best result |
| `n_trials` | int | Total trials completed |
| `n_failed` | int | Number of failed trials |
| `duration_seconds` | float | Total optimization time |
| `all_trials` | list[TrialResult] | All trial results |

### Search Space Types

| Type | Description |
|------|-------------|
| `XGBoostSearchSpace` | TypedDict for XGBoost hyperparameters |
| `LightGBMSearchSpace` | TypedDict for LightGBM hyperparameters |
| `ClearGBMSearchSpace` | TypedDict for ClearGBM hyperparameters |
| `MLPSearchSpace` | TypedDict for MLP hyperparameters |
| `LSTMSearchSpace` | TypedDict for LSTM hyperparameters |
| `FloatRangeSpec` | Float parameter range specification |
| `IntRangeSpec` | Integer parameter range specification |

### Optimizer Types

| Type | Description |
|------|-------------|
| `OptunaXGBoostOptimizer` | XGBoost hyperparameter optimizer |
| `OptunaLightGBMOptimizer` | LightGBM hyperparameter optimizer |
| `OptunaClearGBMOptimizer` | ClearGBM hyperparameter optimizer |
| `OptunaMLPOptimizer` | MLP hyperparameter optimizer |
| `OptunaLSTMOptimizer` | LSTM hyperparameter optimizer |

### DART Boosting

Both XGBoost and LightGBM search spaces include DART (Dropouts meet Multiple Additive Regression Trees).

**XGBoost DART Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `booster` | categorical | `"gbtree"`, `"dart"` | Boosting algorithm |
| `rate_drop` | float | 0.0-0.5 | Dropout rate for trees |
| `skip_drop` | float | 0.0-0.5 | Probability of skipping dropout |

**LightGBM DART Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `boosting_type` | categorical | `"gbdt"`, `"dart"` | Boosting type |
| `drop_rate` | float | 0.0-0.5 | Dropout rate for trees |
| `skip_drop` | float | 0.0-0.5 | Probability of skipping dropout |
| `feature_fraction` | float | 0.02-0.1 | Feature subsampling for DART |

## Protocols

| Protocol | Description |
|----------|-------------|
| `ClassifierBackend` | Backend interface (prepare, train, load, predict) |
| `PreparedClassifier` | Prepared classifier ready for training |
| `ClassifierRegistry` | Backend registry used by `BaseTabularTrainer` |
| `XGBModelProtocol` | XGBoost model with predict_proba |
| `XGBBoosterProtocol` | Low-level XGBoost booster |
| `XGBClassifierFactory` | XGBoost classifier constructor |
| `XGBClassifierLoader` | XGBoost model loader |
| `PredictorProtocol` | Any model with predict_proba |

## Core Types Summary

| Type | Description |
|------|-------------|
| `TrainConfig` | XGBoost training configuration |
| `MLPConfig` | MLP neural network configuration |
| `LSTMConfig` | LSTM sequence model configuration |
| `LightGBMConfig` | LightGBM gradient boosting configuration |
| `ClearGBMConfig` | ClearGBM pure-Python gradient boosting configuration |
| `LogRegConfig` | Logistic regression configuration |
| `RandomForestConfig` | Random forest ensemble configuration |
| `ClassifierTrainConfig` | Union of all backend config types |
| `TrainOutcome` | Complete training result |
| `TrainProgress` | Progress update during training |
| `EvalMetrics` | Evaluation metrics for a split |
| `FeatureImportance` | Feature importance entry |
| `DataSplits` | Train/val/test data splits |
| `PreprocessedDataSplits` | Preprocessed splits with state |
| `ProgressCallback` | Callback type for progress updates |
| `BackendName` | Literal: `"xgboost" | "mlp" | "lstm" | "lightgbm" | "cleargbm" | "logreg" | "random_forest"` |

## Manifest Types

TypedDicts for model manifest serialization:

| Type | Description |
|------|-------------|
| `ClassifierManifest` | Complete model manifest with all metadata |
| `ManifestVersions` | Library versions |
| `ManifestSystem` | System info (platform, device, CUDA) |
| `ManifestDataset` | Dataset info (samples, features, distribution) |
| `ManifestTraining` | Training info (backend, config, rounds, duration) |
| `ManifestMetrics` | Train/val/test metrics and best_val_auc |

## Calibration Types

### CalibratorConfig

| Field | Type | Description |
|-------|------|-------------|
| `method` | str | `"isotonic"` or `"platt"` |
| `clip_proba` | bool | Whether to clip probabilities to [eps, 1-eps] |
| `eps` | float | Epsilon for probability clipping (default 1e-10) |

### IsotonicParams

| Field | Type | Description |
|-------|------|-------------|
| `X_thresholds` | list[float] | Sorted input probability thresholds |
| `y_values` | list[float] | Corresponding calibrated probability values |

### PlattParams

| Field | Type | Description |
|-------|------|-------------|
| `A` | float | Slope parameter (typically negative) |
| `B` | float | Intercept parameter |

### CalibratorState

| Field | Type | Description |
|-------|------|-------------|
| `method` | str | `"isotonic"` or `"platt"` |
| `config` | CalibratorConfig | Calibrator configuration |
| `params` | IsotonicParams \| PlattParams | Learned calibration parameters |

### CalibrationResult

| Field | Type | Description |
|-------|------|-------------|
| `state` | CalibratorState | Serializable calibrator state |
| `train_brier_before` | float | Brier score before calibration |
| `train_brier_after` | float | Brier score after calibration |
| `train_ece_before` | float | Expected calibration error before |
| `train_ece_after` | float | Expected calibration error after |

### CalibratedPredictions

| Field | Type | Description |
|-------|------|-------------|
| `raw_proba` | NDArray[np.float64] | Original uncalibrated probabilities |
| `calibrated_proba` | NDArray[np.float64] | Calibrated probabilities |
| `method` | str | Calibration method used |

### Calibration Functions

| Function | Description |
|----------|-------------|
| `create_isotonic_calibrator` | Create isotonic regression calibrator |
| `create_platt_calibrator` | Create Platt scaling calibrator |
| `encode_calibrator_state` | Encode CalibratorState to JSON-compatible dict |
| `decode_calibrator_state` | Decode JSON-compatible dict to CalibratorState |
