# API Reference

Detailed API documentation for covenant-ml. Covers both classification and regression APIs.

## Classification Configuration Types

### TrainConfig (XGBoost Classifier)

| Field | Type | Description |
|-------|------|-------------|
| `device` | `RequestedDevice` | `"cpu"`, `"cuda"`, or `"auto"` |
| `learning_rate` | `float` | Learning rate (alias: eta) |
| `max_depth` | `int` | Maximum tree depth |
| `n_estimators` | `int` | Number of boosting rounds |
| `subsample` | `float` | Row sampling ratio |
| `colsample_bytree` | `float` | Column sampling ratio |
| `reg_alpha` | `float` | L1 regularization |
| `reg_lambda` | `float` | L2 regularization |
| `train_ratio` | `float` | Training set ratio |
| `val_ratio` | `float` | Validation set ratio |
| `test_ratio` | `float` | Test set ratio |
| `random_state` | `int` | Random seed |
| `early_stopping_rounds` | `int` | Rounds without improvement before stopping |
| `scale_pos_weight` | `float` | Optional: positive class weight for imbalanced data |

### LightGBMConfig (Classifier)

| Field | Type | Description |
|-------|------|-------------|
| `device` | `RequestedDevice` | `"cpu"`, `"cuda"`, or `"auto"` |
| `learning_rate` | `float` | Learning rate (alias: eta) |
| `max_depth` | `int` | Maximum tree depth |
| `n_estimators` | `int` | Number of boosting rounds |
| `num_leaves` | `int` | Maximum leaves per tree |
| `min_child_samples` | `int` | Minimum samples per leaf |
| `subsample` | `float` | Row sampling ratio |
| `colsample_bytree` | `float` | Column sampling ratio |
| `reg_alpha` | `float` | L1 regularization |
| `reg_lambda` | `float` | L2 regularization |
| `train_ratio` | `float` | Training set ratio |
| `val_ratio` | `float` | Validation set ratio |
| `test_ratio` | `float` | Test set ratio |
| `random_state` | `int` | Random seed |
| `early_stopping_rounds` | `int` | Rounds without improvement before stopping |

### ClearGBMConfig

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | `int` | Number of boosting rounds |
| `max_depth` | `int` | Maximum tree depth |
| `learning_rate` | `float` | Shrinkage factor for updates |
| `min_samples_split` | `int` | Minimum samples to split a node |
| `min_samples_leaf` | `int` | Minimum samples in a leaf |
| `max_bins` | `int` | Histogram bins for O(K) split finding (default: 64) |
| `subsample` | `float` | Row subsampling ratio (1.0 = no subsampling) |
| `train_ratio` | `float` | Training set ratio |
| `val_ratio` | `float` | Validation set ratio |
| `test_ratio` | `float` | Test set ratio |
| `random_state` | `int` | Random seed |
| `early_stopping_rounds` | `int` | Rounds without improvement before stopping |

### MLPConfig

Used for both MLP classifier (covenant_nn) and MLP regressor (covenant_nn). Config type lives in covenant_ml.

| Field | Type | Description |
|-------|------|-------------|
| `device` | `RequestedDevice` | `"cpu"`, `"cuda"`, or `"auto"` |
| `precision` | `MLPPrecision` | `"fp32"`, `"fp16"`, `"bf16"`, or `"auto"` |
| `optimizer` | `MLPOptimizer` | `"adamw"`, `"adam"`, or `"sgd"` |
| `hidden_sizes` | `tuple[int, ...]` | Hidden layer sizes (e.g., `(64, 32)`) |
| `learning_rate` | `float` | Learning rate |
| `batch_size` | `int` | Training batch size |
| `n_epochs` | `int` | Maximum training epochs |
| `dropout` | `float` | Dropout rate (0.0-1.0) |
| `train_ratio` | `float` | Training set ratio |
| `val_ratio` | `float` | Validation set ratio |
| `test_ratio` | `float` | Test set ratio |
| `random_state` | `int` | Random seed |
| `early_stopping_patience` | `int` | Epochs without improvement before stopping |

### LSTMConfig

Used for both LSTM classifier (covenant_nn) and LSTM regressor (covenant_nn). Config type lives in covenant_ml.

| Field | Type | Description |
|-------|------|-------------|
| `device` | `RequestedDevice` | `"cpu"`, `"cuda"`, or `"auto"` |
| `precision` | `LSTMPrecision` | `"fp32"`, `"fp16"`, `"bf16"`, or `"auto"` |
| `hidden_size` | `int` | LSTM hidden state size |
| `num_layers` | `int` | Number of stacked LSTM layers |
| `dropout` | `float` | Dropout rate between layers (0.0-1.0) |
| `bidirectional` | `bool` | Use bidirectional LSTM |
| `sequence_length` | `int` | Number of time periods per sequence |
| `learning_rate` | `float` | Learning rate |
| `batch_size` | `int` | Training batch size |
| `n_epochs` | `int` | Maximum training epochs |
| `train_ratio` | `float` | Training set ratio |
| `val_ratio` | `float` | Validation set ratio |
| `test_ratio` | `float` | Test set ratio |
| `random_state` | `int` | Random seed |
| `early_stopping_patience` | `int` | Epochs without improvement before stopping |

### LogRegConfig

| Field | Type | Description |
|-------|------|-------------|
| `solver` | `str` | `"lbfgs"`, `"liblinear"`, `"newton-cg"`, `"newton-cholesky"`, `"sag"`, `"saga"` |
| `penalty` | `str` | `"l1"`, `"l2"`, `"elasticnet"`, `"none"` |
| `C` | `float` | Inverse regularization strength (smaller = stronger reg) |
| `max_iter` | `int` | Maximum solver iterations |
| `tol` | `float` | Tolerance for stopping criteria |
| `class_weight_balanced` | `bool` | If True, weights inversely proportional to class frequencies |
| `train_ratio` | `float` | Training set ratio |
| `val_ratio` | `float` | Validation set ratio |
| `test_ratio` | `float` | Test set ratio |
| `random_state` | `int` | Random seed |
| `l1_ratio` | `float` | ElasticNet mixing (0=L2, 1=L1). Only with `penalty="elasticnet"` |

### RandomForestConfig

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | `int` | Number of trees in the forest |
| `max_depth` | `int \| None` | Maximum tree depth (None = unlimited) |
| `min_samples_split` | `int` | Minimum samples to split an internal node |
| `min_samples_leaf` | `int` | Minimum samples required in a leaf node |
| `max_features` | `str \| float \| int \| None` | Features per split: `"sqrt"`, `"log2"`, fraction, count, or None |
| `bootstrap` | `bool` | Whether to use bootstrap samples |
| `class_weight_balanced` | `bool` | If True, weights inversely proportional to class frequencies |
| `n_jobs` | `int` | Number of parallel workers (-1 = all cores) |
| `train_ratio` | `float` | Training set ratio |
| `val_ratio` | `float` | Validation set ratio |
| `test_ratio` | `float` | Test set ratio |
| `random_state` | `int` | Random seed |
| `oob_score` | `bool` | Whether to compute out-of-bag score (requires bootstrap=True) |

## Regression Configuration Types

Regression backends reuse the same config TypedDicts as classification (`TrainConfig`, `LightGBMConfig`, `MLPConfig`, `LSTMConfig`). The `scale_pos_weight` field in `TrainConfig` is ignored for regression (no class imbalance concept).

### RegressorTrainConfig

```python
RegressorTrainConfig = TrainConfig | MLPConfig | LSTMConfig | LightGBMConfig
```

### RegressorBackendName

```python
RegressorBackendName = Literal["xgboost_reg", "lightgbm_reg", "mlp_reg", "lstm_reg"]
```

## Classification Result Types

### TrainOutcome

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | Unique model identifier |
| `model_path` | `str` | Path to saved model file |
| `samples_total` | `int` | Total samples |
| `samples_train` | `int` | Training samples |
| `samples_val` | `int` | Validation samples |
| `samples_test` | `int` | Test samples |
| `best_val_auc` | `float` | Best validation AUC |
| `best_round` | `int` | Round with best AUC |
| `total_rounds` | `int` | Total training rounds |
| `early_stopped` | `bool` | Whether training stopped early |
| `train_metrics` | `EvalMetrics` | Training set metrics |
| `val_metrics` | `EvalMetrics` | Validation set metrics |
| `test_metrics` | `EvalMetrics` | Test set metrics |
| `feature_importances` | `list[FeatureImportance]` | Ranked feature importances |
| `config` | `ClassifierTrainConfig` | Training configuration |
| `scale_pos_weight_computed` | `float` | Auto-calculated class weight |

### EvalMetrics

| Field | Type | Description |
|-------|------|-------------|
| `loss` | `float` | Log loss (cross-entropy) |
| `ppl` | `float` | Perplexity (exp(loss)) |
| `auc` | `float` | Area under ROC curve |
| `accuracy` | `float` | Classification accuracy |
| `precision` | `float` | Precision for breach class |
| `recall` | `float` | Recall for breach class |
| `f1_score` | `float` | F1 score |

### TrainProgress

| Field | Type | Description |
|-------|------|-------------|
| `round` | `int` | Current training round |
| `total_rounds` | `int` | Total training rounds |
| `train_loss` | `float` | Training loss |
| `train_auc` | `float` | Training AUC |
| `val_loss` | `float \| None` | Validation loss |
| `val_auc` | `float \| None` | Validation AUC |

## Regression Result Types

### RegressionTrainOutcome

| Field | Type | Description |
|-------|------|-------------|
| `model_path` | `str` | Path to saved model file |
| `model_id` | `str` | Unique model identifier |
| `samples_total` | `int` | Total samples |
| `samples_train` | `int` | Training samples |
| `samples_val` | `int` | Validation samples |
| `samples_test` | `int` | Test samples |
| `train_metrics` | `RegressionMetrics` | Training set metrics |
| `val_metrics` | `RegressionMetrics` | Validation set metrics |
| `test_metrics` | `RegressionMetrics` | Test set metrics |
| `best_val_rmse` | `float` | Best validation RMSE (lower is better) |
| `best_round` | `int` | Round with best RMSE |
| `total_rounds` | `int` | Total training rounds |
| `early_stopped` | `bool` | Whether training stopped early |
| `config` | `RegressorTrainConfig` | Training configuration |
| `feature_importances` | `list[FeatureImportance]` | Ranked feature importances |

Note: Unlike `TrainOutcome`, there is no `scale_pos_weight_computed` field (regression has no class imbalance).

### RegressionMetrics

| Field | Type | Description |
|-------|------|-------------|
| `mse` | `float` | Mean squared error (lower is better) |
| `rmse` | `float` | Root mean squared error (lower is better) |
| `mae` | `float` | Mean absolute error (lower is better) |
| `r_squared` | `float` | R-squared / coefficient of determination (higher is better, max 1.0) |
| `mape` | `float` | Mean absolute percentage error (lower is better) |

### RegressionTrainProgress

| Field | Type | Description |
|-------|------|-------------|
| `round` | `int` | Current training round |
| `total_rounds` | `int` | Total training rounds |
| `train_rmse` | `float` | Training RMSE |
| `val_rmse` | `float \| None` | Validation RMSE |

### FeatureImportance

Shared by both classification and regression.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Feature name |
| `importance` | `float` | Importance score (gain-based) |
| `rank` | `int` | Rank (1 = most important) |

## Regression Metric Functions

All functions accept `y_true: NDArray[np.float64]` and `y_pred: NDArray[np.float64]`:

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_mse(y_true, y_pred)` | `float` | Mean squared error |
| `compute_rmse(y_true, y_pred)` | `float` | Root mean squared error |
| `compute_mae(y_true, y_pred)` | `float` | Mean absolute error |
| `compute_r_squared(y_true, y_pred)` | `float` | R-squared (coefficient of determination) |
| `compute_mape(y_true, y_pred, eps=1e-8)` | `float` | Mean absolute percentage error |
| `compute_all_regression_metrics(y_true, y_pred)` | `RegressionMetrics` | All 5 metrics at once |
| `format_regression_metrics_str(metrics)` | `str` | Human-readable string |

## Regression Trainer

```python
def train_regression_model_with_validation(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    config: TrainConfig,
    output_dir: Path,
    feature_names: list[str],
    progress_callback: RegressorProgressCallback | None = None,
) -> RegressionTrainOutcome
```

Uses `objective="reg:squarederror"`, `eval_metric="rmse"`. Early stops on val RMSE (lower is better).

### RegressionDataSplits

```python
def regression_split(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> RegressionDataSplits
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `x_train` | `NDArray[np.float64]` | Training features |
| `y_train` | `NDArray[np.float64]` | Training targets (float64) |
| `x_val` | `NDArray[np.float64]` | Validation features |
| `y_val` | `NDArray[np.float64]` | Validation targets |
| `x_test` | `NDArray[np.float64]` | Test features |
| `y_test` | `NDArray[np.float64]` | Test targets |

## Protocols

### Classification Protocols

| Protocol | Description |
|----------|-------------|
| `ClassifierBackend` | Backend interface (prepare, train, evaluate, save, load) |
| `PreparedClassifier` | Trained classifier with `predict_proba()` |
| `ClassifierRegistry` | Backend registry for classifier backends |
| `ProgressCallback` | `Callable[[TrainProgress], None]` |
| `XGBModelProtocol` | XGBoost model with predict_proba |
| `XGBBoosterProtocol` | Low-level XGBoost booster |
| `XGBClassifierFactory` | XGBoost classifier constructor |

### Regression Protocols

| Protocol | Description |
|----------|-------------|
| `RegressorBackend` | Backend interface (prepare, train, evaluate, save, load) |
| `PreparedRegressor` | Trained regressor with `predict()` → 1D array |
| `RegressorRegistry` | Backend registry for regressor backends |
| `RegressorProgressCallback` | `Callable[[RegressionTrainProgress], None]` |
| `XGBRegressorModelProtocol` | XGBoost regressor model with predict |
| `XGBRegressorFactory` | XGBoost regressor constructor |
| `RegressorBackendFactory` | Factory protocol for creating regressor backends |

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
| `n_features` | `int` | Number of features |
| `outlier_bounds` | `tuple[OutlierBounds, ...]` | Per-feature lower/upper bounds |
| `special_codes` | `tuple[SpecialCodeSpec, ...]` | Per-feature detected special codes |
| `imputation_values` | `tuple[ImputationSpec, ...]` | Per-feature imputation values |
| `feature_means` | `NDArray[np.float64]` | Per-feature means for z-score |
| `feature_stds` | `NDArray[np.float64]` | Per-feature stds for z-score |

### PreprocessedDataSplits

| Attribute | Type | Description |
|-----------|------|-------------|
| `x_train` | `NDArray[np.float64]` | Preprocessed training features |
| `y_train` | `NDArray[np.int64]` | Training labels |
| `x_val` | `NDArray[np.float64]` | Preprocessed validation features |
| `y_val` | `NDArray[np.int64]` | Validation labels |
| `x_test` | `NDArray[np.float64]` | Preprocessed test features |
| `y_test` | `NDArray[np.int64]` | Test labels |
| `state` | `PreprocessingState` | Fitted preprocessing state |

## Dataset Loading

### DatasetConfig

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Dataset identifier |
| `path` | `str` | Relative path within data directory |
| `format` | `FileFormat` | `"csv"` or `"arff"` |
| `target_column` | `TargetColumnSpec` | Target column config |
| `encoding` | `FileEncoding` | `"utf-8"`, `"latin-1"`, etc. |
| `description` | `str` | Human-readable description |

### LoadedDataset (Classification)

| Field | Type | Description |
|-------|------|-------------|
| `meta` | `DatasetMeta` | Dataset metadata with statistics |
| `x` | `NDArray[np.float64]` | Feature matrix (n_samples, n_features) |
| `y` | `NDArray[np.int64]` | Labels (n_samples,) — 0=healthy, 1=breach |

### RegressionLoadedDataset

| Field | Type | Description |
|-------|------|-------------|
| `meta` | `DatasetMeta` | Dataset metadata with statistics |
| `x` | `NDArray[np.float64]` | Feature matrix (n_samples, n_features) |
| `y` | `NDArray[np.float64]` | Continuous targets (n_samples,) |

### DatasetMeta

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Dataset identifier |
| `n_samples` | `int` | Total number of samples |
| `n_features` | `int` | Number of feature columns |
| `n_positive` | `int` | Number of positive class samples |
| `n_negative` | `int` | Number of negative class samples |
| `positive_ratio` | `float` | Fraction of positive samples |
| `feature_names` | `tuple[str, ...]` | Ordered tuple of feature column names |

### Regression Test Dataset

```python
from covenant_ml.datasets.testing import create_fake_regression_dataset_loader

loader = create_fake_regression_dataset_loader(n_samples=200, n_features=10, random_state=42)
dataset = loader.load(config, Path("/fake"))
# dataset["x"]: NDArray[np.float64] shape (200, 10)
# dataset["y"]: NDArray[np.float64] shape (200,)
```

## Time-Series Dataset Loading

### TimeSeriesSpec

| Field | Type | Description |
|-------|------|-------------|
| `entity_column` | `str` | Column identifying unique entities |
| `time_column` | `str` | Column for temporal ordering |
| `aggregation` | `AggregationStrategy` | `"last"`, `"first"`, `"mean"`, or `"statistics"` |
| `labels_file` | `str` | Separate CSV file containing entity labels |
| `labels_entity_column` | `str` | Entity column name in labels file |
| `include_rank_features` | `bool` | Add per-entity percentile rank features |
| `include_diff_features` | `bool` | Add row-to-row difference features |
| `include_window_features` | `bool` | Add window aggregation features |
| `window_sizes` | `tuple[int, ...]` | Window sizes for window features |

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

## Classification Cross-Validation

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

## Regression Cross-Validation

### Types

| Type | Description |
|------|-------------|
| `RegressionFoldResult` | Single fold result with train/val RMSE and predictions |
| `RegressionCVResult` | Complete CV result with per-fold RMSE, mean/std, OOF predictions |

### RegressionFoldResult

| Field | Type | Description |
|-------|------|-------------|
| `fold_number` | `int` | Fold index |
| `train_rmse` | `float` | Training RMSE for this fold |
| `val_rmse` | `float` | Validation RMSE for this fold |
| `val_indices` | `NDArray[np.intp]` | Validation sample indices |
| `val_predictions` | `NDArray[np.float64]` | Predictions on validation set |

### RegressionCVResult

| Field | Type | Description |
|-------|------|-------------|
| `n_folds` | `int` | Number of folds |
| `fold_results` | `tuple[RegressionFoldResult, ...]` | Per-fold results |
| `mean_val_rmse` | `float` | Mean validation RMSE across folds |
| `std_val_rmse` | `float` | Std of validation RMSE across folds |
| `oof_predictions` | `NDArray[np.float64]` | Out-of-fold predictions |

### Functions

| Function | Description |
|----------|-------------|
| `kfold_split` | Create KFold splits (not stratified) |
| `get_regression_fold_data` | Extract train/val data for a fold |
| `run_regression_cross_validation` | Execute k-fold regression CV |

### FoldRegressorTrainer Protocol

```python
class FoldRegressorTrainer(Protocol):
    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        fold_number: int,
    ) -> TrainedRegressor: ...
```

## Classification Ensemble

### Types

| Type | Description |
|------|-------------|
| `EnsembleOOFData` | Out-of-fold data for ensemble weight optimization |
| `OptimizationConfig` | Config for weight optimization (metric, method, tolerance) |
| `OptimizationResult` | Optimization result with weights and convergence info |
| `EnsembleWeights` | Model name to weight mapping |

### Functions

| Function | Description |
|----------|-------------|
| `optimize_ensemble_weights` | Optimize weights to maximize AUC |
| `validate_oof_data` | Validate OOF data structure |
| `compute_weighted_predictions` | Apply weights to model predictions |

## Regression Ensemble

### RegressionEnsembleOOFData

| Field | Type | Description |
|-------|------|-------------|
| `model_predictions` | `tuple[ModelOOFPredictions, ...]` | Per-model OOF predictions |
| `labels` | `NDArray[np.float64]` | True continuous targets |
| `n_samples` | `int` | Total samples |
| `n_models` | `int` | Number of models |

### RegressionOptimizationConfig

| Field | Type | Description |
|-------|------|-------------|
| `metric` | `Literal["neg_rmse", "neg_mae", "r_squared"]` | Optimization objective |
| `method` | `Literal["SLSQP", "trust-constr"]` | Scipy minimize method |
| `max_iterations` | `int` | Maximum optimizer iterations |
| `tolerance` | `float` | Convergence tolerance |
| `random_state` | `int` | Random seed |

### RegressionOptimizationResult

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `EnsembleWeights` | Optimized model weights |
| `best_score` | `float` | Best objective value |
| `n_iterations` | `int` | Iterations taken |
| `converged` | `bool` | Whether optimization converged |
| `initial_score` | `float` | Score with equal weights |

### Functions

| Function | Description |
|----------|-------------|
| `optimize_regression_ensemble_weights` | Optimize weights on regression OOF data |
| `validate_regression_oof_data` | Validate regression OOF structure |
| `extract_regression_prediction_matrix` | Extract prediction matrix from OOF data |
| `create_regression_equal_weights` | Create uniform weights |
| `make_default_regression_optimization_config` | Default config (neg_rmse, SLSQP) |

## Feature Engineering

### FeatureEngineeringConfig

| Field | Type | Description |
|-------|------|-------------|
| `use_ratios` | `bool` | Include pairwise ratio features |
| `use_products` | `bool` | Include pairwise product features |
| `use_log_transforms` | `bool` | Include log-transformed features |
| `max_ratio_features` | `int` | Limit ratio features (0 = no limit) |
| `max_product_features` | `int` | Limit product features (0 = no limit) |

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
| `GradientConfig` | Config for gradient explainer (covenant_nn) |
| `IntegratedGradientsConfig` | Config for integrated gradients (covenant_nn) |

## Hyperparameter Optimization

### OptimizationConfig

| Field | Type | Description |
|-------|------|-------------|
| `n_trials` | `int` | Number of optimization trials |
| `timeout_seconds` | `int \| None` | Optional timeout |
| `n_jobs` | `int` | Parallel jobs (-1 = all cores) |
| `direction` | `str` | `"maximize"` or `"minimize"` |
| `sampler_seed` | `int` | Random seed for reproducibility |

### OptimizationSummary

| Field | Type | Description |
|-------|------|-------------|
| `best_value` | `float` | Best objective value (e.g., AUC or neg RMSE) |
| `best_params` | `dict` | Best hyperparameters found |
| `best_trial_number` | `int` | Trial number of best result |
| `n_trials` | `int` | Total trials completed |
| `n_failed` | `int` | Number of failed trials |
| `duration_seconds` | `float` | Total optimization time |
| `all_trials` | `list[TrialResult]` | All trial results |

### Classification Optimizers

| Type | Description |
|------|-------------|
| `OptunaXGBoostOptimizer` | XGBoost classifier optimizer |
| `OptunaLightGBMOptimizer` | LightGBM classifier optimizer |
| `OptunaClearGBMOptimizer` | ClearGBM classifier optimizer |

MLP and LSTM classification optimizers live in `covenant_nn`.

### Regression Objectives

| Class | Library | Backend |
|-------|---------|---------|
| `XGBoostRegressorObjective` | covenant_ml | XGBoost |
| `LightGBMRegressorObjective` | covenant_ml | LightGBM |
| `MLPRegressorObjective` | covenant_nn | MLP |
| `LSTMRegressorObjective` | covenant_nn | LSTM |

All regression objectives return negative RMSE (higher = better for Optuna maximization).

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

## Backend Name Types

### Classification

```python
BackendName = Literal["xgboost", "mlp", "lstm", "lightgbm", "cleargbm", "logreg", "random_forest"]
```

Note: `"mlp"` and `"lstm"` backends live in `covenant_nn`, not `covenant_ml`.

### Regression

```python
RegressorBackendName = Literal["xgboost_reg", "lightgbm_reg", "mlp_reg", "lstm_reg"]
```

`"mlp_reg"` and `"lstm_reg"` backends live in `covenant_nn`.

## Config Union Types

### Classification

```python
ClassifierTrainConfig = (
    TrainConfig | MLPConfig | LSTMConfig | LightGBMConfig | ClearGBMConfig | LogRegConfig | RandomForestConfig
)
```

### Regression

```python
RegressorTrainConfig = TrainConfig | MLPConfig | LSTMConfig | LightGBMConfig
```

## Calibration Types

### CalibratorConfig

| Field | Type | Description |
|-------|------|-------------|
| `method` | `str` | `"isotonic"` or `"platt"` |
| `clip_proba` | `bool` | Whether to clip probabilities to [eps, 1-eps] |
| `eps` | `float` | Epsilon for probability clipping (default 1e-10) |

### IsotonicParams

| Field | Type | Description |
|-------|------|-------------|
| `X_thresholds` | `list[float]` | Sorted input probability thresholds |
| `y_values` | `list[float]` | Corresponding calibrated probability values |

### PlattParams

| Field | Type | Description |
|-------|------|-------------|
| `A` | `float` | Slope parameter (typically negative) |
| `B` | `float` | Intercept parameter |

### CalibratorState

| Field | Type | Description |
|-------|------|-------------|
| `method` | `str` | `"isotonic"` or `"platt"` |
| `config` | `CalibratorConfig` | Calibrator configuration |
| `params` | `IsotonicParams \| PlattParams` | Learned calibration parameters |

### CalibrationResult

| Field | Type | Description |
|-------|------|-------------|
| `state` | `CalibratorState` | Serializable calibrator state |
| `train_brier_before` | `float` | Brier score before calibration |
| `train_brier_after` | `float` | Brier score after calibration |
| `train_ece_before` | `float` | Expected calibration error before |
| `train_ece_after` | `float` | Expected calibration error after |

### CalibratedPredictions

| Field | Type | Description |
|-------|------|-------------|
| `raw_proba` | `NDArray[np.float64]` | Original uncalibrated probabilities |
| `calibrated_proba` | `NDArray[np.float64]` | Calibrated probabilities |
| `method` | `str` | Calibration method used |

### Calibration Functions

| Function | Description |
|----------|-------------|
| `create_isotonic_calibrator` | Create isotonic regression calibrator |
| `create_platt_calibrator` | Create Platt scaling calibrator |
| `encode_calibrator_state` | Encode CalibratorState to JSON-compatible dict |
| `decode_calibrator_state` | Decode JSON-compatible dict to CalibratorState |

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

## Testing Utilities

### Classification Config Factories

| Function | Returns | Description |
|----------|---------|-------------|
| `make_xgboost_config(...)` | `TrainConfig` | XGBoost classifier test config |
| `make_lightgbm_config(...)` | `LightGBMConfig` | LightGBM classifier test config |
| `make_mlp_config(...)` | `MLPConfig` | MLP classifier test config |
| `make_lstm_config(...)` | `LSTMConfig` | LSTM classifier test config |

### Regression Config Factories

| Function | Returns | Description |
|----------|---------|-------------|
| `make_xgboost_regressor_config(...)` | `TrainConfig` | XGBoost regressor test config |
| `make_lightgbm_regressor_config(...)` | `LightGBMConfig` | LightGBM regressor test config |
| `make_mlp_regressor_config(...)` | `MLPConfig` | MLP regressor test config |
| `make_lstm_regressor_config(...)` | `LSTMConfig` | LSTM regressor test config |

### Hooks

| Function | Description |
|----------|-------------|
| `set_cuda_hook(hook)` | Override CUDA availability check |
| `set_torch_device_hook(hook)` | Override torch.device creation |
| `set_torch_cuda_is_available_hook(hook)` | Override torch.cuda.is_available |

## Temporal Feature Extraction (McKinnon PNAS 2024)

### TemporalFeatureConfig

| Field | Type | Description |
|-------|------|-------------|
| `n_fourier_harmonics` | `int` | Number of Fourier harmonics for seasonal cycle removal (McKinnon uses 5) |
| `hot_cutoff_percentile` | `float` | Percentile for hot-tail threshold (0-100 exclusive) |
| `cold_cutoff_percentile` | `float` | Percentile for cold-tail threshold (0-100 exclusive) |
| `season` | `SeasonDefinition` | `"warm"`, `"cold"`, or `"full_year"` |
| `season_months` | `tuple[int, ...]` | Month numbers defining the season (1-12) |
| `compute_ar1` | `bool` | Whether to compute lag-1 autocorrelation metric |

### SeasonalCycleCoefficients

| Field | Type | Description |
|-------|------|-------------|
| `n_harmonics` | `int` | Number of harmonics used |
| `cos_coefficients` | `tuple[tuple[float, ...], ...]` | Cosine coefficients, shape (n_harmonics, n_locations) |
| `sin_coefficients` | `tuple[tuple[float, ...], ...]` | Sine coefficients, shape (n_harmonics, n_locations) |
| `mean` | `tuple[float, ...]` | Mean value per location |
| `n_days_per_year` | `int` | Days per year for frequency calculation (365) |

### TailThresholds

| Field | Type | Description |
|-------|------|-------------|
| `hot_threshold` | `tuple[float, ...]` | Hot-tail threshold per location |
| `cold_threshold` | `tuple[float, ...]` | Cold-tail threshold per location |
| `hot_percentile` | `float` | Percentile used for hot threshold |
| `cold_percentile` | `float` | Percentile used for cold threshold |

### TemporalFeatureState

| Field | Type | Description |
|-------|------|-------------|
| `config` | `TemporalFeatureConfig` | Configuration used for fitting |
| `seasonal_cycle` | `SeasonalCycleCoefficients` | Fitted Fourier coefficients |
| `thresholds` | `TailThresholds` | Pre-computed tail thresholds |
| `median_baseline` | `tuple[float, ...]` | Mean of within-season medians per location |
| `n_locations` | `int` | Number of spatial locations |

### HeatMetricResult

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | `str` | Entity identifier |
| `n_years` | `int` | Number of years computed |
| `metric_names` | `tuple[str, ...]` | Ordered tuple of computed metric names |
| `values` | `tuple[tuple[float, ...], ...]` | Metric values, shape (n_years, n_metrics) |

### Heat Metric Names

9 canonical heat metrics (from `HEAT_METRIC_NAMES`):

| Metric | Description | Ranking Group |
|--------|-------------|---------------|
| `seasonal_max` | Maximum daily value in season | HOT (negate) |
| `seasonal_min` | Minimum daily value in season | COLD (direct) |
| `cum_excess_hot` | Cumulative excess above hot threshold | HOT (negate) |
| `avg_excess_hot` | Average excess above hot threshold | HOT (negate) |
| `ndays_excess_hot` | Days exceeding hot threshold | HOT (negate) |
| `cum_excess_cold` | Cumulative excess below cold threshold | COLD (direct) |
| `avg_excess_cold` | Average excess below cold threshold | COLD (direct) |
| `ndays_excess_cold` | Days exceeding cold threshold | HOT (negate) |
| `ar1` | Lag-1 autocorrelation of residuals | HOT (negate) |

### Temporal Feature Functions

| Function | Description |
|----------|-------------|
| `fit_seasonal_cycle(values, doy, n_harmonics)` | Fit Fourier seasonal cycle coefficients |
| `remove_seasonal_cycle(values, doy, coefficients)` | Remove fitted seasonal cycle from data |
| `compute_within_season_medians(residuals, year_labels)` | Compute median residuals per year per location |
| `compute_residuals(residuals, medians, year_labels)` | Subtract within-season medians from residuals |
| `fit_tail_thresholds(residuals, config)` | Compute hot/cold percentile thresholds |
| `compute_heat_metrics(residuals, year_labels, thresholds, config)` | Compute all 9 heat metrics |
| `fit_temporal_features(values, doy, year_labels, config)` | Fit complete pipeline, return state |
| `transform_temporal_features(values, doy, year_labels, state)` | Transform using fitted state |
| `build_temporal_feature_names(config)` | Build ordered list of feature names |

## Rank-Trend Hypothesis Testing (McKinnon PNAS 2024, Steps 4-7)

### RankTrendConfig

| Field | Type | Description |
|-------|------|-------------|
| `n_null_samples` | `int` | Number of Monte Carlo permutation samples (must be >= 1) |
| `random_seed` | `int` | Random seed for reproducibility (must be >= 0) |

### MetricTrendResult

| Field | Type | Description |
|-------|------|-------------|
| `metric_name` | `str` | Name of the heat metric tested |
| `observed_slope` | `float` | OLS slope of rank-vs-year regression |
| `p_value` | `float` | Two-sided p-value from null distribution |
| `is_significant` | `bool` | Whether p_value < 0.05 |
| `n_years` | `int` | Number of years in the time series |
| `spatial_dof` | `int` | Estimated spatial degrees of freedom (Bretherton et al. 1999) |

### RankTrendResult

| Field | Type | Description |
|-------|------|-------------|
| `metric_results` | `tuple[MetricTrendResult, ...]` | Per-metric trend test results |
| `n_null_samples` | `int` | Number of Monte Carlo samples used |
| `random_seed` | `int` | Random seed used for reproducibility |

### Rank-Trend Functions

| Function | Description |
|----------|-------------|
| `compute_ols_slope(x, y)` | Manual OLS regression slope |
| `rank_metric_series(values, negate)` | Convert 1D values to ranks (1 = most extreme) |
| `rank_heat_metrics(metrics, metric_names)` | Rank all metrics with HOT/COLD sign conventions + composite averages |
| `compute_latitude_weights(latitudes)` | Cosine-based area weights normalized to sum to 1 |
| `compute_weighted_spatial_mean(values, weights)` | Weighted mean across spatial locations |
| `estimate_spatial_dof(rank_series, weights)` | Bretherton et al. (1999) DOF: trace(C)^2 / \|\|C\|\|_F^2 |
| `generate_null_trend_slopes(dof, n_years, n_samples, seed)` | Monte Carlo null distribution of rank trend slopes |
| `compute_trend_pvalue(observed_slope, null_slopes)` | Two-sided p-value from null distribution |
| `run_rank_trend_analysis(metrics, metric_names, latitudes, config)` | Full analysis orchestrator (steps 4-7) |

### Encode/Decode/Require Functions

All temporal and rank-trend TypedDicts have full JSON serialization support:

| Function | Description |
|----------|-------------|
| `encode_temporal_feature_state(state)` | Encode to JSON-serializable dict |
| `require_temporal_feature_state(data)` | Validate and extract from parsed JSON |
| `encode_heat_metric_result(result)` | Encode to JSON-serializable dict |
| `require_heat_metric_result(data)` | Validate and extract from parsed JSON |
| `encode_metric_trend_result(result)` | Encode to JSON-serializable dict |
| `require_metric_trend_result(data)` | Validate and extract from parsed JSON |
| `encode_rank_trend_result(result)` | Encode to JSON-serializable dict |
| `require_rank_trend_result(data)` | Validate and extract from parsed JSON |
| `make_rank_trend_config(n_null_samples, random_seed)` | Factory with validation |
| `require_rank_trend_config(data, key)` | Validate and extract from parsed JSON |
| `make_metric_trend_result(...)` | Factory with validation |
| `make_rank_trend_result(...)` | Factory with validation |

### Synthetic Test Data Factories

| Function | Description |
|----------|-------------|
| `create_synthetic_daily_timeseries(n_years, n_locations, n_harmonics, seed)` | Multi-location daily data with known Fourier seasonal cycle + noise |
| `create_synthetic_trending_metrics(n_years, n_locations, seed)` | Multi-location metrics with known linear trends for rank-trend testing |
