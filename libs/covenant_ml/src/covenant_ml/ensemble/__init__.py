"""Ensemble module for combining multiple model predictions.

Provides weighted ensemble that optimizes model weights using OOF predictions
to maximize the AMEX competition metric.

Key components:
- EnsembleOOFData: Container for OOF predictions from multiple models
- EnsembleWeights: Optimized or equal weights for models
- optimize_ensemble_weights: Find optimal weights using scipy
- compute_weighted_predictions: Apply weights to get ensemble predictions

Example:
    >>> from covenant_ml.ensemble import (
    ...     create_oof_data,
    ...     optimize_ensemble_weights,
    ...     compute_weighted_predictions,
    ...     make_default_optimization_config,
    ...     use_real_scipy,
    ... )
    >>> # Set up scipy at application startup
    >>> use_real_scipy()
    >>>
    >>> # Create OOF data from model predictions
    >>> oof_data = create_oof_data(model_predictions, labels)
    >>>
    >>> # Optimize weights
    >>> config = make_default_optimization_config()
    >>> result = optimize_ensemble_weights(oof_data, config)
    >>> best_score = result['best_score']  # Access optimized score
    >>>
    >>> # Apply weights
    >>> ensemble_pred = compute_weighted_predictions(oof_data, result['weights'])
"""

from covenant_ml.ensemble.optimizer import (
    optimize_ensemble_weights,
    set_minimize_hook,
    use_real_scipy,
)
from covenant_ml.ensemble.regression_optimizer import (
    create_regression_equal_weights,
    extract_regression_prediction_matrix,
    optimize_regression_ensemble_weights,
    validate_regression_oof_data,
)
from covenant_ml.ensemble.regression_types import (
    RegressionEnsembleOOFData,
    RegressionOptimizationConfig,
    RegressionOptimizationResult,
    make_default_regression_optimization_config,
)
from covenant_ml.ensemble.types import (
    EnsembleOOFData,
    EnsemblePrediction,
    EnsembleWeights,
    ModelOOFPredictions,
    OptimizationConfig,
    OptimizationResult,
    make_default_optimization_config,
)
from covenant_ml.ensemble.weighted import (
    compute_weighted_predictions,
    create_equal_weights,
    create_oof_data,
    extract_prediction_matrix,
    validate_oof_data,
    validate_weights,
)

__all__ = [
    "EnsembleOOFData",
    "EnsemblePrediction",
    "EnsembleWeights",
    "ModelOOFPredictions",
    "OptimizationConfig",
    "OptimizationResult",
    "RegressionEnsembleOOFData",
    "RegressionOptimizationConfig",
    "RegressionOptimizationResult",
    "compute_weighted_predictions",
    "create_equal_weights",
    "create_oof_data",
    "create_regression_equal_weights",
    "extract_prediction_matrix",
    "extract_regression_prediction_matrix",
    "make_default_optimization_config",
    "make_default_regression_optimization_config",
    "optimize_ensemble_weights",
    "optimize_regression_ensemble_weights",
    "set_minimize_hook",
    "use_real_scipy",
    "validate_oof_data",
    "validate_regression_oof_data",
    "validate_weights",
]
