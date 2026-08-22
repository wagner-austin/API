"""Test hooks for worker components, ML registry injection, and dataset loading.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from covenant_ml.backends.registry import (
    BackendFactory,
    BackendRegistration,
    ClassifierRegistry,
    default_registry,
)
from covenant_ml.datasets import (
    DatasetConfig,
    DatasetRegistry,
    LoadedDataset,
    TimeSeriesDatasetConfig,
    TimeSeriesDatasetRegistry,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.explainers.registry import ExplainerRegistry
from covenant_ml.types import PredictorProtocol

from covenant_radar_api.worker._hook_defaults import (
    _real_data_bank_uploader,
    _real_dataset_loader,
    _real_dataset_registry,
    _real_explainer_registry,
    _real_lightgbm_loader,
    _real_logreg_loader,
    _real_lstm_loader,
    _real_mlp_loader,
    _real_objective_factory,
    _real_optimizer_registry,
    _real_random_forest_loader,
    _real_timeseries_loader,
    _real_timeseries_registry,
)
from covenant_radar_api.worker._hook_protocols import (
    DataBankUploaderProtocol,
    DatasetLoaderCallable,
    DatasetRegistryFactoryProtocol,
    ExplainerRegistryFactoryProtocol,
    LightGBMLoaderProtocol,
    LogRegLoaderProtocol,
    LSTMLoaderProtocol,
    MLPLoaderProtocol,
    ObjectiveFactoryProtocol,
    ObjectiveWithFeatureCount,
    OptimizerRegistryFactoryProtocol,
    RandomForestLoaderProtocol,
    RegistryFactory,
    TimeSeriesLoaderCallable,
    TimeSeriesRegistryFactoryProtocol,
)


def _full_registry() -> ClassifierRegistry:
    """Build registry with all backends including PyTorch (covenant_nn).

    Returns:
        ClassifierRegistry with tree-based backends from covenant_ml
        and neural backends (MLP, LSTM) from covenant_nn.
    """
    reg = default_registry()
    nn_mod = __import__("covenant_nn", fromlist=["create_mlp_backend", "create_lstm_backend"])
    create_mlp: BackendFactory = nn_mod.create_mlp_backend
    create_lstm: BackendFactory = nn_mod.create_lstm_backend
    reg.register("mlp", BackendRegistration(create_mlp))
    reg.register("lstm", BackendRegistration(create_lstm))
    return reg


registry_factory: RegistryFactory = _full_registry

dataset_registry_factory: DatasetRegistryFactoryProtocol = _real_dataset_registry

dataset_loader: DatasetLoaderCallable = _real_dataset_loader

timeseries_registry_factory: TimeSeriesRegistryFactoryProtocol = _real_timeseries_registry

timeseries_loader: TimeSeriesLoaderCallable = _real_timeseries_loader

explainer_registry_factory: ExplainerRegistryFactoryProtocol = _real_explainer_registry

optimizer_registry_factory: OptimizerRegistryFactoryProtocol = _real_optimizer_registry

objective_factory: ObjectiveFactoryProtocol = _real_objective_factory

mlp_loader: MLPLoaderProtocol = _real_mlp_loader

lstm_loader: LSTMLoaderProtocol = _real_lstm_loader

lightgbm_loader: LightGBMLoaderProtocol = _real_lightgbm_loader

logreg_loader: LogRegLoaderProtocol = _real_logreg_loader

random_forest_loader: RandomForestLoaderProtocol = _real_random_forest_loader

data_bank_uploader: DataBankUploaderProtocol = _real_data_bank_uploader

__all__ = [
    "DataBankUploaderProtocol",
    "DatasetConfig",
    "DatasetLoaderCallable",
    "DatasetRegistry",
    "DatasetRegistryFactoryProtocol",
    "ExplainerRegistry",
    "ExplainerRegistryFactoryProtocol",
    "LSTMLoaderProtocol",
    "LightGBMLoaderProtocol",
    "LoadedDataset",
    "LogRegLoaderProtocol",
    "MLPLoaderProtocol",
    "ObjectiveFactoryProtocol",
    "ObjectiveWithFeatureCount",
    "OptimizerRegistryFactoryProtocol",
    "PredictorProtocol",
    "ProgressCallbackProtocol",
    "RandomForestLoaderProtocol",
    "RegistryFactory",
    "TimeSeriesDatasetConfig",
    "TimeSeriesDatasetRegistry",
    "TimeSeriesLoaderCallable",
    "TimeSeriesRegistryFactoryProtocol",
    "data_bank_uploader",
    "dataset_loader",
    "dataset_registry_factory",
    "explainer_registry_factory",
    "lightgbm_loader",
    "logreg_loader",
    "lstm_loader",
    "mlp_loader",
    "objective_factory",
    "optimizer_registry_factory",
    "random_forest_loader",
    "registry_factory",
    "timeseries_loader",
    "timeseries_registry_factory",
]
