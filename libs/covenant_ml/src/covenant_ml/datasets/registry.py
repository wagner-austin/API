"""Dataset registry for the pluggable dataset loading system.

Provides registries of known dataset configurations.
Immutable after construction, thread-safe for reads.

Registries:
    DatasetRegistry: For standard DatasetConfig (CSV, ARFF)
    TimeSeriesDatasetRegistry: For TimeSeriesDatasetConfig (time-series CSV)
"""

from __future__ import annotations

from covenant_ml.datasets.types import (
    DatasetConfig,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)


class DatasetRegistry:
    """Registry of known dataset configurations.

    Immutable after construction. Thread-safe for reads.
    Provides lookup by dataset name with strict validation.
    """

    def __init__(self, configs: tuple[DatasetConfig, ...]) -> None:
        """Initialize with a tuple of dataset configs.

        Args:
            configs: Immutable tuple of DatasetConfig entries.

        Raises:
            ValueError: If duplicate dataset names found.
        """
        self._configs: dict[str, DatasetConfig] = {}
        for cfg in configs:
            name = cfg["name"]
            if name in self._configs:
                raise ValueError(f"Duplicate dataset name: {name}")
            self._configs[name] = cfg

    def get(self, name: str) -> DatasetConfig:
        """Get configuration for a dataset by name.

        Args:
            name: Dataset name (e.g., "kaggle_company_bankruptcy").

        Returns:
            DatasetConfig for the requested dataset.

        Raises:
            KeyError: If dataset not found in registry.
        """
        if name not in self._configs:
            available = ", ".join(sorted(self._configs.keys()))
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")
        return self._configs[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered dataset names.

        Returns:
            Sorted tuple of dataset names.
        """
        return tuple(sorted(self._configs.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if dataset is registered.

        Args:
            name: Dataset name to check.

        Returns:
            True if dataset is in registry.
        """
        return name in self._configs

    def __len__(self) -> int:
        """Get number of registered datasets.

        Returns:
            Number of datasets in registry.
        """
        return len(self._configs)


class TimeSeriesDatasetRegistry:
    """Registry of time-series dataset configurations.

    Stores TimeSeriesDatasetConfig entries for datasets with
    multiple observations per entity over time.

    Immutable after construction. Thread-safe for reads.
    """

    def __init__(self, configs: tuple[TimeSeriesDatasetConfig, ...]) -> None:
        """Initialize with a tuple of time-series dataset configs.

        Args:
            configs: Immutable tuple of TimeSeriesDatasetConfig entries.

        Raises:
            ValueError: If duplicate dataset names found.
        """
        self._configs: dict[str, TimeSeriesDatasetConfig] = {}
        for cfg in configs:
            name = cfg["name"]
            if name in self._configs:
                raise ValueError(f"Duplicate dataset name: {name}")
            self._configs[name] = cfg

    def get(self, name: str) -> TimeSeriesDatasetConfig:
        """Get time-series configuration for a dataset by name.

        Args:
            name: Dataset name (e.g., "kaggle_amex_default").

        Returns:
            TimeSeriesDatasetConfig for the requested dataset.

        Raises:
            KeyError: If dataset not found in registry.
        """
        if name not in self._configs:
            available = ", ".join(sorted(self._configs.keys()))
            raise KeyError(f"Time-series dataset '{name}' not found. Available: {available}")
        return self._configs[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered time-series dataset names.

        Returns:
            Sorted tuple of dataset names.
        """
        return tuple(sorted(self._configs.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if time-series dataset is registered.

        Args:
            name: Dataset name to check.

        Returns:
            True if dataset is in registry.
        """
        return name in self._configs

    def __len__(self) -> int:
        """Get number of registered time-series datasets.

        Returns:
            Number of datasets in registry.
        """
        return len(self._configs)


def make_default_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Create registry with verified time-series dataset configurations.

    Returns:
        TimeSeriesDatasetRegistry with production configs.
    """
    return TimeSeriesDatasetRegistry(_VERIFIED_TIMESERIES_CONFIGS)


# Verified time-series dataset configurations
_VERIFIED_TIMESERIES_CONFIGS: tuple[TimeSeriesDatasetConfig, ...] = (
    # AMEX Default Prediction (Kaggle competition)
    # ~458K customers, 188 features, ~13 time steps per customer
    TimeSeriesDatasetConfig(
        name="kaggle_amex_default",
        display_name="AMEX Default Prediction",
        folder="kaggle_amex_default",
        file_name="train_data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=458913,
        n_features_expected=188,
        positive_class_ratio_expected=0.26,
        time_series=TimeSeriesSpec(
            entity_column="customer_ID",
            time_column="S_2",
            aggregation="last",
            labels_file="train_labels.csv",
            labels_entity_column="customer_ID",
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=False,
            window_sizes=(),
        ),
    ),
)


def make_default_registry() -> DatasetRegistry:
    """Create registry with all verified dataset configurations.

    Returns:
        DatasetRegistry with production dataset configs.
    """
    return DatasetRegistry(_VERIFIED_CONFIGS)


# Verified dataset configurations (generated by discovery script, reviewed by human)
_VERIFIED_CONFIGS: tuple[DatasetConfig, ...] = (
    # Taiwan bankruptcy (original)
    DatasetConfig(
        name="taiwan",
        display_name="Taiwan Bankruptcy (Original)",
        folder="taiwan_data",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="Bankrupt?",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=6819,
        n_features_expected=95,
        positive_class_ratio_expected=0.033,
    ),
    # US bankruptcy (original)
    DatasetConfig(
        name="us",
        display_name="US Bankruptcy (Original)",
        folder="us_data",
        file_name="american_bankruptcy.csv",
        file_format="csv",
        encoding="utf-8-sig",
        target=TargetColumnSpec(
            column_name="status_label",
            label_type="binary_str",
            positive_values=("failed",),
            negative_values=("alive",),
        ),
        exclude_columns=("company_name", "year"),
        n_samples_expected=78682,
        n_features_expected=18,
        positive_class_ratio_expected=0.025,
    ),
    # Polish bankruptcy (original)
    DatasetConfig(
        name="polish",
        display_name="Polish Bankruptcy (Original)",
        folder="polish_data",
        file_name="1year.arff",
        file_format="arff",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="class",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=7027,
        n_features_expected=64,
        positive_class_ratio_expected=0.043,
    ),
    # Kaggle company bankruptcy (Taiwan copy)
    DatasetConfig(
        name="kaggle_company_bankruptcy",
        display_name="Kaggle Company Bankruptcy",
        folder="kaggle_company_bankruptcy",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="Bankrupt?",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=6819,
        n_features_expected=95,
        positive_class_ratio_expected=0.033,
    ),
    # UCI Credit Card Default (Taiwan credit card clients)
    DatasetConfig(
        name="kaggle_credit_default",
        display_name="UCI Credit Card Default",
        folder="kaggle_credit_default",
        file_name="UCI_Credit_Card.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="default.payment.next.month",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=("ID",),
        n_samples_expected=30000,
        n_features_expected=23,
        positive_class_ratio_expected=0.221,
    ),
    # Credit Risk Dataset
    DatasetConfig(
        name="kaggle_credit_risk",
        display_name="Credit Risk Dataset",
        folder="kaggle_credit_risk",
        file_name="credit_risk_dataset.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="loan_status",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=32581,
        n_features_expected=11,
        positive_class_ratio_expected=0.218,
    ),
    # HELOC (Home Equity Line of Credit)
    DatasetConfig(
        name="kaggle_heloc",
        display_name="HELOC Risk Performance",
        folder="kaggle_heloc",
        file_name="heloc_dataset_v1 (1).csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="RiskPerformance",
            label_type="binary_str",
            positive_values=("Bad",),
            negative_values=("Good",),
        ),
        exclude_columns=(),
        n_samples_expected=10459,
        n_features_expected=23,
        positive_class_ratio_expected=0.522,
    ),
    # Give Me Credit (Kaggle competition)
    DatasetConfig(
        name="kaggle_give_me_credit",
        display_name="Give Me Credit",
        folder="kaggle_give_me_credit",
        file_name="cs-training.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="SeriousDlqin2yrs",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=("",),  # Unnamed index column
        n_samples_expected=150000,
        n_features_expected=10,
        positive_class_ratio_expected=0.067,
    ),
    # Loan Default Dataset
    DatasetConfig(
        name="kaggle_loan_default",
        display_name="Loan Default Prediction",
        folder="kaggle_loan_default",
        file_name="Loan_Default.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="Status",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=("ID", "year"),
        n_samples_expected=148670,
        n_features_expected=31,
        positive_class_ratio_expected=0.246,
    ),
)


__all__ = [
    "DatasetRegistry",
    "TimeSeriesDatasetRegistry",
    "make_default_registry",
    "make_default_timeseries_registry",
]
