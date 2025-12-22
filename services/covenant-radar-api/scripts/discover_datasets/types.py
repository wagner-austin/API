"""Types for dataset discovery.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# Detection status literals
DetectionStatus = Literal["success", "warning", "error"]


class TargetColumnCandidate(TypedDict, total=True):
    """A candidate target column found during scanning.

    Attributes:
        column_name: Name of the column.
        unique_values: Sample of unique values found.
        n_unique: Number of unique values.
        is_binary: Whether the column appears to be binary.
    """

    column_name: str
    unique_values: tuple[str, ...]
    n_unique: int
    is_binary: bool


class DiscoveredDataset(TypedDict, total=True):
    """Result of scanning a single dataset folder.

    Attributes:
        folder_name: Name of the folder in data/external/.
        file_name: Detected primary data file name.
        file_format: Detected file format (csv, arff, excel).
        encoding: Detected or assumed file encoding.
        n_rows: Number of rows in the dataset.
        n_columns: Number of columns in the dataset.
        target_candidates: Possible target columns detected.
        recommended_target: Best guess for target column, or empty string.
        recommended_exclude: Columns that should be excluded (IDs, names, etc.).
        target_positive_value: Detected positive class value, or empty string.
        target_negative_value: Detected negative class value, or empty string.
        target_label_type: Type of target labels (binary_int or binary_str).
        positive_class_ratio: Ratio of positive class in sample (0.0 if unknown).
        status: Detection status (success, warning, error).
        message: Status message or error description.
    """

    folder_name: str
    file_name: str
    file_format: Literal["csv", "arff", "xlsx", "xls", "data", "unknown"]
    encoding: Literal["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    n_rows: int
    n_columns: int
    target_candidates: tuple[TargetColumnCandidate, ...]
    recommended_target: str
    recommended_exclude: tuple[str, ...]
    target_positive_value: str
    target_negative_value: str
    target_label_type: Literal["binary_int", "binary_str"]
    positive_class_ratio: float
    status: DetectionStatus
    message: str


class DiscoverySummary(TypedDict, total=True):
    """Summary of all discovered datasets.

    Attributes:
        n_total: Total number of dataset folders.
        n_success: Number successfully scanned.
        n_warning: Number with warnings.
        n_error: Number with errors.
        datasets: Tuple of all discovered datasets.
    """

    n_total: int
    n_success: int
    n_warning: int
    n_error: int
    datasets: tuple[DiscoveredDataset, ...]


class ValidationResult(TypedDict, total=True):
    """Result of validating a discovered dataset config.

    Attributes:
        folder_name: Dataset folder name.
        valid: Whether the config is valid.
        target_exists: Whether target column exists in data.
        positive_value_found: Whether positive value exists in target.
        negative_value_found: Whether negative value exists in target.
        calculated_ratio: Actual positive class ratio from full data.
        expected_ratio: Expected ratio from discovery.
        ratio_diff: Absolute difference between calculated and expected.
        errors: List of validation error messages.
    """

    folder_name: str
    valid: bool
    target_exists: bool
    positive_value_found: bool
    negative_value_found: bool
    calculated_ratio: float
    expected_ratio: float
    ratio_diff: float
    errors: tuple[str, ...]


__all__ = [
    "DetectionStatus",
    "DiscoveredDataset",
    "DiscoverySummary",
    "TargetColumnCandidate",
    "ValidationResult",
]
