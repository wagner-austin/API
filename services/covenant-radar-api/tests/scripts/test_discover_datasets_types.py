"""Tests for discover_datasets types module."""

from __future__ import annotations

from scripts.discover_datasets.types import (
    DetectionStatus,
    DiscoveredDataset,
    DiscoverySummary,
    TargetColumnCandidate,
)


class TestTargetColumnCandidate:
    """Tests for TargetColumnCandidate TypedDict."""

    def test_create_with_all_fields(self) -> None:
        """Test creating TargetColumnCandidate with all required fields."""
        candidate: TargetColumnCandidate = {
            "column_name": "target",
            "unique_values": ("0", "1"),
            "n_unique": 2,
            "is_binary": True,
        }
        assert candidate["column_name"] == "target"
        assert candidate["unique_values"] == ("0", "1")
        assert candidate["n_unique"] == 2
        assert candidate["is_binary"] is True

    def test_create_non_binary_candidate(self) -> None:
        """Test creating non-binary TargetColumnCandidate."""
        candidate: TargetColumnCandidate = {
            "column_name": "class",
            "unique_values": ("A", "B", "C"),
            "n_unique": 3,
            "is_binary": False,
        }
        assert candidate["column_name"] == "class"
        assert candidate["is_binary"] is False


class TestDiscoveredDataset:
    """Tests for DiscoveredDataset TypedDict."""

    def test_create_success_dataset(self) -> None:
        """Test creating a successful DiscoveredDataset."""
        dataset: DiscoveredDataset = {
            "folder_name": "test_dataset",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 1000,
            "n_columns": 10,
            "target_candidates": (
                {
                    "column_name": "target",
                    "unique_values": ("0", "1"),
                    "n_unique": 2,
                    "is_binary": True,
                },
            ),
            "recommended_target": "target",
            "recommended_exclude": ("id", "name"),
            "target_positive_value": "1",
            "target_negative_value": "0",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.3,
            "status": "success",
            "message": "Single data file found",
        }
        assert dataset["folder_name"] == "test_dataset"
        assert dataset["status"] == "success"
        assert len(dataset["target_candidates"]) == 1
        assert dataset["target_positive_value"] == "1"
        assert dataset["positive_class_ratio"] == 0.3

    def test_create_error_dataset(self) -> None:
        """Test creating an error DiscoveredDataset."""
        dataset: DiscoveredDataset = {
            "folder_name": "empty_folder",
            "file_name": "",
            "file_format": "unknown",
            "encoding": "utf-8",
            "n_rows": 0,
            "n_columns": 0,
            "target_candidates": (),
            "recommended_target": "",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "error",
            "message": "No data files found",
        }
        assert dataset["status"] == "error"
        assert dataset["file_name"] == ""


class TestDiscoverySummary:
    """Tests for DiscoverySummary TypedDict."""

    def test_create_empty_summary(self) -> None:
        """Test creating empty DiscoverySummary."""
        summary: DiscoverySummary = {
            "n_total": 0,
            "n_success": 0,
            "n_warning": 0,
            "n_error": 0,
            "datasets": (),
        }
        assert summary["n_total"] == 0
        assert len(summary["datasets"]) == 0

    def test_create_summary_with_datasets(self) -> None:
        """Test creating DiscoverySummary with datasets."""
        dataset: DiscoveredDataset = {
            "folder_name": "test",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "warning",
            "message": "No target column found",
        }
        summary: DiscoverySummary = {
            "n_total": 1,
            "n_success": 0,
            "n_warning": 1,
            "n_error": 0,
            "datasets": (dataset,),
        }
        assert summary["n_total"] == 1
        assert summary["n_warning"] == 1
        assert summary["datasets"][0]["folder_name"] == "test"


class TestDetectionStatus:
    """Tests for DetectionStatus type alias."""

    def test_valid_status_values(self) -> None:
        """Test that valid status values can be assigned."""
        status_success: DetectionStatus = "success"
        status_warning: DetectionStatus = "warning"
        status_error: DetectionStatus = "error"

        assert status_success == "success"
        assert status_warning == "warning"
        assert status_error == "error"
