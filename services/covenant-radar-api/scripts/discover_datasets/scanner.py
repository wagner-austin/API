"""Dataset scanning logic.

Scans dataset folders to detect file format, target columns, and metadata.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from scripts.discover_datasets.detection import (
    calculate_positive_ratio,
    detect_positive_negative_values,
    find_exclude_columns,
    find_target_candidates,
    recommend_target,
)
from scripts.discover_datasets.encoding import detect_encoding
from scripts.discover_datasets.parsers import (
    read_arff_header_and_sample,
    read_csv_header_and_sample,
    read_data_header_and_sample,
    read_excel_header_and_sample,
    read_xls_header_and_sample,
)
from scripts.discover_datasets.types import (
    DiscoveredDataset,
    DiscoverySummary,
    TargetColumnCandidate,
)

# Type alias for file format
FileFormat = Literal["csv", "arff", "xlsx", "xls", "data", "unknown"]

# Type alias for file encoding
FileEncoding = Literal["utf-8", "utf-8-sig", "latin-1", "cp1252"]


class TargetInfo(TypedDict, total=True):
    """Information about detected target column values.

    Attributes:
        positive_value: Detected positive class value.
        negative_value: Detected negative class value.
        label_type: Type of target labels (binary_int or binary_str).
        positive_ratio: Ratio of positive class in sample.
    """

    positive_value: str
    negative_value: str
    label_type: Literal["binary_int", "binary_str"]
    positive_ratio: float


class FileData(TypedDict, total=True):
    """Data read from a dataset file.

    Attributes:
        columns: Column names from the file.
        n_rows: Number of data rows.
        sample_rows: Sample of data rows.
    """

    columns: tuple[str, ...]
    n_rows: int
    sample_rows: tuple[tuple[str, ...], ...]


def _detect_file_format(path: Path) -> FileFormat:
    """Detect file format from extension.

    Args:
        path: Path to the file.

    Returns:
        Detected file format literal.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".arff":
        return "arff"
    if suffix == ".xlsx":
        return "xlsx"
    if suffix == ".xls":
        return "xls"
    if suffix == ".data":
        return "data"
    return "unknown"


def _find_data_file(folder: Path) -> tuple[Path | None, str]:
    """Find the primary data file in a folder.

    Args:
        folder: Dataset folder to scan.

    Returns:
        Tuple of (file path or None, status message).
    """
    csv_files = list(folder.glob("*.csv"))
    arff_files = list(folder.glob("*.arff"))
    excel_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    data_files = list(folder.glob("*.data"))

    all_files = csv_files + arff_files + excel_files + data_files

    if len(all_files) == 0:
        return None, "No data files found"

    if len(all_files) == 1:
        return all_files[0], "Single data file found"

    # Prefer files with common names
    preferred_names = ("data.csv", "train.csv", "dataset.csv")
    for preferred in preferred_names:
        for f in all_files:
            if f.name.lower() == preferred:
                return f, f"Selected {f.name} from multiple files"

    # Return largest file as primary
    largest = all_files[0]
    largest_size = largest.stat().st_size
    for f in all_files[1:]:
        size = f.stat().st_size
        if size > largest_size:
            largest = f
            largest_size = size
    return largest, f"Selected largest file: {largest.name}"


def _read_file_by_format(
    data_file: Path,
    file_format: FileFormat,
    encoding: FileEncoding,
) -> FileData:
    """Read header and sample rows from a file based on its format.

    Args:
        data_file: Path to the data file.
        file_format: Detected format of the file.
        encoding: File encoding to use.

    Returns:
        FileData with columns, row count, and sample rows.
    """
    if file_format == "csv":
        columns, n_rows, sample_rows = read_csv_header_and_sample(data_file, encoding)
    elif file_format == "data":
        columns, n_rows, sample_rows = read_data_header_and_sample(data_file, encoding)
    elif file_format == "arff":
        columns, n_rows, sample_rows = read_arff_header_and_sample(data_file)
    elif file_format == "xlsx":
        columns, n_rows, sample_rows = read_excel_header_and_sample(data_file)
    else:  # xls (legacy Excel format)
        columns, n_rows, sample_rows = read_xls_header_and_sample(data_file)

    return FileData(columns=columns, n_rows=n_rows, sample_rows=sample_rows)


def _detect_target_info(
    target_candidates: tuple[TargetColumnCandidate, ...],
    recommended_target: str,
    sample_rows: tuple[tuple[str, ...], ...],
    columns: tuple[str, ...],
) -> TargetInfo:
    """Detect positive/negative values and ratio for the target column.

    Args:
        target_candidates: All target column candidates found.
        recommended_target: The recommended target column name.
        sample_rows: Sample data rows.
        columns: All column names.

    Returns:
        TargetInfo with detected values and ratio.
    """
    if not recommended_target:
        return TargetInfo(
            positive_value="",
            negative_value="",
            label_type="binary_int",
            positive_ratio=0.0,
        )

    # Find the target candidate to get its unique values
    for candidate in target_candidates:
        if candidate["column_name"] == recommended_target and candidate["is_binary"]:
            pos, neg, ltype = detect_positive_negative_values(candidate["unique_values"])
            positive_ratio = 0.0
            if pos:
                positive_ratio = calculate_positive_ratio(
                    sample_rows, columns, recommended_target, pos
                )
            return TargetInfo(
                positive_value=pos,
                negative_value=neg,
                label_type=ltype,
                positive_ratio=positive_ratio,
            )

    return TargetInfo(
        positive_value="",
        negative_value="",
        label_type="binary_int",
        positive_ratio=0.0,
    )


def _determine_scan_status(
    target_candidates: tuple[TargetColumnCandidate, ...],
    file_message: str,
) -> tuple[Literal["success", "warning", "error"], str]:
    """Determine the scan status based on target detection.

    Args:
        target_candidates: Detected target column candidates.
        file_message: Message from file detection.

    Returns:
        Tuple of (status, message).
    """
    if len(target_candidates) == 0:
        return "warning", "No target column candidates found"
    if not any(c["is_binary"] for c in target_candidates):
        return "warning", "No binary target column found"
    return "success", file_message


def _create_empty_result(folder_name: str, message: str) -> DiscoveredDataset:
    """Create an empty/error result for a folder with no data files.

    Args:
        folder_name: Name of the folder.
        message: Error message describing why no data was found.

    Returns:
        DiscoveredDataset with error status.
    """
    return DiscoveredDataset(
        folder_name=folder_name,
        file_name="",
        file_format="unknown",
        encoding="utf-8",
        n_rows=0,
        n_columns=0,
        target_candidates=(),
        recommended_target="",
        recommended_exclude=(),
        target_positive_value="",
        target_negative_value="",
        target_label_type="binary_int",
        positive_class_ratio=0.0,
        status="error",
        message=message,
    )


def scan_dataset_folder(folder: Path) -> DiscoveredDataset:
    """Scan a single dataset folder.

    Args:
        folder: Path to dataset folder.

    Returns:
        DiscoveredDataset with scan results.
    """
    # Find data file
    data_file, file_message = _find_data_file(folder)

    if data_file is None:
        return _create_empty_result(folder.name, file_message)

    # Detect file format and encoding
    file_format = _detect_file_format(data_file)
    needs_encoding = file_format in ("csv", "data")
    encoding: FileEncoding = detect_encoding(data_file) if needs_encoding else "utf-8"

    # Read file data
    file_data = _read_file_by_format(data_file, file_format, encoding)

    # Analyze columns
    target_candidates = find_target_candidates(file_data["columns"], file_data["sample_rows"])
    recommended_target = recommend_target(target_candidates)
    recommended_exclude = find_exclude_columns(file_data["columns"])

    # Detect target info
    target_info = _detect_target_info(
        target_candidates,
        recommended_target,
        file_data["sample_rows"],
        file_data["columns"],
    )

    # Determine status
    status, message = _determine_scan_status(target_candidates, file_message)

    return DiscoveredDataset(
        folder_name=folder.name,
        file_name=data_file.name,
        file_format=file_format,
        encoding=encoding,
        n_rows=file_data["n_rows"],
        n_columns=len(file_data["columns"]),
        target_candidates=target_candidates,
        recommended_target=recommended_target,
        recommended_exclude=recommended_exclude,
        target_positive_value=target_info["positive_value"],
        target_negative_value=target_info["negative_value"],
        target_label_type=target_info["label_type"],
        positive_class_ratio=target_info["positive_ratio"],
        status=status,
        message=message,
    )


def scan_external_dir(external_dir: Path) -> DiscoverySummary:
    """Scan all dataset folders in the external directory.

    Args:
        external_dir: Path to data/external/ directory.

    Returns:
        DiscoverySummary with all scan results.
    """
    if not external_dir.exists():
        return DiscoverySummary(
            n_total=0,
            n_success=0,
            n_warning=0,
            n_error=0,
            datasets=(),
        )

    folders = sorted(
        [p for p in external_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )

    datasets: list[DiscoveredDataset] = []
    n_success = 0
    n_warning = 0
    n_error = 0

    for folder in folders:
        result = scan_dataset_folder(folder)
        datasets.append(result)

        if result["status"] == "success":
            n_success += 1
        elif result["status"] == "warning":
            n_warning += 1
        else:
            n_error += 1

    return DiscoverySummary(
        n_total=len(folders),
        n_success=n_success,
        n_warning=n_warning,
        n_error=n_error,
        datasets=tuple(datasets),
    )


__all__ = [
    "scan_dataset_folder",
    "scan_external_dir",
]
