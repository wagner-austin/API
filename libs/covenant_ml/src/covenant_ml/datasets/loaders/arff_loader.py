"""ARFF dataset loader.

Loads ARFF (Weka) datasets into LoadedDataset format.
ARFF format consists of @relation, @attribute, @data sections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from covenant_ml.datasets.loaders._parsing import (
    encode_label,
    find_column_index,
    parse_numeric_value,
)
from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
)


class ARFFLoader:
    """Loads ARFF (Weka) datasets into LoadedDataset format.

    ARFF format:
    - @relation <name>
    - @attribute <name> <type>
    - @data
    - comma-separated values

    Handles missing values marked with '?' as per ARFF spec.
    """

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load ARFF dataset.

        Args:
            config: Dataset configuration.
            external_dir: Root directory for datasets.

        Returns:
            LoadedDataset ready for ML.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If format invalid or parsing fails.
        """
        file_path = external_dir / config["folder"] / config["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Parse ARFF
        if config.get("group_column") is not None:
            # Silently dropping groups would let a grouped dataset row-split
            # and leak; refuse until this loader learns to carry them.
            raise ValueError("group_column is only supported by the CSV loader")

        attributes, data_rows = self._parse_arff(file_path, config["encoding"])

        # Find target column
        target_spec = config["target"]
        target_idx = find_column_index(attributes, target_spec["column_name"])

        # Build feature indices (exclude target and any exclude_columns)
        exclude_set = set(config["exclude_columns"])
        exclude_set.add(target_spec["column_name"].lower())

        feature_indices: list[int] = []
        feature_names: list[str] = []
        for i, attr_name in enumerate(attributes):
            if attr_name.lower() not in exclude_set:
                feature_indices.append(i)
                feature_names.append(attr_name)

        # Convert to arrays
        n_samples = len(data_rows)
        n_features = len(feature_indices)

        x_array = np.zeros((n_samples, n_features), dtype=np.float64)
        y_array = np.zeros(n_samples, dtype=np.int64)

        for row_idx, row in enumerate(data_rows):
            # Extract features
            for feat_idx, col_idx in enumerate(feature_indices):
                value = row[col_idx] if col_idx < len(row) else ""
                x_array[row_idx, feat_idx] = parse_numeric_value(value)

            # Extract and encode label
            target_value = row[target_idx] if target_idx < len(row) else ""
            y_array[row_idx] = encode_label(target_value, target_spec, row_idx, file_path)

        # Replace any remaining NaN/inf with 0.0
        x_array = np.nan_to_num(x_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute metadata
        n_positive = int(np.sum(y_array))
        n_negative = n_samples - n_positive
        positive_ratio = n_positive / n_samples if n_samples > 0 else 0.0

        meta = DatasetMeta(
            name=config["name"],
            n_samples=n_samples,
            n_features=n_features,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_ratio=positive_ratio,
            feature_names=tuple(feature_names),
            categorical_encodings=(),  # ARFF numeric attributes only
        )

        return LoadedDataset(meta=meta, x=x_array, y=y_array, groups=None)

    def _parse_arff(
        self,
        file_path: Path,
        encoding: str,
    ) -> tuple[list[str], list[list[str]]]:
        """Parse ARFF file and return attribute names and data rows.

        Args:
            file_path: Path to ARFF file.
            encoding: File encoding to use.

        Returns:
            Tuple of (attribute_names, data_rows).

        Raises:
            ValueError: If no data rows found or format invalid.
        """
        attributes: list[str] = []
        data_rows: list[list[str]] = []
        in_data = False

        with open(file_path, encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue

                if line.lower() == "@data":
                    in_data = True
                    continue

                if not in_data:
                    # Parse attribute definition
                    if line.lower().startswith("@attribute"):
                        parts = line.split()
                        if len(parts) >= 2:
                            attr_name = parts[1]
                            attributes.append(attr_name)
                else:
                    # Parse data row
                    values = line.split(",")
                    data_rows.append([v.strip() for v in values])

        if not data_rows:
            raise ValueError(f"No data rows found in {file_path}")

        return attributes, data_rows


def create_arff_loader() -> ARFFLoader:
    """Factory function for creating ARFF loader.

    Returns:
        New ARFFLoader instance.
    """
    return ARFFLoader()


__all__ = [
    "ARFFLoader",
    "create_arff_loader",
]
