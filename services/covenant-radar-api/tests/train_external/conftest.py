"""Shared fixtures for external training tests.

The copies of the repository's real dataset files live in
``tests/_real_datasets.py``, shared with the optimize and explain tests.
"""

from __future__ import annotations

from pathlib import Path


def write_taiwan_dataset(base_dir: Path) -> Path:
    """Write a minimal Taiwan-style CSV dataset for testing."""
    taiwan_dir = base_dir / "taiwan_data"
    taiwan_dir.mkdir(parents=True, exist_ok=True)
    path = taiwan_dir / "data.csv"
    rows = [" Bankrupt?, Feat1, Feat2, Feat3"]
    for i in range(15):
        label = 1 if i < 5 else 0
        rows.append(f"{label},{i * 0.1:.1f},{i * 0.2:.1f},{i * 0.3:.1f}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def write_us_dataset(base_dir: Path) -> Path:
    """Write a minimal US-style CSV dataset for testing."""
    us_dir = base_dir / "us_data"
    us_dir.mkdir(parents=True, exist_ok=True)
    path = us_dir / "american_bankruptcy.csv"
    headers = ["company_name", "status_label", "year"] + [f"X{i}" for i in range(1, 19)]
    rows = [",".join(headers)]
    for i in range(15):
        status = "failed" if i < 5 else "alive"
        values = [f"company_{i}", status, "2020"] + [f"{i * 0.1:.1f}" for _ in range(18)]
        rows.append(",".join(values))
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def write_polish_dataset(base_dir: Path) -> Path:
    """Write a minimal Polish-style ARFF dataset for testing."""
    polish_dir = base_dir / "polish_data"
    polish_dir.mkdir(parents=True, exist_ok=True)
    path = polish_dir / "1year.arff"
    attrs = "\n".join([f"@attribute Attr{i} numeric" for i in range(1, 65)])
    rows = ["@relation test", attrs, "@attribute class {0,1}", "", "@data"]
    for i in range(15):
        label = 1 if i < 5 else 0
        features = ",".join([f"{i * 0.01:.2f}" for _ in range(64)])
        rows.append(f"{features},{label}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path
