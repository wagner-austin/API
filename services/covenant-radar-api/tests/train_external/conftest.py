"""Shared fixtures for external training tests."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile


def copy_real_taiwan(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Taiwan dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent.parent / "data" / "external" / "taiwan_data" / "data.csv"
    if not src.exists():
        raise FileNotFoundError("Taiwan dataset not found in repository data")
    dst_dir = external_root / "taiwan_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "data.csv"
    copyfile(str(src), str(dst))
    header = (dst.read_text(encoding="utf-8").splitlines())[0]
    cols = [c.strip() for c in header.split(",")]
    feature_names = cols[1:]
    n_rows = sum(1 for _ in dst.open(encoding="utf-8")) - 1
    return dst, n_rows, feature_names


def copy_real_us(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full US dataset into external_root and return (path, n_rows, feature_names)."""
    src = (
        Path(__file__).parent.parent.parent
        / "data"
        / "external"
        / "us_data"
        / "american_bankruptcy.csv"
    )
    if not src.exists():
        raise FileNotFoundError("US dataset not found in repository data")
    dst_dir = external_root / "us_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "american_bankruptcy.csv"
    copyfile(str(src), str(dst))
    header = (dst.read_text(encoding="utf-8-sig").splitlines())[0]
    cols = [c.strip() for c in header.split(",")]
    feature_names = [c for c in cols if c.startswith("X")]
    n_rows = sum(1 for _ in dst.open(encoding="utf-8-sig")) - 1
    return dst, n_rows, feature_names


def copy_real_polish(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Polish dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent.parent / "data" / "external" / "polish_data" / "1year.arff"
    if not src.exists():
        raise FileNotFoundError("Polish dataset not found in repository data")
    dst_dir = external_root / "polish_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "1year.arff"
    copyfile(str(src), str(dst))
    lines = dst.read_text(encoding="utf-8").splitlines()
    data_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_idx = i
            break
    if data_idx < 0:
        raise RuntimeError("ARFF file missing @data section")
    n_rows = len(lines) - (data_idx + 1)
    feature_names: list[str] = []
    for line in lines[: data_idx + 1]:
        s = line.strip()
        if s.lower().startswith("@attribute"):
            parts = s.split()
            if len(parts) >= 2 and parts[1].lower() != "class":
                feature_names.append(parts[1])
    return dst, n_rows, feature_names


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
