"""The repository's real dataset files, copied into a test's tree as a sample.

Tests that drive the real loaders, trainers and explainers read the files
committed under ``data/external``. Copying each file whole made them the
heaviest tests in the suite: on serendipity (fleet job 5242e59b, board task
dbbb9758) the four that read the 78,682-row US file took 23 to 27 s each, and
the permutation and SHAP explanations over all 6,819 Taiwan rows 9 to 34 s.
What each of them checks is that the real format reaches the real code and
comes back whole, which every tenth row of the file shows as well as all of
them do.

So every copy here keeps the file's header lines and then every
:data:`SAMPLE_EVERY`-th data row, byte for byte: the copy is the real format,
the real columns and the real class mix (Taiwan keeps 682 rows with 21
bankruptcies), only shorter. The US file keeps every
:data:`US_SAMPLE_EVERY`-th row instead, 1,574 rows with 97 failures: at every
tenth row its 7,869 rows still cost 1.6 s per load on the hub, because
covenant_ml's CSV loader checks every cell in Python, and the four tests that
load it took 2.0 to 5.7 s each. Each function returns the copy's path, the
number of data rows the copy holds and the feature columns, read back from the
copy rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

#: Where the repository keeps the real files.
EXTERNAL_DATA: Final[Path] = Path(__file__).parent.parent / "data" / "external"

#: One data row in this many is kept, starting with the first.
SAMPLE_EVERY: Final[int] = 10

#: The stride for the US file, the only one long enough to need a wider one.
US_SAMPLE_EVERY: Final[int] = 50

#: The columns of the Financial Distress panel that are not features.
_FINANCIAL_DISTRESS_NON_FEATURES: Final[tuple[str, ...]] = (
    "Company",
    "Time",
    "Financial Distress",
)


def _sample_real_file(relative: Path, external_root: Path, header_lines: int, every: int) -> Path:
    """Write the header and every ``every``-th data row of a real file.

    Args:
        relative: The file's path under :data:`EXTERNAL_DATA`, which is also
            its path under ``external_root``, where the loaders look for it.
        external_root: The test's own external data directory.
        header_lines: How many leading lines are header, kept whole.
        every: One data row in this many is kept, starting with the first.

    Returns:
        The path of the sampled copy.

    Raises:
        FileNotFoundError: The repository does not hold the file.
    """
    source = EXTERNAL_DATA / relative
    if not source.exists():
        raise FileNotFoundError(f"real dataset {relative.as_posix()} is not in {EXTERNAL_DATA}")
    lines = source.read_bytes().splitlines(keepends=True)
    kept = lines[:header_lines] + lines[header_lines::every]
    destination = external_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(kept))
    return destination


def _csv_header(path: Path, encoding: str) -> list[str]:
    """The stripped column names on a CSV file's first line.

    Args:
        path: The CSV file.
        encoding: Its text encoding (``utf-8-sig`` drops a byte order mark).

    Returns:
        The column names in file order.
    """
    header = path.read_text(encoding=encoding).splitlines()[0]
    return [column.strip() for column in header.split(",")]


def _csv_rows(path: Path, encoding: str) -> int:
    """The number of data rows in a CSV file with a one-line header.

    Args:
        path: The CSV file.
        encoding: Its text encoding.

    Returns:
        Its line count less the header.
    """
    with path.open(encoding=encoding) as handle:
        return sum(1 for _ in handle) - 1


def copy_real_taiwan(external_root: Path) -> tuple[Path, int, list[str]]:
    """Sample the Taiwan bankruptcy CSV into ``external_root/taiwan_data``.

    Args:
        external_root: The test's own external data directory.

    Returns:
        The copy's path, its data row count, and every column after the label.
    """
    path = _sample_real_file(Path("taiwan_data") / "data.csv", external_root, 1, SAMPLE_EVERY)
    return path, _csv_rows(path, "utf-8"), _csv_header(path, "utf-8")[1:]


def copy_real_us(external_root: Path) -> tuple[Path, int, list[str]]:
    """Sample the US bankruptcy CSV into ``external_root/us_data``.

    Args:
        external_root: The test's own external data directory.

    Returns:
        The copy's path, its data row count, and the ``X`` feature columns.
    """
    path = _sample_real_file(
        Path("us_data") / "american_bankruptcy.csv", external_root, 1, US_SAMPLE_EVERY
    )
    columns = _csv_header(path, "utf-8-sig")
    return path, _csv_rows(path, "utf-8-sig"), [c for c in columns if c.startswith("X")]


def copy_real_financial_distress(external_root: Path) -> tuple[Path, int, list[str]]:
    """Sample the Financial Distress panel into ``external_root/kaggle_financial_distress``.

    Args:
        external_root: The test's own external data directory.

    Returns:
        The copy's path, its data row count, and every column except the
        company, the period and the target.
    """
    path = _sample_real_file(
        Path("kaggle_financial_distress") / "Financial Distress.csv",
        external_root,
        1,
        SAMPLE_EVERY,
    )
    columns = _csv_header(path, "utf-8")
    features = [c for c in columns if c not in _FINANCIAL_DISTRESS_NON_FEATURES]
    return path, _csv_rows(path, "utf-8"), features


def copy_real_polish(external_root: Path) -> tuple[Path, int, list[str]]:
    """Sample the Polish bankruptcy ARFF file into ``external_root/polish_data``.

    The header is everything up to and including the ``@data`` line, so the
    attribute declarations survive whole.

    Args:
        external_root: The test's own external data directory.

    Returns:
        The copy's path, its data row count, and every attribute but ``class``.

    Raises:
        RuntimeError: The file has no ``@data`` line.
    """
    relative = Path("polish_data") / "1year.arff"
    source_lines = (EXTERNAL_DATA / relative).read_text(encoding="utf-8").splitlines()
    markers = [i for i, line in enumerate(source_lines) if line.strip().lower() == "@data"]
    if not markers:
        raise RuntimeError("ARFF file missing @data section")
    header_lines = markers[0] + 1
    path = _sample_real_file(relative, external_root, header_lines, SAMPLE_EVERY)
    lines = path.read_text(encoding="utf-8").splitlines()
    features: list[str] = []
    for line in lines[:header_lines]:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].lower() == "@attribute" and parts[1].lower() != "class":
            features.append(parts[1])
    return path, len(lines) - header_lines, features


__all__ = [
    "EXTERNAL_DATA",
    "SAMPLE_EVERY",
    "US_SAMPLE_EVERY",
    "copy_real_financial_distress",
    "copy_real_polish",
    "copy_real_taiwan",
    "copy_real_us",
]
