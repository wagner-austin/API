"""Real refresh subprocesses exercise client exit, kernel limits, deadlines and cleanup.

The executable client copies controlled CSV output; it does not test PostgreSQL
query semantics. Native synthetic database rehearsal remains a separate check.
"""

from __future__ import annotations

import csv
import io
import signal
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)
from platform_core.run_record import decode_observation
from scripts import crsp_extraction

from covenant_ml.markets.crsp_snapshot import PSQL

from ._crsp_diagnostic_process import (
    diagnostic_arguments,
    prepare_allocation,
    run_diagnostic_process,
)
from ._crsp_extraction_fixtures import FakeDatabase, database_rows, installed
from ._crsp_refresh_fixtures import RefreshCase, refresh_case

@pytest.fixture()
def allocation(tmp_path: Path) -> Generator[RefreshCase, None, None]:
    case = refresh_case(tmp_path)
    environment = dict(config_test_hooks.get_environment())
    environment.update(prepare_allocation(case))
    previous = config_test_hooks.get_environment
    config_test_hooks.get_environment = lambda: environment
    yield case
    config_test_hooks.get_environment = previous


def test_refresh_cli_writes_native_observations_in_process(allocation: RefreshCase) -> None:
    with installed(FakeDatabase(database_rows(), securities=(1,))):
        arguments = diagnostic_arguments(allocation, 120, 2_000_000, "refresh")
        assert crsp_extraction.main(arguments) == 0
    root = allocation.cohort.extraction.root / "native-output"
    result = narrow_json_to_dict(load_json_str((root / "comparison.json").read_text()))
    counts = narrow_json_to_dict(result["rows"])
    metrics = tuple(
        decode_observation(value)
        for value in narrow_json_to_list(load_json_str((root / "metrics.json").read_text()))
    )
    assert metrics == tuple(
        decode_observation({"name": role + "." + name, "value": value})
        for role, fields in counts.items()
        for name, value in narrow_json_to_dict(fields).items()
    )
    assert list(allocation.cohort.extraction.scratch.iterdir()) == []


def _client_output(root: Path) -> Path:
    """Write the controlled CSVs and transaction readings consumed by the real child."""
    source = root / "copy-source"
    (source / "train-validation").mkdir(parents=True)
    (source / "extract.sql").write_text("")
    FakeDatabase(database_rows(), securities=(1,))(PSQL, source, seconds=1, max_file_bytes=1)
    (source / "extract.sql").unlink()
    return source


def test_real_refresh_process_captures_and_encrypts_client_output(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    source = _client_output(tmp_path)
    result = run_diagnostic_process(case, f'exec cp -R "{source}/." .\n', 120, 2_000_000, "refresh")
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    output = tmp_path / "native-output"
    public = narrow_json_to_dict(load_json_str((output / "comparison.json").read_text()))
    assert public["kind"] == "CRSP_DIAGNOSTIC_REFRESH"
    assert public["plaintext_removed"] is True
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    assert (output / "comparison" / "bundle.json").is_file()
    metrics = tuple(
        decode_observation(value)
        for value in narrow_json_to_list(load_json_str((output / "metrics.json").read_text()))
    )
    assert len(metrics) == 24
    assert {item["name"] for item in metrics} == {
        role + "." + name
        for role in ("membership", "returns", "calendar", "delists")
        for name in ("old", "fresh", "unchanged", "added", "removed", "changed")
    }
    assert all(item["value"] == 0 for item in metrics if item["name"].endswith(".changed"))


@pytest.mark.parametrize("field", ["permno", "dlyret", "dlycaldt"])
def test_real_refresh_refuses_private_scalar_without_public_leak(
    tmp_path: Path, field: str
) -> None:
    case = refresh_case(tmp_path)
    source = _client_output(tmp_path)
    path = source / "train-validation" / "returns.csv"
    rows = list(csv.reader(io.StringIO(path.read_text("utf-8"))))
    sentinel = "PRIVATE_SOURCE_SCALAR_123456"
    rows[1][rows[0].index(field)] = sentinel
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    result = run_diagnostic_process(case, f'exec cp -R "{source}/." .\n', 120, 2_000_000, "refresh")
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines()[-1] == (
        f"ValueError: [CRSP-007] invalid {field} scalar in returns CSV"
    )
    assert sentinel not in result.stderr
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    assert list((tmp_path / "native-output").iterdir()) == []


@pytest.mark.parametrize(
    ("body", "seconds", "max_bytes", "error"),
    [
        (
            "echo 'private SQL' >&2\nexit 3\n",
            120,
            2_000_000,
            "ValueError: [CRSP-011] psql exited 3; no bundle was published",
        ),
        (
            "exec dd if=/dev/zero of=overflow.bin bs=1000000 count=1 status=none\n",
            120,
            5000,
            f"ValueError: [CRSP-011] psql exited {-signal.SIGXFSZ}; no bundle was published",
        ),
        (
            "exec sleep 30\n",
            1,
            2_000_000,
            f"subprocess.TimeoutExpired: Command '{PSQL}'",
        ),
    ],
)
def test_real_client_failures_remove_authenticated_plaintext_without_leaking_sql(
    tmp_path: Path, body: str, seconds: int, max_bytes: int, error: str
) -> None:
    case = refresh_case(tmp_path)
    result = run_diagnostic_process(case, body, seconds, max_bytes, "refresh")
    assert result.returncode == 1
    assert result.stdout == ""
    failure = result.stderr.splitlines()[-1]
    if seconds == 1:
        command, separator, deadline = failure.partition(" timed out after ")
        assert command == error
        assert separator == " timed out after "
        duration, unit = deadline.split()
        assert unit == "seconds"
        assert 0 < float(duration) <= seconds
    else:
        assert failure == error
    assert "private SQL" not in result.stderr
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    assert list((tmp_path / "native-output").iterdir()) == []
