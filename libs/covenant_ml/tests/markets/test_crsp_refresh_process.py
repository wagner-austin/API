"""Real refresh subprocesses exercise client exit, kernel limits, deadlines and cleanup.

The executable client copies controlled CSV output; it does not test PostgreSQL
query semantics. Native synthetic database rehearsal remains a separate check.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)
from platform_core.run_record import decode_observation

from covenant_ml.markets.crsp_cohort_records import encode_crsp_cohort_plan
from covenant_ml.markets.crsp_snapshot import PSQL

from ._crsp_extraction_fixtures import FakeDatabase, database_rows
from ._crsp_process_fixtures import client_environment
from ._crsp_refresh_fixtures import RefreshCase, refresh_case


def _run(
    case: RefreshCase, body: str, seconds: int, max_bytes: int
) -> subprocess.CompletedProcess[str]:
    root = case.cohort.extraction.root
    (root / "plan.json").write_text(dump_json_str(encode_crsp_cohort_plan(case.cohort.plan)))
    (root / "native-output").mkdir()
    environment = client_environment(root, body)
    environment.update(
        WRDS_OUTPUT_DIR=str(root / "native-output"), TMPDIR=str(case.cohort.extraction.scratch)
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.crsp_extraction",
            "refresh",
            "--cohort",
            str(case.cohort.destination / "bundle.json"),
            "--cohort-sha256",
            case.digest,
            "--plan",
            str(root / "plan.json"),
            "--access",
            str(root / "access.json"),
            "--max-rows",
            "20000",
            "--max-relation-bytes",
            str(max_bytes),
            "--max-partition-bytes",
            "4000000",
            "--seconds",
            str(seconds),
            "--scratch-env",
            "TMPDIR",
            "--output-env",
            "WRDS_OUTPUT_DIR",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_real_refresh_process_captures_and_encrypts_client_output(tmp_path: Path) -> None:
    case = refresh_case(tmp_path)
    source = tmp_path / "copy-source"
    (source / "train-validation").mkdir(parents=True)
    (source / "extract.sql").write_text("")
    FakeDatabase(database_rows(), securities=(1,))(PSQL, source, seconds=1, max_file_bytes=1)
    (source / "extract.sql").unlink()
    result = _run(case, f'exec cp -R "{source}/." .\n', 120, 2_000_000)
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


@pytest.mark.parametrize(
    ("body", "seconds", "max_bytes", "error"),
    [
        ("echo 'private SQL' >&2\nexit 3\n", 120, 2_000_000, "psql exited 3"),
        (
            "exec dd if=/dev/zero of=overflow.bin bs=1000000 count=1 status=none\n",
            120,
            5000,
            f"psql exited {-signal.SIGXFSZ}",
        ),
        ("exec sleep 30\n", 1, 2_000_000, "TimeoutExpired"),
    ],
)
def test_real_client_failures_remove_authenticated_plaintext_without_leaking_sql(
    tmp_path: Path, body: str, seconds: int, max_bytes: int, error: str
) -> None:
    case = refresh_case(tmp_path)
    result = _run(case, body, seconds, max_bytes)
    assert result.returncode == 1
    assert result.stdout == ""
    assert error in result.stderr
    assert "private SQL" not in result.stderr
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    assert list((tmp_path / "native-output").iterdir()) == []
