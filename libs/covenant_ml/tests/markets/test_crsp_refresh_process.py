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
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from covenant_ml.markets.crsp_cohort_records import encode_crsp_cohort_plan
from covenant_ml.markets.crsp_extraction_records import encode_crsp_limits
from covenant_ml.markets.crsp_snapshot import PSQL

from ._crsp_extraction_fixtures import FakeDatabase, database_rows, limits
from ._crsp_process_fixtures import client_environment
from ._crsp_refresh_fixtures import RefreshCase, refresh_case

PROGRAM = """
import sys
from pathlib import Path
from platform_core.json_utils import dump_json_str, require_str
from covenant_ml.markets.crsp_cohort_records import decode_crsp_cohort_plan
from covenant_ml.markets.crsp_extraction_records import decode_crsp_limits
from covenant_ml.markets.crsp_refresh import compare_crsp_vintages
from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
from covenant_ml.markets.validation import read_document
root = Path(sys.argv[1])
request = read_document(root / 'refresh-request.json')
result = compare_crsp_vintages(
    root / 'cohort' / 'bundle.json', require_str(request, 'digest'),
    decode_crsp_cohort_plan(request['plan']),
    EvaluatorKeys(decode_research_access(read_document(root / 'access.json'))),
    scratch=root / 'scratch', destination=root / 'fresh',
    limits=decode_crsp_limits(request['limits']), setup=None,
)
print(dump_json_str(result))
"""


def _run(
    case: RefreshCase, body: str, seconds: int, max_bytes: int
) -> subprocess.CompletedProcess[str]:
    root = case.cohort.extraction.root
    (root / "refresh-request.json").write_text(
        dump_json_str(
            {
                "digest": case.digest,
                "plan": encode_crsp_cohort_plan(case.cohort.plan),
                "limits": encode_crsp_limits(limits(seconds=seconds, max_relation_bytes=max_bytes)),
            }
        )
    )
    return subprocess.run(
        [sys.executable, "-c", PROGRAM, str(root)],
        env=client_environment(root, body),
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
    assert (result.returncode, result.stderr) == (0, "")
    public = narrow_json_to_dict(load_json_str(result.stdout))
    assert public["kind"] == "CRSP_DIAGNOSTIC_REFRESH"
    assert public["plaintext_removed"] is True
    assert list(case.cohort.extraction.scratch.iterdir()) == []
    assert (case.destination / "bundle.json").is_file()


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
    assert not case.destination.exists()
