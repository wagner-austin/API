"""Controller reports are published only after measured integrity and checks pass."""

import runpy
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import controller_inspect
from hpc3.core.digest import sha256_hex
from tests.conftest import FakeRun
from tests.controller_fixtures import SELFCHECK, prepare
from tests.wrds_board import TASK, accepting_board


def arguments(root: Path, fake_run: FakeRun) -> list[str]:
    """Prepare the same provenance the public command reads."""
    provenance = root / "provenance.json"
    provenance.write_text(dump_json_str(prepare(root, fake_run)))
    return ["--provenance", str(provenance), "--wheels-dir", str(root),
            "--selfcheck", str(root / "selfcheck.py"),
            "--selfcheck-sha256", sha256_hex(SELFCHECK), "--report-task", TASK]


def test_publication_matches_measured_files(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    board = accepting_board(2)
    assert controller_inspect.main(arguments(tmp_path, fake_run)) == 0
    report = narrow_json_to_dict(load_json_str(emitted[0]))
    assert report["kind"] == "HPC3_CONTROLLER_EVIDENCE"
    assert report["installed_files"] == 1
    assert report["provenance_sha256"] == sha256_hex((tmp_path / "provenance.json").read_bytes())
    assert sent_arguments(board.bodies[-1])["body"] == "HPC3 CONTROLLER LOAD-REPORT\n" + emitted[0]


def test_refusal_never_publishes(tmp_path: Path, fake_run: FakeRun) -> None:
    board = accepting_board(2)
    args = arguments(tmp_path, fake_run)
    (tmp_path / "site/example.py").write_text("changed")
    with pytest.raises(ValueError, match="CONTROLLER_DIGEST_MISMATCH"):
        controller_inspect.main(args)
    assert board.bodies == []


def test_module_entrypoint(tmp_path: Path, fake_run: FakeRun, argv: list[str]) -> None:
    argv[:] = ["inspect", *arguments(tmp_path, fake_run)]
    accepting_board(2)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hpc3.cli.controller_inspect", run_name="__main__")
    assert caught.value.code == 0
