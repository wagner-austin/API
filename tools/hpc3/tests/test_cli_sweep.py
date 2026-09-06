"""Tests for the sweep CLI: many members, one sbatch call.

The triage CLI's tests were split out to ``test_cli_triage.py`` on 2026-09-06,
when this file passed the 600-line ceiling. The two shared a fixture and
nothing else.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.cli import sweep as sweep_cli
from tests.against_hpc3 import read_ledger
from tests.conftest import (
    PREFLIGHT_LINE,
    FakeRun,
    budget_document,
    gpus,
    project_config,
    workspace_document,
    write_file,
    write_workspace,
)


def _payload(count: int = 3, **overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid sweep document.

    Args:
        count: How many members.
        **overrides: Fields to add or replace.

    Returns:
        The document.
    """
    members: list[JSONValue] = [
        {"suffix": f"s{i}", "command": f"python train.py --seed {i}", "artifact": None}
        for i in range(count)
    ]
    document: dict[str, JSONValue] = {
        "project": "abl",
        "name": "rung",
        "members": members,
        "experiment": {"rung": "774M"},
    }
    document.update(overrides)
    return document


def _write(path: pathlib.Path, payload: JSONValue) -> None:
    """Write a document for the CLI to read.

    Args:
        path: File to write.
        payload: Document to serialise.
    """
    write_file(path, dump_json_str(payload).encode("utf-8"))


def _args(tmp_path: pathlib.Path, **overrides: JSONValue) -> list[str]:
    """Build a full sweep argument list.

    Args:
        tmp_path: Directory holding the documents.
        **overrides: Workspace fields to replace.

    Returns:
        Arguments excluding the program name.
    """
    config = write_workspace(tmp_path / "hpc3.json", workspace_document(**overrides))
    return ["--config", config, "--run", str(tmp_path / "s.json")]


def _healthy(fake: FakeRun) -> None:
    """Script a cluster that admits every member.

    Args:
        fake: The runner to script.
    """
    fake.add("test -d", stdout="PRESENT\n")
    fake.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")


class TestSweepCli:
    def test_it_reports_every_member_and_a_watch_command(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "s.json", _payload())
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 101\n")

        assert sweep_cli.main(_args(tmp_path)) == 0
        assert emitted[0].startswith("budget OK: projected 1.5 GPU-hours")
        assert emitted[1:4] == [
            "submitted 101_0 abl.rung-s0",
            "submitted 101_1 abl.rung-s1",
            "submitted 101_2 abl.rung-s2",
        ]
        assert emitted[-1].endswith("--job 101_0,101_1,101_2")

    def test_it_reports_where_every_members_log_landed(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "s.json", _payload())
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        sweep_cli.main(_args(tmp_path))
        assert "logs /pub/w/abl/logs" in emitted

    def test_every_member_reaches_the_ledger(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "s.json", _payload())
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 101\n")

        sweep_cli.main(_args(tmp_path))
        entries = read_ledger(tmp_path / "ledger.jsonl")
        assert [e["job_id"] for e in entries] == ["101_0", "101_1", "101_2"]

    def test_a_budget_overrun_stops_it_before_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write(tmp_path / "s.json", _payload())
        with pytest.raises(AppError) as excinfo:
            sweep_cli.main(
                _args(
                    tmp_path,
                    projects={"abl": project_config(budget=budget_document(gpu_hours=1.0))},
                )
            )
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED
        assert fake_run.calls == []
        assert emitted == []

    def test_an_oversized_sweep_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write(tmp_path / "s.json", _payload(count=25))
        with pytest.raises(AppError) as excinfo:
            sweep_cli.main(_args(tmp_path))
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING
        assert fake_run.calls == []

    def test_the_report_names_the_partition_and_that_it_is_free(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "s.json", _payload(count=2))
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 1\n")
        sweep_cli.main(
            _args(
                tmp_path,
                projects={"abl": project_config(partition="free-gpu32", gpu=gpus("L40S"))},
            )
        )
        assert any("free-gpu32 (free)" in line for line in emitted)

    def test_the_config_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            sweep_cli.main(["--run", str(tmp_path / "s.json")])

    def test_the_entrypoint_reads_the_process_arguments(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        argv: list[str],
        frozen_clock: str,
    ) -> None:
        _write(tmp_path / "s.json", _payload(count=1))
        _healthy(fake_run)
        fake_run.add("sbatch", stdout="Submitted batch job 9\n")
        argv[:] = ["prog", *_args(tmp_path)]

        with pytest.raises(SystemExit) as excinfo:
            sweep_cli.entrypoint()
        assert excinfo.value.code == 0
        assert emitted[1] == "submitted 9_0 abl.rung-s0"
