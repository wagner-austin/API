"""Tests for the chain CLI.

The reported lines are the assertions, because what an operator reads is the
only account they get of which stage waits on which. Once the jobs age out of
``squeue`` the wiring is not recoverable from the cluster at all.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.cli import chain as chain_cli
from tests.against_hpc3 import read_ledger
from tests.conftest import (
    PREFLIGHT_LINE,
    FakeRun,
    budget_document,
    project_config,
    workspace_document,
    write_file,
    write_workspace,
)


def _payload(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid chain document.

    Args:
        **overrides: Fields to add or replace.

    Returns:
        The document.
    """
    document: dict[str, JSONValue] = {
        "project": "abl",
        "name": "pipeline",
        "experiment": {"sample": "batch7"},
        "stages": [
            {"suffix": "extract", "command": "python extract.py", "artifact": None},
            {"suffix": "evaluate", "command": "python evaluate.py", "minutes": 45},
        ],
    }
    document.update(overrides)
    return document


def _args(tmp_path: pathlib.Path, **overrides: JSONValue) -> list[str]:
    """Build a full chain argument list.

    Args:
        tmp_path: Directory holding the documents.
        **overrides: Workspace fields to replace.

    Returns:
        Arguments excluding the program name.
    """
    config = write_workspace(tmp_path / "hpc3.json", workspace_document(**overrides))
    return ["--config", config, "--run", str(tmp_path / "c.json")]


def _write(path: pathlib.Path, payload: JSONValue) -> None:
    """Write a document for the CLI to read.

    Args:
        path: File to write.
        payload: Document to serialise.
    """
    write_file(path, dump_json_str(payload).encode("utf-8"))


def _healthy(fake: FakeRun) -> None:
    """Script a cluster that admits every stage.

    Args:
        fake: The runner to script.
    """
    fake.add("test -d", stdout="PRESENT\n")
    fake.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")
    fake.add("sbatch abl.pipeline-extract", stdout="Submitted batch job 101\n")
    fake.add("sbatch abl.pipeline-evaluate", stdout="Submitted batch job 102\n")


class TestChainCli:
    def test_it_reports_each_stage_and_what_it_waits_on(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)

        assert chain_cli.main(_args(tmp_path)) == 0
        assert "submitted 101 abl.pipeline-extract" in emitted[1]
        assert "starts when ready" in emitted[1]
        assert "submitted 102 abl.pipeline-evaluate" in emitted[2]
        assert "after 101" in emitted[2]

    def test_it_says_that_a_failure_cancels_the_rest(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """The difference between "blocked" and "gone" is what an operator
        reading a half-finished pipeline needs."""
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)

        chain_cli.main(_args(tmp_path))
        assert any("--kill-on-invalid-dep" in line for line in emitted)

    def test_it_reports_the_stage_count_and_the_partition(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)

        chain_cli.main(_args(tmp_path))
        assert "2 stage(s) on free-gpu (free)" in emitted

    def test_it_offers_a_watch_command_naming_every_stage(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)

        chain_cli.main(_args(tmp_path))
        assert any(line.endswith("--job 101,102") for line in emitted)

    def test_every_stage_reaches_the_ledger(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, frozen_clock: str
    ) -> None:
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)

        chain_cli.main(_args(tmp_path))
        recorded = read_ledger(tmp_path / "ledger.jsonl")
        assert [entry["job_id"] for entry in recorded] == ["101", "102"]

    def test_the_whole_pipeline_is_budgeted_not_just_its_first_stage(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], frozen_clock: str
    ) -> None:
        """Stages are sequential in time and simultaneous in commitment: 30
        minutes plus 45 is 1.25 GPU-hours, and a budget consulted one stage at
        a time would approve a pipeline it would refuse whole."""
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)

        chain_cli.main(_args(tmp_path))
        assert emitted[0].startswith("budget OK: projected 1.2 GPU-hours")

    def test_a_pipeline_over_the_budget_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, frozen_clock: str
    ) -> None:
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)

        with pytest.raises(AppError) as excinfo:
            chain_cli.main(
                _args(
                    tmp_path,
                    projects={"abl": project_config(budget=budget_document(gpu_hours=1.0))},
                )
            )
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED
        assert not any("&& sbatch " in command for command in fake_run.commands())

    def test_a_billing_stage_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, frozen_clock: str
    ) -> None:
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)

        with pytest.raises(AppError) as excinfo:
            chain_cli.main(
                _args(tmp_path, projects={"abl": project_config(partition="gpu")}),
            )
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_BILLS
        assert not any("&& sbatch " in command for command in fake_run.commands())

    def test_the_entrypoint_reads_the_process_arguments(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        argv: list[str],
        frozen_clock: str,
    ) -> None:
        """Exercised for real rather than excluded from coverage: reading
        sys.argv and raising SystemExit only happens through this door."""
        _write(tmp_path / "c.json", _payload())
        _healthy(fake_run)
        argv[:] = ["prog", *_args(tmp_path)]

        with pytest.raises(SystemExit) as excinfo:
            chain_cli.entrypoint()
        assert excinfo.value.code == 0
        assert "submitted 101 abl.pipeline-extract" in emitted[1]

    def test_the_config_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError):
            chain_cli.main(["--run", str(tmp_path / "c.json")])

    def test_the_run_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        with pytest.raises(ValueError):
            chain_cli.main(["--config", config])
