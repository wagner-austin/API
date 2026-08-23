"""Tests for the preflight CLI."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.cli import preflight as preflight_cli
from tests.conftest import FakeRun, workspace_document, write_file, write_workspace

_LINE = (
    "sbatch: Job 1 to start at 2026-08-22T03:23:00 a using 4 processors "
    "on nodes hpc3-gpu-16-02 in partition free-gpu"
)


def _run_payload(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid run document.

    Args:
        **overrides: Fields to add or replace.

    Returns:
        The document.
    """
    document: dict[str, JSONValue] = {
        "project": "abl",
        "name": "arm-b-42",
        "command": "python train.py",
        "experiment": {"arm": "B"},
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


def _ok(fake_run: FakeRun) -> None:
    """Script a cluster that admits everything.

    Args:
        fake_run: The runner to script.
    """
    fake_run.add("test -d", stdout="PRESENT\n")
    fake_run.add("--test-only", stdout=_LINE + "\nrc=0\n")


def _args(tmp_path: pathlib.Path, flag: str) -> list[str]:
    """Build a full argument list naming one document.

    Args:
        tmp_path: Directory holding the documents.
        flag: Either ``--run`` or ``--sweep``.

    Returns:
        Arguments excluding the program name.
    """
    return [
        "--config",
        write_workspace(tmp_path / "hpc3.json", workspace_document()),
        flag,
        str(tmp_path / "doc.json"),
    ]


class TestPreflightCli:
    def test_a_single_run_reports_the_verdict(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write(tmp_path / "doc.json", _run_payload())
        _ok(fake_run)

        assert preflight_cli.main(_args(tmp_path, "--run")) == 0
        assert emitted[0] == (
            "OK abl.arm-b-42: would start 2026-08-22T03:23:00 on hpc3-gpu-16-02 (4 cpu, free-gpu)"
        )
        assert emitted[1] == "1 spec(s) would be admitted; nothing was queued"

    def test_it_says_the_estimate_is_not_a_reservation(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Measured on this cluster: a 3.4-hour estimate started in 5 seconds."""
        _write(tmp_path / "doc.json", _run_payload())
        _ok(fake_run)
        preflight_cli.main(_args(tmp_path, "--run"))
        assert "not a reservation" in emitted[-1]

    def test_a_sweep_preflights_every_member(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        members: list[JSONValue] = [
            {"suffix": f"s{i}", "command": f"python t.py --seed {i}"} for i in range(3)
        ]
        _write(
            tmp_path / "doc.json",
            {
                "project": "abl",
                "name": "rung",
                "members": members,
                "experiment": {"rung": "774M"},
            },
        )
        _ok(fake_run)

        assert preflight_cli.main(_args(tmp_path, "--sweep")) == 0
        assert [line.split(":")[0] for line in emitted[:3]] == [
            "OK abl.rung-s0",
            "OK abl.rung-s1",
            "OK abl.rung-s2",
        ]
        assert emitted[3] == "3 spec(s) would be admitted; nothing was queued"

    def test_an_oversized_sweep_never_contacts_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        members: list[JSONValue] = [
            {"suffix": f"s{i}", "command": "python t.py"} for i in range(25)
        ]
        _write(
            tmp_path / "doc.json",
            {
                "project": "abl",
                "name": "rung",
                "members": members,
                "experiment": {"rung": "774M"},
            },
        )

        with pytest.raises(AppError) as excinfo:
            preflight_cli.main(_args(tmp_path, "--sweep"))
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING
        assert fake_run.calls == []

    def test_a_rejected_spec_is_not_reported_as_admitted(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write(tmp_path / "doc.json", _run_payload())
        fake_run.add("test -d", stdout="PRESENT\n")
        fake_run.add("--test-only", stdout="allocation failure: bad account\nrc=1\n")

        with pytest.raises(AppError) as excinfo:
            preflight_cli.main(_args(tmp_path, "--run"))
        assert excinfo.value.code is Hpc3ErrorCode.PREFLIGHT_REJECTED
        assert emitted == []

    def test_an_unknown_override_never_contacts_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _write(tmp_path / "doc.json", _run_payload(minute=600))
        with pytest.raises(AppError) as excinfo:
            preflight_cli.main(_args(tmp_path, "--run"))
        assert excinfo.value.code is Hpc3ErrorCode.RUN_FIELD_UNKNOWN
        assert fake_run.calls == []

    def test_naming_neither_document_is_refused(self, tmp_path: pathlib.Path) -> None:
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        with pytest.raises(ValueError, match="exactly one of --run or --sweep"):
            preflight_cli.main(["--config", config])

    def test_naming_both_documents_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Defaulting to either would preflight something the caller did not name."""
        _write(tmp_path / "doc.json", _run_payload())
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        with pytest.raises(ValueError, match="exactly one of --run or --sweep"):
            preflight_cli.main(
                [
                    "--config",
                    config,
                    "--run",
                    str(tmp_path / "doc.json"),
                    "--sweep",
                    str(tmp_path / "doc.json"),
                ]
            )

    def test_a_missing_config_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            preflight_cli.main(["--run", str(tmp_path / "doc.json")])

    def test_the_entrypoint_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], argv: list[str]
    ) -> None:
        _write(tmp_path / "doc.json", _run_payload())
        _ok(fake_run)
        argv[:] = ["prog", *_args(tmp_path, "--run")]

        with pytest.raises(SystemExit) as excinfo:
            preflight_cli.entrypoint()
        assert excinfo.value.code == 0
        assert emitted[0].startswith("OK abl.arm-b-42")
