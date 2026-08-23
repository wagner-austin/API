"""Tests for the stage, submit and watch CLIs.

Documents are written to real files and read through the production hook, so
these exercise the same path a user does. Only the command runner, the report
sink and the clock are faked.

Every command reads the workspace, and the workspace carries the budget and
the ledger. There is no test that omits them, because there is no flag that
could.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from hpc3.cli import stage as stage_cli
from hpc3.cli import submit as submit_cli
from hpc3.cli import watch as watch_cli
from tests.against_hpc3 import decode_job_status, read_ledger
from tests.conftest import (
    FakeRun,
    budget_document,
    cluster,
    gpus,
    project_config,
    script_healthy_cluster,
    workspace_document,
    write_file,
    write_workspace,
)

_PAYLOAD = b"the marker predicts extraction accuracy.\n"
_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()

_ROW = (
    "55519937|abl-verify|free-gpu32|COMPLETED|48|"
    "billing=11,cpu=11,gres/gpu=1,mem=64G,node=1|hpc3-gpu-n54-00"
)


def _write_json(path: pathlib.Path, payload: JSONValue) -> None:
    """Write a JSON document for the CLI to read.

    Args:
        path: File to write.
        payload: Document to serialise.
    """
    write_file(path, dump_json_str(payload).encode("utf-8"))


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
        "experiment": {"arm": "B", "seed": "42"},
    }
    document.update(overrides)
    return document


def _config(tmp_path: pathlib.Path, **overrides: JSONValue) -> str:
    """Write a workspace and return its path.

    Args:
        tmp_path: Directory holding the documents.
        **overrides: Workspace fields to replace.

    Returns:
        The path, ready to pass as ``--config``.
    """
    return write_workspace(tmp_path / "hpc3.json", workspace_document(**overrides))


def _submit_args(tmp_path: pathlib.Path, *, gpu_hours: float = 100.0) -> list[str]:
    """Build a full submit argument list.

    Args:
        tmp_path: Directory holding the documents.
        gpu_hours: GPU-hour cap for the workspace budget.

    Returns:
        Arguments excluding the program name.
    """
    return [
        "--config",
        _config(tmp_path, budget=budget_document(gpu_hours=gpu_hours)),
        "--run",
        str(tmp_path / "run.json"),
    ]


def _watch_args(tmp_path: pathlib.Path, jobs: str, *, gpu_hours: float = 100.0) -> list[str]:
    """Build a full watch argument list.

    Args:
        tmp_path: Directory holding the workspace.
        jobs: The ``--job`` value.
        gpu_hours: GPU-hour cap for the workspace budget.

    Returns:
        Arguments excluding the program name.
    """
    budget = budget_document(gpu_hours=gpu_hours, units=100.0)
    return ["--config", _config(tmp_path, budget=budget), "--job", jobs]


def _stage_manifest(destination: str = "/pub/wagnera3/corpora") -> dict[str, JSONValue]:
    """Build a manifest document naming the standard payload.

    Args:
        destination: Cluster directory to receive it.

    Returns:
        The document.
    """
    return {
        "destination": destination,
        "files": [{"name": "armB.txt", "sha256": _DIGEST, "size_bytes": len(_PAYLOAD)}],
        "provenance": {"wiki_commit": "176bb8c", "emitter": "emit_corpus.py"},
    }


def _stage_args(tmp_path: pathlib.Path, *, expected: str | None = None) -> list[str]:
    """Build a full stage argument list, writing the published-digest record.

    Args:
        tmp_path: Directory holding the documents.
        expected: Digest the record vouches for; defaults to the real one.

    Returns:
        Arguments excluding the program name.
    """
    record = tmp_path / "file_ids.txt"
    write_file(record, ((expected if expected is not None else _DIGEST) + "\n").encode())
    return [
        "--config",
        _config(tmp_path),
        "--manifest",
        str(tmp_path / "m.json"),
        "--source-dir",
        str(tmp_path / "src"),
        "--expect-from",
        str(record),
    ]


class TestStageCli:
    def test_it_stages_and_reports_each_file(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        write_file(tmp_path / "src" / "armB.txt", _PAYLOAD)
        _write_json(tmp_path / "m.json", _stage_manifest())
        fake_run.add("sha256sum", stdout=f"{_DIGEST}  x\n")

        assert stage_cli.main(_stage_args(tmp_path)) == 0
        assert emitted[-2:] == [
            "staged /pub/wagnera3/corpora/armB.txt",
            "verified 1 file(s) on hpc3:/pub/wagnera3/corpora",
        ]

    def test_it_reports_the_provenance_it_staged_under(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        write_file(tmp_path / "src" / "armB.txt", _PAYLOAD)
        _write_json(tmp_path / "m.json", _stage_manifest())
        fake_run.add("sha256sum", stdout=f"{_DIGEST}  x\n")

        stage_cli.main(_stage_args(tmp_path))
        assert emitted[1] == "provenance emitter=emit_corpus.py wiki_commit=176bb8c"

    def test_bytes_the_published_record_does_not_name_never_leave_the_machine(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """A corpus regenerated from the wrong source state stages clean
        against itself; only the external record catches it."""
        write_file(tmp_path / "src" / "armB.txt", _PAYLOAD)
        _write_json(tmp_path / "m.json", _stage_manifest())

        with pytest.raises(AppError) as excinfo:
            stage_cli.main(_stage_args(tmp_path, expected="a" * 64))

        assert excinfo.value.code is Hpc3ErrorCode.STAGED_DIGEST_UNEXPECTED
        assert fake_run.calls == []
        assert emitted == []

    def test_a_digest_mismatch_is_not_reported_as_success(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        write_file(tmp_path / "src" / "armB.txt", b"a different corpus\n")
        _write_json(tmp_path / "m.json", _stage_manifest(destination="/pub/x"))

        with pytest.raises(AppError) as excinfo:
            stage_cli.main(_stage_args(tmp_path))
        assert excinfo.value.code is Hpc3ErrorCode.DIGEST_MISMATCH

    def test_a_manifest_without_provenance_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        write_file(tmp_path / "src" / "armB.txt", _PAYLOAD)
        document = _stage_manifest()
        del document["provenance"]
        _write_json(tmp_path / "m.json", document)

        with pytest.raises(JSONTypeError):
            stage_cli.main(_stage_args(tmp_path))
        assert fake_run.calls == []

    def test_the_expect_from_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        """A check that runs only when remembered is not protection."""
        with pytest.raises(ValueError, match="--expect-from is required"):
            stage_cli.main(
                [
                    "--config",
                    _config(tmp_path),
                    "--manifest",
                    str(tmp_path / "m.json"),
                    "--source-dir",
                    str(tmp_path / "src"),
                ]
            )

    def test_a_missing_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--source-dir is required"):
            stage_cli.main(["--config", _config(tmp_path), "--manifest", str(tmp_path / "m.json")])


class TestSubmitCli:
    def test_it_submits_and_reports_the_job(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write_json(tmp_path / "run.json", _run_payload())
        script_healthy_cluster(fake_run)

        assert submit_cli.main(_submit_args(tmp_path)) == 0
        assert emitted[0] == "submitted 55519937 abl.arm-b-42"
        assert "A100x1 on free-gpu (free)" in emitted[1]

    def test_it_reports_where_the_logs_landed(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """A job nobody can find the output of is not inspectable."""
        _write_json(tmp_path / "run.json", _run_payload())
        script_healthy_cluster(fake_run)
        submit_cli.main(_submit_args(tmp_path))
        assert emitted[2] == "  logs /pub/w/abl/logs"

    def test_it_prints_the_command_that_watches_the_job(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write_json(tmp_path / "run.json", _run_payload())
        script_healthy_cluster(fake_run)
        submit_cli.main(_submit_args(tmp_path))
        assert emitted[3].endswith("--job 55519937")
        assert emitted[3].startswith("watch: hpc3-watch --config ")

    def test_it_records_the_job_in_the_ledger_the_workspace_names(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
    ) -> None:
        """The ledger path is not a flag, so it cannot differ from triage's."""
        _write_json(tmp_path / "run.json", _run_payload())
        script_healthy_cluster(fake_run)

        submit_cli.main(_submit_args(tmp_path))

        entries = read_ledger(tmp_path / "ledger.jsonl")
        assert [e["job_id"] for e in entries] == ["55519937"]
        assert entries[0]["submitted_at"] == frozen_clock
        assert entries[0]["name"] == "abl.arm-b-42"

    def test_a_budget_that_the_job_would_break_stops_it(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """0.1 GPU-hours cannot hold a 30-minute one-GPU job."""
        _write_json(tmp_path / "run.json", _run_payload())
        with pytest.raises(AppError) as excinfo:
            submit_cli.main(_submit_args(tmp_path, gpu_hours=0.1))
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED
        assert fake_run.calls == []
        assert emitted == []

    def test_a_billing_project_says_so_in_the_report(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write_json(tmp_path / "run.json", _run_payload())
        config = write_workspace(
            tmp_path / "hpc3.json",
            workspace_document(
                projects={
                    "abl": project_config(
                        partition="free-gpu32", gpu=gpus("L40S"), accept_billing=True
                    )
                }
            ),
        )
        script_healthy_cluster(fake_run, job_id="1")
        submit_cli.main(["--config", config, "--run", str(tmp_path / "run.json")])
        assert "BILLS service units" in emitted[1]

    def test_an_invalid_resolution_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _write_json(tmp_path / "run.json", _run_payload(gpu=gpus("gpu")))
        with pytest.raises(AppError) as excinfo:
            submit_cli.main(_submit_args(tmp_path))
        assert excinfo.value.code is Hpc3ErrorCode.GPU_TYPE_UNPINNED
        assert fake_run.calls == []

    def test_an_undeclared_project_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        _write_json(tmp_path / "run.json", _run_payload(project="sirius"))
        with pytest.raises(AppError) as excinfo:
            submit_cli.main(_submit_args(tmp_path))
        assert excinfo.value.code is Hpc3ErrorCode.WORKSPACE_PROJECT_UNKNOWN
        assert fake_run.calls == []

    def test_the_config_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        """Without it there is no host, no ledger and no cap."""
        with pytest.raises(ValueError, match="--config is required"):
            submit_cli.main(["--run", str(tmp_path / "run.json")])

    def test_the_run_flag_is_not_optional(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--run is required"):
            submit_cli.main(["--config", _config(tmp_path)])

    def test_a_retired_flag_is_refused_rather_than_ignored(self, tmp_path: pathlib.Path) -> None:
        """--host used to exist; silently ignoring it would target the wrong cluster."""
        with pytest.raises(ValueError, match="unknown argument"):
            submit_cli.main(["--config", _config(tmp_path), "--host", "hpc3"])


class TestWatchCli:
    def test_it_reports_state_and_real_cost(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout=_ROW + "\n")

        assert watch_cli.main(_watch_args(tmp_path, "55519937")) == 0
        assert emitted[0].startswith("final 55519937 abl-verify COMPLETED 48s 0.1467 SU")
        assert emitted[1] == "total 0.1467 SU across 1 row(s)"
        assert emitted[3] == "states COMPLETED=1"

    def test_it_reports_gpu_hours(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout=_ROW + "\n")
        watch_cli.main(_watch_args(tmp_path, "55519937"))
        assert emitted[2] == "gpu-hours 0.01"

    def test_an_overrun_is_reported_after_the_rows(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The rows are worth seeing whether or not the budget was broken."""
        fake_run.add("sacct", stdout=_ROW + "\n")

        with pytest.raises(AppError) as excinfo:
            watch_cli.main(_watch_args(tmp_path, "55519937", gpu_hours=0.001))
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED
        assert emitted[0].startswith("final 55519937")

    def test_the_cap_it_enforces_is_the_one_the_submitter_projected_against(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """One document, so the two commands cannot disagree about the ceiling."""
        _write_json(tmp_path / "run.json", _run_payload())
        script_healthy_cluster(fake_run)
        config = _config(tmp_path, budget=budget_document(gpu_hours=0.001))

        with pytest.raises(AppError) as submitted:
            submit_cli.main(["--config", config, "--run", str(tmp_path / "run.json")])
        assert submitted.value.code is Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED

        fake_run.add("sacct", stdout=_ROW + "\n")
        with pytest.raises(AppError) as watched:
            watch_cli.main(["--config", config, "--job", "55519937"])
        assert watched.value.code is Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED

    def test_a_pending_job_is_marked_live_with_no_node(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="1|arm|free-gpu|PENDING|0||\n")
        watch_cli.main(_watch_args(tmp_path, "1"))
        assert emitted[0].startswith("live  1 arm PENDING 0s")
        assert emitted[0].endswith("@ -")

    def test_an_unknown_job_is_a_failure_not_an_empty_report(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="")
        with pytest.raises(ValueError, match="knows no job"):
            watch_cli.main(_watch_args(tmp_path, "999"))
        assert emitted == []

    def test_the_config_flag_is_not_optional(self) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            watch_cli.main(["--job", "1"])


class TestFormatStatus:
    def test_a_terminal_row_is_marked_final(self) -> None:
        status = decode_job_status(
            {
                "job_id": "1",
                "name": "arm",
                "partition": "free-gpu",
                "state": "COMPLETED",
                "elapsed_seconds": 1,
                "billing_tres": 0,
                "gpu_count": 1,
                "cpu_count": 8,
                "node_list": "n1",
            }
        )
        assert watch_cli.format_status(status, cluster()).startswith("final ")

    def test_a_requeued_row_is_marked_live_because_protection_worked(self) -> None:
        status = decode_job_status(
            {
                "job_id": "1",
                "name": "arm",
                "partition": "free-gpu",
                "state": "REQUEUED",
                "elapsed_seconds": 1,
                "billing_tres": 0,
                "gpu_count": 1,
                "cpu_count": 8,
                "node_list": "n1",
            }
        )
        assert watch_cli.format_status(status, cluster()).startswith("live  ")


class TestEntrypoints:
    """Each ``entrypoint`` reads ``sys.argv`` and raises; that only happens
    when a process starts through it, so it is exercised for real rather than
    excluded from coverage.
    """

    def test_stage_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], argv: list[str]
    ) -> None:
        write_file(tmp_path / "src" / "armB.txt", _PAYLOAD)
        _write_json(tmp_path / "m.json", _stage_manifest(destination="/pub/x"))
        fake_run.add("sha256sum", stdout=f"{_DIGEST}  x\n")
        argv[:] = ["prog", *_stage_args(tmp_path)]

        with pytest.raises(SystemExit) as excinfo:
            stage_cli.entrypoint()
        assert excinfo.value.code == 0
        assert emitted[-1] == "verified 1 file(s) on hpc3:/pub/x"

    def test_submit_reads_the_process_arguments(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        argv: list[str],
        frozen_clock: str,
    ) -> None:
        _write_json(tmp_path / "run.json", _run_payload())
        script_healthy_cluster(fake_run)
        argv[:] = ["prog", *_submit_args(tmp_path)]

        with pytest.raises(SystemExit) as excinfo:
            submit_cli.entrypoint()
        assert excinfo.value.code == 0
        assert emitted[0] == "submitted 55519937 abl.arm-b-42"

    def test_watch_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], argv: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="1|arm|free-gpu|COMPLETED|10|billing=4,gres/gpu=1|n1\n")
        argv[:] = ["prog", *_watch_args(tmp_path, "1", gpu_hours=10.0)]

        with pytest.raises(SystemExit) as excinfo:
            watch_cli.entrypoint()
        assert excinfo.value.code == 0
        assert emitted[-1] == "budget OK"


class TestMultiJobWatch:
    def test_one_sacct_call_covers_the_whole_sweep(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Six separate calls would observe six different moments."""
        rows = "\n".join(
            f"10{i}|rung-s{i}|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|node{i}" for i in range(3)
        )
        fake_run.add("sacct", stdout=rows + "\n")

        code = watch_cli.main(_watch_args(tmp_path, "101,102,103", gpu_hours=10.0))
        assert code == 0
        assert len(fake_run.calls) == 1
        assert "sacct -j 101,102,103" in fake_run.calls[0].remote_command

    def test_it_tallies_states_across_the_sweep(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        rows = (
            "101|a|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n"
            "102|b|free-gpu|RUNNING|30|billing=8,gres/gpu=1|n2\n"
            "103|c|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n3\n"
        )
        fake_run.add("sacct", stdout=rows)
        watch_cli.main(_watch_args(tmp_path, "101,102,103", gpu_hours=10.0))
        assert emitted[5] == "states COMPLETED=2 RUNNING=1"

    def test_an_id_accounting_does_not_know_is_named_not_hidden(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="101|a|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n")
        watch_cli.main(_watch_args(tmp_path, "101,999", gpu_hours=10.0))
        assert emitted[4] == "NOT FOUND 999"

    def test_trailing_commas_are_ignored(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        fake_run.add("sacct", stdout="101|a|free-gpu|COMPLETED|60|billing=8,gres/gpu=1|n1\n")
        watch_cli.main(_watch_args(tmp_path, "101,,", gpu_hours=10.0))
        assert "sacct -j 101 " in fake_run.calls[0].remote_command

    def test_a_job_argument_of_only_commas_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        with pytest.raises(ValueError, match="at least one job id"):
            watch_cli.main(_watch_args(tmp_path, ",,"))
