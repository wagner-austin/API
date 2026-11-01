"""Handover preserves request, journal and native evidence while refusing drift."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_str,
)
from platform_core.mcp_client import McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, sent_arguments

from fleet.cli import research_native
from fleet.core import _test_hooks, research_handover
from tests._handover_fixtures import TASK, freeze, pin, replies, reporting, setup


def test_completed_handover_reports_both_identities_without_changing_inputs(
    tmp_path: pathlib.Path,
) -> None:
    request, document, plan, measured = setup(tmp_path)
    digest = freeze(tmp_path, plan, "handover.json")
    runner = replies(measured)
    _test_hooks.run = runner
    poster = reporting()
    original = (tmp_path / request["spec_sha256"] / "request.json").read_bytes()
    result = research_handover.poll(tmp_path, request, document, digest, TASK, ())
    assert result == {
        "state": "complete",
        "native_job_id": "12345",
        "record_sha256": plan["record_sha256"],
        "handover_sha256": digest,
    }
    assert (tmp_path / request["spec_sha256"] / "request.json").read_bytes() == original
    assert [call[2] for call in runner.calls] == [
        "hpc3.cli.controller_inspect",
        "--policy",
        "hpc3.cli.execution_inspect",
    ]
    assert runner.calls[1][-1] == "status"
    assert runner.calls[2][7:9] == ("--artifact", "/cluster/study/attempt-1/result.json")
    body = require_str(sent_arguments(poster.bodies[-1]), "body")
    report = narrow_json_to_dict(load_json_str(body.split("\n", 1)[1]))
    assert report["request_sha256"] == request["spec_sha256"]
    assert report["replacement_sha256"] == plan["replacement_sha256"]
    assert report["original_python"] == document["python"]
    assert report["replacement_controller"] == measured


@pytest.mark.parametrize(
    "field",
    [
        "extra",
        "request_sha256",
        "replacement_sha256",
        "recipe",
        "checksums",
        "provenance",
        "selfcheck",
        "native_job_id",
        "record_sha256",
    ],
)
def test_plan_drift_refuses_before_publication(tmp_path: pathlib.Path, field: str) -> None:
    request, document, plan, measured = setup(tmp_path)
    if field in {"recipe", "checksums", "provenance", "selfcheck"}:
        require_dict(plan, field)["sha256"] = "a" * 64
    elif field == "replacement_sha256":
        target = {**document, "python": "/different", "ledger": "/different/journal"}
        plan[field] = freeze(tmp_path, target, "request.json")
    else:
        plan[field] = "a" * 64
    digest = freeze(tmp_path, plan, "handover.json")
    runner = replies(measured)
    _test_hooks.run = runner
    poster = reporting()
    with pytest.raises(ValueError, match=r"RESEARCH_(HANDOVER|INPUT)_"):
        research_handover.poll(tmp_path, request, document, digest, TASK, ())
    assert poster.bodies == []


@pytest.mark.parametrize(
    "changed", ["backend", "phase", "same-python", "recipe-path", "provenance-python"]
)
def test_other_lifecycle_or_paths_cannot_be_handed_over(
    tmp_path: pathlib.Path, changed: str
) -> None:
    request, document, plan, measured = setup(tmp_path)
    if changed in {"backend", "phase"}:
        document[changed] = "wrds" if changed == "backend" else "final"
    elif changed == "same-python":
        plan["replacement_sha256"] = request["spec_sha256"]
    elif changed == "recipe-path":
        other = tmp_path / "other-run.json"
        other.write_text((tmp_path / "work/run.json").read_text())
        plan["recipe"] = pin(other)
    else:
        source = tmp_path / "provenance.json"
        source.write_text('{"controller_python":"/another/python"}')
        plan["provenance"] = pin(source)
    digest = freeze(tmp_path, plan, "handover.json")
    runner = replies(measured)
    _test_hooks.run = runner
    with pytest.raises(ValueError, match="RESEARCH_HANDOVER_"):
        research_handover.poll(tmp_path, request, document, digest, TASK, ())
    assert runner.calls == []


@pytest.mark.parametrize(
    "field", ["controller_python", "controller_python_sha256", "provenance_sha256"]
)
def test_changed_controller_measurement_refuses(tmp_path: pathlib.Path, field: str) -> None:
    request, document, plan, measured = setup(tmp_path)
    measured[field] = "changed"
    _test_hooks.run = replies(measured)
    with pytest.raises(ValueError, match="RESEARCH_HANDOVER_CONTROLLER"):
        research_handover.poll(
            tmp_path, request, document, freeze(tmp_path, plan, "handover.json"), TASK, ()
        )


def test_actual_replacement_process_failure_retains_original_request(
    tmp_path: pathlib.Path,
) -> None:
    request, document, plan, _ = setup(tmp_path)
    digest = freeze(tmp_path, plan, "handover.json")
    _test_hooks.run = _test_hooks._default_run
    with pytest.raises(ValueError, match="RESEARCH_CONTROLLER_PROCESS: exit 7"):
        research_handover.poll(tmp_path, request, document, digest, TASK, ())
    assert not (tmp_path / request["spec_sha256"] / "handover.sha256").exists()
    log = tmp_path / request["spec_sha256"] / "controller.jsonl"
    assert narrow_json_to_dict(load_json_str(log.read_text()))["exit"] == 7


def test_marker_is_exclusive_and_requires_exact_readback(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "handover.sha256"
    research_handover.record(path, "a" * 64)
    assert path.read_text() == "a" * 64 + "\n"
    with pytest.raises(FileExistsError):
        research_handover.record(path, "b" * 64)
    assert path.read_text() == "a" * 64 + "\n"

    def changed_read(path: pathlib.Path) -> str:
        return "changed " + path.read_text()

    _test_hooks.read_text = changed_read
    with pytest.raises(ValueError, match="RESEARCH_HANDOVER_WRITE"):
        research_handover.record(tmp_path / "other", "a" * 64)


def test_native_cli_records_once_then_normal_poll_remeasures(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request, _, plan, measured = setup(tmp_path)
    digest = freeze(tmp_path, plan, "handover.json")
    env = tmp_path / "env"
    env.write_text("")
    args = [
        "--root",
        str(tmp_path),
        "--backend",
        "hpc3",
        "--digest",
        request["spec_sha256"],
        "--attempt",
        request["attempt"],
        "--tenant",
        "ours",
        "--task",
        TASK,
        "--env-file",
        str(env),
    ]
    for extra in (["--handover", digest], []):
        _test_hooks.run = replies(measured)
        reporting()
        assert research_native.main([*args, *extra]) == 0
        report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
        assert report["handover_sha256"] == digest
        assert report["native_job_id"] == "12345"
    assert (tmp_path / request["spec_sha256"] / "handover.sha256").read_text() == digest + "\n"
    with pytest.raises(ValueError, match="RESEARCH_HANDOVER_EXISTS"):
        research_native.main([*args, "--handover", digest])
    assert not (tmp_path / request["spec_sha256"] / "poll.lock").exists()


def test_report_failure_never_activates_handover(tmp_path: pathlib.Path) -> None:
    request, _, plan, measured = setup(tmp_path)
    digest = freeze(tmp_path, plan, "handover.json")
    env = tmp_path / "env"
    env.write_text("")
    _test_hooks.run = replies(measured)
    reporting()
    _test_hooks.http_post = FakeHttpPost(
        [McpHttpResponse(status=401, body="denied", content_type="text/plain")]
    )
    with pytest.raises(AppError):
        research_native.main(
            [
                "--root",
                str(tmp_path),
                "--backend",
                "hpc3",
                "--digest",
                request["spec_sha256"],
                "--attempt",
                request["attempt"],
                "--tenant",
                "ours",
                "--task",
                TASK,
                "--env-file",
                str(env),
                "--handover",
                digest,
            ]
        )
    assert not (tmp_path / request["spec_sha256"] / "handover.sha256").exists()
    assert not (tmp_path / request["spec_sha256"] / "poll.lock").exists()
