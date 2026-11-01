"""Real proposal CLI persistence, refusal exit codes and numerical startup boundary."""

import runpy
import subprocess
import sys
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.errors import AppError
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments, stack_endpoint_files
from platform_core.session_label import API_KEY_NAME, STACK_ENV_PATH, TENANT_ID_NAME
from scripts.research_proposals import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.research_journal import research_journal
from covenant_ml.markets.research_records import encode_research_policy
from covenant_ml.markets.research_selection_report import report_selection
from covenant_ml.markets.research_trial_records import decode_research_trial, encode_research_trial
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._research_final_fixtures import completed_study, write_trial
from ._research_fixtures import policy, proposal

TASK = "a9b2927c-3775-46cc-b86e-2aeb0c2c4479"


@pytest.fixture()
def board() -> Generator[FakeHttpPost, None, None]:
    """Capture the real reporter's writes at the existing transport seam."""
    files = stack_endpoint_files()
    files[STACK_ENV_PATH] = f"{API_KEY_NAME}=test-key\n{TENANT_ID_NAME}=test-tenant\n"
    previous = (_test_hooks.http_post, _test_hooks.stack_text)
    post = FakeHttpPost([posted_ok() for _ in range(6)])
    _test_hooks.http_post, _test_hooks.stack_text = post, files.__getitem__
    yield post
    _test_hooks.http_post, _test_hooks.stack_text = previous


def _files(root: Path) -> list[str]:
    policy_path = root / "policy.json"
    policy_path.write_text(dump_json_str(encode_research_policy(policy())), encoding="utf-8")
    return ["--policy", str(policy_path), "--ledger", str(root / "attempts.sqlite")]


def test_init_admit_refuse_and_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _files(tmp_path)
    assert main([*args, "init"], stand_in_pin) == 0
    assert narrow_json_to_dict(load_json_str(capsys.readouterr().out))["attempts"] == []
    request = tmp_path / "proposal.json"
    request.write_text(dump_json_str(proposal()), encoding="utf-8")
    call = [*args, "propose", "--proposal", str(request)]
    assert main(call, stand_in_pin) == 0
    admitted = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert narrow_json_to_dict(require_list(admitted, "attempts")[0])["status"] == "admitted"
    assert main(call, stand_in_pin) == 2
    refused = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert narrow_json_to_dict(require_list(refused, "attempts")[0])["reason"] == (
        "candidate_seed_consumed"
    )
    assert main([*args, "status"], stand_in_pin) == 0
    status = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert require_list(status, "attempts") == [
        *require_list(admitted, "attempts"),
        *require_list(refused, "attempts"),
    ]


def test_real_process_refusal_and_restart(tmp_path: Path) -> None:
    args = _files(tmp_path)
    command = [sys.executable, "-m", "scripts.research_proposals", *args]
    initialized = subprocess.run(
        [*command, "init"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert narrow_json_to_dict(load_json_str(initialized.stdout))["attempts"] == []
    request = proposal()
    request["partition"] = "final"
    request_path = tmp_path / "proposal.json"
    request_path.write_text(dump_json_str(request), encoding="utf-8")
    refused = subprocess.run(
        [*command, "propose", "--proposal", str(request_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert refused.returncode == 2
    report = narrow_json_to_dict(load_json_str(refused.stdout))
    row = narrow_json_to_dict(require_list(report, "attempts")[0])
    assert row["reason"] == "holdout_forbidden"
    restarted = subprocess.run(
        [*command, "status"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert narrow_json_to_dict(load_json_str(restarted.stdout)) == report


def test_evaluator_selection_seals_the_journal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], board: FakeHttpPost
) -> None:
    ledger, trial, manifest = completed_study(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(dump_json_str(encode_research_policy(trial["policy"])), "utf-8")
    request = write_trial(trial, tmp_path / "request.json")
    args = ["--policy", str(policy_path), "--ledger", str(ledger)]
    selection = [
        "select",
        "--trial",
        str(request),
        "--final-manifest",
        str(manifest),
        "--report-task",
        TASK,
    ]
    before = sha256(ledger.read_bytes()).hexdigest()
    assert main([*args, *selection], stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert report["selected"] == [
        {
            "number": 1,
            "seed": 11,
            "candidate": research_journal(trial["policy"]).read(ledger)[0]["candidate"],
            "selection": "selected",
            "final_manifest_digest": sha256(manifest.read_bytes()).hexdigest(),
        }
    ]
    assert report["journal_sha256_before"] == before
    assert report["journal_sha256_after"] == sha256(ledger.read_bytes()).hexdigest()
    assert report["journal_sha256_after"] != before
    assert (report["exit_status"], report["messages"]) == (0, [])
    assert sent_arguments(board.bodies[1])["body"] == (
        "RESEARCH SELECTION LOAD-REPORT\n" + dump_json_str(report)
    )
    assert sent_arguments(board.bodies[1])["agent"] == "hpc-report-a9b2927c"
    assert main([*args, *selection], stand_in_pin) == 2
    refused = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert refused["messages"] == [
        "usage: [RESEARCH-008] selection requires settled, unsealed validation"
    ]
    assert refused["journal_sha256_before"] == refused["journal_sha256_after"]
    assert refused["selected"] == report["selected"]
    release = [
        "release",
        "--trial",
        str(request),
        "--final-manifest",
        str(manifest),
        "--access",
        str(tmp_path / "validation/access.json"),
        "--directory",
        str(tmp_path / "grant"),
    ]
    assert main([*args, *release], stand_in_pin) == 0
    released = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert narrow_json_to_dict(require_list(released, "attempts")[0])["selection"] == "released"
    assert (tmp_path / "grant/final.agekey").is_file()
    assert set(released) == {"policy_digest", "attempts"}
    with pytest.raises(ValueError, match="RESEARCH-009"):
        main([*args, *release], stand_in_pin)


def test_wrong_seed_is_reported_without_changing_the_journal(
    tmp_path: Path, board: FakeHttpPost
) -> None:
    ledger, trial, manifest = completed_study(tmp_path)
    document = encode_research_trial(trial)
    document["number"] = 2
    require_dict(document, "proposal")["seed"] = 23
    before = ledger.read_bytes()
    code, report = report_selection(ledger, decode_research_trial(document), manifest, TASK)
    assert code == report["exit_status"] == 2
    assert report["messages"] == [
        "usage: RESEARCH_SELECTION_WINNER: require the winner's fixed final seed"
    ]
    assert report["selected"] == []
    assert (report["attempt"], report["seed"]) == (2, 23)
    assert report["journal_sha256_before"] == report["journal_sha256_after"]
    assert ledger.read_bytes() == before
    assert sent_arguments(board.bodies[1])["body"] == (
        "RESEARCH SELECTION LOAD-REPORT\n" + dump_json_str(report)
    )


def test_reporting_failure_cannot_restore_selection(tmp_path: Path, board: FakeHttpPost) -> None:
    ledger, trial, manifest = completed_study(tmp_path)
    refusal = McpHttpResponse(status=401, body="bad key", content_type=EVENT_STREAM_MEDIA_TYPE)
    _test_hooks.http_post = FakeHttpPost([posted_ok(), refusal])
    with pytest.raises(AppError, match="HTTP 401"):
        report_selection(ledger, trial, manifest, TASK)
    assert research_journal(trial["policy"]).read(ledger)[0]["selection"] == "selected"


def test_malformed_task_is_refused_before_selection(tmp_path: Path, board: FakeHttpPost) -> None:
    ledger, trial, manifest = completed_study(tmp_path)
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        report_selection(ledger, trial, manifest, "bad-task")
    assert ledger.read_bytes() == before
    assert board.bodies == []


def test_entry_refuses_late_numeric_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.research_proposals", run_name="__main__")
