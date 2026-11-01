"""The evaluator's preparation reading: real journal, real recipe, real reporter."""

import runpy
import subprocess
import sys
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments, stack_endpoint_files
from platform_core.session_label import API_KEY_NAME, STACK_ENV_PATH, TENANT_ID_NAME
from scripts.research_preparation import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.research_cluster import prepare_bundle
from covenant_ml.markets.research_journal import research_journal, research_policy_digest
from covenant_ml.markets.research_preparation_report import (
    decode_prepared_recipe,
    encode_prepared_recipe,
    report_preparation,
)
from covenant_ml.markets.research_records import encode_research_policy
from covenant_ml.markets.research_trial_records import ResearchTrial
from covenant_ml.markets.validation import read_document

from ._research_execution_fixtures import setup

TASK = "e1621f6f-7c31-4ec1-bae1-8657053e707f"


@pytest.fixture()
def board() -> Generator[FakeHttpPost, None, None]:
    """Capture the real reporter's writes at the existing transport seam."""
    files = stack_endpoint_files()
    files[STACK_ENV_PATH] = f"{API_KEY_NAME}=test-key\n{TENANT_ID_NAME}=test-tenant\n"
    previous = (_test_hooks.http_post, _test_hooks.stack_text)
    post = FakeHttpPost([posted_ok(), posted_ok()])
    _test_hooks.http_post, _test_hooks.stack_text = post, files.__getitem__
    yield post
    _test_hooks.http_post, _test_hooks.stack_text = previous


def _prepared(root: Path) -> tuple[Path, Path, Path, ResearchTrial]:
    """Admit attempt 1, freeze its policy and let ``prepare_bundle`` write its recipe."""
    ledger, trial, cluster, directory = setup(root)
    policy = root / "policy.json"
    policy.write_text(dump_json_str(encode_research_policy(trial["policy"])), "utf-8")
    prepare_bundle(directory, cluster, trial)
    return policy, ledger, directory / "run.json", trial


def _rewrite(recipe: Path, change: JSONObject, experiment: JSONObject) -> None:
    """Alter a prepared recipe's top-level and experiment fields in place."""
    document = read_document(recipe)
    document.update(change)
    require_dict(document, "experiment").update(experiment)
    recipe.write_text(dump_json_str(document), "utf-8")


def test_reading_is_posted_by_the_program_and_printed(
    tmp_path: Path, board: FakeHttpPost, capsys: pytest.CaptureFixture[str]
) -> None:
    policy, ledger, recipe, trial = _prepared(tmp_path)
    before = ledger.read_bytes()
    command = ["--policy", str(policy), "--ledger", str(ledger), "--recipe", str(recipe)]
    assert main([*command, "--report-task", TASK]) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    frozen = trial["policy"]
    digest = research_policy_digest(frozen)
    row = research_journal(frozen).read(ledger)[0]
    assert report["kind"] == "RESEARCH_PREPARATION_EVIDENCE"
    assert report["policy_digest"] == digest
    assert report["seeds"] == list(frozen["seeds"])
    assert (report["cpus"], report["mem_gb"], report["max_seconds"]) == (
        frozen["cpus"],
        frozen["mem_gb"],
        frozen["max_seconds"],
    )
    assert report["max_candidates"] == frozen["max_candidates"]
    assert report["attempts"] == [
        {
            "number": 1,
            "status": "admitted",
            "seed": trial["proposal"]["seed"],
            "candidate": row["candidate"],
            "job_id": "",
        }
    ]
    assert report["recipe"] == {
        "name": f"research-{digest[:12]}-1",
        "research_attempt": f"{digest}:1",
        "request_sha256": sha256((recipe.parent / "request.json").read_bytes()).hexdigest(),
        "cpus": frozen["cpus"],
        "mem_gb": frozen["mem_gb"],
        "minutes": (trial["proposal"]["seconds"] + 64) // 60,
        "partition": "free",
        "gpu_requested": False,
        "requeue": False,
    }
    assert report["recipe_sha256"] == sha256(recipe.read_bytes()).hexdigest()
    assert report["journal_sha256"] == sha256(before).hexdigest()
    assert report["ledger"] == str(ledger.resolve())
    assert ledger.read_bytes() == before
    posted = sent_arguments(board.bodies[1])
    assert posted["body"] == "RESEARCH PREPARATION LOAD-REPORT\n" + dump_json_str(report)
    assert (posted["agent"], posted["taskId"]) == ("hpc-report-e1621f6f", TASK)


def test_final_recipe_of_an_admitted_attempt_is_read(tmp_path: Path, board: FakeHttpPost) -> None:
    policy, ledger, recipe, trial = _prepared(tmp_path)
    final = f"{research_policy_digest(trial['policy'])}:1:final"
    _rewrite(recipe, {}, {"research_attempt": final})
    report = report_preparation(policy, ledger, recipe, TASK)
    assert require_dict(report, "recipe")["research_attempt"] == final


@pytest.mark.parametrize(
    ("change", "attempt", "error"),
    [
        ({}, "e" * 64 + ":1", "RESEARCH_PREPARATION_POLICY: recipe names policy eeee"),
        ({}, "{digest}:2", "RESEARCH_PREPARATION_ATTEMPT: attempt 2 was not admitted"),
        ({"cpus": 128}, "{digest}:1", "RESEARCH_PREPARATION_RESOURCES"),
        ({"mem_gb": 512}, "{digest}:1", "RESEARCH_PREPARATION_RESOURCES"),
    ],
)
def test_foreign_recipe_is_refused_before_posting(
    tmp_path: Path, board: FakeHttpPost, change: JSONObject, attempt: str, error: str
) -> None:
    policy, ledger, recipe, trial = _prepared(tmp_path)
    digest = research_policy_digest(trial["policy"])
    _rewrite(recipe, change, {"research_attempt": attempt.format(digest=digest)})
    with pytest.raises(ValueError, match=error):
        report_preparation(policy, ledger, recipe, TASK)
    assert board.bodies == []


def test_journal_of_another_policy_is_refused(tmp_path: Path, board: FakeHttpPost) -> None:
    policy, ledger, recipe, _trial = _prepared(tmp_path)
    document = read_document(policy)
    document["max_seconds"] = 7200
    policy.write_text(dump_json_str(document), "utf-8")
    with pytest.raises(ValueError, match="JOURNAL-001"):
        report_preparation(policy, ledger, recipe, TASK)
    assert board.bodies == []


def test_malformed_task_is_refused_before_any_file_is_read(
    tmp_path: Path, board: FakeHttpPost
) -> None:
    absent = tmp_path / "absent.json"
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        report_preparation(absent, absent, absent, "e1621f6f")
    assert board.bodies == []


def test_board_refusal_fails_the_reading(tmp_path: Path, board: FakeHttpPost) -> None:
    policy, ledger, recipe, _trial = _prepared(tmp_path)
    refusal = McpHttpResponse(status=401, body="bad key", content_type=EVENT_STREAM_MEDIA_TYPE)
    _test_hooks.http_post = FakeHttpPost([posted_ok(), refusal])
    with pytest.raises(AppError, match="HTTP 401"):
        report_preparation(policy, ledger, recipe, TASK)


@pytest.mark.parametrize(
    ("gpu", "requested"), [(None, False), ({"type": "a100", "count": 1}, True)]
)
def test_recipe_codec_reads_the_gpu_request(
    tmp_path: Path, gpu: JSONValue, requested: bool
) -> None:
    _policy, _ledger, recipe, _trial = _prepared(tmp_path)
    document = read_document(recipe)
    document["gpu"] = gpu
    encoded = encode_prepared_recipe(decode_prepared_recipe(document))
    assert encoded["gpu_requested"] is requested
    assert (encoded["partition"], encoded["requeue"]) == ("free", False)


def test_recipe_without_a_gpu_declaration_is_refused(tmp_path: Path) -> None:
    _policy, _ledger, recipe, _trial = _prepared(tmp_path)
    document = read_document(recipe)
    del document["gpu"]
    with pytest.raises(ValueError, match="RESEARCH_PREPARATION_RECIPE: gpu must be declared"):
        decode_prepared_recipe(document)


def test_real_process_refuses_a_malformed_task(tmp_path: Path) -> None:
    absent = str(tmp_path / "absent.json")
    command = ["--policy", absent, "--ledger", absent, "--recipe", absent]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.research_preparation", *command, "--report-task", "x"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr.splitlines()[-1] == (
        "ValueError: WRDS_REPORT_TASK: expected a full task id, got 'x'"
    )


def test_entrypoint_requires_its_flags() -> None:
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("scripts.research_preparation", run_name="__main__")
    assert exited.value.code == 2
