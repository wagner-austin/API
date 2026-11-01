"""The evaluator's summary reading: real journal, real summary, real reporter."""

import runpy
import shutil
import subprocess
import sys
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
)
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments, stack_endpoint_files
from platform_core.run_record import run_record_sidecar
from platform_core.session_label import API_KEY_NAME, STACK_ENV_PATH, TENANT_ID_NAME
from platform_core.testing import sample_run_fingerprint
from scripts.research_summary_report import main

from covenant_ml.markets import _test_hooks, research_summary
from covenant_ml.markets.research_journal import (
    admit_proposal,
    encode_research_attempt,
    research_journal,
    research_policy_digest,
)
from covenant_ml.markets.research_records import encode_research_proposal
from covenant_ml.markets.research_summary import write_validation_summary
from covenant_ml.markets.research_summary_report import SUMMARY_LABEL, report_summary
from covenant_ml.markets.research_trial_records import (
    ResearchTrial,
    decode_research_trial,
    encode_research_trial,
)
from covenant_ml.markets.validation import read_document

from ._research_final_fixtures import completed_study, write_trial

TASK = "673a4bc4-8693-4764-8a8d-cca0483e279f"


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Run the real study once and let the real summarizer write its summary."""
    root = tmp_path_factory.mktemp("study")
    ledger, trial, _final = completed_study(root)
    request = write_trial(trial, root / "trial.json")
    output = root / "summary" / "result.json"
    write_validation_summary(ledger, trial, output, sample_run_fingerprint())
    return ledger, request, output


@pytest.fixture()
def copied(study: tuple[Path, Path, Path], tmp_path: Path) -> tuple[Path, Path, Path]:
    """Give each test its own journal and summary, written once by the real owners."""
    ledger, request, output = study
    shutil.copyfile(ledger, tmp_path / "journal.sqlite")
    shutil.copytree(output.parent, tmp_path / "summary")
    return tmp_path / "journal.sqlite", request, tmp_path / "summary" / "result.json"


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


def _trial(request: Path) -> ResearchTrial:
    """Read the frozen request the summarizer was given."""
    return decode_research_trial(read_document(request))


def test_verified_reading_is_posted_by_the_program_and_printed(
    copied: tuple[Path, Path, Path], board: FakeHttpPost, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, request, output = copied
    before = ledger.read_bytes()
    command = ["--ledger", str(ledger), "--trial", str(request), "--summary", str(output)]
    assert main([*command, "--report-task", TASK]) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    trial = _trial(request)
    rows = research_journal(trial["policy"]).read(ledger)
    document = read_document(output)
    sidecar = run_record_sidecar(output)
    record = narrow_json_to_dict(load_json_str(sidecar.read_text()))
    assert report["kind"] == "RESEARCH_SUMMARY_EVIDENCE"
    assert report["policy_digest"] == research_policy_digest(trial["policy"])
    assert report["required_seeds"] == list(trial["policy"]["seeds"])
    owner = Path(research_summary.__file__).read_bytes()
    assert report["summary_code_sha256"] == sha256(owner).hexdigest()
    assert report["numpy"] == np.__version__
    assert report["journal_sha256"] == sha256(before).hexdigest()
    assert report["journal_sha256_after"] == report["journal_sha256"]
    assert ledger.read_bytes() == before
    assert (
        report["journal_snapshot_sha256"]
        == sha256(
            dump_json_str([encode_research_attempt(row) for row in rows]).encode()
        ).hexdigest()
        == document["journal_snapshot_sha256"]
    )
    assert report["attempts"] == [
        {
            "number": row["number"],
            "status": "complete",
            "seed": seed,
            "candidate": row["candidate"],
            "job_id": f"slurm:{row['number']}",
            "record_sha256": sha256(row["record"].encode()).hexdigest(),
        }
        for row, seed in zip(rows, trial["policy"]["seeds"], strict=True)
    ]
    assert report["payload_sha256"] == sha256(output.read_bytes()).hexdigest()
    assert report["record_sha256"] == sha256(sidecar.read_bytes()).hexdigest()
    assert (report["experiment"], report["payload_digest"]) == (
        record["experiment"],
        record["payload_digest"],
    )
    assert record["label"] == SUMMARY_LABEL
    assert report["interpretation"] == document["interpretation"]
    assert report["final_scoring"] == document["final_scoring"]
    assert report["recomputed_matches"] is True
    written = [narrow_json_to_dict(item) for item in require_list(document, "results")]
    readings = [narrow_json_to_dict(item) for item in require_list(report, "results")]
    assert [item["arm"] for item in readings] == [item["arm"] for item in written]
    assert len(readings) == 4
    for reading, result in zip(readings, written, strict=True):
        assert reading["interval"] == result["interval"]
        assert (reading["n_seeds"], reading["n_months"]) == (2, 16)
        assert reading["attempts"] == [1, 2]
        assert reading["seeds"] == list(trial["policy"]["seeds"])
        assert reading["candidate"] == result["candidate"]
        assert reading["reference"] == "momentum"
    posted = sent_arguments(board.bodies[1])
    assert posted["body"] == "RESEARCH SUMMARY LOAD-REPORT\n" + dump_json_str(report)
    assert (posted["agent"], posted["taskId"]) == ("hpc-report-673a4bc4", TASK)


def _rewrite(output: Path, change: JSONObject) -> None:
    """Alter the written summary payload's top-level fields in place."""
    document = read_document(output)
    document.update(change)
    output.write_text(dump_json_str(document, indent=1), encoding="utf-8")


def _replace_first_means(output: Path) -> None:
    """Replace the first result's averaged monthly differences with a constant series."""
    document = read_document(output)
    first = narrow_json_to_dict(require_list(document, "results")[0])
    months = len(require_list(first, "mean_paired_differences"))
    first["mean_paired_differences"] = [1.0 for _ in range(months)]
    output.write_text(dump_json_str(document, indent=1), encoding="utf-8")


def _replace_first_observation(output: Path) -> None:
    """Replace one observation value in the summary's RunRecord sidecar."""
    sidecar = run_record_sidecar(output)
    record = narrow_json_to_dict(load_json_str(sidecar.read_text()))
    first = narrow_json_to_dict(require_list(record, "observations")[0])
    first["value"] = 1.0e9
    sidecar.write_text(dump_json_str(record, indent=1), encoding="utf-8")


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("code", "RESEARCH_SUMMARY_CODE"),
        ("journal", "RESEARCH_SUMMARY_JOURNAL"),
        ("policy", "RESEARCH_SUMMARY_POLICY"),
        ("results", "RESEARCH_SUMMARY_RESULTS"),
        ("record", "RESEARCH_SUMMARY_RECORD"),
    ],
)
def test_summary_its_journal_does_not_reproduce_is_refused_before_posting(
    copied: tuple[Path, Path, Path], board: FakeHttpPost, fault: str, error: str
) -> None:
    ledger, request, output = copied
    trial = _trial(request)
    if fault == "code":
        _rewrite(output, {"reporter_sha256": "e" * 64})
    elif fault == "journal":
        duplicate = encode_research_proposal(trial["proposal"])
        refused = admit_proposal(ledger, trial["policy"], duplicate, 5000)
        assert refused["status"] == "refused"
    elif fault == "policy":
        _rewrite(output, {"policy_digest": "e" * 64})
    elif fault == "results":
        _replace_first_means(output)
    else:
        _replace_first_observation(output)
    with pytest.raises(ValueError, match=error):
        report_summary(ledger, request, output, TASK)
    assert board.bodies == []


def test_trial_the_journal_never_admitted_is_refused(
    copied: tuple[Path, Path, Path], board: FakeHttpPost, tmp_path: Path
) -> None:
    ledger, request, output = copied
    document = encode_research_trial(_trial(request))
    document["number"] = 3
    foreign = tmp_path / "trial-3.json"
    foreign.write_text(dump_json_str(document), encoding="utf-8")
    with pytest.raises(ValueError, match="RESEARCH_EXECUTION_REQUEST"):
        report_summary(ledger, foreign, output, TASK)
    assert board.bodies == []


def test_malformed_task_is_refused_before_any_file_is_read(
    tmp_path: Path, board: FakeHttpPost
) -> None:
    absent = tmp_path / "absent.json"
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        report_summary(absent, absent, absent, "673a4bc4")
    assert board.bodies == []


def test_board_refusal_fails_the_reading(
    copied: tuple[Path, Path, Path], board: FakeHttpPost
) -> None:
    ledger, request, output = copied
    refusal = McpHttpResponse(status=401, body="bad key", content_type=EVENT_STREAM_MEDIA_TYPE)
    _test_hooks.http_post = FakeHttpPost([posted_ok(), refusal])
    before = (ledger.read_bytes(), output.read_bytes())
    with pytest.raises(AppError, match="HTTP 401"):
        report_summary(ledger, request, output, TASK)
    assert (ledger.read_bytes(), output.read_bytes()) == before


def test_real_process_refuses_a_malformed_task(tmp_path: Path) -> None:
    absent = str(tmp_path / "absent.json")
    command = ["--ledger", absent, "--trial", absent, "--summary", absent]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.research_summary_report", *command, "--report-task", "x"],
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
        runpy.run_module("scripts.research_summary_report", run_name="__main__")
    assert exited.value.code == 2
