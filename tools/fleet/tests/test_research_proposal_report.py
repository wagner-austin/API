"""Service reports bind a pinned proposal specification to its admission outcome."""

import hashlib
from pathlib import Path

import pytest
from board_watch import _test_hooks as board_hooks
from platform_core.json_utils import dump_json_str
from platform_core.mcp_testing import (
    DECLARED_TASKBOARD_URL,
    FakeHttpPost,
    posted_ok,
    sent_arguments,
)

from fleet.cli import research_propose
from fleet.core import _test_hooks
from tests.conftest import FakeRun, ok
from tests.test_research_proposal import PROPOSAL, report, specification
from tests.test_research_proposer import replies

TASK = "45ff6e73-ec8c-41ff-aae4-39d00423dddb"


def test_service_publishes_the_controller_outcome(tmp_path: Path) -> None:
    document = specification(tmp_path)
    document["host"] = "colossus"
    payload = dump_json_str(document)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (tmp_path / digest).mkdir()
    (tmp_path / digest / "proposal.json").write_text(payload)
    _test_hooks.run = FakeRun([*replies(dump_json_str(PROPOSAL)), ok(report("admitted", ""))])
    poster = FakeHttpPost([posted_ok(), posted_ok()])
    _test_hooks.http_post = poster
    original = board_hooks.env
    board_hooks.env = {
        "TASKBOARD_MCP_API_KEY": "test-key", "CORVIS_TENANT_ID": "tenant",
        "BOARD_WATCH_URL": DECLARED_TASKBOARD_URL,
    }.get
    try:
        assert research_propose.main([
            "--root", str(tmp_path), "--digest", digest, "--report-task", TASK,
        ]) == 0
    finally:
        board_hooks.env = original
    arguments = sent_arguments(poster.bodies[-1])
    assert arguments["agent"] == "fleet-research-proposer"
    assert arguments["taskId"] == TASK
    assert arguments["body"] == "RESEARCH PROPOSAL LOAD-REPORT\n" + dump_json_str({
        "spec_sha256": digest, "host": "colossus", "image": document["image"],
        "result": {
            "number": 3, "status": "admitted", "reason": "", "candidate": "c" * 64,
            "proposal_sha256": hashlib.sha256(dump_json_str(PROPOSAL).encode()).hexdigest(),
        },
    })


def test_invalid_report_task_runs_no_proposal() -> None:
    runner = FakeRun([])
    _test_hooks.run = runner
    with pytest.raises(ValueError, match="UUID"):
        research_propose.main(["--report-task", "invalid"])
    assert runner.calls == []
