"""Scheduler absence, rejection and failure are never reported as success."""

from __future__ import annotations

import pathlib

import pytest

from hpc3.core.wrds_scheduler import account_state, job_argv, parse_job_id, queue_state, status
from tests.conftest import FakeRun
from tests.wrds_support import example

EMPTY = "<job_info><queue_info/><job_info/></job_info>"
HELD = (
    "<job_info><queue_info/><job_info><job_list><JB_job_number>123</JB_job_number>"
    "<state>hqw</state></job_list></job_info></job_info>"
)
DONE = "jobnumber 123\nfailed 0\nexit_status 0\n"


def test_resource_request_is_quoted_and_bounded(tmp_path: pathlib.Path) -> None:
    run = example(tmp_path)
    argv = job_argv(run, "/home/uci/awagner14/run with spaces", "one", "a" * 64)
    assert argv[argv.index("-pe") + 1 : argv.index("-pe") + 3] == ["onenode", "1"]
    assert argv[argv.index("-l") + 1] == "h_rt=30,m_mem_free=2G"
    assert argv[argv.index("-shell") + 1] == "n"
    assert "-V" not in argv
    assert argv[-2] == "-c"
    assert "sha256sum -c -" in argv[-1]
    assert "\n" not in argv[-1]
    assert "'" in argv[-1]
    assert "run.json' one" in argv[-1]


@pytest.mark.parametrize("text", ["", "bad", "123\n456", "12.1-2:1"])
def test_non_single_job_receipts(text: str) -> None:
    with pytest.raises(ValueError, match="WRDS_JOB_ID"):
        parse_job_id(text)


def test_queue_requires_exact_job_and_well_formed_state() -> None:
    assert parse_job_id("123\n") == "123"
    assert queue_state(EMPTY, "123") == "absent"
    assert queue_state(HELD, "123") == "hqw"
    assert queue_state(HELD, "456") == "absent"
    for xml in (
        "<unrelated/>",
        HELD.replace("hqw", ""),
        HELD.replace(
            "</job_info></job_info>",
            HELD.split("<job_list>")[1].replace("</job_info></job_info>", "")
            + "</job_info></job_info>",
        ),
    ):
        with pytest.raises((ValueError, SyntaxError)):
            queue_state(xml, "123")
    duplicate = HELD.replace(
        "</job_list>",
        "</job_list><job_list><JB_job_number>123</JB_job_number><state>r</state></job_list>",
    )
    with pytest.raises(ValueError, match="WRDS_QUEUE"):
        queue_state(duplicate, "123")


@pytest.mark.parametrize(
    "text", ["", "jobnumber 456\nfailed 0\nexit_status 0", DONE + "jobnumber 123\n"]
)
def test_invalid_accounting(text: str) -> None:
    with pytest.raises(ValueError, match="WRDS_ACCOUNTING"):
        account_state(text, "123")


def test_accounting_requires_both_scheduler_and_payload_success() -> None:
    assert account_state("=====\nqname all.q\n" + DONE, "123") == "completed"
    assert account_state(DONE.replace("failed 0", "failed 37 : lost host"), "123") == "failed"
    assert account_state(DONE.replace("exit_status 0", "exit_status 7"), "123") == "failed"


def test_status_active_and_completed(fake_run: FakeRun) -> None:
    fake_run.add("qstat", stdout=HELD, once=True)
    fake_run.add("qstat", stdout=EMPTY)
    fake_run.add("qacct", stdout=DONE)
    assert status("wrds", "123") == "hqw"
    assert status("wrds", "123") == "completed"


def test_accounting_delay_is_distinct_from_a_query_failure(fake_run: FakeRun) -> None:
    fake_run.add("qstat", stdout=EMPTY)
    fake_run.add("qacct", returncode=1, stderr="error: job id 123 not found", once=True)
    fake_run.add("qacct", returncode=1, stderr="database unavailable")
    assert status("wrds", "123") == "accounting-pending"
    with pytest.raises(ValueError, match="database unavailable"):
        status("wrds", "123")
