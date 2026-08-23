"""Tests for the audit events and the logging hook behind them.

The default hook is exercised against the real platform logger, so the seam
is verified rather than assumed: a fake matching a protocol the production
implementation does not satisfy would otherwise pass everything.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pytest
from platform_core.json_utils import JSONValue

from hpc3.contracts.job import JobSpec
from hpc3.core import _test_hooks, audit
from tests.against_hpc3 import decode_job_spec
from tests.conftest import LoggedEvent, cluster


def _spec(**overrides: JSONValue) -> JobSpec:
    """Build a decoded job spec.

    Args:
        **overrides: Fields to replace.

    Returns:
        A validated spec.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "arm-b-42",
        "partition": "free-gpu",
        "gpu": "A100",
        "gpu_count": 2,
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "accept_billing": False,
        "env_path": "/pub/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B"},
        "command": "python train.py",
    }
    base.update(overrides)
    return decode_job_spec(base)


class TestJobSubmittedEvent:
    def test_it_records_every_field_a_reader_needs(self, logged: list[LoggedEvent]) -> None:
        audit.job_submitted(_spec(), host="hpc3", job_id="55519937", cluster=cluster())

        assert len(logged) == 1
        assert logged[0].event == audit.JOB_SUBMITTED
        assert logged[0].fields == {
            "job_id": "55519937",
            "job_name": "abl.arm-b-42",
            "project": "abl",
            "host": "hpc3",
            "cluster": "hpc3",
            "partition": "free-gpu",
            "gpu": "A100",
            "gpu_count": 2,
            "cpus": 8,
            "minutes": 30,
            "bills": False,
            "requeue": False,
            "checkpoint_steps": 0,
        }

    def test_the_recorded_name_is_the_one_the_cluster_shows(
        self, logged: list[LoggedEvent]
    ) -> None:
        """An audit trail naming something squeue never showed is not a trail."""
        audit.job_submitted(_spec(project="sirius"), host="hpc3", job_id="1", cluster=cluster())
        assert logged[0].fields["job_name"] == "sirius.arm-b-42"

    def test_a_billing_submission_is_marked_as_spending(self, logged: list[LoggedEvent]) -> None:
        """A billed submission is a spending decision; the record must say so."""
        spec = _spec(partition="free-gpu32", gpu="L40S", accept_billing=True)
        audit.job_submitted(spec, host="hpc3", job_id="1", cluster=cluster())
        assert logged[0].fields["bills"] is True

    def test_a_protected_run_records_its_protection(self, logged: list[LoggedEvent]) -> None:
        spec = _spec(minutes=600, requeue=True, checkpoint_steps=50)
        audit.job_submitted(spec, host="hpc3", job_id="1", cluster=cluster())
        assert logged[0].fields["requeue"] is True
        assert logged[0].fields["checkpoint_steps"] == 50


class TestSweepSubmittedEvent:
    def test_it_records_the_member_ids(self, logged: list[LoggedEvent]) -> None:
        audit.sweep_submitted(host="hpc3", project="abl", base_name="rung", job_ids=["1", "2", "3"])
        assert logged[0].event == audit.SWEEP_SUBMITTED
        assert logged[0].fields == {
            "host": "hpc3",
            "project": "abl",
            "base_name": "abl.rung",
            "members": 3,
            "job_ids": "1,2,3",
        }


class TestFilesStagedEvent:
    def test_it_records_the_destination_and_count(self, logged: list[LoggedEvent]) -> None:
        audit.files_staged(
            host="hpc3",
            destination="/pub/corpora",
            count=2,
            provenance="wiki_commit=176bb8c",
        )
        assert logged[0].event == audit.FILES_STAGED
        assert logged[0].fields == {
            "host": "hpc3",
            "destination": "/pub/corpora",
            "files": 2,
            "provenance": "wiki_commit=176bb8c",
        }


class TestDefaultLogHook:
    def test_the_production_hook_writes_through_the_platform_logger(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="hpc3"):
            _test_hooks.log_event("hpc3_test_event", {"a": "x", "n": 1, "flag": True})

        records = [r for r in caplog.records if r.message == "hpc3_test_event"]
        assert len(records) == 1
        assert records[0].levelno == logging.INFO

    def test_the_hook_resets_to_the_production_implementation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        captured: list[str] = []

        def _capture(event: str, fields: Mapping[str, str | int | bool]) -> None:
            captured.append(event)

        _test_hooks.log_event = _capture
        _test_hooks.log_event("held", {})
        assert captured == ["held"]

        _test_hooks.reset_hooks()
        with caplog.at_level(logging.INFO, logger="hpc3"):
            _test_hooks.log_event("after_reset", {})

        # The fake stopped receiving and the real logger started.
        assert captured == ["held"]
        assert [r.message for r in caplog.records] == ["after_reset"]
