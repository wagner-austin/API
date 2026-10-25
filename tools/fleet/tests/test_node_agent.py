"""The claim half of a node runner's tick on the queue's node lane (MCPs
board task fd5cabfa), against a faked queue, a faked board and a faked fleet.

Everything but the two boundaries this machine cannot cross in a test is
real: the real workspace decode, the real capacity check, the real ledger and
lease files, the real staging and launch scripts. The queue speaks the
JSON-RPC-over-SSE shape the live endpoint speaks, and git and ssh answer
through the one command hook, so what is asserted is the exact sequence of
commands a claim issues and the exact requests it makes. The collect half,
the announce and the entry points are ``test_node_agent_collect.py``; a
project carrying a companion is ``test_node_agent_companions.py``.
"""

from __future__ import annotations

import pathlib
import uuid

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str, narrow_json_to_str

from fleet.cli import _config, node_agent, node_prepare
from fleet.contracts.dispatch import decode_job
from fleet.contracts.node import NodePlatform
from fleet.core import _test_hooks, dialect, mirror_fetch, records, staging
from tests._node_agent_fixtures import (
    NOTHING_LAUNCHED,
    NPM_CI,
    PROBED,
    REMOTE,
    SERVED_HOLDING_NOTHING,
    _credentials_in_env,
    _sourced_config,
    claim_replies,
    node_argv,
    prebuilt_export,
)
from tests._queue_fakes import DEFAULT_SHA, FakeQueue, queue_job
from tests._toolchain_fixtures import LAVENDER_2026_09_23
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID, FakeRun, failed, ok

__all__ = ["_credentials_in_env", "_sourced_config"]


def _claim_and_start(config_path: pathlib.Path, *, commit_present: bool = True) -> FakeRun:
    """Run one tick that claims the default job and launches it.

    Args:
        config_path: The workspace document.
        commit_present: Whether the mirror already holds the commit.

    Returns:
        The command runner, with every call recorded.
    """
    payload = prebuilt_export(config_path)
    runner = FakeRun(
        claim_replies(staging.digest(payload), registry=config_path, commit_present=commit_present)
    )
    _test_hooks.run = runner
    _test_hooks.http_post = FakeQueue(
        [
            dump_json_str({"jobs": []}),
            dump_json_str({"claimed": queue_job(status="claimed")}),
            dump_json_str({"job": queue_job(status="running", node="lavender")}),
        ]
    )
    assert node_agent.main(node_argv(config_path)) == 0
    return runner


def _mirror(config_path: pathlib.Path) -> str:
    """Where the tick keeps the bare mirror of the demo project's remote.

    Args:
        config_path: The workspace document.

    Returns:
        The mirror's path as git is given it.
    """
    return str(config_path.parent / "runs" / "mirrors" / "wagner-austin-API.git")


class TestIdentity:
    def test_the_label_and_session_are_derived_from_the_node(self) -> None:
        label, session = node_agent.node_identity("sedona", elevated=False)

        assert label == "fleet-node-sedona"
        assert session == str(uuid.uuid5(uuid.NAMESPACE_URL, "fleet-node-agent/sedona"))
        assert uuid.UUID(session).version == 5
        # Stable across calls, so every tick is one session on the ledger.
        assert node_agent.node_identity("sedona", elevated=False) == (label, session)


class TestClaiming:
    def test_a_claimed_check_is_exported_staged_launched_and_reported_started(
        self, sourced_config: pathlib.Path
    ) -> None:
        payload = prebuilt_export(sourced_config)
        runner = FakeRun(
            claim_replies(staging.digest(payload), registry=sourced_config, commit_present=True)
        )
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report"]
        claim = endpoint.arguments[1]
        assert claim["lane"] == "node"
        assert claim["tags"] == ["windows"]
        assert claim["node"] == "lavender"
        assert claim["agent"] == "fleet-node-lavender"
        started = endpoint.arguments[2]
        assert started["action"] == "start"
        assert started["node"] == "lavender"
        assert started["runId"] == DEMO_RUN_ID
        # The node's two probes, capacity then toolchain (two ssh calls each),
        # BEFORE the claim, then the mirror, the commit probe and the archive,
        # then the fleet's own sequence: no tar of any working tree anywhere,
        # and no second probe.
        mirror = _mirror(sourced_config)
        probes = len(PROBED)
        assert [call[0] for call in runner.calls[:probes]] == ["ssh"] * probes
        assert runner.calls[probes] == ("git", "init", "--bare", "--quiet", mirror)
        assert runner.calls[probes + 1] == (
            "git",
            "-C",
            mirror,
            "cat-file",
            "-e",
            f"{DEFAULT_SHA}^{{commit}}",
        )
        archive = runner.calls[probes + 2]
        assert archive[:6] == ("git", "-C", mirror, "archive", "--format=tar.gz", "-o")
        assert archive[7] == DEFAULT_SHA
        assert not any(call[0] == "tar" for call in runner.calls)

    def test_the_build_script_carries_the_install_steps_and_the_node_cache(
        self, sourced_config: pathlib.Path
    ) -> None:
        runner = _claim_and_start(sourced_config)

        # The build script is the third-from-last send before the re-probe
        # that follows every launch (MCPs board task 48842bfd); its body travelled
        # on stdin, and it is exactly the dialect's rendering for this
        # target with the registry's install steps and the node's cache
        # root, the workers being those the capacity check granted.
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        rows = records.read_ledger(loaded.ledger)
        assert len(rows) == 1
        expected = dialect.for_platform(NodePlatform.WINDOWS).build_script(
            target=f"C:/fleet/stage/{DEMO_RUN_ID}",
            path=DEMO_PROJECT,
            workers=rows[0]["workers"],
            install=(NPM_CI,),
            cache_root="C:/fleet/stage/cache",
            isolated_docker=False,
            elevated=False,
            agent="opus-dispatch-0905",
            loopback_unit=None,
        )
        assert runner.stdin[-3 - len(PROBED)] == expected.encode("utf-8")
        # The claimed job's submitter, exported so a hold the suite takes is
        # attributed to that session (MCPs board task 6c4516af A4).
        assert "$env:BOARD_AGENT_LABEL = 'opus-dispatch-0905'" in expected
        assert "[string[]]$Install = @('npm ci')," in expected
        assert "[string]$CacheRoot = 'C:/fleet/stage/cache'," in expected
        assert f"[string]$Recipe = 'C:/fleet/stage/{DEMO_RUN_ID}/{DEMO_PROJECT}'," in expected

    def test_a_commit_the_mirror_lacks_is_fetched_from_the_declared_remote(
        self, sourced_config: pathlib.Path
    ) -> None:
        runner = _claim_and_start(sourced_config, commit_present=False)

        mirror = _mirror(sourced_config)
        assert runner.calls[len(PROBED) + 2] == (
            "git",
            "-C",
            mirror,
            "fetch",
            "--quiet",
            "--no-tags",
            "--force",
            REMOTE,
            f"+{DEFAULT_SHA}:refs/fleet/fetched",
        )

    def test_the_ledger_row_names_the_submitter_and_the_feed_names_the_commit(
        self, sourced_config: pathlib.Path
    ) -> None:
        payload = prebuilt_export(sourced_config)
        _test_hooks.run = FakeRun(
            claim_replies(staging.digest(payload), registry=sourced_config, commit_present=True)
        )
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str(
                    {
                        "claimed": queue_job(
                            status="claimed",
                            submittedBy="opus-weight-injection-0902",
                            sessionId="acc774c0-3bc3-4cce-9dda-c7a12fb99519",
                        )
                    }
                ),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        node_agent.main(node_argv(sourced_config))

        ledger = (sourced_config.parent / "runs" / "ledger.jsonl").read_text(encoding="utf-8")
        assert "opus-weight-injection-0902" in ledger
        assert "acc774c0-3bc3-4cce-9dda-c7a12fb99519" in ledger
        feed = (sourced_config.parent / "runs" / "feed.jsonl").read_text(encoding="utf-8")
        assert f"git archive of {DEFAULT_SHA}" in feed


class TestClaimingNothing:
    """The gate before the claim: a node that cannot run anything takes
    nothing off the lane (the 2026-09-21T10:00:02Z incident in
    ``capacity.room_for_any``'s docstring)."""

    def test_a_node_that_does_not_answer_claims_nothing(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _test_hooks.run = FakeRun([failed(255, "ssh: connect to host lavender: timed out")])
        endpoint = FakeQueue([dump_json_str({"jobs": []})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list"]
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-3].startswith("lavender did not answer; claiming nothing: ")
        assert messages[-2:] == [NOTHING_LAUNCHED, SERVED_HOLDING_NOTHING]

    def test_a_node_with_room_for_nothing_claims_nothing(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """2.2 GB free against a 4.0 GB owner reservation, as lavender read
        at 09:58Z that day: nothing fits, so no job is taken to be refused.
        The toolchain is asked first, since the tags it finds decide which
        projects the room is sized for."""
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok("free_ram_gb=2.2\nfree_disk_gb=243.0\nreserved_cores=0\n"),
                ok(""),
                ok(LAVENDER_2026_09_23),
            ]
        )
        endpoint = FakeQueue([dump_json_str({"jobs": []})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list"]
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            message.startswith(
                "lavender has room for nothing; claiming nothing: NODE_OWNER_RESERVED: "
                "lavender has 2.2 GB free"
            )
            for message in messages
        )


class TestRefusals:
    def _refused_detail(self, config_path: pathlib.Path, job: JSONObject) -> str:
        """Run a tick that claims ``job`` and read the refusal it reports.

        Args:
            config_path: The workspace document.
            job: The wire row the queue hands back.

        Returns:
            The close report's detail.
        """
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": job}),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint
        assert node_agent.main(node_argv(config_path)) == 0
        closed = endpoint.arguments[2]
        assert closed["action"] == "close"
        assert closed["status"] == "refused"
        assert "exitCode" not in closed
        return narrow_json_to_str(closed["detail"])

    def test_an_unknown_project_is_refused_rather_than_crashing_the_loop(
        self, sourced_config: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun(PROBED)

        detail = self._refused_detail(
            sourced_config, queue_job(status="claimed", project="libs/absent")
        )

        assert FleetErrorCode.WORKSPACE_PROJECT_UNKNOWN in detail

    def test_a_tag_the_registry_does_not_declare_is_refused_before_any_git_command(
        self, sourced_config: pathlib.Path
    ) -> None:
        runner = FakeRun(PROBED)
        _test_hooks.run = runner

        detail = self._refused_detail(
            sourced_config, queue_job(status="claimed", requiredTags=["gpu"])
        )

        assert detail == (
            "PROJECT_TAGS_MISMATCH: the job requires [gpu], which "
            "registry/projects/libs/demo.json does not declare (it declares []); resubmit "
            "with the registry's tags"
        )
        assert [call[0] for call in runner.calls] == ["ssh"] * len(PROBED)

    def test_a_job_naming_fewer_tags_than_its_project_passes_the_gate(self) -> None:
        """What it needs is the declaration, which the claim's fits already
        assessed against this runner (MCPs board task 939ec5c7, job 7c16305c)."""
        row = queue_job(status="claimed", requiredTags=["windows"])
        job = decode_job(row, answer=dump_json_str(row))

        assert node_prepare.tags_refusal(job, ("elevated", "windows")) is None

    def test_a_project_with_no_remote_is_refused_by_name(self, config_path: pathlib.Path) -> None:
        """The shared fixture's project declares ``source: null``."""
        runner = FakeRun(PROBED)
        _test_hooks.run = runner

        detail = self._refused_detail(config_path, queue_job(status="claimed"))

        assert detail.startswith("PROJECT_REMOTE_MISSING: project 'libs/demo' declares no source")
        assert [call[0] for call in runner.calls] == ["ssh"] * len(PROBED)

    def test_a_sha_the_remote_lacks_is_refused_with_nothing_leased(
        self, sourced_config: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun(
            [
                *PROBED,
                ok(""),  # git init --bare
                failed(128, "missing"),  # cat-file: absent
                failed(128, f"fatal: couldn't find remote ref {DEFAULT_SHA}"),  # fetch
            ]
        )

        detail = self._refused_detail(sourced_config, queue_job(status="claimed"))

        assert detail.startswith(f"SHA_NOT_ON_REMOTE: {REMOTE} does not serve {DEFAULT_SHA}")
        assert "one git push away" in detail
        assert not (sourced_config.parent / "runs" / "leases.json").exists()

    def test_a_fetch_that_failed_for_another_reason_carries_gits_words(
        self, sourced_config: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun(
            [
                *PROBED,
                ok(""),
                failed(128, "missing"),
                *[failed(128, "fatal: unable to access: could not resolve host")]
                * mirror_fetch.FETCH_ATTEMPTS,
            ]
        )

        detail = self._refused_detail(sourced_config, queue_job(status="claimed"))

        assert detail.startswith("EXPORT_FAILED: git fetch")
        assert "attempt 3: fatal: unable to access: could not resolve host" in detail

    def test_a_node_lane_job_without_a_sha_is_a_contract_fault_not_a_refusal(
        self, sourced_config: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun(PROBED)
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed", sha=None)}),
            ]
        )

        with pytest.raises(AppError) as raised:
            node_agent.main(node_argv(sourced_config))

        assert raised.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED
        assert "carries no sha" in raised.value.message
