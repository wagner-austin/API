"""The claim half of a node runner's tick on the queue's node lane (MCPs
board task fd5cabfa), against a faked queue, a faked board and a faked fleet.

Everything but the two boundaries this machine cannot cross in a test is
real: the real workspace decode, the real capacity check, the real ledger and
lease files, the real staging and launch scripts. The queue speaks the
JSON-RPC-over-SSE shape the live endpoint speaks, and git and ssh answer
through the one command hook, so what is asserted is the exact sequence of
commands a claim issues and the exact requests it makes. The collect half,
the announce and the entry points are ``test_node_agent_collect.py``.
"""

from __future__ import annotations

import pathlib
import uuid

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str, narrow_json_to_str

from fleet.cli import _config, node_agent
from fleet.contracts.source import ProjectCompanion
from fleet.core import _test_hooks, dialect, records, staging
from tests._node_agent_fixtures import (
    COMPANION_DIRECTORY,
    COMPANION_REF,
    COMPANION_REMOTE,
    COMPANION_SHA,
    PROBED,
    REMOTE,
    _credentials_in_env,
    _sourced_config,
    claim_replies,
    node_argv,
    prebuilt_companion,
    prebuilt_export,
    sourced_document,
)
from tests._queue_fakes import DEFAULT_SHA, FakeQueue, queue_job
from tests.conftest import DEMO_PROJECT, FakeRun, failed, ok

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
    runner = FakeRun(claim_replies(staging.digest(payload), commit_present=commit_present))
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
    """Where the tick keeps the demo project's bare mirror.

    Args:
        config_path: The workspace document.

    Returns:
        The mirror's path as git is given it.
    """
    return str(config_path.parent / "runs" / "mirrors" / "libs-demo.git")


def _companion_config(config_path: pathlib.Path) -> pathlib.Path:
    """Rewrite the workspace so the demo project declares one companion.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    config_path.write_text(
        dump_json_str(
            sourced_document(
                (("npm", "ci"),),
                (
                    ProjectCompanion(
                        remote=COMPANION_REMOTE, ref=COMPANION_REF, directory=COMPANION_DIRECTORY
                    ),
                ),
            )
        ),
        encoding="utf-8",
    )
    return config_path


def _companion_replies(
    export_digest: str, companion_digest: str
) -> list[_test_hooks.CommandResult]:
    """Every command a claim tick runs for a project carrying one companion.

    Args:
        export_digest: What the node reports for the project's archive.
        companion_digest: What it reports for the companion's.

    Returns:
        One result per call, in order.
    """
    return [
        *PROBED,
        ok(""),  # the project's mirror: git init --bare
        ok(""),  # the project's mirror: git cat-file -e, the commit is held
        ok(""),  # the companion's mirror: git init --bare
        ok(""),  # the companion: git fetch the ref
        ok(f"{COMPANION_SHA}\n"),  # the companion: git rev-parse the tip
        ok(""),  # the companion: git archive
        ok(""),  # the project: git archive
        ok(""),  # stage: send mkdir script
        ok(""),  # stage: run mkdir
        ok(""),  # stage: send the base64 payload
        ok(""),  # stage: send reassemble script
        ok(export_digest),  # stage: run reassemble
        ok(""),  # stage: send extract script
        ok(""),  # stage: run extract
        ok(""),  # stage: send the git-init script
        ok(""),  # stage: run git init
        ok(""),  # companion: send reset script
        ok(""),  # companion: run reset
        ok(""),  # companion: send mkdir script
        ok(""),  # companion: run mkdir
        ok(""),  # companion: send the base64 payload
        ok(""),  # companion: send reassemble script
        ok(companion_digest),  # companion: run reassemble
        ok(""),  # companion: send extract script
        ok(""),  # companion: run extract
        ok(""),  # companion: send the commit script
        ok(""),  # companion: run the commit
        ok(""),  # launch: send the build script
        ok(""),  # launch: send the registration script
        ok("launched"),  # launch: run the registration script
    ]


class TestCompanions:
    """A project whose check reads a second repository (MCPs board task
    0515040d). slime lints its lifted code against the committed workspace
    beside it, and a node that was handed slime's commit alone stopped at that
    gate on every sha: job f25cb840 on sedona, exit 2 in under three minutes,
    'no git checkout at C:\\fleet\\stage\\MCPs'."""

    def test_the_companion_is_fetched_at_its_tip_before_the_lease_and_staged_beside_the_run(
        self, config_path: pathlib.Path
    ) -> None:
        """The whole path, through the real prepare, the real export and the
        real staging: the ref resolved to a commit, that commit archived, and
        the tree landed at ``<stage_root>/MCPs`` -- which is ``../MCPs`` from
        the export root, the one spelling that answers on a node and on a
        workstation alike."""
        sourced = _companion_config(config_path)
        export_payload = prebuilt_export(sourced)
        companion_payload = prebuilt_companion(sourced)
        runner = FakeRun(
            _companion_replies(staging.digest(export_payload), staging.digest(companion_payload))
        )
        _test_hooks.run = runner
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        assert node_agent.main(node_argv(sourced)) == 0

        mirror = str(config_path.parent / "runs" / "mirrors" / "companion-wagner-austin-MCPs.git")
        assert runner.calls[4] == ("git", "init", "--bare", "--quiet", mirror)
        assert runner.calls[5] == (
            "git",
            "-C",
            mirror,
            "fetch",
            "--quiet",
            "--no-tags",
            "--force",
            COMPANION_REMOTE,
            f"+{COMPANION_REF}:refs/fleet/companion",
        )
        assert runner.calls[6][3] == "rev-parse"
        assert runner.calls[7][3] == "archive"
        assert runner.calls[7][-1] == COMPANION_SHA
        sent = [body or b"" for body in runner.stdin]
        assert (
            dialect.companion_repository_script("C:/fleet/stage/MCPs", COMPANION_SHA).encode()
            in sent
        )

    def test_the_companion_lands_before_the_build_script_is_sent(
        self, config_path: pathlib.Path
    ) -> None:
        """The install steps are part of that script, so a project whose
        install or recipe reads its companion would otherwise find nothing
        there."""
        sourced = _companion_config(config_path)
        export_payload = prebuilt_export(sourced)
        companion_payload = prebuilt_companion(sourced)
        runner = FakeRun(
            _companion_replies(staging.digest(export_payload), staging.digest(companion_payload))
        )
        _test_hooks.run = runner
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        assert node_agent.main(node_argv(sourced)) == 0

        sent = [body or b"" for body in runner.stdin]
        committed = sent.index(
            dialect.companion_repository_script("C:/fleet/stage/MCPs", COMPANION_SHA).encode()
        )
        built = next(index for index, body in enumerate(sent) if b"make check" in body)
        assert committed < built

    def test_the_feed_names_the_commit_the_companion_was_measured_with(
        self, config_path: pathlib.Path
    ) -> None:
        """A companion is the tip of a ref, so two runs of one sha can
        legitimately measure different things; the line is where that shows."""
        sourced = _companion_config(config_path)
        export_payload = prebuilt_export(sourced)
        companion_payload = prebuilt_companion(sourced)
        _test_hooks.run = FakeRun(
            _companion_replies(staging.digest(export_payload), staging.digest(companion_payload))
        )
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        assert node_agent.main(node_argv(sourced)) == 0

        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced)})
        staged = [
            event["detail"] for event in records.read_feed(loaded.feed) if event["kind"] == "staged"
        ]
        assert len(staged) == 2
        assert staged[1] == (
            f"{len(companion_payload)} bytes, the MCPs companion at {COMPANION_SHA}, "
            "to C:/fleet/stage/MCPs"
        )


class TestIdentity:
    def test_the_label_and_session_are_derived_from_the_node(self) -> None:
        label, session = node_agent.node_identity("sedona")

        assert label == "fleet-node-sedona"
        assert session == str(uuid.uuid5(uuid.NAMESPACE_URL, "fleet-node-agent/sedona"))
        assert uuid.UUID(session).version == 5
        # Stable across calls, so every tick is one session on the ledger.
        assert node_agent.node_identity("sedona") == (label, session)


class TestClaiming:
    def test_a_claimed_check_is_exported_staged_launched_and_reported_started(
        self, sourced_config: pathlib.Path
    ) -> None:
        payload = prebuilt_export(sourced_config)
        runner = FakeRun(claim_replies(staging.digest(payload), commit_present=True))
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
        assert started["runId"] == "libs-demo-1757000000"
        # The node's probe (two ssh calls) BEFORE the claim, then the mirror,
        # the commit probe and the archive, then the fleet's own sequence:
        # no tar of any working tree anywhere, and no second probe.
        mirror = _mirror(sourced_config)
        assert runner.calls[0][0] == "ssh"
        assert runner.calls[1][0] == "ssh"
        assert runner.calls[2] == ("git", "init", "--bare", "--quiet", mirror)
        assert runner.calls[3] == (
            "git",
            "-C",
            mirror,
            "cat-file",
            "-e",
            f"{DEFAULT_SHA}^{{commit}}",
        )
        archive = runner.calls[4]
        assert archive[:6] == ("git", "-C", mirror, "archive", "--format=tar.gz", "-o")
        assert archive[7] == DEFAULT_SHA
        assert not any(call[0] == "tar" for call in runner.calls)

    def test_the_build_script_carries_the_install_steps_and_the_node_cache(
        self, sourced_config: pathlib.Path
    ) -> None:
        runner = _claim_and_start(sourced_config)

        # The build script is the third-from-last send; its body travelled
        # on stdin, and it is exactly the dialect's rendering for this
        # target with the registry's install steps and the node's cache
        # root, the workers being those the capacity check granted.
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        rows = records.read_ledger(loaded.ledger)
        assert len(rows) == 1
        expected = dialect.for_platform("windows").build_script(
            target="C:/fleet/stage/libs-demo-1757000000",
            path=DEMO_PROJECT,
            workers=rows[0]["workers"],
            install=(("npm", "ci"),),
            cache_root="C:/fleet/stage/cache",
        )
        assert runner.stdin[-3] == expected.encode("utf-8")
        assert "npm ci *>> 'C:/fleet/stage/libs-demo-1757000000/result.txt.log'" in expected
        assert "$env:npm_config_cache = 'C:/fleet/stage/cache/npm'" in expected
        assert (
            f"Set-Location -LiteralPath 'C:/fleet/stage/libs-demo-1757000000/{DEMO_PROJECT}'"
            in expected
        )

    def test_a_commit_the_mirror_lacks_is_fetched_from_the_declared_remote(
        self, sourced_config: pathlib.Path
    ) -> None:
        runner = _claim_and_start(sourced_config, commit_present=False)

        mirror = _mirror(sourced_config)
        assert runner.calls[4] == (
            "git",
            "-C",
            mirror,
            "fetch",
            "--quiet",
            "--no-tags",
            REMOTE,
            DEFAULT_SHA,
        )

    def test_the_ledger_row_names_the_submitter_and_the_feed_names_the_commit(
        self, sourced_config: pathlib.Path
    ) -> None:
        payload = prebuilt_export(sourced_config)
        _test_hooks.run = FakeRun(claim_replies(staging.digest(payload), commit_present=True))
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
        silent = "lavender did not answer; claiming nothing: "
        assert any(message.startswith(silent) for message in messages)
        assert "nothing in the node lane for lavender" in messages

    def test_a_node_with_room_for_nothing_claims_nothing(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """2.2 GB free against a 4.0 GB owner reservation, as lavender read
        at 09:58Z that day: nothing fits, so no job is taken to be refused."""
        _test_hooks.run = FakeRun([ok(""), ok("free_ram_gb=2.2\nfree_disk_gb=243.0\n")])
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

    def test_tags_that_disagree_with_the_registry_are_refused_before_any_git_command(
        self, sourced_config: pathlib.Path
    ) -> None:
        runner = FakeRun(PROBED)
        _test_hooks.run = runner

        detail = self._refused_detail(
            sourced_config, queue_job(status="claimed", requiredTags=["gpu"])
        )

        assert detail == (
            "PROJECT_TAGS_MISMATCH: the job requires [gpu] but fleet.json declares [] for "
            "libs/demo; resubmit with the registry's tags"
        )
        assert [call[0] for call in runner.calls] == ["ssh", "ssh"]

    def test_a_project_with_no_remote_is_refused_by_name(self, config_path: pathlib.Path) -> None:
        """The shared fixture's project declares ``source: null``."""
        runner = FakeRun(PROBED)
        _test_hooks.run = runner

        detail = self._refused_detail(config_path, queue_job(status="claimed"))

        assert detail.startswith("PROJECT_REMOTE_MISSING: project 'libs/demo' declares no source")
        assert [call[0] for call in runner.calls] == ["ssh", "ssh"]

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
                failed(128, "fatal: unable to access: could not resolve host"),
            ]
        )

        detail = self._refused_detail(sourced_config, queue_job(status="claimed"))

        assert detail.startswith("EXPORT_FAILED: git fetch")
        assert "could not resolve host" in detail

    def test_a_node_with_room_but_too_little_for_the_project_refuses_with_the_engines_code(
        self, sourced_config: pathlib.Path
    ) -> None:
        """6.0 GB free is 2.0 GB past the owner's 4.0 GB reservation: one
        1.1 GB worker, so the node passes the pre-claim gate (room for
        something) and the claimed project, whose minimum is two workers,
        is refused after the claim with the engine's own code."""
        _test_hooks.run = FakeRun(
            [ok(""), ok("free_ram_gb=6.0\nfree_disk_gb=860.0\n"), ok(""), ok("")]
        )

        detail = self._refused_detail(sourced_config, queue_job(status="claimed"))

        assert detail.startswith(
            f"{FleetErrorCode.NODE_MEMORY_EXHAUSTED.value}: lavender affords 1 worker(s) for a "
            "suite that declares a minimum of 2"
        )

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
