"""The restart lane's pure half: argv composition, refusals, result prose.

The tick-level behaviour -- a claimed ``restart-session`` job reported
started, executed, and closed -- lives in ``test_agent_restart.py``; this
module holds what needs no queue: the exact session-audit invocation against
the published tree, the refusals that keep a lawless target out of an argv
element, and the closing detail's tail.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest

from fleet.core import _test_hooks, rebuild, restart
from fleet.core.published_tree import PublishedTree
from tests.conftest import FakeRun, ok

TARGET = "934d9975-0d65-4e68-83de-b74f8c4df0c4"
ROOT: Final = pathlib.Path("C:/Users/Test/PROJECTS/MCPs")
REGISTRY: Final = "C:/scratch/fleet-session-audit/abc/mcp-shared/src/source-registry"
TREE: Final = PublishedTree(
    commit="6e920e3e17f7ed9275422b703c33ca4325c3f45b",
    python_path="C:/scratch/sa/src;C:/scratch/shared/src",
    registry_dir=REGISTRY,
)

#: Every invocation opens the same way: the checkout's session-audit
#: environment, through poetry.
PREFIX: Final = ("poetry", "-C", str(ROOT / "packages" / "session-audit"), "run", "session-audit")


class TestRestartArgv:
    def test_composes_the_exact_session_audit_invocation(self) -> None:
        """The register the verb reads is the extracted one, last on the line."""
        assert restart.restart_argv(ROOT, REGISTRY, TARGET) == (
            *PREFIX,
            "rollover",
            "--apply",
            "--session",
            TARGET,
            "--registry-dir",
            REGISTRY,
        )

    def test_composes_the_exact_revive_invocation(self) -> None:
        """The second session verb (MCPs mig 525, board task 1fe89973): one
        table maps it to session-audit's revive mode, carrying the submitter
        into the brief the revived session is typed."""
        assert restart.revive_argv(ROOT, REGISTRY, TARGET, "fable-system-audit-0915") == (
            *PREFIX,
            "revive",
            "--session",
            TARGET,
            "--requested-by",
            "fable-system-audit-0915",
            "--registry-dir",
            REGISTRY,
        )
        assert restart.SESSION_COMMANDS == (
            "restart-session",
            "revive-session",
            "kill-session",
            "kill-session-hard",
            "compact-session",
        )

    def test_composes_the_exact_compact_invocation(self) -> None:
        """MCPs mig 564, board task 01f31e4a: the compact carries the
        submitter so session-audit prints who asked, and no kill flag."""
        assert restart.compact_argv(ROOT, REGISTRY, TARGET, "opus-super-fleet-0926") == (
            *PREFIX,
            "compact",
            "--session",
            TARGET,
            "--requested-by",
            "opus-super-fleet-0926",
            "--registry-dir",
            REGISTRY,
        )

    def test_composes_the_exact_kill_invocations_and_only_the_hard_verb_is_hard(self) -> None:
        """MCPs mig 526, board task 660964d9: two verbs, one mode, and the
        --hard flag is the queue's choice, never this runner's."""
        graceful = (*PREFIX, "kill", "--session", TARGET, "--requested-by", "opus-mcps-0917-e7b5")

        assert restart.kill_argv(ROOT, REGISTRY, TARGET, "opus-mcps-0917-e7b5", hard=False) == (
            *graceful,
            "--registry-dir",
            REGISTRY,
        )
        assert restart.kill_argv(ROOT, REGISTRY, TARGET, "opus-mcps-0917-e7b5", hard=True) == (
            *graceful,
            "--hard",
            "--registry-dir",
            REGISTRY,
        )

    def test_maps_every_session_verb_to_exactly_one_invocation_at_the_trees_commit(self) -> None:
        label = "opus-mcps-0917-e7b5"
        pinned = {"commit": TREE["commit"], "python_path": TREE["python_path"]}

        assert restart.session_invocation(ROOT, TREE, "restart-session", TARGET, label) == {
            "verb": "restart",
            "mode": "rollover",
            "argv": restart.restart_argv(ROOT, REGISTRY, TARGET),
            "types_requester": False,
            **pinned,
        }
        assert restart.session_invocation(ROOT, TREE, "revive-session", TARGET, label) == {
            "verb": "revive",
            "mode": "revive",
            "argv": restart.revive_argv(ROOT, REGISTRY, TARGET, label),
            "types_requester": True,
            **pinned,
        }
        assert restart.session_invocation(ROOT, TREE, "kill-session", TARGET, label) == {
            "verb": "kill",
            "mode": "kill",
            "argv": restart.kill_argv(ROOT, REGISTRY, TARGET, label, hard=False),
            "types_requester": True,
            **pinned,
        }
        hard = restart.session_invocation(ROOT, TREE, "kill-session-hard", TARGET, label)
        assert hard["argv"] == restart.kill_argv(ROOT, REGISTRY, TARGET, label, hard=True)
        assert restart.session_invocation(ROOT, TREE, "compact-session", TARGET, label) == {
            "verb": "compact",
            "mode": "compact",
            "argv": restart.compact_argv(ROOT, REGISTRY, TARGET, label),
            "types_requester": True,
            **pinned,
        }

    def test_a_command_that_is_not_a_session_verb_is_refused_by_code(self) -> None:
        refusal = r"^SESSION_COMMAND_UNKNOWN: 'check' is not a session verb$"
        with pytest.raises(ValueError, match=refusal):
            restart.session_invocation(ROOT, TREE, "check", TARGET, "opus-mcps-0917-e7b5")

    def test_a_submitter_that_is_not_a_board_label_is_refused_before_argv(self) -> None:
        refusal = restart.requester_refusal("Not A Label; rm -rf /")
        if refusal is None:
            raise AssertionError("expected a refusal for a lawless submitter")
        assert refusal.startswith(f"{restart.REQUESTER_INVALID_CODE}:")
        assert restart.requester_refusal("fable-system-audit-0915") is None

    def test_the_keystrokes_are_not_here(self) -> None:
        """A3 on the board task: the sequence lives in session_audit.rollover
        and nowhere else. The runner composes one invocation and no key."""
        source = pathlib.Path(restart.__file__).read_text(encoding="utf-8")
        assert "/exit" not in source.replace("``/exit``", "")
        assert "send-keys" not in source


class TestRefusals:
    def test_a_missing_checkout_is_refused_by_code(self, tmp_path: pathlib.Path) -> None:
        absent = tmp_path / "no-such-checkout"

        refusal = restart.refusal_for(absent, TARGET)

        if refusal is None:
            raise AssertionError("expected a refusal for an absent checkout")
        assert refusal.startswith(f"{restart.ROOT_MISSING_CODE}:")
        assert str(absent) in refusal

    def test_a_row_with_no_target_is_refused_by_code(self, tmp_path: pathlib.Path) -> None:
        """The queue's CHECK forbids this row; a runner that met one anyway
        must name the fact rather than invoke session-audit with nothing."""
        refusal = restart.refusal_for(tmp_path, None)

        if refusal is None:
            raise AssertionError("expected a refusal for a targetless row")
        assert refusal.startswith(f"{restart.TARGET_MISSING_CODE}:")

    def test_a_target_outside_the_uuid_grammar_is_refused_before_argv(
        self, tmp_path: pathlib.Path
    ) -> None:
        for lawless in ("mcps-99", TARGET.upper(), f"{TARGET} --apply"):
            refusal = restart.refusal_for(tmp_path, lawless)
            if refusal is None:
                raise AssertionError(f"expected a refusal for {lawless!r}")
            assert refusal.startswith(f"{restart.TARGET_INVALID_CODE}:")

    def test_a_real_checkout_and_a_lawful_target_pass(self, tmp_path: pathlib.Path) -> None:
        assert restart.refusal_for(tmp_path, TARGET) is None


class TestRunAndDescribe:
    def test_run_goes_through_the_command_seam_with_the_extraction_on_pythonpath(self) -> None:
        """The verb withholds the agent's own venv and sets PYTHONPATH to the
        published extraction, and nothing else about its environment."""
        runner = FakeRun([ok("RESTARTED")])
        _test_hooks.run = runner

        result = restart.run_session_job(
            restart.session_invocation(ROOT, TREE, "restart-session", TARGET, "fable-dm-0912")
        )

        assert result["returncode"] == 0
        assert runner.calls == [restart.restart_argv(ROOT, REGISTRY, TARGET)]
        assert runner.unset_env == [restart.SESSION_ENVIRONMENT_EXCLUDED] == [("VIRTUAL_ENV",)]
        assert runner.set_env == [(("PYTHONPATH", TREE["python_path"]),)]
        assert runner.timeouts == [restart.SESSION_JOB_TIMEOUT_SECONDS]

    def test_revive_goes_through_the_command_seam_and_names_its_mode_and_commit(self) -> None:
        runner = FakeRun([ok("REVIVE - REVIVED session x: now pid 5")])
        _test_hooks.run = runner
        invocation = restart.session_invocation(
            ROOT, TREE, "revive-session", TARGET, "fable-system-audit-0915"
        )

        result = restart.run_session_job(invocation)

        assert runner.calls == [
            restart.revive_argv(ROOT, REGISTRY, TARGET, "fable-system-audit-0915")
        ]
        assert restart.describe_result(result, invocation) == (
            f"session-audit revive at {TREE['commit']} exited 0: "
            "REVIVE - REVIVED session x: now pid 5"
        )

    def test_describe_carries_the_exit_code_and_the_combined_output(self) -> None:
        detail = restart.describe_result(
            _test_hooks.CommandResult(
                returncode=1,
                stdout=(
                    "ROLLOVER APPLIED - 1 restart(s) attempted: 0 restarted, 1 skipped, 0 failed\n"
                ),
                stderr="",
                timed_out=False,
            ),
            restart.session_invocation(ROOT, TREE, "restart-session", TARGET, "fable-dm-0912"),
        )

        assert detail == (
            f"session-audit rollover at {TREE['commit']} exited 1: ROLLOVER APPLIED - "
            "1 restart(s) attempted: 0 restarted, 1 skipped, 0 failed"
        )

    def test_describe_keeps_the_tail_where_the_outcome_lines_are(self) -> None:
        head = "plan line\n" * 400
        detail = restart.describe_result(
            _test_hooks.CommandResult(
                returncode=0,
                stdout=f"{head}RESTARTED mcps-99 now pid 41324",
                stderr="",
                timed_out=False,
            ),
            restart.session_invocation(ROOT, TREE, "restart-session", TARGET, "fable-dm-0912"),
        )

        assert detail.endswith("RESTARTED mcps-99 now pid 41324")
        # The tail width is the rebuild lane's, lifted rather than restated.
        opening = f"session-audit rollover at {TREE['commit']} exited 0: "
        assert len(detail) <= len(opening) + rebuild.DETAIL_TAIL_CHARS
