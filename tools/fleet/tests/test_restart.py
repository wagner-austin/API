"""The restart lane's pure half: argv composition, refusals, result prose.

The tick-level behaviour -- a claimed ``restart-session`` job reported
started, executed, and closed -- lives in ``test_agent_restart.py``; this
module holds what needs no queue: the exact session-audit invocation, the
refusals that keep a lawless target out of an argv element, and the closing
detail's tail.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pytest

from fleet.core import _test_hooks, rebuild, restart

TARGET = "934d9975-0d65-4e68-83de-b74f8c4df0c4"


class TestRestartArgv:
    def test_composes_the_exact_session_audit_invocation(self) -> None:
        root = pathlib.Path("C:/Users/Test/PROJECTS/MCPs")

        argv = restart.restart_argv(root, TARGET)

        assert argv == (
            "poetry",
            "-C",
            str(root / "packages" / "session-audit"),
            "run",
            "session-audit",
            "rollover",
            "--apply",
            "--session",
            TARGET,
        )

    def test_composes_the_exact_revive_invocation(self) -> None:
        """The second session verb (MCPs mig 525, board task 1fe89973): one
        table maps it to session-audit's revive mode, carrying the submitter
        into the brief the revived session is typed."""
        root = pathlib.Path("C:/Users/Test/PROJECTS/MCPs")

        argv = restart.revive_argv(root, TARGET, "fable-system-audit-0915")

        assert argv == (
            "poetry",
            "-C",
            str(root / "packages" / "session-audit"),
            "run",
            "session-audit",
            "revive",
            "--session",
            TARGET,
            "--requested-by",
            "fable-system-audit-0915",
        )
        assert restart.SESSION_COMMANDS == (
            "restart-session",
            "revive-session",
            "kill-session",
            "kill-session-hard",
        )

    def test_composes_the_exact_kill_invocations_and_only_the_hard_verb_is_hard(self) -> None:
        """MCPs mig 526, board task 660964d9: two verbs, one mode, and the
        --hard flag is the queue's choice, never this runner's."""
        root = pathlib.Path("C:/Users/Test/PROJECTS/MCPs")
        graceful = (
            "poetry",
            "-C",
            str(root / "packages" / "session-audit"),
            "run",
            "session-audit",
            "kill",
            "--session",
            TARGET,
            "--requested-by",
            "opus-mcps-0917-e7b5",
        )

        assert restart.kill_argv(root, TARGET, "opus-mcps-0917-e7b5", hard=False) == graceful
        assert restart.kill_argv(root, TARGET, "opus-mcps-0917-e7b5", hard=True) == (
            *graceful,
            "--hard",
        )

    def test_maps_every_session_verb_to_exactly_one_invocation(self) -> None:
        root = pathlib.Path("C:/Users/Test/PROJECTS/MCPs")
        label = "opus-mcps-0917-e7b5"

        assert restart.session_invocation(root, "restart-session", TARGET, label) == {
            "verb": "restart",
            "mode": "rollover",
            "argv": restart.restart_argv(root, TARGET),
            "types_requester": False,
        }
        assert restart.session_invocation(root, "revive-session", TARGET, label) == {
            "verb": "revive",
            "mode": "revive",
            "argv": restart.revive_argv(root, TARGET, label),
            "types_requester": True,
        }
        assert restart.session_invocation(root, "kill-session", TARGET, label) == {
            "verb": "kill",
            "mode": "kill",
            "argv": restart.kill_argv(root, TARGET, label, hard=False),
            "types_requester": True,
        }
        assert restart.session_invocation(root, "kill-session-hard", TARGET, label)["argv"] == (
            restart.kill_argv(root, TARGET, label, hard=True)
        )

    def test_a_command_that_is_not_a_session_verb_is_refused_by_code(self) -> None:
        root = pathlib.Path("C:/Users/Test/PROJECTS/MCPs")
        refusal = r"^SESSION_COMMAND_UNKNOWN: 'check' is not a session verb$"
        with pytest.raises(ValueError, match=refusal):
            restart.session_invocation(root, "check", TARGET, "opus-mcps-0917-e7b5")

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
    def test_run_goes_through_the_command_seam(self, tmp_path: pathlib.Path) -> None:
        calls: list[tuple[str, ...]] = []
        withheld: list[tuple[str, ...]] = []

        def fake_run(
            argv: Sequence[str],
            *,
            stdin_bytes: bytes | None = None,
            unset_env: Sequence[str] = (),
        ) -> _test_hooks.CommandResult:
            calls.append(tuple(argv))
            withheld.append(tuple(unset_env))
            return _test_hooks.CommandResult(returncode=0, stdout="RESTARTED", stderr="")

        _test_hooks.run = fake_run

        result = restart.run_session_job(
            restart.session_invocation(tmp_path, "restart-session", TARGET, "fable-dm-0912")
        )

        assert result["returncode"] == 0
        assert calls == [restart.restart_argv(tmp_path, TARGET)]
        assert withheld == [("VIRTUAL_ENV",)]

    def test_revive_goes_through_the_command_seam_and_names_its_mode(
        self, tmp_path: pathlib.Path
    ) -> None:
        calls: list[tuple[str, ...]] = []
        withheld: list[tuple[str, ...]] = []

        def fake_run(
            argv: Sequence[str],
            *,
            stdin_bytes: bytes | None = None,
            unset_env: Sequence[str] = (),
        ) -> _test_hooks.CommandResult:
            calls.append(tuple(argv))
            withheld.append(tuple(unset_env))
            return _test_hooks.CommandResult(
                returncode=0, stdout="REVIVE - REVIVED session x: now pid 5", stderr=""
            )

        _test_hooks.run = fake_run

        result = restart.run_session_job(
            restart.session_invocation(
                tmp_path, "revive-session", TARGET, "fable-system-audit-0915"
            )
        )

        assert calls == [restart.revive_argv(tmp_path, TARGET, "fable-system-audit-0915")]
        assert withheld == [restart.SESSION_ENVIRONMENT_EXCLUDED] == [("VIRTUAL_ENV",)]
        assert restart.describe_result(result, "revive") == (
            "session-audit revive exited 0: REVIVE - REVIVED session x: now pid 5"
        )

    def test_describe_carries_the_exit_code_and_the_combined_output(self) -> None:
        detail = restart.describe_result(
            _test_hooks.CommandResult(
                returncode=1,
                stdout=(
                    "ROLLOVER APPLIED - 1 restart(s) attempted: 0 restarted, 1 skipped, 0 failed\n"
                ),
                stderr="",
            ),
            "rollover",
        )

        assert detail == (
            "session-audit rollover exited 1: ROLLOVER APPLIED - 1 restart(s) attempted: "
            "0 restarted, 1 skipped, 0 failed"
        )

    def test_describe_keeps_the_tail_where_the_outcome_lines_are(self) -> None:
        head = "plan line\n" * 400
        detail = restart.describe_result(
            _test_hooks.CommandResult(
                returncode=0, stdout=f"{head}RESTARTED mcps-99 now pid 41324", stderr=""
            ),
            "rollover",
        )

        assert detail.endswith("RESTARTED mcps-99 now pid 41324")
        # The tail width is the rebuild lane's, lifted rather than restated.
        assert len(detail) <= len("session-audit rollover exited 0: ") + rebuild.DETAIL_TAIL_CHARS
