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
        assert restart.SESSION_COMMANDS == ("restart-session", "revive-session")

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

        result = restart.run_session_restart(tmp_path, session_target=TARGET)

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

        result = restart.run_session_revive(
            tmp_path, session_target=TARGET, requested_by="fable-system-audit-0915"
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
            )
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
            )
        )

        assert detail.endswith("RESTARTED mcps-99 now pid 41324")
        # The tail width is the rebuild lane's, lifted rather than restated.
        assert len(detail) <= len("session-audit rollover exited 0: ") + rebuild.DETAIL_TAIL_CHARS
