"""fleet-runners end to end: its flags, the audit, the render and the entry points.

Every invocation goes through :func:`fleet.cli.runners.main` with a real
roster file in a temporary directory and the command seam faked at
``_test_hooks.run`` -- the same wiring production has, minus the network.
The acts that change a host or a record are tests/test_runner_cli_acts.py's.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.cli import runners
from fleet.core import _test_hooks, runner_render
from fleet.core.runner_registrations import REGISTRATIONS_REASON
from tests._runner_fixtures import all_declared, no_dropped_jobs
from tests._runner_roster_files import (
    clean_transcript,
    manual_asset,
    raw_host,
    roster_host,
    write_roster,
)
from tests.conftest import FakeRun, failed, ok


class TestFlags:
    """Refusals before any work happens."""

    def test_a_missing_spec_flag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--spec is required"):
            runners.main([])

    def test_render_without_host_is_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="--render requires --host"):
            runners.main(["--spec", spec_path, "--render", str(tmp_path / "out")])

    def test_an_unknown_host_is_a_typed_error_naming_the_declared_ones(
        self, tmp_path: pathlib.Path
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender"), raw_host("sedona")])
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", spec_path, "--host", "loki"])
        assert fault.value.code is FleetErrorCode.RUNNER_HOST_UNKNOWN
        assert fault.value.message == (
            "the roster declares no host named 'loki'; it declares: lavender, sedona"
        )

    def test_an_invalid_roster_is_a_typed_error(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "runners.json"
        path.write_text('{"hosts": [{}]}', encoding="utf-8")
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", str(path)])
        assert fault.value.code is FleetErrorCode.RUNNER_SPEC_UNREADABLE

    def test_a_missing_roster_raises_from_the_reader(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(OSError):
            runners.main(["--spec", str(tmp_path / "absent.json")])

    def test_two_mode_flags_are_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="--render and --sample are different acts"):
            runners.main(
                [
                    "--spec",
                    spec_path,
                    "--host",
                    "lavender",
                    "--sample",
                    str(tmp_path / "l.jsonl"),
                    "--render",
                    str(tmp_path / "out"),
                ]
            )


class TestAudit:
    """The audit mode's exit codes and coverage."""

    def test_a_clean_fleet_exits_zero(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender"), raw_host("sedona")])
        lavender, sedona = roster_host(spec_path, "lavender"), roster_host(spec_path, "sedona")
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(clean_transcript(lavender)),
                *no_dropped_jobs(lavender),
                ok(""),
                ok(clean_transcript(sedona)),
                *no_dropped_jobs(sedona),
                *all_declared([lavender, sedona]),
            ]
        )
        with caplog.at_level("INFO"):
            assert runners.main(["--spec", spec_path]) == 0
        messages = [record.getMessage() for record in caplog.records]
        assert [message for message in messages if " OK " not in message] == []
        assert messages.index("lavender OK memory:31gb") < messages.index(
            "sedona OK dropped-jobs:sedona"
        )
        assert messages[-1] == "roster OK registrations:wagner-austin/API"

    def test_host_filters_to_one_machine(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender"), raw_host("sedona")])
        sedona = roster_host(spec_path, "sedona")
        runner = FakeRun(
            [
                ok(""),
                ok(clean_transcript(sedona)),
                *no_dropped_jobs(sedona),
                *all_declared([sedona]),
            ]
        )
        _test_hooks.run = runner
        assert runners.main(["--spec", spec_path, "--host", "sedona"]) == 0
        # The audit's send and run, the dropped-job row's two reads, then the
        # registrations row's one.
        assert len(runner.calls) == 5
        assert "sedona" in runner.calls[0]
        assert not any("lavender" in call for call in runner.calls)

    def test_a_registration_no_host_declares_drifts_the_roster_row(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """lavender-wsl's registrations outlived the WSL guest on 2026-10-20."""
        spec_path = write_roster(tmp_path, [raw_host("lavender"), raw_host("sedona")])
        sedona = roster_host(spec_path, "sedona")
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(clean_transcript(sedona)),
                *no_dropped_jobs(sedona),
                ok("1 sedona online false\n21 lavender-wsl offline false\n"),
            ]
        )
        with caplog.at_level("INFO"):
            assert runners.main(["--spec", spec_path, "--host", "sedona"]) == 1
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-2] == (
            "roster DRIFT registrations:wagner-austin/API -- examined 2 registration(s) of "
            "wagner-austin/API against the 2 runner name(s) the roster declares for it; 1 no "
            "roster host declares: lavender-wsl (id 21, offline, idle); fleet-runners --prune "
            f"yes removes the offline idle ones ({REGISTRATIONS_REASON})"
        )
        assert messages[-1] == "1 fault(s); the fleet does not match its roster"

    def test_drift_exits_one_with_the_roster_s_reason_beside_it(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        lavender = roster_host(spec_path, "lavender")
        unit = "actions.runner.wagner-austin-API.lavender.service"
        transcript = clean_transcript(lavender).replace(
            f"service:{unit} OK", f"service:{unit} DRIFT systemctl is-active said: inactive"
        )
        _test_hooks.run = FakeRun(
            [ok(""), ok(transcript), *no_dropped_jobs(lavender), *all_declared([lavender])]
        )
        with caplog.at_level("INFO"):
            assert runners.main(["--spec", spec_path]) == 1
        messages = [record.getMessage() for record in caplog.records]
        assert (
            f"lavender DRIFT service:{unit} -- systemctl is-active said: inactive "
            "(runs the wagner-austin/API runner lavender)"
        ) in messages
        assert messages[-1] == "1 fault(s); the fleet does not match its roster"

    def test_an_unreachable_host_is_a_line_beside_the_others(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender"), raw_host("sedona")])
        sedona = roster_host(spec_path, "sedona")
        _test_hooks.run = FakeRun(
            [
                failed(255, "No route to host"),
                ok(""),
                ok(clean_transcript(sedona)),
                *no_dropped_jobs(sedona),
                *all_declared([sedona]),
            ]
        )
        with caplog.at_level("INFO"):
            assert runners.main(["--spec", spec_path]) == 1
        messages = [record.getMessage() for record in caplog.records]
        assert any(m.startswith("lavender UNREACHABLE ") for m in messages)
        assert "sedona OK memory:31gb" in messages

    def test_an_unscorable_transcript_stops_the_audit(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        _test_hooks.run = FakeRun([ok(""), ok("garbage\n")])
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", spec_path])
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE


class TestRender:
    """The render mode's artifacts and run order."""

    def test_render_writes_both_scripts_and_prints_the_run_order(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        host = raw_host("lavender")
        host["assets"] = [manual_asset("the provenance jar")]
        spec_path = write_roster(tmp_path, [host])
        out_dir = tmp_path / "out"
        runner = FakeRun([])
        _test_hooks.run = runner
        with caplog.at_level("INFO"):
            code = runners.main(
                ["--spec", spec_path, "--host", "lavender", "--render", str(out_dir)]
            )
        assert code == 0
        assert runner.calls == []
        rendered = runner_render.render_provision(roster_host(spec_path, "lavender"))
        assert (out_dir / "base.sh").read_text(encoding="utf-8") == rendered["base_script"]
        assert (out_dir / "provision.sh").read_text(encoding="utf-8") == (
            rendered["provision_script"]
        )
        assert [record.getMessage() for record in caplog.records] == [
            f"wrote {out_dir / 'base.sh'}",
            f"wrote {out_dir / 'provision.sh'}",
            "RUN ORDER for lavender:",
            f"  1. base.sh on the host as root, then reboot if it printed "
            f"{runner_render.REBOOT_MARKER}",
            "  2. provision.sh on the host as root, with each RUNNER_TOKEN_* exported",
            f"  {rendered['manual_steps'][0]}",
            "  3. fleet-runners --spec <roster> --host lavender",
        ]


class TestEntrypoint:
    """The console script boundary."""

    def _render_argv(self, tmp_path: pathlib.Path) -> list[str]:
        """A render invocation, which needs no faked network.

        Args:
            tmp_path: The case's directory, holding the roster and the output.

        Returns:
            The arguments after the program name.
        """
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        return ["--spec", spec_path, "--host", "lavender", "--render", str(tmp_path / "out")]

    def test_entrypoint_exits_with_mains_code(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["fleet-runners", *self._render_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as caught:
                runners.entrypoint()
        finally:
            sys.argv = saved
        assert caught.value.code == 0

    def test_help_prints_the_usage_line_and_exits_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Measured 2026-10-20 on ziggurat: ``fleet-runners --help`` ended in
        an uncaught HelpRequestedError traceback."""
        saved = sys.argv
        sys.argv = ["fleet-runners", "--help"]
        try:
            with pytest.raises(SystemExit) as caught:
                runners.entrypoint()
        finally:
            sys.argv = saved
        assert caught.value.code == 0
        assert capsys.readouterr().out == (
            "usage: --spec <value> --host <value> --render <value> --sample <value> "
            "--report <value> --cores <value> --onboard <value> --rebuild <value> "
            "--prune <value>\n"
        )

    def test_a_refusal_is_one_line_on_stderr_and_exit_two(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        saved = sys.argv
        sys.argv = ["fleet-runners", "--host", "lavender"]
        try:
            with pytest.raises(SystemExit) as caught:
                runners.entrypoint()
        finally:
            sys.argv = saved
        assert caught.value.code == 2
        assert capsys.readouterr().err == (
            "usage: --spec is required: the roster's path is passed, never searched for\n"
        )

    def test_running_as_a_module_actually_runs(self, tmp_path: pathlib.Path) -> None:
        """The half that silently goes missing without an `if __name__` block."""
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.runners", None)
        sys.argv = ["x", *self._render_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.runners", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.runners"] = saved_module
        assert raised.value.code == 0
        assert (tmp_path / "out" / "provision.sh").is_file()
