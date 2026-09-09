"""fleet-runners end to end: flags, exit codes, and both modes.

Every invocation goes through :func:`fleet.cli.runners.main` with a real
roster file in a temporary directory and the command seam faked at
``_test_hooks.run`` -- the same wiring production has, minus the network.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from fleet.cli import runners
from fleet.core import _test_hooks, runner_load
from tests.conftest import FakeRun, failed, ok


def _raw_host(name: str) -> dict[str, JSONValue]:
    """A valid raw host entry for a roster file.

    Args:
        name: The host's name, also used as its ssh alias.

    Returns:
        The raw mapping.
    """
    return {
        "name": name,
        "host": name,
        "wsl_distro": "Ubuntu",
        "keepalive_task": None,
        "wslconfig_min_memory_gb": None,
        "scratch_dir": "C:/fleet/stage",
        "gpu_required": False,
        "systemd_timers": [],
        "installs": [
            {
                "repo": "wagner-austin/API",
                "runner_name": f"{name}-wsl",
                "side": "wsl",
                "service": f"actions.runner.wagner-austin-API.{name}-wsl.service",
                "workdir": "/home/gharunner/actions-runner-1/_work",
                "labels": ["lavender-wsl"],
            }
        ],
        "assets": [],
    }


def _write_roster(tmp_path: pathlib.Path, hosts: list[dict[str, JSONValue]]) -> str:
    """Write a roster file.

    Args:
        tmp_path: The test's temporary directory.
        hosts: The raw host entries.

    Returns:
        The roster's path, as a string for the flag.
    """
    path = tmp_path / "runners.json"
    path.write_text(dump_json_str({"hosts": hosts}), encoding="utf-8")
    return str(path)


def _clean_transcript(name: str) -> str:
    """The transcript of a minimal host with every check passing.

    Args:
        name: The host's name.

    Returns:
        The OK lines.
    """
    return (
        f"CHECK service:wsl:actions.runner.wagner-austin-API.{name}-wsl.service OK\n"
        f"CHECK workdir:wagner-austin/API:wsl:{name}-wsl OK\n"
    )


class TestFlags:
    """Refusals before any work happens."""

    def test_a_missing_spec_flag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--spec is required"):
            runners.main([])

    def test_render_without_host_is_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        with pytest.raises(ValueError, match="--render requires --host"):
            runners.main(["--spec", spec_path, "--render", str(tmp_path / "out")])

    def test_an_unknown_host_is_a_typed_error(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", spec_path, "--host", "loki"])
        assert fault.value.code is FleetErrorCode.RUNNER_HOST_UNKNOWN
        assert "lavender" in fault.value.message

    def test_an_invalid_roster_is_a_typed_error(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "runners.json"
        path.write_text('{"hosts": [{}]}', encoding="utf-8")
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", str(path)])
        assert fault.value.code is FleetErrorCode.RUNNER_SPEC_UNREADABLE

    def test_a_missing_roster_raises_from_the_reader(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(OSError):
            runners.main(["--spec", str(tmp_path / "absent.json")])


class TestAudit:
    """The audit mode's exit codes and coverage."""

    def test_a_clean_fleet_exits_zero(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender"), _raw_host("loki")])
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(_clean_transcript("lavender")),
                ok(""),
                ok(_clean_transcript("loki")),
            ]
        )
        assert runners.main(["--spec", spec_path]) == 0

    def test_host_filters_to_one_machine(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender"), _raw_host("loki")])
        runner = FakeRun([ok(""), ok(_clean_transcript("loki"))])
        _test_hooks.run = runner
        assert runners.main(["--spec", spec_path, "--host", "loki"]) == 0
        assert len(runner.calls) == 2
        assert "loki" in runner.calls[0]
        assert not any("lavender" in call for call in runner.calls)

    def test_drift_exits_one(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        transcript = (
            "CHECK service:wsl:actions.runner.wagner-austin-API.lavender-wsl.service "
            "DRIFT it said: inactive\n"
            "CHECK workdir:wagner-austin/API:wsl:lavender-wsl OK\n"
        )
        _test_hooks.run = FakeRun([ok(""), ok(transcript)])
        assert runners.main(["--spec", spec_path]) == 1

    def test_an_unreachable_host_is_a_line_beside_the_others(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender"), _raw_host("loki")])
        _test_hooks.run = FakeRun(
            [failed(255, "No route to host"), ok(""), ok(_clean_transcript("loki"))]
        )
        assert runners.main(["--spec", spec_path]) == 1

    def test_an_unscorable_transcript_stops_the_audit(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        _test_hooks.run = FakeRun([ok(""), ok("garbage\n")])
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", spec_path])
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE


class TestRender:
    """The render mode's artifacts."""

    def test_render_writes_both_scripts_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        out_dir = tmp_path / "out"
        assert (
            runners.main(["--spec", spec_path, "--host", "lavender", "--render", str(out_dir)]) == 0
        )
        windows = (out_dir / "provision.ps1").read_text(encoding="utf-8")
        linux = (out_dir / "provision.sh").read_text(encoding="utf-8")
        assert "nothing declared" in windows
        assert "RUNNER_TOKEN_API" in linux
        assert "ci-clean" in linux

    def test_render_touches_no_network(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        runner = FakeRun([])
        _test_hooks.run = runner
        runners.main(["--spec", spec_path, "--host", "lavender", "--render", str(tmp_path / "o")])
        assert runner.calls == []

    def test_a_manual_asset_is_printed_as_a_run_order_step(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        host = _raw_host("lavender")
        host["assets"] = [
            {
                "path": "/opt/corvis/rw-game/game-lib.jar",
                "sha256": None,
                "writable": False,
                "reason": "the provenance jar",
                "manual": True,
                "provision_command": None,
            }
        ]
        spec_path = _write_roster(tmp_path, [host])
        with caplog.at_level("INFO"):
            code = runners.main(
                ["--spec", spec_path, "--host", "lavender", "--render", str(tmp_path / "o")]
            )
        assert code == 0
        assert any(
            "PLACE BY HAND: /opt/corvis/rw-game/game-lib.jar" in record.message
            for record in caplog.records
        )


class TestSampleAndReport:
    """The load-sampling modes."""

    _FOREST = "\n".join(
        [
            "1 0 /sbin/init",
            "100 1 /home/gharunner/actions-runner-1/bin/Runner.Worker spawnclient",
            "101 100 python -m pytest",
            "102 101 pt_data_worker",
        ]
    )

    def test_sample_appends_one_line_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        record = tmp_path / "load.jsonl"
        _test_hooks.run = FakeRun([ok(self._FOREST)])
        assert (
            runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)]) == 0
        )
        lines = [line for line in record.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 1
        decoded = runner_load.decode_load_sample(load_json_str(lines[0]))
        assert decoded["total"] == 2
        assert decoded["host"] == "lavender"

    def test_report_reads_the_record_back_and_exits_zero(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        record = tmp_path / "load.jsonl"
        _test_hooks.run = FakeRun([ok(self._FOREST), ok(self._FOREST)])
        runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)])
        runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)])
        with caplog.at_level("INFO"):
            code = runners.main(["--spec", spec_path, "--report", str(record), "--cores", "16"])
        assert code == 0
        messages = [record_.getMessage() for record_ in caplog.records]
        assert "samples: 2" in messages
        assert any("at or under 16 cores: 100.0% of samples" in m for m in messages)

    def test_report_filtered_to_an_unsampled_host_says_it_measured_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender"), _raw_host("loki")])
        record = tmp_path / "load.jsonl"
        _test_hooks.run = FakeRun([ok(self._FOREST)])
        runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)])
        with pytest.raises(ValueError, match="measured nothing"):
            runners.main(
                ["--spec", spec_path, "--host", "loki", "--report", str(record), "--cores", "16"]
            )

    def test_report_without_cores_is_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        with pytest.raises(ValueError, match="--report requires --cores"):
            runners.main(["--spec", spec_path, "--report", str(tmp_path / "x.jsonl")])

    @pytest.mark.parametrize("cores", ["0", "-3", "many"])
    def test_a_non_positive_cores_value_is_refused(
        self, tmp_path: pathlib.Path, cores: str
    ) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        with pytest.raises(ValueError, match="positive integer"):
            runners.main(
                ["--spec", spec_path, "--report", str(tmp_path / "x.jsonl"), "--cores", cores]
            )

    def test_two_mode_flags_are_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        with pytest.raises(ValueError, match="different acts"):
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

    def test_an_unreachable_host_raises_rather_than_recording(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        record = tmp_path / "load.jsonl"
        _test_hooks.run = FakeRun([failed(255, "No route to host")])
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)])
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert not record.exists()


class TestOnboardMode:
    """The --onboard mode: flags, roster rewrite, exit codes."""

    def test_onboard_requires_host(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        with pytest.raises(ValueError, match="--onboard requires --host"):
            runners.main(["--spec", spec_path, "--onboard", "wagner-austin/x"])

    def test_sides_without_onboard_is_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        with pytest.raises(ValueError, match="--sides only means something"):
            runners.main(["--spec", spec_path, "--sides", "wsl"])

    def test_python_without_onboard_is_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        with pytest.raises(ValueError, match="--python only means something"):
            runners.main(["--spec", spec_path, "--python", "3.11.9"])

    def test_onboard_rewrites_the_roster_and_exits_on_the_audit(
        self, tmp_path: pathlib.Path
    ) -> None:
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        # The post-onboard audit covers the pre-existing install AND the new
        # wsl one, in roster order.
        transcript = (
            _clean_transcript("lavender")
            + "CHECK service:wsl:actions.runner.wagner-austin-x.lavender-wsl.service OK\n"
            + "CHECK workdir:wagner-austin/x:wsl:lavender-wsl OK\n"
        )
        _test_hooks.run = FakeRun(
            [
                ok("TOK123\n"),  # gh mint
                ok(""),  # send wsl payload
                ok(""),  # send wsl driver
                ok("done"),  # run wsl driver
                ok(""),  # audit send
                ok(transcript),  # audit run
            ]
        )
        code = runners.main(
            [
                "--spec",
                spec_path,
                "--host",
                "lavender",
                "--onboard",
                "wagner-austin/x",
                "--sides",
                "wsl",
            ]
        )
        assert code == 0
        rewritten = pathlib.Path(spec_path).read_text(encoding="utf-8")
        decoded = runners.load_runner_spec(spec_path)
        assert '"wagner-austin/x"' in rewritten
        repos = [i["repo"] for i in decoded["hosts"][0]["installs"]]
        assert repos == ["wagner-austin/API", "wagner-austin/x"]

    def test_onboard_with_a_drifting_audit_exits_one_and_still_records(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The runners exist and the roster records them either way; the
        exit code carries the audit's verdict, not the provisioning's."""
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        transcript = (
            _clean_transcript("lavender")
            + "CHECK service:wsl:actions.runner.wagner-austin-x.lavender-wsl.service "
            "DRIFT it said: inactive\n" + "CHECK workdir:wagner-austin/x:wsl:lavender-wsl OK\n"
        )
        _test_hooks.run = FakeRun(
            [ok("TOK123\n"), ok(""), ok(""), ok("done"), ok(""), ok(transcript)]
        )
        code = runners.main(
            [
                "--spec",
                spec_path,
                "--host",
                "lavender",
                "--onboard",
                "wagner-austin/x",
                "--sides",
                "wsl",
            ]
        )
        assert code == 1
        decoded = runners.load_runner_spec(spec_path)
        assert [i["repo"] for i in decoded["hosts"][0]["installs"]] == [
            "wagner-austin/API",
            "wagner-austin/x",
        ]


class TestEntrypoint:
    """The console script boundary."""

    def test_entrypoint_exits_with_mains_code(self, tmp_path: pathlib.Path) -> None:
        """A render invocation, so no faked network is needed and the one
        claim asserted is the one the boundary exists for: the SystemExit
        carries main's return value."""
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        saved = sys.argv
        sys.argv = [
            "fleet-runners",
            "--spec",
            spec_path,
            "--host",
            "lavender",
            "--render",
            str(tmp_path / "out"),
        ]
        try:
            with pytest.raises(SystemExit) as caught:
                runners.entrypoint()
        finally:
            sys.argv = saved
        assert caught.value.code == 0

    def test_running_as_a_module_actually_runs(self, tmp_path: pathlib.Path) -> None:
        """The half that silently goes missing without an `if __name__` block."""
        spec_path = _write_roster(tmp_path, [_raw_host("lavender")])
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.runners", None)
        sys.argv = [
            "x",
            "--spec",
            spec_path,
            "--host",
            "lavender",
            "--render",
            str(tmp_path / "out"),
        ]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.runners", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.runners"] = saved_module
        assert raised.value.code == 0
