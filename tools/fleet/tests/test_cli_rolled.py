"""The launcher each scheduled tick runs (board task 465689f5).

The command line is checked by its own refusals; the launch is checked for
real. The real case commits a stand-in ``fleet`` package to a repository,
rolls it, and lets the launcher extract it and run ``python -m
fleet.cli.agent`` with the default command hook: the stand-in answers only
if the extraction on ``PYTHONPATH`` shadowed this checkout's editable
install of the real package, which is the whole claim a roll rests on.
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

import pytest

from fleet.cli import rolled as rolled_cli
from fleet.core import _test_hooks, rolled
from tests._git_fixtures import committed_repository
from tests.conftest import FakeRun, FakeTempRoot, failed, ok

#: A stand-in agent: it names itself, echoes its argv, writes to both
#: streams and exits with a status no real agent uses.
STAND_IN_AGENT = (
    "import sys\n"
    "print('rolled stand-in', *sys.argv[1:])\n"
    "print('stand-in stderr', file=sys.stderr)\n"
    "sys.exit(5)\n"
)

#: A rolled commit whose fleet package is the stand-in.
STAND_IN_FILES: dict[str, str] = {
    "tools/fleet/src/fleet/__init__.py": "",
    "tools/fleet/src/fleet/cli/__init__.py": "",
    "tools/fleet/src/fleet/cli/agent.py": STAND_IN_AGENT,
    "tools/fleet/src/fleet/cli/node_agent.py": STAND_IN_AGENT,
    "libs/platform_core/src/platform_core/__init__.py": "",
    "libs/monorepo_guards/src/monorepo_guards/__init__.py": "",
    "tools/board-watch/src/board_watch/__init__.py": "",
    "tools/fleet/fleet.json": "{}\n",
}


class TestCommandLine:
    def test_the_separator_splits_the_launchers_flags_from_the_agents(self) -> None:
        parsed, passthrough = rolled_cli.split_command_line(
            ["--repo-root", "C:/api", "--agent", "fleet-agent", "--", "--node", "sedona"]
        )
        assert parsed == {"--repo-root": "C:/api", "--agent": "fleet-agent"}
        assert passthrough == ["--node", "sedona"]

    def test_a_command_line_without_the_separator_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            rolled_cli.split_command_line(["--repo-root", "C:/api", "--agent", "fleet-agent"])
        assert str(refused.value) == (
            "FLEET_ROLL_USAGE: -- must separate ['--repo-root', '--agent'] from the agent's own "
            "command line"
        )

    @pytest.mark.parametrize("flag", ["--config", "--records-dir"])
    def test_an_agent_command_line_naming_the_registry_or_records_is_refused(
        self, flag: str
    ) -> None:
        with pytest.raises(ValueError) as refused:
            rolled_cli.split_command_line(["--agent", "fleet-agent", "--", flag, "x"])
        assert str(refused.value) == (
            f"FLEET_ROLL_USAGE: the agent's command line may not carry {flag}; the registry "
            "is the rolled commit's own and the records are the checkout's"
        )

    def test_each_agent_names_its_module_and_nothing_else_is_accepted(self) -> None:
        assert rolled_cli.require_module("fleet-agent") == "fleet.cli.agent"
        assert rolled_cli.require_module("fleet-node-agent") == "fleet.cli.node_agent"
        with pytest.raises(ValueError) as refused:
            rolled_cli.require_module("fleet-run")
        assert str(refused.value) == (
            "FLEET_ROLL_USAGE: --agent is one of ['fleet-agent', 'fleet-node-agent']"
        )


class TestLaunch:
    def test_no_roll_runs_nothing_and_exits_refused(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        runner = FakeRun([failed(128, "fatal: Needed a single revision")])
        _test_hooks.run = runner

        with caplog.at_level("INFO"):
            status = rolled_cli.main(
                ["--repo-root", str(tmp_path), "--agent", "fleet-node-agent", "--", "--node", "x"]
            )

        assert status == rolled_cli.REFUSED_EXIT == 2
        assert len(runner.calls) == 1
        [message] = [record.getMessage() for record in caplog.records]
        assert message.startswith("FLEET_ROLL_REF_UNRESOLVED: refs/fleet/rolled in ")

    def test_the_agent_runs_from_the_extraction_with_its_own_arguments(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        commit = "0" * 40
        scratch = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch)
        destination = scratch / "fleet-rolled" / commit
        for required in rolled.REQUIRED_FILES:
            path = destination / pathlib.PurePosixPath(required)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        runner = FakeRun([ok(f"{commit}\n"), ok(""), ok(""), ok("")])
        _test_hooks.run = runner

        with caplog.at_level("INFO"):
            status = rolled_cli.main(
                ["--repo-root", str(tmp_path), "--agent", "fleet-agent", "--", "--session", "s"]
            )

        assert status == 0
        assert runner.calls[-1] == (
            sys.executable,
            "-m",
            "fleet.cli.agent",
            "--config",
            str(destination / "tools" / "fleet" / "fleet.json"),
            "--records-dir",
            str(tmp_path / "tools" / "fleet"),
            "--session",
            "s",
        )
        assert runner.timeouts[-1] == rolled_cli.AGENT_WALL_SECONDS == 2340
        [(name, python_path)] = runner.set_env[-1]
        assert name == "PYTHONPATH"
        assert python_path.split(os.pathsep)[0] == str(destination / "tools" / "fleet" / "src")
        # An agent that printed nothing is relayed as nothing.
        assert [record.getMessage() for record in caplog.records] == [
            f"fleet-roll: fleet.cli.agent runs from refs/fleet/rolled at {commit}"
        ]

    def test_the_rolled_package_shadows_the_checkouts_for_real(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        api = tmp_path / "api"
        commit = committed_repository(api, STAND_IN_FILES, rolled.ROLLED_REF)
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        config = tmp_path / "scratch" / "fleet-rolled" / commit / "tools" / "fleet" / "fleet.json"

        with caplog.at_level("INFO"):
            status = rolled_cli.main(
                ["--repo-root", str(api), "--agent", "fleet-node-agent", "--", "--node", "sedona"]
            )

        assert status == 5
        assert [record.getMessage() for record in caplog.records] == [
            f"fleet-roll: fleet.cli.node_agent runs from refs/fleet/rolled at {commit}",
            f"rolled stand-in --config {config} --records-dir {api / 'tools' / 'fleet'} "
            "--node sedona",
            "stand-in stderr",
        ]


class TestInvocationForms:
    def test_the_entrypoint_exits_with_mains_status(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        _test_hooks.run = FakeRun([failed(128, "no ref")])
        saved = sys.argv
        sys.argv = ["x", "--repo-root", str(tmp_path), "--agent", "fleet-agent", "--"]
        try:
            with pytest.raises(SystemExit) as raised:
                rolled_cli.entrypoint()
        finally:
            sys.argv = saved
        assert raised.value.code == 2

    def test_running_as_a_module_actually_runs(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        _test_hooks.run = FakeRun([failed(128, "no ref")])
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.rolled", None)
        sys.argv = ["x", "--repo-root", str(tmp_path), "--agent", "fleet-agent", "--"]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.rolled", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.rolled"] = saved_module
        assert raised.value.code == 2
