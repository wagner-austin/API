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
import shutil
import subprocess
import sys

import pytest

from fleet.cli import rolled as rolled_cli
from fleet.core import _test_hooks, rolled, scratch
from tests._git_fixtures import rolled_at_origin
from tests._rolled_fetch_script import fetch_replies
from tests.conftest import DEMO_NOW, FakeClock, FakeRun, FakeTempRoot, failed, ok

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
    "tools/fleet/registry/settings.json": "{}\n",
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
        assert rolled_cli.require_module("fleet-research") == "fleet.cli.research"
        with pytest.raises(ValueError) as refused:
            rolled_cli.require_module("fleet-run")
        assert str(refused.value) == (
            "FLEET_ROLL_USAGE: --agent is one of ['fleet-agent', 'fleet-node-agent', 'fleet-research']"
        )


class TestLaunch:
    def test_no_roll_runs_nothing_and_exits_refused(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        runner = FakeRun(
            [*fetch_replies(tmp_path / "git"), failed(128, "fatal: Needed a single revision")]
        )
        _test_hooks.run = runner

        with caplog.at_level("INFO"):
            status = rolled_cli.main(
                ["--repo-root", str(tmp_path), "--agent", "fleet-node-agent", "--", "--node", "x"]
            )

        assert status == rolled_cli.REFUSED_EXIT == 2
        assert len(runner.calls) == 3
        [message] = [record.getMessage() for record in caplog.records]
        assert message.startswith("FLEET_ROLL_REF_UNRESOLVED: refs/fleet/rolled in ")

    def test_the_agent_runs_from_the_extraction_with_its_own_arguments(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        commit = "0" * 40
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        _test_hooks.now = FakeClock(DEMO_NOW)
        destination = scratch_root / "fleet-rolled" / commit
        for required in rolled.REQUIRED_FILES:
            path = destination / pathlib.PurePosixPath(required)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        left = scratch_root / "fleet-archives" / "dead-run.tgz"
        left.parent.mkdir()
        left.write_bytes(b"left by a failed stage")
        then = DEMO_NOW - scratch.STALE_SECONDS - 1
        os.utime(left, (then, then))
        runner = FakeRun([*fetch_replies(tmp_path / "git"), ok(f"{commit}\n"), ok("")])
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
            str(destination / "tools" / "fleet" / "registry"),
            "--records-dir",
            str(tmp_path / "tools" / "fleet"),
            "--session",
            "s",
        )
        assert runner.timeouts[-1] == rolled_cli.AGENT_WALL_SECONDS == 2340
        [(name, python_path)] = runner.set_env[-1]
        assert name == "PYTHONPATH"
        assert python_path.split(os.pathsep)[0] == str(destination / "tools" / "fleet" / "src")
        # An agent that printed nothing is relayed as nothing; what the turn
        # reaped is named.
        assert [record.getMessage() for record in caplog.records] == [
            f"fleet-roll: fleet.cli.agent runs from refs/fleet/rolled at {commit}",
            "fleet-roll: removed what was left behind: fleet-archives/dead-run.tgz",
        ]
        # The commit's tree outlives the agent: the next tick runs from it
        # (MCPs board task fb505198).
        assert destination.is_dir()
        assert not left.exists()

    def test_the_rolled_package_shadows_the_checkouts_for_real(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        api = tmp_path / "api"
        commit = rolled_at_origin(api, STAND_IN_FILES)
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        launch = tmp_path / "scratch" / "fleet-rolled" / commit
        config = launch / "tools" / "fleet" / "registry"

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
        assert launch.is_dir()

    def test_ticks_launched_together_each_reach_their_agent_from_one_tree(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The 10:24Z collision, re-run: the hub starts its ticks in one second.

        Each launch is a real process with its own id, as each tick is, and
        the scratch root is the real one. Extracting into one directory per
        commit with no turn, two of five ticks were refused
        FLEET_ROLL_EXTRACT_FAILED on a file another tick's tar was writing;
        in the extraction turn one launch extracts and the rest reuse its
        tree (MCPs board task fb505198).
        """
        api = tmp_path / "api"
        commit = rolled_at_origin(api, STAND_IN_FILES)
        tree = _test_hooks._default_temp_root() / "fleet-rolled" / commit
        config = tree / "tools" / "fleet" / "registry"
        command = (
            sys.executable,
            "-m",
            "fleet.cli.rolled",
            "--repo-root",
            str(api),
            "--agent",
            "fleet-node-agent",
            "--",
            "--node",
            "sedona",
        )
        try:
            launches = [
                subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for _ in range(6)
            ]
            outputs = [launch.communicate(timeout=120)[0] for launch in launches]
            outcomes = list(zip([launch.returncode for launch in launches], outputs, strict=True))
            assert [status for status, _ in outcomes] == [5] * 6, outcomes
            assert all(f"rolled stand-in --config {config} " in output for _, output in outcomes), (
                outcomes
            )
            assert sorted(path.name for path in tree.parent.iterdir() if commit in path.name) == [
                commit
            ]
        finally:
            # The real scratch root outlives the test; its tree is this
            # test's alone, named for a commit made in its tmp_path.
            shutil.rmtree(tree)


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
