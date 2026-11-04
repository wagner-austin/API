"""fleet-runners' acts end to end: sampling and its report, onboarding, registering, draining.

Every invocation goes through :func:`fleet.cli.runners.main` with a real
roster file in a temporary directory and the command seam faked at
``_test_hooks.run``, as tests/test_runner_cli.py's are.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import load_json_str

from fleet.cli import runners
from fleet.contracts.runners import HostRunnerSpec
from fleet.core import _test_hooks, runner_load, runner_onboard
from tests._runner_fixtures import register_answers
from tests._runner_roster_files import (
    clean_transcript,
    manual_asset,
    raw_host,
    roster_host,
    write_roster,
)
from tests.conftest import FakeClock, FakeRun, FakeSleep, failed, ok


class TestSampleAndReport:
    """The load-sampling modes."""

    _FOREST = "\n".join(
        [
            "1 0 /sbin/init",
            "100 1 /home/gharunner/actions-runner-api-1/bin/Runner.Worker spawnclient",
            "101 100 python -m pytest",
            "102 101 pt_data_worker",
        ]
    )

    def test_sample_appends_one_line_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
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
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        record = tmp_path / "load.jsonl"
        _test_hooks.run = FakeRun([ok(self._FOREST), ok(self._FOREST)])
        runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)])
        runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)])
        caplog.clear()
        with caplog.at_level("INFO"):
            code = runners.main(["--spec", spec_path, "--report", str(record), "--cores", "16"])
        assert code == 0
        assert [record_.getMessage() for record_ in caplog.records] == [
            "samples: 2",
            "total descendants: min=2 max=2",
            "over 16 cores: 0.0% of samples (longest consecutive streak: 0)",
            "at or under 16 cores: 100.0% of samples",
        ]

    def test_report_filtered_to_an_unsampled_host_says_it_measured_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender"), raw_host("sedona")])
        record = tmp_path / "load.jsonl"
        _test_hooks.run = FakeRun([ok(self._FOREST)])
        runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)])
        with pytest.raises(ValueError, match="measured nothing"):
            runners.main(
                ["--spec", spec_path, "--host", "sedona", "--report", str(record), "--cores", "16"]
            )

    def test_report_without_cores_is_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="--report requires --cores"):
            runners.main(["--spec", spec_path, "--report", str(tmp_path / "x.jsonl")])

    @pytest.mark.parametrize("cores", ["0", "-3", "many"])
    def test_a_non_positive_cores_value_is_refused(
        self, tmp_path: pathlib.Path, cores: str
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="positive integer"):
            runners.main(
                ["--spec", spec_path, "--report", str(tmp_path / "x.jsonl"), "--cores", cores]
            )

    def test_an_unreachable_host_raises_rather_than_recording(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        record = tmp_path / "load.jsonl"
        _test_hooks.run = FakeRun([failed(255, "No route to host")])
        with pytest.raises(AppError) as fault:
            runners.main(["--spec", spec_path, "--host", "lavender", "--sample", str(record)])
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert not record.exists()


def _onboarded(spec_path: str) -> HostRunnerSpec:
    """lavender as onboarding wagner-austin/x leaves it.

    Args:
        spec_path: The roster before onboarding.

    Returns:
        Its host with x's install after API's.
    """
    host = roster_host(spec_path, "lavender")
    host["installs"].append(runner_onboard.plan_onboard(host, "wagner-austin/x")["install"])
    return host


def _register_argv(spec_path: str) -> list[str]:
    """A confirmed registration of lavender's runners.

    Args:
        spec_path: The roster.

    Returns:
        The arguments after the program name.
    """
    return ["--spec", spec_path, "--host", "lavender", "--register", "yes"]


class TestOnboardMode:
    """The --onboard mode: its host, the roster rewrite, and the steps it prints."""

    def test_onboard_requires_host(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="--onboard requires --host"):
            runners.main(["--spec", spec_path, "--onboard", "wagner-austin/x"])

    def test_onboard_rewrites_the_roster_touching_no_host_and_prints_what_follows(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        grown = _onboarded(spec_path)
        runner = FakeRun([])
        _test_hooks.run = runner
        argv = ["--spec", spec_path, "--host", "lavender", "--onboard", "wagner-austin/x"]
        with caplog.at_level("INFO"):
            assert runners.main(argv) == 0
        assert runner.calls == []
        assert roster_host(spec_path, "lavender") == grown
        assert pathlib.Path(spec_path).read_text(encoding="utf-8").endswith("}\n")
        assert [record.getMessage() for record in caplog.records] == [
            "onboarded wagner-austin/x runner lavender (labels: lavender)",
            f"roster rewritten: {spec_path}",
            "NEXT, in order, for lavender:",
            "  1. make render-documents, then commit and push the roster and its render",
            "  2. in MCPs, fleet-reach render, then commit and push fleet-mcp/nixos/roster.json",
            "  3. fleet-runners --spec <roster> --host lavender --drain yes, switch lavender to "
            "the new generation, then the same with --resume yes",
            "  4. fleet-runners --spec <roster> --host lavender --register yes",
        ]


class TestRegisterMode:
    """The --register mode: its confirmation, its host, and its exit codes."""

    def test_register_requires_host(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="--register requires --host"):
            runners.main(["--spec", spec_path, "--register", "yes"])

    def test_register_takes_only_the_literal_yes(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="takes the literal value yes"):
            runners.main(["--spec", spec_path, "--host", "lavender", "--register", "true"])

    def test_a_clean_registration_exits_zero(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        lavender = roster_host(spec_path, "lavender")
        answers = register_answers(clean_transcript(lavender), lavender)
        runner = FakeRun(answers)
        _test_hooks.run = runner
        with caplog.at_level("INFO"):
            code = runners.main(_register_argv(spec_path))
        assert code == 0
        assert len(runner.calls) == len(answers)
        assert [record.getMessage() for record in caplog.records][:2] == [
            "lavender register: tokens: minted for 1 repositories and written to "
            "/var/lib/fleet-runner-tokens",
            "lavender register: registered: 1 runner(s), "
            "actions.runner.wagner-austin-API.lavender.service",
        ]
        assert caplog.records[-1].getMessage() == (
            "registration verified: every check green on lavender"
        )

    def test_a_manual_asset_drifts_until_placed(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        host = raw_host("lavender")
        host["assets"] = [manual_asset("the licensed engine")]
        spec_path = write_roster(tmp_path, [host])
        lavender = roster_host(spec_path, "lavender")
        jar = "/opt/corvis/rw-game/game-lib.jar"
        transcript = clean_transcript(lavender).replace(
            f"asset:{jar} OK", f"asset:{jar} DRIFT test -e {jar}:  (exit 1)"
        )
        _test_hooks.run = FakeRun(register_answers(transcript, lavender))
        with caplog.at_level("INFO"):
            code = runners.main(_register_argv(spec_path))
        assert code == 1
        messages = [record.getMessage() for record in caplog.records]
        assert f"lavender DRIFT asset:{jar} -- test -e {jar}:  (exit 1) (the licensed engine)" in (
            messages
        )
        assert messages[-1] == "1 fault(s) after registering; the drift lines name the repairs"


def _drain_argv(spec_path: str, flag: str) -> list[str]:
    """A confirmed drain or resume of lavender's runners.

    Args:
        spec_path: The roster.
        flag: ``--drain`` or ``--resume``.

    Returns:
        The arguments after the program name.
    """
    return ["--spec", spec_path, "--host", "lavender", flag, "yes"]


class TestDrainAndResumeModes:
    """The --drain and --resume modes: confirmation, host, and what each prints (f19d66a5)."""

    @pytest.mark.parametrize("flag", ["--drain", "--resume"])
    def test_each_requires_host(self, tmp_path: pathlib.Path, flag: str) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match=f"{flag} requires --host"):
            runners.main(["--spec", spec_path, flag, "yes"])

    @pytest.mark.parametrize(
        ("flag", "act"),
        [
            ("--drain", "it takes runners out of GitHub's pool"),
            ("--resume", "it starts runners and puts them in the pool"),
        ],
    )
    def test_each_takes_only_the_literal_yes(
        self, tmp_path: pathlib.Path, flag: str, act: str
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError) as refused:
            runners.main(["--spec", spec_path, "--host", "lavender", flag, "true"])
        assert str(refused.value) == (
            f"{flag} takes the literal value yes, got 'true': {act}, and a value is its "
            "confirmation"
        )

    def test_drain_and_resume_are_different_acts(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        argv = [*_drain_argv(spec_path, "--drain"), "--resume", "yes"]
        with pytest.raises(ValueError, match="are different acts"):
            runners.main(argv)

    def test_a_drain_prints_each_step_and_the_switch_that_follows(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        clock = FakeClock(1_000)
        _test_hooks.now = clock
        _test_hooks.sleep = FakeSleep(clock)
        idle = "1 lavender online false\n"
        answers = [idle, "lavender\n", "", idle, idle, idle, "", "inactive\n"]
        _test_hooks.run = FakeRun([ok(text) for text in answers])
        with caplog.at_level("INFO"):
            assert runners.main(_drain_argv(spec_path, "--drain")) == 0
        assert [record.getMessage() for record in caplog.records] == [
            "lavender drain: labels: removed from 1 runner(s); GitHub assigns them nothing new",
            "lavender drain: stopped: actions.runner.wagner-austin-API.lavender.service, idle "
            "for 30s with no label",
            "drained: every runner of lavender is stopped and out of the pool; switch it, then "
            "fleet-runners --resume yes --host lavender",
        ]

    def test_a_resume_prints_each_step(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        clock = FakeClock(1_000)
        _test_hooks.now = clock
        _test_hooks.sleep = FakeSleep(clock)
        online = "1 lavender online false\n"
        answers = [online, "", "active\n", "lavender\n", online]
        _test_hooks.run = FakeRun([ok(text) for text in answers])
        with caplog.at_level("INFO"):
            assert runners.main(_drain_argv(spec_path, "--resume")) == 0
        assert [record.getMessage() for record in caplog.records] == [
            "lavender resume: started: none, every unit was active",
            "lavender resume: labels: the roster's restored on 1 runner(s)",
            "lavender resume: online: all 1 runner(s), as GitHub reads them",
            "resumed: every runner of lavender is online and in the pool",
        ]


class TestPruneMode:
    """The --prune mode: its confirmation, its scope, and what it removes and keeps."""

    def test_prune_takes_only_the_literal_yes(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError) as refused:
            runners.main(["--spec", spec_path, "--prune", "true"])
        assert str(refused.value) == (
            "--prune takes the literal value yes, got 'true': it deletes registrations from "
            "GitHub, and a value is its confirmation"
        )

    def test_prune_takes_no_host(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="--prune takes no --host"):
            runners.main(["--spec", spec_path, "--host", "lavender", "--prune", "yes"])

    def test_the_offline_undeclared_are_removed_and_the_run_exits_zero(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        runner = FakeRun([ok("1 lavender online true\n21 lavender-wsl offline false\n"), ok("")])
        _test_hooks.run = runner
        with caplog.at_level("INFO"):
            assert runners.main(["--spec", spec_path, "--prune", "yes"]) == 0
        assert tuple(runner.calls[1]) == (
            "gh",
            "api",
            "-X",
            "DELETE",
            "repos/wagner-austin/API/actions/runners/21",
        )
        assert [record.getMessage() for record in caplog.records] == [
            "roster PRUNED wagner-austin/API lavender-wsl (id 21, offline, idle)",
            "prune: examined 2 registration(s); removed 1, kept 0",
        ]

    def test_an_online_undeclared_registration_is_kept_and_the_run_exits_one(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        runner = FakeRun([ok("1 lavender online false\n9 loki online false\n")])
        _test_hooks.run = runner
        with caplog.at_level("INFO"):
            assert runners.main(["--spec", spec_path, "--prune", "yes"]) == 1
        assert len(runner.calls) == 1
        assert [record.getMessage() for record in caplog.records] == [
            "roster KEPT wagner-austin/API loki (id 9, online, idle) -- a live machine no "
            "roster host declares: declare it, or stop its runner and prune again",
            "prune: examined 2 registration(s); removed 0, kept 1",
        ]
