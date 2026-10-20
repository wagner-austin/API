"""fleet-runners' acts end to end: sampling and its report, onboarding, rebuilding.

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
from tests._runner_fixtures import no_dropped_jobs, quiet_rebuild_answers
from tests._runner_roster_files import (
    clean_transcript,
    manual_asset,
    raw_host,
    roster_host,
    write_roster,
)
from tests.conftest import FakeRun, failed, ok


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


class TestOnboardMode:
    """The --onboard mode: flags, roster rewrite, exit codes."""

    def test_onboard_requires_host(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="--onboard requires --host"):
            runners.main(["--spec", spec_path, "--onboard", "wagner-austin/x"])

    @pytest.mark.parametrize(("service", "code"), [("OK", 0), ("DRIFT it said: inactive", 1)])
    def test_onboard_rewrites_the_roster_and_exits_on_the_audit(
        self, tmp_path: pathlib.Path, service: str, code: int
    ) -> None:
        """The runner exists and the roster records it either way; the exit
        code carries the audit's verdict, not the provisioning's."""
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        grown = _onboarded(spec_path)
        unit = "actions.runner.wagner-austin-x.lavender.service"
        transcript = clean_transcript(grown).replace(
            f"service:{unit} OK", f"service:{unit} {service}"
        )
        _test_hooks.run = FakeRun(
            [
                ok("TOK123\n"),  # gh mint
                ok(""),  # send the provision
                ok("done\n"),  # run it
                ok(""),  # audit send
                ok(transcript),  # audit run
                *no_dropped_jobs(grown),
            ]
        )
        argv = ["--spec", spec_path, "--host", "lavender", "--onboard", "wagner-austin/x"]
        assert runners.main(argv) == code
        assert roster_host(spec_path, "lavender") == grown
        assert pathlib.Path(spec_path).read_text(encoding="utf-8").endswith("}\n")


class TestRebuildMode:
    """The --rebuild mode: its confirmation, its host, and its exit codes."""

    def test_rebuild_requires_host(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="--rebuild requires --host"):
            runners.main(["--spec", spec_path, "--rebuild", "yes"])

    def test_rebuild_takes_only_the_literal_yes(self, tmp_path: pathlib.Path) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        with pytest.raises(ValueError, match="takes the literal value yes"):
            runners.main(["--spec", spec_path, "--host", "lavender", "--rebuild", "true"])

    def test_a_clean_rebuild_exits_zero(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec_path = write_roster(tmp_path, [raw_host("lavender")])
        lavender = roster_host(spec_path, "lavender")
        runner = FakeRun(quiet_rebuild_answers(clean_transcript(lavender), lavender, repos=1))
        _test_hooks.run = runner
        with caplog.at_level("INFO"):
            code = runners.main(["--spec", spec_path, "--host", "lavender", "--rebuild", "yes"])
        assert code == 0
        assert len(runner.calls) == 9
        assert caplog.records[-1].getMessage() == "rebuild verified: every check green on lavender"

    def test_a_manual_asset_is_printed_after_the_stages_and_drifts_until_placed(
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
        _test_hooks.run = FakeRun(quiet_rebuild_answers(transcript, lavender, repos=1))
        with caplog.at_level("INFO"):
            code = runners.main(["--spec", spec_path, "--host", "lavender", "--rebuild", "yes"])
        assert code == 1
        messages = [record.getMessage() for record in caplog.records]
        assert messages[:4] == [
            "lavender rebuild: base: laid, and nothing asked for a reboot",
            "lavender rebuild: tokens: minted for 1 repositories",
            "lavender rebuild: provision: ran for 1 runners",
            f"lavender PLACE BY HAND: {jar} -- the licensed engine. This asset can never be "
            "fetched by a script; place it, then re-run fleet-runners audit.",
        ]
        assert messages[-1] == "1 fault(s) after the rebuild; the drift lines name the repairs"
