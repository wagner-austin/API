"""The load sampler: forest parsing, descendant counting, the record, the report.

The forests here are shaped like the measured incident (2026-09-08, load 78:
DataLoader children invisible to every worker literal), and the report tests
pin rule 3 of the spec -- the instrument must be able to say NO.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, load_json_str

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, remote, runner_load
from tests.conftest import FakeRun, failed, ok


def _host() -> HostRunnerSpec:
    """A two-install host, one per repo, mirroring lavender's shape.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=None,
        wslconfig_min_memory_gb=None,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        installs=[
            RunnerInstall(
                repo="wagner-austin/API",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-API.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner-api-1/_work",
                labels=["lavender-wsl"],
            ),
            RunnerInstall(
                repo="wagner-austin/MCPs",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-MCPs.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner/_work",
                labels=["lavender-wsl"],
            ),
        ],
        assets=[],
    )


#: A forest with the incident's shape: an API worker whose xdist child has
#: DataLoader grandchildren, an MCPs worker with one vitest child, and
#: unrelated processes that must count for nobody.
_FOREST = "\n".join(
    [
        "1 0 /sbin/init",
        "100 1 /home/gharunner/actions-runner-api-1/bin/Runner.Listener run",
        "101 100 /home/gharunner/actions-runner-api-1/bin/Runner.Worker spawnclient",
        "102 101 python -m pytest -n 8",
        "103 102 python xdist worker gw0",
        "104 103 pt_data_worker",
        "105 103 pt_data_worker",
        "200 1 /home/gharunner/actions-runner/bin/Runner.Listener run",
        "201 200 /home/gharunner/actions-runner/bin/Runner.Worker spawnclient",
        "202 201 node vitest-worker",
        "300 1 postgres",
        "301 300 postgres checkpointer",
    ]
)


class TestParseProcessForest:
    """The ps parser."""

    def test_a_forest_parses_to_pid_ppid_args(self) -> None:
        forest = runner_load.parse_process_forest(_FOREST)
        assert forest[104] == (103, "pt_data_worker")
        assert forest[1] == (0, "/sbin/init")

    def test_blank_lines_are_ignored(self) -> None:
        assert len(runner_load.parse_process_forest("\n" + _FOREST + "\n\n")) == 12

    @pytest.mark.parametrize("line", ["garbage", "12 not-a-pid python", "\x0012 1 x"])
    def test_a_malformed_line_is_refused_not_skipped(self, line: str) -> None:
        with pytest.raises(AppError) as fault:
            runner_load.parse_process_forest(_FOREST + "\n" + line)
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE


class TestTakeSample:
    """Scoring a forest against the roster."""

    def test_descendants_are_counted_transitively_per_install(self) -> None:
        sample = runner_load.take_sample(_host(), _FOREST, None, at=1757400000)
        assert sample == runner_load.LoadSample(
            at=1757400000,
            host="lavender",
            installs=[
                runner_load.InstallLoad(
                    repo="wagner-austin/API",
                    runner_name="lavender-wsl",
                    busy=True,
                    # pytest + gw0 + two pt_data_worker grandchildren.
                    descendants=4,
                ),
                runner_load.InstallLoad(
                    repo="wagner-austin/MCPs",
                    runner_name="lavender-wsl",
                    busy=True,
                    descendants=1,
                ),
            ],
            total=5,
        )

    def test_an_idle_install_reads_zero_and_not_busy(self) -> None:
        quiet = "\n".join(
            [
                "1 0 /sbin/init",
                "100 1 /home/gharunner/actions-runner-api-1/bin/Runner.Listener run",
                "300 1 postgres",
            ]
        )
        sample = runner_load.take_sample(_host(), quiet, None, at=7)
        assert sample["total"] == 0
        assert [install["busy"] for install in sample["installs"]] == [False, False]

    def test_unrelated_processes_count_for_nobody(self) -> None:
        sample = runner_load.take_sample(_host(), _FOREST, None, at=7)
        # postgres and its checkpointer appear in the forest and in no install.
        assert sample["total"] == 5


def _mixed_host() -> HostRunnerSpec:
    """A host with one install per side, mirroring lavender since tree-bot.

    Returns:
        The spec.
    """
    spec = _host()
    spec["installs"].append(
        RunnerInstall(
            repo="wagner-austin/tree-bot",
            runner_name="lavender",
            side="windows",
            service="actions.runner.wagner-austin-tree-bot.lavender",
            workdir="C:/actions-runner-tree-bot/_work",
            labels=["lavender"],
        )
    )
    return spec


#: A Windows process forest in the same three-column shape, with the
#: backslashed command lines Win32 actually reports, one runner worker with
#: one child, and a pid that COLLIDES with the wsl forest's -- the reason
#: the two forests must never be merged.
_WINDOWS_FOREST = "\n".join(
    [
        "4 0 System",
        "100 4 C:\\actions-runner-tree-bot\\bin\\Runner.Worker.exe spawnclient",
        "101 100 python.exe -m pytest",
        "300 4 svchost.exe",
    ]
)


class TestDualForests:
    """Windows-side installs are scored against the Windows forest."""

    def test_each_side_counts_only_its_own_forest(self) -> None:
        sample = runner_load.take_sample(_mixed_host(), _FOREST, _WINDOWS_FOREST, at=1757400000)
        by_repo = {install["repo"]: install for install in sample["installs"]}
        # wsl pid 100 is a Runner.Listener; windows pid 100 is a Worker.
        # If the forests were merged, the collision would corrupt both.
        assert by_repo["wagner-austin/tree-bot"]["descendants"] == 1
        assert by_repo["wagner-austin/tree-bot"]["busy"] is True
        assert sample["total"] == 6

    def test_a_windows_install_with_no_windows_forest_is_refused(self) -> None:
        with pytest.raises(ValueError, match="looking away"):
            runner_load.take_sample(_mixed_host(), _FOREST, None, at=1)

    def test_sample_host_fetches_the_windows_forest_only_when_needed(self) -> None:
        runner = FakeRun([ok(_FOREST), ok(""), ok(_WINDOWS_FOREST)])
        _test_hooks.run = runner
        sample = runner_load.sample_host(_mixed_host(), at=11)
        assert sample["total"] == 6
        # Three calls: the wsl ps, then send + run of the forest script.
        assert len(runner.calls) == 3
        assert runner.stdin[1] == runner_load.WINDOWS_FOREST_SCRIPT.encode("utf-8")


class TestSampleHost:
    """The ssh seam."""

    def test_the_command_is_ps_inside_the_distro(self) -> None:
        runner = FakeRun([ok(_FOREST)])
        _test_hooks.run = runner
        sample = runner_load.sample_host(_host(), at=11)
        assert sample["total"] == 5
        assert tuple(runner.calls[0]) == (
            "ssh",
            *remote.SSH_OPTIONS,
            "lavender",
            "wsl",
            "-d",
            "Ubuntu",
            "--",
            "ps",
            "-eo",
            "pid=,ppid=,args=",
        )

    def test_an_unreachable_host_raises_rather_than_recording_zero(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "No route to host")])
        with pytest.raises(AppError) as fault:
            runner_load.sample_host(_host(), at=11)
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE


class TestRecordRoundTrip:
    """The JSONL record."""

    def test_encode_then_decode_is_the_identity(self) -> None:
        sample = runner_load.take_sample(_host(), _FOREST, None, at=1757400000)
        line = runner_load.render_sample_line(sample)
        assert runner_load.decode_load_sample(load_json_str(line)) == sample

    def test_a_total_disagreeing_with_its_installs_is_a_corrupt_record(self) -> None:
        sample = runner_load.take_sample(_host(), _FOREST, None, at=1)
        broken = dict(runner_load.encode_load_sample(sample))
        broken["total"] = 99
        with pytest.raises(JSONTypeError, match="corrupt record"):
            runner_load.decode_load_sample(broken)

    def test_a_non_object_line_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            runner_load.decode_load_sample([1, 2])

    def test_a_non_object_install_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            runner_load.decode_load_sample(
                {"at": 1, "host": "lavender", "installs": [3], "total": 0}
            )

    def test_a_file_of_lines_decodes_in_order(self) -> None:
        first = runner_load.take_sample(_host(), _FOREST, None, at=1)
        second = runner_load.take_sample(_host(), _FOREST, None, at=2)
        raw = (
            runner_load.render_sample_line(first)
            + "\n"
            + runner_load.render_sample_line(second)
            + "\n"
        )
        assert [s["at"] for s in runner_load.decode_sample_file(raw)] == [1, 2]


class TestReport:
    """Rule 3: the instrument must be able to say no."""

    def _sample_with_total(self, total: int, *, at: int) -> runner_load.LoadSample:
        """A sample whose one install carries the given load.

        Args:
            total: The descendant count.
            at: The timestamp.

        Returns:
            The sample.
        """
        return runner_load.LoadSample(
            at=at,
            host="lavender",
            installs=[
                runner_load.InstallLoad(
                    repo="wagner-austin/API",
                    runner_name="lavender-wsl",
                    busy=total > 0,
                    descendants=total,
                )
            ],
            total=total,
        )

    def test_a_quiet_box_reports_zero_over_and_full_under(self) -> None:
        samples = [self._sample_with_total(n, at=n) for n in (0, 3, 8)]
        report = runner_load.report_over_cores(samples, cores=16)
        assert report == runner_load.LoadReport(
            samples=3,
            minimum=0,
            maximum=8,
            over_fraction=0.0,
            under_fraction=1.0,
            longest_over_streak=0,
        )

    def test_a_streak_is_consecutive_not_total(self) -> None:
        totals = [40, 40, 2, 40, 40, 40, 2]
        samples = [self._sample_with_total(t, at=i) for i, t in enumerate(totals)]
        report = runner_load.report_over_cores(samples, cores=16)
        assert report["longest_over_streak"] == 3
        assert report["over_fraction"] == pytest.approx(5 / 7)
        assert report["under_fraction"] == pytest.approx(2 / 7)
        assert report["minimum"] == 2

    def test_an_empty_record_is_refused_because_it_reads_idle(self) -> None:
        with pytest.raises(ValueError, match="measured nothing"):
            runner_load.report_over_cores([], cores=16)

    def test_a_non_positive_core_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            runner_load.report_over_cores([self._sample_with_total(1, at=1)], cores=0)
