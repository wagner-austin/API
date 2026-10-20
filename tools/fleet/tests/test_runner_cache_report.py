"""The action cache report counts a host's runners for the node runner's probe (MCPs ce2b92e0).

The report runs under a real bash over a host laid out as files
(tests/_cache_host.py says how), its Worker logs written as the runner
writes them, with real mtimes the host's own ``find`` reads. What it writes
is what the probe forwards (:mod:`fleet.contracts.action_cache`), so each
case reads the status file whole.
"""

from __future__ import annotations

import pathlib

from fleet.contracts.action_cache import PROBE_FIELDS
from tests._cache_host import OTHER, UNIT, Host
from tests._runner_fixtures import an_install

#: The other two MCPs units lavender runs, for the cases that need four.
SECOND = an_install("wagner-austin/MCPs", "lavender-2", slot=2)["service"]
THIRD = an_install("wagner-austin/MCPs", "lavender-3", slot=3)["service"]


def _status(runners: int, reading: int, served: int, missed: int) -> str:
    """The status file's text.

    Args:
        runners: Units examined.
        reading: Those whose process reads the cache.
        served: Those whose newest job found every action in it.
        missed: Those whose newest job missed one.

    Returns:
        The four lines, in the probe's order.
    """
    values = (runners, reading, served, missed)
    return "".join(f"{key}={value}\n" for key, value in zip(PROBE_FIELDS, values, strict=True))


class TestTheReportCountsEachRunnerOnce:
    def test_lavender_s_four_kinds_of_runner_are_each_counted_where_they_belong(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        # Served: its newest job since starting found all three actions. The
        # older log, from before its process started, missed one and is not read.
        host.unit(UNIT, 100, adopted=True, elapsed=1800)
        host.worker_log(UNIT, "20261015-130000", minutes_ago=40, lookups=3, hits=2)
        host.worker_log(UNIT, "20261015-151225", minutes_ago=5, lookups=3, hits=3)
        # Missed: its newest job looked up two and the cache held one.
        host.unit(SECOND, 200, adopted=True, elapsed=1800)
        host.worker_log(SECOND, "20261015-151000", minutes_ago=8, lookups=2, hits=1)
        # Reading, with no job since its process started.
        host.unit(THIRD, 300, adopted=True, elapsed=60)
        host.worker_log(THIRD, "20261015-140000", minutes_ago=30, lookups=2, hits=2)
        # Not reading: its process predates the drop-in.
        host.unit(OTHER, 400, loaded=False)
        host.worker_log(OTHER, "20261015-151100", minutes_ago=3, lookups=0, hits=0)

        ran = host.report(UNIT, SECOND, THIRD, OTHER)

        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == ""
        assert host.status() == _status(runners=4, reading=3, served=1, missed=1)
        assert (tmp_path / "statusdir" / "status.part").exists() is False

    def test_a_stopped_unit_and_a_process_gone_before_ps_asks_count_but_are_not_read(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, running=False, adopted=True)
        host.unit(SECOND, 200, adopted=True, elapsed=None)
        host.worker_log(SECOND, "20261015-151000", minutes_ago=1, lookups=2, hits=2)

        ran = host.report(UNIT, SECOND)

        assert ran.returncode == 0, ran.stderr
        assert host.status() == _status(runners=2, reading=1, served=0, missed=0)

    def test_a_job_that_looked_up_nothing_is_neither_served_nor_missed(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, adopted=True, elapsed=600)
        host.worker_log(UNIT, "20261015-151225", minutes_ago=1, lookups=0, hits=0)

        assert host.report(UNIT).returncode == 0
        assert host.status() == _status(runners=1, reading=1, served=0, missed=0)

    def test_each_run_replaces_the_file_whole(self, tmp_path: pathlib.Path) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, adopted=True, elapsed=600)
        host.worker_log(UNIT, "20261015-151225", minutes_ago=1, lookups=1, hits=0)
        assert host.report(UNIT).returncode == 0
        assert host.status() == _status(runners=1, reading=1, served=0, missed=1)

        host.worker_log(UNIT, "20261015-152500", minutes_ago=0, lookups=1, hits=1)

        assert host.report(UNIT).returncode == 0
        assert host.status() == _status(runners=1, reading=1, served=1, missed=0)

    def test_a_call_naming_no_unit_is_refused_with_its_usage(self, tmp_path: pathlib.Path) -> None:
        ran = Host(tmp_path).report()

        assert ran.returncode == 2
        assert ran.stderr == "fleet-action-cache-report: usage: fleet-action-cache-report UNIT...\n"
        assert (tmp_path / "statusdir").exists() is False
