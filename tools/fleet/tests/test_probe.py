"""The node capacity probe: parsing what the script prints, and sending it.

Split from ``test_core_io.py`` when that file passed the 600-line ceiling:
the probe is its own module (:mod:`fleet.core.probe`) with its own concern,
reading a node's free memory and disk, while the records, the ssh seam and
the default hooks stay together there. The fakes are the same ones, from
``conftest``, implementing the real ``RunProtocol``.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import (
    LiveLoad,
    NodeConfig,
    NodeGpu,
    NodePlatform,
    NodeState,
    describe_node,
)
from fleet.core import _test_hooks, dialect_linux, dialect_windows, probe
from tests.conftest import IDLE, FakeRun, ok


def _node() -> NodeConfig:
    """Build a node for the probe tests.

    Returns:
        The node.
    """
    return NodeConfig(
        host="lavender",
        platform=NodePlatform.WINDOWS,
        stage_root="C:/fleet/stage",
        logical_cores=16,
        ram_gb=32.0,
        gpu=None,
        enabled=True,
        test_database=False,
        rust=None,
        cxx=None,
        docker=None,
        stack=None,
        elevated=False,
        wsl_host=None,
        budget=NodeBudget(
            reserved_cores=2,
            reserved_ram_gb=4.0,
            worker_ram_gb=1.1,
            max_disk_gb=20.0,
            checks_at_once=None,
        ),
    )


class TestProbe:
    def test_it_reads_the_fields_the_script_emits(self) -> None:
        output = "free_ram_gb=27.395\nfree_disk_gb=860.123\nlogical_cores=16\nreserved_cores=4\n"

        held = LiveLoad(runs=2, workers=4, ram_gb=4.4)

        state = probe.parse_probe("lavender", output, live=held)

        assert state == {
            "host": "lavender",
            "free_ram_gb": 27.395,
            "free_disk_gb": 860.123,
            "live": held,
            "ci_slice": None,
            "reserved_cores": 4,
            "action_cache": None,
        }

    def test_a_runner_host_reports_its_action_cache_counts(self) -> None:
        """lavender-wsl's runners on 2026-10-15, as its root timer counts them (MCPs ce2b92e0)."""
        output = (
            "free_ram_gb=8.161\nfree_disk_gb=824.306\nreserved_cores=0\n"
            "action_cache_runners=8\naction_cache_reading=8\n"
            "action_cache_served=7\naction_cache_missed=1\n"
        )

        state = probe.parse_probe("lavender-wsl", output, live=IDLE)

        assert state["action_cache"] == {"runners": 8, "reading": 8, "served": 7, "missed": 1}

    @pytest.mark.parametrize(
        ("extra", "named"),
        [
            ("action_cache_runners=8\n", "without a 'action_cache_reading' field"),
            (
                "action_cache_runners=8\naction_cache_reading=7.5\n"
                "action_cache_served=0\naction_cache_missed=0\n",
                "action_cache_reading=7.5, which is not a whole number of runners",
            ),
            (
                "action_cache_runners=8\naction_cache_reading=9\n"
                "action_cache_served=0\naction_cache_missed=0\n",
                "8 runner(s), 9 reading, 0 served, 0 missed",
            ),
        ],
        ids=["three counts missing", "a fractional count", "more reading than counted"],
    )
    def test_a_partial_or_inconsistent_cache_reading_is_unreadable_not_ignored(
        self, extra: str, named: str
    ) -> None:
        output = "free_ram_gb=8.1\nfree_disk_gb=9.0\nreserved_cores=0\n" + extra

        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("lavender-wsl", output, live=IDLE)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert named in excinfo.value.message

    def test_a_node_with_a_ci_slice_reports_its_current_and_high(self) -> None:
        """lavender-wsl's answer to the new probe, 2026-09-29 (MCPs 5d6e57e7)."""
        output = (
            "free_ram_gb=8.161\nfree_disk_gb=824.306\nlogical_cores=16\n"
            "ci_slice_current_gb=16.000\nci_slice_high_gb=16.000\nreserved_cores=0\n"
        )

        state = probe.parse_probe("lavender-wsl", output, live=IDLE)

        assert state["ci_slice"] == {"current_gb": 16.0, "high_gb": 16.0}
        assert state["free_ram_gb"] == 8.161

    @pytest.mark.parametrize(
        ("extra", "named"),
        [
            ("ci_slice_current_gb=16.000\n", "without a 'ci_slice_high_gb' field"),
            ("ci_slice_current_gb=max\nci_slice_high_gb=16.000\n", "ci_slice_current_gb='max'"),
        ],
    )
    def test_half_a_slice_reading_is_unreadable_not_ignored(self, extra: str, named: str) -> None:
        output = "free_ram_gb=8.1\nfree_disk_gb=9.0\nreserved_cores=0\n" + extra

        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("lavender-wsl", output, live=IDLE)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert named in excinfo.value.message

    def test_a_thousands_separator_is_read(self) -> None:
        """PowerShell's N3 format writes them; the value is still a number."""
        output = "free_ram_gb=1,027.395\nfree_disk_gb=860.000\nreserved_cores=0\n"

        assert probe.parse_probe("lavender", output, live=IDLE)["free_ram_gb"] == 1027.395

    def test_a_line_without_an_equals_is_ignored(self) -> None:
        """PowerShell writes warnings to the same stream."""
        output = "WARNING: something\nfree_ram_gb=1.0\nfree_disk_gb=2.0\nreserved_cores=0\n"

        assert probe.parse_probe("lavender", output, live=IDLE)["free_ram_gb"] == 1.0

    def test_a_missing_field_names_itself(self) -> None:
        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("lavender", "free_ram_gb=1.0\n", live=IDLE)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "free_disk_gb" in excinfo.value.message

    def test_a_non_numeric_field_shows_what_the_node_said(self) -> None:
        """The usual cause is a PowerShell error printed where a number goes."""
        output = "free_ram_gb=Cannot find drive\nfree_disk_gb=1.0\n"

        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("lavender", output, live=IDLE)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "Cannot find drive" in excinfo.value.message

    def test_a_field_with_two_decimal_points_is_not_a_number(self) -> None:
        with pytest.raises(AppError, match="not a number"):
            probe.parse_probe("lavender", "free_ram_gb=1.2.3\nfree_disk_gb=1.0\n", live=IDLE)

    def test_a_signed_value_is_a_number(self) -> None:
        state = probe.parse_probe(
            "lavender", "free_ram_gb=-1.0\nfree_disk_gb=+2.0\nreserved_cores=0\n", live=IDLE
        )

        assert state["free_ram_gb"] == -1.0
        assert state["free_disk_gb"] == 2.0

    def test_a_node_without_its_reservations_line_is_unreadable(self) -> None:
        """MCPs board task 84a2b690: a probe that sums no reservations has not run as written."""
        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("diphtheria", "free_ram_gb=1.0\nfree_disk_gb=2.0\n", live=IDLE)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "without a 'reserved_cores' field" in excinfo.value.message

    def test_a_reservation_file_it_cannot_read_refuses_the_node_by_name(self) -> None:
        output = (
            "free_ram_gb=1.0\nfree_disk_gb=2.0\nlogical_cores=8\n"
            "reservation_unreadable=/home/corvis/.fleet/reservations/vm.reservation\n"
        )

        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("diphtheria", output, live=IDLE)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert excinfo.value.message == (
            "diphtheria holds a reservation file its probe cannot read, "
            "/home/corvis/.fleet/reservations/vm.reservation: it needs whole-number cores= and "
            "until_unix= lines (fleet.core.reservations), and a reservation read as nothing "
            "would let a suite onto the cores it holds"
        )

    @pytest.mark.parametrize("held", ["1.5", "-1"])
    def test_reserved_cores_must_be_a_whole_number_of_cores(self, held: str) -> None:
        output = f"free_ram_gb=1.0\nfree_disk_gb=2.0\nreserved_cores={held}\n"

        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("diphtheria", output, live=IDLE)

        assert excinfo.value.message == (
            f"diphtheria reported reserved_cores={float(held)!r}, "
            "which is not a whole number of cores"
        )

    def test_a_bare_sign_is_not_a_number(self) -> None:
        with pytest.raises(AppError, match="not a number"):
            probe.parse_probe("lavender", "free_ram_gb=-\nfree_disk_gb=1.0\n", live=IDLE)

    def test_probing_a_node_sends_the_script_and_parses_the_answer(self) -> None:
        runner = FakeRun([ok(""), ok("free_ram_gb=27.0\nfree_disk_gb=860.0\nreserved_cores=0\n")])
        _test_hooks.run = runner

        held = LiveLoad(runs=1, workers=2, ram_gb=2.2)

        state = probe.probe_node(_node(), live=held, writer="fleet-run")

        assert state["free_ram_gb"] == 27.0
        assert state["live"] == held
        assert runner.stdin[0] == dialect_windows.CAPACITY_PROBE_SCRIPT.encode("utf-8")
        assert runner.calls[0][-1].endswith(
            "C:/fleet/stage/fleet-capacity-fleet-run.ps1' -Encoding utf8\""
        )

    def test_the_script_arrives_byte_identical_to_the_constant(self) -> None:
        """THE RENDER-AND-SEND RULE, asserted rather than described.

        The bytes on the wire are the constant itself, so no value this
        package holds can carry a quote into a shell. The braces inside it
        are PowerShell's own format operator and are evaluated on the far
        side -- Python never touches them, which is exactly what this
        equality proves.
        """
        runner = FakeRun([ok(""), ok("free_ram_gb=1.0\nfree_disk_gb=1.0\nreserved_cores=0\n")])
        _test_hooks.run = runner

        probe.probe_node(_node(), live=IDLE, writer="fleet-run")

        assert runner.stdin[0] == dialect_windows.CAPACITY_PROBE_SCRIPT.encode("utf-8")
        assert "{0:N3}" in dialect_windows.CAPACITY_PROBE_SCRIPT

    def test_a_linux_node_is_probed_with_the_sh_constant(self) -> None:
        runner = FakeRun(
            [ok(""), ok("free_ram_gb=26.394\nfree_disk_gb=687.729\nreserved_cores=4\n")]
        )
        _test_hooks.run = runner
        node = _node()
        node["platform"] = NodePlatform.LINUX
        node["stage_root"] = "/home/corvis/fleet/stage"

        state = probe.probe_node(node, live=IDLE, writer="lavender-wsl")

        assert state["free_disk_gb"] == 687.729
        assert state["reserved_cores"] == 4
        assert runner.stdin[0] == dialect_linux.CAPACITY_PROBE_SCRIPT.encode("utf-8")
        script = "/home/corvis/fleet/stage/fleet-capacity-lavender-wsl.sh"
        assert runner.calls[0][-1].endswith(f"cat > '{script}'")
        assert runner.calls[1][-2:] == ("/bin/sh", script)


class TestDescribeNode:
    """The line fleet-nodes prints for a probed node, live runs included."""

    STATE = NodeState(
        host="lavender",
        free_ram_gb=27.4,
        free_disk_gb=860.0,
        ci_slice=None,
        live=LiveLoad(runs=1, workers=6, ram_gb=6.6),
        reserved_cores=0,
        action_cache=None,
    )

    def test_it_names_the_architecture_and_what_live_runs_hold(self) -> None:
        node = _node()
        node["gpu"] = NodeGpu(
            model="NVIDIA GeForce GTX 1630",
            vram_mib=4096,
            compute_capability="7.5",
            driver_version="591.86",
        )

        assert describe_node(node, self.STATE) == (
            "lavender: NVIDIA GeForce GTX 1630 sm_7.5, 27.4/32.0 GB RAM free, 860 GB disk free, "
            "1 live run(s) holding 6 worker(s)"
        )

    def test_a_cpu_only_node_says_so(self) -> None:
        assert describe_node(_node(), self.STATE).startswith("lavender: cpu-only, ")
