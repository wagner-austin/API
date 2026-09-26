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
from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.core import _test_hooks, dialect_linux, dialect_windows, probe
from tests.conftest import FakeRun, ok


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
        budget=NodeBudget(
            reserved_cores=2,
            reserved_ram_gb=4.0,
            worker_ram_gb=1.1,
            max_concurrent_runs=2,
            max_disk_gb=20.0,
        ),
    )


class TestProbe:
    def test_it_reads_the_fields_the_script_emits(self) -> None:
        output = "free_ram_gb=27.395\nfree_disk_gb=860.123\nlogical_cores=16\n"

        state = probe.parse_probe("lavender", output, live_runs=2)

        assert state == {
            "host": "lavender",
            "free_ram_gb": 27.395,
            "free_disk_gb": 860.123,
            "live_runs": 2,
        }

    def test_a_thousands_separator_is_read(self) -> None:
        """PowerShell's N3 format writes them; the value is still a number."""
        output = "free_ram_gb=1,027.395\nfree_disk_gb=860.000\n"

        assert probe.parse_probe("lavender", output, live_runs=0)["free_ram_gb"] == 1027.395

    def test_a_line_without_an_equals_is_ignored(self) -> None:
        """PowerShell writes warnings to the same stream."""
        output = "WARNING: something\nfree_ram_gb=1.0\nfree_disk_gb=2.0\n"

        assert probe.parse_probe("lavender", output, live_runs=0)["free_ram_gb"] == 1.0

    def test_a_missing_field_names_itself(self) -> None:
        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("lavender", "free_ram_gb=1.0\n", live_runs=0)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "free_disk_gb" in excinfo.value.message

    def test_a_non_numeric_field_shows_what_the_node_said(self) -> None:
        """The usual cause is a PowerShell error printed where a number goes."""
        output = "free_ram_gb=Cannot find drive\nfree_disk_gb=1.0\n"

        with pytest.raises(AppError) as excinfo:
            probe.parse_probe("lavender", output, live_runs=0)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "Cannot find drive" in excinfo.value.message

    def test_a_field_with_two_decimal_points_is_not_a_number(self) -> None:
        with pytest.raises(AppError, match="not a number"):
            probe.parse_probe("lavender", "free_ram_gb=1.2.3\nfree_disk_gb=1.0\n", live_runs=0)

    def test_a_signed_value_is_a_number(self) -> None:
        state = probe.parse_probe("lavender", "free_ram_gb=-1.0\nfree_disk_gb=+2.0\n", live_runs=0)

        assert state["free_ram_gb"] == -1.0
        assert state["free_disk_gb"] == 2.0

    def test_a_bare_sign_is_not_a_number(self) -> None:
        with pytest.raises(AppError, match="not a number"):
            probe.parse_probe("lavender", "free_ram_gb=-\nfree_disk_gb=1.0\n", live_runs=0)

    def test_probing_a_node_sends_the_script_and_parses_the_answer(self) -> None:
        runner = FakeRun([ok(""), ok("free_ram_gb=27.0\nfree_disk_gb=860.0\n")])
        _test_hooks.run = runner

        state = probe.probe_node(_node(), live_runs=1)

        assert state["free_ram_gb"] == 27.0
        assert state["live_runs"] == 1
        assert runner.stdin[0] == dialect_windows.CAPACITY_PROBE_SCRIPT.encode("utf-8")
        assert runner.calls[0][-1].endswith("C:/fleet/stage/fleet-capacity.ps1' -Encoding utf8\"")

    def test_the_script_arrives_byte_identical_to_the_constant(self) -> None:
        """THE RENDER-AND-SEND RULE, asserted rather than described.

        The bytes on the wire are the constant itself, so no value this
        package holds can carry a quote into a shell. The braces inside it
        are PowerShell's own format operator and are evaluated on the far
        side -- Python never touches them, which is exactly what this
        equality proves.
        """
        runner = FakeRun([ok(""), ok("free_ram_gb=1.0\nfree_disk_gb=1.0\n")])
        _test_hooks.run = runner

        probe.probe_node(_node(), live_runs=0)

        assert runner.stdin[0] == dialect_windows.CAPACITY_PROBE_SCRIPT.encode("utf-8")
        assert "{0:N3}" in dialect_windows.CAPACITY_PROBE_SCRIPT

    def test_a_linux_node_is_probed_with_the_sh_constant(self) -> None:
        runner = FakeRun([ok(""), ok("free_ram_gb=26.394\nfree_disk_gb=687.729\n")])
        _test_hooks.run = runner
        node = _node()
        node["platform"] = NodePlatform.LINUX
        node["stage_root"] = "/home/corvis/fleet/stage"

        state = probe.probe_node(node, live_runs=0)

        assert state["free_disk_gb"] == 687.729
        assert runner.stdin[0] == dialect_linux.CAPACITY_PROBE_SCRIPT.encode("utf-8")
        assert runner.calls[0][-1].endswith("cat > '/home/corvis/fleet/stage/fleet-capacity.sh'")
        assert runner.calls[1][-2:] == ("/bin/sh", "/home/corvis/fleet/stage/fleet-capacity.sh")
