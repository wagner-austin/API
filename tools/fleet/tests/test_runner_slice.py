"""The runners' share of a CI host: the budget's contract and its units (board task 45a4f22b)."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.cli.runners import load_runner_spec
from fleet.contracts.runner_slice import (
    LISTENER_SLICE_NAME,
    CiSlice,
    decode_ci_slice,
    encode_ci_slice,
)
from fleet.core import runner_slice_render
from tests._runner_fixtures import a_ci_slice, ci_slice_json


def _budget(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Lavender's raw budget with fields overridden.

    Args:
        overrides: Field values replacing lavender's.

    Returns:
        The raw mapping.
    """
    raw = ci_slice_json()
    raw.update(overrides)
    return raw


class TestContract:
    """decode_ci_slice's refusals and the round trip."""

    def test_lavender_s_budget_decodes_and_round_trips(self) -> None:
        decoded: CiSlice = decode_ci_slice(ci_slice_json(), host_memory_gb=31)
        assert decoded == a_ci_slice()
        assert encode_ci_slice(decoded) == ci_slice_json()

    @pytest.mark.parametrize(
        "field",
        [
            "memory_high_gb",
            "memory_max_gb",
            "swap_max_gb",
            "cpu_weight",
            "io_read_mb_per_s",
            "io_read_iops",
        ],
    )
    def test_a_missing_field_is_refused(self, field: str) -> None:
        raw = _budget()
        del raw[field]
        with pytest.raises(JSONTypeError, match=field):
            decode_ci_slice(raw, host_memory_gb=31)

    def test_a_throttle_point_of_nothing_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="memory_high_gb must be positive, got 0"):
            decode_ci_slice(_budget(memory_high_gb=0), host_memory_gb=31)

    def test_a_slice_may_be_allowed_no_swap_at_all(self) -> None:
        assert decode_ci_slice(_budget(swap_max_gb=0), host_memory_gb=31)["swap_max_gb"] == 0

    def test_a_negative_swap_bound_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="swap_max_gb must be 0 or more, got -1"):
            decode_ci_slice(_budget(swap_max_gb=-1), host_memory_gb=31)

    @pytest.mark.parametrize("field", ["io_read_mb_per_s", "io_read_iops"])
    @pytest.mark.parametrize("bound", [0, -5])
    def test_a_read_bound_of_nothing_is_refused(self, field: str, bound: int) -> None:
        with pytest.raises(
            JSONTypeError, match=f"ci_slice.{field} must be positive, got {bound}; a read bound"
        ):
            decode_ci_slice(_budget(**{field: bound}), host_memory_gb=31)

    def test_a_ceiling_below_the_throttle_point_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="kill jobs before systemd ever throttled"):
            decode_ci_slice(_budget(memory_max_gb=15), host_memory_gb=31)

    @pytest.mark.parametrize("weight", [0, 10001])
    def test_a_weight_outside_systemd_s_range_is_refused(self, weight: int) -> None:
        with pytest.raises(JSONTypeError, match=f"between 1 and 10000, got {weight}"):
            decode_ci_slice(_budget(cpu_weight=weight), host_memory_gb=31)

    @pytest.mark.parametrize("host_memory_gb", [18, 12])
    def test_a_ceiling_that_leaves_the_host_nothing_is_refused(self, host_memory_gb: int) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_ci_slice(ci_slice_json(), host_memory_gb=host_memory_gb)
        assert str(refused.value) == (
            f"ci_slice.memory_max_gb 18 leaves nothing of the host's {host_memory_gb} GB for "
            "sshd and the fleet's own checks, which is the starvation this ceiling exists to "
            "prevent"
        )


class TestSliceUnits:
    """The slice units and the drop-in that places each runner."""

    def test_the_unit_carries_the_budget(self) -> None:
        assert runner_slice_render.render_slice_unit(a_ci_slice()) == (
            "[Unit]\n"
            "Description=GitHub Actions runners' share of this host, rendered by fleet-runners "
            "(MCPs board tasks 45a4f22b, cb264851)\n"
            "\n"
            "[Slice]\n"
            "MemoryAccounting=yes\n"
            "CPUAccounting=yes\n"
            "IOAccounting=yes\n"
            "MemoryHigh=16G\n"
            "MemoryMax=18G\n"
            "MemorySwapMax=2G\n"
            "CPUWeight=20\n"
            "IOReadBandwidthMax=/ 200M\n"
            "IOReadIOPSMax=/ 5000\n"
        )

    def test_the_roster_s_lavender_unit_bounds_swap_and_root_disk_reads(self) -> None:
        """The unit rendered from runners.json itself, so a roster edit that
        drops either bound fails here rather than on the host."""
        roster = load_runner_spec(str(pathlib.Path(__file__).parent.parent / "runners.json"))
        lavender = next(host for host in roster["hosts"] if host["name"] == "lavender")
        unit = runner_slice_render.render_slice_unit(lavender["ci_slice"])
        assert "\nMemorySwapMax=2G\n" in unit
        assert "\nIOReadBandwidthMax=/ 200M\nIOReadIOPSMax=/ 5000\n" in unit

    def test_a_slice_allowed_no_swap_renders_a_zero_bound(self) -> None:
        budget = a_ci_slice()
        budget["swap_max_gb"] = 0
        assert "\nMemorySwapMax=0G\n" in runner_slice_render.render_slice_unit(budget)

    def test_the_listener_slice_is_accounted_and_bounded_by_nothing(self) -> None:
        """No Memory*, CPUWeight or IO*Max: a bound here would throttle the
        listeners as the CI slice's did (MCPs board task 715b9ec5)."""
        unit = runner_slice_render.LISTENER_SLICE_UNIT
        directives = unit.split("[Slice]\n", 1)[1].splitlines()
        assert directives == ["MemoryAccounting=yes", "CPUAccounting=yes", "IOAccounting=yes"]
        assert LISTENER_SLICE_NAME == "listeners.slice"
        assert "-" not in LISTENER_SLICE_NAME

    def test_each_runner_is_placed_in_the_listener_slice_by_its_drop_in(self) -> None:
        assert runner_slice_render.SLICE_DROP_IN == "[Service]\nSlice=listeners.slice\n"
        assert runner_slice_render.SLICE_DROP_IN_NAME == "fleet-slice.conf"
