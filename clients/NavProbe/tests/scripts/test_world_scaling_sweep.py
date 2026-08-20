"""Tests for the world-count scaling ladder."""

from __future__ import annotations

import pytest
from scripts.arguments import DEFAULT_DEVICE, ScriptArgumentError
from scripts.world_scaling_sweep import (
    PERTURBATION,
    SCENE,
    TRIAL,
    main,
    parse_invocation,
    rung_from,
)

from navprobe.codecs.scaling_run import SCALING_RUN_BANNER, decode_scaling_run
from tests.scripts.conftest import CLOCK_STEP, DEVICES, Harness


class TestParseInvocation:
    """Tests for :func:`parse_invocation`."""

    def test_reads_every_fixed_positional(self) -> None:
        """Mode, cache, record bound and capacity are all required."""
        parsed = parse_invocation(["RUN_TO_RUN", "cache", "64", "256", "2"])
        assert (parsed.mode_name, parsed.cache_dir, parsed.max_records, parsed.capacity) == (
            "RUN_TO_RUN",
            "cache",
            64,
            256,
        )

    def test_reads_the_variadic_world_counts(self) -> None:
        """The ladder is the whole point; every rung is read in order."""
        parsed = parse_invocation(["RUN_TO_RUN", "cache", "64", "256", "2", "64", "512", "4096"])
        assert parsed.world_counts == (2, 64, 512, 4096)

    def test_defaults_the_device(self) -> None:
        """A ladder without the flag runs on the first card."""
        assert parse_invocation(["M", "c", "0", "256", "2"]).device == DEFAULT_DEVICE

    def test_reads_a_device_flag_before_the_variadic_list(self) -> None:
        """Flag-first is the form the variadic tail forces."""
        parsed = parse_invocation(["--device", "cuda:1", "M", "c", "0", "256", "2", "64"])
        assert (parsed.device, parsed.world_counts) == ("cuda:1", (2, 64))

    def test_reads_a_device_flag_after_the_variadic_list(self) -> None:
        """Flag-last must not be swallowed as another world count."""
        parsed = parse_invocation(["M", "c", "0", "256", "2", "64", "--device", "cuda:1"])
        assert (parsed.device, parsed.world_counts) == ("cuda:1", (2, 64))

    def test_rejects_a_command_line_with_no_world_counts(self) -> None:
        """A ladder with no rungs measures nothing."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN", "cache", "64", "256"])
        assert caught.value.code == "NP-ARGS-005"

    def test_rejects_a_zero_capacity(self) -> None:
        """A zero constraint allocation cannot hold a contact."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN", "cache", "64", "0", "2"])
        assert caught.value.code == "NP-ARGS-004"

    def test_rejects_a_zero_world_count(self) -> None:
        """A rung of zero worlds simulates nothing."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN", "cache", "64", "256", "0"])
        assert caught.value.code == "NP-ARGS-004"

    def test_names_the_rung_it_rejected(self) -> None:
        """The message points at the offending rung, not just the list."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN", "cache", "64", "256", "2", "many"])
        assert "WORLDS[1]" in caught.value.message


class TestRungFrom:
    """Tests for :func:`rung_from`."""

    def test_derives_throughput_from_the_trial_design(self) -> None:
        """World-steps per second is the figure the ladder exists to produce."""
        rung = rung_from(64, "digest", True, None, 2.0)
        expected = 64 * TRIAL["step_count"] * TRIAL["repetitions"] / 2.0
        assert rung["world_steps_per_second"] == expected

    def test_carries_the_divergence_point(self) -> None:
        """A rung that failed to reproduce records where it parted."""
        assert rung_from(2, "d", False, 57, 1.0)["first_divergent_step"] == 57


class TestScalingRun:
    """Tests for :func:`main` against the fake runtime."""

    def test_returns_zero(self, harness: Harness) -> None:
        """A completed ladder exits clean."""
        assert main(["RUN_TO_RUN", "cache", "64", "256", "2", "64"]) == 0

    def test_opts_out_of_power_throttling_before_timing_anything(self, harness: Harness) -> None:
        """This ladder IS a wall-clock measurement; the opt-out is the point.

        Without it the throughput figures mix two power regimes, which is
        exactly why the published curve had to be marked provisional.
        """
        main(["RUN_TO_RUN", "cache", "64", "256", "2"])
        assert harness.opt_out.calls == 1

    def test_runs_every_rung_in_order(self, harness: Harness) -> None:
        """The ladder's order is the curve's x-axis."""
        main(["RUN_TO_RUN", "cache", "64", "256", "2", "64", "512"])
        record = decode_scaling_run(harness.writer.documents(SCALING_RUN_BANNER))
        assert [rung["world_count"] for rung in record["rungs"]] == [2, 64, 512]

    def test_passes_each_world_count_to_the_factory(self, harness: Harness) -> None:
        """A ladder that built every rung at one width would measure nothing."""
        main(["RUN_TO_RUN", "cache", "64", "256", "2", "64"])
        assert [call[1] for call in harness.construct.calls] == [2, 64]

    def test_passes_the_capacity_through(self, harness: Harness) -> None:
        """The right-sized capacity is what lets a large rung fit at all."""
        main(["RUN_TO_RUN", "cache", "64", "256", "2"])
        assert harness.construct.calls[0][3] == 256

    def test_records_the_measured_wall_time(self, harness: Harness) -> None:
        """The clock advances by a known step, so the figure is exact."""
        main(["RUN_TO_RUN", "cache", "64", "256", "2"])
        record = decode_scaling_run(harness.writer.documents(SCALING_RUN_BANNER))
        assert record["rungs"][0]["wall_seconds"] == CLOCK_STEP

    def test_records_the_resolved_device(self, harness: Harness) -> None:
        """A throughput figure is meaningless without the card that produced it."""
        main(["--device", "cuda:1", "RUN_TO_RUN", "cache", "64", "256", "2"])
        record = decode_scaling_run(harness.writer.documents(SCALING_RUN_BANNER))
        assert record["device"] == DEVICES["cuda:1"]

    def test_writes_a_decodable_record(self, harness: Harness) -> None:
        """The ladder is comparable against a repeat of itself."""
        main(["RUN_TO_RUN", "cache", "64", "256", "2"])
        record = decode_scaling_run(harness.writer.documents(SCALING_RUN_BANNER))
        assert (record["capacity"], record["scene"], record["spec"], record["perturbation"]) == (
            256,
            SCENE,
            TRIAL,
            PERTURBATION,
        )

    def test_streams_one_line_per_rung(self, harness: Harness) -> None:
        """A long ladder is legible while it runs."""
        main(["RUN_TO_RUN", "cache", "64", "256", "2", "64"])
        assert len([c for c in harness.writer.chunks if c.startswith("nworld=")]) == 2

    def test_reports_divergence_when_the_rollouts_disagree(self, drifting_harness: Harness) -> None:
        """The ladder reports the instrument's verdict, not a hoped-for one."""
        main(["RUN_TO_RUN", "cache", "64", "256", "2"])
        record = decode_scaling_run(drifting_harness.writer.documents(SCALING_RUN_BANNER))
        assert record["rungs"][0]["deterministic"] is False

    def test_raises_on_an_unknown_device(self, harness: Harness) -> None:
        """An absent card fails before the first rung."""
        with pytest.raises(ValueError, match="Invalid device identifier: cuda:7"):
            main(["--device", "cuda:7", "RUN_TO_RUN", "cache", "64", "256", "2"])
