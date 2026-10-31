"""Tests for the ten-scene GPU determinism sweep."""

from __future__ import annotations

import pytest
from scripts.arguments import DEFAULT_DEVICE, ScriptArgumentError
from scripts.gpu_deterministic_sweep import (
    CONSTRAINT_CAPACITY,
    PERTURBATION,
    SCENES,
    WORLD_COUNT,
    main,
    parse_invocation,
)

from navprobe.codecs.sweep_run import SWEEP_RUN_BANNER, decode_sweep_run
from tests.scripts.conftest import DEVICES, Harness


class TestParseInvocation:
    """Tests for :func:`parse_invocation`."""

    def test_reads_the_two_required_positionals(self) -> None:
        """Mode and cache directory are the minimum command line."""
        parsed = parse_invocation(["RUN_TO_RUN", "cache"])
        assert (parsed.mode_name, parsed.cache_dir) == ("RUN_TO_RUN", "cache")

    def test_defaults_the_record_bound_to_zero(self) -> None:
        """Zero leaves Warp's own code-generated bound in place."""
        assert parse_invocation(["RUN_TO_RUN", "cache"]).max_records == 0

    def test_defaults_the_device(self) -> None:
        """The pre-flag command line still selects the first card."""
        assert parse_invocation(["RUN_TO_RUN", "cache"]).device == DEFAULT_DEVICE

    def test_reads_the_record_bound(self) -> None:
        """The bound that cleared the 32-body overflow is accepted."""
        assert parse_invocation(["RUN_TO_RUN", "cache", "64"]).max_records == 64

    def test_reads_the_device_flag(self) -> None:
        """A second card is addressable."""
        assert parse_invocation(["RUN_TO_RUN", "cache", "64", "--device", "cuda:1"]).device == (
            "cuda:1"
        )

    def test_rejects_too_few_positionals(self) -> None:
        """A missing cache directory would silently share a warm cache."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN"])
        assert caught.value.code == "NP-ARGS-001"

    def test_rejects_too_many_positionals(self) -> None:
        """A fourth positional is a typo, not an option."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN", "cache", "64", "extra"])
        assert caught.value.code == "NP-ARGS-001"

    def test_rejects_a_non_numeric_record_bound(self) -> None:
        """A mistyped bound stops before a cold compile."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN", "cache", "sixty-four"])
        assert caught.value.code == "NP-ARGS-003"


class TestProgressLine:
    """Tests for :func:`progress_line`."""

    def test_reports_the_verdict_and_the_divergence_point(self, harness: Harness) -> None:
        """A streamed line carries what the operator is watching for."""
        main(["RUN_TO_RUN", "cache"])
        first = harness.writer.chunks[0]
        assert first.startswith("scene bodies=2 spacing=0.07 deterministic=True first_div=None")

    def test_streams_one_line_per_scene(self, harness: Harness) -> None:
        """A sweep that runs for an hour must be legible while it runs."""
        main(["RUN_TO_RUN", "cache"])
        streamed = [chunk for chunk in harness.writer.chunks if chunk.startswith("scene ")]
        assert len(streamed) == len(SCENES)

    def test_ends_every_streamed_line_with_a_newline(self, harness: Harness) -> None:
        """Lines are written whole rather than accumulating on one row."""
        main(["RUN_TO_RUN", "cache"])
        streamed = [chunk for chunk in harness.writer.chunks if chunk.startswith("scene ")]
        assert [chunk[-1] for chunk in streamed] == ["\n"] * len(SCENES)


class TestSweepRun:
    """Tests for :func:`main` against the fake runtime."""

    def test_returns_zero(self, harness: Harness) -> None:
        """A completed sweep exits clean."""
        assert main(["RUN_TO_RUN", "cache"]) == 0

    def test_passes_the_configuration_to_warp(self, harness: Harness) -> None:
        """The mode, cache directory and record bound reach the initialiser."""
        main(["RUN_TO_RUN", "cache-dir", "64"])
        assert harness.init_warp.calls == [("RUN_TO_RUN", "cache-dir", 64)]

    def test_sweeps_every_scene_in_the_family(self, harness: Harness) -> None:
        """All ten scenes run; a truncated family would be a quieter result."""
        main(["RUN_TO_RUN", "cache"])
        assert len(decode_sweep_run(harness.writer.documents(SWEEP_RUN_BANNER))["entries"]) == len(
            SCENES
        )

    def test_constructs_every_factory_inside_the_device_scope(self, harness: Harness) -> None:
        """Work outside the scope would run on whatever device was current."""
        main(["RUN_TO_RUN", "cache"])
        assert harness.runtime.work_inside_scope == ["construct:1"] * len(SCENES)

    def test_scopes_to_the_requested_device(self, harness: Harness) -> None:
        """The sweep enters the scope once, for the card that was asked for."""
        main(["RUN_TO_RUN", "cache", "--device", "cuda:1"])
        assert harness.runtime.scopes_entered == ["cuda:1"]

    def test_passes_the_perturbation_and_capacity_through(self, harness: Harness) -> None:
        """Defaulting either would measure a silently truncated solve."""
        main(["RUN_TO_RUN", "cache"])
        assert [(call[2], call[3]) for call in harness.construct.calls] == [
            (PERTURBATION, CONSTRAINT_CAPACITY)
        ] * len(SCENES)

    def test_records_the_resolved_device_not_the_requested_one(self, harness: Harness) -> None:
        """A record must name the card that ran, not the name that was typed."""
        main(["RUN_TO_RUN", "cache", "--device", "cuda:1"])
        record = decode_sweep_run(harness.writer.documents(SWEEP_RUN_BANNER))
        assert (record["device"], record["device_request"]) == (DEVICES["cuda:1"], "cuda:1")

    def test_writes_a_decodable_record(self, harness: Harness) -> None:
        """The output is a record another run can be compared against."""
        main(["RUN_TO_RUN", "cache", "64"])
        record = decode_sweep_run(harness.writer.documents(SWEEP_RUN_BANNER))
        assert (record["mode"], record["max_records"], record["world_count"]) == (
            "RUN_TO_RUN",
            64,
            WORLD_COUNT,
        )

    def test_reports_determinism_when_the_rollouts_agree(self, harness: Harness) -> None:
        """The positive control: agreeing simulators must read as deterministic."""
        main(["RUN_TO_RUN", "cache"])
        record = decode_sweep_run(harness.writer.documents(SWEEP_RUN_BANNER))
        assert all(entry["trial"]["deterministic"] for entry in record["entries"])

    def test_reports_divergence_when_the_rollouts_disagree(self, drifting_harness: Harness) -> None:
        """The negative control: the script reports what the instrument found."""
        main(["RUN_TO_RUN", "cache"])
        record = decode_sweep_run(drifting_harness.writer.documents(SWEEP_RUN_BANNER))
        assert [entry["trial"]["deterministic"] for entry in record["entries"]] == [False] * len(
            SCENES
        )

    def test_carries_the_scene_each_verdict_belongs_to(self, harness: Harness) -> None:
        """A verdict without its scene cannot be compared across machines."""
        main(["RUN_TO_RUN", "cache"])
        record = decode_sweep_run(harness.writer.documents(SWEEP_RUN_BANNER))
        assert tuple(entry["scene"] for entry in record["entries"]) == SCENES

    def test_raises_on_an_unknown_device(self, harness: Harness) -> None:
        """An absent card fails before a cold compile rather than after it."""
        with pytest.raises(ValueError, match="Invalid device identifier: cuda:7"):
            main(["RUN_TO_RUN", "cache", "--device", "cuda:7"])

    def test_does_not_enter_a_scope_when_the_device_is_unknown(self, harness: Harness) -> None:
        """Resolution is eager, so nothing is compiled for a bad name."""
        with pytest.raises(ValueError, match="Invalid device identifier"):
            main(["RUN_TO_RUN", "cache", "--device", "cuda:7"])
        assert harness.runtime.scopes_entered == []
