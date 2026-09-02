"""Tests for the collision-pair probe.

The probe's whole reason to exist is that a verdict alone is not enough, so the
test that matters most here is :class:`TestInertSceneIsVisible`: a run whose
repetitions agree perfectly while producing no contacts must be legible as
such, in the streamed line and in the record. Every other test guards the
plumbing around it.
"""

from __future__ import annotations

import pytest
from scripts.arguments import DEFAULT_DEVICE, ScriptArgumentError
from scripts.collision_pair_probe import (
    CONSTRAINT_CAPACITY,
    PERTURBATION,
    TRIAL,
    WORLD_COUNT,
    main,
    measure_witness,
    parse_invocation,
    progress_line,
)

from navprobe.codecs.contact_witness import CONTACT_WITNESS_BANNER, decode_contact_witness_run
from navprobe.collision_pairs import COLLISION_PAIRS
from navprobe.records import (
    ContactWitnessEntry,
    ContactWitnessRunRecord,
    TrialRecord,
    TrialSpec,
)
from tests.scripts.conftest import DEVICES, LIVE_CONTACTS_PER_STEP, WitnessHarness


def _decoded(harness: WitnessHarness) -> ContactWitnessRunRecord:
    """Decode the record a completed run wrote.

    Args:
        harness: The harness the run used.

    Returns:
        The decoded record.
    """
    return decode_contact_witness_run(harness.writer.chunks[-1])


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
        """A command line without the flag still selects the first card."""
        assert parse_invocation(["RUN_TO_RUN", "cache"]).device == DEFAULT_DEVICE

    def test_reads_the_record_bound(self) -> None:
        """The bound the convex-narrowphase runs used is accepted."""
        assert parse_invocation(["RUN_TO_RUN", "cache", "4096"]).max_records == 4096

    def test_reads_the_device_flag(self) -> None:
        """A second card is addressable."""
        parsed = parse_invocation(["RUN_TO_RUN", "cache", "--device", "cuda:1"])
        assert parsed.device == "cuda:1"

    def test_reads_the_linesearch_block_dim_flag(self) -> None:
        """The one setting that decides a coupled-body verdict is pinnable."""
        parsed = parse_invocation(["RUN_TO_RUN", "cache", "--linesearch-block-dim", "64"])
        assert parsed.linesearch_block_dim == 64

    def test_defaults_the_linesearch_block_dim_to_the_vendor_s(self) -> None:
        """Absent the flag the script imposes nothing."""
        assert parse_invocation(["RUN_TO_RUN", "cache"]).linesearch_block_dim is None

    def test_rejects_too_few_positionals(self) -> None:
        """A missing cache directory would silently share a warm cache."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN"])
        assert caught.value.code == "NP-ARGS-008"

    def test_rejects_too_many_positionals(self) -> None:
        """A fourth positional is a typo, not an option."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN", "cache", "4096", "extra"])
        assert caught.value.code == "NP-ARGS-008"

    def test_rejects_a_non_numeric_record_bound(self) -> None:
        """A mistyped bound stops before a cold compile."""
        with pytest.raises(ScriptArgumentError):
            parse_invocation(["RUN_TO_RUN", "cache", "many"])


class TestMeasureWitness:
    """Tests for :func:`measure_witness`."""

    def test_sums_contacts_over_every_step(self, witness_harness: WitnessHarness) -> None:
        """The total is per-step contacts times the step count."""
        factory = witness_harness.construct("xml", WORLD_COUNT, PERTURBATION, CONSTRAINT_CAPACITY)
        total, _ = measure_witness(factory, TRIAL)
        assert total == LIVE_CONTACTS_PER_STEP * TRIAL["step_count"]

    def test_counts_no_zero_steps_when_the_scene_is_live(
        self, witness_harness: WitnessHarness
    ) -> None:
        """A scene in contact throughout reports no idle steps."""
        factory = witness_harness.construct("xml", WORLD_COUNT, PERTURBATION, CONSTRAINT_CAPACITY)
        _, zero_steps = measure_witness(factory, TRIAL)
        assert zero_steps == 0

    def test_counts_every_step_when_the_scene_is_inert(self, inert_harness: WitnessHarness) -> None:
        """A scene that never touches reports every step as idle."""
        factory = inert_harness.construct("xml", WORLD_COUNT, PERTURBATION, CONSTRAINT_CAPACITY)
        total, zero_steps = measure_witness(factory, TRIAL)
        assert (total, zero_steps) == (0, TRIAL["step_count"])


class TestProgressLine:
    """What a run streams while it is running."""

    def test_carries_the_contact_total_beside_the_verdict(self) -> None:
        """A line showing only the verdict would read as a pass."""
        entry = ContactWitnessEntry(
            pair="box_box",
            trial=TrialRecord(
                spec=TrialSpec(seed=7, step_count=40, repetitions=4),
                world_count=2,
                reference_digest="a" * 64,
                deterministic=True,
                first_divergent_step=None,
            ),
            contact_total=0,
            zero_contact_steps=40,
        )
        line = progress_line(entry, 1.5)
        assert "deterministic=True" in line
        assert "contacts=0" in line

    def test_ends_with_a_newline(self) -> None:
        """Streamed lines are written unbuffered and must terminate."""
        entry = ContactWitnessEntry(
            pair="box_box",
            trial=TrialRecord(
                spec=TrialSpec(seed=7, step_count=40, repetitions=4),
                world_count=2,
                reference_digest="a" * 64,
                deterministic=True,
                first_divergent_step=None,
            ),
            contact_total=3,
            zero_contact_steps=0,
        )
        assert progress_line(entry, 1.5).endswith("\n")


class TestRun:
    """A complete run over the pair family."""

    def test_returns_zero(self, witness_harness: WitnessHarness) -> None:
        """A completed run reports success."""
        assert main(["RUN_TO_RUN", "cache", "4096"]) == 0

    def test_opts_out_of_power_throttling(self, witness_harness: WitnessHarness) -> None:
        """Wall time is an output, so the throttle is lifted first."""
        main(["RUN_TO_RUN", "cache", "4096"])
        assert witness_harness.opt_out.calls == 1

    def test_passes_the_configuration_to_warp(self, witness_harness: WitnessHarness) -> None:
        """Mode, cache and record bound reach the initialiser."""
        main(["RUN_TO_RUN", "cache", "4096"])
        assert witness_harness.init_warp.calls == [("RUN_TO_RUN", "cache", 4096)]

    def test_sweeps_every_pair_in_the_family(self, witness_harness: WitnessHarness) -> None:
        """One entry per pair, in the family's declared order."""
        main(["RUN_TO_RUN", "cache", "4096"])
        assert tuple(e["pair"] for e in _decoded(witness_harness)["entries"]) == COLLISION_PAIRS

    def test_constructs_every_factory_inside_the_device_scope(
        self, witness_harness: WitnessHarness
    ) -> None:
        """Building a model outside the scope would target the wrong card."""
        main(["RUN_TO_RUN", "cache", "4096"])
        assert witness_harness.runtime.work_inside_scope == ["construct:1" for _ in COLLISION_PAIRS]

    def test_passes_the_perturbation_and_capacity_through(
        self, witness_harness: WitnessHarness
    ) -> None:
        """Defaulting either one silently changes what was measured."""
        main(["RUN_TO_RUN", "cache", "4096"])
        call = witness_harness.construct.calls[0]
        assert (call[1], call[2], call[3]) == (WORLD_COUNT, PERTURBATION, CONSTRAINT_CAPACITY)

    def test_records_the_resolved_device_not_the_requested_one(
        self, witness_harness: WitnessHarness
    ) -> None:
        """The report names the card that ran, not the name that was typed."""
        main(["RUN_TO_RUN", "cache", "4096", "--device", "cuda:0"])
        record = _decoded(witness_harness)
        assert (record["device"], record["device_request"]) == (DEVICES["cuda:0"], "cuda:0")

    def test_writes_a_decodable_record(self, witness_harness: WitnessHarness) -> None:
        """The document round-trips through its own codec."""
        main(["RUN_TO_RUN", "cache", "4096"])
        assert witness_harness.writer.chunks[-1].startswith(CONTACT_WITNESS_BANNER)

    def test_streams_one_line_per_pair(self, witness_harness: WitnessHarness) -> None:
        """A long run stays legible while it runs."""
        main(["RUN_TO_RUN", "cache", "4096"])
        streamed = [c for c in witness_harness.writer.chunks if c.startswith("pair=")]
        assert len(streamed) == len(COLLISION_PAIRS)

    def test_a_live_scene_records_its_contacts(self, witness_harness: WitnessHarness) -> None:
        """The witness is carried, not dropped between measurement and record."""
        main(["RUN_TO_RUN", "cache", "4096"])
        entry = _decoded(witness_harness)["entries"][0]
        assert entry["contact_total"] == LIVE_CONTACTS_PER_STEP * TRIAL["step_count"]


class TestInertSceneIsVisible:
    """The case the probe exists for.

    Repetitions agree bit for bit and the verdict is ``deterministic: true``,
    while no contact is ever generated. That is what MuJoCo-Warp's convex
    narrowphase does under a deterministic mode, and a report that carried only
    the verdict would present it as a clean result.
    """

    def test_the_verdict_still_says_deterministic(self, inert_harness: WitnessHarness) -> None:
        """The verdict is not wrong -- it is answering a different question."""
        main(["RUN_TO_RUN", "cache", "4096"])
        assert all(e["trial"]["deterministic"] for e in _decoded(inert_harness)["entries"])

    def test_the_witness_reports_zero_contacts(self, inert_harness: WitnessHarness) -> None:
        """And the witness is what says the verdict meant nothing."""
        main(["RUN_TO_RUN", "cache", "4096"])
        assert all(e["contact_total"] == 0 for e in _decoded(inert_harness)["entries"])

    def test_every_step_is_recorded_as_idle(self, inert_harness: WitnessHarness) -> None:
        """The idle count distinguishes 'landed late' from 'never touched'."""
        main(["RUN_TO_RUN", "cache", "4096"])
        assert all(
            e["zero_contact_steps"] == TRIAL["step_count"]
            for e in _decoded(inert_harness)["entries"]
        )

    def test_the_streamed_line_shows_both(self, inert_harness: WitnessHarness) -> None:
        """The failure is legible while the run is still going."""
        main(["RUN_TO_RUN", "cache", "4096"])
        streamed = [c for c in inert_harness.writer.chunks if c.startswith("pair=")]
        assert "deterministic=True contacts=0" in streamed[0]
