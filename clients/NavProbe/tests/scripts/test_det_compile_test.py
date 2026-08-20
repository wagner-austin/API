"""Tests for the deterministic-mode compile gate."""

from __future__ import annotations

import pytest
from scripts.arguments import ScriptArgumentError
from scripts.det_compile_test import (
    CONSTRAINT_CAPACITY,
    DEFAULT_GATE_DEVICE,
    PERTURBATION,
    SCENE,
    WORLD_COUNT,
    main,
    parse_invocation,
)

from navprobe.codecs.compile_gate import COMPILE_GATE_BANNER, decode_compile_gate
from tests.scripts.conftest import CLOCK_STEP, DEVICES, Harness


class TestParseInvocation:
    """Tests for :func:`parse_invocation`."""

    def test_reads_the_mode_and_cache_directory(self) -> None:
        """A cold cache is what makes the gate's result mean anything."""
        parsed = parse_invocation(["RUN_TO_RUN", "fresh-cache"])
        assert (parsed.mode_name, parsed.cache_dir) == ("RUN_TO_RUN", "fresh-cache")

    def test_defaults_to_the_cpu_device(self) -> None:
        """Parse-time rejection needs no GPU, so the gate must not require one."""
        assert parse_invocation(["RUN_TO_RUN", "cache"]).device == DEFAULT_GATE_DEVICE

    def test_reads_an_explicit_device(self) -> None:
        """The same gate can be pointed at a card deliberately."""
        assert parse_invocation(["RUN_TO_RUN", "c", "--device", "cuda:0"]).device == "cuda:0"

    def test_rejects_a_missing_cache_directory(self) -> None:
        """Without a fresh cache the gate could report a cached success."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN"])
        assert caught.value.code == "NP-ARGS-006"

    def test_rejects_an_extra_positional(self) -> None:
        """A third positional is a typo, not an option."""
        with pytest.raises(ScriptArgumentError) as caught:
            parse_invocation(["RUN_TO_RUN", "cache", "extra"])
        assert caught.value.code == "NP-ARGS-006"


class TestCompileGate:
    """Tests for :func:`main` against the fake runtime."""

    def test_returns_zero_when_the_mode_compiles(self, harness: Harness) -> None:
        """A mode that compiles and steps is the record this gate writes."""
        assert main(["RUN_TO_RUN", "cache"]) == 0

    def test_passes_the_mode_and_a_zero_record_bound(self, harness: Harness) -> None:
        """The gate tests compilation, so it leaves the bound at Warp's own."""
        main(["GPU_TO_GPU", "fresh"])
        assert harness.init_warp.calls == [("GPU_TO_GPU", "fresh", 0)]

    def test_runs_on_the_cpu_device_by_default(self, harness: Harness) -> None:
        """The gate is a CPU test of a GPU property."""
        main(["RUN_TO_RUN", "cache"])
        assert harness.runtime.scopes_entered == [DEFAULT_GATE_DEVICE]

    def test_constructs_inside_the_device_scope(self, harness: Harness) -> None:
        """Codegen must happen under the device being tested."""
        main(["RUN_TO_RUN", "cache"])
        assert harness.runtime.work_inside_scope == ["construct:1"]

    def test_drives_the_measured_pipeline(self, harness: Harness) -> None:
        """The gate compiles the same adapter the measurements use.

        A gate that drove the vendor its own way could pass while the measured
        path still failed.
        """
        main(["RUN_TO_RUN", "cache"])
        assert harness.construct.calls == [
            (harness.construct.calls[0][0], WORLD_COUNT, PERTURBATION, CONSTRAINT_CAPACITY)
        ]

    def test_writes_a_decodable_record(self, harness: Harness) -> None:
        """The record is evidence the mode compiled, with its conditions."""
        main(["RUN_TO_RUN", "cache"])
        record = decode_compile_gate(harness.writer.documents(COMPILE_GATE_BANNER))
        assert (record["mode"], record["world_count"], record["scene"]) == (
            "RUN_TO_RUN",
            WORLD_COUNT,
            SCENE,
        )

    def test_records_the_resolved_device(self, harness: Harness) -> None:
        """A pass on one device is not a pass on another."""
        main(["RUN_TO_RUN", "cache", "--device", "cuda:0"])
        record = decode_compile_gate(harness.writer.documents(COMPILE_GATE_BANNER))
        assert record["device"] == DEVICES["cuda:0"]

    def test_records_the_wall_time(self, harness: Harness) -> None:
        """Cold codegen time is the figure that says the cache was cold."""
        main(["RUN_TO_RUN", "cache"])
        record = decode_compile_gate(harness.writer.documents(COMPILE_GATE_BANNER))
        assert record["wall_seconds"] == CLOCK_STEP

    def test_raises_on_an_unknown_device(self, harness: Harness) -> None:
        """A device name that does not exist fails before compiling."""
        with pytest.raises(ValueError, match="Invalid device identifier: cuda:9"):
            main(["RUN_TO_RUN", "cache", "--device", "cuda:9"])

    def test_writes_no_record_when_the_device_is_unknown(self, harness: Harness) -> None:
        """The absence of the record is the gate's other answer."""
        with pytest.raises(ValueError, match="Invalid device identifier"):
            main(["RUN_TO_RUN", "cache", "--device", "cuda:9"])
        assert harness.writer.chunks == []
