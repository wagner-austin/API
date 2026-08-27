"""The attention-cost benchmark and its report.

The timings run for real on the CPU. What a CPU runner cannot establish is
the answer -- how much slower the math path is on a V100 -- and it does not
try; the shapes it times are the cheapest in the sweep and the assertions are
about the instrument, not about the hardware.

The out-of-memory path is exercised with a REAL
``torch.cuda.OutOfMemoryError`` raised by a test-supplied callable. That is
dependency injection, not a mock: the exception is torch's own, and the arm
under test is the production one.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
import torch
from platform_core.determinism_record import TRUE, determinism_record
from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.run_record import (
    Observation,
    RunRecord,
    decode_run_record,
    encode_run_record,
    run_record,
)
from platform_core.testing import sample_run_fingerprint
from torch.nn.attention import SDPBackend

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import sdpa_benchmark as bench_cli
from model_trainer.cli import sdpa_benchmark_report as report_cli
from model_trainer.core.services.model.sdpa_probe import forced_sdpa_output, sdpa_output
from model_trainer.core.services.model.sdpa_shapes import (
    COST_BATCHES,
    COST_LENGTHS,
    DEFAULT_KEY,
    FALSE_VALUE,
    FITTED_SUFFIX,
    PEAK_SUFFIX,
    SDPA_COST_EXPERIMENT,
    SECONDS_SUFFIX,
    SPREAD_SUFFIX,
    TRUE_VALUE,
    SdpaCostShape,
    cost_label,
    cost_prefix,
    cost_shapes,
)
from model_trainer.core.services.model.sdpa_timing import (
    cost_operands,
    measure_sdpa,
    time_sdpa,
)
from model_trainer.core.services.model.timing_harness import (
    NO_PEAK,
    MeasuredCost,
    backend_context,
    peak_reader,
    peak_resetter,
    timed_or_unfitted,
)

TINY = SdpaCostShape(name="test-tiny", batch=1, heads=2, head_dim=8, sequence_len=16)
PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})

#: One trivial call. The declared sweep ends at eight sequences of 4096
#: tokens, where the math path allocates gigabytes; timing that on a CPU
#: runner would not fail, it would hang.
CHEAP: tuple[SdpaCostShape, ...] = (TINY,)


def _cheap_sweep() -> Generator[None, None, None]:
    """Install the one-call sweep for the duration of one test.

    Written as ``pytest.fixture(impl)`` rather than with the decorator,
    matching ``tests/conftest.py``: the decorator form returns an overloaded
    function and trips this package's ``disallow_any_decorated``.

    Yields:
        Nothing; the sweep is installed for the body of the test.
    """
    cli_hooks.cost_shapes_hook = lambda: CHEAP
    try:
        yield
    finally:
        cli_hooks.cost_shapes_hook = cli_hooks._default_cost_shapes


cheap_sweep = pytest.fixture(_cheap_sweep)


class TestTheSweep:
    def test_it_walks_every_batch_against_every_length(self) -> None:
        grid = [s for s in cost_shapes() if s["name"].startswith("grid-")]

        assert len(grid) == len(COST_BATCHES) * len(COST_LENGTHS)

    def test_it_runs_to_four_thousand_tokens(self) -> None:
        # The math path's memory is quadratic in this axis, so the sweep has
        # to reach far enough for the wall to be inside the measured range
        # rather than past its edge.
        assert max(COST_LENGTHS) == 4096

    def test_it_covers_one_sequence_and_a_real_batch(self) -> None:
        assert COST_BATCHES == (1, 8)

    def test_it_also_prices_the_ladder_rungs_the_correctness_result_covers(self) -> None:
        rungs = [s for s in cost_shapes() if s["name"].startswith("rung-")]

        assert [s["name"] for s in rungs] == [
            "rung-tiny",
            "rung-small",
            "rung-medium",
            "rung-large",
            "rung-xl",
            "rung-tiny-len128",
            "rung-tiny-len256",
            "rung-tiny-len512",
        ]

    def test_no_two_shapes_share_a_label(self) -> None:
        labels = [cost_label(s, DEFAULT_KEY, SECONDS_SUFFIX) for s in cost_shapes()]

        assert len(set(labels)) == len(labels)


class TestTheOperands:
    def test_they_carry_the_layout_split_heads_produces(self) -> None:
        query, key, value = cost_operands(TINY, "cpu")

        assert tuple(query.shape) == (1, 2, 16, 8)
        assert not query.is_contiguous()
        assert not key.is_contiguous()
        assert not value.is_contiguous()

    def test_a_batch_of_eight_is_eight_sequences(self) -> None:
        batched = SdpaCostShape(name="b", batch=8, heads=2, head_dim=8, sequence_len=16)

        assert tuple(cost_operands(batched, "cpu")[0].shape) == (8, 2, 16, 8)


class TestThePeakCounters:
    def test_a_cuda_run_uses_torchs_own_counters(self) -> None:
        # Asserted by identity, the same trick `synchroniser` uses: it is
        # what makes the cuda arm checkable on a machine without a card.
        assert peak_resetter("cuda") is torch.cuda.reset_peak_memory_stats
        assert peak_reader("cuda") is torch.cuda.max_memory_allocated

    def test_a_cpu_run_reports_no_peak_rather_than_omitting_one(self) -> None:
        # A cpu record keeps the same shape as a cuda one; the allocator
        # being measured is CUDA's and a cpu run does not use it.
        peak_resetter("cpu")()

        assert peak_reader("cpu")() == NO_PEAK


class TestTheBackendContext:
    def test_forcing_a_backend_restricts_the_dispatcher(self) -> None:
        # The context is what makes the pinned arm pinned, so it has to
        # produce what forcing that backend per call produces -- otherwise
        # the benchmark would be timing a different kernel than the
        # selection probe measured.
        query, key, value = cost_operands(TINY, "cpu")
        per_call = forced_sdpa_output(query, key, value, SDPBackend.MATH)
        with backend_context(SDPBackend.MATH):
            under_context = sdpa_output(query, key, value)

        if per_call is None:
            raise AssertionError("forcing math on cpu must produce a result")
        assert torch.equal(per_call, under_context)

    def test_the_unforced_arm_is_wrapped_too(self) -> None:
        # Both arms enter a context so the enter-and-exit cost is paid on
        # both sides. An earlier revision forced the backend INSIDE the
        # timing loop, which put 27.8 us per call on the pinned arm alone --
        # 20% of the whole measurement at batch 1 and 64 tokens, right where
        # the ladder's rungs are.
        query, key, value = cost_operands(TINY, "cpu")

        with backend_context(None):
            plain = sdpa_output(query, key, value)

        assert torch.equal(plain, sdpa_output(query, key, value))

    def test_a_null_context_does_not_restrict_anything(self) -> None:
        # Under the null context the fused backends are still permitted, so
        # a cpu call that would refuse under a forced fused backend does not.
        query, key, value = cost_operands(TINY, "cpu")

        with backend_context(None):
            assert sdpa_output(query, key, value).shape == (1, 2, 16, 8)


class TestMeasuring:
    def test_it_returns_a_median_a_spread_and_a_peak(self) -> None:
        cost = measure_sdpa(TINY, "cpu", None)

        assert cost["seconds"] > 0.0
        assert cost["spread"] >= 0.0
        assert cost["peak_bytes"] == NO_PEAK

    def test_forcing_the_math_backend_still_measures(self) -> None:
        cost = measure_sdpa(TINY, "cpu", SDPBackend.MATH)

        assert cost["seconds"] > 0.0

    def test_time_sdpa_returns_the_cost_when_it_fits(self) -> None:
        cost = time_sdpa(TINY, "cpu", SDPBackend.MATH)

        if cost is None:
            raise AssertionError("a tiny cpu call must fit")
        assert cost["seconds"] > 0.0


class TestRunningOutOfMemory:
    def test_it_is_reported_as_a_result_not_raised(self) -> None:
        # A real torch exception from a test-supplied callable: the arm under
        # test is production code, and nothing is faked.
        def out_of_memory() -> MeasuredCost:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        assert timed_or_unfitted(out_of_memory) is None

    def test_a_measurement_that_fits_passes_through(self) -> None:
        wanted = MeasuredCost(seconds=1.0, spread=0.5, peak_bytes=2.0)

        def fits() -> MeasuredCost:
            return wanted

        assert timed_or_unfitted(fits) == wanted

    def test_any_other_failure_propagates(self) -> None:
        # An unrelated error recorded as "this did not fit" would be a false
        # fact about the card's memory.
        def broken() -> MeasuredCost:
            raise RuntimeError("something else entirely")

        with pytest.raises(RuntimeError, match="something else entirely"):
            timed_or_unfitted(broken)


class TestWhatOneRecordCarries:
    def test_a_fitted_call_reports_four_numbers(self) -> None:
        cost = MeasuredCost(seconds=1.5, spread=0.25, peak_bytes=1024.0)

        names = [o["name"] for o in bench_cli.cost_observations(cost_prefix(TINY), "math", cost)]

        assert names == [
            cost_label(TINY, "math", FITTED_SUFFIX),
            cost_label(TINY, "math", SECONDS_SUFFIX),
            cost_label(TINY, "math", SPREAD_SUFFIX),
            cost_label(TINY, "math", PEAK_SUFFIX),
        ]

    def test_a_call_that_did_not_fit_reports_only_that(self) -> None:
        observations = bench_cli.cost_observations(cost_prefix(TINY), "math", None)

        assert observations == (
            Observation(name=cost_label(TINY, "math", FITTED_SUFFIX), value=FALSE_VALUE),
        )

    def test_the_pinned_backend_is_the_one_the_probe_found_card_invariant(self) -> None:
        assert bench_cli.PINNED_KEY == "math"

    def test_the_production_hook_walks_the_whole_declared_sweep(self) -> None:
        # The tests install a one-call sweep; the cluster runs the real one,
        # and this is what keeps the two from drifting apart unnoticed.
        assert cli_hooks._default_cost_shapes() == cost_shapes()


def cost_record(
    gpu: str,
    *,
    base_seconds: float = 1.0,
    pinned_seconds: float = 4.0,
    base_peak: float = 100.0,
    pinned_peak: float = 3400.0,
    fits: bool = True,
    spread: float = 0.0,
) -> RunRecord:
    """Build a record whose every shape carries the same made-up cost."""
    observations: list[Observation] = []
    for shape in cost_shapes():
        observations.append(
            Observation(name=cost_label(shape, DEFAULT_KEY, FITTED_SUFFIX), value=TRUE_VALUE)
        )
        observations.append(
            Observation(name=cost_label(shape, DEFAULT_KEY, SECONDS_SUFFIX), value=base_seconds)
        )
        observations.append(
            Observation(name=cost_label(shape, DEFAULT_KEY, SPREAD_SUFFIX), value=spread)
        )
        observations.append(
            Observation(name=cost_label(shape, DEFAULT_KEY, PEAK_SUFFIX), value=base_peak)
        )
        observations.append(
            Observation(
                name=cost_label(shape, "math", FITTED_SUFFIX),
                value=TRUE_VALUE if fits else FALSE_VALUE,
            )
        )
        if fits:
            observations.append(
                Observation(name=cost_label(shape, "math", SECONDS_SUFFIX), value=pinned_seconds)
            )
            observations.append(
                Observation(name=cost_label(shape, "math", SPREAD_SUFFIX), value=spread)
            )
            observations.append(
                Observation(name=cost_label(shape, "math", PEAK_SUFFIX), value=pinned_peak)
            )
    return run_record(
        experiment=SDPA_COST_EXPERIMENT,
        label=bench_cli.SDPA_COST_LABEL,
        fingerprint=sample_run_fingerprint(
            image_digest="sha256:test",
            gpu_model=gpu,
            driver_version="580.82.07",
            determinism=PINNED,
        ),
        observations=tuple(observations),
        payload_digest="",
    )


class TestTheReport:
    def test_a_clean_pair_becomes_a_multiplier(self) -> None:
        values = {o["name"]: o["value"] for o in cost_record("A100")["observations"]}

        assert report_cli.slowdown(values, cost_prefix(cost_shapes()[0])) == "4.0x"

    def test_memory_growth_carries_the_pinned_peak(self) -> None:
        values = {o["name"]: o["value"] for o in cost_record("A100")["observations"]}

        assert report_cli.memory_growth(values, cost_prefix(cost_shapes()[0])) == "34.0x (0 MiB)"

    def test_not_fitting_is_named_rather_than_divided(self) -> None:
        values = {o["name"]: o["value"] for o in cost_record("V100", fits=False)["observations"]}

        assert report_cli.slowdown(values, cost_prefix(cost_shapes()[0])) == report_cli.DID_NOT_FIT
        assert (
            report_cli.memory_growth(values, cost_prefix(cost_shapes()[0]))
            == report_cli.DID_NOT_FIT
        )

    def test_a_noisy_measurement_is_not_divided(self) -> None:
        # The split-K benchmark learned this from a run with 54%, 85% and 90%
        # batch spreads: a median with an enormous spread must not be
        # compared with another one.
        values = {o["name"]: o["value"] for o in cost_record("A30", spread=0.9)["observations"]}

        assert report_cli.slowdown(values, cost_prefix(cost_shapes()[0])) == report_cli.NOISY

    def test_two_measurements_under_the_launch_floor_are_not_divided(self) -> None:
        values = {
            o["name"]: o["value"]
            for o in cost_record("A100", base_seconds=1e-6, pinned_seconds=9e-6)["observations"]
        }

        assert report_cli.slowdown(values, cost_prefix(cost_shapes()[0])) == report_cli.BELOW_FLOOR

    def test_the_floor_is_the_slowest_card_measured_on_this_cluster(self) -> None:
        # 136 us on the V100, the largest of the three, so "clears the floor"
        # means clears it everywhere rather than on the fastest card.
        assert report_cli.OVERHEAD_FLOOR == 136e-6

    def test_a_record_claiming_it_fitted_but_carrying_no_timing_says_so(self) -> None:
        # A fact about the record, not about the card. Reporting a truncated
        # record as a memory limit would put a defect in the file into a
        # table of hardware results.
        full = cost_record("A100")
        shape = cost_shapes()[0]
        wanted = cost_label(shape, "math", SECONDS_SUFFIX)
        values = {o["name"]: o["value"] for o in full["observations"] if o["name"] != wanted}

        assert report_cli.slowdown(values, cost_prefix(shape)) == report_cli.INCOMPLETE
        assert report_cli.INCOMPLETE != report_cli.DID_NOT_FIT

    def test_a_record_missing_the_shape_reads_as_not_fitted(self) -> None:
        empty = run_record(
            experiment=SDPA_COST_EXPERIMENT,
            label=bench_cli.SDPA_COST_LABEL,
            fingerprint=cost_record("A30")["fingerprint"],
            observations=(Observation(name="unrelated", value=1.0),),
            payload_digest="",
        )
        values = {o["name"]: o["value"] for o in empty["observations"]}

        assert not report_cli.fitted(values, cost_prefix(cost_shapes()[0]), "math")
        assert report_cli.slowdown(values, cost_prefix(cost_shapes()[0])) == report_cli.DID_NOT_FIT

    def test_an_unrecorded_base_peak_is_said_rather_than_divided_by_zero(self) -> None:
        values = {o["name"]: o["value"] for o in cost_record("A30", base_peak=0.0)["observations"]}

        assert report_cli.memory_growth(values, cost_prefix(cost_shapes()[0])) == "not recorded"

    def test_the_report_heads_each_run_with_its_card(self) -> None:
        lines = report_cli.report_lines((("v100.json", cost_record("Tesla V100")),))

        assert "Tesla V100" in lines[0]

    def test_every_shape_gets_a_row(self) -> None:
        lines = report_cli.report_lines((("a.json", cost_record("A100")),))
        rows = [line for line in lines if line.startswith("  grid-") or line.startswith("  rung-")]

        assert len(rows) == len(cost_shapes())


class TestTheCommandLines:
    def test_main_writes_a_record_that_decodes_back(
        self, tmp_path: pathlib.Path, cheap_sweep: None
    ) -> None:
        out = tmp_path / "records" / "cost.json"

        assert bench_cli.main(["--device", "cpu", "--out", str(out)]) == 0

        written = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert written["experiment"] == SDPA_COST_EXPERIMENT

    def test_an_absent_device_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--device"):
            bench_cli.main(["--out", str(tmp_path / "cost.json")])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path, cheap_sweep: None) -> None:
        with pytest.raises(ValueError, match="--backend"):
            bench_cli.main(
                ["--device", "cpu", "--out", str(tmp_path / "c.json"), "--backend", "math"]
            )

    def test_the_report_reads_a_directory(self, tmp_path: pathlib.Path) -> None:
        directory = tmp_path / "cost"
        directory.mkdir()
        (directory / "a100.json").write_text(
            dump_json_str(encode_run_record(cost_record("A100"))), encoding="utf-8"
        )

        assert report_cli.main(["--dir", str(directory)]) == 0

    def test_the_report_refuses_a_missing_directory(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="no such directory"):
            report_cli.main(["--dir", str(tmp_path / "absent")])

    def test_the_benchmark_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, cheap_sweep: None
    ) -> None:
        saved = sys.argv
        sys.argv = ["x", "--device", "cpu", "--out", str(tmp_path / "c.json")]
        try:
            with pytest.raises(SystemExit) as raised:
                bench_cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_the_report_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        directory = tmp_path / "cost"
        directory.mkdir()
        (directory / "a.json").write_text(
            dump_json_str(encode_run_record(cost_record("A100"))), encoding="utf-8"
        )
        saved = sys.argv
        sys.argv = ["x", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as raised:
                report_cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_running_the_benchmark_as_main_actually_measures(
        self, tmp_path: pathlib.Path, cheap_sweep: None
    ) -> None:
        out = tmp_path / "c.json"
        module_name = "model_trainer.cli.sdpa_benchmark"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", "--device", "cpu", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert out.is_file()

    def test_running_the_report_as_main_actually_reports(self, tmp_path: pathlib.Path) -> None:
        directory = tmp_path / "cost"
        directory.mkdir()
        (directory / "a.json").write_text(
            dump_json_str(encode_run_record(cost_record("A100"))), encoding="utf-8"
        )
        module_name = "model_trainer.cli.sdpa_benchmark_report"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
