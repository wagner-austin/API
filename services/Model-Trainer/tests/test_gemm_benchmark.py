"""The split-K cost benchmark, exercised on the CPU.

The timings themselves are not asserted -- a wall clock on a shared machine is
not a value a test can pin. What IS asserted is everything that decides whether
the numbers mean anything: that the two conditions are named apart, that the
child is spawned with the variable actually set, and that a child which came
back having measured the WRONG condition is refused rather than averaged in.

That last one is the case worth having. `CUBLASLT_WORKSPACE_SIZE` is read once
per process, so a child that somehow ran the default would produce timings
identical to the parent's and the benchmark would report the fix as free --
the most damaging way this could be wrong.
"""

from __future__ import annotations

import pathlib

import pytest
import torch
from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    decode_run_record,
    encode_run_record,
    run_record,
)

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import gemm_benchmark as bench
from model_trainer.core.run_fingerprint import capture_run_fingerprint
from model_trainer.core.services.model.gemm_shapes import (
    BATCH_COLS,
    GEMM_BATCHED,
    GEMM_COLS,
    GEMM_SHAPES,
    GemmShape,
    probed_shapes,
    timed_shapes,
)
from model_trainer.core.services.model.gemm_timing import (
    BATCHES,
    INNER,
    WARMUP,
    synchroniser,
    time_gemm,
)

SMALL: GemmShape = {"rows": 8, "inner": 16, "cols": 4, "origin": "test"}

#: One shape. Timing is deliberately repetitive -- warmup plus several
#: batches of many calls -- so walking the real 43-shape table per test
#: spent minutes measuring a laptop nobody will read the numbers from.
ONE: tuple[tuple[str, GemmShape], ...] = (("only", SMALL),)


class TestTiming:
    def test_it_reports_a_positive_time_per_call(self) -> None:
        seconds, spread = time_gemm(SMALL, "cpu")

        assert seconds > 0.0
        assert spread >= 0.0

    def test_the_spread_is_the_range_across_batches(self) -> None:
        # Reported rather than discarded: a median with an enormous spread is
        # a number that should not be compared with another one, and only the
        # caller can see both.
        _, spread = time_gemm(SMALL, "cpu")

        assert spread >= 0.0

    def test_it_amortises_launch_overhead_over_a_batch(self) -> None:
        # A single call at the small shapes is dominated by launch overhead,
        # which is identical under both conditions and would swamp the kernel
        # difference this exists to measure.
        assert INNER > 1

    def test_it_discards_warmup_calls(self) -> None:
        # The first call on a shape pays for kernel selection and lazy module
        # loading -- a one-time cost that a per-call time must not carry.
        assert WARMUP > 0

    def test_it_takes_more_than_one_batch_so_a_median_exists(self) -> None:
        assert BATCHES >= 3

    def test_a_cuda_device_waits_on_the_real_cuda_barrier(self) -> None:
        # Asserted by identity, which is how this arm is reachable at all on a
        # machine with no GPU. A branch inside the timing loop would have left
        # it uncovered on every machine that runs the suite.
        assert synchroniser("cuda") is torch.cuda.synchronize

    def test_a_cpu_device_does_not_wait(self) -> None:
        # A CPU has already finished when the call returns, and calling the
        # cuda barrier there would raise.
        assert synchroniser("cpu")() is None
        assert synchroniser("cpu") is not torch.cuda.synchronize


class TestConditionNaming:
    def test_the_two_conditions_are_distinct(self) -> None:
        assert bench.DEFAULT_CONDITION != bench.NOSPLITK_CONDITION

    def test_observations_are_suffixed_with_their_condition(self) -> None:
        observations = bench.timing_observations("cpu", bench.NOSPLITK_CONDITION, ONE)

        assert all(o["name"].endswith(bench.NOSPLITK_CONDITION) for o in observations)

    def test_every_shape_gets_a_time_and_a_spread(self) -> None:
        observations = bench.timing_observations("cpu", bench.DEFAULT_CONDITION, ONE)

        assert len(observations) == 2 * len(ONE)

    def test_the_two_conditions_do_not_collide_in_the_record(self) -> None:
        # If they did, one would overwrite the other and the comparison would
        # silently be a run against itself.
        a = {o["name"] for o in bench.timing_observations("cpu", bench.DEFAULT_CONDITION, ONE)}
        b = {o["name"] for o in bench.timing_observations("cpu", bench.NOSPLITK_CONDITION, ONE)}

        assert a & b == set()


def _child_record(condition: str, path: pathlib.Path) -> None:
    """Write a record such as the child would produce."""
    record = run_record(
        experiment=bench.BENCHMARK_EXPERIMENT,
        label=bench.BENCHMARK_LABEL,
        fingerprint=capture_run_fingerprint("cpu", cli_hooks.apply_determinism_hook()),
        observations=(Observation(name=f"gemm-x-M8-K16-N4|seconds|{condition}", value=1.0),),
        payload_digest=NO_PAYLOAD,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")


class TestTheChild:
    def test_it_is_spawned_with_the_workspace_variable_set(self, tmp_path: pathlib.Path) -> None:
        # THE thing that makes the second condition a second condition. The
        # variable is read once per process, so if it is missing here the
        # child measures the default and the fix reads as free.
        seen: dict[str, str] = {}

        def _spawn(argv: list[str], variable: str, value: str, /) -> int:
            seen[variable] = value
            _child_record(bench.NOSPLITK_CONDITION, pathlib.Path(argv[argv.index("--out") + 1]))
            return 0

        cli_hooks.run_benchmark_child = _spawn
        try:
            bench.run_child("cpu", tmp_path / "child.json")
        finally:
            cli_hooks.run_benchmark_child = cli_hooks._default_run_benchmark_child

        assert seen[bench.WORKSPACE_VAR] == bench.NO_SPLIT_K

    def test_it_is_asked_for_the_nosplitk_condition_by_name(self, tmp_path: pathlib.Path) -> None:
        argv_seen: list[str] = []

        def _spawn(argv: list[str], variable: str, value: str, /) -> int:
            argv_seen.extend(argv)
            _child_record(bench.NOSPLITK_CONDITION, pathlib.Path(argv[argv.index("--out") + 1]))
            return 0

        cli_hooks.run_benchmark_child = _spawn
        try:
            bench.run_child("cpu", tmp_path / "child.json")
        finally:
            cli_hooks.run_benchmark_child = cli_hooks._default_run_benchmark_child

        assert bench.NOSPLITK_CONDITION in argv_seen
        assert argv_seen[argv_seen.index("--condition") + 1] == bench.NOSPLITK_CONDITION

    def test_a_failing_child_is_refused(self, tmp_path: pathlib.Path) -> None:
        cli_hooks.run_benchmark_child = lambda argv, variable, value: 3
        try:
            with pytest.raises(RuntimeError, match="child exited 3"):
                bench.run_child("cpu", tmp_path / "child.json")
        finally:
            cli_hooks.run_benchmark_child = cli_hooks._default_run_benchmark_child

    def test_a_child_that_measured_the_wrong_condition_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The damaging failure: a child that ran the default would produce
        # timings matching the parent's, and the benchmark would report the
        # fix as costing nothing.
        def _spawn(argv: list[str], variable: str, value: str, /) -> int:
            _child_record(bench.DEFAULT_CONDITION, pathlib.Path(argv[argv.index("--out") + 1]))
            return 0

        cli_hooks.run_benchmark_child = _spawn
        try:
            with pytest.raises(RuntimeError, match="another condition"):
                bench.run_child("cpu", tmp_path / "child.json")
        finally:
            cli_hooks.run_benchmark_child = cli_hooks._default_run_benchmark_child

    def test_the_production_hook_runs_a_real_subprocess(self) -> None:
        # What the fake would otherwise hide: that the deployed path spawns
        # something at all.
        assert cli_hooks.run_benchmark_child is cli_hooks._default_run_benchmark_child

    def test_the_production_child_reports_an_exit_code(self, tmp_path: pathlib.Path) -> None:
        import sys

        code = cli_hooks._default_run_benchmark_child(
            [sys.executable, "-c", "raise SystemExit(7)"], "GEMM_BENCH_PROBE", "1"
        )

        assert code == 7


def _one_shape() -> tuple[tuple[str, GemmShape], ...]:
    """The shape table the CLI tests install."""
    return ONE


class TestTheCommandLine:
    def test_a_condition_run_writes_only_that_condition(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "one.json"

        cli_hooks.benchmark_shapes = _one_shape
        try:
            assert (
                bench.main(
                    ["--device", "cpu", "--out", str(out), "--condition", bench.NOSPLITK_CONDITION]
                )
                == 0
            )
        finally:
            cli_hooks.benchmark_shapes = cli_hooks._default_benchmark_shapes

        decoded = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert all(o["name"].endswith(bench.NOSPLITK_CONDITION) for o in decoded["observations"])

    def test_the_parent_records_both_conditions(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "both.json"

        def _spawn(argv: list[str], variable: str, value: str, /) -> int:
            _child_record(bench.NOSPLITK_CONDITION, pathlib.Path(argv[argv.index("--out") + 1]))
            return 0

        cli_hooks.run_benchmark_child = _spawn
        cli_hooks.benchmark_shapes = _one_shape
        try:
            assert bench.main(["--device", "cpu", "--out", str(out)]) == 0
        finally:
            cli_hooks.run_benchmark_child = cli_hooks._default_run_benchmark_child
            cli_hooks.benchmark_shapes = cli_hooks._default_benchmark_shapes

        decoded = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        names = [o["name"] for o in decoded["observations"]]
        assert any(n.endswith(bench.DEFAULT_CONDITION) for n in names)
        assert any(n.endswith(bench.NOSPLITK_CONDITION) for n in names)

    def test_an_absent_device_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--device"):
            bench.main(["--out", str(tmp_path / "x.json")])

    def test_an_absent_out_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            bench.main(["--device", "cpu"])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--repeats"):
            bench.main(["--device", "cpu", "--out", str(tmp_path / "x.json"), "--repeats", "5"])

    def test_the_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        import sys

        saved = sys.argv
        cli_hooks.benchmark_shapes = _one_shape
        sys.argv = [
            "modeltrainer-gemm-benchmark",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "e.json"),
            "--condition",
            bench.DEFAULT_CONDITION,
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                bench.entrypoint()
        finally:
            sys.argv = saved
            cli_hooks.benchmark_shapes = cli_hooks._default_benchmark_shapes

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_benchmarks(self, tmp_path: pathlib.Path) -> None:
        import runpy
        import sys

        out = tmp_path / "m.json"
        module_name = "model_trainer.cli.gemm_benchmark"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        # The one-shape table matters here as much as in the other CLI tests,
        # and this test is where forgetting it hurts most: `runpy` runs the
        # module in-process, so the real table would time an 84-GFLOP batched
        # matmul on the test runner's CPU and hang the suite rather than fail
        # it. It did, once.
        cli_hooks.benchmark_shapes = _one_shape
        sys.argv = [
            "modeltrainer-gemm-benchmark",
            "--device",
            "cpu",
            "--out",
            str(out),
            "--condition",
            bench.DEFAULT_CONDITION,
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            cli_hooks.benchmark_shapes = cli_hooks._default_benchmark_shapes
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0
        assert out.is_file()


class TestTheProductionShapeTable:
    def test_the_deployed_benchmark_times_every_probed_shape(self) -> None:
        # What the one-shape fixture would otherwise hide: that the deployed
        # command measures the real table, not the cheap stand-in.
        assert cli_hooks._default_benchmark_shapes() == timed_shapes()


class TestTheBatchedTable:
    """Timing at one 64-token sequence could not resolve most shapes.

    Every call carries ~100 microseconds of dispatch and launch, and at one
    short sequence the arithmetic is a few microseconds -- so the first
    benchmark reported the same time for a 128x128 matmul as for a
    1600x6400 one. Those shapes are not unaffected: split-K is SELECTED on
    seven of the eight, measured from the traces. They were unmeasured.
    """

    def test_it_mirrors_every_ladder_shape(self) -> None:
        assert len(GEMM_BATCHED) == len(GEMM_SHAPES)

    def test_it_changes_only_the_batch_dimension(self) -> None:
        # The point is to time the SAME matmul with more work per call. A
        # batched twin that also moved M or K would be a different call.
        by_dims = {(s["rows"], s["inner"]) for _, s in GEMM_BATCHED}

        assert by_dims == {(s["rows"], s["inner"]) for s in GEMM_SHAPES.values()}

    def test_every_batched_shape_uses_the_batch_dimension(self) -> None:
        assert [s["cols"] for _, s in GEMM_BATCHED] == [BATCH_COLS for _ in GEMM_BATCHED]

    def test_the_batch_is_far_larger_than_one_short_sequence(self) -> None:
        # It has to buy enough arithmetic to clear the launch-overhead floor;
        # a modest bump would have left the measurement where it was.
        assert BATCH_COLS >= 16 * GEMM_COLS

    def test_the_twins_are_named_apart_from_their_originals(self) -> None:
        # So a record read on its own tells them apart without consulting
        # dimensions.
        assert set(dict(GEMM_BATCHED)) & set(GEMM_SHAPES) == set()

    def test_the_benchmark_times_both_regimes(self) -> None:
        assert len(timed_shapes()) == len(GEMM_SHAPES) + len(GEMM_BATCHED)

    def test_the_sweep_grid_is_not_timed(self) -> None:
        # The sweep answers a question about VALUES, not speed; timing
        # thirty-five more shapes would cost minutes and say nothing.
        assert len(timed_shapes()) < len(probed_shapes())
