"""The training-step benchmark and its report.

The steps run for real on the CPU at a model small enough for a test runner:
a real forward, a real backward and a real AdamW update. What a CPU runner
cannot establish is the answer -- what the attention pin costs a training
step on a V100 -- and it does not try.

What it CAN establish is that the step is the trainer's step. The assertions
below check that gradients actually flow, that the optimizer actually moves
the weights, that the model is in training posture rather than eval, and that
the sweep is sized for the memory a step needs rather than the memory a
forward needs.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
import torch
from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import TRUE, determinism_record
from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    decode_run_record,
    encode_run_record,
    run_record,
)
from torch.nn.attention import SDPBackend

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import sdpa_benchmark_report as shared_report
from model_trainer.cli import train_benchmark as bench_cli
from model_trainer.cli import train_benchmark_report as report_cli
from model_trainer.core.services.model.cost_labels import (
    DEFAULT_KEY,
    FITTED_SUFFIX,
    PEAK_SUFFIX,
    SECONDS_SUFFIX,
    SPREAD_SUFFIX,
    TRUE_VALUE,
    labelled,
)
from model_trainer.core.services.model.forward_cost import GPT2_VOCAB, ForwardCostShape
from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES
from model_trainer.core.services.model.train_cost import (
    TRAIN_BATCHES,
    TRAIN_CLIP,
    TRAIN_INNER,
    TRAIN_LR,
    TRAIN_OPTIMIZER,
    TRAIN_SHAPES,
    TRAIN_WARMUP,
    measure_train_step,
    run_train_step,
    time_train_step,
    train_step_setup,
)

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})

#: A model a test runner can step in a moment. The declared rows need about
#: twelve gigabytes of parameters, gradients and AdamW moments.
CHEAP_SHAPE = ForwardCostShape(
    name="test-tiny", model_size="tiny", batch=2, sequence_len=16, vocab_size=32
)
CHEAP: tuple[ForwardCostShape, ...] = (CHEAP_SHAPE,)


def _cheap_train() -> Generator[None, None, None]:
    """Install the one-row sweep for the duration of one test.

    Yields:
        Nothing; the sweep is installed for the body of the test.
    """
    cli_hooks.train_shapes = lambda: CHEAP
    try:
        yield
    finally:
        cli_hooks.train_shapes = cli_hooks._default_train_shapes


cheap_train = pytest.fixture(_cheap_train)


class TestTheSweep:
    def test_it_is_sized_for_a_step_not_a_forward(self) -> None:
        # A step holds parameters, gradients and two AdamW moments. The
        # forward sweep runs gpt2-large at batch 2; this one drops it to 1.
        large = [s for s in TRAIN_SHAPES if s["model_size"] == "large"]

        assert [s["batch"] for s in large] == [1]

    def test_every_workload_row_carries_the_real_vocabulary(self) -> None:
        workload = [s for s in TRAIN_SHAPES if not s["name"].startswith("gate-")]

        assert {s["vocab_size"] for s in workload} == {GPT2_VOCAB}

    def test_every_row_names_a_model_size_that_exists(self) -> None:
        assert [s for s in TRAIN_SHAPES if s["model_size"] not in GPT2_MODEL_SIZES] == []

    def test_no_two_rows_share_a_prefix(self) -> None:
        prefixes = [bench_cli.train_prefix(s) for s in TRAIN_SHAPES]

        assert len(set(prefixes)) == len(prefixes)

    def test_a_step_prefix_cannot_be_read_as_a_forward_one(self) -> None:
        # Two experiments, two vocabularies of names. A record that could be
        # read as either would be differenced against the wrong table.
        assert bench_cli.train_prefix(CHEAP_SHAPE).startswith("train-")

    def test_it_warms_up_more_than_the_forward_benchmark(self) -> None:
        # AdamW allocates its moment buffers lazily on the FIRST step, and
        # that happens once per run rather than once per step.
        assert (TRAIN_WARMUP, TRAIN_INNER, TRAIN_BATCHES) == (3, 1, 5)


class TestTheStepItself:
    def test_the_model_is_in_training_posture(self) -> None:
        # The forward benchmark left it in eval. Here dropout is live and
        # gradients are tracked, which is a different call to the dispatcher.
        step = train_step_setup(CHEAP_SHAPE, "cpu")

        assert step["model"].training is True

    def test_it_uses_the_optimizer_the_trainer_uses(self) -> None:
        step = train_step_setup(CHEAP_SHAPE, "cpu")

        assert TRAIN_OPTIMIZER == "adamw"
        assert type(step["optimizer"]).__name__ == "AdamW"

    def test_a_step_actually_moves_the_weights(self) -> None:
        # The load-bearing assertion. A "training step" that timed a forward
        # and a no-op optimizer would report a smaller cost and be wrong.
        step = train_step_setup(CHEAP_SHAPE, "cpu")
        first = next(iter(step["model"].parameters()))
        before = first.detach().clone()

        run_train_step(step)

        assert not torch.equal(before, first.detach())

    def test_a_step_actually_produces_gradients(self) -> None:
        step = train_step_setup(CHEAP_SHAPE, "cpu")

        run_train_step(step)
        grads = [p.grad for p in step["model"].parameters() if p.grad is not None]

        assert len(grads) == len(list(step["model"].parameters()))

    def test_the_gradient_is_clipped_to_the_trainers_ceiling(self) -> None:
        assert TRAIN_CLIP == 1.0

    def test_the_learning_rate_reaches_the_update_rule(self) -> None:
        # Read out of the WEIGHTS rather than out of `param_groups`, because
        # a rate sitting in the optimizer's config that never reaches the
        # update would pass the config read and change nothing.
        #
        # On its first step Adam's bias correction makes the ratio
        # m_hat / sqrt(v_hat) exactly +/-1 for every element with a nonzero
        # gradient, so the largest weight change in that step is the learning
        # rate itself, to within the epsilon and the decoupled decay term.
        step = train_step_setup(CHEAP_SHAPE, "cpu")
        first = next(iter(step["model"].parameters()))
        before = first.detach().clone()

        run_train_step(step)
        largest = float((first.detach() - before).abs().max().item())

        assert abs(largest - TRAIN_LR) < TRAIN_LR * 0.05


class TestMeasuringOneStep:
    def test_it_returns_a_median_a_spread_and_a_peak(self) -> None:
        step = train_step_setup(CHEAP_SHAPE, "cpu")

        cost = measure_train_step(step, "cpu", None)

        assert cost["seconds"] > 0.0
        assert cost["spread"] >= 0.0
        assert cost["peak_bytes"] == 0.0

    def test_pinning_the_math_backend_still_steps(self) -> None:
        step = train_step_setup(CHEAP_SHAPE, "cpu")

        cost = measure_train_step(step, "cpu", SDPBackend.MATH)

        assert cost["seconds"] > 0.0

    def test_a_step_is_slower_than_the_forward_it_contains(self) -> None:
        # Sanity on the instrument: a step is a forward plus a backward plus
        # an update, so it cannot be cheaper than the forward alone.
        from model_trainer.core.services.model.forward_cost import (
            forward_model_and_input,
            measure_forward,
        )

        model, ids = forward_model_and_input(CHEAP_SHAPE, "cpu")
        forward_only = measure_forward(model, ids, "cpu", None)
        step_cost = measure_train_step(train_step_setup(CHEAP_SHAPE, "cpu"), "cpu", None)

        assert step_cost["seconds"] > forward_only["seconds"]

    def test_time_train_step_returns_the_cost_when_it_fits(self) -> None:
        cost = time_train_step(train_step_setup(CHEAP_SHAPE, "cpu"), "cpu", None)

        if cost is None:
            raise AssertionError("a tiny cpu step must fit")
        assert cost["seconds"] > 0.0


def train_record(
    gpu: str,
    *,
    base_seconds: float = 0.100,
    # 1.24x rather than 1.15x: the latter lands on the rounding boundary of
    # the one-decimal format, so the fixture would be asserting a tie-break
    # rule instead of the multiplier.
    pinned_seconds: float = 0.124,
    base_peak: float = 4.0e9,
    pinned_peak: float = 6.0e9,
    fits: bool = True,
) -> RunRecord:
    """Build a record whose every declared row carries the same made-up cost."""
    observations: list[Observation] = []
    for shape in TRAIN_SHAPES:
        prefix = bench_cli.train_prefix(shape)
        observations.append(
            Observation(name=labelled(prefix, DEFAULT_KEY, FITTED_SUFFIX), value=TRUE_VALUE)
        )
        observations.append(
            Observation(name=labelled(prefix, DEFAULT_KEY, SECONDS_SUFFIX), value=base_seconds)
        )
        observations.append(
            Observation(name=labelled(prefix, DEFAULT_KEY, SPREAD_SUFFIX), value=0.0)
        )
        observations.append(
            Observation(name=labelled(prefix, DEFAULT_KEY, PEAK_SUFFIX), value=base_peak)
        )
        observations.append(
            Observation(
                name=labelled(prefix, "math", FITTED_SUFFIX),
                value=TRUE_VALUE if fits else 0.0,
            )
        )
        if fits:
            observations.append(
                Observation(name=labelled(prefix, "math", SECONDS_SUFFIX), value=pinned_seconds)
            )
            observations.append(
                Observation(name=labelled(prefix, "math", SPREAD_SUFFIX), value=0.0)
            )
            observations.append(
                Observation(name=labelled(prefix, "math", PEAK_SUFFIX), value=pinned_peak)
            )
    return run_record(
        experiment=bench_cli.TRAIN_COST_EXPERIMENT,
        label=bench_cli.TRAIN_COST_LABEL,
        fingerprint=RunFingerprint(
            image_digest="sha256:test",
            gpu_model=gpu,
            driver_version="580.82.07",
            determinism=PINNED,
        ),
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


class TestTheRecord:
    def test_it_declares_its_own_experiment(self, cheap_train: None) -> None:
        # A step is a forward, a backward and an update; a forward is a
        # forward. Two records answering those must not be differenced.
        record = bench_cli.train_run_record("cpu")

        assert record["experiment"] == "training-step-cost"
        assert record["experiment"] != "forward-pass-cost"

    def test_it_reports_both_arms_for_every_row(self, cheap_train: None) -> None:
        record = bench_cli.train_run_record("cpu")
        names = {o["name"] for o in record["observations"]}
        prefix = bench_cli.train_prefix(CHEAP_SHAPE)

        assert labelled(prefix, DEFAULT_KEY, SECONDS_SUFFIX) in names
        assert labelled(prefix, "math", SECONDS_SUFFIX) in names


class TestTheReport:
    def test_it_reuses_the_same_decision_logic_as_every_other_cost_table(self) -> None:
        assert report_cli.slowdown is shared_report.slowdown
        assert report_cli.memory_growth is shared_report.memory_growth

    def test_a_clean_pair_becomes_a_multiplier(self) -> None:
        values = {o["name"]: o["value"] for o in train_record("A100")["observations"]}

        assert report_cli.slowdown(values, bench_cli.train_prefix(TRAIN_SHAPES[0])) == "1.2x"

    def test_memory_growth_is_reported_for_a_step(self) -> None:
        values = {o["name"]: o["value"] for o in train_record("A100")["observations"]}

        assert report_cli.memory_growth(values, bench_cli.train_prefix(TRAIN_SHAPES[0])).startswith(
            "1.5x"
        )

    def test_a_row_that_did_not_fit_is_named(self) -> None:
        values = {o["name"]: o["value"] for o in train_record("V100", fits=False)["observations"]}

        assert (
            report_cli.slowdown(values, bench_cli.train_prefix(TRAIN_SHAPES[0]))
            == shared_report.DID_NOT_FIT
        )

    def test_an_absent_timing_is_said_rather_than_shown_as_zero(self) -> None:
        # The arm that did not fit has no seconds to render, and a training
        # sweep is where that is EXPECTED rather than exotic: backward keeps
        # the score matrix the fused kernel exists to avoid keeping, so the
        # pinned arm is the one that runs out of memory. A cell reading 0.0
        # would be read as "instant".
        values = {o["name"]: o["value"] for o in train_record("V100", fits=False)["observations"]}
        prefix = bench_cli.train_prefix(TRAIN_SHAPES[0])

        assert report_cli.milliseconds(values, prefix, "math") == report_cli.ABSENT

    def test_every_declared_row_gets_a_line(self) -> None:
        lines = report_cli.report_lines((("a100.json", train_record("A100")),))
        rows = [
            line
            for line in lines
            if line.startswith("  small-")
            or line.startswith("  medium-")
            or line.startswith("  large-")
            or line.startswith("  gate-")
        ]

        assert len(rows) == len(TRAIN_SHAPES)


class TestTheCommandLines:
    def test_main_writes_a_record_that_decodes_back(
        self, tmp_path: pathlib.Path, cheap_train: None
    ) -> None:
        out = tmp_path / "records" / "train.json"

        assert bench_cli.main(["--device", "cpu", "--out", str(out)]) == 0

        written = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert written["label"] == bench_cli.TRAIN_COST_LABEL

    def test_an_absent_device_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--device"):
            bench_cli.main(["--out", str(tmp_path / "t.json")])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--steps"):
            bench_cli.main(["--device", "cpu", "--out", str(tmp_path / "t.json"), "--steps", "3"])

    def test_the_production_hook_walks_the_whole_declared_sweep(self) -> None:
        assert cli_hooks._default_train_shapes() == TRAIN_SHAPES

    def test_the_report_reads_a_directory(self, tmp_path: pathlib.Path) -> None:
        directory = tmp_path / "train"
        directory.mkdir()
        (directory / "a100.json").write_text(
            dump_json_str(encode_run_record(train_record("A100"))), encoding="utf-8"
        )

        assert report_cli.main(["--dir", str(directory)]) == 0

    def test_the_report_refuses_a_missing_directory(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="no such directory"):
            report_cli.main(["--dir", str(tmp_path / "absent")])

    def test_the_benchmark_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, cheap_train: None
    ) -> None:
        saved = sys.argv
        sys.argv = ["x", "--device", "cpu", "--out", str(tmp_path / "t.json")]
        try:
            with pytest.raises(SystemExit) as raised:
                bench_cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_the_report_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        directory = tmp_path / "train"
        directory.mkdir()
        (directory / "a.json").write_text(
            dump_json_str(encode_run_record(train_record("A100"))), encoding="utf-8"
        )
        saved = sys.argv
        sys.argv = ["x", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as raised:
                report_cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_running_the_benchmark_as_main_actually_steps(
        self, tmp_path: pathlib.Path, cheap_train: None
    ) -> None:
        out = tmp_path / "t.json"
        module_name = "model_trainer.cli.train_benchmark"
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
        directory = tmp_path / "train"
        directory.mkdir()
        (directory / "a.json").write_text(
            dump_json_str(encode_run_record(train_record("A100"))), encoding="utf-8"
        )
        module_name = "model_trainer.cli.train_benchmark_report"
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
