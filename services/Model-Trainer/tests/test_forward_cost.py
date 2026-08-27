"""The end-to-end forward benchmark and its report.

The passes run for real on the CPU, at a model small enough for a test
runner. What a CPU runner cannot establish is the answer -- what the
attention pin costs a forward pass on a V100 -- and it does not try. The
assertions are about the instrument: that both arms time the same weights,
that the shapes carry the vocabulary the answer depends on, and that the
report reuses the per-call benchmark's decision logic rather than a second
copy of it.
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
    NO_PAYLOAD,
    Observation,
    RunRecord,
    decode_run_record,
    encode_run_record,
    run_record,
)
from platform_core.testing import sample_run_fingerprint
from torch.nn.attention import SDPBackend

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import forward_benchmark as bench_cli
from model_trainer.cli import forward_benchmark_report as report_cli
from model_trainer.cli import sdpa_benchmark_report as shared_report
from model_trainer.core.services.model.cost_labels import (
    DEFAULT_KEY,
    FITTED_SUFFIX,
    PEAK_SUFFIX,
    SECONDS_SUFFIX,
    SPREAD_SUFFIX,
    TRUE_VALUE,
    labelled,
)
from model_trainer.core.services.model.forward_cost import (
    FORWARD_BATCHES,
    FORWARD_INNER,
    FORWARD_SHAPES,
    FORWARD_WARMUP,
    GPT2_VOCAB,
    ForwardCostShape,
    forward_model_and_input,
    measure_forward,
    time_forward,
)
from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})

#: A model a test runner can build in a moment. The declared rows go to 774
#: million parameters over a 50,257-token vocabulary; on a CPU that is not
#: slow, it is unusable.
CHEAP_SHAPE = ForwardCostShape(
    name="test-tiny", model_size="tiny", batch=2, sequence_len=16, vocab_size=32
)
CHEAP: tuple[ForwardCostShape, ...] = (CHEAP_SHAPE,)


def _cheap_forward() -> Generator[None, None, None]:
    """Install the one-row sweep for the duration of one test.

    Yields:
        Nothing; the sweep is installed for the body of the test.
    """
    cli_hooks.forward_shapes = lambda: CHEAP
    try:
        yield
    finally:
        cli_hooks.forward_shapes = cli_hooks._default_forward_shapes


cheap_forward = pytest.fixture(_cheap_forward)


class TestTheSweep:
    def test_every_workload_row_carries_the_real_gpt2_vocabulary(self) -> None:
        # The output projection scales with the vocabulary, so attention's
        # SHARE of the pass -- which is what the end-to-end multiplier
        # measures -- moves with this choice.
        workload = [s for s in FORWARD_SHAPES if not s["name"].startswith("gate-")]

        assert {s["vocab_size"] for s in workload} == {GPT2_VOCAB}

    def test_the_gate_row_keeps_the_probes_own_configuration(self) -> None:
        gate = [s for s in FORWARD_SHAPES if s["name"].startswith("gate-")]

        assert gate == [
            {
                "name": "gate-tiny-b1-s64",
                "model_size": "tiny",
                "batch": 1,
                "sequence_len": 64,
                "vocab_size": 512,
            }
        ]

    def test_every_row_names_a_model_size_that_exists(self) -> None:
        assert [s for s in FORWARD_SHAPES if s["model_size"] not in GPT2_MODEL_SIZES] == []

    def test_no_two_rows_share_a_prefix(self) -> None:
        prefixes = [bench_cli.forward_prefix(s) for s in FORWARD_SHAPES]

        assert len(set(prefixes)) == len(prefixes)

    def test_the_prefix_carries_the_vocabulary(self) -> None:
        # A record read on its own has to say which vocabulary it used, or it
        # is not reproducible.
        assert bench_cli.forward_prefix(FORWARD_SHAPES[0]).endswith(f"-v{GPT2_VOCAB}")

    def test_the_batching_constants_differ_from_the_per_call_benchmarks(self) -> None:
        # A forward pass issues hundreds of launches internally and has
        # already amortised them; batching twenty would only multiply the
        # wall clock.
        assert (FORWARD_WARMUP, FORWARD_INNER, FORWARD_BATCHES) == (2, 1, 5)


class TestBuildingOneRow:
    def test_the_input_is_the_declared_batch_and_length(self) -> None:
        _, ids = forward_model_and_input(CHEAP_SHAPE, "cpu")

        assert tuple(ids.shape) == (2, 16)

    def test_every_id_is_inside_the_vocabulary(self) -> None:
        # Taken modulo the vocabulary so a sequence longer than it still
        # indexes real tokens, unlike the probe, which refuses that case
        # because there the VALUES matter.
        long_row = ForwardCostShape(
            name="x", model_size="tiny", batch=1, sequence_len=64, vocab_size=8
        )
        _, ids = forward_model_and_input(long_row, "cpu")

        assert int(ids.max().item()) < 8

    def test_the_model_runs_the_input_it_was_built_with(self) -> None:
        model, ids = forward_model_and_input(CHEAP_SHAPE, "cpu")

        with torch.no_grad():
            loss = float(model.forward(input_ids=ids, labels=ids).loss.item())

        assert loss > 0.0

    def test_two_builds_of_one_row_give_the_same_weights(self) -> None:
        first, ids = forward_model_and_input(CHEAP_SHAPE, "cpu")
        second, _ = forward_model_and_input(CHEAP_SHAPE, "cpu")

        with torch.no_grad():
            a = float(first.forward(input_ids=ids, labels=ids).loss.item())
            b = float(second.forward(input_ids=ids, labels=ids).loss.item())

        assert a == b


class TestMeasuringOnePass:
    def test_it_returns_a_median_a_spread_and_a_peak(self) -> None:
        model, ids = forward_model_and_input(CHEAP_SHAPE, "cpu")

        cost = measure_forward(model, ids, "cpu", None)

        assert cost["seconds"] > 0.0
        assert cost["spread"] >= 0.0
        assert cost["peak_bytes"] == 0.0

    def test_pinning_the_math_backend_still_measures(self) -> None:
        model, ids = forward_model_and_input(CHEAP_SHAPE, "cpu")

        cost = measure_forward(model, ids, "cpu", SDPBackend.MATH)

        assert cost["seconds"] > 0.0

    def test_both_arms_time_the_same_model_object(self) -> None:
        # Rebuilding between arms would put a fresh random init into the
        # comparison, and would pay for the construction twice.
        model, ids = forward_model_and_input(CHEAP_SHAPE, "cpu")

        with torch.no_grad():
            before = float(model.forward(input_ids=ids, labels=ids).loss.item())
        measure_forward(model, ids, "cpu", SDPBackend.MATH)
        with torch.no_grad():
            after = float(model.forward(input_ids=ids, labels=ids).loss.item())

        assert before == after

    def test_time_forward_returns_the_cost_when_it_fits(self) -> None:
        model, ids = forward_model_and_input(CHEAP_SHAPE, "cpu")

        cost = time_forward(model, ids, "cpu", None)

        if cost is None:
            raise AssertionError("a tiny cpu pass must fit")
        assert cost["seconds"] > 0.0


def forward_record(
    gpu: str,
    *,
    base_seconds: float = 0.010,
    pinned_seconds: float = 0.012,
    base_peak: float = 1.0e9,
    pinned_peak: float = 1.4e9,
    fits: bool = True,
) -> RunRecord:
    """Build a record whose every declared row carries the same made-up cost."""
    observations: list[Observation] = []
    for shape in FORWARD_SHAPES:
        prefix = bench_cli.forward_prefix(shape)
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
        experiment=bench_cli.FORWARD_COST_EXPERIMENT,
        label=bench_cli.FORWARD_COST_LABEL,
        fingerprint=sample_run_fingerprint(
            image_digest="sha256:test",
            gpu_model=gpu,
            driver_version="580.82.07",
            determinism=PINNED,
        ),
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


class TestTheRecord:
    def test_it_declares_its_own_experiment(self, cheap_forward: None) -> None:
        # "What does one attention call cost" and "what does a forward pass
        # cost" are different questions; two records answering them must not
        # be differenced against each other.
        record = bench_cli.forward_run_record("cpu")

        assert record["experiment"] == "forward-pass-cost"
        assert record["experiment"] != "sdpa-backend-cost"

    def test_it_reports_both_arms_for_every_row(self, cheap_forward: None) -> None:
        record = bench_cli.forward_run_record("cpu")
        names = {o["name"] for o in record["observations"]}
        prefix = bench_cli.forward_prefix(CHEAP_SHAPE)

        assert labelled(prefix, DEFAULT_KEY, SECONDS_SUFFIX) in names
        assert labelled(prefix, "math", SECONDS_SUFFIX) in names


class TestTheReport:
    def test_it_reuses_the_per_call_benchmarks_decision_logic(self) -> None:
        # Not a second copy: one place decides whether a ratio may be printed
        # for either experiment.
        assert report_cli.slowdown is shared_report.slowdown
        assert report_cli.memory_growth is shared_report.memory_growth

    def test_a_clean_pair_becomes_a_multiplier(self) -> None:
        values = {o["name"]: o["value"] for o in forward_record("A100")["observations"]}

        assert report_cli.slowdown(values, bench_cli.forward_prefix(FORWARD_SHAPES[0])) == "1.2x"

    def test_it_prints_absolute_milliseconds_beside_the_ratio(self) -> None:
        # A multiplier alone cannot say whether a cost is affordable: 5x on a
        # two-millisecond pass and 5x on a two-second one are different facts.
        values = {o["name"]: o["value"] for o in forward_record("A100")["observations"]}
        prefix = bench_cli.forward_prefix(FORWARD_SHAPES[0])

        assert report_cli.milliseconds(values, prefix, DEFAULT_KEY) == "10.0"
        assert report_cli.milliseconds(values, prefix, "math") == "12.0"

    def test_an_absent_timing_is_said_rather_than_shown_as_zero(self) -> None:
        values = {o["name"]: o["value"] for o in forward_record("V100", fits=False)["observations"]}
        prefix = bench_cli.forward_prefix(FORWARD_SHAPES[0])

        assert report_cli.milliseconds(values, prefix, "math") == report_cli.ABSENT

    def test_every_declared_row_gets_a_line(self) -> None:
        lines = report_cli.report_lines((("a100.json", forward_record("A100")),))
        rows = [line for line in lines if line.startswith("  small-") or line.startswith("  gate-")]

        assert len(rows) == len([s for s in FORWARD_SHAPES if s["name"][0] in "sg"])

    def test_the_header_names_the_card(self) -> None:
        lines = report_cli.report_lines((("v100.json", forward_record("Tesla V100")),))

        assert "Tesla V100" in lines[0]


class TestTheCommandLines:
    def test_main_writes_a_record_that_decodes_back(
        self, tmp_path: pathlib.Path, cheap_forward: None
    ) -> None:
        out = tmp_path / "records" / "fwd.json"

        assert bench_cli.main(["--device", "cpu", "--out", str(out)]) == 0

        written = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert written["label"] == bench_cli.FORWARD_COST_LABEL

    def test_an_absent_device_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--device"):
            bench_cli.main(["--out", str(tmp_path / "fwd.json")])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--rows"):
            bench_cli.main(["--device", "cpu", "--out", str(tmp_path / "f.json"), "--rows", "2"])

    def test_the_production_hook_walks_the_whole_declared_sweep(self) -> None:
        assert cli_hooks._default_forward_shapes() == FORWARD_SHAPES

    def test_the_report_reads_a_directory(self, tmp_path: pathlib.Path) -> None:
        directory = tmp_path / "fwd"
        directory.mkdir()
        (directory / "a100.json").write_text(
            dump_json_str(encode_run_record(forward_record("A100"))), encoding="utf-8"
        )

        assert report_cli.main(["--dir", str(directory)]) == 0

    def test_the_report_refuses_a_missing_directory(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="no such directory"):
            report_cli.main(["--dir", str(tmp_path / "absent")])

    def test_the_benchmark_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, cheap_forward: None
    ) -> None:
        saved = sys.argv
        sys.argv = ["x", "--device", "cpu", "--out", str(tmp_path / "f.json")]
        try:
            with pytest.raises(SystemExit) as raised:
                bench_cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_the_report_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        directory = tmp_path / "fwd"
        directory.mkdir()
        (directory / "a.json").write_text(
            dump_json_str(encode_run_record(forward_record("A100"))), encoding="utf-8"
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
        self, tmp_path: pathlib.Path, cheap_forward: None
    ) -> None:
        out = tmp_path / "f.json"
        module_name = "model_trainer.cli.forward_benchmark"
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
        directory = tmp_path / "fwd"
        directory.mkdir()
        (directory / "a.json").write_text(
            dump_json_str(encode_run_record(forward_record("A100"))), encoding="utf-8"
        )
        module_name = "model_trainer.cli.forward_benchmark_report"
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
