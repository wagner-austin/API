"""The isolated-GEMM probe, run for real on the CPU.

Nothing is faked. The matmuls run; the digests are taken over the bytes those
matmuls produced.

WHAT THESE TESTS CANNOT COVER, and it is the point of the probe. The question
is whether two CARDS produce the same tensor, and one CPU cannot produce two
cards. What is checkable here is everything the answer depends on: that both
cards would be handed identical operands, that the digest actually
discriminates, and that a call which failed to reproduce itself is refused
rather than recorded.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
import torch
from platform_core.json_utils import load_json_str
from platform_core.known_answer_registry import gate_record
from platform_core.run_record import NO_PAYLOAD, decode_run_record

from model_trainer.cli import gemm_probe as gemm_cli
from model_trainer.core.services.model.gemm_probe import (
    DIGEST_BYTES,
    describe_output,
    gemm_identity,
    gemm_operands,
    gemm_output,
    require_reproduced,
)
from model_trainer.core.services.model.gemm_shapes import (
    DIGEST_SUFFIX,
    GEMM_COLS,
    GEMM_EXPERIMENT,
    GEMM_SHAPES,
    SUM_SUFFIX,
    GemmShape,
    gemm_label,
)

#: A shape small enough to run many times on a CPU, in the same orientation as
#: the real ones.
SMALL: GemmShape = {"rows": 8, "inner": 16, "cols": 4, "origin": "test"}


def _out_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the record path used by the CLI tests."""
    return tmp_path / "records" / "gemm.json"


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Return a complete CPU command line."""
    return ["--device", "cpu", "--out", str(_out_path(tmp_path))]


class TestTheShapeTable:
    def test_every_shape_has_positive_dimensions(self) -> None:
        bad = [
            name for name, s in GEMM_SHAPES.items() if min(s["rows"], s["inner"], s["cols"]) <= 0
        ]

        assert bad == []

    def test_every_shape_shares_the_ladder_sequence_length(self) -> None:
        # The ladder runs one sequence of the gate rung's length, so a shape
        # with a different N is not a call the ladder ever issued and cannot
        # be read against a rung.
        assert [s["cols"] for s in GEMM_SHAPES.values()] == [GEMM_COLS for _ in GEMM_SHAPES]

    def test_every_shape_says_where_it_came_from(self) -> None:
        # A shape with no origin is a number nobody can trace to a rung.
        assert [n for n, s in GEMM_SHAPES.items() if not s["origin"].strip()] == []

    def test_labels_are_unique_across_shapes_and_measurements(self) -> None:
        labels = [
            gemm_label(n, s, suffix)
            for n, s in GEMM_SHAPES.items()
            for suffix in (DIGEST_SUFFIX, SUM_SUFFIX)
        ]

        assert sorted(labels) == sorted(set(labels))

    def test_a_label_carries_the_dimensions_and_the_measurement(self) -> None:
        assert (
            gemm_label("medium-mlp-proj", GEMM_SHAPES["medium-mlp-proj"], DIGEST_SUFFIX)
            == "gemm-medium-mlp-proj-M1024-K4096-N64|digest48"
        )

    def test_resizing_a_shape_renames_its_measurements(self) -> None:
        # The property that keeps a re-dimensioned probe from recording under
        # a name whose earlier value came from different arithmetic.
        widened: GemmShape = {**SMALL, "inner": SMALL["inner"] * 2}

        assert gemm_label("x", SMALL, SUM_SUFFIX) != gemm_label("x", widened, SUM_SUFFIX)


class TestTheOperands:
    def test_both_cards_would_be_handed_identical_operands(self) -> None:
        # THE precondition. Generating on the device would use the per-device
        # CUDA RNG and hand two cards different inputs, producing a difference
        # that says nothing about how they multiply.
        first = gemm_operands(SMALL, "cpu")
        second = gemm_operands(SMALL, "cpu")

        for a, b in zip(first, second, strict=True):
            assert torch.equal(a, b)

    def test_the_operands_have_the_shapes_the_call_needs(self) -> None:
        bias, x, w = gemm_operands(SMALL, "cpu")

        assert list(bias.shape) == [SMALL["rows"]]
        assert list(x.shape) == [SMALL["cols"], SMALL["inner"]]
        assert list(w.shape) == [SMALL["inner"], SMALL["rows"]]

    def test_two_different_shapes_do_not_share_operands(self) -> None:
        wider: GemmShape = {**SMALL, "inner": SMALL["inner"] * 2}

        assert gemm_operands(SMALL, "cpu")[1].shape != gemm_operands(wider, "cpu")[1].shape

    def test_the_output_has_the_orientation_cublas_reports(self) -> None:
        # A logged call `A[M x K] B[K x N]` is addmm(bias[M], x[N,K], w[K,M]),
        # verified against a real trace: addmm(b, x[64,4096], w[4096,1024])
        # logs Adesc=[rows=1024 cols=4096] Bdesc=[rows=4096 cols=64], which is
        # the ladder's medium MLP-projection call exactly.
        out = gemm_output(SMALL, "cpu")

        assert list(out.shape) == [SMALL["cols"], SMALL["rows"]]

    def test_the_bias_is_actually_added(self) -> None:
        # Not decoration: the fused bias epilogue is what routes this to
        # cuBLASLt at all. `torch.mm` was measured to take the legacy
        # cublasSgemm path and log nothing under a trace.
        bias, x, w = gemm_operands(SMALL, "cpu")

        assert torch.equal(gemm_output(SMALL, "cpu"), torch.addmm(bias, x, w))
        assert not torch.equal(gemm_output(SMALL, "cpu"), torch.mm(x, w))


class TestTheDigest:
    def test_it_is_stable_for_one_tensor(self) -> None:
        out = gemm_output(SMALL, "cpu")

        assert describe_output(out) == describe_output(out)

    def test_a_single_last_bit_change_changes_it(self) -> None:
        # The whole reason the digest is recorded rather than only the sum.
        out = gemm_output(SMALL, "cpu")
        nudged = out.clone()
        nudged[0][0] = torch.nextafter(nudged[0][0], torch.tensor(float("inf")))

        assert describe_output(out)[0] != describe_output(nudged)[0]

    def test_it_catches_a_change_a_sum_cannot(self) -> None:
        # Two elements moved by +d and -d leave the sum EXACTLY untouched.
        # A hash cannot cancel, which is why the sum alone is not the check.
        out = gemm_output(SMALL, "cpu").double()
        swapped = out.clone()
        swapped[0][0] += 1.0
        swapped[0][1] -= 1.0

        assert float(swapped.sum()) == float(out.sum())
        assert describe_output(out.float())[0] != describe_output(swapped.float())[0]

    def test_the_folded_digest_is_an_exactly_representable_integer(self) -> None:
        # Taking more bytes would start rounding, and two different tensors
        # could then record the same number -- the one failure this
        # observation must not have.
        digest, _ = describe_output(gemm_output(SMALL, "cpu"))

        assert digest == float(int(digest))
        # 2**48, written out: six bytes of digest.
        assert 0.0 <= digest < 281474976710656.0
        assert DIGEST_BYTES == 6

    def test_the_sum_is_reported_beside_it(self) -> None:
        out = gemm_output(SMALL, "cpu")

        assert describe_output(out)[1] == float(out.double().sum().item())


class TestTheReproducibilityGuard:
    def test_two_identical_runs_pass_and_return_the_first(self) -> None:
        out = gemm_output(SMALL, "cpu")

        assert require_reproduced(out, out.clone(), SMALL, "cpu") is out

    def test_two_differing_runs_are_refused_by_name(self) -> None:
        out = gemm_output(SMALL, "cpu")
        other = out.clone()
        other[0][0] = torch.nextafter(other[0][0], torch.tensor(float("inf")))

        with pytest.raises(RuntimeError, match="did not reproduce itself on cpu"):
            require_reproduced(out, other, SMALL, "cpu")

    def test_the_refusal_names_the_shape(self) -> None:
        other = gemm_output(SMALL, "cpu") + 1.0

        with pytest.raises(RuntimeError, match=r"M8xK16xN4"):
            require_reproduced(gemm_output(SMALL, "cpu"), other, SMALL, "cpu")

    def test_a_real_cpu_call_reproduces_itself(self) -> None:
        assert gemm_identity(SMALL, "cpu") == gemm_identity(SMALL, "cpu")


class TestTheRecord:
    def test_it_carries_two_observations_per_shape(self) -> None:
        record = gemm_cli.gemm_run_record("cpu")

        assert len(record["observations"]) == 2 * len(GEMM_SHAPES)

    def test_every_observation_is_named_for_its_shape_and_measurement(self) -> None:
        record = gemm_cli.gemm_run_record("cpu")
        expected = sorted(
            gemm_label(n, s, suffix)
            for n, s in GEMM_SHAPES.items()
            for suffix in (DIGEST_SUFFIX, SUM_SUFFIX)
        )

        assert sorted(o["name"] for o in record["observations"]) == expected

    def test_the_values_are_what_the_probe_computes(self) -> None:
        record = gemm_cli.gemm_run_record("cpu")
        by_name = {o["name"]: o["value"] for o in record["observations"]}
        name, shape = next(iter(GEMM_SHAPES.items()))
        digest, total = gemm_identity(shape, "cpu")

        assert by_name[gemm_label(name, shape, DIGEST_SUFFIX)] == digest
        assert by_name[gemm_label(name, shape, SUM_SUFFIX)] == total

    def test_no_two_shapes_produced_the_same_digest(self) -> None:
        # If they had, the probe would be measuring one thing repeatedly and
        # every shape would agree across cards for free.
        record = gemm_cli.gemm_run_record("cpu")
        digests = [o["value"] for o in record["observations"] if o["name"].endswith(DIGEST_SUFFIX)]

        assert len(set(digests)) == len(digests)

    def test_it_declares_its_own_experiment(self) -> None:
        assert gemm_cli.gemm_run_record("cpu")["experiment"] == GEMM_EXPERIMENT

    def test_it_carries_no_payload_digest(self) -> None:
        assert gemm_cli.gemm_run_record("cpu")["payload_digest"] == NO_PAYLOAD

    def test_the_registry_refuses_it_by_observation_count(self) -> None:
        record = gemm_cli.gemm_run_record("cpu")

        with pytest.raises(ValueError, match="exactly one observation"):
            gate_record((), record)


class TestTheCommandLine:
    def test_main_writes_a_record_that_decodes_back(self, tmp_path: pathlib.Path) -> None:
        assert gemm_cli.main(_argv(tmp_path)) == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))

        assert decoded["label"] == gemm_cli.GEMM_LABEL
        assert len(decoded["observations"]) == 2 * len(GEMM_SHAPES)

    def test_main_creates_the_parent_directory(self, tmp_path: pathlib.Path) -> None:
        assert not _out_path(tmp_path).parent.exists()

        assert gemm_cli.main(_argv(tmp_path)) == 0

        assert _out_path(tmp_path).is_file()

    def test_an_absent_device_is_refused_and_nothing_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ValueError, match="--device"):
            gemm_cli.main(["--out", str(_out_path(tmp_path))])

        assert not _out_path(tmp_path).exists()

    def test_an_absent_out_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            gemm_cli.main(["--device", "cpu"])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--shape"):
            gemm_cli.main([*_argv(tmp_path), "--shape", "medium-mlp-proj"])

    def test_the_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-gemm-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                gemm_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_probes(self, tmp_path: pathlib.Path) -> None:
        module_name = "model_trainer.cli.gemm_probe"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-gemm-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))
        assert len(decoded["observations"]) == 2 * len(GEMM_SHAPES)
