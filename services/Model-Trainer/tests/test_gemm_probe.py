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

import math
import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
import torch
from platform_core.json_utils import load_json_str
from platform_core.known_answer_registry import gate_record
from platform_core.run_record import NO_PAYLOAD, RunRecord, decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import gemm_probe as gemm_cli
from model_trainer.core.services.model.deterministic_gemm import (
    CUBLAS_ARM,
    FP64_ARM,
    RANK1_ARM,
)
from model_trainer.core.services.model.gemm_probe import (
    DIGEST_BYTES,
    gemm_description,
    gemm_identity,
    gemm_operands,
    gemm_output,
)
from model_trainer.core.services.model.gemm_shapes import (
    DIGEST_SUFFIX,
    GEMM_EXPERIMENT,
    SUM_SUFFIX,
    GemmShape,
    gemm_label,
    probed_shapes,
)
from model_trainer.core.services.model.tensor_digest import describe_tensor

#: A shape small enough to run many times on a CPU, in the same orientation as
#: the real ones.
SMALL: GemmShape = {"rows": 8, "inner": 16, "cols": 4, "origin": "test"}

#: A second shape with different dimensions, so distinctness is testable.
WIDE: GemmShape = {"rows": 8, "inner": 32, "cols": 4, "origin": "test"}

#: The table the record and command-line tests walk. Three entries encode the
#: two properties the real tables have: two DISTINCT dimension sets, and one
#: deliberate twin -- the same dimensions under a second name -- mirroring the
#: ladder/grid overlap the real table keeps on purpose. Installed through
#: ``probed_shapes_hook`` because the real table digests ninety-three shapes
#: at up to N=4096, which on a CPU is minutes per record and these tests
#: build several.
CHEAP_PROBE: tuple[tuple[str, GemmShape], ...] = (
    ("small", SMALL),
    ("wide", WIDE),
    ("small-twin", SMALL),
)


def _cheap_probe() -> Generator[None, None, None]:
    """Install the three-shape table for the duration of one test.

    Written as ``pytest.fixture(impl)`` below rather than with the decorator,
    matching ``tests/conftest.py``: the decorator form returns an overloaded
    function and trips this package's ``disallow_any_decorated``.

    Yields:
        Nothing; the table is installed for the body of the test.
    """
    measurement_hooks.probed_shapes_hook = lambda: CHEAP_PROBE
    try:
        yield
    finally:
        measurement_hooks.probed_shapes_hook = measurement_hooks._default_probed_shapes


cheap_probe = pytest.fixture(_cheap_probe)


def _out_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the record path used by the CLI tests."""
    return tmp_path / "records" / "gemm.json"


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Return a complete CPU command line."""
    return [
        "--device",
        "cpu",
        "--out",
        str(_out_path(tmp_path)),
        "--controls",
        "none",
        "--kernel",
        CUBLAS_ARM,
    ]


def _record() -> RunRecord:
    """Return a CPU record under the untreated posture and the vendor kernel.

    The arm pair every assertion below is indifferent to. Tests that care
    which arm ran spell it out instead.
    """
    return gemm_cli.gemm_run_record("cpu", controls="none", kernel=CUBLAS_ARM)


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
        out = gemm_output(SMALL, "cpu", kernel=CUBLAS_ARM)

        assert list(out.shape) == [SMALL["cols"], SMALL["rows"]]

    def test_the_bias_is_actually_added(self) -> None:
        # Not decoration: the fused bias epilogue is what routes this to
        # cuBLASLt at all. `torch.mm` was measured to take the legacy
        # cublasSgemm path and log nothing under a trace.
        bias, x, w = gemm_operands(SMALL, "cpu")

        assert torch.equal(gemm_output(SMALL, "cpu", kernel=CUBLAS_ARM), torch.addmm(bias, x, w))
        assert not torch.equal(gemm_output(SMALL, "cpu", kernel=CUBLAS_ARM), torch.mm(x, w))


class TestTheDigest:
    def test_it_is_stable_for_one_tensor(self) -> None:
        out = gemm_output(SMALL, "cpu", kernel=CUBLAS_ARM)

        assert describe_tensor(out) == describe_tensor(out)

    def test_a_single_last_bit_change_changes_it(self) -> None:
        # The whole reason the digest is recorded rather than only the sum.
        out = gemm_output(SMALL, "cpu", kernel=CUBLAS_ARM)
        nudged = out.clone()
        nudged[0][0] = torch.nextafter(nudged[0][0], torch.tensor(float("inf")))

        assert describe_tensor(out)[0] != describe_tensor(nudged)[0]

    def test_it_catches_a_change_a_sum_cannot(self) -> None:
        # Two elements moved by +d and -d leave the sum EXACTLY untouched.
        # A hash cannot cancel, which is why the sum alone is not the check.
        out = gemm_output(SMALL, "cpu", kernel=CUBLAS_ARM).double()
        swapped = out.clone()
        swapped[0][0] += 1.0
        swapped[0][1] -= 1.0

        assert float(swapped.sum()) == float(out.sum())
        assert describe_tensor(out.float())[0] != describe_tensor(swapped.float())[0]

    def test_the_folded_digest_is_an_exactly_representable_integer(self) -> None:
        # Taking more bytes would start rounding, and two different tensors
        # could then record the same number -- the one failure this
        # observation must not have.
        digest, _ = describe_tensor(gemm_output(SMALL, "cpu", kernel=CUBLAS_ARM))

        assert digest == float(int(digest))
        # 2**48, written out: six bytes of digest.
        assert 0.0 <= digest < 281474976710656.0
        assert DIGEST_BYTES == 6

    def test_the_sum_is_a_host_reduction_rather_than_a_device_one(self) -> None:
        """The assertion that used to sit here ENCODED A BUG.

        It read ``describe_tensor(out)[1] == float(out.double().sum().item())``
        -- pinning the sum to a torch reduction on the tensor's own device.
        On a CPU-only runner that passes trivially, so it locked the defect in
        rather than catching it, and the docstring beside it claimed the
        opposite: "Summed in float64 on the CPU so the reduction cannot itself
        differ between devices".

        Measured 2026-08-27 on an RTX 3090 Ti: for one 64x50257 output, the
        device reduction gave 13927.145996611645 and the host reduction
        13927.145996611649 -- from BIT-IDENTICAL bytes, confirmed by
        ``torch.equal`` and by an identical digest. Every cross-card ``|sum``
        observation was therefore reporting the device's reduction ORDER, not
        its data, and two cards agreeing bit-for-bit could record different
        sums.

        The fix was to delete the duplicate and use ``describe_tensor``, which
        moves to the host and uses ``math.fsum``. This asserts the property
        that matters and that a CPU runner CAN see: the sum is exact, so it
        equals the exact sum of the values, which a device reduction tree is
        not obliged to give.
        """
        out = gemm_output(SMALL, "cpu", kernel=CUBLAS_ARM)
        _, total = describe_tensor(out)
        values: list[float] = out.flatten().tolist()

        assert total == math.fsum(values)


class TestTheReproducibilityGuard:
    # The guard itself moved to `tensor_digest`, where the sdpa probe also
    # uses it; its two directions are exercised there. What stays here is
    # the part specific to a GEMM: that the refusal names the call.
    def test_the_description_names_the_dimensions(self) -> None:
        assert gemm_description(SMALL, CUBLAS_ARM) == "a cublas GEMM M8xK16xN4"

    def test_two_shapes_do_not_share_a_description(self) -> None:
        widened: GemmShape = {**SMALL, "inner": SMALL["inner"] * 2}

        assert gemm_description(SMALL, CUBLAS_ARM) != gemm_description(widened, CUBLAS_ARM)

    def test_two_arms_do_not_share_a_description(self) -> None:
        # All three arms produce a tensor of the same shape, so a failure
        # reading only the dimensions would not say which arithmetic failed
        # to reproduce itself.
        assert gemm_description(SMALL, CUBLAS_ARM) != gemm_description(SMALL, RANK1_ARM)

    def test_a_real_cpu_call_reproduces_itself(self) -> None:
        assert gemm_identity(SMALL, "cpu", kernel=CUBLAS_ARM) == gemm_identity(
            SMALL, "cpu", kernel=CUBLAS_ARM
        )

    def test_every_arm_reproduces_itself(self) -> None:
        for arm in (CUBLAS_ARM, FP64_ARM, RANK1_ARM):
            assert gemm_identity(SMALL, "cpu", kernel=arm) == gemm_identity(
                SMALL, "cpu", kernel=arm
            )


class TestTheRecord:
    def test_it_carries_two_observations_per_shape(self, cheap_probe: None) -> None:
        record = _record()

        assert len(record["observations"]) == 2 * len(CHEAP_PROBE)

    def test_every_observation_is_named_for_its_shape_and_measurement(
        self, cheap_probe: None
    ) -> None:
        record = _record()
        expected = sorted(
            gemm_label(n, s, suffix)
            for n, s in CHEAP_PROBE
            for suffix in (DIGEST_SUFFIX, SUM_SUFFIX)
        )

        assert sorted(o["name"] for o in record["observations"]) == expected

    def test_the_values_are_what_the_probe_computes(self, cheap_probe: None) -> None:
        record = _record()
        by_name = {o["name"]: o["value"] for o in record["observations"]}
        name, shape = CHEAP_PROBE[0]
        digest, total = gemm_identity(shape, "cpu", kernel=CUBLAS_ARM)

        assert by_name[gemm_label(name, shape, DIGEST_SUFFIX)] == digest
        assert by_name[gemm_label(name, shape, SUM_SUFFIX)] == total

    def test_distinct_dimensions_produced_distinct_digests(self, cheap_probe: None) -> None:
        # If they had not, the probe would be measuring one thing repeatedly
        # and every shape would agree across cards for free.
        record = _record()
        by_name = {o["name"]: o["value"] for o in record["observations"]}
        by_dims = {
            (s["rows"], s["inner"], s["cols"]): by_name[gemm_label(n, s, DIGEST_SUFFIX)]
            for n, s in CHEAP_PROBE
        }

        assert len(set(by_dims.values())) == len(by_dims)

    def test_one_shape_under_two_names_gives_one_digest(self, cheap_probe: None) -> None:
        # The real ladder table and sweep grid overlap -- tiny-attn-proj is
        # M128/K128, which is also a grid point -- and the overlap is measured
        # twice rather than deduplicated. CHEAP_PROBE keeps a twin for the
        # same reason: the two names must agree, which checks the result
        # depends on the DIMENSIONS and not on which table asked for them.
        by_name = {o["name"]: o["value"] for o in _record()["observations"]}

        seen: dict[tuple[int, int, int], str] = {}
        overlaps = 0
        for name, s in CHEAP_PROBE:
            dims = (s["rows"], s["inner"], s["cols"])
            twin = seen.get(dims)
            if twin is not None:
                overlaps += 1
                assert (
                    by_name[gemm_label(name, s, DIGEST_SUFFIX)]
                    == by_name[gemm_label(twin, s, DIGEST_SUFFIX)]
                )
            seen[dims] = name

        # The check is worthless if the table ever stops overlapping.
        assert overlaps > 0

    def test_the_real_table_keeps_an_overlap_for_the_twin_check(self) -> None:
        # The cheap table's twin mirrors a property the real table must not
        # silently lose: at least two entries sharing dimensions.
        dims = [(s["rows"], s["inner"], s["cols"]) for _, s in probed_shapes()]

        assert len(dims) > len(set(dims))

    def test_the_production_hook_walks_the_whole_declared_set(self) -> None:
        assert measurement_hooks._default_probed_shapes() == probed_shapes()

    def test_it_declares_its_own_experiment(self, cheap_probe: None) -> None:
        assert _record()["experiment"] == GEMM_EXPERIMENT

    def test_it_carries_no_payload_digest(self, cheap_probe: None) -> None:
        assert _record()["payload_digest"] == NO_PAYLOAD

    def test_the_registry_refuses_it_by_observation_count(self, cheap_probe: None) -> None:
        record = _record()

        with pytest.raises(ValueError, match="exactly one observation"):
            gate_record((), record)


class TestTheCommandLine:
    def test_main_writes_a_record_that_decodes_back(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
        assert gemm_cli.main(_argv(tmp_path)) == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))

        assert decoded["label"] == gemm_cli.gemm_label_for("none", CUBLAS_ARM)
        assert len(decoded["observations"]) == 2 * len(CHEAP_PROBE)

    def test_main_creates_the_parent_directory(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
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
            gemm_cli.main(["--device", "cpu", "--controls", "none", "--kernel", CUBLAS_ARM])

    def test_an_absent_controls_is_refused(self, tmp_path: pathlib.Path) -> None:
        # No default posture. A record whose arm was guessed names a
        # condition it may not have run under.
        with pytest.raises(ValueError, match="--controls"):
            gemm_cli.main(
                ["--device", "cpu", "--out", str(_out_path(tmp_path)), "--kernel", CUBLAS_ARM]
            )

    def test_an_absent_kernel_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--kernel"):
            gemm_cli.main(
                ["--device", "cpu", "--out", str(_out_path(tmp_path)), "--controls", "none"]
            )

    def test_an_unknown_kernel_is_refused_before_anything_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ValueError, match="kernel must be one of"):
            gemm_cli.main([*_argv(tmp_path)[:-1], "triton"])

        assert not _out_path(tmp_path).exists()

    def test_the_destination_is_resolved_before_anything_computes(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Until 2026-08-29 `--out` was read AFTER the record was built, so a
        # command line missing it ran all fifty-nine GEMMs and then failed.
        # The refusal must arrive before the work, not after it.
        with pytest.raises(ValueError, match="--out"):
            gemm_cli.main(["--device", "cpu", "--controls", "none", "--kernel", "triton"])

    def test_the_label_names_both_arms(self, tmp_path: pathlib.Path, cheap_probe: None) -> None:
        argv = ["--device", "cpu", "--out", str(_out_path(tmp_path))]

        assert gemm_cli.main([*argv, "--controls", "both", "--kernel", RANK1_ARM]) == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))

        assert decoded["label"] == "gemm-attribution-v2-both-rank1"

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--shape"):
            gemm_cli.main([*_argv(tmp_path), "--shape", "medium-mlp-proj"])

    def test_the_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-gemm-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                gemm_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_probes(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
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
        assert len(decoded["observations"]) == 2 * len(CHEAP_PROBE)
