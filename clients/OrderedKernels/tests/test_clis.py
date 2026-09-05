"""The three command lines, on real CUDA work over small installed tables."""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli.gemm_probe import gemm_label_for
from model_trainer.core.services.model.gemm_probe import gemm_identity
from model_trainer.core.services.model.gemm_shapes import DIGEST_SUFFIX, GemmShape, gemm_label
from model_trainer.core.services.model.probe_shapes import require_probe_shape
from model_trainer.core.services.model.trace_plan import WORKSPACE_NAME, WORKSPACE_UNSET
from model_trainer.core.services.model.train_step_plan import train_step_label
from model_trainer.core.services.model.train_step_probe import train_step_once
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from ordered_kernels.cli import bench as bench_cli
from ordered_kernels.cli import gemm_probe as gemm_cli
from ordered_kernels.cli import train_step as train_cli

SMALL: GemmShape = {"rows": 24, "inner": 33, "cols": 8, "origin": "test"}
WIDE: GemmShape = {"rows": 24, "inner": 64, "cols": 8, "origin": "test"}
CHEAP: tuple[tuple[str, GemmShape], ...] = (("small", SMALL), ("wide", WIDE))


def _cheap_probe() -> Generator[None, None, None]:
    """Install a two-shape table; written as pytest.fixture(impl) per house style.

    Yields:
        Nothing; the table holds for the test body.
    """
    # Replacing an attribute that does not exist silently creates it, and the
    # test then reads back its own fake while production reads nothing: that
    # is exactly how the 2026-09-04 hook move (5bea978c) broke gemm_probe on
    # the cluster while this suite stayed green. Prove the production default
    # is where this fixture is about to point before pointing there.
    assert measurement_hooks.probed_shapes_hook is measurement_hooks._default_probed_shapes
    assert measurement_hooks.benchmark_shapes is measurement_hooks._default_benchmark_shapes
    measurement_hooks.probed_shapes_hook = lambda: CHEAP
    measurement_hooks.benchmark_shapes = lambda: CHEAP
    try:
        yield
    finally:
        measurement_hooks.probed_shapes_hook = measurement_hooks._default_probed_shapes
        measurement_hooks.benchmark_shapes = measurement_hooks._default_benchmark_shapes


cheap_probe = pytest.fixture(_cheap_probe)


def _no_workspace() -> Generator[None, None, None]:
    """Pin the split-K env observation to unset, restoring after.

    Yields:
        Nothing; the pin holds for the test body.
    """
    cli_hooks.env_cublaslt_workspace = lambda: None
    try:
        yield
    finally:
        cli_hooks.env_cublaslt_workspace = cli_hooks._default_env_cublaslt_workspace


no_workspace = pytest.fixture(_no_workspace)


class TestTheGemmCli:
    def test_its_record_equals_the_rank1_instrument_digest_for_digest(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
        out = tmp_path / "gemm.json"

        assert gemm_cli.main(["--device", "cuda", "--out", str(out)]) == 0

        decoded = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert decoded["label"] == gemm_label_for("both", "ordered")
        by_name = {o["name"]: o["value"] for o in decoded["observations"]}
        for name, shape in CHEAP:
            digest, total = gemm_identity(shape, "cuda", kernel="rank1")
            assert by_name[gemm_label(name, shape, DIGEST_SUFFIX)] == digest
            assert by_name[gemm_label(name, shape, "sum")] == total

    def test_an_absent_out_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            gemm_cli.main(["--device", "cuda"])

    def test_running_the_module_as_main_actually_probes(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
        module_name = "ordered_kernels.cli.gemm_probe"
        out = tmp_path / "gemm-main.json"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["ordered-gemm-probe", "--device", "cuda", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0
        decoded = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert len(decoded["observations"]) == 2 * len(CHEAP)


class TestTheTrainCli:
    def test_its_record_equals_the_owned_arm_tensor_for_tensor(
        self, tmp_path: pathlib.Path, no_workspace: None
    ) -> None:
        # The step-level oracle: ordered and owned are one arithmetic, so a
        # whole tiny-rung record must match the owned arm's digests exactly.
        out = tmp_path / "train.json"

        args = ["--device", "cuda", "--rungs", "tiny", "--out", str(out), "--attention", "vendor"]

        assert train_cli.main(args) == 0

        decoded = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert decoded["label"] == train_step_label(("tiny",), "both", "ordered")
        by_name = {o["name"]: o["value"] for o in decoded["observations"]}
        assert by_name[WORKSPACE_NAME] == WORKSPACE_UNSET
        tensors, loss = train_step_once("cuda", require_probe_shape("tiny"), kernel="owned")
        assert by_name["tiny|loss"] == loss
        for tensor in tensors:
            name = f"tiny|{tensor['kind']}|{tensor['path']}|digest48"
            assert by_name[name] == tensor["digest"]

    def test_a_repeated_rung_is_refused_before_anything_computes(
        self, tmp_path: pathlib.Path
    ) -> None:
        out = tmp_path / "train.json"

        args = ["--device", "cuda", "--rungs", "tiny,tiny", "--out", str(out)]

        with pytest.raises(ValueError, match="tiny"):
            train_cli.main([*args, "--attention", "vendor"])

        assert not out.exists()

    def test_the_ordered_full_arm_computes_rather_than_passes_through(
        self, tmp_path: pathlib.Path, no_workspace: None
    ) -> None:
        # Same parameters, same observation names, a NEW label -- and the
        # digests must differ from the vendor-attention arm's somewhere,
        # because owned attention is different arithmetic for the same
        # function. A full record that merely echoed the ordered arm would
        # mean the attention walk swapped nothing that mattered.
        vendor_out = tmp_path / "train-vendor.json"
        full_out = tmp_path / "train-full.json"
        base = ["--device", "cuda", "--rungs", "tiny", "--out"]

        assert train_cli.main([*base, str(vendor_out), "--attention", "vendor"]) == 0
        assert train_cli.main([*base, str(full_out), "--attention", "ordered"]) == 0

        vendor = decode_run_record(load_json_str(vendor_out.read_text(encoding="utf-8")))
        full = decode_run_record(load_json_str(full_out.read_text(encoding="utf-8")))
        assert full["label"] == train_step_label(("tiny",), "both", "ordered-full")
        vendor_by_name = {o["name"]: o["value"] for o in vendor["observations"]}
        full_by_name = {o["name"]: o["value"] for o in full["observations"]}
        assert set(vendor_by_name) == set(full_by_name)
        differing = [n for n in vendor_by_name if vendor_by_name[n] != full_by_name[n]]
        # The digests are the witness, deliberately not the loss: measured
        # 2026-09-05, the tiny rung's loss is bit-EQUAL between the arms
        # while 51 of 56 tensor digests differ -- the page-title phenomenon
        # ("a loss agrees where the computation does not") repeating at the
        # training step, and an assertion on the loss would have pinned a
        # coincidence.
        assert len(differing) > len(vendor_by_name) // 2

    def test_an_undeclared_attention_arm_is_refused(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "train.json"

        args = ["--device", "cuda", "--rungs", "tiny", "--out", str(out)]

        with pytest.raises(ValueError, match="--attention"):
            train_cli.main([*args, "--attention", "math"])

        assert not out.exists()

    def test_a_swap_that_matched_nothing_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="replaced nothing"):
            train_cli.require_swapped(0)

    def test_a_real_swap_count_passes_through(self) -> None:
        assert train_cli.require_swapped(9) == 9

    def test_the_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, no_workspace: None
    ) -> None:
        out = tmp_path / "train-entry.json"
        saved = sys.argv
        sys.argv = ["ordered-train-step", "--device", "cuda", "--rungs", "tiny", "--out", str(out)]
        sys.argv += ["--attention", "vendor"]
        try:
            with pytest.raises(SystemExit) as excinfo:
                train_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0


class TestTheBenchCli:
    def test_it_prices_every_installed_shape(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
        out = tmp_path / "bench.json"

        assert bench_cli.main(["--device", "cuda", "--out", str(out)]) == 0

        results = load_json_str(out.read_text(encoding="utf-8"))
        # Narrowing blocks rather than isinstance assertions, per the
        # weak-assertion rule: a failure here is a malformed artifact, and
        # the raise names it.
        if not isinstance(results, dict):
            raise AssertionError(f"bench output must be an object, got {type(results)}")
        assert sorted(results) == sorted(gemm_label(n, s, "ms") for n, s in CHEAP)
        for row in results.values():
            if not isinstance(row, dict):
                raise AssertionError(f"each row must be an object, got {type(row)}")
            vendor = row["vendor_ms"]
            ordered = row["ordered_ms"]
            ratio = row["ratio"]
            if (
                not isinstance(vendor, float)
                or not isinstance(ordered, float)
                or not isinstance(ratio, float)
            ):
                raise AssertionError(f"timings must be floats: {row}")
            assert vendor > 0.0
            assert ordered > 0.0
            assert ratio == ordered / vendor

    def test_an_absent_device_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--device"):
            bench_cli.main(["--out", str(tmp_path / "b.json")])

    def test_the_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
        out = tmp_path / "bench-entry.json"
        saved = sys.argv
        sys.argv = ["ordered-bench", "--device", "cuda", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                bench_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_benches(
        self, tmp_path: pathlib.Path, cheap_probe: None
    ) -> None:
        module_name = "ordered_kernels.cli.bench"
        out = tmp_path / "bench-main.json"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["ordered-bench", "--device", "cuda", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0
        assert out.is_file()


def test_running_train_step_as_main_actually_steps(tmp_path: pathlib.Path) -> None:
    module_name = "ordered_kernels.cli.train_step"
    out = tmp_path / "train-main.json"
    saved_argv = sys.argv
    saved_module = sys.modules.pop(module_name, None)
    sys.argv = ["ordered-train-step", "--device", "cuda", "--rungs", "tiny", "--out", str(out)]
    sys.argv += ["--attention", "vendor"]
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved_argv
        if saved_module is not None:
            sys.modules[module_name] = saved_module

    assert excinfo.value.code == 0
    assert out.is_file()
