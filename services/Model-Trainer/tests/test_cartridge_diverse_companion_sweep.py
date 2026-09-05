"""The diverse-pool sweep entry, exercised on a real model over fake corpora.

Same split as the sibling sweeps' suites. Beyond their assertions, this one
pins what makes the diverse pool diverse: each member trains on its own
corpus (members provably differ), the pool's first member is byte-identical
to the varied provider's first member when the first corpus matches (so all
three records nest), the companion-cross instrument scores every member on
the primary held-out, and the three refusals refuse -- count mismatch,
repeated corpus, and a companion that is also measured.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator, Mapping

import pytest
import torch
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import cartridge_diverse_companion_sweep as sweep
from model_trainer.cli import cartridge_varied_companion_sweep as varied_sweep
from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_plans import (
    DIVERSE_COMPANION_SWEEP_EXPERIMENT,
    DIVERSE_COMPANION_SWEEP_PLANS,
    VARIED_COMPANION_SWEEP_PLANS,
    VariedCompanionSweepPlan,
    varied_companion_sweep_label,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import CacheCapableLMProto, LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

#: A plan small enough to run in a test and shaped like the real one.
TINY_DIVERSE_PLAN: VariedCompanionSweepPlan = {
    "model_id": "tiny-under-test",
    "window": 8,
    "held_out_stride": 3,
    "compartment_counts": (2, 3),
    "slots": 2,
    "probability": 0.5,
    "max_companions": 2,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
}

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader."""
    assert model_id_or_path == TINY_DIVERSE_PLAN["model_id"]
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(model_id_or_path: str, quantization: QuantizationConfig | None) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2."""
    assert model_id_or_path == TINY_DIVERSE_PLAN["model_id"]
    assert quantization is None
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _documents(marker: str) -> tuple[str, ...]:
    """Four documents of 24 characters each, twelve windows of eight.

    Args:
        marker: Character that makes this corpus different from another.

    Returns:
        The corpus bodies.
    """
    return tuple(f"{marker}{index}" * 12 for index in range(4))


def _fake_plans() -> Mapping[str, VariedCompanionSweepPlan]:
    """Stand in for the production plan table, with one runnable plan."""
    return {"tiny": TINY_DIVERSE_PLAN}


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's own name."""
    return _documents(corpus_dir.name[0])


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards."""
    measurement_hooks.diverse_companion_sweep_plans = _fake_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    measurement_hooks.diverse_companion_sweep_plans = (
        measurement_hooks._default_diverse_companion_sweep_plans
    )
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    hf_hooks.Hooks.reset()


def _staged(tmp_path: pathlib.Path, names: tuple[str, ...]) -> list[pathlib.Path]:
    """Create one directory per corpus name.

    Args:
        tmp_path: The test's temporary directory.
        names: Directory names; the fake reader keys corpora on them.

    Returns:
        The created paths, in order.
    """
    created: list[pathlib.Path] = []
    for name in names:
        path = tmp_path / name
        path.mkdir()
        created.append(path)
    return created


def _train_windows(seed: int, rows: int) -> list[torch.Tensor]:
    """Draw deterministic training windows for a provider under test.

    Args:
        seed: Seed for the draw.
        rows: How many windows.

    Returns:
        One (1, 8) id tensor per window.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return [
        torch.randint(0, _VOCAB, (1, 8), generator=generator, dtype=torch.long) for _ in range(rows)
    ]


def _cache_capable_base() -> CacheCapableLMProto:
    """Build the tiny base the provider tests share.

    Returns:
        The cache-capable tiny GPT-2.
    """
    from model_trainer.core.services.finetuning.strategies.cartridge import (
        require_cache_capable,
    )

    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_cache_capable(model)


class TestDiversePoolProvider:
    def test_the_pool_is_one_object_per_seed_and_one_member_per_corpus(self) -> None:
        base = _cache_capable_base()
        provider = sweep._DiversePoolProvider(
            base, [_train_windows(31, 4), _train_windows(37, 4)], TINY_DIVERSE_PLAN
        )

        assert provider.pool(7) is provider.pool(7)
        assert provider.pool(7) is not provider.pool(8)
        assert len(provider.pool(7)) == TINY_DIVERSE_PLAN["max_companions"]

    def test_members_trained_on_different_corpora_differ(self) -> None:
        """The pool is diverse in fact, not only in flag order."""
        base = _cache_capable_base()
        provider = sweep._DiversePoolProvider(
            base, [_train_windows(31, 4), _train_windows(37, 4)], TINY_DIVERSE_PLAN
        )

        first, second = provider.pool(7)
        differing = [
            name
            for name, tensor in first.state_dict().items()
            if not torch.equal(tensor, second.state_dict()[name])
        ]
        assert sorted(differing) == sorted(first.state_dict())

    def test_the_first_member_nests_the_varied_pools_first_member(self) -> None:
        """With the first corpus shared, the two providers' member zero agree.

        Both derive member zero's seed from the same formula, so the diverse
        record's pool contains the varied record's first member byte for
        byte -- the three grids compare as supersets along one chain.
        """
        base = _cache_capable_base()
        shared = _train_windows(31, 4)
        diverse = sweep._DiversePoolProvider(
            base, [shared, _train_windows(37, 4)], TINY_DIVERSE_PLAN
        )
        varied = varied_sweep._PoolProvider(base, shared, TINY_DIVERSE_PLAN)

        mine = diverse.pool(7)[0].state_dict()
        theirs = varied.pool(7)[0].state_dict()
        assert sorted(mine) == sorted(theirs)
        for name, tensor in mine.items():
            assert torch.equal(tensor, theirs[name]), name


class TestMeasureGrid:
    def test_every_arm_is_named_once(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta, echo = _staged(
            tmp_path, ("alpha", "beta", "gamma", "delta", "echo")
        )

        observations, _digest = sweep.measure_grid(
            TINY_DIVERSE_PLAN,
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpora=[delta, echo],
            device="cpu",
        )

        names = [observation["name"] for observation in observations]
        assert len(names) == len(set(names))

    def test_it_names_the_cells_the_companion_crosses_and_the_floor(
        self, tmp_path: pathlib.Path
    ) -> None:
        primary, beta, gamma, delta, echo = _staged(
            tmp_path, ("alpha", "beta", "gamma", "delta", "echo")
        )

        observations, _digest = sweep.measure_grid(
            TINY_DIVERSE_PLAN,
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpora=[delta, echo],
            device="cpu",
        )

        named = {observation["name"] for observation in observations}
        assert "max_companions" in named
        assert "companion-cross-0_mean" in named
        assert "companion-cross-1_mean" in named
        assert "companion-cross-1_spread" in named
        assert "diverse-K2-p0.5-n2-alone_mean" in named
        assert "diverse-K2-p0.5-n3-composed_spread" in named
        assert "diverse-K2-p0.5-n2-cross-0_mean" in named
        assert "diverse_composed_noise_floor" in named

    def test_a_companion_count_mismatching_the_plan_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        with pytest.raises(ValueError, match="pool of 2 companions"):
            sweep.measure_grid(
                TINY_DIVERSE_PLAN,
                corpus=primary,
                other_corpora=[beta, gamma],
                companion_corpora=[delta],
                device="cpu",
            )

    def test_a_repeated_companion_corpus_is_refused(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        with pytest.raises(ValueError, match="same-content pool wearing a diverse label"):
            sweep.measure_grid(
                TINY_DIVERSE_PLAN,
                corpus=primary,
                other_corpora=[beta, gamma],
                companion_corpora=[delta, delta],
                device="cpu",
            )

    def test_a_companion_that_is_also_measured_is_refused(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        with pytest.raises(ValueError, match="partner memorisation"):
            sweep.measure_grid(
                TINY_DIVERSE_PLAN,
                corpus=primary,
                other_corpora=[beta, gamma],
                companion_corpora=[delta, gamma],
                device="cpu",
            )

    def test_too_few_other_corpora_are_refused_up_front(self, tmp_path: pathlib.Path) -> None:
        primary, beta, delta, echo = _staged(tmp_path, ("alpha", "beta", "delta", "echo"))

        with pytest.raises(ValueError, match="needs 2 other corpora; 1 supplied"):
            sweep.measure_grid(
                TINY_DIVERSE_PLAN,
                corpus=primary,
                other_corpora=[beta],
                companion_corpora=[delta, echo],
                device="cpu",
            )


class TestRunRecord:
    def test_it_carries_the_experiment_and_a_corpus_stamped_label(
        self, tmp_path: pathlib.Path
    ) -> None:
        primary, beta, gamma, delta, echo = _staged(
            tmp_path, ("alpha", "beta", "gamma", "delta", "echo")
        )

        record = sweep.diverse_companion_sweep_run_record(
            "tiny",
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpora=[delta, echo],
            device="cpu",
        )

        assert record["experiment"] == DIVERSE_COMPANION_SWEEP_EXPERIMENT
        assert record["label"].startswith(
            "tiny-tiny-under-test-w8-s3-e1-lr0.05-n2.3-c2-p0.5-K2-seeds7.8.9-"
        )

    def test_an_unknown_plan_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="tiny"):
            sweep.diverse_companion_sweep_run_record(
                "no-such-plan",
                corpus=tmp_path,
                other_corpora=[],
                companion_corpora=[],
                device="cpu",
            )


class TestHookDefault:
    def test_the_production_hook_serves_the_declared_table(self) -> None:
        assert (
            measurement_hooks._default_diverse_companion_sweep_plans()
            is DIVERSE_COMPANION_SWEEP_PLANS
        )


class TestProductionPlan:
    def test_the_diverse_plan_matches_the_varied_plan_on_every_field(self) -> None:
        """The two records isolate one difference -- the pool's voices --
        only if every knob agrees."""
        varied = VARIED_COMPANION_SWEEP_PLANS["gpt2-companions-varied"]
        diverse = DIVERSE_COMPANION_SWEEP_PLANS["gpt2-companions-diverse"]
        assert diverse == varied

    def test_the_diverse_label_cannot_collide_with_the_varied_one(self) -> None:
        label = varied_companion_sweep_label(
            "gpt2-companions-diverse",
            DIVERSE_COMPANION_SWEEP_PLANS["gpt2-companions-diverse"],
            digest="0" * 64,
        )
        assert label.startswith(
            "gpt2-companions-diverse-gpt2-w256-s4-e12-lr0.01-n4.8-c64-p0.5-K3-seeds7.8.9-"
        )

    def test_the_scale_rung_differs_from_the_recorded_plan_only_in_the_base(self) -> None:
        """The medium rung isolates parameter count: every other field --
        the schedule deliberately included -- must equal the recorded diverse
        plan, or scale is confounded with tuning."""
        recorded = DIVERSE_COMPANION_SWEEP_PLANS["gpt2-companions-diverse"]
        medium = DIVERSE_COMPANION_SWEEP_PLANS["gpt2-medium-companions-diverse"]
        assert medium["model_id"] == "gpt2-medium"
        assert {**medium, "model_id": recorded["model_id"]} == recorded


def _argv(tmp_path: pathlib.Path, *, others: str, companions: str) -> list[str]:
    """Build a complete command line against staged corpora.

    Args:
        tmp_path: The test's temporary directory.
        others: The ``--other-corpora`` value, verbatim.
        companions: The ``--companion-corpora`` value, verbatim.

    Returns:
        The flags, without a program name.
    """
    return [
        "--plan",
        "tiny",
        "--corpus",
        str(tmp_path / "alpha"),
        "--other-corpora",
        others,
        "--companion-corpora",
        companions,
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "nested" / "record.json"),
    ]


class TestMain:
    def test_it_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma", "delta", "echo"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"
        companions = f"{tmp_path / 'delta'},{tmp_path / 'echo'}"

        code = sweep.main(_argv(tmp_path, others=others, companions=companions))

        assert code == 0
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == DIVERSE_COMPANION_SWEEP_EXPERIMENT

    def test_the_companion_corpora_flag_is_required(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--companion-corpora"):
            sweep.main(
                [
                    "--plan",
                    "tiny",
                    "--corpus",
                    str(tmp_path),
                    "--other-corpora",
                    str(tmp_path),
                    "--device",
                    "cpu",
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )

    def test_a_trailing_comma_names_no_extra_corpus(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma", "delta", "echo"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"
        companions = f"{tmp_path / 'delta'},{tmp_path / 'echo'},"

        assert sweep.main(_argv(tmp_path, others=others, companions=companions)) == 0


class TestInvocationForms:
    """The console entry and `python -m` must both measure and write."""

    def test_the_console_entry_point_runs_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma", "delta", "echo"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"
        companions = f"{tmp_path / 'delta'},{tmp_path / 'echo'}"
        saved = sys.argv
        sys.argv = [
            "modeltrainer-cartridge-diverse-companion-sweep",
            *_argv(tmp_path, others=others, companions=companions),
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                sweep.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma", "delta", "echo"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"
        companions = f"{tmp_path / 'delta'},{tmp_path / 'echo'}"
        module_name = "model_trainer.cli.cartridge_diverse_companion_sweep"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", *_argv(tmp_path, others=others, companions=companions)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()
