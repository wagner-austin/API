"""The varied-count sweep entry, exercised on a real model over fake corpora.

Same split as the sibling sweeps' suites: real tiny GPT-2, real varied
companioned training, real scoring; faked hub loaders, corpus reader and
plan table. The assertions concentrate on the record's names, the held-out
refusals, the provider's pool contract -- one cached pool per seed, whose
first member is byte-identical to the single-companion grid's companion for
that seed, so the pools provably nest the recorded configuration -- and the
production plan's comparability against the recorded grids.
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
from model_trainer.cli import cartridge_companion_sweep as companion_sweep
from model_trainer.cli import cartridge_varied_companion_sweep as sweep
from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_plans import (
    COMPANION_SWEEP_PLANS,
    VARIED_COMPANION_SWEEP_EXPERIMENT,
    VARIED_COMPANION_SWEEP_PLANS,
    CompanionSweepPlan,
    VariedCompanionSweepPlan,
    varied_companion_sweep_label,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

#: A plan small enough to run in a test and shaped like the real one: two
#: counts so the walk has an n-axis and the pool cache is hit, a pool of two
#: so the count draw is live.
TINY_VARIED_PLAN: VariedCompanionSweepPlan = {
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

#: The sibling plan the nesting test trains the single companion under:
#: identical on every field the companion seed formula reads.
TINY_COMPANION_PLAN: CompanionSweepPlan = {
    "model_id": "tiny-under-test",
    "window": 8,
    "held_out_stride": 3,
    "compartment_counts": (2, 3),
    "slots": 2,
    "probabilities": (0.5, 1.0),
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
}

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader."""
    assert model_id_or_path == TINY_VARIED_PLAN["model_id"]
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(model_id_or_path: str, quantization: QuantizationConfig | None) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2."""
    assert model_id_or_path == TINY_VARIED_PLAN["model_id"]
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
    return {"tiny": TINY_VARIED_PLAN}


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's own name."""
    return _documents(corpus_dir.name[0])


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards."""
    measurement_hooks.varied_companion_sweep_plans = _fake_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    measurement_hooks.varied_companion_sweep_plans = (
        measurement_hooks._default_varied_companion_sweep_plans
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


class TestPoolProvider:
    def test_the_pool_is_one_object_per_seed_and_sized_by_the_plan(self) -> None:
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        from model_trainer.core.services.finetuning.strategies.cartridge import (
            require_cache_capable,
        )

        base = require_cache_capable(model)
        provider = sweep._PoolProvider(base, _train_windows(31, 4), TINY_VARIED_PLAN)

        assert provider.pool(7) is provider.pool(7)
        assert provider.pool(7) is not provider.pool(8)
        assert len(provider.pool(7)) == TINY_VARIED_PLAN["max_companions"]

    def test_the_pools_first_member_is_the_recorded_single_companion(self) -> None:
        """The seed formula nests the recorded configuration, byte for byte.

        Member zero trains from ``seed + COMPANION_SEED_STRIDE * len(seeds)``
        -- exactly the seed the single-companion grid's trained companion
        used -- so the varied cells' pools contain the recorded companion,
        and the comparison between the two records is between supersets, not
        strangers.
        """
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        from model_trainer.core.services.finetuning.strategies.cartridge import (
            require_cache_capable,
        )

        base = require_cache_capable(model)
        windows = _train_windows(31, 4)
        pool_provider = sweep._PoolProvider(base, windows, TINY_VARIED_PLAN)
        single_provider = companion_sweep._CompanionProviders(base, windows, TINY_COMPANION_PLAN)

        member_zero = pool_provider.pool(7)[0].state_dict()
        recorded = single_provider.trained(7).state_dict()
        assert sorted(member_zero) == sorted(recorded)
        for name, tensor in member_zero.items():
            assert torch.equal(tensor, recorded[name]), name


class TestMeasureGrid:
    def test_every_arm_is_named_once(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        observations, _digest = sweep.measure_grid(
            TINY_VARIED_PLAN,
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpus=delta,
            device="cpu",
        )

        names = [observation["name"] for observation in observations]
        assert len(names) == len(set(names))

    def test_it_names_the_grid_the_floor_and_the_verdicts(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        observations, _digest = sweep.measure_grid(
            TINY_VARIED_PLAN,
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpus=delta,
            device="cpu",
        )

        named = {observation["name"] for observation in observations}
        assert "slots_per_cartridge" in named
        assert "max_companions" in named
        assert "varied-K2-p0.5-n2-alone_mean" in named
        assert "varied-K2-p0.5-n3-composed_spread" in named
        assert "varied-K2-p0.5-n2-cross-0_mean" in named
        assert "varied-K2-p0.5-n3-untrained-composed_mean" in named
        assert "varied_composed_noise_floor" in named
        assert (
            "varied-K2-p0.5-n2-composed_to_varied-K2-p0.5-n2-untrained-composed_separated" in named
        )
        assert "varied-K2-p0.5-n2-composed_to_varied-K2-p0.5-n3-composed_difference" in named

    def test_too_few_other_corpora_are_refused_up_front(self, tmp_path: pathlib.Path) -> None:
        primary, beta, delta = _staged(tmp_path, ("alpha", "beta", "delta"))

        with pytest.raises(ValueError, match="needs 2 other corpora; 1 supplied"):
            sweep.measure_grid(
                TINY_VARIED_PLAN,
                corpus=primary,
                other_corpora=[beta],
                companion_corpus=delta,
                device="cpu",
            )

    def test_a_companion_corpus_that_is_also_a_partner_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        primary, beta, gamma = _staged(tmp_path, ("alpha", "beta", "gamma"))

        with pytest.raises(ValueError, match="partner memorisation"):
            sweep.measure_grid(
                TINY_VARIED_PLAN,
                corpus=primary,
                other_corpora=[beta, gamma],
                companion_corpus=gamma,
                device="cpu",
            )

    def test_a_companion_corpus_that_is_the_primary_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        primary, beta, gamma = _staged(tmp_path, ("alpha", "beta", "gamma"))

        with pytest.raises(ValueError, match="partner memorisation"):
            sweep.measure_grid(
                TINY_VARIED_PLAN,
                corpus=primary,
                other_corpora=[beta, gamma],
                companion_corpus=primary,
                device="cpu",
            )


class TestRunRecord:
    def test_it_carries_the_experiment_and_a_corpus_stamped_label(
        self, tmp_path: pathlib.Path
    ) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        record = sweep.varied_companion_sweep_run_record(
            "tiny",
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpus=delta,
            device="cpu",
        )

        assert record["experiment"] == VARIED_COMPANION_SWEEP_EXPERIMENT
        assert record["label"].startswith(
            "tiny-tiny-under-test-w8-s3-e1-lr0.05-n2.3-c2-p0.5-K2-seeds7.8.9-"
        )

    def test_an_unknown_plan_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="tiny"):
            sweep.varied_companion_sweep_run_record(
                "no-such-plan",
                corpus=tmp_path,
                other_corpora=[],
                companion_corpus=tmp_path,
                device="cpu",
            )


class TestHookDefault:
    def test_the_production_hook_serves_the_declared_table(self) -> None:
        assert (
            measurement_hooks._default_varied_companion_sweep_plans()
            is VARIED_COMPANION_SWEEP_PLANS
        )


class TestProductionPlan:
    def test_the_varied_plan_matches_the_recorded_grids_on_every_shared_field(self) -> None:
        """The varied row subtracts against the recorded grids only if the
        shared fields are identical, and its probability must be the recipe
        cell's."""
        recorded = COMPANION_SWEEP_PLANS["gpt2-companions"]
        varied = VARIED_COMPANION_SWEEP_PLANS["gpt2-companions-varied"]
        assert varied["compartment_counts"] == (4, 8)
        assert varied["probability"] == 0.5
        assert varied["max_companions"] == 3
        assert varied["model_id"] == recorded["model_id"]
        assert varied["window"] == recorded["window"]
        assert varied["held_out_stride"] == recorded["held_out_stride"]
        assert varied["slots"] == recorded["slots"]
        assert varied["seeds"] == recorded["seeds"]
        assert varied["epochs"] == recorded["epochs"]
        assert varied["learning_rate"] == recorded["learning_rate"]

    def test_the_varied_label_cannot_collide_with_the_recorded_ones(self) -> None:
        label = varied_companion_sweep_label(
            "gpt2-companions-varied",
            VARIED_COMPANION_SWEEP_PLANS["gpt2-companions-varied"],
            digest="0" * 64,
        )
        assert label.startswith(
            "gpt2-companions-varied-gpt2-w256-s4-e12-lr0.01-n4.8-c64-p0.5-K3-seeds7.8.9-"
        )


def _argv(tmp_path: pathlib.Path, *, others: str) -> list[str]:
    """Build a complete command line against staged corpora.

    Args:
        tmp_path: The test's temporary directory.
        others: The ``--other-corpora`` value, verbatim.

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
        "--companion-corpus",
        str(tmp_path / "delta"),
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "nested" / "record.json"),
    ]


class TestMain:
    def test_it_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"

        code = sweep.main(_argv(tmp_path, others=others))

        assert code == 0
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == VARIED_COMPANION_SWEEP_EXPERIMENT

    def test_the_companion_corpus_flag_is_required(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--companion-corpus"):
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
        _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'},"

        assert sweep.main(_argv(tmp_path, others=others)) == 0


class TestInvocationForms:
    """The console entry and `python -m` must both measure and write."""

    def test_the_console_entry_point_runs_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"
        saved = sys.argv
        sys.argv = [
            "modeltrainer-cartridge-varied-companion-sweep",
            *_argv(tmp_path, others=others),
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                sweep.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"
        module_name = "model_trainer.cli.cartridge_varied_companion_sweep"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", *_argv(tmp_path, others=others)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()
