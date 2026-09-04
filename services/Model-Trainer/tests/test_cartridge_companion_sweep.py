"""The companion-sweep entry, exercised on a real model over fake corpora.

Same split as the sibling sweeps' suites: real tiny GPT-2, real companioned
training, real scoring; faked hub loaders, corpus reader and plan table. The
assertions concentrate on the record's names, the held-out-companion
refusals, and the provider contract -- one trained companion per seed,
shared by identity across the grid.
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
from model_trainer.cli import cartridge_companion_sweep as sweep
from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_plans import (
    COMPANION_SWEEP_EXPERIMENT,
    COMPANION_SWEEP_PLANS,
    CompanionSweepPlan,
    companion_sweep_label,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

#: A plan small enough to run in a test and shaped like the real one: two
#: probabilities so the grid has a p-axis, two counts so it has an n-axis.
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
    assert model_id_or_path == TINY_COMPANION_PLAN["model_id"]
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(model_id_or_path: str, quantization: QuantizationConfig | None) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2."""
    assert model_id_or_path == TINY_COMPANION_PLAN["model_id"]
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


def _fake_plans() -> Mapping[str, CompanionSweepPlan]:
    """Stand in for the production plan table, with one runnable plan."""
    return {"tiny": TINY_COMPANION_PLAN}


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's own name."""
    return _documents(corpus_dir.name[0])


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards."""
    measurement_hooks.companion_sweep_plans = _fake_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    measurement_hooks.companion_sweep_plans = measurement_hooks._default_companion_sweep_plans
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


class TestProviders:
    def test_the_trained_companion_is_one_object_per_seed(self) -> None:
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        from model_trainer.core.services.finetuning.strategies.cartridge import (
            require_cache_capable,
        )

        base = require_cache_capable(model)
        generator = torch.Generator()
        generator.manual_seed(31)
        train = [
            torch.randint(0, _VOCAB, (1, 8), generator=generator, dtype=torch.long)
            for _ in range(4)
        ]
        providers = sweep._CompanionProviders(base, train, TINY_COMPANION_PLAN)

        assert providers.trained(7) is providers.trained(7)
        assert providers.trained(7) is not providers.trained(8)

    def test_the_noise_companion_is_a_pure_function_of_the_seed(self) -> None:
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        from model_trainer.core.services.finetuning.strategies.cartridge import (
            require_cache_capable,
        )

        base = require_cache_capable(model)
        providers = sweep._CompanionProviders(base, [], TINY_COMPANION_PLAN)

        first = providers.noise(7).state_dict()
        second = providers.noise(7).state_dict()
        assert sorted(first) == sorted(second)
        for name, tensor in first.items():
            assert torch.equal(tensor, second[name]), name


class TestCellObservations:
    """The pure assembly, driven by constructed arms.

    Split from the grid walk precisely so this failure shape is testable:
    the real grid measured a noise-p1.0 cartridge at -0.68 alone, and the
    first CLI version died in retention() there, taking every later cell
    with it. The tiny rung cannot be forced to fail its own corpus (it
    learns positively even at learning rate 50), so the branch is driven
    with real replicate()-built arms instead.
    """

    def test_a_failed_alone_arm_keeps_its_numbers_and_drops_the_ratio(self) -> None:
        from model_trainer.core.contracts.replicated_measurement import replicate

        alone = replicate("cell-alone", [(7, -0.7), (8, -0.65), (9, -0.68)])
        composed = replicate("cell-composed", [(7, -0.6), (8, -0.66), (9, -0.7)])
        untrained = replicate("cell-untrained-composed", [(7, 0.8), (8, 0.85), (9, 0.86)])

        named = sweep.cell_observations("cell", alone, composed, untrained, ())

        values = {observation["name"]: observation["value"] for observation in named}
        assert values["cell-alone_mean"] == pytest.approx(-0.6766666666666667)
        assert "cell_retention" not in values
        # The interference verdict still computes: untrained mean 0.83667
        # minus composed mean -0.65333 is +1.49 against a 0.1 pair floor.
        assert values["cell-composed_to_cell-untrained-composed_separated"] == 1.0
        assert values["cell-composed_to_cell-untrained-composed_difference"] == pytest.approx(1.49)

    def test_a_positive_alone_arm_carries_its_ratio(self) -> None:
        from model_trainer.core.contracts.replicated_measurement import replicate

        alone = replicate("cell-alone", [(7, 0.8), (8, 0.82), (9, 0.84)])
        composed = replicate("cell-composed", [(7, 0.4), (8, 0.41), (9, 0.42)])
        untrained = replicate("cell-untrained-composed", [(7, 0.3), (8, 0.31), (9, 0.32)])
        cross = (replicate("cell-cross-0", [(7, -0.1), (8, -0.12), (9, -0.11)]),)

        named = sweep.cell_observations("cell", alone, composed, untrained, cross)

        values = {observation["name"]: observation["value"] for observation in named}
        assert values["cell_retention"] == pytest.approx(0.41 / 0.82)
        assert values["cell-cross-0_mean"] == pytest.approx(-0.11)


class TestMeasureGrid:
    def test_every_arm_is_named_once(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        observations, _digest = sweep.measure_grid(
            TINY_COMPANION_PLAN,
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpus=delta,
            device="cpu",
        )

        names = [observation["name"] for observation in observations]
        assert len(names) == len(set(names))

    def test_it_names_the_grid_the_floors_and_the_verdicts(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        observations, _digest = sweep.measure_grid(
            TINY_COMPANION_PLAN,
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpus=delta,
            device="cpu",
        )

        named = {observation["name"] for observation in observations}
        assert "slots_per_cartridge" in named
        assert "noise-p0.5-n2-alone_mean" in named
        assert "trained-p1.0-n3-composed_spread" in named
        assert "noise-p1.0-n2_retention" in named
        assert "trained-p0.5_composed_noise_floor" in named
        assert "noise-p0.5-n2-cross-0_mean" in named
        assert "trained-p1.0-n3-untrained-composed_mean" in named
        assert "noise-p0.5-n2-composed_to_noise-p0.5-n2-untrained-composed_separated" in named
        assert "trained-p1.0-n2-composed_to_trained-p1.0-n3-composed_difference" in named

    def test_too_few_other_corpora_are_refused_up_front(self, tmp_path: pathlib.Path) -> None:
        primary, beta, delta = _staged(tmp_path, ("alpha", "beta", "delta"))

        with pytest.raises(ValueError, match="needs 2 other corpora; 1 supplied"):
            sweep.measure_grid(
                TINY_COMPANION_PLAN,
                corpus=primary,
                other_corpora=[beta],
                companion_corpus=delta,
                device="cpu",
            )

    def test_a_companion_that_is_also_a_partner_is_refused(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma = _staged(tmp_path, ("alpha", "beta", "gamma"))

        with pytest.raises(ValueError, match="partner memorisation"):
            sweep.measure_grid(
                TINY_COMPANION_PLAN,
                corpus=primary,
                other_corpora=[beta, gamma],
                companion_corpus=gamma,
                device="cpu",
            )

    def test_a_companion_that_is_the_primary_is_refused(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma = _staged(tmp_path, ("alpha", "beta", "gamma"))

        with pytest.raises(ValueError, match="partner memorisation"):
            sweep.measure_grid(
                TINY_COMPANION_PLAN,
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

        record = sweep.companion_sweep_run_record(
            "tiny",
            corpus=primary,
            other_corpora=[beta, gamma],
            companion_corpus=delta,
            device="cpu",
        )

        assert record["experiment"] == COMPANION_SWEEP_EXPERIMENT
        assert record["label"].startswith(
            "tiny-tiny-under-test-w8-s3-e1-lr0.05-n2.3-c2-p0.5.1.0-seeds7.8.9-"
        )

    def test_an_unknown_plan_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="tiny"):
            sweep.companion_sweep_run_record(
                "no-such-plan",
                corpus=tmp_path,
                other_corpora=[],
                companion_corpus=tmp_path,
                device="cpu",
            )


class TestHookDefault:
    def test_the_production_hook_serves_the_declared_table(self) -> None:
        assert measurement_hooks._default_companion_sweep_plans() is COMPANION_SWEEP_PLANS


class TestProductionPlans:
    def test_the_n8_plan_matches_the_recorded_grid_on_every_shared_field(self) -> None:
        """The n8 row subtracts against the recorded grid only if the shared
        fields are identical -- a drifted window or schedule would compare
        numbers no run produced."""
        base = COMPANION_SWEEP_PLANS["gpt2-companions"]
        extended = COMPANION_SWEEP_PLANS["gpt2-companions-n8"]
        assert extended["compartment_counts"] == (8,)
        assert extended["probabilities"] == (0.25, 0.5)
        assert extended["model_id"] == base["model_id"]
        assert extended["window"] == base["window"]
        assert extended["held_out_stride"] == base["held_out_stride"]
        assert extended["slots"] == base["slots"]
        assert extended["seeds"] == base["seeds"]
        assert extended["epochs"] == base["epochs"]
        assert extended["learning_rate"] == base["learning_rate"]

    def test_the_n8_label_cannot_collide_with_the_recorded_one(self) -> None:
        label = companion_sweep_label(
            "gpt2-companions-n8",
            COMPANION_SWEEP_PLANS["gpt2-companions-n8"],
            digest="0" * 64,
        )
        assert label.startswith(
            "gpt2-companions-n8-gpt2-w256-s4-e12-lr0.01-n8-c64-p0.25.0.5-seeds7.8.9-"
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
        assert restored["experiment"] == COMPANION_SWEEP_EXPERIMENT

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
        sys.argv = ["modeltrainer-cartridge-companion-sweep", *_argv(tmp_path, others=others)]
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
        module_name = "model_trainer.cli.cartridge_companion_sweep"
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
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == COMPANION_SWEEP_EXPERIMENT
