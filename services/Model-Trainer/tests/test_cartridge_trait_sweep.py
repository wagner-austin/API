"""The trait-composition entry, on a real model over authored fake corpora.

THE SAME SPLIT THE COMPOSITION SWEEP'S SUITE USES, and for the same reason:
the arms are REAL -- a real tiny GPT-2, real cartridges trained and composed,
real contrastive directions read off real activations -- and the faked seams
are the hub loaders, the trait reader and the plan table, because the
production plan is GPU-minutes of cartridges over a base this suite must not
download.

THE TWO PROPERTIES ASSERTED HERE THAT NOTHING ELSE CAN CHECK are both about
ORDER. The power gate has to refuse before a model loads, because its whole
value is being cheaper than the run it prevents; and the solo precondition has
to refuse before the composed cells run, because a composed arm measured
against a solo arm indistinguishable from noise is a ratio with no reading and
the composed cells are where the hours are.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator, Mapping, Sequence

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.power_distributions import McNemarTest
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _trait_hooks as trait_hooks
from model_trainer.cli import cartridge_trait_sweep as sweep
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.contracts.replicated_measurement import replicate
from model_trainer.core.contracts.sweep_checkpoint import decode_sweep_checkpoint
from model_trainer.core.contracts.trait_corpus import TraitCorpus, TraitPairSpec
from model_trainer.core.contracts.trait_plan import TRAIT_SWEEP_EXPERIMENT, TraitPlan
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_sweep_checkpoint import checkpoint_path
from model_trainer.core.services.model.cartridge_trait_plans import TRAIT_SWEEP_PLANS
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.trait_arms import TraitArm
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

#: The site the tiny probe model actually has, tensor-valued so a direction
#: can be both read and applied there.
_SITE = "transformer.h.0.mlp.c_proj"

#: A block, whose output is a tuple. Used only to make the steering cell fail
#: on purpose, so the checkpoint's contents can be asserted.
_TUPLE_SITE = "transformer.h.0"

#: Three traits so an n3 cell exists, two counts so the step verdicts have a
#: pair to compare, and three seeds because fewer is refused. Twelve pairs at
#: stride two hold out six, which is exactly the fewest that can ever reject
#: at alpha 0.05 under the exact test -- so the declared effect is 1.0 and the
#: gate passes by the narrowest margin it can.
TINY_TRAIT_PLAN: TraitPlan = {
    "model_id": "gpt2",  # a real policy id (the fakes return a tiny GPT-2 anyway)
    "traits": ("bullets", "formal-tone", "step-by-step"),
    "held_out_stride": 2,
    "max_seq_len": 32,
    "slots": 2,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
    "compartment_counts": (2, 3),
    "smallest_effect_of_interest": 1.0,
    "alpha": 0.05,
    "mcnemar_test": McNemarTest.EXACT,
    "steering_module": _SITE,
    "steering_strength": 10.0,
}

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]

#: The committed trait corpus, for the one test that reads the real reader.
_CORPUS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "traits"


def _corpus(trait: str, marker: str) -> TraitCorpus:
    """Author one trait's pairs, short enough for the declared budget.

    Args:
        trait: The trait to declare.
        marker: A character making this trait's text unlike the others'.

    Returns:
        The corpus: twelve pairs, so the stride holds out six.
    """
    return TraitCorpus(
        trait=trait,
        pairs=[
            TraitPairSpec(
                prompt=f"p{marker}{index} ",
                expressing=f"{marker * 4}{index}",
                neutral=f"{chr(ord(marker) + 9) * 4}{index}",
            )
            for index in range(12)
        ],
    )


def _fake_trait_reader(
    corpus_dir: pathlib.Path, traits: Sequence[str], /
) -> tuple[TraitCorpus, ...]:
    """Stand in for the trait reader, returning the roster in order.

    Args:
        corpus_dir: Unused; the corpora are authored here.
        traits: The roster requested.

    Returns:
        One corpus per requested trait, in roster order.
    """
    markers = {"bullets": "a", "formal-tone": "b", "step-by-step": "c"}
    return tuple(_corpus(trait, markers[trait]) for trait in traits)


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader.

    Args:
        model_id_or_path: The id the plan declares.

    Returns:
        The fake tokenizer.
    """
    assert model_id_or_path == TINY_TRAIT_PLAN["model_id"]
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(
    model_id_or_path: str, quantization: QuantizationConfig | StoredBf16Precision | None
) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2.

    Args:
        model_id_or_path: The id the plan declares.
        quantization: Must be None for a gpt2-class id.

    Returns:
        The model.
    """
    assert model_id_or_path == TINY_TRAIT_PLAN["model_id"]
    assert quantization is None
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _refusing_model(
    model_id_or_path: str, quantization: QuantizationConfig | StoredBf16Precision | None
) -> LMModelProto:
    """A loader that fails if it is ever reached.

    Args:
        model_id_or_path: Unused.
        quantization: Unused.

    Raises:
        AssertionError: Always. The power gate must refuse before this runs.
    """
    raise AssertionError("the power gate must refuse before a model is loaded")


def _fake_plans() -> Mapping[str, TraitPlan]:
    """Stand in for the production plan table.

    Returns:
        One runnable plan.
    """
    return {"tiny": TINY_TRAIT_PLAN}


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards.

    Yields:
        None, once the fakes are installed.
    """
    measurement_hooks.trait_sweep_plans = _fake_plans
    trait_hooks.read_trait_corpora = _fake_trait_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    measurement_hooks.trait_sweep_plans = measurement_hooks._default_trait_sweep_plans
    trait_hooks.read_trait_corpora = trait_hooks._default_read_trait_corpora
    hf_hooks.Hooks.reset()


def _arm(name: str, gains: tuple[float, float, float]) -> TraitArm:
    """Build one arm from chosen numbers.

    Args:
        name: The arm's name, without a reading suffix.
        gains: Per-seed expression gains; coherence is held at zero.

    Returns:
        The arm.
    """
    seeds = TINY_TRAIT_PLAN["seeds"]
    return TraitArm(
        expression=replicate(f"{name}-expression", list(zip(seeds, gains, strict=True))),
        coherence=replicate(f"{name}-coherence", [(seed, 0.0) for seed in seeds]),
    )


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Build the flags one run takes.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The argument list.
    """
    return [
        "--plan",
        "tiny",
        "--corpus",
        str(tmp_path),
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "nested" / "record.json"),
    ]


class TestTheSoloPrecondition:
    """The arc stops here, and it stops BEFORE the composed cells."""

    def test_a_gain_larger_than_its_own_spread_passes(self) -> None:
        """Clearing the bar is not evidence of a large effect.

        Only that there is an effect to divide by, which is what every
        retention below it needs.
        """
        sweep.require_solo_precondition(
            _arm("solo", (0.90, 0.95, 1.00)), _arm("solo-untrained", (0.0, 0.0, 0.0))
        )

    def test_a_gain_inside_its_own_spread_is_refused(self) -> None:
        """The 7B failure's exact shape: a mean the seeds could have produced.

        Every composed cell divides by this gain, so the refusal has to come
        before them rather than after -- the composed cells are the hours.
        """
        with pytest.raises(AppError) as excinfo:
            sweep.require_solo_precondition(
                _arm("solo", (0.05, 0.50, 0.95)), _arm("solo-untrained", (0.0, 0.0, 0.0))
            )
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_SOLO_PRECONDITION_FAILED

    def test_the_refusal_names_the_untrained_control_too(self) -> None:
        """A solo gain that merely matches an untrained prefix is the other failure.

        A reader needs both numbers to tell the two apart, so the message
        carries the control's mean as well as the arm's.
        """
        with pytest.raises(AppError) as excinfo:
            sweep.require_solo_precondition(
                _arm("solo", (0.05, 0.50, 0.95)), _arm("solo-untrained", (0.44, 0.44, 0.44))
            )
        assert "+0.4400" in excinfo.value.message


class TestThePowerGateRunsFirst:
    """Its whole value is being cheaper than the run it prevents."""

    def test_a_plan_declaring_an_unresolvable_effect_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Six held-out pairs cannot resolve one hundredth of an item.

        Args:
            tmp_path: The test's temporary directory.
        """
        hf_hooks.Hooks.load_hf_model = _refusing_model
        plan: TraitPlan = {**TINY_TRAIT_PLAN, "smallest_effect_of_interest": 0.01}
        with pytest.raises(AppError) as excinfo:
            sweep.measure_grid(
                plan,
                plan_name="tiny",
                corpus=tmp_path,
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED

    def test_the_refusal_names_the_pairs_it_would_take(self, tmp_path: pathlib.Path) -> None:
        """A refusal that names a target is a next action.

        And the unit is PAIRS here rather than items, because a reader sent to
        count the wrong thing is sent to the wrong file.

        Args:
            tmp_path: The test's temporary directory.
        """
        hf_hooks.Hooks.load_hf_model = _refusing_model
        plan: TraitPlan = {**TINY_TRAIT_PLAN, "smallest_effect_of_interest": 0.01}
        with pytest.raises(AppError) as excinfo:
            sweep.measure_grid(
                plan,
                plan_name="tiny",
                corpus=tmp_path,
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )
        assert "pair(s)" in excinfo.value.message
        assert "600 pair(s)" in excinfo.value.message


class TestPreparingTheRoster:
    """Tokenising and splitting happen once, in roster order."""

    def test_every_trait_is_split_into_training_and_scored_pairs(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Twelve pairs at stride two is six and six, per trait.

        Args:
            tmp_path: The test's temporary directory.
        """
        corpora = _fake_trait_reader(tmp_path, TINY_TRAIT_PLAN["traits"])
        prepared = sweep.prepare_traits(corpora, TINY_TRAIT_PLAN, device="cpu")
        assert [split.trait for split in prepared] == list(TINY_TRAIT_PLAN["traits"])
        assert all(len(split.train) == 6 for split in prepared)
        assert all(len(split.held_out) == 6 for split in prepared)


class TestTheWholeGrid:
    """One run, and the rows it must carry."""

    def test_the_record_carries_every_family_of_row(self, tmp_path: pathlib.Path) -> None:
        """The gate's numbers, the solo cell, the composed cells, the steering.

        Asserted by NAME rather than by count, because a count passes while a
        whole family is missing and the name is what a later reader greps for.

        Args:
            tmp_path: The test's temporary directory.
        """
        observations, digest = sweep.measure_grid(
            TINY_TRAIT_PLAN,
            plan_name="tiny",
            corpus=tmp_path,
            device="cpu",
            checkpoints=tmp_path / "checkpoints",
        )
        names = {row["name"] for row in observations}
        assert digest
        assert {"held_out_pairs", "training_pairs", "resolvable_floor", "declared_effect"} <= names
        assert "bullets-solo-expression_mean" in names
        assert "bullets-solo-untrained-expression_mean" in names
        assert "bullets-solo-coherence_mean" in names
        assert "bullets-n2-composed-expression_mean" in names
        assert "bullets-n3-untrained-composed-expression_mean" in names
        assert "bullets-n3-cross-1-expression_mean" in names
        assert "bullets-steer-n1-expression_once" in names
        assert "bullets-steer-n3-coherence_p_value" in names
        assert "composed_expression_noise_floor" in names

    def test_the_gate_numbers_are_the_realised_ones(self, tmp_path: pathlib.Path) -> None:
        """The record must carry the count the run actually scored on.

        A plan's cap is an upper bound the corpus may fall short of, and the
        retracted question-set headline came from exactly that gap.

        Args:
            tmp_path: The test's temporary directory.
        """
        observations, _digest = sweep.measure_grid(
            TINY_TRAIT_PLAN,
            plan_name="tiny",
            corpus=tmp_path,
            device="cpu",
            checkpoints=tmp_path / "checkpoints",
        )
        values = {row["name"]: row["value"] for row in observations}
        assert values["held_out_pairs"] == 6.0
        assert values["training_pairs"] == 6.0
        assert values["resolvable_floor"] == pytest.approx(1.0)

    def test_a_completed_run_leaves_no_checkpoint_behind(self, tmp_path: pathlib.Path) -> None:
        """A leftover file is indistinguishable from an interrupted run.

        The next submission would skip cells it should have re-measured.

        Args:
            tmp_path: The test's temporary directory.
        """
        checkpoints = tmp_path / "checkpoints"
        sweep.measure_grid(
            TINY_TRAIT_PLAN,
            plan_name="tiny",
            corpus=tmp_path,
            device="cpu",
            checkpoints=checkpoints,
        )
        assert not checkpoint_path(checkpoints, "trait-tiny").exists()

    def test_an_interrupted_run_keeps_the_cells_it_finished(self, tmp_path: pathlib.Path) -> None:
        """The hours an eviction cannot take are the ones already written.

        The steering site is pointed at a block here, whose output is a tuple
        and cannot be perturbed, so the cells before it complete and the run
        dies in the last family. That is a real failure of this measurement,
        not a simulated one.

        Args:
            tmp_path: The test's temporary directory.
        """
        checkpoints = tmp_path / "checkpoints"
        plan: TraitPlan = {**TINY_TRAIT_PLAN, "steering_module": _TUPLE_SITE}
        with pytest.raises(AppError) as excinfo:
            sweep.measure_grid(
                plan,
                plan_name="tiny",
                corpus=tmp_path,
                device="cpu",
                checkpoints=checkpoints,
            )
        assert excinfo.value.code is ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED
        saved = decode_sweep_checkpoint(
            narrow_json_to_dict(
                load_json_str(
                    checkpoint_path(checkpoints, "trait-tiny").read_text(encoding="utf-8")
                )
            )
        )
        assert [cell["cell"] for cell in saved["cells"]] == ["solo", "n2", "n3"]


class TestHookDefaults:
    """Every hook this entry reads has a production implementation behind it.

    THE FAKES ABOVE REPLACE BOTH OF THEM, so without this the two production
    paths would be the only lines in the entry that nothing ever runs -- and a
    default nobody exercises is how a seam comes to point at a table that no
    longer exists.
    """

    def test_the_plan_hook_serves_the_declared_table(self) -> None:
        """The committed plans, not a copy of them."""
        assert measurement_hooks._default_trait_sweep_plans() is TRAIT_SWEEP_PLANS

    def test_the_corpus_hook_reads_the_committed_corpus(self) -> None:
        """The real reader against the real files, in roster order.

        Which also checks the two halves agree: the roster the plan declares
        and the filenames the corpus is staged under.
        """
        plan = TRAIT_SWEEP_PLANS["gpt2-traits"]
        corpora = trait_hooks._default_read_trait_corpora(_CORPUS_ROOT, plan["traits"])
        assert [corpus["trait"] for corpus in corpora] == list(plan["traits"])


class TestInvocationForms:
    """The console entry and `python -m` must both measure and write."""

    def test_main_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        """The artifact is the deliverable; a run that wrote nothing is not one.

        Args:
            tmp_path: The test's temporary directory.
        """
        assert sweep.main(_argv(tmp_path)) == 0
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == TRAIT_SWEEP_EXPERIMENT
        assert restored["label"].startswith("tiny-gpt2-traitsbullets.")

    def test_an_unknown_plan_names_the_ones_that_exist(self, tmp_path: pathlib.Path) -> None:
        """A mistyped plan's answer is nearly always the list.

        Args:
            tmp_path: The test's temporary directory.
        """
        with pytest.raises(KeyError, match="tiny"):
            sweep.trait_sweep_run_record(
                "nope",
                corpus=tmp_path,
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )

    def test_the_console_entry_point_runs_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        """The installed command must be the same measurement.

        Args:
            tmp_path: The test's temporary directory.
        """
        saved = sys.argv
        sys.argv = ["modeltrainer-cartridge-trait-sweep", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                sweep.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        """Without the ``__main__`` guard this imports, runs nothing and exits 0.

        Args:
            tmp_path: The test's temporary directory.
        """
        module_name = "model_trainer.cli.cartridge_trait_sweep"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", *_argv(tmp_path)]
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
        assert restored["experiment"] == TRAIT_SWEEP_EXPERIMENT
