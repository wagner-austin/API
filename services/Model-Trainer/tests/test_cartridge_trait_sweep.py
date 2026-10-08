"""The trait-composition entry, on a real model over authored fake corpora.

THE SAME SPLIT THE COMPOSITION SWEEP'S SUITE USES, and for the same reason:
the arms are REAL -- a real tiny GPT-2, real cartridges trained and composed,
real contrastive directions read off real activations -- and the faked seams
are the hub loaders, the trait reader and the plan table, because the
production plan is GPU-minutes of cartridges over a base this suite must not
download.

THE PROPERTY ASSERTED HERE THAT NOTHING ELSE CAN CHECK is about ORDER: the
power gate has to refuse before a model loads, because its whole value is
being cheaper than the run it prevents. The solo precondition's own ordering
-- composed cells only after it clears -- is asserted in
`test_cartridge_trait_precondition.py`.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _trait_hooks as trait_hooks
from model_trainer.cli import cartridge_trait_sweep as sweep
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.contracts.sweep_checkpoint import decode_sweep_checkpoint
from model_trainer.core.contracts.trait_plan import TRAIT_SWEEP_EXPERIMENT, TraitPlan
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_sweep_checkpoint import checkpoint_path
from model_trainer.core.services.model.cartridge_trait_plans import TRAIT_SWEEP_PLANS
from model_trainer.core.services.model.trait_roster import prepare_traits
from model_trainer.core.types import LMModelProto
from tests._trait_sweep_support import (
    TINY_TRAIT_PLAN,
    fake_trait_reader,
    install_fakes,
    restore_fakes,
)

#: A block, whose output is a tuple. Used only to make the steering cell fail
#: on purpose, so the checkpoint's contents can be asserted.
_TUPLE_SITE = "transformer.h.0"

#: The committed trait corpus, for the one test that reads the real reader.
_CORPUS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "traits"


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


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards.

    Yields:
        None, once the fakes are installed.
    """
    install_fakes()
    yield None
    restore_fakes()


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
        plan: TraitPlan = {**TINY_TRAIT_PLAN, "pair_test_floor": 0.01}
        with pytest.raises(AppError) as excinfo:
            sweep.measure_grid(
                plan,
                plan_name="tiny",
                corpus=tmp_path,
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
                merge=None,
            )
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED

    def test_too_few_seeds_for_the_sei_is_refused_before_a_model_loads(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The seed gate is the second refusal, and it is cheap for the same reason.

        Args:
            tmp_path: The test's temporary directory.
        """
        hf_hooks.Hooks.load_hf_model = _refusing_model
        plan: TraitPlan = {**TINY_TRAIT_PLAN, "pilot_paired_differences": (0.0, 0.1, 0.2)}
        with pytest.raises(AppError) as excinfo:
            sweep.measure_grid(
                plan,
                plan_name="tiny",
                corpus=tmp_path,
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
                merge=None,
            )
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED
        assert "composes over 3 seed(s)" in excinfo.value.message

    def test_the_refusal_names_the_pairs_it_would_take(self, tmp_path: pathlib.Path) -> None:
        """A refusal that names a target is a next action.

        And the unit is PAIRS here rather than items, because a reader sent to
        count the wrong thing is sent to the wrong file.

        Args:
            tmp_path: The test's temporary directory.
        """
        hf_hooks.Hooks.load_hf_model = _refusing_model
        plan: TraitPlan = {**TINY_TRAIT_PLAN, "pair_test_floor": 0.01}
        with pytest.raises(AppError) as excinfo:
            sweep.measure_grid(
                plan,
                plan_name="tiny",
                corpus=tmp_path,
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
                merge=None,
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
        corpora = fake_trait_reader(tmp_path, TINY_TRAIT_PLAN["traits"])
        prepared = prepare_traits(corpora, TINY_TRAIT_PLAN, device="cpu")
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
            merge=None,
        )
        names = {row["name"] for row in observations}
        assert digest
        assert {
            "held_out_pairs",
            "training_pairs",
            "resolvable_floor",
            "pair_test_floor",
            "smallest_effect_of_interest",
            "solo_precondition_cleared",
        } <= names
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
            merge=None,
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
            merge=None,
        )
        assert not checkpoint_path(checkpoints, "trait-tiny-anchor").exists()
        assert not checkpoint_path(checkpoints, "trait-tiny-composed").exists()

    def test_an_interrupted_run_keeps_the_cells_it_finished(self, tmp_path: pathlib.Path) -> None:
        """The hours an eviction cannot take are the ones already written.

        The steering site is pointed at a block here, whose output is a tuple
        and cannot be perturbed, so the solo cell completes and the run dies
        in the anchor phase's steering family. That is a real failure of this
        measurement, not a simulated one.

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
                merge=None,
            )
        assert excinfo.value.code is ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED
        saved = decode_sweep_checkpoint(
            narrow_json_to_dict(
                load_json_str(
                    checkpoint_path(checkpoints, "trait-tiny-anchor").read_text(encoding="utf-8")
                )
            )
        )
        assert [cell["cell"] for cell in saved["cells"]] == ["solo@b0"]


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
                merge=None,
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
