"""The solo precondition: decided first, recorded either way, and binding.

THE ARC STOPS AT A FAILED SOLO ARM, and the property that matters is ORDER:
the composed cells are where the hours are, and a composed retention measured
against a solo gain indistinguishable from its own noise is a division
artefact -- the corpus programme's 7B rung is the recorded case. So these
tests assert that a failing solo arm reaches a RECORD (its verdict, its rows
and the steering rows, which divide by nothing the cartridge produced) and
that no composed cell runs behind it.

THE FAILING CORPUS IS REAL, NOT A STUB. The training members of the primary
trait carry the marker its held-out pairs use for their NEUTRAL member, so a
cartridge trained on them leans away from the trait on every held-out pair:
its mean expression gain is negative, below any spread, on a real tiny GPT-2.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Sequence

import pytest

from model_trainer.cli import _trait_hooks as trait_hooks
from model_trainer.cli import cartridge_trait_sweep as sweep
from model_trainer.core.contracts.replicated_measurement import replicate
from model_trainer.core.contracts.trait_corpus import TraitCorpus, TraitPairSpec
from model_trainer.core.services.model.cartridge_sweep_checkpoint import checkpoint_path
from model_trainer.core.services.model.trait_arms import solo_precondition_cleared
from tests._trait_sweep_support import (
    TINY_TRAIT_PLAN,
    install_fakes,
    restore_fakes,
    trait_corpus,
)


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards.

    Yields:
        None, once the fakes are installed.
    """
    install_fakes()
    yield None
    restore_fakes()


def _inverted_bullets() -> TraitCorpus:
    """Author a primary trait whose training teaches the opposite of its scoring.

    The stride holds out the EVEN pairs and trains on the odd ones. Odd pairs
    express with ``aaaa``; even pairs express with ``jjjj`` and carry ``aaaa``
    as their NEUTRAL member. Training on ``aaaa`` therefore makes every
    held-out neutral member easier, which is a preference AWAY from the trait.

    Returns:
        Twelve pairs, six of them held out.
    """
    return TraitCorpus(
        trait="bullets",
        pairs=[
            TraitPairSpec(
                prompt=f"pa{index} ",
                expressing=f"{'aaaa' if index % 2 else 'jjjj'}{index}",
                neutral=f"{'jjjj' if index % 2 else 'aaaa'}{index}",
            )
            for index in range(12)
        ],
    )


def _inverted_reader(corpus_dir: pathlib.Path, traits: Sequence[str], /) -> tuple[TraitCorpus, ...]:
    """Serve the roster with the primary trait inverted.

    Args:
        corpus_dir: Unused; the corpora are authored here.
        traits: The roster requested.

    Returns:
        One corpus per requested trait, the first inverted.
    """
    return (_inverted_bullets(), *(trait_corpus(trait) for trait in traits[1:]))


class TestTheVerdict:
    """The bar is the arm's own spread, the weakest defensible one."""

    def test_a_gain_larger_than_its_own_spread_clears(self) -> None:
        """Clearing the bar is not evidence of a large effect.

        Only that there is an effect to divide by, which is what every
        retention below it needs.
        """
        gain = replicate("solo-expression", [(7, 0.90), (8, 0.95), (9, 1.00)])
        assert solo_precondition_cleared(gain) is True

    def test_a_gain_inside_its_own_spread_fails(self) -> None:
        """The 7B failure's exact shape: a mean the seeds could have produced."""
        gain = replicate("solo-expression", [(7, 0.05), (8, 0.50), (9, 0.95)])
        assert solo_precondition_cleared(gain) is False

    def test_a_gain_equal_to_its_spread_fails(self) -> None:
        """Equality is not clearing: the bar is strict, because a tie is noise."""
        gain = replicate("solo-expression", [(7, 0.5), (8, 1.0), (9, 1.5)])
        assert gain["mean"] == gain["spread"] == 1.0
        assert solo_precondition_cleared(gain) is False


class TestTheLogSentence:
    """A reader of the job log needs the verdict, the gain and the control."""

    def test_a_cleared_arm_says_so_with_both_numbers(self) -> None:
        """The control is named because matching it is the other failure."""
        sentence = sweep.describe_solo_precondition(
            replicate("solo-expression", [(7, 0.90), (8, 0.95), (9, 1.00)]),
            replicate("solo-untrained-expression", [(7, 0.44), (8, 0.44), (9, 0.44)]),
        )
        assert sentence.startswith("solo precondition CLEARED: 'solo-expression'")
        assert "+0.9500" in sentence
        assert "spread 0.1000" in sentence
        assert sentence.endswith("control sits at +0.4400")

    def test_a_failed_arm_says_so(self) -> None:
        """The word a reader greps for is the verdict."""
        sentence = sweep.describe_solo_precondition(
            replicate("solo-expression", [(7, 0.05), (8, 0.50), (9, 0.95)]),
            replicate("solo-untrained-expression", [(7, 0.0), (8, 0.0), (9, 0.0)]),
        )
        assert sentence.startswith("solo precondition FAILED:")


class TestTheGridStopsAtAFailedSoloArm:
    """A failed precondition is a recorded result, and nothing composes behind it."""

    def test_the_record_carries_the_verdict_the_anchor_and_no_composed_cell(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The solo and steering rows are there; the composed rows are not.

        Args:
            tmp_path: The test's temporary directory.
        """
        trait_hooks.read_trait_corpora = _inverted_reader
        observations, _digest = sweep.measure_grid(
            TINY_TRAIT_PLAN,
            plan_name="tiny",
            corpus=tmp_path,
            device="cpu",
            checkpoints=tmp_path / "checkpoints",
            merge=None,
        )
        values = {row["name"]: row["value"] for row in observations}
        assert values["solo_precondition_cleared"] == 0.0
        assert values["bullets-solo-expression_mean"] < 0.0
        untrained_seeds = [
            values[f"bullets-solo-untrained-expression_seed{seed}_gain"]
            for seed in TINY_TRAIT_PLAN["seeds"]
        ]
        assert values["bullets-solo-untrained-expression_mean"] == pytest.approx(
            sum(untrained_seeds) / len(untrained_seeds)
        )
        # Every steering count ran, each scored on all six held-out pairs.
        for count in (1, 2, 3):
            assert values[f"bullets-steer-n{count}-expression_items"] == 6.0
            assert values[f"bullets-steer-n{count}-coherence_items"] == 6.0
        assert not [name for name in values if name.startswith("bullets-n")]
        assert "composed_expression_noise_floor" not in values

    def test_a_cleared_arm_is_recorded_as_cleared(self, tmp_path: pathlib.Path) -> None:
        """The other branch, on the corpus whose training teaches the trait.

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
        assert values["solo_precondition_cleared"] == 1.0
        assert values["bullets-solo-expression_mean"] > values["bullets-solo-expression_spread"]

    def test_a_stopped_run_leaves_no_checkpoint_behind(self, tmp_path: pathlib.Path) -> None:
        """The anchor phase completed, so its file is gone; no composed file was made.

        Args:
            tmp_path: The test's temporary directory.
        """
        trait_hooks.read_trait_corpora = _inverted_reader
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
