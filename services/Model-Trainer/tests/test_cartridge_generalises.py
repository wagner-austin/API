"""Does a cartridge learn the CORPUS, or just the batch it was shown?

Every other test of this strategy shows the loss falling on the items being
trained on. That is memorisation, and a prefix with enough slots can do it
while learning nothing transferable. This file asks the question those tests
cannot: does text the cartridge was NEVER SHOWN, drawn from the same corpus,
become easier to predict?

THE CORPUS IS SYNTHETIC AND THAT IS DELIBERATE. It is a fixed token pattern
with random fillers, so "the structure of this corpus" is a thing that exists,
is learnable, and is absent from a randomly-initialised model. Real prose on a
pretrained model would confound the question: the base already knows English,
so an improvement could be the cartridge learning the corpus OR the base's own
priors being nudged, and this suite cannot separate those.

WHAT THIS THEREFORE DOES NOT SHOW. It does not show that a cartridge learns
real prose, or that it works on a pretrained model, or how it scales. It shows
the MECHANISM transfers structure from training text to held-out text, which is
the precondition for all of those and is worth failing loudly if it breaks.

The control that makes it a result rather than an artefact is the untrained
cartridge: a prefix that has been attached but not trained must score like no
prefix at all. Without that arm, "adding any prefix helps" would explain the
numbers just as well.
"""

from __future__ import annotations

import pytest
import torch

from model_trainer.core.contracts.model import CartridgeConfig, ModelTrainConfig
from model_trainer.core.services.finetuning.strategies.cartridge import CartridgeStrategy
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.cartridge_scoring import score_held_out, train_on
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES

#: The structure the corpus carries: four marker tokens in a fixed order, each
#: followed by a filler drawn from a disjoint range. A model that has learned
#: the corpus predicts the markers; the fillers are unpredictable by
#: construction, so they bound how far the loss can fall and stop the task
#: being trivially memorisable.
_PATTERN = (11, 22, 33, 44)
_FILLER_LOW = 100
_FILLER_HIGH = 140

#: Held-out rows are drawn from seeds far from the training seeds, so no row is
#: shared between the two sets.
_HELD_OUT_SEED_BASE = 1000


def _row(seed: int) -> torch.Tensor:
    """Build one corpus row: the pattern, interleaved with random fillers.

    Args:
        seed: Seed for this row's fillers, so a row is reproducible and rows
            differ from each other.

    Returns:
        Token ids shaped (1, 8).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    row = torch.empty(2 * len(_PATTERN), dtype=torch.long)
    for position, marker in enumerate(_PATTERN):
        row[2 * position] = marker
    row[1::2] = torch.randint(
        _FILLER_LOW, _FILLER_HIGH, (len(_PATTERN),), generator=generator, dtype=torch.long
    )
    return row.unsqueeze(0)


def _cfg(num_slots: int) -> ModelTrainConfig:
    """Build a config selecting the cartridge strategy.

    Args:
        num_slots: Prefix positions.

    Returns:
        The config.
    """
    return {
        "model_family": "hf_lm",
        "model_size": "tiny",
        "max_seq_len": 32,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.01,
        "tokenizer_id": None,
        "corpus_path": "",
        "corpus_format": "lines",
        "holdout_fraction": 0.1,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "precision": "fp32",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 3,
        "test_split_ratio": 0.1,
        "finetune_lr_cap": 1.0,
        "loss_mask_prefix_separator": None,
        "finetuning_strategy": "cartridge",
        "hub_model_id": "gpt2",
        "lora": None,
        "cartridge": CartridgeConfig(enabled=True, num_slots=num_slots, init_seed=7),
        "quantization": None,
        "gguf_export": None,
    }


def _adapt(num_slots: int) -> CartridgeModel:
    """Put a fresh cartridge on a fresh tiny GPT-2.

    Args:
        num_slots: Prefix positions.

    Returns:
        The cartridge-wrapped model.

    Raises:
        TypeError: If the strategy returned something else.
    """
    base, _ = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    wrapper = CartridgeStrategy().adapt(base, "gpt2", _cfg(num_slots)).model
    if not isinstance(wrapper, CartridgeModel):
        raise TypeError("the cartridge strategy must produce a CartridgeModel")
    return wrapper


def _training_rows() -> list[torch.Tensor]:
    """Rows the cartridge is trained on.

    Returns:
        Twenty-four rows.
    """
    return [_row(index) for index in range(24)]


def _held_out_rows() -> list[torch.Tensor]:
    """Rows the cartridge never sees.

    Returns:
        Twelve rows, from seeds disjoint from the training set's.
    """
    return [_row(_HELD_OUT_SEED_BASE + index) for index in range(12)]


class _Experiment:
    """One trained cartridge and the held-out measurement around it.

    Attributes:
        model: The cartridge-wrapped model, after training.
        before: The held-out comparison taken BEFORE training.
        after: The held-out comparison taken after.
        epochs: Mean training loss per epoch.
    """

    def __init__(self, num_slots: int) -> None:
        """Train a cartridge and score held-out text either side of it.

        Args:
            num_slots: Prefix positions.
        """
        self.model = _adapt(num_slots)
        held_out = _held_out_rows()
        self.before, _ = score_held_out(self.model, held_out)
        self.epochs = train_on(self.model, _training_rows(), epochs=12, learning_rate=0.05)
        self.after, self.outcomes = score_held_out(self.model, held_out)


@pytest.fixture(name="experiment", scope="module")
def _experiment() -> _Experiment:
    """Run the experiment once for the assertions that read it.

    Module-scoped because it trains a model: the claims below are about one
    run's outcome, and repeating the training per assertion would multiply the
    cost without testing anything more.

    Returns:
        The completed experiment.
    """
    return _Experiment(num_slots=8)


class TestTheCartridgeLearnsTheCorpus:
    """Held-out text from the training corpus becomes easier to predict."""

    def test_training_converges(self, experiment: _Experiment) -> None:
        """A run that did not train cannot support any claim below it."""
        assert experiment.epochs[-1] < experiment.epochs[0]

    def test_held_out_loss_is_lower_with_the_cartridge(self, experiment: _Experiment) -> None:
        """The claim: rows the cartridge never saw are predicted better."""
        assert experiment.after["mean_treatment"] < experiment.after["mean_baseline"]

    def test_the_improvement_is_not_a_split_decision(self, experiment: _Experiment) -> None:
        """A mean can improve while half the items get worse.

        The pairing is what distinguishes a real effect from an average over a
        mixture, so the direction is asserted item by item.
        """
        assert experiment.after["improved"] > experiment.after["worsened"]

    def test_the_split_is_significant(self, experiment: _Experiment) -> None:
        """Under McNemar's exact conditional test, at one percent."""
        assert experiment.after["p_value"] < 0.01

    def test_every_held_out_item_was_scored_under_both_arms(self, experiment: _Experiment) -> None:
        """A pairing that dropped an item would compare different sets."""
        assert experiment.after["items"] == len(_held_out_rows())
        assert len(experiment.outcomes) == experiment.after["items"]


class TestTheControlArms:
    """What makes it a result rather than an artefact."""

    def test_an_untrained_cartridge_shows_no_significant_direction(
        self, experiment: _Experiment
    ) -> None:
        """The arm that rules out "attaching any prefix helps".

        Before training, the prefix is drawn noise at the model's own
        initialiser scale. Measured: it moves 8 of 12 items the right way,
        which is NOT nothing and is exactly why this is asserted as a p-value
        rather than as a count. Eight of twelve is p = 0.39 under the same
        exact test -- indistinguishable from a coin, where the trained
        cartridge reaches 0.0005 on the identical rows.

        The naive version of this control, "an untrained prefix improves no
        items", fails on real numbers. Asserting it would have meant either a
        flaky test or a weakened claim.
        """
        assert experiment.before["p_value"] > 0.05

    def test_an_untrained_cartridge_is_worth_far_less_than_a_trained_one(
        self, experiment: _Experiment
    ) -> None:
        """The effect sizes are not the same order of magnitude.

        Measured on this corpus: attaching an untrained prefix moves the mean
        held-out loss by about 0.002, and training it moves the mean by about
        1.14 -- roughly six hundred times as much. The bar here is set at ten
        so it tests the separation rather than the exact ratio.
        """
        untrained_gain = experiment.before["mean_baseline"] - experiment.before["mean_treatment"]
        trained_gain = experiment.after["mean_baseline"] - experiment.after["mean_treatment"]
        assert trained_gain > 10.0 * abs(untrained_gain)

    def test_training_is_what_changed_the_held_out_score(self, experiment: _Experiment) -> None:
        """The same model, the same rows, before and after the only difference."""
        assert experiment.after["mean_treatment"] < experiment.before["mean_treatment"]

    def test_the_control_arm_is_unaffected_by_training(self, experiment: _Experiment) -> None:
        """The base scores the same before and after, because it was frozen.

        This is the strongest available statement that the comparison is about
        the prefix: the control arm is the same weights in both measurements,
        and a training run between them moved it by nothing.
        """
        assert experiment.after["mean_baseline"] == pytest.approx(
            experiment.before["mean_baseline"]
        )

    def test_the_training_rows_and_held_out_rows_are_disjoint(self) -> None:
        """The premise of the whole file, asserted rather than assumed.

        An overlap would turn every claim above into a statement about
        memorisation, and it would not be visible in any of them.
        """
        trained = _training_rows()
        held_out = _held_out_rows()
        assert not any(
            torch.equal(training_row, held_out_row)
            for training_row in trained
            for held_out_row in held_out
        )
        assert len(held_out) == 12
