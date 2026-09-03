"""A cartridge run through the REAL trainer, end to end.

The strategy's own tests drive a hand-rolled optimizer loop. This drives
``BaseTrainer.train()``: dataset construction, epochs, gradient clipping,
validation, checkpointing, best-checkpoint restore, and the manifest. Those are
the parts a strategy does not control and cannot test from the inside, and the
first run of this found a real defect -- ``_restore_best_checkpoint`` read the
artifact with ``AutoModelForCausalLM``, which refuses a cartridge directory, so
the run trained to completion and then died.

The claim it establishes is the one the whole strategy exists for, made against
the real loop rather than a loop written for the test: a training run leaves
every base weight byte-identical.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.contracts.cartridge import (
    CARTRIDGE_MANIFEST_NAME,
    CARTRIDGE_WEIGHTS_NAME,
)
from model_trainer.core.contracts.model import (
    CartridgeConfig,
    ModelTrainConfig,
    PreparedLMModel,
)
from model_trainer.core.encoding import Encoder, ListEncoded
from model_trainer.core.services.finetuning.strategies.cartridge import CartridgeStrategy
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.training.base_trainer import BaseTrainer
from model_trainer.core.types import NamedParameter, TracedLMModelProto
from tests.conftest import UNPINNED

#: Vocabulary the tiny rung was built with, which the corpus must stay inside.
_VOCAB = 512


class _ByteEncoder:
    """Encoder mapping characters into the tiny rung's vocabulary.

    Real enough for the dataset builder: it produces ids the model's embedding
    actually has, which is what makes the loss a number rather than an index
    error.
    """

    def encode(self: _ByteEncoder, text: str) -> ListEncoded:
        """Encode one id per character.

        Args:
            text: Text to encode.

        Returns:
            The ids, inside the model's vocabulary.
        """
        return ListEncoded([ord(character) % _VOCAB for character in text] or [1])

    def decode(self: _ByteEncoder, ids: list[int]) -> str:
        """Decode ids back to characters.

        Args:
            ids: Ids to decode.

        Returns:
            The decoded text.
        """
        return "".join(chr(value) for value in ids)

    def token_to_id(self: _ByteEncoder, token: str) -> int | None:
        """Map a token to an id.

        Args:
            token: The token.

        Returns:
            Its id, inside the vocabulary.
        """
        return ord(token[0]) % _VOCAB if token else None

    def get_vocab_size(self: _ByteEncoder) -> int:
        """Return the vocabulary size.

        Returns:
            The size the tiny rung was built with.
        """
        return _VOCAB


def _cfg(corpus_path: Path) -> ModelTrainConfig:
    """Build a training config for a short cartridge run.

    Args:
        corpus_path: File the dataset is read from.

    Returns:
        The config.
    """
    return {
        "model_family": "hf_lm",
        "model_size": "tiny",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.05,
        "tokenizer_id": None,
        "corpus_path": str(corpus_path),
        "corpus_format": "lines",
        "holdout_fraction": 0.2,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "precision": "fp32",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 5,
        "test_split_ratio": 0.2,
        "finetune_lr_cap": 1.0,
        "loss_mask_prefix_separator": None,
        "finetuning_strategy": "cartridge",
        "hub_model_id": "gpt2",
        "lora": None,
        "cartridge": CartridgeConfig(enabled=True, num_slots=4, init_seed=7),
        "quantization": None,
        "gguf_export": None,
    }


def _corpus(tmp_path: Path) -> Path:
    """Write a small corpus for the run to read.

    Args:
        tmp_path: Directory to write into.

    Returns:
        The corpus file.
    """
    path = tmp_path / "corpus.txt"
    path.write_text(
        "\n".join(f"line number {index} about the wiki" for index in range(40)),
        encoding="utf-8",
    )
    return path


class _Run:
    """One completed cartridge training run and what it left behind.

    Attributes:
        base: The frozen transformer.
        wrapper: The cartridge-wrapped model the trainer drove.
        snapshot: Every base weight as it stood before training.
        settings: The settings the run used.
    """

    base: TracedLMModelProto
    wrapper: CartridgeModel
    snapshot: dict[str, torch.Tensor]
    settings: Settings

    def __init__(self, tmp_path: Path) -> None:
        """Adapt a real model and prepare it for the trainer.

        Args:
            tmp_path: Directory the corpus is written into.

        Raises:
            TypeError: If the strategy returned something else.
        """
        base, _ = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        self.base = base
        self.config = _cfg(_corpus(tmp_path))
        adapted = CartridgeStrategy().adapt(base, "gpt2", self.config)
        wrapper = adapted.model
        if not isinstance(wrapper, CartridgeModel):
            raise TypeError("the cartridge strategy must produce a CartridgeModel")
        self.wrapper = wrapper
        self.snapshot = {
            name: parameter.detach().clone() for name, parameter in base.named_parameters()
        }
        self.settings = load_settings()

    def trainer(self, run_id: str) -> BaseTrainer:
        """Build the real trainer over this run's prepared model.

        Args:
            run_id: Identifier for the run.

        Returns:
            The trainer.
        """
        encoder: Encoder = _ByteEncoder()
        prepared = PreparedLMModel(
            model=self.wrapper,
            tokenizer_id=None,
            eos_id=1,
            pad_id=0,
            max_seq_len=16,
            tok_for_dataset=encoder,
            is_peft=False,
            strategy_name="cartridge",
            hub_model_id="gpt2",
            quantization=None,
        )
        return BaseTrainer(
            prepared,
            self.config,
            self.settings,
            run_id=run_id,
            redis_hb=lambda _: None,
            cancelled=lambda: False,
            resume=False,
            progress=None,
            service_name="cartridge-run",
            determinism=UNPINNED,
        )

    def base_parameters(self) -> list[tuple[str, NamedParameter]]:
        """Return the frozen model's named parameters.

        Returns:
            The pairs, in the model's own order.
        """
        return list(self.base.named_parameters())


@pytest.fixture(name="completed")
def _completed(tmp_path: Path) -> tuple[_Run, float, str]:
    """Run one cartridge training run to completion.

    Shared across the assertions below so the loop runs once rather than once
    per claim; the claims are about one run's outcome, not about repetition.

    Args:
        tmp_path: Directory for the corpus.

    Returns:
        The run, its final loss, and its output directory.
    """
    run = _Run(tmp_path)
    outcome = run.trainer("cartridge-integration").train()
    return run, outcome["loss"], outcome["out_dir"]


class TestARealTrainingRun:
    """What ``BaseTrainer.train()`` does with a cartridge."""

    def test_it_completes_and_reports_a_finite_loss(
        self, completed: tuple[_Run, float, str]
    ) -> None:
        """The whole loop runs: dataset, epochs, validation, restore, manifest."""
        _, loss, _ = completed
        assert loss > 0.0
        assert loss == loss

    def test_it_leaves_every_base_weight_byte_identical(
        self, completed: tuple[_Run, float, str]
    ) -> None:
        """The claim the strategy exists for, through the real loop.

        Not the hand-rolled optimizer of the strategy's own tests: this is
        after gradient clipping, checkpoint writes, a validation pass and the
        best-checkpoint restore, any of which could have written a weight.
        """
        run, _, _ = completed
        assert all(
            torch.equal(run.snapshot[name], parameter.detach())
            for name, parameter in run.base_parameters()
        )

    def test_the_artifact_is_a_cartridge_and_not_a_model(
        self, completed: tuple[_Run, float, str]
    ) -> None:
        """A cartridge run ships the prefix, and names its base rather than
        storing an unmodified copy of it."""
        _, _, out_dir = completed
        written = sorted(path.name for path in Path(out_dir).iterdir())
        assert CARTRIDGE_MANIFEST_NAME in written
        assert CARTRIDGE_WEIGHTS_NAME in written
        assert "model.safetensors" not in written
        assert "pytorch_model.bin" not in written

    def test_the_base_stays_frozen_through_the_run(
        self, completed: tuple[_Run, float, str]
    ) -> None:
        """Nothing in the loop re-enables gradients on the base.

        ``freeze_embed`` and the checkpoint restore both touch parameters, and
        either could have flipped a flag back.
        """
        run, _, _ = completed
        assert all(not parameter.requires_grad for _, parameter in run.base_parameters())

    def test_the_trained_cartridge_is_what_was_saved(
        self, completed: tuple[_Run, float, str]
    ) -> None:
        """The blocks in memory at the end match the blocks on disk.

        This is what makes the artifact the run's result rather than a
        snapshot from some earlier step.
        """
        run, _, out_dir = completed
        loaded: dict[str, torch.Tensor] = torch.load(
            Path(out_dir) / CARTRIDGE_WEIGHTS_NAME, weights_only=True
        )
        assert all(
            torch.equal(loaded[name], tensor.detach())
            for name, tensor in run.wrapper.named_parameters()
        )
