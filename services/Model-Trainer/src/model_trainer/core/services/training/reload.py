"""Reload a run's shipped weights into the model already in memory.

WHY THIS IS NOT ``model.from_pretrained(path)``. That call was polymorphic
over the model class, which worked for the two backends whose artifact IS a
model and broke for the one whose artifact is not. A PEFT directory holds an
adapter, a delta, so reconstructing needs the base model too:
``PeftModel.from_pretrained`` takes ``(model, model_id)``. Called with a path
alone it binds the path to ``model`` and raises ``TypeError: missing 1
required positional argument: 'model_id'`` after a whole run has been spent.

So there are FOUR artifact formats, not two, and the polymorphism hid that
rather than handling it:

- a PEFT adapter, read by writing its state dict into the live wrapper
- a cartridge, a manifest and a block of key-value tensors, which is not a
  model either and which ``AutoModelForCausalLM`` rejects the same way
- a HuggingFace model directory, read by ``AutoModelForCausalLM``
- a char-LSTM checkpoint, which is not HuggingFace-shaped at all and whose
  directory ``AutoModelForCausalLM`` rejects with "Unrecognized model"

The cartridge was the fourth, added 2026-09-03, and it arrived the way the
docstring above predicts: a cartridge run trained to completion and then died
in ``_restore_best_checkpoint`` with "Unrecognized model in
.../cartridge-probe", after the whole run had been spent. The same failure
this module was written about, one format later.

The format is a property of what WROTE it, so the reader is chosen by the
strategy that produced the artifact and, for the backends that predate
strategies, by ``model_family``. This is a dispatch over real formats, not a
fallback chain: nothing is tried after something else fails.

Every reader writes into the live model rather than returning a new one. The
trainer scores the reloaded weights with an optimizer, a device placement
and references that all point at the object it already holds.
"""

from __future__ import annotations

from typing import Literal, Protocol

from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.services.finetuning.strategies._test_hooks import (
    Hooks as FinetuningHooks,
)
from model_trainer.core.types import LMModelProto


class _SelfLoadingModelClassProto(Protocol):
    """Protocol for a model class that rebuilds itself from a directory.

    Satisfied by ``AutoModelForCausalLM`` and by ``CharLSTMModel``. NOT
    satisfied by ``PeftModel``, whose ``from_pretrained`` needs a base model
    as well, which is the distinction this module exists to respect.
    """

    def from_pretrained(self, path: str) -> LMModelProto:
        """Rebuild a model from a saved directory.

        Args:
            path: Directory holding the saved model.

        Returns:
            The rebuilt model.
        """
        ...


def _copy_state_into(model: LMModelProto, reloaded: LMModelProto) -> None:
    """Copy a rebuilt model's tensors into the live one.

    Rebuilding and then copying, rather than swapping the object, is what
    keeps the optimizer and device placement valid. Reading from disk rather
    than from an in-memory snapshot is what makes a partial or corrupt save
    surface here instead of in whatever consumes the artifact later.

    Args:
        model: The live model to write into.
        reloaded: The model just rebuilt from disk.
    """
    _ = model.load_state_dict(reloaded.state_dict())


def _reload_hf_model(model: LMModelProto, path: str) -> None:
    """Reload a HuggingFace model directory into the live model.

    Args:
        model: The live model to write into.
        path: Directory holding the saved model.
    """
    transformers = __import__("transformers", fromlist=["AutoModelForCausalLM"])
    model_cls: _SelfLoadingModelClassProto = transformers.AutoModelForCausalLM
    _copy_state_into(model, model_cls.from_pretrained(path))


def _reload_char_lstm(model: LMModelProto, path: str) -> None:
    """Reload a char-LSTM checkpoint into the live model.

    Kept separate from the HuggingFace reader because a char-LSTM artifact
    carries no ``config.json`` with a ``model_type``, so the Auto class
    refuses it by name rather than misreading it.

    Args:
        model: The live model to write into.
        path: Directory holding the saved model.
    """
    from model_trainer.core.services.model.backends.char_lstm.model import CharLSTMModel

    _copy_state_into(model, CharLSTMModel.from_pretrained(path))


def _reload_cartridge(model: LMModelProto, path: str) -> None:
    """Reload a saved key-value prefix into the live model.

    In place, like every reader here, and for a reason specific to this one:
    the cartridge IS the trainable state, so swapping the object would leave
    the optimizer holding the tensors of a cartridge nobody is using any more
    and stepping them into the void.

    The blocks travel through the model's own ``load_state_dict``, which
    checks them against the geometry it was built with. A cartridge saved from
    a differently shaped model is refused here rather than silently attached.

    Args:
        model: The live cartridge-wrapped model to write into.
        path: Directory holding the saved cartridge.
    """
    _ = model.load_state_dict(FinetuningHooks.load_cartridge(path).state_dict())


def reload_shipped_weights(
    prepared: PreparedLMModel,
    model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"],
    path: str,
) -> None:
    """Load the artifact at ``path`` into ``prepared``'s live model, in place.

    Args:
        prepared: The prepared model whose weights are being replaced. Its
            ``strategy_name`` and ``is_peft`` flag record which shape the
            hf_lm backend wrote.
        model_family: The backend that wrote the artifact, which is what
            determines its on-disk format for the families that predate
            strategies.
        path: Directory the artifact was saved to.
    """
    if prepared.strategy_name == "cartridge":
        _reload_cartridge(prepared.model, path)
        return
    if prepared.is_peft:
        FinetuningHooks.reload_adapter_weights(prepared.model, path)
        return
    if model_family == "char_lstm":
        _reload_char_lstm(prepared.model, path)
        return
    _reload_hf_model(prepared.model, path)


__all__ = ["reload_shipped_weights"]
