"""The three artifact readers behind ``reload_shipped_weights``.

This file exists because the defect it covers was invisible to a suite at
100% line coverage. ``_restore_best_checkpoint`` called
``model.from_pretrained(path)``, the only test of it used a fake declaring
``from_pretrained(cls, path)``, and that is exactly the signature a real
``PeftModel`` does NOT have. The line was covered; every PEFT run crashed on
it after training had finished.

A fake that matched the contract we wrote instead of the API we call is what
let that through, so nothing here is faked. Every test builds a real model,
saves a real artifact, and reads it back through the real reader.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import dump_json_str, load_json_str

from model_trainer.core.contracts.cartridge import (
    CARTRIDGE_MANIFEST_NAME,
    decode_cartridge_geometry,
    encode_cartridge_geometry,
)
from model_trainer.core.contracts.model import CartridgeConfig, ModelTrainConfig, PreparedLMModel
from model_trainer.core.encoding import Encoder, ListEncoded
from model_trainer.core.services.finetuning.strategies._hook_protocols import (
    _GetPeftModelFn,
    _LoraConfigClassProto,
    _LoraConfigProto,
)
from model_trainer.core.services.finetuning.strategies._test_hooks import (
    _default_reload_adapter_weights,
)
from model_trainer.core.services.finetuning.strategies.cartridge import CartridgeStrategy
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.backends.char_lstm.model import (
    CharLSTM,
    CharLSTMModel,
)
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    _default_load_hf_model,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.training.reload import reload_shipped_weights
from model_trainer.core.types import LMModelProto

_TINY_GPT2 = "sshleifer/tiny-gpt2"


def _cartridge_cfg() -> ModelTrainConfig:
    """Build the minimum config the cartridge strategy reads.

    Returns:
        A training config selecting the cartridge strategy.
    """
    return {
        "model_family": "hf_lm",
        "model_size": "tiny",
        "max_seq_len": 8,
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
        "finetune_lr_cap": 0.01,
        "loss_mask_prefix_separator": None,
        "finetuning_strategy": "cartridge",
        "hub_model_id": "gpt2",
        "lora": None,
        "cartridge": CartridgeConfig(enabled=True, num_slots=3, init_seed=5),
        "quantization": None,
        "gguf_export": None,
    }


class _Tok:
    """Character encoder, enough to satisfy the prepared model's field."""

    def encode(self: _Tok, text: str) -> ListEncoded:
        """Encode one id per character.

        Args:
            text: Text to encode.

        Returns:
            The encoded ids.
        """
        return ListEncoded([ord(c) for c in text])

    def token_to_id(self: _Tok, token: str) -> int | None:
        """Map a one-character token to its ordinal.

        Args:
            token: Token to look up.

        Returns:
            The ordinal, or None when the token is not one character.
        """
        return ord(token) if len(token) == 1 else None

    def get_vocab_size(self: _Tok) -> int:
        """Report the vocabulary size.

        Returns:
            The number of representable code points.
        """
        return 0x110000

    def decode(self: _Tok, ids: list[int]) -> str:
        """Decode ids back to text.

        Args:
            ids: Ids to decode.

        Returns:
            The decoded string.
        """
        return "".join(chr(i) for i in ids)


def _prepared(model: LMModelProto, *, is_peft: bool) -> PreparedLMModel:
    """Wrap a real model as a PreparedLMModel.

    Args:
        model: The live model.
        is_peft: Whether the artifact is an adapter.

    Returns:
        The prepared model.
    """
    encoder: Encoder = _Tok()
    return PreparedLMModel(
        model=model,
        tokenizer_id=None,
        eos_id=0,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=encoder,
        is_peft=is_peft,
    )


def _scatter(model: LMModelProto) -> None:
    """Overwrite every parameter so a reload has something to undo.

    Args:
        model: The model to disturb, in place.
    """
    state = model.state_dict()
    disturbed = {name: torch.full_like(tensor, 0.5) for name, tensor in state.items()}
    _ = model.load_state_dict(disturbed)


def _first_tensor(model: LMModelProto) -> torch.Tensor:
    """Return one named tensor, for comparing before and after a reload.

    Args:
        model: The model to read.

    Returns:
        The tensor under the first key in sorted order.
    """
    state = model.state_dict()
    return state[sorted(state.keys())[0]]


def _first_adapter_tensor(model: LMModelProto) -> torch.Tensor:
    """Return one LoRA tensor from a wrapped model.

    A ``PeftModel``'s state dict spans the frozen base as well as the
    adapter, but the saved artifact holds only the adapter. So the base
    weights are legitimately NOT restored by an adapter reload, and asserting
    over them would be asserting the wrong contract.

    Args:
        model: The wrapped model to read.

    Returns:
        The tensor under the first LoRA key in sorted order.
    """
    state = model.state_dict()
    lora_keys = sorted(name for name in state if "lora_" in name)
    assert lora_keys, "the wrapped model carries no LoRA parameters"
    return state[lora_keys[0]]


# THE PEFT PROTOCOLS ARE IMPORTED, NOT REDECLARED. This file used to carry its
# own `_LoraConfigClassProto`, and the copy had drifted: it omitted both
# `task_type` and `fan_in_fan_out`, so the fixture below built an adapter the
# strategy would never produce and the type checker was satisfied.
#
# What that cost was a three-per-run PEFT warning nobody could act on --
# "fan_in_fan_out is set to False but the target module is `Conv1D`" -- and,
# behind it, a reload contract exercised against a differently-shaped object.
# A protocol whose whole job is to describe a third-party signature is worth
# exactly as much as its agreement with that signature, and two copies is how
# the agreement is lost.


class _SafetensorsProto(Protocol):
    """Protocol for the safetensors.torch functions used here."""

    def load_file(self, filename: str) -> dict[str, torch.Tensor]:
        """Read a safetensors file.

        Args:
            filename: File to read.

        Returns:
            The tensors it holds.
        """
        ...

    def save_file(self, tensors: dict[str, torch.Tensor], filename: str) -> None:
        """Write a safetensors file.

        Args:
            tensors: Tensors to write.
            filename: Destination file.
        """
        ...


def _real_peft_model() -> LMModelProto:
    """Build a real LoRA-wrapped tiny GPT-2.

    Returns:
        A live PeftModel, the object whose reload contract is under test.
    """
    base = _default_load_hf_model(_TINY_GPT2, None)
    peft = __import__("peft", fromlist=["LoraConfig", "get_peft_model"])
    config_cls: _LoraConfigClassProto = peft.LoraConfig
    get_peft_model: _GetPeftModelFn = peft.get_peft_model
    # `fan_in_fan_out=True` because GPT-2's `c_attn` is a `Conv1D`, which
    # stores its weight as (fan_in, fan_out) where a `Linear` stores the
    # transpose. Omitting it built an adapter the strategy would never
    # produce: `_default_create_peft_model` passes True, so this fixture was
    # testing the reload contract against a differently-shaped object, and
    # PEFT papered over the difference by correcting the flag itself and
    # warning -- three times per run, which is how the mismatch stayed
    # invisible.
    config: _LoraConfigProto = config_cls(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM",
        # True because GPT-2's `c_attn` is a `Conv1D`, which stores its weight
        # as (fan_in, fan_out) where a `Linear` stores the transpose. This
        # mirrors `_default_create_peft_model` exactly: a fixture that builds
        # the adapter differently from the strategy is testing the reload
        # contract against an object the strategy never produces.
        fan_in_fan_out=True,
    )
    return get_peft_model(base, config)


class TestAPeftAdapterReloadsInPlace:
    """The arm that used to crash, against a real PeftModel."""

    def test_a_real_adapter_round_trips(self, tmp_path: Path) -> None:
        """Save a real adapter, disturb the live weights, read it back.

        This is the test the original suite did not have. Against a real
        ``PeftModel`` the old one-argument reconstruction raises
        ``TypeError: missing 1 required positional argument: 'model_id'``.
        """
        model = _real_peft_model()
        saved = tmp_path / "adapter"
        model.save_pretrained(str(saved))
        before = _first_adapter_tensor(model).clone()

        _scatter(model)
        assert not torch.equal(_first_adapter_tensor(model), before)

        reload_shipped_weights(_prepared(model, is_peft=True), "hf_lm", str(saved))

        assert torch.equal(_first_adapter_tensor(model), before)

    def test_the_live_object_survives_the_reload(self, tmp_path: Path) -> None:
        """Object identity is what keeps the optimizer valid."""
        model = _real_peft_model()
        saved = tmp_path / "adapter"
        model.save_pretrained(str(saved))
        prepared = _prepared(model, is_peft=True)

        reload_shipped_weights(prepared, "hf_lm", str(saved))

        assert prepared.model is model

    def test_an_adapter_from_another_model_is_refused(self, tmp_path: Path) -> None:
        """A foreign adapter carries keys this model has no slot for.

        Loading it silently would score parameters that were never trained
        here, so the reader raises instead.
        """
        model = _real_peft_model()
        saved = tmp_path / "adapter"
        model.save_pretrained(str(saved))

        # The tampered copy goes in its own directory. safetensors memory-maps
        # the file it reads, and Windows refuses to rewrite a mapped file
        # ("a file with a user-mapped section open", os error 1224).
        foreign = tmp_path / "foreign"
        shutil.copytree(saved, foreign)
        safetensors: _SafetensorsProto = __import__(
            "safetensors.torch", fromlist=["load_file", "save_file"]
        )
        tensors = safetensors.load_file(str(saved / "adapter_model.safetensors"))
        tensors["base_model.model.transformer.h.0.attn.c_attn.lora_A.stranger.weight"] = (
            torch.zeros(1)
        )
        safetensors.save_file(tensors, str(foreign / "adapter_model.safetensors"))

        with pytest.raises(AppError) as raised:
            _default_reload_adapter_weights(model, str(foreign))

        error: AppError[ModelTrainerErrorCode] = raised.value
        assert error.code is ModelTrainerErrorCode.ADAPTER_RELOAD_MISMATCH
        assert "stranger" in error.message


class TestAFullModelReloadsThroughItsOwnClass:
    """The two non-adapter formats, each read by the class that owns it."""

    @pytest.mark.parametrize("family", ["gpt2", "llama", "qwen", "hf_lm"])
    def test_a_huggingface_artifact_round_trips(
        self,
        family: Literal["gpt2", "llama", "qwen", "hf_lm"],
        tmp_path: Path,
    ) -> None:
        """All four HuggingFace families share one format and one reader."""
        model = _default_load_hf_model(_TINY_GPT2, None)
        saved = tmp_path / f"model-{family}"
        model.save_pretrained(str(saved))
        before = _first_tensor(model).clone()

        _scatter(model)
        assert not torch.equal(_first_tensor(model), before)

        reload_shipped_weights(_prepared(model, is_peft=False), family, str(saved))

        assert torch.equal(_first_tensor(model), before)

    def test_a_char_lstm_artifact_round_trips(self, tmp_path: Path) -> None:
        """A char-LSTM directory carries no config.json for the Auto class.

        Reading it with ``AutoModelForCausalLM`` fails with "Unrecognized
        model", which is what a single HuggingFace-only reader produced.
        """
        inner = CharLSTM(
            vocab_size=10,
            embed_dim=8,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            max_seq_len=8,
        )
        model = CharLSTMModel(inner)
        saved = tmp_path / "char-lstm"
        model.save_pretrained(str(saved))
        before = _first_tensor(model).clone()

        _scatter(model)
        assert not torch.equal(_first_tensor(model), before)

        reload_shipped_weights(_prepared(model, is_peft=False), "char_lstm", str(saved))

        assert torch.equal(_first_tensor(model), before)


class TestTheCartridgeReader:
    """The fourth artifact format, added the way the third was: by crashing.

    A cartridge directory holds a manifest and a block of key-value tensors.
    ``AutoModelForCausalLM`` refuses it with "Unrecognized model", exactly as
    it refuses a char-LSTM checkpoint, so a cartridge run trained to
    completion and then died in ``_restore_best_checkpoint`` -- after the
    whole run had been spent, which is the failure mode this module's
    docstring was written about.

    Nothing here is faked, per this file's standing rule: a real cartridge is
    trained, saved, scattered and read back through the real reader.
    """

    def _cartridge_prepared(self) -> tuple[PreparedLMModel, CartridgeModel]:
        """Build a real cartridge over a real GPT-2, prepared for the trainer.

        Returns:
            The prepared model and the cartridge wrapper inside it.

        Raises:
            TypeError: If the strategy returned something else.
        """
        base, _ = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        adapted = CartridgeStrategy().adapt(base, "gpt2", _cartridge_cfg())
        wrapper = adapted.model
        if not isinstance(wrapper, CartridgeModel):
            raise TypeError("the cartridge strategy must produce a CartridgeModel")
        encoder: Encoder = _Tok()
        return (
            PreparedLMModel(
                model=wrapper,
                tokenizer_id=None,
                eos_id=0,
                pad_id=0,
                max_seq_len=8,
                tok_for_dataset=encoder,
                is_peft=False,
                strategy_name="cartridge",
                hub_model_id="gpt2",
                quantization=None,
            ),
            wrapper,
        )

    def test_a_saved_cartridge_reads_back_into_the_live_model(self, tmp_path: Path) -> None:
        """The path that crashed: save, scatter, restore.

        The blocks are overwritten between the save and the read, so a reader
        that quietly did nothing would leave the scattered values in place and
        fail this.
        """
        prepared, wrapper = self._cartridge_prepared()
        saved = {name: tensor.detach().clone() for name, tensor in wrapper.named_parameters()}
        wrapper.save_pretrained(str(tmp_path))

        for _, tensor in wrapper.named_parameters():
            tensor.detach().fill_(99.0)

        reload_shipped_weights(prepared, "hf_lm", str(tmp_path))

        assert all(
            torch.equal(saved[name], tensor.detach()) for name, tensor in wrapper.named_parameters()
        )

    def test_the_reader_writes_into_the_model_the_trainer_already_holds(
        self, tmp_path: Path
    ) -> None:
        """In place, not by substitution.

        The cartridge IS the trainable state, so an optimizer built before the
        restore must still be stepping the tensors the restore wrote. Asserted
        by identity of the prepared model's object across the call.
        """
        prepared, wrapper = self._cartridge_prepared()
        wrapper.save_pretrained(str(tmp_path))

        reload_shipped_weights(prepared, "hf_lm", str(tmp_path))

        assert prepared.model is wrapper

    def test_a_cartridge_from_a_differently_shaped_model_is_refused(self, tmp_path: Path) -> None:
        """The geometry check runs on the restore path too.

        A saved cartridge whose manifest describes another model would
        otherwise be installed silently, and the run would finish scoring a
        prefix built for something else.
        """
        prepared, wrapper = self._cartridge_prepared()
        wrapper.save_pretrained(str(tmp_path))

        manifest = tmp_path / CARTRIDGE_MANIFEST_NAME
        widened = decode_cartridge_geometry(load_json_str(manifest.read_text(encoding="utf-8")))
        widened["num_kv_heads"] = widened["num_kv_heads"] + 1
        manifest.write_text(dump_json_str(encode_cartridge_geometry(widened)), encoding="utf-8")

        with pytest.raises(AppError) as excinfo:
            reload_shipped_weights(prepared, "hf_lm", str(tmp_path))
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH

    def test_a_directory_holding_no_cartridge_is_refused(self, tmp_path: Path) -> None:
        """An empty artifact directory must not read as an empty cartridge."""
        prepared, _ = self._cartridge_prepared()
        with pytest.raises(FileNotFoundError):
            reload_shipped_weights(prepared, "hf_lm", str(tmp_path))
