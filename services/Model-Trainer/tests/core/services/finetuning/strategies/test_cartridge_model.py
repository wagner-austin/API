"""The wrapper that puts a trained prefix in front of a frozen model.

Driven against a fake base, because what is under test is what the wrapper
PASSES: the cache it builds, the mask it sizes, the labels it declines to
widen, and the cache accumulation it switches off. Each of those is a value a
real model would consume silently, so asserting the values is the only way to
catch them. That the values are the right ones for a real transformer is
established separately, against one, in ``test_cartridge_strategy``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator, Sequence
from pathlib import Path

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.cartridge import (
    CARTRIDGE_MANIFEST_NAME,
    CARTRIDGE_WEIGHTS_NAME,
    CartridgeGeometry,
)
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks, reset_hooks
from model_trainer.core.services.finetuning.strategies.cartridge_model import (
    CartridgeLoadResult,
    CartridgeModel,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import initialise_slots
from model_trainer.core.types import KVCacheProto
from tests.core.services.finetuning.testing import FakeCacheCapableModel

_GEOMETRY = CartridgeGeometry(num_layers=2, num_kv_heads=4, head_dim=8, num_slots=3)


class _EmptyCache(KVCacheProto):
    """A cache carrying nothing, for tests that only inspect what was built."""


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    yield
    reset_hooks()


def wrapped() -> tuple[CartridgeModel, FakeCacheCapableModel]:
    """Build a wrapper over a fake base.

    Returns:
        The wrapper and the base it wraps.
    """
    base = FakeCacheCapableModel(num_layers=2, num_kv_heads=4, head_dim=8)
    return CartridgeModel(base=base, slots=initialise_slots(_GEOMETRY, seed=1)), base


def ids(*, batch_size: int, length: int) -> torch.Tensor:
    """Build a token batch.

    Args:
        batch_size: Rows.
        length: Positions per row.

    Returns:
        Token ids.
    """
    return torch.zeros((batch_size, length), dtype=torch.long)


class TestWhatReachesTheBase:
    """The four arguments the wrapper decides on the caller's behalf."""

    def test_the_mask_covers_the_prefix_and_the_input_together(self) -> None:
        """A mask covering only the input raises a shape error on a real model.

        Both the length and the contents are asserted: a mask of the right
        length holding a zero anywhere over the prefix would mask a slot out
        of attention, so the cartridge would be present and partly ignored.
        """
        model, base = wrapped()
        model.forward(input_ids=ids(batch_size=2, length=6), labels=ids(batch_size=2, length=6))
        mask = base.calls[-1]["attention_mask"]
        if mask is None:
            raise TypeError("a prefix run must supply an attention mask")
        assert tuple(mask.shape) == (2, 9)
        assert int(mask.sum().item()) == 18

    def test_the_cache_is_built_from_the_slots_of_every_layer(self) -> None:
        """The prefix passed is this cartridge's own blocks, not a fresh draw.

        The builder is swapped for one that records what it was handed, so the
        assertion is on the actual tensors rather than on the opaque cache
        object they end up inside. A wrapper that drew new blocks per forward,
        or that passed layer zero's blocks for every layer, would train a
        cartridge whose saved weights do not describe what ran.
        """
        recorded: list[tuple[torch.Tensor, torch.Tensor]] = []

        def _record(blocks: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> KVCacheProto:
            recorded.extend(blocks)
            return _EmptyCache()

        model, _ = wrapped()
        Hooks.build_prefix_cache = _record
        model.forward(input_ids=ids(batch_size=1, length=4), labels=ids(batch_size=1, length=4))

        expected = [tensor for _, tensor in model.named_parameters()]
        passed = [tensor for pair in recorded for tensor in pair]
        assert len(recorded) == 2
        assert all(
            torch.equal(was.detach(), now.detach())
            for was, now in zip(expected, passed, strict=True)
        )

    def test_the_forward_does_not_ask_the_base_to_accumulate_a_cache(self) -> None:
        """Otherwise the cache grows by the sequence length on every step."""
        model, base = wrapped()
        model.forward(input_ids=ids(batch_size=1, length=4), labels=ids(batch_size=1, length=4))
        assert base.calls[-1]["use_cache"] is False
        assert int(base.calls[-1]["input_ids"].sum().item()) == 0

    def test_the_labels_reach_the_base_unwidened(self) -> None:
        """Logits come back at the input's length, so widening would misalign them.

        Asserted on the tensor's identity and its extent: a copy of the right
        length would pass an identity check on shape alone.
        """
        model, base = wrapped()
        labels = ids(batch_size=1, length=4)
        model.forward(input_ids=ids(batch_size=1, length=4), labels=labels)
        passed = base.calls[-1]["labels"]
        if passed is None:
            raise TypeError("labels must reach the base")
        assert torch.equal(passed, labels)
        assert tuple(passed.shape) == (1, 4)

    def test_a_wider_batch_widens_the_mask(self) -> None:
        """The mask is built per call, so a changing batch size must track it."""
        model, base = wrapped()
        model.forward(input_ids=ids(batch_size=5, length=2), labels=ids(batch_size=5, length=2))
        mask = base.calls[-1]["attention_mask"]
        if mask is None:
            raise TypeError("a prefix run must supply an attention mask")
        assert tuple(mask.shape) == (5, 5)
        assert int(mask.sum().item()) == 25


class TestTheTrainableSurface:
    """What an optimizer built from this model can reach."""

    def test_it_exposes_only_the_slots(self) -> None:
        """Two blocks per layer, and nothing belonging to the base."""
        model, _ = wrapped()
        assert len(model.parameters()) == 4

    def test_constructing_it_freezes_every_base_parameter(self) -> None:
        """A single unfrozen weight would be updated by any optimizer built here."""
        base = FakeCacheCapableModel(num_layers=2, num_kv_heads=4, head_dim=8)
        CartridgeModel(base=base, slots=initialise_slots(_GEOMETRY, seed=1))
        assert all(not parameter.requires_grad for _, parameter in base.named_parameters())

    def test_it_reports_its_geometry(self) -> None:
        """Callers need the shape to write a manifest beside the weights."""
        model, _ = wrapped()
        assert model.geometry == _GEOMETRY


class TestTheProtocolSurface:
    """The rest of what a language model is asked to do."""

    def test_eval_reaches_the_base(self) -> None:
        """Dropout in the frozen base still has to be switched off to evaluate."""
        model, base = wrapped()
        model.eval()
        assert base.name == "cache-capable"

    def test_moving_returns_the_same_model(self) -> None:
        """The protocol's ``to`` chains, so a caller can write ``model.to(d)``."""
        model, _ = wrapped()
        assert model.to("cpu") is model

    def test_moving_keeps_the_slots_trainable(self) -> None:
        """A move must not quietly detach the cartridge from autograd."""
        model, _ = wrapped()
        model.to("cpu")
        assert all(tensor.requires_grad for _, tensor in model.named_parameters())

    def test_the_config_is_the_bases_own(self) -> None:
        """A cartridge changes what a model attends to, not what it is."""
        model, base = wrapped()
        assert model.config is base.config

    def test_gradient_checkpointing_is_refused(self) -> None:
        """A checkpointed model discards the prefix, measured against 4.46.3."""
        model, _ = wrapped()
        with pytest.raises(AppError) as excinfo:
            model.gradient_checkpointing_enable()
        assert (
            excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GRADIENT_CHECKPOINTING_UNSUPPORTED
        )


class TestCheckpointing:
    """A cartridge checkpoint is the cartridge, and not the model under it."""

    def test_the_state_dict_carries_only_the_slots(self) -> None:
        """The base is named by its hub id and reloaded from there."""
        model, _ = wrapped()
        assert len(model.state_dict()) == 4

    def test_loading_installs_the_given_blocks(self) -> None:
        """Resuming a run must restore the prefix it was training."""
        model, _ = wrapped()
        replacement = initialise_slots(_GEOMETRY, seed=99).state_dict()
        name = next(iter(replacement))
        model.load_state_dict(replacement)
        assert torch.equal(model.state_dict()[name].detach(), replacement[name].detach())

    def test_loading_reports_a_total_load(self) -> None:
        """The result type is the protocol's, carrying no partial outcome."""
        model, _ = wrapped()
        result = model.load_state_dict(initialise_slots(_GEOMETRY, seed=2).state_dict())
        assert type(result) is CartridgeLoadResult

    def test_loading_blocks_of_the_wrong_shape_is_refused(self) -> None:
        """A checkpoint from a different model must not be installed."""
        model, _ = wrapped()
        state = model.state_dict()
        state[next(iter(state))] = torch.zeros(1, 4, 99, 8)
        with pytest.raises(AppError) as excinfo:
            model.load_state_dict(state)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH

    def test_saving_writes_a_manifest_and_a_weights_file(self) -> None:
        """Two files: one a person can read, one torch can load."""
        model, _ = wrapped()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            assert sorted(p.name for p in Path(tmpdir).iterdir()) == [
                CARTRIDGE_MANIFEST_NAME,
                CARTRIDGE_WEIGHTS_NAME,
            ]
