"""The parameter blocks a cartridge trains, and the shapes they refuse.

Geometry discovery is driven with a fake that REPORTS a cache shape, because
what is under test here is the reading of that shape. Whether a real model
accepts a prefix built to it is tested against a real transformer in
``test_cartridge_strategy``.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.cartridge import CartridgeGeometry
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    compose,
    discover_geometry,
    initialise_slots,
    require_matching_geometry,
    slots_from_state,
)


def make_geometry(
    *,
    num_layers: int = 2,
    num_kv_heads: int = 4,
    head_dim: int = 8,
    num_slots: int = 3,
) -> CartridgeGeometry:
    """Build a geometry for testing.

    Args:
        num_layers: Layers the prefix spans.
        num_kv_heads: Key-value heads per layer.
        head_dim: Width of one head's vectors.
        num_slots: Prefix positions.

    Returns:
        The geometry.
    """
    return CartridgeGeometry(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_slots=num_slots,
    )


def cache_keys(*, layers: int, kv_heads: int, head_dim: int, dims: int = 4) -> list[torch.Tensor]:
    """Build the per-layer key tensors a probe would return.

    Args:
        layers: How many layers to report.
        kv_heads: Key-value heads per layer.
        head_dim: Width of one head's vectors.
        dims: Dimensionality of each tensor. Four is the real layout.

    Returns:
        One key tensor per layer.
    """
    shape = (1, kv_heads, 1, head_dim)[:dims]
    return [torch.zeros(shape) for _ in range(layers)]


class TestGeometryDiscovery:
    """Reading a cartridge's shape off a model's own cache."""

    def test_it_reads_layers_heads_and_width_from_the_cache(self) -> None:
        """The three fields the MODEL decides, taken from what it reported."""
        geometry = discover_geometry(cache_keys(layers=3, kv_heads=2, head_dim=16), num_slots=7)
        assert geometry == CartridgeGeometry(num_layers=3, num_kv_heads=2, head_dim=16, num_slots=7)

    def test_it_takes_the_kv_head_count_not_the_attention_head_count(self) -> None:
        """The grouped-query case, which is the reason this reads a cache at all.

        A Llama with 8 attention heads and 2 key-value heads caches keys with
        a head dimension of 2. Reading ``num_attention_heads`` from a config
        would give 8 and cut every block four times too wide.
        """
        geometry = discover_geometry(cache_keys(layers=2, kv_heads=2, head_dim=4), num_slots=1)
        assert geometry["num_kv_heads"] == 2

    def test_the_slot_count_is_the_callers_and_passes_through(self) -> None:
        """Three fields come from the model; this one never does."""
        geometry = discover_geometry(cache_keys(layers=1, kv_heads=1, head_dim=1), num_slots=512)
        assert geometry["num_slots"] == 512

    def test_a_model_reporting_no_layers_is_refused(self) -> None:
        """Nothing to prepend to means the strategy cannot run at all."""
        with pytest.raises(AppError) as excinfo:
            discover_geometry([], num_slots=4)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE

    def test_a_cache_without_the_standard_layout_is_refused(self) -> None:
        """A two-dimensional key carries no head count to read."""
        with pytest.raises(AppError) as excinfo:
            discover_geometry(cache_keys(layers=1, kv_heads=2, head_dim=4, dims=2), num_slots=4)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE


class TestInitialisation:
    """A freshly drawn cartridge."""

    def test_every_block_has_the_declared_shape(self) -> None:
        """Batch dimension one, because a cartridge is shared by every sequence."""
        geometry = make_geometry()
        slots = initialise_slots(geometry, seed=1)
        for _, tensor in slots.named_parameters():
            assert tuple(tensor.detach().shape) == (1, 4, 3, 8)

    def test_it_produces_two_blocks_per_layer(self) -> None:
        """Keys and values."""
        slots = initialise_slots(make_geometry(num_layers=5), seed=1)
        assert len(slots.parameters()) == 10

    def test_every_block_is_trainable(self) -> None:
        """A block without gradients would sit in the optimizer doing nothing."""
        slots = initialise_slots(make_geometry(), seed=1)
        assert all(tensor.requires_grad for _, tensor in slots.named_parameters())

    def test_the_same_seed_draws_the_same_cartridge(self) -> None:
        """A run that cannot say what it started from cannot be repeated."""
        first = initialise_slots(make_geometry(), seed=99)
        second = initialise_slots(make_geometry(), seed=99)
        assert all(
            torch.equal(a.detach(), b.detach())
            for (_, a), (_, b) in zip(
                first.named_parameters(), second.named_parameters(), strict=True
            )
        )

    def test_a_different_seed_draws_a_different_cartridge(self) -> None:
        """Otherwise the seed is decorative and reproducibility is an illusion."""
        first = initialise_slots(make_geometry(), seed=1)
        second = initialise_slots(make_geometry(), seed=2)
        assert not torch.equal(
            first.named_parameters()[0][1].detach(),
            second.named_parameters()[0][1].detach(),
        )

    def test_keys_and_values_of_a_layer_differ(self) -> None:
        """Drawn from one generator in sequence, so they must not coincide.

        If the generator were reseeded per block, every key would equal its
        value and the cartridge would carry half the capacity it reports.
        """
        slots = initialise_slots(make_geometry(), seed=5)
        named = slots.named_parameters()
        assert not torch.equal(named[0][1].detach(), named[1][1].detach())

    def test_no_block_is_all_zeros(self) -> None:
        """Zero keys give every slot the same attention logit.

        The slots would stay symmetric under gradient descent and the
        cartridge would collapse to one learned position, which is the reason
        the initialiser draws rather than zeroing.
        """
        slots = initialise_slots(make_geometry(), seed=3)
        assert all(bool(tensor.detach().abs().sum() > 0) for _, tensor in slots.named_parameters())


class TestNaming:
    """Block names are the on-disk contract."""

    def test_names_are_unique_across_every_block(self) -> None:
        """A collision would drop a block on save and mis-load on read."""
        slots = initialise_slots(make_geometry(num_layers=4), seed=1)
        names = [name for name, _ in slots.named_parameters()]
        assert len(set(names)) == len(names)

    def test_the_state_dict_holds_every_block(self) -> None:
        """Two per layer."""
        slots = initialise_slots(make_geometry(num_layers=4), seed=1)
        assert len(slots.state_dict()) == 8

    def test_the_state_dict_names_match_the_parameter_names(self) -> None:
        """They are the same objects under the same names, or a load misplaces one."""
        slots = initialise_slots(make_geometry(), seed=1)
        assert sorted(slots.state_dict()) == sorted(name for name, _ in slots.named_parameters())


class TestBatchExpansion:
    """One cartridge serves every row of a batch."""

    def test_blocks_widen_to_the_batch(self) -> None:
        """Attention needs a block per row; the storage stays one."""
        slots = initialise_slots(make_geometry(), seed=1)
        keys, values = slots.layer_blocks(0, batch_size=5)
        assert tuple(keys.shape) == (5, 4, 3, 8)
        assert tuple(values.shape) == (5, 4, 3, 8)

    def test_expansion_does_not_copy_the_stored_block(self) -> None:
        """A batch of 32 must not allocate 32 cartridges.

        Asserted on storage rather than on a memory figure: an expanded view
        shares the base tensor's storage, a copy does not.
        """
        slots = initialise_slots(make_geometry(), seed=1)
        keys, _ = slots.layer_blocks(0, batch_size=32)
        stored = slots.named_parameters()[0][1]
        assert keys.untyped_storage().data_ptr() == stored.detach().untyped_storage().data_ptr()

    def test_each_layer_returns_its_own_blocks(self) -> None:
        """Returning layer zero for every layer would train one block four times."""
        slots = initialise_slots(make_geometry(num_layers=2), seed=1)
        first, _ = slots.layer_blocks(0, batch_size=1)
        second, _ = slots.layer_blocks(1, batch_size=1)
        assert not torch.equal(first, second)


class TestMovingDevices:
    """A cartridge follows its model."""

    def test_moving_keeps_the_blocks_trainable(self) -> None:
        """``Tensor.to`` returns a non-leaf, which no optimizer would update.

        This is the failure the rebind exists to prevent: without it a run
        that moved its model would silently stop training the cartridge.
        """
        slots = initialise_slots(make_geometry(), seed=1)
        slots.to("cpu")
        assert all(tensor.requires_grad for _, tensor in slots.named_parameters())
        assert all(tensor.detach().is_leaf for _, tensor in slots.named_parameters())

    def test_moving_preserves_the_values(self) -> None:
        """A move must not be a reinitialisation."""
        slots = initialise_slots(make_geometry(), seed=1)
        before = slots.named_parameters()[0][1].detach().clone()
        slots.to("cpu")
        assert torch.equal(before, slots.named_parameters()[0][1].detach())


class TestRebuildingFromState:
    """Reading a cartridge back, and refusing one that does not fit."""

    def test_a_saved_cartridge_round_trips(self) -> None:
        """Same values, same names, still trainable."""
        geometry = make_geometry()
        original = initialise_slots(geometry, seed=11)
        rebuilt = slots_from_state(original.state_dict(), geometry)
        assert all(
            torch.equal(a.detach(), b.detach())
            for (_, a), (_, b) in zip(
                original.named_parameters(), rebuilt.named_parameters(), strict=True
            )
        )
        assert all(tensor.requires_grad for _, tensor in rebuilt.named_parameters())

    def test_a_missing_block_is_refused(self) -> None:
        """The manifest promises blocks the file does not carry."""
        geometry = make_geometry()
        state = initialise_slots(geometry, seed=1).state_dict()
        del state[next(iter(state))]
        with pytest.raises(AppError) as excinfo:
            slots_from_state(state, geometry)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_STATE_INCOMPLETE

    def test_a_block_of_the_wrong_shape_is_refused(self) -> None:
        """Present but wrong is worse than absent: it would attach and mis-attend."""
        geometry = make_geometry()
        state = initialise_slots(geometry, seed=1).state_dict()
        name = next(iter(state))
        state[name] = torch.zeros(1, 4, 99, 8)
        with pytest.raises(AppError) as excinfo:
            slots_from_state(state, geometry)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH

    def test_the_shape_refusal_names_both_shapes(self) -> None:
        """Actionable means saying what was found and what was expected."""
        geometry = make_geometry()
        state = initialise_slots(geometry, seed=1).state_dict()
        name = next(iter(state))
        state[name] = torch.zeros(1, 4, 99, 8)
        with pytest.raises(AppError) as excinfo:
            slots_from_state(state, geometry)
        message = str(excinfo.value)
        assert "(1, 4, 99, 8)" in message
        assert "(1, 4, 3, 8)" in message


class TestGeometryMatching:
    """A cartridge belongs to the model it was cut for."""

    def test_identical_geometry_passes(self) -> None:
        """The ordinary case."""
        require_matching_geometry(make_geometry(), make_geometry())

    def test_a_different_slot_count_is_allowed(self) -> None:
        """The slot count is the CALLER's choice and travels with the cartridge.

        A 512-slot cartridge and a 2048-slot one are both legitimate on the
        same model, so comparing this field would refuse valid pairs.
        """
        require_matching_geometry(make_geometry(num_slots=512), make_geometry(num_slots=2048))

    @pytest.mark.parametrize(
        ("field", "value"),
        [("num_layers", 9), ("num_kv_heads", 9), ("head_dim", 9)],
    )
    def test_a_different_model_shape_is_refused(self, field: str, value: int) -> None:
        """Each of the three model-decided fields, individually."""
        saved = make_geometry()
        other = make_geometry()
        if field == "num_layers":
            other["num_layers"] = value
        elif field == "num_kv_heads":
            other["num_kv_heads"] = value
        else:
            other["head_dim"] = value
        with pytest.raises(AppError) as excinfo:
            require_matching_geometry(saved, other)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH

    def test_the_refusal_names_every_field_that_differs(self) -> None:
        """One message rather than one failure at a time."""
        saved = make_geometry(num_layers=2, num_kv_heads=4, head_dim=8)
        other = make_geometry(num_layers=3, num_kv_heads=5, head_dim=8)
        with pytest.raises(AppError) as excinfo:
            require_matching_geometry(saved, other)
        message = str(excinfo.value)
        assert "num_layers 2 vs 3" in message
        assert "num_kv_heads 4 vs 5" in message
        assert "head_dim" not in message


class TestHoldingPrebuiltBlocks:
    """The constructor takes blocks rather than drawing them."""

    def test_it_reports_the_blocks_it_was_given(self) -> None:
        """Both ways a cartridge comes into existence go through here."""
        geometry = make_geometry(num_layers=1)
        keys = [torch.ones(1, 4, 3, 8)]
        values = [torch.zeros(1, 4, 3, 8)]
        slots = CartridgeSlots(geometry=geometry, keys=keys, values=values)
        named = slots.named_parameters()
        assert torch.equal(named[0][1].detach(), torch.ones(1, 4, 3, 8))
        assert torch.equal(named[1][1].detach(), torch.zeros(1, 4, 3, 8))


class TestComposition:
    """Joining two cartridges into one prefix.

    The mechanical half. Whether a composed cartridge still WORKS is measured
    against a real model in ``test_cartridge_composition``.
    """

    def test_the_slot_counts_add(self) -> None:
        """Two contexts laid end to end occupy both their positions."""
        joined = compose(
            initialise_slots(make_geometry(num_slots=3), seed=1),
            initialise_slots(make_geometry(num_slots=5), seed=2),
        )
        assert joined.geometry["num_slots"] == 8

    def test_the_model_dimensions_are_unchanged(self) -> None:
        """Composition lengthens the prefix; it does not reshape the model."""
        joined = compose(
            initialise_slots(make_geometry(num_slots=3), seed=1),
            initialise_slots(make_geometry(num_slots=5), seed=2),
        )
        assert (
            joined.geometry["num_layers"],
            joined.geometry["num_kv_heads"],
            joined.geometry["head_dim"],
        ) == (2, 4, 8)

    def test_every_block_has_the_joined_shape(self) -> None:
        """The concatenation is on the SLOT axis, which is the one that grows.

        Joining on the head axis would produce blocks of the right total size
        describing a model with twice the heads, and attention would read them
        without complaint, so the axis is asserted through the shape.
        """
        joined = compose(
            initialise_slots(make_geometry(num_slots=3), seed=1),
            initialise_slots(make_geometry(num_slots=5), seed=2),
        )
        assert all(
            tuple(tensor.detach().shape) == (1, 4, 8, 8) for _, tensor in joined.named_parameters()
        )

    def test_the_first_cartridge_occupies_the_leading_slots(self) -> None:
        """Order is the caller's and is preserved, block by block."""
        first = initialise_slots(make_geometry(num_slots=3), seed=1)
        second = initialise_slots(make_geometry(num_slots=5), seed=2)
        joined = compose(first, second)
        for index, (_, tensor) in enumerate(joined.named_parameters()):
            original = first.named_parameters()[index][1].detach()
            assert torch.equal(tensor.detach()[:, :, :3, :], original)

    def test_the_second_cartridge_occupies_the_trailing_slots(self) -> None:
        """The other half of the same claim."""
        first = initialise_slots(make_geometry(num_slots=3), seed=1)
        second = initialise_slots(make_geometry(num_slots=5), seed=2)
        joined = compose(first, second)
        for index, (_, tensor) in enumerate(joined.named_parameters()):
            original = second.named_parameters()[index][1].detach()
            assert torch.equal(tensor.detach()[:, :, 3:, :], original)

    def test_the_inputs_are_left_usable(self) -> None:
        """Composing must not consume its operands.

        A caller composing A with B still holds A, and may compose it with C
        next. Sharing storage would make the second composition see whatever
        the first did.
        """
        first = initialise_slots(make_geometry(num_slots=3), seed=1)
        before = first.named_parameters()[0][1].detach().clone()
        joined = compose(first, initialise_slots(make_geometry(num_slots=5), seed=2))
        joined.named_parameters()[0][1].detach().fill_(7.0)
        assert torch.equal(first.named_parameters()[0][1].detach(), before)

    def test_the_result_is_trainable(self) -> None:
        """A composed cartridge can be trained further, so its blocks are leaves."""
        joined = compose(
            initialise_slots(make_geometry(num_slots=3), seed=1),
            initialise_slots(make_geometry(num_slots=5), seed=2),
        )
        assert all(tensor.requires_grad for _, tensor in joined.named_parameters())

    def test_composing_is_not_commutative(self) -> None:
        """Slots sit at positions, so order changes the object.

        Asserted rather than left ambiguous: nothing here sorts the operands,
        and a caller who assumed otherwise would get a different prefix than
        they expected without any error.
        """
        first = initialise_slots(make_geometry(num_slots=3), seed=1)
        second = initialise_slots(make_geometry(num_slots=5), seed=2)
        forward = compose(first, second).named_parameters()[0][1].detach()
        backward = compose(second, first).named_parameters()[0][1].detach()
        assert not torch.equal(forward, backward)

    def test_cartridges_for_differently_shaped_models_cannot_be_joined(self) -> None:
        """Two prefixes for different models cannot occupy one cache."""
        with pytest.raises(AppError) as excinfo:
            compose(
                initialise_slots(make_geometry(num_layers=2), seed=1),
                initialise_slots(make_geometry(num_layers=3), seed=2),
            )
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH
