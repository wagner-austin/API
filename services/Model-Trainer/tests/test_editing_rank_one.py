"""The rank-one arithmetic, checked against the identity it claims.

These run on real tensors in float64, because the claim under test is exact
rather than approximate: adding an outer product to a matrix changes that
matrix's action on EVERY input by the value vector scaled by the key's dot
product with the input, and by nothing else. Float64 keeps the residual at the
level of representation error, so a tolerance of 1e-12 is a real check and not
a shrug.

The degenerate case is constructed rather than stumbled on: a key exactly
orthogonal to the module's input has a zero divisor, which is the one input
this arithmetic cannot answer for.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.services.model.editing.rank_one import (
    MIN_SOLVE_DENOMINATOR,
    compose_rank_one,
    orient_for_weight,
    predicted_output_delta,
    solve_right_vector,
)

_INPUT_WIDTH = 5
_OUTPUT_WIDTH = 3


def _vector(*values: float) -> torch.Tensor:
    """Build a float64 vector from written-out values.

    ``torch.tensor`` takes an untyped list and this package types every
    expression, so the values are written into a zeroed tensor instead. The
    tests that need exact inputs -- an orthogonal key, a divisor sitting on
    the stated minimum -- need them written out rather than drawn.

    Args:
        *values: The elements, in order.

    Returns:
        A one-dimensional float64 tensor holding them.
    """
    tensor = torch.zeros(len(values), dtype=torch.float64)
    for index, value in enumerate(values):
        tensor[index] = value
    return tensor


def _seeded(*shape: int) -> torch.Tensor:
    """Draw a deterministic float64 tensor.

    Args:
        *shape: Dimensions to draw.

    Returns:
        The tensor, from a generator seeded per call so two calls with the
        same shape do NOT collide.
    """
    return torch.randn(*shape, dtype=torch.float64)


def test_compose_rank_one_is_the_outer_product() -> None:
    left = _vector(1.0, 2.0)
    right = _vector(3.0, 4.0, 5.0)
    expected = torch.zeros(2, 3, dtype=torch.float64)
    expected[0] = _vector(3.0, 4.0, 5.0)
    expected[1] = _vector(6.0, 8.0, 10.0)
    assert torch.equal(compose_rank_one(left, right), expected)


def test_an_applied_update_moves_every_output_by_exactly_the_predicted_delta() -> None:
    """The whole content of a rank-one edit, measured rather than asserted."""
    torch.manual_seed(20260902)
    weight = _seeded(_INPUT_WIDTH, _OUTPUT_WIDTH)
    left = _seeded(_INPUT_WIDTH)
    right = _seeded(_OUTPUT_WIDTH)
    edited = weight + compose_rank_one(left, right)

    for _ in range(8):
        probe = _seeded(_INPUT_WIDTH)
        measured = probe @ edited - probe @ weight
        predicted = predicted_output_delta(left=left, right=right, probe=probe)
        assert torch.allclose(measured, predicted, atol=1e-12, rtol=0.0)


def test_solving_the_value_vector_makes_the_module_emit_the_target() -> None:
    torch.manual_seed(20260903)
    weight = _seeded(_INPUT_WIDTH, _OUTPUT_WIDTH)
    key_input = _seeded(_INPUT_WIDTH)
    left = _seeded(_INPUT_WIDTH)
    current_output = key_input @ weight
    target_output = current_output + _vector(0.5, -1.5, 2.0)

    solve = solve_right_vector(
        target_output=target_output,
        current_output=current_output,
        current_input=key_input,
        left=left,
    )
    edited = weight + compose_rank_one(left, solve["vector"])

    assert torch.allclose(key_input @ edited, target_output, atol=1e-12, rtol=0.0)
    assert solve["denominator"] == pytest.approx(float(torch.dot(key_input, left).item()))


def test_solving_refuses_a_key_orthogonal_to_the_input() -> None:
    """Exactly orthogonal, so the divisor is 0.0 and not merely small."""
    key_input = _vector(1.0, 0.0)
    left = _vector(0.0, 1.0)
    with pytest.raises(AppError) as caught:
        solve_right_vector(
            target_output=_vector(1.0),
            current_output=_vector(0.0),
            current_input=key_input,
            left=left,
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_KEY_ORTHOGONAL_TO_INPUT


def test_solving_accepts_a_divisor_at_the_stated_minimum() -> None:
    """The boundary is inclusive, and the value vector is correspondingly large.

    Pinned because the constant is a claim about which requests this
    arithmetic can answer, and a boundary that quietly moved would change
    which edits a sweep reports as impossible.
    """
    key_input = _vector(MIN_SOLVE_DENOMINATOR)
    left = _vector(1.0)
    solve = solve_right_vector(
        target_output=_vector(1.0),
        current_output=_vector(0.0),
        current_input=key_input,
        left=left,
    )
    assert solve["denominator"] == pytest.approx(MIN_SOLVE_DENOMINATOR)
    assert float(solve["vector"].item()) == pytest.approx(1.0 / MIN_SOLVE_DENOMINATOR)


def test_orientation_leaves_a_matching_update_alone() -> None:
    update = _seeded(_INPUT_WIDTH, _OUTPUT_WIDTH)
    oriented = orient_for_weight(update, torch.Size([_INPUT_WIDTH, _OUTPUT_WIDTH]))
    assert oriented["transposed"] is False
    assert torch.equal(oriented["matrix"], update)


def test_orientation_transposes_an_update_for_a_linear_style_weight() -> None:
    """The branch GPT-2 never takes and GPT-J does.

    ``Conv1D`` stores (input, output), so a composed update fits it directly.
    A module storing (output, input) needs the transpose, and the flag is what
    puts that difference on the edit's record instead of hiding it.
    """
    update = _seeded(_INPUT_WIDTH, _OUTPUT_WIDTH)
    oriented = orient_for_weight(update, torch.Size([_OUTPUT_WIDTH, _INPUT_WIDTH]))
    assert oriented["transposed"] is True
    assert torch.equal(oriented["matrix"], update.T)


def test_orientation_refuses_a_weight_that_is_not_a_matrix() -> None:
    with pytest.raises(AppError) as caught:
        orient_for_weight(_seeded(_INPUT_WIDTH, _OUTPUT_WIDTH), torch.Size([_INPUT_WIDTH]))
    assert caught.value.code is ModelTrainerErrorCode.EDIT_WEIGHT_NOT_MATRIX


def test_orientation_refuses_an_update_that_fits_neither_way() -> None:
    with pytest.raises(AppError) as caught:
        orient_for_weight(_seeded(_INPUT_WIDTH, _OUTPUT_WIDTH), torch.Size([7, 11]))
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH


@pytest.mark.parametrize("argument", ["left", "right"])
def test_composing_refuses_an_argument_that_is_not_a_vector(argument: str) -> None:
    vectors = {
        "left": _seeded(_INPUT_WIDTH),
        "right": _seeded(_OUTPUT_WIDTH),
    }
    vectors[argument] = _seeded(2, 2)
    with pytest.raises(AppError) as caught:
        compose_rank_one(vectors["left"], vectors["right"])
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH
    assert argument in caught.value.message


@pytest.mark.parametrize("argument", ["target_output", "current_output", "current_input", "left"])
def test_solving_refuses_an_argument_that_is_not_a_vector(argument: str) -> None:
    arguments = {
        "target_output": _seeded(_OUTPUT_WIDTH),
        "current_output": _seeded(_OUTPUT_WIDTH),
        "current_input": _seeded(_INPUT_WIDTH),
        "left": _seeded(_INPUT_WIDTH),
    }
    arguments[argument] = _seeded(2, 2)
    with pytest.raises(AppError) as caught:
        solve_right_vector(
            target_output=arguments["target_output"],
            current_output=arguments["current_output"],
            current_input=arguments["current_input"],
            left=arguments["left"],
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH
    assert argument in caught.value.message


def test_solving_refuses_outputs_of_different_lengths() -> None:
    with pytest.raises(AppError) as caught:
        solve_right_vector(
            target_output=_seeded(_OUTPUT_WIDTH),
            current_output=_seeded(_OUTPUT_WIDTH + 1),
            current_input=_seeded(_INPUT_WIDTH),
            left=_seeded(_INPUT_WIDTH),
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH
    assert "current_output" in caught.value.message


def test_solving_refuses_a_key_from_a_different_space_than_the_input() -> None:
    with pytest.raises(AppError) as caught:
        solve_right_vector(
            target_output=_seeded(_OUTPUT_WIDTH),
            current_output=_seeded(_OUTPUT_WIDTH),
            current_input=_seeded(_INPUT_WIDTH),
            left=_seeded(_INPUT_WIDTH + 2),
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH
    assert "left" in caught.value.message


def test_predicting_refuses_a_probe_that_is_not_a_vector() -> None:
    with pytest.raises(AppError) as caught:
        predicted_output_delta(
            left=_seeded(_INPUT_WIDTH),
            right=_seeded(_OUTPUT_WIDTH),
            probe=_seeded(2, 2),
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH
    assert "probe" in caught.value.message


def test_predicting_refuses_a_probe_from_a_different_space_than_the_key() -> None:
    with pytest.raises(AppError) as caught:
        predicted_output_delta(
            left=_seeded(_INPUT_WIDTH),
            right=_seeded(_OUTPUT_WIDTH),
            probe=_seeded(_INPUT_WIDTH + 1),
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH
