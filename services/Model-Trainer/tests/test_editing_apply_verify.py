"""Editing a real GPT-2, and proving what the edit did and did not do.

Nothing is faked. The `tiny` rung is built by the production builder, the real
``Conv1D`` weight is edited, the real forward pass runs, and the activations
are read by the production capture. The one hook that is swapped is swapped
for a reason stated at its test: the arm reporting a capture that did not
happen cannot be reached while a real module is running.

WHAT THIS SUITE CAUGHT, and it is the reason to read it before trusting an
injection result. An edit can be exactly right at the module and change
nothing the model outputs. Solving for a target that adds the SAME constant to
every feature moves the module's output by construction and is then removed
downstream by layer normalisation, which subtracts the mean. Measured here:
that edit leaves the forward loss identical while an edit of the same size in
a direction moves it by four hundredths. A harness that verified only the
module output would report a perfect injection either way.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.knowledge_edit import EditSite, resolve_edit_module
from model_trainer.core.services.model.editing import _test_hooks
from model_trainer.core.services.model.editing.activations import capture_module_io
from model_trainer.core.services.model.editing.apply import (
    WeightSnapshot,
    apply_rank_one_edit,
    restore_weight,
    snapshot_weight,
)
from model_trainer.core.services.model.editing.rank_one import solve_right_vector
from model_trainer.core.services.model.editing.sites import (
    require_edit_module,
    weight_parameter_name,
)
from model_trainer.core.services.model.editing.verify import (
    as_input_output_weight,
    changed_parameters,
    parameter_digests,
    verify_rank_one_edit,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import (
    HookValue,
    LMModelProto,
    TracedLMModelProto,
    TracedModuleProto,
)

#: The site every test here edits: the first block's MLP down-projection,
#: which is where the reference implementation writes on GPT-2.
_SITE: EditSite = {
    "layer": 0,
    "module_template": "transformer.h.{}.mlp.c_proj",
    "fact_token": "prompt_last",
}

#: Parameters the tiny rung actually has. Pinned because the count is what
#: makes "nothing else changed" a statement about the whole model.
_TINY_PARAMETER_COUNT = 28

#: Residual left by float32 on this arithmetic. Measured at 9.5e-07 for both
#: the prediction and the key residuals; the bound is an order above that so
#: it fails on a wrong edit rather than on a rounding difference.
_FLOAT32_RESIDUAL = 1e-5


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Put the capture hook back after any test that swapped it."""
    yield
    _test_hooks.reset_hooks()


def _tiny() -> tuple[TracedLMModelProto, torch.Tensor]:
    """Build the tiny rung and its input.

    Returns:
        The model in eval mode on the CPU, and one sequence of token ids.
    """
    return probe_model_and_input("cpu", PROBE_SHAPES["tiny"])


def _solve_edit_for(
    model: TracedLMModelProto,
    input_ids: torch.Tensor,
    target_delta: torch.Tensor,
    position: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Capture at the site and solve for the update that moves the output.

    Args:
        model: The model to read.
        input_ids: One sequence.
        target_delta: How far to move the module's output.
        position: Token to key the edit on.

    Returns:
        ``(left, right, target_output, denominator)``.
    """
    captured = capture_module_io(
        model=model,
        module_name=resolve_edit_module(_SITE),
        input_ids=input_ids,
        position=position,
    )
    target_output = captured["module_output"] + target_delta
    solve = solve_right_vector(
        target_output=target_output,
        current_output=captured["module_output"],
        current_input=captured["module_input"],
        left=captured["module_input"],
    )
    return captured["module_input"], solve["vector"], target_output, solve["denominator"]


def _forward_loss(model: LMModelProto, input_ids: torch.Tensor) -> float:
    """Run the model and read its loss.

    Args:
        model: The model to run.
        input_ids: One sequence, used as its own labels.

    Returns:
        The loss.
    """
    with torch.no_grad():
        return float(model.forward(input_ids=input_ids, labels=input_ids).loss.item())


def test_an_edit_hits_its_target_and_moves_no_other_parameter() -> None:
    model, input_ids = _tiny()
    parameter_name = weight_parameter_name(_SITE)
    left, right, target_output, denominator = _solve_edit_for(
        model, input_ids, torch.full((128,), 1.0)
    )

    before = snapshot_weight(model, parameter_name)
    digests_before = parameter_digests(model)
    record = apply_rank_one_edit(
        model=model,
        site=_SITE,
        item_id="edit-0001",
        left=left,
        right=right,
        denominator=denominator,
    )
    after = snapshot_weight(model, parameter_name)
    digests_after = parameter_digests(model)

    torch.manual_seed(20260902)
    probes = [torch.randn(left.shape[0]) for _ in range(4)]
    verification = verify_rank_one_edit(
        module=parameter_name,
        before=before["values"],
        after=after["values"],
        transposed=record["transposed"],
        left=left,
        right=right,
        probes=probes,
        key_input=left,
        target_output=target_output,
        digests_before=digests_before,
        digests_after=digests_after,
    )

    assert verification["other_parameters_changed"] == ()
    assert verification["max_prediction_error"] < _FLOAT32_RESIDUAL
    assert verification["key_output_error"] < _FLOAT32_RESIDUAL
    assert record["module"] == "transformer.h.0.mlp.c_proj.weight"
    assert record["item_id"] == "edit-0001"


def test_the_record_carries_the_orientation_gpt2_actually_stores() -> None:
    """``Conv1D`` is (input, output), so the composed update needs no transpose.

    Pinned as a fact about this architecture rather than as a preference: a
    model family storing (output, input) flips this flag, and an experiment
    comparing the two needs the difference on the record.
    """
    model, input_ids = _tiny()
    left, right, _, denominator = _solve_edit_for(model, input_ids, torch.full((128,), 1.0))
    record = apply_rank_one_edit(
        model=model,
        site=_SITE,
        item_id="edit-0002",
        left=left,
        right=right,
        denominator=denominator,
    )
    assert record["transposed"] is False
    assert (record["weight_rows"], record["weight_cols"]) == (512, 128)


def test_the_edited_module_emits_the_target_on_a_later_forward_pass() -> None:
    """The before-and-after the whole exercise rests on, at the module."""
    model, input_ids = _tiny()
    left, right, target_output, denominator = _solve_edit_for(
        model, input_ids, torch.full((128,), 1.0)
    )
    apply_rank_one_edit(
        model=model,
        site=_SITE,
        item_id="edit-0003",
        left=left,
        right=right,
        denominator=denominator,
    )
    recaptured = capture_module_io(
        model=model,
        module_name=resolve_edit_module(_SITE),
        input_ids=input_ids,
        position=-1,
    )
    assert torch.allclose(
        recaptured["module_output"], target_output, atol=_FLOAT32_RESIDUAL, rtol=0.0
    )


def test_a_mean_shift_target_is_invisible_downstream_and_a_direction_is_not() -> None:
    """Layer normalisation removes a uniform shift, so the edit does nothing.

    Both edits below are applied identically and both hit their target at the
    module. Only one changes what the model computes, because the residual
    stream passes through layer normalisation, which subtracts the mean of the
    features. A target that adds the same constant to all of them is exactly
    what that operation discards.

    The bounds are asymmetric on purpose. The mean-shift arm measured a diff
    of exactly 0.0 here, and is asserted as indistinguishable rather than
    bit-identical so that a card with a different last bit fails nothing. The
    direction arm measured -3.9e-02, four orders above that.
    """
    uniform_model, input_ids = _tiny()
    before_uniform = _forward_loss(uniform_model, input_ids)
    left, right, _, denominator = _solve_edit_for(uniform_model, input_ids, torch.full((128,), 5.0))
    apply_rank_one_edit(
        model=uniform_model,
        site=_SITE,
        item_id="edit-uniform",
        left=left,
        right=right,
        denominator=denominator,
    )
    uniform_change = abs(_forward_loss(uniform_model, input_ids) - before_uniform)

    direction_model, direction_ids = _tiny()
    before_direction = _forward_loss(direction_model, direction_ids)
    torch.manual_seed(7)
    left, right, _, denominator = _solve_edit_for(direction_model, direction_ids, torch.randn(128))
    apply_rank_one_edit(
        model=direction_model,
        site=_SITE,
        item_id="edit-direction",
        left=left,
        right=right,
        denominator=denominator,
    )
    direction_change = abs(_forward_loss(direction_model, direction_ids) - before_direction)

    assert uniform_change < 1e-6
    assert direction_change > 1e-3


def test_restoring_returns_every_parameter_to_its_exact_bytes() -> None:
    model, input_ids = _tiny()
    parameter_name = weight_parameter_name(_SITE)
    digests_before = parameter_digests(model)
    snapshot = snapshot_weight(model, parameter_name)
    left, right, _, denominator = _solve_edit_for(model, input_ids, torch.full((128,), 1.0))
    apply_rank_one_edit(
        model=model,
        site=_SITE,
        item_id="edit-0004",
        left=left,
        right=right,
        denominator=denominator,
    )
    assert parameter_digests(model) != digests_before

    restore_weight(model, snapshot)
    assert parameter_digests(model) == digests_before


def test_editing_a_parameter_the_model_does_not_expose_names_the_site() -> None:
    """``lm_head.weight`` is tied to the embedding, so it is not a parameter.

    A real trap rather than an invented one: the name exists as a module
    attribute and the edit would look reasonable, but ``named_parameters``
    deduplicates tied weights, so writing "lm_head" is writing nowhere.
    """
    model, _ = _tiny()
    site: EditSite = {
        "layer": 0,
        "module_template": "lm_head{}",
        "fact_token": "prompt_last",
    }
    with pytest.raises(AppError) as caught:
        snapshot_weight(model, weight_parameter_name(site))
    assert caught.value.code is ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND


def test_editing_a_vector_parameter_is_refused() -> None:
    """A layer norm's gain is a real parameter and not a matrix."""
    model, _ = _tiny()
    with pytest.raises(AppError) as caught:
        snapshot_weight(model, "transformer.ln_f.weight")
    assert caught.value.code is ModelTrainerErrorCode.EDIT_WEIGHT_NOT_MATRIX


def test_restoring_a_snapshot_of_the_wrong_shape_is_refused() -> None:
    model, _ = _tiny()
    snapshot: WeightSnapshot = {
        "parameter_name": weight_parameter_name(_SITE),
        "values": torch.zeros(3, 4),
    }
    with pytest.raises(AppError) as caught:
        restore_weight(model, snapshot)
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH


def test_restoring_values_that_cannot_be_confirmed_is_refused() -> None:
    """A NaN-bearing snapshot cannot be verified restored, so it is not accepted.

    ``torch.equal`` is false for NaN against itself, which is precisely the
    case this arm exists for: the write may have succeeded and the code cannot
    say so. Reached with real tensors rather than by intercepting the write.
    """
    model, _ = _tiny()
    parameter_name = weight_parameter_name(_SITE)
    poisoned = snapshot_weight(model, parameter_name)
    poisoned["values"][0, 0] = float("nan")
    with pytest.raises(AppError) as caught:
        restore_weight(model, poisoned)
    assert caught.value.code is ModelTrainerErrorCode.EDIT_RESTORE_MISMATCH


def test_capture_refuses_more_than_one_sequence() -> None:
    model, input_ids = _tiny()
    with pytest.raises(AppError) as caught:
        capture_module_io(
            model=model,
            module_name=resolve_edit_module(_SITE),
            input_ids=torch.cat([input_ids, input_ids], dim=0),
            position=0,
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH


@pytest.mark.parametrize("position", [64, -65])
def test_capture_refuses_a_position_outside_the_sequence(position: int) -> None:
    model, input_ids = _tiny()
    with pytest.raises(AppError) as caught:
        capture_module_io(
            model=model,
            module_name=resolve_edit_module(_SITE),
            input_ids=input_ids,
            position=position,
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH


def test_capture_refuses_a_module_whose_output_is_not_a_tensor() -> None:
    """A transformer block returns a tuple, and there is no activation in it.

    Real module, real forward pass. The guard exists because a caller who
    points the site at a block rather than at a projection gets a plausible
    name and no vector.
    """
    model, input_ids = _tiny()
    with pytest.raises(AppError) as caught:
        capture_module_io(
            model=model, module_name="transformer.h.0", input_ids=input_ids, position=0
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED


def test_capture_reports_a_forward_pass_that_never_ran_the_module() -> None:
    """The one arm a real module cannot reach, reached through the seam.

    A hook attached to a live module always fires, so the only way to exercise
    the report is a forward that runs nothing. Production binds this hook to
    the real forward; nothing else in the suite swaps it.
    """
    model, input_ids = _tiny()

    def _forward_that_runs_nothing(model: LMModelProto, input_ids: torch.Tensor) -> None:
        return None

    _test_hooks.run_capture_forward = _forward_that_runs_nothing
    with pytest.raises(AppError) as caught:
        capture_module_io(
            model=model,
            module_name=resolve_edit_module(_SITE),
            input_ids=input_ids,
            position=0,
        )
    assert caught.value.code is ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED
    assert "did not run" in caught.value.message


def test_requiring_a_module_that_does_not_exist_names_the_site() -> None:
    model, _ = _tiny()
    with pytest.raises(AppError) as caught:
        require_edit_module(model, "transformer.h.99.mlp.c_proj")
    assert caught.value.code is ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND


def test_the_production_capture_returns_what_a_hook_on_that_module_sees() -> None:
    """Cross-check the capture against the module's own output, value by value.

    The module resolved by :func:`require_edit_module` is hooked directly here
    and the numbers it emits are compared with what
    :func:`capture_module_io` reports for the same position. Two paths to one
    tensor: if the capture ever indexed the wrong position or the wrong batch,
    this is what says so.
    """
    model, input_ids = _tiny()
    module = require_edit_module(model, resolve_edit_module(_SITE))
    seen: list[torch.Tensor] = []

    def _record(
        hooked: TracedModuleProto, args: tuple[HookValue, ...], output: HookValue, /
    ) -> None:
        """Keep the module's output tensor.

        Args:
            hooked: The module that ran, unread.
            args: Its positional arguments, unread.
            output: What it returned.

        Raises:
            TypeError: If the projection returned something other than a
                tensor, which would mean this test is hooked to a module it
                did not mean to hook.
        """
        if not torch.is_tensor(output):
            raise TypeError(f"expected a tensor, got {type(output).__name__}")
        seen.append(output.detach().clone())

    handle = module.register_forward_hook(_record)
    with torch.no_grad():
        model.forward(input_ids=input_ids, labels=input_ids)
    handle.remove()

    captured = capture_module_io(
        model=model,
        module_name=resolve_edit_module(_SITE),
        input_ids=input_ids,
        position=-1,
    )
    assert len(seen) == 1
    assert tuple(seen[0].shape) == (1, 64, 128)
    assert torch.equal(seen[0][0, -1], captured["module_output"])


def test_parameter_digests_covers_every_parameter_the_model_has() -> None:
    model, _ = _tiny()
    digests = parameter_digests(model)
    assert len(digests) == _TINY_PARAMETER_COUNT
    assert "transformer.h.0.mlp.c_proj.weight" in digests


def test_changed_parameters_reports_additions_removals_and_moves() -> None:
    before = {"a": 1.0, "b": 2.0, "edited": 3.0, "gone": 4.0}
    after = {"a": 1.0, "b": 9.0, "edited": 99.0, "added": 5.0}
    assert changed_parameters(before, after, "edited") == ("added", "b", "gone")


def test_input_output_orientation_transposes_only_when_told_to() -> None:
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert torch.equal(as_input_output_weight(weight, False), weight)
    assert torch.equal(as_input_output_weight(weight, True), weight.T)
