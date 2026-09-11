"""The published way to put a disposition into a model, for comparison.

WHY THIS ARM EXISTS AT ALL. Without it the trait sweep reports a number about
cartridges; with it the run is a comparison against the only measured account
of dispositional composition there is. The published result is that
composition is expensive -- two steering vectors already cost a large fraction
of trait expression, and every common composition scheme degrades as vectors
are added -- and this arc has a lever that literature does not: the
compartments are trained, not extracted. A comparison at MATCHED trait counts
is the finding whichever way it lands, and a cartridge number with nothing
beside it is not.

WHAT A CONTRASTIVE ACTIVATION VECTOR IS. Run the model over text that
exhibits a trait and over matched text that does not, read one module's output
at the last token of each, and take the mean difference. That direction is
what the model's own activations differ by when the trait is present. Adding a
multiple of it back at the same module is the intervention.

WHY THE MODULE IS DECLARED AND NOT DERIVED. The depth at which a contrastive
direction is readable is a property of the model, and this programme has not
measured it. A derived layer -- two thirds of the depth, say -- would put a
number nobody chose into every record, and the record would not say it was
unchosen. :class:`~model_trainer.core.contracts.trait_plan.TraitPlan` declares
it, so a run states the site it measured at.

WHY THE SITE MUST BE A TENSOR-VALUED MODULE. Extraction reuses
:func:`~model_trainer.core.services.model.editing.activations.capture_module_io`,
which refuses a module that runs with a non-tensor output -- a whole
transformer block returns a tuple -- and that refusal is the honest one to
inherit rather than to work around. The projection that writes a block's
result back into the residual stream is tensor-valued and is where the edit
literature already keys its own interventions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)

from model_trainer.core.services.model.cartridge_scoring import (
    TraitPair,
    TraitPairLosses,
    base_loss,
)
from model_trainer.core.services.model.editing.activations import capture_module_io
from model_trainer.core.services.model.editing.sites import require_edit_module
from model_trainer.core.types import (
    HookHandleProto,
    HookValue,
    LMModelProto,
    SteerableLMProto,
    TracedModuleProto,
)


def require_steerable(model: LMModelProto) -> SteerableLMProto:
    """Narrow a model to one that can be both scored and perturbed.

    The single place the widening happens, so one message explains it.
    ``isinstance`` against a runtime-checkable protocol rather than a cast,
    for the reason its two siblings give: a cast would ASSERT the capability
    and this establishes it.

    Args:
        model: The model to narrow.

    Returns:
        The same model, typed as steerable.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the model presents no
            module graph or cannot be called for a loss. Refused rather than
            skipped: an arm that silently declined to steer would report the
            plain base's numbers under the steering arm's name, which is the
            one failure this comparison cannot survive.
    """
    if isinstance(model, SteerableLMProto):
        return model
    raise AppError(
        ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND,
        (
            "this model cannot be both walked and scored, so a steering arm has no site "
            "to read a direction at or no loss to report; the arm needs a transformer "
            "whose submodules can be reached and which returns a loss when given labels"
        ),
        model_trainer_status_for(ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND),
    )


def extract_steering_vector(
    model: SteerableLMProto, pairs: Sequence[TraitPair], *, module_name: str
) -> torch.Tensor:
    """Read the direction a module's output moves in when a trait is present.

    THE LAST TOKEN, on both members. The trait has to have been expressed by
    then -- it is what the whole continuation carries -- and reading a fixed
    interior position would read a different point of two continuations that
    are not the same length. Position -1 is the one index that means the same
    thing in both.

    Args:
        model: The plain base, traceable. Never a cartridge-wrapped model: a
            steering vector is an alternative to a prefix, not an addition to
            one, and extracting through a prefix would measure the pair of
            them.
        pairs: The TRAINING pairs of one trait. Training rather than held-out,
            for the same reason the cartridge trains on the training half: a
            direction read off the pairs it is scored on measures
            memorisation, and the arms would not be comparable.
        module_name: Dotted path of the module to read.

    Returns:
        The mean difference, one dimensional, on the model's device.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if no pairs are supplied --
            a mean over nothing has no direction, and a zero vector would
            steer nothing while reporting as a steering arm. With
            ``EDIT_MODULE_NOT_FOUND`` or ``EDIT_ACTIVATION_NOT_CAPTURED``
            propagated from the capture.
    """
    if not pairs:
        raise AppError(
            ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE,
            (
                "a steering vector cannot be extracted from zero pairs; the mean "
                "difference over an empty set has no direction, and a zero vector would "
                "steer nothing at all while appearing in the record as a steering arm"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE),
        )
    differences = [
        capture_module_io(
            model=model, module_name=module_name, input_ids=pair["expressing"], position=-1
        )["module_output"]
        - capture_module_io(
            model=model, module_name=module_name, input_ids=pair["neutral"], position=-1
        )["module_output"]
        for pair in pairs
    ]
    return torch.stack(differences).mean(dim=0)


def unit_direction(vector: torch.Tensor) -> torch.Tensor:
    """Scale a steering vector to unit length.

    WHY THE ARMS ARE COMPARED AT UNIT LENGTH. The raw mean difference's
    magnitude is a property of the model's activation scale at the chosen
    module, not of the trait, so two traits' raw vectors are not comparable
    and neither are two modules'. Normalising makes the plan's declared
    strength the single knob, and a record that carries one number a reader
    can reproduce rather than two that interact.

    Args:
        vector: The mean difference.

    Returns:
        The same direction at unit norm.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the vector has no length.
            A zero mean difference means the module's output did not move
            between the two members at all, so there is no direction to
            normalise -- and scaling it would divide by zero and report NaNs
            as an arm.
    """
    norm = math.sqrt(float((vector * vector).sum().item()))
    if norm == 0.0:
        raise AppError(
            ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE,
            (
                "the mean activation difference is exactly zero, so this module's output "
                "does not move between the two members of any pair; there is no direction "
                "to steer along and the site cannot carry this trait"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE),
        )
    return vector / norm


def compose_directions(directions: Sequence[torch.Tensor]) -> torch.Tensor:
    """Combine several trait directions the way the literature composes them.

    SUMMED, AND THE SUM IS THE POINT. Adding vectors is the scheme the
    published composition measurements use, and its cost as vectors are added
    is exactly what this arm exists to compare the cartridge grid against.
    Something cleverer here -- a shared low-dimensional basis, an orthogonal
    projection -- would be a different intervention, and comparing a cartridge
    grid against it would answer a question nobody has published a baseline
    for.

    The sum is re-normalised so a composed arm is applied at the same strength
    as a solo one. Without that, composing at n4 would steer at up to four
    times the magnitude and the arm would measure the strength, not the
    composition.

    Args:
        directions: Unit directions, one per trait, in roster order.

    Returns:
        The composed direction, at unit norm.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if no directions are given,
            or if they sum to zero -- which happens when they cancel, and is
            a real property of the traits rather than an error to smooth over,
            so it is named rather than replaced with a small epsilon.
    """
    if not directions:
        raise AppError(
            ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE,
            (
                "no directions to compose; a composed steering arm over zero traits is "
                "the plain base under another name and must not enter a record as an arm"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE),
        )
    return unit_direction(torch.stack(list(directions)).sum(dim=0))


def _steered(output: HookValue, shift: torch.Tensor, module_name: str) -> torch.Tensor:
    """Add the steering shift to a tensor-valued module's output.

    ONE SHAPE, AND A REFUSAL FOR EVERYTHING ELSE. A transformer block returns
    a tuple whose first element is the hidden state and whose remainder is
    cache, and it would be easy to reach in and perturb element zero. That is
    exactly what this refuses: the site is already required to be
    tensor-valued, because
    :func:`~model_trainer.core.services.model.editing.activations.capture_module_io`
    refuses to READ a direction anywhere else, and an arm that could apply a
    direction where it could not read one would be steering along a vector
    from a different basis. Guessing which element of an unknown structure is
    the activation produces a model that still runs and answers differently,
    which is the failure mode with no symptom.

    Args:
        output: What the module returned.
        shift: The vector to add, broadcast over batch and position.
        module_name: The site, for the refusal message.

    Returns:
        The output with the shift added.

    Raises:
        AppError: With ``EDIT_ACTIVATION_NOT_CAPTURED`` if the output is not a
            tensor.
    """
    if torch.is_tensor(output):
        return output + shift
    raise AppError(
        ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED,
        (
            f"module '{module_name}' returned a {type(output).__name__} rather than a "
            f"tensor, so this site cannot carry a steering direction; the same site must "
            f"be readable by the capture that extracts the direction, and that refuses "
            f"anything but a tensor for the same reason"
        ),
        model_trainer_status_for(ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED),
    )


def attach_steering(
    model: SteerableLMProto, direction: torch.Tensor, *, module_name: str, strength: float
) -> HookHandleProto:
    """Steer a model along a direction until the returned handle is removed.

    A HANDLE RATHER THAN A CONTEXT MANAGER, matching
    :func:`~model_trainer.core.services.model.editing.activations.capture_module_io`,
    which attaches and removes in the same way. The caller removes it; there
    is no ``finally`` arm, because a failure between attach and remove ends
    the measurement anyway and a steered model that outlived its own error
    would be a worse outcome than a leaked hook in a dying process.

    Args:
        model: The model to steer, traceable.
        direction: Unit direction to add along.
        module_name: Dotted path of the module to perturb. The SAME site the
            direction was read at -- a direction read at one module and added
            at another is a vector in the wrong basis.
        strength: Multiplier applied to the unit direction.

    Returns:
        The hook handle, for the caller to remove.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the model has no such
            module.
    """
    module = require_edit_module(model, module_name)
    shift = direction * strength

    def _shift(
        hooked: TracedModuleProto, args: tuple[HookValue, ...], output: HookValue, /
    ) -> torch.Tensor:
        """Add the shift to this module's output.

        Args:
            hooked: The module that ran. Unread: the hook is attached to one
                module, so its identity is already known.
            args: The module's positional arguments. Unread: a steering
                intervention is a function of the output alone.
            output: What it returned.

        Returns:
            The output with the shift added.
        """
        return _steered(output, shift, module_name)

    return module.register_forward_hook(_shift)


def steered_trait_losses(
    model: SteerableLMProto,
    pairs: Sequence[TraitPair],
    direction: torch.Tensor,
    *,
    module_name: str,
    strength: float,
) -> list[TraitPairLosses]:
    """Score every pair steered and unsteered, on one model.

    TWO PASSES RATHER THAN ONE INTERLEAVED LOOP, which is where this differs
    from the cartridge scorer next door. A cartridge arm and its control are
    two different objects -- the wrapper and its own base -- so they can be
    scored pair by pair. A steering arm and its control are the SAME object
    with a hook attached, so the control pass has to finish before the hook
    goes on. The numbers are unaffected: both passes run under ``no_grad`` in
    evaluation mode, where nothing consumes randomness and order carries no
    state.

    THE HOOK IS REMOVED BEFORE ANYTHING IS RETURNED. A steered model that
    outlived this call would silently steer every later arm measured on it,
    and every one of those would look like a result.

    Args:
        model: The base to steer. Never a cartridge-wrapped model: this arm is
            an alternative to a prefix, not an addition to one.
        pairs: The held-out pairs, in order.
        direction: Unit direction to steer along.
        module_name: Dotted path of the module to perturb.
        strength: Multiplier applied to the unit direction.

    Returns:
        One record per pair, in the order given, ready for the same reducers
        the cartridge arms use.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the site does not exist,
            or ``EDIT_ACTIVATION_NOT_CAPTURED`` if it returns a shape this
            cannot perturb.
    """
    model.eval()
    control = [
        (base_loss(model, pair["expressing"]), base_loss(model, pair["neutral"])) for pair in pairs
    ]
    handle = attach_steering(model, direction, module_name=module_name, strength=strength)
    steered = [
        (base_loss(model, pair["expressing"]), base_loss(model, pair["neutral"])) for pair in pairs
    ]
    handle.remove()
    return [
        TraitPairLosses(
            index=index,
            base_expressing=plain_expressing,
            base_neutral=plain_neutral,
            arm_expressing=shifted_expressing,
            arm_neutral=shifted_neutral,
        )
        for index, (
            (plain_expressing, plain_neutral),
            (shifted_expressing, shifted_neutral),
        ) in enumerate(zip(control, steered, strict=True))
    ]


__all__ = [
    "attach_steering",
    "compose_directions",
    "extract_steering_vector",
    "require_steerable",
    "steered_trait_losses",
    "unit_direction",
]
