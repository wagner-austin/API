"""Resolve an edit site to the parameter and module it names.

The site is configuration, so every one of its fields can be wrong, and each
wrong field fails differently: a module template that names nothing is a typo
the caller fixes, and a module whose weight is not a matrix is a real module
that this method cannot edit. They get different codes for that reason.

WHY EXISTENCE IS CHECKED BEFORE ASKING. ``torch.nn.Module.get_parameter``
raises ``AttributeError`` with a message about attribute lookup, which is true
and useless: the caller supplied a layer index and a template, and needs to be
told which of those did not resolve. Membership is therefore established
against the model's own inventory first, and the error names the site. Nothing
here catches anything.
"""

from __future__ import annotations

from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.knowledge_edit import EditSite, resolve_edit_module
from model_trainer.core.types import (
    EditableParameterProto,
    LMModelProto,
    TracedLMModelProto,
    TracedModuleProto,
)

#: The attribute a weight-bearing module keeps its matrix in.
WEIGHT_ATTRIBUTE = "weight"


def weight_parameter_name(site: EditSite) -> str:
    """Name the parameter an edit at this site writes into.

    Args:
        site: The site to resolve.

    Returns:
        The dotted parameter path, e.g.
        ``transformer.h.17.mlp.c_proj.weight``.

    Raises:
        JSONTypeError: If the site's module template does not carry exactly
            one layer placeholder.
    """
    return f"{resolve_edit_module(site)}.{WEIGHT_ATTRIBUTE}"


def require_editable_weight(
    model: TracedLMModelProto, parameter_name: str
) -> EditableParameterProto:
    """Return the weight an edit writes into, refusing anything unusable.

    Args:
        model: The model to resolve against.
        parameter_name: Dotted parameter path.

    Returns:
        The parameter, writable.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the model has no parameter
            of that name, or ``EDIT_WEIGHT_NOT_MATRIX`` if it has one and it
            is not two dimensional.
    """
    known = [name for name, _ in model.named_parameters()]
    if parameter_name not in known:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND,
            message=(
                f"no parameter named '{parameter_name}' in this model; "
                f"the edit site names a module this architecture does not have"
            ),
        )
    parameter = model.get_parameter(parameter_name)
    if len(parameter.shape) != 2:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_WEIGHT_NOT_MATRIX,
            message=(
                f"parameter '{parameter_name}' has shape {tuple(parameter.shape)}; "
                f"a rank-one edit writes into a matrix, so this site names a real "
                f"parameter that this method cannot edit"
            ),
        )
    return parameter


def require_edit_module(model: TracedLMModelProto, module_name: str) -> TracedModuleProto:
    """Return the module an edit captures activations from.

    Args:
        model: The model to resolve against.
        module_name: Dotted module path.

    Returns:
        The submodule, traceable so a hook can be attached.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the model has no module of
            that name.
    """
    known = [name for name, _ in model.named_modules()]
    if module_name not in known:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND,
            message=(
                f"no module named '{module_name}' in this model; "
                f"the edit site names a module this architecture does not have"
            ),
        )
    return model.get_submodule(module_name)


def require_traceable_model(model: LMModelProto) -> TracedLMModelProto:
    """Narrow a loaded model to one whose module graph an edit can reach.

    The single place this widening happens, so one error message explains it.
    ``isinstance`` against a runtime-checkable protocol rather than a cast: a
    cast would ASSERT the module graph and this establishes it.

    What the check can see is that the graph methods are present. What it
    cannot see is their signatures -- no runtime protocol check inspects one --
    so :func:`require_edit_module` is still the check with teeth, and it runs
    against the model's own inventory a moment later.

    Args:
        model: The model to narrow, as the hub loader typed it.

    Returns:
        The same model, typed as traceable.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the model presents no
            module graph at all. Refused rather than skipped: an arm that
            silently declined to edit would report the unedited model's
            accuracy under the edited arm's name.
    """
    if isinstance(model, TracedLMModelProto):
        return model
    raise AppError(
        code=ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND,
        message=(
            "this model presents no module graph, so an edit has no module to write "
            "into; weight editing needs a model whose submodules can be walked"
        ),
    )


__all__ = [
    "WEIGHT_ATTRIBUTE",
    "require_edit_module",
    "require_editable_weight",
    "require_traceable_model",
    "weight_parameter_name",
]
