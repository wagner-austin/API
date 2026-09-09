"""The two refusals a real model cannot reach.

WHAT IS FAKED AND WHY, since this repository does not fake models lightly.
Both arms below fire on a model whose module graph is missing or whose
resolved module is not a torch module -- and every model the loaders return
has a real graph of real modules, so with real weights these branches are
unreachable. That is exactly the case
:mod:`~model_trainer.core.services.model.editing._test_hooks` was written for:
a guard nothing can reach is a guard nobody has checked.

The stand-ins are as small as the protocols allow. Neither pretends to be a
language model; each presents the one shape the narrowing under test asks
about, so a test that passed for the wrong reason would have to have been
written on purpose.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.services.model.editing.sites import require_traceable_model
from model_trainer.core.services.model.editing.value_optimisation import (
    optimise_target_output,
)
from model_trainer.core.types import (
    ConfigLike,
    EditableParameterProto,
    ForwardHookProto,
    ForwardOutProto,
    ForwardPreHookProto,
    HookHandleProto,
    LMModelProto,
    LoadStateDictResultProto,
    NamedParameter,
    ParameterLike,
    TracedLMModelProto,
    TracedModuleProto,
)


class _NoModuleGraph:
    """A language model with no module graph at all.

    Every member :class:`~model_trainer.core.types.LMModelProto` declares and
    none of the four :class:`~model_trainer.core.types.TracedModuleProto`
    does, which is the difference the narrowing exists to detect.
    """

    def train(self) -> None:
        """Do nothing."""

    def eval(self) -> None:
        """Do nothing."""

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Never called.

        Args:
            input_ids: Unused.
            labels: Unused.

        Raises:
            AssertionError: Always. The narrowing refuses this model before
                anything runs it.
        """
        raise AssertionError("a model with no module graph must be refused first")

    def parameters(self) -> Sequence[ParameterLike]:
        """Return nothing."""
        return ()

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return nothing."""
        return ()

    def to(self, device: str) -> LMModelProto:
        """Return self.

        Args:
            device: Unused.

        Returns:
            This model.
        """
        return self

    def save_pretrained(self, out_dir: str) -> None:
        """Never called.

        Args:
            out_dir: Unused.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("nothing here is saved")

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return nothing."""
        return {}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> LoadStateDictResultProto:
        """Never called.

        Args:
            state_dict: Unused.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("nothing here is loaded")

    def gradient_checkpointing_enable(self) -> None:
        """Do nothing."""

    @property
    def config(self) -> ConfigLike:
        """Return a memberless configuration.

        Returns:
            This object, which satisfies the memberless protocol.
        """
        return self


class _GraphHoldingSomethingElse(_NoModuleGraph):
    """A model whose graph names a module that is not a torch module.

    The path resolves, so ``require_edit_module`` is satisfied, and what comes
    back cannot be stood in for -- the gap between "this name exists" and
    "this name is swappable".
    """

    def named_modules(self) -> Generator[tuple[str, TracedModuleProto], None, None]:
        """Yield one name that resolves to a non-module.

        Yields:
            The name, paired with this object.
        """
        yield ("h.0.mlp", self)

    def children(self) -> Generator[TracedModuleProto, None, None]:
        """Yield nothing."""
        return
        yield

    def set_submodule(self, target: str, module: torch.nn.Module) -> None:
        """Never called.

        Args:
            target: Unused.
            module: Unused.

        Raises:
            AssertionError: Always. The narrowing refuses before any swap.
        """
        raise AssertionError("nothing may be swapped into a graph like this")

    def register_forward_hook(self, hook: ForwardHookProto, /) -> HookHandleProto:
        """Never called.

        Args:
            hook: Unused.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("nothing may be hooked on a graph like this")

    def register_forward_pre_hook(self, hook: ForwardPreHookProto, /) -> HookHandleProto:
        """Never called.

        Args:
            hook: Unused.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("nothing may be hooked on a graph like this")

    def to(self, device: str) -> TracedLMModelProto:
        """Return self, still traceable.

        Redeclared because the traced protocol narrows this return type, and
        a stand-in that kept the wider one would not satisfy it.

        Args:
            device: Unused.

        Returns:
            This model.
        """
        return self

    def get_parameter(self, target: str) -> EditableParameterProto:
        """Never called.

        Args:
            target: Unused.

        Raises:
            AssertionError: Always. Nothing here is edited.
        """
        raise AssertionError("nothing here holds a parameter")

    @property
    def training(self) -> bool:
        """Report evaluation mode.

        Returns:
            False, always: nothing here trains.
        """
        return False

    def get_submodule(self, target: str) -> TracedModuleProto:
        """Return something that is not a torch module.

        Args:
            target: The name to resolve.

        Returns:
            This object, which presents the graph methods and is not a
            ``torch.nn.Module``.
        """
        return self


class TestRequireTraceableModel:
    def test_a_model_with_a_graph_is_returned_unchanged(self) -> None:
        model: LMModelProto = _GraphHoldingSomethingElse()

        assert require_traceable_model(model) is model

    def test_a_model_with_no_graph_is_refused(self) -> None:
        """Refused rather than skipped: an arm that silently declined to edit
        would report the unedited model's accuracy under the edited arm's
        name, and nothing in the record would say so.
        """
        with pytest.raises(AppError) as raised:
            require_traceable_model(_NoModuleGraph())

        assert raised.value.code is ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND


class TestOptimiseTargetOutputNarrowing:
    def test_a_resolved_name_that_is_not_a_torch_module_is_refused(self) -> None:
        """`require_edit_module` proves the path resolves. It does not prove
        the thing at the end of it can be stood in for while a value vector
        is learned, and this is the check that does.
        """
        with pytest.raises(AppError) as raised:
            optimise_target_output(
                model=_GraphHoldingSomethingElse(),
                module_name="h.0.mlp",
                prompt_ids=[1, 2],
                target_ids=[3],
                position=1,
                current_output=torch.zeros(4),
                steps=1,
                learning_rate=0.1,
                device="cpu",
            )

        assert raised.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH
