"""Putting a kernel arm inside a model, exercised on the real GPT-2 rung.

Nothing is faked. The real `tiny` model is built by the production builder,
the real modules are swapped, and the real forward pass runs.

WHAT THESE CANNOT COVER, and it is again the point of the instrument: whether
two CARDS agree. What is checkable here is everything that has to be true for
the cross-card answer to mean anything -- that the swap reaches every matmul,
that it changes the arithmetic rather than renaming it, that it does not
change the WEIGHTS, and that the untreated arm is untouched so every trace
taken before the arms existed stays comparable.
"""

from __future__ import annotations

from typing import Protocol

import pytest
import torch

from model_trainer.core.services.model.deterministic_gemm import (
    CUBLAS_ARM,
    KERNEL_ARMS,
    RANK1_ARM,
    matmul_by_arm,
    rank1_addmm,
    rank1_matmul,
)
from model_trainer.core.services.model.kernel_arm_modules import (
    ArmConv1D,
    ArmLinear,
    Conv1DProto,
    SwapTargetProto,
    apply_kernel_arm_to_model,
    require_swappable,
    use_kernel_arm,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import TracedLMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFModel


class _Conv1DCtorProto(Protocol):
    """Constructor for ``transformers`` ``Conv1D``.

    Separate from :class:`Conv1DProto`, which describes an INSTANCE and is
    what ``isinstance`` needs. Mirrors ``_GPT2ConfigCtorProto`` in
    ``backends/gpt2/hf_gpt2.py``, for the same reason: the library ships no
    type information.
    """

    def __call__(self, nf: int, nx: int) -> Conv1DProto: ...


def _make_conv1d() -> Conv1DProto:
    """Build one real transformers Conv1D, through the typed accessor.

    Returns:
        A ``Conv1D(6, 4)`` -- output width 6, input width 4.
    """
    module = __import__("transformers.pytorch_utils", fromlist=["Conv1D"])
    ctor: _Conv1DCtorProto = module.Conv1D
    return ctor(6, 4)


class _BiasedLinearCtorProto(Protocol):
    """Constructor for ``torch.nn.Linear``, narrowed to the biased case."""

    def __call__(self, in_features: int, out_features: int, *, bias: bool) -> torch.nn.Module: ...


class _SequentialCtorProto(Protocol):
    """Constructor for ``torch.nn.Sequential``."""

    def __call__(self, *modules: torch.nn.Module) -> SwapTargetProto: ...


def _biased_tree() -> SwapTargetProto:
    """Build a one-module tree whose Linear carries a bias.

    Reached through a dynamic import for the reason ``Conv1D`` is: a concrete
    ``torch.nn.Sequential`` does not structurally satisfy the protocol under
    this package's typing rules, because torch's real signatures are broader
    than the ones declared for the trace.

    Returns:
        ``Sequential(Linear(4, 4, bias=True))``.
    """
    module = __import__("torch.nn", fromlist=["Sequential", "Linear"])
    linear_ctor: _BiasedLinearCtorProto = module.Linear
    tree_ctor: _SequentialCtorProto = module.Sequential
    return tree_ctor(linear_ctor(4, 4, bias=True))


def _tiny() -> tuple[TracedLMModelProto, torch.Tensor]:
    """Build the real `tiny` rung.

    Returns:
        ``(model, input_ids)`` from the production builder.
    """
    return probe_model_and_input("cpu", PROBE_SHAPES["tiny"])


def _matmul_classes(model: SwapTargetProto) -> set[str]:
    """Names of the classes that carry a matmul in one model.

    Args:
        model: The model to inspect.

    Returns:
        The class names present, restricted to the four this module cares
        about, so an unrelated module appearing cannot change the assertion.
    """
    interesting = {"Conv1D", "Linear", "ArmConv1D", "ArmLinear"}
    return {type(m).__name__ for _, m in model.named_modules()} & interesting


class TestWhatGetsReplaced:
    def test_it_replaces_every_conv1d_and_the_head(self) -> None:
        # GPT-2 tiny is two blocks of four Conv1D, plus lm_head.
        model, _ = _tiny()

        assert use_kernel_arm(model, RANK1_ARM) == 9
        assert _matmul_classes(model) == {"ArmConv1D", "ArmLinear"}

    def test_the_untreated_arm_is_a_no_op(self) -> None:
        # Not "replaced with a wrapper that calls addmm" -- untouched. Every
        # trace taken before the arms existed is a Conv1D record, and
        # rebuilding the untreated path out of wrappers would rename every
        # observation in it for no gain.
        model, _ = _tiny()

        assert use_kernel_arm(model, CUBLAS_ARM) == 0
        assert _matmul_classes(model) == {"Conv1D", "Linear"}

    def test_every_arm_that_is_not_cublas_replaces(self) -> None:
        for arm in KERNEL_ARMS:
            if arm == CUBLAS_ARM:
                continue
            model, _ = _tiny()

            assert use_kernel_arm(model, arm) == 9

    def test_an_unknown_arm_is_refused_before_the_model_is_touched(self) -> None:
        model, _ = _tiny()

        with pytest.raises(ValueError, match="kernel must be one of"):
            use_kernel_arm(model, "triton")

        assert _matmul_classes(model) == {"Conv1D", "Linear"}

    def test_a_biased_linear_is_refused_rather_than_guessed_at(self) -> None:
        # Where F.linear adds a bias relative to its reduction has not been
        # measured, and a probe that guessed would be measuring the guess.
        with pytest.raises(ValueError, match="only replaces bias-free"):
            use_kernel_arm(_biased_tree(), RANK1_ARM)

    def test_the_refusal_names_the_path_it_found(self) -> None:
        # Read on a cluster, detached from the source: "is a Linear" alone is
        # not findable, "0 is a Linear" is.
        with pytest.raises(ValueError, match=r"\A0 is a Linear"):
            use_kernel_arm(_biased_tree(), RANK1_ARM)


class TestReachingAModelThatOnlyDeclaresItselfALanguageModel:
    """The hub loader returns an LMModelProto, which declares no module graph.

    ``FakeHFModel`` is the double: it satisfies ``LMModelProto`` and is NOT a
    torch module, which is precisely the case the narrowing has to handle. It
    is lifted from the hf_lm helpers rather than rewritten, so it cannot drift
    from the protocol the original tracks.

    So the scorer cannot call `use_kernel_arm` directly, and the narrowing has
    to happen somewhere. Where it happens is the behaviour under test: AFTER
    the untreated short-circuit, so that a fake language model stays usable
    for every test that does not exercise an arm.
    """

    def test_the_untreated_arm_never_narrows(self) -> None:
        # A double that is not a torch module at all. If the narrowing ran
        # first this would raise, and every baseline test in the suite would
        # have to build a real module to score nothing.
        assert apply_kernel_arm_to_model(FakeHFModel(), CUBLAS_ARM) == 0

    def test_a_treated_arm_refuses_a_model_it_cannot_reach(self) -> None:
        with pytest.raises(ValueError, match="not a torch module"):
            apply_kernel_arm_to_model(FakeHFModel(), RANK1_ARM)

    def test_an_unknown_arm_is_refused_before_any_narrowing(self) -> None:
        # The arm check comes first, so the error names the real problem
        # rather than complaining about the model.
        with pytest.raises(ValueError, match="kernel must be one of"):
            apply_kernel_arm_to_model(FakeHFModel(), "triton")

    def test_a_real_model_is_reached_and_swapped(self) -> None:
        model, _ = _tiny()

        assert apply_kernel_arm_to_model(model, RANK1_ARM) == 9

    def test_require_swappable_returns_the_same_object(self) -> None:
        model, _ = _tiny()

        assert require_swappable(model) is model


class TestTheSwapPreservesTheModel:
    def test_the_weights_are_the_same_objects(self) -> None:
        # Copied weights would let a difference in the trace be a difference
        # in what was multiplied, which is the one confound that would make
        # the whole measurement unreadable.
        model, _ = _tiny()
        before = dict(model.named_parameters())
        use_kernel_arm(model, RANK1_ARM)
        after = dict(model.named_parameters())

        assert sorted(before) == sorted(after)
        assert all(before[name] is after[name] for name in before)

    def test_the_swapped_model_still_runs_and_reports_a_loss(self) -> None:
        model, ids = _tiny()
        use_kernel_arm(model, RANK1_ARM)

        with torch.no_grad():
            loss = float(model.forward(input_ids=ids, labels=ids).loss.item())

        assert loss > 0.0

    def test_the_swapped_model_reports_the_same_loss_to_rounding(self) -> None:
        # A wrapper that returned zeros would still run and still have the
        # right shapes. The number the probe actually reports has to survive:
        # not bit-identical, since a different reduction order is the whole
        # point, but the same answer.
        model, ids = _tiny()
        with torch.no_grad():
            plain = float(model.forward(input_ids=ids, labels=ids).loss.item())
        swapped_model, swapped_ids = _tiny()
        use_kernel_arm(swapped_model, RANK1_ARM)
        with torch.no_grad():
            treated = float(
                swapped_model.forward(input_ids=swapped_ids, labels=swapped_ids).loss.item()
            )

        assert abs(plain - treated) < 1e-4


class TestTheModulesComputeTheRightThing:
    def test_conv1d_matches_the_transformers_definition(self) -> None:
        # Read from transformers 4.46.3 pytorch_utils.Conv1D.forward: the
        # reshape either side is part of the operation, not decoration.
        torch.manual_seed(7)
        original = _make_conv1d()
        x = torch.randn(2, 3, 4)
        armed = ArmConv1D(original, RANK1_ARM)

        expected = rank1_addmm(original.bias, x.view(-1, 4), original.weight).view(2, 3, 6)

        assert torch.equal(armed.forward(x), expected)

    def test_conv1d_is_close_to_the_module_it_replaces(self) -> None:
        torch.manual_seed(7)
        original = _make_conv1d()
        x = torch.randn(2, 3, 4)

        gap = (ArmConv1D(original, RANK1_ARM).forward(x) - original.forward(x)).abs().max().item()

        assert gap < 1e-5

    def test_linear_matches_a_bias_free_ascending_k_product(self) -> None:
        torch.manual_seed(7)
        original = torch.nn.Linear(4, 6, bias=False)
        x = torch.randn(2, 3, 4)

        expected = rank1_matmul(x.view(-1, 4), original.weight.t()).view(2, 3, 6)

        assert torch.equal(ArmLinear(original, RANK1_ARM).forward(x), expected)

    def test_linear_is_close_to_the_module_it_replaces(self) -> None:
        torch.manual_seed(7)
        original = torch.nn.Linear(4, 6, bias=False)
        x = torch.randn(2, 3, 4)

        gap = (ArmLinear(original, RANK1_ARM).forward(x) - original.forward(x)).abs().max().item()

        assert gap < 1e-5

    def test_the_head_goes_through_the_bias_free_dispatch(self) -> None:
        # A zeroed bias is not free. It can flip the sign bit of a negative
        # zero (0.0 + -0.0 is +0.0), and for the cuBLAS arm it would route
        # lm_head to addmm's fused epilogue -- a DIFFERENT library entry point
        # than the mm an untreated lm_head actually takes, which would make
        # the "untreated" arm untreated in name only.
        torch.manual_seed(7)
        original = torch.nn.Linear(4, 6, bias=False)
        x = torch.randn(2, 3, 4)
        flat = x.view(-1, 4)

        for arm in KERNEL_ARMS:
            expected = matmul_by_arm(arm, flat, original.weight.t()).view(2, 3, 6)

            assert torch.equal(ArmLinear(original, arm).forward(x), expected)

    def test_the_untreated_head_takes_the_same_path_as_an_untouched_linear(self) -> None:
        # The cuBLAS arm of the bias-free dispatch must reproduce F.linear
        # exactly, or the baseline is not the baseline.
        torch.manual_seed(7)
        original = torch.nn.Linear(4, 6, bias=False)
        x = torch.randn(2, 3, 4)

        assert torch.equal(ArmLinear(original, CUBLAS_ARM).forward(x), original.forward(x))

    def test_an_unknown_arm_is_refused_at_construction(self) -> None:
        torch.manual_seed(7)

        with pytest.raises(ValueError, match="kernel must be one of"):
            ArmConv1D(_make_conv1d(), "triton")

    def test_an_unknown_arm_is_refused_at_linear_construction(self) -> None:
        with pytest.raises(ValueError, match="kernel must be one of"):
            ArmLinear(torch.nn.Linear(4, 6, bias=False), "triton")
