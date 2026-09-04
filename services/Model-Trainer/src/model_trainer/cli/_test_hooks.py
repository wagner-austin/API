"""Hooks for the command-line entries - production defaults, tests override.

Production sets these to the real implementations at import. Tests replace
them with fakes before exercising the code under test, so there is no
conditional in the entry itself -- it calls the hook.

Only the two seams that need real weights and a real GPU are here. Everything
else in the scorer is pure and is exercised directly, because a fake in front
of pure code tests the fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from platform_core.continuation_task import EvalPrompt
from platform_core.determinism_record import DeterminismRecord
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)

# Imported rather than restated. This module carried its own copy of
# `ApplyDeterminismProto`, identical to the core one and with no reason to be
# separate -- `PinTorchThreadsProto` was already coming from here, so the
# usual excuse (keeping torch out of a laptop-side import) did not apply.
# The fork was invisible until the core protocol gained an argument and only
# one of the two copies changed.
from model_trainer.core._hook_protocols_ml import (
    ApplyDeterminismProto,
    PinTorchThreadsProto,
)
from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItem
from model_trainer.core.contracts.continuation_sweep import Completion, ContinuationArm
from model_trainer.core.contracts.model import PreparedLMModel


class LoadHubModelProto(Protocol):
    """Protocol for loading an untrained model straight from the hub."""

    def __call__(self, hub_model_id: str, /) -> PreparedLMModel:
        """Load the named model with nothing applied to it."""
        ...


class LoadContinuationArmProto(Protocol):
    """Protocol for loading one arm of a continuation sweep.

    Both arms come from the SAME saved run, which is why this takes an
    artifact path and an arm name rather than a model id: the control for a
    QLoRA adapter is that adapter's own base under that adapter's own
    quantization, and reading both out of one metadata file is what makes
    the pair impossible to mismatch by hand.
    """

    def __call__(self, artifact_path: str, arm: ContinuationArm, /) -> PreparedLMModel:
        """Load the named arm of the saved run."""
        ...


class GenerateContinuationBatchProto(Protocol):
    """Protocol for decoding one batch of continuations.

    Keyword-only, matching the real signature: seven arguments of which four
    are integers, and transposing two of them would produce a sweep that
    runs and is wrong.
    """

    def __call__(
        self,
        *,
        model: PreparedLMModel,
        prompts: Sequence[EvalPrompt],
        max_new_tokens: int,
        max_prompt_tokens: int,
        repetition_penalty: float,
        device: str,
        seed: int,
    ) -> list[Completion]:
        """Continue every prompt in one batch."""
        ...


class ReadCorpusDocumentsProto(Protocol):
    """Protocol for reading a measurement corpus off the filesystem."""

    def __call__(self, corpus_dir: Path, /) -> tuple[str, ...]:
        """Read every document in a corpus directory, in a fixed order."""
        ...


class ScoreClozeProto(Protocol):
    """Protocol for the cloze scorer.

    Keyword-only, matching the real signature: the scorer takes five
    arguments whose order carries no meaning and would be easy to transpose.
    """

    def __call__(
        self,
        *,
        items: list[ClozeItem],
        model: PreparedLMModel,
        device: str,
        max_seq_len: int,
    ) -> ClozeEvalResult:
        """Score every item and report accuracy against the guessing baseline."""
        ...


class EnvCublasltWorkspaceProto(Protocol):
    """Protocol for reading the split-K condition out of the environment.

    Behind a hook because it reads process-global configuration a test must
    be able to set both ways without a real cuBLASLt, and because reading it
    is the only way a record can say which condition produced it -- see
    :data:`~model_trainer.core.services.model.trace_plan.WORKSPACE_NAME`.
    """

    def __call__(self) -> str | None:
        """Return ``CUBLASLT_WORKSPACE_SIZE``, or None when it is not set."""
        ...


def _default_env_cublaslt_workspace() -> str | None:
    """Production reader for the split-K condition - used as default hook.

    Goes through ``config_test_hooks.get_env`` rather than ``os.environ``,
    which is the read this monorepo's env guard exists to stop.

    Returns:
        The variable's value, or None when unset or empty. Empty is treated
        as unset because cuBLASLt itself ignores an empty value, and a record
        claiming a condition its library did not apply would be worse than
        one saying it does not know.
    """
    from platform_core.config import config_test_hooks
    from platform_core.determinism_env import CUBLASLT_WORKSPACE_ENV_VAR

    value = config_test_hooks.get_env(CUBLASLT_WORKSPACE_ENV_VAR)
    return value if value else None


class RunBenchmarkChildProto(Protocol):
    """Protocol for spawning the benchmark's second-condition process.

    Behind a hook because the child is a real subprocess that needs a GPU and
    several seconds of timing. A test can drive the parent's plumbing --
    including the refusal when the child comes back with the wrong
    condition -- without paying for either.
    """

    def __call__(self, argv: list[str], variable: str, value: str, /) -> int:
        """Run the child with one variable set, and return its exit code."""
        ...


def _default_run_benchmark_child(argv: list[str], variable: str, value: str, /) -> int:
    """Production child spawner - used as default hook.

    ``os.putenv`` and an INHERITED environment, rather than building a full
    mapping to hand ``subprocess``. Two reasons, and the first is the monorepo
    rule: reading ``os.environ`` to assemble that mapping is the config read
    the env guard exists to stop, while ``putenv`` is a write -- the same
    distinction ``core/_test_hooks`` already relies on to set
    ``CUBLAS_WORKSPACE_CONFIG``. The second is that putenv reaches the real
    process environment, which is what a child inherits and what cuBLASLt's
    own getenv reads.

    Args:
        argv: The command line to run.
        variable: The variable to set for the child.
        value: What to set it to. This is the whole reason the child exists:
            ``CUBLASLT_WORKSPACE_SIZE`` is read once when the cuBLASLt handle
            is created, so a process that already has one cannot change
            condition -- measured, two calls with it set between them both
            still used split-K.

    Returns:
        The child's exit code.
    """
    import os
    import subprocess

    os.putenv(variable, value)
    return subprocess.run(argv, check=False).returncode


def _default_load_continuation_arm(artifact_path: str, arm: ContinuationArm, /) -> PreparedLMModel:
    """Production arm loader - used as default hook.

    Imported inside the function for the same reason the hub loader's import
    is: parsing a command line and printing a usage error must not pull torch
    into the process.

    Args:
        artifact_path: The saved training run both arms are defined by.
        arm: Which side to load. ``candidate`` reattaches the adapter;
            ``base`` loads the weights it was trained against and attaches
            nothing.

    Returns:
        The prepared model.
    """
    from model_trainer.core.services.model.backends.hf_lm.io import (
        load_base_of_prepared_hf_lm,
        load_prepared_hf_lm_from_handle,
    )

    if arm == "candidate":
        return load_prepared_hf_lm_from_handle(artifact_path, None)
    return load_base_of_prepared_hf_lm(artifact_path)


def _default_generate_continuation_batch(
    *,
    model: PreparedLMModel,
    prompts: Sequence[EvalPrompt],
    max_new_tokens: int,
    max_prompt_tokens: int,
    repetition_penalty: float,
    device: str,
    seed: int,
) -> list[Completion]:
    """Production batched decoder - used as default hook.

    Args:
        model: The loaded arm.
        prompts: The batch, already composed.
        max_new_tokens: Token budget for one completion.
        max_prompt_tokens: How much of each prompt's tail is kept.
        repetition_penalty: Penalty on tokens already emitted.
        device: Where the tensors go.
        seed: Seeds the generator before this batch.

    Returns:
        One completion per prompt, in the order given.
    """
    from model_trainer.core.services.model.continuations import generate_batch

    return generate_batch(
        model=model,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=max_prompt_tokens,
        repetition_penalty=repetition_penalty,
        device=device,
        seed=seed,
    )


def _default_load_hub_model(hub_model_id: str, /) -> PreparedLMModel:
    """Production hub loader - used as default hook.

    Imported inside the function so that importing this module does not pull
    torch into a process that only wanted to parse a command line and print
    a usage error.

    Args:
        hub_model_id: HuggingFace model id, for example ``gpt2-medium``.

    Returns:
        The prepared model, with nothing applied to it.
    """
    from model_trainer.core.services.model.backends.hf_lm.io import (
        load_prepared_hf_lm_from_hub,
    )

    return load_prepared_hf_lm_from_hub(hub_model_id)


def _default_score_cloze(
    *,
    items: list[ClozeItem],
    model: PreparedLMModel,
    device: str,
    max_seq_len: int,
) -> ClozeEvalResult:
    """Production scorer - used as default hook.

    Unpacks the prepared model into the model and its encoder, which is the
    only reason this is not the scorer itself: the entry holds a
    PreparedLMModel and the scorer takes the two halves.

    Args:
        items: The cloze items to score.
        model: The prepared model and its encoder.
        device: Device to score on.
        max_seq_len: Token budget per item.

    Returns:
        The scored result.
    """
    from model_trainer.core.services.model.cloze import score_cloze_items

    return score_cloze_items(
        items=items,
        model=model.model,
        encoder=model.tok_for_dataset,
        device=device,
        max_seq_len=max_seq_len,
    )


def _default_apply_determinism(*, remove_split_k: bool, math_attention: bool) -> DeterminismRecord:
    """Production determinism pin - used as default hook.

    Delegates to the same hook the workers use, so a run scored from the
    command line and one scored through the queue pin identically. A second
    spelling here would be a second posture nobody noticed diverging.

    Args:
        remove_split_k: Forwarded unchanged. The CLI tier is where the two
            populations actually differ -- a scoring command pins like a
            worker, a measurement command deliberately does not -- so this
            passes the caller's choice on rather than making one.
        math_attention: Forwarded unchanged, same reasoning.

    Returns:
        What was actually applied.
    """
    from model_trainer.core import _test_hooks as core_hooks

    return core_hooks.apply_determinism_hook(
        remove_split_k=remove_split_k, math_attention=math_attention
    )


def _default_pin_torch_threads(threads: int) -> int:
    """Production torch thread pin - used as default hook.

    Delegates to the worker's hook for the same reason
    :func:`_default_apply_determinism` does: a probe run from the command
    line and a job run through the queue must pin by the same call, or the
    two postures diverge without anyone noticing.

    Args:
        threads: Count to request.

    Returns:
        The count torch resolved to, which may differ from the request.
    """
    from model_trainer.core import _test_hooks as core_hooks

    return core_hooks.pin_torch_threads(threads)


#: Opening and closing line of a markdown document's YAML frontmatter.
_FRONTMATTER_FENCE = "---\n"


def _strip_frontmatter(text: str, path: Path) -> str:
    """Return a markdown document's body, without its YAML frontmatter.

    The frontmatter is removed rather than trained on. A cartridge measured
    over pages that still carry their frontmatter spends part of its prefix
    learning to predict ``tags:`` and ``fact_checked:``, and the gain it
    reports is then partly a gain at predicting YAML -- which is real, and is
    not the thing anybody asked.

    Args:
        text: The document's full text.
        path: Where it came from, for the error message.

    Returns:
        The body. The text unchanged when it opens no frontmatter fence,
        because a document without frontmatter is all body.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when a document opens a
            fence it never closes. Treating that as body would silently train
            on the YAML this function exists to remove.
    """
    if not text.startswith(_FRONTMATTER_FENCE):
        return text
    closing = text.find("\n" + _FRONTMATTER_FENCE, len(_FRONTMATTER_FENCE) - 1)
    if closing == -1:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE,
            (
                f"{path} opens a frontmatter fence and never closes it, so its body "
                f"cannot be told from its metadata; measuring it would train the "
                f"cartridge on YAML"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE),
        )
    return text[closing + len(_FRONTMATTER_FENCE) + 1 :]


def _default_read_corpus_documents(corpus_dir: Path, /) -> tuple[str, ...]:
    """Production implementation - read a directory of markdown documents.

    Sorted by filename, because the order decides which windows the stride
    holds out, and an unsorted directory listing would make the held-out set
    depend on the filesystem.

    Args:
        corpus_dir: Directory holding the corpus.

    Returns:
        Every document's body, in filename order. Documents that are empty
        once their frontmatter is removed are dropped -- they contribute no
        window, and carrying them would put entries in the corpus digest that
        no measurement can see.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` if the directory holds no
            markdown at all.
    """
    bodies = [
        _strip_frontmatter(path.read_text(encoding="utf-8"), path).strip()
        for path in sorted(corpus_dir.glob("*.md"))
    ]
    kept = tuple(body for body in bodies if body)
    if not kept:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE,
            (
                f"{corpus_dir} holds no markdown document with a body; a cartridge "
                f"measurement needs text to train on"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE),
        )
    return kept


load_hub_model: LoadHubModelProto = _default_load_hub_model

load_continuation_arm: LoadContinuationArmProto = _default_load_continuation_arm

generate_continuation_batch: GenerateContinuationBatchProto = _default_generate_continuation_batch

read_corpus_documents: ReadCorpusDocumentsProto = _default_read_corpus_documents

score_cloze: ScoreClozeProto = _default_score_cloze

apply_determinism_hook: ApplyDeterminismProto = _default_apply_determinism

pin_torch_threads: PinTorchThreadsProto = _default_pin_torch_threads

env_cublaslt_workspace: EnvCublasltWorkspaceProto = _default_env_cublaslt_workspace

run_benchmark_child: RunBenchmarkChildProto = _default_run_benchmark_child


__all__ = [
    "ApplyDeterminismProto",
    "EnvCublasltWorkspaceProto",
    "GenerateContinuationBatchProto",
    "LoadContinuationArmProto",
    "LoadHubModelProto",
    "ReadCorpusDocumentsProto",
    "RunBenchmarkChildProto",
    "ScoreClozeProto",
    "apply_determinism_hook",
    "env_cublaslt_workspace",
    "generate_continuation_batch",
    "load_continuation_arm",
    "load_hub_model",
    "pin_torch_threads",
    "read_corpus_documents",
    "run_benchmark_child",
    "score_cloze",
]
