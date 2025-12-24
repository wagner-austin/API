"""HuggingFace LM text scoring.

Computes loss, perplexity, and optionally per-token surprisal.

Uses hooks from _test_hooks for dependency injection.
Production sets hooks to real implementations at startup.
Tests set hooks to fakes for isolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as functional
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import PreparedLMModel, ScoreConfig, ScoreOutcome
from model_trainer.core.encoding import Encoder

from ._test_hooks import Hooks, ReadTextFileFn


def _read_text_or_path(cfg: ScoreConfig, settings: Settings) -> str:
    """Read text from config text or path, validating path is under data_root.

    Args:
        cfg: Score configuration.
        settings: Application settings.

    Returns:
        Text string to score.

    Raises:
        AppError: If path is outside data_root or neither text nor path provided.
        RuntimeError: If read_text_file hook is not initialized.
    """
    if cfg["text"] is not None:
        return cfg["text"]
    if cfg["path"] is not None:
        data_root = Path(settings["app"]["data_root"])
        resolved = Path(cfg["path"]).resolve()
        if not str(resolved).startswith(str(data_root.resolve())):
            raise AppError(
                ModelTrainerErrorCode.CORPUS_NOT_FOUND,
                "path must be under data_root",
                model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
            )
        read_fn: ReadTextFileFn | None = Hooks.read_text_file
        if read_fn is None:
            raise RuntimeError(
                "Hooks.read_text_file not initialized - call init_production_hooks()"
            )
        return read_fn(resolved)
    raise AppError(
        ModelTrainerErrorCode.CORPUS_NOT_FOUND,
        "either text or path must be provided",
        model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
    )


class _ForwardOut(Protocol):
    """Protocol for model forward output with logits."""

    @property
    def logits(self) -> torch.Tensor:
        """Logits tensor [batch, seq, vocab]."""
        ...


class _ForwardFn(Protocol):
    """Protocol for model forward function."""

    def __call__(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> _ForwardOut:
        """Run forward pass.

        Args:
            input_ids: Input token IDs.
            labels: Labels for loss computation.

        Returns:
            Output with logits.
        """
        ...


def _get_logits_and_loss(
    prepared: PreparedLMModel, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model forward pass to get logits and per-position loss.

    Args:
        prepared: Prepared model.
        input_ids: Input tensor [1, T].

    Returns:
        Tuple of (logits [1, T, V], per_token_loss [T-1]).
    """
    model = prepared.model
    model.eval()

    with torch.no_grad():
        _attr_forward: str = "forward"
        forward_fn: _ForwardFn = getattr(model, _attr_forward)
        output = forward_fn(input_ids=input_ids, labels=input_ids)
        logits: torch.Tensor = output.logits

        if logits.size(1) < 2:
            empty_loss: torch.Tensor = torch.zeros(0)
            return logits, empty_loss

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        flat_labels = shift_labels.reshape(-1)
        per_token_loss = functional.cross_entropy(flat_logits, flat_labels, reduction="none")

        return logits, per_token_loss


def _compute_topk(
    logits: torch.Tensor, encoder: Encoder, k: int
) -> Sequence[Sequence[tuple[str, float]]]:
    """Compute top-k predictions per position.

    Args:
        logits: Logits tensor [1, T, V].
        encoder: Encoder for decoding tokens.
        k: Number of top predictions.

    Returns:
        Sequence of top-k (token, probability) tuples per position.
    """
    probs = functional.softmax(logits[0], dim=-1)
    top_probs, top_indices = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1)

    result: list[list[tuple[str, float]]] = []
    for t in range(top_probs.size(0)):
        position_topk: list[tuple[str, float]] = []
        for i in range(top_probs.size(1)):
            token_id = int(top_indices[t, i].item())
            prob = float(top_probs[t, i].item())
            token_str = encoder.decode([token_id])
            position_topk.append((token_str, prob))
        result.append(position_topk)
    return result


def score_hf_lm(*, prepared: PreparedLMModel, cfg: ScoreConfig, settings: Settings) -> ScoreOutcome:
    """Score text using a HuggingFace LM model.

    Computes loss and perplexity, optionally per-token surprisal and top-k predictions.

    Args:
        prepared: Prepared model from prepare_hf_lm_with_handle.
        cfg: Scoring configuration.
        settings: Application settings.

    Returns:
        ScoreOutcome with loss, perplexity, and optional details.
    """
    text = _read_text_or_path(cfg, settings)
    encoder = prepared.tok_for_dataset

    encoded = encoder.encode(text)
    ids = encoded.ids

    if len(ids) < 2:
        return ScoreOutcome(
            loss=0.0,
            perplexity=1.0,
            surprisal=None,
            topk=None,
            tokens=None,
        )

    if cfg["seed"] is not None:
        torch.manual_seed(cfg["seed"])

    ids_list: list[int] = list(ids)
    batch_ids: list[list[int]] = [ids_list]
    input_ids = torch.tensor(batch_ids, dtype=torch.long)

    if input_ids.size(1) > prepared.max_seq_len:
        input_ids = input_ids[:, : prepared.max_seq_len]

    logits, per_token_loss = _get_logits_and_loss(prepared, input_ids)

    # Handle empty tensor case (single token input or model returns single position)
    if per_token_loss.numel() == 0:
        mean_loss = 0.0
        ppl = 1.0
    else:
        mean_loss = float(per_token_loss.mean().item())
        ppl = float(math.exp(mean_loss)) if mean_loss < 20 else float("inf")

    surprisal: Sequence[float] | None = None
    if cfg["detail_level"] == "per_char":
        num_tokens = int(per_token_loss.numel())
        surprisal_list: list[float] = [
            float(per_token_loss[i].item()) / math.log(2) for i in range(num_tokens)
        ]
        surprisal = surprisal_list

    topk: Sequence[Sequence[tuple[str, float]]] | None = None
    if cfg["top_k"] is not None and cfg["top_k"] > 0:
        topk = _compute_topk(logits, encoder, cfg["top_k"])

    tokens: Sequence[str] | None = None
    if cfg["detail_level"] == "per_char":
        actual_len = int(input_ids.size(1))
        tokens = [encoder.decode([ids_list[i]]) for i in range(actual_len)]

    return ScoreOutcome(
        loss=mean_loss,
        perplexity=ppl,
        surprisal=surprisal,
        topk=topk,
        tokens=tokens,
    )


__all__ = [
    "score_hf_lm",
]
