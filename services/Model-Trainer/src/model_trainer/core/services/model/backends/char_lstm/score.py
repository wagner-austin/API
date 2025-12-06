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


class _ForwardLogitsFn(Protocol):
    """Local protocol for CharLSTM forward_logits method with hidden state."""

    def __call__(
        self: _ForwardLogitsFn,
        *,
        input_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return (logits, hidden_state) tuple.

        Args:
            input_ids: Input tensor [B, T] of token indices.
            hidden: Optional (h_n, c_n) hidden state (unused for scoring).

        Returns:
            Tuple of (logits [B, T, V], (h_n, c_n) hidden state).
        """
        ...


def _read_text_or_path(cfg: ScoreConfig, settings: Settings) -> str:
    """Read text from config text or path, validating path is under data_root."""
    if cfg["text"] is not None:
        return cfg["text"]
    if cfg["path"] is not None:
        data_root = Path(settings["app"]["data_root"])
        resolved = Path(cfg["path"]).resolve()
        # Validate path is under data_root for security
        if not str(resolved).startswith(str(data_root.resolve())):
            raise AppError(
                ModelTrainerErrorCode.CORPUS_NOT_FOUND,
                "path must be under data_root",
                model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
            )
        return resolved.read_text(encoding="utf-8")
    raise AppError(
        ModelTrainerErrorCode.CORPUS_NOT_FOUND,
        "either text or path must be provided",
        model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
    )


def _compute_logits(
    prepared: PreparedLMModel, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model forward pass to get logits and per-position loss.

    Returns (logits, per_token_loss) where:
    - logits: [1, T, V] tensor of vocabulary logits
    - per_token_loss: [T] tensor of per-position cross-entropy loss
    """
    model = prepared.model
    model.eval()
    with torch.no_grad():
        # Access forward_logits via local Protocol pattern per refactor doc line 176
        # Use variable for attr name to prevent ruff SIM910 simplification
        _attr_forward_logits: str = "forward_logits"
        forward_logits_fn: _ForwardLogitsFn = getattr(model, _attr_forward_logits)
        # forward_logits returns (logits, hidden_state) - we ignore hidden for scoring
        logits_and_hidden = forward_logits_fn(input_ids=input_ids, hidden=None)
        logits: torch.Tensor = logits_and_hidden[0]  # [1, T, V]

        # Compute per-position loss: compare logits[t] with label[t+1]
        # For next-char prediction, we shift: input[0:T-1] predicts label[1:T]
        if logits.size(1) < 2:
            # Not enough tokens for loss computation
            empty_loss: torch.Tensor = torch.zeros(0)
            return logits, empty_loss

        shift_logits = logits[:, :-1, :]  # [1, T-1, V]
        shift_labels = input_ids[:, 1:]  # [1, T-1]

        # Compute per-token cross-entropy
        # Reshape for cross_entropy: [T-1, V] and [T-1]
        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))  # [T-1, V]
        flat_labels = shift_labels.reshape(-1)  # [T-1]
        per_token_loss = functional.cross_entropy(flat_logits, flat_labels, reduction="none")

        return logits, per_token_loss


def _compute_topk(
    logits: torch.Tensor, encoder: Encoder, k: int
) -> Sequence[Sequence[tuple[str, float]]]:
    """Compute top-k predictions per position.

    Returns list of lists of (token_str, probability) tuples.
    """
    # logits: [1, T, V]
    probs = functional.softmax(logits[0], dim=-1)  # [T, V]
    top_probs, top_indices = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1)  # [T, k]

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


def score_char_lstm(
    *, prepared: PreparedLMModel, cfg: ScoreConfig, settings: Settings
) -> ScoreOutcome:
    """Score text using a char_lstm model.

    Computes loss and perplexity, optionally per-char surprisal and top-k predictions.
    """
    text = _read_text_or_path(cfg, settings)
    encoder = prepared.tok_for_dataset

    # Encode text
    encoded = encoder.encode(text)
    ids = encoded.ids

    if len(ids) < 2:
        # Not enough tokens for loss computation
        return ScoreOutcome(
            loss=0.0,
            perplexity=1.0,
            surprisal=None,
            topk=None,
            tokens=None,
        )

    # Set seed if provided
    if cfg["seed"] is not None:
        torch.manual_seed(cfg["seed"])

    # Prepare input tensor - explicitly type nested list to avoid Any
    ids_list: list[int] = list(ids)
    batch_ids: list[list[int]] = [ids_list]
    input_ids = torch.tensor(batch_ids, dtype=torch.long)  # [1, T]

    # Compute logits and per-token loss
    logits, per_token_loss = _compute_logits(prepared, input_ids)

    # Compute mean loss and perplexity
    # Note: per_token_loss is guaranteed non-empty here because:
    # - We already returned early if len(ids) < 2
    # - With >= 2 tokens, _compute_logits always returns non-empty per_token_loss
    mean_loss = float(per_token_loss.mean().item())
    ppl = float(math.exp(mean_loss)) if mean_loss < 20 else float("inf")

    # Compute optional per-char surprisal
    surprisal: Sequence[float] | None = None
    if cfg["detail_level"] == "per_char":
        # Surprisal is log2 of loss (convert from nats to bits)
        # Use explicit indexing to avoid Any from tensor iteration
        num_tokens = int(per_token_loss.numel())
        surprisal_list: list[float] = [
            float(per_token_loss[i].item()) / math.log(2) for i in range(num_tokens)
        ]
        surprisal = surprisal_list

    # Compute optional top-k predictions
    topk: Sequence[Sequence[tuple[str, float]]] | None = None
    if cfg["top_k"] is not None and cfg["top_k"] > 0:
        topk = _compute_topk(logits, encoder, cfg["top_k"])

    # Compute tokens list for per_char mode
    tokens: Sequence[str] | None = None
    if cfg["detail_level"] == "per_char":
        tokens = [encoder.decode([tid]) for tid in ids]

    return ScoreOutcome(
        loss=mean_loss,
        perplexity=ppl,
        surprisal=surprisal,
        topk=topk,
        tokens=tokens,
    )
