from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as functional
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import GenerateConfig, GenerateOutcome, PreparedLMModel
from model_trainer.core.encoding import Encoder


class _ForwardLogitsFn(Protocol):
    """Local protocol for CharLSTM forward_logits method with stateful generation."""

    def __call__(
        self: _ForwardLogitsFn,
        *,
        input_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return (logits, hidden_state) for stateful generation.

        Args:
            input_ids: Input tensor [B, T] of token indices.
            hidden: Optional (h_n, c_n) hidden state from previous step.

        Returns:
            Tuple of (logits [B, T, V], (h_n, c_n) hidden state).
        """
        ...


def _read_prompt(cfg: GenerateConfig, settings: Settings) -> str:
    """Read prompt from config prompt_text or prompt_path."""
    if cfg["prompt_text"] is not None:
        return cfg["prompt_text"]
    if cfg["prompt_path"] is not None:
        data_root = Path(settings["app"]["data_root"])
        resolved = Path(cfg["prompt_path"]).resolve()
        if not str(resolved).startswith(str(data_root.resolve())):
            raise AppError(
                ModelTrainerErrorCode.CORPUS_NOT_FOUND,
                "prompt_path must be under data_root",
                model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
            )
        return resolved.read_text(encoding="utf-8")
    raise AppError(
        ModelTrainerErrorCode.CORPUS_NOT_FOUND,
        "either prompt_text or prompt_path must be provided",
        model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
    )


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample a single token from logits using temperature, top-k, and top-p.

    Args:
        logits: [V] tensor of vocabulary logits
        temperature: Temperature for sampling (higher = more random)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Keep tokens with cumulative probability <= top_p (nucleus sampling)

    Returns:
        Sampled token id
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Greedy: return argmax
        return int(logits.argmax().item())

    # Apply top-k filtering
    if top_k > 0:
        top_k_val = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Apply top-p (nucleus) filtering
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Sample from the filtered distribution
    probs = functional.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


def _check_stop_sequence(text: str, stop_sequences: Sequence[str]) -> bool:
    """Check if text ends with any stop sequence."""
    return any(stop and text.endswith(stop) for stop in stop_sequences)


def _generate_single(
    prepared: PreparedLMModel,
    encoder: Encoder,
    prompt_ids: list[int],
    cfg: GenerateConfig,
) -> tuple[str, int, bool]:
    """Generate a single sequence using stateful LSTM generation.

    Uses hidden state passing for efficient autoregressive generation.
    Instead of re-running the entire sequence each step, we:
    1. Run the prompt through once to get initial hidden state
    2. For each new token, only process that single token with previous hidden

    Returns (generated_text, steps, eos_terminated).
    """
    model = prepared.model
    model.eval()

    # Access forward_logits via local Protocol pattern per refactor doc line 176
    # Use variable for attr name to prevent ruff SIM910 simplification
    _attr_forward_logits: str = "forward_logits"
    forward_logits_fn: _ForwardLogitsFn = getattr(model, _attr_forward_logits)

    generated_ids: list[int] = []
    eos_id = prepared.eos_id
    max_len = prepared.max_seq_len
    steps = 0
    eos_terminated = False

    with torch.no_grad():
        # Step 1: Process the prompt to get initial hidden state
        prompt_batch: list[list[int]] = [prompt_ids]
        prompt_tensor = torch.tensor(prompt_batch, dtype=torch.long)
        logits, hidden = forward_logits_fn(input_ids=prompt_tensor, hidden=None)

        # Get logits for last prompt position to sample first generated token
        last_logits: torch.Tensor = logits[0, -1, :]  # [V]

        # Step 2: Autoregressive generation with hidden state
        for _ in range(cfg["max_new_tokens"]):
            # Check max sequence length (prompt + generated so far)
            if len(prompt_ids) + len(generated_ids) >= max_len:
                break

            # Sample next token from current logits
            next_token = _sample_token(
                last_logits,
                temperature=cfg["temperature"],
                top_k=cfg["top_k"],
                top_p=cfg["top_p"],
            )
            steps += 1

            # Check for EOS
            if cfg["stop_on_eos"] and next_token == eos_id:
                eos_terminated = True
                break

            # Append token
            generated_ids.append(next_token)

            # Check stop sequences
            if cfg["stop_sequences"]:
                current_text = encoder.decode(generated_ids)
                if _check_stop_sequence(current_text, cfg["stop_sequences"]):
                    break

            # Forward pass for just the new token, passing hidden state
            # This is the key efficiency: process 1 token instead of entire sequence
            next_batch: list[list[int]] = [[next_token]]
            next_tensor = torch.tensor(next_batch, dtype=torch.long)
            logits, hidden = forward_logits_fn(input_ids=next_tensor, hidden=hidden)

            # Get logits for the single position (the new token)
            last_logits = logits[0, 0, :]  # [V]

    # Decode generated portion
    generated_text = encoder.decode(generated_ids)

    return generated_text, steps, eos_terminated


def generate_char_lstm(
    *, prepared: PreparedLMModel, cfg: GenerateConfig, settings: Settings
) -> GenerateOutcome:
    """Generate text using a char_lstm model.

    Supports multiple return sequences, temperature, top-k, top-p sampling.
    """
    prompt = _read_prompt(cfg, settings)
    encoder = prepared.tok_for_dataset

    # Set seed if provided
    if cfg["seed"] is not None:
        torch.manual_seed(cfg["seed"])

    # Encode prompt - explicitly type as list[int] to avoid Any
    encoded = encoder.encode(prompt)
    prompt_ids: list[int] = list(encoded.ids)

    # Generate num_return_sequences outputs
    outputs: list[str] = []
    eos_terminated_list: list[bool] = []
    total_steps = 0

    for _ in range(cfg["num_return_sequences"]):
        text, steps, eos_term = _generate_single(prepared, encoder, prompt_ids, cfg)
        outputs.append(text)
        eos_terminated_list.append(eos_term)
        total_steps += steps

    return GenerateOutcome(
        outputs=outputs,
        steps=total_steps,
        eos_terminated=eos_terminated_list,
    )
