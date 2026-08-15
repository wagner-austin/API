"""HuggingFace LM text generation.

Generates text using temperature, top-k, top-p sampling.

Uses hooks from _test_hooks for dependency injection.
Production sets hooks to real implementations at startup.
Tests set hooks to fakes for isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import GenerateConfig, GenerateOutcome, PreparedLMModel

from ._test_hooks import Hooks, ReadTextFileFn


def _read_prompt(cfg: GenerateConfig, settings: Settings) -> str:
    """Read prompt from config prompt_text or prompt_path.

    Args:
        cfg: Generation configuration.
        settings: Application settings.

    Returns:
        Prompt text string.

    Raises:
        AppError: If path is outside data_root or neither prompt provided.
        RuntimeError: If read_text_file hook is not initialized.
    """
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
        read_fn: ReadTextFileFn = Hooks.read_text_file
        return read_fn(resolved)
    raise AppError(
        ModelTrainerErrorCode.CORPUS_NOT_FOUND,
        "either prompt_text or prompt_path must be provided",
        model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
    )


class _GenerateFn(Protocol):
    """Protocol for HuggingFace generate method."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Generate sequences.

        Args:
            input_ids: Input token IDs.
            max_new_tokens: Maximum new tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p (nucleus) filtering.
            num_return_sequences: Number of sequences to return.
            eos_token_id: End of sequence token ID.
            pad_token_id: Padding token ID.

        Returns:
            Generated token IDs tensor.
        """
        ...


def generate_hf_lm(
    *, prepared: PreparedLMModel, cfg: GenerateConfig, settings: Settings
) -> GenerateOutcome:
    """Generate text using a HuggingFace LM model.

    Uses HuggingFace's generate() method with temperature, top-k, top-p sampling.

    Args:
        prepared: Prepared model from prepare_hf_lm_with_handle.
        cfg: Generation configuration.
        settings: Application settings.

    Returns:
        GenerateOutcome with generated texts and metadata.
    """
    prompt = _read_prompt(cfg, settings)
    encoder = prepared.tok_for_dataset

    if cfg["seed"] is not None:
        torch.manual_seed(cfg["seed"])

    encoded = encoder.encode(prompt)
    prompt_ids_list: list[int] = list(encoded.ids)

    batch_ids: list[list[int]] = [prompt_ids_list]
    input_ids = torch.tensor(batch_ids, dtype=torch.long)

    max_prompt_len = max(1, prepared.max_seq_len - cfg["max_new_tokens"])
    if input_ids.size(1) > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]

    model = prepared.model
    model.eval()

    _attr_generate: str = "generate"
    generate_fn: _GenerateFn = getattr(model, _attr_generate)

    with torch.no_grad():
        do_sample = cfg["temperature"] > 0
        temp = max(0.01, cfg["temperature"]) if do_sample else 1.0

        output_ids: torch.Tensor = generate_fn(
            input_ids,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=do_sample,
            temperature=temp,
            top_k=cfg["top_k"] if cfg["top_k"] > 0 else 50,
            top_p=cfg["top_p"],
            num_return_sequences=cfg["num_return_sequences"],
            eos_token_id=prepared.eos_id,
            pad_token_id=prepared.pad_id,
        )

    prompt_len = int(input_ids.size(1))
    outputs: list[str] = []
    eos_terminated_list: list[bool] = []
    total_steps = 0

    for seq_idx in range(output_ids.size(0)):
        seq = output_ids[seq_idx]
        generated_ids: list[int] = [int(seq[i].item()) for i in range(prompt_len, seq.size(0))]

        if prepared.eos_id in generated_ids:
            eos_idx = generated_ids.index(prepared.eos_id)
            eos_terminated_list.append(True)
            if cfg["stop_on_eos"]:
                generated_ids = generated_ids[:eos_idx]
        else:
            eos_terminated_list.append(False)

        text = encoder.decode(generated_ids)

        if cfg["stop_sequences"]:
            for stop in cfg["stop_sequences"]:
                if stop and stop in text:
                    idx = text.index(stop)
                    text = text[:idx]
                    break

        outputs.append(text)
        total_steps += len(generated_ids)

    return GenerateOutcome(
        outputs=outputs,
        steps=total_steps,
        eos_terminated=eos_terminated_list,
    )


__all__ = [
    "generate_hf_lm",
]
