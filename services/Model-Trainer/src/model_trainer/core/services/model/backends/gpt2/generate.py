from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import GenerateConfig, GenerateOutcome, PreparedLMModel


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


def _check_stop_sequence(text: str, stop_sequences: Sequence[str]) -> bool:
    """Check if text ends with any stop sequence."""
    return any(stop and text.endswith(stop) for stop in stop_sequences)


class _GenerateFn(Protocol):
    """Local protocol for GPT2 generate method."""

    def __call__(
        self: _GenerateFn,
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
    ) -> torch.Tensor: ...


def generate_gpt2(
    *, prepared: PreparedLMModel, cfg: GenerateConfig, settings: Settings
) -> GenerateOutcome:
    """Generate text using a GPT-2 model.

    Uses HuggingFace's generate() method with temperature, top-k, top-p sampling.
    """
    prompt = _read_prompt(cfg, settings)
    encoder = prepared.tok_for_dataset

    # Set seed if provided
    if cfg["seed"] is not None:
        torch.manual_seed(cfg["seed"])

    # Encode prompt - explicitly type as list[int] to avoid Any
    encoded = encoder.encode(prompt)
    prompt_ids_list: list[int] = list(encoded.ids)

    # Prepare input tensor - explicitly type nested list to avoid Any
    batch_ids: list[list[int]] = [prompt_ids_list]
    input_ids = torch.tensor(batch_ids, dtype=torch.long)

    # Truncate prompt if needed to leave room for generation
    max_prompt_len = max(1, prepared.max_seq_len - cfg["max_new_tokens"])
    if input_ids.size(1) > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]

    model = prepared.model
    model.eval()

    # Access generate method via local Protocol pattern
    # Use variable for attr name to prevent ruff SIM910 simplification
    _attr_generate: str = "generate"
    generate_fn: _GenerateFn = getattr(model, _attr_generate)

    with torch.no_grad():
        # Generate sequences
        # temperature=0 means greedy; HF uses temperature>0 with do_sample=False for greedy
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

    # Process outputs
    prompt_len = int(input_ids.size(1))
    outputs: list[str] = []
    eos_terminated_list: list[bool] = []
    total_steps = 0

    for seq_idx in range(output_ids.size(0)):
        # Get generated portion only - convert to typed list
        seq = output_ids[seq_idx]
        generated_ids: list[int] = [int(seq[i].item()) for i in range(prompt_len, seq.size(0))]

        # Check for EOS termination
        if prepared.eos_id in generated_ids:
            eos_idx = generated_ids.index(prepared.eos_id)
            eos_terminated_list.append(True)
            if cfg["stop_on_eos"]:
                generated_ids = generated_ids[:eos_idx]
        else:
            eos_terminated_list.append(False)

        # Decode
        text = encoder.decode(generated_ids)

        # Check stop sequences and truncate if needed
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
