"""Greedy, batched continuation of a whole batch of held-out files at once.

WHY BATCHED. Four-bit generation is dequantization-bound rather than
compute-bound, so a batch costs barely more than a single row. Measured on an
RTX 3090 Ti at 128 new tokens: batch 1 gives 3.3 tokens/s and batch 16 gives
46.8 tokens/s at 1.2 GiB -- fourteen times the throughput for the same
arithmetic. Unbatched, two arms over the held-out set would take about
eighteen hours, which would have forced a smaller sample and a weaker
comparison rather than a slower one.

Batching does perturb the numerics slightly against an unbatched run, because
padded positions change reduction order. That is acceptable here and would
not be acceptable if the arms were batched differently: both arms use the
same batch size and the same length-sorted batch composition, so an item sits
with the same neighbours in both and the comparison stays paired.

WHY GREEDY. The comparison must be free of sampling noise, so an item's
completion is a function of the weights alone. Greedy alone degenerates
though -- measured over a real sweep, completions that ran to the budget
repeated 65-78% of their lines while completions that terminated repeated
11% -- so a repetition penalty is applied, identically to both arms, where
sampling would otherwise have been the fix.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from platform_core.continuation_task import EvalPrompt

from model_trainer.core.contracts.continuation_sweep import Completion
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.services.model.continuation_batch import (
    left_pad,
    split_at_eos,
    truncate_to_tail,
)


class GenerateProto(Protocol):
    """Protocol for the HuggingFace ``generate`` method this calls.

    Declared here rather than on :class:`LMModelProto` for the reason that
    protocol's own docstring gives: what a model instance offers is not the
    same question as what a caller needs, and a protocol that claims more
    than the concrete class has is believed until it is not.
    """

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        repetition_penalty: float,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Generate continuations for a padded batch.

        Args:
            input_ids: Left-padded prompt ids.
            attention_mask: 1 on real positions, 0 on padding.
            max_new_tokens: Token budget per row.
            do_sample: Whether to sample. Always False here.
            repetition_penalty: Penalty on tokens already emitted.
            pad_token_id: What to pad finished rows with.

        Returns:
            The prompt ids followed by the generated ones, per row.
        """
        ...


def generate_batch(
    *,
    model: PreparedLMModel,
    prompts: Sequence[EvalPrompt],
    max_new_tokens: int,
    max_prompt_tokens: int,
    repetition_penalty: float,
    device: str,
    seed: int,
) -> list[Completion]:
    """Continue every prompt in one batch.

    The seed is applied HERE, once per batch, rather than once per run. That
    makes a batch's result independent of how many batches ran before it, so
    a run resumed after a preemption reproduces the batches it redoes instead
    of merely producing plausible replacements for them. Greedy decoding
    should not consume the generator at all; seeding per batch means the
    guarantee does not rest on that being true of every future decoder.

    Args:
        model: The loaded arm, prepared with its own tokenizer and token ids.
        prompts: The batch, already composed by the caller.
        max_new_tokens: Token budget for one completion.
        max_prompt_tokens: How much of each prompt's tail is kept.
        repetition_penalty: Penalty on tokens already emitted.
        device: Where the tensors go.
        seed: Seeds the generator before this batch.

    Returns:
        One completion per prompt, in the order given. Each carries the
        WHOLE file -- prompt followed by continuation -- because that is
        what a checker has to read.

    Raises:
        ValueError: If the batch is empty, or a prompt encodes to no tokens.
            Propagated from :func:`left_pad`; a batch that shows the model
            nothing would otherwise produce files attributed to items that
            were never asked.
    """
    torch.manual_seed(seed)
    encoder = model.tok_for_dataset

    rows = [
        truncate_to_tail(encoder.encode(prompt["prompt"]).ids, max_prompt_tokens)
        for prompt in prompts
    ]
    padded, mask = left_pad(rows, model.pad_id)

    input_ids = torch.tensor(padded, dtype=torch.long).to(device)
    attention_mask = torch.tensor(mask, dtype=torch.long).to(device)

    model.model.eval()
    _attr_generate: str = "generate"
    generate: GenerateProto = getattr(model.model, _attr_generate)

    with torch.no_grad():
        produced = generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=model.pad_id,
        )

    # Left padding made every prompt the same width, so every row's
    # continuation begins at the same column.
    width = int(input_ids.size(1))
    completions: list[Completion] = []
    for index, prompt in enumerate(prompts):
        row = produced[index]
        new_tokens = [int(row[position].item()) for position in range(width, int(row.size(0)))]
        kept, finished = split_at_eos(new_tokens, model.eos_id)
        completions.append(
            Completion(
                item_id=prompt["item_id"],
                text=prompt["prompt"] + encoder.decode(kept),
                finished=finished,
            )
        )
    return completions


__all__ = ["GenerateProto", "generate_batch"]
