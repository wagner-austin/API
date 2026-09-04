"""Turning a batch of prompts into padded rows, and rows back into completions.

Deliberately free of torch, because none of this is arithmetic. It is the
bookkeeping either side of the one call that is -- and it is exactly the
bookkeeping that is easy to get subtly wrong in a way no exception reports:
pad on the wrong side and every completion begins after a run of pad tokens;
truncate the wrong end and the model is asked to continue from a place the
file does not stop at; forget to cut at the end-of-sequence token and the
scored file ends in whatever the padding decoded to.

Each of those produces a file, and a file is what the scorer scores.
"""

from __future__ import annotations

from collections.abc import Sequence


def truncate_to_tail(ids: Sequence[int], budget: int) -> list[int]:
    """Keep the END of an over-long prompt.

    The completion has to continue from where the prompt stops, so the tail
    is the load-bearing half. Truncating the tail instead would hand the
    model a prompt that stops in a different place than the file it will be
    appended to, and the scored file would then have a seam in the middle
    that no model wrote.

    Args:
        ids: The prompt's token ids.
        budget: How many may be kept.

    Returns:
        At most ``budget`` ids, taken from the end.

    Raises:
        ValueError: If ``budget`` is not positive. A zero-token prompt asks
            the model to write a file from nothing.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    return list(ids[-budget:])


def left_pad(rows: Sequence[Sequence[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    """Pad rows to a common width on the LEFT, with the mask that goes with it.

    Left padding is required for batched decoder-only generation. Right
    padding would put pad tokens between the prompt and the first generated
    token, so the model would be continuing from the padding rather than
    from the file. It also makes the prompt width uniform, which is what
    lets the caller slice every row's completion at the same column.

    Args:
        rows: One list of token ids per prompt.
        pad_id: The token to pad with.

    Returns:
        A tuple of (padded rows, attention mask). The mask is 0 where a
        position is padding and 1 where it is real, which is what stops the
        model attending to tokens nobody wrote.

    Raises:
        ValueError: If ``rows`` is empty, or if any row is. An empty batch
            has no width to pad to, and an empty prompt is the zero-token
            case :func:`truncate_to_tail` already refuses -- both would
            otherwise reach the model and produce a completion attributed to
            an item that was never shown anything.
    """
    if not rows:
        raise ValueError("cannot pad an empty batch")
    for index, row in enumerate(rows):
        if not row:
            raise ValueError(
                f"row {index} is empty; a prompt with no tokens shows the model nothing"
            )
    width = max(len(row) for row in rows)
    padded: list[list[int]] = []
    mask: list[list[int]] = []
    for row in rows:
        gap = width - len(row)
        padded.append([pad_id] * gap + list(row))
        mask.append([0] * gap + [1] * len(row))
    return padded, mask


def split_at_eos(new_tokens: Sequence[int], eos_id: int) -> tuple[list[int], bool]:
    """Cut a completion at its end-of-sequence token, and say whether it had one.

    Cut rather than decoded whole. Everything after the first end-of-sequence
    token is padding the batch needed for its other rows, and decoding it
    appends text to a file the model considered finished.

    Whether the token appeared is RECORDED rather than inferred from the
    text later, because a file that ends on a plausible-looking line and a
    file that ends because the budget did are indistinguishable once the
    tokens are gone -- and the difference is the whole reason the first
    sweep of this comparison was void.

    Args:
        new_tokens: The ids the model generated, prompt excluded.
        eos_id: The end-of-sequence token id.

    Returns:
        A tuple of (ids to decode, whether the model stopped on its own).
    """
    ids = list(new_tokens)
    if eos_id in ids:
        return ids[: ids.index(eos_id)], True
    return ids, False


__all__ = ["left_pad", "split_at_eos", "truncate_to_tail"]
