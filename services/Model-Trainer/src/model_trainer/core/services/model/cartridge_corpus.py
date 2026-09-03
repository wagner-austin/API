"""Turn documents into the fixed windows a cartridge measurement trains on.

Takes ENCODED ids, not text. Tokenizing needs a tokenizer, which needs the
hub or a cache; splitting a list of integers into equal chunks needs neither,
and keeping them apart is what lets the split be tested without either.

WHY THE SPLIT IS A STRIDE AND NOT A TAIL. Holding out the last fraction of a
corpus holds out the last documents, not a sample of it -- on a wiki sorted by
filename that is whichever pages sort last, and a cartridge would be scored
entirely on the subject those happen to cover. Taking every ``n``th window
instead draws held-out rows from every document, so the score is about the
corpus rather than about its alphabetical tail.

THE CONSEQUENCE, STATED BECAUSE IT BOUNDS EVERY NUMBER THIS FEEDS. Adjacent
windows of one document share subject matter, so a held-out window usually has
a training window on either side of it. That makes this a test of
generalisation WITHIN a document, not across documents, and a gain measured
here does not license a claim about text the corpus never approached. The
across-document question needs a split by document, which is a different
split and would answer a different thing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)

#: Smallest stride that leaves anything to train on.
#:
#: At a stride of one every window is held out and the training set is empty.
#: That is not a degenerate configuration to warn about, it is a request for a
#: measurement with no treatment arm.
MIN_HELD_OUT_STRIDE = 2


def _refuse(message: str) -> AppError[ModelTrainerErrorCode]:
    """Build the error a corpus that cannot be measured raises.

    Args:
        message: What is wrong, phrased for the person who chose it.

    Returns:
        The error to raise.
    """
    return AppError(
        ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE,
        message,
        model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE),
    )


def build_windows(
    documents: Sequence[Sequence[int]], *, window: int, device: str
) -> list[torch.Tensor]:
    """Chunk every document into equal windows, dropping each document's tail.

    The tail is dropped rather than padded. A padded window is mostly a
    prediction of the padding token, and its loss would dilute every mean this
    corpus feeds by an amount that depends on how the documents happened to
    divide.

    Chunking does NOT cross a document boundary, so no window is half one
    subject and half another.

    Args:
        documents: Encoded token ids, one sequence per document.
        window: Tokens per window.
        device: Torch device string to build the tensors on.

    Returns:
        One tensor per window, each shaped ``(1, window)``, in document order.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` if the window is not
            positive, or if no document is long enough to fill one window.
    """
    if window <= 0:
        raise _refuse(
            f"a window of {window} tokens describes nothing to score; the window is "
            f"how much text each measured item carries and must be positive"
        )
    built: list[torch.Tensor] = []
    for ids in documents:
        for start in range(0, len(ids) - window + 1, window):
            # Allocated and filled rather than built from a list literal:
            # `torch.tensor([...])` is typed as returning Any, which would put
            # an unchecked value into every caller of this function.
            row = torch.empty((1, window), dtype=torch.long, device=device)
            for offset in range(window):
                row[0, offset] = int(ids[start + offset])
            built.append(row)
    if not built:
        longest = max((len(ids) for ids in documents), default=0)
        raise _refuse(
            f"no document reaches {window} tokens (the longest is {longest}), so the "
            f"corpus yields no windows at all; shorten the window or supply more text"
        )
    return built


def split_by_stride(
    windows: Sequence[torch.Tensor], *, held_out_stride: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Hold out every ``held_out_stride``th window, train on the rest.

    Args:
        windows: Every window in the corpus, in document order.
        held_out_stride: Take one window in this many for the held-out set.

    Returns:
        ``(train, held_out)``.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` if the stride would leave
            either arm empty. Both are checked, because the two failures read
            very differently: no training windows means nothing was learned,
            and no held-out windows means nothing was tested.
    """
    if held_out_stride < MIN_HELD_OUT_STRIDE:
        raise _refuse(
            f"a held-out stride of {held_out_stride} holds out every window and leaves "
            f"nothing to train on; the smallest stride that trains anything is "
            f"{MIN_HELD_OUT_STRIDE}"
        )
    train = [row for index, row in enumerate(windows) if index % held_out_stride != 0]
    held_out = [row for index, row in enumerate(windows) if index % held_out_stride == 0]
    if not held_out:
        raise _refuse(
            f"{len(windows)} window(s) at a stride of {held_out_stride} hold out none of "
            f"them, so the cartridge would be scored on nothing"
        )
    if not train:
        raise _refuse(
            f"{len(windows)} window(s) at a stride of {held_out_stride} leave no training "
            f"windows, so there would be no cartridge to score"
        )
    return train, held_out


__all__ = [
    "MIN_HELD_OUT_STRIDE",
    "build_windows",
    "split_by_stride",
]
