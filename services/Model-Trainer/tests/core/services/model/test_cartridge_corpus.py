"""Windows, and the split that decides what a held-out gain is a gain AT.

The split is the part with a claim in it. Taking every fourth window rather
than the last quarter is what keeps a score from being about whichever
documents sort last, and the cost of that choice -- a held-out window usually
has training windows on either side of it -- bounds every number the
measurement reports.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.services.model.cartridge_corpus import (
    MIN_HELD_OUT_STRIDE,
    build_windows,
    split_by_stride,
)


def _ids(windows: list[torch.Tensor]) -> list[list[int]]:
    """Read every window's token ids back as plain integers.

    ``Tensor.tolist`` is untyped, so reading a window through it would put an
    ``Any`` into every assertion below and the type checker would stop seeing
    what is being compared.

    Args:
        windows: The windows to read.

    Returns:
        One list of ids per window, batch dimension dropped.
    """
    return [[int(row[0][position]) for position in range(int(row.shape[1]))] for row in windows]


class TestBuildWindows:
    def test_it_cuts_equal_windows_in_document_order(self) -> None:
        built = build_windows([[1, 2, 3, 4], [5, 6]], window=2, device="cpu")

        assert _ids(built) == [[1, 2], [3, 4], [5, 6]]

    def test_every_window_is_batch_shaped(self) -> None:
        """Shaped (1, window), which is what the scorer and trainer both take."""
        built = build_windows([[1, 2, 3, 4]], window=4, device="cpu")

        assert [tuple(row.shape) for row in built] == [(1, 4)]
        assert built[0].dtype is torch.long

    def test_a_document_tail_is_dropped_not_padded(self) -> None:
        """A padded window is mostly a prediction of the padding token.

        Its loss would dilute every mean the corpus feeds by an amount that
        depends on how the documents happened to divide, which is a number
        nobody chose.
        """
        built = build_windows([[1, 2, 3, 4, 5]], window=2, device="cpu")

        assert _ids(built) == [[1, 2], [3, 4]]

    def test_a_window_never_spans_two_documents(self) -> None:
        """Otherwise a window is half one subject and half another.

        Both documents here are three tokens against a window of two, so a
        chunker that ran across the join would emit a window of [3, 4] -- the
        tail of the first and the head of the second.
        """
        built = build_windows([[1, 2, 3], [4, 5, 6]], window=2, device="cpu")

        assert _ids(built) == [[1, 2], [4, 5]]

    def test_a_non_positive_window_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            build_windows([[1, 2, 3]], window=0, device="cpu")

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE
        assert "a window of 0 tokens" in excinfo.value.message

    def test_a_corpus_too_short_to_fill_one_window_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            build_windows([[1, 2], [3]], window=5, device="cpu")

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE
        assert "the longest is 2" in excinfo.value.message

    def test_an_empty_corpus_reports_a_longest_of_zero(self) -> None:
        """No documents at all still names a length, rather than raising over max()."""
        with pytest.raises(AppError) as excinfo:
            build_windows([], window=5, device="cpu")

        assert "the longest is 0" in excinfo.value.message


class TestSplitByStride:
    def test_it_holds_out_every_nth_window(self) -> None:
        windows = build_windows([[0, 1, 2, 3, 4, 5, 6, 7]], window=1, device="cpu")

        train, held_out = split_by_stride(windows, held_out_stride=4)

        assert _ids(held_out) == [[0], [4]]
        assert _ids(train) == [[1], [2], [3], [5], [6], [7]]

    def test_the_two_arms_partition_the_corpus(self) -> None:
        """Every window lands in exactly one arm -- none dropped, none shared.

        A window in both would be trained on and then scored as held out,
        which reports memorisation as generalisation.
        """
        windows = build_windows([list(range(12))], window=1, device="cpu")

        train, held_out = split_by_stride(windows, held_out_stride=3)

        assert len(train) + len(held_out) == len(windows)
        assert sorted(int(row[0][0]) for row in [*train, *held_out]) == list(range(12))

    def test_a_stride_of_one_is_refused(self) -> None:
        """It holds out everything, which is a request for no treatment arm."""
        windows = build_windows([[1, 2, 3, 4]], window=1, device="cpu")

        with pytest.raises(AppError) as excinfo:
            split_by_stride(windows, held_out_stride=1)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE
        assert f"the smallest stride that trains anything is {MIN_HELD_OUT_STRIDE}" in (
            excinfo.value.message
        )

    def test_no_windows_at_all_is_refused_as_nothing_to_score(self) -> None:
        with pytest.raises(AppError) as excinfo:
            split_by_stride([], held_out_stride=4)

        assert "would be scored on nothing" in excinfo.value.message

    def test_a_single_window_is_refused_as_nothing_to_train(self) -> None:
        """The two failures are separate because they read differently.

        No held-out windows means nothing was tested; no training windows
        means there is no cartridge to test.
        """
        windows = build_windows([[1, 2]], window=2, device="cpu")

        with pytest.raises(AppError) as excinfo:
            split_by_stride(windows, held_out_stride=2)

        assert "leave no training windows" in excinfo.value.message
