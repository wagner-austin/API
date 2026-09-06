"""Array-task identity: expansion of what the cluster prints, and nothing else.

Every aggregate fixture here is a string the cluster actually emitted --
probe job 55678543, free partition, ``--array=0-3%2``, 2026-09-01 -- because
the whole module exists to read Slurm's real output, and a parser tested
against invented strings tests a different cluster.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.array import (
    array_task_id,
    base_job_id,
    base_job_ids,
    expand_job_id,
    format_array_indices,
)


class TestTaskIds:
    def test_a_task_id_is_base_underscore_index(self) -> None:
        assert array_task_id("55678543", 2) == "55678543_2"


class TestBaseIds:
    """The id you must ASK about, and the inverse of :func:`array_task_id`.

    Expanding what comes back is only half the job: `sacct -j 55765275_0`
    returns NOTHING while task 0 sits inside a pending aggregate, so a query
    built from per-task ids never sees the row it needs to expand.
    """

    def test_a_plain_id_is_its_own_base(self) -> None:
        assert base_job_id("55674054") == "55674054"

    def test_a_task_id_reduces_to_the_submitted_id(self) -> None:
        assert base_job_id("55765275_3") == "55765275"

    def test_an_aggregate_reduces_to_the_same_base(self) -> None:
        assert base_job_id("55765275_[0-5]") == "55765275"

    def test_a_throttled_aggregate_reduces_to_the_same_base(self) -> None:
        assert base_job_id("55678543_[2-3%2]") == "55678543"

    def test_every_task_of_one_array_collapses_to_a_single_asked_id(self) -> None:
        """A 60-task array is one accounting query, not sixty ids."""
        assert base_job_ids([f"55786856_{index}" for index in range(60)]) == ["55786856"]

    def test_distinct_arrays_are_kept_in_first_seen_order(self) -> None:
        recorded = ["55765284_0", "55765275_0", "55765284_1", "101"]
        assert base_job_ids(recorded) == ["55765284", "55765275", "101"]

    def test_no_ids_produce_no_query(self) -> None:
        assert base_job_ids([]) == []


class TestExpandingClusterIds:
    def test_a_plain_job_id_is_itself(self) -> None:
        assert expand_job_id("55674054") == ("55674054",)

    def test_a_running_task_id_is_itself(self) -> None:
        """Measured: RUNNING tasks row individually as base_index."""
        assert expand_job_id("55678543_0") == ("55678543_0",)

    def test_a_pending_aggregate_with_throttle_expands(self) -> None:
        """The exact squeue AND sacct -X shape for still-pending tasks.

        The throttle is discarded: %2 says how fast tasks may start, not
        which tasks exist."""
        assert expand_job_id("55678543_[2-3%2]") == ("55678543_2", "55678543_3")

    def test_a_mixed_list_and_range_aggregate_expands_in_order(self) -> None:
        assert expand_job_id("9_[0,5-7,11]") == ("9_0", "9_5", "9_6", "9_7", "9_11")

    def test_a_single_index_aggregate_expands(self) -> None:
        assert expand_job_id("9_[4]") == ("9_4",)

    def test_an_unclosed_bracket_is_refused(self) -> None:
        with pytest.raises(AppError) as caught:
            expand_job_id("9_[2-3")
        assert caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE
        assert "never closes" in caught.value.message

    def test_an_empty_term_is_refused(self) -> None:
        with pytest.raises(AppError) as caught:
            expand_job_id("9_[2,,3]")
        assert caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE
        assert "empty index term" in caught.value.message

    def test_a_non_numeric_term_is_refused(self) -> None:
        with pytest.raises(AppError) as caught:
            expand_job_id("9_[2,x]")
        assert caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE
        assert "non-numeric" in caught.value.message

    def test_a_non_numeric_range_bound_is_refused(self) -> None:
        with pytest.raises(AppError) as caught:
            expand_job_id("9_[2-x]")
        assert caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE
        assert "non-numeric" in caught.value.message

    def test_a_reversed_range_is_refused(self) -> None:
        """A silent miss here becomes a double submission racing on one
        artifact, which is why every malformation raises instead of
        yielding a partial set."""
        with pytest.raises(AppError) as caught:
            expand_job_id("9_[7-2]")
        assert caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE
        assert "reversed range" in caught.value.message


class TestFormattingIndices:
    def test_a_full_run_compresses_to_one_range(self) -> None:
        assert format_array_indices((0, 1, 2, 3)) == "0-3"

    def test_a_sparse_gap_mixes_singles_and_ranges(self) -> None:
        """The campaign's shape: exactly the missing document positions."""
        assert format_array_indices((3, 17, 18, 19)) == "3,17-19"

    def test_a_single_index_stands_alone(self) -> None:
        assert format_array_indices((5,)) == "5"

    def test_separated_singles_stay_separate(self) -> None:
        assert format_array_indices((1, 3, 5)) == "1,3,5"

    def test_an_empty_selection_is_refused(self) -> None:
        with pytest.raises(AppError) as caught:
            format_array_indices(())
        assert caught.value.code is Hpc3ErrorCode.ARRAY_INDICES_EMPTY

    def test_a_non_increasing_selection_is_refused(self) -> None:
        """Shuffled or duplicated indices mean the member bookkeeping is
        already wrong, and a submission built from it runs the wrong set."""
        with pytest.raises(AppError) as caught:
            format_array_indices((3, 3))
        assert caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE
        with pytest.raises(AppError) as reversed_caught:
            format_array_indices((5, 2))
        assert reversed_caught.value.code is Hpc3ErrorCode.ARRAY_ID_UNPARSABLE


class TestRoundTrip:
    def test_formatting_then_expanding_returns_the_selection(self) -> None:
        """The submitter renders --array from indices; the parsers expand
        the cluster's echo of it. The two must be inverses or a campaign's
        idea of what is live drifts from what it submitted."""
        indices = (0, 2, 3, 4, 9)
        expression = format_array_indices(indices)
        expanded = expand_job_id(f"777_[{expression}%4]")
        assert expanded == tuple(array_task_id("777", index) for index in indices)
