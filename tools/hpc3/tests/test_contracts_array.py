"""Array-task identity: expansion of what the cluster prints, and nothing else.

Every aggregate fixture here is a string the cluster actually emitted --
probe job 55678543, free partition, ``--array=0-3%2``, 2026-09-01 -- because
the whole module exists to read Slurm's real output, and a parser tested
against invented strings tests a different cluster.
"""

from __future__ import annotations

import itertools

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.array import (
    answers,
    array_task_id,
    base_job_id,
    base_job_ids,
    expand_job_id,
    format_array_argument,
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


class TestFormattingTheArgument:
    """The complete ``--array`` value: indices, and the throttle when declared."""

    def test_no_throttle_is_the_bare_index_expression(self) -> None:
        assert format_array_argument((0, 1, 2, 3), None) == "0-3"

    def test_a_throttle_is_appended_with_slurms_separator(self) -> None:
        """``0-18%4``: nineteen tasks, four at a time."""
        assert format_array_argument(tuple(range(19)), 4) == "0-18%4"

    def test_a_sparse_gap_keeps_its_throttle(self) -> None:
        """A campaign resubmitting three members still runs them throttled."""
        assert format_array_argument((3, 17, 18, 19), 2) == "3,17-19%2"

    def test_an_empty_selection_is_refused_before_the_throttle_is_read(self) -> None:
        with pytest.raises(AppError) as excinfo:
            format_array_argument((), 4)
        assert excinfo.value.code is Hpc3ErrorCode.ARRAY_INDICES_EMPTY


class TestRoundTrip:
    def test_a_throttled_argument_expands_to_the_same_selection(self) -> None:
        """The cluster echoes the throttle back in the pending aggregate and
        the parser discards it; what it submitted and what it sees must be
        the same tasks."""
        indices = (0, 2, 3, 4, 9)
        expanded = expand_job_id(f"777_[{format_array_argument(indices, 4)}]")
        assert expanded == tuple(array_task_id("777", index) for index in indices)

    def test_formatting_then_expanding_returns_the_selection(self) -> None:
        """The submitter renders --array from indices; the parsers expand
        the cluster's echo of it. The two must be inverses or a campaign's
        idea of what is live drifts from what it submitted."""
        indices = (0, 2, 3, 4, 9)
        expression = format_array_indices(indices)
        expanded = expand_job_id(f"777_[{expression}%4]")
        assert expanded == tuple(array_task_id("777", index) for index in indices)


class TestAnswers:
    """Which accounting row answers a requested id.

    The ids are array 59867212, eight tasks on the free partition, read from
    HPC3's accounting on 2026-11-20: ``sacct -j 59867212`` printed rows
    ``59867212_0`` to ``59867212_7`` and never one named ``59867212``.
    """

    def test_a_plain_job_answers_its_own_id(self) -> None:
        assert answers("55519937", "55519937")
        assert not answers("55519937", "55519938")

    def test_a_task_row_answers_its_own_task_and_no_other(self) -> None:
        assert answers("59867212_3", "59867212_3")
        assert not answers("59867212_3", "59867212_4")

    def test_a_pending_aggregate_answers_each_task_it_names(self) -> None:
        """``sacct -j 59867212_0`` is empty while the array pends; the
        aggregate is the only row standing for task 0."""
        assert answers("59867212_[0-7]", "59867212_0")
        assert answers("59867212_[0-7]", "59867212_7")
        assert not answers("59867212_[1-7]", "59867212_0")

    def test_every_row_of_an_array_answers_its_base(self) -> None:
        """No row is ever named by the base once the array exists, so a
        request for the base is answered by its tasks and its aggregate."""
        assert answers("59867212_7", "59867212")
        assert answers("59867212_[0-7]", "59867212")
        assert not answers("59867213_0", "59867212")

    def test_a_task_does_not_answer_for_the_whole_array(self) -> None:
        assert not answers("59867212", "59867212_0")

    def test_an_aggregate_answers_exactly_the_tasks_it_was_submitted_as(self) -> None:
        """Every index subset of a six-task array: the aggregate the cluster
        prints for it answers its base and each of its tasks, and no task
        outside it."""
        for size in range(1, 7):
            for indices in itertools.combinations(range(6), size):
                aggregate = f"777_[{format_array_argument(indices, 2)}]"
                assert answers(aggregate, "777")
                for index in range(6):
                    assert answers(aggregate, array_task_id("777", index)) is (index in indices)
