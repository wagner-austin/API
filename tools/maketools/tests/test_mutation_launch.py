"""Tests for :mod:`maketools.mutation.launch` (MCPs board task e52e5148).

The launcher hands the step a fresh scratch directory, removes it however
the step ends, and records the step's time as the check's mutation span
when ``check-budget`` named a span file, saying so when none did.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError

from maketools.bounded_spans import SPANS_ENVIRONMENT_KEY, read_spans
from maketools.commands import CommandResult
from maketools.mutation.launch import mutate_after_suite, record_time
from tests.conftest import World, failed, ok

NOT_APPLICABLE = (
    "mutation: not applicable: 0 lines added to src files against origin/main (merge base "
    "abc123), so 0 mutants were made"
)


def _no_change(argv: tuple[str, ...]) -> CommandResult:
    """Answer git as a commit that adds no source line.

    Args:
        argv: The command.

    Returns:
        The merge base for ``merge-base``, and nothing for the rest.
    """
    return ok("abc123\n" if argv[1] == "merge-base" else "")


def test_the_time_is_recorded_as_the_checks_mutation_span(world: World, tmp_path: Path) -> None:
    spans = tmp_path / "spans"
    world.now_value = 1794087804.5
    record_time({SPANS_ENVIRONMENT_KEY: str(spans)}, 41)
    assert spans.read_text(encoding="utf-8") == "41 1794087804.500 mutation testing\n"
    assert world.lines == ["mutation: 41s, recorded as the check's mutation testing"]


def test_a_run_no_check_budget_started_says_its_time_is_held_to_no_bound(world: World) -> None:
    record_time({SPANS_ENVIRONMENT_KEY: ""}, 3)
    record_time({}, 4)
    assert world.lines == [
        f"mutation: 3s; no check-budget named a span file in {SPANS_ENVIRONMENT_KEY}, so this "
        "run's time is held to no bound",
        f"mutation: 4s; no check-budget named a span file in {SPANS_ENVIRONMENT_KEY}, so this "
        "run's time is held to no bound",
    ]


def test_the_step_runs_in_a_scratch_directory_removed_after_and_its_span_recorded(
    world: World, tmp_path: Path
) -> None:
    world.answer = _no_change
    spans = tmp_path / "spans"
    scratch = tmp_path / "runs" / ".mutation-deadbeef"
    scratch.parent.mkdir()
    environment = {"PATH": "/bin", SPANS_ENVIRONMENT_KEY: str(spans)}
    assert mutate_after_suite(tmp_path, "/pkg/.venv/bin/python", environment, scratch) == 0
    assert scratch.is_dir()
    assert world.removed_trees == [scratch]
    assert world.lines == [
        NOT_APPLICABLE,
        "mutation: 0s, recorded as the check's mutation testing",
    ]
    assert [span.seconds for span in read_spans(spans)] == [0]
    assert {call["cwd"] for call in world.answering_calls} == {tmp_path}
    assert {call["env"][SPANS_ENVIRONMENT_KEY] for call in world.answering_calls} == {str(spans)}


def test_a_refusal_still_removes_the_scratch_directory_and_records_no_span(
    world: World, tmp_path: Path
) -> None:
    world.answer = lambda argv: failed(128, "")
    spans = tmp_path / "spans"
    scratch = tmp_path / ".mutation-deadbeef"
    with pytest.raises(AppError) as refused:
        mutate_after_suite(tmp_path, "/py", {SPANS_ENVIRONMENT_KEY: str(spans)}, scratch)
    assert refused.value.code is MutationErrorCode.BASE_UNAVAILABLE
    assert world.removed_trees == [scratch]
    assert not spans.exists()
