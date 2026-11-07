"""API's boundary of the lifted mutation step (MCPs board task e52e5148).

What is API's own around MCPs' core: the JSON codec over
``platform_core.json_utils`` (its refusals naming the field), the hooks bound
to maketools' own, the span sink ``check-budget`` reads, and the refusal
mapping onto ``platform_core``'s ``MutationErrorCode``. The core's commands
go through the ``world`` fixture's ``run_answering``; the files, the scratch
directory and the span file are real.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core import error_codes_tooling
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError

from maketools.bounded_spans import SPANS_ENVIRONMENT_KEY, read_spans
from maketools.commands import CommandResult
from maketools.mutation.context import MutationErrorCode, MutationStepError, Ran
from maketools.mutation.launch import (
    API_CODEC,
    CHILD_WALL_SECONDS,
    answer,
    dump_spaced,
    integer,
    made,
    mapping,
    mutate_after_suite,
    mutation_hooks,
    optional_text,
    read_text,
    sequence,
    span_sink,
    text,
    write_text,
)
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


def test_the_codec_reads_with_platform_core_and_names_the_field_it_refuses() -> None:
    assert API_CODEC.loads('{"a": [1, "b", null]}') == {"a": [1, "b", None]}
    assert mapping({"k": 1}, "f") == {"k": 1}
    assert sequence([1, "x"], "f") == [1, "x"]
    assert text("x", "f") == "x"
    assert integer(3, "f") == 3
    assert optional_text(None, "f") is None
    assert optional_text("y", "f") == "y"
    with pytest.raises(JSONTypeError, match=r"^files: Expected JSON object, got list$") as one:
        mapping([], "files")
    assert isinstance(one.value.__cause__, JSONTypeError)
    with pytest.raises(JSONTypeError, match=r"^start_pos: Expected JSON array, got dict$"):
        sequence({}, "start_pos")
    with pytest.raises(JSONTypeError, match=r"^operator_name: Expected JSON string, got int$"):
        text(1, "operator_name")
    with pytest.raises(JSONTypeError, match=r"^occurrence: Expected JSON integer, got str$"):
        integer("1", "occurrence")
    with pytest.raises(JSONTypeError, match=r"^diff: Expected JSON string, got int$"):
        optional_text(2, "diff")


def test_the_codec_writes_json_with_the_standard_separator_as_mcps_does() -> None:
    assert dump_spaced(["src/a.py", 'q"']) == '["src/a.py", "q\\""]'
    assert dump_spaced("é\t") == '"\\u00e9\\t"'
    assert API_CODEC.dumps is dump_spaced
    assert (API_CODEC.mapping, API_CODEC.sequence) == (mapping, sequence)
    assert (API_CODEC.text, API_CODEC.optional_text, API_CODEC.integer) == (
        text,
        optional_text,
        integer,
    )


def test_a_command_runs_through_run_answering_in_the_package_under_the_steps_wall(
    world: World, tmp_path: Path
) -> None:
    world.answer = lambda argv: CommandResult(returncode=3, stdout="out", stderr="err")
    assert answer(["git", "status"], {"A": "1"}, tmp_path) == Ran(3, "out", "err")
    (call,) = world.answering_calls
    assert call["argv"] == ("git", "status")
    assert call["cwd"] == tmp_path
    assert call["env"] == {"A": "1"}
    assert call["timeout_seconds"] == CHILD_WALL_SECONDS == 900


def test_the_hooks_reach_maketools_own_and_real_files(world: World, tmp_path: Path) -> None:
    world.answer = lambda argv: ok("printed")
    scratch = tmp_path / "runs" / ".mutation-deadbeef"
    scratch.parent.mkdir()
    hooks = mutation_hooks(tmp_path, scratch)
    assert hooks.run(["true"], {}) == Ran(0, "printed", "")
    assert world.answering_calls[0]["cwd"] == tmp_path
    assert hooks.now() == world.now_value
    assert hooks.make_run_directory() == scratch
    assert scratch.is_dir()
    hooks.write_text(scratch / "a.json", "é")
    assert hooks.read_text(scratch / "a.json") == "é"
    hooks.write("one line")
    assert world.lines == ["one line"]
    hooks.remove_tree(scratch)
    assert world.removed_trees == [scratch]
    assert (hooks.write_text, hooks.read_text) == (write_text, read_text)


def test_making_a_scratch_directory_that_exists_is_refused(tmp_path: Path) -> None:
    assert made(tmp_path / "new") == tmp_path / "new"
    with pytest.raises(FileExistsError):
        made(tmp_path / "new")


def test_a_runs_time_goes_to_the_span_file_check_budget_named(world: World, tmp_path: Path) -> None:
    spans = tmp_path / "spans"
    world.now_value = 1794087804.5
    span_sink({SPANS_ENVIRONMENT_KEY: str(spans)})(41)
    assert spans.read_text(encoding="utf-8") == "41 1794087804.500 mutation testing\n"
    assert world.lines == ["mutation: 41s, recorded as the check's mutation testing"]


def test_a_run_no_check_budget_started_says_its_time_is_held_to_no_bound(world: World) -> None:
    span_sink({SPANS_ENVIRONMENT_KEY: ""})(3)
    span_sink({})(4)
    assert world.lines == [
        f"mutation: 3s; no check-budget named a span file in {SPANS_ENVIRONMENT_KEY}, so this "
        "run's time is held to no bound",
        f"mutation: 4s; no check-budget named a span file in {SPANS_ENVIRONMENT_KEY}, so this "
        "run's time is held to no bound",
    ]


def test_a_commit_adding_no_source_line_runs_in_scratch_and_records_no_span(
    world: World, tmp_path: Path
) -> None:
    world.answer = _no_change
    spans = tmp_path / "spans"
    scratch = tmp_path / ".mutation-deadbeef"
    environment = {"PATH": "/bin", SPANS_ENVIRONMENT_KEY: str(spans)}
    assert mutate_after_suite(tmp_path, "/pkg/.venv/bin/python", environment, scratch) == 0
    assert scratch.is_dir()
    assert world.removed_trees == [scratch]
    assert world.lines == [NOT_APPLICABLE]
    assert read_spans(spans) == ()
    assert {call["env"][SPANS_ENVIRONMENT_KEY] for call in world.answering_calls} == {str(spans)}


@pytest.mark.parametrize("code", list(MutationErrorCode))
def test_every_core_refusal_is_platform_cores_code_of_the_same_value(
    code: MutationErrorCode,
) -> None:
    assert error_codes_tooling.MutationErrorCode(code.value).value == code.value


def test_a_refusal_becomes_an_app_error_and_still_removes_the_scratch_directory(
    world: World, tmp_path: Path
) -> None:
    world.answer = lambda argv: failed(128, "")
    scratch = tmp_path / ".mutation-deadbeef"
    with pytest.raises(AppError) as refused:
        mutate_after_suite(tmp_path, "/py", {}, scratch)
    assert refused.value.code is error_codes_tooling.MutationErrorCode.BASE_UNAVAILABLE
    assert refused.value.message.startswith("refs/remotes/origin/main names no commit here")
    assert isinstance(refused.value.__cause__, MutationStepError)
    assert world.removed_trees == [scratch]
