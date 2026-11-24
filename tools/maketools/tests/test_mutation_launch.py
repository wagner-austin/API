"""API's boundary of the lifted mutation step (MCPs board tasks e52e5148 and 3b386c0d).

What is API's own around MCPs' core: the JSON codec over
``platform_core.json_utils`` (its refusals naming the field), the hooks bound
to maketools' own, the span sink ``check-budget`` reads, the bound the step
holds itself to (``check-budget``'s own for the span), and the refusal
mapping onto ``platform_core``'s ``MutationErrorCode``. The core's commands
go through the ``world`` fixture's ``run_answering``; the files, the scratch
directory and the span file are real. Each package's ``pyproject.toml``
declares ``src`` and ``scripts`` for coverage, as every API package does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core import error_codes_tooling
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str
from platform_core.toml_utils import loads_toml

from maketools.bounded_spans import MUTATION_SPAN, SPANS_ENVIRONMENT_KEY, BoundedSpan, read_spans
from maketools.commands import CommandResult
from maketools.mutation.context import MutationErrorCode, Ran
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
from tests.conftest import NOT_APPLICABLE, World, failed, ok, write_package_pyproject

#: A commit adding one module constant, a line only import runs.
CONSTANT_DIFF = (
    "diff --git a/src/m.py b/src/m.py\nnew file mode 100644\n--- /dev/null\n"
    "+++ b/src/m.py\n@@ -0,0 +1 @@\n+X = 1\n"
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
    assert str(one.value.__cause__) == "Expected JSON object, got list"
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
    assert API_CODEC.loads_toml is loads_toml
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
    write_package_pyproject(tmp_path)
    spans = tmp_path / "spans"
    scratch = tmp_path / ".mutation-deadbeef"
    environment = {"PATH": "/bin", SPANS_ENVIRONMENT_KEY: str(spans)}
    assert mutate_after_suite(tmp_path, "/pkg/.venv/bin/python", environment, scratch) == 0
    assert scratch.is_dir()
    assert world.removed_trees == [scratch]
    assert world.lines == [NOT_APPLICABLE]
    assert read_spans(spans) == ()
    assert {call["env"][SPANS_ENVIRONMENT_KEY] for call in world.answering_calls} == {str(spans)}


def _constant_dump() -> str:
    """cosmic-ray's dump of the one mutant of ``X = 1``: its ``1``.

    Returns:
        The line ``cosmic-ray dump`` prints for it.
    """
    mutation: dict[str, JSONValue] = {
        "module_path": "src/m.py",
        "operator_name": "core/NumberReplacer",
        "occurrence": 0,
        "start_pos": [1, 4],
        "end_pos": [1, 5],
        "operator_args": {},
        "definition_name": "x",
    }
    work_item: dict[str, JSONValue] = {"job_id": "j", "mutations": [mutation]}
    dumped: list[JSONValue] = [work_item, None]
    return dump_json_str(dumped, compact=False)


def _import_only_coverage() -> str:
    """The coverage report of a line only import ran: no test's context names it.

    Returns:
        The JSON ``coverage json --show-contexts`` writes.
    """
    contexts: dict[str, JSONValue] = {"1": [""]}
    measured: dict[str, JSONValue] = {"executed_lines": [1], "contexts": contexts}
    files: dict[str, JSONValue] = {"src/m.py": measured}
    report: dict[str, JSONValue] = {"meta": {"show_contexts": True}, "files": files}
    return dump_json_str(report, compact=False)


def test_the_step_holds_itself_to_check_budgets_bound_and_refuses_when_it_comes(
    world: World, tmp_path: Path
) -> None:
    # The clock reads 1,000,000.0 when the step starts and moves 900 s while
    # cosmic-ray lists the mutants, so the step's own check before its first
    # trial finds the bound, 900 s, come: no baseline and no trial runs, and
    # the span is round(1,000,900.0 - 1,000,000.0) = 900 s.
    write_package_pyproject(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("X = 1\n", encoding="utf-8")
    spans = tmp_path / "spans"
    scratch = tmp_path / ".mutation-deadbeef"
    python = "/pkg/.venv/bin/python"

    def answer_commands(argv: tuple[str, ...]) -> CommandResult:
        if argv[:2] == ("git", "merge-base"):
            return ok("abc123\n")
        if argv[:2] == ("git", "diff"):
            return ok(CONSTANT_DIFF)
        if argv[1:2] == ("dump",):
            world.now_value += 900.0
            return ok(_constant_dump())
        if argv[1:3] == ("-m", "coverage"):
            Path(argv[-1]).write_text(_import_only_coverage(), encoding="utf-8")
        return ok("")

    world.answer = answer_commands
    environment = {"PATH": "/bin", SPANS_ENVIRONMENT_KEY: str(spans)}
    with pytest.raises(AppError) as refused:
        mutate_after_suite(tmp_path, python, environment, scratch)
    assert refused.value.code is error_codes_tooling.MutationErrorCode.OVER_BOUND
    assert refused.value.message == (
        "0 of the 1 mutant(s) the tests run in 1 source file(s) had been run when the step's "
        "900 s bound came, so 1 were not run, beside 0 already uncaught, each named above; "
        "HEAD's bytes are back in each file it mutates and its span is recorded. A change too "
        "large to mutate inside the bound is one to split"
    )
    assert 'MUTANT NOT RUN, THE BOUND CAME src/m.py:1:5 NumberReplacer -> "X = 1"' in world.lines
    ran = [call["argv"] for call in world.answering_calls]
    assert [argv for argv in ran if argv[0] == "sh" or "mutate-and-test" in argv] == []
    assert ran[2][-3:] == ("--", "src", "scripts")
    assert read_spans(spans) == (BoundedSpan(MUTATION_SPAN, 900, 900, 1_000_900.0),)
    assert world.removed_trees == [scratch]


@pytest.mark.parametrize("code", list(MutationErrorCode))
def test_every_core_refusal_is_platform_cores_code_of_the_same_value(
    code: MutationErrorCode,
) -> None:
    assert error_codes_tooling.MutationErrorCode(code.value).value == code.value


def test_a_refusal_becomes_an_app_error_and_still_removes_the_scratch_directory(
    world: World, tmp_path: Path
) -> None:
    world.answer = lambda argv: failed(128, "")
    write_package_pyproject(tmp_path)
    scratch = tmp_path / ".mutation-deadbeef"
    with pytest.raises(AppError) as refused:
        mutate_after_suite(tmp_path, "/py", {}, scratch)
    assert refused.value.code is error_codes_tooling.MutationErrorCode.BASE_UNAVAILABLE
    assert refused.value.message.startswith("refs/remotes/origin/main names no commit here")
    assert str(refused.value.__cause__) == f"MUTATION_BASE_UNAVAILABLE: {refused.value.message}"
    assert world.removed_trees == [scratch]
