"""Tests for :mod:`maketools.mutation.cosmic` (MCPs board task e52e5148).

Every JSON text here is what cosmic-ray 8.7.0 printed on sedona on
2026-11-07 for mcp-shared-py's ``path_spelling.py`` (MCPs board task
fec5851c recorded them for its own step's cases): ``dump`` before a run,
and ``mutate-and-test`` for a surviving mutant, an occurrence past the last
(no mutation) and an operator that does not exist (its worker's exception).
"""

from __future__ import annotations

import pytest
from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError

from maketools.mutation.cosmic import (
    MutantResult,
    RunOutcome,
    Site,
    WorkerOutcome,
    command,
    decode_result,
    decode_sites,
    mutated_line,
    session_config,
)

DUMP = (
    '[{"job_id": "f0e1634c923d400bb85d9c5f463f6364", "mutations": [{"module_path": '
    '"src/mcp_shared_py/path_spelling.py", "operator_name": '
    '"core/ReplaceComparisonOperator_Eq_Gt", "occurrence": 0, "start_pos": [45, 13], '
    '"end_pos": [45, 15], "operator_args": {}, "definition_name": "lies_in"}]}, null]\n'
    '[{"job_id": "e57c57dbda3c457c8bda61b2bbda214e", "mutations": [{"module_path": '
    '"src/mcp_shared_py/path_spelling.py", "operator_name": "core/ReplaceOrWithAnd", '
    '"occurrence": 2, "start_pos": [48, 30], "end_pos": [48, 32], "operator_args": {}, '
    '"definition_name": "lies_in"}]}, null]\n'
)

SURVIVED = (
    '{"worker_outcome": "normal", "output": "", "test_outcome": "survived", "diff": '
    '"--- mutation diff ---\\n--- a/src/mcp_shared_py/path_spelling.py\\n+++ '
    "b/src/mcp_shared_py/path_spelling.py\\n@@ -42,7 +42,7 @@\\n         True when it does, "
    "at a path boundary: ``/r/MCPs-guard`` does not\\n         lie in ``/r/MCPs``.\\n     "
    '\\"\\"\\"\\n-    if place == \\"\\":\\n+    if place < \\"\\":\\n         return False\\n'
    "     candidate = spelled(place)\\n     return candidate == root or "
    'candidate.startswith(root + \\"/\\")"}'
)

NO_TEST = '{"worker_outcome": "no-test", "output": null, "test_outcome": null, "diff": null}'

EXCEPTION = (
    '{"worker_outcome": "exception", "output": "Traceback (most recent call last):\\n'
    'KeyError: \'NoSuchOperator\'\\n", "test_outcome": "incompetent", "diff": null}'
)


def test_the_dump_decodes_to_each_mutant_in_order() -> None:
    assert decode_sites(DUMP) == [
        Site(
            "src/mcp_shared_py/path_spelling.py", 45, 13, "core/ReplaceComparisonOperator_Eq_Gt", 0
        ),
        Site("src/mcp_shared_py/path_spelling.py", 48, 30, "core/ReplaceOrWithAnd", 2),
    ]


def test_an_empty_dump_holds_no_mutant() -> None:
    assert decode_sites("") == []


def test_a_work_item_with_operator_arguments_is_refused_naming_its_line() -> None:
    line = DUMP.splitlines()[1].replace('"operator_args": {}', '"operator_args": {"n": 1}')
    with pytest.raises(AppError, match=r"carries operator arguments mutate-and-test") as refused:
        decode_sites(DUMP.splitlines()[0] + "\n" + line)
    assert refused.value.code is MutationErrorCode.COSMIC_RAY_FAILED
    assert '"n": 1' in refused.value.message


@pytest.mark.parametrize("count", [0, 2])
def test_a_work_item_holding_other_than_one_mutation_is_refused(count: int) -> None:
    first = DUMP.splitlines()[0]
    mutation = first.split('"mutations": [')[1].split("]}, null]")[0]
    changed = first.replace(mutation, ", ".join([mutation] * count))
    with pytest.raises(AppError, match=f"holds {count} mutations, not one") as refused:
        decode_sites(changed)
    assert refused.value.code is MutationErrorCode.COSMIC_RAY_FAILED


def test_a_mistyped_field_is_refused() -> None:
    line = DUMP.splitlines()[0].replace('"start_pos": [45, 13]', '"start_pos": [45, "13"]')
    with pytest.raises(JSONTypeError, match="Expected JSON integer, got str"):
        decode_sites(line)


def test_a_surviving_result_decodes_with_its_diff() -> None:
    result = decode_result(SURVIVED)
    assert result.worker is WorkerOutcome.NORMAL
    assert result.tests is RunOutcome.SURVIVED
    assert result.output == ""
    assert '+    if place < "":' in result.diff.splitlines()


def test_a_result_with_no_mutation_decodes_its_nulls_as_empty() -> None:
    assert decode_result(NO_TEST) == MutantResult(WorkerOutcome.NO_TEST, None, "", "")


def test_a_worker_exception_decodes_with_its_traceback() -> None:
    result = decode_result(EXCEPTION)
    assert result.worker is WorkerOutcome.EXCEPTION
    assert result.tests is RunOutcome.INCOMPETENT
    assert result.output.endswith("KeyError: 'NoSuchOperator'\n")
    assert result.diff == ""


def test_an_unknown_outcome_is_refused() -> None:
    with pytest.raises(ValueError, match="'lost' is not a valid WorkerOutcome"):
        decode_result(NO_TEST.replace("no-test", "lost"))
    with pytest.raises(ValueError, match="'gone' is not a valid RunOutcome"):
        decode_result(EXCEPTION.replace("incompetent", "gone"))


def test_the_mutated_line_is_the_first_added_line_without_the_file_header() -> None:
    assert mutated_line(decode_result(SURVIVED).diff) == 'if place < "":'


def test_a_mutation_over_two_lines_is_shown_by_its_first() -> None:
    diff = (
        "--- a/src/a.py\n+++ b/src/a.py\n@@ -3,2 +3,2 @@\n-x = (1 +\n-     2)\n+x = (1 -\n+     2)"
    )
    assert mutated_line(diff) == "x = (1 -"


def test_cosmic_ray_is_the_script_beside_the_interpreter() -> None:
    assert command("/srv/pkg/.venv/bin/python") == "/srv/pkg/.venv/bin/cosmic-ray"


def test_the_session_config_names_every_file_to_mutate_and_the_local_distributor() -> None:
    assert session_config(["src/a.py", 'src/q"uote.py']) == (
        "[cosmic-ray]\n"
        'module-path = ["src/a.py","src/q\\"uote.py"]\n'
        "timeout = 1.0\n"
        "excluded-modules = []\n"
        'test-command = "false"\n'
        "\n"
        "[cosmic-ray.distributor]\n"
        'name = "local"\n'
    )
