"""The subprocess-timeout rule, driven over real parsed source.

Every case writes a real file and runs the real rule over it: the rule's whole
job is to read Python, so a case that handed it a pre-built AST would be
testing the test's idea of the syntax rather than the syntax.

The two clauses are exercised against each other on purpose. A rule that only
refused a missing ``timeout=`` would pass ``run(input=..., timeout=...)``,
which is the call that cost a fleet staging send 40 minutes on 2026-09-24, and
a rule that refused any payload would refuse ``stdin=``, which is the remedy.
Both directions are pinned below.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.orchestrator import run_for_project
from monorepo_guards.subprocess_timeout_rules import (
    MISSING_DEADLINE,
    UNBOUNDED_PAYLOAD,
    UNPROVEN_DEADLINE,
    SubprocessTimeoutRule,
)
from tests._literal_set_support import write_declared_sets


def _module(tmp_path: Path, body: str, *, where: str = "src") -> Path:
    """Write one module into a scanned directory.

    Args:
        tmp_path: The case's temporary directory.
        body: The module's source.
        where: The directory under it, ``src`` by default.

    Returns:
        The file's path.
    """
    path = tmp_path / where / "subject.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _kinds(path: Path) -> list[str]:
    """Every violation kind the rule reports for one file, in order.

    Args:
        path: The file to scan.

    Returns:
        The kinds.
    """
    return [v.kind for v in SubprocessTimeoutRule().run([path])]


# ---------------------------------------------------------------------------
# Clause one: a call with no deadline
# ---------------------------------------------------------------------------


def test_a_run_without_a_timeout_is_refused(tmp_path: Path) -> None:
    path = _module(tmp_path, "import subprocess\nsubprocess.run(['ls'])\n")
    violations = SubprocessTimeoutRule().run([path])
    assert [v.kind for v in violations] == [MISSING_DEADLINE]
    assert violations[0].line_no == 2
    assert violations[0].file == path


def test_every_awaiting_helper_is_refused_without_a_timeout(tmp_path: Path) -> None:
    # All four take timeout, so all four are judged; a rule that only knew
    # run() would leave check_output the obvious way to write an unbounded one.
    path = _module(
        tmp_path,
        "import subprocess\n"
        "subprocess.run(['a'])\n"
        "subprocess.check_output(['b'])\n"
        "subprocess.check_call(['c'])\n"
        "subprocess.call(['d'])\n",
    )
    assert _kinds(path) == [MISSING_DEADLINE] * 4


def test_a_bare_name_bound_by_a_from_import_is_judged(tmp_path: Path) -> None:
    path = _module(tmp_path, "from subprocess import run\nrun(['ls'])\n")
    assert _kinds(path) == [MISSING_DEADLINE]


def test_an_aliased_from_import_is_judged_under_its_alias(tmp_path: Path) -> None:
    path = _module(tmp_path, "from subprocess import run as spawn\nspawn(['ls'])\n")
    assert _kinds(path) == [MISSING_DEADLINE]


def test_a_bounded_call_is_silent(tmp_path: Path) -> None:
    path = _module(tmp_path, "import subprocess\nsubprocess.run(['ls'], timeout=5)\n")
    assert _kinds(path) == []


# ---------------------------------------------------------------------------
# Clause two: a deadline that does not govern the call
# ---------------------------------------------------------------------------


def test_a_timeout_paired_with_input_is_refused(tmp_path: Path) -> None:
    """The regression. This call passes clause one and is still unbounded."""
    path = _module(
        tmp_path,
        "import subprocess\nsubprocess.run(['ls'], input=payload, timeout=5)\n",
    )
    violations = SubprocessTimeoutRule().run([path])
    assert [v.kind for v in violations] == [UNBOUNDED_PAYLOAD]
    assert violations[0].line_no == 2


def test_stdin_is_not_input_and_is_not_refused(tmp_path: Path) -> None:
    """The remedy must pass, or the rule refuses the fix and keeps the defect.

    This is the shape ``tools/fleet/src/fleet/core/_command.py`` uses: the
    child reads a file itself, so the calling thread writes nothing and the
    deadline governs the whole call.
    """
    path = _module(
        tmp_path,
        "import subprocess\nsubprocess.run(['ls'], stdin=handle, timeout=5)\n",
    )
    assert _kinds(path) == []


def test_input_without_a_timeout_is_reported_once_as_missing(tmp_path: Path) -> None:
    # Not two findings for one call: with no timeout at all the first clause
    # is the honest description, and the fix is a deadline AND a file.
    path = _module(tmp_path, "import subprocess\nsubprocess.run(['ls'], input=b'x')\n")
    assert _kinds(path) == [MISSING_DEADLINE]


def test_communicate_with_input_and_timeout_is_refused(tmp_path: Path) -> None:
    """``run`` is built on ``communicate``, so the hole is the same hole."""
    path = _module(
        tmp_path,
        "proc.communicate(input=data, timeout=30)\n",
    )
    assert _kinds(path) == [UNBOUNDED_PAYLOAD]


def test_communicate_without_a_payload_is_left_to_the_popen_shape(
    tmp_path: Path,
) -> None:
    # Deliberately not judged here: a communicate carrying no input is the
    # ordinary Popen bound, which this rule does not reach into.
    path = _module(tmp_path, "proc.communicate(timeout=30)\n")
    assert _kinds(path) == []


def test_communicate_with_input_and_no_timeout_is_not_this_rules_finding(
    tmp_path: Path,
) -> None:
    path = _module(tmp_path, "proc.communicate(input=data)\n")
    assert _kinds(path) == []


# ---------------------------------------------------------------------------
# A deadline nobody can prove
# ---------------------------------------------------------------------------


def test_splatted_keywords_are_refused_as_an_unproven_deadline(
    tmp_path: Path,
) -> None:
    """No reading of this call says whether it is bounded, so it is not."""
    path = _module(tmp_path, "import subprocess\nsubprocess.run(['ls'], **options)\n")
    assert _kinds(path) == [UNPROVEN_DEADLINE]


def test_a_splat_beside_a_real_timeout_is_still_unproven(tmp_path: Path) -> None:
    # The timeout is visible but the splat may also carry input=, so the call
    # is still not proven bounded and the report says which uncertainty it is.
    path = _module(tmp_path, "import subprocess\nsubprocess.run(['ls'], timeout=5, **rest)\n")
    assert _kinds(path) == [UNPROVEN_DEADLINE]


# ---------------------------------------------------------------------------
# What the rule must NOT claim
# ---------------------------------------------------------------------------


def test_an_unrelated_function_called_run_is_not_judged(tmp_path: Path) -> None:
    """Resolution is by import, not by spelling."""
    path = _module(tmp_path, "def run(argv):\n    return argv\n\nrun(['ls'])\n")
    assert _kinds(path) == []


def test_another_objects_run_attribute_is_not_judged(tmp_path: Path) -> None:
    path = _module(tmp_path, "import subprocess\nrunner.run(['ls'])\n")
    assert _kinds(path) == []


def test_a_dotted_receiver_other_than_subprocess_is_not_judged(tmp_path: Path) -> None:
    path = _module(tmp_path, "import subprocess\npkg.mod.run(['ls'])\n")
    assert _kinds(path) == []


def test_a_subprocess_attribute_that_is_not_an_awaiting_helper_is_not_judged(
    tmp_path: Path,
) -> None:
    path = _module(tmp_path, "import subprocess\nsubprocess.list2cmdline(['ls'])\n")
    assert _kinds(path) == []


def test_a_from_import_of_something_else_binds_nothing(tmp_path: Path) -> None:
    path = _module(tmp_path, "from subprocess import PIPE\nPIPE(['ls'])\n")
    assert _kinds(path) == []


def test_a_from_import_of_another_module_binds_nothing(tmp_path: Path) -> None:
    path = _module(tmp_path, "from shlex import quote as run\nrun(['ls'])\n")
    assert _kinds(path) == []


def test_a_call_whose_callee_is_an_expression_is_not_judged(tmp_path: Path) -> None:
    # Neither an ast.Name nor an ast.Attribute: the last arm of the resolver.
    path = _module(tmp_path, "handlers[0](['ls'])\n")
    assert _kinds(path) == []


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_a_scripts_file_is_judged(tmp_path: Path) -> None:
    path = _module(tmp_path, "import subprocess\nsubprocess.run(['ls'])\n", where="scripts")
    assert _kinds(path) == [MISSING_DEADLINE]


def test_a_tests_file_is_not_judged(tmp_path: Path) -> None:
    """Bounded by the CI job's own clock, which workflow_timeout_rules enforces."""
    path = tmp_path / "src" / "tests" / "test_thing.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("import subprocess\nsubprocess.run(['ls'])\n", encoding="utf-8")
    assert _kinds(path) == []


def test_a_file_outside_src_and_scripts_is_not_judged(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "example.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("import subprocess\nsubprocess.run(['ls'])\n", encoding="utf-8")
    assert _kinds(path) == []


def test_the_rule_reports_its_name(tmp_path: Path) -> None:
    assert SubprocessTimeoutRule().name == "subprocess-timeout"


def test_several_files_are_judged_in_one_run(tmp_path: Path) -> None:
    first = tmp_path / "src" / "one.py"
    second = tmp_path / "src" / "two.py"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("import subprocess\nsubprocess.run(['a'])\n", encoding="utf-8")
    second.write_text(
        "import subprocess\nsubprocess.run(['b'], input=x, timeout=1)\n",
        encoding="utf-8",
    )
    violations = SubprocessTimeoutRule().run([first, second])
    assert [v.kind for v in violations] == [MISSING_DEADLINE, UNBOUNDED_PAYLOAD]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
#
# THIS CASE WAS HELD BACK ON PURPOSE AND LANDS WITH THE REGISTRATION. The rule
# reaches 51 packages through their scripts/guard.py bootstrap rather than the
# 7 that depend on monorepo_guards, and 12 of them held unbounded call sites;
# a registered rule with live violations is not a landed rule, it is a broken
# build for every package it reaches. So the rule was written first,
# registered nowhere, and this case sat here as a comment until the count
# reached zero (2,794 files under src and scripts, 13 violations down to 0,
# board task 0d891468). Asserting registration before then would have been a
# test asserting the future rather than the code.


def _guarded_monorepo(tmp_path: Path) -> tuple[Path, Path]:
    """Build a monorepo root with a guard config and one project inside it.

    Args:
        tmp_path: The case's temporary directory.

    Returns:
        The monorepo root and the project root, the two arguments
        :func:`run_for_project` takes.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "monorepo-guards.toml").write_text(
        '[guards]\ndirectories = ["src"]\nexclude_parts = [".venv"]\n'
        "forbid_pyi = true\nallow_print_in_tests = false\ndataclass_ban_segments = []\n",
        encoding="utf-8",
    )
    write_declared_sets(root)
    project = root / "services" / "subject"
    (project / "src").mkdir(parents=True)
    return root, project


def test_the_orchestrator_actually_runs_this_rule(tmp_path: Path) -> None:
    """REGISTERED IS NOT INVOKED, and only this case tells them apart.

    Every other case in this file constructs the rule and calls it, so all of
    them stay green whether or not the orchestrator has ever heard of it.
    This one drives the real ``run_for_project`` over a real guard config and
    asserts the defect fails the run.
    """
    root, project = _guarded_monorepo(tmp_path)
    (project / "src" / "subject.py").write_text(
        "import subprocess\nsubprocess.run(['ls'])\n", encoding="utf-8"
    )

    assert run_for_project(root, project) == 2


def test_the_orchestrator_passes_the_bounded_form(tmp_path: Path) -> None:
    """The other half, and it is not decoration: a rule that refused
    everything would also make the case above pass."""
    root, project = _guarded_monorepo(tmp_path)
    (project / "src" / "subject.py").write_text(
        "import subprocess\nsubprocess.run(['ls'], timeout=5)\n", encoding="utf-8"
    )

    assert run_for_project(root, project) == 0
