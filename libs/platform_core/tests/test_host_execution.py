"""A host-execution run fails unless every host case bound to its capability ran.

Board task 465689f5. :mod:`platform_core.host_execution` is what makes a
passed execution project mean the host cases executed: it keeps only the
cases bound to the run's capability and turns a skip, or a run of none, into
a failure. A plugin that quietly passed a run of zero would make every gate
that reads the project vouch for nothing, so each case here runs a real
pytest session, in this process so the module's lines are measured, with a
plugin module that forwards the three hooks exactly as a package's does.
"""

from __future__ import annotations

from typing import Final

import pytest

from platform_core.host_execution import HostExecutionPlan

#: A package's plugin module, binding two capabilities and one execution-only
#: case to each, with this run holding ``alpha``.
PLUGIN: Final[str] = """
from platform_core import host_execution

PLAN = host_execution.HostExecutionPlan(
    markers={"host_alpha": "alpha", "host_beta": "beta"},
    execution_only={"host_alpha_only": "alpha", "host_beta_only": "beta"},
    needs={"host_alpha_only": "needs the alpha route", "host_beta_only": "needs the beta route"},
    projects={"alpha": "tools/alpha-execution", "beta": "tools/beta-execution"},
    here="alpha",
)


def pytest_addoption(parser):
    host_execution.add_option(parser)


def pytest_configure(config):
    host_execution.configure(config, PLAN)


def pytest_collection_modifyitems(config, items):
    host_execution.select(config, items, PLAN)
"""


def _session(pytester: pytest.Pytester, body: str, *arguments: str) -> pytest.RunResult:
    """Run one pytest session over ``body`` with the plugin loaded.

    Args:
        pytester: The fixture that owns the session's directory.
        body: The test module's source.
        arguments: Extra command-line arguments.

    Returns:
        The finished run.
    """
    pytester.makepyfile(host_plugin=PLUGIN, test_cases=body)
    pytester.syspathinsert()
    return pytester.runpytest_inprocess("-p", "host_plugin", "-p", "no:cacheprovider", *arguments)


def _module(alpha: str, beta: str) -> str:
    """A module with one case bound to each capability and one unbound.

    Args:
        alpha: The body of the case bound to this run's capability.
        beta: The body of the case bound to the other one.

    Returns:
        The module's source.
    """
    return (
        "import pytest\n"
        f"@pytest.mark.host_alpha\ndef test_alpha():\n    {alpha}\n"
        f"@pytest.mark.host_beta\ndef test_beta():\n    {beta}\n"
        "def test_unbound():\n    pass\n"
    )


def test_a_complete_execution_run_keeps_only_this_capability_and_passes(
    pytester: pytest.Pytester,
) -> None:
    result = _session(pytester, _module("pass", "raise AssertionError"), "--host-execution")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1, deselected=2)
    assert "host execution on alpha: 1 case(s) ran, 0 skipped" in result.outlines
    assert not any("HOST_EXECUTION_INCOMPLETE" in line for line in result.outlines)


def test_a_skipped_host_case_fails_the_execution_run(pytester: pytest.Pytester) -> None:
    result = _session(pytester, _module("pytest.skip('absent')", "pass"), "--host-execution")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(skipped=1, deselected=2)
    assert "host execution on alpha: 0 case(s) ran, 1 skipped" in result.outlines
    assert (
        "HOST_EXECUTION_INCOMPLETE: an execution run must run every host case "
        "bound to alpha and at least one; a skip is not a run"
    ) in result.outlines


def test_an_execution_run_of_no_host_case_fails(pytester: pytest.Pytester) -> None:
    body = "import pytest\n@pytest.mark.host_beta\ndef test_beta():\n    pass\n"
    result = _session(pytester, body, "--host-execution")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(deselected=1)
    assert "host execution on alpha: 0 case(s) ran, 0 skipped" in result.outlines


def test_the_ordinary_run_skips_the_other_capability_naming_its_project(
    pytester: pytest.Pytester,
) -> None:
    result = _session(pytester, _module("pass", "raise AssertionError"), "-rs")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=2, skipped=1)
    assert [line for line in result.outlines if line.startswith("SKIPPED")] == [
        "SKIPPED [1] test_cases.py:5: executes on a beta host; tools/beta-execution runs it there"
    ]
    assert not any(line.startswith("host execution on") for line in result.outlines)


#: One execution-only case bound to each capability.
EXECUTION_ONLY: Final[str] = (
    "import pytest\n"
    "@pytest.mark.host_alpha_only\ndef test_alpha_only():\n    pass\n"
    "@pytest.mark.host_beta_only\ndef test_beta_only():\n    pass\n"
)


def test_the_ordinary_run_skips_every_execution_only_case_naming_its_need(
    pytester: pytest.Pytester,
) -> None:
    """Even the one bound to this run's capability: the ordinary run also
    runs in CI, which never has what an execution-only case needs."""
    result = _session(pytester, EXECUTION_ONLY, "-rs")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(skipped=2)
    assert sorted(line for line in result.outlines if line.startswith("SKIPPED")) == [
        "SKIPPED [1] test_cases.py:2: needs the alpha route; tools/alpha-execution runs it there",
        "SKIPPED [1] test_cases.py:5: needs the beta route; tools/beta-execution runs it there",
    ]


def test_the_execution_run_keeps_an_execution_only_case_of_this_capability(
    pytester: pytest.Pytester,
) -> None:
    result = _session(pytester, EXECUTION_ONLY, "--host-execution")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1, deselected=1)
    assert "host execution on alpha: 1 case(s) ran, 0 skipped" in result.outlines


def test_the_plan_declares_every_marker(pytester: pytest.Pytester) -> None:
    result = _session(pytester, "def test_nothing():\n    pass\n", "--markers")

    assert result.ret == pytest.ExitCode.OK
    for line in (
        "@pytest.mark.host_alpha: executes on a alpha host",
        "@pytest.mark.host_beta: executes on a beta host",
        "@pytest.mark.host_alpha_only: needs the alpha route; only the execution run runs it",
        "@pytest.mark.host_beta_only: needs the beta route; only the execution run runs it",
    ):
        assert line in result.outlines


def _plan(
    *,
    markers: dict[str, str],
    execution_only: dict[str, str],
    needs: dict[str, str],
    projects: dict[str, str],
) -> HostExecutionPlan:
    """Build a plan, holding the run's capability fixed.

    Args:
        markers: Marker to capability.
        execution_only: Execution-only marker to capability.
        needs: Execution-only marker to its need.
        projects: Capability to project.

    Returns:
        The plan.
    """
    return HostExecutionPlan(
        markers=markers,
        execution_only=execution_only,
        needs=needs,
        projects=projects,
        here="alpha",
    )


def test_a_plan_refuses_a_marker_declared_both_ways() -> None:
    with pytest.raises(ValueError, match=r"^HOST_PLAN_MARKER_TWICE: host_alpha$"):
        _plan(
            markers={"host_alpha": "alpha"},
            execution_only={"host_alpha": "alpha"},
            needs={"host_alpha": "needs it"},
            projects={"alpha": "tools/alpha-execution"},
        )


def test_a_plan_refuses_an_execution_only_marker_with_no_need() -> None:
    with pytest.raises(ValueError, match=r"^HOST_PLAN_NEED_MISSING: host_alpha_only$"):
        _plan(
            markers={},
            execution_only={"host_alpha_only": "alpha"},
            needs={},
            projects={"alpha": "tools/alpha-execution"},
        )


def test_a_plan_refuses_a_capability_no_project_runs() -> None:
    with pytest.raises(ValueError, match=r"^HOST_PLAN_PROJECT_MISSING: beta$"):
        _plan(
            markers={"host_alpha": "alpha", "host_beta": "beta"},
            execution_only={},
            needs={},
            projects={"alpha": "tools/alpha-execution"},
        )


def test_a_plan_keeps_its_own_copies() -> None:
    projects = {"alpha": "tools/alpha-execution"}
    plan = _plan(markers={"host_alpha": "alpha"}, execution_only={}, needs={}, projects=projects)
    projects["alpha"] = "changed"

    assert plan.projects == {"alpha": "tools/alpha-execution"}
    assert plan.markers == {"host_alpha": "alpha"}
    assert plan.execution_only == {}
    assert plan.needs == {}
    assert plan.here == "alpha"
