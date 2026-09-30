"""Host-execution runs: cases that execute on a real host, and a run that proves they did.

Board task 465689f5, rule R3 of b6eb30c8. Some cases in a package run what
the package ships against the real thing it touches: rendered PowerShell on
a Windows node, sh on a Linux node, a docker build under a node's rootless
daemon, or an ssh session to a cluster. Each is bound by a marker to the
CAPABILITY it needs, and a package's suite runs in one of two ways:

- The ORDINARY run (``make check``) skips a case whose capability this run
  does not have, naming the fleet project that runs it. An EXECUTION-ONLY
  case is skipped here on every machine, because the ordinary run also runs
  in CI, where the capability it needs (a node's execdocker user, the ssh
  route to a cluster) is never present even when the platform matches.
- The EXECUTION run (``--host-execution``) keeps only the cases bound to
  this run's capability and FAILS when any of them skips or when none ran.
  That run is what a package's execution fleet project checks, so a passed
  row of that project means the host cases executed, not that they were
  collected.

A package declares its markers and projects once, in a :class:`HostExecutionPlan`,
and its conftest-loaded plugin module forwards pytest's three hooks to
:func:`add_option`, :func:`configure` and :func:`select`. The hooks have to
be module-level functions in that module: ``pytest_addoption`` must run
while the command line is still being declared, before any plugin object a
later hook could register.

This module was lifted from tools/fleet's ``tests/_host.py``, where it was
bound to fleet's two platforms, so that tools/hpc3 could bind a case to the
cluster the same way instead of writing a second tally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest

#: The command-line option that makes a run the execution run.
EXECUTION_OPTION: Final[str] = "--host-execution"


class HostExecutionPlan:
    """What one package's host cases need, and which fleet project runs each.

    Attributes:
        markers: Marker name to the capability it binds a case to. The
            ordinary run skips such a case only where this run lacks the
            capability.
        execution_only: Marker name to the capability it binds a case to,
            for cases only the execution run may run.
        needs: Execution-only marker name to what the case needs, the first
            half of the reason the ordinary run gives for skipping it.
        projects: Capability to the fleet project whose check is this
            package's execution run on a machine that has it.
        here: The capability this run has.
    """

    __slots__ = ("execution_only", "here", "markers", "needs", "projects")

    def __init__(
        self,
        *,
        markers: Mapping[str, str],
        execution_only: Mapping[str, str],
        needs: Mapping[str, str],
        projects: Mapping[str, str],
        here: str,
    ) -> None:
        """Bind a package's markers, projects and this run's capability.

        Args:
            markers: Marker to capability, for cases the ordinary run may run.
            execution_only: Marker to capability, for cases it may not.
            needs: Execution-only marker to what its case needs.
            projects: Capability to the fleet project that runs its cases.
            here: The capability this run has.

        Raises:
            ValueError: When a marker is declared twice, an execution-only
                marker has no stated need, or a bound capability has no
                project to name.
        """
        shared = sorted(set(markers) & set(execution_only))
        if shared:
            raise ValueError(f"HOST_PLAN_MARKER_TWICE: {', '.join(shared)}")
        unstated = sorted(set(execution_only) - set(needs))
        if unstated:
            raise ValueError(f"HOST_PLAN_NEED_MISSING: {', '.join(unstated)}")
        bound = set(markers.values()) | set(execution_only.values())
        unrun = sorted(bound - set(projects))
        if unrun:
            raise ValueError(f"HOST_PLAN_PROJECT_MISSING: {', '.join(unrun)}")
        self.markers: Final[Mapping[str, str]] = dict(markers)
        self.execution_only: Final[Mapping[str, str]] = dict(execution_only)
        self.needs: Final[Mapping[str, str]] = dict(needs)
        self.projects: Final[Mapping[str, str]] = dict(projects)
        self.here: Final[str] = here


def bound_capability(item: pytest.Item, plan: HostExecutionPlan) -> str | None:
    """The capability a case is bound to, if any.

    Args:
        item: A collected case.
        plan: The package's plan.

    Returns:
        The capability its marker names, or None for an unbound case.
    """
    for marker, capability in plan.execution_only.items():
        if item.get_closest_marker(marker) is not None:
            return capability
    for marker, capability in plan.markers.items():
        if item.get_closest_marker(marker) is not None:
            return capability
    return None


class ExecutionTally:
    """Counts what an execution run did, on the process that reports it.

    Attributes:
        here: The capability the run has, named in its report.
        passed: Host cases whose call passed.
        skipped: Host cases that skipped in any phase.
    """

    def __init__(self, here: str) -> None:
        """Start from nothing run.

        Args:
            here: The capability the run has.
        """
        self.here = here
        self.passed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Count one phase's outcome.

        Args:
            report: The phase's report.
        """
        if report.skipped:
            self.skipped += 1
        elif report.passed and report.when == "call":
            self.passed += 1

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Fail the run unless every selected host case ran and at least one did.

        Args:
            session: The finished session.
        """
        if self.incomplete():
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        """Report the tally, and name the failure when the run was incomplete.

        The terminal reporter calls this after every ``pytest_sessionfinish``,
        so the exit status is already set when the lines are written.

        Args:
            terminalreporter: The session's terminal reporter.
        """
        terminalreporter.write_line(
            f"host execution on {self.here}: {self.passed} case(s) ran, {self.skipped} skipped"
        )
        if self.incomplete():
            terminalreporter.write_line(
                "HOST_EXECUTION_INCOMPLETE: an execution run must run every host case "
                f"bound to {self.here} and at least one; a skip is not a run"
            )

    def incomplete(self) -> bool:
        """Say whether a host case was skipped or none ran.

        Returns:
            True when the execution run must fail.
        """
        return bool(self.skipped) or not self.passed


def add_option(parser: pytest.Parser) -> None:
    """Declare the execution option; a plugin's ``pytest_addoption`` calls this.

    Args:
        parser: pytest's option parser.
    """
    parser.addoption(
        EXECUTION_OPTION,
        action="store_true",
        help="run only the host cases bound to this run's capability, failing on a skip or on none",
    )


def configure(config: pytest.Config, plan: HostExecutionPlan) -> None:
    """Declare the plan's markers, and count an execution run on the reporting process.

    A plugin's ``pytest_configure`` calls this. The tally registers only on
    the controller, never on an xdist worker, so a run split across workers
    is counted once.

    Args:
        config: The run's configuration.
        plan: The package's plan.
    """
    for marker, capability in plan.markers.items():
        config.addinivalue_line("markers", f"{marker}: executes on a {capability} host")
    for marker, need in plan.needs.items():
        config.addinivalue_line("markers", f"{marker}: {need}; only the execution run runs it")
    execution: bool = config.getoption(EXECUTION_OPTION)
    if execution and not hasattr(config, "workerinput"):
        config.pluginmanager.register(ExecutionTally(plan.here), "host-execution-tally")


def select(config: pytest.Config, items: list[pytest.Item], plan: HostExecutionPlan) -> None:
    """Keep the execution run's cases, or skip what the ordinary run cannot run.

    A plugin's ``pytest_collection_modifyitems`` calls this.

    Args:
        config: The run's configuration.
        items: The collected cases, narrowed in place for an execution run.
        plan: The package's plan.
    """
    execution: bool = config.getoption(EXECUTION_OPTION)
    if execution:
        kept = [item for item in items if bound_capability(item, plan) == plan.here]
        dropped = [item for item in items if bound_capability(item, plan) != plan.here]
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
        return
    for item in items:
        only = [
            marker for marker in plan.execution_only if item.get_closest_marker(marker) is not None
        ]
        if only:
            capability = plan.execution_only[only[0]]
            item.add_marker(
                pytest.mark.skip(
                    reason=f"{plan.needs[only[0]]}; {plan.projects[capability]} runs it there"
                )
            )
            continue
        bound = bound_capability(item, plan)
        if bound is not None and bound != plan.here:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"executes on a {bound} host; {plan.projects[bound]} runs it there"
                )
            )


__all__ = [
    "EXECUTION_OPTION",
    "ExecutionTally",
    "HostExecutionPlan",
    "add_option",
    "bound_capability",
    "configure",
    "select",
]
