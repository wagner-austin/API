"""The execution run fails unless every host case of this platform ran.

Board task 465689f5. ``tests/_host.py`` is what makes a passed
tools/fleet-execution run mean the host cases executed: it keeps only this
platform's cases and turns a skip, or a run of none, into a failure. A
plugin that quietly passed a run of zero would make the deploy gate vouch
for nothing, so each case here runs a real pytest session in a subprocess
with that module's own bytes as its plugin, and reads the exit status and
the tally line it printed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests._host import EXECUTION_ONLY_MARKER, HOST_PLATFORM, MARKERS, PROJECTS

#: The platform this machine is not.
OTHER_PLATFORM: Final[str] = "linux" if HOST_PLATFORM == "windows" else "windows"

#: The plugin's source, loaded into each session under its own name.
PLUGIN: Final[str] = (Path(__file__).resolve().parent / "_host.py").read_text(encoding="utf-8")


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
    return pytester.runpytest_subprocess(
        "-p", "host_plugin", "-p", "no:cacheprovider", "-n", "0", *arguments
    )


def _cases(this: str, other: str) -> str:
    """A module with one case bound to each platform.

    Args:
        this: The body of the case bound to this platform.
        other: The body of the case bound to the other one.

    Returns:
        The module's source.
    """
    return (
        "import pytest\n"
        f"@pytest.mark.{MARKERS[HOST_PLATFORM]}\n"
        f"def test_here():\n    {this}\n"
        f"@pytest.mark.{MARKERS[OTHER_PLATFORM]}\n"
        f"def test_there():\n    {other}\n"
        "def test_unbound():\n    pass\n"
    )


def test_a_complete_run_keeps_only_this_platform_and_passes(pytester: pytest.Pytester) -> None:
    result = _session(pytester, _cases("pass", "raise AssertionError"), "--host-execution")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1, deselected=2)
    assert f"host execution on {HOST_PLATFORM}: 1 case(s) ran, 0 skipped" in result.outlines
    assert not any("HOST_EXECUTION_INCOMPLETE" in line for line in result.outlines)


def test_a_skipped_host_case_fails_the_run(pytester: pytest.Pytester) -> None:
    result = _session(pytester, _cases("pytest.skip('absent')", "pass"), "--host-execution")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(skipped=1, deselected=2)
    assert f"host execution on {HOST_PLATFORM}: 0 case(s) ran, 1 skipped" in result.outlines
    assert (
        "HOST_EXECUTION_INCOMPLETE: an execution run must run every host case "
        f"bound to {HOST_PLATFORM} and at least one; a skip is not a run"
    ) in result.outlines


def test_a_run_of_no_host_case_fails(pytester: pytest.Pytester) -> None:
    body = f"import pytest\n@pytest.mark.{MARKERS[OTHER_PLATFORM]}\ndef test_there():\n    pass\n"
    result = _session(pytester, body, "--host-execution")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(deselected=1)
    assert f"host execution on {HOST_PLATFORM}: 0 case(s) ran, 0 skipped" in result.outlines


def test_the_ordinary_run_skips_the_other_platform_naming_its_project(
    pytester: pytest.Pytester,
) -> None:
    result = _session(pytester, _cases("pass", "raise AssertionError"), "-rs")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=2, skipped=1)
    reason = f"executes on a {OTHER_PLATFORM} host; {PROJECTS[OTHER_PLATFORM]} runs it there"
    assert [line for line in result.outlines if line.startswith("SKIPPED")] == [
        f"SKIPPED [1] test_cases.py:5: {reason}"
    ]
    assert not any(line.startswith("host execution on") for line in result.outlines)


#: A module with one case that needs the node's rootless daemon.
DOCKER_CASE: Final[str] = (
    f"import pytest\n@pytest.mark.{EXECUTION_ONLY_MARKER}\ndef test_isolated():\n    pass\n"
)


def test_the_ordinary_run_skips_a_docker_case_on_every_platform(
    pytester: pytest.Pytester,
) -> None:
    """Even on Linux: API's CI is Linux and has no execdocker user, so only
    the execution run on a docker node may run it."""
    result = _session(pytester, DOCKER_CASE, "-rs")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(skipped=1)
    reason = (
        f"needs a node's execdocker user and its rootless daemon; {PROJECTS['linux']} runs it there"
    )
    assert [line for line in result.outlines if line.startswith("SKIPPED")] == [
        f"SKIPPED [1] test_cases.py:2: {reason}"
    ]


def test_the_execution_run_keeps_a_docker_case_only_on_linux(pytester: pytest.Pytester) -> None:
    result = _session(pytester, DOCKER_CASE, "--host-execution")

    kept = {"linux": {"passed": 1}, "windows": {"deselected": 1}}[HOST_PLATFORM]
    result.assert_outcomes(**kept)


def test_the_tally_counts_on_the_controller_under_xdist(pytester: pytest.Pytester) -> None:
    body = _cases("pass", "pass") + "".join(
        f"@pytest.mark.{MARKERS[HOST_PLATFORM]}\ndef test_more_{index}():\n    pass\n"
        for index in range(3)
    )
    result = _session(pytester, body, "--host-execution", "-n", "2")

    assert result.ret == pytest.ExitCode.OK
    assert f"host execution on {HOST_PLATFORM}: 4 case(s) ran, 0 skipped" in result.outlines
