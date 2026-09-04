"""Every command in this package actually runs when run as a module.

WHAT IT COST HERE. All twelve commands lacked the ``__main__`` guard on
2026-08-28, so ``python -m hpc3.cli.<anything>`` exited 0 having done nothing.
The console scripts poetry generates call ``entrypoint`` directly and have
always worked, so the two invocation forms disagreed in silence.

Preflight is the sharp case. Its entire job is answering "would this job
start?", so a preflight that never runs and exits 0 is not a missing check --
it is a GREEN LIGHT that checked nothing. Four jobs were one step from being
submitted on the strength of exactly that.

WHERE THE SHAPE CHECK LIVES, AND WHY NOT HERE. It lived in a shared scan that
each package CALLED, which moved the problem rather than solving it:
``code_style_eval`` never called it, and on 2026-09-04 lost a scoring pass
over 226 files to the same defect -- the third package, after the note in
``model_trainer``'s test had already recorded that hpc3 had it too.
``monorepo_guards.entrypoint_rules.EntrypointRule`` decides it now, on the
AST, for every package inside every ``make lint``. Nothing has to remember
to call it, so this file no longer asserts it.

WHAT THIS FILE STILL DOES, BECAUSE A GUARD CANNOT. A guard proves the block is
PRESENT. :func:`test_running_a_command_as_a_module_reaches_its_entry_point`
proves it FIRES, which is the property that was actually missing, and it is
what makes the twelve guard lines covered by a test that means something
rather than by a pragma. Its command list comes from
:mod:`platform_core.entrypoint_audit`, which enumerates and no longer
judges.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from platform_core.entrypoint_audit import (
    command_modules,
    defines_entrypoint,
    public_modules,
)

from hpc3.cli import _test_hooks

CLI_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "hpc3" / "cli"

#: What a command exits with when it refuses -- ``_fatal.EXIT_REFUSED``. Run
#: with no arguments, every command here is missing a required flag, raises
#: ValueError, and that is translated to this status.
EXIT_REFUSED = 2


def _command_names() -> tuple[str, ...]:
    """The importable name of every command in this package.

    Returns:
        Dotted module names, sorted.
    """
    return tuple(f"hpc3.cli.{p.stem}" for p in command_modules(CLI_DIR))


def _capture_errors() -> Generator[list[str], None, None]:
    """Route refusals into a list for the duration of one test.

    Yields:
        The lines the command wrote to its error stream.
    """
    captured: list[str] = []
    _test_hooks.emit_error = captured.append
    yield captured
    _test_hooks.reset_hooks()


capture_errors = pytest.fixture(_capture_errors)


def test_the_cli_directory_is_not_empty() -> None:
    # A rule that silently scans nothing passes forever. This is the check
    # that the check has subjects.
    assert len(public_modules(CLI_DIR)) >= 10


def test_every_public_module_here_is_a_command() -> None:
    # Not a restatement of the scan: it records that in THIS package the
    # entry-point predicate and the directory currently select the same set.
    # The model_trainer twin asserts the opposite -- it has a report-helper
    # library under cli/ -- and the shared predicate is what lets one rule
    # serve both without either needing an exemption list.
    assert command_modules(CLI_DIR) == public_modules(CLI_DIR)


def test_a_mention_of_the_entry_point_is_not_a_definition() -> None:
    # The shared predicate parses rather than searches. Asserted from this
    # side too because the whole no-exemption-list property rests on it.
    assert defines_entrypoint("def entrypoint() -> None: ...") is True
    assert defines_entrypoint("def main() -> int: ...") is False
    assert defines_entrypoint('"""calls entrypoint"""\n__all__ = ["entrypoint"]') is False


@pytest.mark.parametrize("module_name", _command_names())
def test_running_a_command_as_a_module_reaches_its_entry_point(
    module_name: str, capture_errors: list[str]
) -> None:
    """Run each command through ``runpy`` exactly as ``python -m`` does.

    Given no arguments every command here lacks a required flag, so it raises
    ValueError, ``_fatal.run`` translates that to :data:`EXIT_REFUSED`, and
    ``entrypoint`` raises SystemExit carrying it. Reaching that status proves
    the whole chain ran: guard, entry point, main, and the typed translator.

    Args:
        module_name: The command to run.
        capture_errors: Collected refusal lines.
    """
    saved_argv = sys.argv
    saved_module = sys.modules.pop(module_name, None)
    sys.argv = [module_name.rsplit(".", 1)[1]]
    try:
        with pytest.raises(SystemExit) as raised:
            runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved_argv
        if saved_module is not None:
            sys.modules[module_name] = saved_module

    assert raised.value.code == EXIT_REFUSED
    assert len(capture_errors) == 1
    assert capture_errors[0].startswith("usage: ")
