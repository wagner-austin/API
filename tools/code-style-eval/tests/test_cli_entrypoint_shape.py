"""Every CLI that can be run as a module actually runs when you do.

THIS PACKAGE IS THE THIRD TO HAVE THIS DEFECT, and the first two both left a
note saying so. `model_trainer`'s version of this file records that "the same
shape had already been hit twice in the sibling ``hpc3`` package -- where it
then sat unfixed until 2026-08-28, when all twelve hpc3 commands turned out
to have it, including a preflight that reported a clean green light without
checking anything". Both packages then added the guard AND a test. This one
had neither.

WHAT IT COST HERE, on 2026-09-04. A scoring run over 226 generated files --
which an A30 had just spent thirty-three minutes producing -- was invoked as
``python -m code_style_eval.cli.evaluate`` with every flag correct. It
imported the module, defined its functions, wrote no outcomes file and exited
**0**. Exit zero, no output, no error. The failure looks identical to a run
that legitimately scored nothing, which is a state this scorer can genuinely
reach when no generated file matches a prompt.

WHERE THE SHAPE CHECK LIVES, AND WHY NOT HERE. Three packages each added
their own copy of it, one at a time, after each was bitten -- which is the
same as saying that a package which had not yet been bitten had no check.
``monorepo_guards.entrypoint_rules.EntrypointRule`` now decides it on the AST
for every package in the repository, inside every ``make lint``, including
packages nobody has written yet. This file does not assert it a second time.

WHAT IS LEFT HERE IS WHAT A GUARD CANNOT DO: run the commands. A guard reads
files and reports on their shape. Only the ``runpy`` test below invokes each
command exactly as ``python -m`` does and watches it refuse, which proves the
chain executes rather than describing it.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.entrypoint_audit import command_modules, public_modules

CLI_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "code_style_eval" / "cli"


def _command_names() -> tuple[str, ...]:
    """The importable name of every command in this package.

    Derived from the directory rather than listed, so a command added later
    is exercised without an edit here -- and a command added later that does
    not run is what this file exists to catch.

    Returns:
        Dotted module names, sorted.
    """
    return tuple(f"code_style_eval.cli.{path.stem}" for path in command_modules(CLI_DIR))


def test_the_scan_has_subjects() -> None:
    # A rule that silently scans nothing passes forever, which is the same
    # failure shape as the command it exists to catch.
    assert len(public_modules(CLI_DIR)) >= 2


def test_every_command_is_a_public_module() -> None:
    # Keyed on defining `entrypoint` rather than on a filename, so a helper
    # module added later falls out of scope by its own shape. The containment
    # says the two derivations are looking at the same directory; which
    # modules are actually commands is proven by running them below.
    assert set(command_modules(CLI_DIR)) <= set(public_modules(CLI_DIR))


@pytest.mark.parametrize("module_name", _command_names())
def test_running_a_command_as_a_module_reaches_its_entry_point(module_name: str) -> None:
    """Run each command through ``runpy`` exactly as ``python -m`` does.

    Given no arguments every command here is missing a required flag, so
    ``parse_arguments`` raises ValueError. Reaching that error proves the
    whole chain ran: main guard, entry point, main, and the parser. The
    silent version of this module reached none of it and exited 0.

    Args:
        module_name: The command to run.
    """
    saved_argv = sys.argv
    saved_module = sys.modules.pop(module_name, None)
    sys.argv = [module_name.rsplit(".", 1)[1]]
    try:
        with pytest.raises(ValueError, match="is required"):
            runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved_argv
        if saved_module is not None:
            sys.modules[module_name] = saved_module
