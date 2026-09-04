"""Tests for enumerating the commands in a command directory.

Built against real directories of real source under ``tmp_path``, not against
this repository's own packages. Keying platform_core's suite to the current
contents of ``model_trainer`` or ``hpc3`` would make this file fail whenever
someone added a command there, which is not this library's business.

The audit used to decide whether each command was GUARDED as well, by
searching its text for two substrings. Those functions are gone, and their
tests with them: the decision is made on the AST by
``monorepo_guards.entrypoint_rules.EntrypointRule``, which runs in every
package's ``make lint`` rather than only in the three packages that
remembered to call this. What remains is enumeration, which a guard cannot
do for a test that wants to RUN each command.
"""

from __future__ import annotations

import pathlib

from platform_core.entrypoint_audit import (
    command_modules,
    defines_entrypoint,
    public_modules,
)

#: A command as it should be written.
GUARDED = '''"""A command."""


def main(argv: object = None) -> int:
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
'''

#: Not a command at all: helpers with no entry point, which correctly need no
#: guard and must fall out of scope by shape rather than by exemption.
LIBRARY = '''"""Helpers, not a command."""


def render(value: str) -> str:
    return value
'''


def _package(root: pathlib.Path, modules: dict[str, str]) -> pathlib.Path:
    """Write a synthetic command directory.

    Args:
        root: Directory to build under.
        modules: File stem to source.

    Returns:
        The directory holding the modules.
    """
    cli = root / "cli"
    cli.mkdir()
    for stem, source in modules.items():
        (cli / f"{stem}.py").write_text(source, encoding="utf-8")
    return cli


def test_a_definition_is_a_definition() -> None:
    assert defines_entrypoint(GUARDED) is True


def test_a_module_without_an_entry_point_is_not_a_command() -> None:
    assert defines_entrypoint(LIBRARY) is False


def test_a_mention_of_the_entry_point_is_not_a_definition() -> None:
    # The reason this parses instead of searching for the word. A docstring or
    # an __all__ entry naming `entrypoint` would make a grep-based predicate
    # demand a guard on a module that has no command in it.
    assert defines_entrypoint('"""calls entrypoint"""\n__all__ = ["entrypoint"]') is False


def test_a_nested_function_named_entrypoint_is_not_a_definition() -> None:
    # The console script binds a MODULE-level name; a closure is not it.
    source = "def outer() -> None:\n    def entrypoint() -> None:\n        return None\n"

    assert defines_entrypoint(source) is False


def test_public_modules_excludes_private_and_dunder(tmp_path: pathlib.Path) -> None:
    cli = _package(tmp_path, {"alpha": GUARDED, "beta": LIBRARY})
    (cli / "__init__.py").write_text("", encoding="utf-8")
    (cli / "_helpers.py").write_text(LIBRARY, encoding="utf-8")

    assert [p.name for p in public_modules(cli)] == ["alpha.py", "beta.py"]


def test_command_modules_selects_less_than_the_directory(tmp_path: pathlib.Path) -> None:
    # The property that makes an exemption list unnecessary.
    cli = _package(tmp_path, {"alpha": GUARDED, "beta": LIBRARY})

    assert [p.name for p in command_modules(cli)] == ["alpha.py"]
    assert [p.name for p in public_modules(cli)] == ["alpha.py", "beta.py"]


def test_a_directory_of_only_libraries_yields_no_commands(tmp_path: pathlib.Path) -> None:
    # A caller parametrizes over this list. Returning the whole directory
    # when nothing in it is a command would have every helper run as a
    # command and fail for a reason that has nothing to do with the defect.
    cli = _package(tmp_path, {"beta": LIBRARY})

    assert command_modules(cli) == ()
