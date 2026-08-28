"""Tests for the command entry-point audit.

Built against real directories of real source under ``tmp_path``, not against
this repository's own packages. Two reasons. The audit's whole job is to
report a defect, and a test that only ever pointed it at correct code could
never see it report one -- every assertion would be `== ()` and the functions
could return `()` unconditionally and pass. And keying platform_core's suite
to the current contents of ``model_trainer`` or ``hpc3`` would make this file
fail whenever someone added a command there, which is not this library's
business.
"""

from __future__ import annotations

import pathlib

from platform_core.entrypoint_audit import (
    GUARD_CALL,
    MAIN_GUARD,
    command_modules,
    defines_entrypoint,
    misguarded_commands,
    public_modules,
    unguarded_commands,
)

#: A command as it should be written.
GUARDED = f'''"""A command."""


def main(argv: object = None) -> int:
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["entrypoint", "main"]


if {MAIN_GUARD}:
{GUARD_CALL}
'''

#: The defect: an entry point with no guard, so `python -m` does nothing.
UNGUARDED = '''"""A command that does nothing when run as a module."""


def main(argv: object = None) -> int:
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
'''

#: The defect one layer in: guarded, but calling the wrong function. `main`
#: returns a status into nothing, so the process exits 0 regardless.
MISGUARDED = f'''"""A command whose guard drops the exit code."""


def main(argv: object = None) -> int:
    return 1


def entrypoint() -> None:
    raise SystemExit(main())


if {MAIN_GUARD}:
    main()
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


def test_an_unguarded_command_is_reported(tmp_path: pathlib.Path) -> None:
    cli = _package(tmp_path, {"alpha": GUARDED, "broken": UNGUARDED})

    assert unguarded_commands(cli) == ("broken.py",)


def test_a_library_without_a_guard_is_not_reported(tmp_path: pathlib.Path) -> None:
    # `LIBRARY` has no guard and must not be demanded to have one.
    cli = _package(tmp_path, {"beta": LIBRARY})

    assert unguarded_commands(cli) == ()


def test_a_fully_guarded_package_reports_nothing(tmp_path: pathlib.Path) -> None:
    cli = _package(tmp_path, {"alpha": GUARDED, "gamma": GUARDED})

    assert unguarded_commands(cli) == ()
    assert misguarded_commands(cli) == ()


def test_a_guard_calling_main_is_reported(tmp_path: pathlib.Path) -> None:
    # Guarded, so `unguarded_commands` is silent -- and still broken, because
    # `main()` returns its status into nothing and the process exits 0.
    cli = _package(tmp_path, {"sneaky": MISGUARDED})

    assert unguarded_commands(cli) == ()
    assert misguarded_commands(cli) == ("sneaky.py",)


def test_an_unguarded_command_is_not_also_reported_as_misguarded(
    tmp_path: pathlib.Path,
) -> None:
    # The two findings are disjoint: a module with no guard cannot have a
    # wrong one, and reporting it twice would double-count one defect.
    cli = _package(tmp_path, {"broken": UNGUARDED})

    assert misguarded_commands(cli) == ()
