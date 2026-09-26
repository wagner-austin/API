"""The command-line entry point ``scripts/run.py`` hands its arguments to.

One command per Makefile recipe shape that used to need a shell. Each is a
named function over the package's modules; this file owns only the argument
parsing and the exit code, and prints a refusal as ``CODE: message`` so a
recipe's failure names its cause.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks, workspace
from maketools.env_run import run_env
from maketools.guard_run import run_guard
from maketools.makefile_banner_rule import lint_banners
from maketools.makefile_grammar import lint_grammar, render_violation
from maketools.reap import DEFAULT_OLDER_THAN_MINUTES, sweep_stale
from maketools.test_run import Runner, run_tests
from maketools.venv_check import check_venv
from maketools.venvs import native_wheel, poetry_build, uv_venv_check, venv_exec

#: The flag that skips the pre-run sweep; for debugging only.
NO_SWEEP_FLAG: Final[str] = "--no-sweep"

#: The flag that runs a suite without xdist.
SERIAL_FLAG: Final[str] = "--serial"

#: The flag that names how pytest is reached.
RUNNER_FLAG: Final[str] = "--runner"

#: The flag that sets the sweep's minimum age.
OLDER_THAN_FLAG: Final[str] = "--older-than-minutes"

#: ``compose-up``'s flags.
GIT_COMMIT_FLAG: Final[str] = "--git-commit"
PROGRESS_FLAG: Final[str] = "--build-progress"

#: ``native-wheel``'s flags.
CRATE_FLAG: Final[str] = "--crate"
PACKAGE_FLAG: Final[str] = "--package"


def command_venv_check(arguments: Sequence[str]) -> int:
    """``venv-check``: remove a stale ``.venv``.

    Args:
        arguments: None expected.

    Returns:
        0; the removal is a side effect, not a verdict.
    """
    require_no_arguments("venv-check", arguments)
    check_venv(Path.cwd())
    return 0


def command_guard(arguments: Sequence[str]) -> int:
    """``guard``: run the package's guard shim when it has one.

    Args:
        arguments: None expected.

    Returns:
        The shim's exit status.
    """
    require_no_arguments("guard", arguments)
    return run_guard(Path.cwd())


def command_test(arguments: Sequence[str]) -> int:
    """``test [--no-sweep] [--serial] [--runner poetry|venv] [pytest args...]``.

    Args:
        arguments: The launcher's flags first, then anything for pytest.

    Returns:
        pytest's exit status.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` on an unknown runner.
    """
    sweep = True
    serial = False
    runner = Runner.POETRY
    remaining = list(arguments)
    while remaining and remaining[0] in (NO_SWEEP_FLAG, SERIAL_FLAG, RUNNER_FLAG):
        flag = remaining.pop(0)
        if flag == NO_SWEEP_FLAG:
            sweep = False
        elif flag == SERIAL_FLAG:
            serial = True
        else:
            runner = require_runner(remaining.pop(0) if remaining else "")
    return run_tests(Path.cwd(), remaining, sweep=sweep, serial=serial, runner=runner)


def require_runner(value: str) -> Runner:
    """Narrow a ``--runner`` value.

    Args:
        value: The text after the flag.

    Returns:
        The runner.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when it is neither runner.
    """
    if value == Runner.POETRY:
        return Runner.POETRY
    if value == Runner.VENV:
        return Runner.VENV
    raise AppError(MaketoolsErrorCode.USAGE, f"{RUNNER_FLAG} must be poetry or venv, got {value!r}")


def command_reap_stale(arguments: Sequence[str]) -> int:
    """``reap-stale [--older-than-minutes N]``: the standalone sweep.

    Args:
        arguments: The optional age flag and its value.

    Returns:
        1 when a target could not be killed, else 0.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` on any other argument.
    """
    older_than = DEFAULT_OLDER_THAN_MINUTES
    if list(arguments[:1]) == [OLDER_THAN_FLAG] and len(arguments) == 2:
        older_than = require_integer(OLDER_THAN_FLAG, arguments[1])
    elif arguments:
        raise AppError(
            MaketoolsErrorCode.USAGE,
            f"reap-stale takes only {OLDER_THAN_FLAG} N, got {list(arguments)}",
        )
    report = sweep_stale(Path.cwd(), older_than_minutes=older_than)
    return 1 if report["failed"] > 0 else 0


def command_lint_makefiles(arguments: Sequence[str]) -> int:
    """``lint-makefiles``: the grammar and the banner rule over every tracked Makefile.

    Args:
        arguments: None expected.

    Returns:
        1 when any Makefile is outside the grammar or a ``check:`` target
        does not print the pass banner last, else 0.
    """
    require_no_arguments("lint-makefiles", arguments)
    repo_root = repository_root()
    examined, grammar_violations = lint_grammar(repo_root)
    violations = [*grammar_violations, *lint_banners(repo_root)]
    for violation in violations:
        _test_hooks.write_error(render_violation(violation, repo_root))
    if violations:
        _test_hooks.write_error(
            f"lint-makefiles: {len(violations)} violation(s) in {examined} tracked Makefile(s)"
        )
        return 1
    _test_hooks.write_line(
        f"lint-makefiles: {examined} tracked Makefile(s) in the portable grammar, "
        "every one beginning with the shell prologue and every check printing the pass banner"
    )
    return 0


def command_env(arguments: Sequence[str]) -> int:
    """``env NAME=VALUE... [--draw NAME=LOW-HIGH] [--then "cmd"] -- argv...``.

    See :mod:`maketools.env_run`.

    Args:
        arguments: The assignments, the draws, the separator and the command.

    Returns:
        The command's status.
    """
    return run_env(arguments, Path.cwd())


def command_fan_out(arguments: Sequence[str]) -> int:
    """``fan-out TARGET PARENT...``: ``make TARGET`` in every package under the parents.

    Args:
        arguments: The target, then the parent directories.

    Returns:
        1 when any package failed, else 0.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` without a target and a parent.
    """
    if len(arguments) < 2:
        raise AppError(MaketoolsErrorCode.USAGE, "fan-out needs a target and at least one parent")
    return workspace.fan_out(arguments[0], [Path(p) for p in arguments[1:]], cwd=Path.cwd())


def command_compose_up(arguments: Sequence[str]) -> int:
    """``compose-up DIR [--git-commit] [--build-progress plain]``.

    Args:
        arguments: The service directory and its flags.

    Returns:
        compose's status.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` on a missing directory or a stray flag.
    """
    if not arguments:
        raise AppError(MaketoolsErrorCode.USAGE, "compose-up needs a service directory")
    git_commit = False
    progress = ""
    rest = list(arguments[1:])
    while rest:
        flag = rest.pop(0)
        if flag == GIT_COMMIT_FLAG:
            git_commit = True
        elif flag == PROGRESS_FLAG and rest:
            progress = rest.pop(0)
        else:
            raise AppError(MaketoolsErrorCode.USAGE, f"compose-up does not take {flag!r}")
    return workspace.compose_up(
        Path.cwd() / arguments[0], build_progress=progress, git_commit=git_commit
    )


def command_compose_down(arguments: Sequence[str]) -> int:
    """``compose-down DIR...``: ``docker compose down`` in each directory.

    Args:
        arguments: The service directories.

    Returns:
        The first non-zero status, or 0.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` without a directory.
    """
    if not arguments:
        raise AppError(MaketoolsErrorCode.USAGE, "compose-down needs at least one directory")
    return workspace.compose_down([Path.cwd() / d for d in arguments])


def command_hooks(arguments: Sequence[str]) -> int:
    """``hooks install|check``: this clone's ``core.hooksPath``.

    Args:
        arguments: The one word.

    Returns:
        The verdict.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` on anything else.
    """
    if list(arguments) == ["install"]:
        return workspace.hooks_install(Path.cwd())
    if list(arguments) == ["check"]:
        return workspace.hooks_check(Path.cwd())
    raise AppError(MaketoolsErrorCode.USAGE, f"hooks takes install or check, got {list(arguments)}")


def command_require_tool(arguments: Sequence[str]) -> int:
    """``require-tool NAME HINT``: refuse when NAME is not on the PATH.

    Args:
        arguments: The tool and the install hint.

    Returns:
        0 when found.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` without both words.
    """
    if len(arguments) != 2:
        raise AppError(MaketoolsErrorCode.USAGE, "require-tool needs NAME and HINT")
    workspace.require_tool(arguments[0], arguments[1])
    return 0


def command_uv_venv_check(arguments: Sequence[str]) -> int:
    """``uv-venv-check``: create or recreate a ``uv`` venv.

    Args:
        arguments: None expected.

    Returns:
        ``uv``'s status when it ran, else 0.
    """
    require_no_arguments("uv-venv-check", arguments)
    return uv_venv_check(Path.cwd())


def command_venv_exec(arguments: Sequence[str]) -> int:
    """``venv-exec NAME ARGS...``: run an executable from ``.venv``.

    Args:
        arguments: The executable and its arguments.

    Returns:
        Its status.
    """
    return venv_exec(Path.cwd(), arguments)


def command_native_wheel(arguments: Sequence[str]) -> int:
    """``native-wheel --crate DIR --package NAME``: see :func:`maketools.venvs.native_wheel`.

    Args:
        arguments: The two flags and their values.

    Returns:
        pip's status when it ran, else 0.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` unless exactly both flags are given.
    """
    if len(arguments) != 4 or arguments[0] != CRATE_FLAG or arguments[2] != PACKAGE_FLAG:
        raise AppError(
            MaketoolsErrorCode.USAGE, f"native-wheel takes {CRATE_FLAG} DIR {PACKAGE_FLAG} NAME"
        )
    return native_wheel(Path.cwd(), crate=Path(arguments[1]), package=arguments[3])


def command_poetry_build(arguments: Sequence[str]) -> int:
    """``poetry-build DIR...``: ``poetry build --quiet`` in each package.

    Args:
        arguments: The package directories.

    Returns:
        The first non-zero status, or 0.
    """
    return poetry_build(Path.cwd(), [Path(p) for p in arguments])


def repository_root() -> Path:
    """The monorepo root, from this package's own location.

    Returns:
        ``tools/maketools/src/maketools/cli.py`` is four levels below it.
    """
    return Path(__file__).resolve().parents[4]


def require_no_arguments(command: str, arguments: Sequence[str]) -> None:
    """Refuse arguments a command does not take.

    Args:
        command: The command, for the message.
        arguments: What was passed.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when any were.
    """
    if arguments:
        raise AppError(
            MaketoolsErrorCode.USAGE, f"{command} takes no arguments, got {list(arguments)}"
        )


def require_integer(flag: str, value: str) -> int:
    """Parse a flag's integer value.

    Args:
        flag: The flag, for the message.
        value: Its text.

    Returns:
        The integer.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when it is not one.
    """
    if not value.isdigit():
        raise AppError(MaketoolsErrorCode.USAGE, f"{flag} needs an integer, got {value!r}")
    return int(value)


COMMANDS: Final[Mapping[str, Callable[[Sequence[str]], int]]] = {
    "venv-check": command_venv_check,
    "guard": command_guard,
    "test": command_test,
    "reap-stale": command_reap_stale,
    "lint-makefiles": command_lint_makefiles,
    "env": command_env,
    "fan-out": command_fan_out,
    "compose-up": command_compose_up,
    "compose-down": command_compose_down,
    "hooks": command_hooks,
    "require-tool": command_require_tool,
    "uv-venv-check": command_uv_venv_check,
    "venv-exec": command_venv_exec,
    "native-wheel": command_native_wheel,
    "poetry-build": command_poetry_build,
}


def dispatch(argv: Sequence[str]) -> int:
    """Route to a command.

    Args:
        argv: The command and its arguments.

    Returns:
        The command's exit status.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when no command or an unknown one was
            named.
    """
    if not argv:
        raise AppError(MaketoolsErrorCode.USAGE, f"a command is required: {sorted(COMMANDS)}")
    handler = COMMANDS.get(argv[0])
    if handler is None:
        raise AppError(
            MaketoolsErrorCode.USAGE, f"unknown command {argv[0]!r}; one of {sorted(COMMANDS)}"
        )
    return handler(argv[1:])


def main(argv: Sequence[str]) -> int:
    """Run a command and turn a refusal into an exit code.

    The one boundary: an :class:`AppError` becomes ``CODE: message`` on
    stderr and exit 1. Anything else propagates with its traceback, because
    an unexpected failure inside a build tool is a defect to read, not a
    message to tidy.

    Args:
        argv: The command and its arguments.

    Returns:
        The exit status.
    """
    try:
        return dispatch(argv)
    except AppError as refusal:
        _test_hooks.write_error(f"{refusal.code}: {refusal.message}")
        return 1


__all__ = [
    "COMMANDS",
    "CRATE_FLAG",
    "GIT_COMMIT_FLAG",
    "NO_SWEEP_FLAG",
    "OLDER_THAN_FLAG",
    "PACKAGE_FLAG",
    "PROGRESS_FLAG",
    "RUNNER_FLAG",
    "SERIAL_FLAG",
    "command_compose_down",
    "command_compose_up",
    "command_env",
    "command_fan_out",
    "command_guard",
    "command_hooks",
    "command_lint_makefiles",
    "command_native_wheel",
    "command_poetry_build",
    "command_reap_stale",
    "command_require_tool",
    "command_test",
    "command_uv_venv_check",
    "command_venv_check",
    "command_venv_exec",
    "dispatch",
    "main",
    "repository_root",
    "require_integer",
    "require_no_arguments",
    "require_runner",
]
