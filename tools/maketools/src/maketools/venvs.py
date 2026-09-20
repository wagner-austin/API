"""Virtual environments that poetry does not own, and first-party wheels.

Three packages step outside the poetry template:

* ``libs/cleargbm_rs`` is a maturin crate whose Python side lives in a
  ``uv`` venv; its recipes ran ``.venv\\Scripts\\ruff`` by Windows path.
  ``venv-exec`` resolves the executable for whichever layout the venv has
  (``Scripts`` on Windows, ``bin`` elsewhere) and ``uv-venv-check``
  recreates the venv when it is missing or cannot answer.
* ``services/covenant-radar-api`` depends on that crate WITHOUT
  ``develop = true`` and with its version frozen, so ``poetry install``
  considers it satisfied forever and rebuilding the crate leaves the venv
  running whatever build it first saw (it presented as ``module
  'cleargbm_rs' has no attribute 'py_gbm_model_to_json_rs'`` at conftest
  import). ``native-wheel`` reinstalls from the crate's newest wheel when
  that wheel is newer than the installed dist-info.
* ``clients/OrderedKernels`` installs first-party wheels rebuilt fresh
  every lint, deliberately not as path dependencies (platform-ml -> shap
  0.50.0 pins llvmlite to a PyPI-deleted beta, so the path-dep chain cannot
  solve on a fresh machine, 2026-09-01). ``poetry-build`` runs ``poetry
  build --quiet`` in each named package and stops at the first failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks
from maketools.venv_check import VENV_DIRECTORY

#: The two layouts a venv puts its executables in.
EXECUTABLE_DIRECTORIES: Final[tuple[str, ...]] = ("Scripts", "bin")

#: What ``uv`` is told to install into a fresh venv.
UV_INSTALL_ARGV: Final[tuple[str, ...]] = ("uv", "pip", "install", "-e", ".[dev]")


def venv_executable(project: Path, name: str) -> Path:
    """Locate an executable inside the project's venv.

    Args:
        project: The package directory.
        name: The executable's name without a suffix.

    Returns:
        Its path.

    Raises:
        AppError: ``MAKETOOLS_ARTIFACT_MISSING`` when neither layout has it.
    """
    venv = project / VENV_DIRECTORY
    for directory in EXECUTABLE_DIRECTORIES:
        for candidate in (venv / directory / name, venv / directory / f"{name}.exe"):
            if candidate.is_file():
                return candidate
    raise AppError(
        MaketoolsErrorCode.ARTIFACT_MISSING,
        f"{venv} has no {name} under {' or '.join(EXECUTABLE_DIRECTORIES)}",
    )


def venv_exec(project: Path, argv: Sequence[str]) -> int:
    """Run an executable from the project's venv.

    Args:
        project: The package directory.
        argv: The executable's name and its arguments.

    Returns:
        Its exit status.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when no executable was named.
    """
    if not argv:
        raise AppError(MaketoolsErrorCode.USAGE, "venv-exec needs an executable name")
    executable = venv_executable(project, argv[0])
    return _test_hooks.run_inheriting(
        [str(executable), *argv[1:]],
        cwd=project,
        env=_test_hooks.environ(),
        new_session=False,
    )


def uv_venv_check(project: Path) -> int:
    """Create the ``uv`` venv when missing, recreate it when it cannot answer.

    Args:
        project: The package directory.

    Returns:
        ``uv``'s status when it had to run, else 0.
    """
    venv = project / VENV_DIRECTORY
    if venv.is_dir():
        probe = _test_hooks.run_capturing(
            [str(venv_executable(project, "python")), "-m", "mypy", "--version"], cwd=project
        )
        if probe["returncode"] == 0:
            _test_hooks.write_line(f"uv-venv-check: .venv answers ({probe['stdout'].strip()})")
            return 0
        _test_hooks.write_line(
            f"uv-venv-check: stale venv detected (mypy --version exited "
            f"{probe['returncode']}); removing {venv}"
        )
        _test_hooks.remove_tree(venv)
    else:
        _test_hooks.write_line("uv-venv-check: no .venv yet; creating one")
    environment = _test_hooks.environ()
    code = _test_hooks.run_inheriting(
        ["uv", "venv"], cwd=project, env=environment, new_session=False
    )
    if code != 0:
        return code
    return _test_hooks.run_inheriting(
        UV_INSTALL_ARGV, cwd=project, env=environment, new_session=False
    )


def newest_wheel(crate: Path) -> Path | None:
    """The most recently written wheel under a crate's ``target/wheels``.

    Args:
        crate: The crate directory.

    Returns:
        The wheel, or None when none has been built.
    """
    wheels = sorted((crate.resolve() / "target" / "wheels").glob("*.whl"), key=modified_at)
    return wheels[-1] if wheels else None


def modified_at(path: Path) -> float:
    """A file's modification time, for ordering.

    Args:
        path: The file.

    Returns:
        Seconds since the epoch.
    """
    return path.stat().st_mtime


def installed_dist_info(project: Path, package: str) -> Path | None:
    """The installed package's dist-info directory, if any.

    Args:
        project: The package whose venv holds it.
        package: The distribution name as pip spells it.

    Returns:
        The directory, or None when not installed.
    """
    found = sorted((project / VENV_DIRECTORY).glob(f"**/site-packages/{package}-*.dist-info"))
    return found[0] if found else None


def native_wheel(project: Path, *, crate: Path, package: str) -> int:
    """Reinstall a native package from its crate's wheel when the wheel is newer.

    Args:
        project: The package whose venv is refreshed.
        crate: The crate directory, relative to the project.
        package: The distribution name.

    Returns:
        pip's status when it had to run, else 0.
    """
    wheel = newest_wheel(project / crate)
    if wheel is None:
        _test_hooks.write_line(
            f"native-wheel: no {package} wheel built under {crate.as_posix()}; skipping"
        )
        return 0
    info = installed_dist_info(project, package)
    if info is not None and info.stat().st_mtime >= wheel.stat().st_mtime:
        _test_hooks.write_line(f"native-wheel: {package} up to date")
        return 0
    _test_hooks.write_line(f"native-wheel: reinstalling {package} from {wheel.name}")
    return _test_hooks.run_inheriting(
        ["poetry", "run", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
        cwd=project,
        env=_test_hooks.environ(),
        new_session=False,
    )


def poetry_build(project: Path, packages: Sequence[Path]) -> int:
    """``poetry build --quiet`` in each package, stopping at the first failure.

    Args:
        project: The recipe's directory, which relative packages resolve against.
        packages: The packages to build.

    Returns:
        The first non-zero status, or 0.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when no package was named.
    """
    if not packages:
        raise AppError(MaketoolsErrorCode.USAGE, "poetry-build needs at least one package")
    for package in packages:
        _test_hooks.write_line(f"poetry-build: {package.as_posix()}")
        code = _test_hooks.run_inheriting(
            ["poetry", "build", "--quiet"],
            cwd=project / package,
            env=_test_hooks.environ(),
            new_session=False,
        )
        if code != 0:
            return code
    return 0


__all__ = [
    "EXECUTABLE_DIRECTORIES",
    "UV_INSTALL_ARGV",
    "installed_dist_info",
    "modified_at",
    "native_wheel",
    "newest_wheel",
    "poetry_build",
    "uv_venv_check",
    "venv_exec",
    "venv_executable",
]
