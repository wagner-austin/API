"""A project must not depend on anything outside the monorepo.

Poetry path dependencies are how one project in this monorepo consumes
another, and they are resolved relative to the project that declares them —
so nothing stops one from climbing past the monorepo root into a sibling
checkout on the developer's disk:

.. code-block:: toml

    [tool.poetry.group.dev.dependencies]
    some-library = { path = "../../../some-library", develop = true }

That builds and tests locally and cannot build anywhere else, because the
directory it names is not part of this repository. Worse, it is invisible:
the dependency keeps working long after it should have been removed, because
``poetry install`` does not uninstall what the lock file no longer mentions,
so the import still resolves out of a stale virtualenv. Use ``poetry sync``
and the virtualenv matches the lock; this rule stops the dependency being
declared in the first place.

A project that genuinely needs code from outside the monorepo should vendor
it, with its provenance recorded, rather than reach across the filesystem.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.config import GuardConfig
from monorepo_guards.toml_reader import extract_path_dependencies, read_pyproject
from monorepo_guards.util import find_monorepo_root

PYPROJECT_FILENAME = "pyproject.toml"

KIND_ESCAPING_PATH = "dependency-path-escapes-monorepo"
KIND_UNROOTED = "dependency-check-outside-monorepo"


class EscapingPathDependencyRule:
    """Fails path dependencies that resolve outside the monorepo.

    Args:
        config (GuardConfig): Guard configuration, whose ``root`` is the
            project being checked.
    """

    def __init__(self, config: GuardConfig) -> None:
        """Store the configuration."""
        self._config = config

    @property
    def name(self) -> str:
        """str: Name shown in the guard summary."""
        return "dependency-escape"

    def run(self, files: list[Path]) -> list[Violation]:
        """Check the project's declared path dependencies.

        Args:
            files (list[Path]): Python files the orchestrator collected.
                Unused: this rule reads ``pyproject.toml``, which is not a
                Python file.

        Returns:
            list[Violation]: One violation per path dependency resolving
            outside the monorepo, or a single violation when the project is
            not inside a guarded monorepo at all. A project with no
            ``pyproject.toml`` declares no dependencies and so yields none.
        """
        project_root = self._config.root
        pyproject = project_root / PYPROJECT_FILENAME
        if not pyproject.is_file():
            return []

        monorepo_root = find_monorepo_root(project_root)
        if monorepo_root is None:
            return [
                Violation(
                    file=pyproject,
                    line_no=1,
                    kind=KIND_UNROOTED,
                    line=f"no ancestor of {project_root} holds the guard config",
                )
            ]

        violations: list[Violation] = []
        for dependency in extract_path_dependencies(read_pyproject(pyproject)):
            resolved = (project_root / dependency.path).resolve()
            if resolved != monorepo_root and monorepo_root not in resolved.parents:
                violations.append(
                    Violation(
                        file=pyproject,
                        line_no=dependency.line_no,
                        kind=KIND_ESCAPING_PATH,
                        line=f"{dependency.name} = {{ path = {dependency.path!r} }} resolves "
                        f"to {resolved}, outside {monorepo_root}",
                    )
                )
        return violations


__all__ = [
    "KIND_ESCAPING_PATH",
    "KIND_UNROOTED",
    "PYPROJECT_FILENAME",
    "EscapingPathDependencyRule",
]
