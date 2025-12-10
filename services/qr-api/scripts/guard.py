from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol


class _RunForProject(Protocol):
    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int: ...


def _default_is_dir(path: Path) -> bool:
    """Production implementation - uses Path.is_dir()."""
    return path.is_dir()


# Hook for testing - allows injecting a fake is_dir check.
_is_dir_hook: Callable[[Path], bool] = _default_is_dir


def _find_monorepo_root(start: Path) -> Path:
    current = start
    while True:
        if _is_dir_hook(current / "libs"):
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _load_orchestrator(monorepo_root: Path) -> _RunForProject:
    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: _RunForProject = mod.run_for_project
    return run_for_project


def main(argv: Sequence[str] | None = None) -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root(project_root)
    run_for_project = _load_orchestrator(monorepo_root)

    args = list(argv) if argv is not None else list(sys.argv[1:])
    root_override: Path | None = None
    verbose = False
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--root" and idx + 1 < len(args):
            root_override = Path(args[idx + 1]).resolve()
            idx += 2
        elif token in ("-v", "--verbose"):
            verbose = True
            idx += 1
        else:
            idx += 1

    target_root = root_override if root_override is not None else project_root
    rc = run_for_project(monorepo_root=monorepo_root, project_root=target_root)
    if verbose:
        sys.stdout.write(f"guard_exit_code code={rc}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(None))
