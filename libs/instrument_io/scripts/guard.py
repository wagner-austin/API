from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from scripts.mock_ban_rule import MockBanRule

# Import hooks from testing module
# Note: we import here instead of from instrument_io.testing to avoid
# circular import during module load
_find_monorepo_root_impl: Callable[[Path], Path] | None = None
_load_orchestrator_impl: Callable[[Path], Callable[[Path, Path], int]] | None = None


def _find_monorepo_root(start: Path) -> Path:
    """Find monorepo root, delegating to testing hooks."""
    from instrument_io.testing import hooks

    return hooks.find_monorepo_root(start)


def _load_orchestrator(monorepo_root: Path) -> Callable[[Path, Path], int]:
    """Load orchestrator, delegating to testing hooks."""
    from instrument_io.testing import hooks

    return hooks.load_orchestrator(monorepo_root)


def _run_local_rules(project_root: Path) -> int:
    """Run instrument_io-specific guard rules.

    Returns:
        0 if no violations found, 2 if violations found
    """
    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        return 0

    files: list[Path] = []
    for path in tests_dir.rglob("*.py"):
        if path.is_file():
            files.append(path)

    rule = MockBanRule()
    violations = rule.run(files)

    if violations:
        sys.stderr.write("Mock ban rule violations found:\n")
        for v in violations:
            sys.stderr.write(f"  {v.file}:{v.line_no}: kind={v.kind}\n")
        return 2

    sys.stdout.write("Mock ban rule: no violations found.\n")
    return 0


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

    rc = run_for_project(monorepo_root, target_root)
    if rc != 0:
        return rc

    local_rc = _run_local_rules(target_root)
    if verbose:
        sys.stdout.write(f"guard_exit_code code={local_rc}\n")
    return local_rc


if __name__ == "__main__":
    raise SystemExit(main(None))
