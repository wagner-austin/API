"""Guard script for monorepo compliance checks.

Loads and runs the monorepo-guards orchestrator to enforce code quality rules.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from scripts import _test_hooks
from scripts.contract_rules import run_contract_rules
from scripts.hook_restore_rules import run_hook_restore_rules
from scripts.layer_rules import run_layer_rules
from scripts.physics_claims import run_physics_claim_rules
from scripts.protocol_constant_rules import run_protocol_constant_rules
from scripts.shim_rules import run_shim_rules
from scripts.state_sentinel_rules import run_mine_layer_rules, run_state_sentinel_rules
from scripts.wiki_rules import run_wiki_rules

# Record the script path at module load time for production use.
_test_hooks.set_script_path(Path(__file__).resolve())


def _find_monorepo_root(start: Path) -> Path:
    """Find the monorepo root by looking for a 'libs' directory.

    Args:
        start: Starting path to search upward from.

    Returns:
        Path to the monorepo root.

    Raises:
        RuntimeError: When no ancestor directory contains 'libs'.
    """
    current = start
    while True:
        if _test_hooks.is_dir(current / "libs"):
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard checks.

    Args:
        argv: Command line arguments. Uses sys.argv[1:] if None.

    Returns:
        Exit code (0 for success, non-zero for violations).
    """
    script_path = _test_hooks.get_script_path()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root(project_root)
    run_for_project = _test_hooks.load_orchestrator(monorepo_root)

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
    # Every local rule runs unconditionally (each reports its own
    # violations); a nonzero orchestrator rc is preserved rather than
    # flattened to 1, so the caller keeps the orchestrator's signal.
    local_violations = (
        run_contract_rules(target_root)
        + run_layer_rules(target_root)
        + run_shim_rules(target_root)
        + run_physics_claim_rules(target_root)
        + run_protocol_constant_rules(target_root)
        + run_state_sentinel_rules(target_root)
        + run_mine_layer_rules(target_root)
        + run_hook_restore_rules(target_root)
        + run_wiki_rules(target_root)
    )
    if local_violations > 0 and rc == 0:
        rc = 1
    if verbose:
        sys.stdout.write(f"guard_exit_code code={rc}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(None))
