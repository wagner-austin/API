"""Run the two wiki gates as a lint step, not only as tests.

Both rules were complete, tested and green -- and reachable from
nothing but their own test suites. ``physics_claims`` opens by calling
itself a "Guard rule"; it was never in ``scripts/guard.py``, which is
the byte-identical shim all forty-one packages share and so cannot
carry a rule belonging to one of them. The rules did gate the
repository, through pytest, which is why a wiki page committed ahead of
its code turned CI red this morning. But a violation surfaced as a test
failure names the assertion rather than the page, and arrives after the
whole suite has run.

This is the shape ``check_undecoded_fields`` already uses in this
package: a ``run`` that returns an exit code, a ``main`` that exits
with it, and a Makefile lint step between the guard and ruff. The
rules keep their pytest tests -- those cover the fixtures and the
committed-tree binding, which a lint step cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.physics_claims import run_physics_claim_rules
from scripts.wiki_rules import run_wiki_rules

#: The package root both rules read ``wiki/pages`` beneath.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(project_root: Path = PROJECT_ROOT) -> int:
    """Run both wiki gates and report the CLI exit code.

    Both run even when the first finds violations: a page can fail
    structure and binding independently, and reporting one of the two
    would send a reader back for a second pass.

    Args:
        project_root: Package root holding ``wiki/pages``.

    Returns:
        ``0`` when both gates are clean; ``1`` otherwise.
    """
    structure = run_wiki_rules(project_root)
    binding = run_physics_claim_rules(project_root)
    total = structure + binding
    if total == 0:
        sys.stdout.write("wiki gates: structure and claim binding both clean\n")
        return 0
    sys.stdout.write(
        f"wiki gates: {structure} structure violation(s), {binding} claim-binding violation(s)\n"
    )
    return 1


def main() -> None:
    """Entry point for the ``tankpit-check-wiki`` script."""
    sys.exit(run())


if __name__ == "__main__":
    main()


__all__ = ["PROJECT_ROOT", "main", "run"]
