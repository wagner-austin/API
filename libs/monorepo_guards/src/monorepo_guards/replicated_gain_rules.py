"""Guard rule: a file that summarises replicated gains must also record them.

``gain_observations`` emits an arm's mean and its spread. ``spread`` is
``max - min``: a two-point summary of n numbers, and the points it drops are
exactly the ones a paired test needs. A record carrying only the summary can
be asked "is this bigger than the noise I saw" and can never afterwards be
asked "would I have detected an effect of size X", because the inputs are
gone.

THE COST, MEASURED RATHER THAN IMAGINED. On 2026-09-08 task 91e12be1 tried to
settle whether Model-Trainer's diverse n8 arm separates from its same-content
pool: a difference of 0.097017 against a floor of 0.069734, which is 1.3912x
the floor. At three seeds a paired t-threshold falls between 1.2421x and
1.4342x the range, so that claim sits inside the band where the two
instruments disagree -- it separates if the middle seed landed near the
midpoint of the range and does not if it landed near an extreme. The record
cannot say which. The per-seed values existed in memory when the record was
written; the cluster artifact and the job logs were both retrieved and neither
carries them. The published verdict is permanently unresolvable, and no amount
of access fixes it.

WHY A GUARD AND NOT A REVIEW NOTE. ``per_seed_observations`` already existed,
was exported, was tested, and its own docstring made this argument. It had
been wired into two of the six callers of one shared assembly helper. Nothing
declared which arrangement was correct and nothing checked, so the four that
lacked it looked exactly like the two that had it. That is the shape a rule
catches and a reader does not.

SCOPE IS THE FILE, unlike ``run-record``, whose pairing legitimately spans
modules. Here the two calls describe the SAME arm at the same moment: a file
that summarises an arm has that arm's replicates in hand, so there is no
correct arrangement in which the per-seed emission lives somewhere else.

WHAT IT DOES NOT CLAIM. Importing both symbols is not proof that every arm
reaches both, and this rule cannot show that -- a file could summarise two
arms and record seeds for one. It closes the gap where a file has no way to
emit per-seed values at all, which is the shape the real failure took in all
four sweeps.

Violations:
- replicated-gain-summary-only: a file emits gain_observations and never
  emits per_seed_observations
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from monorepo_guards import Violation
from monorepo_guards.util import parse_source

#: The summary emitter. Mean and spread; the range discards the replicates.
_SUMMARY_SYMBOL: Final[str] = "gain_observations"

#: The emitter that keeps them. One named scalar per seed.
_PER_SEED_SYMBOL: Final[str] = "per_seed_observations"

#: The module that defines both, which naturally names one without the other.
_DEFINING_MODULE: Final[str] = "replicated_measurement.py"


def _called_names(tree: ast.Module) -> set[str]:
    """Collect every bare function name called in a module.

    Keyed on calls rather than imports so that a file which imports a symbol
    and never uses it does not read as compliant -- an unused import is
    exactly what a half-applied fix leaves behind.

    Args:
        tree: Parsed module.

    Returns:
        The set of names appearing as a direct call target.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


class ReplicatedGainRule:
    """Guard rule pairing gain summaries with their per-seed replicates."""

    name = "replicated-gain"

    def run(self, files: list[Path]) -> list[Violation]:
        """Check every file that summarises a replicated gain.

        Args:
            files: Python source files to check.

        Returns:
            One violation per offending file.
        """
        violations: list[Violation] = []
        for path in sorted(files):
            if path.name == _DEFINING_MODULE:
                continue
            called = _called_names(parse_source(path))
            if _SUMMARY_SYMBOL in called and _PER_SEED_SYMBOL not in called:
                violations.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="replicated-gain-summary-only",
                        line=(
                            f"file calls {_SUMMARY_SYMBOL} but never "
                            f"{_PER_SEED_SYMBOL}; a mean and a max-min spread "
                            f"cannot be asked a paired question afterwards, and "
                            f"the replicates are gone once the record is written"
                        ),
                    )
                )
        return violations


__all__ = ["ReplicatedGainRule"]
