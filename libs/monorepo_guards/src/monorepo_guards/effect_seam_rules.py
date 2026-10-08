"""Guard rule ``effect-seam-twin``: no fake stands alone.

The operator, 2026-10-07, on learning the harness gate's tests faked the
installer: "i thought qe banned fakes? ok sounds like we need to enforce a
fake ana cross the entire mcps and ai cooridnation sysyem". The gate's suite
was at 100 percent because every test reached ``claude.exe install`` through
a ``_test_hooks`` fake; the real hook ran once, on a command that succeeded;
and the real installer, hung and killed mid-swap, left the machine with no
launcher (MCPs board tasks 5895c980 and c96e8791).

So for every EFFECT seam (:mod:`monorepo_guards.effect_seams`) some test must
run the real implementation (:mod:`monorepo_guards.effect_seam_twins`), and
one of those must take it through a failure of its own kind
(:mod:`monorepo_guards.effect_failures`): a process seam a timeout, exit or
kill, a file swap the OS refusing it. There is no allow-list:
a seam that cannot be run failing is a seam whose failure nobody has seen.

The rule and its wording are MCPs' (mcp-shared and mcp-shared-py, board task
c96e8791); API applies them under board task cc7222ca.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.config import GuardConfig
from monorepo_guards.effect_failures import SATISFIED_BY
from monorepo_guards.effect_seam_twins import real_tests
from monorepo_guards.effect_seams import effect_seams, index_package


class EffectSeamTwinRule:
    """Fail every effect seam with no real test, or none that fails it."""

    name = "effect-seam-twin"

    def __init__(self, config: GuardConfig) -> None:
        """Bind the rule to the package it checks.

        Args:
            config: The guard run's configuration; its ``root`` is where
                module names and ``tests`` are found.
        """
        self._root = config.root

    def run(self, files: list[Path]) -> list[Violation]:
        """Report each effect seam without a real twin.

        Args:
            files: The package's ``src``, ``scripts`` and ``tests`` files.

        Returns:
            One violation per seam no test runs for real, or whose real
            tests exercise no failure, naming the seam and its chain.
        """
        index = index_package(files, self._root)
        seams = effect_seams(index)
        twins = real_tests(files, self._root, index, seams)
        violations: list[Violation] = []
        for effect in seams:
            seam = effect.seam
            tests = twins[(seam.module.name, seam.label)]
            effect_kind = effect.reach.primitive.kind
            if any(test.kinds & SATISFIED_BY[effect_kind] for test in tests):
                continue
            chain = " -> ".join((seam.label, *effect.reach.chain))
            where = seam.module.path.relative_to(self._root).as_posix()
            head = f"{where}:{seam.label} ({effect_kind}: {chain})"
            if tests:
                names = ", ".join(test.label for test in tests)
                kind = "effect-seam-twin-no-failure"
                text = f"{head} its real tests {names} exercise no {effect_kind} failure"
            else:
                kind = "effect-seam-twin-missing"
                text = f"{head} no test runs its real implementation"
            violations.append(
                Violation(file=seam.module.path, line_no=seam.line_no, kind=kind, line=text)
            )
        return violations


__all__ = ["EffectSeamTwinRule"]
