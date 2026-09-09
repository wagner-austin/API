"""Guard rule: an entry point that PRODUCES a research record must be registered.

``docs/RESEARCH.md`` opens by declaring what it is for -- "Every body of work
on this machine that produces numbers someone compares" -- and
``tools/hpc3/tests/test_committed_runs.py`` already enforces one direction of
that: a project registered with the hpc3 CLI must agree with what the file
says. Nothing enforced the reverse. A command could emit a full ``RunRecord``,
run for weeks, and appear in no entry at all, because there was no claim to
check and silence is indistinguishable from compliance.

THE REAL INSTANCE, and it is why this rule exists rather than being tidiness.
``cartridge_qa_benchmark`` produced the cartridge programme's headline result
-- "the cartridge arm beats every retriever" -- and on 2026-09-09 it held: 0
of 571 committed run documents, 0 lines of ``docs/RESEARCH.md``, and 0
git-tracked artifacts, its records written to a temporary directory that is
purged. The claim was retracted the same day for being four times below what
its instrument could resolve. Nothing went red at any point, because an
unregistered surface makes no claim a reviewer can score.

A SECOND INSTANCE FOUND BY THIS RULE'S OWN FIRST RUN, which is the argument
for it being mechanical: ``score_baseline`` carries 96 committed run
documents, more than any other entry point in the workspace, and was equally
absent from the registry. The registry entry that should have named it ends
its list of commands with a literal ``...``. PROSE WITH AN ELLIPSIS CANNOT BE
CHECKED, and that ellipsis is the whole defect in one character.

PRODUCING, NOT READING, and the distinction is load-bearing. A report module
imports the ``RunRecord`` TYPE to annotate what it reads back;
``sdpa_benchmark_report`` does exactly that and is not a research surface. A
producer imports the ``run_record`` FACTORY or ``encode_run_record``, because
those are how a record comes to exist. Keying on the type would convict every
correct reporter, and a rule that fires on correct code is one an operator
learns to skip.

WHAT THIS RULE DOES NOT CLAIM. Registration is not power. A registered
entry point can still run an instrument too weak to resolve what it reports,
which is the defect ``docs/RESEARCH.md``'s own minimum-detectable-effect
sweep exists to catch. This rule closes the earlier gap -- a surface that
never entered the record at all -- and the two are siblings rather than
substitutes.

Violations:
- research-entry-point-unregistered: a CLI entry point builds a RunRecord and
  is named nowhere in the research registry
- research-registry-missing: entry points build RunRecords and the registry
  file does not exist
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from monorepo_guards import Violation
from monorepo_guards.config import GuardConfig
from monorepo_guards.util import imported_names, package_of, parse_source

#: Where the workspace declares every body of work that produces compared
#: numbers, relative to the monorepo root.
REGISTRY_RELATIVE_PATH: Final[tuple[str, ...]] = ("docs", "RESEARCH.md")

#: Importing one of these means a module BRINGS A RECORD INTO EXISTENCE.
#: ``run_record`` is the factory and ``encode_run_record`` is how one reaches
#: a file; the ``RunRecord`` type is deliberately absent, because annotating
#: what you read back is not producing it.
_PRODUCER_SYMBOLS: Final[frozenset[str]] = frozenset({"run_record", "encode_run_record"})

#: The directory segment that marks a module as an entry point. A library
#: module may legitimately build a record on behalf of the command that calls
#: it -- ``services/training/run_records.py`` does -- and it is the COMMAND
#: that a registry entry names.
_ENTRY_POINT_SEGMENT: Final[str] = "cli"

#: Packages exempt because they DEFINE the vocabulary rather than produce
#: research with it. ``platform_core`` owns the record types and
#: ``monorepo_guards`` names them only in this rule's own text and tests.
_DEFINING_PACKAGES: Final[frozenset[str]] = frozenset({"platform_core", "monorepo_guards"})


def _is_entry_point(path: Path) -> bool:
    """Decide whether a file is a command rather than a library module.

    Args:
        path: The file.

    Returns:
        True when the file sits under a ``cli`` directory and is not a private
        module. Private modules under ``cli`` are the hook and helper seams
        (``_test_hooks``, ``_measurement_hooks``); they are not invoked by a
        run document and a registry entry cannot name them.
    """
    return _ENTRY_POINT_SEGMENT in path.parts and not path.name.startswith("_")


def _is_registered(module_name: str, registry_text: str) -> bool:
    """Decide whether the registry names a module.

    Matched on a word boundary rather than as a substring. ``gemm_probe`` is a
    substring of ``legacy_gemm_probe``, so a substring test would report the
    shorter one registered on the strength of an entry describing the longer
    one -- two different measurements, one of them silently credited to the
    other's paperwork.

    Args:
        module_name: The entry point's module name, without extension.
        registry_text: The registry file's full text.

    Returns:
        True when the name occurs as a whole word.
    """
    return re.search(rf"\b{re.escape(module_name)}\b", registry_text) is not None


class ResearchRegistrationRule:
    """Guard rule pairing record production with registry membership."""

    name = "research-registration"

    def __init__(self, config: GuardConfig) -> None:
        """Bind the rule to the tree that holds the registry.

        Args:
            config: The guard run's configuration. The registry lives at the
                monorepo root rather than in the package being checked, so the
                rule needs the wider tree the same way ``LiteralSetRule`` does.
        """
        self._monorepo_root = config.monorepo_root

    def run(self, files: list[Path]) -> list[Violation]:
        """Check every entry point that produces a research record.

        Args:
            files: Python source files to check.

        Returns:
            One violation per unregistered producing entry point, or a single
            ``research-registry-missing`` violation when producers exist and
            the registry file does not.
        """
        producers: dict[str, Path] = {}
        for path in files:
            if package_of(path) in _DEFINING_PACKAGES or not _is_entry_point(path):
                continue
            if imported_names(parse_source(path)) & _PRODUCER_SYMBOLS:
                producers[path.stem] = path
        if not producers:
            return []
        registry = self._monorepo_root.joinpath(*REGISTRY_RELATIVE_PATH)
        if not registry.is_file():
            first = sorted(producers.items())[0]
            return [
                Violation(
                    file=first[1],
                    line_no=1,
                    kind="research-registry-missing",
                    line=(
                        f"{len(producers)} entry point(s) build a RunRecord and the research "
                        f"registry at {registry} does not exist; numbers that are compared "
                        f"must be produced by a surface the registry names"
                    ),
                )
            ]
        registry_text = registry.read_text(encoding="utf-8")
        return [
            Violation(
                file=path,
                line_no=1,
                kind="research-entry-point-unregistered",
                line=(
                    f"entry point '{module_name}' builds a RunRecord and is named nowhere in "
                    f"{registry.name}; register it, and name it explicitly rather than "
                    f"extending a list with an ellipsis, which nothing can check"
                ),
            )
            for module_name, path in sorted(producers.items())
            if not _is_registered(module_name, registry_text)
        ]


__all__ = ["REGISTRY_RELATIVE_PATH", "ResearchRegistrationRule"]
