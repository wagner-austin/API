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

from pathlib import Path
from string import ascii_letters, digits
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

#: What a qualified command name may be spelled with. Read directly rather
#: than through a regular expression: this package sets ``disallow_any_expr``,
#: and ``re.findall`` is typed ``list[Any]``, which no amount of annotation at
#: the call site removes.
_NAME_CHARACTERS: Final[frozenset[str]] = frozenset(ascii_letters + digits + "_.")

#: Returned when a token opens no brace group. It is ``-1`` so that an
#: UNCLOSED group -- where ``str.find`` also answers ``-1`` -- takes the same
#: path as no group at all, rather than needing a branch that says the same
#: thing twice.
_NO_BRACE_GROUP: Final[int] = -1


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


def _qualified_name(path: Path) -> str:
    """Name an entry point the way the registry names it.

    Args:
        path: The entry point's file.

    Returns:
        The dotted import path from the package root, e.g.
        ``model_trainer.cli.gemm_probe``, taking everything after the ``src``
        marker. A file outside that layout yields its parent and stem, which
        is the most qualification its path carries.
    """
    parts = path.with_suffix("").as_posix().split("/")
    if "src" in parts:
        return ".".join(parts[parts.index("src") + 1 :])
    return ".".join(parts[-2:])


def _closing_brace(registry_text: str, index: int, token: str) -> int:
    """Locate the brace group a qualified prefix opens, if it opens one.

    Args:
        registry_text: The registry file's full text.
        index: Offset just past ``token``.
        token: The name-characters run that was just read.

    Returns:
        The offset of the ``}`` closing a group this token introduces, or
        :data:`_NO_BRACE_GROUP` when the token introduces no group -- it is
        not followed by ``{``, does not end in the ``.`` that makes it a
        prefix, or opens a group nothing closes.
    """
    if index >= len(registry_text) or registry_text[index] != "{" or not token.endswith("."):
        return _NO_BRACE_GROUP
    return registry_text.find("}", index)


def _registered_names(registry_text: str) -> frozenset[str]:
    """Read every command the registry names, expanding its brace notation.

    THE BARE MODULE STEM IS NOT USABLE HERE, and this is the bug this
    function exists to close. An earlier version matched the stem on a word
    boundary, which passed ``code_style_eval.cli.compare`` because the word
    "compare" appears twice in the registry as ordinary English, and would
    have passed ``scripts.batch`` on the word "batch". A check that a module
    can satisfy by sharing a spelling with prose is one that reads as
    verified while having looked at nothing -- the same failure the
    run-record rule records for collecting ``Name`` nodes.

    So a command counts as named only when its QUALIFIED path appears. The
    registry writes those two ways: in full, and grouped as
    ``pkg.cli.{a, b, c}``, which is the house notation this expands.

    Args:
        registry_text: The registry file's full text.

    Returns:
        Every qualified command name the registry declares.
    """
    names: set[str] = set()
    index = 0
    while index < len(registry_text):
        if registry_text[index] not in _NAME_CHARACTERS:
            index += 1
            continue
        start = index
        while index < len(registry_text) and registry_text[index] in _NAME_CHARACTERS:
            index += 1
        token = registry_text[start:index]
        closing = _closing_brace(registry_text, index, token)
        if closing == _NO_BRACE_GROUP:
            if "." in token:
                names.add(token.strip("."))
            continue
        names.update(
            token + member.strip()
            for member in registry_text[index + 1 : closing].split(",")
            if member.strip()
        )
        index = closing + 1
    return frozenset(names)


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
                producers[_qualified_name(path)] = path
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
        registered = _registered_names(registry.read_text(encoding="utf-8"))
        return [
            Violation(
                file=path,
                line_no=1,
                kind="research-entry-point-unregistered",
                line=(
                    f"entry point '{qualified}' builds a RunRecord and is named nowhere in "
                    f"{registry.name}; name it by its qualified path, explicitly rather than "
                    f"behind an ellipsis, which nothing can check"
                ),
            )
            for qualified, path in sorted(producers.items())
            if qualified not in registered
        ]


__all__ = ["REGISTRY_RELATIVE_PATH", "ResearchRegistrationRule"]
