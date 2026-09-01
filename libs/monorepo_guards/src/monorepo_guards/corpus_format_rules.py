"""Guard rule: the corpus-format tuple and its Literal annotations agree.

WHAT MYPY ALREADY CATCHES, SO THIS DOES NOT. ``CORPUS_FORMATS`` is annotated
``tuple[Literal["lines", "documents"], ...]``, so adding a member to the tuple
that the annotation does not name is a type error at the tuple itself.

WHAT IT DOES NOT CATCH, WHICH IS WHY THIS EXISTS. The format is spelled as an
inline ``Literal[...]`` on roughly a dozen fields and signatures, matching how
``model_family`` and ``finetuning_strategy`` are written in the same service
rather than aliasing a name. Widen the tuple AND its own annotation to add a
third format and every one of those dozen sites still type-checks -- they are
independent annotations, and mypy has no reason to relate them. The decoder
would then accept a format that ``ModelTrainConfig`` cannot hold, and the
failure surfaces as a run trained under the wrong reader rather than as a type
error at the site that was not updated.

That is the drift a type alias would have prevented for free. This rule buys
the same guarantee without the alias: every ``Literal`` that names a corpus
format must name exactly the formats the tuple declares.

Violations:
- corpus-format-literal-drift: a corpus_format Literal disagrees with
  CORPUS_FORMATS
- corpus-format-tuple-missing: the declaring module no longer declares the
  tuple this rule reads, so the rule is checking nothing
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from monorepo_guards import Violation
from monorepo_guards.util import parse_source

#: Where the accepted set is declared. Read from source rather than imported,
#: because the guard runs against packages that do not depend on the service.
_DEFINING_MODULE: Final = "core/contracts/dataset.py"

#: The name the declaring module binds the accepted set to.
_TUPLE_NAME: Final = "CORPUS_FORMATS"

#: Fields, parameters and attributes whose Literal must match the tuple.
_FORMAT_NAMES: Final = frozenset({"corpus_format"})


def _literal_members(node: ast.expr) -> frozenset[str] | None:
    """Read the string members of a ``Literal[...]`` subscript.

    Args:
        node: Annotation expression to inspect.

    Returns:
        The literal's string members, or None when the annotation is not a
        ``Literal`` of strings.
    """
    if not isinstance(node, ast.Subscript):
        return None
    base = node.value
    if isinstance(base, ast.Attribute):
        base_name = base.attr
    elif isinstance(base, ast.Name):
        base_name = base.id
    else:
        return None
    if base_name != "Literal":
        return None
    sliced = node.slice
    elements = sliced.elts if isinstance(sliced, ast.Tuple) else [sliced]
    members: set[str] = set()
    for element in elements:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        members.add(element.value)
    return frozenset(members)


def _declared_formats(tree: ast.Module) -> frozenset[str] | None:
    """Read the accepted set out of the declaring module's tuple.

    Args:
        tree: Parsed declaring module.

    Returns:
        The declared formats, or None when the tuple is absent.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not isinstance(target, ast.Name) or target.id != _TUPLE_NAME:
            continue
        value = node.value
        if not isinstance(value, ast.Tuple):
            return None
        members: set[str] = set()
        for element in value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return None
            members.add(element.value)
        return frozenset(members)
    return None


def _annotation_sites(tree: ast.Module) -> list[tuple[int, ast.expr]]:
    """Collect every annotation attached to a corpus-format name.

    Covers the three shapes the field appears in: a TypedDict or class-body
    annotation, a function parameter, and a function return whose own name
    says it produces a format.

    Args:
        tree: Parsed module to scan.

    Returns:
        Pairs of (line number, annotation expression).
    """
    sites: list[tuple[int, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, ast.Attribute):
                target_name = target.attr
            else:
                target_name = ""
            if target_name in _FORMAT_NAMES:
                sites.append((node.lineno, node.annotation))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            for argument in arguments:
                if argument.arg in _FORMAT_NAMES and argument.annotation is not None:
                    sites.append((argument.lineno, argument.annotation))
            if node.returns is not None and "corpus_format" in node.name:
                sites.append((node.lineno, node.returns))
    return sites


class CorpusFormatLiteralRule:
    """Guard rule keeping corpus-format Literals in step with the tuple."""

    name = "corpus-format-literal"

    def run(self, files: list[Path]) -> list[Violation]:
        """Compare every corpus-format Literal against the declared tuple.

        Args:
            files: Python source files to check.

        Returns:
            Every disagreement found, in file order. Empty when the package
            being checked does not declare the tuple at all, which is every
            package except the one that owns it.
        """
        declaring = [f for f in files if f.as_posix().endswith(_DEFINING_MODULE)]
        if not declaring:
            return []
        declared = _declared_formats(parse_source(declaring[0]))
        if declared is None:
            return [
                Violation(
                    file=declaring[0],
                    line_no=1,
                    kind="corpus-format-tuple-missing",
                    line=(
                        f"{_TUPLE_NAME} is no longer a tuple of string literals here, so "
                        "the corpus-format guard is checking nothing; restore it or "
                        "delete this rule rather than leaving it silently inert"
                    ),
                )
            ]
        expected = ", ".join(sorted(declared))
        found: list[Violation] = []
        for path in files:
            for line_no, annotation in _annotation_sites(parse_source(path)):
                members = _literal_members(annotation)
                if members is None or members == declared:
                    continue
                actual = ", ".join(sorted(members))
                found.append(
                    Violation(
                        file=path,
                        line_no=line_no,
                        kind="corpus-format-literal-drift",
                        line=(
                            f"this corpus_format Literal names {actual}, but "
                            f"{_TUPLE_NAME} declares {expected}; the decoder and the "
                            "config would accept different sets, and a run would train "
                            "under a reader its config cannot describe"
                        ),
                    )
                )
        return found


__all__ = ["CorpusFormatLiteralRule"]
