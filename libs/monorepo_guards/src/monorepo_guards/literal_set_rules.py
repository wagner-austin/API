"""Guard rule: an inline ``Literal`` must agree with the tuple that declares it.

WHAT MYPY ALREADY CATCHES, SO THIS DOES NOT. A declaring module binds its
accepted set to a tuple annotated with the same ``Literal``, so adding a member
to the tuple that the annotation does not name is a type error at the tuple
itself.

WHAT IT DOES NOT CATCH, WHICH IS WHY THIS EXISTS. The same set is often spelled
as an inline ``Literal[...]`` on a dozen fields and signatures rather than
through a shared name. Widen the tuple AND its own annotation and every one of
those sites still type-checks -- they are independent annotations, and mypy has
no reason to relate them. A decoder then accepts a value the config cannot
hold, and the failure surfaces as a run doing the wrong thing rather than as a
type error at the site nobody updated.

That is the drift a type alias would have prevented for free. This rule buys
the same guarantee without the alias: every ``Literal`` naming one of these
fields must name exactly what its tuple declares.

WHY IT IS PARAMETERISED. It was written for ``corpus_format`` and its own
docstring observed that ``model_family`` and ``finetuning_strategy`` were
written the same way in the same service. When ``finetuning_strategy`` was
collapsed onto a shared ``StrategyName``, forking a near-identical rule for it
would have been the very duplication both rules exist to prevent, so the
machinery takes its subject as configuration and is registered once per set.

Violations, per configured subject:
- <subject>-literal-drift: a Literal disagrees with the declaring tuple
- <subject>-tuple-missing: the declaring module no longer declares the tuple
  this rule reads, so the rule is checking nothing
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from monorepo_guards import Violation
from monorepo_guards.util import parse_source


class LiteralSet:
    """One set of string constants, and where it is declared.

    Attributes:
        subject: Kebab-case label for this set, used to build the rule's name
            and its violation kinds.
        defining_module: Repo-relative path suffix of the module that binds
            the tuple. Read from source rather than imported, because the
            guard runs against packages that do not depend on the service.
        tuple_name: The name the declaring module binds the set to.
        field_names: Fields, parameters and attributes whose ``Literal`` must
            match the tuple.
        consequence: What goes wrong when the sets disagree, appended to the
            violation message so the reader learns the stake rather than only
            the rule.
    """

    subject: str
    defining_module: str
    tuple_name: str
    field_names: frozenset[str]
    consequence: str

    def __init__(
        self,
        *,
        subject: str,
        defining_module: str,
        tuple_name: str,
        field_names: frozenset[str],
        consequence: str,
    ) -> None:
        """Describe one declared set of string constants.

        Args:
            subject: Kebab-case label for this set.
            defining_module: Path SUFFIX of the declaring module, matched with
                ``endswith``, and it must include the package segment. Two
                services in this monorepo own a
                ``core/contracts/dataset.py`` -- Model-Trainer, which
                declares CORPUS_FORMATS, and Art-Trainer, whose file is
                about LoRA datasets and never had it. Given the bare
                ``core/contracts/dataset.py``, this rule matched
                Art-Trainer's file while checking Art-Trainer, found no
                tuple, and reported that the guard had gone inert -- a
                red gate on a service that was never in scope.
            tuple_name: Name the declaring module binds the set to.
            field_names: Field and parameter names whose Literal must match.
            consequence: What goes wrong when the sets disagree.
        """
        self.subject = subject
        self.defining_module = defining_module
        self.tuple_name = tuple_name
        self.field_names = field_names
        self.consequence = consequence


CORPUS_FORMAT_SET: Final = LiteralSet(
    subject="corpus-format",
    defining_module="model_trainer/core/contracts/dataset.py",
    tuple_name="CORPUS_FORMATS",
    field_names=frozenset({"corpus_format"}),
    consequence=(
        "the decoder and the config would accept different sets, and a run would "
        "train under a reader its config cannot describe"
    ),
)

STRATEGY_NAME_SET: Final = LiteralSet(
    subject="strategy-name",
    defining_module="model_trainer/core/contracts/strategy_names.py",
    tuple_name="STRATEGY_NAMES",
    field_names=frozenset({"finetuning_strategy"}),
    consequence=(
        "the HTTP layer, the queue decoder and the registry would accept different "
        "sets, so a request naming a strategy would be admitted at one layer and "
        "refused at another -- or admitted everywhere and silently dropped from a "
        "checkpoint's metadata, depending on which copy was stale"
    ),
)


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


def _declared_members(tree: ast.Module, tuple_name: str) -> frozenset[str] | None:
    """Read the accepted set out of the declaring module's tuple.

    Args:
        tree: Parsed declaring module.
        tuple_name: Name the module binds the set to.

    Returns:
        The declared members, or None when the tuple is absent or is not a
        tuple of string constants.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not isinstance(target, ast.Name) or target.id != tuple_name:
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


def _annotation_sites(tree: ast.Module, field_names: frozenset[str]) -> list[tuple[int, ast.expr]]:
    """Collect every annotation attached to one of the watched names.

    Covers the three shapes the field appears in: a TypedDict or class-body
    annotation, a function parameter, and a function return whose own name
    says it produces one of these values.

    Args:
        tree: Parsed module to scan.
        field_names: Names whose annotations are watched.

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
            if target_name in field_names:
                sites.append((node.lineno, node.annotation))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            for argument in arguments:
                if argument.arg in field_names and argument.annotation is not None:
                    sites.append((argument.lineno, argument.annotation))
            if node.returns is not None and any(name in node.name for name in field_names):
                sites.append((node.lineno, node.returns))
    return sites


class LiteralSetRule:
    """Guard rule keeping one set's inline Literals in step with its tuple."""

    name: str
    _declared_set: LiteralSet

    def __init__(self, declared_set: LiteralSet) -> None:
        """Bind the rule to one declared set.

        Args:
            declared_set: The set this instance checks.
        """
        self._declared_set = declared_set
        self.name = f"{declared_set.subject}-literal"

    def run(self, files: list[Path]) -> list[Violation]:
        """Compare every watched Literal against the declared tuple.

        Args:
            files: Python source files to check.

        Returns:
            Every disagreement found, in file order. Empty when the package
            being checked does not declare the tuple at all, which is every
            package except the one that owns it.
        """
        declared_set = self._declared_set
        declaring = [f for f in files if f.as_posix().endswith(declared_set.defining_module)]
        if not declaring:
            return []
        declared = _declared_members(parse_source(declaring[0]), declared_set.tuple_name)
        if declared is None:
            return [
                Violation(
                    file=declaring[0],
                    line_no=1,
                    kind=f"{declared_set.subject}-tuple-missing",
                    line=(
                        f"{declared_set.tuple_name} is no longer a tuple of string "
                        f"literals here, so the {declared_set.subject} guard is checking "
                        "nothing; restore it or delete this rule rather than leaving it "
                        "silently inert"
                    ),
                )
            ]
        expected = ", ".join(sorted(declared))
        found: list[Violation] = []
        for path in files:
            for line_no, annotation in _annotation_sites(
                parse_source(path), declared_set.field_names
            ):
                members = _literal_members(annotation)
                if members is None or members == declared:
                    continue
                actual = ", ".join(sorted(members))
                found.append(
                    Violation(
                        file=path,
                        line_no=line_no,
                        kind=f"{declared_set.subject}-literal-drift",
                        line=(
                            f"this Literal names {actual}, but {declared_set.tuple_name} "
                            f"declares {expected}; {declared_set.consequence}"
                        ),
                    )
                )
        return found


__all__ = [
    "CORPUS_FORMAT_SET",
    "STRATEGY_NAME_SET",
    "LiteralSet",
    "LiteralSetRule",
]
