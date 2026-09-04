"""Guard rule banning a type bound to a second spelling of its own name.

    # Retained so the old spelling still resolves.
    GoogleTokenResponse = OAuthTokenResponse

The workspace already bans ``X: TypeAlias = ...`` (``typing_rules``). This is
the same defect written without the annotation, which is exactly why it
survived: that rule reads the annotation, and here there is none. Three of the
five compatibility aliases deleted on 2026-08-26 were this shape, and each was
found by reading the comment its author left on it -- not a search strategy
that scales, which is why the rule exists.

What it costs to leave in: a reader who greps for ``OAuthTokenResponse``
finds half its uses. A second name does not divide the work, it divides the
evidence.

MEASURED BEFORE BEING BELIEVED, because a guard's first number is a claim
about the guard and not only about the tree. Against 4896 files it reported
27 findings and every one was a real second spelling of an existing type --
no false positives, in five services plus two clients.

The predicate is structural, not a list of blessed names:

- Both sides must be spelled like types (``GenerateRequest``, not
  ``MAX_RETRIES``). A constant bound to another constant is a duplicated
  value: a different problem with a different fix.
- The right-hand side must be a name THIS module binds by import or class
  definition. Requiring that keeps the rule off assignments whose value
  merely happens to be capitalised.

A SECOND check, for functions whose whole body is ``return f(same args)``,
was written and measured alongside this one and deliberately not kept. It
reported 189 findings, and after excluding factories (``create_x_backend``
returning ``XBackend(...)``) and Protocol methods (``get_default_search_space``
returning ``make_lightgbm_default_space(...)``) it still reported 95, of which
most were legitimate under the workspace's own rule that a wrapper may exist
for type narrowing or boundary translation -- ``encode_job_event(JobEvent) ->
str`` calling ``dump_json_str`` is narrowing, not renaming. Telling those
apart needs the callee's signature, which is usually in another file and out
of reach of a single-file AST pass. A check that needs a growing exclusion
list to stay quiet is the wrong predicate, so it is not here rather than here
and muted.
"""

from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import parse_source


def _is_type_spelled(name: str) -> bool:
    """Report whether a name is spelled like a type rather than a constant.

    Args:
        name: Identifier to classify.

    Returns:
        True for ``GenerateRequest``, False for ``MAX_RETRIES`` and ``count``.
    """
    return name[:1].isupper() and any(char.islower() for char in name)


class PassthroughRule:
    """Ban module-level bindings that only rename an existing type."""

    name = "passthrough"

    def _is_src_file(self, path: Path) -> bool:
        """Report whether a path is production source.

        Args:
            path: File being considered.

        Returns:
            True for files under a ``src`` directory. A local name inside a
            test is a fixture; shipping a second public name for one type is
            what this rule is for.
        """
        return "src" in path.parts

    def _bound_type_names(self, tree: ast.Module) -> set[str]:
        """Collect names this module binds by import or class definition.

        Args:
            tree: Parsed module.

        Returns:
            The names an alias could point at.
        """
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname if alias.asname else alias.name)
        return names

    def _check_alias(
        self, path: Path, statement: ast.Assign, type_names: set[str]
    ) -> list[Violation]:
        """Check one module-level assignment for the alias shape.

        Args:
            path: Source file.
            statement: Assignment to inspect.
            type_names: Names bound by an import or class definition here.

        Returns:
            One violation if the assignment renames a type, else none.
        """
        if len(statement.targets) != 1:
            return []
        target = statement.targets[0]
        value = statement.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Name):
            return []
        if not _is_type_spelled(target.id) or not _is_type_spelled(value.id):
            return []
        if value.id not in type_names:
            return []

        return [
            Violation(
                file=path,
                line_no=statement.lineno,
                kind="passthrough-alias",
                line=f"{target.id} = {value.id} -- use {value.id} at the call sites",
            )
        ]

    def run(self, files: list[Path]) -> list[Violation]:
        """Check every production source file.

        Args:
            files: Files to check.

        Returns:
            Every renaming alias found.
        """
        out: list[Violation] = []
        for path in files:
            if not self._is_src_file(path):
                continue
            # Read and parse without a guard: a source file that cannot be
            # read or parsed is a real problem in the tree being checked, and
            # a syntax error must not silently exempt a file from this rule.
            tree = parse_source(path)
            type_names = self._bound_type_names(tree)
            for statement in tree.body:
                if isinstance(statement, ast.Assign):
                    out.extend(self._check_alias(path, statement, type_names))
        return out


__all__ = ["PassthroughRule"]
