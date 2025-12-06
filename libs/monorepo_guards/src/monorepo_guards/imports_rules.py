from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class ImportsRule:
    name = "imports"

    def _check_import_node(self, path: Path, node: ast.Import) -> list[Violation]:
        violations: list[Violation] = []
        for alias in node.names:
            if alias.name == "pydantic" or alias.name.startswith("pydantic."):
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="import-pydantic",
                        line="",
                    )
                )
            if alias.name == "inspect":
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="import-inspect",
                        line="",
                    )
                )
        return violations

    def _check_import_from_node(self, path: Path, node: ast.ImportFrom) -> list[Violation]:
        violations: list[Violation] = []
        forbidden_typing_imports = {"TYPE_CHECKING", "Iterable", "Iterator"}
        forbidden_collections_imports = {"Iterable", "Iterator"}

        if node.module is None:
            return violations

        if node.module == "pydantic" or node.module.startswith("pydantic."):
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="import-pydantic",
                    line="",
                )
            )

        if node.module == "inspect":
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="import-inspect",
                    line="",
                )
            )

        if node.module == "typing":
            for alias in node.names:
                if alias.name in forbidden_typing_imports:
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind=f"import-typing-{alias.name.lower()}",
                            line="",
                        )
                    )

        if node.module == "collections.abc":
            for alias in node.names:
                if alias.name in forbidden_collections_imports:
                    violations.append(
                        Violation(
                            file=path,
                            line_no=node.lineno,
                            kind=f"import-collections-{alias.name.lower()}",
                            line="",
                        )
                    )

        return violations

    def _check_attribute_node(self, path: Path, node: ast.Attribute) -> list[Violation]:
        violations: list[Violation] = []
        forbidden_typing_imports = {"TYPE_CHECKING", "Iterable", "Iterator"}
        forbidden_collections_imports = {"Iterable", "Iterator"}

        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "typing"
            and node.attr in forbidden_typing_imports
        ):
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind=f"typing-{node.attr.lower()}-usage",
                    line="",
                )
            )

        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "collections"
            and node.value.attr == "abc"
            and node.attr in forbidden_collections_imports
        ):
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind=f"collections-abc-{node.attr.lower()}-usage",
                    line="",
                )
            )

        return violations

    def _check_ast(self, path: Path, tree: ast.AST) -> list[Violation]:
        violations: list[Violation] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(self._check_import_node(path, node))
            elif isinstance(node, ast.ImportFrom):
                violations.extend(self._check_import_from_node(path, node))
            elif isinstance(node, ast.Attribute):
                violations.extend(self._check_attribute_node(path, node))

        return violations

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            lines = read_lines(path)
            source = "\n".join(lines)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc
            out.extend(self._check_ast(path, tree))
        return out


__all__ = ["ImportsRule"]
