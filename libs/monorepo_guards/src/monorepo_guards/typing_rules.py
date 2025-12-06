from __future__ import annotations

import ast
import tokenize
from collections.abc import Generator
from io import StringIO
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class TypingRule:
    name = "typing"

    def _iter_tokens(self, text: str) -> Generator[tokenize.TokenInfo, None, None]:
        reader = StringIO(text).readline
        yield from tokenize.generate_tokens(reader)

    def _contains_object_in_annotation(self, node: ast.AST) -> bool:
        return any(isinstance(child, ast.Name) and child.id == "object" for child in ast.walk(node))

    def _contains_unknown_json(self, node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id == "UnknownJson" for child in ast.walk(node)
        )

    def _check_object_annotations(self, path: Path, node: ast.AST) -> list[Violation]:
        violations: list[Violation] = []
        if (
            isinstance(node, ast.AnnAssign)
            and node.annotation is not None
            and self._contains_object_in_annotation(node.annotation)
        ):
            violations.append(
                Violation(file=path, line_no=node.lineno, kind="object-in-annotation", line="")
            )
        if (
            isinstance(node, ast.arg)
            and node.annotation is not None
            and self._contains_object_in_annotation(node.annotation)
        ):
            violations.append(
                Violation(file=path, line_no=node.lineno, kind="object-in-annotation", line="")
            )
        # Check both sync and async function return types
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.returns is not None
            and self._contains_object_in_annotation(node.returns)
        ):
            violations.append(
                Violation(file=path, line_no=node.lineno, kind="object-in-annotation", line="")
            )
        return violations

    def _is_unknown_json_allowed_function(self, func_name: str) -> bool:
        """Check if function is allowed to use UnknownJson (internal helpers only)."""
        allowed_prefixes = ("_load_json", "_decode", "_attach")
        return any(func_name.startswith(prefix) for prefix in allowed_prefixes)

    def _is_typealias_annotation(self, node: ast.expr) -> bool:
        """Check if annotation indicates a TypeAlias definition."""
        return (isinstance(node, ast.Name) and node.id == "TypeAlias") or (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "TypeAlias"
        )

    def _check_unknown_json_function(
        self, path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[Violation]:
        """Check UnknownJson usage in function signatures (both sync and async)."""
        violations: list[Violation] = []
        if self._is_unknown_json_allowed_function(node.name):
            return violations

        # Check return type
        if node.returns is not None and self._contains_unknown_json(node.returns):
            violations.append(
                Violation(file=path, line_no=node.lineno, kind="unknownjson-public-return", line="")
            )

        # Check parameters
        for arg in node.args.args:
            if arg.annotation is not None and self._contains_unknown_json(arg.annotation):
                violations.append(
                    Violation(
                        file=path, line_no=arg.lineno, kind="unknownjson-public-param", line=""
                    )
                )

        return violations

    def _check_unknown_json_misuse(
        self, path: Path, node: ast.AST, module_level_nodes: set[int]
    ) -> list[Violation]:
        violations: list[Violation] = []

        # Check function signatures (both sync and async)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.extend(self._check_unknown_json_function(path, node))

        # Check class attributes
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and item.annotation is not None
                    and self._contains_unknown_json(item.annotation)
                ):
                    violations.append(
                        Violation(
                            file=path, line_no=item.lineno, kind="unknownjson-class-attr", line=""
                        )
                    )

        # Check module-level variables (except TypeAlias definitions and private vars)
        if (
            isinstance(node, ast.AnnAssign)
            and id(node) in module_level_nodes
            and node.annotation is not None
            and self._contains_unknown_json(node.annotation)
            and isinstance(node.target, ast.Name)
            and node.target.id != "UnknownJson"
            and not node.target.id.startswith("_")  # Allow private module-level constants
            and not self._is_typealias_annotation(node.annotation)
        ):
            violations.append(
                Violation(file=path, line_no=node.lineno, kind="unknownjson-module-var", line="")
            )

        return violations

    def _check_typealias_usage(self, path: Path, node: ast.AnnAssign) -> list[Violation]:
        """Check for forbidden TypeAlias usage (all TypeAlias is banned)."""
        violations: list[Violation] = []

        if (
            node.annotation is not None
            and self._is_typealias_annotation(node.annotation)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            violations.append(
                Violation(
                    file=path,
                    line_no=node.lineno,
                    kind="typealias-forbidden",
                    line="",
                )
            )

        return violations

    def _check_ast(self, path: Path, tree: ast.AST) -> list[Violation]:
        violations: list[Violation] = []
        forbidden_imports = {"Any", "cast", "TypeAlias"}

        # Get module-level statements for UnknownJson checks
        # ast.parse() always returns ast.Module, so we can safely assert this
        assert isinstance(tree, ast.Module)
        module_level_nodes: set[int] = set()
        for stmt in tree.body:
            module_level_nodes.add(id(stmt))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                for alias in node.names:
                    if alias.name in forbidden_imports:
                        violations.append(
                            Violation(
                                file=path,
                                line_no=node.lineno,
                                kind=f"typing-import-{alias.name.lower()}",
                                line="",
                            )
                        )
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "typing"
                and node.attr in forbidden_imports
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
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "cast"
            ):
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="cast-call",
                        line="",
                    )
                )
            if isinstance(node, ast.Name) and node.id == "Any":
                violations.append(
                    Violation(
                        file=path,
                        line_no=node.lineno,
                        kind="any-usage",
                        line="",
                    )
                )
            violations.extend(self._check_object_annotations(path, node))
            violations.extend(self._check_unknown_json_misuse(path, node, module_level_nodes))
            if isinstance(node, ast.AnnAssign):
                violations.extend(self._check_typealias_usage(path, node))
        return violations

    def _check_comments(self, path: Path, text: str) -> list[Violation]:
        violations: list[Violation] = []
        for tok in self._iter_tokens(text):
            if tok.type == tokenize.COMMENT and "type: ignore" in tok.string:
                violations.append(
                    Violation(
                        file=path,
                        line_no=tok.start[0],
                        kind="type-ignore",
                        line=tok.line.rstrip("\n"),
                    )
                )
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
            out.extend(self._check_comments(path, source))
        return out


__all__ = ["TypingRule"]
