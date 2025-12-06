from __future__ import annotations

import ast
import tokenize
from io import StringIO
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_lines


class SuppressRule:
    name = "suppress"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            lines = read_lines(path)
            if not lines:
                continue
            source = "\n".join(lines)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise RuntimeError(f"failed to parse {path}: {exc}") from exc
            out.extend(self._scan_tree(path, tree, lines))
            out.extend(self._scan_pragma_comments(path, source))
        return out

    def _scan_tree(self, path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
        def is_suppress(expr: ast.AST) -> bool:
            func = expr.func if isinstance(expr, ast.Call) else expr
            if isinstance(func, ast.Attribute):
                is_contextlib = isinstance(func.value, ast.Name) and func.value.id == "contextlib"
                return is_contextlib and func.attr == "suppress"
            return isinstance(func, ast.Name) and func.id == "suppress"

        out: list[Violation] = []
        seen: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if is_suppress(item.context_expr):
                        line_no = item.context_expr.lineno
                        if line_no in seen:
                            continue
                        seen.add(line_no)
                        idx = line_no - 1
                        text = lines[idx] if 0 <= idx < len(lines) else ""
                        out.append(
                            Violation(
                                file=path,
                                line_no=line_no,
                                kind="suppress-usage",
                                line=text.rstrip("\n"),
                            )
                        )
        return out

    def _scan_pragma_comments(self, path: Path, source: str) -> list[Violation]:
        violations: list[Violation] = []
        reader = StringIO(source).readline
        tokens = tokenize.generate_tokens(reader)
        for tok in tokens:
            if tok.type == tokenize.COMMENT and "pragma" in tok.string:
                violations.append(
                    Violation(
                        file=path,
                        line_no=tok.start[0],
                        kind="pragma-comment",
                        line=tok.line.rstrip("\n"),
                    )
                )
        return violations


__all__ = ["SuppressRule"]
