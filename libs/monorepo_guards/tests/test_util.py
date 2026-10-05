from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.util import iter_py_files, module_nodes, parse_source


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_iter_py_files_excludes_cache_and_handles_missing_dirs(tmp_path: Path) -> None:
    root = tmp_path
    incl = root / "src"
    excl = incl / ".pytest_cache" / "skip.py"
    keep = incl / "keep.py"
    _write(excl, "x=1\n")
    _write(keep, "y=2\n")

    cfg = GuardConfig(
        root=root,
        monorepo_root=root,
        directories=("src", "missing"),  # 'missing' directory does not exist
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )
    files = iter_py_files(cfg)
    # Only 'keep.py' should be included
    assert [p.name for p in files] == ["keep.py"]


def test_module_nodes_walks_a_parsed_file_once_in_ast_walk_order(tmp_path: Path) -> None:
    """The first call walks and the second hands back the same tuple, in
    exactly the order ``ast.walk`` yields, so a rule switched to it sees
    the nodes it saw before and the module is walked once."""
    path = tmp_path / "mod.py"
    _write(path, "import os\n\ndef f(x: int) -> int:\n    return x + 1\n")
    tree = parse_source(path)
    first = module_nodes(tree)
    walked = list(ast.walk(tree))
    assert len(first) == len(walked) == 16
    assert all(mine is theirs for mine, theirs in zip(first, walked, strict=True))
    assert module_nodes(tree) is first
