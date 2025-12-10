from __future__ import annotations

from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.dataclass_rules import DataclassRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_dataclass_rule_flags_banned_paths(tmp_path: Path) -> None:
    root = tmp_path
    target = root / "server" / "model_trainer" / "core" / "contracts" / "types.py"
    code = "from dataclasses import dataclass\n@dataclass\nclass T:\n    x: int\n"
    _write(target, code)

    cfg = GuardConfig(
        root=root,
        directories=("server/model_trainer",),
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=False,
        allow_print_in_tests=False,
        dataclass_ban_segments=(("model_trainer", "core", "contracts"),),
    )
    rule = DataclassRule(cfg)
    violations = rule.run([target])
    kinds = {v.kind for v in violations}
    assert "dataclass-decorator-forbidden" in kinds
    assert "dataclass-import-forbidden" in kinds


def test_dataclass_rule_non_banned_path_noop(tmp_path: Path) -> None:
    root = tmp_path
    target = root / "feature" / "module" / "types.py"
    code = "from dataclasses import dataclass\n@dataclass\nclass T:\n    x: int\n"
    _write(target, code)

    cfg = GuardConfig(
        root=root,
        directories=("feature",),
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=False,
        allow_print_in_tests=False,
        dataclass_ban_segments=(("model_trainer", "core", "contracts"),),
    )
    rule = DataclassRule(cfg)
    violations = rule.run([target])
    assert violations == []
