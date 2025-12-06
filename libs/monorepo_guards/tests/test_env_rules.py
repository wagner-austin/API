from __future__ import annotations

from pathlib import Path

from monorepo_guards.env_rules import EnvRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_env_rule_flags_os_and_tomllib(tmp_path: Path) -> None:
    os_file = tmp_path / "src" / "uses_env.py"
    os_from_file = tmp_path / "src" / "uses_env_from.py"
    toml_file = tmp_path / "src" / "uses_toml.py"
    toml_from_file = tmp_path / "src" / "uses_toml_from.py"
    toml_dyn_file = tmp_path / "src" / "uses_toml_dyn.py"
    os_safe_file = tmp_path / "src" / "uses_env_safe.py"
    _write(os_file, "import os\nx = os.getenv('X')\n")
    _write(os_from_file, "from os import getenv\nx = getenv('Y')\n")
    _write(os_safe_file, "from os import path\np = path\n")
    _write(toml_file, "import tomllib\ndata = tomllib.loads('a=1')\n")
    _write(toml_from_file, "from tomllib import loads\ndata = loads('a=1')\n")
    _write(toml_dyn_file, "mod = __import__('tomllib')\n")

    rule = EnvRule()
    violations = rule.run(
        [os_file, os_from_file, os_safe_file, toml_file, toml_from_file, toml_dyn_file]
    )
    kinds = {v.kind for v in violations}
    assert "env-access-banned" in kinds
    assert "tomllib-banned" in kinds


def test_env_rule_allows_builtins(tmp_path: Path) -> None:
    allowed_core = tmp_path / "src" / "platform_core" / "config" / "_utils.py"
    allowed_guard = tmp_path / "src" / "monorepo_guards" / "config_loader.py"
    allowed_test = tmp_path / "tests" / "test_config_loader.py"
    _write(allowed_core, "import os\nx = os.getenv('X')\n")
    _write(allowed_guard, "import tomllib\nx = tomllib.loads('a=1')\n")
    _write(allowed_test, "import tomllib\nx = tomllib.loads('a=1')\n")

    rule = EnvRule()
    violations = rule.run([allowed_core, allowed_guard, allowed_test])
    assert violations == []
