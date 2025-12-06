from __future__ import annotations

from pathlib import Path

from monorepo_guards.orchestrator import run_for_project


def test_run_for_project_with_config_file(tmp_path: Path) -> None:
    # Create a monorepo root with config
    monorepo_root = tmp_path / "repo"
    monorepo_root.mkdir()
    config_file = monorepo_root / "monorepo-guards.toml"
    config_file.write_text(
        """
[guards]
directories = ["src"]
exclude_parts = [".venv"]
forbid_pyi = true
allow_print_in_tests = false
dataclass_ban_segments = []
""",
        encoding="utf-8",
    )

    # Create a project
    project_root = monorepo_root / "services" / "test-service"
    project_root.mkdir(parents=True)
    src_dir = project_root / "src"
    src_dir.mkdir()

    # Create a clean Python file
    py_file = src_dir / "clean.py"
    py_file.write_text("x: int = 1\n", encoding="utf-8")

    # Run guards
    rc = run_for_project(monorepo_root, project_root)

    assert rc == 0
