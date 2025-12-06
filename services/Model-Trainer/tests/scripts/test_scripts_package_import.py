from __future__ import annotations

import importlib


def test_import_scripts_package() -> None:
    mod = importlib.import_module("scripts")
    # Ensure module loaded and has expected name
    assert mod.__name__ == "scripts", f"Expected module name to be 'scripts', got {mod.__name__}"
