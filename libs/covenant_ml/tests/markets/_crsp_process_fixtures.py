"""Executable database-client boundary fixtures shared by extraction and refresh tests."""

from __future__ import annotations

import os
from pathlib import Path

from platform_core.config import config_test_hooks


def client_environment(root: Path, body: str) -> dict[str, str]:
    """Create a real executable client first on the child process's PATH."""
    directory = root / "bin"
    directory.mkdir()
    client = directory / "psql"
    client.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    client.chmod(0o755)
    environment = dict(config_test_hooks.get_environment())
    environment["PATH"] = f"{directory}{os.pathsep}{environment['PATH']}"
    return environment


__all__ = ["client_environment"]
