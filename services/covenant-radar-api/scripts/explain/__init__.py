"""Feature importance explanation CLI package.

Modular structure:
- cli: Argument parsing and types
- display: Rich console output formatting
- main: Main entry point function
- runner: Core explanation execution
"""

from __future__ import annotations

from scripts.explain.cli import (
    ExplainArgs,
    parse_args,
)
from scripts.explain.display import (
    print_config,
    print_result,
)
from scripts.explain.main import main
from scripts.explain.runner import (
    ExplainRunResult,
    get_project_root,
    run_explanation,
)

__all__ = [
    "ExplainArgs",
    "ExplainRunResult",
    "get_project_root",
    "main",
    "parse_args",
    "print_config",
    "print_result",
    "run_explanation",
]
