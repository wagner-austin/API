"""Hyperparameter optimization CLI package.

DELIBERATELY EMPTY OF RE-EXPORTS, and that is a correctness requirement
rather than a style preference.

``python -m scripts.optimize`` imports THIS module before it imports
``__main__``. While it re-exported twenty symbols it transitively imported
``runner``, which imports ``covenant_ml``, which imports numpy -- and the BLAS
thread variables are read once, when numpy loads. So any pin written in
``__main__`` was written after the only moment it could take effect, and
:func:`~platform_core.determinism_cpu.apply_cpu_determinism` would have
refused rather than report a posture the process does not have.

That is why this package pinned nothing while every benchmark entry point
beside it did: not an oversight in ``__main__``, but a package whose import
made the fix unreachable from there. The same medicine was applied to
``platform_core.__init__`` for the same reason, where it took the import of
one contract module from 466 modules to 102.

Import from the submodule that owns the name -- ``scripts.optimize.main``,
``scripts.optimize.runner`` -- which is also how a reader learns where a
symbol lives.

Modules:
- cli: Argument parsing and types
- display: Rich console output formatting
- history: Run history tracking and comparison
- logging_config: Logging setup and suppression
- main: Main entry point function
- modes: Run modes (single, compare, all-datasets)
- runner: Core optimization execution
- state: Lifecycle state management
"""

from __future__ import annotations

__all__: list[str] = []
