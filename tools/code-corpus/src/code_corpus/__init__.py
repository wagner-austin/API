"""Emit training corpora from the monorepo's own source code.

The package walks git repositories, selects tracked source files, and writes a
JSONL corpus plus a provenance manifest, so a training run's inputs can always
be reconstructed from what was recorded about them.
"""
