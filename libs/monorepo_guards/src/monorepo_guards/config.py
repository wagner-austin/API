from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class GuardConfig(NamedTuple):
    root: Path
    directories: tuple[str, ...]
    exclude_parts: tuple[str, ...]
    forbid_pyi: bool
    allow_print_in_tests: bool
    dataclass_ban_segments: tuple[tuple[str, ...], ...]


__all__ = ["GuardConfig"]
