"""Typed handle on the viewport-probe module for patching.

``tests/action_lab/conftest.py`` and the viewport-probe test modules
both swap attributes on this module. The Protocol gives those swaps a
type; importing the module through ``__import__`` and annotating the
result is what lets mypy check an attribute the module itself only
re-exports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.viewport_probe import ViewportProbe


class _ViewportModuleProtocol(Protocol):
    ViewportProbe: type[ViewportProbe]
    get_terrain_map: Callable[[], TerrainMapProtocol | None]


_viewport_module_import = __import__(
    "tankpit_bot.action_lab.viewport_probe",
    fromlist=["viewport_probe"],
)
viewport_module: _ViewportModuleProtocol = _viewport_module_import
