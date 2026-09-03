"""Data files this distribution carries, addressed as package resources.

A package rather than a bare directory because
:func:`importlib.resources.files` addresses an importable package, and
addressing the data that way is what makes it travel with the wheel.
:mod:`tankpit_bot.resources` is the only module that reads anything here;
nothing imports this package for its own sake.

WHAT LIVES HERE AND WHY IT MOVED. The static XOR key and the field minimap
GIFs used to sit at the repository root, outside ``src/``, and were reached
by a path four parents above the module (the repo root in a checkout,
site-packages after an install) and by bare CWD-relative filenames. Neither
survives ``pip install``, so every consumer rebuilt the data environment by
hand: the fleet container copied the key in and named it by environment
variable, and a cluster job staged forty-six files beside itself and passed
the same variable per run. Both were working around a distribution that did
not carry its own data ([[packaged-data-assets]]).
"""

from __future__ import annotations

__all__: list[str] = []
