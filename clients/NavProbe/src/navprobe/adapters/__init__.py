"""Driven adapters converting a concrete simulator to the probe's port.

This is the only layer that imports a simulator. Everything above it depends on
:class:`navprobe.rollout.SimulatorProtocol` and never on a vendor, which is what
lets the instrument be exercised end to end without one.

An adapter's job is narrow and worth stating: match the vendor signature
exactly, flatten whatever the vendor returns into floats in a stable order, and
translate the vendor's failures into this package's error codes. It performs no
comparison and reaches no verdict — those belong to layers that have no idea
which simulator produced the numbers.
"""

from __future__ import annotations

__all__: list[str] = []
