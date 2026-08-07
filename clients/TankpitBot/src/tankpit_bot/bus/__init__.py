"""Cross-thread session buses and the status contract they carry.

Three primitives let a running session publish to observers without
knowing whether any observer exists: a mode bridge the SPA writes
into, a status bus the tick loop publishes onto, and a frame bus the
screencast relay feeds. A standalone ``make bot`` session gets inert
instances with zero subscribers; the HTTP service injects shared ones.

They live below both ``bot`` and ``service`` on purpose. While they sat
inside ``service`` the tick loop had to import the HTTP package to run
at all, and ``service`` imported ``bot`` back for the mode vocabulary --
a cycle whose real shape was just three misfiled files.

Submodules are imported directly; this file stays a docstring so the
package cannot become a second import surface.
"""

from __future__ import annotations
