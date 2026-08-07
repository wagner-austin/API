"""Wire primitives: the byte layer every codec sits on.

A true leaf -- nothing here imports any other ``tankpit_bot`` package,
which is the point. Both :mod:`tankpit_bot.protocol` and
:mod:`tankpit_bot.container` encode and decode the same wire, and both
need the same byte arithmetic and length validation. While those lived
in ``protocol/helpers.py`` the container package had to import from
protocol, and protocol imports container back (container messages are
a message family decoded from inside ``protocol/decoders/tank.py``) --
a package cycle that ``protocol/decoders/tank.py`` was papering over
with a function-level import.

Submodules import from here directly; this file stays a docstring so
the package cannot become a second import surface.
"""

from __future__ import annotations
