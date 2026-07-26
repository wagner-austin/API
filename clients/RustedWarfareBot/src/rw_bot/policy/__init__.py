"""What the bot decides to do.

Everything here is a pure function of observed state. Transport is
``rw_bot.control``; the wire format is ``rw_bot.wire``; unit costs and
mobility come from ``rw_bot.mechanics``.
"""

from __future__ import annotations
