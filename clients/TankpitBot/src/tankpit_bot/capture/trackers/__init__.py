"""Message trackers for WebSocket capture decoding.

A tracker turns one base64 payload into a human-readable line, or ``None``
when the payload is not its business. That shape is what a capture reader
wants -- it was the January 2026 reverse-engineering toolkit, written to
answer "what is this game even sending?" by eye.

It is not what the bot wants. ``sniffer/world_state_dispatch`` (April 2026)
produces structured ``WorldStateDict`` updates, which is what the AI layer
consumes, and it superseded every tracker except one. The other eleven
survived only because this package re-exported them, which made an
unreferenced class look referenced -- the 2026-08-07 shim sweep
(``0ee86133``) walked straight past them for that reason. They were deleted
2026-08-10; the decoding they encoded lives on in the live path and the
protocol pages.

``MineTracker`` is the only member left, and it survives for a narrower
reason than "the bot needs it" -- it does not. ``sniffer/core.py`` arms it
on the session magic and pipes its line straight to ``log.info`` for
outbound presses; the comment at that call site says so outright ("the bot
never reads it"). It is the same narration as the other eleven. The
difference is who calls it: the sniffer, whose whole purpose is narrating a
capture. Narration nobody calls is dead; narration the capture tool calls
is the capture tool working.
"""

from __future__ import annotations
