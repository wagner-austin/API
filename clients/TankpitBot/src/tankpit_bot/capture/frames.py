"""The one way to turn a captured WebSocket payload into frame bodies.

A payload arrives base64-encoded (that is how CDP hands it over) and
carries one or more length-prefixed frames. Decoding the base64 and
walking the frames is a single step, and before this module thirteen
places did it for themselves ([[session-state-deglobalisation]]):

* nine hand-rolled the walk with a silent ``break`` on a torn tail —
  ``sniffer.decoders``, ``browser.autoscroll``,
  ``action_lab.viewport_probe``, ``capture.viewport_analysis``,
  ``diagnostics.capture_audit``, ``sim.ghost``, ``sim.transport``, and
  both ``validate`` timelines;
* four already called :func:`~tankpit_bot.protocol.framing.split_frames`
  but each re-composed the base64 decode and the error handling around
  it.

What is shared is the COMPOSITION. What is not shared is the policy on
a bad payload — one site counts them, one classifies them as a typed
skip, one lets them propagate — so this raises and leaves the choice
where it belongs.

Measured against the whole archive on 2026-08-06 before the collapse:
407 sessions, 230,323 received and 72,674 sent payloads. The strict
walk and the silent-truncate walk returned IDENTICAL bodies on every
payload that parsed; no payload failed base64; none was shorter than
three bytes. They differ on exactly four payloads, all inside the one
pre-framing capture ``bot-20260331-230406`` — and on those the silent
walk returns nothing anyway. The choice is therefore between silently
nothing and loudly nothing, never between data and no data.
"""

from __future__ import annotations

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol.framing import FramingError, split_frames


def split_payload_frames(payload: str) -> list[bytes]:
    """Split one base64 WebSocket payload into its logical frame bodies.

    Empty bodies are dropped. A zero-length frame is LEGAL framing that
    carries no message, and every consumer here reads ``body[0]`` to
    route it — so passing one on is an IndexError waiting to happen.
    The walks this replaces hid that by treating ``length == 0`` as a
    torn frame and stopping, which silently discarded every LATER frame
    in the same payload too. Dropping just the empty body keeps the
    rest ([[session-state-deglobalisation]]).

    Args:
        payload: Base64-encoded frame payload as captured from the wire.

    Returns:
        Frame bodies in wire order, each without its length prefix,
        each non-empty.

    Raises:
        FramingError: If the payload is not valid base64, or ends
            mid-frame. Both are corruption of a single payload; the
            caller decides whether that is fatal, countable, or a
            typed skip.
    """
    # An EMPTY payload is not corruption: it is valid (empty) base64
    # carrying zero frames. Only a non-empty payload that fails to
    # decode is a fault, and calling that "not valid base64" is then
    # an accurate message rather than a catch-all.
    if not payload:
        return []
    data = decode_base64_safe(payload)
    if data is None:
        raise FramingError(f"payload is not valid base64 ({len(payload)} chars)")
    return [body for body in split_frames(data) if body]


__all__ = [
    "split_payload_frames",
]
