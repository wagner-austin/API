"""The options every ssh and scp from the hub to a node carries.

``BatchMode=yes`` makes a missing key fail immediately instead of prompting
for a password no automated caller can answer -- the failure is the point,
because a prompt would hang a dispatch forever.

``ServerAliveInterval`` and ``ServerAliveCountMax`` make ssh notice a peer
that went away AFTER the handshake: ``ConnectTimeout`` bounds only the
connection, and a laptop that sleeps mid-command leaves the TCP session
ESTABLISHED with nothing on either end to say so. Measured 2026-09-17
11:15Z on pendragon (board tasks 35940277 and 41ac6ed2): one such ssh sat
three days. Four missed probes fifteen seconds apart end it in a minute.

ONE CONNECTION PER CALL, NEVER A SHARED ONE (MCPs board task ee3d6cdc).
loki is reached only through a Cloudflare tunnel, where a login costs a
websocket and a full handshake, and now and then one stalls past
``ConnectTimeout``. Sharing one connection per node (``ControlMaster``) was
tried for that and measured worse on 2026-10-21 from diphtheria: over the
shared connection the fleet's own script send to loki hung 120 s once in 91
sends, and its remote writer outlived the caller holding the file for as
long as the master lived, so 70 of the next 77 sends failed 'The process
cannot access the file'; then over 100 alternating rounds a shared
connection hung 8 of 200 sends past 30 s, in consecutive pairs on one
master, while a fresh connection per send hung none of 100 (the slowest
6.70 s). A stall on a shared connection holds every call riding it until
the keepalive ends the master; a stall on a fresh one costs that one call,
and a launch whose transport fails is given back to the queue rather than
judged (:mod:`fleet.cli.node_launch`). ``ControlPath=none`` makes the
fresh connection explicit, so neither a ``ControlMaster`` line in the hub
account's ssh config nor a master an earlier release left behind is used.
"""

from __future__ import annotations

from typing import Final

#: The options every ssh and scp to a node carries.
SSH_OPTIONS: Final[tuple[str, ...]] = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "ControlPath=none",
)

__all__ = ["SSH_OPTIONS"]
