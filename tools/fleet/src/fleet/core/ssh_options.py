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

ONE CONNECTION PER NODE, SHARED (MCPs board task ee3d6cdc).
Every call of a job opened its own connection, and loki is reached only
through a Cloudflare tunnel, where a login costs a websocket and a full
handshake, about 2 s and now and then more than ``ConnectTimeout``: measured
from diphtheria on 2026-10-21, 322 timed logins with 19 over 3 s and one of
10.6 s, and 9 of loki's 585 jobs in the week before closed on 'Connection
timed out during banner exchange'. ``ControlMaster=auto`` makes the first
call to a node the master and every later one a session on its connection,
``ControlPersist`` keeps that master for :data:`MASTER_PERSIST_SECONDS`
after its last session, and the keepalive above ends a master whose peer
went away, so the next call starts a fresh one. Measured the same morning:
the first call 2.15 s, each of the next seven 0.22 s, a 200 KB answer 0.53
s, and a persisted master holds none of a captured call's pipes open.

THE HUB IS LINUX (API board task 2fab5e74). These options were chosen per
hub platform while austinpc could be the hub, because OpenSSH for Windows
has no multiplexing. Every hub duty is diphtheria's (``registry/machines``),
and no hub job runs on a Windows machine (measured 2026-10-21 on austinpc and
loki), so every connection shares its node's master. Windows nodes such as
loki are the far end of these connections, which multiplexing never
touches.
"""

from __future__ import annotations

from typing import Final

#: How long a node's shared connection outlives its last session, in seconds:
#: longer than the gap between one job's calls, which a collect pass three
#: minutes after the launch reaches too.
MASTER_PERSIST_SECONDS: Final[int] = 300

#: Where a node's shared connection listens, under the hub account's ssh
#: directory: ``%C`` is ssh's hash of the local host, the node, its port and
#: its user, so it names one node's master and stays short of a socket path's
#: length limit.
CONTROL_PATH: Final[str] = "~/.ssh/fleet-mux-%C"

#: The options that bound a connection's waits.
BASE_OPTIONS: Final[tuple[str, ...]] = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
)

#: The options that share one connection per node.
SHARED_CONNECTION_OPTIONS: Final[tuple[str, ...]] = (
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPath={CONTROL_PATH}",
    "-o",
    f"ControlPersist={MASTER_PERSIST_SECONDS}",
)


#: The options every ssh and scp to a node carries: the bounds, then the
#: shared connection.
SSH_OPTIONS: Final[tuple[str, ...]] = (*BASE_OPTIONS, *SHARED_CONNECTION_OPTIONS)

__all__ = [
    "BASE_OPTIONS",
    "CONTROL_PATH",
    "MASTER_PERSIST_SECONDS",
    "SHARED_CONNECTION_OPTIONS",
    "SSH_OPTIONS",
]
