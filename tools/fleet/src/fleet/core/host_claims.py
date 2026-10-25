"""A host's claims under way, and the turn its processes take to claim.

MCPs board task a85ef09e. A node's ordinary and elevated runners were two
processes on the hub (:mod:`fleet.cli.node_agent`), and each charged its
launches under way from its own memory only
(:mod:`fleet.cli.node_launch`), so for the 40 to 80 s between a claim and
its run reaching the ledger the other runner read the host as emptier than
it was. On 2026-10-07 serendipity held three runs and 6 workers against a
pool of 4 cores that way, twice in one evening, and fleet job c858b46b ran
321 s of its 300 s budget. The elevated runner left the fleet with its
Windows nodes (board task e705f08c); the host stays the unit for any two
processes claiming for it, by two things:

THE CLAIMS ARE THE HOST'S. Each claimed job's grant is written to
``<records>/host-claims/<node>/<job>.json``
(:class:`fleet.contracts.host_claim.HostClaim`) when its launch is handed to
the pool, and the file is removed once the launch has ended, by which time
a launched run is on the ledger. Every process claiming for the host
charges every claim written there, whichever process wrote it. One file per
claim, so ending a launch is one removal that needs no turn: the launch ends
on a thread of the pool while its runner may be holding the turn for its
next claim. A runner killed mid-launch never removes its claim, so the next
tick's collect pass, which refuses the job it finds claimed with nothing
launched, removes it then (MCPs board task 3f7d40fe). A claim also stops
counting at its ``until_unix``, the queue job's own claim lease, and the
turn that finds one past it removes it.

THE CLAIM IS TAKEN IN TURN. Reading the host's load and claiming against it
are two steps, and two processes between them would both read the same free
room. So a claim pass runs inside :func:`claim_turn`, which holds
``<records>/host-claims/<node>.lock``: the file's exclusive creation is the
lock, its text names the turn so only its holder removes it, and a lock
older than :data:`TURN_STALE_SECONDS` was left by a runner that died inside
its turn and is taken away. Claims are read and written inside a turn. The
lock itself is :mod:`fleet.core.file_turn`, which the lease file shares.
"""

from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Generator
from typing import Final

from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.logging import get_logger

from fleet.contracts.host_claim import HostClaim, decode_host_claim, encode_host_claim
from fleet.contracts.node import LiveLoad
from fleet.core import _test_hooks, file_turn
from fleet.core.remote import SSH_TIMEOUT_SECONDS

_log = get_logger(__name__)

#: The directory beside the ledger that holds each host's claims and lock.
HOST_CLAIMS_DIRECTORY: Final = "host-claims"

#: How old a turn's lock is when its holder is taken to have died inside it.
#: A turn runs the capacity and toolchain probes, each a staged script and
#: its run over ssh at :data:`fleet.core.remote.SSH_TIMEOUT_SECONDS` apiece,
#: and the queue's claim beside them, so five of those deadlines bound the
#: longest turn a live runner can take.
TURN_STALE_SECONDS: Final = 5 * SSH_TIMEOUT_SECONDS

#: How long a runner waits between looks at a lock another turn holds.
TURN_POLL_SECONDS: Final = 1

#: The claim turn as :mod:`fleet.core.file_turn` holds it.
CLAIM_TURN: Final = file_turn.TurnBounds(
    what="claim turn",
    waited_for="another process claiming for its host",
    stale_seconds=TURN_STALE_SECONDS,
    poll_seconds=TURN_POLL_SECONDS,
)


def claims_directory(directory: pathlib.Path, *, alias: str) -> pathlib.Path:
    """Where a host's claims under way are recorded, one file each.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.

    Returns:
        The host's claims directory.
    """
    return directory / alias


def lock_path(directory: pathlib.Path, *, alias: str) -> pathlib.Path:
    """Where a host's claim turn is held.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.

    Returns:
        The lock file.
    """
    return directory / f"{alias}.lock"


@contextlib.contextmanager
def claim_turn(directory: pathlib.Path, *, alias: str, runner: str) -> Generator[None, None, None]:
    """Hold the host's claim turn for the span of the ``with`` block.

    Waits while another turn holds the lock, taking away a lock older than
    :data:`TURN_STALE_SECONDS`, and gives the lock back on leaving the block
    however it is left.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.
        runner: The label of the runner taking the turn, written into the
            lock so a reader of it can name the holder.

    Yields:
        Nothing; the turn is held while the block runs.

    Raises:
        OSError: When the lock can be neither created nor read for a reason
            other than another turn holding it.
    """
    with file_turn.held(lock_path(directory, alias=alias), holder=runner, bounds=CLAIM_TURN):
        yield


def live_claims(directory: pathlib.Path, *, alias: str) -> tuple[HostClaim, ...]:
    """The host's claims still under way, read inside a claim turn.

    A claim whose ``until_unix`` has passed is removed here, the one place
    a claim its launch never removed is found.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.

    Returns:
        Every recorded claim whose ``until_unix`` has not passed, ordered by
        job id. A host with no claims directory has never claimed.

    Raises:
        JSONTypeError: When a claim file does not hold a valid claim.
    """
    now = _test_hooks.now()
    live: list[HostClaim] = []
    for path in sorted(claims_directory(directory, alias=alias).glob("*.json")):
        text = _read_claim(path)
        if text is None:
            continue
        claim = decode_host_claim(load_json_str(text))
        if claim["until_unix"] <= now:
            _log.info(
                "%s: %s's claim of %s lapsed at %d unlaunched; removing it",
                alias,
                claim["runner"],
                claim["job_id"],
                claim["until_unix"],
            )
            path.unlink(missing_ok=True)
            continue
        live.append(claim)
    return tuple(live)


def _read_claim(path: pathlib.Path) -> str | None:
    """Read one claim file that its launch may be removing as it is read.

    Args:
        path: The claim file.

    Returns:
        Its text, or None when its launch removed it after it was listed:
        that launch has ended, and its run, if it ran, is on the ledger.
    """
    try:
        return _test_hooks.read_text(path)
    except FileNotFoundError:
        return None


def record(directory: pathlib.Path, *, alias: str, claim: HostClaim) -> None:
    """Write a claim for the host, inside the claim turn that made it.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.
        claim: The claimed job's grant.
    """
    _test_hooks.write_text(
        claims_directory(directory, alias=alias) / f"{claim['job_id']}.json",
        dump_json_str(encode_host_claim(claim)),
    )


def discharge(directory: pathlib.Path, *, alias: str, job_id: str) -> None:
    """Remove a job's claim once its launch has ended; no turn is needed.

    A file another process has open is removed all the same on Linux, so a
    removal needs no retry; the Windows hub that refused one while the
    host's elevated runner read it (MCPs board task a85ef09e) left with the
    fleet's last Windows machine (board task e705f08c).

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.
        job_id: The queue job whose launch ended.
    """
    (claims_directory(directory, alias=alias) / f"{job_id}.json").unlink(missing_ok=True)


def host_load(claims: tuple[HostClaim, ...]) -> LiveLoad:
    """What a host's claims under way hold, counted as live runs are.

    Args:
        claims: The host's live claims.

    Returns:
        One run per claim, with their grants and memory summed.
    """
    return LiveLoad(
        runs=len(claims),
        workers=sum(claim["workers"] for claim in claims),
        ram_gb=sum(claim["ram_gb"] for claim in claims),
    )


__all__ = [
    "HOST_CLAIMS_DIRECTORY",
    "TURN_POLL_SECONDS",
    "TURN_STALE_SECONDS",
    "claim_turn",
    "claims_directory",
    "discharge",
    "host_load",
    "live_claims",
    "lock_path",
    "record",
]
