"""The authoritative container claim: an exclusive-create file mutex.

The advisory claim rows in each fleet report steer siblings' planning
away from held containers, but they arrive AFTER the decision — the
five-bot World run measured 1,499 pickup dispatches with 273 tiles
where two bots dispatched within 30 s at a MEDIAN GAP OF ZERO seconds
([[fleet-forage-allocation]]): siblings commit inside the same tick,
and no message protocol can arbitrate a collision that happens before
the message. The filesystem can, because exclusive create
(``O_CREAT | O_EXCL``) is atomic: first-writer-wins on one file per
container tile is a true mutex with no broker, it survives divergent
knowledge (a bot only ever claims what it can see), and it preserves
the single-tank rule — a solo bot creates claims nobody contends and
behaves identically.

Protocol laws:

* **Existence is the lock; content is metadata.** Creation is atomic
  but the content write is not, so a reader that catches the file
  mid-create simply cannot judge ownership or staleness this beat —
  it treats the claim as held and loses one 2 s tick, never a
  journey. Ownership and staleness decisions are made only from
  content that decoded.
* **Claims expire by their own stamp.** The holder refreshes its
  claim every full tick; a stamp older than :data:`CLAIM_TTL_MS` is a
  crashed or wedged holder and any contender may reap it. Two
  contenders may race the reap — the unlink tolerates the loss and
  the retry create arbitrates, so exactly one wins.
* **Only the owner releases.** A release reads the claim first and
  deletes only its own; the mid-create window reads as "not mine" and
  the stale copy is left for the reap, never deleted blind.
"""

from __future__ import annotations

import re
from pathlib import Path

from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.runtime_artifacts import bot_run_dir

CLAIM_TTL_MS = 30_000
"""Age at which a container claim is a crashed holder's leftover.

A live holder refreshes on every FULL tick (~2 s), but early-return
ticks — in-flight actions, pending shot feedback, the measured 8 s
empty-container receipt stalls — refresh nothing, so the horizon must
ride out the longest sanctioned stall with margin. 30 s does, sits in
the same freshness family as the hop-sighting and enemy-sighting
horizons, and bounds how long a dead bot's claim can wall a container
off from the fleet. The wire-silence watchdog kills a wedged session
at 90 s, so every claim a dead process held frees well before its
tank even leaves the field."""

_ROOM_ID = re.compile(r"^[A-Za-z0-9-]{1,32}$")

_CLAIMS_DIRNAME = "_claims"
"""Directory under ``runs/bot`` holding the per-room claim files.

The leading underscore is load-bearing: instance names must match
``^[a-z0-9][a-z0-9_-]{0,31}$`` (:func:`~tankpit_bot.runtime_artifacts.
resolve_bot_instance`), so no bot instance directory can ever collide
with it."""


class ContainerClaimDict(TypedDict):
    """One held container claim's metadata.

    Attributes:
        instance: The holder's instance name ("" for the sole-bot
            namespace — which is by construction the only process in
            that namespace, so the empty name is still unambiguous).
        tank_id: The holder's tank id, for the events stream.
        claimed_ms: Wall-clock ms of the claim's last refresh — the
            stamp the staleness reap judges.
    """

    instance: str
    tank_id: int
    claimed_ms: int


def encode_container_claim(claim: ContainerClaimDict) -> JSONObject:
    """Serialize a container claim to a JSON object.

    Args:
        claim: Claim to serialize.

    Returns:
        JSON object with the claim's fields.
    """
    return {
        "instance": claim["instance"],
        "tank_id": claim["tank_id"],
        "claimed_ms": claim["claimed_ms"],
    }


def decode_container_claim(data: JSONObject) -> ContainerClaimDict:
    """Validate and deserialize a container claim from a JSON object.

    Args:
        data: JSON object with the claim's fields.

    Returns:
        The validated claim.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return ContainerClaimDict(
        instance=require_str(data, "instance"),
        tank_id=require_int(data, "tank_id"),
        claimed_ms=require_int(data, "claimed_ms"),
    )


def claim_path(room: str, x: int, y: int) -> Path:
    """Return the claim file path for one container tile.

    Args:
        room: The wire room id the tile's coordinates belong to —
            claims are per-field for the same reason reports merge
            same-room only (coordinates from another field are
            poison).
        x: Container tile X.
        y: Container tile Y.

    Returns:
        ``runs/bot/_claims/<room>/<x>_<y>.claim``.

    Raises:
        ValueError: If ``room`` is not a plain alphanumeric room id —
            path separators must never reach the filesystem layer.
    """
    if not _ROOM_ID.match(room):
        raise ValueError(f"room id {room!r} is not a valid claim namespace")
    return bot_run_dir("") / _CLAIMS_DIRNAME / room / f"{x}_{y}.claim"


def _claim_content(instance: str, tank_id: int, now_ms: int) -> str:
    """Render the claim file content for one acquisition or refresh.

    Args:
        instance: The claiming bot's instance name.
        tank_id: The claiming bot's tank id.
        now_ms: The stamp to write.

    Returns:
        The encoded claim JSON.
    """
    return dump_json_str(
        encode_container_claim(
            ContainerClaimDict(instance=instance, tank_id=tank_id, claimed_ms=now_ms)
        )
    )


def _read_claim(path: Path) -> ContainerClaimDict | None:
    """Read and decode an existing claim file.

    Args:
        path: The claim file.

    Returns:
        The decoded claim; ``None`` when the file is gone (released
        between the caller's create failure and this read) or its
        content has not landed yet (the documented non-atomic window
        after the holder's exclusive create). ``None`` never means
        "unclaimed" — the caller re-arbitrates with the file's
        existence, where creation is the only atomic truth.
    """
    try:
        text = _test_hooks.read_text(path)
    except FileNotFoundError:
        return None
    try:
        parsed = load_json_str(text)
    except InvalidJsonError:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        return decode_container_claim(parsed)
    except JSONTypeError:
        return None


def acquire_container_claim(
    room: str,
    x: int,
    y: int,
    *,
    instance: str,
    tank_id: int,
    now_ms: int,
) -> bool:
    """Acquire or refresh the authoritative claim on one container.

    Idempotent per holder: the first call wins the tile, every later
    call by the same holder refreshes the stamp — so the tick loop
    calls this once per full tick for the held plan and never tracks
    "acquire" versus "refresh" itself.

    Args:
        room: The wire room id.
        x: Container tile X.
        y: Container tile Y.
        instance: This bot's instance name.
        tank_id: This bot's tank id.
        now_ms: Current wall-clock ms.

    Returns:
        True when this bot holds the claim after the call; False when
        a sibling holds it (including the unreadable mid-create
        window, which denies for one beat by protocol law).
    """
    path = claim_path(room, x, y)
    content = _claim_content(instance, tank_id, now_ms)
    if _test_hooks.create_text_exclusive(path, content):
        return True
    existing = _read_claim(path)
    if existing is None:
        # Released-or-unreadable between the create failure and the
        # read: one retry create arbitrates the released case, and the
        # mid-create case stays a denial because the file still exists.
        return _test_hooks.create_text_exclusive(path, content)
    if existing["instance"] == instance:
        # Refresh. ``replace_text``'s Windows drop-a-beat law is fine
        # here: the previous stamp stays current and the next full
        # tick refreshes again, well inside the 30 s horizon.
        _test_hooks.replace_text(path, content)
        return True
    if now_ms - existing["claimed_ms"] <= CLAIM_TTL_MS:
        return False
    # Stale foreign claim: the holder crashed or wedged. Reap and
    # re-arbitrate — a concurrent reaper may win the recreate, and the
    # exclusive create decides it either way.
    _test_hooks.remove_file(path)
    return _test_hooks.create_text_exclusive(path, content)


def fresh_denied_claim_tiles(denied: dict[str, int], now_ms: int) -> set[str]:
    """Tiles whose claim denial is still standing.

    The denial memory exists because the advisory claimed set is
    replaced WHOLESALE by every merge pass: a denial stamped into it
    would survive only until the same tick's exchange, and a winner
    that crashed right after claiming never publishes the advisory
    row at all — leaving the loser to re-pick the tile and lose one
    beat per tick until the dead claim aged out. Remembering own
    denials locally, bounded by :data:`CLAIM_TTL_MS`, caps that at
    ONE denied beat: past the horizon the denying claim is itself
    reapable, so re-contending becomes legitimate exactly when the
    memory expires.

    Args:
        denied: Session denial memory, tile key (``"x,y"``) to the
            denial's stamp.
        now_ms: Current wall-clock ms.

    Returns:
        The tile keys still inside the denial horizon.
    """
    return {tile for tile, denied_ms in denied.items() if now_ms - denied_ms <= CLAIM_TTL_MS}


def release_container_claim(room: str, x: int, y: int, *, instance: str) -> bool:
    """Release this bot's claim on one container tile.

    Args:
        room: The wire room id.
        x: Container tile X.
        y: Container tile Y.
        instance: This bot's instance name — only the owner's claim is
            ever deleted; a foreign or unreadable claim is left for
            its holder or the staleness reap.

    Returns:
        True when an owned claim was deleted; False when there was
        nothing of this bot's to release.
    """
    path = claim_path(room, x, y)
    existing = _read_claim(path)
    if existing is None or existing["instance"] != instance:
        return False
    _test_hooks.remove_file(path)
    return True


__all__ = [
    "CLAIM_TTL_MS",
    "ContainerClaimDict",
    "acquire_container_claim",
    "claim_path",
    "decode_container_claim",
    "encode_container_claim",
    "fresh_denied_claim_tiles",
    "release_container_claim",
]
