"""Per-enemy damage accounting — dealt and taken, by weapon kind.

User ruling 2026-07-27 (after the first PvP death, run 183703, where
"did we hurt the guest?" took offline archaeology to answer): the bot
tracks damage to enemies and damage taken per enemy, with the weapon
kind, live. Combined with the fuel book's per-kind totals this traces
fuel the whole way through a session.

Sources, all wire-grounded:

* **Dealt** — every own ShootEvent echo carries the weapon byte; a
  one-slot pairing holds it until the shot resolves (the bot fires at
  most one shot per ~2 s tick, so a single slot suffices, mirroring
  ``pending_shot_inventory_snapshot``). On a confirmed hit the victim
  is charged the measured victim cost ([[game-economy]] damage table
  via :mod:`tankpit_bot.physics.damage`). An unpaired hit (echo lost)
  counts under ``unknown`` with zero fuel — counted, never invented.
* **Taken** — every enemy ShootEvent carries shooter and weapon; the
  shot is counted against that shooter immediately, and its victim
  cost is CONFIRMED as damage only when a following fuel reading
  actually drops by at least that cost within the pairing window
  (armor-absorbed or missed incoming shots stay counted but
  unconfirmed).
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.contracts.enforcement import enforce_contract, require
from tankpit_bot.physics.damage import (
    DUAL_HIT_VICTIM_COST,
    HOMING_HIT_VICTIM_COST,
    MISSILE_HIT_VICTIM_COST,
    SINGLE_HIT_VICTIM_COST,
)

_WEAPON_NAMES: dict[int, str] = {0: "single", 1: "dual", 2: "missile", 3: "homing"}

_VICTIM_COSTS: dict[int, int] = {
    0: SINGLE_HIT_VICTIM_COST,
    1: DUAL_HIT_VICTIM_COST,
    2: MISSILE_HIT_VICTIM_COST,
    3: HOMING_HIT_VICTIM_COST,
}

_PENDING_INCOMING_TTL_MS = 4000
"""How long an incoming shot waits for its fuel-reading confirmation.
Fuel syncs arrive every ~2 s; two windows covers charge latency."""

_NO_PENDING_WEAPON = -1
"""Sentinel: no own shot echo is awaiting resolution."""


class EnemyDamageSideDict(TypedDict):
    """One direction of the ledger against one enemy.

    Attributes:
        name: Enemy display name at first observation.
        single: Shot count for weapon 0.
        dual: Shot count for weapon 1.
        missile: Shot count for weapon 2.
        homing: Shot count for weapon 3.
        unknown: Hits whose weapon pairing was unavailable (dealt
            side only; never carries fuel).
        fuel: Damage in fuel — dealt: measured victim costs of
            resolved hits; taken: fuel-reading-confirmed losses.
    """

    name: str
    single: int
    dual: int
    missile: int
    homing: int
    unknown: int
    fuel: int


class PendingIncomingDict(TypedDict):
    """One incoming shot awaiting fuel-reading confirmation."""

    shooter_id: int
    cost: int
    deadline_ms: int


class ConfirmedIncomingDict(TypedDict):
    """One fuel-confirmed incoming hit, timestamped for rate windows.

    ``shooter_id`` carries the attribution from the pending queue so
    the rate window can be scoped to attackers who still exist — the
    2026-07-31 arena soak proved the attacker-agnostic window wrong:
    a freshly killed enemy's hits kept projecting forward for the
    whole 10 s window and the bot blocked three healthy targets as
    "unwinnable" on a dead tank's damage.
    """

    timestamp_ms: int
    cost: int
    shooter_id: int


class DamageBookDict(TypedDict):
    """The two-sided per-enemy damage ledger.

    Attributes:
        dealt: Our resolved hits per enemy, keyed by tank id string.
        taken: Incoming shots per shooter, keyed by tank id string;
            ``fuel`` holds only fuel-confirmed damage.
        pending_dealt_weapon: Weapon byte of the last own shot echo,
            or ``-1`` when no shot awaits resolution.
        pending_incoming: Incoming shots not yet fuel-confirmed.
        confirmed_incoming: Recent fuel-confirmed hits, oldest first.
            Feeds the damage-aware engagement break's incoming-rate
            window; pruned to the window on every read.
    """

    dealt: dict[str, EnemyDamageSideDict]
    taken: dict[str, EnemyDamageSideDict]
    pending_dealt_weapon: int
    pending_incoming: list[PendingIncomingDict]
    confirmed_incoming: list[ConfirmedIncomingDict]


def make_damage_book() -> DamageBookDict:
    """Return an empty damage book.

    Returns:
        A book with no enemies and no pending pairings.
    """
    return DamageBookDict(
        dealt={},
        taken={},
        pending_dealt_weapon=_NO_PENDING_WEAPON,
        pending_incoming=[],
        confirmed_incoming=[],
    )


def _side(
    ledger: dict[str, EnemyDamageSideDict],
    tank_id: int,
    name: str,
) -> EnemyDamageSideDict:
    """Get or create one enemy's row in a ledger direction.

    Args:
        ledger: The ``dealt`` or ``taken`` map.
        tank_id: Enemy tank id.
        name: Enemy display name (kept from first observation).

    Returns:
        The enemy's mutable row.
    """
    key = str(tank_id)
    row = ledger.get(key)
    if row is None:
        row = EnemyDamageSideDict(
            name=name,
            single=0,
            dual=0,
            missile=0,
            homing=0,
            unknown=0,
            fuel=0,
        )
        ledger[key] = row
    return row


class OwnShotEchoContract:
    """Structural invariants on a paired own-shot echo."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "damage_book_own_shot_echo"

    def check(self, book: DamageBookDict, weapon: int) -> None:
        """Validate the echo before it takes the pairing slot.

        Args:
            book: The damage book.
            weapon: Weapon byte from our own ShootEvent echo.

        Raises:
            LedgerInvariantError: If the weapon byte is outside the
                wire vocabulary.
        """
        require(weapon in _WEAPON_NAMES, LedgerInvariantError, weapon=repr(weapon))


class IncomingShotContract:
    """Structural invariants on an incoming enemy shot record."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "damage_book_incoming_shot"

    def check(
        self,
        book: DamageBookDict,
        shooter_id: int,
        shooter_name: str,
        weapon: int,
        now_ms: int,
    ) -> None:
        """Validate an incoming shot before it ledgers.

        Args:
            book: The damage book.
            shooter_id: Who fired.
            shooter_name: Shooter display name.
            weapon: Weapon byte from the enemy ShootEvent.
            now_ms: Current wall-clock ms.

        Raises:
            LedgerInvariantError: If the shooter id is negative, the
                timestamp is negative, or the pending queue is
                runaway-large.
        """
        require(shooter_id >= 0, LedgerInvariantError, shooter_id=repr(shooter_id))
        require(now_ms >= 0, LedgerInvariantError, now_ms=repr(now_ms))
        require(
            len(book["pending_incoming"]) < 10_000,
            LedgerInvariantError,
            pending=repr(len(book["pending_incoming"])),
        )


@enforce_contract(OwnShotEchoContract())
def record_own_shot_echo(book: DamageBookDict, weapon: int) -> None:
    """Hold the weapon byte of our just-echoed shot for hit pairing.

    Args:
        book: The damage book.
        weapon: Weapon byte from our own ShootEvent echo.
    """
    book["pending_dealt_weapon"] = weapon


def resolve_dealt(
    book: DamageBookDict,
    victim_id: int,
    victim_name: str,
    intended_id: int,
) -> None:
    """Charge a confirmed hit to its victim using the paired weapon.

    A hit whose victim tile is unresolvable (``victim_id == -1`` --
    the off-viewport reroute case, [[shoot-event-format]]) is charged
    to the COMMANDED target instead: the reroute law guarantees an
    id-targeted shot lands on the specified id wherever it stands
    (user callout 2026-07-27: "dont we know the intended target?").
    In-viewport seeker retargets resolve a real victim id and are
    unaffected.

    Args:
        book: The damage book.
        victim_id: Tank id the hit resolved against (``-1`` when the
            impact tile is off-viewport).
        victim_name: Victim display name (may be empty).
        intended_id: The commanded target id, used when the wire
            could not resolve the victim.
    """
    weapon = book["pending_dealt_weapon"]
    book["pending_dealt_weapon"] = _NO_PENDING_WEAPON
    charged_id = victim_id if victim_id != -1 else intended_id
    row = _side(book["dealt"], charged_id, victim_name or f"tank-{charged_id}")
    weapon_name = _WEAPON_NAMES.get(weapon)
    if weapon_name is None:
        row["unknown"] += 1
        return
    if weapon_name == "single":
        row["single"] += 1
    elif weapon_name == "dual":
        row["dual"] += 1
    elif weapon_name == "missile":
        row["missile"] += 1
    else:
        row["homing"] += 1
    row["fuel"] += _VICTIM_COSTS[weapon]


@enforce_contract(IncomingShotContract())
def record_incoming_shot(
    book: DamageBookDict,
    shooter_id: int,
    shooter_name: str,
    weapon: int,
    now_ms: int,
) -> None:
    """Count an enemy shot fired at us and queue its confirmation.

    Args:
        book: The damage book.
        shooter_id: Who fired.
        shooter_name: Shooter display name.
        weapon: Weapon byte from the enemy ShootEvent.
        now_ms: Current wall-clock ms.
    """
    row = _side(book["taken"], shooter_id, shooter_name)
    weapon_name = _WEAPON_NAMES.get(weapon)
    if weapon_name == "single":
        row["single"] += 1
    elif weapon_name == "dual":
        row["dual"] += 1
    elif weapon_name == "missile":
        row["missile"] += 1
    elif weapon_name == "homing":
        row["homing"] += 1
    else:
        row["unknown"] += 1
        return
    book["pending_incoming"].append(
        PendingIncomingDict(
            shooter_id=shooter_id,
            cost=_VICTIM_COSTS[weapon],
            deadline_ms=now_ms + _PENDING_INCOMING_TTL_MS,
        )
    )


def confirm_incoming_damage(book: DamageBookDict, fuel_delta: int, now_ms: int) -> None:
    """Confirm queued incoming shots against an observed fuel drop.

    Oldest-first: each pending shot whose victim cost fits inside the
    remaining drop is confirmed and charged to its shooter; expired
    pendings (no covering drop within the TTL) are discarded — they
    stay counted as shots but never as damage.

    Args:
        book: The damage book.
        fuel_delta: The fuel reading's delta (negative on a drop).
        now_ms: Current wall-clock ms.
    """
    live = [p for p in book["pending_incoming"] if p["deadline_ms"] >= now_ms]
    budget = -fuel_delta if fuel_delta < 0 else 0
    remaining: list[PendingIncomingDict] = []
    for pending in live:
        if budget >= pending["cost"]:
            budget -= pending["cost"]
            key = str(pending["shooter_id"])
            book["taken"][key]["fuel"] += pending["cost"]
            book["confirmed_incoming"].append(
                ConfirmedIncomingDict(
                    timestamp_ms=now_ms,
                    cost=pending["cost"],
                    shooter_id=pending["shooter_id"],
                )
            )
        else:
            remaining.append(pending)
    book["pending_incoming"] = remaining


def incoming_damage_window(
    book: DamageBookDict,
    now_ms: int,
    window_ms: int,
    excluded_shooter_ids: frozenset[int],
) -> tuple[int, int]:
    """Return confirmed incoming (hits, fuel) inside the trailing window.

    The instrument behind the damage-aware engagement break: only
    fuel-CONFIRMED hits count (a counted-but-unconfirmed shot never
    inflates the rate), and the log is pruned to the window on every
    read so it cannot grow unbounded.

    Hits from shooters in ``excluded_shooter_ids`` are excluded. The
    book is policy-free about WHY a shooter is excluded — the caller
    (the world service) owns that law: shooters who can no longer
    fire on us must not project into the next engagement. Two classes
    qualify today: registry-DEACTIVATED (2026-07-31 arena soak: a
    freshly killed enemy's 81/tick blocked three healthy follow-up
    targets as "unwinnable at any fuel") and registry-alive but
    wire-silent past the presence standard (flag-triage-20260902 row
    11: a disengaged pair-mate kept pricing the duel for the whole
    10 s window). A shooter the registry cannot vouch for is NEVER in
    the set — a registry gap must not under-report live danger.
    Excluded shooters' entries stay in the log until the window
    prunes them — presence can flip back within the window.

    Args:
        book: The damage book.
        now_ms: Current wall-clock ms.
        window_ms: Trailing window length in ms.
        excluded_shooter_ids: Tank ids whose hits must not count —
            the caller's can-no-longer-fire verdicts.

    Returns:
        ``(hits, fuel)`` confirmed within ``[now_ms - window_ms, now_ms]``
        from shooters not excluded.
    """
    floor = now_ms - window_ms
    book["confirmed_incoming"] = [
        hit for hit in book["confirmed_incoming"] if hit["timestamp_ms"] >= floor
    ]
    counted = [
        hit for hit in book["confirmed_incoming"] if hit["shooter_id"] not in excluded_shooter_ids
    ]
    hits = len(counted)
    fuel = sum(hit["cost"] for hit in counted)
    return hits, fuel


def total_fuel(ledger: dict[str, EnemyDamageSideDict]) -> int:
    """Sum one ledger direction's fuel-confirmed damage across enemies.

    Args:
        ledger: The ``dealt`` or ``taken`` map.

    Returns:
        Total damage in fuel.
    """
    return sum(side["fuel"] for side in ledger.values())


def summarize_side(ledger: dict[str, EnemyDamageSideDict]) -> str:
    """Render one ledger direction as a compact human-readable line.

    Args:
        ledger: The ``dealt`` or ``taken`` map.

    Returns:
        Semicolon-joined per-enemy summaries, or ``"none"``.
    """
    if not ledger:
        return "none"
    parts: list[str] = []
    for key in sorted(ledger, key=int):
        row = ledger[key]
        counts = (
            ("single", row["single"]),
            ("dual", row["dual"]),
            ("missile", row["missile"]),
            ("homing", row["homing"]),
            ("unknown", row["unknown"]),
        )
        weapons = ", ".join(f"{kind}={count}" for kind, count in counts if count > 0)
        parts.append(f"{row['name']}({key}): {weapons or 'no shots'} fuel={row['fuel']}")
    return "; ".join(parts)


__all__ = [
    "ConfirmedIncomingDict",
    "DamageBookDict",
    "EnemyDamageSideDict",
    "IncomingShotContract",
    "OwnShotEchoContract",
    "PendingIncomingDict",
    "confirm_incoming_damage",
    "incoming_damage_window",
    "make_damage_book",
    "record_incoming_shot",
    "record_own_shot_echo",
    "resolve_dealt",
    "summarize_side",
    "total_fuel",
]
