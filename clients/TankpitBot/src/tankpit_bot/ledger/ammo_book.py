"""Live ammo book — the consumption-equals-hit contract, enforced.

Between 0x49 inventory snapshots the book counts what the bot did
(dual/missile/homing shots fired, radar scans dispatched) and what
the wire granted (0x67 equipment gains). At each snapshot every slot
delta must be feasible: weapons may only fall by at most the shots
fired since the last snapshot (misses consume nothing, so fewer is
fine), radars by at most the scans dispatched, and NO slot may rise
without an equipment gain. Armor absorbs incoming hits the bot cannot
predict, so its decreases are unconstrained; its increases still
require a gain. An infeasible delta is a physics divergence on the
ammo channel. See ``wiki/pages/physics-module-roadmap.md`` Phase 3.
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.contracts.enforcement import enforce_contract, require

SLOT_ARMOR = 0
SLOT_DUAL = 1
SLOT_MISSILE = 2
SLOT_HOMING = 3
SLOT_RADAR = 4

_SLOT_NAMES = ("armor", "dual", "missile", "homing", "radar")
_WEAPON_SLOTS: dict[int, int] = {1: SLOT_DUAL, 2: SLOT_MISSILE, 3: SLOT_HOMING}


class AmmoVerdictDict(TypedDict):
    """Reconciliation result for one snapshot-to-snapshot interval."""

    balanced: bool
    detail: str


class AmmoBookDict(TypedDict):
    """The running ammo account between 0x49 snapshots."""

    last_counts: list[int] | None
    shots: list[int]
    enemy_shots: int
    gains: int
    snapshots: int
    divergences: int


def make_ammo_book() -> AmmoBookDict:
    """Return an empty ammo book.

    Returns:
        A book with no anchoring snapshot and zero activity counters.
    """
    return AmmoBookDict(
        last_counts=None,
        shots=[0, 0, 0, 0, 0],
        enemy_shots=0,
        gains=0,
        snapshots=0,
        divergences=0,
    )


class AmmoSnapshotContract:
    """Structural invariants on a 0x49 count snapshot."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "ammo_book_snapshot"

    def check(self, *, book: AmmoBookDict, counts: list[int]) -> None:
        """Validate a snapshot before reconciliation.

        Args:
            book: The book being reconciled.
            counts: The five slot counts from the wire.

        Raises:
            LedgerInvariantError: If the snapshot is not five
                non-negative counts.
        """
        require(len(counts) == 5, LedgerInvariantError, counts=repr(counts))
        require(all(count >= 0 for count in counts), LedgerInvariantError, counts=repr(counts))


class AmmoShotContract:
    """Structural invariants on a counted shot echo."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "ammo_book_shot"

    def check(self, *, book: AmmoBookDict, weapon: int) -> None:
        """Validate a shot echo before counting it.

        Args:
            book: The book being updated.
            weapon: Wire weapon byte.

        Raises:
            LedgerInvariantError: If the weapon byte is negative or
                the book's counter vector is malformed.
        """
        require(weapon >= 0, LedgerInvariantError, weapon=repr(weapon))
        require(len(book["shots"]) == 5, LedgerInvariantError, shots=repr(book["shots"]))


class AmmoActivityContract:
    """Structural invariants on a counted scan or gain."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "ammo_book_activity"

    def check(self, *, book: AmmoBookDict) -> None:
        """Validate the book before counting activity.

        Args:
            book: The book being updated.

        Raises:
            LedgerInvariantError: If the counter vector is malformed.
        """
        require(len(book["shots"]) == 5, LedgerInvariantError, shots=repr(book["shots"]))


@enforce_contract(AmmoShotContract())
def record_ammo_shot(*, book: AmmoBookDict, weapon: int) -> None:
    """Count one own shot echo against its ammo slot.

    Args:
        book: The book being updated.
        weapon: Weapon byte (1=dual, 2=missile, 3=homing; the free
            single, 0, consumes nothing and is ignored).
    """
    slot = _WEAPON_SLOTS.get(weapon)
    if slot is not None:
        book["shots"][slot] += 1


@enforce_contract(AmmoActivityContract())
def record_ammo_scan(*, book: AmmoBookDict) -> None:
    """Count one dispatched radar scan against the radar slot.

    Args:
        book: The book being updated.
    """
    book["shots"][SLOT_RADAR] += 1


@enforce_contract(AmmoActivityContract())
def record_ammo_enemy_shot(*, book: AmmoBookDict) -> None:
    """Count one enemy shot echo (bounds armor consumption).

    Armor absorbs at most 2 shields per incoming hit (a dual), so
    between snapshots the armor slot may fall by at most twice the
    enemy shots observed ([[game-economy]] armor rules, 2026-07-21).

    Args:
        book: The book being updated.
    """
    book["enemy_shots"] += 1


@enforce_contract(AmmoActivityContract())
def record_ammo_gain(*, book: AmmoBookDict) -> None:
    """Count one 0x67 equipment gain (any slots may rise).

    Args:
        book: The book being updated.
    """
    book["gains"] += 1


class AmmoDeathContract:
    """Structural invariants on a booked death penalty."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "ammo_book_death"

    def check(self, *, book: AmmoBookDict, mine_kill: bool) -> None:
        """Validate the book before applying the death transform.

        Args:
            book: The book being updated.
            mine_kill: The 0x41 mine-sentinel attribution.

        Raises:
            LedgerInvariantError: If the counter vector is malformed.
        """
        require(len(book["shots"]) == 5, LedgerInvariantError, shots=repr(book["shots"]))


@enforce_contract(AmmoDeathContract())
def record_ammo_death(*, book: AmmoBookDict, mine_kill: bool) -> None:
    """Apply the death penalty to the book's expectation baseline.

    Wire-verified 2026-08-28 across all six corpus deaths
    ([[equipment-system]]): a tank-kill death sets every slot to
    ``ceil(n/2)`` (the unconsumable armor/missile/radar slots proved
    it with zero exceptions), and the one observed mine death zeroed
    every slot outright (rebuilt afterwards purely via 0x67 gains).
    Without this transform every death's next 0x49 snapshot read as
    an infeasible fall and burned a false ammo divergence (three
    deaths, three divergences, desert 2026-08-26).

    Shots fired before the death in the same interval stay counted:
    halving the pre-shot baseline is always at least as permissive as
    the server's halving of the post-shot count, so feasibility is
    preserved without re-splitting the interval.

    Args:
        book: The book being updated.
        mine_kill: True when the 0x41 carried the mine sentinel.
    """
    last_counts = book["last_counts"]
    if last_counts is None:
        return
    if mine_kill:
        book["last_counts"] = [0, 0, 0, 0, 0]
    else:
        book["last_counts"] = [(count + 1) // 2 for count in last_counts]


@enforce_contract(AmmoSnapshotContract())
def record_ammo_snapshot(*, book: AmmoBookDict, counts: list[int]) -> AmmoVerdictDict | None:
    """Reconcile one 0x49 snapshot against the recorded activity.

    Args:
        book: The book being reconciled.
        counts: The five slot counts [armor, dual, missile, homing,
            radar] from the wire.

    Returns:
        The interval verdict, or None for the anchoring first snapshot.
    """
    last_counts = book["last_counts"]
    shots = book["shots"]
    enemy_shots = book["enemy_shots"]
    gains = book["gains"]
    book["last_counts"] = list(counts)
    book["shots"] = [0, 0, 0, 0, 0]
    book["enemy_shots"] = 0
    book["gains"] = 0
    if last_counts is None:
        return None
    book["snapshots"] += 1
    problems: list[str] = []
    for slot, name in enumerate(_SLOT_NAMES):
        delta = counts[slot] - last_counts[slot]
        if delta > 0 and gains == 0:
            problems.append(f"{name} rose {delta} with no equipment gain")
        elif delta < 0 and slot == SLOT_ARMOR and -delta > 2 * enemy_shots:
            problems.append(f"armor fell {-delta} with only {enemy_shots} enemy shots observed")
        elif delta < 0 and slot != SLOT_ARMOR and -delta > shots[slot]:
            problems.append(f"{name} fell {-delta} with only {shots[slot]} uses recorded")
    balanced = not problems
    if not balanced:
        book["divergences"] += 1
    return AmmoVerdictDict(balanced=balanced, detail="; ".join(problems) or "(balanced)")


__all__ = [
    "SLOT_ARMOR",
    "SLOT_DUAL",
    "SLOT_HOMING",
    "SLOT_MISSILE",
    "SLOT_RADAR",
    "AmmoActivityContract",
    "AmmoBookDict",
    "AmmoDeathContract",
    "AmmoShotContract",
    "AmmoSnapshotContract",
    "AmmoVerdictDict",
    "make_ammo_book",
    "record_ammo_death",
    "record_ammo_enemy_shot",
    "record_ammo_gain",
    "record_ammo_scan",
    "record_ammo_shot",
    "record_ammo_snapshot",
]
