"""Live double-entry fuel book — Phase 3 of the physics roadmap.

Between two absolute wire fuel readings the book accumulates ENTRIES,
each a feasibility interval ``[lo, hi]`` for that event's fuel effect
(physics predicts exact debits for shots/radar/mines, ranged debits
for walks and teleport drift, optional debits for enemy hits, open
credits for pickups). On every reading the measured residual must
fall inside the summed interval; outside is a physics divergence —
either a wiki claim is wrong or the game changed. The caller (the
wire mutation point) emits the diagnostic; this module only keeps the
books. See ``wiki/pages/physics-module-roadmap.md`` Phase 3.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.contracts.enforcement import enforce_contract, require

FuelEntryKind = Literal[
    "shot_single",
    "shot_dual",
    "shot_missile",
    "shot_homing",
    "homing_carry",
    "walk",
    "radar",
    "mine_press",
    "teleport",
    "pickup",
    "enemy_hit",
    "detonation",
]

FUEL_ENTRY_KINDS: tuple[FuelEntryKind, ...] = (
    "shot_single",
    "shot_dual",
    "shot_missile",
    "shot_homing",
    "homing_carry",
    "walk",
    "radar",
    "mine_press",
    "teleport",
    "pickup",
    "enemy_hit",
    "detonation",
)
"""Every entry kind, for validation and iteration."""


class FuelEntryDict(TypedDict):
    """One predicted fuel effect with its feasibility interval."""

    kind: FuelEntryKind
    lo: int
    hi: int


class FuelWindowVerdictDict(TypedDict):
    """Reconciliation result for one reading-to-reading window."""

    balanced: bool
    residual: int
    lo: int
    hi: int
    entry_kinds: str


class FuelBookDict(TypedDict):
    """The running account between QUIET wire fuel readings.

    Live charges lag their cause echoes by up to a few sync windows,
    so per-sync judgement mis-attributes (2026-07-21 first-light soak:
    71 false divergences, every one a debit landing one window after
    its entry). The book therefore accumulates readings into a BLOCK
    and judges only at a quiet boundary — a reading with zero delta
    and no entries recorded since the previous reading — or at the
    forced cap. This is the live twin of the audit's episode method.
    """

    last_fuel: int | None
    block_start_fuel: int | None
    readings_in_block: int
    entries_at_last_reading: int
    entries: list[FuelEntryDict]
    windows: int
    divergences: int


def make_fuel_book() -> FuelBookDict:
    """Return an empty fuel book.

    Returns:
        A book with no confirmed reading and no open entries.
    """
    return FuelBookDict(
        last_fuel=None,
        block_start_fuel=None,
        readings_in_block=0,
        entries_at_last_reading=0,
        entries=[],
        windows=0,
        divergences=0,
    )


class FuelEntryContract:
    """Structural invariants on a fuel-book entry."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "fuel_book_entry"

    def check(self, *, book: FuelBookDict, kind: FuelEntryKind, lo: int, hi: int) -> None:
        """Validate an entry before it enters the book.

        Args:
            book: The book receiving the entry.
            kind: Entry kind.
            lo: Most-negative feasible delta.
            hi: Least-negative (or most-positive) feasible delta.

        Raises:
            LedgerInvariantError: If the interval is inverted or the
                kind is unknown.
        """
        require(lo <= hi, LedgerInvariantError, kind=kind, lo=repr(lo), hi=repr(hi))
        require(kind in FUEL_ENTRY_KINDS, LedgerInvariantError, kind=kind)
        require(len(book["entries"]) < 10_000, LedgerInvariantError, kind=kind)


class FuelReadingContract:
    """Structural invariants on an absolute fuel reading."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "fuel_book_reading"

    def check(self, *, book: FuelBookDict, fuel_total: int) -> None:
        """Validate a reading before reconciliation.

        Args:
            book: The book being reconciled.
            fuel_total: New absolute fuel level.

        Raises:
            LedgerInvariantError: If the reading is negative.
        """
        require(fuel_total >= 0, LedgerInvariantError, fuel_total=repr(fuel_total))


@enforce_contract(FuelEntryContract())
def record_fuel_entry(*, book: FuelBookDict, kind: FuelEntryKind, lo: int, hi: int) -> None:
    """Record one predicted fuel effect into the open window.

    Args:
        book: The book receiving the entry.
        kind: Entry kind.
        lo: Most-negative feasible delta (e.g. -10 for a dual shot).
        hi: Least-negative feasible delta (0 for an optional debit;
            positive ceilings are open credits such as pickups).
    """
    book["entries"].append(FuelEntryDict(kind=kind, lo=lo, hi=hi))


BLOCK_READING_CAP = 50
"""Force-judge a block after this many readings so a never-quiet
combat stretch cannot defer judgement forever."""


@enforce_contract(FuelReadingContract())
def record_fuel_reading(*, book: FuelBookDict, fuel_total: int) -> FuelWindowVerdictDict | None:
    """Fold one absolute wire fuel reading into the open block.

    The first reading anchors the account. Later readings extend the
    block; the block is judged when the boundary is QUIET (this
    reading changed nothing and no entries arrived since the previous
    reading) or when the reading cap forces it. At judgement the
    block's total residual must fall inside the entries' summed
    feasibility interval. Homing shots may split their debit across
    the block boundary, so each one seeds the next block with a
    ``homing_carry`` entry of ``[-5, 0]``.

    Args:
        book: The book being reconciled.
        fuel_total: New absolute fuel level from the wire.

    Returns:
        The block verdict when a block closed, else None.
    """
    last_fuel = book["last_fuel"]
    book["last_fuel"] = fuel_total
    if last_fuel is None:
        book["block_start_fuel"] = fuel_total
        book["readings_in_block"] = 0
        book["entries_at_last_reading"] = 0
        return None
    book["readings_in_block"] += 1
    quiet = (
        fuel_total == last_fuel
        and len(book["entries"]) == book["entries_at_last_reading"]
        and book["readings_in_block"] > 1
    )
    book["entries_at_last_reading"] = len(book["entries"])
    if not quiet and book["readings_in_block"] < BLOCK_READING_CAP:
        return None
    entries = book["entries"]
    block_start = book["block_start_fuel"]
    residual = fuel_total - (block_start if block_start is not None else fuel_total)
    lo = sum(entry["lo"] for entry in entries)
    hi = sum(entry["hi"] for entry in entries)
    balanced = lo <= residual <= hi
    book["windows"] += 1
    if not balanced:
        book["divergences"] += 1
    book["entries"] = [
        FuelEntryDict(kind="homing_carry", lo=-5, hi=0)
        for entry in entries
        if entry["kind"] == "shot_homing"
    ]
    book["block_start_fuel"] = fuel_total
    book["readings_in_block"] = 0
    book["entries_at_last_reading"] = len(book["entries"])
    return FuelWindowVerdictDict(
        balanced=balanced,
        residual=residual,
        lo=lo,
        hi=hi,
        entry_kinds=",".join(entry["kind"] for entry in entries) or "(none)",
    )


__all__ = [
    "BLOCK_READING_CAP",
    "FUEL_ENTRY_KINDS",
    "FuelBookDict",
    "FuelEntryDict",
    "FuelEntryKind",
    "FuelWindowVerdictDict",
    "make_fuel_book",
    "record_fuel_entry",
    "record_fuel_reading",
]
