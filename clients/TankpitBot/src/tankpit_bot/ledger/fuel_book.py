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
    "boundary_strand",
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
    "boundary_strand",
)
"""Every entry kind, for validation and iteration."""

MAX_SERVE_CHARGE = 10
"""The largest single serve charge (a dual/missile/homing shot or a
radar press). A cap-forced block boundary can strand exactly one such
charge on either side of the cut — the 2026-08-28 corpus mine found
156 divergences whose gap was exactly one weapon charge (-10 duals in
ENGAGE, -6 clearance singles in COLLECT, -5 homing halves), each
mirrored by a +charge under-spend in the following block."""


class FuelEntryDict(TypedDict):
    """One predicted fuel effect with its feasibility interval."""

    kind: FuelEntryKind
    lo: int
    hi: int


class FuelKindTotalDict(TypedDict):
    """Cumulative session totals for one entry kind.

    The trace half of the book (user ruling 2026-07-27: fuel traced
    the whole way through): while windows judge physics, the totals
    answer "where did the session's fuel go" -- teleport drain, walk
    drain, shot spend, pickup income -- as summed feasibility bounds.
    """

    count: int
    lo_sum: int
    hi_sum: int


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
    totals: dict[str, FuelKindTotalDict]


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
        totals={},
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
    total = book["totals"].get(kind)
    if total is None:
        book["totals"][kind] = FuelKindTotalDict(count=1, lo_sum=lo, hi_sum=hi)
    else:
        total["count"] += 1
        total["lo_sum"] += lo
        total["hi_sum"] += hi


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
    forced = not quiet
    if forced and book["readings_in_block"] < BLOCK_READING_CAP:
        return None
    entries = book["entries"]
    if forced:
        # A cap-forced cut can land mid charge/echo pair: the last
        # serve's charge in this block with its echo in the next (or
        # the echo here with its charge still in flight). Quiet
        # boundaries cannot strand — the ~100 ms charge/echo lag never
        # spans a zero-delta 2 s reading gap — so the tolerance exists
        # ONLY here, bounded to one serve charge each way (the
        # 2026-08-28 corpus mine: every strand gap was exactly one
        # weapon charge).
        entries = [*entries, FuelEntryDict(kind="boundary_strand", lo=-MAX_SERVE_CHARGE, hi=0)]
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
    if forced:
        # The mirror: an entry judged above may have its charge still
        # in flight, landing as an unexplained debit in the next
        # block; and a stranded charge judged above surfaces as its
        # echo's un-fallen entry there.
        book["entries"].append(
            FuelEntryDict(kind="boundary_strand", lo=-MAX_SERVE_CHARGE, hi=MAX_SERVE_CHARGE)
        )
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


class FuelBookOnlyContract:
    """Structural invariants on a book-only operation."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "fuel_book_only"

    def check(self, *, book: FuelBookDict) -> None:
        """Validate the book before the operation.

        Args:
            book: The book being operated on.

        Raises:
            LedgerInvariantError: If the entry list is malformed.
        """
        require(
            len(book["entries"]) < 10_000,
            LedgerInvariantError,
            entries=repr(len(book["entries"])),
        )


class FuelWidenContract:
    """Structural invariants on a teleport re-pricing."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "fuel_book_widen"

    def check(self, *, book: FuelBookDict, widen_by: int) -> None:
        """Validate a widening before it applies.

        Args:
            book: The book holding the open teleport entry.
            widen_by: Non-negative widening, in fuel.

        Raises:
            LedgerInvariantError: If the widening is negative.
        """
        require(widen_by >= 0, LedgerInvariantError, widen_by=repr(widen_by))


@enforce_contract(FuelBookOnlyContract())
def reset_fuel_book_on_death(*, book: FuelBookDict) -> None:
    """Re-anchor the book across a death: the account does not survive it.

    The killing drain is unbookable (the fatal hits took fuel below
    zero mid-batch) and the respawn refill arrives as an absolute
    reading with no announced cause — the three largest corpus
    divergences (+1351, +1078, +361, all ~20 s after a death,
    2026-08-28 mine) were respawn refills judged against a dead
    block. Death wipes the open entries and the anchor; the next wire
    reading opens a fresh account, exactly like the session's first.

    Args:
        book: The book being re-anchored.
    """
    book["last_fuel"] = None
    book["block_start_fuel"] = None
    book["readings_in_block"] = 0
    book["entries_at_last_reading"] = 0
    book["entries"] = []


@enforce_contract(FuelWidenContract())
def widen_last_teleport_entry(*, book: FuelBookDict, widen_by: int) -> bool:
    """Widen the newest open teleport entry for a displaced landing.

    The server charges the distance to the ACTUAL landing tile
    ([[game-economy]]#teleport-cost); the entry was priced at dispatch
    from the REQUESTED tile with only the routine +/-6-tile drift
    allowance. When the landing confirm proves a bigger displacement
    (refusals land a full field away), the triangle inequality bounds
    the true charge within ``6 * displacement`` fuel of the dispatch
    price — the 2026-08-28 corpus mine matched every teleport
    under-spend gap to a displacement receipt this widening covers.

    Args:
        book: The book holding the open teleport entry.
        widen_by: Fuel to widen each bound by (``teleport_cost`` over
            the requested-to-landed displacement, plus rounding).

    Returns:
        True when an open teleport entry was widened, False when the
        block holding it has already been judged.
    """
    for entry in reversed(book["entries"]):
        if entry["kind"] != "teleport":
            continue
        new_lo = entry["lo"] - widen_by
        new_hi = min(entry["hi"] + widen_by, 0)
        total = book["totals"]["teleport"]
        total["lo_sum"] += new_lo - entry["lo"]
        total["hi_sum"] += new_hi - entry["hi"]
        entry["lo"] = new_lo
        entry["hi"] = new_hi
        return True
    return False


__all__ = [
    "BLOCK_READING_CAP",
    "FUEL_ENTRY_KINDS",
    "MAX_SERVE_CHARGE",
    "FuelBookDict",
    "FuelBookOnlyContract",
    "FuelEntryDict",
    "FuelEntryKind",
    "FuelKindTotalDict",
    "FuelWidenContract",
    "FuelWindowVerdictDict",
    "make_fuel_book",
    "record_fuel_entry",
    "record_fuel_reading",
    "reset_fuel_book_on_death",
    "widen_last_teleport_entry",
]
