"""Engagement ledger: every fight in a run, with economics and outcome.

The issue report answers "did anything break"; the forage economy
answers "where did the collect time go"; this module answers "how did
every FIGHT go" — one row per enemy engaged, with shots, breaks,
outcome, time-to-kill, and the damage trade both ways.

Origin: the 2026-08-26 return-fire/solvency collision. The bot broke
from red-8 for fuel insolvency and then traded six more return shots
with the same tank — a defect that was invisible to every existing
report and took three artifacts to triangulate live. Here it is one
flagged row: an engagement with breaks and a negative damage trade.

Sources, all wire- or diagnostic-grounded, from the events artifact:

* ``WIRE shoot(x,y,id=N)`` lines — per-target shot dispatches.
* ``tank_deactivated`` diagnostics — kills (killer is us; dispatch
  never emits this kind for our own death).
* ``self_deactivated`` diagnostics — our own deaths, from either
  receipt (protocol 0x41, or the Normal-field u16 fuel-wrap).
* ``engagement_break`` diagnostics — solvency/fire breaks per target.
* ``self_alignment_sample`` diagnostics — our own wire id.
* ``tank_identity`` diagnostics — id-to-name resolution.
* the session-end ``damage_ledger`` diagnostic — per-enemy damage
  dealt and taken in fuel, both directions.

CLI: ``tankpit-engagements [events.jsonl]`` — defaults to
``runs/bot/latest.events.jsonl``. Wired into ``make analyze``.
"""

from __future__ import annotations

import re
from pathlib import Path

from platform_core.logging import get_logger
from platform_core.rich_logging import setup_rich_logging
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.forage_economy import _opt_int, _opt_str, _timestamp_seconds
from tankpit_bot.runtime_records import RuntimeEventRecordDict

log = get_logger(__name__)

_DEFAULT_SOURCE = Path("runs/bot/latest.events.jsonl")

_WIRE_SHOOT = re.compile(r"^shoot\(\d+,\d+,id=(\d+)\)$")

#: One per-enemy row of the session-end ``damage_ledger`` string:
#: ``name(id): kind=n, ... fuel=N`` — rows are joined with ``"; "``.
_LEDGER_ROW = re.compile(r"^\s*(?P<name>.+?)\((?P<tank_id>\d+)\):.*fuel=(?P<fuel>\d+)$")


class EngagementDict(TypedDict):
    """One enemy's fight record for the run.

    Attributes:
        target_id: The enemy's wire tank id.
        target_name: Display name (``tank_identity`` / ledger row),
            or ``"?"`` when the run never resolved one.
        first_shot_s: Epoch seconds of our first shot at this enemy,
            or ``None`` when we never fired (they engaged us, or the
            only evidence is a break or the damage ledger).
        last_shot_s: Epoch seconds of our last shot, or ``None``.
        shots: Our wire shot dispatches at this enemy.
        breaks: ``engagement_break`` diagnostics naming this enemy.
        outcome: ``"kill"`` (we deactivated them), ``"killed_us"``
            (they deactivated us), or ``"open"`` (neither by session
            end — survived, departed, or the run ended first).
        seconds_to_kill: First shot to deactivation, or ``None``.
        dealt_fuel: Fuel damage we landed on them (damage ledger).
        taken_fuel: Fuel damage they landed on us (damage ledger).
    """

    target_id: int
    target_name: str
    first_shot_s: float | None
    last_shot_s: float | None
    shots: int
    breaks: int
    outcome: str
    seconds_to_kill: float | None
    dealt_fuel: int
    taken_fuel: int


class EngagementLedgerDict(TypedDict):
    """The run's fight ledger.

    Attributes:
        source_path: Events artifact the ledger was built from.
        self_id: Our wire tank id, or ``None`` when the run emitted
            no ``self_alignment_sample`` (kill/death attribution is
            then impossible and every outcome stays ``"open"``).
        engagements: One row per enemy, ordered by first shot (rows
            we never shot at sort last, by id).
        kills: Rows with outcome ``"kill"``.
        deaths: ``self_deactivated`` records — our own deaths.
        negative_trades: Rows where ``taken_fuel > dealt_fuel``.
        post_break_negative_trades: Negative-trade rows that also
            carry at least one break — the return-fire/solvency
            collision signature this ledger was built to expose.
    """

    source_path: str
    self_id: int | None
    engagements: list[EngagementDict]
    kills: int
    deaths: int
    negative_trades: int
    post_break_negative_trades: int


def _blank_row(target_id: int) -> EngagementDict:
    """Return an empty engagement row for one enemy id.

    Args:
        target_id: The enemy's wire tank id.

    Returns:
        A row with no shots, breaks, damage, or outcome yet.
    """
    return EngagementDict(
        target_id=target_id,
        target_name="?",
        first_shot_s=None,
        last_shot_s=None,
        shots=0,
        breaks=0,
        outcome="open",
        seconds_to_kill=None,
        dealt_fuel=0,
        taken_fuel=0,
    )


def _apply_damage_ledger(
    rows: dict[int, EngagementDict],
    ledger_text: str,
    *,
    side: str,
) -> None:
    """Fold one side of the session damage ledger into the rows.

    Args:
        rows: Engagement rows keyed by enemy id; enemies present only
            in the ledger (they hit us, we never fired) gain a row.
        ledger_text: The ``dealt`` or ``taken`` ledger string —
            ``name(id): ... fuel=N`` rows joined with ``"; "``.
        side: ``"dealt"`` or ``"taken"`` — which fuel field to fill.
    """
    for chunk in ledger_text.split(";"):
        if not chunk.strip():
            continue
        match = _LEDGER_ROW.match(chunk.strip())
        if match is None:
            continue
        target_id = int(match.group("tank_id"))
        row = rows.setdefault(target_id, _blank_row(target_id))
        if row["target_name"] == "?":
            row["target_name"] = match.group("name")
        if side == "dealt":
            row["dealt_fuel"] = int(match.group("fuel"))
        else:
            row["taken_fuel"] = int(match.group("fuel"))


def _consume_wire_shot(rows: dict[int, EngagementDict], record: RuntimeEventRecordDict) -> None:
    """Book one WIRE record into the rows when it is a targeted shot.

    Args:
        rows: Engagement rows keyed by enemy id.
        record: A ``WIRE``-channel event record.
    """
    match = _WIRE_SHOOT.match(record["message"])
    if match is None:
        return
    target_id = int(match.group(1))
    if target_id == 0:
        # Ground fire (mine clearance, last-position shots) carries
        # id 0 -- not an enemy, not an engagement.
        return
    row = rows.setdefault(target_id, _blank_row(target_id))
    seconds = _timestamp_seconds(record)
    if row["first_shot_s"] is None:
        row["first_shot_s"] = seconds
    row["last_shot_s"] = seconds
    row["shots"] += 1


def _consume_break(
    rows: dict[int, EngagementDict],
    fields: dict[str, str | int | float | bool],
) -> None:
    """Book one ``engagement_break`` diagnostic into the rows.

    Args:
        rows: Engagement rows keyed by enemy id.
        fields: The break record's structured payload.
    """
    target_id = _opt_int(fields, "target_id")
    if target_id is None:
        return
    row = rows.setdefault(target_id, _blank_row(target_id))
    row["breaks"] += 1
    name = _opt_str(fields, "target_name")
    if name is not None:
        row["target_name"] = name


def _consume_deactivation(
    rows: dict[int, EngagementDict],
    fields: dict[str, str | int | float | bool],
    self_id: int | None,
    record: RuntimeEventRecordDict,
) -> None:
    """Book one ``tank_deactivated`` diagnostic into the rows.

    Only OTHER tanks' deactivations arrive on this kind — dispatch
    routes our own death to ``self_deactivated`` before this record
    is ever emitted, so a victim-is-us branch here would be dead
    code (and was, until 2026-08-26: deaths read 0 through both of
    arterial's main-map deaths).

    Args:
        rows: Engagement rows keyed by enemy id.
        fields: The deactivation record's structured payload.
        self_id: Our wire id, or ``None`` while unknown (attribution
            is then impossible and the record books nothing).
        record: The full record (timestamp source for time-to-kill).
    """
    victim_id = _opt_int(fields, "victim_id")
    killer_id = _opt_int(fields, "killer_id")
    if victim_id is None or killer_id is None or self_id is None:
        return
    if killer_id == self_id and victim_id in rows:
        row = rows[victim_id]
        row["outcome"] = "kill"
        first = row["first_shot_s"]
        if first is not None:
            row["seconds_to_kill"] = _timestamp_seconds(record) - first


def _consume_self_death(
    rows: dict[int, EngagementDict],
    fields: dict[str, str | int | float | bool],
) -> None:
    """Book our own ``self_deactivated`` receipt's killer, when named.

    The 0x41 receipt carries ``killer_id``; the fuel-wrap receipt
    (which lands first when both arrive, and wins the dedup) cannot
    name the killer, so those deaths count without an outcome
    attribution — honest, not inferred.

    Args:
        rows: Engagement rows keyed by enemy id.
        fields: The self-death record's structured payload.
    """
    killer_id = _opt_int(fields, "killer_id")
    if killer_id is not None and killer_id in rows:
        rows[killer_id]["outcome"] = "killed_us"


def _resolve_names(rows: dict[int, EngagementDict], records: list[RuntimeEventRecordDict]) -> None:
    """Name every still-unnamed row from the run's identity records.

    Runs after the main pass because an identity record can precede
    the first shot at its tank.

    Args:
        rows: Engagement rows keyed by enemy id.
        records: The full decoded record stream.
    """
    for record in records:
        fields = record["fields"]
        if _opt_str(fields, "diagnostic_kind") != "tank_identity":
            continue
        tank_id = _opt_int(fields, "tank_id")
        name = _opt_str(fields, "name")
        if tank_id in rows and name is not None and rows[tank_id]["target_name"] == "?":
            rows[tank_id]["target_name"] = name


def _sort_key(row: EngagementDict) -> tuple[bool, float]:
    """Order rows by first shot; ledger-only rows sort last, by id.

    Args:
        row: Engagement to order.

    Returns:
        Sort key placing shot-at rows chronologically first.
    """
    first = row["first_shot_s"]
    if first is None:
        return (True, float(row["target_id"]))
    return (False, first)


def build_engagement_ledger(source_path: Path) -> EngagementLedgerDict:
    """Build the fight ledger from one run's events artifact.

    Args:
        source_path: JSONL events path to analyze.

    Returns:
        The run's engagement ledger.
    """
    records = load_event_records(source_path)
    rows: dict[int, EngagementDict] = {}
    self_id: int | None = None
    deaths = 0
    for record in records:
        fields = record["fields"]
        if record["channel"] == "WIRE":
            _consume_wire_shot(rows, record)
            continue
        kind = _opt_str(fields, "diagnostic_kind")
        if kind == "self_alignment_sample" and self_id is None:
            self_id = _opt_int(fields, "belief_tank_id")
        elif kind == "engagement_break":
            _consume_break(rows, fields)
        elif kind == "tank_deactivated":
            _consume_deactivation(rows, fields, self_id, record)
        elif kind == "self_deactivated":
            deaths += 1
            _consume_self_death(rows, fields)
        elif kind == "damage_ledger":
            dealt = _opt_str(fields, "dealt")
            taken = _opt_str(fields, "taken")
            if dealt is not None:
                _apply_damage_ledger(rows, dealt, side="dealt")
            if taken is not None:
                _apply_damage_ledger(rows, taken, side="taken")
    _resolve_names(rows, records)
    ordered = sorted(rows.values(), key=_sort_key)
    # Rows we never fired at (an ally's stray hits, an attacker we
    # escaped) are informational -- only fights WE prosecuted can lose
    # a damage trade.
    negative = [
        row for row in ordered if row["shots"] > 0 and row["taken_fuel"] > row["dealt_fuel"]
    ]
    return EngagementLedgerDict(
        source_path=str(source_path),
        self_id=self_id,
        engagements=ordered,
        kills=sum(1 for row in ordered if row["outcome"] == "kill"),
        deaths=deaths,
        negative_trades=len(negative),
        post_break_negative_trades=sum(1 for row in negative if row["breaks"] > 0),
    )


def _row_line(row: EngagementDict) -> str:
    """Render one engagement as a fixed-width report line.

    Args:
        row: Engagement to render.

    Returns:
        One aligned text line.
    """
    t2k = f"{row['seconds_to_kill']:.0f}s" if row["seconds_to_kill"] is not None else "-"
    trade = row["dealt_fuel"] - row["taken_fuel"]
    return (
        f"{row['target_id']:>5}  {row['target_name']:<14.14}"
        f" {row['shots']:>5}  {row['breaks']:>6}  {row['outcome']:<9}"
        f" {t2k:>5}  {row['dealt_fuel']:>6}  {row['taken_fuel']:>6}  {trade:>+6}"
    )


def render_engagement_ledger(ledger: EngagementLedgerDict) -> str:
    """Render the fight ledger as a human-readable report.

    Args:
        ledger: Ledger to render.

    Returns:
        Multi-line report text.
    """
    self_text = str(ledger["self_id"]) if ledger["self_id"] is not None else "unknown"
    lines = [
        "=== ENGAGEMENTS ===",
        f"source: {ledger['source_path']}",
        (
            f"self id: {self_text} | engagements {len(ledger['engagements'])}"
            f" | kills {ledger['kills']} | deaths {ledger['deaths']}"
        ),
        "   id  name            shots  breaks  outcome     t2k   dealt   taken   trade",
    ]
    lines.extend(_row_line(row) for row in ledger["engagements"])
    if ledger["negative_trades"] > 0:
        losers = [
            f"{row['target_name']}({row['target_id']})"
            f" taken {row['taken_fuel']} > dealt {row['dealt_fuel']}"
            + (" AFTER A BREAK" if row["breaks"] > 0 else "")
            for row in ledger["engagements"]
            if row["shots"] > 0 and row["taken_fuel"] > row["dealt_fuel"]
        ]
        lines.append(f"negative trades: {ledger['negative_trades']} -- " + "; ".join(losers))
    if ledger["post_break_negative_trades"] > 0:
        lines.append(
            f"FLAG: {ledger['post_break_negative_trades']} engagement(s) lost the damage"
            " trade after a break -- the solvency law walked away and shots kept flowing"
        )
    return "\n".join(lines)


def main() -> int:
    """Run the ``tankpit-engagements`` CLI entrypoint.

    One path argument (default ``runs/bot/latest.events.jsonl``)
    analyzes that run.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    setup_rich_logging(level="INFO")
    args = list(_test_hooks.get_argv())[1:]
    source = Path(args[0]) if args else _DEFAULT_SOURCE
    log.info("%s", render_engagement_ledger(build_engagement_ledger(source)))
    return 0


__all__ = [
    "EngagementDict",
    "EngagementLedgerDict",
    "build_engagement_ledger",
    "main",
    "render_engagement_ledger",
]
