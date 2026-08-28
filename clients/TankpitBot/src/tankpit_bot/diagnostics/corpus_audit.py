"""Corpus audit: diff every analyzer against wire ground truth, per run.

Born from the 2026-08-28 dead-diagnostic audit's second lesson: every
recent reporting bug survived because nothing DIFFED the reports
against ground truth -- deaths read 0 through three real deaths, the
digest undercounted coordinate-aimed kills, and each was caught only
when an external fact collided with a report. This module makes that
collision a standing instrument: for each archived events artifact it
independently recounts the wire receipts (0x41 kill attributions,
``self_deactivated`` death receipts), runs BOTH analyzers over the
same records, and flags every disagreement -- plus the invariants the
one-off sweep checked by hand (beat discipline, radar-book balance,
tombstone re-aims).

CLI: ``tankpit-corpus-audit [root]`` sweeps ``runs/bot`` (or the given
directory) and prints one line per flagged run plus the corpus
summary. Exit code 1 when any run is flagged, so the sweep can gate.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from platform_core.logging import get_logger
from platform_core.rich_logging import setup_rich_logging
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.run_digest import build_run_digest
from tankpit_bot.diagnostics.session_scorecard import build_session_scorecard
from tankpit_bot.diagnostics.session_scorecard_accumulator import route_scorecard_record
from tankpit_bot.diagnostics.session_scorecard_types import new_scorecard_accumulator
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    require_bool_field,
    require_int_field,
)

log = get_logger(__name__)

#: A repeated teleport aim at a tile that displaced within this window
#: is the orbit signature the displacement-tombstone doctrine kills.
REAIM_WINDOW_S = 30.0

#: The radar book may sit one press out at an inventory-sample boundary
#: (the sample and the dispatch race by one message); anything larger
#: is a tracking defect.
RADAR_DRIFT_TOLERANCE = 1


class RunAuditDict(TypedDict):
    """Ground-truth diff and invariant flags for one events artifact.

    Attributes:
        source: The events artifact audited.
        wire_kills: Independent recount of ``tank_deactivated``
            receipts naming this session's tank as killer.
        wire_deaths: Independent recount of ``self_deactivated``
            receipts.
        digest_kills: What the run digest computed.
        digest_deaths: What the run digest computed.
        scorecard_kills: What the session scorecard computed.
        fast_shots: Shoot dispatches under 2 s after the previous
            shoot -- the serve-cadence law says a correct dispatcher
            produces zero.
        reaims_within_30s: Teleport aims repeated at a tile that
            displaced within the re-aim window.
        radar_drift: Final extra-radar count minus the running
            expectation (first sample, plus gains, minus
            extra-consuming dispatches, death-penalized per the
            wire-verified halving law); 0 when the tracking is sound.
        physics_divergences: ``physics_divergence`` receipts (the fuel
            book's own residual detector).
        kind_counts: Every diagnostic kind observed, tallied.
        flags: Human-readable violation lines; empty when the run is
            clean.
    """

    source: str
    wire_kills: int
    wire_deaths: int
    digest_kills: int
    digest_deaths: int
    scorecard_kills: int
    fast_shots: int
    reaims_within_30s: int
    radar_drift: int
    physics_divergences: int
    kind_counts: dict[str, int]
    flags: list[str]


class CorpusAuditDict(TypedDict):
    """Whole-corpus roll-up.

    Attributes:
        root: Directory swept.
        runs_audited: Artifacts audited.
        runs_flagged: Artifacts with at least one flag.
        runs_skipped: Artifacts that would not audit (empty, or an
            archive era missing a required field), named so a shrinking
            corpus is never silent.
        skipped: The unreadable artifacts with their reasons.
        kind_counts: Corpus-wide diagnostic-kind tallies -- an
            emitted kind absent here after enough runs is either
            dormant or dead, and worth a look either way.
        audits: Per-run audits, flagged runs only.
    """

    root: str
    runs_audited: int
    runs_flagged: int
    runs_skipped: int
    skipped: list[str]
    kind_counts: dict[str, int]
    audits: list[RunAuditDict]


def _ts_seconds(timestamp: str) -> float:
    """Convert an event timestamp to epoch seconds.

    Args:
        timestamp: ISO-format timestamp from an event record.

    Returns:
        Epoch seconds.
    """
    return datetime.fromisoformat(timestamp).timestamp()


class _WireTruth:
    """Single-pass recount of the raw signals the analyzers summarize."""

    def __init__(self) -> None:
        """Zero every tally."""
        self.self_id = -1
        self.kills = 0
        self.deaths = 0
        self.fast_shots = 0
        self.reaims = 0
        self.divergences = 0
        self.radar_expected = -1
        self.radar_last = -1
        self.radar_max_seen = -1
        self.radar_mismatch = 0
        self.kind_counts: Counter[str] = Counter()
        self._last_shot_s: float | None = None
        self._displaced_at: dict[tuple[int, int], float] = {}

    def _consume_combat_kind(self, record: RuntimeEventRecordDict, kind: str, t_s: float) -> bool:
        """Fold one combat-receipt DIAGNOSTIC into the tallies.

        Args:
            record: The event record.
            kind: Its ``diagnostic_kind``.
            t_s: Event epoch seconds.

        Returns:
            True when ``kind`` matched a combat receipt.
        """
        fields = record["fields"]
        if kind == "tank_identity":
            if self.self_id == -1:
                self.self_id = require_int_field(fields, "tank_id")
            return True
        if kind == "tank_deactivated":
            # Pre-fleet archives lack killer_id; the .get comparison
            # counts them never, mirroring the scorecard's split.
            if self.self_id != -1 and fields.get("killer_id") == self.self_id:
                self.kills += 1
            return True
        if kind == "self_deactivated":
            self.deaths += 1
            # Death penalty on the radar expectation (wire-verified:
            # ceil(n/2) on tank kills, zero on the mine sentinel).
            # Archives predate the is_mine_kill field; there the
            # rebased mine-team killer ids 0-3 identify a mine death
            # (real tank ids sit in the hundreds).
            if self.radar_expected != -1:
                killer = fields.get("killer_id")
                mine = fields.get("is_mine_kill") is True or (
                    isinstance(killer, int) and 0 <= killer <= 3
                )
                self.radar_expected = 0 if mine else (self.radar_expected + 1) // 2
            return True
        if kind == "teleport_displacement":
            key = (
                require_int_field(fields, "requested_x"),
                require_int_field(fields, "requested_y"),
            )
            previous = self._displaced_at.get(key)
            if previous is not None and t_s - previous <= REAIM_WINDOW_S:
                self.reaims += 1
            self._displaced_at[key] = t_s
            return True
        return False

    def _consume_resource_kind(self, record: RuntimeEventRecordDict, kind: str) -> None:
        """Fold one resource-ledger DIAGNOSTIC into the tallies.

        Args:
            record: The event record.
            kind: Its ``diagnostic_kind``.
        """
        fields = record["fields"]
        if kind == "physics_divergence":
            self.divergences += 1
        elif kind == "radar_dispatch":
            if require_bool_field(fields, "uses_extra") and self.radar_expected != -1:
                self.radar_expected -= 1
        elif kind == "equipment_gain":
            gained = require_int_field(fields, "radar")
            if self.radar_expected != -1:
                # The server clamps gains at the rank cap. Caps live on
                # the ``20 + 5 * rank`` ladder, so the highest observed
                # sample rounded UP to the ladder is the cap estimate
                # (samples are dense -- every count change emits its
                # own 0x49). A run that sat at cap 25 while gains kept
                # arriving overcounted +24 before this clamp existed
                # (artax 2026-08-26 08:48).
                cap_estimate = max(20, (self.radar_max_seen + 4) // 5 * 5)
                self.radar_expected = min(
                    self.radar_expected + gained,
                    max(cap_estimate, self.radar_expected),
                )
                # Gain and sample are emitted atomically by the same
                # update, but pre-2026-08-28 archives wrote the sample
                # FIRST -- a run ending on that pair left a stale
                # +gain mismatch. Recomputing here retro-explains it;
                # in the fixed order the following sample lands the
                # same answer.
                self.radar_mismatch = self.radar_last - self.radar_expected
        elif kind == "inventory_sample":
            radar = require_int_field(fields, "radar")
            if self.radar_expected == -1:
                self.radar_expected = radar
            self.radar_last = radar
            self.radar_max_seen = max(self.radar_max_seen, radar)
            # Drift is evaluated AT samples -- the only moments the
            # wire states the count -- so trailing dispatches whose
            # confirming 0x49 never arrived before teardown cannot
            # fake a drift (artax 08-26 08:48 read +7 from exactly
            # that end-of-stream skew).
            self.radar_mismatch = radar - self.radar_expected

    def consume(self, record: RuntimeEventRecordDict) -> None:
        """Fold one event record into the tallies.

        Args:
            record: The event record.
        """
        t_s = _ts_seconds(record["timestamp"])
        kind = record["fields"].get("diagnostic_kind")
        if isinstance(kind, str):
            self.kind_counts[kind] += 1
            if not self._consume_combat_kind(record, kind, t_s):
                self._consume_resource_kind(record, kind)
        elif record["channel"] == "WIRE" and record["message"].startswith("shoot("):
            if self._last_shot_s is not None and t_s - self._last_shot_s < 2.0:
                self.fast_shots += 1
            self._last_shot_s = t_s

    def radar_drift(self) -> int:
        """Return the radar book's prediction error at the last sample.

        The expectation runs event-by-event from the first sample:
        gains add (cap-clamped at the highest observed count), paid
        dispatches subtract, and deaths apply the wire-verified
        penalty (ceil-halved, or zeroed on a mine kill) -- so a
        death-run no longer fakes a drift.

        Returns:
            The last sample minus the expectation at that sample, or
            0 when the run had no inventory samples.
        """
        return self.radar_mismatch


def _collect_flags(audit: RunAuditDict) -> list[str]:
    """Derive the violation lines for one audited run.

    Args:
        audit: The audit with every tally filled in.

    Returns:
        Human-readable flag lines, empty when clean.
    """
    flags: list[str] = []
    if audit["digest_kills"] != audit["wire_kills"]:
        flags.append(
            f"digest kills={audit['digest_kills']} disagrees with "
            f"wire recount {audit['wire_kills']}"
        )
    if audit["scorecard_kills"] != audit["wire_kills"]:
        flags.append(
            f"scorecard kills={audit['scorecard_kills']} disagrees with "
            f"wire recount {audit['wire_kills']}"
        )
    if audit["digest_deaths"] != audit["wire_deaths"]:
        flags.append(
            f"digest deaths={audit['digest_deaths']} disagrees with "
            f"wire recount {audit['wire_deaths']}"
        )
    if audit["fast_shots"] > 0:
        flags.append(
            f"{audit['fast_shots']} shoot dispatches under the 2 s serve "
            "beat -- the dispatcher is out-running the cadence law"
        )
    if abs(audit["radar_drift"]) > RADAR_DRIFT_TOLERANCE:
        flags.append(
            f"radar book drift {audit['radar_drift']} -- inventory "
            "tracking disagrees with first+gains-spends"
        )
    if audit["reaims_within_30s"] > 0:
        flags.append(
            f"{audit['reaims_within_30s']} re-aims at displaced tiles "
            "within 30 s -- the displacement tombstone should forbid these"
        )
    return flags


def audit_events_artifact(source_path: Path) -> RunAuditDict:
    """Audit one events artifact against its own wire receipts.

    Args:
        source_path: JSONL events path.

    Returns:
        The audit, with ``flags`` naming every disagreement.

    Raises:
        ValueError: If the artifact holds no events.
    """
    records = load_event_records(source_path)
    if not records:
        raise ValueError(f"no events in {source_path}")
    truth = _WireTruth()
    accumulator = new_scorecard_accumulator()
    for record in records:
        truth.consume(record)
        route_scorecard_record(record, accumulator)
    digest = build_run_digest(source_path)
    scorecard = build_session_scorecard(accumulator)
    audit = RunAuditDict(
        source=str(source_path),
        wire_kills=truth.kills,
        wire_deaths=truth.deaths,
        digest_kills=digest["kills"],
        digest_deaths=digest["deaths"],
        scorecard_kills=scorecard["kills"],
        fast_shots=truth.fast_shots,
        reaims_within_30s=truth.reaims,
        radar_drift=truth.radar_drift(),
        physics_divergences=truth.divergences,
        kind_counts=dict(sorted(truth.kind_counts.items())),
        flags=[],
    )
    audit["flags"] = _collect_flags(audit)
    return audit


def audit_corpus(root: Path) -> CorpusAuditDict:
    """Audit every events artifact under a directory.

    Args:
        root: Directory holding ``*.events.jsonl`` artifacts (swept
            recursively; ``latest.events.jsonl`` mirrors are skipped
            as duplicates of their stamped twins).

    Returns:
        The corpus roll-up carrying only the flagged runs in full.
    """
    kind_counts: Counter[str] = Counter()
    audits: list[RunAuditDict] = []
    skipped: list[str] = []
    runs_audited = 0
    for source_path in sorted(root.rglob("*.events.jsonl")):
        if source_path.name == "latest.events.jsonl":
            continue
        try:
            audit = audit_events_artifact(source_path)
        except (KeyError, ValueError) as error:
            # KeyError: an archive era missing a field a strict reader
            # (this recount or the scorecard accumulator) requires.
            skipped.append(f"{source_path}: {error}")
            continue
        runs_audited += 1
        kind_counts.update(audit["kind_counts"])
        if audit["flags"]:
            audits.append(audit)
    return CorpusAuditDict(
        root=str(root),
        runs_audited=runs_audited,
        runs_flagged=len(audits),
        runs_skipped=len(skipped),
        skipped=skipped,
        kind_counts=dict(sorted(kind_counts.items())),
        audits=audits,
    )


def render_corpus_audit(corpus: CorpusAuditDict) -> str:
    """Render the corpus audit as the human table.

    Args:
        corpus: Computed corpus audit.

    Returns:
        Multi-line report text.
    """
    lines = [
        "=== CORPUS AUDIT ===",
        f"root       {corpus['root']}",
        f"runs       {corpus['runs_audited']} audited, {corpus['runs_flagged']} flagged, "
        f"{corpus['runs_skipped']} skipped",
    ]
    for reason in corpus["skipped"]:
        lines.append(f"  SKIPPED  {reason}")
    for audit in corpus["audits"]:
        lines.append(f"run        {audit['source']}")
        for flag in audit["flags"]:
            lines.append(f"  FLAG     {flag}")
    lines.append("kinds      observed across the corpus:")
    for kind, count in corpus["kind_counts"].items():
        lines.append(f"           {kind} x{count}")
    return "\n".join(lines)


def main() -> int:
    """Run the ``tankpit-corpus-audit`` CLI entrypoint.

    Returns:
        0 when every audited run is clean, 1 when any run is flagged.
    """
    setup_rich_logging(level="INFO")
    argv = _test_hooks.get_argv()
    root = Path(argv[1]) if len(argv) > 1 else Path("runs") / "bot"
    corpus = audit_corpus(root)
    log.info("%s", render_corpus_audit(corpus))
    return 1 if corpus["runs_flagged"] else 0


__all__ = [
    "RADAR_DRIFT_TOLERANCE",
    "REAIM_WINDOW_S",
    "CorpusAuditDict",
    "RunAuditDict",
    "audit_corpus",
    "audit_events_artifact",
    "main",
    "render_corpus_audit",
]
