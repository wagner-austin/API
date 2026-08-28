"""Per-run session scorecard: time budget, combat outcome, fuel, dot ledger.

Distils a filled :class:`ScorecardAccumulatorDict` into the
:class:`SessionScorecardDict` the issue report consumes. The two
halves either side of it are
:mod:`tankpit_bot.diagnostics.session_scorecard_accumulator` (ingest)
and :mod:`tankpit_bot.diagnostics.session_scorecard_render` (output).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from tankpit_bot.diagnostics.issue_report_types import (
    FuelLowWaterEpisodeDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
    TeleportSpendRecordDict,
    make_unsampled_inventory_counts,
)
from tankpit_bot.diagnostics.session_scorecard_types import (
    FuelSampleRecordDict,
    ScorecardAccumulatorDict,
)

# The fuel-critical band: combat needs ~10 fuel per shot and teleports
# cost 6 per tile, so dipping below this means the session nearly
# stranded itself.
_FUEL_FLOOR_THRESHOLD = 100


def _budget_sort_key(record: StateBudgetRecordDict) -> tuple[int, str]:
    """Sort key for the state budget: descending seconds, then name.

    Args:
        record: State budget record to key.

    Returns:
        Tuple of ``(-seconds, state)``.
    """
    return (-record["seconds"], record["state"])


_IDLE_STATE = "IDLE"

_MAP_OPEN_LEG = "IDLE/map_open"
"""Budget bucket for the IDLE stretches that were opening the map."""

_SCOPE_SHIFT_LEG = "IDLE/scope_shift"
"""Budget bucket for the IDLE stretches that steered the scope.

Neither of these is a bot state: the HFSM has none for ``map_open`` or
``scope_shift`` because both are COMMANDS, and a tick that dispatches one
transitions nowhere. That is the whole reason their seconds landed in
IDLE. The names keep the origin visible instead of inventing states the
state machine does not have.

Together they accounted for every second of the 16 IDLE seconds in run
20260812-194435 -- 10 opening the map for a teleport, 6 steering the quad
sweep -- leaving zero seconds in which the bot was actually idle."""


def _idle_bucket(
    previous_moment: datetime,
    moment: datetime,
    map_open_completions: list[datetime],
    scope_shift_sends: list[datetime],
) -> str:
    """Return the bucket an IDLE stretch belongs to.

    The two markers sit at opposite ends of their stretch, so their
    boundary tests differ and neither is arbitrary. A map open is seen
    COMPLETING, which is what releases IDLE, so its timestamp lands on
    the stretch's closing edge: ``(previous, moment]``. A scope shift is
    seen being SENT by the very tick that entered IDLE, so its timestamp
    lands on the opening edge: ``[previous, moment)``.

    Args:
        previous_moment: When the IDLE stretch began.
        moment: When it ended.
        map_open_completions: Map-open completion moments.
        scope_shift_sends: Scope-shift dispatch moments.

    Returns:
        The map-open bucket, the scope-shift bucket, or plain ``IDLE``.
    """
    if any(previous_moment < completion <= moment for completion in map_open_completions):
        return _MAP_OPEN_LEG
    if any(previous_moment <= send < moment for send in scope_shift_sends):
        return _SCOPE_SHIFT_LEG
    return _IDLE_STATE


def _build_state_budget(
    transitions: list[tuple[str, str]],
    map_open_completions_at: list[str],
    scope_shift_sends_at: list[str],
) -> list[StateBudgetRecordDict]:
    """Sum seconds spent in each bot state from STATE-channel transitions.

        The interval between consecutive ``A -> B`` transitions is credited
        to the EARLIER transition's destination -- the state the bot was
        actually in during that interval. Non-transition STATE lines (the
        initial bare state announcement) carry no interval and are skipped.
        Each interval is also one VISIT to its state, so the per-state
        stretch count and longest single visit fall out of the same walk --
        that pair distinguishes tick-boundary residue (many short visits)
        from a stall (one long visit) at no extra cost.

    An IDLE stretch that was dispatching a command is credited to that
        command instead -- see :func:`_idle_bucket` and
        :data:`_SCOPE_SHIFT_LEG`.

        Args:
            transitions: ``(timestamp, message)`` pairs in stream order.
            map_open_completions_at: Timestamps of completed map opens.
            scope_shift_sends_at: Timestamps of dispatched scope shifts.

        Returns:
            Per-state totals sorted by descending seconds then state name.
    """
    completions = [datetime.fromisoformat(moment) for moment in map_open_completions_at]
    sends = [datetime.fromisoformat(moment) for moment in scope_shift_sends_at]
    totals: Counter[str] = Counter()
    visits: Counter[str] = Counter()
    longest: dict[str, int] = {}
    previous_state = ""
    previous_moment: datetime | None = None
    for timestamp, message in transitions:
        if " -> " not in message:
            continue
        _, _, destination = message.partition(" -> ")
        moment = datetime.fromisoformat(timestamp)
        if previous_moment is not None:
            interval = int((moment - previous_moment).total_seconds())
            bucket = previous_state
            if previous_state == _IDLE_STATE:
                bucket = _idle_bucket(previous_moment, moment, completions, sends)
            totals[bucket] += interval
            visits[bucket] += 1
            longest[bucket] = max(longest.get(bucket, 0), interval)
        previous_state = destination
        previous_moment = moment
    records = [
        StateBudgetRecordDict(
            state=state,
            seconds=seconds,
            stretches=visits[state],
            max_seconds=longest[state],
        )
        for state, seconds in totals.items()
    ]
    records.sort(key=_budget_sort_key)
    return records


def _build_teleport_spend(
    spent: dict[str, int],
    drops: dict[str, int],
) -> tuple[list[TeleportSpendRecordDict], int]:
    """Shape the accumulated per-state teleport spend into sorted rows.

    Args:
        spent: Per-``bot_state`` fuel totals from the WORLD receipts.
        drops: Per-``bot_state`` receipt counts.

    Returns:
        Tuple of per-state spend rows (descending fuel, then state)
        and the total spend.
    """
    records = [
        TeleportSpendRecordDict(bot_state=state, drops=drops[state], fuel_spent=fuel)
        for state, fuel in spent.items()
    ]
    records.sort(key=lambda record: (-record["fuel_spent"], record["bot_state"]))
    return records, sum(spent.values())


def _episode_cause(
    samples: list[FuelSampleRecordDict],
    entry_index: int,
    min_index: int,
) -> tuple[str, int, str]:
    """Find the largest fuel drop on the way down into one episode.

    Args:
        samples: Context-stamped fuel samples in stream order.
        entry_index: Index of the first below-threshold sample.
        min_index: Index of the episode's minimum-fuel sample.

    Returns:
        Tuple of ``(cause_kind, cause_drop, cause_state)``. When no
        positive drop exists in the window (a session that STARTED
        below threshold at its minimum), the entry sample's own
        context is returned with a drop of 0.
    """
    best_drop = 0
    best_index = entry_index
    for index in range(max(entry_index, 1), min_index + 1):
        drop = samples[index - 1]["fuel"] - samples[index]["fuel"]
        if drop > best_drop:
            best_drop = drop
            best_index = index
    chosen = samples[best_index]
    return chosen["in_flight"], best_drop, chosen["bot_state"]


def _build_low_water_episodes(
    samples: list[FuelSampleRecordDict],
    threshold: int,
) -> list[FuelLowWaterEpisodeDict]:
    """Split the fuel trajectory into below-threshold episodes.

    Args:
        samples: Context-stamped fuel samples in stream order.
        threshold: Danger line; samples strictly below it are "low".

    Returns:
        One record per maximal contiguous below-threshold run, in
        stream order.
    """
    episodes: list[FuelLowWaterEpisodeDict] = []
    index = 0
    while index < len(samples):
        if samples[index]["fuel"] >= threshold:
            index += 1
            continue
        end = index
        min_index = index
        while end + 1 < len(samples) and samples[end + 1]["fuel"] < threshold:
            end += 1
            if samples[end]["fuel"] < samples[min_index]["fuel"]:
                min_index = end
        cause_kind, cause_drop, cause_state = _episode_cause(samples, index, min_index)
        first = datetime.fromisoformat(samples[index]["timestamp"])
        last = datetime.fromisoformat(samples[end]["timestamp"])
        recovery = samples[end + 1] if end + 1 < len(samples) else None
        episodes.append(
            FuelLowWaterEpisodeDict(
                start_timestamp=samples[index]["timestamp"],
                end_timestamp=samples[end]["timestamp"],
                duration_seconds=int((last - first).total_seconds()),
                entry_fuel=samples[index - 1]["fuel"] if index > 0 else -1,
                min_fuel=samples[min_index]["fuel"],
                cause_kind=cause_kind,
                cause_drop=cause_drop,
                cause_state=cause_state,
                recovery_fuel=recovery["fuel"] if recovery is not None else -1,
                recovery_kind=recovery["in_flight"] if recovery is not None else "",
            )
        )
        index = end + 1
    return episodes


def build_session_scorecard(accumulator: ScorecardAccumulatorDict) -> SessionScorecardDict:
    """Distill the per-run outcome scorecard from the accumulator.

    Args:
        accumulator: Fully routed scorecard accumulator.

    Returns:
        Session scorecard.
    """
    duration_seconds = 0
    if accumulator["first_timestamp"] and accumulator["last_timestamp"]:
        first = datetime.fromisoformat(accumulator["first_timestamp"])
        last = datetime.fromisoformat(accumulator["last_timestamp"])
        duration_seconds = int((last - first).total_seconds())
    fuel_samples = accumulator["fuel_samples"]
    fuel_values = [sample["fuel"] for sample in fuel_samples]
    low_water_threshold = (
        accumulator["max_escape_floor"]
        if accumulator["max_escape_floor"] > 0
        else _FUEL_FLOOR_THRESHOLD
    )
    teleport_spend, teleport_spend_total = _build_teleport_spend(
        accumulator["teleport_spend_fuel"],
        accumulator["teleport_spend_drops"],
    )
    inventory_samples = accumulator["inventory_samples"]
    return SessionScorecardDict(
        duration_seconds=duration_seconds,
        state_budget=_build_state_budget(
            accumulator["state_transitions"],
            accumulator["map_open_completions_at"],
            accumulator["scope_shift_sends_at"],
        ),
        kills=accumulator["kills"],
        shots=accumulator["shots"],
        combat_misses=accumulator["combat_misses"],
        tank_damage_changes=accumulator["tank_damage_changes"],
        fuel_min=min(fuel_values) if fuel_values else -1,
        fuel_last=fuel_values[-1] if fuel_values else -1,
        fuel_sample_count=len(fuel_values),
        inventory_first=(
            inventory_samples[0] if inventory_samples else make_unsampled_inventory_counts()
        ),
        inventory_last=(
            inventory_samples[-1] if inventory_samples else make_unsampled_inventory_counts()
        ),
        inventory_sample_count=len(inventory_samples),
        equipment_gain_events=accumulator["equipment_gain_events"],
        equipment_gained=accumulator["equipment_gained"],
        scans_extra=accumulator["scans_extra"],
        scans_builtin=accumulator["scans_builtin"],
        physics_divergences=accumulator["physics_divergences"],
        action_outcome_counts=dict(sorted(accumulator["action_outcome_counts"].items())),
        fuel_low_water_threshold=low_water_threshold,
        fuel_low_water_episodes=_build_low_water_episodes(fuel_samples, low_water_threshold),
        teleport_spend=teleport_spend,
        teleport_spend_total=teleport_spend_total,
        ledger_teleport_spend_min=accumulator["ledger_teleport_spend_min"],
        ledger_teleport_spend_max=accumulator["ledger_teleport_spend_max"],
        ledger_shot_singles=accumulator["ledger_shot_singles"],
        ledger_shot_duals=accumulator["ledger_shot_duals"],
        ledger_shot_homings=accumulator["ledger_shot_homings"],
        career_destroyed_last=accumulator["career_destroyed_last"],
        career_deactivated_last=accumulator["career_deactivated_last"],
        career_score_last=accumulator["career_score_last"],
        career_playtime_seconds_last=accumulator["career_playtime_seconds_last"],
        container_pickups_full=accumulator["container_pickups_full"],
        container_pickups_partial=accumulator["container_pickups_partial"],
    )


__all__ = [
    "build_session_scorecard",
]
