"""Human-fight episodes and per-event fight rows from one capture.

Built 2026-08-03 after the nope-fight autopsy took four read passes:
every summary layer hid one load-bearing fact (the monitor greps hid
the humans, the "rejected" label hid successful drinks, the raw fuel
column hid whose fuel went where, the shot log hid that the bot never
moved). This module makes the wire's play-by-play a first-class typed
product of the EXISTING extraction — it consumes
:func:`tankpit_bot.validate.shadow_timeline.extract_shadow_timeline`
and adds no second decode path.

Two products:

* :func:`extract_human_episodes` — one record per human opponent with
  exactly-computable engagement facts (shot counts, kills, deaths,
  and the stationary-streak metric that exposed the turret behavior:
  six consecutive shots from one tile while under fire).
* :func:`render_fight_rows` — the chronological play-by-play for a
  window, one typed row per wire event, with self fuel deltas
  attributed by cause.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot.protocol.naming import is_human_name
from tankpit_bot.validate.shadow_timeline import ShadowTimelineDict, ShotEventDict

FuelCause = Literal[
    "own_shot",
    "incoming_single_or_homing",
    "incoming_dual",
    "gain",
    "spend_or_multi",
    "walk_or_misc",
]
"""Attribution of one self fuel delta, from the measured cost table:
own shots debit 10, singles/homings hit for 45, duals for 90; larger
debits are own teleports or multi-event syncs; credits are pickups or
refunds; the remainder is walking."""


class HumanEpisodeDict(TypedDict):
    """One human opponent's engagement facts for a session.

    The window is the human's own first-to-last shot span; the
    stationary streak counts consecutive OWN shots fired from one
    unchanged source tile inside that window — the turret metric
    (nope-fight ground truth: streak 6 while taking a hit per tick).
    """

    tank_id: int
    name: str
    first_shot_ms: int
    last_shot_ms: int
    shots_by_human: int
    our_shots_in_window: int
    max_stationary_streak: int
    kills_of_human: int
    deaths_to_human: int


class FightRowDict(TypedDict):
    """One rendered play-by-play row.

    ``kind`` tags the wire family; ``actor`` names the participant
    (``self``, the tank id, or the human's registry name when known);
    ``description`` is the human-readable cell built by the typed
    renderers below.
    """

    timestamp_ms: int
    kind: Literal["shot", "fuel", "position", "kill"]
    actor: str
    description: str


def classify_fuel_delta(delta: int) -> FuelCause:
    """Attribute one self fuel delta to its measured cause.

    Args:
        delta: Fuel change between consecutive absolute readings.

    Returns:
        The cause bucket for the delta.
    """
    if delta == -10:
        return "own_shot"
    if delta == -45:
        return "incoming_single_or_homing"
    if delta == -90:
        return "incoming_dual"
    if delta > 0:
        return "gain"
    if delta < -90:
        return "spend_or_multi"
    return "walk_or_misc"


def _actor_name(timeline: ShadowTimelineDict, tank_id: int) -> str:
    """Render one tank's display label.

    Args:
        timeline: The extracted session timeline.
        tank_id: The tank to label.

    Returns:
        ``self`` for the session's own tank, the registry name when
        known, else ``tank <id>``.
    """
    if tank_id == timeline["self_id"]:
        return "self"
    name = timeline["names"].get(tank_id, "")
    if name != "":
        return name
    return f"tank {tank_id}"


def _stationary_streak(own_shots: list[ShotEventDict]) -> int:
    """Longest run of consecutive own shots from one unchanged tile.

    Args:
        own_shots: The session's own shots inside one episode window,
            in wire order.

    Returns:
        The maximum consecutive-same-source-tile run length (zero for
        an empty window).
    """
    best = 0
    run = 0
    last_tile: tuple[int, int] | None = None
    for shot in own_shots:
        tile = (shot["source_x"], shot["source_y"])
        run = run + 1 if tile == last_tile else 1
        last_tile = tile
        best = max(best, run)
    return best


def extract_human_episodes(timeline: ShadowTimelineDict) -> list[HumanEpisodeDict]:
    """Summarize every human opponent's engagement in one session.

    A human is any tank OTHER than the session's own whose registry
    name classifies human
    (:func:`tankpit_bot.protocol.naming.is_human_name`) — the first
    real-capture run listed Artax as its own opponent. A human with
    zero shots still yields an episode when the session records a
    kill involving them — a greeted bystander with neither yields
    none (no engagement to describe).

    Args:
        timeline: The extracted session timeline.

    Returns:
        Episodes in first-shot order (kill-only episodes carry a zero
        window and sort first).
    """
    self_id = timeline["self_id"]
    episodes: list[HumanEpisodeDict] = []
    humans = sorted(
        tank_id
        for tank_id, name in timeline["names"].items()
        if is_human_name(name) and tank_id != self_id
    )
    for human_id in humans:
        their_shots = [s for s in timeline["shots"] if s["shooter_id"] == human_id]
        kills_of_human = sum(
            1 for k in timeline["kills"] if k["victim_id"] == human_id and k["killer_id"] == self_id
        )
        deaths_to_human = sum(
            1 for k in timeline["kills"] if k["victim_id"] == self_id and k["killer_id"] == human_id
        )
        if not their_shots and kills_of_human == 0 and deaths_to_human == 0:
            continue
        first_ms = their_shots[0]["timestamp_ms"] if their_shots else 0
        last_ms = their_shots[-1]["timestamp_ms"] if their_shots else 0
        our_shots = [
            s
            for s in timeline["shots"]
            if s["shooter_id"] == self_id and first_ms <= s["timestamp_ms"] <= last_ms
        ]
        episodes.append(
            HumanEpisodeDict(
                tank_id=human_id,
                name=timeline["names"][human_id],
                first_shot_ms=first_ms,
                last_shot_ms=last_ms,
                shots_by_human=len(their_shots),
                our_shots_in_window=len(our_shots),
                max_stationary_streak=_stationary_streak(our_shots),
                kills_of_human=kills_of_human,
                deaths_to_human=deaths_to_human,
            )
        )
    episodes.sort(key=lambda episode: episode["first_shot_ms"])
    return episodes


def _shot_rows(timeline: ShadowTimelineDict, start_ms: int, end_ms: int) -> list[FightRowDict]:
    """Render every in-window shot.

    Args:
        timeline: The extracted session timeline.
        start_ms: Inclusive window start.
        end_ms: Inclusive window end.

    Returns:
        One row per 0x53 in the window.
    """
    return [
        FightRowDict(
            timestamp_ms=shot["timestamp_ms"],
            kind="shot",
            actor=_actor_name(timeline, shot["shooter_id"]),
            description=(
                f"shot ({shot['source_x']},{shot['source_y']}) -> "
                f"({shot['target_x']},{shot['target_y']}) weapon={shot['weapon']}"
            ),
        )
        for shot in timeline["shots"]
        if start_ms <= shot["timestamp_ms"] <= end_ms
    ]


def _fuel_rows(timeline: ShadowTimelineDict, start_ms: int, end_ms: int) -> list[FightRowDict]:
    """Render every in-window self fuel delta with its cause.

    The previous reading tracks across the whole session so a window
    opening mid-fight still attributes its first delta correctly.

    Args:
        timeline: The extracted session timeline.
        start_ms: Inclusive window start.
        end_ms: Inclusive window end.

    Returns:
        One row per changed absolute self fuel reading in the window.
    """
    rows: list[FightRowDict] = []
    previous_fuel: int | None = None
    for sync in timeline["syncs"]:
        if sync["tank_id"] != timeline["self_id"]:
            continue
        fuel = sync["fuel"]
        if fuel is None:
            continue
        in_window = start_ms <= sync["timestamp_ms"] <= end_ms
        if previous_fuel is not None and fuel != previous_fuel and in_window:
            delta = fuel - previous_fuel
            rows.append(
                FightRowDict(
                    timestamp_ms=sync["timestamp_ms"],
                    kind="fuel",
                    actor="self",
                    description=(f"fuel {previous_fuel} -> {fuel} ({classify_fuel_delta(delta)})"),
                )
            )
        previous_fuel = fuel
    return rows


def _position_rows(timeline: ShadowTimelineDict, start_ms: int, end_ms: int) -> list[FightRowDict]:
    """Render every in-window position statement.

    Args:
        timeline: The extracted session timeline.
        start_ms: Inclusive window start.
        end_ms: Inclusive window end.

    Returns:
        One row per position fix in the window.
    """
    return [
        FightRowDict(
            timestamp_ms=position["timestamp_ms"],
            kind="position",
            actor=_actor_name(timeline, position["tank_id"]),
            description=f"at ({position['x']},{position['y']})",
        )
        for position in timeline["positions"]
        if start_ms <= position["timestamp_ms"] <= end_ms
    ]


def _kill_rows(timeline: ShadowTimelineDict, start_ms: int, end_ms: int) -> list[FightRowDict]:
    """Render every in-window deactivation.

    Args:
        timeline: The extracted session timeline.
        start_ms: Inclusive window start.
        end_ms: Inclusive window end.

    Returns:
        One row per 0x41 in the window.
    """
    return [
        FightRowDict(
            timestamp_ms=kill["timestamp_ms"],
            kind="kill",
            actor=_actor_name(timeline, kill["victim_id"]),
            description=f"DEACTIVATED by {_actor_name(timeline, kill['killer_id'])}",
        )
        for kill in timeline["kills"]
        if start_ms <= kill["timestamp_ms"] <= end_ms
    ]


def render_fight_rows(
    timeline: ShadowTimelineDict,
    start_ms: int,
    end_ms: int,
) -> list[FightRowDict]:
    """Render the chronological play-by-play for one window.

    Args:
        timeline: The extracted session timeline.
        start_ms: Inclusive window start (capture epoch milliseconds).
        end_ms: Inclusive window end.

    Returns:
        Rows in timestamp order: every shot, every self fuel delta
        with its cause, every position statement, every deactivation.
    """
    rows = (
        _shot_rows(timeline, start_ms, end_ms)
        + _fuel_rows(timeline, start_ms, end_ms)
        + _position_rows(timeline, start_ms, end_ms)
        + _kill_rows(timeline, start_ms, end_ms)
    )
    rows.sort(key=lambda row: (row["timestamp_ms"], row["kind"]))
    return rows


__all__ = [
    "FightRowDict",
    "FuelCause",
    "HumanEpisodeDict",
    "classify_fuel_delta",
    "extract_human_episodes",
    "render_fight_rows",
]
