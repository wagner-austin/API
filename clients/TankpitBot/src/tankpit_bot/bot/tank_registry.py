"""Per-colour tank record, written from the in-game ``C`` panel.

An account holds four tanks per world, one per colour, each with its
own rank, kills and promotion points ([[game-rules]]). Nothing on the
wire reports the ones you are NOT playing -- the lobby names only the
last-played colour -- so the only way to know all four is to record
each as it is played. This module is that recorder.

The source is the STARTUP panel sample, not an exit one. Measured
2026-08-31 across two live runs: the panel is a LOGIN-TIME SNAPSHOT.
A 150-second session that scored a kill reported identical
``destroyed_enemies`` (1958), ``promotion_points`` (674270) and
``play_time_s`` (365247) at startup and at teardown, while the values
DID move between the two runs. Nothing updates mid-session, so an exit
press costs a tick ([[client-commands]]) to re-read what the startup
press already had. Run N's results land in run N+1's startup sample.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.browser.accounts import resolve_account
from tankpit_bot.runtime_artifacts import TANK_REGISTRY_PATH
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types.constants import TROOP_COLOR_NAMES

log = get_logger(__name__)


def _load_registry() -> JSONObject:
    """Return the registry on disk, or an empty one.

    Returns:
        The parsed registry; empty when absent or unreadable, so a
        first run writes a fresh file instead of failing.
    """
    try:
        raw = _test_hooks.read_text(TANK_REGISTRY_PATH)
    except OSError:
        return {}
    try:
        return narrow_json_to_dict(load_json_str(raw))
    except JSONTypeError as error:
        # A corrupt registry must not take a live session down: the
        # bot is mid-run and this is bookkeeping, not gameplay.
        log.warning(
            "Tank registry at %s is not an object (%s); starting fresh",
            TANK_REGISTRY_PATH,
            error,
        )
        return {}


def record_tank_sample(ws: WorldService, room: str) -> None:
    """Merge this session's tank into the registry under its colour.

    Writes nothing unless the colour, the account name and a visible
    panel are all known: a half-identified row would claim a reading
    the run never took.

    Args:
        ws: The session's world service, holding the panel sample and
            the self tank's team.
        room: The room this session joined.
    """
    self_state = ws.get_world_state()["self_state"]
    account = ws.self_account
    # Keyed by the ACCOUNT, not the wire tank name. They match today,
    # but the control page looks rows up by the accounts.json username
    # it offers in its dropdown, and the wire name is only recorded
    # when a 0x21 identity happens to arrive after self_state is
    # established -- a race that leaves it empty on plenty of runs.
    configured = resolve_account()
    if configured is None:
        log.info("Tank registry: skipped (no configured account to key the row by)")
        return
    name = configured["username"]
    if self_state is None or not account["rank_name"]:
        log.info("Tank registry: skipped (no colour or panel sample yet)")
        return
    team = self_state["team"]
    if not 0 <= team < len(TROOP_COLOR_NAMES):
        log.info("Tank registry: skipped (team %d is not a colour)", team)
        return

    registry = _load_registry()
    accounts = narrow_json_to_dict(registry.get("accounts", {}))
    rooms = narrow_json_to_dict(accounts.get(name, {}))
    colours = narrow_json_to_dict(rooms.get(room, {}))
    # ``rank`` is the panel's own WORD, not the 0..8 index. The index
    # is derivable from the name through RANK_NAMES; the reverse is
    # not, so storing the name keeps a label the table does not know
    # instead of flattening it to a sentinel. Consumers that need the
    # number for fuel_capacity or radar_radius look it up.
    colours[TROOP_COLOR_NAMES[team]] = {
        "rank": account["rank_name"],
        # Descends toward 1 as points accumulate, so it is a POSITION
        # rather than a countdown to the next promotion (operator,
        # 2026-08-31). ``AccountStatsDict.rank_number`` still carries
        # the older reading in its own docstring.
        "leaderboard": account["rank_number"],
        "kills": account["destroyed_enemies"],
        "deaths": account["deactivated_total"],
        "promo": account["promotion_points"],
        "play_time_s": account["play_time_s"],
        "observed_ms": account["stats_observed_ms"],
    }
    rooms[room] = colours
    accounts[name] = rooms
    registry["accounts"] = accounts
    _test_hooks.write_text(TANK_REGISTRY_PATH, dump_json_str(registry, indent=1))
    log.info(
        "Tank registry: %s %s %s -> %s (%d kills, %d deaths)",
        name,
        room,
        TROOP_COLOR_NAMES[team],
        account["rank_name"],
        account["destroyed_enemies"],
        account["deactivated_total"],
    )


__all__ = ["record_tank_sample"]
