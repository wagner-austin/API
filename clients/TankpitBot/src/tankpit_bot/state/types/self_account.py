"""The canonical account-identity model for the bot's own tank.

Built 2026-08-06 (user: "we want to make sure that we can plug in to
the tank information as we add new features"). ``SelfStateDict`` is
the TACTICAL self — position, team, rank tier, fuel — refreshed by
the wire every tick. This module is the ACCOUNT self: the
session-stable identity and progress facts that previously existed
only as diagnostic exhaust (one ``tank_identity`` event at join, one
``session_account_stats`` scrape at startup) with no home a runtime
feature could consult.

Two writers fill it: the 0x21 TankInfo dispatch (name, persistent id,
decoration) and the startup stats-panel scrape (rank_name, the
countdown ``rank_number`` — [[tank-registry]] § rank number —
promotion points, lifetime totals). Anything rank-aware or
identity-aware plugs in HERE instead of re-fishing event streams.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int, require_str
from typing_extensions import TypedDict


class SelfAccountDict(TypedDict):
    """Account-level identity and progress of the bot's own tank.

    Attributes:
        name: In-game tank name (e.g. ``Artax``); empty until the
            self 0x21 TankInfo arrives.
        persistent_tank_id: Cross-session account id (JS ``a.aa``);
            -1 until observed.
        decoration_state_hex: Cosmetic skin bytes, hex-encoded; empty
            until observed.
        rank_name: Stats-panel rank label (e.g. ``private``); empty
            until scraped.
        rank_number: The countdown rank in parentheses after the rank
            name — descends toward 1 as promotion points accumulate
            ([[tank-registry]]); -1 until scraped.
        promotion_points: Lifetime promotion points; -1 until scraped.
        destroyed_enemies: Lifetime kills; -1 until scraped.
        deactivated_total: Lifetime own-deactivations; -1 until
            scraped.
        play_time_s: Lifetime play seconds; -1 until scraped.
        identity_observed_ms: When the identity fields last updated
            (0 = never).
        stats_observed_ms: When the scrape fields last updated
            (0 = never).
    """

    name: str
    persistent_tank_id: int
    decoration_state_hex: str
    rank_name: str
    rank_number: int
    promotion_points: int
    destroyed_enemies: int
    deactivated_total: int
    play_time_s: int
    identity_observed_ms: int
    stats_observed_ms: int


def make_empty_self_account() -> SelfAccountDict:
    """Build the never-observed account model.

    Returns:
        All-sentinel account state (empty strings, -1 counters, zero
        timestamps).
    """
    return SelfAccountDict(
        name="",
        persistent_tank_id=-1,
        decoration_state_hex="",
        rank_name="",
        rank_number=-1,
        promotion_points=-1,
        destroyed_enemies=-1,
        deactivated_total=-1,
        play_time_s=-1,
        identity_observed_ms=0,
        stats_observed_ms=0,
    )


def encode_self_account(account: SelfAccountDict) -> JSONObject:
    """Encode the account model to a JSON-serializable dict.

    Args:
        account: Account state to encode.

    Returns:
        JSON object with every field.
    """
    return {
        "name": account["name"],
        "persistent_tank_id": account["persistent_tank_id"],
        "decoration_state_hex": account["decoration_state_hex"],
        "rank_name": account["rank_name"],
        "rank_number": account["rank_number"],
        "promotion_points": account["promotion_points"],
        "destroyed_enemies": account["destroyed_enemies"],
        "deactivated_total": account["deactivated_total"],
        "play_time_s": account["play_time_s"],
        "identity_observed_ms": account["identity_observed_ms"],
        "stats_observed_ms": account["stats_observed_ms"],
    }


def decode_self_account(data: JSONObject) -> SelfAccountDict:
    """Decode the account model with strict validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated account state.
    """
    return SelfAccountDict(
        name=require_str(data, "name"),
        persistent_tank_id=require_int(data, "persistent_tank_id"),
        decoration_state_hex=require_str(data, "decoration_state_hex"),
        rank_name=require_str(data, "rank_name"),
        rank_number=require_int(data, "rank_number"),
        promotion_points=require_int(data, "promotion_points"),
        destroyed_enemies=require_int(data, "destroyed_enemies"),
        deactivated_total=require_int(data, "deactivated_total"),
        play_time_s=require_int(data, "play_time_s"),
        identity_observed_ms=require_int(data, "identity_observed_ms"),
        stats_observed_ms=require_int(data, "stats_observed_ms"),
    )


__all__ = [
    "SelfAccountDict",
    "decode_self_account",
    "encode_self_account",
    "make_empty_self_account",
]
