"""HUD payload model + renderer for the in-page diagnostic overlay.

One :class:`OverlayStateDict` is assembled per tick from data the tick
loop already holds (decision, self state, inventory, session counters)
and rendered into a flat slot payload the in-page HUD consumes
(:mod:`tankpit_bot.browser.overlay_hud`). All display strings and
colors are computed HERE so the JS side stays a dumb sink that fills
fixed-geometry slots.

The color vocabulary is the fiesta retro theme, channel-for-channel
(``~/PROJECTS/MCPs/fiesta/src/services/theme.ts`` ``RETRO_THEME``):
neon green for full/good, purple for the neutral neon, hot pink for
combat/low, with the dim/foreground grays from the fiesta stylesheet.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

FIESTA_GREEN = "rgb(57, 255, 20)"
"""Retro theme title/highlight green — full stocks, dispatched commands."""

FIESTA_PURPLE = "rgb(200, 0, 200)"
"""Retro theme button purple — the neutral neon."""

FIESTA_PINK = "rgb(255, 20, 147)"
"""Retro theme ATK hot pink — combat mode, low stocks, held commands."""

FIESTA_FG = "#e9e9f0"
"""Fiesta foreground off-white — mid-band values."""

_MODE_COLORS: dict[str, str] = {
    "HUNT": FIESTA_PINK,
    "COLLECT": FIESTA_GREEN,
    "UNSET": FIESTA_PURPLE,
}

_STOCK_SHORTFALL_TOLERANCE = 5
"""Counts within this many of cap render off-white instead of pink.

Mirrors the hunt-readiness contract (2026-07-25): radars within 5 of
cap still satisfy the hunt gate, so only deeper shortfalls go pink.
"""

_FUEL_LOW_PCT = 25
"""Fuel percentage below which the meter goes hot pink.

The wire damage tiers are fuel-capacity quartiles
(:func:`tankpit_bot.physics.capacity.damage_tier`); the bottom
quartile is the near-death shade, so the meter matches it.
"""


class OverlayStateDict(TypedDict):
    """One tick's HUD payload.

    Attributes:
        hfsm_state: Bot state-machine state name.
        ai_mode: Durable AI mode (e.g. ``HUNT``).
        ai_mode_state: Mode sub-state (e.g. ``ENGAGE``).
        behavior_mode: Behavior label of this tick's decision.
        behavior_reason: Reason label of this tick's decision.
        command_type: Command kind dispatched this tick.
        target_x: Decision target X coordinate.
        target_y: Decision target Y coordinate.
        command_sent: Whether the executor actually dispatched.
        in_flight_kind: Current in-flight action kind.
        fuel: Current fuel reading.
        fuel_cap: Rank-derived fuel capacity.
        self_x: Current self X coordinate.
        self_y: Current self Y coordinate.
        armor: Armor shield count.
        duals: Dual shot count.
        missiles: Missile shot count.
        homings: Homing shot count.
        radars: Extra radar count.
        inv_cap: Rank-derived per-slot inventory capacity.
        kills: Session kill count.
        hits: Session hit count.
        misses: Session miss count.
        rejects: Session reject count.
        target_id: Combat target tank id (``-1`` when none).
        target_name: Last shot target name (``""`` when none).
    """

    hfsm_state: str
    ai_mode: str
    ai_mode_state: str
    behavior_mode: str
    behavior_reason: str
    command_type: str
    target_x: int
    target_y: int
    command_sent: bool
    in_flight_kind: str
    fuel: int
    fuel_cap: int
    self_x: int
    self_y: int
    armor: int
    duals: int
    missiles: int
    homings: int
    radars: int
    inv_cap: int
    kills: int
    hits: int
    misses: int
    rejects: int
    target_id: int
    target_name: str


def encode_overlay_state(overlay: OverlayStateDict) -> JSONObject:
    """Encode an overlay payload to JSON.

    Args:
        overlay: Payload to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "hfsm_state": overlay["hfsm_state"],
        "ai_mode": overlay["ai_mode"],
        "ai_mode_state": overlay["ai_mode_state"],
        "behavior_mode": overlay["behavior_mode"],
        "behavior_reason": overlay["behavior_reason"],
        "command_type": overlay["command_type"],
        "target_x": overlay["target_x"],
        "target_y": overlay["target_y"],
        "command_sent": overlay["command_sent"],
        "in_flight_kind": overlay["in_flight_kind"],
        "fuel": overlay["fuel"],
        "fuel_cap": overlay["fuel_cap"],
        "self_x": overlay["self_x"],
        "self_y": overlay["self_y"],
        "armor": overlay["armor"],
        "duals": overlay["duals"],
        "missiles": overlay["missiles"],
        "homings": overlay["homings"],
        "radars": overlay["radars"],
        "inv_cap": overlay["inv_cap"],
        "kills": overlay["kills"],
        "hits": overlay["hits"],
        "misses": overlay["misses"],
        "rejects": overlay["rejects"],
        "target_id": overlay["target_id"],
        "target_name": overlay["target_name"],
    }


def decode_overlay_state(data: JSONObject) -> OverlayStateDict:
    """Decode an overlay payload from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated payload.

    Raises:
        JSONTypeError: When required fields are missing or invalid.
    """
    return OverlayStateDict(
        hfsm_state=require_str(data, "hfsm_state"),
        ai_mode=require_str(data, "ai_mode"),
        ai_mode_state=require_str(data, "ai_mode_state"),
        behavior_mode=require_str(data, "behavior_mode"),
        behavior_reason=require_str(data, "behavior_reason"),
        command_type=require_str(data, "command_type"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        command_sent=require_bool(data, "command_sent"),
        in_flight_kind=require_str(data, "in_flight_kind"),
        fuel=require_int(data, "fuel"),
        fuel_cap=require_int(data, "fuel_cap"),
        self_x=require_int(data, "self_x"),
        self_y=require_int(data, "self_y"),
        armor=require_int(data, "armor"),
        duals=require_int(data, "duals"),
        missiles=require_int(data, "missiles"),
        homings=require_int(data, "homings"),
        radars=require_int(data, "radars"),
        inv_cap=require_int(data, "inv_cap"),
        kills=require_int(data, "kills"),
        hits=require_int(data, "hits"),
        misses=require_int(data, "misses"),
        rejects=require_int(data, "rejects"),
        target_id=require_int(data, "target_id"),
        target_name=require_str(data, "target_name"),
    )


def _band_fill(color: str) -> str:
    """Return the translucent band fill for a bright mode color.

    Args:
        color: A ``rgb(r, g, b)`` color string.

    Returns:
        The matching ``rgba(r, g, b, 0.20)`` fill.
    """
    return color.replace("rgb(", "rgba(").replace(")", ", 0.20)")


def _stock_color(count: int, cap: int) -> str:
    """Return the display color for one inventory slot.

    Args:
        count: Current slot count.
        cap: Rank-derived per-slot capacity.

    Returns:
        Green at cap, off-white within the hunt-gate tolerance,
        hot pink below it.
    """
    if count >= cap:
        return FIESTA_GREEN
    if count >= cap - _STOCK_SHORTFALL_TOLERANCE:
        return FIESTA_FG
    return FIESTA_PINK


def _fuel_pct(fuel: int, fuel_cap: int) -> int:
    """Return the fuel meter fill percentage, clamped to [0, 100].

    Args:
        fuel: Current fuel reading.
        fuel_cap: Rank-derived fuel capacity.

    Returns:
        Integer percentage; ``0`` when the capacity is non-positive
        (an impossible live value — capacity is ``1000 + 100*rank`` —
        but the payload is plain ints and the meter must never divide
        by zero on a malformed fixture).
    """
    if fuel_cap <= 0:
        return 0
    return min(100, max(0, (100 * fuel) // fuel_cap))


def _fuel_color(pct: int) -> str:
    """Return the fuel meter color for a fill percentage.

    Args:
        pct: Fill percentage in [0, 100].

    Returns:
        Green at full, hot pink in the near-death quartile, purple
        between.
    """
    if pct >= 100:
        return FIESTA_GREEN
    if pct < _FUEL_LOW_PCT:
        return FIESTA_PINK
    return FIESTA_PURPLE


def _target_text(overlay: OverlayStateDict) -> str:
    """Return the combat-target slot text.

    Args:
        overlay: This tick's payload.

    Returns:
        ``name #id`` when both are known, ``#id`` when only the id
        is, and an em dash when there is no target.
    """
    if overlay["target_id"] < 0:
        return "—"
    if overlay["target_name"]:
        return f"{overlay['target_name']} #{overlay['target_id']}"
    return f"#{overlay['target_id']}"


def render_overlay_payload(overlay: OverlayStateDict) -> JSONObject:
    """Render one tick's payload into the HUD's flat slot map.

    Every slot is a display-ready string, int, or CSS color; the JS
    template assigns them into fixed-geometry elements without any
    client-side logic.

    Args:
        overlay: Payload to render.

    Returns:
        Slot map the HUD template consumes.
    """
    mode_color = _MODE_COLORS.get(overlay["ai_mode"], FIESTA_PURPLE)
    mode_text = (
        f"{overlay['ai_mode']} · {overlay['ai_mode_state']}"
        if overlay["ai_mode_state"]
        else overlay["ai_mode"]
    )
    pct = _fuel_pct(overlay["fuel"], overlay["fuel_cap"])
    cap = overlay["inv_cap"]
    return {
        "state_text": overlay["hfsm_state"],
        "mode_text": mode_text,
        "mode_color": mode_color,
        "mode_band": _band_fill(mode_color),
        "pos_text": f"{overlay['self_x']},{overlay['self_y']}",
        "fuel_text": f"{overlay['fuel']}/{overlay['fuel_cap']}",
        "fuel_pct": pct,
        "fuel_color": _fuel_color(pct),
        "s0": overlay["armor"],
        "s1": overlay["duals"],
        "s2": overlay["missiles"],
        "s3": overlay["homings"],
        "s4": overlay["radars"],
        "s0c": _stock_color(overlay["armor"], cap),
        "s1c": _stock_color(overlay["duals"], cap),
        "s2c": _stock_color(overlay["missiles"], cap),
        "s3c": _stock_color(overlay["homings"], cap),
        "s4c": _stock_color(overlay["radars"], cap),
        "do_text": (f"{overlay['command_type']} → ({overlay['target_x']},{overlay['target_y']})"),
        "sent_text": "●" if overlay["command_sent"] else "✕",
        "sent_color": FIESTA_GREEN if overlay["command_sent"] else FIESTA_PINK,
        "why_text": f"{overlay['behavior_mode']}: {overlay['behavior_reason']}",
        "tgt_text": _target_text(overlay),
        "act_text": overlay["in_flight_kind"],
        "kills": overlay["kills"],
        "hits": overlay["hits"],
        "misses": overlay["misses"],
        "rejects": overlay["rejects"],
    }


__all__ = [
    "FIESTA_FG",
    "FIESTA_GREEN",
    "FIESTA_PINK",
    "FIESTA_PURPLE",
    "OverlayStateDict",
    "decode_overlay_state",
    "encode_overlay_state",
    "render_overlay_payload",
]
