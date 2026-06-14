"""In-page diagnostic overlay showing what the bot is doing live.

Renders a fixed-position HUD element inside the game page so a human
watching the browser can see the bot's current thinking without tailing
artifacts: HFSM state, AI mode, the decision taken this tick (command,
reason, target), the in-flight action, and fuel. One CDP
``Runtime.evaluate`` per tick updates the element; the payload lines are
rendered in Python so the JS side stays a dumb text sink.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot._test_hooks import CDPSessionProtocol


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
        self_x: Current self X coordinate.
        self_y: Current self Y coordinate.
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
    self_x: int
    self_y: int


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
        "self_x": overlay["self_x"],
        "self_y": overlay["self_y"],
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
        self_x=require_int(data, "self_x"),
        self_y=require_int(data, "self_y"),
    )


def render_overlay_lines(overlay: OverlayStateDict) -> list[str]:
    """Render the HUD text lines for one tick.

    Args:
        overlay: Payload to render.

    Returns:
        Human-readable lines shown in the page overlay.
    """
    sent = "sent" if overlay["command_sent"] else "NOT SENT"
    return [
        f"BOT {overlay['hfsm_state']} | {overlay['ai_mode']}/{overlay['ai_mode_state']}",
        f"pos ({overlay['self_x']},{overlay['self_y']}) fuel {overlay['fuel']}",
        f"do  {overlay['command_type']} -> ({overlay['target_x']},{overlay['target_y']}) [{sent}]",
        f"why {overlay['behavior_mode']}: {overlay['behavior_reason']}",
        f"act {overlay['in_flight_kind']}",
    ]


_UPDATE_TEMPLATE = """
(() => {{
    const data = {payload};
    let el = document.getElementById('tankpit-bot-hud');
    if (!el) {{
        el = document.createElement('div');
        el.id = 'tankpit-bot-hud';
        el.style.cssText =
            'position:fixed;top:8px;right:8px;z-index:99999;' +
            'background:rgba(0,0,0,0.78);color:#7CFC00;' +
            'font:12px/1.5 monospace;padding:8px 10px;border-radius:6px;' +
            'pointer-events:none;white-space:pre;text-align:left;';
        document.body.appendChild(el);
    }}
    el.textContent = data.lines.join('\\n');
    return true;
}})()
"""


def update_bot_overlay(cdp: CDPSessionProtocol, overlay: OverlayStateDict) -> None:
    """Create or update the in-page HUD with this tick's payload.

    Args:
        cdp: Active CDP session attached to the live tankpit page.
        overlay: Payload to render.
    """
    lines: list[JSONValue] = list(render_overlay_lines(overlay))
    payload: JSONObject = {"lines": lines}
    cdp.send(
        "Runtime.evaluate",
        {
            "expression": _UPDATE_TEMPLATE.format(payload=dump_json_str(payload)),
            "returnByValue": True,
        },
    )


__all__ = [
    "OverlayStateDict",
    "decode_overlay_state",
    "encode_overlay_state",
    "render_overlay_lines",
    "update_bot_overlay",
]
