"""Game loop: find enemies, teleport to them, hunt and kill.

Extracted from base.py for separation of concerns. The loop runs one
command per 2-second server tick and has three modes:

1. **Seek** — no enemy in combat range: teleport to the closest enemy.
2. **Hunt** — enemy within range: alternate shooting and moving toward them.
3. **Patrol** — no enemies on the map: walk in a square, terrain-aware.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.commands import encode_teleport_command
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.protocol.commands import (
    CMD_MAP_OPEN,
    TICK_RATE_MS,
    build_move_command,
    build_query_command,
    build_shoot_command,
)
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
    get_world_state,
)
from tankpit_bot.state.types import TankStateDict

log = get_logger(__name__)

COMBAT_RANGE = 8  # Max tiles from self to shoot/move (must be within 18x18 viewport)


def _force_idle(bot: Bot) -> None:
    """Force bot state to IDLE so game loop can always take actions."""
    from tankpit_bot.bot.states import transition_to

    if bot.get_state() != "IDLE":
        bot._state_data = transition_to(bot._state_data, "IDLE")


def _shoot(bot: Bot, x: int, y: int, target_id: int) -> None:
    """Send shoot command directly, bypassing state machine."""
    raw = build_shoot_command(x, y, target_id)
    bot._send_bytes(raw, f"shoot({x},{y},id={target_id})")


def _move(bot: Bot, x: int, y: int) -> None:
    """Send move command directly, bypassing state machine."""
    raw = build_move_command(x, y)
    bot._send_bytes(raw, f"move({x},{y})")



def _open_map_and_sync(bot: Bot, page: _test_hooks.PageProtocol) -> None:
    """Open map and wait for position data to arrive. Two sync passes."""
    encoded = build_query_command(CMD_MAP_OPEN)
    bot._send_bytes(encoded, "map_open")
    page.wait_for_timeout(1500)
    _sync_js_tank_positions(bot)


def _last_combat_status(bot: Bot, target_name: str) -> str:
    """Scrape game log and check combat status against a specific target.

    Returns one of:
        "hit"   — last combat line is "You hit ..."
        "miss"  — last combat line is "You fire"
        "kill"  — target_name "has been deactivated by you"
        ""      — no combat line found
    """
    if bot._cdp is None:
        return ""
    from tankpit_bot.browser.dom_scraper import SCRAPE_GAME_LOG_JS
    try:
        result = bot._cdp.send(
            "Runtime.evaluate",
            {"expression": SCRAPE_GAME_LOG_JS, "returnByValue": True},
        )
        inner = result.get("result")
        raw = inner.get("value", "") if isinstance(inner, dict) else ""
    except (OSError, RuntimeError):
        return ""
    if not raw:
        return ""
    lines = raw.split("\n")
    # Check if our specific target was deactivated
    # Game log format: "red-4\n has been deactivated by you"
    for i, line in enumerate(lines):
        if "deactivated by you" in line.lower():
            # Check if target name is on this line or the previous line
            if target_name.lower() in line.lower():
                return "kill"
            if i > 0 and target_name.lower() in lines[i - 1].lower():
                return "kill"
    # Check last combat line for hit/miss
    for line in reversed(lines):
        low = line.strip().lower()
        if low.startswith("you hit"):
            return "hit"
        if low == "you fire":
            return "miss"
    return ""


# ---------------------------------------------------------------------------
# Main entry point — called from Bot.run()
# ---------------------------------------------------------------------------


def run_game_loop(bot: Bot, page: _test_hooks.PageProtocol) -> None:
    """Run the game loop: seek, hunt, patrol.

    Args:
        bot: Bot instance for sending commands.
        page: Playwright page for waiting.
    """
    pos = bot.get_position()
    if pos is None:
        log.warning("No position available, waiting...")
        page.wait_for_timeout(TICK_RATE_MS)
        pos = bot.get_position()

    (x, y) = (pos[0], pos[1]) if pos is not None else (128, 128)

    terrain = _load_terrain()

    # Patrol state
    direction = 0
    dir_dx = [1, 0, -1, 0]
    dir_dy = [0, 1, 0, -1]
    steps_in_dir = 0

    # Hunt state
    shoot_next = True
    locked_target_id: int | None = None  # stick to this enemy until dead/gone
    shot_at_id: int | None = None  # tank_id we shot at last tick
    killed_ids: dict[int, int] = {}  # tank_id -> tick when killed (skip for N ticks)

    # Install JS hook to capture tank positions from the game client
    _install_tank_position_hook(bot)

    # Open map, get positions, teleport to nearest enemy
    _open_map_and_sync(bot, page)
    target = _find_closest_enemy(x, y)
    if target is not None:
        tx, ty = target["x"], target["y"]
        cmd = make_teleport_command(tx, ty)
        bot._send_bytes(encode_teleport_command(cmd), f"teleport({tx},{ty})")
        locked_target_id = target["tank_id"]
        x, y = tx, ty
        page.wait_for_timeout(TICK_RATE_MS)
        _sync_js_tank_positions(bot)

    tick = 0
    last_teleport_tick = 0  # skip stale world state reads right after teleport
    KILL_COOLDOWN = 10  # ignore corpse for 10 ticks (~20s) after kill

    while True:
        tick += 1
        game_entries = bot._poll_game_log()

        # Expire old kill cooldowns
        killed_ids = {k: t for k, t in killed_ids.items() if tick - t < KILL_COOLDOWN}

        # Sync JS tank positions into our world state every tick
        _sync_js_tank_positions(bot)

        # After shooting, check last combat line in DOM game log
        if shot_at_id is not None:
            shot_tank = get_world_state()["tanks"].get(str(shot_at_id))
            shot_name = shot_tank["name"] if shot_tank else ""
            status = _last_combat_status(bot, shot_name)
            log.info("COMBAT STATUS: %s (target %s id=%d)", status or "none", shot_name, shot_at_id)

            if status == "kill":
                log.info("KILL: target %d deactivated", shot_at_id)
                killed_ids[shot_at_id] = tick
                locked_target_id = None
                shot_at_id = None
                shoot_next = True
                continue
            elif status == "miss":
                log.info("MISS: target %s dodged, re-locating", shot_name)
                # One atomic operation: open map → wait for blob → find → teleport
                _open_map_and_sync(bot, page)
                new_target = _find_closest_enemy(x, y, killed_ids)
                if new_target is not None:
                    tx, ty = new_target["x"], new_target["y"]
                    log.info("RE-LOCATE: %s at (%d,%d)", new_target["name"], tx, ty)
                    locked_target_id = new_target["tank_id"]
                    cmd = make_teleport_command(tx, ty)
                    bot._send_bytes(encode_teleport_command(cmd), f"teleport({tx},{ty})")
                    x, y = tx, ty
                    last_teleport_tick = tick
                else:
                    locked_target_id = None
                shot_at_id = None
                shoot_next = True
                page.wait_for_timeout(TICK_RATE_MS)
                continue
            else:
                # "hit" or no feedback — keep shooting
                shot_at_id = None

        # Force IDLE — game loop owns control, state machine must not block actions
        _force_idle(bot)

        # Read position from world state, but not right after a teleport (stale)
        world = get_world_state()
        if tick - last_teleport_tick >= 5:
            self_state = world["self_state"]
            if self_state is not None and (self_state["x"] != 0 or self_state["y"] != 0):
                x, y = self_state["x"], self_state["y"]

        # Target locking: prefer current target if still valid
        target = _find_target(x, y, locked_target_id, killed_ids)

        # No known enemies — open map to get fresh positions
        if target is None:
            _open_map_and_sync(bot, page)
            target = _find_closest_enemy(x, y, killed_ids)
            if target is not None:
                tx, ty = target["x"], target["y"]
                locked_target_id = target["tank_id"]
                cmd = make_teleport_command(tx, ty)
                bot._send_bytes(encode_teleport_command(cmd), f"teleport({tx},{ty})")
                x, y = tx, ty
                page.wait_for_timeout(TICK_RATE_MS)
                _sync_js_tank_positions(bot)
                shoot_next = True
                continue
            else:
                locked_target_id = None

        if target is not None:
            locked_target_id = target["tank_id"]
            dist = abs(target["x"] - x) + abs(target["y"] - y)

            log.info(
                "TARGET: %s at (%d,%d) dist=%d self=(%d,%d)",
                target["name"],
                target["x"],
                target["y"],
                dist,
                x,
                y,
            )

            if dist > COMBAT_RANGE:
                fuel = bot.get_fuel()
                if fuel < 100:
                    log.info("SEEK: too low fuel (%d) to teleport, waiting", fuel)
                else:
                    _force_idle(bot)
                    _open_map_and_sync(bot, page)
                    # Re-read target position from fresh data
                    target = _find_target(x, y, locked_target_id, killed_ids)
                    if target is not None:
                        tx, ty = target["x"], target["y"]
                        log.info("SEEK: teleport -> (%d,%d) fuel=%d", tx, ty, fuel)
                        cmd = make_teleport_command(tx, ty)
                        bot._send_bytes(encode_teleport_command(cmd), f"teleport({tx},{ty})")
                        x, y = tx, ty
                        last_teleport_tick = tick
                shoot_next = True
                page.wait_for_timeout(TICK_RATE_MS)
                continue
            else:
                # In range — hunt
                x, y, shoot_next, did_shoot = _hunt_tick(
                    bot, x, y, target, shoot_next, terrain,
                )
                shot_at_id = target["tank_id"] if did_shoot else None
        else:
            # No enemies — patrol
            locked_target_id = None
            shoot_next = True
            nx, ny, blocked = _patrol_step(
                bot, x, y, direction, dir_dx, dir_dy, terrain,
            )
            if blocked:
                direction = (direction + 1) % 4
                steps_in_dir = 0
            else:
                x, y = nx, ny
                steps_in_dir += 1
                if steps_in_dir >= 10:
                    direction = (direction + 1) % 4
                    steps_in_dir = 0

        page.wait_for_timeout(TICK_RATE_MS)


# ---------------------------------------------------------------------------
# Terrain loading
# ---------------------------------------------------------------------------


def _install_tank_position_hook(bot: Bot) -> None:
    """Check WS hook and install runtime interception if needed."""
    if bot._cdp is None:
        return

    # Check if our constructor hook is active AND install a runtime
    # hook on WebSocket.prototype to catch WS instances we missed
    hook_js = r"""(() => {
let info = {
  allWS: (window.__allWS || []).length,
  recv: window.__wsRecvCount || 0,
  wsWrapped: typeof window.__rawMsgs !== 'undefined'
};
if (!window.__runtimeHooked) {
  window.__runtimeHooked = true;
  if (!window.__rawMsgs) window.__rawMsgs = [];
  if (!window.__wsRecvCount) window.__wsRecvCount = 0;
  const origAEL = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function(type, fn, opts) {
    if (this instanceof WebSocket && type === 'message') {
      if (!window.__allWS) window.__allWS = [];
      if (window.__allWS.indexOf(this) === -1) window.__allWS.push(this);
      window.__capturedWS = this;
      const origFn = fn;
      const ws = this;
      fn = function(event) {
        window.__wsRecvCount++;
        if (ws.readyState === 1) window.__capturedWS = ws;
        try {
          if (event.data instanceof Blob) {
            const reader = new FileReader();
            reader.onload = function() {
              const bytes = new Uint8Array(reader.result);
              let b = '';
              for (let i = 0; i < bytes.length; i += 8192) {
                b += String.fromCharCode.apply(
                  null, bytes.subarray(i, i + 8192));
              }
              window.__rawMsgs.push(btoa(b));
              if (window.__rawMsgs.length > 500)
                window.__rawMsgs = window.__rawMsgs.slice(-200);
            };
            reader.readAsArrayBuffer(event.data);
          }
        } catch(e) {}
        return origFn.call(this, event);
      };
      info.hooked_msg_listener = true;
    }
    return origAEL.call(this, type, fn, opts);
  };
  info.runtime_hook = 'installed';
} else {
  info.runtime_hook = 'already_installed';
}
return JSON.stringify(info);
})()"""
    try:
        result = bot._cdp.send("Runtime.evaluate", {"expression": hook_js, "returnByValue": True})
        inner = result.get("result")
        hook_val = inner.get("value", "") if isinstance(inner, dict) else ""
        log.info("WS hook: %s", hook_val)
    except (OSError, RuntimeError) as e:
        log.warning("WS hook failed: %s", e)


def _sync_js_tank_positions(bot: Bot) -> None:
    """Drain raw WS messages from JS and feed them through our decoder.

    The JS hook stores raw binary messages as base64 in window.__rawMsgs.
    We read them here, pass through our protocol decoder, and update world state.
    """
    if bot._cdp is None:
        return

    # Atomically drain the message queue
    drain_js = r"""
    (() => {
        let msgs = window.__rawMsgs || [];
        let count = window.__wsRecvCount || 0;
        window.__rawMsgs = [];
        return JSON.stringify({count: count, msgs: msgs});
    })()
    """
    try:
        result = bot._cdp.send("Runtime.evaluate", {"expression": drain_js, "returnByValue": True})
        inner = result.get("result")
        if not isinstance(inner, dict):
            return
        val = inner.get("value")
        if not isinstance(val, str) or not val:
            return

        from tankpit_bot.sniffer.decoders import process_received_message

        data = narrow_json_to_dict(load_json_str(val))
        raw_msgs = data.get("msgs", [])
        if not isinstance(raw_msgs, list):
            return
        if raw_msgs:
            count = data.get("count", 0)
            log.info("JS->Python: %d raw msgs (total recv: %s)", len(raw_msgs), count)
        for b64 in raw_msgs:
            if isinstance(b64, str):
                process_received_message(b64)

    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        log.debug("sync_js_tank_positions failed: %s", exc)


def _load_terrain() -> TerrainMapProtocol | None:
    """Load the correct terrain map for the current room.

    Returns:
        TerrainMap instance, or None if terrain GIF not found.
    """
    room_name = _test_hooks.get_env("TANKPIT_ROOM") or "Practice"
    field_file = "field01_r.gif" if "Practice" in room_name else "field42-r.gif"
    field_path = Path(field_file)
    if _test_hooks.path_exists(field_path):
        terrain = _test_hooks.load_terrain_map(field_path)
        log.info("Loaded terrain for %s from %s", room_name, field_file)
        return terrain

    fallback = get_terrain_map()
    log.warning("Fallback terrain map (wanted %s)", field_file)
    return fallback


# ---------------------------------------------------------------------------
# Enemy finding
# ---------------------------------------------------------------------------


def _find_target(
    x: int, y: int, locked_id: int | None, killed_ids: dict[int, int] | None = None,
) -> TankStateDict | None:
    """Find target: prefer locked target if still alive, else closest enemy."""
    if locked_id is not None and (killed_ids is None or locked_id not in killed_ids):
        world = get_world_state()
        tank = world["tanks"].get(str(locked_id))
        if tank is not None and not (tank["x"] == 0 and tank["y"] == 0):
            return tank
    return _find_closest_enemy(x, y, killed_ids)


def _find_closest_enemy(
    x: int, y: int, killed_ids: dict[int, int] | None = None,
) -> TankStateDict | None:
    """Find the closest enemy tank on the entire map.

    Args:
        x: Our current X position.
        y: Our current Y position.
        killed_ids: Tank IDs on kill cooldown — skip these.

    Returns:
        Closest enemy TankStateDict, or None if no enemies.
    """
    world = get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        return None

    skip = killed_ids or {}
    self_team = self_state["team"]
    best: TankStateDict | None = None
    best_dist = 999

    self_id = self_state["tank_id"]
    for tank in world["tanks"].values():
        if tank["is_self"] or tank["tank_id"] == self_id or tank["team"] == self_team:
            continue
        if tank["tank_id"] in skip:
            continue
        # Skip tanks with no known position (0,0 placeholder from info-only messages)
        if tank["x"] == 0 and tank["y"] == 0:
            continue
        dist = abs(tank["x"] - x) + abs(tank["y"] - y)
        if dist < best_dist:
            best = tank
            best_dist = dist

    return best


# ---------------------------------------------------------------------------
# Teleport near enemy
# ---------------------------------------------------------------------------


def _teleport_near(
    ex: int,
    ey: int,
    terrain: TerrainMapProtocol | None,
) -> tuple[int, int]:
    """Pick a walkable tile ~5 tiles from the enemy to teleport to.

    Tries 4 cardinal offsets, then falls back to the enemy's exact position.

    Args:
        ex: Enemy X.
        ey: Enemy Y.
        terrain: TerrainMap or None.

    Returns:
        (x, y) destination for teleport.
    """
    return ex, ey


# ---------------------------------------------------------------------------
# Hunt tick (shoot or move toward enemy)
# ---------------------------------------------------------------------------


def _hunt_tick(
    bot: Bot,
    x: int,
    y: int,
    target: TankStateDict,
    shoot: bool,
    terrain: TerrainMapProtocol | None,
) -> tuple[int, int, bool, bool]:
    """Execute one hunt tick: shoot or move toward target.

    Args:
        bot: Bot instance for sending commands.
        x: Our current X.
        y: Our current Y.
        target: Enemy tank to hunt.
        shoot: True to shoot this tick, False to move.
        terrain: TerrainMap or None.

    Returns:
        (new_x, new_y, next_shoot_flag, did_shoot).
    """
    tx, ty = target["x"], target["y"]
    dist = abs(tx - x) + abs(ty - y)

    # Adjacent (dist=1) — just shoot
    if dist <= 1:
        _shoot(bot, tx, ty, target["tank_id"])
        log.info("HUNT: shoot %s at (%d,%d) dist=%d", target["name"], tx, ty, dist)
        return x, y, True, True

    if shoot:
        _shoot(bot, tx, ty, target["tank_id"])
        log.info("HUNT: shoot %s at (%d,%d) dist=%d", target["name"], tx, ty, dist)
        return x, y, False, True  # Next tick: move

    # Move directly to the tile adjacent to the enemy
    mx, my = _adjacent_tile(x, y, tx, ty, terrain)
    if (mx, my) != (x, y):
        _move(bot, mx, my)
        log.info("HUNT: move (%d,%d)->(%d,%d) toward %s", x, y, mx, my, target["name"])
        return mx, my, True, False  # Next tick: shoot

    # Blocked — just shoot
    _shoot(bot, tx, ty, target["tank_id"])
    log.info("HUNT: blocked, shooting %s", target["name"])
    return x, y, True, True


def _adjacent_tile(
    x: int,
    y: int,
    tx: int,
    ty: int,
    terrain: TerrainMapProtocol | None,
) -> tuple[int, int]:
    """Pick a passable tile adjacent to the target, closest to us.

    Returns (x, y) unchanged if all adjacent tiles are blocked.
    """
    candidates = [(tx - 1, ty), (tx + 1, ty), (tx, ty - 1), (tx, ty + 1)]
    best = (x, y)
    best_dist = 999
    for cx, cy in candidates:
        if not (0 <= cx < 256 and 0 <= cy < 256):
            continue
        if terrain is not None and not terrain.is_passable(cx, cy):
            continue
        d = abs(cx - x) + abs(cy - y)
        if d < best_dist:
            best = (cx, cy)
            best_dist = d
    return best


# ---------------------------------------------------------------------------
# Step toward target (one tile, terrain-aware)
# ---------------------------------------------------------------------------


def _step_toward(
    x: int,
    y: int,
    tx: int,
    ty: int,
    terrain: TerrainMapProtocol | None,
) -> tuple[int, int]:
    """Move one tile toward target, preferring the axis with the larger gap.

    Args:
        x: Current X.
        y: Current Y.
        tx: Target X.
        ty: Target Y.
        terrain: TerrainMap or None.

    Returns:
        (new_x, new_y) — same as (x, y) if completely blocked.
    """
    dx = 1 if tx > x else (-1 if tx < x else 0)
    dy = 1 if ty > y else (-1 if ty < y else 0)

    if abs(tx - x) >= abs(ty - y):
        candidates = [(x + dx, y), (x, y + dy)] if dx != 0 else [(x, y + dy)]
    else:
        candidates = [(x, y + dy), (x + dx, y)] if dy != 0 else [(x + dx, y)]

    for mx, my in candidates:
        if 0 <= mx < 256 and 0 <= my < 256 and (terrain is None or terrain.is_passable(mx, my)):
            return mx, my

    return x, y


# ---------------------------------------------------------------------------
# Patrol step (walk in a direction, terrain-aware)
# ---------------------------------------------------------------------------


def _patrol_step(
    bot: Bot,
    x: int,
    y: int,
    direction: int,
    dir_dx: list[int],
    dir_dy: list[int],
    terrain: TerrainMapProtocol | None,
) -> tuple[int, int, bool]:
    """Move one tile in the patrol direction.

    Args:
        bot: Bot instance for sending commands.
        x: Current X.
        y: Current Y.
        direction: Current patrol direction (0-3).
        dir_dx: Direction X deltas.
        dir_dy: Direction Y deltas.
        terrain: TerrainMap or None.

    Returns:
        (new_x, new_y, was_blocked).
    """
    tx = x + dir_dx[direction]
    ty = y + dir_dy[direction]

    if not (0 <= tx < 256 and 0 <= ty < 256) or (
        terrain is not None and not terrain.is_passable(tx, ty)
    ):
        log.info("PATROL: (%d,%d) blocked dir=%d, turning", x, y, direction)
        return x, y, True

    _move(bot, tx, ty)
    log.info("PATROL: (%d,%d) -> (%d,%d) dir=%d", x, y, tx, ty, direction)
    return tx, ty, False
