"""Protocol message TypedDicts.

This module contains all TypedDict definitions for protocol messages,
organized by category for easy reference.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot.container.types import ContainerMessage

# =============================================================================
# Text Messages (no XOR encoding)
# =============================================================================


class JoinConfirmDict(TypedDict):
    """Join confirmation (= message) - TEXT format.

    Format: =<team>|<join_date>|<name>|<rank>|<eq1>|<eq2>|<eq3>|<eq4>
    """

    msg_type: Literal[0x3D]
    team: int
    join_date: str
    name: str
    rank: int
    equipment: list[int]


class WorldInfoDict(TypedDict):
    """World/map info (+ message) - TEXT format.

    Format: +<id>|<name>|<field>|<flags>|<team>|<mode>|<image>|<year>
    """

    msg_type: Literal[0x2B]
    world_id: int
    name: str
    field_id: int
    flags: list[int]
    team: int
    mode: str
    image: str
    year: int


# =============================================================================
# Combat Messages
# =============================================================================


class ShootEventDict(TypedDict):
    """Shooting / hit event (0x53 'S' message).

    Layout from tpclient.js Gg.h (V.S), re-verified 2026-06-19 against
    real wire bytes from runs/bot/bot-20260619-050303 msg t+25.47s
    `53 02 15 05 b2 7d b2 7e b2 7e 01`:
      a[0]    = team byte (red=0, purple=1, blue=2, orange=3)
      a[1:3]  = shooter_id (LE u16)  -- the tank that fired
      a[3]    = source_x  -- shooter's tile X (live position)
      a[4]    = source_y  -- shooter's tile Y
      a[5]    = target_x  -- shot's landing tile X (homing's final tile)
      a[6]    = target_y  -- shot's landing tile Y
      a[7]    = aim_x     -- aim tile X (the tile the gun is pointed at)
      a[8]    = aim_y     -- aim tile Y
      a[9]    = weapon (0=single, 1=dual, 2=missile, 3=homing)

    Prior decoder named a[3,4] as target and a[5,6] as projectile_start
    -- reversed. Prior names for a[7..9] as fuel/weapon/ammo were also
    wrong. Three-way validation (enemy src tracking, homing tgt tile,
    wire damage events) confirmed the corrected layout.

    a[7]/a[8] semantics promoted from ``unk1``/``unk2`` to
    ``aim_x``/``aim_y`` 2026-06-20. JS evidence: ``Gg.h`` passes them to
    the projectile-animation constructor ``yf`` as ``z`` and ``O``;
    inside ``yf``, ``this.qa = 24 * z + 12`` and ``this.ta = 16 * O + 8``
    are PIXEL CENTRES of the tile the tank's gun is aimed at, and
    ``yf.start()`` uses ``atan2(this.h - this.qa, this.ta - this.i)`` to
    set the tank's facing direction. For straight shots aim == target;
    for guided weapons (missile/homing) aim is the initial barrel
    direction and target is the homing impact tile.

    Hit detection per JS Gg.prototype.h case 18: shot landed on a named
    tank tile -> hit. That tile lookup uses (target_x, target_y).
    """

    msg_type: Literal[0x53]
    team: int
    shooter_id: int
    source_x: int
    source_y: int
    target_x: int
    target_y: int
    aim_x: int
    aim_y: int
    weapon: int


class DeactivationDict(TypedDict):
    """Kill/deactivation event (0x41 'A' message).

    Layout from tpclient.js Pg.h (V.A), verified 2026-06-19:
      a[0]  = status byte
      a[1:3] = victim_id (LE u16)
      a[3]  = promo_eligible (1=earned extra points)
      a[4:6] = killer_id (LE u16)
      If killer_id >= 65530: mine kill (team = killer_id - 65530)
    """

    msg_type: Literal[0x41]
    status: int
    victim_id: int
    promo_eligible: bool
    killer_id: int
    is_mine_kill: bool


# 0x4B MinePlacement and 0x45 MineDetonation TypedDicts live in
# tankpit_bot.container.types -- both are container-subtype messages
# that never arrive standalone. Protocol versions deleted 2026-06-19.


# =============================================================================
# Movement Messages
# =============================================================================


class MovementDict(TypedDict):
    """Movement path (0x47 'G' message).

    Layout from tpclient.js Lg.h (V.G), verified 2026-06-19:
      a[0:2]  = tank_id (LE u16)
      a[2]    = start_x
      a[3]    = start_y
      a[4]    = direction
      a[5]    = damage_state (assigned to b.u in Lg.prototype.h; NOT damage_state)
      a[6:9]  = lb_score (24-bit BE)
      a[9]    = rank (assigned to b.l in Lg.prototype.h)
      a[10]   = animation flag (passed to Re constructor, not tank state)
      a[11]   = is_carrying (1=true)
      a[12:]  = waypoints (direction chars)

    ``waypoints`` collapses the nsew path to its final position (one
    entry, or empty when stationary); ``path_tiles`` preserves the
    wire's true step count — one fuel per step, exact even on
    non-minimal paths around obstacles ([[game-economy]] walk row) —
    and ``path`` keeps the raw nsew route the SERVER chose, since the
    client only sends a destination click and the server pathfinds.
    """

    msg_type: Literal[0x47]
    tank_id: int
    start_x: int
    start_y: int
    direction: int
    damage_state: int
    lb_score: int
    rank: int
    flag: int
    is_carrying: bool
    waypoints: list[tuple[int, int]]
    path_tiles: int
    path: str


class MovementResponseDict(TypedDict):
    """Movement response (0x3D '=' binary message).

    Layout from tpclient.js Mg.h (V["="]):
      a[0]    = team
      a[1:3]  = tank_id (LE u16)
      a[3]    = x
      a[4]    = y
      a[5]    = direction
      a[6]    = damage_state (assigned to b.u; NOT damage_state)
      a[7]    = rank
      a[8:11] = lb_score (24-bit BE)
      a[11]   = carrying flag
    """

    msg_type: Literal[0x3D]
    team: int
    tank_id: int
    x: int
    y: int
    direction: int
    damage_state: int
    rank: int
    lb_score: int
    carrying: int


# =============================================================================
# Resource Messages (Fuel, Equipment)
# =============================================================================


class FuelGainDict(TypedDict):
    """Fuel gain event (D message). fuel_total is the new absolute fuel level.

    ``flag`` is the raw wire byte at offset 2; ``is_free`` derives from
    it (``flag == 0``). Corpus 2026-07-21 (295 samples): 294 bodies
    carry 0, one carries 0x2B — the byte is a value, not a boolean, so
    the encoder needs it verbatim for byte-identical round-trips.
    """

    msg_type: Literal[0x44]
    fuel_total: int
    is_free: bool
    flag: int


class FuelDepositDict(TypedDict):
    """Fuel deposit event (d message). fuel_total is the new absolute fuel level."""

    msg_type: Literal[0x64]
    fuel_total: int


class InventoryDict(TypedDict):
    """Inventory display (I message)."""

    msg_type: Literal[0x49]
    show: bool
    alternate: bool
    counts: list[int]
    enabled: list[bool]


class EquipmentGainDict(TypedDict):
    """Equipment gain (g message)."""

    msg_type: Literal[0x67]
    show_message: bool
    gained: list[int]


class EquipmentToggleDict(TypedDict):
    """Equipment toggle (t message)."""

    msg_type: Literal[0x74]
    enabled: list[bool]


# =============================================================================
# Radar Messages
# =============================================================================


class RadarResultDict(TypedDict):
    """Radar scan result (F message)."""

    msg_type: Literal[0x46]
    detection_type: int
    found: bool


class EnemyDetectionDict(TypedDict):
    """Enemy detection (H message)."""

    msg_type: Literal[0x48]
    tank_id: int
    x: int
    y: int
    rank: int
    team: int


class RadarContainerDict(TypedDict):
    """Container entry in radar scan result.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        volume: Fuel volume (0-32767), or -1 for equipment.
    """

    x: int
    y: int
    volume: int


class RadarMineDict(TypedDict):
    """Mine entry in radar scan result.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team that placed the mine (0=red, 1=purple, 2=blue, 3=orange).
    """

    x: int
    y: int
    team: int


class RadarMineClearDict(TypedDict):
    """Mine-clear entry in radar scan result.

    An overlay entry whose value is >= 8 (255 in the JS dh detonation
    handler) — the server's statement that the tile has NO mine. The
    JS ch handler writes the value into ``tile.m`` raw; 255 is the
    canonical no-mine sentinel it uses everywhere else.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
    """

    x: int
    y: int


class RadarScanResultDict(TypedDict):
    """Radar scan result (0x4F, JS handler ``ch`` / V.O).

    The 0x4F body is a batch of per-tile writes — a delta sync of the
    scanned area, not an append-only reveal list. Cache entries set a
    tile's container layer (0 = tile now empty, N = fuel volume,
    65535 -> -1 = equipment); overlay entries set the mine layer
    (0-7 = mine with ``team = value & 3``, >= 8 = no mine). Corpus
    scan 2026-07-03 (199 sessions, 1817 bodies): 247 of 2093 cache
    entries were removals (value 0); every body arrived tunneled
    inside 0x2E.

    Attributes:
        msg_type: Message type (0x4F).
        containers: Container entries (volume 0 = authoritative removal).
        mines: Mine entries (overlay value 0-7).
        mine_clears: Tiles the server declared mine-free (overlay >= 8).
    """

    msg_type: Literal[0x4F]
    containers: list[RadarContainerDict]
    mines: list[RadarMineDict]
    mine_clears: list[RadarMineClearDict]


# =============================================================================
# Tank Messages
# =============================================================================


class TankInfoDict(TypedDict):
    """Tank info (0x21 '!' message).

    Trace-verified from tpclient.js Tf.h (line 3896-3901):
      a[0]    = team (a[0] & 255)
      a[1:3]  = tank_id (LE u16)
      a[3:7]  = decoration_state (4 bytes, decoded by yg() into 9 x 2-bit slots)
      a[7:10] = persistent_tank_id (24-bit BE, sets a.aa for profile links)
      a[10:]  = name (UTF-8 string)

    NOTE: This message does NOT contain the tank's current rank.
    """

    msg_type: Literal[0x21]
    tank_id: int
    team: int
    decoration_state: bytes
    persistent_tank_id: int
    name: str


class TankEntryDict(TypedDict):
    """Tank entry (( message).

    Layout from tpclient.js Uf.h (V["("]), verified 2026-06-19:
      a[0]   = flags (255=known tank)
      a[1:3] = tank_id (LE u16)
      a[3]   = packed byte: team(bits 0-1), damage_state(bits 2-3), rank(bits 4-7)
      a[4:7] = score (24-bit BE)
      a[7]   = x position
      a[8]   = y position
    """

    msg_type: Literal[0x28]
    team: int
    tank_id: int
    rank: int
    damage_state: int
    score: int
    x: int
    y: int


class TankRemoveDict(TypedDict):
    """Tank removal from world (0x58 'X' message).

    Trace-verified from tpclient.js Ug.h (V.X):
      a[0:2] = tank_id (LE u16)

    Server-driven removal — clears the tile entry, releases the tank slot,
    and drops the rendered tank. No accompanying display text.
    """

    msg_type: Literal[0x58]
    tank_id: int


class MapTankEntry(TypedDict):
    """One tank slot parsed from the 0x4C MapData blob.

    Trace-verified from the per-entry loop in JS Ig.h (V.L):
      a[c+0]      = x
      a[c+1]      = y
      X(a[c+2:4]) = tank_id (LE u16)
      a[c+4]      = packed:
                    rank   = (byte >> 4) & 0xF
                    damage = (byte >> 2) & 0x3
                    team   =  byte       & 0x3
    """

    x: int
    y: int
    tank_id: int
    rank: int
    damage: int
    team: int


class MapDataDict(TypedDict):
    """Whole-map snapshot (0x4C 'L' message).

    Trace-verified from tpclient.js Ig.h (V.L). The body has two
    sections:

      1. Fuel-dot run-length list -- the map's yellow-pixel fuel
         atlas. Total RLE byte count is ``X(a[0], a[1])`` (LE u16)
         and the cells live in ``a[2 : 2+count]``. A 2-D cursor
         ``(d, e)`` starts at ``(1, 1)``; each byte ``h`` advances
         ``d`` by ``h``, wrapping to ``e += 1, d %= 256`` whenever
         ``d`` exceeds 255. Cells valued 255 are pure continuation --
         they advance the cursor but emit no dot. Every other cell
         emits the cursor as a ``(x, y)`` fuel-dot position. The
         atlas is server-cached per session (byte-identical across
         map opens); ~40% of dots still hold fuel when visited, and
         every verified dot held high-volume fuel. Restored
         2026-07-03 (decoded for length only 2026-06-22 to then).

      2. Tank entries -- 5 bytes each, packed to the end of the body.
         See :class:`MapTankEntry` for the per-entry layout.

    JS Ig.prototype.h stores each tank entry into the map slot at
    ``(x << 8) + y`` and assigns ``team`` / ``damage`` / ``rank``
    (verified against Mg.prototype.h's identical field assignment
    order: ``c.h = team``, ``c.u = damage``, ``c.l = rank``).
    """

    msg_type: Literal[0x4C]
    fuel_dots: list[tuple[int, int]]
    tanks: list[MapTankEntry]


class BuildPickupDict(TypedDict):
    """Obstacle / bridge build / pickup event (0x42 'B' message).

    Trace-verified from tpclient.js Jg.h (V.B):
      X(a[0], a[1]) = tank_id (LE u16)
      a[2]          = source_x  -- where the tank was when the action fired
      a[3]          = source_y
      a[4]          = drop_x    -- target tile receiving the obstacle / bridge
      a[5]          = drop_y
      a[6]          = direction -- new facing for the acting tank (passed to We())
      a[7]          = obstacle_type -- assigned to tile.j; ``1`` means bridge
                       module, other non-zero values are obstacle subtypes.
                       Production captures (2026-06-19) show ``2`` for a
                       regular obstacle drop; ``0`` is the cleared state.
      a[8]          = flag      -- influences pickup-visibility branch (this.s in JS)

    JS Jg.prototype.h:
      * Updates the tank's facing at (source_x, source_y).
      * Stamps ``drop_x, drop_y`` tile's ``j`` field with ``obstacle_type``.
      * For the player's own tank, prints "Bridge module built"
        (when ``obstacle_type == 1``), "Obstacle dropped", or
        "Obstacle picked up" depending on ``a.la`` (carry state).
    """

    msg_type: Literal[0x42]
    tank_id: int
    source_x: int
    source_y: int
    drop_x: int
    drop_y: int
    direction: int
    obstacle_type: int
    flag: int


class DecorationDict(TypedDict):
    """Decoration / award notification (0x4E 'N' message).

    Trace-verified from tpclient.js Sf.h (V.N):
      X(a[0], a[1]) = tank_id (LE u16)
      a[2]          = slot (decoration slot index into the tank's ``v[]`` table)
      a[3]          = level (new decoration level for that slot)

    JS Sf.prototype.h prints a banner only when the new ``level`` raises
    the tank's current ``v[slot]`` -- the new value is assigned
    unconditionally. The decoration label is
    ``nb[3 * slot + level - 1]`` (3 medals per slot).
    """

    msg_type: Literal[0x4E]
    tank_id: int
    slot: int
    level: int


class PromotionDict(TypedDict):
    """Binary promotion notification (0x2B '+' message, gameplay).

    Trace-verified from tpclient.js Rf.h (V["+"]):
      a[0] = new_rank (target rank, indexes into rank-name table)
      a[1] = was_promoted (1 = "You have been promoted!" banner;
                           0 = silent rank set, e.g. on join)

    Distinct from the text-format ``WorldInfoDict`` (also 0x2B) emitted
    by the server at lobby/ROOM_LIST time. The two are disambiguated by
    wire body length: Rf carries exactly 2 XOR-decoded payload bytes.
    """

    msg_type: Literal[0x2B]
    new_rank: int
    was_promoted: bool


class TankExitDict(TypedDict):
    """Tank exit/elimination announcement (0x29 ')' message).

    Trace-verified from tpclient.js Vf.h (V[")"]):
      a[0]   = team
      a[1:3] = tank_id (LE u16)
      a[3]   = was_silent (1 = no display text emitted)
      a[4]   = was_eliminated (1 = "eliminated from the game",
                               0 = "left the game")

    Pure announcement — the renderer prints a log line unless
    ``was_silent``. Separate from 0x58 TankRemove, which physically
    removes the tank from the world.
    """

    msg_type: Literal[0x29]
    team: int
    tank_id: int
    was_silent: bool
    was_eliminated: bool


class TankStatusSyncDict(TypedDict):
    """Tank status sync (0x2E message).

    Layout from tpclient.js Og.h (V["."]), verified 2026-06-19:
      a[0]    = team (subtype)
      a[1:3]  = tank_id (LE u16)
      a[3]    = damage_state (b.u; dual-purpose: rank_category on init, damage during gameplay)
      a[4]    = rank (b.l)
      a[5:8]  = lb_score (24-bit BE)
      a[8]    = promo_state -- present when the body is at least 9 bytes
      a[9]    = has_fuel_bar (if long form)
      a[10:12] = fuel (LE u16, if long form)

    The 9-byte short form carries ``promo_state``; the 13-byte long
    form carries ``fuel`` as well. Production corpus
    (analysis_scripts/crack_tank_status_short.py) confirms 74/74
    9-byte 0x2E bodies have promo_state in ``[0, 5]``.
    """

    msg_type: Literal[0x2E]
    subtype: int
    tank_id: int
    damage_state: int
    rank: int
    lb_score: int
    promo_state: int | None
    fuel: int | None


class TankStatusDict(TypedDict):
    """Full tank status (0x3E '>' message).

    ``damage_state`` is bits 2-3 of the info byte (the packed-byte
    convention shared with TankEntry/MapTankEntry: team 0-1, damage
    2-3, rank 4-7). Corpus 2026-07-21: 223 of 244 bodies carry a
    nonzero value there — dropping it broke byte-identical
    round-trips.
    """

    msg_type: Literal[0x3E]
    team: int
    rank: int
    damage_state: int
    tank_id: int
    decoration_state: bytes
    leaderboard_score: int
    leaderboard_position: int
    name: str


# =============================================================================
# World Messages (Viewport, Terrain, Sync)
# =============================================================================


class SyncDict(TypedDict):
    """Sync/heartbeat (0x3F '?' message)."""

    msg_type: Literal[0x3F]


class CacheUpdateDict(TypedDict):
    """Cache-only tile patch (0x43 'C' message)."""

    msg_type: Literal[0x43]
    updates: list[tuple[int, int, int]]


class ChatAckDict(TypedDict):
    """Chat-toggle acknowledgment (1-byte 0x43 'C' message).

    The 0x43 type byte is overloaded: cache patches are 4-byte
    entries, while the server answers a client chat toggle (Ka,
    "C{enabled}") with a single flag byte. Discovered live
    2026-07-24 when the key probe's Z press crashed the decode
    pipeline; the official client's $g handler reads 4-byte entries
    without length validation and silently mis-parses this frame.
    """

    msg_type: Literal["chat_ack"]
    enabled: bool


class OverlayUpdateDict(TypedDict):
    """Overlay-only tile patch (0x40 '@' message)."""

    msg_type: Literal[0x40]
    updates: list[tuple[int, int, int]]


class ViewportEntityDict(TypedDict):
    """Single tile row in viewport update."""

    col: int
    row: int
    cache_value: int
    overlay_value: int
    terrain_type: int


class ViewportUpdateDict(TypedDict):
    """Viewport/map update (0x5A 'Z' message)."""

    msg_type: Literal[0x5A]
    viewport_left: int
    viewport_top: int
    entities: list[ViewportEntityDict]


class TerrainUpdateDict(TypedDict):
    """Terrain type update (0x4A 'J' message)."""

    msg_type: Literal[0x4A]
    updates: list[tuple[int, int, int]]


class SupervisorTextDict(TypedDict):
    """Free-form server text channel (0x3C '<' message).

    Trace-verified from tpclient.js wg.h (V['<']):
      ``wg.h(a) = new wg(p(a))`` -- a is the entire XOR-decoded body
      and ``p()`` is just byte-to-string conversion
      (``String.fromCharCode(a[i] & 255)``). The renderer prints:
      "Message from the Supervisor:\\n<message>\\n".

    Distinct from 0x52 CommandResult (V.R / xg) which carries a 3-byte
    error code; this is the server's freeform announcement channel.
    """

    msg_type: Literal[0x3C]
    message: str


class SupervisorDict(TypedDict):
    """Command failure response (0x52 'R' message).

    Trace-verified from tpclient.js xg.h (line 4317-4322):
      a[0] = reset_action (1=reset to idle)
      a[1] = close_map (1=close map view)
      a[2] = error_code (index into Gb[] error strings; 128+=custom text)
    """

    msg_type: Literal[0x52]
    reset_action: int
    close_map: int
    error_code: int


# =============================================================================
# Misc Messages
# =============================================================================


class ActionDoneDict(TypedDict):
    """Action completion marker (0x54 'T' message)."""

    msg_type: Literal[0x54]


class ChatMessageDict(TypedDict):
    """Chat message (M message)."""

    msg_type: Literal[0x4D]
    sender_id: int
    message_type: int
    x: int | None
    y: int | None


class StatisticsDict(TypedDict):
    """Statistics display (V message)."""

    msg_type: Literal[0x56]
    playtime_hours: int
    playtime_minutes: int
    playtime_seconds: int
    destroyed: int
    deactivated: int
    score: int


class ActiveForcesDict(TypedDict):
    """Active forces count (* message)."""

    msg_type: Literal[0x2A]
    team_counts: list[int]


class ActivePlayerEntry(TypedDict):
    """One row in an 0x2F ActivePlayers list.

    Attributes:
        tank_id: 16-bit LE tank id from the wire.
        rank: 1-byte rank index used by the JS client to render the
            ``ec[rank]`` label in the active-players banner. Same
            domain as ``TankStateDict.rank``.
    """

    tank_id: int
    rank: int


class ActivePlayersDict(TypedDict):
    """0x2F ``/`` ActivePlayers list.

    Server-broadcast roster of every active player in the room,
    decoded from the JS Yg.h handler at ``tpclient.pretty.js:4653``.
    Wire shape is repeating 3-byte records: ``(tank_id_lo,
    tank_id_hi, rank)``. The bot consumes this to know who's actually
    in the room without needing to spam ``/`` queries.
    """

    msg_type: Literal[0x2F]
    players: list[ActivePlayerEntry]


class Top10EntryDict(TypedDict):
    """One row of the 0x31 ``1`` Top10 leaderboard.

    Attributes:
        position: 1-based leaderboard rank in this Top10 list.
        score: 24-bit BE score.
        team: Team id (0-3).
        rank: Military rank (0-8).
        name: UTF-8 player name.
        tank_id: ``-1`` when the server does not echo the persistent
            tank id on this row; otherwise the value carried by the
            wire. The JS client hyperlinks ``tank_id >= 500`` rows to
            ``/tanks/profile?tank_id=...`` so the value is at least
            sometimes a persistent identifier.
    """

    position: int
    score: int
    team: int
    rank: int
    name: str
    tank_id: int


class Top10Dict(TypedDict):
    """0x31 ``1`` Top10 leaderboard broadcast.

    Wire shape (JS Zg.h at ``tpclient.pretty.js:4679``):
      a[0]      = team_filter (255 = all-team Top10, else team id)
      a[1..3]   = viewer's score (24-bit BE)
      a[4]      = viewer's leaderboard position
      a[5..]    = repeating rows: position(1), score(3 BE), team(1),
                  rank(1), name_len(1), name(name_len bytes)

    Attributes:
        team_filter: ``255`` for the all-team list, else the team id
            this Top10 row applies to.
        viewer_score: 24-bit BE score of the player viewing the list.
        viewer_position: 1-based leaderboard position of the viewer.
        entries: Decoded rows in the order the server sent them
            (top to bottom).
    """

    msg_type: Literal[0x31]
    team_filter: int
    viewer_score: int
    viewer_position: int
    entries: list[Top10EntryDict]


class PingResponseDict(TypedDict):
    """0x60 `` ` `` PingResponse from the server.

    JS V[``\\``] = we (``tpclient.pretty.js:3839``) handler is a no-op:
    the server just acknowledges the bot is still considered
    connected. Decoded for telemetry so the bot's events stream can
    timestamp every heartbeat.
    """

    msg_type: Literal[0x60]


class ConnectionLostDict(TypedDict):
    """0x7E ``~`` ConnectionLost from the server.

    JS V[``~``] = xe (``tpclient.pretty.js:3829``) triggers a
    disconnect. Decoded so the bot's events stream records WHY a
    session ended even when the transport layer doesn't surface a
    structured reason.
    """

    msg_type: Literal[0x7E]


__all__ = [
    "ActionDoneDict",
    "ActiveForcesDict",
    "ActivePlayerEntry",
    "ActivePlayersDict",
    "BinaryMessage",
    "BuildPickupDict",
    "CacheUpdateDict",
    "ChatAckDict",
    "ChatMessageDict",
    "ConnectionLostDict",
    "DeactivationDict",
    "DecodedMessage",
    "DecorationDict",
    "EnemyDetectionDict",
    "EquipmentGainDict",
    "EquipmentToggleDict",
    "FuelDepositDict",
    "FuelGainDict",
    "InventoryDict",
    "JoinConfirmDict",
    "MapDataDict",
    "MapTankEntry",
    "MovementDict",
    "MovementResponseDict",
    "OverlayUpdateDict",
    "PingResponseDict",
    "PromotionDict",
    "RadarContainerDict",
    "RadarMineClearDict",
    "RadarMineDict",
    "RadarResultDict",
    "RadarScanResultDict",
    "ShootEventDict",
    "StatisticsDict",
    "SupervisorDict",
    "SupervisorTextDict",
    "SyncDict",
    "TankEntryDict",
    "TankExitDict",
    "TankInfoDict",
    "TankRemoveDict",
    "TankStatusDict",
    "TankStatusSyncDict",
    "TerrainUpdateDict",
    "TextMessage",
    "Top10Dict",
    "Top10EntryDict",
    "ViewportEntityDict",
    "ViewportUpdateDict",
    "WorldInfoDict",
]


# =============================================================================
# Message union types
# =============================================================================

TextMessage = JoinConfirmDict | WorldInfoDict

BinaryMessage = (
    ShootEventDict
    | DeactivationDict
    | FuelGainDict
    | FuelDepositDict
    | RadarResultDict
    | EnemyDetectionDict
    | InventoryDict
    | EquipmentGainDict
    | EquipmentToggleDict
    | RadarScanResultDict
    | MovementDict
    | TankInfoDict
    | MovementResponseDict
    | SyncDict
    | CacheUpdateDict
    | ChatAckDict
    | OverlayUpdateDict
    | TankEntryDict
    | TankExitDict
    | TankRemoveDict
    | PromotionDict
    | DecorationDict
    | BuildPickupDict
    | MapDataDict
    | ActionDoneDict
    | ChatMessageDict
    | StatisticsDict
    | ActiveForcesDict
    | ActivePlayersDict
    | Top10Dict
    | PingResponseDict
    | ConnectionLostDict
    | TankStatusSyncDict
    | TankStatusDict
    | SupervisorDict
    | SupervisorTextDict
    | TerrainUpdateDict
    | ViewportUpdateDict
    | ContainerMessage
)

DecodedMessage = TextMessage | BinaryMessage
