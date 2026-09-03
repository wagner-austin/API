"""World state service — owns all mutable game state as instance attributes.

Replaces the 16 module-level globals in ``world_state.py`` with a single
injectable service. Dispatch modules receive a ``WorldService`` instance
instead of importing ``world_state`` as ``_ws`` and reaching into private
module attributes.

Production code creates one ``WorldService`` per session; tests create a
fresh instance per test (no global resets needed).
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.facts.source import FactSource
from tankpit_bot.inventory import (
    InventoryItem,
    InventoryState,
    ItemType,
)
from tankpit_bot.ledger.ammo_book import AmmoBookDict, make_ammo_book
from tankpit_bot.ledger.damage_book import DamageBookDict, make_damage_book
from tankpit_bot.ledger.fuel_book import FuelBookDict, make_fuel_book
from tankpit_bot.ledger.service import LedgerService
from tankpit_bot.resources import field_gif_path
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_service_beliefs import WorldServiceBeliefsMixin
from tankpit_bot.sniffer.world_service_movement import WorldServiceMovementMixin
from tankpit_bot.sniffer.world_service_radar import WorldServiceRadarMixin
from tankpit_bot.state import (
    WorldStateDict,
    make_empty_world_state,
    update_self_position,
    update_self_rank,
)
from tankpit_bot.state.types.self_account import SelfAccountDict, make_empty_self_account
from tankpit_bot.state.viewport_geometry import (
    viewport_visible_bounds,
)

log = get_logger(__name__)

ITEM_TYPES: list[ItemType] = [
    "armor_shields",
    "dual_shots",
    "missile_shots",
    "homing_shots",
    "extra_radars",
]

WEAPON_BYTE_TO_ITEM: dict[int, ItemType] = {
    1: "dual_shots",
    2: "missile_shots",
    3: "homing_shots",
}


def _make_empty_inventory() -> InventoryState:
    """Create an empty inventory state with all items at zero.

    Returns:
        InventoryState with all counts at 0 and enabled False.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=0, enabled=False),
        dual_shots=InventoryItem(count=0, enabled=False),
        missile_shots=InventoryItem(count=0, enabled=False),
        homing_shots=InventoryItem(count=0, enabled=False),
        extra_radars=InventoryItem(count=0, enabled=False),
    )


class WorldService(WorldServiceRadarMixin, WorldServiceMovementMixin, WorldServiceBeliefsMixin):
    """Owns all mutable game state for one session.

    Instance attributes mirror the 16 module-level globals that were
    previously in ``world_state.py``. Dispatch modules receive a
    ``WorldService`` instance and mutate it directly.
    """

    def __init__(self) -> None:
        """Initialize empty world state for a new session."""
        self.world_state: WorldStateDict = make_empty_world_state()
        self.fuel_book: FuelBookDict = make_fuel_book()
        self.ammo_book: AmmoBookDict = make_ammo_book()
        self.damage_book: DamageBookDict = make_damage_book()
        # The ledger sits here beside the three books because the wire
        # layer and the command layer must share ONE of it: the executor
        # records a teleport dispatch and the 0x5A dispatch handler reads
        # it back to spot server displacement. The service is the only
        # handle both layers hold ([[session-state-deglobalisation]]).
        self.ledger: LedgerService = LedgerService()
        self.terrain_map: _test_hooks.TerrainMapProtocol | None = None
        self.room_images: dict[str, str] = {}
        self.selected_room: str | None = None
        self.inventory_state: InventoryState = _make_empty_inventory()
        self.got_confirmed_hit: bool = False
        self.got_our_shot_response: bool = False
        # Tank id present on the target tile of our most recent shot, or
        # -1 if the tile was empty. Set by ``mark_combat_hit`` from the
        # 0x53 ShootEvent fields (since the 2026-06-19 decoder
        # unification). Consumed by combat_feedback to distinguish hits
        # on the intended target from incidental hits (e.g. homing
        # seeker locking onto a closer enemy than the bot commanded).
        self.last_shot_victim_id: int = -1
        # Tank id our planner intended to shoot at on the most recent
        # ``shoot`` dispatch (the ``target_id`` arg to ``shoot_at``).
        # Used by the 0x53 ShootEvent dispatcher to attribute homing /
        # missile seeker resolutions: when a tracking weapon resolves
        # at ``(tx, ty)``, the server's seeker chased the locked target
        # to that tile, so ``(tx, ty)`` is a fresh wire datum for the
        # locked target's position (and works even when the target is
        # off-viewport and 0x2E TankStatusSync stops broadcasting). The
        # dispatcher uses it to update the locked target's tracked
        # position so the next shot aims at fresh server-known coords.
        self.last_shot_combat_target_id: int = -1
        # Inventory snapshot taken at the most recent shoot dispatch.
        # Used by combat_feedback to confirm hits via ammo delta: the
        # server only debits dual / missile / homing ammo on a hit, so
        # a decrement in any of those slots between snapshot time and
        # feedback time confirms the shot landed -- even when the
        # 0x53 ``victim_id`` lookup misses because the target is off
        # the bot's viewport (live run 2026-06-24 12:43: 4 pursuit
        # homings logged as misses because ``_find_tank_at_tile`` could
        # not see purple-9 off-viewport). ``None`` when no shoot is
        # pending or the delta has already been consumed.
        self.pending_shot_inventory_snapshot: InventoryState | None = None
        # The in-flight GROUND-AIMED shot (clearance fire at a tile,
        # ``target_id == 0``), or dispatch_ms == 0 when none. Ground
        # shots have no tank target, so the combat feedback classifier
        # never resolves them — their own 0x53 echo is their receipt
        # (accepted, billed, fired) and their final ledger resolution
        # (``shoot:fired``). Before this existed, shoot was the one
        # action kind whose clearance dispatches ALL closed
        # ``superseded`` (2026-08-21 false liveness alarm). Written at
        # the executor's shoot dispatch, consumed by the tick loop's
        # ground-shot resolver.
        self.pending_ground_shot_aim_x: int = 0
        self.pending_ground_shot_aim_y: int = 0
        self.pending_ground_shot_dispatch_ms: int = 0
        # Name of the most recent wire command this session dispatched
        # ("map_open", "teleport", "scope(5)", ...). The map-modal
        # receipt: ANY dispatched action closes the server-side map,
        # and the client overlay can lag that closure by a tick — the
        # teleport precondition must read this, not the overlay alone
        # (run bot-20260901-032936 03:34:21-25: a scope pan between
        # the open and the teleport closed the map server-side while
        # the overlay still rendered, and the teleport drew cant_do).
        self.last_wire_command_name: str = ""
        # Undrained 0x41 deactivations, victim -> killer. The killer id
        # travels with the victim because the two consumers diverge:
        # the dead-tank registry takes every victim, but the session
        # kill count takes only victims THIS tank killed (fleet run
        # 2026-08-14: arterial banked artax's two kills with zero
        # shots fired when this was a bare victim set).
        self.killed_tank_ids: dict[int, int] = {}
        self.teleport_landed: bool = False
        self.radar_scan_complete: bool = False
        self.map_data_processed: bool = False
        self.map_data_ingested_ms: int = 0
        self.viewport_update_processed: bool = False
        # Stamped on every dispatched binary world message -- the only
        # truthful liveness signal for the GAME session. Session 3 of
        # run 20260730: the game socket died at 11:58:32, the page
        # auto-reconnected to the LOBBY (its new socket read OPEN, so
        # the page-health gate passed every tick), and the bot injected
        # map_open into a session the server no longer recognized for
        # 43 minutes (243 consecutive stalls, zero inbound world
        # traffic). Lobby text (ROOM_LIST/SELECT) takes the text route
        # and deliberately does NOT refresh this stamp.
        self.last_game_message_ms: int = 0
        self.pending_radar_empty_delta_ms: int = 0
        self.pending_radar_uses_extra: bool = True
        self.failed_move_targets: dict[str, int] = {}
        self.landing_refusals: dict[str, int] = {}
        self.displacement_tombstones: dict[str, int] = {}
        # Forage-frontier visit tombstones ([[equipment-system]]
        # staleness law, 2026-08-28): arrived-at or seen-empty block
        # centers, TTL'd so the frontier returns after the field has
        # had time to repopulate.
        self.forage_visited: dict[str, int] = {}
        # Unique container-tile sightings, "x,y" -> is_fuel — the
        # dwell-unbiased composition ledger behind the frontier's
        # equipment prior (flag-11 correction, 2026-09-02: block
        # equipment FRACTION is real map signal, p<0.017, while
        # visit-count signals are the retired atlas's dwell trap). A
        # tile records once and re-observation rewrites the same
        # entry, so staring at a block adds nothing. Swept from the
        # container beliefs once per tick
        # (``record_container_sightings``).
        self.container_kind_sightings: dict[str, bool] = {}
        self.movement_rejections: list[int] = []
        # The canonical account-identity model ([[tank-registry]] rank
        # number; state/types/self_account.py) — session-stable "who
        # am I" facts, filled by the self 0x21 TankInfo and the
        # startup stats-panel scrape. Runtime features consult this
        # instead of re-fishing diagnostic streams.
        self.self_account: SelfAccountDict = make_empty_self_account()
        # Tank ids that have sent any (non-self-echo) chat this
        # session. One half of the human-consent combat contract
        # (user ruling 2026-07-30, session 8 killed over it: "to
        # engage in combat, the human must respond hello or engage
        # the bot first"); the other half is the damage book's
        # "taken" side recording who shot us.
        self.chat_seen_tank_ids: set[int] = set()
        # Stamped when a remembered container pickup comes back
        # code=4 (empty) -- the belief the planner acted on is
        # disproven, so the local memory of this area is desynced.
        # User ruling 2026-07-30 ("if one item is stale or out of
        # sync then its worth a radar. not, 3 items"): session 4 spent
        # three larder hops on containers Yuppler had already
        # collected, each landing scan suppressed as verified stock.
        # Cleared by the next radar response, which reconciles the
        # viewport authoritatively.
        self.container_desync_ms: int = 0
        # Last wire-announced gain of our own (fuel-total announce or
        # an inventory count rising) -- the code=4 drain-vs-stale
        # discriminator reads it (world_service_beliefs).
        self.last_own_gain_ms: int = 0
        self.failed_scan_viewports: dict[str, int] = {}
        self.last_command_error: int = -1
        # Last promotion-progress bar value seen on a self 0x2E, so the
        # telemetry emits on CHANGE rather than once per status
        # message. -1 is "nothing seen yet", distinct from a real 0.
        self.last_self_promo_state: int = -1
        # Set by the 0x41 dispatch when the wire announces OUR OWN
        # death; the tick loop converts it into the ``deactivated``
        # session exit (a corpse has no decisions left).
        self.self_deactivated: bool = False
        # Wall-clock stamp of the last 0x45 detonation on OUR OWN
        # tile -- the wire signature of a walk-over mine hit (45
        # fuel, movement arrested). Drives the user's 2026-07-30
        # movement doctrine: walk in-viewport, but after a mine
        # hit approach the SAME destination by teleport (landings
        # are mine-immune by the displacement law), then resume
        # walking.
        self.last_own_mine_hit_ms: int = 0
        # Own-mine-hit reveal-scan latch: set by the walk-over stamp,
        # cleared by the radar response (world_service_radar), read by
        # COLLECT's mine-reveal gate.
        self.mine_reveal_pending_ms: int = 0
        # Container tombstones ([[fleet-coordination]] negative
        # knowledge): {coord key: disproof wall-clock ms} stamped by
        # every local removal (code-4 disproof, emptied pickup,
        # unreachable). The fleet merge admits a remote sighting only
        # when OBSERVED AFTER the disproof -- without this, a
        # teammate that still believes in a dead container re-imports
        # it every exchange and the pickup loop never converges (run
        # arterial 2026-08-14 19:20: (102,85) disproved three times
        # in five seconds, re-imported between each).
        self.container_disproofs: dict[str, int] = {}
        # Teammates' held combat locks from the fleet knowledge
        # exchange ([[fleet-coordination]]): {target_id: freshest
        # report written_ms}. REPLACED wholesale by each merge pass,
        # so a teammate that disengages or goes silent stops steering
        # acquisition within one exchange. Threat ranking prefers
        # these ids inside a priority tier (focus fire).
        self.fleet_engaged_target_ids: dict[int, int] = {}
        # Siblings' shared collect intents ([[fleet-coordination]],
        # 2026-08-28): latched forage-frontier goals by instance, and
        # container tiles a sibling's collect plan holds. Replaced
        # wholesale each merge pass, like the engaged ids above.
        self.fleet_forage_goals: dict[str, tuple[int, int]] = {}
        self.fleet_claimed_containers: set[str] = set()
        # The container tile whose AUTHORITATIVE claim file this
        # session owns (-1,-1 for none) — the exclusive-create mutex
        # of [[fleet-forage-allocation]], reconciled against the held
        # collect plan by the tick loop's claim arbitration. Session
        # bookkeeping like the fields above: the advisory rows steer
        # planning, this pair records what the filesystem granted.
        self.held_claim_x: int = -1
        self.held_claim_y: int = -1
        # Own claim DENIALS, tile key -> stamp — the arbitration's
        # local memory. Needed because the advisory set above is
        # replaced wholesale each merge, so a denial stamped there
        # would not outlive the tick, and a winner that crashed after
        # claiming never publishes its advisory row at all. Pruned
        # each arbitration pass; read through
        # ``fresh_denied_claim_tiles`` at ctx construction.
        self.claim_denied_tiles: dict[str, int] = {}
        # Consent evidence inherited from same-color siblings
        # ([[fleet-coordination]], operator ruling 2026-08-26): ids the
        # FLEET has proof consented to combat. Replaced wholesale per
        # merge, like the engaged ids above.
        self.fleet_consented_tank_ids: set[int] = set()
        # Siblings' own tank ids, from their reports' identity field.
        # Replaced wholesale per merge like every fleet_* field. The
        # settled-knowledge law's exclusion set: fleet bots carry
        # human-style account names, and counting a sibling as "a
        # human is about" would keep every fleet room permanently
        # unsettled ([[flag-triage-20260902]] rows 3-5).
        self.fleet_sibling_tank_ids: set[int] = set()
        # The settled-knowledge watermark: newest observation stamp of
        # any FOREIGN human tank (not self, not a sibling) this
        # session ever saw. Monotonic and never pruned — a human who
        # left the room still bounds how far back scan knowledge is
        # trusted. 0 = never; read through ``knowledge_floor_ms``.
        self.last_foreign_human_seen_ms: int = 0
        # How many same-team siblings currently report themselves
        # war-ready (past the wartime readiness floor under a
        # war-joining doctrine). The swarm muster's quorum input
        # (operator order 2026-09-01); wholesale-replaced per
        # exchange like every fleet_* field.
        self.fleet_war_ready_count: int = 0
        # ContainerPickup de-duplication. The server broadcasts each
        # 0x43 pickup TWICE within ~200 ms (one to the picker, one to
        # the world view) -- empirical 43.9% duplicate rate across 13
        # sniff sessions (2026-06-20). The dispatcher records each
        # pickup signature here with its receipt timestamp; the next
        # arrival of the same signature within
        # ``PICKUP_DEDUP_WINDOW_MS`` is suppressed.
        self.recent_pickup_signatures: dict[tuple[tuple[int, int, int], ...], int] = {}
        # Career stats from the most recent 0x56 Statistics broadcast.
        # The wire sends these every ~10 s; the AI uses ``destroyed`` to
        # gate "I have N kills, time to play conservatively" decisions,
        # ``deactivated`` for the K/D ratio in scorecards, and
        # ``playtime_seconds_total`` to enforce a session length cap
        # without depending on wall-clock drift. ``-1`` means the wire
        # has not yet sent a Statistics broadcast this session.
        self.career_destroyed: int = -1
        self.career_deactivated: int = -1
        self.career_score: int = -1
        self.career_playtime_seconds_total: int = -1
        self.career_stats_last_update_ms: int = 0
        # 0x2F ActivePlayers roster of (tank_id, rank) tuples from the
        # most recent server broadcast. Empty until the server sends
        # the first one (usually after join-confirm).
        self.active_players: list[tuple[int, int]] = []
        # 0x31 Top10 latest snapshot. ``-1`` means the wire hasn't sent
        # a Top10 broadcast yet.
        self.top10_viewer_score: int = -1
        self.top10_viewer_position: int = -1
        self.top10_team_filter: int = -1
        # 0x60 PingResponse wall-clock. The bot's session-health
        # monitor reads this to detect long server silences without
        # racing the WebSocket layer.
        self.last_ping_response_ms: int = 0
        # Message ID of the last 0x4D chat the server echoed back for
        # OUR OWN tank (-1 before the first). The echo is the only
        # confirmation a sent chat survived the server-side flood
        # mute ([[chat-messages]], sniff-20260729-214411).
        self.last_chat_echo_message_id: int = -1
        # 0x4C MapData fuel-dot atlas -- the map's yellow-pixel fuel
        # positions. Server-cached per session (byte-identical across
        # map opens), so each MapData dispatch simply overwrites it.
        # Empty until the first map open. Consumed by the dot-hop
        # restock picker and the dot-relay travel planner.
        self.map_fuel_dots: tuple[tuple[int, int], ...] = ()

    # -----------------------------------------------------------------
    # World state accessors
    # -----------------------------------------------------------------

    def get_world_state(self) -> WorldStateDict:
        """Get the current world state.

        Returns:
            Current WorldStateDict with containers, mines, self_state, etc.
        """
        return self.world_state

    def get_terrain_map(self) -> _test_hooks.TerrainMapProtocol | None:
        """Get the current terrain map, loading if needed.

        Returns:
            TerrainMap instance, or None if terrain GIF not found.
        """
        return self._load_terrain_map_if_needed()

    # -----------------------------------------------------------------
    # Radar / scan event flags
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Radar cache refresh tracking
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Position updates
    # -----------------------------------------------------------------

    def update_world_state_from_position(
        self,
        x: int,
        y: int,
        fact_source: FactSource = "wire_0x3D_movement",
    ) -> None:
        """Update world state with new self position.

        Args:
            x: Self X coordinate.
            y: Self Y coordinate.
            fact_source: Wire channel the position arrived on.
        """
        self.world_state = update_self_position(
            self.world_state, x, y, _test_hooks.get_current_time_ms(), fact_source
        )

    def update_world_state_from_rank(self, rank: int, fact_source: FactSource) -> None:
        """Apply a wire-observed self rank to world state.

        A mid-session promotion flips the rank field of self-addressed
        0x2E/0x47/0x3D statements the same tick as the promoting kill
        (measured bot-20260725-211120); every rank-derived readiness
        bar and capacity reads ``self_state["rank"]``, so the update
        must land the tick it arrives.

        Args:
            rank: Wire-observed rank of the self tank.
            fact_source: Wire channel the rank arrived on.
        """
        current = self.world_state["self_state"]
        if current is not None and current["rank"] != rank:
            log.info("RANK: self rank %d -> %d (%s)", current["rank"], rank, fact_source)
        self.world_state = update_self_rank(
            self.world_state, rank, _test_hooks.get_current_time_ms(), fact_source
        )

    # -----------------------------------------------------------------
    # Failed move / scan tracking
    # -----------------------------------------------------------------

    def record_self_identity(
        self,
        name: str,
        persistent_tank_id: int,
        decoration_state_hex: str,
        timestamp_ms: int,
    ) -> None:
        """Record the self tank's wire identity (0x21 TankInfo).

        Args:
            name: In-game tank name.
            persistent_tank_id: Cross-session account id.
            decoration_state_hex: Cosmetic skin bytes, hex-encoded.
            timestamp_ms: When the identity arrived.
        """
        self.self_account["name"] = name
        self.self_account["persistent_tank_id"] = persistent_tank_id
        self.self_account["decoration_state_hex"] = decoration_state_hex
        self.self_account["identity_observed_ms"] = timestamp_ms

    def record_account_stats(
        self,
        *,
        rank_name: str,
        leaderboard_position: int,
        promotion_points: int,
        destroyed_enemies: int,
        deactivated_total: int,
        play_time_s: int,
        timestamp_ms: int,
    ) -> None:
        """Record the startup stats-panel scrape.

        Args:
            rank_name: Panel rank label.
            leaderboard_position: The countdown rank number.
            promotion_points: Lifetime promotion points.
            destroyed_enemies: Lifetime kills.
            deactivated_total: Lifetime own-deactivations.
            play_time_s: Lifetime play seconds.
            timestamp_ms: When the scrape was taken.
        """
        self.self_account["rank_name"] = rank_name
        self.self_account["leaderboard_position"] = leaderboard_position
        self.self_account["promotion_points"] = promotion_points
        self.self_account["destroyed_enemies"] = destroyed_enemies
        self.self_account["deactivated_total"] = deactivated_total
        self.self_account["play_time_s"] = play_time_s
        self.self_account["stats_observed_ms"] = timestamp_ms

    # -----------------------------------------------------------------
    # Room / terrain map management
    # -----------------------------------------------------------------

    def register_room_image(self, room_id: str, image: str) -> None:
        """Register a room's field image from a ROOM_LIST message.

        Args:
            room_id: Room ID (e.g. "2").
            image: Field image filename (e.g. "field42.gif").
        """
        self.room_images[room_id] = image

    def set_selected_room(self, room_id: str) -> None:
        """Track which room was selected from a SELECT message.

        Resets the terrain map so the correct one loads on next render.

        Args:
            room_id: Room ID that was selected.
        """
        self.selected_room = room_id
        self.terrain_map = None
        image = self.room_images.get(room_id)
        log.info("Selected room %s (field image: %s)", room_id, image or "unknown")
        emit_diagnostic(
            diagnostic_kind="session_room_joined",
            room_id=room_id,
            field_image=image if image is not None else "unknown",
        )

    # -----------------------------------------------------------------
    # Viewport / radar geometry helpers
    # -----------------------------------------------------------------

    def viewport_bounds(self) -> tuple[int, int, int, int]:
        """Return inclusive visible viewport bounds.

        Returns:
            Inclusive ``(left, top, right, bottom)`` viewport bounds.
        """
        return viewport_visible_bounds(self.world_state["viewport"])

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _load_terrain_map_if_needed(self) -> _test_hooks.TerrainMapProtocol | None:
        """Load terrain map for the selected room.

        Returns:
            TerrainMap instance, or None if file not found.
        """
        if self.terrain_map is not None:
            return self.terrain_map

        if self.selected_room is None:
            log.warning("No selected room is available for terrain-map loading")
            return None
        image = self.room_images.get(self.selected_room)
        if image is None:
            log.warning("No registered room image for selected room %s", self.selected_room)
            return None
        gif_path = field_gif_path(image)
        if gif_path is None:
            log.warning("No local GIF found for %s (room %s)", image, self.selected_room)
            return None
        self.terrain_map = _test_hooks.load_terrain_map(gif_path)
        log.info("Loaded terrain map from %s (room %s)", gif_path, self.selected_room)
        return self.terrain_map


__all__ = [
    "ITEM_TYPES",
    "WEAPON_BYTE_TO_ITEM",
    "WorldService",
]
