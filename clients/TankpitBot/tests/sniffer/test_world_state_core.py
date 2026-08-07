"""Tests for world-state core, rendering, rooms, and rejections.

``test_world_state_core.py`` was 701 lines; inventory detail is now a
sibling.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_inventory import (
    get_inventory_state,
    update_inventory_from_gain,
    update_inventory_from_protocol,
)
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestInventoryTracking:
    """Tests for binary protocol inventory tracking (0x49, 0x67, 0x74)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_full_signal_without_self_state_is_noop(self) -> None:
        """Code-7 reconciliation is a no-op when rank is unknown."""
        from tankpit_bot.sniffer.world_state_inventory import (
            update_inventory_from_full_signal,
        )

        assert update_inventory_from_full_signal(get_world_service()) == []
        inv = get_inventory_state(get_world_service())
        assert inv["dual_shots"]["count"] == 0

    def test_full_signal_reconciles_every_slot_to_capacity(self) -> None:
        """Code 7 is an authoritative all-slots-full statement.

        User mechanic (2026-07-18): containers fill whatever is empty
        and the server sends Inventory full only when ALL items are at
        cap -- so the rejection snaps every drifted shadow count up to
        the rank capacity.
        """
        from tankpit_bot.physics.capacity import inventory_capacity
        from tankpit_bot.sniffer.world_state import update_world_state_from_position
        from tankpit_bot.sniffer.world_state_inventory import (
            update_inventory_from_full_signal,
        )

        update_world_state_from_position(100, 100)
        ws = get_world_service()
        changes = update_inventory_from_full_signal(ws)
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("position update must create self_state")
        cap = inventory_capacity(self_state["rank"])
        assert len(changes) == 5
        inv = get_inventory_state(ws)
        assert inv["armor_shields"]["count"] == cap
        assert inv["dual_shots"]["count"] == cap
        assert inv["missile_shots"]["count"] == cap
        assert inv["homing_shots"]["count"] == cap
        assert inv["extra_radars"]["count"] == cap

    def test_initial_inventory_is_empty(self) -> None:
        """Test inventory starts at zero counts, all disabled."""
        inv = get_inventory_state(get_world_service())
        assert inv["armor_shields"]["count"] == 0
        assert inv["armor_shields"]["enabled"] is False
        assert inv["dual_shots"]["count"] == 0
        assert inv["dual_shots"]["enabled"] is False
        assert inv["missile_shots"]["count"] == 0
        assert inv["homing_shots"]["count"] == 0
        assert inv["extra_radars"]["count"] == 0

    def test_update_from_protocol_sets_absolute_counts(self) -> None:
        """Test 0x49 message sets absolute inventory counts."""
        update_inventory_from_protocol(
            get_world_service(),
            counts=[10, 5, 3, 2, 1],
            enabled=[True, False, True, True, False],
        )
        inv = get_inventory_state(get_world_service())
        assert inv["armor_shields"]["count"] == 10
        assert inv["dual_shots"]["count"] == 5
        assert inv["dual_shots"]["enabled"] is False
        assert inv["missile_shots"]["count"] == 3
        assert inv["homing_shots"]["count"] == 2
        assert inv["extra_radars"]["count"] == 1
        assert inv["extra_radars"]["enabled"] is False

    def test_update_from_protocol_returns_changes(self) -> None:
        """Test 0x49 message returns all changes from initial state.

        Initial state has enabled=False for all slots. First protocol
        message sets enabled=True + armor count=10, so all 5 slots
        report enabled_changed and armor also reports a count delta.
        """
        changes = update_inventory_from_protocol(
            get_world_service(),
            counts=[10, 0, 0, 0, 0],
            enabled=[True, True, True, True, True],
        )
        # All 5 slots changed enabled state (False→True), armor also changed count
        assert len(changes) == 5
        armor_change = next(c for c in changes if c["item"] == "armor_shields")
        assert armor_change["delta"] == 10
        assert armor_change["enabled_changed"] is True

    def test_update_from_protocol_no_changes_when_same(self) -> None:
        """Test 0x49 message returns empty when counts unchanged."""
        update_inventory_from_protocol(
            get_world_service(),
            counts=[10, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        changes = update_inventory_from_protocol(
            get_world_service(),
            counts=[10, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        assert changes == []

    def test_update_from_gain_adds_deltas(self) -> None:
        """Test 0x67 message adds gained amounts to current counts."""
        update_inventory_from_protocol(
            get_world_service(),
            counts=[10, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        update_inventory_from_gain(get_world_service(), gained=[5, 0, 2, 0, 3])

        inv = get_inventory_state(get_world_service())
        assert inv["armor_shields"]["count"] == 15
        assert inv["dual_shots"]["count"] == 5
        assert inv["missile_shots"]["count"] == 5
        assert inv["homing_shots"]["count"] == 2
        assert inv["extra_radars"]["count"] == 4

    def test_update_from_gain_returns_changes(self) -> None:
        """Test 0x67 message returns changes for non-zero gains."""
        update_inventory_from_protocol(
            get_world_service(),
            counts=[10, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        changes = update_inventory_from_gain(get_world_service(), gained=[5, 0, 0, 0, 0])
        assert len(changes) == 1
        assert changes[0]["item"] == "armor_shields"
        assert changes[0]["delta"] == 5
        assert changes[0]["old_count"] == 10
        assert changes[0]["new_count"] == 15

    def test_update_from_gain_preserves_enabled(self) -> None:
        """Test 0x67 message does not change enabled flags."""
        update_inventory_from_protocol(
            get_world_service(),
            counts=[10, 5, 3, 2, 1],
            enabled=[True, False, True, False, True],
        )
        update_inventory_from_gain(get_world_service(), gained=[1, 1, 1, 1, 1])

        inv = get_inventory_state(get_world_service())
        assert inv["armor_shields"]["enabled"] is True
        assert inv["dual_shots"]["enabled"] is False
        assert inv["homing_shots"]["enabled"] is False


class TestWorldStateCore:
    """Tests for sniffer world state core operations."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_reset_world_state_clears_state(self) -> None:
        """Test reset_world_state clears world state and terrain map."""
        update_world_state_from_position(100, 100)
        reset_world_state()

        svc = get_world_service()
        assert svc.world_state["self_state"] is None
        assert svc.terrain_map is None

    def test_load_terrain_map_returns_none_if_no_file(self) -> None:
        """Test returns None when no terrain file exists."""
        from tankpit_bot.sniffer.world_state import (
            register_room_image,
            set_selected_room,
        )

        register_room_image("1", "field01.gif")
        set_selected_room("1")
        _test_hooks.path_exists = lambda path: False

        result = get_world_service()._load_terrain_map_if_needed()
        assert result is None

    def test_load_terrain_map_caches_result(self) -> None:
        """Test terrain map is cached after first load."""
        from tankpit_bot.sniffer.world_state import (
            register_room_image,
            set_selected_room,
        )

        fake_terrain = InMemoryTerrainMap()

        register_room_image("1", "field01.gif")
        set_selected_room("1")
        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        svc = get_world_service()
        result1 = svc._load_terrain_map_if_needed()
        assert result1 is fake_terrain
        assert svc.terrain_map is fake_terrain

        result2 = svc._load_terrain_map_if_needed()
        assert result2 is fake_terrain

    def test_update_world_state_from_position(self) -> None:
        """Test updates self position in world state."""
        update_world_state_from_position(128, 64)

        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after position update")
        assert self_state["x"] == 128
        assert self_state["y"] == 64

    def test_update_world_state_from_position_updates_existing(self) -> None:
        """Test updates existing self position in world state."""
        # First call creates self_state
        update_world_state_from_position(100, 100)
        # Second call updates existing self_state
        update_world_state_from_position(200, 150)

        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after position update")
        assert self_state["x"] == 200
        assert self_state["y"] == 150

    def test_update_world_state_from_radar_containers(self) -> None:
        """Test updates containers from radar."""
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict

        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=50, y=60, volume=100),  # fuel with 100 units
            RadarContainerDict(x=55, y=65, volume=-1),  # equipment (volume=-1)
        ]
        mines: list[RadarMineDict] = []

        svc = get_world_service()
        update_world_state_from_radar(svc, containers, mines, [])

        assert "50,60" in svc.world_state["containers"]
        assert svc.world_state["containers"]["50,60"]["is_fuel"] is True
        assert "55,65" in svc.world_state["containers"]
        assert svc.world_state["containers"]["55,65"]["is_fuel"] is False

    def test_update_world_state_from_radar_mines(self) -> None:
        """Test updates mines from radar."""
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict

        containers: list[RadarContainerDict] = []
        mines: list[RadarMineDict] = [
            RadarMineDict(x=70, y=80, team=1),
            RadarMineDict(x=75, y=85, team=2),
        ]

        svc = get_world_service()
        update_world_state_from_radar(svc, containers, mines, [])

        assert "70,80" in svc.world_state["mines"]
        assert svc.world_state["mines"]["70,80"]["team"] == 1
        assert "75,85" in svc.world_state["mines"]


class TestWorldStateRendering:
    """Tests for world state ASCII rendering."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_render_world_state_ascii_returns_none_without_terrain(self) -> None:
        """Test returns None when no terrain file exists."""
        from tankpit_bot.sniffer.world_state import register_room_image, set_selected_room
        from tankpit_bot.sniffer.world_state_tiles import render_world_state_ascii

        register_room_image("1", "field01.gif")
        set_selected_room("1")
        _test_hooks.path_exists = lambda path: False

        result = render_world_state_ascii(get_world_service())
        assert result is None

    def test_render_world_state_ascii_with_terrain(self) -> None:
        """Test renders ASCII with terrain map."""
        from tankpit_bot.sniffer.world_state import register_room_image, set_selected_room
        from tankpit_bot.sniffer.world_state_tiles import render_world_state_ascii

        fake_terrain = InMemoryTerrainMap()
        register_room_image("1", "field01.gif")
        set_selected_room("1")
        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        update_world_state_from_position(128, 128)

        result = render_world_state_ascii(get_world_service())
        if result is None:
            raise AssertionError("expected string, got None")
        assert "Viewport:" in result
        assert "@" in result


class TestRoomTracking:
    """Tests for room image tracking and terrain map selection."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_register_room_image_stores_mapping(self) -> None:
        """Test register_room_image stores room-to-image mapping."""
        from tankpit_bot.sniffer.world_state import register_room_image

        register_room_image("2", "field42.gif")
        assert get_world_service().room_images["2"] == "field42.gif"

    def test_set_selected_room_tracks_selection(self) -> None:
        """Test set_selected_room stores selected room and resets terrain."""
        from tankpit_bot.sniffer.world_state import set_selected_room

        # Pre-load a terrain map so we can verify it gets reset
        fake_terrain = InMemoryTerrainMap()
        get_world_service().terrain_map = fake_terrain

        set_selected_room("2")
        assert get_world_service().selected_room == "2"
        assert get_world_service().terrain_map is None

    def test_load_terrain_uses_selected_room_image(self) -> None:
        """Test terrain loader uses field image from selected room."""
        from tankpit_bot.sniffer.world_state import (
            register_room_image,
            set_selected_room,
        )

        fake_terrain = InMemoryTerrainMap()

        register_room_image("2", "field42.gif")
        set_selected_room("2")

        _test_hooks.path_exists = lambda path: "field42" in str(path)
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        svc = get_world_service()
        result = svc._load_terrain_map_if_needed()
        assert result is fake_terrain
        assert svc.terrain_map is fake_terrain

    def test_load_terrain_tries_underscore_and_hyphen_suffix(self) -> None:
        """Test _find_field_gif tries both _r and -r suffixes."""
        # Only the -r variant exists
        _test_hooks.path_exists = lambda path: str(path) == "field42-r.gif"

        result = get_world_service()._find_field_gif("field42.gif")
        if result is None:
            raise AssertionError("expected Path, got None")
        assert str(result) == "field42-r.gif"

    def test_find_field_gif_underscore_variant(self) -> None:
        """Test _find_field_gif finds _r variant."""
        _test_hooks.path_exists = lambda path: str(path) == "field01_r.gif"

        result = get_world_service()._find_field_gif("field01.gif")
        if result is None:
            raise AssertionError("expected Path, got None")
        assert str(result) == "field01_r.gif"

    def test_find_field_gif_returns_none_when_missing(self) -> None:
        """Test _find_field_gif returns None when no file found."""
        _test_hooks.path_exists = lambda path: False

        result = get_world_service()._find_field_gif("field99.gif")
        assert result is None

    def test_load_terrain_returns_none_without_selected_room(self) -> None:
        """Test terrain loader requires an explicit selected room."""
        result = get_world_service()._load_terrain_map_if_needed()
        assert result is None

    def test_load_terrain_warns_when_gif_missing_for_room(self) -> None:
        """Test terrain loader returns none when selected room GIF is missing."""
        from tankpit_bot.sniffer.world_state import (
            register_room_image,
            set_selected_room,
        )

        register_room_image("5", "field99.gif")
        set_selected_room("5")

        _test_hooks.path_exists = lambda path: False

        result = get_world_service()._load_terrain_map_if_needed()
        assert result is None

    def test_load_terrain_returns_none_when_room_has_no_image(self) -> None:
        """Test terrain loader requires a registered image for the selected room."""
        from tankpit_bot.sniffer.world_state import set_selected_room

        # Select room "9" but never register an image for it
        set_selected_room("9")

        result = get_world_service()._load_terrain_map_if_needed()
        assert result is None

    def test_reset_clears_room_tracking(self) -> None:
        """Test reset_world_state clears room tracking state."""
        from tankpit_bot.sniffer.world_state import register_room_image, set_selected_room

        register_room_image("2", "field42.gif")
        set_selected_room("2")

        reset_world_state()

        assert get_world_service().room_images == {}
        assert get_world_service().selected_room is None


class TestFailedMoveTargets:
    """Tests for failed move target tracking."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_mark_and_check_failed_move(self) -> None:
        """Marking a move target as failed makes it show as failed."""
        from tankpit_bot.sniffer.world_state import (
            is_move_target_failed,
            mark_move_target_failed,
        )

        mark_move_target_failed(73, 158, 50000)
        assert is_move_target_failed(73, 158, 60000) is True

    def test_failed_move_expires_after_ttl(self) -> None:
        """Failed move target expires after the TTL passes."""
        from tankpit_bot.sniffer.world_state import (
            is_move_target_failed,
            mark_move_target_failed,
        )

        mark_move_target_failed(73, 158, 10000)
        # 50000 - 10000 = 40000 > 30000 TTL → expired
        assert is_move_target_failed(73, 158, 50000) is False

    def test_unfailed_target_returns_false(self) -> None:
        """Coordinates never marked as failed return False."""
        from tankpit_bot.sniffer.world_state import is_move_target_failed

        assert is_move_target_failed(50, 50, 100000) is False

    def test_clear_failed_move_targets_resets_all(self) -> None:
        """clear_failed_move_targets removes all recorded failures."""
        from tankpit_bot.sniffer.world_state import (
            is_move_target_failed,
            mark_move_target_failed,
        )

        mark_move_target_failed(73, 158, 90000)
        get_world_service().clear_failed_move_targets()
        assert is_move_target_failed(73, 158, 100000) is False

    def test_radar_refresh_clears_failed_moves(self) -> None:
        """Fresh radar data clears all failed move targets."""
        from tankpit_bot.sniffer.world_state import (
            is_move_target_failed,
            mark_move_target_failed,
        )

        mark_move_target_failed(73, 158, 90000)
        update_world_state_from_radar(get_world_service(), [], [], [])
        assert is_move_target_failed(73, 158, 100000) is False


class TestMovementRejections:
    """Tests for the movement-rejection record."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

    def test_recorded_rejections_count_inside_the_window(self) -> None:
        """Rejections inside the trailing window are counted."""
        from tankpit_bot.sniffer.world_state import (
            recent_movement_rejections,
            record_movement_rejection,
        )

        record_movement_rejection(95000)
        record_movement_rejection(97000)

        assert recent_movement_rejections(100000, 10000) == 2

    def test_stale_rejections_fall_out_of_the_window(self) -> None:
        """Rejections older than the window are pruned and not counted."""
        from tankpit_bot.sniffer.world_state import (
            recent_movement_rejections,
            record_movement_rejection,
        )

        record_movement_rejection(80000)
        record_movement_rejection(97000)

        assert recent_movement_rejections(100000, 10000) == 1
        assert recent_movement_rejections(120000, 10000) == 0

    def test_empty_record_counts_zero(self) -> None:
        """A fresh service has no rejections on record."""
        from tankpit_bot.sniffer.world_state import recent_movement_rejections

        assert recent_movement_rejections(100000, 10000) == 0
