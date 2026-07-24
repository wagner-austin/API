"""Tests for the archive validators (windows, firing, walk, damage, capacity)."""

from __future__ import annotations

from tankpit_bot.protocol.commands import CMD_RADAR
from tankpit_bot.validate.archive import (
    validate_firing_costs,
    validate_fuel_capacity,
    validate_hit_damage,
    validate_radar_cost,
    validate_walk_cost,
)
from tankpit_bot.validate.windows import FuelWindowDict, build_fuel_windows
from tankpit_bot.validate.wire_timeline import (
    FuelReadingDict,
    SelfMoveDict,
    SentActionDict,
    ShotEchoDict,
    WireTimelineDict,
)


def _timeline(
    *,
    fuel_readings: list[FuelReadingDict] | None = None,
    own_shots: list[ShotEchoDict] | None = None,
    enemy_shots: list[ShotEchoDict] | None = None,
    sent_actions: list[SentActionDict] | None = None,
    self_moves: list[SelfMoveDict] | None = None,
    pickups: list[int] | None = None,
    detonations: list[int] | None = None,
    rank: int | None = 1,
) -> WireTimelineDict:
    """Build a timeline literal with the given event streams.

    Args:
        fuel_readings: Absolute fuel readings.
        own_shots: Own 0x53 echoes.
        enemy_shots: Enemy 0x53 echoes.
        sent_actions: Sent commands.
        self_moves: Own movement echoes.
        pickups: Container pickup sequence numbers.
        detonations: Mine detonation sequence numbers.
        rank: Session rank (None = unknown).

    Returns:
        A timeline for window tests.
    """
    return WireTimelineDict(
        session_id="t",
        self_id=7,
        rank=rank,
        fuel_readings=fuel_readings if fuel_readings is not None else [],
        own_shots=own_shots if own_shots is not None else [],
        enemy_shots=enemy_shots if enemy_shots is not None else [],
        sent_actions=sent_actions if sent_actions is not None else [],
        self_moves=self_moves if self_moves is not None else [],
        pickup_timestamps=pickups if pickups is not None else [],
        detonation_timestamps=detonations if detonations is not None else [],
        inventory_snapshots=[],
    )


def _window(
    *,
    delta: int,
    own: list[int] | None = None,
    enemy: list[int] | None = None,
    tiles: int = 0,
    echoes: int = 0,
    spending: int = 0,
    moves: int = 0,
    pickups: int = 0,
    detonations: int = 0,
) -> FuelWindowDict:
    """Build one fuel window literal.

    Args:
        delta: Fuel delta across the window.
        own: Own shot weapons in the window.
        enemy: Enemy shot weapons in the window.
        tiles: Tiles walked in the window.
        echoes: Own 0x47 movement echoes in the window.
        spending: Fuel-spending sent commands in the window.
        moves: CMD_MOVE sends among the spending commands.
        pickups: Container pickups in the window.
        detonations: Mine detonations in the window.

    Returns:
        The window record.
    """
    return FuelWindowDict(
        delta=delta,
        own_shot_weapons=own if own is not None else [],
        enemy_shot_weapons=enemy if enemy is not None else [],
        walked_tiles=tiles,
        move_echoes=echoes,
        spending_commands=spending,
        move_commands=moves,
        pickups=pickups,
        detonations=detonations,
    )


class TestBuildFuelWindows:
    """Window slicing between consecutive fuel readings."""

    def test_window_end_is_inclusive_start_exclusive(self) -> None:
        """Events at the closing reading timestamp count; at the opening they do not."""
        timeline = _timeline(
            fuel_readings=[
                FuelReadingDict(timestamp_ms=10, fuel=500, from_event=False),
                FuelReadingDict(timestamp_ms=20, fuel=490, from_event=False),
            ],
            own_shots=[
                ShotEchoDict(timestamp_ms=10, weapon=1),
                ShotEchoDict(timestamp_ms=15, weapon=1),
                ShotEchoDict(timestamp_ms=20, weapon=3),
            ],
            enemy_shots=[ShotEchoDict(timestamp_ms=16, weapon=0)],
            sent_actions=[
                SentActionDict(timestamp_ms=17, command=CMD_RADAR, x=0, y=0),
                SentActionDict(timestamp_ms=18, command=115, x=0, y=0),
            ],
            self_moves=[SelfMoveDict(timestamp_ms=12, tiles=4)],
            pickups=[19, 25],
            detonations=[13, 30],
        )
        windows = build_fuel_windows(timeline)
        assert windows == [
            FuelWindowDict(
                delta=-10,
                own_shot_weapons=[1, 3],
                enemy_shot_weapons=[0],
                walked_tiles=4,
                move_echoes=1,
                spending_commands=1,
                move_commands=0,
                pickups=1,
                detonations=1,
            )
        ]

    def test_fewer_than_two_readings_yield_no_windows(self) -> None:
        """One reading cannot form a delta."""
        reading = FuelReadingDict(timestamp_ms=1, fuel=500, from_event=False)
        timeline = _timeline(fuel_readings=[reading])
        assert build_fuel_windows(timeline) == []


class TestFiringCosts:
    """One-shot isolation windows re-derive the per-weapon costs."""

    def test_clean_windows_match_each_weapon(self) -> None:
        """Exact deltas for all four weapons produce exact evidence."""
        windows = [
            _window(delta=-6, own=[0]),
            _window(delta=-10, own=[1]),
            _window(delta=-10, own=[2]),
            _window(delta=-10, own=[3]),
        ]
        evidence = validate_firing_costs(windows)
        assert [(e["claim_id"], e["samples"], e["exact"], e["mismatches"]) for e in evidence] == [
            ("single-shot-cost", 1, 1, 0),
            ("dual-shot-cost", 1, 1, 0),
            ("missile-shot-cost", 1, 1, 0),
            ("homing-shot-cost", 1, 1, 0),
        ]

    def test_homing_split_debit_is_a_sample_but_not_a_mismatch(self) -> None:
        """A -5 homing window is the known split debit, not a contradiction."""
        evidence = validate_firing_costs([_window(delta=-5, own=[3])])
        homing = evidence[3]
        assert homing["claim_id"] == "homing-shot-cost"
        assert (homing["samples"], homing["exact"], homing["mismatches"]) == (1, 0, 0)

    def test_wrong_delta_is_a_mismatch(self) -> None:
        """A clean single-shot window at -7 contradicts the claim."""
        evidence = validate_firing_costs([_window(delta=-7, own=[0])])
        single = evidence[0]
        assert (single["samples"], single["exact"], single["mismatches"]) == (1, 0, 1)

    def test_contaminated_windows_are_excluded(self) -> None:
        """Multi-shot, enemy, walking, spending, pickup, and mine windows skip."""
        windows = [
            _window(delta=-12, own=[0, 0]),
            _window(delta=-6, own=[0], enemy=[1]),
            _window(delta=-6, own=[0], tiles=2),
            _window(delta=-6, own=[0], spending=1),
            _window(delta=-6, own=[0], pickups=1),
            _window(delta=-51, own=[0], detonations=1),
        ]
        evidence = validate_firing_costs(windows)
        assert all(record["samples"] == 0 for record in evidence)


class TestHitDamage:
    """Lone enemy-shot windows re-derive the victim costs."""

    def test_hits_zero_deltas_and_mismatches(self) -> None:
        """-45/-90 are exact, 0 is a non-sample, anything else mismatches."""
        windows = [
            _window(delta=-45, enemy=[0]),
            _window(delta=-90, enemy=[1]),
            _window(delta=0, enemy=[0]),
            _window(delta=-44, enemy=[0]),
            _window(delta=-45, enemy=[2]),
            _window(delta=-45, enemy=[0, 0]),
            _window(delta=-45, enemy=[0], own=[0]),
            _window(delta=-45, enemy=[0], tiles=1),
            _window(delta=-45, enemy=[0], spending=1),
            _window(delta=-45, enemy=[0], pickups=1),
            _window(delta=-90, enemy=[0], detonations=1),
            _window(delta=-45, enemy=[9]),
        ]
        evidence = validate_hit_damage(windows)
        single, dual, missile, homing = evidence
        assert single["claim_id"] == "single-hit-victim-cost"
        assert (single["samples"], single["exact"], single["mismatches"]) == (2, 1, 1)
        assert dual["claim_id"] == "dual-hit-victim-cost"
        assert (dual["samples"], dual["exact"], dual["mismatches"]) == (1, 1, 0)
        assert missile["claim_id"] == "missile-hit-victim-cost"
        assert (missile["samples"], missile["exact"], missile["mismatches"]) == (1, 1, 0)
        assert homing["claim_id"] == "homing-hit-victim-cost"
        assert (homing["samples"], homing["exact"], homing["mismatches"]) == (0, 0, 0)


class TestFuelCapacity:
    """Every reading must respect the rank-derived bound."""

    def test_bound_at_cap_and_above_cap(self) -> None:
        """Readings below, at, and above the rank-1 cap of 1100."""
        timelines = [
            _timeline(
                fuel_readings=[
                    FuelReadingDict(timestamp_ms=1, fuel=900, from_event=False),
                    FuelReadingDict(timestamp_ms=2, fuel=1100, from_event=False),
                    FuelReadingDict(timestamp_ms=3, fuel=1101, from_event=False),
                ],
                rank=1,
            ),
            _timeline(
                fuel_readings=[FuelReadingDict(timestamp_ms=1, fuel=9000, from_event=False)],
                rank=None,
            ),
        ]
        evidence = validate_fuel_capacity(timelines)
        assert evidence["claim_id"] == "fuel-capacity"
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (3, 2, 1)
        assert "1 readings AT the cap" in evidence["detail"]


class TestWalkEpisodes:
    """Walk episodes: movement windows summed until a quiet zero window."""

    def test_single_window_episode_is_exact(self) -> None:
        """A 2-tile walk fully drained before the quiet window is exact."""
        windows = [
            _window(delta=-2, tiles=2, echoes=1, spending=1, moves=1),
            _window(delta=0),
        ]
        evidence = validate_walk_cost(windows)
        assert evidence["claim_id"] == "walk-cost"
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (1, 1, 0)

    def test_drain_spread_across_silent_windows_is_exact(self) -> None:
        """A 4-tile walk draining over two silent windows still sums exactly."""
        windows = [
            _window(delta=-2, tiles=4, echoes=1, spending=1, moves=1),
            _window(delta=-2),
            _window(delta=0),
        ]
        evidence = validate_walk_cost(windows)
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (1, 1, 0)

    def test_wrong_total_is_a_mismatch(self) -> None:
        """An episode whose total contradicts 1/tile is a mismatch."""
        windows = [
            _window(delta=-3, tiles=2, echoes=1, spending=1, moves=1),
            _window(delta=0),
        ]
        evidence = validate_walk_cost(windows)
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (1, 0, 1)

    def test_episode_cut_by_foreign_event_is_discarded(self) -> None:
        """A shot before the quiet window voids the episode, unjudged."""
        windows = [
            _window(delta=-2, tiles=2, echoes=1, spending=1, moves=1),
            _window(delta=-6, own=[0]),
            _window(delta=0),
        ]
        evidence = validate_walk_cost(windows)
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)

    def test_episode_reaching_end_of_session_is_discarded(self) -> None:
        """A walk with no closing quiet window cannot be priced."""
        windows = [_window(delta=-2, tiles=2, spending=1, moves=1)]
        evidence = validate_walk_cost(windows)
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)

    def test_multi_echo_episode_is_discarded(self) -> None:
        """A second echo means a re-command; the full paths overcount tiles."""
        windows = [
            _window(delta=-2, tiles=4, echoes=1, spending=1, moves=1),
            _window(delta=-3, tiles=5, echoes=1, spending=1, moves=1),
            _window(delta=0),
        ]
        evidence = validate_walk_cost(windows)
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)

    def test_non_walk_windows_are_skipped(self) -> None:
        """Windows with foreign spending never start an episode."""
        windows = [
            _window(delta=-12, tiles=2, echoes=1, spending=2, moves=1),
            _window(delta=0),
        ]
        evidence = validate_walk_cost(windows)
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)


class TestRadarCost:
    """The lone-radar window recipe from the 2026-07-24 mining sweep."""

    @staticmethod
    def _radar_timeline(
        *,
        sent_actions: list[SentActionDict],
        readings: list[FuelReadingDict] | None = None,
        own_shots: list[ShotEchoDict] | None = None,
        pickups: list[int] | None = None,
    ) -> WireTimelineDict:
        """Build a two-reading timeline around one candidate window.

        Args:
            sent_actions: Sent commands for the session.
            readings: Fuel readings (defaults to a clean -10 window).
            own_shots: Own shot echoes.
            pickups: Pickup timestamps.

        Returns:
            Timeline with one window from t=10000 to t=12000.
        """
        return _timeline(
            fuel_readings=readings
            if readings is not None
            else [
                FuelReadingDict(timestamp_ms=10000, fuel=500, from_event=False),
                FuelReadingDict(timestamp_ms=12000, fuel=490, from_event=False),
            ],
            sent_actions=sent_actions,
            own_shots=own_shots,
            pickups=pickups,
        )

    def test_lone_radar_window_is_exact(self) -> None:
        """One radar, nothing else, delta -10: a clean exact sample."""
        timeline = self._radar_timeline(
            sent_actions=[SentActionDict(timestamp_ms=11000, command=CMD_RADAR, x=0, y=0)],
        )
        evidence = validate_radar_cost([timeline])
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (1, 1, 0)

    def test_wrong_delta_is_a_mismatch(self) -> None:
        """A lone-radar window whose delta is not -10 counts against the claim."""
        timeline = self._radar_timeline(
            sent_actions=[SentActionDict(timestamp_ms=11000, command=CMD_RADAR, x=0, y=0)],
            readings=[
                FuelReadingDict(timestamp_ms=10000, fuel=500, from_event=False),
                FuelReadingDict(timestamp_ms=12000, fuel=497, from_event=False),
            ],
        )
        evidence = validate_radar_cost([timeline])
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (1, 0, 1)

    def test_other_sent_command_excludes_the_window(self) -> None:
        """Any non-radar send in the window disqualifies it."""
        timeline = self._radar_timeline(
            sent_actions=[
                SentActionDict(timestamp_ms=11000, command=CMD_RADAR, x=0, y=0),
                SentActionDict(timestamp_ms=11500, command=115, x=0, y=0),
            ],
        )
        evidence = validate_radar_cost([timeline])
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)

    def test_two_radars_exclude_the_window(self) -> None:
        """Two radar sends in one window cannot price a single scan."""
        timeline = self._radar_timeline(
            sent_actions=[
                SentActionDict(timestamp_ms=10500, command=CMD_RADAR, x=0, y=0),
                SentActionDict(timestamp_ms=11500, command=CMD_RADAR, x=0, y=0),
            ],
        )
        evidence = validate_radar_cost([timeline])
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)

    def test_contamination_inside_the_window_excludes_it(self) -> None:
        """A shot echo inside the window dirties it."""
        timeline = self._radar_timeline(
            sent_actions=[SentActionDict(timestamp_ms=11000, command=CMD_RADAR, x=0, y=0)],
            own_shots=[ShotEchoDict(timestamp_ms=11500, weapon=0)],
        )
        evidence = validate_radar_cost([timeline])
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)

    def test_contamination_in_the_backward_guard_excludes_it(self) -> None:
        """A pickup within 3 s BEFORE the window dirties it (late debit)."""
        timeline = self._radar_timeline(
            sent_actions=[SentActionDict(timestamp_ms=11000, command=CMD_RADAR, x=0, y=0)],
            pickups=[9000],
        )
        evidence = validate_radar_cost([timeline])
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)

    def test_event_carried_reading_is_contamination(self) -> None:
        """A 0x44/0x64-sourced reading closing the window dirties it."""
        timeline = self._radar_timeline(
            sent_actions=[SentActionDict(timestamp_ms=11000, command=CMD_RADAR, x=0, y=0)],
            readings=[
                FuelReadingDict(timestamp_ms=10000, fuel=500, from_event=False),
                FuelReadingDict(timestamp_ms=12000, fuel=490, from_event=True),
            ],
        )
        evidence = validate_radar_cost([timeline])
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (0, 0, 0)

    def test_low_fuel_scan_clamps_to_remaining_fuel(self) -> None:
        """The radar debit is min(10, fuel): fuel 6 pays 6, fuel 0 pays 0."""
        clamp = self._radar_timeline(
            sent_actions=[SentActionDict(timestamp_ms=11000, command=CMD_RADAR, x=0, y=0)],
            readings=[
                FuelReadingDict(timestamp_ms=10000, fuel=6, from_event=False),
                FuelReadingDict(timestamp_ms=12000, fuel=0, from_event=False),
            ],
        )
        free = self._radar_timeline(
            sent_actions=[SentActionDict(timestamp_ms=11000, command=CMD_RADAR, x=0, y=0)],
            readings=[
                FuelReadingDict(timestamp_ms=10000, fuel=0, from_event=False),
                FuelReadingDict(timestamp_ms=12000, fuel=0, from_event=False),
            ],
        )
        evidence = validate_radar_cost([clamp, free])
        assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (2, 2, 0)
