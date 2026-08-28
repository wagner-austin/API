"""Tests for :mod:`tankpit_bot.diagnostics.session_scorecard_accumulator`.

Covers record routing into the accumulator: every per-channel router,
the physics-divergence path, and teleport-spend routing.
"""

from __future__ import annotations

from tests.diagnostics._scorecard_fixtures import (
    _record,
    _routed,
)

from tankpit_bot.diagnostics.issue_report_types import (
    InventoryCountsDict,
)
from tankpit_bot.diagnostics.session_scorecard import build_session_scorecard
from tankpit_bot.diagnostics.session_scorecard_accumulator import route_scorecard_record
from tankpit_bot.diagnostics.session_scorecard_render import (
    collect_scorecard_issues,
    render_scorecard_section,
)
from tankpit_bot.diagnostics.session_scorecard_types import (
    FuelSampleRecordDict,
    new_scorecard_accumulator,
)
from tankpit_bot.runtime_records import RuntimeEventRecordDict


class TestRouting:
    """Tests for route_scorecard_record."""

    def test_routes_state_shots_kills_and_fuel(self) -> None:
        """Every scorecard-relevant record lands in its bucket."""
        accumulator = _routed(
            [
                _record(channel="STATE", message="INITIALIZING -> IDLE"),
                _record(channel="WIRE", message="shoot(136,149,id=529)"),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "tank_identity", "tank_id": 7},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "tank_deactivated",
                        "victim_id": 529,
                        "killer_id": 7,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "self_alignment_sample", "belief_fuel": 740},
                ),
            ]
        )

        assert accumulator["state_transitions"] == [("2026-06-12T06:25:00", "INITIALIZING -> IDLE")]
        assert accumulator["shots"] == 1
        assert accumulator["kills"] == 1
        assert accumulator["fuel_samples"] == [
            FuelSampleRecordDict(
                timestamp="2026-06-12T06:25:00",
                fuel=740,
                bot_state="",
                in_flight="none",
            )
        ]

    def test_tracks_duration_bounds_from_every_record(self) -> None:
        """Even unrelated channels move the duration bounds."""
        accumulator = _routed(
            [
                _record(channel="AI", message="anything", timestamp="2026-06-12T06:25:00"),
                _record(channel="WIRE", message="teleport(1,2)", timestamp="2026-06-12T06:27:30"),
            ]
        )

        assert accumulator["first_timestamp"] == "2026-06-12T06:25:00"
        assert accumulator["last_timestamp"] == "2026-06-12T06:27:30"
        assert accumulator["shots"] == 0

    def test_routes_self_statistics_to_career_totals(self) -> None:
        """``self_statistics`` populates the career-* fields from the wire."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "self_statistics",
                        "playtime_hours": 12,
                        "playtime_minutes": 34,
                        "playtime_seconds": 56,
                        "playtime_seconds_total": 12 * 3600 + 34 * 60 + 56,
                        "destroyed": 4271,
                        "deactivated": 1893,
                        "score": 1003500,
                    },
                )
            ]
        )

        assert accumulator["career_destroyed_last"] == 4271
        assert accumulator["career_deactivated_last"] == 1893
        assert accumulator["career_score_last"] == 1003500
        assert accumulator["career_playtime_seconds_last"] == 12 * 3600 + 34 * 60 + 56

    def test_routes_container_pickup_dispatched_to_full_and_partial(self) -> None:
        """``container_pickup_dispatched`` splits records into full/partial tallies."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "container_pickup_dispatched",
                        "x": 80,
                        "y": 90,
                        "remaining_volume": 0,
                        "is_partial": False,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "container_pickup_dispatched",
                        "x": 81,
                        "y": 91,
                        "remaining_volume": 881,
                        "is_partial": True,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "container_pickup_dispatched",
                        "x": 82,
                        "y": 92,
                        "remaining_volume": 0,
                        "is_partial": False,
                    },
                ),
            ]
        )

        assert accumulator["container_pickups_full"] == 2
        assert accumulator["container_pickups_partial"] == 1

    def test_ignores_irrelevant_diagnostics(self) -> None:
        """Unrelated diagnostic kinds leave the buckets untouched."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "map_positions_parsed", "tank_count": 37},
                )
            ]
        )

        assert accumulator["kills"] == 0
        assert accumulator["fuel_samples"] == []

    def test_routes_inventory_gains_scans_and_approaches(self) -> None:
        """The four observability diagnostics land in their buckets."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "inventory_sample",
                        "armor": 0,
                        "dual": 12,
                        "missile": 0,
                        "homing": 22,
                        "radar": 0,
                        "radar_enabled": True,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "equipment_gain",
                        "armor": 0,
                        "dual": 7,
                        "missile": 0,
                        "homing": 2,
                        "radar": 3,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "equipment_gain",
                        "armor": 1,
                        "dual": 0,
                        "missile": 0,
                        "homing": 3,
                        "radar": 0,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "radar_dispatch", "uses_extra": True},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "radar_dispatch", "uses_extra": False},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "radar_dispatch", "uses_extra": False},
                ),
            ]
        )

        assert accumulator["inventory_samples"] == [
            InventoryCountsDict(armor=0, dual=12, missile=0, homing=22, radar=0)
        ]
        assert accumulator["equipment_gain_events"] == 2
        assert accumulator["equipment_gained"] == InventoryCountsDict(
            armor=1, dual=7, missile=0, homing=5, radar=3
        )
        assert accumulator["scans_extra"] == 1
        assert accumulator["scans_builtin"] == 2

    def test_routes_combat_gate_diagnostics(self) -> None:
        """Combat-gate diagnostics each increment their dedicated counter.

        Locks the wiring added 2026-06-19 alongside the freshness
        refactor. Without these counters the miss path and damage
        transitions emit DIAGNOSTIC events the scorecard ignores --
        regression of this test would re-blind the scorecard. (The
        ghost / stale-position counters were removed 2026-08-28: their
        emitters died in the 0x29-departure and three-timestamp
        freshness reworks, so the counters read zero forever.)
        """
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "combat_miss",
                        "target_name": "orange-8",
                        "target_id": 534,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "combat_miss",
                        "target_name": "red-3",
                        "target_id": 211,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "tank_damage_changed",
                        "tank_id": 534,
                        "tank_name": "orange-8",
                        "previous_damage_state": 0,
                        "damage_state": 1,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "tank_damage_changed",
                        "tank_id": 211,
                        "tank_name": "red-3",
                        "previous_damage_state": 1,
                        "damage_state": 2,
                    },
                ),
            ]
        )

        assert accumulator["combat_misses"] == 2
        assert accumulator["tank_damage_changes"] == 2


def test_physics_divergence_routing_and_issue() -> None:
    """physics_divergence events count into the scorecard and raise an issue."""
    accumulator = new_scorecard_accumulator()
    route_scorecard_record(
        _record(
            channel="DIAGNOSTIC",
            fields={
                "diagnostic_kind": "physics_divergence",
                "residual": -45,
                "feasible_lo": -10,
                "feasible_hi": -10,
                "entry_kinds": "shot_dual",
                "fact_source": "wire_0x2E_tank_status_sync",
            },
        ),
        accumulator,
    )
    assert accumulator["physics_divergences"] == 1
    scorecard = build_session_scorecard(accumulator)
    assert scorecard["physics_divergences"] == 1
    assert "  physics divergences: 1" in render_scorecard_section(scorecard)
    issues = collect_scorecard_issues(scorecard)
    assert any("physics divergences: 1" in issue for issue in issues)


class TestTeleportSpendRouting:
    """Tests for the WORLD fuel-receipt teleport spend attribution."""

    @staticmethod
    def _world_fuel(
        message: str,
        *,
        in_flight: str = "teleport",
        bot_state: str = "HUNT/CLOSE",
    ) -> RuntimeEventRecordDict:
        """Build a WORLD-channel fuel transition record."""
        return _record(
            channel="WORLD",
            message=message,
            fields={"bot_state": bot_state, "in_flight_action_kind": in_flight},
        )

    def test_in_flight_debits_attribute_to_bot_state(self) -> None:
        """Teleport debits group by the ambient bot state."""
        accumulator = _routed(
            [
                self._world_fuel("Fuel: 1090 -> 823 (-267)", bot_state="COLLECT/SEARCH"),
                self._world_fuel("Fuel: 372 -> 214 (-158)"),
                self._world_fuel("Fuel: 214 -> 200 (-14)"),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["teleport_spend"] == [
            {"bot_state": "COLLECT/SEARCH", "drops": 1, "fuel_spent": 267},
            {"bot_state": "HUNT/CLOSE", "drops": 2, "fuel_spent": 172},
        ]
        assert scorecard["teleport_spend_total"] == 439

    def test_non_teleport_and_credit_receipts_are_ignored(self) -> None:
        """Only in-flight teleport DEBITS count."""
        accumulator = _routed(
            [
                self._world_fuel("Fuel: 500 -> 490 (-10)", in_flight="shoot"),
                self._world_fuel("Fuel: 490 -> 1100 (+610)"),
                self._world_fuel("Not a fuel line"),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["teleport_spend"] == []
        assert scorecard["teleport_spend_total"] == 0

    def test_damage_ledger_routes_billing_and_spend_bounds(self) -> None:
        """The end-of-run ledger populates billing counts and bounds."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "damage_ledger",
                        "teleport_fuel_lo": -19290,
                        "teleport_fuel_hi": -11993,
                        "shot_single_count": 6,
                        "shot_dual_count": 170,
                        "shot_homing_count": 72,
                    },
                )
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["ledger_teleport_spend_max"] == 19290
        assert scorecard["ledger_teleport_spend_min"] == 11993
        assert scorecard["ledger_shot_singles"] == 6
        assert scorecard["ledger_shot_duals"] == 170
        assert scorecard["ledger_shot_homings"] == 72

    def test_damage_ledger_without_fuel_fields_keeps_sentinels(self) -> None:
        """A pre-fuel-book ledger event leaves the -1 sentinels."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "damage_ledger", "dealt": "", "taken": ""},
                )
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["ledger_teleport_spend_max"] == -1
        assert scorecard["ledger_teleport_spend_min"] == -1
        assert scorecard["ledger_shot_singles"] == -1
