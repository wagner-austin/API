"""Shadow-law validators: law-obeying and law-violating timelines.

Each law gets an obeying case (exact) and corrupted cases (mismatch)
— the corrupted cases are the negative controls proving the validator
has teeth, plus every skip filter.
"""

from __future__ import annotations

from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.sim.equipment import MERCY_BUNDLE
from tankpit_bot.sim.server import CORPSE_WINDOW_TICKS
from tankpit_bot.validate.shadow_laws import (
    shadow_corpse_window,
    shadow_damage_tier,
    shadow_grant_invariants,
    shadow_mercy_bundle,
    shadow_sync_cadence,
)
from tankpit_bot.validate.shadow_timeline import (
    EquipmentGainEventDict,
    InventoryEventDict,
    KillEventDict,
    ShadowTimelineDict,
    TankExitEventDict,
    TankRemoveEventDict,
    TankSyncEventDict,
)

SELF_ID = 7
ENEMY_ID = 9
VICTIM_ID = 12

CORPSE_MS = CORPSE_WINDOW_TICKS * TICK_RATE_MS


def _timeline(
    syncs: list[TankSyncEventDict] | None = None,
    kills: list[KillEventDict] | None = None,
    gains: list[EquipmentGainEventDict] | None = None,
    removals: list[TankRemoveEventDict] | None = None,
    exits: list[TankExitEventDict] | None = None,
    inventories: list[InventoryEventDict] | None = None,
) -> ShadowTimelineDict:
    return ShadowTimelineDict(
        session_id="shadow-test",
        self_id=SELF_ID,
        names={},
        syncs=syncs if syncs is not None else [],
        kills=kills if kills is not None else [],
        gains=gains if gains is not None else [],
        removals=removals if removals is not None else [],
        exits=exits if exits is not None else [],
        inventories=inventories if inventories is not None else [],
        shots=[],
        positions=[],
    )


def _sync(
    timestamp_ms: int,
    tank_id: int,
    damage_state: int = 3,
    rank: int = 1,
    fuel: int | None = None,
) -> TankSyncEventDict:
    return TankSyncEventDict(
        timestamp_ms=timestamp_ms,
        tank_id=tank_id,
        damage_state=damage_state,
        rank=rank,
        fuel=fuel,
    )


def _syncs(tank_id: int, start: int, gap: int, count: int) -> list[TankSyncEventDict]:
    return [_sync(start + index * gap, tank_id) for index in range(count)]


def _kill(timestamp_ms: int, victim_id: int = VICTIM_ID, killer_id: int = SELF_ID) -> KillEventDict:
    return KillEventDict(
        timestamp_ms=timestamp_ms,
        victim_id=victim_id,
        killer_id=killer_id,
        is_mine_kill=False,
    )


def _gain(timestamp_ms: int, gained: list[int], show_message: bool) -> EquipmentGainEventDict:
    return EquipmentGainEventDict(
        timestamp_ms=timestamp_ms, show_message=show_message, gained=gained
    )


def _inventory(timestamp_ms: int, counts: list[int]) -> InventoryEventDict:
    return InventoryEventDict(timestamp_ms=timestamp_ms, counts=counts)


class TestSyncCadence:
    def test_tick_cadence_is_exact(self) -> None:
        timeline = _timeline(syncs=_syncs(ENEMY_ID, 0, TICK_RATE_MS, 8))
        evidence = shadow_sync_cadence([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_corrupted_cadence_is_a_mismatch(self) -> None:
        timeline = _timeline(syncs=_syncs(ENEMY_ID, 0, 5000, 8))
        evidence = shadow_sync_cadence([timeline])
        assert (evidence["samples"], evidence["mismatches"]) == (1, 1)

    def test_self_tank_is_excluded(self) -> None:
        timeline = _timeline(syncs=_syncs(SELF_ID, 0, TICK_RATE_MS, 8))
        evidence = shadow_sync_cadence([timeline])
        assert evidence["samples"] == 0

    def test_too_few_gaps_is_not_a_sample(self) -> None:
        timeline = _timeline(syncs=_syncs(ENEMY_ID, 0, TICK_RATE_MS, 3))
        evidence = shadow_sync_cadence([timeline])
        assert evidence["samples"] == 0

    def test_absence_holes_do_not_poison_the_median(self) -> None:
        """A 2 s core interrupted by viewport-exit holes stays exact.

        The 2026-08-03 sweep: 74/266 archive tanks "failed" the raw
        median purely from 18-943 s absence holes around clean 2 s
        cores — the holes are line-of-sight, not cadence.
        """
        core = _syncs(ENEMY_ID, 0, TICK_RATE_MS, 6)
        resumed = _syncs(ENEMY_ID, 600_000, TICK_RATE_MS, 6)
        timeline = _timeline(syncs=core + resumed)
        evidence = shadow_sync_cadence([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_holes_alone_leave_no_clean_gaps_to_judge(self) -> None:
        """A tank glimpsed only across absences is unobserved, not off-law."""
        timeline = _timeline(syncs=_syncs(ENEMY_ID, 0, 30_000, 8))
        evidence = shadow_sync_cadence([timeline])
        assert evidence["samples"] == 0


class TestGrantInvariants:
    def test_in_range_stack_roll_is_exact(self) -> None:
        timeline = _timeline(
            gains=[_gain(1000, [0, 7, 0, 0, 0], True)],
            inventories=[_inventory(1100, [5, 17, 5, 5, 3])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_radar_roll_is_exact(self) -> None:
        timeline = _timeline(
            gains=[_gain(1000, [0, 0, 0, 0, 3], True)],
            inventories=[_inventory(1100, [5, 5, 5, 5, 6])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_cap_clip_is_exact(self) -> None:
        timeline = _timeline(
            gains=[_gain(1000, [0, 2, 0, 0, 0], True)],
            inventories=[_inventory(1100, [5, 25, 5, 5, 3])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_two_slot_grant_is_a_mismatch(self) -> None:
        timeline = _timeline(
            gains=[_gain(1000, [0, 7, 0, 1, 0], True)],
            inventories=[_inventory(1100, [5, 17, 5, 6, 3])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert evidence["mismatches"] == 1

    def test_out_of_roll_amount_is_a_mismatch(self) -> None:
        timeline = _timeline(
            gains=[_gain(1000, [0, 2, 0, 0, 0], True)],
            inventories=[_inventory(1100, [5, 12, 5, 5, 3])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert evidence["mismatches"] == 1

    def test_negative_pre_is_a_mismatch(self) -> None:
        timeline = _timeline(
            gains=[_gain(1000, [0, 7, 0, 0, 0], True)],
            inventories=[_inventory(1100, [5, 5, 5, 5, 3])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert evidence["mismatches"] == 1

    def test_grant_at_cap_is_a_mismatch(self) -> None:
        timeline = _timeline(
            gains=[_gain(1000, [0, 0, 0, 0, 1], True)],
            inventories=[_inventory(1100, [5, 5, 5, 5, 26])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert evidence["mismatches"] == 1

    def test_overfull_post_is_a_mismatch(self) -> None:
        timeline = _timeline(
            gains=[_gain(1000, [0, 3, 0, 0, 0], True)],
            inventories=[_inventory(1100, [5, 27, 5, 5, 3])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert evidence["mismatches"] == 1

    def test_silent_and_unpaired_gains_are_skipped(self) -> None:
        timeline = _timeline(
            gains=[
                _gain(1000, list(MERCY_BUNDLE), False),
                _gain(50_000, [0, 7, 0, 0, 0], True),
            ],
            inventories=[_inventory(1100, [5, 7, 5, 5, 4])],
        )
        evidence = shadow_grant_invariants([timeline])
        assert evidence["samples"] == 0


class TestMercyBundle:
    def test_radar_zero_kill_with_bundle_is_exact(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            gains=[_gain(10_000, list(MERCY_BUNDLE), False)],
            inventories=[_inventory(5000, [5, 5, 5, 5, 0])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_radar_zero_kill_without_bundle_is_a_mismatch(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            inventories=[_inventory(5000, [5, 5, 5, 5, 0])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert evidence["mismatches"] == 1

    def test_stocked_kill_without_bundle_is_exact(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            inventories=[_inventory(5000, [5, 5, 5, 5, 3])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_stocked_kill_with_bundle_is_a_mismatch(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            gains=[_gain(10_000, list(MERCY_BUNDLE), False)],
            inventories=[_inventory(5000, [5, 5, 5, 5, 3])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert evidence["mismatches"] == 1

    def test_bundle_amount_outside_rolls_is_a_mismatch(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            gains=[_gain(10_000, [0, 5, 0, 1, 1], False)],
            inventories=[_inventory(5000, [5, 5, 5, 5, 0])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert evidence["mismatches"] == 1

    def test_kill_without_prior_snapshot_is_skipped(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            inventories=[_inventory(10_000, [5, 5, 5, 5, 0])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert evidence["samples"] == 0

    def test_mine_and_foreign_kills_are_skipped(self) -> None:
        mine_kill = KillEventDict(
            timestamp_ms=10_000, victim_id=VICTIM_ID, killer_id=SELF_ID, is_mine_kill=True
        )
        timeline = _timeline(
            kills=[mine_kill, _kill(20_000, killer_id=ENEMY_ID)],
            inventories=[_inventory(5000, [5, 5, 5, 5, 0])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert evidence["samples"] == 0

    def test_silent_bundle_outside_window_is_not_paired(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            gains=[_gain(60_000, list(MERCY_BUNDLE), False)],
            inventories=[_inventory(5000, [5, 5, 5, 5, 3])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_loud_gain_is_not_a_bundle(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            gains=[_gain(10_000, [0, 7, 0, 0, 0], True)],
            inventories=[_inventory(5000, [5, 5, 5, 5, 3])],
        )
        evidence = shadow_mercy_bundle([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)


class TestCorpseWindow:
    def test_window_gap_is_exact(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            removals=[TankRemoveEventDict(timestamp_ms=10_000 + CORPSE_MS, tank_id=VICTIM_ID)],
        )
        evidence = shadow_corpse_window([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_early_removal_is_a_mismatch(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            removals=[TankRemoveEventDict(timestamp_ms=18_000, tank_id=VICTIM_ID)],
        )
        evidence = shadow_corpse_window([timeline])
        assert evidence["mismatches"] == 1

    def test_no_removal_is_skipped(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            removals=[
                TankRemoveEventDict(timestamp_ms=5000, tank_id=VICTIM_ID),
                TankRemoveEventDict(timestamp_ms=40_000, tank_id=ENEMY_ID),
            ],
        )
        evidence = shadow_corpse_window([timeline])
        assert evidence["samples"] == 0

    def test_intervening_quit_is_skipped(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            removals=[TankRemoveEventDict(timestamp_ms=10_000 + CORPSE_MS, tank_id=VICTIM_ID)],
            exits=[TankExitEventDict(timestamp_ms=15_000, tank_id=VICTIM_ID)],
        )
        evidence = shadow_corpse_window([timeline])
        assert evidence["samples"] == 0

    def test_quit_outside_window_does_not_skip(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            removals=[TankRemoveEventDict(timestamp_ms=10_000 + CORPSE_MS, tank_id=VICTIM_ID)],
            exits=[TankExitEventDict(timestamp_ms=60_000, tank_id=VICTIM_ID)],
        )
        evidence = shadow_corpse_window([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)

    def test_id_reuse_sync_is_skipped(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            removals=[TankRemoveEventDict(timestamp_ms=10_000 + CORPSE_MS, tank_id=VICTIM_ID)],
            syncs=[_sync(20_000, VICTIM_ID)],
        )
        evidence = shadow_corpse_window([timeline])
        assert evidence["samples"] == 0

    def test_other_tank_sync_does_not_skip(self) -> None:
        timeline = _timeline(
            kills=[_kill(10_000)],
            removals=[TankRemoveEventDict(timestamp_ms=10_000 + CORPSE_MS, tank_id=VICTIM_ID)],
            syncs=[_sync(20_000, ENEMY_ID)],
        )
        evidence = shadow_corpse_window([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)


class TestDamageTier:
    def test_quartile_boundaries_are_exact(self) -> None:
        timeline = _timeline(
            syncs=[
                _sync(0, ENEMY_ID, damage_state=0, fuel=274),
                _sync(1000, ENEMY_ID, damage_state=1, fuel=275),
                _sync(2000, ENEMY_ID, damage_state=2, fuel=550),
                _sync(3000, ENEMY_ID, damage_state=3, fuel=825),
                _sync(4000, ENEMY_ID, damage_state=3, fuel=1100),
            ]
        )
        evidence = shadow_damage_tier([timeline])
        assert (evidence["samples"], evidence["exact"]) == (5, 5)

    def test_wrong_tier_is_a_mismatch(self) -> None:
        timeline = _timeline(syncs=[_sync(0, ENEMY_ID, damage_state=3, fuel=100)])
        evidence = shadow_damage_tier([timeline])
        assert evidence["mismatches"] == 1

    def test_short_form_sync_is_skipped(self) -> None:
        timeline = _timeline(syncs=[_sync(0, ENEMY_ID, damage_state=0, fuel=None)])
        evidence = shadow_damage_tier([timeline])
        assert evidence["samples"] == 0

    def test_capacity_is_rank_derived(self) -> None:
        timeline = _timeline(syncs=[_sync(0, ENEMY_ID, damage_state=0, rank=8, fuel=440)])
        evidence = shadow_damage_tier([timeline])
        assert (evidence["samples"], evidence["exact"]) == (1, 1)
