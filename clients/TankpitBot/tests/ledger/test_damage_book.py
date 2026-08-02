"""Tests for :mod:`tankpit_bot.ledger.damage_book`."""

from __future__ import annotations

import pytest

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.ledger.damage_book import (
    IncomingShotContract,
    OwnShotEchoContract,
    confirm_incoming_damage,
    incoming_damage_window,
    make_damage_book,
    record_incoming_shot,
    record_own_shot_echo,
    resolve_dealt,
    summarize_side,
)
from tankpit_bot.physics.damage import (
    DUAL_HIT_VICTIM_COST,
    HOMING_HIT_VICTIM_COST,
    MISSILE_HIT_VICTIM_COST,
    SINGLE_HIT_VICTIM_COST,
)


def test_make_damage_book_is_empty() -> None:
    """A fresh book has no enemies and no pendings."""
    book = make_damage_book()
    assert book["dealt"] == {}
    assert book["taken"] == {}
    assert book["pending_dealt_weapon"] == -1
    assert book["pending_incoming"] == []
    assert book["confirmed_incoming"] == []


def test_resolve_dealt_charges_each_weapon_at_its_victim_cost() -> None:
    """Every paired weapon ledgers its measured victim cost."""
    book = make_damage_book()
    expected = {
        0: ("single", SINGLE_HIT_VICTIM_COST),
        1: ("dual", DUAL_HIT_VICTIM_COST),
        2: ("missile", MISSILE_HIT_VICTIM_COST),
        3: ("homing", HOMING_HIT_VICTIM_COST),
    }
    total = 0
    for weapon, (kind, cost) in expected.items():
        record_own_shot_echo(book, weapon)
        resolve_dealt(book, 535, "orange-9", 535)
        total += cost
        row = book["dealt"]["535"]
        counts = {
            "single": row["single"],
            "dual": row["dual"],
            "missile": row["missile"],
            "homing": row["homing"],
        }
        assert counts[kind] == 1
    assert row["fuel"] == total
    assert row["name"] == "orange-9"
    assert book["pending_dealt_weapon"] == -1


def test_resolve_dealt_without_pairing_counts_unknown_with_zero_fuel() -> None:
    """An unpaired hit is counted but never charges invented fuel."""
    book = make_damage_book()
    resolve_dealt(book, -1, "", -1)
    row = book["dealt"]["-1"]
    assert row["unknown"] == 1
    assert row["fuel"] == 0
    assert row["name"] == "tank--1"


def test_record_incoming_shot_counts_and_queues_confirmation() -> None:
    """Incoming shots ledger per shooter per weapon and queue pairing."""
    book = make_damage_book()
    record_incoming_shot(book, 2627, "guest", 1, 1000)
    record_incoming_shot(book, 2627, "guest", 3, 2000)
    row = book["taken"]["2627"]
    assert row["dual"] == 1
    assert row["homing"] == 1
    assert row["fuel"] == 0
    assert [p["cost"] for p in book["pending_incoming"]] == [
        DUAL_HIT_VICTIM_COST,
        HOMING_HIT_VICTIM_COST,
    ]


def test_record_incoming_shot_unknown_weapon_never_queues() -> None:
    """A weapon byte outside the vocabulary counts unknown, no pairing."""
    book = make_damage_book()
    record_incoming_shot(book, 2627, "guest", 9, 1000)
    assert book["taken"]["2627"]["unknown"] == 1
    assert book["pending_incoming"] == []


def test_confirm_incoming_damage_charges_covered_costs_oldest_first() -> None:
    """A fuel drop confirms as many queued shots as it covers."""
    book = make_damage_book()
    record_incoming_shot(book, 2627, "guest", 1, 1000)
    record_incoming_shot(book, 2627, "guest", 3, 1000)
    confirm_incoming_damage(book, -(DUAL_HIT_VICTIM_COST + 10), 2000)
    row = book["taken"]["2627"]
    assert row["fuel"] == DUAL_HIT_VICTIM_COST
    assert len(book["pending_incoming"]) == 1
    confirm_incoming_damage(book, -HOMING_HIT_VICTIM_COST, 3000)
    assert row["fuel"] == DUAL_HIT_VICTIM_COST + HOMING_HIT_VICTIM_COST
    assert book["pending_incoming"] == []


def test_confirm_incoming_damage_discards_expired_pendings() -> None:
    """A pending past its TTL never confirms, even against a big drop."""
    book = make_damage_book()
    record_incoming_shot(book, 2627, "guest", 1, 1000)
    confirm_incoming_damage(book, -500, 99999)
    assert book["taken"]["2627"]["fuel"] == 0
    assert book["pending_incoming"] == []


def test_confirm_incoming_damage_ignores_positive_deltas() -> None:
    """A fuel gain confirms nothing and keeps the queue."""
    book = make_damage_book()
    record_incoming_shot(book, 2627, "guest", 1, 1000)
    confirm_incoming_damage(book, 900, 2000)
    assert book["taken"]["2627"]["fuel"] == 0
    assert len(book["pending_incoming"]) == 1


def test_confirmed_hits_are_timestamped_for_the_rate_window() -> None:
    """Each fuel-confirmed hit lands in the window log at its confirm time."""
    book = make_damage_book()
    record_incoming_shot(book, 2627, "guest", 1, 1000)
    record_incoming_shot(book, 2627, "guest", 3, 1000)
    confirm_incoming_damage(book, -(DUAL_HIT_VICTIM_COST + HOMING_HIT_VICTIM_COST), 2000)
    assert book["confirmed_incoming"] == [
        {"timestamp_ms": 2000, "cost": DUAL_HIT_VICTIM_COST, "shooter_id": 2627},
        {"timestamp_ms": 2000, "cost": HOMING_HIT_VICTIM_COST, "shooter_id": 2627},
    ]


def test_incoming_damage_window_counts_and_prunes() -> None:
    """The window sums only in-window hits and prunes the stale tail."""
    book = make_damage_book()
    for confirm_ms in (1000, 5000, 12000):
        record_incoming_shot(book, 2627, "guest", 3, confirm_ms - 500)
        confirm_incoming_damage(book, -HOMING_HIT_VICTIM_COST, confirm_ms)
    hits, fuel = incoming_damage_window(book, 13000, 10000, frozenset())
    assert (hits, fuel) == (2, 2 * HOMING_HIT_VICTIM_COST)
    assert len(book["confirmed_incoming"]) == 2
    hits, fuel = incoming_damage_window(book, 30000, 10000, frozenset())
    assert (hits, fuel) == (0, 0)
    assert book["confirmed_incoming"] == []


def test_unconfirmed_shots_never_enter_the_rate_window() -> None:
    """A counted-but-unconfirmed shot cannot inflate the break rate."""
    book = make_damage_book()
    record_incoming_shot(book, 2627, "guest", 1, 1000)
    confirm_incoming_damage(book, 900, 2000)
    assert incoming_damage_window(book, 2000, 10000, frozenset()) == (0, 0)


def test_dead_shooters_hits_leave_the_rate_window() -> None:
    """A known-dead attacker's damage never projects into the next fight.

    The 2026-07-31 arena soak: after killing the attacker, its 5-hit
    window rate persisted and three healthy follow-up targets were
    blocked as "unwinnable at any fuel". Known-dead shooters are
    excluded; an UNKNOWN shooter still counts (a registry gap can
    never under-report live danger); and the entries stay in the log,
    so a respawned shooter's liveness flip restores them.
    """
    book = make_damage_book()
    record_incoming_shot(book, 2627, "guest", 1, 1000)
    confirm_incoming_damage(book, -DUAL_HIT_VICTIM_COST, 2000)
    record_incoming_shot(book, 3000, "stranger", 3, 2500)
    confirm_incoming_damage(book, -HOMING_HIT_VICTIM_COST, 3000)

    both = incoming_damage_window(book, 4000, 10000, frozenset())
    assert both == (2, DUAL_HIT_VICTIM_COST + HOMING_HIT_VICTIM_COST)

    guest_dead = incoming_damage_window(book, 4000, 10000, frozenset({2627}))
    assert guest_dead == (1, HOMING_HIT_VICTIM_COST)
    assert len(book["confirmed_incoming"]) == 2

    guest_respawned = incoming_damage_window(book, 4000, 10000, frozenset())
    assert guest_respawned == (2, DUAL_HIT_VICTIM_COST + HOMING_HIT_VICTIM_COST)


def test_summarize_side_renders_rows_and_empty() -> None:
    """The summary line lists nonzero weapons per enemy, or 'none'."""
    book = make_damage_book()
    assert summarize_side(book["dealt"]) == "none"
    record_own_shot_echo(book, 1)
    resolve_dealt(book, 535, "orange-9", 535)
    resolve_dealt(book, 500, "red-1", 500)
    text = summarize_side(book["dealt"])
    assert "red-1(500): unknown=1 fuel=0" in text
    assert f"orange-9(535): dual=1 fuel={DUAL_HIT_VICTIM_COST}" in text


def test_record_own_shot_echo_rejects_unknown_weapon() -> None:
    """The echo contract refuses weapon bytes outside the vocabulary."""
    book = make_damage_book()
    with pytest.raises(LedgerInvariantError):
        record_own_shot_echo(book, 7)


def test_record_incoming_shot_rejects_negative_shooter() -> None:
    """The incoming contract refuses negative shooter ids."""
    book = make_damage_book()
    with pytest.raises(LedgerInvariantError):
        record_incoming_shot(book, -5, "ghost", 1, 1000)


def test_record_incoming_shot_covers_single_and_missile() -> None:
    """Weapon bytes 0 and 2 ledger into their own columns."""
    book = make_damage_book()
    record_incoming_shot(book, 500, "red-1", 0, 1000)
    record_incoming_shot(book, 500, "red-1", 2, 1000)
    row = book["taken"]["500"]
    assert row["single"] == 1
    assert row["missile"] == 1
    assert [p["cost"] for p in book["pending_incoming"]] == [
        SINGLE_HIT_VICTIM_COST,
        MISSILE_HIT_VICTIM_COST,
    ]


def test_contract_names_identify_their_rules() -> None:
    """Each contract exposes its ledger-rule name."""
    assert OwnShotEchoContract().name == "damage_book_own_shot_echo"
    assert IncomingShotContract().name == "damage_book_incoming_shot"


def test_resolve_dealt_reroute_hit_charges_the_commanded_target() -> None:
    """An unresolvable victim (-1) ledgers under the intended id."""
    book = make_damage_book()
    record_own_shot_echo(book, 3)
    resolve_dealt(book, -1, "orange-9", 535)
    row = book["dealt"]["535"]
    assert row["homing"] == 1
    assert row["fuel"] == HOMING_HIT_VICTIM_COST
    assert "-1" not in book["dealt"]
