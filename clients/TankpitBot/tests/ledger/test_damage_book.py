"""Tests for :mod:`tankpit_bot.ledger.damage_book`."""

from __future__ import annotations

import pytest

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.ledger.damage_book import (
    IncomingShotContract,
    OwnShotEchoContract,
    confirm_incoming_damage,
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
        resolve_dealt(book, 535, "orange-9")
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
    resolve_dealt(book, -1, "")
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


def test_summarize_side_renders_rows_and_empty() -> None:
    """The summary line lists nonzero weapons per enemy, or 'none'."""
    book = make_damage_book()
    assert summarize_side(book["dealt"]) == "none"
    record_own_shot_echo(book, 1)
    resolve_dealt(book, 535, "orange-9")
    resolve_dealt(book, 500, "red-1")
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
