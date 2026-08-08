"""Tests for the live double-entry fuel book."""

from __future__ import annotations

import pytest

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.ledger.fuel_book import (
    FUEL_ENTRY_KINDS,
    FuelEntryContract,
    FuelReadingContract,
    make_fuel_book,
    record_fuel_entry,
    record_fuel_reading,
)
from tankpit_bot.sniffer.world_service import WorldService


def test_first_reading_anchors_without_verdict() -> None:
    """The first absolute reading opens the account and judges nothing."""
    book = make_fuel_book()
    assert record_fuel_reading(book=book, fuel_total=1000) is None
    assert book["last_fuel"] == 1000
    assert book["windows"] == 0
    assert book["divergences"] == 0


def test_exact_debit_block_balances_at_quiet_boundary() -> None:
    """A dual shot followed by exactly -10 balances once the wire quiets."""
    book = make_fuel_book()
    record_fuel_reading(book=book, fuel_total=1000)
    record_fuel_entry(book=book, kind="shot_dual", lo=-10, hi=-10)
    assert record_fuel_reading(book=book, fuel_total=990) is None
    verdict = record_fuel_reading(book=book, fuel_total=990)
    if verdict is None:
        raise AssertionError("expected a block verdict")
    assert verdict["balanced"] is True
    assert (verdict["residual"], verdict["lo"], verdict["hi"]) == (-10, -10, -10)
    assert verdict["entry_kinds"] == "shot_dual"
    assert (book["windows"], book["divergences"]) == (1, 0)


def test_unexplained_drain_is_a_divergence() -> None:
    """A fuel drop with no entries is exactly what the book must catch."""
    book = make_fuel_book()
    record_fuel_reading(book=book, fuel_total=1000)
    assert record_fuel_reading(book=book, fuel_total=955) is None
    verdict = record_fuel_reading(book=book, fuel_total=955)
    if verdict is None:
        raise AssertionError("expected a block verdict")
    assert verdict["balanced"] is False
    assert verdict["entry_kinds"] == "(none)"
    assert (book["windows"], book["divergences"]) == (1, 1)


def test_optional_and_ranged_entries_widen_the_interval() -> None:
    """Enemy hits and walks may or may not have cost fuel."""
    book = make_fuel_book()
    record_fuel_reading(book=book, fuel_total=1000)
    record_fuel_entry(book=book, kind="enemy_hit", lo=-90, hi=0)
    record_fuel_entry(book=book, kind="walk", lo=-7, hi=0)
    for fuel_total, expect_balanced in ((1000, True), (903, True), (902, False), (1001, False)):
        fresh = make_fuel_book()
        record_fuel_reading(book=fresh, fuel_total=1000)
        record_fuel_entry(book=fresh, kind="enemy_hit", lo=-90, hi=0)
        record_fuel_entry(book=fresh, kind="walk", lo=-7, hi=0)
        assert record_fuel_reading(book=fresh, fuel_total=fuel_total) is None
        verdict = record_fuel_reading(book=fresh, fuel_total=fuel_total)
        if verdict is None:
            raise AssertionError("expected a block verdict")
        assert verdict["balanced"] is expect_balanced


def test_pickup_credit_explains_gains() -> None:
    """An open pickup credit explains any gain up to its ceiling."""
    book = make_fuel_book()
    record_fuel_reading(book=book, fuel_total=400)
    record_fuel_entry(book=book, kind="pickup", lo=0, hi=1100)
    assert record_fuel_reading(book=book, fuel_total=1100) is None
    verdict = record_fuel_reading(book=book, fuel_total=1100)
    if verdict is None:
        raise AssertionError("expected a block verdict")
    assert verdict["balanced"] is True


def test_homing_shot_seeds_a_carry_into_the_next_window() -> None:
    """A -5/-5 split across the sync boundary balances both windows."""
    book = make_fuel_book()
    record_fuel_reading(book=book, fuel_total=1000)
    record_fuel_entry(book=book, kind="shot_homing", lo=-10, hi=-5)
    assert record_fuel_reading(book=book, fuel_total=995) is None
    first = record_fuel_reading(book=book, fuel_total=995)
    if first is None:
        raise AssertionError("expected a block verdict")
    assert first["balanced"] is True
    assert [entry["kind"] for entry in book["entries"]] == ["homing_carry"]
    assert record_fuel_reading(book=book, fuel_total=990) is None
    second = record_fuel_reading(book=book, fuel_total=990)
    if second is None:
        raise AssertionError("expected a block verdict")
    assert second["balanced"] is True
    assert second["entry_kinds"] == "homing_carry"


def test_inverted_interval_is_rejected() -> None:
    """An entry whose floor exceeds its ceiling violates the contract."""
    book = make_fuel_book()
    with pytest.raises(LedgerInvariantError):
        record_fuel_entry(book=book, kind="walk", lo=0, hi=-5)


def test_negative_reading_is_rejected() -> None:
    """The wire never reports negative fuel; a negative reading is a bug."""
    book = make_fuel_book()
    with pytest.raises(LedgerInvariantError):
        record_fuel_reading(book=book, fuel_total=-1)


def test_entry_flood_is_rejected() -> None:
    """A book that never reconciles must not grow without bound."""
    book = make_fuel_book()
    for _ in range(10_000):
        book["entries"].append({"kind": "walk", "lo": -1, "hi": 0})
    with pytest.raises(LedgerInvariantError):
        record_fuel_entry(book=book, kind="walk", lo=-1, hi=0)


def test_every_kind_is_recordable() -> None:
    """The declared kind vocabulary round-trips through the contract."""
    book = make_fuel_book()
    for kind in FUEL_ENTRY_KINDS:
        record_fuel_entry(book=book, kind=kind, lo=-1, hi=0)
    assert len(book["entries"]) == len(FUEL_ENTRY_KINDS)


def test_contract_names_identify_the_invariants() -> None:
    """Contract names appear in enforcement errors; pin them."""
    assert FuelEntryContract().name == "fuel_book_entry"
    assert FuelReadingContract().name == "fuel_book_reading"


def test_block_cap_forces_judgement_in_never_quiet_combat() -> None:
    """A block that never quiets is judged at the reading cap."""
    from tankpit_bot.ledger.fuel_book import BLOCK_READING_CAP

    book = make_fuel_book()
    record_fuel_reading(book=book, fuel_total=10_000)
    verdict = None
    for step in range(1, BLOCK_READING_CAP + 1):
        record_fuel_entry(book=book, kind="walk", lo=-1, hi=0)
        verdict = record_fuel_reading(book=book, fuel_total=10_000 - step)
        if step < BLOCK_READING_CAP:
            assert verdict is None
    if verdict is None:
        raise AssertionError("expected the cap to force a verdict")
    assert verdict["balanced"] is True
    assert book["windows"] == 1


def test_announced_gains_credit_the_live_book() -> None:
    """0x44 gains explain their own delta at the wire mutation point."""
    from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total

    ws = WorldService()
    update_world_state_from_fuel_total(ws, 1000, "wire_0x44_fuel_gain")
    assert ws.fuel_book["entries"] == []
    update_world_state_from_fuel_total(ws, 1200, "wire_0x44_fuel_gain")
    assert ws.fuel_book["entries"] == [{"kind": "pickup", "lo": 200, "hi": 200}]
    update_world_state_from_fuel_total(ws, 1200, "wire_0x2E_tank_status_sync")
    update_world_state_from_fuel_total(ws, 1200, "wire_0x2E_tank_status_sync")
    assert ws.fuel_book["windows"] == 1
    assert ws.fuel_book["divergences"] == 0
