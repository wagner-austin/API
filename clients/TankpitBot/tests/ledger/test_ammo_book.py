"""Tests for the live ammo book (consumption-equals-hit, enforced)."""

from __future__ import annotations

import pytest

from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.ledger.ammo_book import (
    AmmoSnapshotContract,
    make_ammo_book,
    record_ammo_death,
    record_ammo_gain,
    record_ammo_scan,
    record_ammo_shot,
    record_ammo_snapshot,
)


def test_first_snapshot_anchors_without_verdict() -> None:
    """The first 0x49 only opens the account."""
    book = make_ammo_book()
    assert record_ammo_snapshot(book=book, counts=[3, 25, 25, 25, 20]) is None
    assert book["snapshots"] == 0


def test_consumption_within_shots_balances() -> None:
    """Weapons falling by at most the shots fired is the contract."""
    book = make_ammo_book()
    record_ammo_snapshot(book=book, counts=[3, 25, 25, 25, 20])
    record_ammo_shot(book=book, weapon=1)
    record_ammo_shot(book=book, weapon=1)
    record_ammo_shot(book=book, weapon=3)
    record_ammo_shot(book=book, weapon=0)
    record_ammo_scan(book=book)
    verdict = record_ammo_snapshot(book=book, counts=[3, 23, 25, 25, 19])
    if verdict is None:
        raise AssertionError("expected a snapshot verdict")
    assert verdict["balanced"] is True
    assert verdict["detail"] == "(balanced)"
    assert (book["snapshots"], book["divergences"]) == (1, 0)


def test_unexplained_fall_is_a_divergence() -> None:
    """Ammo vanishing without shots is exactly what the book must catch."""
    book = make_ammo_book()
    record_ammo_snapshot(book=book, counts=[3, 25, 25, 25, 20])
    verdict = record_ammo_snapshot(book=book, counts=[3, 22, 25, 25, 20])
    if verdict is None:
        raise AssertionError("expected a snapshot verdict")
    assert verdict["balanced"] is False
    assert "dual fell 3 with only 0 uses recorded" in verdict["detail"]
    assert book["divergences"] == 1


def test_a_tank_kill_death_halves_the_baseline_rounding_up() -> None:
    """The wire-verified law: every slot goes to ceil(n/2) on death.

    Without the transform, the desert 2026-08-26 run burned one false
    ammo divergence per death (three deaths, three divergences): the
    post-death 0x49 read as an infeasible fall. The odd counts pin the
    round-half-UP direction (45->23, 9->5, 37->19).
    """
    book = make_ammo_book()
    record_ammo_snapshot(book=book, counts=[45, 9, 44, 37, 24])

    record_ammo_death(book=book, mine_kill=False)

    assert book["last_counts"] == [23, 5, 22, 19, 12]
    verdict = record_ammo_snapshot(book=book, counts=[23, 5, 22, 19, 12])
    if verdict is None:
        raise AssertionError("expected a snapshot verdict")
    assert verdict["balanced"] is True


def test_shots_before_the_death_stay_feasible() -> None:
    """Halving the pre-shot baseline never under-allows the real fall.

    Fire one dual from 9, die: the server halves 8 to 4; the book
    halves 9 to 5 and still holds one recorded shot, so the snapshot
    at 4 balances.
    """
    book = make_ammo_book()
    record_ammo_snapshot(book=book, counts=[25, 9, 25, 25, 20])
    record_ammo_shot(book=book, weapon=1)

    record_ammo_death(book=book, mine_kill=False)

    verdict = record_ammo_snapshot(book=book, counts=[13, 4, 13, 13, 10])
    if verdict is None:
        raise AssertionError("expected a snapshot verdict")
    assert verdict["balanced"] is True


def test_a_mine_death_zeroes_the_baseline() -> None:
    """The one observed mine death wiped every slot outright."""
    book = make_ammo_book()
    record_ammo_snapshot(book=book, counts=[35, 35, 35, 35, 29])

    record_ammo_death(book=book, mine_kill=True)

    assert book["last_counts"] == [0, 0, 0, 0, 0]
    verdict = record_ammo_snapshot(book=book, counts=[0, 0, 0, 0, 0])
    if verdict is None:
        raise AssertionError("expected a snapshot verdict")
    assert verdict["balanced"] is True


def test_a_death_before_any_snapshot_is_a_no_op() -> None:
    """With no anchoring 0x49 there is no baseline to penalize."""
    book = make_ammo_book()

    record_ammo_death(book=book, mine_kill=False)

    assert book["last_counts"] is None


def test_rise_requires_an_equipment_gain() -> None:
    """Counts rising without a 0x67 gain is a divergence; with one it balances."""
    book = make_ammo_book()
    record_ammo_snapshot(book=book, counts=[3, 20, 20, 20, 20])
    bad = record_ammo_snapshot(book=book, counts=[3, 25, 20, 20, 20])
    if bad is None:
        raise AssertionError("expected a snapshot verdict")
    assert bad["balanced"] is False
    assert "dual rose 5 with no equipment gain" in bad["detail"]
    record_ammo_gain(book=book)
    good = record_ammo_snapshot(book=book, counts=[4, 30, 25, 25, 25])
    if good is None:
        raise AssertionError("expected a snapshot verdict")
    assert good["balanced"] is True


def test_armor_falls_bounded_by_enemy_shots() -> None:
    """Armor absorbs at most 2 shields per incoming hit (a dual)."""
    from tankpit_bot.ledger.ammo_book import record_ammo_enemy_shot

    book = make_ammo_book()
    record_ammo_snapshot(book=book, counts=[5, 20, 20, 20, 20])
    record_ammo_enemy_shot(book=book)
    record_ammo_enemy_shot(book=book)
    fell = record_ammo_snapshot(book=book, counts=[1, 20, 20, 20, 20])
    if fell is None:
        raise AssertionError("expected a snapshot verdict")
    assert fell["balanced"] is True
    unexplained = record_ammo_snapshot(book=book, counts=[0, 20, 20, 20, 20])
    if unexplained is None:
        raise AssertionError("expected a snapshot verdict")
    assert unexplained["balanced"] is False
    assert "armor fell 1 with only 0 enemy shots observed" in unexplained["detail"]
    rose = record_ammo_snapshot(book=book, counts=[5, 20, 20, 20, 20])
    if rose is None:
        raise AssertionError("expected a snapshot verdict")
    assert rose["balanced"] is False
    assert "armor rose 5 with no equipment gain" in rose["detail"]


def test_malformed_snapshots_are_rejected() -> None:
    """A snapshot must be five non-negative counts."""
    book = make_ammo_book()
    with pytest.raises(LedgerInvariantError):
        record_ammo_snapshot(book=book, counts=[1, 2, 3])
    with pytest.raises(LedgerInvariantError):
        record_ammo_snapshot(book=book, counts=[1, 2, 3, 4, -1])


def test_contract_names_identify_the_invariants() -> None:
    """Contract names appear in enforcement errors; pin them."""
    from tankpit_bot.ledger.ammo_book import (
        AmmoActivityContract,
        AmmoDeathContract,
        AmmoShotContract,
    )

    assert AmmoSnapshotContract().name == "ammo_book_snapshot"
    assert AmmoShotContract().name == "ammo_book_shot"
    assert AmmoActivityContract().name == "ammo_book_activity"
    assert AmmoDeathContract().name == "ammo_book_death"


def test_negative_weapon_byte_is_rejected() -> None:
    """A negative weapon byte cannot come off the wire."""
    book = make_ammo_book()
    with pytest.raises(LedgerInvariantError):
        record_ammo_shot(book=book, weapon=-1)
