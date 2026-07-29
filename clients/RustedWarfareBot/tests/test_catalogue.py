"""Decoding the engine's own unit catalogue.

The headline cases run against the real ``-printunits`` log archived under
``wiki/sources/m0-probe/``, so the decoder is tested against what the engine
actually printed. The malformed cases are hand-built, because the engine does
not emit broken output on request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.mechanics.catalogue import CatalogueError, Weapon, decode_catalogue
from rw_bot.wire.codec import decode_samples

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"


def _catalogue_lines() -> list[str]:
    return _CATALOGUE.read_text(encoding="utf-8", errors="replace").splitlines()


def _weapon(
    *,
    shoot_delay: float,
    attack_range: float,
    direct: float = 0.0,
    direct_volley: float = 0.0,
    area: float = 0.0,
    area_volley: float = 0.0,
) -> Weapon:
    """Build a whole expected weapon, so a test compares one value not four.

    Comparing the entire record is what makes an unexpected non-zero field a
    failure. Asserting field by field would pass while some other damage figure
    silently drifted.

    Args:
        shoot_delay: Frames between shots.
        attack_range: Range in world units.
        direct: Single-target damage per shot.
        direct_volley: Single-target damage per volley.
        area: Splash damage per shot.
        area_volley: Splash damage per volley.

    Returns:
        The expected weapon record.
    """
    return Weapon(
        shoot_delay=shoot_delay,
        attack_range=attack_range,
        direct_damage=direct,
        direct_damage_volley=direct_volley,
        area_damage=area,
        area_damage_volley=area_volley,
    )


def _block(*stats: str, type_name: str = "probe", name: str = "Probe") -> list[str]:
    """Build one unit block with the given stat lines."""
    return [
        '<div class="unit">',
        f'<img src="unit:{type_name}" />',
        f"<h4>{name}</h4>",
        "<p>-A<br/>-B</p>",
        f"<pre>{stats[0]}",
        *stats[1:],
        "</pre></div>",
    ]


# Named parts rather than slices of one tuple: mypy reads a tuple slice as
# carrying Any under disallow_any_expr, and the pieces are each meaningful
# enough to deserve a name.
_PRICE = "Price: $10"
_MASS = "Mass: 30"
_MIDDLE = ("Hp: 20", "Speed: 1.5", "Turn speed: 2")
_REST = (*_MIDDLE, _MASS)
_MINIMAL = (_PRICE, *_REST)


def test_decodes_every_unit_in_the_real_catalogue() -> None:
    units = decode_catalogue(_catalogue_lines())
    assert len(units) == 90


def test_type_names_are_unique_because_they_are_the_join_key() -> None:
    units = decode_catalogue(_catalogue_lines())
    names = [u["type_name"] for u in units]
    assert len(set(names)) == len(names)


def test_the_builder_matches_the_engines_printed_stats() -> None:
    """Cross-checked by eye against the archived log."""
    units = {u["type_name"]: u for u in decode_catalogue(_catalogue_lines())}
    builder = units["builder"]
    assert builder["display_name"] == "Builder"
    assert builder["price"] == 500
    assert builder["hp"] == 170
    assert builder["speed"] == 0.6
    assert builder["mass"] == 3000
    assert builder["weapon"] is None
    assert "Can not attack." in builder["description"]


def test_a_turret_carries_its_upgrade_tier_prices() -> None:
    units = {u["type_name"]: u for u in decode_catalogue(_catalogue_lines())}
    turret = units["c_turret_t1"]
    assert turret["price"] == 500
    assert turret["upgrade_prices"] == (1000,)
    assert turret["speed"] == 0.0


def test_armed_and_unarmed_units_partition_the_catalogue() -> None:
    units = decode_catalogue(_catalogue_lines())
    armed = [u for u in units if u["weapon"] is not None]
    assert len(armed) == 61
    assert len(units) - len(armed) == 29


def test_every_live_roster_type_is_checked_against_the_catalogue() -> None:
    """The join key is only useful where it actually joins, so this asserts the
    real intersection rather than a hand-picked pair.

    ``editorOrBuilder`` appears in the world stream and has no catalogue entry:
    it is the map editor's placeholder, not a buildable unit. Recording it as a
    known exception is the point -- an earlier version of this test named itself
    "all priced" while checking two types, and that is what let the claim that
    the catalogue prices everything the stream reports survive.
    """
    catalogue = {u["type_name"]: u for u in decode_catalogue(_catalogue_lines())}
    samples = decode_samples(
        (_PROJECT_ROOT / "wiki" / "sources" / "m6-wire" / "world-sample.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    live = {e["type_name"] for s in samples for e in s["entities"]}
    assert live, "the archived capture carries a roster"

    unpriced = sorted(t for t in live if t not in catalogue)
    assert unpriced == ["editorOrBuilder"]
    for type_name in live - set(unpriced):
        assert catalogue[type_name]["price"] > 0


def test_the_catalogue_predicts_which_live_units_can_move() -> None:
    """Verified live: see wiki/sources/m7-mobility/mobility-predicate.txt."""
    catalogue = {u["type_name"]: u for u in decode_catalogue(_catalogue_lines())}
    assert catalogue["commandCenter"]["speed"] == 0.0
    assert catalogue["builder"]["speed"] > 0.0


def test_multi_barrel_damage_keeps_per_shot_and_volley_apart() -> None:
    units = {u["type_name"]: u for u in decode_catalogue(_catalogue_lines())}
    multi = [
        u
        for u in units.values()
        if u["weapon"] is not None
        and u["weapon"]["direct_damage_volley"] > u["weapon"]["direct_damage"]
    ]
    assert multi, "the real catalogue contains multi-barrel units"
    ratios = {
        round(u["weapon"]["direct_damage_volley"] / u["weapon"]["direct_damage"], 2)
        for u in multi
        if u["weapon"] is not None
    }
    # More than one ratio is why volley damage cannot be derived from per-shot.
    assert len(ratios) > 1


def test_a_single_barrel_weapon_reports_volley_equal_to_per_shot() -> None:
    block = _block(*_MINIMAL, "Shoot Delay: 40", "Attack Range: 120", "Direct Damage: 10")
    assert decode_catalogue(block)[0]["weapon"] == _weapon(
        shoot_delay=40.0, attack_range=120.0, direct=10.0, direct_volley=10.0
    )


def test_a_volley_total_is_read_as_its_own_figure() -> None:
    block = _block(
        *_MINIMAL, "Shoot Delay: 40", "Attack Range: 120", "Direct Damage: 12 (total:24.0)"
    )
    assert decode_catalogue(block)[0]["weapon"] == _weapon(
        shoot_delay=40.0, attack_range=120.0, direct=12.0, direct_volley=24.0
    )


def test_area_damage_is_read_the_same_way() -> None:
    block = _block(*_MINIMAL, "Shoot Delay: 5", "Attack Range: 90", "Area Damage: 45 (total:270.0)")
    assert decode_catalogue(block)[0]["weapon"] == _weapon(
        shoot_delay=5.0, attack_range=90.0, area=45.0, area_volley=270.0
    )


def test_an_armed_unit_with_no_damage_line_reports_zero_damage() -> None:
    """One unit in the real catalogue is exactly this shape."""
    block = _block(*_MINIMAL, "Shoot Delay: 5", "Attack Range: 90")
    assert decode_catalogue(block)[0]["weapon"] == _weapon(shoot_delay=5.0, attack_range=90.0)


def test_multiple_upgrade_tiers_are_ordered() -> None:
    block = _block(*_MINIMAL, "T2 Upgrade Price: $100", "T3 Upgrade Price: $250")
    assert decode_catalogue(block)[0]["upgrade_prices"] == (100, 250)


def test_a_unit_without_a_description_still_decodes() -> None:
    block = [
        '<div class="unit">',
        '<img src="unit:probe" />',
        "<h4>Probe</h4>",
        f"<pre>{_PRICE}",
        *_REST,
        "</pre></div>",
    ]
    assert decode_catalogue(block)[0]["description"] == ""


def test_lines_outside_unit_blocks_are_ignored() -> None:
    noise = ["2026-07-25 01:32:54.595: File logging started", "INFO:Slick Build #84"]
    assert decode_catalogue([*noise, *_block(*_MINIMAL), *noise]) != ()


def test_an_empty_log_yields_no_units() -> None:
    assert decode_catalogue([]) == ()


def test_a_block_without_a_type_name_is_rejected() -> None:
    block = ['<div class="unit">', "<h4>Probe</h4>", f"<pre>{_PRICE}", "</pre></div>"]
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(block)
    assert caught.value.code == "RW-CATALOGUE-001"


def test_a_malformed_image_tag_is_rejected() -> None:
    block = ['<div class="unit">', '<img src="unit:', f"<pre>{_PRICE}", "</pre></div>"]
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(block)
    assert caught.value.code == "RW-CATALOGUE-001"


def test_a_block_without_a_display_name_is_rejected() -> None:
    block = [
        '<div class="unit">',
        '<img src="unit:probe" />',
        f"<pre>{_PRICE}",
        *_REST,
        "</pre></div>",
    ]
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(block)
    assert caught.value.code == "RW-CATALOGUE-002"


def test_a_stat_line_without_a_separator_is_rejected() -> None:
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(_block(*_MINIMAL, "GarbageLine"))
    assert caught.value.code == "RW-CATALOGUE-003"


def test_a_non_integer_price_is_rejected() -> None:
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(_block("Price: $lots", "Hp: 1", "Speed: 1", "Turn speed: 1", "Mass: 1"))
    assert caught.value.code == "RW-CATALOGUE-003"


def test_a_non_numeric_speed_is_rejected() -> None:
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(_block("Price: $1", "Hp: 1", "Speed: fast", "Turn speed: 1", "Mass: 1"))
    assert caught.value.code == "RW-CATALOGUE-003"


def test_an_unclosed_damage_total_is_rejected() -> None:
    block = _block(*_MINIMAL, "Shoot Delay: 1", "Attack Range: 1", "Direct Damage: 12 (total:24.0")
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(block)
    assert caught.value.code == "RW-CATALOGUE-003"


def test_a_non_numeric_damage_total_is_rejected() -> None:
    block = _block(*_MINIMAL, "Shoot Delay: 1", "Attack Range: 1", "Direct Damage: 12 (total:x)")
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(block)
    assert caught.value.code == "RW-CATALOGUE-003"


def test_a_missing_required_stat_is_rejected() -> None:
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(_block("Price: $1", "Hp: 1", "Speed: 1", "Turn speed: 1"))
    assert caught.value.code == "RW-CATALOGUE-004"
    assert "Mass" in caught.value.message


def test_an_armed_unit_without_a_shoot_delay_is_rejected() -> None:
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(_block(*_MINIMAL, "Attack Range: 120"))
    assert caught.value.code == "RW-CATALOGUE-004"


def test_an_unclosed_block_is_rejected_rather_than_dropped() -> None:
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(['<div class="unit">', '<img src="unit:probe" />'])
    assert caught.value.code == "RW-CATALOGUE-005"
    assert "truncated" in caught.value.message


def test_a_blank_line_inside_the_stats_block_is_skipped() -> None:
    block = _block(_PRICE, "", *_REST)
    assert decode_catalogue(block)[0]["price"] == 10


def test_a_close_tag_sharing_a_stat_line_is_rejected_not_mis_parsed() -> None:
    """The engine puts the close on its own line; anything else must fail loudly."""
    block = [
        '<div class="unit">',
        '<img src="unit:probe" />',
        "<h4>Probe</h4>",
        f"<pre>{_PRICE}",
        *_MIDDLE,
        f"{_MASS}</pre></div>",
    ]
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue(block)
    assert caught.value.code == "RW-CATALOGUE-003"


def test_a_repeated_type_name_is_rejected() -> None:
    with pytest.raises(CatalogueError) as caught:
        decode_catalogue([*_block(*_MINIMAL), *_block(*_MINIMAL)])
    assert caught.value.code == "RW-CATALOGUE-006"
