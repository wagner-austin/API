"""A gameplay style as data, exercised as the decoders it is made of.

No filesystem beyond the shipped presets: a doctrine is lines in and a value
out, and every refusal names what was wrong with the line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.policy.doctrine import (
    DEFAULT_DOCTRINE,
    DERIVE_RESERVE,
    Doctrine,
    DoctrineError,
    decode_doctrine,
    encode_doctrine,
    format_doctrine,
    parse_doctrine_lines,
)
from rw_bot.validation import DecodeError

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _doctrine(name: str = "rush", counter: bool = False) -> Doctrine:
    return Doctrine(
        name=name,
        goals=("extractorT1", "c_tank", "c_tank"),
        heavies=(),
        max_workers=6,
        mass=25,
        reserve=450,
        expand=True,
        counter=counter,
        cover=True,
        intercept=False,
        guard_cap=0,
        aa_cover=False,
        forward=False,
        scout=False,
        raid=0,
        rush=False,
        creep=False,
        riposte=False,
        tech=False,
    )


def test_a_doctrine_round_trips_through_its_payload() -> None:
    assert decode_doctrine(encode_doctrine(_doctrine())) == _doctrine()


def test_a_doctrine_round_trips_through_its_file_form() -> None:
    """What a test or probe writes is exactly what the entry point reads."""
    assert parse_doctrine_lines(format_doctrine(_doctrine(counter=True))) == _doctrine(counter=True)


def test_the_shipped_default_preset_matches_the_constant() -> None:
    """The file exists so an experiment can copy it and edit one line.

    It is only useful as a starting point if it *is* the default rather than a
    drifted copy of it, so the two are pinned to each other here.
    """
    lines = (
        (_PROJECT_ROOT / "doctrines" / "default.doctrine").read_text(encoding="utf-8").splitlines()
    )
    assert parse_doctrine_lines(lines) == DEFAULT_DOCTRINE


def _preset(name: str) -> dict[str, str | int | bool]:
    lines = (_PROJECT_ROOT / "doctrines" / name).read_text(encoding="utf-8").splitlines()
    return encode_doctrine(parse_doctrine_lines(lines))


def test_the_shipped_arms_differ_from_default_in_exactly_one_field() -> None:
    """Two arms that differ in one thing are an A/B; in two, an anecdote."""
    default = encode_doctrine(DEFAULT_DOCTRINE)
    for preset, field in (("counter.doctrine", "counter"), ("no-expand.doctrine", "expand")):
        arm = _preset(preset)
        differing = [key for key in default if key != "name" and arm[key] != default[key]]
        assert differing == [field]


def test_the_duel_arms_form_a_one_field_chain() -> None:
    """aa -> aa-counter -> aa-counter-guard, each one field from the last.

    The chain is what lets a result be attributed: whatever moves between two
    adjacent arms moved because of that field and nothing else.
    """
    chain = (
        ("aa.doctrine", "aa-counter.doctrine", "counter"),
        ("aa-counter.doctrine", "aa-counter-guard.doctrine", "intercept"),
        ("aa-counter-guard.doctrine", "aa-counter-guard-cap.doctrine", "guard_cap"),
        ("aa-counter-guard.doctrine", "aa-counter-guard-nocover.doctrine", "cover"),
        ("aa-counter-guard.doctrine", "aa-counter-guard-aa.doctrine", "aa_cover"),
        ("aa-counter-guard.doctrine", "aa-counter-guard-fwd.doctrine", "forward"),
        ("aa-counter-guard.doctrine", "aa-counter-guard-scout.doctrine", "scout"),
        ("aa-counter-guard.doctrine", "aa-counter-guard-raid.doctrine", "raid"),
    )
    for base_name, arm_name, field in chain:
        base = _preset(base_name)
        arm = _preset(arm_name)
        differing = [key for key in base if key != "name" and arm[key] != base[key]]
        assert differing == [field]


def test_blank_lines_and_comments_are_skipped() -> None:
    """A preset records why its values are what they are, beside the values."""
    lines = ("# why", "", *format_doctrine(_doctrine()), "   ")
    assert parse_doctrine_lines(lines) == _doctrine()


def test_the_default_derives_its_reserve() -> None:
    assert DEFAULT_DOCTRINE["reserve"] == DERIVE_RESERVE


def test_a_line_without_a_value_names_the_shape() -> None:
    with pytest.raises(DoctrineError) as caught:
        parse_doctrine_lines(("name",))
    assert caught.value.code == "RW-DOCTRINE-001"


def test_an_unknown_field_is_named() -> None:
    """A typo becomes an error naming the field, not a knob that silently
    fails to turn.
    """
    with pytest.raises(DoctrineError) as caught:
        parse_doctrine_lines(("masss 25",))
    assert caught.value.code == "RW-DOCTRINE-002"
    assert "masss" in str(caught.value)


def test_a_non_numeric_count_is_named() -> None:
    with pytest.raises(DoctrineError) as caught:
        parse_doctrine_lines(("mass soon",))
    assert caught.value.code == "RW-DOCTRINE-003"


def test_a_flag_takes_only_zero_or_one() -> None:
    with pytest.raises(DoctrineError) as caught:
        parse_doctrine_lines(("expand yes",))
    assert caught.value.code == "RW-DOCTRINE-004"


def test_a_repeated_field_is_refused() -> None:
    """Last-one-wins is a default quietly changing what the arm means."""
    with pytest.raises(DoctrineError) as caught:
        parse_doctrine_lines(("mass 7", "mass 25"))
    assert caught.value.code == "RW-DOCTRINE-005"


def test_a_blank_goal_entry_is_refused() -> None:
    payload = encode_doctrine(_doctrine())
    payload["goals"] = "c_tank,,c_tank"
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-006"


def test_heavies_round_trip_and_none_means_empty() -> None:
    """The extra-composition channel: a list, or the word ``none``.

    A word rather than a blank, because a doctrine line cannot carry an
    empty value and a missing field is an error by design.
    """
    armed = _doctrine()
    armed["heavies"] = ("heavyTank", "heavyTank")
    assert parse_doctrine_lines(format_doctrine(armed)) == armed
    assert encode_doctrine(_doctrine())["heavies"] == "none"
    assert decode_doctrine(encode_doctrine(_doctrine()))["heavies"] == ()


def test_a_blank_heavies_entry_is_refused() -> None:
    payload = encode_doctrine(_doctrine())
    payload["heavies"] = "heavyTank,,heavyTank"
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-010"


def test_a_reserve_below_the_sentinel_is_refused() -> None:
    """-1 means derive; anything lower is a typo, not a deeper derivation."""
    payload = encode_doctrine(_doctrine())
    payload["reserve"] = -2
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-007"


def test_a_negative_guard_cap_is_refused() -> None:
    """Zero already means the whole reserve; below it is a typo, not a
    deeper commitment."""
    payload = encode_doctrine(_doctrine())
    payload["guard_cap"] = -1
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-008"


def test_a_negative_raid_size_is_refused() -> None:
    """Zero already means no raiding; a size below it is a typo, not a
    deeper restraint."""
    payload = encode_doctrine(_doctrine())
    payload["raid"] = -1
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-009"


def test_a_missing_field_is_an_error_not_a_default() -> None:
    """The same rule the sweep's job lines follow, for the same reason."""
    lines = tuple(line for line in format_doctrine(_doctrine()) if not line.startswith("mass "))
    with pytest.raises(DecodeError) as caught:
        parse_doctrine_lines(lines)
    assert caught.value.code == "RW-DECODE-001"
