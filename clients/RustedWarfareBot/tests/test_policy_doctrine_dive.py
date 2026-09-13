"""The dive's doctrine fields, exercised as the codec refusals they carry.

Split from ``test_policy_doctrine`` at the size ceiling: the four fields the
dive added in one day (``dive``, ``divemargin``, ``divecap``, ``diveblood``)
share one shape -- a count, zero for off, negative refused with its own
code -- and read together here ([[policy-raid]]).
"""

from __future__ import annotations

import pytest

from rw_bot.policy.doctrine import DoctrineError
from rw_bot.policy.doctrine_codecs import decode_doctrine, encode_doctrine
from rw_bot.policy.doctrine_default import DEFAULT_DOCTRINE


def test_a_negative_dive_is_refused_and_a_party_size_round_trips() -> None:
    """Zero already means no diving; below it is a typo."""
    payload = encode_doctrine(DEFAULT_DOCTRINE)
    payload["dive"] = -1
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-043"
    payload["dive"] = 3
    assert decode_doctrine(payload)["dive"] == 3


def test_a_negative_dive_margin_is_refused_and_a_standoff_width_round_trips() -> None:
    """Zero already means any outranging gun; below it is a typo."""
    payload = encode_doctrine(DEFAULT_DOCTRINE)
    payload["divemargin"] = -1
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-044"
    payload["divemargin"] = 100
    assert decode_doctrine(payload)["divemargin"] == 100


def test_a_negative_dive_cap_is_refused_and_a_party_budget_round_trips() -> None:
    """Zero already means uncapped under the escalating rung; below it is a typo."""
    payload = encode_doctrine(DEFAULT_DOCTRINE)
    payload["divecap"] = -1
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-045"
    payload["divecap"] = 16
    assert decode_doctrine(payload)["divecap"] == 16


def test_a_negative_dive_blood_is_refused_and_a_death_count_round_trips() -> None:
    """Zero already means the first draft waits for nothing; below it is a typo."""
    payload = encode_doctrine(DEFAULT_DOCTRINE)
    payload["diveblood"] = -1
    with pytest.raises(DoctrineError) as caught:
        decode_doctrine(payload)
    assert caught.value.code == "RW-DOCTRINE-046"
    payload["diveblood"] = 3
    assert decode_doctrine(payload)["diveblood"] == 3


def test_the_default_doctrine_carries_every_dive_field_off() -> None:
    """The champion's presets carry the fields at zero, and so does the constant."""
    assert (
        DEFAULT_DOCTRINE["dive"],
        DEFAULT_DOCTRINE["divemargin"],
        DEFAULT_DOCTRINE["divecap"],
        DEFAULT_DOCTRINE["diveblood"],
    ) == (0, 0, 0, 0)
