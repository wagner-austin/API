"""Both prediction latches behind one feed, read as two booleans.

What is pinned: the doom feed builds the exact doom.COLUMNS tuple, the
brace feed reads its scoreboard-and-coverage figures off the sample
itself, the arming EDGE is returned exactly once, and a sentries with
neither model is inert.
"""

from __future__ import annotations

from rw_bot.policy.head import HeadModel, decode_head_model
from rw_bot.policy.sentries import Sentries
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, player, sample


def _world(*, rival_army: int = 8_000, credits: int = 4_000) -> Sample:
    return sample(
        entity(1, "commandCenter"),
        entity(7, "extractorT1", x=120.0, y=44.0),
        frame=1,
        credits=credits,
        players=(
            player(0, index=0, local=True, hostile=False, army_value=900, income=40),
            player(1, index=1, hostile=True, army_value=rival_army, income=90),
        ),
    )


def _observe(sentries: Sentries, world: Sample) -> bool:
    return sentries.observe(
        world,
        army=3,
        enemies=2,
        extractors=1,
        losses=0,
        producers=1,
        idle=0,
        orders=0,
        refused=0,
        worth=1_500,
        rival_worth=9_000,
        workers=2,
        navy_seen=0,
        air_seen=0,
        navy_blood=0,
    )


def _brace_model(threshold: float) -> HeadModel:
    return decode_head_model(
        [
            f'{{"window": 2, "threshold": {threshold}, "intercept": 4.0}}',
            '{"name": "rival_army_last", "mean": 8000.0, "std": 1000.0, "coef": 0.0}',
        ]
    )


def test_without_models_the_sentries_are_inert() -> None:
    sentries = Sentries(None, None, {})
    assert _observe(sentries, _world()) is False
    assert sentries.predicted is False
    assert sentries.braced is False


def test_the_doom_latch_arms_through_the_shared_feed() -> None:
    """A window-2 model with a huge intercept arms as soon as the doom
    window closes; the brace stays quiet because no brace model rides."""
    doom = decode_head_model(
        [
            '{"window": 2, "threshold": 0.5, "intercept": 6.0}',
            '{"name": "army_mean", "mean": 3.0, "std": 1.0, "coef": 0.0}',
        ]
    )
    sentries = Sentries(doom, None, {})
    _observe(sentries, _world())
    assert sentries.predicted is False  # window of 2 still open
    _observe(sentries, _world())
    assert sentries.predicted is True
    assert sentries.braced is False


def test_the_brace_edge_is_returned_exactly_once() -> None:
    """The loop responds to the EDGE; every later observation reads the
    standing state instead."""
    sentries = Sentries(None, _brace_model(0.5), {})
    assert _observe(sentries, _world()) is False  # window of 2 still open
    assert _observe(sentries, _world()) is True  # the arming edge
    assert sentries.braced is True
    assert _observe(sentries, _world()) is False  # armed already, no edge
    assert sentries.braced is True


def test_a_high_threshold_brace_never_arms_on_a_calm_match() -> None:
    sentries = Sentries(None, _brace_model(0.999), {})
    for _ in range(4):
        assert _observe(sentries, _world()) is False
    assert sentries.braced is False


def test_the_hunt_gate_is_a_continuous_verdict_not_a_latch() -> None:
    """The gate holds while the score clears the threshold and releases the
    moment it does not -- the brace arms once, the gate reads every sample."""
    swing = decode_head_model(
        [
            '{"window": 2, "threshold": 0.5, "intercept": 0.0}',
            '{"name": "rival_army_last", "mean": 8000.0, "std": 1000.0, "coef": 5.0}',
        ]
    )
    sentries = Sentries(None, None, {}, gate=swing)
    assert sentries.hunted_down is False  # window of 2 still open
    _observe(sentries, _world(rival_army=9_000))
    assert sentries.hunted_down is False  # still filling
    _observe(sentries, _world(rival_army=9_000))
    assert sentries.hunted_down is True  # rival army above the mean: doom
    _observe(sentries, _world(rival_army=6_000))
    assert sentries.hunted_down is False  # the score cleared: hunt again


def test_the_gate_keeps_reading_after_the_brace_arms() -> None:
    """One feed, two consumers: the armed brace stops being fed, the gate
    does not -- a recalled party must be releasable when the score clears."""
    swing = decode_head_model(
        [
            '{"window": 2, "threshold": 0.5, "intercept": 0.0}',
            '{"name": "rival_army_last", "mean": 8000.0, "std": 1000.0, "coef": 5.0}',
        ]
    )
    sentries = Sentries(None, _brace_model(0.5), {}, gate=swing)
    _observe(sentries, _world(rival_army=9_000))
    assert _observe(sentries, _world(rival_army=9_000)) is True  # brace edge
    assert sentries.braced is True
    assert sentries.hunted_down is True
    _observe(sentries, _world(rival_army=6_000))
    assert sentries.braced is True  # the latch holds
    assert sentries.hunted_down is False  # the gate released
