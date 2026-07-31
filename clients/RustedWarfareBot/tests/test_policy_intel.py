"""The fog memory, exercised as the bookkeeping it is.

What is tested: a sighting outlives the sample that made it, re-sighting
refreshes rather than duplicates, the window forgets, and the remembered list
reads identically across runs of one seed.
"""

from __future__ import annotations

from rw_bot.policy.intel import INTEL_WINDOW_FRAMES, Intel
from tests.wire_fixtures import enemy, entity, sample


def test_a_sighting_outlives_the_fog() -> None:
    """The whole point: seen once is not seen never."""
    intel = Intel()
    intel.observe(sample(enemy(9, "heli", flying=True), frame=100))
    intel.observe(sample(frame=200))
    remembered = intel.remembered()
    assert [s["unit_id"] for s in remembered] == [9]
    assert remembered[0]["flying"] is True


def test_a_resight_refreshes_rather_than_duplicates() -> None:
    intel = Intel()
    intel.observe(sample(enemy(9, "heli", x=100.0), frame=100))
    intel.observe(sample(enemy(9, "heli", x=900.0), frame=200))
    remembered = intel.remembered()
    assert len(remembered) == 1
    assert remembered[0]["x"] == 900.0
    assert remembered[0]["frame"] == 200


def test_the_window_forgets_what_stayed_unseen() -> None:
    """Intel old enough to be wrong is worse than fog."""
    intel = Intel()
    intel.observe(sample(enemy(9, "heli"), frame=100))
    intel.observe(sample(frame=100 + INTEL_WINDOW_FRAMES + 1))
    assert intel.remembered() == ()


def test_our_own_units_are_not_sightings() -> None:
    intel = Intel()
    intel.observe(sample(entity(1, "c_tank"), enemy(9, "heli"), frame=100))
    assert [s["unit_id"] for s in intel.remembered()] == [9]


def test_remembered_reads_in_identity_order() -> None:
    """Two runs of one seed must read their memory identically."""
    intel = Intel()
    intel.observe(sample(enemy(12, "c_tank"), enemy(9, "heli"), frame=100))
    assert [s["unit_id"] for s in intel.remembered()] == [9, 12]


def test_sightings_taken_counts_first_sights_only() -> None:
    """A scout that never saw anything and one never built read identically
    everywhere else; this is the figure that separates them.

    First sights, not re-sights: counting every upsert billed the standing
    armies in view every sample, and one raid-arm match read
    ``sightings 166554`` -- about 41 a sample, none of it about scouting
    (log: 2026-07-29).
    """
    intel = Intel()
    intel.observe(sample(enemy(9, "heli"), frame=100))
    intel.observe(sample(enemy(9, "heli"), frame=175))
    intel.observe(sample(frame=250))
    assert intel.sightings_taken == 1


def test_a_unit_seen_again_after_expiry_is_news_again() -> None:
    """The window expired means the memory forgot it; a re-sight after that is
    a first sight by the memory's own account."""
    intel = Intel(window_frames=100)
    intel.observe(sample(enemy(9, "heli"), frame=100))
    intel.observe(sample(frame=300))
    intel.observe(sample(enemy(9, "heli"), frame=350))
    assert intel.sightings_taken == 2
