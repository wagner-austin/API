"""The worker-wait knob: timing the workforce, not counting it.

The corpus verdict this knob operationalizes (log 2026-09-07): phase-1
worker growth rides with a longer stand at 12/12 evolve generations while
phase-0 growth rides the other way -- and the VH-era ``workers10`` arm
only ever measured the count axis. The knob gates the CEILING the economy
buys toward; a zero-worker emergency still buys through
:func:`~rw_bot.policy.spending.worker_need`'s first branch, which ignores
the ceiling by design ([[impossible-build-priority-head]]).
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.policy.campaign import play
from rw_bot.wire.state import Sample
from tests.campaign_fixtures import (
    BUILDER,
    CATALOGUE,
    CENTRE,
    PLACEMENTS,
    PROFILES,
    ScriptedPeer,
    verb,
)
from tests.wire_fixtures import lines, option, pool, sample


def _worker_wait_world() -> Sample:
    """One busy-after-tick-one builder, and a Command Center able to make more.

    The plan takes the sole builder on the first observation; from the second
    it stands on its job and reads busy, so :func:`~rw_bot.policy.spending.\
worker_need` sees no free worker, one owned against a ceiling of four, and
    credits to spare -- the exact state that buys a second builder.

    Returns:
        The scripted world.
    """
    return sample(
        CENTRE,
        BUILDER,
        credits=4000,
        pools=(pool(x=300.0),),
        options=(
            option(214, "extractorT1", placed=True),
            option(213, "builder", placed=False),
        ),
    )


def test_a_second_builder_is_bought_once_the_first_is_busy() -> None:
    """The control half of the pair: with no wait, this state buys a builder.

    This is also the withhold test's fail-first witness -- if the gate were
    inert, the same world would buy in both tests and the pair could not
    both pass.
    """
    world = _worker_wait_world()
    peer = ScriptedPeer(lines(*(world for _ in range(3))))
    play(AgentChannel(peer), ("extractorT1",), CATALOGUE, PLACEMENTS, PROFILES, 3)
    assert any('"type":"builder"' in line for line in verb(peer, "produce"))


def test_the_worker_wait_withholds_that_purchase_until_it_expires() -> None:
    """Timing, not count: below the wait the ceiling reads as ONE.

    The same world, the same free-worker drought, the same credits -- and no
    builder is bought, because the doctrine says the workforce grows later.
    """
    world = _worker_wait_world()
    peer = ScriptedPeer(lines(*(world for _ in range(3))))
    play(
        AgentChannel(peer),
        ("extractorT1",),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        3,
        worker_wait=100,
    )
    assert not any('"type":"builder"' in line for line in verb(peer, "produce"))
