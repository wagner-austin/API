"""One worker, many claimants: the arbitration that keeps two commanders
off one builder.

The engine runs whichever waypoint arrived last, so every defect here has the
same shape -- two spenders ordering the same unit and neither order arriving.
Gathering, replacement, job memory and the plan/economy split are all faces
of that one rule.
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.policy.campaign import play
from rw_bot.policy.workforce import EXPAND_RETRY_SAMPLES
from tests.campaign_fixtures import (
    BUILDER,
    CATALOGUE,
    CENTRE,
    ENEMY,
    PLACEMENTS,
    PROFILES,
    WAVE,
    ScriptedPeer,
    defence_world,
    order_lines,
    run_campaign,
    verb,
)
from tests.wire_fixtures import (
    enemy,
    entity,
    lines,
    option,
    pool,
    sample,
)


def test_the_plan_and_the_economy_do_not_both_drive_the_one_builder() -> None:
    """There is one builder, and the engine runs whichever waypoint arrived last.

    A live 400-sample run had the economy re-tasking the builder to its own pool
    between the plan's own extractors: four expansions ordered, and a plan still
    stuck at 3 of 8 ([[policy-loop]]). Whoever holds the builder holds it alone.
    """
    world = sample(
        CENTRE,
        BUILDER,
        credits=4000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    report, peer = run_campaign(world, times=1, plan=("extractorT1",))
    assert len(verb(peer, "build")) == 1
    assert report["expanded"] == 0
    # The economy stands down because the plan's worker was the *only* free one,
    # not because it is barred whenever the plan holds any worker at all. That
    # distinction is the fix: with six workers the old rule skipped the expander
    # on 572 of 800 samples ([[policy-economy]]).
    assert report["expand_reason"] == "the opening plan is using the only free worker"


def test_a_second_worker_keeps_expanding_while_the_plan_holds_the_first() -> None:
    """The fix, and the figure that forced it.

    The plan takes **one** worker and the expander used to answer that by
    standing down entirely -- income, defence and throughput together, however
    many others were free. Instrumented over 800 samples with six workers alive,
    the expander was skipped on **572 of them**: those spenders were not
    declining, they were never asked ([[policy-economy]]).

    Two builders here, and both should be working: the plan places its extractor
    with one and the economy claims the second pool with the other. Neither may
    order the same unit, which is the defect the old gate existed to prevent
    ([[policy-loop]]).
    """
    world = sample(
        CENTRE,
        BUILDER,
        entity(215, "builder", x=0.0, y=0.0),
        credits=4000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(
            option(214, "extractorT1", placed=True),
            option(215, "extractorT1", placed=True),
        ),
    )
    report, peer = run_campaign(world, times=1, plan=("extractorT1",))
    built = verb(peer, "build")
    assert len(built) == 2
    assert report["expanded"] == 1
    # The two orders go to different workers. One unit ordered twice in a tick is
    # the original bug: the engine runs whichever waypoint arrived last, so
    # neither order is carried out.
    assert len({line.split('"unit_id":')[1].split(",")[0] for line in built}) == 2


def test_a_cheaper_defence_does_not_jump_the_queue_while_income_is_merely_short() -> None:
    """The inversion that stalled the economy at Hard.

    Income needs the extractor's 700; a turret needs 500. So on every
    observation where the economy was refused for credits, defence was offered
    the same balance, could afford it, and took it. Measured over a Hard batch:
    **29 turrets bought against 4 extractors, 43 of 47 extractor claims refused
    for credits**, income stuck at 34/s while the opponent compounded
    ([[policy-holding-ground]]).

    A refusal for any *other* reason is a different matter -- every pool taken,
    every route exposed, no worker able to place one -- and the surplus really is
    spare then. That is what defence is for, and the case below it still passes.
    """
    catalogue, placements, profiles = defence_world()
    world = sample(
        CENTRE,
        entity(214, "builder", x=0.0, y=0.0),
        # An uncovered extractor, so defence has somewhere it wants to spend.
        entity(400, "extractorT1", x=900.0, y=0.0),
        # Enough for the 500 turret, not enough for the 700 extractor.
        credits=600,
        pools=(pool(x=300.0),),
        options=(option(214, "c_turret_t1"), option(214, "extractorT1", placed=True)),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, profiles, 1, expand=True)
    assert [line for line in order_lines(peer) if '"kind":"build"' in line] == []


def test_two_workers_do_not_both_claim_the_same_pool() -> None:
    """The waste that unblocking the workforce exposed.

    A pool is judged occupied by what *stands* on it, so one a builder is
    walking toward still reads as free. One worker at a time hid that; several
    at once did not. An instrumented run granted **23 extractor orders, lost
    nothing at all, and ended with four extractors** -- the credits were never
    burnt, since a granted claim is intent, but every duplicate cost a worker
    its travel time ([[policy-holding-ground]]).
    """
    near = pool(x=300.0)
    far = pool(x=900.0, index=1)
    world = sample(
        CENTRE,
        BUILDER,
        entity(215, "builder", x=0.0, y=0.0),
        credits=10_000,
        pools=(near, far),
        options=(
            option(214, "extractorT1", placed=True),
            option(215, "extractorT1", placed=True),
        ),
    )
    _, peer = run_campaign(world, times=1, plan=("extractorT1",))
    sites = {line.split('"x":')[1].split(",")[0] for line in verb(peer, "build")}
    assert len(sites) == 2


def test_the_economy_takes_the_builder_once_the_plan_is_finished() -> None:
    """The complement: standing down is for the opening, not for the match."""
    world = sample(
        CENTRE,
        BUILDER,
        entity(400, "extractorT1"),
        credits=4000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    report, peer = run_campaign(world, times=1, plan=("extractorT1",))
    assert verb(peer, "build") == [
        '{"kind":"build","unit_id":214,"x":300.0,"y":0.0,"type":"extractorT1"}'
    ]
    assert report["expanded"] == 1


def test_the_economy_leaves_the_worker_it_just_sent_to_build() -> None:
    """The worker is busy because it has an outstanding job, across ticks.

    Both rules used to refuse only while their *own* structure was going up,
    and a refusal from one fell straight through to the other, which ordered
    something else and re-tasked the worker off it. Availability is judged once
    now, per worker, by the thing that knows what each was sent to do
    ([[policy-loop]]).
    """
    first = sample(
        CENTRE,
        BUILDER,
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    # Same worker, same place, and now its extractor is going up where it was
    # sent. It must not be handed a second pool.
    building = sample(
        CENTRE,
        BUILDER,
        entity(400, "extractorT1", x=300.0, complete=False),
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = ScriptedPeer(lines(first, building, building))
    report = play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 3)
    assert len(verb(peer, "build")) == 1
    assert report["expand_reason"] == "every worker is already building something"


def test_a_second_worker_builds_while_the_first_is_busy() -> None:
    """What the whole refactor buys: two workers, two jobs at once.

    One builder was an assumption baked into every layer -- the plan found "the"
    builder and so did the economy, both meaning the first in the roster, so a
    second would have stood idle for the entire match ([[policy-production]]).
    """
    second = entity(215, "builder", x=100.0)
    first = sample(
        CENTRE,
        BUILDER,
        second,
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True), option(215, "extractorT1", placed=True)),
    )
    building = sample(
        CENTRE,
        BUILDER,
        second,
        entity(400, "extractorT1", x=300.0, complete=False),
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True), option(215, "extractorT1", placed=True)),
    )
    peer = ScriptedPeer(lines(first, building))
    play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 2)
    ordered = verb(peer, "build")
    assert len(ordered) == 2
    assert '"unit_id":214' in ordered[0]
    assert '"unit_id":215' in ordered[1]


def test_a_walking_builder_is_left_alone_by_the_economy() -> None:
    """The order it is carrying out is the order that would be sent again."""
    world = sample(
        CENTRE,
        entity(214, "builder", x=50.0),
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    moving = sample(
        CENTRE,
        entity(214, "builder", x=90.0),
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = ScriptedPeer(lines(world, moving))
    report = play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 2)
    assert report["expand_reason"] == "every worker is already building something"


def test_the_plan_gets_the_next_free_worker_ahead_of_the_economy() -> None:
    """Worker priority to match the plan's credit priority.

    The regression this pins: the moment defence siting stopped being
    silently refused, a rich Hard match kept every worker employed on
    turrets, the plan's factory never met a free one, and "no free worker"
    was ruled "not playable from here" -- army 0 -> 0, wins 1/10 where the
    same doctrine had won 10/12 (log: 2026-07-31). Now the busy state is a
    wait, the expander stands down while the plan waits, and the next freed
    worker is the plan's.
    """
    pool_a = pool(x=300.0)
    pool_b = pool(x=500.0)
    opts = (
        option(214, "extractorT1", placed=True),
        option(214, "landFactory", placed=True, index=1),
    )
    working = sample(CENTRE, BUILDER, credits=4000, pools=(pool_a, pool_b), options=opts)
    finished = sample(
        CENTRE,
        BUILDER,
        entity(400, "extractorT1", x=300.0, y=10.0),
        credits=4000,
        pools=(pool_b,),
        options=opts,
    )
    # The worker frees only after the retry window judges its finished job
    # done, so the finished world repeats past EXPAND_RETRY_SAMPLES.
    tail = [finished] * (EXPAND_RETRY_SAMPLES + 3)
    peer = ScriptedPeer(lines(working, working, *tail))
    report = play(
        AgentChannel(peer),
        ("extractorT1", "landFactory"),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        2 + len(tail),
    )
    builds = verb(peer, "build")
    # Sample one: the plan takes the worker for its extractor. While the
    # worker is busy, the plan waits for it and the economy -- a free pool and
    # 4,000 credits in hand -- buys NOTHING rather than the plan's worker.
    # Once the worker frees, it goes to the plan's factory, not to the
    # economy's pool.
    assert len(builds) == 2
    assert '"type":"extractorT1"' in builds[0]
    assert '"type":"landFactory"' in builds[1]
    stages = [row["stage"] for row in report["reaches"]]
    assert "plan-first-in-line" in stages


def test_a_disbanded_wave_is_sent_home_again() -> None:
    """The mark is per stint in the reserve, not per match.

    A survivor handed back by a disbanded wave was previously told nothing: not
    cleared to attack, and already marked as rallied from before its first wave.
    It stood where its wave died until enough reinforcements arrived to release
    it again ([[policy-combat]]).
    """
    far = entity(1, "c_tank", x=4000.0)
    # Three tanks release a wave on the first observation; two die, and the
    # survivor is below FIRST_WAVE so it goes back to the reserve.
    first = sample(CENTRE, *WAVE, ENEMY)
    after = sample(CENTRE, far, ENEMY)
    peer = ScriptedPeer(lines(first, after))
    play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 2)
    moves = [line for line in peer.sent if '"kind":"move"' in line]
    assert '{"kind":"move","unit_id":1,"x":0.0,"y":0.0}' in moves


def test_a_worker_sitting_on_an_unstarted_job_is_freed_to_retry() -> None:
    """The engine refuses some placements silently, and says so only in its log.

    A worker that has neither moved nor started building is not on its way
    anywhere. After the same window the plan's stall clock uses, the order is
    presumed lost and the worker is free to be given another.
    """
    world = sample(
        CENTRE,
        BUILDER,
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    # Nothing ever goes up and the worker never moves, so after the retry
    # window the order is reissued.
    peer = ScriptedPeer(lines(*(world for _ in range(EXPAND_RETRY_SAMPLES + 2))))
    play(
        AgentChannel(peer),
        (),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        EXPAND_RETRY_SAMPLES + 2,
    )
    assert len(verb(peer, "build")) == 2


def test_a_worker_that_dies_is_forgotten() -> None:
    """Its bookkeeping must not outlive it, or an id the engine reuses inherits
    a job that was never given to it.
    """
    with_worker = sample(
        CENTRE,
        BUILDER,
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    without = sample(CENTRE, credits=100_000, pools=(pool(x=300.0),))
    reborn = sample(
        CENTRE,
        BUILDER,
        credits=100_000,
        pools=(pool(x=300.0),),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = ScriptedPeer(lines(with_worker, without, reborn))
    play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 3)
    # Ordered on the first observation, and again once a worker exists afresh.
    assert len(verb(peer, "build")) == 2


def test_an_expansion_is_not_reoffered_at_sample_rate() -> None:
    """The same site twice running is the order still being carried out."""
    world = sample(
        CENTRE,
        BUILDER,
        entity(400, "extractorT1", x=300.0, complete=False),
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = ScriptedPeer(lines(world, world, world))
    play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 3)
    assert len(verb(peer, "build")) == 1


def test_an_unrelated_structure_does_not_count_as_a_workers_job() -> None:
    """Ownership, completeness and type all have to match, or an opponent's
    half-built building nearby would hold our worker busy forever.
    """
    world = sample(
        CENTRE,
        BUILDER,
        enemy(900, "extractorT1", x=300.0, complete=False),
        entity(401, "landFactory", x=300.0, complete=False),
        credits=100_000,
        pools=(pool(x=300.0), pool(x=900.0, index=1)),
        options=(option(214, "extractorT1", placed=True),),
    )
    peer = ScriptedPeer(lines(world, world))
    play(AgentChannel(peer), (), CATALOGUE, PLACEMENTS, PROFILES, 2)
    # Neither the enemy's nor the wrong-typed structure is this worker's job,
    # so the second observation still finds it free -- and the site it was sent
    # to is unchanged, which is what the repeat guard suppresses.
    assert len(verb(peer, "build")) == 1
