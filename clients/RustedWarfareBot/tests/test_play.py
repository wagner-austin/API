"""The play entry point, driven against a scripted agent.

The catalogue is the real archived ``-printunits`` dump, so the prices these
assert against are the engine's own.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.play import (
    DEFAULT_GOALS,
    EXIT_BAD_USAGE,
    EXIT_INCOMPLETE,
    EXIT_OK,
    expansion_reserve,
    heavy_reinforcements,
    load_catalogue,
    load_placements,
    main,
    reinforcements,
)

from rw_bot.policy.doctrine import Doctrine, DoctrineError
from rw_bot.policy.doctrine_file import format_doctrine
from tests.wire_fixtures import ScriptedPeer, StubbedConnect

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"
_PLACEMENT_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"


def _entity_line(frame: int, index: int, unit_id: int, type_name: str) -> str:
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":100.0,"y":200.0,'
        f'"team":0,"mine":true,"hostile":false,"movement":"LAND","group":1,"flying":false,"submerged":false,"touching_water":false,"hp":100.0,"max_hp":100.0,"complete":true,"queued":0}}'
    )


def _pool_line(frame: int, index: int, tile_x: int, tile_y: int) -> str:
    return (
        f'{{"kind":"pool","frame":{frame},"index":{index},'
        f'"tile_x":{tile_x},"tile_y":{tile_y},'
        f'"x":{tile_x * 20 + 10}.0,"y":{tile_y * 20 + 10}.0,"group_land":1}}'
    )


#: What the Builder offers by default here -- the plan's own types, which the
#: live capture confirms unit 214 reports.
_BUILDER_OFFERS = ("extractorT1", "landFactory", "c_tank")


def _option_line(frame: int, index: int, unit_id: int, produces: str) -> str:
    return (
        f'{{"kind":"option","frame":{frame},"index":{index},"unit_id":{unit_id},'
        f'"produces":"{produces}","key":"u_x","placed":true,"available":true,"makes_something":true,"price":100}}'
    )


def _sample_lines(
    frame: int,
    credits: int,
    *entities: tuple[int, str],
    pools: tuple[tuple[int, int], ...] = (),
    options: tuple[tuple[int, str], ...] | None = None,
) -> list[str]:
    if options is None:
        options = tuple((214, name) for name in _BUILDER_OFFERS)
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{frame * 3},'
        f'"visible":{len(entities)},"pools":{len(pools)},'
        f'"options":{len(options)},"players":0,'
        f'"credits":{credits},"defeated":false,"wiped":false,"players_left":6}}'
    ]
    for index, (unit_id, type_name) in enumerate(entities):
        lines.append(_entity_line(frame, index, unit_id, type_name))
    for index, (tile_x, tile_y) in enumerate(pools):
        lines.append(_pool_line(frame, index, tile_x, tile_y))
    for index, (unit_id, produces) in enumerate(options):
        lines.append(_option_line(frame, index, unit_id, produces))
    return lines


_BUILDER = (214, "builder")

#: What DEFAULT_GOALS expands to against the real tree from the opening roster.
#: Stated here so a change to either the goals or the tree fails loudly rather
#: than quietly changing what these tests drive.
_EXPANDED = (
    "extractorT1",
    "extractorT1",
    "extractorT1",
    "landFactory",
    "c_tank",
    "c_tank",
    "c_tank",
    "c_tank",
)


def test_the_real_catalogue_prices_every_goal() -> None:
    """A goal the catalogue cannot price blocks the run at once."""
    catalogue = load_catalogue(_CATALOGUE_PATH)
    assert [catalogue[name]["price"] for name in DEFAULT_GOALS] == [
        700,
        700,
        700,
        350,
        350,
        350,
        350,
    ]


def test_the_real_dump_rules_every_goal() -> None:
    """A goal the placement dump does not cover blocks the run at once."""
    placements = load_placements(_PLACEMENT_PATH)
    assert [placements[name]["needs_pool"] for name in DEFAULT_GOALS] == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_the_goals_open_with_the_structures_that_pay_for_the_rest() -> None:
    """An extractor generates credits; everything else spends them."""
    assert DEFAULT_GOALS[0] == "extractorT1"


def test_the_goals_name_no_factory_but_the_plan_has_one() -> None:
    """The distinction expansion exists to make.

    A tank is asked for; the Land Factory that makes one is not, and writing it
    out by hand is exactly what the build tree exists to stop. If this ever
    passes trivially -- because someone put the factory back in the goals --
    the expansion is no longer being exercised by a real run.
    """
    assert "landFactory" not in DEFAULT_GOALS
    assert "c_tank" in DEFAULT_GOALS


def test_a_completed_plan_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    built = [(300 + i, name) for i, name in enumerate(_EXPANDED)]
    # One read for the opening observation the plan is expanded against, then
    # one per loop sample. The loop no longer stops when the plan finishes: a
    # completed opening is where playing starts ([[policy-loop]]).
    peer = ScriptedPeer(_sample_lines(1, 9000, _BUILDER, *built) * 3)
    with StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "2"]) == EXIT_OK
    # The world already holds a finished Land Factory, so expansion inserts
    # nothing -- the goals are reachable as written. That is the same rule the
    # insertion cases exercise, seen from the other side.
    assert capsys.readouterr().out.splitlines() == [
        "doctrine: default",
        "goals: extractorT1 -> extractorT1 -> extractorT1 -> c_tank -> c_tank -> c_tank -> c_tank",
        "plan:  extractorT1 -> extractorT1 -> extractorT1 -> c_tank -> c_tank -> c_tank -> c_tank",
        "  extractorT1 costs 700, goes on a resource pool",
        "  extractorT1 costs 700, goes on a resource pool",
        "  extractorT1 costs 700, goes on a resource pool",
        "  c_tank costs 350, goes on the ring",
        "  c_tank costs 350, goes on the ring",
        "  c_tank costs 350, goes on the ring",
        "  c_tank costs 350, goes on the ring",
        "plan total: 3500 credits, holding 9000",
        "verdict        survived (sample_limit)",
        "plan           7/7 -- done: all 7 plan entries satisfied",
        "build orders   0",
        "reinforced     0",
        # No pool rides on this scripted world, so expansion has nothing to
        # claim and says so rather than reporting a bare zero. The reason is
        # the economy's rather than defence's even though defence is attempted
        # last: "no pool was taken" has five distinct causes and this
        # enumerates them, where defence would report only that it ran last
        # ([[policy-economy]]).
        "expansions     0 (0 factories) (no pool free of 0: 0 occupied, 0 unreachable, 0 exposed)",
        "extractors     3 -> 3",
        "attack orders  0",
        "rallied        0",
        "intercepted    0",
        "sightings      0",
        "raids          0",
        "marches        0",
        "army           4 -> 4",
        # The scripted world carries no player records, so the engine's own
        # scoreboard reads as absent rather than as a measurement of nothing.
        "army value     0 -> 0",
        "total worth    0 -> 0",
        "best rival     0 -> 0 (peak 0, worst dip 0)",
        "workers        1",
        # What is standing, which no other line reports: a turret is neither
        # army nor income, so without this a run that bought defences and one
        # that bought none read identically ([[policy-economy]]).
        "structures     extractorT1 x3, landFactory x1",
        "composition    c_tank x4",
        "enemy fields   none",
        "income         0/s",
        "enemies seen   0 -> 0 (0 engageable)",
        "engaged gone   0",
        "players        6 -> 6 (0 eliminated)",
        "claims refused 0",
        "samples seen   2",
        "frames elapsed 0",
        "engine clock   0 ms",
        "credits left   9000",
        # **Which spender was even asked**, which no other figure reports. A
        # stage that declined three thousand times and one that was never
        # reached both leave a refusal count of zero, and defence was measured
        # and refuted on exactly that ambiguity -- it had fired three times in
        # twelve full matches ([[policy-holding-ground]]).
        "reach          income                  reached     2  acted     0"
        "  last: no pool free of 0: 0 occupied, 0 unreachable, 0 exposed",
        "reach          defence                 reached     2  acted     0"
        "  last: no free worker can place c_turret_t1",
        "reach          throughput              reached     2  acted     0"
        "  last: production is not the constraint",
        # The plan is already satisfied in this world, so nothing is ever
        # claimed -- named rather than left blank, because an empty block reads
        # as a measurement that failed to happen.
        "spend          nothing was ever claimed",
    ]


def test_an_unfinished_plan_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    peer = ScriptedPeer(_sample_lines(1, 10, _BUILDER) * 2)
    with StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "1"]) == EXIT_INCOMPLETE
    # One more header line than before: the plan now announces its total price
    # against the opening balance.
    assert capsys.readouterr().out.splitlines()[12:] == [
        "verdict        survived (sample_limit)",
        # The extractor has no pool in this scripted world, so the plan reaches
        # past it to the factory and reports what blocks *that* -- an entry
        # with nowhere to stand defers rather than stopping the plan, which is
        # what kept two duels from ever building an army
        # ([[policy-holding-ground]]).
        "plan           0/8 -- building: landFactory costs 700, holding 10",
        "build orders   0",
        "reinforced     0",
        # There is one builder and the opening plan is still using it, so the
        # economy stands down rather than re-tasking it to its own pool. Both
        # ordering it in one tick means neither order is carried out, which a
        # live run showed as four expansions and a plan stuck at 3/8.
        "expansions     0 (0 factories) (the opening plan is using the only free worker)",
        "extractors     0 -> 0",
        "attack orders  0",
        "rallied        0",
        "intercepted    0",
        "sightings      0",
        "raids          0",
        "marches        0",
        "army           0 -> 0",
        "army value     0 -> 0",
        "total worth    0 -> 0",
        "best rival     0 -> 0 (peak 0, worst dip 0)",
        "workers        1",
        "structures     none",
        # No army, so no mix -- named rather than left blank, because an empty
        # field reads as a measurement that failed to happen.
        "composition    none",
        "enemy fields   none",
        "income         0/s",
        "enemies seen   0 -> 0 (0 engageable)",
        "engaged gone   0",
        "players        6 -> 6 (0 eliminated)",
        "claims refused 0",
        "samples seen   1",
        "frames elapsed 0",
        "engine clock   0 ms",
        "credits left   10",
        # The gate that switches the whole economy off, seen from outside for
        # the first time: the plan holds the one worker, so income, defence and
        # throughput are not reached at all rather than declining
        # ([[policy-economy]]).
        "reach          plan-holds-only-worker  reached     1  acted     0"
        "  last: the opening plan is using the only free worker",
        "spend          nothing was ever claimed",
    ]


def test_the_sample_budget_defaults_when_not_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    built = [(300 + i, name) for i, name in enumerate(_EXPANDED)]
    peer = ScriptedPeer(_sample_lines(1, 9000, _BUILDER, *built) * 200)
    with StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH)]) == EXIT_OK
    # One more header line than before: the plan now announces its total price
    # against the opening balance.
    assert capsys.readouterr().out.splitlines()[11:] == [
        "verdict        survived (sample_limit)",
        "plan           7/7 -- done: all 7 plan entries satisfied",
        "build orders   0",
        "reinforced     0",
        # No pool rides on this scripted world, so expansion has nothing to
        # claim and says so rather than reporting a bare zero. The reason is
        # the economy's rather than defence's even though defence is attempted
        # last: "no pool was taken" has five distinct causes and this
        # enumerates them, where defence would report only that it ran last
        # ([[policy-economy]]).
        "expansions     0 (0 factories) (no pool free of 0: 0 occupied, 0 unreachable, 0 exposed)",
        "extractors     3 -> 3",
        "attack orders  0",
        "rallied        0",
        "intercepted    0",
        "sightings      0",
        "raids          0",
        "marches        0",
        "army           4 -> 4",
        # The scripted world carries no player records, so the engine's own
        # scoreboard reads as absent rather than as a measurement of nothing.
        "army value     0 -> 0",
        "total worth    0 -> 0",
        "best rival     0 -> 0 (peak 0, worst dip 0)",
        "workers        1",
        "structures     extractorT1 x3, landFactory x1",
        "composition    c_tank x4",
        "enemy fields   none",
        "income         0/s",
        "enemies seen   0 -> 0 (0 engageable)",
        "engaged gone   0",
        "players        6 -> 6 (0 eliminated)",
        "claims refused 0",
        "samples seen   120",
        "frames elapsed 0",
        "engine clock   0 ms",
        "credits left   9000",
        # Reached on every one of the 120 observations and acting on none of
        # them, which is the shape "declined constantly" makes. The opposite
        # shape -- never reached at all -- is the one in the test above, and a
        # refusal count renders both as zero.
        "reach          income                  reached   120  acted     0"
        "  last: no pool free of 0: 0 occupied, 0 unreachable, 0 exposed",
        "reach          defence                 reached   120  acted     0"
        "  last: no free worker can place c_turret_t1",
        "reach          throughput              reached   120  acted     0"
        "  last: production is not the constraint",
        "spend          nothing was ever claimed",
    ]


@pytest.mark.parametrize(
    "args", [[], ["27200"], ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]]
)
def test_a_bad_argument_count_prints_usage(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: play <port> <catalogue-path> <placement-path> "
        "[max-samples] [doctrine-path] [trace-path]\n"
    )


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.play")
    sys.argv = ["play"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.play", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.play"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: play <port> <catalogue-path> <placement-path> "
        "[max-samples] [doctrine-path] [trace-path]\n"
    )


def test_the_style_can_be_given_as_a_doctrine_file(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """So one arm of an experiment differs from another by a file.

    Editing the source between runs is how an A/B stops being an A/B: the two
    arms end up differing by whatever else was in the working tree. The file
    also outlives the run, so the arm that played last week can be re-run
    rather than reconstructed.
    """
    preset = tmp_path / "tanks.doctrine"
    preset.write_text(
        "\n".join(
            format_doctrine(
                Doctrine(
                    name="tanks",
                    goals=("c_tank", "c_tank"),
                    heavies=(),
                    max_workers=4,
                    mass=7,
                    reserve=-1,
                    expand=True,
                    counter=False,
                    cover=True,
                    intercept=False,
                    guard_cap=0,
                    aa_cover=False,
                    forward=False,
                    scout=False,
                    raid=0,
                    rush=False,
                    creep=0,
                    riposte=False,
                    tech=0,
                    lurk=0,
                    allin=0,
                    decoys=0,
                    kite=False,
                    income_ladder=False,
                    hp_floor=0,
                    strike=0,
                    medics=0,
                    bunkers=0,
                    flame=0,
                    close=0,
                    guns=0,
                    nukes=0,
                )
            )
        )
        + "\n",
        encoding="utf-8",
    )
    peer = ScriptedPeer(_sample_lines(1, 9000, _BUILDER, (300, "landFactory")) * 3)
    with StubbedConnect(peer):
        code = main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "1", str(preset)])
    # The exit code is about whether the plan finished, which is not what this
    # is testing; what matters is that the style came from the file.
    assert code in (EXIT_OK, EXIT_INCOMPLETE)
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "doctrine: tanks"
    assert printed[1] == "goals: c_tank -> c_tank"
    assert printed[2] == "plan:  c_tank -> c_tank"


def test_a_dash_means_the_default_doctrine_and_no_trace(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Both optional slots have to be fillable without being used."""
    peer = ScriptedPeer(_sample_lines(1, 9000, _BUILDER, (300, "landFactory")) * 3)
    with StubbedConnect(peer):
        main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "1", "-", "-"])
    assert capsys.readouterr().out.splitlines()[0] == "doctrine: default"
    assert list(tmp_path.iterdir()) == []


def test_the_reserve_covers_the_dearest_thing_the_bot_keeps_making() -> None:
    """Enough to replace a single loss, and the depth measured better than a
    shallower one.

    The objection to the maximum is real: one expensive type raises the barrier
    the whole economy must clear, invisibly, since nothing about a unit list
    looks like a reserve. It was changed to the composition mean on exactly that
    reasoning, moving the standard mix 450 -> 375, and twelve seeds at Very Hard
    called it a **regression: 7 wins became 3**, routs 3 -> 1, outside the noise
    floor ([[policy-holding-ground]]). A shallower reserve starves the
    replacement it exists to fund.

    The confound the objection identified is answered by the override below
    rather than by lowering the figure.
    """
    catalogue = load_catalogue(_CATALOGUE_PATH)
    arty = catalogue["c_artillery"]["price"]
    assert arty > catalogue["c_tank"]["price"]
    assert expansion_reserve(("c_tank", "c_tank", "c_tank", "c_artillery"), catalogue) == arty
    assert expansion_reserve(("c_tank",), catalogue) == catalogue["c_tank"]["price"]


def test_nothing_to_reinforce_reserves_nothing() -> None:
    """No army to protect means every spare credit belongs to the economy."""
    assert expansion_reserve((), load_catalogue(_CATALOGUE_PATH)) == 0


def test_the_derived_reserve_ignores_the_heavies() -> None:
    """A heavy is unbuildable until its tier opens, so reserving its
    replacement price is dead capital -- and worse: a 3,100-credit
    heavyArtillery in the mix raised the floor to 3,100 and the unprotected
    tech claim then needed 5,100 in the bank, starving the very unlock that
    would have made the heavy buildable. Measured: the roster probe never
    bought its unlock at all ([[policy-budget]]).
    """
    catalogue = load_catalogue(_CATALOGUE_PATH)
    goals = ("c_tank", "c_artillery")
    assert catalogue["heavyArtillery"]["price"] > catalogue["c_artillery"]["price"]
    with_heavies = (*reinforcements(goals, catalogue), "heavyArtillery")
    assert expansion_reserve(with_heavies, catalogue) == catalogue["heavyArtillery"]["price"]
    # The entry point derives from the goals alone, so the doctrine's heavies
    # never move the floor.
    assert (
        expansion_reserve(reinforcements(goals, catalogue), catalogue)
        == catalogue["c_artillery"]["price"]
    )


def test_heavies_are_verified_against_the_real_catalogue() -> None:
    """The extra-composition channel skips plan expansion, so its checks
    live here: priced, and producible by a queue rather than placed.

    ``heavyTank`` is the entry the channel was built for -- behind the land
    factory's tier-two unlock, so it must reach the composition without
    ever becoming a plan goal ([[mechanics-build-actions]]).
    """
    catalogue = load_catalogue(_CATALOGUE_PATH)
    assert heavy_reinforcements(("heavyTank", "heavyTank"), catalogue) == (
        "heavyTank",
        "heavyTank",
    )
    with pytest.raises(DoctrineError) as unknown:
        heavy_reinforcements(("heavyTankTypo",), catalogue)
    assert unknown.value.code == "RW-DOCTRINE-011"
    with pytest.raises(DoctrineError) as structure:
        heavy_reinforcements(("landFactory",), catalogue)
    assert structure.value.code == "RW-DOCTRINE-011"
    assert "structure" in structure.value.message


def test_the_opening_settles_by_content_not_by_clock(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A held match world may be sampled before its units spawn.

    The world used to settle on 22 seconds of free-running wall clock, and
    runs began from worlds that already differed ([[policy-determinism]]).
    The roster is what plan expansion reads, so the roster is the condition:
    empty samples are acknowledged and the plan is expanded against the first
    observation that owns something finished.
    """
    built = [(300 + i, name) for i, name in enumerate(_EXPANDED)]
    empty = _sample_lines(1, 4000)
    populated = _sample_lines(2, 9000, _BUILDER, *built)
    # Two empty observations, one populated one the settle consumes, then the
    # loop's own two samples.
    peer = ScriptedPeer(empty + empty + populated * 3)
    with StubbedConnect(peer):
        assert main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "2"]) == EXIT_OK
    printed = capsys.readouterr().out.splitlines()
    # The plan was expanded against the populated world, not the empty one.
    assert printed[2] == (
        "plan:  extractorT1 -> extractorT1 -> extractorT1 -> c_tank -> c_tank -> c_tank -> c_tank"
    )


def test_a_world_that_never_populates_is_a_failed_start_not_a_slow_one() -> None:
    """The settle is bounded: a match whose units never spawn is broken, and
    expanding a plan against an empty world says so loudly rather than
    waiting forever.
    """
    from rw_bot.policy.expand import ExpansionError

    empty = _sample_lines(1, 4000)
    peer = ScriptedPeer(empty * 60)
    with StubbedConnect(peer), pytest.raises(ExpansionError) as caught:
        main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "1"])
    assert caught.value.code == "RW-EXPAND-001"
