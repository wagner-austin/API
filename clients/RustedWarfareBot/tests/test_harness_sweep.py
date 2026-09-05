"""The job model: what a sweep file means, exercised as the pure functions it is.

No filesystem and no game. What a match *is* and which worker plays it are
decisions, and they are decided here.
"""

from __future__ import annotations

import pytest

from rw_bot.harness import play_match_cli
from rw_bot.harness.match import MatchConfig
from rw_bot.harness.results_layout import (
    SWEEP_ROOT,
    TRACE_ROOT,
    match_log_path,
    trace_path,
)
from rw_bot.harness.sweep import (
    JOB_FIELDS,
    LAUNCHER_MODULE,
    SweepError,
    SweepJob,
    assigned,
    decode_sweep_job,
    encode_sweep_job,
    fresh_seeds,
    is_complete,
    job_name,
    make_argv,
    parse_job_line,
    parse_jobs,
    play_args,
    scorecard,
)
from rw_bot.policy.ledger import Outlay, Reach
from rw_bot.policy.match_report import MatchReport, format_report
from rw_bot.validation import DecodeError

#: The interpreter a sweep launches its matches with. Stated, because the
#: harness passes its own rather than naming one.
PY = "/venv/bin/python"

#: Where this batch files, and the two paths a launch is told rather than
#: composing for itself. Built through the layout module the runner uses, so a
#: test cannot pass against a spelling nothing produces.
BATCH = "demo"
OUT_DIR = f"{SWEEP_ROOT}/{BATCH}"
LOG = f"{OUT_DIR}/logs/tank-s777.log"
TRACE = f"{TRACE_ROOT}/{BATCH}/tank-s777.ndjson"

#: A clone's leased channel port.
PORT = 27512

#: The X display that clone leases.
DISPLAY = 91


def _flag(argv: tuple[str, ...], name: str) -> str | None:
    """Return the value a command line gives one flag.

    Args:
        argv: The command.
        name: The flag to read.

    Returns:
        Its value, or None when the flag is absent. Absence is a real answer
        here: several options are omitted rather than passed as zero, because
        a frozen tree predating one rejects the unknown key.
    """
    if name not in argv:
        return None
    return argv[argv.index(name) + 1]


def _job(
    label: str = "tank", seed: int = 1, doctrine: str = "doctrines/default.doctrine"
) -> SweepJob:
    return SweepJob(
        label=label,
        seed=seed,
        doctrine=doctrine,
        samples=1500,
    )


def test_a_job_line_becomes_a_job() -> None:
    job = parse_job_line("tank | 12345 | doctrines/default.doctrine | 1500")
    assert job == SweepJob(
        label="tank",
        seed=12345,
        doctrine="doctrines/default.doctrine",
        samples=1500,
    )


def test_a_line_missing_a_field_names_the_line_and_the_schema() -> None:
    """A sweep file is written by hand, so a default here would quietly change
    what an arm means rather than reporting that the arm is under-specified.
    """
    with pytest.raises(SweepError) as caught:
        parse_job_line("tank|12345|1500")
    assert caught.value.code == "RW-SWEEP-001"
    assert "expected 4" in str(caught.value)
    assert "|".join(JOB_FIELDS) in str(caught.value)


def test_a_non_numeric_field_names_the_field() -> None:
    with pytest.raises(SweepError) as caught:
        parse_job_line("tank|soon|doctrines/default.doctrine|1500")
    assert caught.value.code == "RW-SWEEP-002"
    assert "'seed'" in str(caught.value)


def test_a_field_out_of_range_is_refused_by_the_decoder() -> None:
    """Parsing narrows the text; the decoder is what judges the value."""
    with pytest.raises(DecodeError) as caught:
        parse_job_line("tank|1|doctrines/default.doctrine|0")
    assert caught.value.code == "RW-DECODE-004"


def test_a_blank_label_is_refused() -> None:
    with pytest.raises(DecodeError) as caught:
        parse_job_line("  |1|doctrines/default.doctrine|1500")
    assert caught.value.code == "RW-DECODE-003"


def test_a_negative_seed_is_allowed_because_the_engine_takes_one() -> None:
    """Seeds are not counts, so the positive-integer rule does not apply."""
    assert parse_job_line("tank|-3|doctrines/default.doctrine|1500")["seed"] == -3


def test_blank_lines_and_comments_are_skipped() -> None:
    """An arm is commented out of a sweep rather than deleted from it."""
    jobs = parse_jobs(
        [
            "# the control arm",
            "tank|1|doctrines/default.doctrine|1500",
            "",
            "   ",
            "  # tank|2|doctrines/default.doctrine|1500",
            "arty|1|doctrines/arty.doctrine|1500",
        ]
    )
    assert [job["label"] for job in jobs] == ["tank", "arty"]


def test_a_job_round_trips_through_its_payload() -> None:
    assert decode_sweep_job(encode_sweep_job(_job())) == _job()


def test_a_result_is_filed_under_its_arm_and_seed() -> None:
    assert job_name(_job(label="tank-arty", seed=4242)) == "tank-arty-s4242"


def test_every_job_is_played_exactly_once_across_the_workers() -> None:
    """The partition is what makes a lock unnecessary; overlap would play a
    match twice and a gap would silently drop one from the experiment.
    """
    jobs = [_job(seed=n) for n in range(10)]
    seen = [job for index in range(4) for job in assigned(jobs, index, 4)]
    assert sorted(job["seed"] for job in seen) == list(range(10))


def test_a_worker_with_nothing_to_do_gets_nothing() -> None:
    assert assigned([_job()], 2, 4) == ()


def test_a_sweep_needs_at_least_one_worker() -> None:
    with pytest.raises(SweepError) as caught:
        assigned([_job()], 0, 0)
    assert caught.value.code == "RW-SWEEP-003"


@pytest.mark.parametrize("index", [-1, 4])
def test_a_worker_index_outside_the_pool_is_refused(index: int) -> None:
    with pytest.raises(SweepError) as caught:
        assigned([_job()], index, 4)
    assert caught.value.code == "RW-SWEEP-004"


def test_the_planner_argument_list_carries_every_arm_variable() -> None:
    """Samples, the doctrine that is the whole of the style, and the trace."""
    assert (
        play_args(
            _job(doctrine="doctrines/mass25.doctrine"),
            trace_path(TRACE_ROOT, BATCH, _job(doctrine="doctrines/mass25.doctrine")),
        )
        == "1500 doctrines/mass25.doctrine runs/traces/demo/tank-s1.ndjson"
    )


def test_a_chosen_match_is_carried_to_every_job_in_the_batch() -> None:
    """Which map is played decides how many opponents there are.

    The engine caps teams by the map's own count, so a two-player map is a duel
    whatever else is asked for -- which is why the match belongs to the batch
    rather than to a job line ([[policy-determinism]]).
    """
    duel = MatchConfig(map_path="maps/skirmish/[p2]duel_lake.tmx", opponents=1, difficulty=-2)
    argv = make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY, duel)
    assert _flag(argv, "--map") == "maps/skirmish/[p2]duel_lake.tmx"
    assert _flag(argv, "--opponents") == "1"
    assert _flag(argv, "--difficulty") == "-2"


def test_no_chosen_match_leaves_the_engines_own_default() -> None:
    """Absent means the hardcoded ten-player free-for-all, which is what every
    measurement before the duel was taken in ([[policy-determinism]]).
    """
    argv = make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY)
    assert _flag(argv, "--map") is None
    assert _flag(argv, "--opponents") is None
    assert _flag(argv, "--difficulty") is None


def test_every_match_records_a_trace_named_after_its_job() -> None:
    """Sweeps used to pass ``-`` and keep only the endpoint scorecard.

    Endpoints proved actively misleading: a match reporting ``extractors
    0 -> 0`` had held a peak of 14 and led on total worth at the midpoint
    before collapsing. None of that survives in the scorecard, and re-running
    to recover it produces a different match ([[policy-trace]]).
    """
    assert trace_path(TRACE_ROOT, BATCH, _job(seed=777)) == TRACE
    assert trace_path(TRACE_ROOT, "other", _job(label="air")) == "runs/traces/other/air-s1.ndjson"


def test_a_launch_is_told_where_to_write_rather_than_deriving_it() -> None:
    """Both paths used to be built inside ``make_argv`` against
    repository-relative roots, while the runner created the directories from
    an absolute ``out_dir``. The two agreed only while the process started in
    the repository, so on a compute node the directory that existed and the
    directory written to were different ones -- and the trace, which is the
    entire measurement of a replication panel, went where nothing looked.
    """
    cluster_out = "/pub/wagnera3/rusted/runs/sweeps/demo"
    argv = make_argv(
        PY,
        _job(seed=777),
        ".game-w2",
        75,
        match_log_path(cluster_out, _job(seed=777)),
        trace_path("/pub/wagnera3/rusted/runs/traces", BATCH, _job(seed=777)),
        PORT,
        DISPLAY,
    )
    assert _flag(argv, "--play-log") == f"{cluster_out}/logs/tank-s777.log"
    played = _flag(argv, "--play-args")
    if played is None:
        raise AssertionError("a launch must always carry its planner arguments")
    assert played.endswith("/pub/wagnera3/rusted/runs/traces/demo/tank-s777.ndjson")


def test_the_log_path_hangs_off_the_batchs_own_results_directory() -> None:
    assert match_log_path(OUT_DIR, _job(seed=777)) == LOG


def test_the_command_pins_the_seed_the_clone_and_the_lockstep() -> None:
    """Lockstep is passed per job rather than left to the recipe: free running,
    parallel matches under CPU contention sample at different game-times.
    """
    assert make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY) == (
        PY,
        "-m",
        "rw_bot.harness.play_match_cli",
        "--port",
        str(PORT),
        "--display",
        str(DISPLAY),
        "--game-dir",
        ".game-w2",
        "--seed",
        "777",
        "--lockstep",
        "75",
        "--play-log",
        "runs/sweeps/demo/logs/tank-s777.log",
        "--play-args",
        "1500 doctrines/default.doctrine runs/traces/demo/tank-s777.ndjson",
    )


def test_the_command_is_this_packages_launcher_not_make() -> None:
    """It used to be a ``make play`` line, which put the whole launch behind a
    PowerShell recipe and a PowerShell script -- neither of which can start a
    match on a Linux compute node."""
    argv = make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY)
    assert argv[0] == PY
    assert "make" not in argv


def test_the_launcher_name_is_read_off_the_module_it_names() -> None:
    """Written out, the string and the module drift apart silently and the
    sweep launches something that no longer exists."""
    assert play_match_cli.__name__ == LAUNCHER_MODULE


def test_a_frozen_tree_is_carried_to_every_job_in_the_batch() -> None:
    """A match imports the source tree at launch, so without this an edit
    landed mid-batch meant later matches ran different code from earlier ones
    -- the working tree was frozen for the batch's whole runtime
    ([[policy-loop]])."""
    argv = make_argv(
        PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY, tree="runs/sweeps/demo/.tree"
    )
    assert _flag(argv, "--tree") == "runs/sweeps/demo/.tree"


def test_a_pinned_batch_says_so_and_an_unpinned_one_stays_silent() -> None:
    """Silence is load-bearing: a tree frozen before the option existed runs
    an agent that rejects the unknown key, so an unpinned batch must not
    mention the variable at all ([[policy-determinism]]).
    """
    pinned = make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY, pin_delta=3)
    assert _flag(pinned, "--pin-delta") == "3"
    unpinned = make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY)
    assert _flag(unpinned, "--pin-delta") is None


def test_a_fast_batch_says_so_and_a_realtime_one_stays_silent() -> None:
    """The gym knob rides the same silence rule as the pin: certified
    bit-exact at 10x (log 2026-08-06), but a tree frozen before the option
    existed runs an agent that rejects the unknown key."""
    fast = make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY, fast_forward=10)
    assert _flag(fast, "--fast-forward") == "10"
    realtime = make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY)
    assert _flag(realtime, "--fast-forward") is None


def test_a_frozen_tree_owns_the_doctrine_path_too() -> None:
    """The first snapshot batch proved this within the hour: matches imported
    frozen code but read the working tree's doctrine file, a field was added
    mid-batch, and the frozen parser refused it on sixteen straight matches.
    A doctrine file is as much the experiment as the code is."""
    job = _job(doctrine="doctrines/mass25.doctrine")
    frozen = play_args(job, trace_path(TRACE_ROOT, BATCH, job), f"{SWEEP_ROOT}/{BATCH}/.tree")
    assert frozen == (
        "1500 runs/sweeps/demo/.tree/doctrines/mass25.doctrine runs/traces/demo/tank-s1.ndjson"
    )
    argv = make_argv(
        PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, PORT, DISPLAY, tree="runs/sweeps/demo/.tree"
    )
    assert _flag(argv, "--play-args") == (
        "1500 runs/sweeps/demo/.tree/doctrines/default.doctrine runs/traces/demo/tank-s777.ndjson"
    )


def _report() -> MatchReport:
    """A report with every field populated, so the filter is asked about all of them."""
    return MatchReport(
        grade="survived",
        completed=8,
        planned=8,
        build_orders=8,
        build_outcome="done",
        build_reason="all 8 plan entries satisfied",
        produced=63,
        expanded=128,
        expanded_factories=0,
        expand_reason="every worker is already building something",
        extractors_start=0,
        extractors_end=13,
        attack_orders=208,
        rallied=27,
        intercepts=12,
        sightings=41,
        raids=3,
        hunts=2,
        marches=9,
        army_start=0,
        army_end=20,
        targets_seen=16,
        targets_end=99,
        engageable_end=86,
        killed=16,
        army_value_start=500,
        army_value_end=23500,
        worth_start=3500,
        worth_end=37000,
        rival_worth_start=4700,
        rival_worth_end=27850,
        rival_worth_peak=27850,
        rival_worth_drawdown=700,
        workers_end=33,
        enemy_types_end=(("c_tank", 40), ("extractorT2", 3)),
        composition_end=(("c_tank", 20),),
        units_lost_to=(("c_artillery", 3),),
        buildings_lost_to=(),
        standing_end=(("extractorT1", 13), ("c_turret_t1", 4), ("landFactory", 1)),
        owned_peak=(("c_tank", 24), ("extractorT1", 14), ("seaFactory", 1)),
        income_end=122,
        players_start=5,
        players_end=5,
        eliminated=0,
        refused_claims=144,
        samples_seen=1500,
        frames_elapsed=112144,
        clock_elapsed_ms=1869000,
        credits_at_end=3880,
        outcome="sample_limit",
        outlays=(
            Outlay(
                purpose="produce:c_tank",
                asked=412,
                granted=350,
                spent=122_500,
                refusal="produce:c_tank wanted 350 of 120 available; 700 already committed",
            ),
        ),
        reaches=(
            Reach(
                stage="defence",
                reached=3,
                acted=3,
                reason="",
            ),
        ),
    )


def test_every_figure_the_report_emits_survives_the_filter() -> None:
    """The guard against the drift that already happened once.

    The filter used to be a hardcoded tuple of labels, duplicating knowledge
    that lives in :func:`format_report`. Two figures were added to the report
    and neither was added to the tuple, so every sweep result silently dropped
    them and a whole batch looked like it had measured nothing. Deriving the
    expectation from the report itself is what makes that impossible: a figure
    added tomorrow is covered without anyone remembering to come here.
    """
    rendered = format_report(_report())
    assert scorecard(rendered) == rendered


def test_the_scorecard_is_taken_from_the_transcript_and_the_chatter_is_not() -> None:
    card = scorecard(
        [
            "==> play (headless match on port 27801)",
            "goals: extractorT1 -> c_tank",
            "  c_tank costs 350, goes on the ring",
            "verdict        survived (sample_limit)",
            "army           0 -> 20",
            "composition    c_tank x20",
            "[play] game stopped",
        ]
    )
    assert card == (
        "verdict        survived (sample_limit)",
        "army           0 -> 20",
        "composition    c_tank x20",
    )


def test_a_transcript_without_a_verdict_is_not_a_result() -> None:
    """A match that crashed on boot prints its plan and nothing else, and
    filing that would record a blank as though it were a measurement.
    """
    assert not is_complete(scorecard(["goals: c_tank", "[play] game stopped"]))
    assert is_complete(scorecard(["verdict        survived (sample_limit)"]))


def test_a_sweep_job_carries_the_port_its_clone_leases() -> None:
    """The lease's exclusivity is what replaces a draw. It used to be optional,
    with an absent one meaning "let the recipe draw" -- and two concurrent
    draws collided the first time eight matches launched in one instant, with
    both dying on the bind (imp-creep12, 2026-08-08)."""
    leased = make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, 27512, DISPLAY)
    assert leased[leased.index("--port") + 1] == "27512"


def test_a_job_with_no_lease_is_refused_rather_than_drawn_for() -> None:
    """There is no draw left to fall back to: a launcher that invented a port
    here would collide with a live match's lease and take BOTH matches down,
    which is strictly worse than not starting."""
    with pytest.raises(SweepError) as caught:
        make_argv(PY, _job(seed=777), ".game-w2", 75, LOG, TRACE, 0, DISPLAY)
    assert caught.value.code == "RW-SWEEP-005"


def test_fresh_seeds_are_odd_disjoint_and_spread() -> None:
    """The panel-independence picker: nothing already used, everything
    odd, and the picks stride the whole range instead of clustering at
    its bottom."""
    used = {101, 103, 105}
    picked = fresh_seeds(used, 4, 100, 132)
    assert picked == (107, 113, 119, 125)
    assert set(picked) & used == set()
    assert all(seed % 2 == 1 for seed in picked)
    # An even start rounds UP to odd rather than fielding an even seed.
    assert fresh_seeds(frozenset(), 2, 10, 20) == (11, 15)


def test_fresh_seeds_refuse_a_range_too_small_to_stay_disjoint() -> None:
    with pytest.raises(SweepError) as caught:
        fresh_seeds({11, 13}, 3, 10, 16)
    assert caught.value.code == "RW-SWEEP-006"
    assert "1 unused odd seed(s), 3 were asked for" in caught.value.message
