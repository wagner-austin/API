"""The job model: what a sweep file means, exercised as the pure functions it is.

No filesystem and no game. What a match *is* and which worker plays it are
decisions, and they are decided here.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.match import MatchConfig
from rw_bot.harness.sweep import (
    JOB_FIELDS,
    SweepError,
    SweepJob,
    assigned,
    decode_sweep_job,
    encode_sweep_job,
    is_complete,
    job_name,
    make_argv,
    parse_job_line,
    parse_jobs,
    play_args,
    scorecard,
    trace_path,
)
from rw_bot.policy.ledger import Outlay, Reach
from rw_bot.policy.match_report import MatchReport, format_report
from rw_bot.validation import DecodeError


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
        play_args(_job(doctrine="doctrines/mass25.doctrine"))
        == "1500 doctrines/mass25.doctrine runs/traces/tank-s1.ndjson"
    )


def test_a_chosen_match_is_carried_to_every_job_in_the_batch() -> None:
    """Which map is played decides how many opponents there are.

    The engine caps teams by the map's own count, so a two-player map is a duel
    whatever else is asked for -- which is why the match belongs to the batch
    rather than to a job line ([[policy-determinism]]).
    """
    duel = MatchConfig(map_path="maps/skirmish/[p2]duel_lake.tmx", opponents=1, difficulty=-2)
    argv = make_argv(_job(seed=777), ".game-w2", 75, duel)
    assert argv[-3:] == (
        "PLAY_MAP=maps/skirmish/[p2]duel_lake.tmx",
        "PLAY_OPPONENTS=1",
        "PLAY_DIFFICULTY=-2",
    )


def test_no_chosen_match_leaves_the_engines_own_default() -> None:
    """Absent means the hardcoded ten-player free-for-all, which is what every
    measurement before the duel was taken in ([[policy-determinism]]).
    """
    argv = make_argv(_job(seed=777), ".game-w2", 75)
    assert not [element for element in argv if element.startswith("PLAY_MAP")]


def test_every_match_records_a_trace_named_after_its_job() -> None:
    """Sweeps used to pass ``-`` and keep only the endpoint scorecard.

    Endpoints proved actively misleading: a match reporting ``extractors
    0 -> 0`` had held a peak of 14 and led on total worth at the midpoint
    before collapsing. None of that survives in the scorecard, and re-running
    to recover it produces a different match ([[policy-trace]]).
    """
    assert trace_path(_job(seed=777)) == "runs/traces/tank-s777.ndjson"
    assert trace_path(_job(label="air")) == "runs/traces/air-s1.ndjson"


def test_the_command_pins_the_seed_the_clone_and_the_lockstep() -> None:
    """Lockstep is passed per job rather than left to the recipe: free running,
    parallel matches under CPU contention sample at different game-times.
    """
    assert make_argv(_job(seed=777), ".game-w2", 75) == (
        "make",
        "play",
        "GAME_DIR=.game-w2",
        "PLAY_SEED=777",
        "PLAY_SAMPLES=1500",
        "PLAY_LOCKSTEP=75",
        "PLAY_LOG=runs/tank-s777.log",
        "PLAY_ARGS=1500 doctrines/default.doctrine runs/traces/tank-s777.ndjson",
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
        standing_end=(("extractorT1", 13), ("c_turret_t1", 4), ("landFactory", 1)),
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
