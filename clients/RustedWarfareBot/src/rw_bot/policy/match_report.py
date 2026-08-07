"""What a match achieved, and how it reads.

The report is the whole output of a run and therefore the whole input to an
experiment. Every figure in it earned its place by a question that could not be
answered without it, and several were added because their absence had already
produced a wrong conclusion -- which is why the docstrings here record what the
figure is *for* rather than only what it holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from rw_bot.policy.ledger import Outlay, Reach, format_outlays, format_reaches


class MatchReport(TypedDict):
    """What a match achieved, across every layer that acted in it.

    One report rather than a scorecard and a battle report, because there is one
    loop and splitting the figures would re-imply a seam that no longer exists.

    Attributes:
        grade: The engine's own verdict: won, defeated, wiped, or survived.
        completed: Plan entries standing at the end.
        planned: Plan entries asked for.
        build_orders: Orders the plan issued.
        build_outcome: How the plan stands: building, done, blocked or stalled.
        build_reason: The plan's own words for its last decision.
        produced: Reinforcements ordered.
        expanded: Structures the economy ordered — extractors and factories.
        expanded_factories: How many of those were producers rather than income.
            Split out because the two answer opposite questions: one says the
            bot is earning more, the other that it can spend what it earns, and
            a single count cannot tell a run that grew its economy from one that
            grew its throughput ([[policy-production]]).
        expand_reason: The economy's own words for its last decision, which is
            what tells a match that could not expand from one that chose not to.
        extractors_start: Finished extractors held at the first observation.
        extractors_end: Finished extractors held at the last.
        attack_orders: Attack orders issued.
        rallied: Move orders issued to gather the reserve.
        intercepts: Guard engagements issued -- reserve units turned on a
            raider inside the outpost radius of our own structures. The line
            that tells "interception never fired" from "fired constantly",
            which a verdict alone reads identically
            ([[policy-holding-ground]]).
        sightings: Hostile sightings recorded by the intel memory. A scout
            that never saw anything and a scout that was never built read
            identically in every other figure
            ([[community-play-strategies]]).
        raids: Income objectives assaulted by the raid party. The
            never-fired ambiguity again: a raid that found nothing remembered
            to hit and a raid switched off read identically without it.
        marches: Outbound orders sent to raid party members. The conveyor
            detector: v1's ``raids`` read 2-6 while dozens of lone
            replacements marched to the same objective and died, because
            re-drafts against a standing objective count nothing
            (log: 2026-07-29).
        army_start: Units available to fight at the first observation.
        army_end: Units still available at the last.
        targets_seen: Hostile entities visible at the first observation.
        targets_end: Hostile entities visible at the last.
        engageable_end: How many of those the army could actually shoot. A gap
            between this and ``targets_end`` is an army holding units that
            cannot reach what it is facing.
        killed: Targets ordered against that are no longer visible. Not a kill
            count -- a target that retreated into fog reads the same way, which
            is why the field is named for what was observed.
        army_value_start: The engine's valuation of everything mobile we held at
            the first observation.
        army_value_end: The same at the last.
        worth_start: Everything we held at the first observation, mobile and
            standing alike. The score a bot that can buy defences has to be
            judged on: a turret is booked under building value, so army value
            alone would show the most cost-effective purchase available as a
            pure loss.
        worth_end: The same at the last observation.
        rival_worth_start: The strongest hostile player's total at the first
            observation.
        rival_worth_end: The same at the last. Both ends are carried because
            one end cannot answer the question that matters: whether anything
            we did ever cost an opponent anything. A rival total that only ever
            climbs is a bot that is out-building its opponents without
            fighting them ([[policy-verdict]]).
        rival_worth_peak: The highest total the strongest rival ever reached.
        rival_worth_drawdown: The largest fall **any single opponent** took from
            its own running peak. This is the honest answer to "are we killing
            them", and neither endpoint figure can give it: an opponent that
            lost half its army and rebuilt looks identical at the last
            observation to one that was never touched. Measured per opponent
            rather than against the leader, because grinding a weaker player
            down while a stronger one builds would otherwise report the same
            zero as achieving nothing. A drawdown of zero means nothing we did
            ever cost anybody anything, whatever the attack-order count says.
        workers_end: Builders owned at the last observation. Reported because
            they inflate the figure above them: a builder is mobile, so the
            engine books it under army value while it does no fighting, and
            33 fighting units worth 28,050 is not 33 tanks.
        enemy_types_end: What the opponents were fielding at the last
            observation, commonest first. Carried because a whole tier of the
            game turns on it and nothing else can see it: unit types declare a
            ``techLevel``, and a type's build action is registered only into
            the action lists at or above that level, so at tech 1 a tier-2
            action is absent rather than refused. Whether the opponents hold
            tier-2 types is therefore the difference between "the bot is
            playing the same game badly" and "the bot is playing a smaller
            game" ([[policy-holding-ground]]).
        standing_end: How many of each building were held at the last
            observation, commonest first. The line that says whether a policy
            ever ran: defence is a whole expansion question and its purchases
            were invisible in every figure above -- a turret is neither army nor
            income, so twelve full matches could not answer whether one had ever
            been built ([[policy-holding-ground]]).
        units_lost_to: Killer type and count for every MOBILE unit of ours
            that vanished with a last damager on record, commonest killer
            first. The death ledger's answer to "what kills our units":
            inferred losses without a damager are deliberately absent,
            because a conversion completing is not a kill
            ([[policy-trace]]).
        buildings_lost_to: The same table for our structures -- "what
            destroys our buildings", the perimeter-erosion class named by
            enemy type instead of guessed at.
        composition_end: How many of each armed type were held at the last
            observation, commonest first. A composition is something the caller
            *asks* for, and asking is not getting: a type the engine never
            offers -- gated by tech, or absent from this producer's tree --
            leaves the mix silently at whatever else was makeable. Without this
            the experiment cannot tell a mix that was built from a mix that was
            requested and quietly denied ([[policy-production]]).
        income_end: Credits per second at the last observation.
        players_start: Players still in the match at the first observation.
        players_end: Players still in at the last.
        eliminated: How many players went out while we were watching.
        refused_claims: Credit claims the budget turned down. A high count with
            a healthy balance means the priority order is starving something.
        samples_seen: World samples read.
        frames_elapsed: Engine frames between the first and last observation.
        clock_elapsed_ms: The engine's own millisecond clock between the first
            and last observation. Carried beside the frame count because the
            pair answers a question neither can alone: whether the simulation
            advances per **frame** or per **wall clock**. The engine caps
            itself at 300 frames a second, and matches are measured running at
            about 297 -- so if the clock outruns the wall the cap is a real
            throughput ceiling worth removing, and if it tracks the wall then
            uncapping would buy nothing at all ([[harness-parallel-matches]]).
        credits_at_end: Credits held at the last observation.
        outcome: Why the loop stopped: the engine's verdict when the match was
            decided, or ``"sample_limit"`` when it was still being played.
        outlays: What each purpose asked for and got, dearest first. The
            ``refused_claims`` count above says how often *something* was turned
            down and never what or why, though the budget records both on every
            claim -- about four thousand sentences a match, discarded until this
            was added ([[policy-economy]]).
        reaches: How often each spender was arrived at, in chain order. This is
            the figure that separates "declined three thousand times" from
            "never asked", which a refusal count reads identically. Defence was
            measured and refuted on that ambiguity and had in fact fired three
            times in twelve full matches ([[policy-holding-ground]]).
    """

    grade: str
    completed: int
    planned: int
    build_orders: int
    build_outcome: str
    build_reason: str
    produced: int
    expanded: int
    expanded_factories: int
    expand_reason: str
    extractors_start: int
    extractors_end: int
    attack_orders: int
    rallied: int
    intercepts: int
    sightings: int
    raids: int
    marches: int
    army_start: int
    army_end: int
    targets_seen: int
    targets_end: int
    engageable_end: int
    killed: int
    army_value_start: int
    army_value_end: int
    worth_start: int
    worth_end: int
    rival_worth_start: int
    rival_worth_end: int
    rival_worth_peak: int
    rival_worth_drawdown: int
    workers_end: int
    standing_end: tuple[tuple[str, int], ...]
    composition_end: tuple[tuple[str, int], ...]
    units_lost_to: tuple[tuple[str, int], ...]
    buildings_lost_to: tuple[tuple[str, int], ...]
    enemy_types_end: tuple[tuple[str, int], ...]
    income_end: int
    players_start: int
    players_end: int
    eliminated: int
    refused_claims: int
    samples_seen: int
    frames_elapsed: int
    clock_elapsed_ms: int
    credits_at_end: int
    outcome: str
    outlays: tuple[Outlay, ...]
    reaches: tuple[Reach, ...]


def _format_composition(composition: Sequence[tuple[str, int]]) -> str:
    """Render an army mix as one line.

    Args:
        composition: Type name and count, commonest first.

    Returns:
        ``"c_tank x24, c_artillery x8"``, or ``"none"`` for an empty army --
        which is a real state worth naming rather than an empty field that
        reads as a missing measurement.
    """
    if not composition:
        return "none"
    return ", ".join(f"{name} x{count}" for name, count in composition)


def format_report(report: MatchReport) -> tuple[str, ...]:
    """Render a match report as lines.

    Args:
        report: The report.

    Returns:
        One line per figure, then one line per purpose claimed against and one
        per spender reached. Those two blocks are last because they are the
        long ones and a reader scanning for the verdict should not have to page
        past them.
    """
    return (
        f"verdict        {report['grade']} ({report['outcome']})",
        f"plan           {report['completed']}/{report['planned']}"
        f" -- {report['build_outcome']}: {report['build_reason']}",
        f"build orders   {report['build_orders']}",
        f"reinforced     {report['produced']}",
        f"expansions     {report['expanded']}"
        f" ({report['expanded_factories']} factories) ({report['expand_reason']})",
        f"extractors     {report['extractors_start']} -> {report['extractors_end']}",
        f"attack orders  {report['attack_orders']}",
        f"rallied        {report['rallied']}",
        f"intercepted    {report['intercepts']}",
        f"sightings      {report['sightings']}",
        f"raids          {report['raids']}",
        f"marches        {report['marches']}",
        f"army           {report['army_start']} -> {report['army_end']}",
        f"army value     {report['army_value_start']} -> {report['army_value_end']}",
        f"total worth    {report['worth_start']} -> {report['worth_end']}",
        f"best rival     {report['rival_worth_start']} -> {report['rival_worth_end']}"
        f" (peak {report['rival_worth_peak']}, worst dip {report['rival_worth_drawdown']})",
        f"workers        {report['workers_end']}",
        f"structures     {_format_composition(report['standing_end'])}",
        f"composition    {_format_composition(report['composition_end'])}",
        f"units lost to  {_format_composition(report['units_lost_to'])}",
        f"works lost to  {_format_composition(report['buildings_lost_to'])}",
        f"enemy fields   {_format_composition(report['enemy_types_end'])}",
        f"income         {report['income_end']}/s",
        f"enemies seen   {report['targets_seen']} -> {report['targets_end']}"
        f" ({report['engageable_end']} engageable)",
        f"engaged gone   {report['killed']}",
        f"players        {report['players_start']} -> {report['players_end']}"
        f" ({report['eliminated']} eliminated)",
        f"claims refused {report['refused_claims']}",
        f"samples seen   {report['samples_seen']}",
        f"frames elapsed {report['frames_elapsed']}",
        f"engine clock   {report['clock_elapsed_ms']} ms",
        f"credits left   {report['credits_at_end']}",
        *format_reaches(report["reaches"]),
        *format_outlays(report["outlays"]),
    )


__all__ = ["MatchReport", "format_report"]
