"""Measure what an extractor earns, by building them one at a time and watching.

The one number the economy policy was written without. An extractor costs 700
and the catalogue says only that it "generates credits", so the reserve held
back for the army, and the choice between another extractor and two more tanks,
have both been settled by argument rather than arithmetic
([[policy-economy]]).

The method is deliberately blunt, because a blunt measurement of the right thing
beats a clever estimate. The bot builds one extractor, then does nothing at all
for a stretch of samples while recording credits against the engine clock, then
builds the next. Each idle stretch is a window in which nothing was bought, so
its slope is income and nothing else. Comparing windows gives the marginal value
of an extractor directly, with the base rate from the command centre subtracted
by construction rather than assumed.

Everything is measured inside one match. Five separate runs would each draw a
different opponent mix from the engine's weighted random, and the differences
between them would swamp the effect being measured
([[ai-opponent-strategy]]).

Run against a game started with ``-javaagent:...=channelPort=27200``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rw_bot.control.channel import AgentChannel, open_channel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.income import (
    Reading,
    format_rates,
    marginal,
    measure,
    payback_seconds,
)
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.campaign import play
from rw_bot.policy.economy import EXTRACTOR_TYPE, count_extractors
from scripts.play import load_catalogue, load_combat_profiles, load_placements

#: Extractors to build, and therefore windows to measure -- one before the
#: first is built, then one after each.
DEFAULT_STAGES = 4

#: Samples spent standing still in each window.
#:
#: Long enough that the slope is not dominated by the granularity of a credit,
#: short enough that the bot is not sitting motionless in a skirmish for
#: minutes at a time. The archived capture moved 27 credits in roughly a second
#: with no extractor at all, so a few hundred samples is a few hundred credits
#: of signal ([[wire-contract-ndjson]]).
DEFAULT_IDLE_SAMPLES = 200

#: Samples one extractor may take to go up before the stage is called failed.
BUILD_BUDGET = 400

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_BAD_USAGE = 2


def observe(channel: AgentChannel, window: int, samples: int) -> tuple[Reading, ...]:
    """Read samples while ordering nothing, and record what credits did.

    Ordering nothing is the whole method. Every sample is still acknowledged,
    because in lockstep the acknowledgement is what releases the simulation and
    an unacked sample stalls the game rather than merely skipping a reading
    ([[policy-determinism]]).

    Args:
        channel: An open connection to the agent.
        window: Which idle stretch these readings belong to.
        samples: How many observations to take.

    Returns:
        One reading per sample, in the order taken.

    Raises:
        ChannelError: When the agent closes the connection mid-window.
        OSError: When the connection fails.
    """
    rows: list[Reading] = []
    for _ in range(samples):
        sample = channel.next_sample()
        try:
            rows.append(
                Reading(
                    window=window,
                    extractors=count_extractors(sample),
                    clock_ms=sample["clock_ms"],
                    credits=sample["credits"],
                )
            )
        finally:
            channel.send_ack()
    return tuple(rows)


def format_reading(reading: Reading) -> str:
    """Render one reading as an NDJSON record.

    Written out in full so the archived run is re-analysable without re-playing
    it -- the summary is a conclusion, and the readings are the evidence.

    Args:
        reading: The reading.

    Returns:
        One NDJSON object, without a newline.
    """
    return (
        f'{{"window":{reading["window"]},"extractors":{reading["extractors"]},'
        f'"clock_ms":{reading["clock_ms"]},"credits":{reading["credits"]}}}'
    )


def report(readings: Sequence[Reading], price: int) -> tuple[str, ...]:
    """Turn the readings into the numbers the economy policy needs.

    Args:
        readings: Every reading taken, across all windows.
        price: What one extractor costs.

    Returns:
        The rate table followed by the two derived figures.
    """
    rates = measure(readings)
    lines = list(format_rates(rates))
    per_extractor = marginal(rates)
    if per_extractor is None:
        return (*lines, "", "not enough distinct extractor counts to measure a slope")
    payback = payback_seconds(price, per_extractor)
    lines.append("")
    lines.append(f"per extractor  {per_extractor:.2f} credits/s")
    if payback is None:
        lines.append("payback        never -- it earns nothing measurable")
    else:
        lines.append(f"payback        {payback:.1f}s at {price} credits")
    return tuple(lines)


def _stages(
    channel: AgentChannel,
    stages: int,
    idle: int,
    catalogue: Mapping[str, UnitStats],
    placements: Mapping[str, TypePlacement],
    profiles: Mapping[str, CombatProfile],
) -> tuple[list[Reading], bool]:
    """Alternate idle windows with building one more extractor.

    Args:
        channel: An open connection to the agent.
        stages: How many extractors to build.
        idle: Samples to spend in each window.
        catalogue: Unit stats by type name.
        placements: Placement rules by type name.
        profiles: Combat profiles by type name.

    Returns:
        Every reading taken, and whether all stages completed.

    Raises:
        ChannelError: When the agent closes the connection.
        OSError: When the connection fails.
    """
    readings: list[Reading] = []
    for window in range(stages + 1):
        readings.extend(observe(channel, window, idle))
        sys.stdout.write(f"# window {window} done, {len(readings)} readings\n")
        sys.stdout.flush()
        # The last window is measured and then nothing more is built -- there is
        # no point paying for an extractor no window will observe.
        if window < stages:
            plan = (EXTRACTOR_TYPE,) * (window + 1)
            # The probe must not perturb what it measures, so expansion is off
            # and nothing is reinforced: only the plan runs, and the credit
            # slope across each window is then income and nothing else
            # ([[policy-economy]]).
            report = play(
                channel,
                plan,
                catalogue,
                placements,
                profiles,
                BUILD_BUDGET,
                expand=False,
                stop_when_plan_done=True,
            )
            if report["build_outcome"] != "done":
                sys.stdout.write(f"# stage {window + 1} stopped: {report['build_reason']}\n")
                return readings, False
    return readings, True


def main(argv: Sequence[str] | None = None) -> int:
    """Measure extractor income against a live game and print the rates.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [stages] [idle-samples]``.
            ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when every stage completed, ``EXIT_INCOMPLETE`` when one
        did not, ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4, 5):
        sys.stdout.write(
            "usage: income <port> <catalogue-path> <placement-path> [stages] [idle-samples]\n"
        )
        return EXIT_BAD_USAGE
    stages = int(args[3]) if len(args) >= 4 else DEFAULT_STAGES
    idle = int(args[4]) if len(args) == 5 else DEFAULT_IDLE_SAMPLES

    catalogue = load_catalogue(Path(args[1]))
    placements = load_placements(Path(args[2]))
    profiles = load_combat_profiles(Path(args[2]))

    channel = open_channel(int(args[0]))
    channel.next_sample()
    channel.send_ack()
    readings, complete = _stages(channel, stages, idle, catalogue, placements, profiles)
    channel.close()

    for reading in readings:
        sys.stdout.write(f"{format_reading(reading)}\n")
    sys.stdout.write("\n")
    for line in report(readings, catalogue[EXTRACTOR_TYPE]["price"]):
        sys.stdout.write(f"{line}\n")
    return EXIT_OK if complete else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main(None))
