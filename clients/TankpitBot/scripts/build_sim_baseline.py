"""Generate a one-generation sim corpus and diff it against the archive.

The response-shape differ answers "which wire shapes does the real
server produce that the sim never does, and which does the sim invent".
Both halves are only as good as the sim corpus behind them, and the
shared ``runs/sim`` archive is not a corpus — it is a graveyard. It
accumulates every session ever run, so on 2026-09-02 it held 91
sessions of which 76 predated that morning's fixes, and the differ
dutifully reported 36 invented laws, most of which no longer existed in
any code path. A verdict taken over it describes the union of every sim
ever written.

**The sim is byte-deterministic.** Three sessions of the same scenario
run minutes apart on 2026-09-02 produced identical wire — 168 messages,
the same payloads in the same order, the same file size to the byte.
That is the fact this script is built around, and it kills the obvious
way to "enlarge the baseline": running one scenario N times adds N
copies of one sample and no information at all. The 91-session archive
was never 91 samples either; it was a handful of scenarios repeated
across several generations of code.

So the corpus is widened by SCENARIO and deepened by ROUNDS, never by
repetition. Each entry in :data:`SCENARIOS` drives a different command
vocabulary out of the production bot, which is what a response-shape
diff actually samples.

Usage:
    poetry run python -m scripts.build_sim_baseline [--rounds R]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from scripts import _test_hooks
from tankpit_bot.analysis.response_shapes import (
    analyze_response_shapes,
    format_response_shape_diff,
)
from tankpit_bot.runtime_artifacts import make_run_stamp
from tankpit_bot.sim.run import run_sim_session

#: The real archive the baseline is measured against.
LIVE_DIRECTORIES: tuple[Path, ...] = (Path("runs") / "bot", Path("runs") / "sniff")

#: Root under which each generated baseline gets its own stamped
#: directory. Never swept as a whole — a baseline is one subdirectory.
BASELINE_ROOT = Path("runs") / "sim-baseline"

#: Server ticks per generated session. Depth is one of the only two
#: levers a deterministic sim has, so this is generous by default: a
#: longer session carries the production bot further round its own
#: restock/hunt/engage cycle and into command families a short one
#: never reaches.
DEFAULT_ROUNDS = 400

#: Divergence rows rendered per verdict, matching the differ CLI.
ROW_LIMIT = 20


class BaselineScenarioDict(TypedDict):
    """One scenario the baseline sweeps.

    Attributes:
        label: Short name, used in the per-session stamp so the
            capture filenames say which scenario produced them.
        opponent: Whether the scripted opponent returns fire.
        practice: Face the certified practice-bot roster instead.
        ferry: Play the water-locked ferry forage scenario.
        larder: Play the own-tile collection scenario — the client
            standing ON equipment with empty slots, which is the only
            way the grant-without-a-walk and free-radar branches run.
        opponent_name: Wire name for the scripted opponent. A
            human-shaped name runs the session under the consent gate
            and the fair-fight contracts.
    """

    label: str
    opponent: bool
    practice: bool
    ferry: bool
    larder: bool
    opponent_name: str


#: The sweep. Breadth is the lever that actually buys samples, because
#: each of these drives a DIFFERENT command vocabulary out of the
#: production bot — and command vocabulary is what a response-shape
#: diff windows on. ``larder`` is the worked example: it was added
#: 2026-09-02 to reach two branches no other scenario touches, and it
#: closed two missing-law rows worth 1,940 live windows between them.
#:
#: The atlas forage scenario is deliberately absent. It reseeds the
#: container field from the mined longitudinal atlas, which moves
#: containers rather than changing which commands are sent or which
#: messages answer them — and the differ measures shapes, not
#: positions. It also depends on a 2.7 MB artifact under gitignored
#: ``runs/``, so requiring it would make this script unrunnable on a
#: fresh clone in exchange for no new shapes.
SCENARIOS: tuple[BaselineScenarioDict, ...] = (
    BaselineScenarioDict(
        label="duel", opponent=True, practice=False, ferry=False, larder=False, opponent_name=""
    ),
    BaselineScenarioDict(
        label="solo", opponent=False, practice=False, ferry=False, larder=False, opponent_name=""
    ),
    BaselineScenarioDict(
        label="practice", opponent=True, practice=True, ferry=False, larder=False, opponent_name=""
    ),
    BaselineScenarioDict(
        label="ferry", opponent=False, practice=False, ferry=True, larder=False, opponent_name=""
    ),
    BaselineScenarioDict(
        label="human",
        opponent=True,
        practice=False,
        ferry=False,
        larder=False,
        opponent_name="guest",
    ),
    BaselineScenarioDict(
        label="larder", opponent=False, practice=False, ferry=False, larder=True, opponent_name=""
    ),
)


def _int_flag(argv: list[str], flag: str, fallback: int) -> int:
    """Read one integer flag from the command line.

    Args:
        argv: Command-line tokens excluding the program name.
        flag: The flag to read, e.g. ``--rounds``.
        fallback: The value used when the flag is ABSENT. This is a
            default, not a recovery: a flag present with an unparsable
            value raises rather than falling back, because a round
            count the operator meant to set must never be silently
            replaced by another one.

    Returns:
        The parsed value, or ``fallback`` when the flag is absent.

    Raises:
        ValueError: If the flag is present with a non-integer value.
    """
    if flag not in argv:
        return fallback
    return int(argv[argv.index(flag) + 1])


def build_baseline(rounds: int, stamp: str) -> Path:
    """Play every scenario once into one fresh directory.

    Once each, not several times each: the sim is deterministic, so a
    second run of a scenario is the same bytes and buys nothing.

    Args:
        rounds: Server ticks per session.
        stamp: The baseline's own stamp, naming its directory.

    Returns:
        The directory the sessions were archived in.
    """
    archive_dir = BASELINE_ROOT / stamp
    # No mkdir: the session writer creates parents, so the directory
    # comes into being with its first capture and an abandoned run
    # leaves no empty archive for the differ to read as a silent sim.
    for scenario in SCENARIOS:
        result = run_sim_session(
            rounds,
            archive_dir=archive_dir,
            opponent=scenario["opponent"],
            practice=scenario["practice"],
            ferry=scenario["ferry"],
            larder=scenario["larder"],
            opponent_name=scenario["opponent_name"],
            stamp=f"{stamp}-{scenario['label']}",
        )
        sys.stdout.write(
            f"  {scenario['label']:<9} {result['rounds_played']} rounds, "
            f"{result['commands_sent']} commands, exit={result['exit_reason']}\n"
        )
    return archive_dir


def main() -> None:
    """Build a fresh baseline and print its fidelity diff.

    Raises:
        SystemExit: If a live archive directory is missing. Diffing on
            without it would report every sim shape as invented, which
            is an artefact of an empty comparison rather than a
            fidelity verdict.
    """
    _test_hooks.setup_rich_logging(level="INFO")
    rounds = _int_flag(sys.argv[1:], "--rounds", DEFAULT_ROUNDS)

    for directory in LIVE_DIRECTORIES:
        if not _test_hooks.path_exists(directory):
            sys.stdout.write(f"No such directory: {directory}\n")
            raise SystemExit(1)

    stamp = make_run_stamp()
    sys.stdout.write(f"building baseline {stamp}: {len(SCENARIOS)} scenarios x {rounds} rounds\n")
    archive_dir = build_baseline(rounds, stamp)
    sys.stdout.write(f"baseline archived: {archive_dir}\n\n")

    diff = analyze_response_shapes(list(LIVE_DIRECTORIES), [archive_dir])
    sys.stdout.write(format_response_shape_diff(diff, ROW_LIMIT))
    sys.stdout.write(
        f"\nre-read this baseline later with:\n"
        f"  poetry run python -m scripts.analyze_response_shapes {archive_dir}\n"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_ROOT",
    "DEFAULT_ROUNDS",
    "LIVE_DIRECTORIES",
    "ROW_LIMIT",
    "SCENARIOS",
    "BaselineScenarioDict",
    "build_baseline",
    "main",
]
