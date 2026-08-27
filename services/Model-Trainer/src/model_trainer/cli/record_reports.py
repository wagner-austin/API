"""Reading a directory of run records, and saying how they differ.

Shared by every report that compares several runs of one experiment. It began
inside the ladder report and moved here when the forward trace needed the same
three things: read every record in a directory, difference each run's
configuration against a reference, and name the axes on which a comparison
stops meaning anything. A second copy would be free to drift, and the drift
would be in the part that decides whether a reading is admissible at all.

WHY THE CONFIGURATION DIFFERENCES ARE COMPUTED RATHER THAN PRINTED FOR THE
READER TO CHECK. An earlier revision printed fingerprints and trusted the
caller, on the reasoning that they submitted the runs and know what they are.
That is the same shape as every defect this apparatus exists to catch: a check
a person has to remember to perform by eye is not a check.
"""

from __future__ import annotations

import pathlib

from platform_core.comparability import find_differences
from platform_core.json_utils import load_json_str
from platform_core.known_answer_registry import incomplete_axes
from platform_core.run_record import RunRecord, decode_run_record

#: Digits used when printing a measured value. Seventeen significant digits is
#: what it takes to write an IEEE double so it reads back unchanged, and
#: anything shorter would print two differing values identically -- which is
#: the exact mistake these reports exist to prevent.
VALUE_DIGITS = 17

#: Fingerprint axes whose difference makes a cross-card reading meaningless
#: rather than merely qualified. Two runs in different images that disagree
#: have not told you anything about their cards -- the image is the variable
#: the whole apparatus pins so that the card can be the one that moves.
#: Determinism is deliberately absent: it is a nested record whose differences
#: the per-axis listing already names, and a change to it is a finding rather
#: than a confound.
CONFOUNDING_AXES = ("image_digest",)


def read_run_records(directory: pathlib.Path) -> tuple[tuple[str, RunRecord], ...]:
    """Read every record in a directory, in filename order.

    Every ``.json`` file is read and decoded. Nothing is skipped on the way:
    a file that does not decode fails the command, because a directory of
    records with one unreadable member is a directory whose agreement result
    would silently cover fewer runs than the reader thinks.

    Args:
        directory: Directory holding one record per run.

    Returns:
        ``(filename, record)`` pairs, sorted by filename so the value columns
        are in the same order every time this is run.

    Raises:
        FileNotFoundError: If the directory does not exist, or holds no
            ``.json`` file at all.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"no such directory: {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no .json records in {directory}")
    return tuple(
        (path.name, decode_run_record(load_json_str(path.read_text(encoding="utf-8"))))
        for path in paths
    )


def configuration_lines(named_records: tuple[tuple[str, RunRecord], ...]) -> tuple[str, ...]:
    """Report how each run's configuration differs from the first.

    The first run is the reference simply because a set needs one; which run
    holds the position does not change which axes are named.

    Args:
        named_records: ``(filename, record)`` pairs, in report order.

    Returns:
        One line per run, plus a warning line for every confounding axis found
        anywhere in the set.
    """
    reference = named_records[0][1]["fingerprint"]
    lines = [f"  [0] {named_records[0][0]}  (reference)"]
    confounded: list[str] = []

    for index, (name, record) in enumerate(named_records[1:], start=1):
        differences = find_differences(reference, record["fingerprint"])
        axes = [d["axis"] for d in differences]
        described = ", ".join(axes) if axes else "identical to the reference"
        lines.append(f"  [{index}] {name}  differs on: {described}")
        confounded.extend(axis for axis in axes if axis in CONFOUNDING_AXES)

    # An axis nobody recorded compares EQUAL to the same gap in another run,
    # so two runs that each failed to record their image read as "identical
    # configuration". They are not; they are two runs that cannot say. The
    # empty axis has to be named separately from the differing ones, because
    # the difference machinery is structurally unable to see it.
    for index, (name, record) in enumerate(named_records):
        empty = incomplete_axes(record["fingerprint"])
        if empty:
            lines.append(
                f"  !! [{index}] {name} recorded no {', '.join(empty)}. An unrecorded axis"
            )
            lines.append(
                "     matches every other unrecorded one, so agreement here is not evidence."
            )
            # NOT added to `confounded`. That warning reads "these runs do not
            # all share one X", which is false here -- they share the absence.
            # The line above already says the true thing. A set where only SOME
            # runs recorded the axis reaches `confounded` through the
            # differencing loop, because empty differs from non-empty there.

    for axis in sorted(set(confounded)):
        lines.append(f"  !! these runs do not all share one {axis}. A disagreement between")
        lines.append(
            "     them is not a fact about the cards, and this report cannot separate them."
        )
    return tuple(lines)


__all__ = [
    "CONFOUNDING_AXES",
    "VALUE_DIGITS",
    "configuration_lines",
    "read_run_records",
]
