"""Gate a run record against the known-answer registry, or register it.

Both operations existed as one-off scripts before this: register, gate,
repair the layout, and answer what a new image would do -- four of them,
each with its own copy of "read the file and decode every entry". One wrote
the file with the canonical encoder and collapsed it to a single line. The
collection logic now lives in :mod:`platform_core.known_answer_registry`;
this is the command that drives it.

TWO MODES, NAMED RATHER THAN INFERRED.

``--mode gate`` checks a record against what is already registered and
changes nothing. This is what a job should run before trusting an
environment.

``--mode register`` establishes a new entry -- and refuses unless the entry
DISCRIMINATES. Verifying that an answer matches the measurement it was built
from proves only that :func:`check_known_answer` can subtract; it is very
nearly circular. So registration also requires that a drifted value deviates
and that the same value on another card does not apply. An entry that cannot
fail is not a gate, and the failure is silent: everything passes forever.

Registration is refused for a record whose fingerprint has an empty axis.
That is not hypothetical -- the first probe measured here recorded its card
from the batch prologue rather than from inside the process, the prologue
differed between two jobs, and one row's ``driver_version`` does not exist
and cannot be recovered.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import load_json_str
from platform_core.known_answer import (
    KnownAnswer,
    check_known_answer,
    describe_known_answer_outcome,
)
from platform_core.known_answer_registry import (
    entry_from_record,
    find_entry,
    gate_record,
    read_registry,
    write_registry,
)
from platform_core.logging import get_logger
from platform_core.run_record import RunRecord, decode_run_record

_log = get_logger(__name__)

REGISTRY_FLAG = "--registry"
RECORD_FLAG = "--record"
MODE_FLAG = "--mode"
TOLERANCE_FLAG = "--tolerance"

GATE_MODE = "gate"
REGISTER_MODE = "register"
MODES = (GATE_MODE, REGISTER_MODE)

_FLAGS = (REGISTRY_FLAG, RECORD_FLAG, MODE_FLAG, TOLERANCE_FLAG)

# The card name used to prove an entry still reports a configuration move.
# Any value no real entry carries works; this one is obviously synthetic so a
# reader of the output cannot mistake it for a measurement.
_CONTROL_CARD = "SYNTHETIC CONTROL CARD"

# The drift a bit-exact entry must reject.
_CONTROL_DRIFT = 1e-9


def load_record(path: pathlib.Path) -> RunRecord:
    """Read and validate a run record.

    Args:
        path: The record written by a probe or scorer.

    Returns:
        The validated record.

    Raises:
        JSONTypeError: If the document is not a valid run record.
    """
    return decode_run_record(load_json_str(path.read_text(encoding="utf-8")))


def discrimination_failures(entry: KnownAnswer) -> tuple[str, ...]:
    """Name the ways an entry fails to discriminate.

    Args:
        entry: The candidate entry.

    Returns:
        One message per control the entry failed, empty when it passes all
        three. Reported together rather than raising on the first, because a
        caller fixing an entry wants to know everything wrong with it.
    """
    failures: list[str] = []
    fingerprint = entry["fingerprint"]
    expected = entry["expected"]

    same = check_known_answer(entry, fingerprint, expected)
    if same["kind"] != "matches":
        failures.append(f"does not match its own measurement: {same['kind']}")

    drifted = check_known_answer(entry, fingerprint, expected + _CONTROL_DRIFT)
    if drifted["kind"] != "deviates":
        failures.append(f"does not fire on a drift of {_CONTROL_DRIFT:g}: {drifted['kind']}")

    moved = check_known_answer(
        entry,
        RunFingerprint(
            image_digest=fingerprint["image_digest"],
            gpu_model=_CONTROL_CARD,
            driver_version=fingerprint["driver_version"],
            determinism=fingerprint["determinism"],
        ),
        expected,
    )
    if moved["kind"] != "configuration_differs":
        failures.append(f"does not treat a card change as a move: {moved['kind']}")

    return tuple(failures)


def run_gate(registry_path: pathlib.Path, record: RunRecord) -> int:
    """Report how the registry judges a record.

    Args:
        registry_path: The registry file.
        record: The measured run.

    Returns:
        0 when an entry covering this configuration matched, 1 otherwise --
        including when no entry covers it. "Nothing to compare against" is
        not a pass, and returning 0 there would let an unregistered
        configuration look verified.
    """
    answers = read_registry(registry_path)
    outcomes = gate_record(answers, record)

    if not outcomes:
        _log.info(
            "no registered answer carries label=%s; nothing gated this run",
            record["label"],
        )
        return 1

    matched = False
    for entry, outcome in outcomes:
        _log.info(
            "gate card=%s %s",
            entry["fingerprint"]["gpu_model"],
            describe_known_answer_outcome(entry, outcome),
        )
        if outcome["kind"] == "matches":
            matched = True
    return 0 if matched else 1


def run_register(registry_path: pathlib.Path, record: RunRecord, tolerance: float) -> int:
    """Establish a new entry from a record, refusing one that cannot gate.

    Args:
        registry_path: The registry file.
        record: The measured run to register.
        tolerance: The absolute deviation still counted as a match.

    Returns:
        0 once the entry is stored, or when an identical entry already
        exists.

    Raises:
        ValueError: If the record's fingerprint has an empty axis, if it does
            not carry exactly one observation, or if the resulting entry
            fails any discrimination control.
    """
    answers = read_registry(registry_path)
    entry = entry_from_record(record, tolerance)

    existing = find_entry(answers, entry["label"], entry["fingerprint"])
    if existing is not None:
        if existing == entry:
            _log.info(
                "an identical entry is already registered for label=%s card=%s",
                entry["label"],
                entry["fingerprint"]["gpu_model"],
            )
            return 0
        raise ValueError(
            f"an entry for label {entry['label']!r} on "
            f"{entry['fingerprint']['gpu_model']!r} is already registered with "
            f"expected={existing['expected']!r}, and this record says "
            f"{entry['expected']!r}. Registering both would leave two answers "
            f"for one configuration; decide which run was right."
        )

    failures = discrimination_failures(entry)
    if failures:
        raise ValueError("refusing an entry that does not discriminate: " + "; ".join(failures))

    write_registry(registry_path, (*answers, entry))
    _log.info(
        "registered label=%s card=%s driver=%s image=%s expected=%r tolerance=%r",
        entry["label"],
        entry["fingerprint"]["gpu_model"],
        entry["fingerprint"]["driver_version"],
        entry["fingerprint"]["image_digest"],
        entry["expected"],
        entry["tolerance"],
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Gate or register one run record.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 on success; 1 when a gate found no matching entry.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, a
            required flag is absent, the mode is not one of the two, or
            ``--tolerance`` is not a number.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    mode = cli_args.require_flag(parsed, MODE_FLAG)
    if mode not in MODES:
        raise ValueError(f"{MODE_FLAG} must be one of {list(MODES)}, got {mode!r}")

    registry_path = pathlib.Path(cli_args.require_flag(parsed, REGISTRY_FLAG))
    record = load_record(pathlib.Path(cli_args.require_flag(parsed, RECORD_FLAG)))

    if mode == GATE_MODE:
        return run_gate(registry_path, record)

    raw_tolerance = cli_args.require_flag(parsed, TOLERANCE_FLAG)
    try:
        tolerance = float(raw_tolerance)
    except ValueError as exc:
        raise ValueError(f"{TOLERANCE_FLAG} must be a number, got {raw_tolerance!r}") from exc
    return run_register(registry_path, record, tolerance)


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = [
    "discrimination_failures",
    "entrypoint",
    "load_record",
    "main",
    "run_gate",
    "run_register",
]


if __name__ == "__main__":
    entrypoint()
