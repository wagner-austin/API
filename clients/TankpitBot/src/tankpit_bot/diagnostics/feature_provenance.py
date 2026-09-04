"""What a tick-level feature table was derived from, and the record that says so.

:mod:`tankpit_bot.diagnostics.feature_rows` reshapes an events artifact into
one row per tick, and until now wrote that table with nothing attached saying
what produced it. 539 runs and 132,266 rows is a design matrix somebody will
train on; two exports taken from different working trees were byte-comparable
and otherwise indistinguishable. This module supplies the missing half in the
vocabulary the rest of the monorepo already uses --
:class:`~platform_core.run_record.RunRecord` beside the result, on the
:func:`~platform_core.run_record.run_record_sidecar` convention -- rather than
inventing a second one. ``rusted`` took the same step on 2026-08-29 and
:mod:`rw_bot.provenance` is the model followed here.

THERE ARE TWO CONFIGURATIONS AND ONLY ONE OF THEM IS KNOWABLE. This is the
whole design, so it is stated first:

* What produced the EVENTS -- a live bot run against tankpit.com, with its
  build, doctrine, account and rank. **For artifacts written before
  2026-09-04 it is not recorded anywhere**: their records carry
  ``timestamp``, ``level``, ``logger``, ``mode``, ``channel`` and
  ``message``, and no build stamp, commit or version -- it cannot be
  recovered, and no fingerprint written today can honestly claim it.
  Artifacts written since open with a ``session_build`` diagnostic
  (build ref, distribution version, instance, doctrine, room -- never
  the account name), emitted by ``configure_bot_runtime_logging``
  (board task 7e766d65), so a future derivation can join its source
  axis to a real build instead of only a digest.
* What produced the FEATURE ROWS -- this derivation, running now. Fully
  knowable.

So the record describes the DERIVATION, and identifies the events only by
:data:`SOURCE_DISTRIBUTION`, a digest of the artifact read. That is true and
checkable. Inventing a bot version for the first half is the failure
``docs/RESEARCH.md`` was written to prevent, and the same gap it states
plainly for ``turkic-lstm``: the results already on disk have no sidecar and
cannot be given an honest one retroactively. Stamping the build at emission
time is the fix for FUTURE runs and belongs in the runtime logging path, not
here.

WHY THE DIGESTS ARE OVER CONTENT AND NOT OVER FILE BYTES. Both artifacts are
written through :func:`pathlib.Path.write_text` and read back through
:func:`pathlib.Path.read_text`, which translate line endings in text mode --
so the same logical table is ``\\n`` on Linux and ``\\r\\n`` on this
workstation. Digesting the file as it sits on disk would make a Windows
export and its byte-for-byte-identical Linux copy fingerprint differently,
and every comparison of a cluster run against a workstation-produced corpus
would report a changed input that did not change. The digest is therefore
taken over the decoded text, which is the thing that is actually the same.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint, cpu_run_fingerprint
from platform_core.determinism_record import DeterminismRecord, determinism_record
from platform_core.environment_record import (
    HostProbe,
    PackageVersion,
    VersionReader,
    capture_host_record,
    package_versions,
)
from platform_core.json_utils import dump_json_str
from platform_core.run_record import (
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
    run_record_sidecar,
)

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.feature_row_types import COUNTED_KINDS, NO_ACTION, FeatureRowDict

FEATURE_EXPERIMENT = "tankpit-tick-features"
"""What this line of work IS, for pairing two of its records.

One name across every run and every build, not one per export. The questions
the corpus exists to answer are longitudinal -- has the policy's action mix
moved between builds, did radar dispatch collapse after a change -- and two
records naming different experiments are not comparable at all, which would
make exactly those questions unaskable.
"""

SOURCE_DISTRIBUTION = "tankpit-events"
"""Name the source events artifact is recorded under in the packages axis.

The INPUT to the derivation, and the only honest handle on the live run that
produced it. Recorded as a digest rather than a filename because a path is a
label somebody maintains: ``latest.events.jsonl`` names a different artifact
every session, and two derivations over "the same" path are routinely over
different bytes.
"""

DERIVATION_DISTRIBUTION = "tankpit-bot"
"""Name the reshaping code is recorded under in the packages axis.

The derivation is pure, so its output is decided by exactly two things: the
bytes in and the code that folded them. This is the second. A change to
:data:`~tankpit_bot.diagnostics.feature_row_types.COUNTED_KINDS` or to the
last-outcome-wins rule changes every row without changing the input, and this
axis is what makes that visible instead of silent.
"""

DIGEST_LENGTH = 16
"""How many hex characters of a digest are recorded.

Full length would dominate every rendering of the axis; sixteen is far past
the point where two real artifacts collide. Matches ``rusted``'s choice so the
two projects' records read alike.
"""

DERIVATION_STACK = "tankpit-feature-rows"
"""What the determinism record names as the thing with a posture.

The derivation performs no floating-point arithmetic at all -- it counts
integers and sorts by tick -- so there is no BLAS thread count or kernel
selection to pin, and :data:`~platform_core.determinism_record.UNPINNED_STACK`
would misdescribe it as merely unpinned. What IS true is stated as a setting:
the fold is order-independent and the output is sorted, so the same input
yields byte-identical rows.
"""

ORDERING_SETTING = "row_order"
"""Setting name for how rows are ordered in the output."""

ORDERING_VALUE = "tick_ascending"
"""How rows are ordered: by tick, ascending, with no tie to read order."""

ARITHMETIC_SETTING = "float_arithmetic"
"""Setting name for whether the derivation does floating-point work."""

ARITHMETIC_VALUE = "none"
"""The derivation does no floating-point arithmetic, so none can drift."""


def content_digest(text: str) -> str:
    """Digest a table's decoded content.

    Args:
        text: The decoded text to digest.

    Returns:
        The full SHA-256 hex digest of ``text`` encoded as UTF-8. Full length
        rather than truncated, unlike the package axis: this is the payload
        digest two records are checked for bit-identity on, and truncating a
        value whose only job is exact comparison trades away the thing it is
        for.
    """
    return sha256(text.encode("utf-8")).hexdigest()


def source_artifact(source_path: Path) -> PackageVersion:
    """Identify the events artifact a feature table was derived from.

    Args:
        source_path: The events artifact that was read.

    Returns:
        The artifact, named :data:`SOURCE_DISTRIBUTION` and versioned by the
        first :data:`DIGEST_LENGTH` characters of its content digest.

    Raises:
        FileNotFoundError: When the artifact is absent. Propagated rather
            than recorded as unknown, for the reason this module exists: a
            record that says "some events" identifies nothing, and the
            derivation cannot have run without it.
        OSError: When the artifact cannot be read.
        UnicodeDecodeError: When it does not decode as UTF-8.
    """
    return PackageVersion(
        name=SOURCE_DISTRIBUTION,
        version=content_digest(_test_hooks.read_text(source_path))[:DIGEST_LENGTH],
    )


def derivation_packages(
    source_path: Path, read_version: VersionReader
) -> tuple[PackageVersion, ...]:
    """Name everything whose behaviour decides the rows.

    Args:
        source_path: The events artifact that was read.
        read_version: Reader for one distribution's installed version,
            injected so a test can state a version without owning an
            installation.

    Returns:
        The input and the code that folded it, in the shared axis's canonical
        order. Ordered by
        :func:`~platform_core.environment_record.package_versions` rather than
        here, because that is the order the record's own decoder produces: a
        tuple assembled any other way does not equal its own round trip.

    Raises:
        FileNotFoundError: When the events artifact is absent.
        OSError: When it cannot be read.
        ValueError: When a digest or a resolved version came back empty.
        importlib.metadata.PackageNotFoundError: Propagated from
            ``read_version`` when the distribution is not installed.
    """
    source = source_artifact(source_path)
    return package_versions(
        {
            source["name"]: source["version"],
            DERIVATION_DISTRIBUTION: read_version(DERIVATION_DISTRIBUTION),
        }
    )


def derivation_determinism() -> DeterminismRecord:
    """State the posture the derivation actually has.

    Returns:
        The record: no floating-point arithmetic, and rows in ascending tick
        order regardless of the order the artifact emitted them.
    """
    return determinism_record(
        stack=DERIVATION_STACK,
        settings={
            ARITHMETIC_SETTING: ARITHMETIC_VALUE,
            ORDERING_SETTING: ORDERING_VALUE,
        },
    )


def feature_fingerprint(
    get_env: Callable[[str], str | None],
    probe: HostProbe,
    source_path: Path,
    read_version: VersionReader,
) -> RunFingerprint:
    """Describe the configuration a feature table was derived under.

    Args:
        get_env: Reader for a process environment variable, for the image
            digest a launcher would export. The derivation normally runs on a
            workstation, so it is normally absent and recorded as such.
        probe: Reader for the machine's own facts.
        source_path: The events artifact that was read.
        read_version: Reader for one distribution's installed version.

    Returns:
        The fingerprint. The GPU axes are stated absent rather than omitted:
        the derivation touches no card, and empty differs from every real
        value instead of matching all of them.

    Raises:
        FileNotFoundError: When the events artifact is absent.
        OSError: When it cannot be read.
        ValueError: When the probe reports a value that cannot identify a
            machine, or a package entry came back empty.
        UnknownCoreCountError: Propagated from the probe.
    """
    return cpu_run_fingerprint(
        derivation_determinism(),
        get_env,
        capture_host_record(probe),
        derivation_packages(source_path, read_version),
    )


def run_label(source_path: Path) -> str:
    """Name which run within the experiment a table's rows are.

    Args:
        source_path: The events artifact that was read, e.g.
            ``runs/bot/artax/bot-20260806-210413.events.jsonl``.

    Returns:
        ``<instance>/<stamp>``, e.g. ``artax/bot-20260806-210413``. The
        instance is included because one stamp does not identify a run across
        a fleet -- several bots run concurrently and each writes its own
        artifact under its own directory -- and a label that collides makes
        two different runs look like two readings of one.
    """
    stamp = source_path.name.split(".")[0]
    return f"{source_path.parent.name}/{stamp}"


def _counted_total(rows: list[FeatureRowDict], kind: str) -> int:
    """Sum one counted kind across every row.

    Args:
        rows: The feature rows.
        kind: One of :data:`~tankpit_bot.diagnostics.feature_row_types.COUNTED_KINDS`.

    Returns:
        The total. Literal keys rather than a dynamic index, for the reason
        :mod:`~tankpit_bot.diagnostics.feature_rows` gives: a TypedDict is
        keyed by literal, and indexing one with a variable needs a
        suppression this codebase does not permit.
    """
    if kind == "hop_declined":
        return sum(row["hop_declined"] for row in rows)
    if kind == "radar_dispatch":
        return sum(row["radar_dispatch"] for row in rows)
    if kind == "container_pickup_dispatched":
        return sum(row["container_pickup_dispatched"] for row in rows)
    if kind == "plan_released":
        return sum(row["plan_released"] for row in rows)
    if kind == "command_error":
        return sum(row["command_error"] for row in rows)
    return sum(row["fleet_knowledge_merged"] for row in rows)


def feature_observations(rows: list[FeatureRowDict]) -> tuple[Observation, ...]:
    """Name the numbers a reader of this corpus would subtract.

    Args:
        rows: The feature rows the derivation produced.

    Returns:
        The shape of the table and the totals behind it. ``tick_span`` and
        ``tick_density`` are recorded alongside ``ticks`` for the reason
        ``rusted`` keeps counts beside its win rate: a run that emitted 400
        rows across 400 ticks and one that emitted 400 across 9,000 are the
        same size and are not the same run, and only the density separates
        them.

        An empty table reports zero for all three rather than raising. A run
        that emitted no diagnostics is a real and recordable outcome -- it is
        what a bot that died on boot produces -- and a density over no ticks
        is reported as ``0.0`` because there is no span to divide by, not
        because the ticks were empty.
    """
    ticks = len(rows)
    span = rows[-1]["tick_n"] - rows[0]["tick_n"] + 1 if rows else 0
    density = ticks / span if span > 0 else 0.0
    observations = [
        Observation(name="ticks", value=float(ticks)),
        Observation(name="tick_span", value=float(span)),
        Observation(name="tick_density", value=density),
        Observation(
            name="action_ticks",
            value=float(sum(1 for row in rows if row["action_kind"] != NO_ACTION)),
        ),
    ]
    observations.extend(
        Observation(name=f"total_{kind}", value=float(_counted_total(rows, kind)))
        for kind in COUNTED_KINDS
    )
    return tuple(observations)


def feature_run_record(
    source_path: Path,
    rows: list[FeatureRowDict],
    fingerprint: RunFingerprint,
    payload_digest: str,
) -> RunRecord:
    """Turn one derived table into the shape every experiment emits.

    Args:
        source_path: The events artifact the rows came from.
        rows: The rows produced.
        fingerprint: What they were derived under, from
            :func:`feature_fingerprint`.
        payload_digest: Digest of the written table, from
            :func:`content_digest`.

    Returns:
        The record, observations in canonical order.

    Raises:
        ValueError: When the label came out empty, which would leave the
            record unpairable.
    """
    return run_record(
        experiment=FEATURE_EXPERIMENT,
        label=run_label(source_path),
        fingerprint=fingerprint,
        observations=feature_observations(rows),
        payload_digest=payload_digest,
    )


def write_run_record(features_path: Path, record: RunRecord) -> Path:
    """Write a record beside the table it describes.

    Args:
        features_path: The features table the record describes.
        record: The record to write.

    Returns:
        The sidecar path written, from
        :func:`~platform_core.run_record.run_record_sidecar` -- the suffix is
        appended rather than substituted, so the table's own extension stays
        visible and two experiments whose results differ only by extension do
        not name one sidecar between them.
    """
    destination = run_record_sidecar(features_path)
    _test_hooks.write_text(destination, f"{dump_json_str(encode_run_record(record))}\n")
    return destination


__all__ = [
    "ARITHMETIC_SETTING",
    "ARITHMETIC_VALUE",
    "DERIVATION_DISTRIBUTION",
    "DERIVATION_STACK",
    "DIGEST_LENGTH",
    "FEATURE_EXPERIMENT",
    "ORDERING_SETTING",
    "ORDERING_VALUE",
    "SOURCE_DISTRIBUTION",
    "content_digest",
    "derivation_determinism",
    "derivation_packages",
    "feature_fingerprint",
    "feature_observations",
    "feature_run_record",
    "run_label",
    "source_artifact",
    "write_run_record",
]
