"""What the feature table's run record claims, and what it refuses to.

The record exists so a row can be traced to what produced it, so these
tests are mostly about identity: the same content fingerprints the same
way wherever it sits, different content does not, and the axes the
derivation cannot honestly know are stated absent rather than invented.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.determinism_record import DeterminismRecord
from platform_core.environment_record import HostProbe
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record
from platform_core.testing import FakeHostProbe

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.fs import WriteTextProtocol
from tankpit_bot.diagnostics.feature_provenance import (
    ARITHMETIC_SETTING,
    ARITHMETIC_VALUE,
    DERIVATION_DISTRIBUTION,
    DERIVATION_STACK,
    DIGEST_LENGTH,
    FEATURE_EXPERIMENT,
    ORDERING_SETTING,
    ORDERING_VALUE,
    SOURCE_DISTRIBUTION,
    content_digest,
    derivation_determinism,
    derivation_packages,
    feature_fingerprint,
    feature_observations,
    feature_run_record,
    run_label,
    source_artifact,
    write_run_record,
)
from tankpit_bot.diagnostics.feature_row_types import COUNTED_KINDS, NO_ACTION, FeatureRowDict

_BOT_VERSION = "9.9.9"


def _read_version(distribution: str) -> str:
    """State a distribution's version without owning an installation.

    Args:
        distribution: The distribution asked about.

    Returns:
        A fixed version for the bot, so a record can be asserted
        literally rather than against whatever is installed.

    Raises:
        ValueError: When asked about any other distribution, so a test
            that widened the package axis without meaning to fails
            rather than silently recording a second version.
    """
    if distribution != DERIVATION_DISTRIBUTION:
        raise ValueError(f"unexpected distribution {distribution!r}")
    return _BOT_VERSION


def _probe() -> HostProbe:
    """Build the stated machine every record here is fingerprinted on.

    Returns:
        The fake probe.
    """
    probe: HostProbe = FakeHostProbe(platform="Linux-6.1", machine="x86_64", logical_cores=8)
    return probe


def _no_env(key: str) -> str | None:
    """Report an unset environment, as a workstation run has.

    Args:
        key: The variable asked about.

    Returns:
        None, always.
    """
    _ = key
    return None


def _row(
    tick_n: int,
    *,
    action_kind: str = NO_ACTION,
    hop_declined: int = 0,
    radar_dispatch: int = 0,
    container_pickup_dispatched: int = 0,
    plan_released: int = 0,
    command_error: int = 0,
    fleet_knowledge_merged: int = 0,
) -> FeatureRowDict:
    """Build one feature row, defaulting every field a test does not set.

    Keyword parameters rather than a ``**overrides`` update, because a
    TypedDict updated from an open mapping cannot be checked -- and the
    suppression that would silence it is not permitted here.

    Args:
        tick_n: The tick the row describes.
        action_kind: The action dispatched, if any.
        hop_declined: Declined hop lanes on the tick.
        radar_dispatch: Radar dispatches on the tick.
        container_pickup_dispatched: Pickups dispatched on the tick.
        plan_released: Plan releases on the tick.
        command_error: Command errors on the tick.
        fleet_knowledge_merged: Fleet knowledge merges on the tick.

    Returns:
        The row.
    """
    return FeatureRowDict(
        tick_n=tick_n,
        bot_state="COLLECT/SENSE",
        action_kind=action_kind,
        outcome=NO_ACTION,
        duration_ms=-1,
        attempt_id=-1,
        hop_declined=hop_declined,
        radar_dispatch=radar_dispatch,
        container_pickup_dispatched=container_pickup_dispatched,
        plan_released=plan_released,
        command_error=command_error,
        fleet_knowledge_merged=fleet_knowledge_merged,
    )


def _artifact(tmp_path: Path, text: str) -> Path:
    """Write a source artifact to read back.

    Args:
        tmp_path: Directory to write into.
        text: Content to write.

    Returns:
        The path written.
    """
    path = tmp_path / "bot-20260806-210413.events.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_same_content_digests_the_same_from_two_different_paths(tmp_path: Path) -> None:
    """Identity is the content, not where the file sits.

    Two copies of one artifact under different names must produce one
    source axis, because they ARE one input -- a corpus staged to the
    cluster is the workstation's files at different paths, and a digest
    that moved with the path would report every staged run as a new
    input.
    """
    left = tmp_path / "a" / "bot-1.events.jsonl"
    right = tmp_path / "b" / "renamed.events.jsonl"
    for path in (left, right):
        path.parent.mkdir()
        path.write_text('{"same": "bytes"}\n', encoding="utf-8")

    assert source_artifact(left) == source_artifact(right)


def test_a_changed_artifact_changes_the_source_axis(tmp_path: Path) -> None:
    """One edited event is a different input and must say so."""
    before = source_artifact(_artifact(tmp_path, '{"tick_n": 1}\n'))
    after = source_artifact(_artifact(tmp_path, '{"tick_n": 2}\n'))

    assert before["name"] == after["name"] == SOURCE_DISTRIBUTION
    assert before["version"] != after["version"]
    assert len(after["version"]) == DIGEST_LENGTH


def test_a_missing_artifact_raises_rather_than_recording_an_unknown(tmp_path: Path) -> None:
    """A record that says "some events" identifies nothing.

    The derivation cannot have run without its input, so the absence is
    a failure rather than a value.
    """
    with pytest.raises(FileNotFoundError):
        source_artifact(tmp_path / "absent.events.jsonl")


def test_the_package_axis_names_the_input_and_the_code_that_folded_it(tmp_path: Path) -> None:
    """Both halves decide the rows, so both are recorded, in sorted order."""
    packages = derivation_packages(_artifact(tmp_path, "{}\n"), _read_version)

    assert [entry["name"] for entry in packages] == [
        DERIVATION_DISTRIBUTION,
        SOURCE_DISTRIBUTION,
    ]
    assert packages[0]["version"] == _BOT_VERSION


def test_the_determinism_record_states_the_posture_the_derivation_has() -> None:
    """No floating-point work and a sorted output, rather than "unpinned".

    ``UNPINNED_STACK`` would say nothing pinned this run's arithmetic,
    which is true only because there is no arithmetic to pin -- a
    different claim, and the one a later reader needs.
    """
    record: DeterminismRecord = derivation_determinism()

    assert record["stack"] == DERIVATION_STACK
    assert record["settings"] == (
        (ARITHMETIC_SETTING, ARITHMETIC_VALUE),
        (ORDERING_SETTING, ORDERING_VALUE),
    )


def test_the_gpu_axes_are_stated_absent_rather_than_omitted(tmp_path: Path) -> None:
    """A derivation touching no card must not compare equal to one that did.

    Empty differs from every real value; an omitted axis would match
    any other record missing the same axis.
    """
    fingerprint = feature_fingerprint(_no_env, _probe(), _artifact(tmp_path, "{}\n"), _read_version)

    assert fingerprint["gpu_model"] == ""
    assert fingerprint["driver_version"] == ""
    assert fingerprint["image_digest"] == ""
    assert fingerprint["host"]["machine"] == "x86_64"
    assert fingerprint["host"]["logical_cores"] == 8


def test_the_image_digest_is_recorded_when_a_launcher_exported_one(tmp_path: Path) -> None:
    """On the cluster the payload is told its image, and the record carries it."""

    def _in_image(key: str) -> str | None:
        return "sha256:abc" if key == "IMAGE_DIGEST" else None

    fingerprint = feature_fingerprint(
        _in_image, _probe(), _artifact(tmp_path, "{}\n"), _read_version
    )

    assert fingerprint["image_digest"] == "sha256:abc"


def test_the_label_names_the_instance_as_well_as_the_stamp() -> None:
    """A stamp alone collides across a fleet.

    Several bots run concurrently and each writes its own artifact, so
    a label without the instance would make two different runs look
    like two readings of one.
    """
    assert (
        run_label(Path("runs/bot/artax/bot-20260806-210413.events.jsonl"))
        == "artax/bot-20260806-210413"
    )


def test_the_observations_carry_the_shape_of_the_table_and_its_totals() -> None:
    """Density separates two tables that are the same size.

    Three rows spanning ticks 1..10 and three spanning 1..3 are both
    three rows and are not the same run.
    """
    rows = [
        _row(1, hop_declined=2, action_kind="scan"),
        _row(4, radar_dispatch=1, command_error=3),
        _row(10, container_pickup_dispatched=5, plan_released=1, fleet_knowledge_merged=7),
    ]

    values = {o["name"]: o["value"] for o in feature_observations(rows)}

    assert values["ticks"] == 3.0
    assert values["tick_span"] == 10.0
    assert values["tick_density"] == 0.3
    assert values["action_ticks"] == 1.0
    assert values["total_hop_declined"] == 2.0
    assert values["total_radar_dispatch"] == 1.0
    assert values["total_container_pickup_dispatched"] == 5.0
    assert values["total_plan_released"] == 1.0
    assert values["total_command_error"] == 3.0
    assert values["total_fleet_knowledge_merged"] == 7.0


def test_every_counted_kind_gets_an_observation() -> None:
    """The totals track the vocabulary, so a new kind cannot be silently dropped."""
    names = {o["name"] for o in feature_observations([_row(1)])}

    assert {f"total_{kind}" for kind in COUNTED_KINDS} <= names


def test_an_empty_table_reports_zeros_rather_than_raising() -> None:
    """A run that emitted no diagnostics is a real outcome.

    It is what a bot that died on boot produces, and the density is
    ``0.0`` because there is no span to divide by -- not because the
    ticks were empty.
    """
    values = {o["name"]: o["value"] for o in feature_observations([])}

    assert values["ticks"] == 0.0
    assert values["tick_span"] == 0.0
    assert values["tick_density"] == 0.0
    assert values["action_ticks"] == 0.0


def test_a_single_tick_spans_one_tick_and_is_fully_dense() -> None:
    """The span is inclusive, so one row is a span of one and not of zero."""
    values = {o["name"]: o["value"] for o in feature_observations([_row(7)])}

    assert values["tick_span"] == 1.0
    assert values["tick_density"] == 1.0


def test_the_record_names_the_experiment_and_survives_its_own_round_trip(
    tmp_path: Path,
) -> None:
    """One experiment name across every run, and a record that decodes.

    The round trip is the assertion that matters: a record whose
    observations were assembled in the wrong order does not equal its
    own decode, which is how that class of bug is caught.
    """
    source = _artifact(tmp_path, "{}\n")
    record = feature_run_record(
        source,
        [_row(1, radar_dispatch=2)],
        feature_fingerprint(_no_env, _probe(), source, _read_version),
        content_digest("rows\n"),
    )

    assert record["experiment"] == FEATURE_EXPERIMENT
    assert record["label"] == f"{tmp_path.name}/bot-20260806-210413"
    assert record["payload_digest"] == content_digest("rows\n")

    written: dict[Path, str] = {}

    def _capture(path: Path, content: str) -> None:
        written[path] = content

    fake: WriteTextProtocol = _capture
    saved = _test_hooks.write_text
    _test_hooks.write_text = fake
    try:
        destination = write_run_record(tmp_path / "s.features.jsonl", record)
    finally:
        _test_hooks.write_text = saved

    assert destination.name == "s.features.jsonl.runrecord.json"
    decoded = decode_run_record(narrow_json_to_dict(load_json_str(written[destination])))
    assert decoded == record


def test_the_payload_digest_is_over_content_so_line_endings_cannot_move_it() -> None:
    """A Windows export and its Linux copy are the same table.

    Digesting the file as it sits on disk would fingerprint them
    differently and report a changed input on every cluster-versus-
    workstation comparison, which is the comparison the corpus exists
    for.
    """
    assert content_digest("a\nb\n") != content_digest("a\r\nb\r\n")
    assert content_digest("a\nb\n") == content_digest("a\nb\n")
