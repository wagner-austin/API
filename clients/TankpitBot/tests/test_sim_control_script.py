"""Tests for scripts.sim_control, the N=1 byte-identity control.

The recording tests play the REAL sim -- every baseline scenario and a
ghost replay of a small recorded fight -- into the fake file system, so
the manifest is the digest of what the sim actually wrote. The central
claim is checked the way the script is used: two recordings of one tree
must compare IDENTICAL, which only holds if the clock really is the only
thing that varies between runs.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    dump_json_str,
    load_json_str,
)
from scripts.build_sim_baseline import SCENARIOS
from scripts.sim_control import (
    CLOCK_KEYS,
    CONTROL_LAYOUT,
    CONTROL_POPULATION_SEED,
    CONTROL_ROOT,
    DEFAULT_GHOST_CAPTURE,
    DEFAULT_ROUNDS,
    GHOST_LABEL,
    ControlArtifactDict,
    ControlManifestDict,
    canonical_digest,
    compare_manifests,
    decode_manifest,
    encode_manifest,
    main,
    record_control,
    strip_clock,
)

from scripts import _test_hooks as script_hooks
from tests._baseline_field import baseline_field
from tests.conftest import FakeFileSystem
from tests.sim._ghost_fixtures import _fight_capture

_GHOST = Path("runs/ghost-input.capture_session.json")


@pytest.fixture()
def _field(fake_fs: FakeFileSystem) -> Generator[FakeFileSystem, None, None]:
    """The baseline field plus a recorded fight to replay as ghosts.

    Yields:
        The installed fake file system.
    """
    fake_fs.write_text(_GHOST, _fight_capture())
    real_logging = script_hooks.setup_rich_logging
    script_hooks.setup_rich_logging = _silent
    with baseline_field(fake_fs):
        try:
            yield fake_fs
        finally:
            script_hooks.setup_rich_logging = real_logging


def _silent(level: script_hooks.LogLevel) -> None:
    """Absorb the logging call without configuring handlers.

    Args:
        level: Ignored.
    """


def _manifest(artifacts: list[ControlArtifactDict]) -> ControlManifestDict:
    """A manifest played with the control's own world.

    Args:
        artifacts: Its digests.

    Returns:
        The manifest.
    """
    return ControlManifestDict(
        rounds=3,
        layout=CONTROL_LAYOUT,
        population_seed=CONTROL_POPULATION_SEED,
        ghost_capture=_GHOST.as_posix(),
        artifacts=artifacts,
    )


def test_strip_clock_removes_every_clock_key_at_every_depth() -> None:
    """Nested objects and lists lose their clock and keep everything else."""
    value = load_json_str(
        dump_json_str(
            {
                "start_timestamp_ms": 1,
                "end_timestamp_ms": 2,
                "messages": [{"timestamp_ms": 3, "payload": "AA"}],
                "timestamp": "2026-10-05T02:24:58",
                "kept": 4,
            }
        )
    )
    assert strip_clock(value) == {"messages": [{"payload": "AA"}], "kept": 4}
    assert {"timestamp", "timestamp_ms", "start_timestamp_ms", "end_timestamp_ms"} == CLOCK_KEYS


def test_strip_clock_opens_a_json_text_field() -> None:
    """The survey's ``timestamp_ms`` lives inside a ``*_json`` string."""
    value = load_json_str(dump_json_str({"survey_json": '{"timestamp_ms":9,"busy":false}'}))
    assert strip_clock(value) == {"survey_json": {"busy": False}}


def test_strip_clock_leaves_a_plain_string_alone() -> None:
    """Only a ``*_json`` key promises a document; other text is text."""
    value = load_json_str(dump_json_str({"message": '{"timestamp_ms":9}'}))
    assert strip_clock(value) == {"message": '{"timestamp_ms":9}'}


def test_a_json_text_field_that_is_not_json_raises() -> None:
    """A field named ``*_json`` holding something else is a broken artifact."""
    value = load_json_str(dump_json_str({"survey_json": "not json"}))
    with pytest.raises(InvalidJsonError):
        strip_clock(value)


def test_the_digest_ignores_the_clock_and_nothing_else() -> None:
    """Two captures a clock apart digest alike; a payload byte does not."""
    first = dump_json_str({"start_timestamp_ms": 1, "messages": [{"payload": "AA"}]})
    later = dump_json_str({"start_timestamp_ms": 7, "messages": [{"payload": "AA"}]})
    other = dump_json_str({"start_timestamp_ms": 1, "messages": [{"payload": "AB"}]})
    assert canonical_digest(first, "capture") == canonical_digest(later, "capture")
    assert canonical_digest(first, "capture") != canonical_digest(other, "capture")


def test_the_events_digest_reads_json_lines_and_skips_blank_ones() -> None:
    """An event stream is one record per line, timestamps stripped."""
    early = '{"timestamp":"a","kind":"x"}\n\n{"timestamp":"b","kind":"y"}\n'
    late = '{"timestamp":"c","kind":"x"}\n{"timestamp":"d","kind":"y"}\n'
    reordered = '{"timestamp":"c","kind":"y"}\n{"timestamp":"d","kind":"x"}\n'
    assert canonical_digest(early, "events") == canonical_digest(late, "events")
    assert canonical_digest(early, "events") != canonical_digest(reordered, "events")


def test_a_manifest_round_trips() -> None:
    """encode then decode gives back the manifest it was given."""
    manifest = _manifest(
        [
            ControlArtifactDict(label="duel", kind="capture", sha256="a" * 64),
            ControlArtifactDict(label="duel", kind="world", sha256="b" * 64),
            ControlArtifactDict(label="ghost", kind="events", sha256="c" * 64),
        ]
    )
    text = encode_manifest(manifest)
    assert text.endswith("\n")
    assert decode_manifest(text) == manifest


def test_decode_refuses_an_unknown_kind() -> None:
    """A kind outside the three is refused by name."""
    text = dump_json_str(
        {
            "rounds": 3,
            "layout": CONTROL_LAYOUT,
            "population_seed": 7,
            "ghost_capture": "g",
            "artifacts": [{"label": "duel", "kind": "replay", "sha256": "a"}],
        }
    )
    with pytest.raises(JSONTypeError, match="SIM_CONTROL_KIND"):
        decode_manifest(text)


def test_decode_refuses_a_missing_field() -> None:
    """Every manifest field is required."""
    text = dump_json_str({"rounds": 3, "artifacts": []})
    with pytest.raises(JSONTypeError, match="layout"):
        decode_manifest(text)


def test_compare_finds_nothing_between_equal_manifests() -> None:
    """Identical recordings have no differences."""
    artifacts = [ControlArtifactDict(label="duel", kind="capture", sha256="a" * 64)]
    assert compare_manifests(_manifest(artifacts), _manifest(list(artifacts))) == []


def test_compare_names_every_kind_of_difference() -> None:
    """A changed world, a changed digest and a one-sided artifact each show."""
    before = _manifest(
        [
            ControlArtifactDict(label="duel", kind="capture", sha256="a" * 64),
            ControlArtifactDict(label="solo", kind="world", sha256="b" * 64),
        ]
    )
    after = _manifest(
        [
            ControlArtifactDict(label="duel", kind="capture", sha256="f" * 64),
            ControlArtifactDict(label="ghost", kind="events", sha256="c" * 64),
        ]
    )
    after["rounds"] = 4
    assert compare_manifests(before, after) == [
        "played differently: rounds 3 vs 4",
        f"duel capture: {'a' * 12} vs {'f' * 12}",
        "ghost events: only after",
        "solo world: only before",
    ]


def test_two_recordings_of_one_tree_are_identical(
    _field: FakeFileSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    """The control's founding fact: only the clock varies between runs.

    Every baseline scenario and the ghost replay are played twice by the
    real sim, once through ``record`` and once through
    :func:`record_control` into another directory; if anything but the
    stripped clock varied, a digest would differ here.
    """
    out = Path("runs/sim-control/after.json")
    assert main(["record", str(out), "--rounds", "2", "--ghost", str(_GHOST)]) == 0
    # The sim's own log lines share stdout; the script's verdict is last.
    assert capsys.readouterr().out.endswith(
        f"recorded {3 * (len(SCENARIOS) + 1)} artifacts from "
        f"{len(SCENARIOS) + 1} sessions x 2 rounds: {out}\n"
    )
    first = decode_manifest(_field.read_text(out))
    second = record_control(2, _GHOST, CONTROL_ROOT / "second")

    labels = [scenario["label"] for scenario in SCENARIOS] + [GHOST_LABEL]
    assert [(a["label"], a["kind"]) for a in first["artifacts"]] == [
        (label, kind) for label in labels for kind in ("capture", "world", "events")
    ]
    assert first["rounds"] == 2
    assert first["layout"] == CONTROL_LAYOUT
    assert first["population_seed"] == CONTROL_POPULATION_SEED
    assert first["ghost_capture"] == _GHOST.as_posix()
    assert compare_manifests(first, second) == []
    written = _field.get_written_files()
    assert any(path.startswith(str(CONTROL_ROOT / "after" / GHOST_LABEL)) for path in written)


def test_the_defaults_are_the_standing_control() -> None:
    """150 rounds and the 2026-08-02 capture, as the 08-03 lift used."""
    assert DEFAULT_ROUNDS == 150
    assert DEFAULT_GHOST_CAPTURE.name == "bot-20260802-205105.capture_session.json"


def test_record_refuses_a_missing_ghost(
    _field: FakeFileSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without its ghost the control is not the control, so nothing plays."""
    with pytest.raises(SystemExit) as excinfo:
        main(["record", "runs/sim-control/x.json"])
    assert excinfo.value.code == 1
    assert capsys.readouterr().out.startswith(
        f"SIM_CONTROL_GHOST_MISSING: {DEFAULT_GHOST_CAPTURE} does not exist"
    )
    assert not any("sim-control" in path for path in _field.get_written_files())


@pytest.mark.parametrize(
    "argv",
    [
        ["record"],
        ["record", "--rounds", "3"],
        ["record", "m.json", "--rounds"],
        ["compare", "only-one.json"],
        ["replay"],
        [],
    ],
)
def test_usage_errors_exit_two(
    argv: list[str], _field: FakeFileSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed command line is refused before anything plays."""
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    assert capsys.readouterr().out.startswith("SIM_CONTROL_USAGE:")


def _write_manifest(fake_fs: FakeFileSystem, path: Path, sha: str) -> None:
    """Write one single-artifact manifest.

    Args:
        fake_fs: The installed fake file system.
        path: Where to write it.
        sha: The artifact's digest.
    """
    manifest = _manifest([ControlArtifactDict(label="duel", kind="capture", sha256=sha)])
    fake_fs.write_text(path, encode_manifest(manifest))


def test_compare_reports_identical(
    _field: FakeFileSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    """Equal manifests exit 0 and say how many artifacts matched."""
    _write_manifest(_field, Path("before.json"), "a" * 64)
    _write_manifest(_field, Path("after.json"), "a" * 64)
    assert main(["compare", "before.json", "after.json"]) == 0
    assert capsys.readouterr().out == "IDENTICAL: all 1 artifacts\n"


def test_compare_reports_divergence(
    _field: FakeFileSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    """Differing manifests exit 1 and list each difference."""
    _write_manifest(_field, Path("before.json"), "a" * 64)
    _write_manifest(_field, Path("after.json"), "b" * 64)
    assert main(["compare", "before.json", "after.json"]) == 1
    assert capsys.readouterr().out == (
        f"DIVERGED on 1:\n  duel capture: {'a' * 12} vs {'b' * 12}\n"
    )


def test_module_runs_as_a_script(
    _field: FakeFileSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``__main__`` entry point exits with the subcommand's code."""
    _write_manifest(_field, Path("before.json"), "a" * 64)
    _write_manifest(_field, Path("after.json"), "a" * 64)
    old_argv = sys.argv
    sys.argv = ["sim_control", "compare", "before.json", "after.json"]
    sys.modules.pop("scripts.sim_control", None)
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("scripts.sim_control", run_name="__main__")
    finally:
        sys.argv = old_argv
    assert excinfo.value.code == 0
    assert capsys.readouterr().out == "IDENTICAL: all 1 artifacts\n"
