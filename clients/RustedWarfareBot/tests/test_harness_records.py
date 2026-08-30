"""Filing what a batch's numbers were produced under, beside the numbers.

Driven against the in-memory host, so the real reader, the real aggregate and
the real codec run -- only the filesystem is a fake.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from platform_core.comparability import NO_VALUE
from platform_core.json_utils import load_json_str
from platform_core.run_record import RUN_RECORD_SUFFIX, decode_run_record

from rw_bot.harness.records import (
    batch_fingerprint,
    read_batch_rows,
    read_extractors,
    write_arm_records,
)
from rw_bot.provenance import (
    ASSETS_DISTRIBUTION,
    GAME_DISTRIBUTION,
    JVM_DISTRIBUTION,
    SWEEP_EXPERIMENT,
)
from rw_bot.tree_identity import TreeIdentityError
from tests.harness_fakes import FakeHost
from tests.sample_game import LINUX, SAMPLE_JAVA_VERSION, write_sample_game

#: The other platform, so the runtime axis is exercised as both.
WINDOWS = "win32"

OUT_DIR = Path("runs/sweeps/demo")
TRACES = Path("runs/traces")
BATCH = "demo"

#: A card carrying the figures an arm's aggregate is built from.
CARD = (
    "verdict        won (eliminated)",
    "extractors     4 -> 2",
    "total worth    3500 -> 4700",
    "enemies seen   17 -> 20 (3 engageable)",
    "intercepted    5",
    "income         18/s",
)

#: A trace whose header is followed by sample lines, in the real column order
#: read off ``runs/traces/battery96``: frame, army, credits, enemies,
#: EXTRACTORS, lost, producers, idle, orders, refused, worth, rival. Twelve
#: columns, because fewer is an archive trace whose extractor column is not
#: there to read.
TRACE = (
    "frame army credits enemies extractors lost producers idle orders refused worth rival",
    "0 0 4004 2 3 0 0 0 0 0 3500 3500",
    "75 0 4009 2 9 0 1 1 1 0 3900 3900",
    "150 0 4014 2 4 0 1 1 2 0 4700 4700",
)


def _host_with_batch(*stems: str) -> FakeHost:
    """Build a host holding a filed batch and a real game jar.

    Args:
        *stems: Result filenames without their suffix.

    Returns:
        The host.
    """
    host = FakeHost(platform="linux")
    host.dirs.add(OUT_DIR.as_posix())
    for stem in stems:
        host.files[f"{OUT_DIR.as_posix()}/{stem}.txt"] = CARD
        host.files[f"{TRACES.as_posix()}/{BATCH}/{stem}.ndjson"] = TRACE
    return host


class TestReadingATrace:
    def test_the_peak_and_the_drop_are_both_reported(self) -> None:
        """The scorecard says ``4 -> 2``; the trace says it held nine."""
        assert read_extractors(TRACE) == (9, 5)

    def test_the_header_is_not_a_sample(self) -> None:
        assert read_extractors((TRACE[0],)) == (0, 0)

    def test_a_trace_with_no_samples_reports_nothing_held(self) -> None:
        """What an interrupted match leaves."""
        assert read_extractors(()) == (0, 0)

    def test_a_short_archive_line_is_not_read(self) -> None:
        """A thirteen-column archive trace has no honest value in the
        extractor column, so it is skipped rather than misread."""
        assert read_extractors((TRACE[0], "1 100 5")) == (0, 0)


class TestReadingABatch:
    def test_every_filed_result_becomes_a_row(self) -> None:
        with _host_with_batch("attack-s1", "attack-s2", "defend-s1"):
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
        assert [(row["arm"], row["seed"]) for row in rows] == [
            ("attack", 1),
            ("attack", 2),
            ("defend", 1),
        ]

    def test_rows_carry_the_trace_figures(self) -> None:
        with _host_with_batch("attack-s1"):
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
        assert (rows[0]["peak"], rows[0]["dropped"]) == (9, 5)

    def test_a_result_with_no_trace_still_reads(self) -> None:
        """A match whose trace was lost is still a filed result, and dropping
        it would silently shrink the arm it belonged to."""
        with _host_with_batch("attack-s1") as host:
            del host.files[f"{TRACES.as_posix()}/{BATCH}/attack-s1.ndjson"]
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
        assert (rows[0]["peak"], rows[0]["dropped"]) == (0, 0)

    def test_a_partial_transcript_is_not_a_result(self) -> None:
        """A match that printed no verdict is kept as ``.partial`` precisely
        so it is not counted; reading it would file a blank as a measurement."""
        with _host_with_batch("attack-s1") as host:
            host.files[f"{OUT_DIR.as_posix()}/attack-s9.partial"] = ("### FAILED",)
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
        assert [row["seed"] for row in rows] == [1]

    def test_an_empty_batch_reads_as_no_rows(self) -> None:
        with FakeHost(platform="linux") as host:
            host.dirs.add(OUT_DIR.as_posix())
            assert read_batch_rows(OUT_DIR, TRACES, BATCH) == ()


class TestTheFingerprint:
    def test_it_carries_the_code_the_runtime_and_the_data(self, tmp_path: Path) -> None:
        """All three digested from the bytes that ran, not from a maintained
        label. The runtime and the assets were absent from this axis until
        2026-08-29, which made two Linux batches on different bundled JREs
        fingerprint identically."""
        game = write_sample_game(tmp_path / "game")
        with FakeHost(platform=LINUX):
            fingerprint = batch_fingerprint(str(game))
        assert {p["name"] for p in fingerprint["packages"]} == {
            GAME_DISTRIBUTION,
            JVM_DISTRIBUTION,
            ASSETS_DISTRIBUTION,
        }

    def test_the_runtime_read_is_the_one_this_machine_would_run(self, tmp_path: Path) -> None:
        """The platform comes from the running interpreter here and nowhere
        else in this package: everything else composes a command for a machine
        it is not on, while this describes a batch that already played."""
        game = write_sample_game(tmp_path / "game", platform=WINDOWS)
        with FakeHost(platform=WINDOWS):
            fingerprint = batch_fingerprint(str(game))
        runtime = [p for p in fingerprint["packages"] if p["name"] == JVM_DISTRIBUTION]
        assert runtime[0]["version"].startswith(SAMPLE_JAVA_VERSION)

    def test_an_absent_jar_is_refused_rather_than_recorded_as_unknown(self) -> None:
        """A record saying "some build" about an obfuscated binary says
        nothing, and every claim here is valid for one build only."""
        with FakeHost(platform=LINUX), pytest.raises(FileNotFoundError):
            batch_fingerprint("no-such-directory")

    def test_an_absent_asset_tree_is_refused(self, tmp_path: Path) -> None:
        """The jar alone used to be enough to build a record. It is not: a
        clone missing its maps sends the engine to its boot sandbox and voids
        every scorecard, with the jar digest matching throughout."""
        game = write_sample_game(tmp_path / "game")
        shutil.rmtree(game / "assets")
        with FakeHost(platform=LINUX), pytest.raises(TreeIdentityError) as caught:
            batch_fingerprint(str(game))
        assert caught.value.code == "RW-TREE-001"

    def test_a_workstation_run_states_no_image_and_no_card(self, tmp_path: Path) -> None:
        game = write_sample_game(tmp_path / "game")
        with FakeHost(platform=LINUX):
            fingerprint = batch_fingerprint(str(game))
        assert fingerprint["image_digest"] == NO_VALUE
        assert fingerprint["gpu_model"] == NO_VALUE


class TestWritingTheRecords:
    def _fingerprint(self, tmp_path: Path) -> Path:
        """Plant a game directory so a fingerprint can be built.

        A whole tree rather than a lone jar, because the packages axis names
        the runtime and the assets beside it and would refuse a directory
        holding only code.

        Args:
            tmp_path: Directory to write into.

        Returns:
            The game directory.
        """
        return write_sample_game(tmp_path / "game")

    def test_one_record_is_filed_per_arm(self, tmp_path: Path) -> None:
        game = self._fingerprint(tmp_path)
        with _host_with_batch("attack-s1", "attack-s2", "defend-s1") as host:
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
            arms = write_arm_records(OUT_DIR, BATCH, rows, batch_fingerprint(str(game)))
        assert arms == ("attack", "defend")
        assert f"{OUT_DIR.as_posix()}/attack{RUN_RECORD_SUFFIX}" in host.files
        assert f"{OUT_DIR.as_posix()}/defend{RUN_RECORD_SUFFIX}" in host.files

    def test_the_filed_record_decodes_through_the_shared_codec(self, tmp_path: Path) -> None:
        """Written as JSON a reader can load, not as prose."""
        game = self._fingerprint(tmp_path)
        with _host_with_batch("attack-s1", "attack-s2") as host:
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
            write_arm_records(OUT_DIR, BATCH, rows, batch_fingerprint(str(game)))
            text = "\n".join(host.files[f"{OUT_DIR.as_posix()}/attack{RUN_RECORD_SUFFIX}"])
        record = decode_run_record(load_json_str(text))
        assert record["experiment"] == SWEEP_EXPERIMENT
        assert record["label"] == "demo/attack"

    def test_the_record_carries_the_arms_own_numbers(self, tmp_path: Path) -> None:
        game = self._fingerprint(tmp_path)
        with _host_with_batch("attack-s1", "attack-s2", "defend-s1") as host:
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
            write_arm_records(OUT_DIR, BATCH, rows, batch_fingerprint(str(game)))
            text = "\n".join(host.files[f"{OUT_DIR.as_posix()}/attack{RUN_RECORD_SUFFIX}"])
        record = decode_run_record(load_json_str(text))
        named = {o["name"]: o["value"] for o in record["observations"]}
        assert named["matches"] == 2.0
        assert named["wins"] == 2.0
        assert named["win_rate"] == 1.0

    def test_re_running_replaces_rather_than_appends(self, tmp_path: Path) -> None:
        """A batch is resumable: a pass that plays the last matches must leave
        a record covering all of them, not a second record beside the first."""
        game = self._fingerprint(tmp_path)
        with _host_with_batch("attack-s1") as host:
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
            write_arm_records(OUT_DIR, BATCH, rows, batch_fingerprint(str(game)))
            host.files[f"{OUT_DIR.as_posix()}/attack-s2.txt"] = CARD
            host.files[f"{TRACES.as_posix()}/{BATCH}/attack-s2.ndjson"] = TRACE
            rows = read_batch_rows(OUT_DIR, TRACES, BATCH)
            write_arm_records(OUT_DIR, BATCH, rows, batch_fingerprint(str(game)))
            text = "\n".join(host.files[f"{OUT_DIR.as_posix()}/attack{RUN_RECORD_SUFFIX}"])
        record = decode_run_record(load_json_str(text))
        named = {o["name"]: o["value"] for o in record["observations"]}
        assert named["matches"] == 2.0
