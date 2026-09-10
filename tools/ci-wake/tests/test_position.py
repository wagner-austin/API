"""The bridge's memory: which pushes have already been posted about.

The reader tolerates nothing, and the cost of the alternative is specific: a
line read as absent means the push it names is announced AGAIN, and the
reader of that second post cannot tell it from a genuinely new verdict.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    dump_json_str,
    load_json_str,
)

from ci_wake import _test_hooks
from ci_wake.enrolment import attempt_key
from ci_wake.position import (
    AnnouncedPush,
    append_announced,
    decode_announced_push,
    encode_announced_push,
    position_path,
    read_announced,
)
from tests.conftest import OTHER_SHA, REPO, SHA


def _record(*, sha: str = SHA, state: str = "ripe", at: int = 1788700000) -> AnnouncedPush:
    """Build one position record.

    Args:
        sha: The announced push's sha.
        state: The state that closed the row.
        at: When the post landed.

    Returns:
        The record.
    """
    return AnnouncedPush(key=attempt_key(REPO, sha), state=state, announced_unix=at)


class TestCodec:
    def test_a_record_round_trips(self) -> None:
        record = _record()

        assert decode_announced_push(encode_announced_push(record)) == record

    def test_a_non_object_refuses(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_announced_push("ripe")

    def test_a_missing_field_refuses(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_announced_push({"key": attempt_key(REPO, SHA), "state": "ripe"})


class TestPositionPath:
    def test_it_sits_beside_the_enrolment_record(self) -> None:
        """Derived rather than configured: two files that must describe the
        same set of pushes should not be separately addressable, or moving
        one leaves the other behind and every push is announced twice."""
        assert position_path(pathlib.Path("/runs/pushes.jsonl")) == pathlib.Path(
            "/runs/announced.jsonl"
        )


class TestReadAnnounced:
    def test_an_absent_record_reads_as_empty(self, tmp_path: pathlib.Path) -> None:
        """A machine whose bridge has never run has announced nothing, and
        refusing the first cycle would make the bridge impossible to start."""
        assert read_announced(tmp_path / "announced.jsonl") == frozenset()

    def test_every_key_written_is_read_back(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "announced.jsonl"
        append_announced(path, _record(sha=SHA))
        append_announced(path, _record(sha=OTHER_SHA, state="stalled"))

        assert read_announced(path) == {attempt_key(REPO, SHA), attempt_key(REPO, OTHER_SHA)}

    def test_blank_lines_are_skipped(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "announced.jsonl"
        path.write_text(
            dump_json_str(encode_announced_push(_record())) + "\n\n  \n", encoding="utf-8"
        )

        assert read_announced(path) == {attempt_key(REPO, SHA)}

    def test_a_non_object_line_is_fatal_and_names_its_line_number(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "announced.jsonl"
        path.write_text(
            dump_json_str(encode_announced_push(_record())) + '\n"ripe"\n', encoding="utf-8"
        )

        with pytest.raises(JSONTypeError, match="line 2"):
            read_announced(path)

    def test_a_line_that_is_not_json_at_all_is_fatal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "announced.jsonl"
        path.write_text("}\n", encoding="utf-8")

        with pytest.raises(InvalidJsonError):
            read_announced(path)

    def test_it_reads_through_the_seam_rather_than_the_filesystem(
        self, tmp_path: pathlib.Path
    ) -> None:
        seen: list[pathlib.Path] = []

        def _exists(path: pathlib.Path) -> bool:
            seen.append(path)
            return False

        _test_hooks.file_exists = _exists

        assert read_announced(tmp_path / "announced.jsonl") == frozenset()
        assert seen == [tmp_path / "announced.jsonl"]


class TestAppendAnnounced:
    def test_the_state_that_closed_the_row_is_recorded_beside_the_key(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A push closed as ``stalled`` whose runs later finished is a real
        sequence, and a reader needs to see that the verdict they never
        received was not lost but deliberately not waited for."""
        path = tmp_path / "announced.jsonl"

        append_announced(path, _record(state="stalled"))

        written = decode_announced_push(load_json_str(path.read_text(encoding="utf-8").strip()))
        assert written["state"] == "stalled"
