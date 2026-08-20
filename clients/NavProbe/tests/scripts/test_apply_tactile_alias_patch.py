"""Tests for the tactile alias patch.

Every test here runs the real function against a real file on disk. The property
that matters is byte-level: the patch must round-trip exactly, because a changed
byte changes Warp's module hash and therefore the kernel-cache key for every
kernel in the module -- which would silently recompile, or silently reuse, at the
moment a determinism measurement depends on neither happening.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.apply_tactile_alias_patch import (
    _LAUNCH_OLD,
    _MAX_NEW,
    _MAX_OLD,
    _SIG_OLD,
    PatchSiteError,
    main,
)
from scripts.arguments import ScriptArgumentError

from scripts import _test_hooks
from tests.scripts.vendor import RecordingWriter

#: A file carrying each patch site exactly once, in the vendor's own layout.
_SOURCE = "\n".join(["prelude", _SIG_OLD, "middle", _MAX_OLD, "tail", _LAUNCH_OLD, "end"]) + "\n"


@pytest.fixture()
def writer() -> Generator[RecordingWriter, None, None]:
    """Capture what the script writes, then restore the hook.

    Yields:
        The recording writer installed for the test.
    """
    saved = _test_hooks.write_out
    recorder = RecordingWriter()
    _test_hooks.write_out = recorder
    yield recorder
    _test_hooks.write_out = saved


def _write(path: Path, text: str, line_ending: str) -> bytes:
    """Write a source file with a chosen line ending.

    Args:
        path: File to write.
        text: Contents, using ``\\n`` internally.
        line_ending: The ending to convert to.

    Returns:
        The bytes written.
    """
    raw = text.replace("\n", line_ending).encode("utf-8")
    path.write_bytes(raw)
    return raw


class TestApplyAndRevert:
    """The patch's round-trip behaviour on real files."""

    def test_apply_changes_the_file(self, tmp_path: Path, writer: RecordingWriter) -> None:
        """A patch that changed nothing would report success having done nothing."""
        target = tmp_path / "sensor.py"
        original = _write(target, _SOURCE, "\n")
        main(["apply", str(target)])
        assert target.read_bytes() != original

    def test_apply_routes_the_max_through_the_alias(
        self, tmp_path: Path, writer: RecordingWriter
    ) -> None:
        """The whole point: the atomic_max writes a separately bound array."""
        target = tmp_path / "sensor.py"
        _write(target, _SOURCE, "\n")
        main(["apply", str(target)])
        assert _MAX_NEW in target.read_text(encoding="utf-8")

    def test_round_trips_byte_identically(self, tmp_path: Path, writer: RecordingWriter) -> None:
        """Apply then revert must restore the file exactly.

        A single changed byte changes Warp's module hash, so a lossy revert
        would leave the venv looking canonical while keying a different cache.
        """
        target = tmp_path / "sensor.py"
        original = _write(target, _SOURCE, "\n")
        main(["apply", str(target)])
        main(["revert", str(target)])
        assert target.read_bytes() == original

    def test_preserves_crlf_line_endings(self, tmp_path: Path, writer: RecordingWriter) -> None:
        """A silent EOL rewrite would change the module hash on its own."""
        target = tmp_path / "sensor.py"
        original = _write(target, _SOURCE, "\r\n")
        main(["apply", str(target)])
        main(["revert", str(target)])
        assert target.read_bytes() == original

    def test_keeps_crlf_while_patched(self, tmp_path: Path, writer: RecordingWriter) -> None:
        """The applied file keeps the convention it arrived with."""
        target = tmp_path / "sensor.py"
        _write(target, _SOURCE, "\r\n")
        main(["apply", str(target)])
        assert b"\r\n" in target.read_bytes()

    def test_leaves_no_bare_newlines_in_a_crlf_file(
        self, tmp_path: Path, writer: RecordingWriter
    ) -> None:
        """A mixed-ending file is neither convention and hashes as a third."""
        target = tmp_path / "sensor.py"
        _write(target, _SOURCE, "\r\n")
        main(["apply", str(target)])
        assert target.read_bytes().replace(b"\r\n", b"").count(b"\n") == 0

    def test_reports_what_it_did(self, tmp_path: Path, writer: RecordingWriter) -> None:
        """The operator needs to know which file was touched."""
        target = tmp_path / "sensor.py"
        _write(target, _SOURCE, "\n")
        main(["apply", str(target)])
        assert writer.text == f"apply: OK ({target})\n"

    def test_returns_zero(self, tmp_path: Path, writer: RecordingWriter) -> None:
        """A completed patch exits clean."""
        target = tmp_path / "sensor.py"
        _write(target, _SOURCE, "\n")
        assert main(["apply", str(target)]) == 0


class TestRejections:
    """Command lines and files the script refuses."""

    def test_rejects_an_absent_action(self, writer: RecordingWriter) -> None:
        """Defaulting to apply would patch a venv nobody asked to patch."""
        with pytest.raises(ScriptArgumentError) as caught:
            main([])
        assert caught.value.code == "NP-ARGS-007"

    def test_rejects_an_unknown_action(self, writer: RecordingWriter) -> None:
        """Only apply and revert exist."""
        with pytest.raises(ScriptArgumentError) as caught:
            main(["unpatch"])
        assert caught.value.code == "NP-ARGS-007"

    def test_rejects_a_file_missing_a_site(self, tmp_path: Path, writer: RecordingWriter) -> None:
        """A different vendor revision must not be half-patched."""
        target = tmp_path / "sensor.py"
        _write(target, "prelude\n" + _MAX_OLD + "\n", "\n")
        with pytest.raises(PatchSiteError) as caught:
            main(["apply", str(target)])
        assert caught.value.code == "NP-PATCH-001"

    def test_rejects_a_file_with_a_duplicated_site(
        self, tmp_path: Path, writer: RecordingWriter
    ) -> None:
        """Two occurrences means one would be missed or both corrupted."""
        target = tmp_path / "sensor.py"
        _write(target, _SOURCE + _SIG_OLD + "\n", "\n")
        with pytest.raises(PatchSiteError) as caught:
            main(["apply", str(target)])
        assert caught.value.code == "NP-PATCH-001"

    def test_leaves_the_file_untouched_when_a_site_is_missing(
        self, tmp_path: Path, writer: RecordingWriter
    ) -> None:
        """A refused patch must not have written a partial result."""
        target = tmp_path / "sensor.py"
        original = _write(target, "prelude\n" + _MAX_OLD + "\n", "\n")
        with pytest.raises(PatchSiteError):
            main(["apply", str(target)])
        assert target.read_bytes() == original
