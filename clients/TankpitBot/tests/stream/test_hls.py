"""HTTP answers for HLS files, against real files on disk."""

from __future__ import annotations

from pathlib import Path

from tankpit_bot.stream.hls import (
    PLAYLIST_CACHE_CONTROL,
    PLAYLIST_CONTENT_TYPE,
    SEGMENT_CACHE_CONTROL,
    SEGMENT_CONTENT_TYPE,
    WARMUP_RETRY_SECONDS,
    hls_web_response,
    read_hls_file,
)


class TestReadHlsFile:
    """Status semantics per filename class."""

    def test_present_playlist_is_served_uncached(self, tmp_path: Path) -> None:
        """The playlist is the live edge; no cache may hold it."""
        (tmp_path / "index.m3u8").write_bytes(b"#EXTM3U\n")
        answer = read_hls_file(tmp_path, "index.m3u8")
        assert answer["status"] == 200
        assert answer["content_type"] == PLAYLIST_CONTENT_TYPE
        assert answer["body"] == b"#EXTM3U\n"
        assert answer["cache_control"] == PLAYLIST_CACHE_CONTROL
        assert answer["retry_after_seconds"] == 0

    def test_absent_playlist_is_warming(self, tmp_path: Path) -> None:
        """Before the encoder's first write, the answer is come-back."""
        answer = read_hls_file(tmp_path, "index.m3u8")
        assert answer["status"] == 503
        assert answer["retry_after_seconds"] == WARMUP_RETRY_SECONDS
        assert b"no playlist yet" in answer["body"]

    def test_present_segment_is_served_immutable(self, tmp_path: Path) -> None:
        """A segment's name is never reused, so caches may keep it."""
        (tmp_path / "seg00042.ts").write_bytes(b"\x47mpegts")
        answer = read_hls_file(tmp_path, "seg00042.ts")
        assert answer["status"] == 200
        assert answer["content_type"] == SEGMENT_CONTENT_TYPE
        assert answer["body"] == b"\x47mpegts"
        assert answer["cache_control"] == SEGMENT_CACHE_CONTROL
        assert answer["retry_after_seconds"] == 0

    def test_rotated_out_segment_is_404(self, tmp_path: Path) -> None:
        """A segment past the live window points back at the playlist."""
        answer = read_hls_file(tmp_path, "seg00001.ts")
        assert answer["status"] == 404
        assert b"no longer in the live window" in answer["body"]

    def test_a_name_outside_the_grammar_is_refused_before_any_read(self, tmp_path: Path) -> None:
        """Only the playlist name and the segment shape reach the disk."""
        (tmp_path / "xvfb.log").write_bytes(b"secret")
        for name in ("xvfb.log", "seg1.ts", "seg000010.ts", "SEG00001.TS", "a/../b"):
            answer = read_hls_file(tmp_path, name)
            assert answer["status"] == 404
            assert answer["body"] == b"no such stream file"


class TestWebResponseAdapter:
    """The one translation into aiohttp."""

    def test_headers_carry_cache_policy(self, tmp_path: Path) -> None:
        """Cache-Control always travels; Retry-After only when non-zero."""
        (tmp_path / "index.m3u8").write_bytes(b"#EXTM3U\n")
        ok = hls_web_response(read_hls_file(tmp_path, "index.m3u8"))
        assert ok.status == 200
        assert ok.headers["Cache-Control"] == PLAYLIST_CACHE_CONTROL
        assert "Retry-After" not in ok.headers

        warming = hls_web_response(read_hls_file(tmp_path / "absent", "index.m3u8"))
        assert warming.status == 503
        assert warming.headers["Retry-After"] == str(WARMUP_RETRY_SECONDS)
