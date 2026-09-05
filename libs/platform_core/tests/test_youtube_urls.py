"""Reading a video id out of a YouTube URL.

Two packages parsed YouTube URLs and each had its own copy. The outcome type
is what let them share the parsing without sharing an error policy, so the
tests below are mostly about the FOUR failure kinds being distinguishable --
that distinction is the whole reason this returns rather than raises.
"""

from __future__ import annotations

import pytest

from platform_core.youtube_urls import (
    VIDEO_ID_PATTERN,
    YOUTUBE_HOSTS,
    canonical_watch_url,
    read_video_id,
)

_ID = "dQw4w9WgXcQ"
"""An eleven-character id, the length YouTube actually uses."""


class TestTheAcceptedShapes:
    @pytest.mark.parametrize(
        "url",
        [
            f"https://www.youtube.com/watch?v={_ID}",
            f"https://youtube.com/watch?v={_ID}",
            f"https://m.youtube.com/watch?v={_ID}",
            f"https://www.youtube.com/shorts/{_ID}",
            f"https://www.youtube.com/live/{_ID}",
            f"https://youtu.be/{_ID}",
            f"https://www.youtu.be/{_ID}",
        ],
    )
    def test_every_shape_yields_the_same_id(self, url: str) -> None:
        assert read_video_id(url) == {"kind": "ok", "video_id": _ID}

    def test_a_url_without_a_scheme_is_accepted(self) -> None:
        """People paste `youtu.be/<id>` without the https, and both packages
        accepted it before this was shared."""
        assert read_video_id(f"youtu.be/{_ID}")["video_id"] == _ID

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert read_video_id(f"  https://youtu.be/{_ID}  ")["video_id"] == _ID

    def test_extra_query_parameters_do_not_disturb_it(self) -> None:
        url = f"https://www.youtube.com/watch?v={_ID}&t=42s&list=PL1"

        assert read_video_id(url)["video_id"] == _ID


class TestTheFourFailures:
    """Each is named, because each earns a different refusal in a caller."""

    def test_an_empty_url_is_empty_not_unparseable(self) -> None:
        assert read_video_id("") == {"kind": "empty", "video_id": None}

    def test_whitespace_only_counts_as_empty(self) -> None:
        assert read_video_id("   ")["kind"] == "empty"

    def test_a_url_the_stdlib_refuses_is_unparseable(self) -> None:
        """A real ValueError from urlsplit, not a simulated one. clubbot had a
        DI hook whose only job was to fake this; an invalid IPv6 bracket
        produces it for real."""
        assert read_video_id("http://[")["kind"] == "unparseable"

    def test_another_video_site_is_not_youtube(self) -> None:
        assert read_video_id(f"https://vimeo.com/{_ID}")["kind"] == "not_youtube"

    def test_a_lookalike_host_is_not_youtube(self) -> None:
        """`youtube.com.evil.test` ends with a YouTube host as a SUBSTRING and
        is not one. The check is on the whole netloc for that reason."""
        assert read_video_id(f"https://youtube.com.evil.test/watch?v={_ID}")["kind"] == (
            "not_youtube"
        )

    def test_a_watch_url_with_no_v_parameter_has_a_bad_id(self) -> None:
        assert read_video_id("https://www.youtube.com/watch")["kind"] == "bad_video_id"

    def test_an_empty_v_parameter_has_a_bad_id(self) -> None:
        assert read_video_id("https://www.youtube.com/watch?v=")["kind"] == "bad_video_id"

    def test_a_ten_character_id_is_refused(self) -> None:
        """Truncated ids are the common paste error and they are exactly the
        length check's job."""
        assert read_video_id("https://youtu.be/dQw4w9WgXc")["kind"] == "bad_video_id"

    def test_an_id_with_a_disallowed_character_is_refused(self) -> None:
        assert read_video_id("https://youtu.be/dQw4w9WgX!Q")["kind"] == "bad_video_id"

    def test_a_bare_youtube_host_has_a_bad_id(self) -> None:
        assert read_video_id("https://www.youtube.com/")["kind"] == "bad_video_id"

    def test_an_unknown_path_shape_has_a_bad_id(self) -> None:
        assert read_video_id(f"https://www.youtube.com/embed/{_ID}")["kind"] == "bad_video_id"

    def test_shorts_with_no_id_after_it_has_a_bad_id(self) -> None:
        assert read_video_id("https://www.youtube.com/shorts")["kind"] == "bad_video_id"

    def test_a_failure_never_carries_a_video_id(self) -> None:
        """A caller reads `video_id is None` to decide whether to refuse, so a
        failure that carried an id would be admitted as a success."""
        failures = ["", "http://[", "https://vimeo.com/x", "https://youtu.be/short"]

        assert [read_video_id(u)["video_id"] for u in failures] == [None, None, None, None]


class TestTheDeclaredVocabulary:
    def test_the_hosts_are_the_five_youtube_serves(self) -> None:
        assert (
            frozenset(
                {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
            )
            == YOUTUBE_HOSTS
        )

    def test_the_id_pattern_is_anchored_at_both_ends(self) -> None:
        """Unanchored, `dQw4w9WgXcQextra` would match and a wrong id would be
        accepted from a malformed path."""
        candidates = [_ID, f"{_ID}extra", f"extra{_ID}", _ID[:10]]

        assert [bool(VIDEO_ID_PATTERN.match(c)) for c in candidates] == [True, False, False, False]


class TestCanonicalWatchUrl:
    def test_it_renders_the_watch_form(self) -> None:
        assert canonical_watch_url(_ID) == f"https://www.youtube.com/watch?v={_ID}"

    def test_its_output_reads_back_as_the_same_id(self) -> None:
        """The round trip is the property worth holding: canonicalising twice
        must not drift."""
        assert read_video_id(canonical_watch_url(_ID))["video_id"] == _ID


__all__ = [
    "TestCanonicalWatchUrl",
    "TestTheAcceptedShapes",
    "TestTheDeclaredVocabulary",
    "TestTheFourFailures",
]
