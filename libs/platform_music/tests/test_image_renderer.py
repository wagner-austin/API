from __future__ import annotations

from platform_music import WrappedResult
from platform_music.image_gen import build_renderer


def test_build_renderer_and_render_png() -> None:
    r = build_renderer()
    result: WrappedResult = {
        "service": "lastfm",
        "year": 2024,
        "generated_at": "2024-12-31T00:00:00Z",
        "total_scrobbles": 10,
        "top_artists": [],
        "top_songs": [],
        "top_by_month": [],
    }
    out = r.render_wrapped(result)
    assert bytes(out).startswith(b"\x89PNG\r\n\x1a\n")
