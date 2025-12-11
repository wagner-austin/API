"""Tests for production client factory paths.

These tests verify that the default hooks (production implementations) work correctly.
The hooks are initialized to production implementations at module load, so calling
the internal _get_* functions with default hooks exercises the real factory functions.
"""

from __future__ import annotations

from platform_music.jobs import (
    _get_apple_client,
    _get_lastfm_client,
    _get_spotify_client,
    _get_youtube_client,
)


def test_get_lastfm_client_production_path() -> None:
    """Verify production lastfm_client hook creates the stub (no real Last.fm impl)."""
    client = _get_lastfm_client(api_key="k", api_secret="s", session_key="sk")
    # Last.fm uses a stub since there's no real implementation in this lib
    assert client.__class__.__name__ == "_Stub"


def test_get_spotify_client_production_path() -> None:
    """Verify production spotify_client hook creates _SpotifyClient."""
    client = _get_spotify_client(access_token="at", refresh_token="rt", expires_in=3600)
    assert client.__class__.__name__ == "_SpotifyClient"


def test_get_apple_client_production_path() -> None:
    """Verify production apple_client hook creates _AppleClient."""
    client = _get_apple_client(music_user_token="ut", developer_token="dt")
    assert client.__class__.__name__ == "_AppleClient"


def test_get_youtube_client_production_path() -> None:
    """Verify production youtube_client hook creates _YouTubeClient."""
    client = _get_youtube_client(sapisid="sid", cookies="c=1")
    assert client.__class__.__name__ == "_YouTubeClient"
