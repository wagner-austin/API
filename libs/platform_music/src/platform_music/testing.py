from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from platform_workers.redis import RedisStrProto, redis_for_kv

from platform_music.models import PlayRecord, ServiceName, Track
from platform_music.services.apple import AppleHttpGetHook, AppleMusicProto
from platform_music.services.lastfm import LastFmProto
from platform_music.services.spotify import SpotifyHttpGetHook, SpotifyProto
from platform_music.services.youtube import YouTubeHttpPostHook, YouTubeMusicProto

# =============================================================================
# Fake Music Service Implementations
# =============================================================================


class FakeLastFm(LastFmProto):
    """Simple in-memory Last.fm fake for testing."""

    def __init__(self) -> None:
        self._plays: list[PlayRecord] = []

    def add_play(
        self,
        *,
        track_id: str,
        title: str,
        artist_name: str,
        played_at: str,
    ) -> None:
        track: Track = {
            "id": track_id,
            "title": title,
            "artist_name": artist_name,
            "duration_ms": 0,
            "service": "lastfm",
        }
        self._plays.append({"track": track, "played_at": played_at, "service": "lastfm"})

    def get_listening_history(
        self,
        *,
        start_date: str,
        end_date: str,
        limit: int | None = None,
    ) -> list[PlayRecord]:
        # ISO 8601 string compare is valid for UTC timestamps
        filtered = [p for p in self._plays if start_date <= p["played_at"] <= end_date]
        if limit is not None:
            filtered = filtered[:limit]
        return list(filtered)


class FakeSpotify(SpotifyProto):
    """Simple in-memory Spotify fake for testing."""

    def __init__(self) -> None:
        self._plays: list[PlayRecord] = []

    def add_play(
        self,
        *,
        track_id: str,
        title: str,
        artist_name: str,
        played_at: str,
        duration_ms: int = 1000,
    ) -> None:
        track: Track = {
            "id": track_id,
            "title": title,
            "artist_name": artist_name,
            "duration_ms": duration_ms,
            "service": "spotify",
        }
        self._plays.append({"track": track, "played_at": played_at, "service": "spotify"})

    def get_listening_history(
        self,
        *,
        start_date: str,
        end_date: str,
        limit: int | None = None,
    ) -> list[PlayRecord]:
        filtered = [p for p in self._plays if start_date <= p["played_at"] <= end_date]
        if limit is not None:
            filtered = filtered[:limit]
        return list(filtered)


class FakeAppleMusic(AppleMusicProto):
    """Simple in-memory Apple Music fake for testing."""

    def __init__(self) -> None:
        self._plays: list[PlayRecord] = []

    def add_play(
        self,
        *,
        track_id: str,
        title: str,
        artist_name: str,
        played_at: str,
        duration_ms: int = 1000,
    ) -> None:
        track: Track = {
            "id": track_id,
            "title": title,
            "artist_name": artist_name,
            "duration_ms": duration_ms,
            "service": "apple_music",
        }
        self._plays.append({"track": track, "played_at": played_at, "service": "apple_music"})

    def get_listening_history(
        self,
        *,
        start_date: str,
        end_date: str,
        limit: int | None = None,
    ) -> list[PlayRecord]:
        filtered = [p for p in self._plays if start_date <= p["played_at"] <= end_date]
        if limit is not None:
            filtered = filtered[:limit]
        return list(filtered)


class FakeYouTubeMusic(YouTubeMusicProto):
    """Simple in-memory YouTube Music fake for testing."""

    def __init__(self) -> None:
        self._plays: list[PlayRecord] = []

    def add_play(
        self,
        *,
        track_id: str,
        title: str,
        artist_name: str,
        played_at: str,
        duration_ms: int = 1000,
    ) -> None:
        track: Track = {
            "id": track_id,
            "title": title,
            "artist_name": artist_name,
            "duration_ms": duration_ms,
            "service": "youtube_music",
        }
        self._plays.append({"track": track, "played_at": played_at, "service": "youtube_music"})

    def get_listening_history(
        self,
        *,
        start_date: str,
        end_date: str,
        limit: int | None = None,
    ) -> list[PlayRecord]:
        filtered = [p for p in self._plays if start_date <= p["played_at"] <= end_date]
        if limit is not None:
            filtered = filtered[:limit]
        return list(filtered)


# =============================================================================
# Hook Type Definitions
# =============================================================================

# jobs.py hooks
RedisClientHook = Callable[[str], RedisStrProto]
LastFmClientHook = Callable[[str, str, str], LastFmProto]
SpotifyClientHook = Callable[[str, str, int | str], SpotifyProto]
AppleClientHook = Callable[[str, str], AppleMusicProto]
YouTubeClientHook = Callable[[str, str], YouTubeMusicProto]

# Guard script hooks
RunForProjectProto = Callable[[Path, Path], int]  # Positional args version
LoadOrchestratorHook = Callable[[Path], RunForProjectProto]


# =============================================================================
# Hooks Container
# =============================================================================


class HooksContainer:
    """Container for hooks. Production sets defaults, tests override with fakes."""

    # jobs.py dependencies - set to production implementations at module load
    redis_client: RedisClientHook
    lastfm_client: LastFmClientHook
    spotify_client: SpotifyClientHook
    apple_client: AppleClientHook
    youtube_client: YouTubeClientHook

    # HTTP layer hooks for adapters (low-level) - set to production implementations
    apple_http_get: AppleHttpGetHook
    spotify_http_get: SpotifyHttpGetHook
    youtube_http_post: YouTubeHttpPostHook

    # Guard script hook - optional, only used by guard script
    load_orchestrator: LoadOrchestratorHook | None = None


hooks = HooksContainer()


# =============================================================================
# Production Implementations (set as defaults)
# =============================================================================


def _prod_redis_client(url: str) -> RedisStrProto:
    """Production redis client factory."""
    return redis_for_kv(url)


def _prod_lastfm_client(api_key: str, api_secret: str, session_key: str) -> LastFmProto:
    """Production lastfm client factory."""
    from platform_music.services.lastfm import lastfm_client as _lastfm_client

    return _lastfm_client(api_key=api_key, api_secret=api_secret, session_key=session_key)


def _prod_spotify_client(
    access_token: str, refresh_token: str, expires_in: int | str
) -> SpotifyProto:
    """Production spotify client factory."""
    from platform_music.services.spotify import spotify_client as _spotify_client

    return _spotify_client(
        access_token=access_token, refresh_token=refresh_token, expires_in=expires_in
    )


def _prod_apple_client(music_user_token: str, developer_token: str) -> AppleMusicProto:
    """Production apple client factory."""
    from platform_music.services.apple import apple_client as _apple_client

    return _apple_client(music_user_token=music_user_token, developer_token=developer_token)


def _prod_youtube_client(sapisid: str, cookies: str) -> YouTubeMusicProto:
    """Production youtube client factory."""
    from platform_music.services.youtube import youtube_client as _youtube_client

    return _youtube_client(sapisid=sapisid, cookies=cookies)


def _prod_spotify_http_get(url: str, access_token: str, timeout: float) -> str:
    """Production HTTP GET for Spotify."""
    from platform_music.services.spotify import http_get_impl

    return http_get_impl(url, access_token=access_token, timeout=timeout)


def _prod_apple_http_get(url: str, developer_token: str, user_token: str, timeout: float) -> str:
    """Production HTTP GET for Apple Music."""
    from platform_music.services.apple import http_get_impl

    return http_get_impl(
        url, developer_token=developer_token, user_token=user_token, timeout=timeout
    )


def _prod_youtube_http_post(
    url: str, sapisid: str, cookies: str, origin: str, timeout: float, body: str
) -> str:
    """Production HTTP POST for YouTube Music."""
    from platform_music.services.youtube import http_post_impl

    return http_post_impl(
        url, sapisid=sapisid, cookies=cookies, origin=origin, timeout=timeout, body=body
    )


def _init_production_hooks() -> None:
    """Initialize hooks with production implementations. Called at module load."""
    hooks.redis_client = _prod_redis_client
    hooks.lastfm_client = _prod_lastfm_client
    hooks.spotify_client = _prod_spotify_client
    hooks.apple_client = _prod_apple_client
    hooks.youtube_client = _prod_youtube_client
    hooks.spotify_http_get = _prod_spotify_http_get
    hooks.apple_http_get = _prod_apple_http_get
    hooks.youtube_http_post = _prod_youtube_http_post
    hooks.load_orchestrator = None


# Initialize production hooks at module load
_init_production_hooks()


def reset_hooks() -> None:
    """Reset all hooks to production defaults. Call in test teardown."""
    _init_production_hooks()


# =============================================================================
# Factory Helpers for Tests
# =============================================================================


def make_plays(service: ServiceName, count: int = 12) -> list[PlayRecord]:
    """Create a list of fake play records for testing."""
    out: list[PlayRecord] = []
    for i in range(count):
        out.append(
            {
                "track": {
                    "id": f"{service}:t{i}",
                    "title": f"Song{i}",
                    "artist_name": f"Artist{i % 3}",
                    "duration_ms": 1000,
                    "service": service,
                },
                "played_at": f"2024-{(i % 12) + 1:02d}-01T00:00:00Z",
                "service": service,
            }
        )
    return out


def make_fake_redis_client(
    fake_redis: RedisStrProto,
) -> RedisClientHook:
    """Create a redis client hook that returns the provided fake."""

    def _hook(url: str) -> RedisStrProto:
        return fake_redis

    return _hook


def make_fake_lastfm_client(
    fake_lastfm: LastFmProto,
) -> LastFmClientHook:
    """Create a lastfm client hook that returns the provided fake."""

    def _hook(api_key: str, api_secret: str, session_key: str) -> LastFmProto:
        return fake_lastfm

    return _hook


def make_fake_spotify_client(
    fake_spotify: SpotifyProto,
) -> SpotifyClientHook:
    """Create a spotify client hook that returns the provided fake."""

    def _hook(access_token: str, refresh_token: str, expires_in: int | str) -> SpotifyProto:
        return fake_spotify

    return _hook


def make_fake_apple_client(
    fake_apple: AppleMusicProto,
) -> AppleClientHook:
    """Create an apple client hook that returns the provided fake."""

    def _hook(music_user_token: str, developer_token: str) -> AppleMusicProto:
        return fake_apple

    return _hook


def make_fake_youtube_client(
    fake_youtube: YouTubeMusicProto,
) -> YouTubeClientHook:
    """Create a youtube client hook that returns the provided fake."""

    def _hook(sapisid: str, cookies: str) -> YouTubeMusicProto:
        return fake_youtube

    return _hook


def make_fake_apple_http_get(response_json: str) -> AppleHttpGetHook:
    """Create an apple http_get hook that returns the provided JSON response."""

    def _hook(url: str, developer_token: str, user_token: str, timeout: float) -> str:
        return response_json

    return _hook


def make_fake_spotify_http_get(response_json: str) -> SpotifyHttpGetHook:
    """Create a spotify http_get hook that returns the provided JSON response."""

    def _hook(url: str, access_token: str, timeout: float) -> str:
        return response_json

    return _hook


def make_fake_spotify_http_get_pages(pages: list[str]) -> SpotifyHttpGetHook:
    """Create a spotify http_get hook that returns pages in sequence.

    After all pages are exhausted, returns the last page repeatedly.
    """
    state = {"index": 0}

    def _hook(url: str, access_token: str, timeout: float) -> str:
        idx = state["index"]
        result = pages[idx]
        if idx < len(pages) - 1:
            state["index"] = idx + 1
        return result

    return _hook


def make_fake_youtube_http_post(response_json: str) -> YouTubeHttpPostHook:
    """Create a youtube http_post hook that returns the provided JSON response."""

    def _hook(url: str, sapisid: str, cookies: str, origin: str, timeout: float, body: str) -> str:
        return response_json

    return _hook


def make_raising_apple_http_get(exc: BaseException) -> AppleHttpGetHook:
    """Create an apple http_get hook that raises the provided exception."""

    def _hook(url: str, developer_token: str, user_token: str, timeout: float) -> str:
        raise exc

    return _hook


def make_raising_spotify_http_get(exc: BaseException) -> SpotifyHttpGetHook:
    """Create a spotify http_get hook that raises the provided exception."""

    def _hook(url: str, access_token: str, timeout: float) -> str:
        raise exc

    return _hook


def make_raising_youtube_http_post(exc: BaseException) -> YouTubeHttpPostHook:
    """Create a youtube http_post hook that raises the provided exception."""

    def _hook(url: str, sapisid: str, cookies: str, origin: str, timeout: float, body: str) -> str:
        raise exc

    return _hook


def make_fake_load_orchestrator(exit_code: int = 0) -> LoadOrchestratorHook:
    """Create a load_orchestrator hook that returns a fake runner."""

    def _loader(monorepo_root: Path) -> RunForProjectProto:
        def _runner(monorepo_root: Path, project_root: Path) -> int:
            return exit_code

        return _runner

    return _loader


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AppleClientHook",
    "FakeAppleMusic",
    "FakeLastFm",
    "FakeSpotify",
    "FakeYouTubeMusic",
    "HooksContainer",
    "LastFmClientHook",
    "LoadOrchestratorHook",
    "RedisClientHook",
    "RunForProjectProto",
    "SpotifyClientHook",
    "YouTubeClientHook",
    "hooks",
    "make_fake_apple_client",
    "make_fake_apple_http_get",
    "make_fake_lastfm_client",
    "make_fake_load_orchestrator",
    "make_fake_redis_client",
    "make_fake_spotify_client",
    "make_fake_spotify_http_get",
    "make_fake_spotify_http_get_pages",
    "make_fake_youtube_client",
    "make_fake_youtube_http_post",
    "make_plays",
    "make_raising_apple_http_get",
    "make_raising_spotify_http_get",
    "make_raising_youtube_http_post",
    "reset_hooks",
]
