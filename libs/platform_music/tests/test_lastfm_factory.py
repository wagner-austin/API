from __future__ import annotations

import pytest

from platform_music.services.lastfm import lastfm_client


def test_lastfm_client_stub_raises() -> None:
    client = lastfm_client(api_key="a", api_secret="b", session_key="c")
    with pytest.raises(NotImplementedError):
        client.get_listening_history(
            start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
        )
