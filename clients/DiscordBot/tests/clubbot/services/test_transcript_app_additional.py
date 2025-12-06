from __future__ import annotations

import pytest

from clubbot.services.transcript.client import TranscriptResult, _parse_langs


def test_transcriptresult_getitem_all_keys_and_keyerror() -> None:
    r = TranscriptResult(url="https://v", video_id="vid", text="hello")
    assert r["url"] == "https://v"
    assert r["video_id"] == "vid"
    assert r["text"] == "hello"
    with pytest.raises(KeyError):
        _ = r["nope"]


def test_parse_langs_defaults_when_empty() -> None:
    out = _parse_langs("   ")
    # Expect non-empty default list and contains en
    assert isinstance(out, list) and "en" in out
