from __future__ import annotations

from platform_discord.transcript.embeds import build_transcript_embed, build_transcript_error_embed
from platform_discord.transcript.types import TranscriptInfo


def test_transcript_embeds() -> None:
    info: TranscriptInfo = {"url": "https://y", "video_id": "vid", "chars": 123}
    e = build_transcript_embed(info=info)
    d = e.to_dict()
    assert d.get("title") == "Transcript Ready"
    # Cover branch when chars is missing or zero
    info2: TranscriptInfo = {"url": "https://y", "video_id": "vid"}
    e_no = build_transcript_embed(info=info2)
    d_no = e_no.to_dict()
    names = [f["name"] for f in d_no.get("fields", [])]
    assert "Characters" not in names
    e2 = build_transcript_error_embed(message="boom")
    d2 = e2.to_dict()
    assert d2.get("title") == "Transcript Failed"
