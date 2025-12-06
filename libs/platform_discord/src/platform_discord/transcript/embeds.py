from __future__ import annotations

from platform_discord.discord_types import Embed as _Embed
from platform_discord.embed_helpers import add_field as _add_field
from platform_discord.embed_helpers import create_embed as _create_embed

from .types import TranscriptInfo


def build_transcript_embed(*, info: TranscriptInfo) -> _Embed:
    e = _create_embed(
        title="Transcript Ready",
        description="Your transcript is attached.",
        color=0x57F287,
    )
    _add_field(e, name="Video", value=f"`{info['video_id']}`", inline=True)
    _add_field(e, name="URL", value=f"`{info['url']}`", inline=False)
    chars = info.get("chars")
    if isinstance(chars, int) and chars > 0:
        _add_field(e, name="Characters", value=f"`{chars}`", inline=True)
    return e


def build_transcript_error_embed(*, message: str) -> _Embed:
    e = _create_embed(
        title="Transcript Failed",
        description="Unable to process transcript.",
        color=0xED4245,
    )
    _add_field(e, name="Error", value=f"```{message}```", inline=False)
    return e


__all__ = ["build_transcript_embed", "build_transcript_error_embed"]
