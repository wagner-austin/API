from __future__ import annotations

from platform_discord.discord_types import Embed as _Embed
from platform_discord.embed_helpers import add_field as _add_field
from platform_discord.embed_helpers import create_embed as _create_embed


def build_qr_embed(*, url: str) -> _Embed:
    e = _create_embed(
        title="QR Code Generated",
        description="Your QR code is ready.",
        color=0x57F287,
    )
    _add_field(e, name="URL", value=f"`{url}`", inline=False)
    return e


def build_qr_error_embed(*, message: str) -> _Embed:
    e = _create_embed(
        title="QR Generation Failed",
        description="Unable to create QR code.",
        color=0xED4245,
    )
    _add_field(e, name="Error", value=f"```{message}```", inline=False)
    return e


__all__ = ["build_qr_embed", "build_qr_error_embed"]
