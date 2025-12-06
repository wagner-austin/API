"""Discord type definitions for platform services.

This module re-exports the public Protocol from embed_helpers
and provides additional type protocols for Discord entities.
"""

from __future__ import annotations

from typing import Protocol

from platform_discord.embed_helpers import EmbedProto

# Re-export EmbedProto as Embed for backwards compatibility
Embed = EmbedProto


class File(Protocol):
    """Protocol defining the minimal discord.File interface we depend on."""

    @property
    def filename(self) -> str | None: ...


class User(Protocol):
    """Protocol defining the minimal discord.User interface we depend on."""

    @property
    def id(self) -> int: ...


__all__ = ["Embed", "EmbedProto", "File", "User"]
