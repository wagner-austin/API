"""Tests for discord_types module."""

from __future__ import annotations

from platform_discord.discord_types import Embed, EmbedProto, File, User
from platform_discord.embed_helpers import create_embed


def test_embed_alias_is_embed_proto() -> None:
    """Verify Embed is an alias for EmbedProto."""
    assert Embed is EmbedProto


def test_embed_type_with_create_embed() -> None:
    """Verify create_embed returns something compatible with Embed type."""
    embed: Embed = create_embed(title="Test")
    assert embed.title == "Test"


class _FakeFile:
    @property
    def filename(self) -> str | None:
        return "test.txt"


class _FakeUser:
    @property
    def id(self) -> int:
        return 12345


def test_file_protocol_structure() -> None:
    """Verify File Protocol can be implemented."""
    file_obj: File = _FakeFile()
    assert file_obj.filename == "test.txt"


def test_user_protocol_structure() -> None:
    """Verify User Protocol can be implemented."""
    user_obj: User = _FakeUser()
    assert user_obj.id == 12345
