"""Typed helpers for Discord embed operations.

This module provides strictly typed functions for creating and accessing
Discord embed properties, avoiding the Any types in discord.py.

Follows the same adapter pattern as platform_workers/redis.py:
- Internal Protocols for discord.py types
- Public Protocols for our interface
- Adapter classes that wrap discord objects
- Factory functions that return Protocol types

Usage:
    from platform_discord.embed_helpers import (
        create_embed,
        get_field,
        get_footer_text,
        get_color_value,
        EmbedFieldData,
    )

    # Create embed with typed parameters
    embed = create_embed(title="Hello", description="World", color=0x57F287)
    add_field(embed, name="Status", value="Active", inline=True)
    set_footer(embed, text="Request ID: abc123")

    # Read properties with proper typing
    field = get_field(embed, "Status")
    if field is not None:
        value = field["value"]  # Typed as str
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class EmbedFieldData(TypedDict):
    """Typed dict for embed field data."""

    name: str
    value: str
    inline: bool


class EmbedFooterData(TypedDict):
    """Typed dict for embed footer data."""

    text: str
    icon_url: str | None


class EmbedAuthorData(TypedDict):
    """Typed dict for embed author data."""

    name: str
    icon_url: str | None
    url: str | None


class EmbedData(TypedDict, total=False):
    """Typed dict representing full embed data."""

    title: str | None
    description: str | None
    color: int | None
    fields: list[EmbedFieldData]
    footer: EmbedFooterData | None
    author: EmbedAuthorData | None


# Internal Protocols for discord.py types (not exposed publicly)
class _DiscordFieldProxy(Protocol):
    """Protocol for discord embed field proxy."""

    @property
    def name(self) -> str | None: ...

    @property
    def value(self) -> str | None: ...

    @property
    def inline(self) -> bool: ...


class _DiscordFooterProxy(Protocol):
    """Protocol for discord embed footer proxy."""

    @property
    def text(self) -> str | None: ...


class _DiscordColorValue(Protocol):
    """Protocol for discord Color value property."""

    @property
    def value(self) -> int: ...


class _DiscordEmbedClient(Protocol):
    """Protocol for discord.Embed internal client."""

    @property
    def title(self) -> str | None: ...

    @property
    def description(self) -> str | None: ...

    @property
    def color(self) -> _DiscordColorValue | None: ...

    @property
    def footer(self) -> _DiscordFooterProxy: ...

    @property
    def fields(self) -> list[_DiscordFieldProxy]: ...

    def add_field(self, *, name: str, value: str, inline: bool = True) -> _DiscordEmbedClient: ...

    def set_footer(self, *, text: str) -> _DiscordEmbedClient: ...


class _DiscordColorCtor(Protocol):
    """Protocol for discord.Color constructor."""

    def __call__(self, value: int) -> _DiscordColorValue: ...


class _DiscordEmbedCtor(Protocol):
    """Protocol for discord.Embed constructor."""

    def __call__(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        color: _DiscordColorValue | None = None,
    ) -> _DiscordEmbedClient: ...


class _DiscordModule(Protocol):
    """Protocol for discord module."""

    Embed: _DiscordEmbedCtor
    Color: _DiscordColorCtor


def _load_discord_module() -> _DiscordModule:
    """Load discord module dynamically."""
    mod: _DiscordModule = __import__("discord")
    return mod


# Public Protocol - this is what we expose
@runtime_checkable
class EmbedProto(Protocol):
    """Public protocol for Discord embeds used by platform services."""

    @property
    def title(self) -> str | None: ...

    @property
    def description(self) -> str | None: ...

    @property
    def color_value(self) -> int | None: ...

    @property
    def footer_text(self) -> str | None: ...

    @property
    def field_count(self) -> int: ...

    def add_field(self, *, name: str, value: str, inline: bool = True) -> None: ...

    def set_footer(self, *, text: str) -> None: ...

    def get_field(self, name: str) -> EmbedFieldData | None: ...

    def has_field(self, name: str) -> bool: ...

    def get_all_fields(self) -> list[EmbedFieldData]: ...

    def get_field_value(self, name: str) -> str | None: ...

    def to_dict(self) -> EmbedData: ...


class _EmbedAdapter(EmbedProto):
    """Adapter that wraps discord.Embed and exposes our strict interface."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _DiscordEmbedClient) -> None:
        self._inner = inner

    @property
    def title(self) -> str | None:
        raw = self._inner.title
        return str(raw) if raw is not None else None

    @property
    def description(self) -> str | None:
        raw = self._inner.description
        return str(raw) if raw is not None else None

    @property
    def color_value(self) -> int | None:
        color = self._inner.color
        if color is None:
            return None
        return int(color.value)

    @property
    def footer_text(self) -> str | None:
        footer = self._inner.footer
        text = footer.text
        return str(text) if text else None

    @property
    def field_count(self) -> int:
        return len(self._inner.fields)

    def add_field(self, *, name: str, value: str, inline: bool = True) -> None:
        self._inner.add_field(name=name, value=value, inline=inline)

    def set_footer(self, *, text: str) -> None:
        self._inner.set_footer(text=text)

    def get_field(self, name: str) -> EmbedFieldData | None:
        for field in self._inner.fields:
            field_name = field.name
            if field_name == name:
                field_data: EmbedFieldData = {
                    "name": str(field_name) if field_name else "",
                    "value": str(field.value) if field.value else "",
                    "inline": bool(field.inline),
                }
                return field_data
        return None

    def has_field(self, name: str) -> bool:
        return self.get_field(name) is not None

    def get_all_fields(self) -> list[EmbedFieldData]:
        result: list[EmbedFieldData] = []
        for field in self._inner.fields:
            field_data: EmbedFieldData = {
                "name": str(field.name) if field.name else "",
                "value": str(field.value) if field.value else "",
                "inline": bool(field.inline),
            }
            result.append(field_data)
        return result

    def get_field_value(self, name: str) -> str | None:
        field = self.get_field(name)
        return field["value"] if field is not None else None

    def to_dict(self) -> EmbedData:
        """Return dict representation for discord.py API compatibility.

        Discord.py calls to_dict() when serializing embeds for the API.
        We build the dict from our typed properties to avoid Any types.
        """
        result: EmbedData = {}
        title = self.title
        if title is not None:
            result["title"] = title
        desc = self.description
        if desc is not None:
            result["description"] = desc
        color = self.color_value
        if color is not None:
            result["color"] = color
        fields = self.get_all_fields()
        if fields:
            result["fields"] = fields
        footer_text = self.footer_text
        if footer_text:
            result["footer"] = {"text": footer_text, "icon_url": None}
        return result


def create_embed(
    *,
    title: str | None = None,
    description: str | None = None,
    color: int | None = None,
) -> EmbedProto:
    """Create a Discord embed with typed parameters.

    Args:
        title: The embed title.
        description: The embed description.
        color: The embed color as an integer (e.g., 0x57F287 for green).

    Returns:
        An EmbedProto instance wrapping the discord.Embed.
    """
    from .testing import hooks

    discord_mod = hooks.load_discord_module()
    color_obj = discord_mod.Color(color) if color is not None else None
    inner = discord_mod.Embed(title=title, description=description, color=color_obj)
    return _EmbedAdapter(inner)


# Convenience functions that work with EmbedProto
def add_field(embed: EmbedProto, *, name: str, value: str, inline: bool = True) -> None:
    """Add a field to an embed.

    Args:
        embed: The embed to add the field to.
        name: The field name.
        value: The field value.
        inline: Whether the field should be inline.
    """
    embed.add_field(name=name, value=value, inline=inline)


def set_footer(embed: EmbedProto, *, text: str) -> None:
    """Set the embed footer.

    Args:
        embed: The embed to set the footer on.
        text: The footer text.
    """
    embed.set_footer(text=text)


def get_title(embed: EmbedProto) -> str | None:
    """Get the embed title."""
    return embed.title


def get_description(embed: EmbedProto) -> str | None:
    """Get the embed description."""
    return embed.description


def get_color_value(embed: EmbedProto) -> int | None:
    """Get the embed color value as an integer."""
    return embed.color_value


def get_footer_text(embed: EmbedProto) -> str | None:
    """Get the embed footer text."""
    return embed.footer_text


def get_field(embed: EmbedProto, name: str) -> EmbedFieldData | None:
    """Get a field by name."""
    return embed.get_field(name)


def has_field(embed: EmbedProto, name: str) -> bool:
    """Check if the embed has a field with the given name."""
    return embed.has_field(name)


def get_all_fields(embed: EmbedProto) -> list[EmbedFieldData]:
    """Get all fields from the embed."""
    return embed.get_all_fields()


def get_field_value(embed: EmbedProto, name: str) -> str | None:
    """Get a field's value by name."""
    return embed.get_field_value(name)


def unwrap_embed(embed: EmbedProto) -> _DiscordEmbedClient:
    """Extract the underlying discord.Embed from an EmbedProto adapter.

    Use this when you need to pass an embed to discord.py APIs that require
    the concrete discord.Embed type (e.g., interaction.followup.send).

    Args:
        embed: An EmbedProto instance (typically from create_embed()).

    Returns:
        The underlying discord.Embed object.

    Raises:
        TypeError: If the embed is not from our adapter.
    """
    if isinstance(embed, _EmbedAdapter):
        return embed._inner
    msg = "Expected embed created via create_embed(), got external embed"
    raise TypeError(msg)


__all__ = [
    "EmbedAuthorData",
    "EmbedData",
    "EmbedFieldData",
    "EmbedFooterData",
    "EmbedProto",
    "add_field",
    "create_embed",
    "get_all_fields",
    "get_color_value",
    "get_description",
    "get_field",
    "get_field_value",
    "get_footer_text",
    "get_title",
    "has_field",
    "set_footer",
    "unwrap_embed",
]
