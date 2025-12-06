"""Tests for embed_helpers module."""

from __future__ import annotations

import pytest

from platform_discord.embed_helpers import (
    EmbedData,
    EmbedFieldData,
    EmbedProto,
    add_field,
    create_embed,
    get_all_fields,
    get_color_value,
    get_description,
    get_field,
    get_field_value,
    get_footer_text,
    get_title,
    has_field,
    set_footer,
)


def test_create_embed_with_all_params() -> None:
    embed = create_embed(title="Test", description="Desc", color=0x57F287)
    assert get_title(embed) == "Test"
    assert get_description(embed) == "Desc"
    assert get_color_value(embed) == 0x57F287


def test_create_embed_with_none_params() -> None:
    embed = create_embed()
    assert get_title(embed) is None
    assert get_description(embed) is None
    assert get_color_value(embed) is None


def test_add_field() -> None:
    embed = create_embed(title="Test")
    add_field(embed, name="Field1", value="Value1", inline=True)
    add_field(embed, name="Field2", value="Value2", inline=False)
    fields = get_all_fields(embed)
    assert len(fields) == 2
    assert fields[0]["name"] == "Field1"
    assert fields[0]["value"] == "Value1"
    assert fields[0]["inline"] is True
    assert fields[1]["inline"] is False


def test_set_footer() -> None:
    embed = create_embed(title="Test")
    set_footer(embed, text="Footer text")
    assert get_footer_text(embed) == "Footer text"


def test_get_title() -> None:
    embed = create_embed(title="My Title")
    assert get_title(embed) == "My Title"

    embed_no_title = create_embed()
    assert get_title(embed_no_title) is None


def test_get_description() -> None:
    embed = create_embed(description="My Desc")
    assert get_description(embed) == "My Desc"

    embed_no_desc = create_embed()
    assert get_description(embed_no_desc) is None


def test_get_color_value() -> None:
    embed = create_embed(color=0xFF0000)
    assert get_color_value(embed) == 0xFF0000

    embed_no_color = create_embed()
    assert get_color_value(embed_no_color) is None


def test_get_footer_text() -> None:
    embed = create_embed(title="Test")
    set_footer(embed, text="My Footer")
    assert get_footer_text(embed) == "My Footer"


def test_get_footer_text_no_footer() -> None:
    embed = create_embed()
    # Footer exists but text is empty/None
    result = get_footer_text(embed)
    assert result is None or result == ""


def test_get_field() -> None:
    embed = create_embed(title="Test")
    add_field(embed, name="Status", value="Active", inline=True)
    add_field(embed, name="Count", value="42", inline=False)

    field = get_field(embed, "Status")
    if field is None:
        pytest.fail("expected field 'Status'")
    assert field["name"] == "Status"
    assert field["value"] == "Active"
    assert field["inline"] is True

    field2 = get_field(embed, "Count")
    if field2 is None:
        pytest.fail("expected field 'Count'")
    assert field2["inline"] is False

    missing = get_field(embed, "NonExistent")
    assert missing is None


def test_has_field() -> None:
    embed = create_embed(title="Test")
    add_field(embed, name="Exists", value="Yes")

    assert has_field(embed, "Exists") is True
    assert has_field(embed, "Missing") is False


def test_get_all_fields() -> None:
    embed = create_embed(title="Test")
    add_field(embed, name="A", value="1", inline=True)
    add_field(embed, name="B", value="2", inline=False)

    fields = get_all_fields(embed)
    assert len(fields) == 2
    assert fields[0]["name"] == "A"
    assert fields[0]["value"] == "1"
    assert fields[1]["name"] == "B"
    assert fields[1]["value"] == "2"


def test_get_all_fields_empty() -> None:
    embed = create_embed(title="Test")
    fields = get_all_fields(embed)
    assert fields == []


def test_get_field_value() -> None:
    embed = create_embed(title="Test")
    add_field(embed, name="Key", value="MyValue")

    assert get_field_value(embed, "Key") == "MyValue"
    assert get_field_value(embed, "Missing") is None


def test_embed_field_data_typing() -> None:
    field: EmbedFieldData = {"name": "Test", "value": "Val", "inline": True}
    assert field["name"] == "Test"
    assert field["value"] == "Val"
    assert field["inline"] is True


def test_embed_proto_type_alias() -> None:
    # Just verify the type alias is exported and usable
    embed: EmbedProto = create_embed(title="Test")
    assert get_title(embed) == "Test"


def test_embed_adapter_field_count() -> None:
    embed = create_embed(title="Test")
    assert embed.field_count == 0
    add_field(embed, name="A", value="1")
    assert embed.field_count == 1
    add_field(embed, name="B", value="2")
    assert embed.field_count == 2


def test_embed_adapter_properties_directly() -> None:
    embed = create_embed(title="Title", description="Desc", color=0x123456)
    set_footer(embed, text="Footer")

    # Test accessing properties directly on the adapter
    assert embed.title == "Title"
    assert embed.description == "Desc"
    assert embed.color_value == 0x123456
    assert embed.footer_text == "Footer"


def test_embed_adapter_methods_directly() -> None:
    embed = create_embed(title="Test")
    embed.add_field(name="Direct", value="Call", inline=False)
    embed.set_footer(text="Direct Footer")

    assert embed.has_field("Direct") is True
    field = embed.get_field("Direct")
    if field is None:
        pytest.fail("expected field 'Direct'")
    assert field["value"] == "Call"
    assert embed.footer_text == "Direct Footer"


def test_embed_to_dict_full() -> None:
    """Test to_dict with all properties set."""
    embed = create_embed(title="Title", description="Desc", color=0xFF0000)
    add_field(embed, name="Field1", value="Value1", inline=True)
    add_field(embed, name="Field2", value="Value2", inline=False)
    set_footer(embed, text="Footer text")

    result = embed.to_dict()

    assert result["title"] == "Title"
    assert result["description"] == "Desc"
    assert result["color"] == 0xFF0000
    fields = result["fields"]
    if fields is None:
        pytest.fail("expected fields")
    assert len(fields) == 2
    assert fields[0]["name"] == "Field1"
    assert fields[0]["value"] == "Value1"
    assert fields[0]["inline"] is True
    assert fields[1]["name"] == "Field2"
    assert fields[1]["inline"] is False
    footer = result["footer"]
    if footer is None:
        pytest.fail("expected footer")
    assert footer["text"] == "Footer text"


def test_embed_to_dict_minimal() -> None:
    """Test to_dict with minimal properties."""
    embed = create_embed()
    result = embed.to_dict()

    # Empty embed should have empty dict or only set fields
    assert "title" not in result or result.get("title") is None
    assert "description" not in result or result.get("description") is None
    assert "color" not in result or result.get("color") is None
    assert "fields" not in result or result.get("fields") == []
    assert "footer" not in result or result.get("footer") is None


def test_embed_to_dict_partial() -> None:
    """Test to_dict with only some properties set."""
    embed = create_embed(title="Only Title")
    result = embed.to_dict()

    assert result["title"] == "Only Title"
    assert "description" not in result or result.get("description") is None


def test_unwrap_embed_returns_inner() -> None:
    """Test that unwrap_embed returns the underlying discord.Embed."""
    from platform_discord.embed_helpers import unwrap_embed

    embed = create_embed(title="Test Title", color=0xFF0000)
    inner = unwrap_embed(embed)

    # The inner should be the actual discord.Embed - verify properties work
    assert inner.title == "Test Title"
    # Verify it can add fields (method exists on discord.Embed)
    inner.add_field(name="Test", value="Value")
    assert len(inner.fields) == 1


class _NonAdapterEmbed:
    """A fake embed that doesn't use our adapter."""

    @property
    def title(self) -> str | None:
        return "Fake"

    @property
    def description(self) -> str | None:
        return None

    @property
    def color_value(self) -> int | None:
        return None

    @property
    def footer_text(self) -> str | None:
        return None

    @property
    def field_count(self) -> int:
        return 0

    def add_field(self, *, name: str, value: str, inline: bool = True) -> None:
        pass

    def set_footer(self, *, text: str) -> None:
        pass

    def get_field(self, name: str) -> EmbedFieldData | None:
        return None

    def has_field(self, name: str) -> bool:
        return False

    def get_all_fields(self) -> list[EmbedFieldData]:
        return []

    def get_field_value(self, name: str) -> str | None:
        return None

    def to_dict(self) -> EmbedData:
        return {}


def test_unwrap_embed_raises_for_non_adapter() -> None:
    """Test that unwrap_embed raises TypeError for non-adapter embeds."""
    from platform_discord.embed_helpers import unwrap_embed

    fake: EmbedProto = _NonAdapterEmbed()
    with pytest.raises(TypeError, match="Expected embed created via create_embed"):
        unwrap_embed(fake)
