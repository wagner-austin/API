"""Tests for probe constants."""

from __future__ import annotations

from tankpit_bot.probe import DEFAULT_MOUSE_POSITIONS, DEFAULT_PROBE_KEYS


def test_default_probe_keys_contains_known_commands() -> None:
    """Test DEFAULT_PROBE_KEYS contains keys with known command mappings."""
    # Only keys with known command IDs are included
    assert "s" in DEFAULT_PROBE_KEYS  # Radar (CMD_RADAR = 102)
    assert "d" in DEFAULT_PROBE_KEYS  # Mine (CMD_MINE = 107)
    assert "f" in DEFAULT_PROBE_KEYS  # Map open (CMD_MAP_OPEN = 108)
    assert "q" in DEFAULT_PROBE_KEYS  # Quit (plain command '-')
    # Keys without known commands are NOT included
    assert "w" not in DEFAULT_PROBE_KEYS  # Unknown
    assert " " not in DEFAULT_PROBE_KEYS  # Unknown
    # Exact expected list (matches test_map_command.py order)
    assert DEFAULT_PROBE_KEYS == ["f", "f", "s", "d", "q"]


def test_default_mouse_positions_is_empty() -> None:
    """Test DEFAULT_MOUSE_POSITIONS is empty (no mouse probing by default)."""
    assert len(DEFAULT_MOUSE_POSITIONS) == 0
