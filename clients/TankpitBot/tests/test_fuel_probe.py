"""Tests for fuel_probe module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tankpit_bot.fuel_probe import (
    FuelProber,
    probe_all,
    probe_common_paths,
    probe_dom_bars,
    probe_game_variables,
    probe_numeric_globals,
)


@pytest.fixture
def mock_cdp() -> MagicMock:
    """Create a mock CDP session."""
    return MagicMock()


def make_cdp_result(value: Any) -> dict[str, Any]:
    """Create a CDP result wrapper."""
    return {"result": {"value": value}}


class TestProbeDomBars:
    """Tests for probe_dom_bars function."""

    def test_returns_empty_list_when_no_bars(self, mock_cdp: MagicMock) -> None:
        """Returns empty list when no bars found."""
        mock_cdp.send.return_value = make_cdp_result([])
        result = probe_dom_bars(mock_cdp)
        assert result == []

    def test_parses_bar_elements(self, mock_cdp: MagicMock) -> None:
        """Parses bar elements correctly."""
        mock_cdp.send.return_value = make_cdp_result([
            {
                "tag": "DIV",
                "id": "fuel-bar",
                "class_name": "progress-bar",
                "width": "75%",
                "computed_width": "150px",
                "parent_class": "bar-container",
            }
        ])
        result = probe_dom_bars(mock_cdp)
        assert len(result) == 1
        assert result[0]["tag"] == "DIV"
        assert result[0]["id"] == "fuel-bar"
        assert result[0]["width"] == "75%"

    def test_handles_invalid_result(self, mock_cdp: MagicMock) -> None:
        """Handles invalid CDP result gracefully."""
        mock_cdp.send.return_value = {"result": None}
        result = probe_dom_bars(mock_cdp)
        assert result == []


class TestProbeGameVariables:
    """Tests for probe_game_variables function."""

    def test_returns_empty_list_when_no_vars(self, mock_cdp: MagicMock) -> None:
        """Returns empty list when no variables found."""
        mock_cdp.send.return_value = make_cdp_result([])
        result = probe_game_variables(mock_cdp)
        assert result == []

    def test_parses_game_variables(self, mock_cdp: MagicMock) -> None:
        """Parses game variables correctly."""
        mock_cdp.send.return_value = make_cdp_result([
            {"name": "fuel", "value": 850, "path": "game.player.fuel"},
            {"name": "hp", "value": 100, "path": "player.hp"},
        ])
        result = probe_game_variables(mock_cdp)
        assert len(result) == 2
        assert result[0]["name"] == "fuel"
        assert result[0]["value"] == 850
        assert result[0]["path"] == "game.player.fuel"


class TestProbeNumericGlobals:
    """Tests for probe_numeric_globals function."""

    def test_returns_empty_list_when_no_globals(self, mock_cdp: MagicMock) -> None:
        """Returns empty list when no numeric globals."""
        mock_cdp.send.return_value = make_cdp_result([])
        result = probe_numeric_globals(mock_cdp)
        assert result == []

    def test_parses_numeric_tuples(self, mock_cdp: MagicMock) -> None:
        """Parses numeric global tuples."""
        mock_cdp.send.return_value = make_cdp_result([
            ["fuelLevel", 750],
            ["maxFuel", 1000],
        ])
        result = probe_numeric_globals(mock_cdp)
        assert len(result) == 2
        assert result[0] == ("fuelLevel", 750.0)
        assert result[1] == ("maxFuel", 1000.0)


class TestProbeCommonPaths:
    """Tests for probe_common_paths function."""

    def test_returns_empty_list_when_no_paths(self, mock_cdp: MagicMock) -> None:
        """Returns empty list when no paths found."""
        mock_cdp.send.return_value = make_cdp_result([])
        result = probe_common_paths(mock_cdp)
        assert result == []

    def test_parses_path_values(self, mock_cdp: MagicMock) -> None:
        """Parses path-value pairs."""
        mock_cdp.send.return_value = make_cdp_result([
            ["game.player.fuel", 500],
        ])
        result = probe_common_paths(mock_cdp)
        assert len(result) == 1
        assert result[0] == ("game.player.fuel", 500.0)


class TestProbeAll:
    """Tests for probe_all function."""

    def test_combines_all_probes(self, mock_cdp: MagicMock) -> None:
        """Combines results from all probe functions."""
        mock_cdp.send.return_value = make_cdp_result([])
        result = probe_all(mock_cdp)
        assert "dom_bars" in result
        assert "js_variables" in result
        assert "numeric_globals" in result
        assert result["dom_bars"] == []
        assert result["js_variables"] == []
        assert result["numeric_globals"] == []


class TestFuelProber:
    """Tests for FuelProber class."""

    def test_probe_returns_result(self, mock_cdp: MagicMock) -> None:
        """Probe method returns FuelProbeResult."""
        mock_cdp.send.return_value = make_cdp_result([])
        prober = FuelProber(mock_cdp)
        result = prober.probe()
        assert "dom_bars" in result
        assert "js_variables" in result
        assert "numeric_globals" in result

    def test_log_results_calls_probe(self, mock_cdp: MagicMock) -> None:
        """log_results method calls probe and processes results."""
        mock_cdp.send.side_effect = [
            make_cdp_result([
                {
                    "tag": "DIV",
                    "id": "hp-bar",
                    "class_name": "health",
                    "width": "80%",
                    "computed_width": "200px",
                    "parent_class": "",
                }
            ]),
            make_cdp_result([
                {"name": "fuel", "value": 800, "path": "player.fuel"}
            ]),
            make_cdp_result([
                ["fuelValue", 800]
            ]),
        ]
        prober = FuelProber(mock_cdp)
        prober.log_results()
        # Verify probe was called (cdp.send called 3 times)
        assert mock_cdp.send.call_count == 3
