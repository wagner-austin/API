"""Tests for fuel_probe module."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue

from tankpit_bot.browser.fuel_probe import (
    FuelProber,
    probe_all,
    probe_common_paths,
    probe_dom_bars,
    probe_game_variables,
    probe_numeric_globals,
)
from tests.conftest import FakeCDPSessionSimple


def make_cdp_result(value: JSONValue) -> JSONObject:
    """Create a CDP result wrapper.

    Args:
        value: The value to wrap in result structure.

    Returns:
        CDP response with nested result.value structure.
    """
    result_inner: JSONObject = {"value": value}
    return {"result": result_inner}


class TestProbeDomBars:
    """Tests for probe_dom_bars function."""

    def test_returns_empty_list_when_no_bars(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when no bars found."""
        fake_cdp.add_response(make_cdp_result([]))
        result = probe_dom_bars(fake_cdp)
        assert result == []

    def test_parses_bar_elements(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Parses bar elements correctly."""
        bar_data: JSONObject = {
            "tag": "DIV",
            "id": "fuel-bar",
            "class_name": "progress-bar",
            "width": "75%",
            "computed_width": "150px",
            "parent_class": "bar-container",
        }
        fake_cdp.add_response(make_cdp_result([bar_data]))
        result = probe_dom_bars(fake_cdp)
        assert len(result) == 1
        assert result[0]["tag"] == "DIV"
        assert result[0]["id"] == "fuel-bar"
        assert result[0]["width"] == "75%"

    def test_handles_invalid_result(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Handles invalid CDP result gracefully."""
        fake_cdp.add_response({"result": None})
        result = probe_dom_bars(fake_cdp)
        assert result == []

    def test_returns_empty_when_value_not_list(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when result value is not a list."""
        fake_cdp.add_response(make_cdp_result("not a list"))
        result = probe_dom_bars(fake_cdp)
        assert result == []

    def test_skips_non_dict_items(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Skips non-dict items in the list."""
        bar_data: JSONObject = {
            "tag": "DIV",
            "id": "fuel-bar",
            "class_name": "progress-bar",
            "width": "75%",
            "computed_width": "150px",
            "parent_class": "container",
        }
        fake_cdp.add_response(make_cdp_result([bar_data, "not a dict", 123]))
        result = probe_dom_bars(fake_cdp)
        assert len(result) == 1
        assert result[0]["id"] == "fuel-bar"


class TestProbeGameVariables:
    """Tests for probe_game_variables function."""

    def test_returns_empty_list_when_no_vars(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when no variables found."""
        fake_cdp.add_response(make_cdp_result([]))
        result = probe_game_variables(fake_cdp)
        assert result == []

    def test_parses_game_variables(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Parses game variables correctly."""
        var1: JSONObject = {"name": "fuel", "value": 850, "path": "game.player.fuel"}
        var2: JSONObject = {"name": "hp", "value": 100, "path": "player.hp"}
        fake_cdp.add_response(make_cdp_result([var1, var2]))
        result = probe_game_variables(fake_cdp)
        assert len(result) == 2
        assert result[0]["name"] == "fuel"
        assert result[0]["value"] == 850
        assert result[0]["path"] == "game.player.fuel"

    def test_returns_empty_when_value_not_list(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when result value is not a list."""
        fake_cdp.add_response(make_cdp_result("not a list"))
        result = probe_game_variables(fake_cdp)
        assert result == []

    def test_skips_non_dict_items(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Skips non-dict items in the list."""
        var1: JSONObject = {"name": "fuel", "value": 850, "path": "game.player.fuel"}
        fake_cdp.add_response(make_cdp_result([var1, "not a dict", 123]))
        result = probe_game_variables(fake_cdp)
        assert len(result) == 1
        assert result[0]["name"] == "fuel"

    def test_handles_non_dict_result_obj(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when result_obj is not a dict."""
        fake_cdp.add_response({"result": None})
        result = probe_game_variables(fake_cdp)
        assert result == []


class TestProbeNumericGlobals:
    """Tests for probe_numeric_globals function."""

    def test_returns_empty_list_when_no_globals(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when no numeric globals."""
        fake_cdp.add_response(make_cdp_result([]))
        result = probe_numeric_globals(fake_cdp)
        assert result == []

    def test_parses_numeric_tuples(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Parses numeric global tuples."""
        tuple1: list[JSONValue] = ["fuelLevel", 750]
        tuple2: list[JSONValue] = ["maxFuel", 1000]
        fake_cdp.add_response(make_cdp_result([tuple1, tuple2]))
        result = probe_numeric_globals(fake_cdp)
        assert len(result) == 2
        assert result[0] == ("fuelLevel", 750.0)
        assert result[1] == ("maxFuel", 1000.0)

    def test_returns_empty_when_value_not_list(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when result value is not a list."""
        fake_cdp.add_response(make_cdp_result("not a list"))
        result = probe_numeric_globals(fake_cdp)
        assert result == []

    def test_skips_non_list_items(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Skips non-list items in the results."""
        tuple1: list[JSONValue] = ["fuelLevel", 750]
        fake_cdp.add_response(make_cdp_result([tuple1, "not a list", {"dict": "item"}]))
        result = probe_numeric_globals(fake_cdp)
        assert len(result) == 1
        assert result[0] == ("fuelLevel", 750.0)

    def test_skips_tuples_with_wrong_length(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Skips tuples that don't have exactly 2 elements."""
        tuple_ok: list[JSONValue] = ["fuel", 500]
        tuple_short: list[JSONValue] = ["short"]
        tuple_long: list[JSONValue] = ["too", "many", "values"]
        fake_cdp.add_response(make_cdp_result([tuple_ok, tuple_short, tuple_long]))
        result = probe_numeric_globals(fake_cdp)
        assert len(result) == 1
        assert result[0] == ("fuel", 500.0)

    def test_skips_tuples_with_non_numeric_value(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Skips tuples where second element is not numeric."""
        tuple_numeric: list[JSONValue] = ["fuel", 500]
        tuple_string: list[JSONValue] = ["name", "not a number"]
        fake_cdp.add_response(make_cdp_result([tuple_numeric, tuple_string]))
        result = probe_numeric_globals(fake_cdp)
        assert len(result) == 1
        assert result[0] == ("fuel", 500.0)

    def test_handles_non_dict_result_obj(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when result_obj is not a dict."""
        fake_cdp.add_response({"result": None})
        result = probe_numeric_globals(fake_cdp)
        assert result == []


class TestProbeCommonPaths:
    """Tests for probe_common_paths function."""

    def test_returns_empty_list_when_no_paths(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when no paths found."""
        fake_cdp.add_response(make_cdp_result([]))
        result = probe_common_paths(fake_cdp)
        assert result == []

    def test_parses_path_values(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Parses path-value pairs."""
        path_tuple: list[JSONValue] = ["game.player.fuel", 500]
        fake_cdp.add_response(make_cdp_result([path_tuple]))
        result = probe_common_paths(fake_cdp)
        assert len(result) == 1
        assert result[0] == ("game.player.fuel", 500.0)

    def test_returns_empty_when_value_not_list(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when result value is not a list."""
        fake_cdp.add_response(make_cdp_result("not a list"))
        result = probe_common_paths(fake_cdp)
        assert result == []

    def test_skips_non_list_items(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Skips non-list items in the results."""
        path_tuple: list[JSONValue] = ["game.fuel", 750]
        fake_cdp.add_response(make_cdp_result([path_tuple, "not a list", {"dict": "item"}]))
        result = probe_common_paths(fake_cdp)
        assert len(result) == 1
        assert result[0] == ("game.fuel", 750.0)

    def test_skips_tuples_with_wrong_length(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Skips tuples that don't have exactly 2 elements."""
        tuple_ok: list[JSONValue] = ["path.to.var", 500]
        tuple_short: list[JSONValue] = ["short"]
        tuple_long: list[JSONValue] = ["too", "many", "values"]
        fake_cdp.add_response(make_cdp_result([tuple_ok, tuple_short, tuple_long]))
        result = probe_common_paths(fake_cdp)
        assert len(result) == 1
        assert result[0] == ("path.to.var", 500.0)

    def test_skips_tuples_with_non_numeric_value(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Skips tuples where second element is not numeric."""
        tuple_numeric: list[JSONValue] = ["game.fuel", 500]
        tuple_string: list[JSONValue] = ["game.name", "not a number"]
        fake_cdp.add_response(make_cdp_result([tuple_numeric, tuple_string]))
        result = probe_common_paths(fake_cdp)
        assert len(result) == 1
        assert result[0] == ("game.fuel", 500.0)

    def test_handles_non_dict_result_obj(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Returns empty list when result_obj is not a dict."""
        fake_cdp.add_response({"result": None})
        result = probe_common_paths(fake_cdp)
        assert result == []


class TestProbeAll:
    """Tests for probe_all function."""

    def test_combines_all_probes(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Combines results from all probe functions."""
        fake_cdp.add_response(make_cdp_result([]))
        fake_cdp.add_response(make_cdp_result([]))
        fake_cdp.add_response(make_cdp_result([]))
        result = probe_all(fake_cdp)
        assert "dom_bars" in result
        assert "js_variables" in result
        assert "numeric_globals" in result
        assert result["dom_bars"] == []
        assert result["js_variables"] == []
        assert result["numeric_globals"] == []


class TestFuelProber:
    """Tests for FuelProber class."""

    def test_probe_returns_result(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """Probe method returns FuelProbeResult."""
        fake_cdp.add_response(make_cdp_result([]))
        fake_cdp.add_response(make_cdp_result([]))
        fake_cdp.add_response(make_cdp_result([]))
        prober = FuelProber(fake_cdp)
        result = prober.probe()
        assert "dom_bars" in result
        assert "js_variables" in result
        assert "numeric_globals" in result

    def test_log_results_calls_probe(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """log_results method calls probe and processes results."""
        bar_data: JSONObject = {
            "tag": "DIV",
            "id": "hp-bar",
            "class_name": "health",
            "width": "80%",
            "computed_width": "200px",
            "parent_class": "",
        }
        var_data: JSONObject = {"name": "fuel", "value": 800, "path": "player.fuel"}
        numeric_tuple: list[JSONValue] = ["fuelValue", 800]

        fake_cdp.add_response(make_cdp_result([bar_data]))
        fake_cdp.add_response(make_cdp_result([var_data]))
        fake_cdp.add_response(make_cdp_result([numeric_tuple]))

        prober = FuelProber(fake_cdp)
        prober.log_results()
        # Verify probe was called (cdp.send called 3 times)
        assert fake_cdp.call_count == 3

    def test_log_results_with_empty_dom_bars(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """log_results handles empty dom_bars list."""
        var_data: JSONObject = {"name": "fuel", "value": 800, "path": "player.fuel"}
        numeric_tuple: list[JSONValue] = ["fuelValue", 800]

        fake_cdp.add_response(make_cdp_result([]))  # Empty dom_bars
        fake_cdp.add_response(make_cdp_result([var_data]))
        fake_cdp.add_response(make_cdp_result([numeric_tuple]))

        prober = FuelProber(fake_cdp)
        prober.log_results()
        assert fake_cdp.call_count == 3

    def test_log_results_with_empty_js_variables(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """log_results handles empty js_variables list."""
        bar_data: JSONObject = {
            "tag": "DIV",
            "id": "hp-bar",
            "class_name": "health",
            "width": "80%",
            "computed_width": "200px",
            "parent_class": "",
        }
        numeric_tuple: list[JSONValue] = ["fuelValue", 800]

        fake_cdp.add_response(make_cdp_result([bar_data]))
        fake_cdp.add_response(make_cdp_result([]))  # Empty js_variables
        fake_cdp.add_response(make_cdp_result([numeric_tuple]))

        prober = FuelProber(fake_cdp)
        prober.log_results()
        assert fake_cdp.call_count == 3

    def test_log_results_with_empty_numeric_globals(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """log_results handles empty numeric_globals list."""
        bar_data: JSONObject = {
            "tag": "DIV",
            "id": "hp-bar",
            "class_name": "health",
            "width": "80%",
            "computed_width": "200px",
            "parent_class": "",
        }
        var_data: JSONObject = {"name": "fuel", "value": 800, "path": "player.fuel"}

        fake_cdp.add_response(make_cdp_result([bar_data]))
        fake_cdp.add_response(make_cdp_result([var_data]))
        fake_cdp.add_response(make_cdp_result([]))  # Empty numeric_globals

        prober = FuelProber(fake_cdp)
        prober.log_results()
        assert fake_cdp.call_count == 3

    def test_log_results_all_empty(self, fake_cdp: FakeCDPSessionSimple) -> None:
        """log_results handles all empty lists."""
        fake_cdp.add_response(make_cdp_result([]))
        fake_cdp.add_response(make_cdp_result([]))
        fake_cdp.add_response(make_cdp_result([]))

        prober = FuelProber(fake_cdp)
        prober.log_results()
        assert fake_cdp.call_count == 3
