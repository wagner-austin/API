"""Tests for weather observation event schemas."""

from __future__ import annotations

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str

from covenant_radar_api.domains.weather.schemas import (
    WeatherEventV1,
    decode_weather_event,
    encode_weather_event,
    make_weather_event,
)


class TestMakeWeatherEvent:
    """Tests for make_weather_event factory."""

    def test_creates_event(self) -> None:
        """Create weather event with all fields."""
        event = make_weather_event(
            event_id="evt-001",
            station_id="station-alpha",
            day_of_year=172,
            temperature=28.5,
            timestamp="2025-06-21T14:00:00Z",
        )

        assert event["type"] == "weather.observation.v1"
        assert event["event_id"] == "evt-001"
        assert event["station_id"] == "station-alpha"
        assert event["day_of_year"] == 172
        assert event["temperature"] == 28.5
        assert event["timestamp"] == "2025-06-21T14:00:00Z"

    def test_negative_temperature(self) -> None:
        """Create weather event with negative temperature."""
        event = make_weather_event(
            event_id="evt-002",
            station_id="station-beta",
            day_of_year=15,
            temperature=-12.3,
            timestamp="2025-01-15T08:00:00Z",
        )

        assert event["temperature"] == -12.3

    def test_day_of_year_boundaries(self) -> None:
        """Create events with day_of_year at boundaries."""
        event_jan1 = make_weather_event(
            event_id="evt-003",
            station_id="s1",
            day_of_year=1,
            temperature=0.0,
            timestamp="2025-01-01T00:00:00Z",
        )
        event_dec31 = make_weather_event(
            event_id="evt-004",
            station_id="s1",
            day_of_year=365,
            temperature=0.0,
            timestamp="2025-12-31T00:00:00Z",
        )

        assert event_jan1["day_of_year"] == 1
        assert event_dec31["day_of_year"] == 365


class TestEncodeWeatherEvent:
    """Tests for encode_weather_event."""

    def test_round_trip(self) -> None:
        """Encode then decode produces identical event."""
        original = make_weather_event(
            event_id="evt-100",
            station_id="station-gamma",
            day_of_year=200,
            temperature=35.7,
            timestamp="2025-07-19T12:00:00Z",
        )

        json_str = encode_weather_event(original)
        decoded = decode_weather_event(json_str)

        assert decoded["type"] == original["type"]
        assert decoded["event_id"] == original["event_id"]
        assert decoded["station_id"] == original["station_id"]
        assert decoded["day_of_year"] == original["day_of_year"]
        assert decoded["temperature"] == original["temperature"]
        assert decoded["timestamp"] == original["timestamp"]

    def test_produces_json_with_type_field(self) -> None:
        """Encoder returns JSON containing the event type."""
        event = make_weather_event(
            event_id="evt-101",
            station_id="s1",
            day_of_year=1,
            temperature=0.0,
            timestamp="2025-01-01T00:00:00Z",
        )

        result = encode_weather_event(event)
        assert "weather.observation.v1" in result
        assert "evt-101" in result


class TestDecodeWeatherEvent:
    """Tests for decode_weather_event."""

    def test_decodes_valid_payload(self) -> None:
        """Decode a valid JSON payload."""
        payload_dict: WeatherEventV1 = {
            "type": "weather.observation.v1",
            "event_id": "evt-200",
            "station_id": "station-delta",
            "day_of_year": 90,
            "temperature": 15.2,
            "timestamp": "2025-03-31T10:00:00Z",
        }
        payload = dump_json_str(payload_dict)

        event = decode_weather_event(payload)

        assert event["type"] == "weather.observation.v1"
        assert event["event_id"] == "evt-200"
        assert event["station_id"] == "station-delta"
        assert event["day_of_year"] == 90
        assert event["temperature"] == 15.2
        assert event["timestamp"] == "2025-03-31T10:00:00Z"

    def test_wrong_type_raises(self) -> None:
        """Raises JSONTypeError for wrong event type."""
        payload = dump_json_str(
            {
                "type": "something.else.v1",
                "event_id": "evt-300",
                "station_id": "s1",
                "day_of_year": 1,
                "temperature": 0.0,
                "timestamp": "2025-01-01T00:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError, match=r"Expected 'weather\.observation\.v1'"):
            decode_weather_event(payload)

    def test_missing_field_raises(self) -> None:
        """Raises JSONTypeError for missing required field."""
        payload = dump_json_str(
            {
                "type": "weather.observation.v1",
                "event_id": "evt-400",
                # missing station_id
                "day_of_year": 1,
                "temperature": 0.0,
                "timestamp": "2025-01-01T00:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError):
            decode_weather_event(payload)

    def test_invalid_json_raises(self) -> None:
        """Raises InvalidJsonError on invalid JSON."""
        with pytest.raises(InvalidJsonError):
            decode_weather_event("not valid json")

    def test_non_object_json_raises(self) -> None:
        """Raises JSONTypeError on JSON that is not an object."""
        with pytest.raises(JSONTypeError):
            decode_weather_event('"just a string"')
