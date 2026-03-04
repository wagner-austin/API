"""Weather observation event schemas for streaming pipeline.

Provides TypedDict definitions and encoder/decoder functions for weather
observation events used in the temporal feature extraction pipeline.

Event types:
- WeatherEventV1: Input weather observation events from Kafka

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)

# =============================================================================
# Event Type Discriminator
# =============================================================================

WeatherEventType = Literal["weather.observation.v1"]

# =============================================================================
# Input Event: WeatherEventV1
# =============================================================================


class WeatherEventV1(TypedDict):
    """Single weather observation event consumed from Kafka.

    Represents a single temperature observation at a weather station
    on a specific day. Used as input to WeatherFeatureExtractor for
    McKinnon-style temporal feature extraction.

    Attributes:
        type: Event type discriminator.
        event_id: UUID for deduplication.
        station_id: Weather station identifier (partition key).
        day_of_year: Day of year (1-366) for seasonal cycle removal.
        temperature: Temperature observation in degrees Celsius.
        timestamp: ISO datetime when observation was recorded.
    """

    type: WeatherEventType
    event_id: str
    station_id: str
    day_of_year: int
    temperature: float
    timestamp: str


# =============================================================================
# Factory Function
# =============================================================================


def make_weather_event(
    *,
    event_id: str,
    station_id: str,
    day_of_year: int,
    temperature: float,
    timestamp: str,
) -> WeatherEventV1:
    """Create a weather observation event.

    Args:
        event_id: UUID for deduplication.
        station_id: Weather station identifier.
        day_of_year: Day of year (1-366).
        temperature: Temperature in degrees Celsius.
        timestamp: ISO datetime when recorded.

    Returns:
        WeatherEventV1 instance.
    """
    return {
        "type": "weather.observation.v1",
        "event_id": event_id,
        "station_id": station_id,
        "day_of_year": day_of_year,
        "temperature": temperature,
        "timestamp": timestamp,
    }


# =============================================================================
# Encoder Function
# =============================================================================


def encode_weather_event(event: WeatherEventV1) -> str:
    """Serialize a weather event to JSON string.

    Args:
        event: WeatherEventV1 to serialize.

    Returns:
        Compact JSON string.
    """
    return dump_json_str(event)


# =============================================================================
# Decoder Function
# =============================================================================


def _parse_weather_event_type(raw: str) -> WeatherEventType:
    """Parse weather event type from string.

    Args:
        raw: Raw string value.

    Returns:
        Validated WeatherEventType literal.

    Raises:
        JSONTypeError: If value is not valid.
    """
    if raw == "weather.observation.v1":
        return "weather.observation.v1"
    raise JSONTypeError(f"Expected 'weather.observation.v1', got '{raw}'")


def decode_weather_event(payload: str) -> WeatherEventV1:
    """Parse and validate a weather event from JSON string.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated WeatherEventV1.

    Raises:
        JSONTypeError: If payload is not a valid weather event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    type_raw = require_str(decoded, "type")
    event_type = _parse_weather_event_type(type_raw)
    event_id = require_str(decoded, "event_id")
    station_id = require_str(decoded, "station_id")
    day_of_year = require_int(decoded, "day_of_year")
    temperature = require_float(decoded, "temperature")
    timestamp = require_str(decoded, "timestamp")
    return {
        "type": event_type,
        "event_id": event_id,
        "station_id": station_id,
        "day_of_year": day_of_year,
        "temperature": temperature,
        "timestamp": timestamp,
    }


__all__ = [
    "WeatherEventType",
    "WeatherEventV1",
    "decode_weather_event",
    "encode_weather_event",
    "make_weather_event",
]
