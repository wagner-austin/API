"""Test hooks for the generic streaming worker entry point.

Every external dependency the entry point reaches for is behind a hook here:
reading the fitted state and station map off disk, loading the ML model, and
constructing the Gemini client. Production sets them to the real
implementations at import; tests replace them with fakes.

There are no conditionals in the entry point as a result -- it calls the hook
and gets whatever is wired.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.datasets.types_temporal import (
    TemporalFeatureState,
    require_temporal_feature_state,
)
from covenant_ml.types import PredictorProtocol
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_int,
)
from platform_core.logging import get_logger

from .integrations.google_ai import GeminiConfig, create_gemini_client
from .streaming._test_hooks_generic_worker import TextGeneratorProtocol

# =============================================================================
# Logger Protocol
# =============================================================================


class LoggerProtocol(Protocol):
    """Protocol for the structured logger the entry point writes to."""

    def info(self, msg: str) -> None:
        """Log an informational message.

        Args:
            msg: Message text.
        """
        ...

    def error(self, msg: str) -> None:
        """Log an error message.

        Args:
            msg: Message text.
        """
        ...


class LoggerFactoryProtocol(Protocol):
    """Protocol for creating a logger by name."""

    def __call__(self, name: str) -> LoggerProtocol:
        """Create a logger.

        Args:
            name: Logger name, conventionally the module name.

        Returns:
            Logger instance.
        """
        ...


def _real_logger_factory(name: str) -> LoggerProtocol:
    """Create a real structured logger.

    Args:
        name: Logger name.

    Returns:
        platform_core logger.
    """
    return get_logger(name)


# =============================================================================
# Temporal State Loader
# =============================================================================


class TemporalStateLoaderProtocol(Protocol):
    """Protocol for reading a fitted temporal feature state from disk."""

    def __call__(self, path: Path) -> TemporalFeatureState:
        """Load and validate a fitted state.

        Args:
            path: Path to the JSON state file.

        Returns:
            Validated TemporalFeatureState.
        """
        ...


def _real_temporal_state_loader(path: Path) -> TemporalFeatureState:
    """Read a fitted temporal feature state from a JSON file.

    The file is the output of encode_temporal_feature_state, and is validated
    on the way back in rather than trusted: a state whose coefficient arrays
    disagree with n_locations would otherwise fail much later, inside feature
    extraction, with an index error naming nothing useful.

    Args:
        path: Path to the JSON state file.

    Returns:
        Validated TemporalFeatureState.

    Raises:
        FileNotFoundError: If the file does not exist.
        InvalidJsonError: If the file is not valid JSON.
        JSONTypeError: If the top-level payload is not an object.
        ValueError: If a required field is missing or has the wrong type.
    """
    payload = path.read_text(encoding="utf-8")
    decoded: dict[str, JSONValue] = narrow_json_to_dict(load_json_str(payload))
    return require_temporal_feature_state(decoded)


# =============================================================================
# Station Map Loader
# =============================================================================


class StationMapLoaderProtocol(Protocol):
    """Protocol for reading the station-to-location mapping from disk."""

    def __call__(self, path: Path) -> dict[str, int]:
        """Load and validate a station mapping.

        Args:
            path: Path to the JSON mapping file.

        Returns:
            Mapping from station identifier to location index.
        """
        ...


def _real_station_map_loader(path: Path) -> dict[str, int]:
    """Read the station-to-location mapping from a JSON file.

    Every value is required to be an integer index. A string index here
    would surface as a numpy indexing error deep inside extraction, naming
    the array rather than the station that was wrong.

    Args:
        path: Path to the JSON mapping file.

    Returns:
        Mapping from station identifier to location index.

    Raises:
        FileNotFoundError: If the file does not exist.
        InvalidJsonError: If the file is not valid JSON.
        JSONTypeError: If the payload is not an object of integer values.
    """
    payload = path.read_text(encoding="utf-8")
    decoded: dict[str, JSONValue] = narrow_json_to_dict(load_json_str(payload))
    if not decoded:
        raise JSONTypeError("Station map is empty; no station could be featurised")
    return {station: require_int(decoded, station) for station in decoded}


# =============================================================================
# Model Loader
# =============================================================================


class ModelLoaderProtocol(Protocol):
    """Protocol for loading the ML model the worker runs."""

    def __call__(self, model_path: str) -> PredictorProtocol:
        """Load a model from disk.

        Args:
            model_path: Path to the saved model file.

        Returns:
            Model exposing predict_proba.
        """
        ...


def _real_model_loader(model_path: str) -> PredictorProtocol:
    """Load an XGBoost model from disk.

    Args:
        model_path: Path to the saved model file.

    Returns:
        Loaded model.
    """
    from covenant_ml.predictor import load_model

    return load_model(model_path)


# =============================================================================
# Text Generator Factory
# =============================================================================


class TextGeneratorFactoryProtocol(Protocol):
    """Protocol for constructing the alert text generator."""

    def __call__(self, api_key: str, model: str) -> TextGeneratorProtocol:
        """Create a text generator.

        Args:
            api_key: Google AI API key.
            model: Gemini model name.

        Returns:
            Generator exposing generate_text.
        """
        ...


def _real_text_generator_factory(api_key: str, model: str) -> TextGeneratorProtocol:
    """Create a Gemini-backed text generator.

    Args:
        api_key: Google AI API key.
        model: Gemini model name.

    Returns:
        GeminiClient, which satisfies TextGeneratorProtocol structurally.
    """
    config: GeminiConfig = {"api_key": api_key, "model": model}
    return create_gemini_client(config)


# =============================================================================
# Hooks (production wiring)
# =============================================================================

logger_factory: LoggerFactoryProtocol = _real_logger_factory
temporal_state_loader: TemporalStateLoaderProtocol = _real_temporal_state_loader
station_map_loader: StationMapLoaderProtocol = _real_station_map_loader
model_loader: ModelLoaderProtocol = _real_model_loader
text_generator_factory: TextGeneratorFactoryProtocol = _real_text_generator_factory


__all__ = [
    "LoggerFactoryProtocol",
    "LoggerProtocol",
    "ModelLoaderProtocol",
    "StationMapLoaderProtocol",
    "TemporalStateLoaderProtocol",
    "TextGeneratorFactoryProtocol",
    "logger_factory",
    "model_loader",
    "station_map_loader",
    "temporal_state_loader",
    "text_generator_factory",
]
