"""Tests for the generic streaming worker entry point.

The entry point is what makes a domain runnable: it reads configuration,
builds and registers the domain, resolves the model and text generator, and
hands everything to GenericStreamingWorker. Nothing ran the generic worker
before this existed.

Every external dependency goes through a hook, so these tests substitute
fakes and exercise the real resolution logic rather than mocking it out.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from covenant_ml.datasets.types import TemporalFeatureState, encode_temporal_feature_state
from covenant_ml.types import PredictorProtocol
from numpy.typing import NDArray
from platform_core.json_utils import dump_json_str
from platform_core.testing import make_fake_env

from covenant_radar_api import generic_worker_entry_hooks as _hooks
from covenant_radar_api.domains.esports.domain import ESPORTS_DOMAIN_NAME
from covenant_radar_api.domains.weather.domain import WEATHER_DOMAIN_NAME, make_weather_domain
from covenant_radar_api.generic_worker_entry import (
    GenericWorkerDeps,
    build_dependencies,
    build_domain_registry,
    create_worker,
    main,
)
from covenant_radar_api.streaming._test_hooks import use_fake_kafka, use_real_kafka
from covenant_radar_api.streaming.config import StreamingConfig, load_streaming_config
from covenant_radar_api.streaming.generic_worker import make_generic_worker_config
from tests.domains.weather._test_weather_fixtures import make_flat_state

# =============================================================================
# Fakes
# =============================================================================


class _FakePredictor:
    """Predictor returning a fixed positive-class probability."""

    def __init__(self, probability: float = 0.10) -> None:
        """Initialize with the probability every prediction returns.

        Args:
            probability: Positive-class probability to report.
        """
        self._probability = probability

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a fixed two-column probability array.

        Args:
            x: Feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        result: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        result[:, 0] = 1.0 - self._probability
        result[:, 1] = self._probability
        return result


class _FakeTextGenerator:
    """Text generator recording the prompts it receives."""

    def __init__(self) -> None:
        """Initialize with an empty prompt log."""
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        """Return a fixed summary and record the prompt.

        Args:
            prompt: Prompt describing the alert context.

        Returns:
            Fixed summary text.
        """
        self.prompts.append(prompt)
        return "summary"


class _RecordingLogger:
    """Logger capturing what the entry point reports."""

    def __init__(self) -> None:
        """Initialize with empty message logs."""
        self.infos: list[str] = []
        self.errors: list[str] = []

    def info(self, msg: str) -> None:
        """Record an informational message.

        Args:
            msg: Message text.
        """
        self.infos.append(msg)

    def error(self, msg: str) -> None:
        """Record an error message.

        Args:
            msg: Message text.
        """
        self.errors.append(msg)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def weather_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write a fitted state and station map to disk.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Paths to the state file and the station map file.
    """
    state: TemporalFeatureState = make_flat_state()
    state_path = tmp_path / "state.json"
    state_path.write_text(dump_json_str(encode_temporal_feature_state(state)), encoding="utf-8")

    map_path = tmp_path / "stations.json"
    map_path.write_text(dump_json_str({"station-a": 0}), encoding="utf-8")
    return state_path, map_path


@pytest.fixture(autouse=True)
def restore_hooks() -> Generator[None, None, None]:
    """Restore every entry hook after each test."""
    original_state = _hooks.temporal_state_loader
    original_map = _hooks.station_map_loader
    original_model = _hooks.model_loader
    original_text = _hooks.text_generator_factory
    yield
    _hooks.temporal_state_loader = original_state
    _hooks.station_map_loader = original_map
    _hooks.model_loader = original_model
    _hooks.text_generator_factory = original_text
    use_real_kafka()


def _wire_fakes() -> _FakeTextGenerator:
    """Point the model and text-generator hooks at fakes.

    Returns:
        The text generator the hook now returns.
    """
    generator = _FakeTextGenerator()

    def _model_loader(model_path: str) -> PredictorProtocol:
        return _FakePredictor()

    def _text_factory(api_key: str, model: str) -> _FakeTextGenerator:
        return generator

    _hooks.model_loader = _model_loader
    _hooks.text_generator_factory = _text_factory
    return generator


# =============================================================================
# Tests
# =============================================================================


class TestBuildDomainRegistry:
    """The registry is populated from configuration, not hard-coded."""

    def test_registers_every_available_domain(self) -> None:
        """Both domains are offered, and neither needs configuration to be.

        Nothing registered a domain before this, so the registry was
        permanently empty and the generic worker had nothing to run.
        """
        make_fake_env({})

        registry = build_domain_registry()

        assert registry.list_names() == (ESPORTS_DOMAIN_NAME, WEATHER_DOMAIN_NAME)

    def test_registration_reads_no_configuration(self) -> None:
        """Registering must not build, or every deployment pays for every domain.

        Weather reads a fitted state and station map off disk. If
        registration constructed it, a deployment running only esports
        would fail at startup demanding files it never opens.
        """
        make_fake_env({})

        registry = build_domain_registry()

        assert registry.get(ESPORTS_DOMAIN_NAME).config["name"] == ESPORTS_DOMAIN_NAME

    def test_esports_threshold_is_configurable(self) -> None:
        """A deployment can tune esports alerts without weather configured."""
        make_fake_env({"ESPORTS__ALERT_THRESHOLD": "0.60"})

        domain = build_domain_registry().get(ESPORTS_DOMAIN_NAME)

        assert domain.config["alert_threshold"] == pytest.approx(0.60)

    def test_alert_threshold_is_configurable(self, weather_files: tuple[Path, Path]) -> None:
        """A deployment can tune how often alerts fire without a rebuild."""
        state_path, map_path = weather_files
        make_fake_env(
            {
                "WEATHER__STATE_PATH": str(state_path),
                "WEATHER__STATION_MAP_PATH": str(map_path),
                "WEATHER__ALERT_THRESHOLD": "0.25",
            }
        )

        domain = build_domain_registry().get(WEATHER_DOMAIN_NAME)

        assert domain.config["alert_threshold"] == pytest.approx(0.25)

    def test_missing_state_path_raises_when_weather_is_requested(self) -> None:
        """A required path that is unset fails when it is needed, naming itself.

        The failure belongs at get(), not at register(): the same registry
        serves an esports deployment that never sets these variables.
        """
        make_fake_env({})
        registry = build_domain_registry()

        with pytest.raises(RuntimeError, match="WEATHER__STATE_PATH"):
            registry.get(WEATHER_DOMAIN_NAME)


class TestBuildDependencies:
    """Dependencies are resolved from the environment through the hooks."""

    def test_resolves_domain_model_and_generator(
        self,
        weather_files: tuple[Path, Path],
    ) -> None:
        """Everything the worker needs is present and typed."""
        state_path, map_path = weather_files
        make_fake_env(
            {
                "WEATHER__STATE_PATH": str(state_path),
                "WEATHER__STATION_MAP_PATH": str(map_path),
                "MODEL_PATH": "/models/model.ubj",
                "GEMINI_API_KEY": "key",
            }
        )
        _wire_fakes()

        deps = build_dependencies()

        assert deps["domain"].config["name"] == WEATHER_DOMAIN_NAME
        assert deps["worker_config"]["model_version"] == "v1.0.0"

    def test_model_version_and_timeout_are_configurable(
        self,
        weather_files: tuple[Path, Path],
    ) -> None:
        """Both reach the worker config the predictions are stamped with."""
        state_path, map_path = weather_files
        make_fake_env(
            {
                "WEATHER__STATE_PATH": str(state_path),
                "WEATHER__STATION_MAP_PATH": str(map_path),
                "MODEL_PATH": "/models/model.ubj",
                "GEMINI_API_KEY": "key",
                "MODEL_VERSION": "v9.9.9",
                "STREAMING__POLL_TIMEOUT_SECONDS": "2.5",
            }
        )
        _wire_fakes()

        deps = build_dependencies()

        assert deps["worker_config"]["model_version"] == "v9.9.9"
        assert deps["worker_config"]["poll_timeout_seconds"] == pytest.approx(2.5)

    def test_unknown_domain_is_reported(self, weather_files: tuple[Path, Path]) -> None:
        """Naming a domain that is not registered fails with the available set."""
        state_path, map_path = weather_files
        make_fake_env(
            {
                "WEATHER__STATE_PATH": str(state_path),
                "WEATHER__STATION_MAP_PATH": str(map_path),
                "MODEL_PATH": "/models/model.ubj",
                "GEMINI_API_KEY": "key",
                "STREAMING__DOMAIN": "curling",
            }
        )
        _wire_fakes()

        with pytest.raises(KeyError, match="curling"):
            build_dependencies()

    def test_esports_runs_without_any_weather_configuration(self) -> None:
        """An esports deployment must not be asked for weather's files.

        This is what the lazy registry buys. Registering weather eagerly
        would make this raise on WEATHER__STATE_PATH, for a domain the
        deployment never touches.
        """
        make_fake_env(
            {
                "MODEL_PATH": "/models/model.ubj",
                "GEMINI_API_KEY": "key",
                "STREAMING__DOMAIN": ESPORTS_DOMAIN_NAME,
            }
        )
        _wire_fakes()

        deps = build_dependencies()

        assert deps["domain"].config["name"] == ESPORTS_DOMAIN_NAME
        assert deps["domain"].config["input_topic"] == "esports.match_state.v1"

    def test_missing_gemini_key_raises(self, weather_files: tuple[Path, Path]) -> None:
        """Alerts cannot be summarised without a key, so it is required."""
        state_path, map_path = weather_files
        make_fake_env(
            {
                "WEATHER__STATE_PATH": str(state_path),
                "WEATHER__STATION_MAP_PATH": str(map_path),
                "MODEL_PATH": "/models/model.ubj",
            }
        )
        _wire_fakes()

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            build_dependencies()


def _make_deps() -> GenericWorkerDeps:
    """Build worker dependencies without reading the environment.

    Returns:
        GenericWorkerDeps wired to a real weather domain and fake model.
    """
    return {
        "domain": make_weather_domain(
            state=make_flat_state(),
            station_to_location={"station-a": 0},
        ),
        "model": _FakePredictor(),
        "text_generator": _FakeTextGenerator(),
        "worker_config": make_generic_worker_config(
            model_version="v1.0.0",
            poll_timeout_seconds=1.0,
        ),
    }


class TestCreateWorker:
    """The worker is constructed against the domain's own topics."""

    def test_builds_a_worker(self) -> None:
        """A worker is produced and is not yet running."""
        make_fake_env({"CONFLUENT__SECURITY_PROTOCOL": "PLAINTEXT"})
        config: StreamingConfig = load_streaming_config()

        worker = create_worker(config, _make_deps())

        assert worker.is_running is False


class TestMain:
    """main() gates on the streaming flag before touching Kafka."""

    def test_returns_one_when_streaming_disabled(self) -> None:
        """The flag defaults off, so an unconfigured deployment exits cleanly.

        Reaching Kafka first would fail with a connection error instead of
        saying the feature is switched off.
        """
        make_fake_env({})
        config: StreamingConfig = load_streaming_config()
        logger = _RecordingLogger()

        exit_code = main(streaming_config=config, deps=None, logger=logger)

        assert exit_code == 1
        assert any("STREAMING__ENABLED" in message for message in logger.errors)

    def test_runs_and_shuts_down_cleanly(self) -> None:
        """The success path builds a worker, runs it, and shuts it down.

        Bounded to zero iterations so the daemon loop exits immediately;
        unbounded is what the container runs.
        """
        use_fake_kafka()
        make_fake_env(
            {
                "STREAMING__ENABLED": "true",
                "CONFLUENT__SECURITY_PROTOCOL": "PLAINTEXT",
            }
        )
        config: StreamingConfig = load_streaming_config()
        logger = _RecordingLogger()

        exit_code = main(
            streaming_config=config,
            deps=_make_deps(),
            logger=logger,
            max_iterations=0,
        )

        assert exit_code == 0
        assert logger.errors == []

    def test_reports_the_domain_and_topic_on_start(self) -> None:
        """The startup line names what is being consumed, for operators."""
        use_fake_kafka()
        make_fake_env(
            {
                "STREAMING__ENABLED": "true",
                "CONFLUENT__SECURITY_PROTOCOL": "PLAINTEXT",
            }
        )
        config: StreamingConfig = load_streaming_config()
        logger = _RecordingLogger()

        main(
            streaming_config=config,
            deps=_make_deps(),
            logger=logger,
            max_iterations=0,
        )

        assert any(WEATHER_DOMAIN_NAME in message for message in logger.infos)
        assert any("weather.observations.v1" in message for message in logger.infos)


class TestModuleEntry:
    """The module runs as a script, which is how the container starts it."""

    def test_main_guard_exits_with_the_disabled_code(self) -> None:
        """Running as __main__ propagates main()'s exit code through sys.exit.

        The container CMD invokes the console script, so this guard is the
        real production path. Streaming is left disabled, which makes the
        run terminate immediately with the documented exit code instead of
        entering the daemon loop.
        """
        import runpy
        import sys
        from types import ModuleType

        make_fake_env({})
        module_name = "covenant_radar_api.generic_worker_entry"
        saved: ModuleType | None = sys.modules.pop(module_name, None)
        try:
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module(module_name, run_name="__main__", alter_sys=True)
            assert exc_info.value.code == 1
        finally:
            if saved is not None:
                sys.modules[module_name] = saved
