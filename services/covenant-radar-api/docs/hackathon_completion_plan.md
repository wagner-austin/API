# Hackathon Completion Plan — AI Partner Catalyst

**Hackathon:** AI Partner Catalyst: Accelerate Innovation
**Deadline:** December 31, 2025 @ 2:00 PM PST
**Track:** Confluent (primary)
**Prize:** $12,500 first place

---

## 1) Current State

### Completed (Phases 1-4)

| Component | Status | Location |
|-----------|--------|----------|
| Datadog APM integration | ✅ | `integrations/datadog/` |
| Datadog custom metrics | ✅ | `integrations/datadog/metrics.py` |
| Kafka config TypedDicts | ✅ | `streaming/config.py` |
| Kafka event schemas | ✅ | `streaming/schemas.py` |
| Kafka producer wrapper | ✅ | `streaming/producer.py` |
| Kafka consumer wrapper | ✅ | `streaming/consumer.py` |
| StreamingWorker | ✅ | `streaming/worker.py` (890 lines) |
| Test hooks (Kafka fakes) | ✅ | `streaming/_test_hooks.py` |
| Repository fakes | ✅ | `streaming/_test_hooks_repositories.py` |
| Model fakes | ✅ | `streaming/_test_hooks_model.py` |
| Full test coverage | ✅ | `tests/streaming/` |

### Remaining (Phases 5-7)

| Component | Status | Effort |
|-----------|--------|--------|
| Gemini integration | ❌ | ~200 lines + tests |
| Data replay script | ❌ | ~150 lines + tests |
| Worker entry point | ❌ | ~50 lines + tests |
| Web UI dashboard | ❌ | ~300 lines HTML/JS |
| Railway deployment | ❌ | Config only |
| Demo video | ❌ | Recording |
| Devpost submission | ❌ | Writing |

---

## 2) Hackathon Requirements Checklist

### Confluent Track (Required)

- [ ] Real-time data stream via Confluent Cloud
- [ ] AI/ML models applied to streaming data
- [ ] Predictions generated from stream
- [ ] Demonstrate real-world problem solving

### Google Cloud AI (Required for ALL tracks)

- [ ] Integrate Google Cloud AI tools (Vertex AI, Gemini, or BigQuery ML)
- [ ] Must use Gemini/Vertex AI for some functionality

### Submission Requirements

- [ ] Hosted project URL for testing
- [ ] Public open-source repository with LICENSE
- [ ] 3-minute demo video
- [ ] Devpost form with description

---

## 3) Implementation Plan

### Phase 5A: Gemini Integration

**Purpose:** Generate human-readable alert summaries when risk exceeds threshold.

**Location:** `src/covenant_radar_api/integrations/google_ai/`

#### Files to Create

```
integrations/google_ai/
├── __init__.py              # Package exports
├── _test_hooks.py           # DI hooks for testing
├── schemas.py               # TypedDicts for request/response
└── client.py                # GeminiClient wrapper
```

#### schemas.py

```python
"""TypedDict schemas for Gemini API integration.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)


class GeminiConfig(TypedDict, total=True):
    """Configuration for Gemini client.

    Fields:
        api_key: Google AI API key.
        model: Model name (e.g., "gemini-1.5-flash").
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0.0-1.0).
    """

    api_key: str
    model: str
    max_tokens: int
    temperature: float


class AlertContext(TypedDict, total=True):
    """Context for generating alert summary.

    Fields:
        deal_id: Deal identifier.
        deal_name: Human-readable deal name.
        borrower: Borrower name.
        sector: Business sector.
        risk_probability: ML-predicted default probability.
        evaluation_status: Covenant evaluation result.
        breaches_count: Number of covenant breaches.
        covenants_evaluated: Total covenants checked.
    """

    deal_id: str
    deal_name: str
    borrower: str
    sector: str
    risk_probability: float
    evaluation_status: str
    breaches_count: int
    covenants_evaluated: int


class GeminiRequest(TypedDict, total=True):
    """Request to Gemini API.

    Fields:
        prompt: Text prompt for generation.
        context: Alert context for formatting.
    """

    prompt: str
    context: AlertContext


class GeminiResponse(TypedDict, total=True):
    """Response from Gemini API.

    Fields:
        text: Generated text response.
        tokens_used: Total tokens consumed.
        latency_ms: API call latency in milliseconds.
    """

    text: str
    tokens_used: int
    latency_ms: int


# Factory functions
def make_gemini_config(
    *,
    api_key: str,
    model: str = "gemini-1.5-flash",
    max_tokens: int = 100,
    temperature: float = 0.3,
) -> GeminiConfig:
    """Create Gemini configuration.

    Args:
        api_key: Google AI API key.
        model: Model name.
        max_tokens: Maximum response tokens.
        temperature: Sampling temperature.

    Returns:
        GeminiConfig instance.
    """
    return {
        "api_key": api_key,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def make_alert_context(
    *,
    deal_id: str,
    deal_name: str,
    borrower: str,
    sector: str,
    risk_probability: float,
    evaluation_status: str,
    breaches_count: int,
    covenants_evaluated: int,
) -> AlertContext:
    """Create alert context.

    Args:
        deal_id: Deal identifier.
        deal_name: Human-readable deal name.
        borrower: Borrower name.
        sector: Business sector.
        risk_probability: ML-predicted probability.
        evaluation_status: Covenant evaluation result.
        breaches_count: Number of breaches.
        covenants_evaluated: Total covenants.

    Returns:
        AlertContext instance.
    """
    return {
        "deal_id": deal_id,
        "deal_name": deal_name,
        "borrower": borrower,
        "sector": sector,
        "risk_probability": risk_probability,
        "evaluation_status": evaluation_status,
        "breaches_count": breaches_count,
        "covenants_evaluated": covenants_evaluated,
    }


def make_gemini_response(
    *,
    text: str,
    tokens_used: int,
    latency_ms: int,
) -> GeminiResponse:
    """Create Gemini response.

    Args:
        text: Generated text.
        tokens_used: Tokens consumed.
        latency_ms: API latency.

    Returns:
        GeminiResponse instance.
    """
    return {
        "text": text,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
    }


# Encode functions
def encode_alert_context(context: AlertContext) -> str:
    """Serialize alert context to JSON string.

    Args:
        context: AlertContext to serialize.

    Returns:
        Compact JSON string.
    """
    return dump_json_str(context)


# Decode functions
def decode_alert_context(payload: str) -> AlertContext:
    """Parse and validate alert context from JSON string.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated AlertContext.

    Raises:
        JSONTypeError: If payload is invalid.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    return {
        "deal_id": require_str(decoded, "deal_id"),
        "deal_name": require_str(decoded, "deal_name"),
        "borrower": require_str(decoded, "borrower"),
        "sector": require_str(decoded, "sector"),
        "risk_probability": require_float(decoded, "risk_probability"),
        "evaluation_status": require_str(decoded, "evaluation_status"),
        "breaches_count": require_int(decoded, "breaches_count"),
        "covenants_evaluated": require_int(decoded, "covenants_evaluated"),
    }
```

#### _test_hooks.py

```python
"""Dependency injection hooks for Gemini integration.

Production code sets hooks to real implementations at startup.
Tests set hooks to fakes.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Protocol

from .schemas import AlertContext, GeminiConfig, GeminiResponse


class GeminiClientProtocol(Protocol):
    """Protocol for Gemini client implementations."""

    def generate_alert_summary(
        self,
        context: AlertContext,
    ) -> GeminiResponse:
        """Generate alert summary from context.

        Args:
            context: Alert context with deal/risk info.

        Returns:
            GeminiResponse with generated text.

        Raises:
            GeminiAPIError: If API call fails.
        """
        ...


class FakeGeminiClient:
    """Fake Gemini client for testing.

    Returns deterministic responses without API calls.
    """

    def __init__(self, config: GeminiConfig) -> None:
        """Initialize fake client.

        Args:
            config: Gemini configuration (unused but required for interface).
        """
        self._config = config
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Get number of calls made."""
        return self._call_count

    def generate_alert_summary(
        self,
        context: AlertContext,
    ) -> GeminiResponse:
        """Generate fake alert summary.

        Args:
            context: Alert context.

        Returns:
            Deterministic fake response.
        """
        self._call_count += 1
        risk_pct = int(context["risk_probability"] * 100)
        text = (
            f"ALERT: {context['deal_name']} ({context['borrower']}) shows "
            f"{risk_pct}% default risk with {context['breaches_count']} "
            f"covenant breaches. Immediate review recommended."
        )
        return {
            "text": text,
            "tokens_used": len(text.split()),
            "latency_ms": 50,
        }


# Module-level hook
_gemini_client: GeminiClientProtocol | None = None


def get_gemini_client() -> GeminiClientProtocol:
    """Get the current Gemini client.

    Returns:
        Active GeminiClientProtocol implementation.

    Raises:
        RuntimeError: If no client is configured.
    """
    if _gemini_client is None:
        msg = "Gemini client not configured. Call set_gemini_client() first."
        raise RuntimeError(msg)
    return _gemini_client


def set_gemini_client(client: GeminiClientProtocol) -> None:
    """Set the Gemini client implementation.

    Args:
        client: GeminiClientProtocol implementation.
    """
    global _gemini_client
    _gemini_client = client


def clear_gemini_client() -> None:
    """Clear the Gemini client (for test cleanup)."""
    global _gemini_client
    _gemini_client = None
```

#### client.py

```python
"""Gemini API client for alert summary generation.

Uses google-generativeai SDK with strict typing.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import time

from .schemas import AlertContext, GeminiConfig, GeminiResponse, make_gemini_response


class GeminiAPIError(Exception):
    """Raised when Gemini API call fails."""

    pass


# Protocol for the genai module to avoid Any
class GenerativeModelProtocol:
    """Protocol for google.generativeai.GenerativeModel."""

    def generate_content(self, prompt: str) -> object:
        """Generate content from prompt."""
        ...


class GeminiClient:
    """Client for Gemini API calls.

    Wraps google-generativeai SDK with strict typing.
    """

    def __init__(self, config: GeminiConfig) -> None:
        """Initialize Gemini client.

        Args:
            config: Gemini configuration with API key and model.

        Raises:
            GeminiAPIError: If SDK initialization fails.
        """
        self._config = config
        self._model = self._initialize_model()

    def _initialize_model(self) -> GenerativeModelProtocol:
        """Initialize the generative model.

        Returns:
            Configured GenerativeModel instance.

        Raises:
            GeminiAPIError: If initialization fails.
        """
        try:
            genai_module = __import__("google.generativeai", fromlist=["configure", "GenerativeModel"])
            configure_fn = getattr(genai_module, "configure")
            model_class = getattr(genai_module, "GenerativeModel")

            configure_fn(api_key=self._config["api_key"])
            model: GenerativeModelProtocol = model_class(self._config["model"])
            return model
        except ImportError as e:
            raise GeminiAPIError(f"google-generativeai not installed: {e}") from e
        except Exception as e:
            raise GeminiAPIError(f"Failed to initialize Gemini: {e}") from e

    def _build_prompt(self, context: AlertContext) -> str:
        """Build prompt for alert summary generation.

        Args:
            context: Alert context with deal/risk info.

        Returns:
            Formatted prompt string.
        """
        return f"""Generate a concise 1-sentence alert summary for a loan covenant monitoring system.

Deal: {context['deal_name']}
Borrower: {context['borrower']}
Sector: {context['sector']}
Default Risk: {context['risk_probability']:.1%}
Covenant Status: {context['evaluation_status']}
Breaches: {context['breaches_count']} of {context['covenants_evaluated']}

Write a single professional sentence summarizing the risk and recommended action."""

    def generate_alert_summary(
        self,
        context: AlertContext,
    ) -> GeminiResponse:
        """Generate alert summary from context.

        Args:
            context: Alert context with deal/risk info.

        Returns:
            GeminiResponse with generated text.

        Raises:
            GeminiAPIError: If API call fails.
        """
        prompt = self._build_prompt(context)
        start_time = time.perf_counter()

        try:
            response = self._model.generate_content(prompt)
            text = str(getattr(response, "text", ""))
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Estimate tokens (rough approximation)
            tokens_used = len(prompt.split()) + len(text.split())

            return make_gemini_response(
                text=text.strip(),
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )
        except Exception as e:
            raise GeminiAPIError(f"Gemini API call failed: {e}") from e


def create_gemini_client(config: GeminiConfig) -> GeminiClient:
    """Factory function to create Gemini client.

    Args:
        config: Gemini configuration.

    Returns:
        Configured GeminiClient.

    Raises:
        GeminiAPIError: If client creation fails.
    """
    return GeminiClient(config)
```

#### Tests Required

```
tests/integrations/google_ai/
├── __init__.py
├── test_schemas.py          # encode/decode, make_* functions
├── test_hooks.py            # FakeGeminiClient, hook functions
└── test_client.py           # GeminiClient with fake SDK
```

---

### Phase 5B: Data Replay Script

**Purpose:** Stream existing dataset (AMEX, Taiwan, etc.) to Kafka as measurement events.

**Location:** `scripts/replay_data/`

#### Files to Create

```
scripts/replay_data/
├── __init__.py              # Package marker
├── __main__.py              # CLI entry point
├── _test_hooks.py           # DI hooks for testing
├── types.py                 # TypedDicts for replay config
├── loader.py                # Dataset loading wrapper
└── runner.py                # Main replay logic
```

#### types.py

```python
"""TypedDict definitions for data replay script.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Literal, TypedDict


ReplaySpeed = Literal["realtime", "fast", "instant"]


class ReplayConfig(TypedDict, total=True):
    """Configuration for data replay.

    Fields:
        dataset: Dataset name from registry.
        topic: Kafka topic to publish to.
        speed: Replay speed mode.
        batch_size: Events per batch.
        deal_id_prefix: Prefix for generated deal IDs.
        max_events: Maximum events to replay (0 = unlimited).
    """

    dataset: str
    topic: str
    speed: ReplaySpeed
    batch_size: int
    deal_id_prefix: str
    max_events: int


class ReplayStats(TypedDict, total=True):
    """Statistics from replay run.

    Fields:
        events_sent: Total events published.
        batches_sent: Number of batches.
        elapsed_seconds: Total runtime.
        events_per_second: Throughput.
    """

    events_sent: int
    batches_sent: int
    elapsed_seconds: float
    events_per_second: float


def make_replay_config(
    *,
    dataset: str,
    topic: str = "covenant.measurements.v1",
    speed: ReplaySpeed = "fast",
    batch_size: int = 100,
    deal_id_prefix: str = "replay",
    max_events: int = 0,
) -> ReplayConfig:
    """Create replay configuration.

    Args:
        dataset: Dataset name (e.g., "taiwan", "kaggle_amex_default").
        topic: Kafka topic.
        speed: Replay speed.
        batch_size: Events per batch.
        deal_id_prefix: Prefix for deal IDs.
        max_events: Maximum events (0 = unlimited).

    Returns:
        ReplayConfig instance.
    """
    return {
        "dataset": dataset,
        "topic": topic,
        "speed": speed,
        "batch_size": batch_size,
        "deal_id_prefix": deal_id_prefix,
        "max_events": max_events,
    }


def make_replay_stats(
    *,
    events_sent: int,
    batches_sent: int,
    elapsed_seconds: float,
) -> ReplayStats:
    """Create replay statistics.

    Args:
        events_sent: Total events.
        batches_sent: Total batches.
        elapsed_seconds: Runtime.

    Returns:
        ReplayStats instance.
    """
    eps = events_sent / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "events_sent": events_sent,
        "batches_sent": batches_sent,
        "elapsed_seconds": elapsed_seconds,
        "events_per_second": eps,
    }
```

#### runner.py (Core Logic)

```python
"""Data replay runner for streaming datasets to Kafka.

Loads dataset via covenant_ml.datasets, converts rows to measurement events,
and publishes to Kafka topic.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

from covenant_ml.datasets import DatasetLoader
from covenant_ml.datasets.types import LoadedDataset

from covenant_radar_api.streaming.producer import StreamingProducer
from covenant_radar_api.streaming.schemas import (
    MeasurementEventV1,
    make_measurement_event,
)

from .types import ReplayConfig, ReplayStats, make_replay_stats


def _generate_event_id() -> str:
    """Generate unique event ID."""
    return str(uuid.uuid4())


def _current_iso_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row_to_events(
    row_index: int,
    feature_names: tuple[str, ...],
    feature_values: tuple[float, ...],
    deal_id_prefix: str,
) -> Iterator[MeasurementEventV1]:
    """Convert a dataset row to measurement events.

    Args:
        row_index: Row number (used for deal_id and period).
        feature_names: Column names.
        feature_values: Feature values.
        deal_id_prefix: Prefix for deal ID.

    Yields:
        MeasurementEventV1 for each feature.
    """
    deal_id = f"{deal_id_prefix}-{row_index:06d}"
    period_start = f"2024-{((row_index % 12) + 1):02d}-01"
    period_end = f"2024-{((row_index % 12) + 1):02d}-28"
    timestamp = _current_iso_timestamp()

    for name, value in zip(feature_names, feature_values, strict=True):
        yield make_measurement_event(
            event_id=_generate_event_id(),
            deal_id=deal_id,
            period_start=period_start,
            period_end=period_end,
            metric_name=name,
            metric_value=float(value),
            timestamp=timestamp,
        )


def _get_delay_seconds(speed: str) -> float:
    """Get delay between batches for replay speed.

    Args:
        speed: Replay speed mode.

    Returns:
        Delay in seconds.
    """
    if speed == "realtime":
        return 1.0
    if speed == "fast":
        return 0.1
    return 0.0  # instant


class DataReplayRunner:
    """Replays dataset rows as Kafka measurement events.

    Loads dataset via covenant_ml.datasets, iterates rows,
    converts to measurement events, and publishes to Kafka.
    """

    def __init__(
        self,
        producer: StreamingProducer,
        loader: DatasetLoader,
        config: ReplayConfig,
    ) -> None:
        """Initialize replay runner.

        Args:
            producer: Kafka producer for publishing events.
            loader: Dataset loader instance.
            config: Replay configuration.
        """
        self._producer = producer
        self._loader = loader
        self._config = config

    def run(self) -> ReplayStats:
        """Execute data replay.

        Returns:
            ReplayStats with run statistics.

        Raises:
            KeyError: If dataset not found.
        """
        start_time = time.perf_counter()

        # Load dataset
        dataset: LoadedDataset = self._loader.load(self._config["dataset"])
        feature_names = tuple(dataset["feature_names"])
        features = dataset["features"]

        events_sent = 0
        batches_sent = 0
        delay = _get_delay_seconds(self._config["speed"])
        max_events = self._config["max_events"]

        batch: list[MeasurementEventV1] = []

        for row_idx in range(len(features)):
            if max_events > 0 and events_sent >= max_events:
                break

            row_values = tuple(float(v) for v in features[row_idx])

            for event in _row_to_events(
                row_index=row_idx,
                feature_names=feature_names,
                feature_values=row_values,
                deal_id_prefix=self._config["deal_id_prefix"],
            ):
                batch.append(event)
                events_sent += 1

                if len(batch) >= self._config["batch_size"]:
                    self._send_batch(batch)
                    batches_sent += 1
                    batch = []

                    if delay > 0:
                        time.sleep(delay)

                if max_events > 0 and events_sent >= max_events:
                    break

        # Send remaining batch
        if batch:
            self._send_batch(batch)
            batches_sent += 1

        # Flush producer
        self._producer.flush(timeout_seconds=10.0)

        elapsed = time.perf_counter() - start_time
        return make_replay_stats(
            events_sent=events_sent,
            batches_sent=batches_sent,
            elapsed_seconds=elapsed,
        )

    def _send_batch(self, batch: list[MeasurementEventV1]) -> None:
        """Send batch of events to Kafka.

        Args:
            batch: Events to send.
        """
        for event in batch:
            self._producer.produce_event(
                topic=self._config["topic"],
                key=event["deal_id"],
                event=event,
            )
        self._producer.poll(0.0)
```

#### CLI Entry Point (__main__.py)

```python
"""CLI entry point for data replay script.

Usage:
    poetry run python -m scripts.replay_data --dataset taiwan --speed fast
"""

from __future__ import annotations

import argparse
import sys

from covenant_ml.datasets import DatasetLoader

from covenant_radar_api.streaming.config import load_streaming_config
from covenant_radar_api.streaming.producer import create_streaming_producer

from .runner import DataReplayRunner
from .types import ReplaySpeed, make_replay_config


def _parse_speed(value: str) -> ReplaySpeed:
    """Parse replay speed argument.

    Args:
        value: Speed string.

    Returns:
        Validated ReplaySpeed.

    Raises:
        argparse.ArgumentTypeError: If invalid.
    """
    if value == "realtime":
        return "realtime"
    if value == "fast":
        return "fast"
    if value == "instant":
        return "instant"
    raise argparse.ArgumentTypeError(f"Invalid speed: {value}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Replay dataset as Kafka measurement events"
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Dataset name (e.g., taiwan, kaggle_amex_default)",
    )
    parser.add_argument(
        "--speed", "-s",
        type=_parse_speed,
        default="fast",
        help="Replay speed: realtime, fast, instant (default: fast)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Events per batch (default: 100)",
    )
    parser.add_argument(
        "--max-events", "-m",
        type=int,
        default=1000,
        help="Maximum events to send (default: 1000, 0 = unlimited)",
    )
    parser.add_argument(
        "--deal-prefix", "-p",
        default="replay",
        help="Deal ID prefix (default: replay)",
    )

    args = parser.parse_args()

    # Load streaming config from environment
    streaming_config = load_streaming_config()

    # Create producer
    producer = create_streaming_producer(streaming_config)

    # Create loader
    loader = DatasetLoader()

    # Create replay config
    config = make_replay_config(
        dataset=args.dataset,
        speed=args.speed,
        batch_size=args.batch_size,
        max_events=args.max_events,
        deal_id_prefix=args.deal_prefix,
    )

    # Run replay
    runner = DataReplayRunner(producer, loader, config)
    stats = runner.run()

    print(f"Replay complete:")
    print(f"  Events sent: {stats['events_sent']}")
    print(f"  Batches sent: {stats['batches_sent']}")
    print(f"  Elapsed: {stats['elapsed_seconds']:.2f}s")
    print(f"  Throughput: {stats['events_per_second']:.1f} events/sec")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### Phase 5C: Worker Entry Point

**Purpose:** CLI to start the streaming inference worker.

**Location:** `pyproject.toml` + `covenant_radar_api/streaming/main.py`

#### Add to pyproject.toml

```toml
[tool.poetry.scripts]
covenant-rq-worker = "covenant_radar_api.worker_entry:main"
covenant-stream-worker = "covenant_radar_api.streaming.main:main"
```

#### streaming/main.py

```python
"""Entry point for streaming inference worker.

Initializes all dependencies and runs the StreamingWorker.

Usage:
    poetry run covenant-stream-worker
"""

from __future__ import annotations

import sys

from covenant_ml.predictor import load_predictor
from covenant_persistence import (
    create_covenant_repository,
    create_covenant_result_repository,
    create_deal_repository,
    create_measurement_repository,
)

from ..core.config import settings_from_env
from ..integrations.datadog.metrics import create_metrics_client
from ..integrations.google_ai._test_hooks import set_gemini_client
from ..integrations.google_ai.client import create_gemini_client
from ..integrations.google_ai.schemas import make_gemini_config
from .config import load_streaming_config
from .consumer import create_streaming_consumer
from .producer import create_streaming_producer
from .worker import StreamingWorker, make_default_worker_config


def main() -> int:
    """Main entry point for streaming worker.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    # Load configuration
    settings = settings_from_env()
    streaming_config = load_streaming_config()

    # Initialize Gemini client
    gemini_api_key = settings["gemini"]["api_key"]
    gemini_config = make_gemini_config(api_key=gemini_api_key)
    gemini_client = create_gemini_client(gemini_config)
    set_gemini_client(gemini_client)

    # Initialize Kafka
    consumer = create_streaming_consumer(streaming_config)
    producer = create_streaming_producer(streaming_config)

    # Subscribe to measurements topic
    consumer.subscribe([streaming_config["topics"]["measurements"]])

    # Initialize repositories
    db_url = settings["database"]["url"]
    deal_repo = create_deal_repository(db_url)
    covenant_repo = create_covenant_repository(db_url)
    measurement_repo = create_measurement_repository(db_url)
    result_repo = create_covenant_result_repository(db_url)

    # Load ML model
    model_path = settings["app"]["active_model_path"]
    model = load_predictor(model_path)

    # Initialize metrics
    metrics = create_metrics_client(settings["datadog"])

    # Sector/region encoders (from seeding or config)
    sector_encoder = {"Technology": 0, "Finance": 1, "Healthcare": 2}
    region_encoder = {"North America": 0, "Europe": 1, "Asia": 2}

    # Create worker
    worker_config = make_default_worker_config()
    worker = StreamingWorker(
        consumer=consumer,
        producer=producer,
        metrics=metrics,
        model=model,
        deal_repo=deal_repo,
        covenant_repo=covenant_repo,
        measurement_repo=measurement_repo,
        result_repo=result_repo,
        sector_encoder=sector_encoder,
        region_encoder=region_encoder,
        config=worker_config,
    )

    print("Starting streaming worker...")
    print(f"  Consuming from: {streaming_config['topics']['measurements']}")
    print(f"  Producing to: {streaming_config['topics']['predictions']}")
    print(f"  Alert threshold: {worker_config['alert_threshold']}")

    try:
        worker.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        worker.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### Phase 6: Web UI Dashboard

**Purpose:** Display live predictions and alerts from the streaming pipeline.

**Location:** `static/` directory in service root

#### Architecture

Simple HTML + JavaScript that:
1. Polls `/api/predictions/recent` every 2 seconds
2. Polls `/api/alerts/recent` every 2 seconds
3. Updates UI with latest predictions and alerts

#### New API Endpoints Required

Add to `api/routes/streaming.py`:

```python
"""Streaming dashboard endpoints.

Provides recent predictions and alerts for UI consumption.
"""

from __future__ import annotations

from typing import TypedDict

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["streaming"])


class RecentPrediction(TypedDict):
    """Recent prediction for dashboard."""

    deal_id: str
    deal_name: str
    risk_probability: float
    risk_tier: str
    evaluation_status: str
    processed_at: str


class RecentAlert(TypedDict):
    """Recent alert for dashboard."""

    deal_id: str
    severity: str
    risk_probability: float
    gemini_summary: str
    triggered_at: str


@router.get("/predictions/recent")
def get_recent_predictions() -> list[RecentPrediction]:
    """Get recent predictions for dashboard.

    Returns:
        List of recent predictions.
    """
    # TODO: Implement with in-memory cache or Redis
    return []


@router.get("/alerts/recent")
def get_recent_alerts() -> list[RecentAlert]:
    """Get recent alerts for dashboard.

    Returns:
        List of recent alerts.
    """
    # TODO: Implement with in-memory cache or Redis
    return []
```

#### HTML Dashboard (static/index.html)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Covenant Radar - Live Dashboard</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        .grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
        }
        .card h2 {
            color: #00d4ff;
            margin-top: 0;
            font-size: 18px;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
        th { color: #888; font-weight: normal; }
        .risk-low { color: #4caf50; }
        .risk-medium { color: #ff9800; }
        .risk-high { color: #f44336; }
        .risk-critical { color: #ff1744; font-weight: bold; }
        .alert-item {
            background: #1f1f3a;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 4px solid #ff1744;
        }
        .alert-item.warning { border-left-color: #ff9800; }
        .alert-time { color: #666; font-size: 12px; }
        .metrics { display: flex; gap: 20px; margin-bottom: 20px; }
        .metric {
            background: #16213e;
            padding: 15px 25px;
            border-radius: 8px;
        }
        .metric-value { font-size: 32px; color: #00d4ff; }
        .metric-label { color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <h1>Covenant Radar - Live Risk Monitor</h1>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value" id="total-deals">--</div>
            <div class="metric-label">Active Deals</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="high-risk">--</div>
            <div class="metric-label">High Risk</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="alerts-today">--</div>
            <div class="metric-label">Alerts Today</div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Deal Risk Overview</h2>
            <table>
                <thead>
                    <tr>
                        <th>Deal</th>
                        <th>Risk</th>
                        <th>Status</th>
                        <th>Updated</th>
                    </tr>
                </thead>
                <tbody id="predictions-table">
                    <tr><td colspan="4">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Live Alert Feed</h2>
            <div id="alerts-feed">
                <div class="alert-item">Waiting for alerts...</div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = "";

        function getRiskClass(tier) {
            return "risk-" + tier.toLowerCase();
        }

        function formatTime(iso) {
            const d = new Date(iso);
            return d.toLocaleTimeString();
        }

        async function fetchPredictions() {
            try {
                const resp = await fetch(API_BASE + "/api/predictions/recent");
                const data = await resp.json();
                renderPredictions(data);
            } catch (e) {
                console.error("Failed to fetch predictions:", e);
            }
        }

        async function fetchAlerts() {
            try {
                const resp = await fetch(API_BASE + "/api/alerts/recent");
                const data = await resp.json();
                renderAlerts(data);
            } catch (e) {
                console.error("Failed to fetch alerts:", e);
            }
        }

        function renderPredictions(predictions) {
            const tbody = document.getElementById("predictions-table");
            if (predictions.length === 0) {
                tbody.innerHTML = "<tr><td colspan='4'>No predictions yet</td></tr>";
                return;
            }

            let html = "";
            let highRisk = 0;
            for (const p of predictions) {
                const riskClass = getRiskClass(p.risk_tier);
                if (p.risk_tier === "HIGH" || p.risk_tier === "CRITICAL") highRisk++;
                html += `<tr>
                    <td>${p.deal_name || p.deal_id}</td>
                    <td class="${riskClass}">${(p.risk_probability * 100).toFixed(0)}% ${p.risk_tier}</td>
                    <td>${p.evaluation_status}</td>
                    <td>${formatTime(p.processed_at)}</td>
                </tr>`;
            }
            tbody.innerHTML = html;

            document.getElementById("total-deals").textContent = predictions.length;
            document.getElementById("high-risk").textContent = highRisk;
        }

        function renderAlerts(alerts) {
            const feed = document.getElementById("alerts-feed");
            if (alerts.length === 0) {
                feed.innerHTML = "<div class='alert-item'>No alerts yet</div>";
                return;
            }

            let html = "";
            for (const a of alerts) {
                const severityClass = a.severity === "warning" ? "warning" : "";
                html += `<div class="alert-item ${severityClass}">
                    <strong>${a.deal_id}</strong> - ${(a.risk_probability * 100).toFixed(0)}%
                    <p>${a.gemini_summary}</p>
                    <div class="alert-time">${formatTime(a.triggered_at)}</div>
                </div>`;
            }
            feed.innerHTML = html;

            document.getElementById("alerts-today").textContent = alerts.length;
        }

        // Initial fetch
        fetchPredictions();
        fetchAlerts();

        // Poll every 2 seconds
        setInterval(fetchPredictions, 2000);
        setInterval(fetchAlerts, 2000);
    </script>
</body>
</html>
```

---

## 4) Test Requirements

Every new module requires tests achieving 100% statement and branch coverage.

### Test Files Required

```
tests/integrations/google_ai/
├── __init__.py
├── test_schemas.py              # All TypedDicts, make_*, encode/decode
├── test_hooks.py                # FakeGeminiClient, hook functions
└── test_client.py               # GeminiClient with mocked SDK

tests/scripts/replay_data/
├── __init__.py
├── test_types.py                # TypedDict factories
├── test_runner.py               # DataReplayRunner with fakes
└── test_main.py                 # CLI parsing and main()

tests/streaming/
├── test_main.py                 # Worker entry point (new)
```

### Test Pattern

```python
"""Tests for Gemini schemas.

Tests use fake implementations via _test_hooks.py.
No mocks, no weak assertions.
"""

from __future__ import annotations

import pytest

from covenant_radar_api.integrations.google_ai.schemas import (
    make_alert_context,
    encode_alert_context,
    decode_alert_context,
)
from platform_core.json_utils import JSONTypeError


class TestMakeAlertContext:
    """Tests for make_alert_context factory."""

    def test_creates_valid_context(self) -> None:
        """Factory creates context with all fields."""
        context = make_alert_context(
            deal_id="deal-001",
            deal_name="Test Deal",
            borrower="Test Corp",
            sector="Technology",
            risk_probability=0.85,
            evaluation_status="BREACH",
            breaches_count=2,
            covenants_evaluated=5,
        )

        assert context["deal_id"] == "deal-001"
        assert context["deal_name"] == "Test Deal"
        assert context["borrower"] == "Test Corp"
        assert context["sector"] == "Technology"
        assert context["risk_probability"] == 0.85
        assert context["evaluation_status"] == "BREACH"
        assert context["breaches_count"] == 2
        assert context["covenants_evaluated"] == 5


class TestEncodeDecodeAlertContext:
    """Tests for encode/decode round-trip."""

    def test_roundtrip_preserves_data(self) -> None:
        """Encode then decode returns identical data."""
        original = make_alert_context(
            deal_id="deal-001",
            deal_name="Test Deal",
            borrower="Test Corp",
            sector="Technology",
            risk_probability=0.85,
            evaluation_status="BREACH",
            breaches_count=2,
            covenants_evaluated=5,
        )

        encoded = encode_alert_context(original)
        decoded = decode_alert_context(encoded)

        assert decoded == original

    def test_decode_invalid_json_raises(self) -> None:
        """Invalid JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError):
            decode_alert_context("not json")

    def test_decode_missing_field_raises(self) -> None:
        """Missing required field raises JSONTypeError."""
        with pytest.raises(JSONTypeError):
            decode_alert_context('{"deal_id": "x"}')
```

---

## 5) Deployment

### Railway Configuration

Add `Procfile` for Railway:

```
web: poetry run hypercorn 'covenant_radar_api.api.main:create_app()' --bind 0.0.0.0:$PORT
worker: poetry run covenant-stream-worker
```

### Environment Variables Required

```bash
# Database
DATABASE_URL=postgresql://...

# Redis
REDIS_URL=redis://...

# Confluent Cloud
CONFLUENT__BOOTSTRAP_SERVERS=pkc-xxx.us-east-1.aws.confluent.cloud:9092
CONFLUENT__API_KEY=...
CONFLUENT__API_SECRET=...

# Kafka Topics
KAFKA__TOPIC_MEASUREMENTS=covenant.measurements.v1
KAFKA__TOPIC_PREDICTIONS=covenant.predictions.v1
KAFKA__TOPIC_ALERTS=covenant.alerts.v1

# Gemini
GEMINI__API_KEY=...
GEMINI__MODEL=gemini-1.5-flash

# Datadog
DATADOG__ENABLED=true
DATADOG__SERVICE=covenant-radar-api
DATADOG__ENV=production
```

---

## 6) Demo Scenario

### Script for 3-Minute Video

1. **0:00-0:30** - Show dashboard with empty state
2. **0:30-1:00** - Start data replay: `poetry run python -m scripts.replay_data -d taiwan -s fast -m 500`
3. **1:00-1:30** - Show predictions appearing in dashboard
4. **1:30-2:00** - Point out a CRITICAL risk alert with Gemini summary
5. **2:00-2:30** - Show Datadog dashboard with metrics
6. **2:30-3:00** - Explain architecture: CSV → Kafka → ML → Gemini → UI

---

## 7) Devpost Submission

### Title
Covenant Radar: Real-Time Loan Risk Intelligence

### Summary (1-2 sentences)
Covenant Radar streams financial measurements through Confluent Cloud, applies XGBoost/LightGBM ML models for default prediction, generates human-readable alerts via Gemini, and visualizes risk in a live dashboard with Datadog observability.

### What It Does
- Monitors loan covenant compliance in real-time
- Predicts default risk using gradient boosting ML
- Generates natural language alerts with Google Gemini
- Streams data through Confluent Cloud Kafka
- Provides end-to-end observability via Datadog

### How We Built It
- Python 3.11 + FastAPI
- Confluent Cloud for Kafka streaming
- XGBoost + LightGBM for ML inference
- Google Gemini for alert generation
- Datadog for APM and metrics
- 100% test coverage, strict typing

### Challenges
- Integrating three partner technologies coherently
- Maintaining strict type safety with untyped SDKs
- Real-time prediction latency optimization

### What We Learned
- Streaming architecture patterns with Kafka
- LLM integration for domain-specific text generation
- Observability-first development practices

---

## 8) Validation Checklist

### Code Quality

- [ ] `make check` passes
- [ ] 100% statement coverage
- [ ] 100% branch coverage
- [ ] No `Any`, `cast`, `type: ignore`
- [ ] No `.pyi` stub files
- [ ] No `# noqa` comments

### TypedDict Standards

- [ ] All structured data uses TypedDict
- [ ] `make_*` factory for each TypedDict
- [ ] `encode_*` for serialization
- [ ] `decode_*` with `require_*` validation
- [ ] TypeGuard functions where needed

### Testing Standards

- [ ] `_test_hooks.py` for DI in service modules
- [ ] Production sets hooks to real implementations
- [ ] Tests set hooks to fakes
- [ ] No mocks in tests
- [ ] Strong assertions on exact values

### Hackathon Requirements

- [ ] Confluent Cloud streaming works
- [ ] Gemini generates alert text
- [ ] Dashboard displays live predictions
- [ ] Demo video recorded
- [ ] Devpost form submitted

---

*Last updated: December 30, 2025*
