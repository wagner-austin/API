# Multi-Domain Streaming ML Platform Plan

## Overview

Refactor covenant-radar-api into a **domain-agnostic streaming ML platform** that can be adapted for multiple hackathon submissions and use cases.

**Target Hackathons:**

| Hackathon | Deadline | Prize | Domain Fit |
|-----------|----------|-------|------------|
| AI Partner Catalyst | Dec 31, 2025 | $75K | Fintech (covenant-radar) |
| LMA EDGE | Jan 14, 2026 | $25K | Fintech/Blockchain |
| Tableau | Jan 12, 2026 | $45K | Data Visualization |
| Hex-a-thon | Jan 22, 2026 | $10K | Databases/ML |
| Cloud9 x JetBrains | Feb 03, 2026 | $25K | Gaming/Esports |
| Gemini 3 | Feb 09, 2026 | $100K | Social Good |

**Goal:** One codebase, multiple domains, many submissions.

---

## Current Architecture (Domain-Specific)

```
covenant-radar-api/
├── streaming/
│   ├── schemas.py          # MeasurementEventV1, PredictionEventV1 (covenant-specific)
│   ├── worker.py           # StreamingWorker (covenant evaluation + ML)
│   ├── consumer.py         # Consumes MeasurementEventV1
│   └── producer.py         # Produces PredictionEventV1, AlertEventV1
├── integrations/
│   ├── datadog/            # Observability (domain-agnostic)
│   └── google_ai/          # Gemini (domain-agnostic)
└── api/
    └── routes/             # REST endpoints (domain-specific)
```

**Problem:** Schemas, worker logic, and feature extraction are tightly coupled to loan covenant domain.

---

## Target Architecture (Domain-Agnostic)

```
streaming-ml-platform/
├── core/                           # Domain-agnostic infrastructure
│   ├── streaming/
│   │   ├── base_schemas.py         # BaseEventV1, BasePredictionV1
│   │   ├── base_worker.py          # GenericStreamingWorker
│   │   ├── consumer.py             # Generic consumer
│   │   └── producer.py             # Generic producer
│   ├── ml/
│   │   ├── predictor.py            # Generic prediction interface
│   │   └── feature_protocol.py     # FeatureExtractorProtocol
│   └── integrations/
│       ├── datadog/                # Observability
│       ├── google_ai/              # Gemini
│       └── confluent/              # Kafka config
│
├── domains/                        # Pluggable domain implementations
│   ├── covenant/                   # Loan covenant monitoring
│   │   ├── schemas.py              # MeasurementEventV1, CovenantPredictionV1
│   │   ├── features.py             # CovenantFeatureExtractor
│   │   ├── evaluator.py            # Deterministic covenant rules
│   │   └── worker.py               # CovenantStreamingWorker
│   │
│   ├── esports/                    # LoL match prediction
│   │   ├── schemas.py              # MatchEventV1, WinPredictionV1
│   │   ├── features.py             # EsportsFeatureExtractor
│   │   └── worker.py               # EsportsStreamingWorker
│   │
│   ├── weather/                    # Storm/climate prediction
│   │   ├── schemas.py              # WeatherEventV1, StormPredictionV1
│   │   ├── features.py             # WeatherFeatureExtractor
│   │   └── worker.py               # WeatherStreamingWorker
│   │
│   └── stocks/                     # Price movement prediction
│       ├── schemas.py              # PriceEventV1, TrendPredictionV1
│       ├── features.py             # StockFeatureExtractor
│       └── worker.py               # StockStreamingWorker
│
└── apps/                           # Hackathon-specific deployments
    ├── covenant_radar/             # AI Partner Catalyst, LMA EDGE
    ├── esports_radar/              # Cloud9 x JetBrains
    └── storm_radar/                # Gemini 3 (social good angle)
```

---

## Phase 1: Core Abstractions

### 1.1 Base Event Schema

```python
# core/streaming/base_schemas.py
"""Base event schemas for domain-agnostic streaming.

All domain-specific events inherit from these base types.
Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_str,
)


class BaseInputEventV1(TypedDict, total=True):
    """Base input event consumed from Kafka.

    All domain-specific input events must include these fields.

    Fields:
        type: Event type discriminator (domain.event_name.version).
        event_id: UUID for deduplication.
        entity_id: Primary entity identifier (deal_id, match_id, etc.).
        timestamp: ISO datetime when event was emitted.
    """

    type: str
    event_id: str
    entity_id: str
    timestamp: str


class BasePredictionEventV1(TypedDict, total=True):
    """Base prediction event published to Kafka.

    All domain-specific prediction events must include these fields.

    Fields:
        type: Event type discriminator.
        event_id: UUID for this event.
        entity_id: Primary entity identifier.
        prediction_value: Primary prediction output (probability, score, etc.).
        confidence: Model confidence in prediction (0.0-1.0).
        model_version: Version string of the ML model used.
        latency_ms: Inference latency in milliseconds.
        processed_at: ISO datetime when processing completed.
    """

    type: str
    event_id: str
    entity_id: str
    prediction_value: float
    confidence: float
    model_version: str
    latency_ms: int
    processed_at: str


class BaseAlertEventV1(TypedDict, total=True):
    """Base alert event for high-severity predictions.

    Fields:
        type: Event type discriminator.
        event_id: UUID for this event.
        entity_id: Primary entity identifier.
        alert_type: Category of alert.
        severity: Alert severity level.
        prediction_value: Value that triggered alert.
        gemini_summary: Human-readable summary from Gemini.
        triggered_at: ISO datetime when alert was triggered.
    """

    type: str
    event_id: str
    entity_id: str
    alert_type: str
    severity: Literal["info", "warning", "critical"]
    prediction_value: float
    gemini_summary: str
    triggered_at: str


# Encode/decode functions for base events
def encode_base_input_event(event: BaseInputEventV1) -> str:
    """Serialize base input event to JSON string."""
    return dump_json_str(event)


def decode_base_input_event(payload: str) -> BaseInputEventV1:
    """Parse base input event from JSON string.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated BaseInputEventV1.

    Raises:
        JSONTypeError: If required fields are missing.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    return {
        "type": require_str(decoded, "type"),
        "event_id": require_str(decoded, "event_id"),
        "entity_id": require_str(decoded, "entity_id"),
        "timestamp": require_str(decoded, "timestamp"),
    }
```

### 1.2 Feature Extractor Protocol

```python
# core/ml/feature_protocol.py
"""Protocol for domain-specific feature extraction.

Each domain implements this protocol to transform raw events
into feature vectors for ML prediction.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray


InputEvent = TypeVar("InputEvent", contravariant=True)


class FeatureExtractorProtocol(Protocol[InputEvent]):
    """Protocol for extracting ML features from domain events.

    Each domain implements this to transform its specific event
    types into numeric feature vectors.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return ordered tuple of feature names."""
        ...

    @property
    def n_features(self) -> int:
        """Return number of features produced."""
        ...

    def extract(self, event: InputEvent) -> NDArray[np.float64]:
        """Extract feature vector from input event.

        Args:
            event: Domain-specific input event.

        Returns:
            1D numpy array of shape (n_features,).
        """
        ...

    def extract_batch(
        self,
        events: list[InputEvent],
    ) -> NDArray[np.float64]:
        """Extract features from multiple events.

        Args:
            events: List of domain-specific input events.

        Returns:
            2D numpy array of shape (n_events, n_features).
        """
        ...
```

### 1.3 Domain Registry

```python
# core/domains/registry.py
"""Registry for pluggable domain implementations.

Domains are registered at startup and selected by configuration.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Protocol, TypedDict


class DomainConfig(TypedDict, total=True):
    """Configuration for a registered domain.

    Fields:
        name: Domain identifier (e.g., "covenant", "esports").
        display_name: Human-readable name.
        input_topic: Kafka topic for input events.
        prediction_topic: Kafka topic for predictions.
        alert_topic: Kafka topic for alerts.
        alert_threshold: Prediction value threshold for alerts.
    """

    name: str
    display_name: str
    input_topic: str
    prediction_topic: str
    alert_topic: str
    alert_threshold: float


class DomainProtocol(Protocol):
    """Protocol that all domain implementations must satisfy."""

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        ...

    @property
    def feature_extractor(self) -> FeatureExtractorProtocol:
        """Return feature extractor for this domain."""
        ...

    def decode_input_event(self, payload: str) -> BaseInputEventV1:
        """Decode domain-specific input event."""
        ...

    def encode_prediction_event(self, event: BasePredictionEventV1) -> str:
        """Encode domain-specific prediction event."""
        ...

    def generate_alert_context(
        self,
        entity_id: str,
        prediction_value: float,
    ) -> dict[str, str]:
        """Generate context for Gemini alert summary."""
        ...


class DomainRegistry:
    """Registry of available domain implementations.

    Thread-safe for reads after construction.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._domains: dict[str, DomainProtocol] = {}

    def register(self, domain: DomainProtocol) -> None:
        """Register a domain implementation.

        Args:
            domain: Domain to register.

        Raises:
            ValueError: If domain name already registered.
        """
        name = domain.config["name"]
        if name in self._domains:
            raise ValueError(f"Domain '{name}' already registered")
        self._domains[name] = domain

    def get(self, name: str) -> DomainProtocol:
        """Get domain by name.

        Args:
            name: Domain identifier.

        Returns:
            Registered DomainProtocol.

        Raises:
            KeyError: If domain not found.
        """
        if name not in self._domains:
            available = ", ".join(sorted(self._domains.keys()))
            raise KeyError(f"Domain '{name}' not found. Available: {available}")
        return self._domains[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered domain names."""
        return tuple(sorted(self._domains.keys()))
```

### 1.4 Generic Streaming Worker

```python
# core/streaming/base_worker.py
"""Generic streaming worker that delegates to domain implementations.

The worker handles Kafka consume/produce and ML inference.
Domain-specific logic is delegated to the DomainProtocol.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import time
import uuid
from typing import TypedDict

from covenant_ml.predictor import predict_probabilities
from covenant_ml.types import PredictorProtocol

from .base_schemas import BaseAlertEventV1, BasePredictionEventV1
from .consumer import StreamingConsumer
from .producer import StreamingProducer
from ..domains.registry import DomainProtocol
from ..integrations.datadog.metrics import MetricsClient
from ..integrations.google_ai.client import GeminiClientProtocol


class GenericWorkerConfig(TypedDict, total=True):
    """Configuration for generic streaming worker.

    Fields:
        model_version: Version string for the ML model.
        batch_size: Max messages to poll per iteration.
        poll_timeout_seconds: Kafka poll timeout.
        commit_interval: Number of messages between commits.
    """

    model_version: str
    batch_size: int
    poll_timeout_seconds: float
    commit_interval: int


class GenericStreamingWorker:
    """Domain-agnostic streaming worker.

    Consumes events from Kafka, extracts features using the domain's
    feature extractor, runs ML prediction, and produces results.

    Domain-specific logic is delegated to the DomainProtocol.
    """

    def __init__(
        self,
        domain: DomainProtocol,
        consumer: StreamingConsumer,
        producer: StreamingProducer,
        model: PredictorProtocol,
        metrics: MetricsClient,
        gemini: GeminiClientProtocol,
        config: GenericWorkerConfig,
    ) -> None:
        """Initialize the generic worker.

        Args:
            domain: Domain implementation for event handling.
            consumer: Kafka consumer.
            producer: Kafka producer.
            model: ML model for predictions.
            metrics: Datadog metrics client.
            gemini: Gemini client for alert summaries.
            config: Worker configuration.
        """
        self._domain = domain
        self._consumer = consumer
        self._producer = producer
        self._model = model
        self._metrics = metrics
        self._gemini = gemini
        self._config = config
        self._running = False

    def process_event(self, payload: str) -> BasePredictionEventV1:
        """Process a single input event and generate prediction.

        Args:
            payload: Raw JSON payload from Kafka.

        Returns:
            Prediction event to publish.
        """
        # Decode using domain-specific decoder
        input_event = self._domain.decode_input_event(payload)
        entity_id = input_event["entity_id"]

        # Extract features using domain's feature extractor
        start_time = time.perf_counter()
        features = self._domain.feature_extractor.extract(input_event)

        # Run ML prediction
        features_2d = features.reshape(1, -1)
        probabilities = predict_probabilities(self._model, features_2d)
        prediction_value = float(probabilities[0])
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Build prediction event
        prediction: BasePredictionEventV1 = {
            "type": f"{self._domain.config['name']}.prediction.v1",
            "event_id": str(uuid.uuid4()),
            "entity_id": entity_id,
            "prediction_value": prediction_value,
            "confidence": self._calculate_confidence(prediction_value),
            "model_version": self._config["model_version"],
            "latency_ms": latency_ms,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Check if alert needed
        threshold = self._domain.config["alert_threshold"]
        if prediction_value >= threshold:
            alert = self._generate_alert(entity_id, prediction_value)
            self._producer.produce(
                topic=self._domain.config["alert_topic"],
                key=entity_id,
                value=self._domain.encode_alert_event(alert),
            )

        # Record metrics
        self._metrics.record_prediction_latency(
            entity_id,
            self._classify_tier(prediction_value),
            float(latency_ms),
        )

        return prediction

    def _calculate_confidence(self, prediction_value: float) -> float:
        """Calculate model confidence from prediction value."""
        # Confidence is higher when prediction is further from 0.5
        return abs(prediction_value - 0.5) * 2

    def _classify_tier(self, prediction_value: float) -> str:
        """Classify prediction into tier for metrics."""
        if prediction_value >= 0.8:
            return "CRITICAL"
        if prediction_value >= 0.5:
            return "HIGH"
        if prediction_value >= 0.25:
            return "MEDIUM"
        return "LOW"

    def _generate_alert(
        self,
        entity_id: str,
        prediction_value: float,
    ) -> BaseAlertEventV1:
        """Generate alert with Gemini summary."""
        # Get domain-specific context for Gemini
        context = self._domain.generate_alert_context(entity_id, prediction_value)

        # Generate summary with Gemini
        gemini_response = self._gemini.generate_summary(context)

        severity: Literal["info", "warning", "critical"]
        if prediction_value >= 0.9:
            severity = "critical"
        elif prediction_value >= 0.8:
            severity = "warning"
        else:
            severity = "info"

        return {
            "type": f"{self._domain.config['name']}.alert.v1",
            "event_id": str(uuid.uuid4()),
            "entity_id": entity_id,
            "alert_type": "high_risk",
            "severity": severity,
            "prediction_value": prediction_value,
            "gemini_summary": gemini_response["text"],
            "triggered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> None:
        """Run the worker main loop."""
        self._running = True
        domain_config = self._domain.config

        self._consumer.subscribe([domain_config["input_topic"]])

        while self._running:
            message = self._consumer.poll(self._config["poll_timeout_seconds"])
            if message is None:
                continue

            prediction = self.process_event(message["payload"])

            self._producer.produce(
                topic=domain_config["prediction_topic"],
                key=prediction["entity_id"],
                value=self._domain.encode_prediction_event(prediction),
            )

    def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self._consumer.close()
        self._producer.flush(timeout_seconds=10.0)
```

---

## Phase 2: Domain Implementations

### 2.1 Covenant Domain (Existing)

```python
# domains/covenant/domain.py
"""Covenant domain implementation for loan monitoring.

Wraps existing covenant-radar functionality into the domain protocol.
"""

from __future__ import annotations

from core.domains.registry import DomainConfig, DomainProtocol
from core.ml.feature_protocol import FeatureExtractorProtocol
from core.streaming.base_schemas import BaseInputEventV1

from .features import CovenantFeatureExtractor
from .schemas import (
    MeasurementEventV1,
    decode_measurement_event,
    encode_prediction_event,
)


class CovenantDomain:
    """Loan covenant monitoring domain."""

    def __init__(self) -> None:
        """Initialize covenant domain."""
        self._config: DomainConfig = {
            "name": "covenant",
            "display_name": "Loan Covenant Monitor",
            "input_topic": "covenant.measurements.v1",
            "prediction_topic": "covenant.predictions.v1",
            "alert_topic": "covenant.alerts.v1",
            "alert_threshold": 0.8,
        }
        self._feature_extractor = CovenantFeatureExtractor()

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        return self._config

    @property
    def feature_extractor(self) -> FeatureExtractorProtocol:
        """Return covenant feature extractor."""
        return self._feature_extractor

    def decode_input_event(self, payload: str) -> BaseInputEventV1:
        """Decode measurement event."""
        event = decode_measurement_event(payload)
        return {
            "type": event["type"],
            "event_id": event["event_id"],
            "entity_id": event["deal_id"],
            "timestamp": event["timestamp"],
        }

    def encode_prediction_event(self, event: BasePredictionEventV1) -> str:
        """Encode covenant prediction event."""
        return encode_prediction_event(event)

    def generate_alert_context(
        self,
        entity_id: str,
        prediction_value: float,
    ) -> dict[str, str]:
        """Generate context for Gemini alert."""
        return {
            "domain": "loan_covenant",
            "entity_type": "deal",
            "entity_id": entity_id,
            "prediction_type": "default_risk",
            "prediction_value": f"{prediction_value:.1%}",
            "action": "review loan terms and borrower financials",
        }
```

### 2.2 Esports Domain (New)

```python
# domains/esports/domain.py
"""Esports domain implementation for match prediction.

Predicts match outcomes based on player/team statistics.
"""

from __future__ import annotations

from core.domains.registry import DomainConfig, DomainProtocol
from core.ml.feature_protocol import FeatureExtractorProtocol
from core.streaming.base_schemas import BaseInputEventV1

from .features import EsportsFeatureExtractor
from .schemas import (
    MatchEventV1,
    decode_match_event,
    encode_win_prediction_event,
)


class EsportsDomain:
    """League of Legends match prediction domain."""

    def __init__(self) -> None:
        """Initialize esports domain."""
        self._config: DomainConfig = {
            "name": "esports",
            "display_name": "LoL Match Predictor",
            "input_topic": "esports.matches.v1",
            "prediction_topic": "esports.predictions.v1",
            "alert_topic": "esports.alerts.v1",
            "alert_threshold": 0.85,  # High confidence predictions
        }
        self._feature_extractor = EsportsFeatureExtractor()

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        return self._config

    @property
    def feature_extractor(self) -> FeatureExtractorProtocol:
        """Return esports feature extractor."""
        return self._feature_extractor

    def decode_input_event(self, payload: str) -> BaseInputEventV1:
        """Decode match event."""
        event = decode_match_event(payload)
        return {
            "type": event["type"],
            "event_id": event["event_id"],
            "entity_id": event["match_id"],
            "timestamp": event["timestamp"],
        }

    def generate_alert_context(
        self,
        entity_id: str,
        prediction_value: float,
    ) -> dict[str, str]:
        """Generate context for Gemini commentary."""
        return {
            "domain": "esports",
            "entity_type": "match",
            "entity_id": entity_id,
            "prediction_type": "win_probability",
            "prediction_value": f"{prediction_value:.1%}",
            "action": "provide analyst-style commentary on the prediction",
        }
```

### 2.3 Esports Schemas

```python
# domains/esports/schemas.py
"""Kafka event schemas for esports domain.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)


MatchEventType = Literal["esports.match.v1"]
WinPredictionEventType = Literal["esports.prediction.v1"]


class MatchEventV1(TypedDict, total=True):
    """Match statistics event from Kafka.

    Fields:
        type: Event type discriminator.
        event_id: UUID for deduplication.
        match_id: Unique match identifier.
        game_number: Game number in series (1-5).
        timestamp: ISO datetime when stats were recorded.
        game_time_seconds: Current game time.
        blue_team: Blue side team name.
        red_team: Red side team name.
        blue_kills: Blue team total kills.
        red_kills: Red team total kills.
        blue_gold: Blue team total gold.
        red_gold: Red team total gold.
        blue_towers: Blue team towers destroyed.
        red_towers: Red team towers destroyed.
        blue_dragons: Blue team dragons taken.
        red_dragons: Red team dragons taken.
        blue_barons: Blue team barons taken.
        red_barons: Red team barons taken.
    """

    type: MatchEventType
    event_id: str
    match_id: str
    game_number: int
    timestamp: str
    game_time_seconds: int
    blue_team: str
    red_team: str
    blue_kills: int
    red_kills: int
    blue_gold: int
    red_gold: int
    blue_towers: int
    red_towers: int
    blue_dragons: int
    red_dragons: int
    blue_barons: int
    red_barons: int


class WinPredictionEventV1(TypedDict, total=True):
    """Match win prediction event.

    Fields:
        type: Event type discriminator.
        event_id: UUID for this event.
        match_id: Match identifier.
        blue_win_probability: Probability blue team wins (0.0-1.0).
        predicted_winner: Team name predicted to win.
        confidence: Model confidence.
        model_version: ML model version.
        latency_ms: Inference latency.
        processed_at: ISO datetime of processing.
    """

    type: WinPredictionEventType
    event_id: str
    match_id: str
    blue_win_probability: float
    predicted_winner: str
    confidence: float
    model_version: str
    latency_ms: int
    processed_at: str


def make_match_event(
    *,
    event_id: str,
    match_id: str,
    game_number: int,
    timestamp: str,
    game_time_seconds: int,
    blue_team: str,
    red_team: str,
    blue_kills: int,
    red_kills: int,
    blue_gold: int,
    red_gold: int,
    blue_towers: int,
    red_towers: int,
    blue_dragons: int,
    red_dragons: int,
    blue_barons: int,
    red_barons: int,
) -> MatchEventV1:
    """Create a match event."""
    return {
        "type": "esports.match.v1",
        "event_id": event_id,
        "match_id": match_id,
        "game_number": game_number,
        "timestamp": timestamp,
        "game_time_seconds": game_time_seconds,
        "blue_team": blue_team,
        "red_team": red_team,
        "blue_kills": blue_kills,
        "red_kills": red_kills,
        "blue_gold": blue_gold,
        "red_gold": red_gold,
        "blue_towers": blue_towers,
        "red_towers": red_towers,
        "blue_dragons": blue_dragons,
        "red_dragons": red_dragons,
        "blue_barons": blue_barons,
        "red_barons": red_barons,
    }


def encode_match_event(event: MatchEventV1) -> str:
    """Serialize match event to JSON string."""
    return dump_json_str(event)


def decode_match_event(payload: str) -> MatchEventV1:
    """Parse match event from JSON string.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated MatchEventV1.

    Raises:
        JSONTypeError: If required fields are missing.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    return {
        "type": "esports.match.v1",
        "event_id": require_str(decoded, "event_id"),
        "match_id": require_str(decoded, "match_id"),
        "game_number": require_int(decoded, "game_number"),
        "timestamp": require_str(decoded, "timestamp"),
        "game_time_seconds": require_int(decoded, "game_time_seconds"),
        "blue_team": require_str(decoded, "blue_team"),
        "red_team": require_str(decoded, "red_team"),
        "blue_kills": require_int(decoded, "blue_kills"),
        "red_kills": require_int(decoded, "red_kills"),
        "blue_gold": require_int(decoded, "blue_gold"),
        "red_gold": require_int(decoded, "red_gold"),
        "blue_towers": require_int(decoded, "blue_towers"),
        "red_towers": require_int(decoded, "red_towers"),
        "blue_dragons": require_int(decoded, "blue_dragons"),
        "red_dragons": require_int(decoded, "red_dragons"),
        "blue_barons": require_int(decoded, "blue_barons"),
        "red_barons": require_int(decoded, "red_barons"),
    }
```

### 2.4 Esports Feature Extractor

```python
# domains/esports/features.py
"""Feature extraction for esports match prediction.

Extracts ML features from match statistics events.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.ml.feature_protocol import FeatureExtractorProtocol

from .schemas import MatchEventV1


class EsportsFeatureExtractor:
    """Extract ML features from LoL match statistics."""

    def __init__(self) -> None:
        """Initialize feature extractor."""
        self._feature_names: tuple[str, ...] = (
            "kill_diff",
            "gold_diff",
            "gold_diff_per_minute",
            "tower_diff",
            "dragon_diff",
            "baron_diff",
            "blue_kill_ratio",
            "blue_gold_ratio",
            "game_time_minutes",
            "blue_objectives",
            "red_objectives",
            "objective_diff",
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return ordered tuple of feature names."""
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Return number of features produced."""
        return len(self._feature_names)

    def extract(self, event: MatchEventV1) -> NDArray[np.float64]:
        """Extract feature vector from match event.

        Args:
            event: Match statistics event.

        Returns:
            1D numpy array of shape (n_features,).
        """
        game_time_minutes = event["game_time_seconds"] / 60.0

        # Kill metrics
        kill_diff = event["blue_kills"] - event["red_kills"]
        total_kills = event["blue_kills"] + event["red_kills"]
        blue_kill_ratio = (
            event["blue_kills"] / total_kills if total_kills > 0 else 0.5
        )

        # Gold metrics
        gold_diff = event["blue_gold"] - event["red_gold"]
        gold_diff_per_minute = gold_diff / game_time_minutes if game_time_minutes > 0 else 0.0
        total_gold = event["blue_gold"] + event["red_gold"]
        blue_gold_ratio = (
            event["blue_gold"] / total_gold if total_gold > 0 else 0.5
        )

        # Objective metrics
        tower_diff = event["blue_towers"] - event["red_towers"]
        dragon_diff = event["blue_dragons"] - event["red_dragons"]
        baron_diff = event["blue_barons"] - event["red_barons"]

        blue_objectives = event["blue_towers"] + event["blue_dragons"] + event["blue_barons"]
        red_objectives = event["red_towers"] + event["red_dragons"] + event["red_barons"]
        objective_diff = blue_objectives - red_objectives

        return np.array([
            float(kill_diff),
            float(gold_diff),
            gold_diff_per_minute,
            float(tower_diff),
            float(dragon_diff),
            float(baron_diff),
            blue_kill_ratio,
            blue_gold_ratio,
            game_time_minutes,
            float(blue_objectives),
            float(red_objectives),
            float(objective_diff),
        ], dtype=np.float64)

    def extract_batch(
        self,
        events: list[MatchEventV1],
    ) -> NDArray[np.float64]:
        """Extract features from multiple events.

        Args:
            events: List of match events.

        Returns:
            2D numpy array of shape (n_events, n_features).
        """
        features_list = [self.extract(event) for event in events]
        return np.vstack(features_list)
```

---

## Phase 3: Hackathon Mapping

### 3.1 Domain-to-Hackathon Matrix

| Domain | Dec 31 | Jan 12 | Jan 14 | Jan 22 | Feb 03 | Feb 09 |
|--------|--------|--------|--------|--------|--------|--------|
| **covenant** | AI Partner | Tableau | LMA EDGE | Hex-a-thon | - | - |
| **esports** | - | - | - | - | Cloud9 | Gemini 3 |
| **weather** | - | - | - | - | - | Gemini 3 |

### 3.2 Submission Strategy

**AI Partner Catalyst (Dec 31):**
- Domain: covenant
- Story: Real-time loan risk monitoring
- Required: Confluent + Google Cloud AI (Gemini)

**LMA EDGE (Jan 14):**
- Domain: covenant
- Story: Fintech innovation in lending
- Angle: Blockchain-ready architecture (future integration)

**Cloud9 x JetBrains (Feb 03):**
- Domain: esports
- Story: Real-time esports analytics
- Angle: Gaming + ML + streaming

**Gemini 3 (Feb 09):**
- Domain: esports OR weather
- Story: Social good through accessible analytics
- Angle: Democratizing pro-level analysis

---

## Phase 4: Implementation Order

### 4.1 For Dec 31 (AI Partner Catalyst)

| Task | Priority | Hours |
|------|----------|-------|
| Gemini integration | P0 | 2 |
| Data replay script | P0 | 1 |
| Web UI dashboard | P0 | 2 |
| Worker entry point | P0 | 0.5 |
| Deploy to Railway | P0 | 1 |
| Demo video | P0 | 1 |
| Devpost submission | P0 | 0.5 |

**Total: ~8 hours**

### 4.2 For Jan 14 (LMA EDGE)

| Task | Priority | Hours |
|------|----------|-------|
| Polish covenant domain | P1 | 4 |
| Add visualization endpoints | P1 | 4 |
| Improve UI | P1 | 4 |
| Re-record demo | P1 | 2 |

### 4.3 For Feb 03 (Cloud9)

| Task | Priority | Hours |
|------|----------|-------|
| Implement esports domain | P1 | 8 |
| Download LoL dataset | P1 | 1 |
| Train esports model | P1 | 4 |
| Esports UI | P1 | 4 |
| Demo video | P1 | 2 |

### 4.4 For Feb 09 (Gemini 3)

| Task | Priority | Hours |
|------|----------|-------|
| Pick domain angle | P1 | 1 |
| Polish chosen domain | P1 | 8 |
| Social good framing | P1 | 2 |
| Demo video | P1 | 2 |

---

## Code Standards Checklist

Every new file must satisfy:

- [ ] No `Any` type annotations
- [ ] No `cast()` calls
- [ ] No `type: ignore` comments
- [ ] No `.pyi` stub files
- [ ] No `# noqa` comments
- [ ] TypedDict for all structured data (total=True)
- [ ] `encode_*()` function for each TypedDict
- [ ] `decode_*()` function with `require_*` validation
- [ ] `_test_hooks.py` for DI in service modules
- [ ] Production code sets hooks to real implementations
- [ ] Tests set hooks to fakes (no mocks)
- [ ] Google-style docstrings with Args, Returns, Raises
- [ ] 100% statement coverage
- [ ] 100% branch coverage

---

## Success Metrics

| Hackathon | Goal | Success Criteria |
|-----------|------|------------------|
| AI Partner Catalyst | Submit | Working demo + video |
| LMA EDGE | Place | Top 10 |
| Cloud9 x JetBrains | Place | Top 5 (gaming fit) |
| Gemini 3 | Win | Top 3 ($100K pot) |

---

*Last updated: December 30, 2025*
