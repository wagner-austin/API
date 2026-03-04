# Multi-Domain Streaming ML Platform Plan

## Overview

Refactor covenant-radar-api into a **domain-agnostic streaming ML platform** where new domains plug into GenericStreamingWorker via DomainProtocol.

**Goal:** One codebase, multiple domains. Each domain provides schemas, feature extraction, and alert context. The core handles Kafka consume/produce, ML inference, Datadog metrics, and Gemini alerts.

---

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core abstractions (base schemas, protocols, registry, generic worker) | ✅ Complete (2080 tests) |
| 2 | Domain implementations (weather, esports) | Partial — weather exists, esports not started |

---

## Architecture

```
domains/
  base_schemas.py        # BaseInputEventV1, BasePredictionEventV1, BaseAlertEventV1
  protocols.py           # DomainProtocol, FeatureExtractorProtocol
  registry.py            # DomainRegistry
  _test_hooks.py         # DI hooks for domain registry

  weather/               # ✅ Implemented
    schemas.py           # WeatherEventV1 + encode/decode
    features.py          # WeatherFeatureExtractor
    _test_hooks.py       # Weather-specific hooks

  esports/               # ❌ Not started
    schemas.py           # MatchEventV1 + encode/decode
    features.py          # EsportsFeatureExtractor
    domain.py            # EsportsDomain class

streaming/
  generic_worker.py      # GenericStreamingWorker (domain-agnostic)
  _test_hooks_generic_worker.py
```

---

## Phase 2: Domain Implementations (Remaining Work)

### Design Fix (from Phase 2 execution)

The original DomainProtocol had `feature_extractor` property returning `FeatureExtractorProtocol` whose `extract()` takes `BaseInputEventV1`. But real extractors need domain-specific types (e.g. `WeatherEventV1`, `MatchEventV1`). No cast-free adapter can bridge this.

**Fix:** Replace `decode_input_event()` + `feature_extractor.extract()` with a single `decode_and_extract(payload: str) -> tuple[BaseInputEventV1, NDArray[np.float64]]`. Each domain decodes to its own type internally.

### Step 1: Refactor DomainProtocol

**Modify:** `src/covenant_radar_api/domains/protocols.py`

Replace `feature_extractor` property + `decode_input_event` with:
- `feature_names: tuple[str, ...]` property
- `n_features: int` property
- `decode_and_extract(payload: str) -> tuple[BaseInputEventV1, NDArray[np.float64]]`

Keep: `config`, `encode_prediction_event`, `generate_alert_context`.

### Step 2: Update GenericStreamingWorker

**Modify:** `src/covenant_radar_api/streaming/generic_worker.py`

Change `process_event` to call `self._domain.decode_and_extract(payload)` instead of separate decode + extract.

### Step 3: Extract Weather Test Fixtures (DRY)

**New:** `tests/domains/weather/_test_weather_fixtures.py` — extract shared state builders from `test_features.py`.

### Step 4: Implement WeatherDomain

**New:** `src/covenant_radar_api/domains/weather/domain.py`

`WeatherDomain` class implementing the refactored `DomainProtocol`. Factory: `make_weather_domain()`.

### Step 5: Implement Esports Domain

**New files:**
- `src/covenant_radar_api/domains/esports/__init__.py`
- `src/covenant_radar_api/domains/esports/schemas.py` — MatchEventV1 (match_id, game_number, game_time_seconds, blue/red kills/gold/towers/dragons/barons) + encode/decode
- `src/covenant_radar_api/domains/esports/features.py` — EsportsFeatureExtractor (12 features: kill_diff, gold_diff, gold_diff_per_minute, tower_diff, dragon_diff, baron_diff, blue_kill_ratio, blue_gold_ratio, game_time_minutes, blue_objectives, red_objectives, objective_diff)
- `src/covenant_radar_api/domains/esports/domain.py` — EsportsDomain class
- `src/covenant_radar_api/domains/esports/_test_hooks.py`
- Tests for all of the above

### Verification

`make check` after each step. All files: mypy strict, ruff, 100% stmt+branch coverage, Google docstrings.

---

*Last updated: March 2026*
