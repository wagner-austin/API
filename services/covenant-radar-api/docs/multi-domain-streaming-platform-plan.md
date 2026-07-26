# Multi-Domain Streaming ML Platform Plan

## Overview

Refactor covenant-radar-api into a **domain-agnostic streaming ML platform** where new domains plug into GenericStreamingWorker via DomainProtocol.

**Goal:** One codebase, multiple domains. Each domain provides schemas, feature extraction, and alert context. The core handles Kafka consume/produce, ML inference, Datadog metrics, and Gemini alerts.

---

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core abstractions (base schemas, protocols, registry, generic worker) | ✅ Complete |
| 2 | Weather domain + deployment | ✅ Complete — runs on a real broker |
| 3 | Esports domain | Not started |

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
    domain.py            # WeatherDomain + make_weather_domain
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

### Design Fix (from Phase 2 execution) — ✅ done

The original DomainProtocol had `feature_extractor` property returning `FeatureExtractorProtocol` whose `extract()` takes `BaseInputEventV1`. But real extractors need domain-specific types (e.g. `WeatherEventV1`, `MatchEventV1`). No cast-free adapter can bridge this.

**Fix:** Replace `decode_input_event()` + `feature_extractor.extract()` with a single `decode_and_extract(payload: str) -> tuple[BaseInputEventV1, NDArray[np.float64]]`. Each domain decodes to its own type internally.

`FeatureExtractorProtocol` was deleted with the change: once the domain owns the combined step, nothing implements or consumes it.

### Step 1: Refactor DomainProtocol — ✅ done

**Modify:** `src/covenant_radar_api/domains/protocols.py`

Replace `feature_extractor` property + `decode_input_event` with:
- `feature_names: tuple[str, ...]` property
- `n_features: int` property
- `decode_and_extract(payload: str) -> tuple[BaseInputEventV1, NDArray[np.float64]]`

Keep: `config`, `encode_prediction_event`, `generate_alert_context`.

### Step 2: Update GenericStreamingWorker — ✅ done

**Modify:** `src/covenant_radar_api/streaming/generic_worker.py`

Change `process_event` to call `self._domain.decode_and_extract(payload)` instead of separate decode + extract.

### Step 3: Extract Weather Test Fixtures (DRY) — ✅ done

**New:** `tests/domains/weather/_test_weather_fixtures.py` — extract shared state builders from `test_features.py`.

### Step 4: Implement WeatherDomain — ✅ done

**New:** `src/covenant_radar_api/domains/weather/domain.py`

`WeatherDomain` class implementing the refactored `DomainProtocol`. Factory: `make_weather_domain()`.

### Step 5: Implement Esports Domain — not started

**New files:**
- `src/covenant_radar_api/domains/esports/__init__.py`
- `src/covenant_radar_api/domains/esports/schemas.py` — MatchEventV1 (match_id, game_number, game_time_seconds, blue/red kills/gold/towers/dragons/barons) + encode/decode
- `src/covenant_radar_api/domains/esports/features.py` — EsportsFeatureExtractor (12 features: kill_diff, gold_diff, gold_diff_per_minute, tower_diff, dragon_diff, baron_diff, blue_kill_ratio, blue_gold_ratio, game_time_minutes, blue_objectives, red_objectives, objective_diff)
- `src/covenant_radar_api/domains/esports/domain.py` — EsportsDomain class
- `src/covenant_radar_api/domains/esports/_test_hooks.py`
- Tests for all of the above

### Deployment — ✅ done

The worker is reachable, not just built:

- `covenant-streaming-worker` console script -> `generic_worker_entry:main`
- `streaming` Dockerfile target; one image serves every domain, since
  `STREAMING__DOMAIN` selects which one runs
- `streaming-worker` service in the covenant-radar compose file
- `platform-kafka` (KRaft, single node) in the root compose, so the stack has
  a broker; `CONFLUENT__SECURITY_PROTOCOL` selects PLAINTEXT for it or
  SASL_SSL for Confluent Cloud
- `WEATHER__STATE_PATH` and `WEATHER__STATION_MAP_PATH` supply the fitted
  seasonal state, which comes from training and cannot be derived at startup

`STREAMING__ENABLED` still defaults to false, so nothing starts consuming
until a deployment opts in.

### Verification

`make check` after each step. All files: mypy strict, ruff, 100% stmt+branch
coverage, Google docstrings.

Verified against a real broker: two observations produced to
`weather.observations.v1`, one extreme and one ordinary. The extreme scored
0.9945 and raised a critical alert; the ordinary scored 0.0073 and raised
none. Both prediction events carried `entity_id=station-a`, confirming the
station_id -> entity_id mapping, and the configured model version.

---

*Last updated: July 2026*
