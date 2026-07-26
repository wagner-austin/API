# Multi-Domain Streaming ML Platform Plan

## Overview

Refactor covenant-radar-api into a **domain-agnostic streaming ML platform** where new domains plug into GenericStreamingWorker via DomainProtocol.

**Goal:** One codebase, multiple domains. Each domain provides schemas, feature extraction, and alert context. The core handles Kafka consume/produce, ML inference, Datadog metrics, and Gemini alerts.

---

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core abstractions (base schemas, protocols, registry, generic worker) | ✅ Complete |
| 2 | Weather domain + deployment | ✅ Complete — runs on a real broker, verified on real ERA5 observations |
| 3 | Esports domain | ✅ Complete — runs on a real broker, verified on real competitive matches |

### Real-data verification

Both domains have been run end to end on real observations through the
production entry point, not just on fixtures.

**Weather** — ERA5 daily maximum temperature, Phoenix and Irvine,
1990-2024. Fitted on 1990-2015, streamed the held-out 2016-2024 summers.
Out-of-sample exceedance of the 95th-percentile threshold was 9.78% and
13.16% against the 5% a stationary climate implies, and held-out AUC on
next-day extremes was 0.9118.

**Esports** — 7,620 competitive League of Legends matches (Oracle's Elixir
export via Kaggle `chuckephron/leagueoflegends`, CC0, 2014-2018). Snapshots
reconstructed at minutes 10/15/20/25 from per-minute gold plus timestamped
kill, turret, dragon and baron events; the label is the real match result.
Split temporally and by match: fit on 2014-2016, evaluate on 2017-2018, so
no match contributes to both. Held-out AUC 0.8163 over 14,057 snapshots,
rising with game time (0.7112 at minute 10 to 0.8921 at minute 25) as a win
probability should. The 0.5435 blue-side base rate matches the known
historical blue advantage, which is the check that the sides are not
transposed.

Both datasets were validated before use, not merely downloaded: cross-column
identities (`golddiff == goldblue - goldred`, exact over 282,036 values),
monotonicity of cumulative counters, rule-fixed bounds, event timestamps
inside the match, and a leakage check that no snapshot counts an event
after its own timestamp. Two checks initially failed. One was a defect in
the check — 37 match ids are Riot's Japanese regional match-history URLs,
which carry no `gameHash`. The other was real: 8 of 7,620 matches (all
2015, all in the training split) recorded `FOUNTAIN_TURRET` kills beyond
the structures that exist. Turret counts now use an allowlist of the eleven
real turrets, and 14 exactly-duplicated event rows are collapsed. Held-out
AUC moved 0.8167 to 0.8163 under the correction.

Caveat worth keeping: later minutes describe longer games, because a
snapshot at minute 25 only exists for matches that reached it. That
conditions the population on game length, which is a future fact. It does
not leak the winner — blue's win rate across those populations is flat at
0.5441/0.5441/0.5438/0.5418 — but minute-25 numbers are not drawn from the
same match population as minute-10 numbers.

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

  esports/               # ✅ Implemented
    schemas.py           # MatchEventV1 + encode/decode
    features.py          # EsportsFeatureExtractor
    domain.py            # EsportsDomain + make_esports_domain
    _test_hooks.py       # No hookable dependencies; the extractor is pure

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

### Step 5: Implement Esports Domain — ✅ done

**New files:**
- `src/covenant_radar_api/domains/esports/__init__.py`
- `src/covenant_radar_api/domains/esports/schemas.py` — MatchEventV1 (match_id, game_number, game_time_seconds, blue/red kills/gold/towers/dragons/barons) + encode/decode
- `src/covenant_radar_api/domains/esports/features.py` — EsportsFeatureExtractor (12 features: kill_diff, gold_diff, gold_diff_per_minute, tower_diff, dragon_diff, baron_diff, blue_kill_ratio, blue_gold_ratio, game_time_minutes, blue_objectives, red_objectives, objective_diff)
- `src/covenant_radar_api/domains/esports/domain.py` — EsportsDomain class
- `src/covenant_radar_api/domains/esports/_test_hooks.py`
- Tests for all of the above

The core needed no change to accept it, which is what the protocol was for.
Every count is an integer, so a snapshot round-trips through JSON exactly;
gold as a float would have made the round trip approximate for no gain.

### Design Fix (from Step 5 execution): lazy domain registration — ✅ done

`DomainRegistry` held built domains, so `build_domain_registry()` had to
construct every domain to offer any. Weather reads a fitted state and a
station map off disk, so an esports-only deployment would have failed at
startup demanding `WEATHER__STATE_PATH` for a domain it never runs.

The registry now holds factories and runs them in `get()`, mirroring
`BackendRegistration` in the two ML registries. Registration reads no
configuration; a domain's requirements are read only when it is selected.
Verified by running the esports smoke with the weather variables unset.

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
