---
title: Adding a risk domain to Covenant Radar is a registration — and covenant itself is not one
tags: [services, covenant-radar, domains, protocol, registry, streaming, drift]
related:
  - "[[covenant-radar-kafka-streaming]]"
  - "[[covenant-radar-service-architecture]]"
  - "[[covenant-radar-backend-registry]]"
source_paths:
  - services/covenant-radar-api/src/covenant_radar_api/domains/protocols.py
  - services/covenant-radar-api/src/covenant_radar_api/domains/registry.py
  - services/covenant-radar-api/src/covenant_radar_api/generic_worker_entry.py
  - services/covenant-radar-api/src/covenant_radar_api/domains/weather/domain.py
  - services/covenant-radar-api/src/covenant_radar_api/domains/esports/domain.py
  - services/covenant-radar-api/src/covenant_radar_api/domains/weather/features.py
  - services/covenant-radar-api/src/covenant_radar_api/domains/esports/features.py
  - services/covenant-radar-api/src/covenant_radar_api/streaming_worker_entry.py
  - services/covenant-radar-api/README.md
source_git_blobs:
  "services/covenant-radar-api/src/covenant_radar_api/domains/protocols.py": 0295a0aa16ee23e087c572c18d3087de4ebdae74
  "services/covenant-radar-api/src/covenant_radar_api/domains/registry.py": a0ca30e34f31a2cb6b7a4e655d88d5a96daccc0d
  "services/covenant-radar-api/src/covenant_radar_api/generic_worker_entry.py": 4ab93469361f573434c42b3d9394183e79e59c72
  "services/covenant-radar-api/src/covenant_radar_api/domains/weather/domain.py": 8a270a58250970a0e00b7a116befd7d360c23f89
  "services/covenant-radar-api/src/covenant_radar_api/domains/esports/domain.py": b140a06032dfb7244c88af7f8629246b84ca6302
  "services/covenant-radar-api/src/covenant_radar_api/domains/weather/features.py": e970927608df339b4749078cd9ae60f1536f2204
  "services/covenant-radar-api/src/covenant_radar_api/domains/esports/features.py": 99e525ca49fd2fd92cbf85f251a912fad1c7fb5a
  "services/covenant-radar-api/src/covenant_radar_api/streaming_worker_entry.py": 572e204f048d68e38b64c6778e6c17bcca023fa8
  "services/covenant-radar-api/README.md": 34f48ed308cbe2cbffc9e867e2b4a41748c9a90b
provenance:
  - "read from code at API commit 2e0511ee7 on 2026-10-05; 'two domains are registered' is the body of build_domain_registry, and 'no covenant package' is the directory listing of src/covenant_radar_api/domains/ (esports/, weather/, plus the shared modules)"
fact_checked: "2026-10-05"
confidence: high
hubs: [services]
---

# Adding a risk domain is a registration — and covenant is not one

The README's leading claim is that Covenant Radar generalised from loan
covenants to "multi-domain risk prediction behind one protocol", where
"adding a risk domain is a *registration*, not a fork", and that "three ship
today — `covenant`, `weather`, `esports` — each its own package".[^readme]
The first two sentences are true of the code. **The third is not: two
domains implement the protocol, and covenant runs on a separate,
covenant-specific worker.**

## What a domain is

`DomainProtocol` is five members and nothing else:[^proto]

- `config` → `DomainConfig`: name, display name, the three Kafka topics
  (input, prediction, alert) and the alert threshold;
- `feature_names` / `n_features`;
- `decode_and_extract(payload) -> (BaseInputEventV1, NDArray[float64])`;
- `encode_prediction_event(event)`;
- `generate_alert_context(entity_id, prediction_value)` — the strings the
  Gemini alert prompt is built from.

**Decoding and feature extraction are deliberately one method.** A domain's
extractor reads its own event type (`WeatherEventV1`, not the base), and a
protocol method declared to take the base type cannot be satisfied by one
taking a narrower type; two methods forced a cast at the seam, one method lets
each domain decode to its own type internally and hand back only the base
event.[^proto] Anyone tempted to "clean this up" into decode + extract will
reintroduce the cast this design removed.

## How one is registered

`DomainRegistry` holds **factories, not instances**: `register(name, factory)`
refuses a duplicate name, and `get(name)` builds the domain at lookup and
raises `KeyError` naming the available domains.[^reg] That laziness is
load-bearing — weather needs a fitted seasonal state and a station map off
disk (`WEATHER__STATE_PATH`, `WEATHER__STATION_MAP_PATH`), and an eager
registry would demand those files from a deployment running only esports.[^entry]

`build_domain_registry` is the one place domains are wired, and it registers
exactly two:[^entry]

| domain | input topic | features | alert threshold |
|---|---|---|---|
| `weather` | `weather.observations.v1` | 5 — anomaly, hot/cold excess, hot/cold-extreme flags | 0.80 |
| `esports` | `esports.match_state.v1` | 12 — kill/gold/tower/dragon/baron diffs, ratios, game time, objectives | 0.85 |

Each domain's prediction and alert topics follow the same `<domain>.*.v1`
naming.[^weather][^esports] `STREAMING__DOMAIN` selects one at start-up,
defaulting to `weather`.[^entry]

**So adding a domain is:** a package under `domains/` implementing the five
members, plus one `registry.register(...)` line in `build_domain_registry`.
The streaming worker, the topic subscription and the alert path need no change
— the worker subscribes to `domain.config["input_topic"]` itself.[^entry][^proto]

## Where covenant actually is

There is no `domains/covenant/`. Covenant's streaming path is
`streaming_worker_entry.py` → `StreamingWorker`, which buffers measurements by
`(deal_id, period_start, period_end)`, queries PostgreSQL for the deal and its
covenants, runs the deterministic rules, then predicts.[^cov] That shape —
stateful, database-backed, multi-message aggregation — does not fit a protocol
whose unit is one payload in, one feature vector out, which is the likely
reason it was never ported. It also has no console script and is not what the
container runs; [[covenant-radar-kafka-streaming]] covers what that means.

The durable fix for the README is the same as the backend count in
[[covenant-radar-backend-registry]]: derive the sentence from
`build_domain_registry().list_names()` rather than restate it.[^reg][^readme]

[^readme]: `services/covenant-radar-api/README.md` § "Covenant Radar API" (opening paragraph) and § "The design decisions worth knowing" ("A domain is a package, not a branch").
[^proto]: `services/covenant-radar-api/src/covenant_radar_api/domains/protocols.py` / `DomainProtocol`, `DomainConfig`, and the module docstring's paragraph beginning "Decoding and feature extraction are one operation".
[^reg]: `services/covenant-radar-api/src/covenant_radar_api/domains/registry.py` / `DomainRegistry.register`, `DomainRegistry.get`.
[^entry]: `services/covenant-radar-api/src/covenant_radar_api/generic_worker_entry.py` / `build_domain_registry` (two `registry.register` calls), `build_dependencies` (`_parse_str("STREAMING__DOMAIN", "weather")`).
[^weather]: `services/covenant-radar-api/src/covenant_radar_api/domains/weather/domain.py` / `WEATHER_INPUT_TOPIC`, `WEATHER_ALERT_THRESHOLD`; `services/covenant-radar-api/src/covenant_radar_api/domains/weather/features.py` / `WEATHER_FEATURE_NAMES`.
[^esports]: `services/covenant-radar-api/src/covenant_radar_api/domains/esports/domain.py` / `ESPORTS_INPUT_TOPIC`, `ESPORTS_ALERT_THRESHOLD`; `services/covenant-radar-api/src/covenant_radar_api/domains/esports/features.py` / `ESPORTS_FEATURE_NAMES`.
[^cov]: `services/covenant-radar-api/src/covenant_radar_api/streaming_worker_entry.py` (module docstring, "Usage: poetry run python -m covenant_radar_api.streaming_worker_entry") and `_create_worker`.
