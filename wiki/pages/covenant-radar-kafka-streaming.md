---
title: Covenant Radar's Kafka path — the hardened worker is not the one the container runs
tags: [services, covenant-radar, kafka, streaming, offsets, dead-letter, drift]
related:
  - "[[covenant-radar-domain-protocol]]"
  - "[[covenant-radar-service-architecture]]"
source_paths:
  - services/covenant-radar-api/src/covenant_radar_api/streaming/generic_worker.py
  - services/covenant-radar-api/src/covenant_radar_api/streaming/worker.py
  - services/covenant-radar-api/src/covenant_radar_api/streaming/worker_buffers.py
  - services/covenant-radar-api/src/covenant_radar_api/streaming/worker_events.py
  - services/covenant-radar-api/src/covenant_radar_api/streaming/consumer.py
  - services/covenant-radar-api/src/covenant_radar_api/streaming/config.py
  - services/covenant-radar-api/src/covenant_radar_api/streaming_worker_entry.py
  - services/covenant-radar-api/src/covenant_radar_api/generic_worker_entry.py
  - services/covenant-radar-api/Dockerfile
  - services/covenant-radar-api/docker-compose.yml
  - services/covenant-radar-api/pyproject.toml
  - services/covenant-radar-api/README.md
source_git_blobs:
  "services/covenant-radar-api/src/covenant_radar_api/streaming/generic_worker.py": e840541c8b686903d1a9a209fae17118b94cced3
  "services/covenant-radar-api/src/covenant_radar_api/streaming/worker.py": 062e3c143904221dd2b3a0c17aadbe538095debe
  "services/covenant-radar-api/src/covenant_radar_api/streaming/worker_buffers.py": 8bd4cdcb5eceadfe2df2b208bf8aa8f81965fd00
  "services/covenant-radar-api/src/covenant_radar_api/streaming/worker_events.py": 5608eeb27a8d996567b0d1f595c4d5d64586643e
  "services/covenant-radar-api/src/covenant_radar_api/streaming/consumer.py": 775a26317f5a4861f52f6966dadd6bb74ab93b35
  "services/covenant-radar-api/src/covenant_radar_api/streaming/config.py": 038d77eb6ced250cd562d8f11e0a3f5ff7ffc751
  "services/covenant-radar-api/src/covenant_radar_api/streaming_worker_entry.py": 572e204f048d68e38b64c6778e6c17bcca023fa8
  "services/covenant-radar-api/src/covenant_radar_api/generic_worker_entry.py": 4ab93469361f573434c42b3d9394183e79e59c72
  "services/covenant-radar-api/Dockerfile": 1609785a17251ca8a306cd640c83a117aa099883
  "services/covenant-radar-api/docker-compose.yml": f18715070a89c105837afcff1bcef89b6c315896
  "services/covenant-radar-api/pyproject.toml": 8f420f4aa5698d2bb311e5dba284bb81acbe72ca
  "services/covenant-radar-api/README.md": 34f48ed308cbe2cbffc9e867e2b4a41748c9a90b
provenance:
  - "read from code at API commit 2e0511ee7 on 2026-10-05. The replay and crash-loop consequences below are DERIVED from the code and the compose restart policy, not observed on a broker; no streaming worker was started for this page."
fact_checked: "2026-10-05"
confidence: medium
hubs: [services]
---

# Covenant Radar's Kafka path

There are **two streaming workers**, and the properties the README advertises
belong to the one that is not deployed.[^wiring][^covent][^readme]

| | `StreamingWorker` (covenant) | `GenericStreamingWorker` (weather, esports) |
|---|---|---|
| started by | `python -m covenant_radar_api.streaming_worker_entry` | `covenant-streaming-worker` console script |
| container | none | the Dockerfile's `streaming` target, compose `streaming-worker` |
| unit of work | a buffered `(deal, period)` of measurements | one message |
| undecodable payload | routed to the dead-letter topic, offset advanced | raises out of `process_event` |
| offsets | explicit commit every `commit_interval` (10) messages, excluding anything still buffered | never committed |

Sources: the console script and the container CMD both name
`generic_worker_entry:main`;[^wiring] the covenant entry has no script and
documents itself as `python -m`.[^covent]

## The covenant worker — what was built carefully

`StreamingWorker.run_once` polls one message; the consumer returns either a
`ConsumedMeasurement` or an `UndecodableMessage` rather than raising, and the
second is dead-lettered so its offset can be moved past — without that "the
same message would be redelivered on every restart".[^consumer][^dlq] Ready
buffers are evaluated (rules, then model) and produced, and only then are
their offsets released; commits pass explicit positions because an
argument-less commit "would advance every assigned partition to its consumed
position, acknowledging messages still held in memory".[^worker][^consumer]
Shutdown flushes the producer before the final commit.[^worker] Topics
default to `covenant.{measurements,predictions,alerts,dlq}.v1`.[^config]

## The generic worker — what the container actually runs

`GenericStreamingWorker.run_once` polls, decodes the payload as UTF-8, calls
`process_event`, and produces the prediction and any alert.[^generic] It has
no dead-letter branch and no `commit` call anywhere in the module.[^generic]
Two defaults in `load_streaming_config` then decide what that means:[^config]

- `KAFKA__ENABLE_AUTO_COMMIT` defaults to **false**, so nothing commits the
  consumer group's position at all;
- `KAFKA__AUTO_OFFSET_RESET` defaults to **earliest**.

**Derived consequence, not observed:** with no committed offset, each start of
the worker resumes from the earliest retained message and re-predicts the
whole topic, re-emitting every prediction and alert. And a payload
`decode_and_extract` rejects raises out of `run`, the process exits, compose's
`restart: unless-stopped` restarts it, and it replays to the same message
again.[^compose] The covenant worker's docstrings name exactly this failure as
the reason its dead-letter path exists.[^consumer]

One more default is shared rather than per-domain: the consumer group is
`KAFKA__CONSUMER_GROUP_ID`, default `covenant-radar-api`, for every
domain.[^config]

## What to do with this

A session extending streaming should treat the generic worker as **not yet
at parity** with the covenant one, and the fix as a lift rather than a fork:
the `UndecodableMessage` / explicit-commit machinery is already written and
tested in `consumer.py` and `worker_buffers.py`, keyed to measurement events.
The README's "dead-letter topic so an undecodable payload can't stall the
stream" is true of code no deployment runs.[^readme]

[^wiring]: `services/covenant-radar-api/pyproject.toml:70` (`covenant-streaming-worker = "covenant_radar_api.generic_worker_entry:main"`); `services/covenant-radar-api/Dockerfile:76` (`CMD ["sh", "-c", "exec covenant-streaming-worker"]`); `services/covenant-radar-api/docker-compose.yml` § `streaming-worker` (`target: streaming`).
[^covent]: `services/covenant-radar-api/src/covenant_radar_api/streaming_worker_entry.py` — module docstring, "Usage: poetry run python -m covenant_radar_api.streaming_worker_entry".
[^consumer]: `services/covenant-radar-api/src/covenant_radar_api/streaming/consumer.py` / `UndecodableMessage` (docstring), `StreamingConsumer.poll`, `StreamingConsumer.commit` (docstring).
[^dlq]: `services/covenant-radar-api/src/covenant_radar_api/streaming/worker_buffers.py` / `_dead_letter_undecodable`, `_commit_positions`.
[^worker]: `services/covenant-radar-api/src/covenant_radar_api/streaming/worker.py` / `StreamingWorker.run_once`, `StreamingWorker._process_ready_buffers` ("Only now may these offsets be committed"), `StreamingWorker.shutdown`; `services/covenant-radar-api/src/covenant_radar_api/streaming/worker_events.py` / `make_default_worker_config` (`"commit_interval": 10`).
[^generic]: `services/covenant-radar-api/src/covenant_radar_api/streaming/generic_worker.py` / `GenericStreamingWorker.run_once`, `GenericStreamingWorker.process_event`, `GenericStreamingWorker.shutdown` (flush and close, no commit).
[^config]: `services/covenant-radar-api/src/covenant_radar_api/streaming/config.py` / `load_streaming_config` — `KAFKA__TOPIC_*`, `KAFKA__AUTO_OFFSET_RESET` default `"earliest"`, `KAFKA__ENABLE_AUTO_COMMIT` default `False`, `KAFKA__CONSUMER_GROUP_ID` default `"covenant-radar-api"`.
[^compose]: `services/covenant-radar-api/docker-compose.yml` § `streaming-worker` — `restart: unless-stopped`.
[^readme]: `services/covenant-radar-api/README.md` § "The design decisions worth knowing" ("Kafka on Confluent Cloud ... with a dead-letter topic").
