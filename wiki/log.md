# Wiki Operation Log

Append-only. Log structural operations (new hubs, decomposition, audits, cleanups). Routine page edits don't need a log entry — git history covers those.

## [2026-07-06] init | api monorepo wiki scaffolded
Hubs created: services, clients, libs, infrastructure
Notes: initial scaffold via /wiki-init. Empty pages/ — content added as subsystems get documented.

## [2026-07-07] first-batch | 3 starter pages
Pages written: monorepo-discipline, platform-workers-rq-pattern, service-port-map
Hubs updated: services (+1), libs (+1), infrastructure (+1)
Notes: all pages audited claim-by-claim against the code before landing — Redis client factory names, RQ harness helpers, monorepo-guards.toml location, service port assignments all verified in the source. Skipped writing more api pages this batch because deeper subsystem context (Kafka streaming in covenant-radar-api, Kohya-ss backend in Art-Trainer, MMS-LID in platform_langid) would require reading service internals; those pages should be written by someone who's touched the subsystem code, not paraphrased from READMEs.
