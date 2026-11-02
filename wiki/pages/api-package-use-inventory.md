---
title: Which API packages are used, measured 2026-11-01, to pick what to archive
tags: [monorepo, inventory, archive, railway, fleet, infrastructure]
related:
  - "[[service-port-map]]"
  - "[[monorepo-discipline]]"
source_paths:
  - docs/RESEARCH.md
  - tools/hpc3-execution/Makefile
  - services/covenant-radar-api/data/external
source_git_blobs:
  "docs/RESEARCH.md": c7b63ba14027eadb70362b5ec5459e58535f37e9
  "tools/hpc3-execution/Makefile": 87ea0da9f4efa4e161e77d8d62c6ec5f2bcb8af2
provenance:
  - "The archived packages' own files are cited at API da1c233ec, the commit the tag archive/api-packages-2026-11-01 names, since they left HEAD with the archive. tools/fleet/registry/projects/tools/maketools-execution.json is cited at API 5f661ee41, the last commit with its eight install steps, which are the measurement; the archive cut it to four."
fact_checked: "2026-11-01"
confidence: medium
hubs: [infrastructure]
---

# Which API packages are used, measured 2026-11-01

The operator asked on 2026-11-01 whether the monorepo carries old code nobody uses that could be archived.[^ask] This page is that inventory: every top-level package under `services/`, `libs/`, `clients/` and `tools/` at API `adda8c020`, what each is measured to be used by, and a recommendation. **Nothing moved until the operator picked**: the picks were recorded as rulings on board task `a276f973` first, and are below.

## What the size actually is

Two packages hold most of it. `services/covenant-radar-api` tracks 352 MB, 330 MB of it Kaggle loan datasets under `data/external`, which the ClearGBM and covenant_ml research reads, so they are needed but do not belong in git. `libs/instrument_io` tracks 119 MB, mostly instrument fixtures under `tests/fixtures`, and nothing imports it. Every other package is under 45 MB.[^size] Archiving the small services saves little fleet time: each uses 4 to 15 minutes of fleet checks a month. The exception is `tools/maketools-execution`, whose check deploys 8 of the services, at 1052 minutes a month.[^fleet][^mxe]

## How each column was read

- **Deploy**: a GET of `/healthz` on each service's default Railway host. `200` is live. `Railway fallback` is Railway's own 404 with `x-railway-fallback: true`: no service at that host. No Railway credential is on the session host, so a service under another hostname is unread, and DiscordBot serves no HTTP. No API service runs as a container on demure, diphtheria or emerald.[^probe]
- **Imported by**: path dependencies in every package's `pyproject.toml`, `package.json` and `Cargo.toml`, a count when over four. No MCPs package names an API library.[^deps]
- **Last own change**: the newest commit that touches only this package and whose subject is not shared by commits to 3 or more packages. The one-commit-per-folder rule splits a sweep into one commit per package, and this drops those. Guard-driven single-package edits still count, so the date is an upper bound on real use.[^solo]
- **Fleet min/30d**: claimed-to-closed minutes of every fleet job for the package in the 30 days to 2026-11-01.[^fleet]
- **RESEARCH.md**: whether `docs/RESEARCH.md` names the path.[^research]

## The table

One row per package, each column read as the list above says, at API `adda8c020` and the instants named in the footnotes.[^probe][^deps][^solo][^fleet][^size][^research]

| Package | Deploy | Imported by | Last own change | Fleet min/30d | Size | RESEARCH.md | Recommendation |
|---|---|---|---|---|---|---|---|
| `clients/DiscordBot` | none (Railway fallback) | none | 2026-10-08 `8cc3872e6` | 7 | 545 KB | no | archive: no deploy at its default host, no importer, serves no HTTP |
| `clients/NavProbe` | none | none | 2026-10-31 `dda19cac4` | 67 | 1 MB | no | keep: research instrument, own wiki |
| `clients/OrderedKernels` | none | none | 2026-10-30 `f561db26d` | 32 | 671 KB | yes | keep: research |
| `clients/RustedWarfareBot` | none | none | 2026-10-25 `2a6920546` | 85 | 11 MB | yes | keep: research |
| `clients/TankpitBot` | none | none | 2026-10-30 `f3d7367e2` | 362 | 26 MB | yes | keep: research |
| `clients/TankpitBot-colossus-execution` | none | none | 2026-10-18 `3d92aa80b` | 6 | 1 KB | no | keep until board task 05d4f095 settles it |
| `clients/TankpitBot-execution` | none | none | 2026-10-05 `0411bc0c8` | 47 | 1 KB | no | keep: fleet recipe |
| `libs/cleargbm` | library | covenant_ml | 2026-09-26 `bdc043690` | 7 | 4 MB | yes | keep: research |
| `libs/cleargbm_rs` | library | cleargbm, covenant-radar-api | 2026-10-25 `4a59de86e` | 17 | 2 MB | yes | keep: research |
| `libs/covenant_domain` | library | covenant-radar-api, covenant_ml, covenant_persistence | 2025-12-24 `964a7620a` | 7 | 179 KB | no | keep: research |
| `libs/covenant_ml` | library | covenant-radar-api, covenant_nn | 2026-11-01 `ef969f32d` | 992 | 42 MB | yes | keep: research |
| `libs/covenant_nn` | library | covenant-radar-api | 2026-10-07 `916cddf69` | 30 | 11 MB | no | keep: research |
| `libs/covenant_persistence` | library | covenant-radar-api | 2026-08-14 `34dac09b9` | 4 | 241 KB | no | keep: research |
| `libs/instrument_io` | library | none | 2026-10-26 `d90bca755` | 55 | 119 MB | no | archive: no importer anywhere, 119 MB |
| `libs/monorepo_guards` | library | 5 | 2026-10-08 `42471411d` | 36 | 979 KB | no | keep: imported |
| `libs/platform_calendar` | library | opportunity-radar-api | 2026-10-18 `7ce9c2d81` | 9 | 496 KB | no | keep: imported |
| `libs/platform_codebase` | library | github-stats-api, opportunity-radar-api, platform_devpost, platform_kaggle | 2026-09-26 `b64cad69f` | 6 | 200 KB | no | keep: imported |
| `libs/platform_core` | library | 44 | 2026-10-31 `e3ab602f2` | 166 | 2 MB | yes | keep: imported |
| `libs/platform_devpost` | library | opportunity-radar-api | 2026-09-26 `2d3dfa7c7` | 6 | 261 KB | no | keep: imported |
| `libs/platform_discord` | library | DiscordBot, handwriting-ai | 2026-09-25 `9e829fb05` | 5 | 427 KB | no | archive with DiscordBot and handwriting-ai, its only importers |
| `libs/platform_email` | library | none | 2026-10-18 `92da96636` | 13 | 621 KB | no | archive: no importer; corvis has outlook_ tools |
| `libs/platform_kaggle` | library | github-stats-api, opportunity-radar-api | 2026-09-26 `a7a9108e8` | 8 | 456 KB | no | keep: imported |
| `libs/platform_langid` | library | grandma-api | 2026-11-01 `cce1217d2` | 87 | 334 KB | no | keep: imported |
| `libs/platform_ml` | library | 5 | 2026-10-07 `f8021aed1` | 28 | 646 KB | no | keep: imported |
| `libs/platform_music` | library | music-wrapped-api | 2026-09-26 `e63f69788` | 6 | 245 KB | no | archive with services/music-wrapped-api |
| `libs/platform_stt` | library | grandma-api, transcript-api | 2026-10-08 `7a62fbcf6` | 12 | 413 KB | no | keep: imported |
| `libs/platform_translate` | library | grandma-api | 2026-08-14 `37f3790c5` | 8 | 201 KB | no | keep: imported |
| `libs/platform_workers` | library | 12 | 2026-09-25 `73a72c49b` | 7 | 287 KB | no | keep: imported |
| `libs/procart` | library | procart-api | 2026-09-26 `b81df9e0f` | 6 | 228 KB | no | archive with services/procart-api |
| `services/Art-Trainer` | none (Railway fallback) | none | 2026-10-11 `d05227f68` | 89 | 875 KB | no | archive: no deploy, no importer |
| `services/Model-Trainer` | none (Railway fallback) | none | 2026-10-31 `823d9ecf2` | 161 | 6 MB | yes | keep: research (mi, floor) |
| `services/covenant-radar-api` | none (Railway fallback) | none | 2026-10-30 `6820fc2d8` | 193 | 352 MB | yes | keep: research (cleargbm); move data/external out of git |
| `services/data-bank-api` | none (Railway fallback) | none | 2026-09-26 `c6b1f6b75` | 13 | 292 KB | no | archive with the service mode that calls it: no deploy, called only by covenant-radar-api and Model-Trainer through platform_core |
| `services/github-stats-api` | Railway (200) | none | 2026-09-26 `8710836c0` | 9 | 513 KB | no | keep: live |
| `services/grandma-api` | Railway (200) and Pages | none | 2026-11-01 `5933006ef` | 107 | 760 KB | no | keep: live |
| `services/handwriting-ai` | none (Railway fallback) | none | 2026-10-25 `fce5d6be8` | 68 | 1 MB | no | archive: no deploy, no importer, called by DiscordBot |
| `services/music-wrapped-api` | none (Railway fallback) | none | 2026-10-08 `235dfe7fb` | 8 | 269 KB | no | archive with libs/platform_music: no deploy, no importer |
| `services/opportunity-radar-api` | Railway (200) | none | 2026-09-26 `21858e490` | 8 | 257 KB | no | keep: live |
| `services/procart-api` | none (Railway fallback) | none | 2026-10-08 `c8165824f` | 42 | 2 MB | no | archive with libs/procart: no deploy, no importer |
| `services/qr-api` | none (Railway fallback) | none | 2026-09-26 `2c49437b2` | 13 | 166 KB | no | archive: no deploy, no importer, called by DiscordBot |
| `services/transcript-api` | Railway (200) | none | 2026-09-26 `6bddd3497` | 13 | 602 KB | no | keep: live |
| `services/turkic-api` | none (Railway fallback) | none | 2026-10-08 `55c6a02d0` | 18 | 907 KB | no | archive: no deploy, no importer |
| `tools/board-watch` | sessions | 6 | 2026-10-29 `e0f3f1242` | 10 | 204 KB | no | keep: imported |
| `tools/ci-wake` | hub pump | none | 2026-10-29 `d52fd1a6e` | 37 | 434 KB | no | keep: fleet tooling |
| `tools/code-corpus` | none | none | 2026-10-07 `897d8ffa9` | 8 | 207 KB | no | keep: research |
| `tools/code-style-eval` | none | none | 2026-10-07 `503266a48` | 11 | 3 MB | yes | keep: research |
| `tools/commit-scope` | none | none | 2026-10-07 `6670ec0e2` | 6 | 164 KB | no | keep: fleet tooling |
| `tools/fleet` | every fleet node | fleet-wake | 2026-11-01 `19d7ec032` | 1373 | 4 MB | no | keep: imported |
| `tools/fleet-execution` | none | none | 2026-10-25 `bb7515d22` | 637 | 1 KB | no | keep: fleet tooling |
| `tools/fleet-execution-hub` | none | none | 2026-10-18 `2da3fae82` | 34 | 1 KB | no | keep: fleet tooling |
| `tools/fleet-execution-runners` | none | none | 2026-10-19 `2e522b4e7` | 50 | 1 KB | no | keep: fleet tooling |
| `tools/fleet-health-wake` | hub pump | none | 2026-10-23 `451ae3a6f` | 8 | 165 KB | no | keep: fleet tooling |
| `tools/fleet-wake` | fleet | none | 2026-10-30 `b83b023a0` | 64 | 182 KB | no | keep: fleet tooling |
| `tools/hpc-wake` | hub pump | none | 2026-10-28 `fb77f0d61` | 68 | 256 KB | no | keep: fleet tooling |
| `tools/hpc3` | none | RustedWarfareBot, hpc-wake | 2026-11-01 `9bdb73ad0` | 218 | 15 MB | yes | keep: research |
| `tools/hpc3-execution` | none | none | 2026-09-30 `4e8a0f44f` | 0 | 1 KB | no | delete: no registry entry, never run |
| `tools/lock-wake` | hub pump | none | 2026-10-10 `35778d50c` | 7 | 239 KB | no | keep: fleet tooling |
| `tools/maketools` | none | none | 2026-11-01 `1ed3ada33` | 125 | 475 KB | no | keep: fleet tooling |
| `tools/maketools-execution` | none | none | 2026-11-01 `78c46a140` | 1052 | 3 KB | no | keep: fleet tooling |

`clients/docs` is one 19 KB document, not a package, and is left out.[^size]

## The Discord group was the operator's question

`clients/DiscordBot` is a bot with slash commands for QR codes, YouTube transcripts and digit recognition, so it calls `qr-api`, `transcript-api` and `handwriting-ai`.[^discord] None of those has a live default Railway host except `transcript-api`, which is kept. Whether the bot still ran on Railway under some other service name could not be read without a Railway credential, so the group (DiscordBot, `handwriting-ai`, `qr-api`, `turkic-api` and `libs/platform_discord`) was put to the operator. `data-bank-api` was first put in the group by mistake: it is the file store covenant-radar-api's train and predict workers and Model-Trainer's corpus fetch call through `platform_core.data_bank_client`, while the research itself runs as batch jobs on local data and never calls it.

## What the operator picked

On 2026-11-01 the operator picked every archive recommendation above, the whole Discord group, and `data-bank-api` with the service mode that calls it.[^picks] So thirteen packages leave the monorepo, `services/data-bank-api` last, once nothing calls it: `libs/instrument_io`, `libs/platform_email`, `libs/platform_music`, `libs/platform_discord`, `libs/procart`, `services/Art-Trainer`, `services/handwriting-ai`, `services/music-wrapped-api`, `services/procart-api`, `services/qr-api`, `services/turkic-api`, `services/data-bank-api` and `clients/DiscordBot`. Their history is at the tag `archive/api-packages-2026-11-01`, on API `da1c233ec`, the last commit holding all thirteen. `tools/hpc3-execution` was not asked about and stays.

Removing `data-bank-api` meant replacing what its callers stored through it rather than only deleting a directory. covenant-radar-api now reads and writes its models only at the local paths its settings name, with no download at startup and no upload after training. Model-Trainer's service mode now uses the filesystem stores its cluster mode already used: a corpus is staged under `/data/corpus` by its SHA-256 and a finished run is packed into `/data/artifacts/store`. `platform_core`'s data-bank client, protocol, events and settings and `platform_ml`'s HTTP `ArtifactStore` went with it.[^dbank]

## Archiving is not just deleting a directory

`libs/platform_core` names several services in shared configuration and client code, so archiving a service also trims those references, and every package that imported it must pass `make check` on the fleet after. Each archived package's history stays at a named tag. `tools/hpc3-execution` is a recipe for a fleet project that was never registered, so it has no history worth tagging.[^hpc3x]

[^ask]: [synthesis] -- MCPs board task a276f973, its spec's OPERATOR block, dated 2026-11-01.
[^dbank]: [synthesis] -- MCPs board task a9566810, which carried the operator's ruling "Archive it and strip the service mode"; the change is described from its own diff over `services/covenant-radar-api`, `services/Model-Trainer`, `libs/platform_core`, `libs/platform_ml` and `libs/monorepo_guards`.
[^picks]: [synthesis] -- MCPs board task a276f973, notes 226e1b90 and bc8f96d4: the two RULING lines recording the operator's picks and the correction that put data-bank-api to the operator on its own.
[^size]: [synthesis] -- MCPs board task a276f973, note e96f59e1: `git ls-tree -r -l origin/main` summed per package; `services/covenant-radar-api/data/external` holds 33 dataset directories.
[^fleet]: [synthesis] -- MCPs board task a276f973, note e96f59e1: `fleet_dispatch_jobs` closed after 2026-10-02T20:44Z and at or before 2026-11-01T20:44Z, summed closed_at minus claimed_at per project and floored to whole minutes. The first reading, at 20:44Z over a window it did not pin, gave `tools/hpc3` 217 and `tools/maketools-execution` 1043; the cells are the pinned window's, re-read on 2026-11-02.
[^mxe]: API `5f661ee41`, `tools/fleet/registry/projects/tools/maketools-execution.json` -- eight `install` steps, one `image-<service>` phase per deployed service. API `0ebe2e4dd` cut them to the four kept services.
[^probe]: [synthesis] -- MCPs board task a276f973, note e96f59e1: `curl -D - https://<host>.up.railway.app/healthz` at 2026-11-01T20:46:56Z, and `docker ps` on demure, diphtheria and emerald at 20:41Z. The served case `execution/tests/served/api-railway-health.test.ts` in MCPs asks the same fourteen hosts from a fleet lane and fails when any answers otherwise.
[^deps]: [synthesis] -- MCPs board task a276f973, note e96f59e1: 101 `path = "..."` dependency edges across every manifest at origin/main.
[^solo]: [synthesis] -- MCPs board task a276f973, note e96f59e1: the last-own-change rule as stated there.
[^research]: `docs/RESEARCH.md` -- searched for each package's path.
[^discord]: API `da1c233ec` (the archive tag), `clients/DiscordBot/README.md:3` -- "Modular Python Discord bot with slash commands for QR code generation, YouTube transcripts, and handwritten digit recognition."
[^hpc3x]: `tools/hpc3-execution/Makefile:7` -- "the fleet project tools/hpc3-execution, which requires the `hpc3` capability"; no file under `tools/fleet/registry/projects` declares that path (MCPs board task a276f973, note e96f59e1).
