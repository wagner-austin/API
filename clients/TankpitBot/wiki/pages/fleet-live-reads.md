---
title: Fleet Live Reads
tags: [fleet, architecture, telemetry, performance]
related:
  - "[[fleet-lifecycle]]"
  - "[[fleet-coordination]]"
  - "[[bot-service-architecture]]"
source_paths:
  - "src/tankpit_bot/diagnostics/event_tail.py"
  - "src/tankpit_bot/diagnostics/run_digest_fold.py"
  - "src/tankpit_bot/service/fleet_stream.py"
  - "src/tankpit_bot/service/fleet_telemetry.py"
source_git_blobs:
  "src/tankpit_bot/diagnostics/event_tail.py": "4d97cda6957481cd5fd578002f9b20923c2ce2e0"
  "src/tankpit_bot/diagnostics/run_digest_fold.py": "a6a3850fe48cadf840f64d826320254752d4ce29"
  "src/tankpit_bot/service/fleet_stream.py": "9a1fcdcaba2c7edb98cb97a73765118c44c3214c"
  "src/tankpit_bot/service/fleet_telemetry.py": "6b59d1826c8f34a554f8ce6566e271b556f4b449"
fact_checked: "2026-09-01"
confidence: high
hubs: [architecture]
---

# Fleet live reads: following a run instead of re-reading it

*Established 2026-09-01, after an operator reported that refreshing
the control page "took forever to load back and connect to the running
windows".*

## The cost that was there all along

Both fleet summaries — the digest numbers and the activity feed — were
computed by reading the instance's **entire** `latest.events.jsonl` and
decoding every line, on every cache miss.[^1] For a finished run that
is free. For a LIVE run it is not: the artifact grows for the whole
session, and the page polls once a second against a two-second
cache.[^2]

Measured on a real run: `runs/bot/yuppler/latest.events.jsonl` reached
**13.5 MB after about six minutes** of play.[^3] A five-bot fleet was
therefore re-reading and re-parsing on the order of 65 MB every two
seconds, and the page's poll loop awaited each bot **in turn**, so a
cold load paid the sum of the fleet rather than its slowest member.

The cost was invisible for a year because the CLI (`make digest`) runs
once, on a finished artifact.

## What replaced it

**The digest became a fold that can resume.** The reduction always was
a fold — one pass over records mutating a digest plus four pieces of
carry state — but it was welded to "read the whole file first".
`RunDigestAccumulator` makes the carry state explicit and durable, so
a caller that has folded the first N records can fold record N+1
alone.[^4] `build_run_digest` is now a thin caller of it, so the CLI
and the fleet compute identical numbers by construction.

The guarantee is tested as a property: folding a run in one pass and
folding it split at **every possible boundary** produce the identical
digest.[^5] That matters because the split points are decided by the
bot's write timing, not by anything the reader controls.

**The reader follows the file.** `EventTail` keeps a byte cursor and
decodes only what was appended.[^6] Two details make it safe rather
than merely fast:

- **Line boundaries.** A poll can land mid-append, so bytes after the
  last newline are withheld until the rest arrives. A multi-byte UTF-8
  character split across two polls survives, because it can only ever
  sit inside that withheld tail.
- **Run identity.** A new session re-creates the same path, which would
  otherwise leave the cursor pointing into the middle of a different
  run. The file's own filesystem index is compared on every read, so a
  replaced artifact is REPORTED as a restart and both folds reset.

**Reads are transactional.** The cursor is committed only after the
decode succeeds, and a fold that fails part-way **spoils** that run's
stream so every later poll fails the same way.[^7] Both exist for the
same reason: advancing past a line the strict decoder rejected would
serve a digest with a hole in it — numbers that look plausible and are
wrong — while claiming nothing was amiss.

One cursor feeds both summaries, so whichever of stats/activity is
asked for second finds nothing new to fold.[^8] The page also polls its
bots **concurrently** now, so first paint costs the slowest bot rather
than the sum.[^9]

[^1]: The pre-change `FleetTelemetry.stats` called `build_run_digest`
      (whole file) and `activity` called `load_event_records` (whole
      file) on every miss.
[^2]: `TELEMETRY_CACHE_TTL_MS = 2000`
      (`src/tankpit_bot/service/fleet_telemetry.py`); the page's
      `setInterval(poll, 1000)` (`service/fleet_page.py`).
[^3]: Measured 2026-09-01 by listing `runs/bot/*/latest.events.jsonl`;
      the yuppler run of 2026-08-28 19:58–20:04 is 13,542,295 bytes.
[^4]: `RunDigestAccumulator` in
      `src/tankpit_bot/diagnostics/run_digest_fold.py`.
[^5]: `tests/diagnostics/test_run_digest_fold.py`, parameterised over
      every split index, plus the one-record-at-a-time extreme.
[^6]: `EventTail.next_records` in
      `src/tankpit_bot/diagnostics/event_tail.py`; exercised against
      real files in `tests/diagnostics/test_event_tail.py`.
[^7]: `InstanceStream.refresh` in
      `src/tankpit_bot/service/fleet_stream.py`.
[^8]: `FleetTelemetry._refreshed_stream`
      (`service/fleet_telemetry.py`).
[^9]: `poll` fans out with `Promise.all(names.map(pollBot))`
      (`service/fleet_page.py`); it previously awaited each bot in a
      `for` loop.
