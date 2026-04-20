# Run Artifacts

This project uses one canonical runtime artifact layout for each mode.

The stable inspection paths are always under `runs/`, and every run also gets a
timestamped archive copy.

## `make bot`

`make bot` now writes six canonical artifacts:

- latest text log: `runs/bot/latest.log`
- archived text log: `runs/bot/bot-YYYYMMDD-HHMMSS.log`
- latest structured events: `runs/bot/latest.events.jsonl`
- archived structured events: `runs/bot/bot-YYYYMMDD-HHMMSS.events.jsonl`
- latest capture session: `runs/bot/latest.capture_session.json`
- archived capture session: `runs/bot/bot-YYYYMMDD-HHMMSS.capture_session.json`

For autonomous debugging, the three important files are:

```text
runs/bot/latest.log
runs/bot/latest.events.jsonl
runs/bot/latest.capture_session.json
```

`latest.log` is the operator-readable timeline. `latest.events.jsonl` is the
machine-readable event stream for `AI`, `SYNC`, `STATE`, `WIRE`, and `WORLD`
records.

## `make sniff`

`make sniff` also uses canonical latest and archive artifacts:

- latest text log: `runs/sniff/latest.log`
- archived text log: `runs/sniff/sniff-YYYYMMDD-HHMMSS.log`
- latest structured events: `runs/sniff/latest.events.jsonl`
- archived structured events: `runs/sniff/sniff-YYYYMMDD-HHMMSS.events.jsonl`
- latest capture session: `runs/sniff/latest.capture_session.json`
- latest raw capture: `runs/sniff/latest.raw_capture.json`
- latest session summary: `runs/sniff/latest.session_summary.json`
- archived capture session: `runs/sniff/sniff-YYYYMMDD-HHMMSS.capture_session.json`
- archived raw capture: `runs/sniff/sniff-YYYYMMDD-HHMMSS.raw_capture.json`
- archived session summary: `runs/sniff/sniff-YYYYMMDD-HHMMSS.session_summary.json`

If `TANKPIT_OUTPUT` or `OUTPUT=...` is provided, the requested output file is
still written, and the same run is mirrored into the canonical latest/archive
paths above.

The stable inspection paths are:

```text
runs/sniff/latest.log
runs/sniff/latest.events.jsonl
runs/sniff/latest.capture_session.json
runs/sniff/latest.raw_capture.json
runs/sniff/latest.session_summary.json
```

## Why This Matters

The bot and sniffer no longer rely on terminal scrollback as the primary source
of truth.

- `make bot` always leaves one deterministic latest text log, one
  deterministic latest structured event stream, and one latest capture
  session.
- `make sniff` always leaves one deterministic latest capture bundle plus the
  corresponding text/event logs.
- The CLI and the docs now point to the same locations, so future debugging can
  inspect one stable place instead of guessing where a run went.
