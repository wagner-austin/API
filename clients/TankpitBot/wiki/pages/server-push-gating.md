---
title: Server Push Gating (Play-to-Receive)
tags: [protocol, wire, server-behavior, observation]
related:
  - "[[tank-freshness-model]]"
  - "[[enemy-bot-behavior]]"
  - "[[connection-protocol]]"
  - "[[client-commands]]"
  - "[[make-targets]]"
source_paths:
  - "bot_watch_probe.capture_session.json"
  - "src/tankpit_bot/action_lab/enemy_teleport.py"
source_git_blobs:
  "bot_watch_probe.capture_session.json": "694fa343cbc2c92ad1fb30b4c7fb30d2bcbf58f6"
  "src/tankpit_bot/action_lab/enemy_teleport.py": "79f215b0c07d3ed82d83de632b6a93bdf140a0fe"
fact_checked: "2026-08-06"
confidence: high
hubs: [protocol]
---

# Server Push Gating (Play-to-Receive)

The server streams its periodic push traffic — 0x2E tank status syncs,
the empty 0x3F `MSG_SYNC` tick, 0x47 movement broadcasts — **only to
clients that are taking real gameplay actions**. Direct
request→responses are never muted. Queries (`'i'` inventory, `'h'`
nearest-enemy, `'l'` map open) and socket keep-alives do not count as
gameplay and do not hold the stream open.[^1]

## The seven-run discrimination (all 2026-07-24)

One variable per run; every run is a live `make bot-watch`-family
session with the full CDP capture on disk.[^1]

| Run | Design | Push stream |
|---|---|---|
| 1 | map + teleport, silent 10-min dwell | dead ~9 s after landing |
| 2 | idle at spawn, no map, silent | stragglers to 158 s, then dead |
| 3 | 3× map opens, 3-min idle dwells | each request answered; dead ~2 s after each |
| 4 | (landed-path bug: dwell silent) | dead ~8 s |
| 5 | teleport + 1.5 s inventory-query heartbeat | 366 responses, zero push traffic |
| 6 | no map, no teleport, query heartbeat | same — queries never count |
| 7 | teleport + 1.5 s **walk** heartbeat | **open for the full 617 s** |

Run 7 is the confirmation: a client walking one tile every 1.5 s
received self 0x2E syncs every ~3 s (188), 0x3F ticks every ~6 s (104),
and its own 0x47 movement echoes (205) for the entire ten-minute dwell,
where designs 1–6 all went silent inside a minute.[^1]

## The window is one server tick (precision run, n=101)

Run 8 slowed the heartbeat to 6 s so each action's window stands
alone: 101 walk beats, and every single beat produced exactly two
push messages whose last arrival landed at median 1.21 s / p90
1.86 s / **max 2.06 s** after the walk — then silence until the next
beat. The post-action push window is one ~2 s server tick; a 1.5 s
action cadence merely re-opens it before it closes, which is why run
7 looked continuous.[^3]

## The connection itself is rate-gated (four-session discrimination)

Beyond the per-action push window, the CONNECTION has its own play
gate: **~12 minutes after join, the server disconnects a client
whose action rate is too sparse** — and sparse actions do NOT reset
the clock. Four radar-watch sessions varied one factor each:[^4]

| Session | Design | Outcome |
|---|---|---|
| 1 | queries only, map polls | dead at ~716 s |
| 2 | real walk every 15 s, map polls | dead at ~701 s |
| 3 | free walk every 15 s, NO map | dead at ~714 s |
| 4 | walk+scan every **1.5 s**, no map | **alive through the full 909 s** |

Idleness, the map-open state, and per-action resets are all
falsified; only the dense cadence survived. The threshold sits
somewhere between 1.5 s and 15 s per action (unbracketed). The
production bot acts every ~2 s, which is why it never sees this —
the archive's 45-minute bot session is the long-duration
cross-check. Corollary for observation probes: the 1.5 s walk
shuffle is not optional politeness, it is the connection
keepalive.[^4]

## What still counts as "playing"

Server-rejected actions hold the stream open. Run 7's dwell never
drained its message buffer, so the shuffle aimed from the frozen
landing position: the west-bound half of its walks targeted the watched
bot's occupied tile and drew 154× `CANT_GO` plus 45× `ALREADY_THERE`
supervisor (0x52) rejections — and the push stream stayed open
regardless. Rejected walks also cost no fuel: total spend 788→505
matches the ~205 accepted 1-tile walks plus activity drift.[^1]

## What a playing observer actually receives

A held-open stream is **self-telemetry plus events, not a world
snapshot**. Ten minutes adjacent to an undisturbed bot produced zero
0x2E syncs, zero movements, and zero refuels for that bot — other tanks
enter the stream only when *they* act ([[enemy-bot-behavior]]:
undisturbed bots do nothing). Consequences:[^1]

- The "global ~2 s 0x2E broadcast" premise in [[tank-freshness-model]]
  is per-tank activity-conditional: the archive measured it during
  combat, where every bot was acting.
- Passively reading an idle tank's fuel is impossible; only its
  join-time 0x21 roster entry and its own future actions reveal state.
- 0x21 identity announcements arrive as a join-time roster dump (self +
  the 36-bot roster), not as mute-piercing events.

## Bot-side impact

The production bot acts every tick, so it never experiences the mute.
Observation probes must genuinely play: the watch dwell walks a 1-tile
shuffle per beat (`TANKPIT_ENEMY_TELEPORT_HEARTBEAT_MS`, ~40 fuel/min),
draining the CDP buffer each beat so the shuffle tracks the tank's true
position instead of repeating run 7's frozen-origin rejections.[^2]

[^1]: decisive capture `bot_watch_probe.capture_session.json`
    (2026-07-24, tank "Artax" id 1301, 1,198 messages, 617 s): 401
    five-byte walk frames t+7.2→615.8 s; received t>60 s = 188× 0x2E
    (all self), 188× 0x47, 199× 0x52 (154 code 1 `CANT_GO`, 45 code 6
    `ALREADY_THERE`), 104× 0x3F, 38× 0x21 (join roster). Runs 1–6:
    wiki/log.md entries of 2026-07-24 (anomaly → falsifications →
    law), captures `bot_watch_nomap_probe`, `bot_watch_wake_probe`,
    `bot_watch_nomap_hb_probe` at repo root.
[^2]: `src/tankpit_bot/action_lab/enemy_teleport.py:186` —
    `_heartbeat_action` (per-beat drain) and `_settle_dwell` at `:207`
    (shuffle); driven by `make bot-watch` (`Makefile:172`), see
    [[make-targets]]. Re-verified 2026-08-06 and repinned: the file
    changed only by threading a clock argument
    (`action_hooks.get_current_time_ms()`) into one landing call at
    `:396`, which touches neither method this footnote cites.
[^4]: radar-watch captures 2026-07-24/25:
    `radar_watch_probe.capture_session.json` (sessions 1–2 overwrote
    the same path; session 2 is the committed one),
    `radar_watch_nomap_probe.capture_session.json` (session 3, fuel 0
    throughout), `radar_watch_fast_probe.capture_session.json`
    (session 4, 587 free walk+scan beats, receive traffic steady to
    909 s). Kill times 716/701/714 s; the 45-minute archive session
    is `runs/bot/bot-20260610-011333.capture_session.json`.
[^3]: precision capture `bot_watch_pw6_probe.capture_session.json`
    (2026-07-24, 6 s walk heartbeat, same design as the decisive run
    otherwise): 101 five-byte walk frames at 6.0 s modal gap; per-beat
    push count exactly 2 for 101/101 beats; last-push offsets median
    1.21 s, p90 1.86 s, max 2.06 s; dwell kinds 92× each of
    0x2E/0x47/0x3F.
