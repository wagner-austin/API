---
title: Deactivation (0x41) Wire Format
tags: [protocol, combat, wire]
related:
  - "[[shoot-event-format]]"
  - "[[weapon-log-markers]]"
source_paths:
  - "tpclient.js"
  - "src/tankpit_bot/physics/capacity.py"
  - "runs/bot"
  - "runs/sniff"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
  "src/tankpit_bot/physics/capacity.py": "07208a2a67947512d7f592177dc968e1dd9e484c"
fact_checked: "2026-07-19"
confidence: high
hubs: [protocol]
---

# Deactivation (0x41) Wire Format

Kill confirmation message. The server emits 0x41 when one tank deactivates another, carrying both IDs, promotion eligibility, and a mine-kill sentinel.[^4]

## Wire layout

JS handler `Pg.h` (V.A), verified 2026-06-19. 6-byte body after the 0x41 type byte:[^4]

```
[0]    status                     — Pg constructor's first param (kill-type / cause)
[1:3]  victim_id (LE u16)         — X(a[1], a[2])
[3]    promo_eligible (1 = eligible)
[4:6]  killer_id_raw (LE u16)     — X(a[4], a[5]); see mine sentinel below
```

Tunneled inside 0x2E: outer subtype is `0x41`, inner is 6 bytes — same layout.[^2]

## Mine kills

If `killer_id_raw >= 65530`, the kill was caused by a mine. Post-processing:[^4]

```python
is_mine_kill = killer_id_raw >= 65530
killer_id = killer_id_raw - 65530 if is_mine_kill else killer_id_raw
```

When `is_mine_kill` is true, the residual `killer_id` is the **mine team** (0=red, 1=purple, 2=blue, 3=orange), not a tank ID. This sentinel encoding is from JS `Pg.h` directly.[^4]

## ID offset history

The victim/killer IDs were originally decoded at wrong offsets (offset 0 instead of 1), producing garbage IDs like 62976 instead of 502. Fixed by reading at offset 1,2 for victim and 4,5 for killer.[^1] The full 5-field expansion (adding `status`, `promo_eligible`, `is_mine_kill`) landed 2026-06-19 with the unification audit.

## Own kills

0x41 **fires for own kills**, always tunneled inside a 0x2E envelope.[^2] The June 2026 claim that it "never fires for own kills" was a decoder blind spot, not server behavior: the June-10-era decoder had no 0x2E subtype dispatcher and could not unwrap tunneled 0x41s, so the diagnostics stayed at zero across on-screen kills and the wrong conclusion was codified here. The DOM game-log kill banner ("has been deactivated by you") is the *client's rendering* of this same message — it lags the wire by one to two ticks (render + scraper poll) and carries less information (name only, no killer id, no promo flag, no mine sentinel).

Consequence (2026-07-19): the game-log kill-scraping channel was deleted from the bot; 0x41 is the single kill source. The DOM log is retained as a capture-artifact witness only.[^5]

[^1]: Protocol analysis of 2026-06-10, which moved `victim_id` from offset 0→1 and `killer_id` from 3→4. Both corrections are what the decoder implements today: `src/tankpit_bot/protocol/decoders/combat.py:98` reads `victim_id=x16(data[1], data[2])` and `:92` reads `raw_killer = x16(data[4], data[5])`, against the layout comment at `:90-91` — `[status:1] [victim_id:2 LE] [promo_eligible:1] [killer_id:2 LE]`, sourced from `tpclient.js` `Pg.h (V.A)`. Re-verified 2026-08-06.
[^4]: JS truth: `Pg.h` handler in `tpclient.js` (blob-pinned in frontmatter) — 6-byte field walk and the 65530 mine sentinel both read directly from the handler body; decode mirrored in the bot's 0x41 decoder and exercised by the [^2] capture replays.
[^5]: `wiki/log.md:1198` — "[2026-07-19] falsification + teardown | 0x41 fires for own kills; DOM game-log consumption deleted (wire is the single actor)". The own-kill case is visible in the decoder itself: `src/tankpit_bot/protocol/decoders/combat.py:92-94` treats `raw_killer >= 65530` as the mine-kill encoding and rebases it (`killer_id = raw_killer - 65530`), surfacing `is_mine_kill` on the decoded record — so own kills are carried on the wire rather than inferred from the DOM.
[^2]: capture replay 2026-07-19 with the modern 0x2E-tunneled decoder: `bot-20260610-005248` contains 1 own-kill 0x41 (victim 512, killer 1301 = the bot) and `bot-20260610-011333` contains 19 (killer 1301 on every one) — the very captures the "never fires" claim was drawn from. Run `bot-20260719-004608` cross-confirms live: all 4 own kills decoded as 0x41 (victims 500/513/506/511, killer_id 1301) matching the game-log banners 1:1, banners trailing by 0–2 s.

## The corpse window is exactly 22 seconds (corpus-swept 2026-07-22)

The single 2026-06-20 observation ("0x58 TankRemove arrives ~22 s
later") is now a corpus constant: 37 kill→remove pairs across all
246 sessions give min = median = **exactly 22.0 s** (the
distribution tail is id-reuse pairing noise — a respawned tank's
later viewport-exit 0x58 matched against the old kill). The sim
implements it as ``CORPSE_WINDOW_TICKS = 11`` at the 2 s cadence
([[physics-module-roadmap]]); the corpse 0x58 does NOT start the
law-4 reroute clock — rerouting only follows LIVING departures.

## SOLVED (2026-07-23): there is no healing — the damage tier IS the fuel quartile

The 2026-07-22 "healing measured but unresolved" section below-the-fold
was a MISREADING, corrected by the user the next day: **tanks do not
heal over time. Fuel is the health pool**, recovered only by fuel
pickups, and the rendered damage shade (mouse-over: lighter = more
HP) is a pure fuel indicator.[^3]

Corpus fit, same day: every 0x2E sync carrying BOTH the damage tier
and the absolute fuel (the long form) is a supervised pair — 19,658
samples across 246 sessions, **zero exceptions**:[^3]

| tier | fuel range (rank 1, cap 1100) | meaning |
|---|---|---|
| 3 | 825–1100 | top quartile — healthy, lightest shade |
| 2 | 550–824 | |
| 1 | 275–549 | |
| 0 | 0–274 | bottom quartile — near death, darkest |

The law: ``damage_tier = min(3, 4 * fuel // fuel_capacity(rank))``
(`physics/capacity.py:damage_tier`, blob-pinned in frontmatter; the
claim block below is machine-checked by the `physics_claims` guard
stage; the sim derives the tier from fuel at every emission point
and stores no tier state; the shadow comparator re-derives the law
on every ``make shadow``).[^3]

This retro-explains every confusion in the old reading: the "quiet
heals" (1→3 ×257, 2→3 ×199) were fuel pickups jumping quartiles;
the "~6–10 s dwell before repair" was time-to-drive-to-a-container;
"1→0 ×143" was fuel LOSS, not healing; June's "tiers count down
0→3→2→1 and kills die from tier 1" was a fresh (briefly unsynced)
tank draining 3→2→1, dying from tier 1 because the killing hit took
fuel below zero before a tier-0 sync could broadcast. The bot-side
consequences are real: ``DAMAGE_*`` constants and the finish-off
ordering in ``bot/ai/threats.py`` were inverted and are now fixed —
tier 0 is the kill-shot target, and an unknown tank defaults to
tier 3 (assume healthy), not "full = 0".[^3]

```json claims
{
  "claims": [
    {
      "id": "damage-tier",
      "code": "tankpit_bot.physics.capacity:damage_tier",
      "formula": "min(3, 4 * fuel // fuel_capacity(rank))",
      "probes": [
        {"args": [0, 1], "expect": 0},
        {"args": [274, 1], "expect": 0},
        {"args": [275, 1], "expect": 1},
        {"args": [549, 1], "expect": 1},
        {"args": [550, 1], "expect": 2},
        {"args": [824, 1], "expect": 2},
        {"args": [825, 1], "expect": 3},
        {"args": [1100, 1], "expect": 3},
        {"args": [440, 8], "expect": 0},
        {"args": [450, 8], "expect": 1}
      ]
    }
  ]
}
```

## Self-death receipts: the 0x41 arrives everywhere, and the wrap precedes it

Live-corrected 2026-08-26/27 against `runs/bot/desert/bot-20260826-182204`
(three deaths, each carrying `origin=protocol_0x41`): **the self 0x41
DOES arrive on Normal fields** — the earlier "no 0x41 for self on
Normal" claim was an inference from a dead reporting counter, falsified
by the artifact. What actually happens at a self-death, in wire order
inside one drain batch:

1. A fuel total drives THROUGH zero and the u16 field wraps (readings
   65460/65475/65482/65530 observed). The wrap is a valid earlier
   receipt of the death — and must never be ingested as a +65k pickup
   (it polluted the fuel book three times in that run).
2. The self 0x41 lands one message later with the killer id.

Both producers raise the same `ws.self_deactivated` flag and dedup on
it — one death, one `self_deactivated` diagnostic, whichever receipt
lands first (the wrap usually wins; it carries no killer id, honestly).
Every reporting consumer (run digest, engagement ledger) counts that
single diagnostic; the old free-text regex and the
tank_deactivated-victim-is-us branch were dead code that read deaths=0
through three real deaths. Detection lives in
`src/tankpit_bot/sniffer/world_state_containers.py` (wrap) and
`world_state_dispatch.py` (0x41); the respawn contract
(`_handle_own_deactivation`) consumes the flag either way.

[^3]: user (Austin), 2026-07-23 — "tanks dont heal. they only can recover health/fuel from picking up fuel containers... when i mouse over a tank on the map, it shows them lighter more hp or darker lower hp". Corpus fit same day: 19,658 tier+fuel pairs, boundaries exactly 275/550/825 = capacity quartiles at rank 1, zero exceptions. The consequence — fuel IS health, so the only recovery path is a fuel pickup — is why the pickup predicate refuses nothing on volume grounds: `minimum_volume=1` at `src/tankpit_bot/bot/ai/collect_pickups.py:253`. Deactivation itself is read off the wire, not inferred from health: `src/tankpit_bot/combat.py:114` and `:119`.
