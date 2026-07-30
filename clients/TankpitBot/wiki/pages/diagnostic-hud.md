---
title: Diagnostic HUD + Human Flag Channel
tags: [architecture, observability, hud, diagnostics]
related:
  - "[[self-observing-architecture]]"
  - "[[bot-service-architecture]]"
  - "[[rendering-pipeline]]"
source_paths:
  - "src/tankpit_bot/browser/overlay.py"
  - "src/tankpit_bot/browser/overlay_hud.py"
  - "src/tankpit_bot/browser/flag_capture.py"
fact_checked: "2026-07-29"
confidence: high
hubs: [architecture]
---

# Diagnostic HUD + Human Flag Channel

The in-page HUD a human sees during `make run`, rebuilt 2026-07-29 from
the auto-sized green-on-black text block into a **fixed-geometry**
fiesta-styled glass card, plus a click-to-flag channel that turns "I
just saw a bug" into a queryable ledger event.[^1]

## Fixed geometry (the anti-jitter contract)

The old HUD resized every tick because its box hugged five variable-
length text lines. The new card never changes size: the DOM +
stylesheet are installed once, and every later tick only assigns
`textContent`/colors into pre-sized slots — fixed 272px width, fixed
row heights, `tabular-nums` digits, ellipsis clipping on the variable
slots (why/tgt), and a width-transition fuel meter.[^1]

## Design lineage

Palette and surface carried channel-for-channel from the fiesta
streaming SPA: the stippled frosted-glass panel (blue tint
`rgba(24,34,80,0.28)`, 9px dot grid, `blur(6px) saturate(1.1)`,
two-tone bevel border, blue halo) and the console-button face for the
flag button. Retro theme colors: neon green `rgb(57,255,20)` =
full/good/COLLECT, purple `rgb(200,0,200)` = neutral/UNSET, hot pink
`rgb(255,20,147)` = HUNT/low/held.[^2]

## What it shows (one card, nine rows)

Header (HFSM state) · mode banner (color-coded HUNT/COLLECT/UNSET with
substate) · position + fuel vs rank cap · fuel meter (green full /
purple mid / pink under the damage-tier-0 quartile) · five stock slots
vs rank cap (green at cap, off-white within the hunt-gate tolerance of
5, pink below) · decision + sent/held dot · typed reason · combat
target + in-flight action · session K/H/M/RJ + the flag button.[^1]

## The human flag channel

Clicking **⚑ FLAG** calls the `__botFlagDeliver` CDP binding (bindings,
not loopback fetches — Chrome's Local Network Access gate hangs
page→127.0.0.1 fetches forever, the same reason the live-view caster
uses a binding). The service emits a `human_flag` DIAGNOSTIC event to
`runs/bot/latest.events.jsonl` carrying `flag_seq`, `clicked_at_ms`,
and `recent_ticks` — a JSON snapshot of the last 8 HUD payloads (~16s
of bot thinking). `make analyze` and JSONL queries can anchor on
`diagnostic_kind == "human_flag"` instead of a human reconstructing
the moment from memory.[^3]

The card is `pointer-events: none` so it can never eat a game input;
only the flag button opts back in. The HUD lives in the DOM, not on
the game canvases, so it is invisible to the live-view caster (which
composites canvases only) — the phone stream shows the clean game.[^1]

[^1]: `src/tankpit_bot/browser/overlay.py` (payload + slot renderer),
    `overlay_hud.py` (install-once template, `build_hud_expression`),
    tick wiring in `bot/tick_loop.py` step 9.
[^2]: fiesta SPA `~/PROJECTS/MCPs/fiesta/src/style.css`
    (`.screen-panel` glass recipe, `.console-button` face) and
    `src/services/theme.ts` (`RETRO_THEME`, user-pinned 2026-07-05).
[^3]: `src/tankpit_bot/browser/flag_capture.py`
    (`FlagCaptureService`, ring size 8); binding-vs-fetch rationale
    measured 2026-07-29 in `browser/live_view.py` module docs.
