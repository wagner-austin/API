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
source_git_blobs:
  "src/tankpit_bot/browser/overlay.py": "d7c588bb375aa605518f368dfefb9c1e69f8dfe1"
  "src/tankpit_bot/browser/overlay_hud.py": "67877546967b5fc8025baf94aa84c2af3bcab3fd"
  "src/tankpit_bot/browser/flag_capture.py": "29b74c4d65e877914ede877679ff867d91aba273"
fact_checked: "2026-08-06"
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

## Tracing a flag (triage recipe)

Every `human_flag` event auto-carries `tick_n`, `bot_state`, and the
runtime timestamp alongside its own fields[^4], so a flag pins an exact
spot in the event stream:

1. `grep '"human_flag"' runs/bot/<run>.events.jsonl` → note the
   flag's `flag_seq` and `tick_n` (call it T).
2. **Before** the click: the event's own `recent_ticks` (last 8 HUD
   payloads — mode, position, fuel, stocks, decision, reason, target).
3. **After** the click: filter the same JSONL to `tick_n >= T` — every
   decision, `hop_selected`/`hop_declined`, `acquisition_candidates`,
   gain, and outcome that followed, in order.
4. File findings on a triage page (first: [[flag-triage-20260729]])
   with a fix-status table; close rows only with run/sim receipts.[^1]

[^1]: `src/tankpit_bot/browser/overlay.py` (payload + slot renderer);
    `src/tankpit_bot/browser/overlay_hud.py:181` —
    `build_hud_expression`, the install-once template; tick wiring at
    `src/tankpit_bot/bot/tick_body.py:195` (step 9, "Update the in-page
    HUD so a human watching the browser sees what…"), which calls
    `update_bot_overlay` at `:230` and mirrors the same payload to
    `hud.json` at `:231-234`. All three re-verified and pinned
    2026-08-07; the tick wiring moved from `tick_loop.py` when the tick
    loop was split into a dispatcher and a per-tick body.
[^2]: The fiesta SPA, in a DIFFERENT repository —
    `~/PROJECTS/MCPs/fiesta/src/style.css` (`.screen-panel` glass
    recipe, `.console-button` face; 4 matching selectors) and
    `~/PROJECTS/MCPs/fiesta/src/services/theme.ts` (`RETRO_THEME`,
    user-pinned 2026-07-05). Both files confirmed present 2026-08-06.
    They sit outside this project's `workspaceRoot`, so they cannot
    appear in `source_paths` or carry a `source_git_blobs` pin here —
    cited by path only, and drift in them is NOT detectable by this
    wiki's gate.
[^3]: `src/tankpit_bot/browser/flag_capture.py:50` —
    `FlagCaptureService`, with `FLAG_RING_SIZE = 8` at `:46`;
    binding-vs-fetch rationale measured 2026-07-29 in
    `src/tankpit_bot/browser/live_view.py` module docs.
[^4]: `src/tankpit_bot/runtime_logging.py:45` — `RUNTIME_CONTEXT_KEYS: frozenset[str] = frozenset({"tick_n", "bot_state", "in_flight_action_kind"})`, the context stamped onto every emitted diagnostic (field docs at `:57-67`). The flag's own fields come from `src/tankpit_bot/browser/flag_capture.py:100-105`: `emit_diagnostic(diagnostic_kind="human_flag", flag_seq=..., clicked_at_ms=..., recent_ticks=...)`.
