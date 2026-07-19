---
title: Deactivation (0x41) Wire Format
tags: [protocol, combat, wire]
related: [[shoot-event-format]], [[weapon-log-markers]]
sources: [tpclient.js Pg.h / V.A, capture replay 2026-07-19, see footnotes]
fact_checked: 2026-07-19
confidence: high
---

# Deactivation (0x41) Wire Format

Kill confirmation message. The server emits 0x41 when one tank deactivates another, carrying both IDs, promotion eligibility, and a mine-kill sentinel.

## Wire layout

JS handler `Pg.h` (V.A), verified 2026-06-19. 6-byte body after the 0x41 type byte:

```
[0]    status                     — Pg constructor's first param (kill-type / cause)
[1:3]  victim_id (LE u16)         — X(a[1], a[2])
[3]    promo_eligible (1 = eligible)
[4:6]  killer_id_raw (LE u16)     — X(a[4], a[5]); see mine sentinel below
```

Tunneled inside 0x2E: outer subtype is `0x41`, inner is 6 bytes — same layout.

## Mine kills

If `killer_id_raw >= 65530`, the kill was caused by a mine. Post-processing:

```python
is_mine_kill = killer_id_raw >= 65530
killer_id = killer_id_raw - 65530 if is_mine_kill else killer_id_raw
```

When `is_mine_kill` is true, the residual `killer_id` is the **mine team** (0=red, 1=purple, 2=blue, 3=orange), not a tank ID. This sentinel encoding is from JS `Pg.h` directly.

## ID offset history

The victim/killer IDs were originally decoded at wrong offsets (offset 0 instead of 1), producing garbage IDs like 62976 instead of 502. Fixed by reading at offset 1,2 for victim and 4,5 for killer.[^1] The full 5-field expansion (adding `status`, `promo_eligible`, `is_mine_kill`) landed 2026-06-19 with the unification audit.

## Own kills

0x41 **fires for own kills**, always tunneled inside a 0x2E envelope.[^2] The June 2026 claim that it "never fires for own kills" was a decoder blind spot, not server behavior: the June-10-era decoder had no 0x2E subtype dispatcher and could not unwrap tunneled 0x41s, so the diagnostics stayed at zero across on-screen kills and the wrong conclusion was codified here. The DOM game-log kill banner ("has been deactivated by you") is the *client's rendering* of this same message — it lags the wire by one to two ticks (render + scraper poll) and carries less information (name only, no killer id, no promo flag, no mine sentinel).

Consequence (2026-07-19): the game-log kill-scraping channel was deleted from the bot; 0x41 is the single kill source. The DOM log is retained as a capture-artifact witness only.

[^1]: protocol analysis 2026-06-10 — offset correction from 0→1 for victim_id, 3→4 for killer_id
[^2]: capture replay 2026-07-19 with the modern 0x2E-tunneled decoder: `bot-20260610-005248` contains 1 own-kill 0x41 (victim 512, killer 1301 = the bot) and `bot-20260610-011333` contains 19 (killer 1301 on every one) — the very captures the "never fires" claim was drawn from. Run `bot-20260719-004608` cross-confirms live: all 4 own kills decoded as 0x41 (victims 500/513/506/511, killer_id 1301) matching the game-log banners 1:1, banners trailing by 0–2 s.
