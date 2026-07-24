---
title: b.u Dual-Purpose Field (rank_category on init, damage_state during gameplay)
tags: [protocol, bug, decoder]
related:
  - "[[v-table-complete]]"
  - "[[decode-coverage]]"
  - "[[js-source-map]]"
source_paths:
  - "tpclient.js:127"
  - "tpclient.js:158"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (traced b.u assignment across all 5 V-table handlers)
hubs: [protocol]
---

# rank_category Mislabeled as damage_state

## The Field

The Xc (tank entity) field `b.u` is **dual-purpose**:[^1]
1. **On initialization** (TankEntry, sd() bot setup): set to rank_category (static, from packed byte bits 2-3)
2. **During gameplay** (TankStatusSync, Movement, MovementResponse): overwritten with damage_state (dynamic; its full semantics were solved 2026-07-23 — the value is the fuel-quartile tier, [[deactivation-format]] §SOLVED)

Capture data proves the gameplay values change under combat, not static rank category. The initial misidentification as "always rank_category" was wrong — the field changes meaning after initialization.[^2]

## V-table Handlers

| Handler | Message | JS Source | Field Set | Wire Byte |
|---------|---------|-----------|-----------|-----------|
| Uf (TankEntry, 0x28) | `b.u = this.i` | `a[3]>>2&3` | rank_category | packed byte bits 2-3 |
| Qf (TankStatusFull, 0x3E) | `b.u = this.m` | `a[0]>>2&3` | rank_category | packed byte bits 2-3 |
| Og (TankStatusSync, 0x2E) | `b.u = this.s` | `a[3]` | rank_category | byte 3 |
| Lg (Movement, 0x47) | `b.u = this.i` | `a[5]` | rank_category | byte 5 |
| Mg (MovementResponse, 0x3D) | `b.u = this.l` | `a[6]` | rank_category | byte 6 |

In TankEntry and TankStatusFull, the value is explicitly extracted as `(packed >> 2) & 3` — a 2-bit field from a byte that also contains team (bits 0-1) and rank (bits 4-7). This is unambiguously rank_category, not health.[^1]

The same field (`b.u`) is set in all 5 handlers — but NOT with the same meaning. (The sentence that previously ended this section, "It is always rank_category," was the intermediate analysis this page's Resolution retracts: the Og/Lg/Mg gameplay writes carry the dynamic tier, not rank_category.)[^1]

## Proof: sd() Bot Initialization (line 43)

```javascript
function sd(a) {
  for (var b = 0; 4 > b; b++)
    for (var c = 0; 9 > c; c++) {
      var d = ud(a.P, 9*b+c+1);
      d.name = lc[b]+(c+1);
      var e = kc[c]|b;
      d.l = e>>4&15;    // rank
      d.h = e&3;        // team
      d.u = e>>2&3;     // rank_category (INITIAL value only)
    }
}
```

`kc = [44,44,28,28,28,12,12,12,12]`. For bot index 0: `kc[0]=44=0b00101100`, `(44|0)>>2&3 = 11&3 = 3`. This sets b.u to a static rank_category during practice bot initialization.[^1]

But during gameplay, Movement/MovementResponse/TankStatusSync OVERWRITE b.u with dynamic values (transitions observed in June combat captures). The field is correctly named `damage_state` in our decoders; what the number MEANS was closed 2026-07-23 — it is the fuel quartile, corpus-fitted 19,658/19,658 ([[deactivation-format]] §SOLVED), so the June "damage progression 0→3→2→1" reading was itself a misinterpretation of fuel draining.[^2]

## Resolution

The initial analysis (renaming to rank_category) was **wrong**. The field is dual-purpose but the gameplay value IS damage_state. The original decoders were correct. Reverted all renames. Only the TankEntry decoder byte order fix and the EnemyDetection byte order fix are real bugs (those were applied correctly).[^1][^2]

[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned lines 127/158; `sd()` at line 43 with the `kc` table) — every table row and code fence above carries its assignment expression inline, re-checkable by grep; trace date in the frontmatter `verified:` field.
[^2]: June 2026 combat captures recorded the b.u transitions on tanks under fire that forced the revert; the gameplay value's semantics are now fully solved as the fuel-quartile tier — law, claim block, and 19,658-pair corpus fit in [[deactivation-format]] §SOLVED (2026-07-23).
