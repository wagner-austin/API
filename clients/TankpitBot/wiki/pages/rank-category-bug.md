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

The Xc (tank entity) field `b.u` is **dual-purpose**:
1. **On initialization** (TankEntry, sd() bot setup): set to rank_category (static, from packed byte bits 2-3)
2. **During gameplay** (TankStatusSync, Movement, MovementResponse): overwritten with damage_state (dynamic, tracks combat damage 0→3→2→1)

Capture data proves the gameplay values track damage progression during combat, not static rank category. The initial misidentification as "always rank_category" was wrong — the field changes meaning after initialization.

## V-table Handlers

| Handler | Message | JS Source | Field Set | Wire Byte |
|---------|---------|-----------|-----------|-----------|
| Uf (TankEntry, 0x28) | `b.u = this.i` | `a[3]>>2&3` | rank_category | packed byte bits 2-3 |
| Qf (TankStatusFull, 0x3E) | `b.u = this.m` | `a[0]>>2&3` | rank_category | packed byte bits 2-3 |
| Og (TankStatusSync, 0x2E) | `b.u = this.s` | `a[3]` | rank_category | byte 3 |
| Lg (Movement, 0x47) | `b.u = this.i` | `a[5]` | rank_category | byte 5 |
| Mg (MovementResponse, 0x3D) | `b.u = this.l` | `a[6]` | rank_category | byte 6 |

In TankEntry and TankStatusFull, the value is explicitly extracted as `(packed >> 2) & 3` — a 2-bit field from a byte that also contains team (bits 0-1) and rank (bits 4-7). This is unambiguously rank_category, not health.

The same field (`b.u`) is set in all 5 handlers. It is always rank_category.

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

`kc = [44,44,28,28,28,12,12,12,12]`. For bot index 0: `kc[0]=44=0b00101100`, `(44|0)>>2&3 = 11&3 = 3`. This sets b.u to a static rank_category during practice bot initialization.

But during gameplay, Movement/MovementResponse/TankStatusSync OVERWRITE b.u with dynamic values that track damage progression (0→3→2→1 observed in captures during combat). The field is correctly named `damage_state` in our decoders.

## Resolution

The initial analysis (renaming to rank_category) was **wrong**. The field is dual-purpose but the gameplay value IS damage_state. The original decoders were correct. Reverted all renames. Only the TankEntry decoder byte order fix and the EnemyDetection byte order fix are real bugs (those were applied correctly).
