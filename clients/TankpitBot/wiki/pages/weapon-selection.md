---
title: Weapon Selection (Server-Side)
tags: [combat, weapons, protocol]
related: [[shoot-event-format]], [[weapon-log-markers]], [[shot-range]]
sources: [see footnotes]
fact_checked: 2026-07-02
confidence: high
---

# Weapon Selection

Weapon selection is **server-side**, not client-side. The fire command sends an (x,y) coordinate. The server decides which weapon to use based on what's at that tile.[^1]

## Selection rules

| What's at the tile | Weapon used | Byte | Effect |
|-------------------|-------------|------|--------|
| Enemy tank (stationary) | Dual shot | 0x01 | Hit, damage applied |
| Enemy tank (has pending move on same tick) | Homing shot | 0x03 | Auto-tracks to new position |
| Empty ground / terrain / water | Single shot | 0x00 | Miss — nothing there |

## Key implications

- **weapon_byte=0 is a genuine miss** — the target was not at the fired coordinates[^1]
- **Homing fires automatically** when the enemy moves on the same tick as your shot. You don't choose homing vs dual — the server does[^1]
- **Homing does NOT consume a dual shot** — it's a separate weapon[^1]
- **Keep duals enabled at all times** during combat. The server selects the right weapon[^1]
- **Homing also tracks teleporting enemies** — fire at their last position, server auto-homings if they teleported[^1]

## The weapon byte is the per-shot ammo ledger — and therefore the hit oracle

The server only spends dual / missile / homing ammo on a shot that
lands, and it records the spend in the ShootEvent ``weapon`` field.
The page client's inventory display decrements from exactly this
field. So per shot:[^2]

- ``weapon > 0`` → one consumable debited → **hit** (even when the
  impact tile is off the local viewport and ``victim_id`` resolves to
  ``-1`` — the tile-occupancy lookup cannot see off-viewport targets)
- ``weapon = 0`` → free single, nothing debited → **miss**

Wire proof (run 2026-07-02 01:21): five pursuit homings each carried
``weapon=3`` while ``victim_id=-1``; orange-3 died to the fifth. The
pre-2026-07-02 bot classified those winning shots as misses because
it keyed hits on ``victim_id`` and derived the ammo decrement from
that guess — leaving the ammo-delta cross-check circularly dependent
on the signal it existed to correct. The classifier now keys directly
on consumption; ``victim_id`` is kill-attribution metadata only.

A consumption-miss at a registry position that has NOT moved means
the target is genuinely gone from that tile (frozen registry entry or
unwitnessed corpse) — the bot blocks the target instead of repeating
the shot (run 2026-07-02 01:23: 25+ ``weapon=0`` shots looped at a
stale tile before this rule).

## Miss causes

A miss (weapon_byte=0) means one of:[^1]
1. The enemy moved before your shot resolved (and no homing was available)
2. The position data was stale — enemy wasn't actually there
3. The shot hit terrain/water (target behind obstacle without missile shots)

## What a miss is NOT

- Shields: return **positive hit** (dual, weapon_byte=1). Shields absorb damage but the shot registers as a hit.
- Corpses: return **positive hit**. A dead tank still "receives" the shot at the wire level.
- A miss does NOT mean "hit but no damage" — it means "nothing was at that coordinate."

[^1]: user (Austin), 2026-06-16 — full weapon selection explanation: "homing shot is used whenever you click on an enemy tank but the enemy had already submitted a move command... it will automatically use a homing shot"
[^2]: user (Austin), 2026-07-02 — "check the inventory delta for each shot. that is how we measure hits vs misses"; wire-verified in capture 2026-07-02 01:20 (orange-3 pursuit kill via 5 weapon=3 debits, orange-1 weapon=0 stream with zero debits)
