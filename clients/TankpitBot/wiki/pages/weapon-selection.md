---
title: Weapon Selection (Server-Side)
tags: [combat, weapons, protocol]
related:
  - "[[shoot-event-format]]"
  - "[[weapon-log-markers]]"
  - "[[shot-range]]"
source_paths:
  - "runs/bot"
  - "src/tankpit_bot/sim/combat.py"
source_git_blobs:
  "src/tankpit_bot/sim/combat.py": "5efd5cffbd5da642572539b483c42f628d95bb69"
fact_checked: "2026-07-02"
confidence: high
hubs: [combat, protocol]
---

# Weapon Selection

Weapon selection is **server-side**, not client-side. The fire command sends an (x,y) coordinate. The server decides which weapon to use based on what's at that tile.[^1]

## Selection rules

| What's at the tile | Weapon used | Byte | Effect |
|-------------------|-------------|------|--------|
| Enemy tank (stationary) | Dual shot | 0x01 | Hit, damage applied |
| Enemy tank (has pending move on same tick) | Homing shot | 0x03 | Auto-tracks to new position |
| Enemy tank with terrain OR another tank in the line of sight | Missile | 0x02 | Fires over the obstruction (missiles slot must be enabled) |
| Empty ground / terrain / water | Single shot | 0x00 | Miss — nothing there |

## Missile trigger rule (user contract 2026-07-20)

User (verbatim): "missiles only fire when you shoot at an enemy, on
the visible viewport, and there is terrain or a tank in between you
and the target enemy. friendly or foe inbetween will trigger
missiles. same if you shoot at an enemy on the other side of a rock
wall. you can use dual shots across water ofc."

So the obstruction test is: rock/terrain walls and ANY tank (friendly
or enemy) on the line of sight → missile; open ground and water are
NOT obstructions → dual/homing as usual. Movable concrete blocks are
also line-of-sight obstructions for non-missile shots (see
[[movable-blocks]]) — missiles shoot over them. Missile firing
requires the missile equipment slot (3) enabled; the bot currently
keeps it off (`tactics.py`), which is why `weapon=2` never appeared
in 204 archived bot sessions.

**Wire-verified 2026-07-20** (manual capture sniff-20260720-213208):
the user fired at enemies behind rock with missiles enabled — 10
shots echoed as `weapon=2` exactly as the contract predicts, cost 10
fuel + 1 missile each ([[game-economy]]).

**Missiles are ENEMY-ONLY (user contract 2026-07-21)**: "missiles
only work for enemies, not for shooting mines or the ground." A shot
at a mine or empty tile behind terrain does NOT escalate to a missile
— it fires a single that stops at the obstruction (wire-verified,
sniff-20260721-212348 t+169.91: click on a solo mine at (55,167)
behind a mountain echoed `weapon=0` with the CLIPPED impact tile
(46,165) on the shooter→click ray, billed the full −6, destroyed
nothing). Consequence: mines behind terrain cannot be shot from
cover; the bot must gain line of sight first. See
[[shoot-event-format]] for the impact-tile clipping semantics.

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
