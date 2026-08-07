---
title: Package Layering
tags: [architecture, refactor, imports]
related:
  - "[[coding-standards]]"
  - "[[module-map]]"
  - "[[inheritance-chain]]"
  - "[[session-state-deglobalisation]]"
source_paths:
  - "src/tankpit_bot"
fact_checked: "2026-08-06"
confidence: high
hubs: [architecture]
---

# Package Layering

`src/tankpit_bot` had **no layering at all**. A Tarjan SCC over the
package-level import graph put all 17 packages in ONE strongly-connected
component: every package could reach every other. There was no acyclic
order to violate, and 139 function-level imports existed largely to
paper over the resulting load-order problems.[^1]

## Leaf packages

Four packages now import no other `tankpit_bot` package (or one leaf).
They are the base every other layer rests on:

| package | outbound | holds |
|---|---|---|
| `types/` | none | wire/game vocabulary + TypedDicts |
| `wire/` | none | byte arithmetic, length validators, codec errors |
| `contracts/` | none | contract-enforcement machinery |
| `facts/` | `contracts` | `Fact[T]`, `FactSource`, provenance, confidence |

## Cycles removed

Each fix was a module move, not a rewrite: no logic changed, no
behavioural difference, no shim left behind.

**`protocol ↔ container`.** `container/encoders.py` reached into
`protocol/helpers.py` for `pack16`, while `protocol` imports `container`
because container messages are a message family decoded from inside
`protocol/decoders/tank.py`. `protocol/helpers.py` was not
protocol-specific at all — it is the byte layer both codecs sit on, with
zero `tankpit_bot` imports. Moved to the new leaf `wire/helpers.py` (39
import sites). `container` now has exactly ONE outbound edge, to `wire`.
The function-level `container` import inside `protocol/decoders/tank.py`
was a cycle workaround and is now hoisted to module scope.[^2]

**`facts ↔ state`.** `facts/` conflated two layers: the leaf vocabulary
(`fact`, `source`, `provenance`, `confidence`), which all eight
`state/types` modules import, and three projection modules
(`container_facts`, `tank_facts`, `world_facts`) that read `state.types`
back. `facts/__init__.py` already exported only the vocabulary half —
the projections were never in the barrel. Moved them to
`state/projections/`, which is where a read model over state belongs.[^3]

**`physics ↔ state`.** Two separate causes, both fixed by moving the
misfiled module rather than the dependency:

- `state/types/constants.py` was never state. It is the wire/game
  vocabulary — terrain codes, team codes, damage codes, ASCII glyphs,
  liveness literals — with no imports beyond `platform_core`. Moved to
  the leaf `types/constants.py`; both `state` barrels had their
  re-exports stripped rather than left as pass-throughs.
- `physics/line_of_sight.py` was the ONLY module in `physics/` importing
  `state` (and `_test_hooks`). The other five physics modules import
  nothing. Line-of-sight is a QUERY over terrain state — it reads
  `TerrainTileDict` and a `TerrainMapProtocol` — not a rule constant, so
  it moved to `state/line_of_sight.py`. `physics/` now depends only on
  `protocol`.[^4]

## Thin wrappers found and collapsed

`bot/ai/combat_strategy.py` carried four public/private pairs where the
public function was pure delegation — `def engage_target(ctx, target):
return _combat_shoot(ctx, target)` and three identical siblings for
map-open, teleport, and close. The private half held the real
documentation; the public half held a generic stub. Collapsed to one
function each, keeping the substantive docstring and grafting the
public's `Args:`/`Returns:` sections onto it.[^5]

## Counting caveat

A first pass reported 10 mutual pairs. That undercounted: an analyzer
that only follows `from tankpit_bot.X import ...` misses
`from tankpit_bot import X`, which is how `state/renderer.py` reaches
`_test_hooks`. Counting both forms, the true starting figure was **12**.
Any future layering guard must handle both import forms or it will
under-report the same way.

[^1]: Measured 2026-08-06 by AST-walking every module under
`src/tankpit_bot`, mapping each import to its top-level package, and
running Tarjan SCC. The single SCC contained `<top>`, `_test_hooks`,
`action_lab`, `bot`, `browser`, `capture`, `container`, `diagnostics`,
`facts`, `ledger`, `physics`, `protocol`, `service`, `sim`, `sniffer`,
`state`, `validate`. The 139 figure is function-level (deferred)
`tankpit_bot` imports across `src/`, counted the same day by the same
walk; not all are cycle workarounds — `protocol/decoders/tank.py` also
defers sibling decoder imports because the `decoders/__init__.py` barrel
re-enters during package init.
[^2]: `container/` outbound edges after the move: `wire` only, verified
by re-running the package graph. `pack16` is defined at
`src/tankpit_bot/wire/helpers.py`; the module has no `tankpit_bot`
imports. The hoisted import is at the top of
`src/tankpit_bot/protocol/decoders/tank.py`; it was previously inside
the 0x2E dispatch function.
[^3]: The three projection modules had NO source consumers — every
`state/types/*.py` mention of them is a docstring cross-reference, not an
import. Only two test modules imported them, now at
`tests/state/test_projection_entities.py` and
`tests/state/test_projection_world.py`. `facts/` outbound after the move:
`contracts` only.
[^4]: `physics/` module audit 2026-08-06: `capacity.py`, `combat.py`,
`costs.py`, `damage.py`, and `map.py` have zero `tankpit_bot` imports;
`supervisor.py` imports `protocol.constants`; `line_of_sight.py` was the
sole `state` importer. `line_of_sight`'s two claim bindings in
[[mine-mechanics]] moved with it, and `scripts/physics_claims.py` gained
`LINE_OF_SIGHT_MODULE` as a bare claim target — bare, so only its own
`__all__` (`is_shot_line_clear`, `shot_line_tiles`) is bound and reverse
coverage stays satisfied by the two existing law claims.
[^5]: The four pairs were `_combat_open_map`/`open_map_for_target`,
`_combat_teleport`/`teleport_to_target`, `_combat_close`/`close_target`,
`_combat_shoot`/`engage_target`. Each public body was exactly
`return _private(ctx, target)`. Collapsing them was required by the
standing no-thin-wrappers rule in [[coding-standards]]; it also removed
a genuine module cycle, because the approach stages and the fire stage
call each other and could not be separated while the wrappers existed.
