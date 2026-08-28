# Architecture

Codebase design decisions, patterns, and coding standards. (19 pages)

[Inheritance Chain](../pages/inheritance-chain.md) -- Bot -> DispatchMixin -> CompletionsMixin -> SessionBase, composition over inheritance
[Coding Standards](../pages/coding-standards.md) -- no Any/cast/TYPE_CHECKING, no mocks, _test_hooks DI, MonkeyPatchBanRule
[Tank Freshness Model](../pages/tank-freshness-model.md) -- three independent freshness timestamps + observation-based mutator; the architecture that makes the stale-position combat bug impossible to reintroduce
[Bot Behavior Contract](../pages/bot-behavior-contract.md) -- MUST/MUST NOT/Verified-by table for every bot behavior; consult before proposing fixes; locks in anti-pattern prevention
[Fleet Coordination](../pages/fleet-coordination.md) -- the shared knowledge layer: same-team bots exchanging beliefs via atomic knowledge.json files, focus-fire ranking, fighter/gatherer roles, color assignment
[Self-Observing Bot Architecture](../pages/self-observing-architecture.md) -- fail-hard-on-state-entry philosophy, four-layer stack (Facts/Decisions/Ledger/Memory), the 15 blind spots the 20:47:31 deadlock exposed, phase roadmap
[Bot Service Architecture](../pages/bot-service-architecture.md) -- the SPA-driven long-running service: ModeBridge + StatusBus + SessionRunner primitives, five aiohttp routes, session lifecycle, DI hooks in service/_test_hooks.py
[Executor Rejection Silent Loops](../pages/executor-rejection-loops.md) -- structural pattern behind the 2026-07-06 20:47:31 deadlock class: AI-state rollback + unwired rejection paths let executor validators loop silently; mine class killed at the root 2026-07-20, instances #2/#3 (stale anchors, pickup races) still open
[Terrain Composition — Single-Owner Walkability](../pages/terrain-composition.md) -- "can I walk here?" has ONE owner: the composed decision terrain (static map + ferries + hostile mines); why the blocked_mines parameter and the executor mine veto are gone, the invariant table, and the rule for future dynamic obstacles
[Physics Module Roadmap](../pages/physics-module-roadmap.md) -- the wiki-as-executable-truth plan: physics/ module + machine-checked wiki claim binding in make check (Phase 1, IMPLEMENTED 2026-07-20 — see as-built notes), validators vs the capture archive (Phase 2), live divergence counting (Phase 3), executor staleness track

[Committed Intent](../pages/committed-intent.md) -- plans that survive the tick boundary: `bot/ai/intent.py` as the single owner of collect-plan semantics (typed plan + completes-here + validity + reasoned release with `plan_released` events); phase 1 SHIPPED 2026-07-30 (s8-2 fix), phase 2 = hunt/clearance plans
[Diagnostic HUD + Human Flag Channel](../pages/diagnostic-hud.md) -- the fixed-geometry fiesta-styled in-page HUD (2026-07-29 rebuild) and the click-to-flag channel that lands a human_flag diagnostic with an 8-tick lead-up snapshot; includes the flag-tracing triage recipe
[Flag Triage 2026-07-29](../pages/flag-triage-20260729.md) -- first live flag session: 10 flags, 4 root causes (direction-blind top-off hop, 63% zero-yield hop churn, missing mine-shot clearance, mine-ring acquisition cloak), fix-status table

[Larder Plan](../pages/larder-plan.md) -- IMPLEMENTED and live-proven 2026-07-27: harvest radar-verified containers the bot already remembers as a COLLECT cascade priority; own-tile equipment pickup probe answered YES 3/3; the under-fire refuel now shares the same query

[Session-State De-globalisation](../pages/session-state-deglobalisation.md) -- the 17 modules holding per-session state at module scope, why the divergence exists (a 40-second package split froze an unfinished convergence), the archive measurements that make the cipher swap a lift, and the 11-step ordered plan whose completion criterion is deleting the conftest reset list

[Package Layering](../pages/package-layering.md) -- src/tankpit_bot had ONE strongly-connected component containing all 17 packages; the four leaf packages (types, wire, contracts, facts), the three cycles removed by moving misfiled modules (protocol/container, facts/state, physics/state), the four thin wrappers collapsed in combat_strategy, and why a layering guard must count `from tankpit_bot import X` too

[Quad-Sweep Doctrine](../pages/quad-sweep-doctrine.md) -- stationary recon then loop harvest: four ATOMIC scope shifts from one anchor tile the 31x31 the ANCHOR law reaches, each paying one extra radar; the harvest loop orders clusters by walkable distance and exits by teleport to the next block anchor; archive-mined proof that a shift-framed window is the acceptance window, so pickups need no teleport frame
[Guard Mutation Sweep](../pages/guard-mutation-sweep.md) -- replacing a guard's return with pass and asking whether the suite notices: 18% of 474 guards survived at 100% coverage, the four resolutions and why "structural" pins nothing, and the two faults 6,000 tests could not see
[Project History](../pages/project-history.md) -- the eight-month arc: which package arrived when, what superseded what, and the three lessons learned more than once
