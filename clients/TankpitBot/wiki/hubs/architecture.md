# Architecture

Codebase design decisions, patterns, and coding standards.

[Inheritance Chain](../pages/inheritance-chain.md) -- Bot -> DispatchMixin -> CompletionsMixin -> SessionBase, composition over inheritance
[Coding Standards](../pages/coding-standards.md) -- no Any/cast/TYPE_CHECKING, no mocks, _test_hooks DI, MonkeyPatchBanRule
[Tank Freshness Model](../pages/tank-freshness-model.md) -- three independent freshness timestamps + observation-based mutator; the architecture that makes the stale-position combat bug impossible to reintroduce
[Bot Behavior Contract](../pages/bot-behavior-contract.md) -- MUST/MUST NOT/Verified-by table for every bot behavior; consult before proposing fixes; locks in anti-pattern prevention
[Self-Observing Bot Architecture](../pages/self-observing-architecture.md) -- fail-hard-on-state-entry philosophy, four-layer stack (Facts/Decisions/Ledger/Memory), the 15 blind spots the 20:47:31 deadlock exposed, phase roadmap
[Bot Service Architecture](../pages/bot-service-architecture.md) -- the SPA-driven long-running service: ModeBridge + StatusBus + SessionRunner primitives, five aiohttp routes, session lifecycle, DI hooks in service/_test_hooks.py
[Executor Rejection Silent Loops](../pages/executor-rejection-loops.md) -- structural pattern behind the 2026-07-06 20:47:31 deadlock class: AI-state rollback + unwired rejection paths let executor validators loop silently; three known live instances after Phase 0
