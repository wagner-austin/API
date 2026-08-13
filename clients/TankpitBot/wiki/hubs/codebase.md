# Codebase

How the code is organized, how the services wire together, how to test, and how to run things. Read `module-map` first for orientation, then drill into the topic you need.

[Module Map](../pages/module-map.md) -- top-level packages, what each owns, dependency flow
[Services](../pages/services.md) -- WorldService, CDPService, CommandService: ownership, injection, factory wiring
[Testing Patterns](../pages/testing-patterns.md) -- _test_hooks DI, protocol-matching implementations, MonkeyPatchBanRule
[Guard Mutation Sweep](../pages/guard-mutation-sweep.md) -- replacing a guard's return with pass and asking whether the suite notices: 18% of 474 guards survived at 100% coverage, the four resolutions and why "structural" pins nothing, and the two faults 6,000 tests could not see
[Make Targets](../pages/make-targets.md) -- what each target does, which ones touch live servers, safe defaults
[Adding a Probe](../pages/adding-a-probe.md) -- step-by-step: create probe, wire factory, add CLI, write tests
[Project History](../pages/project-history.md) -- the eight-month arc: which package arrived when, what superseded what, and the three lessons learned more than once
