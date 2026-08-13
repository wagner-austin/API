---
title: Guard Mutation Sweep
tags: [testing, quality, method]
related:
  - "[[testing-patterns]]"
  - "[[coding-standards]]"
  - "[[decode-coverage]]"
source_paths:
  - "tests/sniffer/test_dispatch_exclusivity.py"
  - "src/tankpit_bot/sniffer/constants.py"
  - "src/tankpit_bot/validate/roundtrip.py"
source_git_blobs:
  "tests/sniffer/test_dispatch_exclusivity.py": "2cba91a2286da57d48361cbfdc0784a95884f9f9"
  "src/tankpit_bot/sniffer/constants.py": "fe859e16ecdc3670bdc9e150e290dc607d1bac72"
  "src/tankpit_bot/validate/roundtrip.py": "b57592a4a886e62d202674421642d65ac38ddb49"
fact_checked: "2026-08-12"
confidence: high
verified: "every claimed kill re-mutated against the final tree (31/31); structural cases re-mutated and confirmed still surviving"
hubs: [codebase, architecture]
---

# Guard Mutation Sweep

Replace one defensive guard's `return` with `pass`, run the suite, and ask whether anything noticed. **Killed** means a test failed. **Survived** means no test in ~6,000 could tell the guard from absent.

**A covered line is not a pinned line.** The suite held 100% statement and branch coverage throughout, and 18% of guards survived (76 of 474). A second pass over the 106 guards the first collector structurally could not see — its `if` body had to be exactly one statement, so anything that logged before returning was invisible — survived at 25% (27 of 113).[^1] Guards that log before returning are the least pinned code in the repo, because a log line is the kind of effect assertions skip.

## Four resolutions, and only three of them pin anything

| Category | Action |
|---|---|
| **Redundant** | Delete. A downstream check already decides it. |
| **Wrong-answer untested** | Test the case. The return value differs; nobody asked. |
| **Side-effect-only** | Test the side effect. Return value identical, log/ledger/emission differs. |
| **Structural** | Pin the enabling invariant. The mutant is genuinely unobservable. |

Structural is not "pinned", and the distinction matters more than the total. Of the 27 second-pass survivors, **14 were killed and 13 were structural**: dispatch-chain and cascade arms whose removal changes nothing because no arm can match after an earlier one did. Verified, not assumed — removing all four scorecard cascade returns and re-routing 1,403,706 archived records produced byte-identical accumulators; removing the three `sim/ghost` returns recompiled all 34 capture sessions to byte-identical ghost specs.[^2] No test can kill those. What is enforced instead is the property that makes them unobservable: the arms must test pairwise-distinct values, checked by parsing the source.[^3] **Delete one of those 13 returns and nothing fails.**

## Rules that earned their place

- **Probe before classifying.** Reading the code produced the wrong classification roughly 40% of the time. Three guards labelled load-bearing were redundant; one docstring asserting an existing test caught a regression was disproved by injecting that regression.
- **Paired controls.** Every "nothing happened" assertion needs a companion proving the setup *can* produce something. Three controls failed on first attempt, each revealing the test was not reaching the code it named.
- **Negative-control the detector.** A source-parsing check that silently reads nothing passes for the wrong reason. Re-run the control after refactoring detection logic — a rewrite can stop detecting without any test noticing.
- **Verify against a green baseline.** The harness reports KILLED whenever the mutant run fails *for any reason*, so a broken test in the same file manufactures a false kill. This happened once.

## Two faults the suite could not see

A survived mutant in `_capture_static_key` truncated the tracked 160 KB `tpclient.js` to zero bytes while every test passed: a fetch returning nothing became `""` and was written to a CWD-relative path, and the guard was the only thing keeping two tests off the real filesystem.[^4] Separately, `MSG_MIN_LENGTHS` admitted `0x45` and `0x4B` — container-only subtypes with no top-level decoder — so membership promised something three consumers relied on and it was false.[^5]

Loop **exit routes** cannot be swept against the whole suite: removing one leaves the loop non-terminating, one hanging test consumes the per-mutant timeout, and the guard retires with no verdict. Targeting only the files that drive the loop makes the hang land in seconds and attributes it — all five such guards are killed by non-termination.[^6]

[^1]: sweep of `src/tankpit_bot/**/*.py` early-return guards, 2026-08-11/12; second pass collected `if` bodies of 2+ statements ending in a bare/None/False return.
[^2]: 426 archived `runs/bot/*.events.jsonl` re-routed through `route_scorecard_record`; 34 `runs/sniff/*.capture_session.json` recompiled through `compile_ghost_spec`.
[^3]: commits `df7f9ef0`, `6efbab0a` — `_CASCADES` in `tests/sniffer/test_dispatch_exclusivity.py`; duplicate-arm injection fails the test.
[^4]: commit `9a7a63cc` — `src/tankpit_bot/browser/lifecycle.py`, `_capture_static_key`.
[^5]: commit `c25891d5` — both entries removed; invariant enforced by `test_every_declared_type_is_decodable_at_top_level`.
[^6]: `tick_loop.py` 156/169/173/178 and `action_lab/enemy_teleport.py:112`, verified 2026-08-12 against `tests/bot/test_tick_loop_lifecycle.py`, `test_tick_loop_crash.py`, `test_enemy_teleport_settle.py`.
