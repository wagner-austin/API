---
title: Session-State De-globalisation
tags: [architecture, refactor, sessions, plan]
related:
  - "[[services]]"
  - "[[bot-service-architecture]]"
  - "[[coding-standards]]"
  - "[[module-map]]"
source_paths:
  - "src/tankpit_bot/sniffer"
  - "src/tankpit_bot/ledger"
  - "src/tankpit_bot/capture/xor.py"
  - "src/tankpit_bot/protocol/codec.py"
  - "tests/conftest.py"
source_git_blobs:
  "src/tankpit_bot/sniffer": "9d5d8223ba026a595b4c5ff9fd9b0953cf6987ae"
  "src/tankpit_bot/ledger": "f39b02960545eebbd7da5b879f4eaa63a4649b94"
  "src/tankpit_bot/capture/xor.py": "43df3e756872949f3ebe0571afb7f4a3d53c6880"
  "src/tankpit_bot/protocol/codec.py": "76ba3790ab90e303383c2f7a66dd48c96e30467c"
  "tests/conftest.py": "a6d34fab76028fe59e8e24fc684c138211184262"
fact_checked: "2026-08-07"
confidence: high
hubs: [architecture]
---

# Session-State De-globalisation

Seventeen modules hold **per-session** state at module scope. Each is a
place where two sessions in one process would overwrite each other, and
each is a value that logically belongs to one game session and nothing
else. This page is the inventory, the evidence, the archaeology of how
it happened, and the ordered plan to finish it.[^3]

The plan is not conditional on multi-bot. `tankpit-fleet` already runs
N bots as child processes, deliberately, so an orchestrator crash
cannot kill a live tank ([[bot-service-architecture]] fleet section).
That makes this work a correctness and architecture change: a module
global holding one session's cipher is wrong whether or not a second
session exists in the process.[^1]

## The proof it is real: the reset list

`tests/conftest.py` carries a ten-call reset between every test, plus
more in sibling fixtures. It exists precisely because this state leaks
across "sessions" — the test suite has been simulating the multi-session
problem all along, and the resets are the workaround.[^2]

Deleting that reset list is the completion criterion for this work. If
state is instance-scoped, no reset is possible to forget.[^2]

**Progress: 10 calls → 8 (list length), of which 7 are session state.** Step 1 removed `reset_xor_state` and put
`reset_static_key_cache` in its place, so the count is unchanged at
first glance — but the two are not the same kind of thing. The old call
reset a *session's* cipher; the new one resets a *process-wide key
cache*, which by the "Legitimately process-level" test below stays. The
honest count of session-state resets is therefore nine, and the
replacement is a line the finished refactor will still carry. Five
per-file `_isolate` fixtures in `tests/action_lab/` that duplicated the
same reset were deleted outright; one of them documented the exact leak
this step removed — "the replay harness builds a global XOR table from
the capture's magic; without an explicit teardown reset, that table
would leak into subsequent tests that decode bytes with a different
key."[^13]

## Inventory

Two mechanisms hold module state: rebinding (`global` declared) and
in-place mutation of a module container (no `global` needed). Both are
counted.[^3]

### Per-session — must be de-globalised

| module | state | mechanism |
|---|---|---|
| ~~`sniffer/xor.py`~~ | ~~`_global_xor_table`, `_global_static_key`~~ | **module deleted, step 1** |
| `sniffer/world_state.py` | `_service` (a `WorldService`) | rebind |
| ~~`sniffer/viewport.py`~~ | ~~`_viewport_left`, `_viewport_top`~~ | **module deleted, step 2** |
| `sniffer/trackers.py:41` | `ALL_TRACKERS` (tuple of tracker instances) | in-place |
| `ledger/events.py` | `_event_counter` | rebind |
| `ledger/outcome/teleport.py` | `_pending` | rebind |
| `ledger/outcome/_emit.py:27-29` | `_attempt_counters`, `_pending_decisions`, `_resolved_decision_ids` | in-place |
| `ledger/ring.py:51` | `_rings` | in-place |
| `ledger/decision.py:60` | `_decisions` | in-place |
| `ledger/mode_transition.py:38` | `_transitions` | in-place |
| ~~`bot/ai/collect_common.py`~~ | ~~`_blacklisted_container_keys`~~ | **deleted, step 7 — never had a writer** |
| ~~`diagnostics/self_alignment.py`~~ | ~~`_last_emitted_belief`~~ | **instance state, step 3** |
| ~~`diagnostics/entity_alignment.py`~~ | ~~`_last_emitted_signature`~~ | **instance state, step 3** |
| ~~`browser/cdp_utils.py`~~ | ~~`_cdp_time_offset_ms`~~ | **instance state, step 4** |
| ~~`browser/client_structure.py`~~ | ~~`_survey_emitted`~~ | **instance state, step 5** |
| `runtime_logging.py` | `_BOT_ARTIFACTS`, `_SNIFF_ARTIFACTS`, `_PROBE_ARTIFACTS` | rebind |
| `runtime_context.py:25` | `_RUNTIME_CONTEXT_TICK_N`, `_..._BOT_STATE`, `_..._IN_FLIGHT_ACTION_KIND` | rebind |

### Legitimately process-level — NOT in scope

| module | state | why it stays |
|---|---|---|
| `sniffer/decoders.py` | `_PROTOCOL_FRAME_LOGGING_ENABLED` | a logging switch for the process, not a session fact |
| `bot/tick_loop.py` | `_INTERRUPT_REQUESTED` | Ctrl+C is a process signal |
| `analysis/_test_hooks.py` | `read_text`, `list_session_paths` | the DI seam itself ([[testing-patterns]]) |
| `capture/xor.py` `_static_key_cache` (`:28`) | the key read from `xor_static_key.txt` | identical for every session; caching it is a property of the key. (Was `sniffer/xor.py`, deleted in step 1; the cache moved, the reasoning did not.) |

## Why it exists: an interrupted convergence

This is not two competing designs. It is **one design that never
finished converging**, frozen in place by a package split.[^4]

The pre-split `sniffer.py` monolith (3,204 lines) already contained
BOTH idioms at once: 73 references passing an `xor_table` explicitly —
the trackers, which each hold `self._xor_table` — and a module global
introduced later under the comment `# Module-level XOR table for
unified decoder`. The global was a shortcut at one call site: the
unified decoder was a free function with no object in scope, and
threading the table would have meant touching the whole chain.[^4]

Then two commits **40 seconds apart** on 2026-01-12 split the monolith:
`a1c32e9a` at 13:04:18 carved out `sniffer/`, taking the global into
`sniffer/xor.py`; `389231df` at 13:04:58 added `capture/`, taking the
table-passing helper into `capture/xor.py`. Neither commit created the
divergence — they made it structural, turning two local helpers in one
file into two modules with two APIs in two packages, where nobody had
reason to notice they were the same function twice.[^4]

`get_world_service()` is the same shortcut at the same kind of call
site. `WorldService`'s own docstring records that the 16 world globals
were ALREADY collapsed into per-session instance state, and
`dispatch_world_state_update(ws, ...)` already receives an instance —
the singleton is what is left of the unfinished half.[^5]

## Evidence gathered before touching anything

Three measurements, all archive-wide, all 2026-08-06:[^6]

- **The global decoder's bounds-tolerance is dead.** The XOR table is
  1000 bytes; the longest real frame payload is 931; **0 of 279,771**
  payloads exceed the table. So `xor_decode(body)` and
  `xor_decode_body(body, table, offset=1)` are equivalent on all real
  traffic, and the swap is a lift rather than a rewrite.[^6]
- **Strict framing costs nothing.** 217,678 received payloads split
  cleanly under `split_frames` with zero raises; 62,095 sent payloads
  raise exactly twice, both inside `bot-20260331-230406`.[^6]
- **`unframed_payload` is unreachable.** Every capture with unframed
  sent payloads also lacks magic, and the magic check returns first, so
  that skip arm cannot fire on the archive.[^7]

## Call-site sizing

| target | call sites | files | note |
|---|---:|---:|---|
| `xor_decode` | 8 | 6 | `sniffer/decoders.py` is the only live one |
| `get_world_service()` | 74 | 23 | re-counted 2026-08-07; the old "21 in `bot/tick_loop.py` alone" no longer holds — that file was split, and the sites now spread across `tick_combat_feedback.py` (14), `tick_loop_command_errors.py` (6), `executor.py` (6), `tick_body.py` (5) and `bot_dispatch.py` (5) |

The live chain is shallower than the count suggests:
`drain_messages(bot)` already holds the session, and
`dispatch_world_state_update(ws, ...)` already takes the instance. Only
the boundary function in between reaches for globals.[^5]

**Calibration from step 1: this table undercounts by about 10×.** The
XOR row predicted 8 sites across 6 files; the commit touched **79 files
— 21 under `src/`, 57 under `tests/`.** The estimate was not wrong
about `xor_decode` itself, it was wrong about what a signature change
costs: threading one parameter through `process_received_message`
rippled into `_test_hooks` protocol members, every fake that satisfies
those protocols, and every fixture that built a table. The `src` figure
(21 vs 6) is the honest error in the *reading*; the `tests` figure is
the part the table never modelled at all. Read the `get_world_service`
row with the same correction: 74 sites in 23 files is the *src* floor
for step 8, not its size.[^12]

## Plan

Ordered by coupling, least first. Every step ends with `make check`
green and a commit, so a collision with concurrent work costs one step
rather than the refactor.[^3]

1. ~~**XOR cipher.**~~ **SHIPPED 2026-08-06** (`7481cdca`). Deleted
   `sniffer/xor.py` outright — no shim, no deprecation. Its callers now
   take the table as a parameter, built by the new
   `capture/xor.py::build_session_xor_table` and stored on
   `SessionBase`, which hands the *same* table to `CommandService`
   instead of building it twice from identical inputs.[^8]

   Three silent failures became loud, which is the real payoff:[^14]

   - A missing `xor_static_key.txt` now raises
     `XorStaticKeyUnavailableError`. `build_global_xor_table` returned
     early leaving the table `None`, and `xor_decode` then returned
     `body[1:]` **undeciphered** — garbage that decoded into plausible
     world state rather than an error.
   - `drain_messages` returns 0 and keeps the buffer while the session
     has no table, instead of dispatching pre-magic frames through that
     same identity decode.
   - `WebSocketSniffer` overrode `_on_magic_captured` *without*
     `super()`, so live decode depended on the global that
     `init_trackers_with_magic` built as a side effect. The override is
     gone.

   Two tests had been asserting the silent behaviour and now assert the
   real cipher — `test_binary_promotion_takes_binary_route` read "XOR
   table is None in tests so `body[1:]` passes through verbatim", and
   `tests/replay/test_script.py`'s fake filesystem carried no key file
   at all. Both are stronger tests now than the ones they replace.[^14]

   Carried in the same commit because they were the same edit: three
   redeclarations of `_cdp_message_buffer` folded into inheriting
   `BufferedMessageSourceProtocol`, and the `build_global_xor_table`
   parameter dropped from `prepare_probe_replay` — all five harnesses
   passed the production function, so it was a seam with exactly one
   implementation. `reset_xor_state` left the conftest reset list
   (step 11's first entry to fall); `reset_static_key_cache` took its
   place, because the **key** is process-wide — one key builds every
   session's table — while only the **table** was session state.[^12]
2. ~~**`sniffer/viewport.py`**~~ **SHIPPED 2026-08-07 — by DELETION,
   not by threading.** The step was written as "move the two globals
   onto the session's viewport state". They were already there. The
   single writer (`sniffer/world_state_tiles.py`) called
   `update_viewport_origin(left, top)` and then, on the very next
   statement, wrote the same pair into `ws.world_state["viewport"]`
   via `make_visible_viewport_state`. Every production consumer reads
   the world-state copy through `viewport_visible_bounds`;
   `get_viewport_left()` / `get_viewport_top()` had **zero callers in
   `src/`, `scripts/` or `analysis_scripts/`** — only tests. The
   module was write-only in production, and `reset_viewport_tracking()`
   (called at four session boundaries) reset state nobody consulted.
   Deleted outright: the module, its single write, its four resets, and
   the two tests that existed only to cover it. One test
   (`test_world_state_functions.py`) had been setting BOTH the global
   and the world-state copy — the assertion depended only on the
   latter, which is the duplication in miniature.

   **This does not move the completion criterion.** These globals were
   never in the conftest reset list, so the count stays at nine. Step 2
   removed a duplicate, not a leak.
3. ~~**Diagnostics dedupe memories**~~ **SHIPPED 2026-08-07.** Both
   gates became `SelfAlignmentEmitter` / `EntityAlignmentEmitter`,
   constructed by `Bot` and called as `bot._self_alignment.maybe_emit(...)`.
   Both `reset_*_emitter` functions are DELETED, and the two tests that
   proved "reset clears the gate" now prove the stronger property a
   reset can never prove: a SECOND emitter emits the same belief. The
   sibling `tests/diagnostics/conftest.py` fixture lost both calls.
4. ~~**`browser/cdp_utils.py`**~~ **SHIPPED 2026-08-07.** The offset
   became `CDPClock`, owned by `CDPService` — which was already
   per-session, so the state simply moved to where its lifetime already
   was. The anchor matters per session: CDP timestamps are monotonic
   seconds from an arbitrary origin, so a second session reading
   through the first session's offset would misdate every frame.
   `reset_cdp_time_offset` is deleted along with its four call sites.
5. ~~**`browser/client_structure.py`**~~ **SHIPPED 2026-08-07.**
   `_survey_emitted` became `ClientStructureSurveyor`, owned by `Bot`.
   "Once per session" now means once per SESSION rather than once per
   process. **This is the first step to move the completion criterion:**
   `reset_client_structure_survey` was in the ten-call conftest list,
   so that list is now NINE calls (eight session resets plus the
   process-wide `reset_static_key_cache`).
6. **Ledger cluster** — events counter, teleport `_pending`, `_rings`,
   `_emit` trackers, `_decisions`, `_transitions`. These form one
   bookkeeping layer and move together.
7. ~~**`bot/ai/collect_common.py`**~~ **SHIPPED 2026-08-07 — by
   DELETION.** The container blacklist was never de-globalised because
   it was never alive: `blacklist_container` has **no caller in
   `src/` in any commit in this repository's history**, and
   `reset_container_blacklist` — whose docstring said it ran "on
   death/respawn" — had none either. Only tests called both. The
   reader `is_container_blacklisted` therefore always answered False,
   so the five decision sites consulting it (two hop selectors, the
   equipment pickup, the quad sweep, the scope scout) were filtering
   on a set that could never fill, and the `is_blacklisted` predicate
   threaded through `larder.select_fuel_larder_hop` carried the same
   nothing. Removing all of it is behaviour-identical.

   **This is the second step to move the completion criterion:**
   `reset_container_blacklist` was in the conftest list, so the list
   is now EIGHT calls. If per-session blacklisting is wanted, it needs
   a writer first — a reader without one is a decision nobody makes.
8. **`WorldService`** — delete `_service` and `get_world_service()`;
   thread the instance through the 74 sites. The largest step.

   **Decode boundary cut 2026-08-07.** `process_received_message`,
   `_process_single_message`, `try_decode_binary` and
   `try_decode_received` now take the session's service. The replay
   hook does not: threading it through `_test_hooks` closes an import
   cycle (`state → … → _test_hooks → sniffer.world_service → state`),
   so `_real_process_received_message` resolves the service inside the
   function body with the reason recorded there. That is the last
   singleton reach on the replay path and it falls with this step's
   final flip.

   **Test-side migration 2026-08-07: `reset_world_state()` 496 → 3.**
   The blocker was believed to be 496 call sites needing a hand-built
   service. It was not. `tests/conftest.py::_isolate_protocol_singletons`
   is autouse and already resets the singleton before *and* after every
   test, so **444 of the 496 were dead ritual** — 195 prologue calls,
   229 whole `setup_method`/`teardown_method` bodies, and 20 epilogues,
   plus 20 `try/finally` blocks that existed solely to guarantee a reset
   the fixture already guaranteed. Deleting them removed 202 lifecycle
   methods, 212 unused imports, and the now-empty `real_inventory` and
   `_isolate_world_state` fixtures. The test count did not move: 6191
   before, 6191 after.

   What made the deletion safe to prove rather than guess: **no fixture
   anywhere populates world state.** All seven `conftest.py` files were
   checked — the only writers are the root autouse reset and two
   `action_lab` fixtures that reset-and-yield without populating. Had
   any fixture seeded state, a test's prologue reset would have been
   deliberately wiping it, and deleting the reset would have silently
   changed what the test measured.

   `tests/sniffer/test_replay_pipeline.py` is now fully off the
   singleton: its helpers return the `WorldService` they decoded into.
   The five "reset clears X" tests were rewritten to assert the durable
   invariant — *a freshly constructed service starts clean* — which is
   the property that outlives `reset_world_state` rather than dying
   with it.

   **The 3 survivors, and why.** Two are the conftest fixture itself
   (the seam). The third is
   `tests/bot/test_executor_dispatch.py:288`, which is genuinely
   load-bearing: `_make_bot()` seeds position and fuel, and the test
   needs *no* self-belief, so it wipes the service after construction.
   It unblocks when `Bot` takes a `WorldService` — src-side work, not
   test work.

   **The estimate was wrong in the useful direction.** The step-1
   calibration note above warns that the table *undercounts* by ~10×.
   Here the opposite held: the test-side figure overcounted by ~150×,
   because it counted call sites without asking whether they did
   anything. Count what a call *does*, not that it appears.
9. **`sniffer/trackers.py`** — `ALL_TRACKERS`.
10. **`runtime_logging.py` + `runtime_context.py`** — per-session
    artifacts and context. The three tick-context globals moved to
    `runtime_context.py` when `runtime_logging.py` was split
    (2026-08-07); `runtime_context.py` is their sole owner and the
    emitter reads them through `get_runtime_context()`, so the split
    did not duplicate them.
11. **Delete the conftest reset list.** Its removal is the proof.

## Found while shipping step 1: the cipher is forked four ways

Deleting `sniffer/xor.py` removed one fork and revealed that it was
never the only one. Four implementations of the same XOR existed, and
they did **not** agree at the edges:[^11]

| site | signature | past the table end | status |
|---|---|---|---|
| `protocol/codec.py::xor_bytes` | `(table, data, offset=0)` | raises `ValueError`, named and explicit | survivor |
| `capture/xor.py::xor_decode_body` | `(body, table, offset=0)` | raises `IndexError`, incidentally | remains |
| `diagnostics/capture_audit.py::_xor_with_table` | `(body, table)`, offset fixed at 1 | passes through in the clear | remains |
| ~~`sim/transport.py::_xor_with_table`~~ | ~~`(table, data)`, offset 0~~ | ~~passes through in the clear~~ | **deleted** — calls `capture.xor.xor_decode_body` |

Note the argument order flipped between them — `(body, table)` in two,
`(table, data)` in the other two — so a wrong-order call type-checked
cleanly and silently produced garbage. The `sim/transport.py` fork is
gone (2026-08-07): it now imports `xor_decode_body` and passes
`offset=1` explicitly at its client-command site, so one of the two
argument orders is retired.[^11]

The duplicated static-key path is **also closed**.[^12] Both
`capture/xor.py` and `protocol/codec.py` used to compute it with the
byte-identical expression
`Path(__file__).parent.parent.parent.parent / "xor_static_key.txt"`;
`capture` inlined it inside `load_xor_static_key`. That function is
deleted — `capture/xor.py` now imports `build_xor_table`,
`load_static_key` and `DEFAULT_STATIC_KEY_PATH` from `protocol.codec`
(`:21-22`, used at `:80` and `:103`) and keeps only the
session-scoped concern: build the table for ONE session's magic, cache
the process-wide key behind it, and the base64 helpers. Its module
docstring now says so.

**Two forks left, not four.** Folding them into `protocol/codec.py` is
still its own step: the pass-through tail is a real semantic
difference, measured dead only for the received-decode path (0 of
279,771 archived payloads exceed the 1000-byte table), and that
measurement says nothing about the audit reader — the one remaining
site that relies on it.[^11]

## Carried in the same sweep

Two defects found while mapping this, both in scope:[^9]

- **An eleventh fork of the frame walk**, this time in production code:
  `sniffer/decoders.py::process_received_message` re-derives
  `data[offset] | (data[offset + 1] << 8)` instead of calling
  `split_frames`. The other ten forks were in `analysis_scripts/`.[^9]
- ~~**No file-size enforcement.**~~ **CLOSED 2026-08-07.**
  [[coding-standards]] sets a 400-600 line ceiling and the 2026-07-31
  log entry recorded a backlog of 40 files over 600. Measured
  2026-08-06: **77**, and all four of the worst named in that backlog
  had GROWN — `test_fuel_probe.py` 2698→2948, `test_cdp.py` 2400→2816,
  `world_state_dispatch.py` 1156→1280, `tick_loop.py` 1037→1211.
  Nothing in `make check` checked file size, so the rule was
  documented and unenforced. `scripts/file_size_rules.py` now runs in
  the guard with no allowlist and no baseline, the backlog is 0, and
  the two files named here as the worst offenders are
  `world_state_dispatch.py` at 471 lines and `tick_loop.py` at 516 —
  both split rather than trimmed. That ordering was the right one for
  the reason given: they are the two heaviest consumers of the globals
  this page removes, and splitting them first means step 8 threads the
  `WorldService` instance through modules that are already
  single-purpose.[^10]

[^1]: `src/tankpit_bot/service/fleet.py:1-28` — the fleet manager's module docstring states the process-per-bot rationale verbatim: it "runs in a terminal the operator owns, so an orchestration harness dying can never kill a live tank (the 41-kill session died at 46 minutes exactly that way)". Registered as `tankpit-fleet` in `pyproject.toml`.
[^2]: `tests/conftest.py` — the `_isolate_protocol_singletons` autouse fixture calls, in order: `reset_world_state`, `reset_xor_state`, `reset_event_ids`, `reset_action_outcome_tracking`, `reset_outcome_rings`, `reset_teleport_dispatch_tracking`, `reset_decision_records`, `reset_mode_transitions`, `reset_container_blacklist`, `reset_client_structure_survey`. Its own docstring gives the reason — "tests that run later on the same xdist worker can decode bytes with a stale XOR key or read containers seeded by a prior run". Sibling fixtures `_restore_hooks` and `_restore_runtime_logging_state` reset the hook surface and the runtime-logging artifacts.
[^3]: Inventory taken 2026-08-06 by two sweeps. Sweep one, every `global` declaration under `src/tankpit_bot`: 13 modules, listed in the table. Sweep two, module-level mutable containers mutated in place — which need no `global` and so are invisible to sweep one: `src/tankpit_bot/ledger/ring.py:51` (`_rings`), `src/tankpit_bot/ledger/decision.py:60` (`_decisions`), `src/tankpit_bot/ledger/mode_transition.py:38` (`_transitions`), `src/tankpit_bot/ledger/outcome/_emit.py:27-29` (three), `src/tankpit_bot/bot/ai/collect_common.py:15` (`_blacklisted_container_keys`), and `src/tankpit_bot/sniffer/trackers.py:41` (`ALL_TRACKERS`). A `global`-only sweep undercounts by those six modules, which is why both were run.
[^4]: `git show a1c32e9a~1:...sniffer.py` — the pre-split monolith, 3,204 lines. `_global_xor_table` is declared at line 1827 under the comment "# Module-level XOR table for unified decoder", with `_build_global_xor_table` at 1843 and the private `_xor_decode` at 1855; the same file carries 73 `xor_table` references for the table-passing path. Split commits: `a1c32e9a` "Split sniffer.py into sniffer/ package" (13:04:18) and `389231df` "Add capture/ package" (13:04:58), both 2026-01-12.
[^5]: `src/tankpit_bot/sniffer/world_service.py:80-86` — `class WorldService`, docstring: "Owns all mutable game state for one session. Instance attributes mirror the 16 module-level globals that were previously in ``world_state.py``. Dispatch modules receive a ``WorldService`` instance and mutate it directly." The remaining singleton is `sniffer/world_state.py:19`, `_service = WorldService()`. Live chain: `bot/world_sync.py::drain_messages(bot)` -> `sniffer/decoders.py::process_received_message` -> `_process_single_message`, which is the only step that reaches for the globals.
[^6]: Measured 2026-08-06 over `runs/bot` (287 captures) with the production primitives: static key and table both 1000 bytes (`capture/xor.py::build_xor_table`), longest frame payload 931 bytes, 0 of 279,771 payloads longer than the table. Framing: 217,678 received payloads split cleanly under `protocol/framing.py::split_frames` with zero `FramingError`; 62,095 sent payloads raised twice. Re-derivable by re-running the sweep over the archive; the counts are not stored.
[^7]: Same sweep: the only capture with unframed sent payloads is `bot-20260331-230406`, which also carries `magic: null`. `analysis/scan.py::scan_session` checks magic first and returns `no_magic`, so the `unframed_payload` arm cannot be reached from the archive.
[^8]: Written before the step, describing the state it removed: `src/tankpit_bot/browser/session_base.py::_on_magic_captured` built the table twice from the same inputs — `init_trackers_with_magic(magic)` (which called `build_global_xor_table`) for the decode side, and `self._commands.xor_table = build_xor_table(static_key, magic)` for the send side. `CommandService.xor_table` at `bot/command_service.py:62` was already instance state, consumed at `:76`. As of `7481cdca` the method builds one table and assigns it to both.
[^12]: Commit `7481cdca` "Make the XOR table a session value, not a module global", 2026-08-06. `git show --stat` reports **79 files changed, 611 insertions(+), 588 deletions(-)** — 21 paths under `src/`, 57 under `tests/`, plus `scripts/decode.py`. Verification at that commit: `make lint` reports 0 violations across every guard rule (as of 2026-08-07: 31 from the monorepo orchestrator, plus eight local rules in `scripts/guard.py:127-136` that run unconditionally and report only on violation, two of which -- `file_size_rules` and `layer_rules` -- were added by the layering work); `python -m pytest -n auto` reports 5939 passed. Three failures remained in `tests/test_check_undecoded_fields.py`, caused by a concurrent session's in-flight split of `protocol/types.py` into a package — `scripts/check_undecoded_fields.py` `DEFAULT_TARGETS` still named the deleted file. **Fixed 2026-08-07** with the split it belonged to: the checker's targets accept a package directory as well as a module, through the existing `glob_paths` DI hook rather than a second code path.
[^14]: `tests/sniffer/test_decoders.py::TestProcessReceivedMessage::test_binary_promotion_takes_binary_route` carried the comment `# XOR table is None in tests so body[1:] passes through verbatim.` (removed in `7481cdca`); it asserted `self_state["rank"] == 5` from a body written as plaintext. It now builds that body through `make_binary_payload`, which ciphers it under the same table the decoder is handed, so the assertion exercises the real cipher round-trip instead of an identity pass. `tests/replay/test_script.py::_install_fake_fs` installed a `_FakeFS` with no `xor_static_key.txt` at all, which the old builder tolerated by leaving the table `None`; it now seeds the key and resets the cache, and its three `TestMainCLI` tests fail loudly without it.
[^13]: `tests/conftest.py::_isolate_protocol_singletons` after `7481cdca`: the reset list runs `reset_world_state`, `reset_static_key_cache`, `reset_event_ids`, `reset_action_outcome_tracking`, `reset_outcome_rings`, `reset_teleport_dispatch_tracking`, `reset_decision_records`, `reset_mode_transitions`, `reset_container_blacklist`, `reset_client_structure_survey` — ten calls, of which `reset_static_key_cache` guards a process-wide key cache rather than session state. The five deleted per-file duplicates were `_isolate` / `_isolate_world_state` in `tests/action_lab/test_equipment_probe_branches.py`, `test_replay_equipment_probe.py`, `test_replay_fuel_probe.py`, `test_replay_movement_probe.py`, and `test_replay_teleport_probe.py`; the quoted docstring is the one that stood in `test_replay_equipment_probe.py`, `test_replay_fuel_probe.py`, and `test_replay_movement_probe.py` verbatim.
[^9]: `src/tankpit_bot/sniffer/decoders.py::process_received_message` — the frame loop re-derives the 2-byte little-endian length prefix inline rather than calling `protocol/framing.py::split_frames`, which has owned that arithmetic since the protocol layer was written (`decode_frame_header`, `:37`).
[^10]: Counted 2026-08-06 with `find src scripts tests -name '*.py' | xargs wc -l | awk '$1>600'`: 77 files. The 2026-07-31 backlog of 40 is enumerated at `wiki/log.md:2437`. At that count `bot/tick_loop.py` held 21 of the 73 `get_world_service()` call sites; after the split the total is 74 across 23 files, with no single file holding more than 14.
[^11]: Read 2026-08-06 while shipping step 1, re-read 2026-08-07. `src/tankpit_bot/protocol/codec.py:83` `xor_bytes(table, data, offset=0)` raises `InvalidKeyError` on an empty table and `ValueError` when `offset + len(data) > len(table)` — the only one of the four that names the condition. `src/tankpit_bot/capture/xor.py:149` `xor_decode_body(body, xor_table, offset=0)` indexes `xor_table[i]` unguarded, so the same condition surfaces as a bare `IndexError`. `src/tankpit_bot/diagnostics/capture_audit.py:53` `_xor_with_table(body, table)` carries an explicit `else` branch copying the byte through unciphered, and is now the ONLY remaining pass-through site. **Two changes since the 2026-08-06 reading, both narrowing the fork:** `sim/transport.py` no longer defines its own `_xor_with_table` — it imports `xor_decode_body` (`:25`) and calls it at `:62` and `:165`, passing `offset=1` explicitly at the client-command site — and `capture/xor.py` no longer computes the static-key path or loads the key itself; it imports `build_xor_table`, `load_static_key` and `DEFAULT_STATIC_KEY_PATH` from `protocol.codec` (`:21-22`, used at `:80` and `:103`), so `DEFAULT_STATIC_KEY_PATH` at `protocol/codec.py:20` is the single definition. The 279,771-payload measurement covers only the received-decode path and was taken before step 1; it does not license deleting the pass-through arm from the audit reader.
