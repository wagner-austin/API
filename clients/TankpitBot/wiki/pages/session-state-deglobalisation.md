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
  "src/tankpit_bot/sniffer": "630b741c05e1c21db252da1f045ce9e887d8df39"
  "src/tankpit_bot/ledger": "9fb698ec36f37e2458b1c3528042c04773fa446f"
  "src/tankpit_bot/capture/xor.py": "1685851c8623f51b461b58fc418ebe8df6c72873"
  "src/tankpit_bot/protocol/codec.py": "76ba3790ab90e303383c2f7a66dd48c96e30467c"
  "tests/conftest.py": "1142bca727c8e8d2cf191d1941537e76e40c1e46"
fact_checked: "2026-08-06"
confidence: high
hubs: [architecture]
---

# Session-State De-globalisation

Seventeen modules hold **per-session** state at module scope. Each is a
place where two sessions in one process would overwrite each other, and
each is a value that logically belongs to one game session and nothing
else. This page is the inventory, the evidence, the archaeology of how
it happened, and the ordered plan to finish it.

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
state is instance-scoped, no reset is possible to forget.

**Progress: 10 calls → 9.** Step 1 removed `reset_xor_state` and put
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
| `sniffer/viewport.py` | `_viewport_left`, `_viewport_top` | rebind |
| `sniffer/trackers.py:42` | `ALL_TRACKERS` (tuple of tracker instances) | in-place |
| `ledger/events.py` | `_event_counter` | rebind |
| `ledger/outcome/teleport.py` | `_pending` | rebind |
| `ledger/outcome/_emit.py:27-29` | `_attempt_counters`, `_pending_decisions`, `_resolved_decision_ids` | in-place |
| `ledger/ring.py:51` | `_rings` | in-place |
| `ledger/decision.py:60` | `_decisions` | in-place |
| `ledger/mode_transition.py:38` | `_transitions` | in-place |
| `bot/ai/collect_common.py:15` | `_blacklisted_container_keys` | in-place |
| `diagnostics/self_alignment.py` | `_last_emitted_belief` | rebind |
| `diagnostics/entity_alignment.py` | `_last_emitted_signature` | rebind |
| `browser/cdp_utils.py` | `_cdp_time_offset_ms` | rebind |
| `action_lab/client_structure.py` | `_survey_emitted` | rebind |
| `runtime_logging.py` | `_BOT_ARTIFACTS`, `_SNIFF_ARTIFACTS`, `_PROBE_ARTIFACTS` | rebind |
| `runtime_logging.py` | `_RUNTIME_CONTEXT_TICK_N`, `_..._BOT_STATE`, `_..._IN_FLIGHT_ACTION_KIND` | rebind |

### Legitimately process-level — NOT in scope

| module | state | why it stays |
|---|---|---|
| `sniffer/decoders.py` | `_PROTOCOL_FRAME_LOGGING_ENABLED` | a logging switch for the process, not a session fact |
| `bot/tick_loop.py` | `_INTERRUPT_REQUESTED` | Ctrl+C is a process signal |
| `analysis/_test_hooks.py` | `read_text`, `list_session_paths` | the DI seam itself ([[testing-patterns]]) |
| `sniffer/xor.py` static key | the key read from `xor_static_key.txt` | identical for every session; caching it is a property of the key |

## Why it exists: an interrupted convergence

This is not two competing designs. It is **one design that never
finished converging**, frozen in place by a package split.

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

Three measurements, all archive-wide, all 2026-08-06:

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
| `get_world_service()` | 73 | 20 | 21 in `bot/tick_loop.py` alone |

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
row with the same correction: 73 sites in 20 files is the *src* floor
for step 8, not its size.[^12]

## Plan

Ordered by coupling, least first. Every step ends with `make check`
green and a commit, so a collision with concurrent work costs one step
rather than the refactor.

1. ~~**XOR cipher.**~~ **SHIPPED 2026-08-06** (`7481cdca`). Deleted
   `sniffer/xor.py` outright — no shim, no deprecation. Its callers now
   take the table as a parameter, built by the new
   `capture/xor.py::build_session_xor_table` and stored on
   `SessionBase`, which hands the *same* table to `CommandService`
   instead of building it twice from identical inputs.[^8]

   Three silent failures became loud, which is the real payoff:

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
2. **`sniffer/viewport.py`** — `_viewport_left` / `_viewport_top` onto
   the session's viewport state.
3. **Diagnostics dedupe memories** — `self_alignment`,
   `entity_alignment`.
4. **`browser/cdp_utils.py`** — `_cdp_time_offset_ms` onto the CDP
   service, which is already per-session.
5. **`action_lab/client_structure.py`** — `_survey_emitted`.
6. **Ledger cluster** — events counter, teleport `_pending`, `_rings`,
   `_emit` trackers, `_decisions`, `_transitions`. These form one
   bookkeeping layer and move together.
7. **`bot/ai/collect_common.py`** — container blacklist.
8. **`WorldService`** — delete `_service` and `get_world_service()`;
   thread the instance through the 73 sites. The largest step.
9. **`sniffer/trackers.py`** — `ALL_TRACKERS`.
10. **`runtime_logging.py`** — per-session artifacts and context.
11. **Delete the conftest reset list.** Its removal is the proof.

## Found while shipping step 1: the cipher is forked four ways

Deleting `sniffer/xor.py` removed one fork and revealed that it was
never the only one. Four implementations of the same XOR remain, and
they do **not** agree at the edges:[^11]

| site | signature | past the table end |
|---|---|---|
| `protocol/codec.py::xor_bytes` | `(table, data, offset=0)` | raises `ValueError`, named and explicit |
| `capture/xor.py::xor_decode_body` | `(body, table, offset=0)` | raises `IndexError`, incidentally |
| `diagnostics/capture_audit.py::_xor_with_table` | `(body, table)`, offset fixed at 1 | passes through in the clear |
| `sim/transport.py::_xor_with_table` | `(table, data)`, offset fixed at 0 | passes through in the clear |

Note the argument order flips between them — `(body, table)` in two,
`(table, data)` in the other two — so a wrong-order call type-checks
cleanly and silently produces garbage.[^11]

`capture/xor.py` and `protocol/codec.py` additionally each compute the
static-key path with the byte-identical expression
`Path(__file__).parent.parent.parent.parent / "xor_static_key.txt"`,
resolving to the same `<repo>/xor_static_key.txt`. `codec` at least
exports it as `DEFAULT_STATIC_KEY_PATH`; `capture` inlines it inside
`load_xor_static_key`. That is a duplicated constant rather than a
disagreement — but it is the same fork, and the tests already reach
across it: `tests/capture/test_xor.py` imports the path from
`protocol.codec` to seed a file that `capture.xor` reads.[^11]

`protocol/codec.py` is the survivor: it is the lower layer and the only
one with validation. Folding the other three into it is **its own
step**, not absorbed silently into another, because the pass-through
tail is a real semantic difference that needs measuring first. The tail
was already measured dead for the *received-decode* path — 0 of 279,771
archived payloads exceed the 1000-byte table — but that measurement
says nothing about the sim's envelope encoder or the audit reader, which
are the two sites that rely on it. Measure those before deleting the
arm.[^11]

## Carried in the same sweep

Two defects found while mapping this, both in scope:

- **An eleventh fork of the frame walk**, this time in production code:
  `sniffer/decoders.py::process_received_message` re-derives
  `data[offset] | (data[offset + 1] << 8)` instead of calling
  `split_frames`. The other ten forks were in `analysis_scripts/`.[^9]
- **No file-size enforcement.** [[coding-standards]] sets a 400-600
  line ceiling and the 2026-07-31 log entry recorded a backlog of 40
  files over 600. Measured 2026-08-06: **77**, and all four of the
  worst named in that backlog have GROWN — `test_fuel_probe.py`
  2698→2948, `test_cdp.py` 2400→2816, `world_state_dispatch.py`
  1156→1280, `tick_loop.py` 1037→1211. Nothing in `make check` checks
  file size, so the rule is documented and unenforced. A guard rule
  belongs with this work because `world_state_dispatch.py` and
  `tick_loop.py` are simultaneously the two worst offenders and the two
  heaviest consumers of the globals being removed — splitting them
  before de-globalising would only spread the coupling.[^10]

[^1]: `src/tankpit_bot/service/fleet.py:1-28` — the fleet manager's module docstring states the process-per-bot rationale verbatim: it "runs in a terminal the operator owns, so an orchestration harness dying can never kill a live tank (the 41-kill session died at 46 minutes exactly that way)". Registered as `tankpit-fleet` in `pyproject.toml`.
[^2]: `tests/conftest.py` — the `_isolate_protocol_singletons` autouse fixture calls, in order: `reset_world_state`, `reset_xor_state`, `reset_event_ids`, `reset_action_outcome_tracking`, `reset_outcome_rings`, `reset_teleport_dispatch_tracking`, `reset_decision_records`, `reset_mode_transitions`, `reset_container_blacklist`, `reset_client_structure_survey`. Its own docstring gives the reason — "tests that run later on the same xdist worker can decode bytes with a stale XOR key or read containers seeded by a prior run". Sibling fixtures `_restore_hooks` and `_restore_runtime_logging_state` reset the hook surface and the runtime-logging artifacts.
[^3]: Inventory taken 2026-08-06 by two sweeps. Sweep one, every `global` declaration under `src/tankpit_bot`: 13 modules, listed in the table. Sweep two, module-level mutable containers mutated in place — which need no `global` and so are invisible to sweep one: `src/tankpit_bot/ledger/ring.py:51` (`_rings`), `src/tankpit_bot/ledger/decision.py:60` (`_decisions`), `src/tankpit_bot/ledger/mode_transition.py:38` (`_transitions`), `src/tankpit_bot/ledger/outcome/_emit.py:27-29` (three), `src/tankpit_bot/bot/ai/collect_common.py:15` (`_blacklisted_container_keys`), and `src/tankpit_bot/sniffer/trackers.py:42` (`ALL_TRACKERS`). A `global`-only sweep undercounts by those six modules, which is why both were run.
[^4]: `git show a1c32e9a~1:...sniffer.py` — the pre-split monolith, 3,204 lines. `_global_xor_table` is declared at line 1827 under the comment "# Module-level XOR table for unified decoder", with `_build_global_xor_table` at 1843 and the private `_xor_decode` at 1855; the same file carries 73 `xor_table` references for the table-passing path. Split commits: `a1c32e9a` "Split sniffer.py into sniffer/ package" (13:04:18) and `389231df` "Add capture/ package" (13:04:58), both 2026-01-12.
[^5]: `src/tankpit_bot/sniffer/world_service.py:80-86` — `class WorldService`, docstring: "Owns all mutable game state for one session. Instance attributes mirror the 16 module-level globals that were previously in ``world_state.py``. Dispatch modules receive a ``WorldService`` instance and mutate it directly." The remaining singleton is `sniffer/world_state.py:19`, `_service = WorldService()`. Live chain: `bot/world_sync.py::drain_messages(bot)` -> `sniffer/decoders.py::process_received_message` -> `_process_single_message`, which is the only step that reaches for the globals.
[^6]: Measured 2026-08-06 over `runs/bot` (287 captures) with the production primitives: static key and table both 1000 bytes (`capture/xor.py::build_xor_table`), longest frame payload 931 bytes, 0 of 279,771 payloads longer than the table. Framing: 217,678 received payloads split cleanly under `protocol/framing.py::split_frames` with zero `FramingError`; 62,095 sent payloads raised twice. Re-derivable by re-running the sweep over the archive; the counts are not stored.
[^7]: Same sweep: the only capture with unframed sent payloads is `bot-20260331-230406`, which also carries `magic: null`. `analysis/scan.py::scan_session` checks magic first and returns `no_magic`, so the `unframed_payload` arm cannot be reached from the archive.
[^8]: Written before the step, describing the state it removed: `src/tankpit_bot/browser/session_base.py::_on_magic_captured` built the table twice from the same inputs — `init_trackers_with_magic(magic)` (which called `build_global_xor_table`) for the decode side, and `self._commands.xor_table = build_xor_table(static_key, magic)` for the send side. `CommandService.xor_table` at `bot/command_service.py:62` was already instance state, consumed at `:76`. As of `7481cdca` the method builds one table and assigns it to both.
[^12]: Commit `7481cdca` "Make the XOR table a session value, not a module global", 2026-08-06. `git show --stat` reports **79 files changed, 611 insertions(+), 588 deletions(-)** — 21 paths under `src/`, 57 under `tests/`, plus `scripts/decode.py`. Verification at that commit: `make lint` reports 0 violations across all 28 guard rules; `python -m pytest -n auto` reports 5939 passed. Three failures remain in `tests/test_check_undecoded_fields.py`, caused by a concurrent session's in-flight split of `protocol/types.py` into a package — `scripts/check_undecoded_fields.py:34` `DEFAULT_TARGETS` still names the deleted file. That script is untouched by either session and its fix belongs to the split, not to this work.
[^14]: `tests/sniffer/test_decoders.py::TestProcessReceivedMessage::test_binary_promotion_takes_binary_route` carried the comment `# XOR table is None in tests so body[1:] passes through verbatim.` (removed in `7481cdca`); it asserted `self_state["rank"] == 5` from a body written as plaintext. It now builds that body through `make_binary_payload`, which ciphers it under the same table the decoder is handed, so the assertion exercises the real cipher round-trip instead of an identity pass. `tests/replay/test_script.py::_install_fake_fs` installed a `_FakeFS` with no `xor_static_key.txt` at all, which the old builder tolerated by leaving the table `None`; it now seeds the key and resets the cache, and its three `TestMainCLI` tests fail loudly without it.
[^13]: `tests/conftest.py::_isolate_protocol_singletons` after `7481cdca`: the reset list runs `reset_world_state`, `reset_static_key_cache`, `reset_event_ids`, `reset_action_outcome_tracking`, `reset_outcome_rings`, `reset_teleport_dispatch_tracking`, `reset_decision_records`, `reset_mode_transitions`, `reset_container_blacklist`, `reset_client_structure_survey` — ten calls, of which `reset_static_key_cache` guards a process-wide key cache rather than session state. The five deleted per-file duplicates were `_isolate` / `_isolate_world_state` in `tests/action_lab/test_equipment_probe_branches.py`, `test_replay_equipment_probe.py`, `test_replay_fuel_probe.py`, `test_replay_movement_probe.py`, and `test_replay_teleport_probe.py`; the quoted docstring is the one that stood in `test_replay_equipment_probe.py`, `test_replay_fuel_probe.py`, and `test_replay_movement_probe.py` verbatim.
[^9]: `src/tankpit_bot/sniffer/decoders.py::process_received_message` — the frame loop re-derives the 2-byte little-endian length prefix inline rather than calling `protocol/framing.py::split_frames`, which has owned that arithmetic since the protocol layer was written (`decode_frame_header`, `:37`).
[^10]: Counted 2026-08-06 with `find src scripts tests -name '*.py' | xargs wc -l | awk '$1>600'`: 77 files. The 2026-07-31 backlog of 40 is enumerated at `wiki/log.md:2437`. `bot/tick_loop.py` holds 21 of the 73 `get_world_service()` call sites.
[^11]: Read 2026-08-06 while shipping step 1. `src/tankpit_bot/protocol/codec.py:83` `xor_bytes(table, data, offset=0)` raises `InvalidKeyError` on an empty table (line 100) and `ValueError` when `offset + len(data) > len(table)` (lines 106-110) — the only one of the four that names the condition. `src/tankpit_bot/capture/xor.py:136` `xor_decode_body(body, xor_table, offset=0)` indexes `xor_table[i]` unguarded, so the same condition surfaces as a bare `IndexError`. `src/tankpit_bot/diagnostics/capture_audit.py:51` `_xor_with_table(body, table)` and `src/tankpit_bot/sim/transport.py:26` `_xor_with_table(table, data)` each carry an explicit `else` branch copying the byte through unciphered. The key path is the same expression in both `src/tankpit_bot/protocol/codec.py:20` and `src/tankpit_bot/capture/xor.py:28`. The 279,771-payload measurement covers only the received-decode path and was taken before step 1; it does not license deleting the pass-through arm from the sim encoder or the audit reader.
