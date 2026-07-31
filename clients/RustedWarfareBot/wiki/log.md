# Wiki Operation Log

Append-only. Log structural operations (new hubs, decomposition, audits, cleanups) and live runs that produced findings. Routine page edits don't need a log entry — git history covers those.

## [2026-07-25] init | wiki scaffolded
Hubs created: engine-internals, headless-harness, game-mechanics, bot-architecture, multiplayer
Notes: initial scaffold via /wiki-init, SCHEMA v1.0 plus two domain additions (`game_version` pinning, observed-behaviour-over-inference citation hierarchy).

## [2026-07-25] probe | M0 headless feasibility
Pages written: harness-nodisplay, engine-name-oracle, multiplayer-portability-invariants
Artifacts: `wiki/sources/m0-probe/` — `nodisplay-boot.log` (336), `printunits.log` (1533), `jar-classes.txt` (1698), `main-strings.txt` (345)
Notes: `-nodisplay` boots the full engine in 1.32 s and ticks a live match with no window; asset index self-identifies as `dedicatedServer`. Boot log names engine objects and prints one obfuscated mapping directly. `-printunits` emits a complete unit stat catalogue. Game copied from the Steam install to `.game/` (2095 files, 451 MB) so runs never mutate the Steam tree and the build stays pinned against auto-update.

## [2026-07-25] probe | M1' — `-sandbox` reaches a live skirmish, blocked on one GUI method
Artifacts: `wiki/sources/m1-sandbox/sandbox-crash.log`, `sandbox-10x10-crash.log`
Notes: `-sandbox` boots straight into `maps/skirmish/[z;p10]Crossing Large (10p).tmx` — the same default `singleplayer.rml` wires to that button — with PathEngine costs built and units present. It requires an explicit `-width/-height`; `-nodisplay`'s default 10x10 fails earlier. The only blocker is `com.corrodinggames.rts.java.d.a.EnableScissorRegion`, which dereferences a null `slick.Graphics` on the first in-game GUI frame; the method only calls `setWorldClip`/`clearWorldClip`, so it is safe to no-op. `-safemode`, `-canvasgl`, and `-nopostprocessing -disable_atlas` all still hit it. The RML menu tree also yielded the full control API in unobfuscated names: `ScriptEngine.getInstance().getRoot()` reaches `startNew(String)`, `loadConfigAndStartNew(String)`, `setValueById`, and `hostStart(boolean)`. Also: the first crash uploaded a stack trace to corrodinggames.com; `sendReports` is now `false` in the pinned copy.

## [2026-07-25] build | Python harness scaffolded to the monorepo standard
Notes: `pyproject.toml`, `Makefile`, `scripts/guard.py` lifted from the sibling client rather than authored fresh; guard runs over src/tests/scripts inside `make lint`. First slice is `rw_bot.validation` (require_* validators), `rw_bot.harness.launch` (LaunchConfig with encode/decode and the verified flag set), `rw_bot.harness.boot_log` (structured decoding of the engine log), and the `rw-boot-log` CLI. `make check` green: guard 0 violations across 31 rule categories, ruff clean, mypy strict clean, 61 tests, 100% statements and branches.

Five guard rules shaped the design rather than being worked around: a local `errors.py` is banned (exceptions now live beside the code that raises them, with the shared base in `rw_bot/__init__.py`); `object` is banned in annotations (payloads are typed `Mapping[str, str | int | bool]`); `Iterator`/`Iterable` from `collections.abc` are banned; `except` must log or re-raise (the CLI no longer maps a decode failure to an exit code — it propagates); and weak assertions are banned. `pytest.fixture` additionally fails `disallow_any_expr`, so hook binding uses explicitly typed context managers.

Two real bugs were caught by tests against the archived logs: the timestamp prefix was stripped one character short, and the monorepo-root search test passed spuriously because pytest's basetemp lives inside the repo.

## [2026-07-25] decision | language + runtime split
Pages written: runtime-split-java-agent-python-brain
Pages updated: multiplayer-portability-invariants (rewritten for paragraph-citation), engine-name-oracle, harness-nodisplay
Notes: agent = plain Java (forced — javaagent in the game's OpenJDK 13, minimal deps to avoid classloader conflict); brain = Python (matches both sibling clients, reuses `libs/` + the enforced clients standard, numpy for grid-shaped planner state). TypeScript was the close runner-up — better types and one shared wire contract, but cannot consume the Python `libs/` and would fork the api/clients Poetry convention. Kotlin-everything rejected: zero IPC is real, but it discards the whole toolchain for a bottleneck that is discovery, not compute. Decisions run decimated at 4–10 Hz with the agent free-running between them, so IPC is not the ceiling; per-tick reaction goes through the engine's own standing orders rather than agent-side logic, which keeps pure dispatch intact.

Also swept all four pages to zero uncited paragraphs. The `paragraph-citation` family requires every prose paragraph to carry a `[^N]`, `[@id]`, or `[[slug]]` — currently advisory for concept pages but explicitly slated to go fatal, and the rule's own docstring names a 2026-07-23 tankpitbot incident (confident uncited inference) as the reason. Verified locally: 0 uncited, 0 dangling footnote refs, 0 unused definitions, all 9 source_paths resolve with line anchors in bounds, all 4 wikilinks resolve.

## [2026-07-25] build | M2 — the javaagent exists and the engine reaches a live headless skirmish
Pages written: agent-render-callback-noop
Pages updated: index (5 pages), hubs/bot-architecture, hubs/headless-harness
Artifacts: `wiki/sources/m2-agent/` — `sandbox-agent-running.log` (403), `sandbox-agent-stdout.txt` (49), `sandbox-agent-jstack.txt` (31)
Notes: the M1' blocker is cleared. `EnableScissorRegion(Z)V` is replaced with a bare `return` by a javaagent, and `-sandbox` now runs indefinitely: map loaded, PathEngine ready, `--- setRunning ---`, minimap built, `--- Mouse API succeeded`, no exception. Two thread dumps twelve seconds apart put the game thread at 4,984 ms then 7,968 ms of CPU, both inside `Display.sync` — the loop is live and frame-rate limited, not stalled. Uncapping that sync is the lever if the bot ever needs faster-than-realtime play.

Disassembly changed the design mid-build. The obvious patch — preserve `h = enabled` and drop only the `Graphics` calls — is wrong: flag `h` has exactly three references, two writes here and one read guarding a branch that dereferences field `g`. A bare `return` leaves `h` permanently false and disarms that branch; preserving it would have armed a second null dereference. The narrow fix was the safe one only because the field census was done first.

Patched without ASM or Javassist. The agent loads beside obfuscated classes where every dependency is a conflict surface, and one property makes hand-rolling tractable: a no-op body references no constant-pool entries, so the pool is parsed only to find its end and copied byte-for-byte, leaving the edit local to a single `Code` attribute. Correctness is certified by defining and linking each patched class — HotSpot's verifier is the oracle, not a second reading by the code that wrote the bytes. Independently confirmed with `javap`: the method is `0: return` and every other method is byte-identical.

Not settled: `RenderGeometryPossiblyCompiled` dereferences the same null `j` at ~10 further offsets. The render path is proven to survive the frames observed, not proven clean. `Targets` is a list for that reason — the next callback is a one-line addition.

## [2026-07-25] probe | M3 — the simulation tick method and clock, found by running the game
Pages written: engine-tick-and-clock
Pages updated: index (6 pages), hubs/engine-internals, hubs/bot-architecture
Artifacts: `wiki/sources/m3-discovery/` — `engine-snapshots.log` (1239), `gameengine-tick-method.txt` (105)
Notes: `com.corrodinggames.rts.game.i.a(float)` is the simulation tick. Its body contains a literal `getfield bx / iconst_1 / iadd / putfield bx`, so `bx` counts invocations exactly; `by` is stored from an `f2i` just above and is a derived millisecond clock. Measured over two ten-second intervals on the live engine: `by` +9,994 then +9,999 (1 kHz), `bx` +2,993 then +2,998 (299.8 Hz).

The method was found by running the game, not by reading it. `gameFramework.l` carries ~360 members of which exactly one kept a readable name, so correlating by type is expensive; the live object graph is cheaper and unambiguous. The agent gained a `discoverAtSeconds=` option that snapshots the engine reflectively at given elapsed times — read-only, structural summaries only, `toString` called on nothing but strings and primitives so a probe cannot side-effect a live simulation. `gameFramework.l.B()` is the handle, and its whole body being `return al;` is what makes it safe.

Multiple snapshots were the point. A single dump cannot separate match state from static configuration; the diff can. It also showed the sandbox map is fully loaded before t=5s, so only scalars moved between snapshots — which is what isolated the counters in the first place.

Two false leads worth recording. Rate alone cannot distinguish a sim tick from a rendered frame, and the render loop is frame-rate limited in the same ~300 Hz range — only the bytecode settled it. And five other methods write `bx`: three restore a saved pair around the state deserialiser `y.a(k,…)`, one writes literal zero as the new-game reset, one is the constructor. Taking any of them for the tick would put the clock read on a path that fires only on load. A first scan missed the real writer entirely because it matched the field owner as `l.bx`, and the increment is compiled against the subclass as a bare `bx` — owner-qualified greps are not safe on an obfuscated hierarchy.

Not settled: two parallel eleven-element collections on the engine, `X` of `game.units.al` and `W` of `game.units.e.b`. `X` is the likely unit list, but eleven is unreconciled and neither accounts for the 206 trees the same load logged. Lead, not finding. `CommandController`'s order entry point remains open — the last of the three prerequisites `engine-name-oracle` named.

## [2026-07-25] audit | cross-check: the engine calls `bx` a frame, not a tick
Pages updated: engine-tick-and-clock
Notes: two concurrent probes reached the same method from different directions, and comparing their artifacts corrected a naming error neither would have caught alone. The command-path probe archived the decompiled `game.i.a(float)`, which contains the engine's own debug line: `"updateAllGame1: deltaSpeed:" + f2 + " frame:" + this.bx + " network.currentStepRate:" + this.bX.c()`. So `bx` is a **frame** counter in the engine's vocabulary, and the lockstep step rate is a separate quantity on the network engine.

The measurement stands — `bx` still advances 299.8/s and is still incremented exactly once per update call — but the framing mattered: a decision cadence decimated against `bx` decimates against local frames, while the quantity agreed between peers is the network step rate. That is the one a multiplayer-legal planner has to key off. `bx` remains correct for "has the simulation advanced".

Same artifact sharpened `by`: it accumulates `f2 * 16.666666f` (ms per frame at a 60 Hz baseline, scaled by delta), so its measured 1 kHz is a consequence of that formula rather than an independent clock. The page said "derived millisecond value", which was right but vaguer than the source allows.

## [2026-07-25] build | M7 — the unit catalogue, and the mobility predicate falls out of it
Pages written: mechanics-unit-catalogue
Pages updated: index (11 pages), hubs/game-mechanics, hubs/bot-architecture
Notes: `rw_bot.mechanics.catalogue` decodes the engine's own `-printunits` output into typed records — 90 units with price, HP, speed, mass, upgrade tiers and weapons. No new capture was needed: the artifact has been sitting in `wiki/sources/m0-probe/printunits.log` since M0, unmined. `make check` green, 100% statements and branches on the new module.

The catalogue answers a question that was previously answered the expensive way. 38 of the 90 units have speed exactly zero, and `commandCenter` is one of them — which is the catalogue-level explanation for the first real order this project issued moving nothing. That was diagnosed empirically by sampling position before and after (see the issuing-orders entry). `speed > 0` is a read mobility predicate, so selection no longer has to discover immobility by ordering a building to walk.

The type name is the join key and it already appears on three surfaces: `unit:builder` in the catalogue, the `type` field on the world stream, and the argument the type registry accepts when placing a building. No mapping table is needed between them.

Two shapes had to be modelled rather than flattened. A unit is armed only if the engine prints an attack range (61 of 90); damage of a kind not printed is zero, which is a fact about the unit rather than a missing reading. And per-shot and per-volley damage are independent figures — the engine writes `Direct Damage: 12 (total:24.0)` for multi-barrel weapons, and the ratio is not fixed: 2x, 4x, 6x, and one unit at 1.84x. Assuming a barrel count would be wrong for most of them.

Worth recording how that second shape was found: not by surveying the stat keys, which suggested a plain number and looked complete. The decoder was written against the key survey, run against the real log, and failed on `heavyInterceptor` with "non-numeric Direct Damage: '12 (total:24.0)'". The key names were surveyed; the value shapes were not. Running it against the real artifact is what closed the gap.

## [2026-07-25] build | M6 — the wire contract exists on both sides
Pages written: wire-contract-ndjson
Pages updated: index (10 pages), hubs/bot-architecture
Artifacts: `wiki/sources/m6-wire/world-sample.ndjson` (12) — a real capture from a live headless skirmish
Notes: the agent serialises the owned roster as NDJSON and the planner decodes it into typed samples. `make check` green: 148 tests (was 88), 100% statements and branches, `rw_bot.wire.ndjson` 105/40 and `rw_bot.wire.state` 80/28 fully covered.

The format's defining property came from the consumer, not the producer. `json.loads` is unusable under `disallow_any_expr` — its return is `Any`, every expression touching it errors, `isinstance` does not rescue it because the call itself is the offending expression, and suppressions are banned. Verified by probe before designing around it rather than assumed. So the reader is hand-written, and a hand-written reader is only cheap if the grammar is small: every record is therefore a flat object of scalars, no nesting, no arrays, no null. Constraining the Java producer is what buys a fully typed Python consumer; the two are one decision.

The reader coerces nothing and repairs nothing — six traceable codes, one per rejection, including duplicate keys and trailing content after the object. The check that earns its place is the declared-count check: a sample promising three entities and carrying two is a truncated capture, the ordinary result of reading while the agent is still writing, and yielding it silently would let a planner act on a roster it cannot fully see.

Two real bugs found while building, both by running rather than reading. `Orders.onGameThread` enqueues without running, so reading the rendered sample straight after the call produced null every time — the render is now awaited on a latch with a bounded wait that fails loudly. And the first capture attempt wrote nothing at all, which the "state sample was not produced" line reported honestly instead of writing an empty file.

`validation.py` gained `require_finite_float` (`RW-DECODE-006`) and its payload union widened to carry floats; the union is covariant so no existing caller changed. Non-finite values are rejected on both sides — JSON cannot carry them, so a NaN means the producer emitted something the format does not have, which is a bug rather than a datum.

## [2026-07-25] probe | M4 — decompiled the jar; the entity model falls out in three greps
Pages written: engine-entity-model
Pages updated: engine-tick-and-clock (its unit-list guess was wrong and is now corrected), index (7 pages), hubs/engine-internals, hubs/game-mechanics
Artifacts: `wiki/sources/m4-entities/` — `entity-count-loop.txt` (37), `player-class.txt` (23), `live-graph-search.txt` (13)
Notes: the model is `com.corrodinggames.rts.game.units.am` for every world object, static `am.bE` as the master list, `al` as the **tree** class, `am.bX` as the owning player and `am.eo`/`am.ep` as position. `game.n` is the player class, carrying `public double o = 4000.0` (starting credits) next to a string literal addressed to modders that confirms the lockstep model in passing.

The decisive move was searching the decompiled source for a log string the engine already prints — "there are N units on this map and N trees" — which lands directly in the census loop that uses the list, the base class, the tree subclass, the owner field and the position pair all at once.

Two prior readings were wrong and are recorded rather than quietly replaced. A sprite registry was taken for the unit list because its size (11) sat near the map's unit count (10); size coincidence is not identification. A graph node class was then taken for the unit class because its query methods had the right shape. Both were guesses dressed as findings, and the wiki page that carried the first one has been corrected in place.

Method note: `javap` is right for narrow questions and proved the tick increment, but it does not make relationships legible, and relationships were the whole question. CFR 0.152 decompiles all 1698 classes cleanly. The decompiled tree is a derived work of a commercial game and is deliberately NOT versioned — it regenerates into gitignored `runs/decompiled/`, and only cited excerpts live under `wiki/sources/`.

Two fixes in `describeElements` specifically, distinct from the `findCollections` traversal defect logged separately below. It listed only an element's own declared fields and not inherited ones — which is exactly what made `al` instances look featureless and helped sell the wrong reading, since engine entity state is declared on the base `am`. It also reflected into JDK internals when an element was a platform type; platform classes are now summarised instead. A selftest case asserts an inherited field appears, so the failure mode that cost this milestone two wrong answers cannot return silently.

## [2026-07-25] build | agent wired into the Python contract and the check gate
Notes: the Java half is no longer a side artifact. `LaunchConfig` gained a required `agent_jar`, rendered as `-javaagent` ahead of `-cp` (the JVM stops parsing its own options at the main class, so position is load-bearing and is asserted by test). Required rather than optional: a launch without the agent is not a supported mode, and modelling it as optional adds a branch whose only reachable outcome is a crash.

New `rw_bot.harness.agent` closes the cross-language gap the type system cannot see: it parses the manifest the JVM will read and fails with traceable codes for the three reachable drifts — no `Premain-Class` (`RW-AGENT-001`), jar not built (`RW-AGENT-002`), and the attribute naming a class whose source no longer exists (`RW-AGENT-003`), which builds cleanly and then aborts the JVM at launch. Two tests read this repo's own tracked manifest, Java source and Makefile, so a rename or a moved build path fails in `make check` rather than in an engine crash log.

New validator `require_absolute_path` (`RW-DECODE-005`). It also closed an existing gap: `log_path` was documented as absolute and never enforced, so relative paths silently resolved against the game tree. Now enforced for both paths, and the existing tests were updated to match rather than the rule being relaxed.

`make check` now chains `agent-selftest`, so a patcher regression or an obfuscated name that moved in a game update fails at the gate. `make check` green: guard 0 violations, ruff clean, mypy strict clean, 88 tests (was 61), 100% statements and branches, agent-selftest OK.

## [2026-07-25] audit | staleness sweep across all 14 pages
Pages updated: multiplayer-portability-invariants (medium -> high), harness-nodisplay, issuing-orders, engine-entity-model, mechanics-unit-catalogue, perception-visibility

Notes: the pages were written as each milestone landed, and several had been overtaken by later ones. Four real corrections, none cosmetic.

`multiplayer-portability-invariants` was written at `medium` confidence and named its own falsification test: read `CommandController`'s dispatch path. M5 did exactly that, and the dispatch path confirms both halves of the lockstep model — every command is stamped with the `by` millisecond clock at construction, and the enqueue forks on network role with a server-side check on one arm only. A single-player engine with no notion of peers would need neither. Raised to `high`; the four invariants are unchanged, because they were written to be correct either way. Invariant 3 also stopped being theoretical: the stream now carries opponents, so ownership is a per-entity fact the planner must consult.

`issuing-orders` opened with "The bot can now play." It could not, and I said so at the time without fixing the page. One hardcoded order fired by a wall-clock timer at a roster index typed on the command line is a working dispatch path and nothing else. The claim is now stated as what it was, pointing at `policy-loop` for what playing actually took.

`harness-nodisplay` still listed "starting a skirmish headless has not been done" as an open question, and described `-sandbox` and the software-rendering flags as never run — all false for a while. The resolved question is now written up as resolved and the exercised-flag count corrected from five to nine. Two footnotes orphaned by the rewrite were removed rather than left dangling.

Three pages carried uncited prose paragraphs, which the `paragraph-citation` family flags as advisory today and is slated to make fatal.

Verified after: 14 pages, 0 uncited paragraphs, 0 dangling footnote references, 0 unused definitions, every `source_paths` entry resolving with line anchors in bounds, every wikilink resolving.

## [2026-07-25] milestone | M9 — the policy loop; the bot plays a build order unattended
Pages written: policy-loop
Pages updated: index (14 pages), hubs/bot-architecture, hubs/game-mechanics
Artifacts: `wiki/sources/m9-policy/` — `plan-completed.txt` (10), `plan-stalled-on-laboratory.txt` (10), `engine-refused-the-laboratory.txt` (5), `visible-includes-opponents.ndjson` (4)

Notes: three structures, three orders, no waste — `done (all 3 structures built)` in 13 samples and 3,603 frames, credits managed the whole way. The deciding half is a pure function of a sample, a plan and the catalogue, so the playing logic is tested exhaustively without a game; the loop around it only reads, asks, acts and repeats.

Credits had to reach the wire first: `n.o`, sitting beside the engine's own note to modders about not cheating in multiplayer. Floored to whole currency, because the engine spends in whole units and 99 credits does not buy a 100-credit structure.

Two ownership traps were closed before they could bite. The stream now carries every visible entity, not just the player's — 19 across five teams in one capture, of which three were ours — so without an ownership check an opponent's factory would have advanced our plan and an enemy builder could have been selected to receive an order. The frame count was also renamed `owned` to `visible`, because it had stopped being the former the moment enemies were included and a consumer reading it as a roster size would have been silently wrong.

The run that taught the most failed. A plan ending in a laboratory ran 300 samples and 89,290 frames, banked 11,258 credits, and reported "building laboratory (3 of 3)" throughout. Nothing was wrong with credits, placement or the channel: the engine had refused the order outright and said so only in its own log — `Unit 'builder' can not queue build:laboratory`. A builder cannot construct one, which is not derivable from the catalogue (prices and stats, no build lists) and produces no roster change the planner can see. Ordering each plan slot at most once protects against double-spend and, on its own, turns a refusal into a bot that looks busy forever. The loop now reports `stalled` after a bounded number of samples with no progress, naming what it waited on.

Scoring exists so runs can be compared: outcome, completed against planned, orders sent, samples, frames elapsed, credits left. `orders sent` against `completed` is the sharp one — equal means nothing was wasted, higher is the shape a refusal makes.

`make check` green: guard 0 violations, ruff + mypy clean, 266 tests, 100% statements and branches, agent-selftest OK.

Not a player yet in any full sense. It executes a fixed sequence against five opponents who scout, expand and fight, and it has no notion of winning. What it has is the shape: observe, decide, act, verify, score, and fail loudly when the world disagrees.

## [2026-07-25] milestone | M7 — the command channel; orders now originate in Python
Pages written: command-channel
Pages updated: index (12 pages), hubs/bot-architecture, hubs/engine-internals
Artifacts: `wiki/sources/m7-channel/` — `planner-drove-a-live-game.txt` (13), `world-sample-with-ids.ndjson` (12), `scriptengine-drain.txt` (26)

Notes: a planner connected to a live sandbox, read one sample, selected the builder by type name, computed a destination from that builder's own position, sent a build order, and watched the roster gain a `landFactory` three samples later. Nothing in the sequence was a constant — the subject came from the sample and the destination came from the subject.

One design decision was made before any wire code and is the reason the loop is usable: units are addressed by engine identity, not roster position. The Python contract had documented index as "the handle an order is dispatched against", which would have hardened a bug — index renumbers the moment anything is built or dies, which in this game is constant. The engine's `eh` is assigned once behind an "ID for GameObject is already set" guard and is what the engine uses for network identity. Entity records now carry both, plus the readable type name, which is what makes selection possible at all.

Backpressure is the constraint the channel is shaped around. Samples are produced on the game thread and written by another through a bounded queue that drops its oldest entry when full; blocking a socket write on the game thread would let a slow planner stall the simulation. Asserted in the selftest rather than left as a comment, because a match that pauses whenever the planner is busy would be blamed on the game long before the queue.

Both parsers reject independently. The Java side accepts a flat object of scalars and nothing else — nesting, arrays, duplicate keys, trailing text, non-finite coordinates and a move carrying a build type are all errors, fourteen rejection cases asserted. The Python encoder refuses the same shapes before sending. Neither side trusts the other.

Also fixed in passing, all caught by the gate rather than by review: a local `_parse_float` in `mechanics/catalogue.py` (the guard bans that name as a reimplemented config helper — renamed to `_parse_stat_number`, which is what it actually is, with the reason recorded at the definition), four `assert ... is not None` narrowing guards replaced by whole-record comparisons that also catch drift in fields the old assertions ignored, two `string in output` checks replaced by exact line comparisons, and four tuple-slice expressions that trip `disallow_any_expr` replaced by named parts.

`make check` green: guard 0 violations, ruff + mypy clean, 214 tests, 100% statements and branches, agent-selftest OK.

Still not a player. The probe orders once and watches; there is no goal, no scoring, and no loop that reconsiders. What exists is the substrate: perceive, decide, act, observe, all in Python against a live match.

## [2026-07-25] milestone | M8 — the build verb; a builder placed a factory
Pages written: building-structures
Pages updated: index (9 pages), hubs/engine-internals, hubs/game-mechanics
Artifacts: `wiki/sources/m8-build/` — `build-succeeded.txt` (28), `build-rejected-selector-zero.txt` (6), `waypoint-validator.txt` (38), `build-action-lookup.txt` (18), `placement-setter.txt` (8), `buildable-type-names.txt` (90)

Notes: construction is the economic verb — a bot that can only move cannot open a game — and it turned out to reuse the move machinery entirely, differing in one setter and one integer. It is not a special action: the `gui.actions.*` translation table enumerates 55 keys of unit abilities and carries nothing for placing a structure. Placement rides the same waypoint slot in a third target kind, the same one the engine's system-spawn path requires.

The integer cost a run by being read as a rotation. It is a build-action selector: a builder holds a list of build actions and the engine matches by type **and** selector, short-circuiting only on `-1`. Passing `0` asks for the action whose own index is 0, matches nothing, and the order is dropped by waypoint validation — builder never moved, no structure, no error. `-1` means "any action that builds this type" and is the only value that does not require knowing a builder's internal action ordering.

Diagnosis came from the engine naming its own refusal. `isValidNewWaypoint==false on: builder(pos:4250,2610 id:214 t:0)` said a waypoint was refused but not why; the validator's four distinct rejection strings — missing build type, cannot queue, locked, unavailable — are the actual diagnostic, and finding them was a decompiler read, not a guess.

With `-1` the same code worked: builder pathfinds from (4250, 2610) toward the site, and by t+5s the roster gains `units.d.m` at exactly (4450.0, 2730.0), the requested coordinates. Its drawables are `land_factory_*`, which identifies it independently of the type name asked for. The builder then stops in range and constructs.

Also recorded: 90 buildable type names extracted from the `-printunits` catalogue, and the reason they resolve at all. The registry lookup tries mods, then a built-in enum, then aliases — and the enum arm is dead, because its constants are obfuscated to single letters while it compares against `Enum.name()`. Every name resolves through the `.ini` registry, which is where the built-in units live too.

Still not playing. Two verbs exist and both are driven by command-line constants; nothing chooses, and Python has still never seen a game state.

## [2026-07-25] milestone | M5 — the bot issued a real order and a unit moved
Pages written: issuing-orders
Pages updated: index (8 pages), hubs/engine-internals, hubs/bot-architecture
Artifacts: `wiki/sources/m5-order/` — `order-accepted-unit-moved.txt` (11), `order-accepted-building-did-not-move.txt` (6), `controller-create-and-enqueue.txt` (22), `command-setters.txt` (22), `scriptengine-update.txt` (26), `builtin-ai-order-idiom.txt` (6)

Notes: the last of the three prerequisites is closed. The path is three calls — `cf.a(team)` to construct and enqueue, `a(unit)` to add a subject, `a(x,y)` to set a destination — and nothing dispatches afterwards, because the tick drains the queue. It is the built-in AI's own idiom rather than a reconstruction of one, which is what makes the bot's input the same class of thing a player's input is.

Threading was decided by the engine, not by us. Commands land in a plain `ArrayList` the tick drains, so a probe-thread write would race the simulation; `ScriptEngine.addRunnableToQueue` appends under a lock and `ScriptEngine.update` runs the work on the thread that marks itself the main script thread. Every engine touch now goes through it, reads included — a position sampled mid-tick tears as easily as a write corrupts.

The first order proved the probe design rather than the order path. Issued cleanly, no exception, and nothing happened: the subject was `units.d.e`, drawables `base`, the Command Center. Three samples at the same coordinates. Had the success criterion been "no exception thrown", that would have been recorded as working. Re-targeted at the Builder the same code moved it 300 units, with a y excursion at t+5s corrected by t+10s — pathfinding around an obstacle, which also rules out a direct field write — arriving 2.5 units from the destination. That settles `e.a(float,float)` as move-to-point by observation, where the target-kind enum is single letters and reading it statically would have been slower and weaker.

Selection deliberately stayed out of the agent. It publishes the owned-entity roster and dispatches against an index; a mobility predicate guessed in the dispatch layer is exactly the decision logic the agent must not hold. That is also why the building order was possible to make, and worth having made.

Two multiplayer invariants got independent confirmation from the decompiled enqueue rather than from reasoning: commands are stamped with the `by` millisecond clock at construction, and the enqueue forks on network role with a server-side check on one branch only.

New drift guard: `Orders.verifyBindings` resolves six classes, six fields and four method signatures against the jar with no game running, asserted by `agent-selftest` inside `make check`. Not settled: the third owned entity, parked at (-1000, -1000), is unexplained — a lead, not a finding.

## [2026-07-25] tooling | whole-jar decompile, and the command path it found in one grep
Artifacts: `wiki/sources/m4-commands/` — `CommandController-c.java.txt` (114), `Command-e-public-surface.txt` (56), `Command-add-unit.txt` (20), `engine-construction-site.txt` (16), `engine-tick-decompiled.txt` (19)

Notes: prompted by the right question — were we reading this code or guessing at it? The honest answer was neither quite: everything to date came from `javap` disassembly, which is the most literal reading possible, but of exactly **one** method body plus class listings, with cross-referencing done by grepping that text. That method had already failed once, recorded in the M3 entry: the real writer of the tick counter was missed because the increment compiles against the subclass as a bare field name.

CFR 0.152 over the pinned jar: 1,698 source files from 1,698 classes in 19 seconds, pinned by version in `make decompile` because a decompiler is a heuristic and an unpinned one would silently change the source being reasoned about. Output is gitignored — derived work, regenerable in one command. Excerpts that a page cites are copied into `wiki/sources/`, the same discipline the javap artifacts already follow.

Validated against known ground truth before being trusted. `++this.bx` matches the `getfield/iconst_1/iadd/putfield` javap showed, and `this.by = (int)((float)this.by + f2 * 16.666666f)` explains the measured 1 kHz clock outright — 16.667 is 1000/60.

Then it closed M4 immediately. `this.h("CommandController"); this.cf = new com.corrodinggames.rts.gameFramework.c();` maps the label the boot log prints to the class, and `this.cf.c()` in the tick method shows it pumped every frame. `c.b(team)` is the order entry point: it constructs a command, stamps `e2.d = l2.by` with the same millisecond clock, and enqueues to one of two lists depending on `bX.B` — a server path that calls `prepareAndCheckOnServer()` first, and a client path that does not. The command object `e` carries the action (`k`), a target point (`l`), a target unit (`m`), and takes its subject units through `a(y)`, which is where the "gave an order to unit with team:" warning lives.

That is the last of the three prerequisites `engine-name-oracle` named, and it took one grep. The same question against disassembly would have meant finding every caller of an unnamed class by hand.

Standing rule going forward: decompiled Java is a **reconstruction** — plausible output that can be subtly wrong on obfuscated input. It is for navigation and hypothesis. Load-bearing claims stay pinned to javap bytecode or observed runtime behaviour, which is the citation hierarchy `SCHEMA.md` already sets.

## [2026-07-25] build | element expansion, and a traversal defect the JVM reported
Notes: the agent gained `inspectFields=`, expanding a named field's elements one level so the unresolved `X`/`W` collections become decidable rather than merely counted.

Building it surfaced a defect in the existing graph search. `findCollections` enqueued Collections deliberately, to descend through them — then walked their *declared fields*, reaching elements via `Arrays$ArrayList.a`, `ArrayList.elementData` and `AbstractList.modCount`. It worked, and `make check` passed, because JDK 13 only warns: "All illegal access operations will be denied in a future release." Two things were wrong with it beyond the expiry date — the route is implementation-specific (each Collection stores contents differently, so it is luck that any given one is reachable this way), and the guard against exactly this already existed and was applied in the sibling path, with a comment explaining why. One path had it, one did not.

Containers are now traversed by their elements: `Collection` directly, `Map` by `values()`, object arrays through `java.lang.reflect.Array`. Every other platform object is a leaf. Both routes share one `consider` so a match is recognised identically however it was reached; paths gained an element index (`.holder[0].tag`).

Worth recording how it was found: not by review. `make check` was green and the warning was scrolling past in the selftest output. `--illegal-access=debug` turned three anonymous warnings into a stack trace naming `Discovery.readQuietly:224` and `findCollections:174` in one step — after two wrong guesses from reading the code, both of which blamed a path that was already correctly guarded. Verified after: 0 warnings, was 3+. Four selftest cases lock it in — one that descent through a collection still reaches its elements, three that the platform internals never appear in a path.

## [2026-07-25] structure | evidence moved out of runs/
Notes: probe artifacts relocated `runs/m0-probe/` → `wiki/sources/m0-probe/` and all citations repointed. Cause: the api monorepo's root `.gitignore` excludes `**/runs/` as a DIRECTORY, and git cannot re-include a path below an excluded directory — so cited evidence under `runs/` could never be versioned, and every citation would have failed `source-path-exists` anywhere but this machine. `wiki/sources/` also matches the archive convention the workspace's registered wikis already use. The root `*.log` file-pattern exclusion is negated locally via `!wiki/sources/**`, which works precisely because it is a file pattern rather than a directory one.

## [2026-07-25] build | extractors on resource pools, and a stall window that was measuring the wrong thing
Artifacts: `wiki/sources/m11-pools/` — `type-flags.ndjson` (173), `pool-build-run.log` (414); `wiki/sources/m6-wire/world-sample.ndjson` regenerated (195)

Notes: the bot could build factories and had no economy, because the one structure that makes credits may only stand on a resource pool and the bot could not see one. Pools are terrain, so they are in no entity list, and the rule is in no stat dump.

Both gaps were closed by reading the engine rather than around it. A pool is a tile whose tileset declares `res_pool`; the loader turns that into a boolean on the tile and the extractor placement check reads that same boolean, and each link of that chain was read out of the decompiled loader rather than assumed. The placement rule comes from `placeOnlyOnResPool` in the extractor's shipped `.ini`, which the loader stores on the unit type; `make type-flags` now asks every registered type for its own predicate and dumps the answers. 173 types, exactly eight pool-bound.

Deliberately not parsed: the sentence "Can only be built on resource pools" in `Strings.properties`. It is hand-written per language and has no connection to the flag the engine enforces — the same words appear untranslated in the Japanese bundle. A bot reading it would be parsing marketing copy.

The tile scan cross-validates. The agent walks the live grid reflectively and reports 46 pools; decompressing the map's `Items` layer and counting the marked gid gives 46 at the same coordinates, tile (115, 6) both ways. Two unrelated routes to one answer is what makes the binding trustworthy.

One engine asymmetry worth remembering: a type's reported name does not always resolve back through the registry. One built-in reports `marker` while the registry matches built-ins on their enum constant name, so `ar.a("marker")` returns null. The dump asks each type object directly instead of round-tripping its name.

Then the live run failed, and the failure was not about pools. Three extractors and two factories were ordered; the third extractor was declared refused and the run stopped at 3/5. It had not been refused — it completed seconds later, and the roster afterwards showed it standing exactly on the pool tile it was sent to. The stall window ran from the moment of the order, so at a measured 12.4 world units per sample, 45 samples reached 558 units and silently capped how far the bot could build. The two near pools are within 230 units; the third is 588.

Timing one far build fixed the shape of it rather than the number. Ordering an extractor 622 units out, the builder travelled 50 samples and the structure appeared on the very sample it stopped moving — construction costs nothing measurable at this rate, travel is the whole delay. So the clock only runs while the builder stands still, which needs no speed constant, no frame rate and no assumption about map size. Raising the constant instead would have bought the same run and the same bug on a bigger map.

Also calibrated in passing: the catalogue's `speed` is not world units per frame. The builder is listed at 0.6 and measures about 50 world units per second.

Verified: 5/5 structures, 5 orders, no waste, reproduced twice. Three extractors on pool tile centres (203,130), (223,130) and (184,118), all three confirmed `res_pool` in the map file; two factories on ring offsets from the Command Center.

Wiki: new page `mechanics-resource-pools`. `wire-contract-ndjson` corrected — it still documented an `owned` count renamed to `visible` at M9 and quoted a capture line that no longer existed. `policy-loop` and `perception-visibility` updated.

Fog: settled rather than assumed. `perception-visibility` had offered "`-sandbox` does not apply fog" as an explicitly-labelled reading; the agent now reads the map's fog flag and the player's grid and reports the answer on every map scan. It is **disabled** on this map, so both visibility filters have been passing everything all along. The mechanism stays legitimate by construction; the behaviour under fog stays untested. The page keeps `medium` — what changed is that the gap is measured rather than suspected.

make check green: guard 0 violations, ruff + mypy clean, 100% statements and branches, agent-selftest OK.

## [2026-07-25] policy | the planner asks who can make a thing, and gains a second verb
Artifacts: `wiki/sources/m12-produce/` — `produce-timing.txt` (34), `produce-run.log` (414); `wiki/sources/m6-wire/world-sample.ndjson` regenerated (567)

Notes: the planner selected a unit to order by matching the type name `builder`. That was a constant standing in for a question the engine answers per unit, and it had already cost a three-hundred-sample run on a laboratory no builder can construct. The engine's build actions now ride in every world sample, so the question is asked before an order is spent.

Reading them wrong is silent, and the first cut did read them wrong. The predicate that looks like "makes something" — `a.s.g()` — is **false** on `a.v`, the action by which a builder places a structure, and **true** on `a.l`, the action by which a factory produces a unit. It means "produced without placement". Used as a filter it drops every structure in the game: the capture reported the bot's own Builder as able to make nothing, while the Command Center reported two. The correct test is the union — `y() != null` (places something) or `g()` (produces something) — and after it the Builder reports its thirteen real structures. `laboratory` is not among them, which is the historical stall explained rather than merely fixed.

The same field settles which verb to use. `y()` is non-null exactly for placed builds, so `placed` is the engine's own distinction rather than a guess from the produced type's speed, and a produce order carries no coordinate at all.

One exclusion turned out to be load-bearing. `editorOrBuilder` — the map editor's placeholder, owned, parked at (-1000, -1000), 170,000 hp — answers for 108 types against the Builder's 13, a strict superset including the laboratory. Selecting a producer by capability without excluding it would make almost no plan entry unbuildable, so the new pre-flight check would pass on types nothing playable can make and the order would go to a unit that is not in the game. A check that looks like protection while removing it is worse than no check.

The stall window could not be reused as-is: it resets on builder movement, and a factory never moves. Three answers were tried. Elapsed samples caps what the bot can afford, since production time is linear in price — a Builder ($500) took 34 samples and a Scout ($700) 45, or 14.7 and 15.6 credits per sample — so any fixed window silently forbids expensive units exactly as the old travel window forbade distant ones. Falling credits does not work either: the scout run read 4243 → 3678 → 3813 → 3849, rising through most of production as income outpaced the drain. The producing building's own queue depth does work, and it is now in the stream: measured, the Command Center reported queued=1 for all 45 samples the Scout took and zero on the sample it appeared. So both verbs share one rule — the clock runs only while nothing observable is happening, with movement the evidence for a placed build and a non-empty queue for a produced unit. No rate constant, no safety factor, no cap on cost.

Verified live: 6/6, six orders, no waste — three extractors and two factories placed, one Scout produced, with the dispatch log naming `a.l` as the action used. That closes the loop between the bytecode reading and the runtime behaviour. Reproduced across three runs, the last of them on the queue-based stall rule.

One more thing the editor placeholder broke, and this one is fixed. Reading a unit's production queue found its field by name alone, and `z` is not unique: an anonymous action class on that placeholder carries an unrelated field of the same name with no `c` inside it, so the read crashed the game outright -- `field c not found on units.h$10`. Obfuscation reuses single letters freely, so a name is not an identity; the field is now matched by declared type as well, against the queue class. The crash was the design working, in that the agent failed loudly rather than returning a plausible number, and it is the second time in this entry the placeholder has had to be excluded from something.

Worth recording how the factory case was diagnosed, because four separate attempts read as "the engine refuses production" and none of them was. A building joins the roster the moment construction *starts*, so a factory with an id, a position and a full option list can still be unfinished -- and an unfinished factory accepts a production order into its queue and never advances it, because the queue only ticks once the building reports complete. The roster cannot tell that apart from a refusal. Queue depth and the completion flag can, and both are now in the stream: measured directly, the factory appeared at t=4.0s with `complete=false`, finished at t=21s, took the order, spent credits 4440 -> 4289, and delivered the tank at t=29.5s.

That completion flag exposed one more defect, this time in the planner: a building joins the roster when construction *starts*, so counting on presence reported a plan finished while a factory was still a shell. Progress now counts only finished structures — and the correction had to land with its own consequence, because the builder stops moving the instant it arrives and the shell appears at the same moment, so movement stops being evidence exactly when construction starts. A rising owned structure now counts as in-flight too, or the fix would have traded a wrong scorecard for a false stall. The run is visibly slower and more honest for it: 559 samples for the same six entries where presence-counting took 215.

make check green: guard 0 violations, ruff + mypy clean, 338 tests, 100% statements and branches, agent-selftest OK.

## [2026-07-25] policy | goals instead of a build order, and the first look at what the bot does afterwards
Artifacts: `wiki/sources/m13-expand/` — `expanded-run.log` (438), `idle-after-plan.txt` (34); `wiki/sources/m11-pools/type-flags.ndjson` regenerated (487, now 173 types + 314 build edges)

Notes: the plan was a list a human wrote, prerequisites and all. Ask for a tank and the planner answered `blocked` — correct, and useless, since the way to get one was in the engine's own registry.

Two sources, and the split is deliberate. The option stream answers what each *owned* unit can make and is the right source for dispatch, because it carries the engine id an order is addressed to. It cannot answer what a plan asks, because a plan reasons about things that do not exist: nothing owned can make a tank until a factory stands. So the static half is dumped from the registry — every type asked for its own action list, each action for the type it makes. 314 edges over 173 types, riding in the same file as the placement flags because both are one pass over one registry and two files could drift against different builds.

The two cross-check, which is what makes either trustworthy: the registry gives the Builder thirteen structures, and the live per-entity stream reports exactly those thirteen by a completely unrelated route. The dump also settles the laboratory outright — no type in the registry produces one at tech 1, so that plan was never executable, and it is now refused before a socket opens rather than after three hundred samples of reported progress.

Expansion is a pure function run once, before the loop. Three properties earned their tests: availability accumulates, so two tanks insert one factory rather than two; goal order survives, so an extractor asked for first still opens the plan and pays for the rest; and the search terminates over a cyclic graph — a factory makes a builder and a builder makes a factory — by tracking what it is already resolving. Asked for a tank while owning nothing it names the real producers and refuses.

Verified live: goals of three extractors and two tanks, no factory named. The plan gained one, and six orders produced six entries with nothing wasted — the tanks made by unit 267, the factory that did not exist when the plan was written.

Then the question that had never been asked: what does it do afterwards? Every run to date stopped the moment the plan completed, which measures whether the bot can execute a list, not whether it can play. Observing 800 samples past a completed plan: nothing lost, not one hit point of damage taken, credits climbing 8,539 → 21,164, and visible enemy units going 54 → 126. "It survived" is true and misleading — it was not attacked. Nothing shows it can take a hit, return one, or notice it is being approached. The bot banks an economy it never spends while five opponents double their army.

That is the next thing, and it is not a planner gap: the planner does what it is asked. There is no policy for what to do with what it built.

make check green: guard 0 violations, ruff + mypy clean, 363 tests, 100% statements and branches, agent-selftest OK.

## [2026-07-25] policy | pools chosen by who can shoot the way there, and the produce diagnostic corrected
Artifacts: `wiki/sources/m6-wire/world-sample.ndjson` regenerated (567, entity records now carry `hostile`); new pages `policy-threat`, `mechanics-build-actions`

Notes: the planner picked resource pools by distance and nothing else. That is the whole answer on an empty map, and this one is not empty — an earlier run sent a builder 4,293 units out through two opponents' bases and it was killed before arriving. The pool was legal, unoccupied and the nearest one left.

Screening destinations would not have fixed it. The builder died in transit and the pool it was walking to was fine, so the test had to be applied to the walk: a pool is rejected when a visible hostile's attack range covers any point of the straight line from the builder to it. Because the pool is that line's endpoint, a pool inside a turret's field of fire falls out of the same test and needs no separate check.

Two origins, deliberately. Exposure is measured from the builder, because the builder is what gets shot and starts wherever it is standing. Distance is still measured from the anchor, because the economy should grow outward from the base rather than trail whichever pool the builder last walked past. One origin for both would answer one of the two questions wrong.

Hostility is the engine's, not the negation of ownership. `n.c(n)` compares alliance *group* rather than the team number the wire already carried, and returns false whenever either side is the neutral team — so an ally's tank and a neutral map object are both non-threats, and "everything that is not mine" gets both wrong. That flag now rides on every entity record. Reach comes from the catalogue's declared attack range, so no radius in the module is invented; an unarmed hostile has none, because an enemy builder on the route is an obstacle rather than a gun.

The refusal had to learn to say which refusal it was. "Every pool in sight is occupied" was the only sentence available, and it becomes a lie exactly on the runs where the reason matters — pools all built on is progress, pools all covered is losing ground. Both remain waits rather than blocks: a killed enemy stops covering a route the same way a destroyed extractor frees a pool.

Then the produce diagnostic, which was investigated on the suspicion that it ran on every order rather than only on failure. It does, it has to, and it was wrong about something else.

There is no failure to hang it on. The engine's two complaints on the dispatch path go through a logger whose rate limiter is a static counter that is never reset — four messages for the lifetime of the process, then it returns early forever. The other three conditions were never logged at all. So the reading must happen before dispatch or not at all.

The comment claiming the queue-add path "checks three predicates" was wrong on the count and on which. It checks five: the action resolves, it is available, it applies, the player is under the unit cap, and the cost is paid. The unit cap was missing entirely and is the one nothing else reveals — the engine's own count is `unitCountExcludingBuildingsIncludingQueued`, so a factory with a full queue is at the cap before any of it has rolled out, and the refusal that follows is indistinguishable from an order that never arrived. It is reported now.

The fifth gate cannot be reported, and finding out why was the useful part. `B().c(unit)` is check-*then-charge*: asking it deducts the cost. So a diagnostic can read four gates and must not touch the fifth, and an order passing everything the agent can read may still be refused at the till — stated rather than papered over. The same trap sits one argument away in the first gate, where `applies(unit, true)` routes affordability through the charging helper while `applies(unit, false)` routes it through a pure read. Every call site pins the false, and the reason now lives where the name is declared.

Reading gates and then dispatching anyway was the remaining half-measure. A closed gate now stops the order in the agent, logged as an error naming which one, rather than sending a command the engine drops in silence. It refuses rather than throws: being at the unit cap or short of credits is an ordinary state of a game in progress and clears on its own, where a missing action cannot and still throws.

One defect found in the build itself, and it had been lying for some time. `make agent` catches a failure to replace the jar and reports which cause to look for, but `Move-Item` fails non-terminatingly by default: PowerShell printed the error, skipped the catch, left the failure flag empty, and the target reported a successful build over a jar it never replaced. Observed directly — a running game held the jar open and `make agent` exited 0 with a stale artifact. `make check` was never affected, since the self-checks compile to a per-invocation directory and never touch the jar, which is also why this survived so long.

make check green: guard 0 violations, ruff + mypy clean, 380 tests, 100% statements and branches, agent-selftest OK.

## [2026-07-26] research | reading the opponent's own code instead of reading about strategy
Artifacts: `wiki/sources/m14-ai/` — `attack-group-staging.txt` (80), `unit-mix.txt` (73), `ai-state-dump.txt` (52)

Notes: prompted by "should we research strategies". The answer was no to external strategy writing and yes to one primary source, because the bot's losses are not caused by picking the wrong plan -- they are caused by building four tanks and stopping. Knowing an optimal build order does not help something with no continuous production. The binding constraint was mechanism, not knowledge.

The source worth reading ships in the jar. `com.corrodinggames.rts.game.a` is the AI package and identifies itself: two error strings begin `"AI: "`, and it draws its own state over the map with labels naming its fields -- `attackingCount`, `Turtling`, `StagingForAttack`, `AttackDelay`, `StagingTimer`, `UnitsWanted`. Those labels are what made the rest legible; nothing here was inferred from obfuscated field names alone.

Production is a weighted mix rather than a build order. Every registered type is offered to a predicate, admitted types get a weight (10.0 by default), and picking is weighted-random filtered by movement class and tech level. It never finishes, so it never needs to decide what comes next -- a different shape of decision from our ordered list entirely.

Attacking is delayed, then staged, then committed. A group's attack delay starts at 1000; when it expires the group masses and will not commit while any member is more than 170 world units from the rally point (28900 squared). Staging ends three ways: everyone gathered, seventeen seconds elapsed, or **any member taking damage within the last second** -- the engine logs "Not staging due to damage" and attacks at once. That last one is a reaction we have no equivalent of.

Two dispatch details worth copying. Committed groups re-issue every 800 ms rather than only on change, so a stale waypoint cannot outlive the world moving under it. And the AI attacks the *ground* four times in five: it rolls 0-99 and below 80 targets the position rather than the unit. A position attack does not chase, so a formation stays together; a unit attack follows a target that may be running and pulls the group apart.

Page is `confidence: medium` on purpose. All of it is decompiled source, which is strong for what the code can do and weaker than a run for what it does, and the standing rule from M4 is that decompiled Java is a reconstruction. Nothing here has been confirmed by watching the AI do it, and the page says so along with the cheap check that would: log an opponent's unit count and engagement timing against these constants.

Also corrected while linking: `index.md` claimed 17 content pages against an actual 19. Hub counts were right and there were no orphans, so the drift was in the total alone.

## [2026-07-26] research | reading the AI we lose to, and one claim that did not survive checking
Artifacts: none — this is a reading of `runs/decompiled`, which is derived and gitignored. New pages `engine-ai-zones`, `engine-ai-triggers`

Notes: the bot has no spatial model and no notion of when a force is ready. The AI it plays against has both, and the code is right there: `com.corrodinggames.rts.game.a`, about 5,300 lines over 28 classes. No new decompilation was needed and no community source was consulted, deliberately — a page sourced from forum strategy could not be fact-checked against anything, and every other page here can be.

The AI names itself. It draws a debug overlay over its own zones and those string literals survived ProGuard, so `unsafeBaseTimer`, `lastAttemptedBuilding`, `StagingForAttack`, `AttackDelay`, `Idle Builders` and the rest are the engine's own labels rather than inference. Two class names leak the same way: one calls itself PlainZone in a fallback log line, another TransporterGroup in a load-time complaint. Where a name had to be inferred the pages say so, and the state enums stayed obfuscated -- the overlay genuinely prints `State: a`.

A zone is a circle that owns units. Five kinds, closed set, ids fixed by the save format. Two things in the base class are worth taking regardless of strategy: containment for a unit inflates the radius by that unit's own collision radius, and a unit carries a back-pointer to its zone so joining one detaches it from the last -- a unit belongs to exactly one zone, and assignment is the arbitration. The bot currently picks a producer per decision with nothing stopping two decisions picking the same builder.

Placement is the sharpest contrast. Where the bot walks a fixed ring of eight offsets the engine can and does refuse, the AI rejection-samples: fifteen random points in the zone, each accepted only if the engine's own placement predicate accepts it, and null if none do. One unit type gets a search radius that grows 100 per failed attempt rather than giving up in place.

Then the claim that did not survive. The pool-expansion seeding really is unreachable -- guarded by a once-only flag set before the work is attempted, gated on already owning an extractor, inside a test that only holds on the first tick, with the flag written once and read once in 1,910 lines. What was wrong was the conclusion drawn from it. Two other sites create base zones on recurring cooldowns, and the live one draws a resource pool **uniformly at random** from the same map list, on a 2,000 cooldown, capped at two concurrent claims. So the AI does expand onto pools; the dead branch is a vestigial bootstrap. Worth noting for our own selection: a nearest-pool helper sits directly beside the random draw and the expansion path does not call it.

Its viability filter for a pool rejects a site within 300 of a hostile Command Center or 320 of an allied one -- the ally exclusion is the larger of the two -- and it uses the same alliance predicates the bot adopted yesterday, which is some independent evidence the right pair got picked.

On triggers: buildings are priced rather than thresholded. A build delay resets to 270 plus the zone id modulo 15 -- the id used as a phase offset so several bases do not all build on the same tick, which is a one-token fix for a thundering herd the bot will need shortly -- and the gate is a six-rung credit ladder, 1,300 credits when the base is near capacity rising to 4,800 when it is nearly out of room. A refused attempt cuts the delay by 120 and retries sooner than a success does.

Units come from a budget rather than a timer: a float that accumulates, fastest by a wide margin while the base is unsafe, cut sevenfold once two defensive groups exist, capped at 3.5 and clamped to 1.2 while poor and safe. It is spent in a burst -- up to twelve production attempts in one tick until the budget falls below 3.

Attacks are fill-then-commit. A group is created empty with a target size and recruits until full: three units for the first wave, five for the next few, seven after that, and 14 rising to 18 on the hardest difficulty. Exactly one attack group at a time. Staging ends when no member is more than 170 world units from the centre -- a rendezvous test, not a timer -- with two early exits that are more interesting than the happy path: any member taking damage within the last 1,000 frames cancels staging outright (`"Not staging due to damage"`), and a 17,000 timeout attacks regardless. While attacking it re-issues every 800, drops members that cannot reach, aborts when none can, and 80% of the time attack-moves to the target's position rather than ordering against the unit.

Both pages are `medium` and say why in the body: the timer constants are read but unobserved, and confirming any of them means watching an opponent's zone list over a live game. The agent reads its own team, not another player's AI object, so that is a real piece of work rather than a probe.

make check: not re-run for this entry -- documentation only, no source touched. The gate was last green on the threat work; it is currently red on another session's in-flight combat files.

## [2026-07-26] policy | continuous production, and the target churn it exposed
Artifacts: `wiki/sources/m15-production/` — `before-after.txt` (50), `sustained-run.log` (1293)

Notes: the third gap named by reading the opponents' code, and the one they said mattered most: they never stop producing, we stopped at the end of a list. Closing it was a new pure module rather than a change to the planner -- `production.sustain` decides what idle producers should start, `campaign.fight` sends it, and `build_order` is untouched.

What to make is deliberately not a new judgement. Reinforcement repeats the units the goals already asked for, because ranking units by some invented notion of combat worth is a guess with a number attached. Two limits come free from the engine instead of being modelled here: an option reports itself unavailable at the unit cap and under tech gating, since the agent asks the engine's own predicate. Credits are budgeted across the batch, because two factories that can each afford a tank cannot always afford two, and issuing both would leave the second refused for a reason the log could not explain.

Measured on the same map and opening. Without reinforcement: army 4 -> 0, phase over at 466 samples, nothing left. With it: army 4 -> 2, the phase ran its full 1500-sample budget, 45 reinforcements produced, engaged-targets-gone doubled from 7 to 14. It survives where it used to be wiped.

It is still losing, and the numbers say so plainly -- losses outrun replacements and the opponents went 47 -> 142 over the same window. Reinforcement bought survival, not parity.

The run also exposed a defect that was invisible while the army died early: 743 attack orders across 48 attacking units against only 24 distinct targets, about fifteen re-orders per unit. Targets are picked nearest-to-army-centre and that centre moves every time a unit dies or rolls out, so the whole army is re-tasked on a flip that may be a few world units wide. The engine's AI holds a target and refreshes on an 800 ms timer instead, which spends the same order volume on a stable choice. That is the next fix.

Also written: `policy-combat`, which two committed source comments already referenced as `(wiki: policy-combat)` before the page existed. That dangling reference was mine, from the previous commit.

make check green: guard 0 violations, ruff + mypy clean, 422 tests, 100% statements and branches, agent-selftest OK.

## [2026-07-26] research | a probe for the opposing AI's zones, built so the planner cannot use it
Artifacts: `wiki/sources/m15-ai-zones/zone-dump.txt` (75, distilled from a 4,601-line run); new page `engine-ai-probe`

Notes: both AI pages were `medium` for the same reason -- the constants were read from the decompile and never watched. Watching them means reading the opposing AI's zone objects, which is trivially possible and would be cheating if the planner ever saw it. So the probe exists and the constraints are the design.

Three reasons the planner must never see it, in increasing order of weight. A zone is intent rather than observation -- where the AI plans to expand, how big a group must be before it commits -- and a human infers that from units instead, so there is no observable counterpart to launder it through. Zones carry no visibility model at all: every entity has the engine's own per-player fog test, and the zone base class has containment and distance helpers and nothing resembling one, so reading zones does not stretch the fog rule but steps around it. And it would not work anyway -- zones exist only on AI players, the local human is constructed as a different subclass entirely, so a policy resting on this would beat the shipped AI and silently do nothing against a person. That is the objection that matters: not that it is unsporting, but that it produces a bot which cannot be evaluated against anything but itself.

Discipline is not a mechanism, so the separation is structural. The dump goes to the agent log and the planner reads the NDJSON stream and nothing else, so the Python side cannot consume it -- not does not, cannot; wiring it in would need a record kind, a decoder and a validator, which is a diff rather than a slip. And it is a boolean option defaulting off, with a self-check asserting the default, so an archived capture cannot quietly contain it.

Two choices inside the probe earned themselves immediately. Players are reached through the entities they own rather than through the engine's player table, which held the new pinned surface to three names. And fields are rendered generically rather than by pinned name -- for a probe whose job is to confirm what the obfuscated letters mean, a dump that applied the current reading could only ever agree with it.

Which is how the first run caught a wrong claim. Four AI players at 40, 90 and 150 seconds: radii confirmed at 420 and 360, expansion zones confirmed arriving over time on the cooldown path rather than at bootstrap -- so the dead one-shot bootstrap really does cost nothing -- and the capacity ratio confirmed bounded in [0,1], which had been the shakiest inference on the page. The unit budget turned out to start at -1.0 and range negative, which the source reading missed: it is a debt-then-credit counter, not a stock. And the reading that a shipped AI opens with an attack group of three is **not supported** -- every attack group observed targets five, from the first sample on. That is marked unsupported rather than quietly corrected, because the branch selecting 3 is really in the source and the discrepancy is unexplained.

Still unobserved: the third zone kind at radius 310 never appeared in 150 seconds, and no group reached its target size, so staging, the damage abort and the 17,000 timeout remain source-only. One limitation of the probe itself -- enum fields render as their class name, because the shared renderer deliberately calls toString on nothing but strings, primitives and wrappers, so the zone state the AI's own overlay prints by name still cannot be read from a dump. Naming enum constants is safe and is the next change to it.

agent-selftest green: bindings resolve including the three new names, and the option checks cover the default-off case. Python untouched.

## [2026-07-26] policy | target commitment, a Makefile target for playing, and a variance lesson
Artifacts: `wiki/sources/m15-production/` — `target-churn.txt` (33), `committed-run.log`

Notes: prompted by a fair question -- how come ours seems dumb? The answer was not that we needed to read more of the opponents' code. We already had the operative fact. Ours was dumb architecturally: the combat policy was a pure function of one sample with no memory, so it recomputed "nearest enemy to the army centre" every observation and never committed to anything. Their AI holds a target in a group object. The mistake was mine and it was conceptual -- I had conflated *pure* with *stateless*, when the build loop next door had been passing its own prior progress in as an argument all along.

Fix: the previously chosen target is an argument to `choose_target`, kept while it remains visible and hostile. Re-orders per attacking unit fell from 15.5 to between 2.2 and 5.9 across two runs.

The outcome did not follow, and that is the more useful finding. Two runs of identical code gave army 4 -> 0 at 1013 samples and army 4 -> 7 over the full 1500 -- the worst and the best the bot has produced. The opponents' unit mix is weighted-random by construction, so no two runs are the same experiment. Two runs cannot separate a regression from noise, and the page says so rather than picking whichever run flattered the change. The churn claim stands because it is a direct consequence of the change and holds in both runs.

Also added: `make play`. Every live run so far has been a hand-assembled PowerShell incantation, and one of them quietly reintroduced a bug `make agent` had already fixed -- writing the jar in place while a game held it open. The target builds a per-invocation jar and deliberately does not depend on `agent`, because a jar held by any running game cannot be replaced, including one started by another agent in the same tree.

Two bugs found while writing it, both worth recording. `PLAY_PORT ?= $(shell ...)` is recursively expanded, so the random port was regenerated on every reference: the game bound one port, the readiness poll watched a second, the planner dialled a third, and the failure message named a fourth. `:=` expands once and fixes it. And Windows will not let a jar be replaced while a JVM has it attached, which is why the per-invocation build is not merely tidy.

make check green: guard 0 violations, ruff + mypy clean, 426 tests, 100% statements and branches, agent-selftest OK.

## [2026-07-26] research | the probe corrects itself twice, and every enum in the jar turns out to name itself
Artifacts: `wiki/sources/m15-ai-zones/zone-dump-330s.txt` (74), `wiki/sources/m16-enums/enum-names.txt` (69, regenerable from the jar alone)

Notes: two things the probe reported that nobody asked it for changed conclusions, which is the argument for dumping every field rather than the ones the pages care about.

The first was a retraction that was itself wrong. Yesterday's dump showed every attack-flagged group targeting five where the source says the first wave is three, and the triggers page recorded the escalation ladder as unsupported. The dump also carried `B`, a flag no page mentioned: every one of those groups had it set, and it marks a sea group, whose target is five by a different branch entirely -- and this map is a water map. A longer run to 330 seconds caught the real thing, a fourth group appearing with A=3, h=true, B=false. A probe rendering only the fields the pages cared about would have shown two indistinguishable fives and left a correct claim retracted. The trace of that mistake is kept on the page rather than tidied away.

The second: the unit-production accumulator initialises to -1.0 and ranges negative, so it is a debt counter rather than a stock. A new zone owes a unit's worth of accumulation before it may make anything. The source reading had it filling toward a cap.

Then the enum rendering, which started as a probe limitation and ended somewhere much larger. The renderer printed enums as their class, losing the values; it now calls Enum.name(), which is safe under the rule it sits beside -- name() is final on java.lang.Enum and returns a stored string, so no engine code runs on the probe thread. The zone state and kind immediately read as words.

They read as words because **ProGuard renamed the enum fields and left the constant name strings alone**. The decompile shows `enum j { a, b, c; }`; the bytecode still carries `Pre`, `Prepare`, `Active`. That is a second naming oracle for the whole jar, needing no running game and no decompiler -- `javap -p -c | grep '// String '` recovers the constants of all 53 game enums, and the sweep is archived.

Several are worth more than the class names the oracle page was originally written about. `units.ao` is `NONE LAND BUILDING AIR WATER HOVER OVER_CLIFF OVER_CLIFF_WATER` -- the movement-layer model that mechanics-resource-pools records as the missing half of the reachability problem, sitting in the jar the whole time. `units.av` is the complete order vocabulary, seventeen verbs. `units.a` is the attack stances. `units.a.t` and `a.u` are the action taxonomy behind the two build verbs.

For the AI specifically: kind is `Main`, `ResourceOutpost`, `ForwardOutpost`, and state is `Pre`, `Prepare`, `Active`. So the radius-360 expansions are resource outposts by the engine's own name, and the unobserved radius-310 third kind is a forward outpost -- which explains why it is sited from a unit rather than a map point and screened by an aggressive filter rather than an economic one. Observed lifecycle: a fresh outpost reads Pre and an established one Active; Prepare has not been caught.

The general rule this leaves: before inferring what an obfuscated enum means, check whether the value already says. Between the constant strings and the debug-overlay literals, most of what the engine calls its own concepts is recoverable without guessing.

make check green: 426 tests, 100% statements and branches, guard 0 violations, agent-selftest passing with enum rendering covered.

## [2026-07-26] mechanics | reachability closes, and it was a comparison rather than a search
Artifacts: `wiki/sources/m17-movement/reachability.txt` (regenerable from the archived capture); `wiki/sources/m6-wire/world-sample.ndjson` recaptured (564, entity records now carry `movement` and `group`, pool records `group_land`); new page `mechanics-movement-layers`

Notes: the pool selector knew how far a pool was and who could shoot the way there, and had no idea whether the builder could get there. The pools page recorded that as wanting "a movement-layer model the planner does not have". It was in the jar.

`units.ao` is NONE, LAND, BUILDING, AIR, WATER, HOVER, OVER_CLIFF, OVER_CLIFF_WATER -- recovered through the enum-name oracle, since the decompile shows only a..h. And the engine precomputes connected components per layer, so its own reachability predicate, which names itself pathPossible in a log line, is: air and none answer true unconditionally, everything else compares two component ids for equality. Exact, free, and the same answer its AI uses to decide whether a group can reach a target.

Negatives are not ids: -1 impassable, -2 off the map, -3 grids never built. The engine rejects the first two then compares, which leaves a hole -- two -3s compare equal and answer true. The bot rejects every negative, which is more conservative and costs at most a site it might have allowed.

One thing nearly sank it. The first capture read group_land -1 for **all forty-six** pools: a resource-pool tile is not walkable ground, so taken literally every pool was unreachable and the economy would have stopped dead. What matters is whether a builder can stand beside it, so the four neighbouring tiles are sampled when the centre has none -- which is exactly what the engine's own AI does for the same reason, testing a zone centre and then four points around it. Tile step read from the map, not assumed.

With that, the map has something worth knowing: 34 of the 46 pools are in component 1, the mainland, and the other 12 sit in six two-pool components -- six island pairs on a symmetric ten-player map -- that no land unit can walk to. So the filter is not theoretical. Distance-only selection would have aimed a builder at one of those twelve as soon as the near ground filled.

Two sentinels confirmed themselves from units whose situation was already known. The Command Center reads NONE / -3: a building does not move, so no layer grid covers it. The map editor's placeholder, parked at (-1000, -1000), reads LAND / -2 -- precisely the off-map sentinel, from the one unit independently known to be off the map.

The land framing is deliberate and named in the field. A builder that does not travel on land is not judged at all rather than judged wrongly: its component indexes a different grid, so the comparison would be a confident wrong answer. Occupancy and threat still apply to it.

Two defects found on the way, both in shared files. `make wire-capture` and every other probe target had been failing with "unknown agent option discoverAtSeconds": a new option was added by opening a second if/else chain below the first, so every key handled by the first half fell through to the second half's else. Only typeFlagsPath escaped, by a `continue`. Merged back into one chain, with a self-check that parses an early key alongside a late one. And the pool record renderer had started reaching for a live engine, which broke the wire self-checks that exercise these shapes without a game; connectivity is passed in as an argument now.

make check green: 430 tests, 100% statements and branches, guard 0 violations, agent-selftest passing.

## [2026-07-26] mechanics | attack range for all 173 types, and the reason the old source had 90
Artifacts: `wiki/sources/m18-reach/attack-range.txt`; `wiki/sources/m11-pools/type-flags.ndjson` regenerated (660, now carrying a `unitcombat` record per type)

Notes: the threat model read reach from `-printunits` and treated an absent type as harmless. Checking that assumption is what started this, and the assumption was wrong in a way worth writing down.

`-printunits` emits 90 of the engine's 173 registered types. The filter is in `ar.s()` itself: not an orderable unit, name starting with "bug", shadowed by another type, a custom type without its listing flag, one specific exclusion, and an explicit sixteen-name blocklist. So 83 types had no entry and 48 of those are armed -- every turret among them. A threat model reading that source calls them all harmless, which is the one direction it must never guess in.

The fix is not a better default. The registry dump -- already one pass over one registry for placement flags -- now also asks each type for its attack range, read off the engine's prototype for that type, which is a map lookup rather than a construction, so nothing is spawned. Its own record kind rather than another field on the placement record: the two answer unrelated questions, and a type named for placement carrying an attack range is a lie the decoder then has to keep telling.

The cross-check is the part that makes it trustworthy: on the 90 types both sources describe there are **zero** disagreements. The registry is a wider reading of the same fact, not a competing one. And `reach_of` now indexes the table instead of defaulting through a miss, so a stale dump fails loudly and names the type.

The gap turned out to be reachable by ordinary play rather than by mods, which was the surprise. Twenty-seven buildable types are armed and missing from the old source, and most are the same unit in another mode -- the mode being the type name a live entity reports while in it. A deployed mech bunker at 240, a surfaced submarine at 240, a landed experimental gunship at 390, an amphibious jet underwater at 100. Plus the bug faction on any map with bug players, and the modular spider tree at up to 400.

Measured honestly, it changed nothing on the sample to hand. Of 55 visible hostiles, 20 read as armed under either source and the pool survey was identical -- occupied 15, unreachable 12, exposed 1, same tile chosen. The five AI players had built only priced types by then. This closes a latent hole, not a firing one, and the log says so rather than claiming a save.

Two smaller corrections fell out. The unit-catalogue page presented 90 as the catalogue, which reads as "the game's units" and is about half of them; it now says which 83 are missing and why. And the fifty `--- ERROR: running printForHelp()` lines in the printunits log are not errors -- they are a banner loop at the top of the printer.

make check green: 436 tests, 100% statements and branches, guard 0 violations, agent-selftest passing. Live 7/8 on a 500-sample run with the wider table, which was the risk worth checking: 48 more armed types could have made every pool read as exposed, and did not.

## [2026-07-26] refactor | one tick, one budget, one owner per fact
Artifacts: `wiki/sources/m11-pools/type-flags.ndjson` regenerated (660, now carrying four layer predicates per type); `wiki/sources/m6-wire/world-sample.ndjson` recaptured (579, entity records carry `flying`/`submerged`/`touching_water`, and a new `player` record carries the engine's own scoreboard); new pages `mechanics-combat-profile`, `policy-budget`, `policy-verdict`

Notes: an audit went looking for the divergent planners the run reports implied and did not find any. What it found was the opposite, and worse: layers that never ran.

**The seam was the defect.** The build loop ran the opening plan to completion and handed over to a fight loop. While building there was no army and no economy; once fighting there was no build policy at all, so `extractorT1` was the only structure that could ever be placed again and the factory count was frozen for the rest of the match -- which is the arithmetic behind the run that banked 7,013 credits behind a single Land Factory. And a plan that stalled meant a match that never fought, because the handover was conditional on the plan finishing. There is one loop now; `runner.py` kept the only part of itself that was never about looping, the judgement about whether an order already given is still being carried out.

**Two spenders, one balance.** Inside a single observation the production pass budgeted across every idle producer against `sample["credits"]`, and the expansion pass then asked the same field whether it could afford an extractor. Both correct alone; the pair committed one credit twice. With one factory the overlap was small enough to hide. `policy/budget.py` makes spending a single ordered decision, and every refusal now carries its reason into the run report.

**`c_tank` cannot shoot aircraft, and nothing read that.** It is in the unit's own `.ini`, three lines under `[attack]`. Combat selected on *having* a weapon and never on the weapon *reaching the target*, so on a water map the army could commit to a helicopter, hold it for as long as it stayed visible because commitment keeps a visible target, and never fire. The engine's own attackability test is four branches; it is now transcribed rather than modelled, with the attacker's predicates on the type record and the target's three states on the entity record so neither side is inferred.

Two rows of the regenerated dump changed real behaviour beyond that. `antiAirTurret` reports `hits_land: false` -- armed, 250 units of reach, and unable to touch a builder walking past, so it had been ruling out pools it could not defend. And every submarine in the game reports `hits_land_out_of_water: false`: torpedoes, which have to strike something in the water. Four hulls, and they are exactly the four that matter on the water map we play.

**The engine had the scoreboard the whole time.** `gameFramework.g.f` computes income, army value and building value per player and writes them into its own save; the enum constants name themselves through the ProGuard oracle, so the reader matches on `"income"` rather than on an ordinal. The bot had been regressing a credit balance against the clock across deliberately idle windows to estimate the first of those. First live capture: our army value 500 against four opponents on 1,000, and `income 42/s` -- which is base 18 plus three extractors at the `generation_resources: credits=8` each declares. Two unrelated routes to the same figure.

**The live run found what the tests could not.** With the loop unified, the plan and the economy both drove the one builder: the engine runs whichever waypoint arrived last, so 400 samples produced four expansion orders and a plan still stuck at 3 of 8. Whoever holds the builder now holds it alone. Same seed, same budget, after: plan 4 of 8, one expansion, factory up and producing.

Three defects fell out of coverage rather than review: a branch handling a budget refusal that could never happen, a guarded lookup whose miss was impossible by construction, and -- the one worth keeping -- `idle_producers` counting every owned unit, so "is every producer busy?" answered no for as long as the player owned a Command Center, and the throughput rule could never fire at all.

`tests/test_architecture.py` is new and is the part meant to outlast this: it asserts that exactly one module reads from the agent, that every other policy module is pure, that no exported policy *function* is unreachable from the rest of the tree, and that every record decoder has an encoder. The unreachable-function check is the one that would have caught `production_bound`, which was written, tested, documented with the measured evidence for it, and called by nothing -- while the suite stayed green.

**Measured, on the engine, same seed.** A 1500-sample match with everything in place (`wiki/sources/m20-one-tick/unified-loop-1500.txt`): plan 8 of 8, ten extractors, 29 reinforcements, 311 attack orders, army 0 -> 17, army value 500 -> 6,450 against a leader on 10,250. The reference the audit started from, on the same map: army 4 -> 2 while the opponents went 47 -> 142, three extractors for the whole match, 21,164 credits banked. Income read 98/s, which is base 18 plus ten extractors at the 8 each declares -- the third independent confirmation of that constant.

Of 112 visible hostiles, 104 read as engageable and 8 did not. Before this the army would have been free to commit to any of those 8 and hold it.

**The throughput rule, and a guess that measured worse.** Chasing the leftover credits turned up a second copy of the bug just fixed: `production_bound` called any unit offering a non-placed action a producer, and the Command Center always offers a Builder and is idle almost permanently -- so "is every producer busy" answered *no* on every observation of every match, and the rule had never fired once in the life of the bot. Restricting it to producers of a wanted type made it fire exactly once, because a factory is idle only on the single tick it finishes, which is the tick the rule is evaluated on.

So the test was rewritten to ask the budget instead: is there a surplus left after everything else has claimed? That fired freely, drained the bank -- and was the worst of the four shapes tried (`wiki/sources/m20-one-tick/factory-rule-ab.txt`). Income is the low-variance figure, being a deterministic function of extractor count, and it fell monotonically as factories rose: 98, 90, 74 credits/s, with army value going 6,450 -> 4,000 -> 3,300. **There is one builder, and every factory it places is an extractor it does not.** Income gates production, so buying capacity with the credits that would have bought income trades the thing that was working for the thing that was idle. The shipped rule is the conjunction of both tests, which restores 90/s and an army worth 5,400.

**The trace answered it, and the answer was not the one being guessed at** (`wiki/sources/m20-one-tick/throttle-trace.txt`). Widening the per-sample record to carry the production pipeline itself -- capable producers, how many idle, orders issued, claims refused -- settled it in one run: **one factory, busy on 97% of the observations it existed for**, and produce orders issued on exactly the 28 ticks it was free. Not the unit cap, since capability never lapsed. Not credits, since the budget refused nothing. Every free tick was filled.

The same trace surfaced a defect nobody had asked about: capacity never reached two, although the run reported seven factories expanded. The agent log showed all eight ring positions ordered once each and none completed. A factory joins the roster the moment construction starts, so its position reads as taken and the next order goes to the next one -- but it is not a *producer* until finished, so the throughput rule kept firing, and each order re-tasked the one builder off the last. `expand_production` had neither guard its sibling always had; and adding them was still not enough, because a refusal from the throughput rule fell straight through to the pool rule, which re-tasked the builder anyway. Builder availability is settled once now, by the dispatcher that owns the builder, before either rule is asked.

**Removing the throttle made the bot worse**, which is the finding worth keeping. With factories completing properly the bank drained to 315 credits and production doubled to 62 units -- against a *smaller* army, 17 rather than 22, worth 6,450 rather than 8,200. The bank was never the waste; it was the symptom of an economy ahead of what the army could usefully spend. Extractors compound and factories do not, so taking the builder off pools trades the asset that grows for the one that does not.

The expansion order was inverted to buy income before throughput on that reasoning. **It is not demonstrated.** Single-seed runs on this map are noisy -- three runs of identical code have given armies of 3, 6 and 14 -- and a later run under the shipped ordering produced 4 extractors rather than 9. The shipped order is the theoretically sounder default, consistent with the one clean comparison, and that is the whole of the claim. Settling it needs repeated seeds per arm.

make check green: 593 tests, 100% statements and branches, guard 0 violations, agent-selftest passing with the widened records covered.

## [2026-07-26] policy | a priority list cannot express an army, and a squared damage figure nearly sent the fix the wrong way

The question was whether the bot has a strategy. It did not, and the reason was structural rather than a matter of tuning: `sustain` took the first wanted type a producer could make and stopped, and `reinforcements` collapsed duplicate goals on the reading that four tanks meant one preference stated four times. Between them **a mixed army was not expressible**. Every idle producer reached the same head of the same list, so three 1500-sample matches ended with 33 identical `c_tank`.

That matters because of what `c_tank` is. It cannot shoot at aircraft at all, and roughly 15 of ~99 visible enemies per match were unreachable for that reason alone. Its 130 reach is the shortest of anything worth fielding, and every static defence in the game out-ranges it -- `c_turret_t1` by 1.27x, `c_turret_t1_artillery` by 2.69x, `c_turret_t2_artillery` by 3.54x. The loss table says the same thing in corpses: 96% of losses more than 2,000 world units from home, none at all within 900 of it.

Repeats are the ratio now. Each idle producer builds whatever the roster is furthest short of, measured as a **share** so one rule covers an army of three and an army of three hundred. Two details are load-bearing. Orders decided earlier in the same tick count toward the roster -- without that a batch of idle factories all see the identical shortfall and all fill it identically, which is the old bug rediscovered one tick at a time. And the worker fallback is a separate argument rather than the tail of the composition, because a share is owed to everything in the mix and a builder owed a share of a 34-unit roster is a land factory ordering builders.

**The unit value table was wrong, and it was wrong in the direction that mattered.** `Weapon` carries damage twice -- per shot and per volley -- and the engine prints a separate volley total only when it differs, so the decoder copies the per-shot figure across when it does not. For every single-barrel unit in the game the two fields therefore hold the same number. A derived ranking multiplied them, which squares the damage, and squaring reorders any two units whose damage and firing rate differ in opposite directions. Under it `c_artillery` read as the better buy over `c_tank` (2.96 against 2.38); corrected, it is 2.25 against 5.71 and less than half as good. The conclusion drawn from the bad table -- "the tank is the worst thing we can build" -- was exactly backwards. **The tank is the best thing the land factory makes.**

What caught it was printing the raw fields rather than the derived ones and noticing `direct_damage == direct_damage_volley` in all seven units sampled. The corrected figures reproduce numbers recorded from an independent earlier pass (`c_turret_t1` 16.40, `c_tank` 5.71), which is what makes them trustworthy now rather than merely different.

Two things follow from the corrected table. Everything ranked above `c_tank` is built by a **builder**, not a factory -- so the bot's spending ceiling is set by how many builders it holds, not how many factories. And `hoverTank`, briefly the candidate answer to the air problem on the strength of the bad numbers, is worse than a tank at everything except being able to shoot upward.

**The builder ceiling does not survive measurement** (`wiki/sources/m22-workers/worker-ceiling-ab.txt`). It was added on the argument that a builder is 500 credits of thing that does not fight, after a run bought 33 of them. Capping them caps extractors, which caps income, and the surplus then has nowhere to go -- the land factory can only start one 350-credit unit at a time:

    ceiling   extractors  income   worth   best rival  ratio   banked
    4               10     98/s   24,600      25,350    0.97   17,938
    8               10     98/s   23,450      26,850    0.87   15,229
    uncapped        13    122/s   40,850      26,650    1.53    4,620

Both capped arms end the match sitting on 15,000-18,000 credits they could not spend, and both are at or behind the strongest opponent. The ceiling stays a parameter rather than a constant precisely so this could be asked of a run instead of an argument.

`composition` is now a reported figure, because a composition is something the caller *asks* for and asking is not getting: a type the engine never offers leaves the mix silently at whatever else was makeable, and without the breakdown an experiment cannot tell a mix that was built from one that was requested and quietly denied.

**Still not established: whether we are killing anything.** `players 5 -> 5` in every run recorded so far, and the strongest rival's worth is only ever read at the first and last observation. Nobody has been eliminated in any match yet.

make check green: 617 tests, 100% statements and branches, guard 0 violations, agent-selftest passing.

## [2026-07-27] measurement | the bot loses, and every experiment so far was scored on a transient

Played to a verdict instead of to a sample limit, the bot is defeated or wiped in three of four matches. At 1,500 samples the same code reads as 1.26x ahead on total worth; by 3,500 its extractors have gone 14 to 0, its workers 36 to 0, its income 130/s to 0, and the strongest opponent has compounded from 4,700 to between 131,000 and 151,000 while fielding over 500 visible units.

The worker ceiling, the army composition and the wave mass were all measured at 1,500. None of those measurements is wrong; they answer a question that turned out not to be the question. Full-length runs are affordable now only because matches run four at a time.

**The build policy named the cause without new instrumentation:** `44 are built on, 0 cannot be walked to, and 2 can only be reached through enemy fire`. The opponents hold 44 of the map's 46 pools. The same run records 275 expansion orders and one surviving extractor -- the bot is not failing to expand, it expands constantly and loses every claim, at 700 credits a time.

So expansion without defence is a credit shredder, and the obvious ranking inverts. Upgrading an extractor to tier 3 is a real 2.3x on income -- 8 credits a second becomes 20, and the extractor upgrades *itself* with no builder and no new pool -- and it is worth nothing while 246 of 247 extractors die.

**The bot cannot build a defensive structure at all, by construction rather than by policy.** Three gates compose: `economy.py` names exactly two placeable types, `reinforcements` drops every immobile type from the composition, and `sustain` skips placed options. A builder can place thirteen things and the bot places two. `c_turret_t1` is the best damage per credit and the best hit points per credit in the game, costs less than the extractor it would defend, and has never been built. Seventeen upgrade paths are unreachable for the same reason.

**One real bug found and fixed, which changed nothing.** A builder is produced by the Command Center *and by every Land Factory*. The policy asked for one only as a fallback, and a fallback is reached only by a producer that can make nothing in the composition -- which a Land Factory never is, because it can always make a tank. When the Command Center died, twenty-two factories built tanks while the player had no builder and no way back, ending `plan blocked: nothing the player owns can make extractorT1` with `workers 0`. At zero builders the builder now goes into the composition instead. Measured on the same four seeds: one survival became two, and one seed died 294 samples earlier. Noise. The fix stays because a permanent-death trap is indefensible whatever the scoreboard says, but it is not why the bot loses.

No test caught it: every fixture gave the builder option to the Command Center alone, so the fallback always fired and the real case -- a producer that can make *both* -- was never constructed.

**Infrastructure.** Matches now run several at a time, one cloned game directory per worker, because a running match writes three fixed-name paths inside its own directory and everything else was already per-invocation. Jobs are a file, results are one file per match, so a batch is resumable by construction and crash-isolated. Lockstep is mandatory for a batch: free-running, parallel matches under CPU contention would sample at different game-times and the act of running them in parallel would change their results.

`campaign.py` went from 1,207 lines to 453, split into `scoreboard`, `match_report`, `workforce`, `recorder`, `dispatch` and `spending`. The architecture guard caught that the first attempt broke the "exactly one module touches the wire" invariant, so `dispatch` and `spending` are pure -- they return typed orders and the loop sends them -- rather than the guard being weakened to fit the refactor.

make check green: 697 tests, 100% statements and branches, guard 0 violations, agent-selftest passing.

## [2026-07-27] agent | the upgrade the bot could not see, and the filter of ours that hid it

The opponents hold twelve upgraded extractors against four un-upgraded ones. Ours held none, and three separate investigations concluded that upgrading was out of reach. All three were wrong, and each was reached by reasoning correctly from real evidence.

The first said the engine never offers it: a probe played the real opening until four extractors were standing, asked what every owned structure offered, and got nothing at all. The observation was right. The second said it was gated behind a tech level -- `extractorT2` declares `techLevel: 2`, and the engine registers a type's build action only into the action lists at or above its tier. Both facts are true and neither is why the extractor was silent. The third said it needed a tier-two builder: `mechEngineer` is produced by nothing in this build and `combatEngineer` only by experimental units, so the chain read as 44,500 credits of prerequisites. Also true, also not the answer.

**The agent was dropping the action before it reached the wire.** `BuildOptions` discarded any action that neither placed something nor answered true to the engine's "makes something" predicate, on the reading that the remainder were stops and rallies. An upgrade is neither: the asset declares it as `convertTo`, a conversion. So it was filtered out silently, and an extractor that was offering an upgrade published no options at all.

That filter was a policy decision living in the agent, whose stated job is to publish what the engine offers and let the planner decide. Every action is now published, with `makes_something` carried as a wire field. With the filter gone, all four extractors offer `extractorT2` and the engine calls it **available** -- at tier one, with no prerequisite of any kind.

**Two faults surfaced on the way to a working upgrade, and both crashed the match rather than degrading it.**

The same filter existed in two places: the listing path and `actionMaking`, which resolves an order to an action. Removing it from one gave the worst of both -- the planner was offered an upgrade it could see, the agent could not find it to dispatch, and it threw inside the engine's script thread: `extractorT1 has no action making 'extractorT2'; it can make nothing`. A predicate duplicated across a producer and a consumer is the same shape as the sweep filter that silently dropped two report figures earlier the same day.

Then: a conversion does not fill the production queue. `queued` stays at zero for as long as it runs, so the structure keeps offering the upgrade it is already performing, and the order went out again every observation. One duplicate landed after the conversion finished, addressed to a unit that was now an `extractorT2` and could only make an `extractorT3` -- which is how the second crash announced that the upgrade had worked. Ordered once per structure now, the same way every other re-issued order in this codebase is guarded.

Measured live over 800 samples with both fixed: **income 54/s, which is the base 18 plus three tier-two extractors at 12 each.** The first income this bot has produced above the tier-one ceiling.

A reporting fault was caught in the same run. `count_extractors` matched `extractorT1` alone, so a player holding three upgraded extractors was reported as holding none -- a figure quietly meaning something other than what it says, which is exactly how the 1,500-sample reading went wrong. Every tier counts now.

**Also settled, and both negative.** The 300 fps cap is not a throughput ceiling: a full match reports 112,425 frames against an engine clock of 406,149 ms and about 400 seconds of wall time, so the simulation is paced by the wall rather than by frames and uncapping it would buy nothing. No bytecode patch was written, which is the point of measuring first. And the wave-mass, defence and turtle arms are all refuted at full length -- four hypotheses closed by measurement, none of which changed a verdict.

The wire capture was regenerated rather than patched, because `makes_something` cannot honestly be synthesised for records written before it existed. 369 option records became 483.

make check green: 735 tests, 100% statements and branches, guard 0 violations, agent-selftest passing with the widened option record covered byte-for-byte.

## [2026-07-27] agent | the upgrade path priced wrong, the chain shaped wrong, and a statistic that was never measured
Pages updated: mechanics-unit-value (new section: conversion pricing and the fork), policy-holding-ground (two corrections)
Artifacts: `runs/sweeps/upgrade-fixed/` — six full-length matches, six seeds

Three faults, found by reading `.game/assets/units/extractor/*.ini` and the `-printunits` dump directly rather than by reasoning from the code.

**The plan fix that had to come first.** `completed_count` was taught that an upgraded structure still satisfies the entry that built it. `next_unsatisfied_index` answers a different question off the same roster and had the identical exact-name-match bug, unfixed — so the count advanced while the index did not, which is a plan reporting progress it will not act on. Both go through `satisfies` now.

**A conversion is not priced at what the result costs to build.** `upgrade_income` claimed `extractorT2`'s `price` — 2,100, the cost to *build* one, a transaction nothing in this game can perform, because the builder places tier ones and nothing above. The engine charges 1,400, printed as `T2 Upgrade Price: $1400` on the tier one and declared as `action_upgradeT2`'s price in the asset. So every upgrade over-reserved 700 credits and was refused outright whenever the balance sat between the two figures, against a budget already refusing 1,185–1,685 claims a match. The price now comes from the **holder** and by **position**: the first entry of a unit's `upgrade_prices` is the cost of its own next conversion (1,400, 4,000, 8,000 down the line). Position rather than label, deliberately — the dump prints a tier three's overclock cost under `T2 Upgrade Price`, and a tier two carries both `T2` and `T3 Upgrade Price` at 4,000 for a single declared action.

The campaign fixture had priced `extractorT2` at 1,400 — the conversion cost — which described a world the game cannot produce and made the buggy reading look correct. That is why no test caught it. The fixture now carries both real figures, and `tests/test_policy_spending.py` asserts against the archived dump rather than against a fixture.

**The extractor line is not a line.** `extractorT3.ini` declares two conversions off the tier three — `action_overclock` to 30 credits a second at 1,100 hit points, `action_reinforce` to 20 at 4,700 with an 800 shield — and neither leads to the other; both carry only an `action_refund` back down. Modelled as one five-long chain, the code asserted that an overclocked extractor was an upgrade of a reinforced one, false in both directions. Two paths sharing a prefix state the truth instead, and `satisfies` needs no special case: it requires both types in the same path, and no path holds both siblings. `next_tier` walks as far as the paths agree and returns nothing at the fork, because which branch is worth more depends on whether the ground is contested and that is a measurement nobody has taken.

**A statistic that was never measured, corrected in four places.** "246 of 247 extractors the bot placed were destroyed" appeared in `policy-holding-ground`, in two docstrings and in a test. It was the only claim on that page carrying no footnote, and no file under `wiki/sources/` contains the figure: it is `275 − 28 = 247` **expansion orders** restated as placed-and-destroyed structures. An order granted by the budget is not a structure that went up — the builder still has to walk there and the engine refuses a placement silently. The endpoint scorecard cannot separate "built and destroyed" from "never built", and those call for opposite fixes, so the page now says so and points at `policy-trace` for the instrument that can.

**Measured: the plan fix alone, six seeds at 4,000 samples.** Three survived, three defeated, none wiped, against a four-seed baseline of one survived, two defeated, one wiped. On the four shared seeds it is a wash — 12345 and 4242 improve from wiped to survived, 777 and 31337 regress the other way. No signal.

A new symptom is unambiguous, though: **every one of the six built zero factories**, against 18–53 in the baseline, and finished with 12,480–35,984 credits banked, an army of at most one tank, and no worker at all in four of the six. Whatever else is true, the bot has stopped converting credits into anything. Not diagnosed here — the next run carries a per-sample trace, because that is exactly the question the endpoints cannot answer.

## [2026-07-27] measure | throughput-before-income refuted, and the frame limiter reopened
Artifacts: `runs/sweeps/throughput/` — six full-length matches; `runs/trace-12345.ndjson` — the first per-sample trace of a full match

**The trace changed what the problem looked like, and then the fix drawn from it failed.**

A per-sample trace of seed 12345 contradicts the endpoint scorecard on nearly every point. The scorecard reports `extractors 0 -> 0`; the trace shows the bot **holding a peak of 14**, and reaching **56,650 total worth at the midpoint against the strongest rival's 38,650 — ahead**. It then falls to 850 while the bank climbs to 24,866. The producer count never exceeds **two** for the entire match. Losses were 115 tanks, 98 builders and **34 extractors** — not the 246 a previous entry claimed, which was never a measurement.

The reading taken from that: two factories cannot spend that income, so income never becomes army. `expand_production` self-gates on `production_bound` (every producer busy *and* surplus enough for a factory) but ran last, reachable only when no pool was claimable, which with pools churning is almost never. So the gate was reordered to run first, on the argument that it would only fire when income genuinely could not be spent.

**Measured over six seeds it was the worst arm yet: three wiped, three defeated, none survived**, against three survivals on the same seeds without it. Expansion collapsed from 307–509 orders to **2–6**, every one a factory; extractors finished at 0 or 1 and income at 0/s. With one or two producers `production_bound` holds on nearly every observation, so the rule took the builder nearly every time it came free.

`production_bound`'s own docstring had said this and it was dismissed as a 1,500-sample artifact answering a question that had since changed. **It was not.** There is one builder, and every factory it places is an extractor it does not — and that holds at full length exactly as it held at 1,500. Reverted.

The banked credits are real and remain unexplained. What they are evidence for is too few *builders*, not a different use of the one. That is also what the engine's own AI answers with: it runs several bases, each targeting two builders, each claiming what is near it ([[ai-opponent-strategy]]).

**Scoreboard to date: no change has yet improved a verdict.** More builders, turrets before income, turrets from surplus, turtling, anti-air, wave mass, upgrades and the plan fix are all a wash or worse. What has improved is correctness and instrumentation.

**The frame limiter is reopened, having been closed on an inverted inference.** It was closed on the reasoning that a match reports an engine clock tracking wall time, so the simulation is wall-paced and uncapping buys nothing. The clock tracks the wall *because the limiter makes it*: `java/u.java:117` sets `setTargetFrameRate(300)`, `u.java:141-145` re-sets it every frame from `highRefreshRate`, and `java/b.java:122` then calls `Display.sync(targetFPS)` unless the target is -1. `Display.sync` sleeps. `Main.java:479` builds the same container under `-nodisplay`, so killing the renderer never removed it. Achieved rate is ~277 fps against a 300 cap and the JVMs sit at 13–21% CPU — sleeping, not computing. A 4,000-sample match at lockstep 75 is 288,750 frames, about 962 s of game time, roughly half the observed wall clock.

## [2026-07-27] measure | the noise floor, and why seeding did not remove it
Artifacts: `runs/sweeps/noise/` and `runs/sweeps/noise-seeded/` — twelve runs each of ONE identical job specification

**Every arm measured before this was read against nothing.** Twelve matches from an identical specification -- same seed, same arguments, same code -- came back:

| | unseeded | seeded |
|---|---|---|
| verdict | 3 survived / 9 defeated | 3 survived / 9 defeated |
| total worth | 350 – 15,350 (sd 5,158) | 500 – 15,850 (sd 4,848) |
| income | 0 – 54/s | 0 – 60/s |
| extractors | 0 – 3 | 0 – 4 |
| samples seen | 3,232 – 4,000 (sd 284) | 3,005 – 4,000 (sd 354) |

A 25% survival rate on an identical specification. So `eight` at one survival in six against `one` at none, `tank` and `air` at none in six, and the upgrade arm at three in six are all consistent with the base rate and none of them measured anything. The single run reported as evidence that more builders worked -- *survived, four extractors, 66 credits a second, worth 13,800* -- is the same specification as these twenty-four and sits above all of them. It was a top-tail sample read as a result.

**Two findings do survive, because they fall outside the floor.** More builders raised expansion orders from 16–22 to 107–182 where the floor's own expansion spread is 118–194, non-overlapping against the low arm. And throughput-before-income wiped three matches in six against **zero** wipes in twenty-four here.

**The cause is not what either candidate hypothesis said.** Twelve engine call sites use `java.lang.Math.random()` -- a JVM-global generator `EngineRandom` never touched, driving the AI's choice of which unit to plant a base at (`game/a/a.java:1713,1737,1761`), its site and worker-destination positioning on a random disc (`game/a/o.java:96-97,166-167`), and unit scatter (`game/units/y.java:4811-4837`). That is real, and it is now seeded -- verified in isolation across three JVM launches and live in a match logging both generators pinned.

**It changed nothing measurable.** Seeding fixes the *sequence*, not which draw each consumer receives: if the number of calls before a given decision varies, every consumer downstream shifts. And the map settles for 22 seconds of free-running wall clock before the planner attaches, on a simulation that advances by measured delta -- so runs begin from already-different worlds. The earlier delta-jitter reading was wrong about the mechanism and right that the wall clock is implicated.

**So the harness is statistical, and the data says how to use it.** Coefficient of variation across identical runs: total worth 1.1, income 1.0, expansions 0.15, **samples seen 0.09**. Survival time is the lowest-variance figure the scorecard carries and has the right shape -- longer is better, censored at the sample limit. Twelve runs give a standard error near 87 samples, so a 250-sample difference is detectable; detecting a change in survival rate from 25% to 50% would need about 58 matches an arm.

Experiments are paired across seeds from here, scored on survival time, and a one-match-per-arm screen of twelve compositions is not worth running: at this noise level it would report about three survivals whatever the arms were.

Every sweep match now writes a per-sample trace rather than passing `-`, because the endpoints proved actively misleading -- a match reporting `extractors 0 -> 0` had held a peak of fourteen and led the strongest rival at the midpoint ([[policy-trace]]).

## [2026-07-27] measure | the bot leads for sixty per cent of every match, then cannot spend
Artifacts: `runs/traces/r01..r12-s12345.ndjson` — twelve traced runs of one identical specification

**Averaged over twelve identical runs, and the shape is the same in all of them.**

| progress | our worth | strongest rival | army | extractors | producers | credits |
|---|---|---|---|---|---|---|
| 20% | 17,354 | 16,217 | 12.1 | 5.6 | 1.0 | 2,440 |
| 40% | 35,125 | 30,575 | 15.3 | 10.0 | 1.0 | 1,621 |
| 50% | **50,296** | 42,488 | 22.6 | 10.5 | 1.3 | 2,299 |
| 60% | 58,696 | 56,558 | 21.2 | 10.1 | 1.5 | 6,058 |
| 70% | 52,012 | 73,362 | 17.8 | 7.8 | 1.2 | 12,660 |
| 100% | 7,238 | 135,508 | 0.4 | 1.2 | 0.2 | 22,429 |

Peak worth averages 67,650 and arrives 63% of the way through; final worth averages 7,237. **The bot builds a leading position in every match and loses ninety per cent of it.**

**The cause is throughput, and it is not subtle.** The producer count is 1.0 for the whole first half and never passes 1.7, while idle producers sit at zero -- one factory, permanently saturated. Income keeps compounding to about ten extractors, and at the crossover the credits it earns stop becoming army: 2,299 at the halfway mark, then 6,058, 12,660, and 22,429 at the end. The rival's worth grows on a near-straight line through all of it.

So the bot does not lose fights. It loses because it cannot spend what it earns, and the army stops at twenty-two units while an opponent's does not.

That is the same diagnosis the throughput arm was built on and it remains right; what was wrong was the remedy. Re-prioritising the *one* builder's time toward factories only displaced the extractors funding everything ([[policy-production]]). The builder count has since been fixed -- a wanted builder joins the army composition rather than a channel only the Command Center could reach -- and eight builders raise expansion orders from 16–22 to 107–182. Whether that is enough to also buy factories is the next measurement, not an assumption.

**A better score fell out of the same traces.** Coefficient of variation across the twelve identical runs: final worth 0.67, peak worth 0.20, extractor peak 0.12, survival time 0.098, army peak 0.094, and the mean share of worth held against the strongest rival **0.066**. The endpoint figure every scorecard has reported all along is ten times noisier than a score computable from the trace beside it.

## [2026-07-28] measure | the bot won a match
Artifacts: `runs/traces/duel-veryeasy.ndjson`; `runs/duel-full.log`

```
verdict        won (won)
players        2 -> 1 (1 eliminated)
plan           8/8 -- done: all 8 plan entries satisfied
total worth    3500 -> 60950
best rival     5400 -> 2550 (peak 12500, worst dip 9950)
income         158/s
army           0 -> 23
extractors     0 -> 7
workers        7
samples seen   2387
```

The opponent was eliminated at sample 2,387, well inside the limit. Against the four-opponent game every figure here is unrecognisable: worth 60,950 and climbing rather than peaking at 67,000 and collapsing to 7,000; income 158/s against 26–66; seven workers against nought to two; an army of 23 alive at the end rather than decaying to nothing.

**The bot had never played the game it was being judged on.** `-sandbox` hardcodes a ten-player map, and the setup is read from a GUI document with no values headless, so every figure falls through to a Java default: four opponents, at Medium, on *Crossing Large (10p)*. No opponent is ever eliminated there, so the "best rival" reaching 135,508 was one of **four** growing unopposed while the bot fought all of them. Leading the strongest of four to the sixty per cent mark reads differently in that light.

**Getting there took two wrong readings of the engine, both caught by its own log.** Calling the match-setup helper directly with its last argument true set the players up and started nothing -- the argument defers the start, and without it the map's units are skipped for want of a player and both sides are wiped on the first tick. Then `bS.y()`, guessed as the starter, turned out to be a stopper. What worked was to stop reimplementing the startup and queue the engine's own script with the map substituted -- **and the opponent count then needs no override at all**, because the helper caps teams by the map's own count. Choosing a two-player map *is* choosing one opponent.

Difficulty is set after the load rather than before, because `loadConfigCommon` overwrites the field from the GUI's unread default and saves -- which is why `preferences.ini` reads `aiDifficulty:0` however it is edited. It is an **income multiplier on the AI alone**: 0.4x Very Easy, 0.7x Easy, 1.0x Medium, 1.4x Hard, 1.8x Very Hard, 3.7x Impossible. At Medium an opponent earns exactly what the bot does, which is the setting every prior measurement used.

**What this does not establish.** One match, at the bottom rung of six, and the noise floor for duels is unmeasured -- in the four-opponent game it was a 25% survival rate across twelve identical specifications. Twelve seeds at Very Easy are running to turn this into a rate. `expansions 67 (0 factories)` and 11,224 credits left say the throughput fault is still there and merely no longer fatal.

**Also fixed on the way.** A map path carrying a space split the `-javaagent` flag and the JVM aborted with `processing of -javaagent failed` before the agent loaded; the launch now renders the flag and its options as one argv element rather than a shell string, which is what `harness/launch.py` existed for and the Makefile recipe was quietly duplicating.

## [2026-07-28] measure | twelve of twelve at Easy, and the two faults that were costing the other nine
Artifacts: `runs/sweeps/duel-easy/`, `duel-easy-fixed/`, `duel-easy-throughput/` — three arms, twelve seeds each, paired

| arm at Easy (0.7x AI income) | won | timed out | lost |
|---|---|---|---|
| before | 3/12 | 9 | 0 |
| + plan deferral | 5/12 | 7 | 0 |
| **+ throughput** | **12/12** | 0 | 0 |

**The bot is never beaten at these rungs.** Every non-win was the sample limit, at both Very Easy and Easy, across every arm. What was failing was the ability to *finish*, and it failed in two distinct ways that the paired seeds separated cleanly.

**Fault one: the opening plan waited forever.** It is a sequence -- three extractors, then the factory, then the army -- and it stopped on whichever entry it had reached. When the third extractor had no free pool the wait was permanent: the factory was never built, no army was ever produced, and a match ran to the limit with five idle builders, 60,676 credits and nothing to fight with. Seed 90210 finished at 18 credits a second with total worth unchanged from its opening 3,500, which is the base rate with **zero** extractors. An entry with nowhere to stand now defers to the next. Only placement defers: being unaffordable or having no producer is a fact about the whole plan, and skipping past those would spend the next entry's credits because this one is short.

**Fault two: the bank was never spent.** The healthy-economy timeouts finished with a completed plan, an army of 26, five extractors -- and **44,660 credits against a single factory**, having knocked the opponent from a peak of 37,750 down to 6,650 without finishing it. One spare builder now buys throughput. The measured effect is exactly the intended mechanism: **4-8 factories** where there had been none, and **44-347 credits left** where there had been 40,000-52,000.

Worth noting what that trade looks like, because it is not what a scorecard reader would guess: income *fell* from 98-158/s to 50-66/s and total worth from 44-66k to 27-32k. Fewer extractors, more factories, a smaller position -- and every match won. Banking an economy was never the goal.

**This is the same change that was the worst arm ever measured**, when it reordered the whole chain: three wiped, three defeated, expansion collapsing from 307-509 orders to 2-6. The defect was arithmetic rather than priority -- there was **one** builder, so every factory it placed was an extractor it did not. Duels run with seven or eight now, so it takes one and leaves the rest, and a floor of two free workers withholds it entirely in the opening. Re-running it before the builder count was fixed would have repeated a known-bad arm.

**A fault of my own, caught only by pairing.** Deferral made seed 4242 go from five extractors and an army of 25 to none of either, reporting `stalled: extractorT1 was ordered but never appeared`. `OrderTracker` keyed an outstanding order by the completed count alone, so the deferred extractor and the factory after it -- both pending at the same count -- collided on one slot. The second target was never issued, and the clock then ran against an order that had never been sent. Keyed by what is being built now, and the clock restarts when the plan turns to another entry. **Verdict counts alone would have read 3 to 5 as unambiguous progress and hidden it.**

## [2026-07-28] measure | the duel ladder: twelve, twelve, and nine of twelve
Artifacts: `runs/sweeps/duel-veryeasy-fixed/`, `duel-easy-throughput/`, `duel-medium/` — twelve seeds a rung, paired

| rung | AI income | won | timed out | lost |
|---|---|---|---|---|
| Very Easy | 0.4x | **12/12** | 0 | 0 |
| Easy | 0.7x | **12/12** | 0 | 0 |
| Medium | **1.0x** | **9/12** | 3 | 0 |

**Medium is the bar that matters: the multiplier is one, so the opponent earns exactly what the bot does.** Every measurement this project took before the duel work ran at that setting -- against *four* such opponents at once, on a ten-player map, with none of them ever eliminated. What looked like a bot that could not win was a bot that had never played a winnable game.

**Zero losses at every rung, in every arm.** Every non-win is the four-thousand-sample limit expiring, never a defeat. Medium wins finish in 1,598-2,792 samples and Very Easy wins in 1,306-1,823 -- a five-hundred-sample band, which is what a policy that reliably executes looks like rather than one that sometimes stalls. Very Easy was 9/12 before the fixes and is 12/12 after.

**The remaining failure mode has not changed in kind.** All three Medium non-wins sit at 18-38 credits a second against 50-66 in every win: the same economic signature the plan-deferral and throughput fixes attacked, so there is headroom left in that direction rather than a new problem to find.

**A warning worth carrying forward.** The change that took Easy from 3/12 to 12/12 *lowered* income from 98-158/s to 50-66/s and total worth from 44-66k to 27-32k. Fewer extractors, more factories, a smaller position, every match won. Any search optimising income or worth -- the two figures a scorecard makes most prominent -- would have rejected it. The only figure that improved was whether the match was finished.

## [2026-07-28] measure | the duels were never a pool race — every seed builds three extractors, most lose them, and turrets do not stop it
Artifacts: `wiki/sources/m28-holding/extractor-survival.txt`, `pools.py`, `holding.py`, `where.py`, `deaths.py`; `runs/sweeps/duel-hard-defence/` against `runs/sweeps/duel-hard/`
Notes: Read from the per-sample traces the twelve Hard duels had already written, so it cost no runs — the "built and destroyed" versus "never built" distinction [[policy-trace]] exists to make, finally asked of a batch that carried it.

**Every one of the twelve seeds reaches a peak of three extractors**, inside the first quarter of the match (first at 1-4% of the run, peak at 9-22%). There is no race being lost. The verdict follows what happens after the peak, with no overlap at all:

| extractors lost | seeds | verdict |
|---|---|---|
| 0 | 4242, 555, 777 | **won** |
| 1 | 31337 | **won** |
| 2 | 60613, 8675309, 90210, 99991 | not won |
| 3 | 12345, 1337, 8128, 24601 | not won |

Every run also regains three or four, so the bot is not failing to re-expand either — it re-buys the same ground at 700 credits a time and cannot keep it. That answers the question [[policy-holding-ground]] has carried since it was written, about what 247 expansion orders bought: **built and destroyed**, not never built.

**The ending-income threshold recorded yesterday was a consequence, not a lever.** Income of 18/38/50/58 per second is the base 18 plus whatever extractors survived — a restatement of the same count, which is why it separated wins so cleanly and why treating it as the target would have been chasing the symptom.

**The obvious follow-up was tried and lost.** `undefended` offered every immobile structure nearest-anchor-first, "so the base is covered before the frontier", and the per-loss table from m21 refutes that ordering outright — **not one unit died within 900 world units of the base across either traced run**, with 96% and 72% of losses deeper than 2,000, and the structures among them extractors out where the pools are. Restricting cover to extractors, same twelve seeds and same rung: **wins 4 -> 0, drops 21 -> 24, and the first two losses in fifty-two duels.** Reverted; the reasoning is recorded because it was good and still lost. Four defence arms have now failed.

**The arm was not a fair test, and finding that out is what the new instrumentation bought.** The `structures` line shows three turrets standing across all twelve matches — two in one seed, one built and destroyed in another. Defence has never been a policy that ran, only one that was reached. Unverified candidates for why: it fires only when income declines, seldom on a nine-pool map; and its site is a bare `+60` offset never checked against terrain, which at a pool would be refused silently at the cost of a walk and a stall window per attempt.

**What the army cannot do at all.** `c_tank` declares `canAttackFlyingUnits: false` and prints as "Can attack ground only"; `c_turret_t1` is "Attacks ground units." That is the whole army and the whole defence. In all four winning duels the surviving enemies are **0 engageable** — every one is something the bot cannot shoot; in the losers, 20-25% of the opponent's force is. Not yet shown to be what kills the extractors, since 75-80% of their units are ground, but no turret we can place stops the other fifth.

**A measurement gap closed in the same change, and it is the more important half.** Nothing reported whether a turret had ever been built: our own buildings appeared in no report line, the trace has no column for them, and the expander keeps the *income* reason when defence declines, so the defence reason never reached a log at all. Asked whether one turret had been built across twelve full matches, the honest answer was that the run output could not say. The report now carries a `structures` line, and the question is answerable rather than arguable.

## [2026-07-28] measure | the economy was switched off most of every match, and the ladder moved two rungs
Artifacts: `wiki/sources/m30-ladder/ladder.txt`, `wiki/sources/m29-antiair/`, `wiki/sources/m28-holding/diag-post-worker-fix.ndjson`, `runs/sweeps/duel-{medium,hard,veryhard}*`
Notes: Five changes, in the order they were found. The first is instrumentation and it is what made the rest visible.

**Nothing reported what the bot was trying to buy.** `Budget.claim` had always recorded each request's purpose, amount and refusal reason, and `format_ledger` had always rendered it — neither was ever called outside its own tests. The loop kept a *count* of refusals: about four thousand sentences a match, discarded. Two records now survive to the report, and the load-bearing one is **which spender was even reached**, because "declined three thousand times" and "never asked once" were previously the same number. Four defence experiments had been judged on exactly that ambiguity.

**One busy worker switched off every spender.** `Expander.step` returned nothing at all while the opening plan held a worker. The plan holds one; the bot runs four to eight. Instrumented over 800 samples: **the expander was skipped on 572 of them**, and it fires while the plan is merely *waiting to afford* something, so a plan parked on price switched the economy off for the rest of the match. Fixed by naming the held worker rather than flagging it — expansion orders 4 → 21, income actions 7 → 32.

**Freed, the workers all walked to the same pool.** Occupancy is judged by what *stands* on a pool, so one being walked toward reads as free. A run granted **23 extractor orders, lost nothing at all, and finished with four extractors**. A defect the previous fix exposed rather than created — and the shape this wiki recorded as "275 expansion orders against a single surviving extractor" without being able to explain it.

**A turret costs 500 and an extractor 700, so defence won on price.** At Hard: **29 turrets bought against 4 extractors**, with 43 of 47 extractor claims refused for credits and income stuck at 34/s. Expansion now marks a refusal that was purely about money, and nothing cheaper is offered the balance the economy was short of.

**And the army was eating the economy.** `replace_losses` claims protected and unbounded; expansion claimed unprotected. The reserve protects the army *from* the economy and nothing did the reverse: at Very Hard **2,800 credits of roughly 65,000 reached the economy**, 129 units were produced and two survived. Expansion now claims protected below four extractors — a floor taken from the outcome data, where income ≥50/s won 36 of 36 and 50/s is base 18 plus four extractors.

| rung | AI income | won | lost | routs | median win |
|---|---|---|---|---|---|
| Medium | 1.0x | 9/11 | 0 | **4** | 2,000 |
| Hard | 1.4x | **6/12** | 0 | **2** | 2,434 |
| Very Hard | 1.8x | 5/11 | 1 | 1 | 2,907 |

**The bot now wins more at Very Hard than its own baseline did at Hard** (5/11 at 1.8x against 4/12 at 1.4x), and Hard went 4/12 → 6/12 with no losses. "Routs" are matches leaving the opponent with nothing; neither baseline produced one.

**Still binding, unchanged at both rungs:** the non-wins lose four to eight extractors and end at 0.0–0.3x the opponent's worth. Holding ground, not claiming it.

**Also this session:** four modules split along real seams — `siting` out of `build_order`, `defence` out of `economy`, `expander` out of `spending`, `codec` out of `state` — and an architecture test added so module size cannot drift back. Nothing is over 600 lines.

## [2026-07-28] measure | three unit-value arguments, three refutations, and the pattern behind them
Artifacts: `runs/sweeps/duel-veryhard-{floor,order,mechgun}`, `wiki/sources/m30-ladder/ladder.txt`
Notes: A stretch of negative results, kept because the *shape* of the error repeated and is worth more than any of the three findings.

**The tech question, answered properly.** Tech is per unit type, not a player level: a unit exposes only its own tier's action list, so tier-2 units need a tier-2 *builder*. That chain is real and expensive — builder → `experimentalLandFactory` (11,000) → `experimentalDropship` (30,000) → `combatEngineer` (3,500). **But it is not the relevant path.** Querying the registry directly, `mechFactory` costs **1,000** and our Builder can already place it; it makes `mechGun`, `mechMissile`, `mechArtillery` and builders. `policy.expand` inserts the prerequisite automatically, so reaching it needs no code at all. It was never built for a mundane reason: `economy.py` names two structure types and this is not one of them.

**mechGun refuted.** 0 won of 7 against the tank arm's 7 of 12, two of the losses being seeds that won comfortably before. The argument for it was that it beats `c_tank` on hit points per credit *and* damage per credit — 83 against 60, 7.7 against 5.7. It does. It is also **27% slower** (0.8 against 1.1), and on a ladder decided entirely by extractor survival a slower army defends scattered pools worse.

**Expansion-before-upgrades refuted.** Pools are six times better per credit than conversions — 87 credits per +1/s against 350, which [[policy-economy]] states as a rule and the game's own assets annotate. Reordered on exactly that arithmetic: **7 won → 5**, same two losses, routs 3 → 2, median win 2,207 → 2,362. Inside the noise floor, so not a refutation — but not the improvement the figure promised. What the arithmetic omits is **risk**: a new extractor is income that can be destroyed, a conversion is income on ground already held, and this ladder is decided by extractors *lost*. The original docstring said so — "the one income the map cannot take away" — and was read past.

**The pattern.** Three arguments, three per-credit metrics, three omitted dimensions: **range** (why `c_artillery` ranks last and outranges every enemy ground defence), **speed** (mechGun), **survivability** (upgrades). A table that ranks units by cost efficiency is silent about the thing that decides the match, and it has now been wrong three times in one session in three different ways.

**Two unpinned decisions found while doing it.** Swapping the upgrade and expansion calls broke no test; `expansion_reserve` had no test at all. Both are pinned now. The reserve also turned out to return the **maximum** price in the composition, so one expensive unit raised the barrier for the whole economy — a 1,400-credit artillery took it 450 → 1,400 and expansion was refused 232 times of 237. It is the mean now, which is what stops a composition A/B from silently being a reserve A/B as well.

## [2026-07-28] build | the style becomes a file: doctrines, a decomposed loop, and counter-composition
Notes: Structural session, no live runs — `make check` green throughout (837 tests, 100% statements and branches, guard 0). Three changes, one motive: testing a gameplay style should be an argument, not an edit.

**A doctrine is one file naming the whole style.** `rw_bot.policy.doctrine` carries goals, worker ceiling, wave mass, reserve (with `-1` = derive, the same sentinel the CLI override used), the expansion switch and the new counter switch, decoded with the same `require_*` discipline as every other flat payload. Presets live in `doctrines/`; `default.doctrine` is pinned to the `DEFAULT_DOCTRINE` constant by a test so the copy-and-edit starting point cannot drift, and the two shipped arms (`counter`, `no-expand`) are pinned to differ from default in exactly one field. `scripts/play.py` shrinks from ten positional slots to `[max-samples] [doctrine-path] [trace-path]` — the tail that grew one slot per question stops growing. Sweep job lines shrink with it: `label|seed|goals|max_workers|samples|mass|reserve` becomes `label|seed|doctrine|samples`, so what an arm *was* survives as a file rather than a job line reconstructed afterwards.

**The campaign loop is three objects instead of forty locals.** `play()` carried every report figure as a hand-threaded local — around forty of them — plus six more for wave state. The figures moved to `policy.scorekeeper.Scorekeeper` (observe per sample, assemble the report at the end); the wave memory — released, rallying, holding, last-sent pairings — moved to `policy.dispatch.WaveController`, the same shape `OrderTracker` and `Workforce` already have. Wave discipline is now exercised directly in `tests/test_policy_dispatch.py` rather than only by playing a whole match; the campaign's 62 tests pass unchanged, which is what says the refactor preserved behaviour. One naming note: the controller's method is `command`, not `step` — the monorepo test-quality guard reads `.step()` in a test as an ML optimizer pattern.

**Production can now read the record the loop already kept.** `enemy_types_end` sat on every report while the mix stayed blind to it — the 33-tanks-under-an-air-force failure ([[mechanics-combat-profile]]) was fixed by allowing a static mix, and the mix stayed static however the opponent played. `policy.counter.counter_composition` tilts the composition until its anti-air share covers the share of the visible threat that flies, repeating only types already asked for (a doctrine with no anti-air is left alone — which unit answers air stays the doctrine's question). All-air threats drop armed ground-only types; unarmed stay, a builder is in the mix for the economy. **Off by default and unmeasured**: `doctrines/counter.doctrine` is the A/B arm, and until seeds are run against `default.doctrine` the tilt is an argument, not a finding.

**Follow-ups this leaves open:** the docstring essays this session did *not* move to wiki pages (deliberately — separate pass, separate diff); `policy-loop` and the bot-architecture hub need repinning to the decomposed loop once this lands; and the counter A/B needs its sweep.

## [2026-07-28] correction | the reserve is the maximum again, and the log said otherwise
Notes: The previous measure entry ends "It is the mean now." That was true when written and was reverted the same day: twelve seeds at Very Hard called the mean a regression — **7 wins became 3**, routs 3 → 1, outside the noise floor — so `expansion_reserve` returns the **maximum** again and its docstring carries the full story. The confound the mean was meant to fix (a composition A/B silently also being a reserve A/B) is answered instead by a fixed per-arm reserve, which now lives in the doctrine file (`reserve` field, `-1` = derive).

## [2026-07-28] measure | counter-composition A/B — the mechanism works, the ladder doesn't move
Artifacts: `runs/sweeps/counter-ab-hard`, `runs/traces/{aa,counter}-s*.ndjson`, `sweeps/counter-ab.txt`
Notes: First doctrine-format sweep: twelve seeds at Hard on duel_lake, `doctrines/aa.doctrine` (fixed 3 tank : 1 hovertank) against `doctrines/aa-counter.doctrine`, one field apart. 24/24 filed, zero worker failures.

**On the ladder's deciding metrics, a wash.** Wins 7/12 → 8/12, losses 0 → 0, extractor drops 34 → 36 (peak-minus-end over the traces). Per-seed the flips go 4 to counter, 3 against — inside the noise floor, and stated so.

**On its own metric, decisive.** Summing `targets - engageable` at match end across the arm: **58 unengageable survivors → 3**. The fixed mix ends its worst non-win staring at 24 enemies it cannot shoot (seed 1337: extractors 0, rival at 127,600); with the tilt, every surviving opponent force is one the army can engage — the counter arm's non-wins are *outnumbered*, not *unarmed against the layer*. Median end worth 29,350 → 33,050 agrees. The tilt also visibly moved production: compositions run up to 2:1 tank:hover against the stated 3:1 when air showed.

**Reading: adopt, without claiming a rung.** The counter switch converts the unfixable failure mode ("cannot shoot what is left") into the fixable one ("not enough of what works"), costs nothing when nothing flies, and does not regress wins or drops. What it is *not* is a ladder move at Hard — the losses it fixes were already non-losses. Whether "outnumbered" is then answered by mass, defence or a second factory is the next question, and it is now askable per doctrine file.

**A trace-format note for the next reader:** `runs/traces/*.ndjson` are fixed-width tables, not NDJSON, despite the extension — column 5 is extractors.

## [2026-07-29] build | interception: the reserve answers the raid the turrets never could
Notes: The counter A/B's diagnosis (extractors bleed 5-7 -> 0-2 mid-match while the reserve stands at the rally point) meets the zone reading ([[engine-ai-zones]]): the engine's own AI answers raids with *mobile* Defensive groups tied to zones, and four static-turret arms are already refuted. `policy.guard.deepest_intruder` calls a hostile an intruder when it stands inside the engine's own resource-outpost radius -- **360**, the same figure the opponent uses for "territory" -- of any of our structures, and picks the deepest one at least one reserve unit can shoot, ties on health then id so a seed repeats. `WaveController` gains the doctrine-flagged branch: intrusion pulls the whole reserve onto the intruder, the raid's end sends the guards home again by the same forget-the-rally rule a disbanded wave uses, and only intrusion bypasses the wave gate -- attacking out stays the wave's business. The report grows an `intercepted` line so "never fired" and "fired constantly" stop reading identically, which is the ambiguity defence was misjudged on ([[policy-holding-ground]]). Doctrine field `intercept`, presets `aa-counter-guard` one field from `aa-counter`, `sweeps/guard-ab.txt` queued at Hard on the standard twelve seeds. Gate green: 100% statements and branches, guard 0.

## [2026-07-29] measure | interception A/B — drops down 43%, the death spiral endings gone
Artifacts: `runs/sweeps/guard-ab-hard`, `runs/traces/{counter,guard}-s*.ndjson`, `sweeps/guard-ab.txt`
Notes: Twelve seeds at Hard on duel_lake, `aa-counter` against `aa-counter-guard`, one field apart (`intercept`). 24/24 filed after one port-collision retry (`Address already in use` on a worker's back-to-back matches -- the resume path handled it as designed).

**Wins 9/12 -> 10/12, losses 0 -> 0, extractor drops 35 -> 20.** The interception fired for real -- 2,810 guard engagements across the arm, 25 to 870 per match -- and the report's own `intercepted` line is what says so, closing the never-fired/fired-constantly ambiguity defence was misjudged on. The deep endings are what moved: the control arm's worst non-wins finish at 0-1 extractors with the rival at 51,000-108,000; the guard arm's two non-wins hold 2-3 extractors with the rival at 19,000-37,900. Seed 8128 flips outright -- survived at 0 extractors, rival 108,650 -> **won**.

**The exchange rate did not move, and that is the finding under the finding.** Priced from the traces' loss tables, both arms trade at 0.45-0.46 drawdown-credits per credit lost. Interception does not fight better; it fights **where the income is**, and extractor survival is the whole difference. Consistent with the ladder law: matches are decided by extractors lost, not by battle efficiency ([[policy-holding-ground]]).

**The cost case exists and is measurable:** seed 8675309 (won -> survived) logged 870 intercepts -- a match spent entirely repelling raids, holding 3 extractors but never massing an attack. A reserve that always answers the raid is a reserve that sometimes never leaves home; whether a cap or a detachment (answer with part of the reserve) beats answer-with-everything is the next one-field question if interception's cost ever shows at scale.

**Method note:** the control arm re-ran fresh beside the guard arm rather than reusing yesterday's batch, and that was forced: seed 24601 flipped verdicts between identically-specified batches, so cross-batch comparisons are dead until the settle window is pinned ([[policy-determinism]]). Within-batch pairing stands.

## [2026-07-29] build | the settle window is pinned: hold at liveness, zero the frame counter, rebase to match age
Notes: Five acceptance iterations, each a two-run diff of the same seed, each eliminating one noise layer ([[policy-determinism]]).

**What was wrong, in layers.** (1) The 22-second wall-clock settle let the match world free-run before the planner attached -- the known source. (2) The premain seeding spent itself on the pre-match world: both pinned generators are global and the menu consumed a wall-dependent number of draws before the match existed. (3) The difficulty landed on a wall-clock frame, four seconds of sleep after a start whose own frame varied. (4) The deepest: the engine's *synced* randomness -- `f.a(min,max,salt)` -- hashes the match seed with the **boot-absolute frame counter** `bx`, mixing it into the result four ways, so two runs whose menus lasted a different number of frames drew differently on every call however thoroughly the generators were seeded.

**What the fix does.** The match script is queued from the game thread with the generators freshly seeded (pinning the load's own draws, spawn scatter included); a per-tick watcher detects liveness as engine state -- local player exists and owns units, since the engine singleton is constructed exactly once and "the game object was replaced" was never observable -- and on that tick reseeds, applies difficulty, **zeroes frame and clock** (the engine's own new-game convention, performed where this start path skips it), opens the channel, and holds the world for the first planner. A readiness probe cannot release the hold: only an ack counts, and a boundary nobody consumed stays on its frame. The recipe's settle sleep is gone in match mode; the planner rides out the unit-less opening by acking until the roster appears. Traces are rebased to match age, the coordinate the match actually evolves in.

**What two runs of one seed now share, measured:** identical content to the credit for thousands of frames, orders and events aligned. **What remains:** a genuine ±1-tick quantization -- the async loader posts the spawn at a wall moment, so which tick it lands in varies by one, and an opponent build event occasionally shows up one sample apart between runs (observed once at age 2925 in 600 samples). Whether that skew butterflies into different outcomes is not arguable from here: `sweeps/noise-held.txt` plays twelve full-length replicas of one seed, and the spread of those twelve scorecards **is** the new noise floor. The old floor, for contrast: a 25% survival rate on identical specifications.

**Wrong turns kept for the record:** holding on a game-object flip waits forever (`l.a(Context,n)` returns the existing instance, always); metering the load itself through lockstep made runs diverge *more*, because the load does wall-clock work off the game thread and pacing the game thread changes what interleaves with what.

## [2026-07-29] measure | the noise floor after pinning everything reachable — determinism refuted, and priced
Artifacts: `runs/sweeps/noise-held`, `sweeps/noise-held.txt`
Notes: Twelve full-length replicas of one seed, every identified source pinned: world held at first live tick, generators reseeded there, frame counter zeroed (the synced-random input, [[engine-tick-and-clock]]), logic step fixed at 3ms so CPU load leaves the physics. **Result: 8 won, 4 survived, zero losses, twelve unique scorecards.** Full-match determinism is refuted at this depth. Short runs are another matter -- 600-sample sequential pairs agree to the credit apart from rare single-event skews -- so the remaining source is something that compounds: the prime suspect is the path engine's worker threads, whose results land on a wall-timed tick, and pinning thread interleaving is engine surgery this stack cannot do from an agent.

**What the campaign bought, priced honestly:** every match starts ~26 wall-seconds sooner (the settle and its safety margin are gone); a seed now means one match for its opening minutes rather than for nothing; probes of a few hundred samples are effectively reproducible; and the floor is *known* -- verdicts on one seed sit in a won/survived band with no losses, so **verdict-level A/Bs still need twelve seeds and margins beyond two to three wins**, while mechanism metrics (drops, engageable, intercepted) remain the sharper per-match reads. The old floor, for scale: identical specifications could not hold a verdict at all, and seeding alone never moved it ([[policy-determinism]]).

**Operational note:** twelve parallel workers were repeatedly killed on this box -- twelve held JVMs is on the order of 12 GB -- and four workers completed cleanly. Sweeps run at four from here, which also keeps the certification representative of sweep conditions.

## [2026-07-29] research | how humans win — the community strategy corpus, and what it says we lack
Pages written: community-play-strategies
Notes: Four community guides mined (Steam's Multiplayer Basics and RTS Defence foremost) and recorded verbatim at low confidence — claims by players, each one a sweep waiting to run. The consensus loop: expand map control with military FORWARD and extractors born WITH turret cover -> military sized just above the opponent's -> surplus into fabricators. Scouting continuously is stated as the win condition itself ("KEEP. DOING. THIS."); a full counter matrix keys off it; the one micro that matters is kiting ("walk backwards when engaging"); expansions get two ground + two AA turrets with overlapping ranges and a repair bay. Everything the corpus names as essential -- scout, forward posture, counter-picking, kiting, raiding, fabricators, turret tiers, repair -- is something this bot has never done. Every named unit (fabricatorT1-3, amphibiousJet, scout, c_interceptor, repairbay) already exists in our catalogue; nothing needs the engine cracked to try. One wire gap found while checking: the command channel speaks four verbs (move, build, produce, attack) and the game's own controls include attack-move (double right-click) -- a raid or a forward rally without attack-move walks past enemies. Verb probe queued.

## [2026-07-29] measure | interception replicates at Very Hard — the death spiral is cured, the finish is not
Artifacts: `runs/sweeps/guard-ab-veryhard`, `runs/traces/guard-ab-veryhard/`, `sweeps/guard-veryhard.txt`
Notes: Twelve seeds at Very Hard (1.8x AI income), `aa-counter` against `aa-counter-guard`, one field apart, four workers, read against the measured floor ([[policy-determinism]]).

**Wins 3/12 -> 4/12** -- inside the floor, stated so. **Everything underneath moved, and in the same direction and magnitude as at Hard.** Extractor drops 35 -> 20 (Hard: 35 -> 20 -- the same 43% cut, replicated at a rung with double the pressure). Median end worth 18,050 -> 26,750. Zero-extractor endings 3 -> 0. The control arm's non-wins finish at 18-38/s income -- below the 50/s line that decides matches ([[policy-holding-ground]]) -- while the guard arm's non-wins finish at 46-70/s, economies still alive and fighting. Unengageable leftovers 165 -> 61. Interception fired 4,654 times across the arm.

**The bottleneck has moved.** Before interception the modal loss was the death spiral: income raided to nothing, army unrebuildable, rival compounding to 60,000-100,000. That ending no longer occurs in the guard arm at either rung. What replaces it is the healthy stalemate: five to six extractors, 50-70/s, worth in the high twenty-thousands -- **an intact economy that never converts**. The disease is now offence: waves that suicide into defended ground ([[policy-combat]]), no raiding, no scouting, no kiting -- precisely the gaps the community corpus ranks first ([[community-play-strategies]]).

**Doctrine ruling: `aa-counter-guard` is the line.** Two rungs, one mechanism story, no regression anywhere. The next arms build on it: attack-move (the engine encodes it as one flag on the move command -- `e.h = true`, read from the double-click dispatch), scouting with fog-memory intel, then raiding.

## [2026-07-29] build+probe | attack-move and scouting -- the first two verbs of an offence
Artifacts: `runs/attackmove-probe.log`, `scripts/attackmove_probe.py`, `sweeps/scout-ab.txt`
Notes: Two community-corpus gaps closed in one session ([[community-play-strategies]]).

**Attack-move, proven live.** The wire gains its fifth verb (`attack_move`: unit, x, y); the agent encodes it as the engine's own double-right-click does -- a move command with `e.h = true` written before enqueue, the flag named by the `doubleClickToAttackMove` setting that gates the dispatch. The probe produced a scout, ordered it past the enemy base, and read the ruling off the sample series: at frame 13,500 the enemy extractor's health fell 800 -> 769 with the scout holding 88 world units away and far short of its destination -- **it engaged what it met**, then fought past the land factory to the enemy Command Center, 350 hp to 70. A plain move walks through without a shot. Raids and forward rallies are now expressible.

**Scouting, with the memory that makes it worth anything.** Doctrine flag `scout`: one scout kept alive by the composition exactly as the builder is, walking the pool circuit farthest-first -- the far pools are the opponent's side. Its sightings land in `policy.intel.Intel`, a fog memory with a 9,000-frame trust window, and the counter tilt now reads *remembered* threats: the end-to-end test pins the payoff, a helicopter seen once and fogged still tilts production to anti-air where the unscouted arm reverts to tanks. The report gains `sightings`, because a scout that saw nothing and a scout never built read identically everywhere else. The scout is excluded from the army so no wave ever marches it in.

**Queued:** `sweeps/scout-ab.txt` -- aa-counter-guard against aa-counter-guard-scout at Very Hard, one field apart. The corpus claims scouting is the win condition; the batch prices the minimal version of it.

## [2026-07-29] measure | scouting v1 refuted -- the intel was real and both of its consumers choked on it
Artifacts: `runs/sweeps/scout-ab-veryhard`, `sweeps/scout-ab.txt`
Notes: Twelve seeds at Very Hard, `aa-counter-guard` against `aa-counter-guard-scout`, one field apart, detached run, 24/24. **Wins 3/12 both arms; everything else moved against the scout.** Non-win rivals finish at 97,000-113,000 against the control's 33,000-82,000; extractors end 0-3 against 3-7; median worth 18,150 against 21,600.

**The intel arrived and poisoned both consumers.** The compositions say how. (1) The tilt reads the remembered set's flying share, and a scouted enemy base is mostly buildings and boats -- so remembering MORE collapsed the air share toward zero and the arm finished with hoverTank x0-x1 where the unscouted control held x3-x14. Seeing everything made the mix blinder than seeing only what attacked. (2) The scout dies on its circuit -- a plain move walked into the enemy base -- so it is permanently the furthest-behind share, and the Command Center, the one producer of builders, spent whole matches replacing scouts instead: workers 4/5/1 against the control's 6/8/7. Seed 31337 ran its economy on one builder, which is the single-builder disease reintroduced by the eyes that were meant to help.

**Neither failure is scouting's; both are v1's.** The fixes name themselves: the tilt's threat set filters to mobile units, and the scout yields to the builder rather than outranking it. Re-measure as v2. Kept because the corpus's claim ("scouting is the win condition") now has a measured caveat: intel is only worth what its consumers do with it ([[community-play-strategies]]).

**Also banked from this batch's control arm:** interception wins are FASTER wins -- at Hard the guard arm's ten wins averaged 2,144 samples against the control's 2,485; win speed is now the standing tiebreaker for saturated win rates.

## [2026-07-29] build | scouting v2 and the raid -- the offence gets its first objective
Notes: Two changes, both answers to measured failures, gate green (895 tests, 100%).

**Scouting v2 fixes what v1's A/B convicted.** The tilt's threat set filters to mobile units -- a scouted base is mostly buildings and boats, and feeding them all in collapsed the flying share and left the arm with less anti-air than not scouting -- and the scout yields to the economy at the expander's own two-worker floor, because v1's scout outranked the builder at the Command Center and one match ran its economy on a single builder for it.

**The raid composes the two proven parts.** `policy.raid.Raider`: a first-wave-sized party (the engine's own first-group size) drafted by lowest id, attack-moved at the remembered enemy extractor nearest our anchor -- the frontier one. A raider standing on the memory of an extractor and seeing none reports the death (`Intel.forget`) and the raid advances; income types only, because raiding the army is the waves' job and raiding defences is what waves die to. The party is withheld from the wave controller -- one unit, one commander, the AI's own zone invariant ([[engine-ai-zones]]). Doctrine flag `raid`, report line `raids`, preset one field from the champion.

**Method note that reshaped the next batch:** cross-batch comparisons are dead ([[policy-determinism]]), so the composition arms staged earlier cannot borrow the scout batch's control. Everything queued rides in one batch behind one shared control instead.

## [2026-07-29] wiki | four pages for the session's features, attack-move folded into its owner
Pages written: policy-doctrine, policy-interception, policy-intel-and-scouting, policy-raid
Pages updated: issuing-orders (the fifth verb and its live proof), hubs/bot-architecture, index (32 -> 36 pages)
Notes: The log entries told the build stories; the pages now carry the durable facts -- the one-field discipline and in-batch controls on the doctrine page, both interception A/Bs with the exchange-rate finding, the v1 scouting refutation and the two fixes it forced, and the raid's rules with confidence held at medium until the all-arms batch rules on it. Every page cites its code paths and its measuring batches.

## [2026-07-29] finding | the amphib arm is void, and it found a plan pathology on its way down
Notes: Mid-batch read of `all-arms-veryhard`. The amphib arm's end compositions carry **zero amphibious jets**, and the scorecards say why: the plan expanded `amphibiousJet` through `experimentalLandFactory` at 11,000 credits and stalls at 7/11 "holding 248" -- because in THIS build's tree the airFactory makes lightGunship, c_interceptor and c_helicopter only, and `amphibiousJet` is produced by `combatEngineer` behind the experimental chain. The community counter matrix's favourite unit is a tier the corpus never mentioned ([[community-play-strategies]] now carries the caveat). The registry's full underwater-capable list -- missileShip, lightSub, heavySub, nautilus (all seaFactory), bomber (experimental chain) -- means **the sub answer requires water placement**, a real siting capability we lack, or accepting the naval gap on lake maps.

**The pathology worth more than the arm:** "waiting to afford" is a legitimate plan state with no ceiling, so a goals entry priced beyond the economy's reach holds a worker hostage forever -- `plan-holds-only-worker reached 255` and climbing in every amphib match. The plan needs either an affordability sanity check at expansion time (11,000 against a 4,000-credit start deserves at least a printed warning) or a savings cap that abandons an entry the balance never approaches. Queued.

**Method note:** the arm was registry-checked for the unit's COMBAT profile but not for its PRODUCER's price -- half a premise check. The build-tree walk is now part of staging any goals arm.

## [2026-07-29] finding | raid v1 refuted at 0/12 -- the party is an attrition conveyor, and the traces caught the wrong theory first
Artifacts: `runs/sweeps/all-arms-veryhard` (raid arm final; batch still playing), `runs/traces/all-arms-veryhard`
Notes: **Wins 0/12 against the in-batch control's 5/12** -- outside any reading of the noise floor (identical-doctrine replicas split 7/5, not 5/0). One field, `raid`, cost every win.

**The first theory was wrong and the traces refuted it before it shipped anywhere.** The party is FIRST_WAVE-sized and withheld from the waves, so the first wave cannot muster until the army holds six -- gate doubled, mechanism confirmed in code (`campaign.py`, `raid.py`). But the traces show the early game IDENTICAL between arms: army reaches 3 at sample ~595 and 6 at ~708 in both (production-limited, not gate-limited), rival worth ~22,000 at sample 1,000 in both. The ~113-sample wave delay moves nothing.

**The real mechanism is a mid-game attrition conveyor.** `Raider.strike` tops the party back up as members die -- by drafting ONE replacement, which then attack-moves across the map alone into a base the scorecards show holding 13 anti-air turrets, navy and air. A one-unit trickle, the exact thing FIRST_WAVE exists to forbid, issued forever: raid arms reinforced as much as control (168-176 vs 101-178) and ended with roughly HALF the army value on most seeds, kills no higher. The bleed compounds: fewer units home, weaker interception (1,008 intercept engagements on s777 against control's 238), extractors die, income halves (46/s vs 78/s), the rival snowballs to 57,000-75,600 where control held it near 24,000. The `raids` counter (2-6) hid the conveyor -- re-marches against the same objective count nothing.

**s777 is the counter-case that sharpens the lesson:** the raid arm took 72 kills there, double the control's 30, and still lost the economy race. Kills that do not protect income are not progress at Very Hard.

**What survives:** every part of the mechanism worked -- intel remembered extractors through fog, attack-move fought its way in live, ghost confirmation advanced objectives. What failed is arbitration: nothing asked whether the army could SPARE a party. V2 is designed and queued: a party or nothing (survivors below strength disband to the waves; no single-unit re-drafts), and drafting gated on surplus above the current wave rung's need (`WaveController.need()` + party size, judged in the campaign where the withholding already lives). A `marches` report line rides along so a conveyor can never hide behind a small `raids` count again.

## [2026-07-29] measure | the all-arms batch: four challengers, zero winners, and the economy is the whole story
Artifacts: `runs/sweeps/all-arms-veryhard` (60/60), `runs/traces/all-arms-veryhard`, `sweeps/all-arms.txt`
Notes: Twelve seeds at Very Hard, duel_lake, one in-batch control, every arm one field (or one goals entry) from it.

| arm | wins | extractor drops | median worth | verdict |
|---|---|---|---|---|
| control | **5/12** | 25 | 30,800 | champion stands |
| arty | 0/12 | 48 | 20,050 | refuted, second independent batch |
| amphib | 2/12 | 27 | 20,750 | void (mis-priced goal) + plan pathology data |
| raid | 0/12 | 44 | 21,450 | refuted (own entry above) |
| scout-v2 | 3/12 | 28 | 22,350 | refuted on economy grounds, softer than v1 |

**The ruling: `aa-counter-guard` remains the champion, and every added behaviour paid for itself out of the economy.** The batch's one-line lesson is the noise-held bimodality made causal: wins reach five-plus extractors, losses stall at three, and each challenger arm made the bad branch more likely -- arty by buying range instead of income, amphib by holding a worker hostage for a save that could never happen, raid by bleeding the interception reserve, scout by spending Command Center slots on dying scouts.

**Scout-v2 in detail, because its refutation is the most instructive:** v1's catastrophes are gone (no worker starvation below the floor, tilt no longer drowned), but control peaks at 7 extractors in 11 of 12 matches while the scout arm peaks at 4-6, income medians 46/s against 62-78/s. The scout still competes with the economy for production. And the one bright spot is real: **scout won seed 555, the only challenger flip of a control loss in the whole batch** -- the intel pays when the economy survives it. V3's design question is posed exactly: eyes that cost no production slot (scout only from surplus throughput, or intel from combat contact alone).

**Method note:** the win-speed tiebreaker was not needed -- no arm saturated. Amphib's two wins double as affordability-guard receipts: both seeds won on the nine affordable plan entries while the tenth held a worker hostage, which is the pathology priced, not the doctrine working.

## [2026-07-29] build | the landing: savings clock, first-sight metric, raid v2, and the batch that freezes its own code
Notes: Four changes in one gate (green: ruff, mypy strict, 100% coverage, agent self-test), all designed from the batch's receipts before it finished.

**The savings clock ends the hostage worker.** ``Decision`` carries its shortfall as a number (``deficit``); ``OrderTracker`` rules a price wait blocked when the shortfall sets no new low across ``AFFORD_STALL_SAMPLES = 90`` -- a window that measures *progress*, not duration, so a slow save is allowed to be arbitrarily slow and only an impossible one is called. The economics that make it honest: the plan claims nothing while waiting (``spending.build_plan`` claims only when acting), so production keeps spending the income and an expensive save under attrition is not slow but impossible. The ruling is not latched -- a new credit high-water lifts it and the plan reclaims its worker -- and ``build_plan`` releases the held worker on any non-building outcome, which also frees it on refusal stalls. ``play.py`` now prints the whole plan's price against the opening balance, so a 15,450-credit plan on a 4,000 bank is line three of every log. The other wait families (pool, ring, action-not-offered) remain unbounded and their comments now say so instead of claiming a bound that never existed; the raid batch showed a pool wait holding a worker 335 samples, so the question is open, priced, and deliberately not answered in the same patch.

**Sightings count first sights.** ``Intel.sightings_taken`` billed every re-sight of the standing army every sample -- ``sightings 166554`` on one raid scorecard, ~41/sample, none of it about scouting. First sights only now; a re-sight after window expiry counts again, correctly, because the memory forgot it.

**Raid v2 is the refutation inverted.** A party or nothing: survivors below strength disband and attack-move *home* -- the road back crosses the same ground -- and only a whole party is ever drafted, from units inside the rally radius of the anchor, so it starts together the way a wave does. Whether the army can spare one is the campaign's call against ``WaveController.need() + Raider.size`` -- the wave gate's own figure, read through the same ``wave_size`` muster uses so arbitration cannot drift from the gate. ``marches`` joins the scorecard as the conveyor detector. Re-measuring as ``raid2`` against the same 12 seeds and control (`sweeps/raid2-ab.txt`).

**Batches freeze their own code.** ``prepare_tree`` copies ``src/rw_bot``, ``scripts``, ``doctrines`` and the prebuilt agent jar into ``runs/sweeps/<batch>/.tree`` at launch; every match imports the snapshot (``PLAY_TREE`` -> ``python -P`` with ``PYTHONPATH`` at the tree, agent jar reused instead of compiled per match). The working tree is editable the moment a batch starts, a resumed batch resumes its *original* code whatever happened since, and the batch carries a record of exactly what it ran. The no-coding-while-sweeping freeze this session worked under is gone.

## [2026-07-29] build | guard_cap: the interception cost case becomes a one-field question
Notes: Built against the working tree while the raid2 batch plays from its frozen one -- the first edit the snapshot mechanism paid for. Gate green.

The guard measure's open cost case: one match logged **870 intercepts and never massed an attack** -- answer-with-everything defends by disbanding the offence. ``guard_cap`` is a new required doctrine field (``RW-DOCTRINE-008`` guards its range): zero commits the whole reserve, exactly the behaviour both guard A/Bs measured, so the shipped figure is a value rather than a special case; ``N`` commits the N *nearest* engageable reserve units, because an interception is a race with the damage the intruder is doing and the detachment that arrives first is the one that was closest when the alarm went. The rest of the reserve keeps gathering toward the wave.

Preset ``aa-counter-guard-cap`` pins the cap at **3** -- the first-wave size, because below it a detachment is a trickle by the engine's own rule and ours, and a smaller detachment would re-create at home exactly what raid v1 died of abroad. One field from the champion; queued for the next batch alongside whatever the raid2 verdict demands.

## [2026-07-29] fix | the freeze had a hole, and the first mid-batch edit found it in under an hour
Artifacts: `runs/sweeps/raid2-ab-veryhard` (4 valid results kept; 16 crashed matches cleared and resumed)
Notes: The snapshot copied `doctrines/` but `play_args` still handed matches the job line's doctrine path, which resolves against the repository root -- so matches imported frozen CODE and read working-tree DOCTRINE FILES. The `guard_cap` edit landed mid-batch, the frozen parser refused the unknown field exactly as the required-field discipline says it must, and sixteen straight matches crashed on their first planner line.

Two things worth keeping from the failure. First, **the loudness was designed and it worked**: the strict parser turned a silent contamination (new field quietly read by new code in some matches) into sixteen immediate, identical, attributable crashes. Second, **a doctrine file is as much the experiment as the code is** -- the freeze boundary is "everything a match reads that development changes", not "everything Python imports". `play_args` now rewrites the doctrine path into the tree; catalogue and type-flag dumps stay working-tree reads because they are build-pinned artifacts, and that boundary is now written down rather than implied.

The resume validated the design's other half: the batch reuses its existing frozen tree, so the four results filed before the edit and the twenty after it ran identical code -- one experiment, interrupted, not two experiments stitched.

## [2026-07-30] measure | raid v2: from 0/12 to cost-neutral, and the conveyor is provably dead
Artifacts: `runs/sweeps/raid2-ab-veryhard` (24/24), `runs/traces/raid2-ab-veryhard`, `sweeps/raid2-ab.txt`
Notes: Same twelve seeds, same map and difficulty, doctrine file identical to the refuted arm's -- only the code's arbitration changed. **Wins 3/12 against control's 3/12; drops 30 against 33; median worth 23,800 against 25,500.** Dead even on every figure that convicted v1.

**The conveyor is dead by the scorecard's own arithmetic:** `marches = raids x 3` in all twelve matches -- every raid a whole party, drafted gathered, never a lone replacement. The economy stopped paying: end extractors 2-7 and income 38-78/s where v1 sat starved at 46/s under rivals it had let snowball to 57,000-75,600; v2's non-wins mostly hold the rival to 26,000-54,000 (one 84,300 runaway remains, seed 555).

**Ruling: the raid mechanism is now free, and it is not yet worth anything.** Surplus-only drafting means the raid fires 1-9 times per match at zero measured cost, which converts the question from "does raiding hurt" (v1's answer: fatally) to "what makes it BITE" -- party size as a doctrine knob, richer objectives (builders are income too), or timing (raid the rebuild, not the standing base). The doctrine flag stays available and harmless; the champion is unchanged at even. Also banked from this batch's controls: the savings clock fired twice in 23 matches, both on genuinely impossible saves, both loud -- no over-firing.

**Method note:** control read 3/12 here against 5/12 in the all-arms batch -- the same doctrine, cross-batch, inside the priced +/-2-3 noise floor. The floor keeps earning its keep: without it that swing would have read as a regression hunt.

## [2026-07-30] build | raid becomes a size, and the cap-raid5 batch carries both open questions
Notes: Gate green. The `raid` doctrine field is an **integer now** -- zero for no raiding, N for the party size -- because the v2 measure's open question is precisely the size, and a size question answered by a code edit is an A/B that stopped being one (`RW-DOCTRINE-009` guards the range; the shipped raid preset pins `raid 3`, the size that measured cost-neutral). Same shape `guard_cap` took a day earlier: the flag family keeps shrinking into knobs.

Launched `cap-raid5-veryhard` behind one control, **six workers** -- the box holds 32 GB and the old four-worker habit was priced off a machine state that no longer exists: `guard_cap 3` (does a capped detachment buy back the offence the 870-intercept cost case spent) and `raid 5` (does a party heavy enough to kill covered extractors convert free raiding into wins). Second frozen-tree batch; the working tree stayed editable from the moment it launched.

## [2026-07-30] build | the Makefile stops being a program: recipe bodies become PowerShell files
Notes: The ugliness had one cause -- Make runs each recipe line as ONE shell invocation, so every recipe that needed control flow grew into a single-line PowerShell program; `play` had reached four thousand characters. Recipe bodies now live in `scripts/make/` as real parameterized files (`play.ps1`, `agent.ps1`, `selftest.ps1`, and one `probe.ps1` whose two parameters cover all four probe targets -- they only ever differed in the agent's argument string and where output lands). Every recipe is one readable invocation line; the Makefile keeps the variables and their defaults.

Boundary written down: `scripts/make/*.ps1` is launch plumbing, read from the working tree like the Makefile that invokes it -- what a frozen batch pins is what a match RUNS, not how it is BOOTED. Faithfulness notes from the translation: the play recipe keeps its exit-code semantics (the `income: play` chain depends on them), the agent build keeps its atomic tmp-then-Move-Item with the terminating-error catch (the jar-held-open failure used to report success), and both play branches, the agent build, the self-test and a live discover-probe were smoke-tested before the gate. Gate green.

## [2026-07-30] finding | two corpus claims die of arithmetic before any sweep: kiting and fabricators
Notes: Both settled from the registry and the engine's own unit files, at the cost of a grep each -- the premise check the amphib arm taught, done first this time.

**Kiting is unbuildable in this meta, and it composes two prior findings.** The corpus's kiting unit is the minigun mech, and the registry prices the tactic out three ways at once: `mechMinigun` walks at **0.6 against its chaser's 1.1** -- the gap closes at half a unit per tick and held-range kiting is geometrically impossible, worth at most ~160 ticks of free fire on first contact; the mech-family composition was **already refuted 0/7 on exactly that speed axis** (log 2026-07-28, "mechGun refuted"); and the composition that actually wins has no envelope to hold -- `c_tank` 130 against enemy `c_tank` 130, `hoverTank` 140. A kite verb would be micro for an army we measured out of playing. Not built, and the community page carries the caveat.

**Fabricators cannot pay back inside a match.** `fabricatorT1.ini`: price 2,200, `generation_resources: credits=2` -- an **1,100-second payback** against a full 4,000-sample match's ~890 seconds of engine clock, twelve times worse per credit than an extractor's ~88s. The guides that recommend massing them describe hour-long 100-300-tank games; at our match length a fabricator is dead credits by the engine's own numbers. Not built.

**What the same evidence leaves standing as the next defence arm:** the four refuted turret arms all predate defence actually running (three turrets EVER stood across those twelve matches; today's scorecards show ~27 acted defence purchases per match), and none of them was the corpus's actual advice -- a turret **born with the pool, as one purchase at claim time**. Untested, now testable fairly. Queued behind the cap-raid5 verdict.

## [2026-07-30] fix | defence stops placing turrets blind: the bare offset becomes an occupancy-checked cover ring
Notes: Gate green. `expand_defence`'s site was ``x + RING_SLOT_RADIUS`` for its whole life, reached for without looking -- its own docstring flagged the hazard ("never verified") and a raid-batch scorecard finally priced it: **27 paid turret orders, about five ever standing**, each silent refusal costing a builder walk plus a stall window. The site is now `siting.clear_site_near`: a two-shell cover ring (60/120, eight directions each, all inside the turret's 165 reach -- the base ring's slots sit 233 out, which is why defence could not borrow it), walked with the same occupancy predicate every other placement trusts, the covered structure itself exempt. An exhausted ring is a *stated* wait now, not a silent refusal loop.

**The confound this retroactively spreads:** all four turret-arm refutations ran under blind siting, so "the turret is not the answer" is measured only about a defence that mostly never landed. The next batch's control carries the fix; if drops move on their own, the turret question reopens with fair footing. On open ground the first cover offset equals the old one exactly, so nothing changes where the old code already worked.

## [2026-07-30] measure | cap refuted with its mechanism attached; raid5 says size is not the bite
Artifacts: `runs/sweeps/cap-raid5-veryhard` (36/36, zero failures), `runs/traces/cap-raid5-veryhard`, `sweeps/cap-raid5.txt`
Notes: Twelve seeds, one in-batch control, six workers, second frozen-tree batch.

| arm | wins | drops | median worth | intercepts |
|---|---|---|---|---|
| control | 5/12 | 18 | 29,700 | 3,886 |
| cap (guard_cap 3) | 2/12 | **34** | 24,200 | **1,693** |
| raid5 (raid 5) | 4/12 | 21 | 24,550 | 4,644 |

**guard_cap refuted, and the mechanism is in the row:** capping the interception at three units halved the guard's engagements and the drops nearly doubled. Answer-with-everything IS the guard's mechanism -- the deepest intruder dies to local numbers before the extractor does, and a three-unit detachment loses that race often enough to pay in both the detachment and the pool. The 870-intercepts-never-massed cost case was the price of the thing that works, not waste to reclaim. The question is closed; `guard_cap 0` stands in the champion.

**raid5 is cost-neutral, same as raid3 -- and it raided LESS.** 4/12 against 5/12 is inside the floor; `marches = raids x 5` in all nine matches that raided at all -- but three matches logged zero raids, because a five-unit party needs `need + 5` spare and that gate sometimes never opens at Very Hard. A bigger party raids more rarely and still does not convert, so **size is not the missing ingredient**. What remains untried: the objective set (enemy *builders* are the rebuild engine the raid was conceived against; extractors are only its fuel) and timing. The raid stays available and harmless at either size.

**Ruling: champion unchanged, `aa-counter-guard`.** Its rate now reads 5/12, 3/12, 5/12 across three consecutive batch controls -- stable around 4/12 at 1.8x income handicap, with the noise floor behaving exactly as priced.

## [2026-07-30] build | aa_cover: the first defence that can touch an aircraft
Notes: Gate green. New doctrine flag `aa_cover`: once the opponent has **shown** aircraft -- latched from sight, because sorties leave the viewport and AA that stands down between them is never finished when one arrives -- a `c_antiAirTurret` (600cr, 250 reach, air-only, Builder-placeable from the start) joins the cover beside structures, after ground cover in the spend chain, with its own `aa-cover` reach line. The gap it closes is total: the whole army and the ground turret declare `canAttackFlyingUnits: false`, so nothing the bot could place has ever touched an aircraft, which [[policy-holding-ground]] flagged as the next fair question two days ago. The generic `undefended`/`expand_defence` machinery took a second turret type without modification -- the cover test was already "a turret of THIS type within THIS type's reach".

Launched `aa-cover-veryhard`: control against `aa`, twelve seeds, six workers, third frozen-tree batch. Both arms carry the defence siting fix, so the control doubles as the first fair reading of ground defence since the blind-offset era -- if its drops move against the prior controls' 18-33, the siting fix did that.

## [2026-07-30] measure+fix | aa v1 never fired -- the reach line catches in one batch what once took four
Artifacts: `runs/sweeps/aa-cover-veryhard` (24/24, zero failures)
Notes: Control 6/12 (its best yet: 5, 3, 5, 6 -- and drops 26, turrets standing 4-6, all inside historical bands, so the siting fix shows **no measurable shift**; its certain gain was removing silent-refusal waste, and any win-rate effect sits under the floor). The aa arm read 4/12 -- but the number is noise about nothing, because **not one anti-air turret ever stood in twelve matches.** The arm's own reach line convicts it instantly: `aa-cover reached 50 acted 0 -- wanted 600 of 196 available`. Last in the spend chain, the AA stage was reached only when income and ground cover had both declined, and never with 600 credits still unspent.

This is the exact disease the ground turret had -- "a policy that was reached, never one that ran" -- which took FOUR refuted arms to diagnose because nothing then reported reach. The instrumentation built out of that failure caught the recurrence in one batch.

**V2 inverts the priority on the latch:** once the opponent has shown aircraft, AA cover outranks ground cover -- safe by the same gap that motivates it, since ground raiders already have the guard and nothing else in the bot touches an aircraft -- and falls through to the ground turret when 600 is not there, so cover is never held hostage to the dearer turret. Pinned both ways in tests; gate green; relaunched as `aa-cover2-veryhard` on the same sweep file, fourth frozen-tree batch.

## [2026-07-30] measure | aa v2 doubles its control -- and the noise floor calls it suggestive, not proven
Artifacts: `runs/sweeps/aa-cover2-veryhard` (24/24, zero failures)
Notes: **aa 4/12 against control 2/12, drops 32 against 37** -- the first challenger in the project's history to double its in-batch control. And the honest reading stops short of a champion change, for two reasons the batch itself supplies.

**The mechanism runs but the causality is muddy.** V2's inversion works: `aa-cover` acted 99 times across the arm against v1's zero, AA turrets stood at match end twice and died fighting elsewhere -- built, spent, doing *something*. But two of the four wins barely used it (acted 0 and 2; seed 4242 won without the opponent ever showing air), so the +2 margin is not cleanly the turrets' work.

**The floor got repriced under us.** The same champion doctrine has now closed consecutive batch controls at 5, 3, 5, 6, 2 -- a four-win swing on identical code. A +2 in-batch margin is exactly what that noise can produce on its own. The 12-seed A/B has hit its resolution limit for effects this size.

**Ruling: champion unchanged; the aa question graduates to the first 24-seed A/B** -- double the discriminating power, priced at one long batch, which an effect worth keeping should survive. Method note kept in bold: **verdict margins under 3 wins at 12 seeds are not findings**, and two of this week's rulings (raid2's tie, raid5's minus-one) already conformed without knowing it.

## [2026-07-31] measure | aa refuted at 24 seeds: the 12-seed doubling was the noise the protocol was built to catch
Artifacts: `runs/sweeps/aa24-veryhard` (48/48, zero failures), `sweeps/aa24.txt`
Notes: The first 24-seed A/B, run because aa v2 doubled its 12-seed control inside a floor that had just been shown to produce exactly such doublings. **Control 10/24, aa 5/24 -- refuted at a five-win margin, half the champion's rate.** The 12-seed "4 vs 2" was a mirage; one batch earlier it pointed the opposite direction with the same confidence.

**The mechanism is in two numbers:** drops 66 against 42, ground turrets standing identical (3.8 both arms). The latch spent ~3,000 credits a match on air-only turrets (118 aa-cover actions across the arm) without displacing a single ground turret -- the money came out of the economy instead, and the raids that actually take extractors are GROUND raids the AA turret cannot shoot. Air, at this tier and map, is simply not what beats us; v1's lesson was that the AA stage never ran, v2's is that running it hurts.

**Standing state:** `aa_cover 0` in the champion; the flag stays for maps where air pressure is real. The 24-seed protocol earned permanence in its first outing: it prevented a champion change that would have cost three wins in eight, and control's 10/24 is the first properly-sized baseline -- the reference figure for every verdict after this one.

## [2026-07-31] build | forward posture: the army moves to where the match is decided
Notes: Gate green. New doctrine flag `forward`: the reserve gathers at the **frontier extractor** -- the owned extractor farthest from the anchor, any tier counting by the plan's own `satisfies` test, id tie-break for determinism -- instead of at the base. Placement keeps its anchor; only the rally moves (`dispatch.rally_post`). Falls back to the anchor before any extractor stands.

The motivation is the one invariant six consecutive batches have not moved: matches are decided by extractor drops, the per-loss table puts each death 688-1,766 world units from the army's own fighting cloud, and the army has spent every one of those matches gathered at the base on the other side of that distance. The corpus ranks the same idea second of everything it teaches. Known interaction, recorded not fixed: the raid drafts from units gathered at the *anchor*, so `forward 1` + `raid > 0` would starve the draft -- both champion-off, and the arm carries only `forward`.

Launched `fwd24-veryhard` at the new standard from the start: 24 seeds, >=4-win margin, 48 matches, sixth frozen-tree batch.

## [2026-07-31] measure | forward posture ties exactly, and the invariant survives its most direct attack yet
Artifacts: `runs/sweeps/fwd24-veryhard` (48/48, zero failures), `sweeps/fwd24.txt`
Notes: **8/24 against 8/24 -- a dead tie at full 24-seed resolution.** Drops 54 against 57, worth flat. The one mover is the mechanism line: **intercepts up 28%** (9,791 vs 7,630) -- the reserve posted at the frontier genuinely fights more raids, and it does not matter. The drops happen anyway, at the pools the army is not standing at: one post cannot cover a spread frontier, and proximity was not the missing ingredient.

**This closes the posture family as asked, and it sharpens the ceiling case.** Six-plus batches of one-field arms around `aa-counter-guard` now read: everything refuted or neutral, the win rate parked at ~8-10/24 at Very Hard. The invariant -- matches decided by extractor drops -- has survived turrets (four ways, then fair-sited), interception caps, anti-air, raids at two sizes, scouting twice, and now the army standing directly on the frontier. What remains structurally untried inside this composition: nothing cheap. What remains outside it: denying the OPPONENT's economy hard enough to matter (the raid never bit), trading better (micro measured out at this tier), or changing the question -- Very Hard is a 1.8x income multiplier on equal pools, arithmetically unbeatable by economy, and the actual goal is a HUMAN, who does not cheat income but also does not play like this AI. The training-target question is now open on its merits.

Also banked: consecutive 24-seed controls read 10/24 then 8/24 -- the 24-seed noise band is about +/-1-2, an enormous improvement over the 12-seed +/-2-3-on-12, and the >=4 margin standard looks correctly sized.

## [2026-07-31] fix | the siting fix's shadow: a busy workforce was ruled "not playable", and Hard lost every win to it
Artifacts: `runs/sweeps/hard24-baseline-BUGGED` (kept as evidence), gate green
Notes: The first Hard batch since the landings read **1 win in 10** where the same doctrine had won 10/12 -- and the scorecard named the chain outright: `plan 3/8 -- blocked: nothing the player owns can make landFactory`, `expansions 136 (0 factories)`, `no-free-worker reached 3289`, `army 0 -> 0`, `attack orders 0`. At Hard the bot is rich, and the moment the defence siting fix stopped turret orders being silently refused, defence kept **all eight workers permanently employed** (126 turrets in one match). The plan's Land Factory never met a free worker, and `decide` ruled "no free worker offers this" as "not playable from here" -- a permanent block over a state the world leaves the moment anyone's hands free. The old blind-offset bug had been *accidentally* feeding the plan free workers by wasting defence orders; fixing defence exposed the plan's missing worker priority. Very Hard never showed it because poverty kept workers idle.

**The fix is worker priority to match the plan's credit priority, in three parts.** (1) `decide` splits the two states: nothing owned can make it stays blocked; every capable unit busy is a *wait*, carrying no unit because the plan wants whichever worker frees first (`_any_producer_exists`, ownership-checked -- the busy-blind twin of `find_producer`). (2) `PlanStep.wants_worker` carries the signal. (3) The expander stands down entirely while it is set (`plan-first-in-line` reach line), so the next freed worker is the plan's. Pinned at both layers plus an end-to-end: mid-job the economy buys nothing over a free pool with 4,000 in hand, and the freed worker builds the factory, not the pool.

Cost accepted and recorded: a silently-refused order on a lone worker now takes the workforce retry window *plus* the stall window (~90 samples, was 45) to be declared dead, because the busy state is examined only after the workforce frees the worker. `build_order.py` also crossed the module-size guard and split its progress reads (`completed_count`, `unsatisfied_indices`, `next_unsatisfied_index`) into `policy/progress.py`.

## [2026-07-31] build | the rush verb: the goal pivots to the ladder, and the ladder's top needs the earliest possible fight
Notes: Gate green. The project's goal is restated -- climb the AI ladder toward Impossible, watchably; the human-sparring milestone died when its human turned out never to have played the game (sparring mode stays built and shelved: `make host`). `make watch` opens a real window on any champion match, Impossible by default.

The Impossible record is 0/12 -- four defeats, eight wipes, the bot never reaching the enemy base -- and the arithmetic says why: a 3.7x income multiplier compounds, so nothing done LATE can matter. The counter that remains is the earliest possible fight, and the bot could not start one: **waves attack what is visible, and at match start nothing is**, so an all-in stood at its rally point while the cheat compounded. The new `rush` doctrine flag closes that: released waves attack-move at the **mirror of our anchor through the resource-pool centroid** -- skirmish duels are symmetric, so the reflection of our Command Center is the opponent's, pure geometry over what every sample carries -- until first contact, when the engagement policy re-tasks them (the engine runs the newest waypoint). `doctrines/rush.doctrine` is the all-in composition: one extractor, tanks immediately, mass 3, everything else off. `marches` now pools the raid's and the rush's outbound orders.

Also in this stretch's gate: `play()` crossed the complexity guard and shed `_draft_raid`/`_march_rush` helpers. Staged: `sweeps/impossible24.txt` -- champion control against rush at difficulty 3, 24 seeds, launching when the fixed-code Hard baseline files.

## [2026-07-31] fix | the second half of the workforce freeze: a finished job now frees its worker on sight
Artifacts: `runs/sweeps/hard24-baseline-BUGGED2` (kept), gate green
Notes: The worker-priority fix was necessary and not sufficient: the re-run still showed four of ten matches at `army 0 -> 0`, the plan now correctly WAITING on "every unit that can make landFactory is busy" -- and the worker never coming. The remaining leak was completion detection: a worker was freed only through the quiet window (site shows nothing rising for the whole retry window), and the defence cover ring packs turrets densely enough that a **neighbour's** rising structure inside the job radius kept re-marking finished workers busy -- a chain that freezes the whole workforce exactly in rich matches. `Workforce._is_free` now checks the one unambiguous signal first: **an owned, complete structure of the job's type standing at the job site frees the worker on that tick.** Collision-safe by construction: every siting path already refuses a site with a structure inside the same radius, so a fresh order can never start next to a pre-satisfied site. Also removes the 45-sample post-completion lag every build ever paid.

Third launch of the Hard baseline; its monitor now counts busy-deadlock scorecards per filing so a third leak, if any, is visible immediately. The goal this ladder serves was restated by Austin in plain terms today: **100% win rate at Impossible and every rung below.** The tech-tree page ([[mechanics-tech-tree]]) now prices every builder-reachable path; the artillery turret upgrade (350 reach for 2,100, on the turret defence already buys) is the sharpest unfielded weapon in it.

## [2026-07-31] fix | the third leak was the placeholder: defence was covering the map editor's unit, off-map, forever
Artifacts: `runs/champion-s12345.log` (the smoking gun: `build c_turret_t1 by 1231 at (-940.0, -1000.0)`), gate green
Notes: The completion fix was ALSO necessary and not sufficient -- the third Hard run still deadlocked 7 of 10 with the plan waiting and 51 builders bought against 5 alive. The engine log named it: turret orders at **(-940, -1000)**, beside the editor placeholder parked at (-1000, -1000). `undefended` covers "owned complete immobile structures lacking a turret", and the placeholder qualifies -- owned, immobile, 170,000 hp, eternally bare. Ranked last by distance it was never reached; the moment the cover ring made defence WORK, every real structure got covered and the placeholder became the only bare structure left. Defence then poured turrets at an unbuildable off-map point for the rest of every rich match, walking workers to their deaths across the map and freezing the workforce the plan was waiting on.

The exclusion `find_producer` documents as load-bearing now guards `undefended` too. Three leaks, one shape: **the placeholder trap re-arms in every new consumer of "things the player owns"**, and each of today's fixes (worker priority, completion-on-sight, this) was invisible until the previous one worked. Fourth launch of the Hard baseline; the deadlock counter rides in its monitor.

## [2026-07-31] fix | leaks four and five: a gutted tree reused as frozen, and a factory refused by its own cover
Notes: The cascade continued past the placeholder. **Leak four:** a partial delete under Windows file locks left `.tree/` standing with its doctrines gone, `prepare_tree` judged existence by the directory, and ten matches failed at once on a freeze that reported success -- the tree now writes a `.complete` marker last and reuse is judged by the marker alone. **Leak five, the one that emptied every army:** the fresh Hard batch stood ZERO factories where the pre-cover-ring era stood two to four -- ten identical `landFactory` orders at one ring slot, all silently refused. The cover ring had densified the base, and `next_ring_site`'s 60-unit occupancy check cannot see a factory's footprint (>84 units): a turret 82 away passes the check and fails the engine. `RING_CLEARANCE = 130` now guards ring placements (still under the 144 closest-slot spacing); a live probe confirms `plan 8/8, army 0 -> 5, attack orders 15` where every batch match had read `army 0 -> 0`.

**Five leaks, one day, one root:** the defence siting fix made turrets actually land, and everything downstream had been calibrated -- accidentally -- against a defence that mostly failed. Worker priority, completion detection, the placeholder's eternal bareness, and the ring clearance were all pre-existing latent defects that a working defence armed simultaneously. Every fix probe-validated or batch-validated before the next layer showed. Sixth launch of the Hard baseline is the first on code where a probe has actually shown a factory standing.

## [2026-07-31] measure+build | the healthy bot wins 6/24 at Hard, and the suspect is the repair itself
Artifacts: `runs/sweeps/hard24-baseline` (24/24, zero failures, zero armyless)
Notes: The sixth launch finally measured a fully functioning champion at Hard: every match fields an army and fights, median worth 32,700 -- the highest a control has ever posted -- and **6/24 wins** where the pre-cover-ring bot won 10/12. Richer, healthier, and converting at a third of the old rate. The one deliberate behavioural change standing between the two eras is that defence WORKS now: 25-45k a match flows into turrets that four historical arms already measured as not winning matches. The old bot's blind siting was wasting those orders and thereby -- accidentally -- keeping the money and the workers on army and economy.

So the on-vs-off question is finally askable, and it is now a doctrine field: `cover 0/1` (`RW`-flag family, default on -- the lineage behaviour), preset `aa-counter-guard-nocover` one field from the champion, `rush.doctrine` set to cover 0 as an all-in should be. Launched `cover24-hard`: covered against nocover, 24 seeds, ten workers. If nocover restores the old rate, the defence ledger closes as "attempted-defence was load-bearing waste; landing it was a regression", the champion flips its flag, and the Impossible batch inherits the winner.

## [2026-07-31] research | the determinism assault: the fork is the pathfinding thread, proven in three probes
Artifacts: `runs/traces/detrep/{a,b}.ndjson`, `runs/det-{a,b,c,d}.entities`, `scripts/detprobe.py`, `runs/decompiled/com/corrodinggames/rts/gameFramework/k/o.java`
Notes: Austin: "im sure we could solve it if we really tried." Tried; solved down to the class and the line. The chain of evidence, each step one probe:

1. **The world digest** (new trace column: CRC32 over every entity's id/type/position/hp per sample) localised the fork instantly: two champion replicas of one seed are **bit-identical at frame 0** -- the hold works perfectly -- and diverge by frame 75, the first lockstep window.
2. **The observer probe** (connect, ack, never order): two runs **bit-identical for four windows**. The simulation alone is deterministic. The fork enters with OUR commands.
3. **The one-order probe** (a single identical move at sample 0): the ordered builder itself sits **0.26 world units apart by frame 75** -- it started walking about one tick earlier in one run -- with every other entity identical.

A move order triggers exactly one thing the pure simulation does not: an **asynchronous pathfinding job**. `gameFramework/k/o` is the engine: a dedicated worker thread sleeps on a monitor (`run()` waits on `x`), a request sets the job and `a()` notifies, `b()` computes and delivers -- on whatever tick the OS schedules the worker. The unit stands still until its path arrives, so path-arrival tick = wall-clock race = the whole of the nondeterminism.

**The patch plan:** rewrite `o.a()`'s body to invoke `this.b()` inline -- pathfinding computes synchronously on the requesting thread, the worker never wakes, arrival tick becomes deterministic. The agent's class patcher needs one new capability (emit a self-call instead of a bare return). Then the certification: twelve replicas, twelve identical scorecards. Payoff at stake: one seed = one answer, ~10x cheaper verdicts, bit-exact replay of any match.

## [2026-07-31] research | sparring mode scoped: the bot can host a game a human joins, and the invariants already paid for it
Artifacts: `runs/decompiled/com/corrodinggames/librocket/scripts/{Root,Multiplayer}.java`, `.game/assets/gui/battleroom.rml`
Notes: The training-target pivot's second half. The whole hosting chain sits in the **unobfuscated script surface** -- the same one that provided `-sandbox`: `Root.hostStartWithPassword(false, null)` boots the LAN server (port 5123, `preferences.ini`) and opens the battleroom; the map is set on the network engine (`bX.ay.a` = skirmish type, `ay.b` = map basename); the battleroom's Start button is `mp.multiplayerStart()`, whose server branch just sets the map path and calls **`bX.ae()`** -- no readiness gate. Poll the player list until the human joins, call `ae()`, and the existing liveness watcher, channel and planner take over.

The four portability invariants ([[multiplayer-portability-invariants]]) were held from the start and now pay out: every action already goes through the engine's command queue (no desync vector), the planner already runs free-running when `lockstepFrames=0` (a human cannot be world-held), perception already filters to the visible. Implementation remainder: the player-count field on the network engine, 1v1 team slots, a lobby timeout -- a day, not a weekend. Private LAN against a knowing human is inside the page's own social boundary.

## [2026-07-30] measure | cover24-hard: turret cover is a measured regression, the champion flips to nocover
Artifacts: `runs/sweeps/cover24-hard` (48/48, zero failures), `scripts/analyze_sweep.py`, `scripts/ledger.py`
Notes: (Stamp note: the six entries above are marked [2026-07-31] but were written 07-29/30 -- their order is correct, their dates ran a day ahead.) The on-vs-off question the healthy bot finally made askable is answered, decisively: **covered 4/24, nocover 14/24** at Hard, a 10-win margin against the >=4 standard, the largest any arm has ever posted. The mechanism is exactly the accusation: covered dropped **43** extractors to nocover's **22**, posted 28 unengageable sightings to nocover's 6, and bought its higher median worth (29,950 vs 25,050) in structures that neither held ground nor won matches. Nocover converts seven matches to outright wins with `best rival -> 0` -- the enemy economy erased -- where covered's typical end state is a fat base watching 30+ targets it never engages. The defence ledger closes: **attempted-defence was load-bearing waste, and landing it was a regression.** Cover is now off in the champion lineage (`aa-counter-guard-nocover`), and 14/24 beats the six-batch ~8-10/24 plateau the posture family had parked at -- turning defence off is the single largest one-field gain on record.

Also landed, gated at 100%: the analysis pair Austin asked for ("are we logging all these attempts... in a way that allows us to analyze it"). `analyze_sweep` prints any batch as per-match rows joined with trace-derived extractor drops plus per-arm aggregates; `ledger` flattens every scorecard ever filed into one TSV keyed batch/arm/seed. Writing the tests caught a real defect: `ledger.main` ignored its `root` parameter and always read the live record. Next: `impossible24` launches on the flipped champion -- nocover control vs rush, difficulty 3, 24 seeds.

## [2026-07-30] research+build | determinism solved: three stacked clocks, and the engine's own kill switch closes the last one
Artifacts: `runs/det-{sync,fine,pin,f1,l5,bu,full,clean}-{a,b}.*` (the probe ladder), `runs/traces/det-full-{a,b}.ndjson` (603 lines, bit-identical), gate green
Notes: The pathfinding thread was the mechanism we could SEE, not the whole mechanism. The sync-path patch landed (the agent's class patcher grew a delegation capability: `o.a()V` rewritten to invoke `b()V` inline via the Methodref `run()` already carries, `c()V` no-opped so the PathSolver thread never exists) -- and the one-order probe STILL forked. What followed was a bisection by probe resolution, each layer named by its own experiment:

1. **The frame delta is the wall clock.** `java/u.update(GameContainer,int)` stores Slick's measured delta-ms in field `t`; `render` multiplies it by 0.06 into the `deltaSpeed` every simulation tick scales by ([[engine-tick-and-clock]]). OS scheduling jitter IS simulation divergence -- only a moving unit can show it, which is why observer runs were bit-identical and pathfinding took the blame. Pinning `t` (a store-constant patcher capability, since retired) made lockstep-1 runs **bit-identical over 40 frames** -- and lockstep-5 runs still wobbled.
2. **The tick count is also the wall clock.** The probe grew an engine-clock column (`by` is the integral of the step size), and it caught the residue red-handed: two pinned runs at the same frame read c60 vs c57. Slick sometimes runs ZERO `update()` calls before a render -- and `render` zeroes `t` after consuming it -- so a render-only cycle ticks the simulation with a zero-length step at OS-chosen positions. Pinning what each update carries cannot fix how many updates precede a render.
3. **The engine ships the kill switch.** `gameFramework.l.bu`, default -1, never written by any code in the jar, read at the top of every tick: `if (bu >= 0) f2 = bu`. One reflective float write at the match's first live tick (`MatchSetup`, beside the frame zeroing; `pinDeltaMs` agent option, `PLAY_PINDELTA` make var, 3ms = 0.18 steps -- the measured 300fps container average is 3.33ms, so the pace shift is under ten percent) and every tick is the same fixed quantum of simulation whatever the container measured.

**The certification so far:** lockstep-5 one-order probes bit-identical; the full planner -- economy, expansions, combat, a Hard opponent, 600 samples at production lockstep-75 -- **traces bit-identical to the world digest (603 lines) and scorecards bit-identical (45 lines)**. The one residue is the known liveness race: runs may sit one frame of phase apart in their STAMPS while carrying identical content ([[policy-determinism]] recorded this before the assault began). The interim `u.t` bytecode pin and its patcher capability were removed once `bu` superseded them -- a re-probe on the cleaned agent confirmed content-identity survives the removal. `pinDeltaMs` stays opt-in: watch and host runs need the sim glued to wall time, and trees frozen before the option exists must launch with 0, so `PLAY_PINDELTA` defaults off until `impossible24`'s tree retires. Staged: `sweeps/certify12.txt`, twelve replicas of one seed that must file twelve identical scorecards. Payoff on the table: one seed = one answer, ~10x cheaper verdicts, bit-exact replay of any match.

## [2026-07-30] measure | impossible24, called at 20/48: nobody wins, and the loss has a shape
Artifacts: `runs/sweeps/impossible24` (19 champion + 1 rush filed when called; batch resumable)
Notes: The Impossible question, answered early because the answer stopped moving: **champion 0/19, rush 0/1** at difficulty 3. Not the old instant wipes -- the healthy nocover champion SURVIVES to the sample limit in 17 of 19 -- but strangled: it peaks at 5-6 extractors and ends at 0-2 (102 drops over 19 matches, more than five per match), median worth 5,750 against enemy scores of 100,000-185,000, unengageable sightings 813 (air overhead it cannot shoot), intercepts >10,000 (endless defence, no conversion). The 3.7x economy simply out-replaces every trade. The rush all-in confirmed the other end: 47 tanks in 3-tank waves, `engaged gone 0` -- nothing it attacked died before it did. A rush footnote for the record: on duel_lake the enemy command center is visible from frame 0, so the march verb's "nothing visible yet" premise never held and ordinary wave engagement owned the attack the whole match; the verb is a no-op on this map, the composition question was still measured.

**What this rules out:** timing (all-in at the earliest possible moment) and posture (every one-field variant already measured at Very Hard). What it leaves: mass at contact and reach. The champion feeds 3-tank waves into turret lines the same way the rush does, just later; nothing in either arm ever out-ranged a `c_turret_t1` line backed by artillery. Next levers, in order of evidence: bigger `mass` so waves arrive as armies (the wave-size ladder was tuned at Medium income, not against 3.7x), and the tech-tree page's unfielded standoff weapons ([[mechanics-tech-tree]]: c_artillery in the army mix, the 2,100cr/350-reach artillery turret upgrade for map control). The rung below (Very Hard, 1.8x) is where the win rate ladder actually needs climbing first -- Impossible is unreachable until VH stops being a coin flip.

## [2026-07-31] measure | certify12: solo determinism certified, parallel sweeps keep a residual -- and the protocol changes
Artifacts: `runs/sweeps/certify12` (12/12 filed), `runs/traces/certify12` (the fork evidence)
Notes: Twelve pinned replicas of one seed under the production sweep (10 workers): **three bit-identical, the rest forking at discrete frames** (most at sample ~344, one at ~332, one at ~222, one at sample 5) -- scalar columns identical at the fork, world digest diverged, which is a walking unit's position differing by a load-dependent tick. Every pin is confirmed active by the traces themselves: workers walk from sample one and stay digest-identical for hundreds of samples, impossible with an unpinned clock; `Math.random` and the engine generator log their seeds in all twelve. What remains is **parallel-load scheduling** reaching something unnamed -- the engine's own update loop carries a "JIT bug detected" self-check (`game/i.a`, active only in debug mode) that names deltaSpeed corruption under compilation, and JIT completion timing is exactly the kind of thing ten competing JVMs reschedule. Solo replicas remain bit-identical (proven at 600 full-planner samples, twice); the sweep residual is an open question parked with its evidence archived.

**The protocol changes anyway, on Austin's point: "do we need to run so many games if we're just losing them all?"** No. The 24-seed standard existed to average noise at decision boundaries. Falsifying a bad idea takes three seeds; the loss SHAPE (drops, worth curve, engageable gap -- exact under solo determinism) tells the iteration story; only an arm that starts winning earns the 24-seed confirmation. First screening batch under the new rule: `screen-vh9` -- nocover control vs `aa-counter-guard-arty` (remade on the nocover lineage; its 0/12 predates every defence fix and rode the refuted cover flag) vs `aa-counter-guard-mass40`, three seeds each at Very Hard, pinned. The two arms are the Impossible autopsy's surviving levers: standoff reach and mass at contact.

## [2026-07-31] measure | screening rounds one and two: artillery converts, the dose saturates, and the traces name the real enemy
Artifacts: `runs/sweeps/screen-vh9`, `runs/sweeps/screen-vh9b`, `runs/traces/screen-vh9` (the autopsy)
Notes: The 3-seed protocol's first outing, five arms over two rounds at Very Hard, all on seeds 12345/777/4242, all pinned. Round one: **arty 1/3 (8 drops), control 0/3 (17 drops), mass40 0/3** with one death and the largest unconverted army ever posted (56,450). Round two: **combo (arty+mass40) 1/3, arty2 (double dose) 0/3** -- the second tube starves the tank line and flipped the winning seed to a 6-drop strangle. One artillery per mix is the dose; it halves the drops and is the only thing that converts.

**The trace autopsy is the real product.** All three seeds run EVEN to sample 1000; the match is decided in one window around samples 1000-1500. Winning seed: the army snowballs 9 -> 32 through the window, extractors hold, enemy economy collapses within 1500 more samples. Losing seed: the army bleeds 12 -> 8 in bad trades, then one strike takes THREE extractors at sample ~1400 and the compounding does the rest. The enemy difference: the losing seed's opponent fields **eight gunships, five interceptors, helicopters and a navy** (intercepted 824 -- a match spent chasing what the army cannot catch); the winning seed's opponent never got air up. The gap is not reach or mass -- it is that the mix answers air only while a raider is on screen: hoverTank hits air and the counter tilts toward it, but with `scout 0` the tilt is sighting-scoped and evaporates between raids (the intel-memory path is scout-gated by design). Round three runs the two candidate answers on the same seeds: `arty-scout` (the counter keeps its memory) and `arty-aa` (AA turrets over the extractors, cover otherwise off -- the AA branch precedes the cover gate, so the refuted ground-turret waste stays out).

## [2026-07-31] measure | screening rounds three to five: every side-channel refuted, the composition space is exhausted, and the next lever is behavior
Artifacts: `runs/sweeps/screen-vh9{c,d,e}`
Notes: Eleven arms, thirty-three matches, five rounds, one afternoon -- the 3-seed protocol at full speed. Round three (the air answers): **scout memory 0/3, AA roof 0/3, and both LOST seed 777**, the seed plain arty wins -- defensive spend blunts the mid-game snowball that is the only converting shape ever measured. The roof did what it promised (one drop on its best seed) and it did not matter: protection without conversion is a slower strangle. Round four (the answer inside the fist): **hover-primary without arty refuted** (16 drops, rival ~100k everywhere -- artillery is load-bearing), hovarty 0/3 with the best containments yet (29k, 40k) and no conversion. Round five (the scatter hypothesis): **interception-off refuted hard** -- 0/3, drops doubled, s777 flipped from a win to a zero-extractor 125k strangle. Chasing raiders was load-bearing.

**Standing at the plateau:** `aa-counter-guard-arty` is the best doctrine measured -- half the control's drops, the only economy-kill, 1/3 at Very Hard -- and no single-field move off it improves it. The trace autopsies say why: every match is even to sample 1000 and decided by trade quality in the window after, and trade quality is not a doctrine field. The wave already focuses one target persistently ([[policy-combat]]) -- which at mass 25 is OVERKILL per volley against a spread-firing enemy -- and the artillery walks with the wave into the turret reach it out-ranges. Those are the two behavior levers the next arc owns: kill-sized target groups instead of one-target focus, and artillery that stands off at its own reach. Composition screening stops here; arty is the champion candidate pending a code-level trade-quality pass.

## [2026-07-31] build+measure | kill-sized fire groups: trades improve, kills do not, and the chaos boundary shows itself
Artifacts: `runs/sweeps/screen-vh9{f,g}`, `src/rw_bot/policy/combat.py` (engagements rewritten), gate green
Notes: The first behavior change of the micro arc. `engagements()` no longer sends the whole wave at one target: attackers deal into groups just large enough that one combined volley kills -- the engine's own damage figures, no invented constant -- held per attacker, with overflow reinforcing the nearest group. Round six (unbounded groups): the hardest seed's rival fell **114k -> 19k**, the best containment ever measured, and the winning seed's kill was gone. Round seven (groups capped at two, `MAX_OPEN_GROUPS`): best drop count ever (7), near-parity on the hardest seed (ours 24.9k vs rival 24.3k), s777 worse than unbounded. **No code state dominates per-seed** -- baseline/unbounded/capped shuffle non-monotonically across the three seeds, which is the chaos boundary: three seeds rank composition changes (whose effects are large) but not fire policies (whose effects are subtle and trajectory-coupled). Aggregate signals only: rival-sum 153k baseline, 110k unbounded, 126k capped; drops 8/8/7; wins 1/0/0. The capped code stays (best drops, keeps the fist) pending a wins-based reason to prefer another.

Also killed by data: the artillery-standoff hypothesis. Survivor compositions show artillery dying proportionally, not preferentially -- the engine already halts an attacker at its own weapon range, so the "arty walks into turret reach" story was wrong. What stands after seven rounds: against 1.8x income every even trade loses long-run, so round eight sends the raid verb (never measured with artillery or the new fire code) at the enemy's extractors -- the one target class where a trade pays 1.8x.

## [2026-07-31] build+measure | rounds eight to eleven: the equilibrium named, the saving mechanism built, and the economy finally compounds
Artifacts: `runs/sweeps/screen-vh9{h,i,j,k}`, `src/rw_bot/policy/budget.py` (withhold), gate green
Notes: Round eight (raid 3 on the leader): best aggregate of the doctrine family -- **6 drops, ahead-of-rival on two seeds** -- and the trace tails named the wall: two seeds sit in STABLE EQUILIBRIA, the enemy shaved to ~15k and rebuilding exactly as fast as we shave. Against 1.8x income every even trade loses; equilibrium is what "even" looks like from our side. Round nine (mechMissile, the dedicated 190-reach air answer behind one auto-inserted mechFactory): best economy consistency ever (4 extractors and 54/s on every seed), mediocre containment, 0/3. Round ten (upgrades claim before production): **a no-op, and the ledger said why** -- `upgrade:extractorT3 wanted 4000 of 0 available` on every ask, because a per-tick budget drains to the reserve before a 4,000-credit purchase can ever fit. Claim order cannot buy what income must accumulate toward.

**Round eleven is the structural change: `Budget.withhold`.** A refused conversion keeps its price back from every later claim -- protected ones too, deliberately, since replacing losses is protected and drains the balance to zero each tick -- so the balance climbs across ticks until the conversion fits. Measured: **six T2 and six T3 conversions bought in one match, 32,400 credits of income infrastructure, 98/s at the end -- nearly double any arm ever -- and the highest worth of the campaign (46,950).** Still 0/3: the army pauses while each conversion saves, the enemy's 1.8x compounds meanwhile, and the equilibrium moved UP on both sides rather than breaking. Pinned end to end: withholding hides credits from protected and unprotected claims alike, a refused upgrade saves rather than losing the credits to a tank, and upgrades outrank production (tests on budget, spending and the campaign loop). Round twelve runs the composite -- saving economy plus raid denial on the capped fire code -- and earns the 24-seed promotion if it converts.

## [2026-07-31] measure | round twelve refutes unconditional saving, and the arc goes to its confirmation batch
Artifacts: `runs/sweeps/screen-vh9l`, gate green (the withhold call reverted, the mechanism and its tests kept)
Notes: The composite (saving economy + raid denial + capped fire groups) posted **drops 4 -- the best defensive figure of the campaign -- 78/s income on every seed, and the worst rival scores of the whole arc (75-99k)**. The two compounding engines are opposed: every conversion's saving pause is a tick the enemy's army lives, and their untouched economy at 1.8x outruns any income we buy. Their dips collapsed (raid alone: 15.8k; composite: 4.6k). So the withhold call is reverted with its refutation written where the code was -- the mechanism stays built and tested for a future gated use -- and the arc's standing best is the round-eight configuration: **arty + raid on the capped kill-group code**.

Twelve rounds, ~40 matches, one day, every arm cheap and every refutation banked. Launched: `vh24-arc` -- candidate vs the pre-arc champion (nocover), 24 seeds at Very Hard, the number that says what the arc was worth. The next lever class after this reads out is real-time tactics: timing pushes to the enemy army's absence (intel already sees it), retreat-when-losing, and the win-mechanism constant of every converted match -- pressure that starts the shave by sample 1500.

## [2026-07-31] measure | vh24-arc: the champion flips on shape, and Very Hard's honest baseline is near zero
Artifacts: `runs/sweeps/vh24-arc` (48/48, one resumed flake)
Notes: The arc's confirmation batch, 24 seeds at Very Hard, pinned, in-batch control. **Candidate (arty+raid on the kill-group code): 1/24, zero losses, 66 drops, 107 unengageable. Control (pre-arc nocover champion): 0/24, two losses, 139 drops, 468 unengageable.** The win margin (1 vs 0) is below the >=4 standard and is recorded as such -- what flips the champion is dominance on every secondary figure: half the extractor drops, a quarter of the unengageable sightings (the arty+raid mix can fight what it sees), no losses, and the only conversion. `aa-counter-guard-arty-raid` is the working champion.

**The larger finding is the baseline itself.** Under pinned, healthy-code conditions Very Hard is a ~0-1/24 rung for this bot. The remembered 8-10/24 came from the cover-on posture family on unpinned code in the 12-seed era, and it does not transfer -- some mix of the pace pin, the healthy defence siting, and the old era's lucky unpinned trajectories inflated it. The ladder reads, honestly: Hard 14/24 (nocover, unpinned era), Very Hard ~1/24, Impossible 0/28. The next arc is real-time tactics -- timing pushes to the enemy army's absence (the intel exists), retreat-when-losing, and pressure that starts the shave by sample 1500, the one constant of every converted match ever measured.

## [2026-07-31] research | the AI's brain read at last: random targeting, the raid leash, and the community fortress
Artifacts: `wiki/pages/ai-opponent-strategy.md` (three new sections with decompile cites), `doctrines/fortress.doctrine`, `doctrines/punch.doctrine`
Notes: The mid-sweep research pass Austin asked for, primary source first. Three findings that change the tactics arc. (1) **The AI's attack targeting is `Math.random()` over every targetable unit we own** -- no scoring, no distance, and no fog term anywhere in the eligibility chain, so it targets units it has never seen and our placement IS the distribution of its attacks (`a.java:1726`, `j()` at `:1881`, base `cg()` always true). (2) **Any intruder in one of its base zones recalls a group to "defending base" for a 500-tick timer, before the attack branch runs** (`g.java:423-444`) -- the raid party is a leash on its armies, which is consistent with the raid arm's best-shape record. (3) **The community answer to Impossible is turtle-and-counter with repair bays** -- turrets plus repair bays plus artillery at a choke, counter when a 17-second-bound group breaks on the wall. Our cover24 turtle had no repair; unhealed turrets are turrets bought twice. `repairbay` is builder-buildable and plan-expressible, so `fortress.doctrine` takes the system whole. Screens queued behind the wallet-first round: fortress at VH, punch at Impossible.

## [2026-07-31] build+measure | the riposte ships, the creep dies its first death, and the tech ceiling is finally named
Artifacts: `runs/sweeps/screen-imp3{,b}`, `src/rw_bot/policy/{creep,dispatch,combat}.py`, `doctrines/{punch,creeper,fortress}.doctrine`, gate green
Notes: The assault continues on Austin's directive. **Punch refuted at Impossible** (0/3, all dead -- a 25-unit fist dents 3-5k of a 3.7x economy). **Creep v1 refuted at Impossible, expensively and informatively**: the walk worked -- 164 turrets ordered, 82,000 credits -- and every one died before the next stood, because one-at-a-time turret building is the trickle anti-pattern in masonry. The human creep works in clusters with repair and escort; v2 knows what to be. **The riposte ships**: Austin described his own meta ("stockpile, let them burn their army on my defences, then push") and it is now a doctrine flag -- the WaveController arms on the intruder-present-to-absent edge and the next muster releases the whole reserve below the ladder's rung, anti-trickle floor intact, consumed-or-missed semantics. `fortress.doctrine` is that playstyle whole: cover + AA + repair bay, mass 40, raid 3 to leash their groups, riposte 1. Screening at Very Hard.

**The deepest finding answers "are we using good units?": no -- all tier one, and the ceiling is a missing wire verb.** The land factory's 2,000-credit T2 upgrade is not a type conversion: decompiled `units/d/n.java` flips a flag on the same building (`L()` returns null) and unlocks the heavy roster. Our upgrade machinery matches conversions by target type, so it cannot see it; our command vocabulary (produce-by-type, build-at) cannot invoke a no-type action. The option stream already carries them -- `produces:""`, an action index, available -- so one new verb (`ability`, by action index) unlocks factory T2 and with it heavyTank: 800 credits, 600 hit points, reach 160, and it shoots aircraft. Roughly double the combat value per credit of anything we field, plus the air answer inside the fist. That is the tech arc, and it is next.

## [2026-07-31] build+measure | the ability verb ships end to end, and the fortress fails at the wallet, not the wall
Artifacts: `runs/techprobe.out`, `runs/sweeps/screen-vh9n`, `src/rw_bot/policy/{spending,campaign}.py`, gate green
Notes: **The tech channel is plumbed and the first live probe named its own blocker.** The full path -- option stream carrying the no-type T2 unlock, `unlock_tech` finding the selector, the `ability` wire verb, the agent's `actionBySelector` dispatch -- ran a real match and the ledger read `tech:landFactory asked 37 got 0 -- wanted 2000 of 0 available`. Every gate opened except the wallet: a per-tick budget drains to the reserve before 2,000 accumulates, the same arithmetic that killed claim-order for the T3 conversion in round ten. So the withhold gets its gated use, the one the round-twelve refutation reserved it for: **a refused unlock saves toward itself** -- bounded, once per factory, unlike the unbounded income ladder that lost -- and tech claims before income conversions, because the T2 extractor funds at a 2,300 balance and would snipe every accrual just short of the unlock's 2,900. Pinned: a refused unlock hides its price from later spenders, protected ones included.

**Fortress refuted at Very Hard, and not where expected.** 0/3, zero losses -- the wall holds -- but the horde never forms: on s12345 army value sat at 500 all match, composition `none`, the opening plan held the only worker for 1,154 samples while the fortress goals (turrets, AA, repair bay) ate the whole early economy and VH raids ate the extractors (income 18/s at the end). The riposte never armed because there was nothing to release. Austin's meta fails at the wallet, not the wall: the counter-punch needs the champion's economy underneath it, which is an argument for riposte-on-champion rather than fortress-as-identity.

## [2026-07-31] build+measure | the unlock buys rally points until the wire learns what actions cost
Artifacts: `runs/techprobe{2,3,4}.out`, `runs/sweeps/screen-vh9o`, `wiki/sources/m6-wire/world-sample.ndjson` (recaptured), gate green
Notes: Three probes, each naming the next fault. Probe two proved the withhold accrues -- every later spender read `of 0 available` while the unlock saved -- and simply ran out of match. Probe three, at 1,500 samples, fired the verb live at last: `tech:landFactory got 4, spent 8000`, four ability orders dispatched and accepted, the army still growing while it saved (16,400 vs 5,400 without). Probe four then bought the humbling discovery: **no heavyTank ever rolled out, because the "unlock" was the rally point.** Decompiled, the two are identical on the wire -- no type, not placed, makes nothing -- and `unlock_tech` took the first match (`units/a/o.java`: `c_1`, setRally; `units/d/n.java`: the real T2 upgrade). The reading that separates them is the action's own cost accessor, abstract on the action base class: a rally answers zero, the upgrade answers its tier price. So the option stream now carries **price** -- the engine's own charge, per action -- the selector wants the no-type action that costs something, and the claim is for the wire's figure rather than a catalogue guess. The archived world capture was regenerated through `make wire-capture` (the fixture's own rule: never edit it by hand), and the fresh capture shows the discriminator live: the command centre's rally at price 0 beside its builder at 500. The `heavies` doctrine field also shipped: composition entries outside the plan goals -- the build tree would insert an experimental factory for a heavy goal, where production's live-options path simply stays quiet until the tier opens. Probe five runs the corrected selector.

**Riposte-champ refuted at Very Hard** (`screen-vh9o`, 0/3, zero losses): the counter-punch trigger on the champion economy did not convert either -- seed 12345's economy still dies (18/s), and 1,345 intercepts say the reserve spent the match chasing raiders rather than massing behind the wall. Both riposte housings (fortress, champion) now measured and dead at VH; the verb keeps its tests and waits for a shape that funds it.

## [2026-07-31] build | the index that was not a selector: the ability verb moves to the engine's own keys
Artifacts: `runs/techprobe5.out`, `agent/src/rwbot/agent/{BuildOptions,Orders,CommandRecord,CommandChannel}.java`, `src/rw_bot/wire/{command,state,codec}.py`, `wiki/sources/m6-wire/world-sample.ndjson` (recaptured again), gate green
Notes: Probe five selected the right option -- the priced no-type unlock -- and still fired the rally, which named the deeper fault: **the engine's per-action index is not a selector.** Every action on a unit answers the same figure (the capture shows a command centre's rally, builder and scout all reporting `1`), so "the action at index N" always resolved to the first action on the list. The engine's own executor never uses that index; it resolves commands by each action's interned key (`c_1`, `u_builder`, ...), so the wire now does the same: the option stream publishes `key`, the ability order carries `{"kind":"ability","unit_id":N,"key":"..."}`, and the agent looks the action up by the string the listing published. The retired `action` field is gone from both streams -- a field that looks like a selector and is not one is a trap that already bit ([[mechanics-build-actions]]). Probe six runs the key dispatch, with `heavies heavyTank,heavyTank` waiting in the composition.
