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
