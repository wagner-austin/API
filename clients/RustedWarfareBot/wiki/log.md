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
