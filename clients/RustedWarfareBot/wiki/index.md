# RustedWarfareBot Wiki

**Read this first.** 5 topic hubs, 32 content pages. Follow the hub link for your topic; each hub lists its pages with one-line descriptions.

Pinned game build: **Rusted Warfare 1.15 (code 176, build #28)**, working copy at `.game/`. Claims about engine internals are only valid for this build — see `SCHEMA.md` on `game_version` pinning.

## Hubs

[Engine Internals](hubs/engine-internals.md) -- the obfuscated JVM: class-name mapping, engine objects, the script surface, what survived ProGuard (16 pages)
[Headless Harness](hubs/headless-harness.md) -- running the game without a display: CLI flags, boot behaviour, working copy, run artifacts (3 pages)
[Game Mechanics](hubs/game-mechanics.md) -- the RTS itself: unit stats, economy, build tree, terrain and movement layers, fog (13 pages)
[Bot Architecture](hubs/bot-architecture.md) -- perception, planner, dispatch, coding standards, the contracts that keep the bot honest (22 pages)
[Multiplayer](hubs/multiplayer.md) -- lockstep model, command relay, desync, third-party servers, what SP work must preserve (1 page)

## How this works

**Three tiers:** this index (read every session) → hub pages (read when topic matches) → content pages (read when you need the facts). A content page can be linked from multiple hubs.

**Adding new content:** create the content page in `pages/`, add an inclusion link from the relevant hub(s), bump page counts here. If the topic needs a new hub, create it in `hubs/` and add one line above.

**The rules:** see `SCHEMA.md` — atomicity, frontmatter, `game_version` pinning, citations, hub-link discipline.

**After a live run:** archive the artifacts under `runs/`, update the affected pages, and log the operation in `log.md`.
