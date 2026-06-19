# TankpitBot

Automated bot client for the Tankpit.com browser game.

## Wiki

**Read `wiki/index.md` at the start of every session.** The wiki is the single source of truth for game mechanics, wire protocol, combat strategy, and architecture decisions. Navigate from index -> hub -> content page as needed.

Do NOT rely on `.claude/projects/.../memory/` files for game knowledge — the wiki supersedes them. Memory files may still hold AI behavior preferences but all game/protocol/architecture facts live in the wiki.

After a live run, update wiki pages with new findings and log the operation in `wiki/log.md`.

## Build

```bash
make check    # guard + ruff + mypy + tests + coverage (100% required)
make run      # 5-min timed bot session + scorecard
make analyze  # issue report on latest run
```

## Coding standards

See `wiki/pages/coding-standards.md` for the full list. Summary:
- No Any, cast, type: ignore, TYPE_CHECKING, .pyi, noqa
- No mocks, no monkey-patching — use `_test_hooks` DI
- No back-compat shims, no wrappers, no fallbacks, no legacy code
- 100% test coverage, no weak assertions
- Files under 400 lines where possible
