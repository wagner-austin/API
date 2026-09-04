---
title: Packaged Data Assets
tags: [architecture, decisions, packaging, distribution, reproducibility]
related:
  - "[[sim-world-parameterization]]"
  - "[[feature-corpus-provenance]]"
  - "[[coding-standards]]"
source_paths:
  - "src/tankpit_bot/resources.py"
  - "src/tankpit_bot/protocol/codec.py"
  - "src/tankpit_bot/sniffer/world_service.py"
  - "src/tankpit_bot/sim/run_boot.py"
  - "tests/test_resources.py"
  - "scripts/download_fields.py"
  - "pyproject.toml"
  - "Dockerfile"
source_git_blobs:
  "src/tankpit_bot/resources.py": "97a6416f44ecfdc88f72294ae34e1828d3e4ae2b"
  "src/tankpit_bot/protocol/codec.py": "b69f89bd1bf4e550a56f48a7761f91acbf35f9e1"
  "src/tankpit_bot/sniffer/world_service.py": "48c3ff79c77bfea915869be662621d6f4b0ee492"
  "src/tankpit_bot/sim/run_boot.py": "d28a3b5cc5f86b77fb04f4095696b389e86bb713"
  "tests/test_resources.py": "01f6d2dbedc0608bac6752ec2427554e88a9b4fa"
  "scripts/download_fields.py": "267d3f5ddb0c22e4a2987aff1edbd5f87644f013"
  "pyproject.toml": "99aa1e0b69ac61fe246e3a78735e3aaa4acb4649"
  "Dockerfile": "e41a1035df86e034f28ae521db5a2124908861ae"
provenance:
  - "HPC3 job 55715554 failed on field01_r.gif not found, 2026-09-03"
  - "HPC3 job 55715564 failed on xor_static_key.txt missing despite the file being staged beside the GIF, 2026-09-03"
  - "HPC3 job 55715577 completed only after TANKPIT_XOR_KEY_FILE was passed per-run, 2026-09-03"
  - "field42-r.gif and field42_r.gif measured byte-identical (sha256 73c698d581ac8d125d5dcec06211e967ab858cb8aa23d8ea1e82dbf8b24b3d2b), duplicate deleted 2026-09-03"
  - "Both halves of the spelling assertion exercised against a reintroduced duplicate and against a rename, 2026-09-03: the count catches an added file, the suffix check catches a renamed one"
  - "git notes on 12717125 and bccf5afa recorded this split until 2026-09-03, when all five notes in the repository were deleted and their contents migrated into the wikis; refs/notes/commits had never been pushed"
fact_checked: "2026-09-03"
confidence: high
hubs: [architecture]
---

# Packaged data assets: the wheel carries the key and the minimaps

*Established 2026-09-03, after one packaging defect cost three cluster
submissions and produced two independent workarounds.*[^1]

## The defect

`tankpit_bot` read two families of data files that the distribution did not
contain. `pyproject.toml` shipped only `py.typed`; the assets sat at the
repository root, outside `src/`.[^6]

| asset | how it was found | what happened after `pip install` |
|---|---|---|
| `xor_static_key.txt` | `Path(__file__).parent.parent.parent.parent` | four parents up is site-packages — nothing there |
| `field*_r.gif` (45 then, 44 now) | bare relative filename | resolved against whatever directory the process started in |

Neither survives an install. So every consumer rebuilt the data environment
by hand, and **two independent workarounds grew for one defect**: the fleet
container copied the key in and named it with `TANKPIT_XOR_KEY_FILE`, and a
cluster job staged forty-six files beside itself, ran from that directory,
and passed the same variable on every submission.[^7]

## The two failures are the evidence, and they failed differently

Three HPC3 submissions on 2026-09-03:[^1]

1. **`55715554`** — died on `field01_r.gif not found`. Fixed by staging the
   file and running from its directory, because the GIF resolves via CWD.
2. **`55715564`** — died on `xor_static_key.txt missing` **even though the
   key had just been staged beside the GIF**. The key does not resolve via
   CWD at all; the fix for the first was no fix for the second.
3. **`55715577`** — completed, but only with `TANKPIT_XOR_KEY_FILE` passed
   per run.

That second failure is the one worth keeping. Two assets, two different
resolution mechanisms, one of them invisible until the other was fixed — a
defect that could not be discovered in a single pass.[^1]

## The fix, and what was deleted rather than kept

The assets live in `tankpit_bot.data` and ship in the wheel.
`tankpit_bot.resources` is the one owner of "where is this asset", addressing
them through `importlib.resources`, so the answer is identical under a
checkout, a pip install, a container and a cluster image.[^8]

**Deleted, not kept beside the fix:**[^9]

- the checkout-relative constant
- the `TANKPIT_XOR_KEY_FILE` override
- `world_service._find_field_gif`'s CWD candidate list
- the `Dockerfile`'s `COPY` and `ENV` for both families
- the three tests that pinned the override's behaviour

An environment override is a second answer to a question that must have one,
and it is precisely what let the defect survive: with the escape hatch in
place, every consumer could paper over the packaging and none had to fix it.[^9]

`static_key_file_path` moved from `protocol.codec` to `resources` with the
rest. Locating a resource is not the codec's job, and a re-export left in
`codec` would be a shim, so all 23 importing modules were repointed instead.[^10]

## Two distinctions the module holds deliberately

**`field_gif_path` returns `None`; `require_asset` raises.** A field the
*server* names may honestly ship no minimap — the session runs without
terrain, which the world service already handles. A file the caller
*already names* is a broken install. One error for both would have hidden
which of the two happened.[^11]

**`field42-r.gif` was dead weight, and is gone.** It was byte-identical to
`field42_r.gif` — both sha256 `73c698d581ac8d12…` — and no lookup could reach
it past its underscore sibling, so resolving both spellings was a branch
nothing could execute.[^3] The branch went first; the file followed on
2026-09-03.

It is worth naming which of the two was the original, because the intuition
runs backwards. The **hyphen** spelling came first, added 2026-01-12 as the
single minimap the terrain work started from.[^2] August's bulk import of the
remaining forty-three standardised on the underscore and produced a second
copy of field42 under the new spelling, leaving the January original stranded
as an unreachable twin — a duplicate created by a naming convention arriving
after the file it renamed.

The deletion is held by a test rather than by this paragraph.
`test_every_shipped_minimap_is_reachable_by_its_server_name` used to *skip* a
file that did not end in `_r.gif`; it now fails on one. Both halves were
exercised against a deliberately broken tree rather than assumed: the count
assertion catches a file that is *added*, and the suffix assertion catches one
that is *renamed*, which the count alone cannot see.[^4] A second spelling
cannot return quietly, because the only thing that made the first one
survivable was that nothing checked.

The other way a duplicate could return is the downloader, and it cannot:
`scripts/download_fields.py` writes `f"{field_name}_r.gif"` and has no other
spelling in it.[^5] So the set has one producer and one shape, and the test
holds the shape whether or not the producer is what filled the directory.

## This change is split across two commits, and neither is comprehensible alone

`git log` on any minimap, or on the static key, lands on a commit about a
Model-Trainer cartridge strategy. That is not a mistake in the reader's
search — it is where the files actually are.[^13]

| commit | holds | subject says |
|---|---|---|
| `12717125` | the 46 file renames — 45 minimaps and the key | a Model-Trainer cartridge strategy |
| `bccf5afa` | the code that addresses them — `resources.py`, the deleted override, the deleted CWD list, 23 repointed callers | "the wheel carries the key and the minimaps" |

Two sessions were committing in one working tree twelve minutes apart. A
shared index staged the renames under the first session's pathspec, so the
bytes left in their commit and the code left in ours — and `bccf5afa`'s
subject overclaims on its own terms, since it names minimaps it does not
contain.[^14]

This was recorded as a `git notes` annotation on both commits, following a
convention three earlier commits already used. **Those notes are gone, and
the table above is why they could be.**[^15]

`refs/notes/commits` was never pushed to `origin`, so every note in this
repository — these two and three from July — was visible only on the machine
that wrote it, and no clone ever saw one. That is not a record; it is a
private annotation that looks like one. All five were migrated into the wikis
and deleted on 2026-09-03, on the reasoning that two systems where one is
undistributed is worse than one system that is. The July three moved to
`clients/RustedWarfareBot/wiki/log.md`.

What is genuinely lost is the point-of-use prompt: `git log` on a minimap no
longer explains itself, and a reader has to reach this page instead. That is
the trade, made deliberately.

## What this buys

A distribution that carries its own data works under pip, Docker, apptainer
and a checkout with no per-consumer ritual. It also means **the hpc3 image
contract needs no `assets` concept** — an earlier proposal to bake the files
into the image definition would have been a third workaround at the wrong
layer.[^12]

[^1]: Job ids and failure modes from `/pub/wagnera3/tankpit/logs/`, 2026-09-03. `55715564`'s traceback ends in `XorStaticKeyUnavailableError` raised from `capture/xor.py::require_static_key` after the GIF had already loaded and 5,639 mines had been seeded — the run got far enough to prove the terrain fix worked and the key fix did not exist.

[^2]: `git log --diff-filter=A --follow` on both spellings, read 2026-09-03. `0b17ee63` (2026-01-12) adds `clients/TankpitBot/field42-r.gif` alone, subject "Add field42-r.gif minimap for ASCII terrain rendering"; `54a5e9f6` (2026-08-05, "the remaining forty-three field minimaps") adds `field42_r.gif` alongside them. Both files moved into `src/tankpit_bot/data/` in `12717125`; why forty-six asset renames sit under a Model-Trainer subject is explained in this page's own section on the split, which is where that record now lives.

[^3]: `sha256sum` and `cmp` on both files, 2026-09-03: each is `73c698d581ac8d125d5dcec06211e967ab858cb8aa23d8ea1e82dbf8b24b3d2b`, and `cmp` exits 0. The unreachability is structural rather than measured: `src/tankpit_bot/resources.py:105` builds the candidate as `data_directory() / f"{field_image.removesuffix('.gif')}{FIELD_GIF_SUFFIX}"`, so with `FIELD_GIF_SUFFIX == "_r.gif"` no input can produce the hyphen spelling.
[^4]: Verified 2026-09-03 by breaking the tree twice. Copying `field42_r.gif` back to `field42-r.gif` fails at `tests/test_resources.py:88` on `assert 45 == 44`; *renaming* it — which leaves the count at 44 — passes line 88 and falls through to `tests/test_resources.py:92`, `AssertionError: field42-r.gif ships but no lookup can reach it`. The second case is the one the old `continue` silently permitted.
[^5]: `scripts/download_fields.py`, `download_field_gifs`: `out_path = resolved_dir / f"{field_name}_r.gif"`, the only GIF filename the module constructs.

[^6]: The two lookups as they stood before commit `bccf5afa`: the key was read from `Path(__file__).parent.parent.parent.parent`, four levels above `protocol/codec.py`, which is the repository root from a checkout and `site-packages` from an install; the GIFs were resolved from a bare relative filename in `sniffer/world_service.py::_find_field_gif`. `pyproject.toml:14-18` now ships `src/tankpit_bot/data/*.gif` and `src/tankpit_bot/data/xor_static_key.txt` beside `py.typed`, which previously stood alone.
[^7]: The container workaround is visible in `Dockerfile:121-128`, which now records the two `COPY` lines and the `ENV TANKPIT_XOR_KEY_FILE` it used to carry and no longer does. The cluster workaround is jobs `55715554`/`55715564`/`55715577`, whose submissions staged 46 files into the working directory and passed the variable per run.
[^8]: `src/tankpit_bot/resources.py:63`, `data_directory`, returns `Path(str(files(DATA_PACKAGE)))` with `DATA_PACKAGE = "tankpit_bot.data"` at line 33 — one anchor on the installed package rather than on the caller's working directory or the module's parents.
[^9]: All five deletions are in commit `bccf5afa`, which removes `_CHECKOUT_STATIC_KEY_PATH` and the `TANKPIT_XOR_KEY_FILE` branch from `protocol/codec.py` (-27 lines), the CWD candidate list from `sniffer/world_service.py`, both `COPY`/`ENV` pairs from the `Dockerfile` (-23 lines), and 45 lines from `tests/test_codec.py` — the three tests that pinned the override's behaviour.
[^10]: Commit `bccf5afa` touches 37 files; the import repoint accounts for most of them, and `protocol/codec.py` retains no re-export of the moved function.
[^11]: `src/tankpit_bot/resources.py:89` (`field_gif_path`, returning `Path | None`) and `:111` (`require_asset`, raising `BundledAssetMissingError`). The split is pinned by `test_an_unknown_field_resolves_to_none_rather_than_raising` and `test_a_named_asset_that_does_not_ship_is_refused`.
[^12]: Verified 2026-09-03 by building the wheel and reading it: `tankpit_bot-0.1.0-py3-none-any.whl` carries 44 files under `tankpit_bot/data/*.gif` plus `tankpit_bot/data/xor_static_key.txt`, with no hyphen spelling present. The hpc3 `ImageSpec` (`tools/hpc3/src/hpc3/contracts/image_spec.py:170-182`) declares no assets field.
[^13]: `git log --follow` on `src/tankpit_bot/data/field01_r.gif` and on `xor_static_key.txt`, read 2026-09-03: both list `12717125` as their most recent commit. `git show --name-status 12717125` reports 46 `R100` entries under `clients/` alongside 12 additions and 62 modifications under `services/` and `libs/`. `git ls-remote origin 'refs/notes/*'` returns nothing.
[^14]: Commit timestamps, read 2026-09-03: `12717125` at 01:41:28 -0700, `bccf5afa` at 01:53:39 -0700. `git show --stat bccf5afa` lists 37 files and no `.gif` among them, against a subject reading "TankpitBot ships its own data: the wheel carries the key and the minimaps".
[^15]: `git notes list` returned five entries when read on 2026-09-03 — `12717125` and `bccf5afa` from this change, plus `52379073`, `63ed06e3` and `e0f5ff3a` from 2026-07-26, all three opening with the same line, "Also contains, from a concurrent session and unmentioned in the subject:". All five were removed the same day after their contents were verified present in the wikis; `git notes list` now returns nothing.
