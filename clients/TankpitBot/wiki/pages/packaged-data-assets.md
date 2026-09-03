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
  - "pyproject.toml"
  - "Dockerfile"
source_git_blobs:
  "src/tankpit_bot/resources.py": "a4ba016bd69d2a6ffcff342d4887295525367162"
  "src/tankpit_bot/protocol/codec.py": "b69f89bd1bf4e550a56f48a7761f91acbf35f9e1"
  "src/tankpit_bot/sniffer/world_service.py": "48c3ff79c77bfea915869be662621d6f4b0ee492"
  "src/tankpit_bot/sim/run_boot.py": "d28a3b5cc5f86b77fb04f4095696b389e86bb713"
  "pyproject.toml": "99aa1e0b69ac61fe246e3a78735e3aaa4acb4649"
  "Dockerfile": "e41a1035df86e034f28ae521db5a2124908861ae"
provenance:
  - "HPC3 job 55715554 failed on field01_r.gif not found, 2026-09-03"
  - "HPC3 job 55715564 failed on xor_static_key.txt missing despite the file being staged beside the GIF, 2026-09-03"
  - "HPC3 job 55715577 completed only after TANKPIT_XOR_KEY_FILE was passed per-run, 2026-09-03"
fact_checked: "2026-09-03"
confidence: high
hubs: [architecture]
---

# Packaged data assets: the wheel carries the key and the minimaps

*Established 2026-09-03, after one packaging defect cost three cluster
submissions and produced two independent workarounds.*

## The defect

`tankpit_bot` read two families of data files that the distribution did not
contain. `pyproject.toml` shipped only `py.typed`; the assets sat at the
repository root, outside `src/`.

| asset | how it was found | what happened after `pip install` |
|---|---|---|
| `xor_static_key.txt` | `Path(__file__).parent.parent.parent.parent` | four parents up is site-packages — nothing there |
| `field*_r.gif` (45) | bare relative filename | resolved against whatever directory the process started in |

Neither survives an install. So every consumer rebuilt the data environment
by hand, and **two independent workarounds grew for one defect**: the fleet
container copied the key in and named it with `TANKPIT_XOR_KEY_FILE`, and a
cluster job staged forty-six files beside itself, ran from that directory,
and passed the same variable on every submission.

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
defect that could not be discovered in a single pass.

## The fix, and what was deleted rather than kept

The assets live in `tankpit_bot.data` and ship in the wheel.
`tankpit_bot.resources` is the one owner of "where is this asset", addressing
them through `importlib.resources`, so the answer is identical under a
checkout, a pip install, a container and a cluster image.

**Deleted, not kept beside the fix:**

- the checkout-relative constant
- the `TANKPIT_XOR_KEY_FILE` override
- `world_service._find_field_gif`'s CWD candidate list
- the `Dockerfile`'s `COPY` and `ENV` for both families
- the three tests that pinned the override's behaviour

An environment override is a second answer to a question that must have one,
and it is precisely what let the defect survive: with the escape hatch in
place, every consumer could paper over the packaging and none had to fix it.

`static_key_file_path` moved from `protocol.codec` to `resources` with the
rest. Locating a resource is not the codec's job, and a re-export left in
`codec` would be a shim, so all 23 importing modules were repointed instead.

## Two distinctions the module holds deliberately

**`field_gif_path` returns `None`; `require_asset` raises.** A field the
*server* names may honestly ship no minimap — the session runs without
terrain, which the world service already handles. A file the caller
*already names* is a broken install. One error for both would have hidden
which of the two happened.

**`field42-r.gif` is dead weight and the resolver no longer pretends
otherwise.** It is byte-identical to `field42_r.gif` and no lookup can reach
it past its underscore sibling, so resolving both spellings was a branch
nothing could execute. The branch is gone; the duplicate asset is left in
place, since deleting shipped game data is the operator's call and nothing
depends on it now.

## What this buys

A distribution that carries its own data works under pip, Docker, apptainer
and a checkout with no per-consumer ritual. It also means **the hpc3 image
contract needs no `assets` concept** — an earlier proposal to bake the files
into the image definition would have been a third workaround at the wrong
layer.

[^1]: Job ids and failure modes from `/pub/wagnera3/tankpit/logs/`, 2026-09-03. `55715564`'s traceback ends in `XorStaticKeyUnavailableError` raised from `capture/xor.py::require_static_key` after the GIF had already loaded and 5,639 mines had been seeded — the run got far enough to prove the terrain fix worked and the key fix did not exist.
