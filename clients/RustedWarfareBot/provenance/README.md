# Provenance

What a cluster run is made of, and the records staging is held against. Read
by a person, and by `hpc3-stage --expect-from`.

Two trees are staged, and they are pinned differently because they come from
different places: the game comes from Steam depots, the payload comes from
this repository.

## The game tree — `python -m scripts.stage_tree`

- **`linux-tree.json`** — the document. Every file of the tree, classified
  against Steam's own installed copy: which are byte-identical at the same
  path, which are the same file under the other platform's runtime directory,
  which are copies this project made under another name, and which have no
  second copy anywhere and are stated as assembled from the Linux depot rather
  than claimed to be checked. Carries the app, the build, and both depots
  pinned by manifest GID.
- **`linux-tree-digests.txt`** — the published record. A `sha256sum --text`
  listing of the whole tree, headed by the depot pins and the tally, with the
  archive's own digest on the first data line. `hpc3-stage` requires
  `--expect-from` precisely because a manifest emitted beside its own files
  always agrees with them; this file is written by a different act and is what
  makes that check real.
- **`stage-linux-tree.json`** — the stage manifest, emitted by the same run so
  no digest is ever retyped.

Regenerate with:

```
python -m scripts.stage_tree \
  --tree <assembled linux tree> \
  --reference "C:\Program Files (x86)\Steam\steamapps\common\Rusted Warfare" \
  --build-id 9902063 \
  --content-manifest 9090535937117498741 \
  --linux-manifest 223921525878913700 \
  --archive <out>/rusted-warfare-linux.tar \
  --out provenance/linux-tree.json \
  --digests provenance/linux-tree-digests.txt \
  --manifest provenance/stage-linux-tree.json \
  --destination /pub/wagnera3/rusted/staging
```

## The payload tree — `python -m scripts.stage_payload`

Everything a match imports or READS at launch: `src/rw_bot`, `scripts`,
`doctrines`, `sweeps`, the compiled agent jar, and the two registry dumps the
planner reads (`printunits.log`, `type-flags.ndjson`). Frozen BEFORE
submission and staged, because `prepare_tree` copies from repository-relative
paths and a compute node has no repository — a freeze there would report
success having copied nothing. The agent jar makes it unavoidable: the Linux
depot ships a JRE with no compiler, so no node can build it.

**The two dumps were the last to join, and they were left out for a reason
that read as sound.** A `-printunits` dump is an artifact of the game build
rather than code that changes between batches, so it looked like something a
freeze had no business carrying. But the question a frozen tree answers is
not "does this change" — it is "can a match READ this where the match runs",
and the launcher's default path for both is repository-relative. The first
cluster member to reach the planner died on

```
FileNotFoundError: [Errno 2] No such file or directory: 'wiki/sources/m0-probe/printunits.log'
```

having already patched the engine, seeded it, and held the world at its first
frame. Job 55663569, 2026-08-30.

- **`payload-tree.json`** — the document, pinned by `git_commit`. Given rather
  than read from `git`, because the tree frozen is the WORKING one and a
  command that stamped HEAD onto a dirty tree would print a reassuring lie.
- **`payload-tree-digests.txt`** — the same published-record shape.
- **`stage-payload-tree.json`** — the stage manifest, emitted by the same run.

Two freezes of one source produce identical digests: bytecode caches are
excluded (140 of the first payload's 408 files were `.pyc`, and a `.pyc`
embeds the source's timestamp), and the archive pins member order, timestamps,
owner and modes.

**Neither manifest is hand-written any more, and both were.** The claim above
them used to be that they were "composed from the document so no digest is
ever retyped", which was the intention rather than the truth: they were typed.
The first re-freeze proved the difference — the payload grew from 407 entries
to 410, its archive changed length and digest, and the manifest went on naming
the previous ones. Staging against a stale manifest fails the local digest
check in the good case and stages the wrong tree in the bad one, verifying
happily on both sides. `rw_bot.stage_record` composes them now, from the
archive the same run just wrote.

Run `make agent` first. The jar is not in the repository, and the freeze
refuses a tree without it rather than staging one no match can use.

```
python -m scripts.stage_payload \
  --tree <scratch>/rw-payload \
  --commit $(git rev-parse HEAD) \
  --archive <out>/rw-payload.tar \
  --out provenance/payload-tree.json \
  --digests provenance/payload-tree-digests.txt \
  --manifest provenance/stage-payload-tree.json \
  --destination /pub/wagnera3/rusted/staging
```

`--commit` is a CLAIM, not a reading. The tree frozen is the WORKING one, so
passing `git rev-parse HEAD` while the tree carries uncommitted edits stamps a
commit whose contents are not what was staged. That is deliberate — the
alternative is a command that reads HEAD itself and prints a reassuring lie —
but it means a freeze taken over a dirty tree should be followed by a commit,
and the same applies to `git_commit` in `tools/hpc3/specs/rusted-image.json`.

## The image — `tools/hpc3/specs/rusted-image.json`

Not in this directory, because it is an hpc3 artifact, but it is the third
pinned thing a run depends on. Its system layer was determined by running the
real engine in the real base image under Docker, not by reading documentation:
`xauth`, `libxtst6` and `x11-xserver-utils` are each required, and each was
found by a launch that failed without it. LWJGL 2 enumerates display modes by
shelling out to the `xrandr` BINARY, which is why the third one matters.

## Where a member's game clone goes

Not in this directory, but it is the third thing a submission has to be told,
and it was the second defect the first cluster run found. A clone's name is
relative — `.game-w1` — and `sbatch` sets no working directory, so a member
resolves it against the directory it was SUBMITTED from: one directory per
project, on `/pub`, shared by every node. Every member of the batch therefore
aimed at one clone. The first began copying 307 MB into it, and the second,
nine seconds later on another node, saw the directory already there, skipped
the copy it thought was finished, and died listing
`.game-w1/assets/maps/skirmish` one second in. Jobs 55663569 and 55663571;
the 72 MB the first got through was still sitting in `rusted/scripts/`
afterwards, which is how the mechanism was confirmed rather than guessed.

A member is given `--clones <dir>` now, and takes its ordinal from its own
position in the job file, so the clone directory, the channel port and the X
display are all leases nothing else in the batch holds.

## Reading the panel's verdict

`scripts.replicate` is NOT in the wheel — `pyproject` packages `rw_bot` from
`src` and nothing else — so it cannot run inside the image where the traces
are written. It takes a traces root for exactly this reason: bring the traces
home and read the verdict here.

```
scp -r hpc3:/pub/wagnera3/rusted/runs/traces/replicate runs/traces/
python -m scripts.replicate replicate runs/traces
```

## Neither archive is committed

308 MB and 2 MB respectively, and both are reproducible from their trees by
construction: `rw_bot.tree_archive` pins the member order, timestamps, owner
and modes, so two packs of one tree are the same file byte for byte. Each
digest is in the documents above, which is what lets a repack be compared
rather than trusted.

## Why this directory exists rather than `runs/`

The repository ignores `**/runs/`. The 264 files tracked under
`tools/hpc3/runs/` predate that rule and survive only because git ignores
untracked files, not tracked ones — so anything NEW written there is silently
never committed. A provenance record that vanishes on a fresh clone is worse
than none, because the tree it describes would still be staged and the
document proving what it was would be gone.

## What this does NOT claim

The 257 `assembled-runtime` files — the Linux half of the bundled JRE and the
native `.so` libraries — have no second copy on this machine to be checked
against. The depot download they came from was consumed. The document names
the depot and the manifest they are declared to come from and says they were
assembled, which is the honest form of a claim nothing here can verify.
