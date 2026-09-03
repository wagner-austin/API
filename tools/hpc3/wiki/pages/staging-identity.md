---
title: Staged bytes are held to a record written by a different act
tags: [identity, staging]
hubs: [images-and-staging]
related: ["[[image-build-flow]]", "[[known-answers]]"]
source_paths:
  - "src/hpc3/contracts/stage.py"
  - "src/hpc3/core/expected.py"
  - "src/hpc3/contracts/provenance.py"
  - "src/hpc3/core/stage.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/stage.py": "52a0836d3ecc451ffd05cfe3860580d8182cd9ec"
  "src/hpc3/core/expected.py": "0ad187f7ba85b5b91ee9d1732c6545825f107cb5"
  "src/hpc3/contracts/provenance.py": "8760fc424d4d9775622b00c4f963735b02be6fa5"
  "src/hpc3/core/stage.py": "582badba6fe1feb258dd91d0be2b7735d1c7898b"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
fact_checked: 2026-09-01
confidence: high
---

# Staged bytes are held to a record written by a different act

Three checks in this package verify *transport* — that what arrived is what
left. The staging check verifies *identity* — that what left was the right
thing in the first place. It exists because transport checks cannot catch a
run that completes, reports plausible numbers, and is comparable to nothing.

```bash
hpc3-stage --config hpc3.json --manifest runs/stage.json \
    --source-dir runs/corpora --expect-from runs/file_ids.txt
```

A manifest is self-consistent by construction: whoever emitted the files
computed the digests from those same files, so they always agree. That proves
the emitter was deterministic and nothing else.

`--expect-from` is required and points at a record written by a *different
act* — every digest in the manifest must appear in it. That is a real check
precisely because re-emitting a corpus from the wrong source state produces
new digests, and new digests are not in the record. Any text works: a
`sha256sum` listing, a JSON manifest, a run log.

## The provenance block

Every manifest carries a required, non-empty `provenance` block:

```json
{
  "destination": "/pub/wagnera3/abl/corpora",
  "files": [{ "name": "armB.txt", "sha256": "…", "size_bytes": 41943040 }],
  "provenance": {
    "wiki_commit": "176bb8c",
    "emitter": "extraction-eval/emit_corpus.py",
    "emitter_flags": "--seed 0 --dilution oscar_en.txt --dilution-ratio 7.0"
  }
}
```

Free-form because what identifies a source differs per project, and a fixed
schema would mean writing `"none"` into fields that do not apply. The block is
the record; `--expect-from` is the enforcement.
