---
title: Staged bytes are held to a record written by a different act
tags: [identity, staging]
related: [[image-build-flow]], [[known-answers]]
sources: [contracts/stage.py, core/expected.py, contracts/provenance.py, README.md@4dc63f17]
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
