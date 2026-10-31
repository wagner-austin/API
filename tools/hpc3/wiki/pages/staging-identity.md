---
title: Staged bytes are held to a record written by a different act
tags: [identity, staging]
hubs: [images-and-staging]
related: ["[[image-build-flow]]", "[[known-answers]]"]
source_paths:
  - "src/hpc3/core/expected.py"
  - "src/hpc3/core/stage.py"
  - "src/hpc3/core/reproducible.py"
  - "src/hpc3/cli/stage.py"
  - "tests/test_stage_report.py"
source_git_blobs:
  "src/hpc3/core/expected.py": "7355b6fee112fcba63cc346eef386fead380ea03"
  "src/hpc3/core/stage.py": "88d735b5d967dda49c3c623cbf8187ede2d2ead3"
  "src/hpc3/core/reproducible.py": "8258650f38c32515c80274d6acfe9917b88646b6"
  "src/hpc3/cli/stage.py": "01557e1aa9cc012257a43781a64b2fe3cff6b815"
  "tests/test_stage_report.py": "cc94e608baba49d799471014273df0040b258c6e"
provenance:
  - "platform_core.stage_manifest (libs/platform_core, outside this workspaceRoot)"
  - "platform_core.stage_provenance (libs/platform_core, outside this workspaceRoot)"
fact_checked: 2026-10-31
confidence: high
---

# Staged bytes are held to a record written by a different act

Three checks in this package verify *transport* — that what arrived is what
left. The staging check verifies *identity* — that what left was the right
thing in the first place. It exists because transport checks cannot catch a
run that completes, reports plausible numbers, and is comparable to nothing.[^identity]

```bash
hpc3-stage --config hpc3.json --manifest runs/stage.json \
    --source-dir runs/corpora --expect-from runs/file_ids.txt
```

A manifest is self-consistent by construction: whoever emitted the files
computed the digests from those same files, so they always agree. That proves
the emitter was deterministic and nothing else.[^identity]

`--expect-from` is required and points at a record written by a *different
act* — every digest in the manifest must appear in it. That is a real check
precisely because re-emitting a corpus from the wrong source state produces
new digests, and new digests are not in the record. Any text works: a
`sha256sum` listing, a JSON manifest, a run log.[^identity]

[^identity]: `src/hpc3/core/expected.py`, module rationale, `read_expected_digests` and `check_expected`; `src/hpc3/core/stage.py`, `stage_one` and `stage_manifest`.

`--report-task <UUID>` publishes a program-authored receipt through the shared
HPC board reporter after successful staging. The existing command also reads
back the remote certificate and refuses a changed independent digest record
before publication. The receipt names the manifest and expected-file hashes,
every verified file identity, the certificate checksum, and the executing
Python path and checksum. A transfer failure, wrong certificate or refused
board write fails the command; none produces a success receipt. This uses the
same transport and staging functions as an invocation without reporting.[^report]

[^report]: `src/hpc3/cli/stage.py`, `main`; `tests/test_stage_report.py`, `test_report_names_verified_bytes_and_controller`, `test_failed_transport_or_changed_evidence_never_posts` and `test_board_failure_propagates_after_verified_transport`. API `9c79fc713c4d30e8e89390c3990b735435458475`, fleet `8fd49050-ae8c-4e55-8029-20ea5d718a02`: 2,131 tests, 100% statement and branch coverage. Board task `a622165c-bb63-4714-933b-10d527f21478`.

**`--expect-from` cannot see one thing, and since 2026-09-09 staging refuses
it separately** (re-read 2026-10-08). A manifest records the digest of the
bytes on disk, so a tracked text file checked out with CRLF against an LF blob
is recorded accurately and reproducible by nobody: another checkout hashes
different bytes. The manifest and the expected record are written on one
machine from one checkout, so they agree by construction. `stage_manifest`
now refuses with `STAGE_SOURCE_NOT_REPRODUCIBLE` a tracked file whose bytes
differ from the repository's blob, comparing with `git hash-object
--no-filters`; the first version omitted `--no-filters`, so git's clean filter
normalised the file before hashing, returned HEAD's blob and compared a value
against itself while all ten unit tests passed. Measured when it landed, 363
files across every `*stage.json` under `runs/` held three violations, all in
code-style, left refusing rather than renormalised because those bytes
produced finished jobs.[^repro]

## The provenance block

Every manifest carries a required, non-empty `provenance` block:[^shared]

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
the record; `--expect-from` is the enforcement.[^shared]

The manifest and its provenance block are shapes, not staging. They live in
`platform_core.stage_manifest` and
`platform_core.stage_provenance`. A payload writes a manifest as well as this
package reading one -- rusted's `rw_bot.stage_record` builds them -- and the
payload's image ships `platform_core` already; taking the shape from this
package put the whole submitter inside the compute image. The enforcement,
`src/hpc3/core/expected.py`, stays here, because only the submitter needs it.[^shared]

[^shared]: `src/hpc3/cli/stage.py`, `main` decodes through the shared `platform_core.stage_manifest` owner before invoking `check_expected` and `stage_manifest`. The external modules declared in provenance, `platform_core.stage_manifest` (`StageManifest`, `decode_stage_manifest`) and `platform_core.stage_provenance` (`require_provenance`), define the required shape and dependency boundary.

[^repro]: `src/hpc3/core/stage.py` § `stage_manifest`, whose docstring raises `STAGE_SOURCE_NOT_REPRODUCIBLE`, and `src/hpc3/core/reproducible.py:104`, the `git -C <source_dir> hash-object --no-filters <name>` call, under the comment at `:78` that opens "``--no-filters`` IS LOAD-BEARING, AND ITS ABSENCE IS UNDETECTABLE BY ANY"; API commit `057531331` (2026-09-09), whose message carries the 363-file measurement and the 28ba697b against 1007163890 reading of `code-style-gen-v2-base.json`.
