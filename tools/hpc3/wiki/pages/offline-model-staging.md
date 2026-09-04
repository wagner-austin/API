---
title: A model pinned by commit hash stages a cache that offline loading cannot read
tags: [staging, identity, images, offline]
hubs: [images-and-staging]
related: ["[[staging-identity]]", "[[capture-source-drift]]", "[[image-build-flow]]", "[[known-answers]]"]
source_paths:
  - "runs/code-style-run-train.json"
  - "runs/hpc3-code-style.json"
source_git_blobs:
  "runs/code-style-run-train.json": "d60c063f583e12b75e85651d00ae6c24e54ebc04"
  "runs/hpc3-code-style.json": "1704cc0ecd7d68df31eee6b11eb845803a7c7fe4"
provenance:
  - "huggingface_hub file_download.py, function _cache_commit_hash_for_specific_revision -- read from the installed 0.x in services/Model-Trainer/.venv, outside this wiki's workspaceRoot"
  - "job 55744648, code-style.qlora-qwen-v1 on an A30, 2026-09-04: OSError after 9 seconds, 'couldn't find it in the cached files'"
  - "/pub/wagnera3/hf/hub/models--Qwen--Qwen2.5-Coder-1.5B -- no refs/ directory, against models--gpt2 which has refs/main"
fact_checked: 2026-09-04
confidence: high
---

# A model pinned by commit hash stages a cache that offline loading cannot read

A cluster run reads its base model from a cache on `/pub`, with
`HF_HOME`, `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` set by the run
document. Staging that cache by **pinning the revision** — the careful thing
to do, and the thing this repository's whole provenance posture pushes you
toward — produces a cache that is complete and unreadable.

## What happens

`snapshot_download(repo_id, revision="<40-char commit hash>")` downloads every
file and writes them under `snapshots/<hash>/`. It writes **no `refs/`
directory at all**.

`from_pretrained("<repo_id>")` asks for the default revision, `main`. Offline,
`main` can only be resolved through `refs/main`. So:

```
OSError: We couldn't connect to 'https://huggingface.co' to load this file,
couldn't find it in the cached files and it looks like
Qwen/Qwen2.5-Coder-1.5B is not the path to a directory containing a file
named config.json.
```

Every byte of the model is present. Nothing can name it.

## Why, from the library rather than from the symptom

`_cache_commit_hash_for_specific_revision` writes the pointer only when the
requested revision differs from the resolved commit hash:

```python
if revision != commit_hash:
    ref_path = Path(storage_folder) / "refs" / revision
    ...
    ref_path.write_text(commit_hash)
```

Its own docstring says it "does nothing if `revision` is already a proper
`commit_hash`". Pass a hash and `revision == commit_hash`, so the branch is
never taken.

The read side is symmetric: offline resolution takes `revision` directly when
it matches a commit-hash regex, and otherwise goes through
`refs/<revision>`. Downloading by hash and loading by hash would work. It is
downloading by hash and loading by **repo id** — which defaults to `main` —
that fails, and that is the ordinary way every caller loads a model.

The contrast is visible in the cache itself: `models--gpt2`, fetched without a
pin, has `refs/main`. `models--Qwen--Qwen2.5-Coder-1.5B`, fetched with one,
had no `refs/` at all.

## Three ways out, and which to prefer

1. **Write the ref yourself** after a pinned download — one file containing
   the hash. Keeps the pin, which is the point of pinning, and produces
   exactly what an unpinned download would have.
2. **Download unpinned.** Writes `refs/main`, and gives up the pin: the cache
   then holds whatever `main` was that day, which nothing records.
3. **Load with `revision=` the same hash.** Correct, and it requires every
   caller to know the hash — including callers inside a trainer you did not
   write.

The first is preferred here for the reason the rest of this wiki gives about
identity: the snapshot directory stays the pinned commit, so the cache still
says which weights it holds, and the ref is a pointer rather than a claim.

## What now catches it

`model_trainer.cluster.preflight.check_model_available` resolves the model's
config before training starts, and its refusal names the missing ref — because
the cause is not guessable from the symptom. An operator reading "cannot
resolve" against a cache they can SEE the model sitting in has no reason to
suspect a pointer file.

That check mirrors `check_corpus_certified`, which had existed for a while
against the same argument: a corpus is an input a run cannot recover from
getting wrong. A base model is the same class, and had no check until a job
was allocated an A30, verified every output root, round-tripped the artifact
store, certified its corpus, and died nine seconds later.

## Application note

The generic lesson is not about HuggingFace. **A staged artifact has an
identity and a NAME, and pinning the identity can remove the name.** Check
that the thing you staged is reachable the way its consumer will reach it,
not just that its bytes are present — the same distinction
[[staging-identity]] makes about digests proving arrival rather than
provenance.
