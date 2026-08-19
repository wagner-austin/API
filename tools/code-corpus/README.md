# code-corpus

Emit training corpora from the monorepo's own source code, for the
code-style training pipeline (board task `315a8d38`): a corpus emitter whose
every output can be reconstructed from what it records about itself.

## What it does

`code-corpus-emit` walks one or more git repositories, selects tracked
source files (gitignore-aware by construction: the universe is
`git ls-files`), and writes three artifacts:

- a **training JSONL** — one record per source file,
- a **holdout JSONL** — whole files held out by seeded sample, for the
  guard-pass evaluation's prompts and for held-out perplexity,
- a **manifest** — repository commits (plus a dirty flag), selection and
  exclusion counts, per-language totals, the split parameters, and sha256
  digests of both emitted files.

Excluded and counted: generator output (`document_categories.py` twins),
whitespace-only files, and byte-identical duplicates (the per-project guard
scripts are lifted verbatim across services; training on N copies
overweights them N-fold). Files that are not valid UTF-8 fail the emission
loudly rather than being skipped.

## Why JSONL and not plain text

Model-Trainer's existing corpus reader is line-oriented and strips each
line, which destroys indentation — fatal for Python source. Documents are
therefore carried as JSON strings, where newlines and leading whitespace
survive byte-exactly. Consuming this format needs a document-mode dataset
path in Model-Trainer (part of the training-run component of task
`315a8d38`); the emitter deliberately does not bend the corpus to fit the
line-oriented reader.

Each record:

```json
{"repo": "api", "path": "libs/.../errors.py", "language": "python",
 "sha256": "<digest of the file's normalized content>",
 "tokens_approx": 123, "text": "# api/libs/.../errors.py\n<file content>"}
```

`text` is the training document — a path-header comment followed by the
file's content (UTF-8, LF-normalized). `sha256` digests the content
*without* the header; it identifies the source file for deduplication and
provenance.

## Usage

```
poetry run code-corpus-emit \
  --repo api=C:/Users/Test/PROJECTS/api \
  --repo mcp=C:/Users/Test/PROJECTS/MCPs \
  --out runs/code-corpus.jsonl \
  --holdout-fraction 0.1 --seed 0
```

Defaults: `--language python` (the only language table entry so far),
holdout fraction 0.1, seed 0, holdout path derived as
`<out>.holdout.jsonl`, manifest as `<out>.manifest.json`. Repository order
matters: deduplication keeps the first occurrence of identical content.

The holdout split is **by file**. A within-file split would leak the
held-out remainder's style into training — the same defect tracked for the
trainer's own validation split (board task `f8706e51`).

## Uploading to the data bank

The emitter writes files; it does not talk to the network. Upload the
training corpus the same way the wiki ablation corpora were uploaded
(multipart `POST /files` on data-bank-api) and record the returned
`file_id` next to the manifest, e.g. in `runs/file_ids.txt`:

```
curl -sS -X POST "$DATA_BANK_URL/files" -F "file=@runs/code-corpus.jsonl"
```

A future training run then references that `file_id` as its
`corpus_file_id`.

## Development

`make check` — guards (monorepo_guards via `scripts/guard.py`), ruff, mypy
strict over `src`, `tests` and `scripts`, then pytest with xdist and 100%
statement+branch coverage over `src` and `scripts`.
