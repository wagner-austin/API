# code-style-eval

Scores a model's generated code against the monorepo's own checkers.

House style has no external benchmark. Nothing off the shelf knows this repo
wants TypedDicts with encode/decode pairs, `_test_hooks` seams and no `Any`.
The repo's own `ruff`, `mypy` and `scripts.guard` do, so they are the metric:
a completion passes if the tools the operator already runs would accept it.

## What it measures

Given held-out documents from a `code-corpus` emission, the tool shows a model
the head of each file, takes its continuation, and runs all three checkers over
the result. The per-item outcomes are kept, not just the rate, because two arms
are scored on the same items and the comparison is paired.

Reported per arm: the combined guard-pass rate and a rate per checker, since a
model can be syntactically clean and architecturally wrong.

Reported per pair: the 2x2 table, the net items fixed minus broken, and a
two-sided exact McNemar p-value over the discordant pairs. Not a t-test. The
outcome per item is boolean, the arms share items, and item difficulty
dominates between-item variance, so pooling into two independent samples
measures which files were sampled rather than which model wrote the code.

## Usage

```
poetry run code-style-eval \
  --holdout ../code-corpus/runs/code-corpus-v1.holdout.jsonl \
  --generated-dir runs/gen \
  --interpreter path/to/python \
  --arm candidate \
  --out runs/outcomes.jsonl
```

Generation is not this tool's job. It scores files that already exist, so the
same instrument scores a base model, an adapter, or a human.
