# commit-scope

Refuse a commit carrying staged paths its author did not declare.

## The defect

In a tree several sessions edit at once, **the git index is shared mutable
state with no lock, and `git commit` takes all of it.** `git add <path>`
protects the ADD. Nothing protects the COMMIT. Anything another session stages
between one and the other ships under your message and your authorship.

Measured twice in three hours on 2026-09-04/05, both times by sessions already
following the explicit-path staging rule that was supposed to prevent it:

| commit | what happened |
|---|---|
| `9d945451` | a commit swept another session's staged deletions |
| `09d2a04b` | an `--amend` swept another session's staged files; unwound, relanded as `7e275441` |

`--only` does not save an amend — it rebases the named paths onto a commit
that already contains the swept ones.

## Using it

Declare what is yours, and a sweep is refused:

```
COMMIT_SCOPE="tools/commit-scope,libs/platform_core" git commit -m "..."
```

An entry is a file or a directory prefix. A directory covers a path only as an
exact match or followed by `/`, so declaring `libs/platform_core` does **not**
cover `libs/platform_core_extras` — the sibling a sweep picks up.

## What it honestly does not do

**An undeclared commit is not blocked.** It prints the staged set as a receipt
and proceeds. That is a narrowing, not a close, and it is described that way
everywhere it appears — refusing every undeclared commit would block every
human in the repository, and a check people disable protects nothing. The
receipt buys the one thing it honestly can: the staged set is shown at the
moment of commit, which is where both incidents were visible and unlooked at.

The native close, where it applies, is git's own:

```
git commit -- <paths>
```

which commits exactly those paths regardless of what else is staged. It does
not help an `--amend`.

## Exit codes

| code | meaning |
|---|---|
| `0` | the commit may proceed |
| `1` | the index carries paths outside `COMMIT_SCOPE` |
| `2` | the question could not be asked — git could not answer, or the declaration contains an entry that could never match |

`2` is distinct from `1` deliberately. A broken environment and a refused
commit are different problems, and a hook that returned the same status for
both would teach the author that this check is noisy.

## Layout

| file | holds |
|---|---|
| `scope.py` | the decision, pure — no git, no environment, no output |
| `contracts.py` | the TypedDicts, their decoders, and the validation that refuses unmatchable entries |
| `report.py` | the wording, separated so tests assert it rather than a terminal |
| `_test_hooks.py` | the three seams: git, the environment, stdout |
| `cli/check.py` | order of operations, and the decision-to-exit-code mapping |

Every property is testable without a repository. The tests that do use one
build their own throwaway repo and scope every git call with `-C` — testing
this package by staging into the real index would be committing the exact
defect it exists to prevent.
