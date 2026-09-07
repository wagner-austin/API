# Code standards — the review rubric for this repo

The bar a change here is held to, written so it can be **checked against a
diff**. `CLAUDE.md` says how to work in this repo and routes to the canonical
documents; `docs/RESEARCH.md` is the index of work that produces comparable
numbers. This file is neither. It is the list a reviewer — human, or the audit
manager running its standards arm — reads a `git show` against.

**Per repository, by design.** The manager reads the `CODE_STANDARDS.md`
belonging to the repo a closure names, and never borrows another repo's bar. A
repo without one is reviewed against nothing and the verdict says so. This file
exists because API had no rubric, so every audit of API work returned
`standards: not reviewed` — permanently, for the repo holding the cluster
tooling, Model-Trainer, ClearGBM and every client bot.

## Scope, stated first because it is uneven

48 packages under `clients/`, `libs/`, `services/` and `tools/` each ship the
full harness. **`amex_1st_place/`, `docs/`, `wiki/` and `scripts/` do not** —
no `Makefile`, no guard, no coverage gate. A diff in those four is reviewed
against this file's judgment section alone, and nothing mechanical is implied
about it. Do not read a clean review of that code as a checked one.

**And the harness is Python-shaped while the repo is not.** Everything in the
next section describes Python. `libs/cleargbm_rs` is a Rust crate — its
`[tool.coverage.run] source` is `["scripts"]` alone, which is correct rather
than a carve-out, because its `src/` is `.rs` and the crate carries its own
Rust coverage beside it (`RustCoverageRule`, `RustTestRule`,
`RustExplicitMatchRule` and `RustProptestRule` are what hold that half).
`services/grandma-api/web` is TypeScript and answers to none of it. A rubric
that said "100% branches" without saying "Python packages" would be the
vacuous-green shape one level up: it would read as covering surfaces it never
measured. That distinction is `opus-corpus-docmode-0901`'s, sent 2026-09-07
while this file was being written.

## What this is NOT

It is not a second linter, and in this repo that is not a hedge — it is a
measurement. Every one of the 48 packages runs, at `make check`:

- **`python -m scripts.guard`** — 45 rule classes from `libs/monorepo_guards`,
  including `MockBanRule`, `MonkeyPatchBanRule`, `WeakAssertionRule`,
  `TypingRule`, `SuppressRule`, `ExceptionsRule`, `AllExportsRule`,
  `FileSizeRule`, `RunRecordRule`, `RunFingerprintLiteralRule`. Each package's
  `scripts/guard.py` is a byte-identical shim into that shared package,
  enforced by `GuardShimRule` — so a package cannot quietly run a weaker set.
- **mypy `strict`** plus `disallow_any_expr`, `disallow_any_explicit`,
  `disallow_any_decorated`, `disallow_any_unimported`, over
  **`src`, `tests` and `scripts`** — all three.
- **ruff** (`E,F,I,B,BLE,UP,N,C4,SIM,RET,C90,RUF,ANN`, line length 100) with
  `typing.Any`, `typing.cast` and `typing.TypeAlias` banned outright at the
  import.
- **pytest with coverage at `fail_under = 100`**, branch coverage on, over
  `src` and `scripts` (47 of 48; `libs/cleargbm_rs` is `scripts` alone for the
  reason above). **And there are no carve-outs**: measured 2026-09-07, zero
  `exclude_lines` and zero non-empty `omit` anywhere in first-party code, so
  the 100% is over everything rather than over what was left after exclusions.
  That is worth knowing before you read a coverage number here as soft.

**So do not re-report what that already refuses.** A finding of "uses `Any`",
"mocks a dependency", "weak assertion", "missing `__all__` entry", "file over
the size ceiling" in one of the 48 is almost certainly a misreading of the
diff — those cannot merge. A review whose findings are all mechanical is noise
wearing the shape of rigour.

This rubric earns its place on exactly two things the tooling cannot see: code
in the four unharnessed trees, and **the judgment section below**.

A finding is an **observation**, never an accusation. A closure is not false
because its diff has a standards finding. Say what you saw, point at a file and
a line, and let the author answer. The audit's whole credibility rests on not
manufacturing findings — a checker that invents them gets routed around, and
then it checks nothing.

## The standing directive, and the rule that already refuses each part

The operator's standing instruction, restated verbatim because it is given to
sessions by hand and belongs where the bar is written:

> "I want to ensure no back compat shims, no thin wrappers, no fallbacks, no
> legacy code, no type alias."

**Every clause is already enforced here by name.** This section is a map, not a
new bar — it exists so a reviewer cites the rule that fails the build instead of
paraphrasing the directive into a fresh one, and so a session that reads only
this file knows the clauses are mechanical rather than aspirational:

| clause | what refuses it |
|---|---|
| no fallbacks | `hook-fallback-dispatch`, and `hook-conditional-dispatch` for the sibling shape — a call site that branches on which implementation is installed |
| no back-compat shims | `shim-compatibility-marker` |
| no thin wrappers | `passthrough-alias` |
| no legacy code | `qr-legacy-validators` |
| no type alias | ruff bans `typing.TypeAlias` at the import; `typealias-forbidden` |
| no `Any`, no `cast` | `any-usage`, `cast-call`, `ruff-missing-ban-typing-any`, `ruff-missing-ban-typing-cast`, and the `mypy-disallow-any-*-disabled` rules that stop a package relaxing its own config |
| no `type: ignore`, no `.pyi` | `type-ignore`, `pyi-disallowed` |
| no `if TYPE_CHECKING:` | `imports_rules.py` — `TYPE_CHECKING` is in `forbidden_typing_imports`, catching both `from typing import TYPE_CHECKING` and the `typing.TYPE_CHECKING` attribute form |
| no dataclasses | `dataclass-decorator-forbidden`, `dataclass-import-forbidden` |
| DI through hooks, not branching | `hook-dispatch`, `hook-conditional-dispatch` — production wires the real implementation at startup and tests swap it; no call site asks which one it got |

Two consequences worth stating, because they are what the table is FOR:

- **A directive violation is a build failure, not a review opinion.** If you are
  about to write "this looks like a fallback" as a finding, name the rule
  instead — and if no rule fires, ask whether it is really the thing the
  directive names before saying so.
- **The reverse also holds.** A shape none of these rules catches is a judgment
  item, and belongs in the section below with its reasoning, not asserted as if
  the tooling had already ruled on it.

## The harness a package must SHIP

The section above is a description of the 48, not permission for the 49th to
skip it. A new package **grows one**, modelled on the nearest sibling that
works — otherwise it is compliant by having nothing to check it, which is the
vacuous-green failure this file warns about two sections down.

- `Makefile` and `pyproject.toml` copied from the nearest working sibling. Do
  not invent a layout. **`pyproject.toml` is never edited without the
  operator** — it invalidates the lock hash, and the CUDA-wheel packages
  re-resolve gigabytes. `poetry add` for dependencies.
- `scripts/guard.py` — the shim, generated from
  `monorepo_guards.guard_shim_template`. Edit the template, never the copy.
- ruff and mypy over `src`, `tests` AND `scripts`. A test file outside the type
  checker is where `Any` goes to live, and it is the file most likely to assert
  something false.
- Coverage at 100% statements **and branches**, over `src` and `scripts`.
- Capture output as
  `powershell -Command "& { make check 2>&1 | Select-Object -Last 100 }; exit $LASTEXITCODE"`.
  The trailing `exit` is not optional: PowerShell does not propagate a native
  command's status out of a script block, so without it a failing `make check`
  reports success.

## Judgment — what a reviewer is actually for

None of these is decidable by a linter. The first three are shared with the
MCPs rubric and were **measured to fire on API code** on 2026-09-07, by
`opus-corpus-docmode-0901`, applying that file by hand to two `tools/hpc3`
commits whose ruff, mypy, 45 guard rules and 100% branch coverage were all
green and correctly so.

- **Could this pass while doing nothing?** The failure this monorepo produces
  most is a check that reports success because it examined nothing. The
  measured instance: `require_inputs_present` refuses a run whose declared
  input files are absent, and is silent when a command declares none — so
  renaming `--payload` would make it vacuously green across every run while
  still reporting success. Unit tests could not catch it; they fed the
  extractor command strings the tests themselves wrote, so they agreed with it
  by construction. If a diff adds something that can be vacuously green, say so.
- **Does a new guard have a subject, and is it proved to FIRE?** A guard
  asserted only against clean code has never been seen to work; one whose
  subject was deleted passes forever while guarding nothing. The same session's
  *first fix* for the item above read the flag spellings **from the module under
  test**, so renaming the flag renamed the expectation too and the loop skipped
  every command — it passed while doing nothing, one level up. Found only by
  breaking the extractor on purpose to watch it go red.
- **Is the predicate right, or is there an exemption list?** An exemption list
  is an admission the predicate is wrong. A rule that had to name a file to keep
  passing should have been sharpened instead.
- **Does the test fail if the behaviour regresses**, or only if it crashes?
- **Does it widen a public surface it did not need to?**

### Judgment items specific to this repo

- **Can a number be traced to the job that produced it?** This monorepo's
  output is measurements someone will compare. `platform_core.run_record.RunRecord`
  is the one record shape and `platform_core.comparability` decides whether two
  results may be subtracted — `covenant_ml` carried its own until 2026-08-29,
  which is exactly why nothing could read its numbers beside another
  experiment's. A second record shape is a finding. So is a run that emits a
  number no fingerprint explains. `docs/RESEARCH.md` is the enforced index
  (`tools/hpc3/tests/test_committed_runs.py`); a registered project missing from
  it fails.
- **`"fingerprint": null` is better than a missing key.** It says nobody
  recorded one, rather than leaving the reader to guess. Rows written before a
  pin existed are not retroactively fixable — re-running is what fills the gap.
- **Is a committed file's dependency committed?** An untracked module is
  invisible to everyone else's `make check` and visible only to its author's.
  Measured 2026-09-07: a package went red for the fleet while the session that
  had just run all six packages green reported truthfully about every file it
  could see. Committing early is not tidiness; it is what puts your work inside
  other people's checks.
- **Does this check measure the subject its NAME claims, or the one that
  happened to be on disk?** The mirror of the item above, and the worse half:
  that one is your work being invisible to everyone else's check, this one is
  your check reading a set nobody else has — and it reports success while
  doing it. `tools/hpc3/tests/test_committed_runs.py` asserted floors over
  `runs/*.json` collected by a **filesystem glob**, while `.gitignore` ignores
  `tools/hpc3/runs/*` with five re-inclusion rules. Measured 2026-09-07: 492
  `.json` on disk against 263 tracked at HEAD. The floors were therefore
  calibrated on the author's working tree and could only ever be met on the
  machine that wrote them; CI, the one reader starting from a clean checkout,
  saw 36 and went red. A file named `test_committed_runs.py` that never asks
  git cannot measure committedness — the word in the filename was the claim it
  failed to make. Fixed in `3c27ef82` by extracting `git archive HEAD` and
  reading inside that, which is the general repair: **name the subject, then
  read that subject, not whatever is nearest.**

  The second-order half is why this is a judgment item and not a bug report.
  Whoever commits the missing file turns CI green **without ever having seen
  it red**, and the check returns to being unable to notice the next
  divergence — now with a passing history that reads as evidence it works. A
  check that is wrong is cheaper than one that is wrong and vindicated.
- **Does a check read the command line, or the thing the command line names?**
  A verifier that reads flags does not read inside the file it just verified.
  Stated by its own author as the gap in `require_inputs_present`: preflight
  passed clean on both arms of a run whose spec pointed at a directory that did
  not exist, because `artifact_path` lives in the JSON body and not on the
  command line. Second-order references are the next instance of this class.
- **Does a cluster-touching change survive being asked the wrong way?**
  Expanding a Slurm array aggregate is useless if the query never returns one —
  green unit tests over a wrong query passed twice in two days (`fa8f87f9`,
  `249ee403`). If a diff changes what is *matched*, check whether it also had to
  change what is *asked*.
- **Was the shared index respected?** `git commit` takes the whole index, which
  in this tree carries other sessions' in-flight work. Commit by explicit path
  (`git commit -- <paths>`, which git honours natively) or declare
  `COMMIT_SCOPE`. A commit that swept up files its message does not mention is
  a finding regardless of whether it broke anything.

## Where to look before writing a helper

Lift, don't fork. Search `libs/platform_core` and the other `libs/*` packages
**by function name**, not by concept, before writing a cross-cutting helper —
validation, logging, error formatting, run records, retry. The drift you save
is not yours; it is whoever has to delete the second copy later.
