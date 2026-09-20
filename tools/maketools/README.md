# maketools

What the monorepo's Makefile recipes call instead of a shell.

```bash
make check            # lint + test, 100% statements and branches
make lint-makefiles   # every tracked Makefile against the grammar
```

## Why it exists

Until 2026-09-20 all 55 Makefiles here set `SHELL := powershell.exe` and
`make test` ran `scripts/run-tests.ps1`, so `make check` could run on a
Windows node and nowhere else. The fleet dispatcher (`tools/fleet`) could
not send a project to the one Linux node, and CI's `packages.yml` kept a
hand-maintained duplicate of every Makefile's steps because the Linux
runners could not call the Makefile. Board task `33bb86ce` carries the
measurement and the decision; MCPs did the same a day earlier (`b835753b`).

The fix is not a second Makefile per platform. Each platform keeps its
native shell (`scripts/make/shell.mk` picks it: PowerShell on Windows,
`/bin/sh -eu` elsewhere) and every recipe is written in the **intersection**
of the two. Anything with logic moves into this package and is called by
path with the **system** interpreter, so nothing has to be installed first:

```make
test:
	$(PYTHON) ../../tools/maketools/scripts/run.py test
```

`scripts/run.py` puts `src` and `libs/platform_core/src` on the path by
the same route every `scripts/guard.py` takes to `monorepo_guards`; the
three `platform_core` modules used (`errors`, `error_codes_tooling`,
`json_utils`) are standard library all the way down.

## The grammar

A recipe line is one plain command. Allowed: the command and its arguments,
`@echo "..."`, make's own `$(VAR)` and `$(MAKE)`, single or double quotes.
Banned, and refused by `lint-makefiles`:

| banned | because |
|---|---|
| cmdlets (`Write-Host`, `Set-Location`, ...), `$env:`, `-ForegroundColor`, `.ps1`, backticks, `\` paths | PowerShell only |
| `&&`, `\|\|`, `$$VAR`, `/dev/null`, `[ test ]`, `if`/`for` | sh only |
| `;`, `>`, `<`, `\|`, `cd` | read differently by the two shells |
| `SHELL :=`, `.SHELLFLAGS`, `$(shell ...)` outside `shell.mk` | the platform shell leaking into a Makefile |

One fence is allowed: inside `ifeq ($(OS),Windows_NT)` ... `else` ... `endif`
the Windows arm may use PowerShell, because a target that registers a Task
Scheduler job has no meaning elsewhere; the `else` arm is checked like any
other line and defines the same names so a Linux host gets a refusal that
says why.

## The commands

| command | what it does |
|---|---|
| `venv-check` | remove `.venv` when `poetry run mypy --version` cannot answer, so the `poetry sync` that follows builds a fresh one |
| `guard` | run `poetry run python -m scripts.guard` when the package has a shim; say "not applicable, 0 rules run" when it has none |
| `test [--no-sweep] [pytest args...]` | the shared launcher, the port of `scripts/run-tests.ps1`: a kill-on-close job object on Windows, a new session and a descendant reap on POSIX, the pre-run sweep, `--max-worker-restart=0`, a per-run `COVERAGE_FILE` under `runs/` with a cleanup scoped to that token, and one BLAS thread per worker |
| `reap-stale [--older-than-minutes N]` | the standalone sweep, the port of `scripts/reap-test-processes.ps1`: candidates older than N minutes whose ancestry names this project, reaped only when the whole set is idle across a 5-second CPU sample |
| `lint-makefiles` | the grammar over every tracked Makefile |
| `env [NAME=VALUE \| NAME?=VALUE \| NAME=]... [--draw NAME=LOW-HIGH]... [--then "cmd"] -- argv...` | run a command with variables set, defaulted, unset or drawn from a range, every assigned name substituted into the argv as `@NAME@`; `--then` runs a second command whatever the first's status and the first non-zero status of the two is the verdict |
| `fan-out TARGET PARENT...` | `make -C <child> TARGET` in every child directory of each parent that has a Makefile, running all of them and naming every failure |
| `compose-up DIR [--git-commit] [--build-progress X]`, `compose-down DIR...` | `docker compose` bring-up per service directory (with `GIT_COMMIT` from `git rev-parse HEAD` when asked) and the matching teardown |
| `hooks install\|check` | set or verify `core.hooksPath` |
| `require-tool NAME HINT` | refuse with the hint when a tool is not on PATH |
| `uv-venv-check`, `venv-exec NAME ARGS...` | the `uv` venv of a Rust-backed package and an executable from its `.venv` |
| `native-wheel --crate DIR --package NAME`, `poetry-build DIR...` | install a crate's newest maturin wheel into the package venv; build first-party wheels |

## Where the process table comes from

The reaper's questions are the same on both platforms; the source differs.
Windows answers through PowerShell's `Get-CimInstance Win32_Process`,
rendered as JSON rows; Linux answers from `/proc`. Each reader takes its
input through a seam (the command runner, a `/proc` root), so the parsing is
exercised by the suite on the platform that cannot produce the real input.
