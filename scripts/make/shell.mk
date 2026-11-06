# The shell every Makefile in this monorepo runs its recipes under.
#
# INCLUDED AS THE FIRST LINE OF EVERY MAKEFILE, at the depth the Makefile
# sits (`include scripts/make/shell.mk` at the root, `../scripts/make/shell.mk`
# from libs/, services/, clients/, `../../` from a package). tools/maketools'
# lint-makefiles guard refuses a Makefile that sets SHELL anywhere else, and
# refuses a recipe that uses either shell's private syntax; the grammar is in
# that guard's docstring and in tools/maketools/README.md.
#
# WHY TWO SHELLS AND NOT ONE. On Windows a PowerShell-launched make (the
# operator's own `powershell -Command "& { make check }"`, the dispatcher's
# Task Scheduler task on every CI node) inherits a PATH with git.exe on it
# and sh.exe not, so sh is not a shell make can find there without a path
# baked into this file that differs per machine. On Linux there is no
# PowerShell. So each platform keeps its native shell and every recipe is
# written in the intersection: one plain command per line, `@echo` for
# output, make's own `$(VAR)` for substitution, and a script by path for
# anything with logic. `-eu -c` on sh so an unset variable or a failing
# command in a line fails the line, which is what PowerShell's native
# command exit propagation gives the same recipe on Windows.
#
# The same file, word for word except for the paths, sits at
# scripts/make/shell.mk in ~/PROJECTS/MCPs (board task b835753b); the two
# repositories share no package, so this is the one deliberate copy.
#
# PYTHON is the SYSTEM interpreter, used only to launch
# tools/maketools/scripts/run.py, which needs nothing installed. The name
# differs (`python` on Windows where the launcher owns `python3`, `python3`
# on Debian-family Linux where `python` is not provided), and this variable
# is the one place that difference is spelled.
ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command
PYTHON := python
else
SHELL := /bin/sh
.SHELLFLAGS := -eu -c
PYTHON := python3
endif

# check-fleet: this package's HEAD checked on a fleet node, the verdict
# waited for and printed (MCPs board task 4412816f). A session host runs no
# make check, and API had no way to put one of its own packages on the
# queue but a hand-pushed staging ref and a raw dispatch_submit. The target
# runs the ONE check-fleet, MCPs' maketools, from the MCPs clone beside this
# repository's (this file's directory is two below the root, and the clone
# one above it), which keys an API package by its bare path as the
# registry here does and reads the stack's address and keys from that MCPs
# clone; nothing of it is copied into tools/maketools. A tree with no MCPs
# clone beside it is refused when the target runs, never when make parses
# this file, so a fleet node's export of API parses as before.
# `make check-fleet TASK=<uuid>` addresses the verdict to that task.
MCPS_MAKETOOLS_RUN := $(dir $(lastword $(MAKEFILE_LIST)))../../../MCPs/packages/maketools/scripts/run.py

.PHONY: check-fleet
check-fleet:
	$(if $(wildcard $(MCPS_MAKETOOLS_RUN)),,$(error MAKETOOLS_LAUNCHER_MISSING: check-fleet runs MCPs' maketools from $(MCPS_MAKETOOLS_RUN), and no MCPs clone sits beside this one))
	$(PYTHON) -u $(MCPS_MAKETOOLS_RUN) check-fleet

# Every Makefile includes this file first, so the target above would be each
# package's default goal; clearing it hands that back to the Makefile's own
# first target, as before the target was added.
.DEFAULT_GOAL :=
