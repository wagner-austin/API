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
