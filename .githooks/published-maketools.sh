#!/bin/sh
# Run one MCPs maketools command as MCPs PUBLISHED it, never as a checkout
# holds it.
#
# EVERY COMMIT NAMES ITS TASK (MCPs board task 691b0067). This repository's
# commit-msg and pre-push hooks and its `make commit-tasks` ask MCPs'
# maketools whether a commit names the board task it serves. MCPs is the
# checkout beside this one, and its working tree is shared: other sessions'
# uncommitted edits sit in it, and it moves forward only when the fleet
# deploys a clean hub, so on a given day it can lack the command or carry an
# uncommitted edit to it. A gate read from there could be missing, or
# weakened by an edit nobody committed. So the command is extracted from
# MCPs' origin/main, the published commit its own pre-push already judged,
# which no editor mutates; the checkout supplies only the board's
# credentials and address, which the command reads from it.
#
# MCPs' repository is named with --git-dir, never -C: inside a hook git
# exports GIT_DIR for the repository being committed or pushed, and GIT_DIR
# outranks -C, so `git -C ../MCPs archive` read this repository and found no
# packages/maketools (measured, on this hook's first real commit).
#
# Arguments: the maketools command and its arguments.
# Exit status: the command's own, or 1 when MCPs' origin/main could not
# supply it, which refuses whatever called this.

set -eu

HOOK_REPO="$(cd "$(dirname "$0")/.." && pwd)"
MCPS="$HOOK_REPO/../MCPs"
EXTRACT="$(mktemp -d)"
trap 'rm -rf "$EXTRACT"' EXIT

if ! git --git-dir="$MCPS/.git" archive -o "$EXTRACT/maketools.tar" origin/main packages/maketools; then
    echo "published-maketools: $MCPS has no origin/main carrying packages/maketools, so $1 did not run and what called it is refused" >&2
    exit 1
fi
tar -xf "$EXTRACT/maketools.tar" -C "$EXTRACT"
python "$EXTRACT/packages/maketools/scripts/run.py" "$@"
