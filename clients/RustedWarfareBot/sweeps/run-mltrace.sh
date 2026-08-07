#!/bin/sh
# The first income-pair batch (log 2026-08-05/06), chained behind
# ledger-solo24's engine slot the same way run-xmap-1v1.sh chained
# behind floor-solo24. The chain waits for the LAUNCHER's completion
# line rather than counting scorecards, because a crashed match leaves
# a .partial forever and a count of 24 would then never arrive.
#
# Full 24-seed panel rather than a 6-seed sliver: the schema rides the
# frozen tree, so the first new-schema batch is also the first real
# rw_matches supply (~36k rows), and the trace change is pure
# instrumentation, so it doubles as a ledger-quality panel. Three
# workers -- data supply has no solo-sequential purity condition, and
# the machine keeps headroom (the 2026-08-06 crash lesson).
cd "$(dirname "$0")/.." || exit 1
until grep -q "matches have results" runs/ledger-solo24.out 2>/dev/null; do sleep 120; done
make sweep SWEEP_JOBS=sweeps/vh-solo24.txt SWEEP_NAME=mltrace24 SWEEP_WORKERS=3 SWEEP_LOCKSTEP=75 "SWEEP_MATCH=maps/skirmish/[p2]duel_lake.tmx 2" SWEEP_PINDELTA=3 > runs/mltrace24.out 2>&1
echo "mltrace24 complete"
