#!/bin/sh
# The cross-map validation as it was always meant to run: TRUE 1v1 on every
# map, now that the seating override holds (players 2 -> 2 verified live on
# the four-seat lake_2p, `runs/seat-probe.out`, log 2026-08-05). Chained
# behind floor-solo24's last matches for the engine slot.
until [ "$(ls runs/sweeps/floor-solo24/*.txt 2>/dev/null | wc -l)" -ge 24 ]; do sleep 120; done
cd "$(dirname "$0")/.." || exit 1
for MAP in "maps/skirmish/[p2]lake_2p.tmx" "maps/skirmish/[p2]big_island.tmx" "maps/skirmish/[p2]hills_2p.tmx" "maps/skirmish/[p2]two_cold_sides.tmx"; do
  NAME="duel-$(basename "$MAP" .tmx | sed 's/^\[p2\]//')"
  make sweep SWEEP_JOBS=sweeps/xmap-jobs.txt "SWEEP_NAME=$NAME" SWEEP_WORKERS=1 "SWEEP_MATCH=$MAP 2" SWEEP_PINDELTA=3 > "runs/$NAME.out" 2>&1
done
echo "true-1v1 cross-map validation complete"
