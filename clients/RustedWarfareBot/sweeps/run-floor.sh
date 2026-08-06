#!/bin/sh
# Task #27 validation: the map-derived economy floor, measured against both
# baselines under matched conditions. Four foreign maps exactly as the xmap
# baseline ran them (seeds 12345/777, 2 workers, Very Hard) -- the fix must
# move 0W/5L/3S -- then a duel_lake solo regression pair on two baseline wins,
# where the derivation recovers the literal seven and nothing should move.
cd "$(dirname "$0")/.." || exit 1
for MAP in "maps/skirmish/[p2]lake_2p.tmx" "maps/skirmish/[p2]big_island.tmx" "maps/skirmish/[p2]hills_2p.tmx" "maps/skirmish/[p2]two_cold_sides.tmx"; do
  NAME="floor-$(basename "$MAP" .tmx | sed 's/^\[p2\]//')"
  make sweep SWEEP_JOBS=sweeps/xmap-jobs.txt "SWEEP_NAME=$NAME" SWEEP_WORKERS=2 "SWEEP_MATCH=$MAP 2" SWEEP_PINDELTA=3 > "runs/$NAME.out" 2>&1
done
make sweep SWEEP_JOBS=sweeps/floor-duel.txt SWEEP_NAME=floor-duel SWEEP_WORKERS=1 "SWEEP_MATCH=maps/skirmish/[p2]duel_lake.tmx 2" SWEEP_PINDELTA=3 > runs/floor-duel.out 2>&1
echo "floor validation complete"
