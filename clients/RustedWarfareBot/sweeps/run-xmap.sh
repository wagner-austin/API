#!/bin/sh
# Cross-map validation: the champion on four unexplored maps, two seeds each,
# chained behind the zone2 screen. Each map is one mini-sweep because the
# harness carries the map globally (log 2026-08-04).
until [ "$(ls runs/sweeps/vh-zone2/*.txt 2>/dev/null | wc -l)" -ge 6 ]; do sleep 300; done
for MAP in "maps/skirmish/[p2]lake_2p.tmx" "maps/skirmish/[p2]big_island.tmx" "maps/skirmish/[p2]hills_2p.tmx" "maps/skirmish/[p2]two_cold_sides.tmx"; do
  NAME="xmap-$(basename "$MAP" .tmx | tr -d '[]p')"
  make sweep SWEEP_JOBS=sweeps/xmap-jobs.txt "SWEEP_NAME=$NAME" SWEEP_WORKERS=2 "SWEEP_MATCH=$MAP 2" SWEEP_PINDELTA=3 > "runs/$NAME.out" 2>&1
done
echo "cross-map validation complete"
