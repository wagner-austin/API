"""World-count scaling sweep: determinism and throughput vs nworld, one scene.

Usage: python scripts/world_scaling_sweep.py <MODE> <CACHE_DIR> <MAX_RECORDS> <CAPACITY> <WORLDS...>
  MODE        NOT_GUARANTEED | RUN_TO_RUN | GPU_TO_GPU
  CACHE_DIR   Warp kernel-cache directory
  MAX_RECORDS wp.config.deterministic_max_records (0 = static bound)
  CAPACITY    per-world constraint capacity (njmax/nconmax). The canonical
              sweeps use 8192 at nworld 2; the default sparse-Jacobian
              allocation scales as njmax * nv * nworld, so scaling runs need
              a right-sized value (touching-8 peaks near 20 contacts/world).
  WORLDS...   world counts to ladder through, e.g. 2 64 512 4096

Scene is fixed: touching row of 8 spheres (spacing 0.055, radius 0.03,
timestep 0.005) — solidly inside the coupled-body failure regime. Trial design
matches the canonical sweeps (seed 7, 150 steps, 12 repetitions). Digests are
not comparable across world counts (each world is seed-perturbed, so nworld
changes the state being digested); the per-count outputs are the determinism
verdict and the wall clock, from which world-steps/second is derived.
"""
import json
import sys
import time

mode_name, cache_dir = sys.argv[1], sys.argv[2]
max_records = int(sys.argv[3])
capacity = int(sys.argv[4])
world_counts = [int(x) for x in sys.argv[5:]]

import warp as wp

wp.config.kernel_cache_dir = cache_dir
if mode_name != "NOT_GUARANTEED":
    wp.config.deterministic = getattr(wp.DeterministicMode, mode_name)
if max_records:
    wp.config.deterministic_max_records = max_records
wp.init()

from navprobe.adapters.mjx_warp_state import MjWarpStateSimulatorFactory
from navprobe.records import TrialSpec
from navprobe.scenes import row_scene
from navprobe.sweep import run_scene_sweep

TRIAL = TrialSpec(seed=7, step_count=150, repetitions=12)
SCENE = row_scene(8, 0.055, 0.03, 0.005)
PERTURBATION = 0.01


def build_factory(model_xml: str, world_count: int) -> MjWarpStateSimulatorFactory:
    return MjWarpStateSimulatorFactory(model_xml, world_count, PERTURBATION, capacity)


rows = []
with wp.ScopedDevice("cuda:0"):
    for nworld in world_counts:
        t0 = time.perf_counter()
        try:
            (entry,) = run_scene_sweep(build_factory, [SCENE], TRIAL, nworld)
            wall = time.perf_counter() - t0
            trial = entry["trial"]
            world_steps = nworld * TRIAL["step_count"] * TRIAL["repetitions"]
            rows.append(
                {
                    "nworld": nworld,
                    "deterministic": trial["deterministic"],
                    "first_divergent_step": trial["first_divergent_step"],
                    "reference_digest": trial["reference_digest"],
                    "wall_s": round(wall, 2),
                    "world_steps_per_s": round(world_steps / wall, 1),
                }
            )
            print(
                f"nworld={nworld} deterministic={trial['deterministic']} "
                f"wall={wall:.1f}s world_steps/s={world_steps / wall:.0f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - capacity/overflow failures are findings
            wall = time.perf_counter() - t0
            rows.append({"nworld": nworld, "error": f"{type(exc).__name__}: {exc}"[:400], "wall_s": round(wall, 2)})
            print(f"nworld={nworld} ERROR after {wall:.1f}s: {type(exc).__name__}: {str(exc)[:200]}", flush=True)

print(
    "REPORT "
    + json.dumps(
        {
            "mode": mode_name,
            "max_records": max_records,
            "capacity": capacity,
            "scene": {"bodies": 8, "spacing": 0.055},
            "trial": dict(TRIAL),
            "rows": rows,
        }
    )
)
