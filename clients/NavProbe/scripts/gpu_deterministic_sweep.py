"""GPU determinism sweep for the ten-scene family under a chosen Warp mode.

Usage: python scripts/gpu_deterministic_sweep.py <MODE> <CACHE_DIR> [MAX_RECORDS]
  MODE        NOT_GUARANTEED | RUN_TO_RUN | GPU_TO_GPU
  CACHE_DIR   Warp kernel-cache directory for this run
  MAX_RECORDS optional wp.config.deterministic_max_records override. The
              default 0 uses Warp's code-generated lower bound, which the
              solver's data-dependent contact loops exceed at 32 bodies
              (RuntimeError: deterministic scatter buffer overflow in '_M').

Mirrors the wiki-pinned CPU-control design exactly, on cuda:0: scenes
separated 2/8/16/32 at spacing 0.070 and touching 2/4/5/6/8/32 at 0.055,
radius 0.03, timestep 0.005, TrialSpec(seed=7, step_count=150,
repetitions=12), world_count=2, perturbation=0.01, constraint_capacity=8192.
Prints one ``REPORT {json}`` line: per-scene deterministic verdict, first
divergent step, reference digest, and wall seconds (first scene's wall
includes module compilation for the process).
"""
import json
import sys
import time

mode_name, cache_dir = sys.argv[1], sys.argv[2]

import warp as wp

wp.config.kernel_cache_dir = cache_dir
if mode_name != "NOT_GUARANTEED":
    wp.config.deterministic = getattr(wp.DeterministicMode, mode_name)
max_records = int(sys.argv[3]) if len(sys.argv) > 3 else 0
if max_records:
    wp.config.deterministic_max_records = max_records
wp.init()

from navprobe.adapters.mjx_warp_state import MjWarpStateSimulatorFactory
from navprobe.records import TrialSpec
from navprobe.scenes import build_scene, row_scene
from navprobe.sweep import run_scene_sweep

PERTURBATION = 0.01
CONSTRAINT_CAPACITY = 8192
TRIAL = TrialSpec(seed=7, step_count=150, repetitions=12)
WORLD_COUNT = 2

SCENES = [row_scene(n, 0.070, 0.03, 0.005) for n in (2, 8, 16, 32)] + [
    row_scene(n, 0.055, 0.03, 0.005) for n in (2, 4, 5, 6, 8, 32)
]


def build_factory(model_xml: str, world_count: int) -> MjWarpStateSimulatorFactory:
    return MjWarpStateSimulatorFactory(model_xml, world_count, PERTURBATION, CONSTRAINT_CAPACITY)


device = wp.get_device()
rows = []
with wp.ScopedDevice("cuda:0"):
    for spec in SCENES:
        t0 = time.perf_counter()
        (entry,) = run_scene_sweep(build_factory, [spec], TRIAL, WORLD_COUNT)
        wall = time.perf_counter() - t0
        trial = entry["trial"]
        rows.append(
            {
                "bodies": spec["body_count"],
                "spacing": spec["spacing"],
                "deterministic": trial["deterministic"],
                "first_divergent_step": trial["first_divergent_step"],
                "reference_digest": trial["reference_digest"],
                "wall_s": round(wall, 2),
            }
        )
        print(
            f"scene bodies={spec['body_count']} spacing={spec['spacing']} "
            f"deterministic={trial['deterministic']} first_div={trial['first_divergent_step']} "
            f"wall={wall:.1f}s",
            flush=True,
        )

report = {
    "mode": mode_name,
    "max_records": max_records,
    "device": str(wp.get_device("cuda:0")),
    "trial": dict(TRIAL),
    "world_count": WORLD_COUNT,
    "perturbation": PERTURBATION,
    "rows": rows,
    "all_deterministic": all(r["deterministic"] for r in rows),
}
print("REPORT " + json.dumps(report))
