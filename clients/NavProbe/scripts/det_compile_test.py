"""Deterministic-mode compile gate test for MuJoCo-Warp (CPU device only).

Usage: python det_compile_test.py <MODE> <CACHE_DIR>
  MODE      NOT_GUARANTEED | RUN_TO_RUN | GPU_TO_GPU
  CACHE_DIR fresh directory for the Warp kernel cache (cold codegen guaranteed)

Builds the NavProbe touching-row scene (6 bodies, 0.055 spacing -- the
configuration where GPU runs are irreproducible), then put_model/put_data and
steps twice on the Warp CPU device. Any deterministic-mode codegen rejection
fires during module parse, so no GPU is needed to trigger it and none can
avoid it. Prints PASS/FAIL plus the full error.
"""
import sys
import time

mode_name, cache_dir = sys.argv[1], sys.argv[2]

import warp as wp

wp.config.kernel_cache_dir = cache_dir
if mode_name != "NOT_GUARANTEED":
    wp.config.deterministic = getattr(wp.DeterministicMode, mode_name)
wp.init()

import mujoco
import mujoco_warp as mjw
from navprobe.scenes import build_scene, row_scene

mjm = mujoco.MjModel.from_xml_string(build_scene(row_scene(6, 0.055, 0.03, 0.005)))
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)

t0 = time.perf_counter()
try:
    with wp.ScopedDevice("cpu"):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=2)
        mjw.step(m, d)
        wp.synchronize()
        mjw.step(m, d)
        wp.synchronize()
    dt = time.perf_counter() - t0
    print(f"RESULT: PASS mode={mode_name} nsensor={mjm.nsensor} wall={dt:.1f}s")
except Exception as exc:  # noqa: BLE001 - the whole point is to report the error class
    dt = time.perf_counter() - t0
    print(f"RESULT: FAIL mode={mode_name} wall={dt:.1f}s")
    print(f"ERROR_TYPE: {type(exc).__module__}.{type(exc).__name__}")
    msg = str(exc)
    print("ERROR_TEXT_BEGIN")
    print(msg[:4000])
    print("ERROR_TEXT_END")
