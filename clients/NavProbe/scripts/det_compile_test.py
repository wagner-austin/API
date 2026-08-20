"""Deterministic-mode compile gate for MuJoCo-Warp, on the Warp CPU device.

Run as ``python -m scripts.det_compile_test <MODE> <CACHE_DIR> [--device DEV]``.

  MODE      NOT_GUARANTEED | RUN_TO_RUN | GPU_TO_GPU
  CACHE_DIR fresh directory for the Warp kernel cache (cold codegen guaranteed)
  --device  Warp device to compile on, default ``cpu``. A deterministic-mode
            rejection fires while Warp parses Python into its IR, so no GPU is
            needed to trigger one and none can avoid one: the gate is a CPU
            test of a GPU property. The flag exists so the same gate can be
            pointed at a card when the question is whether *that device's*
            codegen path differs.

Builds the NavProbe touching-row scene (6 bodies, 0.055 spacing -- the
configuration where GPU runs are irreproducible), constructs the state adapter
over it, and steps it twice.

The scene is driven through :mod:`navprobe.adapters.mjx_warp_state` rather than
through hand-written ``put_model``/``put_data`` calls, so the gate exercises the
same pipeline the measurements do. A gate that drove the vendor its own way
could pass while the measured path still failed.

**A rejection is not caught.** Warp's error carries the file, the line and the
reduction families that conflicted; catching it to write a "failed" document
would replace that with a summary, and would let a genuine bug be filed as a
determinism-mode rejection. So this script writes a record when the mode
compiles, and raises the vendor's own error when it does not. The absence of the
record is the other answer.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from navprobe.codecs.compile_gate import encode_compile_gate
from navprobe.records import CompileGateRecord, SceneSpec
from navprobe.scenes import build_scene, row_scene
from scripts import _test_hooks
from scripts.arguments import DEVICE_FLAG, ScriptArgumentError, split_device

#: The device the gate runs on when ``--device`` is absent.
DEFAULT_GATE_DEVICE = "cpu"

#: Half-width of the seed-driven initial offset range.
PERTURBATION = 0.01

#: Upper bound on constraints, contacts and Jacobian non-zeros.
CONSTRAINT_CAPACITY = 8192

#: Parallel worlds the gate allocates.
WORLD_COUNT = 2

#: Seed the gate resets to. Any seed compiles the same kernels.
SEED = 7

#: Steps taken after construction. Two rather than one because the first
#: advance compiles and the second proves the compiled kernel is re-enterable.
STEP_COUNT = 2

#: The scene the gate compiles.
SCENE: SceneSpec = row_scene(6, 0.055, 0.03, 0.005)

#: Positional arguments accepted.
POSITIONAL_COUNT = 2

USAGE = "usage: det_compile_test <MODE> <CACHE_DIR> [--device DEV]"


class Invocation:
    """One parsed command line.

    Args:
        mode_name: Determinism mode to configure.
        cache_dir: Kernel-cache directory for this run.
        device: Warp device identifier to compile on.
    """

    def __init__(self, mode_name: str, cache_dir: str, device: str) -> None:
        self.mode_name = mode_name
        self.cache_dir = cache_dir
        self.device = device


def parse_invocation(args: Sequence[str]) -> Invocation:
    """Parse a command line.

    The device defaults to :data:`DEFAULT_GATE_DEVICE` rather than to the
    shared ``cuda:0``, because parse-time rejection needs no GPU and a gate
    that silently required one would be unrunnable on the machines that most
    need to run it.

    Args:
        args: Arguments excluding the program name.

    Returns:
        The parsed invocation.

    Raises:
        ScriptArgumentError: When the arity is wrong.
    """
    requested, positional = split_device(args)
    if len(positional) != POSITIONAL_COUNT:
        raise ScriptArgumentError(
            "NP-ARGS-006",
            f"{USAGE} -- expected {POSITIONAL_COUNT} positional arguments, got {len(positional)}",
        )
    device = requested if DEVICE_FLAG in args else DEFAULT_GATE_DEVICE
    return Invocation(positional[0], positional[1], device)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compile gate and write its record.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        ``0`` when the mode compiled and stepped.

    Raises:
        ScriptArgumentError: When the command line is unusable.
        ValueError: When Warp does not recognise the requested device.
    """
    invocation = parse_invocation(list(sys.argv[1:]) if argv is None else argv)
    warp = _test_hooks.init_warp(invocation.mode_name, invocation.cache_dir, 0)
    device_label = str(warp.get_device(invocation.device))
    construct = _test_hooks.load_state_factory()
    model_xml = build_scene(SCENE)

    started = _test_hooks.monotonic()
    with warp.ScopedDevice(invocation.device):
        factory = construct(model_xml, WORLD_COUNT, PERTURBATION, CONSTRAINT_CAPACITY)
        simulator = factory()
        simulator.reset(SEED)
        for _ in range(STEP_COUNT):
            simulator.advance()
    elapsed = _test_hooks.monotonic() - started

    _test_hooks.write_out(
        encode_compile_gate(
            CompileGateRecord(
                mode=invocation.mode_name,
                device=device_label,
                device_request=invocation.device,
                max_records=0,
                wall_seconds=elapsed,
                world_count=WORLD_COUNT,
                scene=SCENE,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
