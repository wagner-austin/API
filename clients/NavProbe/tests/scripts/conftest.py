"""Hook wiring shared by the measurement-script tests.

The scripts call :mod:`scripts._test_hooks` symbols unconditionally, so a test
installs fakes by rebinding those module attributes and restores them
afterwards. Restoration is not optional: the hooks are module state, and a test
that left a fake in place would hand it to every test that ran after it in the
same worker.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from scripts import _test_hooks
from tests.scripts.vendor import (
    FakeInitWarp,
    FakeWarpRuntime,
    RecordingFactoryConstructor,
    RecordingOptOut,
    RecordingWriter,
    SteppingClock,
)

#: Devices the fake runtime recognises. ``cuda:0`` reports a label unlike its
#: identifier so a test can tell the two apart in a record.
DEVICES = {
    "cuda:0": "NVIDIA GeForce RTX 3090 Ti",
    "cuda:1": "NVIDIA GeForce GTX 1630",
    "cpu": "cpu",
}

#: Seconds the stepping clock advances per reading.
CLOCK_STEP = 0.25


class Harness:
    """Everything a script test needs to drive one run.

    Args:
        runtime: The Warp stand-in the script was given.
        init_warp: The initialiser, recording the configuration asked for.
        construct: The factory constructor, recording its arguments.
        writer: Everything the script wrote.
        opt_out: Records that power throttling was opted out of.
    """

    def __init__(
        self,
        runtime: FakeWarpRuntime,
        init_warp: FakeInitWarp,
        construct: RecordingFactoryConstructor,
        writer: RecordingWriter,
        opt_out: RecordingOptOut,
    ) -> None:
        self.runtime = runtime
        self.init_warp = init_warp
        self.construct = construct
        self.writer = writer
        self.opt_out = opt_out


def install(deterministic: bool) -> Harness:
    """Rebind every script hook to a fake and return them.

    Args:
        deterministic: Whether the simulators built produce agreeing rollouts.

    Returns:
        The installed harness.
    """
    runtime = FakeWarpRuntime(DEVICES)
    init_warp = FakeInitWarp(runtime)
    construct = RecordingFactoryConstructor(runtime, deterministic)
    writer = RecordingWriter()
    opt_out = RecordingOptOut()
    _test_hooks.opt_out_of_power_throttling = opt_out
    _test_hooks.init_warp = init_warp
    _test_hooks.load_state_factory = lambda: construct
    _test_hooks.write_out = writer
    _test_hooks.monotonic = SteppingClock(CLOCK_STEP)
    return Harness(runtime, init_warp, construct, writer, opt_out)


@pytest.fixture()
def harness() -> Generator[Harness, None, None]:
    """Install fakes producing deterministic rollouts, then restore.

    Yields:
        The installed harness.
    """
    saved = (
        _test_hooks.init_warp,
        _test_hooks.load_state_factory,
        _test_hooks.write_out,
        _test_hooks.monotonic,
        _test_hooks.opt_out_of_power_throttling,
    )
    yield install(deterministic=True)
    (
        _test_hooks.init_warp,
        _test_hooks.load_state_factory,
        _test_hooks.write_out,
        _test_hooks.monotonic,
        _test_hooks.opt_out_of_power_throttling,
    ) = saved


@pytest.fixture()
def drifting_harness() -> Generator[Harness, None, None]:
    """Install fakes producing rollouts that disagree, then restore.

    The negative control: a script must report what the instrument found, not
    what the script hoped for.

    Yields:
        The installed harness.
    """
    saved = (
        _test_hooks.init_warp,
        _test_hooks.load_state_factory,
        _test_hooks.write_out,
        _test_hooks.monotonic,
        _test_hooks.opt_out_of_power_throttling,
    )
    yield install(deterministic=False)
    (
        _test_hooks.init_warp,
        _test_hooks.load_state_factory,
        _test_hooks.write_out,
        _test_hooks.monotonic,
        _test_hooks.opt_out_of_power_throttling,
    ) = saved
