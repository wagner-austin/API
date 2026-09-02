"""A Warp stand-in the measurement scripts can be driven against.

These are not mocks. A mock would record that ``ScopedDevice`` was called and
return whatever the test wanted; what the scripts actually have to get right is
that work happens *inside* the device scope, that an unknown device fails before
any of it, and that the record carries the device that ran rather than the name
that was typed. So this module implements a device registry that behaves like
Warp's in the ways the scripts depend on, and the tests let the scripts reach
their own conclusions about it.

The simulators the scripts sweep are the real ones in :mod:`tests.simulators`,
reached through :mod:`tests.factories`, so a sweep driven here produces genuine
determinism verdicts rather than canned ones.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager

from scripts._test_hooks import (
    GetDeviceProtocol,
    ScopedDeviceProtocol,
    WarpRuntimeProtocol,
)

from navprobe.experiment import SimulatorFactoryProtocol
from navprobe.rollout import SimulatorProtocol
from tests.factories import DriftingSimulatorFactory, LinearSimulatorFactory


class FakeDevice:
    """A resolved device, carrying the label Warp would report.

    Args:
        label: The device's string form.
    """

    def __init__(self, label: str) -> None:
        self._label = label

    def __str__(self) -> str:
        """Return the device's label.

        Returns:
            The label this device was built with.
        """
        return self._label


class FakeWarpRuntime:
    """A Warp module stand-in with a fixed set of devices.

    Args:
        devices: Identifiers this runtime recognises, mapped to the labels it
            reports for them. A label that differs from its identifier is the
            interesting case: it is what proves a report records the card that
            ran rather than the name that was asked for.
    """

    def __init__(self, devices: dict[str, str]) -> None:
        self._devices = devices
        self.scopes_entered: list[str] = []
        self.scope_depth = 0
        self.work_inside_scope: list[str] = []
        # Both hooks are bound as instance attributes rather than defined as
        # methods. The Protocol declares them as settable variables, which is
        # what they are on the real Warp module -- a class method would be a
        # read-only attribute and would not satisfy it. ``ScopedDevice`` has a
        # second reason: Warp's spelling is not a legal method name under the
        # naming rules this repo enforces.
        self.ScopedDevice: ScopedDeviceProtocol = self._scoped_device
        self.get_device: GetDeviceProtocol = self._get_device

    def _get_device(self, ident: str) -> FakeDevice:
        """Resolve a device identifier.

        Args:
            ident: The identifier to resolve.

        Returns:
            The resolved device.

        Raises:
            ValueError: When this runtime does not have that device, with the
                message Warp itself uses, so a caller that matched on the text
                would behave identically against both.
        """
        if ident not in self._devices:
            raise ValueError(f"Invalid device identifier: {ident}")
        return FakeDevice(self._devices[ident])

    @contextmanager
    def _scoped_device(self, ident: str) -> Generator[FakeDevice, None, None]:
        """Scope enclosed work to a device.

        Exposed as ``ScopedDevice`` by :meth:`__init__`, which is the name the
        Protocol and the vendor both use.

        Args:
            ident: The device to scope to.

        Yields:
            The scoped device.
        """
        self.scopes_entered.append(ident)
        self.scope_depth += 1
        try:
            yield self._get_device(ident)
        finally:
            self.scope_depth -= 1


class RecordingFactoryConstructor:
    """Builds real simulator factories, recording the arguments it was given.

    The recorded arguments are how a test asserts that a script passed its
    perturbation and its capacity through rather than defaulting them, which is
    the failure that produced a silently truncated solve once already.

    Args:
        runtime: The runtime whose scope depth is sampled on each call, so a
            test can assert construction happened inside the device scope.
        deterministic: Whether the factories built produce agreeing simulators.
    """

    def __init__(self, runtime: FakeWarpRuntime, deterministic: bool) -> None:
        self._runtime = runtime
        self._deterministic = deterministic
        self.calls: list[tuple[str, int, float, int]] = []
        #: The block-size pin each call carried, in call order. Kept beside
        #: ``calls`` rather than widened into it so that adding a settable
        #: condition did not rewrite every existing assertion on the tuple --
        #: and so a test can assert a script passed ``None`` rather than
        #: silently imposing a value, which is the failure that matters here.
        self.linesearch_block_dims: list[int | None] = []

    def __call__(
        self,
        model_xml: str,
        world_count: int,
        perturbation: float,
        constraint_capacity: int,
        linesearch_block_dim: int | None = None,
    ) -> SimulatorFactoryProtocol:
        """Build a factory for one compiled scene.

        Args:
            model_xml: The compiled scene's MJCF document.
            world_count: Parallel worlds each simulator carries.
            perturbation: Half-width of the seed-driven offset range.
            constraint_capacity: Constraint allocation bound.
            linesearch_block_dim: Block size pinned on the line-search kernel,
                or ``None`` for the vendor default.

        Returns:
            A factory producing real simulators.
        """
        self.calls.append((model_xml, world_count, perturbation, constraint_capacity))
        self.linesearch_block_dims.append(linesearch_block_dim)
        self._runtime.work_inside_scope.append(f"construct:{self._runtime.scope_depth}")
        if self._deterministic:
            return LinearSimulatorFactory(world_count=world_count)
        return DriftingSimulatorFactory(world_count=world_count, diverge_at_step=3)


class WitnessSimulator:
    """A real simulator with a contact count bolted on.

    The contact count is configured rather than simulated, and that is the
    point: the case worth testing is a simulator whose rollouts agree
    perfectly while reporting ZERO contacts, because that is the shape the
    convex-narrowphase failure takes and the one a verdict alone calls a pass.
    A simulator that derived its contacts from its own motion could not be
    asked to produce it.

    Args:
        inner: The simulator whose rollout behaviour is being wrapped.
        contacts_per_step: What :meth:`contact_count` reports after each step.
    """

    def __init__(self, inner: SimulatorProtocol, contacts_per_step: int) -> None:
        self._inner = inner
        self._contacts_per_step = contacts_per_step

    @property
    def world_count(self) -> int:
        """Number of parallel worlds the wrapped simulator carries.

        Returns:
            The inner simulator's world count.
        """
        return self._inner.world_count

    def reset(self, seed: int) -> None:
        """Reset the wrapped simulator.

        Args:
            seed: The seed to pin.
        """
        self._inner.reset(seed)

    def advance(self) -> Sequence[float]:
        """Advance the wrapped simulator one step.

        Returns:
            The inner simulator's observation.
        """
        return self._inner.advance()

    def contact_count(self) -> int:
        """Report the configured contact count.

        Returns:
            Contacts for the step just taken.
        """
        return self._contacts_per_step


class WitnessSimulatorFactory:
    """Builds witness-capable simulators over a real factory.

    Args:
        inner: The factory producing the simulators to wrap.
        contacts_per_step: What each built simulator reports per step.
    """

    def __init__(self, inner: SimulatorFactoryProtocol, contacts_per_step: int) -> None:
        self._inner = inner
        self._contacts_per_step = contacts_per_step

    def __call__(self) -> WitnessSimulator:
        """Construct one simulator.

        Returns:
            A freshly wrapped simulator.
        """
        return WitnessSimulator(self._inner(), self._contacts_per_step)


class RecordingWitnessFactoryConstructor:
    """Builds witness-capable factories, recording the arguments it was given.

    The witness twin of :class:`RecordingFactoryConstructor`. It exists rather
    than widening that one because the two hooks are deliberately separate: a
    sweep that only needs the vendor-agnostic surface should not be handed a
    capability its scenes may not have.

    Args:
        runtime: The runtime whose scope depth is sampled on each call.
        deterministic: Whether the factories built produce agreeing simulators.
        contacts_per_step: Contacts each built simulator reports. Zero models a
            scene that has stopped interacting.
    """

    def __init__(
        self, runtime: FakeWarpRuntime, deterministic: bool, contacts_per_step: int
    ) -> None:
        self._runtime = runtime
        self._deterministic = deterministic
        self._contacts_per_step = contacts_per_step
        self.calls: list[tuple[str, int, float, int]] = []
        self.linesearch_block_dims: list[int | None] = []

    def __call__(
        self,
        model_xml: str,
        world_count: int,
        perturbation: float,
        constraint_capacity: int,
        linesearch_block_dim: int | None = None,
    ) -> WitnessSimulatorFactory:
        """Build a witness-capable factory for one scene.

        Args:
            model_xml: The scene's MJCF document.
            world_count: Parallel worlds each simulator carries.
            perturbation: Half-width of the seed-driven offset range.
            constraint_capacity: Constraint allocation bound.
            linesearch_block_dim: Block size pinned on the line-search kernel,
                or ``None`` for the vendor default.

        Returns:
            A factory producing witness-capable simulators.
        """
        self.calls.append((model_xml, world_count, perturbation, constraint_capacity))
        self.linesearch_block_dims.append(linesearch_block_dim)
        self._runtime.work_inside_scope.append(f"construct:{self._runtime.scope_depth}")
        inner: SimulatorFactoryProtocol = (
            LinearSimulatorFactory(world_count=world_count)
            if self._deterministic
            else DriftingSimulatorFactory(world_count=world_count, diverge_at_step=3)
        )
        return WitnessSimulatorFactory(inner, self._contacts_per_step)


class RecordingWriter:
    """Collects everything a script writes out.

    Args:
        None.
    """

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def __call__(self, text: str) -> None:
        """Record one written chunk.

        Args:
            text: The text the script wrote.
        """
        self.chunks.append(text)

    @property
    def text(self) -> str:
        """Everything written, concatenated.

        Returns:
            The full output stream.
        """
        return "".join(self.chunks)

    def documents(self, banner: str) -> str:
        """Extract the one document beginning with ``banner``.

        Args:
            banner: The banner line the document starts with.

        Returns:
            The document text from its banner to the end of the stream.

        Raises:
            AssertionError: When no chunk starts with that banner. A test that
                silently got no document would pass its later assertions
                vacuously.
        """
        for chunk in self.chunks:
            if chunk.startswith(banner):
                return chunk
        raise AssertionError(f"no document with banner {banner!r} in {self.chunks!r}")


class RecordingOptOut:
    """Records that the power-throttling opt-out was requested.

    A fake rather than the real call because the real one alters the *test
    worker's* power state, which would leak into every later test in that
    worker and quietly change what the suite is measured under.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        """Record one request."""
        self.calls += 1


class SteppingClock:
    """A monotonic clock that advances a fixed amount per reading.

    A real clock would make every wall-time assertion a statement about the
    machine the suite ran on. This one makes elapsed time exact, so a test can
    assert the throughput a record carries.

    Args:
        step: Seconds added between consecutive readings.
    """

    def __init__(self, step: float) -> None:
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        """Read the clock and advance it.

        Returns:
            The current reading, before advancing.
        """
        current = self._now
        self._now += self._step
        return current


class FakeInitWarp:
    """Returns a prepared runtime, recording the configuration asked for.

    Args:
        runtime: The runtime to hand back.
    """

    def __init__(self, runtime: FakeWarpRuntime) -> None:
        self._runtime = runtime
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, mode_name: str, cache_dir: str, max_records: int) -> WarpRuntimeProtocol:
        """Bring the runtime up.

        Declared as the Protocol rather than as the concrete fake, so this
        satisfies :class:`scripts._test_hooks.InitWarpProtocol` structurally --
        which is the point of the fake, and would silently stop being true if
        the Protocol's return type changed.

        Args:
            mode_name: Determinism mode asked for.
            cache_dir: Kernel-cache directory asked for.
            max_records: Deterministic record bound asked for.

        Returns:
            The prepared runtime.
        """
        self.calls.append((mode_name, cache_dir, max_records))
        return self._runtime


def scene_body_counts(document: str, tag: str) -> Sequence[str]:
    """Read the leading field of every tagged row in a document.

    Args:
        document: The encoded document.
        tag: The row tag to select.

    Returns:
        The second token of each matching row, in order.
    """
    return [
        line.split("\t")[1]
        for line in document.strip("\n").split("\n")
        if line.split("\t")[0] == tag
    ]
