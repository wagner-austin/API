"""Dependency-injection hooks for the repo-local measurement scripts.

Same discipline as :mod:`navprobe._test_hooks`: every symbol is bound to its
real implementation at import time and called unconditionally. Production wires
nothing; tests rebind and restore. There is no ``if hook is not None`` branch
anywhere, because a conditional hook is a second code path that production never
exercises.

Vendors are bound the way :mod:`navprobe.adapters.mjx_warp_bindings` binds them:
``__import__`` behind a Protocol annotation, so the untyped module is typed by
the declaration rather than by the import, with no cast and no ``Any``.

Only genuine boundary effects belong here, and for these scripts there are four.

**Warp initialisation is one effect, not three.** ``wp.config.deterministic``
must be set before ``wp.init()``, and ``mujoco_warp`` must be imported after it,
because Warp lowers a module's atomics during codegen at import time. Splitting
that into separate hooks would let a caller wire them in the wrong order and get
a silently non-deterministic run -- the exact failure this package exists to
detect. The hook takes the settings and returns the initialised module, so the
order lives in one place.

**The post-init imports are hooks** for the same reason: they must not happen
before the configuration is applied.

**The clock is a hook** because wall time is an output of these scripts, and a
test asserting on a real clock would be asserting on the machine it ran on.
"""

from __future__ import annotations

import sys
import time
from contextlib import AbstractContextManager
from typing import Protocol

from navprobe.experiment import SimulatorFactoryProtocol

#: Vendor modules, named here so the import sites read as one word.
WARP_MODULE = "warp"


class WarpDeviceProtocol(Protocol):
    """A resolved Warp device.

    Declared for the label alone. These scripts never call a method on a
    device: they resolve one to prove it exists and to record which card
    produced the numbers, and Warp's own ``__str__`` is that record.
    """

    def __str__(self) -> str:
        """Return the device's label, such as ``cuda:0``.

        Returns:
            The label Warp reports for this device.
        """
        ...


class DeterminismModeProtocol(Protocol):
    """One member of Warp's ``DeterministicMode`` enumeration."""

    name: str


class DeterministicModeEnumProtocol(Protocol):
    """Warp's ``DeterministicMode`` enumeration.

    Attributes:
        RUN_TO_RUN: Reproducible across runs on one device.
        GPU_TO_GPU: Reproducible across devices as well.
    """

    RUN_TO_RUN: DeterminismModeProtocol
    GPU_TO_GPU: DeterminismModeProtocol


class GetDeviceProtocol(Protocol):
    """Resolve a Warp device identifier."""

    def __call__(self, ident: str) -> WarpDeviceProtocol:
        """Resolve one device.

        Args:
            ident: A Warp device identifier such as ``cuda:0``.

        Returns:
            The device, whose string form labels the run.

        Raises:
            ValueError: When Warp does not recognise the identifier. Callers
                resolve eagerly so an absent device fails before a long cold
                compile rather than after it.
        """
        ...


class ScopedDeviceProtocol(Protocol):
    """Scope enclosed work to one Warp device."""

    def __call__(self, ident: str) -> AbstractContextManager[WarpDeviceProtocol]:
        """Enter a device scope.

        Args:
            ident: A Warp device identifier.

        Returns:
            A context manager making ``ident`` the current device.
        """
        ...


class WarpConfigProtocol(Protocol):
    """The Warp configuration these scripts set before initialisation.

    Attributes:
        kernel_cache_dir: Directory Warp writes compiled kernels to.
        deterministic: The determinism mode, a ``wp.DeterministicMode`` member.
        deterministic_max_records: Upper bound on deterministic scatter
            records per thread. Zero means Warp's code-generated lower bound.
    """

    kernel_cache_dir: str
    deterministic: DeterminismModeProtocol
    deterministic_max_records: int


class WarpInitProtocol(Protocol):
    """Initialise Warp, after configuration is applied."""

    def __call__(self) -> None:
        """Initialise the runtime and enumerate devices."""
        ...


class WarpRuntimeProtocol(Protocol):
    """Warp as the *scripts* use it, once it is up.

    Deliberately only two members. A script resolves a device and scopes work
    to it; it never touches the configuration, because configuring after
    initialisation would be too late to matter. Keeping the returned type this
    narrow is what lets a test supply an object with two attributes instead of
    impersonating a vendor module -- and it means a script that started
    reaching for ``config`` would not typecheck, which is the intent.

    Attributes:
        get_device: Resolves a device identifier.
        ScopedDevice: Scopes work to a device. Named as Warp names it, because
            a binding that renamed a vendor symbol would no longer document
            what the vendor is called.
    """

    get_device: GetDeviceProtocol
    ScopedDevice: ScopedDeviceProtocol


class WarpModuleProtocol(WarpRuntimeProtocol, Protocol):
    """Warp as the *initialiser* uses it: the runtime plus its configuration.

    Warp ships type information, so binding it through ``__import__`` is not
    about an untyped vendor here. It is about pinning ``init``'s signature,
    which Warp's own stubs leave untyped and which strict mypy therefore
    refuses to call directly.

    Attributes:
        config: The settings applied before initialisation.
        DeterministicMode: Enumeration of Warp's determinism modes.
        init: Initialises the runtime.
    """

    config: WarpConfigProtocol
    DeterministicMode: DeterministicModeEnumProtocol
    init: WarpInitProtocol


class InitWarpProtocol(Protocol):
    """Import Warp, apply a determinism configuration, and initialise it."""

    def __call__(self, mode_name: str, cache_dir: str, max_records: int) -> WarpRuntimeProtocol:
        """Bring Warp up under one determinism configuration.

        Args:
            mode_name: ``NOT_GUARANTEED``, ``RUN_TO_RUN`` or ``GPU_TO_GPU``.
                Anything but ``NOT_GUARANTEED`` is set on
                ``wp.config.deterministic`` before initialisation.
            cache_dir: Kernel-cache directory for this run. A fresh directory
                forces cold codegen, which is what makes a compile-gate result
                mean anything.
            max_records: ``wp.config.deterministic_max_records``. Zero leaves
                Warp's code-generated lower bound in place, which the solver's
                data-dependent contact loops exceed at 32 bodies.

        Returns:
            The initialised module.
        """
        ...


class StateFactoryConstructorProtocol(Protocol):
    """Construct a MuJoCo-Warp state simulator factory for one scene."""

    def __call__(
        self, model_xml: str, world_count: int, perturbation: float, constraint_capacity: int
    ) -> SimulatorFactoryProtocol:
        """Build the factory.

        Args:
            model_xml: The compiled scene's MJCF document.
            world_count: Parallel worlds each simulator carries.
            perturbation: Half-width of the seed-driven initial offset range.
            constraint_capacity: Upper bound on constraints, contacts and
                Jacobian non-zeros the allocation reserves.

        Returns:
            A factory producing freshly constructed simulators for that scene.
        """
        ...


class LoadStateFactoryProtocol(Protocol):
    """Load the MuJoCo-Warp state adapter, after Warp is initialised."""

    def __call__(self) -> StateFactoryConstructorProtocol:
        """Return the adapter's factory constructor.

        Returns:
            The constructor, typed by this declaration rather than by the
            import it comes from.
        """
        ...


class WriteOutProtocol(Protocol):
    """Write report text to standard output."""

    def __call__(self, text: str) -> None:
        """Write ``text`` verbatim.

        Args:
            text: Text to write, including its own trailing newline.
        """
        ...


class MonotonicProtocol(Protocol):
    """Read a monotonic clock, in seconds."""

    def __call__(self) -> float:
        """Return the current value of a monotonic clock.

        Returns:
            Seconds from an unspecified origin.
        """
        ...


def _init_warp_impl(mode_name: str, cache_dir: str, max_records: int) -> WarpRuntimeProtocol:
    """Production implementation of :class:`InitWarpProtocol`.

    Args:
        mode_name: Determinism mode to configure before initialisation.
        cache_dir: Kernel-cache directory for this run.
        max_records: Deterministic record bound, zero for Warp's own.

    Returns:
        The initialised module.
    """
    module: WarpModuleProtocol = __import__(
        WARP_MODULE,
        fromlist=["config", "DeterministicMode", "init", "get_device", "ScopedDevice"],
    )
    module.config.kernel_cache_dir = cache_dir
    if mode_name != "NOT_GUARANTEED":
        module.config.deterministic = getattr(module.DeterministicMode, mode_name)
    if max_records:
        module.config.deterministic_max_records = max_records
    module.init()
    return module


def _load_state_factory_impl() -> StateFactoryConstructorProtocol:
    """Production implementation of :class:`LoadStateFactoryProtocol`.

    Imported here rather than at module scope because ``mujoco_warp`` lowers its
    kernels during import, so it must not be imported before
    :func:`_init_warp_impl` has applied the determinism configuration.

    Returns:
        The adapter's factory constructor.
    """
    from navprobe.adapters.mjx_warp_state import MjWarpStateSimulatorFactory

    constructor: StateFactoryConstructorProtocol = MjWarpStateSimulatorFactory
    return constructor


def _write_out_impl(text: str) -> None:
    """Production implementation of :class:`WriteOutProtocol`.

    Args:
        text: Text to write, including its own trailing newline.
    """
    sys.stdout.write(text)
    sys.stdout.flush()


init_warp: InitWarpProtocol = _init_warp_impl
load_state_factory: LoadStateFactoryProtocol = _load_state_factory_impl
write_out: WriteOutProtocol = _write_out_impl
monotonic: MonotonicProtocol = time.perf_counter


__all__ = [
    "DeterminismModeProtocol",
    "DeterministicModeEnumProtocol",
    "GetDeviceProtocol",
    "InitWarpProtocol",
    "LoadStateFactoryProtocol",
    "MonotonicProtocol",
    "ScopedDeviceProtocol",
    "StateFactoryConstructorProtocol",
    "WarpConfigProtocol",
    "WarpDeviceProtocol",
    "WarpInitProtocol",
    "WarpModuleProtocol",
    "WarpRuntimeProtocol",
    "WriteOutProtocol",
    "init_warp",
    "load_state_factory",
    "monotonic",
    "write_out",
]
