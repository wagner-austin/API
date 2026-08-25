"""Dependency-injection hooks for the repo-local measurement scripts.

Same discipline as :mod:`navprobe._test_hooks`: every symbol is bound to its
real implementation at import time and called unconditionally. Production wires
nothing; tests rebind and restore. There is no ``if hook is not None`` branch
anywhere, because a conditional hook is a second code path that production never
exercises.

Vendors are bound the way :mod:`navprobe.adapters.mjx_warp_bindings` binds them:
``__import__`` behind a Protocol annotation, so the untyped module is typed by
the declaration rather than by the import, with no cast and no ``Any``.

Only genuine boundary effects belong here, and for these scripts there are five.

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

**Power-throttling opt-out is a hook** for the same reason the clock is: it
decides what the clock will read. Windows classifies a long-running console
process as background work and applies EcoQoS throttling to it a few seconds
in; the demotion lands mid-measurement and never lifts, so a sweep that does
not opt out reports two power regimes as one. It cannot be detected by
querying -- a throttled process and one that opted out report identical state
-- so it is applied unconditionally rather than checked.

That last hook is a **deliberate duplication**, and it should not stay one. The
canonical implementation is ``covenant_ml.benchmarking.power``, written
2026-08-19 by the session that root-caused the effect and holds the evidence
for it (the same fit repeated in one process going 0.547s to 7.108s, with leak,
thread growth and thermal recovery all ruled out). NavProbe cannot import it:
``covenant_ml`` depends on xgboost, lightgbm, optuna, polars and scikit-learn,
and a physics-determinism client does not take an ML stack to make one Win32
call. The right end state is a small shared package both depend on. It is not
done here because that package's owner is actively working in that tree, and
restructuring a live one to save thirty lines is how a merge conflict becomes
someone else's afternoon. Proposed on the agent board rather than left silent.
"""

from __future__ import annotations

import ctypes
import sys
import time
from contextlib import AbstractContextManager
from ctypes import wintypes
from typing import Protocol

from navprobe import NavProbeError
from navprobe.experiment import SimulatorFactoryProtocol

#: Vendor modules, named here so the import sites read as one word.
WARP_MODULE = "warp"

#: ``ProcessPowerThrottling`` from ``PROCESS_INFORMATION_CLASS``.
PROCESS_POWER_THROTTLING = 4

#: ``PROCESS_POWER_THROTTLING_EXECUTION_SPEED``.
EXECUTION_SPEED = 0x1

#: ``PROCESS_POWER_THROTTLING_CURRENT_VERSION``.
STATE_VERSION = 1

#: What ``GetCurrentProcess`` returns: the documented ``(HANDLE)-1``
#: pseudo-handle for the calling process. Passed directly rather than fetched
#: -- one fewer untyped boundary, and it cannot be truncated to 32 bits.
CURRENT_PROCESS_PSEUDO_HANDLE = -1


class PowerThrottlingError(NavProbeError):
    """This process could not be opted out of power throttling.

    Args:
        code: Stable identifier in the ``NP-POWER-<NNN>`` range.
        message: Human-readable description, carrying the Win32 error code.
    """


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
        self,
        model_xml: str,
        world_count: int,
        perturbation: float,
        constraint_capacity: int,
        linesearch_block_dim: int | None = None,
    ) -> SimulatorFactoryProtocol:
        """Build the factory.

        Args:
            model_xml: The compiled scene's MJCF document.
            world_count: Parallel worlds each simulator carries.
            perturbation: Half-width of the seed-driven initial offset range.
            constraint_capacity: Upper bound on constraints, contacts and
                Jacobian non-zeros the allocation reserves.
            linesearch_block_dim: CUDA block size to pin the iterative
                line-search kernel to, or ``None`` for the vendor default.
                Optional so the sweeps that predate the block-size finding call
                this unchanged; declared because leaving it out would mean a
                script could not pin the one setting that decides whether a
                coupled-body scene reproduces.

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


class SetProcessInformationProtocol(Protocol):
    """The ``kernel32!SetProcessInformation`` foreign function.

    ``ctypes`` types a DLL attribute as a function pointer whose call returns
    an untyped value; assigning it to this Protocol is where a concrete return
    type comes from, the same way the Warp module is typed above.

    The parameters are spelled as the exact ``ctypes`` instances the call is
    made with rather than as a loose varargs signature, which is what removes
    the need for an ``argtypes`` declaration: each argument marshals at its own
    declared width, so nothing falls back to a default that would truncate a
    64-bit handle to 32 bits.
    """

    def __call__(
        self,
        process: ctypes.c_void_p,
        info_class: ctypes.c_int,
        information: ctypes.c_void_p,
        size: ctypes.c_uint32,
    ) -> int:
        """Set one class of information on a process.

        Args:
            process: Handle to the target process.
            info_class: Which ``PROCESS_INFORMATION_CLASS`` is being set.
            information: Pointer to the class's state structure.
            size: Byte length of that structure.

        Returns:
            Non-zero on success, zero on failure.
        """
        ...


class ProcessInformationSetterProtocol(Protocol):
    """The Win32 boundary that sets this process's power state.

    Carries the three mask fields as plain integers rather than a ``ctypes``
    structure, so building the structure stays inside the one function that
    talks to Win32 and no test reaches through a field descriptor.
    """

    def __call__(self, version: int, control_mask: int, state_mask: int) -> int:
        """Apply a power-throttling state to the current process.

        Args:
            version: ``PROCESS_POWER_THROTTLING_STATE.Version``.
            control_mask: Which policies the process expresses a preference
                about.
            state_mask: The preference itself, for the policies named by
                ``control_mask``.

        Returns:
            The Win32 error code, or zero when the request was accepted.
        """
        ...


class OptOutOfPowerThrottlingProtocol(Protocol):
    """Opt this process out of system-managed power throttling."""

    def __call__(self) -> None:
        """Make the request.

        Returns:
            None. The call is made for its effect on the process.

        Raises:
            PowerThrottlingError: When the platform refuses.
        """
        ...


class PowerThrottlingState(ctypes.Structure):
    """``PROCESS_POWER_THROTTLING_STATE`` from ``processthreadsapi.h``.

    The pair of masks encodes three distinct requests, and the difference
    between two of them is the whole point:

    * ``ControlMask = 0`` -- no preference, Windows decides. The default, and
      the one that throttles.
    * ``ControlMask = EXECUTION_SPEED``, ``StateMask = EXECUTION_SPEED`` --
      always throttle.
    * ``ControlMask = EXECUTION_SPEED``, ``StateMask = 0`` -- never throttle.

    Reading the state back does not reveal whether the process is *currently*
    throttled: a default-managed process reports ``StateMask = 0``, identical
    to one that has explicitly opted out. The condition is detectable only by
    timing, which is why this is applied unconditionally rather than checked.
    """

    _fields_ = (
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    )


def win32_process_information_setter(version: int, control_mask: int, state_mask: int) -> int:
    """Production implementation of :class:`ProcessInformationSetterProtocol`.

    Args:
        version: Structure version.
        control_mask: Policies being expressed a preference about.
        state_mask: The preference itself.

    Returns:
        The Win32 error code, or zero when the request was accepted.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_process_information: SetProcessInformationProtocol = kernel32.SetProcessInformation
    state = PowerThrottlingState(Version=version, ControlMask=control_mask, StateMask=state_mask)
    accepted = set_process_information(
        ctypes.c_void_p(CURRENT_PROCESS_PSEUDO_HANDLE),
        ctypes.c_int(PROCESS_POWER_THROTTLING),
        ctypes.c_void_p(ctypes.addressof(state)),
        ctypes.c_uint32(ctypes.sizeof(state)),
    )
    if accepted != 0:
        return 0
    return ctypes.get_last_error()


def disable_power_throttling(setter: ProcessInformationSetterProtocol) -> None:
    """Request "never throttle this process".

    Requests ``ControlMask = EXECUTION_SPEED`` with ``StateMask = 0``. Setting
    both masks would request the exact opposite, one bit away, so the encoding
    is asserted in tests rather than left to review.

    Args:
        setter: Applies the state. Injected so the refusal path is reachable
            in tests without altering the host's power state.

    Returns:
        None. The call is made for its effect on the process.

    Raises:
        PowerThrottlingError: When the request is refused. Raised rather than
            ignored, with no fallback: a sweep that could not opt out is
            timing an unknown mix of two power regimes, and a wall clock
            nobody can attribute is worse than no wall clock.
    """
    code = setter(STATE_VERSION, EXECUTION_SPEED, 0)
    if code != 0:
        raise PowerThrottlingError(
            "NP-POWER-001",
            f"could not opt out of process power throttling (win32 {code}); "
            f"wall-clock figures would mix two power regimes",
        )


def _opt_out_of_power_throttling_impl() -> None:
    """Production implementation of :class:`OptOutOfPowerThrottlingProtocol`.

    Returns:
        None. The call is made for its effect on the process.

    Raises:
        PowerThrottlingError: When the platform refuses.
    """
    disable_power_throttling(win32_process_information_setter)


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
opt_out_of_power_throttling: OptOutOfPowerThrottlingProtocol = _opt_out_of_power_throttling_impl
write_out: WriteOutProtocol = _write_out_impl
monotonic: MonotonicProtocol = time.perf_counter


__all__ = [
    "CURRENT_PROCESS_PSEUDO_HANDLE",
    "EXECUTION_SPEED",
    "PROCESS_POWER_THROTTLING",
    "STATE_VERSION",
    "DeterminismModeProtocol",
    "DeterministicModeEnumProtocol",
    "GetDeviceProtocol",
    "InitWarpProtocol",
    "LoadStateFactoryProtocol",
    "MonotonicProtocol",
    "OptOutOfPowerThrottlingProtocol",
    "PowerThrottlingError",
    "PowerThrottlingState",
    "ProcessInformationSetterProtocol",
    "ScopedDeviceProtocol",
    "SetProcessInformationProtocol",
    "StateFactoryConstructorProtocol",
    "WarpConfigProtocol",
    "WarpDeviceProtocol",
    "WarpInitProtocol",
    "WarpModuleProtocol",
    "WarpRuntimeProtocol",
    "WriteOutProtocol",
    "disable_power_throttling",
    "init_warp",
    "load_state_factory",
    "monotonic",
    "opt_out_of_power_throttling",
    "win32_process_information_setter",
    "write_out",
]
