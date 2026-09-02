"""Typed boundary onto MuJoCo itself, shared by every adapter.

Both backends this package drives — MJX on JAX and MuJoCo-Warp — start from the
same place: an MJCF document compiled by ``mujoco.MjModel``. That step is the
same vendor, the same function, and the same signature for both, so it is
declared once here rather than in each adapter's own bindings.

The alternative was two declarations of ``from_xml_string``, free to drift into
disagreeing about a signature only one of them had checked. A single declaration
means the drift test for it covers every adapter that compiles a model.

``mujoco`` ships no ``py.typed`` marker, so the module is bound by assigning the
import to the Protocol, which is where the type comes from.
"""

from __future__ import annotations

from typing import Protocol

#: Import path of the MuJoCo top-level module.
MUJOCO_MODULE = "mujoco"


class MjModelProtocol(Protocol):
    """A compiled MuJoCo model.

    Attributes:
        nq: Number of generalised position coordinates.
    """

    nq: int


class MjDataProtocol(Protocol):
    """Simulation state for one compiled model.

    Declared for one field. This package's adapters drive their solvers through
    the vendor backends, not through MuJoCo's own stepper -- what MuJoCo itself
    is needed for is the question "does this scene start in contact at all",
    which no amount of reading the MJCF answers.

    That question is not academic. A scene whose bodies never touch produces
    zero contacts in EVERY determinism mode, which is indistinguishable at a
    glance from a mode that dropped them -- and two such scenes were built by
    hand during the 2026-08-30 collision-pair work before a control caught
    them. ``ncon`` is what turns "I calculated that these overlap" into "MuJoCo
    agrees they do".

    Attributes:
        ncon: Contacts MuJoCo found in the current state.
    """

    ncon: int


class MjDataLoaderProtocol(Protocol):
    """The ``MjData`` class object, used only as a loader."""

    def __call__(self, model: MjModelProtocol) -> MjDataProtocol:
        """Allocate state for a compiled model.

        Args:
            model: The compiled model.

        Returns:
            Freshly allocated state.
        """
        ...


class ForwardProtocol(Protocol):
    """``mujoco.mj_forward``."""

    def __call__(self, m: MjModelProtocol, d: MjDataProtocol) -> None:
        """Populate derived state, including the contact list.

        Args:
            m: The compiled model.
            d: The state to populate in place.
        """
        ...


class MjModelLoaderProtocol(Protocol):
    """The ``MjModel`` class object, used only as a loader."""

    def from_xml_string(self, xml: str) -> MjModelProtocol:
        """Compile a model from MJCF XML.

        Args:
            xml: The MJCF document.

        Returns:
            The compiled model.

        Raises:
            ValueError: When the XML does not compile. MuJoCo raises this
                itself; it is named here so callers know the failure is theirs
                to expect rather than something this package converts.
        """
        ...


class MujocoModuleProtocol(Protocol):
    """The MuJoCo top-level module.

    Attributes:
        MjModel: The model class, used to compile MJCF.
        MjData: The state class, used to allocate state.
        mj_forward: Populates derived state, including contacts.
    """

    MjModel: MjModelLoaderProtocol
    MjData: MjDataLoaderProtocol
    mj_forward: ForwardProtocol


def load_mujoco() -> MujocoModuleProtocol:
    """Load the MuJoCo module behind its Protocol.

    Returns:
        The module, typed by the annotation rather than by the import.
    """
    module: MujocoModuleProtocol = __import__(
        MUJOCO_MODULE, fromlist=["MjModel", "MjData", "mj_forward"]
    )
    return module


__all__ = [
    "MUJOCO_MODULE",
    "ForwardProtocol",
    "MjDataLoaderProtocol",
    "MjDataProtocol",
    "MjModelLoaderProtocol",
    "MjModelProtocol",
    "MujocoModuleProtocol",
    "load_mujoco",
]
