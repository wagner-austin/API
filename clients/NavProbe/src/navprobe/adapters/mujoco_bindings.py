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
    """

    MjModel: MjModelLoaderProtocol


def load_mujoco() -> MujocoModuleProtocol:
    """Load the MuJoCo module behind its Protocol.

    Returns:
        The module, typed by the annotation rather than by the import.
    """
    module: MujocoModuleProtocol = __import__(MUJOCO_MODULE, fromlist=["MjModel"])
    return module


__all__ = [
    "MUJOCO_MODULE",
    "MjModelLoaderProtocol",
    "MjModelProtocol",
    "MujocoModuleProtocol",
    "load_mujoco",
]
