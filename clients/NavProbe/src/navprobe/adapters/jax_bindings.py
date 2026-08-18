"""Typed boundary onto JAX: its arrays, its transforms, and its NumPy surface.

Separated from the MJX boundary because they are different vendors with
different release cycles. JAX owns arrays and the ``jit``/``vmap`` transforms;
MJX owns models, state, and stepping. A single module holding both would make
every JAX signature change look like an MJX change, and vice versa.

The dependency runs one way: :mod:`navprobe.adapters.mjx_bindings` imports the
array Protocols from here, because MJX's state fields *are* JAX arrays. Nothing
here knows MJX exists.

``jax`` does ship type information. It is bound the same way as the untyped
vendors anyway, so that each function is declared with the signature it is
actually called with rather than inherited from stubs whose ``tolist`` returns
an untyped value — which would put ``Any`` back into a package that forbids it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

#: Import path of JAX.
JAX_MODULE = "jax"

#: Import path of JAX's NumPy surface.
JAX_NUMPY_MODULE = "jax.numpy"


class FlatArrayProtocol(Protocol):
    """A one-dimensional device array."""

    def tolist(self) -> list[float]:
        """Copy the array to Python floats.

        Returns:
            One float per element, in the array's own order.
        """
        ...


class BatchedArrayProtocol(Protocol):
    """A two-dimensional device array indexed by world, then by element."""

    def tolist(self) -> list[list[float]]:
        """Copy the array to Python floats.

        Returns:
            One list of floats per world, in world order.
        """
        ...


class AsArrayProtocol(Protocol):
    """``jax.numpy.asarray``, for the batched position array."""

    def __call__(self, a: Sequence[Sequence[float]]) -> BatchedArrayProtocol:
        """Convert nested Python floats to a device array.

        Args:
            a: One sequence of positions per world.

        Returns:
            The device array.
        """
        ...


class JaxNumpyModuleProtocol(Protocol):
    """JAX's NumPy surface.

    Attributes:
        asarray: Converts host values to a device array.
    """

    asarray: AsArrayProtocol


def load_jax_numpy() -> JaxNumpyModuleProtocol:
    """Load JAX's NumPy surface behind its Protocol.

    Returns:
        The module, typed by the annotation rather than by the import.
    """
    module: JaxNumpyModuleProtocol = __import__(JAX_NUMPY_MODULE, fromlist=["asarray"])
    return module


__all__ = [
    "JAX_MODULE",
    "JAX_NUMPY_MODULE",
    "AsArrayProtocol",
    "BatchedArrayProtocol",
    "FlatArrayProtocol",
    "JaxNumpyModuleProtocol",
    "load_jax_numpy",
]
