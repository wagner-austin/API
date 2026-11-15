"""Which NVIDIA NCCL distributions a Poetry lock installs on the same Python.

``nvidia-nccl-cu12`` and ``nvidia-nccl-cu13`` both install
``nvidia/nccl/lib/libnccl.so.2``. An environment holding one of each loses
that library the moment an install removes either of them, and every torch
import after it fails with ``ImportError: libnccl.so.2``: CI run 43700695596
at API 1a7d87bd5 failed twenty torch cases of covenant-radar-api exactly so.
torch and xgboost each bring one of the two, and which one turns on the
Python version (xgboost moves to nccl-cu13 at 3.4.0, which needs Python
3.12), so a package whose lock carries both has to keep their environment
markers apart.

A lock that holds the right versions today can still carry a marker wide
enough to put both on one Python, and nothing that only reads version
strings sees it. :func:`nccl_collisions` reads the markers themselves, as
Poetry wrote them, and names every Python that would install both; a
package's test asserts there is none.
"""

from __future__ import annotations

from typing import Final

from packaging.markers import Marker
from platform_core.json_utils import narrow_json_to_dict, narrow_json_to_list, require_str
from platform_core.toml_utils import loads_toml

#: The two distributions that both install ``libnccl.so.2``.
NCCL_DISTRIBUTIONS: Final[frozenset[str]] = frozenset({"nvidia-nccl-cu12", "nvidia-nccl-cu13"})

#: The Pythons a lock is read for: every minor from the monorepo's floor,
#: 3.11, through 3.16, the last one torch 2.15's triton marker names.
PYTHON_VERSIONS: Final[tuple[str, ...]] = tuple(f"3.{minor}" for minor in range(11, 17))


def _linux_environment(python_version: str) -> dict[str, str]:
    """The marker environment of an x86-64 Linux CPython of one minor.

    Args:
        python_version: A ``major.minor`` version, e.g. ``"3.12"``.

    Returns:
        Every marker variable a lock's markers name, for that interpreter.
    """
    return {
        "python_version": python_version,
        "python_full_version": f"{python_version}.0",
        "platform_system": "Linux",
        "sys_platform": "linux",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }


def nccl_collisions(lock_text: str) -> dict[str, tuple[str, ...]]:
    """Name every Python on which a lock installs both NCCL distributions.

    A lock entry with no ``markers`` key installs everywhere, which is how
    Poetry writes an unconditional dependency.

    Args:
        lock_text: A ``poetry.lock`` file's complete text.

    Returns:
        For each version in :data:`PYTHON_VERSIONS` that would install more
        than one of :data:`NCCL_DISTRIBUTIONS`, the distributions it would
        install, sorted. Empty when no Python installs two.
    """
    installed: dict[str, set[str]] = {version: set() for version in PYTHON_VERSIONS}
    for entry in narrow_json_to_list(loads_toml(lock_text)["package"]):
        package = narrow_json_to_dict(entry)
        name = require_str(package, "name")
        if name not in NCCL_DISTRIBUTIONS:
            continue
        marker = Marker(require_str(package, "markers")) if "markers" in package else None
        for version in PYTHON_VERSIONS:
            if marker is None or marker.evaluate(_linux_environment(version)):
                installed[version].add(name)
    return {version: tuple(sorted(names)) for version, names in installed.items() if len(names) > 1}


__all__ = ["NCCL_DISTRIBUTIONS", "PYTHON_VERSIONS", "nccl_collisions"]
