"""The Dockerfile's model layer installs exactly what its libs declare.

MCPs board task 90c135ac. The runtime stage installs platform_langid's model
stack (torch, torchaudio, transformers) and platform_stt's numpy in a layer
of their own, before the wheel and the libs are copied in, so a change to
the service or to any lib reinstalls only those. That layer names each
requirement, and the names are a copy of what the two libs' pyproject files
declare. A copy that drifts costs no correctness, since the install after
it resolves the libs' own constraints, but it costs the whole point: with
bare names the layer took transformers 5 and numpy 2, and the wheel install
spent 106 s on lavender-wsl's rootless daemon replacing them. This pins the
copy to its source.

The index is pinned too (MCPs board task 3e7cfd51). The layer installs torch
from the PyTorch CPU index, and so do the poetry venvs this service and
platform_langid are checked in, each declaring that index as an explicit
source for torch and torchaudio: Poetry reads only the root project's
sources, and PyPI's Linux torch is the CUDA build with 14 nvidia packages
nothing here loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from platform_core.json_utils import JSONObject, require_dict, require_list, require_str
from platform_core.toml_utils import loads_toml

#: services/grandma-api, the directory holding the Dockerfile.
SERVICE: Final[Path] = Path(__file__).resolve().parents[1]

#: The monorepo's libs directory.
LIBS: Final[Path] = SERVICE.parents[1] / "libs"

#: Each package the model layer installs, and the lib whose pyproject
#: declares its constraint.
DECLARED_BY: Final[dict[str, str]] = {
    "torch": "platform_langid",
    "torchaudio": "platform_langid",
    "transformers": "platform_langid",
    "numpy": "platform_stt",
}

#: The characters a PEP 440 specifier opens with, which end a name.
SPECIFIER_START: Final[str] = "<>=!~"

#: The PyTorch CPU index, and the name each pyproject gives it.
CPU_INDEX: Final[str] = "https://download.pytorch.org/whl/cpu"
CPU_SOURCE: Final[str] = "pytorch-cpu"

#: The packages each pyproject takes from :data:`CPU_INDEX`.
FROM_CPU_INDEX: Final[frozenset[str]] = frozenset({"torch", "torchaudio"})

#: The pyproject files whose venvs install torch: the service and its lib.
TORCH_PYPROJECTS: Final[tuple[Path, ...]] = (
    SERVICE / "pyproject.toml",
    LIBS / "platform_langid" / "pyproject.toml",
)


def _split_requirement(requirement: str) -> tuple[str, str]:
    """Split one requirement into its name and its specifier.

    Args:
        requirement: The quoted text, such as ``numpy<2.0``.

    Returns:
        The name and everything from the first specifier character on.
    """
    cut = min(
        (index for index, char in enumerate(requirement) if char in SPECIFIER_START),
        default=len(requirement),
    )
    return requirement[:cut], requirement[cut:]


def _model_layer() -> dict[str, str]:
    """Read the model layer's requirements out of the Dockerfile.

    Returns:
        Each quoted requirement on the layer's one line, name to specifier.
    """
    lines = (SERVICE / "Dockerfile").read_text(encoding="utf-8").splitlines()
    layer = [line for line in lines if '"torch' in line]
    assert len(layer) == 1, f"expected one model-layer line, found {layer}"
    quoted = layer[0].split('"')[1::2]
    return dict(_split_requirement(requirement) for requirement in quoted)


def _pep440(poetry_constraint: str) -> str:
    """Write a Poetry constraint as the PEP 440 specifier pip reads.

    Args:
        poetry_constraint: The value in ``[tool.poetry.dependencies]``,
            either a caret (``^2.0.0``) or a plain specifier (``<2.0``).

    Returns:
        ``>=X.Y.Z,<X+1.0.0`` for a caret with a non-zero major version, the
        plain specifier unchanged otherwise.
    """
    if not poetry_constraint.startswith("^"):
        return poetry_constraint
    version = poetry_constraint.removeprefix("^")
    major = int(version.split(".")[0])
    assert major > 0, f"a caret on a 0.x version narrows differently: {poetry_constraint}"
    return f">={version},<{major + 1}.0.0"


def _poetry(pyproject: Path) -> JSONObject:
    """The ``[tool.poetry]`` table of one pyproject file.

    Args:
        pyproject: The file.

    Returns:
        The table.
    """
    document = loads_toml(pyproject.read_text(encoding="utf-8"))
    return require_dict(require_dict(document, "tool"), "poetry")


def _constraint(pyproject: Path, package: str) -> str:
    """The Poetry constraint one pyproject declares for one package.

    Args:
        pyproject: The file.
        package: The dependency's name.

    Returns:
        The constraint as written: the ``version`` of the table a package in
        :data:`FROM_CPU_INDEX` is declared by, the string otherwise.
    """
    dependencies = require_dict(_poetry(pyproject), "dependencies")
    if package in FROM_CPU_INDEX:
        return require_str(require_dict(dependencies, package), "version")
    return require_str(dependencies, package)


def _declared(lib: str, package: str) -> str:
    """The constraint one lib's pyproject declares for one package.

    Args:
        lib: The lib directory under libs/.
        package: The dependency's name.

    Returns:
        Its constraint as a PEP 440 specifier.
    """
    return _pep440(_constraint(LIBS / lib / "pyproject.toml", package))


def test_the_model_layer_installs_each_package_as_its_lib_declares_it() -> None:
    expected = {package: _declared(lib, package) for package, lib in DECLARED_BY.items()}
    assert _model_layer() == expected


def test_every_dockerfile_install_reads_the_cpu_index() -> None:
    lines = (SERVICE / "Dockerfile").read_text(encoding="utf-8").splitlines()
    indexed = [line for line in lines if "--extra-index-url" in line]
    assert indexed == [f"RUN pip install --extra-index-url {CPU_INDEX} \\"] * 2


@pytest.mark.parametrize("pyproject", TORCH_PYPROJECTS, ids=("grandma-api", "platform_langid"))
def test_the_venv_takes_torch_from_the_cpu_index(pyproject: Path) -> None:
    poetry = _poetry(pyproject)
    assert require_list(poetry, "source") == [
        {"name": CPU_SOURCE, "url": CPU_INDEX, "priority": "explicit"}
    ]
    dependencies = require_dict(poetry, "dependencies")
    assert {
        package: require_str(require_dict(dependencies, package), "source")
        for package in sorted(FROM_CPU_INDEX)
    } == {"torch": CPU_SOURCE, "torchaudio": CPU_SOURCE}


def test_the_service_declares_torch_at_platform_langid_s_constraints() -> None:
    for package in sorted(FROM_CPU_INDEX):
        assert _constraint(SERVICE / "pyproject.toml", package) == _constraint(
            LIBS / "platform_langid" / "pyproject.toml", package
        )


def test_a_caret_reads_as_the_range_poetry_means_by_it() -> None:
    assert _pep440("^4.30.0") == ">=4.30.0,<5.0.0"
    assert _pep440("<2.0") == "<2.0"
