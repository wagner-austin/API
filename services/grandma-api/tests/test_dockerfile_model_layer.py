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
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from platform_core.json_utils import require_dict, require_str
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


def _declared(lib: str, package: str) -> str:
    """The constraint one lib's pyproject declares for one package.

    Args:
        lib: The lib directory under libs/.
        package: The dependency's name.

    Returns:
        Its constraint as a PEP 440 specifier.
    """
    document = loads_toml((LIBS / lib / "pyproject.toml").read_text(encoding="utf-8"))
    tool = require_dict(document, "tool")
    poetry = require_dict(tool, "poetry")
    dependencies = require_dict(poetry, "dependencies")
    return _pep440(require_str(dependencies, package))


def test_the_model_layer_installs_each_package_as_its_lib_declares_it() -> None:
    expected = {package: _declared(lib, package) for package, lib in DECLARED_BY.items()}
    assert _model_layer() == expected


def test_a_caret_reads_as_the_range_poetry_means_by_it() -> None:
    assert _pep440("^4.30.0") == ">=4.30.0,<5.0.0"
    assert _pep440("<2.0") == "<2.0"
