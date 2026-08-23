"""The shipped example documents must actually work.

An example that no longer parses is worse than no example: it is the first
thing anyone copies, and the failure surfaces as a confusing error in their
own file rather than in ours. These read the real files from ``examples/``
through the real decoders, so a change to any contract that invalidates them
fails here rather than in someone's first attempt.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONValue, load_json_str

from hpc3.contracts.run import resolve_run, resolve_sweep
from hpc3.contracts.sweep import expand_sweep
from hpc3.contracts.workspace import Workspace, decode_workspace

_EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"


def _load(name: str) -> JSONValue:
    """Read one example document.

    Args:
        name: Filename within ``examples/``.

    Returns:
        The parsed document.
    """
    return load_json_str((_EXAMPLES / name).read_text(encoding="utf-8"))


def _fields(name: str) -> list[str]:
    """List the top-level fields one example document states.

    Args:
        name: Filename within ``examples/``.

    Returns:
        The field names, sorted.

    Raises:
        TypeError: If the document is not a JSON object.
    """
    document = _load(name)
    if not isinstance(document, dict):
        raise TypeError(f"{name} must be a JSON object")
    return sorted(document)


def _workspace() -> Workspace:
    """Decode the example workspace.

    Returns:
        The validated workspace.
    """
    return decode_workspace(_load("hpc3.json"), config_dir=_EXAMPLES)


class TestExampleWorkspace:
    def test_it_decodes(self) -> None:
        workspace = _workspace()
        assert workspace["host"] == "hpc3"
        assert sorted(workspace["projects"]) == ["abl"]

    def test_its_ledger_resolves_beside_it(self) -> None:
        assert pathlib.Path(_workspace()["ledger"]).parent == _EXAMPLES


class TestExampleRun:
    def test_it_resolves_against_the_example_workspace(self) -> None:
        spec = resolve_run(_workspace(), _load("run-arm-b.json"))
        assert spec["project"] == "abl"
        assert spec["gpu"] == "A100"

    def test_a_run_states_only_what_is_specific_to_it(self) -> None:
        """The README's central claim, checked against the real file.

        Four fields: three saying what to run, and one saying what the run
        IS so the result can be traced back to it. Everything else -- the
        partition, the GPU, the cores, the environment -- is inherited.
        """
        assert _fields("run-arm-b.json") == ["command", "experiment", "name", "project"]


class TestExampleSweep:
    def test_it_resolves_and_every_member_inherits(self) -> None:
        specs = expand_sweep(resolve_sweep(_workspace(), _load("sweep-scale-rung.json")))
        assert len(specs) == 4
        assert {s["cpus"] for s in specs} == {8}

    def test_its_overrides_reach_every_member(self) -> None:
        specs = expand_sweep(resolve_sweep(_workspace(), _load("sweep-scale-rung.json")))
        assert {s["minutes"] for s in specs} == {900}
        assert {s["checkpoint_steps"] for s in specs} == {250}
