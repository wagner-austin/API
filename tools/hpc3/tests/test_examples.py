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
from tests.against_hpc3 import decode_project_config

_EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"

_README = pathlib.Path(__file__).parent.parent / "README.md"

_FENCE = "```"
_JSON_FENCE = _FENCE + "json"


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


def _readme_json_blocks() -> list[str]:
    """Extract every fenced ``json`` block from the README.

    A block that opens with a quote is a fragment meant to be pasted INTO an
    enclosing object -- the "add one entry to ``projects``" snippet -- so it
    is wrapped in braces here exactly as a reader would paste it. That keeps
    the documented gesture and the tested gesture the same one.

    Returns:
        Each block's text, in document order, parseable as JSON.
    """
    blocks: list[str] = []
    collecting = False
    current: list[str] = []
    for line in _README.read_text(encoding="utf-8").splitlines():
        if collecting and line.startswith(_FENCE):
            body = "\n".join(current)
            blocks.append("{" + body + "}" if body.startswith('"') else body)
            collecting = False
            current = []
        elif collecting:
            current.append(line)
        elif line.startswith(_JSON_FENCE):
            collecting = True
    return blocks


def _readme_project_configs() -> list[JSONValue]:
    """Collect every project entry the README shows.

    Both shapes count: the entries inside a full workspace document, and the
    standalone fragment shown under "Adding a project". They are found by
    carrying a ``partition``, not by which block they came from, so a new
    example is covered the moment it is written.

    Returns:
        One value per documented project entry.
    """
    configs: list[JSONValue] = []
    for block in _readme_json_blocks():
        document = load_json_str(block)
        if not isinstance(document, dict):
            continue
        projects = document.get("projects")
        candidates = projects if isinstance(projects, dict) else document
        for value in candidates.values():
            if isinstance(value, dict) and "partition" in value:
                configs.append(value)
    return configs


def _workspace() -> Workspace:
    """Decode the example workspace.

    Returns:
        The validated workspace.
    """
    return decode_workspace(_load("hpc3.json"), config_dir=_EXAMPLES)


class TestTheReadmeIsAnExampleToo:
    """The README is the first thing anyone copies, so it is tested like one.

    ``examples/`` was already covered; the README was not, and it drifted --
    both of its project snippets went on omitting ``deterministic`` after the
    field became required, so the documented gesture produced a refusal in the
    reader's own file. Covering only the directory nobody opens first is the
    gap this closes.
    """

    def test_every_documented_json_block_parses(self) -> None:
        blocks = _readme_json_blocks()
        assert len(blocks) == 6
        assert [isinstance(load_json_str(block), dict) for block in blocks] == [True] * 6

    def test_every_documented_project_entry_decodes(self) -> None:
        """Including the standalone fragment, which is what a reader pastes."""
        configs = _readme_project_configs()
        assert len(configs) == 3
        assert [decode_project_config(config)["partition"] for config in configs] == [
            "free-gpu",
            "free",
            "free-gpu",
        ]

    def test_the_documented_cpu_only_entry_really_is_cpu_only(self) -> None:
        """The README claims `"gpu": null` is how CPU work is stated. This is
        that claim run through the decoder rather than read."""
        cpu = [c for c in _readme_project_configs() if decode_project_config(c)["gpu"] is None]
        assert len(cpu) == 1
        assert decode_project_config(cpu[0])["partition"] == "free"

    def test_the_documented_workspace_decodes_whole(self) -> None:
        document = load_json_str(_readme_json_blocks()[0])
        assert decode_workspace(document, config_dir=_EXAMPLES)["host"] == "hpc3"

    def test_the_readme_states_what_it_cannot_submit(self) -> None:
        """A tool that only documents its powers reads as having no limits.

        The shapes below are absent by omission rather than by decision, and
        an omission that is not written down is indistinguishable from one
        nobody noticed.
        """
        text = _README.read_text(encoding="utf-8")
        assert "## What this cannot submit" in text
        for shape in ("Multi-node / MPI", "Job array", "Job dependency", "Explicit `--qos`"):
            assert shape in text

    def test_a_limit_that_was_lifted_is_not_still_listed_as_one(self) -> None:
        """This assertion named "CPU-only job" until CPU-only shipped, and
        failed the moment the row came out -- which is the behaviour wanted
        from a list of limits. A stale entry there is worse than no list: it
        sends a reader looking for a workaround to something that works."""
        section = _README.read_text(encoding="utf-8").split("## What this cannot submit")[1]
        table = section.split("None of these")[0]
        assert "CPU-only" not in table


class TestExampleWorkspace:
    def test_it_decodes(self) -> None:
        workspace = _workspace()
        assert workspace["host"] == "hpc3"
        assert sorted(workspace["projects"]) == ["abl", "sirius"]

    def test_it_carries_a_gpu_project_and_a_cpu_project(self) -> None:
        """Both shapes ship as examples, because a reader with a JVM tool and
        a reader with a training script both open this file first."""
        projects = _workspace()["projects"]
        assert projects["abl"]["gpu"] == {"model": "A100", "count": 1}
        assert projects["sirius"]["gpu"] is None
        assert projects["sirius"]["partition"] == "free"

    def test_its_ledger_resolves_beside_it(self) -> None:
        assert pathlib.Path(_workspace()["ledger"]).parent == _EXAMPLES


class TestExampleRun:
    def test_it_resolves_against_the_example_workspace(self) -> None:
        spec = resolve_run(_workspace(), _load("run-arm-b.json"))
        assert spec["project"] == "abl"
        assert spec["gpu"] == {"model": "A100", "count": 1}

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
