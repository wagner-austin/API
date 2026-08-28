"""The committed run documents must resolve against a committed workspace.

``examples/`` is covered by :mod:`tests.test_examples` on the reasoning that
an example nobody can parse is worse than no example. ``runs/`` carries a
stronger claim and had no such cover: commit ``0a9d33c4`` committed the
cleargbm project block and its sweep documents past the ``runs/`` ignore
specifically as "the declarative record" of work that ran. A declarative
record that no longer decodes is worse than an unparseable example, because
nobody copies it -- they cite it.

WHAT THIS COVERS THAT NOTHING DID. Three workspace documents are committed --
``hpc3.json``, ``hpc3-mi.json`` and ``hpc3-floor.json`` -- one per project,
each with its own budget. A run document names a project and only the
workspace declaring that project can resolve it, so "does this record still
decode" was a question no test asked and no single ``--config`` could answer.
Resolving each document against the workspace that declares its project is
what these tests do.

WHAT IT CAUGHT ON ITS FIRST RUN. Six committed cleargbm sweeps failed:
``artifact`` became required on a sweep member after they were written, so
the record of the 320-trial P6 farm rung could no longer be read by the code
that wrote it. ``null`` is the honest value -- every member runs
``--no-save-model`` and writes no file of its own -- and stating it explicitly
is what the contract asks for.

WHAT IT DELIBERATELY DOES NOT ASSERT. That there is one workspace. The
package's own :class:`~hpc3.contracts.workspace.Workspace` documents the
budget as "shared by every project. One pool, because the machine is one
machine", and ``hpc3-watch`` says the ceiling it enforces "is the same one
the submitting command projected against" -- neither of which is true across
three documents declaring 0.5, 12.0 and 1.0 GPU-hours over the same ledger
file. That is a real tension, but the split is deliberate (see ``21e1efd2``,
which unfunded ``mi`` on its own), and resolving it means either merging the
documents or moving the budget into ``ProjectConfig``. A test should not
pick. What it CAN pin without picking is that no project is declared by two
workspaces, because that is the reading where "which document governs this
run" has no answer at all.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONValue, load_json_str

from hpc3.contracts.run import resolve_run, resolve_sweep
from hpc3.contracts.sweep import expand_sweep
from hpc3.contracts.workspace import Workspace, decode_workspace

_RUNS = pathlib.Path(__file__).parent.parent / "runs"

_INDEX = pathlib.Path(__file__).parents[3] / "docs" / "RESEARCH.md"
"""The research index, at the monorepo root rather than in this package.

It names work in other repositories -- LSTM lives outside this one entirely --
so it cannot live under the tool that happens to submit some of it. This
package holds the check because this package holds the registry.
"""


def _documents() -> list[tuple[str, dict[str, JSONValue]]]:
    """Read every JSON object in ``runs/``.

    Returns:
        Each document's filename and parsed body, in filename order.
    """
    found: list[tuple[str, dict[str, JSONValue]]] = []
    for path in sorted(_RUNS.glob("*.json")):
        document = load_json_str(path.read_text(encoding="utf-8"))
        if isinstance(document, dict):
            found.append((path.name, document))
    return found


def _submissions() -> list[tuple[str, str, dict[str, JSONValue]]]:
    """Select the documents that are submissions rather than configuration.

    A submission is identified by naming a ``project``, which is what
    :func:`~hpc3.contracts.run.resolve_run` requires first. That predicate
    excludes the workspaces themselves, image specifications, and any
    document predating the field -- without an exemption list, which is the
    point: a filename is not a reason.

    Returns:
        Each submission's filename, project name, and body.

    Raises:
        TypeError: If a document's ``project`` is not a string.
    """
    found: list[tuple[str, str, dict[str, JSONValue]]] = []
    for name, document in _documents():
        project = document.get("project")
        if project is None:
            continue
        if not isinstance(project, str):
            raise TypeError(f"{name}: 'project' must be a string")
        found.append((name, project, document))
    return found


def _workspaces() -> dict[str, Workspace]:
    """Decode every committed workspace document.

    Returns:
        The validated workspaces, keyed by filename.
    """
    return {
        name: decode_workspace(document, config_dir=_RUNS)
        for name, document in _documents()
        if "projects" in document
    }


def _by_project() -> dict[str, Workspace]:
    """Map each declared project to the workspace declaring it.

    Returns:
        One workspace per project name.

    Raises:
        ValueError: If two workspaces declare the same project, which leaves
            no answer to which one governs a run naming it.
    """
    owners: dict[str, Workspace] = {}
    for name, workspace in _workspaces().items():
        for project in workspace["projects"]:
            if project in owners:
                raise ValueError(f"project {project!r} is declared twice, one place is {name}")
            owners[project] = workspace
    return owners


def _sweep_member_artifacts() -> list[JSONValue]:
    """Read the artifact every sweep member declares.

    Returns:
        One value per member across every sweep document, in document order.

    Raises:
        TypeError: If a member is not a JSON object, which would mean the
            document is not a sweep at all.
    """
    artifacts: list[JSONValue] = []
    for _, _, document in _submissions():
        members = document.get("members")
        if not isinstance(members, list):
            continue
        for member in members:
            if not isinstance(member, dict):
                raise TypeError("a sweep member must be a JSON object")
            artifacts.append(member["artifact"])
    return artifacts


class TestTheCommittedWorkspaces:
    """Three documents, one per project, sharing one ledger file."""

    def test_they_declare_every_project_exactly_once(self) -> None:
        assert sorted(_by_project()) == ["cleargbm", "floor", "mi", "turkic-lstm"]

    def test_each_workspace_declares_the_project_its_filename_implies(self) -> None:
        declared = {name: sorted(w["projects"]) for name, w in _workspaces().items()}
        assert declared == {
            "hpc3-floor.json": ["floor"],
            "hpc3-mi.json": ["mi"],
            "hpc3-turkic-lstm.json": ["turkic-lstm"],
            "hpc3.json": ["cleargbm"],
        }

    def test_every_workspace_resolves_its_ledger_to_the_same_file(self) -> None:
        """The forks diverge on budget but converge on the ledger, which is
        what makes one pool's worth of work land in one record."""
        ledgers = {pathlib.Path(w["ledger"]) for w in _workspaces().values()}
        assert ledgers == {_RUNS / "ledger.jsonl"}


class TestTheResearchIndexNamesEveryProject:
    """``docs/RESEARCH.md`` is the list a new session reads first.

    A list that is only prose rots the way the required-symbol assertion in
    ``test_contracts_image`` rotted -- nine commits past the point it stopped
    being true. So the registered half of it is checked: every project a
    committed workspace declares must appear in the index, and every repo
    path a project declares must exist.

    The reverse direction is deliberately NOT asserted. The index carries
    surfaces that are not registered anywhere -- LSTM, RustedWarfareBot --
    and that is the whole reason it is worth reading; a test demanding the
    two match exactly would be satisfied by deleting the entries that matter.
    """

    def test_every_declared_project_appears_in_the_index(self) -> None:
        text = _INDEX.read_text(encoding="utf-8")
        missing = sorted(name for name in _by_project() if f"`{name}`" not in text)
        assert missing == []

    def test_every_declared_repo_exists(self) -> None:
        """A path nobody checks is a path that is eventually wrong."""
        absent = sorted(
            name
            for name, workspace in _workspaces().items()
            for config in workspace["projects"].values()
            if not pathlib.Path(config["repo"]).is_dir()
        )
        assert absent == []

    def test_the_index_states_which_surfaces_are_unregistered(self) -> None:
        """The entries no tool can see are the ones a reader most needs.

        ``LSTM`` was named here until it was onboarded as ``turkic-lstm`` on
        2026-08-28, at which point asserting its presence in this section
        would have kept a true sentence in a section that had stopped
        applying to it. The section itself is what must not disappear.
        """
        text = _INDEX.read_text(encoding="utf-8")
        unregistered = text.split("## Not registered anywhere")[1]
        assert "RustedWarfareBot" in unregistered
        assert "sirius" in unregistered


class TestEveryCommittedSubmissionResolves:
    """The regression guard: a record that no longer decodes is not a record."""

    def test_every_project_named_by_a_submission_is_declared_somewhere(self) -> None:
        named = {project for _, project, _ in _submissions()}
        assert sorted(named - set(_by_project())) == []

    def test_every_run_document_resolves(self) -> None:
        owners = _by_project()
        runs = [(p, d) for _, p, d in _submissions() if "command" in d and "members" not in d]
        resolved = [resolve_run(owners[project], doc)["project"] for project, doc in runs]
        assert sorted(set(resolved)) == ["floor", "mi", "turkic-lstm"]
        assert len(resolved) == len(runs)

    def test_every_sweep_document_resolves_and_expands(self) -> None:
        owners = _by_project()
        sweeps = [(n, p, d) for n, p, d in _submissions() if "members" in d]
        expanded = [len(expand_sweep(resolve_sweep(owners[p], d))) for _, p, d in sweeps]
        assert sorted(name for name, _, _ in sweeps) == [
            "sweep-cleargbm-p6-rung1.json",
            "sweep-cleargbm-p6-rung2.json",
            "sweep-cleargbm-p6-rung3.json",
            "sweep-cleargbm-p6-rung4.json",
            "sweep-cleargbm-p6-rung4b.json",
            "sweep-cleargbm-p6-rung5.json",
            "sweep-turkic-bases-resume-1.json",
            "sweep-turkic-bases.json",
        ]
        assert sum(expanded) == 119

    def test_no_sweep_member_leaves_its_artifact_unstated(self) -> None:
        """Stated, which is not the same as null.

        This asserted ``== {None}`` while the only sweeps were cleargbm's,
        where null is the honest value because every member runs
        ``--no-save-model`` and writes no file of its own. That made an
        accident of the corpus look like a rule, and the first sweep that
        DOES produce a file -- ``sweep-turkic-bases``, whose members each
        write a checkpoint -- would have failed a test that was never about
        them. What the contract requires is that the key is present and
        deliberate; what its value should be is the member's business.
        """
        artifacts = _sweep_member_artifacts()
        stated = sorted(str(a) for a in artifacts if a is not None)
        assert len(artifacts) == 119
        assert sum(1 for a in artifacts if a is None) == 108
        assert stated[0] == "/pub/wagnera3/LSTM/checkpoints/az_best.pt"
        # 7 from the original sweep, 4 more from the resume round that
        # followed the 2026-08-28 preemption wave. `free-gpu` is
        # PreemptMode=CANCEL, so a preempted member does not come back on its
        # own and is resubmitted as a new record naming the jobs it resumes --
        # which is why a resume round is a committed document rather than a
        # command someone re-ran.
        assert len(stated) == 11
