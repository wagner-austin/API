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


_ORIGINAL_WORKSPACE = "hpc3.json"
"""The first workspace, which predates one-file-per-project.

It declares ``cleargbm`` rather than a project named after itself. Every
workspace added since is ``hpc3-<project>.json``.
"""

_WORKSPACE_PREFIX = "hpc3-"
_WORKSPACE_SUFFIX = ".json"


def _expected_projects_for(filename: str) -> list[str]:
    """The projects a workspace filename commits it to declaring.

    Args:
        filename: A workspace document's filename, e.g. ``hpc3-rusted.json``.

    Returns:
        The single project the name implies, or ``cleargbm`` for the original
        workspace. A one-element list rather than a bare name, so the caller
        compares it against ``sorted(workspace["projects"])`` directly and a
        document declaring TWO projects fails on the same comparison.
    """
    if filename == _ORIGINAL_WORKSPACE:
        return ["cleargbm"]
    return [filename[len(_WORKSPACE_PREFIX) : -len(_WORKSPACE_SUFFIX)]]


class TestTheCommittedWorkspaces:
    """One document per project, all sharing one ledger file.

    These were two hardcoded inventories until 2026-09-03 -- a six-name list
    and a six-entry filename map -- so REGISTERING A PROJECT MEANT EDITING
    THIS FILE, and the edit was discovered by meeting a red test rather than
    by following a step. That is the same defect
    :mod:`hpc3.core.research_index` was built to remove from ``RESEARCH.md``:
    a restatement of what the workspace documents already declare, kept in
    sync by hand. What cannot drift is what nobody retypes.

    They are now the PROPERTIES those lists were standing in for, so a
    seventh project needs no edit here and still cannot violate either rule.
    """

    def test_no_project_is_declared_by_two_workspaces(self) -> None:
        """ "Which document governs this run" must have exactly one answer.

        The module docstring calls this the reading that must not be
        permitted; it is the one thing pinnable without deciding whether the
        budget split should be merged.
        """
        declaring: dict[str, list[str]] = {}
        for filename, workspace in _workspaces().items():
            for project in workspace["projects"]:
                declaring.setdefault(project, []).append(filename)

        assert {p: sorted(f) for p, f in declaring.items() if len(f) > 1} == {}

    def test_each_workspace_declares_exactly_the_project_its_filename_names(self) -> None:
        """``hpc3-<project>.json`` declares ``<project>``, and nothing else.

        ``hpc3.json`` is the original workspace and predates the convention;
        it declares ``cleargbm``. Naming that one exception is what lets the
        rule be a rule instead of a list -- and it is checked here rather
        than assumed, so the day it stops being true this fails.
        """
        wrong = {
            filename: sorted(workspace["projects"])
            for filename, workspace in _workspaces().items()
            if sorted(workspace["projects"]) != _expected_projects_for(filename)
        }

        assert wrong == {}

    def test_the_registry_is_not_empty(self) -> None:
        """A derived rule passes vacuously over nothing; this says it did not."""
        assert len(_by_project()) >= 6

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
        2026-08-28, and ``RustedWarfareBot`` until it was onboarded as
        ``rusted`` on 2026-08-29. Each time, asserting its presence in this
        section would have kept a true sentence in a section that had stopped
        applying to it. Twice is the pattern: what must not disappear is the
        SECTION, and the entries in it are expected to leave one at a time.

        ``sirius`` is what remains, and it is a different case again -- it is
        named there as a deliberate non-registration rather than as a backlog
        item, so it does not leave by being onboarded.
        """
        text = _INDEX.read_text(encoding="utf-8")
        unregistered = text.split("## Not registered anywhere")[1]
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
            "sweep-turkic-bases-v4.json",
            "sweep-turkic-bases.json",
        ]
        assert sum(expanded) == 126

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
        assert len(artifacts) == 126
        assert sum(1 for a in artifacts if a is None) == 108
        assert stated[0] == "/pub/wagnera3/LSTM/checkpoints/az_best.pt"
        # 7 from the original sweep, 4 more from the resume round that
        # followed the 2026-08-28 preemption wave. `free-gpu` is
        # PreemptMode=CANCEL, so a preempted member does not come back on its
        # own and is resubmitted as a new record naming the jobs it resumes --
        # which is why a resume round is a committed document rather than a
        # command someone re-ran.
        #
        # 7 more from sweep-turkic-bases-v4, which retrains every language on
        # the corrected corpus. Those write to checkpoints_v4 rather than over
        # checkpoints: five of the seven v3 corpora are byte-identical to v4,
        # so those checkpoints stay valid, and keeping them is what lets the
        # v4 run measure run-to-run variability against a real baseline.
        assert len(stated) == 18
