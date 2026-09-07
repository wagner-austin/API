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

import io
import pathlib
import subprocess
import tarfile
import tempfile

from platform_core.json_utils import JSONValue, load_json_str

from hpc3.contracts.run import resolve_run, resolve_sweep
from hpc3.contracts.sweep import expand_sweep
from hpc3.contracts.workspace import Workspace, decode_workspace
from hpc3.core.inputs import declared_inputs

_RUNS = pathlib.Path(__file__).parent.parent / "runs"

_REPO = pathlib.Path(__file__).parents[3]

_RUNS_IN_REPO = "tools/hpc3/runs"
"""``runs/`` as git spells it, which is the only spelling git accepts."""

_INDEX = pathlib.Path(__file__).parents[3] / "docs" / "RESEARCH.md"
"""The research index, at the monorepo root rather than in this package.

It names work in other repositories -- LSTM lives outside this one entirely --
so it cannot live under the tool that happens to submit some of it. This
package holds the check because this package holds the registry.
"""


def _documents() -> list[tuple[str, dict[str, JSONValue]]]:
    """Read every JSON object COMMITTED under ``runs/``.

    READS THE COMMITTED TREE, NOT THE WORKING DIRECTORY, and the module's
    name is the reason. ``.gitignore`` ignores ``tools/hpc3/runs/*`` and then
    RE-INCLUDES five families by pattern -- ``hpc3*.json``, ``*-digests.txt``,
    ``*-stage.json``, ``code-style-*.json``, ``cartridge-*.json`` -- which is
    why a tracked set exists at all and why the two counts can diverge far
    without either looking wrong. Globbing the filesystem measured a set that
    exists only on the machine that wrote it.

    A snapshot, not an invariant: on 2026-09-07 a working tree held 488 JSON
    documents against 259 at HEAD, and a re-measure the same day read 492 and
    263 as four more landed. Both spreads make the point; neither is a number
    to check this against, which is why the assertions below use floors.

    That is not hypothetical: a floor calibrated at 138 locally arrived on CI
    as ``assert 36 >= 100`` (run 34104178998). Every developer's ``make
    check`` was green, because every working tree is self-consistent and CI
    is the only reader that starts from a clean checkout. A module named
    ``test_committed_runs`` that never asked git could not measure the
    property it is named for.

    ``git archive HEAD`` rather than a per-file ``git show``: one subprocess
    instead of 259, and it works on the shallow clone ``actions/checkout``
    produces by default, because HEAD's tree is present even at depth 1. An
    older revision would not be, which is a separate trap this repo has
    already paid for once.

    Returns:
        Each document's filename and parsed body, in filename order.
    """
    archive = subprocess.run(
        ["git", "archive", "HEAD", "--", _RUNS_IN_REPO],
        cwd=_REPO,
        capture_output=True,
        check=True,
    ).stdout
    found: list[tuple[str, dict[str, JSONValue]]] = []
    with tempfile.TemporaryDirectory() as scratch:
        root = pathlib.Path(scratch)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
        for path in sorted((root / _RUNS_IN_REPO).glob("*.json")):
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

    def test_every_declared_repo_inside_this_monorepo_exists(self) -> None:
        """A path nobody checks is a path that is eventually wrong.

        Only the ones this repository can actually know about. ``turkic-lstm``
        declares ``../../../../LSTM`` -- a SIBLING checkout, one level above
        the monorepo root -- because that project genuinely lives outside this
        tree, which ``docs/RESEARCH.md`` states plainly. Whether that directory
        exists is a property of the machine, not of this repository, so a CI
        runner that checks out only this repo will never have it.

        This test asserted every path unconditionally until 2026-09-04, which
        meant it passed on exactly one machine and failed the first time
        anything else ran it. Narrowing it to in-repo paths keeps what it was
        built to catch -- a typo'd or stale ``repo`` in a workspace document --
        and stops claiming to check what it cannot see.
        """
        root = _RUNS.parents[2].resolve()
        absent = sorted(
            name
            for name, workspace in _workspaces().items()
            for config in workspace["projects"].values()
            if (resolved := pathlib.Path(config["repo"]).resolve()).is_relative_to(root)
            and not resolved.is_dir()
        )

        assert absent == []

    def test_a_repo_outside_this_monorepo_is_named_in_the_index(self) -> None:
        """What the test above stops checking, this one refuses to lose.

        An out-of-tree ``repo`` is unverifiable here, so the compensating
        control is that the project must still be described in the research
        index -- which is where a reader learns the checkout is expected
        beside this one rather than inside it.
        """
        root = _RUNS.parents[2].resolve()
        text = _INDEX.read_text(encoding="utf-8")
        outside = [
            name
            for workspace in _workspaces().values()
            for name, config in workspace["projects"].items()
            if not pathlib.Path(config["repo"]).resolve().is_relative_to(root)
        ]

        assert outside != []
        assert [name for name in outside if f"`{name}`" not in text] == []

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
        """Resolution IS the assertion: ``resolve_run`` raises rather than returns.

        THE PROJECT LIST THAT USED TO BE HERE WAS THE THIRD HARDCODED
        INVENTORY IN THIS FILE, and the two above it were derived on
        2026-09-03 precisely because "registering a project meant editing this
        file, and the edit was discovered by meeting a red test rather than by
        following a step". This one was left, so registering ``code-style``
        met it as exactly that -- a fourth surprise failure, in the file whose
        own docstring says a seventh project needs no edit here.

        What the list was standing in for is stronger stated as a property,
        and is asserted below: every document resolves against the workspace
        declaring the project it NAMES, and comes back naming that same
        project. A set comparison could not catch a document that resolved to
        the wrong project as long as some other document named the right one.
        """
        owners = _by_project()
        runs = [(p, d) for _, p, d in _submissions() if "command" in d and "members" not in d]
        resolved = [resolve_run(owners[project], doc)["project"] for project, doc in runs]

        # A rule that silently resolves nothing passes forever.
        assert runs != []
        assert resolved == [project for project, _ in runs]

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


class TestTheInputCheckHasASubject:
    """``declared_inputs`` must be seen to find something.

    ``require_inputs_present`` refuses a run whose declared inputs are not on
    the cluster. It is silent when a command declares nothing -- correctly,
    since most do -- which means the whole check goes vacuously green the
    moment the extractor stops recognising a flag. Rename ``--payload`` to
    ``--payload-file`` and every code-style run declares nothing, every
    submission passes, and the gap that killed job 55806418 is back with the
    guard still reporting success.

    The unit tests in :mod:`tests.test_inputs` cannot catch that: they pass
    the extractor command strings they wrote themselves, so they agree with
    it by construction. Only the committed documents are an independent
    subject.
    """

    #: The flag spellings the committed commands actually use, written out
    #: here rather than imported from :data:`~hpc3.core.inputs.INPUT_FLAGS`.
    #: Reading them from the module under test would make this circular --
    #: renaming the flag would rename the expectation with it, and the loop
    #: would skip every command instead of failing. These are the strings on
    #: the command lines, and they are the reason the extractor exists.
    _SPELLINGS = ("--payload", "--spec", "--items", "--corpus")

    def test_a_command_naming_an_input_flag_yields_that_input(self) -> None:
        """The sharp form, with no count to keep up to date.

        A committed command containing ``--payload`` and a ``/pub`` path MUST
        produce a declared input. This fires on exactly the regression that
        matters -- the extractor no longer recognising a flag that is right
        there in the command -- with no magic number and no exemption list.
        """
        checked = 0
        for name, _project, document in _submissions():
            command = document.get("command")
            if not isinstance(command, str):
                continue
            for flag in self._SPELLINGS:
                if f"{flag} /pub/" not in command:
                    continue
                checked += 1
                assert declared_inputs(command), (
                    f"{name}: command names {flag} with a /pub path, "
                    f"but declared_inputs found nothing: {command}"
                )

        # The loop above is vacuous if no committed command spells any of
        # them, which is the reading this whole class exists to refuse.
        # 36 committed submissions spell one, measured against HEAD rather
        # than against a working tree -- see _documents. The floor sits well
        # under that so adding or retiring a run document does not fail this,
        # and well over zero because zero is the failure it exists to catch:
        # a renamed flag collapses this count to 0, not to 19.
        assert checked >= 20

    def test_the_committed_documents_actually_exercise_the_extractor(self) -> None:
        """The subject exists at all.

        Every assertion above is over a loop that a deleted or emptied
        ``runs/`` would make vacuous.

        MEASURED AGAINST HEAD, 2026-09-07: 195 committed documents, 158 of
        them submissions, 36 declaring at least one input across 10 unique
        cluster paths.

        The numbers this docstring first carried -- "138 committed
        submissions ... 29 unique paths" -- were measured by globbing a
        WORKING TREE and were wrong for any clean checkout. They put the
        floor at 100, which CI met as ``assert 36 >= 100``. The word
        "committed" was doing no work: nothing here asked git. Both halves
        are fixed -- the set in :func:`_documents`, the floor here -- and the
        floor is the lesser half. Raising it alone would have made the number
        agree with CI while the check still measured the wrong set.
        """
        declaring = [
            name
            for name, _project, document in _submissions()
            if isinstance(document.get("command"), str)
            and declared_inputs(str(document["command"]))
        ]

        assert len(declaring) >= 20
        assert "code-style-run-train-v2.json" in declaring
        assert "code-style-run-gen-v2-base.json" in declaring

    def test_an_output_flag_is_not_mistaken_for_an_input(self) -> None:
        """The committed documents as an independent check on the split.

        ``--record`` and ``--out-dir`` name files a job WRITES. If either were
        ever added to ``INPUT_FLAGS`` the first run of everything would be
        refused, and it would be refused in production rather than here.
        """
        for name, _project, document in _submissions():
            command = document.get("command")
            if not isinstance(command, str):
                continue
            for path in declared_inputs(command):
                assert "/results/" not in path, f"{name}: {path} is written, not read"
                assert "/generated/" not in path, f"{name}: {path} is written, not read"
