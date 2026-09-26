"""The archive scope, and the real repository it was measured against.

Two halves, deliberately. The first drives :mod:`fleet.core.archive_scope`
over declarations built here, so every branch of the rule is exercised on
inputs a reader can see. The second runs ``git archive`` against THIS
repository at HEAD and looks at the tar that comes out, because the rule is
worth nothing if the bytes it produces are missing something a check reads --
and because the numbers in the module's docstring and on board task 140e7042
are claims about a real repository, which is the sort of claim that goes
stale silently.

THE SIZE FLOOR IS THE POINT OF THE SECOND HALF. The one quiet failure of a
declared scope is a heavy directory nobody declares: it ships in full, the
payload creeps back up, and no assertion anywhere notices. So a test here
walks the commit and fails when a tracked directory crosses a floor without
being declared, naming it. That is the drift alarm the module's docstring
says lives outside it.
"""

from __future__ import annotations

import collections
import pathlib
import subprocess
import tarfile
from typing import Final

import pytest
from platform_core.json_utils import load_json_str

from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core.archive_scope import EXCLUDE, WHOLE_TREE, archive_pathspec, owns
from fleet.core.dialect import EXPORT_AUTHOR_EMAIL

API: Final[str] = "https://github.com/wagner-austin/API.git"
MCPS: Final[str] = "https://github.com/wagner-austin/MCPs.git"

#: This repository, four hops up from ``tools/fleet/tests/test_*.py``.
REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[3]

#: The registry this package ships, which is the one the hub dispatches with.
REGISTRY: Final[pathlib.Path] = REPO_ROOT / "tools" / "fleet" / "fleet.json"

#: A deadline for every git command here. Generous because the unscoped
#: archive of this repository takes 43 seconds on the hub and a cold cache
#: is slower; finite because a test that hangs a suite is worse than one
#: that fails.
GIT_TIMEOUT_SECONDS: Final[int] = 900

#: A tracked directory holding at least this many bytes is worth naming when
#: a payload has grown. Five megabytes: small enough that the list is not
#: only the obvious two, large enough that it stays a list somebody reads
#: rather than a directory dump. It bounds a DIAGNOSTIC and not a verdict,
#: so it is a readability choice (:func:`undeclared_heavy_directories`).
SIZE_FLOOR_BYTES: Final[int] = 5_000_000


def _git(*args: str) -> str:
    """Run git in this repository and return its standard output.

    Args:
        *args: The arguments after ``git``.

    Returns:
        Standard output, decoded.

    Raises:
        AssertionError: If git exits non-zero, carrying its stderr, so a
            broken invocation reads as a broken invocation rather than as an
            empty result.
    """
    done = subprocess.run(
        ("git", "-C", str(REPO_ROOT), *args),
        capture_output=True,
        text=True,
        check=False,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    assert done.returncode == 0, f"git {' '.join(args)} exited {done.returncode}: {done.stderr}"
    return done.stdout


def _declared() -> tuple[str, ...]:
    """The API repository's declared data directories, from the real registry.

    Returns:
        The paths, in declaration order.
    """
    workspace = decode_fleet_workspace(load_json_str(REGISTRY.read_text(encoding="utf-8")))
    return workspace["data_paths"][API]


def _is_a_git_checkout() -> bool:
    """Whether this tree has the history the measurements here read.

    IT DOES NOT ON A FLEET NODE, and that is the point of asking. A dispatch
    stages ``git archive`` of a commit: the node receives FILES and no
    ``.git``, so ``git ls-tree HEAD`` there exits 128 with "Not a valid
    object name HEAD". Found the honest way, by dispatching this suite to
    serendipity at 3482c418 and reading the log: 3 failed, 8 errors, every
    one of them this.

    AND "HEAD RESOLVES" STOPPED MEANING "CHECKOUT" ON 2026-09-26 (MCPs board
    task 6bbfd171): the runner now commits the staged export once, as
    :data:`~fleet.core.dialect.EXPORT_AUTHOR_EMAIL`, because MCPs
    packages/db's migrator reads ``HEAD``. The first dispatch of this suite
    after that change, c8e38453 on sedona, failed six cases here that took
    the export for a checkout. So a tree is a checkout when ``HEAD``
    resolves to a commit someone other than the runner authored.

    Returns:
        True when ``HEAD`` resolves to a commit not authored by the fleet
        runner, which is a developer checkout and CI.
    """
    done = subprocess.run(
        ("git", "-C", str(REPO_ROOT), "log", "-1", "--format=%ae", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    return done.returncode == 0 and done.stdout.strip() != EXPORT_AUTHOR_EMAIL


def _require_git() -> None:
    """Skip a measurement that needs history, saying exactly what it needed.

    Not a silent skip. A pass over zero rows must not read like a clean
    result, so the reason names the tree it is standing in and what still
    ran without it: on a node, :class:`TestTheStagedTreeItself` checks the
    same property against the files that actually arrived, which is the
    stronger evidence of the two.
    """
    if not _is_a_git_checkout():
        pytest.skip(
            f"not applicable: {REPO_ROOT} has no history but the runner's own export "
            "commit, so this is a staged export "
            "rather than a checkout. The scope's effect on this tree is asserted directly "
            "by TestTheStagedTreeItself, which reads the files the dispatch delivered."
        )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


class TestOwnership:
    def test_a_project_keeps_a_directory_inside_itself(self) -> None:
        assert owns("libs/instrument_io", "libs/instrument_io/tests/fixtures")

    def test_a_project_keeps_a_directory_that_is_exactly_itself(self) -> None:
        assert owns("tools/hpc3", "tools/hpc3")

    def test_a_project_does_not_own_another_project_s_directory(self) -> None:
        assert not owns("libs/monorepo_guards", "libs/instrument_io/tests/fixtures")

    def test_a_sibling_sharing_a_prefix_is_not_owned(self) -> None:
        """``libs/instrument`` must not claim ``libs/instrument_io``'s data.

        The bug a plain ``startswith`` would have: the separator is what
        makes one path a parent of another, and without it every project is
        a parent of every project whose name it prefixes.
        """
        assert not owns("libs/instrument", "libs/instrument_io/tests/fixtures")

    def test_a_project_that_is_the_repository_owns_everything_in_it(self) -> None:
        """slime declares path '' and its payload must not change."""
        assert owns("", "anything/at/all")


class TestPathspec:
    def test_an_undeclared_repository_is_scoped_to_nothing(self) -> None:
        """Not ``('.',)``: the command must stay the one it always was."""
        declared = {API: ("tools/hpc3/artifacts",)}

        assert archive_pathspec(declared, remote=MCPS, project_path="x") == ()

    def test_a_repository_declaring_an_empty_list_is_scoped_to_nothing(self) -> None:
        assert archive_pathspec({MCPS: ()}, remote=MCPS, project_path="packages/maketools") == ()

    def test_the_whole_tree_leads_and_every_exclusion_follows_it(self) -> None:
        spec = archive_pathspec(
            {API: ("a/data", "b/data")}, remote=API, project_path="libs/monorepo_guards"
        )

        assert spec == (WHOLE_TREE, f"{EXCLUDE}a/data", f"{EXCLUDE}b/data")

    def test_declaration_order_is_kept(self) -> None:
        """A reader comparing the command against the registry reads down it."""
        spec = archive_pathspec({API: ("z/data", "a/data")}, remote=API, project_path="p")

        assert spec == (WHOLE_TREE, f"{EXCLUDE}z/data", f"{EXCLUDE}a/data")

    def test_the_project_s_own_data_survives_its_own_check(self) -> None:
        spec = archive_pathspec(
            {API: ("libs/instrument_io/tests/fixtures", "tools/hpc3/artifacts")},
            remote=API,
            project_path="libs/instrument_io",
        )

        assert spec == (WHOLE_TREE, f"{EXCLUDE}tools/hpc3/artifacts")

    def test_a_project_owning_every_declared_directory_is_scoped_to_nothing(self) -> None:
        """Down to the empty tuple, not down to a bare ``.``: there is
        nothing to subtract, so there is nothing to say."""
        spec = archive_pathspec(
            {API: ("libs/instrument_io/tests/fixtures",)},
            remote=API,
            project_path="libs/instrument_io",
        )

        assert spec == ()

    def test_a_project_that_is_the_repository_is_scoped_to_nothing(self) -> None:
        assert archive_pathspec({API: ("a/data",)}, remote=API, project_path="") == ()


# ---------------------------------------------------------------------------
# The real repository
# ---------------------------------------------------------------------------


def _tracked_sizes() -> dict[str, int]:
    """Every tracked file at HEAD and its blob size.

    Returns:
        Repo-relative posix path to size in bytes.
    """
    _require_git()
    sizes: dict[str, int] = {}
    for line in _git("ls-tree", "-r", "--long", "HEAD").splitlines():
        fields = line.split(None, 4)
        if len(fields) < 5:
            continue
        sizes[fields[4].strip()] = int(fields[3])
    return sizes


def _directory_weights(sizes: dict[str, int]) -> collections.Counter[str]:
    """Total tracked bytes under every directory prefix.

    Args:
        sizes: Per-file sizes from :func:`_tracked_sizes`.

    Returns:
        Directory path to the bytes beneath it, every depth counted.
    """
    weights: collections.Counter[str] = collections.Counter()
    for name, size in sizes.items():
        segments = name.split("/")
        for depth in range(1, len(segments)):
            weights["/".join(segments[:depth])] += size
    return weights


def undeclared_heavy_directories(declared: tuple[str, ...]) -> list[tuple[int, str]]:
    """Heavy directories the registry says nothing about, heaviest first.

    DIAGNOSTIC, NEVER A VERDICT, and that is a deliberate retreat. The first
    version of this classified each directory as data or as source so it
    could fail on its own, using mean bytes per file. Measured across this
    repository the two classes sit at 32 KB and 45 KB per file -- source
    tops out at ``libs/covenant_ml`` and data bottoms out at
    ``tools/hpc3/artifacts`` -- and a threshold with 12 KB of daylight
    either side is a coin toss that would start failing honest commits. So
    nothing here decides anything. The payload's SIZE is the verdict
    (:meth:`TestAgainstThisRepository.test_the_scoped_payload_stays_small`),
    and this list exists to tell whoever reads that failure WHAT grew,
    which a byte count on its own cannot.

    A directory is reported only where the weight actually SITS: when a
    single child holds more than 90% of it, the child is the real subject
    and the parent is just the road to it.

    Args:
        declared: The data directories the registry declares. Passed rather
            than read so the walk can be watched FINDING something: handed
            an empty declaration it must name the directories the registry
            really carries.

    Returns:
        ``(bytes, path)`` per undeclared directory over
        :data:`SIZE_FLOOR_BYTES`, heaviest first.
    """
    weights = _directory_weights(_tracked_sizes())
    found: list[tuple[int, str]] = []
    for path, weight in weights.items():
        if weight < SIZE_FLOOR_BYTES:
            continue
        if any(path == d or path.startswith(f"{d}/") or d.startswith(f"{path}/") for d in declared):
            continue
        children = [
            c for c in weights if c.startswith(f"{path}/") and c.count("/") == path.count("/") + 1
        ]
        if children and max(weights[c] for c in children) > 0.9 * weight:
            continue
        found.append((weight, path))
    return sorted(found, reverse=True)


class TestAgainstThisRepository:
    def test_every_declared_directory_still_exists(self) -> None:
        """A declaration for a path nobody has any more excludes nothing, and
        reads like protection that is not there."""
        tracked = _tracked_sizes()
        missing = [
            path for path in _declared() if not any(name.startswith(f"{path}/") for name in tracked)
        ]

        assert missing == []

    def test_the_walk_names_the_real_directories_when_nothing_is_declared(self) -> None:
        """WATCH IT FIND SOMETHING, against this repository, not a fixture.

        The diagnostic is only useful in the moment a payload has grown and
        nobody knows why, which is a moment nothing else here rehearses. A
        walk whose heuristics quietly stopped matching would return nothing
        and read exactly like a clean repository. Handed an empty
        declaration it must name the two directories this row is about, with
        the weights git reports for them.

        ``libs`` sits between the two in the answer and that is correct
        rather than noise: with nothing declared it really does hold 166 MB
        that no declaration accounts for, and its heaviest child is only 75%
        of it, so it is where weight sits. Which is the reason this walk
        does not get to fail a build on its own.
        """
        found = undeclared_heavy_directories(())
        weights = {path: weight for weight, path in found}
        tracked = _directory_weights(_tracked_sizes())

        assert found[0][1] == "services/covenant-radar-api/data/external"
        assert (
            weights["services/covenant-radar-api/data/external"]
            == (tracked["services/covenant-radar-api/data/external"])
        )
        assert (
            weights["libs/instrument_io/tests/fixtures"]
            == (tracked["libs/instrument_io/tests/fixtures"])
        )

    def test_the_scoped_payload_stays_small(self, scoped_tar: pathlib.Path) -> None:
        """THE DRIFT ALARM, and the only loud thing standing against the one
        quiet failure this design has.

        A new corpus, fixture set or artifact dump lands, nobody declares
        it, and it rides along in every payload for every project forever.
        Nothing in the scope notices, because a declaration cannot know
        about what was never declared. This does: the payload is measured,
        and when it has grown the message names the undeclared directories
        by weight so the reader is not left grepping a 600 MB tree.

        The ceiling is absolute rather than relative to the repository,
        because a relative one rises to meet the very thing it is watching
        for. 40 MB against today's 27.5 leaves room for honest growth, and
        raising it is a number in a diff that somebody has to justify --
        which is the decision this alarm exists to force.
        """
        scoped = scoped_tar.stat().st_size
        worst = undeclared_heavy_directories(_declared())[:5]
        report = "\n".join(f"  {weight:>13,}  {path}" for weight, path in worst)

        assert scoped < 40_000_000, (
            f"the scoped payload is {scoped:,} bytes, over the 40,000,000 ceiling. "
            f"The heaviest undeclared directories are:\n{report}\n"
            "Declare the data one in fleet.json's data_paths, or raise the ceiling "
            "here and say why."
        )

    def test_the_declared_directories_are_most_of_this_repository(self) -> None:
        """Non-vacuity. If the declarations stopped matching real paths this
        would still be green with an empty exclusion doing nothing, so the
        weight is asserted rather than the count."""
        sizes = _tracked_sizes()
        weights = _directory_weights(sizes)
        declared_bytes = sum(weights[path] for path in _declared())

        assert declared_bytes > 0.75 * sum(sizes.values())


@pytest.fixture(scope="module")
def scoped_tar(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build the archive a dispatch of ``libs/monorepo_guards`` would stage.

    Args:
        tmp_path_factory: pytest's session temporary directory factory.

    Returns:
        Path to the written tarball.
    """
    _require_git()
    spec = archive_pathspec({API: _declared()}, remote=API, project_path="libs/monorepo_guards")
    destination = tmp_path_factory.mktemp("scoped") / "scoped.tgz"
    _git("archive", "--format=tar.gz", "-o", str(destination), "HEAD", "--", *spec)
    return destination


@pytest.fixture(scope="module")
def names(scoped_tar: pathlib.Path) -> frozenset[str]:
    """Every path inside the scoped archive.

    Args:
        scoped_tar: The built tarball.

    Returns:
        The member names.
    """
    with tarfile.open(scoped_tar, "r:gz") as archive:
        return frozenset(archive.getnames())


class TestTheStagedTreeItself:
    """The shape, read off the filesystem, wherever this suite is running.

    THIS IS THE STRONGER EVIDENCE OF THE TWO AND IT ONLY EXISTS BECAUSE A
    DISPATCH FAILED. The tar inspection below proves what the hub would
    build; this proves what a node actually received, because on a node
    this tree IS the scoped export, unpacked. Every assertion here holds in
    a checkout too, so nothing is only ever checked in one place.
    """

    def test_the_tree_holds_every_package_s_manifest_and_guard_shim(self) -> None:
        """``libs/monorepo_guards`` asserts at least forty package roots and
        no unshimmed package, over the whole monorepo. If a scope ever stops
        carrying the shape, that package fails on a node and this says why."""
        manifests = list(REPO_ROOT.glob("*/*/pyproject.toml"))
        shims = list(REPO_ROOT.glob("*/*/scripts/guard.py"))

        assert len(manifests) >= 40
        assert len(shims) >= 40

    def test_the_tree_holds_the_workflows_that_package_parses(self) -> None:
        workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.y*ml"))

        assert len(workflows) > 5

    def test_the_tree_holds_the_build_files_every_makefile_reaches_for(self) -> None:
        """Neither is a poetry path dependency, and every project's Makefile
        dies at parse time without them. An include list would have dropped
        both, which is the measured reason this scope excludes instead."""
        assert (REPO_ROOT / "scripts" / "make" / "shell.mk").is_file()
        assert (REPO_ROOT / "tools" / "maketools" / "scripts" / "run.py").is_file()

    def test_the_tree_holds_the_root_files_a_check_reads(self) -> None:
        assert (REPO_ROOT / "monorepo-guards.toml").is_file()
        assert (REPO_ROOT / ".ruff.toml").is_file()
        assert (REPO_ROOT / "Makefile").is_file()

    def test_a_staged_export_carries_none_of_the_declared_data(self) -> None:
        """ON A NODE THIS IS THE WHOLE CLAIM, MEASURED ON THE DELIVERED BYTES.

        ``tools/fleet`` owns none of the eight declared directories, so a
        dispatch of it must arrive with all eight absent. In a checkout they
        are all present and correctly so, which is why this asserts only
        where the tree is an export -- and says which of the two it saw
        rather than passing quietly either way.
        """
        present = [path for path in _declared() if (REPO_ROOT / path).exists()]

        if _is_a_git_checkout():
            assert present == list(_declared()), (
                "a checkout must still hold its own data directories; the scope removes them "
                "from an ARCHIVE and must never touch the working tree"
            )
        else:
            assert present == [], (
                "this tree is a staged export of a project that owns none of the declared data "
                f"directories, so all of them must be absent; these arrived anyway: {present}"
            )


class TestTheArchiveItKeeps:
    def test_it_keeps_every_package_s_manifest_and_guard_shim(self, names: frozenset[str]) -> None:
        """WHY THIS IS NOT AN INCLUDE LIST, asserted rather than explained.

        ``libs/monorepo_guards`` asserts at least forty package roots and no
        unshimmed package, over the whole monorepo. Its dispatched check sees
        exactly this tar, so if the scope ever stops carrying the shape, that
        package fails on a node and the reason is here.
        """
        manifests = [n for n in names if n.endswith("/pyproject.toml")]
        shims = [n for n in names if n.endswith("/scripts/guard.py")]

        assert len(manifests) >= 40
        assert len(shims) >= 40

    def test_it_keeps_the_workflows_that_package_parses(self, names: frozenset[str]) -> None:
        workflows = [
            n for n in names if n.startswith(".github/workflows/") and n.endswith((".yml", ".yaml"))
        ]

        assert len(workflows) > 5

    def test_it_keeps_the_build_files_every_makefile_reaches_for(
        self, names: frozenset[str]
    ) -> None:
        """Neither is a poetry path dependency, and every project's Makefile
        dies at parse time without them."""
        assert "scripts/make/shell.mk" in names
        assert "tools/maketools/scripts/run.py" in names

    def test_it_keeps_the_root_files_a_check_reads(self, names: frozenset[str]) -> None:
        assert "monorepo-guards.toml" in names
        assert ".ruff.toml" in names
        assert "Makefile" in names

    def test_it_keeps_the_source_of_the_projects_whose_data_it_drops(
        self, names: frozenset[str]
    ) -> None:
        """Dropping a project's data must not drop the project."""
        assert "libs/instrument_io/pyproject.toml" in names
        assert any(n.startswith("libs/instrument_io/src/") for n in names)

    def test_it_drops_every_declared_directory(self, names: frozenset[str]) -> None:
        carried = [path for path in _declared() if any(n.startswith(f"{path}/") for n in names)]

        assert carried == []

    def test_it_is_far_smaller_than_the_commit_it_came_from(self, scoped_tar: pathlib.Path) -> None:
        """The measurement the whole row is about, re-taken on every run.

        Measured 2026-09-25 at f560de52: 216,826,747 bytes unscoped against
        27,462,327 scoped, 87.3% off. Asserted as a ratio rather than as
        those two numbers, because the repository grows and a pinned byte
        count would fail for the wrong reason; what must not come back is
        the ORDER OF MAGNITUDE.
        """
        scoped = scoped_tar.stat().st_size
        tracked = sum(_tracked_sizes().values())

        assert scoped < 0.1 * tracked
