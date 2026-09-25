"""Which of a repository's files travel with a check, and which stay home.

THE PAYLOAD WAS THE WHOLE REPOSITORY, AND ALMOST ALL OF IT WAS DATA. Until
board task 140e7042 :func:`fleet.core.export.archive_commit` ran ``git
archive`` with no pathspec, so a dispatch of ``libs/monorepo_guards`` -- a
package of about a quarter of a megabyte -- staged every byte the API
monorepo tracks. Measured 2026-09-25 at ``f560de52``: 216,826,747 bytes,
43.2 seconds to build the tar, and that archive then had to be base64'd,
sent over ssh and digested at the far end for every single check.

WHAT IS ACTUALLY BIG. The monorepo tracks 610,853,185 bytes and eight
directories hold 521 MB of them: training corpora, instrument fixtures
recorded off real hardware, a vendored parser distribution, archived run
artifacts. None of it is read by any project's check but the one that owns
it. Declaring those eight in the registry takes the payload to 27,462,327
bytes and 2.7 seconds -- 87.3% smaller, 16 times faster to build.

WHY THIS EXCLUDES RATHER THAN INCLUDES, WHICH IS THE DESIGN DECISION HERE
AND THE ONE THAT WAS GOT WRONG FIRST. An include list -- the project's own
path, its poetry closure, the root files -- is the obvious shape and it does
not work, for two measured reasons:

* ``libs/monorepo_guards`` AUDITS THE REPOSITORY'S SHAPE. Its suite asserts
  ``len(package_roots(REPO_ROOT)) >= 40``, that no package lacks a guard
  shim, and that every real workflow under ``.github/workflows`` parses and
  is bounded. Handed its own path and its closure it would see one package
  root and fail. The whole point of that package is that it reads everything.
* EVERY PROJECT'S BUILD REACHES OUT OF ITS OWN TREE. Each Makefile opens
  ``include ../../scripts/make/shell.mk`` and calls
  ``../../tools/maketools/scripts/run.py``, and neither is a poetry path
  dependency, so no dependency closure names them. An include list would
  have died at make parse time on every node.

The shape of a repository is cheap -- 9,130 entries and 27 MB with the data
gone, including all 51 ``pyproject.toml``, all 51 ``scripts/guard.py`` and
all 7 workflow files. The data is what costs. So the data is what goes, and
everything else is carried without anyone having to have declared it.

A PROJECT KEEPS ITS OWN DATA. A declared directory inside the project being
checked is not excluded: ``libs/instrument_io``'s fixtures are 114 MB of
dead weight in every other project's archive and are the thing its own suite
reads. That is what makes this a scope rather than a blanket.

THE ONE QUIET FAILURE, STATED BECAUSE IT IS REAL. A heavy directory nobody
declares ships in full and nothing here notices; the registry is a
declaration and declarations drift. It is made loud outside this module, by
a test that fails when a tracked directory crosses a size floor without
being declared, in the same spirit as the guard shim audit above. This
module is deliberately not the place for that check -- it holds no opinion
about a working tree and runs against a bare mirror on the hub.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

#: The pathspec term for "everything", which the exclusions are subtracted
#: from. ``git archive`` needs a positive term before a negative one; with
#: only ``:(exclude)`` terms it matches nothing and writes an empty tar.
WHOLE_TREE: Final[str] = "."

#: Git's magic prefix for a negative pathspec term. Long form rather than
#: ``:!``, which a shell can read as a history expansion, and these terms are
#: also written into logs and board posts where the long form reads.
EXCLUDE: Final[str] = ":(exclude)"


def owns(project_path: str, data_path: str) -> bool:
    """Whether a data directory belongs to the project being checked.

    Args:
        project_path: The project's repo-relative path, ``""`` for a project
            that is the whole repository.
        data_path: A declared data directory, never the repository root.

    Returns:
        True when the directory lies inside the project, so the project's own
        check keeps it. A project that IS the repository owns every data
        directory in it, which is why ``slime`` and the MCPs packages keep
        the payload they have always had.

        Compared against ``project_path + "/"`` rather than by string prefix,
        so ``libs/instrument`` does not claim ``libs/instrument_io``'s
        fixtures.
    """
    if not project_path:
        return True
    return data_path == project_path or data_path.startswith(f"{project_path}/")


def archive_pathspec(
    data_paths: Mapping[str, tuple[str, ...]], *, remote: str, project_path: str
) -> tuple[str, ...]:
    """The pathspec ``git archive`` is given for one project's export.

    Args:
        data_paths: The workspace's declarations, keyed by remote.
        remote: The remote the project's commits come from, which selects
            the repository whose declarations apply. A repo-relative path
            means nothing outside its own repository, so a declaration made
            for one is never applied to another.
        project_path: The project's repo-relative path, ``""`` for a project
            that is the whole repository.

    Returns:
        ``(".", ":(exclude)<path>", ...)`` for a repository with data to
        leave out, and the EMPTY TUPLE when there is none -- for a repository
        that declares no data paths, and for a project that owns every one
        its repository declares.

        Empty rather than ``(".",)`` so the caller runs the command it has
        always run, byte for byte, when nothing is being scoped. A project
        whose payload is not meant to change must not depend on ``.`` and no
        pathspec being equivalent; here it does not have to be.

        The exclusions come back in declaration order, which is the order the
        registry lists them in and so the order a reader comparing the
        command against the registry expects.
    """
    declared = data_paths.get(remote, ())
    excluded = tuple(path for path in declared if not owns(project_path, path))
    if not excluded:
        return ()
    return (WHOLE_TREE, *(f"{EXCLUDE}{path}" for path in excluded))


__all__ = ["EXCLUDE", "WHOLE_TREE", "archive_pathspec", "owns"]
