"""The registry's data-path declaration, and every way to write it wrong.

Its own module rather than a class in ``test_contracts.py``, which is at the
600-line ceiling: this is one cohesive subject -- what a repository declares
as data, and what the decoder refuses -- and it reads better beside
:mod:`tests.test_archive_scope`, which tests what the declaration is then
used FOR.

EVERY REFUSAL HERE IS A REGISTRY LINE THAT WOULD OTHERWISE BE FOUND ON A
NODE. A bad declaration does not fail where it is written; it strips the
wrong bytes off an archive, which is base64'd, sent over ssh, unpacked
somewhere else and then fails as a missing file, minutes later, with a
message about whatever was missing rather than about the line that removed
it. The decoder is the last cheap place to catch any of this.

The workspace documents are built here from literals rather than borrowed,
so what each case declares is visible in the case.
"""

from __future__ import annotations

from typing import Final

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from fleet.contracts.workspace import decode_fleet_workspace

#: The API monorepo, whose eight data directories are the real subject.
API: Final[str] = "https://github.com/wagner-austin/API.git"

#: A second repository, for the case that proves declarations do not leak
#: between repositories.
MCPS: Final[str] = "https://github.com/wagner-austin/MCPs.git"


def _document(
    *, data_paths: JSONValue, project_path: str = "libs/monorepo_guards", remote: str = API
) -> JSONObject:
    """A minimal but complete workspace document.

    Args:
        data_paths: What to put under ``data_paths``, valid or not.
        project_path: The single project's repo-relative path, ``""`` for a
            project that is the whole repository.
        remote: The remote that project's commits come from.

    Returns:
        The document, ready for :func:`decode_fleet_workspace`.
    """
    return {
        "nodes": {
            "lavender": {
                "host": "lavender",
                "platform": "windows",
                "stage_root": "C:/fleet/stage",
                "logical_cores": 24,
                "ram_gb": 63.7,
                "gpu": None,
                "enabled": True,
                "budget": {
                    "reserved_cores": 4,
                    "reserved_ram_gb": 8.0,
                    "worker_ram_gb": 1.1,
                    "max_concurrent_runs": 2,
                    "max_disk_gb": 20.0,
                },
            }
        },
        "not_dispatchable": {},
        "projects": {
            "the-project": {
                "worker_ram_gb": 1.1,
                "minimum_workers": 4,
                "expected_minutes": 5,
                "required_tags": [],
                "source": {
                    "remote": remote,
                    "path": project_path,
                    "install": [],
                    "companions": [],
                },
            }
        },
        "data_paths": data_paths,
        "ledger": "runs/ledger.jsonl",
        "feed": "runs/feed.jsonl",
        "leases": "runs/leases.json",
    }


class TestWhatItAccepts:
    def test_a_repository_s_directories_decode_in_declaration_order(self) -> None:
        """Order is kept because the pathspec is built in it, and a reader
        comparing the command against the registry reads down the list."""
        decoded = decode_fleet_workspace(
            _document(data_paths={API: ["services/covenant-radar-api/data/external", "libs/x/y"]})
        )

        assert decoded["data_paths"] == {
            API: ("services/covenant-radar-api/data/external", "libs/x/y")
        }

    def test_a_repository_with_nothing_to_leave_behind_says_so(self) -> None:
        """An empty list is a decision. An absent key is not, which is why
        the field is required rather than defaulted."""
        assert decode_fleet_workspace(_document(data_paths={MCPS: []}))["data_paths"] == {MCPS: ()}

    def test_a_fleet_declaring_nothing_at_all_is_accepted(self) -> None:
        assert decode_fleet_workspace(_document(data_paths={}))["data_paths"] == {}


class TestWhatItRefuses:
    def test_an_absent_field_is_refused_rather_than_read_as_none(self) -> None:
        """ "Nobody considered this" and "there is no data here" stage 217 MB
        and 27 MB respectively while looking identical in the file."""
        document = _document(data_paths={})
        del document["data_paths"]

        with pytest.raises(JSONTypeError, match="data_paths"):
            decode_fleet_workspace(document)

    def test_a_key_that_is_not_a_remote_is_refused(self) -> None:
        """The key selects which repository a path is relative to, so a key
        that is not a remote is a scope applied to nothing."""
        with pytest.raises(JSONTypeError, match="data_paths key"):
            decode_fleet_workspace(_document(data_paths={"the api repo": ["libs/x"]}))

    def test_a_repository_whose_value_is_not_a_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a list of repo-relative directories"):
            decode_fleet_workspace(_document(data_paths={API: "libs/x"}))

    def test_a_path_that_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match=r"data_paths\[.*\]\[0\] must be a string"):
            decode_fleet_workspace(_document(data_paths={API: [7]}))

    def test_the_repository_root_is_refused_as_a_data_path(self) -> None:
        """``""`` is a legal PROJECT path, meaning the whole repository, and
        would decode fine as one. As a data path it means exclude
        everything: the archive comes out empty and fails on the node as a
        missing Makefile, a long way from the line that emptied it."""
        with pytest.raises(JSONTypeError, match="is the repository root"):
            decode_fleet_workspace(_document(data_paths={API: [""]}))

    def test_a_path_climbing_out_of_the_repository_is_refused(self) -> None:
        """These become git pathspec terms. The path grammar is what stops a
        declaration reaching somewhere the archive was never meant to go."""
        with pytest.raises(JSONTypeError, match="data_paths"):
            decode_fleet_workspace(_document(data_paths={API: ["../elsewhere"]}))

    def test_a_path_containing_a_dispatchable_project_is_refused(self) -> None:
        """The workspace would be saying one directory is both work to
        dispatch and data no export carries. The export wins silently, by
        staging a tree with the project missing from it."""
        with pytest.raises(JSONTypeError, match="which contains the dispatchable project"):
            decode_fleet_workspace(
                _document(data_paths={API: ["libs"]}, project_path="libs/monorepo_guards")
            )

    def test_a_path_that_is_exactly_a_project_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="which contains the dispatchable project"):
            decode_fleet_workspace(
                _document(
                    data_paths={API: ["libs/monorepo_guards"]},
                    project_path="libs/monorepo_guards",
                )
            )

    def test_any_path_is_refused_for_a_project_that_is_the_repository(self) -> None:
        """A project declaring path ``""`` owns the whole tree, so every data
        directory lies inside it and none would ever be excluded from its
        export. Refused rather than accepted and ignored: a declaration that
        silently does nothing is worse than one that is rejected, because
        somebody wrote it believing it did something."""
        with pytest.raises(JSONTypeError, match="which contains the dispatchable project"):
            decode_fleet_workspace(_document(data_paths={API: ["any/data"]}, project_path=""))


class TestScopeDoesNotLeakBetweenRepositories:
    def test_a_project_of_another_remote_does_not_block_a_declaration(self) -> None:
        """``services`` declared for MCPs must not be judged against a
        project of the API monorepo that happens to live under that name. A
        repo-relative path means nothing outside its own repository."""
        decoded = decode_fleet_workspace(
            _document(
                data_paths={MCPS: ["services"]},
                project_path="services/Model-Trainer",
                remote=API,
            )
        )

        assert decoded["data_paths"] == {MCPS: ("services",)}

    def test_a_project_sharing_a_prefix_with_a_path_does_not_block_it(self) -> None:
        """``libs/instrument`` is not a parent of ``libs/instrument_io``, and
        a prefix comparison without the separator would say it is."""
        decoded = decode_fleet_workspace(
            _document(
                data_paths={API: ["libs/instrument_io/tests/fixtures"]},
                project_path="libs/instrument",
            )
        )

        assert decoded["data_paths"] == {API: ("libs/instrument_io/tests/fixtures",)}
