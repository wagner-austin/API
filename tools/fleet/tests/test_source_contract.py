"""The ``source`` a project declares (MCPs board task fd5cabfa, A2): its
remote, its path inside the repository and its install steps, every grammar
refusal named, and the queue row's ``requiredTags`` and ``sha`` fields the
same runner reads.

Every refusal here is a fleet.json line somebody will write wrong once. The
messages are asserted because they are what that person reads.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from fleet.contracts.dispatch import decode_claim
from fleet.contracts.project import decode_project_config, encode_project_config
from fleet.contracts.source import (
    INSTALL_TOKEN,
    PATH_PATTERN,
    REMOTE_PATTERN,
    ProjectCompanion,
    ProjectSource,
    decode_companion_directory,
    decode_companion_ref,
    decode_companions,
    decode_install,
    decode_path,
    decode_project_source,
    decode_remote,
    encode_project_source,
)
from tests._queue_fakes import DEFAULT_SHA, queue_job
from tests.conftest import DEMO_PROJECT
from tests.test_contracts import _project
from tests.test_dispatch_contracts import claimed_job

HTTPS = "https://github.com/wagner-austin/MCPs.git"
SSH_SCP = "git@github.com:wagner-austin/MCPs.git"
SSH_URL = "ssh://git@github.com/wagner-austin/MCPs.git"


def _source(
    install: tuple[tuple[str, ...], ...] = (("npm", "ci"),),
    companions: tuple[ProjectCompanion, ...] = (),
) -> ProjectSource:
    """A source for the MCPs wiki-search package.

    Args:
        install: The install steps.
        companions: The repositories exported beside it.

    Returns:
        The source.
    """
    return ProjectSource(
        remote=HTTPS, path="packages/wiki-search", install=install, companions=companions
    )


class TestRemote:
    @pytest.mark.parametrize("remote", [HTTPS, SSH_SCP, SSH_URL])
    def test_the_three_remote_forms_pass(self, remote: str) -> None:
        assert decode_remote(remote, field="source.remote") == remote
        assert [found.group(0) for found in REMOTE_PATTERN.finditer(remote)] == [remote]

    @pytest.mark.parametrize(
        "remote",
        [
            "http://github.com/wagner-austin/MCPs.git",  # not https
            "https://github.com",  # no path
            "https://github.com/wagner austin/MCPs.git",  # whitespace
            "https://github.com/wagner-austin/MCPs.git;rm",  # metacharacter
            "/c/Users/Test/PROJECTS/MCPs",  # a local path is not a remote
            "",
        ],
    )
    def test_anything_outside_the_grammar_is_refused_by_field(self, remote: str) -> None:
        with pytest.raises(JSONTypeError, match=r"source\.remote must be an https://") as raised:
            decode_remote(remote, field="source.remote")

        assert repr(remote) in str(raised.value)


class TestPath:
    @pytest.mark.parametrize("path", ["", "tools/fleet", "packages/wiki-search", "a.b_c-d/e"])
    def test_the_root_and_slash_joined_segments_pass(self, path: str) -> None:
        assert decode_path(path, field="source.path") == path
        assert [found.group(0) for found in PATH_PATTERN.finditer(path)] == [path]

    @pytest.mark.parametrize(
        "path",
        ["/tools", "tools/", "tools//fleet", "../fleet", "./fleet", "tools fleet", ".hidden"],
    )
    def test_a_dot_segment_an_empty_segment_or_a_stranger_is_refused(self, path: str) -> None:
        with pytest.raises(JSONTypeError, match=r"source\.path must be '' for the repository root"):
            decode_path(path, field="source.path")


class TestInstall:
    def test_steps_decode_to_tuples_of_tokens(self) -> None:
        decoded = decode_install(
            [["npm", "ci"], ["npx", "playwright", "install", "chromium"]], field="source.install"
        )

        assert decoded == (("npm", "ci"), ("npx", "playwright", "install", "chromium"))

    def test_an_empty_list_is_a_project_that_installs_itself(self) -> None:
        assert decode_install([], field="source.install") == ()

    def test_an_absent_key_is_refused_rather_than_read_as_no_steps(self) -> None:
        with pytest.raises(JSONTypeError, match=r"source\.install is required: \[\]"):
            decode_install(None, field="source.install")

    def test_a_non_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a list of argv lists, got str"):
            decode_install("npm ci", field="source.install")

    def test_a_step_that_is_not_a_list_is_refused_by_index(self) -> None:
        with pytest.raises(JSONTypeError, match=r"source\.install\[1\] must be a list of argv"):
            decode_install([["npm", "ci"], "npm rebuild"], field="source.install")

    def test_an_empty_step_is_refused_by_index(self) -> None:
        with pytest.raises(JSONTypeError, match=r"source\.install\[0\] is empty"):
            decode_install([[]], field="source.install")

    @pytest.mark.parametrize(
        "token", ["npm ci", "rm;", "$HOME", "'quoted'", "a|b", "@splat", 7, None, "a\nb", ""]
    )
    def test_a_token_carrying_shell_syntax_is_refused_by_position(
        self, token: str | int | None
    ) -> None:
        with pytest.raises(JSONTypeError, match=r"source\.install\[0\]\[1\] must be a string"):
            decode_install([["npm", token]], field="source.install")

    @pytest.mark.parametrize(
        "token",
        [
            "npm",
            "node",
            "scripts/ci-build.mjs",
            "--ignore-scripts",
            "-g",
            "KEY=value",
            "a:b",
            "1.2.3",
            "playwright@1.47",
        ],
    )
    def test_the_tokens_the_registry_uses_pass(self, token: str) -> None:
        assert [found.group(0) for found in INSTALL_TOKEN.finditer(token)] == [token]
        assert decode_install([["npm", token]], field="source.install") == (("npm", token),)


class TestCompanions:
    """The repositories a project's export carries beside it (MCPs board task
    0515040d). Every refusal here is a fleet.json line somebody will write
    wrong once, and the message is what that person reads."""

    def test_a_declaration_decodes_to_its_three_fields(self) -> None:
        assert decode_companions(
            [{"remote": HTTPS, "ref": "main", "directory": "MCPs"}], field="source.companions"
        ) == (ProjectCompanion(remote=HTTPS, ref="main", directory="MCPs"),)

    def test_an_empty_list_is_a_project_whose_check_reads_its_own_tree(self) -> None:
        assert decode_companions([], field="source.companions") == ()

    def test_two_companions_in_two_directories_decode_in_declared_order(self) -> None:
        """Order is kept because it is the order they are staged in, and two
        different directories are exactly what the repeat check must allow."""
        assert decode_companions(
            [
                {"remote": HTTPS, "ref": "main", "directory": "MCPs"},
                {"remote": SSH_URL, "ref": "refs/heads/main", "directory": "workspace"},
            ],
            field="source.companions",
        ) == (
            ProjectCompanion(remote=HTTPS, ref="main", directory="MCPs"),
            ProjectCompanion(remote=SSH_URL, ref="refs/heads/main", directory="workspace"),
        )

    def test_an_absent_key_is_refused_rather_than_read_as_none(self) -> None:
        with pytest.raises(JSONTypeError, match=r"source\.companions is required: \[\]"):
            decode_companions(None, field="source.companions")

    def test_a_non_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a list of companions, got str"):
            decode_companions(HTTPS, field="source.companions")

    def test_an_element_that_is_not_an_object_is_refused_by_index(self) -> None:
        with pytest.raises(JSONTypeError, match=r"source\.companions\[0\] must be an object"):
            decode_companions([HTTPS], field="source.companions")

    def test_two_companions_in_one_directory_are_refused_by_index(self) -> None:
        """The second would stage over the first and the recipe would read
        whichever finished last."""
        with pytest.raises(
            JSONTypeError, match=r"source\.companions\[1\]\.directory repeats 'MCPs'"
        ):
            decode_companions(
                [
                    {"remote": HTTPS, "ref": "main", "directory": "MCPs"},
                    {"remote": SSH_URL, "ref": "master", "directory": "MCPs"},
                ],
                field="source.companions",
            )

    @pytest.mark.parametrize("ref", ["main", "master", "refs/heads/main", "release-1.2"])
    def test_the_refs_a_registry_would_name_pass(self, ref: str) -> None:
        assert decode_companion_ref(ref, field="source.companions[0].ref") == ref

    @pytest.mark.parametrize(
        "ref",
        ["", "/main", "main/", "../main", "-main", "refs//heads", "ma in", "main;rm", "$HEAD"],
    )
    def test_a_ref_outside_the_grammar_is_refused_with_the_two_spellings(self, ref: str) -> None:
        with pytest.raises(JSONTypeError, match=r"such as 'main' or 'refs/heads/main'"):
            decode_companion_ref(ref, field="source.companions[0].ref")

    @pytest.mark.parametrize("directory", ["MCPs", "workspace-1", "a.b_c-d"])
    def test_one_segment_passes_as_a_directory(self, directory: str) -> None:
        assert (
            decode_companion_directory(directory, field="source.companions[0].directory")
            == directory
        )

    @pytest.mark.parametrize(
        "directory", ["", "..", "../MCPs", "a/b", "/MCPs", "C:/fleet/stage/MCPs", ".hidden", "-x"]
    )
    def test_a_directory_that_is_a_path_is_refused(self, directory: str) -> None:
        """It is joined to the node's stage root, so a value that could spell
        ``..`` or an absolute path would write outside the staging area the
        runner owns."""
        with pytest.raises(JSONTypeError, match=r"must be ONE directory name"):
            decode_companion_directory(directory, field="source.companions[0].directory")

    def test_a_source_carrying_one_survives_encoding(self) -> None:
        original = _source(
            companions=(ProjectCompanion(remote=HTTPS, ref="main", directory="MCPs"),)
        )

        assert encode_project_source(original) == {
            "remote": HTTPS,
            "path": "packages/wiki-search",
            "install": [["npm", "ci"]],
            "companions": [{"remote": HTTPS, "ref": "main", "directory": "MCPs"}],
        }
        assert decode_project_source(encode_project_source(original), field="source") == original


class TestProjectSource:
    def test_a_source_survives_encoding(self) -> None:
        original = _source()

        assert (
            decode_project_source(
                load_json_str(dump_json_str(encode_project_source(original))), field="source"
            )
            == original
        )

    def test_a_source_with_no_steps_survives_encoding(self) -> None:
        original = _source(install=())

        assert encode_project_source(original) == {
            "remote": HTTPS,
            "path": "packages/wiki-search",
            "install": [],
            "companions": [],
        }
        assert decode_project_source(encode_project_source(original), field="source") == original

    def test_a_declared_null_is_none_both_ways(self) -> None:
        assert decode_project_source(None, field="source") is None
        assert encode_project_source(None) is None

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="source must be an object or null, got list"):
            decode_project_source([HTTPS], field="source")

    def test_a_missing_remote_is_refused_by_name(self) -> None:
        with pytest.raises(JSONTypeError, match="remote"):
            decode_project_source({"path": "", "install": []}, field="source")

    def test_the_field_path_reaches_the_nested_refusal(self) -> None:
        with pytest.raises(JSONTypeError, match=r"projects\.x\.source\.path must be ''"):
            decode_project_source(
                {"remote": HTTPS, "path": "../x", "install": []}, field="projects.x.source"
            )


class TestProjectCarriesItsSource:
    def test_a_project_with_a_source_survives_encoding(self) -> None:
        original = {**_project(), "source": _source()}
        encoded = encode_project_config(_project())
        encoded["source"] = encode_project_source(_source())
        project = decode_project_config(encoded)

        assert project == original
        assert encode_project_config(project)["source"] == encode_project_source(_source())

    def test_an_absent_source_key_is_refused_not_read_as_no_remote(self) -> None:
        encoded = encode_project_config(_project())
        del encoded["source"]

        with pytest.raises(JSONTypeError, match="project must declare 'source', using null"):
            decode_project_config(encoded)


class TestQueueRowCheckoutFields:
    def test_a_node_lane_row_carries_its_sha_tags_and_task(self) -> None:
        job = claimed_job(
            dump_json_str(
                {
                    "claimed": queue_job(
                        status="claimed",
                        requiredTags=["gpu", "linux"],
                        taskId="fd5cabfa-a328-48f4-b5e9-3a02dd531ea5",
                    )
                }
            )
        )

        assert job["sha"] == DEFAULT_SHA
        assert job["required_tags"] == ("gpu", "linux")
        assert job["task_id"] == "fd5cabfa-a328-48f4-b5e9-3a02dd531ea5"
        assert job["project"] == DEMO_PROJECT

    def test_required_tags_that_are_not_an_array_are_refused(self) -> None:
        with pytest.raises(AppError, match="field 'requiredTags' is str, not an array"):
            decode_claim(dump_json_str({"claimed": queue_job(requiredTags="gpu")}))

    def test_a_missing_required_tags_key_is_refused(self) -> None:
        row = queue_job()
        del row["requiredTags"]

        with pytest.raises(AppError, match="field 'requiredTags' is NoneType, not an array"):
            decode_claim(dump_json_str({"claimed": row}))

    def test_a_tag_outside_the_derived_vocabulary_is_refused_by_index(self) -> None:
        with pytest.raises(AppError, match=r"requiredTags\[1\] 'macos' is not one of"):
            decode_claim(dump_json_str({"claimed": queue_job(requiredTags=["windows", "macos"])}))
