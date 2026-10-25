"""Tags are derived from what a node measures, and a project's requirement is a closed set.

Every case here is the reason the tags are not a column on the node: a
declared ``gpu`` tag would outlive the card it described, while one derived
from ``gpu`` on the node contract is exactly as true as that field.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig, NodeGpu
from fleet.contracts.tags import (
    TOOL_TAG,
    NodeTag,
    decode_node_tag,
    decode_required_tags,
    encode_tags,
    missing_tags,
    node_tags,
    tool_tags,
)
from fleet.contracts.toolchain import ToolReport

#: sedona's card as the registry declares it.
RTX_3070_TI = NodeGpu(
    model="NVIDIA GeForce RTX 3070 Ti Laptop GPU",
    vram_mib=8192,
    compute_capability="8.6",
    driver_version="551.23",
)


def _node(
    *,
    gpu: NodeGpu | None = None,
    test_database: bool = False,
    rust: str | None = None,
    cxx: str | None = None,
    docker: str | None = None,
    stack: str | None = None,
) -> NodeConfig:
    """Build a node declaration with the six fields tags derive from.

    Args:
        gpu: Its CUDA device, or None for a CPU-only node.
        test_database: Whether it runs the fleet test database.
        rust: The cargo version it declares, or None.
        cxx: The C++ toolchain version it declares, or None.
        docker: The rootless execdocker daemon version it declares, or None.
        stack: The stack daemon version it declares, or None.

    Returns:
        The node.
    """
    return NodeConfig(
        host="sedona",
        stage_root="/home/corvis/fleet/stage",
        logical_cores=20,
        ram_gb=16.0,
        gpu=gpu,
        enabled=True,
        test_database=test_database,
        rust=rust,
        cxx=cxx,
        docker=docker,
        stack=stack,
        budget=NodeBudget(
            reserved_cores=4,
            reserved_ram_gb=4.0,
            worker_ram_gb=1.1,
            max_disk_gb=40.0,
            checks_at_once=None,
            pinned_checks_only=False,
        ),
    )


class TestNodeTags:
    def test_a_cpu_only_node_carries_linux_alone(self) -> None:
        assert node_tags(_node()) == frozenset({NodeTag.LINUX})

    def test_a_node_with_a_card_carries_both(self) -> None:
        assert node_tags(_node(gpu=RTX_3070_TI)) == frozenset({NodeTag.LINUX, NodeTag.GPU})

    def test_a_node_running_the_fleet_test_database_carries_testdb(self) -> None:
        """diphtheria's shape once provisioned (MCPs board task 6bbfd171)."""
        assert node_tags(_node(gpu=RTX_3070_TI, test_database=True)) == (
            frozenset({NodeTag.LINUX, NodeTag.GPU, NodeTag.TESTDB})
        )

    def test_a_node_declaring_a_rust_toolchain_carries_rust(self) -> None:
        """diphtheria's shape once rustup was installed (MCPs board task 1e2da299)."""
        assert node_tags(_node(test_database=True, rust="1.98.1")) == (
            frozenset({NodeTag.LINUX, NodeTag.TESTDB, NodeTag.RUST})
        )
        assert NodeTag.RUST not in node_tags(_node())

    def test_a_node_declaring_a_cxx_toolchain_carries_cxx(self) -> None:
        """diphtheria's g++ 13.3.0, the one node node-gyp could build on
        (MCPs board task 3f19c136); a node declaring none carries no cxx."""
        assert node_tags(_node(rust="1.98.1", cxx="13.3.0")) == (
            frozenset({NodeTag.LINUX, NodeTag.RUST, NodeTag.CXX})
        )
        assert NodeTag.CXX not in node_tags(_node())

    def test_a_node_declaring_the_rootless_exec_daemon_carries_docker(self) -> None:
        """diphtheria once provisioned with execdocker's rootless daemon (MCPs
        board task 6c4516af); a node declaring none carries no docker."""
        assert node_tags(_node(docker="29.8.1")) == frozenset({NodeTag.LINUX, NodeTag.DOCKER})
        assert NodeTag.DOCKER not in node_tags(_node())

    def test_a_node_declaring_the_stack_carries_stack(self) -> None:
        """diphtheria, whose own daemon holds mcp-network and the stack's
        images, carries it; lavender-wsl, the second testdb node, declares
        none and carries no stack (MCPs board task 554bffc1)."""
        diphtheria = _node(test_database=True, cxx="13.3.0", docker="29.8.1", stack="29.8.1")
        lavender_wsl = _node(test_database=True, cxx="13.3.0", docker="29.1.3")
        assert node_tags(diphtheria) == frozenset(
            {NodeTag.LINUX, NodeTag.TESTDB, NodeTag.CXX, NodeTag.DOCKER, NodeTag.STACK}
        )
        assert node_tags(lavender_wsl) == frozenset(
            {NodeTag.LINUX, NodeTag.TESTDB, NodeTag.CXX, NodeTag.DOCKER}
        )
        required = (NodeTag.TESTDB, NodeTag.CXX, NodeTag.STACK)
        assert missing_tags(node_tags(lavender_wsl), required) == (NodeTag.STACK,)
        assert missing_tags(node_tags(diphtheria), required) == ()
        assert missing_tags(node_tags(lavender_wsl), (NodeTag.TESTDB, NodeTag.CXX)) == ()

    def test_the_vocabulary_is_linux_and_eleven_capabilities(self) -> None:
        """The dispatch queue's CHECK (MCPs migrations 532, 563, 569, 570, 571,
        615, 622, 639, 648, 668, 707 and 722) admits these twelve words, so
        the members' values are pinned in order, and neither ``windows`` nor
        ``elevated`` is one of them (board task e705f08c)."""
        assert [tag.value for tag in NodeTag] == [
            "linux",
            "gpu",
            "testdb",
            "rust",
            "cxx",
            "docker",
            "stack",
            "ffmpeg",
            "hooks",
            "go",
            "chrome",
            "pdftotext",
        ]

    def test_no_declaration_carries_a_tool_tag(self) -> None:
        """ffmpeg, hooks, go, chrome and pdftotext come only from the toolchain probe
        (:func:`tool_tags`), so a node with every declaration set carries
        none of them."""
        loaded = _node(
            gpu=RTX_3070_TI,
            test_database=True,
            rust="1.98.1",
            cxx="13.3.0",
            docker="29.8.1",
            stack="29.8.1",
        )
        assert NodeTag.FFMPEG not in node_tags(loaded)
        assert NodeTag.HOOKS not in node_tags(loaded)
        assert NodeTag.GO not in node_tags(loaded)
        assert NodeTag.CHROME not in node_tags(loaded)
        assert NodeTag.PDFTOTEXT not in node_tags(loaded)


class TestToolTags:
    def test_a_probe_that_found_ffmpeg_gives_its_tag(self) -> None:
        """MCPs board task 939ec5c7: a node claims an ffmpeg suite on the
        tick its probe first finds the executable."""
        found = (
            ToolReport(name="git", present=True, version="2.51.0"),
            ToolReport(name="ffmpeg", present=True, version="6.1.1-3ubuntu5"),
        )
        assert tool_tags(found) == frozenset({NodeTag.FFMPEG})

    def test_a_probe_that_found_the_hooks_environment_gives_its_tag(self) -> None:
        """A node once its interpreter carries the check's tools (MCPs board
        task ec895824): the line answers with the route file it found."""
        found = (
            ToolReport(
                name="hooks", present=True, version="/home/corvis/.claude/corvis-hooks.json"
            ),
        )
        assert tool_tags(found) == frozenset({NodeTag.HOOKS})
        assert tool_tags((ToolReport(name="hooks", present=False, version=""),)) == frozenset()

    def test_an_absent_or_unreported_tool_gives_nothing(self) -> None:
        absent = (ToolReport(name="ffmpeg", present=False, version=""),)
        assert tool_tags(absent) == frozenset()
        assert tool_tags(()) == frozenset()

    def test_a_probe_that_found_go_gives_its_tag(self) -> None:
        """The hub's answer, go1.27.1, the one machine with go on 2026-10-05
        (MCPs board task 1da15750); every fleet node answered go=no=."""
        found = (ToolReport(name="go", present=True, version="go version go1.27.1 linux/amd64"),)
        assert tool_tags(found) == frozenset({NodeTag.GO})
        assert tool_tags((ToolReport(name="go", present=False, version=""),)) == frozenset()

    def test_a_probe_that_found_chrome_gives_its_tag(self) -> None:
        """diphtheria after its 2026-10-09 05:48Z install (MCPs board task
        2f596185): the line answers with the binary's own --version, and the
        node that answered chrome=no= all of 2026-10-04 and until then
        carries nothing for it."""
        found = (ToolReport(name="chrome", present=True, version="Google Chrome 155.0.8059.39"),)
        assert tool_tags(found) == frozenset({NodeTag.CHROME})
        assert tool_tags((ToolReport(name="chrome", present=False, version=""),)) == frozenset()

    def test_a_probe_that_found_pdftotext_gives_its_tag(self) -> None:
        """diphtheria, colossus and lavender-wsl on 2026-10-11 (MCPs board
        task 6dd73ada): the line answers with poppler's own -v, and a node
        without the tool carries nothing for it."""
        found = (ToolReport(name="pdftotext", present=True, version="pdftotext version 24.02.0"),)
        assert tool_tags(found) == frozenset({NodeTag.PDFTOTEXT})
        assert tool_tags((ToolReport(name="pdftotext", present=False, version=""),)) == frozenset()

    def test_only_the_tagged_tools_have_tags(self) -> None:
        assert TOOL_TAG == {
            "ffmpeg": NodeTag.FFMPEG,
            "hooks": NodeTag.HOOKS,
            "go": NodeTag.GO,
            "chrome": NodeTag.CHROME,
            "pdftotext": NodeTag.PDFTOTEXT,
        }
        assert tool_tags((ToolReport(name="make", present=True, version="4.4"),)) == frozenset()


class TestMissingTags:
    def test_a_satisfied_requirement_is_empty(self) -> None:
        carried = node_tags(_node(gpu=RTX_3070_TI))
        assert missing_tags(carried, (NodeTag.GPU, NodeTag.LINUX)) == ()

    def test_the_missing_tags_come_back_in_the_project_s_order(self) -> None:
        plain = node_tags(_node())
        assert missing_tags(plain, (NodeTag.TESTDB, NodeTag.GPU)) == (
            NodeTag.TESTDB,
            NodeTag.GPU,
        )
        assert missing_tags(plain, (NodeTag.GPU, NodeTag.TESTDB)) == (
            NodeTag.GPU,
            NodeTag.TESTDB,
        )

    def test_a_database_suite_is_missing_testdb_on_a_node_without_one(self) -> None:
        with_database = node_tags(_node(test_database=True))
        plain = node_tags(_node())
        assert missing_tags(plain, (NodeTag.TESTDB,)) == (NodeTag.TESTDB,)
        assert missing_tags(with_database, (NodeTag.TESTDB,)) == ()

    def test_a_crate_build_is_missing_rust_on_a_node_without_cargo(self) -> None:
        with_cargo = node_tags(_node(rust="1.98.1"))
        plain = node_tags(_node())
        assert missing_tags(plain, (NodeTag.RUST,)) == (NodeTag.RUST,)
        assert missing_tags(with_cargo, (NodeTag.RUST,)) == ()

    def test_an_audio_suite_is_missing_ffmpeg_until_the_probe_finds_it(self) -> None:
        """TankpitBot-execution's requirement against a node before and after
        its probe found ffmpeg (MCPs board task 939ec5c7)."""
        declared = node_tags(_node())
        audio = (NodeTag.FFMPEG,)
        assert missing_tags(declared, audio) == (NodeTag.FFMPEG,)
        assert missing_tags(declared | {NodeTag.FFMPEG}, audio) == ()

    def test_a_project_requiring_nothing_is_never_missing_anything(self) -> None:
        assert missing_tags(node_tags(_node()), ()) == ()


class TestDecodeNodeTag:
    def test_each_word_in_the_set_decodes_to_its_member(self) -> None:
        for tag in NodeTag:
            assert decode_node_tag(tag.value, field="t") is tag

    def test_a_word_outside_the_set_is_refused_with_the_set(self) -> None:
        with pytest.raises(
            JSONTypeError,
            match=r"^t must be one of linux, gpu, testdb, rust, cxx, docker, stack, ffmpeg, "
            r"hooks, go, chrome, pdftotext, got 'podman'; .* a Rust or C\+\+ toolchain, the "
            r"execution suite's rootless Docker daemon, the corvis compose stack, or a tool, "
            r"the hooks check's environment or Google Chrome its probe found\), and one it "
            r"does not carry could never be satisfied: no node runs windows or an elevated "
            r"runner since board task e705f08c$",
        ):
            decode_node_tag("podman", field="t")

    @pytest.mark.parametrize("word", ["windows", "elevated"])
    def test_the_retired_words_are_refused_like_any_other(self, word: str) -> None:
        """A project row still naming either would wait forever, so it fails
        at decode instead (board task e705f08c)."""
        with pytest.raises(JSONTypeError, match=rf"got '{word}'; .* since board task e705f08c$"):
            decode_node_tag(word, field="t")

    def test_a_non_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="t must be a string, got int"):
            decode_node_tag(7, field="t")


class TestDecodeRequiredTags:
    def test_an_empty_list_is_a_suite_any_node_may_run(self) -> None:
        assert decode_required_tags([], field="p.required_tags") == ()

    def test_tags_keep_their_declared_order(self) -> None:
        assert decode_required_tags(["gpu", "linux"], field="p") == (
            NodeTag.GPU,
            NodeTag.LINUX,
        )

    def test_an_absent_key_is_refused_not_defaulted(self) -> None:
        with pytest.raises(
            JSONTypeError,
            match=r"p\.required_tags is required: \[\] for a suite any node may run",
        ):
            decode_required_tags(None, field="p.required_tags")

    def test_a_non_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="p must be a list of tags, got str"):
            decode_required_tags("gpu", field="p")

    def test_an_unknown_tag_is_refused_by_index(self) -> None:
        with pytest.raises(
            JSONTypeError, match=r"p\[1\] must be one of linux, gpu, testdb, rust, cxx"
        ):
            decode_required_tags(["gpu", "cuda"], field="p")

    def test_a_repeated_tag_is_refused_by_index_naming_its_word(self) -> None:
        with pytest.raises(JSONTypeError, match=r"^p\[1\] repeats 'gpu'$"):
            decode_required_tags(["gpu", "gpu"], field="p")


class TestEncodeTags:
    def test_tags_encode_as_their_words_in_order(self) -> None:
        encoded = encode_tags((NodeTag.GPU, NodeTag.LINUX))
        assert encoded == ["gpu", "linux"]
        assert [type(word) for word in encoded] == [str, str]
        assert encode_tags(()) == []
