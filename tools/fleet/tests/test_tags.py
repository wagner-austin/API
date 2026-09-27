"""Tags are derived from what a node measures, and a project's requirement is a closed set.

Every case here is the reason the tags are not a column on the node: a
declared ``gpu`` tag would outlive the card it described, while one derived
from ``gpu`` on the node contract is exactly as true as that field.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig, NodeGpu, NodePlatform
from fleet.contracts.tags import (
    NodeTag,
    decode_node_tag,
    decode_required_tags,
    encode_tags,
    missing_tags,
    node_tags,
)

#: sedona's card as fleet.json declares it.
RTX_3070_TI = NodeGpu(
    model="NVIDIA GeForce RTX 3070 Ti Laptop GPU",
    vram_mib=8192,
    compute_capability="8.6",
    driver_version="551.23",
)


def _node(
    *,
    platform: NodePlatform = NodePlatform.WINDOWS,
    gpu: NodeGpu | None = None,
    test_database: bool = False,
    rust: str | None = None,
    cxx: str | None = None,
    docker: str | None = None,
) -> NodeConfig:
    """Build a node declaration with the six fields tags derive from.

    Args:
        platform: The node's dialect.
        gpu: Its CUDA device, or None for a CPU-only node.
        test_database: Whether it runs the fleet test database.
        rust: The cargo version it declares, or None.
        cxx: The C++ toolchain version it declares, or None.
        docker: The rootless execdocker daemon version it declares, or None.

    Returns:
        The node.
    """
    return NodeConfig(
        host="sedona",
        platform=platform,
        stage_root="C:/fleet/stage",
        logical_cores=20,
        ram_gb=16.0,
        gpu=gpu,
        enabled=True,
        test_database=test_database,
        rust=rust,
        cxx=cxx,
        docker=docker,
        budget=NodeBudget(
            reserved_cores=4,
            reserved_ram_gb=4.0,
            worker_ram_gb=1.1,
            max_concurrent_runs=1,
            max_disk_gb=40.0,
        ),
    )


class TestNodeTags:
    def test_a_cpu_only_windows_node_carries_its_platform_alone(self) -> None:
        assert node_tags(_node()) == frozenset({NodeTag.WINDOWS})

    def test_a_linux_node_with_a_card_carries_both(self) -> None:
        assert node_tags(_node(platform=NodePlatform.LINUX, gpu=RTX_3070_TI)) == frozenset(
            {NodeTag.LINUX, NodeTag.GPU}
        )

    def test_a_node_running_the_fleet_test_database_carries_testdb(self) -> None:
        """diphtheria's shape once provisioned (MCPs board task 6bbfd171)."""
        assert node_tags(
            _node(platform=NodePlatform.LINUX, gpu=RTX_3070_TI, test_database=True)
        ) == (frozenset({NodeTag.LINUX, NodeTag.GPU, NodeTag.TESTDB}))

    def test_a_node_declaring_a_rust_toolchain_carries_rust(self) -> None:
        """diphtheria's shape once rustup was installed (MCPs board task 1e2da299)."""
        assert node_tags(_node(platform=NodePlatform.LINUX, test_database=True, rust="1.98.1")) == (
            frozenset({NodeTag.LINUX, NodeTag.TESTDB, NodeTag.RUST})
        )
        assert NodeTag.RUST not in node_tags(_node(platform=NodePlatform.LINUX))

    def test_a_node_declaring_a_cxx_toolchain_carries_cxx(self) -> None:
        """diphtheria's g++ 13.3.0, the one node node-gyp can build on
        (MCPs board task 3f19c136); a Windows node without the VC tools
        declares none and carries no cxx."""
        assert node_tags(_node(platform=NodePlatform.LINUX, rust="1.98.1", cxx="13.3.0")) == (
            frozenset({NodeTag.LINUX, NodeTag.RUST, NodeTag.CXX})
        )
        assert NodeTag.CXX not in node_tags(_node())

    def test_a_node_declaring_the_rootless_exec_daemon_carries_docker(self) -> None:
        """diphtheria once provisioned with execdocker's rootless daemon (MCPs
        board task 6c4516af); a node declaring none carries no docker."""
        assert node_tags(_node(platform=NodePlatform.LINUX, docker="29.8.1")) == (
            frozenset({NodeTag.LINUX, NodeTag.DOCKER})
        )
        assert NodeTag.DOCKER not in node_tags(_node(platform=NodePlatform.LINUX))

    def test_every_platform_carries_the_tag_spelled_as_its_own_word(self) -> None:
        """The platform-to-tag table has a row for every platform, so a third
        platform fails here before a node of it could be tagged."""
        for platform in NodePlatform:
            (tag,) = node_tags(_node(platform=platform))
            assert tag.value == platform.value

    def test_the_vocabulary_is_the_two_platforms_gpu_testdb_rust_cxx_and_docker(self) -> None:
        """The dispatch queue's CHECK (MCPs migrations 532, 563, 569, 570 and
        571) is these seven words, so the members' values are pinned in order."""
        assert [tag.value for tag in NodeTag] == [
            "windows",
            "linux",
            "gpu",
            "testdb",
            "rust",
            "cxx",
            "docker",
        ]


class TestMissingTags:
    def test_a_satisfied_requirement_is_empty(self) -> None:
        assert missing_tags(_node(gpu=RTX_3070_TI), (NodeTag.GPU, NodeTag.WINDOWS)) == ()

    def test_the_missing_tags_come_back_in_the_project_s_order(self) -> None:
        linux = _node(platform=NodePlatform.LINUX)
        assert missing_tags(linux, (NodeTag.WINDOWS, NodeTag.GPU)) == (
            NodeTag.WINDOWS,
            NodeTag.GPU,
        )
        assert missing_tags(linux, (NodeTag.GPU, NodeTag.WINDOWS)) == (
            NodeTag.GPU,
            NodeTag.WINDOWS,
        )

    def test_a_database_suite_is_missing_testdb_on_a_node_without_one(self) -> None:
        with_database = _node(platform=NodePlatform.LINUX, test_database=True)
        assert missing_tags(_node(platform=NodePlatform.LINUX), (NodeTag.TESTDB,)) == (
            NodeTag.TESTDB,
        )
        assert missing_tags(with_database, (NodeTag.TESTDB,)) == ()

    def test_a_crate_build_is_missing_rust_on_a_node_without_cargo(self) -> None:
        with_cargo = _node(platform=NodePlatform.LINUX, rust="1.98.1")
        assert missing_tags(_node(platform=NodePlatform.LINUX), (NodeTag.RUST,)) == (NodeTag.RUST,)
        assert missing_tags(with_cargo, (NodeTag.RUST,)) == ()

    def test_a_project_requiring_nothing_is_never_missing_anything(self) -> None:
        assert missing_tags(_node(platform=NodePlatform.LINUX), ()) == ()


class TestDecodeNodeTag:
    def test_each_word_in_the_set_decodes_to_its_member(self) -> None:
        for tag in NodeTag:
            assert decode_node_tag(tag.value, field="t") is tag

    def test_a_word_outside_the_set_is_refused_with_the_set(self) -> None:
        with pytest.raises(
            JSONTypeError,
            match=r"t must be one of windows, linux, gpu, testdb, rust, cxx, docker, got 'podman'; "
            r".* a Rust or C\+\+ toolchain, or the execution suite's rootless Docker daemon",
        ):
            decode_node_tag("podman", field="t")

    def test_a_non_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="t must be a string, got int"):
            decode_node_tag(7, field="t")


class TestDecodeRequiredTags:
    def test_an_empty_list_is_a_suite_any_node_may_run(self) -> None:
        assert decode_required_tags([], field="p.required_tags") == ()

    def test_tags_keep_their_declared_order(self) -> None:
        assert decode_required_tags(["gpu", "windows"], field="p") == (
            NodeTag.GPU,
            NodeTag.WINDOWS,
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
            JSONTypeError, match=r"p\[1\] must be one of windows, linux, gpu, testdb, rust, cxx"
        ):
            decode_required_tags(["gpu", "cuda"], field="p")

    def test_a_repeated_tag_is_refused_by_index_naming_its_word(self) -> None:
        with pytest.raises(JSONTypeError, match=r"^p\[1\] repeats 'gpu'$"):
            decode_required_tags(["gpu", "gpu"], field="p")

    def test_both_platforms_at_once_is_refused_because_no_node_is_both(self) -> None:
        with pytest.raises(JSONTypeError, match="p names both windows and linux; no node is both"):
            decode_required_tags(["linux", "gpu", "windows"], field="p")


class TestEncodeTags:
    def test_tags_encode_as_their_words_in_order(self) -> None:
        encoded = encode_tags((NodeTag.GPU, NodeTag.WINDOWS))
        assert encoded == ["gpu", "windows"]
        assert [type(word) for word in encoded] == [str, str]
        assert encode_tags(()) == []
