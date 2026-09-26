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
    NODE_TAGS,
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
    platform: NodePlatform = "windows",
    gpu: NodeGpu | None = None,
    test_database: bool = False,
) -> NodeConfig:
    """Build a node declaration with the three fields tags derive from.

    Args:
        platform: The node's dialect.
        gpu: Its CUDA device, or None for a CPU-only node.
        test_database: Whether it runs the fleet test database.

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
        assert node_tags(_node()) == frozenset({"windows"})

    def test_a_linux_node_with_a_card_carries_both(self) -> None:
        assert node_tags(_node(platform="linux", gpu=RTX_3070_TI)) == frozenset({"linux", "gpu"})

    def test_a_node_running_the_fleet_test_database_carries_testdb(self) -> None:
        """diphtheria's shape once provisioned (MCPs board task 6bbfd171)."""
        assert node_tags(_node(platform="linux", gpu=RTX_3070_TI, test_database=True)) == (
            frozenset({"linux", "gpu", "testdb"})
        )

    def test_the_vocabulary_is_the_two_platforms_gpu_and_testdb(self) -> None:
        assert NODE_TAGS == ("windows", "linux", "gpu", "testdb")


class TestMissingTags:
    def test_a_satisfied_requirement_is_empty(self) -> None:
        assert missing_tags(_node(gpu=RTX_3070_TI), ("gpu", "windows")) == ()

    def test_the_missing_tags_come_back_in_the_project_s_order(self) -> None:
        assert missing_tags(_node(platform="linux"), ("windows", "gpu")) == ("windows", "gpu")
        assert missing_tags(_node(platform="linux"), ("gpu", "windows")) == ("gpu", "windows")

    def test_a_database_suite_is_missing_testdb_on_a_node_without_one(self) -> None:
        assert missing_tags(_node(platform="linux"), ("testdb",)) == ("testdb",)
        assert missing_tags(_node(platform="linux", test_database=True), ("testdb",)) == ()

    def test_a_project_requiring_nothing_is_never_missing_anything(self) -> None:
        assert missing_tags(_node(platform="linux"), ()) == ()


class TestDecodeNodeTag:
    def test_each_word_in_the_set_decodes_to_itself(self) -> None:
        for tag in NODE_TAGS:
            assert decode_node_tag(tag, field="t") == tag

    def test_a_word_outside_the_set_is_refused_with_the_set(self) -> None:
        with pytest.raises(
            JSONTypeError, match="t must be one of windows, linux, gpu, testdb, got 'docker'"
        ):
            decode_node_tag("docker", field="t")

    def test_a_non_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="t must be a string, got int"):
            decode_node_tag(7, field="t")


class TestDecodeRequiredTags:
    def test_an_empty_list_is_a_suite_any_node_may_run(self) -> None:
        assert decode_required_tags([], field="p.required_tags") == ()

    def test_tags_keep_their_declared_order(self) -> None:
        assert decode_required_tags(["gpu", "windows"], field="p") == ("gpu", "windows")

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
            JSONTypeError, match=r"p\[1\] must be one of windows, linux, gpu, testdb"
        ):
            decode_required_tags(["gpu", "cuda"], field="p")

    def test_a_repeated_tag_is_refused_by_index(self) -> None:
        with pytest.raises(JSONTypeError, match=r"p\[1\] repeats 'gpu'"):
            decode_required_tags(["gpu", "gpu"], field="p")

    def test_both_platforms_at_once_is_refused_because_no_node_is_both(self) -> None:
        with pytest.raises(JSONTypeError, match="p names both windows and linux; no node is both"):
            decode_required_tags(["linux", "gpu", "windows"], field="p")


class TestEncodeTags:
    def test_tags_encode_as_a_list_in_order(self) -> None:
        assert encode_tags(("gpu", "windows")) == ["gpu", "windows"]
        assert encode_tags(()) == []
