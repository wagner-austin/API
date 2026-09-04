"""The rule that would have caught a five-hour wait, and the parser under it.

THE INCIDENT, because the rule only makes sense next to it. On 2026-09-04 a
ten-minute job pinned `gres/gpu:A100:1` on `free-gpu`. Preflight said
`OK ... would start 2026-09-13` and exited zero. All eight of that partition's
A100s were allocated; it also held 72 A30 and 56 V100 GPUs, several free with
idle cores beside them. The job sat PENDING for five hours. Resubmitted
against a V100, it started in about a hundred seconds.

So the fixture below is not invented: it is the shape `sinfo` actually emits,
including the index lists on the used column.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from hpc3.contracts.gpu_supply import (
    GpuSupply,
    decode_gpu_supply,
    describe_supply,
    encode_gpu_supply,
    free_of,
    parse_gpu_supply,
)
from hpc3.contracts.job import JobSpec
from hpc3.core.gpu_supply import check_requested_gpu_available

#: Real `sinfo -p free-gpu -O "Gres,GresUsed"` output, trimmed to five nodes.
#: Two A100 nodes fully allocated, one A30 node with a spare card, one V100
#: node with two spare, one node reporting no GPU at all.
_SINFO = """\
gpu:A100:2                    gpu:A100:2(IDX:0-1)
gpu:A100:2                    gpu:A100:2(IDX:0-1)
gpu:A30:4                     gpu:A30:3(IDX:0-1,3)
gpu:V100:4                    gpu:V100:2(IDX:0,2)
(null)                        (null)
"""


def _spec(model: str | None, count: int = 1) -> JobSpec:
    """A job spec carrying only what this rule reads.

    Built by hand rather than through `decode_job_spec` because the rule
    touches three fields and a full valid spec would bury them.
    """
    return JobSpec(
        project="mi",
        name="cartridge-gpt2-wiki",
        partition="free-gpu",
        gpu=None if model is None else {"model": model, "count": count},
        cpus=8,
        mem_gb=16,
        minutes=30,
        requeue=True,
        checkpoint_steps=0,
        command="true",
        image=None,
        env_path="/opt/env",
        pinned_packages={},
        deterministic=True,
        depends_on=None,
        experiment={},
        artifact=None,
    )


class TestParseGpuSupply:
    def test_it_sums_nodes_into_a_partition_inventory(self) -> None:
        assert parse_gpu_supply(_SINFO) == (
            {"model": "A100", "total": 4, "used": 4, "free": 0},
            {"model": "A30", "total": 4, "used": 3, "free": 1},
            {"model": "V100", "total": 4, "used": 2, "free": 2},
        )

    def test_index_lists_on_the_used_column_are_not_counted(self) -> None:
        """`gpu:A30:3(IDX:0-1,3)` is three cards, not three plus an index."""
        supply = parse_gpu_supply("gpu:A30:4    gpu:A30:3(IDX:0-1,3)\n")

        assert supply == ({"model": "A30", "total": 4, "used": 3, "free": 1},)

    def test_a_node_with_no_gpu_is_skipped_rather_than_fatal(self) -> None:
        """A CPU-only node in a GPU partition is ordinary.

        Refusing to read the partition because one node has no card would make
        this unusable exactly where it is most needed.
        """
        assert parse_gpu_supply("(null)   (null)\n") == ()

    def test_models_come_back_in_a_stable_order(self) -> None:
        """Two reads of one partition must compare equal."""
        shuffled = "gpu:V100:4  gpu:V100:1\ngpu:A30:4  gpu:A30:1\n"

        assert tuple(entry["model"] for entry in parse_gpu_supply(shuffled)) == ("A30", "V100")

    def test_a_node_with_nothing_allocated_reports_everything_free(self) -> None:
        assert parse_gpu_supply("gpu:A30:4   gpu:A30:0\n") == (
            {"model": "A30", "total": 4, "used": 0, "free": 4},
        )

    def test_empty_output_is_an_empty_inventory(self) -> None:
        assert parse_gpu_supply("") == ()


class TestReadingAnInventory:
    def test_free_of_finds_a_model(self) -> None:
        assert free_of(parse_gpu_supply(_SINFO), "A30") == 1

    def test_free_of_an_absent_model_is_zero(self) -> None:
        assert free_of(parse_gpu_supply(_SINFO), "H100") == 0

    def test_the_summary_names_free_against_total(self) -> None:
        assert describe_supply(parse_gpu_supply(_SINFO)) == (
            "A100 0/4 free, A30 1/4 free, V100 2/4 free"
        )

    def test_an_empty_inventory_describes_as_nothing(self) -> None:
        assert describe_supply(()) == ""


class TestCheckRequestedGpuAvailable:
    def test_the_incident_is_refused(self) -> None:
        """An A100 pin against an exhausted A100 supply, with A30 and V100 free."""
        with pytest.raises(AppError) as excinfo:
            check_requested_gpu_available(_spec("A100"), parse_gpu_supply(_SINFO))

        assert excinfo.value.code is Hpc3ErrorCode.GPU_MODEL_EXHAUSTED
        assert "A30 (1 free)" in excinfo.value.message
        assert "V100 (2 free)" in excinfo.value.message

    def test_a_model_with_free_cards_passes(self) -> None:
        check_requested_gpu_available(_spec("V100"), parse_gpu_supply(_SINFO))

    def test_a_cpu_only_job_is_not_this_rule_s_business(self) -> None:
        check_requested_gpu_available(_spec(None), parse_gpu_supply(_SINFO))

    def test_a_partition_with_no_gpus_at_all_passes(self) -> None:
        """Nothing to compare against, so nothing to advise."""
        check_requested_gpu_available(_spec("A100"), ())

    def test_a_wholly_busy_partition_passes(self) -> None:
        """THE CASE THE RULE MUST NOT FIRE ON.

        When nothing is free the partition is simply busy and waiting is the
        only option -- that is not a mistake, and refusing would leave no way
        to queue work at all on a full cluster. The rule fires only on the
        combination: queueing for the ONE model that is exhausted while others
        idle.
        """
        busy = parse_gpu_supply("gpu:A100:2  gpu:A100:2\ngpu:A30:4  gpu:A30:4\n")

        check_requested_gpu_available(_spec("A100"), busy)

    def test_asking_for_more_cards_than_are_free_is_refused(self) -> None:
        """Two free cards do not satisfy a request for four."""
        with pytest.raises(AppError) as excinfo:
            check_requested_gpu_available(_spec("V100", count=4), parse_gpu_supply(_SINFO))

        assert excinfo.value.code is Hpc3ErrorCode.GPU_MODEL_EXHAUSTED

    def test_a_model_the_partition_does_not_have_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_requested_gpu_available(_spec("H100"), parse_gpu_supply(_SINFO))

        assert excinfo.value.code is Hpc3ErrorCode.GPU_MODEL_EXHAUSTED

    def test_the_message_explains_why_slurm_still_admits_the_job(self) -> None:
        """The reader's next thought is 'but preflight said OK'.

        It did, and it will again: `sbatch --test-only` answers admissibility.
        The message has to close that gap or it reads as a contradiction of
        the tool printing beside it.
        """
        with pytest.raises(AppError) as excinfo:
            check_requested_gpu_available(_spec("A100"), parse_gpu_supply(_SINFO))

        assert "ADMIT" in excinfo.value.message
        assert "backfill" in excinfo.value.message


class TestRoundTrip:
    def test_an_entry_survives_encoding(self) -> None:
        original = GpuSupply(model="A30", total=72, used=63, free=9)

        assert decode_gpu_supply(load_json_str(dump_json_str(encode_gpu_supply(original)))) == (
            original
        )

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_gpu_supply(["A30"])

    def test_numbers_that_do_not_reconcile_are_refused(self) -> None:
        """All three are recorded, so a reader is entitled to use any of them."""
        with pytest.raises(JSONTypeError, match="do not reconcile"):
            decode_gpu_supply({"model": "A30", "total": 72, "used": 63, "free": 40})
