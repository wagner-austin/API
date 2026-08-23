"""Tests for the cluster contract and the measured-cluster registry.

Two jobs here, and the second is the one that matters.

The first is ordinary: HPC3's numbers are what the machine reported, and the
registry refuses a slug nobody has measured.

The second is a claim about the *shape* of this package -- that the rules are
asked of a cluster rather than of a constant. A synthetic cluster is
constructed here with different partition names, different GPUs, a fractional
usage factor and much lower ceilings, and the same production code paths are
driven against it. If any rule had HPC3's values baked in, these fail. The
synthetic cluster is deliberately NOT registered: it is a test fixture, not a
claim that a second real machine has been measured.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.clusters import CLUSTERS, require_cluster
from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.cluster import (
    ClusterFacts,
    PartitionFacts,
    partition_bills,
    partition_facts,
    partition_names,
)
from hpc3.contracts.job import decode_job_spec
from hpc3.contracts.status import decode_job_status, service_units
from hpc3.contracts.sweep import decode_sweep_spec
from tests.conftest import gpus

OTHER: ClusterFacts = ClusterFacts(
    slug="other",
    description="a fictional cluster that exists only in this file",
    gpus=("H100", "MI300X"),
    partitions={
        "batch": PartitionFacts(
            usage_factor=0.5,
            preemptible=False,
            max_hours=8,
            gpus=("H100",),
            max_gpus_per_user=2,
            max_cpus_per_user=None,
            max_jobs_per_user=2,
        ),
        "scavenge": PartitionFacts(
            usage_factor=0.0,
            preemptible=True,
            max_hours=4,
            gpus=("MI300X",),
            max_gpus_per_user=1,
            max_cpus_per_user=None,
            max_jobs_per_user=1,
        ),
        "serial": PartitionFacts(
            usage_factor=0.25,
            preemptible=False,
            max_hours=2,
            gpus=(),
            max_gpus_per_user=None,
            max_cpus_per_user=12,
            max_jobs_per_user=3,
        ),
    },
)
"""Nothing about this overlaps HPC3: not a partition name, not a GPU, not a
ceiling, and its billing partition charges half rate rather than full."""


def _spec(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a job document for the synthetic cluster.

    Args:
        **overrides: Fields to replace.

    Returns:
        The document.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "run",
        "partition": "batch",
        "gpu": gpus("H100"),
        "cpus": 4,
        "mem_gb": 16,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "accept_billing": True,
        "env_path": "/scratch/env",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B"},
        "command": "python train.py",
    }
    base.update(overrides)
    return base


class TestHpc3Facts:
    def test_the_free_partition_carries_a_zero_usage_factor(self) -> None:
        """Measured with sacctmgr: this is why a free-gpu job costs nothing."""
        assert partition_facts(HPC3, "free-gpu")["usage_factor"] == 0.0

    def test_the_partition_that_reads_free_actually_bills(self) -> None:
        """free-gpu32 carries UsageFactor 1.0 despite the name."""
        assert partition_bills(HPC3, "free-gpu32") is True
        assert partition_bills(HPC3, "free-gpu") is False

    def test_every_partition_is_listed(self) -> None:
        assert partition_names(HPC3) == (
            "free",
            "free-gpu",
            "free-gpu32",
            "gpu",
            "gpu32",
            "standard",
        )

    def test_an_absent_partition_is_refused_by_name(self) -> None:
        with pytest.raises(AppError) as excinfo:
            partition_facts(HPC3, "turbo")
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN
        assert "'free-gpu'" in excinfo.value.message


class TestRegistry:
    def test_hpc3_resolves_by_slug(self) -> None:
        assert require_cluster("hpc3") == HPC3

    def test_an_unmeasured_cluster_is_refused(self) -> None:
        """Guessing a default would submit using another machine's ceilings."""
        with pytest.raises(AppError) as excinfo:
            require_cluster("frontier")
        assert excinfo.value.code is Hpc3ErrorCode.CLUSTER_UNKNOWN

    def test_the_message_lists_what_has_been_measured(self) -> None:
        with pytest.raises(AppError) as excinfo:
            require_cluster("frontier")
        assert "'hpc3'" in excinfo.value.message

    def test_the_synthetic_cluster_is_not_registered(self) -> None:
        """A fixture must never be reachable as if it had been measured."""
        assert "other" not in CLUSTERS


class TestRegistryShape:
    """Checked for every registered cluster, so a new module cannot be added
    in a shape the rest of the package cannot read. Only measurement can check
    the values; this checks that they are self-consistent.
    """

    def test_every_key_matches_its_own_slug(self) -> None:
        assert all(slug == facts["slug"] for slug, facts in CLUSTERS.items())

    def test_every_cluster_describes_itself(self) -> None:
        assert all(facts["description"] != "" for facts in CLUSTERS.values())

    def test_the_registry_lists_exactly_what_has_been_measured(self) -> None:
        """A cluster appears here only when someone read its numbers off it."""
        assert sorted(CLUSTERS) == ["hpc3"]

    def test_every_cluster_declares_partitions_and_gpus(self) -> None:
        for facts in CLUSTERS.values():
            assert partition_names(facts) != ()
            assert facts["gpus"] != ()

    def test_no_partition_claims_a_gpu_the_cluster_does_not_have(self) -> None:
        """Otherwise a spec could name a GPU that passes one check and not the other."""
        for facts in CLUSTERS.values():
            for name in partition_names(facts):
                assert set(partition_facts(facts, name)["gpus"]) <= set(facts["gpus"])

    def test_every_gpu_a_cluster_lists_is_reachable_on_some_partition(self) -> None:
        """A GPU no partition carries can never be requested; listing it lies."""
        for facts in CLUSTERS.values():
            reachable: set[str] = set()
            for name in partition_names(facts):
                reachable |= set(partition_facts(facts, name)["gpus"])
            assert reachable == set(facts["gpus"])

    def test_every_partition_declares_the_ceiling_its_own_work_pends_against(self) -> None:
        """This replaced "every partition carries a GPU", which stopped being
        true the moment CPU partitions were measured -- and which was really
        protecting this: a partition must bound the resource its jobs compete
        for, or a sweep on it has nothing that can catch an overcommit.

        Which resource that is follows from the partition rather than being
        declared twice: GPU work pends against ``gres/gpu``, CPU work against
        ``cpu``. Asserting the RELATION rather than either literal is what
        keeps this from expiring again the next time a machine is measured.
        """
        for facts in CLUSTERS.values():
            for name in partition_names(facts):
                measured = partition_facts(facts, name)
                bound = (
                    measured["max_gpus_per_user"]
                    if measured["gpus"] != ()
                    else measured["max_cpus_per_user"]
                )
                if bound is None:
                    raise AssertionError(f"{facts['slug']}.{name} bounds neither GPUs nor cores")
                assert bound >= 1

    def test_every_ceiling_that_is_declared_admits_at_least_one_job(self) -> None:
        """A zero ceiling would refuse every sweep on that partition.

        An undeclared ceiling is skipped rather than defaulted: the QOS says
        nothing about that resource, and a number invented here would be
        enforced against jobs the cluster would have run.
        """
        for facts in CLUSTERS.values():
            for name in partition_names(facts):
                measured = partition_facts(facts, name)
                for ceiling in (measured["max_gpus_per_user"], measured["max_cpus_per_user"]):
                    if ceiling is not None:
                        assert ceiling >= 1
                assert measured["max_jobs_per_user"] >= 1
                assert measured["max_hours"] >= 1

    def test_no_usage_factor_is_negative(self) -> None:
        for facts in CLUSTERS.values():
            for name in partition_names(facts):
                assert partition_facts(facts, name)["usage_factor"] >= 0.0


class TestTheRulesFollowTheCluster:
    """The claim that this package is not HPC3-shaped, driven against a
    cluster whose every value differs.
    """

    def test_a_spec_valid_on_the_other_cluster_decodes(self) -> None:
        assert decode_job_spec(_spec(), OTHER)["partition"] == "batch"

    def test_an_hpc3_partition_is_unknown_on_the_other_cluster(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(partition="free-gpu", gpu=gpus("A100")), OTHER)
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN

    def test_an_hpc3_gpu_is_unknown_on_the_other_cluster(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(gpu=gpus("A100")), OTHER)
        assert excinfo.value.code is Hpc3ErrorCode.GPU_TYPE_UNPINNED

    def test_the_other_clusters_shorter_walltime_ceiling_binds(self) -> None:
        """8 hours there; the same 10-hour job fits HPC3's 72."""
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(minutes=600), OTHER)
        assert excinfo.value.code is Hpc3ErrorCode.TIME_LIMIT_EXCEEDS_PARTITION

    def test_its_preemptible_partition_still_demands_protection(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(
                _spec(partition="scavenge", gpu=gpus("MI300X"), minutes=200, accept_billing=False),
                OTHER,
            )
        assert excinfo.value.code is Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED

    def test_its_billing_partition_still_demands_consent(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(accept_billing=False), OTHER)
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_BILLS_WITHOUT_CONSENT

    def test_its_much_lower_sweep_ceiling_binds(self) -> None:
        """Three members fit HPC3's 24 GPUs and not this cluster's 2."""
        members: list[JSONValue] = [{"suffix": f"s{i}", "command": "python t.py"} for i in range(3)]
        with pytest.raises(AppError) as excinfo:
            decode_sweep_spec({"base": _spec(), "members": members}, OTHER)
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING

    def test_a_fractional_usage_factor_is_applied_rather_than_rounded(self) -> None:
        """The reason the factor is stored as a number and not a bills flag."""
        row: dict[str, JSONValue] = {
            "job_id": "1",
            "name": "run",
            "partition": "batch",
            "state": "COMPLETED",
            "elapsed_seconds": 3600,
            "billing_tres": 4,
            "gpu_count": 1,
            "cpu_count": 4,
            "node_list": "n1",
        }
        assert service_units(decode_job_status(row, OTHER), OTHER) == 2.0

    def test_its_zero_factor_partition_costs_nothing(self) -> None:
        row: dict[str, JSONValue] = {
            "job_id": "1",
            "name": "run",
            "partition": "scavenge",
            "state": "COMPLETED",
            "elapsed_seconds": 36000,
            "billing_tres": 64,
            "gpu_count": 1,
            "cpu_count": 4,
            "node_list": "n1",
        }
        assert service_units(decode_job_status(row, OTHER), OTHER) == 0.0
