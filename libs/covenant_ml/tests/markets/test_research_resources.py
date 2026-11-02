"""Frozen native allocations bind admission, staged bytes and final recipes."""

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue, require_dict

from covenant_ml.markets.research_cluster import (
    FinalExecution,
    decode_research_cluster,
    encode_research_cluster,
    prepare_bundle,
)
from covenant_ml.markets.research_execution import submit_attempt
from covenant_ml.markets.research_journal import research_policy_digest
from covenant_ml.markets.research_records import (
    decode_research_policy,
    decode_research_proposal,
    encode_research_policy,
)
from covenant_ml.markets.research_trial_records import decode_research_trial, encode_research_trial
from covenant_ml.markets.validation import read_document

from ._research_execution_fixtures import setup
from ._research_fixtures import policy, proposal


@pytest.mark.parametrize("binds", [[], ["/pub"], ["/home/evaluator", "/project/data"]])
@pytest.mark.parametrize("final", [False, True])
def test_recipe_preserves_only_operator_declared_mounts(
    tmp_path: Path, binds: list[str], final: bool
) -> None:
    _path, trial, cluster, directory = setup(tmp_path)
    document = encode_research_cluster(cluster)
    document["binds"] = list(binds)
    configured = decode_research_cluster(document)
    execution = FinalExecution(
        manifest=trial["manifest"],
        manifest_digest=trial["policy"]["dataset_digest"],
        grant_directory="/private/final-grant",
    )
    prepare_bundle(directory, configured, trial, final=execution if final else None)
    assert require_dict(read_document(directory / "run.json"), "image") == {
        "path": cluster["image"],
        "sha256": trial["policy"]["image_digest"],
        "binds": binds,
    }
    assert decode_research_cluster(encode_research_cluster(configured)) == configured


@pytest.mark.parametrize("binds", [["relative"], ["/pub/../escape"]])
def test_mount_paths_cannot_be_relative_or_traversing(tmp_path: Path, binds: list[str]) -> None:
    _path, _trial, cluster, _directory = setup(tmp_path)
    document = encode_research_cluster(cluster)
    document["binds"] = list(binds)
    with pytest.raises(ValueError, match="RESEARCH_CLUSTER_CONFIG"):
        decode_research_cluster(document)


def test_omitted_mount_configuration_is_not_defaulted(tmp_path: Path) -> None:
    _path, _trial, cluster, _directory = setup(tmp_path)
    document = encode_research_cluster(cluster)
    del document["binds"]
    with pytest.raises(ValueError, match="RESEARCH_CLUSTER_CONFIG"):
        decode_research_cluster(document)


@pytest.mark.parametrize("field", ["cpus", "mem_gb"])
@pytest.mark.parametrize("value", [0, -1, 2**31, True, 1.5, "4", None])
def test_invalid_allocation_is_refused(field: str, value: JSONValue) -> None:
    document = encode_research_policy(policy())
    document[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        decode_research_policy(document)


@pytest.mark.parametrize("field", ["cpus", "mem_gb"])
def test_allocation_is_required_without_defaults(field: str) -> None:
    document = encode_research_policy(policy())
    del document[field]
    with pytest.raises(ValueError, match="RESEARCH-002"):
        decode_research_policy(document)


@pytest.mark.parametrize("field", ["cpus", "mem_gb"])
def test_proposer_cannot_supply_allocation(field: str) -> None:
    document = proposal()
    document[field] = 128
    with pytest.raises(ValueError, match="RESEARCH-003"):
        decode_research_proposal(document)


@pytest.mark.parametrize("field", ["cpus", "mem_gb", "max_seconds"])
def test_changed_allocation_cannot_reuse_admission(tmp_path: Path, field: str) -> None:
    path, trial, cluster, directory = setup(tmp_path)
    document = encode_research_trial(trial)
    require_dict(document, "policy")[field] = 128
    changed = decode_research_trial(document)
    assert research_policy_digest(changed["policy"]) != research_policy_digest(trial["policy"])
    before = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="JOURNAL-001"):
        submit_attempt(path, changed, cluster, directory)
    assert sha256(path.read_bytes()).hexdigest() == before
    assert not directory.exists()


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize(("cpus", "memory", "seconds"), [(4, 128, 7200), (2, 16, 61)])
def test_native_recipe_uses_frozen_allocation(
    tmp_path: Path, final: bool, cpus: int, memory: int, seconds: int
) -> None:
    _path, trial, cluster, directory = setup(tmp_path)
    document = encode_research_trial(trial)
    fixed = require_dict(document, "policy")
    fixed.update({"cpus": cpus, "mem_gb": memory, "max_seconds": seconds})
    require_dict(document, "proposal")["seconds"] = seconds
    frozen = decode_research_trial(document)
    execution = FinalExecution(
        manifest=trial["manifest"],
        manifest_digest=trial["policy"]["dataset_digest"],
        grant_directory="/private/final-grant",
    )
    prepare_bundle(directory, cluster, frozen, final=execution if final else None)
    recipe = read_document(directory / "run.json")
    assert (recipe["cpus"], recipe["mem_gb"], recipe["minutes"]) == (
        cpus,
        memory,
        121 if seconds == 7200 else 2,
    )
    assert f"timeout --signal=TERM --kill-after=5s {seconds}s python" in str(recipe["command"])
    assert decode_research_trial(read_document(directory / "request.json")) == frozen
    assert (
        require_dict(recipe, "experiment")["request_sha256"]
        == sha256((directory / "request.json").read_bytes()).hexdigest()
    )
