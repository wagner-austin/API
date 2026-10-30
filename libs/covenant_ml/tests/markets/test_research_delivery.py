"""Journal authority and private grant consumption precede every delivery effect."""

import sys
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from platform_core.stage_manifest import decode_stage_manifest
from scripts.research_proposals import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.campaign_process import run_command
from covenant_ml.markets.research_cluster import (
    ResearchCluster,
    decode_research_cluster,
    encode_research_cluster,
)
from covenant_ml.markets.research_delivery import deliver_final_grant
from covenant_ml.markets.research_final import release_final, select_candidate
from covenant_ml.markets.research_journal import research_journal
from covenant_ml.markets.research_records import encode_research_policy
from covenant_ml.markets.research_trial_records import ResearchTrial, encode_research_trial
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._encrypted_fixtures import evaluator_keys
from ._research_execution_fixtures import setup
from ._research_final_fixtures import completed_study


@pytest.fixture(autouse=True)
def restore_runner() -> Generator[None, None, None]:
    previous = _test_hooks.campaign_runner
    yield
    _test_hooks.campaign_runner = previous


def _released(root: Path) -> tuple[Path, ResearchTrial, Path, Path, ResearchCluster]:
    ledger, trial, final = completed_study(root)
    select_candidate(ledger, trial, final)
    grant = root / "grant"
    release_final(ledger, trial, final, grant, keys=evaluator_keys(root / "validation"))
    (root / "config").mkdir()
    _, _, cluster, _ = setup(root / "config")
    return ledger, trial, final, grant, cluster


class Delivery:
    """Inspect stdin-only transfer; real subprocess input and failure tested below."""

    def __init__(self, grant: Path, cluster: ResearchCluster, code: int = 0) -> None:
        self.grant, self.cluster, self.code = grant, cluster, code
        self.calls = 0

    def __call__(
        self,
        command: tuple[str, ...],
        directory: Path,
        log: Path,
        *,
        stdin_bytes: bytes | None = None,
    ) -> int:
        self.calls += 1
        assert directory == self.grant
        assert (directory / "consumed").read_bytes() == b"released\n"
        assert not (directory / "final.agekey").exists()
        assert command == (
            self.cluster["python"],
            "-m",
            "hpc3.cli.private_stage",
            "--config",
            self.cluster["workspace"],
            "--manifest",
            str(directory / "delivery.json"),
        )
        manifest = decode_stage_manifest(read_document(directory / "delivery.json"))
        assert stdin_bytes is not None
        offset = 0
        for file in manifest["files"]:
            data = stdin_bytes[offset : offset + file["size_bytes"]]
            assert sha256(data).hexdigest() == file["sha256"]
            offset += file["size_bytes"]
        assert offset == len(stdin_bytes)
        assert b"AGE-SECRET-KEY-" in stdin_bytes
        assert "AGE-SECRET-KEY-" not in " ".join(command)
        log.write_text("private delivery verified\n" if self.code == 0 else "transfer failed\n")
        return self.code


def test_delivery_retains_authority_and_cannot_be_replayed(tmp_path: Path) -> None:
    ledger, trial, final, grant, cluster = _released(tmp_path)
    before = research_journal(trial["policy"]).read(ledger)
    delivery = Delivery(grant, cluster)
    _test_hooks.campaign_runner = delivery
    assert deliver_final_grant(ledger, trial, cluster, grant, final, "/private/final") == before[0]
    assert research_journal(trial["policy"]).read(ledger) == before
    with pytest.raises(FileExistsError):
        deliver_final_grant(ledger, trial, cluster, grant, final, "/private/retry")
    assert delivery.calls == 1


def test_failed_delivery_leaves_spent_local_authority(tmp_path: Path) -> None:
    ledger, trial, final, grant, cluster = _released(tmp_path)
    delivery = Delivery(grant, cluster, 7)
    _test_hooks.campaign_runner = delivery
    with pytest.raises(RuntimeError, match="RESEARCH_FINAL_DELIVERY"):
        deliver_final_grant(ledger, trial, cluster, grant, final, "/private/final")
    with pytest.raises(FileExistsError):
        deliver_final_grant(ledger, trial, cluster, grant, final, "/private/retry")
    assert delivery.calls == 1


@pytest.mark.parametrize("case", ["unreleased", "different"])
def test_journal_refusals_do_not_read_or_consume_key(tmp_path: Path, case: str) -> None:
    ledger, trial, final, grant, cluster = _released(tmp_path)
    if case == "unreleased":
        (tmp_path / "other").mkdir()
        ledger, trial, final = completed_study(tmp_path / "other")
    else:
        final.write_bytes(b"different")
    with pytest.raises(ValueError, match="RESEARCH_FINAL_DELIVERY"):
        deliver_final_grant(ledger, trial, cluster, grant, final, "/private/final")
    assert not (grant / "consumed").exists()
    assert (grant / "final.agekey").is_file()


def test_real_process_delivery_failure_leaves_key_consumed(tmp_path: Path) -> None:
    ledger, trial, final, grant, cluster = _released(tmp_path)
    document = encode_research_cluster(cluster)
    # A missing controller executable fails at the real process launch boundary.
    document["python"] = str(tmp_path / "missing-python")
    cluster = decode_research_cluster(document)
    _test_hooks.campaign_runner = run_command
    with pytest.raises(FileNotFoundError):
        deliver_final_grant(ledger, trial, cluster, grant, final, "/private/final")
    assert (grant / "consumed").read_bytes() == b"released\n"
    assert not (grant / "final.agekey").exists()


def test_real_controller_input_is_private_and_child_nonzero_is_preserved(tmp_path: Path) -> None:
    payload = b"synthetic-key\x00\xff"
    program = (
        "import sys; from hashlib import sha256; "
        "print(sha256(sys.stdin.buffer.read()).hexdigest()); sys.exit(7)"
    )
    log = tmp_path / "child.log"
    assert run_command((sys.executable, "-c", program), tmp_path, log, stdin_bytes=payload) == 7
    assert log.read_text() == sha256(payload).hexdigest() + "\n"


def test_cli_delivers_only_the_journals_released_request(tmp_path: Path) -> None:
    ledger, trial, final, grant, cluster = _released(tmp_path)
    for name, document in (
        ("policy", encode_research_policy(trial["policy"])),
        ("trial", encode_research_trial(trial)),
        ("cluster", encode_research_cluster(cluster)),
    ):
        (tmp_path / f"{name}.json").write_text(dump_json_str(document))
    delivery = Delivery(grant, cluster)
    _test_hooks.campaign_runner = delivery
    assert (
        main(
            [
                "--policy",
                str(tmp_path / "policy.json"),
                "--ledger",
                str(ledger),
                "deliver",
                "--trial",
                str(tmp_path / "trial.json"),
                "--cluster",
                str(tmp_path / "cluster.json"),
                "--directory",
                str(grant),
                "--final-manifest",
                str(final),
                "--destination",
                "/private/final",
            ],
            stand_in_pin,
        )
        == 0
    )
    assert delivery.calls == 1
