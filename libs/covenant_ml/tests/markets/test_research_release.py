"""Real private grants are request-bound and consumed once across processes."""

import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from covenant_ml.markets import _test_hooks_encryption as hooks
from covenant_ml.markets.research_release import (
    FinalGrant,
    consume_final_grant,
    decode_final_grant,
    encode_final_grant,
    write_final_grant,
)
from covenant_ml.markets.research_trial_records import (
    ResearchTrial,
    decode_research_trial,
    encode_research_trial,
)

from ._trial_fixtures import trial_document


def _grant(root: Path) -> tuple[Path, FinalGrant, ResearchTrial, Path]:
    """Write a real private key grant with exact request and ciphertext bindings."""
    trial = decode_research_trial(trial_document(root))
    manifest = Path(trial["manifest"])
    identity = hooks.x25519.Identity.generate()
    grant = decode_final_grant(
        {
            "request_digest": sha256(
                dump_json_str(encode_research_trial(trial)).encode()
            ).hexdigest(),
            "manifest_digest": sha256(manifest.read_bytes()).hexdigest(),
            "recipient": str(identity.to_public()),
            "max_plaintext_bytes": 1000000,
        }
    )
    directory = root / "grant"
    write_final_grant(directory, grant, identity)
    return directory, grant, trial, manifest


def test_grant_is_private_and_consumed_before_the_identity_is_returned(tmp_path: Path) -> None:
    directory, grant, trial, manifest = _grant(tmp_path)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "final.agekey").stat().st_mode) == 0o600
    assert stat.S_IMODE((directory / "release.json").stat().st_mode) == 0o600
    observed, identity = consume_final_grant(directory, trial, manifest)
    assert observed == grant
    assert str(identity.to_public()) == grant["recipient"]
    assert (directory / "consumed").read_bytes() == b"released\n"
    assert stat.S_IMODE((directory / "consumed").stat().st_mode) == 0o600
    assert not (directory / "final.agekey").exists()
    with pytest.raises(FileExistsError):
        consume_final_grant(directory, trial, manifest)


@pytest.mark.parametrize("case", ["request", "manifest"])
def test_changed_binding_cannot_consume_the_grant(tmp_path: Path, case: str) -> None:
    directory, _, trial, manifest = _grant(tmp_path)
    if case == "request":
        document = encode_research_trial(trial)
        document["number"] = 2
        trial = decode_research_trial(document)
    else:
        manifest.write_text("changed")
    with pytest.raises(ValueError, match="RESEARCH_FINAL_GRANT"):
        consume_final_grant(directory, trial, manifest)
    assert not (directory / "consumed").exists()
    assert (directory / "final.agekey").is_file()


def test_missing_key_consumes_grant_without_restoring_permission(tmp_path: Path) -> None:
    directory, _, trial, manifest = _grant(tmp_path)
    (directory / "final.agekey").unlink()
    with pytest.raises(FileNotFoundError):
        consume_final_grant(directory, trial, manifest)
    assert (directory / "consumed").read_bytes() == b"released\n"
    with pytest.raises(FileExistsError):
        consume_final_grant(directory, trial, manifest)


def test_symlinked_key_is_refused_after_consumption(tmp_path: Path) -> None:
    directory, _, trial, manifest = _grant(tmp_path)
    key = directory / "final.agekey"
    retained = directory / "retained.agekey"
    key.rename(retained)
    key.symlink_to(retained)
    with pytest.raises(OSError):
        consume_final_grant(directory, trial, manifest)
    assert (directory / "consumed").read_bytes() == b"released\n"
    assert retained.is_file()


def test_an_existing_delivery_cannot_be_overwritten(tmp_path: Path) -> None:
    directory, grant, _, _ = _grant(tmp_path)
    key = directory / "final.agekey"
    original = key.read_bytes()
    identity = hooks.x25519.Identity.from_str(original.decode().strip())
    with pytest.raises(FileExistsError):
        write_final_grant(directory, grant, identity)
    assert key.read_bytes() == original
    with pytest.raises(ValueError, match="RESEARCH_FINAL_GRANT"):
        write_final_grant(tmp_path / "wrong", grant, hooks.x25519.Identity.generate())
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("case", ["extra", "budget"])
def test_grant_metadata_refuses_unknown_fields_and_unbounded_storage(
    tmp_path: Path, case: str
) -> None:
    _, grant, _, _ = _grant(tmp_path)
    document: JSONObject = encode_final_grant(grant)
    document["extra" if case == "extra" else "max_plaintext_bytes"] = 0
    with pytest.raises(ValueError, match="RESEARCH_FINAL_GRANT"):
        decode_final_grant(document)


def test_two_real_processes_cannot_redeem_the_same_grant(tmp_path: Path) -> None:
    directory, _, trial, manifest = _grant(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(dump_json_str(encode_research_trial(trial)))
    program = """
import sys
from pathlib import Path
from covenant_ml.markets.research_release import consume_final_grant
from covenant_ml.markets.research_trial_records import decode_research_trial
from covenant_ml.markets.validation import read_document
directory, request, manifest = map(Path, sys.argv[1:])
consume_final_grant(directory, decode_research_trial(read_document(request)), manifest)
sys.stdout.write('consumed\\n')
"""
    command = [sys.executable, "-c", program, str(directory), str(request), str(manifest)]
    with (
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as first,
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as second,
    ):
        outcomes = [first.communicate(timeout=30), second.communicate(timeout=30)]
        assert sorted([first.returncode, second.returncode]) == [0, 1]
    assert sorted(stdout for stdout, _ in outcomes) == [b"", b"consumed\n"]
    failures = [stderr.decode().splitlines()[-1] for _, stderr in outcomes if stderr]
    assert failures == [f"FileExistsError: [Errno 17] File exists: {str(directory / 'consumed')!r}"]
    assert not (directory / "final.agekey").exists()
    assert (directory / "consumed").read_bytes() == b"released\n"
