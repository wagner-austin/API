"""Placing manifest-described files on the cluster, proven on both sides.

The sequence is fixed and every step of it is load-bearing:

1. Read each file locally and verify its length and digest against the
   manifest. This catches an emitter that produced the wrong corpus -- the
   733-page-versus-773-page case -- before any bytes cross the network.
2. Send the exact bytes.
3. Digest the file again ON THE CLUSTER and compare. This is the only step
   that proves what the job will read, and it is the reason a local-only
   check is insufficient.

There is no partial success. If any file fails, the operation raises and the
caller learns which file and why; it does not report how many succeeded,
because a corpus staged in part is not a corpus.
"""

from __future__ import annotations

import pathlib

from hpc3.contracts.provenance import format_provenance
from hpc3.contracts.stage import StagedFile, StageManifest
from hpc3.core import audit, digest, remote
from hpc3.core.reproducible import require_sources_reproducible


def stage_one(host: str, source_dir: pathlib.Path, destination: str, staged: StagedFile) -> str:
    """Place one manifest-described file and prove it arrived intact.

    Args:
        host: SSH destination.
        source_dir: Local directory holding the file.
        destination: Absolute directory on the cluster receiving it.
        staged: Manifest record naming the file and the bytes expected.

    Returns:
        The file's absolute path on the cluster.

    Raises:
        AppError: With ``MANIFEST_FILE_MISSING`` if the local file is
            absent, ``DIGEST_MISMATCH`` if the local bytes or the arrived
            bytes differ from the manifest, or ``REMOTE_COMMAND_FAILED`` if
            the write or the remote digest could not run.
    """
    payload = digest.read_and_verify(source_dir, staged)
    remote_path = f"{destination}/{staged['name']}"
    remote.put_bytes(host, remote_path, payload)
    output = remote.remote_digest(host, remote_path)
    arrived = digest.parse_remote_digest(output, staged["name"])
    digest.check_remote_digest(staged["name"], staged["sha256"], arrived)
    return remote_path


#: Suffix of the certification record a staged corpus is admitted by.
#:
#: Read by ``model_trainer.cluster.preflight.check_corpus_certified``, which
#: refuses a corpus whose digest no ``*-digests.txt`` beside it names. That
#: check's docstring has always said "``hpc3-stage`` writes one, and a file put
#: there by hand has none" -- and until 2026-09-04 this package wrote nothing,
#: so the supported staging path produced corpora the supported training path
#: refused, and the only way through was to hand-write the very file the rule
#: exists to distinguish from a hand-placed one. The constant is duplicated
#: rather than imported because hpc3 does not depend on Model-Trainer; the
#: pairing is asserted by a test that reads the consumer's own value.
CERTIFICATION_SUFFIX = "-digests.txt"


def certification_text(manifest: StageManifest) -> str:
    """Render the record that admits these files to a training run.

    One ``<digest>  <name>`` line per staged file, then the provenance. The
    consumer scans for 64-hex tokens anywhere in the text, so the provenance
    is free-form and the digests are what carry meaning.

    Args:
        manifest: The manifest whose files were staged.

    Returns:
        The record's text, newline-terminated.
    """
    lines = [f"{staged['sha256']}  {staged['name']}" for staged in manifest["files"]]
    lines.append(f"provenance {format_provenance(manifest['provenance'])}")
    return "\n".join(lines) + "\n"


def certification_path(manifest: StageManifest, record_name: str) -> str:
    """Where this staging operation's certification record lands.

    Args:
        manifest: The manifest being staged.
        record_name: Stem for the record, normally the manifest's own
            filename stem, so two manifests staging into one destination do
            not overwrite each other's record.

    Returns:
        The record's absolute path on the cluster.
    """
    return f"{manifest['destination']}/{record_name}{CERTIFICATION_SUFFIX}"


def stage_manifest(
    host: str, source_dir: pathlib.Path, manifest: StageManifest, *, record_name: str
) -> list[str]:
    """Place every file a manifest describes, verifying each on both sides.

    The certification record is written LAST, after every file has been
    verified on the cluster. Written first, it would admit a corpus whose
    transfer then failed -- a certification for bytes that are not there.

    It is deliberately NOT in the returned list. The caller reports that list
    as the files it verified on both sides, and the record is neither: it is
    written once and never read back, so counting it would inflate a number
    whose whole meaning is how many files were proven.

    Args:
        host: SSH destination.
        source_dir: Local directory holding the files.
        manifest: What to place and where.
        record_name: Stem for the certification record.

    Returns:
        Absolute cluster paths of the verified files, in manifest order.

    Raises:
        AppError: With ``STAGE_SOURCE_NOT_REPRODUCIBLE`` when a tracked file's
            bytes differ from the repository's, checked BEFORE anything is
            transferred so a refusal leaves the cluster untouched. Or on the
            first file that cannot be verified or transferred: earlier files
            remain on the cluster; they are individually correct, and the
            caller is told which file stopped the run rather than being
            handed a partial success to interpret.
    """
    # First, and before the directory exists: a digest that only this machine
    # can reproduce is worse than no staging at all, because it certifies.
    require_sources_reproducible(source_dir, manifest)
    remote.make_directory(host, manifest["destination"])
    placed = [
        stage_one(host, source_dir, manifest["destination"], staged) for staged in manifest["files"]
    ]
    remote.put_bytes(
        host,
        certification_path(manifest, record_name),
        certification_text(manifest).encode("utf-8"),
    )
    # Logged only after every file verified on the cluster: an event emitted
    # per file would record a partial stage as a sequence of successes.
    audit.files_staged(
        host=host,
        destination=manifest["destination"],
        count=len(placed),
        provenance=format_provenance(manifest["provenance"]),
    )
    return placed


__all__ = [
    "CERTIFICATION_SUFFIX",
    "certification_path",
    "certification_text",
    "stage_manifest",
    "stage_one",
]
