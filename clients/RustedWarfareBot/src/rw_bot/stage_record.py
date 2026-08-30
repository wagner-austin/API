"""Turning a packed archive into the manifest that stages it.

WHY THIS IS CODE AND NOT A FILE SOMEBODY KEEPS UP TO DATE. The two stage
manifests under ``provenance/`` were written by hand, and the README beside
them said they were "composed from the document so no digest is ever
retyped", which was the intention and not the truth. The first re-freeze
proved the difference: the payload gained two files, its archive changed
size and digest, and the manifest went on naming the previous ones. Staging
against it would have failed the local digest check -- the good case -- or,
had the digest still matched a stale archive on disk, staged the wrong tree
and verified it happily on both sides.

A manifest is three facts about an archive that was just written. Composing
it where the archive is written costs nothing and removes the copy.
"""

from __future__ import annotations

from collections.abc import Mapping

from hpc3.contracts.stage import StagedFile, StageManifest

from rw_bot.tree_archive import ArchiveResult


def stage_manifest(
    destination: str, name: str, archive: ArchiveResult, provenance: Mapping[str, str]
) -> StageManifest:
    """Return the manifest that places one packed archive on the cluster.

    Args:
        destination: Absolute cluster directory receiving the archive.
        name: The archive's bare filename, as the manifest names it.
        archive: What packing produced -- its digest and its length.
        provenance: Where these bytes came from. Free-form, required and
            never empty: the digests prove the bytes on the cluster are the
            bytes named here, and only this says what "here" is.

    Returns:
        The manifest, ready to encode beside the document it describes.
    """
    staged = StagedFile(name=name, sha256=archive["sha256"], size_bytes=archive["size_bytes"])
    return StageManifest(destination=destination, files=[staged], provenance=dict(provenance))


__all__ = ["stage_manifest"]
