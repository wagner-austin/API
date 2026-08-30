"""What the staged game tree IS, as a document rather than as a directory.

A directory pinned by convention cannot say what it held when a run used it.
``hpc3``'s image contract names that as the concrete failure it was written
against, and the tree this project stages was on course to repeat it: 311 MB
assembled by hand, once, described by nothing.

Staging already proves the bytes on the cluster are the bytes named in the
manifest. What it cannot prove is that the manifest names the right bytes, and
that is what this document is for.

THREE ORIGINS, BECAUSE THE TREE HAS THREE KINDS OF FILE. Each entry is
classified against a REFERENCE tree -- Steam's own installed copy of the same
build -- and the classification is a measurement, not a declaration:

The tally below is what the real tree measured on 2026-08-29: 1,445 + 59 + 12
+ 257 + 1 = 1,774 files.

* :data:`ORIGIN_VERIFIED` -- the same path in the reference, byte for byte.
  Content depot 647961 carries no ``oslist``, so its files are identical on
  either platform, and a match here means an independent copy that Steam
  itself placed agrees. 1,445 files.
* :data:`ORIGIN_MIRRORED` -- the same FILENAME elsewhere in the reference,
  byte for byte. 59 files, essentially all vendor data the two bundled JVMs
  both ship: ``jvm-linux/COPYRIGHT`` against ``jvm/COPYRIGHT``, fonts,
  ``cacerts``. Verified just as firmly as the above; the path differs because
  the platform's runtime directory does.
* :data:`ORIGIN_RENAMED` -- a DIFFERENT filename holding the same bytes, which
  means somebody made a copy. 12 files, and the seven under ``assets/`` are
  why this module exists. The launcher passes a map on a command line, so
  this project keeps shell-safe copies -- ``[p2]duel_lake.tmx`` is
  ``[p2]Lake (2p).tmx``. EVERY sweep in this project plays that copy, and it
  is not in the Steam install under that name: rebuilding the tree's content
  half "cleanly" from Steam would have deleted the map every batch runs on
  and sent the engine to its boot sandbox, which is the failure this project
  has already paid for once.

  The other five are the vendor's own coincidences inside the JRE -- three
  identical cursor GIFs, a ``README`` that gained a ``.txt``, two identical
  Chinese locale files. The rule is mechanical and says so; it does not try to
  guess intent, and a reader after the project's own copies filters on
  ``assets/``.

  Split from the mirrored case because lumping the two together buried twelve
  among seventy-one.
* :data:`ORIGIN_ASSEMBLED` -- no counterpart anywhere in the reference. 257
  files: the bundled JRE's Linux-only half and the native ``.so`` libraries.
  There is no second local copy of these, so the document states them as
  assembled from depot 647963 rather than pretending they were checked.

* :data:`ORIGIN_PINNED_STATE` -- a file the game REWRITES on every boot, so
  the two copies are expected to differ and a comparison would say nothing.
  ``preferences.ini`` is the case, and it is staged rather than dropped: the
  runner resets every clone by copying it out of the source directory, so a
  tree without it fails that reset on the compute node. What is staged is this
  workstation's copy, which makes it the experiment's pinned starting state
  rather than an accident of whichever node ran first.

A FIFTH CASE IS REFUSED. Same path, different bytes, and NOT a file the game
rewrites, means the tree and the reference genuinely disagree -- and neither
the digest nor the transfer would notice, because each matches the copy it
came from. The rewritten paths are named by the caller from
:data:`rw_bot.harness.clone.VOLATILE_FILES`, so the exception is a declared
rule with one owner rather than an allowance made here.

WHAT THE DOCUMENT DOES NOT CLAIM. It does not say the assembled files are
Steam's. Nothing on this machine can say that any more -- the depot download
was consumed -- and asserting it would be the kind of sentence this index was
built to stop. It says which depot they came from, when they were assembled,
and that everything else was checked.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from rw_bot import RwBotError
from rw_bot.tree_identity import (
    TreeEntry,
    decode_tree_entry,
    digest_record,
    encode_tree_entry,
    tree_digest,
)

#: The Steam application, and the depots this tree is drawn from.
#:
#: The content depot declares no ``oslist``, which is what makes it the same
#: bytes on either platform and therefore what makes the reference comparison
#: below meaningful at all.
APP_ID = "647960"
CONTENT_DEPOT = "647961"
LINUX_DEPOT = "647963"

#: What a depot contributes, as the document records it.
ROLE_CONTENT = "content"
ROLE_LINUX_RUNTIME = "linux-runtime"

#: How an entry got into the tree.
ORIGIN_VERIFIED = "verified-content"
ORIGIN_MIRRORED = "mirrored-content"
ORIGIN_RENAMED = "renamed-copy"
ORIGIN_ASSEMBLED = "assembled-runtime"
ORIGIN_PINNED_STATE = "pinned-state"

ORIGINS = (
    ORIGIN_VERIFIED,
    ORIGIN_MIRRORED,
    ORIGIN_RENAMED,
    ORIGIN_ASSEMBLED,
    ORIGIN_PINNED_STATE,
)

#: A file with no bytes has no content to be identified by, and every empty
#: file shares one digest. Attribution by digest therefore says nothing about
#: an empty file and must not be attempted: the first run of this classified
#: ``jvm-linux/lib/security/trusted.libraries`` as a copy of a map's demo
#: marker, which is a confident wrong answer of exactly the shape a provenance
#: document exists to prevent.
_EMPTY = 0

#: What a rename's source field holds when there is no source: an entry with
#: no counterpart states that, rather than naming itself and reading as though
#: it had been checked against something.
NO_SOURCE = ""

_DIVERGED = "RW-TREE-101"
_UNKNOWN_ORIGIN = "RW-TREE-102"
_NO_ENTRIES = "RW-TREE-103"


class StagedTreeError(RwBotError):
    """A tree could not be described against its reference.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what disagreed.
    """


class DepotRef(TypedDict):
    """One Steam depot at one pinned manifest.

    Attributes:
        depot_id: The depot's numeric id.
        manifest: The manifest GID, which is what actually pins the contents.
            A depot id alone names a moving target.
        role: What it contributes, one of :data:`ROLE_CONTENT` or
            :data:`ROLE_LINUX_RUNTIME`.
    """

    depot_id: str
    manifest: str
    role: str


class ClassifiedEntry(TreeEntry):
    """One file of the tree, with how it came to be there.

    Attributes:
        origin: One of :data:`ORIGINS`.
        source: The reference path this entry is a copy of, for
            :data:`ORIGIN_RENAMED`; its own path for :data:`ORIGIN_VERIFIED`;
            :data:`NO_SOURCE` for :data:`ORIGIN_ASSEMBLED`.
    """

    origin: str
    source: str


class StagedTreeSpec(TypedDict):
    """Everything a reader needs to know what was staged and where it is from.

    Attributes:
        app_id: The Steam application.
        build_id: The build the reference install was at when the tree was
            described, from ``appmanifest_<app>.acf``.
        depots: The depots the tree draws on, each pinned by manifest.
        reference: Where the verification was done against, as a human-
            readable location. Recorded because "verified" is only meaningful
            alongside "against what".
        entries: Every file, classified, in listing order.
        tree_sha256: The tree's identity, from
            :func:`~rw_bot.tree_identity.tree_digest`.
        archive_name: The single file the tree is staged as.
        archive_sha256: That file's digest, which the stage manifest names and
            the cluster recomputes on arrival.
        archive_size_bytes: Its length.
    """

    app_id: str
    build_id: str
    depots: list[DepotRef]
    reference: str
    entries: list[ClassifiedEntry]
    tree_sha256: str
    archive_name: str
    archive_sha256: str
    archive_size_bytes: int


def classify_entries(
    tree: Sequence[TreeEntry], reference: Sequence[TreeEntry], *, rewritten: Sequence[str]
) -> tuple[ClassifiedEntry, ...]:
    """Say, for every file of a tree, how it relates to a reference copy.

    Args:
        tree: The tree being described, in listing order.
        reference: An independent copy of the same build's content, whose
            files are the ones a match can be verified against.
        rewritten: Paths the game rewrites on every boot, from
            :data:`rw_bot.harness.clone.VOLATILE_FILES`. Required and passed
            in rather than known here: this is the one exception to the
            same-path-must-agree rule, and an exception with a second owner is
            an exception that grows.

    Returns:
        One classified entry per tree entry, in the order given.

    Raises:
        StagedTreeError: ``RW-TREE-101`` when a path exists in both, their
            bytes differ, and the game does not rewrite it. That is the case
            no later check would catch: the digest matches the file it came
            from and the transfer matches the digest, so a tree carrying the
            wrong version of a file stages clean and runs a different
            simulation.
        StagedTreeError: ``RW-TREE-103`` when the tree has no entries.
    """
    if len(tree) == 0:
        raise StagedTreeError(
            _NO_ENTRIES, "a tree with no entries describes nothing and stages nothing"
        )
    at_path = {entry["path"]: entry["sha256"] for entry in reference}
    # First path wins, and the reference arrives byte-sorted, so a file
    # duplicated in the reference always names the same source here.
    at_digest: dict[str, str] = {}
    for entry in reference:
        if entry["size_bytes"] > _EMPTY:
            at_digest.setdefault(entry["sha256"], entry["path"])
    volatile = set(rewritten)

    classified: list[ClassifiedEntry] = []
    for entry in tree:
        twin = at_path.get(entry["path"])
        elsewhere = at_digest.get(entry["sha256"]) if entry["size_bytes"] > _EMPTY else None
        if entry["path"] in volatile:
            classified.append(_classified(entry, ORIGIN_PINNED_STATE, NO_SOURCE))
        elif twin is not None and twin != entry["sha256"]:
            raise StagedTreeError(
                _DIVERGED,
                f"{entry['path']} is {entry['sha256']} here and {twin} in the reference: "
                "the two copies of one file disagree, and nothing downstream would notice "
                "-- the digest matches the bytes it came from either way",
            )
        elif twin is not None:
            classified.append(_classified(entry, ORIGIN_VERIFIED, entry["path"]))
        elif elsewhere is not None:
            classified.append(_classified(entry, _elsewhere_origin(entry, elsewhere), elsewhere))
        else:
            classified.append(_classified(entry, ORIGIN_ASSEMBLED, NO_SOURCE))
    return tuple(classified)


def _elsewhere_origin(entry: TreeEntry, source: str) -> str:
    """Say which kind of same-bytes-elsewhere an entry is.

    The distinction is what a reader is actually asking about. Sixty-three of
    these are vendor files the two bundled JVMs both ship -- ``COPYRIGHT``,
    fonts, ``cacerts`` -- sitting at ``jvm-linux/x`` here and ``jvm/x`` in the
    reference. Seven are copies this PROJECT made, under shell-safe names,
    because the launcher passes a map on a command line. Calling all seventy
    "renamed" buries the seven that a person decided.

    Told apart by the filename: the same name in another directory is the
    vendor shipping one file to two trees; a different name is somebody here
    having made a copy.

    Args:
        entry: The tree entry.
        source: The reference path holding the same bytes.

    Returns:
        :data:`ORIGIN_MIRRORED` or :data:`ORIGIN_RENAMED`.
    """
    if entry["path"].split("/")[-1] == source.split("/")[-1]:
        return ORIGIN_MIRRORED
    return ORIGIN_RENAMED


def _classified(entry: TreeEntry, origin: str, source: str) -> ClassifiedEntry:
    """Attach an origin to a tree entry.

    Args:
        entry: The entry.
        origin: One of :data:`ORIGINS`.
        source: The reference path it was checked against, or
            :data:`NO_SOURCE`.

    Returns:
        The classified entry.
    """
    return ClassifiedEntry(
        path=entry["path"],
        sha256=entry["sha256"],
        size_bytes=entry["size_bytes"],
        origin=origin,
        source=source,
    )


def count_origins(entries: Sequence[ClassifiedEntry]) -> dict[str, int]:
    """Count how many entries each origin accounts for.

    The number a reader checks first, and the reason it is computed rather
    than written into prose: "1,445 verified" is a claim that goes stale the
    moment the tree moves, and a document that carries its own tally cannot.

    Args:
        entries: The classified entries.

    Returns:
        A count per origin, every origin present even at zero -- an absent key
        and a zero read alike otherwise, and only one of them means the tree
        has no such file.
    """
    return {origin: sum(1 for entry in entries if entry["origin"] == origin) for origin in ORIGINS}


def encode_depot(depot: DepotRef) -> JSONObject:
    """Encode a depot reference to a JSON object.

    Args:
        depot: The reference.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {"depot_id": depot["depot_id"], "manifest": depot["manifest"], "role": depot["role"]}


def decode_depot(value: JSONValue) -> DepotRef:
    """Decode and validate a depot reference.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated reference.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing,
            not a string, or empty. A depot with no manifest names a moving
            target, which is the one thing the document must not do.
    """
    obj = narrow_json_to_dict(value)
    return DepotRef(
        depot_id=_require_filled(obj, "depot_id"),
        manifest=_require_filled(obj, "manifest"),
        role=_require_filled(obj, "role"),
    )


def _require_filled(obj: JSONObject, key: str) -> str:
    """Read a required string field that must carry characters.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def encode_classified_entry(entry: ClassifiedEntry) -> JSONObject:
    """Encode a classified entry to a JSON object.

    Args:
        entry: The entry.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        **encode_tree_entry(entry),
        "origin": entry["origin"],
        "source": entry["source"],
    }


def decode_classified_entry(value: JSONValue) -> ClassifiedEntry:
    """Decode and validate a classified entry.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated entry.

    Raises:
        JSONTypeError: If the underlying tree entry is invalid, or ``source``
            is absent.
        StagedTreeError: ``RW-TREE-102`` when the origin is not one of
            :data:`ORIGINS`. Refused rather than carried through, because a
            reader tallying origins would silently drop it.
    """
    obj = narrow_json_to_dict(value)
    entry = decode_tree_entry(value)
    origin = require_str(obj, "origin")
    if origin not in ORIGINS:
        raise StagedTreeError(
            _UNKNOWN_ORIGIN,
            f"{origin!r} is not an origin this document defines; expected one of "
            f"{', '.join(ORIGINS)}",
        )
    return _classified(entry, origin, require_str(obj, "source"))


def encode_staged_tree_spec(spec: StagedTreeSpec) -> JSONObject:
    """Encode a staged-tree spec to a JSON object.

    Args:
        spec: The spec.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    depots: list[JSONValue] = [encode_depot(depot) for depot in spec["depots"]]
    entries: list[JSONValue] = [encode_classified_entry(entry) for entry in spec["entries"]]
    return {
        "app_id": spec["app_id"],
        "build_id": spec["build_id"],
        "depots": depots,
        "reference": spec["reference"],
        "entries": entries,
        "tree_sha256": spec["tree_sha256"],
        "archive_name": spec["archive_name"],
        "archive_sha256": spec["archive_sha256"],
        "archive_size_bytes": spec["archive_size_bytes"],
    }


def decode_staged_tree_spec(value: JSONValue) -> StagedTreeSpec:
    """Decode and validate a staged-tree spec.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated spec.

    Raises:
        JSONTypeError: If the document is not an object, a field is missing or
            mistyped, or the depot or entry lists are empty.
        StagedTreeError: ``RW-TREE-102`` when an entry names an unknown
            origin, or ``RW-TREE-103`` when the tree digest does not match the
            entries the document carries. The second is the load-bearing one:
            a document whose listing and whose digest disagree describes two
            different trees, and the digest is what everything downstream
            compares against.
    """
    obj = narrow_json_to_dict(value)
    depots = [decode_depot(item) for item in require_list(obj, "depots")]
    if depots == []:
        raise JSONTypeError("Field 'depots' must name at least one depot")
    entries = [decode_classified_entry(item) for item in require_list(obj, "entries")]
    if entries == []:
        raise JSONTypeError("Field 'entries' must not be empty")

    digest = _require_filled(obj, "tree_sha256")
    recomputed = tree_digest(entries)
    if recomputed != digest:
        raise StagedTreeError(
            _NO_ENTRIES,
            f"the document states tree_sha256 {digest} but its own {len(entries)} entries "
            f"digest to {recomputed}: the listing and the identity describe different trees",
        )
    size = require_int(obj, "archive_size_bytes")
    if size < 1:
        raise JSONTypeError(f"Field 'archive_size_bytes' must be at least 1, got {size}")
    return StagedTreeSpec(
        app_id=_require_filled(obj, "app_id"),
        build_id=_require_filled(obj, "build_id"),
        depots=depots,
        reference=_require_filled(obj, "reference"),
        entries=entries,
        tree_sha256=digest,
        archive_name=_require_filled(obj, "archive_name"),
        archive_sha256=_require_filled(obj, "archive_sha256"),
        archive_size_bytes=size,
    )


def render_digest_record(spec: StagedTreeSpec) -> tuple[str, ...]:
    """Render the published record staging is held against.

    ``hpc3-stage`` requires ``--expect-from``: a document, written by a
    different act than the staging, naming the digests the work is entitled
    to. It reads loosely -- every 64-character lowercase hex token in the text
    counts -- so this is written for a person and read by the tool.

    Args:
        spec: The described tree.

    Returns:
        Header lines naming the depots and the tree, then one
        ``sha256sum --text`` line per entry, then the archive's own digest.
        The archive line is what the stage manifest is actually checked
        against; the per-file lines are what let a reader find which file
        moved when the tree digest changes.

    Raises:
        StagedTreeError: ``RW-TREE-103`` when the spec carries no entries.
    """
    if spec["entries"] == []:
        raise StagedTreeError(_NO_ENTRIES, "a spec with no entries vouches for nothing")
    counts = count_origins(spec["entries"])
    return digest_record(
        [
            f"# {spec['app_id']} build {spec['build_id']}, verified against {spec['reference']}",
            *(
                f"# depot {depot['depot_id']} manifest {depot['manifest']} ({depot['role']})"
                for depot in spec["depots"]
            ),
            *(f"# {counts[origin]} {origin}" for origin in ORIGINS),
            f"# tree {spec['tree_sha256']}",
        ],
        spec["archive_name"],
        spec["archive_sha256"],
        spec["entries"],
    )


__all__ = [
    "APP_ID",
    "CONTENT_DEPOT",
    "LINUX_DEPOT",
    "NO_SOURCE",
    "ORIGINS",
    "ORIGIN_ASSEMBLED",
    "ORIGIN_MIRRORED",
    "ORIGIN_PINNED_STATE",
    "ORIGIN_RENAMED",
    "ORIGIN_VERIFIED",
    "ROLE_CONTENT",
    "ROLE_LINUX_RUNTIME",
    "ClassifiedEntry",
    "DepotRef",
    "StagedTreeError",
    "StagedTreeSpec",
    "classify_entries",
    "count_origins",
    "decode_classified_entry",
    "decode_depot",
    "decode_staged_tree_spec",
    "encode_classified_entry",
    "encode_depot",
    "encode_staged_tree_spec",
    "render_digest_record",
]
