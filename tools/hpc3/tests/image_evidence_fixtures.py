"""Pinned image build inputs and native readings for the shared inspection tests."""

from platform_core.json_utils import JSONValue

from hpc3.contracts.ledger import LedgerEntry
from hpc3.core.digest import sha256_hex
from tests.conftest import FakeRun
from tests.execution_fixtures import IMAGE, INSTANT, accounting_row, entry, serve_text

DIRECTORY = "/pub/build"
IMAGE_PATH = DIRECTORY + "/runtime.sif"
LOG_PATH = DIRECTORY + "/build-101.out"
FILES = ("build.sbatch", "build.sh", "image.def", "requirements.txt", "selfcheck.py", "GIT_COMMIT")
LOG = "\n".join(
    (
        "image self-check OK",
        f"{IMAGE}  runtime.sif",
        "image self-check OK",
        "re-verify exit 0",
        "--- smoke 1/1: /opt/env/bin/python -m pip check ---",
        "No broken requirements found.",
        "smoke 1 OK",
        "exit      0",
        "",
    )
)


def build_entry(**changes: JSONValue) -> LedgerEntry:
    """Construct a ledger-recorded build, including explicit invalid variants."""
    fields: dict[str, JSONValue] = {
        "experiment": {"kind": "image-build", "image_dir": DIRECTORY},
        "log_dir": DIRECTORY,
        "artifact": IMAGE_PATH,
        "image_digest": "",
    }
    fields.update(changes)
    return entry(**fields)


def pins(log: str = LOG) -> dict[str, str]:
    """Include every generated build file, the image and its native output."""
    return {
        **{f"{DIRECTORY}/{name}": "b" * 64 for name in FILES},
        IMAGE_PATH: IMAGE,
        LOG_PATH: sha256_hex(log.encode()),
    }


def serve_build(fake: FakeRun, log: str = LOG) -> None:
    """Answer actual command forms through the existing transport seam."""
    fake.add("sacct", stdout=accounting_row(build_entry()))
    for path, digest in pins(log).items():
        fake.add(f"sha256sum '{path}'", stdout=f"{digest}  {path}\n")
    serve_text(fake, LOG_PATH, log)
    fake.add("stat -c %Y", stdout=str(INSTANT))
