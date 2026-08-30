"""The committed replication panel, held to what the generator emits today.

A campaign document is codegen that is COMMITTED, which is the shape that
rots quietly: the generator moves, the file on disk does not, and nothing
notices until a batch runs the old command. That is not hypothetical here.
The panel committed under ``provenance/replicate-campaign.json`` was written
before the payload tree became a staged thing, so every one of its 24 members
still said

    --jobs /pub/wagnera3/rusted/sweeps/replicate.txt

with no ``--tree`` at all, while the generator had moved to

    --jobs /pub/wagnera3/rusted/payload/sweeps/replicate.txt --tree /pub/wagnera3/rusted/payload

Submitted as committed, all 24 members would have failed on a job file that is
not at that path any more -- 24 jobs' worth of queue time to learn something a
regeneration answers for free.

So this rebuilds the document from the committed workspace and the committed
job file and holds the committed bytes to it. The experiment block is read
from the document rather than typed here: what is being checked is that the
COMMANDS match the generator for the experiment this panel declares, and a
map name retyped in a test is one more copy to go stale.
"""

from __future__ import annotations

from pathlib import Path

from hpc3.contracts.workspace import decode_workspace, require_project_config
from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_str,
)
from scripts.campaign_doc import DUEL_OPPONENTS, PROJECT, campaign_document

from rw_bot.harness.match import decode_match_config
from rw_bot.harness.sweep import parse_jobs

#: This repository's root, from this file rather than a working directory.
_ROOT = Path(__file__).resolve().parents[1]

#: The workspace the panel was generated against. Outside this repository
#: because a statement about the cluster belongs in the hpc3 workspace, and
#: read rather than mirrored for exactly that reason.
_WORKSPACE = _ROOT.parents[1] / "tools" / "hpc3" / "runs" / "hpc3-rusted.json"

#: The committed panel, and the job file it was generated from. The job file
#: is named repository-relative because that is the form the generator was
#: given; naming the wrong one fails the comparison rather than passing it.
_DOCUMENT = _ROOT / "provenance" / "replicate-campaign.json"
_JOBS_FILE = "sweeps/replicate.txt"


def _committed(path: Path) -> JSONObject:
    """Read one committed JSON document as an object.

    Args:
        path: The file to read.

    Returns:
        The decoded document.
    """
    return narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))


class TestTheCommittedPanel:
    """The panel on disk is the panel the generator produces."""

    def test_every_member_matches_what_the_generator_emits_today(self) -> None:
        """The whole document, not a sampled member: a drift that moved one
        flag moved it on all 24, and a drift that moved only some is worse."""
        document = _committed(_DOCUMENT)
        experiment = require_dict(document, "experiment")
        workspace = decode_workspace(
            load_json_str(_WORKSPACE.read_text(encoding="utf-8")), config_dir=_WORKSPACE.parent
        )
        config = require_project_config(workspace, PROJECT)
        jobs = parse_jobs((_ROOT / _JOBS_FILE).read_text(encoding="utf-8").splitlines())
        match = decode_match_config(
            {
                "map_path": require_str(experiment, "map"),
                "opponents": DUEL_OPPONENTS,
                "difficulty": int(require_str(experiment, "difficulty")),
            }
        )
        rebuilt = campaign_document(
            workspace["root"],
            config["env_path"],
            _JOBS_FILE,
            require_str(experiment, "batch"),
            jobs,
            int(require_str(experiment, "lockstep")),
            match,
        )
        assert rebuilt == document

    def test_the_project_declares_the_image_its_members_run_inside(self) -> None:
        """Without one, ``env_path: /opt/env`` names an interpreter that
        exists in no image and on no compute node, and every member of the
        panel dies on 'no such file or directory' before the game starts."""
        workspace = decode_workspace(
            load_json_str(_WORKSPACE.read_text(encoding="utf-8")), config_dir=_WORKSPACE.parent
        )
        image = require_project_config(workspace, PROJECT)["image"]
        if image is None:
            raise AssertionError(f"project {PROJECT!r} declares no image")
        assert image["path"].endswith(".sif")
        assert image["binds"] == [workspace["root"]]
