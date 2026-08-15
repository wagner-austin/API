from __future__ import annotations

import os
import time
import uuid

from platform_core.json_utils import JSONValue
from platform_core.trainer_keys import heartbeat_key, status_key

from model_trainer.core import _test_hooks


class RunStore:
    artifacts_root: str

    def __init__(self: RunStore, artifacts_root: str) -> None:
        self.artifacts_root = artifacts_root

    def create_run(self: RunStore, model_family: str, model_size: str) -> str:
        """Allocate a run id and its artifact directory.

        Args:
            model_family: Model architecture family for the run.
            model_size: Model size variant for the run.

        Returns:
            The newly allocated run id.

        Raises:
            FileExistsError: If the artifact directory already exists, which
                means the id was not unique.
        """
        ts = int(time.time())
        # The id carried only a one-second timestamp, so two runs of the same
        # family and size submitted within the same second received the same
        # id -- and `exist_ok=True` let them silently share one artifact
        # directory, overwriting each other's manifest and weights. Submitting
        # a four-arm ablation in a loop reproduced it every time. The random
        # suffix makes the id unique, and `exist_ok=False` turns any remaining
        # collision into a loud failure rather than silent data loss.
        run_id = f"{model_family}-{model_size}-{ts}-{uuid.uuid4().hex[:8]}"
        # Create run directory under artifacts to avoid relying on a separate runs_root
        artifacts_dir = os.path.join(self.artifacts_root, "models", run_id)
        os.makedirs(artifacts_dir, exist_ok=False)
        # Write a small manifest for reproducibility and cross-links in the artifacts dir
        manifest_path = os.path.join(artifacts_dir, "manifest.json")
        body: dict[str, JSONValue] = {
            "run_id": run_id,
            "created_at": ts,
            "model_family": model_family,
            "model_size": model_size,
            "artifacts_dir": artifacts_dir,
            "logs_path": os.path.join(artifacts_dir, "logs.jsonl"),
            "status_key": status_key(run_id),
            "heartbeat_key": heartbeat_key(run_id),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(_test_hooks.dump_json_str(body))
        return run_id
