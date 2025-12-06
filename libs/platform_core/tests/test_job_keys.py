from __future__ import annotations

from platform_core import job_keys
from platform_core.trainer_keys import status_key as _status_key
from platform_core.turkic_jobs import turkic_job_key as _tjk


def test_job_keys_reexport_and_helpers() -> None:
    assert job_keys.turkic_job_key("x") == _tjk("x")
    assert job_keys.trainer_run_status_key("y") == _status_key("y")
    assert job_keys.artifact_file_id_key("z").endswith(":file_id")
