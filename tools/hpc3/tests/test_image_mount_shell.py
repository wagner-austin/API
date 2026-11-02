"""Execute the rendered shell boundary; native mount behavior needs HPC3 evidence."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hpc3.contracts.image import ImageReference, encode_image_reference
from hpc3.core.image_exec import run_inside_image
from hpc3.core.sbatch import payload_lines
from tests._sbatch_support import spec
from tests.bash_discovery import posix_bash


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("payload", [False, True])
def test_shell_preserves_paths_removes_ambient_mounts_and_propagates_exit(
    tmp_path: Path, exit_code: int, payload: bool
) -> None:
    recorder = tmp_path / "apptainer"
    recorder.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\0" "$@"\n'
        'printf "%s\\0" "${APPTAINER_BIND-unset}" "${APPTAINER_BINDPATH-unset}" '
        '"${APPTAINER_MOUNT-unset}" "${SINGULARITY_BIND-unset}" '
        '"${SINGULARITY_BINDPATH-unset}" "${SINGULARITY_MOUNT-unset}"\n'
        f"exit {exit_code}\n"
    )
    recorder.chmod(0o700)
    image = ImageReference(
        path="/home/image $(touch UNWANTED).sif",
        sha256="ab" * 32,
        binds=["/home/data $(touch UNWANTED)"],
    )
    environment = dict(os.environ)
    environment["PATH"] = str(tmp_path) + os.pathsep + environment["PATH"]
    for prefix in ("APPTAINER", "SINGULARITY"):
        for suffix in ("BIND", "BINDPATH", "MOUNT"):
            environment[f"{prefix}_{suffix}"] = "/undeclared/filesystem"
    if payload:
        job = spec(
            image=encode_image_reference(image),
            env_path="/opt/env",
            command="true",
            gpu=None,
            partition="free",
        )
        rendered = "\n".join(payload_lines(job))
        command = ["env", f"PATH=/opt/env/bin:{environment['PATH']}", "true"]
    else:
        rendered = "module() { return 0; }\n" + run_inside_image(image, "true")
        command = ["sh", "-c", "true"]
    result = subprocess.run(
        [posix_bash(), "-c", rendered],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr.decode()
    assert result.stdout.decode().split("\0") == [
        "exec",
        "--no-mount",
        "bind-paths,hostfs",
        "--bind",
        "/home/data $(touch UNWANTED):/home/data $(touch UNWANTED)",
        image["path"],
        *command,
        *["unset"] * 6,
        "",
    ]
    assert not (tmp_path / "UNWANTED").exists()
