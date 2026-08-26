"""Rendering a job that runs inside an image rather than out of a directory.

Split from ``test_sbatch`` when that module passed the 600-line ceiling. Both
build their specs from ``_sbatch_support`` so the image rendering and the
host rendering stay two views of the SAME job -- an image test that drifted
onto its own baseline would stop testing the difference it exists to test.

Every assertion here has a measured failure behind it. The image route
shipped with three of them live at once: the payload could not see its data
(no binds), the preflight could not see the environment (host probe), and the
payload could not see the GPU (no ``--nv``). Each one starts cleanly and
fails somewhere that does not name the cause.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.core.sbatch import job_comment, render_sbatch
from tests._sbatch_support import IMAGE, LOG_DIR, SIF, spec

_DIGEST = "9ed4e27fd0d8207de3f84e833b98e0cf7e6ab09af66726849ca1cf023326cd51"


def _image(**overrides: JSONValue) -> JSONValue:
    """Build an image reference with optional overrides.

    Args:
        **overrides: Fields to replace in the baseline reference.

    Returns:
        A JSON image reference.
    """
    base: dict[str, JSONValue] = {"path": SIF, "sha256": _DIGEST, "binds": ["/pub/wagnera3"]}
    base.update(overrides)
    return base


class TestThePayloadRunsThroughApptainer:
    def test_the_image_is_the_argument_apptainer_receives(self) -> None:
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR)
        assert f'    "{SIF}" \\' in script.splitlines()

    def test_the_apptainer_module_is_loaded(self) -> None:
        """`which apptainer` returns nothing on HPC3 until the module loads."""
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR)
        assert "module load apptainer/1.4.5" in script.splitlines()

    def test_path_is_set_inside_the_container(self) -> None:
        """env_path names a container directory once an image is present."""
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR)
        assert '    env PATH="/opt/env/bin:$PATH" \\' in script.splitlines()

    def test_an_image_run_refuses_a_host_bound_env_path(self) -> None:
        """The baseline env_path is a HOST path, so pairing it with an image
        must be refused rather than rendered into a script that silently
        resolves the wrong interpreter."""
        with pytest.raises(JSONTypeError, match="bind-mounts over"):
            _ = spec(image=IMAGE)

    def test_a_host_run_loads_no_module_and_calls_no_apptainer(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert "module load apptainer/1.4.5" not in script
        assert "apptainer exec" not in script

    def test_a_host_run_exports_path_directly(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR).splitlines()
        assert 'export PATH="/pub/wagnera3/envs/abl-pinned/bin:$PATH"' in script


class TestTheDataIsBoundWhereThePayloadExpectsIt:
    """Without this the job starts and finds none of its data.

    /pub on HPC3 is a symlink to /dfs6b/pub; apptainer carries the BeeGFS
    mounts but not the symlink, so an unbound /pub/... does not resolve
    inside the container at all. Measured 2026-08-25.
    """

    def test_a_bind_is_mounted_at_its_own_path(self) -> None:
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR)
        assert '    --bind "/pub/wagnera3:/pub/wagnera3" \\' in script.splitlines()

    def test_every_declared_bind_is_emitted(self) -> None:
        image = _image(binds=["/pub/wagnera3", "/dfs7/scratch"])
        script = render_sbatch(spec(image=image, env_path="/opt/env"), log_dir=LOG_DIR)
        assert script.count("--bind") == 2

    def test_the_binds_precede_the_image(self) -> None:
        """apptainer reads options before the image argument."""
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR).splitlines()
        bind = script.index('    --bind "/pub/wagnera3:/pub/wagnera3" \\')
        assert bind < script.index(f'    "{SIF}" \\')

    def test_an_image_with_no_binds_emits_none(self) -> None:
        """A self-contained computation needs nothing mounted."""
        script = render_sbatch(spec(image=_image(binds=[]), env_path="/opt/env"), log_dir=LOG_DIR)
        assert "--bind" not in script


class TestTheGpuIsReachableFromInsideTheImage:
    """A container carries CUDA but not the DRIVER, which is the host kernel's.

    Job 55589876 died in six seconds with "Found no NVIDIA driver on your
    system" on hpc3-gpu-l54-05 -- whose own prologue had printed "NVIDIA A100
    80GB PCIe, 81920 MiB" one line above the traceback, because the prologue
    runs on the host and the payload did not. Measured both ways on a
    free-gpu A100: `torch.cuda.is_available()` is False without `--nv` and
    True with it.
    """

    def test_a_gpu_job_in_an_image_binds_the_hosts_driver(self) -> None:
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR)
        assert "    --nv \\" in script.splitlines()

    def test_the_driver_flag_precedes_the_image(self) -> None:
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR).splitlines()
        assert script.index("    --nv \\") < script.index(f'    "{SIF}" \\')

    def test_a_cpu_job_in_an_image_asks_for_no_driver(self) -> None:
        """--nv on a node with no driver is an error, not a no-op."""
        script = render_sbatch(
            spec(image=IMAGE, env_path="/opt/env", gpu=None, partition="free"),
            log_dir=LOG_DIR,
        )
        assert "--nv" not in script

    def test_a_host_gpu_run_has_no_driver_flag_to_pass(self) -> None:
        """There is no container between the payload and the driver."""
        assert "--nv" not in render_sbatch(spec(), log_dir=LOG_DIR)


class TestTheRunCanSayWhichEnvironmentProducedIt:
    def test_the_commit_stamp_is_read_from_inside_the_image(self) -> None:
        """A bare `cat` would look on the host, find nothing, and export empty
        -- reporting an image that does know its commit as unstamped."""
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR)
        assert (
            'export GIT_COMMIT="$(apptainer exec "/pub/wagnera3/images/abl.sif" '
            "cat /opt/env/GIT_COMMIT 2>/dev/null || echo '')\"" in script.splitlines()
        )

    def test_the_module_loads_before_the_stamp_is_read(self) -> None:
        lines = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR).splitlines()
        stamp = next(i for i, line in enumerate(lines) if line.startswith("export GIT_COMMIT="))
        assert lines.index("module load apptainer/1.4.5") < stamp

    def test_the_image_digest_is_exported_for_the_payload(self) -> None:
        """capture_run_fingerprint reads this to decide comparability.

        An image cannot compute its own digest from inside itself, so the
        launcher -- which pins it in the spec -- is where it comes from.
        """
        script = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR)
        assert f'export IMAGE_DIGEST="{_DIGEST}"' in script.splitlines()

    def test_a_host_run_exports_no_digest(self) -> None:
        """Unset reads as unknown, which is true: there is no image."""
        assert "IMAGE_DIGEST" not in render_sbatch(spec(), log_dir=LOG_DIR)

    def test_the_digest_is_exported_before_the_payload(self) -> None:
        lines = render_sbatch(spec(image=IMAGE, env_path="/opt/env"), log_dir=LOG_DIR).splitlines()
        export = next(i for i, line in enumerate(lines) if line.startswith("export IMAGE_DIGEST="))
        payload = next(i for i, line in enumerate(lines) if line.startswith("apptainer exec"))
        assert export < payload

    def test_the_queue_records_the_digest_not_the_path(self) -> None:
        """A path names a place that can be rebuilt; a digest names bytes."""
        assert ";env=sif:9ed4e27fd0d8;" in job_comment(spec(image=IMAGE, env_path="/opt/env"))

    def test_a_host_run_records_its_directory(self) -> None:
        assert ";env=/pub/wagnera3/envs/abl-pinned;" in job_comment(spec())
