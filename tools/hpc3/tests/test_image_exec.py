"""Tests for the one place that wraps a command in a job's image.

The defect these were written against was live: preflight probed the host for
an environment that exists only inside the ``.sif``, so every image job was
refused for a directory that was never meant to be on the cluster. The
wrapping has to agree with what the batch script does, because a preflight
that verifies a different world than the job runs in is worse than no
preflight -- it reports green and the job still fails.
"""

from __future__ import annotations

from hpc3.contracts.image import ImageReference
from hpc3.core.image_exec import (
    APPTAINER_MODULE,
    bind_arguments,
    describe_location,
    gpu_arguments,
    run_inside_image,
)

_IMAGE = ImageReference(
    path="/pub/wagnera3/images/v3/abl.sif",
    sha256="3bd2e694857821c383c5e90b1c23b63c706ece20069a6bf34431512ef1c041d4",
    binds=["/pub/wagnera3"],
)

_NO_BINDS = ImageReference(path="/i/x.sif", sha256="ab" * 32, binds=[])

_TWO_BINDS = ImageReference(
    path="/i/x.sif", sha256="ab" * 32, binds=["/pub/wagnera3", "/dfs9/scratch"]
)


class TestBindArguments:
    def test_each_bind_is_mounted_at_its_own_path(self) -> None:
        """A payload's absolute paths must mean the same thing on both sides."""
        assert bind_arguments(_IMAGE) == ['--bind "/pub/wagnera3:/pub/wagnera3"']

    def test_binds_keep_their_declared_order(self) -> None:
        assert bind_arguments(_TWO_BINDS) == [
            '--bind "/pub/wagnera3:/pub/wagnera3"',
            '--bind "/dfs9/scratch:/dfs9/scratch"',
        ]

    def test_an_image_declaring_no_binds_renders_no_arguments(self) -> None:
        assert bind_arguments(_NO_BINDS) == []


class TestGpuArguments:
    """A container carries CUDA but not the driver, which is the host's.

    Job 55589876 landed on hpc3-gpu-l54-05, whose prologue printed "NVIDIA
    A100 80GB PCIe, 81920 MiB" -- and six seconds later the payload died with
    "Found no NVIDIA driver on your system". The prologue runs on the host;
    the payload ran inside the image without --nv. Measured both ways on a
    free-gpu A100: is_available() False without, True with.
    """

    def test_a_gpu_job_binds_the_hosts_driver(self) -> None:
        assert gpu_arguments({"model": "A100", "count": 1}) == ["--nv"]

    def test_the_count_does_not_change_the_flag(self) -> None:
        """--nv binds the driver stack, not one device."""
        assert gpu_arguments({"model": "A100", "count": 4}) == ["--nv"]

    def test_a_cpu_job_asks_for_no_driver(self) -> None:
        """--nv on a node without a driver is an error, not a no-op."""
        assert gpu_arguments(None) == []


class TestRunInsideImage:
    def test_it_loads_the_module_before_reaching_for_apptainer(self) -> None:
        """`which apptainer` returns nothing on a login node until loaded."""
        assert run_inside_image(_IMAGE, "true").startswith(
            f"module load {APPTAINER_MODULE} && apptainer exec "
        )

    def test_the_module_load_is_chained_so_its_failure_is_its_own(self) -> None:
        """Sequencing with ';' would run apptainer anyway.

        A probe that reports absence by printing a token would then print
        ABSENT because apptainer was not found, and the caller would
        diagnose the image instead of the module.
        """
        assert f"module load {APPTAINER_MODULE} && apptainer" in run_inside_image(_IMAGE, "true")

    def test_the_image_is_named_by_path_because_apptainer_takes_a_file(self) -> None:
        assert '"/pub/wagnera3/images/v3/abl.sif"' in run_inside_image(_IMAGE, "true")

    def test_the_command_is_handed_to_a_shell_inside_the_container(self) -> None:
        """A host probe keeps its meaning without being rewritten for argv."""
        line = run_inside_image(_IMAGE, "test -d '/opt/env/bin' && echo PRESENT || echo ABSENT")
        assert line.endswith(
            "sh -c 'test -d '\"'\"'/opt/env/bin'\"'\"' && echo PRESENT || echo ABSENT'"
        )

    def test_the_wrapped_command_survives_embedded_double_quotes(self) -> None:
        """The package probe is `'<env>/bin/python' -c "..."` -- both quotes."""
        inner = "'/opt/env/bin/python' -c \"import sys;print(sys.version)\""
        line = run_inside_image(_IMAGE, inner)
        assert "/opt/env/bin/python" in line
        assert "import sys;print(sys.version)" in line

    def test_an_image_with_no_binds_renders_no_stray_separator(self) -> None:
        """An empty bind list must not leave a double space or a bare --bind."""
        line = run_inside_image(_NO_BINDS, "true")
        assert "--bind" not in line
        assert "  " not in line

    def test_the_whole_line_is_exactly_what_a_remote_shell_receives(self) -> None:
        assert run_inside_image(_NO_BINDS, "true") == (
            f'module load {APPTAINER_MODULE} && apptainer exec "/i/x.sif" sh -c true'
        )


class TestDescribeLocation:
    def test_a_host_check_names_the_host(self) -> None:
        assert describe_location(None, "hpc3") == "on hpc3"

    def test_an_image_check_names_the_image_not_the_host(self) -> None:
        """ "on hpc3" for a container path sends the reader to the one place
        that cannot answer, which is how this defect read as a broken path."""
        assert describe_location(_IMAGE, "hpc3") == "inside /pub/wagnera3/images/v3/abl.sif"
