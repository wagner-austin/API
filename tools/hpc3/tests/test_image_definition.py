"""Tests for the rendered definition and requirements file.

The assertions here are about the submitted text, because that text is what
apptainer acts on -- the same posture as the sbatch tests. Three of them
exist to prove a contract rule survived into the output rather than merely
being validated on the way in.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue

from hpc3.contracts.image_spec import ImageSpec, decode_image_spec
from hpc3.core.image_definition import render_definition, render_requirements
from hpc3.core.image_layout import (
    COMMIT_NAME,
    REQUIREMENTS_NAME,
    SELFCHECK_NAME,
    SPEC_DIR,
    WHEEL_DIR,
)


def _spec(**overrides: JSONValue) -> ImageSpec:
    """Build a valid spec with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        The decoded spec.
    """
    base: dict[str, JSONValue] = {
        "base_image": "python:3.11.16-slim-bookworm",
        "env_prefix": "/opt/env",
        "git_commit": "d11efacd231ef92426eaf92483c33a8504bd770f",
        "system_packages": [],
        "extra_index_urls": ["https://download.pytorch.org/whl/cu124"],
        "requirements": ["torch==2.6.0+cu124", "transformers==4.46.3"],
        "wheels": ["model_trainer_server-0.1.0-py3-none-any.whl"],
        "expected_versions": {"torch": "2.6.0+cu124"},
        "required_symbols": [{"module": "model_trainer", "attribute": "__file__"}],
        "smoke_commands": [],
        "labels": {"org.corvis.captured": "2026-08-25"},
    }
    base.update(overrides)
    return decode_image_spec(base)


class TestTheOperatingSystemLayer:
    """Not every dependency is a wheel: a JVM, an X server and a software GL
    stack cannot be pip-installed, and an image that could only describe its
    Python layer forced a hand-built base nobody could reproduce."""

    def test_an_image_with_no_packages_renders_no_apt_call_at_all(self) -> None:
        """``apt-get update`` alone costs a minute of build time and a network
        dependency, for no result."""
        rendered = render_definition(_spec(system_packages=[]))
        assert "apt-get" not in rendered

    def test_the_declared_packages_are_installed(self) -> None:
        packages: JSONValue = [
            "xvfb=2:21.1.4-2ubuntu1.7",
            "openjdk-17-jre-headless=17.0.13+11-2",
        ]
        rendered = render_definition(_spec(system_packages=packages))
        assert "xvfb=2:21.1.4-2ubuntu1.7" in rendered
        assert "openjdk-17-jre-headless=17.0.13+11-2" in rendered

    def test_it_installs_before_the_virtualenv_is_built(self) -> None:
        """The interpreter the virtualenv is built from may itself be one of
        these packages, so the order is load-bearing rather than tidy."""
        rendered = render_definition(_spec(system_packages=["xvfb=2:21.1.4-2ubuntu1.7"]))
        assert rendered.index("install -y --no-install-recommends") < rendered.index(
            "python -m venv"
        )

    def test_nothing_recommended_is_installed(self) -> None:
        """A recommended package is by definition one nothing declared, and
        the spec is meant to be the whole list."""
        rendered = render_definition(_spec(system_packages=["xvfb=2:21.1.4-2ubuntu1.7"]))
        assert "--no-install-recommends" in rendered

    def test_the_package_index_is_not_left_in_the_layer(self) -> None:
        rendered = render_definition(_spec(system_packages=["xvfb=2:21.1.4-2ubuntu1.7"]))
        assert "rm -rf /var/lib/apt/lists/*" in rendered

    def test_the_install_is_non_interactive(self) -> None:
        """A build job has no terminal; a package that asks a question would
        hang until the scheduler killed it."""
        rendered = render_definition(_spec(system_packages=["xvfb=2:21.1.4-2ubuntu1.7"]))
        assert "DEBIAN_FRONTEND=noninteractive" in rendered

    def test_a_single_package_needs_no_trailing_continuation(self) -> None:
        """A dangling backslash before ``rm`` would swallow the next line."""
        rendered = render_definition(_spec(system_packages=["xvfb=2:21.1.4-2ubuntu1.7"]))
        lines = [line.strip() for line in rendered.splitlines()]
        assert "xvfb=2:21.1.4-2ubuntu1.7" in lines

    def test_the_definition_still_ends_with_a_newline(self) -> None:
        assert render_definition(_spec(system_packages=["xvfb=2:21.1.4-2ubuntu1.7"])).endswith("\n")


class TestRequirements:
    """Ordering matters: pip reads index options file-wide."""

    def test_the_index_url_precedes_every_requirement(self) -> None:
        lines = render_requirements(_spec()).splitlines()
        index = lines.index("--extra-index-url https://download.pytorch.org/whl/cu124")
        torch = lines.index("torch==2.6.0+cu124")
        assert index < torch

    def test_a_local_version_survives_verbatim(self) -> None:
        """``+cu124`` resolves only against the index above it."""
        assert "torch==2.6.0+cu124" in render_requirements(_spec()).splitlines()

    def test_no_index_line_when_none_declared(self) -> None:
        rendered = render_requirements(_spec(extra_index_urls=[]))
        assert "--extra-index-url" not in rendered
        assert rendered.splitlines()[0] == "torch==2.6.0+cu124"

    def test_it_ends_with_a_newline(self) -> None:
        assert render_requirements(_spec()).endswith("\n")


class TestDefinition:
    """The rules the contract validates must reach the emitted text."""

    def test_the_env_prefix_is_where_the_venv_is_created(self) -> None:
        assert "    python -m venv /opt/env" in render_definition(_spec()).splitlines()

    def test_a_different_prefix_moves_every_reference(self) -> None:
        rendered = render_definition(_spec(env_prefix="/srv/research"))
        assert "/opt/env" not in rendered
        assert "    python -m venv /srv/research" in rendered.splitlines()
        assert "    export PATH=/srv/research/bin:$PATH" in rendered.splitlines()

    def test_first_party_wheels_install_without_resolving_dependencies(self) -> None:
        """--no-deps is what keeps the pinned set pinned."""
        line = f"    /opt/env/bin/pip install --no-cache-dir --no-deps {WHEEL_DIR}/*.whl"
        assert line in render_definition(_spec()).splitlines()

    def test_the_third_party_layer_installs_from_the_rendered_file(self) -> None:
        line = f"    /opt/env/bin/pip install --no-cache-dir -r {SPEC_DIR}/{REQUIREMENTS_NAME}"
        assert line in render_definition(_spec()).splitlines()

    def test_the_selfcheck_runs_inside_post(self) -> None:
        rendered = render_definition(_spec())
        lines = rendered.splitlines()
        post = lines.index("%post")
        check = lines.index(f"    /opt/env/bin/python {SPEC_DIR}/{SELFCHECK_NAME}")
        environment = lines.index("%environment")
        assert post < check < environment

    def test_the_selfcheck_runs_after_every_install(self) -> None:
        """Checking before installing would pass on an empty image."""
        lines = render_definition(_spec()).splitlines()
        wheels = lines.index(
            f"    /opt/env/bin/pip install --no-cache-dir --no-deps {WHEEL_DIR}/*.whl"
        )
        check = lines.index(f"    /opt/env/bin/python {SPEC_DIR}/{SELFCHECK_NAME}")
        assert wheels < check

    def test_the_commit_is_copied_into_the_environment(self) -> None:
        line = f"    cp {SPEC_DIR}/{COMMIT_NAME} /opt/env/{COMMIT_NAME}"
        assert line in render_definition(_spec()).splitlines()

    def test_the_spec_and_environment_are_made_world_readable(self) -> None:
        """Otherwise the image's own check cannot be re-run.

        ``%files`` preserves host permissions and ``%post`` runs as root, so a
        file staged mode 640 lands root-owned. The first real build produced
        ``/opt/spec/selfcheck.py`` as ``nobody nogroup`` 640 and re-running the
        verification failed with EACCES, while the environment imported fine.
        """
        line = f"    chmod -R a+rX {SPEC_DIR} /opt/env"
        assert line in render_definition(_spec()).splitlines()

    def test_permissions_are_fixed_before_the_selfcheck_runs(self) -> None:
        lines = render_definition(_spec()).splitlines()
        chmod = lines.index(f"    chmod -R a+rX {SPEC_DIR} /opt/env")
        check = lines.index(f"    /opt/env/bin/python {SPEC_DIR}/{SELFCHECK_NAME}")
        assert chmod < check

    def test_the_chmod_follows_the_env_prefix(self) -> None:
        lines = render_definition(_spec(env_prefix="/srv/research")).splitlines()
        assert f"    chmod -R a+rX {SPEC_DIR} /srv/research" in lines

    def test_the_base_image_is_the_bootstrap_source(self) -> None:
        assert "From: python:3.11.16-slim-bookworm" in render_definition(_spec()).splitlines()

    def test_labels_are_emitted(self) -> None:
        assert "    org.corvis.captured 2026-08-25" in render_definition(_spec()).splitlines()

    def test_no_labels_leaves_the_section_present_but_bare(self) -> None:
        rendered = render_definition(_spec(labels={}))
        assert "%labels" in rendered.splitlines()
        assert "org.corvis" not in rendered

    def test_post_sets_strict_shell_options(self) -> None:
        assert "    set -eu" in render_definition(_spec()).splitlines()

    def test_it_carries_no_carriage_returns(self) -> None:
        assert "\r" not in render_definition(_spec())

    def test_it_ends_with_a_newline(self) -> None:
        assert render_definition(_spec()).endswith("\n")


class TestAptRunsWithoutDroppingPrivileges:
    """The first image to declare an OS layer failed on this and nothing else.

    Apt re-executes its download method as the unprivileged ``_apt`` user.
    An unprivileged apptainer build has no uid to drop to -- HPC3 lists no
    subuid mapping for the user and ships no fakeroot -- so ``seteuid(42)``
    fails and the fetch dies:

        E: setgroups 65534 failed - setgroups (1: Operation not permitted)
        E: Method http has died unexpectedly!
        E: Sub-process http returned an error code (112)

    Measured as job 55662349 on 2026-08-30, which reached the ``%post``
    section and got no further.
    """

    def test_every_apt_call_disables_the_sandbox(self) -> None:
        rendered = render_definition(_spec(system_packages=["xvfb=2:21.1.7-3+deb12u13"]))
        calls = [line for line in rendered.splitlines() if "apt-get" in line]
        assert calls != []
        for call in calls:
            assert "-o APT::Sandbox::User=root" in call, call

    def test_both_the_update_and_the_install_carry_it(self) -> None:
        """One without the other still dies: the update fetches too."""
        rendered = render_definition(_spec(system_packages=["xvfb=2:21.1.7-3+deb12u13"]))
        assert "apt-get -o APT::Sandbox::User=root update" in rendered
        assert "apt-get -o APT::Sandbox::User=root install -y" in rendered

    def test_an_image_with_no_os_layer_runs_no_apt_at_all(self) -> None:
        """The option is on the calls, not on the image: a wheels-only image
        renders no apt line to carry it."""
        rendered = render_definition(_spec(system_packages=[]))
        assert "apt-get" not in rendered

    def test_nothing_is_written_into_the_built_image(self) -> None:
        """Set as an option rather than into /etc/apt/apt.conf.d, so the
        image carries no configuration that outlives its own build."""
        rendered = render_definition(_spec(system_packages=["xvfb=2:21.1.7-3+deb12u13"]))
        assert "apt.conf" not in rendered
