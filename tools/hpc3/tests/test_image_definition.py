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
        "extra_index_urls": ["https://download.pytorch.org/whl/cu124"],
        "requirements": ["torch==2.6.0+cu124", "transformers==4.46.3"],
        "wheels": ["model_trainer_server-0.1.0-py3-none-any.whl"],
        "expected_versions": {"torch": "2.6.0+cu124"},
        "required_symbols": [{"module": "model_trainer", "attribute": "__file__"}],
        "labels": {"org.corvis.captured": "2026-08-25"},
    }
    base.update(overrides)
    return decode_image_spec(base)


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
