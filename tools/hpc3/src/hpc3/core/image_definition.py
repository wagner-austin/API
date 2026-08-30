"""Rendering the Apptainer definition and the pinned requirements beside it.

Two decisions in the emitted text are load-bearing and neither is obvious from
reading it.

The virtualenv is created at the spec's ``env_prefix``, which the image
contract has already refused to place under a bind-mounted root. HPC3 mounts
``/pub`` into every container, so an environment built there would be
replaced at runtime by the host directory and the image's own interpreter
would cease to exist inside its own image.

The first-party wheels install with ``--no-deps``. Every dependency is pinned
in the requirements file above them, and letting a wheel resolve its own would
float a version off the captured set -- which is the drift the image exists to
remove, arriving through the one step that looked harmless.
"""

from __future__ import annotations

from hpc3.contracts.image_spec import ImageSpec
from hpc3.core.image_layout import (
    COMMIT_NAME,
    REQUIREMENTS_NAME,
    SELFCHECK_NAME,
    SPEC_DIR,
    WHEEL_DIR,
)

#: How every apt call in a rendered definition is spelled.
#:
#: ``APT::Sandbox::User=root`` is not a convenience. Apt normally re-executes
#: its download method as the unprivileged ``_apt`` user, which an
#: unprivileged container build cannot permit -- there is no uid to drop to
#: inside a root-mapped namespace, so the call fails and the fetch dies rather
#: than degrading. Set as an option rather than written into
#: ``/etc/apt/apt.conf.d`` so the built image carries no configuration that
#: outlives the build.
_APT = "apt-get -o APT::Sandbox::User=root"


def render_requirements(spec: ImageSpec) -> str:
    """Render the pinned third-party layer as a pip requirements file.

    Args:
        spec: The image being built.

    Returns:
        The file text, LF-terminated. Extra index URLs come first because pip
        reads them as options applying to the whole file, and a local version
        such as ``torch==2.6.0+cu124`` resolves only once its index is known.
    """
    lines = [f"--extra-index-url {url}" for url in spec["extra_index_urls"]]
    if lines:
        lines.append("")
    lines.extend(spec["requirements"])
    return "\n".join(lines) + "\n"


def _render_system_packages(packages: list[str]) -> list[str]:
    """Render the operating-system layer, or nothing when there is none.

    Args:
        packages: Exactly pinned package specifications.

    Returns:
        The lines installing them, or an empty list. An image whose whole
        dependency set is wheels renders no apt call at all rather than an
        install of nothing -- ``apt-get update`` alone costs a minute of
        build time and a network dependency for no result.

        ``--no-install-recommends`` because a recommended package is by
        definition one nothing declared, and the spec is meant to be the
        whole list. The lists are removed afterwards so the layer does not
        carry a package index nothing will read.
    """
    if not packages:
        return []
    return [
        "    # Operating-system layer. Some dependencies are not wheels: a JVM,",
        "    # an X server and a software GL stack cannot be pip-installed.",
        "    export DEBIAN_FRONTEND=noninteractive",
        "",
        "    # APT drops privileges to the '_apt' user before fetching, and in",
        "    # an unprivileged apptainer build that is not allowed: HPC3 lists",
        "    # no subuid mapping for the user and ships no fakeroot, so the",
        "    # build runs in a root-mapped namespace where seteuid(42) fails",
        "    # and apt's http method dies with 'Sub-process http returned an",
        "    # error code (112)'. Measured on the first image to declare an OS",
        "    # layer at all (job 55662349, 2026-08-30). Telling apt not to drop",
        "    # privileges is the whole fix; as real root it is a no-op, so one",
        "    # rendering serves a privileged build and an unprivileged one.",
        f"    {_APT} update",
        f"    {_APT} install -y --no-install-recommends \\",
        *[f"        {package} \\" for package in packages[:-1]],
        f"        {packages[-1]}",
        "    rm -rf /var/lib/apt/lists/*",
        "",
    ]


def _render_post(spec: ImageSpec, env: str) -> list[str]:
    """Render the ``%post`` section.

    Args:
        spec: The image being built.
        env: Absolute container path receiving the virtualenv.

    Returns:
        The section's lines, including its header. The operating-system layer
        comes first because the interpreter the virtualenv is built from may
        itself be one of its packages.
    """
    return [
        "%post",
        "    set -eu",
        "",
        *_render_system_packages(spec["system_packages"]),
        f"    python -m venv {env}",
        f"    {env}/bin/pip install --no-cache-dir --upgrade pip",
        "",
        "    # Third-party layer, every version pinned by the spec.",
        f"    {env}/bin/pip install --no-cache-dir -r {SPEC_DIR}/{REQUIREMENTS_NAME}",
        "",
        "    # First-party layer. --no-deps because the pins above ARE the",
        "    # captured environment; resolving again would silently move one.",
        f"    {env}/bin/pip install --no-cache-dir --no-deps {WHEEL_DIR}/*.whl",
        "",
        "    # Provenance the trainer stamps into every manifest it writes.",
        f"    cp {SPEC_DIR}/{COMMIT_NAME} {env}/{COMMIT_NAME}",
        "",
        "    # %files preserves the HOST's permissions and %post runs as root,",
        "    # so a file staged mode 640 lands root-owned and unreadable by the",
        "    # unprivileged user who later runs the image. Measured: the first",
        "    # build produced /opt/spec/selfcheck.py as 'nobody nogroup' 640,",
        "    # and re-running the verification failed with EACCES while the",
        "    # environment itself imported fine. An image whose own check",
        "    # cannot be re-run is an image nobody can re-verify, so the modes",
        "    # are set here rather than left to whatever umask staged them.",
        f"    chmod -R a+rX {SPEC_DIR} {env}",
        "",
        "    # Fail the build, not the first job to use the image.",
        f"    {env}/bin/python {SPEC_DIR}/{SELFCHECK_NAME}",
    ]


def render_definition(spec: ImageSpec) -> str:
    """Render the Apptainer definition file.

    Args:
        spec: The image being built.

    Returns:
        The definition text, LF-terminated.
    """
    env = spec["env_prefix"]
    lines = [
        "Bootstrap: docker",
        f"From: {spec['base_image']}",
        "",
        "# Generated by hpc3.core.image_definition.render_definition. Do not",
        "# hand-edit: the self-check shipped alongside is rendered from the same",
        "# spec, and editing one reintroduces the drift an image removes.",
        "",
        "%files",
        f"    {REQUIREMENTS_NAME} {SPEC_DIR}/{REQUIREMENTS_NAME}",
        f"    {SELFCHECK_NAME} {SPEC_DIR}/{SELFCHECK_NAME}",
        f"    {COMMIT_NAME} {SPEC_DIR}/{COMMIT_NAME}",
        f"    wheels {WHEEL_DIR}",
        "",
        *_render_post(spec, env),
        "",
        "%environment",
        f"    export PATH={env}/bin:$PATH",
        "    export PYTHONDONTWRITEBYTECODE=1",
        "",
        "%labels",
        *(f"    {name} {value}" for name, value in spec["labels"].items()),
        "",
        "%help",
        "    Rendered from an hpc3 image spec. Corpora and artifacts are",
        "    bind-mounted, never baked in; their identity is carried by digest",
        "    verification at staging time.",
        "",
        f"    Self-check:  apptainer exec <image> {env}/bin/python {SPEC_DIR}/{SELFCHECK_NAME}",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["render_definition", "render_requirements"]
