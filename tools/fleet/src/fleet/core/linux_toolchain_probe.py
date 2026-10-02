"""The ``sh`` toolchain probe: what a Linux node has installed.

Split out of :mod:`fleet.core.dialect_linux` when the ``gpu`` and ``testdb``
lines took that module past the 600-line ceiling (MCPs board task 939ec5c7),
as :mod:`fleet.core.windows_toolchain_probe` was split from its dialect. It
is the probe's body without the dialect's prologue, which
:data:`fleet.core.dialect_linux.TOOLCHAIN_PROBE_SCRIPT` prepends, the shape
:mod:`fleet.core.linux_capacity_probe` already has.
"""

from __future__ import annotations

from fleet.contracts.capability import STACK_IMAGES, STACK_NETWORK
from fleet.contracts.detection import TESTDB_CONTAINER
from fleet.contracts.toolchain import HOOKS_CHECK_MODULES, HOOKS_ROUTE_FILE

#: The toolchain probe's body, verbatim.
#:
#: THE INTERPRETER IS ``python3`` AND IS REPORTED AS ``python``. The required
#: tool is named ``python`` because that is what a Windows PATH calls it; on
#: Linux the Makefile prologue (``scripts/make/shell.mk``) calls ``python3``,
#: so that is what the probe asks for, and it reports the answer under the
#: name the contract requires so :func:`fleet.contracts.toolchain.python_is_right`
#: reads one report on every platform. The managers asked about are the two
#: that exist here: ``apt-get`` for git and make, ``pipx`` for poetry. ``node``
#: is asked and never installed here: Ubuntu 24.04's archive carries Node
#: 18.19.1 where the fleet runs 24, and diphtheria's 24.21.0 came from the
#: NodeSource repository (both read from ``apt-cache policy nodejs`` there,
#: 2026-09-23), so a Linux node's Node is installed by hand.
#:
#: THE ``docker`` LINE ASKS ONE DAEMON BY ITS SOCKET, never ``docker`` on the
#: runner's PATH, which is the stack's daemon on diphtheria. It is the
#: execdocker user's rootless daemon (MCPs board task 6c4516af), asked as
#: that user with ``sudo -n``, so a node without the user, without sudo for
#: it or whose daemon does not say ``name=rootless`` answers ``docker=no=``.
#: So does one whose user lacks the compose or buildx CLI plugin, since the
#: lane the tag routes runs ``docker compose`` and builds images: lavender-wsl
#: carried the tag with neither, and MCPs/execution-deploy job 7ad01613 died
#: there at ``docker: unknown command: docker compose`` (2026-09-29).
#: The ``sudo`` is an ``if`` condition because the prologue's ``set -e``
#: does not fire there: as a bare assignment, a ``sudo`` refusing for want
#: of a password ended the whole probe with exit 1 (measured as execdocker on
#: diphtheria, 2026-09-29, MCPs board task a8ee9b21), not the ``no`` line.
#:
#: THE ``stack`` LINE ASKS THE PATH's DAEMON, the one the stack suites'
#: ``docker run`` reaches, and answers its ServerVersion only when that
#: daemon holds :data:`fleet.contracts.capability.STACK_NETWORK` and every
#: one of the images in ``STACK_IMAGES`` (MCPs board task 554bffc1). Its
#: three asks are one ``if`` condition for the same ``set -e`` reason, so a
#: node with no docker, no network or one image missing answers
#: ``stack=no=``. Measured 2026-09-29: diphtheria holds all four on 29.8.1,
#: and lavender-wsl's daemon has no ``mcp-network``.
#:
#: THE ``gpu`` AND ``testdb`` LINES are what the runner claims those tags
#: with (MCPs board task 939ec5c7, :mod:`fleet.contracts.detection`). ``gpu``
#: is nvidia-smi's first device as ``<name>, <compute capability>``, and
#: ``no`` when nvidia-smi is absent, fails or lists none; ``testdb`` is the
#: image of the PATH daemon's ``corvis-fleet-testdb`` container, and ``no``
#: when there is no such container or no daemon. Both are ``if`` conditions
#: for the ``set -e`` reason above. Measured 2026-10-02: diphtheria answers
#: ``NVIDIA RTX A2000 12GB, 8.6`` and lavender-wsl ``NVIDIA GeForce GTX 1630,
#: 7.5``, and both run the container from ``pgvector/pgvector:pg16-bookworm``.
#:
#: THE ``hooks`` LINE is the MCPs claude-hooks check's environment (MCPs
#: board task ec895824): the route file under the account's home and one
#: ``python3 -c`` importing every module of
#: :data:`fleet.contracts.toolchain.HOOKS_CHECK_MODULES`, one ``if``
#: condition for the ``set -e`` reason above.
TOOLCHAIN_PROBE_BODY = (
    "report() {\n"
    '  if command -v "$2" > /dev/null 2>&1; then\n'
    '    printf \'%s=yes=%s\\n\' "$1" "$("$2" --version 2>&1 | head -n 1 | tr -d \'\\r\')"\n'
    "  else\n"
    "    printf '%s=no=\\n' \"$1\"\n"
    "  fi\n"
    "}\n"
    "report python python3\n"
    "report poetry poetry\n"
    "report git git\n"
    "report make make\n"
    "report node node\n"
    "report ffmpeg ffmpeg\n"
    "report tar tar\n"
    "report cargo cargo\n"
    "if command -v g++ > /dev/null 2>&1; then\n"
    "  printf 'cxx=yes=%s\\n' \"$(g++ -dumpfullversion)\"\n"
    "else\n"
    "  printf 'cxx=no=\\n'\n"
    "fi\n"
    "if id execdocker > /dev/null 2>&1 &&"
    ' d="$(sudo -n -u execdocker docker -H "unix:///run/user/$(id -u execdocker)/docker.sock"'
    " info --format '{{.ServerVersion}} {{json .SecurityOptions}}' 2>/dev/null)\"; then\n"
    '  case "$d" in\n'
    "    *name=rootless*)\n"
    "      if sudo -n -u execdocker docker compose version > /dev/null 2>&1 &&"
    " sudo -n -u execdocker docker buildx version > /dev/null 2>&1; then\n"
    "        printf 'docker=yes=%s\\n' \"${d%% *}\"\n"
    "      else\n"
    "        printf 'docker=no=\\n'\n"
    "      fi ;;\n"
    "    *) printf 'docker=no=\\n' ;;\n"
    "  esac\n"
    "else\n"
    "  printf 'docker=no=\\n'\n"
    "fi\n"
    f"if docker network inspect {STACK_NETWORK} > /dev/null 2>&1 &&"
    f" docker image inspect {' '.join(STACK_IMAGES)} > /dev/null 2>&1 &&"
    " v=\"$(docker version --format '{{.Server.Version}}' 2>/dev/null)\"; then\n"
    "  printf 'stack=yes=%s\\n' \"$v\"\n"
    "else\n"
    "  printf 'stack=no=\\n'\n"
    "fi\n"
    "if command -v nvidia-smi > /dev/null 2>&1 &&"
    ' g="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null'
    ' | head -n 1)" && [ -n "$g" ]; then\n'
    "  printf 'gpu=yes=%s\\n' \"$g\"\n"
    "else\n"
    "  printf 'gpu=no=\\n'\n"
    "fi\n"
    f"if t=\"$(docker container inspect --format '{{{{.Config.Image}}}}' {TESTDB_CONTAINER}"
    ' 2>/dev/null)"; then\n'
    "  printf 'testdb=yes=%s\\n' \"$t\"\n"
    "else\n"
    "  printf 'testdb=no=\\n'\n"
    "fi\n"
    f'if [ -f "$HOME/{"/".join(HOOKS_ROUTE_FILE)}" ] &&'
    f' python3 -c "import {", ".join(HOOKS_CHECK_MODULES)}" > /dev/null 2>&1; then\n'
    f"  printf 'hooks=yes=%s\\n' \"$HOME/{'/'.join(HOOKS_ROUTE_FILE)}\"\n"
    "else\n"
    "  printf 'hooks=no=\\n'\n"
    "fi\n"
    "report apt-get apt-get\n"
    "report pipx pipx\n"
)


__all__ = ["TOOLCHAIN_PROBE_BODY"]
