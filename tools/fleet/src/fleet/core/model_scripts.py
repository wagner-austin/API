"""The scripts a node runs to serve, stop and watch a catalog model.

MCPs board task b9235198. Each is rendered whole and sent to the node by
path (:mod:`fleet.core.remote`), every value in it single-quoted by
:func:`shlex.quote`, and each prints the lines :mod:`fleet.core.model_serve`
reads, one verdict word first, so nothing a node prints is guessed at.

THE SERVER IS A TRANSIENT USER UNIT, ``fleet-model-<id>.service``, started by
``systemd-run --user`` under the account's lingering manager, which is how
chat's own Glimmer runs on colossus (``chat-a3-glimmer``, a unit this
package never names and so never stops). It binds the node's tailnet
address alone, on the lowest free port of the node's declared range, with
``-ngl 999 -fit off`` so every layer goes on the GPU or the load fails,
never a partial offload ``--fit`` chose, and ``-lv 4`` so its log carries
llama.cpp's ``offloaded N/M layers to GPU`` line, which is how the load
knows every layer went there (measured on colossus 2026-11-02: at the
default verbosity the line is not written; one 200-token request and ten
polls add 3.5 KB to the log at ``-lv 4``).

ITS STATE IS ON THE NODE, in ``~/.fleet/models/<id>/``: the job that loaded
it, its port, its server log, the last use, and ``ended`` once it stopped
with no unload job. The last use copies chat's idle rule (chat
``src/chat/api/activity.py``, deleted with the Python app in 71edefa): a
use is a request a person made, never a probe, which here is llama.cpp's
prompt and generated token counters moving, or a request in flight, so the
``/health`` and ``/metrics`` reads the runner itself makes never count. It
is kept in ``/proc/uptime`` seconds, monotonic as chat's ledger was, and a
start counts as a use, so a server just loaded is not idle.
"""

from __future__ import annotations

import shlex
from typing import Final

from fleet.contracts.model_host import CatalogModel, NodeModels
from fleet.core.dialect import PROLOGUE

#: How long a started server has to answer ``/health`` before the load
#: stops it and fails: the 21.7 GB Ornith 35B took under a minute to load
#: on colossus (board task 62b1bb0a), so five is room, not a guess.
HEALTH_SECONDS: Final[int] = 300

#: Where a node keeps its served models' state, under the account's home.
STATE_ROOT: Final[str] = '"$HOME/.fleet/models"'

#: Clear a failed unit's record so its name can be started again; a unit
#: that is not failed is left alone, since ``reset-failed`` of one exits 1.
_RESET_FAILED: Final[str] = (
    'if systemctl --user is-failed --quiet "$unit"; then\n'
    '  systemctl --user reset-failed "$unit"\n'
    "fi\n"
)

#: Sum llama.cpp's prompt and generated token counters, and read the
#: requests in flight, from its ``/metrics`` text on standard input.
_COUNT_USE: Final[str] = (
    "awk '/^llamacpp:(prompt_tokens_total|tokens_predicted_total) / {s += $2}\n"
    "  /^llamacpp:requests_processing / {b = $2}\n"
    '  END {printf "%d %d", s, b}\''
)


def model_unit(model_id: str) -> str:
    """The transient user unit serving one catalog model.

    Args:
        model_id: The catalog id.

    Returns:
        ``fleet-model-<id>.service``.
    """
    return f"fleet-model-{model_id}.service"


def _state(model_id: str) -> str:
    """The shell expression for one model's state directory.

    Args:
        model_id: The catalog id, which the contract holds to letters,
            digits, '.' and '-'.

    Returns:
        A double-quoted expansion of it under :data:`STATE_ROOT`.
    """
    return f'"$HOME/.fleet/models/{model_id}"'


def _indented(text: str, spaces: int) -> str:
    """Indent every line of a fragment.

    Args:
        text: Lines, each ending in a newline.
        spaces: How far.

    Returns:
        The fragment, indented.
    """
    pad = " " * spaces
    return "".join(f"{pad}{line}\n" for line in text.splitlines())


def _check_file(model: CatalogModel) -> str:
    """The lines that refuse a file that is not the catalog's, before anything starts.

    Args:
        model: The catalog row.

    Returns:
        The fragment, printing ``absent``, ``size`` or ``sha256`` and
        exiting on the first difference.
    """
    q = shlex.quote
    return (
        'if [ ! -f "$file" ]; then\n'
        "  printf 'absent %s\\n' \"$file\"\n"
        "  exit 0\n"
        "fi\n"
        'size="$(stat -c %s "$file")"\n'
        f'if [ "$size" != {q(str(model["size_bytes"]))} ]; then\n'
        "  printf 'size %s\\n' \"$size\"\n"
        "  exit 0\n"
        "fi\n"
        'digest="$(sha256sum "$file" | cut -d \' \' -f 1)"\n'
        f'if [ "$digest" != {q(model["sha256"])} ]; then\n'
        "  printf 'sha256 %s\\n' \"$digest\"\n"
        "  exit 0\n"
        "fi\n"
    )


def _start(*, models: NodeModels, model: CatalogModel, job_id: str) -> str:
    """The lines that start the server on the lowest free port, unless one is active.

    Args:
        models: The node's declaration.
        model: The catalog row.
        job_id: The load, which names the unit's description.

    Returns:
        The fragment, setting ``port``, or printing ``ports`` and exiting.
    """
    q = shlex.quote
    first, last = models["first_port"], models["last_port"]
    description = q(f"catalog model {model['model_id']} for dispatch load {job_id}")
    return (
        'if systemctl --user is-active --quiet "$unit"; then\n'
        '  port="$(cat "$dir/port")"\n'
        "else\n"
        f"{_indented(_RESET_FAILED, 2)}"
        "  port=''\n"
        f"  for candidate in $(seq {first} {last}); do\n"
        "    if ! ss -ltnH | awk '{print $4}' | grep -q \":$candidate$\"; then\n"
        '      port="$candidate"\n'
        "      break\n"
        "    fi\n"
        "  done\n"
        '  if [ -z "$port" ]; then\n'
        f"    printf 'ports %s\\n' '{first}-{last}'\n"
        "    exit 0\n"
        "  fi\n"
        '  printf \'%s\\n\' "$port" > "$dir/port"\n'
        '  rm -f "$dir/ended"\n'
        '  printf \'%s 0\\n\' "$(cut -d . -f 1 /proc/uptime)" > "$dir/use"\n'
        f'  systemd-run --user --quiet --unit="$unit" --description={description} \\\n'
        '    -p StandardOutput=truncate:"$dir/server.log" -p StandardError=inherit -- \\\n'
        f'    {q(models["server"])} -m "$file" --host "$addr" --port "$port" \\\n'
        f"    -c {model['context_tokens']} -ngl 999 -fit off -fa on --jinja --metrics -lv 4\n"
        "fi\n"
    )


def load_script(*, models: NodeModels, model: CatalogModel, job_id: str) -> str:
    """Check the model's file, serve it, and wait for it to answer.

    Prints exactly one line: ``absent <path>``, ``size <bytes>`` or
    ``sha256 <digest>`` when the file is not the catalog's, before anything
    starts; ``ports <first>-<last>`` when none is free; ``exited <status>
    <last log line>`` when the server stopped while loading; ``unhealthy
    <seconds>`` once it was stopped for not answering; else ``loaded
    <address> <port> <offloaded>``, the last ``N/M`` of the log's offload
    line or ``none``. A unit already active for the model is adopted, its
    port kept, so a load claimed again after its runner restarted serves
    the same server.

    Args:
        models: The node's declaration.
        model: The catalog row.
        job_id: The load, written as the model's job.

    Returns:
        The script.
    """
    q = shlex.quote
    return (
        f"{PROLOGUE}"
        f"unit={q(model_unit(model['model_id']))}\n"
        f"dir={_state(model['model_id'])}\n"
        f"file={q(models['cache'] + '/' + model['file'])}\n"
        f"{_check_file(model)}"
        'addr="$(tailscale ip -4 | head -n 1)"\n'
        'mkdir -p "$dir"\n'
        f"{_start(models=models, model=model, job_id=job_id)}"
        f"printf '%s\\n' {q(job_id)} > \"$dir/job\"\n"
        "waited=0\n"
        'until curl -fsS -m 2 "http://$addr:$port/health" > /dev/null 2>&1; do\n'
        '  if ! systemctl --user is-active --quiet "$unit"; then\n'
        '    status="$(systemctl --user show "$unit" -p ExecMainStatus --value)"\n'
        '    printf \'exited %s %s\\n\' "$status" "$(tail -n 1 "$dir/server.log")"\n'
        "    exit 0\n"
        "  fi\n"
        f'  if [ "$waited" -ge {HEALTH_SECONDS} ]; then\n'
        '    systemctl --user stop "$unit"\n'
        "    printf 'unhealthy %s\\n' \"$waited\"\n"
        "    exit 0\n"
        "  fi\n"
        "  sleep 1\n"
        "  waited=$((waited + 1))\n"
        "done\n"
        'offload="$(grep -o \'offloaded [0-9]*/[0-9]* layers to GPU\' "$dir/server.log" \\\n'
        "  | tail -n 1 | cut -d ' ' -f 2)\"\n"
        'if [ -z "$offload" ]; then\n'
        "  offload=none\n"
        "fi\n"
        'printf \'loaded %s %s %s\\n\' "$addr" "$port" "$offload"\n'
    )


def unload_script(model_id: str) -> str:
    """Stop the model's server and forget its state.

    Prints ``stopped <job>``, naming the load that started it, or
    ``absent`` when no server of it was active.

    Args:
        model_id: The catalog id.

    Returns:
        The script.
    """
    return (
        f"{PROLOGUE}"
        f"unit={shlex.quote(model_unit(model_id))}\n"
        f"dir={_state(model_id)}\n"
        'if systemctl --user is-active --quiet "$unit"; then\n'
        '  job="$(cat "$dir/job")"\n'
        '  systemctl --user stop "$unit"\n'
        '  rm -rf "$dir"\n'
        "  printf 'stopped %s\\n' \"$job\"\n"
        "else\n"
        f"{_indented(_RESET_FAILED, 2)}"
        '  rm -rf "$dir"\n'
        "  printf 'absent\\n'\n"
        "fi\n"
    )


def scan_script(models: NodeModels) -> str:
    """Read every served model, stop each idle one, and say which have ended.

    Prints one line per model with state on the node: ``serving <id> <job>
    <port> <idle seconds>``, or ``ended <id> <job> <why>`` for one whose
    server exited or which this scan stopped for idleness. An ended model
    stays ``ended`` until :func:`forget_script` removes it, so an end the
    queue did not hear is said again at the next scan. A server whose
    ``/metrics`` does not answer gets no use counted.

    Args:
        models: The node's declaration, whose idle span applies.

    Returns:
        The script.
    """
    idle = models["idle_seconds"]
    return (
        f"{PROLOGUE}"
        f"root={STATE_ROOT}\n"
        'if [ ! -d "$root" ]; then\n'
        "  exit 0\n"
        "fi\n"
        'now="$(cut -d . -f 1 /proc/uptime)"\n'
        'addr="$(tailscale ip -4 | head -n 1)"\n'
        'for dir in "$root"/*; do\n'
        '  if [ ! -d "$dir" ]; then\n'
        "    continue\n"
        "  fi\n"
        '  id="$(basename "$dir")"\n'
        '  unit="fleet-model-$id.service"\n'
        '  job="$(cat "$dir/job")"\n'
        '  if [ ! -f "$dir/ended" ] && ! systemctl --user is-active --quiet "$unit"; then\n'
        '    status="$(systemctl --user show "$unit" -p ExecMainStatus --value)"\n'
        '    result="$(systemctl --user show "$unit" -p Result --value)"\n'
        "    printf 'llama-server exited with status %s, result %s, with no unload job\\n' \\\n"
        '      "$status" "$result" > "$dir/ended"\n'
        "  fi\n"
        '  if [ -f "$dir/ended" ]; then\n'
        '    printf \'ended %s %s %s\\n\' "$id" "$job" "$(cat "$dir/ended")"\n'
        "    continue\n"
        "  fi\n"
        '  port="$(cat "$dir/port")"\n'
        '  read -r last seen < "$dir/use"\n'
        '  if metrics="$(curl -fsS -m 5 "http://$addr:$port/metrics")"; then\n'
        f'    counted="$(printf \'%s\\n\' "$metrics" | {_COUNT_USE})"\n'
        '    total="${counted% *}"\n'
        '    busy="${counted#* }"\n'
        '    if [ "$total" != "$seen" ] || [ "$busy" != 0 ]; then\n'
        '      last="$now"\n'
        '      printf \'%s %s\\n\' "$now" "$total" > "$dir/use"\n'
        "    fi\n"
        "  fi\n"
        "  idle=$((now - last))\n"
        f'  if [ "$idle" -ge {idle} ]; then\n'
        '    systemctl --user stop "$unit"\n'
        f"    printf 'idle %s s with no request, past its {idle} s; llama-server stopped' \\\n"
        '      "$idle" > "$dir/ended"\n'
        '    printf \'ended %s %s %s\\n\' "$id" "$job" "$(cat "$dir/ended")"\n'
        "  else\n"
        '    printf \'serving %s %s %s %s\\n\' "$id" "$job" "$port" "$idle"\n'
        "  fi\n"
        "done\n"
    )


def forget_script(ended: tuple[tuple[str, str], ...]) -> str:
    """Remove the state of models whose end the queue has recorded.

    A directory is removed only while it still names the job that ended,
    so a load of the model that began since keeps its own.

    Args:
        ended: ``(model id, job id)`` per model.

    Returns:
        The script, printing ``forgot <id>`` per directory it removed.
    """
    q = shlex.quote
    lines = [PROLOGUE]
    for model_id, job_id in ended:
        lines.append(
            f"dir={_state(model_id)}\n"
            f'if [ -f "$dir/job" ] && [ "$(cat "$dir/job")" = {q(job_id)} ]; then\n'
            f"  unit={q(model_unit(model_id))}\n"
            f"{_indented(_RESET_FAILED, 2)}"
            '  rm -rf "$dir"\n'
            f"  printf 'forgot %s\\n' {q(model_id)}\n"
            "fi\n"
        )
    return "".join(lines)


__all__ = [
    "HEALTH_SECONDS",
    "STATE_ROOT",
    "forget_script",
    "load_script",
    "model_unit",
    "scan_script",
    "unload_script",
]
