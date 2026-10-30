"""The installed proposer path, run for real against an evaluator's private files.

The evaluator's CLI resolves a pinned specification, runs the proposal program on
the node's rootless daemon and hands the output to an evaluator controller. The
program probes every evaluator asset by its real host path: the policy, the
journal, the final-period extract, transport credentials, the daemon socket,
the controller's code, and the network. Each is absent or refused inside the
process, and every host file keeps its exact bytes.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from fleet.cli import research_propose
from tests.test_research_proposer_host import frozen_runtime

pytestmark = pytest.mark.host_linux_docker

__all__ = ["frozen_runtime"]

PROBE_PROGRAM = """
import json, os, pathlib, socket, sys
context = json.loads(sys.argv[1])
read = {}
for name in context['private']:
    try:
        pathlib.Path(name).read_bytes()
    except OSError as error:
        read[name] = error.errno
    else:
        read[name] = 0
write = {}
for name in context['evaluator']:
    try:
        pathlib.Path(name).write_text('replaced by the proposer')
    except OSError as error:
        write[name] = error.errno
    else:
        write[name] = 0
try:
    socket.create_connection(('1.1.1.1', 443), timeout=0.2)
except OSError as error:
    network = error.errno
else:
    network = 0
print(json.dumps({
    'read': read, 'write': write, 'network': network,
    'environment': sorted(os.environ), 'uid': os.getuid(),
    'proposal': context['proposal'],
}))
"""

CONTROLLER = """
import json, pathlib, sys
proposal = pathlib.Path(sys.argv[sys.argv.index('--proposal') + 1])
pathlib.Path(sys.argv[sys.argv.index('--ledger') + 1]).with_suffix('.received').write_bytes(
    proposal.read_bytes()
)
row = {'number': 1, 'status': 'refused', 'reason': 'not_admitted', 'candidate': ''}
print(json.dumps({'attempts': [row]}))
raise SystemExit(2)
"""


def _pinned(path: pathlib.Path, text: str) -> JSONObject:
    """Write one evaluator file and pin its exact bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def test_installed_proposer_cannot_reach_evaluator_assets(
    tmp_path: pathlib.Path, runtime: tuple[str, str]
) -> None:
    socket, image = runtime
    evaluator = tmp_path / "evaluator"
    policy = evaluator / "policy.json"
    ledger = evaluator / "journal.sqlite"
    final = evaluator / "final" / "returns.csv"
    credentials = evaluator / ".pgpass"
    controller = evaluator / "research_proposals.py"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("journal sentinel")
    final.parent.mkdir()
    final.write_text("permno,dlycaldt,dlyret\n1,2022-01-03,0.5\n")
    credentials.write_text("wrds-pgdata.wharton.upenn.edu:9737:wrds:evaluator:secret\n")
    private = (
        str(policy),
        str(ledger),
        str(final),
        str(credentials),
        socket,
        str(pathlib.Path.home() / ".ssh"),
        "/proc/1/root" + str(final),
    )
    targets = (str(controller), str(policy))
    proposal: JSONObject = {"partition": "validation", "seed": 11}
    probed: list[JSONValue] = [*private]
    replaced: list[JSONValue] = [*targets]
    context: JSONObject = {"private": probed, "evaluator": replaced, "proposal": proposal}
    document: JSONObject = {
        "python": sys.executable,
        "script": _pinned(controller, CONTROLLER),
        "policy": _pinned(policy, '{"max_candidates":2}'),
        "ledger": str(ledger),
        "socket": socket,
        "image": image,
        "program": _pinned(evaluator / "program.py", PROBE_PROGRAM),
        "context": _pinned(evaluator / "context.json", dump_json_str(context)),
        "timeout": 15,
    }
    payload = dump_json_str(document).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / digest).mkdir()
    (tmp_path / digest / "proposal.json").write_bytes(payload)
    files = [policy, ledger, final, credentials, controller]
    before = [path.read_bytes() for path in files]
    original = config_test_hooks.get_environment
    environment = original()
    config_test_hooks.get_environment = lambda: {**environment, "PGPASSWORD": "canary"}
    try:
        assert research_propose.main(["--root", str(tmp_path), "--digest", digest]) == 0
    finally:
        config_test_hooks.get_environment = original
    received = load_json_str(ledger.with_suffix(".received").read_text())
    assert received == {
        "read": dict.fromkeys(private, 2),
        "write": dict.fromkeys(targets, 2),
        "network": 101,
        "environment": ["LC_CTYPE", "PATH"],
        "uid": 65534,
        "proposal": proposal,
    }
    assert [path.read_bytes() for path in files] == before
    transcript = (tmp_path / digest / "admission.jsonl").read_text().splitlines()
    assert [narrow_json_to_dict(load_json_str(line))["exit"] for line in transcript] == [2]
