"""Freeze the existing candidate matrix under the audited accounting amendment."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, require_dict, require_list, require_str
from platform_core.load_report import post_load_report
from covenant_ml.markets import _test_hooks
from covenant_ml.markets.research_cluster import decode_research_cluster, prepare_bundle
from covenant_ml.markets.research_journal import admit_proposal, research_journal, research_policy_digest
from covenant_ml.markets.research_preparation_report import report_preparation
from covenant_ml.markets.research_records import candidate_digest, decode_research_policy, decode_research_proposal
from covenant_ml.markets.research_trial_records import decode_research_evaluation, decode_research_trial, evaluation_digest
from covenant_ml.markets.validation import read_document

root = Path(__file__).parent
previous = root.parent / 'licensed-v2-09113184'
task = 'f61db3a7-a9dd-468c-96d2-d520271a9c3b'
old_journal = previous / 'attempts.sqlite'
old_digest = sha256(old_journal.read_bytes()).hexdigest()
assert old_digest == '82450a7994d16019b9f699104b5e8891d943ad426becd93c5fb56402b0854a40'


def write_document(path: Path, document: JSONObject) -> None:
    """Write one frozen metadata document exclusively, refusing replacement."""
    with path.open('x', encoding='utf-8') as stream:
        stream.write(dump_json_str(document))


old_policy = read_document(previous / 'policy.json')
evaluation_document = read_document(previous / 'evaluation.json')
old_design = require_dict(evaluation_document, 'design')
design = read_document(root / 'controller/crsp-monthly.json')
assert {key for key in design if design[key] != old_design.get(key)} == {'version', 'held_return_rule'}
evaluation_document['design'] = design
evaluation = decode_research_evaluation(evaluation_document)
protocol_digest = evaluation_digest(evaluation)
workspace = read_document(root / 'hpc3.json')
image = require_dict(require_dict(require_dict(workspace, 'projects'), 'cleargbm'), 'image')
stage = read_document(root / 'validation-stage.json')
bundle = require_str(stage, 'destination') + '/bundle.json'
policy_document = dict(old_policy)
policy_document.update({
    'protocol_digest': protocol_digest,
    'dataset_digest': 'd8e66902e452c8f0e8790d3d17e31c648b684fbfa2606bea7de363889f94fa6d',
    'image_digest': require_str(image, 'sha256'),
})
policy = decode_research_policy(policy_document)
assert policy['selection'] == decode_research_policy(old_policy)['selection']
assert (policy['cpus'], policy['mem_gb'], policy['max_seconds']) == (8, 128, 7200)
cluster_document = read_document(previous / 'cluster.json')
old_cluster = decode_research_cluster(cluster_document)
old_checks = dict(line.split('  ', 1)[::-1] for line in old_cluster['checksums'].splitlines())
checks = [f"{image['sha256']}  {image['path']}"]
for raw in require_list(stage, 'files'):
    file = require_dict({'file': raw}, 'file')
    checks.append(f"{file['sha256']}  {stage['destination']}/{file['name']}")
for name in ('__init__.py', 'research_trial.py'):
    remote = old_cluster['entrypoint'] + '/scripts/' + name
    source = root.parent / 'protocol-v3-rehearsal-284bc096/source/libs/covenant_ml/scripts' / name
    assert sha256(source.read_bytes()).hexdigest() == old_checks[remote]
    checks.append(f'{old_checks[remote]}  {remote}')
checks.append(f"{old_cluster['access_digest']}  {old_cluster['access']}")
cluster_document.update({
    'python': '/home/corvis/research-control/wrds-protocol-v3-087f67b4c/bin/python',
    'workspace': str(root / 'hpc3.json'),
    'root': '/data/homezvol1/wagnera3/research/licensed-v3-f61db3a7/attempts',
    'image': require_str(image, 'path'), 'checksums': '\n'.join(checks) + '\n',
    'report_task': task,
})
cluster = decode_research_cluster(cluster_document)
write_document(root / 'policy.json', policy_document)
write_document(root / 'evaluation.json', evaluation_document)
write_document(root / 'cluster.json', cluster_document)
ledger = root / 'attempts.sqlite'
research_journal(policy).initialize(ledger)
matrix: list[JSONValue] = []
for number in range(1, 16):
    proposal_document = read_document(previous / f'proposal-{number}.json')
    original_candidate = candidate_digest(decode_research_proposal(proposal_document))
    proposal_document['protocol_digest'] = protocol_digest
    proposal = decode_research_proposal(proposal_document)
    assert candidate_digest(proposal) == original_candidate
    admitted = admit_proposal(ledger, policy, proposal_document, int(datetime.now(UTC).timestamp() * 1000))
    assert admitted['number'] == number and admitted['status'] == 'admitted'
    trial_document: JSONObject = {
        'policy': policy_document, 'proposal': proposal_document, 'evaluation': evaluation_document,
        'number': number, 'manifest': bundle,
    }
    trial = decode_research_trial(trial_document)
    write_document(root / f'proposal-{number}.json', proposal_document)
    write_document(root / f'trial-{number}.json', trial_document)
    prepare_bundle(root / f'preparation-{number}', cluster, trial)
    matrix.append({'number': number, 'candidate': original_candidate, 'seed': proposal['seed'],
                   'request_sha256': sha256((root / f'trial-{number}.json').read_bytes()).hexdigest()})
assert sha256(old_journal.read_bytes()).hexdigest() == old_digest
prepared = report_preparation(root / 'policy.json', ledger, root / 'preparation-1/run.json', task)
write_document(root / 'preparation-report.json', prepared)
report: JSONObject = {
    'kind': 'LICENSED_V3_STUDY_FREEZE', 'observed_at': datetime.now(UTC).isoformat(),
    'reporter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
    'old_policy': old_policy, 'new_policy': policy_document,
    'policy_digest': research_policy_digest(policy), 'original_journal_sha256': old_digest,
    'original_journal_unchanged': True, 'candidate_count': 3, 'request_count': 15,
    'policy_fields_changed': [key for key in old_policy if old_policy[key] != policy_document[key]],
    'design_fields_changed': ['version', 'held_return_rule'],
    'selection_unchanged': True, 'estimator_unchanged': True, 'matrix': matrix,
    'native_evaluation_submitted': False, 'final_released': False,
}
write_document(root / 'freeze-report.json', report)
post_load_report(post=_test_hooks.http_post, read_text=_test_hooks.stack_text,
                 task_id=task, family='hpc',
                 body='LICENSED V3 STUDY FREEZE LOAD-REPORT\n' + dump_json_str(report))
