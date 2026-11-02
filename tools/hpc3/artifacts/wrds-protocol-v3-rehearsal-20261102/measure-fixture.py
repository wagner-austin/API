"""Regenerate the exact synthetic fixture and bind its encrypted preparation."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, require_dict, require_str
from platform_core.load_report import post_load_report
from covenant_ml.markets import _test_hooks
from covenant_ml.markets.encrypted_storage import verify_encrypted_bundle
from covenant_ml.markets.research_preparation_report import report_preparation
from covenant_ml.markets.validation import read_document
from tests.markets._crsp_fixtures import write_sources
from tests.markets._crsp_study_fixtures import missing_price_rows, study_protocol

root = Path(__file__).parent
generated = root / 'fixture-regenerated'
generated.mkdir(mode=0o700)
rows = missing_price_rows()
manifest_path, manifest = write_sources(generated, rows, study_protocol())
bundle_path = root / 'validation/bundle.json'
bundle_digest = sha256(bundle_path.read_bytes()).hexdigest()
bundle = verify_encrypted_bundle(bundle_path, bundle_digest)
plain_files: list[JSONValue] = []
for item in bundle['files']:
    expected = item['plaintext']
    name = expected['name']
    content = (generated / name).read_bytes()
    assert content == (root / 'synthetic-source' / name).read_bytes()
    assert sha256(content).hexdigest() == expected['sha256']
    assert len(content) == expected['size_bytes']
    plain_files.append(dict(expected))
assert sha256(manifest_path.read_bytes()).hexdigest() == bundle['manifest']['sha256']
spec = read_document(root / 'queue-spec.json')
for key in ('script', 'policy', 'trial', 'cluster'):
    item = require_dict(spec, key)
    assert sha256(Path(require_str(item, 'path')).read_bytes()).hexdigest() == require_str(item, 'sha256')
assert require_str(spec, 'python') == require_str(read_document(root / 'cluster.json'), 'python')
prepared = report_preparation(root / 'policy.json', root / 'attempts.sqlite',
                              root / 'recipe-preflight/run.json',
                              '284bc096-4b05-446b-9463-5ef5c49fa520')
(root / 'preparation-report.json').write_text(dump_json_str(prepared))
report: JSONObject = {
    'kind': 'SYNTHETIC_V3_FIXTURE_AND_QUEUE_EVIDENCE',
    'observed_at': datetime.now(UTC).isoformat(),
    'reporter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_commit': '087f67b4c4aae9a731c913330840dcc665b51701',
    'fixture': 'tests.markets._crsp_study_fixtures.missing_price_rows',
    'source_files': {str(path.relative_to(root / 'source')): sha256(path.read_bytes()).hexdigest()
                     for path in sorted((root / 'source').rglob('*')) if path.is_file()},
    'row_counts': {name: len(values) for name, values in rows.items()},
    'entry_delisting': rows['delists'],
    'missing_price_case': [row for row in rows['returns'] if row['permno'] == 7
                           and row['dlycaldt'] in ('2019-03-11', '2019-03-12')],
    'plaintext_files': plain_files,
    'bundle_sha256': bundle_digest,
    'validation_stage': read_document(root / 'validation-stage.json'),
    'script_stage': read_document(root / 'scripts-stage.json'),
    'access_stage': read_document(root / 'access-stage.json'),
    'queue_spec_sha256': sha256((root / 'queue-spec.json').read_bytes()).hexdigest(),
    'queue_spec': spec,
    'scope': 'Regenerated synthetic input matches every encrypted bundle plaintext identity. Ciphertext and queue inputs verified without decrypting or reading keys; one admitted attempt, no native submission or final release.',
}
payload = dump_json_str(report)
(root / 'fixture-provenance.json').write_text(payload)
post_load_report(post=_test_hooks.http_post, read_text=_test_hooks.stack_text,
                 task_id='284bc096-4b05-446b-9463-5ef5c49fa520', family='hpc',
                 body='SYNTHETIC V3 FIXTURE AND QUEUE LOAD-REPORT\n' + payload)
