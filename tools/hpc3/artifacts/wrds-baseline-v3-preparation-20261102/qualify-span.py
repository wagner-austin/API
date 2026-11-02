"""Measure the previously failed training span with the frozen accounting owner."""

import csv
import io
import os
import sys
from hashlib import sha256
from math import isclose
from pathlib import Path

from platform_core.determinism_cpu import pin_single_thread

determinism = pin_single_thread()

from platform_core.comparability import encode_run_fingerprint
from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    JSONObject, dump_json_str, load_json_str, narrow_json_to_dict,
    require_int, require_float, require_str,
)
from covenant_ml.benchmarking.harness import write_observed_manifest
from covenant_ml.markets.crsp_extract import decode_crsp_extract
from covenant_ml.markets.crsp_protocol import decode_crsp_protocol
from covenant_ml.markets.crsp_return_path import crsp_compound_held
from covenant_ml.markets.crsp_rows import decode_crsp_rows
from covenant_ml.markets.encrypted_storage import decrypted_bundle
from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
from covenant_ml.markets.validation import read_document

request = read_document(Path(sys.argv[1]))
target = read_document(Path(require_str(request, 'selector')))
security = require_int(target, 'permno')
day = require_str(target, 'day')
access = decode_research_access(read_document(Path(require_str(request, 'access'))))
keys = EvaluatorKeys(access)
bundle_digest = require_str(request, 'bundle_sha256')
with decrypted_bundle(
    Path(require_str(request, 'bundle')), bundle_digest,
    partition='train-validation', identity=keys.acquire('train-validation'),
    scratch=Path(os.environ['TMPDIR']), max_plaintext_bytes=access['max_plaintext_bytes'],
) as manifest_path:
    private = manifest_path.parent
    manifest = decode_crsp_extract(load_json_str(manifest_path.read_text('utf-8')))
    protocol = decode_crsp_protocol(narrow_json_to_dict(load_json_str(
        (private / manifest['protocol']['name']).read_text('utf-8')
    )))
    assert protocol['version'] == 3
    assert day < protocol['validation_start']
    calendar = decode_crsp_rows(
        (private / manifest['calendar']['csv']['name']).read_bytes(),
        'calendar', protocol['calendar_columns'], manifest['extracted_at'],
    )
    sessions = [require_str(row, 'caldt') for row in calendar
                if row['tradingflg'] == 'Y' and row['exchangegroup'] == protocol['calendar_group']]
    position = sessions.index(day)
    assert 0 < position < len(sessions) - 1
    previous, following = sessions[position - 1], sessions[position + 1]
    sample = io.StringIO()
    writer = csv.writer(sample, lineterminator='\n')
    columns = protocol['returns_columns']
    writer.writerow(columns)
    with (private / manifest['returns']['csv']['name']).open(newline='') as stream:
        reader = csv.reader(stream, strict=True)
        assert next(reader) == list(columns)
        security_index, day_index = columns.index('permno'), columns.index('dlycaldt')
        for values in reader:
            assert len(values) == len(columns)
            if values[security_index] == str(security) and values[day_index] in (day, following):
                writer.writerow(values)
    rows = decode_crsp_rows(sample.getvalue().encode(), 'returns', columns, manifest['extracted_at'])
    assert len(rows) == 2
    indexed = {(require_int(row, 'permno'), require_str(row, 'dlycaldt')): row for row in rows}
    missing, closing = indexed[(security, day)], indexed[(security, following)]
    assert (missing['dlyretmissflg'], missing['dlyretdurflg']) == ('MP', 'MR')
    assert closing['dlyretdurflg'] == 'P1'
    growth = crsp_compound_held(indexed, security, previous, (day, following))
    expected = 1 + require_float(closing, 'dlyret')
    assert isclose(growth, expected, rel_tol=0, abs_tol=1e-12)
assert not private.exists()
fingerprint = market_fingerprint(determinism, config_test_hooks.get_env)
measurement: JSONObject = {
    'kind': 'licensed-training-span-qualification-v3',
    'native_job_id': os.environ['SLURM_JOB_ID'],
    'source_commit': '087f67b4c4aae9a731c913330840dcc665b51701',
    'bundle_sha256': bundle_digest,
    'selector_sha256': sha256(Path(require_str(request, 'selector')).read_bytes()).hexdigest(),
    'program_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
    'protocol_version': 3, 'training_only': True, 'observed_rows': len(rows),
    'missing_flag': 'MP', 'missing_duration': 'MR', 'closing_duration': 'P1',
    'complete_two_session_span': True, 'growth_matches_once': True,
    'absolute_tolerance': 1e-12, 'plaintext_removed': True,
    'scope': 'Previously failed training interval only; no fitting, selection, final access, raw prices, returns or security identifier published. Does not establish coverage of other intervals.',
}
document: JSONObject = {
    'results': [measurement], 'diagnosis_of': '58740674',
    'fingerprint': encode_run_fingerprint(fingerprint), 'dataset': {'sha256': bundle_digest},
}
write_observed_manifest(document, fingerprint, Path(sys.argv[2]),
                        sha256(dump_json_str(document).encode()).hexdigest(),
                        'licensed-training-span-qualification-v3', ())
print('Training span qualification passed; private plaintext removed.')
