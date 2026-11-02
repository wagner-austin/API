"""Inspect the synthetic native result without executing models or decrypting data."""

from datetime import UTC, datetime
from hashlib import sha256
from math import isclose, prod
from pathlib import Path

from platform_core.json_utils import (
    JSONObject, JSONValue, dump_json_str, narrow_json_to_dict,
    require_float, require_list, require_str,
)
from platform_core.load_report import post_load_report
from covenant_ml.markets import _test_hooks
from covenant_ml.markets.research_preparation_report import report_preparation
from covenant_ml.markets.validation import read_document
from tests.markets._crsp_study_fixtures import pre_entry_rows

root = Path(__file__).parent
result = read_document(root / 'evaluation.json')
cases = [narrow_json_to_dict(value) for value in require_list(result, 'results')]
assert len(cases) == 15
assert len(require_list(result, 'comparisons')) == 4
inspected: list[JSONValue] = []
observations = 0
for case in cases:
    months = [narrow_json_to_dict(value) for value in require_list(case, 'months')]
    assert len(months) == 16
    for month in months:
        held = [require_list({'row': value}, 'row') for value in require_list(month, 'held')]
        weights = [require_float({'value': pair[1]}, 'value') for pair in held]
        claims = [narrow_json_to_dict(value) for value in require_list(month, 'receivables')]
        total = require_float(month, 'cash') + sum(weights) + sum(
            require_float(claim, 'amount') for claim in claims
        )
        assert isclose(total, 1.0, abs_tol=1e-12, rel_tol=0)
        expected_cost = require_float(month, 'turnover') * require_float(case, 'cost_bps') / 10000
        assert isclose(require_float(month, 'cost'), expected_cost, abs_tol=1e-12, rel_tol=0)
        observations += 1
    if case['arm'] == 'equal_weight' and case['cost_bps'] == 10:
        previous, = [month for month in months if month['decision'] == '2019-01-31']
        current, = [month for month in months if month['decision'] == '2019-02-28']
        following, = [month for month in months if month['decision'] == '2019-03-29']
        assert current['selected'] == [3, 4, 5, 6, 7, 8]
        assert current['unfilled'] == [8]
        cash = require_float(current, 'cash')
        assert cash > 0
        prior_claim, = [narrow_json_to_dict(item) for item in require_list(previous, 'receivables')]
        current_claim, = [narrow_json_to_dict(item) for item in require_list(current, 'receivables')]
        assert (current_claim['permno'], current_claim['valued']) == (8, '2019-03-01')
        assert isclose(require_float(current_claim, 'amount'),
                       require_float(prior_claim, 'amount') / (1 + require_float(current, 'net_return')),
                       abs_tol=1e-12, rel_tol=0)
        assert following['receivables'] == []
        holding, = [require_list({'row': value}, 'row')
                    for value in require_list(current, 'held')
                    if require_list({'row': value}, 'row')[0] == 7]
        expected_growth = prod(1 + require_float(row, 'dlyret')
                               for row in pre_entry_rows()['returns']
                               if row['permno'] == 7 and
                               '2019-03-01' < require_str(row, 'dlycaldt') <= '2019-04-01')
        observed_growth = require_float({'value': holding[1]}, 'value') / cash
        assert isclose(observed_growth, expected_growth, abs_tol=1e-12, rel_tol=0)
        inspected.append({'arm': case['arm'], 'cost_bps': case['cost_bps'],
                          'entry_case': current, 'gap_expected_growth': expected_growth,
                          'gap_observed_growth': observed_growth})
assert len(inspected) == 1
completed = report_preparation(root / 'policy.json', root / 'attempts.sqlite',
                               root / 'attempt-1/run.json',
                               '284bc096-4b05-446b-9463-5ef5c49fa520')
attempt, = [narrow_json_to_dict(value) for value in require_list(completed, 'attempts')]
assert (attempt['status'], attempt['job_id'], attempt['selection'],
        attempt['final_manifest_digest']) == ('complete', '58747974', '', '')
(root / 'completed-report.json').write_text(dump_json_str(completed))
report: JSONObject = {
    'kind': 'SYNTHETIC_V3_RESULT_EVIDENCE',
    'observed_at': datetime.now(UTC).isoformat(),
    'reporter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
    'payload_sha256': sha256((root / 'evaluation.json').read_bytes()).hexdigest(),
    'sidecar_sha256': sha256((root / 'evaluation.json.runrecord.json').read_bytes()).hexdigest(),
    'arm_cost_cases': len(cases), 'period_observations': observations,
    'months_per_case': 16, 'paired_comparisons': 4,
    'absolute_tolerance': 1e-12, 'inspected_cases': inspected,
    'completed_journal': completed,
    'scope': 'Readback only of synthetic native results. All 240 period accounting identities checked. The retained-cash slot and stock-7 ending weight independently recover the full original daily-product return across MP/MR-to-P1. No evaluator, model fit, licensed decryption, selection or final release ran here.',
}
payload = dump_json_str(report)
(root / 'result-inspection.json').write_text(payload)
post_load_report(post=_test_hooks.http_post, read_text=_test_hooks.stack_text,
                 task_id='284bc096-4b05-446b-9463-5ef5c49fa520', family='hpc',
                 body='SYNTHETIC V3 RESULT LOAD-REPORT\n' + payload)
