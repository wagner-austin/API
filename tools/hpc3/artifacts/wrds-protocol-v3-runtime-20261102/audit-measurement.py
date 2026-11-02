"""Read source, wheel and installation-scope evidence without running a study."""

import io
import sqlite3
import tarfile
import zipfile
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.validation import read_document
from fleet.core.export import archive_commit
from hpc3.cli._config import load_workspace_connection
from hpc3.clusters import require_cluster
from hpc3.contracts.ledger import encode_ledger_entry
from hpc3.core.ledger import read
from platform_core.event_journal import EventJournal
from platform_core.json_utils import (
    JSONObject, JSONValue, dump_json_str, load_json_str, narrow_json_to_dict,
    require_dict, require_int, require_str,
)
from platform_core.load_report import post_load_report

ROOT = Path(__file__).parent
RUNTIME = ROOT.parent
REPO = Path('/home/corvis/PROJECTS/API-codex-wrds-runs-1027')
COMMIT = '087f67b4c4aae9a731c913330840dcc665b51701'
OLD_COMMIT = '8ef7e0f1769e7127feabb30008d72bc70e908111'
START = datetime.fromisoformat('2026-11-02T03:36:25+00:00')
END = datetime.fromisoformat('2026-11-02T03:45:36+00:00')


def archive_files(payload: bytes) -> dict[str, bytes]:
    """Read regular archive members in memory, without extracting paths."""
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        result: dict[str, bytes] = {}
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None, 'RUNTIME_AUDIT_ARCHIVE_MEMBER'
                assert member.name not in result, 'RUNTIME_AUDIT_DUPLICATE_SOURCE'
                result[member.name] = stream.read()
        return result


def event_projection(history: tuple[JSONObject, ...], event: JSONObject) -> tuple[JSONObject, ...]:
    """Keep only action and clock from each event after shared hash-chain validation."""
    return (*history, {
        'action': require_str(event, 'action'),
        'observed_ms': require_int(event, 'observed_ms'),
    })


source = archive_commit(REPO, COMMIT, ROOT / 'source.tgz', ('libs', 'tools/hpc3'))
source_digest = sha256(source).hexdigest()
assert source_digest == '00c32ee4627537bf8e0c2b2664501039636b50c9e01e30943e0ac9f8fac6188a'
assert source_digest == sha256((RUNTIME / 'source.tgz').read_bytes()).hexdigest()
files = archive_files(source)
identities = read_document(RUNTIME / 'wheel-identities.json')
provenance = read_document(RUNTIME / 'controller-provenance.json')
assert require_str(provenance, 'source_commit') == COMMIT
assert require_dict(provenance, 'first_party_wheel_sha256') == identities
members: JSONObject = {}
packages: JSONObject = {}
for package, prefix in (
    ('cleargbm', 'libs/cleargbm/src/'),
    ('covenant_domain', 'libs/covenant_domain/src/'),
    ('covenant_ml', 'libs/covenant_ml/src/'),
    ('hpc3', 'tools/hpc3/src/'),
    ('platform_core', 'libs/platform_core/src/'),
    ('platform_ml', 'libs/platform_ml/src/'),
):
    name = f'{package}-0.1.0-py3-none-any.whl'
    wheel = (RUNTIME / 'controller-wheels' / name).read_bytes()
    digest = sha256(wheel).hexdigest()
    assert digest == require_str(identities, name), 'RUNTIME_AUDIT_WHEEL_DIGEST'
    matched: list[str] = []
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for member in archive.namelist():
            if member.split('/')[0].endswith('.dist-info'):
                continue
            assert member not in matched, 'RUNTIME_AUDIT_DUPLICATE_WHEEL_MEMBER'
            source_path = prefix + member
            content = archive.read(member)
            assert files[source_path] == content, 'RUNTIME_AUDIT_SOURCE_WHEEL_MISMATCH'
            members[source_path] = sha256(content).hexdigest()
            matched.append(member)
    assert matched, 'RUNTIME_AUDIT_EMPTY_WHEEL'
    packages[name] = {'sha256': digest, 'source_members': len(matched)}
assert len(members) == 622, 'RUNTIME_AUDIT_APPLICATION_MEMBER_COUNT'
member_payload = dump_json_str(members)
(ROOT / 'source-members.json').write_text(member_payload)

old_source = archive_commit(REPO, OLD_COMMIT, ROOT / 'old-rust-source.tgz', ('libs/cleargbm_rs',))
old_files = archive_files(old_source)
rust_files = {name: data for name, data in files.items() if name.startswith('libs/cleargbm_rs/')}
assert rust_files == old_files, 'RUNTIME_AUDIT_RUST_SOURCE_CHANGED'
rust_manifest: JSONObject = {name: sha256(data).hexdigest() for name, data in sorted(rust_files.items())}
rust_payload = dump_json_str(rust_manifest)
(ROOT / 'rust-source-members.json').write_text(rust_payload)
rust_name = 'cleargbm_rs-0.1.0-cp311-cp311-linux_x86_64.whl'
rust_wheel = RUNTIME / 'controller-wheels' / rust_name
old_wheel = RUNTIME.parent / 'protocol-v2-8ef7e0f17' / 'controller-wheels' / rust_name
rust_digest = sha256(rust_wheel.read_bytes()).hexdigest()
assert rust_digest == require_str(identities, rust_name)
assert rust_digest == sha256(old_wheel.read_bytes()).hexdigest(), 'RUNTIME_AUDIT_RUST_BINARY_CHANGED'

workspace = load_workspace_connection({'--config': str(RUNTIME / 'capture-config.json')})
ledger_path = Path(workspace['ledger'])
ledger_before = sha256(ledger_path.read_bytes()).hexdigest()
entries = read(ledger_path, require_cluster(workspace['cluster']))
window = [entry for entry in entries if entry['submitter'] == 'codex-wrds-1027'
          and START <= datetime.fromisoformat(entry['submitted_at']) <= END]
assert [entry['job_id'] for entry in window] == ['58742562'], 'RUNTIME_AUDIT_UNEXPECTED_JOB'
assert window[0]['experiment']['kind'] == 'image-build', 'RUNTIME_AUDIT_UNEXPECTED_KIND'
assert sha256(ledger_path.read_bytes()).hexdigest() == ledger_before

experiments = Path('/home/corvis/research-data/wrds/experiments')
paths = sorted(experiments.rglob('*.sqlite'))
assert len(paths) >= 13, 'RUNTIME_AUDIT_EMPTY_JOURNAL_INVENTORY'
journals: list[JSONValue] = []
for path in paths:
    before = sha256(path.read_bytes()).hexdigest()
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        metadata_rows: list[tuple[str]] = connection.execute('SELECT payload FROM metadata').fetchall()
    assert len(metadata_rows) == 1, 'RUNTIME_AUDIT_JOURNAL_METADATA'
    metadata = narrow_json_to_dict(load_json_str(metadata_rows[0][0]))
    assert require_str(metadata, 'mode') == 'research_validation'
    events = EventJournal(metadata, lambda: (), event_projection).read(path)
    in_window = [event for event in events if START.timestamp() * 1000 <=
                 require_int(event, 'observed_ms') <= END.timestamp() * 1000]
    assert not in_window, 'RUNTIME_AUDIT_JOURNAL_EVENT_DURING_INSTALL'
    assert sha256(path.read_bytes()).hexdigest() == before, 'RUNTIME_AUDIT_JOURNAL_CHANGED'
    journals.append({
        'path': str(path.relative_to(experiments)), 'sha256': before,
        'metadata_sha256': sha256(metadata_rows[0][0].encode()).hexdigest(),
        'events_examined': len(events), 'events_in_window': len(in_window),
        'final_releases_total': sum(require_str(event, 'action') == 'release_final' for event in events),
        'mtime': datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
    })
assert paths == sorted(experiments.rglob('*.sqlite')), 'RUNTIME_AUDIT_JOURNAL_INVENTORY_CHANGED'
grants: list[JSONValue] = []
for path in sorted(experiments.rglob('release.json')):
    stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    assert stamp < START, 'RUNTIME_AUDIT_NEW_RELEASE_METADATA'
    grants.append({'path': str(path.relative_to(experiments)),
                   'sha256': sha256(path.read_bytes()).hexdigest(), 'mtime': stamp.isoformat()})
assert not list(RUNTIME.rglob('*.sqlite')), 'RUNTIME_AUDIT_RUNTIME_JOURNAL'
assert not list(RUNTIME.rglob('release.json')), 'RUNTIME_AUDIT_RUNTIME_GRANT'

report: JSONObject = {
    'kind': 'RUNTIME_SOURCE_AND_SCOPE_EVIDENCE',
    'observed_at': datetime.now(UTC).isoformat(),
    'reporter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_commit': COMMIT, 'source_archive_sha256': source_digest,
    'archive_exporter': 'fleet.core.export.archive_commit',
    'python_wheels': packages, 'source_members': len(members),
    'source_member_manifest_sha256': sha256(member_payload.encode()).hexdigest(),
    'retained_rust': {'previous_commit': OLD_COMMIT, 'source_files': len(rust_files),
                     'source_manifest_sha256': sha256(rust_payload.encode()).hexdigest(),
                     'wheel_sha256': rust_digest, 'source_trees_equal': True,
                     'previous_wheel_equal': True},
    'window_start': START.isoformat(), 'window_end': END.isoformat(),
    'ledger': str(ledger_path), 'ledger_sha256': ledger_before,
    'ledger_entries_examined': len(entries),
    'submissions_in_window': [encode_ledger_entry(entry) for entry in window],
    'journal_inventory_root': str(experiments), 'journals': journals,
    'release_metadata': grants,
    'runtime_journals': [], 'runtime_grants': [],
    'scope': 'Read-only metadata measurement. No job submission, study replay execution, decryption or key access. Journal event chains are validated by the shared EventJournal reader; this is not a new historical filesystem or build-time attestation. Rust source equality and retained binary identity are measured, not a reproducible Rust rebuild.',
}
payload = dump_json_str(report)
(ROOT / 'report.json').write_text(payload)
post_load_report(post=_test_hooks.http_post, read_text=_test_hooks.stack_text,
                 task_id='eab6beb4-3c2c-44b2-92b1-9c8f6c853099', family='hpc',
                 body='RUNTIME SOURCE AND SCOPE LOAD-REPORT\n' + payload)
