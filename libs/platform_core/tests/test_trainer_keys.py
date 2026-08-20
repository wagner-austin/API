from __future__ import annotations

from platform_core.trainer_keys import (
    ARTIFACT_FILE_ID_PREFIX,
    BASELINE_CLOZE_KEY_PREFIX,
    CANCEL_KEY_PREFIX,
    CLOZE_KEY_PREFIX,
    CONVERSATION_KEY_PREFIX,
    CONVERSATION_META_KEY_PREFIX,
    EVAL_KEY_PREFIX,
    GENERATE_KEY_PREFIX,
    HEARTBEAT_KEY_PREFIX,
    JOB_ID_KEY_PREFIX,
    MSG_KEY_PREFIX,
    PROGRESS_KEY_PREFIX,
    SCORE_KEY_PREFIX,
    STATUS_KEY_PREFIX,
    artifact_file_id_key,
    baseline_cloze_key,
    cancel_key,
    cloze_key,
    conversation_key,
    conversation_meta_key,
    eval_key,
    generate_key,
    heartbeat_key,
    job_id_key,
    message_key,
    progress_key,
    score_key,
    status_key,
)


def test_trainer_key_helpers() -> None:
    rid = "r1"
    req_id = "req1"
    assert heartbeat_key(rid) == f"{HEARTBEAT_KEY_PREFIX}{rid}"
    assert status_key(rid) == f"{STATUS_KEY_PREFIX}{rid}"
    assert eval_key(rid) == f"{EVAL_KEY_PREFIX}{rid}"
    assert message_key(rid) == f"{MSG_KEY_PREFIX}{rid}"
    assert artifact_file_id_key(rid) == f"{ARTIFACT_FILE_ID_PREFIX}{rid}:file_id"
    assert cancel_key(rid) == f"{CANCEL_KEY_PREFIX}{rid}:cancelled"
    assert score_key(rid, req_id) == f"{SCORE_KEY_PREFIX}{rid}:{req_id}"
    assert cloze_key(rid, req_id) == f"{CLOZE_KEY_PREFIX}{rid}:{req_id}"
    assert generate_key(rid, req_id) == f"{GENERATE_KEY_PREFIX}{rid}:{req_id}"
    session_id = "s1"
    assert conversation_key(rid, session_id) == f"{CONVERSATION_KEY_PREFIX}{rid}:{session_id}"
    expected_conversation_meta_key = f"{CONVERSATION_META_KEY_PREFIX}{rid}:{session_id}"
    assert conversation_meta_key(rid, session_id) == expected_conversation_meta_key
    assert progress_key(rid) == f"{PROGRESS_KEY_PREFIX}{rid}"
    assert job_id_key(rid) == f"{JOB_ID_KEY_PREFIX}{rid}"
    assert baseline_cloze_key("gpt2", "f1") == f"{BASELINE_CLOZE_KEY_PREFIX}gpt2:f1"


def test_job_id_key_does_not_collide_with_the_cancel_key() -> None:
    """Both hang off `runs:`, and a collision would make cancel unusable.

    `cancel_key` uses a bare `runs:` prefix with a suffix, so a new key under
    the same namespace has to be checked rather than assumed distinct.
    """
    rid = "r1"
    assert job_id_key(rid) != cancel_key(rid)
    assert job_id_key(rid) != status_key(rid)


def test_baseline_cloze_key_is_not_in_the_run_namespace() -> None:
    """A baseline is not a run and must never be readable as one.

    Both key families concern the same measurement type, so the separation is
    asserted rather than assumed: a baseline that collided into `runs:cloze:`
    would appear in the run ledger as if something had trained.
    """
    assert not baseline_cloze_key("gpt2", "f1").startswith("runs:")
    assert baseline_cloze_key("gpt2", "f1") != cloze_key("gpt2", "f1")
