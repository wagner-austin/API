from __future__ import annotations

from platform_core.trainer_keys import (
    ARTIFACT_FILE_ID_PREFIX,
    CANCEL_KEY_PREFIX,
    CONVERSATION_KEY_PREFIX,
    CONVERSATION_META_KEY_PREFIX,
    EVAL_KEY_PREFIX,
    GENERATE_KEY_PREFIX,
    HEARTBEAT_KEY_PREFIX,
    MSG_KEY_PREFIX,
    SCORE_KEY_PREFIX,
    STATUS_KEY_PREFIX,
    artifact_file_id_key,
    cancel_key,
    conversation_key,
    conversation_meta_key,
    eval_key,
    generate_key,
    heartbeat_key,
    message_key,
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
    assert generate_key(rid, req_id) == f"{GENERATE_KEY_PREFIX}{rid}:{req_id}"
    session_id = "s1"
    assert conversation_key(rid, session_id) == f"{CONVERSATION_KEY_PREFIX}{rid}:{session_id}"
    expected_conversation_meta_key = f"{CONVERSATION_META_KEY_PREFIX}{rid}:{session_id}"
    assert conversation_meta_key(rid, session_id) == expected_conversation_meta_key
