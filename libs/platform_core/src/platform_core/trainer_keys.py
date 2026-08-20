from __future__ import annotations

from typing import Final

HEARTBEAT_KEY_PREFIX: Final[str] = "runs:hb:"
STATUS_KEY_PREFIX: Final[str] = "runs:status:"
EVAL_KEY_PREFIX: Final[str] = "runs:eval:"
MSG_KEY_PREFIX: Final[str] = "runs:message:"
ARTIFACT_FILE_ID_PREFIX: Final[str] = "runs:artifact:"
CANCEL_KEY_PREFIX: Final[str] = "runs:"
JOB_ID_KEY_PREFIX: Final[str] = "runs:job:"
SCORE_KEY_PREFIX: Final[str] = "runs:score:"
CLOZE_KEY_PREFIX: Final[str] = "runs:cloze:"
GENERATE_KEY_PREFIX: Final[str] = "runs:gen:"
CONVERSATION_KEY_PREFIX: Final[str] = "runs:conv:"
CONVERSATION_META_KEY_PREFIX: Final[str] = "runs:conv:meta:"
PROGRESS_KEY_PREFIX: Final[str] = "runs:progress:"


def heartbeat_key(run_id: str) -> str:
    return f"{HEARTBEAT_KEY_PREFIX}{run_id}"


def status_key(run_id: str) -> str:
    return f"{STATUS_KEY_PREFIX}{run_id}"


def eval_key(run_id: str) -> str:
    return f"{EVAL_KEY_PREFIX}{run_id}"


def message_key(run_id: str) -> str:
    return f"{MSG_KEY_PREFIX}{run_id}"


def artifact_file_id_key(run_id: str) -> str:
    return f"{ARTIFACT_FILE_ID_PREFIX}{run_id}:file_id"


def cancel_key(run_id: str) -> str:
    return f"{CANCEL_KEY_PREFIX}{run_id}:cancelled"


def job_id_key(run_id: str) -> str:
    """Key holding the queue job id of a run's current execution.

    A run id is not a job id: one run can be enqueued more than once through
    resume, and the queue only knows the job. Cancelling a run that has not
    started yet has to remove that job from the queue, so the mapping has to
    outlive the enqueue call that returned it.

    Args:
        run_id: The training run.

    Returns:
        The Redis key holding the run's current queue job id.
    """
    return f"{JOB_ID_KEY_PREFIX}{run_id}"


def score_key(run_id: str, request_id: str) -> str:
    return f"{SCORE_KEY_PREFIX}{run_id}:{request_id}"


def cloze_key(run_id: str, request_id: str) -> str:
    return f"{CLOZE_KEY_PREFIX}{run_id}:{request_id}"


def generate_key(run_id: str, request_id: str) -> str:
    return f"{GENERATE_KEY_PREFIX}{run_id}:{request_id}"


def conversation_key(run_id: str, session_id: str) -> str:
    return f"{CONVERSATION_KEY_PREFIX}{run_id}:{session_id}"


def conversation_meta_key(run_id: str, session_id: str) -> str:
    return f"{CONVERSATION_META_KEY_PREFIX}{run_id}:{session_id}"


def progress_key(run_id: str) -> str:
    return f"{PROGRESS_KEY_PREFIX}{run_id}"


__all__ = [
    "ARTIFACT_FILE_ID_PREFIX",
    "CANCEL_KEY_PREFIX",
    "CLOZE_KEY_PREFIX",
    "CONVERSATION_KEY_PREFIX",
    "CONVERSATION_META_KEY_PREFIX",
    "EVAL_KEY_PREFIX",
    "GENERATE_KEY_PREFIX",
    "HEARTBEAT_KEY_PREFIX",
    "JOB_ID_KEY_PREFIX",
    "MSG_KEY_PREFIX",
    "PROGRESS_KEY_PREFIX",
    "SCORE_KEY_PREFIX",
    "STATUS_KEY_PREFIX",
    "artifact_file_id_key",
    "cancel_key",
    "cloze_key",
    "conversation_key",
    "conversation_meta_key",
    "eval_key",
    "generate_key",
    "heartbeat_key",
    "job_id_key",
    "message_key",
    "progress_key",
    "score_key",
    "status_key",
]
