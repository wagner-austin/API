"""Chat job processing with conversation memory."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platform_core.errors import (
    AppError,
    ErrorCode,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import (
    artifact_file_id_key,
    conversation_key,
    conversation_meta_key,
)
from platform_workers.redis import RedisStrProto
from typing_extensions import TypedDict

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.contracts.conversation import (
    ChatJobPayload,
    ConversationMessage,
)
from model_trainer.core.contracts.model import GenerateConfig
from model_trainer.core.infra.paths import models_dir
from model_trainer.core.services.container import ServiceContainer
from model_trainer.worker.job_utils import (
    load_tokenizer_for_training,
    redis_client,
    setup_job_logging,
)
from model_trainer.worker.manifest import as_model_family, load_manifest_from_text

_CHAT_RESULT_KEY_PREFIX = "runs:chat:"


def _chat_result_key(run_id: str, session_id: str, request_id: str) -> str:
    return f"{_CHAT_RESULT_KEY_PREFIX}{run_id}:{session_id}:{request_id}"


class _ChatCacheModel(TypedDict, total=False):
    status: Literal["queued", "running", "completed", "failed"]
    response: str | None


def _decode_messages(raw: str) -> list[ConversationMessage]:
    """Decode conversation messages from JSON string."""
    obj = load_json_str(raw)
    if not isinstance(obj, list):
        return []
    messages: list[ConversationMessage] = []
    for item in obj:
        if isinstance(item, dict):
            role_v = item.get("role")
            content_v = item.get("content")
            if role_v in ("user", "assistant") and isinstance(content_v, str):
                role: Literal["user", "assistant"] = "user" if role_v == "user" else "assistant"
                messages.append({"role": role, "content": content_v})
    return messages


def _encode_messages(messages: list[ConversationMessage]) -> str:
    """Encode conversation messages to JSON string."""
    msg_list: list[dict[str, str]] = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    return dump_json_str(msg_list)


def _build_prompt_from_history(messages: list[ConversationMessage]) -> str:
    """Build a prompt string from conversation history."""
    parts: list[str] = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"User: {msg['content']}")
        else:
            parts.append(f"Assistant: {msg['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _ensure_model_downloaded(settings: Settings, r: RedisStrProto, run_id: str) -> Path:
    """Download model artifact if not present, return normalized path."""
    file_id = r.get(artifact_file_id_key(run_id))
    if not isinstance(file_id, str) or file_id.strip() == "":
        raise AppError(
            ModelTrainerErrorCode.DATA_NOT_FOUND,
            "artifact pointer not found for chat",
            model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
        )

    from platform_ml import ArtifactStore

    api_url = settings["app"]["data_bank_api_url"]
    api_key = settings["app"]["data_bank_api_key"]
    store = ArtifactStore(api_url, api_key)
    models_root = models_dir(settings)
    expected_root = f"model-{run_id}"
    normalized = models_root / run_id

    if not normalized.exists():
        out_root = store.download_artifact(
            file_id.strip(),
            dest_dir=models_root,
            request_id=run_id,
            expected_root=expected_root,
        )
        out_root.rename(normalized)

    return normalized


def _run_generation(
    settings: Settings, model_path: Path, prompt: str, payload: ChatJobPayload
) -> str:
    """Load model and run generation, return response text."""
    manifest_path = model_path / "manifest.json"
    if not manifest_path.exists():
        raise AppError(
            ModelTrainerErrorCode.MODEL_NOT_FOUND,
            f"manifest missing for run_id={payload['run_id']}",
            model_trainer_status_for(ModelTrainerErrorCode.MODEL_NOT_FOUND),
        )

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = load_manifest_from_text(manifest_text)

    tok_handle = load_tokenizer_for_training(settings, manifest["tokenizer_id"])
    container = ServiceContainer.from_settings(settings)
    backend = container.model_registry.get(as_model_family(manifest["model_family"]))

    cfg = GenerateConfig(
        prompt_text=prompt,
        prompt_path=None,
        max_new_tokens=payload["max_new_tokens"],
        temperature=payload["temperature"],
        top_k=payload["top_k"],
        top_p=payload["top_p"],
        stop_on_eos=True,
        stop_sequences=["User:", "\nUser:"],
        seed=None,
        num_return_sequences=1,
    )

    prepared = backend.load(str(model_path), settings, tokenizer=tok_handle)
    result = backend.generate(prepared=prepared, cfg=cfg, settings=settings)

    if result["outputs"] and len(result["outputs"]) > 0:
        return result["outputs"][0].strip()
    return ""


def _update_session_ttl(r: RedisStrProto, run_id: str, session_id: str) -> None:
    """Extend TTL on conversation keys if configured."""
    meta_key = conversation_meta_key(run_id, session_id)
    raw_meta = r.get(meta_key)
    if raw_meta is not None and isinstance(raw_meta, str):
        meta_obj = load_json_str(raw_meta)
        if isinstance(meta_obj, dict):
            ttl_v = meta_obj.get("session_ttl_sec")
            if isinstance(ttl_v, int) and ttl_v > 0:
                conv_key = conversation_key(run_id, session_id)
                r.expire(conv_key, ttl_v)
                r.expire(meta_key, ttl_v)


def process_chat_job(payload: ChatJobPayload) -> None:
    """Process a chat job with conversation memory."""
    settings = load_settings()
    setup_job_logging(settings)

    log = get_logger(__name__)
    r = redis_client(settings)
    run_id = payload["run_id"]
    session_id = payload["session_id"]
    request_id = payload["request_id"]

    result_key = _chat_result_key(run_id, session_id, request_id)
    running: _ChatCacheModel = {"status": "running", "response": None}
    r.set(result_key, dump_json_str(running))

    try:
        # Load conversation history and add user message
        conv_key = conversation_key(run_id, session_id)
        raw_messages = r.get(conv_key)
        messages: list[ConversationMessage] = (
            _decode_messages(raw_messages)
            if raw_messages is not None and isinstance(raw_messages, str)
            else []
        )
        user_msg: ConversationMessage = {"role": "user", "content": payload["prompt"]}
        messages.append(user_msg)

        # Build prompt and generate response
        full_prompt = _build_prompt_from_history(messages)
        model_path = _ensure_model_downloaded(settings, r, run_id)
        response_text = _run_generation(settings, model_path, full_prompt, payload)

        # Update history with assistant response
        assistant_msg: ConversationMessage = {"role": "assistant", "content": response_text}
        messages.append(assistant_msg)
        r.set(conv_key, _encode_messages(messages))
        _update_session_ttl(r, run_id, session_id)

        out: _ChatCacheModel = {"status": "completed", "response": response_text}
    except Exception as e:
        out_failed: _ChatCacheModel = {"status": "failed", "response": None}
        log.exception(
            "Chat failed run_id=%s session_id=%s request_id=%s error=%s",
            run_id,
            session_id,
            request_id,
            e,
        )
        r.set(result_key, dump_json_str(out_failed))
        raise
    else:
        r.set(result_key, dump_json_str(out))
        log.info(
            "Chat job completed run_id=%s session_id=%s request_id=%s",
            run_id,
            session_id,
            request_id,
        )


def decode_chat_job_payload(obj: JSONValue) -> ChatJobPayload:
    """Decode a chat job payload from JSON."""
    if not isinstance(obj, dict):
        raise AppError(ErrorCode.INVALID_INPUT, "chat payload must be dict", 400)
    run_id_v = obj.get("run_id")
    session_id_v = obj.get("session_id")
    request_id_v = obj.get("request_id")
    prompt_v = obj.get("prompt")
    max_new_tokens_v = obj.get("max_new_tokens")
    temperature_v = obj.get("temperature")
    top_k_v = obj.get("top_k")
    top_p_v = obj.get("top_p")

    if not isinstance(run_id_v, str):
        raise AppError(ErrorCode.INVALID_INPUT, "run_id must be string", 400)
    if not isinstance(session_id_v, str):
        raise AppError(ErrorCode.INVALID_INPUT, "session_id must be string", 400)
    if not isinstance(request_id_v, str):
        raise AppError(ErrorCode.INVALID_INPUT, "request_id must be string", 400)
    if not isinstance(prompt_v, str):
        raise AppError(ErrorCode.INVALID_INPUT, "prompt must be string", 400)
    if not isinstance(max_new_tokens_v, int):
        raise AppError(ErrorCode.INVALID_INPUT, "max_new_tokens must be int", 400)
    if not isinstance(temperature_v, int | float):
        raise AppError(ErrorCode.INVALID_INPUT, "temperature must be number", 400)
    if not isinstance(top_k_v, int):
        raise AppError(ErrorCode.INVALID_INPUT, "top_k must be int", 400)
    if not isinstance(top_p_v, int | float):
        raise AppError(ErrorCode.INVALID_INPUT, "top_p must be number", 400)

    return {
        "run_id": run_id_v,
        "session_id": session_id_v,
        "request_id": request_id_v,
        "prompt": prompt_v,
        "max_new_tokens": max_new_tokens_v,
        "temperature": float(temperature_v),
        "top_k": top_k_v,
        "top_p": float(top_p_v),
    }


__all__ = ["decode_chat_job_payload", "process_chat_job"]
