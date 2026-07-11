from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch
from omegaconf import DictConfig


def _find_retool_video_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "run_e2e.py").exists() and (parent / "planner").exists():
            return parent
    raise RuntimeError("Unable to locate Retool_Video_RL project root from rollout package.")


PROJECT_ROOT = _find_retool_video_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_FILE_SUFFIXES = (
    ".mp4",
    ".webm",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
    ".flv",
    ".wmv",
    ".mpeg",
    ".mpg",
    ".ts",
    ".m2ts",
)

from config.settings import settings
from core.llm import (
    AcceleratorFatalError,
    BaseLLM,
    GenerationTimeoutError,
    get_llm_provider,
    is_accelerator_fatal_error,
)
from planner.engine import RecursiveEngine
from run_e2e import (
    _build_planner_grounding_context,
    _build_query_run_artifact_paths,
    _build_run_result_payload,
    _grounding_payload_to_text,
    _json_safe,
    _log_trajectory,
    _write_json_file,
    preprocess_video,
)
from runtime.session import runtime_session
from toolkit.impl.visual import prepare_planner_image_attachment
from toolkit.registry import registry
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.utils.logging_utils import format_human_readable_timestamp

logger = logging.getLogger(__name__)
_EPISODE_STREAM_REGISTRY_LOCK = threading.Lock()
_EPISODE_STDOUT_TARGETS: Dict[int, List[Any]] = {}
_EPISODE_STDERR_TARGETS: Dict[int, List[Any]] = {}
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_THREAD_AWARE_EPISODE_STREAMS_INSTALLED = False


def _run_async_blocking(async_callable: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run an async callable from sync code, even if this thread already has a running loop."""

    def _runner() -> Any:
        return asyncio.run(async_callable(*args, **kwargs))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _runner()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_runner).result()


_EPISODE_TRACE_LOGGER_NAMES = (
    "",
    "core.llm.provider",
    "core.memory.context_manager",
    "planner.engine",
    "planner.react_agent",
    "run_e2e",
)


def _promote_episode_trace_logging(min_level: int = logging.INFO) -> List[tuple[logging.Logger, int]]:
    overrides: List[tuple[logging.Logger, int]] = []
    for logger_name in _EPISODE_TRACE_LOGGER_NAMES:
        target_logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        current_level = int(target_logger.level)
        if int(target_logger.getEffectiveLevel()) <= min_level:
            continue
        target_logger.setLevel(min_level)
        overrides.append((target_logger, current_level))
    return overrides


def _restore_logger_levels(overrides: Sequence[tuple[logging.Logger, int]]) -> None:
    for target_logger, previous_level in reversed(list(overrides or [])):
        try:
            target_logger.setLevel(previous_level)
        except Exception:
            continue


class _ThreadAwareEpisodeStream:
    def __init__(self, primary_stream: Any, registry: Dict[int, List[Any]]) -> None:
        self._primary_stream = primary_stream
        self._registry = registry

    def write(self, text: str) -> int:
        normalized = str(text or "")
        self._primary_stream.write(normalized)
        mirrors: List[Any] = []
        thread_id = threading.get_ident()
        with _EPISODE_STREAM_REGISTRY_LOCK:
            mirrors = list(self._registry.get(thread_id, ()))
        for mirror_stream in mirrors:
            try:
                mirror_stream.write(normalized)
            except Exception:
                continue
        return len(normalized)

    def flush(self) -> None:
        self._primary_stream.flush()
        thread_id = threading.get_ident()
        with _EPISODE_STREAM_REGISTRY_LOCK:
            mirrors = list(self._registry.get(thread_id, ()))
        for mirror_stream in mirrors:
            try:
                mirror_stream.flush()
            except Exception:
                continue

    def isatty(self) -> bool:
        try:
            return bool(self._primary_stream.isatty())
        except Exception:
            return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary_stream, name)

    @property
    def encoding(self) -> str:
        return "utf-8"


class _ThreadBoundLogFilter(logging.Filter):
    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self._thread_id = int(thread_id)

    def filter(self, record: logging.LogRecord) -> bool:
        return int(getattr(record, "thread", -1)) == self._thread_id


def _ensure_thread_aware_episode_streams_installed() -> None:
    global _THREAD_AWARE_EPISODE_STREAMS_INSTALLED
    with _EPISODE_STREAM_REGISTRY_LOCK:
        if _THREAD_AWARE_EPISODE_STREAMS_INSTALLED:
            return
        sys.stdout = _ThreadAwareEpisodeStream(_ORIGINAL_STDOUT, _EPISODE_STDOUT_TARGETS)
        sys.stderr = _ThreadAwareEpisodeStream(_ORIGINAL_STDERR, _EPISODE_STDERR_TARGETS)
        _THREAD_AWARE_EPISODE_STREAMS_INSTALLED = True


def _register_episode_thread_stream(thread_id: int, target: Any, *, is_stderr: bool = False) -> None:
    registry = _EPISODE_STDERR_TARGETS if is_stderr else _EPISODE_STDOUT_TARGETS
    with _EPISODE_STREAM_REGISTRY_LOCK:
        streams = registry.setdefault(int(thread_id), [])
        streams.append(target)


def _unregister_episode_thread_stream(thread_id: int, target: Any, *, is_stderr: bool = False) -> None:
    registry = _EPISODE_STDERR_TARGETS if is_stderr else _EPISODE_STDOUT_TARGETS
    with _EPISODE_STREAM_REGISTRY_LOCK:
        streams = registry.get(int(thread_id))
        if not streams:
            return
        registry[int(thread_id)] = [item for item in streams if item is not target]
        if not registry[int(thread_id)]:
            registry.pop(int(thread_id), None)


@contextmanager
def _capture_episode_run_log(log_path: Path) -> Iterator[logging.Handler]:
    resolved_log_path = Path(log_path).resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_thread_aware_episode_streams_installed()
    level_overrides = _promote_episode_trace_logging()
    thread_id = threading.get_ident()
    with open(resolved_log_path, "w", encoding="utf-8-sig") as log_handle:
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(log_handle)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        handler.addFilter(_ThreadBoundLogFilter(thread_id))
        root_logger.addHandler(handler)
        _register_episode_thread_stream(thread_id, log_handle, is_stderr=False)
        _register_episode_thread_stream(thread_id, log_handle, is_stderr=True)
        try:
            yield handler
        finally:
            _unregister_episode_thread_stream(thread_id, log_handle, is_stderr=False)
            _unregister_episode_thread_stream(thread_id, log_handle, is_stderr=True)
            root_logger.removeHandler(handler)
            handler.flush()
            handler.close()
            _restore_logger_levels(level_overrides)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _resolve_effective_time_budget_sec(sample: Dict[str, Any], rollout_config: DictConfig) -> float:
    base_budget_sec = _coerce_float(sample.get("time_budget_sec", 0.0), 0.0)
    if base_budget_sec <= 0:
        return 0.0

    multiplier = _coerce_float(rollout_config.get("wall_clock_budget_multiplier", 1.0), 1.0)
    extra_seconds = _coerce_float(rollout_config.get("wall_clock_budget_extra_seconds", 0.0), 0.0)
    min_seconds = _coerce_float(rollout_config.get("wall_clock_budget_min_seconds", 0.0), 0.0)

    effective_budget_sec = max(0.0, base_budget_sec * max(0.0, multiplier) + max(0.0, extra_seconds))
    planner_model_b = _resolve_role_model_parameter_billions("planner")
    large_model_threshold_b = _coerce_float(rollout_config.get("large_model_parameter_threshold_b", 0.0), 0.0)
    large_model_wall_clock_scale = 1.0
    large_model_wall_clock_min_seconds = 0.0
    if large_model_threshold_b > 0 and planner_model_b >= large_model_threshold_b:
        large_model_wall_clock_scale = max(
            1.0,
            _coerce_float(rollout_config.get("large_model_wall_clock_budget_multiplier", 1.0), 1.0),
        )
        large_model_wall_clock_min_seconds = max(
            0.0,
            _coerce_float(rollout_config.get("large_model_wall_clock_budget_min_seconds", 0.0), 0.0),
        )
        effective_budget_sec *= large_model_wall_clock_scale
    if min_seconds > 0:
        effective_budget_sec = max(effective_budget_sec, min_seconds)
    if large_model_wall_clock_min_seconds > 0:
        effective_budget_sec = max(effective_budget_sec, large_model_wall_clock_min_seconds)
    logger.info(
        "[Episode][BudgetProfile][DIAG] sample_time_budget_sec=%.1f base_multiplier=%.3f base_extra_seconds=%.1f "
        "planner_model_b=%.1f large_model_threshold_b=%.1f large_model_wall_clock_scale=%.3f "
        "base_min_seconds=%.1f large_model_min_seconds=%.1f effective_budget_sec=%.1f",
        base_budget_sec,
        multiplier,
        extra_seconds,
        planner_model_b,
        large_model_threshold_b,
        large_model_wall_clock_scale,
        min_seconds,
        large_model_wall_clock_min_seconds,
        effective_budget_sec,
    )
    return effective_budget_sec


def _extract_model_parameter_billions(value: Any) -> float:
    raw_text = str(value or "").strip()
    if not raw_text:
        return 0.0
    candidates = [raw_text]
    path_name = Path(raw_text).name.strip()
    if path_name and path_name not in candidates:
        candidates.append(path_name)
    for candidate in candidates:
        match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*[bB](?!\w)", candidate)
        if match is None:
            continue
        try:
            return max(0.0, float(match.group(1)))
        except Exception:
            continue
    return 0.0


def _resolve_role_model_parameter_billions(role: str) -> float:
    normalized_role = str(role or "").strip().lower()
    if normalized_role == "planner":
        candidates = (settings.planner_model_subpath, settings.planner_model_path)
    elif normalized_role == "subagent":
        candidates = (settings.subagent_model_subpath, settings.subagent_model_path)
    else:
        candidates = ()
    for candidate in candidates:
        model_b = _extract_model_parameter_billions(candidate)
        if model_b > 0:
            return model_b
    return 0.0


def _truncate_progress_text(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _strip_think_blocks(text: Any) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _flatten_content(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type == "text":
                text_value = str(item.get("text", "") or "").strip()
                if text_value:
                    parts.append(text_value)
            elif item_type == "image":
                sample_times = item.get("sample_times")
                if isinstance(sample_times, list) and sample_times:
                    parts.append(
                        "[image sample_times="
                        + ",".join(str(value) for value in sample_times[:4])
                        + "]"
                    )
                else:
                    parts.append("[image]")
            elif item_type == "video":
                parts.append("[video]")
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _stringify_message_content(content: Any) -> str:
    return _flatten_content(content)


def _ensure_user_message(
    messages: Sequence[Dict[str, Any]],
    *,
    fallback_text: str = "",
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    saw_user = False
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role == "user":
            saw_user = True
        normalized.append(message)
    if saw_user:
        return normalized
    normalized.append({"role": "user", "content": str(fallback_text or "")})
    return normalized


def _render_text_chat(
    messages: Sequence[Dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    lines: List[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip() or "user"
        content = _flatten_content(message.get("content"))
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("assistant:")
    return "\n".join(lines)


def _apply_chat_template_with_optional_thinking(
    client: Any,
    messages: Sequence[Dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: Optional[bool] = None,
) -> Optional[str]:
    apply_chat_template = getattr(client, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None

    base_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    attempts: List[Dict[str, Any]] = []
    if enable_thinking is not None:
        attempts.append(
            {
                **base_kwargs,
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
            }
        )
        attempts.append(
            {
                **base_kwargs,
                "enable_thinking": bool(enable_thinking),
            }
        )
    attempts.append(dict(base_kwargs))

    last_error: Optional[BaseException] = None
    for request_kwargs in attempts:
        has_thinking_control = (
            "enable_thinking" in request_kwargs
            or "chat_template_kwargs" in request_kwargs
        )
        try:
            rendered = apply_chat_template(list(messages), **request_kwargs)
        except Exception as exc:
            last_error = exc
            continue
        if enable_thinking is False and not has_thinking_control:
            logger.warning(
                "[ThinkingBoundary][DIAG] source=rollout.apply_chat_template role=subagent no-think chat template arguments were rejected by %s; rendered prompt without template-level thinking control. last_error=%s",
                type(client).__name__,
                last_error,
            )
        return str(rendered)
    if enable_thinking is False:
        logger.warning(
            "[ThinkingBoundary][DIAG] source=rollout.apply_chat_template role=subagent no-think chat template arguments were rejected by %s; falling back without template-level thinking control. error=%s",
            type(client).__name__,
            last_error,
        )
    return None


def _normalize_visual_reference(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    if text.startswith(("http://", "https://", "file://", "data:")):
        return text
    candidate = Path(text).expanduser()
    try:
        return candidate.resolve().as_uri() if candidate.exists() else text
    except Exception:
        return text


def _prepare_rollout_image_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_image_value = item.get("image") or item.get("path") or item.get("url")
    image_text = str(raw_image_value or "").strip()
    if not image_text:
        return None
    if image_text.startswith(("http://", "https://", "file://", "data:")):
        normalized_item = {
            "type": "image",
            "image": _normalize_visual_reference(image_text),
        }
        for passthrough_key in ("min_pixels", "max_pixels", "sample_times"):
            passthrough_value = item.get(passthrough_key)
            if passthrough_value not in (None, "", []):
                normalized_item[passthrough_key] = passthrough_value
        return normalized_item

    prepared = prepare_planner_image_attachment(
        {
            "image": image_text,
            "timestamp": item.get("timestamp"),
            "sample_times": list(item.get("sample_times") or []),
            "source_images": list(item.get("source_images") or []),
        }
    )
    if prepared is None:
        return None

    normalized_item = {
        "type": "image",
        "image": _normalize_visual_reference(prepared.get("image") or image_text),
    }
    source_images = [
        str(Path(source).expanduser().resolve())
        for source in (prepared.get("source_images") or [])
        if str(source).strip()
    ]
    if source_images:
        normalized_item["source_images"] = source_images
    if prepared.get("timestamp") is not None:
        normalized_item["timestamp"] = prepared.get("timestamp")
    sample_times = [
        value
        for value in (prepared.get("sample_times") or [])
        if value is not None
    ]
    if sample_times:
        normalized_item["sample_times"] = sample_times
    for passthrough_key in ("min_pixels", "max_pixels"):
        passthrough_value = item.get(passthrough_key)
        if passthrough_value not in (None, "", []):
            normalized_item[passthrough_key] = passthrough_value
    return normalized_item


def _normalize_multimodal_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip() or "user"
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append({"role": role, "content": _stringify_message_content(content)})
            continue

        content_items: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type == "text":
                text_value = str(item.get("text", "") or "").strip()
                if text_value:
                    content_items.append({"type": "text", "text": text_value})
            elif item_type == "image":
                normalized_item = _prepare_rollout_image_item(item)
                if normalized_item:
                    content_items.append(normalized_item)
            elif item_type == "video":
                video_value = _normalize_visual_reference(
                    item.get("video") or item.get("path") or item.get("url")
                )
                if video_value:
                    normalized_item = {"type": "video", "video": video_value}
                    for passthrough_key in (
                        "fps",
                        "max_frames",
                        "min_pixels",
                        "max_pixels",
                        "do_sample_frames",
                        "sample_times",
                    ):
                        passthrough_value = item.get(passthrough_key)
                        if passthrough_value not in (None, "", []):
                            normalized_item[passthrough_key] = passthrough_value
                    content_items.append(normalized_item)
        if not content_items:
            content_items.append({"type": "text", "text": ""})
        normalized.append({"role": role, "content": content_items})
    return normalized


def _message_contains_visual_inputs(message: Dict[str, Any]) -> bool:
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).strip().lower() in {"image", "video"}:
            return True
    return False


def _messages_contain_visual_inputs(messages: List[Dict[str, Any]]) -> bool:
    return any(_message_contains_visual_inputs(message) for message in messages or [])


def _canonical_message(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _longest_common_message_prefix(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and _canonical_message(left[index]) == _canonical_message(right[index]):
        index += 1
    return index


def _copy_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(list(messages or []))


def _sanitize_decoded_text(text: Any) -> str:
    if text is None:
        return ""

    if isinstance(text, bytes):
        normalized = text.decode("utf-8", errors="replace")
    else:
        normalized = str(text)

    normalized = normalized.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    normalized = normalized.replace("\x00", "")
    try:
        normalized = unicodedata.normalize("NFC", normalized)
    except Exception:
        pass
    return normalized


def _decode_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    ids = [int(token_id) for token_id in token_ids]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    while ids and pad_token_id is not None and ids[-1] == pad_token_id:
        ids.pop()
    while ids and eos_token_id is not None and ids[-1] == eos_token_id:
        ids.pop()
    if not ids:
        return ""

    try:
        text = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        logger.warning("Tokenizer decode failed in rollout text decode: %s", exc)
        return ""
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        text = text.replace(eos_token, "")
    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token and pad_token != eos_token:
        text = text.replace(pad_token, "")
    all_special_tokens = getattr(tokenizer, "all_special_tokens", None)
    if isinstance(all_special_tokens, (list, tuple, set)):
        for special_token in all_special_tokens:
            normalized_special_token = str(special_token or "")
            if normalized_special_token:
                text = text.replace(normalized_special_token, "")
    return _sanitize_decoded_text(text).strip()


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and int(left[index]) == int(right[index]):
        index += 1
    return index


def _resolve_path(value: Any, *, base: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _resolve_workspace_root(sample: Dict[str, Any], rollout_config: DictConfig) -> Path:
    workspace_value = sample.get("workspace_root")
    if workspace_value:
        return _resolve_path(workspace_value, base=PROJECT_ROOT)

    processed_value = sample.get("processed_root")
    if processed_value:
        processed_path = _resolve_path(processed_value, base=PROJECT_ROOT)
        for parent in processed_path.parents:
            if parent.name.lower() == "queries":
                return parent.parent
        if processed_path.parent.name == "processed":
            return processed_path.parent.parent
        if processed_path.name == "processed":
            return processed_path.parent
        return processed_path.parent

    fallback_root = rollout_config.get("workspace_root", "./videorag_cache/rl")
    return _resolve_path(fallback_root, base=PROJECT_ROOT)

def _sanitize_workspace_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "sample"
    text = text.replace("\\", "_").replace("/", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "sample"


def _compact_workspace_name(
    value: Any,
    *,
    fallback: str = "sample",
    max_length: int = 48,
) -> str:
    sanitized = _sanitize_workspace_name(value)
    if len(sanitized) <= max_length:
        return sanitized

    raw_value = str(value or "").strip() or fallback
    digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:10]
    prefix_length = max(12, max_length - len(digest) - 1)
    prefix = sanitized[:prefix_length].rstrip("._-")
    if not prefix:
        return f"{fallback}_{digest}"
    return f"{prefix}_{digest}"


def _resolve_optional_path(value: Any, *, base: Optional[Path] = None) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return ((base or PROJECT_ROOT) / candidate).resolve()


def _build_episode_artifact_paths(run_workspace: Path, sample_id: str, episode_id: str) -> Dict[str, Path]:
    sample_dir_name = _compact_workspace_name(sample_id, fallback="sample", max_length=40)
    episode_dir_name = _compact_workspace_name(episode_id, fallback="episode", max_length=64)
    root = Path(run_workspace).resolve() / "episodes" / sample_dir_name / episode_dir_name
    artifact_paths = _build_query_run_artifact_paths(root)
    for key in ("workspace", "inputs_dir", "logs_dir", "results_dir"):
        artifact_paths[key].mkdir(parents=True, exist_ok=True)
    return artifact_paths


def _update_json_file(path: Path, updater: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    resolved_path = Path(path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "a+", encoding="utf-8") as handle:
        lock_module = None
        try:
            import fcntl as lock_module  # type: ignore

            lock_module.flock(handle.fileno(), lock_module.LOCK_EX)
        except Exception:
            lock_module = None
        try:
            handle.seek(0)
            raw_payload = handle.read().strip()
            payload: Dict[str, Any] = {}
            if raw_payload:
                try:
                    loaded = json.loads(raw_payload)
                    if isinstance(loaded, dict):
                        payload = loaded
                except Exception:
                    payload = {}
            updated_payload = updater(dict(payload))
            handle.seek(0)
            handle.truncate()
            json.dump(updated_payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
            return updated_payload
        finally:
            if lock_module is not None:
                try:
                    lock_module.flock(handle.fileno(), lock_module.LOCK_UN)
                except Exception:
                    pass


def _summarize_launcher_result_episodes(episodes: Any) -> Dict[str, Any]:
    if not isinstance(episodes, list):
        episodes = []
    summary = {
        "total": len(episodes),
        "running": 0,
        "completed": 0,
        "failed": 0,
        "run_successful": 0,
    }
    for item in episodes:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "") or "").strip().lower()
        if status in summary:
            summary[status] += 1
        if bool(item.get("run_success", item.get("success"))):
            summary["run_successful"] += 1
    return summary


def _build_launcher_rollout_stats(reward_features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    features = reward_features if isinstance(reward_features, dict) else {}
    summary = {
        "time_budget_sec": features.get("time_budget_sec"),
        "root_step_budget": features.get("root_step_budget"),
        "subagent_step_budget": features.get("subagent_step_budget"),
        "generation_call_count": features.get("generation_call_count"),
        "planner_generation_turns": features.get("planner_generation_turns"),
        "subagent_generation_turns": features.get("subagent_generation_turns"),
        "prompt_token_count": features.get("prompt_token_count"),
        "response_token_count": features.get("response_token_count"),
        "grounding_text_chars": features.get("grounding_text_chars"),
        "elapsed_seconds": features.get("elapsed_seconds"),
    }
    return {
        key: _json_safe(value)
        for key, value in summary.items()
        if value not in (None, "", [], {})
    }


def _build_launcher_tool_layer_summary(layer_summary: Any) -> Dict[str, Any]:
    if not isinstance(layer_summary, dict):
        return {}
    summary = {
        "unique_tool_count": layer_summary.get("unique_tool_count"),
        "unique_tools": list(layer_summary.get("unique_tools") or []),
        "tool_usage_summary": dict(layer_summary.get("tool_usage_summary") or {}),
        "tool_family_counts": dict(layer_summary.get("tool_family_counts") or {}),
        "tool_type_counts": dict(layer_summary.get("tool_type_counts") or {}),
        "base_meta_coverage": dict(
            layer_summary.get("base_meta_coverage")
            or layer_summary.get("biz_meta_coverage")
            or {}
        ),
    }
    return {
        key: _json_safe(value)
        for key, value in summary.items()
        if value not in (None, "", [], {})
    }


def _build_launcher_execution_summary(
    *,
    run_result: Optional[Dict[str, Any]],
    reward_features: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = run_result.get("metrics", {}) if isinstance(run_result, dict) else {}
    preprocessing = run_result.get("preprocessing", {}) if isinstance(run_result, dict) else {}
    acceptance_metrics = metrics.get("acceptance_metrics", {}) if isinstance(metrics, dict) else {}
    tool_usage_layers = metrics.get("tool_usage_layers", {}) if isinstance(metrics, dict) else {}
    summary = {
        "num_segments": preprocessing.get("num_segments"),
        "planner_iterations": metrics.get("planner_iterations"),
        "subagent_iterations": metrics.get("subagent_iterations"),
        "delegated_root_steps": metrics.get("delegated_root_steps"),
        "decomposed_root_steps": metrics.get("decomposed_root_steps"),
        "subagent_decomposition_rate": metrics.get("subagent_decomposition_rate"),
        "parallel_concrete_batches": metrics.get("parallel_concrete_batches"),
        "parallel_concrete_batch_rate": metrics.get("parallel_concrete_batch_rate"),
        "tool_layers": {
            "planner_proposed": _build_launcher_tool_layer_summary(tool_usage_layers.get("planner_proposed")),
            "subagent_rewritten": _build_launcher_tool_layer_summary(tool_usage_layers.get("subagent_rewritten")),
            "runtime_executed": _build_launcher_tool_layer_summary(tool_usage_layers.get("runtime_executed")),
        },
        "acceptance": dict(acceptance_metrics or {}),
        "cache_stats": metrics.get("cache_stats", {}),
        "timing_breakdown_seconds": metrics.get("timing_breakdown_seconds", {}),
        "elapsed_seconds": (
            reward_features.get("elapsed_seconds")
            if isinstance(reward_features, dict)
            else None
        ),
    }
    return {
        key: _json_safe(value)
        for key, value in summary.items()
        if value not in (None, "", [], {})
    }


def _build_planner_preview(run_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trajectory = run_result.get("agent", {}).get("trajectory") if isinstance(run_result, dict) else {}
    if not isinstance(trajectory, dict):
        return []

    preview: List[Dict[str, Any]] = []
    for step in trajectory.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        if int(step.get("depth", 0) or 0) != 0:
            continue
        preview.append(
            {
                "step_id": step.get("step_id"),
                "action": str(step.get("action", "")).strip(),
                "thought": str(step.get("thought", "")).strip(),
                "plan": str(step.get("plan", "")).strip(),
                "action_args": _json_safe(step.get("action_args", {})),
                "observation": str(step.get("observation", "")).strip(),
            }
        )
        if len(preview) >= 3:
            break
    return [item for item in preview if item]


def _build_launcher_episode_entry(
    *,
    episode_id: str,
    sample_id: str,
    query: str,
    status: str,
    success: bool,
    final_answer: str,
    worker_pid: Optional[int] = None,
    error: str = "",
    video_path: Optional[Path] = None,
    reward_features: Optional[Dict[str, Any]] = None,
    run_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_workspace = str((run_result or {}).get("workspace") or "").strip() if isinstance(run_result, dict) else ""
    artifact_paths = {}
    if run_workspace:
        workspace_path = Path(run_workspace).resolve()
        artifact_paths = {
            "workspace": str(workspace_path),
            "run_log": str((workspace_path / "logs" / "run.log").resolve()),
            "raw_trace_jsonl": str((workspace_path / "logs" / "raw_llm_trace.jsonl").resolve()),
            "run_result": str((workspace_path / "results" / "run_result.json").resolve()),
        }
    entry: Dict[str, Any] = {
        "episode_id": str(episode_id or "").strip(),
        "sample_id": str(sample_id or "").strip(),
        "query": str(query or "").strip(),
        "status": str(status or "").strip(),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "run_success": bool(success),
        "final_answer": str(final_answer or "").strip(),
        "worker_pid": int(worker_pid or 0) or None,
        "video_path": str(video_path.resolve()) if isinstance(video_path, Path) else (str(video_path) if video_path else None),
        "rollout_stats": _build_launcher_rollout_stats(
            reward_features if isinstance(reward_features, dict) else None
        ),
        "execution_summary": _json_safe(
            _build_launcher_execution_summary(
                run_result=run_result if isinstance(run_result, dict) else None,
                reward_features=reward_features if isinstance(reward_features, dict) else None,
            )
        ),
        "planner_preview": _build_planner_preview(run_result if isinstance(run_result, dict) else None),
        "artifact_paths": artifact_paths,
    }
    if error:
        entry["error"] = str(error)
    return entry


def _build_episode_run_id(sample: Dict[str, Any]) -> str:
    sample_id = _compact_workspace_name(sample.get("sample_id") or "sample", fallback="sample", max_length=24)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    return f"{timestamp}_{os.getpid()}_{sample_id}"


def _coerce_episode_options(options: Any) -> List[Dict[str, str]]:
    if isinstance(options, dict):
        return [
            {"label": str(key).strip(), "text": str(value or "").strip()}
            for key, value in options.items()
            if str(value or "").strip()
        ]

    if not isinstance(options, (list, tuple)):
        return []

    normalized: List[Dict[str, str]] = []
    for index, item in enumerate(options):
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("option") or "").strip()
            text = str(item.get("text") or item.get("value") or item.get("content") or "").strip()
        else:
            label = chr(ord("A") + index)
            text = str(item or "").strip()
        if text:
            normalized.append({"label": label, "text": text})
    return normalized


def _build_episode_prompt(question: str, task_type: str, options: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    if task_type:
        lines.append(f"Task Type: {task_type}")
    lines.append(f"Question: {question}")
    if options:
        lines.append("Options:")
        for item in options:
            label = str(item.get("label", "")).strip()
            text = str(item.get("text", "")).strip()
            lines.append(f"{label}. {text}" if label else text)
        lines.append("Choose exactly one best option and justify it with grounded evidence.")
    return "\n".join(line for line in lines if line).strip()


def _resolve_episode_query(sample: Dict[str, Any]) -> str:
    for key in ("episode_query", "submitted_user_query", "query", "prompt", "question"):
        text = str(sample.get(key) or "").strip()
        if text:
            return text

    extra_info = sample.get("extra_info")
    if isinstance(extra_info, dict):
        fallback_question = str(extra_info.get("question") or "").strip()
        if fallback_question:
            task_type = str(sample.get("task_type") or extra_info.get("task_type") or "").strip()
            options = _coerce_episode_options(sample.get("options") or extra_info.get("options"))
            return _build_episode_prompt(fallback_question, task_type, options)

    sample_id = str(sample.get("sample_id") or "").strip() or "<unknown>"
    available_keys = ", ".join(sorted(str(key) for key in sample.keys()))
    raise ValueError(
        "Video-agent RL sample is missing query text. "
        f"sample_id={sample_id}, available_keys=[{available_keys}]"
    )


def _resolve_video_path(sample: Dict[str, Any], workspace_root: Path, rollout_config: DictConfig) -> Path:
    video_value = sample.get("video_path") or sample.get("video")
    if not video_value:
        raise ValueError("Video-agent RL sample is missing `video_path`.")
    sample_id = str(sample.get("sample_id") or "").strip() or "<unknown>"
    video_path = Path(str(video_value)).expanduser()

    candidate_paths: List[Path] = []

    def _append_candidate(candidate: Path) -> None:
        resolved_candidate = candidate.resolve()
        if resolved_candidate not in candidate_paths:
            candidate_paths.append(resolved_candidate)

    def _append_candidate_variants(candidate: Path) -> None:
        _append_candidate(candidate)
        parent = candidate.parent
        if not parent.exists() or not parent.is_dir():
            return

        base_name = candidate.stem if candidate.suffix else candidate.name
        normalized_base_name = base_name.lower()
        matching_files = [path for path in parent.iterdir() if path.is_file() and path.stem.lower() == normalized_base_name]
        preferred_files = [path for path in matching_files if path.suffix.lower() in VIDEO_FILE_SUFFIXES]
        ranked_files = preferred_files or matching_files
        for path in sorted(
            ranked_files,
            key=lambda item: (
                VIDEO_FILE_SUFFIXES.index(item.suffix.lower()) if item.suffix.lower() in VIDEO_FILE_SUFFIXES else len(VIDEO_FILE_SUFFIXES),
                item.name.lower(),
            ),
        ):
            _append_candidate(path)

    if video_path.is_absolute():
        _append_candidate_variants(video_path)
        for candidate in candidate_paths:
            if candidate.is_file():
                return candidate

    root_values: List[Any] = [
        sample.get("dataset_root"),
        sample.get("video_root"),
        rollout_config.get("dataset_root"),
        rollout_config.get("video_root"),
        PROJECT_ROOT,
        workspace_root,
    ]
    for raw_root in root_values:
        if raw_root in (None, ""):
            continue
        root_path = _resolve_path(raw_root, base=PROJECT_ROOT) if not isinstance(raw_root, Path) else raw_root.resolve()
        _append_candidate_variants(_resolve_path(video_path, base=root_path))
        bench_root = root_path / "Bench"
        if bench_root != root_path:
            _append_candidate_variants(_resolve_path(video_path, base=bench_root))

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate

    searched_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        "Unable to resolve video file for video-agent RL sample. "
        f"sample_id={sample_id}, video_path={video_value}, searched=[{searched_paths}]"
    )


@dataclass
class GenerationRecord:
    role: str
    prompt_token_count: int
    response_token_count: int
    prompt_delta_token_count: int
    response_start: int
    response_end: int
    prompt_preview: str
    response_preview: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "prompt_token_count": self.prompt_token_count,
            "response_token_count": self.response_token_count,
            "prompt_delta_token_count": self.prompt_delta_token_count,
            "response_start": self.response_start,
            "response_end": self.response_end,
            "prompt_preview": self.prompt_preview,
            "response_preview": self.response_preview,
        }


@dataclass
class EpisodeArtifacts:
    prompt_ids: List[int]
    response_ids: List[int]
    loss_mask: List[int]
    final_answer: str
    success: bool
    trajectory: Dict[str, Any]
    run_result: Dict[str, Any]
    reward_features: Dict[str, Any]
    conversation_messages: List[Dict[str, Any]] = field(default_factory=list)
    generation_records: List[Dict[str, Any]] = field(default_factory=list)


class EpisodeRecorder:
    def __init__(self) -> None:
        self.initial_prompt_ids: List[int] = []
        self.transcript_ids: List[int] = []
        self.loss_mask: List[int] = []
        self.records: List[GenerationRecord] = []
        self.conversation_messages: List[Dict[str, Any]] = []

    def record_generation(
        self,
        *,
        role: str,
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        prompt_preview: str,
        response_preview: str,
        prompt_messages: Optional[Sequence[Dict[str, Any]]] = None,
        response_text: str = "",
    ) -> None:
        prompt_token_ids = [int(token_id) for token_id in prompt_ids]
        response_token_ids = [int(token_id) for token_id in response_ids]
        normalized_prompt_messages = _copy_messages(prompt_messages or [])
        prompt_delta_count = 0

        if not self.initial_prompt_ids:
            self.initial_prompt_ids = list(prompt_token_ids)
            self.transcript_ids = list(prompt_token_ids)
            self.conversation_messages = normalized_prompt_messages
        else:
            prefix_length = _longest_common_prefix(self.transcript_ids, prompt_token_ids)
            prompt_delta = prompt_token_ids[prefix_length:]
            prompt_delta_count = len(prompt_delta)
            if prompt_delta:
                self.transcript_ids.extend(prompt_delta)
                self.loss_mask.extend([0] * len(prompt_delta))
            if normalized_prompt_messages:
                prompt_prefix = _longest_common_message_prefix(
                    self.conversation_messages,
                    normalized_prompt_messages,
                )
                if prompt_prefix < len(normalized_prompt_messages):
                    self.conversation_messages.extend(
                        _copy_messages(normalized_prompt_messages[prompt_prefix:])
                    )

        response_start = max(0, len(self.transcript_ids) - len(self.initial_prompt_ids))
        if response_token_ids:
            self.transcript_ids.extend(response_token_ids)
            self.loss_mask.extend([1] * len(response_token_ids))
        response_end = response_start + len(response_token_ids)
        self.conversation_messages.append(
            {
                "role": "assistant",
                "content": str(response_text or ""),
            }
        )

        self.records.append(
            GenerationRecord(
                role=role,
                prompt_token_count=len(prompt_token_ids),
                response_token_count=len(response_token_ids),
                prompt_delta_token_count=prompt_delta_count,
                response_start=response_start,
                response_end=response_end,
                prompt_preview=prompt_preview[:400],
                response_preview=response_preview[:400],
            )
        )

    def finalize(self) -> tuple[List[int], List[int], List[int]]:
        prompt_ids = list(self.initial_prompt_ids)
        response_ids = list(self.transcript_ids[len(prompt_ids):])
        loss_mask = list(self.loss_mask)
        if len(response_ids) != len(loss_mask):
            raise RuntimeError("Episode recorder produced misaligned response_ids and loss_mask.")
        return prompt_ids, response_ids, loss_mask

    def build_conversation_messages(self) -> List[Dict[str, Any]]:
        return _copy_messages(self.conversation_messages)

    def build_stats(self) -> Dict[str, Any]:
        planner_turns = sum(1 for item in self.records if item.role == "planner")
        subagent_turns = sum(1 for item in self.records if item.role == "subagent")
        return {
            "generation_call_count": len(self.records),
            "planner_generation_turns": planner_turns,
            "subagent_generation_turns": subagent_turns,
            "prompt_token_count": len(self.initial_prompt_ids),
            "response_token_count": max(0, len(self.transcript_ids) - len(self.initial_prompt_ids)),
        }


def _log_llm_activity_summary(recorder: Optional[EpisodeRecorder]) -> None:
    if recorder is None:
        logger.info("[LLM_ACTIVITY_SUMMARY] Recorder unavailable.")
        logger.info("[Subagent][STATUS] Recorder unavailable; subagent activity could not be summarized.")
        return

    stats = recorder.build_stats()
    logger.info("[LLM_ACTIVITY_SUMMARY] %s", json.dumps(stats, ensure_ascii=False))

    planner_turns = int(stats.get("planner_generation_turns") or 0)
    subagent_turns = int(stats.get("subagent_generation_turns") or 0)
    logger.info("[Planner][STATUS] planner_generation_turns=%d", planner_turns)

    if subagent_turns <= 0:
        logger.info("[Subagent][STATUS] No subagent LLM turns were executed in this episode.")
        return

    subagent_previews: List[Dict[str, Any]] = []
    for turn_index, record in enumerate(recorder.records, start=1):
        if record.role != "subagent":
            continue
        subagent_previews.append(
            {
                "turn_index": turn_index,
                "prompt_tokens": int(record.prompt_token_count),
                "response_tokens": int(record.response_token_count),
                "prompt_preview": str(record.prompt_preview or "").strip(),
                "response_preview": str(record.response_preview or "").strip(),
            }
        )

    logger.info("[Subagent][STATUS] subagent_generation_turns=%d", subagent_turns)
    logger.info(
        "[Subagent][TURN_PREVIEW_BEGIN]\n%s\n[Subagent][TURN_PREVIEW_END]",
        json.dumps(subagent_previews, ensure_ascii=False, indent=2),
    )


class RolloutChatAdapter(BaseLLM):
    supports_internal_generation_heartbeat = True

    def __init__(
        self,
        *,
        tokenizer: Any,
        model_name: str,
        role: str,
        recorder: EpisodeRecorder,
        max_context_length: int,
        response_generator: Optional[Callable[..., Dict[str, Any]]] = None,
        delegate_llm: Optional[BaseLLM] = None,
        generation_session_id: str = "",
    ) -> None:
        super().__init__(model_name=model_name)
        self.tokenizer = tokenizer
        self.role = role
        self.recorder = recorder
        self.max_context_length = max(1, int(max_context_length))
        self.response_generator = response_generator
        self.delegate_llm = delegate_llm
        self.generation_session_id = str(generation_session_id or "").strip()
        if self.delegate_llm is None and not callable(self.response_generator):
            raise ValueError("RolloutChatAdapter requires either response_generator or delegate_llm.")
        # RL rollout uses a local in-process actor instead of a remote/provider API.
        # Running local CUDA generation inside planner.engine's helper thread has
        # proven unstable, so this adapter requests inline generation and emits
        # its own progress heartbeats from the rollout loop.
        self.supports_internal_generation_heartbeat = self.delegate_llm is None

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _normalize_multimodal_messages(messages)

    def _default_enable_thinking(self) -> Optional[bool]:
        normalized_role = str(self.role or "").strip().lower()
        if normalized_role == "planner":
            return True
        if normalized_role == "subagent":
            return False
        config = settings.get_generation_config(self.role) or {}
        if "enable_thinking" not in config:
            return None
        return bool(config.get("enable_thinking"))

    def _build_prompt_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        normalized = _ensure_user_message(self._normalize_messages(messages))
        if not normalized:
            normalized = [{"role": "user", "content": ""}]
        text_messages = []
        for message in normalized:
            text_messages.append(
                {
                    "role": str(message.get("role", "user")).strip() or "user",
                    "content": _flatten_content(message.get("content")),
                }
            )
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = _apply_chat_template_with_optional_thinking(
                self.tokenizer,
                text_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            if rendered is not None:
                return rendered
        return _render_text_chat(text_messages, add_generation_prompt=True)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        generation_kwargs = dict(kwargs or {})
        generation_deadline = generation_kwargs.get("generation_deadline_monotonic")
        wall_clock_deadline = generation_kwargs.get("wall_clock_deadline_monotonic")
        normalized_role = str(self.role or "").strip().lower()
        requested_enable_thinking = generation_kwargs.get("enable_thinking")
        if normalized_role == "planner" and requested_enable_thinking is None:
            generation_kwargs["enable_thinking"] = True
        if normalized_role == "subagent":
            generation_kwargs["enable_thinking"] = False

        def _remaining_seconds_text(raw_deadline: Any) -> str:
            if raw_deadline in (None, ""):
                return "unbounded"
            try:
                return f"{float(raw_deadline) - time.perf_counter():.3f}s"
            except Exception:
                return "invalid"

        logger.info(
            "[RolloutChatAdapter][DIAG] role=%s model_name=%s message_count=%d max_tokens=%s temperature=%s supports_internal_generation_heartbeat=%s has_progress_callback=%s generation_deadline_remaining=%s wall_clock_remaining=%s requested_enable_thinking=%s effective_enable_thinking=%s extra_generation_keys=%s",
            self.role,
            self.model_name,
            len(messages or []),
            max_tokens,
            temperature,
            bool(getattr(self, "supports_internal_generation_heartbeat", False)),
            callable(generation_kwargs.get("progress_callback")),
            _remaining_seconds_text(generation_deadline),
            _remaining_seconds_text(wall_clock_deadline),
            requested_enable_thinking,
            generation_kwargs.get("enable_thinking"),
            sorted(
                key for key in generation_kwargs.keys()
                if key not in {
                    "progress_callback",
                    "generation_deadline_monotonic",
                    "wall_clock_deadline_monotonic",
                    "generation_heartbeat_interval_seconds",
                }
            ),
        )
        started = time.perf_counter()
        prompt_messages = self._normalize_messages(messages)
        prompt_preview = ""
        response_text = ""
        prompt_ids: List[int] = []
        response_ids: List[int] = []

        if self.delegate_llm is not None:
            prompt_preview = self._build_prompt_text(
                messages,
                enable_thinking=self._default_enable_thinking(),
            )
            try:
                response_text = str(
                    self.delegate_llm.generate(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **generation_kwargs,
                    )
                    or ""
                )
            except Exception:
                logger.exception(
                    "[RolloutChatAdapter][DIAG] delegate_llm.generate failed role=%s model_name=%s after %.3fs",
                    self.role,
                    self.model_name,
                    time.perf_counter() - started,
                )
                raise
            if normalized_role == "subagent":
                response_text = _strip_think_blocks(response_text)
            prompt_ids = self.tokenizer.encode(prompt_preview, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        else:
            try:
                generation_payload = self.response_generator(
                    messages=messages,
                    role=self.role,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    generation_session_id=self.generation_session_id,
                    **generation_kwargs,
                )
            except Exception as exc:
                if isinstance(exc, AcceleratorFatalError) or is_accelerator_fatal_error(exc):
                    logger.exception(
                        "[RolloutChatAdapter][FATAL] response_generator hit a fatal accelerator error role=%s model_name=%s after %.3fs; propagating to abort this worker instead of continuing with a poisoned CUDA context.",
                        self.role,
                        self.model_name,
                        time.perf_counter() - started,
                    )
                    raise AcceleratorFatalError(str(exc)) from exc
                logger.exception(
                    "[RolloutChatAdapter][DIAG] response_generator failed role=%s model_name=%s after %.3fs",
                    self.role,
                    self.model_name,
                    time.perf_counter() - started,
                )
                raise
            prompt_ids = [
                int(token_id)
                for token_id in generation_payload.get("prompt_ids", [])
            ]
            response_ids = [
                int(token_id)
                for token_id in generation_payload.get("response_ids", [])
            ]
            response_text = _decode_text(self.tokenizer, response_ids)
            if normalized_role == "subagent":
                cleaned_response_text = _strip_think_blocks(response_text)
                if cleaned_response_text != response_text:
                    response_text = cleaned_response_text
                    response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            prompt_preview = str(generation_payload.get("prompt_preview", ""))
            prompt_messages = generation_payload.get("prompt_messages") or prompt_messages
            stop_reason = str(generation_payload.get("stop_reason") or "").strip()
            if bool(generation_payload.get("timed_out", False)):
                elapsed_seconds = max(0.0, time.perf_counter() - started)
                logger.warning(
                    "[RolloutChatAdapter][TIMEOUT] role=%s stop_reason=%s elapsed=%.3fs partial_tokens=%d partial_chars=%d",
                    self.role,
                    stop_reason or "model_generate_timeout",
                    elapsed_seconds,
                    len(response_ids),
                    len(response_text),
                )
                raise GenerationTimeoutError(
                    "Rollout model.generate stopped at the per-turn deadline before producing a complete response.",
                    partial_response=response_text,
                    elapsed_seconds=elapsed_seconds,
                    role=normalized_role or self.role,
                    phase="model_generate",
                )

        logger.info(
            "[RolloutChatAdapter][DIAG] role=%s generation_completed duration=%.3fs prompt_tokens=%d response_tokens=%d response_chars=%d",
            self.role,
            time.perf_counter() - started,
            len(prompt_ids),
            len(response_ids),
            len(response_text),
        )
        self.recorder.record_generation(
            role=self.role,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_preview=prompt_preview,
            response_preview=response_text,
            prompt_messages=prompt_messages,
            response_text=response_text,
        )
        return response_text

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self.generate(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(str(text or ""), add_special_tokens=False))

    def count_message_tokens(self, messages: List[Dict[str, Any]], **_: Any) -> int:
        prompt_text = self._build_prompt_text(
            messages,
            enable_thinking=self._default_enable_thinking(),
        )
        return len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

    def get_max_context_length(self) -> int:
        return self.max_context_length


class LazyLLMDelegate(BaseLLM):
    """Delay provider construction until the first real generation call."""

    def __init__(
        self,
        *,
        model_name: str,
        model_role: str,
        factory: Callable[[], BaseLLM],
    ) -> None:
        super().__init__(model_name=model_name)
        self.model_role = str(model_role or "").strip()
        self._factory = factory
        self._delegate: Optional[BaseLLM] = None
        self._lock = threading.Lock()

    def _get_delegate(self) -> BaseLLM:
        if self._delegate is not None:
            return self._delegate
        with self._lock:
            if self._delegate is None:
                started = time.perf_counter()
                logger.info(
                    "[LazyLLMDelegate][DIAG] initializing role=%s model_name=%s",
                    self.model_role,
                    self.model_name,
                )
                self._delegate = self._factory()
                self.model_name = str(getattr(self._delegate, "model_name", self.model_name) or self.model_name)
                logger.info(
                    "[LazyLLMDelegate][DIAG] initialized role=%s model_name=%s duration=%.3fs delegate_type=%s",
                    self.model_role,
                    self.model_name,
                    time.perf_counter() - started,
                    type(self._delegate).__name__,
                )
        return self._delegate

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        return self._get_delegate().generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self.generate(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def count_tokens(self, text: str) -> int:
        delegate = self._delegate
        if delegate is not None:
            return delegate.count_tokens(text)
        return super().count_tokens(text)

    def count_message_tokens(self, messages: List[Dict[str, str]], **kwargs: Any) -> int:
        delegate = self._delegate
        if delegate is not None:
            return delegate.count_message_tokens(messages, **kwargs)
        return super().count_message_tokens(messages, **kwargs)

    def get_max_context_length(self) -> int:
        delegate = self._delegate
        if delegate is not None:
            return delegate.get_max_context_length()
        return super().get_max_context_length()

    def prepare(self) -> None:
        delegate = self._get_delegate()
        prepare_fn = getattr(delegate, "prepare", None)
        if callable(prepare_fn):
            prepare_fn()


class VideoAgentEpisodeRunner:
    def __init__(
        self,
        *,
        tokenizer: Any,
        rollout_config: DictConfig,
        response_generator: Callable[..., Dict[str, Any]],
        preprocess_cache: Optional[Dict[str, Dict[str, Any]]] = None,
        preprocess_cache_max_entries: int = 2,
        enable_preprocess_payload_memory_cache: bool = True,
        accelerator_lock: Any = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.rollout_config = rollout_config
        self.response_generator = response_generator
        self._accelerator_lock = accelerator_lock
        self._preprocess_cache: Dict[str, Dict[str, Any]] = (
            preprocess_cache if preprocess_cache is not None else {}
        )
        self._preprocess_cache_max_entries = max(0, int(preprocess_cache_max_entries))
        self._enable_preprocess_payload_memory_cache = bool(
            enable_preprocess_payload_memory_cache and self._preprocess_cache_max_entries > 0
        )
        self.max_context_length = int(
            _coerce_int(rollout_config.prompt_length, 2048)
            + _coerce_int(rollout_config.response_length, 4096)
        )
        self._independent_subagent_service_llm: Optional[BaseLLM] = None
        self._independent_subagent_service_model_name: str = ""
        self._interaction_config_path = self._resolve_interaction_config_path()
        self._interaction_map: Dict[str, BaseInteraction] = {}
        if self._interaction_config_path:
            self._interaction_map = self._initialize_interaction_map(self._interaction_config_path)

    def _resolve_interaction_config_path(self) -> Optional[str]:
        configured_path = self.rollout_config.get("interaction_config_path")
        multi_turn_config = self.rollout_config.get("multi_turn")
        if configured_path in (None, "") and hasattr(multi_turn_config, "get"):
            try:
                configured_path = multi_turn_config.get("interaction_config_path")
            except Exception:
                configured_path = None
        configured_text = str(configured_path or "").strip()
        if not configured_text or configured_text.lower() in {"none", "null"}:
            return None
        resolved_path = _resolve_optional_path(configured_text, base=PROJECT_ROOT)
        return str(resolved_path) if resolved_path is not None else None

    def _initialize_interaction_map(self, interaction_config_path: str) -> Dict[str, BaseInteraction]:
        interaction_map = initialize_interactions_from_config(interaction_config_path)
        if not isinstance(interaction_map, dict):
            raise ValueError(
                "Interaction config initialization must return a mapping from interaction names to interaction instances."
            )
        logger.info(
            "[Episode][Interaction][DIAG] initialized interaction_map from %s with names=%s",
            interaction_config_path,
            sorted(str(name) for name in interaction_map.keys()),
        )
        return interaction_map

    def _resolve_sample_interaction_kwargs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        interaction_kwargs = sample.get("interaction_kwargs")
        if not isinstance(interaction_kwargs, dict):
            extra_info = sample.get("extra_info")
            if isinstance(extra_info, dict):
                interaction_kwargs = extra_info.get("interaction_kwargs")
        if not isinstance(interaction_kwargs, dict):
            return {}
        return dict(interaction_kwargs)

    def _start_episode_interaction(
        self,
        *,
        sample: Dict[str, Any],
        episode_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._interaction_map:
            return None

        interaction_kwargs = self._resolve_sample_interaction_kwargs(sample)
        if not interaction_kwargs:
            return None

        interaction_name = str(interaction_kwargs.get("name") or "").strip()
        if not interaction_name:
            raise ValueError(
                "interaction_kwargs.name is required when interaction-system is enabled via interaction_config_path."
            )
        if interaction_name not in self._interaction_map:
            raise ValueError(
                f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: {list(self._interaction_map.keys())}"
            )

        interaction = self._interaction_map[interaction_name]
        interaction_instance_id = f"{episode_id}_interaction"
        _run_async_blocking(interaction.start_interaction, interaction_instance_id, **interaction_kwargs)
        logger.info(
            "[Episode][Interaction][DIAG] started interaction name=%s instance_id=%s",
            interaction_name,
            interaction_instance_id,
        )
        return {
            "interaction": interaction,
            "interaction_name": interaction_name,
            "interaction_instance_id": interaction_instance_id,
            "interaction_kwargs": interaction_kwargs,
            "turn_records": [],
            "last_turn_key": "",
        }

    def _run_episode_interaction(
        self,
        *,
        interaction_context: Dict[str, Any],
        messages: List[Dict[str, Any]],
        turn_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        interaction = interaction_context["interaction"]
        interaction_name = str(interaction_context.get("interaction_name") or "").strip()
        interaction_instance_id = str(interaction_context.get("interaction_instance_id") or "").strip()
        interaction_kwargs = dict(interaction_context.get("interaction_kwargs") or {})
        normalized_turn_event = dict(turn_event or {})
        turn_key_parts = [interaction_name]
        for field in ("role", "depth", "iteration", "phase", "status"):
            value = normalized_turn_event.get(field)
            if value in (None, ""):
                continue
            turn_key_parts.append(f"{field}={value}")
        turn_key = "|".join(turn_key_parts)
        if turn_key and str(interaction_context.get("last_turn_key") or "") == turn_key:
            return {}
        should_terminate_sequence, interaction_response, reward, metrics = _run_async_blocking(
            interaction.generate_response,
            interaction_instance_id,
            messages,
            **interaction_kwargs,
        )
        try:
            reward_value = float(reward) if reward is not None else None
        except Exception:
            reward_value = None
        metrics_dict = dict(metrics) if isinstance(metrics, dict) else {}
        response_preview = _truncate_progress_text(interaction_response, limit=220)
        interaction_record = {
            "interaction_enabled": True,
            "interaction_name": interaction_name,
            "interaction_should_terminate_sequence": bool(should_terminate_sequence),
            "interaction_turn_reward": reward_value,
            "interaction_response_preview": response_preview,
            "interaction_metrics": metrics_dict,
            "interaction_turn_role": str(normalized_turn_event.get("role") or ""),
            "interaction_turn_depth": normalized_turn_event.get("depth"),
            "interaction_turn_iteration": normalized_turn_event.get("iteration"),
            "interaction_turn_phase": str(normalized_turn_event.get("phase") or ""),
            "interaction_turn_status": str(normalized_turn_event.get("status") or ""),
        }
        if turn_key:
            interaction_context["last_turn_key"] = turn_key
        turn_records = interaction_context.setdefault("turn_records", [])
        if isinstance(turn_records, list):
            turn_records.append(dict(interaction_record))
        logger.info(
            "[Episode][Interaction][DIAG] name=%s turn_key=%s should_terminate=%s reward=%s metrics=%s response_preview=%s",
            interaction_name,
            turn_key,
            bool(should_terminate_sequence),
            reward_value,
            json.dumps(metrics_dict, ensure_ascii=False, default=str),
            response_preview,
        )
        if bool(should_terminate_sequence):
            logger.info(
                "[Episode][Interaction][DIAG] interaction requested terminate at turn-level; ignored to preserve planner execution boundary."
            )
        return interaction_record

    def _build_episode_interaction_summary(self, interaction_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not interaction_context:
            return {}
        interaction_name = str(interaction_context.get("interaction_name") or "").strip()
        raw_turn_records = interaction_context.get("turn_records")
        turn_records = [dict(item) for item in (raw_turn_records or []) if isinstance(item, dict)]
        if not turn_records:
            return {
                "interaction_enabled": True,
                "interaction_name": interaction_name,
                "interaction_turn_count": 0,
                "interaction_should_terminate_sequence": False,
                "interaction_turn_reward": None,
                "interaction_turn_reward_total": 0.0,
                "interaction_turn_reward_avg": 0.0,
                "interaction_response_preview": "",
                "interaction_metrics": {},
                "interaction_turn_records": [],
            }

        reward_values: List[float] = []
        for record in turn_records:
            reward_value = record.get("interaction_turn_reward")
            if isinstance(reward_value, (int, float)):
                reward_values.append(float(reward_value))

        reward_total = float(sum(reward_values)) if reward_values else 0.0
        reward_avg = (reward_total / len(reward_values)) if reward_values else 0.0
        last_record = dict(turn_records[-1])
        should_terminate_sequence = any(
            bool(record.get("interaction_should_terminate_sequence", False))
            for record in turn_records
        )
        return {
            "interaction_enabled": True,
            "interaction_name": interaction_name,
            "interaction_turn_count": len(turn_records),
            "interaction_should_terminate_sequence": bool(should_terminate_sequence),
            "interaction_turn_reward": last_record.get("interaction_turn_reward"),
            "interaction_turn_reward_total": round(reward_total, 6),
            "interaction_turn_reward_avg": round(reward_avg, 6),
            "interaction_response_preview": str(last_record.get("interaction_response_preview") or ""),
            "interaction_metrics": _json_safe(last_record.get("interaction_metrics", {})),
            "interaction_turn_records": _json_safe(turn_records[-8:]),
        }

    def _finalize_episode_interaction(self, interaction_context: Optional[Dict[str, Any]]) -> None:
        if not interaction_context:
            return
        interaction = interaction_context.get("interaction")
        interaction_instance_id = str(interaction_context.get("interaction_instance_id") or "").strip()
        interaction_name = str(interaction_context.get("interaction_name") or "").strip()
        interaction_kwargs = dict(interaction_context.get("interaction_kwargs") or {})
        if interaction is None or not interaction_instance_id:
            return
        try:
            _run_async_blocking(interaction.finalize_interaction, interaction_instance_id, **interaction_kwargs)
            logger.info(
                "[Episode][Interaction][DIAG] finalized interaction name=%s instance_id=%s",
                interaction_name,
                interaction_instance_id,
            )
        except Exception as exc:
            logger.warning(
                "[Episode][Interaction][DIAG] failed to finalize interaction name=%s instance_id=%s error=%s",
                interaction_name,
                interaction_instance_id,
                exc,
            )

    def _independent_subagent_service_enabled(self) -> bool:
        return _coerce_bool(self.rollout_config.get("enable_independent_subagent_service"), True)

    def _independent_subagent_service_allow_fallback(self) -> bool:
        return _coerce_bool(
            self.rollout_config.get("independent_subagent_service_allow_fallback_to_rollout_actor"),
            False,
        )

    def _resolve_independent_subagent_service_model_name(self) -> str:
        configured_value = str(self.rollout_config.get("independent_subagent_service_model_name") or "").strip()
        if configured_value:
            return configured_value
        return str(settings.subagent_model_path)

    def _should_disable_independent_subagent_service_for_local_process(self, model_name: Optional[str] = None) -> bool:
        resolved_model_name = str(model_name or self._resolve_independent_subagent_service_model_name() or "").strip()
        if not resolved_model_name:
            return False

        provider_value = (
            os.environ.get("TOOL_STAR_SUBAGENT_SERVICE_MODEL_PROVIDER")
            or os.environ.get("TOOL_STAR_SUBAGENT_MODEL_PROVIDER")
            or os.environ.get("TOOL_STAR_NON_PLANNER_MODEL_PROVIDER")
            or os.environ.get("TOOL_STAR_AUX_MODEL_PROVIDER")
            or getattr(settings, "non_planner_model_provider", "auto")
        )
        normalized_provider = str(provider_value or "").strip().lower()
        provider_aliases = {
            "": "auto",
            "local_process": "process",
            "subprocess": "process",
            "service": "openai",
            "sidecar": "openai",
            "remote": "openai",
            "api": "openai",
        }
        normalized_provider = provider_aliases.get(normalized_provider, normalized_provider)
        if normalized_provider != "process":
            return False

        try:
            resolved_subagent_model = str(Path(resolved_model_name).expanduser().resolve())
        except Exception:
            resolved_subagent_model = resolved_model_name
        try:
            resolved_planner_model = str(Path(str(settings.planner_model_path)).expanduser().resolve())
        except Exception:
            resolved_planner_model = str(settings.planner_model_path)
        return resolved_subagent_model == resolved_planner_model

    def _get_independent_subagent_service_llm(self) -> BaseLLM:
        model_name = self._resolve_independent_subagent_service_model_name()
        if (
            self._independent_subagent_service_llm is not None
            and self._independent_subagent_service_model_name == model_name
        ):
            return self._independent_subagent_service_llm

        llm = get_llm_provider(
            model_name=model_name,
            # Use a dedicated role key so the provider cache does not share
            # state with planner when both point to the same model path.
            model_role="subagent_service",
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            enable_thinking=False,
            max_tokens=int(getattr(settings, "subagent_max_generation_tokens", 2048) or 2048),
        )
        self._independent_subagent_service_llm = llm
        self._independent_subagent_service_model_name = model_name
        return llm

    def _prepare_delegate_llm(
        self,
        delegate_llm: Optional[BaseLLM],
        *,
        role: str,
        episode_id: str,
    ) -> None:
        if delegate_llm is None:
            return
        prepare_fn = getattr(delegate_llm, "prepare", None)
        if not callable(prepare_fn):
            return
        started = time.perf_counter()
        logger.info(
            "[Episode][DelegatePrepare][DIAG] begin role=%s episode_id=%s delegate_type=%s model_name=%s",
            role,
            episode_id,
            type(delegate_llm).__name__,
            str(getattr(delegate_llm, "model_name", "") or "<unknown>"),
        )
        prepare_fn()
        logger.info(
            "[Episode][DelegatePrepare][DIAG] done role=%s episode_id=%s delegate_type=%s duration=%.3fs",
            role,
            episode_id,
            type(delegate_llm).__name__,
            time.perf_counter() - started,
        )

    def _get_cached_preprocessing_payload(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self._enable_preprocess_payload_memory_cache:
            return None
        cached_payload = self._preprocess_cache.get(cache_key)
        if cached_payload is None:
            return None
        move_to_end = getattr(self._preprocess_cache, "move_to_end", None)
        if callable(move_to_end):
            try:
                move_to_end(cache_key)
            except Exception:
                pass
        return cached_payload

    def _set_cached_preprocessing_payload(self, cache_key: str, payload: Dict[str, Any]) -> None:
        if not self._enable_preprocess_payload_memory_cache:
            return
        self._preprocess_cache[cache_key] = payload
        move_to_end = getattr(self._preprocess_cache, "move_to_end", None)
        if callable(move_to_end):
            try:
                move_to_end(cache_key)
            except Exception:
                pass

        while len(self._preprocess_cache) > self._preprocess_cache_max_entries:
            popitem = getattr(self._preprocess_cache, "popitem", None)
            if callable(popitem):
                try:
                    popitem(last=False)
                    continue
                except TypeError:
                    pass
                except Exception:
                    break
            try:
                oldest_key = next(iter(self._preprocess_cache))
            except StopIteration:
                break
            self._preprocess_cache.pop(oldest_key, None)

    def _build_shared_preprocess_workspace(self, sample: Dict[str, Any]) -> Path:
        # Match run_e2e semantics: query workspaces can stay sample-scoped, but
        # video preprocessing artifacts must live under the shared workspace root
        # so multiple questions over the same video reuse the same cache.
        return _resolve_workspace_root(sample, self.rollout_config)

    def _resolve_launcher_run_dir(self) -> Optional[Path]:
        return _resolve_optional_path(self.rollout_config.get("launcher_run_dir"), base=PROJECT_ROOT)

    def _resolve_launcher_log_path(self) -> Optional[Path]:
        return _resolve_optional_path(self.rollout_config.get("launcher_log_path"), base=PROJECT_ROOT)

    def _resolve_launcher_result_path(self) -> Optional[Path]:
        return _resolve_optional_path(self.rollout_config.get("launcher_result_path"), base=PROJECT_ROOT)

    def _distributed_rollout_stage_sync_enabled(self) -> bool:
        raw_value = self.rollout_config.get("enable_pre_planner_distributed_barrier", False)
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        return bool(raw_value)

    def _synchronize_distributed_rollout_stage(self, *, stage: str) -> None:
        stage_name = str(stage or "unknown").strip() or "unknown"
        if not self._distributed_rollout_stage_sync_enabled():
            return
        if not torch.cuda.is_available():
            return
        dist = getattr(torch, "distributed", None)
        if dist is None or not callable(getattr(dist, "is_available", None)) or not dist.is_available():
            return
        if not callable(getattr(dist, "is_initialized", None)) or not dist.is_initialized():
            return
        if not callable(getattr(dist, "get_world_size", None)) or int(dist.get_world_size()) <= 1:
            return

        started = time.perf_counter()
        rank = -1
        world_size = -1
        try:
            if callable(getattr(dist, "get_rank", None)):
                rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
            logger.info(
                "[Episode][DIST_SYNC] entering barrier stage=%s rank=%s world_size=%s",
                stage_name,
                rank,
                world_size,
            )
            dist.barrier()
            logger.info(
                "[Episode][DIST_SYNC] exited barrier stage=%s rank=%s world_size=%s duration=%.3fs",
                stage_name,
                rank,
                world_size,
                time.perf_counter() - started,
            )
        except Exception:
            logger.exception(
                "[Episode][DIST_SYNC] distributed barrier failed stage=%s rank=%s world_size=%s after %.3fs",
                stage_name,
                rank,
                world_size,
                time.perf_counter() - started,
            )
            raise

    def _persist_episode_entry(
        self,
        *,
        launcher_result_path: Optional[Path],
        entry: Dict[str, Any],
    ) -> None:
        if launcher_result_path is None:
            return

        def _updater(payload: Dict[str, Any]) -> Dict[str, Any]:
            episodes = payload.get("episodes")
            if not isinstance(episodes, list):
                episodes = []

            episode_id = str(entry.get("episode_id") or "").strip()
            updated_episodes: List[Dict[str, Any]] = []
            replaced = False
            for item in episodes:
                if not isinstance(item, dict):
                    continue
                if str(item.get("episode_id") or "").strip() == episode_id:
                    replaced_entry = dict(entry)
                    created_at = str(item.get("created_at") or "").strip()
                    if created_at and not str(replaced_entry.get("created_at") or "").strip():
                        replaced_entry["created_at"] = created_at
                    started_at = str(item.get("started_at") or "").strip()
                    if started_at and not str(replaced_entry.get("started_at") or "").strip():
                        replaced_entry["started_at"] = started_at
                    updated_episodes.append(replaced_entry)
                    replaced = True
                else:
                    updated_episodes.append(item)
            if not replaced:
                updated_episodes.append(dict(entry))

            payload["episodes"] = updated_episodes
            payload["episode_summary"] = _summarize_launcher_result_episodes(updated_episodes)
            payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
            return payload

        _update_json_file(launcher_result_path, _updater)

    def _build_episode_reward_features(
        self,
        *,
        episode_id: str,
        sample: Dict[str, Any],
        recorder: EpisodeRecorder,
        elapsed_seconds: float,
        grounding_text: str,
        launcher_result_path: Optional[Path],
        worker_pid: int,
        effective_time_budget_sec: float,
    ) -> Dict[str, Any]:
        runtime_metrics = runtime_session.get_query_metrics()
        cache_stats = dict(runtime_metrics.get("cache_stats") or {})
        preprocess_output_cache_hits = int(cache_stats.get("preprocess.output_dir.hit", 0) or 0)
        preprocess_memory_cache_hits = int(cache_stats.get("rollout.preprocess_payload_memory.hit", 0) or 0)
        original_time_budget_sec = _coerce_float(
            sample.get("time_budget_sec", 0.0),
            0.0,
        )
        return {
            "episode_id": str(episode_id or "").strip(),
            "sample_id": str(sample.get("sample_id") or "").strip(),
            "dataset_name": str(sample.get("dataset_name") or sample.get("data_source") or "").strip(),
            "worker_pid": int(worker_pid),
            "time_budget_sec": round(float(effective_time_budget_sec), 6),
            "original_time_budget_sec": round(float(original_time_budget_sec), 6),
            "root_step_budget": int(settings.max_iterations),
            "subagent_step_budget": int(settings.subagent_max_iterations),
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "grounding_text_chars": len(grounding_text),
            "launcher_result_path": str(launcher_result_path) if launcher_result_path is not None else "",
            "preprocess_output_cache_hit": bool(preprocess_output_cache_hits > 0),
            "preprocess_memory_cache_hit": bool(preprocess_memory_cache_hits > 0),
            "preprocess_cache_reused": bool(preprocess_output_cache_hits > 0 or preprocess_memory_cache_hits > 0),
            **recorder.build_stats(),
        }

    def _build_padding_episode_artifacts(
        self,
        *,
        sample: Dict[str, Any],
        query: str,
        episode_id: str,
        worker_pid: int,
    ) -> EpisodeArtifacts:
        eos_token_id = int(getattr(self.tokenizer, "eos_token_id", None) or getattr(self.tokenizer, "pad_token_id", 0) or 0)
        try:
            prompt_ids = self.tokenizer.encode(str(query or ""), add_special_tokens=False)
        except Exception:
            prompt_ids = []
        if not prompt_ids:
            prompt_ids = [eos_token_id]

        reward_features = {
            "episode_id": str(episode_id or "").strip(),
            "sample_id": str(sample.get("sample_id") or "").strip(),
            "dataset_name": str(sample.get("dataset_name") or sample.get("data_source") or "").strip(),
            "worker_pid": int(worker_pid),
            "time_budget_sec": _coerce_float(sample.get("time_budget_sec", 0.0), 0.0),
            "root_step_budget": int(settings.max_iterations),
            "subagent_step_budget": int(settings.subagent_max_iterations),
            "elapsed_seconds": 0.0,
            "grounding_text_chars": 0,
            "generation_call_count": 0,
            "planner_generation_turns": 0,
            "subagent_generation_turns": 0,
            "prompt_token_count": len(prompt_ids),
            "response_token_count": 0,
            "is_padding_prompt": True,
            "preprocess_cache_reused": False,
        }
        run_result = {
            "video_path": str(sample.get("video_path") or ""),
            "query": str(query or ""),
            "workspace": "",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "preprocessing": {
                "video_name": None,
                "num_segments": 0,
                "processed_output_dir": "",
                "segment_index_sample": [],
            },
            "agent": {
                "final_answer": "",
                "trajectory": {
                    "success": False,
                    "is_padding_prompt": True,
                    "termination_reason": "padding_prompt_skipped",
                },
            },
            "metrics": {
                "planner_iterations": 0,
                "subagent_iterations": 0,
                "delegated_root_steps": 0,
                "decomposed_root_steps": 0,
                "parallel_concrete_batches": 0,
                "parallel_concrete_batch_rate": 0.0,
                "cache_stats": {
                    "rollout.padding_prompt.skip.hit": 1,
                },
                "timing_breakdown_seconds": {},
                "acceptance_metrics": {
                    "acceptance_passed": False,
                    "tool_chain_valid": False,
                    "tool_backed_finish_satisfied": False,
                },
            },
        }
        return EpisodeArtifacts(
            prompt_ids=prompt_ids,
            response_ids=[eos_token_id],
            loss_mask=[0],
            final_answer="",
            success=False,
            trajectory={
                "success": False,
                "is_padding_prompt": True,
                "termination_reason": "padding_prompt_skipped",
            },
            run_result=run_result,
            reward_features=reward_features,
            conversation_messages=[],
            generation_records=[],
        )

    def _build_failed_episode_artifacts(
        self,
        *,
        sample: Dict[str, Any],
        query: str,
        episode_id: str,
        worker_pid: int,
        error_message: str,
        elapsed_seconds: float,
        grounding_text: str,
        recorder: Optional[EpisodeRecorder],
        resolved_video_path: Optional[Path],
        processed_output_dir: Optional[Path],
        run_workspace: Optional[Path],
        trajectory: Optional[Dict[str, Any]] = None,
    ) -> EpisodeArtifacts:
        eos_token_id = int(getattr(self.tokenizer, "eos_token_id", None) or getattr(self.tokenizer, "pad_token_id", 0) or 0)
        prompt_ids: List[int] = []
        conversation_messages: List[Dict[str, Any]] = []
        generation_records: List[Dict[str, Any]] = []
        recorder_stats: Dict[str, Any] = {}
        if recorder is not None:
            try:
                prompt_ids, _, _ = recorder.finalize()
            except Exception:
                prompt_ids = list(getattr(recorder, "initial_prompt_ids", []) or [])
            try:
                conversation_messages = recorder.build_conversation_messages()
            except Exception:
                conversation_messages = []
            try:
                generation_records = [record.to_dict() for record in recorder.records]
            except Exception:
                generation_records = []
            try:
                recorder_stats = dict(recorder.build_stats() or {})
            except Exception:
                recorder_stats = {}

        if not prompt_ids:
            try:
                prompt_ids = self.tokenizer.encode(str(query or ""), add_special_tokens=False)
            except Exception:
                prompt_ids = []
        if not prompt_ids:
            prompt_ids = [eos_token_id]

        failure_trajectory = dict(trajectory or {})
        failure_trajectory.setdefault("success", False)
        failure_trajectory.setdefault("termination_reason", "episode_exception")
        failure_trajectory.setdefault("error_message", str(error_message or "").strip())

        reward_features = {
            "episode_id": str(episode_id or "").strip(),
            "sample_id": str(sample.get("sample_id") or "").strip(),
            "dataset_name": str(sample.get("dataset_name") or sample.get("data_source") or "").strip(),
            "worker_pid": int(worker_pid),
            "time_budget_sec": _coerce_float(sample.get("time_budget_sec", 0.0), 0.0),
            "root_step_budget": int(settings.max_iterations),
            "subagent_step_budget": int(settings.subagent_max_iterations),
            "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 6),
            "grounding_text_chars": len(str(grounding_text or "")),
            "run_failed_soft": True,
            "failure_reason": str(error_message or "").strip(),
            "preprocess_cache_reused": False,
            **recorder_stats,
        }
        reward_features.setdefault("generation_call_count", 0)
        reward_features.setdefault("planner_generation_turns", 0)
        reward_features.setdefault("subagent_generation_turns", 0)
        reward_features.setdefault("prompt_token_count", len(prompt_ids))
        reward_features.setdefault("response_token_count", 0)

        run_result = {
            "video_path": str(resolved_video_path or sample.get("video_path") or ""),
            "query": str(query or ""),
            "workspace": str(run_workspace.resolve()) if isinstance(run_workspace, Path) else str(run_workspace or ""),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "error": str(error_message or "").strip(),
            "preprocessing": {
                "video_name": (resolved_video_path.stem if isinstance(resolved_video_path, Path) else None),
                "num_segments": 0,
                "processed_output_dir": (
                    str(processed_output_dir.resolve()) if isinstance(processed_output_dir, Path) else str(processed_output_dir or "")
                ),
                "segment_index_sample": [],
            },
            "agent": {
                "final_answer": str(error_message or "").strip(),
                "trajectory": failure_trajectory,
            },
            "metrics": {
                "planner_iterations": 0,
                "subagent_iterations": 0,
                "delegated_root_steps": 0,
                "decomposed_root_steps": 0,
                "parallel_concrete_batches": 0,
                "parallel_concrete_batch_rate": 0.0,
                "cache_stats": dict(runtime_session.get_query_metrics().get("cache_stats") or {}),
                "timing_breakdown_seconds": dict(runtime_session.get_query_metrics().get("timing_breakdown_seconds") or {}),
                "acceptance_metrics": {
                    "acceptance_passed": False,
                    "tool_chain_valid": False,
                    "tool_backed_finish_satisfied": False,
                },
            },
        }
        return EpisodeArtifacts(
            prompt_ids=prompt_ids,
            response_ids=[eos_token_id],
            loss_mask=[0],
            final_answer=str(error_message or "").strip(),
            success=False,
            trajectory=failure_trajectory,
            run_result=run_result,
            reward_features=reward_features,
            conversation_messages=conversation_messages,
            generation_records=generation_records,
        )

    def _load_preprocessing_payload(
        self,
        *,
        video_path: Path,
        preprocess_workspace_root: Path,
        device_type: Any,
        reuse_cache: bool,
    ) -> Dict[str, Any]:
        cache_key = json.dumps(
            {
                "video_path": str(video_path.resolve()),
                "preprocess_workspace_root": str(preprocess_workspace_root.resolve()),
                "device_type": str(device_type or ""),
                "reuse_cache": bool(reuse_cache),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached_payload = self._get_cached_preprocessing_payload(cache_key)
        if cached_payload is not None:
            runtime_session.record_cache("rollout.preprocess_payload_memory", True)
            return cached_payload

        from config.settings import settings

        lock_context = nullcontext()
        if bool(getattr(settings, "preprocess_video_file_lock_enabled", False)):
            # This avoids duplicate first-step preprocessing work, but can leave
            # other GPUs idle when many rollout workers request the same video.
            lock_dir = preprocess_workspace_root / "_locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_suffix = hashlib.md5(str(video_path.resolve()).encode("utf-8")).hexdigest()[:16]
            lock_name = f"{_sanitize_workspace_name(video_path.stem)}_{lock_suffix}.preprocess.lock"
            from filelock import FileLock

            lock_context = FileLock(str(lock_dir / lock_name))

        with lock_context:
            cached_payload = self._get_cached_preprocessing_payload(cache_key)
            if cached_payload is not None:
                runtime_session.record_cache("rollout.preprocess_payload_memory", True)
                return cached_payload

            runtime_session.record_cache("rollout.preprocess_payload_memory", False)

            preprocessing_payload = _run_async_blocking(
                preprocess_video,
                str(video_path),
                preprocess_workspace_root,
                device_type=device_type,
                embedding_device=self.rollout_config.get(
                    "preprocess_embedding_device",
                    self.rollout_config.get("grounding_embedding_device", None),
                ),
                reuse_cache=reuse_cache,
                accelerator_lock=self._accelerator_lock,
            )
            runtime_session.release_providers_for_roles("preprocess")
            self._set_cached_preprocessing_payload(cache_key, preprocessing_payload)
            return preprocessing_payload

    def _resolve_generation_timeout_seconds(
        self,
        *,
        role: str,
        effective_time_budget_sec: float,
    ) -> float:
        normalized_role = str(role or "").strip().lower()
        key = f"{normalized_role}_generation_timeout_seconds"
        raw_value = self.rollout_config.get(key)
        has_explicit_timeout = raw_value not in (None, "")
        base_timeout_seconds = (
            max(1.0, _coerce_float(raw_value, 1.0))
            if has_explicit_timeout
            else (600.0 if normalized_role == "planner" else 300.0)
        )
        resolved_timeout_seconds = base_timeout_seconds
        model_b = _resolve_role_model_parameter_billions(normalized_role)
        large_model_threshold_b = _coerce_float(self.rollout_config.get("large_model_parameter_threshold_b", 0.0), 0.0)
        timeout_scale = 1.0
        adjustment_tags: List[str] = []
        autoscale_toggle_key = f"{normalized_role}_allow_timeout_autoscale_when_explicit"
        allow_timeout_autoscale_when_explicit = _coerce_bool(
            self.rollout_config.get(
                autoscale_toggle_key,
                self.rollout_config.get("allow_timeout_autoscale_when_explicit", False),
            ),
            False,
        )
        apply_timeout_autoscale = (not has_explicit_timeout) or allow_timeout_autoscale_when_explicit

        role_generation_config = settings.get_generation_config(normalized_role) or {}
        enable_thinking = bool(role_generation_config.get("enable_thinking"))
        if normalized_role == "subagent":
            if enable_thinking:
                adjustment_tags.append("subagent_thinking_forced_off")
            enable_thinking = False

        if apply_timeout_autoscale:
            if large_model_threshold_b > 0 and model_b >= large_model_threshold_b:
                scale_key = f"{normalized_role}_large_model_timeout_multiplier"
                min_key = f"{normalized_role}_large_model_timeout_min_seconds"
                large_model_timeout_scale = max(
                    1.0,
                    _coerce_float(
                        self.rollout_config.get(scale_key, self.rollout_config.get("large_model_timeout_multiplier", 1.0)),
                        1.0,
                    ),
                )
                large_model_timeout_min_seconds = max(
                    0.0,
                    _coerce_float(
                        self.rollout_config.get(min_key, self.rollout_config.get("large_model_timeout_min_seconds", 0.0)),
                        0.0,
                    ),
                )
                timeout_scale *= large_model_timeout_scale
                if large_model_timeout_scale > 1.0:
                    adjustment_tags.append(f"large_model_x{large_model_timeout_scale:.2f}")
                if large_model_timeout_min_seconds > 0:
                    resolved_timeout_seconds = max(resolved_timeout_seconds, large_model_timeout_min_seconds)
                    adjustment_tags.append(f"large_model_min={large_model_timeout_min_seconds:.1f}")

            if enable_thinking:
                thinking_scale_key = f"{normalized_role}_thinking_timeout_multiplier"
                thinking_timeout_scale = max(
                    1.0,
                    _coerce_float(
                        self.rollout_config.get(
                            thinking_scale_key,
                            self.rollout_config.get("thinking_timeout_multiplier", 1.0),
                        ),
                        1.0,
                    ),
                )
                timeout_scale *= thinking_timeout_scale
                if thinking_timeout_scale > 1.0:
                    adjustment_tags.append(f"thinking_x{thinking_timeout_scale:.2f}")
        else:
            adjustment_tags.append("explicit_timeout_no_autoscale")

        resolved_timeout_seconds = max(1.0, resolved_timeout_seconds * timeout_scale)
        timeout_cap_key = f"{normalized_role}_generation_timeout_cap_seconds"
        timeout_cap_seconds = max(
            0.0,
            _coerce_float(
                self.rollout_config.get(
                    timeout_cap_key,
                    self.rollout_config.get("generation_timeout_cap_seconds", 0.0),
                ),
                0.0,
            ),
        )
        if timeout_cap_seconds > 0:
            resolved_timeout_seconds = max(1.0, min(resolved_timeout_seconds, timeout_cap_seconds))
            adjustment_tags.append(f"cap={timeout_cap_seconds:.1f}")
        if effective_time_budget_sec > 0:
            resolved_timeout_seconds = max(60.0, min(resolved_timeout_seconds, float(effective_time_budget_sec)))
            adjustment_tags.append(f"bounded_by_wall_clock={effective_time_budget_sec:.1f}")
        logger.info(
            "[Episode][GenerationProfile][DIAG] role=%s model_b=%.1f enable_thinking=%s base_timeout_seconds=%.1f "
            "timeout_scale=%.3f effective_time_budget_sec=%.1f resolved_timeout_seconds=%.1f adjustments=%s",
            normalized_role or "llm",
            model_b,
            enable_thinking,
            base_timeout_seconds,
            timeout_scale,
            effective_time_budget_sec,
            resolved_timeout_seconds,
            adjustment_tags or ["none"],
        )
        return resolved_timeout_seconds

    def _resolve_generation_heartbeat_interval_seconds(self) -> float:
        raw_value = self.rollout_config.get("generation_heartbeat_interval_seconds")
        if raw_value in (None, ""):
            return 30.0
        return max(5.0, _coerce_float(raw_value, 30.0))

    def _is_generation_timeout_interception_enabled(self) -> bool:
        return _coerce_bool(
            self.rollout_config.get("enable_generation_timeout_interception"),
            False,
        )

    def _is_rollout_wall_clock_timeout_enabled(self) -> bool:
        return _coerce_bool(
            self.rollout_config.get("enable_rollout_wall_clock_timeout"),
            True,
        )

    def _build_episode_progress_callback(
        self,
        *,
        sample_id: str,
        episode_id: str,
        worker_pid: int,
    ) -> Callable[[Dict[str, Any]], None]:
        def _emit(event: Dict[str, Any]) -> None:
            if not isinstance(event, dict):
                return
            parts = [
                "[Episode][PROGRESS]",
                f"sample_id={sample_id}",
                f"episode_id={episode_id}",
                f"worker_pid={worker_pid}",
            ]
            for key in (
                "role",
                "depth",
                "iteration",
                "phase",
                "status",
                "elapsed_seconds",
                "generation_remaining_seconds",
                "wall_clock_remaining_seconds",
                "current_tool",
                "source",
            ):
                value = event.get(key)
                if value in (None, "", [], {}):
                    continue
                parts.append(f"{key}={value}")
            result_preview = _truncate_progress_text(event.get("result_preview", ""), limit=220)
            if result_preview:
                parts.append(f"result_preview={json.dumps(result_preview, ensure_ascii=False)}")
            print(" ".join(parts), flush=True)

        return _emit

    def run_episode(
        self,
        sample: Dict[str, Any],
        turn_yield_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        progress_event_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> EpisodeArtifacts:
        query = _resolve_episode_query(sample)
        sample_id = str(sample.get("sample_id") or "").strip() or "<unknown>"
        episode_id = _build_episode_run_id(sample)
        worker_pid = os.getpid()
        is_padding_sample = bool(sample.get("__verl_is_padding", False))
        if is_padding_sample:
            logger.info(
                "[Episode][PADDING_SKIP][DIAG] sample_id=%s worker_pid=%s role_generation_skipped=True",
                sample_id,
                worker_pid,
            )
            return self._build_padding_episode_artifacts(
                sample=sample,
                query=query,
                episode_id=episode_id,
                worker_pid=worker_pid,
            )
        shared_preprocess_workspace = self._build_shared_preprocess_workspace(sample)
        launcher_run_dir = self._resolve_launcher_run_dir()
        launcher_log_path = self._resolve_launcher_log_path()
        launcher_result_path = self._resolve_launcher_result_path()
        persist_episode_artifacts = bool(self.rollout_config.get("persist_episode_artifacts", False))
        persist_episode_details = (
            persist_episode_artifacts
            and launcher_result_path is not None
            and not is_padding_sample
        )
        run_workspace = launcher_run_dir or shared_preprocess_workspace
        episode_artifact_paths = (
            _build_episode_artifact_paths(run_workspace, sample_id, episode_id)
            if persist_episode_artifacts and not is_padding_sample
            else None
        )
        resolved_video_path: Optional[Path] = None
        processed_output_dir: Optional[Path] = None
        previous_data_cache_dir = str(settings.data_cache_dir)
        settings.data_cache_dir = str(shared_preprocess_workspace)
        episode_file_handler: Optional[logging.Handler] = None
        episode_log_stack = ExitStack()
        recorder: Optional[EpisodeRecorder] = None
        llm_activity_logged = False
        interaction_context: Optional[Dict[str, Any]] = None
        engine: Optional[RecursiveEngine] = None
        episode_started = time.perf_counter()
        grounding_text = ""

        try:
            if episode_artifact_paths is not None:
                episode_file_handler = episode_log_stack.enter_context(
                    _capture_episode_run_log(episode_artifact_paths["run_log_path"])
                )
            if persist_episode_details:
                episode_started_at = datetime.utcnow().isoformat() + "Z"
                self._persist_episode_entry(
                    launcher_result_path=launcher_result_path,
                    entry={
                        "episode_id": episode_id,
                        "sample_id": sample_id,
                        "query": query,
                        "status": "running",
                        "worker_pid": worker_pid,
                        "created_at": episode_started_at,
                        "started_at": episode_started_at,
                        "updated_at": episode_started_at,
                    },
                )
                print(
                    "[Episode][START] "
                    f"sample_id={sample_id} "
                    f"episode_id={episode_id} "
                    f"worker_pid={worker_pid} "
                    f"started_at={format_human_readable_timestamp(episode_started_at)}",
                    flush=True,
                )
                logger.info(
                    "Episode persistence enabled. sample_id=%s episode_id=%s worker_pid=%s launcher_log=%s launcher_result=%s",
                    sample_id,
                    episode_id,
                    worker_pid,
                    launcher_log_path,
                    launcher_result_path,
                )
            if episode_artifact_paths is not None:
                logger.info("=" * 80)
                logger.info("Tool Star RL - Rollout Episode Trace")
                logger.info("=" * 80)
                logger.info("Episode sample_id: %s", sample_id)
                logger.info("Episode id: %s", episode_id)
                logger.info("Episode workspace: %s", episode_artifact_paths["workspace"])
                logger.info("Episode run log: %s", episode_artifact_paths["run_log_path"])
                logger.info("Episode raw trace: %s", episode_artifact_paths["raw_trace_jsonl_path"])
                logger.info("Episode run result: %s", episode_artifact_paths["run_result_path"])
                logger.info("=" * 80)

            video_path = _resolve_video_path(sample, shared_preprocess_workspace, self.rollout_config)
            resolved_video_path = Path(video_path).resolve()
            grounding_top_k = int(getattr(settings, "planner_grounding_top_k", 1) or 1)
            reuse_cache = bool(self.rollout_config.get("reuse_cache", True))
            device_type = self.rollout_config.get("device_type")

            runtime_session.reset_query_state()
            runtime_session.ensure_registry_discovered(registry)

            preprocessing_payload = self._load_preprocessing_payload(
                video_path=video_path,
                preprocess_workspace_root=shared_preprocess_workspace,
                device_type=device_type,
                reuse_cache=reuse_cache,
            )
            processed_output_dir = Path(preprocessing_payload["output_dir"]).resolve()

            grounding_embedding_device = self.rollout_config.get("grounding_embedding_device", None)
            grounding_embedding_provider = self.rollout_config.get("grounding_embedding_provider", None)
            grounding_lock = (
                self._accelerator_lock
                if self._accelerator_lock is not None
                and str(grounding_embedding_provider or "").strip().lower() not in {"service", "openai", "api", "remote", "external"}
                and str(grounding_embedding_device or "").strip().lower() not in {"", "cpu"}
                else nullcontext()
            )
            with grounding_lock:
                grounding_context = _build_planner_grounding_context(
                    query=query,
                    preprocessing_result=preprocessing_payload["result"],
                    processed_dir=preprocessing_payload["output_dir"],
                    top_k=max(1, grounding_top_k),
                    embedding_device=grounding_embedding_device,
                    embedding_provider=grounding_embedding_provider,
                )
            grounding_text = _grounding_payload_to_text(grounding_context)
            self._synchronize_distributed_rollout_stage(stage="post_preprocess_pre_planner_generate")

            recorder = EpisodeRecorder()
            planner_llm = RolloutChatAdapter(
                tokenizer=self.tokenizer,
                model_name=str(self.rollout_config.get("model_name", "video_agent_policy")),
                role="planner",
                recorder=recorder,
                max_context_length=self.max_context_length,
                response_generator=self.response_generator,
                generation_session_id=episode_id,
            )
            independent_subagent_service_enabled = self._independent_subagent_service_enabled()
            independent_subagent_service_allow_fallback = self._independent_subagent_service_allow_fallback()
            independent_subagent_delegate_llm: Optional[BaseLLM] = None
            if independent_subagent_service_enabled:
                service_model_name = self._resolve_independent_subagent_service_model_name()
                if self._should_disable_independent_subagent_service_for_local_process(service_model_name):
                    logger.warning(
                        "[Episode][SubagentService][DIAG] disabling independent subagent process provider and falling back to rollout actor because subagent_service model matches planner model under local process mode: model=%s planner_model=%s provider=%s",
                        service_model_name,
                        str(settings.planner_model_path),
                        (
                            os.environ.get("TOOL_STAR_SUBAGENT_SERVICE_MODEL_PROVIDER")
                            or os.environ.get("TOOL_STAR_SUBAGENT_MODEL_PROVIDER")
                            or os.environ.get("TOOL_STAR_NON_PLANNER_MODEL_PROVIDER")
                            or os.environ.get("TOOL_STAR_AUX_MODEL_PROVIDER")
                            or getattr(settings, "non_planner_model_provider", "auto")
                        ),
                    )
                elif independent_subagent_service_allow_fallback:
                    try:
                        independent_subagent_delegate_llm = self._get_independent_subagent_service_llm()
                    except Exception as exc:
                        logger.warning(
                            "[Episode][SubagentService][DIAG] independent subagent service init failed; falling back to rollout actor path: %s",
                            exc,
                        )
                else:
                    independent_subagent_delegate_llm = LazyLLMDelegate(
                        model_name=service_model_name,
                        model_role="subagent_service",
                        factory=self._get_independent_subagent_service_llm,
                    )
                    logger.info(
                        "[Episode][SubagentService][DIAG] independent subagent service lazy init enabled model=%s fallback_allowed=%s",
                        service_model_name,
                        independent_subagent_service_allow_fallback,
                    )

            subagent_llm = RolloutChatAdapter(
                tokenizer=self.tokenizer,
                model_name=(
                    str(getattr(independent_subagent_delegate_llm, "model_name", "") or "")
                    if independent_subagent_delegate_llm is not None
                    else str(self.rollout_config.get("model_name", "video_agent_policy"))
                ),
                role="subagent",
                recorder=recorder,
                max_context_length=self.max_context_length,
                response_generator=(None if independent_subagent_delegate_llm is not None else self.response_generator),
                delegate_llm=independent_subagent_delegate_llm,
                generation_session_id=episode_id,
            )
            logger.info(
                "[Episode][LLM_ADAPTERS][DIAG] planner_type=%s planner_inline=%s subagent_type=%s subagent_inline=%s subagent_independent_service_enabled=%s subagent_independent_service_active=%s subagent_independent_service_model=%s response_generator_type=%s",
                type(planner_llm).__name__,
                bool(getattr(planner_llm, "supports_internal_generation_heartbeat", False)),
                type(subagent_llm).__name__,
                bool(getattr(subagent_llm, "supports_internal_generation_heartbeat", False)),
                independent_subagent_service_enabled,
                independent_subagent_delegate_llm is not None,
                str(getattr(independent_subagent_delegate_llm, "model_name", "") or "<rollout_actor>"),
                type(self.response_generator).__name__,
            )
            self._prepare_delegate_llm(
                independent_subagent_delegate_llm,
                role="subagent_service",
                episode_id=episode_id,
            )
            interaction_context = self._start_episode_interaction(
                sample=sample,
                episode_id=episode_id,
            )

            runtime_session.record_timing("query.total_seconds", 0.0)
            resolved_time_budget_sec = _resolve_effective_time_budget_sec(sample, self.rollout_config)
            wall_clock_timeout_enabled = self._is_rollout_wall_clock_timeout_enabled()
            time_budget_sec = resolved_time_budget_sec if wall_clock_timeout_enabled else 0.0
            wall_clock_deadline = time.perf_counter() + time_budget_sec if time_budget_sec > 0 else None
            if not wall_clock_timeout_enabled:
                logger.info(
                    "[Episode][WallClockTimeout][DIAG] enabled=False rollout wall-clock timeout disabled by rollout config; resolved_budget_sec=%.1f effective_budget_sec=0.0 no wall_clock_deadline will be passed.",
                    resolved_time_budget_sec,
                )
            generation_timeout_interception_enabled = self._is_generation_timeout_interception_enabled()
            if generation_timeout_interception_enabled:
                planner_generation_timeout_seconds = self._resolve_generation_timeout_seconds(
                    role="planner",
                    effective_time_budget_sec=time_budget_sec,
                )
                subagent_generation_timeout_seconds = self._resolve_generation_timeout_seconds(
                    role="subagent",
                    effective_time_budget_sec=time_budget_sec,
                )
            else:
                planner_generation_timeout_seconds = 0.0
                subagent_generation_timeout_seconds = 0.0
                logger.info(
                    "[Episode][GenerationInterception][DIAG] enabled=False planner/subagent per-turn timeout interception disabled by rollout config; wall_clock_timeout_enabled=%s wall_clock_budget_sec=%.1f.",
                    wall_clock_timeout_enabled,
                    time_budget_sec,
                )
            base_progress_callback = self._build_episode_progress_callback(
                sample_id=sample_id,
                episode_id=episode_id,
                worker_pid=worker_pid,
            )
            def progress_callback(event: Dict[str, Any]) -> None:
                normalized_event = dict(event or {})
                base_progress_callback(normalized_event)
                if callable(progress_event_callback):
                    try:
                        progress_event_callback(dict(normalized_event))
                    except Exception as exc:
                        logger.warning(
                            "[Episode][ProgressRelay][DIAG] external progress callback failed: %s",
                            exc,
                        )
            cooperative_turn_yield_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
            interaction_turn_callback_enabled = interaction_context is not None and recorder is not None
            if interaction_turn_callback_enabled or callable(turn_yield_callback):
                active_data_cache_dir = str(shared_preprocess_workspace)
                inactive_data_cache_dir = previous_data_cache_dir

                def _cooperative_turn_yield_callback(event: Dict[str, Any]) -> Any:
                    normalized_event = dict(event or {})
                    if interaction_turn_callback_enabled and interaction_context is not None and recorder is not None:
                        try:
                            self._run_episode_interaction(
                                interaction_context=interaction_context,
                                messages=recorder.build_conversation_messages(),
                                turn_event=normalized_event,
                            )
                        except Exception as exc:
                            logger.warning(
                                "[Episode][Interaction][DIAG] turn-level interaction callback failed: %s",
                                exc,
                            )

                    if not callable(turn_yield_callback):
                        return {"paused_seconds": 0.0}

                    snapshot = runtime_session.snapshot_query_state()
                    pause_result: Any = None
                    try:
                        runtime_session.reset_query_state()
                        settings.data_cache_dir = inactive_data_cache_dir
                        pause_result = turn_yield_callback(normalized_event)
                    finally:
                        runtime_session.restore_query_state(snapshot)
                        settings.data_cache_dir = active_data_cache_dir
                    return pause_result

                cooperative_turn_yield_callback = _cooperative_turn_yield_callback
            engine = RecursiveEngine(
                llm=planner_llm,
                subagent_llm=subagent_llm,
                raw_trace_jsonl_path=(
                    str(episode_artifact_paths["raw_trace_jsonl_path"])
                    if episode_artifact_paths is not None
                    else None
                ),
                session_query=query,
                wall_clock_budget_seconds=time_budget_sec,
                wall_clock_deadline_monotonic=wall_clock_deadline,
                planner_generation_timeout_seconds=planner_generation_timeout_seconds,
                subagent_generation_timeout_seconds=subagent_generation_timeout_seconds,
                generation_heartbeat_interval_seconds=self._resolve_generation_heartbeat_interval_seconds(),
                progress_callback=progress_callback,
                turn_yield_callback=cooperative_turn_yield_callback,
            )
            final_answer = engine.solve(goal=query, extra_context=grounding_context)
            trajectory = engine.get_full_trajectory() or {}
            interaction_summary: Dict[str, Any] = {}
            if interaction_context is not None and recorder is not None:
                if not list(interaction_context.get("turn_records") or []):
                    self._run_episode_interaction(
                        interaction_context=interaction_context,
                        messages=recorder.build_conversation_messages(),
                        turn_event={
                            "role": "planner",
                            "depth": 0,
                            "iteration": 0,
                            "phase": "episode_final",
                            "status": "completed",
                        },
                    )
                interaction_summary = self._build_episode_interaction_summary(interaction_context)

            elapsed_seconds = time.perf_counter() - episode_started
            runtime_session.record_timing("query.total_seconds", elapsed_seconds)
            success = bool(trajectory.get("success")) if isinstance(trajectory, dict) else False
            _log_trajectory(trajectory if isinstance(trajectory, dict) else None)
            run_result = _build_run_result_payload(
                workspace=(
                    episode_artifact_paths["workspace"]
                    if episode_artifact_paths is not None
                    else run_workspace
                ),
                video_path=str(video_path),
                query=query,
                preprocessing_result=preprocessing_payload["result"],
                processed_dir=preprocessing_payload["output_dir"],
                trajectory=trajectory,
                final_answer=final_answer,
                success=success,
            )
            if interaction_summary:
                run_result_metrics = run_result.get("metrics")
                if isinstance(run_result_metrics, dict):
                    run_result_metrics["interaction"] = _json_safe(
                        {
                            "name": interaction_summary.get("interaction_name"),
                            "turn_count": interaction_summary.get("interaction_turn_count", 0),
                            "should_terminate_sequence": interaction_summary.get("interaction_should_terminate_sequence"),
                            "turn_reward": interaction_summary.get("interaction_turn_reward"),
                            "turn_reward_total": interaction_summary.get("interaction_turn_reward_total", 0.0),
                            "turn_reward_avg": interaction_summary.get("interaction_turn_reward_avg", 0.0),
                            "response_preview": interaction_summary.get("interaction_response_preview"),
                            "metrics": interaction_summary.get("interaction_metrics", {}),
                            "turn_records": interaction_summary.get("interaction_turn_records", []),
                        }
                    )
            if episode_artifact_paths is not None:
                _write_json_file(episode_artifact_paths["run_result_path"], run_result)
                logger.info("[RUN_RESULT_JSON] %s", json.dumps(run_result, ensure_ascii=False))

            reward_features = self._build_episode_reward_features(
                episode_id=episode_id,
                sample=sample,
                recorder=recorder,
                elapsed_seconds=elapsed_seconds,
                grounding_text=grounding_text,
                launcher_result_path=launcher_result_path,
                worker_pid=worker_pid,
                effective_time_budget_sec=time_budget_sec,
            )
            if interaction_summary:
                reward_features.update(
                    {
                        "interaction_enabled": True,
                        "interaction_name": interaction_summary.get("interaction_name", ""),
                        "interaction_turn_count": int(interaction_summary.get("interaction_turn_count", 0) or 0),
                        "interaction_should_terminate_sequence": bool(
                            interaction_summary.get("interaction_should_terminate_sequence", False)
                        ),
                        "interaction_turn_reward": interaction_summary.get("interaction_turn_reward"),
                        "interaction_turn_reward_total": interaction_summary.get("interaction_turn_reward_total", 0.0),
                        "interaction_turn_reward_avg": interaction_summary.get("interaction_turn_reward_avg", 0.0),
                        "interaction_response_preview": str(
                            interaction_summary.get("interaction_response_preview") or ""
                        ),
                        "interaction_metrics": _json_safe(interaction_summary.get("interaction_metrics", {})),
                        "interaction_turn_records": _json_safe(interaction_summary.get("interaction_turn_records", [])),
                    }
                )
            prompt_ids, response_ids, loss_mask = recorder.finalize()
            conversation_messages = recorder.build_conversation_messages()
            generation_records = [record.to_dict() for record in recorder.records]
            _log_llm_activity_summary(recorder)
            llm_activity_logged = True

            if persist_episode_details:
                self._persist_episode_entry(
                    launcher_result_path=launcher_result_path,
                    entry=_build_launcher_episode_entry(
                        episode_id=episode_id,
                        sample_id=sample_id,
                        query=query,
                        status="completed",
                        success=bool(success),
                        final_answer=final_answer,
                        worker_pid=worker_pid,
                        video_path=resolved_video_path,
                        reward_features=reward_features,
                        run_result=run_result,
                    ),
                )
            rollout_finished_at = datetime.utcnow().isoformat() + "Z"
            if not is_padding_sample:
                print(
                    "[Episode][COMPLETE] "
                    f"sample_id={sample_id} "
                    f"episode_id={episode_id} "
                    f"worker_pid={worker_pid} "
                    f"success={bool(success)} "
                    f"finished_at={format_human_readable_timestamp(rollout_finished_at)} "
                    f"elapsed_seconds={elapsed_seconds:.6f}",
                    flush=True,
                )

            return EpisodeArtifacts(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                loss_mask=loss_mask,
                final_answer=final_answer,
                success=success,
                trajectory=trajectory,
                run_result=run_result,
                reward_features=reward_features,
                conversation_messages=conversation_messages,
                generation_records=generation_records,
            )
        except Exception as exc:
            if recorder is not None and not llm_activity_logged:
                _log_llm_activity_summary(recorder)
            failed_at = datetime.utcnow().isoformat() + "Z"
            error_text = str(exc).strip() or exc.__class__.__name__
            if episode_artifact_paths is not None:
                _write_json_file(
                    episode_artifact_paths["error_path"],
                    {
                        "episode_id": episode_id,
                        "sample_id": sample_id,
                        "query": query,
                        "error": error_text,
                        "failed_at": failed_at,
                    },
                )
            if not is_padding_sample:
                print(
                    "[Episode][FAIL] "
                    f"sample_id={sample_id} "
                    f"episode_id={episode_id} "
                    f"worker_pid={worker_pid} "
                    f"failed_at={format_human_readable_timestamp(failed_at)} "
                    f"error={error_text}",
                    flush=True,
                )
            if isinstance(exc, AcceleratorFatalError) or is_accelerator_fatal_error(exc):
                raise

            partial_trajectory: Optional[Dict[str, Any]] = None
            if engine is not None:
                try:
                    partial_trajectory = engine.get_full_trajectory() or {}
                except Exception:
                    partial_trajectory = None
            elapsed_seconds = max(0.0, time.perf_counter() - episode_started)
            failed_artifacts = self._build_failed_episode_artifacts(
                sample=sample,
                query=query,
                episode_id=episode_id,
                worker_pid=worker_pid,
                error_message=f"Error: {error_text}",
                elapsed_seconds=elapsed_seconds,
                grounding_text=grounding_text,
                recorder=recorder,
                resolved_video_path=resolved_video_path,
                processed_output_dir=processed_output_dir,
                run_workspace=run_workspace,
                trajectory=partial_trajectory,
            )
            if episode_artifact_paths is not None:
                _write_json_file(episode_artifact_paths["run_result_path"], failed_artifacts.run_result)
            if persist_episode_details:
                self._persist_episode_entry(
                    launcher_result_path=launcher_result_path,
                    entry=_build_launcher_episode_entry(
                        episode_id=episode_id,
                        sample_id=sample_id,
                        query=query,
                        status="failed",
                        success=False,
                        final_answer=failed_artifacts.final_answer,
                        worker_pid=worker_pid,
                        error=error_text,
                        video_path=resolved_video_path,
                        reward_features=failed_artifacts.reward_features,
                        run_result=failed_artifacts.run_result,
                    ),
                )
            logger.warning(
                "[Episode][SOFT_FAIL] Returning failed episode artifacts instead of aborting the whole rollout. sample_id=%s episode_id=%s worker_pid=%s error=%s",
                sample_id,
                episode_id,
                worker_pid,
                error_text,
            )
            return failed_artifacts
        finally:
            self._finalize_episode_interaction(interaction_context)
            episode_log_stack.close()
            settings.data_cache_dir = previous_data_cache_dir
