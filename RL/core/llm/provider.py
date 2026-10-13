"""LLM provider implementations with retry logic."""
from __future__ import annotations
import copy
import gc
import json
import logging
import re
import time
import unicodedata
from contextlib import contextmanager, nullcontext
from typing import List, Dict, Any, Optional, Iterator, Tuple

import os
from pathlib import Path
from PIL import Image, ImageFile
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    torch = None
try:
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    transformers = None
    AutoConfig = AutoModelForCausalLM = AutoProcessor = AutoTokenizer = GenerationConfig = None
try:
    from openai import APIError, RateLimitError, APIConnectionError
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    APIError = RateLimitError = APIConnectionError = Exception
try:
    from anthropic import Anthropic
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    Anthropic = None
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_random_exponential,
        retry_if_exception_type,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_random_exponential(*args, **kwargs):
        return None

    def retry_if_exception_type(*args, **kwargs):
        return None

from .base import BaseLLM
from .process_provider import ProcessBackedLLMProvider
from core.accelerator_queue import SharedAcceleratorQueueLock, resolve_local_accelerator_slot_key
from core.qwen_runtime import disable_qwen35_fused_kernels_if_requested
from core.sidecar_pool import OpenAIEndpointPool, format_endpoint_list, split_endpoint_list
from config.settings import settings
from config.device_config import get_accelerator_snapshot

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

_SYNTHETIC_ROLE_STOP_SEQUENCES = (
    "\nuser\n",
    "\nuser:",
    "\n<|im_start|>user",
    "\nsystem\n",
    "\nsystem:",
    "\n<|im_start|>system",
    "\ntool\n",
    "\ntool:",
    "\n<|im_start|>tool",
)


def _torch_is_tensor(value: Any) -> bool:
    return bool(torch is not None and torch.is_tensor(value))


class _StopOnStructuredJsonOrSyntheticRoleTurn(transformers.StoppingCriteria if transformers is not None else object):
    def __init__(self, tokenizer: Any, prompt_length: int, tail_tokens: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = max(0, int(prompt_length or 0))
        self.tail_tokens = max(8, int(tail_tokens or 96))
        self.role_pattern = re.compile(
            r"(?im)(?:^|\n)\s*(?:<\|im_start\|>\s*)?(user|system|tool)(?:\s*<\|im_end\|>)?\s*(?:\n|:|$)"
        )
        self.schema_key_pattern = re.compile(
            r'"(?:Thought|thought|Plan|plan|Evidence|evidence|Actions|actions|Action|action|Finish|finish|Result|result|'
            r'observation_quality|evidence_updates|recommended_action|confidence|decision|can_finish|'
            r'missing_information|unresolved_claims|evidence_assessment|mutation|expected_effect|'
            r'regression_risk)"\s*:'
        )

    def _think_block_is_open(self, text: str) -> bool:
        lowered = str(text or "").lower()
        return lowered.rfind("<think>") > lowered.rfind("</think>")

    def _contains_completed_response_json(self, text: str) -> bool:
        normalized = str(text or "")
        if not normalized or self._think_block_is_open(normalized):
            return False
        normalized = re.sub(r"<think>.*?</think>", "", normalized, flags=re.IGNORECASE | re.DOTALL)
        depth = 0
        start = -1
        in_string = False
        escape = False
        for idx, ch in enumerate(normalized):
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = normalized[start : idx + 1]
                    if self.schema_key_pattern.search(candidate):
                        return True
                    start = -1
                elif depth < 0:
                    depth = 0
                    start = -1
        return False

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
        if torch is None or not torch.is_tensor(input_ids) or input_ids.ndim < 2:
            return False
        generated_ids = input_ids[0, self.prompt_length :]
        if generated_ids.numel() <= 0:
            return False
        tail_ids = generated_ids[-self.tail_tokens :]
        try:
            tail_text = self.tokenizer.decode(
                tail_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            return False
        if self._contains_completed_response_json(str(tail_text or "")):
            return True
        return bool(self.role_pattern.search(str(tail_text or "")))


def _require_transformers_stack() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if transformers is None:
        missing.append("transformers")
    if missing:
        raise ModuleNotFoundError(
            "TransformersProvider requires optional dependencies: " + ", ".join(missing)
        )


def _require_anthropic_sdk() -> None:
    if Anthropic is None:
        raise ModuleNotFoundError("AnthropicProvider requires optional dependency: anthropic")

_TEXT_ONLY_MODEL_ROLES = frozenset({
    "subagent",
    "preprocess",
    "query_rewrite",
    "python_coder",
    "fold_memory",
    "reflector",
    "evolver",
})
_VISION_MODEL_ROLES = frozenset({
    "qwen_vl",
    "ocr",
    "observer",
})
_VIDEO_METADATA_KEY_ALIASES = {
    "video_fps": "fps",
    "sample_fps": "fps",
    "frames": "total_num_frames",
    "frame_count": "total_num_frames",
    "num_frames": "total_num_frames",
}
_VIDEO_METADATA_ALLOWED_KEYS = frozenset({"fps", "duration", "total_num_frames"})
_VIDEO_PROCESSOR_CORE_KWARGS = frozenset(
    {
        "text",
        "padding",
        "return_tensors",
        "images",
        "videos",
        "video_metadata",
        "fps",
        "max_frames",
        "do_sample_frames",
    }
)
_FLOAT_MULTIMODAL_TENSOR_KEYS = frozenset({"pixel_values", "pixel_values_videos"})
_VISUAL_MULTIMODAL_TENSOR_KEYS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
)
_QWEN_VIDEO_GRID_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
    }
)
_LANGUAGE_INPUT_TENSOR_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
        "position_ids",
        "labels",
    }
)


def _get_configured_enable_thinking(
    model_role: Optional[str],
    provider_name: Optional[str] = None,
) -> Optional[bool]:
    role_config = _get_role_generation_config(model_role, provider_name)
    if "enable_thinking" not in role_config:
        return None
    return bool(role_config.get("enable_thinking"))


def _resolve_enable_thinking_for_role(
    model_role: Optional[str],
    requested_enable_thinking: Any,
    *,
    fallback: bool,
    source: str,
    provider_name: Optional[str] = None,
    respect_role_config: bool = True,
) -> bool:
    normalized_role = str(model_role or "").strip().lower()
    effective = bool(fallback) if requested_enable_thinking is None else bool(requested_enable_thinking)
    if not respect_role_config:
        return effective
    configured = _get_configured_enable_thinking(model_role, provider_name)
    if configured is None:
        return effective
    if effective != configured:
        logger.warning(
            "[ThinkingBoundary][DIAG] source=%s role=%s requested enable_thinking=%s overridden to config=%s.",
            source,
            normalized_role or "<unknown>",
            effective,
            configured,
        )
    return configured


def _looks_like_aliyun_openai_endpoint(base_url: Any) -> bool:
    normalized = str(base_url or "").strip().lower()
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "dashscope",
            "aliyuncs",
            "bailian",
            "aliyun",
            "alibabacloud",
        )
    )


def _resolve_openai_thinking_control_mode(base_url: Any) -> str:
    configured = str(getattr(settings, "openai_thinking_control_mode", "auto") or "auto").strip().lower()
    aliases = {
        "chat_template": "chat_template_kwargs",
        "chat_template_kwargs": "chat_template_kwargs",
        "qwen": "chat_template_kwargs",
        "enable_thinking": "enable_thinking",
        "bailian": "enable_thinking",
        "aliyun": "enable_thinking",
        "both": "both",
        "auto": "auto",
    }
    mode = aliases.get(configured, "auto")
    if mode == "auto":
        return "enable_thinking" if _looks_like_aliyun_openai_endpoint(base_url) else "chat_template_kwargs"
    return mode


def _inject_openai_thinking_control(
    request_extra_body: Dict[str, Any],
    *,
    enable_thinking: bool,
    base_url: Any,
) -> Dict[str, Any]:
    updated = dict(request_extra_body or {})
    mode = _resolve_openai_thinking_control_mode(base_url)
    if mode in {"chat_template_kwargs", "both"}:
        chat_template_kwargs = dict(updated.get("chat_template_kwargs", {}) or {})
        chat_template_kwargs["enable_thinking"] = bool(enable_thinking)
        updated["chat_template_kwargs"] = chat_template_kwargs
    else:
        chat_template_kwargs = dict(updated.get("chat_template_kwargs", {}) or {})
        if "enable_thinking" in chat_template_kwargs:
            chat_template_kwargs.pop("enable_thinking", None)
            if chat_template_kwargs:
                updated["chat_template_kwargs"] = chat_template_kwargs
            else:
                updated.pop("chat_template_kwargs", None)

    if mode in {"enable_thinking", "both"}:
        updated["enable_thinking"] = bool(enable_thinking)
    else:
        updated.pop("enable_thinking", None)
    return updated


def _truncate_for_log(value: Any, limit: int = 4000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _summarize_openai_exception(exc: BaseException) -> str:
    parts = [str(exc)]
    for attr in ("status_code", "code", "type"):
        value = getattr(exc, attr, None)
        if value is not None:
            parts.append(f"{attr}={value}")
    body = getattr(exc, "body", None)
    if body:
        parts.append("body=" + _truncate_for_log(body))
    response = getattr(exc, "response", None)
    text = getattr(response, "text", None) if response is not None else None
    if text:
        parts.append("response_text=" + _truncate_for_log(text))
    return " | ".join(parts)


def _summarize_openai_messages(messages: List[Dict[str, Any]]) -> str:
    image_count = 0
    image_url_chars = 0
    roles: List[str] = []
    for message in messages or []:
        roles.append(str(message.get("role", "")))
        content = message.get("content")
        items = content if isinstance(content, list) else []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_count += 1
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict):
                image_url_chars += len(str(image_url.get("url", "")))
    return f"roles={roles} image_count={image_count} image_url_chars={image_url_chars}"


def _local_sidecar_log_tail(base_url: str, model_role: Optional[str], line_count: int = 120) -> str:
    if "127.0.0.1" not in str(base_url) and "localhost" not in str(base_url).lower():
        return ""
    if str(model_role or "").strip().lower() not in {"qwen_vl", "ocr"}:
        return ""
    log_dir = Path(__file__).resolve().parents[2] / "rl_workdir" / "sidecars" / "logs"
    if not log_dir.exists():
        return ""
    try:
        base_port = int(os.environ.get("TOOL_STAR_NON_PLANNER_SIDECAR_PORT", "18000") or "18000")
    except (TypeError, ValueError):
        base_port = 18000
    try:
        port_stride = max(1, int(getattr(settings, "sidecar_port_stride", 10) or 10))
    except (TypeError, ValueError):
        port_stride = 10
    selected_paths: List[Path] = []
    port_match = re.search(r":(\d+)(?:/|$)", str(base_url))
    if port_match:
        port = int(port_match.group(1))
        replica_index = (port - base_port) // port_stride
        expected_path = log_dir / f"non_planner_llm_{replica_index}.log"
        if replica_index >= 0 and expected_path.exists():
            selected_paths.append(expected_path)
    recent_paths = sorted(log_dir.glob("non_planner_llm_*.log"), key=lambda item: item.stat().st_mtime)
    for path in reversed(recent_paths):
        if path not in selected_paths:
            selected_paths.append(path)
        if len(selected_paths) >= 3:
            break

    chunks: List[str] = []
    error_pattern = re.compile(
        r"ERROR|Traceback|Exception|RuntimeError|InternalServerError|CUDA|OOM|out of memory|502",
        re.IGNORECASE,
    )
    for path in selected_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        error_lines = [line for line in lines if error_pattern.search(line)]
        if error_lines:
            chunks.append(f"== {path.name} recent error lines ==\n" + "\n".join(error_lines[-40:]))
        chunks.append(f"== {path.name} tail ==\n" + "\n".join(lines[-line_count:]))
    return "\n".join(chunks)


def _get_role_generation_config(model_role: Optional[str], provider_name: Optional[str]) -> Dict[str, Any]:
    getter = getattr(settings, "get_generation_config", None)
    if not callable(getter):
        return {}

    resolved_provider = str(provider_name or "").strip().lower()
    try:
        config = dict(getter(model_role, provider=provider_name) or {})
    except TypeError:
        # Backward compatibility for environments where settings.py still exposes
        # the older get_generation_config(model_role) signature.
        config = dict(getter(model_role) or {})

    if resolved_provider in {"transformers", "local"}:
        config.pop("presence_penalty", None)

    return config


def _normalize_video_metadata_item(metadata: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None

    normalized: Dict[str, Any] = {}
    for raw_key, value in metadata.items():
        resolved_key = _VIDEO_METADATA_KEY_ALIASES.get(str(raw_key), str(raw_key))
        if resolved_key not in _VIDEO_METADATA_ALLOWED_KEYS or value is None:
            continue
        try:
            if resolved_key == "total_num_frames":
                parsed_value = int(round(float(value)))
                if parsed_value <= 0:
                    continue
                normalized[resolved_key] = parsed_value
            else:
                normalized[resolved_key] = float(value)
        except (TypeError, ValueError):
            continue

    return normalized or None


def _drop_none_mapping_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        if value is None:
            continue
        cleaned[key] = value
    return cleaned


def _sanitize_decoded_text(text: Any) -> str:
    if text is None:
        return ""

    if isinstance(text, bytes):
        normalized = text.decode("utf-8", errors="replace")
    else:
        normalized = str(text)

    # Keep generated text UTF-8 writable even if a tokenizer returns
    # malformed surrogate code points.
    normalized = normalized.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    normalized = normalized.replace("\x00", "")
    try:
        normalized = unicodedata.normalize("NFC", normalized)
    except Exception:
        pass
    return normalized


def _is_readable_visual_file(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def _rebuild_visual_reference_from_sources(
    target_path: Optional[Path],
    source_images: List[str],
) -> Optional[str]:
    readable_sources = [
        str(Path(source).expanduser().resolve())
        for source in source_images
        if str(source).strip()
        and Path(str(source).expanduser()).exists()
        and _is_readable_visual_file(Path(str(source).expanduser()).resolve())
    ]
    if not readable_sources:
        return None
    if len(readable_sources) == 1:
        return readable_sources[0]

    target = target_path
    if target is None:
        first_source = Path(readable_sources[0])
        target = first_source.parent / f"{first_source.stem}_rebuilt_collage.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)

    tile_size = 448
    canvas = Image.new("RGB", (tile_size * 2, tile_size * 2), color=(0, 0, 0))
    slot_positions = [(0, 0), (1, 0), (1, 1), (0, 1)]
    opened_images: List[Image.Image] = []
    for source in readable_sources[:4]:
        with Image.open(source) as image:
            loaded = image.convert("RGB")
            loaded.thumbnail((tile_size, tile_size), resample=Image.Resampling.LANCZOS)
            frame = Image.new("RGB", (tile_size, tile_size), color=(0, 0, 0))
            x_offset = max(0, (tile_size - loaded.width) // 2)
            y_offset = max(0, (tile_size - loaded.height) // 2)
            frame.paste(loaded, (x_offset, y_offset))
            opened_images.append(frame)
    if not opened_images:
        return None
    for image, (col, row) in zip(opened_images, slot_positions):
        canvas.paste(image, (col * tile_size, row * tile_size))
    canvas.save(target, format="JPEG", quality=90)
    return str(target.resolve())


def _normalize_processor_sequence_tensors(inputs: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    normalized = dict(inputs or {})
    tracked_shapes: Dict[str, int] = {}
    for key in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids", "labels"):
        value = normalized.get(key)
        if torch.is_tensor(value) and value.ndim >= 2:
            tracked_shapes[key] = int(value.shape[-1])

    if len(tracked_shapes) <= 1 or len(set(tracked_shapes.values())) == 1:
        return normalized

    common_length = min(tracked_shapes.values())
    if common_length <= 0:
        return normalized

    logger.warning(
        "Normalizing multimodal processor sequence tensors from %s: %s -> common_length=%d",
        source,
        tracked_shapes,
        common_length,
    )
    for key, value in list(normalized.items()):
        if not torch.is_tensor(value) or value.ndim < 2:
            continue
        if key in tracked_shapes or key == "position_ids":
            normalized[key] = value[..., :common_length]
    return normalized


def _config_model_type(config_dict: Optional[Dict[str, Any]]) -> str:
    return str((config_dict or {}).get("model_type", "") or "").strip().lower()


def _repeat_optional_sequence_like_temporal_dim(
    value: Any,
    temporal_sizes: Any,
) -> Any:
    if value is None or torch is None:
        return value
    if torch.is_tensor(value):
        if value.ndim == 0 or int(value.shape[0]) != int(temporal_sizes.shape[0]):
            return value
        return torch.repeat_interleave(value, temporal_sizes.to(device=value.device), dim=0)
    if isinstance(value, (list, tuple)) and len(value) == int(temporal_sizes.shape[0]):
        repeated: List[Any] = []
        counts = temporal_sizes.detach().cpu().tolist() if torch.is_tensor(temporal_sizes) else list(temporal_sizes)
        for item, count in zip(value, counts):
            try:
                repeat_count = max(1, int(count))
            except Exception:
                repeat_count = 1
            repeated.extend([item] * repeat_count)
        return tuple(repeated) if isinstance(value, tuple) else repeated
    return value


def _normalize_qwen_video_grid_thw(
    inputs: Dict[str, Any],
    *,
    config_dict: Optional[Dict[str, Any]],
    source: str,
) -> Dict[str, Any]:
    if torch is None:
        return dict(inputs or {})
    normalized = dict(inputs or {})
    if _config_model_type(config_dict) not in _QWEN_VIDEO_GRID_MODEL_TYPES:
        return normalized

    video_grid_thw = normalized.get("video_grid_thw")
    if not torch.is_tensor(video_grid_thw) or video_grid_thw.ndim != 2 or int(video_grid_thw.shape[1]) < 1:
        return normalized

    temporal_sizes = video_grid_thw[:, 0]
    if temporal_sizes.numel() == 0:
        return normalized
    try:
        temporal_sizes_long = temporal_sizes.to(dtype=torch.long)
    except Exception:
        return normalized
    if bool(torch.any(temporal_sizes_long <= 0)):
        return normalized
    if int(torch.max(temporal_sizes_long).item()) <= 1:
        return normalized

    expanded = torch.repeat_interleave(video_grid_thw, temporal_sizes_long, dim=0)
    expanded = expanded.clone()
    expanded[:, 0] = 1
    normalized["video_grid_thw"] = expanded
    if "second_per_grid_ts" in normalized:
        normalized["second_per_grid_ts"] = _repeat_optional_sequence_like_temporal_dim(
            normalized.get("second_per_grid_ts"),
            temporal_sizes_long,
        )
    logger.info(
        "Normalized Qwen video_grid_thw for %s: %s -> %s",
        source,
        tuple(video_grid_thw.shape),
        tuple(expanded.shape),
    )
    return normalized


def _normalize_local_model_path(model_name: Optional[str]) -> str:
    resolved_name = str(model_name or settings.planner_model_path)
    try:
        return str(Path(resolved_name).expanduser().resolve())
    except Exception:
        return resolved_name


def _provider_device_purpose(model_role: Optional[str]) -> str:
    normalized_role = str(model_role or "").strip().lower()
    if normalized_role in {"planner", "subagent"}:
        return "llm_provider_planner_subagent_shared"
    if normalized_role:
        return f"llm_provider_{normalized_role}"
    return "llm_provider_default"


def _resolve_provider_runtime_device(
    device: Optional[str],
    *,
    purpose: str = "llm_provider_default",
) -> str:
    from runtime.session import runtime_session
    from config.device_config import detect_device, get_device_string

    if isinstance(device, int):
        requested_index = max(0, int(device))
        detected_device, detected_count = detect_device()
        if detected_device in {"cuda", "npu"}:
            if detected_count is not None and detected_count > 0:
                requested_index = min(requested_index, int(detected_count) - 1)
            return get_device_string(detected_device, requested_index)

    requested_device = str(device or "").strip()
    if ":" in requested_device:
        return requested_device
    if requested_device.isdigit():
        requested_index = int(requested_device)
        detected_device, detected_count = detect_device()
        if detected_device in {"cuda", "npu"}:
            if detected_count is not None and detected_count > 0:
                requested_index = min(requested_index, int(detected_count) - 1)
            return get_device_string(detected_device, requested_index)
    if requested_device in {"cuda", "npu"}:
        return get_device_string(requested_device, 0)
    if requested_device:
        return runtime_session.get_stable_device(purpose, device=requested_device)
    raw_local_rank = str(os.environ.get("LOCAL_RANK") or "").strip()
    if raw_local_rank:
        try:
            local_rank = int(raw_local_rank)
        except Exception:
            local_rank = -1
        if local_rank >= 0:
            detected_device, detected_count = detect_device()
            if detected_device in {"cuda", "npu"}:
                if detected_count is not None and detected_count > 0:
                    local_rank = min(local_rank, int(detected_count) - 1)
                return get_device_string(detected_device, local_rank)
    detected_device, _ = detect_device()
    return runtime_session.get_stable_device(purpose, device=detected_device)


def _should_share_planner_subagent_provider(
    model_name: Optional[str],
    model_role: Optional[str],
) -> bool:
    normalized_role = str(model_role or "").strip().lower()
    if normalized_role not in {"planner", "subagent"}:
        return False

    normalized_model_name = _normalize_local_model_path(model_name)
    normalized_planner_model = _normalize_local_model_path(settings.planner_model_path)
    normalized_subagent_model = _normalize_local_model_path(settings.subagent_model_path)
    return (
        normalized_planner_model == normalized_subagent_model
        and normalized_model_name == normalized_planner_model
    )


def _normalize_role_env_key(model_role: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(model_role or "").strip().upper()).strip("_")


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _settings_role_mapping_value(mapping_name: str, model_role: Optional[str]) -> str:
    mapping = getattr(settings, mapping_name, {}) or {}
    if not isinstance(mapping, dict):
        return ""
    normalized_role = str(model_role or "").strip().lower()
    role_key = _normalize_role_env_key(model_role).lower()
    for key in (normalized_role, role_key):
        if not key:
            continue
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _default_non_planner_service_model(model_role: Optional[str]) -> str:
    normalized_role = str(model_role or "").strip().lower()
    role_to_attr = {
        "subagent": "subagent_model_path",
        "subagent_service": "subagent_model_path",
        "preprocess": "preprocess_model_path",
        "qwen_vl": "qwen_vl_model_path",
        "ocr": "ocr_model_path",
        "query_rewrite": "query_rewrite_model_path",
        "python_coder": "python_coder_model_path",
        "fold_memory": "fold_memory_model_path",
    }
    attr_name = role_to_attr.get(normalized_role)
    if not attr_name:
        return ""
    value = getattr(settings, attr_name, "")
    return str(value or "").strip()


def _normalize_aux_provider_name(provider_name: Any) -> str:
    normalized = str(provider_name or "").strip().lower()
    aliases = {
        "": "auto",
        "sidecar": "openai",
        "service": "openai",
        "remote": "openai",
        "api": "openai",
        "process": "process",
        "local_process": "process",
        "subprocess": "process",
        "disable": "disabled",
        "off": "disabled",
        "false": "disabled",
        "0": "disabled",
    }
    return aliases.get(normalized, normalized)


def _non_planner_role_can_use_sidecar(model_role: Optional[str]) -> bool:
    normalized_role = str(model_role or "").strip().lower()
    return bool(normalized_role) and normalized_role != "planner"


def _runtime_profile_forces_local_non_planner_runtime() -> bool:
    normalized = str(os.environ.get("TOOL_STAR_RUNTIME_PROFILE", "") or "").strip().lower()
    return normalized in {
        "evaluation_local",
        "evaluation-local",
        "local_evaluation",
        "local-evaluation",
        "evaluation",
        "eval",
    }


def _non_planner_service_provider(model_role: Optional[str]) -> str:
    role_key = _normalize_role_env_key(model_role)
    role_values = []
    if role_key:
        role_values.extend(
            [
                os.environ.get(f"TOOL_STAR_{role_key}_MODEL_PROVIDER"),
                os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_PROVIDER"),
            ]
        )
    return _normalize_aux_provider_name(
        _first_nonempty(
            *role_values,
            os.environ.get("TOOL_STAR_NON_PLANNER_MODEL_PROVIDER"),
            os.environ.get("TOOL_STAR_AUX_MODEL_PROVIDER"),
            _settings_role_mapping_value("non_planner_role_providers", model_role),
            getattr(settings, "non_planner_model_provider", "auto"),
        )
    )


def _non_planner_service_api_base(model_role: Optional[str], *, include_openai_fallback: bool) -> str:
    role_key = _normalize_role_env_key(model_role)
    role_values = []
    if role_key:
        role_values.extend(
            [
                os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_API_BASE"),
                os.environ.get(f"TOOL_STAR_{role_key}_API_BASE"),
            ]
        )
    values = [
        *role_values,
        os.environ.get("TOOL_STAR_NON_PLANNER_SERVICE_API_BASE"),
        os.environ.get("TOOL_STAR_NON_PLANNER_API_BASE"),
        os.environ.get("TOOL_STAR_AUX_SERVICE_API_BASE"),
        os.environ.get("TOOL_STAR_AUX_API_BASE"),
        _settings_role_mapping_value("non_planner_role_api_bases", model_role),
        getattr(settings, "non_planner_api_base", ""),
    ]
    if include_openai_fallback:
        values.append(getattr(settings, "openai_api_base", ""))
    return _first_nonempty(*values)


def _non_planner_service_api_bases(model_role: Optional[str], *, include_openai_fallback: bool) -> str:
    role_key = _normalize_role_env_key(model_role)
    role_values = []
    if role_key:
        role_values.extend(
            [
                os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_API_BASES"),
                os.environ.get(f"TOOL_STAR_{role_key}_API_BASES"),
                os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_API_BASE"),
                os.environ.get(f"TOOL_STAR_{role_key}_API_BASE"),
            ]
        )
    values = [
        *role_values,
        os.environ.get("TOOL_STAR_NON_PLANNER_SERVICE_API_BASES"),
        os.environ.get("TOOL_STAR_NON_PLANNER_API_BASES"),
        os.environ.get("TOOL_STAR_NON_PLANNER_SERVICE_API_BASE"),
        os.environ.get("TOOL_STAR_NON_PLANNER_API_BASE"),
        os.environ.get("TOOL_STAR_AUX_SERVICE_API_BASES"),
        os.environ.get("TOOL_STAR_AUX_API_BASES"),
        os.environ.get("TOOL_STAR_AUX_SERVICE_API_BASE"),
        os.environ.get("TOOL_STAR_AUX_API_BASE"),
        _settings_role_mapping_value("non_planner_role_api_bases", model_role),
        getattr(settings, "non_planner_api_bases", ""),
        getattr(settings, "non_planner_api_base", ""),
    ]
    if include_openai_fallback:
        values.append(getattr(settings, "openai_api_base", ""))
    return format_endpoint_list(_first_nonempty(*values))


def _non_planner_service_api_key(model_role: Optional[str]) -> str:
    role_key = _normalize_role_env_key(model_role)
    role_values = []
    if role_key:
        role_values.extend(
            [
                os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_API_KEY"),
                os.environ.get(f"TOOL_STAR_{role_key}_API_KEY"),
            ]
        )
    return _first_nonempty(
        *role_values,
        _settings_role_mapping_value("non_planner_role_api_keys", model_role),
        getattr(settings, "non_planner_api_key", ""),
        os.environ.get("TOOL_STAR_NON_PLANNER_SERVICE_API_KEY"),
        os.environ.get("TOOL_STAR_NON_PLANNER_API_KEY"),
        os.environ.get("TOOL_STAR_AUX_SERVICE_API_KEY"),
        os.environ.get("TOOL_STAR_AUX_API_KEY"),
        getattr(settings, "openai_api_key", ""),
        "EMPTY",
    )


def _non_planner_service_model(model_role: Optional[str], *, include_openai_fallback: bool) -> str:
    role_key = _normalize_role_env_key(model_role)
    role_values = []
    if role_key:
        role_values.extend(
            [
                os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_MODEL"),
                os.environ.get(f"TOOL_STAR_{role_key}_MODEL"),
            ]
        )
    values = [
        *role_values,
        os.environ.get("TOOL_STAR_NON_PLANNER_SERVICE_MODEL"),
        os.environ.get("TOOL_STAR_NON_PLANNER_MODEL"),
        os.environ.get("TOOL_STAR_AUX_SERVICE_MODEL"),
        os.environ.get("TOOL_STAR_AUX_MODEL"),
        _settings_role_mapping_value("non_planner_role_models", model_role),
        getattr(settings, "non_planner_model", ""),
        _default_non_planner_service_model(model_role),
    ]
    if include_openai_fallback:
        values.append(getattr(settings, "openai_model", ""))
    return _first_nonempty(*values)


def _resolve_non_planner_service_kwargs(
    *,
    model_role: Optional[str],
    current_kwargs: Dict[str, Any],
    explicit: bool,
) -> Optional[Dict[str, Any]]:
    if not _non_planner_role_can_use_sidecar(model_role):
        return None
    if _runtime_profile_forces_local_non_planner_runtime():
        logger.info(
            "Skipping non-planner sidecar routing for role=%s because TOOL_STAR_RUNTIME_PROFILE=%s forces local evaluation runtime.",
            str(model_role or "").strip().lower(),
            str(os.environ.get("TOOL_STAR_RUNTIME_PROFILE", "") or "").strip(),
        )
        return None

    provider_name = _non_planner_service_provider(model_role)
    if provider_name in {"disabled", "local", "transformers", "process"}:
        return None
    if provider_name not in {"auto", "openai"}:
        return None

    include_openai_fallback = provider_name == "openai" or explicit
    service_base = _non_planner_service_api_bases(
        model_role,
        include_openai_fallback=include_openai_fallback,
    )
    service_model = _non_planner_service_model(
        model_role,
        include_openai_fallback=include_openai_fallback,
    )
    if not service_base or not service_model:
        return None

    updated = dict(current_kwargs)
    updated["model_name"] = service_model
    updated["base_url"] = service_base
    updated["api_key"] = _non_planner_service_api_key(model_role)
    updated.setdefault("model_role", model_role)
    logger.info(
        "Routing non-planner role=%s to OpenAI-compatible sidecar model=%s base=%s",
        str(model_role or "").strip().lower(),
        service_model,
        service_base,
    )
    return updated


def build_transformers_provider_cache_key(
    *,
    model_name: Optional[str],
    model_role: Optional[str],
    device: Optional[str],
    device_map: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    normalized_role = str(model_role or "").strip().lower()
    share_planner_subagent = _should_share_planner_subagent_provider(
        model_name,
        normalized_role,
    )
    effective_role = "planner" if share_planner_subagent else normalized_role
    cache_payload = {
        "provider": "transformers",
        "model_name": _normalize_local_model_path(model_name),
        "model_role": "planner_subagent_shared" if share_planner_subagent else normalized_role,
        "device": str(device or "").strip(),
        "runtime_device": _resolve_provider_runtime_device(
            device,
            purpose=_provider_device_purpose(effective_role),
        ),
        "device_map": str(device_map or "").strip(),
        "force_single_device": bool(getattr(settings, "force_single_device", False)),
        "per_device_max_memory_gb": getattr(settings, "per_device_max_memory_gb", None),
    }
    if not share_planner_subagent:
        cache_payload.update(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "enable_thinking": enable_thinking,
            }
        )
    return json.dumps(cache_payload, ensure_ascii=False, sort_keys=True)


def build_process_provider_cache_key(
    *,
    model_name: Optional[str],
    model_role: Optional[str],
    device: Optional[str],
    device_map: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    normalized_role = str(model_role or "").strip().lower()
    cache_payload = {
        "provider": "process",
        "model_name": _normalize_local_model_path(model_name),
        "model_role": normalized_role,
        "device": str(device or "").strip(),
        "runtime_device": _resolve_provider_runtime_device(
            device,
            purpose=_provider_device_purpose(normalized_role),
        ),
        "device_map": str(device_map or "").strip(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "enable_thinking": enable_thinking,
    }
    return json.dumps(cache_payload, ensure_ascii=False, sort_keys=True)


class LocalProvider(BaseLLM):
    """Lightweight local provider used when no remote LLM credentials are configured."""

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        model_name = model_name or settings.planner_model_path
        super().__init__(model_name=model_name, **kwargs)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Return a minimal ReAct-compatible response to keep the pipeline running
        when external LLMs are unavailable.
        """
        logger.warning(
            "LocalProvider in use (no remote LLM credentials). Returning stub Finish response."
        )
        return (
            '{"Thought":"Collected context.",'
            '"Plan":"Return the local stub response.",'
            '"Actions":[],'
            '"Finish":{"chain_complete":true,"completion_basis":"stub->answer",'
            '"answer":"Local stub answer (no remote LLM configured)."}}'
        )

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """Yield the same stub response as a single chunk."""
        yield self.generate(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)


class TransformersProvider(BaseLLM):
    """Local transformers provider for chat-style generation."""

    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        device_map: Optional[str] = None,
        device: Optional[str] = None,
        model_role: Optional[str] = None,
        **kwargs
    ):
        from config.device_config import detect_device, get_device_string, initialize_device

        model_path = Path(model_name).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path not found: {model_path}")
        provider_started = time.perf_counter()
        _require_transformers_stack()
        # Apply the same explicit Orion safety switch to text-only Evolver and
        # auxiliary roles. The video caption path used to own this logic,
        # leaving direct TransformersProvider callers exposed to the same LLVM
        # abort during Qwen3.5 generation.
        qwen_runtime_policy = disable_qwen35_fused_kernels_if_requested(
            transformers_module=transformers,
        )
        # Force offline/local loading to avoid HF repo_id validation on absolute paths
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        role_defaults = _get_role_generation_config(model_role, "transformers")
        super().__init__(model_name=str(model_path), model_role=model_role, **kwargs)
        self.model_role = model_role
        self._requested_device_map = device_map
        self.temperature = role_defaults.get("temperature", 0.7) if temperature is None else temperature
        default_max_tokens = role_defaults.get("max_new_tokens", kwargs.get("max_tokens", 8192))
        self.max_tokens = max_tokens if max_tokens is not None else default_max_tokens
        self.top_p = kwargs.get("top_p", role_defaults.get("top_p"))
        self.top_k = kwargs.get("top_k", role_defaults.get("top_k"))
        self.min_p = kwargs.get("min_p", role_defaults.get("min_p"))
        self.presence_penalty = kwargs.get("presence_penalty", role_defaults.get("presence_penalty"))
        self.repetition_penalty = kwargs.get("repetition_penalty", role_defaults.get("repetition_penalty"))
        self.enable_thinking = _resolve_enable_thinking_for_role(
            model_role,
            kwargs.get("enable_thinking", role_defaults.get("enable_thinking", False)),
            fallback=bool(role_defaults.get("enable_thinking", False)),
            source="TransformersProvider.__init__",
            provider_name="transformers",
        )
        resolved_runtime_device = _resolve_provider_runtime_device(
            device,
            purpose=_provider_device_purpose(model_role),
        )

        resolved_runtime_device_text = str(resolved_runtime_device or "").strip().lower()
        if ":" in resolved_runtime_device_text:
            resolved_device_type = resolved_runtime_device_text.split(":", 1)[0]
        else:
            resolved_device_type = resolved_runtime_device_text or "cpu"
        if resolved_device_type not in ("cuda", "npu", "cpu"):
            logger.warning("Unknown device '%s', fallback to cpu", device)
            resolved_device_type = "cpu"
            resolved_runtime_device = "cpu"
        initialize_device(resolved_device_type)
        self.device = resolved_device_type
        self.runtime_device = str(resolved_runtime_device)
        self.qwen_runtime_policy = qwen_runtime_policy
        detected_device_type, detected_device_count = detect_device()
        self.accelerator_count = detected_device_count or 0
        try:
            queue_log_wait_seconds = float(kwargs.get("shared_accelerator_queue_log_wait_seconds", 5.0) or 5.0)
        except Exception:
            queue_log_wait_seconds = 5.0
        self._accelerator_lock = SharedAcceleratorQueueLock(
            name=f"transformers_provider_{str(model_role or 'default').strip() or 'default'}",
            queue_root=kwargs.get("shared_accelerator_queue_root"),
            slot_key=resolve_local_accelerator_slot_key(self.runtime_device),
            enabled=bool(kwargs.get("enable_shared_accelerator_queue", True)) and str(self.runtime_device).startswith("cuda"),
            log_wait_seconds=queue_log_wait_seconds,
        )
        if self._accelerator_lock is not None:
            logger.info(
                "[AcceleratorQueue] provider role=%s requested_capacity=%s effective_capacity=%s "
                "memory_guard=%s slot=%s",
                str(model_role or "default"),
                getattr(self._accelerator_lock, "requested_capacity", getattr(self._accelerator_lock, "capacity", 1)),
                getattr(self._accelerator_lock, "capacity", 1),
                bool(getattr(self._accelerator_lock, "memory_guard_enabled", False)),
                getattr(self._accelerator_lock, "slot_key", ""),
            )

        # Resident workers may remain concurrent, but a generation request is
        # the high-water-mark allocation for Qwen3.5.  The old provider queue
        # capacity was also used for generation, so three workers could enter
        # three multimodal generations at once and exhaust one physical GPU.
        # Keep the resident/model-load queue separate and arbitrate all
        # Planner/Observer/Reflector/Video_Clip_QA generations through one
        # shared cross-process slot.  Role views are shallow copies and share
        # this lock with the loaded kernel provider.
        raw_generation_capacity = str(
            os.environ.get("TOOLSTAR_ACCELERATOR_GENERATION_QUEUE_CAPACITY") or "1"
        ).strip()
        try:
            generation_capacity = max(1, min(8, int(raw_generation_capacity)))
        except (TypeError, ValueError):
            generation_capacity = 1
        raw_generation_guard = str(
            os.environ.get("TOOLSTAR_ACCELERATOR_GENERATION_MEMORY_GUARD") or "1"
        ).strip().lower()
        generation_guard_enabled = raw_generation_guard not in {"0", "false", "no", "off"}
        try:
            generation_min_free_gib = max(
                0.0,
                float(os.environ.get("TOOLSTAR_ACCELERATOR_GENERATION_MIN_FREE_GIB") or 1.0),
            )
        except (TypeError, ValueError):
            generation_min_free_gib = 1.0
        try:
            generation_reserve_gib = max(
                0.0,
                float(os.environ.get("TOOLSTAR_ACCELERATOR_GENERATION_RESERVE_GIB") or 1.0),
            )
        except (TypeError, ValueError):
            generation_reserve_gib = 1.0
        self._generation_lock = SharedAcceleratorQueueLock(
            name="transformers_generation",
            queue_root=kwargs.get("shared_accelerator_queue_root"),
            slot_key=resolve_local_accelerator_slot_key(self.runtime_device),
            enabled=bool(kwargs.get("enable_shared_accelerator_queue", True)) and str(self.runtime_device).startswith("cuda"),
            log_wait_seconds=queue_log_wait_seconds,
            capacity=generation_capacity,
            memory_guard_enabled=generation_guard_enabled,
            memory_min_free_mib=generation_min_free_gib * 1024.0,
            memory_reserve_mib=generation_reserve_gib * 1024.0,
        )
        if self._generation_lock is not None:
            logger.info(
                "[AcceleratorQueue] generation requested_capacity=%s effective_capacity=%s "
                "memory_guard=%s threshold_gib=%.2f slot=%s",
                getattr(self._generation_lock, "requested_capacity", getattr(self._generation_lock, "capacity", 1)),
                getattr(self._generation_lock, "capacity", 1),
                bool(getattr(self._generation_lock, "memory_guard_enabled", False)),
                float(getattr(self._generation_lock, "memory_admission_threshold_mib", 0.0)) / 1024.0,
                getattr(self._generation_lock, "slot_key", ""),
            )
        
        # Determine device_map strategy
        self.device_map = self._determine_device_map(
            device_map, 
            self.device, 
            self.accelerator_count
        )
        # Validate minimal files to avoid HF repo_id fallback
        tokenizer_cfg = model_path / "tokenizer_config.json"
        if not tokenizer_cfg.exists():
            raise FileNotFoundError(
                f"tokenizer_config.json not found under local model path: {model_path}. "
                f"Ensure the full model files are present locally."
            )
        self._model_config_dict = self._load_model_config_dict(str(model_path))
        self.device_map = self._finalize_device_map(
            str(model_path),
            self.device_map,
            self._model_config_dict,
        )
        if (
            self.device == "cuda"
            and self.accelerator_count > 1
            and self.device_map is None
            and self._requested_device_map is None
        ):
            logger.info(
                "TransformersProvider prefers single-device CUDA inference for latency role=%s runtime_device=%s; multi-device sharding is reserved for explicit overrides or single-device OOM fallback.",
                str(self.model_role or "").strip().lower() or "default",
                self.runtime_device,
            )
        self.processor = None
        self._process_vision_info = None
        self._supports_multimodal_generation = self._should_enable_multimodal_generation(
            self._model_config_dict
        )
        # Force local load to avoid HF repo_id validation when passing absolute paths
        tokenizer_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
            cache_dir=str(model_path),
        )
        if hasattr(self.tokenizer, "truncation_side"):
            try:
                # Preserve the latest turns when long chat history hits the context cap.
                self.tokenizer.truncation_side = "left"
            except Exception:
                pass
        logger.info(
            "TransformersProvider tokenizer ready role=%s elapsed=%.2fs model=%s",
            str(self.model_role or "").strip().lower() or "default",
            time.perf_counter() - tokenizer_started,
            model_path,
        )
        # Determine optimal dtype (bfloat16 preferred for Qwen3, float16 fallback)
        model_dtype = self._get_optimal_dtype()

        def _build_model_kwargs() -> Dict[str, Any]:
            model_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "cache_dir": str(model_path),
                "torch_dtype": model_dtype,
                "low_cpu_mem_usage": True,
            }
            if self.device_map is not None:
                model_kwargs["device_map"] = self.device_map
                # Only auto-configure max_memory for string-based strategies
                # (balanced / sequential). Explicit dict maps already pin layers
                # to devices, so max_memory is unnecessary and can conflict.
                if isinstance(self.device_map, str) and self.accelerator_count > 1:
                    max_memory = self._get_max_memory_config()
                    if max_memory:
                        model_kwargs["max_memory"] = max_memory
            return model_kwargs

        model_cls = self._select_auto_model_class(
            str(model_path),
            config_dict=self._model_config_dict,
        )
        lock_context = self._accelerator_lock if self._accelerator_lock is not None else nullcontext()
        with lock_context:
            model_load_started = time.perf_counter()
            logger.info(
                "TransformersProvider loading model weights role=%s model=%s dtype=%s device=%s device_map=%s",
                str(self.model_role or "").strip().lower() or "default",
                model_path,
                model_dtype,
                self.runtime_device,
                self.device_map,
            )
            try:
                self.model = self._load_hf_model_with_dtype_fallback(
                    model_cls,
                    str(model_path),
                    _build_model_kwargs(),
                    model_dtype,
                )
                logger.info(
                    "TransformersProvider model weights ready role=%s elapsed=%.2fs",
                    str(self.model_role or "").strip().lower() or "default",
                    time.perf_counter() - model_load_started,
                )
                if self.device_map is None:
                    device_move_started = time.perf_counter()
                    logger.info(
                        "TransformersProvider moving model to device role=%s device=%s",
                        str(self.model_role or "").strip().lower() or "default",
                        self.runtime_device,
                    )
                    self.model.to(self.runtime_device)
                    logger.info(
                        "TransformersProvider model moved to device role=%s elapsed=%.2fs",
                        str(self.model_role or "").strip().lower() or "default",
                        time.perf_counter() - device_move_started,
                    )
                self.model.eval()
            except Exception as exc:
                if self._is_out_of_memory_error(exc) and self._can_retry_load_on_multi_device():
                    self.model = None
                    self._clear_model_load_buffers()
                    self.device_map = self._finalize_device_map(
                        str(model_path),
                        "balanced",
                        self._model_config_dict,
                    )
                    logger.warning(
                        "Single-device provider load ran out of memory; retrying with multi-device device_map=%s role=%s model=%s.",
                        self.device_map,
                        str(self.model_role or "").strip().lower() or "default",
                        str(model_path),
                    )
                    self.model = self._load_hf_model_with_dtype_fallback(
                        model_cls,
                        str(model_path),
                        _build_model_kwargs(),
                        model_dtype,
                    )
                    self.model.eval()
                else:
                    raise

        if self._supports_multimodal_generation:
            try:
                processor_started = time.perf_counter()
                processor_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": True,
                    "cache_dir": str(model_path),
                }
                processor_kwargs.update(self._visual_processor_load_kwargs())
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        str(model_path),
                        **processor_kwargs,
                    )
                except TypeError:
                    for key in ("min_pixels", "max_pixels"):
                        processor_kwargs.pop(key, None)
                    self.processor = AutoProcessor.from_pretrained(
                        str(model_path),
                        **processor_kwargs,
                    )
                processor_tokenizer = getattr(self.processor, "tokenizer", None)
                if processor_tokenizer is not None:
                    self.tokenizer = processor_tokenizer
                logger.info(
                    "TransformersProvider processor ready role=%s elapsed=%.2fs",
                    str(self.model_role or "").strip().lower() or "default",
                    time.perf_counter() - processor_started,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize multimodal processor for %s; planner/tool observations will fall back to text-only prompts: %s",
                    model_path,
                    exc,
                )
                self.processor = None
        self._logged_unsupported_generation_keys = set()
        self._supported_generation_keys = self._discover_supported_generation_keys()

        # Sanitise generation_config to avoid spurious warnings.
        # Some model checkpoints ship generation_config.json with sampling
        # parameters (temperature/top_p/top_k) that trigger
        # "generation flags are not valid" when do_sample=False.
        # Set these to None instead of deleting to prevent them from being reloaded.
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            # Set sampling parameters to None to prevent auto-application
            gen_cfg.temperature = None
            gen_cfg.top_p = None
            gen_cfg.top_k = None
            gen_cfg.do_sample = None
            # Also set do_sample explicitly to avoid defaults
            if hasattr(gen_cfg, "repetition_penalty"):
                gen_cfg.repetition_penalty = None

        # Log device distribution for debugging multi-card issues
        hf_map = getattr(self.model, "hf_device_map", None)
        actual_param_dtype = "unknown"
        try:
            actual_param_dtype = str(next(self.model.parameters()).dtype)
        except Exception:
            pass
        if hf_map:
            logger.info(
                "Model distributed across devices: %s (requested_dtype=%s actual_param_dtype=%s)",
                hf_map,
                model_dtype,
                actual_param_dtype,
            )
        else:
            target = self.runtime_device if self.device_map is None else self.device_map
            logger.info(
                "Model loaded on: %s (requested_dtype=%s actual_param_dtype=%s)",
                target,
                model_dtype,
                actual_param_dtype,
            )
        logger.info("LLM accelerator snapshot: %s", get_accelerator_snapshot(self.device))
        logger.info(
            "TransformersProvider initialization complete role=%s elapsed=%.2fs",
            str(self.model_role or "").strip().lower() or "default",
            time.perf_counter() - provider_started,
        )

    def _load_hf_model_with_dtype_fallback(
        self,
        model_cls: Any,
        model_path: str,
        model_kwargs: Dict[str, Any],
        model_dtype: torch.dtype,
    ) -> Any:
        try:
            return model_cls.from_pretrained(model_path, **model_kwargs)
        except TypeError as exc:
            if "torch_dtype" not in str(exc):
                raise

            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("torch_dtype", None)
            fallback_kwargs["dtype"] = model_dtype
            logger.warning(
                "Model class %s does not accept torch_dtype; retrying with dtype=%s.",
                getattr(model_cls, "__name__", str(model_cls)),
                model_dtype,
            )
            return model_cls.from_pretrained(model_path, **fallback_kwargs)

    def _visual_processor_load_kwargs(self) -> Dict[str, Any]:
        role = str(self.model_role or "").strip().lower()
        use_visual_bounds = role in _VISION_MODEL_ROLES
        if not use_visual_bounds and role == "planner":
            # Planner may share the same local Qwen-VL checkpoint with the
            # vision runtime. Keep its processor pixel bounds aligned when the
            # checkpoint is actually multimodal, without forcing text planners
            # into the vision-only path.
            try:
                use_visual_bounds = self._is_multimodal_config(
                    getattr(self, "_model_config_dict", {}) or {}
                )
            except Exception:
                use_visual_bounds = False
        if not use_visual_bounds:
            return {}
        return {
            "min_pixels": int(getattr(settings, "qwen_vl_image_min_pixels", 256 * 256)),
            "max_pixels": int(getattr(settings, "qwen_vl_image_max_pixels", 1280 * 28 * 28)),
        }

    def _discover_supported_generation_keys(self) -> set:
        supported_keys = set()

        try:
            supported_keys.update(GenerationConfig().to_dict().keys())
        except Exception:
            pass

        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "to_dict"):
            try:
                supported_keys.update(generation_config.to_dict().keys())
            except Exception:
                pass

        supported_keys.update(
            {
                "use_cache",
                "remove_invalid_values",
                "renormalize_logits",
                "synced_gpus",
                "stopping_criteria",
            }
        )
        return supported_keys

    def _filter_generation_kwargs(self, generation_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._supported_generation_keys:
            return generation_kwargs

        filtered_kwargs: Dict[str, Any] = {}
        dropped_keys = []
        for key, value in generation_kwargs.items():
            if value is None:
                continue
            if key in self._supported_generation_keys:
                filtered_kwargs[key] = value
            else:
                dropped_keys.append(key)

        for key in dropped_keys:
            if key in self._logged_unsupported_generation_keys:
                continue
            logger.warning(
                "Skipping unsupported local generation kwarg '%s' for model_role=%s.",
                key,
                self.model_role or "unknown",
            )
            self._logged_unsupported_generation_keys.add(key)

        return filtered_kwargs

    def _summarize_input_tensor_shapes(self, inputs: Any) -> str:
        if not isinstance(inputs, dict):
            return "inputs=unknown"
        parts: List[str] = []
        for key in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        ):
            value = inputs.get(key)
            if torch.is_tensor(value):
                parts.append(f"{key}={tuple(value.shape)}")
            elif value is not None:
                parts.append(f"{key}={type(value).__name__}")
        return ", ".join(parts) if parts else "inputs=no_known_tensors"

    def _extract_unsupported_generation_keys(
        self,
        exc: Exception,
        generation_kwargs: Dict[str, Any],
    ) -> List[str]:
        message = str(exc)
        unsupported_keys = [key for key in generation_kwargs if key in message]
        if unsupported_keys:
            return unsupported_keys

        match = re.search(r"\[(.*?)\]", message)
        if not match:
            return []

        extracted = []
        for raw_key in match.group(1).split(","):
            key = raw_key.strip().strip("'\"")
            if key and key in generation_kwargs:
                extracted.append(key)
        return extracted

    def _should_apply_multimodal_generation_stability_guard(
        self,
        *,
        use_multimodal_prompt: bool,
    ) -> bool:
        return (
            bool(use_multimodal_prompt)
            and self.device == "cuda"
            and self.accelerator_count > 1
            and self.device_map is not None
        )

    def _load_model_config_dict(self, model_path: str) -> Dict[str, Any]:
        try:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            if hasattr(config, "to_dict"):
                return config.to_dict()
        except Exception as exc:
            logger.warning("Failed to read model config for %s: %s", model_path, exc)
        return {}

    def _should_enable_multimodal_generation(self, config_dict: Dict[str, Any]) -> bool:
        role = str(self.model_role or "").strip().lower()
        if role in _TEXT_ONLY_MODEL_ROLES:
            return False
        if role in _VISION_MODEL_ROLES:
            return True
        return self._is_multimodal_config(config_dict)

    def _stringify_message_content(self, content: Any) -> str:
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "")).strip().lower()
                if item_type == "text":
                    text_value = " ".join(str(item.get("text", "")).split()).strip()
                    if text_value:
                        parts.append(text_value)
                elif item_type == "image":
                    parts.append("[image]")
                elif item_type == "video":
                    parts.append("[video]")
            return " ".join(part for part in parts if part).strip()
        return str(content or "").strip()

    def _normalize_text_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            normalized.append(
                {
                    "role": str(message.get("role", "user")).strip() or "user",
                    "content": self._stringify_message_content(message.get("content")),
                }
            )
        return normalized

    def _messages_include_user_turn(self, messages: List[Dict[str, Any]]) -> bool:
        return any(
            isinstance(message, dict)
            and str(message.get("role", "")).strip().lower() == "user"
            and str(self._stringify_message_content(message.get("content"))).strip()
            for message in messages or []
        )

    def _supports_chatml_prompt_fallback(self) -> bool:
        config_dict = getattr(self, "_model_config_dict", {}) or {}
        model_type = str(config_dict.get("model_type", "")).strip().lower()
        architectures = " ".join(str(item) for item in config_dict.get("architectures", []) or []).lower()
        chat_template = str(getattr(getattr(self, "tokenizer", None), "chat_template", "") or "")
        return "qwen" in model_type or "qwen" in architectures or "<|im_start|>" in chat_template

    def _render_text_prompt_fallback(
        self,
        text_messages: List[Dict[str, str]],
        *,
        enable_thinking: Optional[bool],
    ) -> str:
        if self._supports_chatml_prompt_fallback():
            parts: List[str] = []
            for message in text_messages:
                role = str(message.get("role", "user")).strip().lower() or "user"
                if role not in {"system", "user", "assistant", "tool"}:
                    role = "user"
                content = str(message.get("content", "") or "").strip()
                if not content:
                    continue
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            assistant_prefix = "<|im_start|>assistant\n"
            if enable_thinking is False:
                assistant_prefix += "<think>\n\n</think>\n\n"
            parts.append(assistant_prefix)
            return "\n".join(parts)

        joined = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in text_messages
            if str(message.get("content", "")).strip()
        )
        return joined + "\nassistant:"

    def _message_contains_visual_inputs(self, message: Dict[str, Any]) -> bool:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return False
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip().lower() in {"image", "video"}:
                return True
        return False

    def _messages_contain_visual_inputs(self, messages: List[Dict[str, Any]]) -> bool:
        return any(self._message_contains_visual_inputs(message) for message in messages or [])

    def _messages_contain_video_inputs(self, messages: List[Dict[str, Any]]) -> bool:
        for message in messages or []:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).strip().lower() == "video":
                    return True
        return False

    def _normalize_visual_reference(self, value: Any) -> str:
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

    def _normalize_multimodal_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user")).strip() or "user"
            content = message.get("content")
            if not isinstance(content, list):
                normalized.append({"role": role, "content": self._stringify_message_content(content)})
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
                    raw_image_value = item.get("image") or item.get("path") or item.get("url")
                    image_value = self._normalize_visual_reference(raw_image_value)
                    if image_value and not image_value.startswith(("http://", "https://", "file://", "data:")):
                        image_path = Path(str(image_value)).expanduser()
                        if image_path.exists() and not _is_readable_visual_file(image_path.resolve()):
                            source_images = list(item.get("source_images") or [])
                            logger.warning(
                                "Unreadable multimodal image input detected: %s. Trying to rebuild from source_images=%s",
                                image_path,
                                source_images,
                            )
                            rebuilt_image_value = _rebuild_visual_reference_from_sources(
                                image_path.resolve(),
                                source_images,
                            )
                            if rebuilt_image_value:
                                logger.warning(
                                    "Recovered unreadable multimodal image input %s -> %s",
                                    image_path,
                                    rebuilt_image_value,
                                )
                                image_value = self._normalize_visual_reference(rebuilt_image_value)
                            else:
                                logger.warning(
                                    "Skipping unreadable multimodal image input %s because rebuild failed. "
                                    "Its timestamp will not be passed to the model.",
                                    image_path,
                                )
                                image_value = ""
                    if image_value:
                        image_item = {"type": "image", "image": image_value}
                        for passthrough_key in ("min_pixels", "max_pixels"):
                            passthrough_value = item.get(passthrough_key)
                            if passthrough_value not in (None, ""):
                                image_item[passthrough_key] = passthrough_value
                        content_items.append(image_item)
                elif item_type == "video":
                    raw_video_value = item.get("video") or item.get("path") or item.get("url")
                    if isinstance(raw_video_value, (list, tuple)):
                        video_value = []
                        for frame in raw_video_value:
                            if frame is None:
                                continue
                            if isinstance(frame, Image.Image):
                                video_value.append(frame)
                                continue
                            normalized_frame = self._normalize_visual_reference(frame)
                            if normalized_frame:
                                video_value.append(normalized_frame)
                    else:
                        video_value = self._normalize_visual_reference(raw_video_value)
                    if video_value:
                        video_item = {"type": "video", "video": video_value}
                        for passthrough_key in (
                            "fps",
                            "sample_fps",
                            "raw_fps",
                            "max_frames",
                            "min_pixels",
                            "max_pixels",
                            "do_sample_frames",
                        ):
                            passthrough_value = item.get(passthrough_key)
                            if passthrough_value not in (None, ""):
                                video_item[passthrough_key] = passthrough_value
                        content_items.append(video_item)

            if not content_items:
                content_items.append({"type": "text", "text": ""})
            normalized.append({"role": role, "content": content_items})
        return normalized

    def _resolve_process_vision_info(self):
        if self._process_vision_info is not None:
            return self._process_vision_info
        try:
            from qwen_vl_utils.vision_process import process_vision_info  # type: ignore
        except Exception:
            from qwen_vl_utils import process_vision_info  # type: ignore
        self._process_vision_info = process_vision_info
        return self._process_vision_info

    def _iter_model_float_parameters(self, module: Any):
        if module is None:
            return
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return
        try:
            iterator = parameters()
        except Exception:
            return
        for param in iterator:
            if torch.is_tensor(param) and param.is_floating_point():
                yield param

    def _resolve_generation_input_float_dtype(self) -> Optional[torch.dtype]:
        for module in (
            getattr(self.model, "visual", None),
            getattr(self.model, "vision_model", None),
            self.model,
        ):
            for param in self._iter_model_float_parameters(module):
                return param.dtype

        try:
            return self._get_optimal_dtype()
        except Exception:
            return None

    def _apply_chat_template(
        self,
        chat_client: Any,
        messages: List[Dict[str, Any]],
        enable_thinking: Optional[bool] = None,
    ) -> str:
        thinking_flag = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        apply_chat_template = getattr(chat_client, "apply_chat_template", None)
        text_messages = self._normalize_text_messages(messages)
        if callable(apply_chat_template) and self._messages_include_user_turn(text_messages):
            try:
                # Hugging Face forwards template variables as top-level
                # keyword arguments.  Passing a nested `chat_template_kwargs`
                # silently leaves Qwen's `enable_thinking` undefined, which
                # makes the template enter its default thinking branch.
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking_flag,
                )
            except Exception as first_exc:
                logger.debug(
                    "Chat template rejected top-level enable_thinking=%s for %s; retrying chat_template_kwargs. error=%s",
                    thinking_flag,
                    type(chat_client).__name__,
                    first_exc,
                )
            second_error = ""
            try:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": thinking_flag},
                )
            except Exception as second_exc:
                second_error = str(second_exc)
                logger.debug(
                    "Chat template rejected chat_template_kwargs.enable_thinking=%s for %s; retrying without thinking control. error=%s",
                    thinking_flag,
                    type(chat_client).__name__,
                    second_exc,
                )
            if not (thinking_flag is False and self._supports_chatml_prompt_fallback()):
                try:
                    return apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as third_exc:
                    if thinking_flag is False:
                        logger.warning(
                            "[ThinkingBoundary][DIAG] source=TransformersProvider._apply_chat_template role=%s no-think chat template arguments were rejected by %s; falling back to local ChatML/plain prompt. error=%s",
                            self.model_role or "<unknown>",
                            type(chat_client).__name__,
                            third_exc,
                        )
            else:
                logger.debug(
                    "Skipping uncontrolled chat-template retry for no-think ChatML-compatible model after error=%s",
                    second_error,
                )
        elif callable(apply_chat_template):
            logger.debug("Skipping chat template rendering because messages contain no user turn.")
        return self._render_text_prompt_fallback(text_messages, enable_thinking=thinking_flag)

    def _move_inputs_to_generation_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        target_device = self._resolve_input_device()
        target_float_dtype = self._resolve_generation_input_float_dtype()
        normalized_inputs = _normalize_processor_sequence_tensors(
            dict(inputs or {}),
            source="provider.move_inputs",
        )
        for key, value in normalized_inputs.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                if (
                    key in _FLOAT_MULTIMODAL_TENSOR_KEYS
                    and value.is_floating_point()
                    and target_float_dtype is not None
                ):
                    moved[key] = value.to(device=target_device, dtype=target_float_dtype)
                else:
                    moved[key] = value.to(target_device)
            else:
                moved[key] = value
        return moved

    def _load_process_vision_payload(
        self,
        normalized_messages: List[Dict[str, Any]],
    ) -> tuple[Any, Any, Dict[str, Any]]:
        process_vision_info = self._resolve_process_vision_info()
        has_video_inputs = self._messages_contain_video_inputs(normalized_messages)
        def _is_empty_video_payload(payload: Any) -> bool:
            if payload is None:
                return True
            if isinstance(payload, (list, tuple, dict, str, bytes)):
                return len(payload) == 0
            return False

        kwargs = {
            "image_patch_size": 16,
            "return_video_metadata": True,
            "return_video_kwargs": True,
        }
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info([normalized_messages], **kwargs)
            if has_video_inputs and _is_empty_video_payload(video_inputs):
                raise RuntimeError("process_vision_info returned no video inputs for a video message.")
            return image_inputs, video_inputs, _drop_none_mapping_values(dict(video_kwargs or {}))
        except StopIteration as exc:
            logger.warning(
                "process_vision_info raised StopIteration for video payload; retrying without metadata/video kwargs."
            )
            try:
                image_inputs, video_inputs = process_vision_info([normalized_messages])
            except StopIteration as retry_exc:
                raise RuntimeError(
                    "Multimodal video preprocessing raised StopIteration before model generation."
                ) from retry_exc
            if has_video_inputs and _is_empty_video_payload(video_inputs):
                raise RuntimeError("process_vision_info returned no video inputs for a video message.")
            return image_inputs, video_inputs, {}
        except TypeError:
            image_inputs, video_inputs = process_vision_info([normalized_messages])
            if has_video_inputs and _is_empty_video_payload(video_inputs):
                raise RuntimeError("process_vision_info returned no video inputs for a video message.")
            return image_inputs, video_inputs, {}

    def _call_multimodal_processor(
        self,
        processor_kwargs: Dict[str, Any],
        *,
        require_video_grid: bool,
    ) -> Dict[str, Any]:
        cleaned_kwargs = _drop_none_mapping_values(dict(processor_kwargs or {}))
        attempts: List[tuple[str, Dict[str, Any]]] = [("full_payload", cleaned_kwargs)]

        def _append_attempt(label: str, payload: Dict[str, Any]) -> None:
            candidate = _drop_none_mapping_values(payload)
            if any(candidate == existing_payload for _, existing_payload in attempts):
                return
            attempts.append((label, candidate))

        auxiliary_keys = [
            key for key in cleaned_kwargs.keys()
            if key not in _VIDEO_PROCESSOR_CORE_KWARGS
        ]
        if auxiliary_keys:
            reduced_aux = {
                key: value
                for key, value in cleaned_kwargs.items()
                if key in _VIDEO_PROCESSOR_CORE_KWARGS
            }
            _append_attempt("without_video_aux_kwargs", reduced_aux)

        if "video_metadata" in cleaned_kwargs:
            reduced = dict(cleaned_kwargs)
            reduced.pop("video_metadata", None)
            _append_attempt("without_video_metadata", reduced)

        if any(key in cleaned_kwargs for key in ("fps", "max_frames", "do_sample_frames")):
            reduced_more = dict(attempts[-1][1])
            for key in ("fps", "max_frames", "do_sample_frames"):
                reduced_more.pop(key, None)
            _append_attempt("without_video_sampling_kwargs", reduced_more)

        last_exc: Optional[Exception] = None
        for index, (label, attempt_kwargs) in enumerate(attempts):
            try:
                inputs = self.processor(**attempt_kwargs)
                inputs = _normalize_processor_sequence_tensors(
                    dict(inputs or {}),
                    source=f"provider_multimodal_processor.{label}",
                )
                if require_video_grid and (
                    "video_grid_thw" not in inputs
                    or not torch.is_tensor(inputs["video_grid_thw"])
                    or inputs["video_grid_thw"].numel() == 0
                ):
                    raise RuntimeError(
                        "Multimodal processor did not return video_grid_thw for a video input. "
                        "This usually means the local qwen_vl_utils / transformers stack did not parse the clip correctly."
                    )
                return inputs
            except StopIteration as exc:
                last_exc = exc
                if index + 1 < len(attempts):
                    logger.warning(
                        "Multimodal processor raised StopIteration during %s for role=%s; retrying with a simpler video payload.",
                        label,
                        self.model_role or "unknown",
                    )
                    continue
                raise RuntimeError(
                    "Multimodal processor raised StopIteration while preparing video inputs."
                ) from exc
            except TypeError as exc:
                last_exc = exc
                if index + 1 < len(attempts):
                    logger.warning(
                        "Multimodal processor rejected %s payload for role=%s (%s); retrying with a simpler video payload.",
                        label,
                        self.model_role or "unknown",
                        exc,
                    )
                    continue
                raise
            except AttributeError as exc:
                last_exc = exc
                if index + 1 < len(attempts):
                    logger.warning(
                        "Multimodal processor crashed during %s for role=%s (%s); retrying with a simpler video payload.",
                        label,
                        self.model_role or "unknown",
                        exc,
                    )
                    continue
                raise
            except RuntimeError as exc:
                last_exc = exc
                if index + 1 < len(attempts) and "video_grid_thw" in str(exc):
                    logger.warning(
                        "Multimodal processor returned incomplete %s payload for role=%s (%s); retrying with a simpler video payload.",
                        label,
                        self.model_role or "unknown",
                        exc,
                    )
                    continue
                raise

        if isinstance(last_exc, StopIteration):
            raise RuntimeError(
                "Multimodal processor raised StopIteration while preparing video inputs."
            ) from last_exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Multimodal processor failed without returning inputs.")

    def _resolve_config_num_hidden_layers(self, config_dict: Optional[Dict[str, Any]] = None) -> Optional[int]:
        resolved = config_dict or self._model_config_dict or {}
        candidates = [
            resolved.get("num_hidden_layers"),
            ((resolved.get("text_config") or {}) if isinstance(resolved.get("text_config"), dict) else {}).get("num_hidden_layers"),
            ((resolved.get("language_config") or {}) if isinstance(resolved.get("language_config"), dict) else {}).get("num_hidden_layers"),
            ((resolved.get("llm_config") or {}) if isinstance(resolved.get("llm_config"), dict) else {}).get("num_hidden_layers"),
        ]
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _build_explicit_multimodal_device_map(
        self,
        model_path: str,
        device_count: int,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        num_layers = self._resolve_config_num_hidden_layers(config_dict)
        if not num_layers or device_count <= 1:
            logger.warning(
                "Cannot build explicit multimodal device map (layers=%s, devices=%d). Falling back to balanced.",
                num_layers,
                device_count,
            )
            return "balanced"

        device_map: Dict[str, int] = {
            "model.visual": 0,
            "model.language_model.embed_tokens": 0,
            "model.language_model.rotary_emb": 0,
            "model.language_model.norm": 0,
            "lm_head": 0,
        }

        if num_layers == 1:
            device_map["model.language_model.layers.0"] = 0
            return device_map

        device_map["model.language_model.layers.0"] = 0
        device_map[f"model.language_model.layers.{num_layers - 1}"] = 0

        middle_layers = list(range(1, max(1, num_layers - 1)))
        worker_devices = list(range(1, device_count)) or [0]
        if middle_layers and worker_devices:
            base_count = len(middle_layers) // len(worker_devices)
            remainder = len(middle_layers) % len(worker_devices)
            cursor = 0
            for device_index, dev_id in enumerate(worker_devices):
                count = base_count + (1 if device_index < remainder else 0)
                for _ in range(count):
                    if cursor >= len(middle_layers):
                        break
                    layer_idx = middle_layers[cursor]
                    device_map[f"model.language_model.layers.{layer_idx}"] = dev_id
                    cursor += 1

        logger.info(
            "Explicit multimodal CUDA device map: %d layers -> %d devices (visual/embed/head on dev0).",
            num_layers,
            device_count,
        )
        return device_map

    def _finalize_device_map(
        self,
        model_path: str,
        resolved_device_map: Any,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if getattr(self, "_requested_device_map", None) is not None:
            return resolved_device_map

        role = str(self.model_role or "").strip().lower()
        if (
            resolved_device_map is None
            or self.device != "cuda"
            or self.accelerator_count <= 1
        ):
            return resolved_device_map

        if role in _TEXT_ONLY_MODEL_ROLES:
            return self._build_explicit_device_map(
                model_path,
                self.accelerator_count,
                config_dict=config_dict,
            )

        if role in _VISION_MODEL_ROLES or self._is_multimodal_config(config_dict or {}):
            return self._build_explicit_multimodal_device_map(
                model_path,
                self.accelerator_count,
                config_dict=config_dict,
            )
        return self._build_explicit_device_map(
            model_path,
            self.accelerator_count,
            config_dict=config_dict,
        )

    @staticmethod
    def _is_out_of_memory_error(error: BaseException) -> bool:
        if torch is not None and isinstance(error, torch.OutOfMemoryError):
            return True
        cause = getattr(error, "__cause__", None)
        if cause is not None and cause is not error and TransformersProvider._is_out_of_memory_error(cause):
            return True
        message = str(error or "").lower()
        return "out of memory" in message or "ran out of memory" in message

    def _can_retry_load_on_multi_device(self) -> bool:
        return (
            self.device == "cuda"
            and self.accelerator_count > 1
            and self.device_map is None
            and self._requested_device_map is None
            and not bool(getattr(settings, "force_single_device", False))
        )

    def _clear_model_load_buffers(self) -> None:
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if self.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.empty_cache()
            elif self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _prepare_multimodal_inputs(
        self,
        messages: List[Dict[str, Any]],
        enable_thinking: Optional[bool] = None,
        move_to_device: bool = True,
    ) -> Dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Multimodal processor is not initialized.")

        normalized_messages = self._normalize_multimodal_messages(messages)
        try:
            prompt_text = self._apply_chat_template(
                self.processor,
                normalized_messages,
                enable_thinking=enable_thinking,
            )
        except StopIteration as exc:
            raise RuntimeError(
                "Multimodal chat template raised StopIteration while preparing video inputs."
            ) from exc
        image_inputs, video_inputs, video_kwargs = self._load_process_vision_payload(normalized_messages)
        videos = None
        video_metadata = None
        has_video_inputs = self._messages_contain_video_inputs(normalized_messages)
        if isinstance(video_inputs, (list, tuple)) and video_inputs:
            first_item = video_inputs[0]
            if isinstance(first_item, tuple) and len(first_item) >= 2:
                videos = [item[0] for item in video_inputs]
                normalized_metadata = []
                for item in video_inputs:
                    normalized_item = _normalize_video_metadata_item(item[1])
                    if normalized_item:
                        normalized_metadata.append(normalized_item)
                video_metadata = normalized_metadata or None
            else:
                videos = video_inputs
        elif has_video_inputs:
            raise RuntimeError("Multimodal video preprocessing returned an empty video payload.")
        processor_kwargs: Dict[str, Any] = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
            "do_resize": False,
        }
        processor_kwargs.update(video_kwargs)
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if videos is not None:
            processor_kwargs["videos"] = videos
        if video_metadata:
            processor_kwargs["video_metadata"] = video_metadata
        processor_kwargs = _drop_none_mapping_values(processor_kwargs)
        inputs = self._call_multimodal_processor(
            processor_kwargs,
            require_video_grid=videos is not None,
        )
        inputs = _normalize_qwen_video_grid_thw(
            dict(inputs or {}),
            config_dict=getattr(self, "_model_config_dict", {}) or {},
            source="provider.prepare_multimodal_inputs",
        )
        return self._move_inputs_to_generation_device(inputs) if move_to_device else dict(inputs)

    def _release_multimodal_generation_buffers(self, *payloads: Any) -> None:
        for payload in payloads:
            if isinstance(payload, dict):
                payload.clear()
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if self.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.empty_cache()
            elif self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _decode_generation_tokens(
        self,
        token_ids: Any,
        *,
        use_multimodal_prompt: bool,
    ) -> str:
        if token_ids is None:
            return ""

        decode_input = token_ids
        if torch.is_tensor(token_ids):
            if token_ids.ndim == 1:
                if token_ids.numel() == 0:
                    return ""
                if (
                    use_multimodal_prompt
                    and self.processor is not None
                    and hasattr(self.processor, "batch_decode")
                ):
                    decode_input = token_ids.unsqueeze(0)
            elif token_ids.ndim >= 2 and token_ids.shape[-1] == 0:
                return ""

        try:
            if (
                use_multimodal_prompt
                and self.processor is not None
                and hasattr(self.processor, "batch_decode")
            ):
                decoded = self.processor.batch_decode(
                    decode_input,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return _sanitize_decoded_text(decoded[0] if decoded else "").strip()
            return _sanitize_decoded_text(
                self.tokenizer.decode(
                decode_input,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                )
            ).strip()
        except StopIteration:
            logger.warning(
                "Decoder raised StopIteration for %s output; treating it as empty and retrying fallback decode.",
                "multimodal" if use_multimodal_prompt else "text",
            )
            return ""

    def _select_auto_model_class(
        self,
        model_path: str,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """Select a compatible AutoModel class based on local config."""
        resolved_config_dict = config_dict or self._load_model_config_dict(model_path)
        if not resolved_config_dict:
            return AutoModelForCausalLM
        role = str(self.model_role or "").strip().lower()

        if role in _TEXT_ONLY_MODEL_ROLES:
            logger.info(
                "Selecting AutoModelForCausalLM for text role '%s' (model_type=%s, architectures=%s)",
                role,
                resolved_config_dict.get("model_type"),
                resolved_config_dict.get("architectures"),
            )
            return AutoModelForCausalLM

        if role in _VISION_MODEL_ROLES:
            model_cls = self._select_multimodal_auto_class(resolved_config_dict)
            if model_cls is not None:
                logger.info(
                    "Selecting multimodal model class %s for vision role '%s'",
                    getattr(model_cls, "__name__", str(model_cls)),
                    role,
                )
                return model_cls
            logger.warning(
                "Vision role '%s' did not resolve a multimodal loader from config=%s; falling back to AutoModelForCausalLM.",
                role,
                resolved_config_dict,
            )
            return AutoModelForCausalLM

        if self._is_multimodal_config(resolved_config_dict):
            model_cls = self._select_multimodal_auto_class(resolved_config_dict)
            if model_cls is not None:
                logger.info(
                    "Detected multimodal config (model_type=%s, architectures=%s), using %s",
                    resolved_config_dict.get("model_type"),
                    resolved_config_dict.get("architectures"),
                    getattr(model_cls, "__name__", str(model_cls)),
                )
                return model_cls

        return AutoModelForCausalLM

    def _is_multimodal_config(self, config_dict: Dict[str, Any]) -> bool:
        model_type = str(config_dict.get("model_type", "")).lower()
        if any(key in model_type for key in ("vl", "vision", "mllama", "multimodal")):
            return True

        if any(
            key in config_dict
            for key in (
                "vision_config",
                "visual",
                "image_token_id",
                "image_token_index",
                "image_start_id",
                "image_end_id",
                "video_token_id",
                "mm_processor_kwargs",
                "image_processor_type",
            )
        ):
            return True

        auto_map = config_dict.get("auto_map", {}) or {}
        auto_map_text = (
            " ".join(str(value).lower() for value in auto_map.values())
            if isinstance(auto_map, dict)
            else str(auto_map).lower()
        )
        if any(key in auto_map_text for key in ("vision", "image", "vl", "multimodal")):
            return True

        architectures = config_dict.get("architectures", []) or []
        for arch in architectures:
            arch_name = str(arch).lower()
            if any(key in arch_name for key in ("vl", "vision", "image", "multimodal")):
                return True
        return False

    def _select_multimodal_auto_class(self, config_dict: Dict[str, Any]):
        architectures = config_dict.get("architectures", []) or []
        for arch in architectures:
            if hasattr(transformers, arch):
                return getattr(transformers, arch)

        for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
            model_cls = getattr(transformers, cls_name, None)
            if model_cls is not None:
                return model_cls
        return None

    def _get_optimal_dtype(self) -> torch.dtype:
        """
        Determine optimal dtype based on device capabilities.
        
        Returns:
            torch.bfloat16 if supported by device, otherwise torch.float16
        """
        try:
            if self.device == "cuda":
                # Check CUDA device capability
                if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            elif self.device == "npu":
                # Check NPU bfloat16 support
                if hasattr(torch, 'npu') and hasattr(torch.npu, 'is_bf16_supported'):
                    if torch.npu.is_bf16_supported():
                        return torch.bfloat16
                # Fallback: Ascend NPUs generally support bfloat16 on 910B and later
                else:
                    logger.info("NPU bfloat16 support check unavailable, assuming support")
                    return torch.bfloat16
            if self.device == "cpu":
                return torch.float32
        except Exception as e:
            logger.debug(f"Error checking bfloat16 support: {e}")

        if self.device == "cpu":
            return torch.float32
        return torch.float16

    def _determine_device_map(
        self, 
        user_device_map: Optional[str], 
        device_type: str, 
        device_count: int
    ) -> Optional[str]:
        """
        Determine optimal device_map strategy based on device type and configuration.
        
        Strategy:
        - NPU: Single device by default (推理场景更稳定，避免设备间通信OOM)
        - CUDA: Single-device by default for batch=1 latency; explicit multi-device
          only when caller requests it or single-device load later proves insufficient
        - User override: Respect explicit device_map parameter
        
        Returns:
            device_map string or None for single device
        """
        from config.settings import settings
        
        # User explicitly set device_map
        if user_device_map is not None:
            return user_device_map
        
        # Single device or CPU: no need for device_map
        if device_type == "cpu" or device_count <= 1:
            return None
        
        # Force single device mode
        if settings.force_single_device:
            logger.info("force_single_device=True, using single device")
            return None

        # NPU: ALWAYS single-device for inference.
        # Ascend NPU does NOT support accelerate's device_map in any form
        # (balanced / sequential / explicit dict).  AlignDevicesHook performs
        # cross-device tensor copies via SDMA, which triggers 507001 copy_d2d
        # errors on all tested Ascend 910B firmware versions.
        # NPU multi-card is ONLY possible via DDP/HCCL for *training*.
        if device_type == "npu":
            if settings.enable_npu_multi_device:
                logger.warning(
                    "ENABLE_NPU_MULTI_DEVICE=true is set but Ascend NPU does NOT "
                    "support accelerate device_map (causes 507001 SDMA copy_d2d "
                    "errors).  Forcing single-device inference.  NPU multi-card "
                    "is only supported via DDP/HCCL for training workloads."
                )
            logger.info(
                "NPU detected (%d NPU(s)). Using single-device inference on %s.",
                device_count, self.runtime_device,
            )
            return None
        
        # CUDA multi-device: stay single-device by default. If a caller
        # explicitly requests sharding we preserve it; otherwise __init__
        # may still retry on multi-device after a real single-device OOM.
        if device_type == "cuda":
            logger.info(
                "CUDA multi-device detected (%d GPUs), defaulting to single-device local inference unless an explicit device_map or OOM fallback selects sharding.",
                device_count,
            )
            return None
        
        return None

    def _build_explicit_device_map(
        self,
        model_path: str,
        device_count: int,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Build explicit layer-to-device mapping for text-model CUDA multi-card distribution.

        Auto-allocation strategies (balanced/sequential) may place the entire model
        on a single device when it numerically fits.  This method reads
        ``num_hidden_layers`` from the model config and pins transformer blocks
        to specific devices round-robin, guaranteeing multi-card utilisation.

        Handles **weight tying**: many LLMs (including Qwen3) share weights
        between ``model.embed_tokens`` and ``lm_head``.  If
        ``tie_word_embeddings`` is true the two modules MUST reside on the
        same device, otherwise accelerate's ``dispatch_model`` will attempt a
        cross-device copy that crashes on NPU (507001 copy_d2d).

        Returns:
            Dict[str, int] mapping module names to device ids,
            or ``"sequential"`` as fallback when config is unavailable.
        """
        try:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            logger.warning("Failed to load model config for device map: %s", e)
            return "sequential"
        num_layers = self._resolve_config_num_hidden_layers(config_dict or config.to_dict())
        if not num_layers or num_layers < device_count:
            logger.warning(
                "Cannot build explicit text device map (layers=%s, devices=%d). "
                "Falling back to sequential.",
                num_layers, device_count,
            )
            return "sequential"

        # Check weight tying: lm_head must be on the same device as embed_tokens
        tie_word_embeddings = getattr(config, "tie_word_embeddings", True)

        device_map: Dict[str, int] = {}
        # Embedding layer -> first device
        device_map["model.embed_tokens"] = 0

        # Distribute transformer layers evenly across devices
        layers_per_device = num_layers // device_count
        remainder = num_layers % device_count

        layer_idx = 0
        for dev_id in range(device_count):
            count = layers_per_device + (1 if dev_id < remainder else 0)
            for _ in range(count):
                device_map[f"model.layers.{layer_idx}"] = dev_id
                layer_idx += 1

        last_device = device_count - 1

        if tie_word_embeddings:
            # lm_head shares weights with embed_tokens -> must be on dev0.
            # Place model.norm on dev0 as well (feeds directly into lm_head).
            device_map["model.norm"] = 0
            device_map["lm_head"] = 0
            head_dev = 0
        else:
            # No tying: norm + head on last device for pipeline balance.
            device_map["model.norm"] = last_device
            device_map["lm_head"] = last_device
            head_dev = last_device

        logger.info(
            "Explicit text CUDA device map: %d layers -> %d devices "
            "(~%d layers/device, embed->dev0, head->dev%d, tie_weights=%s)",
            num_layers, device_count, layers_per_device, head_dev,
            tie_word_embeddings,
        )
        return device_map

    def _get_max_memory_config(self) -> Optional[Dict[int, str]]:
        """
        Configure per-device maximum memory to prevent OOM.
        
        Returns:
            Dict mapping device id to memory limit string (e.g., {0: '20GB', 1: '20GB'})
        """
        from config.settings import settings
        
        if settings.per_device_max_memory_gb is None:
            return None
        
        max_memory = {}
        for i in range(self.accelerator_count):
            max_memory[i] = f"{settings.per_device_max_memory_gb}GB"
        
        logger.info(f"Setting max memory per device: {max_memory}")
        return max_memory

    # Context management overrides

    def count_tokens(self, text: str) -> int:
        """Count tokens using the actual tokenizer."""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, len(text) // 4)

    def count_message_tokens(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> int:
        """Count tokens after rendering the provider chat template."""
        if self.processor is not None and self._messages_contain_visual_inputs(messages):
            try:
                inputs = self._prepare_multimodal_inputs(
                    messages,
                    enable_thinking=kwargs.get("enable_thinking", self.enable_thinking),
                    move_to_device=False,
                )
                input_ids = inputs.get("input_ids")
                if torch.is_tensor(input_ids):
                    return int(input_ids.shape[1])
            except Exception as exc:
                logger.debug("Falling back to text-only token counting after multimodal count failure: %s", exc)
        try:
            prompt = self._build_prompt(
                messages,
                enable_thinking=kwargs.get("enable_thinking", self.enable_thinking),
            )
        except Exception:
            return super().count_message_tokens(messages, **kwargs)
        return self.count_tokens(prompt)

    def get_max_context_length(self) -> int:
        """Return the effective context length used by runtime budgeting.

        The runtime's hard admission check in ``generate`` is governed by
        ``settings.max_input_tokens``. Budgeting and folding decisions must use
        the same ceiling; otherwise logs can report a smaller window (for
        example 8192 from tokenizer metadata) while generation actually accepts
        a larger configured limit.
        """
        from config.settings import settings

        configured_limit = int(getattr(settings, "max_input_tokens", 0) or 0)
        if configured_limit > 0:
            return max(512, configured_limit)

        length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(length, int) and 0 < length <= 1_000_000:
            return length
        config = getattr(self.model, "config", None)
        if config:
            for attr in ("max_position_embeddings", "n_positions", "seq_length"):
                val = getattr(config, attr, None)
                if isinstance(val, int) and val > 0:
                    return val
        return 8192

    @staticmethod
    def _normalize_hf_mapped_device(mapped_device: Any, *, device_type: Optional[str] = None) -> Optional[str]:
        if isinstance(mapped_device, int):
            normalized_type = str(device_type or "cuda").strip().lower() or "cuda"
            return f"{normalized_type}:{mapped_device}"
        if isinstance(mapped_device, str) and mapped_device not in ("cpu", "disk"):
            return mapped_device
        return None

    def _resolve_mapped_module_device(self, prefixes: Tuple[str, ...]) -> Optional[str]:
        hf_map = getattr(self.model, "hf_device_map", None)
        if not isinstance(hf_map, dict) or not hf_map:
            return None

        normalized_prefixes = tuple(
            str(prefix).strip()
            for prefix in prefixes
            if str(prefix).strip()
        )
        for prefix in normalized_prefixes:
            for module_name, mapped_device in hf_map.items():
                normalized_name = str(module_name or "").strip()
                if normalized_name == prefix or normalized_name.startswith(prefix + "."):
                    resolved = self._normalize_hf_mapped_device(
                        mapped_device,
                        device_type=self.device,
                    )
                    if resolved:
                        return resolved

        for mapped_device in hf_map.values():
            resolved = self._normalize_hf_mapped_device(
                mapped_device,
                device_type=self.device,
            )
            if resolved:
                return resolved
        return None

    def _resolve_visual_input_device(self) -> str:
        resolved = self._resolve_mapped_module_device(
            (
                "model.visual",
                "visual",
            )
        )
        return resolved or self.runtime_device

    def _resolve_input_device(self) -> str:
        """Resolve where input tensors should be placed for generation."""
        hf_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_map, dict) and hf_map:
            for mapped_device in hf_map.values():
                resolved = self._normalize_hf_mapped_device(
                    mapped_device,
                    device_type=self.device,
                )
                if resolved:
                    return resolved
        return self.runtime_device

    def _build_prompt(self, messages: List[Dict[str, str]], enable_thinking: Optional[bool] = None) -> str:
        thinking_flag = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        if self.processor is not None and self._messages_contain_visual_inputs(messages):
            normalized_messages = self._normalize_multimodal_messages(messages)
            try:
                return self._apply_chat_template(
                    self.processor,
                    normalized_messages,
                    enable_thinking=thinking_flag,
                )
            except Exception as exc:
                logger.debug("Failed to render multimodal chat template; degrading to text-only prompt: %s", exc)

        text_messages = self._normalize_text_messages(messages)
        if self._messages_include_user_turn(text_messages):
            try:
                # See `_apply_chat_template`: template variables belong at
                # the top level for Hugging Face chat templates.
                return self.tokenizer.apply_chat_template(
                    text_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking_flag,
                )
            except Exception as first_exc:
                logger.debug(
                    "Tokenizer rejected top-level enable_thinking=%s; retrying chat_template_kwargs. error=%s",
                    thinking_flag,
                    first_exc,
                )
                try:
                    return self.tokenizer.apply_chat_template(
                        text_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": thinking_flag},
                    )
                except Exception as second_exc:
                    second_error = str(second_exc)
                    logger.debug(
                        "Tokenizer rejected chat_template_kwargs.enable_thinking=%s; retrying without thinking control. error=%s",
                        thinking_flag,
                        second_exc,
                    )
                if not (thinking_flag is False and self._supports_chatml_prompt_fallback()):
                    try:
                        return self.tokenizer.apply_chat_template(
                            text_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception as third_exc:
                        if thinking_flag is False:
                            logger.warning(
                                "[ThinkingBoundary][DIAG] source=TransformersProvider._build_prompt role=%s no-think chat template arguments were rejected by tokenizer; falling back to local ChatML/plain prompt. error=%s",
                                self.model_role or "<unknown>",
                                third_exc,
                            )
                else:
                    logger.debug(
                        "Skipping uncontrolled tokenizer chat-template retry for no-think ChatML-compatible model after error=%s",
                        second_error,
                    )
        else:
            logger.debug("Skipping tokenizer chat template rendering because messages contain no user turn.")
        return self._render_text_prompt_fallback(text_messages, enable_thinking=thinking_flag)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        from config.settings import settings

        generation_enable_thinking = _resolve_enable_thinking_for_role(
            self.model_role,
            kwargs.pop("enable_thinking", None),
            fallback=bool(self.enable_thinking),
            source="TransformersProvider.generate",
            provider_name="transformers",
            respect_role_config=bool(kwargs.pop("respect_role_generation_config", True)),
        )
        wall_clock_deadline_monotonic = kwargs.pop("wall_clock_deadline_monotonic", None)
        generation_deadline_monotonic = kwargs.pop("generation_deadline_monotonic", None)
        use_multimodal_prompt = (
            self.processor is not None
            and self._supports_multimodal_generation
            and self._messages_contain_visual_inputs(messages)
        )
        prompt = ""
        is_npu_multi_device = (
            self.device == "npu"
            and self.accelerator_count > 1
            and self.device_map is not None
        )
        
        # Use conservative max input length to avoid OOM
        max_input_length = settings.max_input_tokens
        if is_npu_multi_device:
            max_input_length = min(max_input_length, 2048)

        if not use_multimodal_prompt:
            prompt = self._build_prompt(messages, enable_thinking=generation_enable_thinking)
        
        # Use conservative max output tokens
        max_new_tokens = int(max_tokens or self.max_tokens)
        if (
            bool(getattr(settings, "enforce_max_output_tokens", False))
            and settings.max_output_tokens
            and int(settings.max_output_tokens) > 0
        ):
            max_new_tokens = min(max_new_tokens, int(settings.max_output_tokens))

        if is_npu_multi_device:
            gen_temperature = 1.0  # Use default value instead of 0 to avoid edge cases
            gen_do_sample = False
        else:
            explicit_do_sample = kwargs.pop("do_sample", None)
            gen_temperature = temperature if temperature is not None else self.temperature
            if explicit_do_sample is None:
                # Default sampling behavior follows the effective temperature.
                gen_do_sample = gen_temperature > 0
            else:
                gen_do_sample = bool(explicit_do_sample)
                if gen_do_sample:
                    try:
                        if gen_temperature is None or float(gen_temperature) <= 0:
                            gen_temperature = 1.0
                    except Exception:
                        gen_temperature = 1.0

        gen_top_p = kwargs.pop("top_p", self.top_p)
        gen_top_k = kwargs.pop("top_k", self.top_k)
        gen_min_p = kwargs.pop("min_p", self.min_p)
        gen_repetition_penalty = kwargs.pop("repetition_penalty", self.repetition_penalty)
        gen_presence_penalty = kwargs.pop("presence_penalty", self.presence_penalty)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "use_cache": not is_npu_multi_device,
        }
        apply_multimodal_stability_guard = self._should_apply_multimodal_generation_stability_guard(
            use_multimodal_prompt=use_multimodal_prompt,
        )

        # Critical: do_sample must be set BEFORE temperature
        # to ensure HuggingFace accepts the temperature parameter
        if gen_do_sample:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = gen_temperature
            if gen_top_p is not None:
                generation_kwargs["top_p"] = gen_top_p
            if gen_top_k is not None:
                generation_kwargs["top_k"] = gen_top_k
            if gen_min_p is not None:
                generation_kwargs["min_p"] = gen_min_p
        else:
            generation_kwargs["do_sample"] = False
            # When do_sample=False, explicitly avoid setting temperature/top_p/top_k

        if gen_repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = gen_repetition_penalty
        if gen_presence_penalty is not None:
            generation_kwargs["presence_penalty"] = gen_presence_penalty
        if apply_multimodal_stability_guard:
            # Align local multimodal multi-GPU generation with the rollout-side
            # NaN/Inf guard instead of changing config-owned decode parameters.
            generation_kwargs["remove_invalid_values"] = True
            generation_kwargs["renormalize_logits"] = True
            generation_kwargs["synced_gpus"] = False
        effective_deadline_monotonic = None
        for raw_deadline in (wall_clock_deadline_monotonic, generation_deadline_monotonic):
            if raw_deadline in (None, ""):
                continue
            try:
                candidate_deadline = float(raw_deadline)
            except Exception:
                continue
            if effective_deadline_monotonic is None or candidate_deadline < effective_deadline_monotonic:
                effective_deadline_monotonic = candidate_deadline
        if effective_deadline_monotonic is not None:
            try:
                remaining_seconds = effective_deadline_monotonic - time.perf_counter()
            except Exception:
                remaining_seconds = None
            if remaining_seconds is not None and remaining_seconds > 0:
                existing_max_time = generation_kwargs.get("max_time")
                if existing_max_time is None:
                    generation_kwargs["max_time"] = remaining_seconds
                else:
                    generation_kwargs["max_time"] = min(float(existing_max_time), remaining_seconds)
        generation_kwargs = self._filter_generation_kwargs(generation_kwargs)
        inputs: Any = {}
        output = None
        generated_ids = None
        lock_context = self._generation_lock if self._generation_lock is not None else nullcontext()
        try:
            with lock_context:
                if use_multimodal_prompt:
                    inputs = self._prepare_multimodal_inputs(
                        messages,
                        enable_thinking=generation_enable_thinking,
                        move_to_device=True,
                    )
                else:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=False,
                    )
                    inputs = inputs.to(self._resolve_input_device())
                input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
                if not torch.is_tensor(input_ids):
                    raise RuntimeError("Provider inputs are missing input_ids after prompt preprocessing.")
                input_length = int(input_ids.shape[1])
                if input_length > max_input_length:
                    raise ValueError(
                        f"Prompt requires {input_length} tokens, exceeding configured max_input_tokens="
                        f"{max_input_length}. Silent truncation is disabled; reduce context or enable memory folding."
                    )
                if transformers is not None and "stopping_criteria" not in generation_kwargs:
                    try:
                        generation_kwargs["stopping_criteria"] = transformers.StoppingCriteriaList(
                            [_StopOnStructuredJsonOrSyntheticRoleTurn(self.tokenizer, prompt_length=input_length)]
                        )
                    except Exception as exc:
                        logger.debug(
                            "Failed to install synthetic role-turn stopping criteria for model_role=%s: %s",
                            self.model_role or "unknown",
                            exc,
                        )

                logger.info(
                    "Starting generation: role=%s prompt_mode=%s input_tokens=%d max_new_tokens=%d do_sample=%s use_cache=%s thinking=%s",
                    self.model_role or "unknown",
                    "multimodal" if use_multimodal_prompt else "text",
                    input_length,
                    int(max_new_tokens),
                    bool(gen_do_sample),
                    bool(generation_kwargs.get("use_cache", False)),
                    bool(generation_enable_thinking),
                )
                if apply_multimodal_stability_guard:
                    logger.info(
                        "[MultimodalGenerate][DIAG] Applying local CUDA multimodal stability guard: "
                        "use_cache=%s remove_invalid_values=True renormalize_logits=True synced_gpus=False",
                        bool(generation_kwargs.get("use_cache", False)),
                    )
                with torch.no_grad():
                    generation_started = time.perf_counter()
                    try:
                        output = self.model.generate(
                            **inputs,
                            **generation_kwargs,
                        )
                    except StopIteration as exc:
                        input_summary = self._summarize_input_tensor_shapes(inputs)
                        if use_multimodal_prompt:
                            logger.warning(
                                "Model.generate raised StopIteration for role=%s multimodal prompt (%s); retrying once with use_cache=False.",
                                self.model_role or "unknown",
                                input_summary,
                            )
                            retry_variants: List[Tuple[str, Dict[str, Any]]] = []

                            retry_generation_kwargs = dict(generation_kwargs)
                            retry_generation_kwargs["use_cache"] = False
                            retry_variants.append(("use_cache_false", retry_generation_kwargs))

                            retry_without_stop = dict(retry_generation_kwargs)
                            retry_without_stop.pop("stopping_criteria", None)
                            retry_variants.append(("use_cache_false_no_stopping", retry_without_stop))

                            retry_minimal = dict(retry_without_stop)
                            retry_minimal.pop("remove_invalid_values", None)
                            retry_minimal.pop("renormalize_logits", None)
                            retry_minimal.pop("synced_gpus", None)
                            retry_variants.append(("minimal_multimodal_kwargs", retry_minimal))

                            retry_compact = dict(retry_minimal)
                            try:
                                retry_compact["max_new_tokens"] = max(
                                    128,
                                    min(int(retry_compact.get("max_new_tokens", max_new_tokens) or max_new_tokens), 512),
                                )
                            except Exception:
                                retry_compact["max_new_tokens"] = 512
                            retry_variants.append(("minimal_multimodal_kwargs_compact_output", retry_compact))

                            input_variants: List[Tuple[str, Dict[str, Any]]] = [("base_inputs", dict(inputs or {}))]
                            sanitized_inputs = dict(inputs or {})
                            removed_input_keys: List[str] = []
                            for unstable_key in ("mm_token_type_ids", "second_per_grid_ts"):
                                if unstable_key in sanitized_inputs:
                                    sanitized_inputs.pop(unstable_key, None)
                                    removed_input_keys.append(unstable_key)
                            if removed_input_keys:
                                input_variants.append(("drop_optional_inputs", sanitized_inputs))

                            retry_exc: Optional[BaseException] = exc
                            for retry_label, retry_kwargs in retry_variants:
                                for input_variant_label, retry_inputs in input_variants:
                                    try:
                                        self._clear_model_load_buffers()
                                    except Exception:
                                        pass
                                    try:
                                        variant_summary = self._summarize_input_tensor_shapes(retry_inputs)
                                        logger.warning(
                                            "Retrying multimodal generate after StopIteration for role=%s with variant=%s input_variant=%s (%s).",
                                            self.model_role or "unknown",
                                            retry_label,
                                            input_variant_label,
                                            variant_summary,
                                        )
                                        output = self.model.generate(
                                            **retry_inputs,
                                            **retry_kwargs,
                                        )
                                        generation_kwargs = retry_kwargs
                                        inputs = retry_inputs
                                        retry_exc = None
                                        break
                                    except StopIteration as variant_exc:
                                        retry_exc = variant_exc
                                        continue
                                if retry_exc is None:
                                    break
                            if retry_exc is not None:
                                raise RuntimeError(
                                    "Model.generate raised StopIteration for multimodal prompt. "
                                    "This usually indicates an unstable multimodal/device_map runtime. "
                                    f"Input summary: {input_summary}"
                                ) from retry_exc
                        else:
                            raise RuntimeError(
                                "Model.generate raised StopIteration for text prompt."
                            ) from exc
                    except (TypeError, ValueError) as exc:
                        unsupported_keys = self._extract_unsupported_generation_keys(exc, generation_kwargs)
                        if not unsupported_keys:
                            raise
                        for key in unsupported_keys:
                            logger.warning("Model.generate does not support '%s'; retrying without it.", key)
                            generation_kwargs.pop(key, None)
                        output = self.model.generate(
                            **inputs,
                            **generation_kwargs,
                        )
                    logger.info(
                        "Generation finished: prompt_mode=%s duration=%.3fs",
                        "multimodal" if use_multimodal_prompt else "text",
                        time.perf_counter() - generation_started,
                    )
            try:
                generated_token_count = int(output[0].shape[0] - input_length)
                if generated_token_count >= int(max_new_tokens):
                    logger.warning(
                        "Generation likely hit max_new_tokens limit: generated=%d, limit=%d. "
                        "Consider raising MAX_OUTPUT_TOKENS or tightening prompt verbosity.",
                        generated_token_count,
                        int(max_new_tokens),
                    )
                else:
                    logger.info(
                        "Generation tokens: generated=%d, limit=%d",
                        generated_token_count,
                        int(max_new_tokens),
                    )
            except Exception as e:
                logger.debug("Failed to compute generation token diagnostics: %s", e)

            generated_ids = output[0][input_length:]
            text = self._decode_generation_tokens(
                generated_ids,
                use_multimodal_prompt=use_multimodal_prompt,
            )
            if text:
                return text
            if torch.is_tensor(generated_ids) and generated_ids.numel() == 0:
                logger.warning(
                    "Model returned zero generated tokens for %s prompt; falling back to full-output decode.",
                    "multimodal" if use_multimodal_prompt else "text",
                )

            text = self._decode_generation_tokens(
                output if use_multimodal_prompt else output[0],
                use_multimodal_prompt=use_multimodal_prompt,
            )
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            elif prompt and prompt in text[: len(prompt) + 256]:
                text = text.split(prompt, 1)[1].strip()
            return text
        finally:
            if use_multimodal_prompt:
                input_ids = None
                output = None
                generated_ids = None
                self._release_multimodal_generation_buffers(inputs, output, generated_ids)

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """Simple streaming: yield the full generated text as one chunk."""
        yield self.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def close(self) -> None:
        self._release_multimodal_generation_buffers()


class OpenAIProvider(BaseLLM):
    """OpenAI LLM provider with retry mechanism."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_role: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenAI provider.
        
        Args:
            model_name: Model name (defaults to settings)
            api_key: API key (defaults to settings)
            base_url: API base URL (defaults to settings)
            **kwargs: Additional configuration
        """
        model_name = model_name or settings.openai_model
        super().__init__(model_name, **kwargs)
        role_defaults = _get_role_generation_config(model_role, "openai")
        self.model_role = model_role
        configured_base_urls = split_endpoint_list(base_url or settings.openai_api_base)
        if not configured_base_urls:
            configured_base_urls = split_endpoint_list(settings.openai_api_base) or [str(base_url or settings.openai_api_base)]
        self._endpoint_pool = OpenAIEndpointPool(
            api_key=api_key or settings.openai_api_key,
            base_urls=configured_base_urls,
            namespace=f"openai_provider:{str(model_role or model_name or 'default').strip().lower() or 'default'}",
        )
        self.base_url = self._endpoint_pool.display_base_urls
        self.client = None
        self.temperature = kwargs.get("temperature", role_defaults.get("temperature", 0.7))
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.top_p = kwargs.get("top_p", role_defaults.get("top_p"))
        self.top_k = kwargs.get("top_k", role_defaults.get("top_k"))
        self.min_p = kwargs.get("min_p", role_defaults.get("min_p"))
        self.presence_penalty = kwargs.get("presence_penalty", role_defaults.get("presence_penalty"))
        self.repetition_penalty = kwargs.get("repetition_penalty", role_defaults.get("repetition_penalty"))
        self.enable_thinking = _resolve_enable_thinking_for_role(
            model_role,
            kwargs.get("enable_thinking", role_defaults.get("enable_thinking", False)),
            fallback=bool(role_defaults.get("enable_thinking", False)),
            source="OpenAIProvider.__init__",
        )

    @contextmanager
    def _borrow_chat_client(self) -> Iterator[Tuple[Any, str]]:
        with self._endpoint_pool.borrow() as borrowed:
            yield borrowed
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3)
    )
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response with retry logic.
        
        Args:
            messages: Message list
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        request_base_url = "<unselected>"
        request_summary = "<unbuilt>"
        try:
            with self._borrow_chat_client() as (client, request_base_url):
                request_kwargs = dict(kwargs)
                request_kwargs.pop("wall_clock_deadline_monotonic", None)
                request_kwargs.pop("generation_deadline_monotonic", None)
                request_extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
                enable_thinking_value = _resolve_enable_thinking_for_role(
                    self.model_role,
                    request_kwargs.pop("enable_thinking", None),
                    fallback=bool(self.enable_thinking),
                    source="OpenAIProvider.generate",
                    respect_role_config=bool(request_kwargs.pop("respect_role_generation_config", True)),
                )
                request_extra_body = _inject_openai_thinking_control(
                    request_extra_body,
                    enable_thinking=enable_thinking_value,
                    base_url=request_base_url,
                )
                top_k_value = request_kwargs.pop("top_k", self.top_k)
                min_p_value = request_kwargs.pop("min_p", self.min_p)
                repetition_penalty_value = request_kwargs.pop("repetition_penalty", self.repetition_penalty)
                if top_k_value is not None:
                    request_extra_body["top_k"] = top_k_value
                if min_p_value is not None:
                    request_extra_body["min_p"] = min_p_value
                if repetition_penalty_value is not None:
                    request_extra_body["repetition_penalty"] = repetition_penalty_value
                request_payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature if temperature is None else temperature,
                    "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                    "extra_body": request_extra_body,
                }
                stop_value = request_kwargs.pop("stop", None)
                if stop_value is None:
                    request_payload["stop"] = list(_SYNTHETIC_ROLE_STOP_SEQUENCES)
                else:
                    request_payload["stop"] = stop_value
                top_p_value = request_kwargs.pop("top_p", self.top_p)
                presence_penalty_value = request_kwargs.pop("presence_penalty", self.presence_penalty)
                if top_p_value is not None:
                    request_payload["top_p"] = top_p_value
                if presence_penalty_value is not None:
                    request_payload["presence_penalty"] = presence_penalty_value

                request_summary = (
                    f"model={self.model_name} max_tokens={request_payload.get('max_tokens')} "
                    f"temperature={request_payload.get('temperature')} top_p={request_payload.get('top_p')} "
                    f"presence_penalty={request_payload.get('presence_penalty')} "
                    f"extra_body_keys={sorted(request_extra_body.keys())} "
                    f"{_summarize_openai_messages(messages)}"
                )
                response = client.chat.completions.create(
                    **request_payload,
                    **request_kwargs
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(
                "OpenAI API error on base_url=%s role=%s request=%s: %s",
                request_base_url,
                str(self.model_role or "").strip().lower() or "default",
                request_summary,
                _summarize_openai_exception(e),
            )
            sidecar_tail = _local_sidecar_log_tail(request_base_url, self.model_role)
            if sidecar_tail:
                logger.error("Local non-planner sidecar log tail after OpenAI API failure:\n%s", sidecar_tail)
            raise
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3)
    )
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """Generate streaming response with retry logic.
        
        Args:
            messages: Message list
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional parameters
            
        Yields:
            Text chunks
        """
        request_base_url = "<unselected>"
        try:
            with self._borrow_chat_client() as (client, request_base_url):
                request_kwargs = dict(kwargs)
                request_kwargs.pop("wall_clock_deadline_monotonic", None)
                request_kwargs.pop("generation_deadline_monotonic", None)
                request_extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
                enable_thinking_value = _resolve_enable_thinking_for_role(
                    self.model_role,
                    request_kwargs.pop("enable_thinking", None),
                    fallback=bool(self.enable_thinking),
                    source="OpenAIProvider.stream_generate",
                    respect_role_config=bool(request_kwargs.pop("respect_role_generation_config", True)),
                )
                request_extra_body = _inject_openai_thinking_control(
                    request_extra_body,
                    enable_thinking=enable_thinking_value,
                    base_url=request_base_url,
                )
                top_k_value = request_kwargs.pop("top_k", self.top_k)
                min_p_value = request_kwargs.pop("min_p", self.min_p)
                repetition_penalty_value = request_kwargs.pop("repetition_penalty", self.repetition_penalty)
                if top_k_value is not None:
                    request_extra_body["top_k"] = top_k_value
                if min_p_value is not None:
                    request_extra_body["min_p"] = min_p_value
                if repetition_penalty_value is not None:
                    request_extra_body["repetition_penalty"] = repetition_penalty_value
                request_payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature if temperature is None else temperature,
                    "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                    "extra_body": request_extra_body,
                    "stream": True,
                }
                top_p_value = request_kwargs.pop("top_p", self.top_p)
                presence_penalty_value = request_kwargs.pop("presence_penalty", self.presence_penalty)
                if top_p_value is not None:
                    request_payload["top_p"] = top_p_value
                if presence_penalty_value is not None:
                    request_payload["presence_penalty"] = presence_penalty_value

                stream = client.chat.completions.create(
                    **request_payload,
                    **request_kwargs
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(
                "OpenAI streaming error on base_url=%s role=%s: %s",
                request_base_url,
                str(self.model_role or "").strip().lower() or "default",
                e,
            )
            raise


class AnthropicProvider(BaseLLM):
    """Anthropic Claude provider with retry mechanism."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        model_role: Optional[str] = None,
        **kwargs
    ):
        """Initialize Anthropic provider.
        
        Args:
            model_name: Model name (defaults to settings)
            api_key: API key (defaults to settings)
            **kwargs: Additional configuration
        """
        model_name = model_name or settings.anthropic_model
        super().__init__(model_name, **kwargs)
        role_defaults = _get_role_generation_config(model_role, "anthropic")
        self.model_role = model_role
        _require_anthropic_sdk()
        self.client = Anthropic(api_key=api_key or settings.anthropic_api_key)
        self.temperature = kwargs.get("temperature", role_defaults.get("temperature", 0.7))
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.top_p = kwargs.get("top_p", role_defaults.get("top_p"))
        self.top_k = kwargs.get("top_k", role_defaults.get("top_k"))
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3)
    )
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response with retry logic.
        
        Args:
            messages: Message list
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        # Extract system message if present
        system_msg = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        try:
            request_kwargs = dict(kwargs)
            request_kwargs.pop("wall_clock_deadline_monotonic", None)
            request_kwargs.pop("generation_deadline_monotonic", None)
            request_kwargs.pop("enable_thinking", None)
            request_kwargs.pop("min_p", None)
            request_kwargs.pop("repetition_penalty", None)
            request_payload = {
                "model": self.model_name,
                "system": system_msg if system_msg else None,
                "messages": user_messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            }
            top_p_value = request_kwargs.pop("top_p", self.top_p)
            top_k_value = request_kwargs.pop("top_k", self.top_k)
            if top_p_value is not None:
                request_payload["top_p"] = top_p_value
            if top_k_value is not None:
                request_payload["top_k"] = top_k_value
            response = self.client.messages.create(
                **request_payload,
                **request_kwargs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3)
    )
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """Generate streaming response with retry logic.
        
        Args:
            messages: Message list
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional parameters
            
        Yields:
            Text chunks
        """
        # Extract system message if present
        system_msg = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        try:
            request_kwargs = dict(kwargs)
            request_kwargs.pop("wall_clock_deadline_monotonic", None)
            request_kwargs.pop("generation_deadline_monotonic", None)
            request_kwargs.pop("enable_thinking", None)
            request_kwargs.pop("min_p", None)
            request_kwargs.pop("repetition_penalty", None)
            request_payload = {
                "model": self.model_name,
                "system": system_msg if system_msg else None,
                "messages": user_messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            }
            top_p_value = request_kwargs.pop("top_p", self.top_p)
            top_k_value = request_kwargs.pop("top_k", self.top_k)
            if top_p_value is not None:
                request_payload["top_p"] = top_p_value
            if top_k_value is not None:
                request_payload["top_k"] = top_k_value
            with self.client.messages.stream(
                **request_payload,
                **request_kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise


def _infer_provider_name_from_instance(provider: Any) -> Optional[str]:
    if isinstance(provider, TransformersProvider):
        return "transformers"
    if isinstance(provider, OpenAIProvider):
        return "openai"
    if isinstance(provider, AnthropicProvider):
        return "anthropic"
    return None


def _apply_role_generation_defaults_to_provider_view(
    provider_view: Any,
    *,
    model_role: Optional[str],
    provider_name: Optional[str],
) -> None:
    role_defaults = _get_role_generation_config(model_role, provider_name)
    if not role_defaults:
        return

    attribute_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "presence_penalty": "presence_penalty",
        "repetition_penalty": "repetition_penalty",
    }
    for attr_name, config_key in attribute_map.items():
        if hasattr(provider_view, attr_name) and config_key in role_defaults:
            setattr(provider_view, attr_name, role_defaults.get(config_key))
    if hasattr(provider_view, "max_tokens") and "max_new_tokens" in role_defaults:
        provider_view.max_tokens = role_defaults.get("max_new_tokens")


def build_role_specific_provider_view(
    provider: Any,
    model_role: Optional[str],
    *,
    enable_thinking: Optional[bool] = None,
) -> Any:
    normalized_role = str(model_role or "").strip().lower()
    if not normalized_role:
        return provider

    provider_name = _infer_provider_name_from_instance(provider)
    if provider_name is None:
        return provider

    fallback_enable_thinking = bool(getattr(provider, "enable_thinking", False))
    resolved_enable_thinking = _resolve_enable_thinking_for_role(
        normalized_role,
        enable_thinking,
        fallback=fallback_enable_thinking,
        source="build_role_specific_provider_view",
        provider_name=provider_name,
    )

    current_role = str(getattr(provider, "model_role", "") or "").strip().lower()
    if current_role == normalized_role and (
        not hasattr(provider, "enable_thinking")
        or bool(getattr(provider, "enable_thinking", False)) == resolved_enable_thinking
    ):
        _apply_role_generation_defaults_to_provider_view(
            provider,
            model_role=normalized_role,
            provider_name=provider_name,
        )
        return provider

    try:
        provider_view = copy.copy(provider)
    except Exception as exc:
        logger.warning(
            "Failed to build role-specific provider view for role=%s provider=%s; returning the original instance unchanged. error=%s",
            normalized_role,
            type(provider).__name__,
            exc,
        )
        return provider

    if hasattr(provider_view, "model_role"):
        provider_view.model_role = normalized_role
    if hasattr(provider_view, "enable_thinking"):
        provider_view.enable_thinking = resolved_enable_thinking
    provider_view._shared_provider_source = provider
    provider_view._shared_provider_source_role = current_role
    _apply_role_generation_defaults_to_provider_view(
        provider_view,
        model_role=normalized_role,
        provider_name=provider_name,
    )
    return provider_view


def cache_existing_transformers_provider(provider: Any) -> Any:
    if not isinstance(provider, TransformersProvider):
        return provider

    model_name = getattr(provider, "model_name", None)
    model_role = str(getattr(provider, "model_role", "") or "").strip().lower()
    if not model_name or not model_role:
        return provider

    from runtime.session import runtime_session

    share_planner_subagent = _should_share_planner_subagent_provider(
        model_name,
        model_role,
    )
    if share_planner_subagent:
        provider._shared_model_roles = {"planner", "subagent"}

    cache_key = build_transformers_provider_cache_key(
        model_name=model_name,
        model_role=model_role,
        device=getattr(provider, "runtime_device", None) or getattr(provider, "device", None),
        device_map=getattr(provider, "device_map", None),
        temperature=getattr(provider, "temperature", None),
        max_tokens=getattr(provider, "max_tokens", None),
        top_p=getattr(provider, "top_p", None),
        top_k=getattr(provider, "top_k", None),
        min_p=getattr(provider, "min_p", None),
        repetition_penalty=getattr(provider, "repetition_penalty", None),
        enable_thinking=getattr(provider, "enable_thinking", None),
    )
    runtime_session.cache_provider(cache_key, provider)
    return provider


def get_llm_provider(provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> BaseLLM:
    """Factory function to get LLM provider.
    
    Args:
        provider: Provider name ("openai" or "anthropic")
        **kwargs: Additional configuration
        
    Returns:
        LLM provider instance
    """
    resolved_provider = provider or settings.llm_provider
    requested_role = str(kwargs.get("model_role", "") or "").strip().lower()
    raw_requested_provider = str(resolved_provider or "").strip().lower()
    requested_provider = _normalize_aux_provider_name(resolved_provider)
    if requested_provider != raw_requested_provider:
        resolved_provider = requested_provider
    if provider is None and model_name is not None:
        non_planner_provider = _non_planner_service_provider(requested_role)
        sidecar_kwargs = _resolve_non_planner_service_kwargs(
            model_role=requested_role,
            current_kwargs=kwargs,
            explicit=False,
        )
        if sidecar_kwargs is not None:
            resolved_provider = "openai"
            kwargs = sidecar_kwargs
        elif non_planner_provider == "process":
            resolved_provider = "process"
            kwargs = {**kwargs, "model_name": model_name}
        else:
            # Explicit local model paths remain the planner/FSDP path unless a
            # non-planner sidecar is configured for this role.
            resolved_provider = "transformers"
            kwargs = {**kwargs, "model_name": model_name}
    elif raw_requested_provider in {"sidecar", "service", "remote", "api"}:
        sidecar_kwargs = _resolve_non_planner_service_kwargs(
            model_role=requested_role,
            current_kwargs=kwargs,
            explicit=True,
        )
        if sidecar_kwargs is not None:
            resolved_provider = "openai"
            kwargs = sidecar_kwargs
    elif requested_provider == "process":
        resolved_provider = "process"
    elif model_name:
        kwargs = {**kwargs, "model_name": model_name}

    configured_enable_thinking = _get_configured_enable_thinking(requested_role, resolved_provider)
    if configured_enable_thinking is not None:
        kwargs["enable_thinking"] = _resolve_enable_thinking_for_role(
            requested_role,
            kwargs.get("enable_thinking"),
            fallback=bool(configured_enable_thinking),
            source="get_llm_provider",
            provider_name=resolved_provider,
        )

    if resolved_provider == "openai":
        return OpenAIProvider(**kwargs)
    elif resolved_provider == "anthropic":
        return AnthropicProvider(**kwargs)
    elif resolved_provider == "transformers":
        from runtime.session import runtime_session

        requested_role = str(kwargs.get("model_role", "") or "").strip().lower()
        share_planner_subagent = _should_share_planner_subagent_provider(
            kwargs.get("model_name"),
            requested_role,
        )
        cache_key = build_transformers_provider_cache_key(
            model_name=kwargs.get("model_name"),
            model_role=requested_role,
            device=kwargs.get("device"),
            device_map=kwargs.get("device_map"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            min_p=kwargs.get("min_p"),
            repetition_penalty=kwargs.get("repetition_penalty"),
            enable_thinking=kwargs.get("enable_thinking"),
        )
        cache_role = requested_role or "default"
        cached = runtime_session.get_cached_provider(cache_key)
        if cached is not None:
            runtime_session.record_cache(f"provider.{cache_role}", True)
            if share_planner_subagent:
                return build_role_specific_provider_view(
                    cached,
                    requested_role,
                    enable_thinking=kwargs.get("enable_thinking"),
                )
            return cached

        runtime_session.record_cache(f"provider.{cache_role}", False)
        with runtime_session.time_block(f"runtime_load.provider_{cache_role}_seconds"):
            provider_kwargs = dict(kwargs)
            if share_planner_subagent:
                provider_kwargs["model_role"] = "planner"
            provider_instance = TransformersProvider(**provider_kwargs)
        if share_planner_subagent:
            provider_instance._shared_model_roles = {"planner", "subagent"}
        logger.info(
            "Provider cache miss -> loaded role=%s model=%s device=%s device_map=%s runtime_device=%s",
            cache_role,
            getattr(provider_instance, "model_name", kwargs.get("model_name")),
            kwargs.get("device"),
            getattr(provider_instance, "device_map", kwargs.get("device_map")),
            getattr(provider_instance, "runtime_device", kwargs.get("device")),
        )
        cached_provider = runtime_session.cache_provider(cache_key, provider_instance)
        if share_planner_subagent:
            return build_role_specific_provider_view(
                cached_provider,
                requested_role,
                enable_thinking=kwargs.get("enable_thinking"),
            )
        return cached_provider
    elif resolved_provider == "process":
        from runtime.session import runtime_session

        requested_role = str(kwargs.get("model_role", "") or "").strip().lower()
        share_planner_subagent = _should_share_planner_subagent_provider(
            kwargs.get("model_name"),
            requested_role,
        )
        if not str(kwargs.get("device") or "").strip():
            kwargs = {
                **kwargs,
                "device": _resolve_provider_runtime_device(
                    None,
                    purpose=_provider_device_purpose(requested_role),
                ),
            }
        cache_key = build_process_provider_cache_key(
            model_name=kwargs.get("model_name"),
            model_role=requested_role,
            device=kwargs.get("device"),
            device_map=kwargs.get("device_map"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            min_p=kwargs.get("min_p"),
            repetition_penalty=kwargs.get("repetition_penalty"),
            enable_thinking=kwargs.get("enable_thinking"),
        )
        cache_role = requested_role or "default"
        cached = runtime_session.get_cached_provider(cache_key)
        if cached is not None:
            runtime_session.record_cache(f"provider.{cache_role}", True)
            if share_planner_subagent and requested_role == "subagent":
                return build_role_specific_provider_view(
                    cached,
                    requested_role,
                    enable_thinking=kwargs.get("enable_thinking"),
                )
            return cached

        runtime_session.record_cache(f"provider.{cache_role}", False)
        with runtime_session.time_block(f"runtime_load.provider_process_{cache_role}_seconds"):
            provider_instance = ProcessBackedLLMProvider(**kwargs)
        logger.info(
            "Process-backed provider cache miss -> loaded role=%s model=%s device=%s",
            cache_role,
            getattr(provider_instance, "model_name", kwargs.get("model_name")),
            kwargs.get("device"),
        )
        cached_provider = runtime_session.cache_provider(cache_key, provider_instance)
        if share_planner_subagent and requested_role == "subagent":
            return build_role_specific_provider_view(
                cached_provider,
                requested_role,
                enable_thinking=kwargs.get("enable_thinking"),
            )
        return cached_provider
    elif resolved_provider == "local":
        # Use a lightweight local provider; caller can override model_name/model_path via kwargs
        return LocalProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {resolved_provider}")
