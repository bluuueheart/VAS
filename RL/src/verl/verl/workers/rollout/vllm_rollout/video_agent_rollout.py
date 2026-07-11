from __future__ import annotations

import ast
from contextlib import contextmanager, nullcontext
from collections import OrderedDict, deque
from copy import deepcopy
import heapq
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensordict import TensorDict
from transformers import StoppingCriteria, StoppingCriteriaList
try:
    from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
except Exception:  # pragma: no cover - older transformers builds
    InfNanRemoveLogitsProcessor = None  # type: ignore[assignment]
    LogitsProcessorList = None  # type: ignore[assignment]

from core.accelerator_queue import SharedAcceleratorQueueLock, resolve_local_accelerator_slot_key
from core.llm import AcceleratorFatalError, GenerationTimeoutError, is_accelerator_fatal_error
from verl import DataProto
from verl.protocol import to_1d_object_array
from verl.utils.device import set_expandable_segments
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout

from .video_agent_episode import (
    EpisodeArtifacts,
    PROJECT_ROOT,
    VideoAgentEpisodeRunner,
    _apply_chat_template_with_optional_thinking,
    _decode_text,
    _ensure_user_message,
    _flatten_content,
    _messages_contain_visual_inputs,
    _normalize_multimodal_messages,
    _render_text_chat,
)

logger = logging.getLogger(__name__)
_RESPONSE_SCHEMA_KEYS = frozenset(
    {
        "Actions",
        "actions",
        "Action",
        "action",
        "Thought",
        "thought",
        "Plan",
        "plan",
        "Finish",
        "finish",
        "Result",
        "result",
    }
)
_ACTIONABLE_RESPONSE_SCHEMA_KEYS = frozenset(
    {
        "Actions",
        "actions",
        "Action",
        "action",
        "Finish",
        "finish",
    }
)
_STATIC_MULTIMODAL_TENSOR_KEYS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_sizes",
        "video_sizes",
        "second_per_grid_ts",
    }
)
_FLOAT_MULTIMODAL_TENSOR_KEYS = frozenset({"pixel_values", "pixel_values_videos"})
_QUIETLY_DROPPED_PROMPT_ALIGNED_AUXILIARY_TENSOR_KEYS = frozenset(
    {
        # Qwen-style processors may emit prompt-only helper tensors that are
        # useful for the initial prefill but become stale as soon as rollout
        # appends new tokens.
        "mm_token_type_ids",
    }
)


def _normalize_generation_role(role: Any) -> str:
    return str(role or "").strip().lower()


def _coerce_monotonic_deadline(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_out_of_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc or "").lower()
    markers = (
        "out of memory",
        "cuda out of memory",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "hip out of memory",
        "npu out of memory",
    )
    return any(marker in message for marker in markers)


def _coerce_token_ids(value: Any) -> List[int]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        token_ids: List[int] = []
        for item in value:
            token_ids.extend(_coerce_token_ids(item))
        return token_ids
    try:
        return [int(value)]
    except Exception:
        return []


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _coerce_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return float(default)
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _clamp_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    parsed = _coerce_float(value, default)
    return min(float(maximum), max(float(minimum), parsed))


def _clamp_optional_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return min(int(maximum), max(int(minimum), parsed))


def _read_bool_config(config: Any, key: str, default: bool = False) -> bool:
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return bool(getter(key, default))
        except Exception:
            pass
    try:
        return bool(getattr(config, key))
    except Exception:
        return bool(default)


def _force_config_bool_false(config: Any, key: str, *, reason: str) -> None:
    if _read_bool_config(config, key, False):
        logger.warning("Ignoring %s=True because %s", key, reason)
    try:
        config[key] = False
        return
    except Exception:
        pass
    try:
        setattr(config, key, False)
    except Exception:
        pass


def _sanitize_cache_key_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_cache_key_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _sanitize_cache_key_value(subvalue)
            for key, subvalue in value.items()
        }
    return str(value)


def _stable_sha1(payload: Any) -> str:
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        serialized = str(payload)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _longest_common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    common = 0
    for left_token, right_token in zip(left or (), right or ()):
        if int(left_token) != int(right_token):
            break
        common += 1
    return common


def _parse_dict_like(text: str) -> Optional[Dict[str, Any]]:
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _coerce_row_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_row_string_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        normalized: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _is_actionable_response_payload(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    if _ACTIONABLE_RESPONSE_SCHEMA_KEYS.intersection(payload.keys()):
        return True
    tool_name = payload.get("tool") or payload.get("tool_name")
    if isinstance(tool_name, str) and tool_name.strip():
        return True
    name_value = payload.get("name")
    if isinstance(name_value, str) and name_value.strip() and any(
        key in payload for key in ("params", "arguments", "args")
    ):
        return True
    return False


def _extract_structured_json_payload_block(
    text: str,
    *,
    actionable_only: bool = False,
) -> Optional[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for idx, ch in enumerate(str(text or "")):
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
            continue
        if ch != "}":
            continue
        depth -= 1
        if depth == 0 and start >= 0:
            candidate = text[start : idx + 1]
            parsed = _parse_dict_like(candidate)
            if isinstance(parsed, dict) and _RESPONSE_SCHEMA_KEYS.intersection(parsed.keys()):
                if (not actionable_only) or _is_actionable_response_payload(parsed):
                    candidates.append((candidate, start, idx + 1))
        if depth < 0:
            depth = 0

    if not candidates:
        return None
    return candidates[-1]


def _truncate_response_to_structured_payload(text: str) -> Tuple[str, bool]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return normalized_text, False
    block = _extract_structured_json_payload_block(
        normalized_text,
        actionable_only=True,
    )
    if block is None:
        return normalized_text, False
    _, _, end_idx = block
    truncated_text = normalized_text[:end_idx].strip()
    if not truncated_text:
        return normalized_text, False
    return truncated_text, truncated_text != normalized_text


def _response_has_complete_structured_payload(text: str) -> bool:
    block = _extract_structured_json_payload_block(
        text,
        actionable_only=True,
    )
    if block is None:
        return False
    _, _, end_idx = block
    return not str(text[end_idx:] or "").strip()


def _looks_like_structured_payload_fragment(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized or "{" not in normalized:
        return False
    markers = (
        '"actions"',
        '"action"',
        '"finish"',
        '"tool"',
        '"plan"',
        '"thought"',
        "actions",
        "action",
        "finish",
        "tool",
    )
    return any(marker in normalized for marker in markers)


class _StructuredPayloadStoppingCriteria(StoppingCriteria):
    """Stop full-sequence generation once the JSON action payload is complete."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_length: int,
        check_interval_tokens: int,
        min_generated_tokens: int = 8,
        arm_probe_interval_tokens: int = 8,
        tail_probe_tokens: int = 128,
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._prompt_length = max(0, int(prompt_length))
        self._check_interval_tokens = max(1, int(check_interval_tokens))
        self._min_generated_tokens = max(1, int(min_generated_tokens))
        self._arm_probe_interval_tokens = max(1, int(arm_probe_interval_tokens))
        self._tail_probe_tokens = max(32, int(tail_probe_tokens))
        self._last_checked_generated_tokens = -1
        self._structured_probe_armed = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> bool:
        if not torch.is_tensor(input_ids) or input_ids.ndim < 2 or input_ids.shape[0] <= 0:
            return False
        generated_tokens = max(0, int(input_ids.shape[-1]) - self._prompt_length)
        if generated_tokens <= 0:
            return False
        if generated_tokens == self._last_checked_generated_tokens:
            return False
        if generated_tokens % self._check_interval_tokens != 0:
            return False
        self._last_checked_generated_tokens = generated_tokens

        if not self._structured_probe_armed:
            if generated_tokens < self._min_generated_tokens:
                return False
            if generated_tokens % self._arm_probe_interval_tokens != 0:
                return False
            tail_start = max(
                self._prompt_length,
                int(input_ids.shape[-1]) - self._tail_probe_tokens,
            )
            try:
                tail_text = _decode_text(
                    self._tokenizer,
                    input_ids[0, tail_start:].tolist(),
                )
            except Exception:
                return False
            if not _looks_like_structured_payload_fragment(tail_text):
                return False
            self._structured_probe_armed = True

        try:
            decoded_text = _decode_text(
                self._tokenizer,
                input_ids[0, self._prompt_length :].tolist(),
            )
        except Exception:
            return False
        return _response_has_complete_structured_payload(decoded_text)


def _apply_top_k_top_p_filter(
    logits: torch.Tensor,
    *,
    top_k: int = -1,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    filtered = logits
    if top_k is not None and int(top_k) > 0 and int(top_k) < filtered.size(-1):
        topk_values, _ = torch.topk(filtered, int(top_k), dim=-1)
        threshold = topk_values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None and float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > float(top_p)
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        filtered = filtered.scatter(
            -1,
            sorted_indices,
            sorted_logits.masked_fill(sorted_mask, float("-inf")),
        )

    if min_p is not None and float(min_p) > 0.0:
        probs = torch.softmax(filtered, dim=-1)
        filtered = filtered.masked_fill(probs < float(min_p), float("-inf"))

    return filtered


def _sanitize_logits_for_sampling(logits: torch.Tensor) -> torch.Tensor:
    """Keep multinomial away from CUDA device asserts when upstream logits are bad."""
    if not torch.is_tensor(logits) or logits.numel() == 0:
        return logits
    if not logits.is_floating_point():
        return logits

    working = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits.clone()
    finite_mask = torch.isfinite(working)
    if bool(finite_mask.all().item()):
        return working

    flat_logits = working.reshape(-1, working.shape[-1])
    flat_finite_mask = finite_mask.reshape(-1, finite_mask.shape[-1])
    for row_idx in range(int(flat_logits.shape[0])):
        row = flat_logits[row_idx]
        row_finite_mask = flat_finite_mask[row_idx]
        if bool(row_finite_mask.any().item()):
            finite_values = row[row_finite_mask]
            low_replacement = finite_values.min()
            high_replacement = finite_values.max()
            row.copy_(
                torch.where(
                    torch.isnan(row) | torch.isneginf(row),
                    low_replacement,
                    torch.where(torch.isposinf(row), high_replacement, row),
                )
            )
        else:
            row.zero_()
    return working


def _safe_multinomial_from_logits(logits: torch.Tensor) -> torch.Tensor:
    working = _sanitize_logits_for_sampling(logits)
    probs = torch.softmax(working, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    prob_sums = probs.sum(dim=-1, keepdim=True)
    bad_rows = (~torch.isfinite(prob_sums)) | (prob_sums <= 0)
    if bool(bad_rows.any().item()):
        fallback_token = torch.argmax(working, dim=-1, keepdim=True)
        safe_probs = torch.where(bad_rows, torch.zeros_like(probs), probs / prob_sums.clamp_min(1e-12))
        fallback_probs = torch.zeros_like(probs)
        fallback_probs.scatter_(dim=-1, index=fallback_token, value=1.0)
        safe_probs = torch.where(bad_rows, fallback_probs, safe_probs)
        sampled = torch.multinomial(safe_probs, num_samples=1)
        return torch.where(bad_rows, fallback_token, sampled)
    probs = probs / prob_sums.clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    *,
    input_ids: Optional[torch.Tensor],
    repetition_penalty: Optional[float],
) -> torch.Tensor:
    if repetition_penalty in (None, ""):
        return logits
    try:
        penalty = float(repetition_penalty)
    except Exception:
        return logits
    if penalty <= 0.0 or abs(penalty - 1.0) < 1e-6:
        return logits
    if not torch.is_tensor(input_ids) or input_ids.numel() == 0:
        return logits

    gathered_scores = torch.gather(logits, 1, input_ids)
    adjusted_scores = torch.where(
        gathered_scores < 0,
        gathered_scores * penalty,
        gathered_scores / penalty,
    )
    return logits.scatter(1, input_ids, adjusted_scores)


def _tensor_to_token_list(value: torch.Tensor) -> List[int]:
    if value.ndim == 0:
        return [int(value.item())]
    return [int(token_id) for token_id in value.tolist()]


def _truncate_text(text: Any, max_chars: int = 240) -> str:
    value = " ".join(str(text or "").split()).strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)] + "..."


_SCHEDULER_PROGRESS_INT_FIELDS = ("prompt_tokens", "image_count", "video_count")


def _estimate_session_cost_hint_from_sample(sample: Dict[str, Any]) -> float:
    payload = dict(sample or {})
    query_text = str(payload.get("prompt") or payload.get("question") or payload.get("query") or "").strip()
    sample_id = str(payload.get("sample_id") or "").strip().lower()
    extra_info = _coerce_row_object(payload.get("extra_info"))
    task_type = str(payload.get("task_type") or extra_info.get("task_type") or "").strip().lower()
    tool_selection = _coerce_row_string_list(payload.get("tool_selection") or extra_info.get("tool_selection"))
    estimated_cost = max(64.0, float(len(query_text)) * 2.0)
    if "multi" in sample_id or "multi" in task_type:
        estimated_cost += 512.0
    if str(payload.get("video_path") or payload.get("video") or "").strip():
        estimated_cost += 128.0
    estimated_cost += float(len(tool_selection)) * 64.0
    return estimated_cost


def _extract_session_progress_cost_hint(
    event: Dict[str, Any],
    *,
    prompt_token_weight: float,
    image_weight: float,
    video_weight: float,
) -> Optional[Dict[str, float]]:
    if not isinstance(event, dict):
        return None
    preview = str(event.get("result_preview") or "").strip()
    if not preview:
        return None
    extracted_values: Dict[str, float] = {}
    for field_name in _SCHEDULER_PROGRESS_INT_FIELDS:
        match = re.search(rf"\b{re.escape(field_name)}=(\d+)\b", preview)
        if match is None:
            continue
        try:
            extracted_values[field_name] = float(int(match.group(1)))
        except Exception:
            continue
    if not extracted_values:
        return None

    prompt_tokens = float(extracted_values.get("prompt_tokens", 0.0))
    image_count = float(extracted_values.get("image_count", 0.0))
    video_count = float(extracted_values.get("video_count", 0.0))
    estimated_cost = (
        prompt_tokens * max(0.0, float(prompt_token_weight))
        + image_count * max(0.0, float(image_weight))
        + video_count * max(0.0, float(video_weight))
    )
    phase = str(event.get("phase") or "").strip().lower()
    if "structured_repair" in phase:
        estimated_cost *= 0.35
    elif phase.startswith("generation_"):
        estimated_cost *= 1.0
    else:
        estimated_cost *= 0.8
    return {
        "estimated_cost": max(64.0, float(estimated_cost)),
        "prompt_tokens": prompt_tokens,
        "image_count": image_count,
        "video_count": video_count,
    }


def _state_scheduler_priority(
    state: Dict[str, Any],
    *,
    prioritize_new_sessions: bool,
    cost_aware_dispatch: bool,
) -> Tuple[Any, ...]:
    new_session_bias = 0 if prioritize_new_sessions and int(state.get("scheduler_dispatch_count", 0)) <= 0 else 1
    fairness_bias = int(state.get("turn_yields", 0))
    aging_bias = -min(int(state.get("scheduler_wait_count", 0)), 32)
    output_order = int(state.get("output_order", -1))
    if cost_aware_dispatch:
        cost_hint = _coerce_float(state.get("scheduler_cost_hint"), 1024.0)
        return (new_session_bias, fairness_bias, aging_bias, cost_hint, output_order)
    return (new_session_bias, fairness_bias, aging_bias, output_order)


def _summarize_tensor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        if torch.is_tensor(value):
            summary[str(key)] = {
                "shape": tuple(int(dim) for dim in value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        else:
            summary[str(key)] = type(value).__name__
    return summary


def _split_video_inputs_and_metadata(video_inputs: Any) -> Tuple[Any, Optional[List[Dict[str, Any]]]]:
    if video_inputs is None:
        return None, None
    if not isinstance(video_inputs, (list, tuple)):
        return video_inputs, None

    videos: List[Any] = []
    metadata_items: List[Optional[Dict[str, Any]]] = []
    saw_metadata = False
    for item in list(video_inputs):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
            videos.append(item[0])
            metadata_items.append(dict(item[1]))
            saw_metadata = True
        else:
            videos.append(item)
            metadata_items.append(None)

    if not saw_metadata:
        return videos, None

    normalized_metadata = [item for item in metadata_items if item]
    return videos, (normalized_metadata or None)


def _extract_past_key_values(output: Any) -> Any:
    def _looks_like_past_key_values(candidate: Any) -> bool:
        if candidate is None or torch.is_tensor(candidate):
            return False
        if isinstance(candidate, (tuple, list)):
            if not candidate:
                return False
            first_item = candidate[0]
            if torch.is_tensor(first_item):
                return False
            return isinstance(first_item, (tuple, list, dict)) or hasattr(first_item, "shape")
        if isinstance(candidate, dict):
            return True
        return any(
            hasattr(candidate, attr)
            for attr in ("to_legacy_cache", "get_seq_length", "seen_tokens")
        )

    for attr_name in ("past_key_values", "past_key_value", "cache", "cache_params"):
        candidate = getattr(output, attr_name, None)
        if _looks_like_past_key_values(candidate):
            return candidate

    if isinstance(output, dict):
        for key in ("past_key_values", "past_key_value", "cache", "cache_params"):
            candidate = output.get(key)
            if _looks_like_past_key_values(candidate):
                return candidate

    if isinstance(output, (tuple, list)):
        for idx in range(1, min(len(output), 4)):
            candidate = output[idx]
            if _looks_like_past_key_values(candidate):
                return candidate

    return None


def _iter_rollout_inference_modules(root_module: Any) -> List[Any]:
    modules: List[Any] = []
    seen: set[int] = set()
    queue: List[Any] = [root_module]
    child_attrs = (
        "module",
        "model",
        "base_model",
        "language_model",
        "llm",
        "_fsdp_wrapped_module",
    )

    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        modules.append(current)
        for attr_name in child_attrs:
            try:
                child = getattr(current, attr_name, None)
            except Exception:
                child = None
            if child is not None and id(child) not in seen:
                queue.append(child)
    return modules


def _module_gradient_checkpointing_enabled(module: Any) -> bool:
    marker = getattr(module, "is_gradient_checkpointing", None)
    if isinstance(marker, bool):
        return marker
    local_flag = getattr(module, "gradient_checkpointing", None)
    if isinstance(local_flag, bool):
        return local_flag
    try:
        for child in module.modules():
            child_flag = getattr(child, "gradient_checkpointing", None)
            if isinstance(child_flag, bool) and child_flag:
                return True
    except Exception:
        return False
    return False


@contextmanager
def _temporary_rollout_inference_mode(
    module: Any,
    *,
    disable_gradient_checkpointing: bool = True,
    force_use_cache: bool = True,
) -> Any:
    state = {
        "candidate_module_count": 0,
        "cache_flags_forced": 0,
        "gradient_checkpointing_modules_disabled": 0,
        "skipped_reason": "",
    }
    # Rollout generation enters eval mode immediately before this context manager.
    # For HF/Qwen-style models, gradient checkpointing is only active during
    # training forwards, and rollout passes explicit use_cache flags to both
    # incremental forward() and model.generate(). Avoid per-turn config/hook
    # mutation once the module is already in eval mode; that extra work can
    # dominate short planner turns and was the source of regressions.
    if not bool(getattr(module, "training", False)):
        state["skipped_reason"] = "module_already_eval"
        yield state
        return
    if (not disable_gradient_checkpointing) and (not force_use_cache):
        state["skipped_reason"] = "no_runtime_mutation_requested"
        yield state
        return

    candidate_modules = _iter_rollout_inference_modules(module)
    state["candidate_module_count"] = len(candidate_modules)
    previous_cache_values: List[Tuple[Any, Any]] = []
    seen_cache_holders: set[int] = set()
    disabled_gradient_checkpoint_modules: List[Any] = []

    if force_use_cache:
        for candidate in candidate_modules:
            for holder_name in ("config", "generation_config"):
                holder = getattr(candidate, holder_name, None)
                if holder is None or not hasattr(holder, "use_cache"):
                    continue
                holder_id = id(holder)
                if holder_id in seen_cache_holders:
                    continue
                seen_cache_holders.add(holder_id)
                try:
                    previous_value = getattr(holder, "use_cache")
                    previous_cache_values.append((holder, previous_value))
                    if previous_value is not True:
                        setattr(holder, "use_cache", True)
                        state["cache_flags_forced"] += 1
                except Exception:
                    continue

    if disable_gradient_checkpointing:
        for candidate in candidate_modules:
            disable_fn = getattr(candidate, "gradient_checkpointing_disable", None)
            enable_fn = getattr(candidate, "gradient_checkpointing_enable", None)
            if not callable(disable_fn) or not callable(enable_fn):
                continue
            if not _module_gradient_checkpointing_enabled(candidate):
                continue
            try:
                disable_fn()
            except Exception:
                continue
            disabled_gradient_checkpoint_modules.append(candidate)

    state["gradient_checkpointing_modules_disabled"] = len(disabled_gradient_checkpoint_modules)

    try:
        yield state
    finally:
        for holder, previous_value in reversed(previous_cache_values):
            try:
                setattr(holder, "use_cache", previous_value)
            except Exception:
                continue
        for candidate in reversed(disabled_gradient_checkpoint_modules):
            enable_fn = getattr(candidate, "gradient_checkpointing_enable", None)
            if not callable(enable_fn):
                continue
            try:
                enable_fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                try:
                    enable_fn()
                except Exception:
                    continue
            except Exception:
                continue


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
        "Normalizing rollout multimodal processor tensors from %s: %s -> common_length=%d",
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


def _is_prompt_aligned_auxiliary_tensor(
    key: str,
    value: Any,
    *,
    prompt_length: int,
) -> bool:
    if not torch.is_tensor(value):
        return False
    if str(key or "") in _STATIC_MULTIMODAL_TENSOR_KEYS:
        return False
    if value.ndim <= 0:
        return False
    # Multimodal processors may return prompt-shaped helper tensors whose
    # sequence axis is not always the trailing dimension, e.g.
    # `cross_attention_mask` or `aspect_ratio_mask`. Once rollout appends a new
    # token, reusing those stale prompt-aligned tensors causes the classic
    # `mask [N]` vs `indexed tensor [N-1]` failure on the next forward pass.
    return any(int(dim) == int(prompt_length) for dim in value.shape)


def _build_generation_static_kwargs(
    model_inputs: Dict[str, Any],
    *,
    prompt_length: int,
) -> Dict[str, Any]:
    static_kwargs: Dict[str, Any] = {}
    dropped_keys: List[str] = []
    for key, value in dict(model_inputs or {}).items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if _is_prompt_aligned_auxiliary_tensor(key, value, prompt_length=prompt_length):
            dropped_keys.append(str(key))
            continue
        static_kwargs[key] = value

    if dropped_keys:
        dropped_keys = sorted(set(dropped_keys))
        unexpected_dropped_keys = [
            key for key in dropped_keys if key not in _QUIETLY_DROPPED_PROMPT_ALIGNED_AUXILIARY_TENSOR_KEYS
        ]
        quietly_dropped_keys = [
            key for key in dropped_keys if key in _QUIETLY_DROPPED_PROMPT_ALIGNED_AUXILIARY_TENSOR_KEYS
        ]
        if quietly_dropped_keys:
            logger.debug(
                "Skipping known prompt-aligned optional generation tensors during rollout autoregressive generation: %s",
                quietly_dropped_keys,
            )
        if unexpected_dropped_keys:
            logger.warning(
                "Dropping prompt-aligned auxiliary tensors during rollout autoregressive generation: %s",
                unexpected_dropped_keys,
            )
    return static_kwargs


class vLLMVideoAgentRollout(BaseRollout):
    def __init__(self, actor_module, config, tokenizer, model_hf_config, processor=None, **kwargs):
        # Keep video-agent rollout config as DictConfig to preserve custom keys
        # used by planner/subagent runtime logic.
        self.device_mesh = kwargs.get("device_mesh", None)
        self._local_rollout_ulysses_device_mesh = kwargs.get("local_rollout_ulysses_device_mesh", None)
        self.model_config = model_hf_config
        self._actor_module_source = "direct_module"
        self.actor_module_fsdp = None
        if isinstance(actor_module, FSDP):
            # Keep the FSDP owner for full-param materialization, while using
            # the unwrapped module for the patched Qwen generate/forward path.
            self.actor_module_fsdp = actor_module
            self.actor_module = actor_module._fsdp_wrapped_module
            self._actor_module_source = "fsdp_wrapped_module"
        else:
            self.actor_module = actor_module
        self._fatal_accelerator_error: Optional[str] = None
        self.config = config
        self._accelerator_runtime_device = kwargs.get("runtime_device")
        queue_root_value = self.config.get(
            "shared_accelerator_queue_root",
            str((PROJECT_ROOT / "rl_workdir" / "cache" / "_accelerator_queue").resolve()),
        )
        queue_enabled = bool(self.config.get("enable_shared_accelerator_queue", True))
        queue_log_wait_seconds = _coerce_float(
            self.config.get("shared_accelerator_queue_log_wait_seconds", 5.0),
            5.0,
        )
        accelerator_slot_key = resolve_local_accelerator_slot_key(self._accelerator_runtime_device)
        self._actor_fsdp_generate_lock = SharedAcceleratorQueueLock(
            name="video_agent_rollout",
            queue_root=queue_root_value,
            slot_key=accelerator_slot_key,
            enabled=queue_enabled,
            log_wait_seconds=queue_log_wait_seconds,
        )
        current_device_index: Optional[int] = None
        try:
            if torch.cuda.is_available():
                current_device_index = int(torch.cuda.current_device())
        except Exception:
            current_device_index = None
        logger.info(
            "[AcceleratorQueue][DIAG] enabled=%s slot=%s root=%s local_rank=%s rank=%s cuda_visible_devices=%s current_device=%s runtime_device=%s",
            queue_enabled,
            self._actor_fsdp_generate_lock.slot_key,
            self._actor_fsdp_generate_lock.queue_root,
            str(os.environ.get("LOCAL_RANK") or "").strip() or "<unset>",
            str(os.environ.get("RANK") or "").strip() or "<unset>",
            str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip() or "<unset>",
            current_device_index if current_device_index is not None else "<unknown>",
            str(self._accelerator_runtime_device) if self._accelerator_runtime_device is not None else "<unset>",
        )
        local_rollout_ulysses_enabled = self._local_rollout_ulysses_device_mesh is not None
        if local_rollout_ulysses_enabled:
            try:
                local_rollout_ulysses_dp_rank = int(self._local_rollout_ulysses_device_mesh["dp"].get_local_rank())
                local_rollout_ulysses_sp_rank = int(self._local_rollout_ulysses_device_mesh["sp"].get_local_rank())
                local_rollout_ulysses_sp_size = int(self._local_rollout_ulysses_device_mesh["sp"].size())
            except Exception:
                local_rollout_ulysses_dp_rank = 0
                local_rollout_ulysses_sp_rank = 0
                local_rollout_ulysses_sp_size = 1
        else:
            local_rollout_ulysses_dp_rank = 0
            local_rollout_ulysses_sp_rank = 0
            local_rollout_ulysses_sp_size = 1
        logger.info(
            "[LocalRolloutUlysses][DIAG] capable=%s dp_rank=%s sp_rank=%s sp_size=%s",
            local_rollout_ulysses_enabled,
            local_rollout_ulysses_dp_rank,
            local_rollout_ulysses_sp_rank,
            local_rollout_ulysses_sp_size,
        )
        self._active_local_rollout_ulysses = False
        self._fsdp_full_params_local_shard_active = False
        # Preserve multi-session rollout scheduling for bz>1 while ensuring the
        # in-process FSDP/Ulysses actor path is still driven by one thread at a
        # time inside this worker.
        self._fsdp_response_generation_lock = threading.RLock()
        _force_config_bool_false(
            self.config,
            "enable_prefix_caching",
            reason="video-agent rollout must keep prefix caching disabled across samples and turns",
        )
        _force_config_bool_false(
            self.config,
            "enable_rollout_prefix_cache",
            reason="video-agent rollout must not reuse shared rollout prefix cache",
        )
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_hf_config = model_hf_config
        self._validate_actor_model_class_matches_config()
        self.max_context_length = int(config.prompt_length + config.response_length)
        self.pad_token_id = (
            tokenizer.pad_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None
            else getattr(tokenizer, "eos_token_id", 0)
        )
        self._process_vision_info = None
        self._shared_preprocess_cache_max_entries = max(
            0,
            int(self.config.get("preprocess_payload_cache_max_entries", 2) or 2),
        )
        self._enable_shared_preprocess_cache = bool(
            self.config.get("enable_preprocess_payload_memory_cache", True)
        ) and self._shared_preprocess_cache_max_entries > 0
        # Keep preprocessing payload cache at rollout instance scope so
        # repeated queries for the same video can reuse in-process artifacts.
        self._shared_preprocess_cache: Dict[str, Dict[str, Any]] = OrderedDict()
        self._shared_vision_payload_cache_max_entries = max(
            0,
            int(self.config.get("vision_payload_cache_max_entries", 4) or 4),
        )
        self._enable_shared_vision_payload_cache = bool(
            self.config.get("enable_vision_payload_memory_cache", True)
        ) and self._shared_vision_payload_cache_max_entries > 0
        self._shared_vision_payload_cache: Dict[str, Dict[str, Any]] = OrderedDict()
        self._shared_rollout_prefix_cache_max_entries = 0
        self._enable_shared_rollout_prefix_cache = False
        self._shared_rollout_prefix_cache: Dict[str, Dict[str, Any]] = OrderedDict()
        self._supported_generate_kwargs: Optional[set[str]] = None
        self._logged_unsupported_generate_kwargs: set[str] = set()
        self._trace_max_samples_per_step_per_worker: Optional[int] = None
        self._init_rollout_trace_runtime()

    def _get_local_rollout_ulysses_group(self):
        if self._local_rollout_ulysses_device_mesh is None:
            return None
        try:
            return self._local_rollout_ulysses_device_mesh["sp"].get_group()
        except Exception:
            return None

    def _get_local_rollout_ulysses_world_size(self) -> int:
        group = self._get_local_rollout_ulysses_group()
        if group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return 1
        try:
            return int(torch.distributed.get_world_size(group))
        except Exception:
            return 1

    def _is_local_rollout_ulysses_enabled(self) -> bool:
        return bool(self._active_local_rollout_ulysses) and self._get_local_rollout_ulysses_world_size() > 1

    def _is_local_rollout_ulysses_leader(self) -> bool:
        if not self._is_local_rollout_ulysses_enabled():
            return True
        try:
            return bool(self._local_rollout_ulysses_device_mesh["sp"].get_local_rank() == 0)
        except Exception:
            return True

    def _broadcast_local_rollout_ulysses_command(self, command: Dict[str, Any]) -> None:
        group = self._get_local_rollout_ulysses_group()
        if group is None or self._get_local_rollout_ulysses_world_size() <= 1:
            return
        leader_global_rank = torch.distributed.get_global_rank(group, 0)
        payload = [dict(command or {})]
        torch.distributed.broadcast_object_list(payload, src=leader_global_rank, group=group)

    def _recv_local_rollout_ulysses_command(self) -> Dict[str, Any]:
        group = self._get_local_rollout_ulysses_group()
        if group is None or self._get_local_rollout_ulysses_world_size() <= 1:
            return {"type": "shutdown"}
        leader_global_rank = torch.distributed.get_global_rank(group, 0)
        payload = [None]
        torch.distributed.broadcast_object_list(payload, src=leader_global_rank, group=group)
        command = payload[0]
        if not isinstance(command, dict):
            raise RuntimeError(
                "Expected dict command for local rollout Ulysses follower loop, "
                f"but received {type(command).__name__}."
            )
        return command

    def _build_local_rollout_ulysses_generate_request(
        self,
        *,
        messages: List[Dict[str, Any]],
        do_sample: bool,
        role: Optional[str],
        max_tokens: Any,
        temperature: Any,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        serializable_kwargs: Dict[str, Any] = {}
        for key, value in dict(kwargs or {}).items():
            if callable(value):
                continue
            if str(key).startswith("_local_rollout_ulysses_"):
                continue
            serializable_kwargs[str(key)] = deepcopy(value)
        serializable_kwargs.pop("progress_callback", None)
        return {
            "type": "generate",
            "messages": deepcopy(messages),
            "do_sample": bool(do_sample),
            "role": role,
            "max_tokens": deepcopy(max_tokens),
            "temperature": deepcopy(temperature),
            "kwargs": serializable_kwargs,
        }

    def _empty_rollout_dataproto(self) -> DataProto:
        return DataProto(batch=None, non_tensor_batch={}, meta_info={})

    def _run_local_rollout_ulysses_follower_loop(self) -> DataProto:
        logger.info(
            "[LocalRolloutUlysses][DIAG] follower loop start rank=%s sp_world_size=%s",
            str(os.environ.get("RANK") or "").strip() or "<unset>",
            self._get_local_rollout_ulysses_world_size(),
        )
        while True:
            command = self._recv_local_rollout_ulysses_command()
            command_type = str(command.get("type") or "").strip().lower()
            if command_type in {"shutdown", "stop"}:
                logger.info("[LocalRolloutUlysses][DIAG] follower loop shutdown received.")
                break
            if command_type != "generate":
                raise RuntimeError(f"Unsupported local rollout Ulysses command: {command_type}")
            try:
                follower_kwargs = dict(command.get("kwargs") or {})
                follower_kwargs["_local_rollout_ulysses_follower_mode"] = True
                self._generate_response_from_messages(
                    messages=list(command.get("messages") or []),
                    do_sample=bool(command.get("do_sample", True)),
                    role=command.get("role"),
                    max_tokens=command.get("max_tokens"),
                    temperature=command.get("temperature"),
                    **follower_kwargs,
                )
            except Exception:
                logger.error(
                    "[LocalRolloutUlysses][ERROR] follower generation failed but will stay alive for leader shutdown.\n%s",
                    traceback.format_exc(),
                )
        return self._empty_rollout_dataproto()

    def _validate_actor_model_class_matches_config(self) -> None:
        config = self.model_hf_config
        model_type = str(getattr(config, "model_type", "") or "").strip().lower()
        if model_type not in {"qwen3_5", "qwen3_vl"}:
            return
        architectures = [
            str(arch).strip()
            for arch in (getattr(config, "architectures", None) or [])
            if str(arch).strip()
        ]
        actor_class = type(self.actor_module).__name__
        if architectures and actor_class in architectures:
            return
        if model_type == "qwen3_5" and actor_class.startswith("Qwen3VL"):
            raise RuntimeError(
                "Refusing to run rollout with Qwen3-VL actor class for qwen3_5 checkpoint. "
                f"actor_class={actor_class}; architectures={architectures}; "
                "this architecture mismatch can create missing/mismatched weights and CUDA illegal memory access."
            )
        if model_type == "qwen3_vl" and actor_class.startswith("Qwen3_5"):
            raise RuntimeError(
                "Refusing to run rollout with Qwen3.5 actor class for qwen3_vl checkpoint. "
                f"actor_class={actor_class}; architectures={architectures}."
            )

    def _init_rollout_trace_runtime(self) -> None:
        trace_config = self.config.get("trace", {})
        trace_getter = getattr(trace_config, "get", None)

        def _trace_get(key: str, default: Any = None) -> Any:
            if callable(trace_getter):
                try:
                    return trace_getter(key, default)
                except Exception:
                    return default
            if isinstance(trace_config, dict):
                return trace_config.get(key, default)
            return default

        raw_backend = str(_trace_get("backend", "") or "").strip().lower()
        backend = raw_backend if raw_backend and raw_backend not in {"none", "null"} else None
        project_name = str(_trace_get("project_name", "") or "").strip() or "retool_video_rl"
        experiment_name = str(_trace_get("experiment_name", "") or "").strip() or "video_agent_rollout"
        token2text = _coerce_bool(_trace_get("token2text", False), default=False)
        max_samples_per_worker = _coerce_optional_int(_trace_get("max_samples_per_step_per_worker", None))
        if max_samples_per_worker is not None and max_samples_per_worker < 0:
            max_samples_per_worker = None

        RolloutTraceConfig.init(
            project_name=project_name,
            experiment_name=experiment_name,
            backend=backend,
            token2text=token2text,
            max_samples_per_step_per_worker=max_samples_per_worker,
        )
        self._trace_max_samples_per_step_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        logger.info(
            "[RolloutTrace][DIAG] backend=%s project=%s experiment=%s max_samples_per_step_per_worker=%s",
            RolloutTraceConfig.get_backend(),
            project_name,
            experiment_name,
            self._trace_max_samples_per_step_per_worker,
        )

    def _select_traced_sample_indices(self, sample_indices: List[int]) -> set[int]:
        if not sample_indices:
            return set()
        unique_sample_indices = list(dict.fromkeys(int(sample_index) for sample_index in sample_indices))
        max_samples_per_worker = self._trace_max_samples_per_step_per_worker
        if max_samples_per_worker is None or max_samples_per_worker >= len(unique_sample_indices):
            return set(unique_sample_indices)
        if max_samples_per_worker <= 0:
            return set()
        selected_samples = np.random.choice(
            np.array(unique_sample_indices, dtype=np.int64),
            size=int(max_samples_per_worker),
            replace=False,
        )
        return set(int(item) for item in selected_samples.tolist())

    async def resume(self, tags: list[str]):
        return None

    async def update_weights(self, weights, **kwargs):
        return None

    async def release(self):
        return None

    def _resolve_forward_autocast_dtype(self) -> Optional[torch.dtype]:
        raw_dtype = str(self.config.get("dtype", "") or "").strip().lower()
        configured = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
        }.get(raw_dtype)
        if configured is not None:
            return configured

        try:
            first_param = next(self.actor_module.parameters())
        except Exception:
            return None
        if torch.is_tensor(first_param) and first_param.is_floating_point():
            if first_param.dtype in (torch.float16, torch.bfloat16):
                return first_param.dtype
        return None

    def _resolve_model_input_float_dtype(self) -> Optional[torch.dtype]:
        for module in (getattr(self.actor_module, "visual", None), self.actor_module):
            if module is None:
                continue
            parameters = getattr(module, "parameters", None)
            if not callable(parameters):
                continue
            try:
                for param in parameters():
                    if torch.is_tensor(param) and param.is_floating_point():
                        return param.dtype
            except Exception:
                continue
        return self._resolve_forward_autocast_dtype()

    def _filter_supported_generate_kwargs(self, generation_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self._supported_generate_kwargs is None:
            supported = {
                "max_new_tokens",
                "pad_token_id",
                "use_cache",
                "do_sample",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "eos_token_id",
                "max_time",
                "synced_gpus",
                "stopping_criteria",
                "remove_invalid_values",
                "renormalize_logits",
                "logits_processor",
            }
            generation_config = getattr(self.actor_module, "generation_config", None)
            if generation_config is not None and hasattr(generation_config, "to_dict"):
                try:
                    supported.update(generation_config.to_dict().keys())
                except Exception:
                    pass
            self._supported_generate_kwargs = supported

        filtered: Dict[str, Any] = {}
        for key, value in dict(generation_kwargs or {}).items():
            if value is None:
                continue
            if key in self._supported_generate_kwargs:
                filtered[key] = value
                continue
            if key not in self._logged_unsupported_generate_kwargs:
                logger.warning("Skipping unsupported rollout generate kwarg '%s'.", key)
                self._logged_unsupported_generate_kwargs.add(key)
        return filtered

    def _resolve_stop_token_ids(self) -> Tuple[int, ...]:
        token_ids: List[int] = []
        for source in (
            getattr(self.actor_module, "generation_config", None),
            self.model_hf_config,
            self.tokenizer,
        ):
            if source is None:
                continue
            token_ids.extend(_coerce_token_ids(getattr(source, "eos_token_id", None)))
            token_ids.extend(_coerce_token_ids(getattr(source, "eos_token_ids", None)))
        if not token_ids:
            token_ids.extend(_coerce_token_ids(getattr(self.tokenizer, "pad_token_id", None)))
        return tuple(dict.fromkeys(int(token_id) for token_id in token_ids))

    def _resolve_device(self) -> torch.device:
        try:
            return next(self.actor_module.parameters()).device
        except Exception:
            return torch.device("cuda")

    def _capture_device_memory_snapshot(self, device: torch.device) -> Optional[Dict[str, int]]:
        device_type = str(getattr(device, "type", "") or "").strip().lower()
        if device_type not in {"cuda", "npu"}:
            return None
        device_module = getattr(torch, device_type, None)
        if device_module is None:
            return None
        is_available_fn = getattr(device_module, "is_available", None)
        if not callable(is_available_fn) or not bool(is_available_fn()):
            return None

        device_index = getattr(device, "index", None)
        if device_index is None:
            current_device_fn = getattr(device_module, "current_device", None)
            if callable(current_device_fn):
                try:
                    device_index = int(current_device_fn())
                except Exception:
                    device_index = 0
            else:
                device_index = 0

        mem_get_info = getattr(device_module, "mem_get_info", None)
        memory_allocated = getattr(device_module, "memory_allocated", None)
        memory_reserved = getattr(device_module, "memory_reserved", None)
        if not callable(mem_get_info) or not callable(memory_allocated) or not callable(memory_reserved):
            return None

        try:
            free_bytes, total_bytes = mem_get_info(int(device_index))
        except TypeError:
            free_bytes, total_bytes = mem_get_info()
        except Exception:
            return None

        try:
            allocated_bytes = int(memory_allocated(int(device_index)))
        except TypeError:
            allocated_bytes = int(memory_allocated())
        except Exception:
            allocated_bytes = -1

        try:
            reserved_bytes = int(memory_reserved(int(device_index)))
        except TypeError:
            reserved_bytes = int(memory_reserved())
        except Exception:
            reserved_bytes = -1

        return {
            "device_index": int(device_index),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "allocated_bytes": int(allocated_bytes),
            "reserved_bytes": int(reserved_bytes),
        }

    def _estimate_generate_memory_required_bytes(
        self,
        *,
        prompt_length: int,
        image_count: int,
        video_count: int,
        role: str,
        decode_mode: str,
    ) -> int:
        prompt_k_tokens = max(0.0, float(prompt_length) / 1024.0)
        required_gb = _coerce_float(
            self.config.get("generate_memory_admission_base_gb", 1.5),
            1.5,
        )
        required_gb += prompt_k_tokens * _coerce_float(
            self.config.get("generate_memory_admission_per_1k_prompt_tokens_gb", 0.25),
            0.25,
        )
        required_gb += max(0, int(image_count)) * _coerce_float(
            self.config.get("generate_memory_admission_per_image_gb", 0.35),
            0.35,
        )
        required_gb += max(0, int(video_count)) * _coerce_float(
            self.config.get("generate_memory_admission_per_video_gb", 1.0),
            1.0,
        )
        if str(role or "").strip().lower() == "planner":
            required_gb += _coerce_float(
                self.config.get("generate_memory_admission_planner_extra_gb", 0.5),
                0.5,
            )
        if "hf_generate_full_sequence" in str(decode_mode or ""):
            required_gb += _coerce_float(
                self.config.get("generate_memory_admission_full_sequence_extra_gb", 1.0),
                1.0,
            )
        return max(0, int(required_gb * (1024**3)))

    def _wait_for_generate_memory_admission(
        self,
        *,
        device: torch.device,
        role: str,
        prompt_length: int,
        image_count: int,
        video_count: int,
        decode_mode: str,
        generation_deadline: Optional[float],
        wall_clock_deadline: Optional[float],
        extra_margin_bytes: int = 0,
    ) -> None:
        if not bool(self.config.get("enable_generate_memory_admission", True)):
            return

        memory_snapshot = self._capture_device_memory_snapshot(device)
        if memory_snapshot is None:
            return

        poll_seconds = max(
            0.1,
            _coerce_float(self.config.get("generate_memory_admission_poll_seconds", 5.0), 5.0),
        )
        log_wait_seconds = max(
            poll_seconds,
            _coerce_float(self.config.get("generate_memory_admission_log_wait_seconds", 30.0), 30.0),
        )
        configured_timeout_seconds = _coerce_float(
            self.config.get("generate_memory_admission_timeout_seconds", 0.0),
            0.0,
        )
        margin_bytes = max(
            0,
            int(
                _coerce_float(self.config.get("generate_memory_admission_margin_gb", 0.5), 0.5)
                * (1024**3)
            ),
        )
        required_bytes = self._estimate_generate_memory_required_bytes(
            prompt_length=prompt_length,
            image_count=image_count,
            video_count=video_count,
            role=role,
            decode_mode=decode_mode,
        )
        required_with_margin_bytes = required_bytes + margin_bytes + max(0, int(extra_margin_bytes))
        started = time.perf_counter()
        last_wait_log = started

        while True:
            try:
                if str(getattr(device, "type", "") or "").strip().lower() == "cuda":
                    set_expandable_segments(True)
            except Exception:
                pass
            aggressive_empty_cache(force_sync=True, max_retries=1)
            memory_snapshot = self._capture_device_memory_snapshot(device)
            if memory_snapshot is None:
                return

            free_bytes = int(memory_snapshot.get("free_bytes", 0) or 0)
            if free_bytes >= required_with_margin_bytes:
                logger.info(
                    "[RolloutGenerate][MemoryAdmission][DIAG] role=%s admitted device=%s prompt_tokens=%d image_count=%d video_count=%d decode_mode=%s free_gb=%.2f required_gb=%.2f margin_gb=%.2f elapsed=%.2fs",
                    role or "llm",
                    device,
                    int(prompt_length),
                    int(image_count),
                    int(video_count),
                    decode_mode,
                    free_bytes / (1024**3),
                    required_bytes / (1024**3),
                    (required_with_margin_bytes - required_bytes) / (1024**3),
                    time.perf_counter() - started,
                )
                return

            now = time.perf_counter()
            remaining_limits: List[float] = []
            if configured_timeout_seconds > 0:
                remaining_limits.append(configured_timeout_seconds - (now - started))
            if generation_deadline is not None:
                remaining_limits.append(float(generation_deadline) - now)
            if wall_clock_deadline is not None:
                remaining_limits.append(float(wall_clock_deadline) - now)
            remaining_wait_seconds = min(remaining_limits) if remaining_limits else None
            if remaining_wait_seconds is not None and remaining_wait_seconds <= 0:
                raise GenerationTimeoutError(
                    "Waiting for enough free accelerator memory exceeded the current rollout deadline.",
                    partial_response="",
                    elapsed_seconds=now - started,
                    role=role or "llm",
                    phase="memory_admission_wait",
                )

            if (now - last_wait_log) >= log_wait_seconds:
                logger.info(
                    "[RolloutGenerate][MemoryAdmission][DIAG] role=%s waiting device=%s prompt_tokens=%d image_count=%d video_count=%d decode_mode=%s free_gb=%.2f required_gb=%.2f margin_gb=%.2f allocated_gb=%.2f reserved_gb=%.2f remaining_wait_seconds=%s",
                    role or "llm",
                    device,
                    int(prompt_length),
                    int(image_count),
                    int(video_count),
                    decode_mode,
                    free_bytes / (1024**3),
                    required_bytes / (1024**3),
                    (required_with_margin_bytes - required_bytes) / (1024**3),
                    int(memory_snapshot.get("allocated_bytes", 0) or 0) / (1024**3),
                    int(memory_snapshot.get("reserved_bytes", 0) or 0) / (1024**3),
                    (
                        f"{float(remaining_wait_seconds):.1f}"
                        if remaining_wait_seconds is not None
                        else "unbounded"
                    ),
                )
                last_wait_log = now

            sleep_seconds = poll_seconds
            if remaining_wait_seconds is not None:
                sleep_seconds = min(sleep_seconds, max(0.1, float(remaining_wait_seconds)))
            time.sleep(sleep_seconds)

    def _validate_rollout_model_inputs(
        self,
        model_inputs: Dict[str, Any],
        *,
        prompt_length: int,
        image_count: int,
        video_count: int,
        role: str,
    ) -> None:
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise RuntimeError(f"Rollout {role or 'llm'} prompt input_ids must be a rank-2 tensor.")
        if int(prompt_length) <= 0:
            raise RuntimeError(f"Rollout {role or 'llm'} prompt rendered to zero tokens.")
        if torch.is_tensor(attention_mask) and tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise RuntimeError(
                f"Rollout {role or 'llm'} prompt attention_mask shape {tuple(attention_mask.shape)} "
                f"does not match input_ids shape {tuple(input_ids.shape)}."
            )

        if bool(self.config.get("validate_rollout_token_ids", True)):
            min_token_id = int(input_ids.min().item())
            max_token_id = int(input_ids.max().item())
            vocab_size = getattr(self.model_hf_config, "vocab_size", None)
            if vocab_size is None:
                vocab_size = getattr(getattr(self.actor_module, "config", None), "vocab_size", None)
            if min_token_id < 0 or (vocab_size is not None and max_token_id >= int(vocab_size)):
                raise RuntimeError(
                    f"Rollout {role or 'llm'} prompt contains invalid token ids: "
                    f"min={min_token_id} max={max_token_id} vocab_size={vocab_size}."
                )

        if bool(self.config.get("validate_rollout_multimodal_inputs", True)):
            for key in _FLOAT_MULTIMODAL_TENSOR_KEYS:
                tensor_value = model_inputs.get(key)
                if torch.is_tensor(tensor_value) and tensor_value.is_floating_point():
                    if not bool(torch.isfinite(tensor_value).all().item()):
                        raise RuntimeError(f"Rollout {role or 'llm'} input tensor {key} contains NaN or Inf.")

            image_token_id = getattr(self.model_hf_config, "image_token_id", None)
            if image_token_id is None:
                image_token_id = getattr(getattr(self.actor_module, "config", None), "image_token_id", None)
            video_token_id = getattr(self.model_hf_config, "video_token_id", None)
            if video_token_id is None:
                video_token_id = getattr(getattr(self.actor_module, "config", None), "video_token_id", None)
            if image_count > 0 and image_token_id is not None and int((input_ids == int(image_token_id)).sum().item()) <= 0:
                raise RuntimeError(
                    f"Rollout {role or 'llm'} has image tensors but no image placeholder tokens in the prompt."
                )
            if video_count > 0 and video_token_id is not None and int((input_ids == int(video_token_id)).sum().item()) <= 0:
                raise RuntimeError(
                    f"Rollout {role or 'llm'} has video tensors but no video placeholder tokens in the prompt."
                )

    def _resolve_process_vision_info(self):
        if self._process_vision_info is not None:
            return self._process_vision_info
        try:
            from qwen_vl_utils.vision_process import process_vision_info  # type: ignore
        except Exception:
            from qwen_vl_utils import process_vision_info  # type: ignore
        self._process_vision_info = process_vision_info
        return process_vision_info

    def _get_lru_cache_entry(
        self,
        cache: Dict[str, Dict[str, Any]],
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        cached_payload = cache.get(cache_key)
        if cached_payload is None:
            return None
        move_to_end = getattr(cache, "move_to_end", None)
        if callable(move_to_end):
            move_to_end(cache_key)
        return cached_payload

    def _set_lru_cache_entry(
        self,
        cache: Dict[str, Dict[str, Any]],
        cache_key: str,
        payload: Dict[str, Any],
        *,
        max_entries: int,
    ) -> None:
        if not cache_key or max_entries <= 0:
            return
        cache[cache_key] = payload
        move_to_end = getattr(cache, "move_to_end", None)
        if callable(move_to_end):
            move_to_end(cache_key)
        while len(cache) > max_entries:
            popitem = getattr(cache, "popitem", None)
            if callable(popitem):
                popitem(last=False)
                continue
            oldest_key = next(iter(cache), None)
            if oldest_key is None:
                break
            cache.pop(oldest_key, None)

    def _build_visual_payload_cache_key(
        self,
        normalized_messages: List[Dict[str, Any]],
    ) -> str:
        projected_messages: List[Dict[str, Any]] = []
        for message in normalized_messages or []:
            projected_items: List[Dict[str, Any]] = []
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type", "")).strip().lower()
                    if item_type == "text":
                        continue
                    projected_items.append(
                        {
                            str(key): _sanitize_cache_key_value(value)
                            for key, value in item.items()
                            if str(key) != "text"
                        }
                    )
            elif isinstance(content, dict):
                item_type = str(content.get("type", "")).strip().lower()
                if item_type and item_type != "text":
                    projected_items.append(
                        {
                            str(key): _sanitize_cache_key_value(value)
                            for key, value in content.items()
                            if str(key) != "text"
                        }
                    )
            if projected_items:
                projected_messages.append(
                    {
                        "role": str(message.get("role", "user")).strip() or "user",
                        "content": projected_items,
                    }
                )
        if not projected_messages:
            return ""
        return _stable_sha1(projected_messages)

    def _build_rollout_prefix_cache_key(self, session_id: Any, role: Any) -> str:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return ""
        normalized_role = _normalize_generation_role(role) or "llm"
        return f"{normalized_session_id}:{normalized_role}"

    def _apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        normalized_messages = _ensure_user_message(_normalize_multimodal_messages(messages))
        if not normalized_messages:
            if not add_generation_prompt:
                return ""
            normalized_messages = [{"role": "user", "content": ""}]

        client = self.processor or self.tokenizer
        rendered = _apply_chat_template_with_optional_thinking(
            client,
            normalized_messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        if rendered is not None:
            return rendered

        text_messages = []
        for message in normalized_messages:
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
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
            if rendered is not None:
                return rendered
        return _render_text_chat(text_messages, add_generation_prompt=add_generation_prompt)

    def _load_process_vision_payload(
        self,
        normalized_messages: List[Dict[str, Any]],
        *,
        cache_key: str = "",
    ) -> Tuple[Any, Any, Dict[str, Any], bool]:
        if cache_key and self._enable_shared_vision_payload_cache:
            cached_payload = self._get_lru_cache_entry(
                self._shared_vision_payload_cache,
                cache_key,
            )
            if cached_payload is not None:
                return (
                    cached_payload.get("image_inputs"),
                    cached_payload.get("video_inputs"),
                    dict(cached_payload.get("video_kwargs") or {}),
                    True,
                )
        process_vision_info = self._resolve_process_vision_info()
        kwargs = {
            "image_patch_size": 16,
            "return_video_metadata": True,
            "return_video_kwargs": True,
        }
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info([normalized_messages], **kwargs)
            resolved_video_kwargs = dict(video_kwargs or {})
        except TypeError:
            image_inputs, video_inputs = process_vision_info([normalized_messages])
            resolved_video_kwargs = {}
        if cache_key and self._enable_shared_vision_payload_cache:
            self._set_lru_cache_entry(
                self._shared_vision_payload_cache,
                cache_key,
                {
                    "image_inputs": image_inputs,
                    "video_inputs": video_inputs,
                    "video_kwargs": dict(resolved_video_kwargs or {}),
                },
                max_entries=self._shared_vision_payload_cache_max_entries,
            )
        return image_inputs, video_inputs, resolved_video_kwargs, False

    def _prepare_message_inputs(
        self,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool,
        device: torch.device,
        enable_thinking: Optional[bool] = None,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_messages = _normalize_multimodal_messages(messages)
        normalized_role = _normalize_generation_role(role)
        contains_visual_payload = _messages_contain_visual_inputs(normalized_messages)
        if normalized_role == "planner" and contains_visual_payload and self.processor is None:
            raise RuntimeError(
                "Planner received visual inputs during RL rollout, but no multimodal processor was initialized. "
                "RL planner must run in VLM mode when grounding/images are attached."
            )
        prompt_preview = self._apply_chat_template(
            normalized_messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        contains_visual = bool(
            self.processor is not None and contains_visual_payload
        )
        visual_signature = (
            self._build_visual_payload_cache_key(normalized_messages)
            if contains_visual
            else ""
        )
        vision_payload_cache_hit = False

        if contains_visual:
            image_inputs, video_inputs, video_kwargs, vision_payload_cache_hit = self._load_process_vision_payload(
                normalized_messages,
                cache_key=visual_signature,
            )
            videos, video_metadata = _split_video_inputs_and_metadata(video_inputs)
            processor_kwargs: Dict[str, Any] = {
                "text": [prompt_preview],
                "padding": True,
                "return_tensors": "pt",
            }
            if image_inputs is not None:
                processor_kwargs["images"] = image_inputs
            if videos is not None:
                processor_kwargs["videos"] = videos
            if video_metadata:
                processor_kwargs["video_metadata"] = video_metadata
            processor_kwargs.update(
                {
                    key: value
                    for key, value in dict(video_kwargs or {}).items()
                    if value is not None
                }
            )
            inputs = _normalize_processor_sequence_tensors(
                dict(self.processor(**processor_kwargs) or {}),
                source="vllm_video_agent_rollout.prepare_message_inputs",
            )
        else:
            prompt_ids = self.tokenizer.encode(prompt_preview, add_special_tokens=False)
            inputs = {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long),
            }

        model_inputs: Dict[str, Any] = {}
        target_float_dtype = self._resolve_model_input_float_dtype()
        for key, value in _normalize_processor_sequence_tensors(
            dict(inputs or {}),
            source="vllm_video_agent_rollout.model_inputs",
        ).items():
            if value is None:
                continue
            if torch.is_tensor(value):
                tensor_value = value.to(device)
                if (
                    key in _FLOAT_MULTIMODAL_TENSOR_KEYS
                    and tensor_value.is_floating_point()
                    and target_float_dtype is not None
                ):
                    tensor_value = tensor_value.to(dtype=target_float_dtype)
                model_inputs[key] = tensor_value
            else:
                model_inputs[key] = value

        prompt_tensor = model_inputs["input_ids"][0]
        image_count = 0
        video_count = 0
        if torch.is_tensor(model_inputs.get("image_grid_thw")):
            image_count = int(model_inputs["image_grid_thw"].shape[0])
        elif torch.is_tensor(model_inputs.get("pixel_values")):
            image_count = int(model_inputs["pixel_values"].shape[0])
        if torch.is_tensor(model_inputs.get("video_grid_thw")):
            video_count = int(model_inputs["video_grid_thw"].shape[0])
        elif torch.is_tensor(model_inputs.get("pixel_values_videos")):
            video_count = int(model_inputs["pixel_values_videos"].shape[0])

        return {
            "prompt_messages": normalized_messages,
            "prompt_preview": prompt_preview,
            "prompt_ids": _tensor_to_token_list(prompt_tensor),
            "model_inputs": model_inputs,
            "image_count": image_count,
            "video_count": video_count,
            "visual_signature": visual_signature,
            "vision_payload_cache_hit": vision_payload_cache_hit,
        }

    def _actor_forward_with_cache(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        forward_autocast_dtype: Optional[torch.dtype],
        use_cache: bool,
        static_kwargs: Optional[Dict[str, Any]] = None,
        past_key_values: Any = None,
    ) -> Tuple[Any, torch.Tensor, Any]:
        model_call_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": bool(use_cache),
            "return_dict": True,
        }
        if past_key_values is not None:
            model_call_kwargs["past_key_values"] = past_key_values
        else:
            model_call_kwargs.update(dict(static_kwargs or {}))

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=forward_autocast_dtype)
            if device.type == "cuda" and forward_autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        try:
            with autocast_context:
                output = self.actor_module(**model_call_kwargs)
        except TypeError as exc:
            if "return_dict" in model_call_kwargs and "return_dict" in str(exc):
                model_call_kwargs.pop("return_dict", None)
                with autocast_context:
                    output = self.actor_module(**model_call_kwargs)
            else:
                raise

        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, dict):
            logits = output.get("logits")
        if logits is None and isinstance(output, (list, tuple)) and output:
            logits = output[0]
        if logits is None:
            raise RuntimeError("Actor module returned no logits during rollout generation.")

        next_past_key_values = _extract_past_key_values(output) if use_cache else None
        return output, logits, next_past_key_values

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        input_ids: Optional[torch.Tensor] = None,
        do_sample: bool,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> torch.Tensor:
        working = _apply_repetition_penalty(
            logits,
            input_ids=input_ids,
            repetition_penalty=repetition_penalty,
        )
        working = _sanitize_logits_for_sampling(working)
        if not do_sample:
            return torch.argmax(working, dim=-1, keepdim=True)
        if temperature is not None:
            safe_temperature = max(1e-5, _coerce_float(temperature, 1.0))
            working = working / safe_temperature
        working = _apply_top_k_top_p_filter(
            working,
            top_k=_clamp_optional_int(top_k, default=-1, minimum=-1, maximum=int(working.shape[-1])),
            top_p=_clamp_float(top_p, default=1.0, minimum=1e-6, maximum=1.0),
            min_p=_clamp_float(min_p, default=0.0, minimum=0.0, maximum=1.0),
        )
        return _safe_multinomial_from_logits(working)

    def _generate_response_from_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        do_sample: bool,
        role: Optional[str] = None,
        max_tokens: Any = None,
        temperature: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self._fatal_accelerator_error:
            raise AcceleratorFatalError(
                "Rollout worker accelerator context is poisoned by a previous fatal error: "
                f"{self._fatal_accelerator_error}"
            )
        local_rollout_ulysses_follower_mode = bool(
            kwargs.pop("_local_rollout_ulysses_follower_mode", False)
        )
        device = self._resolve_device()
        normalized_role = _normalize_generation_role(role)
        if normalized_role == "subagent":
            # Enforce planner/subagent boundary: subagent must remain no-think
            # even if external overrides or stale runtime defaults request think.
            kwargs["enable_thinking"] = False
        subagent_inference_only = bool(
            normalized_role == "subagent"
            and self.config.get("subagent_inference_only", False)
        )
        if subagent_inference_only:
            if bool(self.config.get("subagent_force_greedy", False)):
                do_sample = False
                temperature = None
            elif bool(self.config.get("subagent_match_e2e_sampling_profile", True)):
                # Keep subagent decoding behavior close to run_e2e defaults:
                # temperature-driven sampling, no-think, and default sampling controls.
                temperature = 1.0
                do_sample = True
                kwargs["enable_thinking"] = False
                kwargs["top_p"] = 0.95
                kwargs["top_k"] = 20
                kwargs["min_p"] = 0.0
                kwargs["repetition_penalty"] = 1.0
        if (
            self._is_local_rollout_ulysses_enabled()
            and self._is_local_rollout_ulysses_leader()
            and not local_rollout_ulysses_follower_mode
        ):
            logger.info(
                "[LocalRolloutUlysses][DIAG] leader dispatching distributed generate role=%s message_count=%d do_sample=%s",
                normalized_role or "llm",
                len(messages or []),
                bool(do_sample),
            )
            self._broadcast_local_rollout_ulysses_command(
                self._build_local_rollout_ulysses_generate_request(
                    messages=messages,
                    do_sample=do_sample,
                    role=role,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    kwargs=kwargs,
                )
            )
        emit_role_diag_logs = not (
            subagent_inference_only
            and bool(self.config.get("subagent_reduce_diagnostic_logging", True))
        )
        if local_rollout_ulysses_follower_mode:
            emit_role_diag_logs = False
        wall_clock_deadline = _coerce_monotonic_deadline(
            kwargs.get("wall_clock_deadline_monotonic")
        )
        generation_deadline = _coerce_monotonic_deadline(
            kwargs.get("generation_deadline_monotonic")
        )
        progress_callback = kwargs.get("progress_callback") if callable(kwargs.get("progress_callback")) else None
        if subagent_inference_only and bool(self.config.get("subagent_disable_progress_events", True)):
            progress_callback = None
        if local_rollout_ulysses_follower_mode:
            progress_callback = None
        try:
            heartbeat_interval_seconds = max(
                5.0,
                float(kwargs.get("generation_heartbeat_interval_seconds", 30.0) or 30.0),
            )
        except Exception:
            heartbeat_interval_seconds = 30.0
        generation_started = time.perf_counter()
        next_heartbeat_monotonic = generation_started + heartbeat_interval_seconds
        generated_ids: List[int] = []
        forward_autocast_dtype = self._resolve_forward_autocast_dtype()
        if emit_role_diag_logs:
            logger.info(
                "[RolloutGenerate][DIAG] role=%s actor_type=%s actor_source=%s device=%s processor_enabled=%s do_sample=%s temperature=%s max_tokens=%s response_length=%s heartbeat_interval_seconds=%.3f generation_deadline=%s wall_clock_deadline=%s forward_autocast_dtype=%s",
                normalized_role or "llm",
                type(self.actor_module).__name__,
                self._actor_module_source,
                device,
                bool(self.processor is not None),
                bool(do_sample),
                temperature,
                max_tokens,
                getattr(self.config, "response_length", None),
                heartbeat_interval_seconds,
                (
                    f"{generation_deadline - time.perf_counter():.3f}s"
                    if generation_deadline is not None
                    else "unbounded"
                ),
                (
                    f"{wall_clock_deadline - time.perf_counter():.3f}s"
                    if wall_clock_deadline is not None
                    else "unbounded"
                ),
                str(forward_autocast_dtype).replace("torch.", "") if forward_autocast_dtype is not None else "disabled",
            )

        def _diag_log(message: str, *args: Any) -> None:
            if emit_role_diag_logs:
                logger.info(message, *args)

        def _emit_generation_progress(*, source: str, result_preview: str, status: str = "running") -> None:
            callback = progress_callback
            if not callable(callback):
                return
            remaining_generation_seconds: Optional[float] = None
            if generation_deadline is not None:
                try:
                    remaining_generation_seconds = generation_deadline - time.perf_counter()
                except Exception:
                    remaining_generation_seconds = None
            remaining_wall_clock_seconds: Optional[float] = None
            if wall_clock_deadline is not None:
                try:
                    remaining_wall_clock_seconds = wall_clock_deadline - time.perf_counter()
                except Exception:
                    remaining_wall_clock_seconds = None
            callback(
                {
                    "status": str(status or "running").strip() or "running",
                    "source": str(source or "llm").strip() or "llm",
                    "elapsed_seconds": round(float(time.perf_counter() - generation_started), 3),
                    "generation_remaining_seconds": (
                        round(float(remaining_generation_seconds), 3)
                        if remaining_generation_seconds is not None
                        else None
                    ),
                    "wall_clock_remaining_seconds": (
                        round(float(remaining_wall_clock_seconds), 3)
                        if remaining_wall_clock_seconds is not None
                        else None
                    ),
                    "result_preview": _truncate_text(str(result_preview or "").strip(), 220),
                }
            )

        _emit_generation_progress(
            source="prepare_inputs_start",
            result_preview=(
                f"starting rollout generation role={normalized_role or 'llm'} "
                f"messages={len(messages or [])}"
            ),
        )
        prepared = self._prepare_message_inputs(
            messages,
            add_generation_prompt=True,
            device=device,
            enable_thinking=kwargs.get("enable_thinking"),
            role=role,
        )
        model_inputs = prepared["model_inputs"]
        current_input_ids = model_inputs["input_ids"]
        current_attention_mask = model_inputs["attention_mask"]
        prompt_length = int(current_input_ids.shape[-1])
        image_count = int(prepared.get("image_count", 0) or 0)
        video_count = int(prepared.get("video_count", 0) or 0)
        self._validate_rollout_model_inputs(
            model_inputs,
            prompt_length=prompt_length,
            image_count=image_count,
            video_count=video_count,
            role=normalized_role or "llm",
        )
        static_kwargs = _build_generation_static_kwargs(
            model_inputs,
            prompt_length=prompt_length,
        )
        _emit_generation_progress(
            source="prepare_inputs_done",
            result_preview=(
                f"prompt_tokens={prompt_length} "
                f"image_count={int(prepared.get('image_count', 0) or 0)} "
                f"video_count={int(prepared.get('video_count', 0) or 0)} "
                f"vision_cache_hit={bool(prepared.get('vision_payload_cache_hit', False))}"
            ),
        )
        _diag_log(
            "[RolloutGenerate][DIAG] role=%s prepared_inputs prompt_tokens=%d image_count=%d video_count=%d vision_cache_hit=%s model_inputs=%s static_kwargs=%s",
            normalized_role or "llm",
            prompt_length,
            int(prepared.get("image_count", 0) or 0),
            int(prepared.get("video_count", 0) or 0),
            bool(prepared.get("vision_payload_cache_hit", False)),
            json.dumps(_summarize_tensor_payload(model_inputs), ensure_ascii=False),
            json.dumps(_summarize_tensor_payload(static_kwargs), ensure_ascii=False),
        )

        remaining_budget = max(1, self.max_context_length - prompt_length)
        token_budget = min(
            int(max_tokens or self.config.response_length or 1),
            int(self.config.response_length),
            remaining_budget,
        )

        stop_token_ids = self._resolve_stop_token_ids()
        stop_token_id_set = set(stop_token_ids)
        disable_gradient_checkpointing_during_rollout = bool(
            kwargs.get(
                "disable_gradient_checkpointing_during_rollout",
                self.config.get("disable_gradient_checkpointing_during_rollout", True),
            )
        )
        force_use_cache_during_rollout = bool(
            kwargs.get(
                "force_use_cache_during_rollout",
                self.config.get("force_use_cache_during_rollout", True),
            )
        )
        enable_planner_incremental_decode = bool(
            self.config.get("enable_planner_incremental_decode", False)
        )
        prefer_subagent_model_generate_for_e2e = bool(
            subagent_inference_only
            and self.config.get("subagent_prefer_model_generate_for_e2e", True)
        )
        use_incremental_kv_cache = (
            (normalized_role == "subagent" and not prefer_subagent_model_generate_for_e2e)
            or (normalized_role == "planner" and enable_planner_incremental_decode)
        )
        use_text_only_kv_cache = (
            use_incremental_kv_cache
            and image_count <= 0
            and video_count <= 0
            and not bool(static_kwargs)
        )
        if prefer_subagent_model_generate_for_e2e:
            decode_mode = "hf_generate_subagent_e2e"
        elif not use_incremental_kv_cache:
            decode_mode = "hf_generate_full_sequence"
        elif use_text_only_kv_cache:
            decode_mode = "text_kv_cache"
        else:
            decode_mode = "multimodal_kv_cache"
        # Check structured payload frequently so generation stops close to the
        # first complete actionable JSON object and avoids long garbage tails.
        default_structured_stop_check_interval = 1 if normalized_role == "subagent" else 2
        structured_stop_check_interval = max(
            1,
            int(
                kwargs.get(
                    "structured_stop_check_interval",
                    self.config.get(
                        "structured_stop_check_interval",
                        default_structured_stop_check_interval,
                    ),
                )
                or default_structured_stop_check_interval
            ),
        )
        if temperature is not None:
            temperature = _clamp_float(temperature, default=1.0, minimum=1e-5, maximum=100.0)
        top_k = _clamp_optional_int(
            kwargs.get("top_k", getattr(self.config, "top_k", -1)),
            default=-1,
            minimum=-1,
            maximum=1_000_000,
        )
        top_p = _clamp_float(
            kwargs.get("top_p", getattr(self.config, "top_p", 1.0)),
            default=1.0,
            minimum=1e-6,
            maximum=1.0,
        )
        min_p = _clamp_float(
            kwargs.get("min_p", getattr(self.config, "min_p", 0.0)),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )
        raw_repetition_penalty = kwargs.get(
            "repetition_penalty",
            getattr(self.config, "repetition_penalty", None),
        )
        repetition_penalty = None
        if raw_repetition_penalty not in (None, ""):
            repetition_penalty = max(1e-6, _coerce_float(raw_repetition_penalty, 1.0))
        enable_rollout_prefix_cache = bool(self._enable_shared_rollout_prefix_cache)
        rollout_prefix_cache_min_tokens = (
            max(
                1,
                int(self.config.get("rollout_prefix_cache_min_tokens", 256) or 256),
            )
            if enable_rollout_prefix_cache
            else 0
        )
        generation_session_id = str(kwargs.get("generation_session_id") or "").strip()
        rollout_prefix_cache_key = (
            self._build_rollout_prefix_cache_key(
                generation_session_id,
                normalized_role,
            )
            if enable_rollout_prefix_cache
            else ""
        )
        _diag_log(
            "[RolloutGenerate][DIAG] role=%s token_budget=%d remaining_budget=%d decode_mode=%s stop_token_ids=%s top_k=%s top_p=%s min_p=%s repetition_penalty=%s disable_gradient_checkpointing=%s force_use_cache=%s prefix_cache_enabled=%s prefix_cache_min_tokens=%d session_id=%s",
            normalized_role or "llm",
            token_budget,
            remaining_budget,
            decode_mode,
            list(stop_token_ids),
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            disable_gradient_checkpointing_during_rollout,
            force_use_cache_during_rollout,
            enable_rollout_prefix_cache,
            rollout_prefix_cache_min_tokens,
            generation_session_id or "<none>",
        )
        def _raise_generation_timeout() -> None:
            elapsed_seconds = time.perf_counter() - generation_started
            partial_text = _decode_text(self.tokenizer, generated_ids)
            role_text = normalized_role or "llm"
            raise GenerationTimeoutError(
                f"{role_text} generation exceeded per-turn timeout after {elapsed_seconds:.1f}s.",
                partial_response=partial_text,
                elapsed_seconds=elapsed_seconds,
                role=role_text,
                phase="rollout_generate",
            )

        self.actor_module.eval()
        stop_reason = "token_budget_exhausted"
        cache_supported = use_incremental_kv_cache
        past_key_values = None
        fallback_to_model_generate = False
        prompt_ready_logits: Optional[torch.Tensor] = None
        prefix_cache_candidate_input_ids: Optional[torch.Tensor] = None
        prefix_cache_candidate_past_key_values: Any = None
        prefix_cache_candidate_logits: Optional[torch.Tensor] = None
        with _temporary_rollout_inference_mode(
            self.actor_module,
            disable_gradient_checkpointing=disable_gradient_checkpointing_during_rollout,
            force_use_cache=force_use_cache_during_rollout,
        ) as inference_mode_state, torch.no_grad():
            _diag_log(
                "[RolloutGenerate][DIAG] role=%s inference_mode candidate_modules=%d cache_flags_forced=%d gradient_checkpointing_modules_disabled=%d skipped_reason=%s",
                normalized_role or "llm",
                int(inference_mode_state.get("candidate_module_count", 0) or 0),
                int(inference_mode_state.get("cache_flags_forced", 0) or 0),
                int(inference_mode_state.get("gradient_checkpointing_modules_disabled", 0) or 0),
                str(inference_mode_state.get("skipped_reason") or "none"),
            )
            prefix_cache_hit = False
            prefix_cache_source = "disabled"
            prefix_cache_prefix_tokens = 0
            prefix_cache_delta_tokens = 0
            if use_incremental_kv_cache and enable_rollout_prefix_cache and rollout_prefix_cache_key:
                cached_prefix_state = self._shared_rollout_prefix_cache.pop(
                    rollout_prefix_cache_key,
                    None,
                )
                if cached_prefix_state is None:
                    prefix_cache_source = "miss"
                else:
                    cached_input_ids = [
                        int(token_id)
                        for token_id in list(cached_prefix_state.get("input_ids", []) or [])
                    ]
                    cached_past_key_values = cached_prefix_state.get("past_key_values")
                    cached_next_token_logits = cached_prefix_state.get("next_token_logits")
                    cached_visual_signature = str(cached_prefix_state.get("visual_signature") or "")
                    current_visual_signature = str(prepared.get("visual_signature") or "")
                    prefix_cache_prefix_tokens = _longest_common_prefix_length(
                        cached_input_ids,
                        prepared["prompt_ids"],
                    )
                    prefix_cache_delta_tokens = max(
                        0,
                        len(prepared["prompt_ids"]) - prefix_cache_prefix_tokens,
                    )
                    if cached_visual_signature != current_visual_signature:
                        prefix_cache_source = "visual_mismatch"
                    elif cached_past_key_values is None:
                        prefix_cache_source = "missing_past_key_values"
                    elif prefix_cache_prefix_tokens != len(cached_input_ids):
                        prefix_cache_source = (
                            f"prefix_mismatch:{prefix_cache_prefix_tokens}/{len(cached_input_ids)}"
                        )
                    elif prefix_cache_prefix_tokens < rollout_prefix_cache_min_tokens:
                        prefix_cache_source = f"prefix_too_short:{prefix_cache_prefix_tokens}"
                    elif prefix_cache_delta_tokens == 0:
                        if torch.is_tensor(cached_next_token_logits):
                            prompt_ready_logits = cached_next_token_logits
                            past_key_values = cached_past_key_values
                            prefix_cache_candidate_input_ids = current_input_ids.detach().clone()
                            prefix_cache_candidate_past_key_values = cached_past_key_values
                            prefix_cache_candidate_logits = cached_next_token_logits
                            prefix_cache_hit = True
                            prefix_cache_source = "exact"
                        else:
                            prefix_cache_source = "missing_next_token_logits"
                    else:
                        delta_input_ids = current_input_ids[:, prefix_cache_prefix_tokens:]
                        try:
                            _, delta_logits, delta_past_key_values = self._actor_forward_with_cache(
                                input_ids=delta_input_ids,
                                attention_mask=current_attention_mask,
                                device=device,
                                forward_autocast_dtype=forward_autocast_dtype,
                                use_cache=True,
                                past_key_values=cached_past_key_values,
                            )
                        except Exception:
                            prefix_cache_source = "delta_prefill_error"
                            logger.exception(
                                "[RolloutGenerate][DIAG] role=%s failed to prefill prompt delta from prefix cache prefix_tokens=%d delta_tokens=%d",
                                normalized_role or "llm",
                                prefix_cache_prefix_tokens,
                                prefix_cache_delta_tokens,
                            )
                        else:
                            if delta_past_key_values is None:
                                prefix_cache_source = "delta_missing_past_key_values"
                            else:
                                prompt_ready_logits = delta_logits[:, -1, :]
                                past_key_values = delta_past_key_values
                                prefix_cache_candidate_input_ids = current_input_ids.detach().clone()
                                prefix_cache_candidate_past_key_values = delta_past_key_values
                                prefix_cache_candidate_logits = prompt_ready_logits
                                prefix_cache_hit = True
                                prefix_cache_source = "delta"
                                decode_mode = f"{decode_mode}+prefix_reuse"
            _diag_log(
                "[RolloutGenerate][DIAG] role=%s prefix_cache source=%s hit=%s prefix_tokens=%d delta_tokens=%d",
                normalized_role or "llm",
                prefix_cache_source,
                prefix_cache_hit,
                prefix_cache_prefix_tokens,
                prefix_cache_delta_tokens,
            )
            if use_incremental_kv_cache:
                for step_idx in range(max(1, token_budget)):
                    if generation_deadline is not None and time.perf_counter() >= generation_deadline:
                        _raise_generation_timeout()
                    if wall_clock_deadline is not None and time.perf_counter() >= wall_clock_deadline:
                        logger.warning(
                            "Stopping rollout generation early because the wall-clock deadline was reached."
                        )
                        stop_reason = "wall_clock_deadline"
                        break

                    if step_idx == 0:
                        _emit_generation_progress(
                            source="before_first_forward",
                            result_preview=(
                                (
                                    f"reusing prefix cache source={prefix_cache_source} "
                                    f"prefix_tokens={prefix_cache_prefix_tokens} "
                                    f"delta_tokens={prefix_cache_delta_tokens}"
                                )
                                if prompt_ready_logits is not None
                                else (
                                    f"starting first actor forward prompt_tokens={prompt_length} "
                                    f"image_count={int(prepared.get('image_count', 0) or 0)} "
                                    f"video_count={int(prepared.get('video_count', 0) or 0)}"
                                )
                            ),
                        )
                        _diag_log(
                            "[RolloutGenerate][DIAG] role=%s first_forward input_ids_shape=%s attention_mask_shape=%s prefix_cache_source=%s",
                            normalized_role or "llm",
                            tuple(int(dim) for dim in current_input_ids.shape),
                            tuple(int(dim) for dim in current_attention_mask.shape),
                            prefix_cache_source,
                        )
                    output = None
                    if step_idx == 0 and prompt_ready_logits is not None:
                        logits = prompt_ready_logits.unsqueeze(1)
                    else:
                        actor_input_ids = (
                            current_input_ids[:, -1:]
                            if cache_supported and past_key_values is not None
                            else current_input_ids
                        )
                        actor_static_kwargs = (
                            None
                            if cache_supported and past_key_values is not None
                            else static_kwargs
                        )
                        output, logits, next_past_key_values = self._actor_forward_with_cache(
                            input_ids=actor_input_ids,
                            attention_mask=current_attention_mask,
                            device=device,
                            forward_autocast_dtype=forward_autocast_dtype,
                            use_cache=bool(cache_supported),
                            static_kwargs=actor_static_kwargs,
                            past_key_values=past_key_values,
                        )
                        if cache_supported:
                            if next_past_key_values is None:
                                cache_supported = False
                                if step_idx == 0:
                                    if normalized_role == "planner":
                                        decode_mode = "hf_generate_full_sequence_fallback"
                                        fallback_to_model_generate = True
                                        logger.warning(
                                            "Planner rollout did not return past_key_values on the first forward; "
                                            "falling back to model.generate for this turn. output_type=%s",
                                            type(output).__name__ if output is not None else "<none>",
                                        )
                                        break
                                    decode_mode = "full_sequence_fallback"
                                    logger.warning(
                                        "Text-only subagent rollout did not return past_key_values on the first forward; "
                                        "falling back to full-sequence decode."
                                    )
                            else:
                                past_key_values = next_past_key_values
                                prefix_cache_candidate_input_ids = current_input_ids.detach().clone()
                                prefix_cache_candidate_past_key_values = next_past_key_values
                                prefix_cache_candidate_logits = logits[:, -1, :]
                    if step_idx == 0:
                        _emit_generation_progress(
                            source="after_first_forward",
                            result_preview=(
                                f"first actor forward returned logits decode_mode={decode_mode}"
                                if prompt_ready_logits is None
                                else (
                                    f"prefix cache ready source={prefix_cache_source} "
                                    f"decode_mode={decode_mode}"
                                )
                            ),
                        )
                        _diag_log(
                            "[RolloutGenerate][DIAG] role=%s first_forward logits_shape=%s decode_mode=%s prefix_cache_source=%s",
                            normalized_role or "llm",
                            tuple(int(dim) for dim in logits.shape),
                            decode_mode,
                            prefix_cache_source,
                        )

                    next_token = self._sample_next_token(
                        logits[:, -1, :],
                        input_ids=current_input_ids,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                    )
                    token_id = int(next_token[0, 0].item())
                    generated_ids.append(token_id)
                    if step_idx == 0:
                        _emit_generation_progress(
                            source="first_token",
                            result_preview=(
                                f"first token sampled token_id={token_id} "
                                f"generated_tokens={len(generated_ids)}"
                            ),
                        )
                        _diag_log(
                            "[RolloutGenerate][DIAG] role=%s first_token token_id=%d generated_tokens=%d",
                            normalized_role or "llm",
                            token_id,
                            len(generated_ids),
                        )
                    current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                    current_attention_mask = torch.cat(
                        [
                            current_attention_mask,
                            torch.ones(
                                (current_attention_mask.shape[0], 1),
                                dtype=current_attention_mask.dtype,
                                device=current_attention_mask.device,
                            ),
                        ],
                        dim=-1,
                    )
                    if time.perf_counter() >= next_heartbeat_monotonic:
                        _emit_generation_progress(
                            source="decode_loop",
                            result_preview=(
                                f"generated_tokens={len(generated_ids)} "
                                f"prompt_tokens={prompt_length}"
                            ),
                        )
                        next_heartbeat_monotonic = time.perf_counter() + heartbeat_interval_seconds

                    if stop_token_id_set and token_id in stop_token_id_set:
                        stop_reason = f"stop_token:{token_id}"
                        break
                    if (
                        generated_ids
                        and ((step_idx + 1) % structured_stop_check_interval == 0 or step_idx + 1 >= token_budget)
                    ):
                        decoded_generation = _decode_text(self.tokenizer, generated_ids)
                        truncated_generation, response_was_truncated = _truncate_response_to_structured_payload(
                            decoded_generation
                        )
                        if _response_has_complete_structured_payload(truncated_generation):
                            if response_was_truncated:
                                generated_ids = self.tokenizer.encode(
                                    truncated_generation,
                                    add_special_tokens=False,
                                )
                            _diag_log(
                                "Stopping rollout generation after the first complete structured response. role=%s generated_tokens=%d truncated=%s",
                                normalized_role or "llm",
                                len(generated_ids),
                                response_was_truncated,
                            )
                            stop_reason = (
                                "structured_payload_truncated"
                                if response_was_truncated
                                else "structured_payload_complete"
                            )
                            break
                    if generation_deadline is not None and time.perf_counter() >= generation_deadline:
                        _raise_generation_timeout()
            if (not use_incremental_kv_cache) or fallback_to_model_generate:
                effective_deadline = None
                for raw_deadline in (wall_clock_deadline, generation_deadline):
                    if raw_deadline is None:
                        continue
                    if effective_deadline is None or raw_deadline < effective_deadline:
                        effective_deadline = raw_deadline
                remaining_seconds = (
                    max(0.0, effective_deadline - time.perf_counter())
                    if effective_deadline is not None
                    else None
                )
                generation_kwargs: Dict[str, Any] = {
                    "max_new_tokens": int(token_budget),
                    "pad_token_id": int(self.pad_token_id),
                    "use_cache": True,
                    "remove_invalid_values": True,
                    "renormalize_logits": True,
                    # Each DP rank drives an independent episode/tool loop during rollout;
                    # forcing generate to synchronize rank completion would make shorter
                    # episodes wait for unrelated long-tail planner/subagent turns.
                    "synced_gpus": False,
                }
                if (
                    bool(self.config.get("enable_rollout_logits_nan_guard", True))
                    and not bool(generation_kwargs.get("remove_invalid_values", False))
                    and InfNanRemoveLogitsProcessor is not None
                    and LogitsProcessorList is not None
                ):
                    generation_kwargs["logits_processor"] = LogitsProcessorList(
                        [InfNanRemoveLogitsProcessor()]
                    )
                if do_sample:
                    generation_kwargs["do_sample"] = True
                    if temperature is not None:
                        generation_kwargs["temperature"] = float(temperature)
                    generation_kwargs["top_p"] = float(top_p)
                    generation_kwargs["top_k"] = max(0, int(top_k))
                    generation_kwargs["min_p"] = float(min_p)
                else:
                    generation_kwargs["do_sample"] = False
                if repetition_penalty is not None:
                    generation_kwargs["repetition_penalty"] = float(repetition_penalty)

                def _build_full_sequence_generation_kwargs(*, use_cache: bool) -> Dict[str, Any]:
                    call_generation_kwargs = dict(generation_kwargs)
                    call_generation_kwargs["use_cache"] = bool(use_cache)
                    if stop_token_ids:
                        call_generation_kwargs["eos_token_id"] = (
                            list(stop_token_ids) if len(stop_token_ids) > 1 else int(stop_token_ids[0])
                        )
                    if remaining_seconds is not None and remaining_seconds > 0:
                        call_generation_kwargs["max_time"] = float(remaining_seconds)
                    call_generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [
                            _StructuredPayloadStoppingCriteria(
                                tokenizer=self.tokenizer,
                                prompt_length=prompt_length,
                                check_interval_tokens=structured_stop_check_interval,
                            )
                        ]
                    )
                    return self._filter_supported_generate_kwargs(call_generation_kwargs)

                prefer_safe_multimodal_no_cache = bool(
                    self.actor_module_fsdp is not None and (image_count > 0 or video_count > 0)
                )
                model_generate_use_cache = not prefer_safe_multimodal_no_cache
                model_generate_slow_path_reason = (
                    "multimodal_fsdp_use_cache_disabled" if prefer_safe_multimodal_no_cache else ""
                )
                if prefer_safe_multimodal_no_cache:
                    logger.info(
                        "[RolloutGenerate][DIAG] role=%s using use_cache=False for multimodal FSDP model.generate to avoid known CUDA illegal-memory-access failure modes.",
                        normalized_role or "llm",
                    )
                generation_kwargs = _build_full_sequence_generation_kwargs(
                    use_cache=model_generate_use_cache
                )
                logged_generation_kwargs = dict(generation_kwargs)
                if "stopping_criteria" in logged_generation_kwargs:
                    logged_generation_kwargs["stopping_criteria"] = {
                        "type": type(logged_generation_kwargs["stopping_criteria"]).__name__,
                        "check_interval_tokens": structured_stop_check_interval,
                    }
                if "logits_processor" in logged_generation_kwargs:
                    processor_list = logged_generation_kwargs["logits_processor"]
                    try:
                        processor_names = [type(item).__name__ for item in list(processor_list)]
                    except Exception:
                        processor_names = [type(processor_list).__name__]
                    logged_generation_kwargs["logits_processor"] = {
                        "type": type(processor_list).__name__,
                        "processors": processor_names,
                    }
                _diag_log(
                    "[RolloutGenerate][DIAG] role=%s switching to model.generate for multimodal/full-sequence parity generation_kwargs=%s",
                    normalized_role or "llm",
                    json.dumps(logged_generation_kwargs, ensure_ascii=False),
                )
                _emit_generation_progress(
                    source="before_model_generate",
                    result_preview=(
                        f"starting model.generate prompt_tokens={prompt_length} "
                        f"image_count={image_count} "
                        f"video_count={video_count} "
                        f"decode_mode={decode_mode} "
                        f"token_budget={token_budget} "
                        f"use_cache={str(bool(model_generate_use_cache)).lower()}"
                        + (
                            f" slow_path_reason={model_generate_slow_path_reason}"
                            if model_generate_slow_path_reason
                            else ""
                        )
                    ),
                )
                full_generation_inputs = dict(model_inputs or {})
                optional_auxiliary_generation_input_keys = sorted(
                    key
                    for key in full_generation_inputs
                    if key in _QUIETLY_DROPPED_PROMPT_ALIGNED_AUXILIARY_TENSOR_KEYS
                )
                fallback_generation_inputs = {
                    key: value
                    for key, value in full_generation_inputs.items()
                    if key not in _QUIETLY_DROPPED_PROMPT_ALIGNED_AUXILIARY_TENSOR_KEYS
                }
                model_generate_inputs = full_generation_inputs
                if optional_auxiliary_generation_input_keys:
                    model_generate_inputs = fallback_generation_inputs
                    _diag_log(
                        "[RolloutGenerate][DIAG] role=%s dropping optional prompt-aligned tensor(s) before model.generate=%s",
                        normalized_role or "llm",
                        optional_auxiliary_generation_input_keys,
                    )

                def _generate_once(
                    call_generation_kwargs: Dict[str, Any],
                    *,
                    generation_inputs: Dict[str, Any],
                ) -> Any:
                    autocast_context = (
                        torch.autocast(device_type=device.type, dtype=forward_autocast_dtype)
                        if device.type == "cuda" and forward_autocast_dtype in (torch.float16, torch.bfloat16)
                        else nullcontext()
                    )
                    with autocast_context:
                        if self.actor_module_fsdp is not None and not bool(
                            self._fsdp_full_params_local_shard_active
                        ):
                            original_forward = self.actor_module.forward
                            forward_dispatch_state = {"in_fsdp_dispatch": False}

                            def _forward_via_fsdp(*args: Any, **kwargs: Any) -> Any:
                                if bool(forward_dispatch_state.get("in_fsdp_dispatch", False)):
                                    return original_forward(*args, **kwargs)
                                forward_dispatch_state["in_fsdp_dispatch"] = True
                                try:
                                    return self.actor_module_fsdp(*args, **kwargs)
                                finally:
                                    forward_dispatch_state["in_fsdp_dispatch"] = False

                            # Keep GenerationMixin on the wrapped model class so
                            # mode-specific helpers like `_sample` remain available,
                            # but route each forward through FSDP wrapper lifecycle.
                            self.actor_module.forward = _forward_via_fsdp
                            try:
                                return self.actor_module.generate(**generation_inputs, **call_generation_kwargs)
                            finally:
                                self.actor_module.forward = original_forward
                        return self.actor_module.generate(**generation_inputs, **call_generation_kwargs)

                def _emit_model_generate_wait_progress(
                    *,
                    elapsed_wait: float,
                    fsdp_lock_waiting: bool,
                    distributed_admission_waiting: bool,
                ) -> None:
                    _emit_generation_progress(
                        source="model_generate_wait",
                        result_preview=(
                            f"blocking model.generate still running "
                            f"elapsed_seconds={elapsed_wait:.1f} "
                            f"prompt_tokens={prompt_length} "
                            f"image_count={image_count} "
                            f"video_count={video_count} "
                            f"decode_mode={decode_mode} "
                            f"token_budget={token_budget} "
                            f"use_cache={str(bool(model_generate_use_cache)).lower()} "
                            f"fsdp_lock_waiting={str(bool(fsdp_lock_waiting)).lower()} "
                            f"distributed_admission_waiting={str(bool(distributed_admission_waiting)).lower()}"
                            + (
                                f" slow_path_reason={model_generate_slow_path_reason}"
                                if model_generate_slow_path_reason
                                else ""
                            )
                        ),
                    )
                    _diag_log(
                        "[RolloutGenerate][DIAG] role=%s blocking model.generate still running elapsed=%.3fs prompt_tokens=%d image_count=%d video_count=%d fsdp_lock_waiting=%s distributed_admission_waiting=%s",
                        normalized_role or "llm",
                        elapsed_wait,
                        prompt_length,
                        image_count,
                        video_count,
                        bool(fsdp_lock_waiting),
                        bool(distributed_admission_waiting),
                    )

                def _run_blocking_generate_with_heartbeat(
                    call_generation_kwargs: Dict[str, Any],
                    *,
                    generation_inputs: Dict[str, Any],
                    extra_memory_margin_bytes: int = 0,
                ) -> Any:
                    if self.actor_module_fsdp is not None:
                        if self._fsdp_full_params_local_shard_active:
                            _diag_log(
                                "[RolloutGenerate][DIAG] role=%s running model.generate on full-params local-shard path without per-forward FSDP dispatch.",
                                normalized_role or "llm",
                            )
                        else:
                            _diag_log(
                                "[RolloutGenerate][DIAG] role=%s running model.generate on FSDP owner-thread path via temporary forward->FSDP dispatch.",
                                normalized_role or "llm",
                            )
                        heartbeat_stop_event = threading.Event()
                        lock_state = {"acquired": False, "admission_ready": False}

                        def _heartbeat_target() -> None:
                            while not heartbeat_stop_event.wait(timeout=heartbeat_interval_seconds):
                                elapsed_wait = time.perf_counter() - generation_started
                                _emit_model_generate_wait_progress(
                                    elapsed_wait=elapsed_wait,
                                    fsdp_lock_waiting=not bool(lock_state.get("acquired", False)),
                                    distributed_admission_waiting=bool(lock_state.get("acquired", False))
                                    and not bool(lock_state.get("admission_ready", False)),
                                )

                        heartbeat_thread = threading.Thread(
                            target=_heartbeat_target,
                            name=f"rollout-generate-heartbeat-{normalized_role or 'llm'}",
                            daemon=True,
                        )
                        heartbeat_thread.start()
                        try:
                            with self._actor_fsdp_generate_lock:
                                lock_state["acquired"] = True
                                self._wait_for_generate_memory_admission(
                                    device=device,
                                    role=normalized_role or "llm",
                                    prompt_length=prompt_length,
                                    image_count=image_count,
                                    video_count=video_count,
                                    decode_mode=decode_mode,
                                    generation_deadline=generation_deadline,
                                    wall_clock_deadline=wall_clock_deadline,
                                    extra_margin_bytes=extra_memory_margin_bytes,
                                )
                                # Local FSDP shards can be serving different samples at the
                                # same time, so a per-generate cross-rank admission barrier
                                # can deadlock at ready=1/world_size.
                                lock_state["admission_ready"] = True
                                return _generate_once(
                                    call_generation_kwargs,
                                    generation_inputs=generation_inputs,
                                )
                        finally:
                            heartbeat_stop_event.set()
                            heartbeat_thread.join(timeout=1.0)

                    generate_state: Dict[str, Any] = {"output": None, "exc": None}

                    def _target() -> None:
                        try:
                            generate_state["output"] = _generate_once(
                                call_generation_kwargs,
                                generation_inputs=generation_inputs,
                            )
                        except Exception as exc:
                            generate_state["exc"] = exc

                    generate_thread = threading.Thread(
                        target=_target,
                        name=f"rollout-generate-{normalized_role or 'llm'}",
                        daemon=True,
                    )
                    generate_thread.start()
                    while generate_thread.is_alive():
                        generate_thread.join(timeout=heartbeat_interval_seconds)
                        if not generate_thread.is_alive():
                            break
                        elapsed_wait = time.perf_counter() - generation_started
                        _emit_model_generate_wait_progress(
                            elapsed_wait=elapsed_wait,
                            fsdp_lock_waiting=False,
                            distributed_admission_waiting=False,
                        )
                    if generate_state["exc"] is not None:
                        raise generate_state["exc"]
                    return generate_state["output"]

                def _run_model_generate(
                    call_generation_kwargs: Dict[str, Any],
                    *,
                    extra_memory_margin_bytes: int = 0,
                ) -> Any:
                    return _run_blocking_generate_with_heartbeat(
                        call_generation_kwargs,
                        generation_inputs=model_generate_inputs,
                        extra_memory_margin_bytes=extra_memory_margin_bytes,
                    )

                def _run_model_generate_with_memory_retry(
                    call_generation_kwargs: Dict[str, Any],
                ) -> Any:
                    max_retries = max(
                        0,
                        int(self.config.get("generate_oom_retry_count", 1) or 1),
                    )
                    retry_extra_margin_bytes = max(
                        0,
                        int(
                            _coerce_float(
                                self.config.get("generate_oom_retry_extra_margin_gb", 1.0),
                                1.0,
                            )
                            * (1024**3)
                        ),
                    )
                    extra_memory_margin_bytes = 0
                    for attempt_idx in range(max_retries + 1):
                        try:
                            return _run_model_generate(
                                call_generation_kwargs,
                                extra_memory_margin_bytes=extra_memory_margin_bytes,
                            )
                        except Exception as exc:
                            if (not _is_out_of_memory_error(exc)) or attempt_idx >= max_retries:
                                raise
                            logger.warning(
                                "[RolloutGenerate][OOMRetry][DIAG] role=%s attempt=%d/%d prompt_tokens=%d image_count=%d video_count=%d decode_mode=%s error=%s",
                                normalized_role or "llm",
                                attempt_idx + 1,
                                max_retries + 1,
                                prompt_length,
                                image_count,
                                video_count,
                                decode_mode,
                                exc,
                            )
                            aggressive_empty_cache(force_sync=True)
                            extra_memory_margin_bytes += retry_extra_margin_bytes
                    raise RuntimeError("Unexpected OOM retry loop termination without result.")

                try:
                    output = _run_model_generate_with_memory_retry(generation_kwargs)
                    self._synchronize_after_rollout_generate(
                        device=device,
                        role=normalized_role or "llm",
                    )
                except StopIteration as exc:
                    if image_count <= 0 and video_count <= 0:
                        raise RuntimeError("Actor model.generate raised StopIteration during rollout generation.") from exc
                    retry_generation_kwargs = _build_full_sequence_generation_kwargs(use_cache=False)
                    logger.warning(
                        "Rollout model.generate raised StopIteration for multimodal prompt; retrying once with use_cache=False."
                    )
                    output = _run_model_generate_with_memory_retry(retry_generation_kwargs)
                    self._synchronize_after_rollout_generate(
                        device=device,
                        role=normalized_role or "llm",
                    )
                    generation_kwargs = retry_generation_kwargs
                except Exception as exc:
                    if isinstance(exc, AcceleratorFatalError) or is_accelerator_fatal_error(exc):
                        self._fatal_accelerator_error = str(exc)
                        logger.exception(
                            "[RolloutGenerate][FATAL] role=%s model.generate raised a fatal accelerator error; marking this rollout worker poisoned and re-raising.",
                            normalized_role or "llm",
                        )
                        raise AcceleratorFatalError(str(exc)) from exc
                    raise
                model_generate_finished_monotonic = time.perf_counter()
                deadline_exhausted = bool(
                    effective_deadline is not None
                    and model_generate_finished_monotonic >= (float(effective_deadline) - 0.05)
                )
                sequences = getattr(output, "sequences", output)
                if not torch.is_tensor(sequences):
                    raise RuntimeError("Actor model.generate returned no tensor sequences during rollout generation.")
                if sequences.ndim == 1:
                    sequences = sequences.unsqueeze(0)
                generated_ids = [int(token_id) for token_id in sequences[0, prompt_length:].tolist()]
                generated_text = _decode_text(self.tokenizer, generated_ids)
                truncated_text, response_was_truncated = _truncate_response_to_structured_payload(generated_text)
                if _response_has_complete_structured_payload(truncated_text):
                    if response_was_truncated:
                        generated_ids = self.tokenizer.encode(truncated_text, add_special_tokens=False)
                    stop_reason = (
                        "structured_payload_truncated"
                        if response_was_truncated
                        else "structured_payload_complete"
                    )
                else:
                    stop_reason = "model_generate_timeout" if deadline_exhausted else "model_generate"
                _diag_log(
                    "[RolloutGenerate][DIAG] role=%s model.generate exit deadline_exhausted=%s stop_reason=%s",
                    normalized_role or "llm",
                    deadline_exhausted,
                    stop_reason,
                )
                _emit_generation_progress(
                    source="after_model_generate",
                    result_preview=(
                        f"generated_tokens={len(generated_ids)} "
                        f"prompt_tokens={prompt_length} "
                        f"use_cache={bool(generation_kwargs.get('use_cache', False))}"
                    ),
                )

        finalized_text = ""
        final_was_truncated = False
        if generated_ids:
            finalized_text = _decode_text(self.tokenizer, generated_ids)
            finalized_truncated_text, final_was_truncated = _truncate_response_to_structured_payload(finalized_text)
            if final_was_truncated:
                generated_ids = self.tokenizer.encode(finalized_truncated_text, add_special_tokens=False)
                finalized_text = finalized_truncated_text
                if stop_reason not in {"structured_payload_complete", "structured_payload_truncated"}:
                    stop_reason = "structured_payload_truncated"
        if len(generated_ids) >= int(token_budget) and stop_reason not in {
            "structured_payload_complete",
            "structured_payload_truncated",
        }:
            logger.warning(
                "[RolloutGenerate][TRUNCATION] role=%s generated_tokens=%d reached_token_budget=%d stop_reason=%s "
                "structured_payload_complete=%s response_preview=%s",
                normalized_role or "llm",
                len(generated_ids),
                int(token_budget),
                stop_reason,
                _response_has_complete_structured_payload(finalized_text),
                _truncate_text(finalized_text, 240),
            )
        if (
            use_incremental_kv_cache
            and enable_rollout_prefix_cache
            and rollout_prefix_cache_key
            and not fallback_to_model_generate
            and cache_supported
            and prefix_cache_candidate_input_ids is not None
            and prefix_cache_candidate_past_key_values is not None
            and prefix_cache_candidate_logits is not None
        ):
            normalized_generated_ids = (
                self.tokenizer.encode(finalized_text, add_special_tokens=False)
                if generated_ids
                else []
            )
            expected_cached_ids = list(prepared["prompt_ids"]) + list(generated_ids)
            cached_input_ids_to_store: Optional[List[int]] = None
            cached_past_key_values_to_store = prefix_cache_candidate_past_key_values
            cached_next_token_logits_to_store: Optional[torch.Tensor] = prefix_cache_candidate_logits
            if generated_ids and normalized_generated_ids != generated_ids:
                _diag_log(
                    "[RolloutGenerate][DIAG] role=%s skipping prefix cache save because normalized transcript ids mismatch generated ids normalized=%d generated=%d",
                    normalized_role or "llm",
                    len(normalized_generated_ids),
                    len(generated_ids),
                )
            elif final_was_truncated:
                _diag_log(
                    "[RolloutGenerate][DIAG] role=%s skipping prefix cache save because the finalized transcript was truncated.",
                    normalized_role or "llm",
                )
            else:
                candidate_length = int(prefix_cache_candidate_input_ids.shape[-1])
                if candidate_length == len(expected_cached_ids):
                    cached_input_ids_to_store = expected_cached_ids
                elif generated_ids and candidate_length + 1 == len(expected_cached_ids):
                    commit_attention_mask = torch.ones(
                        (1, len(expected_cached_ids)),
                        dtype=current_attention_mask.dtype,
                        device=device,
                    )
                    try:
                        _, commit_logits, commit_past_key_values = self._actor_forward_with_cache(
                            input_ids=torch.tensor(
                                [[int(generated_ids[-1])]],
                                dtype=torch.long,
                                device=device,
                            ),
                            attention_mask=commit_attention_mask,
                            device=device,
                            forward_autocast_dtype=forward_autocast_dtype,
                            use_cache=True,
                            past_key_values=prefix_cache_candidate_past_key_values,
                        )
                    except Exception:
                        logger.exception(
                            "[RolloutGenerate][DIAG] role=%s failed to commit final transcript token into prefix cache.",
                            normalized_role or "llm",
                        )
                    else:
                        if commit_past_key_values is not None:
                            cached_input_ids_to_store = expected_cached_ids
                            cached_past_key_values_to_store = commit_past_key_values
                            cached_next_token_logits_to_store = commit_logits[:, -1, :]
                else:
                    _diag_log(
                        "[RolloutGenerate][DIAG] role=%s skipping prefix cache save because candidate_length=%d expected_length=%d",
                        normalized_role or "llm",
                        candidate_length,
                        len(expected_cached_ids),
                    )

            if (
                cached_input_ids_to_store
                and cached_next_token_logits_to_store is not None
            ):
                self._set_lru_cache_entry(
                    self._shared_rollout_prefix_cache,
                    rollout_prefix_cache_key,
                    {
                        "input_ids": list(cached_input_ids_to_store),
                        "past_key_values": cached_past_key_values_to_store,
                        "next_token_logits": cached_next_token_logits_to_store.detach().clone(),
                        "visual_signature": str(prepared.get("visual_signature") or ""),
                        "image_count": image_count,
                        "video_count": video_count,
                    },
                    max_entries=self._shared_rollout_prefix_cache_max_entries,
                )
                _diag_log(
                    "[RolloutGenerate][DIAG] role=%s saved_prefix_cache tokens=%d generated_tokens=%d session_id=%s",
                    normalized_role or "llm",
                    len(cached_input_ids_to_store),
                    len(generated_ids),
                    generation_session_id or "<none>",
                )

        logger.info(
            "[RolloutGenerate][DIAG] role=%s completed duration=%.3fs generated_tokens=%d stop_reason=%s",
            normalized_role or "llm",
            time.perf_counter() - generation_started,
            len(generated_ids),
            stop_reason,
        )
        return {
            "prompt_ids": prepared["prompt_ids"],
            "response_ids": generated_ids,
            "prompt_preview": prepared["prompt_preview"],
            "prompt_messages": prepared["prompt_messages"],
            "stop_reason": stop_reason,
            "timed_out": bool(stop_reason == "model_generate_timeout"),
        }

    def _fallback_prompt_ids(self, sample: Dict[str, Any]) -> List[int]:
        text = str(sample.get("prompt") or sample.get("question") or "").strip()
        if not text:
            text = "Question unavailable."
        return self.tokenizer.encode(text, add_special_tokens=False) or [
            int(getattr(self.tokenizer, "eos_token_id", 0) or 0)
        ]

    def _build_soft_failed_episode_artifacts(
        self,
        *,
        sample: Dict[str, Any],
        error_message: str,
    ) -> EpisodeArtifacts:
        eos_token_id = int(getattr(self.tokenizer, "eos_token_id", None) or getattr(self.tokenizer, "pad_token_id", 0) or 0)
        prompt_ids = list(self._fallback_prompt_ids(sample))
        normalized_error = str(error_message or "").strip() or "episode_failed"
        reward_features = {
            "sample_id": str(sample.get("sample_id") or "").strip(),
            "dataset_name": str(sample.get("dataset_name") or sample.get("data_source") or "").strip(),
            "generation_call_count": 0,
            "planner_generation_turns": 0,
            "subagent_generation_turns": 0,
            "prompt_token_count": len(prompt_ids),
            "response_token_count": 0,
            "episode_soft_failed": True,
            "failure_reason": normalized_error,
        }
        run_result = {
            "video_path": str(sample.get("video_path") or ""),
            "query": str(sample.get("prompt") or sample.get("question") or ""),
            "workspace": "",
            "success": False,
            "error": normalized_error,
            "agent": {
                "final_answer": normalized_error,
                "trajectory": {
                    "success": False,
                    "termination_reason": "cooperative_scheduler_soft_fail",
                    "error_message": normalized_error,
                },
            },
            "metrics": {
                "planner_iterations": 0,
                "subagent_iterations": 0,
                "delegated_root_steps": 0,
                "decomposed_root_steps": 0,
                "parallel_concrete_batches": 0,
                "parallel_concrete_batch_rate": 0.0,
                "cache_stats": {},
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
            final_answer=normalized_error,
            success=False,
            trajectory={
                "success": False,
                "termination_reason": "cooperative_scheduler_soft_fail",
                "error_message": normalized_error,
            },
            run_result=run_result,
            reward_features=reward_features,
            conversation_messages=[],
            generation_records=[],
        )

    def _synchronize_after_rollout_generate(self, *, device: torch.device, role: str) -> None:
        if device.type != "cuda":
            return
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:
            if isinstance(exc, AcceleratorFatalError) or is_accelerator_fatal_error(exc):
                self._fatal_accelerator_error = str(exc)
                logger.exception(
                    "[RolloutGenerate][FATAL] role=%s CUDA synchronize after model.generate surfaced a fatal accelerator error; marking this rollout worker poisoned.",
                    role or "llm",
                )
                raise AcceleratorFatalError(str(exc)) from exc
            raise

    def _episode_to_row(
        self,
        *,
        sample: Dict[str, Any],
        artifacts: Any,
        device: torch.device,
        attention_dtype: torch.dtype,
        eos_token_id: int,
        rollout_policy_version: int,
    ) -> Dict[str, Any]:
        output_device = torch.device("cpu")
        prompt_ids = list(artifacts.prompt_ids or self._fallback_prompt_ids(sample))
        response_ids = list(artifacts.response_ids)
        loss_mask = list(artifacts.loss_mask)

        reward_features = deepcopy(artifacts.reward_features)
        reward_features.setdefault("prompt_truncated", False)
        reward_features.setdefault("response_truncated", False)

        if len(response_ids) != len(loss_mask):
            min_length = min(len(response_ids), len(loss_mask))
            response_ids = response_ids[:min_length]
            loss_mask = loss_mask[:min_length]

        if not response_ids:
            eos_fallback = eos_token_id if eos_token_id is not None else int(self.pad_token_id or 0)
            response_ids = [int(eos_fallback)]
            loss_mask = [1]

        original_prompt_length = len(prompt_ids)
        original_response_length = len(response_ids)

        if len(prompt_ids) > int(self.config.prompt_length):
            prompt_limit = int(self.config.prompt_length)
            prompt_ids = prompt_ids[-int(self.config.prompt_length):]
            reward_features["prompt_truncated"] = True
            reward_features["prompt_overflow_tokens"] = original_prompt_length - len(prompt_ids)
            logger.warning(
                "[RolloutPack][TRUNCATION] prompt_ids truncated: original=%d kept=%d overflow=%d limit=%d",
                original_prompt_length,
                len(prompt_ids),
                reward_features["prompt_overflow_tokens"],
                prompt_limit,
            )

        if len(response_ids) > int(self.config.response_length):
            response_limit = int(self.config.response_length)
            response_ids = response_ids[: int(self.config.response_length)]
            loss_mask = loss_mask[: int(self.config.response_length)]
            reward_features["response_truncated"] = True
            reward_features["response_overflow_tokens"] = original_response_length - len(response_ids)
            logger.warning(
                "[RolloutPack][TRUNCATION] response_ids truncated: original=%d kept=%d overflow=%d limit=%d",
                original_response_length,
                len(response_ids),
                reward_features["response_overflow_tokens"],
                response_limit,
            )

        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=output_device)
        prompt_attention = torch.ones_like(prompt_tensor, dtype=attention_dtype)
        prompt_tensor = pad_sequence_to_length(
            prompt_tensor,
            int(self.config.prompt_length),
            int(self.pad_token_id),
            left_pad=True,
        )
        prompt_attention = pad_sequence_to_length(
            prompt_attention,
            int(self.config.prompt_length),
            0,
            left_pad=True,
        )

        response_tensor = torch.tensor(response_ids, dtype=torch.long, device=output_device)
        response_attention = get_response_mask(
            response_id=response_tensor.unsqueeze(0),
            eos_token=eos_token_id,
            dtype=attention_dtype,
        )[0]
        loss_mask_tensor = torch.tensor(loss_mask, dtype=attention_dtype, device=output_device)

        response_tensor = pad_sequence_to_length(
            response_tensor,
            int(self.config.response_length),
            int(self.pad_token_id),
        )
        response_attention = pad_sequence_to_length(
            response_attention,
            int(self.config.response_length),
            0,
        )
        loss_mask_tensor = pad_sequence_to_length(
            loss_mask_tensor,
            int(self.config.response_length),
            0,
        )
        loss_mask_tensor = loss_mask_tensor * response_attention

        input_ids = torch.cat([prompt_tensor, response_tensor], dim=-1)
        attention_mask = torch.cat([prompt_attention, response_attention], dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

        full_expected = torch.tensor(prompt_ids + response_ids, dtype=torch.long, device=output_device)
        conversation_messages = list(artifacts.conversation_messages or [])
        reward_features["missing_conversation_messages"] = not bool(conversation_messages)
        if conversation_messages:
            conversation_inputs = self._prepare_message_inputs(
                conversation_messages,
                add_generation_prompt=False,
                device=output_device,
            )
            conversation_prompt_ids = torch.tensor(
                conversation_inputs["prompt_ids"],
                dtype=torch.long,
                device=output_device,
            )
            reward_features["multimodal_conversation_tokenization_mismatch"] = bool(
                conversation_prompt_ids.numel() != full_expected.numel()
                or not torch.equal(conversation_prompt_ids, full_expected)
            )
        else:
            conversation_inputs = {
                "prompt_ids": [],
                "image_count": 0,
                "video_count": 0,
            }
            reward_features["multimodal_conversation_tokenization_mismatch"] = bool(full_expected.numel() > 0)

        sample_extra_info = _coerce_row_object(sample.get("extra_info"))
        agent_name = str(
            sample.get("agent_name")
            or sample_extra_info.get("agent_name")
            or "single_turn_agent"
        ).strip()
        if not agent_name:
            agent_name = "single_turn_agent"

        interaction_kwargs = _coerce_row_object(sample.get("interaction_kwargs"))
        if not interaction_kwargs:
            interaction_kwargs = _coerce_row_object(sample_extra_info.get("interaction_kwargs"))

        tool_selection = _coerce_row_string_list(sample.get("tool_selection"))
        if not tool_selection:
            tool_selection = _coerce_row_string_list(sample_extra_info.get("tool_selection"))

        row: Dict[str, Any] = {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask_tensor,
            "sample_id": str(sample.get("sample_id") or ""),
            "dataset_name": str(sample.get("dataset_name") or sample.get("data_source") or ""),
            "agent_name": agent_name,
            "interaction_kwargs": deepcopy(interaction_kwargs),
            "tool_selection": list(tool_selection),
            "conversation_messages": deepcopy(conversation_messages),
            "image_input_count": torch.tensor(
                int(conversation_inputs.get("image_count", 0)),
                dtype=torch.long,
                device=output_device,
            ),
            "video_input_count": torch.tensor(
                int(conversation_inputs.get("video_count", 0)),
                dtype=torch.long,
                device=output_device,
            ),
            "episode_result": artifacts.run_result,
            "episode_trajectory": artifacts.trajectory,
            "episode_final_answer": artifacts.final_answer,
            "episode_success": bool(artifacts.success),
            "episode_reward_features": reward_features,
            "episode_generation_records": artifacts.generation_records,
            "rollout_policy_version": int(rollout_policy_version),
        }

        return row

    def _build_episode_runner(self, *, do_sample: bool, is_validate: bool) -> VideoAgentEpisodeRunner:
        enable_preprocess_payload_memory_cache = bool(
            self._enable_shared_preprocess_cache and not is_validate
        )
        preprocess_cache = (
            self._shared_preprocess_cache
            if enable_preprocess_payload_memory_cache
            else None
        )

        def _response_generator(**call_kwargs: Any) -> Dict[str, Any]:
            # planner.engine may pass do_sample in per-call generation kwargs
            # (e.g., structured_repair path). Avoid duplicating the keyword by
            # resolving it once here and forwarding a single value.
            resolved_do_sample = bool(call_kwargs.pop("do_sample", do_sample))
            if self.actor_module_fsdp is not None:
                with self._fsdp_response_generation_lock:
                    return self._generate_response_from_messages(
                        do_sample=resolved_do_sample,
                        **call_kwargs,
                    )
            return self._generate_response_from_messages(
                do_sample=resolved_do_sample,
                **call_kwargs,
            )

        return VideoAgentEpisodeRunner(
            tokenizer=self.tokenizer,
            rollout_config=self.config,
            response_generator=_response_generator,
            preprocess_cache=preprocess_cache,
            preprocess_cache_max_entries=self._shared_preprocess_cache_max_entries,
            enable_preprocess_payload_memory_cache=enable_preprocess_payload_memory_cache,
            accelerator_lock=self._actor_fsdp_generate_lock,
        )

    def _is_cooperative_turn_scheduler_enabled(self, *, total_sessions: int) -> bool:
        if total_sessions <= 1:
            return False
        return bool(self.config.get("enable_cooperative_turn_scheduler", False))

    def _generate_sequences_with_cooperative_turn_scheduler(
        self,
        *,
        session_specs: List[Dict[str, Any]],
        do_sample: bool,
        is_validate: bool,
        device: torch.device,
        attention_dtype: torch.dtype,
        eos_token_id: int,
        rollout_policy_version: int,
    ) -> List[Dict[str, Any]]:
        total_sessions = len(session_specs)
        if total_sessions <= 0:
            return []

        max_active_sessions = max(
            1,
            int(self.config.get("cooperative_turn_scheduler_max_active_sessions", 2) or 2),
        )
        max_active_sessions = min(max_active_sessions, total_sessions)
        rows_per_dispatch = max(
            1,
            int(self.config.get("cooperative_turn_scheduler_rows_per_dispatch", 1) or 1),
        )
        event_wait_timeout_raw = self.config.get(
            "cooperative_turn_scheduler_event_wait_timeout_seconds",
            300,
        )
        try:
            event_wait_timeout_seconds = float(event_wait_timeout_raw)
        except Exception:
            event_wait_timeout_seconds = 300.0
        if event_wait_timeout_seconds <= 0:
            event_wait_timeout_seconds = 300.0
        prioritize_new_sessions = bool(
            self.config.get("cooperative_turn_scheduler_prioritize_new_sessions", True)
        )
        cost_aware_dispatch = bool(
            self.config.get("cooperative_turn_scheduler_enable_cost_aware_dispatch", True)
        )
        prompt_token_weight = _coerce_float(
            self.config.get("cooperative_turn_scheduler_prompt_token_cost_weight", 1.0),
            1.0,
        )
        image_weight = _coerce_float(
            self.config.get("cooperative_turn_scheduler_image_cost_weight", 1536.0),
            1536.0,
        )
        video_weight = _coerce_float(
            self.config.get("cooperative_turn_scheduler_video_cost_weight", 4096.0),
            4096.0,
        )

        def _resolve_session_sample_key(spec: Dict[str, Any]) -> str:
            sample_payload = spec.get("sample")
            if isinstance(sample_payload, dict):
                sample_id = str(sample_payload.get("sample_id") or "").strip()
                if sample_id:
                    return f"sample_id:{sample_id}"
            trace_context_payload = spec.get("trace_context")
            if isinstance(trace_context_payload, dict):
                sample_index = trace_context_payload.get("sample_index")
                if sample_index is not None:
                    try:
                        return f"sample_index:{int(sample_index)}"
                    except Exception:
                        return f"sample_index:{sample_index}"
            try:
                output_order = int(spec.get("output_order", -1))
            except Exception:
                output_order = -1
            return f"output_order:{output_order}"

        pending_state_items: List[Dict[str, Any]] = [
            {
                "output_order": int(spec["output_order"]),
                "sample": deepcopy(spec["sample"]),
                "trace": bool(spec.get("trace", True)),
                "trace_context": dict(spec.get("trace_context") or {}),
                "sample_parallel_key": _resolve_session_sample_key(spec),
                "thread": None,
                "resume_event": threading.Event(),
                "started": False,
                "turn_yields": 0,
                "scheduler_cost_hint": _estimate_session_cost_hint_from_sample(spec.get("sample") or {}),
                "scheduler_dispatch_count": 0,
                "scheduler_wait_count": 0,
                "scheduler_runnable_version": 0,
                "scheduler_last_progress": {},
            }
            for spec in session_specs
        ]

        grouped_pending_states: Dict[str, deque[Dict[str, Any]]] = {}
        for state in pending_state_items:
            sample_parallel_key = str(state.get("sample_parallel_key") or "")
            if sample_parallel_key not in grouped_pending_states:
                grouped_pending_states[sample_parallel_key] = deque()
            grouped_pending_states[sample_parallel_key].append(state)

        interleaved_pending_state_items: List[Dict[str, Any]] = []
        while grouped_pending_states:
            for sample_parallel_key in list(grouped_pending_states.keys()):
                bucket = grouped_pending_states.get(sample_parallel_key)
                if not bucket:
                    grouped_pending_states.pop(sample_parallel_key, None)
                    continue
                interleaved_pending_state_items.append(bucket.popleft())
                if not bucket:
                    grouped_pending_states.pop(sample_parallel_key, None)

        unique_sample_parallel_slots = len(
            {
                str(state.get("sample_parallel_key") or "")
                for state in interleaved_pending_state_items
            }
        )
        if unique_sample_parallel_slots > 0 and max_active_sessions > unique_sample_parallel_slots:
            logger.info(
                "[TurnScheduler][DIAG] clamping max_active_sessions from %d to %d to keep sample-level parallelism and avoid concurrent group-rollouts for a single sample on one worker.",
                max_active_sessions,
                unique_sample_parallel_slots,
            )
            max_active_sessions = unique_sample_parallel_slots
        if self.actor_module_fsdp is not None and max_active_sessions > 1:
            logger.info(
                "[TurnScheduler][DIAG] local FSDP actor rollout keeps max_active_sessions=%d; "
                "episode threads may advance concurrently, while in-process actor "
                "generation is serialized by the FSDP response-generation lock.",
                max_active_sessions,
            )

        logger.info(
            "[TurnScheduler][DIAG] enabled=True total_sessions=%d unique_sample_slots=%d max_active_sessions=%d rows_per_dispatch=%d event_wait_timeout_seconds=%.1f prioritize_new_sessions=%s cost_aware_dispatch=%s validate=%s",
            total_sessions,
            unique_sample_parallel_slots,
            max_active_sessions,
            rows_per_dispatch,
            event_wait_timeout_seconds,
            prioritize_new_sessions,
            cost_aware_dispatch,
            is_validate,
        )

        scheduler_events: "queue.Queue[Tuple[str, int, Any]]" = queue.Queue()
        cancel_event = threading.Event()
        pending_states = deque(interleaved_pending_state_items)
        states_by_order = {int(state["output_order"]): state for state in list(pending_states)}
        runnable_states: List[Tuple[Tuple[Any, ...], int, int, Dict[str, Any]]] = []
        rows_by_output_order: Dict[int, Dict[str, Any]] = {}
        completed_count = 0
        active_session_count = 0
        total_turn_yields = 0
        timeout_warnings = 0
        runnable_entry_sequence = 0

        def _await_resume_permission(resume_event: threading.Event) -> None:
            while True:
                if resume_event.wait(timeout=0.1):
                    resume_event.clear()
                    if cancel_event.is_set():
                        raise RuntimeError("Cooperative turn scheduler cancelled pending session.")
                    return
                if cancel_event.is_set():
                    raise RuntimeError("Cooperative turn scheduler cancelled pending session.")

        def _session_main(state: Dict[str, Any]) -> None:
            try:
                _await_resume_permission(state["resume_event"])
                runner = self._build_episode_runner(do_sample=do_sample, is_validate=is_validate)

                def _progress_event_callback(event: Dict[str, Any]) -> None:
                    scheduler_events.put(("progress", int(state["output_order"]), dict(event or {})))

                def _turn_yield_callback(event: Dict[str, Any]) -> Dict[str, Any]:
                    state["turn_yields"] = int(state.get("turn_yields", 0)) + 1
                    scheduler_events.put(("yield", int(state["output_order"]), dict(event or {})))
                    wait_started = time.perf_counter()
                    _await_resume_permission(state["resume_event"])
                    return {"paused_seconds": max(0.0, time.perf_counter() - wait_started)}

                trace_context = dict(state.get("trace_context") or {})
                with rollout_trace_attr(
                    step=trace_context.get("step"),
                    sample_index=trace_context.get("sample_index"),
                    rollout_n=trace_context.get("rollout_n"),
                    validate=bool(trace_context.get("validate", False)),
                    name="video_agent_episode",
                    trace=bool(state.get("trace", True)),
                ):
                    artifacts = runner.run_episode(
                        sample=state["sample"],
                        turn_yield_callback=_turn_yield_callback,
                        progress_event_callback=_progress_event_callback,
                    )
                scheduler_events.put(("done", int(state["output_order"]), artifacts))
            except Exception as exc:
                if isinstance(exc, AcceleratorFatalError) or is_accelerator_fatal_error(exc):
                    scheduler_events.put(("error", int(state["output_order"]), exc))
                    return
                logger.warning(
                    "[TurnScheduler][SOFT_FAIL] Converting non-fatal session exception into a failed episode row. output_order=%s sample_id=%s error=%s",
                    int(state.get("output_order", -1)),
                    str((state.get("sample") or {}).get("sample_id") or "").strip(),
                    exc,
                )
                scheduler_events.put(
                    (
                        "done",
                        int(state["output_order"]),
                        self._build_soft_failed_episode_artifacts(
                            sample=dict(state.get("sample") or {}),
                            error_message=f"Error: {str(exc).strip() or exc.__class__.__name__}",
                        ),
                    )
                )

        def _push_runnable_state(state: Dict[str, Any]) -> None:
            nonlocal runnable_entry_sequence
            state["scheduler_runnable_version"] = int(state.get("scheduler_runnable_version", 0)) + 1
            runnable_entry_sequence += 1
            heapq.heappush(
                runnable_states,
                (
                    _state_scheduler_priority(
                        state,
                        prioritize_new_sessions=prioritize_new_sessions,
                        cost_aware_dispatch=cost_aware_dispatch,
                    ),
                    runnable_entry_sequence,
                    int(state["scheduler_runnable_version"]),
                    state,
                ),
            )

        def _pop_next_runnable_state() -> Optional[Dict[str, Any]]:
            while runnable_states:
                _, _, queued_version, state = heapq.heappop(runnable_states)
                if int(state.get("scheduler_runnable_version", 0)) != int(queued_version):
                    continue
                state["scheduler_wait_count"] = 0
                return state
            return None

        def _activate_next_pending(*, front: bool = False) -> bool:
            nonlocal active_session_count
            if not pending_states:
                return False
            state = pending_states.popleft()
            state["started"] = True
            thread = threading.Thread(
                target=_session_main,
                args=(state,),
                name=f"rollout-turn-session-{int(state['output_order'])}",
                daemon=True,
            )
            state["thread"] = thread
            thread.start()
            active_session_count += 1
            if front:
                state["scheduler_dispatch_count"] = 0
            _push_runnable_state(state)
            return True

        def _count_alive_worker_threads() -> int:
            alive_count = 0
            for state in states_by_order.values():
                thread = state.get("thread")
                if isinstance(thread, threading.Thread) and thread.is_alive():
                    alive_count += 1
            return alive_count

        try:
            while completed_count < total_sessions:
                while active_session_count < max_active_sessions and pending_states:
                    _activate_next_pending(front=prioritize_new_sessions)

                dispatch_count = 0
                while dispatch_count < rows_per_dispatch and runnable_states:
                    next_state = _pop_next_runnable_state()
                    if next_state is None:
                        break
                    next_state["scheduler_dispatch_count"] = int(next_state.get("scheduler_dispatch_count", 0)) + 1
                    next_state["resume_event"].set()
                    dispatch_count += 1

                try:
                    event_type, output_order, payload = scheduler_events.get(
                        timeout=event_wait_timeout_seconds,
                    )
                except queue.Empty:
                    timeout_warnings += 1
                    alive_worker_threads = _count_alive_worker_threads()
                    if timeout_warnings == 1 or timeout_warnings % 5 == 0:
                        logger.warning(
                            "[TurnScheduler][DIAG] event_wait_timeout=%.1fs warnings=%d completed=%d/%d active_sessions=%d alive_threads=%d runnable=%d pending=%d queued_events=%d",
                            event_wait_timeout_seconds,
                            timeout_warnings,
                            completed_count,
                            total_sessions,
                            active_session_count,
                            alive_worker_threads,
                            len(runnable_states),
                            len(pending_states),
                            scheduler_events.qsize(),
                        )
                    if alive_worker_threads <= 0 and not runnable_states and not pending_states:
                        raise RuntimeError(
                            "Cooperative turn scheduler stalled: no alive worker threads and no runnable/pending sessions."
                        )
                    continue

                state = states_by_order.get(int(output_order))
                if state is None:
                    raise RuntimeError(
                        f"Cooperative turn scheduler received event for unknown output_order={output_order}."
                    )

                if event_type == "progress":
                    normalized_payload = dict(payload or {})
                    state["scheduler_last_progress"] = normalized_payload
                    extracted_cost = _extract_session_progress_cost_hint(
                        normalized_payload,
                        prompt_token_weight=prompt_token_weight,
                        image_weight=image_weight,
                        video_weight=video_weight,
                    )
                    if extracted_cost is not None:
                        state["scheduler_cost_hint"] = max(
                            _coerce_float(state.get("scheduler_cost_hint"), extracted_cost["estimated_cost"]),
                            float(extracted_cost["estimated_cost"]),
                        )
                    continue

                if event_type == "yield":
                    total_turn_yields += 1
                    state["scheduler_wait_count"] = int(state.get("scheduler_wait_count", 0)) + 1
                    _push_runnable_state(state)
                    continue

                if event_type == "done":
                    active_session_count = max(0, active_session_count - 1)
                    state["scheduler_runnable_version"] = int(state.get("scheduler_runnable_version", 0)) + 1
                    rows_by_output_order[int(output_order)] = self._episode_to_row(
                        sample=state["sample"],
                        artifacts=payload,
                        device=device,
                        attention_dtype=attention_dtype,
                        eos_token_id=eos_token_id,
                        rollout_policy_version=rollout_policy_version,
                    )
                    completed_count += 1
                    continue

                if event_type == "error":
                    raise payload

                raise RuntimeError(f"Unsupported cooperative turn scheduler event: {event_type}")
        finally:
            cancel_event.set()
            for state in states_by_order.values():
                resume_event = state.get("resume_event")
                if isinstance(resume_event, threading.Event):
                    resume_event.set()
            for state in states_by_order.values():
                thread = state.get("thread")
                if isinstance(thread, threading.Thread) and thread.is_alive():
                    thread.join(timeout=1.0)

        rows = [
            rows_by_output_order[int(output_order)]
            for output_order in range(total_sessions)
        ]
        logger.info(
            "[TurnScheduler][DIAG] enabled=True total_sessions=%d completed_sessions=%d total_turn_yields=%d timeout_warnings=%d",
            total_sessions,
            completed_count,
            total_turn_yields,
            timeout_warnings,
        )
        return rows

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        previous_local_rollout_ulysses_state = bool(self._active_local_rollout_ulysses)
        requested_local_rollout_ulysses = bool(
            prompts.meta_info.get("__local_rollout_ulysses_enabled", False)
        )
        self._active_local_rollout_ulysses = bool(
            requested_local_rollout_ulysses and self._get_local_rollout_ulysses_world_size() > 1
        )
        batch_size = len(prompts)
        do_sample = bool(prompts.meta_info.get("do_sample", True))
        is_validate = bool(prompts.meta_info.get("validate", False))
        dispatch_mode = str(prompts.meta_info.get("__local_rollout_dispatch_mode") or "legacy_rank_shard")
        unique_sample_count = int(prompts.meta_info.get("__local_rollout_unique_sample_count", batch_size) or 0)
        configured_repeat_times = int(self.config.n if do_sample else 1)
        repeat_times = max(
            1,
            int(prompts.meta_info.get("__rollout_repeat_override", configured_repeat_times) or configured_repeat_times),
        )
        rollout_policy_version = int(prompts.meta_info.get("policy_version", 0) or 0)
        prompt_batch = prompts.batch
        device = self._resolve_device()
        attention_dtype = torch.long
        if prompt_batch is not None:
            try:
                if "input_ids" in prompt_batch.keys():
                    device = prompt_batch["input_ids"].device
            except Exception:
                pass
            try:
                if "attention_mask" in prompt_batch.keys():
                    attention_dtype = prompt_batch["attention_mask"].dtype
            except Exception:
                pass
        eos_token_id = int(
            prompts.meta_info.get(
                "eos_token_id",
                getattr(self.tokenizer, "eos_token_id", None),
            )
            or getattr(self.tokenizer, "pad_token_id", 0)
            or 0
        )
        global_step = int(prompts.meta_info.get("global_steps", -1) or -1)
        raw_sample_indices = list(prompts.non_tensor_batch.get("index", np.arange(batch_size)))
        sample_indices: List[int] = []
        for index in range(batch_size):
            raw_value = raw_sample_indices[index] if index < len(raw_sample_indices) else index
            try:
                sample_indices.append(int(raw_value))
            except Exception:
                sample_indices.append(int(index))
        traced_sample_indices = self._select_traced_sample_indices(sample_indices)
        sample_ids = [
            str(item)
            for item in list(prompts.non_tensor_batch.get("sample_id", np.empty((0,), dtype=object)))
        ]
        padding_flags = [
            bool(item)
            for item in list(prompts.non_tensor_batch.get("__verl_is_padding", np.empty((0,), dtype=object)))
        ]
        logger.info(
            "[RolloutBatch][DIAG] batch_size=%d do_sample=%s validate=%s repeat_times=%d dispatch_mode=%s local_rollout_ulysses=%s unique_samples=%d sample_ids=%s padding_flags=%s",
            batch_size,
            do_sample,
            is_validate,
            repeat_times,
            dispatch_mode,
            self._active_local_rollout_ulysses,
            unique_sample_count,
            sample_ids,
            padding_flags,
        )
        logger.info(
            "[RolloutTrace][DIAG] backend=%s traced_unique_samples=%d total_unique_samples=%d repeat_times=%d global_step=%d validate=%s",
            RolloutTraceConfig.get_backend(),
            len(traced_sample_indices),
            len(set(sample_indices)),
            repeat_times,
            global_step,
            is_validate,
        )
        session_specs: List[Dict[str, Any]] = []
        for index in range(batch_size):
            item = prompts[index]
            sample = {key: value for key, value in item.non_tensor_batch.items()}
            sample_index = int(sample_indices[index])
            trace_this_sample = sample_index in traced_sample_indices
            for rollout_n in range(repeat_times):
                session_specs.append(
                    {
                        "output_order": len(session_specs),
                        "sample": deepcopy(sample),
                        "trace": bool(trace_this_sample),
                        "trace_context": {
                            "step": global_step,
                            "sample_index": sample_index,
                            "rollout_n": int(rollout_n),
                            "validate": bool(is_validate),
                        },
                    }
                )

        if self._is_local_rollout_ulysses_enabled() and not self._is_local_rollout_ulysses_leader():
            try:
                return self._run_local_rollout_ulysses_follower_loop()
            finally:
                self._active_local_rollout_ulysses = previous_local_rollout_ulysses_state

        rows: List[Dict[str, Any]] = []
        try:
            if self._is_cooperative_turn_scheduler_enabled(total_sessions=len(session_specs)):
                rows = self._generate_sequences_with_cooperative_turn_scheduler(
                    session_specs=session_specs,
                    do_sample=do_sample,
                    is_validate=is_validate,
                    device=device,
                    attention_dtype=attention_dtype,
                    eos_token_id=eos_token_id,
                    rollout_policy_version=rollout_policy_version,
                )
            else:
                runner = self._build_episode_runner(do_sample=do_sample, is_validate=is_validate)
                for session_spec in session_specs:
                    sample = session_spec["sample"]
                    trace_context = dict(session_spec.get("trace_context") or {})
                    with rollout_trace_attr(
                        step=trace_context.get("step"),
                        sample_index=trace_context.get("sample_index"),
                        rollout_n=trace_context.get("rollout_n"),
                        validate=bool(trace_context.get("validate", False)),
                        name="video_agent_episode",
                        trace=bool(session_spec.get("trace", True)),
                    ):
                        artifacts = runner.run_episode(sample)
                    rows.append(
                        self._episode_to_row(
                            sample=sample,
                            artifacts=artifacts,
                            device=device,
                            attention_dtype=attention_dtype,
                            eos_token_id=eos_token_id,
                            rollout_policy_version=rollout_policy_version,
                        )
                    )
        finally:
            if self._is_local_rollout_ulysses_enabled() and self._is_local_rollout_ulysses_leader():
                logger.info("[LocalRolloutUlysses][DIAG] leader broadcasting shutdown.")
                self._broadcast_local_rollout_ulysses_command({"type": "shutdown"})
            self._active_local_rollout_ulysses = previous_local_rollout_ulysses_state

        batch_tensors = {
            "prompts": torch.stack([row["prompts"] for row in rows], dim=0),
            "responses": torch.stack([row["responses"] for row in rows], dim=0),
            "input_ids": torch.stack([row["input_ids"] for row in rows], dim=0),
            "attention_mask": torch.stack([row["attention_mask"] for row in rows], dim=0),
            "position_ids": torch.stack([row["position_ids"] for row in rows], dim=0),
            "loss_mask": torch.stack([row["loss_mask"] for row in rows], dim=0),
            "image_input_count": torch.stack([row["image_input_count"] for row in rows], dim=0),
            "video_input_count": torch.stack([row["video_input_count"] for row in rows], dim=0),
        }

        batch = TensorDict(batch_tensors, batch_size=len(rows))

        non_tensor_batch = {
            "sample_id": to_1d_object_array([row["sample_id"] for row in rows]),
            "dataset_name": to_1d_object_array([row["dataset_name"] for row in rows]),
            "agent_name": to_1d_object_array([row["agent_name"] for row in rows]),
            "interaction_kwargs": to_1d_object_array([row["interaction_kwargs"] for row in rows]),
            "tool_selection": to_1d_object_array([row["tool_selection"] for row in rows]),
            "episode_result": to_1d_object_array([row["episode_result"] for row in rows]),
            "episode_trajectory": to_1d_object_array([row["episode_trajectory"] for row in rows]),
            "episode_final_answer": to_1d_object_array([row["episode_final_answer"] for row in rows]),
            "episode_success": to_1d_object_array([row["episode_success"] for row in rows]),
            "episode_reward_features": to_1d_object_array([row["episode_reward_features"] for row in rows]),
            "episode_generation_records": to_1d_object_array([row["episode_generation_records"] for row in rows]),
            "conversation_messages": to_1d_object_array([row["conversation_messages"] for row in rows]),
            "rollout_policy_version": to_1d_object_array([row["rollout_policy_version"] for row in rows]),
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
