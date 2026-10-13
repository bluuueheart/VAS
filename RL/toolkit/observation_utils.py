"""Shared helpers for planner-facing tool observations."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from toolkit.base import format_tool_result_observation, summarize_tool_result


MAX_OBSERVATION_CHARS = 4000
_PREFERRED_TEXT_KEYS = (
    "result",
    "answer",
    "summary",
    "compressed_summary",
    "findings",
    "caption",
    "transcript",
    "extracted_text",
    "description",
    "context",
    "error",
    "warning",
)
_PREFERRED_META_KEYS = (
    "status",
    "label",
    "value",
    "citation",
    "evidence_status",
    "segment_id",
    "segment_ref",
    "raw_segment_id",
    "segment_aliases",
    "video",
    "source_video",
    "source_segment",
    "time",
    "time_range",
    "search_window",
    "timestamp",
    "t_start",
    "t_end",
    "transcript",
    "extracted_text",
    "frames_analyzed",
    "sample_times",
    "frame_times",
    "frame_paths",
    "image_evidence",
    "image_count",
    "source_image_count",
    "image_timestamps",
    "matched_modalities",
    "similarity_score",
    "score",
    "probe_label",
    "mode",
    "query",
    "matched_segments",
    "resolved_tool",
    "resolved_segment_id",
    "resolved_video",
    "requested_segment_id",
    "candidate_tools",
    "available_fields",
    "normalized_fields",
    "tool",
)
_TRANSPORT_RESULT_RE = re.compile(
    r"\b("
    r"completed with status|"
    r"found\s+\d+\s+relevant segments|"
    r"returned\s+\d+\s+matching segments|"
    r"parallel execution completed|"
    r"tool search"
    r")\b",
    re.IGNORECASE,
)
_PLANNER_TRANSPORT_KEYS = {
    "mode",
    "success",
    "executed",
    "executed_tool",
    "executed_params",
    "tool_status",
    "observation_text",
    "observation_payload",
    "final_result",
    "source",
    "abstract_tool",
    "attempt",
    "successful_items",
    "failed_items",
    "item_count",
    "candidate_tool_count",
    "matched_segment_count",
}
_PLANNER_VISUAL_RESULT_KEYS = (
    "image_evidence",
    "image_timestamps",
)
_PLANNER_RESULT_ONLY_TOOL_NAMES = {"Video_Clip_QA"}
_PLANNER_VISUAL_TEXT_KEYS = (
    "tool",
    "answer",
    "answer_value",
    "observed_value",
    "description",
    "action_description",
    "findings",
    "target",
    "source",
    "relation",
    "direction",
    "qualifiers",
    "uncertainty",
    "negation",
    "order",
    "sequence",
    "before",
    "during",
    "after",
    "dominant_emotion",
    "emotion_scores",
    "modalities_analyzed",
    "query",
    "language",
    "confidence",
    "density",
    "duration",
    "source_video_duration_seconds",
    "source_video_path",
    "mode",
    "timestamp",
    "t_start",
    "t_end",
    "requested_segment_id",
    "resolved_segment_id",
    "resolved_segment_ref",
    "bbox",
    "resolved_bbox",
    "roi",
    "resolved_roi",
    "zoom_factor",
)


def _preserve_full_observations_enabled() -> bool:
    return bool(getattr(settings, "preserve_full_observations", False))


def _should_preserve_full_result_for_tool(tool_name: str = "") -> bool:
    return str(tool_name or "").strip() in _PLANNER_RESULT_ONLY_TOOL_NAMES


def _safe_json_loads(value: Any) -> Optional[Any]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _truncate_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def infer_observation_status(payload: Any, default: str = "success") -> str:
    if isinstance(payload, dict):
        status = str(payload.get("status") or "").strip().lower()
        if status:
            return status
        if payload.get("error"):
            return "error"
        result_text = str(payload.get("result") or payload.get("answer") or "")
        lowered = result_text.lower()
        if "did not provide usable evidence" in lowered or "no matching" in lowered:
            return "no_valid_result"
        return default
    if isinstance(payload, str):
        lowered = payload.lower()
        if lowered.startswith("error"):
            return "error"
        if "did not provide usable evidence" in lowered or "no matching" in lowered:
            return "no_valid_result"
    if isinstance(payload, list) and not payload:
        return "no_valid_result"
    return default


def _is_transport_result_text(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    return bool(_TRANSPORT_RESULT_RE.search(normalized))


def _extract_segment_result_texts(segments: Any, limit: Optional[int] = 2) -> List[str]:
    texts: List[str] = []
    if not isinstance(segments, list):
        return texts
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_text = ""
        for key in ("result", "findings", "caption", "transcript", "snippet", "description", "context"):
            value = segment.get(key)
            if isinstance(value, str) and value.strip():
                segment_text = value.strip()
                break
        if not segment_text:
            continue
        texts.append(segment_text)
        if limit is not None and len(texts) >= limit:
            break
    return texts


def _extract_planner_result_text_from_payload(payload: Any, tool_name: str = "") -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        parsed = _safe_json_loads(payload)
        if parsed is not None:
            return _extract_planner_result_text_from_payload(parsed, tool_name=tool_name)
        return payload.strip()
    if isinstance(payload, list):
        parts: List[str] = []
        items = payload if _preserve_full_observations_enabled() else payload[:3]
        for item in items:
            text = _extract_planner_result_text_from_payload(item, tool_name=tool_name)
            if text and text not in parts:
                parts.append(text)
        return " | ".join(parts)
    if not isinstance(payload, dict):
        return str(payload).strip()

    if isinstance(payload.get("image_evidence"), list) and payload.get("image_evidence"):
        timestamps = payload.get("image_timestamps")
        if isinstance(timestamps, list):
            compact_timestamps: List[float] = []
            for value in timestamps:
                try:
                    compact_timestamps.append(round(float(value), 3))
                except Exception:
                    continue
            if compact_timestamps:
                return "image_timestamps=" + json.dumps(compact_timestamps, ensure_ascii=False)

    for key in ("answer", "findings", "caption", "transcript", "extracted_text", "description", "context"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    result_text = ""
    for key in ("result", "summary", "compressed_summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            result_text = value.strip()
            if not _is_transport_result_text(result_text):
                return result_text
            break

    segment_texts = _extract_segment_result_texts(
        payload.get("matched_segments"),
        limit=None if _preserve_full_observations_enabled() else 2,
    )
    if segment_texts:
        return " | ".join(segment_texts)

    item_texts: List[str] = []
    item_values = payload.get("items", []) if isinstance(payload.get("items"), list) else []
    if not _preserve_full_observations_enabled():
        item_values = item_values[:3]
    for item in item_values:
        text = _extract_planner_result_text_from_payload(item, tool_name=tool_name)
        if text and text not in item_texts:
            item_texts.append(text)
    if item_texts:
        return " | ".join(item_texts)

    if result_text:
        return result_text
    if payload.get("error"):
        return str(payload.get("error")).strip()
    return summarize_tool_result(payload, tool_name=tool_name)


def extract_planner_result_text(result: Any, tool_name: str = "") -> str:
    payload = _normalize_payload(result, tool_name=tool_name)
    return _extract_planner_result_text_from_payload(payload, tool_name=tool_name).strip()


def _compact_scalar(value: Any, aggressive: bool = False) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, 220 if aggressive else 420)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate_text(value, 220 if aggressive else 420)


def _compact_list_preview(values: List[Any], aggressive: bool = False) -> List[Any]:
    limit = 2 if aggressive else 3
    preview: List[Any] = []
    for item in values[:limit]:
        preview.append(_compact_value(item, aggressive=aggressive, depth=1))
    if len(values) > limit:
        preview.append(f"... ({len(values)} items total)")
    return preview


def _compact_item_dict(payload: Dict[str, Any], aggressive: bool = False) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for key in ("tool", "status", "result", "answer", "summary", "caption", "transcript", "findings", "extracted_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            compacted["result" if key in {"answer", "summary", "caption", "transcript", "findings", "extracted_text"} else key] = (
                _truncate_text(value, 240 if aggressive else 420)
            )
            if key in {"answer", "summary", "caption", "transcript", "findings", "extracted_text"}:
                break
    for key in (
        "label",
        "value",
        "citation",
        "claim_id",
        "scope",
        "target",
        "source",
        "relation",
        "direction",
        "qualifiers",
        "observed_value",
        "uncertainty",
        "negation",
        "evidence_status",
        "segment_id",
        "segment_ref",
        "raw_segment_id",
        "video",
        "source_video",
        "source_segment",
        "time",
        "time_range",
        "search_window",
        "timestamp",
        "t_start",
        "t_end",
        "extracted_text",
        "matched_segments",
        "matched_modalities",
        "frame_paths",
        "image_evidence",
        "image_count",
        "source_image_count",
        "image_timestamps",
        "similarity_score",
        "score",
        "probe_label",
        "resolved_tool",
    ):
        if key in compacted:
            continue
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        compacted[key] = _compact_value(value, aggressive=True, depth=1)

    if "result" not in compacted:
        summary = summarize_tool_result(payload, tool_name=str(payload.get("tool", "")))
        if summary:
            compacted["result"] = _truncate_text(summary, 240 if aggressive else 420)
    return compacted


def _compact_value(value: Any, aggressive: bool = False, depth: int = 0) -> Any:
    if isinstance(value, dict):
        return _compact_dict(value, aggressive=aggressive, depth=depth + 1)
    if isinstance(value, list):
        return _compact_list_preview(value, aggressive=aggressive)
    return _compact_scalar(value, aggressive=aggressive)


def _compact_dict(payload: Dict[str, Any], aggressive: bool = False, depth: int = 0) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for key in _PREFERRED_TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            compacted[key] = _truncate_text(value, 700 if aggressive else 1400)
            if key != "warning":
                break

    if "status" in payload:
        compacted["status"] = infer_observation_status(payload, default=str(payload.get("status") or "success"))

    for key in _PREFERRED_META_KEYS:
        if key in compacted or key not in payload:
            continue
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        compacted[key] = _compact_value(value, aggressive=aggressive, depth=depth + 1)

    items = payload.get("items")
    if isinstance(items, list) and items:
        compacted["items"] = [
            _compact_item_dict(item, aggressive=True)
            for item in items[: (2 if aggressive else 4)]
            if isinstance(item, dict)
        ]
        if len(items) > len(compacted["items"]):
            compacted["item_count"] = len(items)

    matched_segments = payload.get("matched_segments")
    if isinstance(matched_segments, list) and matched_segments:
        compacted["matched_segments"] = [
            _compact_item_dict(item, aggressive=True)
            for item in matched_segments[:2]
            if isinstance(item, dict)
        ]
        if len(matched_segments) > len(compacted["matched_segments"]):
            compacted["matched_segment_count"] = len(matched_segments)

    candidates = payload.get("candidate_tools")
    if isinstance(candidates, list) and candidates:
        compacted["candidate_tools"] = [
            _compact_item_dict(item, aggressive=True)
            for item in candidates[:4]
            if isinstance(item, dict)
        ]
        if len(candidates) > len(compacted["candidate_tools"]):
            compacted["candidate_tool_count"] = len(candidates)

    if not compacted.get("result"):
        summary = summarize_tool_result(payload)
        if summary:
            compacted["result"] = _truncate_text(summary, 700 if aggressive else 1400)

    if not aggressive:
        scalar_keys: List[str] = []
        for key, value in payload.items():
            if key in compacted or value in (None, "", [], {}) or isinstance(value, (dict, list)):
                continue
            scalar_keys.append(key)
            compacted[key] = _compact_scalar(value, aggressive=False)
            if len(scalar_keys) >= 4:
                break

    omitted = [
        key for key in payload.keys()
        if key not in compacted and payload.get(key) not in (None, "", [], {})
    ]
    if omitted and not aggressive:
        compacted["omitted_fields"] = omitted[:8]
    return compacted


def _normalize_payload(result: Any, tool_name: str = "") -> Any:
    if isinstance(result, str):
        parsed = _safe_json_loads(result)
        if parsed is not None:
            return parsed
        return {"result": result.strip() or summarize_tool_result(result, tool_name=tool_name)}

    if bool(getattr(settings, "preserve_full_observations", False)) and isinstance(result, (dict, list)):
        return result

    formatted = format_tool_result_observation(result, tool_name=tool_name)
    parsed = _safe_json_loads(formatted)
    if parsed is not None:
        return parsed
    return {"result": formatted.strip() or summarize_tool_result(result, tool_name=tool_name)}


def _build_planner_payload(payload: Any, tool_name: str = "") -> Any:
    if isinstance(payload, list):
        return [
            _build_planner_payload(item, tool_name=tool_name)
            for item in payload
        ]
    if not isinstance(payload, dict):
        if isinstance(payload, str):
            parsed = _safe_json_loads(payload)
            if parsed is not None:
                return _build_planner_payload(parsed, tool_name=tool_name)
            return {"result": payload.strip()} if payload.strip() else {"result": summarize_tool_result(payload, tool_name=tool_name)}
        return {"result": str(payload).strip()} if payload not in (None, "") else {"result": summarize_tool_result(payload, tool_name=tool_name)}

    planner_payload: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in _PLANNER_TRANSPORT_KEYS:
            continue
        if isinstance(value, dict):
            planner_payload[key] = _build_planner_payload(value, tool_name=tool_name)
        elif isinstance(value, list):
            planner_payload[key] = [
                _build_planner_payload(item, tool_name=tool_name) if isinstance(item, (dict, list)) else item
                for item in value
            ]
        else:
            planner_payload[key] = value

    planner_result = _extract_planner_result_text_from_payload(planner_payload, tool_name=tool_name)
    if planner_result:
        planner_payload["result"] = planner_result
    elif "result" not in planner_payload:
        planner_payload["result"] = summarize_tool_result(planner_payload, tool_name=tool_name)
    result_only_payload = _compact_result_only_planner_payload(planner_payload, tool_name=tool_name)
    if result_only_payload is not None and not _preserve_full_observations_enabled():
        return result_only_payload
    visual_payload = _compact_visual_planner_payload(planner_payload, tool_name=tool_name)
    if visual_payload is not None:
        return visual_payload
    return planner_payload


def _compact_result_only_planner_payload(payload: Any, tool_name: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if str(tool_name or "").strip() not in _PLANNER_RESULT_ONLY_TOOL_NAMES:
        return None

    compact_payload: Dict[str, Any] = {
        "result": _extract_planner_result_text_from_payload(payload, tool_name=tool_name)
        or summarize_tool_result(payload, tool_name=tool_name),
    }
    status = infer_observation_status(payload)
    compact_payload["status"] = status
    if payload.get("error") not in (None, ""):
        compact_payload["error"] = str(payload.get("error")).strip()
    if status != "success" and payload.get("warning") not in (None, ""):
        compact_payload["warning"] = str(payload.get("warning")).strip()
    return compact_payload


def _compact_visual_planner_payload(payload: Any, tool_name: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    image_evidence = payload.get("image_evidence")
    if not isinstance(image_evidence, list) or not image_evidence:
        return None

    compact_images: List[Dict[str, Any]] = []
    derived_timestamps: List[float] = []

    for item in image_evidence:
        if not isinstance(item, dict):
            continue
        image_ref = str(item.get("image", "") or item.get("path", "") or "").strip()
        if not image_ref:
            continue
        compact_item: Dict[str, Any] = {"image": image_ref}

        sample_times: List[float] = []
        for value in item.get("sample_times") or []:
            try:
                sample_times.append(round(float(value), 3))
            except Exception:
                continue
        if sample_times:
            compact_item["sample_times"] = sample_times
            for value in sample_times:
                if value not in derived_timestamps:
                    derived_timestamps.append(value)
        elif item.get("timestamp") is not None:
            try:
                timestamp = round(float(item.get("timestamp")), 3)
                compact_item["timestamp"] = timestamp
                if timestamp not in derived_timestamps:
                    derived_timestamps.append(timestamp)
            except Exception:
                pass

        pack_mode = str(item.get("pack_mode", "") or "").strip()
        if pack_mode:
            compact_item["pack_mode"] = pack_mode

        compact_images.append(compact_item)

    if not compact_images:
        return None

    compact_payload: Dict[str, Any] = {
        "result": _extract_planner_result_text_from_payload(payload, tool_name=tool_name),
        "image_evidence": compact_images,
    }
    for key in _PLANNER_VISUAL_RESULT_KEYS:
        if key == "image_evidence":
            continue
        if key == "image_timestamps":
            compact_payload[key] = derived_timestamps
            continue
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        compact_payload[key] = value

    items = payload.get("items")
    if isinstance(items, list) and items:
        compact_payload["items"] = [
            _compact_item_dict(item, aggressive=True)
            for item in items[:2]
            if isinstance(item, dict)
        ]
        if len(items) > len(compact_payload["items"]):
            compact_payload["item_count"] = len(items)

    matched_segments = payload.get("matched_segments")
    if isinstance(matched_segments, list) and matched_segments:
        compact_payload["matched_segments"] = [
            _compact_item_dict(item, aggressive=True)
            for item in matched_segments[:2]
            if isinstance(item, dict)
        ]
        if len(matched_segments) > len(compact_payload["matched_segments"]):
            compact_payload["matched_segment_count"] = len(matched_segments)

    for key in _PLANNER_VISUAL_TEXT_KEYS:
        if key in compact_payload:
            continue
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        compact_payload[key] = _compact_value(value, aggressive=True, depth=1)

    status = infer_observation_status(payload)
    compact_payload["status"] = status
    if status != "success" and payload.get("error") not in (None, ""):
        compact_payload["error"] = str(payload.get("error")).strip()
    if status != "success" and payload.get("warning") not in (None, ""):
        compact_payload["warning"] = str(payload.get("warning")).strip()
    return compact_payload


def build_planner_observation_record(
    tool_name: str,
    result: Any,
    status: Optional[str] = None,
    max_chars: int = MAX_OBSERVATION_CHARS,
) -> Dict[str, Any]:
    normalized = _normalize_payload(result, tool_name=tool_name)
    planner_payload = _build_planner_payload(normalized, tool_name=tool_name)
    inferred_status = infer_observation_status(planner_payload)
    final_status = str(status or inferred_status).strip().lower() or "success"
    if final_status == "success" and inferred_status != "success":
        final_status = inferred_status

    if _preserve_full_observations_enabled() or _should_preserve_full_result_for_tool(tool_name):
        record: Dict[str, Any] = {"tool": tool_name}
        if isinstance(planner_payload, dict):
            for key, value in planner_payload.items():
                if key == "tool":
                    continue
                record[key] = value
        elif isinstance(planner_payload, list):
            record["items"] = planner_payload
            record["item_count"] = len(planner_payload)
        else:
            record["result"] = str(planner_payload or "").strip()

        if not isinstance(record.get("result"), str) or not str(record.get("result", "")).strip():
            record["result"] = extract_planner_result_text(planner_payload, tool_name=tool_name)
        record["status"] = final_status
        return record

    candidate_payloads: List[Any] = [planner_payload]
    if isinstance(planner_payload, dict):
        candidate_payloads.append(_compact_dict(planner_payload, aggressive=False))
        candidate_payloads.append(_compact_dict(planner_payload, aggressive=True))
    elif isinstance(planner_payload, list):
        candidate_payloads.append(
            {
                "items": _compact_list_preview(planner_payload, aggressive=False),
                "item_count": len(planner_payload),
                "result": extract_planner_result_text(planner_payload, tool_name=tool_name),
            }
        )
        candidate_payloads.append(
            {
                "items_preview": _compact_list_preview(planner_payload, aggressive=True),
                "item_count": len(planner_payload),
                "result": extract_planner_result_text(planner_payload, tool_name=tool_name),
            }
        )

    for candidate in candidate_payloads:
        record: Dict[str, Any] = {"tool": tool_name}
        if isinstance(candidate, dict):
            for key, value in candidate.items():
                if key == "tool":
                    continue
                record[key] = value
        elif isinstance(candidate, list):
            record["items"] = candidate
            record["item_count"] = len(candidate)
        else:
            record["result"] = str(candidate or "").strip()

        if not isinstance(record.get("result"), str) or not str(record.get("result", "")).strip():
            record["result"] = extract_planner_result_text(candidate, tool_name=tool_name)
        record["status"] = final_status

        if _serialized_size(record) <= max_chars:
            return record

    fallback = {
        "tool": tool_name,
        "result": _truncate_text(extract_planner_result_text(planner_payload, tool_name=tool_name), max_chars=max(200, max_chars - 80)),
    }
    fallback["status"] = final_status
    return fallback


def format_compact_observation(
    result: Any,
    tool_name: str = "",
    max_chars: int = MAX_OBSERVATION_CHARS,
) -> Tuple[str, str, Any]:
    payload = _normalize_payload(result, tool_name=tool_name)
    planner_payload = _build_planner_payload(payload, tool_name=tool_name)
    if _preserve_full_observations_enabled() or _should_preserve_full_result_for_tool(tool_name):
        serialized = (
            planner_payload
            if isinstance(planner_payload, str)
            else json.dumps(planner_payload, ensure_ascii=False, indent=2)
        )
        return serialized, infer_observation_status(planner_payload), planner_payload

    candidates: List[Any] = [planner_payload]

    if isinstance(planner_payload, dict):
        candidates.append(_compact_dict(planner_payload, aggressive=False))
        candidates.append(_compact_dict(planner_payload, aggressive=True))
    elif isinstance(planner_payload, list):
        summary = summarize_tool_result(planner_payload, tool_name=tool_name)
        candidates.append(
            {
                "result": summary,
                "items": _compact_list_preview(planner_payload, aggressive=False),
                "item_count": len(planner_payload),
            }
        )
        candidates.append(
            {
                "result": _truncate_text(summary, 900),
                "items_preview": [
                    _compact_item_dict(item, aggressive=True)
                    if isinstance(item, dict)
                    else _truncate_text(summarize_tool_result(item, tool_name=tool_name), 180)
                    for item in planner_payload[:3]
                ],
                "item_count": len(planner_payload),
            }
        )

    serialized = ""
    chosen = planner_payload
    for candidate in candidates:
        serialized = json.dumps(candidate, ensure_ascii=False, indent=2)
        if len(serialized) <= max_chars:
            chosen = candidate
            break
    else:
        summary = summarize_tool_result(planner_payload, tool_name=tool_name)
        status = infer_observation_status(planner_payload)
        chosen = {
            "result": _truncate_text(summary, max(200, max_chars - 120)),
            "status": status,
        }
        serialized = json.dumps(chosen, ensure_ascii=False, indent=2)

    status = infer_observation_status(chosen)
    return serialized, status, chosen


def _serialized_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, indent=2))
    except Exception:
        return len(str(value))


def build_observation_record(
    tool_name: str,
    result: Any,
    status: Optional[str] = None,
    max_chars: int = MAX_OBSERVATION_CHARS,
) -> Dict[str, Any]:
    serialized, inferred_status, payload = format_compact_observation(
        result,
        tool_name=tool_name,
        max_chars=max_chars,
    )
    final_status = str(status or inferred_status or "success")
    if final_status == "success" and inferred_status != "success":
        final_status = inferred_status
    preserve_more = _preserve_full_observations_enabled()
    if preserve_more:
        record: Dict[str, Any] = {
            "tool": tool_name,
            "status": final_status,
        }
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in {"tool", "status"}:
                    continue
                record[key] = value
        elif isinstance(payload, list):
            record["items"] = payload
            record["item_count"] = len(payload)
        else:
            record["result"] = str(payload or "")

        if not isinstance(record.get("result"), str) or not str(record.get("result", "")).strip():
            record["result"] = serialized
        return record

    result_text = _truncate_text(
        summarize_tool_result(payload, tool_name=tool_name),
        1400 if preserve_more else 900,
    )
    if not result_text:
        result_text = _truncate_text(serialized, 1400 if preserve_more else 900)
    record: Dict[str, Any] = {
        "tool": tool_name,
        "status": final_status,
        "result": result_text,
    }
    if isinstance(payload, dict):
        preferred_keys = (
            "label",
            "value",
            "citation",
            "evidence_status",
            "segment_id",
            "segment_ref",
            "raw_segment_id",
            "video",
            "source_video",
            "source_segment",
            "time",
            "time_range",
            "search_window",
            "timestamp",
            "t_start",
            "t_end",
            "extracted_text",
            "matched_segments",
            "matched_modalities",
            "frame_paths",
            "image_evidence",
            "image_count",
            "resolved_tool",
            "items",
            "candidate_tools",
            "available_fields",
            "normalized_fields",
            "match_count",
            "successful_items",
            "failed_items",
            "similarity_score",
            "score",
            "probe_label",
            "query",
        )
        aggressive = not preserve_more
        for key in preferred_keys:
            value = payload.get(key)
            if value not in (None, "", [], {}):
                compacted = _compact_value(value, aggressive=aggressive)
                candidate = dict(record)
                candidate[key] = compacted
                if _serialized_size(candidate) <= max_chars:
                    record[key] = compacted
        if preserve_more:
            omitted: List[str] = []
            for key, value in payload.items():
                if key in record or key == "status" or value in (None, "", [], {}):
                    continue
                compacted = _compact_value(
                    value,
                    aggressive=isinstance(value, (dict, list)),
                )
                candidate = dict(record)
                candidate[key] = compacted
                if _serialized_size(candidate) > max_chars:
                    omitted.append(key)
                    continue
                record[key] = compacted
            if omitted and "omitted_fields" not in record:
                record["omitted_fields"] = omitted[:8]
        return record
    if isinstance(payload, list):
        preview_limit = 4 if preserve_more else 3
        preview_items: List[Any] = []
        for item in payload[:preview_limit]:
            if isinstance(item, dict):
                preview_items.append(_compact_item_dict(item, aggressive=not preserve_more))
            else:
                preview_items.append(
                    _truncate_text(
                        summarize_tool_result(item, tool_name=tool_name),
                        180 if preserve_more else 120,
                    )
                )
        if preview_items:
            record["items"] = preview_items
            if len(payload) > len(preview_items):
                record["item_count"] = len(payload)
    return record


def build_parallel_observation(
    items: List[Dict[str, Any]],
    max_chars: int = MAX_OBSERVATION_CHARS,
) -> Tuple[str, str, Dict[str, Any]]:
    normalized_items: List[Dict[str, Any]] = []
    summary_parts: List[str] = []
    failed = 0
    no_valid = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool", "Unknown")).strip() or "Unknown"
        item_status = str(item.get("status", "")).strip() or None
        item_result = item.get("payload", item.get("result"))
        serialized, inferred_status, payload = format_compact_observation(
            item_result,
            tool_name=tool_name,
            max_chars=1400,
        )
        final_status = str(item_status or inferred_status or "success")
        if final_status == "success" and inferred_status != "success":
            final_status = inferred_status
        if final_status != "success":
            failed += 1
            if final_status == "no_valid_result":
                no_valid += 1
        item_entry = build_planner_observation_record(
            tool_name=tool_name,
            result=item_result,
            status=final_status,
            max_chars=max_chars,
        )
        if final_status != "success":
            item_entry["status"] = final_status
        normalized_items.append(item_entry)
        if final_status == "success":
            summary_parts.append(f"{tool_name}: {item_entry['result']}")
        else:
            summary_parts.append(f"{tool_name} [{final_status}]: {item_entry['result']}")

    successful = max(0, len(normalized_items) - failed)
    if successful > 0:
        overall_status = "success"
    elif failed == no_valid and failed > 0:
        overall_status = "no_valid_result"
    elif failed > 0:
        overall_status = "error"
    else:
        overall_status = "success"

    payload = {
        "result": " | ".join(summary_parts) if summary_parts else "Parallel execution completed.",
        "items": normalized_items,
        "status": overall_status,
        "successful_items": successful,
        "failed_items": failed,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(serialized) <= max_chars or any(
        _should_preserve_full_result_for_tool(str(item.get("tool", "")))
        for item in normalized_items
        if isinstance(item, dict)
    ):
        return serialized, overall_status, payload
    return format_compact_observation(payload, tool_name="Parallel", max_chars=max_chars)
