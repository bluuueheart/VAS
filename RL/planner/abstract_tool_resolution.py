"""Helpers for converging abstract tool calls into executable tools."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from config.settings import settings


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _preferred_route_families_for_abstract_tool(abstract_tool: str) -> List[str]:
    normalized = _normalize_token(abstract_tool)
    route_family_aliases = {
        "query rewrite": ["query_rewrite"],
        "evidence retrieval": ["evidence_retrieval"],
        "direct evidence": ["direct_evidence"],
        "verification": ["verification"],
        "verify": ["verification"],
        "cross modal verify": ["verification"],
        "compute": ["compute"],
        "retrieve video frames": ["direct_evidence"],
        "retrieve video frame": ["direct_evidence"],
        "retrieve frames": ["direct_evidence"],
        "retrieve frame": ["direct_evidence"],
    }
    return list(route_family_aliases.get(normalized, []))


def _normalize_segment_field_alias(field: str) -> str:
    normalized = _normalize_token(str(field))
    if not normalized:
        return ""

    mapping = {
        "knowledge graph": "graph",
        "knowledge_graph": "graph",
        "knowledge-graph": "graph",
        "kg": "graph",
        "captions": "caption",
        "caption": "caption",
        "captionss": "caption",
        "asr": "transcript",
        "transcript": "transcript",
        "transcripts": "transcript",
        "frame": "frame_paths",
        "frames": "frame_paths",
        "image": "frame_paths",
        "images": "frame_paths",
        "timestamps": "frame_times",
        "timestamp": "frame_times",
        "metadata": "",
        "meta": "",
        "segment metadata": "",
        "segment_metadata": "",
    }
    if normalized in mapping:
        return mapping[normalized]

    tokens = set(normalized.split())
    if tokens & {
        "graph", "entity", "entities", "relation", "relations", "trend", "trends",
        "chart", "charts", "plot", "plots", "price", "prices", "value", "values",
        "percent", "percentage", "percentages", "change", "changes",
    }:
        return "graph"
    if tokens & {
        "caption", "captions", "scene", "visual", "description", "descriptions",
        "summary", "summaries",
    }:
        return "caption"
    if tokens & {
        "transcript", "transcripts", "subtitle", "subtitles", "speech", "spoken",
        "dialogue", "dialog", "utterance", "utterances",
    }:
        return "transcript"
    if tokens & {"frame", "frames", "image", "images", "screenshot", "screenshots"}:
        return "frame_paths"
    if tokens & {"timestamp", "timestamps", "time", "times"}:
        return "frame_times"
    return normalized.replace(" ", "_")


def build_abstract_signature(
    tool_name: str,
    args: Dict[str, Any],
    description: str = "",
) -> str:
    normalized_name = _normalize_token(tool_name)
    normalized_desc = _normalize_token(description)[:240]
    try:
        serialized_args = json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        serialized_args = str(args)
    return f"{normalized_name}|{normalized_desc}|{serialized_args}"


def extract_candidate_tools(raw_results: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_results, list):
        return []
    candidates: List[Dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        candidates.append(
            {
                "name": name,
                "description": str(item.get("description", "")).strip(),
                "parameters": item.get("parameters", {}) if isinstance(item.get("parameters"), dict) else {},
                "score": float(item.get("score", 0.0) or 0.0),
                "final_score": float(item.get("final_score", item.get("score", 0.0)) or 0.0),
                "tool_type": str(item.get("tool_type", "")).strip(),
                "tags": list(item.get("tags") or []),
                "route_family": str(item.get("route_family", "")).strip(),
                "route_reason": str(item.get("route_reason", "")).strip(),
                "route_score": float(item.get("route_score", 0.0) or 0.0),
                "novelty_score": float(item.get("novelty_score", 0.0) or 0.0),
                "recommended_next_tools": list(item.get("recommended_next_tools") or []),
                "route_plan": list(item.get("route_plan") or []),
            }
        )
    return candidates


def _score_candidate(
    candidate: Dict[str, Any],
    abstract_tool: str,
    description: str,
) -> float:
    target = _normalize_token(f"{abstract_tool} {description}")
    normalized_abstract_tool = _normalize_token(abstract_tool)
    cand_name = _normalize_token(candidate.get("name", ""))
    raw_cand_name = str(candidate.get("name", "")).strip()
    cand_desc = _normalize_token(candidate.get("description", ""))
    score = float(candidate.get("final_score", candidate.get("score", 0.0)) or 0.0)
    target_tokens = set(target.split())
    route_family = str(candidate.get("route_family", "")).strip().lower()

    if cand_name == _normalize_token(abstract_tool):
        score += 10.0
    if target_tokens.intersection({"search", "find", "locate", "retrieve", "segment", "timestamp"}):
        if route_family == "evidence_retrieval":
            score += 4.5
        elif route_family == "registry_search":
            score -= 0.5
        elif route_family == "external_web":
            score -= 4.5
    if target_tokens.intersection({"rewrite", "normalize", "expand", "decompose"}):
        if route_family == "query_rewrite":
            score += 4.0
    if target_tokens.intersection({"frame", "screen", "visual", "visible", "image", "text"}):
        if route_family == "direct_evidence":
            score += 4.0
        elif route_family == "external_web":
            score -= 5.0
        if raw_cand_name in {"Inspect_Frame", "OCR_Extract"}:
            score += 1.25
    if normalized_abstract_tool in {"retrieve video frames", "retrieve video frame", "retrieve frames", "retrieve frame"}:
        if raw_cand_name == "Inspect_Frame":
            score += 5.0
        elif raw_cand_name == "OCR_Extract" and not target_tokens.intersection({"text", "ocr", "label", "subtitle", "caption", "written"}):
            score -= 3.0
    if target_tokens.intersection({"observe", "progression", "motion", "moving", "action"}):
        if raw_cand_name == "Inspect_Frame":
            score += 1.75
        elif raw_cand_name == "OCR_Extract":
            score -= 1.25
    if target_tokens.intersection({
        "before", "after", "when", "while", "happened", "opened", "closed",
        "washed", "cleaned", "tidied", "took", "grabbed", "watched",
    }):
        if raw_cand_name == "Action_Recognition":
            score += 2.25
        elif raw_cand_name == "OCR_Extract":
            score -= 1.75
        elif raw_cand_name == "Entity_Track":
            score -= 0.85
    if target_tokens.intersection({"ocr", "text", "label", "subtitle", "caption", "written"}):
        if raw_cand_name == "OCR_Extract":
            score += 2.0
        elif raw_cand_name == "Action_Recognition":
            score -= 1.0
    if target_tokens.intersection({"track", "tracking", "trajectory", "appears", "visible"}):
        if raw_cand_name == "Entity_Track":
            score += 1.8
    if target_tokens.intersection({
        "clip", "video", "segment", "window", "overall", "holistic", "interaction",
        "interactions", "order", "sequence", "scene", "state", "compare", "comparison",
        "continuous", "navigation",
    }):
        if raw_cand_name == "Video_Clip_QA":
            score += 3.2
        elif raw_cand_name == "OCR_Extract":
            score -= 1.25
        elif raw_cand_name == "Entity_Track" and not target_tokens.intersection({"track", "tracking", "trajectory"}):
            score -= 0.6
    if target_tokens.intersection({"verify", "confirm", "consistency", "contradiction"}):
        if route_family == "verification":
            score += 4.0
    if target_tokens.intersection({"calculate", "count", "sum", "total", "compare"}):
        if route_family == "compute":
            score += 3.5
        elif raw_cand_name == "Video_Clip_QA":
            score += 1.0

    score += float(candidate.get("route_score", 0.0) or 0.0)
    score += float(candidate.get("novelty_score", 0.0) or 0.0)

    overlap = set(target.split()).intersection(set(f"{cand_name} {cand_desc}".split()))
    score += 0.15 * len(overlap)
    return score


def _normalize_field_aliases(fields: List[str]) -> List[str]:
    normalized: List[str] = []
    for field in fields:
        value = _normalize_segment_field_alias(field)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _coerce_field_list(fields: Any) -> List[str]:
    if isinstance(fields, list):
        return [str(item) for item in fields]
    if isinstance(fields, str):
        return [part for part in re.split(r"[\s,]+", fields) if part]
    return []


def _prune_params_to_schema(params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    properties = {}
    if isinstance(schema, dict):
        properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    if not properties:
        return dict(params)
    pruned: Dict[str, Any] = {}
    for key, value in params.items():
        if key in properties:
            pruned[key] = value
    return pruned


def _prepare_candidate_pool(
    candidates: List[Dict[str, Any]],
    target_tool_name: str = "",
    target_tool_schema: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = [dict(item) for item in (candidates or []) if isinstance(item, dict)]
    desired_name = str(target_tool_name or "").strip()
    schema_name = ""
    if isinstance(target_tool_schema, dict):
        schema_name = str(target_tool_schema.get("name", "")).strip()
    desired_name = desired_name or schema_name
    if not desired_name or not isinstance(target_tool_schema, dict):
        return pool
    if any(str(item.get("name", "")).strip() == desired_name for item in pool):
        return pool
    injected = {
        "name": desired_name,
        "description": str(target_tool_schema.get("description", "")).strip(),
        "parameters": target_tool_schema.get("parameters", {}) if isinstance(target_tool_schema.get("parameters"), dict) else {},
        "score": float(target_tool_schema.get("score", 0.0) or 0.0),
        "final_score": float(target_tool_schema.get("final_score", target_tool_schema.get("score", 0.0)) or 0.0),
        "tool_type": str(target_tool_schema.get("tool_type", "")).strip(),
        "tags": list(target_tool_schema.get("tags") or []),
        "route_family": str(target_tool_schema.get("route_family", "")).strip(),
        "route_reason": str(target_tool_schema.get("route_reason", "")).strip(),
        "route_score": float(target_tool_schema.get("route_score", 0.0) or 0.0),
        "novelty_score": float(target_tool_schema.get("novelty_score", 0.0) or 0.0),
        "recommended_next_tools": list(target_tool_schema.get("recommended_next_tools") or []),
        "route_plan": list(target_tool_schema.get("route_plan") or []),
    }
    return [injected] + pool


def _adapt_segment_retrieve_args(
    abstract_args: Dict[str, Any],
    abstract_tool: str,
    description: str,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    segment_id = (
        abstract_args.get("segment_id")
        or abstract_args.get("segment")
        or abstract_args.get("id")
        or abstract_args.get("segment_ref")
    )
    if segment_id not in (None, ""):
        params["segment_id"] = str(segment_id)

    if abstract_args.get("time_range") not in (None, ""):
        params["time_range"] = abstract_args.get("time_range")
    elif abstract_args.get("t_start") not in (None, "") and abstract_args.get("t_end") not in (None, ""):
        params["time_range"] = f"{abstract_args.get('t_start')}-{abstract_args.get('t_end')}"
    elif abstract_args.get("timestamp") not in (None, "") and abstract_args.get("duration") not in (None, ""):
        center = float(abstract_args["timestamp"])
        duration = max(0.0, float(abstract_args["duration"]))
        params["time_range"] = f"{max(0.0, center - duration / 2.0)}-{center + duration / 2.0}"

    fields = []
    requested_fields = abstract_args.get("fields")
    fields.extend(_coerce_field_list(requested_fields))
    target = _normalize_token(f"{abstract_tool} {description}")
    if "transcript" in target:
        fields.append("transcript")
    if "caption" in target or "visual" in target:
        fields.append("caption")
    if "graph" in target or "entity" in target:
        fields.append("graph")
    if "frame" in target:
        fields.extend(["frame_times", "frame_paths"])
    default_fields = ["caption", "transcript"]
    if bool(settings.graph_enable):
        default_fields.append("graph")
    normalized_fields = _normalize_field_aliases(fields or default_fields)
    if normalized_fields:
        params["fields"] = normalized_fields
    return params


def _adapt_semantic_retrieve_args(
    abstract_args: Dict[str, Any],
    abstract_tool: str,
    description: str,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    query = (
        abstract_args.get("query")
        or abstract_args.get("term")
        or abstract_args.get("keyword")
        or abstract_args.get("intent_description")
        or description
        or abstract_tool
    )
    params["query"] = str(query)

    if abstract_args.get("time_range") not in (None, ""):
        params["time_range"] = abstract_args.get("time_range")
    elif abstract_args.get("t_start") not in (None, "") and abstract_args.get("t_end") not in (None, ""):
        params["time_range"] = f"{abstract_args.get('t_start')}-{abstract_args.get('t_end')}"

    if abstract_args.get("top_k") not in (None, ""):
        params["top_k"] = abstract_args.get("top_k")
    return params


def _adapt_visual_probe_args(abstract_args: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in ("timestamp", "duration", "t_start", "t_end", "density", "query", "video", "video_path"):
        value = abstract_args.get(key)
        if value not in (None, ""):
            params[key] = value
    timestamp_range = abstract_args.get("timestamp_range") or abstract_args.get("search_window")
    if timestamp_range not in (None, ""):
        time_range = str(timestamp_range)
        parts = time_range.split("-", 1)
        if len(parts) == 2:
            params.setdefault("t_start", parts[0])
            params.setdefault("t_end", parts[1])
    if "time_range" in abstract_args and abstract_args.get("time_range") not in (None, ""):
        time_range = str(abstract_args["time_range"])
        parts = time_range.split("-", 1)
        if len(parts) == 2:
            params.setdefault("t_start", parts[0])
            params.setdefault("t_end", parts[1])
    num_frames = abstract_args.get("num_frames")
    if (
        num_frames not in (None, "")
        and params.get("density") in (None, "")
        and params.get("t_start") not in (None, "")
        and params.get("t_end") not in (None, "")
    ):
        try:
            frame_count = max(1, int(float(num_frames)))
            start = float(params["t_start"])
            end = float(params["t_end"])
            window = max(0.0, end - start)
            if frame_count <= 1 or window <= 0:
                params["density"] = 0.5
            else:
                params["density"] = max(window / float(frame_count), 1e-3)
        except Exception:
            params["density"] = 0.5
    segment_ref = (
        abstract_args.get("segment_ref")
        or abstract_args.get("segment_id")
        or abstract_args.get("segment")
        or abstract_args.get("id")
    )
    absolute_segment_timestamps = _resolve_segment_relative_timestamps(
        segment_ref=segment_ref,
        timestamps=abstract_args.get("time_windows"),
        video_hint=abstract_args.get("video") or abstract_args.get("video_path"),
    )
    if absolute_segment_timestamps:
        params["timestamp"] = absolute_segment_timestamps
    elif segment_ref not in (None, "") and params.get("timestamp") not in (None, ""):
        normalized_timestamps = _resolve_segment_relative_timestamps(
            segment_ref=segment_ref,
            timestamps=params.get("timestamp"),
            video_hint=abstract_args.get("video") or abstract_args.get("video_path"),
        )
        if normalized_timestamps:
            params["timestamp"] = normalized_timestamps if isinstance(params.get("timestamp"), list) else normalized_timestamps[0]
    if segment_ref not in (None, "") and params.get("t_start") not in (None, "") and params.get("t_end") not in (None, ""):
        normalized_starts = _resolve_segment_relative_timestamps(
            segment_ref=segment_ref,
            timestamps=params.get("t_start"),
            video_hint=abstract_args.get("video") or abstract_args.get("video_path"),
        )
        normalized_ends = _resolve_segment_relative_timestamps(
            segment_ref=segment_ref,
            timestamps=params.get("t_end"),
            video_hint=abstract_args.get("video") or abstract_args.get("video_path"),
        )
        if normalized_starts:
            params["t_start"] = normalized_starts if isinstance(params.get("t_start"), list) else normalized_starts[0]
        if normalized_ends:
            params["t_end"] = normalized_ends if isinstance(params.get("t_end"), list) else normalized_ends[0]
    elif segment_ref not in (None, "") and params.get("timestamp") in (None, "") and params.get("t_start") in (None, "") and params.get("t_end") in (None, ""):
        params.setdefault("timestamp", f"segment_id={segment_ref}")

    # Planner often emits timestamp + t_end when it actually means a window probe.
    if params.get("timestamp") not in (None, ""):
        if params.get("t_end") not in (None, "") and params.get("t_start") in (None, ""):
            params["t_start"] = params["timestamp"]
            params.pop("timestamp", None)
        elif params.get("t_start") not in (None, "") and params.get("t_end") in (None, ""):
            params["t_end"] = params["timestamp"]
            params.pop("timestamp", None)
        elif params.get("t_start") not in (None, "") and params.get("t_end") not in (None, ""):
            params.pop("timestamp", None)
    return params


def _parse_time_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    normalized_text = re.sub(
        r"\s*(seconds?|secs?|sec|s)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if not normalized_text:
        return None
    try:
        if ":" in normalized_text:
            parts = [float(p) for p in normalized_text.split(":")]
            if len(parts) == 3:
                h, m, s = parts
                return h * 3600 + m * 60 + s
            if len(parts) == 2:
                m, s = parts
                return m * 60 + s
        return float(normalized_text)
    except Exception:
        return None


def _segment_reference_candidates(segment_ref: Any) -> List[str]:
    text = str(segment_ref or "").strip()
    if not text:
        return []
    candidates = [text]
    for prefix in ("segment_ref=", "segment_id=", "raw_segment_id="):
        if text.lower().startswith(prefix):
            candidates.append(text.split("=", 1)[1].strip())
    match = re.match(r"^(?P<video>.+)_(?P<segment>\d+)$", text)
    if match:
        candidates.append(match.group("segment"))
        candidates.append(f"{match.group('video')}_{match.group('segment')}")
    deduped: List[str] = []
    for candidate in candidates:
        normalized = str(candidate or "").strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _resolve_segment_row(segment_ref: Any, video_hint: Any = None) -> Optional[Dict[str, Any]]:
    if segment_ref in (None, ""):
        return None
    try:
        from toolkit.impl.visual import _iter_processed_segments
    except Exception:
        return None
    rows = _iter_processed_segments(video_hint)
    candidates = _segment_reference_candidates(segment_ref)
    if not candidates:
        return None

    best_row: Optional[Dict[str, Any]] = None
    best_score = -1
    for row in rows:
        if not isinstance(row, dict):
            continue
        aliases = {
            str(alias).strip()
            for alias in (row.get("segment_aliases") or [])
            if str(alias).strip()
        }
        segment_id = str(row.get("segment_id", "")).strip()
        segment_ref_value = str(row.get("segment_ref", "")).strip()
        video_name = str(row.get("video", "")).strip()
        if segment_id:
            aliases.add(segment_id)
        if segment_ref_value:
            aliases.add(segment_ref_value)
        if video_name and segment_id:
            aliases.add(f"{video_name}_{segment_id}")

        for candidate in candidates:
            if candidate not in aliases:
                continue
            score = 3 if "_" in candidate or candidate.startswith("segment_") else 2
            if score > best_score:
                best_row = row
                best_score = score
    return best_row


def _resolve_segment_relative_timestamps(
    *,
    segment_ref: Any,
    timestamps: Any,
    video_hint: Any = None,
) -> List[float]:
    raw_items = timestamps if isinstance(timestamps, list) else ([timestamps] if timestamps not in (None, "") else [])
    parsed_items: List[float] = []
    for item in raw_items:
        parsed = _parse_time_seconds(item)
        if parsed is not None:
            parsed_items.append(parsed)
    if not parsed_items:
        return []

    row = _resolve_segment_row(segment_ref, video_hint=video_hint)
    if not isinstance(row, dict):
        return [round(value, 3) for value in parsed_items]

    target = row.get("time_parsed") if isinstance(row.get("time_parsed"), dict) else None
    if not isinstance(target, dict) or "start" not in target or "end" not in target:
        return [round(value, 3) for value in parsed_items]

    segment_start = float(target["start"])
    segment_end = float(target["end"])
    segment_duration = max(0.0, segment_end - segment_start)

    absolute_times: List[float] = []
    for value in parsed_items:
        absolute_value = value
        if 0.0 <= value <= segment_duration + 1e-6:
            absolute_value = segment_start + value
        absolute_times.append(round(float(absolute_value), 3))
    return absolute_times


def _adapt_action_recognition_args(
    abstract_args: Dict[str, Any],
    abstract_tool: str,
    description: str,
) -> Dict[str, Any]:
    visual_args = _adapt_visual_probe_args(abstract_args)
    params: Dict[str, Any] = {}
    action_description = (
        abstract_args.get("action_description")
        or abstract_args.get("query")
        or description
        or abstract_tool
    )
    params["action_description"] = str(action_description).strip()
    if abstract_args.get("time_range") not in (None, ""):
        params["time_range"] = abstract_args.get("time_range")
    elif visual_args.get("t_start") not in (None, "") and visual_args.get("t_end") not in (None, ""):
        params["time_range"] = f"{visual_args.get('t_start')}-{visual_args.get('t_end')}"
    elif visual_args.get("timestamp") not in (None, ""):
        params["time_range"] = visual_args.get("timestamp")
    return params


def _adapt_entity_track_args(
    abstract_args: Dict[str, Any],
    abstract_tool: str,
    description: str,
) -> Dict[str, Any]:
    visual_args = _adapt_visual_probe_args(abstract_args)
    params: Dict[str, Any] = {}
    entity_description = (
        abstract_args.get("visual_object_description")
        or abstract_args.get("query")
        or description
        or abstract_tool
    )
    params["visual_object_description"] = str(entity_description).strip()
    if abstract_args.get("time_range") not in (None, ""):
        params["time_range"] = abstract_args.get("time_range")
    elif visual_args.get("t_start") not in (None, "") and visual_args.get("t_end") not in (None, ""):
        params["time_range"] = f"{visual_args.get('t_start')}-{visual_args.get('t_end')}"
    elif visual_args.get("timestamp") not in (None, ""):
        params["time_range"] = visual_args.get("timestamp")
    return params


def _adapt_video_clip_qa_args(
    abstract_args: Dict[str, Any],
    abstract_tool: str,
    description: str,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    query = (
        abstract_args.get("query")
        or abstract_args.get("question")
        or abstract_args.get("prompt")
        or description
        or abstract_tool
    )
    params["query"] = str(query).strip()

    for key in ("video", "video_path"):
        value = abstract_args.get(key)
        if value not in (None, ""):
            params[key] = value

    segment_ref = (
        abstract_args.get("segment_ref")
        or abstract_args.get("segment_id")
        or abstract_args.get("segment")
        or abstract_args.get("id")
    )
    if segment_ref not in (None, ""):
        params["segment_id"] = str(segment_ref)

    raw_start = abstract_args.get("start_time", abstract_args.get("t_start"))
    raw_end = abstract_args.get("end_time", abstract_args.get("t_end"))
    if raw_start in (None, "") or raw_end in (None, ""):
        time_range = abstract_args.get("time_range")
        if time_range not in (None, ""):
            parts = str(time_range).split("-", 1)
            if len(parts) == 2:
                raw_start = raw_start if raw_start not in (None, "") else parts[0]
                raw_end = raw_end if raw_end not in (None, "") else parts[1]
    if raw_start in (None, "") or raw_end in (None, ""):
        timestamp = abstract_args.get("timestamp")
        duration = abstract_args.get("duration")
        if timestamp not in (None, "") and duration not in (None, ""):
            center = _parse_time_seconds(timestamp)
            half_duration = max(0.0, float(duration)) / 2.0
            if center is not None:
                raw_start = raw_start if raw_start not in (None, "") else round(center - half_duration, 3)
                raw_end = raw_end if raw_end not in (None, "") else round(center + half_duration, 3)

    if segment_ref not in (None, "") and raw_start not in (None, "") and raw_end not in (None, ""):
        normalized_starts = _resolve_segment_relative_timestamps(
            segment_ref=segment_ref,
            timestamps=raw_start,
            video_hint=abstract_args.get("video") or abstract_args.get("video_path"),
        )
        normalized_ends = _resolve_segment_relative_timestamps(
            segment_ref=segment_ref,
            timestamps=raw_end,
            video_hint=abstract_args.get("video") or abstract_args.get("video_path"),
        )
        if normalized_starts:
            raw_start = normalized_starts[0]
        if normalized_ends:
            raw_end = normalized_ends[0]

    if raw_start not in (None, ""):
        params["start_time"] = raw_start
    if raw_end not in (None, ""):
        params["end_time"] = raw_end
    return params


def _adapt_args_for_tool(
    tool_name: str,
    abstract_args: Dict[str, Any],
    abstract_tool: str,
    description: str,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if tool_name == "Segment_Retrieve":
        adapted = _adapt_segment_retrieve_args(abstract_args, abstract_tool, description)
    elif tool_name == "Semantic_Retrieve":
        adapted = _adapt_semantic_retrieve_args(abstract_args, abstract_tool, description)
    elif tool_name in {"Inspect_Frame", "OCR_Extract"}:
        adapted = _adapt_visual_probe_args(abstract_args)
    elif tool_name == "Action_Recognition":
        adapted = _adapt_action_recognition_args(abstract_args, abstract_tool, description)
    elif tool_name == "Entity_Track":
        adapted = _adapt_entity_track_args(abstract_args, abstract_tool, description)
    elif tool_name == "Video_Clip_QA":
        adapted = _adapt_video_clip_qa_args(abstract_args, abstract_tool, description)
    else:
        adapted = dict(abstract_args or {})
    return _prune_params_to_schema(adapted, schema or {})


def resolve_abstract_tool_call(
    *,
    abstract_tool: str,
    abstract_args: Dict[str, Any],
    description: str,
    candidates: List[Dict[str, Any]],
    target_tool_name: str = "",
    target_tool_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidate_pool = _prepare_candidate_pool(
        candidates,
        target_tool_name=target_tool_name,
        target_tool_schema=target_tool_schema,
    )
    if not candidate_pool:
        return {
            "resolved": False,
            "observation": {
                "status": "error",
                "result": f"No executable tools were discovered for abstract tool '{abstract_tool}'.",
                "candidate_tools": [],
            },
        }

    ranked = sorted(
        candidate_pool,
        key=lambda item: _score_candidate(item, abstract_tool=abstract_tool, description=description),
        reverse=True,
    )
    normalized_abstract_tool = _normalize_token(abstract_tool)
    desired_name = str(target_tool_name or "").strip()
    if not desired_name and isinstance(target_tool_schema, dict):
        desired_name = str(target_tool_schema.get("name", "")).strip()
    preferred_best: Optional[Dict[str, Any]] = None
    preferred_route_families = (
        _preferred_route_families_for_abstract_tool(abstract_tool)
        if not desired_name
        else []
    )
    if preferred_route_families:
        family_ranked = [
            candidate
            for candidate in ranked
            if str(candidate.get("route_family", "")).strip().lower() in preferred_route_families
        ]
        if family_ranked:
            preferred_best = family_ranked[0]
        else:
            family_text = ", ".join(preferred_route_families)
            next_allowed_tools = [item.get("name") for item in ranked[:5]]
            return {
                "resolved": False,
                "observation": {
                    "status": "no_valid_result",
                    "result": (
                        f"Abstract tool '{abstract_tool}' requires a route-family match ({family_text}), "
                        "but Tool_Search did not discover one. Choose another capability family instead of "
                        "silently switching to a mismatched executable tool."
                    ),
                    "candidate_tools": ranked[:5],
                    "next_allowed_tools": next_allowed_tools,
                },
            }
    best = preferred_best or ranked[0]
    if desired_name:
        for candidate in ranked:
            if str(candidate.get("name", "")).strip() == desired_name:
                best = candidate
                break
    elif normalized_abstract_tool:
        exact_name_matches = [
            candidate
            for candidate in ranked
            if _normalize_token(candidate.get("name", "")) == normalized_abstract_tool
        ]
        if not exact_name_matches:
            return {
                "resolved": False,
                "observation": {
                    "status": "no_valid_result",
                    "result": (
                        f"Planner emitted tool '{abstract_tool}', but Tool_Search did not discover an exact same-name "
                        "executable tool. Do not use fuzzy matching; route this action through subagent rewriting."
                    ),
                    "candidate_tools": ranked[:5],
                    "next_allowed_tools": [item.get("name") for item in ranked[:5]],
                    "requires_exact_name_match": True,
                },
            }
        best = exact_name_matches[0]
    best_name = str(best.get("name", "")).strip()

    if len(ranked) > 1 and preferred_best is None and not (desired_name and best_name == desired_name):
        best_score = _score_candidate(best, abstract_tool=abstract_tool, description=description)
        second_score = _score_candidate(ranked[1], abstract_tool=abstract_tool, description=description)
        best_route_family = str(best.get("route_family", "")).strip().lower()
        if best_score - second_score < 0.35 and best_route_family in {"other", "registry_search"}:
            return {
                "resolved": False,
                "observation": {
                    "status": "no_valid_result",
                    "result": (
                        f"Abstract tool '{abstract_tool}' must be rewritten using one of the discovered executable tools."
                    ),
                    "candidate_tools": ranked[:5],
                    "next_allowed_tools": [item.get("name") for item in ranked[:5]],
                },
            }

    best_schema = best.get("parameters", {}) if isinstance(best.get("parameters"), dict) else {}
    params = _adapt_args_for_tool(
        best_name,
        abstract_args,
        abstract_tool,
        description,
        schema=best_schema,
    )

    return {
        "resolved": True,
        "tool_name": best_name,
        "params": params,
        "candidate_tools": ranked[:5],
        "observation": {
            "result": f"Resolved abstract tool '{abstract_tool}' to executable tool '{best_name}'.",
            "resolved_tool": best_name,
            "target_tool": desired_name or best_name,
            "candidate_tools": ranked[:5],
        },
    }
