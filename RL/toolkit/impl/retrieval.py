"""Retrieval and graph exploration tools."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from config.device_config import detect_device, initialize_device, get_accelerator_snapshot
from config.settings import settings
from core.accelerator_queue import SharedAcceleratorQueueLock, resolve_local_accelerator_slot_key
from runtime.session import runtime_session
from core.sidecar_pool import (
    OpenAIEndpointPool,
    embeddings_api_marked_unhealthy,
    format_endpoint_list,
    mark_embeddings_api_unhealthy,
    request_pooling_embedding,
)
from toolkit.base import BaseTool, enrich_tool_result, infer_primary_tool_type

logger = logging.getLogger(__name__)


def _torch_module() -> Any:
    """Load torch only when a retrieval embedding operation actually runs.

    Tool discovery and the immutable registry baseline need retrieval schemas,
    not a tensor runtime.  Keeping this import lazy prevents the audit
    preflight subprocess from importing the several-gigabyte CUDA/Transformers
    stack before workers are admitted.
    """

    import torch

    return torch


def _transformer_model_classes() -> Tuple[Any, Any]:
    """Load Transformers model classes only for local embedding inference."""

    from transformers import AutoModel, AutoTokenizer

    return AutoModel, AutoTokenizer
_SEGMENT_COUNT_CACHE: Dict[str, int] = {}
_TOOL_SEARCH_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}
_TIMESTAMPED_TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)s\s*->\s*(?P<end>\d+(?:\.\d+)?)s\]\s*(?P<text>.*\S)\s*$",
    flags=re.MULTILINE,
)
_APPROXIMATE_EVIDENCE_RE = re.compile(
    r"\b(about|around|approx(?:\.|imately)?|roughly|nearly|almost|circa)\b|~",
    flags=re.IGNORECASE,
)


def _align_hidden_and_attention_mask(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    source: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch = _torch_module()
    aligned_hidden = hidden
    aligned_mask = attention_mask
    if aligned_hidden.shape[0] == aligned_mask.shape[0]:
        return aligned_hidden, aligned_mask

    common_length = min(int(aligned_hidden.shape[0]), int(aligned_mask.shape[0]))
    logger.warning(
        "Normalizing retrieval embedding tensors from %s: hidden=%s attention_mask=%s -> common_length=%d",
        source,
        tuple(aligned_hidden.shape),
        tuple(aligned_mask.shape),
        common_length,
    )
    if common_length <= 0:
        return aligned_hidden[:0], aligned_mask[:0]
    return aligned_hidden[:common_length], aligned_mask[:common_length]


def _preferred_text_embedding_dtype(device: str) -> torch.dtype:
    torch = _torch_module()
    if str(device).startswith("cpu"):
        return torch.float32
    if str(device).startswith("cuda"):
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def _resolve_model_float_dtype(model: Any) -> torch.dtype:
    torch = _torch_module()
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            for param in parameters():
                if torch.is_tensor(param) and param.is_floating_point():
                    return param.dtype
        except Exception:
            pass
    return torch.float32


def _infer_tool_type(tags: Optional[List[str]]) -> str:
    return infer_primary_tool_type(tags)


def _infer_tool_capability_tier(tags: Optional[List[str]]) -> str:
    tagset = {str(t).strip().lower() for t in (tags or [])}
    if {"debug_placeholder", "placeholder_debug", "placeholder"} & tagset:
        return "debug"
    if {"weak_proxy", "heuristic_proxy", "weak"} & tagset:
        return "weak"
    return "production"


def _is_evaluation_ready(tags: Optional[List[str]]) -> bool:
    return _infer_tool_capability_tier(tags) == "production"


def _tool_schema_signature(tool: BaseTool) -> str:
    payload = {
        "name": str(tool.name or "").strip(),
        "description": str(tool.description or "").strip(),
        "parameters": tool.parameters if isinstance(tool.parameters, dict) else {},
        "tags": list(tool.tags or []),
        "tool_type": _infer_tool_type(tool.tags),
        "capability_tier": _infer_tool_capability_tier(tool.tags),
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _tokenize_text(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", (text or "").lower()) if t]


def _collect_searchable_segment_text(payload: Any) -> str:
    collected: List[str] = []
    seen: set[str] = set()
    skip_keys = {
        "frame_paths",
        "image_paths",
        "image_evidence",
        "matched_segments",
        "frames",
    }

    def _visit(value: Any, *, parent_key: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key or "").strip() in skip_keys:
                    continue
                _visit(item, parent_key=str(key or "").strip())
            return
        if isinstance(value, (list, tuple)):
            for item in value[:64]:
                _visit(item, parent_key=parent_key)
            return
        if not isinstance(value, str):
            return

        text = value.strip()
        if not text or parent_key in {"time", "time_range"}:
            return
        if len(text) > 4000:
            text = text[:4000]
        if text in seen:
            return
        seen.add(text)
        collected.append(text)

    _visit(payload)
    return " ".join(collected).strip()


_SEMANTIC_RELEVANCE_STOPWORDS = {
    "a", "an", "and", "are", "be", "by", "can", "does", "doing", "for",
    "from", "he", "how", "in", "is", "it", "main", "of", "on", "or",
    "person", "show", "the", "their", "this", "to", "video", "what", "with",
    "woman", "man", "character", "thing", "scene", "there",
}
_SEMANTIC_TEXT_MODALITIES = {
    "caption", "description", "transcript", "asr", "text", "graph", "ocr",
}


def _semantic_relevance_diagnostic(
    *,
    query: Any,
    match: Dict[str, Any],
    segment_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Separate text-grounded retrieval from an unverified visual candidate.

    Dense image similarity is useful for localization, but it is not sufficient
    to cite a segment as the answer-bearing evidence. In particular, a query
    about a computer/hand can retrieve visually unrelated wildlife footage. A
    text hit or a strong text-modality hit remains usable; visual-only hits are
    returned as candidates that require Inspect_Frame/Video_Clip_QA.
    """
    query_tokens = {
        token
        for token in _tokenize_text(str(query or ""))
        if token not in _SEMANTIC_RELEVANCE_STOPWORDS and len(token) > 1
    }
    content_tokens = set(_tokenize_text(_collect_searchable_segment_text(segment_payload)))
    matched_tokens = sorted(query_tokens.intersection(content_tokens))
    lexical_overlap = len(matched_tokens) / max(1, len(query_tokens))
    similarity = float(match.get("similarity_score", 0.0) or 0.0)
    modalities = {
        str(value).strip().lower()
        for value in (match.get("matched_modalities") or [])
        if str(value).strip()
    }
    text_modality_hit = bool(modalities.intersection(_SEMANTIC_TEXT_MODALITIES))

    if lexical_overlap >= 0.25:
        label = "lexical_supported"
        usable = True
        evidence_strength = "text_grounded"
        needs_visual_verification = False
    elif text_modality_hit and similarity >= 0.42:
        label = "dense_text_supported"
        usable = True
        evidence_strength = "dense_text_grounded"
        needs_visual_verification = True
    elif similarity >= 0.55:
        label = "dense_visual_candidate"
        usable = False
        evidence_strength = "unverified_visual_candidate"
        needs_visual_verification = True
    else:
        label = "low_signal"
        usable = False
        evidence_strength = "none"
        needs_visual_verification = True

    return {
        "relevance_label": label,
        "relevance_usable": usable,
        "lexical_overlap": round(float(lexical_overlap), 4),
        "matched_tokens": matched_tokens,
        "similarity_score": round(similarity, 4),
        "matched_modalities": sorted(modalities),
        "evidence_strength": evidence_strength,
        "needs_visual_verification": needs_visual_verification,
        "recommended_next_tools": ["Inspect_Frame", "Video_Clip_QA"]
        if needs_visual_verification
        else [],
    }


def _time_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if ":" in text:
            parts = [float(p) for p in text.split(":")]
            if len(parts) == 3:
                h, m, s = parts
                return h * 3600 + m * 60 + s
            if len(parts) == 2:
                m, s = parts
                return m * 60 + s
        return float(text)
    except Exception:
        return None


def _parse_time_range(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        start = _time_to_seconds(value[0])
        end = _time_to_seconds(value[1])
        if start is not None and end is not None:
            return (min(start, end), max(start, end))
        return None
    if not value:
        return None
    text = str(value).strip()
    if "-" not in text:
        return None
    left, right = text.split("-", 1)
    start = _time_to_seconds(left)
    end = _time_to_seconds(right)
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


def _extract_query_time_prior(query: Any) -> Optional[Tuple[float, float]]:
    """Read the explicitly labelled benchmark localization prior, if present."""
    text = str(query or "")
    match = re.search(
        r"(?:reference\s+time(?:\s+window)?|time_reference|time\s+prior)"
        r"[^0-9]{0,120}"
        r"(\d{1,3}(?::\d{2}(?::\d{2}(?:\.\d+)?)?)?)\s*[-–]\s*"
        r"(\d{1,3}(?::\d{2}(?::\d{2}(?:\.\d+)?)?)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    start = _time_to_seconds(match.group(1))
    end = _time_to_seconds(match.group(2))
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


_SEMANTIC_FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "be", "by", "can", "does", "doing", "for",
    "from", "he", "how", "in", "is", "it", "main", "male", "of", "on",
    "or", "person", "show", "the", "their", "this", "to", "video", "what",
    "wear", "wearing", "with", "woman", "man", "character", "clothes", "clothing",
}


def _normalize_asr_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None

    start = _time_to_seconds(entry.get("start"))
    end = _time_to_seconds(entry.get("end"))
    if start is None:
        return None
    if end is None:
        end = start

    start, end = min(start, end), max(start, end)
    text = str(entry.get("text", "") or "").strip()
    if not text:
        return None

    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "time": f"{start:.2f}-{end:.2f}",
        "text": text,
    }


def _dedupe_asr_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[float, float, str]] = set()
    for entry in entries or []:
        normalized = _normalize_asr_entry(entry)
        if normalized is None:
            continue
        key = (
            float(normalized["start"]),
            float(normalized["end"]),
            str(normalized["text"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    deduped.sort(key=lambda item: (float(item["start"]), float(item["end"]), str(item["text"])))
    return deduped


def _parse_timestamped_transcript_entries(transcript: Any) -> List[Dict[str, Any]]:
    text = str(transcript or "").strip()
    if not text:
        return []

    entries: List[Dict[str, Any]] = []
    for match in _TIMESTAMPED_TRANSCRIPT_LINE_RE.finditer(text):
        entries.append(
            {
                "start": match.group("start"),
                "end": match.group("end"),
                "text": match.group("text"),
            }
        )
    return _dedupe_asr_entries(entries)


def _safe_json_loads(text: Any) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _extract_first_json_payload(text: Any) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    direct = _safe_json_loads(stripped)
    if direct is not None:
        return direct

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        parsed = _safe_json_loads(block)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[idx:])
            return parsed
        except Exception:
            continue
    return None


def _normalize_string_list(value: Any, *, limit: int = 8) -> List[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = []

    results: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = " ".join(str(item or "").split()).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        results.append(text)
        if len(results) >= limit:
            break
    return results


def _normalize_evidence_status(value: Any = "", citation: Any = "", evidence_status: Any = "") -> str:
    normalized_status = str(evidence_status or "").strip().lower()
    if normalized_status in {"verified", "exact"}:
        return "verified"
    if normalized_status in {"approximate", "missing"}:
        return normalized_status

    value_text = str(value or "").strip()
    citation_text = str(citation or "").strip()
    if not value_text and not citation_text:
        return "missing"
    if not value_text or not citation_text:
        return "approximate"
    if _APPROXIMATE_EVIDENCE_RE.search(f"{value_text} {citation_text}"):
        return "approximate"
    return "verified"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _normalize_evidence_items(raw_items: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_items, list):
        return []

    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "").strip() or f"slot_{idx}"
        value = str(item.get("value", "") or "").strip()
        citation = str(item.get("citation", "") or "").strip()
        status = _normalize_evidence_status(
            value=value,
            citation=citation,
            evidence_status=item.get("evidence_status", ""),
        )
        normalized.append(
            {
                "label": label,
                "value": value,
                "citation": citation,
                "evidence_status": status,
            }
        )
    return normalized


def _extract_evidence_items(payload: Any) -> List[Dict[str, str]]:
    if isinstance(payload, list):
        return _normalize_evidence_items(payload)
    if not isinstance(payload, dict):
        return []
    for key in ("evidence", "Evidence", "evidence_items", "evidenceSlots", "evidence_slots"):
        if key in payload:
            return _normalize_evidence_items(payload.get(key))
    return []


def _serialize_for_prompt(value: Any, *, max_chars: int = 12000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        serialized = value.strip()
    else:
        try:
            serialized = json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            serialized = str(value)
    serialized = serialized.strip()
    if len(serialized) <= max_chars:
        return serialized
    return serialized[:max_chars] + "\n...[truncated]"


def _build_evidence_verify_prompt(
    *,
    original_question: str,
    planner_context_text: str,
    evidence_items: List[Dict[str, str]],
    answer_draft: str,
    verification_prompt: str,
) -> str:
    evidence_text = _serialize_for_prompt(evidence_items, max_chars=8000) or "[]"
    context_text = planner_context_text or "[empty]"
    answer_text = answer_draft or "[empty]"
    extra_instruction = verification_prompt or (
        "Judge whether the evidence chain is complete, whether the planner has logic gaps or looping behavior, "
        "and whether finishing now would be grounded or overconfident."
    )
    return "\n\n".join(
        [
            "You are validating a planner's evidence chain for a local-video QA system.",
            "Use only the provided question, planner context, evidence list, and answer draft.",
            extra_instruction,
            "Return strict JSON with this schema only:",
            json.dumps(
                {
                    "chain_complete": False,
                    "can_finish": False,
                    "overconfident": False,
                    "has_logic_issue": False,
                    "loop_risk": False,
                    "confidence_level": "low|medium|high",
                    "logic_issues": ["short concrete issue"],
                    "missing_evidence": ["missing fact or citation"],
                    "recommendation": "one concise action-oriented recommendation",
                },
                ensure_ascii=False,
                indent=2,
            ),
            f"[Original Question]\n{original_question}",
            f"[Planner Context]\n{context_text}",
            f"[Evidence Items]\n{evidence_text}",
            f"[Current Answer Draft]\n{answer_text}",
        ]
    ).strip()


def _coarse_transcript_entry(transcript: Any, time_range: Any) -> Optional[Dict[str, Any]]:
    text = str(transcript or "").strip()
    parsed_range = _parse_time_range(time_range)
    if not text or parsed_range is None:
        return None
    start, end = parsed_range
    return _normalize_asr_entry({"start": start, "end": end, "text": text})


def _time_overlap_ratio(candidate: Optional[Tuple[float, float]], target: Optional[Tuple[float, float]]) -> float:
    if candidate is None or target is None:
        return 1.0
    c0, c1 = candidate
    t0, t1 = target
    inter = max(0.0, min(c1, t1) - max(c0, t0))
    if inter <= 0:
        return 0.0
    union = max(c1, t1) - min(c0, t0)
    return inter / union if union > 0 else 0.0


def _normalize_segment_fields(fields: Optional[List[str]]) -> List[str]:
    coerced_fields: List[str] = []
    if isinstance(fields, str):
        coerced_fields = [part for part in re.split(r"[\s,]+", fields) if part]
    else:
        coerced_fields = list(fields or [])
    allowed = {"caption", "transcript", "graph", "frame_times", "frame_paths"}

    def _normalize_field(field: Any) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(field or "").lower()).strip()
        if not normalized:
            return ""

        mapping = {
            "knowledge graph": "graph",
            "knowledge_graph": "graph",
            "knowledge-graph": "graph",
            "kg": "graph",
            "asr": "transcript",
            "caption": "caption",
            "captions": "caption",
            "transcript": "transcript",
            "transcripts": "transcript",
            "frame": "frame_paths",
            "frames": "frame_paths",
            "image": "frame_paths",
            "images": "frame_paths",
            "timestamp": "frame_times",
            "timestamps": "frame_times",
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

    normalized: List[str] = []
    for field in coerced_fields:
        key = _normalize_field(field)
        if key and key not in normalized:
            normalized.append(key)
    invalid = [field for field in normalized if field not in allowed]
    if invalid:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported Segment_Retrieve fields: {', '.join(invalid)}. "
            f"Allowed fields are: {allowed_text}."
        )
    return normalized


def _intent_looks_like_evidence_lookup(intent_description: str) -> bool:
    text = " ".join(str(intent_description or "").strip().lower().split())
    if not text:
        return False
    discovery_markers = (
        "which tool", "what tool", "find tool", "discover tool",
        "tool for", "tool capable", "registry", "available tool",
    )
    if any(marker in text for marker in discovery_markers):
        return False
    evidence_markers = (
        "segment", "timestamp", "time range", "time window", "frame", "frames",
        "caption", "transcript", "ocr", "screen", "video", "clip", "scene",
        "on-screen", "onscreen", "subtitle", "subtitle", "raw_segment_id",
        "shown", "visible", "appears", "appearing", "nearby window",
    )
    return any(marker in text for marker in evidence_markers)


def _iter_scoped_video_dirs(video_hint: Optional[str] = None):
    cache_dir = Path(settings.data_cache_dir) / "processed"
    if not cache_dir.exists():
        return

    scoped_names = runtime_session.resolve_video_scope(video_hint)
    for video_dir in cache_dir.iterdir():
        if not video_dir.is_dir():
            continue
        if scoped_names and video_dir.name not in scoped_names:
            continue
        yield video_dir


def _load_cached_full_asr_entries(video_dir: Path) -> List[Dict[str, Any]]:
    cache_path = Path(settings.data_cache_dir) / "_cache" / video_dir.name / "full_transcript_segments.json"
    if not cache_path.exists():
        return []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except Exception as exc:
        logger.warning("asr: failed to load %s: %s", cache_path, exc)
        return []

    raw_entries = payload.get("segments")
    if not isinstance(raw_entries, list):
        return []
    return _dedupe_asr_entries(list(raw_entries))


def _collect_processed_asr_entries(video_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    segments_path = video_dir / "segments.json"
    segments_payload: Dict[str, Any] = {}
    if segments_path.exists():
        try:
            with open(segments_path, "r", encoding="utf-8") as f:
                loaded = json.load(f) or {}
            if isinstance(loaded, dict):
                segments_payload = loaded
        except Exception as exc:
            logger.warning("asr: failed to load %s: %s", segments_path, exc)

    for segment in (segments_payload or {}).values():
        if not isinstance(segment, dict):
            continue
        transcript = segment.get("transcript", "")
        parsed_entries = _parse_timestamped_transcript_entries(transcript)
        if parsed_entries:
            entries.extend(parsed_entries)
            continue
        coarse_entry = _coarse_transcript_entry(transcript, segment.get("time"))
        if coarse_entry is not None:
            entries.append(coarse_entry)

    transcripts_path = video_dir / "transcripts.json"
    if transcripts_path.exists():
        try:
            with open(transcripts_path, "r", encoding="utf-8") as f:
                transcripts_payload = json.load(f) or {}
        except Exception as exc:
            logger.warning("asr: failed to load %s: %s", transcripts_path, exc)
            transcripts_payload = {}

        if isinstance(transcripts_payload, dict):
            for sid, transcript in transcripts_payload.items():
                parsed_entries = _parse_timestamped_transcript_entries(transcript)
                if parsed_entries:
                    entries.extend(parsed_entries)
                    continue
                segment = segments_payload.get(str(sid), {}) if isinstance(segments_payload, dict) else {}
                coarse_entry = _coarse_transcript_entry(
                    transcript,
                    segment.get("time") if isinstance(segment, dict) else None,
                )
                if coarse_entry is not None:
                    entries.append(coarse_entry)

    return _dedupe_asr_entries(entries)


def _segment_count_for_video_dir(video_dir: Path) -> int:
    segments_path = video_dir / "segments.json"
    cache_key = str(segments_path.resolve())
    cached = _SEGMENT_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        with open(segments_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        count = len(payload) if isinstance(payload, dict) else 0
    except Exception:
        count = 0
    _SEGMENT_COUNT_CACHE[cache_key] = count
    return count


def _scoped_segment_count(video_hint: Optional[str] = None) -> Optional[int]:
    if not video_hint:
        active_count = runtime_session.get_active_segment_count()
        if active_count is not None:
            return active_count
    scoped_dirs = list(_iter_scoped_video_dirs(video_hint))
    if not scoped_dirs:
        return None
    return sum(_segment_count_for_video_dir(video_dir) for video_dir in scoped_dirs)


def _is_sparse_segment_scope(video_hint: Optional[str] = None) -> tuple[bool, Optional[int]]:
    segment_count = _scoped_segment_count(video_hint)
    if segment_count is None:
        return False, None
    threshold = max(1, int(getattr(settings, "sparse_video_segment_threshold", 3) or 3))
    return segment_count <= threshold, segment_count


def _sparse_scope_hidden_tool_names() -> set[str]:
    configured = getattr(settings, "sparse_video_hidden_tool_names", ()) or ()
    return {
        str(name).strip()
        for name in configured
        if str(name).strip()
    }


def get_sparse_scope_hidden_tool_payload(
    tool_name: str,
    *,
    video_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    normalized_name = str(tool_name or "").strip()
    if not normalized_name or normalized_name not in _sparse_scope_hidden_tool_names():
        return None

    sparse_scope, segment_count = _is_sparse_segment_scope(video_hint)
    if not sparse_scope:
        return None

    resolved_scope = sorted(runtime_session.resolve_video_scope(video_hint))
    segment_total = segment_count if segment_count is not None else "unknown"
    return enrich_tool_result(
        {
            "tool": normalized_name,
            "resolved_video_scope": resolved_scope,
            "segment_count": segment_count,
            "status": "no_valid_result",
            "result": (
                f"{normalized_name} is hidden for the current local video scope because only {segment_total} segment(s) "
                "are available. In this sparse scope it should be treated as unavailable, because grounding already "
                "contains the same segment-level evidence. Use direct evidence or verification tools instead."
            ),
        },
        tool_name=normalized_name,
    )


def _tool_route_family(name: str, tags: Optional[List[str]] = None) -> str:
    normalized_name = str(name or "").strip()
    tagset = {str(tag).strip().lower() for tag in (tags or [])}
    if normalized_name == "Tool_Search":
        return "registry_search"
    if normalized_name == "Web_Search":
        return "external_web"
    if normalized_name == "Query_Rewrite" or "rewrite" in tagset:
        return "query_rewrite"
    if normalized_name == "Cross_Modal_Verify" or "verification" in tagset or "crossmodal" in tagset:
        return "verification"
    if normalized_name == "Python_Executor" or "compute" in tagset:
        return "compute"
    if {"visual", "ocr", "audio"} & tagset:
        return "direct_evidence"
    if "retrieval" in tagset or "graph" in tagset or normalized_name in {
        "Semantic_Retrieve", "Keyword_Retrieve", "Segment_Retrieve", "Graph_Search", "Clean_Results",
    }:
        return "evidence_retrieval"
    return "other"


def _intent_requests_tool_discovery(intent_description: str) -> bool:
    text = " ".join(str(intent_description or "").strip().lower().split())
    if not text:
        return False
    markers = (
        "which tool", "what tool", "find tool", "discover tool",
        "tool for", "tool capable", "available tool", "registry",
    )
    return any(marker in text for marker in markers)


def _extract_declared_task_type(intent_description: str) -> str:
    text = str(intent_description or "")
    match = re.search(r"task\s*type\s*:\s*([^\n|]+)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\[mvbench\s*task\s*type\]\s*([^\n|]+)", text, flags=re.IGNORECASE)
    return str(match.group(1)).strip() if match else ""


def _intent_route_features(intent_description: str) -> Dict[str, bool]:
    text = " ".join(str(intent_description or "").strip().lower().split())
    declared_task_type = _extract_declared_task_type(intent_description).lower()
    count_reasoning = any(token in text for token in ("how many", "count", "times", "number of"))
    action_localization = any(
        token in text
        for token in ("before", "after", "when", "which segment", "what happened", "what happens next", "locate", "localize")
    )
    object_existence = any(
        token in text
        for token in ("is there", "are there", "does the", "present", "visible", "appear", "shown")
    )
    egocentric_navigation = any(
        token in text
        for token in (
            "camera wearer",
            "first person",
            "egocentric",
            "turn left",
            "turn right",
            "go straight",
            "walk through",
            "move forward",
            "navigation",
        )
    )
    if "object existence" in declared_task_type:
        object_existence = True
    if "action localization" in declared_task_type:
        action_localization = True
    if any(token in declared_task_type for token in ("action count", "moving count")):
        count_reasoning = True
    if "egocentric navigation" in declared_task_type:
        egocentric_navigation = True
    return {
        "tool_discovery": _intent_requests_tool_discovery(text),
        "evidence_lookup": _intent_looks_like_evidence_lookup(text),
        "external_knowledge": any(
            marker in text
            for marker in (
                "web", "internet", "online", "news", "latest", "current", "background knowledge", "external",
            )
        ),
        "exact_value": any(token in text for token in ("exact", "precise", "verbatim", "price", "number", "label", "id", "score", "value")),
        "visual_probe": any(token in text for token in ("frame", "screen", "visible", "onscreen", "appearance", "image", "object", "scene")),
        "temporal_probe": any(token in text for token in ("segment", "timestamp", "time range", "time window", "before", "after", "when", "window")),
        "verification": any(token in text for token in ("verify", "confirm", "cross modal", "consisten", "contradict", "check whether")),
        "rewrite": any(token in text for token in ("rewrite", "decompose", "normalize", "expand", "clarify")),
        "compute": any(token in text for token in ("calculate", "count", "total", "sum", "ratio", "compare")),
        "count_reasoning": count_reasoning,
        "action_localization": action_localization,
        "object_existence": object_existence,
        "egocentric_navigation": egocentric_navigation,
        "needs_meta_companion": count_reasoning or action_localization or any(
            token in declared_task_type
            for token in ("action count", "moving count", "object interaction", "state change")
        ),
        "declared_mvbench_task": bool(declared_task_type),
    }


def _intent_requires_grounded_local_evidence(features: Dict[str, bool]) -> bool:
    if not isinstance(features, dict):
        return False
    if features.get("tool_discovery") or features.get("external_knowledge"):
        return False
    return any(
        bool(features.get(key))
        for key in (
            "evidence_lookup",
            "visual_probe",
            "temporal_probe",
            "exact_value",
            "verification",
            "count_reasoning",
            "action_localization",
            "object_existence",
            "egocentric_navigation",
            "declared_mvbench_task",
        )
    )


def _tool_specific_route_bonus(tool_name: str, intent_description: str) -> Tuple[float, List[str]]:
    text = " ".join(str(intent_description or "").strip().lower().split())
    if not text:
        return 0.0, []

    action_markers = (
        "before", "after", "when", "while", "happened", "happen", "doing",
        "opened", "closed", "washed", "cleaned", "tidied", "tidying",
        "took", "picked", "grabbed", "watched", "read", "sat", "stood",
    )
    text_markers = (
        "ocr", "text", "subtitle", "caption", "label", "written", "screen text", "onscreen text",
    )
    tracking_markers = (
        "track", "tracking", "trajectory", "where is", "where does", "appears", "visible object",
    )
    clip_reasoning_markers = (
        "clip", "video clip", "video segment", "segment", "window", "short video",
        "whole clip", "entire clip", "entire segment", "overall", "holistic",
        "what is happening", "what happened", "what happens", "describe the clip",
        "summarize the clip", "interaction", "interactions", "temporal order",
        "sequence", "order", "scene change", "scene changes", "state change",
        "state changes", "compare", "comparison", "camera wearer", "first person",
    )

    is_action_query = any(marker in text for marker in action_markers)
    is_text_query = any(marker in text for marker in text_markers)
    is_tracking_query = any(marker in text for marker in tracking_markers)
    is_clip_reasoning_query = any(marker in text for marker in clip_reasoning_markers)
    normalized_tool = str(tool_name or "").strip()

    bonus = 0.0
    reasons: List[str] = []
    if is_action_query:
        if normalized_tool == "Action_Recognition":
            bonus += 2.2
            reasons.append("intent describes an action/event transition")
        elif normalized_tool == "OCR_Extract":
            bonus -= 1.75
            reasons.append("action query should not default to OCR")
        elif normalized_tool == "Entity_Track" and not is_tracking_query:
            bonus -= 0.85
            reasons.append("action query is weaker fit for entity tracking")
    if is_text_query:
        if normalized_tool == "OCR_Extract":
            bonus += 2.0
            reasons.append("intent explicitly asks for text extraction")
        elif normalized_tool == "Action_Recognition":
            bonus -= 1.0
            reasons.append("text extraction intent is weaker fit for action recognition")
    if is_tracking_query and normalized_tool == "Entity_Track":
        bonus += 1.8
        reasons.append("intent explicitly asks for entity tracking")
    if is_clip_reasoning_query:
        if normalized_tool == "Video_Clip_QA":
            bonus += 2.8
            reasons.append("intent asks for holistic clip-level video understanding")
        elif normalized_tool == "OCR_Extract":
            bonus -= 1.1
            reasons.append("holistic clip reasoning is weaker fit for OCR-first routing")
        elif normalized_tool == "Entity_Track" and not is_tracking_query:
            bonus -= 0.6
            reasons.append("holistic clip reasoning is weaker fit for narrow tracking")

    return bonus, reasons


def _route_candidate_bonus(
    schema: Dict[str, Any],
    *,
    intent_description: str,
    recent_tools: List[str],
) -> Dict[str, Any]:
    route_family = _tool_route_family(
        str(schema.get("name", "")).strip(),
        list(schema.get("tags") or []),
    )
    tool_name = str(schema.get("name", "")).strip()
    tool_type = str(schema.get("tool_type", "")).strip() or _infer_tool_type(schema.get("tags"))
    capability_tier = str(schema.get("capability_tier", "")).strip() or _infer_tool_capability_tier(schema.get("tags"))
    tagset = {str(tag).strip().lower() for tag in (schema.get("tags") or [])}
    features = _intent_route_features(intent_description)
    grounded_local_evidence = _intent_requires_grounded_local_evidence(features)
    route_score = 0.0
    reasons: List[str] = []
    sparse_scope, segment_count = _is_sparse_segment_scope()

    if features["tool_discovery"]:
        if route_family == "registry_search":
            route_score += 2.5
            reasons.append("intent explicitly asks for tool discovery")
    else:
        if route_family == "registry_search":
            route_score -= 1.0
            reasons.append("prefer executable evidence tools over registry recursion")

    if features["rewrite"] and route_family == "query_rewrite":
        route_score += 1.75
        reasons.append("intent benefits from rewrite/decomposition")
    if features["evidence_lookup"] and route_family == "evidence_retrieval":
        route_score += 1.75
        reasons.append("intent targets local video evidence retrieval")
    if features["visual_probe"] and route_family == "direct_evidence":
        route_score += 1.35
        reasons.append("intent mentions direct visual probing")
    if features["exact_value"] and route_family in {"evidence_retrieval", "direct_evidence"}:
        route_score += 0.85
        reasons.append("intent requires exact cited evidence")
    if features["verification"] and route_family == "verification":
        route_score += 1.5
        reasons.append("intent asks for consistency/verification")
    if features["compute"] and route_family == "compute":
        route_score += 1.25
        reasons.append("intent needs downstream computation")
    if route_family == "compute" and grounded_local_evidence:
        route_score -= 1.9
        reasons.append("local-video tasks should gather grounded evidence before computation")
        if tool_name == "Python_Executor":
            route_score -= 0.85
            reasons.append("Python_Executor is a late-stage compute tool, not an evidence search tool")
    if features["temporal_probe"] and route_family == "evidence_retrieval":
        route_score += 0.7
        reasons.append("intent includes temporal/segment cues")
    if features["count_reasoning"]:
        if tool_type == "meta" and tagset & {"count", "aggregation", "group", "time", "merge", "window"}:
            route_score += 2.2
            reasons.append("count-style query benefits from meta aggregation/time tools")
        elif route_family in {"direct_evidence", "evidence_retrieval"}:
            route_score += 1.1
            reasons.append("count-style query still needs grounded evidence collection")
    if features["action_localization"]:
        if tool_name == "Video_Clip_QA":
            route_score += 2.4
            reasons.append("action localization benefits from clip-level video inspection")
        elif route_family in {"evidence_retrieval", "direct_evidence"}:
            route_score += 1.35
            reasons.append("action localization needs time-window refinement")
        if tool_type == "meta" and tagset & {"time", "merge", "window"}:
            route_score += 1.5
            reasons.append("action localization benefits from time-range meta tools")
    if features["object_existence"]:
        if tool_name in {"Inspect_Frame", "Video_Clip_QA"} or route_family == "direct_evidence":
            route_score += 1.9
            reasons.append("object existence should be confirmed visually")
        elif tool_type == "meta":
            route_score -= 0.6
            reasons.append("object existence is weaker fit for pure meta tools")
    if features["egocentric_navigation"]:
        if tool_name == "Video_Clip_QA":
            route_score += 2.8
            reasons.append("egocentric navigation benefits from continuous clip-level evidence")
        elif route_family == "direct_evidence":
            route_score += 1.5
            reasons.append("egocentric navigation relies on visual continuity")
        elif tool_name == "OCR_Extract":
            route_score -= 0.8
            reasons.append("egocentric navigation is weaker fit for OCR-first routing")
    broad_visual_feature_count = sum(
        1
        for enabled in (
            features.get("visual_probe"),
            features.get("temporal_probe"),
            features.get("action_localization"),
            features.get("object_existence"),
            features.get("egocentric_navigation"),
            features.get("verification"),
        )
        if enabled
    )
    if tool_name == "Video_Clip_QA" and broad_visual_feature_count >= 2:
        route_score += 1.6
        reasons.append("intent mixes multiple visual/temporal needs suited to holistic clip QA")
    if tool_name == "Video_Clip_QA" and features["count_reasoning"]:
        route_score += 0.9
        reasons.append("count-style query may still need continuous clip context before aggregation")
    if route_family == "external_web":
        if features["external_knowledge"]:
            route_score += 1.5
            reasons.append("intent explicitly asks for external or current information")
        else:
            route_score -= 4.5
            reasons.append("local-video evidence task should not route to web search")
    if capability_tier == "weak":
        route_score -= 1.75
        reasons.append("tool is currently graded as weak proxy evidence")
    elif capability_tier == "debug":
        route_score -= 4.0
        reasons.append("tool is currently graded as debug/placeholder only")
    if sparse_scope:
        if tool_name in {"Semantic_Retrieve", "Keyword_Retrieve", "Graph_Search"}:
            route_score -= 4.0
            reasons.append(
                f"sparse local video scope ({segment_count} segment(s)) makes segment-level retrieval low-yield"
            )
        elif route_family in {"direct_evidence", "verification"}:
            route_score += 1.75
            reasons.append(
                f"sparse local video scope ({segment_count} segment(s)) favors direct evidence or verification"
            )
        elif route_family == "compute" and grounded_local_evidence:
            route_score -= 0.75
            reasons.append(
                f"sparse local video scope ({segment_count} segment(s)) should probe evidence directly before compute"
            )
        if tool_name in _sparse_scope_hidden_tool_names():
            route_score -= 100.0
            reasons.append(
                f"sparse local video scope ({segment_count} segment(s)) hides this segment-level retrieval tool"
            )

    normalized_recent = [str(name).strip() for name in recent_tools if str(name).strip()]
    recent_count = normalized_recent.count(tool_name)
    novelty_score = 0.35 if recent_count == 0 else max(-0.45, -0.15 * recent_count)
    if recent_count == 0:
        reasons.append("promotes tool diversity within this query")
    elif recent_count > 0:
        reasons.append("penalizes recently repeated tool usage")

    tool_bonus, tool_bonus_reasons = _tool_specific_route_bonus(tool_name, intent_description)
    if tool_bonus:
        route_score += tool_bonus
        reasons.extend(tool_bonus_reasons[:2])

    route_plan = _build_route_plan(
        route_family=route_family,
        features=features,
    )
    return {
        "route_family": route_family,
        "route_reason": "; ".join(reasons[:3]) if reasons else "semantic capability match",
        "route_score": round(route_score, 4),
        "novelty_score": round(novelty_score, 4),
        "route_plan": route_plan,
        "capability_tier": capability_tier,
        "evaluation_ready": _is_evaluation_ready(schema.get("tags")),
    }


def _build_route_plan(
    *,
    route_family: str,
    features: Dict[str, bool],
) -> List[str]:
    plan: List[str] = []

    def _add(step: str) -> None:
        normalized = str(step or "").strip()
        if normalized and normalized not in plan:
            plan.append(normalized)

    if features.get("rewrite") or route_family == "query_rewrite":
        _add("query_rewrite")

    if route_family == "evidence_retrieval" or features.get("evidence_lookup") or features.get("temporal_probe"):
        _add("evidence_retrieval")

    if route_family == "direct_evidence" or features.get("visual_probe") or features.get("exact_value"):
        _add("direct_evidence")

    if route_family == "verification" or features.get("verification") or features.get("exact_value"):
        _add("verification")

    if route_family == "compute" or features.get("compute"):
        _add("compute")

    if features.get("count_reasoning"):
        _add("time")
        _add("aggregation")

    if features.get("action_localization"):
        _add("time_refinement")

    if features.get("egocentric_navigation"):
        _add("continuous_visual_trace")

    if not plan:
        _add(route_family or "other")

    return plan


def _recommended_next_tools_for_item(
    current: Dict[str, Any],
    selected: List[Dict[str, Any]],
) -> List[str]:
    current_name = str(current.get("name", "")).strip()
    current_family = str(current.get("route_family", "other")).strip() or "other"
    desired_families = [
        str(step).strip()
        for step in (current.get("route_plan") or [])
        if str(step).strip() and str(step).strip() != current_family
    ]

    ranked: List[tuple[int, int, float, str]] = []
    for item in selected:
        name = str(item.get("name", "")).strip()
        if not name or name == current_name:
            continue
        family = str(item.get("route_family", "other")).strip() or "other"
        if family in desired_families:
            family_rank = desired_families.index(family)
        elif family != current_family:
            family_rank = len(desired_families)
        else:
            family_rank = len(desired_families) + 1
        score = -float(item.get("final_score", item.get("score", 0.0)) or 0.0)
        ranked.append((family_rank, 0 if family != current_family else 1, score, name))

    ranked.sort()
    ordered: List[str] = []
    for _, _, _, name in ranked:
        if name not in ordered:
            ordered.append(name)
        if len(ordered) >= 3:
            break
    return ordered


def _select_diverse_candidates(candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    remaining = [dict(item) for item in candidates]
    selected: List[Dict[str, Any]] = []
    family_counts: Dict[str, int] = {}
    while remaining and len(selected) < max(1, top_k):
        def _selection_score(item: Dict[str, Any]) -> float:
            family = str(item.get("route_family", "other"))
            diversity_bonus = 0.3 if family_counts.get(family, 0) == 0 else max(-0.4, -0.12 * family_counts[family])
            return float(item.get("final_score", item.get("score", 0.0)) or 0.0) + diversity_bonus

        best = max(remaining, key=_selection_score)
        selected.append(best)
        family = str(best.get("route_family", "other"))
        family_counts[family] = family_counts.get(family, 0) + 1
        remaining.remove(best)

    for item in selected:
        item["recommended_next_tools"] = _recommended_next_tools_for_item(item, selected)
    return selected


def _tool_schema_text(tool: BaseTool) -> str:
    return (
        f"name: {tool.name}\n"
        f"description: {tool.description}\n"
        f"tags: {', '.join(tool.tags or [])}\n"
        f"parameters: {json.dumps(tool.parameters, ensure_ascii=False, sort_keys=True)}"
    )


def _get_tool_search_schema_records(
    *,
    model_path: str,
    tools: List[BaseTool],
) -> List[Dict[str, Any]]:
    normalized_model_path = str(model_path or "").strip()
    cache_bucket = _TOOL_SEARCH_SCHEMA_CACHE.setdefault(normalized_model_path, {})
    registry_signature = "|".join(
        sorted(
            _tool_schema_signature(tool)
            for tool in tools
        )
    )
    cache_hit = cache_bucket.get("registry_signature") == registry_signature
    runtime_session.record_cache("tool_search_schema_index", cache_hit)
    if cache_hit and isinstance(cache_bucket.get("records"), list):
        return list(cache_bucket.get("records") or [])

    embedder = _get_tool_search_embedder(model_path)
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []
    for tool in tools:
        tags = list(tool.tags or [])
        schema = tool.get_schema()
        schema["tags"] = tags
        schema["tool_type"] = _infer_tool_type(tags)
        schema["capability_tier"] = _infer_tool_capability_tier(tags)
        schema["evaluation_ready"] = _is_evaluation_ready(tags)
        schema["tool_signature"] = _tool_schema_signature(tool)
        records.append(
            {
                "schema": schema,
                "embedding": embedder.encode(_tool_schema_text(tool)),
            }
        )
    runtime_session.record_timing("tool_search.schema_index_seconds", time.perf_counter() - started)
    cache_bucket["registry_signature"] = registry_signature
    cache_bucket["records"] = records
    return list(records)


def _score_tool_search_records(
    *,
    records: List[Dict[str, Any]],
    intent_description: str,
    recent_tools: List[str],
    query_vec: torch.Tensor,
) -> List[Dict[str, Any]]:
    torch = _torch_module()
    scored: List[Dict[str, Any]] = []
    for record in records:
        schema = dict(record.get("schema") or {})
        tool_vec = record.get("embedding")
        if not torch.is_tensor(tool_vec) or query_vec.numel() == 0 or tool_vec.numel() == 0:
            score = -1.0
        else:
            score = float(torch.dot(query_vec, tool_vec).item())
        schema["score"] = score
        route_meta = _route_candidate_bonus(
            schema,
            intent_description=intent_description,
            recent_tools=recent_tools,
        )
        schema.update(route_meta)
        schema["final_score"] = round(
            float(score)
            + float(route_meta.get("route_score", 0.0))
            + float(route_meta.get("novelty_score", 0.0)),
            4,
        )
        scored.append(schema)
    scored.sort(
        key=lambda item: float(item.get("final_score", item.get("score", 0.0)) or 0.0),
        reverse=True,
    )
    return scored


def _stage_two_meta_budget(tool_type: str, features: Dict[str, bool], top_k: int) -> int:
    if tool_type == "meta":
        return max(1, int(top_k))
    if tool_type != "all":
        return 0
    if not features.get("needs_meta_companion"):
        return 0
    return min(max(1, int(top_k) // 2), 3)


def _interleave_tool_search_stages(
    *,
    primary_items: List[Dict[str, Any]],
    meta_items: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in primary_items:
        name = str(item.get("name", "")).strip()
        if name and name not in seen:
            item["route_stage"] = "primary_base"
            merged.append(item)
            seen.add(name)
    for item in meta_items:
        name = str(item.get("name", "")).strip()
        if name and name not in seen:
            item["route_stage"] = "meta_companion"
            merged.append(item)
            seen.add(name)
    return merged[: max(1, top_k)]


def _explicit_tool_name_from_intent(intent_description: Any) -> str:
    """Extract a planner-requested executable name from a routing intent."""
    text = str(intent_description or "")
    match = re.search(
        r"\b(?:abstract[_ ]tool|target[_ ]tool|tool)\s*:\s*([A-Za-z][A-Za-z0-9_]*)",
        text,
        flags=re.IGNORECASE,
    )
    return str(match.group(1) or "").strip() if match else ""


def _inject_explicit_tool_candidate(
    *,
    results: List[Dict[str, Any]],
    all_tools: List[BaseTool],
    intent_description: Any,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Keep an explicit tool request from being lost below semantic top-k."""
    requested_name = _explicit_tool_name_from_intent(intent_description)
    if not requested_name:
        return results
    normalized_requested = requested_name.lower()
    matching_tool = next(
        (
            tool
            for tool in all_tools
            if str(getattr(tool, "name", "") or "").strip().lower() == normalized_requested
        ),
        None,
    )
    if matching_tool is None:
        return results
    if any(str(item.get("name", "")).strip().lower() == normalized_requested for item in results):
        return results

    schema = matching_tool.get_schema()
    schema["tags"] = list(matching_tool.tags or [])
    schema["tool_type"] = _infer_tool_type(matching_tool.tags)
    schema["capability_tier"] = _infer_tool_capability_tier(matching_tool.tags)
    schema["evaluation_ready"] = _is_evaluation_ready(matching_tool.tags)
    schema["score"] = max(float(schema.get("score", 0.0) or 0.0), 1.0)
    schema["final_score"] = max(float(schema.get("final_score", 0.0) or 0.0), 1.0)
    schema["route_reason"] = f"exact_name_requested:{requested_name}->{matching_tool.name}"
    schema["route_stage"] = "exact_name"
    return [schema] + list(results)[: max(0, int(top_k) - 1)]


def _get_cached_vector_store(video_dir: Path):
    """Load a complete dense index; repair missing/corrupt indexes with embeddings."""
    from filelock import FileLock
    from preprocessing.storage.vector_store import VideoSegmentVectorStore
    import tempfile
    import shutil

    vectors_path = video_dir / "vectors"
    require_video = os.environ.get('TOOLSTAR_SEMANTIC_INDEX_VIDEO', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
    segments = _load_segments_payload(video_dir)
    if not segments:
        raise ValueError("Cannot index a video without persisted segments")
    with FileLock(str(video_dir / '.dense_index.lock')):
        metadata_path = vectors_path / 'metadata.json'
        metadata_bytes = metadata_path.read_bytes() if metadata_path.is_file() else b''
        cache_key = str(vectors_path.resolve()) + ':' + hashlib.sha256(metadata_bytes + str(require_video).encode()).hexdigest()
        cached = runtime_session.get_cached_vector_store(cache_key)
        if cached is not None and cached.document_count == len(segments):
            runtime_session.record_cache("vector_store", True)
            return cached
        runtime_session.record_cache("vector_store", False)
        started = time.perf_counter()
        store = VideoSegmentVectorStore()
        try:
            metadata_payload = json.loads(metadata_bytes)
            metadata = metadata_payload.get('metadata') if isinstance(metadata_payload, dict) else None
            if not isinstance(metadata, list) or len(metadata) != len(segments) or not all(
                isinstance(item, dict) and item.get('indexed_modalities')
                and (not require_video or 'video' in item['indexed_modalities']) for item in metadata
            ):
                raise ValueError('Dense index metadata is incomplete')
            store.load(str(vectors_path))
            if store.document_count != len(segments):
                raise ValueError('Dense index document count does not match segments')
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
            logger.warning('Rebuilding dense index for %s: %s', video_dir.name, exc)
            store.index_segments(segments, video_dir.name, processed_root=str(video_dir))
            if store.document_count != len(segments):
                raise ValueError('Dense index repair did not index every segment')
            staging = Path(tempfile.mkdtemp(prefix='.dense_index_', dir=video_dir))
            previous = staging / 'previous'
            replacement = staging / 'replacement'
            try:
                store.save(str(replacement))
                built_metadata = json.loads((replacement / 'metadata.json').read_text())['metadata']
                if require_video and not all('video' in item.get('indexed_modalities', []) for item in built_metadata):
                    raise ValueError('Multimodal dense indexing requires real video embeddings for every segment')
                if vectors_path.exists():
                    os.replace(vectors_path, previous)
                try:
                    os.replace(replacement, vectors_path)
                except BaseException:
                    if previous.exists():
                        os.replace(previous, vectors_path)
                    raise
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            metadata_bytes = metadata_path.read_bytes()
            cache_key = str(vectors_path.resolve()) + ':' + hashlib.sha256(metadata_bytes + str(require_video).encode()).hexdigest()
        runtime_session.record_timing("runtime_load.vector_store_seconds", time.perf_counter() - started)
        return runtime_session.cache_vector_store(cache_key, store)


def _parse_segment_reference(segment_id: Optional[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    if segment_id is None:
        return None, None, []
    text = str(segment_id).strip()
    if not text:
        return None, None, []

    video_hint = None
    raw_segment_id = text
    aliases = [text]

    match = re.match(r"^(?P<video>.+)_(?P<segment>\d+)$", text)
    if match:
        video_hint = match.group("video")
        raw_segment_id = match.group("segment")
        aliases.append(raw_segment_id)

    if raw_segment_id not in aliases:
        aliases.append(raw_segment_id)
    return video_hint, raw_segment_id, list(dict.fromkeys(aliases))


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _load_segments_payload(video_dir: Path) -> Dict[str, Dict[str, Any]]:
    segments_path = video_dir / "segments.json"
    if not segments_path.exists():
        return {}
    try:
        with open(segments_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load segment payload from %s: %s", segments_path, exc)
        return {}


def _load_segment_times_map(video_dir: Path) -> Dict[str, Dict[str, Any]]:
    segment_times_path = video_dir / "segment_times_info.json"
    if not segment_times_path.exists():
        return {}
    try:
        with open(segment_times_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load segment timing payload from %s: %s", segment_times_path, exc)
        return {}

    if not isinstance(payload, dict):
        return {}
    segment_times_info = payload.get("segment_times_info")
    return segment_times_info if isinstance(segment_times_info, dict) else {}


def _ensure_knowledge_graph_payload(video_dir: Path) -> Dict[str, Any]:
    from preprocessing.knowledge_graph import ensure_knowledge_graph_for_processed_video

    with runtime_session.time_block("runtime_load.knowledge_graph_seconds"):
        payload = ensure_knowledge_graph_for_processed_video(video_dir)
    return payload if isinstance(payload, dict) else {}


def _load_segment_graph_entities(video_dir: Path) -> Dict[str, List[str]]:
    graph_path = video_dir / "knowledge_graph.json"
    if not graph_path.exists():
        return {}
    try:
        with graph_path.open("r", encoding="utf-8") as handle:
            kg = json.load(handle) or {}
    except Exception as exc:
        logger.warning("Failed to load graph payload for %s: %s", video_dir, exc)
        return {}

    graph_entities: Dict[str, List[str]] = {}
    try:
        for ent_name, ent_data in (kg.get("entities") or {}).items():
            if not isinstance(ent_data, dict):
                continue
            for sid in ent_data.get("source_segments", []) or []:
                graph_entities.setdefault(str(sid), []).append(
                    f"{ent_name}: {ent_data.get('description', '')}"
                )
        for rel in kg.get("relations", []) or []:
            if not isinstance(rel, dict):
                continue
            rel_type = rel.get("relation_type") or rel.get("relation") or rel.get("type") or ""
            rel_text = f"{rel.get('source', '')} -{rel_type}-> {rel.get('target', '')}"
            for sid in rel.get("source_segments", []) or []:
                graph_entities.setdefault(str(sid), []).append(rel_text)
    except Exception:
        return graph_entities
    return graph_entities


def _normalize_graph_query_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _graph_entity_match_score(entity_name: str, entity_payload: Optional[Dict[str, Any]], query: str) -> float:
    normalized_query = _normalize_graph_query_token(query)
    if not normalized_query:
        return 0.0

    query_tokens = set(normalized_query.split())
    haystacks = [
        _normalize_graph_query_token(entity_name),
        _normalize_graph_query_token((entity_payload or {}).get("label", "")),
        _normalize_graph_query_token((entity_payload or {}).get("description", "")),
        _normalize_graph_query_token((entity_payload or {}).get("type", "")),
        _normalize_graph_query_token(" ".join((entity_payload or {}).get("aliases", []) or [])),
    ]
    joined_haystack = " ".join(part for part in haystacks if part).strip()
    if not joined_haystack:
        return 0.0
    if normalized_query == _normalize_graph_query_token(entity_name):
        return 3.0
    if normalized_query in haystacks:
        return 2.5
    if normalized_query in joined_haystack:
        return 2.0

    haystack_tokens = set(joined_haystack.split())
    overlap = len(query_tokens.intersection(haystack_tokens))
    if overlap <= 0:
        return 0.0
    return 1.0 + (overlap / max(1, len(query_tokens)))


def _build_segment_image_payload(
    video_name: str,
    frame_paths: List[Any],
    frame_times: List[Any],
    *,
    text_hint: str = "",
) -> Tuple[List[Dict[str, Any]], List[float]]:
    if not frame_paths:
        return [], []

    try:
        from toolkit.impl.visual import (
            pack_planner_image_evidence,
            resolve_processed_frame_image_path,
            summarize_planner_image_evidence,
        )
    except Exception as exc:
        logger.debug("Segment image payload helper unavailable: %s", exc)
        return [], []

    image_evidence: List[Dict[str, Any]] = []
    for index, rel_path in enumerate(frame_paths):
        resolved_path = resolve_processed_frame_image_path(video_name, rel_path)
        if not resolved_path:
            continue
        timestamp = None
        if index < len(frame_times):
            try:
                timestamp = round(float(frame_times[index]), 3)
            except Exception:
                timestamp = None
        image_evidence.append(
            {
                "image": resolved_path,
                "timestamp": timestamp,
                "source_images": [resolved_path],
                "pack_mode": "cached_single",
            }
        )
    packed = pack_planner_image_evidence(image_evidence, text_hint=text_hint)
    if not packed:
        return [], []
    image_stats = summarize_planner_image_evidence(packed)
    return packed, list(image_stats.get("image_timestamps") or [])


def _segment_sort_key(entry: Dict[str, Any]) -> Tuple[float, float, str]:
    parsed = _parse_time_range(entry.get("time_range") or entry.get("time"))
    if parsed is None:
        return (float("inf"), float("inf"), str(entry.get("segment_ref", "")))
    return (float(parsed[0]), float(parsed[1]), str(entry.get("segment_ref", "")))


def _build_candidate_probe_window(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Expose an executable full-segment probe for dense retrieval candidates.

    A short probe around the first returned timestamp can miss a brief event
    later in the same prepared segment.  Retrieval is not answer evidence, but
    it does know the segment boundary; make that boundary explicit so the
    Planner can ask Inspect_Frame/Video_Clip_QA to verify the whole candidate.
    """
    parsed = _parse_time_range(entry.get("time_range") or entry.get("time"))
    if parsed is None:
        return None
    start, end = float(parsed[0]), float(parsed[1])
    if end <= start:
        return None
    segment_ref = str(
        entry.get("segment_ref")
        or entry.get("segment_id")
        or entry.get("id")
        or ""
    ).strip()
    if not segment_ref:
        return None
    return {
        "segment_ref": segment_ref,
        "start_time": round(start, 3),
        "end_time": round(end, 3),
        "time_range": f"{start:.3f}-{end:.3f}",
        "verification_required": True,
        "recommended_tools": ["Inspect_Frame", "Video_Clip_QA"],
    }


def _build_segment_bundle(
    video_dir: Path,
    seg_id: Any,
    seg_data: Dict[str, Any],
    *,
    times_map: Optional[Dict[str, Dict[str, Any]]] = None,
    graph_entities: Optional[Dict[str, List[str]]] = None,
    include_fields: Optional[set[str]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    text_hint: str = "",
) -> Dict[str, Any]:
    time_info = {}
    if isinstance(times_map, dict):
        time_info = times_map.get(str(seg_id), {}) or {}
    if not isinstance(time_info, dict):
        time_info = {}

    aliases = [
        str(alias)
        for alias in (
            time_info.get("segment_aliases")
            or seg_data.get("segment_aliases")
            or [str(seg_id), f"{video_dir.name}_{seg_id}"]
        )
        if str(alias).strip()
    ]
    frame_times = list(time_info.get("frame_times", seg_data.get("frame_times", [])) or [])
    frame_paths = list(time_info.get("frame_paths", seg_data.get("frame_paths", [])) or [])
    image_evidence, image_timestamps = _build_segment_image_payload(
        video_dir.name,
        frame_paths,
        frame_times,
        text_hint=text_hint or f"segment {video_dir.name}_{seg_id}",
    )

    want = set(include_fields or {"caption", "transcript", "graph"})
    graph_payload = []
    if "graph" in want and isinstance(graph_entities, dict):
        graph_payload = list(graph_entities.get(str(seg_id), []) or [])

    entry: Dict[str, Any] = {
        "segment_id": f"{video_dir.name}_{seg_id}",
        "segment_ref": f"{video_dir.name}_{seg_id}",
        "raw_segment_id": str(seg_id),
        "segment_aliases": aliases,
        "video": video_dir.name,
        "time": seg_data.get("time", ""),
        "time_range": seg_data.get("time", ""),
        "frame_times": frame_times,
        "frame_paths": frame_paths,
        "image_evidence": image_evidence,
        "image_timestamps": image_timestamps,
    }
    segment_video_path = str(time_info.get("segment_video_path", "") or "").strip()
    if segment_video_path:
        entry["segment_video_path"] = segment_video_path
    if "caption" in want:
        entry["caption"] = str(seg_data.get("caption", "") or "")
    if "transcript" in want:
        entry["transcript"] = str(seg_data.get("transcript", "") or "")
    if "graph" in want:
        entry["graph"] = graph_payload

    snippet = _first_nonempty(
        entry.get("caption"),
        entry.get("transcript"),
        " | ".join(graph_payload[:2]) if graph_payload else "",
    )
    if snippet:
        entry["snippet"] = snippet
    entry["result"] = (
        f"Segment {entry['segment_ref']} at {entry['time_range']}: {snippet}"
        if snippet
        else f"Segment {entry['segment_ref']} at {entry['time_range']}"
    ).strip()

    for key, value in dict(extra_fields or {}).items():
        if value in (None, "", [], {}):
            continue
        entry[key] = value
    return entry


def _normalize_embedding_provider_name(provider: Any) -> str:
    normalized = str(provider or "").strip().lower()
    aliases = {
        "": "auto",
        "sidecar": "service",
        "remote": "service",
        "openai": "service",
        "api": "service",
        "disable": "disabled",
        "off": "disabled",
        "false": "disabled",
        "0": "disabled",
    }
    return aliases.get(normalized, normalized)


def _settings_embedding_mapping_value(mapping_name: str, role: str = "tool_search") -> str:
    mapping = getattr(settings, mapping_name, {}) or {}
    if not isinstance(mapping, dict):
        return ""
    normalized_role = str(role or "").strip().lower()
    role_key = re.sub(r"[^a-z0-9]+", "_", normalized_role).strip("_")
    for key in (normalized_role, role_key):
        if not key:
            continue
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _tool_search_embedding_provider() -> str:
    return _normalize_embedding_provider_name(
        _first_nonempty(
            os.environ.get("TOOL_STAR_TOOL_SEARCH_EMBEDDING_PROVIDER"),
            _settings_embedding_mapping_value("non_planner_embedding_role_providers"),
            getattr(settings, "non_planner_embedding_provider", "auto"),
            os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_PROVIDER"),
        )
    )


def _tool_search_embedding_service_base() -> str:
    return format_endpoint_list(
        _first_nonempty(
            os.environ.get("TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASES"),
            os.environ.get("TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASE"),
            os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASES"),
            os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASE"),
            os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASES"),
            os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASE"),
            _settings_embedding_mapping_value("non_planner_embedding_role_api_bases"),
            getattr(settings, "non_planner_embedding_api_base", ""),
            getattr(settings, "planner_grounding_embedding_api_base", ""),
        )
    )


def _tool_search_embedding_service_key() -> str:
    return _first_nonempty(
        os.environ.get("TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_KEY"),
        _settings_embedding_mapping_value("non_planner_embedding_role_api_keys"),
        getattr(settings, "non_planner_embedding_api_key", ""),
        os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_API_KEY"),
        os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_KEY"),
        getattr(settings, "planner_grounding_embedding_api_key", ""),
        getattr(settings, "openai_api_key", ""),
        "EMPTY",
    )


def _tool_search_embedding_service_model() -> str:
    return _first_nonempty(
        os.environ.get("TOOL_STAR_TOOL_SEARCH_EMBEDDING_MODEL"),
        _settings_embedding_mapping_value("non_planner_embedding_role_models"),
        getattr(settings, "non_planner_embedding_model", ""),
        getattr(settings, "embedding_model_path", ""),
        os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_MODEL"),
        os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_MODEL"),
        getattr(settings, "planner_grounding_embedding_model", ""),
    )


def _tool_search_embedding_service_enabled() -> bool:
    provider = _tool_search_embedding_provider()
    if provider in {"disabled", "local", "transformers"}:
        return False
    if provider not in {"auto", "service"}:
        return False
    if provider == "auto":
        return bool(_tool_search_embedding_service_base() and _tool_search_embedding_service_model())
    return True


class _ToolSearchEmbedder:
    """Lazy embedding encoder for tool semantic retrieval."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._tokenizer = None
        self._model = None
        self._service_pool = None
        self._service_model = _tool_search_embedding_service_model()
        self._service_base = _tool_search_embedding_service_base()
        self._service_enabled = _tool_search_embedding_service_enabled()
        self._embedding_cache: Dict[str, Any] = {}
        if self._service_enabled:
            if not self._service_base:
                raise RuntimeError(
                    "Tool_Search embedding service requires TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASE "
                    "or TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASE."
                )
            if not self._service_model:
                raise RuntimeError(
                    "Tool_Search embedding service requires TOOL_STAR_TOOL_SEARCH_EMBEDDING_MODEL "
                    "or TOOL_STAR_NON_PLANNER_EMBEDDING_MODEL."
                )
            self._device_type = "service"
            self._accelerator_count = 0
            self._device = "service"
            logger.info(
                "Tool_Search embedder runtime: provider=service model=%s base=%s",
                self._service_model,
                self._service_base,
            )
            self._accelerator_lock = None
            return
        device_type, device_count = detect_device()
        initialize_device(device_type)
        self._device_type = device_type
        self._accelerator_count = device_count or 0
        self._device = runtime_session.get_stable_device("embedding", device=device_type)
        self._accelerator_lock = SharedAcceleratorQueueLock(
            name="tool_search_embedding",
            slot_key=resolve_local_accelerator_slot_key(self._device),
            enabled=str(self._device).startswith("cuda"),
        )

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        torch = _torch_module()
        resolved_model_path = Path(self.model_path).expanduser()
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Local embedding model path not found: {resolved_model_path}")
        model_path_str = str(resolved_model_path.resolve())
        cache_key = json.dumps(
            {
                "runtime": "tool_search_text_embedding",
                "model_path": model_path_str,
                "device": self._device,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached_runtime = runtime_session.get_cached_artifact(cache_key)
        if cached_runtime is None:
            lock_context = self._accelerator_lock if self._accelerator_lock is not None else nullcontext()
            with lock_context:
                cached_runtime = runtime_session.get_cached_artifact(cache_key)
                if cached_runtime is None:
                    AutoModel, AutoTokenizer = _transformer_model_classes()
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        model_path_str,
                        trust_remote_code=True,
                        local_files_only=True,
                        cache_dir=model_path_str,
                    )
                    self._model = AutoModel.from_pretrained(
                        model_path_str,
                        trust_remote_code=True,
                        local_files_only=True,
                        cache_dir=model_path_str,
                        torch_dtype=_preferred_text_embedding_dtype(self._device),
                    )
                    self._model.to(self._device)
                    target_dtype = _preferred_text_embedding_dtype(self._device)
                    if target_dtype != torch.float32:
                        try:
                            self._model.to(dtype=target_dtype)
                        except Exception as exc:
                            logger.warning(
                                "Tool_Search embedder could not switch to %s on %s; using loaded dtype. error=%s",
                                target_dtype,
                                self._device,
                                exc,
                            )
                    self._model.eval()
                    cached_runtime = runtime_session.cache_artifact(
                        cache_key,
                        {
                            "tokenizer": self._tokenizer,
                            "model": self._model,
                            "device": self._device,
                        },
                    )
                    logger.info(
                        "Tool_Search embedder runtime: device=%s dtype=%s visible_accelerators=%s model=%s",
                        self._device,
                        _resolve_model_float_dtype(self._model),
                        self._accelerator_count,
                        model_path_str,
                    )
                    logger.info(
                        "Tool_Search embedder accelerator snapshot: %s",
                        get_accelerator_snapshot(self._device_type),
                    )
        self._tokenizer = cached_runtime["tokenizer"]
        self._model = cached_runtime["model"]
        self._device = cached_runtime["device"]

    def _lazy_load_service(self) -> OpenAIEndpointPool:
        if self._service_pool is not None:
            return self._service_pool
        cache_key = json.dumps(
            {
                "runtime": "tool_search_embedding_service_pool",
                "base": self._service_base,
                "model": self._service_model,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached_pool = runtime_session.get_cached_artifact(cache_key)
        if cached_pool is None:
            cached_pool = runtime_session.cache_artifact(
                cache_key,
                OpenAIEndpointPool(
                    api_key=_tool_search_embedding_service_key(),
                    base_urls=self._service_base,
                    namespace="tool_search_embedding",
                ),
            )
        self._service_pool = cached_pool
        return self._service_pool

    def _encode_via_service(self, text: str) -> torch.Tensor:
        torch = _torch_module()
        with self._lazy_load_service().borrow() as (client, endpoint):
            if embeddings_api_marked_unhealthy(endpoint):
                embedding = request_pooling_embedding(
                    api_key=_tool_search_embedding_service_key(),
                    base_url=endpoint,
                    model=self._service_model,
                    text=text,
                    timeout=float(getattr(settings, "sidecar_smoke_timeout_seconds", 120) or 120),
                )
            else:
                try:
                    response = client.embeddings.create(model=self._service_model, input=text)
                    if not response.data:
                        raise RuntimeError("Tool_Search embedding service returned no vectors.")
                    embedding = response.data[0].embedding
                except Exception as exc:
                    logger.warning(
                        "Tool_Search OpenAI embeddings endpoint failed on %s; retrying via vLLM /pooling: %s",
                        endpoint,
                        exc,
                    )
                    embedding = request_pooling_embedding(
                        api_key=_tool_search_embedding_service_key(),
                        base_url=endpoint,
                        model=self._service_model,
                        text=text,
                        timeout=float(getattr(settings, "sidecar_smoke_timeout_seconds", 120) or 120),
                    )
                    mark_embeddings_api_unhealthy(endpoint)
        vector = torch.tensor(embedding, dtype=torch.float32)
        if vector.numel() == 0:
            raise RuntimeError("Tool_Search embedding service returned an empty vector.")
        return torch.nn.functional.normalize(vector, p=2, dim=0)

    def encode(self, text: str) -> torch.Tensor:
        torch = _torch_module()
        normalized_text = (text or "").strip()
        if not normalized_text:
            return torch.zeros(1, dtype=torch.float32)
        cached = self._embedding_cache.get(normalized_text)
        if cached is not None:
            return cached

        if self._service_enabled:
            encoded = self._encode_via_service(normalized_text).detach().cpu().to(torch.float32)
            self._embedding_cache[normalized_text] = encoded
            return encoded

        lock_context = self._accelerator_lock if self._accelerator_lock is not None else nullcontext()
        with lock_context:
            self._lazy_load()
            inputs = self._tokenizer(
                normalized_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                model_dtype = _resolve_model_float_dtype(self._model)
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=model_dtype)
                    if str(self._device).startswith("cuda") and model_dtype in (torch.float16, torch.bfloat16)
                    else nullcontext()
                )
                with autocast_context:
                    outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state[0]  # (seq_len, dim)
                mask = inputs["attention_mask"][0].unsqueeze(-1).to(hidden.dtype)
                hidden, mask = _align_hidden_and_attention_mask(
                    hidden,
                    mask,
                    source="tool_search_embedder.encode",
                )
                denom = torch.clamp(mask.sum(), min=1.0)
                pooled = (hidden * mask).sum(dim=0) / denom
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
        encoded = pooled.detach().cpu().to(torch.float32)
        self._embedding_cache[normalized_text] = encoded
        return encoded


_TOOL_SEARCH_EMBEDDER: Optional[_ToolSearchEmbedder] = None


def _get_tool_search_embedder(model_path: str) -> _ToolSearchEmbedder:
    global _TOOL_SEARCH_EMBEDDER
    service_enabled = _tool_search_embedding_service_enabled()
    service_base = _tool_search_embedding_service_base()
    service_model = _tool_search_embedding_service_model()
    cache_hit = (
        _TOOL_SEARCH_EMBEDDER is not None
        and _TOOL_SEARCH_EMBEDDER.model_path == model_path
        and bool(getattr(_TOOL_SEARCH_EMBEDDER, "_service_enabled", False)) == service_enabled
        and str(getattr(_TOOL_SEARCH_EMBEDDER, "_service_base", "") or "") == service_base
        and str(getattr(_TOOL_SEARCH_EMBEDDER, "_service_model", "") or "") == service_model
    )
    runtime_session.record_cache("tool_search_embedder", cache_hit)
    if not cache_hit:
        started = time.perf_counter()
        _TOOL_SEARCH_EMBEDDER = _ToolSearchEmbedder(model_path=model_path)
        runtime_session.record_timing("runtime_load.tool_search_embedder_seconds", time.perf_counter() - started)
    return _TOOL_SEARCH_EMBEDDER


class ToolSearch(BaseTool):
    """Search for relevant tools based on intent description."""
    
    name: str = "Tool_Search"
    description: str = (
        "Discover executable tools from the registry by capability. "
        "This tool does not search the video corpus, timestamps, frames, captions, "
        "transcripts, or OCR output."
    )
    tags: list = ["base", "retrieval", "search"]
    
    @property
    def model_path(self) -> str:
        return settings.embedding_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "intent_description": {
                "type": "string",
                "description": "Description of what you want to accomplish"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of tools to retrieve",
                "default": 5
            },
            "tags": {
                "type": "array",
                "description": "Optional list of tags to filter tools (e.g., ['base'] or ['meta'])",
                "items": {"type": "string"},
                "default": None
            },
            "tool_type": {
                "type": "string",
                "enum": ["base", "meta", "all"],
                "description": "Primary tool class filter. Use base for executable/base tools, meta for atomic/meta tools, all for both.",
                "default": "all"
            }
        },
        "required": ["intent_description"]
    }
    
    def run(
        self,
        intent_description: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
        tool_type: str = "all",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute tool search.
        
        Args:
            intent_description: What the user wants to do
            top_k: Number of results to return
            tags: Optional tags to filter tools (any match)
            **kwargs: Additional arguments
            
        Returns:
            List of relevant tool definitions
        """
        from toolkit.registry import registry
        
        normalized_type = str(tool_type or "all").strip().lower()
        if normalized_type == "biz":
            normalized_type = "base"
        if normalized_type not in {"base", "meta", "all"}:
            normalized_type = "all"

        all_tools = registry.get_all_tools(tags=tags)
        if not _intent_requests_tool_discovery(intent_description):
            all_tools = [tool for tool in all_tools if tool.name != "Tool_Search"]
        all_tools = [tool for tool in all_tools if tool.name != "Finish"]
        hidden_names = {
            tool.name
            for tool in all_tools
            if get_sparse_scope_hidden_tool_payload(tool.name) is not None
        }
        if hidden_names:
            logger.info(
                "Tool_Search hiding %d sparse-scope tool(s) from routing: %s",
                len(hidden_names),
                sorted(hidden_names),
            )
            all_tools = [tool for tool in all_tools if tool.name not in hidden_names]
        if not all_tools:
            return []

        recent_tools = runtime_session.get_recent_tools(limit=8)
        features = _intent_route_features(intent_description)
        try:
            embedder = _get_tool_search_embedder(self.model_path)
            query_vec = embedder.encode(intent_description)
            schema_records = _get_tool_search_schema_records(
                model_path=self.model_path,
                tools=all_tools,
            )
            filtered_records = list(schema_records)
            if normalized_type != "all":
                filtered_records = [
                    record
                    for record in filtered_records
                    if str((record.get("schema") or {}).get("tool_type", "")).strip() == normalized_type
                ]
            if not filtered_records:
                return []

            base_records = [
                record
                for record in filtered_records
                if str((record.get("schema") or {}).get("tool_type", "")).strip() == "base"
            ]
            meta_records = [
                record
                for record in filtered_records
                if str((record.get("schema") or {}).get("tool_type", "")).strip() == "meta"
            ]

            if normalized_type == "meta":
                meta_scored = _score_tool_search_records(
                    records=meta_records,
                    intent_description=intent_description,
                    recent_tools=recent_tools,
                    query_vec=query_vec,
                )
                results = _select_diverse_candidates(meta_scored, top_k=max(1, top_k))
            elif normalized_type == "base":
                base_scored = _score_tool_search_records(
                    records=base_records,
                    intent_description=intent_description,
                    recent_tools=recent_tools,
                    query_vec=query_vec,
                )
                results = _select_diverse_candidates(base_scored, top_k=max(1, top_k))
            else:
                meta_budget = _stage_two_meta_budget(normalized_type, features, top_k=max(1, top_k))
                base_budget = max(1, int(top_k) - meta_budget)
                base_scored = _score_tool_search_records(
                    records=base_records or filtered_records,
                    intent_description=intent_description,
                    recent_tools=recent_tools,
                    query_vec=query_vec,
                )
                primary_results = _select_diverse_candidates(
                    base_scored,
                    top_k=max(1, base_budget),
                )
                meta_results: List[Dict[str, Any]] = []
                if meta_budget > 0 and meta_records:
                    meta_scored = _score_tool_search_records(
                        records=meta_records,
                        intent_description=intent_description,
                        recent_tools=recent_tools,
                        query_vec=query_vec,
                    )
                    meta_results = _select_diverse_candidates(
                        meta_scored,
                        top_k=max(1, meta_budget),
                    )
                if not primary_results and meta_records:
                    primary_results = meta_results
                    meta_results = []
                results = _interleave_tool_search_stages(
                    primary_items=primary_results,
                    meta_items=meta_results,
                    top_k=max(1, top_k),
                )
            results = _inject_explicit_tool_candidate(
                results=results,
                all_tools=all_tools,
                intent_description=intent_description,
                top_k=max(1, top_k),
            )
            logger.info(
                "Tool_Search semantic retrieval returned %d/%d tools for intent='%s' using %s (recent_tools=%s, tool_type=%s, stage2_meta=%s)",
                len(results),
                len(all_tools),
                intent_description,
                self.model_path,
                recent_tools,
                normalized_type,
                _stage_two_meta_budget(normalized_type, features, top_k=max(1, top_k)),
            )
            return results
        except Exception as exc:
            raise RuntimeError("Tool_Search semantic embedding/routing failed") from exc


class QueryRewrite(BaseTool):
    """Rewrite complex or ambiguous queries."""
    
    name: str = "Query_Rewrite"
    description: str = (
        "Rewrite, expand, or decompose complex or ambiguous user queries. "
        "Converts colloquial questions to normalized queries suitable for retrieval, "
        "or splits them into multiple sub-queries for multi-hop reasoning."
    )
    tags: list = ["base", "retrieval", "rewrite"]
    
    @property
    def model_path(self) -> str:
        return settings.query_rewrite_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Original query to rewrite"
            },
            "mode": {
                "type": "string",
                "enum": ["normalize", "expand", "decompose"],
                "description": "Rewrite mode",
                "default": "normalize"
            }
        },
        "required": ["query"]
    }
    
    def run(self, query: str, mode: str = "normalize", **kwargs) -> Dict[str, Any]:
        """Execute query rewrite.
        
        Args:
            query: Original query
            mode: Rewrite mode (normalize/expand/decompose)
            **kwargs: Additional arguments
            
        Returns:
            Rewritten query or queries
        """
        text = " ".join(str(query or "").strip().split())
        if not text:
            return {"original": query, "normalized": "", "mode": mode}

        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = re.sub(r"\bpls\b", "please", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bu\b", "you", normalized, flags=re.IGNORECASE)

        if mode == "decompose":
            chunks = re.split(r"\b(?:and|then|after that|also|plus|;|\.)\b", normalized, flags=re.IGNORECASE)
            sub_queries = [c.strip(" ,") for c in chunks if c and c.strip(" ,")]
            if not sub_queries:
                sub_queries = [normalized]
            result = {
                "original": query,
                "mode": mode,
                "normalized": normalized,
                "sub_queries": sub_queries,
            }
        elif mode == "expand":
            synonyms = {
                "argue": ["quarrel", "dispute", "conflict"],
                "gunshot": ["shot fired", "muzzle blast", "firearm discharge"],
                "police": ["officer", "law enforcement"],
                "red shirt": ["man in red", "person wearing red clothing"],
            }
            expanded_terms: List[str] = []
            low = normalized.lower()
            for key, vals in synonyms.items():
                if key in low:
                    expanded_terms.extend(vals)
            expansion = normalized
            if expanded_terms:
                expansion = f"{normalized}; related terms: {', '.join(dict.fromkeys(expanded_terms))}"
            result = {
                "original": query,
                "mode": mode,
                "normalized": normalized,
                "expanded": expansion,
            }
        else:  # normalize
            result = {
                "original": query,
                "mode": "normalize",
                "normalized": normalized,
            }
        result = enrich_tool_result(result, tool_name=self.name)
        
        logger.info(f"Query_Rewrite ({mode}): {query}")
        return result


class GraphSearch(BaseTool):
    """Multi-hop exploration on temporal knowledge graph."""
    
    name: str = "Graph_Search"
    description: str = (
        "Perform multi-hop exploration on a temporal 'scene-entity' knowledge graph. "
        "Start from a given entity and traverse relationship chains (e.g., temporal adjacency, "
        "co-occurrence) to discover potential association paths and implicit clues."
    )
    tags: list = ["base", "retrieval", "graph"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Starting entity for graph traversal"
            },
            "relation": {
                "type": "string",
                "description": "Relationship type to follow (optional)",
                "default": None
            },
            "steps": {
                "type": "integer",
                "description": "Number of hops for traversal",
                "default": 2
            }
        },
        "required": ["entity"]
    }
    
    def run(self, entity: str, relation: Optional[str] = None, steps: int = 2, **kwargs) -> Dict[str, Any]:
        """Execute graph search.
        
        Args:
            entity: Starting entity
            relation: Relationship type
            steps: Number of hops
            **kwargs: Additional arguments
            
        Returns:
            Graph traversal results
        """
        video_hint = kwargs.get("video") or kwargs.get("video_path")
        hidden_payload = get_sparse_scope_hidden_tool_payload(self.name, video_hint=video_hint)
        if hidden_payload is not None:
            return hidden_payload

        scoped_videos = list(_iter_scoped_video_dirs(video_hint))
        if not scoped_videos:
            return {"error": "No processed data found. Please run preprocessing first."}

        results = {
            "start_entity": entity,
            "relation": relation,
            "steps": steps,
            "paths": [],
            "discovered_entities": set(),
            "matched_start_nodes": [],
            "warnings": [],
            "resolved_video_scope": sorted(runtime_session.resolve_video_scope(video_hint)),
        }
        searched_graphs = 0

        for video_dir in scoped_videos:
            try:
                kg_data = _ensure_knowledge_graph_payload(video_dir)
                searched_graphs += 1
                self._bfs_search(kg_data, entity, relation, steps, results, video_name=video_dir.name)
            except Exception as e:
                error_text = f"{video_dir}: {e}"
                logger.error("Error reading KG from %s", error_text)
                results["warnings"].append(error_text)

        results["discovered_entities"] = sorted(results["discovered_entities"])
        results["matched_start_nodes"] = sorted(set(results["matched_start_nodes"]))
        results["paths"].sort(
            key=lambda item: (
                -float(item.get("path_score", 0.0) or 0.0),
                len(item.get("path", []) or []),
                str(item.get("video", "")),
            )
        )
        if results["paths"]:
            status = "success"
            summary = (
                f"Graph search from '{entity}' found {len(results['paths'])} paths "
                f"and {len(results['discovered_entities'])} related entities."
            )
        elif results["warnings"] and searched_graphs == 0:
            status = "error"
            summary = (
                f"Graph search for '{entity}' failed because no readable knowledge graph was available. "
                "Inspect the returned warnings for the root cause."
            )
        else:
            status = "no_valid_result" if searched_graphs > 0 else "error"
            summary = (
                f"Graph search from '{entity}' found 0 paths and 0 related entities."
                if searched_graphs > 0
                else f"Graph search for '{entity}' could not run because no readable knowledge graph was available."
            )
        results = enrich_tool_result(
            {
                **results,
                "searched_graph_count": searched_graphs,
                "status": status,
                "result": summary,
            },
            tool_name=self.name,
        )
        logger.info(f"Graph_Search from {entity}: found {len(results['paths'])} paths")
        return results

    def _bfs_search(
        self,
        kg: Dict,
        start_entity: str,
        target_relation: Optional[str],
        max_steps: int,
        results: Dict,
        *,
        video_name: str,
    ):
        """Helper for BFS search on graph."""
        entities = kg.get("entities", {}) or {}
        relations = kg.get("relations", []) or []

        start_query = str(start_entity or "").strip().lower()
        if not start_query:
            return

        start_candidates: List[Tuple[str, float]] = []
        for name, entity_payload in entities.items():
            match_score = _graph_entity_match_score(str(name), entity_payload, start_query)
            if match_score > 0:
                start_candidates.append((str(name), match_score))
        start_candidates.sort(key=lambda item: (-item[1], item[0]))
        start_nodes = [name for name, _ in start_candidates[:8]]
        if not start_nodes:
            return

        for node in start_nodes:
            results["matched_start_nodes"].append(node)

        adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            if not source or not target:
                continue
            relation_type = str(rel.get("relation_type") or rel.get("relation") or rel.get("type") or "").strip() or "RELATED_TO"
            relation_description = str(rel.get("description") or relation_type).strip()
            source_segments = [str(seg) for seg in (rel.get("source_segments") or []) if str(seg).strip()]
            edge_payload = {
                "relation_type": relation_type,
                "description": relation_description,
                "weight": float(rel.get("weight", 1.0) or 1.0),
                "source_segments": source_segments,
                "time_range": rel.get("time_range"),
            }
            adjacency[source].append({**edge_payload, "neighbor": target})
            adjacency[target].append({**edge_payload, "neighbor": source})

        start_node_score_map = {name: score for name, score in start_candidates}
        normalized_target_relation = _normalize_graph_query_token(target_relation)
        queue: List[Tuple[str, List[str], List[Dict[str, Any]], set[str], float]] = [
            (node, [node], [], {node}, float(start_node_score_map.get(node, 0.0))) for node in start_nodes[:5]
        ]
        seen_edges: set[Tuple[str, str, int]] = set()

        while queue:
            current, path, relation_chain, local_visited, score = queue.pop(0)
            if len(path) - 1 >= max_steps:
                continue

            for edge in adjacency.get(current, []):
                relation_text = _normalize_graph_query_token(
                    f"{edge.get('relation_type', '')} {edge.get('description', '')}"
                )
                if normalized_target_relation and normalized_target_relation not in relation_text:
                    continue
                neighbor = str(edge.get("neighbor", "")).strip()
                if not neighbor or neighbor in local_visited:
                    continue

                edge_key = (current, neighbor, len(path))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                new_path = path + [neighbor]
                new_chain = relation_chain + [
                    {
                        "relation_type": str(edge.get("relation_type", "")).strip() or "RELATED_TO",
                        "description": str(edge.get("description", "")).strip(),
                        "source_segments": list(edge.get("source_segments") or []),
                        "time_range": edge.get("time_range"),
                    }
                ]
                neighbor_payload = entities.get(neighbor, {})
                entity_specificity = min(
                    1.0,
                    len(str((neighbor_payload or {}).get("description", "") or "").split()) / 12.0,
                )
                new_score = score + float(edge.get("weight", 1.0) or 1.0) + entity_specificity
                supporting_segments = sorted(
                    {
                        seg_id
                        for relation_meta in new_chain
                        for seg_id in (relation_meta.get("source_segments") or [])
                        if str(seg_id).strip()
                    }
                )
                results["paths"].append(
                    {
                        "video": video_name,
                        "path": new_path,
                        "relation_chain": new_chain,
                        "supporting_segments": supporting_segments,
                        "final_entity_info": neighbor_payload,
                        "path_score": round(new_score, 4),
                    }
                )
                results["discovered_entities"].add(neighbor)
                if len(results["paths"]) >= 64:
                    return
                queue.append((neighbor, new_path, new_chain, local_visited | {neighbor}, new_score))


class KeywordRetrieve(BaseTool):
    """Exact keyword matching retrieval."""
    
    name: str = "Keyword_Retrieve"
    description: str = (
        "Perform exact matching retrieval for proper nouns, specific IDs, or rare terms. "
        "Returns the matched segment's preprocessed evidence bundle, including caption, transcript, "
        "timestamps, and cached frame images."
    )
    tags: list = ["base", "retrieval", "keyword"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "Keyword or term to search for"
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether search is case-sensitive",
                "default": False
            }
        },
        "required": ["term"]
    }
    
    def run(self, term: str, case_sensitive: bool = False, **kwargs) -> Dict[str, Any]:
        """Execute keyword retrieval.
        
        Args:
            term: Term to search
            case_sensitive: Case sensitivity flag
            **kwargs: Additional arguments
            
        Returns:
            List of exact matches
        """
        from pathlib import Path
        from config.settings import settings

        cache_dir = Path(settings.data_cache_dir) / "processed"
        results: List[Dict[str, Any]] = []

        raw_term = str(term or "")
        normalized_term = re.sub(r"[^a-zA-Z0-9]+", " ", raw_term).strip()
        normalized_compact = normalized_term.replace(" ", "")
        query_tokens = [tok for tok in normalized_term.lower().split() if tok]
        video_hint = kwargs.get("video") or kwargs.get("video_path")
        hidden_payload = get_sparse_scope_hidden_tool_payload(self.name, video_hint=video_hint)
        if hidden_payload is not None:
            hidden_payload.update(
                {
                    "term": term,
                    "case_sensitive": case_sensitive,
                    "match_count": 0,
                    "matched_segments": [],
                }
            )
            return hidden_payload
        resolved_scope = sorted(runtime_session.resolve_video_scope(video_hint))
        
        if not cache_dir.exists():
            return {
                "term": term,
                "case_sensitive": case_sensitive,
                "match_count": 0,
                "matched_segments": [],
                "status": "no_valid_result",
                "result": "Keyword_Retrieve did not provide usable evidence. Planner should try another relevant tool, broaden/narrow the time range, or reformulate the probe.",
            }
            
        for video_dir in _iter_scoped_video_dirs(video_hint):
            segments = _load_segments_payload(video_dir)
            if not segments:
                continue
            times_map = _load_segment_times_map(video_dir)
            graph_entities = _load_segment_graph_entities(video_dir)

            for seg_id, seg_data in segments.items():
                if not isinstance(seg_data, dict):
                    continue
                content = _collect_searchable_segment_text(seg_data)

                match = False
                if case_sensitive:
                    if raw_term in content:
                        match = True
                else:
                    lowered_content = content.lower()
                    normalized_content = re.sub(r"[^a-zA-Z0-9]+", " ", lowered_content).strip()
                    compact_content = normalized_content.replace(" ", "")
                    if raw_term.lower() in lowered_content:
                        match = True
                    elif normalized_term and normalized_term.lower() in normalized_content:
                        match = True
                    elif normalized_compact and normalized_compact.lower() in compact_content:
                        match = True
                    elif query_tokens and all(token in normalized_content.split() for token in query_tokens):
                        match = True

                if not match:
                    continue

                bundle = _build_segment_bundle(
                    video_dir,
                    seg_id,
                    seg_data,
                    times_map=times_map,
                    graph_entities=graph_entities,
                    include_fields={"caption", "transcript", "graph"},
                    extra_fields={
                        "id": f"{video_dir.name}_{seg_id}",
                        "term": term,
                        "match_type": "keyword_exact",
                        "confidence": 1.0,
                    },
                    text_hint=f"keyword {term}",
                )
                results.append(bundle)

        logger.info(f"Keyword_Retrieve for term '{term}': found {len(results)} matches")
        results.sort(key=_segment_sort_key)
        status = "success" if results else "no_valid_result"
        payload = {
            "term": term,
            "case_sensitive": case_sensitive,
            "resolved_video_scope": resolved_scope,
            "match_count": len(results),
            "matched_segments": results,
            "status": status,
            "result": (
                f"Keyword_Retrieve found {len(results)} matching segments for '{term}'."
                if results
                else "Keyword_Retrieve did not provide usable evidence. Planner should try another relevant tool, broaden/narrow the time range, or reformulate the probe."
            ),
        }
        return enrich_tool_result(payload, tool_name=self.name)


class SemanticRetrieve(BaseTool):
    """Semantic fuzzy retrieval."""
    
    name: str = "Semantic_Retrieve"
    description: str = (
        "Perform fuzzy semantic retrieval based on natural language description. "
        "Returns the retrieved segment's preprocessed evidence bundle, including caption, transcript, "
        "timestamps, and cached frame images. Supports time-range constraints. Dense visual-only matches "
        "are returned as unverified candidates and are not cited as answer evidence until Inspect_Frame "
        "or Video_Clip_QA confirms them. Each candidate also exposes its full executable probe window; "
        "verify that window before narrowing to a short timestamp."
    )
    tags: list = ["base", "retrieval", "semantic"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query"
            },
            "time_range": {
                "type": "string",
                "description": "Time range constraint (format: 'HH:MM:SS-HH:MM:SS' or 'SS-SS')",
                "default": None
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }
    
    def run(self, query: str, time_range: Optional[str] = None, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Execute semantic retrieval.
        
        Args:
            query: Semantic query
            time_range: Optional time constraint
            top_k: Number of results
            **kwargs: Additional arguments
            
        Returns:
            List of semantically relevant results
        """
        cache_dir = Path(settings.data_cache_dir) / "processed"
        results: List[Dict[str, Any]] = []
        load_errors: List[str] = []
        video_hint = kwargs.get("video") or kwargs.get("video_path")
        resolved_scope = sorted(runtime_session.resolve_video_scope(video_hint))
        hidden_payload = get_sparse_scope_hidden_tool_payload(self.name, video_hint=video_hint)
        if hidden_payload is not None:
            hidden_payload.update(
                {
                    "query": query,
                    "time_range": time_range,
                    "top_k": top_k,
                    "match_count": 0,
                    "matched_segments": [],
                }
            )
            return hidden_payload
        
        if not cache_dir.exists():
            return {
                "error": "No processed data found",
                "status": "error",
                "query": query,
                "time_range": time_range,
            }
            
        # Parse time range. For benchmark samples with an explicitly labelled
        # localization prior, apply it even when the planner omitted the
        # optional tool argument so a distant semantic hit cannot displace the
        # dataset's scene locator.
        effective_time_range = time_range
        time_prior = None
        if not effective_time_range:
            time_prior = _extract_query_time_prior(runtime_session.get_active_query_text())
            if time_prior is not None:
                effective_time_range = f"{time_prior[0]:.3f}-{time_prior[1]:.3f}"
        time_filter = None
        if effective_time_range:
            try:
                parts = effective_time_range.split('-')
                if len(parts) == 2:
                    # Handle HH:MM:SS or SS
                    def parse_time(t_str):
                        if ':' in t_str:
                            h, m, s = map(float, t_str.split(':'))
                            return h*3600 + m*60 + s
                        return float(t_str)
                        
                    start = parse_time(parts[0])
                    end = parse_time(parts[1])
                    time_filter = (start, end)
            except Exception as e:
                logger.warning(f"Failed to parse time range {effective_time_range}: {e}")
        
        for video_dir in _iter_scoped_video_dirs(video_hint):
            segments = _load_segments_payload(video_dir)
            if not segments:
                continue
            try:
                store = _get_cached_vector_store(video_dir)
                times_map = _load_segment_times_map(video_dir)
                graph_entities = _load_segment_graph_entities(video_dir)
                search_started = time.perf_counter()
                matches = store.search_segments(
                    query=query,
                    k=top_k,
                    time_range=time_filter
                )
                runtime_session.record_timing("retrieval.semantic_search_seconds", time.perf_counter() - search_started)
                
                for match in matches:
                    segment_ref = str(
                        match.get("segment_ref")
                        or match.get("segment_id")
                        or match.get("id")
                        or ""
                    ).strip()
                    _, raw_segment_id, _ = _parse_segment_reference(segment_ref)
                    if not raw_segment_id:
                        raw_segment_id = str(match.get("raw_segment_id", "") or "").strip()
                    seg_data = segments.get(str(raw_segment_id), {})
                    if not isinstance(seg_data, dict) or not seg_data:
                        seg_data = {
                            "time": match.get("time_range") or match.get("time") or "",
                            "caption": match.get("caption", ""),
                            "transcript": match.get("transcript", ""),
                            "frame_times": match.get("frame_times", []),
                            "frame_paths": match.get("frame_paths", []),
                            "segment_aliases": match.get("segment_aliases", []),
                        }
                    relevance = _semantic_relevance_diagnostic(
                        query=query,
                        match=match,
                        segment_payload=seg_data,
                    )
                    bundle = _build_segment_bundle(
                        video_dir,
                        raw_segment_id or segment_ref,
                        seg_data,
                        times_map=times_map,
                        graph_entities=graph_entities,
                        include_fields={"caption", "transcript", "graph"},
                        extra_fields={
                            "id": segment_ref or f"{video_dir.name}_{raw_segment_id}",
                            "query": query,
                            "match_type": "semantic",
                            "similarity_score": float(match.get("similarity_score", 0.0) or 0.0),
                            "matched_modalities": list(match.get("matched_modalities") or []),
                            "modality_scores": dict(match.get("modality_scores") or {}),
                            "relevance_label": relevance["relevance_label"],
                            "relevance_usable": relevance["relevance_usable"],
                            "lexical_overlap": relevance["lexical_overlap"],
                            "matched_tokens": relevance["matched_tokens"],
                            "evidence_strength": relevance["evidence_strength"],
                            "needs_visual_verification": relevance["needs_visual_verification"],
                            "recommended_next_tools": relevance["recommended_next_tools"],
                        },
                        text_hint=f"semantic {query}",
                    )
                    results.append(bundle)

            except Exception as e:
                logger.error(f"Error searching vectors in {video_dir}: {e}")
                load_errors.append(f"{video_dir.name}: {e}")

        results.sort(
            key=lambda item: (
                -float(item.get("similarity_score", 0.0) or 0.0),
                *_segment_sort_key(item),
            )
        )
        if results:
            candidate_segments = results[:top_k]
            candidate_probe_windows = [
                window
                for item in candidate_segments
                for window in [_build_candidate_probe_window(item)]
                if window is not None
            ]
            usable_results = [
                item for item in results
                if bool(item.get("relevance_usable", True))
            ]
            limited = usable_results[:top_k]
            relevance_diagnostics = [
                {
                    "segment_ref": item.get("segment_ref"),
                    "relevance_label": item.get("relevance_label", "unknown"),
                    "similarity_score": item.get("similarity_score", 0.0),
                    "lexical_overlap": item.get("lexical_overlap", 0.0),
                    "matched_modalities": item.get("matched_modalities", []),
                    "needs_visual_verification": bool(item.get("needs_visual_verification", False)),
                }
                for item in candidate_segments
            ]
            if not limited:
                return enrich_tool_result(
                    {
                        "query": query,
                        "time_range": time_range,
                        "top_k": top_k,
                        "resolved_video_scope": resolved_scope,
                        "match_count": 0,
                        "matched_segments": [],
                        "candidate_segments": candidate_segments,
                        "candidate_probe_windows": candidate_probe_windows,
                        "relevance_diagnostics": relevance_diagnostics,
                        "status": "no_valid_result",
                        "time_prior_applied": time_prior is not None and not time_range,
                        "recommended_next_tools": ["Inspect_Frame", "Video_Clip_QA"],
                        "result": (
                            f"Semantic_Retrieve produced {len(candidate_segments)} dense candidate segments for "
                            f"'{query}', but none is text-grounded enough to cite. The candidates require direct "
                            "visual verification; they are not answer evidence."
                        ),
                    },
                    tool_name=self.name,
                )
            payload = {
                "query": query,
                "time_range": time_range,
                "top_k": top_k,
                "resolved_video_scope": resolved_scope,
                "match_count": len(limited),
                "matched_segments": limited,
                "candidate_segments": [
                    item for item in candidate_segments
                    if not bool(item.get("relevance_usable", True))
                ],
                "candidate_probe_windows": candidate_probe_windows,
                "relevance_diagnostics": relevance_diagnostics,
                "status": "success",
                "time_prior_applied": time_prior is not None and not time_range,
                "result": (
                    f"Semantic_Retrieve found {len(limited)} text-grounded segments for '{query}'."
                    + (
                        f" Withheld {len(candidate_segments) - len(limited)} dense-only candidates pending direct visual verification."
                        if len(candidate_segments) > len(limited)
                        else ""
                    )
                ),
            }
            return enrich_tool_result(payload, tool_name=self.name)
        if load_errors:
            return {
                "error": "Failed to load multimodal vector store.",
                "details": load_errors,
                "status": "error",
                "query": query,
                "time_range": effective_time_range,
                "resolved_video_scope": resolved_scope,
            }
        return enrich_tool_result(
            {
                "query": query,
                "time_range": effective_time_range,
                "top_k": top_k,
                "resolved_video_scope": resolved_scope,
                "match_count": 0,
                "matched_segments": [],
                "status": "no_valid_result",
                "time_prior_applied": time_prior is not None and not time_range,
                "result": "Semantic_Retrieve did not provide usable evidence. Planner should try another relevant tool, broaden/narrow the time range, or reformulate the probe.",
            },
            tool_name=self.name,
        )


class VideoSummary(BaseTool):
    """Summarize the full video's preprocessed caption/transcript evidence."""

    name: str = "Video_Summary"
    description: str = (
        "Summarize all preprocessed segment captions and transcripts for the active local video scope. "
        "Uses a text-generation model and returns a concise holistic summary grounded in the video evidence."
    )
    tags: list = ["base", "retrieval", "summary"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @property
    def model_path(self) -> str:
        return settings.subagent_model_path

    def run(self, **kwargs) -> Dict[str, Any]:
        from core.llm.provider import get_llm_provider

        video_hint = kwargs.get("video") or kwargs.get("video_path")
        resolved_scope = sorted(runtime_session.resolve_video_scope(video_hint))
        segment_rows: List[Dict[str, Any]] = []

        for video_dir in _iter_scoped_video_dirs(video_hint):
            segments = _load_segments_payload(video_dir)
            times_map = _load_segment_times_map(video_dir)
            for seg_id, seg_data in segments.items():
                if not isinstance(seg_data, dict):
                    continue
                bundle = _build_segment_bundle(
                    video_dir,
                    seg_id,
                    seg_data,
                    times_map=times_map,
                    graph_entities={},
                    include_fields={"caption", "transcript"},
                    text_hint="video summary",
                )
                segment_rows.append(bundle)

        if not segment_rows:
            return {
                "status": "no_valid_result",
                "resolved_video_scope": resolved_scope,
                "segment_count": 0,
                "summary": "",
                "result": "Video_Summary did not provide usable evidence. Planner should try another relevant tool or verify preprocessing outputs.",
            }

        segment_rows.sort(key=_segment_sort_key)
        evidence_lines: List[str] = []
        for row in segment_rows:
            caption = str(row.get("caption", "") or "").strip()
            transcript = str(row.get("transcript", "") or "").strip()
            evidence_lines.append(
                "\n".join(
                    [
                        f"[Segment {row.get('segment_ref')}] time={row.get('time_range', '')}",
                        f"caption: {caption or 'None'}",
                        f"transcript: {transcript or 'None'}",
                    ]
                ).strip()
            )
        evidence_text = "\n\n".join(evidence_lines)
        prompt = "\n\n".join(
            [
                "Summarize the following video evidence.",
                "Use only the provided captions and transcripts.",
                "Write one concise paragraph that covers the main events, entities, scene changes, and any important spoken or on-screen content.",
                evidence_text,
            ]
        ).strip()

        try:
            provider = get_llm_provider(
                model_name=self.model_path,
                model_role="subagent",
            )
            raw_summary = provider.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                do_sample=False,
                enable_thinking=False,
                max_tokens=min(
                    int(getattr(settings, "subagent_max_generation_tokens", 2048) or 2048),
                    1024,
                ),
            )
            summary_text = str(raw_summary or "").strip()
        except Exception as exc:
            logger.warning("Video_Summary failed to generate model summary: %s", exc)
            return {
                "status": "error",
                "resolved_video_scope": resolved_scope,
                "segment_count": len(segment_rows),
                "error": str(exc),
                "result": "Video_Summary failed while calling the summary model.",
            }

        if not summary_text:
            return {
                "status": "no_valid_result",
                "resolved_video_scope": resolved_scope,
                "segment_count": len(segment_rows),
                "summary": "",
                "result": "Video_Summary did not provide usable evidence. Planner should try another relevant tool or verify preprocessing outputs.",
            }

        return enrich_tool_result(
            {
                "status": "success",
                "resolved_video_scope": resolved_scope,
                "segment_count": len(segment_rows),
                "summary": summary_text,
                "source_segments": [
                    {
                        "segment_ref": row.get("segment_ref"),
                        "time_range": row.get("time_range"),
                    }
                    for row in segment_rows
                ],
                "result": summary_text,
            },
            tool_name=self.name,
        )


class EvidenceVerify(BaseTool):
    """Verify whether planner evidence is sufficient and well-grounded."""

    name: str = "evidence_verify"
    description: str = (
        "Assess the planner's current evidence chain, evidence_status fields, and draft answer. "
        "Determines whether the chain is complete, whether the planner is looping or logically inconsistent, "
        "and whether finishing now would be grounded or overconfident."
    )
    tags: list = ["base", "retrieval", "verification", "planner_context"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "original_question": {
                "type": "string",
                "description": "The user's original question for the planner.",
            },
            "question": {
                "type": "string",
                "description": "Alias for original_question.",
            },
            "planner_context": {
                "type": ["string", "object", "array", "null"],
                "description": "Planner context/history including prior observations, evidence, and reasoning state.",
                "default": None,
            },
            "history": {
                "type": ["string", "object", "array", "null"],
                "description": "Alias for planner_context.",
                "default": None,
            },
            "evidence": {
                "type": ["array", "null"],
                "description": "Evidence items including label/value/citation/evidence_status.",
                "default": None,
            },
            "evidence_items": {
                "type": ["array", "null"],
                "description": "Alias for evidence.",
                "default": None,
            },
            "answer_draft": {
                "type": ["string", "null"],
                "description": "Current draft answer from the planner.",
                "default": None,
            },
            "current_answer": {
                "type": ["string", "null"],
                "description": "Alias for answer_draft.",
                "default": None,
            },
            "verification_prompt": {
                "type": ["string", "null"],
                "description": "Optional extra instruction for how to judge the evidence chain.",
                "default": None,
            },
            "user_prompt": {
                "type": ["string", "null"],
                "description": "Alias for verification_prompt.",
                "default": None,
            },
        },
        "required": ["original_question"],
    }

    @property
    def model_path(self) -> str:
        return settings.subagent_model_path

    def run(
        self,
        original_question: str,
        planner_context: Any = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        answer_draft: Optional[str] = None,
        verification_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from core.llm.provider import get_llm_provider

        question = str(
            original_question
            or kwargs.get("question")
            or kwargs.get("query")
            or ""
        ).strip()
        context_payload = planner_context
        if context_payload in (None, "", []):
            context_payload = kwargs.get("history")
        if context_payload in (None, "", []):
            context_payload = kwargs.get("context")

        explicit_evidence = evidence
        if explicit_evidence in (None, []):
            explicit_evidence = kwargs.get("evidence_items")
        evidence_items = _normalize_evidence_items(explicit_evidence)
        if not evidence_items:
            evidence_items = _extract_evidence_items(context_payload)

        answer_text = str(
            answer_draft
            or kwargs.get("current_answer")
            or kwargs.get("draft_answer")
            or kwargs.get("candidate_answer")
            or ""
        ).strip()
        verify_instruction = str(
            verification_prompt
            or kwargs.get("user_prompt")
            or kwargs.get("prompt")
            or ""
        ).strip()

        planner_context_text = _serialize_for_prompt(context_payload, max_chars=12000)
        prompt = _build_evidence_verify_prompt(
            original_question=question,
            planner_context_text=planner_context_text,
            evidence_items=evidence_items,
            answer_draft=answer_text,
            verification_prompt=verify_instruction,
        )
        parsed = None
        raw_text = ""
        try:
            provider = get_llm_provider(model_name=self.model_path, model_role="subagent")
            messages = [{"role": "user", "content": prompt}]
            for attempt in range(2):
                raw_text = str(provider.generate(messages, temperature=0.0, do_sample=False,
                    enable_thinking=False, max_tokens=min(
                        int(getattr(settings, "subagent_max_generation_tokens", 2048) or 2048), 1024)) or '').strip()
                parsed = _extract_first_json_payload(raw_text)
                if isinstance(parsed, dict) and all(
                    isinstance(parsed.get(key), bool) or str(parsed.get(key)).strip().lower() in {'true', 'false'}
                    for key in ('chain_complete', 'can_finish')
                ):
                    break
                parsed = None
                messages = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": raw_text},
                            {"role": "user", "content": 'Return the actual assessment as a JSON object with explicit boolean chain_complete and can_finish. Preserve your evidence-based judgment; do not infer approval from supplied evidence labels.'}]
        except Exception as exc:
            return {"status": "error", "error": str(exc), "chain_complete": False,
                    "can_finish": False, "result": "Verification model execution failed."}
        if parsed is None:
            return {"status": "error", "parse_status": "invalid_schema", "raw_assessment": raw_text,
                    "chain_complete": False, "can_finish": False,
                    "result": "Verification model did not return a valid assessment after format repair."}

        logic_issues = _normalize_string_list(
            parsed.get("logic_issues")
            or parsed.get("issues")
            or parsed.get("logic_gaps")
        )
        missing_evidence = _normalize_string_list(
            parsed.get("missing_evidence")
            or parsed.get("missing_slots")
            or parsed.get("gaps")
        )
        recommendation = " ".join(
            str(
                parsed.get("recommendation")
                or parsed.get("result")
                or parsed.get("summary")
                or ""
            ).split()
        ).strip()
        normalized = {
            "status": "success",
            "parse_status": "json",
            "original_question": question,
            "evidence_count": len(evidence_items),
            "chain_complete": _coerce_bool(parsed.get("chain_complete"), False),
            "can_finish": _coerce_bool(parsed.get("can_finish"), False),
            "overconfident": _coerce_bool(parsed.get("overconfident"), False),
            "has_logic_issue": _coerce_bool(parsed.get("has_logic_issue"), bool(logic_issues)),
            "loop_risk": _coerce_bool(parsed.get("loop_risk"), False),
            "confidence_level": str(
                parsed.get("confidence_level")
                or "unknown"
            ).strip() or "unknown",
            "logic_issues": logic_issues,
            "missing_evidence": missing_evidence,
            "recommendation": recommendation or "See the structured model assessment.",
            "parsed_json": parsed,
            "result": recommendation or "See the structured model assessment.",
        }
        return enrich_tool_result(normalized, tool_name=self.name)


class CleanResults(BaseTool):
    """Clean and rerank retrieval results."""
    
    name: str = "Clean_Results"
    description: str = (
        "Clean, deduplicate, and rerank retrieval results that were already returned by another "
        "retrieval tool. Requires non-empty retrieved_data and does not search the video corpus, "
        "segments, captions, or transcripts by itself."
    )
    tags: list = ["base", "retrieval", "rerank"]
    
    @property
    def model_path(self) -> str:
        return settings.query_rewrite_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "retrieved_data": {
                "type": "array",
                "description": "Raw retrieval results to clean"
            },
            "query_intent": {
                "type": "string",
                "description": "Original query intent for relevance scoring"
            }
        },
        "required": ["retrieved_data", "query_intent"]
    }
    
    def run(self, retrieved_data: List[Any], query_intent: str, **kwargs) -> List[Any]:
        """Execute result cleaning.
        
        Args:
            retrieved_data: Raw results
            query_intent: Original intent
            **kwargs: Additional arguments
            
        Returns:
            Cleaned and reranked results
        """
        video_hint = None
        for key in ("video", "video_path", "segment_id"):
            value = kwargs.get(key)
            if value not in (None, ""):
                video_hint = str(value)
                break
        hidden_payload = get_sparse_scope_hidden_tool_payload(self.name, video_hint=video_hint)
        if hidden_payload is not None:
            return hidden_payload

        normalized_retrieved_data = list(retrieved_data or [])
        if not normalized_retrieved_data:
            return enrich_tool_result(
                {
                    "status": "no_valid_result",
                    "result": (
                        "Clean_Results requires non-empty retrieved_data from a previous retrieval tool. "
                        "It cannot discover new evidence by itself, so switch to another retrieval, "
                        "verification, or direct-evidence tool instead of retrying Clean_Results."
                    ),
                    "query_intent": str(query_intent or "").strip(),
                    "input_count": 0,
                    "output_count": 0,
                },
                tool_name=self.name,
            )

        query_tokens = set(_tokenize_text(query_intent))
        target_range = _parse_time_range(kwargs.get("time_range"))

        dedup_map: Dict[str, Dict[str, Any]] = {}
        passthrough: List[Any] = []

        for item in normalized_retrieved_data:
            if not isinstance(item, dict):
                passthrough.append(item)
                continue

            if item.get("error"):
                continue

            item_id = item.get("id") or item.get("url") or item.get("segment_id")
            if not item_id:
                item_id = json.dumps(item, sort_keys=True, ensure_ascii=False)

            text_fields = " ".join(
                str(item.get(k, ""))
                for k in ("title", "snippet", "description", "caption", "transcript", "content")
            )
            text_tokens = set(_tokenize_text(text_fields))
            lexical = 0.0
            if query_tokens:
                lexical = len(query_tokens.intersection(text_tokens)) / len(query_tokens)

            base = float(item.get("score", item.get("similarity_score", item.get("confidence", 0.0))) or 0.0)
            cand_range = _parse_time_range(item.get("time_range") or item.get("time"))
            overlap = _time_overlap_ratio(cand_range, target_range)

            rank_score = 0.55 * base + 0.35 * lexical + 0.10 * overlap
            enriched = dict(item)
            enriched["rank_score"] = rank_score

            prev = dedup_map.get(str(item_id))
            if prev is None or enriched["rank_score"] > prev.get("rank_score", -1.0):
                dedup_map[str(item_id)] = enriched

        ranked = sorted(dedup_map.values(), key=lambda x: x.get("rank_score", 0.0), reverse=True)
        cleaned: List[Any] = ranked + passthrough
        logger.info(f"Clean_Results: {len(normalized_retrieved_data)} -> {len(cleaned)} items")
        if not cleaned:
            return enrich_tool_result(
                {
                    "status": "no_valid_result",
                    "result": (
                        "Clean_Results did not retain any usable evidence after deduplication/reranking. "
                        "Treat the current tool as ineffective for this branch and switch to another tool "
                        "instead of retrying Clean_Results with adjusted parameters."
                    ),
                    "query_intent": str(query_intent or "").strip(),
                    "input_count": len(normalized_retrieved_data),
                    "output_count": 0,
                },
                tool_name=self.name,
            )
        return cleaned


class WebSearch(BaseTool):
    """Web search for external information."""
    
    name: str = "Web_Search"
    description: str = (
        "Perform open-domain web search to retrieve real-time information and general "
        "background knowledge beyond the local video library."
    )
    tags: list = ["base", "retrieval", "web"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }
    
    def run(self, query: str, num_results: int = 5, **kwargs) -> List[Dict[str, str]]:
        """Execute web search using DuckDuckGo.
        
        Args:
            query: Search query
            num_results: Number of results
            **kwargs: Additional arguments
            
        Returns:
            List of search results
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("duckduckgo_search not installed. Please install it with: pip install duckduckgo-search")
            return [{"error": "duckduckgo_search library not found"}]

        logger.info(f"Web_Search for: {query}")
        results = []
        try:
            with DDGS() as ddgs:
                # Use text search from duckduckgo_search
                ddgs_gen = ddgs.text(query, max_results=num_results)
                if ddgs_gen:
                    for r in ddgs_gen:
                        results.append({
                            "title": r.get('title', ''),
                            "url": r.get('href', ''),
                            "snippet": r.get('body', '')
                        })
        except Exception as e:
            logger.error(f"Web_Search failed: {str(e)}")
            return [{"error": f"Search failed: {str(e)}"}]
            
        return results


class ASRRetrieve(BaseTool):
    """Locate ASR spans for a requested time window."""

    name: str = "asr"
    description: str = (
        "Locate subtitle or ASR transcript spans from preprocessing outputs for a requested "
        "time range, and return the matched text together with utterance-level timestamps."
    )
    tags: list = ["base", "retrieval", "asr", "transcript"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "time_range": {
                "type": "string",
                "description": "Required target time range (for example '30-60' or '00:01:00-00:02:00').",
            },
        },
        "required": ["time_range"],
    }

    def run(self, time_range: str, **kwargs) -> Dict[str, Any]:
        target_range = _parse_time_range(time_range)
        video_hint = kwargs.get("video") or kwargs.get("video_path")

        if target_range is None:
            return {
                "error": "Invalid or missing time_range. Expected formats like '30-60' or '00:01:00-00:02:00'.",
                "status": "error",
                "time_range": time_range,
            }

        cache_dir = Path(settings.data_cache_dir) / "processed"
        if not cache_dir.exists():
            return {
                "error": "No processed data found. Run preprocessing first.",
                "status": "error",
                "time_range": time_range,
            }

        hidden_payload = get_sparse_scope_hidden_tool_payload(self.name, video_hint=video_hint)
        if hidden_payload is not None:
            hidden_payload.update(
                {
                    "time_range": time_range,
                    "match_count": 0,
                    "matched_entries": [],
                    "transcript": "",
                }
            )
            return hidden_payload

        resolved_scope = sorted(runtime_session.resolve_video_scope(video_hint))
        matched_entries: List[Dict[str, Any]] = []
        seen_matches: set[Tuple[str, float, float, str]] = set()

        for video_dir in _iter_scoped_video_dirs(video_hint):
            entries = _load_cached_full_asr_entries(video_dir)
            source = "cached_full_transcript"
            if not entries:
                entries = _collect_processed_asr_entries(video_dir)
                source = "processed_transcripts"

            for entry in entries:
                entry_range = (
                    float(entry.get("start", 0.0) or 0.0),
                    float(entry.get("end", entry.get("start", 0.0)) or 0.0),
                )
                if _time_overlap_ratio(entry_range, target_range) <= 0:
                    continue

                match_key = (
                    video_dir.name,
                    float(entry["start"]),
                    float(entry["end"]),
                    str(entry["text"]),
                )
                if match_key in seen_matches:
                    continue
                seen_matches.add(match_key)
                matched_entries.append(
                    {
                        "video": video_dir.name,
                        "start": float(entry["start"]),
                        "end": float(entry["end"]),
                        "time": str(entry["time"]),
                        "text": str(entry["text"]),
                        "source": source,
                    }
                )

        matched_entries.sort(
            key=lambda item: (
                str(item.get("video", "")),
                float(item.get("start", 0.0) or 0.0),
                float(item.get("end", 0.0) or 0.0),
            )
        )

        if matched_entries:
            transcript = "\n".join(
                f"[{float(item['start']):.2f}s -> {float(item['end']):.2f}s] {str(item['text'])}"
                for item in matched_entries
            )
            return enrich_tool_result(
                {
                    "resolved_video_scope": resolved_scope,
                    "time_range": time_range,
                    "match_count": len(matched_entries),
                    "matched_entries": matched_entries,
                    "transcript": transcript + ("\n" if transcript else ""),
                    "status": "success",
                    "result": (
                        f"asr returned {len(matched_entries)} matching subtitle span(s) "
                        f"for time_range={time_range}."
                    ),
                },
                tool_name=self.name,
            )

        return {
            "status": "no_valid_result",
            "resolved_video_scope": resolved_scope,
            "time_range": time_range,
            "match_count": 0,
            "matched_entries": [],
            "transcript": "",
            "result": "asr did not provide usable evidence. Planner should try another relevant tool, broaden/narrow the time range, or reformulate the probe.",
        }


class SegmentRetrieve(BaseTool):
    """Retrieve segment-level caption, transcript, and graph data by ID or time range."""

    name: str = "Segment_Retrieve"
    description: str = (
        "Retrieve a segment's preprocessed evidence bundle by segment ID or time range. "
        "Returns caption, transcript, timestamps, cached frame images, and optional graph entities."
    )
    tags: list = ["base", "retrieval", "segment"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "segment_id": {
                "type": "string",
                "description": "Segment identifier. Supports raw IDs like '7' and composite IDs like 'example_7'.",
                "default": None,
            },
            "time_range": {
                "type": "string",
                "description": "Time range to match (e.g. '30-60' or '00:01:00-00:02:00'). Intersects with segment_id filters when both are provided.",
                "default": None,
            },
            "fields": {
                "type": ["array", "string", "null"],
                "items": {"type": "string"},
                "description": (
                    "Which fields to return. Allowed values: "
                    "['caption', 'transcript', 'graph', 'frame_times', 'frame_paths']. "
                    "May be provided as a list or comma-separated string."
                ),
                "default": ["caption", "transcript", "graph", "frame_times", "frame_paths"],
            },
        },
        "required": [],
    }

    def run(
        self,
        segment_id: Optional[str] = None,
        time_range: Optional[str] = None,
        fields: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from pathlib import Path
        from config.settings import settings

        normalized_fields = _normalize_segment_fields(fields or ["caption", "transcript", "graph", "frame_times", "frame_paths"])
        want = set(normalized_fields or ["caption", "transcript", "graph", "frame_times", "frame_paths"])

        cache_dir = Path(settings.data_cache_dir) / "processed"
        if not cache_dir.exists():
            return {
                "error": "No processed data found. Run preprocessing first.",
                "status": "error",
                "requested_segment_id": segment_id,
                "time_range": time_range,
                "normalized_fields": sorted(want),
            }

        target_range = _parse_time_range(time_range)
        results: List[Dict[str, Any]] = []
        requested_video, resolved_segment_id, requested_aliases = _parse_segment_reference(segment_id)
        video_hint = requested_video or kwargs.get("video") or kwargs.get("video_path")
        hidden_payload = get_sparse_scope_hidden_tool_payload(self.name, video_hint=video_hint)
        if hidden_payload is not None:
            hidden_payload.update(
                {
                    "requested_segment_id": segment_id,
                    "resolved_segment_id": resolved_segment_id,
                    "resolved_video": requested_video,
                    "time_range": time_range,
                    "normalized_fields": sorted(want),
                    "available_fields": ["caption", "transcript", "graph", "frame_times", "frame_paths"],
                    "match_count": 0,
                    "matched_segments": [],
                }
            )
            return hidden_payload
        resolved_scope = sorted(runtime_session.resolve_video_scope(video_hint))

        for video_dir in _iter_scoped_video_dirs(video_hint):
            segments = _load_segments_payload(video_dir)
            if not segments:
                continue
            times_map = _load_segment_times_map(video_dir)
            graph_entities = _load_segment_graph_entities(video_dir) if "graph" in want else {}

            for sid, seg in segments.items():
                if not isinstance(seg, dict):
                    continue
                time_info = times_map.get(str(sid), {}) if isinstance(times_map, dict) else {}
                aliases = [str(alias) for alias in (time_info.get("segment_aliases") or seg.get("segment_aliases") or [str(sid), f"{video_dir.name}_{sid}"]) if str(alias).strip()]

                if segment_id is not None:
                    if str(sid) != str(resolved_segment_id) and str(segment_id) not in aliases and not set(aliases).intersection(requested_aliases):
                        continue
                if target_range is not None:
                    seg_range = _parse_time_range(seg.get("time"))
                    if seg_range is None or _time_overlap_ratio(seg_range, target_range) <= 0:
                        continue

                entry = _build_segment_bundle(
                    video_dir,
                    sid,
                    seg,
                    times_map=times_map,
                    graph_entities=graph_entities,
                    include_fields=want,
                    text_hint=f"segment {video_dir.name}_{sid}",
                )
                results.append(entry)

        logger.info(
            "Segment_Retrieve(segment_id=%s, time_range=%s): %d segments returned",
            segment_id, time_range, len(results),
        )
        results.sort(key=_segment_sort_key)
        if results:
            return enrich_tool_result(
                {
                    "requested_segment_id": segment_id,
                    "resolved_segment_id": resolved_segment_id,
                    "resolved_video": requested_video,
                    "resolved_video_scope": resolved_scope,
                    "time_range": time_range,
                    "normalized_fields": sorted(want),
                    "match_count": len(results),
                    "matched_segments": results,
                    "status": "success",
                    "result": (
                        f"Segment_Retrieve returned {len(results)} matching segments "
                        f"for segment_id={segment_id} time_range={time_range}."
                    ),
                },
                tool_name=self.name,
            )
        return {
            "error": "No matching segments found.",
            "status": "no_valid_result",
            "requested_segment_id": segment_id,
            "resolved_segment_id": resolved_segment_id,
            "resolved_video": requested_video,
            "resolved_video_scope": resolved_scope,
            "time_range": time_range,
            "normalized_fields": sorted(want),
            "available_fields": ["caption", "transcript", "graph", "frame_times", "frame_paths"],
            "result": "Segment_Retrieve did not provide usable evidence. Planner should try another relevant tool, broaden/narrow the time range, or reformulate the probe.",
        }
