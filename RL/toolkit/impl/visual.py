"""Visual and audio analysis tools."""
import gc
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageFile

from config.settings import settings
from preprocessing.video.frame_assets import export_video_subclip
from runtime.session import runtime_session
from toolkit.base import BaseTool, enrich_tool_result, summarize_tool_result
from .visual_prompts import (
    build_video_clip_user_message,
    looks_like_truncated_video_clip_qa_answer,
    normalize_video_clip_qa_answer,
    strip_think_blocks,
)

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

_VISUAL_PROCESSORS: Dict[str, Any] = {}
_PROCESSED_SEGMENT_ROWS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_VIDEO_TIMING_CACHE: Dict[str, Dict[str, float]] = {}
PLANNER_IMAGE_ORDER_RULE_TEXT = (
    "If an attached image is a multi-frame collage, frames are arranged in chronological order "
    "clockwise from the top-left: top-left, top-right, bottom-right, bottom-left."
)


class RequestedVideoTimeRangeError(ValueError):
    """Raised when a requested timestamp/window starts beyond the source video length."""


def _strip_think_blocks(text: Any) -> str:
    """Remove <think>...</think> content while preserving the remaining answer."""
    return strip_think_blocks(text)


_VIDEO_QA_DIMENSION_PATTERNS = {
    "action": (
        "action",
        "movement",
        "motion",
        "gesture",
        "pose",
        "dance",
        "doing",
    ),
    "style": (
        "style",
        "genre",
        "type",
        "category",
        "kind",
        "classify",
        "classification",
    ),
    "audio_caption": (
        "audio",
        "sound",
        "music",
        "subtitle",
        "caption",
        "transcript",
        "lyrics",
        "spoken",
        "say",
        "ocr",
        "text on screen",
    ),
    "scene_distinction": (
        "scene",
        "background",
        "setting",
        "compare",
        "difference",
        "distinguish",
        "versus",
        "vs",
        "before",
        "after",
        "transition",
        "between",
    ),
}


def _normalize_summary_snippet(text: Any, *, limit: int = 220) -> str:
    snippet = " ".join(str(text or "").split()).strip()
    if not snippet:
        return ""
    if len(snippet) <= limit:
        return snippet
    return snippet[: max(0, limit - 3)].rstrip() + "..."


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    normalized_items: List[str] = []
    seen = set()
    for item in items:
        snippet = _normalize_summary_snippet(item)
        if not snippet:
            continue
        marker = snippet.lower()
        if marker in seen:
            continue
        seen.add(marker)
        normalized_items.append(snippet)
    return normalized_items


def _detect_video_qa_requested_dimensions(query: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    if not normalized:
        return []
    detected: List[str] = []
    for dimension, keywords in _VIDEO_QA_DIMENSION_PATTERNS.items():
        for keyword in keywords:
            if keyword in normalized:
                detected.append(dimension)
                break
    return detected


def _extract_video_qa_timestamped_evidence(answer: str, *, limit: int = 4) -> List[str]:
    snippets: List[str] = []
    lines = [line.strip(" -\t") for line in str(answer or "").splitlines() if line.strip()]
    for line in lines:
        if re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", line):
            snippets.append(line)
        if len(snippets) >= limit:
            break
    return _dedupe_preserve_order(snippets)[:limit]


def _extract_video_qa_quoted_cues(answer: str, *, limit: int = 4) -> List[str]:
    matches = re.findall(r"[\"“”'']([^\"“”''\n]{2,160})[\"“”'']", str(answer or ""))
    return _dedupe_preserve_order(matches)[:limit]


def _extract_video_qa_matching_lines(answer: str, keywords: List[str], *, limit: int = 4) -> List[str]:
    snippets: List[str] = []
    lines = [line.strip(" -\t") for line in str(answer or "").splitlines() if line.strip()]
    for line in lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            snippets.append(line)
        if len(snippets) >= limit:
            break
    if snippets:
        return _dedupe_preserve_order(snippets)[:limit]

    sentences = re.split(r"(?<=[\.\!\?])\s+", str(answer or ""))
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            snippets.append(sentence)
        if len(snippets) >= limit:
            break
    return _dedupe_preserve_order(snippets)[:limit]


def _build_video_qa_structured_summary(query: str, answer: str) -> Dict[str, Any]:
    cleaned_answer = _strip_think_blocks(answer)
    requested_dimensions = _detect_video_qa_requested_dimensions(query)
    timestamped_evidence = _extract_video_qa_timestamped_evidence(cleaned_answer)
    audio_caption_cues = _extract_video_qa_quoted_cues(cleaned_answer)
    classification_signals = _extract_video_qa_matching_lines(
        cleaned_answer,
        ["style", "genre", "type", "category", "classif", "fits", "resembles"],
    )
    distinction_cues = _extract_video_qa_matching_lines(
        cleaned_answer,
        [" not ", "instead", "while", "versus", "vs", "rather than", "exclude", "different"],
    )

    key_findings = _dedupe_preserve_order(
        classification_signals[:2] + timestamped_evidence[:2] + distinction_cues[:2]
    )
    if not key_findings:
        key_findings = _extract_video_qa_matching_lines(
            cleaned_answer,
            ["shows", "appears", "seen", "visible", "indicates", "suggests"],
            limit=3,
        )
    if not key_findings:
        fallback = re.split(r"(?<=[\.\!\?])\s+", cleaned_answer)
        key_findings = _dedupe_preserve_order(fallback[:2])

    summary_parts: List[str] = []
    if requested_dimensions:
        summary_parts.append("requested_dimensions=" + ", ".join(requested_dimensions))
    if key_findings:
        summary_parts.append("key_findings=" + " | ".join(key_findings[:3]))
    if timestamped_evidence:
        summary_parts.append("timestamped_evidence=" + " | ".join(timestamped_evidence[:2]))
    if audio_caption_cues:
        summary_parts.append("audio_or_text_cues=" + " | ".join(audio_caption_cues[:2]))
    if distinction_cues:
        summary_parts.append("distinctions=" + " | ".join(distinction_cues[:2]))

    summary_text = "; ".join(part for part in summary_parts if part).strip()
    if not summary_text:
        summary_text = _normalize_summary_snippet(cleaned_answer, limit=320)

    compressed_parts: List[str] = []
    if requested_dimensions:
        compressed_parts.append("dims=" + ",".join(requested_dimensions))
    if key_findings:
        compressed_parts.append("findings=" + " | ".join(key_findings[:2]))
    compressed_summary = "; ".join(compressed_parts).strip() or _normalize_summary_snippet(
        cleaned_answer,
        limit=200,
    )

    return {
        "summary": summary_text,
        "compressed_summary": compressed_summary,
        "structured_evidence": {
            "requested_dimensions": requested_dimensions,
            "key_findings": key_findings[:4],
            "timestamped_evidence": timestamped_evidence,
            "audio_caption_cues": audio_caption_cues,
            "classification_signals": classification_signals,
            "distinction_cues": distinction_cues,
        },
    }


def _coerce_numeric_scalar_or_list(name: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_coerce_numeric_scalar_or_list(name, item) for item in value]
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception as exc:
        raise ValueError(
            f"{name} must be numeric seconds-per-frame (for example 0.5 or [0.5, 1.0]), "
            f"but received {value!r}."
        ) from exc


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


def _parse_range(value: Any) -> Optional[Dict[str, float]]:
    if not value:
        return None
    if isinstance(value, dict) and "start" in value and "end" in value:
        start = _parse_time_seconds(value.get("start"))
        end = _parse_time_seconds(value.get("end"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start = _parse_time_seconds(value[0])
        end = _parse_time_seconds(value[1])
    else:
        text = str(value).strip()
        if "-" not in text:
            return None
        left, right = text.split("-", 1)
        start = _parse_time_seconds(left)
        end = _parse_time_seconds(right)
    if start is None or end is None:
        return None
    return {"start": min(start, end), "end": max(start, end)}


def _format_range(start: float, end: float) -> str:
    return f"{int(start)}-{int(end)}"


def _format_precise_time(value: Any) -> str:
    parsed = _parse_time_seconds(value)
    if parsed is None:
        return str(value)
    rounded = round(float(parsed), 3)
    if abs(rounded - round(rounded)) < 1e-6:
        return f"{rounded:.1f}"
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def _format_exception_message(exc: Exception) -> str:
    text = str(exc or "").strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return repr(exc)


def _load_rgb_image(image_path: Any, *, context: str) -> Image.Image:
    resolved_path = Path(str(image_path or "")).expanduser().resolve()
    try:
        with Image.open(resolved_path) as image:
            image.load()
            return image.convert("RGB").copy()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {context} image from {resolved_path}: {_format_exception_message(exc)}"
        ) from exc


def _is_readable_image(path: Path) -> bool:
    try:
        _load_rgb_image(str(path), context="cached")
        return True
    except Exception:
        return False


def _recover_packed_image_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_path = str(item.get("image", "") or "").strip()
    timestamp = item.get("timestamp")
    sample_times = [
        round(float(value), 3)
        for value in (item.get("sample_times") or [])
        if value is not None
    ]
    source_images = [
        str(Path(source).expanduser().resolve())
        for source in (item.get("source_images") or [])
        if str(source).strip()
    ]
    if image_path:
        resolved_image_path = Path(image_path).expanduser().resolve()
        if resolved_image_path.exists() and _is_readable_image(resolved_image_path):
            normalized = dict(item)
            normalized["image"] = str(resolved_image_path)
            normalized["source_images"] = source_images or [str(resolved_image_path)]
            return normalized

    logger.warning(
        "Unreadable packed image encountered. image=%s timestamp=%s source_images=%s. Trying to rebuild it.",
        image_path,
        timestamp,
        source_images,
    )
    readable_source_images: List[str] = []
    for source_image in source_images:
        source_path = Path(source_image)
        if not source_path.exists() or not _is_readable_image(source_path):
            continue
        readable_source_images.append(str(source_path))

    if len(readable_source_images) > 1 and image_path:
        rebuilt_chunk: List[Dict[str, Any]] = []
        for index, source_image in enumerate(readable_source_images):
            rebuilt_chunk.append(
                {
                    "image": source_image,
                    "timestamp": sample_times[index] if index < len(sample_times) else None,
                    "source_images": [source_image],
                }
            )
        rebuilt_collage_path = Path(image_path).expanduser().resolve()
        rebuilt_collage = _build_probe_image_collage(rebuilt_chunk, rebuilt_collage_path)
        if rebuilt_collage:
            rebuilt_item = dict(item)
            rebuilt_item["image"] = rebuilt_collage
            rebuilt_item["source_images"] = readable_source_images
            rebuilt_item["sample_times"] = [
                chunk_item["timestamp"]
                for chunk_item in rebuilt_chunk
                if chunk_item.get("timestamp") is not None
            ]
            rebuilt_item["timestamp"] = (
                rebuilt_item["sample_times"][0]
                if rebuilt_item["sample_times"]
                else None
            )
            logger.warning(
                "Recovered unreadable packed image %s by rebuilding collage from %d readable source images.",
                image_path,
                len(readable_source_images),
            )
            return rebuilt_item

    for source_image in readable_source_images:
        source_path = Path(source_image)
        rebuilt = _persist_standardized_single_image(
            str(source_path),
            timestamp=timestamp,
            source_images=readable_source_images or [str(source_path)],
        )
        if rebuilt is None:
            continue
        rebuilt_path = Path(str(rebuilt.get("image", "") or "")).expanduser().resolve()
        if rebuilt_path.exists() and _is_readable_image(rebuilt_path):
            logger.warning(
                "Recovered unreadable packed image %s by rebuilding from source %s.",
                image_path,
                source_path,
            )
            return rebuilt

    logger.warning(
        "Skipping unreadable packed image. image=%s timestamp=%s source_images=%s. "
        "The skipped image timestamp will not be included in planner input.",
        image_path,
        timestamp,
        source_images,
    )
    return None


def _release_visual_runtime_memory() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _overlap(candidate: Optional[Dict[str, float]], target: Optional[Dict[str, float]]) -> bool:
    if target is None:
        return True
    if candidate is None:
        return False
    return min(candidate["end"], target["end"]) >= max(candidate["start"], target["start"])


def _iter_processed_segments(video_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    cache_dir = Path(settings.data_cache_dir).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (settings.project_root / cache_dir).resolve()
    cache_dir = cache_dir / "processed"
    rows: List[Dict[str, Any]] = []
    if not cache_dir.exists():
        return rows

    scoped_names = runtime_session.resolve_video_scope(video_hint)
    cache_key = f"{cache_dir.resolve()}::{','.join(sorted(scoped_names or []))}"
    cached_rows = _PROCESSED_SEGMENT_ROWS_CACHE.get(cache_key)
    if cached_rows is not None:
        return cached_rows

    for video_dir in cache_dir.iterdir():
        if not video_dir.is_dir():
            continue
        if scoped_names and video_dir.name not in scoped_names:
            continue
        segments_path = video_dir / "segments.json"
        if not segments_path.exists():
            continue
        segment_times_payload = {}
        segment_times_path = video_dir / "segment_times_info.json"
        if segment_times_path.exists():
            try:
                with open(segment_times_path, "r", encoding="utf-8") as f:
                    segment_times_payload = json.load(f) or {}
            except Exception as exc:
                logger.warning("Failed to read segment_times_info from %s: %s", video_dir, exc)

        video_metadata = {}
        video_metadata_path = video_dir / "video_metadata.json"
        if video_metadata_path.exists():
            try:
                with open(video_metadata_path, "r", encoding="utf-8") as f:
                    video_metadata = json.load(f) or {}
            except Exception as exc:
                logger.warning("Failed to read video_metadata from %s: %s", video_dir, exc)

        times_map = segment_times_payload.get("segment_times_info", {}) if isinstance(segment_times_payload, dict) else {}
        source_video_path = _resolve_video_file_path(
            video_dir.name,
            preferred_path=video_metadata.get("source_video_path") if isinstance(video_metadata, dict) else None,
        )
        try:
            with open(segments_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read segments from %s: %s", video_dir, exc)
            continue

        if not isinstance(segments, dict):
            continue

        for seg_id, seg_data in segments.items():
            if not isinstance(seg_data, dict):
                continue
            time_info = times_map.get(str(seg_id), {}) if isinstance(times_map, dict) else {}
            caption = str(seg_data.get("caption", ""))
            transcript = str(seg_data.get("transcript", ""))
            ocr = str(seg_data.get("ocr", ""))
            time_text = str(seg_data.get("time", ""))
            frame_times = time_info.get("frame_times", seg_data.get("frame_times", []))
            frame_paths = time_info.get("frame_paths", seg_data.get("frame_paths", []))
            segment_aliases = time_info.get("segment_aliases", seg_data.get("segment_aliases", []))
            timestamp = time_info.get("timestamp")
            parsed_time = _parse_range(timestamp) or _parse_range(time_text)
            segment_video_path = _resolve_processed_video_asset_path(
                video_dir,
                seg_data.get("segment_video_path") or time_info.get("segment_video_path"),
            )
            rows.append(
                {
                    "video": video_dir.name,
                    "segment_id": str(seg_id),
                    "segment_ref": f"{video_dir.name}_{seg_id}",
                    "time": time_text,
                    "time_parsed": parsed_time,
                    "timestamp": timestamp,
                    "frame_times": frame_times,
                    "frame_paths": frame_paths,
                    "segment_aliases": segment_aliases,
                    "video_path": source_video_path,
                    "segment_video_path": segment_video_path,
                    "caption": caption,
                    "transcript": transcript,
                    "ocr": ocr,
                    "text": f"{caption} {transcript} {ocr}".strip(),
                }
            )
    _PROCESSED_SEGMENT_ROWS_CACHE[cache_key] = rows
    return rows


def _row_reference_aliases(row: Dict[str, Any]) -> set[str]:
    aliases = {
        str(alias).strip()
        for alias in (row.get("segment_aliases") or [])
        if str(alias).strip()
    }
    segment_id = str(row.get("segment_id", "")).strip()
    video = str(row.get("video", "")).strip()
    if segment_id:
        aliases.add(segment_id)
    if video and segment_id:
        aliases.add(f"{video}_{segment_id}")
    return aliases


def _resolve_reference_target(value: Any, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return None

    candidates = [text]
    for prefix in ("segment_ref=", "segment_id=", "raw_segment_id="):
        if text.lower().startswith(prefix):
            candidates.append(text.split("=", 1)[1].strip())
    candidates = [candidate for candidate in candidates if candidate]

    best_row: Optional[Dict[str, Any]] = None
    best_score = -1
    for row in rows:
        aliases = _row_reference_aliases(row)
        if not aliases:
            continue
        for candidate in candidates:
            if candidate not in aliases:
                continue
            score = 3 if "_" in candidate else 2
            if score > best_score:
                best_row = row
                best_score = score

    if not best_row:
        return None
    target = best_row.get("time_parsed")
    if not isinstance(target, dict) or "start" not in target or "end" not in target:
        return None
    return {
        "target": {"start": float(target["start"]), "end": float(target["end"])},
        "probe_label": text,
    }


def _resolve_segment_row_reference(value: Any, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return None

    candidates = [text]
    for prefix in ("segment_ref=", "segment_id=", "raw_segment_id="):
        if text.lower().startswith(prefix):
            candidates.append(text.split("=", 1)[1].strip())
    candidates = [candidate for candidate in candidates if candidate]

    best_row: Optional[Dict[str, Any]] = None
    best_score = -1
    for row in rows:
        aliases = _row_reference_aliases(row)
        segment_ref = str(row.get("segment_ref", "")).strip()
        if segment_ref:
            aliases.add(segment_ref)
        if not aliases:
            continue
        for candidate in candidates:
            if candidate not in aliases:
                continue
            score = 3 if "_" in candidate or candidate.startswith("segment_") else 2
            if score > best_score:
                best_row = row
                best_score = score
    return best_row


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", (text or "").lower()) if t]


def _match_score(text: str, query: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(text_tokens))
    return overlap / max(1, len(query_tokens))


def _resolve_video_file_path(video_name: str, preferred_path: Optional[str] = None) -> Optional[str]:
    candidates: List[Path] = []
    if preferred_path:
        candidates.append(Path(preferred_path))

    roots = [
        Path(settings.data_cache_dir),
        Path(settings.data_cache_dir).parent,
        Path.cwd(),
        Path(__file__).resolve().parents[2] / "tests",
    ]
    exts = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
    for root in roots:
        for ext in exts:
            candidates.append(root / f"{video_name}{ext}")

    seen: set = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.exists():
                return str(candidate)
        except Exception:
            continue
    return None


def _resolve_processed_video_asset_path(video_dir: Path, value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                (video_dir / candidate).resolve(),
                (Path(settings.data_cache_dir).expanduser() / candidate).resolve(),
                (settings.project_root / candidate).resolve(),
            ]
        )

    for option in candidates:
        try:
            if option.exists():
                return str(option.resolve())
        except Exception:
            continue
    return None


def _build_direct_video_row(
    video_hint: Optional[str],
    target_range: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    hint = str(video_hint or "").strip()
    if not hint:
        return None

    candidate = Path(hint).expanduser()
    resolved_path: Optional[str]
    if candidate.exists():
        resolved_path = str(candidate.resolve())
    else:
        resolved_path = _resolve_video_file_path(candidate.stem or hint, preferred_path=hint)
    if not resolved_path:
        return None

    duration = 0.0
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip

        with VideoFileClip(resolved_path) as video:
            duration = float(getattr(video, "duration", 0.0) or 0.0)
    except Exception as exc:
        logger.warning("Failed to inspect raw video duration for %s: %s", resolved_path, exc)

    if duration <= 0 and target_range:
        duration = max(float(target_range.get("end", 0.0) or 0.0), 1.0)
    if duration <= 0:
        duration = 1.0

    video_name = Path(resolved_path).stem
    return {
        "video": video_name,
        "segment_id": "raw",
        "segment_ref": f"{video_name}_raw",
        "time": _format_range(0.0, duration),
        "time_parsed": {"start": 0.0, "end": duration},
        "timestamp": None,
        "frame_times": [],
        "frame_paths": [],
        "segment_aliases": [f"{video_name}_raw", "raw"],
        "video_path": resolved_path,
        "caption": "",
        "transcript": "",
        "ocr": "",
        "text": "",
    }


def _resolve_video_clip_qa_target_range(
    *,
    start: float,
    end: float,
    segment_id: Any = None,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    target_range = {
        "start": min(float(start), float(end)),
        "end": max(float(start), float(end)),
    }
    result = {
        "target_range": target_range,
        "requested_segment_id": str(segment_id) if segment_id not in (None, "", [], {}) else None,
        "resolved_segment_id": None,
        "resolved_segment_ref": None,
    }
    if segment_id in (None, "", [], {}):
        return result

    segment_row = _resolve_segment_row_reference(segment_id, rows or [])
    if not isinstance(segment_row, dict):
        return result
    result["resolved_segment_id"] = str(segment_row.get("segment_id", "")).strip() or None
    result["resolved_segment_ref"] = str(segment_row.get("segment_ref", "")).strip() or None

    segment_range = segment_row.get("time_parsed") if isinstance(segment_row.get("time_parsed"), dict) else None
    if not isinstance(segment_range, dict) or "start" not in segment_range or "end" not in segment_range:
        return result

    segment_start = float(segment_range["start"])
    segment_duration = max(0.0, float(segment_range["end"]) - segment_start)
    if (
        0.0 <= target_range["start"] <= segment_duration + 1e-6
        and 0.0 <= target_range["end"] <= segment_duration + 1e-6
    ):
        result["target_range"] = {
            "start": round(segment_start + target_range["start"], 3),
            "end": round(segment_start + target_range["end"], 3),
        }
    return result


def _range_length(value: Optional[Dict[str, float]]) -> float:
    if not value:
        return 0.0
    return max(0.0, float(value["end"]) - float(value["start"]))


def _sample_count_from_density(window_duration: float, density: Optional[float]) -> Optional[int]:
    if density is None:
        return None
    density_value = max(float(density), 1e-3)
    if window_duration <= 0:
        return 1
    return max(1, int(math.ceil(float(window_duration) / density_value - 1e-9)))


def _range_contains(
    outer: Optional[Dict[str, float]],
    inner: Optional[Dict[str, float]],
    *,
    tolerance: float = 0.05,
) -> bool:
    if not outer or not inner:
        return False
    return (
        float(inner["start"]) >= float(outer["start"]) - tolerance
        and float(inner["end"]) <= float(outer["end"]) + tolerance
    )


def _is_list_like(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if _is_list_like(value):
        return list(value)
    return [value]


def _is_scalar_time_range_pair(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    if any(isinstance(item, (list, tuple, dict)) for item in value):
        return False
    return _parse_time_seconds(value[0]) is not None and _parse_time_seconds(value[1]) is not None


def _normalize_time_range_probes(value: Any) -> List[Any]:
    if value is None:
        return []
    if _is_scalar_time_range_pair(value):
        return [list(value)]
    values = _coerce_list(value)
    if len(values) == 2 and all(_parse_time_seconds(item) is not None for item in values):
        return [list(values)]
    return values


def _probe_label(value: Any, default: str = "full_video") -> str:
    parsed = _parse_range(value)
    if parsed is not None:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"{_format_precise_time(value[0])}-{_format_precise_time(value[1])}"
        if isinstance(value, dict) and "start" in value and "end" in value:
            return f"{_format_precise_time(value.get('start'))}-{_format_precise_time(value.get('end'))}"
        text = str(value or "").strip()
        return text or f"{_format_precise_time(parsed['start'])}-{_format_precise_time(parsed['end'])}"
    text = str(value or "").strip()
    return text or default


def _coerce_broadcast_list(value: Any, size: int, default: Any = None) -> List[Any]:
    if size <= 0:
        return []
    values = _coerce_list(value)
    if not values:
        return [default] * size
    if len(values) == 1 and size > 1:
        return values * size
    if len(values) != size:
        raise ValueError(f"Expected {size} values but received {len(values)}.")
    return values


def _build_batch_result(tool_name: str, probes: List[Dict[str, Any]], input_kind: str = "probes") -> Dict[str, Any]:
    def _probe_status(probe: Dict[str, Any]) -> str:
        status = str(probe.get("status", "")).strip().lower()
        if status in {"success", "error", "no_valid_result"}:
            return status
        text = " ".join(
            str(
                probe.get("result")
                or probe.get("findings")
                or probe.get("extracted_text")
                or ""
            ).split()
        ).strip().lower()
        if probe.get("warning", "").startswith("Failed"):
            return "error"
        if any(
            marker in text
            for marker in (
                "did not provide usable evidence",
                "visual caption unavailable",
                "no legible text detected",
                "no processed segment evidence matched",
            )
        ):
            return "no_valid_result"
        return "success" if text else "no_valid_result"

    successful = [probe for probe in probes if _probe_status(probe) == "success"]
    summary_parts = []
    probe_source = successful[:5] if successful else probes[:3]
    for probe in probe_source:
        label = probe.get("probe_label") or probe.get("search_window") or probe.get("time_range") or probe.get("timestamp")
        text = probe.get("result") or probe.get("findings") or probe.get("extracted_text") or ""
        if text:
            summary_parts.append(f"{label}: {text}" if label else str(text))
    status = "success" if successful else "no_valid_result"
    if successful:
        result_text = " | ".join(summary_parts) if summary_parts else f"{tool_name} returned {len(successful)} usable {input_kind}."
    else:
        result_text = f"{tool_name} did not provide usable evidence across the requested {input_kind}."
        if summary_parts:
            result_text += " Diagnostics: " + " | ".join(summary_parts)
    return enrich_tool_result(
        {
            input_kind: probes,
            "probe_count": len(probes),
            "successful_probes": len(successful),
            "failed_probes": max(0, len(probes) - len(successful)),
            "status": status,
            "result": result_text,
        },
        tool_name=tool_name,
    )


def _build_visual_probe_specs(
    *,
    timestamp: Any = None,
    t_start: Any = None,
    t_end: Any = None,
    density: Any = 0.5,
    duration: Optional[float] = None,
    image_region: Any = None,
    segment_id: Any = None,
    time_windows: Any = None,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    density = _coerce_numeric_scalar_or_list("density", density)

    if segment_id not in (None, "", [], {}) and time_windows not in (None, "", [], {}):
        segment_row = _resolve_segment_row_reference(segment_id, rows or [])
        segment_range = (
            segment_row.get("time_parsed")
            if isinstance(segment_row, dict) and isinstance(segment_row.get("time_parsed"), dict)
            else None
        )
        segment_start = float(segment_range["start"]) if isinstance(segment_range, dict) and "start" in segment_range else 0.0
        segment_end = float(segment_range["end"]) if isinstance(segment_range, dict) and "end" in segment_range else segment_start
        segment_duration = max(0.0, segment_end - segment_start)
        for item in _coerce_list(time_windows):
            relative_time = _parse_time_seconds(item)
            if relative_time is None:
                continue
            absolute_time = float(relative_time)
            if segment_range is not None and 0.0 <= float(relative_time) <= segment_duration + 1e-6:
                absolute_time = segment_start + float(relative_time)
            absolute_time = round(float(absolute_time), 3)
            specs.append(
                {
                    "mode": "single_frame",
                    "target": {"start": absolute_time, "end": absolute_time},
                    "density": None,
                    "probe_label": f"{segment_id}@{absolute_time}",
                    "timestamp": absolute_time,
                    "t_start": None,
                    "t_end": None,
                    "requested_segment_id": str(segment_id),
                    "resolved_segment_id": str(segment_row.get('segment_id', '')) if isinstance(segment_row, dict) else str(segment_id),
                    "resolved_segment_ref": str(segment_row.get('segment_ref', '')) if isinstance(segment_row, dict) else "",
                }
            )
        if specs:
            return specs

    timestamp_segment_row: Optional[Dict[str, Any]] = None
    timestamps = _coerce_list(timestamp)
    if segment_id not in (None, "", [], {}) and timestamps:
        timestamp_segment_row = _resolve_segment_row_reference(segment_id, rows or [])
        segment_range = (
            timestamp_segment_row.get("time_parsed")
            if isinstance(timestamp_segment_row, dict) and isinstance(timestamp_segment_row.get("time_parsed"), dict)
            else None
        )
        if isinstance(segment_range, dict) and "start" in segment_range and "end" in segment_range:
            segment_start = float(segment_range["start"])
            segment_duration = max(0.0, float(segment_range["end"]) - segment_start)
            normalized_timestamps: List[Any] = []
            for item in timestamps:
                parsed = _parse_time_seconds(item)
                if parsed is None:
                    normalized_timestamps.append(item)
                    continue
                absolute_value = float(parsed)
                if 0.0 <= float(parsed) <= segment_duration + 1e-6:
                    absolute_value = segment_start + float(parsed)
                normalized_timestamps.append(round(float(absolute_value), 3))
            timestamps = normalized_timestamps
    for item in timestamps:
        center = _parse_time_seconds(item)
        if center is None and rows:
            resolved = _resolve_reference_target(item, rows)
            segment_row = _resolve_segment_row_reference(item, rows)
            if resolved is not None:
                target = resolved["target"]
                specs.append(
                    {
                        "mode": "range",
                        "target": target,
                        "density": 0.5,
                        "probe_label": resolved["probe_label"],
                        "timestamp": None,
                        "t_start": target["start"],
                        "t_end": target["end"],
                        "requested_segment_id": str(item),
                        "resolved_segment_id": str(segment_row.get('segment_id', '')) if isinstance(segment_row, dict) else str(item),
                        "resolved_segment_ref": str(segment_row.get('segment_ref', '')) if isinstance(segment_row, dict) else "",
                    }
                )
                continue
        if center is None:
            continue
        if duration is not None and float(duration) > 0:
            half = max(0.0, float(duration)) / 2.0
            target = {"start": max(0.0, center - half), "end": center + half}
            mode = "legacy_window"
        else:
            target = {"start": max(0.0, center), "end": max(0.0, center)}
            mode = "single_frame"
        specs.append(
            {
                "mode": mode,
                "target": target,
                "density": None,
                "probe_label": str(item),
                "timestamp": item,
                "t_start": None,
                "t_end": None,
                "requested_segment_id": str(segment_id) if segment_id not in (None, "", [], {}) else None,
                "resolved_segment_id": str(timestamp_segment_row.get('segment_id', '')) if isinstance(timestamp_segment_row, dict) else None,
                "resolved_segment_ref": str(timestamp_segment_row.get('segment_ref', '')) if isinstance(timestamp_segment_row, dict) else None,
            }
        )

    starts = _coerce_list(t_start)
    ends = _coerce_list(t_end)
    range_segment_row: Optional[Dict[str, Any]] = None
    if segment_id not in (None, "", [], {}) and (starts or ends):
        range_segment_row = _resolve_segment_row_reference(segment_id, rows or [])
        segment_range = (
            range_segment_row.get("time_parsed")
            if isinstance(range_segment_row, dict) and isinstance(range_segment_row.get("time_parsed"), dict)
            else None
        )
        if isinstance(segment_range, dict) and "start" in segment_range and "end" in segment_range:
            segment_start = float(segment_range["start"])
            segment_duration = max(0.0, float(segment_range["end"]) - segment_start)

            def _normalize_segment_relative_range_value(value: Any) -> Any:
                parsed = _parse_time_seconds(value)
                if parsed is None:
                    return value
                absolute_value = float(parsed)
                if 0.0 <= float(parsed) <= segment_duration + 1e-6:
                    absolute_value = segment_start + float(parsed)
                return round(float(absolute_value), 3)

            starts = [_normalize_segment_relative_range_value(item) for item in starts]
            ends = [_normalize_segment_relative_range_value(item) for item in ends]
    if starts or ends:
        if len(starts) != len(ends):
            raise ValueError("t_start and t_end must have the same number of items.")
        densities = _coerce_broadcast_list(density, len(starts), default=0.5)
        for start_value, end_value, density_value in zip(starts, ends, densities):
            start = _parse_time_seconds(start_value)
            end = _parse_time_seconds(end_value)
            if start is None or end is None:
                continue
            specs.append(
                {
                    "mode": "range",
                    "target": {"start": min(start, end), "end": max(start, end)},
                    "density": max(float(density_value or 0.5), 1e-3),
                    "probe_label": f"{start_value}-{end_value}",
                    "timestamp": None,
                    "t_start": start_value,
                    "t_end": end_value,
                    "requested_segment_id": str(segment_id) if segment_id not in (None, "", [], {}) else None,
                    "resolved_segment_id": str(range_segment_row.get('segment_id', '')) if isinstance(range_segment_row, dict) else None,
                    "resolved_segment_ref": str(range_segment_row.get('segment_ref', '')) if isinstance(range_segment_row, dict) else None,
                }
            )

    if not specs and image_region is not None:
        entries = _coerce_list(image_region)
        densities = _coerce_broadcast_list(density, len(entries), default=0.5)
        for entry, density_value in zip(entries, densities):
            parsed_range = _parse_range(entry)
            target = _normalize_range_from_value(entry, default_duration=duration or 0.0)
            if target is None and rows:
                resolved = _resolve_reference_target(entry, rows)
                if resolved is not None:
                    target = resolved["target"]
                    parsed_range = target
            if target is None:
                continue
            segment_row = _resolve_segment_row_reference(entry, rows or [])
            specs.append(
                {
                    "mode": "range" if parsed_range else "single_frame",
                    "target": target,
                    "density": max(float(density_value or 0.5), 1e-3) if parsed_range else None,
                    "probe_label": str(entry),
                    "timestamp": entry if not parsed_range else None,
                    "t_start": None,
                    "t_end": None,
                    "requested_segment_id": str(entry) if segment_row is not None else None,
                    "resolved_segment_id": str(segment_row.get('segment_id', '')) if isinstance(segment_row, dict) else None,
                    "resolved_segment_ref": str(segment_row.get('segment_ref', '')) if isinstance(segment_row, dict) else None,
                }
            )

    return specs


def _run_text_match_tool_batch(
    *,
    tool_name: str,
    label_key: str,
    label_value: Any,
    rows: List[Dict[str, Any]],
    target_parser,
    row_builder,
) -> Any:
    labels = _coerce_list(label_value)
    if len(labels) <= 1:
        return row_builder(labels[0] if labels else label_value)

    probes = []
    for item in labels:
        probe_result = row_builder(item)
        if isinstance(probe_result, dict):
            probe_result.setdefault("probe_label", str(item))
        probes.append(probe_result)
    return _build_batch_result(tool_name, probes, input_kind="probes")


def _range_distance(candidate: Optional[Dict[str, float]], target: Optional[Dict[str, float]]) -> float:
    if not target:
        return 0.0
    if not candidate:
        return float("inf")
    if _overlap(candidate, target):
        overlap_start = max(candidate["start"], target["start"])
        overlap_end = min(candidate["end"], target["end"])
        return -max(0.0, overlap_end - overlap_start)
    if candidate["end"] < target["start"]:
        return target["start"] - candidate["end"]
    return candidate["start"] - target["end"]


def _build_uniform_sample_times(start: float, end: float, sample_count: int) -> List[float]:
    start = max(0.0, float(start))
    end = max(start, float(end))
    sample_count = max(1, int(sample_count))
    if sample_count == 1 or end <= start:
        return [round((start + end) / 2, 3)]

    step = (end - start) / sample_count
    if step <= 0:
        return [round(start, 3)] * sample_count
    return [round(start + step * (idx + 0.5), 3) for idx in range(sample_count)]


def _safe_frame_upper_bound(duration: float, fps: float) -> float:
    safe_duration = max(0.0, float(duration or 0.0))
    safe_fps = max(0.0, float(fps or 0.0))
    frame_interval = (1.0 / safe_fps) if safe_fps > 1e-6 else 1e-2
    end_margin = max(1e-3, frame_interval)
    return max(0.0, safe_duration - end_margin)


def _merge_warning_messages(*messages: Any) -> Optional[str]:
    parts: List[str] = []
    for raw in messages:
        text = " ".join(str(raw or "").split()).strip()
        if not text or text in parts:
            continue
        parts.append(text)
    return "; ".join(parts) if parts else None


def _inspect_video_timing(video_path: str) -> Dict[str, float]:
    from moviepy.video.io.VideoFileClip import VideoFileClip

    resolved_video_path = Path(video_path).expanduser().resolve()
    cache_key = str(resolved_video_path)
    cached = _VIDEO_TIMING_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    if not resolved_video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {resolved_video_path}")

    with VideoFileClip(str(resolved_video_path)) as video:
        duration = float(getattr(video, "duration", 0.0) or 0.0)
        fps = float(getattr(video, "fps", 0.0) or 0.0)
    if duration <= 0:
        raise RuntimeError(f"Video has invalid duration: {duration}")

    timing = {
        "duration": duration,
        "fps": fps,
        "max_time": _safe_frame_upper_bound(duration, fps),
    }
    _VIDEO_TIMING_CACHE[cache_key] = timing
    return dict(timing)


def _normalize_requested_video_range(
    *,
    video_path: str,
    start: float,
    end: float,
    tool_name: str,
) -> Dict[str, Any]:
    timing = _inspect_video_timing(video_path)
    duration = float(timing["duration"])
    max_time = float(timing["max_time"])
    requested_start = float(start)
    requested_end = float(end)
    effective_start = max(0.0, requested_start)
    effective_end = max(effective_start, requested_end)

    if effective_start > duration + 1e-6:
        raise RequestedVideoTimeRangeError(
            f"{tool_name} received start time {effective_start:.3f}s beyond video length {duration:.3f}s. "
            "Provide a timestamp/window within the video."
        )

    warning: Optional[str] = None
    if effective_end > duration + 1e-6:
        effective_end = duration
        warning = (
            f"{tool_name} received end time {requested_end:.3f}s beyond video length {duration:.3f}s; "
            f"aligned end time to the video end at {duration:.3f}s."
        )

    safe_start = min(effective_start, max_time)
    safe_end = min(max(effective_end, safe_start), max_time)
    return {
        "requested_start": round(requested_start, 3),
        "requested_end": round(requested_end, 3),
        "effective_start": round(effective_start, 3),
        "effective_end": round(effective_end, 3),
        "safe_start": round(safe_start, 3),
        "safe_end": round(safe_end, 3),
        "duration": round(duration, 3),
        "warning": warning,
    }


def _normalize_range_from_value(value: Any, default_duration: float = 1.0) -> Optional[Dict[str, float]]:
    parsed = _parse_range(value)
    if parsed is not None:
        return parsed
    center = _parse_time_seconds(value)
    if center is None:
        return None
    half = max(0.0, float(default_duration)) / 2.0
    return {
        "start": max(0.0, center - half),
        "end": center + half,
    }


def _select_relevant_rows(
    rows: List[Dict[str, Any]],
    target_range: Optional[Dict[str, float]] = None,
    query: Optional[str] = None,
    video_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    hint = str(video_hint or "").strip()
    filtered = rows
    if hint and not hint.startswith("$"):
        filtered = [
            row for row in rows
            if row.get("video") == hint
            or str(row.get("video_path") or "") == hint
            or Path(str(row.get("video_path") or "")).name == Path(hint).name
        ]

    if target_range is not None:
        overlapping = [row for row in filtered if _overlap(row.get("time_parsed"), target_range)]
        if overlapping:
            filtered = overlapping
        else:
            filtered = sorted(filtered, key=lambda row: _range_distance(row.get("time_parsed"), target_range))[:5]

    query_text = (query or "").strip()
    filtered.sort(
        key=lambda row: (
            _match_score(row.get("text", ""), query_text),
            len(_tokenize(row.get("ocr", "") or row.get("caption", "") or row.get("transcript", ""))),
            -_range_distance(row.get("time_parsed"), target_range),
        ),
        reverse=True,
    )
    return filtered


def _summarize_rows(rows: List[Dict[str, Any]], mode: str = "inspect", max_items: int = 3) -> str:
    snippets: List[str] = []
    for row in rows[:max_items]:
        if mode == "ocr":
            snippet = row.get("ocr", "") or row.get("caption", "")
        else:
            snippet = row.get("caption", "") or row.get("transcript", "")
        snippet = str(snippet or "").strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
    if snippets:
        return " ".join(snippets)
    return "No processed segment evidence matched this window."


def _merge_transcript_context(rows: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    chunks: List[str] = []
    for row in rows:
        transcript = str(row.get("transcript", "")).strip()
        if not transcript:
            continue
        chunks.append(transcript)
    return "\n".join(chunks)


def _is_placeholder_visual_text(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    return normalized.startswith("visual caption unavailable")


def _get_visual_processor(model_path: str, model_name: Optional[str] = None) -> Any:
    raw_key = str(model_path or model_name or settings.qwen_vl_model_path)
    try:
        cache_key = str(Path(raw_key).expanduser().resolve())
    except Exception:
        cache_key = raw_key
    processor = _VISUAL_PROCESSORS.get(cache_key)
    if processor is not None:
        return processor

    from preprocessing.video.caption import create_visual_processor

    resolved_model_name = model_name or cache_key
    processor = create_visual_processor(model_name=resolved_model_name, model_path=cache_key)
    _VISUAL_PROCESSORS[cache_key] = processor
    return processor


def _get_cached_visual_generation_provider(
    *,
    model_path: str,
    model_role: str = "qwen_vl",
):
    from core.llm.provider import get_llm_provider

    cache_key = json.dumps(
        {
            "provider": "llm_provider",
            "model_name": str(model_path),
            "model_role": str(model_role),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cached = runtime_session.get_cached_provider(cache_key)
    if cached is not None:
        runtime_session.record_cache(f"provider.{model_role}", True)
        return cached

    runtime_session.record_cache(f"provider.{model_role}", False)
    started = time.perf_counter()
    provider = get_llm_provider(
        model_name=model_path,
        model_role=model_role,
    )
    runtime_session.record_timing(
        f"runtime_load.provider_{model_role}_seconds",
        time.perf_counter() - started,
    )
    return runtime_session.cache_provider(cache_key, provider)


def _safe_resolved_path_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def _same_model_path(left: Any, right: Any) -> bool:
    return _safe_resolved_path_text(left) == _safe_resolved_path_text(right)


def _video_clip_qa_provider_role(model_path: str) -> str:
    # When the visual checkpoint is the same one used by planner, use the
    # planner TransformersProvider path. That path already knows how to load
    # Qwen3.5/Qwen3-VL multimodal checkpoints in this project.
    if _same_model_path(model_path, settings.planner_model_path):
        return "planner"
    if _same_model_path(model_path, getattr(settings, "subagent_model_path", "")):
        return "subagent"
    return "qwen_vl"


def _video_clip_qa_reuses_loaded_llm_checkpoint(model_path: str) -> bool:
    return _same_model_path(model_path, settings.planner_model_path) or _same_model_path(
        model_path,
        getattr(settings, "subagent_model_path", ""),
    )


def _sample_video_clip_frames_for_provider(
    clip_path: str,
    *,
    fps: float,
    max_frames: int,
) -> tuple[List[Image.Image], List[float], float]:
    from moviepy.video.io.VideoFileClip import VideoFileClip

    resolved_clip_path = Path(clip_path).expanduser().resolve()
    frames: List[Image.Image] = []
    sample_times: List[float] = []
    source_fps = 0.0
    with VideoFileClip(str(resolved_clip_path)) as video:
        duration = max(0.0, float(getattr(video, "duration", 0.0) or 0.0))
        source_fps = max(0.0, float(getattr(video, "fps", 0.0) or 0.0))
        if duration <= 0:
            raise RuntimeError(f"Video clip has invalid duration: {duration}")
        requested_fps = max(0.1, float(fps or 1.0))
        requested_frames = max(1, int(math.ceil(duration * requested_fps)))
        sample_count = max(1, min(int(max_frames), requested_frames))
        max_time = _safe_frame_upper_bound(duration, source_fps or requested_fps)
        if sample_count == 1:
            sample_times = [round(min(max_time, duration / 2.0), 3)]
        else:
            sample_times = [
                round(min(max_time, max_time * idx / float(sample_count - 1)), 3)
                for idx in range(sample_count)
            ]
        for sample_time in sample_times:
            frame = video.get_frame(max(0.0, min(float(sample_time), max_time)))
            frames.append(Image.fromarray(frame).convert("RGB"))
    if not frames:
        raise RuntimeError("No frames were sampled from the video clip.")
    return frames, sample_times, source_fps


def _get_video_clip_qa_provider(model_path: str) -> tuple[Any, str, bool]:
    from core.llm.provider import build_role_specific_provider_view, get_llm_provider

    provider_role = _video_clip_qa_provider_role(model_path)
    provider = get_llm_provider(
        model_name=model_path,
        model_role=provider_role,
    )
    reuses_loaded_llm_checkpoint = provider_role in {"planner", "subagent"}
    if reuses_loaded_llm_checkpoint:
        try:
            provider = build_role_specific_provider_view(
                provider,
                "qwen_vl",
                enable_thinking=False,
            )
        except Exception as exc:
            logger.debug("Failed to build qwen_vl role view over shared planner provider: %s", exc)
    return provider, provider_role, reuses_loaded_llm_checkpoint


def _find_loaded_shared_video_clip_qa_provider(model_path: str) -> Any:
    if not _video_clip_qa_reuses_loaded_llm_checkpoint(model_path):
        return None

    normalized_model_path = _safe_resolved_path_text(model_path)
    provider_cache = getattr(runtime_session, "_provider_cache", {}) or {}
    for provider in provider_cache.values():
        if not _same_model_path(getattr(provider, "model_name", None), normalized_model_path):
            continue
        provider_roles = {
            str(getattr(provider, "model_role", "") or "").strip().lower()
        }
        shared_roles = getattr(provider, "_shared_model_roles", None)
        if isinstance(shared_roles, (list, tuple, set, frozenset)):
            provider_roles.update(
                str(role or "").strip().lower()
                for role in shared_roles
                if str(role or "").strip()
            )
        if not (provider_roles & {"planner", "subagent"}):
            continue
        if getattr(provider, "model", None) is None or getattr(provider, "processor", None) is None:
            continue
        return provider
    return None


def _get_shared_video_clip_qa_visual_processor(model_path: str) -> Any:
    if not _video_clip_qa_reuses_loaded_llm_checkpoint(model_path):
        return None

    from preprocessing.video.caption import VisualProcessor
    from core.llm.provider import build_role_specific_provider_view

    provider = _find_loaded_shared_video_clip_qa_provider(model_path)
    if provider is None:
        return None
    provider_role = str(getattr(provider, "model_role", "") or "").strip().lower()
    if not bool(getattr(provider, "_supports_multimodal_generation", False)):
        logger.warning(
            "Shared Video_Clip_QA runtime for role=%s does not support multimodal generation; cannot build direct-video fallback view.",
            provider_role,
        )
        return None
    try:
        provider = build_role_specific_provider_view(
            provider,
            "qwen_vl",
            enable_thinking=False,
        )
        return VisualProcessor.from_loaded_llm_provider(
            provider,
            model_name=model_path,
            model_path=model_path,
        )
    except Exception as exc:
        logger.warning(
            "Failed to build shared direct-video VisualProcessor view for %s: %s",
            model_path,
            exc,
        )
        return None


def _answer_video_clip_qa_with_provider(
    *,
    clip_path: str,
    start: float,
    end: float,
    query: str,
    model_path: str,
    max_new_tokens: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> str:
    provider, provider_role, _ = _get_video_clip_qa_provider(model_path)
    if not bool(getattr(provider, "_supports_multimodal_generation", False)):
        raise RuntimeError(
            f"Provider role={provider_role} does not support multimodal generation for {model_path}."
        )
    if getattr(provider, "processor", None) is None:
        raise RuntimeError(
            f"Provider role={provider_role} has no multimodal processor for {model_path}."
        )

    official_sampling_fps = max(
        0.1,
        float(getattr(settings, "video_clip_qa_fps", 2.0) or 2.0),
    )
    official_max_frames = max(
        1,
        int(getattr(settings, "video_clip_qa_max_frames", 80) or 80),
    )
    video_item: Dict[str, Any] = {
        "type": "video",
        "video": str(Path(clip_path).expanduser().resolve()),
        "fps": official_sampling_fps,
        "max_frames": official_max_frames,
        "do_sample_frames": True,
    }
    official_min_pixels = max(
        32 * 32,
        int(getattr(settings, "video_clip_qa_min_pixels", 4 * 32 * 32) or (4 * 32 * 32)),
    )
    if min_pixels is not None:
        video_item["min_pixels"] = max(official_min_pixels, int(min_pixels))
    if max_pixels is not None:
        floor_pixels = (
            max(official_min_pixels, int(min_pixels))
            if min_pixels is not None
            else official_min_pixels
        )
        video_item["max_pixels"] = max(floor_pixels, int(max_pixels))

    prompt_text = build_video_clip_user_message(query)
    answer = provider.generate(
        [
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=int(max_new_tokens),
        do_sample=False,
        enable_thinking=False,
    )
    raw_answer = str(answer or "").strip()
    if raw_answer:
        logger.info("[Video_Clip_QA][RAW_RESPONSE_BEGIN][backend=planner_provider_video]\n%s\n[Video_Clip_QA][RAW_RESPONSE_END]", raw_answer)
    return raw_answer


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _resolve_video_clip_qa_processor_pixel_bounds(model_path: str) -> tuple[int, int]:
    configured_min_pixels = max(
        32 * 32,
        int(getattr(settings, "video_clip_qa_min_pixels", 4 * 32 * 32) or (4 * 32 * 32)),
    )
    configured_video_max_pixels = int(
        getattr(
            settings,
            "video_clip_qa_processor_max_pixels",
            65536 * 32 * 32,
        )
        or (65536 * 32 * 32)
    )
    configured_max_pixels = max(
        configured_min_pixels,
        configured_video_max_pixels,
    )

    try:
        provider, _, _ = _get_video_clip_qa_provider(model_path)
    except Exception:
        return configured_min_pixels, configured_max_pixels

    processor = getattr(provider, "processor", None)
    candidates = [
        processor,
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    ]

    discovered_min_pixels = configured_min_pixels
    discovered_max_pixels = configured_max_pixels
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_min = _coerce_positive_int(getattr(candidate, "min_pixels", None))
        candidate_max = _coerce_positive_int(getattr(candidate, "max_pixels", None))
        if candidate_min is not None:
            discovered_min_pixels = max(discovered_min_pixels, candidate_min)
        if candidate_max is not None:
            discovered_max_pixels = min(discovered_max_pixels, candidate_max)

    discovered_max_pixels = max(discovered_min_pixels, discovered_max_pixels)
    return discovered_min_pixels, discovered_max_pixels


def _build_video_clip_qa_sampling_plans(*, min_pixels: int, max_pixels: int) -> List[Dict[str, Any]]:
    default_max_new_tokens = int(getattr(settings, "video_clip_qa_max_new_tokens", 2048) or 2048)
    configured_max_pixels = max(
        int(min_pixels),
        min(
            int(max_pixels),
            int(getattr(settings, "video_clip_qa_max_pixels", 768 * 32 * 32) or (768 * 32 * 32)),
        ),
    )
    minimal_max_pixels = max(
        int(min_pixels),
        min(int(max_pixels), min(configured_max_pixels, 256 * 32 * 32)),
    )

    candidates = [
        {
            "label": "default_request",
            "max_new_tokens": default_max_new_tokens,
            "min_pixels": None,
            "max_pixels": None,
        },
        {
            "label": "compact_output",
            "max_new_tokens": min(default_max_new_tokens, 1024),
            "min_pixels": int(min_pixels),
            "max_pixels": configured_max_pixels,
        },
        {
            "label": "minimal_output",
            "max_new_tokens": min(default_max_new_tokens, 768),
            "min_pixels": int(min_pixels),
            "max_pixels": minimal_max_pixels,
        },
    ]

    plans: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        signature = (
            int(candidate["max_new_tokens"]),
            int(candidate["min_pixels"]) if candidate["min_pixels"] is not None else None,
            int(candidate["max_pixels"]) if candidate["max_pixels"] is not None else None,
        )
        if signature in seen:
            continue
        seen.add(signature)
        plans.append(candidate)
    return plans


def _finalize_video_clip_qa_answer(raw_answer: Any, *, backend: str) -> Dict[str, Any]:
    normalized = normalize_video_clip_qa_answer(raw_answer)
    result: Dict[str, Any] = {
        "answer": normalized["planner_text"],
        "backend": backend,
        "parsed_json": normalized["parsed_json"],
    }
    parsed_json = normalized.get("parsed_json")
    if isinstance(parsed_json, dict):
        if "answer" in parsed_json:
            result["answer_value"] = parsed_json["answer"]
        if "evidence" in parsed_json:
            result["evidence_items"] = parsed_json["evidence"]
        if "uncertainty" in parsed_json:
            result["uncertainty_value"] = parsed_json["uncertainty"]
        if "video_summary" in parsed_json:
            result["video_summary_value"] = parsed_json["video_summary"]
    warning_text = str(normalized.get("warning", "") or "").strip()
    if warning_text:
        result["warning"] = warning_text
    return result


def _persist_probe_video_clip(
    *,
    video_path: str,
    start: float,
    end: float,
    tool_name: str,
) -> str:
    from moviepy.video.io.VideoFileClip import VideoFileClip

    resolved_video_path = Path(video_path).expanduser().resolve()
    if not resolved_video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {resolved_video_path}")

    with VideoFileClip(str(resolved_video_path)) as video:
        clip_duration = float(getattr(video, "duration", 0.0) or 0.0)
        if clip_duration <= 0:
            raise RuntimeError(f"Video has invalid duration: {clip_duration}")
        safe_start = max(0.0, min(float(start), max(0.0, clip_duration - 1e-3)))
        safe_end = max(safe_start + 1e-3, min(float(end), clip_duration))
        if safe_end <= safe_start:
            raise ValueError(f"Invalid clip window: {safe_start}-{safe_end}")
        if safe_start <= 0.05 and safe_end >= clip_duration - 0.05:
            return str(resolved_video_path)

        signature = hashlib.sha1(
            f"v4|{tool_name}|{resolved_video_path}|{safe_start:.3f}|{safe_end:.3f}".encode("utf-8")
        ).hexdigest()[:12]
        output_dir = _resolve_visual_observation_root() / resolved_video_path.stem / f"{tool_name}_{signature}"
        output_dir.mkdir(parents=True, exist_ok=True)
        clip_path = output_dir / f"{resolved_video_path.stem}_{tool_name}.mp4"
        if clip_path.exists():
            return str(clip_path.resolve())
        export_video_subclip(
            video_path=str(resolved_video_path),
            start=safe_start,
            end=safe_end,
            output_path=clip_path,
        )

    return str(clip_path.resolve())

def _run_video_clip_qa(
    *,
    video_path: str,
    start: float,
    end: float,
    query: str,
    model_path: str,
) -> Dict[str, Any]:
    clip_path = _persist_probe_video_clip(
        video_path=video_path,
        start=start,
        end=end,
        tool_name="video_clip_qa",
    )
    with runtime_session.time_block("tool.video_clip_qa_seconds"):
        raw_answer = ""
        last_error_text = ""
        plan_warning = ""
        official_sampling_fps = max(
            0.1,
            float(getattr(settings, "video_clip_qa_fps", 2.0) or 2.0),
        )
        official_max_frames = max(
            1,
            int(getattr(settings, "video_clip_qa_max_frames", 80) or 80),
        )
        effective_min_pixels, effective_max_pixels = _resolve_video_clip_qa_processor_pixel_bounds(model_path)
        for plan in _build_video_clip_qa_sampling_plans(
            min_pixels=effective_min_pixels,
            max_pixels=effective_max_pixels,
        ):
            try:
                logger.info(
                    "Video_Clip_QA direct clip attempt=%s model_sampling_fps=%s model_max_frames=%s model_min_pixels=%s model_max_pixels=%s do_sample_frames=True max_new_tokens=%s clip=%s",
                    plan["label"],
                    official_sampling_fps,
                    official_max_frames,
                    plan["min_pixels"] if plan["min_pixels"] is not None else "source",
                    plan["max_pixels"] if plan["max_pixels"] is not None else "source",
                    plan["max_new_tokens"],
                    clip_path,
                )
                raw_answer = _answer_video_clip_qa_with_provider(
                    clip_path=clip_path,
                    start=start,
                    end=end,
                    query=query,
                    model_path=model_path,
                    max_new_tokens=int(plan["max_new_tokens"]),
                    min_pixels=int(plan["min_pixels"]) if plan["min_pixels"] is not None else None,
                    max_pixels=int(plan["max_pixels"]) if plan["max_pixels"] is not None else None,
                )
            except Exception as exc:
                last_error_text = _format_exception_message(exc)
                logger.warning(
                    "Video_Clip_QA direct clip execution failed for %s on attempt=%s: %s",
                    clip_path,
                    plan["label"],
                    last_error_text,
                )
                _release_visual_runtime_memory()
                continue

            raw_answer_text = str(raw_answer or "").strip()
            if raw_answer_text:
                if plan["label"] != "default_request":
                    plan_warning = (
                        f"Video_Clip_QA succeeded only after fallback sampling controls ({plan['label']}: "
                        f"max_new_tokens={plan['max_new_tokens']}, "
                        f"min_pixels={plan['min_pixels'] if plan['min_pixels'] is not None else 'source'}, "
                        f"max_pixels={plan['max_pixels'] if plan['max_pixels'] is not None else 'source'})."
                    )
                break

            last_error_text = f"Video_Clip_QA returned an empty answer on attempt={plan['label']}."
            logger.warning("%s clip=%s", last_error_text, clip_path)
            _release_visual_runtime_memory()

        if not str(raw_answer or "").strip():
            return {
                "clip_path": clip_path,
                "answer": "",
                "error": last_error_text or "Video_Clip_QA failed to produce a response for the clip.",
                "backend": "planner_provider_video",
            }

        raw_answer_text = str(raw_answer or "").strip()

        result = _finalize_video_clip_qa_answer(
            raw_answer_text,
            backend="planner_provider_video",
        )
        result["clip_path"] = clip_path
        if plan_warning:
            existing_warning = str(result.get("warning", "") or "").strip()
            result["warning"] = (
                f"{existing_warning} {plan_warning}".strip()
                if existing_warning
                else plan_warning
            )
        if looks_like_truncated_video_clip_qa_answer(raw_answer_text):
            existing_warning = str(result.get("warning", "") or "").strip()
            truncated_warning = "Video_Clip_QA response appears truncated."
            result["warning"] = (
                f"{existing_warning} {truncated_warning}".strip()
                if existing_warning
                else truncated_warning
            )
        _release_visual_runtime_memory()
        return result


def _analyze_video_window(
    video_path: str,
    start: float,
    end: float,
    transcript: str,
    query: Optional[str],
    mode: str = "inspect",
    sample_count: Optional[int] = None,
    sample_density: Optional[float] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    del transcript, query, model_path  # raw-frame tools now return visual evidence directly.
    if not video_path:
        raise FileNotFoundError("Video path is empty.")

    from moviepy.video.io.VideoFileClip import VideoFileClip

    resolved_video_path = Path(video_path).expanduser().resolve()
    normalized_range = _normalize_requested_video_range(
        video_path=str(resolved_video_path),
        start=float(start),
        end=float(end),
        tool_name=str(mode or "visual_probe"),
    )

    with VideoFileClip(str(resolved_video_path)) as video:
        clip_duration = float(getattr(video, "duration", 0.0) or 0.0)
        clip_fps = float(getattr(video, "fps", 0.0) or 0.0)
        if clip_duration <= 0:
            raise RuntimeError(f"Video has invalid duration: {clip_duration}")

        max_time = _safe_frame_upper_bound(clip_duration, clip_fps)
        clip_start = max(0.0, min(float(normalized_range["safe_start"]), max_time))
        clip_end = max(clip_start, min(float(normalized_range["safe_end"]), max_time))
        window_duration = max(0.0, clip_end - clip_start)
        density_sample_count = _sample_count_from_density(window_duration, sample_density)
        effective_sample_count = (
            density_sample_count
            if density_sample_count is not None
            else (
                max(1, int(sample_count))
                if sample_count is not None
                else (min(24, max(6, int(round(window_duration * 8)))) if window_duration > 0 else 1)
            )
        )
        effective_sample_count = min(effective_sample_count, 24)
        sample_times = _build_uniform_sample_times(clip_start, clip_end, effective_sample_count)

        frames: List[Image.Image] = []
        realized_times: List[float] = []
        for sample_time in sample_times:
            try:
                safe_sample_time = min(max(0.0, float(sample_time)), max_time)
                frame = video.get_frame(safe_sample_time)
                frames.append(Image.fromarray(frame.astype("uint8")).convert("RGB"))
                realized_times.append(round(float(safe_sample_time), 3))
            except Exception as exc:
                logger.warning("Failed to extract probe frame at %s from %s: %s", sample_time, resolved_video_path, exc)

    persisted_frames = _persist_probe_frame_images(
        video_path=str(resolved_video_path),
        start=clip_start,
        end=clip_end,
        sample_times=realized_times,
        frames=frames,
        tool_name=mode or "inspect",
    )
    packed_frames = _pack_probe_frame_images(
        persisted_frames,
        text_hint=f"{mode} {clip_start:.3f}-{clip_end:.3f}",
    )
    return {
        "text": "",
        "sample_times": realized_times,
        "frames_analyzed": len(realized_times),
        "image_evidence": packed_frames,
        "image_count": len(packed_frames),
        "effective_range": {
            "start": float(normalized_range["effective_start"]),
            "end": float(normalized_range["effective_end"]),
        },
        "warning": normalized_range.get("warning"),
    }


def _resolve_visual_observation_root() -> Path:
    cache_root = Path(settings.data_cache_dir).expanduser()
    if not cache_root.is_absolute():
        cache_root = (settings.project_root / cache_root).resolve()
    evidence_root = cache_root / str(settings.planner_visual_observation_subdir or "planner_evidence")
    evidence_root.mkdir(parents=True, exist_ok=True)
    return evidence_root


def _planner_single_image_size() -> int:
    return max(256, int(getattr(settings, "planner_visual_single_image_size", 896) or 896))


def _planner_collage_tile_size() -> int:
    return max(128, int(getattr(settings, "planner_visual_collage_tile_size", 448) or 448))


def _planner_visual_max_pixels() -> int:
    configured = getattr(settings, "planner_visual_max_pixels", None)
    if configured not in (None, ""):
        try:
            return max(128 * 128, int(configured))
        except Exception:
            pass
    fallback = getattr(settings, "visual_service_max_pixels", 336 * 336)
    return max(128 * 128, int(fallback or (336 * 336)))


def get_planner_image_order_rule_text() -> str:
    return PLANNER_IMAGE_ORDER_RULE_TEXT


def _normalize_image_to_canvas(image: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", canvas_size, color=(0, 0, 0))
    fitted = image.convert("RGB")
    fitted.thumbnail(canvas_size, resample=Image.Resampling.LANCZOS)
    x_offset = max(0, (canvas_size[0] - fitted.width) // 2)
    y_offset = max(0, (canvas_size[1] - fitted.height) // 2)
    canvas.paste(fitted, (x_offset, y_offset))
    return canvas


def _save_jpeg_with_lock(image: Image.Image, output_path: Path, *, quality: int = 90) -> Path:
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    from filelock import FileLock

    lock_path = resolved_output_path.with_suffix(resolved_output_path.suffix + ".lock")
    with FileLock(str(lock_path)):
        if resolved_output_path.exists() and _is_readable_image(resolved_output_path):
            return resolved_output_path

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{resolved_output_path.stem}_",
            suffix=resolved_output_path.suffix or ".jpg",
            dir=str(resolved_output_path.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            image.convert("RGB").save(temp_path, format="JPEG", quality=quality)
            if not _is_readable_image(temp_path):
                raise RuntimeError(f"Atomic JPEG write produced unreadable temp file {temp_path}.")
            os.replace(temp_path, resolved_output_path)
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    return resolved_output_path


def _persist_standardized_single_image(
    image_path: str,
    *,
    timestamp: Optional[float] = None,
    source_images: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    raw_path = str(image_path or "").strip()
    if not raw_path:
        return None
    resolved_image_path = Path(raw_path).expanduser().resolve()
    if not resolved_image_path.exists():
        return None

    output_path = resolved_image_path.parent / f"{resolved_image_path.stem}_planner_single.jpg"
    should_regenerate = True
    if output_path.exists():
        should_regenerate = not _is_readable_image(output_path)
        if should_regenerate:
            logger.warning(
                "Regenerating unreadable planner image %s from source %s.",
                output_path,
                resolved_image_path,
            )

    if should_regenerate:
        try:
            source_image = _load_rgb_image(str(resolved_image_path), context="planner source")
        except Exception as exc:
            logger.warning("%s", exc)
            return None
        normalized = _normalize_image_to_canvas(
            source_image,
            (_planner_single_image_size(), _planner_single_image_size()),
        )
        try:
            output_path = _save_jpeg_with_lock(normalized, output_path, quality=90)
        except Exception as exc:
            logger.warning(
                "Failed to persist standardized planner image %s: %s",
                output_path,
                exc,
            )
            return None

    normalized_sources = [
        str(Path(item).expanduser().resolve())
        for item in (source_images or [str(resolved_image_path)])
        if str(item).strip()
    ]
    normalized_timestamp = round(float(timestamp), 3) if timestamp is not None else None
    return {
        "image": str(output_path.resolve()),
        "timestamp": normalized_timestamp,
        "sample_times": [normalized_timestamp] if normalized_timestamp is not None else [],
        "source_images": normalized_sources or [str(resolved_image_path)],
        "pack_mode": "single",
        "max_pixels": _planner_visual_max_pixels(),
        "image_order_rule": get_planner_image_order_rule_text(),
    }


def _clockwise_collage_slots(count: int) -> List[tuple[int, int]]:
    base_slots = [(0, 0), (1, 0), (1, 1), (0, 1)]
    count = max(0, min(int(count), len(base_slots)))
    return base_slots[:count]


def _persist_probe_frame_images(
    *,
    video_path: str,
    start: float,
    end: float,
    sample_times: List[float],
    frames: List[Image.Image],
    tool_name: str,
) -> List[Dict[str, Any]]:
    if not frames:
        return []

    video_name = Path(video_path).stem or "video"
    signature = hashlib.sha1(
        f"{tool_name}|{video_path}|{start:.3f}|{end:.3f}|{','.join(f'{item:.3f}' for item in sample_times)}".encode("utf-8")
    ).hexdigest()[:12]
    probe_dir = _resolve_visual_observation_root() / video_name / f"{tool_name}_{signature}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    persisted: List[Dict[str, Any]] = []
    for index, (frame, timestamp) in enumerate(zip(frames, sample_times)):
        safe_stamp = str(round(float(timestamp), 3)).replace(".", "p")
        frame_path = probe_dir / f"frame_{index:03d}_{safe_stamp}.jpg"
        if not frame_path.exists():
            frame.convert("RGB").save(frame_path, format="JPEG", quality=90)
        persisted.append(
            {
                "image": str(frame_path.resolve()),
                "timestamp": round(float(timestamp), 3),
                "source_images": [str(frame_path.resolve())],
                "pack_mode": "single",
            }
        )
    return persisted


def _visual_observation_image_limit(text_hint: str = "") -> int:
    token_budget = max(1024, int(getattr(settings, "planner_visual_observation_token_budget", 12288) or 12288))
    image_token_estimate = max(
        256,
        int(getattr(settings, "planner_visual_observation_image_token_estimate", 1536) or 1536),
    )
    text_tokens = max(256, len(" ".join(str(text_hint or "").split())) // 4)
    remaining_budget = max(image_token_estimate, token_budget - text_tokens)
    return max(1, remaining_budget // image_token_estimate)


def _build_probe_image_collage(chunk: List[Dict[str, Any]], collage_path: Path) -> Optional[str]:
    if len(chunk) <= 1:
        return str(chunk[0].get("image", "") or "") if chunk else None

    opened_images: List[Image.Image] = []
    valid_image_paths: List[str] = []
    try:
        for item in chunk:
            image_path = str(item.get("image", "") or "").strip()
            if not image_path:
                continue
            try:
                opened_images.append(_load_rgb_image(image_path, context="probe"))
                valid_image_paths.append(image_path)
            except Exception as exc:
                logger.warning("%s", exc)
        if not opened_images:
            return None
        if len(opened_images) == 1:
            return valid_image_paths[0]

        tile_size = _planner_collage_tile_size()
        canvas = Image.new("RGB", (tile_size * 2, tile_size * 2), color=(0, 0, 0))
        slot_positions = _clockwise_collage_slots(len(opened_images))

        for image, (col, row) in zip(opened_images, slot_positions):
            tile = _normalize_image_to_canvas(image, (tile_size, tile_size))
            canvas.paste(tile, (col * tile_size, row * tile_size))

        saved_path = _save_jpeg_with_lock(canvas, collage_path, quality=90)
        return str(saved_path.resolve())
    except Exception as exc:
        logger.warning("Failed to build probe collage at %s: %s", collage_path, exc)
        return None


def _pack_probe_frame_images(
    image_evidence: List[Dict[str, Any]],
    text_hint: str = "",
) -> List[Dict[str, Any]]:
    if not image_evidence:
        return []

    max_images = _visual_observation_image_limit(text_hint=text_hint)
    group_size = max(2, int(getattr(settings, "planner_visual_observation_pack_group_size", 4) or 4))
    packed: List[Dict[str, Any]] = []
    for chunk_index in range(0, len(image_evidence), group_size):
        raw_chunk = image_evidence[chunk_index:chunk_index + group_size]
        chunk = []
        for item in raw_chunk:
            normalized_item = _recover_packed_image_item(item)
            if normalized_item is not None:
                chunk.append(normalized_item)
        if not chunk:
            logger.warning(
                "Skipping packed planner image chunk %d because every source image was unreadable.",
                chunk_index // group_size,
            )
            continue
        if len(chunk) == 1:
            single = _persist_standardized_single_image(
                str(chunk[0].get("image", "") or "").strip(),
                timestamp=chunk[0].get("timestamp"),
                source_images=list(chunk[0].get("source_images") or []),
            )
            if single is not None:
                packed.append(single)
            continue
        first_image = str(chunk[0].get("image", "") or "").strip()
        if not first_image:
            continue
        collage_path = Path(first_image).parent / f"pack_{chunk_index // group_size:03d}.jpg"
        resolved_collage = _build_probe_image_collage(chunk, collage_path)
        if not resolved_collage:
            continue
        sample_times = [
            round(float(item.get("timestamp")), 3)
            for item in chunk
            if item.get("timestamp") is not None
        ]
        packed.append(
            {
                "image": resolved_collage,
                "timestamp": sample_times[0] if sample_times else None,
                "sample_times": sample_times,
                "source_images": [
                    str(item.get("image", "") or "").strip()
                    for item in chunk
                    if str(item.get("image", "") or "").strip()
                ],
                "pack_mode": f"{len(chunk)}up",
                "max_pixels": _planner_visual_max_pixels(),
                "image_order_rule": get_planner_image_order_rule_text(),
            }
        )

    if len(packed) > max_images:
        step = max(1, math.ceil(len(packed) / max_images))
        packed = packed[::step][:max_images]
    return packed


def _summarize_image_evidence(image_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_items = [
        item for item in (image_evidence or [])
        if isinstance(item, dict)
    ]
    image_count = len(normalized_items)
    source_image_count = 0
    pack_modes: List[str] = []
    contains_packed_collage = False
    image_timestamps: List[float] = []

    for item in normalized_items:
        pack_mode = str(item.get("pack_mode", "") or "").strip()
        if pack_mode and pack_mode not in pack_modes:
            pack_modes.append(pack_mode)
        if pack_mode and pack_mode not in {"single", "cached_single"}:
            contains_packed_collage = True

        sample_times = [
            round(float(value), 3)
            for value in (item.get("sample_times") or [])
            if value is not None
        ]
        if sample_times:
            for value in sample_times:
                if value not in image_timestamps:
                    image_timestamps.append(value)
        elif item.get("timestamp") is not None:
            timestamp = round(float(item.get("timestamp")), 3)
            if timestamp not in image_timestamps:
                image_timestamps.append(timestamp)

        source_images = [
            str(source).strip()
            for source in (item.get("source_images") or [])
            if str(source).strip()
        ]
        if source_images:
            source_image_count += len(source_images)
        elif str(item.get("image", "") or "").strip():
            source_image_count += 1

    if image_count > 0 and source_image_count <= 0:
        source_image_count = image_count

    if image_count <= 0:
        image_pack_summary = "0 image evidence item(s)"
    elif source_image_count > image_count:
        image_pack_summary = (
            f"{image_count} packaged image evidence item(s) "
            f"representing {source_image_count} source frame(s)"
        )
    else:
        image_pack_summary = f"{image_count} image evidence item(s)"

    if contains_packed_collage and pack_modes:
        image_pack_summary += f" via {', '.join(pack_modes)}"

    return {
        "image_count": image_count,
        "source_image_count": source_image_count,
        "image_pack_modes": pack_modes,
        "contains_packed_collage": contains_packed_collage,
        "image_pack_summary": image_pack_summary,
        "image_timestamps": image_timestamps,
        "image_order_rule": get_planner_image_order_rule_text(),
    }


def pack_planner_image_evidence(image_evidence: List[Dict[str, Any]], text_hint: str = "") -> List[Dict[str, Any]]:
    return _pack_probe_frame_images(image_evidence, text_hint=text_hint)


def summarize_planner_image_evidence(image_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _summarize_image_evidence(image_evidence)


def prepare_planner_image_attachment(image_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(image_item, dict):
        return None
    prepared = _recover_packed_image_item(dict(image_item))
    if prepared is None:
        return None
    if prepared.get("preserve_native_pixels"):
        prepared.pop("max_pixels", None)
    elif prepared.get("max_pixels") in (None, "", 0):
        prepared["max_pixels"] = _planner_visual_max_pixels()
    return prepared


def _is_present_result_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _planner_visual_result_text(
    image_timestamps: List[float],
    *,
    fallback_message: str,
) -> str:
    if image_timestamps:
        return "image_timestamps=" + json.dumps(image_timestamps, ensure_ascii=False)
    return fallback_message


def _build_planner_visual_result(
    *,
    image_evidence: List[Dict[str, Any]],
    image_stats: Optional[Dict[str, Any]] = None,
    status: str,
    fallback_message: str,
    extra_fields: Optional[Dict[str, Any]] = None,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    stats = image_stats or _summarize_image_evidence(image_evidence)
    result = {
        "image_evidence": image_evidence,
        "image_count": stats["image_count"],
        "source_image_count": stats["source_image_count"],
        "image_pack_modes": stats["image_pack_modes"],
        "contains_packed_collage": stats["contains_packed_collage"],
        "image_timestamps": stats["image_timestamps"],
        "image_order_rule": stats["image_order_rule"],
        "status": status,
        "result": _planner_visual_result_text(
            stats["image_timestamps"],
            fallback_message=fallback_message,
        ),
    }
    for key, value in (extra_fields or {}).items():
        if _is_present_result_value(value):
            result[key] = value
    if warning:
        result["warning"] = warning
    return result


def resolve_processed_frame_image_path(video_name: str, frame_path: Any) -> Optional[str]:
    return _resolve_cached_frame_path(video_name, frame_path)


def planner_image_attachment_limit(text_hint: str = "") -> int:
    return _visual_observation_image_limit(text_hint=text_hint)


def _resolve_cached_frame_path(video_name: str, frame_path: Any) -> Optional[str]:
    text = str(frame_path or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve()) if candidate.exists() else None

    cache_root = Path(settings.data_cache_dir).expanduser()
    if not cache_root.is_absolute():
        cache_root = (settings.project_root / cache_root).resolve()
    resolved = (cache_root / "processed" / str(video_name or "").strip() / candidate).resolve()
    return str(resolved) if resolved.exists() else None


def _fallback_probe_images_from_rows(
    rows: List[Dict[str, Any]],
    text_hint: str = "",
) -> List[Dict[str, Any]]:
    image_evidence: List[Dict[str, Any]] = []
    for row in rows:
        video_name = str(row.get("video", "")).strip()
        sample_times = list(row.get("frame_times") or [])
        for index, rel_path in enumerate(row.get("frame_paths") or []):
            resolved_path = _resolve_cached_frame_path(video_name, rel_path)
            if not resolved_path:
                continue
            timestamp = sample_times[index] if index < len(sample_times) else None
            image_evidence.append(
                {
                    "image": resolved_path,
                    "timestamp": round(float(timestamp), 3) if timestamp is not None else None,
                    "source_images": [resolved_path],
                    "pack_mode": "cached_single",
                }
            )
    return _pack_probe_frame_images(image_evidence, text_hint=text_hint)


def _merge_row_ranges(rows: List[Dict[str, Any]], limit: int = 4) -> Optional[Dict[str, float]]:
    ranges = [
        row.get("time_parsed")
        for row in rows[:limit]
        if isinstance(row.get("time_parsed"), dict)
    ]
    if not ranges:
        return None
    starts = [float(item["start"]) for item in ranges]
    ends = [float(item["end"]) for item in ranges]
    return {"start": min(starts), "end": max(ends)}


def _sample_indices(length: int, limit: int) -> List[int]:
    if length <= 0 or limit <= 0:
        return []
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [0]
    last_index = length - 1
    step = last_index / max(1, limit - 1)
    return sorted({min(last_index, int(round(index * step))) for index in range(limit)})


def _build_row_frame_hits(row: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    frame_paths = list(row.get("frame_paths") or [])
    frame_times = list(row.get("frame_times") or [])
    for index in _sample_indices(len(frame_paths), limit):
        resolved_path = _resolve_cached_frame_path(row.get("video", ""), frame_paths[index])
        if not resolved_path:
            continue
        timestamp = frame_times[index] if index < len(frame_times) else None
        hits.append(
            {
                "segment_id": row.get("segment_id"),
                "segment_ref": row.get("segment_ref"),
                "time": row.get("time"),
                "timestamp": round(float(timestamp), 3) if timestamp is not None else None,
                "image": resolved_path,
            }
        )
    return hits


def _build_temporal_occurrences(
    rows: List[Dict[str, Any]],
    *,
    label: str,
    confidence_threshold: float = 0.0,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    occurrences: List[Dict[str, Any]] = []
    for row in rows:
        confidence = float(row.get("_match_confidence", 0.0) or 0.0)
        if confidence < confidence_threshold:
            continue
        parsed = row.get("time_parsed") or {}
        frame_hits = _build_row_frame_hits(row, limit=3)
        sample_timestamps = [
            item["timestamp"]
            for item in frame_hits
            if item.get("timestamp") is not None
        ]
        occurrences.append(
            {
                "entity": label,
                "video": row.get("video"),
                "segment_id": row.get("segment_id"),
                "segment_ref": row.get("segment_ref"),
                "time": row.get("time"),
                "start": round(float(parsed.get("start", 0.0)), 3) if parsed else None,
                "end": round(float(parsed.get("end", 0.0)), 3) if parsed else None,
                "confidence": round(confidence, 4),
                "sample_timestamps": sample_timestamps,
                "frame_hits": frame_hits,
                "snippet": row.get("caption", "") or row.get("transcript", "") or row.get("ocr", ""),
            }
        )

    occurrences.sort(
        key=lambda item: (
            float(item.get("start") or 0.0),
            -float(item.get("confidence") or 0.0),
        )
    )
    return occurrences[:limit]


def _collect_visual_probe_evidence(
    *,
    rows: List[Dict[str, Any]],
    query: Optional[str],
    target_range: Optional[Dict[str, float]],
    video_hint: Optional[str],
    mode: str,
    sample_cap: int = 6,
) -> Dict[str, Any]:
    ranked_rows = _select_relevant_rows(
        rows,
        target_range=target_range,
        query=query,
        video_hint=video_hint,
    )
    if not ranked_rows:
        direct_row = _build_direct_video_row(video_hint, target_range=target_range)
        if direct_row is not None:
            ranked_rows = [direct_row]

    chosen_rows = ranked_rows[:5]
    analysis_range = target_range or _merge_row_ranges(chosen_rows, limit=3)
    image_evidence: List[Dict[str, Any]] = []
    sample_times: List[float] = []
    frames_analyzed = 0
    warning_message: Optional[str] = None
    source_row = chosen_rows[0] if chosen_rows else None

    if source_row and source_row.get("video_path") and analysis_range:
        try:
            window_duration = _range_length(analysis_range)
            sample_count = 1 if window_duration <= 0.25 else min(sample_cap, max(2, int(round(window_duration)) + 1))
            analysis = _analyze_video_window(
                video_path=str(source_row["video_path"]),
                start=analysis_range["start"],
                end=analysis_range["end"],
                transcript="",
                query=query,
                mode=mode,
                sample_count=sample_count,
            )
            image_evidence = list(analysis.get("image_evidence") or [])
            sample_times = list(analysis.get("sample_times") or [])
            frames_analyzed = int(analysis.get("frames_analyzed") or 0)
            warning_message = _merge_warning_messages(warning_message, analysis.get("warning"))
        except RequestedVideoTimeRangeError as exc:
            return {
                "selected_rows": ranked_rows,
                "chosen_rows": chosen_rows,
                "matched_segments": _build_matched_segments(chosen_rows, limit=5),
                "image_evidence": [],
                "image_count": 0,
                "source_image_count": 0,
                "image_pack_modes": [],
                "contains_packed_collage": False,
                "image_pack_summary": "0 image evidence item(s)",
                "image_timestamps": [],
                "image_order_rule": get_planner_image_order_rule_text(),
                "sample_times": [],
                "frames_analyzed": 0,
                "search_window": (
                    _format_range(analysis_range["start"], analysis_range["end"])
                    if analysis_range is not None
                    else None
                ),
                "analysis_range": analysis_range,
                "source_video": source_row.get("video") if source_row else None,
                "source_segment": source_row.get("segment_id") if source_row else None,
                "error": str(exc),
                "warning": None,
            }
        except Exception as exc:
            warning_message = str(exc)
            logger.warning("%s raw frame extraction failed for %s: %s", mode, query or target_range, exc)

    if not image_evidence:
        image_evidence = _fallback_probe_images_from_rows(
            chosen_rows,
            text_hint=f"{mode} {query or ''}",
        )
    image_stats = _summarize_image_evidence(image_evidence)

    return {
        "selected_rows": ranked_rows,
        "chosen_rows": chosen_rows,
        "matched_segments": _build_matched_segments(chosen_rows, limit=5),
        "image_evidence": image_evidence,
        "image_count": image_stats["image_count"],
        "source_image_count": image_stats["source_image_count"],
        "image_pack_modes": image_stats["image_pack_modes"],
        "contains_packed_collage": image_stats["contains_packed_collage"],
        "image_pack_summary": image_stats["image_pack_summary"],
        "image_timestamps": image_stats["image_timestamps"],
        "image_order_rule": image_stats["image_order_rule"],
        "sample_times": sample_times,
        "frames_analyzed": frames_analyzed or image_stats["source_image_count"] or image_stats["image_count"],
        "search_window": (
            _format_range(analysis_range["start"], analysis_range["end"])
            if analysis_range is not None
            else None
        ),
        "analysis_range": analysis_range,
        "source_video": source_row.get("video") if source_row else None,
        "source_segment": source_row.get("segment_id") if source_row else None,
        "warning": warning_message,
    }


def _resolve_image_reference(image: Any) -> Optional[Dict[str, Any]]:
    if isinstance(image, dict):
        direct_path = str(image.get("image", "") or image.get("path", "") or "").strip()
        if direct_path:
            return {
                "path": direct_path,
                "timestamp": image.get("timestamp"),
                "source_images": list(image.get("source_images") or []),
            }
        nested = image.get("image_evidence")
        if nested:
            return _resolve_image_reference(nested)

    if isinstance(image, list):
        for item in image:
            resolved = _resolve_image_reference(item)
            if resolved is not None:
                return resolved
        return None

    text = str(image or "").strip()
    if not text:
        return None
    return {"path": text, "timestamp": None, "source_images": [text]}


def _resolve_image_path(image: Any) -> Optional[Dict[str, Any]]:
    reference = _resolve_image_reference(image)
    if reference is None:
        return None

    raw_path = str(reference.get("path", "") or "").strip()
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        cache_root = Path(settings.data_cache_dir).expanduser()
        if not cache_root.is_absolute():
            cache_root = (settings.project_root / cache_root).resolve()
        candidates.extend(
            [
                Path.cwd() / candidate,
                cache_root / candidate,
                settings.project_root / candidate,
            ]
        )

    for option in candidates:
        try:
            if option.exists():
                resolved = dict(reference)
                resolved["path"] = str(option.resolve())
                return resolved
        except Exception:
            continue
    return None


def _normalize_xywh_region(region: List[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    if len(region) != 4:
        raise ValueError("Region must be [x, y, width, height].")

    x, y, width, height = [float(value) for value in region]
    if max(abs(x), abs(y), abs(width), abs(height)) <= 1.0:
        x *= image_width
        y *= image_height
        width *= image_width
        height *= image_height

    left = max(0, min(image_width - 1, int(math.floor(x))))
    top = max(0, min(image_height - 1, int(math.floor(y))))
    right = max(left + 1, min(image_width, int(math.ceil(x + width))))
    bottom = max(top + 1, min(image_height, int(math.ceil(y + height))))

    if right <= left or bottom <= top:
        raise ValueError("Region resolves to an empty crop.")
    return left, top, right, bottom


def _persist_transformed_image(
    *,
    source_path: str,
    image: Image.Image,
    tool_name: str,
    signature_payload: str,
    timestamp: Optional[float] = None,
    source_images: Optional[List[str]] = None,
) -> Dict[str, Any]:
    resolved_source = Path(source_path).expanduser().resolve()
    source_images = source_images or [str(resolved_source)]
    signature = hashlib.sha1(
        f"{tool_name}|{resolved_source}|{signature_payload}".encode("utf-8")
    ).hexdigest()[:12]
    output_dir = _resolve_visual_observation_root() / resolved_source.stem / f"{tool_name}_{signature}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{resolved_source.stem}_{tool_name.lower()}.jpg"
    if not output_path.exists():
        output_path = _save_jpeg_with_lock(image, output_path, quality=95)
    return {
        "image": str(output_path.resolve()),
        "timestamp": round(float(timestamp), 3) if timestamp is not None else None,
        "sample_times": [round(float(timestamp), 3)] if timestamp is not None else [],
        "source_images": [str(Path(item).expanduser().resolve()) for item in source_images],
        "pack_mode": "single",
        "preserve_native_pixels": True,
        "native_resolution": {
            "width": int(image.width),
            "height": int(image.height),
        },
        "image_order_rule": get_planner_image_order_rule_text(),
    }


def _build_matched_segments(rows: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    return [
        {
            "video": row.get("video"),
            "segment_id": row.get("segment_id"),
            "segment_ref": row.get("segment_ref") or f"{row.get('video')}_{row.get('segment_id')}",
            "raw_segment_id": row.get("segment_id"),
            "time": row.get("time"),
            "frame_paths": row.get("frame_paths", []),
            "snippet": row.get("caption", "") or row.get("transcript", "") or row.get("ocr", ""),
            "result": (
                f"Segment {row.get('segment_id')} at {row.get('time')}: "
                f"{row.get('caption', '') or row.get('transcript', '') or row.get('ocr', '')}"
            ).strip(),
        }
        for row in rows[:limit]
    ]


class EntityTrack(BaseTool):
    """Track visual objects across video timeline."""
    
    name: str = "Entity_Track"
    description: str = (
        "Generate continuous trajectories for specific visual objects across the video timeline. "
        "Input object description, output all start-end time segments and corresponding IDs, "
        "establishing inter-frame associations."
    )
    tags: list = ["base", "visual", "tracking", "weak_proxy"]
    
    @property
    def model_path(self) -> str:
        return settings.entity_track_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "visual_object_description": {
                "type": "string",
                "description": "Description of the visual object to track (e.g., 'man in red shirt')"
            },
            "time_range": {
                "type": ["string", "array", "null"],
                "description": "Optional single time range or list of time ranges to search within",
                "default": None
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum confidence score for detections",
                "default": 0.7
            }
        },
        "required": ["visual_object_description"]
    }
    
    def run(
        self,
        visual_object_description: str,
        time_range: Any = None,
        confidence_threshold: float = 0.7,
        **kwargs,
    ) -> Any:
        """Execute entity tracking.
        
        Args:
            visual_object_description: Object to track
            confidence_threshold: Minimum confidence
            **kwargs: Additional arguments
            
        Returns:
            List of tracking results with time segments and IDs
        """
        rows = _iter_processed_segments(kwargs.get("video") or kwargs.get("video_path"))

        def _run_single(range_value: Any) -> Dict[str, Any]:
            target = _parse_range(range_value)
            ranked_rows = _select_relevant_rows(
                rows,
                target_range=target,
                query=visual_object_description,
                video_hint=kwargs.get("video") or kwargs.get("video_path"),
            )

            matched_rows: List[Dict[str, Any]] = []
            for row in ranked_rows:
                score = max(
                    _match_score(row.get("caption", ""), visual_object_description),
                    _match_score(row.get("ocr", ""), visual_object_description),
                    _match_score(row.get("text", ""), visual_object_description),
                )
                if score <= 0:
                    continue
                confidence = min(0.99, 0.55 + score * 0.45)
                if confidence < confidence_threshold:
                    continue
                enriched_row = dict(row)
                enriched_row["_match_confidence"] = confidence
                matched_rows.append(enriched_row)

            evidence = _collect_visual_probe_evidence(
                rows=rows,
                query=visual_object_description,
                target_range=target or _merge_row_ranges(matched_rows, limit=3),
                video_hint=kwargs.get("video") or kwargs.get("video_path"),
                mode="entity_track",
                sample_cap=8,
            )
            if evidence.get("error"):
                return enrich_tool_result(
                    _build_planner_visual_result(
                        image_evidence=[],
                        status="error",
                        fallback_message=str(evidence.get("error")),
                        extra_fields={
                            "description": visual_object_description,
                            "confidence_threshold": confidence_threshold,
                        },
                    ),
                    tool_name=self.name,
                )
            occurrences = _build_temporal_occurrences(
                matched_rows,
                label=visual_object_description,
                confidence_threshold=confidence_threshold,
                limit=8,
            )
            probe_label = _probe_label(range_value)
            status = "success" if occurrences or evidence["image_evidence"] else "no_valid_result"
            return enrich_tool_result(
                _build_planner_visual_result(
                    image_evidence=evidence.get("image_evidence") or [],
                    image_stats=evidence,
                    status=status,
                    fallback_message=(
                        f"Entity_Track did not find usable image evidence for "
                        f"'{visual_object_description}' in {probe_label}."
                    ),
                    extra_fields={
                        "description": visual_object_description,
                        "confidence_threshold": confidence_threshold,
                        "occurrences": occurrences,
                        "items": occurrences,
                        "first_seen": occurrences[0]["time"] if occurrences else None,
                        "last_seen": occurrences[-1]["time"] if occurrences else None,
                    },
                    warning=evidence.get("warning"),
                ),
                tool_name=self.name,
            )

        probes = _normalize_time_range_probes(time_range)
        logger.info(f"Entity_Track: Tracking '{visual_object_description}' using {self.model_path}")
        if len(probes) <= 1:
            return _run_single(probes[0] if probes else time_range)
        return _build_batch_result(self.name, [_run_single(probe) for probe in probes])


class AudioEventDetect(BaseTool):
    """Detect non-speech audio events."""
    
    name: str = "Audio_Event_Detect"
    description: str = (
        "Identify non-speech environmental audio events. "
        "Input sound description (e.g., 'explosion', 'siren', 'piano music'), "
        "output accurate time segments where the sound occurs, "
        "supplementing information not covered by ASR captions."
    )
    tags: list = ["base", "audio", "detection", "weak_proxy"]
    
    @property
    def model_path(self) -> str:
        return settings.audio_event_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "sound_description": {
                "type": "string",
                "description": "Description of sound to detect"
            },
            "time_range": {
                "type": ["string", "array", "null"],
                "description": "Optional single time range or list of time ranges to search within",
                "default": None
            }
        },
        "required": ["sound_description"]
    }
    
    def run(self, sound_description: str, time_range: Any = None, **kwargs) -> Any:
        """Execute audio event detection.
        
        Args:
            sound_description: Sound to detect
            time_range: Optional time constraint
            **kwargs: Additional arguments
            
        Returns:
            List of detected audio events with timestamps
        """
        rows = _iter_processed_segments()

        def _run_single(range_value: Any) -> Dict[str, Any]:
            target = _parse_range(range_value)
            probe_label = _probe_label(range_value)
            results: List[Dict[str, Any]] = []
            for row in rows:
                if not _overlap(row.get("time_parsed"), target):
                    continue
                score = _match_score(row["transcript"], sound_description)
                if score <= 0:
                    continue
                rng = row.get("time_parsed")
                timestamp_value = int((rng["start"] + rng["end"]) / 2) if rng else None
                results.append(
                    {
                        "event": sound_description,
                        "timestamp": timestamp_value,
                        "time": row["time"],
                        "video": row["video"],
                        "segment_id": row["segment_id"],
                        "confidence": round(min(0.99, 0.5 + score * 0.5), 4),
                        "time_range": probe_label,
                    }
                )
            results.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
            return {
                "probe_label": probe_label,
                "time_range": probe_label,
                "items": results,
                "result": summarize_tool_result(results, tool_name=self.name),
            }

        ranges = _normalize_time_range_probes(time_range)
        if len(ranges) <= 1:
            result = _run_single(ranges[0] if ranges else time_range)
            logger.info(f"Audio_Event_Detect: Detecting '{sound_description}' using {self.model_path}")
            return result["items"]

        probes = [_run_single(range_value) for range_value in ranges]
        logger.info(f"Audio_Event_Detect: Detecting '{sound_description}' using {self.model_path}")
        return _build_batch_result(self.name, probes)


class OCRExtract(BaseTool):
    """Return direct frame evidence for OCR-style inspection."""
    
    name: str = "OCR_Extract"
    description: str = (
        "Return direct frame evidence for OCR-style inspection of visible text. "
        "Targets embedded text beyond caption streams (e.g., road signs, screen content, banners) "
        "so the planner can read exact on-screen numbers or labels from the returned images. "
        "This tool does not run a separate OCR model in the middle; it only samples and returns frames. "
        "`density` must be numeric seconds per frame."
    )
    tags: list = ["base", "ocr", "visual"]
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "image_region": {
                "type": ["string", "number", "array", "null"],
                "description": "Single timestamp/range or a list of timestamps/ranges to extract text from",
                "default": None
            },
            "timestamp": {
                "type": ["string", "number", "array", "null"],
                "description": "Single timestamp or list of timestamps for single-frame OCR",
                "default": None
            },
            "t_start": {
                "type": ["string", "number", "array", "null"],
                "description": "Start time or list of start times for OCR windows",
                "default": None
            },
            "t_end": {
                "type": ["string", "number", "array", "null"],
                "description": "End time or list of end times for OCR windows",
                "default": None
            },
            "density": {
                "type": ["number", "array", "null"],
                "description": "Sampling density in seconds per frame for OCR windows; scalar or list of numeric values",
                "default": 0.5
            },
            "language": {
                "type": "string",
                "description": "Expected language of text",
                "default": "en"
            }
        },
        "required": []
    }

    def validate_args(self, **kwargs) -> bool:
        super().validate_args(**kwargs)
        image_region = kwargs.get("image_region")
        timestamp = kwargs.get("timestamp")
        t_start = kwargs.get("t_start")
        t_end = kwargs.get("t_end")
        density = kwargs.get("density", 0.5)

        has_image_region = image_region not in (None, "", [], {})
        has_timestamp = timestamp not in (None, "", [], {})
        has_time_window = t_start not in (None, "", [], {}) and t_end not in (None, "", [], {})

        if not (has_image_region or has_timestamp or has_time_window):
            raise ValueError("OCR_Extract requires image_region, timestamp, or both t_start and t_end.")
        _coerce_numeric_scalar_or_list("density", density)
        return True
    
    def run(
        self,
        image_region: Any = None,
        language: str = "en",
        timestamp: Any = None,
        t_start: Any = None,
        t_end: Any = None,
        density: Any = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute OCR extraction.
        
        Args:
            image_region: Region or timestamp
            language: Text language
            **kwargs: Additional arguments
            
        Returns:
            Image evidence and timestamps for downstream OCR reading
        """
        rows = _iter_processed_segments()
        specs = _build_visual_probe_specs(
            timestamp=timestamp if timestamp is not None else kwargs.get("timestamp"),
            t_start=t_start,
            t_end=t_end,
            density=density,
            image_region=image_region,
            segment_id=kwargs.get("segment_id"),
            time_windows=kwargs.get("time_windows"),
            rows=rows,
        )
        if not specs:
            raise ValueError("OCR_Extract requires image_region, timestamp, or both t_start and t_end.")

        def _run_probe(spec: Dict[str, Any]) -> Dict[str, Any]:
            region_range = spec["target"]
            ranked_rows = _select_relevant_rows(
                rows,
                target_range=region_range,
                query=kwargs.get("query") or spec.get("probe_label") or image_region,
                video_hint=kwargs.get("video") or kwargs.get("video_path"),
            )
            if not ranked_rows:
                direct_row = _build_direct_video_row(
                    kwargs.get("video") or kwargs.get("video_path"),
                    target_range=region_range,
                )
                if direct_row is not None:
                    ranked_rows = [direct_row]

            chosen = ranked_rows[:5]
            best_src = chosen[0] if chosen else None
            analysis_range = region_range or (best_src.get("time_parsed") if best_src else None)
            image_evidence: List[Dict[str, Any]] = []
            frames_analyzed = 0
            sample_times: List[float] = []
            error_message = None

            if analysis_range and best_src and best_src.get("video_path"):
                try:
                    analysis = _analyze_video_window(
                        video_path=str(best_src["video_path"]),
                        start=analysis_range["start"],
                        end=analysis_range["end"],
                        transcript="",
                        query=kwargs.get("query") or spec.get("probe_label") or image_region,
                        mode="ocr",
                        sample_count=1 if spec["mode"] == "single_frame" else None,
                        sample_density=None if spec["mode"] == "single_frame" else spec.get("density"),
                    )
                    image_evidence = list(analysis.get("image_evidence") or [])
                    frames_analyzed = analysis["frames_analyzed"]
                    sample_times = analysis["sample_times"]
                    analysis_range = analysis.get("effective_range") or analysis_range
                    error_message = _merge_warning_messages(error_message, analysis.get("warning"))
                except RequestedVideoTimeRangeError as exc:
                    return enrich_tool_result(
                        _build_planner_visual_result(
                            image_evidence=[],
                            status="error",
                            fallback_message=str(exc),
                            extra_fields={
                                "region": spec.get("probe_label") or image_region,
                                "timestamp": spec.get("timestamp"),
                                "t_start": spec.get("t_start"),
                                "t_end": spec.get("t_end"),
                                "requested_segment_id": spec.get("requested_segment_id"),
                                "resolved_segment_id": spec.get("resolved_segment_id"),
                                "resolved_segment_ref": spec.get("resolved_segment_ref"),
                                "density": spec.get("density"),
                                "mode": spec.get("mode"),
                                "language": language,
                            },
                        ),
                        tool_name=self.name,
                    )
                except Exception as exc:
                    error_message = str(exc)
                    logger.warning("OCR frame extraction failed for region %s: %s", spec.get("probe_label"), exc)

            if not image_evidence:
                image_evidence = _fallback_probe_images_from_rows(
                    chosen,
                    text_hint=f"ocr {spec.get('probe_label') or image_region or ''}",
                )
            image_stats = _summarize_image_evidence(image_evidence)
            effective_start = (
                round(float(analysis_range["start"]), 3)
                if isinstance(analysis_range, dict) and analysis_range.get("start") is not None
                else spec.get("t_start")
            )
            effective_end = (
                round(float(analysis_range["end"]), 3)
                if isinstance(analysis_range, dict) and analysis_range.get("end") is not None
                else spec.get("t_end")
            )

            return enrich_tool_result(
                _build_planner_visual_result(
                    image_evidence=image_evidence,
                    image_stats=image_stats,
                    status="success" if image_evidence else "no_valid_result",
                    fallback_message="OCR_Extract did not provide usable image evidence for the requested region.",
                    extra_fields={
                        "region": spec.get("probe_label") or image_region,
                        "timestamp": spec.get("timestamp"),
                        "t_start": effective_start,
                        "t_end": effective_end,
                        "requested_t_start": spec.get("t_start"),
                        "requested_t_end": spec.get("t_end"),
                        "requested_segment_id": spec.get("requested_segment_id"),
                        "resolved_segment_id": spec.get("resolved_segment_id"),
                        "resolved_segment_ref": spec.get("resolved_segment_ref"),
                        "density": spec.get("density"),
                        "mode": spec.get("mode"),
                        "language": language,
                    },
                    warning=error_message,
                ),
                tool_name=self.name,
            )

        probe_results = [_run_probe(spec) for spec in specs]
        probe_labels = [
            str(spec.get("probe_label") or _format_range(spec["target"]["start"], spec["target"]["end"]))
            for spec in specs
        ]
        logger.info("OCR_Extract probes=%s", ", ".join(probe_labels))
        return probe_results[0] if len(probe_results) == 1 else _build_batch_result(self.name, probe_results)


class SpeakerDiarization(BaseTool):
    """Speaker separation and identification."""
    
    name: str = "Speaker_Diarization"
    description: str = (
        "Speaker separation and identification. "
        "Input audio segment, output 'who (Speaker A/B/C) spoke during which time segments'."
    )
    tags: list = ["base", "audio", "speaker", "debug_placeholder"]
    
    @property
    def model_path(self) -> str:
        return settings.speaker_diarization_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "audio_segment": {
                "type": ["string", "array"],
                "description": "Audio segment identifier/time range, or a list of them"
            }
        },
        "required": ["audio_segment"]
    }
    
    def run(self, audio_segment: Any, **kwargs) -> Any:
        """Execute speaker diarization.
        
        Args:
            audio_segment: Audio segment to analyze
            **kwargs: Additional arguments
            
        Returns:
            Speaker timeline information
        """
        rows = _iter_processed_segments()

        def _run_single(segment_value: Any) -> Dict[str, Any]:
            target = _parse_range(segment_value)
            candidates = [r for r in rows if _overlap(r.get("time_parsed"), target)]
            speakers = ["Speaker_A", "Speaker_B", "Speaker_C"]
            results: List[Dict[str, Any]] = []
            for idx, row in enumerate(candidates[:6]):
                speaker = speakers[idx % len(speakers)]
                token_count = len(_tokenize(row.get("transcript", "")))
                conf = round(min(0.98, 0.55 + min(token_count, 30) / 100), 4)
                results.append(
                    {
                        "speaker": speaker,
                        "time": row.get("time"),
                        "video": row.get("video"),
                        "segment_id": row.get("segment_id"),
                        "confidence": conf,
                    }
                )
            return {
                "probe_label": str(segment_value),
                "audio_segment": segment_value,
                "items": results,
                "result": summarize_tool_result(results, tool_name=self.name),
            }

        segments = _normalize_time_range_probes(audio_segment)
        if len(segments) <= 1:
            result = _run_single(segments[0] if segments else audio_segment)
            logger.info(f"Speaker_Diarization for segment: {audio_segment} using {self.model_path}")
            return result["items"]

        probes = [_run_single(segment_value) for segment_value in segments]
        logger.info(f"Speaker_Diarization for segment: {audio_segment} using {self.model_path}")
        return _build_batch_result(self.name, probes)


class EmotionAnalysis(BaseTool):
    """Emotion analysis combining multiple modalities."""
    
    name: str = "Emotion_Analysis"
    description: str = (
        "Emotion analysis combining facial expressions, body language, and voice tone. "
        "Determine emotional states (anger, tension, happiness) and scene atmosphere."
    )
    tags: list = ["base", "visual", "audio", "emotion"]
    
    @property
    def audio_model_path(self) -> str:
        return settings.emotion_audio_model_path
    
    @property
    def image_model_path(self) -> str:
        return settings.emotion_image_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "video_clip": {
                "type": ["string", "array"],
                "description": "Video clip identifier/time range, or a list of them"
            },
            "modality": {
                "type": "array",
                "items": {"type": "string", "enum": ["face", "voice", "body"]},
                "description": "Modalities to analyze",
                "default": ["face", "voice"]
            }
        },
        "required": ["video_clip"]
    }
    
    def run(self, video_clip: Any, modality: List[str] = None, **kwargs) -> Any:
        """Execute emotion analysis.
        
        Args:
            video_clip: Video clip to analyze
            modality: Modalities to use
            **kwargs: Additional arguments
            
        Returns:
            Emotion analysis results
        """
        modality = modality or ["face", "voice"]
        rows = _iter_processed_segments()

        def _run_single(clip_value: Any) -> Dict[str, Any]:
            target = _parse_range(clip_value)
            candidates = [r for r in rows if _overlap(r.get("time_parsed"), target)]
            text = " ".join(r.get("text", "") for r in candidates).lower()

            anger_keys = ["argue", "shout", "fight", "angry", "yell", "conflict"]
            happy_keys = ["smile", "laugh", "happy", "cheer", "celebrate"]
            tense_keys = ["tense", "panic", "rush", "alarm", "urgent", "warning", "police"]

            anger = sum(text.count(k) for k in anger_keys)
            happy = sum(text.count(k) for k in happy_keys)
            tension = sum(text.count(k) for k in tense_keys)
            total = max(1, anger + happy + tension)
            scores = {
                "anger": round(anger / total, 4),
                "tension": round(tension / total, 4),
                "happiness": round(happy / total, 4),
            }
            dominant = max(scores, key=scores.get)
            evidence = _collect_visual_probe_evidence(
                rows=rows,
                query="emotion analysis " + " ".join(modality),
                target_range=target or _merge_row_ranges(candidates, limit=3),
                video_hint=kwargs.get("video") or kwargs.get("video_path"),
                mode="emotion_analysis",
                sample_cap=8,
            )
            if evidence.get("error"):
                return enrich_tool_result(
                    _build_planner_visual_result(
                        image_evidence=[],
                        status="error",
                        fallback_message=str(evidence.get("error")),
                        extra_fields={
                            "modalities_analyzed": modality,
                        },
                    ),
                    tool_name=self.name,
                )
            status = "success" if evidence["image_evidence"] else "no_valid_result"
            return enrich_tool_result(
                _build_planner_visual_result(
                    image_evidence=evidence.get("image_evidence") or [],
                    image_stats=evidence,
                    status=status,
                    fallback_message=f"Emotion_Analysis did not provide usable image evidence for {clip_value}.",
                    extra_fields={
                        "modalities_analyzed": modality,
                        "dominant_emotion": dominant,
                        "emotion_scores": scores,
                        "confidence": (
                            round(0.5 + scores[dominant] * 0.5, 4)
                            if evidence["image_evidence"]
                            else 0.0
                        ),
                    },
                    warning=evidence.get("warning"),
                ),
                tool_name=self.name,
            )

        clips = _normalize_time_range_probes(video_clip)
        if len(clips) <= 1:
            result = _run_single(clips[0] if clips else video_clip)
            logger.info(f"Emotion_Analysis for clip: {video_clip}")
            return result

        probes = [_run_single(clip_value) for clip_value in clips]
        logger.info(f"Emotion_Analysis for clip: {video_clip}")
        return _build_batch_result(self.name, probes)


class ActionRecognition(BaseTool):
    """Recognize action categories in video."""
    
    name: str = "Action_Recognition"
    description: str = (
        "Identify action categories in video (e.g., 'running', 'handshake', 'fighting', 'cooking')."
    )
    tags: list = ["base", "visual", "action", "weak_proxy"]
    
    @property
    def model_path(self) -> str:
        return settings.qwen_vl_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "action_description": {
                "type": "string",
                "description": "Description of action to recognize"
            },
            "time_range": {
                "type": ["string", "array", "null"],
                "description": "Optional single time range or list of time ranges to search within",
                "default": None
            }
        },
        "required": ["action_description"]
    }
    
    def run(self, action_description: str, time_range: Any = None, **kwargs) -> Any:
        """Execute action recognition.
        
        Args:
            action_description: Action to recognize
            time_range: Optional time constraint
            **kwargs: Additional arguments
            
        Returns:
            List of recognized actions with timestamps
        """
        rows = _iter_processed_segments(kwargs.get("video") or kwargs.get("video_path"))

        def _run_single(range_value: Any) -> Dict[str, Any]:
            target = _parse_range(range_value)
            probe_label = _probe_label(range_value)
            ranked_rows = _select_relevant_rows(
                rows,
                target_range=target,
                query=action_description,
                video_hint=kwargs.get("video") or kwargs.get("video_path"),
            )

            occurrences: List[Dict[str, Any]] = []
            matched_rows: List[Dict[str, Any]] = []
            for row in ranked_rows:
                score = max(
                    _match_score(row.get("caption", ""), action_description),
                    _match_score(row.get("text", ""), action_description),
                )
                if score <= 0:
                    continue
                confidence = round(min(0.99, 0.5 + score * 0.5), 4)
                enriched_row = dict(row)
                enriched_row["_match_confidence"] = confidence
                matched_rows.append(enriched_row)
                occurrences.append(
                    {
                        "action": action_description,
                        "time": row.get("time"),
                        "video": row.get("video"),
                        "segment_id": row.get("segment_id"),
                        "segment_ref": row.get("segment_ref"),
                        "confidence": confidence,
                        "sample_timestamps": list(row.get("frame_times") or [])[:4],
                        "time_range_searched": probe_label,
                    }
                )

            evidence = _collect_visual_probe_evidence(
                rows=rows,
                query=action_description,
                target_range=target or _merge_row_ranges(matched_rows, limit=3),
                video_hint=kwargs.get("video") or kwargs.get("video_path"),
                mode="action_recognition",
                sample_cap=8,
            )
            if evidence.get("error"):
                return enrich_tool_result(
                    _build_planner_visual_result(
                        image_evidence=[],
                        status="error",
                        fallback_message=str(evidence.get("error")),
                        extra_fields={
                            "action_description": action_description,
                        },
                    ),
                    tool_name=self.name,
                )
            status = "success" if occurrences or evidence["image_evidence"] else "no_valid_result"
            return enrich_tool_result(
                _build_planner_visual_result(
                    image_evidence=evidence.get("image_evidence") or [],
                    image_stats=evidence,
                    status=status,
                    fallback_message=(
                        f"Action_Recognition did not find usable image evidence for "
                        f"'{action_description}' in {probe_label}."
                    ),
                    extra_fields={
                        "action_description": action_description,
                        "items": occurrences,
                        "occurrences": occurrences,
                    },
                    warning=evidence.get("warning"),
                ),
                tool_name=self.name,
            )

        ranges = _normalize_time_range_probes(time_range)
        logger.info(f"Action_Recognition: '{action_description}' using {self.model_path}")
        if len(ranges) <= 1:
            return _run_single(ranges[0] if ranges else time_range)

        return _build_batch_result(self.name, [_run_single(range_value) for range_value in ranges])


class VideoClipQA(BaseTool):
    """Inspect a requested video window by clipping it first, then sending the clip directly to the local VLM."""

    name: str = "Video_Clip_QA"
    description: str = (
        "Comprehensive short-window video understanding tool. Clip the requested start/end window first, "
        "then send that clip file directly to the local VLM and answer open-ended or ambiguous questions "
        "about the clip, including actions, interactions, temporal order, scene changes, object states, "
        "counts, comparisons, and overall visual events. Prefer this broad fallback when the request needs "
        "holistic reasoning over a short video segment instead of only OCR, single frames, tracking, or a "
        "narrow specialized visual probe."
    )
    tags: list = ["base", "visual", "video", "qa"]

    @property
    def model_path(self) -> str:
        return settings.qwen_vl_model_path

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "start_time": {
                "type": ["string", "number"],
                "description": "Clip start time in seconds",
            },
            "end_time": {
                "type": ["string", "number"],
                "description": "Clip end time in seconds",
            },
            "query": {
                "type": "string",
                "description": "Question to answer from the clipped video",
            },
            "segment_id": {
                "type": ["string", "number", "null"],
                "description": "Optional segment reference. If provided, segment-relative start/end times will be converted to full-video absolute times.",
                "default": None,
            },
            "video": {
                "type": ["string", "null"],
                "description": "Optional source video hint",
                "default": None,
            },
            "video_path": {
                "type": ["string", "null"],
                "description": "Optional source video path hint",
                "default": None,
            },
        },
        "required": ["start_time", "end_time", "query"],
    }

    def run(self, start_time: Any, end_time: Any, query: str, **kwargs) -> Dict[str, Any]:
        start = _parse_time_seconds(start_time)
        end = _parse_time_seconds(end_time)
        if start is None or end is None:
            raise ValueError("Video_Clip_QA requires numeric start_time and end_time.")
        video_hint = kwargs.get("video") or kwargs.get("video_path")
        rows = _iter_processed_segments(video_hint)
        normalized_target = _resolve_video_clip_qa_target_range(
            start=start,
            end=end,
            segment_id=kwargs.get("segment_id"),
            rows=rows,
        )
        target_range = normalized_target["target_range"]
        if target_range["end"] <= target_range["start"]:
            raise ValueError("Video_Clip_QA requires end_time greater than start_time.")
        max_window_seconds = max(
            1.0,
            float(getattr(settings, "video_clip_qa_max_window_seconds", 90.0) or 90.0),
        )
        requested_window_seconds = _range_length(target_range)
        if requested_window_seconds > max_window_seconds:
            return enrich_tool_result(
                {
                    "start_time": round(float(target_range["start"]), 3),
                    "end_time": round(float(target_range["end"]), 3),
                    "search_window": _format_range(target_range["start"], target_range["end"]),
                    "query": query,
                    "status": "no_valid_result",
                    "result": (
                        f"Video_Clip_QA accepts windows up to {max_window_seconds:g} seconds, but the "
                        f"requested window is {requested_window_seconds:g} seconds. Use Semantic_Retrieve "
                        "or Segment_Retrieve to localize relevant evidence, then inspect one focused window."
                    ),
                    "recommended_next_tools": ["Semantic_Retrieve", "Segment_Retrieve"],
                },
                tool_name=self.name,
            )

        selected = _select_relevant_rows(
            rows,
            target_range=target_range,
            query=query,
            video_hint=video_hint,
        )
        if not selected:
            active_video_path = runtime_session.get_query_metrics().get("active_video_path")
            direct_row = _build_direct_video_row(
                video_hint or active_video_path,
                target_range=target_range,
            )
            if direct_row is not None:
                selected = [direct_row]

        source_row = selected[0] if selected else None
        if source_row is None or not (source_row.get("video_path") or source_row.get("segment_video_path")):
            raise FileNotFoundError("Video_Clip_QA could not resolve a source video for the requested time window.")

        source_video_path = str(source_row.get("video_path") or source_row.get("segment_video_path") or "").strip()
        normalized_request = _normalize_requested_video_range(
            video_path=source_video_path,
            start=float(target_range["start"]),
            end=float(target_range["end"]),
            tool_name=self.name,
        )
        target_range = {
            "start": float(normalized_request["effective_start"]),
            "end": float(normalized_request["effective_end"]),
        }

        clip_source_path = str(source_row.get("video_path") or "").strip()
        clip_start = target_range["start"]
        clip_end = target_range["end"]
        segment_clip_path = str(source_row.get("segment_video_path") or "").strip()
        segment_range = source_row.get("time_parsed")
        if segment_clip_path and _range_contains(segment_range, target_range):
            segment_start = float(segment_range["start"])
            clip_source_path = segment_clip_path
            clip_start = max(0.0, target_range["start"] - segment_start)
            clip_end = max(clip_start, target_range["end"] - segment_start)

        qa_result = _run_video_clip_qa(
            video_path=clip_source_path,
            start=clip_start,
            end=clip_end,
            query=query,
            model_path=self.model_path,
        )
        answer = str(qa_result.get("answer", "") or "").strip()
        error_text = str(qa_result.get("error", "") or "").strip()
        status = "error" if error_text else ("success" if answer else "no_valid_result")
        result = {
            "start_time": round(float(target_range["start"]), 3),
            "end_time": round(float(target_range["end"]), 3),
            "requested_segment_id": normalized_target.get("requested_segment_id"),
            "resolved_segment_id": normalized_target.get("resolved_segment_id"),
            "resolved_segment_ref": normalized_target.get("resolved_segment_ref"),
            "search_window": _format_range(target_range["start"], target_range["end"]),
            "query": query,
            "source_video": source_row.get("video"),
            "source_segment": source_row.get("segment_id"),
            "matched_segments": _build_matched_segments(selected, limit=3),
            "clip_path": qa_result.get("clip_path"),
            "backend": qa_result.get("backend", "video"),
            "sample_times": list(qa_result.get("sample_times") or []),
            "frames_analyzed": qa_result.get("frames_analyzed"),
            "status": status,
            "result": error_text or answer or "Video_Clip_QA did not return a usable answer for the requested clip.",
        }
        parsed_json = qa_result.get("parsed_json")
        if isinstance(parsed_json, (dict, list)):
            result["parsed_json"] = parsed_json
            if isinstance(parsed_json, dict):
                if "answer" in parsed_json:
                    result["answer_value"] = parsed_json["answer"]
                if "evidence" in parsed_json:
                    result["evidence_items"] = parsed_json["evidence"]
                if "uncertainty" in parsed_json:
                    result["uncertainty_value"] = parsed_json["uncertainty"]
                if "video_summary" in parsed_json:
                    result["video_summary_value"] = parsed_json["video_summary"]
        if error_text:
            result["error"] = error_text
        combined_warning = _merge_warning_messages(
            normalized_request.get("warning"),
            qa_result.get("warning"),
        )
        if combined_warning:
            result["warning"] = combined_warning
        logger.info(
            "Video_Clip_QA window=%s query=%s using %s",
            result["search_window"],
            query,
            self.model_path,
        )
        return enrich_tool_result(result, tool_name=self.name)


class CrossModalVerify(BaseTool):
    """Cross-modal consistency verification."""
    
    name: str = "Cross_Modal_Verify"
    description: str = (
        "Cross-modal consistency verification. "
        "Input visual evidence and textual claim, determine if they contradict. "
        "E.g., caption says 'I got the red book' but video shows blue book."
    )
    tags: list = ["base", "crossmodal", "verification", "weak_proxy"]
    
    @property
    def model_path(self) -> str:
        return settings.qwen_vl_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "visual_evidence": {
                "type": "string",
                "description": "Visual evidence description"
            },
            "textual_claim": {
                "type": "string",
                "description": "Textual claim to verify"
            }
        },
        "required": ["visual_evidence", "textual_claim"]
    }
    
    def run(self, visual_evidence: str, textual_claim: str, **kwargs) -> Dict[str, Any]:
        """Execute cross-modal verification.
        
        Args:
            visual_evidence: Visual description
            textual_claim: Text to verify
            **kwargs: Additional arguments
            
        Returns:
            Verification result
        """
        visual_tokens = set(_tokenize(visual_evidence))
        claim_tokens = set(_tokenize(textual_claim))
        shared = len(visual_tokens.intersection(claim_tokens))
        denom = max(1, len(claim_tokens))
        consistency_score = shared / denom

        negation_pairs = [
            ("red", "blue"),
            ("inside", "outside"),
            ("day", "night"),
            ("with", "without"),
        ]
        contradiction = False
        for left, right in negation_pairs:
            if left in visual_tokens and right in claim_tokens:
                contradiction = True
                break
            if right in visual_tokens and left in claim_tokens:
                contradiction = True
                break

        is_consistent = (consistency_score >= 0.2) and (not contradiction)
        result = {
            "visual_evidence": visual_evidence,
            "textual_claim": textual_claim,
            "is_consistent": is_consistent,
            "confidence": round(1.0 - abs(0.5 - consistency_score), 4),
            "explanation": (
                "Visual and textual modalities are consistent"
                if is_consistent
                else "Potential contradiction or weak overlap between visual evidence and claim"
            ),
        }
        result = enrich_tool_result(result, tool_name=self.name)
        
        logger.info(f"Cross_Modal_Verify: '{textual_claim}' vs '{visual_evidence}' using {self.model_path}")
        return result


class InspectFrame(BaseTool):
    """High-frequency resampling of specific time points."""
    
    name: str = "Inspect_Frame"
    description: str = (
        "Inspect a specific video moment or a short time range via uniform frame sampling. "
        "Supports two modes: Inspect_Frame(timestamp) for a single high-resolution frame, "
        "or Inspect_Frame(t_start, t_end, density) for uniformly sampled frames across a time range "
        "(density is numeric seconds per frame). Returns direct frame evidence for downstream planners."
    )
    tags: list = ["base", "visual", "refine"]
    
    @property
    def model_path(self) -> str:
        return settings.qwen_vl_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": ["string", "number", "array", "null"],
                "description": "Single timestamp or list of timestamps to inspect as single-frame snapshots",
                "default": None
            },
            "t_start": {
                "type": ["string", "number", "array", "null"],
                "description": "Start time or list of start times for inspection windows",
                "default": None
            },
            "t_end": {
                "type": ["string", "number", "array", "null"],
                "description": "End time or list of end times for inspection windows",
                "default": None
            },
            "density": {
                "type": ["number", "array", "null"],
                "description": "Sampling density in seconds per frame for window mode; scalar or list of numeric values",
                "default": 0.5
            },
            "duration": {
                "type": ["number", "null"],
                "description": "Deprecated legacy window size around timestamp; kept for backward compatibility",
                "default": None
            },
            "query": {
                "type": "string",
                "description": "What to look for in the frames",
                "default": None
            }
        },
        "required": []
    }

    def validate_args(self, **kwargs) -> bool:
        super().validate_args(**kwargs)
        timestamp = kwargs.get("timestamp")
        t_start = kwargs.get("t_start")
        t_end = kwargs.get("t_end")
        density = kwargs.get("density", 0.5)

        has_timestamp = timestamp not in (None, "", [], {})
        has_time_window = t_start not in (None, "", [], {}) and t_end not in (None, "", [], {})
        if not (has_timestamp or has_time_window):
            raise ValueError("Inspect_Frame requires timestamp, or both t_start and t_end.")
        _coerce_numeric_scalar_or_list("density", density)
        return True
    
    def run(
        self,
        timestamp: Any = None,
        duration: Optional[float] = None,
        query: Optional[str] = None,
        t_start: Any = None,
        t_end: Any = None,
        density: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute frame inspection.
        
        Args:
            timestamp: Time to inspect as a single frame
            duration: Legacy duration around timestamp
            query: What to look for
            t_start: Window start time
            t_end: Window end time
            density: Seconds per sampled frame for window mode
            **kwargs: Additional arguments
            
        Returns:
            Detailed frame analysis
        """
        rows = _iter_processed_segments()
        specs = _build_visual_probe_specs(
            timestamp=timestamp,
            t_start=t_start,
            t_end=t_end,
            density=density,
            duration=duration,
            segment_id=kwargs.get("segment_id"),
            time_windows=kwargs.get("time_windows"),
            rows=rows,
        )
        if not specs:
            raise ValueError("Inspect_Frame requires timestamp, or both t_start and t_end.")

        def _run_probe(spec: Dict[str, Any]) -> Dict[str, Any]:
            target = spec["target"]
            selected = _select_relevant_rows(
                rows,
                target_range=target,
                query=query,
                video_hint=kwargs.get("video") or kwargs.get("video_path"),
            )
            if not selected:
                direct_row = _build_direct_video_row(
                    kwargs.get("video") or kwargs.get("video_path"),
                    target_range=target,
                )
                if direct_row is not None:
                    selected = [direct_row]
            matched_segments = _build_matched_segments(selected)
            image_evidence: List[Dict[str, Any]] = []
            window_duration = max(0.0, target["end"] - target["start"])
            density_value = max(float(spec.get("density") or 0.5), 1e-3)
            if spec["mode"] == "single_frame":
                frames_analyzed = 1
            elif spec["mode"] == "range":
                frames_analyzed = _sample_count_from_density(window_duration, density_value) or 1
            else:
                frames_analyzed = max(1, int(max(0.0, float(duration or 0.0)) * 24))
            sample_times: List[float] = []
            error_message = None

            source_row = selected[0] if selected else None
            if source_row and source_row.get("video_path"):
                try:
                    analysis = _analyze_video_window(
                        video_path=str(source_row["video_path"]),
                        start=target["start"],
                        end=target["end"],
                        transcript="",
                        query=query,
                        mode="inspect",
                        sample_count=1 if spec["mode"] == "single_frame" else None,
                        sample_density=None if spec["mode"] == "single_frame" else density_value,
                        model_path=self.model_path,
                    )
                    image_evidence = list(analysis.get("image_evidence") or [])
                    sample_times = analysis["sample_times"]
                    frames_analyzed = analysis["frames_analyzed"]
                    target = analysis.get("effective_range") or target
                    error_message = _merge_warning_messages(error_message, analysis.get("warning"))
                except RequestedVideoTimeRangeError as exc:
                    return _build_planner_visual_result(
                        image_evidence=[],
                        status="error",
                        fallback_message=str(exc),
                        extra_fields={
                            "timestamp": spec.get("timestamp"),
                            "t_start": spec.get("t_start"),
                            "t_end": spec.get("t_end"),
                            "requested_segment_id": spec.get("requested_segment_id"),
                            "resolved_segment_id": spec.get("resolved_segment_id"),
                            "resolved_segment_ref": spec.get("resolved_segment_ref"),
                            "duration": duration,
                            "density": density_value,
                            "mode": spec["mode"],
                            "query": query,
                        },
                    )
                except Exception as exc:
                    error_message = str(exc)
                    logger.warning("Inspect_Frame raw frame extraction failed at %s: %s", spec.get("probe_label"), exc)

            if not image_evidence:
                image_evidence = _fallback_probe_images_from_rows(
                    selected,
                    text_hint=f"inspect {spec.get('probe_label') or query or ''}",
                )
            image_stats = _summarize_image_evidence(image_evidence)
            effective_start = round(float(target["start"]), 3)
            effective_end = round(float(target["end"]), 3)

            if not sample_times:
                sample_times = (
                    [effective_start]
                    if spec["mode"] == "single_frame"
                    else _build_uniform_sample_times(effective_start, effective_end, frames_analyzed)
                )

            return _build_planner_visual_result(
                image_evidence=image_evidence,
                image_stats=image_stats,
                status="success" if image_evidence else "no_valid_result",
                fallback_message="Inspect_Frame did not provide usable image evidence for the requested region.",
                extra_fields={
                    "timestamp": spec.get("timestamp"),
                    "t_start": effective_start if spec["mode"] != "single_frame" else spec.get("timestamp"),
                    "t_end": effective_end if spec["mode"] != "single_frame" else spec.get("timestamp"),
                    "requested_t_start": spec.get("t_start"),
                    "requested_t_end": spec.get("t_end"),
                    "requested_segment_id": spec.get("requested_segment_id"),
                    "resolved_segment_id": spec.get("resolved_segment_id"),
                    "resolved_segment_ref": spec.get("resolved_segment_ref"),
                    "duration": duration,
                    "density": density_value,
                    "mode": spec["mode"],
                    "query": query,
                },
                warning=error_message,
            )

        probe_results = [_run_probe(spec) for spec in specs]
        probe_labels = [
            str(spec.get("probe_label") or _format_range(spec["target"]["start"], spec["target"]["end"]))
            for spec in specs
        ]
        logger.info("Inspect_Frame probes=%s", ", ".join(probe_labels))
        return probe_results[0] if len(probe_results) == 1 else _build_batch_result(self.name, probe_results)


class Crop(BaseTool):
    """Crop image to remove background interference."""
    
    name: str = "Crop"
    description: str = (
        "Crop image or extract region to remove background interference. "
        "Limit attention to bounding box selected area, exclude surrounding noise, "
        "help model focus on specific object."
    )
    tags: list = ["base", "visual", "edit"]
    
    @property
    def model_path(self) -> str:
        return settings.birefnet_model_path
    
    @property
    def ocr_model_path(self) -> str:
        return settings.ocr_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image identifier or path"
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Bounding box [x, y, width, height]"
            }
        },
        "required": ["image", "bbox"]
    }
    
    def run(self, image: str, bbox: List[float], **kwargs) -> Dict[str, Any]:
        """Execute image cropping.
        
        Args:
            image: Image identifier
            bbox: Bounding box coordinates
            **kwargs: Additional arguments
            
        Returns:
            Cropped image information
        """
        resolved = _resolve_image_path(image)
        if resolved is None:
            raise FileNotFoundError(f"Crop could not resolve image input: {image!r}")

        source_path = str(resolved["path"])
        timestamp = resolved.get("timestamp")
        source_images = list(resolved.get("source_images") or [source_path])
        try:
            rgb_image = _load_rgb_image(source_path, context="crop source")
        except Exception as exc:
            message = str(exc)
            return enrich_tool_result(
                {
                    "original_image": source_path,
                    "bbox": bbox,
                    "error": message,
                    "status": "error",
                    "result": message,
                },
                tool_name=self.name,
            )

        left, top, right, bottom = _normalize_xywh_region(bbox, rgb_image.width, rgb_image.height)
        cropped = rgb_image.crop((left, top, right, bottom))

        output_evidence = _persist_transformed_image(
            source_path=source_path,
            image=cropped,
            tool_name="crop",
            signature_payload=json.dumps({"bbox": [left, top, right, bottom]}, ensure_ascii=False),
            timestamp=timestamp,
            source_images=source_images,
        )
        width = max(0.0, float(right - left))
        height = max(0.0, float(bottom - top))
        area = width * height
        image_evidence = [output_evidence]
        image_stats = _summarize_image_evidence(image_evidence)
        result = _build_planner_visual_result(
            image_evidence=image_evidence,
            image_stats=image_stats,
            status="success",
            fallback_message="Crop did not produce usable image evidence.",
            extra_fields={
                "original_image": source_path,
                "bbox": bbox,
                "resolved_bbox": [left, top, right - left, bottom - top],
                "cropped_image": output_evidence["image"],
                "dimensions": {"width": width, "height": height},
                "crop_area": area,
            },
        )
        
        logger.info(f"Crop image {source_path} with bbox {bbox} using {self.model_path}")
        return enrich_tool_result(result, tool_name=self.name)


class ZoomIn(BaseTool):
    """Zoom in on image region for detail enhancement."""
    
    name: str = "Zoom_in"
    description: str = (
        "Zoom in on local image region with resolution enhancement. "
        "Used to identify tiny visual features (e.g., distinguishing distant object color, texture details), "
        "simulating human 'looking closer' behavior."
    )
    tags: list = ["base", "visual", "enhance"]
    
    @property
    def model_path(self) -> str:
        return settings.super_resolution_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image identifier or path"
            },
            "region_of_interest": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Region to zoom [x, y, width, height]"
            },
            "zoom_factor": {
                "type": "number",
                "description": "Zoom factor",
                "default": 2.0
            }
        },
        "required": ["image", "region_of_interest"]
    }
    
    def run(self, image: str, region_of_interest: List[float], zoom_factor: float = 2.0, **kwargs) -> Dict[str, Any]:
        """Execute zoom operation.
        
        Args:
            image: Image identifier
            region_of_interest: ROI coordinates
            zoom_factor: Magnification factor
            **kwargs: Additional arguments
            
        Returns:
            Zoomed image information
        """
        resolved = _resolve_image_path(image)
        if resolved is None:
            raise FileNotFoundError(f"Zoom_in could not resolve image input: {image!r}")

        source_path = str(resolved["path"])
        timestamp = resolved.get("timestamp")
        source_images = list(resolved.get("source_images") or [source_path])
        zoom_factor = max(1.0, float(zoom_factor))
        try:
            rgb_image = _load_rgb_image(source_path, context="zoom source")
        except Exception as exc:
            message = str(exc)
            return enrich_tool_result(
                {
                    "original_image": source_path,
                    "roi": region_of_interest,
                    "error": message,
                    "status": "error",
                    "result": message,
                },
                tool_name=self.name,
            )

        left, top, right, bottom = _normalize_xywh_region(
            region_of_interest,
            rgb_image.width,
            rgb_image.height,
        )
        cropped = rgb_image.crop((left, top, right, bottom))
        zoomed = cropped.resize(
            (
                max(1, int(round(cropped.width * zoom_factor))),
                max(1, int(round(cropped.height * zoom_factor))),
            ),
            resample=Image.Resampling.LANCZOS,
        )

        output_evidence = _persist_transformed_image(
            source_path=source_path,
            image=zoomed,
            tool_name="zoom_in",
            signature_payload=json.dumps(
                {
                    "roi": [left, top, right, bottom],
                    "zoom_factor": zoom_factor,
                },
                ensure_ascii=False,
            ),
            timestamp=timestamp,
            source_images=source_images,
        )
        width = float(right - left)
        height = float(bottom - top)
        image_evidence = [output_evidence]
        image_stats = _summarize_image_evidence(image_evidence)
        result = _build_planner_visual_result(
            image_evidence=image_evidence,
            image_stats=image_stats,
            status="success",
            fallback_message="Zoom_in did not produce usable image evidence.",
            extra_fields={
                "original_image": source_path,
                "roi": region_of_interest,
                "resolved_roi": [left, top, right - left, bottom - top],
                "zoom_factor": zoom_factor,
                "zoomed_image": output_evidence["image"],
                "estimated_output_size": {
                    "width": round(width * zoom_factor, 2),
                    "height": round(height * zoom_factor, 2),
                },
            },
        )
        
        logger.info(f"Zoom_in on image {source_path} at ROI {region_of_interest} by {zoom_factor}x using {self.model_path}")
        return enrich_tool_result(result, tool_name=self.name)
