"""
Video segmentation module.

Uses ASR timestamp boundaries by default, with scene/fixed fallback only when
timestamp evidence is unavailable.
"""
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from .frame_assets import export_audio_subclip, export_video_subclip

logger = logging.getLogger(__name__)

try:
    from scenedetect import detect, SceneManager, split_video_ffmpeg  # type: ignore
    from scenedetect.detectors import ContentDetector  # type: ignore
except ImportError:  # pragma: no cover
    detect = None  # type: ignore
    SceneManager = None  # type: ignore
    ContentDetector = None  # type: ignore
    split_video_ffmpeg = None  # type: ignore


class VideoSegmenter:
    """Handles timestamp-aware video segmentation and cached audio extraction."""

    CACHE_VERSION = 10

    def __init__(
        self,
        segment_length: int = 30,
        frame_interval: float = 6.0,
        audio_format: str = "mp3",
        video_format: str = "mp4",
        min_segment_duration: int = 15,
        max_segment_duration: int = 45,
        hard_max_segment_duration: int = 60,
    ):
        self.segment_length = segment_length
        self.frame_interval = frame_interval
        self.audio_format = audio_format
        self.video_format = video_format
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max(max_segment_duration, min_segment_duration)
        self.hard_max_segment_duration = max(hard_max_segment_duration, self.max_segment_duration)
        self.scene_manager_cls = SceneManager
        self._valid_segment_split_reward = 3.0
        self._extended_segment_penalty = 24.0
        self._short_single_segment_penalty = 48.0

    def segment_video(
        self,
        video_path: str,
        working_dir: str,
        use_scene_detection: bool = True,
        asr_segments: Optional[List[Dict[str, Any]]] = None,
        reuse_cache: bool = True,
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """
        Segment video into reusable chunks.

        Returns:
            Tuple of (segment_index2name, segment_times_info).
        """
        video_name = Path(video_path).stem
        cache_path = Path(working_dir) / "_cache" / video_name
        strategy = "asr_semantic_v4" if asr_segments else ("scene_v2" if use_scene_detection else "fixed_v1")

        if reuse_cache:
            cached = self._try_load_cached_segments(cache_path, expected_strategy=strategy)
            if cached:
                return cached

        if cache_path.exists():
            shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=False)

        segment_index2name: Dict[str, str] = {}
        segment_times_info: Dict[str, Dict[str, Any]] = {}

        with VideoFileClip(video_path) as video:
            total_duration = float(video.duration or 0.0)
            ranges = self._build_segment_ranges(
                video_path=video_path,
                total_duration=total_duration,
                use_scene_detection=use_scene_detection,
                asr_segments=asr_segments,
            )

            for segment_idx, (start, end) in enumerate(tqdm(ranges, desc=f"Segmenting {video_name}")):
                duration = max(0.0, end - start)
                num_frames = max(1, int(duration / self.frame_interval))
                frame_times = np.linspace(start, end, num_frames, endpoint=False)

                segment_name = self._build_segment_name(video_name, segment_idx, start, end)
                segment_index2name[str(segment_idx)] = segment_name
                segment_times_info[str(segment_idx)] = {
                    "frame_times": frame_times,
                    "timestamp": (round(float(start), 3), round(float(end), 3)),
                }

                audio_path = cache_path / f"{segment_name}.{self.audio_format}"
                if not audio_path.exists():
                    self._extract_audio(video_path, start, end, audio_path)

        self._persist_cache(
            cache_path=cache_path,
            segment_index2name=segment_index2name,
            segment_times_info=segment_times_info,
            strategy=strategy,
            total_duration=total_duration,
        )
        return segment_index2name, segment_times_info

    def _build_segment_ranges(
        self,
        video_path: str,
        total_duration: float,
        use_scene_detection: bool,
        asr_segments: Optional[List[Dict[str, Any]]],
    ) -> List[Tuple[float, float]]:
        if asr_segments:
            return self._build_semantic_ranges_from_asr(asr_segments, total_duration)

        if use_scene_detection:
            return self._build_scene_ranges(video_path=video_path, total_duration=total_duration)

        start_times = list(range(0, int(total_duration), self.segment_length))
        return self._build_ranges_from_start_times(start_times, total_duration)

    def _build_scene_ranges(
        self,
        video_path: str,
        total_duration: float,
    ) -> List[Tuple[float, float]]:
        scene_boundaries = self._detect_scenes(video_path)
        if not scene_boundaries:
            return self._build_ranges_from_start_times(
                list(range(0, int(total_duration), self.segment_length)),
                total_duration,
            )

        normalized_boundaries = sorted(
            {
                round(max(0.0, float(boundary)), 3)
                for boundary in scene_boundaries
                if float(boundary) < float(total_duration) - 1e-3
            }
        )
        if not normalized_boundaries or normalized_boundaries[0] > 1e-3:
            normalized_boundaries.insert(0, 0.0)

        shot_ranges: List[Tuple[float, float]] = []
        for idx, start in enumerate(normalized_boundaries):
            end = normalized_boundaries[idx + 1] if idx + 1 < len(normalized_boundaries) else float(total_duration)
            if end - start <= 1e-3:
                continue
            shot_ranges.append((float(start), float(end)))

        if not shot_ranges:
            return self._build_ranges_from_start_times(
                list(range(0, int(total_duration), self.segment_length)),
                total_duration,
            )

        units: List[Dict[str, Any]] = []
        for start, end in shot_ranges:
            for split_start, split_end in self._subdivide_long_range(float(start), float(end)):
                units.append({"start": split_start, "end": split_end})

        return self._plan_ranges_from_units(
            units=units,
            total_duration=total_duration,
            boundary_strengths=[0.0 for _ in units],
            cover_full_span=True,
            bridge_gaps=False,
        )

    def _subdivide_long_range(
        self,
        start: float,
        end: float,
    ) -> List[Tuple[float, float]]:
        duration = max(0.0, float(end) - float(start))
        if duration <= self.hard_max_segment_duration + 1e-3:
            return [(float(start), float(end))]

        ranges: List[Tuple[float, float]] = []
        cursor = float(start)
        final_end = float(end)
        target = float(self.segment_length)

        while final_end - cursor > self.hard_max_segment_duration + 1e-3:
            proposed_end = cursor + target
            remaining_after = final_end - proposed_end
            if remaining_after < self.min_segment_duration:
                proposed_end = final_end - float(self.min_segment_duration)
            proposed_end = min(proposed_end, cursor + float(self.max_segment_duration))
            proposed_end = max(proposed_end, cursor + float(self.min_segment_duration))
            if proposed_end - cursor > self.hard_max_segment_duration:
                proposed_end = cursor + float(self.hard_max_segment_duration)
            ranges.append((cursor, proposed_end))
            cursor = proposed_end

        ranges.append((cursor, final_end))

        if len(ranges) > 1 and (ranges[-1][1] - ranges[-1][0]) < self.min_segment_duration:
            prev_start, _ = ranges[-2]
            ranges[-2] = (prev_start, ranges[-1][1])
            ranges.pop()
        return ranges

    def _duration_cost(
        self,
        duration: float,
        *,
        allow_short_single: bool,
    ) -> Optional[float]:
        target = float(self.segment_length)
        if duration <= 1e-3:
            return None
        if duration < self.min_segment_duration - 1e-3:
            if allow_short_single:
                return self._short_single_segment_penalty + (float(self.min_segment_duration) - duration)
            return None
        if duration <= self.max_segment_duration + 1e-3:
            return abs(duration - target) - self._valid_segment_split_reward
        if duration <= self.hard_max_segment_duration + 1e-3:
            return self._extended_segment_penalty + abs(duration - target) - self._valid_segment_split_reward
        return None

    def _plan_ranges_from_units(
        self,
        units: List[Dict[str, Any]],
        total_duration: float,
        boundary_strengths: Optional[List[float]] = None,
        *,
        cover_full_span: bool,
        bridge_gaps: bool,
    ) -> List[Tuple[float, float]]:
        if not units:
            return self._build_ranges_from_start_times(
                list(range(0, int(total_duration), self.segment_length)),
                total_duration,
            )

        normalized_units = [
            {
                "start": max(0.0, float(unit.get("start", 0.0) or 0.0)),
                "end": max(0.0, float(unit.get("end", 0.0) or 0.0)),
            }
            for unit in units
            if float(unit.get("end", 0.0) or 0.0) - float(unit.get("start", 0.0) or 0.0) > 1e-3
        ]
        if not normalized_units:
            return [(0.0, round(float(total_duration), 3))]

        strengths = list(boundary_strengths or [])
        if len(strengths) < len(normalized_units):
            strengths.extend([0.0] * (len(normalized_units) - len(strengths)))

        memo: Dict[int, Optional[Tuple[Tuple[float, int, float, float], List[Tuple[float, float]]]]] = {}

        def solve(start_index: int) -> Optional[Tuple[Tuple[float, int, float, float], List[Tuple[float, float]]]]:
            if start_index >= len(normalized_units):
                return ((0.0, 0, 0.0, 0.0), [])
            if start_index in memo:
                return memo[start_index]

            best: Optional[Tuple[Tuple[float, int, float, float], List[Tuple[float, float]]]] = None
            segment_start = float(normalized_units[start_index]["start"])
            for end_index in range(start_index, len(normalized_units)):
                segment_end = float(normalized_units[end_index]["end"])
                duration = segment_end - segment_start
                if duration > self.hard_max_segment_duration + 1e-3:
                    break

                is_last = end_index == len(normalized_units) - 1
                if is_last and duration < self.min_segment_duration - 1e-3 and start_index > 0:
                    continue

                current_cost = self._duration_cost(
                    duration,
                    allow_short_single=is_last and start_index == 0,
                )
                if current_cost is None:
                    continue

                tail_solution = solve(end_index + 1)
                if tail_solution is None:
                    continue
                tail_key, tail_ranges = tail_solution
                boundary_bonus = 0.0
                if not is_last:
                    boundary_bonus = 0.75 * max(0.0, float(strengths[end_index] or 0.0))
                total_cost = current_cost + tail_key[0] - boundary_bonus
                candidate_ranges = [(segment_start, segment_end)] + tail_ranges
                candidate_key = (
                    round(total_cost, 6),
                    -(len(candidate_ranges)),
                    round(abs(duration - float(self.segment_length)), 6),
                    round(segment_end, 6),
                )
                if best is None or candidate_key < best[0]:
                    best = (candidate_key, candidate_ranges)

            memo[start_index] = best
            return best

        planned = solve(0)
        if planned is None:
            fallback_ranges = [(float(normalized_units[0]["start"]), float(normalized_units[-1]["end"]))]
            return self._normalize_ranges(
                fallback_ranges,
                total_duration,
                fill_trailing_gap=False,
                cover_full_span=cover_full_span,
                bridge_gaps=bridge_gaps,
            )

        return self._normalize_ranges(
            planned[1],
            total_duration,
            fill_trailing_gap=False,
            cover_full_span=cover_full_span,
            bridge_gaps=bridge_gaps,
        )

    def _token_boundary_score(self, token: str, next_token: str = "") -> int:
        current = str(token or "").strip()
        upcoming = str(next_token or "").strip().strip("\"'()[]{}").lower()
        if not current:
            return 0
        if re.search(r'[.!?\u3002\uff01\uff1f]["\')\]]*$', current):
            return 4
        if re.search(r'[;:\uff1b\uff1a]["\')\]]*$', current):
            return 3
        if re.search(r'[,\uff0c]["\')\]]*$', current):
            return 2
        if upcoming in {
            "and", "but", "so", "because", "then", "now", "when", "while",
            "however", "therefore", "meanwhile", "instead", "finally",
        }:
            return 1
        return 0

    def _find_semantic_cut(
        self,
        tokens: List[str],
        start_idx: int,
        min_size: int,
        target_size: int,
        max_size: int,
    ) -> int:
        end_idx = min(len(tokens), start_idx + max_size)
        lower_idx = min(len(tokens), start_idx + max(1, min_size))
        if end_idx <= lower_idx:
            return end_idx

        best_cut = end_idx
        best_key: Optional[Tuple[int, int, int]] = None
        for cut_idx in range(lower_idx, end_idx + 1):
            prev_token = tokens[cut_idx - 1] if cut_idx - 1 < len(tokens) else ""
            next_token = tokens[cut_idx] if cut_idx < len(tokens) else ""
            boundary_score = self._token_boundary_score(prev_token, next_token)
            chunk_size = cut_idx - start_idx
            distance = abs(chunk_size - target_size)
            candidate_key = (boundary_score, -distance, -chunk_size)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_cut = cut_idx
        return best_cut

    def _split_long_asr_segment(
        self,
        segment: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        start = max(0.0, float(segment.get("start", 0.0) or 0.0))
        end = max(start, float(segment.get("end", start) or start))
        duration = end - start
        text = str(segment.get("text", "") or "").strip()
        if duration <= self.hard_max_segment_duration + 1e-3:
            return [segment]

        tokens = [tok for tok in text.split() if tok]
        if len(tokens) < 2:
            slices: List[Dict[str, Any]] = []
            cursor = start
            target_span = float(self.segment_length)
            while cursor < end - 1e-3:
                next_end = min(end, cursor + target_span)
                if end - next_end < self.min_segment_duration:
                    next_end = end
                slices.append(
                    {
                        "start": round(cursor, 3),
                        "end": round(next_end, 3),
                        "text": text,
                    }
                )
                cursor = next_end
            return slices or [segment]

        tokens_per_second = len(tokens) / max(duration, 1e-6)
        min_tokens = max(4, int(round(tokens_per_second * self.min_segment_duration)))
        target_tokens = max(min_tokens, int(round(tokens_per_second * self.segment_length)))
        max_tokens = max(target_tokens, int(round(tokens_per_second * self.hard_max_segment_duration)))

        refined: List[Dict[str, Any]] = []
        cursor = 0
        total_tokens = len(tokens)
        while cursor < total_tokens:
            remaining = total_tokens - cursor
            if remaining <= max_tokens:
                cut_idx = total_tokens
            else:
                cut_idx = self._find_semantic_cut(
                    tokens=tokens,
                    start_idx=cursor,
                    min_size=min_tokens,
                    target_size=target_tokens,
                    max_size=max_tokens,
                )
            chunk_start = start + duration * (cursor / total_tokens)
            chunk_end = start + duration * (cut_idx / total_tokens)
            chunk_text = " ".join(tokens[cursor:cut_idx]).strip()
            if chunk_end - chunk_start > 1e-3:
                refined.append(
                    {
                        "start": round(chunk_start, 3),
                        "end": round(chunk_end, 3),
                        "text": chunk_text,
                    }
                )
            cursor = max(cut_idx, cursor + 1)

        return refined or [segment]

    def _split_segment_on_sentence_boundaries(
        self,
        segment: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        start = max(0.0, float(segment.get("start", 0.0) or 0.0))
        end = max(start, float(segment.get("end", start) or start))
        text = str(segment.get("text", "") or "").strip()
        duration = end - start
        normalized = {
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        }
        if duration <= 1e-3 or not text:
            return [normalized]

        split_offsets = [
            match.end()
            for match in re.finditer(r'[.!?\u3002\uff01\uff1f]["\')\]]*(?:\s+|$)', text)
            if 0 < match.end() < len(text)
        ]
        if not split_offsets:
            return [normalized]

        weights = [0]
        for char in text:
            weights.append(weights[-1] + (0 if char.isspace() else 1))
        total_weight = max(1, weights[-1])

        slices: List[Dict[str, Any]] = []
        cursor = 0
        for split_offset in split_offsets + [len(text)]:
            piece = text[cursor:split_offset].strip()
            if not piece:
                cursor = split_offset
                continue
            piece_start_weight = weights[cursor]
            piece_end_weight = weights[split_offset]
            if piece_end_weight <= piece_start_weight:
                cursor = split_offset
                continue
            piece_start = start + duration * (piece_start_weight / total_weight)
            piece_end = start + duration * (piece_end_weight / total_weight)
            slices.append(
                {
                    "start": round(piece_start, 3),
                    "end": round(piece_end, 3),
                    "text": piece,
                }
            )
            cursor = split_offset
        return slices or [normalized]

    def _refine_asr_segments(
        self,
        asr_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        refined: List[Dict[str, Any]] = []
        for segment in asr_segments:
            sentence_aligned = self._split_segment_on_sentence_boundaries(segment)
            for unit in sentence_aligned:
                refined.extend(self._split_long_asr_segment(unit))
        return refined or asr_segments

    def prepare_asr_timestamp_units(
        self,
        asr_segments: Optional[List[Dict[str, Any]]],
        total_duration: float,
    ) -> List[Dict[str, Any]]:
        """Normalize and refine Whisper timestamp units before downstream use.

        The same refined units must drive both semantic segmentation and
        per-segment transcript projection; otherwise we can split a sentence at
        the segmentation boundary and then reintroduce the old broken Whisper
        chunks while rebuilding transcripts.
        """
        cleaned_segments: List[Dict[str, Any]] = []
        for segment in asr_segments or []:
            try:
                start = max(0.0, float(segment.get("start", 0.0)))
                end = max(start, float(segment.get("end", start)))
            except Exception:
                continue
            if end - start <= 1e-3:
                continue
            cleaned_segments.append(
                {
                    "start": start,
                    "end": min(float(total_duration), end),
                    "text": str(segment.get("text", "")).strip(),
                }
            )

        if not cleaned_segments:
            return []
        return self._refine_asr_segments(cleaned_segments)

    def _build_segment_name(self, video_name: str, segment_idx: int, start: float, end: float) -> str:
        safe_video_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", video_name).strip("_") or "video"
        start_ms = int(round(float(start) * 1000))
        end_ms = int(round(float(end) * 1000))
        return f"{safe_video_name}-seg{segment_idx:03d}-{start_ms:09d}-{end_ms:09d}"

    def _extract_audio(self, video_path: str, start: float, end: float, output_path: os.PathLike) -> None:
        """Extract audio track for a given time range."""
        try:
            exported = export_audio_subclip(
                video_path=video_path,
                start=start,
                end=end,
                output_path=Path(output_path),
            )
            if not exported:
                logger.debug("No audio track found for segment %s-%s", start, end)
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning("Failed to extract audio for %s-%s: %s", start, end, exc)

    def _detect_scenes_once(self, video_path: str, threshold: float) -> List[float]:
        detector = ContentDetector(threshold=threshold) if ContentDetector else None
        try:
            boundaries: List[float] = [0.0]
            scene_list = []
            if self.scene_manager_cls and detector:
                from scenedetect import open_video  # lazy import to avoid hard dep

                video_manager = open_video(video_path)
                manager = self.scene_manager_cls()
                manager.add_detector(detector)
                if hasattr(video_manager, "set_downscale_factor"):
                    video_manager.set_downscale_factor()
                if hasattr(video_manager, "start"):
                    video_manager.start()
                manager.detect_scenes(video_manager)
                scene_list = getattr(manager, "get_scene_list", lambda: [])()
                if hasattr(video_manager, "release"):
                    video_manager.release()
            elif detect and detector:
                scene_list = detect(video_path, detector)

            for scene in scene_list or []:
                start_candidate = None
                if isinstance(scene, (list, tuple)) and scene:
                    start_part = scene[0]
                    if hasattr(start_part, "get_seconds"):
                        start_candidate = start_part.get_seconds()
                    else:
                        start_candidate = float(start_part)
                elif hasattr(scene, "get_seconds"):
                    start_candidate = scene.get_seconds()
                if start_candidate is not None:
                    boundaries.append(float(start_candidate))
            unique_boundaries = sorted(set(boundaries))
            if len(unique_boundaries) <= 1:
                return []
            return unique_boundaries
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("Scene detection failed: %s, falling back to fixed intervals", exc)
            return []

    def _detect_scenes(self, video_path: str, threshold: float = 27.0) -> List[float]:
        thresholds: List[float] = []
        for candidate in (threshold, 22.0, 18.0):
            try:
                normalized = float(candidate)
            except Exception:
                continue
            if normalized <= 0:
                continue
            if any(abs(existing - normalized) <= 1e-6 for existing in thresholds):
                continue
            thresholds.append(normalized)

        best_boundaries: List[float] = []
        for current_threshold in thresholds:
            boundaries = self._detect_scenes_once(video_path, current_threshold)
            if len(boundaries) > len(best_boundaries):
                best_boundaries = boundaries
            if len(boundaries) > 1:
                if current_threshold != thresholds[0]:
                    logger.info(
                        "Scene detection recovered %d boundaries for %s at threshold %.1f.",
                        len(boundaries),
                        video_path,
                        current_threshold,
                    )
                return boundaries
        return best_boundaries if len(best_boundaries) > 1 else []

    def _build_ranges_from_start_times(
        self,
        start_times: List[float],
        total_duration: float,
    ) -> List[Tuple[float, float]]:
        if total_duration <= 0:
            return [(0.0, 0.0)]

        normalized = sorted(
            {
                round(max(0.0, float(start)), 3)
                for start in start_times
                if float(start) < float(total_duration) + 1e-3
            }
        )
        if not normalized or normalized[0] > 1e-3:
            normalized.insert(0, 0.0)

        ranges: List[Tuple[float, float]] = []
        for idx, start in enumerate(normalized):
            end = normalized[idx + 1] if idx + 1 < len(normalized) else total_duration
            if end - start <= 1e-3:
                continue
            ranges.append((float(start), float(end)))
        return self._normalize_ranges(ranges, total_duration, fill_trailing_gap=True)

    def _boundary_strength(self, text: str, gap_after: float = 0.0) -> int:
        cleaned = str(text or "").strip()
        if not cleaned:
            return 0
        if re.search("[.!?\u3002\uff01\uff1f][\\\"')\\]]*$", cleaned):
            return 3
        if gap_after >= 0.8:
            return 2
        if re.search("[;:\uff1b\uff1a][\\\"')\\]]*$", cleaned):
            return 2
        if re.search("[,\uff0c][\\\"')\\]]*$", cleaned):
            return 1
        return 0

    def _semantic_boundary_strength(self, text: str, gap_after: float = 0.0) -> int:
        cleaned = str(text or "").strip()
        if not cleaned:
            return 0
        if re.search(r'[.!?\u3002\uff01\uff1f]["\')\]]*$', cleaned):
            return 3
        if gap_after >= 0.8:
            return 2
        if re.search(r'[;:\uff1b\uff1a]["\')\]]*$', cleaned):
            return 2
        if re.search(r'[,\uff0c]["\')\]]*$', cleaned):
            return 1
        return 0

    def _select_semantic_boundary_candidate(
        self,
        target_candidates: List[Tuple[int, float, float, int, float]],
        extended_candidates: List[Tuple[int, float, float, int, float]],
    ) -> Optional[Tuple[int, float, float, int, float]]:
        target_strong = [candidate for candidate in target_candidates if candidate[0] >= 2]
        extended_strong = [candidate for candidate in extended_candidates if candidate[0] >= 2]
        candidates = target_strong or extended_strong
        if not candidates:
            return None
        return max(candidates)

    def _build_semantic_ranges_from_asr(
        self,
        asr_segments: List[Dict[str, Any]],
        total_duration: float,
    ) -> List[Tuple[float, float]]:
        cleaned_segments = self.prepare_asr_timestamp_units(
            asr_segments=asr_segments,
            total_duration=total_duration,
        )

        if not cleaned_segments:
            return self._build_ranges_from_start_times(
                list(range(0, int(total_duration), self.segment_length)),
                total_duration,
            )

        units: List[Dict[str, Any]] = []
        strengths: List[float] = []
        for index, segment in enumerate(cleaned_segments):
            start = max(0.0, float(segment.get("start", 0.0) or 0.0))
            end = max(start, float(segment.get("end", start) or start))
            if end - start <= 1e-3:
                continue
            next_start = (
                float(cleaned_segments[index + 1].get("start", end) or end)
                if index + 1 < len(cleaned_segments)
                else float(total_duration)
            )
            gap_after = max(0.0, next_start - end)
            units.append({"start": start, "end": end})
            strengths.append(self._semantic_boundary_strength(segment.get("text", ""), gap_after=gap_after))

        if not units:
            return [(0.0, round(float(total_duration), 3))]

        return self._plan_ranges_from_units(
            units=units,
            total_duration=total_duration,
            boundary_strengths=strengths,
            cover_full_span=True,
            bridge_gaps=True,
        )

    def _normalize_ranges(
        self,
        ranges: List[Tuple[float, float]],
        total_duration: float,
        fill_trailing_gap: bool = True,
        cover_full_span: bool = False,
        bridge_gaps: bool = False,
    ) -> List[Tuple[float, float]]:
        if total_duration <= 0:
            return [(0.0, 0.0)]

        normalized: List[Tuple[float, float]] = []
        for start, end in ranges:
            start = max(0.0, float(start))
            end = min(float(total_duration), max(start, float(end)))
            if end - start <= 1e-3:
                continue
            # Merge only true overlaps. Adjacent segments that touch at a boundary
            # are valid semantic cuts and must remain separate.
            if normalized and start < normalized[-1][1] - 1e-6:
                normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
            else:
                normalized.append((start, end))

        if not normalized:
            normalized = [(0.0, float(total_duration))]

        if bridge_gaps and len(normalized) > 1:
            bridged: List[Tuple[float, float]] = [normalized[0]]
            for start, end in normalized[1:]:
                prev_start, prev_end = bridged[-1]
                if start > prev_end + 1e-6:
                    bridged[-1] = (prev_start, start)
                elif start < prev_end - 1e-6:
                    start = prev_end
                bridged.append((start, end))
            normalized = bridged

        if cover_full_span and normalized:
            first_start, first_end = normalized[0]
            normalized[0] = (0.0, first_end)
            last_start, last_end = normalized[-1]
            normalized[-1] = (last_start, float(total_duration))

        if fill_trailing_gap and normalized[-1][1] < total_duration:
            last_start, _ = normalized[-1]
            normalized[-1] = (last_start, float(total_duration))

        if len(normalized) > 1 and (normalized[-1][1] - normalized[-1][0]) < self.min_segment_duration:
            prev_start, _ = normalized[-2]
            normalized[-2] = (prev_start, normalized[-1][1])
            normalized.pop()

        return [
            (round(start, 3), round(end, 3))
            for start, end in normalized
            if end - start > 1e-3
        ]

    def _cache_meta_path(self, cache_path: Path) -> Path:
        return cache_path / "segments_meta.json"

    def _segment_times_info_path(self, cache_path: Path) -> Path:
        return cache_path / "segment_times_info.json"

    def _persist_cache(
        self,
        cache_path: Path,
        segment_index2name: Dict[str, str],
        segment_times_info: Dict[str, Dict],
        strategy: str,
        total_duration: float,
    ) -> None:
        meta = {
            "cache_version": self.CACHE_VERSION,
            "strategy": strategy,
            "video_duration": round(float(total_duration or 0.0), 3),
            "segment_index2name": segment_index2name,
            "segment_times_info": {
                idx: {
                    "frame_times": info["frame_times"].tolist(),
                    "timestamp": info["timestamp"],
                }
                for idx, info in segment_times_info.items()
            },
        }
        with open(self._cache_meta_path(cache_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(self._segment_times_info_path(cache_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _try_load_cached_segments(
        self,
        cache_path: Path,
        expected_strategy: Optional[str] = None,
    ) -> Optional[Tuple[Dict[str, str], Dict[str, Dict]]]:
        meta_path = self._segment_times_info_path(cache_path)
        if not meta_path.exists():
            meta_path = self._cache_meta_path(cache_path)
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if int(meta.get("cache_version", 0) or 0) != self.CACHE_VERSION:
                return None
            if expected_strategy and str(meta.get("strategy", "")).strip() != expected_strategy:
                return None

            segment_index2name = meta.get("segment_index2name", {})
            raw_times = meta.get("segment_times_info", {})
            cached_duration = float(meta.get("video_duration", 0.0) or 0.0)

            for _, name in segment_index2name.items():
                audio_path = cache_path / f"{name}.{self.audio_format}"
                if not audio_path.exists():
                    return None

            segment_times_info: Dict[str, Dict[str, Any]] = {}
            for idx, info in raw_times.items():
                frame_times = np.array(info.get("frame_times", []), dtype=float)
                segment_times_info[idx] = {
                    "frame_times": frame_times,
                    "timestamp": tuple(info.get("timestamp", (0, 0))),
                }
            ordered_ranges: List[Tuple[float, float]] = []
            for info in segment_times_info.values():
                timestamp = info.get("timestamp", (0, 0))
                if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                    return None
                try:
                    ordered_ranges.append((float(timestamp[0]), float(timestamp[1])))
                except Exception:
                    return None
            ordered_ranges.sort()
            if ordered_ranges:
                if ordered_ranges[0][0] > 0.05:
                    logger.warning(
                        "Rejecting stale segment cache at %s: first segment starts at %.2fs instead of 0.",
                        cache_path,
                        ordered_ranges[0][0],
                    )
                    return None
                previous_end = ordered_ranges[0][0]
                for start, end in ordered_ranges:
                    if start > previous_end + 0.05:
                        logger.warning(
                            "Rejecting stale segment cache at %s: gap detected between %.2fs and %.2fs.",
                            cache_path,
                            previous_end,
                            start,
                        )
                        return None
                    previous_end = max(previous_end, end)
                if cached_duration > 0.0 and ordered_ranges[-1][1] < cached_duration - 0.05:
                    logger.warning(
                        "Rejecting stale segment cache at %s: cached segments end at %.2fs but video lasts %.2fs.",
                        cache_path,
                        ordered_ranges[-1][1],
                        cached_duration,
                    )
                    return None
            if len(segment_times_info) == 1:
                only_info = next(iter(segment_times_info.values()), {})
                timestamp = only_info.get("timestamp", (0, 0))
                if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                    try:
                        duration = float(timestamp[1]) - float(timestamp[0])
                    except Exception:
                        duration = 0.0
                    if duration > float(self.hard_max_segment_duration) + 1e-3:
                        logger.warning(
                            "Rejecting stale segment cache at %s: single segment spans %.2fs.",
                            cache_path,
                            duration,
                        )
                        return None
            return segment_index2name, segment_times_info
        except Exception:
            return None

    def _merge_short_scenes(self, boundaries: List[int], total_duration: int) -> List[int]:
        if not boundaries:
            return list(range(0, total_duration, self.segment_length))
        if len(boundaries) <= 1:
            return list(range(0, total_duration, self.segment_length))

        merged = [boundaries[0]]
        for boundary in boundaries[1:]:
            if boundary - merged[-1] >= self.segment_length:
                merged.append(boundary)

        if merged[0] != 0:
            merged.insert(0, 0)
        return merged

    def save_video_segments(
        self,
        video_path: str,
        working_dir: str,
        segment_index2name: Dict[str, str],
        segment_times_info: Dict[str, Dict],
    ) -> None:
        video_name = Path(video_path).stem
        cache_path = Path(working_dir) / "_cache" / video_name

        for index in tqdm(segment_index2name, desc=f"Saving segments for {video_name}"):
            start, end = segment_times_info[index]["timestamp"]
            video_file = f"{segment_index2name[index]}.{self.video_format}"
            export_video_subclip(
                video_path=video_path,
                start=start,
                end=end,
                output_path=cache_path / video_file,
            )
