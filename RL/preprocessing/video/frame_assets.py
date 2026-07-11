"""Persist sampled segment frames for downstream multimodal retrieval."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip


def build_segment_aliases(video_name: str, segment_id: Any) -> List[str]:
    raw_segment_id = str(segment_id)
    aliases = [raw_segment_id, f"{video_name}_{raw_segment_id}"]
    return list(dict.fromkeys(aliases))


def _normalize_frame_times(value: Any) -> List[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    normalized: List[float] = []
    for item in value:
        try:
            normalized.append(round(float(item), 3))
        except Exception:
            continue
    return normalized


def _frame_filename(index: int, timestamp: float) -> str:
    stamp = str(timestamp).replace(".", "p")
    return f"frame_{index:03d}_{stamp}.jpg"


def _clip_filename(segment_id: str) -> str:
    safe_segment_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(segment_id or "")).strip("_") or "segment"
    return f"{safe_segment_id}.mp4"


def _is_materialized_media_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _build_subclip(video: VideoFileClip, start: float, end: float):
    if hasattr(video, "subclip"):
        return video.subclip(start, end)
    if hasattr(video, "subclipped"):
        return video.subclipped(start, end)
    raise RuntimeError("moviepy VideoFileClip does not expose subclip/subclipped")


def _safe_frame_upper_bound(duration: float, fps: float) -> float:
    safe_duration = max(0.0, float(duration or 0.0))
    safe_fps = max(0.0, float(fps or 0.0))
    frame_interval = (1.0 / safe_fps) if safe_fps > 1e-6 else 1e-2
    end_margin = max(1e-3, frame_interval)
    return max(0.0, safe_duration - end_margin)


def _add_verbose_flag(write_fn, write_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        write_signature = inspect.signature(write_fn)
        if "verbose" in write_signature.parameters:
            write_kwargs["verbose"] = False
    except Exception:
        pass
    return write_kwargs


def _add_temp_audiofile_kwargs(write_fn, output_path: Path, write_kwargs: Dict[str, Any]) -> Path | None:
    try:
        write_signature = inspect.signature(write_fn)
    except Exception:
        return None

    temp_audio_path = output_path.with_name(f"{output_path.stem}__temp_audio__.m4a")
    parameters = write_signature.parameters

    if "temp_audiofile" in parameters:
        write_kwargs["temp_audiofile"] = str(temp_audio_path)
    elif "temp_audiofile_path" in parameters:
        write_kwargs["temp_audiofile_path"] = str(output_path.parent)
    else:
        return None

    if "remove_temp" in parameters:
        write_kwargs["remove_temp"] = True
    return temp_audio_path


def export_audio_subclip(video_path: str, start: float, end: float, output_path: Path) -> bool:
    """Write one audio segment by reopening the source video for this export only."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(video_path)) as video:
        video_duration = float(getattr(video, "duration", 0.0) or 0.0)
        if video_duration <= 0:
            raise RuntimeError(f"Video has invalid duration: {video_duration}")

        safe_start = max(0.0, min(float(start), max(0.0, video_duration - 1e-3)))
        safe_end = max(safe_start + 1e-3, min(float(end), video_duration))
        clip_segment = _build_subclip(video, safe_start, safe_end)

        try:
            audio_track = getattr(clip_segment, "audio", None)
            if audio_track is None:
                return False

            audio_codec = output_path.suffix.lstrip(".") or "mp3"
            write_kwargs = {
                "codec": "pcm_s16le" if audio_codec == "wav" else audio_codec,
                "logger": None,
            }
            write_kwargs = _add_verbose_flag(audio_track.write_audiofile, write_kwargs)
            audio_track.write_audiofile(str(output_path), **write_kwargs)
            return True
        finally:
            close_audio = getattr(getattr(clip_segment, "audio", None), "close", None)
            if callable(close_audio):
                close_audio()
            close_segment = getattr(clip_segment, "close", None)
            if callable(close_segment):
                close_segment()


def _set_clip_fps(clip_segment: Any, target_fps: float) -> Any:
    setter = getattr(clip_segment, "with_fps", None)
    if callable(setter):
        return setter(float(target_fps))
    setter = getattr(clip_segment, "set_fps", None)
    if callable(setter):
        return setter(float(target_fps))
    return clip_segment


def _resize_clip_segment(clip_segment: Any, target_size: tuple[int, int]) -> Any:
    resize_method = getattr(clip_segment, "resized", None)
    if callable(resize_method):
        try:
            return resize_method(new_size=target_size)
        except TypeError:
            return resize_method(target_size)
    resize_method = getattr(clip_segment, "resize", None)
    if callable(resize_method):
        try:
            return resize_method(newsize=target_size)
        except TypeError:
            return resize_method(target_size)
    return clip_segment


def export_video_subclip(
    video_path: str,
    start: float,
    end: float,
    output_path: Path,
    *,
    target_fps: float | None = None,
    max_side: int | None = None,
) -> None:
    """Write one video segment by reopening the source video for this export only."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_audio_path: Path | None = None

    with VideoFileClip(str(video_path)) as video:
        video_duration = float(getattr(video, "duration", 0.0) or 0.0)
        if video_duration <= 0:
            raise RuntimeError(f"Video has invalid duration: {video_duration}")

        safe_start = max(0.0, min(float(start), max(0.0, video_duration - 1e-3)))
        safe_end = max(safe_start + 1e-3, min(float(end), video_duration))
        clip_segment = _build_subclip(video, safe_start, safe_end)

        try:
            if target_fps is not None and float(target_fps) > 0:
                clip_segment = _set_clip_fps(clip_segment, float(target_fps))
            if max_side is not None and int(max_side) > 0:
                width = int(getattr(clip_segment, "w", 0) or 0)
                height = int(getattr(clip_segment, "h", 0) or 0)
                longest_side = max(width, height)
                if longest_side > int(max_side) and width > 0 and height > 0:
                    scale = float(int(max_side)) / float(longest_side)
                    target_size = (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    )
                    clip_segment = _resize_clip_segment(clip_segment, target_size)
            write_kwargs = {
                "codec": "libx264",
                "logger": None,
            }
            if target_fps is not None and float(target_fps) > 0:
                write_kwargs["fps"] = float(target_fps)
            if getattr(clip_segment, "audio", None) is not None:
                write_kwargs["audio_codec"] = "aac"
                temp_audio_path = _add_temp_audiofile_kwargs(
                    clip_segment.write_videofile,
                    output_path,
                    write_kwargs,
                )
            else:
                write_kwargs["audio"] = False
            write_kwargs = _add_verbose_flag(clip_segment.write_videofile, write_kwargs)
            clip_segment.write_videofile(str(output_path), **write_kwargs)
        finally:
            if temp_audio_path is not None and temp_audio_path.exists():
                try:
                    temp_audio_path.unlink()
                except Exception:
                    pass
            close_audio = getattr(getattr(clip_segment, "audio", None), "close", None)
            if callable(close_audio):
                close_audio()
            close_segment = getattr(clip_segment, "close", None)
            if callable(close_segment):
                close_segment()


def persist_segment_frame_assets(
    video_path: str,
    video_name: str,
    segment_times_info: Dict[str, Dict[str, Any]],
    output_dir: Path,
    frames_subdir: str,
) -> Dict[str, Dict[str, Any]]:
    """Persist sampled frames and annotate segment metadata with relative paths."""
    frames_root = output_dir / frames_subdir
    frames_root.mkdir(parents=True, exist_ok=True)

    if not segment_times_info:
        return segment_times_info

    with VideoFileClip(video_path) as video:
        video_duration = float(getattr(video, "duration", 0.0) or 0.0)
        video_fps = float(getattr(video, "fps", 0.0) or 0.0)
        max_time = _safe_frame_upper_bound(video_duration, video_fps)

        for raw_segment_id, info in segment_times_info.items():
            seg_id = str(raw_segment_id)
            frame_times = _normalize_frame_times(info.get("frame_times"))
            realized_frame_times: List[float] = []
            frame_paths: List[str] = []
            segment_dir = frames_root / seg_id
            segment_dir.mkdir(parents=True, exist_ok=True)

            for index, timestamp in enumerate(frame_times):
                safe_time = min(max(0.0, float(timestamp)), max_time)
                realized_frame_times.append(round(safe_time, 3))
                frame_path = segment_dir / _frame_filename(index, safe_time)
                if not frame_path.exists():
                    frame = video.get_frame(safe_time)
                    Image.fromarray(frame).save(frame_path, format="JPEG", quality=90)
                frame_paths.append(frame_path.relative_to(output_dir).as_posix())

            info["frame_times"] = realized_frame_times
            info["frame_paths"] = frame_paths
            info["segment_aliases"] = build_segment_aliases(video_name, seg_id)

    return segment_times_info


def persist_segment_video_assets(
    video_path: str,
    segment_times_info: Dict[str, Dict[str, Any]],
    output_dir: Path,
    clips_subdir: str,
) -> Dict[str, Dict[str, Any]]:
    """Persist per-segment clips and annotate metadata with relative paths."""
    clips_root = output_dir / clips_subdir
    clips_root.mkdir(parents=True, exist_ok=True)

    if not segment_times_info:
        return segment_times_info

    for raw_segment_id, info in segment_times_info.items():
        seg_id = str(raw_segment_id)
        timestamp = info.get("timestamp", (0.0, 0.0))
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            raise RuntimeError(
                f"Segment {seg_id} is missing a valid timestamp tuple; cannot persist planner-grounding clip assets."
            )
        start = max(0.0, float(timestamp[0] or 0.0))
        end = max(start, float(timestamp[1] or start))
        clip_path = clips_root / _clip_filename(seg_id)

        if not _is_materialized_media_file(clip_path):
            export_video_subclip(video_path=video_path, start=start, end=end, output_path=clip_path)
        if not _is_materialized_media_file(clip_path):
            raise RuntimeError(
                f"Segment {seg_id} clip export did not materialize a usable file at {clip_path} "
                f"for timestamp ({start}, {end})."
            )

        info["segment_video_path"] = clip_path.relative_to(output_dir).as_posix()

    return segment_times_info
