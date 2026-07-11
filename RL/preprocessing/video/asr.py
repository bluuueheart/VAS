"""
Automatic Speech Recognition Module
Extracts transcripts from video audio using Whisper.
"""
import ctypes
import ctypes.util
import inspect
import json
import logging
import os
import site
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from tqdm import tqdm
from faster_whisper import WhisperModel
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from config.device_config import detect_device, initialize_device, get_accelerator_snapshot
from config.settings import settings
from core.accelerator_queue import SharedAcceleratorQueueLock, resolve_local_accelerator_slot_key
from runtime.session import runtime_session


logger = logging.getLogger(__name__)


class ASRProcessor:
    """Handles automatic speech recognition for video segments."""

    TRANSCRIPT_CACHE_VERSION = 5
    _RUNTIME_CACHE: ClassVar[Dict[Tuple[str, str, str, int], WhisperModel]] = {}
    
    _CUDNN_CANDIDATES = {
        "cudnn_ops": [
            "libcudnn_ops.so.9.1.0",
            "libcudnn_ops.so.9.1",
            "libcudnn_ops.so.9",
            "libcudnn_ops.so",
        ],
        "cudnn_cnn": [
            "libcudnn_cnn.so.9.1.0",
            "libcudnn_cnn.so.9.1",
            "libcudnn_cnn.so.9",
            "libcudnn_cnn.so",
        ],
        "cudnn": [
            "libcudnn.so.9.1.0",
            "libcudnn.so.9.1",
            "libcudnn.so.9",
            "libcudnn.so",
        ],
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = None,  # None = auto-detect
        compute_type: str = "float16",
        cudnn_paths: Optional[List[str]] = None,
        cpu_chunk_seconds: int = 90,
        cpu_chunk_threshold_seconds: int = 150,
        cpu_threads: Optional[int] = None,
        accelerator_lock: Any = None,
    ):
        """
        Initialize ASR processor.
        
        Args:
            model_name: Whisper model name
            model_path: Local path to model (if available)
            device: Device to run on ('cuda', 'npu', 'cpu'). Auto-detect if None.
            compute_type: Computation precision
            cudnn_paths: Optional list of directories or files containing cuDNN libraries
            cpu_chunk_seconds: Chunk size used for long CPU ASR
            cpu_chunk_threshold_seconds: Minimum duration before chunked CPU ASR is enabled
            cpu_threads: CPU threads passed to faster-whisper when supported
        """
        # Auto-detect device if not specified
        if device is None:
            device_type, _ = detect_device()
            device = device_type
        requested_device = device
        
        # Validate device for Whisper compatibility
        # Faster-Whisper: Supports 'cuda' and 'cpu'. For NPU, we'll use 'cpu' with fallback.
        if device == "npu":
            logging.warning("Whisper (faster-whisper) does not natively support NPU. Using CPU instead.")
            device = "cpu"
        elif device not in ("cuda", "cpu"):
            logging.warning(f"Unknown device '{device}', falling back to CPU")
            device = "cpu"
        
        # Initialize device environment
        initialize_device(device if device != "cpu" else None)
        
        resolved_model_path = model_path or settings.asr_model_path
        resolved_model_name = model_name or resolved_model_path
        self.model_name = resolved_model_name
        self.model_path = resolved_model_path
        self.device = device
        self.requested_device = requested_device
        self.fallback_from_npu = requested_device == "npu" and device == "cpu"
        self.compute_type = compute_type
        self._model = None
        self.cudnn_paths = self._resolve_cudnn_paths(cudnn_paths)
        self._cudnn_loaded = False
        self.cpu_chunk_seconds = max(0, int(cpu_chunk_seconds or 0))
        self.cpu_chunk_threshold_seconds = max(0, int(cpu_chunk_threshold_seconds or 0))
        self.cpu_threads = max(1, int(cpu_threads or (os.cpu_count() or 1)))
        self._accelerator_lock = accelerator_lock
        if self._accelerator_lock is None:
            self._accelerator_lock = SharedAcceleratorQueueLock(
                name="asr_whisper",
                slot_key=resolve_local_accelerator_slot_key("cuda:0" if self.device == "cuda" else self.device),
                enabled=self.device == "cuda",
            )
    
    def _resolve_cudnn_paths(self, provided: Optional[List[str]]) -> List[str]:
        """Merge explicit paths with env var CUDNN_LIBRARY_PATHS and system paths."""
        merged: List[str] = []
        env_paths = os.getenv("CUDNN_LIBRARY_PATHS")
        if env_paths:
            merged.extend([p.strip() for p in env_paths.split(os.pathsep) if p.strip()])
        if provided:
            merged.extend(provided)
        
        # Auto-detect from python environment (conda/venv)
        merged.append(os.path.join(sys.prefix, "lib"))
        if sys.base_prefix != sys.prefix:
            merged.append(os.path.join(sys.base_prefix, "lib"))
            
        # Check site-packages for nvidia packages (pip installed)
        site_packages = []
        try:
            site_packages.extend(site.getsitepackages())
        except AttributeError:
            pass
        
        # Fallback for site-packages if getsitepackages not available
        site_packages.append(os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"))
            
        for sp in site_packages:
            # Common locations for nvidia-cudnn-* packages
            merged.append(os.path.join(sp, "nvidia", "cudnn", "lib"))
            merged.append(os.path.join(sp, "nvidia", "cudnn_cu12", "lib"))
            merged.append(os.path.join(sp, "nvidia", "cudnn_cu11", "lib"))
        
        seen = set()
        normalized: List[str] = []
        for path in merged:
            expanded = os.path.abspath(os.path.expanduser(path))
            if expanded not in seen and os.path.exists(expanded):
                normalized.append(expanded)
                seen.add(expanded)
        return normalized
    
    def _dlopen(self, lib_path: str):
        """Load shared library with RTLD_GLOBAL when available."""
        mode = getattr(ctypes, "RTLD_GLOBAL", None)
        if mode is not None:
            return ctypes.CDLL(lib_path, mode=mode)
        return ctypes.CDLL(lib_path)
    
    def _load_library_from_path(self, lib_candidates: List[str]) -> bool:
        """Attempt to load one cuDNN component from explicit paths."""
        for base_path in self.cudnn_paths:
            if os.path.isdir(base_path):
                for candidate in lib_candidates:
                    candidate_path = os.path.join(base_path, candidate)
                    if os.path.isfile(candidate_path):
                        try:
                            self._dlopen(candidate_path)
                            return True
                        except OSError as err:
                            logging.warning("Failed loading %s: %s", candidate_path, err)
            elif os.path.isfile(base_path):
                filename = os.path.basename(base_path)
                if filename in lib_candidates or filename.startswith("libcudnn"):
                    try:
                        self._dlopen(base_path)
                        return True
                    except OSError as err:
                        logging.warning("Failed loading %s: %s", base_path, err)
        return False
    
    def _preload_cudnn(self) -> bool:
        """Try to preload cuDNN libraries from the configured paths."""
        if self._cudnn_loaded:
            return True
        if self.device != "cuda" or not self.cudnn_paths:
            return False
        
        for logical_name, candidates in self._CUDNN_CANDIDATES.items():
            if not self._load_library_from_path(candidates):
                logging.warning(
                    "Unable to preload %s from configured cuDNN paths. "
                    "Will continue with system lookup.",
                    logical_name,
                )
                return False
        
        self._cudnn_loaded = True
        return True
    
    def _has_cuda_support(self) -> bool:
        """Best-effort check for CUDA availability before loading model."""
        if self.device != "cuda":
            return False
        
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
    
    def _has_cudnn_runtime(self) -> bool:
        """Check if cuDNN shared libraries are discoverable."""
        if self.device != "cuda":
            return False
        
        if self._preload_cudnn():
            return True
        
        required_libs = ("cudnn_ops", "cudnn_cnn", "cudnn")
        for lib in required_libs:
            if ctypes.util.find_library(lib) is None:
                return False
        return True

    def _runtime_cache_key(self) -> Tuple[str, str, str, int]:
        return (
            str(self.model_path or self.model_name),
            str(self.device),
            str(self.compute_type),
            int(self.cpu_threads),
        )

    def _build_model_kwargs(self) -> Dict[str, Any]:
        model_kwargs: Dict[str, Any] = {}
        if self.device == "cpu":
            try:
                whisper_signature = inspect.signature(WhisperModel.__init__)
            except Exception:
                whisper_signature = None
            if whisper_signature is not None:
                if "cpu_threads" in whisper_signature.parameters:
                    model_kwargs["cpu_threads"] = self.cpu_threads
                if "num_workers" in whisper_signature.parameters:
                    model_kwargs["num_workers"] = 1
        return model_kwargs

    def _normalize_runtime_settings(self) -> None:
        if self.device in ("cpu", "npu") and self.compute_type in ("float16", "int8_float16"):
            logging.warning(
                "Compute type %s not supported on %s. Falling back to int8 for Whisper.",
                self.compute_type,
                self.device
            )
            self.compute_type = "int8"
        if self.device == "cpu" and self.fallback_from_npu and self.compute_type == "float32":
            logging.info("NPU fallback detected. Switching Whisper compute_type from float32 to int8 on CPU.")
            self.compute_type = "int8"
        if self.device == "cuda":
            if not self._has_cuda_support():
                logging.warning("CUDA not available in environment. Falling back to CPU for Whisper.")
                self.device = "cpu"
                self.compute_type = "int8"
            elif not self._has_cudnn_runtime():
                logging.warning(
                    "cuDNN runtime libraries not found (need libcudnn_ops / libcudnn_cnn). "
                    "Falling back to CPU to avoid crash."
                )
                self.device = "cpu"
                self.compute_type = "int8"

    def _try_reuse_cached_runtime(self) -> bool:
        cache_key = self._runtime_cache_key()
        cached_model = self._RUNTIME_CACHE.get(cache_key)
        runtime_session.record_cache("asr_runtime", cached_model is not None)
        if cached_model is None:
            return False
        self._model = cached_model
        logging.info(
            "Reusing cached Whisper runtime '%s' on device=%s compute_type=%s",
            self.model_path or self.model_name,
            self.device,
            self.compute_type,
        )
        return True
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is not None:
            return

        self._normalize_runtime_settings()
        if self._try_reuse_cached_runtime():
            return

        model_source = self.model_path or self.model_name
        logging.info(
            "Loading Whisper model '%s' on device=%s compute_type=%s",
            model_source,
            self.device,
            self.compute_type,
        )
        lock_context = (
            self._accelerator_lock
            if self.device == "cuda" and self._accelerator_lock is not None
            else nullcontext()
        )
        with lock_context:
            with runtime_session.time_block("runtime_load.asr_seconds"):
                model_kwargs = self._build_model_kwargs()
                try:
                    self._model = WhisperModel(
                        model_source,
                        device=self.device,
                        compute_type=self.compute_type,
                        **model_kwargs,
                    )
                except (RuntimeError, ValueError) as err:
                    if "cudnn" in str(err).lower() or "cuda" in str(err).lower():
                        logging.warning(
                            "CUDA/cuDNN unavailable (%s). Falling back to CPU with int8 for Whisper.", err
                        )
                        self.device = "cpu"
                        self.compute_type = "int8"
                        if self._try_reuse_cached_runtime():
                            return
                        self._model = WhisperModel(
                            model_source,
                            device=self.device,
                            compute_type=self.compute_type,
                            **self._build_model_kwargs(),
                        )
                    elif "float16" in str(err).lower():
                        logging.warning(
                            "Float16 unsupported for device '%s' (%s). Falling back to int8.",
                            self.device,
                            err
                        )
                        self.compute_type = "int8"
                        if self._try_reuse_cached_runtime():
                            return
                        self._model = WhisperModel(
                            model_source,
                            device=self.device,
                            compute_type=self.compute_type,
                            **self._build_model_kwargs(),
                        )
                    else:
                        raise
        if self._model is not None:
            self._RUNTIME_CACHE[self._runtime_cache_key()] = self._model
            self._model.logger.setLevel(logging.WARNING)
            logging.info("ASR accelerator snapshot: %s", get_accelerator_snapshot(self.device))

    def _extract_full_audio(self, video_path: str, audio_path: Path) -> None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with VideoFileClip(video_path) as video:
            if video.audio is None:
                raise ValueError(f"No audio track found in video: {video_path}")
            self._write_audio_clip(video.audio, audio_path)

    def _write_audio_clip(self, audio_clip, output_path: Path) -> None:
        audio_codec = output_path.suffix.lstrip(".") or "mp3"
        write_kwargs = {
            "codec": "pcm_s16le" if audio_codec == "wav" else audio_codec,
            "logger": None,
        }
        try:
            write_sig = inspect.signature(audio_clip.write_audiofile)
            if "verbose" in write_sig.parameters:
                write_kwargs["verbose"] = False
        except Exception:
            pass
        audio_clip.write_audiofile(str(output_path), **write_kwargs)

    def _export_audio_chunk(
        self,
        audio_path: Path,
        start: float,
        end: float,
        output_path: Path,
    ) -> None:
        """Export one chunk from the full audio file using a fresh clip handle.

        MoviePy subclips can share the same reader process as the parent clip.
        Closing a derived subclip may invalidate the parent reader and break the
        next chunk export with errors such as `'NoneType' object has no attribute
        'stdout'`. Re-opening the source audio for each chunk avoids that shared
        reader lifetime bug and keeps long-video ASR deterministic.
        """
        with AudioFileClip(str(audio_path)) as audio:
            if hasattr(audio, "subclip"):
                chunk_audio = audio.subclip(start, end)
            elif hasattr(audio, "subclipped"):
                chunk_audio = audio.subclipped(start, end)
            else:
                raise AttributeError("Audio clip object has no subclip/subclipped method")
            try:
                self._write_audio_clip(chunk_audio, output_path)
            finally:
                try:
                    chunk_audio.close()
                except Exception:
                    pass

    def _build_transcribe_kwargs(self) -> Dict[str, Any]:
        transcribe_kwargs: Dict[str, Any] = {"language": "en"}
        if self.device == "cpu":
            transcribe_kwargs.update(
                {
                    "beam_size": 1,
                    "best_of": 1,
                    "condition_on_previous_text": False,
                }
            )
        return transcribe_kwargs

    def _transcribe_audio_file(self, audio_path: Path, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        lock_context = (
            self._accelerator_lock
            if self.device == "cuda" and self._accelerator_lock is not None
            else nullcontext()
        )
        with lock_context:
            segments, _ = self._model.transcribe(str(audio_path), **self._build_transcribe_kwargs())
        return self._collect_transcript_entries(segments, time_offset=time_offset)

    def _transcribe_video_in_cpu_chunks(self, audio_path: Path) -> List[Dict[str, Any]]:
        with AudioFileClip(str(audio_path)) as audio:
            duration = float(audio.duration or 0.0)
            if (
                self.device != "cpu"
                or self.cpu_chunk_seconds <= 0
                or self.cpu_chunk_threshold_seconds <= 0
                or duration <= float(self.cpu_chunk_threshold_seconds)
            ):
                return self._transcribe_audio_file(audio_path)

            chunk_dir = audio_path.parent / "_tmp_asr_chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            entries: List[Dict[str, Any]] = []
            chunk_start = 0.0
            chunk_index = 0
            total_chunks = max(1, int((duration + float(self.cpu_chunk_seconds) - 1e-6) // float(self.cpu_chunk_seconds)))
            logging.info(
                "CPU Whisper fallback enabled for long audio (duration=%.1fs). "
                "Transcribing in %d chunks of up to %ds.",
                duration,
                total_chunks,
                self.cpu_chunk_seconds,
            )

            while chunk_start < duration - 1e-3:
                chunk_end = min(duration, chunk_start + float(self.cpu_chunk_seconds))
                if duration - chunk_end < min(10.0, float(self.cpu_chunk_seconds) / 3.0):
                    chunk_end = duration
                logging.info(
                    "ASR CPU chunk %d/%d: %.1fs -> %.1fs",
                    chunk_index + 1,
                    total_chunks,
                    chunk_start,
                    chunk_end,
                )

                chunk_path = chunk_dir / f"{audio_path.stem}.chunk{chunk_index:03d}.wav"
                try:
                    self._export_audio_chunk(audio_path, chunk_start, chunk_end, chunk_path)
                    entries.extend(self._transcribe_audio_file(chunk_path, time_offset=chunk_start))
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to transcribe ASR chunk {chunk_index + 1}/{total_chunks} "
                        f"({chunk_start:.1f}s -> {chunk_end:.1f}s): {exc}"
                    ) from exc
                finally:
                    try:
                        if chunk_path.exists():
                            chunk_path.unlink()
                    except Exception:
                        pass

                chunk_index += 1
                chunk_start = chunk_end

            try:
                chunk_dir.rmdir()
            except Exception:
                pass
            return entries

    def _collect_transcript_entries(self, segments, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        try:
            for segment in list(segments):
                start = max(0.0, float(segment.start) + time_offset)
                end = max(start, float(segment.end) + time_offset)
                text = str(getattr(segment, "text", "") or "").strip()
                if not text and end - start <= 1e-3:
                    continue
                entries.append(
                    {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": text,
                    }
                )
        except Exception as e:
            logger.warning(f"Error collecting transcript entries: {e}")
            return []
        return entries

    def _format_transcript_entries(self, entries: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for entry in entries:
            try:
                start = max(0.0, float(entry.get("start", 0.0)))
                end = max(start, float(entry.get("end", start)))
            except Exception:
                continue
            text = str(entry.get("text", "") or "").strip()
            if not text:
                continue
            lines.append(f"[{start:.2f}s -> {end:.2f}s] {text}")
        return "\n".join(lines) + ("\n" if lines else "")

    def _clip_entries_to_window(
        self,
        entries: List[Dict[str, Any]],
        window: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            return []
        try:
            window_start = float(window[0])
            window_end = float(window[1])
        except Exception:
            return []

        clipped_entries: List[Dict[str, Any]] = []
        for entry in entries or []:
            try:
                entry_start = float(entry.get("start", 0.0) or 0.0)
                entry_end = float(entry.get("end", entry_start) or entry_start)
            except Exception:
                continue

            overlap = self._range_overlap_duration((entry_start, entry_end), (window_start, window_end))
            if overlap <= 1e-3:
                continue

            clipped_start = max(entry_start, window_start)
            clipped_end = min(entry_end, window_end)
            if clipped_end - clipped_start <= 1e-3:
                continue

            text = str(entry.get("text", "") or "").strip()
            if not text:
                continue

            clipped_entries.append(
                {
                    "start": round(clipped_start, 3),
                    "end": round(clipped_end, 3),
                    "text": text,
                }
            )
        return clipped_entries

    def _ranges_overlap(self, left: Any, right: Any) -> bool:
        if not isinstance(left, (list, tuple)) or len(left) != 2:
            return False
        if not isinstance(right, (list, tuple)) or len(right) != 2:
            return False
        try:
            left_start = float(left[0])
            left_end = float(left[1])
            right_start = float(right[0])
            right_end = float(right[1])
        except Exception:
            return False
        return min(left_end, right_end) > max(left_start, right_start)

    def _range_overlap_duration(self, left: Any, right: Any) -> float:
        if not self._ranges_overlap(left, right):
            return 0.0
        try:
            left_start = float(left[0])
            left_end = float(left[1])
            right_start = float(right[0])
            right_end = float(right[1])
        except Exception:
            return 0.0
        return max(0.0, min(left_end, right_end) - max(left_start, right_start))

    def _load_cached_full_transcript(self, cache_file: Path) -> Optional[List[Dict[str, Any]]]:
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if int(payload.get("cache_version", 0) or 0) != self.TRANSCRIPT_CACHE_VERSION:
                return None
            entries = payload.get("segments")
            if not isinstance(entries, list):
                return None
            normalized: List[Dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                normalized.append(
                    {
                        "start": round(float(entry.get("start", 0.0)), 3),
                        "end": round(float(entry.get("end", entry.get("start", 0.0))), 3),
                        "text": str(entry.get("text", "") or "").strip(),
                    }
                )
            return normalized
        except Exception:
            logging.warning("Failed to read cached full-video ASR from %s", cache_file)
            return None

    def _save_cached_full_transcript(self, cache_file: Path, segments: List[Dict[str, Any]]) -> None:
        payload = {
            "cache_version": self.TRANSCRIPT_CACHE_VERSION,
            "segments": segments,
        }
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("Failed to write full-video ASR cache to %s: %s", cache_file, exc)

    def transcribe_video(
        self,
        video_path: str,
        working_dir: str,
        video_name: Optional[str] = None,
        audio_format: str = "mp3",
        reuse_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Transcribe the full video once and reuse timestamps for semantic segmentation."""
        resolved_video_name = video_name or Path(video_path).stem
        cache_dir = Path(working_dir) / "_cache" / resolved_video_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "full_transcript_segments.json"
        if reuse_cache:
            cached_segments = self._load_cached_full_transcript(cache_file)
            if cached_segments is not None:
                return cached_segments

        audio_file = cache_dir / f"{resolved_video_name}.full.{audio_format}"
        if not reuse_cache or not audio_file.exists() or audio_file.stat().st_size < 1000:
            self._extract_full_audio(video_path, audio_file)

        self._load_model()
        entries = self._transcribe_video_in_cpu_chunks(audio_file)
        self._save_cached_full_transcript(cache_file, entries)
        return entries

    def build_segment_transcripts(
        self,
        segment_times_info: Dict[str, Dict],
        transcript_entries: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Project global ASR entries into every overlapping segment window.

        We clip timestamps to the segment window so the stored transcript never
        advertises times outside the segment's own `time` field.
        """
        segment_windows: Dict[str, Any] = {}
        transcripts: Dict[str, str] = {}
        grouped_entries: Dict[str, List[Dict[str, Any]]] = {}

        for segment_id, info in segment_times_info.items():
            timestamp = info.get("timestamp", (0, 0))
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                transcripts[str(segment_id)] = ""
                continue
            try:
                start = float(timestamp[0])
                end = float(timestamp[1])
            except Exception:
                transcripts[str(segment_id)] = ""
                continue
            normalized_window = (start, end)
            segment_windows[str(segment_id)] = normalized_window
            grouped_entries[str(segment_id)] = []
            transcripts[str(segment_id)] = ""

        sorted_entries = sorted(
            transcript_entries or [],
            key=lambda item: (
                float(item.get("start", 0.0) or 0.0),
                float(item.get("end", item.get("start", 0.0)) or 0.0),
            ),
        )

        for segment_id, entries in grouped_entries.items():
            clipped_entries = self._clip_entries_to_window(sorted_entries, segment_windows[segment_id])
            grouped_entries[segment_id] = clipped_entries
            transcripts[segment_id] = self._format_transcript_entries(clipped_entries)
        return transcripts
    
    def transcribe_segments(
        self,
        video_name: str,
        working_dir: str,
        segment_index2name: Dict[str, str],
        audio_format: str = 'mp3',
        segment_times_info: Optional[Dict[str, Dict]] = None,
        reuse_cache: bool = True,
    ) -> Dict[str, str]:
        """
        Transcribe all video segments.
        
        Args:
            video_name: Name of the video
            working_dir: Working directory containing cache
            segment_index2name: Mapping from index to segment name
            audio_format: Audio file format
            
        Returns:
            Dictionary mapping segment index to transcript
        """
        cache_dir = Path(working_dir) / '_cache' / video_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "transcripts.json"
        cached_transcripts = (
            self._load_cached_transcripts(cache_file, segment_index2name=segment_index2name)
            if reuse_cache
            else {}
        )
        
        transcripts = {}
        missing_indices = []
        
        for index in segment_index2name:
            if cached_transcripts and index in cached_transcripts:
                transcripts[index] = cached_transcripts[index]
            else:
                missing_indices.append(index)
        
        if not missing_indices:
            return transcripts
        
        self._load_model()
        
        for index in tqdm(missing_indices, desc=f"Transcribing {video_name}"):
            segment_name = segment_index2name[index]
            audio_file = cache_dir / f"{segment_name}.{audio_format}"
            
            # Skip if audio file doesn't exist
            if not audio_file.exists():
                logger.debug(f"Audio file not found for segment {index}: {audio_file}")
                transcripts[index] = ""
                continue
            
            # Check if audio file is valid
            if audio_file.stat().st_size < 1000:  # Less than 1KB
                logger.debug(f"Audio file too small (likely empty) for segment {index}")
                transcripts[index] = ""
                continue
            
            try:
                time_offset = 0.0
                segment_window = None
                if segment_times_info and index in segment_times_info:
                    timestamp = segment_times_info[index].get("timestamp", (0, 0))
                    if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                        try:
                            time_offset = float(timestamp[0])
                            segment_window = (float(timestamp[0]), float(timestamp[1]))
                        except (TypeError, ValueError):
                            time_offset = 0.0
                            segment_window = None
                entries = self._transcribe_audio_file(audio_file, time_offset=time_offset)
                if segment_window is not None:
                    entries = self._clip_entries_to_window(entries, segment_window)
                result = self._format_transcript_entries(entries)
                transcripts[index] = result
            except Exception as e:
                logger.warning(f"Transcription failed for segment {index} ({audio_file}): {e}")
                transcripts[index] = ""
        
        self._save_transcripts_cache(cache_file, transcripts, segment_index2name=segment_index2name)
        return transcripts

    def _load_cached_transcripts(
        self,
        cache_file: Path,
        segment_index2name: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        if not cache_file.exists():
            return {}
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {}
                if "transcripts" in data:
                    if int(data.get("cache_version", 0) or 0) != self.TRANSCRIPT_CACHE_VERSION:
                        return {}
                    transcripts = data.get("transcripts", {})
                    cached_names = data.get("segment_index2name", {})
                    if (
                        isinstance(segment_index2name, dict)
                        and isinstance(cached_names, dict)
                        and any(str(cached_names.get(str(idx), "")) != str(name) for idx, name in segment_index2name.items())
                    ):
                        return {}
                    if isinstance(transcripts, dict):
                        return {str(k): str(v) for k, v in transcripts.items()}
                    return {}
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            logging.warning("Failed to read cached transcripts from %s", cache_file)
        return {}

    def _save_transcripts_cache(
        self,
        cache_file: Path,
        transcripts: Dict[str, str],
        segment_index2name: Optional[Dict[str, str]] = None,
    ) -> None:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                payload = {
                    "cache_version": self.TRANSCRIPT_CACHE_VERSION,
                    "segment_index2name": {str(k): str(v) for k, v in (segment_index2name or {}).items()},
                    "transcripts": {str(k): str(v) for k, v in transcripts.items()},
                }
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("Failed to write transcripts cache to %s: %s", cache_file, exc)
    
    def _format_transcript(self, segments, time_offset: float = 0.0) -> str:
        """
        Format Whisper segments into readable transcript.
        
        Args:
            segments: Whisper segment iterator (generator)
            
        Returns:
            Formatted transcript string
        """
        result = ""
        try:
            entries = self._collect_transcript_entries(segments, time_offset=time_offset)
            result = self._format_transcript_entries(entries)
        except Exception as e:
            logger.warning(f"Error formatting transcript: {e}")
            return ""
        return result
    
    def transcribe_single(self, audio_path: str) -> str:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Formatted transcript
        """
        self._load_model()
        
        try:
            return self._format_transcript_entries(self._transcribe_audio_file(Path(audio_path)))
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return ""
