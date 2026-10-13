"""
Video Preprocessing Pipeline
Main orchestrator for video data ingestion and processing.
"""
import asyncio
import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from config.settings import settings
from core.llm import get_llm_provider
from .video.segmentation import VideoSegmenter
from .video.asr import ASRProcessor
from .video.caption import VisualProcessor, create_visual_processor
from .video.frame_assets import (
    build_segment_aliases,
    persist_segment_frame_assets,
    persist_segment_video_assets,
)
from .knowledge_graph import KnowledgeGraphBuilder
from .storage.vector_store import VideoSegmentVectorStore

logger = logging.getLogger(__name__)


_SUBTITLE_REPEAT_COVERAGE_THRESHOLD = 0.8
_SUBTITLE_MIN_DENSITY_CHARS_PER_SECOND = 2.0


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _normalize_subtitle_text(text: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    normalized = re.sub(r"^[^\w]+|[^\w]+$", "", normalized, flags=re.UNICODE)
    return normalized


def _subtitle_char_count(text: str) -> int:
    return len(re.sub(r"[^\w]+", "", str(text or ""), flags=re.UNICODE))


def _format_segment_time_value(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    return f"{numeric:.2f}"


def _parse_time_range_bounds(value: Any) -> Optional[tuple[float, float]]:
    text = str(value or "").strip()
    if "-" not in text:
        return None
    left, right = text.split("-", 1)
    try:
        start = float(left.strip())
        end = float(right.strip())
    except Exception:
        return None
    return (min(start, end), max(start, end))


def _extract_transcript_bounds(transcript: Any) -> Optional[tuple[float, float]]:
    matches = re.findall(r"\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]", str(transcript or ""))
    if not matches:
        return None
    try:
        start = float(matches[0][0])
        end = float(matches[-1][1])
    except Exception:
        return None
    return (min(start, end), max(start, end))


def _segments_look_stale(segments: Any) -> bool:
    if not isinstance(segments, dict) or not segments:
        return True

    transcript_zero_time_mismatch = 0
    for seg_id, seg in segments.items():
        if not isinstance(seg, dict):
            return True

        caption = str(seg.get("caption", "") or "").strip().lower()
        transcript = str(seg.get("transcript", "") or "")
        time_range = str(seg.get("time", "0-0") or "0-0")
        segment_bounds = _parse_time_range_bounds(time_range)
        transcript_bounds = _extract_transcript_bounds(transcript)

        if "caption" not in seg or "transcript" not in seg:
            return True
        if _is_low_information_caption_text(caption):
            return True
        if "transcript evidence:" in caption:
            return True
        if seg_id != "0" and time_range and "-" in time_range and transcript.startswith("[0.00s ->"):
            transcript_zero_time_mismatch += 1
        if segment_bounds and transcript_bounds:
            seg_start, seg_end = segment_bounds
            transcript_start, transcript_end = transcript_bounds
            if transcript_start < seg_start - 0.05 or transcript_end > seg_end + 0.05:
                return True

    return transcript_zero_time_mismatch > 0


def _vector_index_cache_is_complete(vectors_dir: Path, segment_count: int) -> bool:
    """Return whether a cached dense index covers every processed segment.

    Older runs could persist ``vectors/metadata.json`` with an empty metadata
    list when local vector indexing was disabled.  That file is structurally
    valid, so ``VideoSegmentVectorStore.load`` accepts it, but semantic
    retrieval then silently returns no results for every query.  Treat an
    empty or partial index as an invalid cache and rebuild it while the
    already persisted captions/frames remain reusable.
    """
    metadata_path = vectors_dir / "metadata.json"
    if not metadata_path.exists() or segment_count <= 0:
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except Exception as exc:
        logger.warning("Failed to inspect cached vector metadata at %s: %s", metadata_path, exc)
        return False

    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, list) or len(metadata) != int(segment_count):
        return False
    for item in metadata:
        if not isinstance(item, dict):
            return False
        indexed_modalities = item.get("indexed_modalities") or []
        if not isinstance(indexed_modalities, list) or not indexed_modalities:
            return False
        if os.environ.get('TOOLSTAR_SEMANTIC_INDEX_VIDEO', '1').strip().lower() not in {'0', 'false', 'no', 'off'} and 'video' not in indexed_modalities:
            return False
    return True


def _is_placeholder_caption_text(text: Any) -> bool:
    normalized = " ".join(str(text or "").split()).strip().lower()
    return normalized.startswith("visual caption unavailable")


def _is_low_information_caption_text(text: Any) -> bool:
    normalized = " ".join(str(text or "").split()).strip().lower()
    if not normalized:
        return True
    if _is_placeholder_caption_text(normalized):
        return True
    content_chars = re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)
    if not content_chars:
        return True
    punctuation_count = len(re.findall(r"[^\w\s]", normalized, flags=re.UNICODE))
    if re.search(r"([^\w\s])\1{7,}", normalized, flags=re.UNICODE):
        return True
    if punctuation_count >= 12 and punctuation_count * 2 >= len(normalized):
        return True
    return len(content_chars) <= 1 and len(normalized) <= 3


def is_effective_subtitle(asr_segments: Optional[List[Dict[str, Any]]]) -> tuple[bool, str, Dict[str, Any]]:
    normalized_entries: List[Dict[str, Any]] = []
    repeated_durations: Dict[str, float] = {}
    covered_duration = 0.0
    char_count = 0

    for segment in asr_segments or []:
        if not isinstance(segment, dict):
            continue
        normalized_text = _normalize_subtitle_text(segment.get("text", ""))
        if not normalized_text:
            continue
        try:
            start = max(0.0, float(segment.get("start", 0.0) or 0.0))
            end = max(start, float(segment.get("end", start) or start))
        except Exception:
            continue
        duration = max(0.0, end - start)
        if duration <= 1e-3:
            continue
        normalized_entries.append(
            {
                "text": normalized_text,
                "duration": duration,
            }
        )
        covered_duration += duration
        char_count += _subtitle_char_count(normalized_text)
        repeated_durations[normalized_text] = repeated_durations.get(normalized_text, 0.0) + duration

    stats = {
        "non_empty_segments": len(normalized_entries),
        "unique_texts": len(repeated_durations),
        "covered_duration": round(covered_duration, 3),
        "char_count": char_count,
        "density_chars_per_second": round(char_count / max(covered_duration, 1e-6), 3) if covered_duration > 0 else 0.0,
        "top_repeat_coverage_ratio": round(
            (max(repeated_durations.values()) / max(covered_duration, 1e-6)) if repeated_durations and covered_duration > 0 else 0.0,
            3,
        ),
    }

    if not normalized_entries or char_count <= 0:
        return False, "empty", stats

    duplicate = (
        (len(normalized_entries) >= 2 and len(repeated_durations) == 1)
        or (
            len(normalized_entries) >= 2
            and covered_duration > 0
            and (max(repeated_durations.values()) / covered_duration) >= _SUBTITLE_REPEAT_COVERAGE_THRESHOLD
        )
    )
    low_density = covered_duration > 0 and (char_count / covered_duration) < _SUBTITLE_MIN_DENSITY_CHARS_PER_SECOND

    if duplicate and low_density:
        return False, "duplicate_low_density", stats
    if duplicate:
        return False, "duplicate", stats
    if low_density:
        return False, "low_density", stats
    return True, "valid", stats


@dataclass
class PipelineConfig:
    """Configuration for preprocessing pipeline."""
    
    # Device configuration (auto-detect if None)
    device: Optional[str] = None  # None = auto-detect 'cuda'/'npu'/'cpu'
    
    # Video segmentation
    segment_length: int = 30
    frame_interval: float = 6.0  # Time interval in seconds between sampled frames (5-10s per PRD)
    use_scene_detection: bool = True
    use_asr_semantic_segmentation: bool = True
    # A benchmark may exclude transcript text while retaining the source
    # video's audio track. In that mode ASR is disabled for segmentation and
    # planner-facing transcript evidence.
    include_audio_transcript: bool = True
    min_segment_duration: int = 15
    max_segment_duration: int = 45
    hard_max_segment_duration: int = 60
    
    # ASR
    whisper_model: str = field(default_factory=lambda: settings.asr_model_path)
    whisper_model_path: Optional[str] = None  # Will use settings.asr_model_path if None
    whisper_compute_type: str = "float16"
    whisper_cudnn_paths: Optional[List[str]] = field(default_factory=list)
    whisper_cpu_chunk_seconds: int = field(default_factory=lambda: max(0, int(settings.asr_cpu_chunk_seconds)))
    whisper_cpu_chunk_threshold_seconds: int = field(default_factory=lambda: max(0, int(settings.asr_cpu_chunk_threshold_seconds)))
    whisper_cpu_threads: int = field(default_factory=lambda: max(1, int(settings.asr_cpu_threads)))
    asr_device: Optional[str] = field(
        default_factory=lambda: str(
            os.environ.get("TOOLSTAR_ASR_DEVICE")
            or settings.preprocess_asr_device
            or "cpu"
        )
    )
    
    # Visual processing (Frame Captions)
    vision_model: str = field(default_factory=lambda: settings.qwen_vl_model_path)
    vision_model_path: Optional[str] = None  # Will auto-select settings.ocr_model_path or settings.qwen_vl_model_path
    
    # Vector store (Embeddings)
    embedding_model_path: str = ""  # Will use settings.qwen3_vl_embedding_model_path if empty
    embedding_device: Optional[str] = None
    embedding_dim: int = 1152
    build_vector_index: bool = True
    # Sparse videos retain real captions/frames but do not need a dense query
    # index because the planner exposes every usable segment directly.
    build_sparse_vector_index: bool = False
    
    # Knowledge graph
    enable_knowledge_graph: bool = True
    max_entity_extraction_rounds: int = 1
    
    # Metadata synthesis (deprecated: directory-view metadata removed)
    enable_metadata_synthesis: bool = False
    
    # Output
    working_dir: str = "./videorag_cache"
    save_intermediate: bool = True
    reuse_cache: bool = True


class VideoPreprocessingPipeline:
    """Main pipeline for video preprocessing."""
    
    def __init__(
        self,
        config: PipelineConfig,
        llm_provider=None,
        accelerator_lock: Any = None,
        visual_llm_provider: Any = None,
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Pipeline configuration
            llm_provider: LLM provider for entity extraction and metadata synthesis
            visual_llm_provider: Optional already-loaded multimodal provider to
                share between captioning and planner inference
        """
        self.config = config
        # Default预处理LLM使用独立模型（无帧输入），可外部覆盖
        self.llm_provider = llm_provider
        self.accelerator_lock = accelerator_lock
        
        # Initialize components
        self.video_segmenter = VideoSegmenter(
            segment_length=config.segment_length,
            frame_interval=config.frame_interval,
            min_segment_duration=config.min_segment_duration,
            max_segment_duration=config.max_segment_duration,
            hard_max_segment_duration=config.hard_max_segment_duration,
        )
        
        self.asr_processor = ASRProcessor(
            model_name=config.whisper_model,
            model_path=config.whisper_model_path or settings.asr_model_path,
            device=config.asr_device or config.device,
            compute_type=config.whisper_compute_type,
            cudnn_paths=config.whisper_cudnn_paths,
            cpu_chunk_seconds=config.whisper_cpu_chunk_seconds,
            cpu_chunk_threshold_seconds=config.whisper_cpu_chunk_threshold_seconds,
            cpu_threads=config.whisper_cpu_threads,
            accelerator_lock=accelerator_lock,
        )
        
        default_vision_path = (
            settings.ocr_model_path
            if "deepseek" in (config.vision_model or "").lower() and "ocr" in (config.vision_model or "").lower()
            else settings.qwen_vl_model_path
        )

        resolved_vision_path = config.vision_model_path or default_vision_path
        self.visual_processor = None
        if (
            visual_llm_provider is not None
            and getattr(visual_llm_provider, "model", None) is not None
            and getattr(visual_llm_provider, "processor", None) is not None
            and bool(getattr(visual_llm_provider, "_supports_multimodal_generation", False))
        ):
            try:
                self.visual_processor = VisualProcessor.from_loaded_llm_provider(
                    visual_llm_provider,
                    model_name=config.vision_model,
                    model_path=resolved_vision_path,
                )
                logger.info(
                    "Sharing the already-loaded planner multimodal runtime with captioning: model=%s provider_role=%s.",
                    resolved_vision_path,
                    str(getattr(visual_llm_provider, "model_role", "") or "").strip().lower() or "unknown",
                )
            except Exception as exc:
                logger.warning(
                    "Could not build a shared caption view over the planner runtime; falling back to the caption runtime cache: %s",
                    exc,
                )
        if self.visual_processor is None:
            self.visual_processor = create_visual_processor(
                model_name=config.vision_model,
                model_path=resolved_vision_path,
                device=config.device,
                accelerator_lock=accelerator_lock,
            )
        
        self.vector_store = VideoSegmentVectorStore(
            embedding_dim=config.embedding_dim,
            embedding_model_path=config.embedding_model_path or settings.qwen3_vl_embedding_model_path,
            device=config.embedding_device if config.embedding_device is not None else config.device,
            accelerator_lock=accelerator_lock,
        )
        
        if config.enable_knowledge_graph and self.llm_provider is not None:
            self.kg_builder = KnowledgeGraphBuilder(llm_provider=self.llm_provider)
        else:
            self.kg_builder = None
        
        self.metadata_synthesizer = None

    def _ensure_llm_provider(self):
        """Load the preprocess LLM on demand to avoid overlapping caption/KG VRAM use."""
        if self.llm_provider is None:
            logger.info("Initializing preprocess LLM provider on demand...")
            self.llm_provider = get_llm_provider(
                model_name=settings.preprocess_model_path,
                model_role="preprocess",
            )
        return self.llm_provider

    def _ensure_kg_builder(self):
        """Create the KG builder lazily after captioning finishes."""
        if not self.config.enable_knowledge_graph:
            return None
        if self.kg_builder is None:
            self.kg_builder = KnowledgeGraphBuilder(
                llm_provider=self._ensure_llm_provider()
            )
        return self.kg_builder

    def _segment_strategy_label(self, strategy_name: Optional[str] = None) -> str:
        normalized = str(strategy_name or "").strip()
        if normalized:
            return normalized
        if self.config.use_scene_detection:
            return "scene_v2"
        return "fixed_v1"
    
    async def process_video(
        self,
        video_path: str,
        video_name: Optional[str] = None
    ) -> Dict:
        """
        Process a single video through the pipeline.
        
        Args:
            video_path: Path to video file
            video_name: Optional video name (defaults to filename)
            
        Returns:
            Processing results including compact segment data and indexing artifacts
        """
        if video_name is None:
            video_name = Path(video_path).stem
        
        logger.info(f"Processing video: {video_name}")

        output_dir = Path(self.config.working_dir) / "processed" / video_name
        if self.config.reuse_cache:
            cached = self._try_load_processed(output_dir, video_name)
            if cached is not None:
                logger.info(f"Loaded processed outputs from cache for {video_name}")
                return cached
        
        # Step 1: ASR-guided segmentation
        logger.info("Step 1: Building ASR-guided segments...")
        global_asr_segments: List[Dict[str, Any]] = []
        subtitle_valid = False
        subtitle_validity_reason = "not_attempted"
        subtitle_validity_stats: Dict[str, Any] = {}
        semantic_asr_segments: List[Dict[str, Any]] = []
        segment_strategy = self._segment_strategy_label(
            "scene_v2" if self.config.use_scene_detection else "fixed_v1"
        )
        cached_segment_metadata: Optional[Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]] = None
        reuse_cached_segment_metadata = (
            self.config.include_audio_transcript
            and
            self.config.reuse_cache
            and str(os.environ.get("TOOLSTAR_REUSE_SEGMENT_METADATA", "")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if reuse_cached_segment_metadata:
            cache_path = Path(self.config.working_dir) / "_cache" / video_name
            try:
                cached_segment_metadata = self.video_segmenter._try_load_cached_segments(
                    cache_path,
                    expected_strategy="asr_semantic_v4",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to inspect cached segment metadata for %s; falling back to ASR: %s",
                    video_name,
                    exc,
                )
                cached_segment_metadata = None

        if cached_segment_metadata is not None:
            segment_index2name, segment_times_info = cached_segment_metadata
            segment_strategy = self._segment_strategy_label("asr_semantic_v4")
            logger.info(
                "Reusing cached segment metadata for %s: segments=%s; skipping full-video ASR.",
                video_name,
                len(segment_index2name),
            )
        elif self.config.use_asr_semantic_segmentation and self.config.include_audio_transcript:
            try:
                global_asr_segments = self.asr_processor.transcribe_video(
                    video_path=video_path,
                    working_dir=self.config.working_dir,
                    video_name=video_name,
                    reuse_cache=self.config.reuse_cache,
                )
                logger.info("Loaded %d full-video ASR timestamp units for semantic segmentation.", len(global_asr_segments))
                subtitle_valid, subtitle_validity_reason, subtitle_validity_stats = is_effective_subtitle(global_asr_segments)
                logger.info(
                    "Subtitle validity for %s: valid=%s reason=%s stats=%s",
                    video_name,
                    subtitle_valid,
                    subtitle_validity_reason,
                    json.dumps(subtitle_validity_stats, ensure_ascii=False),
                )
                if subtitle_valid:
                    semantic_asr_segments = global_asr_segments
                    segment_strategy = self._segment_strategy_label("asr_semantic_v4")
                else:
                    logger.warning(
                        "Detected ineffective subtitles for %s; switching segmentation to %s.",
                        video_name,
                        self._segment_strategy_label("scene_v2" if self.config.use_scene_detection else "fixed_v1"),
                    )
            except Exception as exc:
                logger.warning("Full-video ASR semantic segmentation fallback activated: %s", exc)
                global_asr_segments = []
                subtitle_validity_reason = "asr_failed"
                subtitle_validity_stats = {"error": str(exc)}

        if cached_segment_metadata is None:
            segment_index2name, segment_times_info = self.video_segmenter.segment_video(
                video_path,
                self.config.working_dir,
                use_scene_detection=self.config.use_scene_detection,
                asr_segments=semantic_asr_segments or None,
                reuse_cache=self.config.reuse_cache,
                extract_audio=bool(self.config.include_audio_transcript),
            )

        # Step 2: Attach per-segment transcripts only when the benchmark
        # contract explicitly permits transcript text. A source audio track
        # may still be retained in the exported segment video.
        logger.info(
            "Step 2: Building per-segment transcripts (audio=%s)...",
            "included" if self.config.include_audio_transcript else "excluded",
        )
        def _transcribe_segments_or_empty() -> Dict[str, str]:
            """Keep visual preprocessing available when optional ASR is unavailable."""
            try:
                return self.asr_processor.transcribe_segments(
                    video_name,
                    self.config.working_dir,
                    segment_index2name,
                    segment_times_info=segment_times_info,
                    reuse_cache=self.config.reuse_cache,
                )
            except Exception as exc:
                logger.warning(
                    "Segment-wise ASR unavailable for %s; continuing with empty transcripts: %s",
                    video_name,
                    exc,
                )
                return {str(segment_id): "" for segment_id in segment_times_info}

        if not self.config.include_audio_transcript:
            transcripts = {str(segment_id): "" for segment_id in segment_index2name}
            logger.info(
                "Transcript-excluded input contract active; skipping ASR transcript extraction for %s (source audio is not stripped).",
                video_name,
            )
        elif cached_segment_metadata is not None:
            cached_transcripts = self.asr_processor._load_cached_transcripts(
                Path(self.config.working_dir) / "_cache" / video_name / "transcripts.json",
                segment_index2name=segment_index2name,
            )
            transcripts = {
                str(segment_id): str(cached_transcripts.get(str(segment_id), "") or "")
                for segment_id in segment_index2name
            }
            logger.info(
                "Reusing cached segment metadata for %s with cached_transcripts=%s/%s; no ASR model load.",
                video_name,
                sum(bool(text.strip()) for text in transcripts.values()),
                len(transcripts),
            )
        elif semantic_asr_segments:
            total_duration = 0.0
            for info in segment_times_info.values():
                if not isinstance(info, dict):
                    continue
                timestamp = info.get("timestamp", (0.0, 0.0))
                if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                    continue
                try:
                    total_duration = max(total_duration, float(timestamp[1] or 0.0))
                except Exception:
                    continue
            if total_duration <= 0.0:
                for entry in semantic_asr_segments:
                    try:
                        total_duration = max(total_duration, float(entry.get("end", 0.0) or 0.0))
                    except Exception:
                        continue
            refined_asr_segments = self.video_segmenter.prepare_asr_timestamp_units(
                asr_segments=semantic_asr_segments,
                total_duration=total_duration,
            )
            transcripts = self.asr_processor.build_segment_transcripts(
                segment_times_info=segment_times_info,
                transcript_entries=refined_asr_segments or semantic_asr_segments,
            )
            if not any(str(text or "").strip() for text in transcripts.values()):
                logger.warning("Projected global ASR transcripts are empty; falling back to segment-wise ASR.")
                transcripts = _transcribe_segments_or_empty()
        else:
            transcripts = _transcribe_segments_or_empty()
        
        # A large-memory reusable batch worker can keep Whisper resident so
        # each new video does not pay the ASR cold-load cost again.  Preserve
        # the historical release behavior unless the batch caller opts in;
        # small cards can explicitly set TOOLSTAR_PERSIST_ASR_RUNTIME=0.
        persist_asr_runtime = str(
            os.environ.get("TOOLSTAR_PERSIST_ASR_RUNTIME", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if persist_asr_runtime:
            logger.info("Keeping GPU Whisper runtime cached for reusable worker samples.")
        else:
            self.asr_processor.unload_model()

        # Step 3: Generate captions
        logger.info("Step 3: Generating captions...")
        persist_visual_runtime = str(
            os.environ.get("TOOLSTAR_PERSIST_VISUAL_RUNTIME", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            captions = self.visual_processor.caption_segments(
                video_name,
                video_path,
                segment_index2name,
                transcripts,
                segment_times_info,
                working_dir=self.config.working_dir,
                reuse_cache=self.config.reuse_cache,
            )
        finally:
            # A reusable batch worker keeps the loaded visual runtime in the
            # class cache between samples.  The normal one-shot pipeline keeps
            # its previous release semantics and clears the cache explicitly.
            self.visual_processor.unload_model(
                clear_runtime_cache=not persist_visual_runtime
            )

        logger.info("Step 3.5: Persisting segment clips and sampled frames for multimodal retrieval...")
        clip_assets_task = asyncio.to_thread(
            persist_segment_video_assets,
            video_path=video_path,
            segment_times_info=deepcopy(segment_times_info),
            output_dir=output_dir,
            clips_subdir=settings.processed_clips_subdir,
        )
        frame_assets_task = asyncio.to_thread(
            persist_segment_frame_assets,
            video_path=video_path,
            video_name=video_name,
            segment_times_info=deepcopy(segment_times_info),
            output_dir=output_dir,
            frames_subdir=settings.processed_frames_subdir,
        )
        clip_times_info, frame_times_info = await asyncio.gather(
            clip_assets_task,
            frame_assets_task,
        )
        merged_segment_times_info: Dict[str, Dict[str, Any]] = deepcopy(segment_times_info)
        for seg_id in merged_segment_times_info.keys():
            merged_info = merged_segment_times_info.setdefault(seg_id, {})
            clip_info = clip_times_info.get(seg_id, {}) if isinstance(clip_times_info, dict) else {}
            frame_info = frame_times_info.get(seg_id, {}) if isinstance(frame_times_info, dict) else {}
            if isinstance(clip_info, dict):
                if clip_info.get("segment_video_path"):
                    merged_info["segment_video_path"] = clip_info.get("segment_video_path")
                if clip_info.get("timestamp") is not None:
                    merged_info["timestamp"] = clip_info.get("timestamp")
            if isinstance(frame_info, dict):
                for key in ("frame_times", "frame_paths", "segment_aliases"):
                    if frame_info.get(key) not in (None, ""):
                        merged_info[key] = frame_info.get(key)
        segment_times_info = merged_segment_times_info
        
        # Step 4: Build segment data
        logger.info("Step 4: Building segment data...")
        segments = self._merge_segment_information(
            segment_index2name,
            segment_times_info,
            transcripts,
            captions
        )
        self._validate_segment_contract(segments)
        
        # Step 5: Keep the planner-facing segment schema compact and explicit.
        logger.info("Step 5: Normalizing compact segment schema...")
        enhanced_segments = segments
        
        # Register segments for KG fallback usage (time, caption, transcript)
        kg_builder = self._ensure_kg_builder()
        if kg_builder:
            for seg_id, seg_data in enhanced_segments.items():
                kg_builder.register_segment(seg_id, seg_data)
        
        # Step 6: Build knowledge graph
        if kg_builder:
            logger.info("Step 6: Building knowledge graph...")
            await self._build_knowledge_graph(enhanced_segments)
            self._attach_graph_text(enhanced_segments)
        
        # Step 7: Index in vector store
        sparse_threshold = max(1, int(getattr(settings, "sparse_video_segment_threshold", 3) or 3))
        skip_sparse_vector_index = (
            self.config.build_vector_index
            and not self.config.build_sparse_vector_index
            and len(enhanced_segments) <= sparse_threshold
        )
        if self.config.build_vector_index and not skip_sparse_vector_index:
            logger.info("Step 7: Indexing in vector store...")
            self.vector_store.index_segments(
                enhanced_segments,
                video_name,
                processed_root=str(output_dir),
            )
        elif skip_sparse_vector_index:
            logger.info(
                "Step 7: Skipping dense index for sparse video (segments=%s threshold=%s); "
                "retaining caption/transcript/frame evidence.",
                len(enhanced_segments),
                sparse_threshold,
            )
        else:
            logger.info(
                "Step 7: Local vector index disabled; retaining caption/transcript grounding inputs."
            )
        
        # Step 8: Save outputs
        if self.config.save_intermediate:
            logger.info("Step 8: Saving outputs...")
            self._save_outputs(
                video_name,
                video_path,
                enhanced_segments,
                segment_index2name=segment_index2name,
                segment_times_info=segment_times_info,
                transcripts=transcripts,
                segment_strategy=segment_strategy,
            )
        
        logger.info(f"Processing complete for {video_name}")
        
        return {
            "video_name": video_name,
            "video_path": video_path,
            "segments": enhanced_segments,
            "num_segments": len(enhanced_segments)
        }
    
    async def process_multiple_videos(
        self,
        video_paths: List[str]
    ) -> List[Dict]:
        """
        Process multiple videos.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of processing results
        """
        results = []
        for video_path in video_paths:
            result = await self.process_video(video_path)
            results.append(result)
        return results
    
    def _merge_segment_information(
        self,
        segment_index2name: Dict[str, str],
        segment_times_info: Dict[str, Dict],
        transcripts: Dict[str, str],
        captions: Dict[str, str]
    ) -> Dict[str, Dict]:
        """Merge all segment information."""
        segments = {}
        
        for index in segment_index2name:
            segment_name = segment_index2name[index]
            time_info = segment_times_info.get(index, {})
            
            timestamp = time_info.get("timestamp")
            if timestamp and len(timestamp) == 2:
                start, end = timestamp
                time_range = f"{_format_segment_time_value(start)}-{_format_segment_time_value(end)}"
            else:
                time_parts = segment_name.split('-')[-2:]
                if len(time_parts) == 2:
                    time_range = f"{time_parts[0]}-{time_parts[1]}"
                else:
                    time_range = "0-0"
            
            caption_text = captions.get(index, "")
            transcript_text = transcripts.get(index, "")
            
            frame_times = time_info.get("frame_times", [])
            if hasattr(frame_times, "tolist"):
                frame_times = frame_times.tolist()
            frame_paths = list(time_info.get("frame_paths", []) or [])
            segment_aliases = list(time_info.get("segment_aliases", []) or [str(index)])
            segment_video_path = str(time_info.get("segment_video_path", "") or "")
            
            segments[index] = {
                "time": time_range,
                "caption": caption_text,
                "transcript": transcript_text,
                "frame_times": frame_times,
                "frame_paths": frame_paths,
                "segment_aliases": [alias for alias in segment_aliases if alias],
                "segment_video_path": segment_video_path,
            }
        
        return segments

    def _validate_segment_contract(self, segments: Dict[str, Dict[str, Any]]) -> None:
        """Enforce the planner-facing preprocessing contract before downstream indexing."""
        invalid_segments: List[str] = []
        for seg_id, seg_data in segments.items():
            if not isinstance(seg_data, dict):
                invalid_segments.append(str(seg_id))
                continue
            if "caption" not in seg_data or "transcript" not in seg_data:
                invalid_segments.append(str(seg_id))
                continue
            if _is_low_information_caption_text(seg_data.get("caption", "")):
                invalid_segments.append(str(seg_id))
                continue
            if not str(seg_data.get("segment_video_path", "") or "").strip():
                invalid_segments.append(str(seg_id))

        if invalid_segments:
            joined = ", ".join(invalid_segments[:8])
            raise RuntimeError(
                "Preprocessing produced invalid segment artifacts for planner grounding/indexing. "
                f"Affected segments: {joined}. "
                "Each segment must contain a real caption, transcript field, and segment_video_path."
            )

    def _attach_graph_text(self, segments: Dict[str, Dict[str, Any]]) -> None:
        """Attach per-segment graph text for graph-side tools and debugging."""
        if not self.kg_builder:
            return

        graph_text_map: Dict[str, List[str]] = {}
        for entity_name, entity in self.kg_builder.entities.items():
            for seg_id in entity.source_segments or []:
                graph_text_map.setdefault(str(seg_id), []).append(entity_name)
                if entity.description:
                    graph_text_map[str(seg_id)].append(entity.description)

        for relation in self.kg_builder.relations:
            relation_text = " ".join(
                part for part in [relation.relation_type, relation.description] if part
            ).strip()
            if not relation_text:
                continue
            for seg_id in relation.source_segments or []:
                graph_text_map.setdefault(str(seg_id), []).append(relation_text)

        for seg_id, seg_data in segments.items():
            graph_text = " ".join(graph_text_map.get(str(seg_id), []))
            if graph_text:
                seg_data["graph_text"] = graph_text
    
    async def _build_knowledge_graph(self, segments: Dict) -> None:
        """Build knowledge graph from segments."""
        segment_order = sorted(segments.keys(), key=lambda x: int(x))
        
        for seg_id in segment_order:
            # Combine caption and transcript for entity extraction
            caption = segments[seg_id].get("caption", "")
            transcript = segments[seg_id].get("transcript", "")
            content = f"Caption: {caption}\nTranscript: {transcript}"
            # LLM-based extraction
            entities, relations = await self.kg_builder.extract_entities_from_segment(
                content,
                seg_id,
                max_gleaning=self.config.max_entity_extraction_rounds
            )
            # Harvest simple keyword entities from caption + transcript for alignment
            time_tuple = self.kg_builder._parse_time_range(segments[seg_id].get("time", "0-0"))
            harvested_caption = self.kg_builder._harvest_text_entities(
                segments[seg_id].get("caption", ""),
                seg_id,
                time_tuple,
                modality="visual",
            )
            harvested_asr = self.kg_builder._harvest_text_entities(
                segments[seg_id].get("transcript", ""),
                seg_id,
                time_tuple,
                modality="audio",
            )
            entities.extend(harvested_caption + harvested_asr)

            # Cross-modal alignment and interaction/holding cues
            entities, align_relations = self.kg_builder.align_modal_entities(
                entities,
                time_tuple,
                seg_id,
                iou_threshold=0.2,
            )
            self.kg_builder.add_interaction_relations(
                entities,
                seg_id,
                time_tuple,
                text=f"{segments[seg_id].get('caption', '')} {segments[seg_id].get('transcript', '')}",
            )
            relations.extend(align_relations)

            # Co-occurrence edges within segment
            self.kg_builder.add_cooccurrence_relations(entities, seg_id, time_tuple)
            
            for entity in entities:
                self.kg_builder.add_entity(entity)
            for relation in relations:
                self.kg_builder.add_relation(relation)
        
        # Build temporal connections
        self.kg_builder.build_temporal_connections(segment_order)
    
    def _save_outputs(
        self,
        video_name: str,
        video_path: str,
        segments: Dict,
        segment_index2name: Optional[Dict[str, str]] = None,
        segment_times_info: Optional[Dict[str, Dict]] = None,
        transcripts: Optional[Dict[str, str]] = None,
        segment_strategy: Optional[str] = None,
    ) -> None:
        """Save processing outputs."""
        output_dir = Path(self.config.working_dir) / "processed" / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save segments
        _atomic_write_json(output_dir / "segments.json", segments)

        # Save source video metadata so runtime tools can re-open the original video.
        video_metadata = {
            "video_name": video_name,
            "source_video_path": str(video_path),
        }
        _atomic_write_json(output_dir / "video_metadata.json", video_metadata)

        # Save transcripts in a normalized format
        if transcripts is not None:
            normalized = {str(k): str(v) for k, v in transcripts.items()}
            _atomic_write_json(output_dir / "transcripts.json", normalized)

        # Save segment_times_info for alignment/debugging
        if segment_index2name is not None and segment_times_info is not None:
            safe_times_info = {}
            for idx, info in segment_times_info.items():
                frame_times = info.get("frame_times", [])
                if hasattr(frame_times, "tolist"):
                    frame_times = frame_times.tolist()
                safe_times_info[str(idx)] = {
                    "frame_times": frame_times,
                    "frame_paths": list(info.get("frame_paths", []) or []),
                    "segment_video_path": str(info.get("segment_video_path", "") or ""),
                    "segment_aliases": list(info.get("segment_aliases", []) or build_segment_aliases(video_name, idx)),
                    "timestamp": info.get("timestamp", (0, 0))
                }
            payload = {
                "segment_cache_version": self.video_segmenter.CACHE_VERSION,
                "segment_strategy": segment_strategy or self._segment_strategy_label(False),
                "input_modalities": (
                    ["video", "audio_transcript"]
                    if self.config.include_audio_transcript
                    else ["video"]
                ),
                "audio_policy": (
                    "included" if self.config.include_audio_transcript else "excluded"
                ),
                "audio_track_policy": "source_preserved",
                "audio_transcript_policy": (
                    "included" if self.config.include_audio_transcript else "excluded"
                ),
                "segment_index2name": {str(k): str(v) for k, v in segment_index2name.items()},
                "segment_times_info": safe_times_info
            }
            _atomic_write_json(output_dir / "segment_times_info.json", payload)
        
        # Save knowledge graph
        if self.kg_builder:
            self.kg_builder.save_graph(str(output_dir / "knowledge_graph.json"))
        
        # Save vector store
        self.vector_store.save(str(output_dir / "vectors"))
    
    def _try_load_processed(
        self,
        output_dir: Path,
        video_name: str
    ) -> Optional[Dict]:
        """Attempt to load processed artifacts to avoid repeated preprocessing."""
        segments_path = output_dir / "segments.json"
        transcripts_path = output_dir / "transcripts.json"
        segment_times_path = output_dir / "segment_times_info.json"
        vectors_dir = output_dir / "vectors"

        if not segments_path.exists():
            return None

        try:
            with open(segments_path, "r", encoding="utf-8") as f:
                segments = json.load(f)

            transcripts = None
            if transcripts_path.exists():
                with open(transcripts_path, "r", encoding="utf-8") as f:
                    transcripts = json.load(f)

            segment_times_info = None
            if segment_times_path.exists():
                with open(segment_times_path, "r", encoding="utf-8") as f:
                    segment_times_info = json.load(f)

            if not self._processed_segment_cache_is_current(segment_times_info):
                logger.info(
                    "Cached preprocessing outputs for %s were produced by an older segmentation version; regenerating.",
                    video_name,
                )
                return None

            expected_audio_policy = (
                "included" if self.config.include_audio_transcript else "excluded"
            )
            cached_audio_policy = str(
                (segment_times_info or {}).get("audio_transcript_policy")
                or (segment_times_info or {}).get("audio_policy")
                or ""
            ).strip().lower()
            if cached_audio_policy != expected_audio_policy:
                logger.info(
                    "Cached preprocessing outputs for %s use audio_policy=%s, expected=%s; regenerating.",
                    video_name,
                    cached_audio_policy or "missing",
                    expected_audio_policy,
                )
                return None

            if _segments_look_stale(segments):
                logger.info(
                    "Cached preprocessing outputs for %s have stale segment timing; regenerating.",
                    video_name,
                )
                return None

            if not self._cache_has_multimodal_assets(output_dir, segments, segment_times_info):
                logger.info("Cached preprocessing outputs for %s are missing frame assets or aliases; regenerating.", video_name)
                return None

            video_path = None
            video_metadata_path = output_dir / "video_metadata.json"
            if video_metadata_path.exists():
                with open(video_metadata_path, "r", encoding="utf-8") as f:
                    video_metadata = json.load(f)
                video_path = video_metadata.get("source_video_path")

            sparse_threshold = max(1, int(getattr(settings, "sparse_video_segment_threshold", 3) or 3))
            allow_sparse_empty_index = (
                not self.config.build_sparse_vector_index
                and len(segments) <= sparse_threshold
            )
            if vectors_dir.exists() and (vectors_dir / "metadata.json").exists():
                if not _vector_index_cache_is_complete(vectors_dir, len(segments)) and not allow_sparse_empty_index:
                    logger.warning(
                        "Cached vector index at %s is empty or incomplete for %d segments; rebuilding it.",
                        vectors_dir,
                        len(segments),
                    )
                    return None
                try:
                    self.vector_store.load(str(vectors_dir))
                except Exception as e:
                    logger.warning(f"Failed to load cached vector store: {e}")
                    return None
            else:
                return None

            return {
                "video_name": video_name,
                "video_path": video_path,
                "segments": segments,
                "num_segments": len(segments),
                "transcripts": transcripts,
                "segment_times_info": segment_times_info,
            }
        except Exception as e:
            logger.warning(f"Failed to load cached outputs from {output_dir}: {e}")
            return None

    def _processed_segment_cache_is_current(
        self,
        segment_times_payload: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(segment_times_payload, dict):
            return False
        try:
            return int(segment_times_payload.get("segment_cache_version", 0) or 0) == self.video_segmenter.CACHE_VERSION
        except Exception:
            return False

    def _cache_has_multimodal_assets(
        self,
        output_dir: Path,
        segments: Dict[str, Dict[str, Any]],
        segment_times_payload: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(segments, dict) or not segments:
            return False
        if not isinstance(segment_times_payload, dict):
            return False

        times_map = segment_times_payload.get("segment_times_info")
        if not isinstance(times_map, dict):
            return False

        for seg_id, seg_data in segments.items():
            if not isinstance(seg_data, dict):
                return False
            time_info = times_map.get(str(seg_id), {})
            if not isinstance(time_info, dict):
                return False

            if "caption" not in seg_data or "transcript" not in seg_data:
                return False
            if _is_low_information_caption_text(seg_data.get("caption", "")):
                return False

            frame_paths = list(seg_data.get("frame_paths") or time_info.get("frame_paths") or [])
            segment_video_path = str(seg_data.get("segment_video_path") or time_info.get("segment_video_path") or "").strip()
            segment_aliases = list(seg_data.get("segment_aliases") or time_info.get("segment_aliases") or [])
            if not frame_paths or not segment_aliases or not segment_video_path:
                return False

            for rel_path in frame_paths:
                frame_path = output_dir / str(rel_path)
                if not frame_path.exists():
                    return False
            clip_path = output_dir / segment_video_path
            if not clip_path.exists():
                return False
        return True
    
    def search(
        self,
        query: str,
        k: int = 5,
        time_range: Optional[tuple] = None
    ) -> List[Dict]:
        """
        Search for relevant segments.
        
        Args:
            query: Search query
            k: Number of results
            time_range: Optional time range filter
            
        Returns:
            List of matching segments
        """
        return self.vector_store.search_segments(query, k=k, time_range=time_range)
