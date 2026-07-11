"""Settings configuration using Pydantic BaseSettings."""
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings for the Tool Star RL framework."""
    
    # OpenAI Configuration
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4"
    # auto: Aliyun/DashScope-style endpoints use extra_body.enable_thinking;
    # other OpenAI-compatible Qwen endpoints use extra_body.chat_template_kwargs.enable_thinking.
    openai_thinking_control_mode: str = "auto"
    
    # Anthropic Configuration
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    
    # System Configuration
    max_recursion_depth: int = 5
    max_iterations: int = 15
    subagent_max_iterations: int = 3
    enable_streaming: bool = False
    planner_max_generation_tokens: int = 8192
    planner_min_thinking_generation_tokens: int = 8192
    subagent_max_generation_tokens: int = 2048
    planner_temperature: float = 1.0
    planner_top_p: float = 0.95
    planner_top_k: int = 20
    planner_min_p: float = 0.0
    planner_presence_penalty: float = 1.5
    planner_repetition_penalty: float = 1.0
    subagent_temperature: float = 0.7
    subagent_top_p: float = 0.8
    subagent_top_k: int = 20
    subagent_min_p: float = 0.0
    subagent_presence_penalty: float = 1.5
    subagent_repetition_penalty: float = 1.0
    structured_repair_timeout_seconds: float = 600.0
    planner_enable_thinking: bool = True
    subagent_enable_thinking: bool = False
    graph_enable: bool = False
    web_search_enabled: bool = False
    entity_track_enabled: bool = False
    disable_meta_tools: bool = True
    disabled_tool_names_base: Tuple[str, ...] = ("Debt_Interest_Analyzer",)
    disabled_graph_tool_names: Tuple[str, ...] = (
        "Graph_Search",
        "Neighbor_List_Graph",
        "Degree_Count_Graph",
        "Edge_Filter_Graph",
        "Path_Length_Filter",
    )
    sparse_video_segment_threshold: int = 3
    sparse_video_hidden_tool_names: Tuple[str, ...] = (
        "Semantic_Retrieve",
        "Keyword_Retrieve",
        "Segment_Retrieve",
        "asr",
        "Clean_Results",
        "Graph_Search",
    )
    
    # Memory Optimization
    max_input_tokens: int = 262144
    # <= 0 means no global output-token cap.
    max_output_tokens: int = 0
    # False by default: do not enforce global output cap even if MAX_OUTPUT_TOKENS is set.
    enforce_max_output_tokens: bool = False
    preserve_full_observations: bool = True
    enable_auto_memory_folding: bool = False
    planner_grounding_top_k: int = 3
    planner_grounding_embedding_provider: str = "local"
    planner_grounding_embedding_api_key: str = "EMPTY"
    planner_grounding_embedding_api_base: str = "http://127.0.0.1:18002/v1"
    planner_grounding_embedding_model: str = ""
    non_planner_model_provider: str = "auto"
    non_planner_api_key: str = "EMPTY"
    non_planner_api_base: str = "http://127.0.0.1:18000/v1"
    non_planner_model: str = ""
    sidecar_load_balance_policy: str = "least_inflight"
    sidecar_endpoint_failure_cooldown_seconds: float = 20.0
    sidecar_port_stride: int = 10
    sidecar_startup_timeout_seconds: int = 900
    sidecar_health_check_interval_seconds: int = 5
    sidecar_gpu_quiesce_seconds: int = 20
    sidecar_smoke_check_enabled: bool = True
    sidecar_smoke_timeout_seconds: int = 120
    sidecar_vllm_use_v1: str = ""
    sidecar_llm_disable_chunked_prefill: bool = False
    sidecar_llm_disable_chunked_mm_input: bool = True
    sidecar_llm_disable_async_scheduling: bool = True
    sidecar_refuse_when_ray_worker_active: bool = True
    sidecar_cleanup_stale_processes: bool = True
    sidecar_kill_orphan_vllm_workers: bool = True
    sidecar_spare_llm_gpu_memory_utilization: float = 0.30
    sidecar_spare_embedding_gpu_memory_utilization: float = 0.16
    sidecar_shared_llm_gpu_memory_utilization: float = 0.30
    sidecar_shared_embedding_gpu_memory_utilization: float = 0.10
    sidecar_embedding_max_model_len: int = 8192
    sidecar_spare_llm_max_model_len: int = 12288
    sidecar_shared_llm_max_model_len: int = 8192
    sidecar_spare_llm_max_batched_tokens: int = 8192
    sidecar_shared_llm_max_batched_tokens: int = 8192
    sidecar_spare_llm_kv_cache_memory_bytes: str = ""
    sidecar_shared_llm_kv_cache_memory_bytes: str = "2G"
    sidecar_llm_max_tokens_per_mm_item: int = 16384
    sidecar_spare_embedding_max_batched_tokens: int = 1024
    sidecar_shared_embedding_max_batched_tokens: int = 512
    sidecar_spare_llm_cpu_offload_gb: int = 0
    sidecar_shared_llm_cpu_offload_gb: int = 0
    sidecar_spare_llm_tensor_parallel_size: int = 1
    sidecar_shared_llm_tensor_parallel_size: int = 4
    sidecar_spare_llm_replica_count: int = 2
    sidecar_shared_llm_replica_count: int = 1
    non_planner_sidecar_replica_count: int = 1
    tool_search_embedding_sidecar_replica_count: int = 1
    grounding_embedding_sidecar_replica_count: int = 1
    # Optional per-role overrides. Edit these in config instead of exporting
    # TOOL_STAR_* variables for every training run.
    non_planner_role_providers: Dict[str, str] = {}
    non_planner_role_api_keys: Dict[str, str] = {}
    non_planner_role_api_bases: Dict[str, str] = {}
    non_planner_role_models: Dict[str, str] = {}
    non_planner_embedding_provider: str = "local"
    non_planner_embedding_api_key: str = "EMPTY"
    non_planner_embedding_api_base: str = "http://127.0.0.1:18002/v1"
    non_planner_embedding_model: str = ""
    non_planner_embedding_role_providers: Dict[str, str] = {
        "tool_search": "local",
        "preprocess_vector": "local",
    }
    non_planner_embedding_role_api_keys: Dict[str, str] = {
        "tool_search": "EMPTY",
        "preprocess_vector": "EMPTY",
    }
    non_planner_embedding_role_api_bases: Dict[str, str] = {
        "tool_search": "http://127.0.0.1:18001/v1",
        "preprocess_vector": "http://127.0.0.1:18002/v1",
    }
    non_planner_embedding_role_models: Dict[str, str] = {}
    # Preprocessing models are inference-only. Keep them off the RL actor CUDA
    # context unless explicitly changed here.
    preprocess_asr_device: str = "cpu"
    per_device_max_memory_gb: Optional[float] = None
    planner_visual_observation_token_budget: int = 12288
    planner_visual_observation_image_token_estimate: int = 1536
    planner_visual_observation_pack_group_size: int = 4
    planner_visual_observation_subdir: str = "planner_evidence"
    planner_grounding_frame_interval_seconds: float = 5.0
    planner_visual_single_image_size: int = 672
    planner_visual_collage_tile_size: int = 336
    planner_visual_max_pixels: int = 336 * 336
    acceptance_min_unique_runtime_tools: int = 4
    qwen_vl_image_min_pixels: int = 256 * 256
    qwen_vl_image_max_pixels: int = 1280 * 28 * 28
    visual_service_max_frames: int = 1
    visual_service_max_pixels: int = 336 * 336
    video_clip_qa_fps: float = 2.0
    # Qwen3.5 video defaults follow the official Alibaba Cloud guidance:
    # 32x32 pixels/token, min 4 tokens per frame, up to 80 sampled frames,
    # about 768 tokens/frame, and 65536 tokens total across the video.
    video_clip_qa_min_pixels: int = 4 * 32 * 32
    video_clip_qa_processor_max_pixels: int = 65536 * 32 * 32
    video_clip_qa_max_frames: int = 80
    video_clip_qa_max_pixels: int = 768 * 32 * 32
    video_clip_qa_max_new_tokens: int = 8192
    video_clip_qa_think_max_chars: int = 4000
    
    # Multi-Device Configuration
    # NPU: Multi-device enabled by default (ensure HCCL environment is configured)
    # CUDA: Auto multi-device is enabled by default
    force_single_device: bool = False  # Set to True to force single device even with multiple visible
    enable_npu_multi_device: bool = True  # Enable NPU multi-device inference
    caption_multi_device_enabled: bool = False  # Keep caption VLM single-device by default; enable only when caption throughput benefits outweigh pipeline stalls
    caption_device_map: str = ""  # Caption-only override: "", "auto", "balanced", or "sequential"
    caption_multi_device_in_distributed: bool = False  # When caption multi-device is enabled globally, keep this False to still force single-device inside Ray/FSDP workers
    preprocess_video_file_lock_enabled: bool = False  # False improves first-step GPU utilization on repeated videos
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "toolrl.log"
    enable_llm_output_debug: bool = True
    llm_output_debug_max_chars: int = 8000
    
    # Data Storage
    data_cache_dir: str = "./videorag_cache"
    processed_frames_subdir: str = "frames"
    processed_clips_subdir: str = "clips"
    
    # Model Base Path - User only needs to modify this ONE line

    model_base_path: str = "/mnt/data/LiuXiao/models"
    planner_model_subpath: str = "Qwen/Qwen3.5-9B"
    subagent_model_subpath: str = "Qwen/Qwen3.5-9B"
    preprocess_model_subpath: str = "Qwen/Qwen3.5-9B"
    qwen3_vl_embedding_model_subpath: str = "Qwen/Qwen3-VL-Embedding-2B"
    qwen_vl_model_subpath: str = "Qwen/Qwen3.5-9B"
    ocr_model_subpath: str = "Qwen/Qwen3.5-9B"
    
    # LLM Provider: "openai", "anthropic", or "local"
    llm_provider: str = "local"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @property
    def config_dir(self) -> Path:
        """Get config directory path."""
        return Path(__file__).parent
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self.config_dir.parent
    
    # ========== Model Path Helpers (Computed Properties) ==========
    # All model paths are computed from model_base_path + relative subpath
    
    # Planner and subagent intentionally reuse the same checkpoint by default.
    # Their behavior diverges through role-specific prompts and generation knobs.

    @property
    def planner_model_path(self) -> str:
        """Planner LLM model path. Think mode; uses planner_* generation knobs."""
        return str(Path(self.model_base_path) / self.planner_model_subpath)

    @property
    def subagent_model_path(self) -> str:
        """Subagent LLM model path. No-think mode; uses subagent_* generation knobs."""
        return str(Path(self.model_base_path) / self.subagent_model_subpath)
    
    @property
    def preprocess_model_path(self) -> str:
        """Preprocessing LLM model path. no-think mode. temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"""
        return str(Path(self.model_base_path) / self.preprocess_model_subpath)
    
    @property
    def qwen_vl_model_path(self) -> str:
        """Shared visual model path used by preprocessing captioning and image-returning visual tools."""
        return str(Path(self.model_base_path) / self.qwen_vl_model_subpath)

    @property
    def qwen3_vl_embedding_model_path(self) -> str:
        """Dense multimodal embedding model path used by preprocessing vector search."""
        return str(Path(self.model_base_path) / self.qwen3_vl_embedding_model_subpath)

    @property
    def embedding_model_path(self) -> str:
        """Embedding model path (Tool_Search)."""
        return str(Path(self.model_base_path) / "Qwen/Qwen3-Embedding-0.6B")
    
    @property
    def query_rewrite_model_path(self) -> str:
        """Query rewriting model path. temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"""
        return str(Path(self.model_base_path) / "Qwen/Qwen3.5-0.8B")
    
    @property
    def python_coder_model_path(self) -> str:
        """Python code executor model path."""
        return str(Path(self.model_base_path) / "Qwen/Qwen2.5-Coder-7B-Instruct")
    
    @property
    def entity_track_model_path(self) -> str:
        """Entity tracking (Grounding DINO) model path."""
        return str(Path(self.model_base_path) / "IDEA-Research/grounding-dino-base")
    
    @property
    def audio_event_model_path(self) -> str:
        """Audio event detection (CLAP) model path."""
        return str(Path(self.model_base_path) / "laion/clap-htsat-fused")
    
    @property
    def ocr_model_path(self) -> str:
        """OCR extraction model path. temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"""
        return str(Path(self.model_base_path) / self.ocr_model_subpath)

    @property
    def speaker_diarization_model_path(self) -> str:
        """Speaker diarization model path."""
        return str(Path(self.model_base_path) / "pyannote/speaker-diarization-community-1")
    
    @property
    def emotion_audio_model_path(self) -> str:
        """Emotion analysis (audio) model path."""
        return str(Path(self.model_base_path) / "speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
    
    @property
    def emotion_image_model_path(self) -> str:
        """Emotion analysis (image) model path."""
        return str(Path(self.model_base_path) / "Xenova/facial_emotions_image_detection")
    
    @property
    def birefnet_model_path(self) -> str:
        """BiRefNet (background removal, switch to RMBG-2.0) model path."""
        return str(Path(self.model_base_path) / "briaai/RMBG-2.0")
    
    @property
    def super_resolution_model_path(self) -> str:
        """Super resolution model path."""
        return str(Path(self.model_base_path) / "iic/cv_rrdb_image-super-resolution")
    
    @property
    def asr_model_path(self) -> str:
        """ASR (Faster-Whisper) model path."""
        return str(Path(self.model_base_path) / "Systran/faster-whisper-large-v3")

    asr_cpu_chunk_seconds: int = 90
    asr_cpu_chunk_threshold_seconds: int = 150
    asr_cpu_threads: int = 8
    
    @property
    def fold_memory_model_path(self) -> str:
        """Memory folding model path. temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"""
        return str(Path(self.model_base_path) / "Qwen/Qwen3.5-0.8B")

    @property
    def planner_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.planner_temperature,
            "top_p": self.planner_top_p,
            "top_k": self.planner_top_k,
            "min_p": self.planner_min_p,
            "presence_penalty": self.planner_presence_penalty,
            "repetition_penalty": self.planner_repetition_penalty,
            "enable_thinking": bool(self.planner_enable_thinking),
            "max_new_tokens": int(self.planner_max_generation_tokens),
        }

    @property
    def subagent_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.subagent_temperature,
            "top_p": self.subagent_top_p,
            "top_k": self.subagent_top_k,
            "min_p": self.subagent_min_p,
            "presence_penalty": self.subagent_presence_penalty,
            "repetition_penalty": self.subagent_repetition_penalty,
            "enable_thinking": bool(self.subagent_enable_thinking),
            "max_new_tokens": int(self.subagent_max_generation_tokens),
        }

    @property
    def preprocess_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "enable_thinking": False,
        }

    @property
    def qwen_vl_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "enable_thinking": False,
            "max_new_tokens": 2048,
        }

    @property
    def query_rewrite_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "enable_thinking": False,
        }

    @property
    def ocr_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "enable_thinking": False,
            "max_new_tokens": 2048,
        }

    @property
    def fold_memory_generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "enable_thinking": False,
            "max_new_tokens": 768,
        }

    @property
    def disabled_tool_names(self) -> Tuple[str, ...]:
        names = [str(name).strip() for name in (self.disabled_tool_names_base or ()) if str(name).strip()]
        if not bool(self.graph_enable):
            names.extend(
                str(name).strip()
                for name in (self.disabled_graph_tool_names or ())
                if str(name).strip()
            )
        if not bool(self.web_search_enabled):
            names.append("Web_Search")
        if not bool(self.entity_track_enabled):
            names.append("Entity_Track")
        return tuple(dict.fromkeys(names))

    @staticmethod
    def _normalize_tool_tags(tool_tags: Optional[Iterable[Any]]) -> Tuple[str, ...]:
        if tool_tags is None:
            return ()
        normalized = []
        for tag in tool_tags:
            text = str(tag or "").strip().lower()
            if text:
                if text == "biz":
                    text = "base"
                normalized.append(text)
        return tuple(dict.fromkeys(normalized))

    def is_tool_enabled(
        self,
        tool_name: str,
        tool_tags: Optional[Iterable[Any]] = None,
    ) -> bool:
        normalized_name = str(tool_name or "").strip()
        if not normalized_name:
            return False
        if normalized_name in set(self.disabled_tool_names):
            return False
        if bool(self.disable_meta_tools):
            normalized_tags = self._normalize_tool_tags(tool_tags)
            if "meta" in normalized_tags:
                return False
        return True

    @property
    def always_available_tool_names(self) -> Tuple[str, ...]:
        names = ["Tool_Search", "Python_Executor"]
        if bool(self.web_search_enabled):
            names.append("Web_Search")
        return tuple(names)

    def get_generation_config(
        self,
        model_role: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        role = str(model_role or "").strip().lower()
        resolved_provider = str(provider or "").strip().lower()
        if role == "planner":
            config = dict(self.planner_generation_config)
        elif role == "subagent":
            config = dict(self.subagent_generation_config)
        elif role == "preprocess":
            config = dict(self.preprocess_generation_config)
        elif role == "qwen_vl":
            config = dict(self.qwen_vl_generation_config)
        elif role == "query_rewrite":
            config = dict(self.query_rewrite_generation_config)
        elif role == "ocr":
            config = dict(self.ocr_generation_config)
        elif role == "fold_memory":
            config = dict(self.fold_memory_generation_config)
        else:
            config = {}

        if resolved_provider in {"transformers", "local"}:
            config.pop("presence_penalty", None)

        return config


# Global settings instance
settings = Settings()
