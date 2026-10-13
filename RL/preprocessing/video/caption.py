"""
Visual Captioning and Feature Extraction Module
Generates frame descriptions and visual features for video segments.
"""
import os
import json
import logging
import base64
import gc
import inspect
import re
import sys
from contextlib import nullcontext
from io import BytesIO
import torch
import numpy as np
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

from config.settings import settings
from core.llm.base import is_accelerator_fatal_error
from core.qwen_runtime import disable_qwen35_fused_kernels_if_requested
from core.sidecar_pool import format_endpoint_list
from toolkit.impl.visual_prompts import (
    looks_like_truncated_video_clip_qa_answer,
    merge_instruction_into_user_message,
    normalize_video_clip_qa_answer,
)

logger = logging.getLogger(__name__)

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

DEFAULT_VISUAL_CAPTION_PROMPT = (
    "Output only the final visual caption for this video segment. "
    "Carefully describe the visible entities, people, objects, their actions, positions, states, scene setting, and changes across the sampled images. "
    "Transcribe important visible words, numbers, chart titles, axis labels, legends, logos, and on-screen annotations exactly when readable. "
    "Write direct factual sentences, for example use 'A woman smiles beside a table' instead of 'The image shows a woman smiling beside a table'. "
    "Do not mention the user, prompt, image numbers, frames, analysis, reasoning, steps, bullets, Markdown, hidden thoughts, or labels such as Caption: or Answer:."
)
_CAPTION_MAX_SENTENCES = 4
_CAPTION_MAX_WORDS = 80
_CAPTION_MAX_CHARS = 420


def _resolve_caption_max_new_tokens() -> int:
    """Resolve the compact-caption token budget without changing other QA budgets.

    The Orion-safe Torch fallback is substantially slower than the fused Qwen
    kernels during autoregressive decoding.  Keep the historical 256-token
    default, but allow a batch caller to choose a smaller, still grounded
    caption budget while retaining the same OOM retry and cache semantics.
    """
    raw_value = os.environ.get("TOOLSTAR_CAPTION_MAX_NEW_TOKENS", "256")
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid TOOLSTAR_CAPTION_MAX_NEW_TOKENS=%r; using 256.",
            raw_value,
        )
        return 256


_CAPTION_DEFAULT_MAX_NEW_TOKENS = _resolve_caption_max_new_tokens()
_CAPTION_DEFAULT_MAX_FRAMES = 8


def _resolve_caption_max_frames() -> int:
    """Resolve the bounded frame budget used only for semantic captions.

    Segment frame assets remain complete and are persisted separately for
    retrieval/Video_Clip_QA.  This budget limits the expensive caption prompt
    while keeping every caption grounded in real sampled visual frames.
    """
    raw_value = os.environ.get("TOOLSTAR_CAPTION_MAX_FRAMES", str(_CAPTION_DEFAULT_MAX_FRAMES))
    try:
        return max(1, min(8, int(raw_value)))
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid TOOLSTAR_CAPTION_MAX_FRAMES=%r; using %d.",
            raw_value,
            _CAPTION_DEFAULT_MAX_FRAMES,
        )
        return _CAPTION_DEFAULT_MAX_FRAMES


_CAPTION_OOM_RETRY_PLANS: Tuple[Tuple[str, Optional[int], float, int, bool], ...] = (
    ("default", None, 1.0, _CAPTION_DEFAULT_MAX_NEW_TOKENS, True),
    ("reduced_frames", 6, 0.75, min(_CAPTION_DEFAULT_MAX_NEW_TOKENS, 192), True),
    ("compact_frames", 4, 0.5, min(_CAPTION_DEFAULT_MAX_NEW_TOKENS, 128), False),
    ("minimal_frames", 2, 0.35, min(_CAPTION_DEFAULT_MAX_NEW_TOKENS, 96), False),
)
_MIN_FRAME_EDGE = 28
_BICUBIC_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
_VIDEO_METADATA_KEY_ALIASES = {
    "video_fps": "fps",
    "sample_fps": "fps",
    "frames": "total_num_frames",
    "frame_count": "total_num_frames",
    "num_frames": "total_num_frames",
}
_VIDEO_METADATA_ALLOWED_KEYS = frozenset({"fps", "duration", "total_num_frames"})
_FLOAT_MULTIMODAL_TENSOR_KEYS = frozenset({"pixel_values", "pixel_values_videos"})
_VISUAL_MULTIMODAL_TENSOR_KEYS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
)
_LANGUAGE_INPUT_TENSOR_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
        "position_ids",
        "labels",
    }
)
_VIDEO_PROCESSOR_CORE_KWARGS = frozenset(
    {
        "text",
        "padding",
        "return_tensors",
        "images",
        "videos",
        "video_metadata",
        "fps",
        "max_frames",
        "do_sample_frames",
    }
)


def _read_nonnegative_int_env(name: str) -> Optional[int]:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _is_distributed_worker_context() -> bool:
    world_size = _read_nonnegative_int_env("WORLD_SIZE")
    local_world_size = _read_nonnegative_int_env("LOCAL_WORLD_SIZE")
    if world_size is not None and world_size > 1:
        return True
    if local_world_size is not None and local_world_size > 1:
        return True
    return "RANK" in os.environ or "LOCAL_RANK" in os.environ


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
        "Normalizing visual processor sequence tensors from %s: %s -> common_length=%d",
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


def _is_placeholder_caption_text(text: Any) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    return normalized.startswith("visual caption unavailable")


def _sanitize_caption_text(text: Any) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[*_`]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove common chain-of-thought/task preambles emitted by instruct models.
    preamble_patterns = (
        r"^the user (?:wants|asks|provided|is asking)[^.?!]*[.?!]\s*",
        r"^i (?:need|will|should|can)[^.?!]*[.?!]\s*",
        r"^we need to [^.?!]*[.?!]\s*",
        r"^\d+\.\s*analy[sz]e the images?\s*:\s*",
        r"^\d+\.\s*analy[sz]e the frames?\s*:\s*",
        r"^analy[sz]e the images?\s*:\s*",
        r"^analy[sz]e the frames?\s*:\s*",
        r"^caption\s*:\s*",
        r"^answer\s*:\s*",
    )
    changed = True
    while changed:
        changed = False
        for pattern in preamble_patterns:
            new_cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
            if new_cleaned != cleaned:
                cleaned = new_cleaned.strip()
                changed = True

    # Convert frame-by-frame bullets into plain visual statements when possible.
    cleaned = re.sub(
        r"(^|\s)(?:frame|frames|image|images)\s*\d+(?:\s*(?:[-,&]|and)\s*\d+)*\s*:\s*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"(^|\s)(?:the\s+)?(?:image|picture|frame|scene|video)\s+(?:shows|displays|depicts|features|contains)\s+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(^|\s)(?:it|this)\s+(?:shows|displays|depicts|features|contains)\s+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(^|\s)(?:these|they)\s+(?:show|display|depict|feature|contain)\s+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(^|\s)[\-]\s+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n-*:;")
    return cleaned


def _is_low_information_caption_text(text: Any) -> bool:
    normalized = re.sub(r"\s+", " ", _sanitize_caption_text(text)).strip().lower()
    if not normalized:
        return True
    if _is_placeholder_caption_text(normalized):
        return True

    # Qwen can occasionally answer the instruction itself instead of describing
    # the frames.  Treat these prompt-leak responses as invalid so the caller
    # retries the segment with the single-segment fallback path.
    if re.search(
        r"^i (?:need|want|should|must) to (?:avoid|not)\b|"
        r"^(?:as an ai|i (?:cannot|can't|can not))\b",
        normalized,
    ):
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


def _compact_segment_caption(
    text: Any,
    *,
    max_sentences: int = _CAPTION_MAX_SENTENCES,
    max_words: int = _CAPTION_MAX_WORDS,
    max_chars: int = _CAPTION_MAX_CHARS,
) -> str:
    normalized = _sanitize_caption_text(text)
    normalized = re.sub(r"\s+", " ", str(normalized or "").strip())
    if not normalized:
        return ""
    if _is_low_information_caption_text(normalized):
        return ""

    sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", normalized) if piece.strip()]
    compact = " ".join(sentences[:max_sentences]) if sentences else normalized
    if not compact:
        compact = normalized

    words = compact.split()
    if max_words > 0 and len(words) > max_words:
        compact = " ".join(words[:max_words]).rstrip(",;:")
        if compact and compact[-1] not in ".!?":
            compact += "."

    if max_chars > 0 and len(compact) > max_chars:
        shortened = compact[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
        compact = (shortened or compact[:max_chars]).rstrip(" ,;:")
        if compact and compact[-1] not in ".!?":
            compact += "..."
    return compact.strip()


def _looks_like_truncated_video_qa_answer(text: Any) -> bool:
    return looks_like_truncated_video_clip_qa_answer(text)

try:
    import transformers  # type: ignore
    from transformers import (  # type: ignore
        AutoProcessor,
        AutoModel,
        AutoModelForCausalLM,
        AutoConfig,
        PretrainedConfig,
    )
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # type: ignore
    except Exception:  # pragma: no cover
        CONFIG_MAPPING = None  # type: ignore
except ImportError:  # pragma: no cover
    transformers = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoConfig = None  # type: ignore
    PretrainedConfig = None  # type: ignore
    CONFIG_MAPPING = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from modelscope import AutoProcessor as ModelscopeAutoProcessor  # type: ignore
    from modelscope import Qwen3VLForConditionalGeneration as ModelscopeQwen3VL  # type: ignore
except Exception:  # pragma: no cover
    ModelscopeAutoProcessor = None  # type: ignore
    ModelscopeQwen3VL = None  # type: ignore


QWEN3_VL_CONFIG_CLASS = None
if PretrainedConfig is not None:
    class Qwen3VLConfig(PretrainedConfig):  # type: ignore[misc]
        model_type = "qwen3_vl"
        keys_to_ignore_at_inference = ["past_key_values"]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

    QWEN3_VL_CONFIG_CLASS = Qwen3VLConfig


def _register_qwen3_config_if_missing() -> None:
    """Register a lightweight Qwen3-VL config so older transformers builds can load the checkpoints."""
    if AutoConfig is None or QWEN3_VL_CONFIG_CLASS is None:
        return

    # 1. Try standard AutoConfig registration
    try:
        if hasattr(AutoConfig, "register"):
            AutoConfig.register("qwen3_vl", QWEN3_VL_CONFIG_CLASS)
    except (ValueError, AttributeError):
        pass
    except Exception as e:
        logger.debug(f"AutoConfig.register failed: {e}")

    # 2. Try direct injection into CONFIG_MAPPING
    if CONFIG_MAPPING is not None:
        # Try registering via method if available
        if hasattr(CONFIG_MAPPING, "register"):
            try:
                CONFIG_MAPPING.register("qwen3_vl", QWEN3_VL_CONFIG_CLASS)
            except Exception:
                pass
        
        # Try direct assignment (for dict-like mappings)
        try:
            CONFIG_MAPPING["qwen3_vl"] = QWEN3_VL_CONFIG_CLASS
        except Exception:
            # Some mappings might be immutable or fail on assignment
            pass

    # 3. Verify registration
    try:
        # Force check if it's resolvable
        if "qwen3_vl" in CONFIG_MAPPING:
            return
    except Exception:
        pass
        
    # 4. Last resort: monkey-patch CONFIG_MAPPING getitem if possible? 
    # (Too risky, sticking to registration)


_register_qwen3_config_if_missing()


def _patch_transformers_flash_attention_compat() -> None:
    """
    Patch symbol compatibility for remote-code models that still import
    `LlamaFlashAttention2` from older transformers internals.
    """
    if transformers is None:
        return

    try:
        from transformers.models.llama import modeling_llama as llama_modeling  # type: ignore
    except Exception:
        return

    if hasattr(llama_modeling, "LlamaFlashAttention2"):
        return

    fallback_cls = (
        getattr(llama_modeling, "LlamaSdpaAttention", None)
        or getattr(llama_modeling, "LlamaAttention", None)
    )
    if fallback_cls is None:
        return

    setattr(llama_modeling, "LlamaFlashAttention2", fallback_cls)
    logger.warning(
        "Applied transformers compatibility shim: aliased LlamaFlashAttention2 to %s",
        getattr(fallback_cls, "__name__", str(fallback_cls)),
    )


def _disable_qwen_fused_kernels_if_requested() -> None:
    """Disable Qwen3.5 Triton/CUDA fast paths on runtimes with unstable JIT support.

    Qwen3.5's native Transformers implementation opportunistically imports
    ``causal_conv1d`` and ``flash-linear-attention``.  The latter uses Triton
    kernels for the gated-delta layers, while the former and the fused gated
    RMSNorm are also compiled extensions.  On the Orion CUDA proxy these
    kernels can abort the whole process inside LLVM instead of raising a
    catchable Python exception.  Keep this opt-in so regular CUDA installations
    retain the faster path.
    """
    disable_qwen35_fused_kernels_if_requested(transformers_module=globals().get("transformers"))


class VisualProcessor:
    """Handles visual captioning and feature extraction for video segments."""

    CAPTION_CACHE_VERSION = 5
    _RUNTIME_CACHE: ClassVar[Dict[Tuple[str, str, str, str], Dict[str, Any]]] = {}
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: str = "",  # Empty string to use settings default
        device: str = None,  # None = auto-detect
        max_slice_nums: int = 2,
        require_local: bool = True,
        accelerator_lock: Any = None,
    ):
        """
        Initialize visual processor.
        
        Args:
            model_name: Vision-language model name
            model_path: Local path to model (if empty, uses settings.qwen_vl_model_path)
            device: Device to run on ('cuda', 'npu', 'cpu'). Auto-detect if None.
            max_slice_nums: Maximum image slices for processing (kept for compatibility)
            require_local: If True, fail fast when local checkpoint is missing instead of downloading
        """
        from config.settings import settings
        from config.device_config import detect_device, initialize_device
        
        # Auto-detect device if not specified
        if device is None:
            device_type, _ = detect_device()
            device = device_type
        
        # Validate device for transformers/QWen compatibility
        # Qwen models support: cuda, npu (via torch_npu), cpu
        if device not in ("cuda", "npu", "cpu"):
            import logging
            logging.warning(f"Unknown device '{device}', falling back to CPU")
            device = "cpu"
        
        # Initialize device environment
        initialize_device(device if device != "cpu" else None)
        
        resolved_model_name = model_name or model_path or settings.qwen_vl_model_path
        self.model_name = resolved_model_name
        resolved_default_path = (
            settings.ocr_model_path
            if "deepseek" in resolved_model_name.lower() and "ocr" in resolved_model_name.lower()
            else settings.qwen_vl_model_path
        )
        self.model_path = model_path or resolved_default_path
        self.device = device
        self.runtime_device = self._resolve_process_local_runtime_device(device)
        disable_cudnn = os.environ.get("TOOLSTAR_DISABLE_CUDNN", "").strip().lower()
        if disable_cudnn in {"1", "true", "yes", "on"}:
            # The Orion CUDA proxy can abort inside its cuDNN runtime-compiled
            # engine path with an LLVM MemoryBuffer assertion.  Keep this
            # opt-in so non-Orion training/inference entry points retain the
            # normal cuDNN path.
            torch.backends.cudnn.enabled = False
            os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
            logger.warning(
                "Disabled cuDNN for the visual caption runtime via TOOLSTAR_DISABLE_CUDNN."
            )
        _disable_qwen_fused_kernels_if_requested()
        _, detected_device_count = detect_device()
        self.accelerator_count = detected_device_count or 0
        valid_device_maps = ("auto", "balanced", "sequential")
        caption_multi_device_enabled = bool(getattr(settings, "caption_multi_device_enabled", False))
        configured_device_map = str(getattr(settings, "caption_device_map", "") or "").strip().lower()
        requested_device_map = configured_device_map if configured_device_map in valid_device_maps else ""
        if configured_device_map and configured_device_map not in valid_device_maps:
            logger.warning(
                "Ignoring invalid settings.caption_device_map=%s for caption runtime. Expected one of %s.",
                configured_device_map,
                valid_device_maps,
            )
        self._requested_device_map = requested_device_map if requested_device_map in valid_device_maps else None
        requested_attn_impl = os.environ.get("TOOLSTAR_ATTN_IMPLEMENTATION", "").strip().lower()
        self._requested_attn_implementation = (
            requested_attn_impl if requested_attn_impl in ("eager", "sdpa", "flash_attention_2") else None
        )
        self.attn_implementation = self._requested_attn_implementation
        self._distributed_worker_context = _is_distributed_worker_context()
        if requested_device_map in valid_device_maps:
            self.device_map = requested_device_map
        elif getattr(settings, "force_single_device", False):
            self.device_map = None
        elif self._distributed_worker_context and not getattr(
            settings,
            "caption_multi_device_in_distributed",
            True,
        ):
            # Keep this escape hatch for runs where captioning overlaps too heavily
            # with planner rollout on the same visible accelerators.
            self.device_map = None
        elif self.device == "npu":
            self.device_map = None
        elif (
            self.device != "cpu"
            and self.accelerator_count > 1
            and caption_multi_device_enabled
        ):
            self.device_map = "balanced"
        else:
            self.device_map = None
        self._model = None
        self._processor = None
        self._is_qwen3_model: Optional[bool] = None
        self._loaded_config_dict: Optional[Dict[str, Any]] = None
        self._logged_qwen_multimodal_mode: Optional[str] = None
        self._process_vision_info = None
        self._active_cache_key: Optional[Tuple[str, str, str, str]] = None
        self._unsupported_optional_generation_input_keys: Set[str] = set()
        self._last_qwen_load_error: Optional[str] = None
        self._accelerator_context_poisoned = False
        self.require_local = require_local
        self._accelerator_lock = accelerator_lock
        if self._distributed_worker_context and self.device != "cpu":
            logger.info(
                "Caption runtime detected distributed worker context; using process-local device=%s with device_map=%s.",
                self.runtime_device,
                self.device_map,
            )

    @classmethod
    def from_loaded_llm_provider(
        cls,
        llm_provider: Any,
        *,
        model_name: Optional[str] = None,
        model_path: str = "",
    ) -> "VisualProcessor":
        provider_model = getattr(llm_provider, "model", None)
        provider_processor = getattr(llm_provider, "processor", None)
        if provider_model is None or provider_processor is None:
            raise RuntimeError("Loaded LLM provider does not expose a multimodal model+processor runtime.")

        resolved_model_path = model_path or str(getattr(llm_provider, "model_name", "") or model_name or "")
        resolved_model_name = model_name or resolved_model_path
        processor = cls(
            model_name=resolved_model_name,
            model_path=resolved_model_path,
            device=str(getattr(llm_provider, "device", "cpu") or "cpu"),
            require_local=True,
            accelerator_lock=getattr(llm_provider, "_accelerator_lock", None),
        )
        processor.runtime_device = str(
            getattr(llm_provider, "runtime_device", processor.runtime_device) or processor.runtime_device
        )
        processor.accelerator_count = int(
            getattr(llm_provider, "accelerator_count", processor.accelerator_count) or processor.accelerator_count
        )
        processor.device_map = getattr(llm_provider, "device_map", processor.device_map)
        processor._requested_device_map = getattr(
            llm_provider,
            "_requested_device_map",
            processor._requested_device_map,
        )
        processor.attn_implementation = getattr(
            llm_provider,
            "attn_implementation",
            processor.attn_implementation,
        )
        processor._requested_attn_implementation = getattr(
            llm_provider,
            "_requested_attn_implementation",
            processor._requested_attn_implementation,
        )
        processor._model = provider_model
        processor._processor = provider_processor
        provider_config_dict = getattr(llm_provider, "_model_config_dict", None)
        processor._loaded_config_dict = (
            dict(provider_config_dict)
            if isinstance(provider_config_dict, dict)
            else None
        )
        processor._is_qwen3_model = getattr(llm_provider, "_is_qwen3_model", None)
        processor._process_vision_info = getattr(llm_provider, "_process_vision_info", None)
        provider_unsupported_keys = getattr(llm_provider, "_unsupported_optional_generation_input_keys", None)
        if isinstance(provider_unsupported_keys, (list, tuple, set, frozenset)):
            processor._unsupported_optional_generation_input_keys = {
                str(key) for key in provider_unsupported_keys if str(key).strip()
            }
        processor._shared_runtime_provider = llm_provider
        return processor

    @staticmethod
    def _resolve_process_local_runtime_device(device: str) -> str:
        if device == "cpu":
            return "cpu"

        if device == "cuda":
            try:
                if torch.cuda.is_available():
                    visible_count = int(torch.cuda.device_count())
                    local_rank = _read_nonnegative_int_env("LOCAL_RANK")
                    if local_rank is not None and 0 <= local_rank < visible_count:
                        return f"cuda:{local_rank}"
                    current_device = int(torch.cuda.current_device())
                    if 0 <= current_device < visible_count:
                        return f"cuda:{current_device}"
            except Exception as exc:
                logger.debug("Failed to resolve process-local CUDA device: %s", exc)
            return "cuda:0"

        if device == "npu":
            local_rank = _read_nonnegative_int_env("LOCAL_RANK")
            if local_rank is not None:
                return f"npu:{local_rank}"
            try:
                current_device = getattr(torch.npu, "current_device", None) if hasattr(torch, "npu") else None
                if callable(current_device):
                    return f"npu:{int(current_device())}"
            except Exception as exc:
                logger.debug("Failed to resolve process-local NPU device: %s", exc)
            return "npu:0"

        from config.device_config import get_device_string

        return get_device_string(device, 0)

    def _resolve_input_device(self) -> str:
        """Resolve the proper device for processor inputs under model sharding."""
        if self._model is None:
            return self.runtime_device
        hf_map = getattr(self._model, "hf_device_map", None)
        if isinstance(hf_map, dict) and hf_map:
            for mapped_device in hf_map.values():
                resolved = self._normalize_hf_mapped_device(
                    mapped_device,
                    device_type=self.device,
                )
                if resolved:
                    return resolved
        return self.runtime_device

    @staticmethod
    def _normalize_hf_mapped_device(mapped_device: Any, *, device_type: Optional[str] = None) -> Optional[str]:
        if isinstance(mapped_device, int):
            normalized_type = str(device_type or "cuda").strip().lower() or "cuda"
            return f"{normalized_type}:{mapped_device}"
        if isinstance(mapped_device, str) and mapped_device not in ("cpu", "disk"):
            return mapped_device
        return None

    def _resolve_mapped_module_device(self, prefixes: Tuple[str, ...]) -> Optional[str]:
        if self._model is None:
            return None
        hf_map = getattr(self._model, "hf_device_map", None)
        if not isinstance(hf_map, dict) or not hf_map:
            return None

        normalized_prefixes = tuple(
            str(prefix).strip()
            for prefix in prefixes
            if str(prefix).strip()
        )
        for prefix in normalized_prefixes:
            for module_name, mapped_device in hf_map.items():
                normalized_name = str(module_name or "").strip()
                if normalized_name == prefix or normalized_name.startswith(prefix + "."):
                    resolved = self._normalize_hf_mapped_device(
                        mapped_device,
                        device_type=self.device,
                    )
                    if resolved:
                        return resolved

        for mapped_device in hf_map.values():
            resolved = self._normalize_hf_mapped_device(
                mapped_device,
                device_type=self.device,
            )
            if resolved:
                return resolved
        return None

    def _resolve_visual_input_device(self) -> str:
        resolved = self._resolve_mapped_module_device(
            (
                "model.visual",
                "visual",
            )
        )
        return resolved or self.runtime_device

    def _generate_with_accelerator_lock(self, inputs: Dict[str, Any], generation_kwargs: Dict[str, Any]) -> Any:
        if self._model is None:
            raise RuntimeError("Caption model is not loaded.")
        lock_context = self._accelerator_lock if self._accelerator_lock is not None else nullcontext()
        with lock_context:
            try:
                return self._model.generate(**inputs, **generation_kwargs)
            except Exception as exc:
                self._mark_accelerator_context_poisoned(exc)
                raise

    @staticmethod
    def _exception_chain_contains_fatal_accelerator_error(error: BaseException) -> bool:
        current: Optional[BaseException] = error
        visited: Set[int] = set()
        while current is not None and id(current) not in visited:
            if is_accelerator_fatal_error(current):
                return True
            visited.add(id(current))
            next_error = getattr(current, "__cause__", None)
            if next_error is None:
                next_error = getattr(current, "__context__", None)
            current = next_error
        return False

    def _mark_accelerator_context_poisoned(self, error: BaseException) -> None:
        if not self._exception_chain_contains_fatal_accelerator_error(error):
            return
        if not getattr(self, "_accelerator_context_poisoned", False):
            logger.error(
                "Caption accelerator context marked poisoned after fatal generation error: %s",
                error,
            )
        self._accelerator_context_poisoned = True
        active_cache_key = getattr(self, "_active_cache_key", None)
        if active_cache_key is not None:
            self._RUNTIME_CACHE.pop(active_cache_key, None)

    def _should_apply_multimodal_generation_stability_guard(self, inputs: Dict[str, Any]) -> bool:
        if self.device != "cuda" or self.accelerator_count <= 1 or self.device_map is None:
            return False
        if not isinstance(inputs, dict):
            return False
        return any(
            torch.is_tensor(inputs.get(key)) and inputs.get(key).numel() > 0
            for key in _VISUAL_MULTIMODAL_TENSOR_KEYS
        )

    @staticmethod
    def _apply_caption_multimodal_generation_stability_guard(
        generation_kwargs: Dict[str, Any],
    ) -> Tuple[bool, bool]:
        sampled_decode_requested = bool(generation_kwargs.get("do_sample", False))
        generation_kwargs["remove_invalid_values"] = True
        generation_kwargs["renormalize_logits"] = True
        generation_kwargs["synced_gpus"] = False
        if sampled_decode_requested:
            generation_kwargs["do_sample"] = False
            for key in ("temperature", "top_p", "top_k", "min_p"):
                generation_kwargs.pop(key, None)
        return sampled_decode_requested, bool(generation_kwargs.get("use_cache", True))

    def _sanitize_model_generation_config(self) -> None:
        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is None:
            return
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        if hasattr(generation_config, "min_p"):
            generation_config.min_p = None
        generation_config.do_sample = None
        if hasattr(generation_config, "repetition_penalty"):
            generation_config.repetition_penalty = None

    def _resolve_model_identifier(self) -> str:
        """
        Prefer a local checkpoint path if it exists, otherwise fall back to the model name.
        """
        if self.model_path and os.path.exists(self.model_path):
            return str(Path(self.model_path).expanduser().resolve())
        if self.require_local:
            raise RuntimeError(
                f"Local checkpoint not found at {self.model_path}. "
                "require_local=True prevents remote download. Please place Qwen3-VL weights locally."
            )
        return self.model_name

    def _get_generation_config(self) -> Dict[str, Any]:
        from config.settings import settings

        model_identifier = str(self.model_path or self.model_name or "").lower()
        explicit_ocr = "deepseek" in model_identifier and "ocr" in model_identifier
        is_ocr_model = (
            explicit_ocr
            or (
                str(self.model_path or "") == settings.ocr_model_path
                and settings.ocr_model_path != settings.qwen_vl_model_path
            )
        )
        return (
            settings.ocr_generation_config
            if is_ocr_model
            else settings.qwen_vl_generation_config
        )

    def _detect_qwen3_model(self, config_dict: Optional[Dict[str, Any]]) -> bool:
        if self._is_qwen3_model is not None:
            return self._is_qwen3_model

        lowered_name = (self.model_name or "").lower()
        is_qwen3 = any(token in lowered_name for token in ("qwen3.5", "qwen3_5", "qwen3-vl"))
        if not is_qwen3 and config_dict:
            model_type = str(config_dict.get("model_type", "")).lower()
            architectures = [str(arch).lower() for arch in (config_dict.get("architectures", []) or [])]
            is_qwen3 = model_type in {"qwen3_vl", "qwen3_5"} or any(
                token in arch for arch in architectures for token in ("qwen3vl", "qwen3_5")
            )

        self._is_qwen3_model = is_qwen3
        return is_qwen3

    def _is_multimodal_config(self, config_dict: Optional[Dict[str, Any]]) -> bool:
        if not config_dict:
            return False

        model_type = str(config_dict.get("model_type", "")).lower()
        if any(key in model_type for key in ("vl", "vision", "mllama", "multimodal")):
            return True

        if any(
            key in config_dict
            for key in (
                "vision_config",
                "visual",
                "image_token_id",
                "image_token_index",
                "image_start_id",
                "image_end_id",
                "video_token_id",
                "mm_processor_kwargs",
                "image_processor_type",
            )
        ):
            return True

        auto_map = config_dict.get("auto_map", {}) or {}
        auto_map_text = (
            " ".join(str(value).lower() for value in auto_map.values())
            if isinstance(auto_map, dict)
            else str(auto_map).lower()
        )
        if any(key in auto_map_text for key in ("vision", "image", "vl", "multimodal")):
            return True

        architectures = config_dict.get("architectures", []) or []
        for arch in architectures:
            arch_name = str(arch).lower()
            if any(key in arch_name for key in ("vl", "vision", "image", "multimodal")):
                return True
        return False

    def _select_model_class(self, config_dict: Optional[Dict[str, Any]]):
        if transformers is None:
            return AutoModelForCausalLM

        architectures = (config_dict or {}).get("architectures", []) or []
        model_type = str((config_dict or {}).get("model_type", "")).lower()
        if self._is_multimodal_config(config_dict):
            for arch in (config_dict or {}).get("architectures", []) or []:
                if hasattr(transformers, arch):
                    logger.info("Using multimodal/conditional architecture class from config: %s", arch)
                    return getattr(transformers, arch)

            for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
                model_cls = getattr(transformers, cls_name, None)
                if model_cls is not None:
                    logger.info("Using multimodal auto class: %s", cls_name)
                    return model_cls

        return AutoModelForCausalLM

    def _select_qwen3_transformers_class_names(self, config_dict: Optional[Dict[str, Any]]) -> List[str]:
        """Return exact Qwen3/Qwen3.5 model classes allowed for this checkpoint."""
        config = config_dict or {}
        model_type = str(config.get("model_type", "")).strip().lower()
        architectures = [str(arch).strip() for arch in (config.get("architectures", []) or []) if str(arch).strip()]
        if model_type == "qwen3_5":
            return [arch for arch in architectures if arch.startswith("Qwen3_5")] or ["Qwen3_5ForConditionalGeneration"]
        if model_type == "qwen3_vl":
            return [arch for arch in architectures if arch.startswith("Qwen3VL")] or ["Qwen3VLForConditionalGeneration"]
        return architectures or ["Qwen3_5ForConditionalGeneration", "Qwen3VLForConditionalGeneration"]

    def _assert_qwen_loaded_class_matches_config(self) -> None:
        if self._model is None:
            return
        config_dict = self._loaded_config_dict or {}
        model_type = str(config_dict.get("model_type", "")).strip().lower()
        if model_type not in {"qwen3_5", "qwen3_vl"}:
            return

        loaded_class = type(self._model).__name__
        expected_class_names = self._select_qwen3_transformers_class_names(config_dict)
        if loaded_class in expected_class_names:
            return
        if model_type == "qwen3_5" and loaded_class.startswith("Qwen3VL"):
            raise RuntimeError(
                "Refusing to use a Qwen3-VL model class for a qwen3_5 checkpoint. "
                f"Loaded class={loaded_class}; expected one of {expected_class_names}; "
                f"model_path={self.model_path}. Loading this mismatched architecture creates missing/mismatched "
                "weights and can trigger CUDA illegal memory access during generation."
            )
        if model_type == "qwen3_vl" and loaded_class.startswith("Qwen3_5"):
            raise RuntimeError(
                "Refusing to use a Qwen3.5 model class for a qwen3_vl checkpoint. "
                f"Loaded class={loaded_class}; expected one of {expected_class_names}; model_path={self.model_path}."
            )

    def _is_qwen_family_model(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        resolved_config = config_dict or self._loaded_config_dict or {}
        model_type = str(resolved_config.get("model_type", "")).lower()
        architectures = resolved_config.get("architectures", []) or []
        model_name_lower = str(self.model_name or "").lower()
        model_path_lower = str(self.model_path or "").lower()

        return (
            "qwen" in model_type
            or any("qwen" in str(arch).lower() for arch in architectures)
            or "qwen" in model_name_lower
            or "qwen" in model_path_lower
        )

    def _processor_supports_images(self) -> bool:
        if self._processor is None:
            return False

        if getattr(self._processor, "image_processor", None) is not None:
            return True

        processor_call = getattr(self._processor, "__call__", None)
        if not callable(processor_call):
            return False

        try:
            processor_signature = inspect.signature(processor_call)
        except Exception:
            return True

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in processor_signature.parameters.values()
        ):
            return True

        return "images" in processor_signature.parameters

    def _model_accepts_visual_inputs(self) -> bool:
        if self._model is None:
            return False

        visual_keys = {
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
            "pixel_values_videos",
            "video_grid_thw",
            "images",
        }
        for attr_name in ("forward", "prepare_inputs_for_generation"):
            method = getattr(self._model, attr_name, None)
            if not callable(method):
                continue

            try:
                method_signature = inspect.signature(method)
            except Exception:
                return True

            if any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in method_signature.parameters.values()
            ):
                return True

            method_keys = set(method_signature.parameters.keys())
            if visual_keys & method_keys:
                return True

        return False

    def _supports_multimodal_runtime(self) -> bool:
        config_is_multimodal = self._is_multimodal_config(self._loaded_config_dict)
        return config_is_multimodal or (
            self._processor_supports_images() and self._model_accepts_visual_inputs()
        )

    def _resolve_pad_token_id(self) -> Optional[int]:
        tokenizer = getattr(self._processor, "tokenizer", None) if self._processor is not None else None
        for source in (tokenizer, self._processor, getattr(self._model, "generation_config", None)):
            if source is None:
                continue
            pad_token_id = getattr(source, "pad_token_id", None)
            if pad_token_id is not None:
                return int(pad_token_id)
            eos_token_id = getattr(source, "eos_token_id", None)
            if eos_token_id is not None:
                return int(eos_token_id)
        return None

    def _resolve_process_vision_info(self):
        if self._process_vision_info is not None:
            return self._process_vision_info
        try:
            from qwen_vl_utils.vision_process import process_vision_info  # type: ignore
        except Exception:
            from qwen_vl_utils import process_vision_info  # type: ignore
        self._process_vision_info = process_vision_info
        return self._process_vision_info

    def _iter_model_float_parameters(self, module: Any):
        if module is None:
            return
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return
        try:
            iterator = parameters()
        except Exception:
            return
        for param in iterator:
            if torch.is_tensor(param) and param.is_floating_point():
                yield param

    def _resolve_generation_input_float_dtype(self) -> Optional[torch.dtype]:
        model = self._model
        if model is None:
            return None

        for module in (getattr(model, "visual", None), getattr(model, "vision_model", None), model):
            for param in self._iter_model_float_parameters(module):
                return param.dtype

        try:
            return self._resolve_preferred_torch_dtype()
        except Exception:
            return None

    def _move_inputs_to_generation_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        target_device = self._resolve_input_device()
        target_float_dtype = self._resolve_generation_input_float_dtype()
        normalized_inputs = _normalize_processor_sequence_tensors(
            dict(inputs or {}),
            source="visual_processor.move_inputs",
        )
        for key, value in normalized_inputs.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                if (
                    key in _FLOAT_MULTIMODAL_TENSOR_KEYS
                    and value.is_floating_point()
                    and target_float_dtype is not None
                ):
                    moved[key] = value.to(device=target_device, dtype=target_float_dtype)
                else:
                    moved[key] = value.to(target_device)
            else:
                moved[key] = value
        return moved

    def _summarize_input_tensor_shapes(self, inputs: Any) -> str:
        if not isinstance(inputs, dict):
            return "inputs=unknown"
        parts: List[str] = []
        for key in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        ):
            value = inputs.get(key)
            if torch.is_tensor(value):
                parts.append(f"{key}={tuple(value.shape)}")
            elif value is not None:
                parts.append(f"{key}={type(value).__name__}")
        return ", ".join(parts) if parts else "inputs=no_known_tensors"

    @staticmethod
    def _normalize_video_metadata_item(metadata: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(metadata, dict):
            return None

        normalized: Dict[str, Any] = {}
        for raw_key, value in metadata.items():
            resolved_key = _VIDEO_METADATA_KEY_ALIASES.get(str(raw_key), str(raw_key))
            if resolved_key not in _VIDEO_METADATA_ALLOWED_KEYS or value is None:
                continue
            try:
                if resolved_key == "total_num_frames":
                    parsed_value = int(round(float(value)))
                    if parsed_value <= 0:
                        continue
                    normalized[resolved_key] = parsed_value
                else:
                    normalized[resolved_key] = float(value)
            except (TypeError, ValueError):
                continue

        return normalized or None

    @staticmethod
    def _drop_none_mapping_values(payload: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in dict(payload or {}).items():
            if value is None:
                continue
            cleaned[key] = value
        return cleaned

    def _get_unsupported_optional_generation_input_keys(self) -> Set[str]:
        keys = getattr(self, "_unsupported_optional_generation_input_keys", None)
        if isinstance(keys, set):
            return keys
        if isinstance(keys, (list, tuple, frozenset)):
            normalized = {str(item) for item in keys if str(item).strip()}
        else:
            normalized = set()
        self._unsupported_optional_generation_input_keys = normalized
        return normalized

    def _filter_known_optional_generation_inputs(
        self,
        processor_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        filtered_inputs = dict(processor_inputs or {})
        unsupported_keys = self._get_unsupported_optional_generation_input_keys()
        if not unsupported_keys:
            return filtered_inputs

        for key in tuple(unsupported_keys):
            filtered_inputs.pop(key, None)
        return filtered_inputs

    def _remember_unsupported_optional_generation_inputs(
        self,
        removed_keys: List[str],
        *,
        label: str,
        input_summary: str,
    ) -> None:
        unsupported_keys = self._get_unsupported_optional_generation_input_keys()
        new_keys = [key for key in removed_keys if key not in unsupported_keys]
        if not new_keys:
            return

        unsupported_keys.update(new_keys)
        logger.info(
            "Direct video QA learned to drop unstable optional processor inputs %s after %s. Input summary: %s",
            new_keys,
            label,
            input_summary,
        )
        if self._active_cache_key is not None:
            self._cache_loaded_runtime(self._active_cache_key)

    def _get_vision_pixel_bounds(self) -> Tuple[int, int]:
        from config.settings import settings

        default_min_pixels = 256 * 256
        default_max_pixels = 1280 * 28 * 28
        min_pixels = max(
            _MIN_FRAME_EDGE * _MIN_FRAME_EDGE,
            int(getattr(settings, "qwen_vl_image_min_pixels", default_min_pixels) or default_min_pixels),
        )
        max_pixels = max(
            min_pixels,
            int(getattr(settings, "qwen_vl_image_max_pixels", default_max_pixels) or default_max_pixels),
        )
        return min_pixels, max_pixels

    def _resolve_preferred_torch_dtype(self) -> torch.dtype:
        """Resolve a stable inference dtype that avoids float32 fallback on accelerators."""
        if self.device == "cpu":
            return torch.float32

        try:
            if self.device == "cuda":
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            if self.device == "npu":
                torch_npu = getattr(torch, "npu", None)
                if torch_npu is not None and hasattr(torch_npu, "is_bf16_supported"):
                    if torch_npu.is_bf16_supported():
                        return torch.bfloat16
                return torch.float16
        except Exception as exc:
            logger.debug("Failed to probe preferred caption dtype: %s", exc)

        return torch.float16

    @staticmethod
    def _sample_frames_evenly(frames: List[Image.Image], max_frames: Optional[int]) -> List[Image.Image]:
        if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
            return list(frames)
        sampled_indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int).tolist()
        sampled_indices = list(dict.fromkeys(int(idx) for idx in sampled_indices))
        return [frames[idx] for idx in sampled_indices]

    @staticmethod
    def _resize_frames_to_max_pixels(
        frames: List[Image.Image],
        max_pixels: Optional[int],
    ) -> List[Image.Image]:
        if max_pixels is None or max_pixels <= 0:
            return list(frames)

        resized_frames: List[Image.Image] = []
        for frame in frames:
            width, height = frame.size
            current_pixels = max(1, int(width) * int(height))
            if current_pixels <= max_pixels:
                resized_frames.append(frame)
                continue

            scale = float(max_pixels / float(current_pixels)) ** 0.5
            new_width = max(_MIN_FRAME_EDGE, int(round(width * scale)))
            new_height = max(_MIN_FRAME_EDGE, int(round(height * scale)))
            if new_width >= width and new_height >= height:
                resized_frames.append(frame)
                continue

            resized_frames.append(frame.resize((new_width, new_height), _BICUBIC_RESAMPLE))
        return resized_frames

    @staticmethod
    def _summarize_frame_batch(frames: List[Image.Image]) -> str:
        if not frames:
            return "frames=0"
        sample_sizes = ", ".join(f"{frame.width}x{frame.height}" for frame in frames[:3])
        suffix = ", ..." if len(frames) > 3 else ""
        return f"frames={len(frames)} sizes=[{sample_sizes}{suffix}]"

    @staticmethod
    def _clear_accelerator_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                logger.warning("Skipping torch.cuda.empty_cache() after caption runtime cleanup failure: %s", exc)
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "empty_cache", None)):
            try:
                torch.npu.empty_cache()
            except Exception as exc:
                logger.warning("Skipping torch.npu.empty_cache() after caption runtime cleanup failure: %s", exc)

    @classmethod
    def _is_out_of_memory_error(cls, error: BaseException) -> bool:
        if isinstance(error, torch.OutOfMemoryError):
            return True
        cause = getattr(error, "__cause__", None)
        if cause is not None and cause is not error and cls._is_out_of_memory_error(cause):
            return True
        message = str(error or "").lower()
        return "out of memory" in message or "ran out of memory" in message

    @classmethod
    def _is_accelerator_runtime_error(cls, error: BaseException) -> bool:
        """Identify recoverable local CUDA/cuDNN failures during captioning."""
        message = str(error or "").lower()
        return any(marker in message for marker in ("cudnn", "cuda error", "cuda kernel"))

    def _generate_segment_caption_with_oom_retry(self, frames: List[Image.Image]) -> str:
        if not frames:
            return ""

        _, base_max_pixels = self._get_vision_pixel_bounds()
        last_error: Optional[Exception] = None
        last_invalid_caption = ""
        seen_attempts: set[Tuple[int, int, int, int]] = set()

        for attempt_idx, (label, max_frames, pixel_ratio, max_new_tokens, use_cache) in enumerate(
            _CAPTION_OOM_RETRY_PLANS,
            start=1,
        ):
            attempt_frames = self._sample_frames_evenly(frames, max_frames)
            attempt_max_pixels = max(
                _MIN_FRAME_EDGE * _MIN_FRAME_EDGE,
                int(round(float(base_max_pixels) * float(pixel_ratio))),
            )
            attempt_frames = self._resize_frames_to_max_pixels(attempt_frames, attempt_max_pixels)
            attempt_signature = (len(attempt_frames), attempt_max_pixels, max_new_tokens, int(use_cache))
            if attempt_signature in seen_attempts:
                continue
            seen_attempts.add(attempt_signature)

            try:
                if attempt_idx > 1:
                    logger.warning(
                        "Retrying segment caption after OOM with plan=%s, %s, max_pixels=%s, max_new_tokens=%s, use_cache=%s.",
                        label,
                        self._summarize_frame_batch(attempt_frames),
                        attempt_max_pixels,
                        max_new_tokens,
                        use_cache,
                    )
                caption = self._generate_caption(
                    attempt_frames,
                    "",
                    allow_text_only_fallback=False,
                    max_new_tokens_override=max_new_tokens,
                    use_cache_override=use_cache,
                )
                normalized_caption = _compact_segment_caption(caption)
                if normalized_caption and not _is_low_information_caption_text(normalized_caption):
                    return normalized_caption
                last_invalid_caption = normalized_caption
                logger.warning(
                    "Segment caption attempt %s returned empty or low-information text; retrying with the next plan.",
                    label,
                )
            except Exception as exc:
                if not self._is_out_of_memory_error(exc):
                    raise
                last_error = exc
                logger.warning(
                    "Segment caption attempt %s failed with OOM (%s, device_map=%s).",
                    label,
                    self._summarize_frame_batch(attempt_frames),
                    self.device_map,
                )
                self._clear_accelerator_cache()

        if last_error is not None:
            raise last_error
        return last_invalid_caption

    def _build_local_video_metadata_fallback(
        self,
        video_path: Path,
        *,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        try:
            with VideoFileClip(str(video_path)) as clip:
                clip_fps = getattr(clip, "fps", None)
                duration = getattr(clip, "duration", None)
                if clip_fps is not None:
                    metadata["fps"] = float(clip_fps)
                if duration is not None:
                    metadata["duration"] = float(duration)
                if clip_fps is not None and duration is not None:
                    metadata["total_num_frames"] = max(1, int(round(float(clip_fps) * float(duration))))
        except Exception as exc:
            logger.debug("Failed to derive local video metadata for %s: %s", video_path, exc)

        if "fps" not in metadata and fps is not None:
            metadata["fps"] = float(fps)
        if "total_num_frames" not in metadata and max_frames is not None:
            metadata["total_num_frames"] = int(max_frames)
        return self._normalize_video_metadata_item(metadata) or {}

    @staticmethod
    def _is_missing_video_fps_error(error: BaseException) -> bool:
        message = str(error or "")
        if isinstance(error, KeyError) and any(str(item) == "video_fps" for item in getattr(error, "args", ())):
            return True
        cause = getattr(error, "__cause__", None)
        if cause is not None and cause is not error:
            return VisualProcessor._is_missing_video_fps_error(cause)
        return "video_fps" in message

    def _build_local_video_tensor_payload(
        self,
        video_path: Path,
        *,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        with VideoFileClip(str(video_path)) as video:
            clip_duration = float(getattr(video, "duration", 0.0) or 0.0)
            clip_fps = float(getattr(video, "fps", 0.0) or 0.0)
            if clip_duration <= 0:
                raise RuntimeError(f"Video has invalid duration for local tensor fallback: {clip_duration}")

            target_fps = float(
                fps
                if fps is not None
                else getattr(settings, "video_clip_qa_fps", 2.0) or 2.0
            )
            if target_fps <= 0:
                target_fps = 1.0

            target_limit = int(
                max_frames
                if max_frames is not None
                else getattr(settings, "video_clip_qa_max_frames", 80) or 80
            )
            if target_limit <= 0:
                target_limit = 1

            requested_count = max(1, int(round(clip_duration * target_fps)))
            requested_count = min(requested_count, target_limit)

            frame_interval = (1.0 / clip_fps) if clip_fps > 1e-6 else 1e-2
            safe_end = max(0.0, clip_duration - max(1e-3, frame_interval))
            frame_times = np.linspace(0.0, safe_end, num=requested_count, endpoint=True)
            frames = self.encode_video_frames(video, frame_times)

        normalized_frames: List[Image.Image] = []
        for idx, frame in enumerate(frames):
            normalized = self._normalize_frame_to_rgb(frame, idx)
            if normalized is not None:
                normalized_frames.append(normalized)
        if not normalized_frames:
            raise RuntimeError("Local video tensor fallback could not decode any valid video frames.")

        video_tensor = torch.stack(
            [
                torch.from_numpy(np.ascontiguousarray(np.asarray(frame, dtype=np.uint8).transpose(2, 0, 1)))
                for frame in normalized_frames
            ]
        ).float()
        metadata = self._normalize_video_metadata_item(
            {
                "fps": clip_fps if clip_fps > 0 else target_fps,
                "duration": clip_duration,
                "total_num_frames": max(
                    len(normalized_frames),
                    int(round((clip_fps if clip_fps > 0 else target_fps) * clip_duration)),
                ),
            }
        )
        video_input = (video_tensor, metadata) if metadata else video_tensor
        return None, [video_input], {"do_sample_frames": False}

    def _prepare_direct_video_inputs_via_chat_template(
        self,
        video_path: Path,
        question_text: str,
        *,
        system_prompt: str,
        enable_thinking: bool,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if self._processor is None:
            return None

        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        if not callable(apply_chat_template):
            return None

        def _build_messages(video_value: str) -> List[Dict[str, Any]]:
            messages: List[Dict[str, Any]] = []
            if str(system_prompt or "").strip():
                messages.append({"role": "system", "content": str(system_prompt or "").strip()})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_value},
                        {"type": "text", "text": question_text},
                    ],
                }
            )
            return messages

        message_variants: List[List[Dict[str, Any]]] = [
            _build_messages(str(video_path)),
        ]
        try:
            file_uri = video_path.as_uri()
        except ValueError:
            file_uri = None
        if file_uri:
            message_variants.append(_build_messages(file_uri))

        sampling_variants: List[Dict[str, Any]] = []

        def _append_sampling_variant(candidate: Dict[str, Any]) -> None:
            cleaned = self._drop_none_mapping_values(dict(candidate or {}))
            if any(cleaned == existing for existing in sampling_variants):
                return
            sampling_variants.append(cleaned)

        if fps is not None:
            _append_sampling_variant({"fps": float(fps)})
        if max_frames is not None:
            _append_sampling_variant({"num_frames": int(max_frames)})
        _append_sampling_variant({})

        last_exc: Optional[Exception] = None
        base_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }

        for current_messages in message_variants:
            for sampling_kwargs in sampling_variants:
                attempt_kwargs = [
                    {**sampling_kwargs, "chat_template_kwargs": {"enable_thinking": enable_thinking}},
                    {**sampling_kwargs, "enable_thinking": enable_thinking},
                    dict(sampling_kwargs),
                ]
                seen_attempts: List[Dict[str, Any]] = []
                for extra_kwargs in attempt_kwargs:
                    if any(extra_kwargs == existing for existing in seen_attempts):
                        continue
                    seen_attempts.append(extra_kwargs)
                    try:
                        inputs = apply_chat_template(
                            current_messages,
                            **base_kwargs,
                            **extra_kwargs,
                        )
                    except TypeError as exc:
                        last_exc = exc
                        continue
                    except Exception as exc:
                        last_exc = exc
                        continue

                    if hasattr(inputs, "items"):
                        prepared = dict(inputs.items())
                    elif hasattr(inputs, "to_dict"):
                        prepared = dict(inputs.to_dict())
                    elif isinstance(inputs, dict):
                        prepared = dict(inputs)
                    else:
                        last_exc = RuntimeError(
                            "Direct processor chat template did not return a mapping for video inputs."
                        )
                        continue

                    prepared = self._drop_none_mapping_values(prepared)
                    if "video_grid_thw" not in prepared:
                        last_exc = RuntimeError(
                            "Direct processor video tokenization did not return video_grid_thw."
                        )
                        continue
                    return prepared

        if last_exc is not None:
            logger.warning(
                "Direct processor video tokenization failed for video QA (%s); falling back to qwen_vl_utils path.",
                last_exc,
            )
        return None

    def _load_video_process_vision_payload(
        self,
        qwen_messages: List[Dict[str, Any]],
        *,
        video_path: Optional[Path] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        process_vision_info = self._resolve_process_vision_info()

        def _is_empty_video_payload(payload: Any) -> bool:
            if payload is None:
                return True
            if isinstance(payload, (list, tuple, dict, str, bytes)):
                return len(payload) == 0
            return False

        kwargs = {
            "image_patch_size": 16,
            "return_video_metadata": True,
            "return_video_kwargs": True,
        }
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info([qwen_messages], **kwargs)
            if _is_empty_video_payload(video_inputs):
                raise RuntimeError("process_vision_info returned no video inputs for video QA.")
            return image_inputs, video_inputs, self._drop_none_mapping_values(dict(video_kwargs or {}))
        except StopIteration as exc:
            logger.warning(
                "process_vision_info raised StopIteration for direct video QA; retrying without metadata/video kwargs."
            )
            try:
                image_inputs, video_inputs = process_vision_info([qwen_messages])
            except StopIteration as retry_exc:
                raise RuntimeError(
                    "Multimodal video preprocessing raised StopIteration before model generation."
                ) from retry_exc
            if _is_empty_video_payload(video_inputs):
                raise RuntimeError("process_vision_info returned no video inputs for video QA.")
            return image_inputs, video_inputs, {}
        except TypeError:
            image_inputs, video_inputs = process_vision_info([qwen_messages])
            if _is_empty_video_payload(video_inputs):
                raise RuntimeError("process_vision_info returned no video inputs for video QA.")
            return image_inputs, video_inputs, {}
        except KeyError as exc:
            if video_path is None or not self._is_missing_video_fps_error(exc):
                raise
            logger.warning(
                "process_vision_info hit missing `video_fps` for direct video QA; rebuilding the video payload locally and continuing through the model's video input path."
            )
            return self._build_local_video_tensor_payload(
                video_path,
                fps=fps,
                max_frames=max_frames,
            )

    def _call_video_multimodal_processor(
        self,
        processor_kwargs: Dict[str, Any],
        *,
        require_video_grid: bool,
    ) -> Dict[str, Any]:
        cleaned_kwargs = self._drop_none_mapping_values(dict(processor_kwargs or {}))
        attempts: List[Tuple[str, Dict[str, Any]]] = [("full_payload", cleaned_kwargs)]

        def _append_attempt(label: str, payload: Dict[str, Any]) -> None:
            candidate = self._drop_none_mapping_values(payload)
            if any(candidate == existing_payload for _, existing_payload in attempts):
                return
            attempts.append((label, candidate))

        auxiliary_keys = [
            key for key in cleaned_kwargs.keys()
            if key not in _VIDEO_PROCESSOR_CORE_KWARGS
        ]
        if auxiliary_keys:
            reduced_aux = {
                key: value
                for key, value in cleaned_kwargs.items()
                if key in _VIDEO_PROCESSOR_CORE_KWARGS
            }
            _append_attempt("without_video_aux_kwargs", reduced_aux)

        if "video_metadata" in cleaned_kwargs:
            reduced = dict(cleaned_kwargs)
            reduced.pop("video_metadata", None)
            _append_attempt("without_video_metadata", reduced)

        if any(key in cleaned_kwargs for key in ("fps", "max_frames", "do_sample_frames")):
            reduced_more = dict(attempts[-1][1])
            for key in ("fps", "max_frames", "do_sample_frames"):
                reduced_more.pop(key, None)
            _append_attempt("without_video_sampling_kwargs", reduced_more)

        last_exc: Optional[Exception] = None
        for index, (label, attempt_kwargs) in enumerate(attempts):
            try:
                inputs = self._processor(**attempt_kwargs)
                inputs = _normalize_processor_sequence_tensors(
                    dict(inputs or {}),
                    source=f"visual_processor.video_processor.{label}",
                )
                if require_video_grid and (
                    "video_grid_thw" not in inputs
                    or not torch.is_tensor(inputs["video_grid_thw"])
                    or inputs["video_grid_thw"].numel() == 0
                ):
                    raise RuntimeError(
                        "Multimodal processor did not return video_grid_thw for a video input. "
                        "This usually means the local qwen_vl_utils / transformers stack did not parse the clip correctly."
                    )
                return inputs
            except StopIteration as exc:
                last_exc = exc
                if index + 1 < len(attempts):
                    logger.warning(
                        "Multimodal processor raised StopIteration during %s for direct video QA; retrying with a simpler video payload.",
                        label,
                    )
                    continue
                raise RuntimeError(
                    "Multimodal processor raised StopIteration while preparing direct video inputs."
                ) from exc
            except TypeError as exc:
                last_exc = exc
                if index + 1 < len(attempts):
                    logger.warning(
                        "Multimodal processor rejected %s payload for direct video QA (%s); retrying with a simpler video payload.",
                        label,
                        exc,
                    )
                    continue
                raise
            except AttributeError as exc:
                last_exc = exc
                if index + 1 < len(attempts):
                    logger.warning(
                        "Multimodal processor crashed during %s for direct video QA (%s); retrying with a simpler video payload.",
                        label,
                        exc,
                    )
                    continue
                raise
            except RuntimeError as exc:
                last_exc = exc
                if index + 1 < len(attempts) and "video_grid_thw" in str(exc):
                    logger.warning(
                        "Multimodal processor returned incomplete %s payload for direct video QA (%s); retrying with a simpler video payload.",
                        label,
                        exc,
                    )
                    continue
                raise

        if isinstance(last_exc, StopIteration):
            raise RuntimeError(
                "Multimodal processor raised StopIteration while preparing direct video inputs."
            ) from last_exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Multimodal processor failed without returning direct video inputs.")

    def _build_tqdm_kwargs(self) -> Dict[str, Any]:
        return {
            "dynamic_ncols": True,
            "leave": True,
            "position": 0,
            "file": sys.stdout,
        }

    def _resolve_config_num_hidden_layers(self, config_dict: Optional[Dict[str, Any]] = None) -> Optional[int]:
        resolved = config_dict or self._loaded_config_dict or {}
        candidates = [
            resolved.get("num_hidden_layers"),
            ((resolved.get("text_config") or {}) if isinstance(resolved.get("text_config"), dict) else {}).get("num_hidden_layers"),
            ((resolved.get("language_config") or {}) if isinstance(resolved.get("language_config"), dict) else {}).get("num_hidden_layers"),
            ((resolved.get("llm_config") or {}) if isinstance(resolved.get("llm_config"), dict) else {}).get("num_hidden_layers"),
        ]
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _build_explicit_multimodal_device_map(
        self,
        model_path: str,
        device_count: int,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        num_layers = self._resolve_config_num_hidden_layers(config_dict)
        if not num_layers or device_count <= 1:
            logger.warning(
                "Cannot build explicit caption multimodal device map (layers=%s, devices=%d). Falling back to balanced.",
                num_layers,
                device_count,
            )
            return "balanced"

        device_map: Dict[str, int] = {
            "model.visual": 0,
            "model.language_model.embed_tokens": 0,
            "model.language_model.rotary_emb": 0,
            "model.language_model.norm": 0,
            "lm_head": 0,
        }

        if num_layers == 1:
            device_map["model.language_model.layers.0"] = 0
            return device_map

        device_map["model.language_model.layers.0"] = 0
        device_map[f"model.language_model.layers.{num_layers - 1}"] = 0

        middle_layers = list(range(1, max(1, num_layers - 1)))
        worker_devices = list(range(1, device_count)) or [0]
        if middle_layers and worker_devices:
            base_count = len(middle_layers) // len(worker_devices)
            remainder = len(middle_layers) % len(worker_devices)
            cursor = 0
            for device_index, dev_id in enumerate(worker_devices):
                count = base_count + (1 if device_index < remainder else 0)
                for _ in range(count):
                    if cursor >= len(middle_layers):
                        break
                    layer_idx = middle_layers[cursor]
                    device_map[f"model.language_model.layers.{layer_idx}"] = dev_id
                    cursor += 1

        logger.info(
            "Explicit caption multimodal CUDA device map: %d layers -> %d devices (visual/embed/head on dev0).",
            num_layers,
            device_count,
        )
        return device_map

    def _finalize_device_map(
        self,
        model_path: str,
        resolved_device_map: Any,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if self._requested_device_map is not None:
            return resolved_device_map

        if (
            resolved_device_map is not None
            and self.device == "cuda"
            and self.accelerator_count > 1
            and self._is_qwen_family_model(config_dict)
        ):
            return self._build_explicit_multimodal_device_map(
                model_path,
                self.accelerator_count,
                config_dict=config_dict,
            )
        return resolved_device_map

    def _can_retry_caption_load_on_multi_device(self) -> bool:
        return (
            self.device == "cuda"
            and self.accelerator_count > 1
            and self.device_map is None
            and self._requested_device_map is None
            and not bool(getattr(settings, "force_single_device", False))
        )

    def _can_retry_video_qa_on_single_device(self) -> bool:
        return (
            self.device == "cuda"
            and self.accelerator_count > 1
            and self.device_map is not None
            and self._requested_device_map is None
        )

    def _can_retry_video_qa_with_eager_attention(self) -> bool:
        return (
            self.device == "cuda"
            and str(self.attn_implementation or "").lower() != "eager"
            and self._requested_attn_implementation is None
        )

    def _retry_video_qa_with_eager_attention(
        self,
        *,
        video_path: str,
        query: str,
        fps: Optional[float],
        max_frames: Optional[int],
        system_prompt: str,
        max_new_tokens: Optional[int],
    ) -> str:
        original_attn_implementation = self.attn_implementation
        logger.warning(
            "Reloading direct video QA runtime with attn_implementation='eager' after multimodal generation instability."
        )
        self.unload_model(clear_runtime_cache=True)
        self.attn_implementation = "eager"
        try:
            return self.answer_question_about_video(
                video_path,
                query,
                fps=fps,
                max_frames=max_frames,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                _allow_single_device_reload_retry=False,
            )
        except Exception:
            self.unload_model(clear_runtime_cache=True)
            self.attn_implementation = original_attn_implementation
            raise

    def _retry_video_qa_on_single_device(
        self,
        *,
        video_path: str,
        query: str,
        fps: Optional[float],
        max_frames: Optional[int],
        system_prompt: str,
        max_new_tokens: Optional[int],
    ) -> str:
        original_device_map = self.device_map
        original_requested_device_map = self._requested_device_map
        original_attn_implementation = self.attn_implementation
        logger.warning(
            "Reloading direct video QA runtime on a single CUDA device after multimodal multi-device instability."
        )
        self.unload_model(clear_runtime_cache=True)
        self.device_map = None
        self._requested_device_map = None
        try:
            return self.answer_question_about_video(
                video_path,
                query,
                fps=fps,
                max_frames=max_frames,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                _allow_single_device_reload_retry=False,
            )
        except Exception as exc:
            if (
                self._can_retry_video_qa_with_eager_attention()
                and "Model.generate raised StopIteration for direct video QA" not in str(exc)
            ):
                try:
                    return self._retry_video_qa_with_eager_attention(
                        video_path=video_path,
                        query=query,
                        fps=fps,
                        max_frames=max_frames,
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception:
                    pass
            self.unload_model(clear_runtime_cache=True)
            self.device_map = original_device_map
            self._requested_device_map = original_requested_device_map
            self.attn_implementation = original_attn_implementation
            raise

    def _get_max_memory_config(self) -> Optional[Dict[int, str]]:
        """Apply optional per-device limits for multi-GPU caption model loading."""
        from config.settings import settings

        if settings.per_device_max_memory_gb is None or self.accelerator_count <= 1:
            return None

        max_memory = {
            idx: f"{settings.per_device_max_memory_gb}GB"
            for idx in range(self.accelerator_count)
        }
        logger.info("Applying caption max memory per device: %s", max_memory)
        return max_memory

    def _runtime_cache_key(self, model_identifier: str) -> Tuple[str, str, str, str]:
        return (
            str(model_identifier),
            str(self.runtime_device),
            str(self.device_map or ""),
            str(self.attn_implementation or ""),
        )

    def _apply_cached_runtime(self, cached_runtime: Dict[str, Any]) -> None:
        self._model = cached_runtime.get("model")
        self._processor = cached_runtime.get("processor")
        cached_config = cached_runtime.get("config_dict")
        self._loaded_config_dict = dict(cached_config) if isinstance(cached_config, dict) else None
        cached_is_qwen3 = cached_runtime.get("is_qwen3_model")
        self._is_qwen3_model = bool(cached_is_qwen3) if cached_is_qwen3 is not None else None
        cached_unsupported_input_keys = cached_runtime.get("unsupported_optional_generation_input_keys")
        if isinstance(cached_unsupported_input_keys, (list, tuple, set, frozenset)):
            self._unsupported_optional_generation_input_keys = {
                str(key) for key in cached_unsupported_input_keys if str(key).strip()
            }
        else:
            self._unsupported_optional_generation_input_keys = set()
        self._logged_qwen_multimodal_mode = None

    def _cache_loaded_runtime(self, cache_key: Tuple[str, str, str, str]) -> None:
        if self._model is None or self._processor is None:
            return
        self._RUNTIME_CACHE[cache_key] = {
            "model": self._model,
            "processor": self._processor,
            "config_dict": dict(self._loaded_config_dict or {}),
            "is_qwen3_model": self._is_qwen3_model,
            "unsupported_optional_generation_input_keys": sorted(
                self._get_unsupported_optional_generation_input_keys()
            ),
        }

    def _log_runtime_capabilities(self) -> None:
        from config.device_config import get_accelerator_snapshot

        model_cls = type(self._model).__name__ if self._model is not None else "None"
        processor_cls = type(self._processor).__name__ if self._processor is not None else "None"
        config_dict = self._loaded_config_dict or {}
        hf_device_map = getattr(self._model, "hf_device_map", None) if self._model is not None else None
        model_param_dtype = "unknown"
        if self._model is not None:
            try:
                model_param_dtype = str(next(self._model.parameters()).dtype)
            except Exception:
                pass
        logger.info(
            "Caption model runtime: model=%s processor=%s device=%s runtime_device=%s device_map=%s hf_device_map=%s param_dtype=%s model_type=%s architectures=%s multimodal_config=%s processor_supports_images=%s model_accepts_visual_inputs=%s",
            model_cls,
            processor_cls,
            self.device,
            self.runtime_device,
            self.device_map,
            hf_device_map,
            model_param_dtype,
            config_dict.get("model_type"),
            config_dict.get("architectures"),
            self._is_multimodal_config(config_dict),
            self._processor_supports_images(),
            self._model_accepts_visual_inputs(),
        )
        logger.info("Caption accelerator snapshot: %s", get_accelerator_snapshot(self.device))

    def unload_model(self, clear_runtime_cache: bool = False) -> None:
        """Release caption model resources so later pipeline stages can reuse VRAM."""
        if self._model is None and self._processor is None:
            return

        if clear_runtime_cache and self._active_cache_key is not None:
            self._RUNTIME_CACHE.pop(self._active_cache_key, None)
            self._active_cache_key = None

        self._model = None
        self._processor = None
        self._loaded_config_dict = None
        self._is_qwen3_model = None
        self._logged_qwen_multimodal_mode = None
        self._unsupported_optional_generation_input_keys = set()
        poisoned_context = bool(getattr(self, "_accelerator_context_poisoned", False))
        self._accelerator_context_poisoned = False
        gc.collect()
        if poisoned_context:
            logger.warning("Skipping caption accelerator cache cleanup because the CUDA/NPU context is poisoned.")
            return
        self._clear_accelerator_cache()

    def _retry_generate_without_unused_kwargs(
        self,
        inputs: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        error: Exception,
    ):
        message = str(error)
        if "not used by the model" not in message:
            raise error

        unused_keys: List[str] = []
        if "[" in message and "]" in message:
            fragment = message.split("[", 1)[1].split("]", 1)[0]
            unused_keys = [item.strip().strip("'\"") for item in fragment.split(",") if item.strip()]
        if not unused_keys:
            raise error

        filtered_inputs = dict(inputs)
        removed_keys = []
        for key in unused_keys:
            if key in filtered_inputs:
                filtered_inputs.pop(key, None)
                removed_keys.append(key)

        if not removed_keys:
            raise error

        critical_visual_keys = {
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "images",
        }
        if critical_visual_keys & set(removed_keys):
            model_cls = type(self._model).__name__ if self._model is not None else "None"
            config_dict = self._loaded_config_dict or {}
            raise RuntimeError(
                "Caption model rejected critical visual inputs "
                f"{sorted(critical_visual_keys & set(removed_keys))}. "
                f"Loaded model class: {model_cls}; model_type={config_dict.get('model_type')}; "
                f"architectures={config_dict.get('architectures')}. "
                "This usually means settings.qwen_vl_model_path points to a text-only checkpoint "
                "or the checkpoint was loaded with the wrong model class."
            ) from error

        logger.warning(
            "Retrying caption generation after dropping unsupported multimodal kwargs: %s",
            removed_keys,
        )
        filtered_inputs = self._move_inputs_to_generation_device(filtered_inputs)
        return self._generate_with_accelerator_lock(filtered_inputs, generation_kwargs)

    def _load_model_with_dtype_fallback(
        self,
        model_cls: Any,
        model_identifier: str,
        load_kwargs: Dict[str, Any],
    ) -> Any:
        preferred_dtype = self._resolve_preferred_torch_dtype()
        preferred_kwargs = dict(load_kwargs)
        preferred_kwargs.pop("dtype", None)
        preferred_kwargs["torch_dtype"] = preferred_dtype

        try:
            return model_cls.from_pretrained(model_identifier, **preferred_kwargs)
        except TypeError as exc:
            if "torch_dtype" not in str(exc):
                raise

            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs["dtype"] = preferred_dtype
            logger.warning(
                "Caption model class %s does not accept torch_dtype; falling back to dtype=%s.",
                getattr(model_cls, "__name__", str(model_cls)),
                preferred_dtype,
            )
            return model_cls.from_pretrained(model_identifier, **fallback_kwargs)

    def _retry_direct_video_generate_without_optional_inputs(
        self,
        processor_inputs: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
    ) -> Tuple[Any, List[str], str]:
        optional_retry_groups: List[Tuple[str, Tuple[str, ...]]] = [
            ("without_mm_token_type_ids", ("mm_token_type_ids",)),
        ]

        for label, keys_to_remove in optional_retry_groups:
            if not any(key in processor_inputs for key in keys_to_remove):
                continue

            trimmed_inputs = dict(processor_inputs)
            removed_keys: List[str] = []
            for key in keys_to_remove:
                if key in trimmed_inputs:
                    trimmed_inputs.pop(key, None)
                    removed_keys.append(key)
            if not removed_keys:
                continue

            logger.info(
                "Retrying direct video QA generation %s after dropping optional processor inputs: %s",
                label,
                removed_keys,
            )
            trimmed_inputs = self._move_inputs_to_generation_device(trimmed_inputs)
            return self._generate_with_accelerator_lock(trimmed_inputs, generation_kwargs), removed_keys, label

        raise StopIteration()

    def _load_qwen3_via_transformers(self, model_identifier: str) -> bool:
        if transformers is None:
            return False

        self._last_qwen_load_error = None

        qwen3_cls = None
        class_names = self._select_qwen3_transformers_class_names(self._loaded_config_dict)
        for class_name in class_names:
            candidate = getattr(transformers, class_name, None)
            if candidate is not None:
                qwen3_cls = candidate
                break

        if qwen3_cls is None or AutoProcessor is None:
            logger.warning(
                "No exact transformers class is available for Qwen checkpoint model_type=%s architectures=%s. "
                "Refusing to fall back across Qwen3.5/Qwen3-VL architecture families.",
                (self._loaded_config_dict or {}).get("model_type"),
                (self._loaded_config_dict or {}).get("architectures"),
            )
            return False

        def _build_load_kwargs() -> Dict[str, Any]:
            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "cache_dir": str(model_identifier),
                "low_cpu_mem_usage": True,
            }
            if self.attn_implementation:
                load_kwargs["attn_implementation"] = self.attn_implementation
            if self.device_map is not None:
                load_kwargs["device_map"] = self.device_map
                if isinstance(self.device_map, str):
                    max_memory = self._get_max_memory_config()
                    if max_memory:
                        load_kwargs["max_memory"] = max_memory
            return load_kwargs

        def _load_once() -> None:
            self._model = self._load_model_with_dtype_fallback(
                qwen3_cls,
                model_identifier,
                _build_load_kwargs(),
            )
            self._assert_qwen_loaded_class_matches_config()
            if self.device_map is None:
                self._model.to(self.runtime_device)
            self._processor = AutoProcessor.from_pretrained(
                model_identifier,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=str(model_identifier),
                min_pixels=256 * 256,
                max_pixels=1280 * 28 * 28,
            )
            self._model.eval()
            self._sanitize_model_generation_config()
            self._log_runtime_capabilities()
            logger.info(
                "Loaded Qwen checkpoint via transformers native implementation: class=%s model_type=%s architectures=%s",
                type(self._model).__name__,
                (self._loaded_config_dict or {}).get("model_type"),
                (self._loaded_config_dict or {}).get("architectures"),
            )

        try:
            _load_once()
            return True
        except Exception as exc:
            if self._is_out_of_memory_error(exc) and self._can_retry_caption_load_on_multi_device():
                try:
                    self._clear_accelerator_cache()
                except Exception:
                    pass
                self._model = None
                self._processor = None
                self.device_map = self._build_explicit_multimodal_device_map(
                    model_identifier,
                    self.accelerator_count,
                    config_dict=self._loaded_config_dict,
                )
                logger.warning(
                    "Caption single-device load ran out of memory; retrying with multi-device device_map=%s.",
                    self.device_map,
                )
                try:
                    _load_once()
                    return True
                except Exception as retry_exc:
                    logger.warning(
                        "Caption multi-device reload after single-device OOM also failed: %s",
                        retry_exc,
                    )
                    exc = retry_exc
            logger.warning(f"Transformers Qwen3-VL loading failed: {exc}")
            self._last_qwen_load_error = str(exc)
            self._model = None
            self._processor = None
            return False

    def _load_qwen3_via_modelscope(self, model_identifier: str) -> bool:
        if ModelscopeQwen3VL is None or ModelscopeAutoProcessor is None:
            return False
        model_type = str((self._loaded_config_dict or {}).get("model_type", "")).strip().lower()
        if model_type == "qwen3_5":
            logger.warning(
                "Skipping ModelScope Qwen3VL fallback for qwen3_5 checkpoint %s; "
                "Qwen3.5 must be loaded with Qwen3_5ForConditionalGeneration.",
                model_identifier,
            )
            return False

        load_kwargs = {
            "dtype": "auto",
        }
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        try:
            self._model = ModelscopeQwen3VL.from_pretrained(
                model_identifier,
                **load_kwargs,
            )
            self._assert_qwen_loaded_class_matches_config()
            if self.device_map is None:
                self._model.to(self.runtime_device)
            min_pixels, max_pixels = self._get_vision_pixel_bounds()
            try:
                self._processor = ModelscopeAutoProcessor.from_pretrained(
                    model_identifier,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            except TypeError:
                # Fallback if ModelScope processor doesn't support these parameters
                logger.info("ModelScope processor doesn't support min/max_pixels, using defaults")
                self._processor = ModelscopeAutoProcessor.from_pretrained(model_identifier)
            self._model.eval()
            self._sanitize_model_generation_config()
            self._log_runtime_capabilities()
            logger.info("Loaded Qwen3-VL via ModelScope implementation.")
            return True
        except Exception as exc:
            logger.warning(f"ModelScope Qwen3-VL loading failed: {exc}")
            self._model = None
            self._processor = None
            return False
    
    def _load_model(self):
        """Lazy load the vision-language model."""
        if self._model is None:
            config_dict = None
            try:
                config_path = os.path.join(self.model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_dict = json.load(f)
                    qwen_classes = [x for x in dir(transformers) if "Qwen" in x] if transformers else []
                    logger.info(f"Available Qwen classes in transformers: {qwen_classes}")
                    logger.info(f"Model architectures: {config_dict.get('architectures')}")
                    logger.info(f"Model type: {config_dict.get('model_type')}")
            except Exception as e:
                logger.warning(f"Failed to read model config: {e}")
            self._loaded_config_dict = config_dict

            model_type = str((config_dict or {}).get("model_type", "")).lower()
            if model_type not in {"qwen3_5"}:
                _patch_transformers_flash_attention_compat()

            model_identifier = self._resolve_model_identifier()
            self.device_map = self._finalize_device_map(
                model_identifier,
                self.device_map,
                config_dict=config_dict,
            )
            if (
                self.device == "cuda"
                and self.accelerator_count > 1
                and self.device_map is None
                and self._requested_device_map is None
            ):
                logger.info(
                    "Caption runtime prefers single-device CUDA inference for latency; multi-device sharding is reserved for explicit overrides or single-device OOM fallback."
                )
            cache_key = self._runtime_cache_key(model_identifier)
            cached_runtime = self._RUNTIME_CACHE.get(cache_key)
            if isinstance(cached_runtime, dict):
                self._apply_cached_runtime(cached_runtime)
                self._active_cache_key = cache_key
                logger.info(
                    "Reusing cached caption runtime: model=%s device=%s device_map=%s",
                    model_identifier,
                    self.runtime_device,
                    self.device_map,
                )
                return
            is_qwen3 = self._detect_qwen3_model(config_dict)

            if is_qwen3:
                if self._load_qwen3_via_transformers(model_identifier):
                    self._active_cache_key = cache_key
                    self._cache_loaded_runtime(cache_key)
                    return
                if model_type != "qwen3_5" and self._load_qwen3_via_modelscope(model_identifier):
                    self._active_cache_key = cache_key
                    self._cache_loaded_runtime(cache_key)
                    return
                load_detail = str(self._last_qwen_load_error or "").strip()
                message = (
                    "Unable to load the local Qwen3/Qwen3.5 checkpoint with an exact architecture class. "
                    "The configured architecture is Qwen3_5ForConditionalGeneration; "
                    "check the underlying runtime error before changing transformers."
                )
                if load_detail:
                    message += f" Underlying load error: {load_detail}"
                raise RuntimeError(message)

            if AutoProcessor is None:
                raise RuntimeError("transformers AutoProcessor not available")

            config = None
            if config_dict and config_dict.get("model_type") == "qwen3_vl" and QWEN3_VL_CONFIG_CLASS is not None:
                logger.info("Detected qwen3_vl model_type, manually loading config to bypass registry.")
                config = QWEN3_VL_CONFIG_CLASS.from_dict(config_dict)

            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "cache_dir": str(model_identifier),
                "low_cpu_mem_usage": True,
            }
            if self.device_map is not None:
                load_kwargs["device_map"] = self.device_map
                if isinstance(self.device_map, str):
                    max_memory = self._get_max_memory_config()
                    if max_memory:
                        load_kwargs["max_memory"] = max_memory
            if config is not None:
                load_kwargs["config"] = config

            model_cls = self._select_model_class(config_dict)
            try:
                self._model = self._load_model_with_dtype_fallback(
                    model_cls,
                    model_identifier,
                    load_kwargs,
                )
                self._assert_qwen_loaded_class_matches_config()
            except Exception as e:
                logger.warning(f"Primary model loader failed: {e}. Attempting fallbacks...")
                fallback_kwargs = {k: v for k, v in load_kwargs.items() if k != "config"}
                model_type = str((config_dict or {}).get("model_type", "")).lower()
                model_name_lower = (self.model_name or "").lower()
                is_qwen_family = ("qwen" in model_type) or ("qwen" in model_name_lower)

                if self._model is None and self._is_multimodal_config(config_dict) and transformers is not None:
                    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
                        model_cls = getattr(transformers, cls_name, None)
                        if model_cls is None:
                            continue
                        try:
                            logger.info("Falling back to multimodal auto class: %s", cls_name)
                            self._model = self._load_model_with_dtype_fallback(
                                model_cls,
                                model_identifier,
                                fallback_kwargs,
                            )
                            self._assert_qwen_loaded_class_matches_config()
                            break
                        except Exception as multimodal_exc:
                            logger.warning("%s fallback failed: %s", cls_name, multimodal_exc)

                if self._model is None and AutoModel is not None:
                    try:
                        logger.info("Falling back to AutoModel with trust_remote_code")
                        self._model = self._load_model_with_dtype_fallback(
                            AutoModel,
                            model_identifier,
                            fallback_kwargs,
                        )
                        self._assert_qwen_loaded_class_matches_config()
                    except Exception as auto_model_exc:
                        logger.warning(f"AutoModel fallback failed: {auto_model_exc}")

                if config_dict and config_dict.get("architectures") and transformers is not None:
                    for arch in config_dict.get("architectures", []):
                        if hasattr(transformers, arch):
                            logger.info(f"Falling back to specific class: {arch}")
                            cls = getattr(transformers, arch)
                            self._model = self._load_model_with_dtype_fallback(
                                cls,
                                model_identifier,
                                fallback_kwargs,
                            )
                            self._assert_qwen_loaded_class_matches_config()
                            break

                if (
                    self._model is None
                    and is_qwen_family
                    and transformers is not None
                    and hasattr(transformers, "Qwen2VLForConditionalGeneration")
                ):
                    logger.info("Falling back to Qwen2VLForConditionalGeneration")
                    from transformers import Qwen2VLForConditionalGeneration
                    self._model = self._load_model_with_dtype_fallback(
                        Qwen2VLForConditionalGeneration,
                        model_identifier,
                        fallback_kwargs,
                    )
                    self._assert_qwen_loaded_class_matches_config()

                if self._model is None and transformers is not None and hasattr(transformers, "AutoModelForVision2Seq"):
                    logger.info("Falling back to AutoModelForVision2Seq")
                    from transformers import AutoModelForVision2Seq
                    self._model = self._load_model_with_dtype_fallback(
                        AutoModelForVision2Seq,
                        model_identifier,
                        fallback_kwargs,
                    )
                    self._assert_qwen_loaded_class_matches_config()

                if self._model is None:
                    raise

            if self.device_map is None:
                self._model.to(self.runtime_device)

            # Initialize processor with vision parameters if applicable
            processor_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "cache_dir": str(model_identifier),
            }
            model_type = str((config_dict or {}).get("model_type", "")).lower()
            if "qwen" in model_type and "vl" in model_type:
                min_pixels, max_pixels = self._get_vision_pixel_bounds()
                processor_kwargs.update({
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                })
            
            self._processor = AutoProcessor.from_pretrained(
                model_identifier,
                **processor_kwargs
            )
            self._model.eval()
            self._sanitize_model_generation_config()
            self._log_runtime_capabilities()
            self._active_cache_key = cache_key
            self._cache_loaded_runtime(cache_key)

    def _ensure_tokenizer_chat_template(self) -> bool:
        """Ensure tokenizer has a usable chat template for apply_chat_template."""
        if self._processor is None:
            return False

        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            return False

        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            return False

        try:
            tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
            logger.info("Injected default tokenizer.chat_template for caption generation.")
            return True
        except Exception as exc:
            logger.warning("Failed to inject default tokenizer.chat_template: %s", exc)
            return False

    def _render_prompt_text(self, messages: List[Dict[str, Any]], num_images: int, query: str) -> str:
        """
        Render prompt text for multimodal generation.
        Prefer processor chat template; fallback to a deterministic plain template when missing.
        """
        if self._processor is None:
            return query

        self._ensure_tokenizer_chat_template()

        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        generation_config = self._get_generation_config()
        thinking_flag = False
        if callable(apply_chat_template):
            try:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": thinking_flag},
                )
            except TypeError:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking_flag,
                )
            except Exception as exc:
                err = str(exc)
                if "tokenizer.chat_template is not set" not in err:
                    raise
                logger.warning(
                    "Processor chat_template missing; using fallback prompt renderer for captioning."
                )

        tokenizer = getattr(self._processor, "tokenizer", None)
        image_token = "<image>"
        if tokenizer is not None:
            image_token = (
                getattr(tokenizer, "image_token", None)
                or getattr(tokenizer, "image_token_id", None)
                or "<image>"
            )
            if not isinstance(image_token, str):
                image_token = "<image>"

        image_prefix = "\n".join([image_token] * max(0, num_images))
        if image_prefix:
            return f"User:\n{image_prefix}\n{query}\nAssistant:\n"
        return f"User:\n{query}\nAssistant:\n"
    
    def encode_video_frames(self, video: VideoFileClip, frame_times: np.ndarray) -> List[Image.Image]:
        """
        Extract and encode frames from video at specified times.
        
        Args:
            video: VideoFileClip object
            frame_times: Array of timestamps to extract frames
            
        Returns:
            List of PIL Images
        """
        frames = []
        video_duration = float(getattr(video, "duration", 0.0) or 0.0)
        video_fps = float(getattr(video, "fps", 0.0) or 0.0)
        safe_max_time = None
        if video_duration > 0:
            frame_interval = (1.0 / video_fps) if video_fps > 1e-6 else 1e-2
            end_margin = max(1e-3, frame_interval)
            safe_max_time = max(0.0, video_duration - end_margin)

        for t in frame_times:
            safe_t = float(t)
            if safe_max_time is not None:
                safe_t = min(max(0.0, safe_t), safe_max_time)
            try:
                frame = video.get_frame(safe_t)
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame.astype('uint8'))
                frames.append(pil_frame)
            except Exception as e:
                logger.warning(f"Error extracting frame at {safe_t}: {e}")
        return frames

    def _normalize_frame_to_rgb(self, frame: Any, idx: int) -> Optional[Image.Image]:
        """Normalize an input frame to a valid RGB PIL image for processor ingestion."""
        if isinstance(frame, Image.Image):
            try:
                return frame if frame.mode == "RGB" else frame.convert("RGB")
            except Exception as err:
                logger.warning(f"Failed to normalize PIL frame {idx}: {err}")
                return None

        try:
            frame_arr = np.asarray(frame)
        except Exception as err:
            logger.warning(f"Failed to convert frame {idx} to ndarray: {err}")
            return None

        if frame_arr.ndim == 0:
            logger.warning(f"Skipping frame {idx}: scalar input is not a valid image payload.")
            return None

        if frame_arr.ndim == 2:
            frame_arr = np.stack([frame_arr] * 3, axis=-1)
        elif frame_arr.ndim == 3 and frame_arr.shape[-1] == 1:
            frame_arr = np.repeat(frame_arr, 3, axis=-1)
        elif frame_arr.ndim == 3 and frame_arr.shape[-1] == 4:
            frame_arr = frame_arr[..., :3]
        elif frame_arr.ndim != 3 or frame_arr.shape[-1] != 3:
            logger.warning(f"Skipping frame {idx}: unsupported shape {frame_arr.shape}.")
            return None

        if frame_arr.dtype != np.uint8:
            if np.issubdtype(frame_arr.dtype, np.floating):
                frame_arr = np.clip(frame_arr, 0.0, 255.0)
            else:
                frame_arr = np.clip(frame_arr, 0, 255)
            frame_arr = frame_arr.astype(np.uint8)

        frame_arr = np.ascontiguousarray(frame_arr)

        try:
            return Image.fromarray(frame_arr, mode="RGB")
        except Exception as err:
            logger.warning(f"Failed to create PIL image for frame {idx}: {err}")
            return None

    def _pil_to_base64_data_url(self, image: Image.Image) -> str:
        """Convert PIL image to base64 data URL for Qwen vision utils compatibility."""
        rgb_image = image.convert("RGB")
        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _caption_cache_model_fingerprint(self) -> str:
        """Bind cached captions to the checkpoint and caption sampling contract."""
        return "|".join(
            (
                str(self.model_path or self.model_name or "").strip(),
                f"max_frames={_resolve_caption_max_frames()}",
                f"max_new_tokens={_CAPTION_DEFAULT_MAX_NEW_TOKENS}",
            )
        )
    
    def caption_segments(
        self,
        video_name: str,
        video_path: str,
        segment_index2name: Dict[str, str],
        transcripts: Dict[str, str],
        segment_times_info: Dict[str, Dict],
        working_dir: Optional[str] = None,
        reuse_cache: bool = True,
    ) -> Dict[str, str]:
        """
        Generate captions for all video segments.
        
        Args:
            video_name: Name of the video
            video_path: Path to video file
            segment_index2name: Mapping from index to segment name
            transcripts: Transcripts for each segment
            segment_times_info: Timing information for segments
            
        Returns:
            Dictionary mapping segment index to caption
        """
        cache_file = None
        if working_dir:
            cache_dir = Path(working_dir) / "_cache" / video_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "captions.json"

        cached_captions = (
            self._load_cached_captions(cache_file, segment_index2name)
            if cache_file and reuse_cache
            else {}
        )
        captions = {}
        pending_indices: List[str] = []
        for index in segment_index2name:
            if index in cached_captions:
                captions[index] = cached_captions[index]
            else:
                pending_indices.append(index)

        if not pending_indices:
            return captions

        self._load_model()
        if not self._supports_multimodal_runtime():
            config_dict = self._loaded_config_dict or {}
            raise RuntimeError(
                "Preprocessing caption model must be a real multimodal checkpoint. "
                f"Resolved model_path={self.model_path!r}, model_name={self.model_name!r}, "
                f"loaded_model_class={type(self._model).__name__ if self._model is not None else 'None'}, "
                f"model_type={config_dict.get('model_type')!r}, "
                f"architectures={config_dict.get('architectures')!r}. "
                "A text-only checkpoint cannot produce planner-grounding captions."
            )

        try:
            caption_batch_size = max(1, int(os.environ.get("TOOLSTAR_CAPTION_BATCH_SIZE", "1") or 1))
        except (TypeError, ValueError):
            caption_batch_size = 1
        use_caption_batch = caption_batch_size > 1 and self._is_qwen_family_model()
        if use_caption_batch:
            logger.info(
                "Using opt-in visual caption batching: video=%s batch_size=%s max_frames=%s max_new_tokens=%s pending_segments=%s.",
                video_name,
                caption_batch_size,
                _resolve_caption_max_frames(),
                _CAPTION_DEFAULT_MAX_NEW_TOKENS,
                len(pending_indices),
            )
        caption_max_frames = _resolve_caption_max_frames()

        caption_runtime_unavailable = False

        def _caption_one(
            index: str,
            video_frames: List[Image.Image],
            timestamp_text: str,
        ) -> Tuple[str, bool]:
            nonlocal caption_runtime_unavailable
            accelerator_caption_failure = False
            if caption_runtime_unavailable:
                accelerator_caption_failure = True
                caption = f"Visual frames available for direct inspection at {timestamp_text}."
            else:
                try:
                    caption = _compact_segment_caption(
                        self._generate_segment_caption_with_oom_retry(video_frames)
                    )
                except Exception as exc:
                    if not self._is_accelerator_runtime_error(exc):
                        raise
                    caption_runtime_unavailable = True
                    accelerator_caption_failure = True
                    caption = f"Visual frames available for direct inspection at {timestamp_text}."
                    logger.warning(
                        "Caption accelerator failed for video=%s segment=%s; preserving frame evidence "
                        "and continuing without semantic captions for the remaining segments: %s",
                        video_name,
                        index,
                        exc,
                    )
            caption = str(caption or "").strip()
            if (
                not caption
                or _is_low_information_caption_text(caption)
            ) and not accelerator_caption_failure:
                raise RuntimeError(
                    "Preprocessing caption generation returned an invalid visual caption. "
                    f"video={video_name!r}, segment_id={index!r}, model_path={self.model_path!r}. "
                    "Planner grounding requires a real visually grounded caption, not empty, placeholder, or punctuation-only text."
                )
            return caption, accelerator_caption_failure

        with VideoFileClip(video_path) as video:
            progress = tqdm(
                total=len(pending_indices),
                desc=f"Captioning {video_name}",
                **self._build_tqdm_kwargs(),
            )
            try:
                for batch_start in range(0, len(pending_indices), caption_batch_size if use_caption_batch else 1):
                    batch_indices = pending_indices[
                        batch_start : batch_start + (caption_batch_size if use_caption_batch else 1)
                    ]
                    batch_frames: List[List[Image.Image]] = []
                    batch_timestamps: Dict[str, str] = {}
                    for index in batch_indices:
                        frame_times = np.asarray(segment_times_info[index]["frame_times"], dtype=float)
                        if frame_times.size > caption_max_frames:
                            selected_positions = np.linspace(
                                0,
                                frame_times.size - 1,
                                num=caption_max_frames,
                                dtype=int,
                            )
                            frame_times = frame_times[selected_positions]
                        batch_frames.append(self.encode_video_frames(video, frame_times))
                        batch_timestamps[index] = ", ".join(
                            f"{float(time):.3f}s" for time in frame_times
                        )

                    generated_batch: Optional[List[str]] = None
                    if len(batch_indices) > 1 and not caption_runtime_unavailable:
                        try:
                            generated_batch = self._generate_caption_batch(
                                batch_frames,
                                # Keep batched generation aligned with the
                                # single-segment retry plan.  The global Qwen
                                # generation config allows 2048 tokens, which
                                # is useful for QA but unnecessarily expensive
                                # for compact preprocessing captions.
                                max_new_tokens_override=_CAPTION_DEFAULT_MAX_NEW_TOKENS,
                                use_cache_override=True,
                            )
                            if len(generated_batch) != len(batch_indices):
                                raise RuntimeError(
                                    "Caption batch returned an unexpected number of outputs: "
                                    f"expected={len(batch_indices)} actual={len(generated_batch)}"
                                )
                        except Exception as exc:
                            generated_batch = None
                            if self._is_accelerator_runtime_error(exc):
                                caption_runtime_unavailable = True
                                logger.warning(
                                    "Caption batch accelerator failed for video=%s segments=%s; preserving frame evidence "
                                    "and continuing without semantic captions: %s",
                                    video_name,
                                    batch_indices,
                                    exc,
                                )
                            else:
                                logger.warning(
                                    "Caption batch failed for video=%s segments=%s; retrying each segment individually: %s",
                                    video_name,
                                    batch_indices,
                                    exc,
                                )
                            self._clear_accelerator_cache()

                    for item_index, index in enumerate(batch_indices):
                        caption: str
                        accelerator_caption_failure: bool
                        batch_caption = (
                            _compact_segment_caption(generated_batch[item_index])
                            if generated_batch is not None
                            else ""
                        )
                        if batch_caption and not _is_low_information_caption_text(batch_caption):
                            caption = batch_caption
                            accelerator_caption_failure = False
                        else:
                            caption, accelerator_caption_failure = _caption_one(
                                index,
                                batch_frames[item_index],
                                batch_timestamps[index],
                            )
                        if (
                            not caption
                            or _is_low_information_caption_text(caption)
                        ) and not accelerator_caption_failure:
                            raise RuntimeError(
                                "Preprocessing caption generation returned an invalid visual caption. "
                                f"video={video_name!r}, segment_id={index!r}, model_path={self.model_path!r}. "
                                "Planner grounding requires a real visually grounded caption, not empty, placeholder, or punctuation-only text."
                            )
                        captions[index] = caption.strip()
                        # Persist after every segment so a worker interruption can
                        # resume visual captioning from the last completed segment
                        # instead of restarting the whole video.  The cache writer
                        # uses a same-directory temporary file and os.replace, so a
                        # killed worker cannot leave a half-written captions.json.
                        if cache_file:
                            self._save_cached_captions(cache_file, captions, segment_index2name)
                        progress.update(1)
            finally:
                progress.close()

        if cache_file:
            self._save_cached_captions(cache_file, captions, segment_index2name)
        return captions

    def _load_cached_captions(
        self,
        cache_file: Optional[Path],
        segment_index2name: Dict[str, str],
    ) -> Dict[str, str]:
        if cache_file is None or not cache_file.exists():
            return {}
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return {}
            if int(payload.get("cache_version", 0) or 0) != self.CAPTION_CACHE_VERSION:
                return {}
            captions = payload.get("captions")
            cached_names = payload.get("segment_index2name")
            cached_fingerprint = str(payload.get("model_fingerprint", "") or "").strip()
            if not isinstance(captions, dict) or not isinstance(cached_names, dict):
                return {}
            if cached_fingerprint != self._caption_cache_model_fingerprint():
                return {}
            if any(str(cached_names.get(str(idx), "")) != str(name) for idx, name in segment_index2name.items()):
                return {}
            filtered: Dict[str, str] = {}
            for key, value in captions.items():
                text = _compact_segment_caption(value)
                if not text or _is_low_information_caption_text(text):
                    continue
                filtered[str(key)] = text
            return filtered
        except Exception as exc:
            logger.warning("Failed to load cached captions from %s: %s", cache_file, exc)
            return {}

    def _save_cached_captions(
        self,
        cache_file: Path,
        captions: Dict[str, str],
        segment_index2name: Dict[str, str],
    ) -> None:
        payload = {
            "cache_version": self.CAPTION_CACHE_VERSION,
            "model_fingerprint": self._caption_cache_model_fingerprint(),
            "segment_index2name": {str(k): str(v) for k, v in segment_index2name.items()},
            "captions": {str(k): str(v) for k, v in captions.items()},
        }
        try:
            tmp_file = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, cache_file)
        except Exception as exc:
            logger.warning("Failed to write cached captions to %s: %s", cache_file, exc)
    
    def _generate_caption(
        self,
        frames: List[Image.Image],
        transcript: str,
        query_template: Optional[str] = None,
        allow_text_only_fallback: bool = True,
        max_new_tokens_override: Optional[int] = None,
        use_cache_override: Optional[bool] = None,
        _allow_eager_attention_retry: bool = True,
    ) -> str:
        """
        Generate caption for a video segment.
        
        Args:
            frames: List of video frames
            transcript: Deprecated compatibility argument. Preprocessing captions must
                rely only on frames plus the prompt text.
            query_template: Optional custom query template
            
        Returns:
            Generated caption
        """
        if not frames:
            return ""
            
        if query_template is None:
            query = DEFAULT_VISUAL_CAPTION_PROMPT
        else:
            query = query_template.format(transcript="")
        
        # Construct messages for Qwen-VL
        # Ensure all frames are PIL.Image objects in RGB mode
        content = []
        safe_frames: List[Image.Image] = []
        for idx, frame in enumerate(frames):
            normalized = self._normalize_frame_to_rgb(frame, idx)
            if normalized is not None:
                safe_frames.append(normalized)

        if not safe_frames:
            logger.warning("No valid frames after preprocessing; returning empty caption.")
            return ""

        for _ in safe_frames:
            content.append({"type": "image"})
        content.append({"type": "text", "text": query})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        try:
            text = self._render_prompt_text(messages, len(safe_frames), query)

            # Detect if processor supports multimodal inputs (images parameter)
            supports_images = self._processor_supports_images()

            def _prepare_inputs(image_payload):
                if supports_images:
                    try:
                        logger.debug(f"Processing {len(image_payload)} images with processor")
                        processor_inputs = self._processor(
                            text=[text],
                            images=image_payload,
                            padding=True,
                            return_tensors="pt"
                        )
                        logger.debug("Processor successfully created inputs with images")
                    except TypeError as te:
                        if "unexpected keyword argument 'images'" in str(te):
                            logger.warning(
                                "Processor does not accept 'images' parameter; falling back to text-only mode."
                            )
                            processor_inputs = self._processor(
                                text=[text],
                                padding=True,
                                return_tensors="pt"
                            )
                        else:
                            raise
                else:
                    # Processor is text-only (e.g., pure tokenizer)
                    logger.info("Using text-only processor mode (no 'images' parameter detected).")
                    processor_inputs = self._processor(
                        text=[text],
                        padding=True,
                        return_tensors="pt"
                    )
                
                processor_inputs = self._move_inputs_to_generation_device(processor_inputs)
                return processor_inputs

            def _prepare_inputs_via_qwen_utils() -> Optional[Dict[str, Any]]:
                if not self._is_qwen_family_model():
                    return None

                if not self._supports_multimodal_runtime():
                    logger.warning(
                        "Skipping qwen_vl_utils multimodal path because loaded caption runtime does not advertise visual input support."
                    )
                    return None

                try:
                    from qwen_vl_utils import process_vision_info  # type: ignore
                except Exception as import_error:
                    logger.warning(
                        "qwen_vl_utils unavailable, skip official Qwen path and use compatibility fallback: %s",
                        import_error,
                    )
                    return None

                qwen_modes = (
                    ("base64", True),
                    ("pil", False),
                )
                last_qwen_error: Optional[Exception] = None

                for mode_name, use_base64 in qwen_modes:
                    qwen_content: List[Dict[str, Any]] = []
                    for frame in safe_frames:
                        image_value: Any = (
                            self._pil_to_base64_data_url(frame) if use_base64 else frame
                        )
                        qwen_content.append({"type": "image", "image": image_value})
                    qwen_content.append({"type": "text", "text": query})

                    qwen_messages = [{"role": "user", "content": qwen_content}]
                    thinking_flag = False
                    try:
                        generation_config = self._get_generation_config()
                        thinking_flag = bool(generation_config.get("enable_thinking", False))
                        qwen_text = self._processor.apply_chat_template(
                            qwen_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            chat_template_kwargs={"enable_thinking": thinking_flag},
                        )
                    except TypeError:
                        qwen_text = self._processor.apply_chat_template(
                            qwen_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=thinking_flag,
                        )
                        image_inputs, video_inputs = process_vision_info([qwen_messages])
                        processor_inputs = self._processor(
                            text=[qwen_text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        processor_inputs = self._move_inputs_to_generation_device(processor_inputs)
                        if self._logged_qwen_multimodal_mode != mode_name:
                            logger.debug("Prepared multimodal inputs via qwen_vl_utils path (mode=%s).", mode_name)
                            self._logged_qwen_multimodal_mode = mode_name
                        return processor_inputs
                    try:
                        image_inputs, video_inputs = process_vision_info([qwen_messages])
                        processor_inputs = self._processor(
                            text=[qwen_text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        processor_inputs = self._move_inputs_to_generation_device(processor_inputs)
                        if self._logged_qwen_multimodal_mode != mode_name:
                            logger.debug("Prepared multimodal inputs via qwen_vl_utils path (mode=%s).", mode_name)
                            self._logged_qwen_multimodal_mode = mode_name
                        return processor_inputs
                    except Exception as qwen_error:
                        last_qwen_error = qwen_error
                        logger.debug(
                            "Official Qwen vision path failed in mode '%s' (%s): %s",
                            mode_name,
                            type(qwen_error).__name__,
                            qwen_error,
                        )

                if last_qwen_error is not None:
                    logger.warning(
                        "All official Qwen vision modes failed (%s): %s",
                        type(last_qwen_error).__name__,
                        last_qwen_error,
                    )
                return None

            inputs: Optional[Dict[str, Any]] = None

            # Try to process with images
            try:
                inputs = _prepare_inputs_via_qwen_utils()
                if inputs is None and self._supports_multimodal_runtime():
                    last_error: Optional[Exception] = None
                    float32_frames: Optional[List[np.ndarray]] = None
                    for payload_name in ("pil_flat", "pil_batched", "float32_flat", "float32_batched"):
                        if payload_name == "pil_flat":
                            payload = safe_frames
                        elif payload_name == "pil_batched":
                            payload = [safe_frames]
                        else:
                            if float32_frames is None:
                                float32_frames = [np.asarray(frame, dtype=np.float32) for frame in safe_frames]
                            payload = float32_frames if payload_name == "float32_flat" else [float32_frames]
                        try:
                            inputs = _prepare_inputs(payload)
                            if last_error is not None:
                                logger.info(
                                    "Recovered image processing using payload variant '%s'.",
                                    payload_name,
                                )
                            last_error = None
                            break
                        except Exception as payload_error:
                            last_error = payload_error
                            logger.debug(
                                "Image payload variant '%s' rejected (%s): %s",
                                payload_name,
                                type(payload_error).__name__,
                                payload_error,
                            )
                    if inputs is None and last_error is not None:
                        raise last_error
                if inputs is None:
                    raise RuntimeError("No compatible multimodal processor input path produced caption inputs.")
            except Exception as preprocess_error:
                logger.warning(f"Image processing failed: {type(preprocess_error).__name__}: {preprocess_error}")
                if not allow_text_only_fallback:
                    raise RuntimeError(
                        "Multimodal caption preprocessing failed for sampled frames. "
                        "Text-only fallback is disabled because preprocessing captions must stay grounded in visual evidence."
                    ) from preprocess_error
                # Fall back to text-only mode if image processing fails
                logger.info("Falling back to text-only caption generation.")
                try:
                    inputs = self._processor(
                        text=[text],
                        padding=True,
                        return_tensors="pt"
                    )
                    inputs = self._move_inputs_to_generation_device(inputs)
                except Exception as text_only_error:
                    logger.error(f"Text-only processing also failed: {text_only_error}")
                    if not allow_text_only_fallback:
                        raise RuntimeError(
                            "Caption generation failed after multimodal preprocessing failure and text-only fallback was disabled."
                        ) from text_only_error
                    return self._fallback_caption(safe_frames, transcript)

            # Generate
            generation_config = self._get_generation_config()
            generation_kwargs = {
                "max_new_tokens": int(
                    max_new_tokens_override
                    if max_new_tokens_override is not None
                    else generation_config.get("max_new_tokens", 2048)
                ),
            }
            pad_token_id = self._resolve_pad_token_id()
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            if use_cache_override is not None:
                generation_kwargs["use_cache"] = bool(use_cache_override)
            temperature = generation_config.get("temperature")
            top_p = generation_config.get("top_p")
            top_k = generation_config.get("top_k")
            min_p = generation_config.get("min_p")
            repetition_penalty = generation_config.get("repetition_penalty")
            if temperature is not None and float(temperature) > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature
            else:
                generation_kwargs["do_sample"] = False
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
            if top_k is not None:
                generation_kwargs["top_k"] = top_k
            if min_p is not None:
                generation_kwargs["min_p"] = min_p
            if repetition_penalty is not None:
                generation_kwargs["repetition_penalty"] = repetition_penalty
            inputs = _normalize_processor_sequence_tensors(
                dict(inputs or {}),
                source="visual_processor.caption_generate",
            )
            inputs = self._move_inputs_to_generation_device(inputs)
            if self._should_apply_multimodal_generation_stability_guard(inputs):
                sampled_decode_requested, resolved_use_cache = (
                    self._apply_caption_multimodal_generation_stability_guard(generation_kwargs)
                )
                logger.info(
                    "Applying caption multimodal multi-device stability guard: "
                    "remove_invalid_values=True renormalize_logits=True synced_gpus=False "
                    "sampled_decode_requested=%s do_sample=%s use_cache=%s",
                    sampled_decode_requested,
                    bool(generation_kwargs.get("do_sample", False)),
                    resolved_use_cache,
                )
            try:
                generated_ids = self._generate_with_accelerator_lock(inputs, generation_kwargs)
            except ValueError as generate_error:
                generated_ids = self._retry_generate_without_unused_kwargs(
                    inputs=inputs,
                    generation_kwargs=generation_kwargs,
                    error=generate_error,
                )
            except Exception as generate_error:
                cudnn_failure = "cudnn" in str(generate_error or "").lower()
                if (
                    not cudnn_failure
                    and _allow_eager_attention_retry
                    and self._can_retry_video_qa_with_eager_attention()
                ):
                    original_attn_implementation = self.attn_implementation
                    logger.warning(
                        "Reloading caption runtime with attn_implementation='eager' after multimodal generation instability."
                    )
                    self.unload_model(clear_runtime_cache=True)
                    self.attn_implementation = "eager"
                    try:
                        self._load_model()
                        return self._generate_caption(
                            frames,
                            transcript,
                            query_template=query_template,
                            allow_text_only_fallback=allow_text_only_fallback,
                            max_new_tokens_override=max_new_tokens_override,
                            use_cache_override=use_cache_override,
                            _allow_eager_attention_retry=False,
                        )
                    except Exception:
                        self.unload_model(clear_runtime_cache=True)
                        self.attn_implementation = original_attn_implementation
                        raise
                raise
            if generated_ids is None:
                logger.warning("Model returned None from generate(); returning empty caption.")
                return ""

            input_ids = None
            if isinstance(inputs, dict):
                input_ids = inputs.get("input_ids")
            else:
                input_ids = getattr(inputs, "input_ids", None)

            if isinstance(generated_ids, torch.Tensor) and generated_ids.ndim == 1:
                generated_ids = generated_ids.unsqueeze(0)

            if input_ids is not None:
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
            else:
                generated_ids_trimmed = generated_ids

            decoded_outputs = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not decoded_outputs:
                logger.warning("Processor returned empty decoded outputs.")
                return ""

            output_text = decoded_outputs[0] or ""
            output_text = _compact_segment_caption(output_text)
            if _is_low_information_caption_text(output_text):
                return ""
            return output_text
            
        except torch.OutOfMemoryError as e:
            logger.warning(f"Error generating caption: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not allow_text_only_fallback:
                raise RuntimeError("Visual caption generation ran out of memory.") from e
            return ""
        except Exception as e:
            logger.warning(f"Error generating caption: {e}")
            if not allow_text_only_fallback:
                raise
            return ""

    def _generate_caption_batch(
        self,
        frames_batch: List[List[Image.Image]],
        query_template: Optional[str] = None,
        max_new_tokens_override: Optional[int] = None,
        use_cache_override: Optional[bool] = None,
    ) -> List[str]:
        """Generate independent visual captions in one multimodal batch.

        This is an opt-in performance path for the local Qwen runtime.  The
        normal segment loop remains single-item and is deliberately kept as
        the fallback because some processors do not preserve image-to-text
        grouping when multiple conversations are passed at once.
        """
        if not frames_batch:
            return []
        if self._model is None or self._processor is None:
            raise RuntimeError("Caption model is not loaded.")
        if not self._is_qwen_family_model():
            raise RuntimeError("Caption batch path currently requires a Qwen-family multimodal model.")

        query = DEFAULT_VISUAL_CAPTION_PROMPT if query_template is None else query_template.format(transcript="")
        safe_batches: List[List[Image.Image]] = []
        prompt_messages: List[List[Dict[str, Any]]] = []
        prompt_texts: List[str] = []
        for frame_group_index, frames in enumerate(frames_batch):
            safe_frames: List[Image.Image] = []
            for frame_index, frame in enumerate(frames):
                normalized = self._normalize_frame_to_rgb(frame, frame_index)
                if normalized is not None:
                    safe_frames.append(normalized)
            if not safe_frames:
                raise RuntimeError(f"No valid frames for caption batch item {frame_group_index}.")
            content: List[Dict[str, Any]] = [{"type": "image"} for _ in safe_frames]
            content.append({"type": "text", "text": query})
            messages = [{"role": "user", "content": content}]
            safe_batches.append(safe_frames)
            prompt_messages.append(messages)
            prompt_texts.append(self._render_prompt_text(messages, len(safe_frames), query))

        processor_inputs: Optional[Dict[str, Any]] = None
        last_qwen_error: Optional[Exception] = None
        process_vision_info = self._resolve_process_vision_info()
        for use_base64 in (True, False):
            qwen_messages_batch: List[List[Dict[str, Any]]] = []
            for safe_frames in safe_batches:
                qwen_content: List[Dict[str, Any]] = []
                for frame in safe_frames:
                    image_value: Any = (
                        self._pil_to_base64_data_url(frame) if use_base64 else frame
                    )
                    qwen_content.append({"type": "image", "image": image_value})
                qwen_content.append({"type": "text", "text": query})
                qwen_messages_batch.append([{"role": "user", "content": qwen_content}])
            try:
                qwen_texts: List[str] = []
                thinking_flag = False
                generation_config = self._get_generation_config()
                thinking_flag = bool(generation_config.get("enable_thinking", False))
                for qwen_messages in qwen_messages_batch:
                    try:
                        qwen_texts.append(
                            self._processor.apply_chat_template(
                                qwen_messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                chat_template_kwargs={"enable_thinking": thinking_flag},
                            )
                        )
                    except TypeError:
                        qwen_texts.append(
                            self._processor.apply_chat_template(
                                qwen_messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=thinking_flag,
                            )
                        )
                image_inputs, video_inputs = process_vision_info(qwen_messages_batch)
                processor_inputs = self._processor(
                    text=qwen_texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                processor_inputs = self._move_inputs_to_generation_device(processor_inputs)
                break
            except Exception as qwen_error:
                last_qwen_error = qwen_error
                logger.debug(
                    "Official Qwen batch vision path failed (base64=%s): %s",
                    use_base64,
                    qwen_error,
                )

        if processor_inputs is None:
            # Keep a compatibility fallback for processors that expose an
            # image processor but do not accept the qwen_vl_utils batch shape.
            try:
                flat_images = [frame for group in safe_batches for frame in group]
                processor_inputs = self._processor(
                    text=prompt_texts,
                    images=flat_images,
                    padding=True,
                    return_tensors="pt",
                )
                processor_inputs = self._move_inputs_to_generation_device(processor_inputs)
            except Exception as processor_error:
                if last_qwen_error is not None:
                    raise RuntimeError(
                        "Both official Qwen batch processing and compatibility batch processing failed."
                    ) from last_qwen_error
                raise processor_error

        generation_config = self._get_generation_config()
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(
                max_new_tokens_override
                if max_new_tokens_override is not None
                else generation_config.get("max_new_tokens", 2048)
            ),
        }
        pad_token_id = self._resolve_pad_token_id()
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if use_cache_override is not None:
            generation_kwargs["use_cache"] = bool(use_cache_override)
        temperature = generation_config.get("temperature")
        top_p = generation_config.get("top_p")
        top_k = generation_config.get("top_k")
        min_p = generation_config.get("min_p")
        repetition_penalty = generation_config.get("repetition_penalty")
        if temperature is not None and float(temperature) > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k
        if min_p is not None:
            generation_kwargs["min_p"] = min_p
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = repetition_penalty

        processor_inputs = _normalize_processor_sequence_tensors(
            dict(processor_inputs or {}),
            source="visual_processor.caption_generate_batch",
        )
        processor_inputs = self._move_inputs_to_generation_device(processor_inputs)
        if self._should_apply_multimodal_generation_stability_guard(processor_inputs):
            self._apply_caption_multimodal_generation_stability_guard(generation_kwargs)
        generated_ids = self._generate_with_accelerator_lock(processor_inputs, generation_kwargs)
        if generated_ids is None:
            return [""] * len(safe_batches)
        if isinstance(generated_ids, torch.Tensor) and generated_ids.ndim == 1:
            generated_ids = generated_ids.unsqueeze(0)

        input_ids = processor_inputs.get("input_ids") if isinstance(processor_inputs, dict) else None
        generated_ids_trimmed: List[Any] = []
        for item_index, output_ids in enumerate(generated_ids):
            if torch.is_tensor(input_ids) and item_index < int(input_ids.shape[0]):
                generated_ids_trimmed.append(output_ids[len(input_ids[item_index]) :])
            else:
                generated_ids_trimmed.append(output_ids)
        decoded_outputs = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        captions = [
            _compact_segment_caption(output)
            for output in list(decoded_outputs or [])
        ]
        captions.extend([""] * max(0, len(safe_batches) - len(captions)))
        return captions[: len(safe_batches)]

    def _fallback_caption(self, frames: List[Image.Image], transcript: str) -> str:
        """Build a deterministic non-empty caption when VLM generation fails."""
        frame_count = len(frames)
        return f"Visual caption unavailable; sampled {frame_count} key frames."

    def answer_question_about_frames(
        self,
        frames: List[Image.Image],
        query: str,
        transcript: str = "",
    ) -> str:
        """Answer a focused question from an ordered set of video frames."""
        self._load_model()
        answer = str(self._generate_caption(frames, transcript, query) or "").strip()
        if answer:
            return answer
        return ""

    def answer_question_about_video(
        self,
        video_path: str,
        query: str,
        *,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        system_prompt: str = "",
        max_new_tokens: Optional[int] = None,
        _allow_single_device_reload_retry: bool = True,
    ) -> str:
        """Answer a focused question from a raw video clip without converting it to image QA."""
        self._load_model()

        if self._processor is None or self._model is None:
            raise RuntimeError("VisualProcessor runtime is not loaded.")
        if not self._supports_multimodal_runtime():
            raise RuntimeError("Loaded visual runtime does not advertise multimodal video support.")

        resolved_video_path = Path(str(video_path or "")).expanduser().resolve()
        if not resolved_video_path.exists():
            raise FileNotFoundError(f"Video path does not exist: {resolved_video_path}")

        question_text = str(query or "").strip()
        system_prompt_text = str(system_prompt or "").strip()
        if not question_text:
            raise ValueError("Video QA requires a non-empty query.")
        question_text = merge_instruction_into_user_message(system_prompt_text, question_text)
        system_prompt_text = ""

        generation_config = self._get_generation_config()
        thinking_flag = bool(generation_config.get("enable_thinking", False))
        direct_processor_inputs = self._prepare_direct_video_inputs_via_chat_template(
            resolved_video_path,
            question_text,
            system_prompt=system_prompt_text,
            enable_thinking=thinking_flag,
            fps=fps,
            max_frames=max_frames,
        )
        if direct_processor_inputs is not None:
            processor_inputs = self._move_inputs_to_generation_device(direct_processor_inputs)
        else:
            video_item: Dict[str, Any] = {
                "type": "video",
                "video": str(resolved_video_path),
                "do_sample_frames": True,
            }
            if fps is not None:
                video_item["fps"] = float(fps)
            if max_frames is not None:
                video_item["max_frames"] = int(max_frames)

            qwen_messages = []
            if system_prompt_text:
                qwen_messages.append({"role": "system", "content": system_prompt_text})
            qwen_messages.append(
                {
                    "role": "user",
                    "content": [
                        video_item,
                        {"type": "text", "text": question_text},
                    ],
                }
            )
            try:
                rendered_text = self._processor.apply_chat_template(
                    qwen_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": thinking_flag},
                )
            except TypeError:
                rendered_text = self._processor.apply_chat_template(
                    qwen_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking_flag,
                )
            except StopIteration as exc:
                raise RuntimeError(
                    "Multimodal chat template raised StopIteration while preparing direct video inputs."
                ) from exc

            image_inputs, video_inputs, video_kwargs = self._load_video_process_vision_payload(
                qwen_messages,
                video_path=resolved_video_path,
                fps=fps,
                max_frames=max_frames,
            )

            videos = None
            video_metadata = None
            if isinstance(video_inputs, (list, tuple)) and video_inputs:
                first_item = video_inputs[0]
                if isinstance(first_item, tuple) and len(first_item) >= 2:
                    videos = [item[0] for item in video_inputs]
                    video_metadata = [item[1] for item in video_inputs]
                else:
                    videos = list(video_inputs)
            else:
                raise RuntimeError("Multimodal video preprocessing returned an empty video payload for video QA.")

            local_video_metadata = self._build_local_video_metadata_fallback(
                resolved_video_path,
                fps=fps,
                max_frames=max_frames,
            )
            if videos is not None and not video_metadata and local_video_metadata:
                video_metadata = [dict(local_video_metadata)]
            elif video_metadata:
                normalized_metadata: List[Dict[str, Any]] = []
                for item in video_metadata:
                    merged = dict(local_video_metadata)
                    if isinstance(item, dict):
                        merged.update(item)
                    normalized_item = self._normalize_video_metadata_item(merged)
                    if normalized_item:
                        normalized_metadata.append(normalized_item)
                video_metadata = normalized_metadata or None

            processor_kwargs: Dict[str, Any] = {
                "text": [rendered_text],
                "padding": True,
                "return_tensors": "pt",
                "do_resize": False,
            }
            processor_kwargs.update(video_kwargs)
            if image_inputs is not None:
                processor_kwargs["images"] = image_inputs
            if videos is not None:
                processor_kwargs["videos"] = videos
            if video_metadata:
                processor_kwargs["video_metadata"] = video_metadata
            processor_kwargs = self._drop_none_mapping_values(processor_kwargs)

            processor_inputs = self._call_video_multimodal_processor(
                processor_kwargs,
                require_video_grid=videos is not None,
            )
            processor_inputs = self._move_inputs_to_generation_device(processor_inputs)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(
                max_new_tokens
                if max_new_tokens is not None
                else generation_config.get("max_new_tokens", 2048)
            ),
            "do_sample": False,
            "use_cache": True,
        }
        pad_token_id = self._resolve_pad_token_id()
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        repetition_penalty = generation_config.get("repetition_penalty")
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        processor_inputs = _normalize_processor_sequence_tensors(
            dict(processor_inputs or {}),
            source="visual_processor.video_qa_generate",
        )
        processor_inputs = self._filter_known_optional_generation_inputs(processor_inputs)
        processor_inputs = self._move_inputs_to_generation_device(processor_inputs)

        try:
            generated_ids = self._generate_with_accelerator_lock(processor_inputs, generation_kwargs)
        except StopIteration as exc:
            input_summary = self._summarize_input_tensor_shapes(processor_inputs)
            retry_generation_kwargs = dict(generation_kwargs)
            retry_exc: Optional[StopIteration] = exc
            if retry_exc is not None:
                try:
                    generated_ids, removed_optional_keys, removed_label = (
                        self._retry_direct_video_generate_without_optional_inputs(
                            processor_inputs=processor_inputs,
                            generation_kwargs=retry_generation_kwargs,
                        )
                    )
                except StopIteration:
                    pass
                else:
                    self._remember_unsupported_optional_generation_inputs(
                        removed_optional_keys,
                        label=removed_label,
                        input_summary=input_summary,
                    )
                    retry_exc = None
            if retry_exc is None:
                pass
            elif _allow_single_device_reload_retry and self._can_retry_video_qa_on_single_device():
                logger.warning(
                    "Caption model.generate raised StopIteration for direct video QA (%s); retrying with single-device reload instead of use_cache=False.",
                    input_summary,
                )
                return self._retry_video_qa_on_single_device(
                    video_path=str(resolved_video_path),
                    query=query,
                    fps=fps,
                    max_frames=max_frames,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                )
            elif self._can_retry_video_qa_with_eager_attention():
                logger.warning(
                    "Caption model.generate raised StopIteration for direct video QA (%s); retrying with eager attention instead of use_cache=False.",
                    input_summary,
                )
                return self._retry_video_qa_with_eager_attention(
                    video_path=str(resolved_video_path),
                    query=query,
                    fps=fps,
                    max_frames=max_frames,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                )
            else:
                raise RuntimeError(
                    "Model.generate raised StopIteration for direct video QA. "
                    "This usually indicates an unstable multimodal/device_map runtime. "
                    f"Input summary: {input_summary}"
                ) from retry_exc
        except ValueError as generate_error:
            generated_ids = self._retry_generate_without_unused_kwargs(
                inputs=processor_inputs,
                generation_kwargs=generation_kwargs,
                error=generate_error,
            )

        if generated_ids is None:
            return ""

        input_ids = processor_inputs.get("input_ids") if isinstance(processor_inputs, dict) else None
        if isinstance(generated_ids, torch.Tensor) and generated_ids.ndim == 1:
            generated_ids = generated_ids.unsqueeze(0)

        if input_ids is not None:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
        else:
            generated_ids_trimmed = generated_ids

        decoded_outputs = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded_outputs:
            return ""
        answer_text = str(decoded_outputs[0] or "").strip()
        if answer_text:
            logger.info("[Video_Clip_QA][RAW_RESPONSE_BEGIN][backend=video]\n%s\n[Video_Clip_QA][RAW_RESPONSE_END]", answer_text)
        normalized_answer = normalize_video_clip_qa_answer(answer_text)
        needs_retry = _looks_like_truncated_video_qa_answer(answer_text) or not bool(normalized_answer.get("has_valid_json"))
        if _allow_single_device_reload_retry and needs_retry:
            retry_reason = "truncated" if _looks_like_truncated_video_qa_answer(answer_text) else "non-JSON"
            logger.warning(
                "Direct video QA answer appears %s; retrying once with a larger generation budget for %s.",
                retry_reason,
                resolved_video_path,
            )
            expanded_budget = max(
                int(generation_kwargs.get("max_new_tokens", 0) or 0) * 2,
                int(getattr(settings, "video_clip_qa_max_new_tokens", 2048) or 2048),
            )
            if expanded_budget > int(generation_kwargs.get("max_new_tokens", 0) or 0):
                retry_answer = self.answer_question_about_video(
                    str(resolved_video_path),
                    query,
                    fps=fps,
                    max_frames=max_frames,
                    system_prompt=system_prompt,
                    max_new_tokens=expanded_budget,
                    _allow_single_device_reload_retry=False,
                )
                retry_answer_text = str(retry_answer or "").strip()
                retry_normalized = normalize_video_clip_qa_answer(retry_answer_text)
                if retry_normalized.get("has_valid_json") or len(retry_answer_text) > len(answer_text):
                    return retry_answer_text
        return answer_text
    
    def caption_retrieved_segments(
        self,
        retrieved_segments: List[str],
        video_paths: Dict[str, str],
        segment_info: Dict[str, Dict],
        refine_knowledge: str,
        frame_interval: float = 6.0
    ) -> Dict[str, str]:
        """
        Generate refined captions for retrieved segments.
        
        Args:
            retrieved_segments: List of segment IDs
            video_paths: Mapping from video name to path
            segment_info: Segment information
            refine_knowledge: Query context for refinement
            frame_interval: Time interval in seconds between sampled frames (5-10s per PRD)
            
        Returns:
            Dictionary mapping segment ID to refined caption
        """
        self._load_model()
        
        caption_result = {}
        for segment_id in tqdm(
            retrieved_segments,
            desc="Refining captions",
            **self._build_tqdm_kwargs(),
        ):
            video_name = '_'.join(segment_id.split('_')[:-1])
            index = segment_id.split('_')[-1]
            
            video_path = video_paths.get(video_name)
            if not video_path or not Path(video_path).exists():
                continue
            
            segment_data = segment_info.get(video_name, {}).get(index, {})
            time_range = segment_data.get("time", "0-0").split('-')
            start, end = int(time_range[0]), int(time_range[1])
            
            with VideoFileClip(video_path) as video:
                # 按照固定时间间隔抽帧（每frame_interval秒一帧）
                segment_duration = end - start
                num_frames = max(1, int(segment_duration / frame_interval))
                frame_times = np.linspace(start, end, num_frames, endpoint=False)
                frames = self.encode_video_frames(video, frame_times)
                
                query = (
                    "Provide a very detailed description (caption) of the video in English "
                    f"and extract relevant information about: {refine_knowledge}"
                )
                
                caption = self._generate_caption(frames, "", query)
                caption_result[segment_id] = f"Caption:\n{caption}\n\n"
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return caption_result


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_service_provider_name(provider: Any) -> str:
    normalized = str(provider or "").strip().lower()
    aliases = {
        "": "auto",
        "sidecar": "service",
        "remote": "service",
        "openai": "service",
        "api": "service",
        "process": "process",
        "local_process": "process",
        "subprocess": "process",
        "disable": "disabled",
        "off": "disabled",
        "false": "disabled",
        "0": "disabled",
    }
    return aliases.get(normalized, normalized)


def _runtime_profile_forces_local_visual_runtime() -> bool:
    normalized = str(os.environ.get("TOOL_STAR_RUNTIME_PROFILE", "") or "").strip().lower()
    return normalized in {
        "evaluation_local",
        "evaluation-local",
        "local_evaluation",
        "local-evaluation",
        "evaluation",
        "eval",
    }


def _visual_role_env_key(model_role: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(model_role or "").strip().upper()).strip("_")


def _settings_role_mapping_value(mapping_name: str, model_role: str) -> str:
    mapping = getattr(settings, mapping_name, {}) or {}
    if not isinstance(mapping, dict):
        return ""
    normalized_role = str(model_role or "").strip().lower()
    role_key = _visual_role_env_key(model_role).lower()
    for key in (normalized_role, role_key):
        if not key:
            continue
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _infer_visual_model_role(model_name: Optional[str], model_path: Optional[str]) -> str:
    raw = f"{model_name or ''} {model_path or ''}".lower()
    try:
        resolved_path = str(Path(str(model_path or "")).expanduser().resolve())
        resolved_ocr = str(Path(settings.ocr_model_path).expanduser().resolve())
        resolved_qwen_vl = str(Path(settings.qwen_vl_model_path).expanduser().resolve())
    except Exception:
        resolved_path = str(model_path or "")
        resolved_ocr = settings.ocr_model_path
        resolved_qwen_vl = settings.qwen_vl_model_path
    if "ocr" in raw or "deepseek" in raw:
        return "ocr"
    if resolved_path == resolved_ocr and resolved_path != resolved_qwen_vl:
        return "ocr"
    return "qwen_vl"


def _default_visual_service_model(model_role: str) -> str:
    normalized_role = str(model_role or "").strip().lower()
    if normalized_role == "ocr":
        return str(settings.ocr_model_path or "").strip()
    return str(settings.qwen_vl_model_path or "").strip()


def _visual_service_provider(model_role: str) -> str:
    role_key = _visual_role_env_key(model_role)
    return _normalize_service_provider_name(
        _first_nonempty(
            os.environ.get(f"TOOL_STAR_{role_key}_MODEL_PROVIDER"),
            os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_PROVIDER"),
            os.environ.get("TOOL_STAR_NON_PLANNER_MODEL_PROVIDER"),
            os.environ.get("TOOL_STAR_AUX_MODEL_PROVIDER"),
            _settings_role_mapping_value("non_planner_role_providers", model_role),
            getattr(settings, "non_planner_model_provider", "auto"),
        )
    )


def _visual_service_base(model_role: str) -> str:
    role_key = _visual_role_env_key(model_role)
    return format_endpoint_list(_first_nonempty(
        os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_API_BASES"),
        os.environ.get(f"TOOL_STAR_{role_key}_API_BASES"),
        os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_API_BASE"),
        os.environ.get(f"TOOL_STAR_{role_key}_API_BASE"),
        os.environ.get("TOOL_STAR_NON_PLANNER_SERVICE_API_BASES"),
        os.environ.get("TOOL_STAR_NON_PLANNER_API_BASES"),
        os.environ.get("TOOL_STAR_NON_PLANNER_SERVICE_API_BASE"),
        os.environ.get("TOOL_STAR_NON_PLANNER_API_BASE"),
        os.environ.get("TOOL_STAR_AUX_SERVICE_API_BASES"),
        os.environ.get("TOOL_STAR_AUX_API_BASES"),
        os.environ.get("TOOL_STAR_AUX_SERVICE_API_BASE"),
        os.environ.get("TOOL_STAR_AUX_API_BASE"),
        _settings_role_mapping_value("non_planner_role_api_bases", model_role),
        getattr(settings, "non_planner_api_base", ""),
    ))


def _visual_service_model(model_role: str) -> str:
    role_key = _visual_role_env_key(model_role)
    return _first_nonempty(
        os.environ.get(f"TOOL_STAR_{role_key}_SERVICE_MODEL"),
        os.environ.get(f"TOOL_STAR_{role_key}_MODEL"),
        os.environ.get("TOOL_STAR_NON_PLANNER_SERVICE_MODEL"),
        os.environ.get("TOOL_STAR_NON_PLANNER_MODEL"),
        os.environ.get("TOOL_STAR_AUX_SERVICE_MODEL"),
        os.environ.get("TOOL_STAR_AUX_MODEL"),
        _settings_role_mapping_value("non_planner_role_models", model_role),
        getattr(settings, "non_planner_model", ""),
        _default_visual_service_model(model_role),
    )


def _visual_service_enabled(model_role: str) -> bool:
    if _runtime_profile_forces_local_visual_runtime():
        logger.info(
            "Visual sidecar runtime disabled for role=%s because TOOL_STAR_RUNTIME_PROFILE=%s forces local evaluation runtime.",
            str(model_role or "").strip().lower(),
            str(os.environ.get("TOOL_STAR_RUNTIME_PROFILE", "") or "").strip(),
        )
        return False
    provider = _visual_service_provider(model_role)
    if provider in {"disabled", "local", "transformers"}:
        return False
    if provider == "process":
        return True
    if provider not in {"auto", "service"}:
        return False
    if provider == "auto":
        return bool(_visual_service_base(model_role) and _visual_service_model(model_role))
    return True


class ServiceVisualProcessor(VisualProcessor):
    """Delegated non-planner visual caption/QA runtime."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: str = "",
        device: str = None,
        max_slice_nums: int = 2,
        require_local: bool = True,
        accelerator_lock: Any = None,
    ):
        del device, max_slice_nums, require_local, accelerator_lock
        self.model_role = _infer_visual_model_role(model_name, model_path)
        self.model_name = model_name or model_path or settings.qwen_vl_model_path
        self.model_path = model_path or (
            settings.ocr_model_path if self.model_role == "ocr" else settings.qwen_vl_model_path
        )
        self.device = "service"
        self.runtime_device = "service"
        self.device_map = None
        self.accelerator_count = 0
        self._llm = None
        provider_name = _visual_service_provider(self.model_role)
        logger.info(
            "Visual processor runtime: provider=%s role=%s model_hint=%s base=%s",
            provider_name,
            self.model_role,
            _visual_service_model(self.model_role) or self.model_name,
            _visual_service_base(self.model_role),
        )

    def _caption_cache_model_fingerprint(self) -> str:
        return "{}:{}:{}:{}".format(
            _visual_service_provider(self.model_role),
            self.model_role,
            _visual_service_base(self.model_role),
            _visual_service_model(self.model_role) or self.model_name,
        )

    def _load_model(self) -> None:
        if self._llm is not None:
            return
        from core.llm.provider import get_llm_provider

        self._llm = get_llm_provider(
            model_name=self.model_path,
            model_role=self.model_role,
        )
        if self._llm.__class__.__name__ == "TransformersProvider":
            raise RuntimeError(
                f"Visual delegated runtime for role={self.model_role} was requested, but provider routing returned TransformersProvider."
            )

    def _supports_multimodal_runtime(self) -> bool:
        return True

    def unload_model(self, clear_runtime_cache: bool = False) -> None:
        del clear_runtime_cache
        return None

    def _frame_to_data_url(self, frame: Image.Image, max_pixels: Optional[int]) -> str:
        normalized = self._normalize_frame_to_rgb(frame, 0)
        if normalized is None:
            raise ValueError("Invalid frame for visual service request.")
        resized = self._resize_frames_to_max_pixels([normalized], max_pixels)[0]
        buffer = BytesIO()
        resized.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _env_positive_int(name: str, fallback: int) -> int:
        try:
            value = os.environ.get(name)
            if value is None or str(value).strip() == "":
                return max(1, int(fallback))
            return max(1, int(value))
        except (TypeError, ValueError):
            return max(1, int(fallback))

    def _get_visual_service_request_budget(self) -> Tuple[int, int]:
        _, global_max_pixels = self._get_vision_pixel_bounds()
        min_pixels = _MIN_FRAME_EDGE * _MIN_FRAME_EDGE
        configured_max_frames = int(getattr(settings, "visual_service_max_frames", 1) or 1)
        configured_max_pixels = int(getattr(settings, "visual_service_max_pixels", 336 * 336) or (336 * 336))
        max_frames = self._env_positive_int("TOOL_STAR_VISUAL_SERVICE_MAX_FRAMES", configured_max_frames)
        max_pixels = self._env_positive_int(
            "TOOL_STAR_VISUAL_SERVICE_MAX_PIXELS",
            min(global_max_pixels, max(min_pixels, configured_max_pixels)),
        )
        return max_frames, max(min_pixels, min(global_max_pixels, max_pixels))

    def _generate_from_frames(
        self,
        frames: List[Image.Image],
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if not frames:
            return ""
        self._load_model()
        max_frames, max_pixels = self._get_visual_service_request_budget()
        sampled_frames = self._sample_frames_evenly(frames, max_frames)
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": self._frame_to_data_url(frame, max_pixels)}}
            for frame in sampled_frames
        ]
        instruction = system_prompt or (
            "You are a visual captioner. Return only the final answer text. "
            "Do not include reasoning, hidden thoughts, bullets, Markdown, or task commentary."
        )
        content.append({"type": "text", "text": str(prompt or "").strip()})
        messages: List[Dict[str, Any]] = []
        if str(instruction or "").strip():
            messages.append({"role": "system", "content": str(instruction or "").strip()})
        messages.append({"role": "user", "content": content})
        response = self._llm.generate(
            messages,
            temperature=0.0,
            max_tokens=int(max_new_tokens or _CAPTION_DEFAULT_MAX_NEW_TOKENS),
            enable_thinking=False,
        )
        return str(response or "").strip()

    def _generate_caption(
        self,
        frames: List[Image.Image],
        transcript: str,
        query_template: Optional[str] = None,
        allow_text_only_fallback: bool = True,
        max_new_tokens_override: Optional[int] = None,
        use_cache_override: Optional[bool] = None,
    ) -> str:
        del transcript, allow_text_only_fallback, use_cache_override
        prompt = query_template.format(transcript="") if query_template else DEFAULT_VISUAL_CAPTION_PROMPT
        return self._generate_from_frames(
            frames,
            prompt,
            max_new_tokens=max_new_tokens_override or _CAPTION_DEFAULT_MAX_NEW_TOKENS,
        )

    def _generate_segment_caption_with_oom_retry(self, frames: List[Image.Image]) -> str:
        return self._generate_caption(
            frames,
            "",
            max_new_tokens_override=_CAPTION_DEFAULT_MAX_NEW_TOKENS,
        )

    def answer_question_about_video(
        self,
        video_path: str,
        query: str,
        *,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        system_prompt: str = "",
        max_new_tokens: Optional[int] = None,
        _allow_single_device_reload_retry: bool = True,
    ) -> str:
        del _allow_single_device_reload_retry
        resolved_video_path = Path(video_path).expanduser().resolve()
        if not resolved_video_path.exists():
            raise FileNotFoundError(f"Video path does not exist: {resolved_video_path}")
        with VideoFileClip(str(resolved_video_path)) as video:
            duration = float(getattr(video, "duration", 0.0) or 0.0)
            clip_fps = float(getattr(video, "fps", 0.0) or 0.0)
            if duration <= 0:
                return ""
            service_max_frames, _ = self._get_visual_service_request_budget()
            requested_count = int(max_frames or max(1, round(duration * float(fps or 1.0))))
            requested_limit = min(service_max_frames, int(max_frames or service_max_frames))
            requested_count = max(1, min(requested_count, requested_limit))
            frame_interval = (1.0 / clip_fps) if clip_fps > 1e-6 else 1e-2
            safe_end = max(0.0, duration - max(1e-3, frame_interval))
            frame_times = np.linspace(0.0, safe_end, num=requested_count, endpoint=True)
            frames = self.encode_video_frames(video, frame_times)
        return self._generate_from_frames(
            frames,
            str(query or "").strip(),
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens or int(getattr(settings, "video_clip_qa_max_new_tokens", 512) or 512),
        )


def create_visual_processor(
    model_name: Optional[str] = None,
    model_path: str = "",
    device: str = None,
    max_slice_nums: int = 2,
    require_local: bool = True,
    accelerator_lock: Any = None,
) -> VisualProcessor:
    model_role = _infer_visual_model_role(model_name, model_path)
    if _visual_service_enabled(model_role):
        return ServiceVisualProcessor(
            model_name=model_name,
            model_path=model_path,
            device=device,
            max_slice_nums=max_slice_nums,
            require_local=require_local,
            accelerator_lock=accelerator_lock,
        )
    return VisualProcessor(
        model_name=model_name,
        model_path=model_path,
        device=device,
        max_slice_nums=max_slice_nums,
        require_local=require_local,
        accelerator_lock=accelerator_lock,
    )
