"""Offline Qwen3-VL-Embedding runtime adapter for segment retrieval."""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from config.device_config import (
    detect_device,
    get_accelerator_snapshot,
    initialize_device,
)
from config.settings import settings
from core.accelerator_queue import SharedAcceleratorQueueLock, resolve_local_accelerator_slot_key
from runtime.session import runtime_session

logger = logging.getLogger(__name__)

_WEIGHT_FILE_CANDIDATES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_SUSPICIOUS_POINTER_SIZE_BYTES = 4096
_VIDEO_METADATA_KEY_ALIASES = {
    "video_fps": "fps",
    "sample_fps": "fps",
    "frames": "total_num_frames",
    "frame_count": "total_num_frames",
    "num_frames": "total_num_frames",
}
_VIDEO_METADATA_ALLOWED_KEYS = frozenset({"fps", "duration", "total_num_frames"})


class _Qwen3VLConfigCompat(PretrainedConfig):
    model_type = "qwen3_vl"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _register_qwen3_config_if_missing() -> None:
    try:
        from transformers import AutoConfig as AutoConfigClass
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception:
        return

    try:
        if hasattr(AutoConfigClass, "register"):
            AutoConfigClass.register("qwen3_vl", _Qwen3VLConfigCompat)
    except Exception:
        pass

    try:
        if hasattr(CONFIG_MAPPING, "register"):
            CONFIG_MAPPING.register("qwen3_vl", _Qwen3VLConfigCompat)
    except Exception:
        pass

    try:
        CONFIG_MAPPING["qwen3_vl"] = _Qwen3VLConfigCompat
    except Exception:
        pass


def _resolve_process_vision_info():
    try:
        from qwen_vl_utils.vision_process import process_vision_info  # type: ignore

        return process_vision_info
    except Exception:
        pass

    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        return process_vision_info
    except Exception as exc:
        raise RuntimeError(
            "qwen_vl_utils.process_vision_info is required for Qwen3-VL-Embedding video inputs."
        ) from exc


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


def _drop_none_mapping_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        if value is None:
            continue
        cleaned[key] = value
    return cleaned


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
        "Normalizing Qwen3-VL-Embedding processor tensors from %s: %s -> common_length=%d",
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


@dataclass
class _Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


_register_qwen3_config_if_missing()

try:
    from transformers import Qwen3VLModel, Qwen3VLPreTrainedModel, Qwen3VLProcessor
except Exception as exc:  # pragma: no cover - import depends on target runtime
    raise RuntimeError(
        "transformers is missing Qwen3VLModel/Qwen3VLProcessor support required by "
        "Qwen/Qwen3-VL-Embedding-2B."
    ) from exc


class _ToolStarQwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False

    def __init__(self, config) -> None:
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value) -> None:
        self.model.set_input_embeddings(value)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> _Qwen3VLForEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return _Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class Qwen3VLEmbeddingRuntimeEncoder:
    """Encodes text and segment clips with Qwen3-VL-Embedding-2B."""

    _RUNTIME_CACHE: ClassVar[Dict[Tuple[str, str], Dict[str, Any]]] = {}
    DEFAULT_INSTRUCTION: ClassVar[str] = "Represent the user's input."
    QUERY_INSTRUCTION: ClassVar[str] = "Retrieve relevant video segments for the user's query."
    MAX_LENGTH: ClassVar[int] = 8192
    MIN_PIXELS: ClassVar[int] = 4096
    MAX_PIXELS: ClassVar[int] = 1843200
    TOTAL_VIDEO_PIXELS: ClassVar[int] = 7864320
    VIDEO_FPS: ClassVar[float] = 1.0
    MAX_FRAMES: ClassVar[int] = 64
    MAX_IMAGE_INPUTS: ClassVar[int] = 4

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        accelerator_lock: Any = None,
    ) -> None:
        self.model_path = str(model_path or settings.qwen3_vl_embedding_model_path)
        explicit_runtime_device = None
        if device is None:
            device_type, device_count = detect_device()
            self.device = device_type
            self.accelerator_count = device_count or 0
        else:
            explicit_runtime_device = str(device).strip()
            self.device = explicit_runtime_device.split(":", 1)[0] if ":" in explicit_runtime_device else explicit_runtime_device
            _, device_count = detect_device()
            self.accelerator_count = device_count or 0

        initialize_device(self.device)
        self.runtime_device = (
            explicit_runtime_device
            if explicit_runtime_device and ":" in explicit_runtime_device
            else runtime_session.get_stable_device("retrieval", device=self.device)
        )
        self._accelerator_lock = accelerator_lock
        if self._accelerator_lock is None:
            self._accelerator_lock = SharedAcceleratorQueueLock(
                name="qwen3_vl_embedding",
                slot_key=resolve_local_accelerator_slot_key(self.runtime_device),
                enabled=str(self.runtime_device).startswith("cuda"),
            )
        self.model = None
        self.processor = None
        self._process_vision_info = _resolve_process_vision_info()

    def _validate_checkpoint_layout(self, checkpoint_path: Path) -> Dict[str, Any]:
        config_path = checkpoint_path / "config.json"
        if not config_path.exists():
            if (checkpoint_path / "src").exists() and (checkpoint_path / "scripts").exists():
                raise RuntimeError(
                    "settings.qwen3_vl_embedding_model_path appears to point to the Qwen3-VL-Embedding code repo, "
                    f"not the model checkpoint: {checkpoint_path}. The official example imports "
                    "`scripts.qwen3_vl_embedding.Qwen3VLEmbedder` from the repo, but `model_name_or_path` must point "
                    "to the downloaded checkpoint directory that contains config.json and model weights."
                )
            raise RuntimeError(
                f"Qwen3-VL-Embedding checkpoint is missing config.json: {checkpoint_path}"
            )

        found_weight_files = [
            name for name in _WEIGHT_FILE_CANDIDATES if (checkpoint_path / name).exists()
        ]
        if not found_weight_files:
            raise RuntimeError(
                "Qwen3-VL-Embedding checkpoint is incomplete: no model weight files were found. "
                f"Expected one of {list(_WEIGHT_FILE_CANDIDATES)} under {checkpoint_path}. "
                "The official checkpoint contains `model.safetensors`; if you only synced tokenizer/config files "
                "or pointed at the repo checkout instead of the model snapshot, loading will fail."
            )

        primary_weight_path = checkpoint_path / found_weight_files[0]
        if primary_weight_path.name == "model.safetensors":
            try:
                file_size = primary_weight_path.stat().st_size
            except OSError:
                file_size = -1
            if 0 <= file_size < _SUSPICIOUS_POINTER_SIZE_BYTES:
                raise RuntimeError(
                    "Qwen3-VL-Embedding checkpoint contains a suspiciously tiny `model.safetensors` file, which "
                    "looks like an LFS/Xet pointer rather than the actual weights. "
                    f"path={primary_weight_path} size_bytes={file_size}. "
                    "Please sync the real model weights instead of the pointer file."
                )

        index_path = checkpoint_path / "model.safetensors.index.json"
        if index_path.exists():
            try:
                index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read Qwen3-VL-Embedding shard index: {index_path}"
                ) from exc
            if isinstance(index_payload, dict):
                weight_map = index_payload.get("weight_map", {})
                if isinstance(weight_map, dict):
                    missing_shards = sorted(
                        {
                            shard_name
                            for shard_name in weight_map.values()
                            if not (checkpoint_path / str(shard_name)).exists()
                        }
                    )
                    if missing_shards:
                        raise RuntimeError(
                            "Qwen3-VL-Embedding sharded checkpoint is incomplete: some shard files are missing. "
                            f"checkpoint={checkpoint_path} missing_shards={missing_shards[:5]}"
                        )

        return {
            "config_path": str(config_path),
            "weight_files": found_weight_files,
        }

    def _resolve_local_path(self, value: str) -> Path:
        candidate = Path(str(value or "")).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (settings.project_root / candidate).resolve()

    def _preferred_dtype(self) -> torch.dtype:
        if self.device == "cpu":
            return torch.float32
        if self.device == "cuda":
            try:
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16

    def _load_process_vision_payload(
        self,
        conversations: List[List[Dict[str, Any]]],
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        kwargs = {
            "image_patch_size": 16,
            "return_video_metadata": True,
            "return_video_kwargs": True,
        }
        try:
            images, video_inputs, video_kwargs = self._process_vision_info(conversations, **kwargs)
            return images, video_inputs, _drop_none_mapping_values(dict(video_kwargs or {}))
        except TypeError:
            images, video_inputs = self._process_vision_info(conversations)
            return images, video_inputs, {}

    def _lazy_load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        resolved_model_path = self._resolve_local_path(self.model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"Local Qwen3-VL-Embedding model path not found: {resolved_model_path}"
            )
        checkpoint_layout = self._validate_checkpoint_layout(resolved_model_path)

        config = AutoConfig.from_pretrained(
            str(resolved_model_path),
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=str(resolved_model_path),
        )
        model_type = str(getattr(config, "model_type", "") or "").strip().lower()
        if model_type != "qwen3_vl":
            raise RuntimeError(
                f"Qwen3-VL-Embedding runtime expected a qwen3_vl checkpoint, but got model_type={model_type!r} "
                f"from {resolved_model_path / 'config.json'}"
            )

        cache_key = (str(resolved_model_path), self.runtime_device)
        cached_runtime = self._RUNTIME_CACHE.get(cache_key)
        runtime_session.record_cache("qwen3_vl_embedding_runtime", cached_runtime is not None)
        if cached_runtime is None:
            started = time.perf_counter()
            processor = Qwen3VLProcessor.from_pretrained(
                str(resolved_model_path),
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=str(resolved_model_path),
                padding_side="right",
            )
            model = _ToolStarQwen3VLForEmbedding.from_pretrained(
                str(resolved_model_path),
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=str(resolved_model_path),
            )
            model.to(self.runtime_device)
            preferred_dtype = self._preferred_dtype()
            if preferred_dtype != torch.float32:
                try:
                    model.to(dtype=preferred_dtype)
                except Exception:
                    logger.warning(
                        "Qwen3-VL-Embedding runtime could not switch to %s on %s; keeping default dtype.",
                        preferred_dtype,
                        self.runtime_device,
                    )
            model.eval()
            cached_runtime = {
                "model": model,
                "processor": processor,
                "model_path": str(resolved_model_path),
            }
            self._RUNTIME_CACHE[cache_key] = cached_runtime
            logger.info(
                "Qwen3-VL-Embedding runtime initialized: device=%s visible_accelerators=%s model=%s weight_files=%s",
                self.runtime_device,
                self.accelerator_count,
                resolved_model_path,
                checkpoint_layout["weight_files"],
            )
            logger.info(
                "Qwen3-VL-Embedding accelerator snapshot: %s",
                get_accelerator_snapshot(self.device),
            )
            runtime_session.record_timing(
                "runtime_load.qwen3_vl_embedding_seconds",
                time.perf_counter() - started,
            )

        self.model = cached_runtime["model"]
        self.processor = cached_runtime["processor"]

    @staticmethod
    def _as_file_uri(path: str) -> str:
        text = str(path or "").strip()
        if text.startswith(("http://", "https://", "file://")):
            return text
        return Path(text).expanduser().resolve().as_uri()

    def _format_input(
        self,
        *,
        text: Optional[str] = None,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []

        if video:
            content.append(
                {
                    "type": "video",
                    "video": self._as_file_uri(video),
                    "fps": self.VIDEO_FPS,
                    "max_frames": self.MAX_FRAMES,
                }
            )

        image_values: List[str] = []
        if images:
            image_values.extend([str(value) for value in images if str(value or "").strip()])
        if image:
            image_values.append(str(image))

        for image_value in image_values:
            content.append(
                {
                    "type": "image",
                    "image": self._as_file_uri(image_value),
                    "min_pixels": self.MIN_PIXELS,
                    "max_pixels": self.MAX_PIXELS,
                }
            )

        if text:
            content.append({"type": "text", "text": text})

        if not content:
            content.append({"type": "text", "text": "NULL"})

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction or self.DEFAULT_INSTRUCTION}],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

    def _prepare_inputs(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        self._lazy_load()
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        try:
            images, video_inputs, video_kwargs = self._load_process_vision_payload(conversations)
        except Exception as exc:
            logger.error("Error in processing vision info: %s", exc)
            images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}
            text = self.processor.apply_chat_template(
                [[{"role": "user", "content": [{"type": "text", "text": "NULL"}]}]],
                add_generation_prompt=True,
                tokenize=False,
            )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            normalized_metadata = []
            for item in video_metadata:
                normalized_item = _normalize_video_metadata_item(item)
                if normalized_item:
                    normalized_metadata.append(normalized_item)
            video_metadata = normalized_metadata or None
        else:
            videos, video_metadata = None, None

        processor_kwargs = {
            "text": text,
            "images": images,
            "videos": videos,
            "truncation": True,
            "max_length": self.MAX_LENGTH,
            "padding": True,
            "do_resize": False,
            "return_tensors": "pt",
            **video_kwargs,
        }
        if video_metadata:
            processor_kwargs["video_metadata"] = video_metadata
        processor_kwargs = _drop_none_mapping_values(processor_kwargs)

        inputs = _normalize_processor_sequence_tensors(
            dict(self.processor(**processor_kwargs) or {}),
            source="qwen3_vl_embedding.prepare_inputs",
        )
        if videos is not None and (
            "video_grid_thw" not in inputs
            or not torch.is_tensor(inputs["video_grid_thw"])
            or inputs["video_grid_thw"].numel() == 0
        ):
            raise RuntimeError(
                "Qwen3-VL-Embedding processor did not return video_grid_thw for a video input. "
                "This usually means the local qwen_vl_utils / transformers combination is incompatible "
                "with the current Qwen3-VL checkpoint."
            )
        prepared: Dict[str, Any] = {}
        for key, value in dict(inputs or {}).items():
            if value is None:
                continue
            prepared[key] = value.to(self.model.device) if torch.is_tensor(value) else value
        return prepared

    @staticmethod
    def _pool_last_token(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden_state.shape[0] != attention_mask.shape[0]:
            common_batch = min(int(hidden_state.shape[0]), int(attention_mask.shape[0]))
            logger.warning(
                "Qwen3-VL-Embedding batch mismatch before pooling: hidden_state=%s attention_mask=%s. "
                "Truncating batch to %d.",
                tuple(hidden_state.shape),
                tuple(attention_mask.shape),
                common_batch,
            )
            hidden_state = hidden_state[:common_batch]
            attention_mask = attention_mask[:common_batch]
        if hidden_state.shape[1] != attention_mask.shape[1]:
            common_length = min(int(hidden_state.shape[1]), int(attention_mask.shape[1]))
            logger.warning(
                "Qwen3-VL-Embedding sequence mismatch before pooling: hidden_state=%s attention_mask=%s. "
                "Truncating to %d.",
                tuple(hidden_state.shape),
                tuple(attention_mask.shape),
                common_length,
            )
            hidden_state = hidden_state[:, :common_length, :]
            attention_mask = attention_mask[:, :common_length]
        flipped_mask = attention_mask.flip(dims=[1])
        last_one_positions = flipped_mask.argmax(dim=1)
        cols = attention_mask.shape[1] - last_one_positions - 1
        rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[rows, cols]

    def _encode_conversations(self, conversations: List[List[Dict[str, Any]]]) -> torch.Tensor:
        lock_context = self._accelerator_lock if self._accelerator_lock is not None else nullcontext()
        with lock_context:
            self._lazy_load()
            inputs = self._prepare_inputs(conversations)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._pool_last_token(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                embeddings = F.normalize(embeddings, p=2, dim=-1)
        vector = embeddings[0].detach().to(device="cpu", dtype=torch.float32)
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        return vector

    def encode_query(self, query: str) -> torch.Tensor:
        normalized = " ".join(str(query or "").split()).strip()
        if not normalized:
            raise ValueError("Qwen3-VL-Embedding query encoding requires non-empty text.")
        conversation = self._format_input(
            text=normalized,
            instruction=self.QUERY_INSTRUCTION,
        )
        return self._encode_conversations([conversation])

    def encode_text(self, text: str) -> torch.Tensor:
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            raise ValueError("Qwen3-VL-Embedding text encoding requires non-empty text.")
        conversation = self._format_input(text=normalized)
        return self._encode_conversations([conversation])

    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        resolved_paths = [
            str(self._resolve_local_path(path))
            for path in (image_paths or [])
            if str(path or "").strip()
        ]
        existing_paths = [path for path in resolved_paths if Path(path).exists()]
        if not existing_paths:
            raise ValueError("Qwen3-VL-Embedding image encoding requires at least one existing image path.")

        if len(existing_paths) > self.MAX_IMAGE_INPUTS:
            step = max(1, len(existing_paths) // self.MAX_IMAGE_INPUTS)
            sampled = existing_paths[::step][: self.MAX_IMAGE_INPUTS]
        else:
            sampled = existing_paths

        conversation = self._format_input(images=sampled)
        return self._encode_conversations([conversation])

    def encode_video(self, video_path: str) -> torch.Tensor:
        resolved_video_path = self._resolve_local_path(video_path)
        if not resolved_video_path.exists():
            raise FileNotFoundError(
                f"Segment clip not found for Qwen3-VL-Embedding indexing: {resolved_video_path}"
            )
        conversation = self._format_input(video=str(resolved_video_path))
        return self._encode_conversations([conversation])
