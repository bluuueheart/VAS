"""Dense multimodal vector storage for Qwen3-VL-Embedding-based retrieval."""

from __future__ import annotations

import json
import logging
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from config.settings import settings
from core.accelerator_queue import SharedAcceleratorQueueLock, resolve_local_accelerator_slot_key
from core.sidecar_pool import (
    OpenAIEndpointPool,
    embeddings_api_marked_unhealthy,
    format_endpoint_list,
    mark_embeddings_api_unhealthy,
    request_pooling_embedding,
)
from .qwen3_vl_embedding_runtime import Qwen3VLEmbeddingRuntimeEncoder

logger = logging.getLogger(__name__)

VECTOR_STORE_SCHEMA_VERSION = 5
VECTOR_STORE_LOCAL_BACKEND = "qwen3_vl_embedding_dense_frames_v1"
VECTOR_STORE_SERVICE_BACKEND = "openai_compatible_text_embedding_dense_v1"
VECTOR_STORE_BACKEND = VECTOR_STORE_LOCAL_BACKEND


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
            parts = [float(part) for part in text.split(":")]
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return hours * 3600 + minutes * 60 + seconds
            if len(parts) == 2:
                minutes, seconds = parts
                return minutes * 60 + seconds
        return float(text)
    except Exception:
        return None


def _parse_time_range(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        start = _time_to_seconds(value[0])
        end = _time_to_seconds(value[1])
        if start is None or end is None:
            return None
        return (min(start, end), max(start, end))
    text = str(value or "").strip()
    if "-" not in text:
        return None
    left, right = text.split("-", 1)
    start = _time_to_seconds(left)
    end = _time_to_seconds(right)
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


def _ranges_overlap(candidate: Optional[Tuple[float, float]], target: Optional[Tuple[float, float]]) -> bool:
    if candidate is None or target is None:
        return True
    return max(candidate[0], target[0]) <= min(candidate[1], target[1])


def _resolve_cache_root() -> Path:
    cache_root = Path(settings.data_cache_dir).expanduser()
    if not cache_root.is_absolute():
        cache_root = (settings.project_root / cache_root).resolve()
    return cache_root


def _resolve_processed_asset_path(
    value: Any,
    video_name: Optional[str] = None,
    processed_root: Optional[Any] = None,
) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None

    path = Path(text)
    candidates: List[Path] = []
    cache_root = _resolve_cache_root()

    if path.is_absolute():
        candidates.append(path)
    else:
        processed_root_text = str(processed_root or "").strip()
        if processed_root_text:
            processed_root_path = Path(processed_root_text).expanduser()
            if not processed_root_path.is_absolute():
                processed_root_path = (settings.project_root / processed_root_path).resolve()
            candidates.append((processed_root_path / path).resolve())
        if video_name:
            candidates.append((cache_root / "processed" / video_name / path).resolve())
            queries_root = cache_root / "queries"
            if queries_root.exists():
                for workspace in sorted(queries_root.iterdir(), reverse=True):
                    if not workspace.is_dir():
                        continue
                    candidates.append((workspace / "processed" / video_name / path).resolve())
        candidates.append((cache_root / path).resolve())
        candidates.append((settings.project_root / path).resolve())

    resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[0] if candidates else None)
    return str(resolved) if resolved is not None else None


def _normalize_available_modalities(scores: Dict[str, float]) -> List[str]:
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_score = float(ranked[0][1])
    threshold = best_score - 0.05
    selected = [name for name, score in ranked if score >= threshold and score > 0]
    if selected:
        return selected
    return [ranked[0][0]]


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_embedding_provider_name(provider: Any) -> str:
    normalized = str(provider or "").strip().lower()
    aliases = {
        "": "auto",
        "openai": "service",
        "api": "service",
        "remote": "service",
        "sidecar": "service",
        "external": "service",
        "transformers": "local",
        "qwen3_vl": "local",
        "qwen3-vl": "local",
        "off": "disabled",
        "false": "disabled",
        "0": "disabled",
    }
    return aliases.get(normalized, normalized)


def _settings_embedding_mapping_value(mapping_name: str, role: str = "preprocess_vector") -> str:
    mapping = getattr(settings, mapping_name, {}) or {}
    if not isinstance(mapping, dict):
        return ""
    normalized_role = str(role or "").strip().lower()
    role_key = normalized_role.replace("-", "_").replace(" ", "_")
    for key in (normalized_role, role_key):
        if not key:
            continue
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _preprocess_embedding_service_provider() -> str:
    return _normalize_embedding_provider_name(
        _first_nonempty(
            _settings_embedding_mapping_value("non_planner_embedding_role_providers"),
            getattr(settings, "non_planner_embedding_provider", "auto"),
            os.environ.get("TOOL_STAR_PREPROCESS_EMBEDDING_PROVIDER"),
            os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_PROVIDER"),
            os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_PROVIDER"),
            getattr(settings, "planner_grounding_embedding_provider", "auto"),
        )
    )


def _preprocess_embedding_service_base() -> str:
    return format_endpoint_list(
        _first_nonempty(
            os.environ.get("TOOL_STAR_PREPROCESS_EMBEDDING_API_BASES"),
            os.environ.get("TOOL_STAR_PREPROCESS_EMBEDDING_API_BASE"),
            os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASES"),
            os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASE"),
            os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASES"),
            os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASE"),
            _settings_embedding_mapping_value("non_planner_embedding_role_api_bases"),
            getattr(settings, "non_planner_embedding_api_base", ""),
            getattr(settings, "planner_grounding_embedding_api_base", ""),
        )
    )


def _preprocess_embedding_service_key() -> str:
    return _first_nonempty(
        _settings_embedding_mapping_value("non_planner_embedding_role_api_keys"),
        getattr(settings, "non_planner_embedding_api_key", ""),
        os.environ.get("TOOL_STAR_PREPROCESS_EMBEDDING_API_KEY"),
        os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_API_KEY"),
        os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_KEY"),
        getattr(settings, "planner_grounding_embedding_api_key", ""),
        getattr(settings, "openai_api_key", ""),
        "EMPTY",
    )


def _preprocess_embedding_service_model() -> str:
    return _first_nonempty(
        _settings_embedding_mapping_value("non_planner_embedding_role_models"),
        getattr(settings, "non_planner_embedding_model", ""),
        getattr(settings, "qwen3_vl_embedding_model_path", ""),
        os.environ.get("TOOL_STAR_PREPROCESS_EMBEDDING_MODEL"),
        os.environ.get("TOOL_STAR_NON_PLANNER_EMBEDDING_MODEL"),
        os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_MODEL"),
        getattr(settings, "planner_grounding_embedding_model", ""),
    )


def _preprocess_embedding_service_enabled() -> bool:
    provider = _preprocess_embedding_service_provider()
    if provider in {"disabled", "local", "transformers", "qwen3_vl", "qwen3-vl"}:
        return False
    if provider not in {"auto", "service"}:
        logger.warning("Unknown preprocess embedding provider=%s; using local.", provider)
        return False
    if provider == "auto":
        return bool(_preprocess_embedding_service_base() and _preprocess_embedding_service_model())
    return True


class Qwen3VLEmbeddingGenerator:
    """Runtime wrapper around the local Qwen3-VL-Embedding checkpoint."""

    backend = VECTOR_STORE_LOCAL_BACKEND

    def __init__(self, model_path: str, device: Optional[str] = None, accelerator_lock: Any = None):
        self.model_path = model_path
        self.runtime = Qwen3VLEmbeddingRuntimeEncoder(
            model_path=model_path,
            device=device,
            accelerator_lock=accelerator_lock,
        )

    def encode_query(self, query: str) -> torch.Tensor:
        return self.runtime.encode_query(query)

    def encode_document_embeddings(self, modal_payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        payload = dict(modal_payload or {})
        vectors: Dict[str, torch.Tensor] = {}
        processed_root = payload.get("processed_root")

        frame_paths = [
            resolved
            for resolved in (
                _resolve_processed_asset_path(
                    path,
                    video_name=str(payload.get("video_name") or payload.get("video") or "").strip() or None,
                    processed_root=processed_root,
                )
                for path in (payload.get("frame_paths") or [])
            )
            if resolved and Path(resolved).exists()
        ]
        if frame_paths:
            vectors["video"] = self.runtime.encode_images(frame_paths)

        segment_video_path = _resolve_processed_asset_path(
            payload.get("segment_video_path"),
            video_name=str(payload.get("video_name") or payload.get("video") or "").strip() or None,
            processed_root=processed_root,
        )
        if segment_video_path and "video" not in vectors:
            vectors["video"] = self.runtime.encode_video(segment_video_path)

        caption = " ".join(str(payload.get("caption", "")).split()).strip()
        if caption:
            vectors["description"] = self.runtime.encode_text(caption)

        transcript = " ".join(str(payload.get("transcript", "")).split()).strip()
        if transcript:
            vectors["asr"] = self.runtime.encode_text(transcript)

        return vectors


class ServiceTextEmbeddingGenerator:
    """OpenAI-compatible text embedding sidecar for preprocessing vector indexes."""

    backend = VECTOR_STORE_SERVICE_BACKEND

    def __init__(self) -> None:
        self.api_base = _preprocess_embedding_service_base()
        self.api_key = _preprocess_embedding_service_key()
        self.model_name = _preprocess_embedding_service_model()
        if not self.api_base:
            raise RuntimeError(
                "Preprocess embedding service requires non_planner_embedding_api_base "
                "or planner_grounding_embedding_api_base in settings.py."
            )
        if not self.model_name:
            raise RuntimeError(
                "Preprocess embedding service requires non_planner_embedding_model "
                "or planner_grounding_embedding_model in settings.py."
            )
        self._service_pool = None

    def _client(self) -> OpenAIEndpointPool:
        from runtime.session import runtime_session

        cache_key = json.dumps(
            {
                "runtime": "preprocess_embedding_service_pool",
                "api_base": self.api_base,
                "model": self.model_name,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = runtime_session.get_cached_artifact(cache_key)
        if cached is None:
            cached = runtime_session.cache_artifact(
                cache_key,
                OpenAIEndpointPool(
                    api_key=self.api_key,
                    base_urls=self.api_base,
                    namespace="preprocess_embedding",
                ),
            )
        self._service_pool = cached
        return cached

    def _encode_text(self, text: str, *, kind: str) -> torch.Tensor:
        from runtime.session import runtime_session

        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return torch.zeros(1, dtype=torch.float32)
        cache_key = json.dumps(
            {
                "runtime": "preprocess_embedding_service_vector",
                "api_base": self.api_base,
                "model": self.model_name,
                "kind": kind,
                "text_sha1": hashlib.sha1(normalized.encode("utf-8")).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = runtime_session.get_cached_artifact(cache_key)
        if cached is not None:
            return cached
        with self._client().borrow() as (client, endpoint):
            if embeddings_api_marked_unhealthy(endpoint):
                embedding = request_pooling_embedding(
                    api_key=self.api_key,
                    base_url=endpoint,
                    model=self.model_name,
                    text=normalized,
                    timeout=float(getattr(settings, "sidecar_smoke_timeout_seconds", 120) or 120),
                )
            else:
                try:
                    response = client.embeddings.create(model=self.model_name, input=normalized)
                    if not response.data:
                        raise RuntimeError("Preprocess embedding service returned no vectors.")
                    embedding = response.data[0].embedding
                except Exception as exc:
                    logger.warning(
                        "Preprocess OpenAI embeddings endpoint failed on %s; retrying via vLLM /pooling: %s",
                        endpoint,
                        exc,
                    )
                    embedding = request_pooling_embedding(
                        api_key=self.api_key,
                        base_url=endpoint,
                        model=self.model_name,
                        text=normalized,
                        timeout=float(getattr(settings, "sidecar_smoke_timeout_seconds", 120) or 120),
                    )
                    mark_embeddings_api_unhealthy(endpoint)
        vector = torch.tensor(embedding, dtype=torch.float32)
        if vector.numel() == 0:
            raise RuntimeError("Preprocess embedding service returned an empty vector.")
        vector = torch.nn.functional.normalize(vector, p=2, dim=0)
        return runtime_session.cache_artifact(cache_key, vector.detach().cpu().to(torch.float32))

    def encode_query(self, query: str) -> torch.Tensor:
        return self._encode_text(query, kind="query")

    def encode_document_embeddings(self, modal_payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        payload = dict(modal_payload or {})
        vectors: Dict[str, torch.Tensor] = {}
        caption = " ".join(str(payload.get("caption", "")).split()).strip()
        transcript = " ".join(str(payload.get("transcript", "")).split()).strip()
        graph_text = " ".join(str(payload.get("graph_text", "")).split()).strip()
        if caption:
            vectors["description"] = self._encode_text(caption, kind="description")
        if transcript:
            vectors["asr"] = self._encode_text(transcript, kind="asr")
        if graph_text:
            vectors["graph"] = self._encode_text(graph_text, kind="graph")
        if not vectors and (caption or transcript or graph_text):
            combined = " ".join(part for part in (caption, transcript, graph_text) if part).strip()
            vectors["text"] = self._encode_text(combined, kind="text")
        return vectors


class VideoSegmentVectorStore:
    """Specialized Qwen3-VL-Embedding-backed vector store for video segments."""

    def __init__(
        self,
        embedding_dim: int = 0,
        embedding_model_path: str = "",
        device: Optional[str] = None,
        accelerator_lock: Any = None,
    ) -> None:
        del embedding_dim
        resolved_embedding_model_path = embedding_model_path or settings.qwen3_vl_embedding_model_path
        self.embedding_model_path = resolved_embedding_model_path
        if _preprocess_embedding_service_enabled():
            self.embedding_generator = ServiceTextEmbeddingGenerator()
            logger.info(
                "Video segment vector store uses embedding sidecar model=%s base=%s",
                self.embedding_generator.model_name,
                self.embedding_generator.api_base,
            )
        else:
            local_accelerator_lock = accelerator_lock
            if local_accelerator_lock is None:
                local_accelerator_lock = SharedAcceleratorQueueLock(
                    name="preprocess_embedding",
                    slot_key=resolve_local_accelerator_slot_key(device),
                    enabled=str(device or "").strip().lower().startswith("cuda"),
                )
            self.embedding_generator = Qwen3VLEmbeddingGenerator(
                model_path=resolved_embedding_model_path,
                device=device,
                accelerator_lock=local_accelerator_lock,
            )
        self.vector_store = DenseVectorStore(
            device=device or "cpu",
            backend=getattr(self.embedding_generator, "backend", VECTOR_STORE_LOCAL_BACKEND),
        )

    def _build_segment_payload(
        self,
        seg_id: str,
        seg_data: Dict[str, Any],
        video_name: str,
        processed_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw_segment_id = str(seg_id)
        segment_ref = f"{video_name}_{raw_segment_id}"
        segment_aliases = [str(alias) for alias in (seg_data.get("segment_aliases") or []) if str(alias).strip()]
        if not segment_aliases:
            segment_aliases = [raw_segment_id, segment_ref]

        segment_video_path = str(seg_data.get("segment_video_path", "") or "").strip()
        if not segment_video_path:
            raise ValueError(
                f"Segment {segment_ref} is missing segment_video_path. "
                "Re-run preprocessing to persist segment clips for Qwen3-VL-Embedding indexing."
            )

        return {
            "segment_id": segment_ref,
            "segment_ref": segment_ref,
            "raw_segment_id": raw_segment_id,
            "video_name": video_name,
            "video": video_name,
            "time_range": str(seg_data.get("time", "")),
            "time": str(seg_data.get("time", "")),
            "caption": str(seg_data.get("caption", "")),
            "transcript": str(seg_data.get("transcript", "")),
            "graph_text": str(seg_data.get("graph_text", "")),
            "frame_times": list(seg_data.get("frame_times") or []),
            "frame_paths": [str(path) for path in (seg_data.get("frame_paths") or []) if str(path).strip()],
            "segment_aliases": segment_aliases,
            "segment_video_path": segment_video_path,
            "processed_root": str(processed_root or "").strip(),
        }

    def index_segments(
        self,
        segments: Dict[str, Dict[str, Any]],
        video_name: str,
        processed_root: Optional[str] = None,
    ) -> None:
        self.vector_store.clear()
        for seg_id, seg_data in segments.items():
            payload = self._build_segment_payload(
                str(seg_id),
                seg_data or {},
                video_name,
                processed_root=processed_root,
            )
            vectors = self.embedding_generator.encode_document_embeddings(payload)
            if not vectors:
                raise ValueError(
                    f"Segment {payload['segment_ref']} produced no Qwen3-VL-Embedding vectors. "
                    "At least one of video/caption/transcript must be available."
                )

            metadata = {
                **{key: value for key, value in payload.items() if key != "processed_root"},
                "indexed_modalities": sorted(vectors.keys()),
            }
            self.vector_store.add_document(vectors=vectors, metadata=metadata)

    def search_segments(
        self,
        query: str,
        k: int = 5,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_generator.encode_query(query)
        return self.vector_store.search(query_vector, k=k, time_range=time_range)

    def save(self, save_dir: str) -> None:
        self.vector_store.save(save_dir)

    def load(self, load_dir: str) -> None:
        self.vector_store.load(load_dir)


class DenseVectorStore:
    """Dense per-modality store for multimodal segment retrieval."""

    def __init__(self, device: Optional[str] = None, backend: str = VECTOR_STORE_LOCAL_BACKEND) -> None:
        self.device = device or "cpu"
        self.backend = backend or VECTOR_STORE_LOCAL_BACKEND
        self.docs: List[Dict[str, torch.Tensor]] = []
        self.metadata: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.docs = []
        self.metadata = []

    def add_document(self, vectors: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> None:
        stored_vectors: Dict[str, torch.Tensor] = {}
        for modality, vector in (vectors or {}).items():
            if not torch.is_tensor(vector):
                raise TypeError(f"Document vector for modality '{modality}' must be a tensor.")
            if vector.ndim != 1:
                raise ValueError(f"Document vector for modality '{modality}' must be 1D.")
            stored_vectors[str(modality)] = vector.to(self.device, dtype=torch.float32)
        if not stored_vectors:
            raise ValueError("DenseVectorStore.add_document requires at least one modality vector.")
        self.docs.append(stored_vectors)
        self.metadata.append(dict(metadata or {}))

    def _score_document(
        self,
        query_vector: torch.Tensor,
        doc_vectors: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        modality_scores: Dict[str, float] = {}
        for modality, doc_vector in doc_vectors.items():
            if int(query_vector.numel()) != int(doc_vector.numel()):
                logger.warning(
                    "Skipping vector modality=%s because query/doc dimensions differ: %d != %d",
                    modality,
                    int(query_vector.numel()),
                    int(doc_vector.numel()),
                )
                continue
            modality_scores[str(modality)] = float(torch.dot(query_vector, doc_vector).item())
        return modality_scores

    def search(
        self,
        query_vector: torch.Tensor,
        k: int = 5,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.docs:
            return []

        query_tensor = query_vector.to(self.device, dtype=torch.float32)
        results: List[Dict[str, Any]] = []

        for doc_vectors, metadata in zip(self.docs, self.metadata):
            segment_range = _parse_time_range(metadata.get("time_range") or metadata.get("time"))
            if time_range and not _ranges_overlap(segment_range, time_range):
                continue

            modality_scores = self._score_document(query_tensor, doc_vectors)
            if not modality_scores:
                continue
            similarity_score = max(modality_scores.values())
            matched_modalities = _normalize_available_modalities(modality_scores)

            results.append(
                {
                    **metadata,
                    "similarity_score": similarity_score,
                    "matched_modalities": matched_modalities,
                    "modality_scores": modality_scores,
                }
            )

        results.sort(key=lambda item: float(item.get("similarity_score", 0.0)), reverse=True)
        return results[: max(1, k)]

    def save(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        serialized_metadata: List[Dict[str, Any]] = []
        for index, (vectors, metadata) in enumerate(zip(self.docs, self.metadata)):
            indexed_modalities = sorted(vectors.keys())
            serialized_metadata.append(
                {
                    **metadata,
                    "indexed_modalities": indexed_modalities,
                }
            )
            for modality in indexed_modalities:
                np.save(save_path / f"doc_{index}.{modality}.npy", vectors[modality].detach().cpu().numpy())

        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": VECTOR_STORE_SCHEMA_VERSION,
                    "backend": self.backend,
                    "metadata": serialized_metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, load_dir: str) -> None:
        load_path = Path(load_dir)
        metadata_path = load_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Vector store metadata missing: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}

        schema_version = int(payload.get("schema_version", 0) or 0)
        if schema_version < VECTOR_STORE_SCHEMA_VERSION:
            raise RuntimeError(
                "Cached vector store is stale and missing the Qwen3-VL-Embedding dense schema. "
                "Please re-run preprocessing."
            )
        backend = str(payload.get("backend", "") or "").strip()
        if backend != self.backend:
            raise RuntimeError(
                "Cached vector store backend does not match the active dense retriever. "
                "Please re-run preprocessing."
            )

        raw_metadata = payload.get("metadata", [])
        if not isinstance(raw_metadata, list):
            raise RuntimeError("Invalid vector store metadata payload.")

        self.clear()
        for index, metadata in enumerate(raw_metadata):
            indexed_modalities = [
                str(modality)
                for modality in ((metadata or {}).get("indexed_modalities") or [])
                if str(modality).strip()
            ]
            if not indexed_modalities:
                raise RuntimeError(
                    f"Vector store cache is incomplete for document {index}: indexed_modalities missing."
                )

            vectors: Dict[str, torch.Tensor] = {}
            for modality in indexed_modalities:
                vector_path = load_path / f"doc_{index}.{modality}.npy"
                if not vector_path.exists():
                    raise RuntimeError(
                        f"Vector store cache is incomplete for document {index} modality '{modality}'. "
                        "Please re-run preprocessing."
                    )
                vector = torch.tensor(np.load(vector_path), device=self.device, dtype=torch.float32)
                if vector.ndim != 1:
                    raise RuntimeError("Loaded vector store tensor has invalid shape.")
                vectors[modality] = vector

            self.docs.append(vectors)
            self.metadata.append(metadata if isinstance(metadata, dict) else {})
