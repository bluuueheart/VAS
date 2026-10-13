"""Process-level runtime/session cache for MVBench and interactive runs."""

from __future__ import annotations

import logging
import time
import gc
import os
import shutil
import subprocess
import threading
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, Optional, Set

from config.device_config import detect_device, get_preferred_device_string, initialize_device

logger = logging.getLogger(__name__)


def _normalize_video_scope_names(values: Iterable[Any]) -> Set[str]:
    names: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text.startswith("$"):
            continue
        path = Path(text)
        if path.suffix:
            names.add(path.stem)
            names.add(path.name)
        names.add(text)
        names.add(path.stem or text)
    return {name for name in names if name}


class RuntimeSession:
    """Holds long-lived model/runtime caches and per-query runtime state."""

    def __init__(self) -> None:
        self._provider_cache: Dict[str, Any] = {}
        self._vector_store_cache: Dict[str, Any] = {}
        self._artifact_cache: Dict[str, Any] = {}
        self._stable_devices: Dict[str, str] = {}
        self._registry_auto_discovered = False
        self._gpu_sampler_stop: Optional[threading.Event] = None
        self._gpu_sampler_thread: Optional[threading.Thread] = None
        self._gpu_sampler_lock = threading.Lock()
        self._gpu_samples: list[Dict[str, float]] = []
        self._gpu_telemetry: Dict[str, Any] = {"status": "not_started"}
        self.reset_query_state(start_gpu_telemetry=False)

    def reset_query_state(self, *, start_gpu_telemetry: bool = True) -> None:
        self._stop_gpu_telemetry()
        self._active_video_names: Set[str] = set()
        self._active_video_path: Optional[str] = None
        self._active_processed_dir: Optional[str] = None
        self._active_segment_count: Optional[int] = None
        self._active_query_text: Optional[str] = None
        self._timings: Dict[str, float] = defaultdict(float)
        self._cache_stats: Counter[str] = Counter()
        self._recent_tools: Deque[str] = deque(maxlen=24)
        if start_gpu_telemetry:
            self._start_gpu_telemetry()

    def _start_gpu_telemetry(self) -> None:
        """Sample GPU utilization for the current query without owning CUDA."""
        enabled = str(os.environ.get("TOOLSTAR_GPU_TELEMETRY", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"} or shutil.which("nvidia-smi") is None:
            self._gpu_telemetry = {
                "status": "disabled" if enabled in {"0", "false", "no", "off"} else "unavailable",
                "sample_count": 0,
            }
            return
        interval_text = str(os.environ.get("TOOLSTAR_GPU_TELEMETRY_INTERVAL_S", "2.0"))
        try:
            interval_s = max(0.5, float(interval_text))
        except (TypeError, ValueError):
            interval_s = 2.0
        stop = threading.Event()
        self._gpu_sampler_stop = stop
        self._gpu_samples = []
        self._gpu_telemetry = {
            "status": "sampling",
            "sample_count": 0,
            "interval_s": interval_s,
        }

        def _sample_once() -> None:
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return
            if completed.returncode != 0:
                return
            for line in completed.stdout.splitlines():
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 4:
                    continue
                try:
                    gpu_index = float(fields[0])
                    utilization = float(fields[1])
                    memory_used_mib = float(fields[2])
                    memory_total_mib = float(fields[3])
                except (TypeError, ValueError):
                    continue
                if memory_total_mib <= 0:
                    continue
                sample = {
                    "gpu_index": gpu_index,
                    "utilization_percent": max(0.0, utilization),
                    "memory_used_gib": max(0.0, memory_used_mib / 1024.0),
                    "memory_total_gib": max(0.0, memory_total_mib / 1024.0),
                }
                sample["memory_percent"] = (
                    100.0 * sample["memory_used_gib"] / sample["memory_total_gib"]
                )
                with self._gpu_sampler_lock:
                    if len(self._gpu_samples) < 1800:
                        self._gpu_samples.append(sample)

        def _sampler() -> None:
            while not stop.is_set():
                _sample_once()
                stop.wait(interval_s)

        thread = threading.Thread(
            target=_sampler,
            name="toolstar-gpu-telemetry",
            daemon=True,
        )
        self._gpu_sampler_thread = thread
        thread.start()

    def _stop_gpu_telemetry(self) -> Dict[str, Any]:
        stop = self._gpu_sampler_stop
        thread = self._gpu_sampler_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self._gpu_sampler_stop = None
        self._gpu_sampler_thread = None
        with self._gpu_sampler_lock:
            samples = list(self._gpu_samples)
        if not samples:
            if self._gpu_telemetry.get("status") == "sampling":
                self._gpu_telemetry = {
                    **self._gpu_telemetry,
                    "status": "no_samples",
                    "sample_count": 0,
                }
            return dict(self._gpu_telemetry)
        utilizations = [item["utilization_percent"] for item in samples]
        memory_used = [item["memory_used_gib"] for item in samples]
        memory_percent = [item["memory_percent"] for item in samples]
        per_gpu: Dict[str, Dict[str, Any]] = {}
        for sample in samples:
            key = str(int(sample["gpu_index"]))
            entry = per_gpu.setdefault(key, {"samples": []})
            entry["samples"].append(sample)
        per_gpu_summary: Dict[str, Dict[str, Any]] = {}
        for key, entries in per_gpu.items():
            gpu_utils = [item["utilization_percent"] for item in entries["samples"]]
            gpu_memory = [item["memory_used_gib"] for item in entries["samples"]]
            per_gpu_summary[key] = {
                "sample_count": len(entries["samples"]),
                "avg_utilization_percent": round(sum(gpu_utils) / len(gpu_utils), 3),
                "peak_utilization_percent": round(max(gpu_utils), 3),
                "peak_memory_used_gib": round(max(gpu_memory), 3),
            }
        self._gpu_telemetry = {
            "status": "ok",
            "sample_count": len(samples),
            "interval_s": self._gpu_telemetry.get("interval_s"),
            "gpu_count": len(per_gpu_summary),
            "avg_utilization_percent": round(sum(utilizations) / len(utilizations), 3),
            "peak_utilization_percent": round(max(utilizations), 3),
            "peak_memory_used_gib": round(max(memory_used), 3),
            "peak_memory_percent": round(max(memory_percent), 3),
            "per_gpu": per_gpu_summary,
        }
        return dict(self._gpu_telemetry)

    def ensure_registry_discovered(self, registry: Any) -> None:
        if self._registry_auto_discovered:
            return
        registry.auto_discover()
        self._registry_auto_discovered = True

    def get_stable_device(
        self,
        purpose: str,
        device: Optional[str] = None,
    ) -> str:
        requested = str(device or "").strip()
        if ":" in requested:
            return requested
        if requested:
            initialize_device(requested)
            stable = self._stable_devices.get(purpose)
            if stable and stable.startswith(requested):
                return stable
            selected = get_preferred_device_string(requested, purpose=purpose)
            self._stable_devices[purpose] = selected
            return selected

        stable = self._stable_devices.get(purpose)
        if stable:
            return stable
        detected_device, _ = detect_device()
        initialize_device(detected_device)
        selected = get_preferred_device_string(detected_device, purpose=purpose)
        self._stable_devices[purpose] = selected
        return selected

    def set_active_video_context(
        self,
        *,
        video_name: Optional[str] = None,
        video_path: Optional[str] = None,
        processed_dir: Optional[Path] = None,
        segment_count: Optional[int] = None,
    ) -> None:
        names = _normalize_video_scope_names(
            [
                video_name,
                video_path,
                processed_dir.name if processed_dir else None,
            ]
        )
        self._active_video_names = names
        self._active_video_path = str(video_path) if video_path else None
        self._active_processed_dir = str(processed_dir) if processed_dir else None
        self._active_segment_count = (
            max(0, int(segment_count))
            if segment_count is not None
            else None
        )

    def get_active_video_names(self) -> Set[str]:
        return set(self._active_video_names)

    def get_active_segment_count(self) -> Optional[int]:
        return self._active_segment_count

    def set_active_query_text(self, query: Optional[str]) -> None:
        self._active_query_text = str(query or "").strip() or None

    def get_active_query_text(self) -> str:
        return str(self._active_query_text or "")

    def resolve_video_scope(self, explicit_hint: Optional[str] = None) -> Set[str]:
        hinted = _normalize_video_scope_names([explicit_hint])
        if hinted:
            return hinted
        return self.get_active_video_names()

    def note_tool_usage(self, tool_name: str) -> None:
        normalized = str(tool_name or "").strip()
        if normalized:
            self._recent_tools.append(normalized)

    def get_recent_tools(self, limit: int = 8) -> list[str]:
        if limit <= 0:
            return []
        return list(self._recent_tools)[-limit:]

    def record_timing(self, name: str, seconds: float) -> None:
        key = str(name or "").strip()
        if not key:
            return
        self._timings[key] += max(0.0, float(seconds or 0.0))

    def record_cache(self, name: str, hit: bool) -> None:
        key = f"{str(name or '').strip()}.{('hit' if hit else 'miss')}"
        self._cache_stats[key] += 1

    def snapshot_query_state(self) -> Dict[str, Any]:
        return {
            "active_video_names": set(self._active_video_names),
            "active_video_path": self._active_video_path,
            "active_processed_dir": self._active_processed_dir,
            "active_segment_count": self._active_segment_count,
            "active_query_text": self._active_query_text,
            "timings": dict(self._timings),
            "cache_stats": dict(self._cache_stats),
            "recent_tools": list(self._recent_tools),
        }

    def restore_query_state(self, snapshot: Optional[Dict[str, Any]]) -> None:
        if not isinstance(snapshot, dict):
            self.reset_query_state()
            return

        self._active_video_names = _normalize_video_scope_names(
            snapshot.get("active_video_names") or ()
        )
        active_video_path = snapshot.get("active_video_path")
        self._active_video_path = str(active_video_path) if active_video_path else None
        active_processed_dir = snapshot.get("active_processed_dir")
        self._active_processed_dir = (
            str(active_processed_dir) if active_processed_dir else None
        )
        active_segment_count = snapshot.get("active_segment_count")
        self._active_segment_count = (
            max(0, int(active_segment_count))
            if active_segment_count is not None
            else None
        )
        active_query_text = snapshot.get("active_query_text")
        self._active_query_text = str(active_query_text).strip() if active_query_text else None

        timings = defaultdict(float)
        for key, value in dict(snapshot.get("timings") or {}).items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            timings[normalized_key] += max(0.0, float(value or 0.0))
        self._timings = timings

        cache_stats = Counter()
        for key, value in dict(snapshot.get("cache_stats") or {}).items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            cache_stats[normalized_key] = max(0, int(value or 0))
        self._cache_stats = cache_stats

        self._recent_tools = deque(
            [
                str(tool or "").strip()
                for tool in list(snapshot.get("recent_tools") or [])
                if str(tool or "").strip()
            ],
            maxlen=24,
        )

    @contextmanager
    def time_block(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_timing(name, time.perf_counter() - started)

    def get_cached_provider(self, cache_key: str) -> Any:
        return self._provider_cache.get(cache_key)

    def cache_provider(self, cache_key: str, provider: Any) -> Any:
        self._provider_cache[cache_key] = provider
        return provider

    def release_providers_for_roles(self, *roles: str) -> int:
        normalized_roles = {
            str(role or "").strip()
            for role in roles
            if str(role or "").strip()
        }
        if not normalized_roles:
            return 0

        dropped = 0
        for cache_key in list(self._provider_cache.keys()):
            provider = self._provider_cache.get(cache_key)
            provider_roles = set()
            direct_role = str(getattr(provider, "model_role", "") or "").strip()
            if direct_role:
                provider_roles.add(direct_role)
            shared_roles = getattr(provider, "_shared_model_roles", None)
            if isinstance(shared_roles, (list, tuple, set, frozenset)):
                provider_roles.update(
                    str(role or "").strip()
                    for role in shared_roles
                    if str(role or "").strip()
                )
            if not provider_roles:
                provider_roles.update(
                    role for role in normalized_roles
                    if f'"model_role": "{role}"' in cache_key
                )
            if normalized_roles.isdisjoint(provider_roles):
                continue
            provider = self._provider_cache.pop(cache_key, None)
            if provider is None:
                continue
            dropped += 1
            self._dispose_cached_provider(provider)
        if dropped:
            logger.info(
                "Released %d cached provider(s) for roles=%s",
                dropped,
                sorted(normalized_roles),
            )
        return dropped

    def _dispose_cached_provider(self, provider: Any) -> None:
        close_fn = getattr(provider, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.debug("Cached provider close() failed: %s", exc)

        for attr in ("model", "processor", "tokenizer", "_process_vision_info"):
            try:
                value = getattr(provider, attr, None)
                if value is not None:
                    setattr(provider, attr, None)
                    del value
            except Exception:
                continue

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

    def get_cached_vector_store(self, cache_key: str) -> Any:
        return self._vector_store_cache.get(cache_key)

    def cache_vector_store(self, cache_key: str, store: Any) -> Any:
        self._vector_store_cache[cache_key] = store
        return store

    def get_cached_artifact(self, cache_key: str) -> Any:
        return self._artifact_cache.get(cache_key)

    def cache_artifact(self, cache_key: str, artifact: Any) -> Any:
        self._artifact_cache[cache_key] = artifact
        return artifact

    def get_query_metrics(self) -> Dict[str, Any]:
        gpu_telemetry = self._stop_gpu_telemetry()
        return {
            "active_video_names": sorted(self._active_video_names),
            "active_video_path": self._active_video_path,
            "active_processed_dir": self._active_processed_dir,
            "active_segment_count": self._active_segment_count,
            "timing_breakdown_seconds": {
                key: round(float(value), 6)
                for key, value in sorted(self._timings.items())
            },
            "cache_stats": dict(self._cache_stats),
            "recent_tools": list(self._recent_tools),
            "gpu_telemetry": gpu_telemetry,
        }


runtime_session = RuntimeSession()
