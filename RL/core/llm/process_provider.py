"""Local process-backed LLM provider for non-planner inference."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional

from ..accelerator_queue import SharedAcceleratorQueueLock, resolve_local_accelerator_slot_key
from .base import BaseLLM, is_accelerator_fatal_error
from config.device_config import detect_device, get_device_string

logger = logging.getLogger(__name__)


def _sanitize_ipc_payload(value: Any, *, path: str = "payload") -> tuple[Any, List[str]]:
    dropped_paths: List[str] = []

    if isinstance(value, dict):
        sanitized_dict: Dict[str, Any] = {}
        for key, item in value.items():
            child_path = f"{path}.{key}"
            sanitized_item, child_dropped = _sanitize_ipc_payload(item, path=child_path)
            dropped_paths.extend(child_dropped)
            if sanitized_item is _DROP_FROM_IPC:
                continue
            sanitized_dict[key] = sanitized_item
        return sanitized_dict, dropped_paths

    if isinstance(value, list):
        sanitized_list: List[Any] = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            sanitized_item, child_dropped = _sanitize_ipc_payload(item, path=child_path)
            dropped_paths.extend(child_dropped)
            if sanitized_item is _DROP_FROM_IPC:
                continue
            sanitized_list.append(sanitized_item)
        return sanitized_list, dropped_paths

    if isinstance(value, tuple):
        sanitized_list: List[Any] = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            sanitized_item, child_dropped = _sanitize_ipc_payload(item, path=child_path)
            dropped_paths.extend(child_dropped)
            if sanitized_item is _DROP_FROM_IPC:
                continue
            sanitized_list.append(sanitized_item)
        return tuple(sanitized_list), dropped_paths

    if callable(value):
        dropped_paths.append(path)
        return _DROP_FROM_IPC, dropped_paths

    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        dropped_paths.append(path)
        return _DROP_FROM_IPC, dropped_paths

    return value, dropped_paths


_DROP_FROM_IPC = object()


def _resolve_process_local_device(requested_device: Any = None) -> Optional[str]:
    if isinstance(requested_device, int):
        requested_device = max(0, int(requested_device))
    requested_text = str(requested_device or "").strip()
    if isinstance(requested_device, int) or requested_text.isdigit():
        device_type, device_count = detect_device()
        if device_type in {"cuda", "npu"}:
            requested_index = int(requested_device if isinstance(requested_device, int) else requested_text)
            if device_count is not None and device_count > 0:
                requested_index = min(requested_index, int(device_count) - 1)
            return get_device_string(device_type, requested_index)

    resolved_text = str(requested_device or "").strip()
    if ":" in resolved_text:
        return resolved_text
    if resolved_text == "cpu":
        return "cpu"
    if resolved_text in {"cuda", "npu"}:
        device_type, device_count = detect_device()
        normalized_type = resolved_text if resolved_text in {"cuda", "npu"} else device_type
        if normalized_type in {"cuda", "npu"}:
            if device_count is not None and device_count > 0:
                return get_device_string(normalized_type, 0)
            return get_device_string(normalized_type, 0)

    raw_local_rank = str(os.environ.get("LOCAL_RANK") or "").strip()
    if raw_local_rank:
        try:
            local_rank = int(raw_local_rank)
        except Exception:
            local_rank = -1
        if local_rank >= 0:
            device_type, device_count = detect_device()
            if device_type in {"cuda", "npu"}:
                if device_count is not None and device_count > 0:
                    local_rank = min(local_rank, int(device_count) - 1)
                return get_device_string(device_type, local_rank)

    if resolved_text:
        return resolved_text
    return None


def _run_transformers_worker(
    provider_kwargs: Dict[str, Any],
    request_queue: Any,
    response_queue: Any,
) -> None:
    """Run a lazy TransformersProvider inside a dedicated child process."""
    provider = None
    provider_init_kwargs = dict(provider_kwargs or {})
    resolved_process_device = _resolve_process_local_device(provider_kwargs.get("device"))
    if resolved_process_device:
        provider_init_kwargs["device"] = resolved_process_device
    provider_init_kwargs["enable_shared_accelerator_queue"] = False
    try:
        queue_log_wait_seconds = float(provider_kwargs.get("shared_accelerator_queue_log_wait_seconds", 5.0) or 5.0)
    except Exception:
        queue_log_wait_seconds = 5.0
    accelerator_queue_lock = SharedAcceleratorQueueLock(
        name=f"process_provider_{str(provider_kwargs.get('model_role') or 'default').strip() or 'default'}",
        queue_root=provider_kwargs.get("shared_accelerator_queue_root"),
        slot_key=resolve_local_accelerator_slot_key(resolved_process_device),
        enabled=bool(provider_kwargs.get("enable_shared_accelerator_queue", True)),
        log_wait_seconds=queue_log_wait_seconds,
    )
    logger.info(
        "Process-backed worker bootstrap role=%s requested_device=%s resolved_device=%s slot=%s queue_enabled=%s",
        str(provider_kwargs.get("model_role") or "default").strip() or "default",
        provider_kwargs.get("device"),
        resolved_process_device or "<auto>",
        accelerator_queue_lock.slot_key,
        bool(provider_kwargs.get("enable_shared_accelerator_queue", True)),
    )
    try:
        while True:
            request = request_queue.get()
            if not isinstance(request, dict):
                continue
            request_id = str(request.get("request_id") or "")
            method = str(request.get("method") or "").strip().lower()
            logger.info(
                "Process-backed worker received role=%s method=%s request_id=%s provider_ready=%s",
                str(provider_kwargs.get("model_role") or "default").strip() or "default",
                method,
                request_id,
                provider is not None,
            )
            if method == "close":
                response_queue.put({"request_id": request_id, "ok": True, "result": None})
                return
            if provider is None:
                from .provider import TransformersProvider

                with accelerator_queue_lock:
                    init_started = time.perf_counter()
                    provider = TransformersProvider(**provider_init_kwargs)
                    logger.info(
                        "Process-backed worker initialized provider role=%s model=%s duration=%.3fs runtime_device=%s",
                        str(provider_kwargs.get("model_role") or "default").strip() or "default",
                        provider_kwargs.get("model_name"),
                        time.perf_counter() - init_started,
                        getattr(provider, "runtime_device", provider_init_kwargs.get("device")),
                    )
            try:
                if method == "generate":
                    with accelerator_queue_lock:
                        generate_started = time.perf_counter()
                        result = provider.generate(
                            request.get("messages") or [],
                            temperature=request.get("temperature"),
                            max_tokens=request.get("max_tokens"),
                            **dict(request.get("kwargs") or {}),
                        )
                        logger.info(
                            "Process-backed worker completed generate role=%s request_id=%s duration=%.3fs",
                            str(provider_kwargs.get("model_role") or "default").strip() or "default",
                            request_id,
                            time.perf_counter() - generate_started,
                        )
                elif method == "count_tokens":
                    result = provider.count_tokens(str(request.get("text") or ""))
                elif method == "count_message_tokens":
                    result = provider.count_message_tokens(request.get("messages") or [])
                elif method == "get_max_context_length":
                    result = provider.get_max_context_length()
                else:
                    raise ValueError(f"Unsupported process-backed LLM method: {method}")
                response_queue.put({"request_id": request_id, "ok": True, "result": result})
            except Exception as exc:
                response_queue.put(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "fatal": bool(is_accelerator_fatal_error(exc)),
                    }
                )
    except KeyboardInterrupt:
        return
    finally:
        try:
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass


class ProcessBackedLLMProvider(BaseLLM):
    """Run a local TransformersProvider in a separate spawned process."""

    def __init__(
        self,
        model_name: str,
        model_role: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, model_role=model_role, **kwargs)
        self.model_role = str(model_role or "").strip()
        self._provider_kwargs = {"model_name": model_name, "model_role": model_role, **kwargs}
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._lock = threading.Lock()
        self._prepared = False

    def _queue_cleanup(self, handle: Any) -> None:
        if handle is None:
            return
        for method_name in ("close", "join_thread", "cancel_join_thread"):
            method = getattr(handle, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception:
                continue

    def _dead_worker_message_locked(self, *, action: str) -> str:
        process = self._process
        exitcode = getattr(process, "exitcode", None) if process is not None else None
        pid = getattr(process, "pid", None) if process is not None else None
        return (
            f"Process-backed LLM worker exited while {action} role={self.model_role or 'default'} "
            f"model={self.model_name} pid={pid} exitcode={exitcode}."
        )

    def _start_process_locked(self) -> None:
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_run_transformers_worker,
            args=(self._provider_kwargs, self._request_queue, self._response_queue),
            daemon=False,
            name=f"tool_star_llm_{self.model_role or 'default'}",
        )
        self._process.start()
        logger.info(
            "Started process-backed LLM role=%s pid=%s model=%s",
            self.model_role or "default",
            getattr(self._process, "pid", None),
            self.model_name,
        )

    def _ensure_process_locked(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        if self._process is not None:
            logger.warning(
                "Process-backed LLM worker died; restarting role=%s model=%s pid=%s exitcode=%s",
                self.model_role or "default",
                self.model_name,
                getattr(self._process, "pid", None),
                getattr(self._process, "exitcode", None),
            )
            self._shutdown_locked(force=True)
        self._start_process_locked()

    def _shutdown_locked(self, *, force: bool) -> None:
        process = self._process
        request_queue = self._request_queue
        response_queue = self._response_queue
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._prepared = False
        if process is None:
            return
        try:
            if not force and process.is_alive() and request_queue is not None:
                request_queue.put({"request_id": str(uuid.uuid4()), "method": "close"})
                process.join(timeout=5.0)
        except Exception:
            pass
        if process.is_alive():
            try:
                process.terminate()
                process.join(timeout=5.0)
            except Exception:
                pass
        for handle in (request_queue, response_queue):
            self._queue_cleanup(handle)

    def close(self) -> None:
        with self._lock:
            self._shutdown_locked(force=False)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _request(self, method: str, **payload: Any) -> Any:
        with self._lock:
            self._ensure_process_locked()
            request_id = str(uuid.uuid4())
            assert self._request_queue is not None
            assert self._response_queue is not None
            started = time.perf_counter()
            last_wait_log = started
            request_payload, dropped_paths = _sanitize_ipc_payload(
                {"request_id": request_id, "method": method, **payload},
                path="request",
            )
            if dropped_paths:
                logger.warning(
                    "Dropped non-IPC-safe fields before sending request to process-backed LLM role=%s model=%s method=%s dropped=%s",
                    self.model_role or "default",
                    self.model_name,
                    method,
                    dropped_paths,
                )
            try:
                self._request_queue.put(request_payload)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to enqueue request for process-backed LLM role={self.model_role or 'default'} "
                    f"model={self.model_name} method={method}: {exc}"
                ) from exc
            logger.info(
                "Process-backed LLM request enqueued role=%s model=%s method=%s pid=%s",
                self.model_role or "default",
                self.model_name,
                method,
                getattr(self._process, "pid", None),
            )
            while True:
                try:
                    response = self._response_queue.get(timeout=1.0)
                except queue.Empty:
                    if self._process is None or not self._process.is_alive():
                        dead_worker_message = self._dead_worker_message_locked(action=f"handling method={method}")
                        self._shutdown_locked(force=True)
                        raise RuntimeError(dead_worker_message)
                    now = time.perf_counter()
                    if (now - last_wait_log) >= 30.0:
                        logger.info(
                            "Process-backed LLM still waiting role=%s model=%s method=%s pid=%s elapsed=%.1fs",
                            self.model_role or "default",
                            self.model_name,
                            method,
                            getattr(self._process, "pid", None),
                            now - started,
                        )
                        last_wait_log = now
                    continue
                if str(response.get("request_id") or "") != request_id:
                    continue
                if bool(response.get("ok")):
                    return response.get("result")
                if bool(response.get("fatal")):
                    self._shutdown_locked(force=True)
                error_message = str(response.get("error") or "Unknown process-backed LLM error.")
                error_traceback = str(response.get("traceback") or "").strip()
                if error_traceback:
                    raise RuntimeError(f"{error_message}\n{error_traceback}")
                raise RuntimeError(error_message)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        started = time.perf_counter()
        result = self._request(
            "generate",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            kwargs=kwargs,
        )
        logger.info(
            "Process-backed LLM completed role=%s model=%s duration=%.3fs",
            self.model_role or "default",
            self.model_name,
            time.perf_counter() - started,
        )
        return str(result or "")

    def prepare(self) -> None:
        if self._prepared:
            return
        started = time.perf_counter()
        self._request("get_max_context_length")
        self._prepared = True
        logger.info(
            "Process-backed LLM prepared role=%s model=%s duration=%.3fs",
            self.model_role or "default",
            self.model_name,
            time.perf_counter() - started,
        )

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self.generate(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def count_tokens(self, text: str) -> int:
        try:
            return int(self._request("count_tokens", text=str(text or "")) or 0)
        except Exception:
            return super().count_tokens(text)

    def count_message_tokens(self, messages: List[Dict[str, str]], **kwargs: Any) -> int:
        del kwargs
        try:
            return int(self._request("count_message_tokens", messages=messages) or 0)
        except Exception:
            return super().count_message_tokens(messages)

    def get_max_context_length(self) -> int:
        try:
            return int(self._request("get_max_context_length") or 8192)
        except Exception:
            return super().get_max_context_length()
