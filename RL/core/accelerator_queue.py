from __future__ import annotations

import atexit
import logging
import os
import re
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except Exception:  # pragma: no cover
    FileLock = None  # type: ignore[assignment]
    FileLockTimeout = TimeoutError  # type: ignore[assignment]


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_ACCELERATOR_QUEUE_ROOT = PROJECT_ROOT / "rl_workdir" / "cache" / "_accelerator_queue"


def _sanitize_lock_component(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = text.replace("\\", "_").replace("/", "_").replace(":", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def resolve_shared_accelerator_queue_root(configured_root: Any = None) -> Path:
    raw_value = str(configured_root or os.environ.get("TOOL_STAR_SHARED_ACCELERATOR_QUEUE_ROOT") or "").strip()
    if raw_value:
        path = Path(raw_value).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    return DEFAULT_SHARED_ACCELERATOR_QUEUE_ROOT.resolve()


def _resolve_device_index(device: Any = None) -> Optional[int]:
    if device is not None:
        if isinstance(device, int):
            return max(0, int(device))
        index = getattr(device, "index", None)
        if index is not None:
            try:
                return int(index)
            except Exception:
                pass
        device_text = str(device).strip().lower()
        if device_text.isdigit():
            try:
                return int(device_text)
            except Exception:
                pass
        match = re.search(r"(?:cuda|npu):(\d+)", device_text)
        if match is not None:
            try:
                return int(match.group(1))
            except Exception:
                pass
        if device_text in {"cuda", "npu"}:
            return 0

    raw_local_rank = str(os.environ.get("LOCAL_RANK") or "").strip()
    if raw_local_rank:
        try:
            local_rank = int(raw_local_rank)
        except Exception:
            local_rank = -1
        if local_rank >= 0:
            return local_rank

    if torch is None or not callable(getattr(torch, "cuda", None)):
        return None
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None or not callable(getattr(cuda_module, "is_available", None)):
        return None
    if not cuda_module.is_available():
        return None
    try:
        return int(cuda_module.current_device())
    except Exception:
        return None


def resolve_local_accelerator_slot_key(device: Any = None) -> str:
    host_key = _sanitize_lock_component(socket.gethostname(), fallback="host")
    device_index = _resolve_device_index(device)
    if device_index is None:
        return f"{host_key}_cpu"

    raw_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if raw_visible_devices:
        visible_tokens = [
            _sanitize_lock_component(token, fallback=f"gpu{idx}")
            for idx, token in enumerate(raw_visible_devices.split(","))
            if str(token).strip()
        ]
        if 0 <= int(device_index) < len(visible_tokens):
            return f"{host_key}_gpu_{visible_tokens[int(device_index)]}"

    return f"{host_key}_cuda_{int(device_index)}"


class SharedAcceleratorQueueLock:
    """Reentrant local lock optionally backed by a cross-process file lock."""

    def __init__(
        self,
        *,
        name: str,
        queue_root: Any = None,
        slot_key: Optional[str] = None,
        enabled: bool = True,
        log_wait_seconds: float = 5.0,
        priority: str = "normal",
        poll_interval_seconds: float = 0.25,
        priority_marker_stale_after_seconds: float = 0.0,
    ) -> None:
        self.name = str(name or "accelerator").strip() or "accelerator"
        self.queue_root = resolve_shared_accelerator_queue_root(queue_root)
        self.slot_key = _sanitize_lock_component(
            slot_key or resolve_local_accelerator_slot_key(),
            fallback="slot",
        )
        self.enabled = bool(enabled)
        self.log_wait_seconds = max(0.0, float(log_wait_seconds))
        normalized_priority = str(priority or "normal").strip().lower()
        self.priority = normalized_priority if normalized_priority in {"normal", "high"} else "normal"
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.priority_marker_stale_after_seconds = max(0.0, float(priority_marker_stale_after_seconds))
        self._local_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._depth = 0
        self._file_lock = None
        self._legacy_priority_gate_path: Optional[Path] = None
        self._priority_marker_prefix = f"{self.slot_key}.priority."
        self._owned_priority_markers: list[Path] = []

        if self.enabled and FileLock is not None:
            self.queue_root.mkdir(parents=True, exist_ok=True)
            lock_filename = f"{self.slot_key}.accelerator.lock"
            self._file_lock = FileLock(str(self.queue_root / lock_filename))
            self._legacy_priority_gate_path = self.queue_root / f"{self.slot_key}.priority"
            atexit.register(self._cleanup_owned_priority_markers)
        elif self.enabled and FileLock is None:
            logger.warning(
                "Shared accelerator queue requested for name=%s slot=%s but filelock is unavailable; "
                "falling back to in-process locking only.",
                self.name,
                self.slot_key,
            )

    def _iter_priority_marker_paths(self) -> list[Path]:
        if self._file_lock is None:
            return []
        try:
            return [path for path in self.queue_root.glob(f"{self._priority_marker_prefix}*") if path.is_file()]
        except Exception:
            return []

    def _read_priority_marker_metadata(self, marker_path: Path) -> dict[str, str]:
        metadata: dict[str, str] = {}
        try:
            for raw_line in marker_path.read_text(encoding="utf-8").splitlines():
                if "=" not in raw_line:
                    continue
                key, value = raw_line.split("=", 1)
                metadata[str(key).strip()] = str(value).strip()
        except Exception:
            return metadata
        return metadata

    def _pid_is_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return True
        return True

    def _cleanup_owned_priority_markers(self) -> None:
        owned_paths: list[Path] = []
        with self._state_lock:
            if self._owned_priority_markers:
                owned_paths = list(self._owned_priority_markers)
                self._owned_priority_markers.clear()
        for marker_path in owned_paths:
            self._remove_priority_marker(marker_path)

    def _should_remove_stale_priority_marker(self, marker_path: Path) -> bool:
        metadata = self._read_priority_marker_metadata(marker_path)
        owner_pid_text = metadata.get("pid", "")
        owner_host = _sanitize_lock_component(metadata.get("host", ""), fallback="")
        current_host = _sanitize_lock_component(socket.gethostname(), fallback="")
        if owner_host and current_host and owner_host != current_host:
            return False
        try:
            owner_pid = int(owner_pid_text)
        except Exception:
            owner_pid = -1
        if owner_pid > 0 and not self._pid_is_alive(owner_pid):
            logger.warning(
                "[AcceleratorQueue][WARN] removing stale priority marker=%s for dead pid=%s slot=%s",
                marker_path,
                owner_pid,
                self.slot_key,
            )
            return True

        if self.priority_marker_stale_after_seconds > 0:
            try:
                age_seconds = max(0.0, time.time() - float(marker_path.stat().st_mtime))
            except Exception:
                age_seconds = 0.0
            if age_seconds >= self.priority_marker_stale_after_seconds:
                logger.warning(
                    "[AcceleratorQueue][WARN] removing expired priority marker=%s age=%.1fs slot=%s stale_after=%.1fs",
                    marker_path,
                    age_seconds,
                    self.slot_key,
                    self.priority_marker_stale_after_seconds,
                )
                return True
        return False

    def _prune_stale_priority_markers(self) -> None:
        for marker_path in self._iter_priority_marker_paths():
            if self._should_remove_stale_priority_marker(marker_path):
                self._remove_priority_marker(marker_path)

    def _prune_legacy_priority_gate(self) -> None:
        legacy_gate_path = self._legacy_priority_gate_path
        if legacy_gate_path is None or not legacy_gate_path.exists():
            return
        logger.warning(
            "[AcceleratorQueue][WARN] removing legacy priority gate=%s slot=%s; current queue implementation uses per-request markers only.",
            legacy_gate_path,
            self.slot_key,
        )
        self._remove_priority_marker(legacy_gate_path)

    def _priority_gate_exists(self) -> bool:
        self._prune_stale_priority_markers()
        self._prune_legacy_priority_gate()
        if self._legacy_priority_gate_path is not None and self._legacy_priority_gate_path.exists():
            return True
        try:
            return any(self.queue_root.glob(f"{self._priority_marker_prefix}*"))
        except Exception:
            return False

    def _create_priority_marker(self, *, reason: str) -> Optional[Path]:
        if self._file_lock is None:
            return None
        self.queue_root.mkdir(parents=True, exist_ok=True)
        marker_path = self.queue_root / (
            f"{self._priority_marker_prefix}{reason}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        )
        marker_path.write_text(
            (
                f"name={self.name}\n"
                f"pid={os.getpid()}\n"
                f"host={socket.gethostname()}\n"
                f"started_at={time.time():.6f}\n"
            ),
            encoding="utf-8",
        )
        return marker_path

    def _remove_priority_marker(self, marker_path: Optional[Path]) -> None:
        if marker_path is None:
            return
        try:
            marker_path.unlink(missing_ok=True)
        except Exception:
            try:
                if marker_path.exists():
                    marker_path.unlink()
            except Exception:
                pass

    def _wait_for_priority_gate_to_clear(self) -> None:
        while self._priority_gate_exists():
            time.sleep(self.poll_interval_seconds)

    def _acquire_file_lock_with_priority(self) -> None:
        if self._file_lock is None:
            return
        while True:
            if self.priority != "high":
                self._wait_for_priority_gate_to_clear()
            try:
                self._file_lock.acquire(timeout=self.poll_interval_seconds)
                return
            except FileLockTimeout:
                continue

    def acquire(self) -> "SharedAcceleratorQueueLock":
        started = time.perf_counter()
        self._local_lock.acquire()
        acquired_file_lock = False
        priority_marker_path: Optional[Path] = None
        try:
            should_acquire_file_lock = False
            with self._state_lock:
                if self._depth == 0 and self._file_lock is not None:
                    should_acquire_file_lock = True
                self._depth += 1

            if should_acquire_file_lock:
                if self.priority == "high":
                    priority_marker_path = self._create_priority_marker(reason="acquire")
                    if priority_marker_path is not None:
                        with self._state_lock:
                            self._owned_priority_markers.append(priority_marker_path)
                self._acquire_file_lock_with_priority()
                acquired_file_lock = True
                waited_seconds = max(0.0, time.perf_counter() - started)
                if waited_seconds >= self.log_wait_seconds:
                    logger.info(
                        "[AcceleratorQueue][DIAG] name=%s slot=%s priority=%s waited %.3fs for shared accelerator slot lock=%s",
                        self.name,
                        self.slot_key,
                        self.priority,
                        waited_seconds,
                        self._file_lock.lock_file,
                    )
            return self
        except Exception:
            if acquired_file_lock and self._file_lock is not None:
                try:
                    self._file_lock.release()
                except Exception:
                    pass
            if priority_marker_path is not None:
                with self._state_lock:
                    try:
                        self._owned_priority_markers.remove(priority_marker_path)
                    except ValueError:
                        pass
                self._remove_priority_marker(priority_marker_path)
            with self._state_lock:
                self._depth = max(0, self._depth - 1)
            self._local_lock.release()
            raise

    def release(self) -> None:
        should_release_file_lock = False
        priority_marker_path: Optional[Path] = None
        with self._state_lock:
            if self._depth > 0:
                self._depth -= 1
            if self._depth == 0 and self._file_lock is not None:
                should_release_file_lock = True
                if self.priority == "high" and self._owned_priority_markers:
                    priority_marker_path = self._owned_priority_markers.pop()

        try:
            if should_release_file_lock:
                self._file_lock.release()
            if priority_marker_path is not None:
                self._remove_priority_marker(priority_marker_path)
        finally:
            self._local_lock.release()

    def __enter__(self) -> "SharedAcceleratorQueueLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
