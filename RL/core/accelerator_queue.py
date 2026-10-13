from __future__ import annotations

import atexit
import logging
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# Torch is imported lazily only when the caller asks us to infer a device.  The
# batch scheduler passes an explicit visible device id, so queue admission must
# not make the CPU-side scheduler load the multimodal runtime.
torch = None  # type: ignore[assignment]

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except Exception:  # pragma: no cover
    FileLock = None  # type: ignore[assignment]
    FileLockTimeout = TimeoutError  # type: ignore[assignment]


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_ACCELERATOR_QUEUE_ROOT = PROJECT_ROOT / "rl_workdir" / "cache" / "_accelerator_queue"


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = str(os.environ.get(name) or "").strip().lower()
    if not raw_value:
        return bool(default)
    if raw_value in {"1", "true", "yes", "y", "on"}:
        return True
    if raw_value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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

    global torch
    if torch is None:
        try:
            import torch as torch_module

            torch = torch_module
        except Exception:  # pragma: no cover
            torch = None
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
        capacity: Optional[int] = None,
        memory_guard_enabled: Optional[bool] = None,
        memory_min_free_mib: Optional[float] = None,
        memory_reserve_mib: Optional[float] = None,
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
        raw_capacity = capacity
        if raw_capacity is None:
            raw_capacity = (
                os.environ.get("TOOLSTAR_ACCELERATOR_QUEUE_CAPACITY")
                or os.environ.get("TOOLSTAR_BATCH_ACCELERATOR_QUEUE_CAPACITY")
                or 1
            )
        try:
            normalized_capacity = int(raw_capacity)
        except (TypeError, ValueError):
            normalized_capacity = 1
        # Every slot represents an independent model working set. Keep the
        # upper bound conservative and require an explicit caller opt-in.
        self.requested_capacity = max(1, min(8, normalized_capacity))
        self.memory_guard_enabled = (
            bool(memory_guard_enabled)
            if memory_guard_enabled is not None
            else _env_bool("TOOLSTAR_ACCELERATOR_QUEUE_MEMORY_GUARD", False)
        )
        self.memory_guard_scope = self.name in {
            "mix_sample_worker_admission",
            # Generation is a separate, short-lived memory phase.  It needs
            # the same cross-process free-memory wait even though the resident
            # worker admission slot is already held by the caller.
            "transformers_generation",
        }
        raw_min_free_mib = str(
            os.environ.get("TOOLSTAR_ACCELERATOR_QUEUE_MIN_FREE_MIB") or ""
        ).strip()
        raw_min_free_gib = str(
            os.environ.get("TOOLSTAR_ACCELERATOR_QUEUE_MIN_FREE_GIB") or ""
        ).strip()
        try:
            if memory_min_free_mib is not None:
                min_free_mib = float(memory_min_free_mib)
            elif raw_min_free_mib:
                min_free_mib = float(raw_min_free_mib)
            elif raw_min_free_gib:
                min_free_mib = float(raw_min_free_gib) * 1024.0
            else:
                min_free_mib = 4096.0 if self.memory_guard_enabled else 0.0
        except (TypeError, ValueError):
            min_free_mib = 4096.0 if self.memory_guard_enabled else 0.0
        self.memory_min_free_mib = max(0.0, min_free_mib)
        raw_reserve_mib = str(
            os.environ.get("TOOLSTAR_ACCELERATOR_QUEUE_RESERVE_MIB") or ""
        ).strip()
        raw_reserve_gib = str(
            os.environ.get("TOOLSTAR_ACCELERATOR_QUEUE_RESERVE_GIB") or ""
        ).strip()
        try:
            if memory_reserve_mib is not None:
                reserve_mib = float(memory_reserve_mib)
            elif raw_reserve_mib:
                reserve_mib = float(raw_reserve_mib)
            elif raw_reserve_gib:
                reserve_mib = float(raw_reserve_gib) * 1024.0
            else:
                reserve_mib = 0.0
        except (TypeError, ValueError):
            reserve_mib = 0.0
        self.memory_reserve_mib = max(0.0, reserve_mib)
        self.memory_admission_threshold_mib = self.memory_min_free_mib + self.memory_reserve_mib
        # A memory guard controls admission timing; it must not silently turn
        # the requested persistent-worker topology into serial execution.
        # Logical workers can therefore remain at three while only workers
        # whose current free-memory budget is safe acquire a resident slot.
        self.capacity = self.requested_capacity
        self.memory_guard_forced_serial = False
        self._local_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._depth = 0
        self._file_lock = None
        self._candidate_file_locks: list[Any] = []
        self._selected_file_lock_index: Optional[int] = None
        self._legacy_priority_gate_path: Optional[Path] = None
        self._priority_marker_prefix = f"{self.slot_key}.priority."
        self._owned_priority_markers: list[Path] = []

        if self.enabled and FileLock is not None:
            self.queue_root.mkdir(parents=True, exist_ok=True)
            if self.capacity <= 1:
                lock_filename = f"{self.slot_key}.accelerator.lock"
                self._file_lock = FileLock(str(self.queue_root / lock_filename))
                self._legacy_priority_gate_path = self.queue_root / f"{self.slot_key}.priority"
            else:
                # Capacity-N admission uses N independent lock files. A
                # process atomically owns one file, so total model working
                # sets on a physical device never exceed the configured cap.
                self._candidate_file_locks = [
                    FileLock(
                        str(self.queue_root / f"{self.slot_key}.accelerator.{index}.lock")
                    )
                    for index in range(self.capacity)
                ]
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
        if self._candidate_file_locks:
            while True:
                for index, candidate_lock in enumerate(self._candidate_file_locks):
                    try:
                        candidate_lock.acquire(timeout=self.poll_interval_seconds)
                    except FileLockTimeout:
                        continue
                    self._file_lock = candidate_lock
                    self._selected_file_lock_index = index
                    return
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

    def _memory_guard_device_index(self) -> Optional[int]:
        raw_device = str(
            os.environ.get("TOOLSTAR_DEVICE_ID")
            or os.environ.get("LOCAL_RANK")
            or ""
        ).strip()
        if raw_device.isdigit():
            return int(raw_device)
        raw_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
        if raw_visible_devices:
            first_token = raw_visible_devices.split(",", 1)[0].strip()
            if first_token.isdigit():
                return int(first_token)
        return None

    def _query_free_memory_mib(self) -> Optional[float]:
        """Return free memory for the visible GPU, or None if unavailable.

        A missing/unusable nvidia-smi must not deadlock CPU scheduling.  The
        queue remains the correctness boundary in that environment and the
        guard becomes diagnostic-only.
        """
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        rows: list[tuple[int, float]] = []
        for line in str(completed.stdout or "").splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 2:
                continue
            try:
                rows.append((int(fields[0]), float(fields[1])))
            except (TypeError, ValueError):
                continue
        if not rows:
            return None
        target_index = self._memory_guard_device_index()
        if target_index is not None:
            for index, free_mib in rows:
                if index == target_index:
                    return free_mib
        return rows[0][1]

    def _wait_for_memory_before_admission(self) -> None:
        if not (
            self.memory_guard_enabled
            and self.memory_guard_scope
            and self.memory_admission_threshold_mib > 0
        ):
            return
        wait_started = time.perf_counter()
        last_free_mib: Optional[float] = None
        while True:
            free_mib = self._query_free_memory_mib()
            if free_mib is None or free_mib >= self.memory_admission_threshold_mib:
                waited_seconds = max(0.0, time.perf_counter() - wait_started)
                if waited_seconds >= self.poll_interval_seconds:
                    logger.info(
                        "[AcceleratorQueue][MEMORY_GUARD] admitted name=%s slot=%s "
                        "free_mib=%s min_free_mib=%.0f reserve_mib=%.0f threshold_mib=%.0f waited=%.3fs",
                        self.name,
                        self.slot_key,
                        "unavailable" if free_mib is None else f"{free_mib:.0f}",
                        self.memory_min_free_mib,
                        self.memory_reserve_mib,
                        self.memory_admission_threshold_mib,
                        waited_seconds,
                    )
                return
            if last_free_mib is None or abs(free_mib - last_free_mib) >= 256:
                logger.info(
                    "[AcceleratorQueue][MEMORY_GUARD] waiting name=%s slot=%s "
                    "free_mib=%.0f min_free_mib=%.0f reserve_mib=%.0f threshold_mib=%.0f",
                    self.name,
                    self.slot_key,
                    free_mib,
                    self.memory_min_free_mib,
                    self.memory_reserve_mib,
                    self.memory_admission_threshold_mib,
                )
                last_free_mib = free_mib
            time.sleep(self.poll_interval_seconds)

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
                elif self._depth == 0 and self._candidate_file_locks:
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
                self._wait_for_memory_before_admission()
                waited_seconds = max(0.0, time.perf_counter() - started)
                if waited_seconds >= self.log_wait_seconds:
                    logger.info(
                        "[AcceleratorQueue][DIAG] name=%s slot=%s capacity=%s priority=%s waited %.3fs for shared accelerator slot lock=%s",
                        self.name,
                        self.slot_key,
                        self.capacity,
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
                if self._candidate_file_locks:
                    self._file_lock = None
                    self._selected_file_lock_index = None
            if priority_marker_path is not None:
                self._remove_priority_marker(priority_marker_path)
        finally:
            self._local_lock.release()

    def __enter__(self) -> "SharedAcceleratorQueueLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
