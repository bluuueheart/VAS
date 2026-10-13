"""Online Evolver controller for the mixed benchmark scheduler.

The controller is deliberately a small process boundary.  A completed sample
is first committed to the normal Mix checkpoint, then accumulated into a
bounded evolution window.  At a checkpoint boundary a child process mines a
proposal, runs the paired trigger/neighbor/regression/held-out gate, and
returns an accepted snapshot.  The parent publishes that snapshot with
``os.replace`` so workers can observe one immutable runtime version at a time.

Keeping proposal and replay work outside the scheduler avoids importing a
second multimodal model into the process that owns the live worker.  The
active snapshot pointer is intentionally a single JSON path; workers reload it
between samples and never change an in-flight trajectory.
"""

from __future__ import annotations

import json
import fcntl
import os
import subprocess
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence


TERMINAL_STATUSES = frozenset({"completed", "wrong", "unanswerable", "failed"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _option(options: Mapping[str, Any], key: str, default: Any) -> Any:
    value = options.get(key, default)
    return default if value is None else value


def _option_with_env(
    options: Mapping[str, Any],
    key: str,
    env_name: str,
    default: Any,
) -> Any:
    raw_env = str(os.environ.get(env_name) or "").strip()
    return raw_env if raw_env else _option(options, key, default)


def _bounded_int(options: Mapping[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_option(options, key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_int_with_env(
    options: Mapping[str, Any],
    key: str,
    env_name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    return _bounded_int(
        {key: _option_with_env(options, key, env_name, default)},
        key,
        default,
        minimum,
        maximum,
    )


def _process_identity(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else ""
    except (OSError, IndexError):
        return ""


class _RecoveredCycle:
    """Observe an orphaned, identity-checked cycle without launching a duplicate."""
    def __init__(self, record):
        self.pid = int(record["pid"])
        self.identity = record["identity"]
        self.output = Path(record["root"]) / "cycle_result.json"
        self.started_at = float(record.get("started_at") or 0)
        self.returncode = None

    def poll(self):
        if not self.pid:
            root = self.output.parent
            with (root / 'launch.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                lease = root / 'cycle_lease.json'
                if lease.is_file():
                    record = _read_json(lease)
                    self.pid, self.identity = int(record['pid']), record['identity']
                elif time.time() - self.started_at < 10:
                    return None
                else:
                    (root / 'launch_cancelled').touch()
        if self.identity and _process_identity(self.pid) == self.identity:
            return None
        try:
            payload = _read_json(self.output) if self.output.is_file() else {}
        except (OSError, ValueError):
            payload = {}
        self.returncode = 0 if payload.get("cycle_committed") is True else 1
        return self.returncode

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("recovered cycle", timeout)
            time.sleep(.1)
        return self.returncode


class OnlineEvolutionController:
    """Collect terminal samples and publish replay-gated snapshots online."""

    def __init__(
        self,
        *,
        config_path: Path,
        workspaces_root: Path,
        evolution_config: Optional[Path] = None,
        output_root: Optional[Path] = None,
        baseline_path: Optional[Path] = None,
        visible_device_id: str = "0",
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.workspaces_root = Path(workspaces_root).resolve()
        self.options = dict(options or {})
        # Online evolution is opt-out. A caller can still disable it for a
        # measured baseline by setting ``online_evolution=false``.
        raw_enabled = _option(self.options, "online_evolution", True)
        explicit_env = str(os.environ.get("ONLINE_EVOLUTION") or "").strip().lower()
        if explicit_env:
            raw_enabled = explicit_env in {"1", "true", "yes", "on"}
        if isinstance(raw_enabled, str):
            raw_enabled = raw_enabled.strip().lower() in {"1", "true", "yes", "on"}
        self.enabled = bool(raw_enabled)
        default_root = self.workspaces_root / "_online_evolution"
        self.root = Path(output_root or default_root).resolve()
        # Keep each cycle as the first directory below the audit root. The
        # existing Evolver auditor can then pair one report with that cycle's
        # replay/suffix artifacts instead of merging every cycle together.
        self.cycles_root = self.root
        self.registry_root = self.root / "registry"
        self.state_path = self.root / "online_state.json"
        self.events_path = self.root / "online_events.jsonl"
        self.records_path = self.root / "online_records.json"
        self.active_snapshot_path = self.root / "active_snapshot.json"
        self.evolution_config = Path(
            evolution_config
            or (Path(__file__).resolve().parents[1] / "config" / "evolution.yaml")
        ).resolve()
        from .verifier import load_evolution_config
        self.excessive_tool_calls = int(load_evolution_config(self.evolution_config).get("diagnosis", {}).get("excessive_tool_calls", 8))
        self.baseline_path = Path(baseline_path).resolve() if baseline_path else None
        self.visible_device_id = str(visible_device_id or "0")
        self.trigger_every = _bounded_int_with_env(
            self.options, "online_evolution_trigger_every", "ONLINE_EVOLUTION_TRIGGER_EVERY", 8, 1, 256
        )
        self.window_size = _bounded_int_with_env(
            self.options, "online_evolution_window", "ONLINE_EVOLUTION_WINDOW", 32, 4, 256
        )
        self.group_size = _bounded_int_with_env(
            self.options, "online_evolution_group_size", "ONLINE_EVOLUTION_GROUP_SIZE", 256, 1, 1024
        )
        self.timeout_seconds = _bounded_int_with_env(
            self.options, "online_evolution_timeout_seconds", "ONLINE_EVOLUTION_TIMEOUT_SECONDS", 7200, 60, 172800
        )
        poll_value = _option_with_env(
            self.options,
            "online_evolution_poll_seconds",
            "ONLINE_EVOLUTION_POLL_SECONDS",
            0.25,
        )
        try:
            self.poll_seconds = min(2.0, max(0.1, float(poll_value)))
        except (TypeError, ValueError):
            self.poll_seconds = 0.25
        self._lock = RLock()
        self._records: Dict[str, Dict[str, Any]] = {}
        self._last_trigger_count = 0
        self._pending_cycle = False
        self._cycle_index = 0
        self._process: Optional[subprocess.Popen[Any]] = None
        self._process_cycle_root: Optional[Path] = None
        self._process_started_at = 0.0
        self._active_snapshot_id = "snapshot_000"
        self._cycle_summaries: list[Dict[str, Any]] = []
        self._closed = False
        self.worker_timeout_seconds = self.timeout_seconds
        self.timeout_seconds = _bounded_int_with_env(
            self.options, "online_evolution_cycle_timeout_seconds",
            "ONLINE_EVOLUTION_CYCLE_TIMEOUT_SECONDS", 172800, 60, 1209600
        )
        self._pending_candidate: Dict[str, Any] = {}
        self._monitor: Dict[str, Any] = {}
        self._last_monitor_count = 0
        self._inflight: Dict[str, Any] = {}
        if self.state_path.is_file():
            previous = _read_json(self.state_path)
            self._last_trigger_count = int(previous.get("last_trigger_count") or 0)
            self._pending_cycle = bool(previous.get("pending_cycle"))
            self._cycle_summaries = list(previous.get("cycles") or [])
            self._pending_candidate = dict(previous.get("pending_candidate") or {})
            self._monitor = dict(previous.get("monitor") or {})
            self._last_monitor_count = int(previous.get("last_monitor_count") or 0)
            self._inflight = dict(previous.get("inflight") or {})
            if self._inflight:
                self._process = _RecoveredCycle(self._inflight)
                self._process_cycle_root = Path(self._inflight["root"])
                self._process_started_at = time.monotonic() - max(0, time.time() - float(self._inflight["started_at"]))
        if self.records_path.is_file():
            self._records = dict(_read_json(self.records_path).get("records") or {})
        self.root.mkdir(parents=True, exist_ok=True)
        self.cycles_root.mkdir(parents=True, exist_ok=True)
        self.registry_root.mkdir(parents=True, exist_ok=True)
        existing_cycles = []
        for path in self.root.glob("evolution_online_cycle_*"):
            try:
                existing_cycles.append(int(path.name.rsplit("_", 1)[-1]))
            except (TypeError, ValueError):
                continue
        self._cycle_index = max(existing_cycles or [0])
        if self.active_snapshot_path.is_file():
            try:
                active_payload = _read_json(self.active_snapshot_path)
                if str(active_payload.get("status") or "").lower() == "accepted":
                    self._active_snapshot_id = str(
                        active_payload.get("snapshot_id") or self._active_snapshot_id
                    )
                    if self._monitor.get("snapshot_id") != self._active_snapshot_id:
                        publication = self.registry_root / f"{self._active_snapshot_id}_accepted" / "publication.json"
                        if publication.is_file():
                            saved = _read_json(publication)
                            self._monitor = dict(saved["monitor"])
                            self._last_monitor_count = int(saved["record_count"])
                        elif self._active_snapshot_id == "snapshot_000":
                            self._monitor = {}
            except (OSError, ValueError, json.JSONDecodeError):
                # A malformed active pointer is never silently selected; the
                # worker will fail closed when it tries to resolve it.
                pass
        # Workers inherit the pointer but never receive CFV_REGISTRY_SNAPSHOT_PATH
        # from the scheduler. They resolve the pointer before each sample.
        os.environ["CFV_ONLINE_ACTIVE_SNAPSHOT_PATH"] = str(self.active_snapshot_path)
        self._write_state(status="disabled" if not self.enabled else "ready")

    @property
    def active_snapshot_id(self) -> str:
        with self._lock:
            return self._active_snapshot_id

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    def _write_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event_id": f"online_{int(time.time_ns())}",
            "event_type": str(event_type),
            "timestamp": _utc_now(),
            "active_snapshot_id": self._active_snapshot_id,
            "payload": dict(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_state(self, *, status: str, last_result: Optional[Mapping[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "kind": "online_evolution_state",
            "layout_version": 1,
            "updated_at": _utc_now(),
            "enabled": bool(self.enabled),
            "status": status,
            "active_snapshot_id": self._active_snapshot_id,
            "active_snapshot_path": str(self.active_snapshot_path),
            "record_count": len(self._records),
            "last_trigger_count": self._last_trigger_count,
            "pending_cycle": bool(self._pending_cycle),
            "pending_candidate": self._pending_candidate,
            "monitor": self._monitor,
            "last_monitor_count": self._last_monitor_count,
            "inflight": self._inflight,
            "cycle_count": len(self._cycle_summaries),
            "cycles": list(self._cycle_summaries[-32:]),
        }
        if last_result is not None:
            payload["last_result"] = dict(last_result)
        _atomic_write_json(self.state_path, payload)

    def _result_path(self, payload: Mapping[str, Any]) -> Optional[Path]:
        raw = str(payload.get("run_result_path") or "").strip()
        if raw:
            path = Path(raw).expanduser().resolve()
            if path.is_file():
                return path
        workspace = str(payload.get("workspace_root") or "").strip()
        if workspace:
            candidate = Path(workspace).expanduser().resolve() / "results" / "run_result.json"
            if candidate.is_file():
                return candidate
        return None

    def observe_result(self, payload: Mapping[str, Any]) -> None:
        """Record a terminal result without blocking the live worker."""
        if not self.enabled or self._closed:
            return
        item = dict(payload)
        uid = str(item.get("sample_uid") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if not uid or status not in TERMINAL_STATUSES:
            return
        result_path = self._result_path(item)
        if result_path is None:
            # A scheduler failure without a trajectory cannot diagnose or
            # replay safely. It remains in the normal Mix checkpoint only.
            return
        item["run_result_path"] = str(result_path)
        from .selective_replay import _tool_failed
        trajectory = (_read_json(result_path).get("agent") or {}).get("trajectory") or {}
        item["evolution_diagnostic_signal"] = status in {"failed", "wrong", "unanswerable"} or any(
            _tool_failed(step) for step in trajectory.get("steps") or [] if isinstance(step, Mapping))
        from .analyzer import _tool_records
        item["evolution_diagnostic_signal"] |= len(_tool_records(trajectory)) > self.excessive_tool_calls
        with self._lock:
            self.poll()
            if uid in self._records:
                return
            self._records[uid] = item
            _atomic_write_json(self.records_path, {"records": self._records})
            self._write_event("TrajectoryObservedEvent", {
                "sample_uid": uid,
                "status": status,
                "run_result_path": str(result_path),
            })
            new_records = len(self._records) - self._last_trigger_count
            # Trigger on a diagnostic signal in the newly observed stride. A
            # historical failure must not keep re-triggering proposals whose
            # current window contains only successful trajectories.
            new_items = list(self._records.values())[-new_records:] if new_records > 0 else []
            new_failure = any(
                bool(record.get("evolution_diagnostic_signal")) or str(record.get("status") or "").lower() in {"failed", "wrong", "unanswerable"}
                for record in new_items
            )
            if new_records >= self.trigger_every and (new_failure or self._pending_candidate) or self._monitor_due():
                self._pending_cycle = True
            self._write_state(status="ready" if self._process is None else "running", last_result=item)

    def _cycle_records(self) -> list[Dict[str, Any]]:
        items = list(self._records.values())
        # Unconsumed failures cannot disappear when the recent window moves.
        failures = [item for item in items[self._last_trigger_count:]
                    if item.get("evolution_diagnostic_signal") or str(item.get("status") or "").lower() in {"failed", "wrong", "unanswerable"}]
        retained = set(self._pending_candidate.get("record_uids") or [])
        if self._monitor_due():
            retained.update(self._monitor.get("record_uids") or [])
        records = {str(item["sample_uid"]): item for item in items
                   if str(item["sample_uid"]) in retained}
        records.update({str(item["sample_uid"]): item for item in failures + items[-self.window_size:]})
        return [dict(item) for item in records.values()] if failures or self._pending_candidate or self._monitor_due() else []

    def _monitor_due(self) -> bool:
        interval = max(self.trigger_every, int(self.options.get("online_evolution_regression_every", 128)))
        return bool(self._monitor) and len(self._records) - self._last_monitor_count >= interval

    def _start_cycle(self) -> None:
        if not self.enabled or self._process is not None or not self._pending_cycle:
            return
        records = self._cycle_records()
        if not records:
            self._pending_cycle = False
            self._write_state(status="ready")
            return
        self._cycle_index += 1
        cycle_root = self.cycles_root / f"evolution_online_cycle_{self._cycle_index:05d}"
        cycle_root.mkdir(parents=True, exist_ok=True)
        request_path = cycle_root / "request.json"
        output_path = cycle_root / "cycle_result.json"
        baseline_snapshot_id = self._active_snapshot_id
        if self.active_snapshot_path.is_file():
            try:
                baseline_snapshot_id = str(_read_json(self.active_snapshot_path).get("snapshot_id") or baseline_snapshot_id)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        frozen_parent = cycle_root / "parent_snapshot.json"
        _atomic_write_json(frozen_parent, _read_json(self.active_snapshot_path)
                           if self.active_snapshot_path.is_file()
                           else {"snapshot_id": "snapshot_000", "status": "accepted"})
        request = {
            "kind": "online_evolution_request",
            "layout_version": 1,
            "created_at": _utc_now(),
            "cycle_index": self._cycle_index,
            "workspaces_root": str(self.workspaces_root),
            "config_path": str(self.config_path),
            "evolution_config": str(self.evolution_config),
            "output_root": str(cycle_root),
            "parent_snapshot_id": baseline_snapshot_id,
            "active_snapshot_path": str(frozen_parent) if baseline_snapshot_id != "snapshot_000" else "",
            "baseline_path": str(self.baseline_path) if self.baseline_path else "",
            "visible_device_id": self.visible_device_id,
            "timeout_seconds": self.worker_timeout_seconds,
            "options": dict(self.options),
            "records": records,
            "pending_candidate": self._pending_candidate,
            "monitor_request": self._monitor if self._monitor_due() else {},
        }
        _atomic_write_json(request_path, request)
        log_path = cycle_root / "online_cycle.log"
        environment = os.environ.copy()
        environment.pop("CFV_REGISTRY_SNAPSHOT_PATH", None)
        environment["CFV_ONLINE_EVOLUTION_CYCLE"] = "1"
        rl_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = str(environment.get("PYTHONPATH") or "").strip()
        environment["PYTHONPATH"] = rl_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        command = [
            sys.executable or "python",
            str(Path(__file__).with_name("cycle_entry.py")),
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ]
        # Durable launch intent precedes process creation. A child lease lets
        # recovery reconcile the spawn-to-state-write crash window.
        self._inflight = {"pid": 0, "identity": "", "root": str(cycle_root),
                          "started_at": time.time(), "record_count": len(self._records),
                          "monitor": bool(request["monitor_request"])}
        self._write_state(status="launching")
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            self._inflight = {}
            self._write_state(status="ready")
            raise
        # Keep the handle reachable through the process object until the
        # child exits; Popen does not close a caller-owned file descriptor.
        setattr(process, "_online_log_handle", log_handle)
        self._process = process
        self._process_cycle_root = cycle_root
        self._process_started_at = time.monotonic()
        self._inflight = {"pid": process.pid, "identity": _process_identity(process.pid),
                          "root": str(cycle_root), "started_at": time.time(),
                          "record_count": len(self._records), "monitor": bool(request["monitor_request"])}
        self._pending_cycle = False
        self._write_event("EvolutionCycleStartedEvent", {
            "cycle_index": self._cycle_index,
            "request_path": str(request_path),
            "record_count": len(records),
            "parent_snapshot_id": baseline_snapshot_id,
        })
        self._write_state(status="running")

    def _publish_snapshot(self, snapshot_path: Path, cycle_result: Mapping[str, Any]) -> bool:
        try:
            snapshot = _read_json(snapshot_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if str(snapshot.get("status") or "").lower() != "accepted":
            return False
        snapshot_id = str(snapshot.get("snapshot_id") or "").strip()
        if not snapshot_id:
            return False
        if snapshot_id == self._active_snapshot_id:
            return self.active_snapshot_path.is_file() and _read_json(self.active_snapshot_path) == snapshot
        if str(snapshot.get("parent_snapshot_id") or "snapshot_000") != self._active_snapshot_id:
            return False
        # Keep accepted snapshots immutable in the audit registry and expose a
        # separate atomically replaced pointer to workers.
        registry_path = self.registry_root / f"{snapshot_id}_accepted" / "snapshot.json"
        _atomic_write_json(registry_path, snapshot)
        cycle_root = snapshot_path.parent
        monitor = {"snapshot_path": str(registry_path),
                         "parent_path": str(cycle_root / "parent_snapshot.json"),
                         "snapshot_id": snapshot_id,
                         "record_uids": list(cycle_result.get("record_uids") or []),
                         "parent_monitor": dict(self._monitor)}
        # Publication metadata must survive a crash immediately after the
        # active-pointer replace, before online_state.json is committed.
        _atomic_write_json(registry_path.with_name("publication.json"), {
            "monitor": monitor, "record_count": len(self._records)})
        _atomic_write_json(self.active_snapshot_path, snapshot)
        self._active_snapshot_id = snapshot_id
        self._monitor = monitor
        self._last_monitor_count = len(self._records)
        self._write_event("EvolutionSnapshotPublishedEvent", {
            "snapshot_id": snapshot_id,
            "snapshot_path": str(registry_path),
            "gate": dict(cycle_result.get("gate") or {}),
        })
        return True

    def poll(self) -> Optional[Dict[str, Any]]:
        """Collect a completed child cycle and publish an accepted snapshot."""
        with self._lock:
            process = self._process
            if process is None:
                return None
            if process.poll() is None:
                if self._process_started_at and time.monotonic() - self._process_started_at > self.timeout_seconds:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    # The group leader can exit while a replay grandchild
                    # ignores TERM. Kill the remaining group in either case.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
                else:
                    return None
            log_handle = getattr(process, "_online_log_handle", None)
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:
                    pass
            cycle_root = self._process_cycle_root
            self._process = None
            self._process_cycle_root = None
            output_path = cycle_root / "cycle_result.json" if cycle_root else None
            result: Dict[str, Any]
            if output_path is not None and output_path.is_file():
                try:
                    result = _read_json(output_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    result = {"status": "invalid_cycle_result", "error": f"{type(exc).__name__}: {exc}"}
            else:
                result = {"status": "cycle_process_failed", "return_code": process.returncode}
            result["process_return_code"] = process.returncode
            inflight = self._inflight
            self._inflight = {}
            if not inflight.get("monitor"):
                if process.returncode == 0 and result.get("status") not in {"cycle_error", "replay_error", "evolver_failed"}:
                    self._last_trigger_count = int(inflight.get("record_count", self._last_trigger_count))
                else:
                    self._pending_cycle = True
            if result.get("kind") == "online_evolution_monitor_result":
                if process.returncode == 0 and result.get("status") in {"passed", "regression_detected"}:
                    self._last_monitor_count = len(self._records)
                if process.returncode == 0 and result.get("status") == "regression_detected" and self._monitor.get("snapshot_id") == self._active_snapshot_id:
                    parent = _read_json(Path(self._monitor["parent_path"]))
                    _atomic_write_json(self.active_snapshot_path, parent)
                    self._active_snapshot_id = str(parent["snapshot_id"])
                    self._pending_candidate = {}
                    self._monitor = dict(self._monitor.get("parent_monitor") or {})
                    self._write_event("EvolutionSnapshotRolledBackEvent", result)
                self._write_state(status="ready", last_result=result)
                return result
            transient = process.returncode != 0 or result.get("status") in {
                "cycle_error", "replay_error", "evolver_failed", "invalid_cycle_result", "cycle_process_failed"
            }
            journal = cycle_root / "candidate_pending.json" if cycle_root else None
            if transient and journal is not None and journal.is_file():
                frozen = _read_json(journal)
                self._pending_candidate = {key: frozen.get(key) for key in (
                    "report_path", "trigger_proof_path", "candidate_snapshot_path",
                    "record_uids", "proposal_record_uids", "local_replay_root")}
                self._pending_cycle = True
            if result.get("status") in {"awaiting_future_validation", "insufficient_coverage"}:
                self._pending_candidate = {
                    "report_path": result.get("report_path"),
                    "trigger_proof_path": result.get("trigger_proof_path"),
                    "candidate_snapshot_path": result.get("candidate_snapshot_path"),
                    "record_uids": result.get("record_uids") or [],
                    "proposal_record_uids": result.get("proposal_record_uids") or [],
                    "local_replay_root": result.get("local_replay_root"),
                }
            elif process.returncode == 0 and result.get("status") not in {
                "cycle_error", "replay_error", "evolver_failed", "invalid_cycle_result", "cycle_process_failed"
            }:
                self._pending_candidate = {}
            published = False
            snapshot_path = str(result.get("accepted_snapshot_path") or "").strip()
            if process.returncode == 0 and str(result.get("status") or "") == "accepted" and snapshot_path:
                published = self._publish_snapshot(Path(snapshot_path).resolve(), result)
                if published and str(result.get("report_path") or "").strip():
                    report_path = Path(str(result["report_path"])).expanduser().resolve()
                    try:
                        report_payload = _read_json(report_path)
                        report_payload.update({
                            "online_publish_status": "published",
                            "online_active_snapshot_id": self._active_snapshot_id,
                            "online_active_snapshot_path": str(self.active_snapshot_path),
                        })
                        _atomic_write_json(report_path, report_payload)
                    except (OSError, ValueError, json.JSONDecodeError):
                        # The cycle result remains authoritative if an audit
                        # report was removed between publication and update.
                        pass
            result["published"] = published
            if published:
                result["online_publish_status"] = "published"
            elif str(result.get("status") or "") == "accepted":
                result["online_publish_status"] = "publish_failed"
            else:
                result["online_publish_status"] = "not_published"
            self._cycle_summaries.append({
                "cycle_index": result.get("cycle_index"),
                "status": result.get("status"),
                "gate_status": (result.get("gate") or {}).get("status"),
                "published": published,
                "snapshot_id": result.get("snapshot_id") or "",
                "completed_at": _utc_now(),
            })
            self._write_event("EvolutionCycleFinishedEvent", result)
            self._write_state(status="ready", last_result=result)
            return result

    def on_boundary(self) -> Optional[Dict[str, Any]]:
        """Run a pending cycle between worker groups.

        The worker has released its accelerator admission slot at this point,
        so Evolver generation and candidate replay cannot compete with an
        in-flight model. Waiting at the boundary is what makes publication
        deterministic: the next group always starts with the newest accepted
        snapshot.
        """
        if not self.enabled or self._closed:
            return None
        with self._lock:
            self.poll()
            if self._process is None and self._pending_cycle:
                self._start_cycle()
        while True:
            with self._lock:
                result = self.poll()
                process_alive = self._process is not None
                if not process_alive:
                    return result
            time.sleep(self.poll_seconds)

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        if wait and self.enabled:
            try:
                # Do not strand a final failure window merely because the run
                # ended before the periodic trigger stride. This is a flush
                # at the end of the run; all earlier windows were handled at
                # live worker boundaries.
                with self._lock:
                    new_items = list(self._records.values())[self._last_trigger_count:]
                    if new_items and (self._pending_candidate or any(
                        bool(item.get("evolution_diagnostic_signal")) or str(item.get("status") or "").lower() in {"failed", "wrong", "unanswerable"}
                        for item in new_items
                    )):
                        self._pending_cycle = True
                self.on_boundary()
            except Exception as exc:
                with self._lock:
                    self._write_event("EvolutionControllerErrorEvent", {
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    self._write_state(status="error")
        else:
            with self._lock:
                self.poll()
        self._closed = True
        with self._lock:
            self._write_state(status="closed")

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self.enabled),
                "status": "running" if self._process is not None else "ready",
                "active_snapshot_id": self._active_snapshot_id,
                "active_snapshot_path": str(self.active_snapshot_path),
                "record_count": len(self._records),
                "cycle_count": len(self._cycle_summaries),
                "pending_cycle": bool(self._pending_cycle),
                "pending_candidate": self._pending_candidate,
                "trigger_every": self.trigger_every,
                "window_size": self.window_size,
                "group_size": self.group_size,
                "cycles": list(self._cycle_summaries[-32:]),
            }
