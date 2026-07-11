#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PID_DIR="${TOOL_STAR_SIDECAR_PID_DIR:-$PROJECT_ROOT/rl_workdir/sidecars/pids}"
ENV_FILE="${TOOL_STAR_SIDECAR_ENV_FILE:-$PROJECT_ROOT/rl_workdir/sidecars/env.sh}"
STOP_TIMEOUT_SECONDS="${TOOL_STAR_SIDECAR_STOP_TIMEOUT_SECONDS:-20}"
HOST="${TOOL_STAR_SIDECAR_HOST:-127.0.0.1}"
LLM_PORT="${TOOL_STAR_NON_PLANNER_SIDECAR_PORT:-18000}"
TOOL_SEARCH_EMBEDDING_PORT="${TOOL_STAR_TOOL_SEARCH_EMBEDDING_SIDECAR_PORT:-18001}"
GROUNDING_EMBEDDING_PORT="${TOOL_STAR_GROUNDING_EMBEDDING_SIDECAR_PORT:-18002}"
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

eval "$(
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from config.settings import settings

def positive_int(value, fallback):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else fallback
    except Exception:
        return fallback

def bool_flag(value, fallback):
    if value is None:
        return "1" if fallback else "0"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "1"
    if text in {"0", "false", "no", "n", "off"}:
        return "0"
    return "1" if fallback else "0"

print("CONFIG_SIDECAR_PORT_STRIDE=" + repr(positive_int(getattr(settings, "sidecar_port_stride", 10), 10)))
print("CONFIG_NON_PLANNER_SIDECAR_REPLICA_COUNT=" + repr(positive_int(getattr(settings, "non_planner_sidecar_replica_count", 2), 2)))
print("CONFIG_TOOL_SEARCH_EMBEDDING_SIDECAR_REPLICA_COUNT=" + repr(positive_int(getattr(settings, "tool_search_embedding_sidecar_replica_count", 2), 2)))
print("CONFIG_GROUNDING_EMBEDDING_SIDECAR_REPLICA_COUNT=" + repr(positive_int(getattr(settings, "grounding_embedding_sidecar_replica_count", 2), 2)))
print("CONFIG_SIDECAR_KILL_ORPHAN_VLLM_WORKERS=" + repr(bool_flag(getattr(settings, "sidecar_kill_orphan_vllm_workers", True), True)))
PY
)"

PORT_STRIDE="${TOOL_STAR_SIDECAR_PORT_STRIDE:-$CONFIG_SIDECAR_PORT_STRIDE}"
NON_PLANNER_REPLICA_COUNT="${TOOL_STAR_NON_PLANNER_SIDECAR_REPLICA_COUNT:-$CONFIG_NON_PLANNER_SIDECAR_REPLICA_COUNT}"
TOOL_SEARCH_REPLICA_COUNT="${TOOL_STAR_TOOL_SEARCH_EMBEDDING_SIDECAR_REPLICA_COUNT:-$CONFIG_TOOL_SEARCH_EMBEDDING_SIDECAR_REPLICA_COUNT}"
GROUNDING_REPLICA_COUNT="${TOOL_STAR_GROUNDING_EMBEDDING_SIDECAR_REPLICA_COUNT:-$CONFIG_GROUNDING_EMBEDDING_SIDECAR_REPLICA_COUNT}"
SIDECAR_KILL_ORPHAN_VLLM_WORKERS="${TOOL_STAR_SIDECAR_KILL_ORPHAN_VLLM_WORKERS:-$CONFIG_SIDECAR_KILL_ORPHAN_VLLM_WORKERS}"

rm -f "$ENV_FILE"

list_descendants() {
  parent_pid="$1"
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi
  for child_pid in $(pgrep -P "$parent_pid" 2>/dev/null || true); do
    printf '%s\n' "$child_pid"
    list_descendants "$child_pid"
  done
}

any_pid_alive() {
  for candidate_pid in "$@"; do
    [ -n "$candidate_pid" ] || continue
    if kill -0 "$candidate_pid" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

terminate_process_tree() {
  root_pid="$1"
  [ -n "$root_pid" ] || return 0

  descendants="$(list_descendants "$root_pid" | sort -rn | tr '\n' ' ')"
  targets="$descendants $root_pid"

  # Newer sidecars are launched via setsid, so the root pid is also the process
  # group id. Kill the group first, then fall back to explicit descendants.
  kill -- "-$root_pid" 2>/dev/null || true
  for target_pid in $targets; do
    kill "$target_pid" 2>/dev/null || true
  done

  start_ts="$(date +%s)"
  deadline_ts="$((start_ts + STOP_TIMEOUT_SECONDS))"
  while any_pid_alive $targets; do
    now_ts="$(date +%s)"
    [ "$now_ts" -lt "$deadline_ts" ] || break
    sleep 1
  done

  if any_pid_alive $targets; then
    echo "Force killing lingering sidecar process tree rooted at pid=$root_pid" >&2
    kill -KILL -- "-$root_pid" 2>/dev/null || true
    for target_pid in $targets; do
      kill -KILL "$target_pid" 2>/dev/null || true
    done
  fi
}

managed_sidecar_ports() {
  python - "$LLM_PORT" "$TOOL_SEARCH_EMBEDDING_PORT" "$GROUNDING_EMBEDDING_PORT" "$PORT_STRIDE" "$NON_PLANNER_REPLICA_COUNT" "$TOOL_SEARCH_REPLICA_COUNT" "$GROUNDING_REPLICA_COUNT" <<'PY'
import sys

llm_port, tool_port, grounding_port, stride, llm_count, tool_count, grounding_count = [int(value) for value in sys.argv[1:]]
ports = []
for base, count in (
    (llm_port, llm_count),
    (tool_port, tool_count),
    (grounding_port, grounding_count),
):
    for index in range(max(1, int(count))):
        ports.append(base + index * stride)
print(" ".join(str(port) for port in sorted(set(ports))))
PY
}

pids_for_tcp_port() {
  port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
    return 0
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' || true
    return 0
  fi
  python - "$port" <<'PY'
import os
import sys

port = int(sys.argv[1])
target_inodes = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(table, "r", encoding="utf-8", errors="ignore").read().splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        local_address, state, inode = parts[1], parts[3], parts[9]
        if state != "0A":
            continue
        try:
            local_port = int(local_address.rsplit(":", 1)[1], 16)
        except Exception:
            continue
        if local_port == port:
            target_inodes.add(inode)

if not target_inodes:
    raise SystemExit(0)

for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    fd_dir = f"/proc/{pid}/fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        continue
    for fd in fds:
        try:
            target = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in target_inodes:
            print(pid)
            break
PY
}

cleanup_managed_ports() {
  for port in $(managed_sidecar_ports); do
    pids="$(pids_for_tcp_port "$port" | sort -u | tr '\n' ' ')"
    [ -n "$pids" ] || continue
    echo "Stopping sidecar process(es) listening on ${HOST}:${port}: $pids"
    for port_pid in $pids; do
      terminate_process_tree "$port_pid"
    done
  done
}

cleanup_orphan_vllm_workers() {
  [ "$SIDECAR_KILL_ORPHAN_VLLM_WORKERS" = "1" ] || return 0
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi
  pids="$(pgrep -u "$(id -u)" -f 'VLLM::EngineCore|VLLM::Worker|EngineCore|Worker_TP[0-9]|VllmWorker' 2>/dev/null | sort -u | tr '\n' ' ' || true)"
  [ -n "$pids" ] || return 0
  echo "Stopping orphan VLLM worker process(es): $pids"
  for vllm_pid in $pids; do
    terminate_process_tree "$vllm_pid"
  done
}

for pid_file in "$PID_DIR"/*.pid; do
  [ -e "$pid_file" ] || continue
  name="$(basename "$pid_file" .pid)"
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $name pid=$pid"
    terminate_process_tree "$pid"
  else
    echo "$name is not running"
  fi
  rm -f "$pid_file"
done

cleanup_managed_ports
cleanup_orphan_vllm_workers
