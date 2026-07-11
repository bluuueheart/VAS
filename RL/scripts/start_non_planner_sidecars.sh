#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${TOOL_STAR_SIDECAR_LOG_DIR:-$PROJECT_ROOT/rl_workdir/sidecars/logs}"
PID_DIR="${TOOL_STAR_SIDECAR_PID_DIR:-$PROJECT_ROOT/rl_workdir/sidecars/pids}"
ENV_FILE="${TOOL_STAR_SIDECAR_ENV_FILE:-$PROJECT_ROOT/rl_workdir/sidecars/env.sh}"
HOST="${TOOL_STAR_SIDECAR_HOST:-127.0.0.1}"
API_KEY="${TOOL_STAR_SIDECAR_API_KEY:-EMPTY}"
LLM_PORT="${TOOL_STAR_NON_PLANNER_SIDECAR_PORT:-18000}"
TOOL_SEARCH_EMBEDDING_PORT="${TOOL_STAR_TOOL_SEARCH_EMBEDDING_SIDECAR_PORT:-18001}"
GROUNDING_EMBEDDING_PORT="${TOOL_STAR_GROUNDING_EMBEDDING_SIDECAR_PORT:-18002}"
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ "${TOOL_STAR_ENABLE_NON_PLANNER_SIDECARS:-0}" != "1" ]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  cat > "$ENV_FILE" <<'EOF'
#!/usr/bin/env sh
export TOOL_STAR_NON_PLANNER_MODEL_PROVIDER=process
export TOOL_STAR_AUX_MODEL_PROVIDER=process
export TOOL_STAR_SUBAGENT_SERVICE_MODEL_PROVIDER=process
export TOOL_STAR_NON_PLANNER_EMBEDDING_PROVIDER=local
export TOOL_STAR_TOOL_SEARCH_EMBEDDING_PROVIDER=local
export TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_PROVIDER=local
export TOOL_STAR_PREPROCESS_EMBEDDING_PROVIDER=local
export TOOL_STAR_SKIP_SIDECAR_PREFLIGHT=1
export TOOL_STAR_AUTO_START_SIDECARS=0
EOF
  echo "Non-planner sidecars disabled by default; wrote process/local runtime env to: $ENV_FILE"
  echo "Set TOOL_STAR_ENABLE_NON_PLANNER_SIDECARS=1 to start vLLM sidecars explicitly."
  exit 0
fi
ENABLE_SUBAGENT_SERVICE_SIDECAR="${TOOL_STAR_ENABLE_SUBAGENT_SERVICE_SIDECAR:-1}"
eval "$(
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from config.settings import settings
def positive_int(value, fallback):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else fallback
    except Exception:
        return fallback
def positive_float(value, fallback):
    try:
        parsed = float(value)
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
print("CONFIG_SIDECAR_LOAD_BALANCE_POLICY=" + repr(str(getattr(settings, "sidecar_load_balance_policy", "least_inflight") or "least_inflight")))
print("CONFIG_SIDECAR_STARTUP_TIMEOUT_SECONDS=" + repr(positive_int(getattr(settings, "sidecar_startup_timeout_seconds", 900), 900)))
print("CONFIG_SIDECAR_HEALTH_CHECK_INTERVAL_SECONDS=" + repr(positive_int(getattr(settings, "sidecar_health_check_interval_seconds", 5), 5)))
print("CONFIG_SIDECAR_GPU_QUIESCE_SECONDS=" + repr(positive_int(getattr(settings, "sidecar_gpu_quiesce_seconds", 8), 8)))
print("CONFIG_SIDECAR_SMOKE_CHECK_ENABLED=" + repr(bool_flag(getattr(settings, "sidecar_smoke_check_enabled", True), True)))
print("CONFIG_SIDECAR_SMOKE_TIMEOUT_SECONDS=" + repr(positive_int(getattr(settings, "sidecar_smoke_timeout_seconds", 120), 120)))
print("CONFIG_VISUAL_SERVICE_MAX_FRAMES=" + repr(positive_int(getattr(settings, "visual_service_max_frames", 1), 1)))
print("CONFIG_VISUAL_SERVICE_MAX_PIXELS=" + repr(positive_int(getattr(settings, "visual_service_max_pixels", 336 * 336), 336 * 336)))
print("CONFIG_SIDECAR_VLLM_USE_V1=" + repr(str(getattr(settings, "sidecar_vllm_use_v1", "") or "")))
print("CONFIG_SIDECAR_LLM_DISABLE_CHUNKED_PREFILL=" + repr(bool_flag(getattr(settings, "sidecar_llm_disable_chunked_prefill", False), False)))
print("CONFIG_SIDECAR_LLM_DISABLE_CHUNKED_MM_INPUT=" + repr(bool_flag(getattr(settings, "sidecar_llm_disable_chunked_mm_input", True), True)))
print("CONFIG_SIDECAR_LLM_DISABLE_ASYNC_SCHEDULING=" + repr(bool_flag(getattr(settings, "sidecar_llm_disable_async_scheduling", True), True)))
print("CONFIG_SIDECAR_REFUSE_WHEN_RAY_WORKER_ACTIVE=" + repr(bool_flag(getattr(settings, "sidecar_refuse_when_ray_worker_active", True), True)))
print("CONFIG_SIDECAR_CLEANUP_STALE_PROCESSES=" + repr(bool_flag(getattr(settings, "sidecar_cleanup_stale_processes", True), True)))
print("CONFIG_SIDECAR_SPARE_LLM_GPU_MEMORY_UTILIZATION=" + repr(positive_float(getattr(settings, "sidecar_spare_llm_gpu_memory_utilization", 0.30), 0.30)))
print("CONFIG_SIDECAR_SPARE_EMBEDDING_GPU_MEMORY_UTILIZATION=" + repr(positive_float(getattr(settings, "sidecar_spare_embedding_gpu_memory_utilization", 0.16), 0.16)))
print("CONFIG_SIDECAR_SHARED_LLM_GPU_MEMORY_UTILIZATION=" + repr(positive_float(getattr(settings, "sidecar_shared_llm_gpu_memory_utilization", 0.30), 0.30)))
print("CONFIG_SIDECAR_SHARED_EMBEDDING_GPU_MEMORY_UTILIZATION=" + repr(positive_float(getattr(settings, "sidecar_shared_embedding_gpu_memory_utilization", 0.10), 0.10)))
print("CONFIG_SIDECAR_EMBEDDING_MAX_MODEL_LEN=" + repr(positive_int(getattr(settings, "sidecar_embedding_max_model_len", 8192), 8192)))
print("CONFIG_SIDECAR_SPARE_LLM_MAX_MODEL_LEN=" + repr(positive_int(getattr(settings, "sidecar_spare_llm_max_model_len", 12288), 12288)))
print("CONFIG_SIDECAR_SHARED_LLM_MAX_MODEL_LEN=" + repr(positive_int(getattr(settings, "sidecar_shared_llm_max_model_len", 8192), 8192)))
print("CONFIG_SIDECAR_SPARE_LLM_MAX_BATCHED_TOKENS=" + repr(positive_int(getattr(settings, "sidecar_spare_llm_max_batched_tokens", 8192), 8192)))
print("CONFIG_SIDECAR_SHARED_LLM_MAX_BATCHED_TOKENS=" + repr(positive_int(getattr(settings, "sidecar_shared_llm_max_batched_tokens", 8192), 8192)))
print("CONFIG_SIDECAR_SPARE_LLM_KV_CACHE_MEMORY_BYTES=" + repr(str(getattr(settings, "sidecar_spare_llm_kv_cache_memory_bytes", "") or "")))
print("CONFIG_SIDECAR_SHARED_LLM_KV_CACHE_MEMORY_BYTES=" + repr(str(getattr(settings, "sidecar_shared_llm_kv_cache_memory_bytes", "2G") or "2G")))
print("CONFIG_SIDECAR_LLM_MAX_TOKENS_PER_MM_ITEM=" + repr(positive_int(getattr(settings, "sidecar_llm_max_tokens_per_mm_item", 16384), 16384)))
print("CONFIG_SIDECAR_SPARE_EMBEDDING_MAX_BATCHED_TOKENS=" + repr(positive_int(getattr(settings, "sidecar_spare_embedding_max_batched_tokens", 1024), 1024)))
print("CONFIG_SIDECAR_SHARED_EMBEDDING_MAX_BATCHED_TOKENS=" + repr(positive_int(getattr(settings, "sidecar_shared_embedding_max_batched_tokens", 512), 512)))
print("CONFIG_QWEN_VL_IMAGE_MAX_PIXELS=" + repr(positive_int(getattr(settings, "qwen_vl_image_max_pixels", 1280 * 28 * 28), 1280 * 28 * 28)))
print("CONFIG_SIDECAR_SPARE_LLM_CPU_OFFLOAD_GB=" + repr(positive_int(getattr(settings, "sidecar_spare_llm_cpu_offload_gb", 0), 0)))
print("CONFIG_SIDECAR_SHARED_LLM_CPU_OFFLOAD_GB=" + repr(positive_int(getattr(settings, "sidecar_shared_llm_cpu_offload_gb", 0), 0)))
print("CONFIG_SIDECAR_SPARE_LLM_TENSOR_PARALLEL_SIZE=" + repr(positive_int(getattr(settings, "sidecar_spare_llm_tensor_parallel_size", 1), 1)))
print("CONFIG_SIDECAR_SHARED_LLM_TENSOR_PARALLEL_SIZE=" + repr(positive_int(getattr(settings, "sidecar_shared_llm_tensor_parallel_size", 4), 4)))
print("CONFIG_SIDECAR_SPARE_LLM_REPLICA_COUNT=" + repr(positive_int(getattr(settings, "sidecar_spare_llm_replica_count", 2), 2)))
print("CONFIG_SIDECAR_SHARED_LLM_REPLICA_COUNT=" + repr(positive_int(getattr(settings, "sidecar_shared_llm_replica_count", 1), 1)))
print("CONFIG_NON_PLANNER_SIDECAR_REPLICA_COUNT=" + repr(positive_int(getattr(settings, "non_planner_sidecar_replica_count", 2), 2)))
print("CONFIG_TOOL_SEARCH_EMBEDDING_SIDECAR_REPLICA_COUNT=" + repr(positive_int(getattr(settings, "tool_search_embedding_sidecar_replica_count", 1), 1)))
print("CONFIG_GROUNDING_EMBEDDING_SIDECAR_REPLICA_COUNT=" + repr(positive_int(getattr(settings, "grounding_embedding_sidecar_replica_count", 1), 1)))
PY
)"
PORT_STRIDE="${TOOL_STAR_SIDECAR_PORT_STRIDE:-$CONFIG_SIDECAR_PORT_STRIDE}"
LOAD_BALANCE_POLICY="${TOOL_STAR_SIDECAR_LOAD_BALANCE_POLICY:-$CONFIG_SIDECAR_LOAD_BALANCE_POLICY}"
SIDECAR_STARTUP_TIMEOUT_SECONDS="${TOOL_STAR_SIDECAR_STARTUP_TIMEOUT_SECONDS:-$CONFIG_SIDECAR_STARTUP_TIMEOUT_SECONDS}"
SIDECAR_HEALTH_CHECK_INTERVAL_SECONDS="${TOOL_STAR_SIDECAR_HEALTH_CHECK_INTERVAL_SECONDS:-$CONFIG_SIDECAR_HEALTH_CHECK_INTERVAL_SECONDS}"
SIDECAR_GPU_QUIESCE_SECONDS="${TOOL_STAR_SIDECAR_GPU_QUIESCE_SECONDS:-$CONFIG_SIDECAR_GPU_QUIESCE_SECONDS}"
SIDECAR_SMOKE_CHECK_ENABLED="${TOOL_STAR_SIDECAR_SMOKE_CHECK_ENABLED:-$CONFIG_SIDECAR_SMOKE_CHECK_ENABLED}"
SIDECAR_SMOKE_TIMEOUT_SECONDS="${TOOL_STAR_SIDECAR_SMOKE_TIMEOUT_SECONDS:-$CONFIG_SIDECAR_SMOKE_TIMEOUT_SECONDS}"
SIDECAR_VLLM_USE_V1="${TOOL_STAR_SIDECAR_VLLM_USE_V1:-$CONFIG_SIDECAR_VLLM_USE_V1}"
SIDECAR_LLM_DISABLE_CHUNKED_PREFILL="${TOOL_STAR_SIDECAR_LLM_DISABLE_CHUNKED_PREFILL:-$CONFIG_SIDECAR_LLM_DISABLE_CHUNKED_PREFILL}"
SIDECAR_LLM_DISABLE_CHUNKED_MM_INPUT="${TOOL_STAR_SIDECAR_LLM_DISABLE_CHUNKED_MM_INPUT:-$CONFIG_SIDECAR_LLM_DISABLE_CHUNKED_MM_INPUT}"
SIDECAR_LLM_DISABLE_ASYNC_SCHEDULING="${TOOL_STAR_SIDECAR_LLM_DISABLE_ASYNC_SCHEDULING:-$CONFIG_SIDECAR_LLM_DISABLE_ASYNC_SCHEDULING}"
SIDECAR_REFUSE_WHEN_RAY_WORKER_ACTIVE="${TOOL_STAR_SIDECAR_REFUSE_WHEN_RAY_WORKER_ACTIVE:-$CONFIG_SIDECAR_REFUSE_WHEN_RAY_WORKER_ACTIVE}"
SIDECAR_CLEANUP_STALE_PROCESSES="${TOOL_STAR_SIDECAR_CLEANUP_STALE_PROCESSES:-$CONFIG_SIDECAR_CLEANUP_STALE_PROCESSES}"
VISUAL_SERVICE_MAX_FRAMES="${TOOL_STAR_VISUAL_SERVICE_MAX_FRAMES:-$CONFIG_VISUAL_SERVICE_MAX_FRAMES}"
VISUAL_SERVICE_MAX_PIXELS="${TOOL_STAR_VISUAL_SERVICE_MAX_PIXELS:-$CONFIG_VISUAL_SERVICE_MAX_PIXELS}"
QWEN_VL_IMAGE_MAX_PIXELS="${TOOL_STAR_QWEN_VL_IMAGE_MAX_PIXELS:-$CONFIG_QWEN_VL_IMAGE_MAX_PIXELS}"
SIDECAR_LLM_MAX_TOKENS_PER_MM_ITEM="${TOOL_STAR_SIDECAR_LLM_MAX_TOKENS_PER_MM_ITEM:-$CONFIG_SIDECAR_LLM_MAX_TOKENS_PER_MM_ITEM}"
detect_gpu_list() {
  if [ -n "${TOOL_STAR_AVAILABLE_GPU_IDS:-}" ]; then
    printf '%s\n' "$TOOL_STAR_AVAILABLE_GPU_IDS"
    return 0
  fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -
    return 0
  fi
  printf '0\n'
}
normalize_positive_int() {
  value="$1"
  fallback="$2"
  python - "$value" "$fallback" <<'PY'
import sys
value = sys.argv[1]
fallback = int(sys.argv[2]) if len(sys.argv) > 2 else 1
try:
  parsed = int(value)
  print(parsed if parsed > 0 else fallback)
except Exception:
  print(fallback)
PY
}
normalize_min_int() {
  value="$1"
  floor="$2"
  python - "$value" "$floor" <<'PY'
import sys
value = sys.argv[1]
floor = int(sys.argv[2]) if len(sys.argv) > 2 else 1
try:
  parsed = int(value)
  print(parsed if parsed >= floor else floor)
except Exception:
  print(floor)
PY
}
pick_gpu_sequence() {
  list="$1"
  policy="$2"
  offset="$3"
  count="$4"
  python - "$list" "$policy" "$offset" "$count" <<'PY'
import subprocess
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
if not items:
    items = ["0"]
policy = (sys.argv[2] or "spread_last").strip().lower()
try:
    offset = int(sys.argv[3] or "0")
except Exception:
    offset = 0
try:
  count = int(sys.argv[4] or "1")
except Exception:
  count = 1
count = max(1, count)
def rotate(values, n):
    if not values:
        return values
    n = n % len(values)
    return values[n:] + values[:n]
ranked = list(items)
if policy == "spread_first":
  ranked = rotate(list(items), offset)
elif policy in {"least_used", "least-free", "least_free"}:
    used = {}
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                used[parts[0]] = int(float(parts[1]))
    except Exception:
        used = {}
    ranked = sorted(items, key=lambda item: (used.get(item, 0), items.index(item)))
    ranked = rotate(ranked, offset)
elif policy == "round_robin":
    ranked = rotate(list(items), offset)
else:
    ranked = rotate(list(reversed(items)), offset)
sequence = [ranked[index % len(ranked)] for index in range(count)]
print(",".join(sequence))
PY
}
pick_gpu_groups() {
  list="$1"
  policy="$2"
  offset="$3"
  group_size="$4"
  count="$5"
  python - "$list" "$policy" "$offset" "$group_size" "$count" <<'PY'
import subprocess
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
if not items:
    items = ["0"]
policy = (sys.argv[2] or "spread_last").strip().lower()
try:
    offset = int(sys.argv[3] or "0")
except Exception:
    offset = 0
try:
    group_size = int(sys.argv[4] or "1")
except Exception:
    group_size = 1
try:
    count = int(sys.argv[5] or "1")
except Exception:
    count = 1
group_size = max(1, min(group_size, len(items)))
count = max(1, count)
def rotate(values, n):
    if not values:
        return values
    n = n % len(values)
    return values[n:] + values[:n]
ranked = list(items)
if policy == "spread_first":
    ranked = rotate(list(items), offset)
elif policy in {"least_used", "least-free", "least_free"}:
    used = {}
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                used[parts[0]] = int(float(parts[1]))
    except Exception:
        used = {}
    ranked = sorted(items, key=lambda item: (used.get(item, 0), items.index(item)))
    ranked = rotate(ranked, offset)
elif policy == "round_robin":
    ranked = rotate(list(items), offset)
else:
    ranked = rotate(list(reversed(items)), offset)
groups = []
cursor = 0
for _ in range(count):
    group = [ranked[(cursor + inner) % len(ranked)] for inner in range(group_size)]
    groups.append(",".join(group))
    cursor += group_size
print(";".join(groups) + ";")
PY
}
normalize_cuda_groups() {
  sequence="$1"
  group_size="$2"
  python - "$sequence" "$group_size" <<'PY'
import re
import sys
sequence = sys.argv[1]
try:
    group_size = int(sys.argv[2] or "1")
except Exception:
    group_size = 1
group_size = max(1, group_size)
raw_groups = [part.strip() for part in re.split(r"[;\n]+", sequence or "") if part.strip()]
if raw_groups:
    groups = raw_groups
else:
    devices = [part.strip() for part in str(sequence or "").split(",") if part.strip()]
    if not devices:
        devices = ["0"]
    if group_size <= 1:
        groups = devices
    else:
        groups = [",".join(devices[index:index + group_size]) for index in range(0, len(devices), group_size)]
print(";".join(groups) + ";")
PY
}
GPU_LIST="$(detect_gpu_list)"
GPU_COUNT="$(python - "$GPU_LIST" <<'PY'
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(items) or 1)
PY
)"
if [ "$GPU_COUNT" -ge 7 ]; then
  SIDECAR_COLOCATION_MODE="spare-gpu"
  SIDECAR_GPU_POLICY="${TOOL_STAR_SIDECAR_GPU_POLICY:-spread_last}"
  SIDECAR_GPU_OFFSET="${TOOL_STAR_SIDECAR_GPU_OFFSET:-0}"
  DEFAULT_LLM_REPLICA_COUNT="$(normalize_positive_int "${TOOL_STAR_NON_PLANNER_SIDECAR_REPLICA_COUNT:-$CONFIG_SIDECAR_SPARE_LLM_REPLICA_COUNT}" 1)"
  DEFAULT_TOOL_SEARCH_REPLICA_COUNT="$(normalize_positive_int "${TOOL_STAR_TOOL_SEARCH_EMBEDDING_SIDECAR_REPLICA_COUNT:-$CONFIG_TOOL_SEARCH_EMBEDDING_SIDECAR_REPLICA_COUNT}" 1)"
  DEFAULT_GROUNDING_REPLICA_COUNT="$(normalize_positive_int "${TOOL_STAR_GROUNDING_EMBEDDING_SIDECAR_REPLICA_COUNT:-$CONFIG_GROUNDING_EMBEDDING_SIDECAR_REPLICA_COUNT}" 1)"
  DEFAULT_LLM_TENSOR_PARALLEL_SIZE="$(normalize_positive_int "${TOOL_STAR_NON_PLANNER_SIDECAR_TENSOR_PARALLEL_SIZE:-$CONFIG_SIDECAR_SPARE_LLM_TENSOR_PARALLEL_SIZE}" 1)"
  DEFAULT_LLM_GPU_MEMORY_UTILIZATION="${TOOL_STAR_NON_PLANNER_SIDECAR_GPU_MEMORY_UTILIZATION:-$CONFIG_SIDECAR_SPARE_LLM_GPU_MEMORY_UTILIZATION}"
  DEFAULT_EMBEDDING_GPU_MEMORY_UTILIZATION="${TOOL_STAR_EMBEDDING_SIDECAR_GPU_MEMORY_UTILIZATION:-$CONFIG_SIDECAR_SPARE_EMBEDDING_GPU_MEMORY_UTILIZATION}"
  DEFAULT_EMBEDDING_MAX_MODEL_LEN="${TOOL_STAR_EMBEDDING_SIDECAR_MAX_MODEL_LEN:-$CONFIG_SIDECAR_EMBEDDING_MAX_MODEL_LEN}"
  DEFAULT_LLM_MAX_MODEL_LEN="${TOOL_STAR_NON_PLANNER_SIDECAR_MAX_MODEL_LEN:-$CONFIG_SIDECAR_SPARE_LLM_MAX_MODEL_LEN}"
  DEFAULT_LLM_MAX_BATCHED_TOKENS="$(normalize_positive_int "${TOOL_STAR_NON_PLANNER_SIDECAR_MAX_BATCHED_TOKENS:-$CONFIG_SIDECAR_SPARE_LLM_MAX_BATCHED_TOKENS}" "$CONFIG_SIDECAR_SPARE_LLM_MAX_BATCHED_TOKENS")"
  DEFAULT_LLM_KV_CACHE_MEMORY_BYTES="${TOOL_STAR_NON_PLANNER_SIDECAR_KV_CACHE_MEMORY_BYTES:-$CONFIG_SIDECAR_SPARE_LLM_KV_CACHE_MEMORY_BYTES}"
  DEFAULT_EMBEDDING_MAX_BATCHED_TOKENS="${TOOL_STAR_EMBEDDING_SIDECAR_MAX_BATCHED_TOKENS:-$CONFIG_SIDECAR_SPARE_EMBEDDING_MAX_BATCHED_TOKENS}"
  DEFAULT_CPU_OFFLOAD_GB="${TOOL_STAR_NON_PLANNER_SIDECAR_CPU_OFFLOAD_GB:-$CONFIG_SIDECAR_SPARE_LLM_CPU_OFFLOAD_GB}"
else
  SIDECAR_COLOCATION_MODE="shared-4gpu"
  SIDECAR_GPU_POLICY="${TOOL_STAR_SIDECAR_GPU_POLICY:-spread_last}"
  SIDECAR_GPU_OFFSET="${TOOL_STAR_SIDECAR_GPU_OFFSET:-0}"
  DEFAULT_LLM_REPLICA_COUNT="$(normalize_positive_int "${TOOL_STAR_NON_PLANNER_SIDECAR_REPLICA_COUNT:-$CONFIG_SIDECAR_SHARED_LLM_REPLICA_COUNT}" 1)"
  DEFAULT_TOOL_SEARCH_REPLICA_COUNT="$(normalize_positive_int "${TOOL_STAR_TOOL_SEARCH_EMBEDDING_SIDECAR_REPLICA_COUNT:-$CONFIG_TOOL_SEARCH_EMBEDDING_SIDECAR_REPLICA_COUNT}" 1)"
  DEFAULT_GROUNDING_REPLICA_COUNT="$(normalize_positive_int "${TOOL_STAR_GROUNDING_EMBEDDING_SIDECAR_REPLICA_COUNT:-$CONFIG_GROUNDING_EMBEDDING_SIDECAR_REPLICA_COUNT}" 1)"
  DEFAULT_LLM_TENSOR_PARALLEL_SIZE="$(normalize_positive_int "${TOOL_STAR_NON_PLANNER_SIDECAR_TENSOR_PARALLEL_SIZE:-$CONFIG_SIDECAR_SHARED_LLM_TENSOR_PARALLEL_SIZE}" 1)"
  DEFAULT_LLM_GPU_MEMORY_UTILIZATION="${TOOL_STAR_NON_PLANNER_SIDECAR_GPU_MEMORY_UTILIZATION:-$CONFIG_SIDECAR_SHARED_LLM_GPU_MEMORY_UTILIZATION}"
  DEFAULT_EMBEDDING_GPU_MEMORY_UTILIZATION="${TOOL_STAR_EMBEDDING_SIDECAR_GPU_MEMORY_UTILIZATION:-$CONFIG_SIDECAR_SHARED_EMBEDDING_GPU_MEMORY_UTILIZATION}"
  DEFAULT_EMBEDDING_MAX_MODEL_LEN="${TOOL_STAR_EMBEDDING_SIDECAR_MAX_MODEL_LEN:-$CONFIG_SIDECAR_EMBEDDING_MAX_MODEL_LEN}"
  DEFAULT_LLM_MAX_MODEL_LEN="${TOOL_STAR_NON_PLANNER_SIDECAR_MAX_MODEL_LEN:-$CONFIG_SIDECAR_SHARED_LLM_MAX_MODEL_LEN}"
  DEFAULT_LLM_MAX_BATCHED_TOKENS="$(normalize_positive_int "${TOOL_STAR_NON_PLANNER_SIDECAR_MAX_BATCHED_TOKENS:-$CONFIG_SIDECAR_SHARED_LLM_MAX_BATCHED_TOKENS}" "$CONFIG_SIDECAR_SHARED_LLM_MAX_BATCHED_TOKENS")"
  DEFAULT_LLM_KV_CACHE_MEMORY_BYTES="${TOOL_STAR_NON_PLANNER_SIDECAR_KV_CACHE_MEMORY_BYTES:-$CONFIG_SIDECAR_SHARED_LLM_KV_CACHE_MEMORY_BYTES}"
  DEFAULT_EMBEDDING_MAX_BATCHED_TOKENS="${TOOL_STAR_EMBEDDING_SIDECAR_MAX_BATCHED_TOKENS:-$CONFIG_SIDECAR_SHARED_EMBEDDING_MAX_BATCHED_TOKENS}"
  DEFAULT_CPU_OFFLOAD_GB="${TOOL_STAR_NON_PLANNER_SIDECAR_CPU_OFFLOAD_GB:-$CONFIG_SIDECAR_SHARED_LLM_CPU_OFFLOAD_GB}"
fi
if [ "$SIDECAR_COLOCATION_MODE" = "shared-4gpu" ] \
  && [ "${TOOL_STAR_FORCE_SHARED_GPU_LLM_SIDECAR:-0}" != "1" ] \
  && [ "$ENABLE_SUBAGENT_SERVICE_SIDECAR" != "1" ]; then
  DEFAULT_LLM_REPLICA_COUNT=0
fi
if [ "$DEFAULT_LLM_TENSOR_PARALLEL_SIZE" -gt "$GPU_COUNT" ]; then
  DEFAULT_LLM_TENSOR_PARALLEL_SIZE="$GPU_COUNT"
fi
if [ "$SIDECAR_LLM_DISABLE_CHUNKED_PREFILL" != "1" ] && [ "$SIDECAR_LLM_DISABLE_CHUNKED_MM_INPUT" = "1" ]; then
  DEFAULT_LLM_MAX_BATCHED_TOKENS="$(normalize_min_int "$DEFAULT_LLM_MAX_BATCHED_TOKENS" "$SIDECAR_LLM_MAX_TOKENS_PER_MM_ITEM")"
fi
LLM_GPU_SLOT_COUNT="$((DEFAULT_LLM_REPLICA_COUNT * DEFAULT_LLM_TENSOR_PARALLEL_SIZE))"
NON_PLANNER_GPU_SEQUENCE="$(pick_gpu_groups "$GPU_LIST" "$SIDECAR_GPU_POLICY" "$SIDECAR_GPU_OFFSET" "$DEFAULT_LLM_TENSOR_PARALLEL_SIZE" "$DEFAULT_LLM_REPLICA_COUNT")"
TOOL_SEARCH_GPU_SEQUENCE="$(pick_gpu_sequence "$GPU_LIST" "$SIDECAR_GPU_POLICY" "$((SIDECAR_GPU_OFFSET + LLM_GPU_SLOT_COUNT))" "$DEFAULT_TOOL_SEARCH_REPLICA_COUNT")"
GROUNDING_GPU_SEQUENCE="$(pick_gpu_sequence "$GPU_LIST" "$SIDECAR_GPU_POLICY" "$((SIDECAR_GPU_OFFSET + LLM_GPU_SLOT_COUNT + DEFAULT_TOOL_SEARCH_REPLICA_COUNT))" "$DEFAULT_GROUNDING_REPLICA_COUNT")"
LLM_CUDA_VISIBLE_DEVICES="$(if [ "$DEFAULT_LLM_REPLICA_COUNT" -eq 0 ]; then printf ''; else normalize_cuda_groups "${TOOL_STAR_NON_PLANNER_SIDECAR_CUDA_VISIBLE_DEVICES:-$NON_PLANNER_GPU_SEQUENCE}" "$DEFAULT_LLM_TENSOR_PARALLEL_SIZE"; fi)"
TOOL_SEARCH_EMBEDDING_CUDA_VISIBLE_DEVICES="${TOOL_STAR_TOOL_SEARCH_EMBEDDING_CUDA_VISIBLE_DEVICES:-$TOOL_SEARCH_GPU_SEQUENCE}"
GROUNDING_EMBEDDING_CUDA_VISIBLE_DEVICES="${TOOL_STAR_GROUNDING_EMBEDDING_CUDA_VISIBLE_DEVICES:-$GROUNDING_GPU_SEQUENCE}"
DEFAULT_LLM_ARGS="--trust-remote-code --enforce-eager --max-model-len $DEFAULT_LLM_MAX_MODEL_LEN --max-num-seqs 1 --max-num-batched-tokens $DEFAULT_LLM_MAX_BATCHED_TOKENS --limit-mm-per-prompt '{\"image\": 1, \"video\": 0, \"audio\": 0, \"vision_chunk\": 0}' --gpu-memory-utilization $DEFAULT_LLM_GPU_MEMORY_UTILIZATION --swap-space ${TOOL_STAR_NON_PLANNER_SIDECAR_SWAP_SPACE_GB:-16}"
if [ "$SIDECAR_LLM_DISABLE_CHUNKED_PREFILL" = "1" ]; then
  DEFAULT_LLM_ARGS="$DEFAULT_LLM_ARGS --no-enable-chunked-prefill"
elif [ "$SIDECAR_LLM_DISABLE_CHUNKED_MM_INPUT" = "1" ]; then
  DEFAULT_LLM_ARGS="$DEFAULT_LLM_ARGS --disable-chunked-mm-input"
fi
if [ "$SIDECAR_LLM_DISABLE_ASYNC_SCHEDULING" = "1" ]; then
  DEFAULT_LLM_ARGS="$DEFAULT_LLM_ARGS --no-async-scheduling"
fi
if [ "$DEFAULT_CPU_OFFLOAD_GB" -gt 0 ]; then
  DEFAULT_LLM_ARGS="$DEFAULT_LLM_ARGS --cpu-offload-gb $DEFAULT_CPU_OFFLOAD_GB"
fi
if [ -n "$DEFAULT_LLM_KV_CACHE_MEMORY_BYTES" ]; then
  DEFAULT_LLM_ARGS="$DEFAULT_LLM_ARGS --kv-cache-memory-bytes $DEFAULT_LLM_KV_CACHE_MEMORY_BYTES"
fi
if [ "$DEFAULT_LLM_TENSOR_PARALLEL_SIZE" -gt 1 ]; then
  DEFAULT_LLM_ARGS="$DEFAULT_LLM_ARGS --tensor-parallel-size $DEFAULT_LLM_TENSOR_PARALLEL_SIZE --distributed-executor-backend mp"
fi
DEFAULT_EMBEDDING_ARGS="--trust-remote-code --enforce-eager --runner pooling --max-model-len $DEFAULT_EMBEDDING_MAX_MODEL_LEN --max-num-seqs 1 --max-num-batched-tokens $DEFAULT_EMBEDDING_MAX_BATCHED_TOKENS --gpu-memory-utilization $DEFAULT_EMBEDDING_GPU_MEMORY_UTILIZATION --swap-space ${TOOL_STAR_EMBEDDING_SIDECAR_SWAP_SPACE_GB:-8}"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$PID_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found. Install/activate the serving environment before starting sidecars." >&2
  exit 1
fi
active_ray_worker_processes() {
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi
  # Only block on signatures that are specific to rollout/FSDP training.
  # Generic ray::TaskRunner.run also appears in unrelated Ray tasks and caused e2e eval false positives.
  pgrep -au "$(id -u)" -f 'ray::.*WorkerDict|actor_rollout_init_model|verl\.trainer\.main_ppo|fsdp_workers|vllm_rollout|video_agent_rollout|video_agent_episode' 2>/dev/null || true
}
if [ "$SIDECAR_REFUSE_WHEN_RAY_WORKER_ACTIVE" = "1" ]; then
  active_ray_processes="$(active_ray_worker_processes)"
  if [ -n "$active_ray_processes" ]; then
    echo "Refusing to start sidecars while Ray/FSDP training workers are active." >&2
    echo "Start non-planner sidecars before launching training, or stop the current Ray/training job first." >&2
    echo "Active Ray/training process(es):" >&2
    printf '%s\n' "$active_ray_processes" >&2
    echo "Override only if you know the GPU memory profile is stable: TOOL_STAR_SIDECAR_REFUSE_WHEN_RAY_WORKER_ACTIVE=0" >&2
    exit 1
  fi
fi
if [ "$SIDECAR_CLEANUP_STALE_PROCESSES" = "1" ]; then
  echo "Cleaning stale managed sidecar processes before launch..."
  TOOL_STAR_SIDECAR_KILL_ORPHAN_VLLM_WORKERS="${TOOL_STAR_SIDECAR_KILL_ORPHAN_VLLM_WORKERS:-1}" \
    sh "$SCRIPT_DIR/stop_non_planner_sidecars.sh" || true
fi
filter_vllm_serve_args() {
  model_path="$1"
  raw_args="$2"
  python - "$model_path" "$raw_args" <<'PY'
import shlex
import subprocess
import sys
model_path = sys.argv[1] if len(sys.argv) > 1 else ""
raw_args = sys.argv[2] if len(sys.argv) > 2 else ""
args = shlex.split(raw_args)
help_outputs = []
for command in (
    ["vllm", "serve", "--help"],
    ["vllm", "serve", model_path, "--help"] if model_path else [],
):
    if not command:
        continue
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        help_outputs.append(completed.stdout or "")
    except Exception:
        pass
help_text = "\n".join(help_outputs)
def is_flag(token: str) -> bool:
    return str(token or "").startswith("-")
known_supported_probe_flags = {
    "--host",
    "--port",
    "--api-key",
    "--served-model-name",
}
known_unsupported_flags = {
    "--swap-space",
    "--task",
}
help_is_reliable = sum(1 for flag in known_supported_probe_flags if flag in help_text) >= 2
filtered = []
skipped = []
index = 0
while index < len(args):
    token = args[index]
    if not is_flag(token):
        filtered.append(token)
        index += 1
        continue
    flag = token.split("=", 1)[0]
    should_skip = flag in known_unsupported_flags
    if help_is_reliable:
        should_skip = should_skip or flag not in help_text
    if not should_skip:
        filtered.append(token)
        if "=" not in token and index + 1 < len(args) and not is_flag(args[index + 1]):
            index += 1
            filtered.append(args[index])
        index += 1
        continue
    skipped.append(token)
    if "=" not in token and index + 1 < len(args) and not is_flag(args[index + 1]):
        skipped.append(args[index + 1])
        index += 1
    index += 1
if skipped:
    print(
        "Skipping unsupported vllm serve arg(s): " + " ".join(shlex.quote(item) for item in skipped),
        file=sys.stderr,
    )
print(" ".join(shlex.quote(item) for item in filtered))
PY
}
eval "$(
python - <<'PY'
from config.settings import settings
print("LLM_MODEL_PATH=" + repr(settings.subagent_model_path))
print("TOOL_SEARCH_EMBEDDING_MODEL_PATH=" + repr(settings.embedding_model_path))
print("GROUNDING_EMBEDDING_MODEL_PATH=" + repr(settings.qwen3_vl_embedding_model_path))
PY
)"
echo "Sidecar GPU coordination mode: $SIDECAR_COLOCATION_MODE (visible GPUs: $GPU_LIST)"
if [ "$SIDECAR_COLOCATION_MODE" = "shared-4gpu" ]; then
  echo "Shared GPU assignment policy=${SIDECAR_GPU_POLICY:-spread_last} offset=${SIDECAR_GPU_OFFSET:-0}: non_planner_llm=$LLM_CUDA_VISIBLE_DEVICES tool_search_embedding=$TOOL_SEARCH_EMBEDDING_CUDA_VISIBLE_DEVICES grounding_embedding=$GROUNDING_EMBEDDING_CUDA_VISIBLE_DEVICES"
  echo "Warning: sidecars will share GPUs with planner training. Defaults keep concurrency low while avoiding vLLM CPU-offload and multimodal chunk splitting instability: LLM replicas=$DEFAULT_LLM_REPLICA_COUNT, tensor-parallel-size=$DEFAULT_LLM_TENSOR_PARALLEL_SIZE, gpu-memory-utilization=$DEFAULT_LLM_GPU_MEMORY_UTILIZATION, kv-cache-memory-bytes=${DEFAULT_LLM_KV_CACHE_MEMORY_BYTES:-auto}, max-num-batched-tokens=$DEFAULT_LLM_MAX_BATCHED_TOKENS, max-tokens-per-mm-item=$SIDECAR_LLM_MAX_TOKENS_PER_MM_ITEM, disable-chunked-prefill=$SIDECAR_LLM_DISABLE_CHUNKED_PREFILL, disable-chunked-mm-input=$SIDECAR_LLM_DISABLE_CHUNKED_MM_INPUT, disable-async-scheduling=$SIDECAR_LLM_DISABLE_ASYNC_SCHEDULING, cpu-offload-gb=$DEFAULT_CPU_OFFLOAD_GB, visual-service-max-frames=$VISUAL_SERVICE_MAX_FRAMES, visual-service-max-pixels=$VISUAL_SERVICE_MAX_PIXELS." >&2
  echo "If co-location OOM occurs after sidecars become healthy, first reduce trainer batch size or sidecar replica count; setting gpu-memory-utilization below the model footprint can prevent vLLM from allocating KV cache blocks." >&2
fi
if [ "$SIDECAR_GPU_QUIESCE_SECONDS" -gt 0 ]; then
  echo "Waiting ${SIDECAR_GPU_QUIESCE_SECONDS}s for previous sidecar GPU memory releases to settle..."
  sleep "$SIDECAR_GPU_QUIESCE_SECONDS"
fi
start_vllm() {
  name="$1"
  port="$2"
  model_path="$3"
  cuda_devices="$4"
  task="$5"
  extra_args="$6"
  pid_file="$PID_DIR/$name.pid"
  log_file="$LOG_DIR/$name.log"
  if [ -f "$pid_file" ]; then
    old_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
      echo "$name already running: pid=$old_pid base_url=http://$HOST:$port/v1"
      return 0
    fi
  fi
  echo "Starting $name on http://$HOST:$port/v1"
  echo "  model=$model_path"
  eval "set -- $(filter_vllm_serve_args "$model_path" "$extra_args")"
  launch_prefix=""
  if command -v setsid >/dev/null 2>&1; then
    launch_prefix="setsid"
  fi
  if [ -n "$SIDECAR_VLLM_USE_V1" ]; then
    export VLLM_USE_V1="$SIDECAR_VLLM_USE_V1"
  else
    unset VLLM_USE_V1
  fi
  CUDA_VISIBLE_DEVICES="$cuda_devices" nohup $launch_prefix vllm serve "$model_path" \
    --host "$HOST" \
    --port "$port" \
    --api-key "$API_KEY" \
    --served-model-name "$model_path" \
    "$@" >"$log_file" 2>&1 &
  echo "$!" > "$pid_file"
  echo "  pid=$(cat "$pid_file") log=$log_file"
}
check_http_health() {
  url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 -H "Authorization: Bearer ${API_KEY:-EMPTY}" "$url" >/dev/null 2>&1
    return $?
  fi
  python - "$url" "${API_KEY:-EMPTY}" <<'PY'
import sys
import urllib.request
url = sys.argv[1]
api_key = sys.argv[2] if len(sys.argv) > 2 else "EMPTY"
request = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
    method="GET",
)
try:
    with urllib.request.urlopen(request, timeout=2) as response:
        code = getattr(response, "status", 200)
    raise SystemExit(0 if 200 <= int(code) < 300 else 1)
except Exception:
    raise SystemExit(1)
PY
}
check_openai_sidecar_smoke() {
  name="$1"
  port="$2"
  model_path="$3"
  task="$4"
  [ "$SIDECAR_SMOKE_CHECK_ENABLED" = "1" ] || return 0
  python - \
    "http://$HOST:$port/v1" \
    "$API_KEY" \
    "$model_path" \
    "$task" \
    "$name" \
    "$SIDECAR_SMOKE_TIMEOUT_SECONDS" \
    "$VISUAL_SERVICE_MAX_PIXELS" \
    "$QWEN_VL_IMAGE_MAX_PIXELS" <<'PY'
import json, sys
import urllib.error, urllib.request
import base64, binascii, math, struct, zlib
base_url = sys.argv[1].rstrip("/")
api_key = sys.argv[2]
model = sys.argv[3]
task = sys.argv[4]
name = sys.argv[5]
try:
    timeout = max(5.0, float(sys.argv[6]))
except Exception:
    timeout = 120.0
try:
    smoke_pixels = max(28 * 28, int(sys.argv[7]))
except Exception:
    smoke_pixels = 336 * 336
try:
    qwen_vl_smoke_pixels = max(28 * 28, int(sys.argv[8]))
except Exception:
    qwen_vl_smoke_pixels = smoke_pixels
smoke_pixels = min(smoke_pixels, qwen_vl_smoke_pixels)
pooling_url = (base_url[:-3] if base_url.endswith("/v1") else base_url) + "/pooling"
def png_data_url(max_pixels: int) -> str:
    side = max(28, int(math.ceil(math.sqrt(max_pixels))))
    raw = b"".join(b"\0" + (b"\xff\xff\xff" * side) for _ in range(side))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
def post_json(url: str, payload: dict) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            body = response.read(4096)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    if not (200 <= status < 300):
        raise RuntimeError(f"HTTP {status}: {body.decode('utf-8', errors='replace')}")
def post_first_success(candidates) -> None:
    errors = []
    for url, payload in candidates:
        try:
            post_json(url, payload)
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))
try:
    if task == "embed":
        post_first_success(
            (
                (base_url + "/embeddings", {"model": model, "input": "sidecar health"}),
                (base_url + "/embeddings", {"model": model, "input": ["sidecar health"]}),
                (pooling_url, {"model": model, "input": "sidecar health", "task": "embed"}),
                (pooling_url, {"model": model, "input": ["sidecar health"], "task": "embed"}),
            )
        )
    else:
        multimodal_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": png_data_url(smoke_pixels)},
                        },
                        {"type": "text", "text": "Reply with ok."},
                    ],
                }
            ],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        post_json(base_url + "/chat/completions", multimodal_payload)
except Exception as exc:
    print(f"{name} OpenAI-compatible smoke check failed for task={task}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}
wait_for_sidecar_ready() {
  name="$1"
  port="$2"
  model_path="$3"
  task="$4"
  pid_file="$PID_DIR/$name.pid"
  log_file="$LOG_DIR/$name.log"
  health_url="http://$HOST:$port/health"
  start_ts="$(date +%s)"
  last_progress_ts="$start_ts"
  deadline_ts="$((start_ts + SIDECAR_STARTUP_TIMEOUT_SECONDS))"
  echo "Waiting for $name health: $health_url"
  while :; do
    if [ -f "$pid_file" ]; then
      pid="$(cat "$pid_file" 2>/dev/null || true)"
      if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        echo "$name exited before becoming healthy. Last log lines:" >&2
        tail -n 80 "$log_file" >&2 2>/dev/null || true
        exit 1
      fi
    fi
    ready_probe=""
    check_http_health "$health_url" && ready_probe="health"
    [ -n "$ready_probe" ] || ! grep -q "Application startup complete" "$log_file" 2>/dev/null || ready_probe="startup-log"
    if [ -n "$ready_probe" ]; then
      echo "$name $ready_probe passed; running OpenAI-compatible smoke check..."
      if ! check_openai_sidecar_smoke "$name" "$port" "$model_path" "$task"; then
        echo "$name $ready_probe passed, but OpenAI-compatible smoke check failed. Last log lines:" >&2
        tail -n 80 "$log_file" >&2 2>/dev/null || true
        exit 1
      fi
      echo "$name ready: http://$HOST:$port/v1"
      return 0
    fi
    now_ts="$(date +%s)"
    if [ "$now_ts" -ge "$deadline_ts" ]; then
      echo "$name did not become healthy within ${SIDECAR_STARTUP_TIMEOUT_SECONDS}s. Last log lines:" >&2
      tail -n 80 "$log_file" >&2 2>/dev/null || true
      exit 1
    fi
    if [ "$((now_ts - last_progress_ts))" -ge 30 ]; then
      echo "$name still starting after $((now_ts - start_ts))s; recent log lines:" >&2
      tail -n 12 "$log_file" 2>/dev/null | sed 's/^/  /' >&2 || true
      last_progress_ts="$now_ts"
    fi
    sleep "$SIDECAR_HEALTH_CHECK_INTERVAL_SECONDS"
  done
}
start_vllm_pool() {
  name="$1"
  port_base="$2"
  model_path="$3"
  cuda_sequence="$4"
  task="$5"
  extra_args="$6"
  replica_count="$7"
  endpoints=""
  index=0
  case "$cuda_sequence" in
    *';'*)
      cuda_lines="$(printf '%s' "$cuda_sequence" | tr ';' '\n')"
      ;;
    *)
      cuda_lines="$(printf '%s' "$cuda_sequence" | tr ',' '\n')"
      ;;
  esac
  while IFS= read -r cuda_device; do
    [ -n "$cuda_device" ] || continue
    [ "$index" -ge "$replica_count" ] && break
    port="$((port_base + index * PORT_STRIDE))"
    replica_name="${name}_${index}"
    start_vllm "$replica_name" "$port" "$model_path" "$cuda_device" "$task" "$extra_args" >&2
    wait_for_sidecar_ready "$replica_name" "$port" "$model_path" "$task" >&2
    endpoints="${endpoints:+$endpoints,}http://$HOST:$port/v1"
    index=$((index + 1))
  done <<EOF
$cuda_lines
EOF
  printf '%s\n' "$endpoints"
}
write_runtime_env_file() {
  non_planner_api_bases="$1"
  tool_search_embedding_api_bases="$2"
  grounding_embedding_api_bases="$3"
  python - "$ENV_FILE" "$non_planner_api_bases" "$tool_search_embedding_api_bases" "$grounding_embedding_api_bases" "$LOAD_BALANCE_POLICY" "$VISUAL_SERVICE_MAX_FRAMES" "$VISUAL_SERVICE_MAX_PIXELS" <<'PY'
import shlex
import sys
from pathlib import Path
env_file = Path(sys.argv[1])
non_planner_api_bases = sys.argv[2]
tool_search_embedding_api_bases = sys.argv[3]
grounding_embedding_api_bases = sys.argv[4]
load_balance_policy = sys.argv[5] or "least_inflight"
visual_service_max_frames = sys.argv[6]
visual_service_max_pixels = sys.argv[7]
def first_endpoint(value: str) -> str:
    parts = [part.strip() for part in str(value or "").replace(";", ",").split(",") if part.strip()]
    return parts[0] if parts else ""
values = {
    "TOOL_STAR_SIDECAR_LOAD_BALANCE_POLICY": load_balance_policy,
    "TOOL_STAR_NON_PLANNER_MODEL_PROVIDER": "service" if non_planner_api_bases else "process",
    "TOOL_STAR_AUX_MODEL_PROVIDER": "service" if non_planner_api_bases else "process",
    "TOOL_STAR_SUBAGENT_SERVICE_MODEL_PROVIDER": "service" if non_planner_api_bases else "process",
    "TOOL_STAR_SUBAGENT_SERVICE_API_BASES": non_planner_api_bases,
    "TOOL_STAR_SUBAGENT_SERVICE_API_BASE": first_endpoint(non_planner_api_bases),
    "TOOL_STAR_NON_PLANNER_API_BASES": non_planner_api_bases,
    "TOOL_STAR_NON_PLANNER_API_BASE": first_endpoint(non_planner_api_bases),
    "TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASES": tool_search_embedding_api_bases,
    "TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASE": first_endpoint(tool_search_embedding_api_bases),
    "TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASES": tool_search_embedding_api_bases,
    "TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASE": first_endpoint(tool_search_embedding_api_bases),
    "TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASES": grounding_embedding_api_bases,
    "TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASE": first_endpoint(grounding_embedding_api_bases),
    "TOOL_STAR_PREPROCESS_EMBEDDING_API_BASES": grounding_embedding_api_bases,
    "TOOL_STAR_PREPROCESS_EMBEDDING_API_BASE": first_endpoint(grounding_embedding_api_bases),
    "TOOL_STAR_VISUAL_SERVICE_MAX_FRAMES": visual_service_max_frames,
    "TOOL_STAR_VISUAL_SERVICE_MAX_PIXELS": visual_service_max_pixels,
}
env_file.parent.mkdir(parents=True, exist_ok=True)
with env_file.open("w", encoding="utf-8") as handle:
    handle.write("#!/usr/bin/env sh\n")
    for key, value in values.items():
        if not value:
            continue
        handle.write(f"export {key}={shlex.quote(str(value))}\n")
PY
}
NON_PLANNER_API_BASES="$(start_vllm_pool \
  "non_planner_llm" \
  "$LLM_PORT" \
  "$LLM_MODEL_PATH" \
  "$LLM_CUDA_VISIBLE_DEVICES" \
  "chat" \
  "${TOOL_STAR_NON_PLANNER_SIDECAR_VLLM_ARGS:-$DEFAULT_LLM_ARGS}" \
  "$DEFAULT_LLM_REPLICA_COUNT")"
TOOL_SEARCH_EMBEDDING_API_BASES="$(start_vllm_pool \
  "tool_search_embedding" \
  "$TOOL_SEARCH_EMBEDDING_PORT" \
  "$TOOL_SEARCH_EMBEDDING_MODEL_PATH" \
  "$TOOL_SEARCH_EMBEDDING_CUDA_VISIBLE_DEVICES" \
  "embed" \
  "$(if [ -n "${TOOL_STAR_TOOL_SEARCH_EMBEDDING_VLLM_ARGS:-}" ]; then printf '%s --runner pooling --max-model-len %s' "$TOOL_STAR_TOOL_SEARCH_EMBEDDING_VLLM_ARGS" "$DEFAULT_EMBEDDING_MAX_MODEL_LEN"; else printf '%s' "$DEFAULT_EMBEDDING_ARGS"; fi)" \
  "$DEFAULT_TOOL_SEARCH_REPLICA_COUNT")"
GROUNDING_EMBEDDING_API_BASES="$(start_vllm_pool \
  "grounding_embedding" \
  "$GROUNDING_EMBEDDING_PORT" \
  "$GROUNDING_EMBEDDING_MODEL_PATH" \
  "$GROUNDING_EMBEDDING_CUDA_VISIBLE_DEVICES" \
  "embed" \
  "$(if [ -n "${TOOL_STAR_GROUNDING_EMBEDDING_VLLM_ARGS:-}" ]; then printf '%s --runner pooling --max-model-len %s' "$TOOL_STAR_GROUNDING_EMBEDDING_VLLM_ARGS" "$DEFAULT_EMBEDDING_MAX_MODEL_LEN"; else printf '%s' "$DEFAULT_EMBEDDING_ARGS"; fi)" \
  "$DEFAULT_GROUNDING_REPLICA_COUNT")"
write_runtime_env_file "$NON_PLANNER_API_BASES" "$TOOL_SEARCH_EMBEDDING_API_BASES" "$GROUNDING_EMBEDDING_API_BASES"
echo "Sidecar base URLs:"
echo "  non_planner_api_bases=$NON_PLANNER_API_BASES"
echo "  tool_search_embedding_api_bases=$TOOL_SEARCH_EMBEDDING_API_BASES"
echo "  grounding_embedding_api_bases=$GROUNDING_EMBEDDING_API_BASES"
echo "Runtime env written to: $ENV_FILE"
