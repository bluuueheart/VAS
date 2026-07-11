#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
# This script still uses verl's historical `main_ppo.py` entrypoint and config
# naming, but the actual algorithm is controlled by `algorithm.adv_estimator`.
# The current default config sets `algorithm.adv_estimator=grpo`.
CONFIG_NAME="${VIDEO_AGENT_TRAINER_CONFIG_NAME:-${VIDEO_AGENT_PPO_CONFIG_NAME:-ppo_video_agent_trainer}}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src/verl:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOOL_STAR_SHARED_ACCELERATOR_QUEUE_ROOT="${TOOL_STAR_SHARED_ACCELERATOR_QUEUE_ROOT:-$PROJECT_ROOT/rl_workdir/cache/_accelerator_queue}"
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
SIDECAR_ENV_FILE="${TOOL_STAR_SIDECAR_ENV_FILE:-$PROJECT_ROOT/rl_workdir/sidecars/env.sh}"
ENABLE_NON_PLANNER_SIDECARS="${TOOL_STAR_ENABLE_NON_PLANNER_SIDECARS:-0}"
DEFAULT_NON_PLANNER_PROVIDER="$(if [ "$ENABLE_NON_PLANNER_SIDECARS" = "1" ]; then printf 'service'; else printf 'process'; fi)"
DEFAULT_EMBEDDING_PROVIDER="$(if [ "$ENABLE_NON_PLANNER_SIDECARS" = "1" ]; then printf 'service'; else printf 'local'; fi)"
load_sidecar_env() {
  [ -f "$SIDECAR_ENV_FILE" ] || return 1
  # shellcheck disable=SC1090
  . "$SIDECAR_ENV_FILE"
}
if [ "$ENABLE_NON_PLANNER_SIDECARS" = "1" ]; then
  load_sidecar_env || true
  if [ -z "${TOOL_STAR_NON_PLANNER_API_BASES:-${TOOL_STAR_NON_PLANNER_API_BASE:-}}" ] \
    && [ -z "${TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASES:-${TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASE:-}}" ] \
    && [ -z "${TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASES:-${TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASE:-}}" ]; then
    unset TOOL_STAR_NON_PLANNER_MODEL_PROVIDER TOOL_STAR_AUX_MODEL_PROVIDER
    unset TOOL_STAR_NON_PLANNER_EMBEDDING_PROVIDER TOOL_STAR_TOOL_SEARCH_EMBEDDING_PROVIDER
    unset TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_PROVIDER TOOL_STAR_PREPROCESS_EMBEDDING_PROVIDER
    unset TOOL_STAR_SKIP_SIDECAR_PREFLIGHT TOOL_STAR_AUTO_START_SIDECARS
  fi
fi

apply_local_provider_defaults() {
  [ "$ENABLE_NON_PLANNER_SIDECARS" != "1" ] || return 0
  echo "Non-planner sidecars disabled; using local process/local providers." >&2
  export TOOL_STAR_NON_PLANNER_MODEL_PROVIDER="process"
  export TOOL_STAR_AUX_MODEL_PROVIDER="process"
  export TOOL_STAR_SUBAGENT_SERVICE_MODEL_PROVIDER="${TOOL_STAR_SUBAGENT_SERVICE_MODEL_PROVIDER:-process}"
  export TOOL_STAR_NON_PLANNER_EMBEDDING_PROVIDER="local"
  export TOOL_STAR_TOOL_SEARCH_EMBEDDING_PROVIDER="local"
  export TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_PROVIDER="local"
  export TOOL_STAR_PREPROCESS_EMBEDDING_PROVIDER="local"
  export TOOL_STAR_SKIP_SIDECAR_PREFLIGHT="1"
  export TOOL_STAR_AUTO_START_SIDECARS="0"
  unset TOOL_STAR_NON_PLANNER_API_BASES TOOL_STAR_NON_PLANNER_API_BASE
  unset TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASES TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASE
  unset TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASES TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASE
  unset TOOL_STAR_PREPROCESS_EMBEDDING_API_BASES TOOL_STAR_PREPROCESS_EMBEDDING_API_BASE
  unset TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASES TOOL_STAR_NON_PLANNER_EMBEDDING_API_BASE
}
apply_local_provider_defaults

export TOOL_STAR_ENABLE_GENERATION_TIMEOUT_INTERCEPTION="${TOOL_STAR_ENABLE_GENERATION_TIMEOUT_INTERCEPTION:-1}"
export TOOL_STAR_ENABLE_ROLLOUT_WALL_CLOCK_TIMEOUT="${TOOL_STAR_ENABLE_ROLLOUT_WALL_CLOCK_TIMEOUT:-0}"

visible_gpu_count() {
  raw="${CUDA_VISIBLE_DEVICES:-}"
  if [ -n "$raw" ]; then
    python - "$raw" <<'PY'
import sys
print(len([x for x in sys.argv[1].split(",") if x.strip()]) or 0)
PY
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
    return 0
  fi
  printf '0\n'
}
apply_shared_gpu_memory_defaults() {
  [ "${TOOL_STAR_FORCE_SHARED_GPU_LLM_SIDECAR:-0}" != "1" ] || return 0
  gpu_count="$(visible_gpu_count)"
  [ "$gpu_count" -gt 0 ] && [ "$gpu_count" -lt 7 ] || return 0
  echo "Shared-GPU memory guard: keeping global non-planner LLM on local process provider." >&2
  export TOOL_STAR_NON_PLANNER_MODEL_PROVIDER="process"
  export TOOL_STAR_AUX_MODEL_PROVIDER="process"
  unset TOOL_STAR_NON_PLANNER_API_BASES TOOL_STAR_NON_PLANNER_API_BASE
  unset TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASES TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASE
  unset TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASES TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASE
}
apply_shared_gpu_memory_defaults

provider_uses_service() {
  provider="$1"
  case "$(printf '%s' "$provider" | tr '[:upper:]' '[:lower:]')" in
    disabled|disable|off|false|0|local|transformers|process|local_process|subprocess)
      return 1
      ;;
  esac
  return 0
}
sidecar_provider_enabled() {
  provider_uses_service "${TOOL_STAR_NON_PLANNER_MODEL_PROVIDER:-${TOOL_STAR_AUX_MODEL_PROVIDER:-$DEFAULT_NON_PLANNER_PROVIDER}}"
}
subagent_service_provider_enabled() {
  provider_uses_service "${TOOL_STAR_SUBAGENT_SERVICE_MODEL_PROVIDER:-${TOOL_STAR_SUBAGENT_MODEL_PROVIDER:-${TOOL_STAR_NON_PLANNER_MODEL_PROVIDER:-${TOOL_STAR_AUX_MODEL_PROVIDER:-$DEFAULT_NON_PLANNER_PROVIDER}}}}"
}
tool_search_embedding_provider_enabled() {
  provider_uses_service "${TOOL_STAR_TOOL_SEARCH_EMBEDDING_PROVIDER:-${TOOL_STAR_NON_PLANNER_EMBEDDING_PROVIDER:-$DEFAULT_EMBEDDING_PROVIDER}}"
}
grounding_embedding_provider_enabled() {
  provider_uses_service "${TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_PROVIDER:-${TOOL_STAR_PREPROCESS_EMBEDDING_PROVIDER:-${TOOL_STAR_NON_PLANNER_EMBEDDING_PROVIDER:-$DEFAULT_EMBEDDING_PROVIDER}}}"
}
any_sidecar_provider_enabled() {
  sidecar_provider_enabled || subagent_service_provider_enabled || tool_search_embedding_provider_enabled || grounding_embedding_provider_enabled
}

preflight_sidecar_endpoints() {
  name="$1"
  endpoints="$2"
  api_key="$3"
  [ -n "$endpoints" ] || return 0
  [ "${TOOL_STAR_SKIP_SIDECAR_PREFLIGHT:-0}" != "1" ] || return 0
  python - "$name" "$endpoints" "$api_key" <<'PY'
import sys
import urllib.request

name = sys.argv[1]
raw_endpoints = sys.argv[2]
api_key = sys.argv[3] if len(sys.argv) > 3 else "EMPTY"
endpoints = []
for item in raw_endpoints.replace(";", ",").split(","):
    endpoint = item.strip().rstrip("/")
    if not endpoint or endpoint in endpoints:
        continue
    endpoints.append(endpoint)

failed = []
for endpoint in endpoints:
    base = endpoint[:-3] if endpoint.endswith("/v1") else endpoint
    health_url = base.rstrip("/") + "/health"
    models_url = endpoint.rstrip("/") + "/models"
    failures = []
    try:
        request = urllib.request.Request(
            health_url,
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            status = int(getattr(response, "status", 200))
        if not (200 <= status < 300):
            failures.append(f"health_status={status}")
    except Exception as exc:
        failures.append(f"health_error={exc}")
    if failures:
        try:
            request = urllib.request.Request(
                models_url,
                headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                status = int(getattr(response, "status", 200))
            if 200 <= status < 300:
                continue
            failures.append(f"models_status={status}")
        except Exception as exc:
            failures.append(f"models_error={exc}")
        failed.append(f"{endpoint} " + ", ".join(failures))

if failed:
    print(
        f"Sidecar endpoint preflight failed for {name}. "
        "Run scripts/start_non_planner_sidecars.sh again and ensure the runtime env file is fresh.",
        file=sys.stderr,
    )
    for failure in failed:
        print(f"  {failure}", file=sys.stderr)
    raise SystemExit(1)
PY
}

run_sidecar_preflight() {
  [ "${TOOL_STAR_SKIP_SIDECAR_PREFLIGHT:-0}" != "1" ] || return 0
  non_planner_endpoints="${TOOL_STAR_NON_PLANNER_API_BASES:-${TOOL_STAR_NON_PLANNER_API_BASE:-}}"
  if [ -z "$non_planner_endpoints" ]; then
    non_planner_endpoints="${TOOL_STAR_SUBAGENT_SERVICE_API_BASES:-${TOOL_STAR_SUBAGENT_SERVICE_API_BASE:-}}"
  fi
  tool_search_endpoints="${TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASES:-${TOOL_STAR_TOOL_SEARCH_EMBEDDING_API_BASE:-}}"
  grounding_endpoints="${TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASES:-${TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASE:-}}"
  sidecar_api_key="${TOOL_STAR_SIDECAR_API_KEY:-EMPTY}"
  missing=0
  if sidecar_provider_enabled && [ -z "$non_planner_endpoints" ]; then echo "Missing non-planner sidecar endpoint runtime env." >&2; missing=1; fi
  if subagent_service_provider_enabled && [ -z "$non_planner_endpoints" ]; then echo "Missing subagent_service sidecar endpoint runtime env." >&2; missing=1; fi
  if tool_search_embedding_provider_enabled && [ -z "$tool_search_endpoints" ]; then echo "Missing Tool_Search embedding sidecar endpoint runtime env." >&2; missing=1; fi
  if grounding_embedding_provider_enabled && [ -z "$grounding_endpoints" ]; then echo "Missing grounding/preprocess embedding sidecar endpoint runtime env." >&2; missing=1; fi
  [ "$missing" -eq 0 ] || return 1
  if sidecar_provider_enabled || subagent_service_provider_enabled; then preflight_sidecar_endpoints "non_planner" "$non_planner_endpoints" "$sidecar_api_key" || return 1; fi
  if tool_search_embedding_provider_enabled; then preflight_sidecar_endpoints "tool_search_embedding" "$tool_search_endpoints" "$sidecar_api_key" || return 1; fi
  if grounding_embedding_provider_enabled; then preflight_sidecar_endpoints "grounding_embedding" "$grounding_endpoints" "$sidecar_api_key" || return 1; fi
}

ensure_sidecars_ready() {
  any_sidecar_provider_enabled || return 0
  if run_sidecar_preflight; then
    return 0
  fi
  if [ "${TOOL_STAR_AUTO_START_SIDECARS:-1}" != "1" ]; then
    echo "Sidecar preflight failed and TOOL_STAR_AUTO_START_SIDECARS is disabled." >&2
    echo "Run scripts/start_non_planner_sidecars.sh before launching training." >&2
    exit 1
  fi
  echo "Starting non-planner sidecars before Ray/FSDP workers are created..."
  sh "$SCRIPT_DIR/start_non_planner_sidecars.sh"
  load_sidecar_env || {
    echo "Sidecar startup completed but runtime env file is missing: $SIDECAR_ENV_FILE" >&2
    exit 1
  }
  run_sidecar_preflight
}

ensure_sidecars_ready

if [ -n "${VIDEO_AGENT_WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY="${VIDEO_AGENT_WANDB_API_KEY}"
fi

has_use_remove_padding_override=0
for arg in "$@"; do
  case "$arg" in
    actor_rollout_ref.model.use_remove_padding=*)
      has_use_remove_padding_override=1
      break
      ;;
  esac
done
if [ "$has_use_remove_padding_override" -ne 1 ]; then
  # Keep the video-agent entrypoint aligned with the historical train.sh scripts:
  # actor/ref sequence-parallel paths require remove-padding, and the user-facing
  # launch command should not need to remember that hidden dependency.
  set -- "$@" "actor_rollout_ref.model.use_remove_padding=True"
fi

if [ -n "${VIDEO_AGENT_RAY_DASHBOARD_ADDRESS:-}" ]; then
  RAY_RUNTIME_ENV="${VIDEO_AGENT_RAY_RUNTIME_ENV:-$PROJECT_ROOT/src/verl/verl/trainer/runtime_env.yaml}"
  ray job submit \
    --address "${VIDEO_AGENT_RAY_DASHBOARD_ADDRESS}" \
    --runtime-env "${RAY_RUNTIME_ENV}" \
    --no-wait \
    -- \
    python -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" "$@"
else
  python -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" "$@"
fi
