import argparse
import asyncio
import concurrent.futures
import importlib
import hashlib
import inspect
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RL_ROOT = Path(os.environ.get("CFV_RUNTIME_SOURCE_ROOT") or _PROJECT_ROOT / "RL").resolve()
if _RL_ROOT.exists() and str(_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RL_ROOT))

try:
    from core.accelerator_queue import (
        SharedAcceleratorQueueLock,
        resolve_local_accelerator_slot_key,
    )
except Exception:  # pragma: no cover - the scheduler remains usable without filelock.
    SharedAcceleratorQueueLock = None  # type: ignore[assignment]
    resolve_local_accelerator_slot_key = None  # type: ignore[assignment]

_ONLINE_EVOLUTION_IMPORT_ERROR: Optional[BaseException] = None
try:
    from evolution.online import OnlineEvolutionController
    from evolution.source_mutation import materialize_source_snapshot
except Exception as exc:  # pragma: no cover - isolated unit tests may omit RL deps.
    OnlineEvolutionController = None  # type: ignore[assignment]
    materialize_source_snapshot = None  # type: ignore[assignment]
    _ONLINE_EVOLUTION_IMPORT_ERROR = exc

from mix_datasets import (
    DatasetInventory,
    MixSample,
    load_mlvu_inventory,
    load_mvbench_inventory,
    load_videomme_inventory,
    resolve_path,
)
from mix_state import (
    SAMPLE_RESULT_FILENAME,
    has_participated,
    import_existing_correct_result,
    import_existing_participated_result,
    load_progress,
    load_result_items_from_progress,
    save_progress,
    summarize_progress,
    upsert_progress_record,
)
from reward_metrics import summarize_mix_reward_features


LOGGER = logging.getLogger("mix_batch")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
RUN_RESULT_LOG_MARKER = "[RUN_RESULT_JSON]"
ONLINE_RUNTIME_RESTART_EXIT_CODE = 75
SUPPORTED_DATASET_ORDER = ("lvbench", "videomme", "egoschema", "mlvu", "mvbench")
DEFAULT_DATASET_ORDER = ("lvbench", "videomme", "egoschema", "mlvu", "mvbench")


@dataclass(frozen=True)
class BatchConfig:
    config_path: Path
    workspaces_root: Path
    progress_path: Path
    results_path: Path
    videomme_official_output_path: Path
    shared_preprocess_root: Path
    ffmpeg_binary: str
    parallel_samples: Optional[int]
    allow_same_device_parallel: bool
    accelerator_queue_enabled: bool
    accelerator_queue_root: Optional[Path]
    sample_worker_admission_capacity: int
    resident_model_workers_per_device: int
    accelerator_queue_memory_guard: bool
    accelerator_queue_min_free_gib: float
    accelerator_queue_reserve_gib: float
    persistent_worker_start_stagger_seconds: float
    reuse_worker_samples: int
    asr_beam_size: int
    asr_best_of: int
    asr_condition_on_previous_text: bool
    asr_vad_filter: bool
    caption_batch_size: int
    caption_max_frames: int
    caption_max_new_tokens: int
    datasets: Dict[str, Dict[str, Any]]
    dataset_order: Tuple[str, ...]
    pipeline_options: Dict[str, Any]


@dataclass(frozen=True)
class RunningSampleTask:
    sample_uid: str
    cache_key: str
    visible_device_id: str
    launch_index: int


_RUN_E2E_MODULE = None
_ONLINE_RUNTIME_SNAPSHOT_ID = ""
_ONLINE_RUNTIME_SOURCE_ROOT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the mixed LVBench/Video-MME/EgoSchema/MLVU/MVBench evaluator."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("mix_config.json")),
        help="Path to the Mix batch config JSON.",
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=None,
        help="Limit execution to the next N pending samples. Default: unlimited.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable preprocessing cache reuse for each launched run_e2e.py invocation.",
    )
    parser.add_argument(
        "--parallel-samples",
        type=int,
        default=None,
        help="Maximum number of samples to run concurrently. Default: auto-detect from visible CUDA devices.",
    )
    parser.add_argument(
        "--sample-uids",
        default="",
        help=(
            "Comma-separated Mix sample_uids to run exactly, in the given order. "
            "Useful for trigger/neighbor/regression replay."
        ),
    )
    parser.add_argument(
        "--skip_true",
        "--skip-true",
        dest="skip_true",
        action="store_true",
        help="Import already-correct existing per-sample results into progress state before scheduling.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Reset the current Mix progress checkpoint (archiving it first) and run every inventory sample. "
            "Use for a formal architecture/configuration change; per-sample artifacts are retained."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the resolved batch order without starting inference or writing results.",
    )
    parser.add_argument(
        "--reuse-worker-samples",
        type=int,
        default=None,
        help=(
            "In serial mode, process this many samples per isolated worker so "
            "the visual model can be reused between samples. Default: config value."
        ),
    )
    parser.add_argument(
        "--worker-sample-uid",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-sample-uids",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-sample-uids-file",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-sample-spec-file",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-device-id",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-paired-ablation-role",
        choices=("observer", "reflector"),
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-paired-ablation-roles",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-paired-ablation-root",
        default="",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)


def _unwrap_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) < 3:
        return text
    if not lines[0].startswith("```") or lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1])


def _strip_json_line_comments(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines)


def _load_json_object(config_path: Path) -> Dict[str, Any]:
    raw_text = config_path.read_text(encoding="utf-8-sig")
    normalized_text = _strip_json_line_comments(_unwrap_markdown_code_fence(raw_text)).strip()
    if not normalized_text:
        raise ValueError(f"Mix config is empty: {config_path}")
    payload = json.loads(normalized_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Mix config must be a JSON object: {config_path}")
    return payload


def _resolve_ffmpeg_binary(configured: Optional[str] = None) -> str:
    """Return an executable ffmpeg path, including shared-site-package fallback.

    Some workers can read the bundled imageio-ffmpeg binary from the shared
    filesystem but cannot execute it because the shared file mode is 0660.
    Copying that exact binary to a private temporary path and adding execute
    permission keeps video decoding real while avoiding a fake success or a
    dependency on a system package that may not exist.
    """

    requested = str(os.environ.get("FFMPEG_BINARY") or configured or "").strip()
    candidates: List[str] = []
    if requested and requested != "ffmpeg":
        candidates.append(requested)
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        candidates.append(which_ffmpeg)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path.resolve())

    try:
        import imageio_ffmpeg  # type: ignore

        binary_dir = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
        bundled = sorted(binary_dir.glob("ffmpeg-*"))
    except Exception:
        bundled = []
    source = next(
        (path for path in bundled if path.is_file() and path.stat().st_size > 0),
        None,
    )
    if source is not None:
        private_dir = Path(tempfile.gettempdir()) / "toolstar_runtime" / "ffmpeg"
        private_dir.mkdir(parents=True, exist_ok=True)
        target = private_dir / source.name
        try:
            if (
                not target.exists()
                or target.stat().st_size != source.stat().st_size
                or not os.access(target, os.X_OK)
            ):
                temporary = private_dir / f".{source.name}.{os.getpid()}.tmp"
                shutil.copy2(source, temporary)
                os.chmod(temporary, 0o755)
                os.replace(temporary, target)
            if os.access(target, os.X_OK):
                return str(target)
        except OSError:
            pass
    return requested or "ffmpeg"


def _parse_parallel_samples_value(raw_value: Any) -> Optional[int]:
    if raw_value in {None, ""}:
        return None
    if isinstance(raw_value, str) and str(raw_value).strip().lower() == "auto":
        return None
    value = int(raw_value)
    if value <= 0:
        return None
    return value


def _parse_dataset_order(
    raw_value: Any,
    datasets: Mapping[str, Dict[str, Any]],
) -> Tuple[str, ...]:
    default_order = tuple(name for name in DEFAULT_DATASET_ORDER if name in datasets) or DEFAULT_DATASET_ORDER
    if raw_value in (None, ""):
        return default_order
    if not isinstance(raw_value, (list, tuple)):
        raise ValueError("Mix config `dataset_order` must be an array of dataset names.")
    normalized_order: List[str] = []
    seen: Set[str] = set()
    for item in raw_value:
        dataset_name = str(item or "").strip().lower()
        if not dataset_name:
            continue
        if dataset_name not in SUPPORTED_DATASET_ORDER:
            raise ValueError(
                f"Unsupported Mix dataset `{dataset_name}` in `dataset_order`. "
                f"Supported values: {', '.join(SUPPORTED_DATASET_ORDER)}"
            )
        if dataset_name not in datasets:
            raise ValueError(
                f"Mix config `dataset_order` references `{dataset_name}` but `datasets.{dataset_name}` is missing."
            )
        if dataset_name in seen:
            raise ValueError(f"Mix config `dataset_order` contains duplicate dataset `{dataset_name}`.")
        normalized_order.append(dataset_name)
        seen.add(dataset_name)
    if not normalized_order:
        raise ValueError("Mix config `dataset_order` must contain at least one dataset.")
    return tuple(normalized_order)


def _load_pipeline_options(config_dir: Path, raw_value: Any) -> Dict[str, Any]:
    if raw_value in (None, ""):
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("Mix config `pipeline_options` must be a JSON object when provided.")
    options = dict(raw_value)
    prompt_path = options.get("planner_direct_answer_prompt_path")
    if prompt_path not in (None, ""):
        options["planner_direct_answer_prompt_path"] = resolve_path(config_dir, str(prompt_path))
    return options


def _env_path_override(config_dir: Path, env_name: str, fallback: Path) -> Path:
    raw_value = str(os.environ.get(env_name) or "").strip()
    if not raw_value:
        return fallback
    return resolve_path(config_dir, raw_value)


def _env_bool(env_name: str, default: bool = False) -> bool:
    raw_value = str(os.environ.get(env_name) or "").strip().lower()
    if not raw_value:
        return bool(default)
    if raw_value in {"1", "true", "yes", "y", "on"}:
        return True
    if raw_value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _queue_capacity_from_env() -> int:
    raw_value = (
        os.environ.get("TOOLSTAR_ACCELERATOR_QUEUE_CAPACITY")
        or os.environ.get("TOOLSTAR_BATCH_ACCELERATOR_QUEUE_CAPACITY")
        or "1"
    )
    try:
        return max(1, min(8, int(raw_value)))
    except (TypeError, ValueError):
        return 1


def _provider_queue_capacity_from_config(raw_config: Mapping[str, Any]) -> int:
    """Resolve concurrent provider slots independently from worker admission.

    Persistent workers already own the outer resident-model admission slots.
    Leaving the inner provider queue at its legacy default of one would make
    three admitted workers serialize every planner generation through one
    shared file lock, defeating the formal same-GPU topology.
    """

    raw_options = raw_config.get("pipeline_options")
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    raw_value = raw_options.get(
        "provider_accelerator_queue_capacity",
        raw_config.get(
            "provider_accelerator_queue_capacity",
            raw_config.get("sample_worker_admission_capacity", 2),
        ),
    )
    try:
        return max(1, min(8, int(raw_value)))
    except (TypeError, ValueError):
        return 2


def _generation_queue_capacity_from_config(raw_config: Mapping[str, Any]) -> int:
    """Resolve the global memory-sensitive generation queue capacity.

    This is intentionally separate from provider/model-load capacity.  A
    persistent worker may stay resident while only a bounded number of
    high-water-mark ``generate`` calls enter one physical GPU.
    """

    raw_options = raw_config.get("pipeline_options")
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    raw_value = raw_options.get(
        "generation_accelerator_queue_capacity",
        raw_config.get("generation_accelerator_queue_capacity", 1),
    )
    try:
        return max(1, min(8, int(raw_value)))
    except (TypeError, ValueError):
        return 1


def _bounded_config_int(
    payload: Mapping[str, Any],
    key: str,
    env_name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.environ.get(env_name)
    if raw_value in (None, ""):
        raw_value = payload.get(key, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_config_float(
    payload: Mapping[str, Any],
    key: str,
    env_name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.environ.get(env_name)
    if raw_value in (None, ""):
        raw_value = payload.get(key, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _inventory_dataset_order(inventories: Sequence[DatasetInventory]) -> List[str]:
    return [
        str(inventory.dataset_name).strip()
        for inventory in inventories
        if str(inventory.dataset_name).strip()
    ]


def _ordered_dataset_names(dataset_order: Sequence[str], present_names: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for item in dataset_order:
        dataset_name = str(item or "").strip()
        if dataset_name and dataset_name not in seen:
            ordered.append(dataset_name)
            seen.add(dataset_name)
    extras = sorted(
        {
            str(item or "").strip()
            for item in present_names
            if str(item or "").strip() and str(item or "").strip() not in seen
        }
    )
    ordered.extend(extras)
    return ordered


def load_config(config_path: Path) -> BatchConfig:
    config_path = config_path.resolve()
    config_dir = config_path.parent
    payload = _load_json_object(config_path)
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Mix config requires a top-level `datasets` object.")
    normalized_datasets = {str(key): dict(value) for key, value in datasets.items() if isinstance(value, dict)}
    workspaces_root = resolve_path(config_dir, str(payload.get("workspaces_root") or "./workspaces"))
    progress_path = resolve_path(config_dir, str(payload.get("progress_path") or "./workspaces/mix_progress.json"))
    results_path = resolve_path(config_dir, str(payload.get("results_path") or "./Mix_result.json"))
    videomme_official_output_path = resolve_path(
        config_dir,
        str(payload.get("videomme_official_output_path") or "./VideoMME_official_output.json"),
    )
    shared_preprocess_root = resolve_path(
        config_dir,
        str(payload.get("shared_preprocess_root") or "./workspaces/_shared_cache"),
    )
    pipeline_options = _load_pipeline_options(config_dir, payload.get("pipeline_options"))
    try:
        asr_beam_size = max(1, min(5, int(
            pipeline_options.get("asr_beam_size", payload.get("asr_beam_size", 5))
        )))
    except (TypeError, ValueError):
        asr_beam_size = 5
    try:
        asr_best_of = max(1, min(5, int(
            pipeline_options.get("asr_best_of", payload.get("asr_best_of", 5))
        )))
    except (TypeError, ValueError):
        asr_best_of = 5
    asr_condition_on_previous_text = bool(
        pipeline_options.get(
            "asr_condition_on_previous_text",
            payload.get("asr_condition_on_previous_text", True),
        )
    )
    asr_vad_filter = bool(
        pipeline_options.get("asr_vad_filter", payload.get("asr_vad_filter", False))
    )
    try:
        caption_batch_size = max(1, min(32, int(
            pipeline_options.get("caption_batch_size", payload.get("caption_batch_size", 1))
        )))
    except (TypeError, ValueError):
        caption_batch_size = 1
    try:
        caption_max_frames = max(1, min(8, int(
            pipeline_options.get("caption_max_frames", payload.get("caption_max_frames", 8))
        )))
    except (TypeError, ValueError):
        caption_max_frames = 8
    try:
        caption_max_new_tokens = max(32, min(512, int(
            pipeline_options.get("caption_max_new_tokens", payload.get("caption_max_new_tokens", 256))
        )))
    except (TypeError, ValueError):
        caption_max_new_tokens = 256
    sample_worker_admission_capacity = _bounded_config_int(
        payload,
        "sample_worker_admission_capacity",
        "TOOLSTAR_SAMPLE_WORKER_ADMISSION_CAPACITY",
        default=2,
        minimum=1,
        maximum=8,
    )
    resident_model_workers_per_device = _bounded_config_int(
        payload,
        "resident_model_workers_per_device",
        "TOOLSTAR_RESIDENT_MODEL_WORKERS_PER_DEVICE",
        default=1,
        minimum=1,
        maximum=8,
    )
    accelerator_queue_memory_guard = _env_bool(
        "TOOLSTAR_ACCELERATOR_QUEUE_MEMORY_GUARD",
        bool(payload.get("accelerator_queue_memory_guard", True)),
    )
    accelerator_queue_min_free_gib = _bounded_config_float(
        payload,
        "accelerator_queue_min_free_gib",
        "TOOLSTAR_ACCELERATOR_QUEUE_MIN_FREE_GIB",
        default=4.0,
        minimum=0.0,
        maximum=80.0,
    )
    accelerator_queue_reserve_gib = _bounded_config_float(
        payload,
        "accelerator_queue_reserve_gib",
        "TOOLSTAR_ACCELERATOR_QUEUE_RESERVE_GIB",
        default=6.0,
        minimum=0.0,
        maximum=40.0,
    )
    persistent_worker_start_stagger_seconds = _bounded_config_float(
        payload,
        "persistent_worker_start_stagger_seconds",
        "TOOLSTAR_PERSISTENT_WORKER_START_STAGGER_SECONDS",
        default=60.0,
        minimum=0.0,
        maximum=600.0,
    )
    return BatchConfig(
        config_path=config_path,
        workspaces_root=_env_path_override(config_dir, "TOOLSTAR_WORKSPACES_ROOT_OVERRIDE", workspaces_root),
        progress_path=_env_path_override(config_dir, "TOOLSTAR_PROGRESS_PATH_OVERRIDE", progress_path),
        results_path=_env_path_override(config_dir, "TOOLSTAR_RESULTS_PATH_OVERRIDE", results_path),
        videomme_official_output_path=_env_path_override(
            config_dir,
            "TOOLSTAR_VIDEOMME_OUTPUT_PATH_OVERRIDE",
            videomme_official_output_path,
        ),
        shared_preprocess_root=_env_path_override(
            config_dir,
            "TOOLSTAR_SHARED_PREPROCESS_ROOT_OVERRIDE",
            shared_preprocess_root,
        ),
        ffmpeg_binary=_resolve_ffmpeg_binary(str(payload.get("ffmpeg_binary") or "ffmpeg")),
        parallel_samples=_parse_parallel_samples_value(payload.get("parallel_samples")),
        # Same-device parallelism is still opt-in.  The environment switch is
        # useful for a measured capacity experiment without mutating the
        # canonical config; a queue capacity above one is itself that opt-in.
        allow_same_device_parallel=(
            bool(payload.get("allow_same_device_parallel", False))
            or _env_bool("TOOLSTAR_ALLOW_SAME_DEVICE_PARALLEL", False)
            or _queue_capacity_from_env() > 1
        ),
        accelerator_queue_enabled=bool(payload.get("accelerator_queue_enabled", True)),
        accelerator_queue_root=(
            _env_path_override(
                config_dir,
                "TOOLSTAR_BATCH_ACCELERATOR_QUEUE_ROOT",
                resolve_path(config_dir, str(payload.get("accelerator_queue_root"))),
            )
            if payload.get("accelerator_queue_root") not in (None, "")
            else None
        ),
        sample_worker_admission_capacity=sample_worker_admission_capacity,
        resident_model_workers_per_device=resident_model_workers_per_device,
        accelerator_queue_memory_guard=accelerator_queue_memory_guard,
        accelerator_queue_min_free_gib=accelerator_queue_min_free_gib,
        accelerator_queue_reserve_gib=accelerator_queue_reserve_gib,
        persistent_worker_start_stagger_seconds=persistent_worker_start_stagger_seconds,
        reuse_worker_samples=max(1, int(payload.get("reuse_worker_samples") or 1)),
        asr_beam_size=asr_beam_size,
        asr_best_of=asr_best_of,
        asr_condition_on_previous_text=asr_condition_on_previous_text,
        asr_vad_filter=asr_vad_filter,
        caption_batch_size=caption_batch_size,
        caption_max_frames=caption_max_frames,
        caption_max_new_tokens=caption_max_new_tokens,
        datasets=normalized_datasets,
        dataset_order=_parse_dataset_order(payload.get("dataset_order"), normalized_datasets),
        pipeline_options=pipeline_options,
    )


def _retool_video_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate_dirs = (
        repo_root / "RL",
        repo_root / "Retool_Video_RL",
    )
    for candidate_dir in candidate_dirs:
        if (candidate_dir / "run_e2e.py").exists():
            return candidate_dir
    raise FileNotFoundError(
        "Unable to locate run_e2e.py. Checked: "
        + ", ".join(str(candidate_dir / "run_e2e.py") for candidate_dir in candidate_dirs)
    )


def _sync_online_runtime_snapshot() -> Dict[str, Any]:
    """Resolve the active Evolver snapshot at a sample boundary.

    A persistent worker must not capture the active policy once at process
    startup: the scheduler can publish a replay-gated snapshot while the
    worker is processing an earlier sample. Schema and verifier policies are
    read by ``run_e2e`` for each query. Source edits additionally request a
    clean worker restart because Python modules are process scoped.
    """
    global _RL_ROOT, _ONLINE_RUNTIME_SNAPSHOT_ID, _ONLINE_RUNTIME_SOURCE_ROOT
    if str(os.environ.get("CFV_ONLINE_EVOLUTION_REPLAY") or "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return {
            "snapshot_id": str(os.environ.get("CFV_REGISTRY_SNAPSHOT_PATH") or "") or "snapshot_000",
            "source_root": str(_RL_ROOT),
            "restart_required": False,
        }
    active_raw = str(os.environ.get("CFV_ONLINE_ACTIVE_SNAPSHOT_PATH") or "").strip()
    base_root = (_PROJECT_ROOT / "RL").resolve()
    target_snapshot_id = "snapshot_000"
    target_source_root = base_root
    active_path: Optional[Path] = None
    if active_raw:
        candidate_path = Path(active_raw).expanduser().resolve()
        if candidate_path.is_file():
            try:
                snapshot_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Online Evolver active snapshot is unreadable: {candidate_path}"
                ) from exc
            if not isinstance(snapshot_payload, dict):
                raise RuntimeError(f"Online Evolver active snapshot must be an object: {candidate_path}")
            snapshot_status = str(snapshot_payload.get("status") or "").strip().lower()
            if snapshot_status != "accepted":
                raise RuntimeError(
                    "Online Evolver active snapshot has an invalid status: "
                    f"{snapshot_status!r} path={candidate_path}"
                )
            target_snapshot_id = str(snapshot_payload.get("snapshot_id") or "").strip() or "snapshot_000"
            # Bind this entire sample/ablation bundle to the bytes read at
            # its boundary. The controller may publish another active snapshot
            # while the model is still processing this bundle.
            frozen_bytes = json.dumps(snapshot_payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
            frozen_root = candidate_path.parent / 'worker_snapshots'
            frozen_root.mkdir(parents=True, exist_ok=True)
            active_path = frozen_root / (hashlib.sha256(frozen_bytes).hexdigest() + '.json')
            if not active_path.exists():
                with tempfile.NamedTemporaryFile(dir=frozen_root, delete=False) as handle:
                    handle.write(frozen_bytes)
                    temporary_path = Path(handle.name)
                os.replace(temporary_path, active_path)
            source_overrides = snapshot_payload.get("source_overrides")
            if source_overrides:
                if materialize_source_snapshot is None:
                    raise RuntimeError("Online source mutation support is unavailable in this worker")
                runtime_root_raw = str(os.environ.get("CFV_ONLINE_RUNTIME_ROOT") or "").strip()
                runtime_root = (
                    Path(runtime_root_raw).expanduser().resolve()
                    if runtime_root_raw
                    else candidate_path.parent / "runtime"
                )
                target_source_root = runtime_root / target_snapshot_id
                if not (target_source_root / "source_snapshot.json").is_file():
                    if target_source_root.exists():
                        shutil.rmtree(target_source_root)
                    target_source_root.parent.mkdir(parents=True, exist_ok=True)
                    materialize_source_snapshot(
                        snapshot_payload,
                        base_root,
                        target_source_root,
                    )
    if target_source_root.resolve() != _RL_ROOT.resolve() and _RUN_E2E_MODULE is None:
        os.environ["CFV_RUNTIME_SOURCE_ROOT"] = str(target_source_root.resolve())
        if active_path is not None:
            os.environ["CFV_REGISTRY_SNAPSHOT_PATH"] = str(active_path)
        os.execv(sys.executable, [sys.executable, *sys.argv])
    previous_source_root = Path(_ONLINE_RUNTIME_SOURCE_ROOT).resolve() if _ONLINE_RUNTIME_SOURCE_ROOT else None
    restart_required = bool(
        _RUN_E2E_MODULE is not None
        and previous_source_root is not None
        and previous_source_root != target_source_root.resolve()
    )
    if _RUN_E2E_MODULE is None:
        _RL_ROOT = target_source_root.resolve()
        if str(_RL_ROOT) in sys.path:
            sys.path.remove(str(_RL_ROOT))
        sys.path.insert(0, str(_RL_ROOT))
    if target_source_root.resolve() != base_root:
        os.environ["CFV_RUNTIME_SOURCE_ROOT"] = str(target_source_root.resolve())
    else:
        os.environ.pop("CFV_RUNTIME_SOURCE_ROOT", None)
    if active_path is not None:
        os.environ["CFV_REGISTRY_SNAPSHOT_PATH"] = str(active_path)
    else:
        os.environ.pop("CFV_REGISTRY_SNAPSHOT_PATH", None)
    _ONLINE_RUNTIME_SNAPSHOT_ID = target_snapshot_id
    _ONLINE_RUNTIME_SOURCE_ROOT = str(target_source_root.resolve())
    return {
        "snapshot_id": target_snapshot_id,
        "source_root": str(target_source_root.resolve()),
        "restart_required": restart_required,
    }


def _load_run_e2e_module():
    global _RUN_E2E_MODULE
    if _RUN_E2E_MODULE is not None:
        return _RUN_E2E_MODULE
    project_root = _RL_ROOT
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    _RUN_E2E_MODULE = importlib.import_module("run_e2e")
    return _RUN_E2E_MODULE


def load_inventories(config: BatchConfig) -> List[DatasetInventory]:
    mix_dir = config.config_path.parent
    inventories: List[DatasetInventory] = []
    from extended_datasets import load_egoschema_inventory, load_lvbench_inventory

    for dataset_name in config.dataset_order:
        dataset_config = config.datasets.get(dataset_name, {})
        if not dataset_config or not bool(dataset_config.get("enabled", True)):
            continue
        if dataset_name == "lvbench":
            inventories.append(
                load_lvbench_inventory(
                    mix_dir=mix_dir,
                    dataset_config=dataset_config,
                    workspaces_root=config.workspaces_root,
                )
            )
        elif dataset_name == "egoschema":
            inventories.append(
                load_egoschema_inventory(
                    mix_dir=mix_dir,
                    dataset_config=dataset_config,
                    workspaces_root=config.workspaces_root,
                )
            )
        elif dataset_name == "mvbench":
            inventories.append(
                load_mvbench_inventory(
                    mix_dir=mix_dir,
                    dataset_config=dataset_config,
                    workspaces_root=config.workspaces_root,
                )
            )
        elif dataset_name == "mlvu":
            inventories.append(
                load_mlvu_inventory(
                    mix_dir=mix_dir,
                    dataset_config=dataset_config,
                    workspaces_root=config.workspaces_root,
                )
            )
        elif dataset_name == "videomme":
            inventories.append(
                load_videomme_inventory(
                    mix_dir=mix_dir,
                    dataset_config=dataset_config,
                    workspaces_root=config.workspaces_root,
                )
            )
    return inventories


def _iter_all_samples(inventories: Sequence[DatasetInventory]) -> Iterable[MixSample]:
    for inventory in inventories:
        for subset_name in inventory.ordered_subset_names:
            for sample in inventory.samples_by_subset.get(subset_name, ()):
                yield sample


def import_skip_true_state(
    *,
    progress_payload: MutableMapping[str, Any],
    inventories: Sequence[DatasetInventory],
) -> int:
    imported = 0
    for sample in _iter_all_samples(inventories):
        if has_participated(progress_payload, sample.sample_uid):
            continue
        imported_result = import_existing_correct_result(
            progress_payload,
            sample_uid=sample.sample_uid,
            workspace_root=sample.workspace_root,
        )
        if imported_result is not None:
            imported += 1
    return imported


def _build_inventory_pending_sequence(
    *,
    inventory: DatasetInventory,
    progress_payload: Mapping[str, Any],
) -> List[MixSample]:
    pending_by_subset: Dict[str, Deque[MixSample]] = {}
    pending_totals: Dict[str, int] = {}
    emitted_counts: Dict[str, int] = {}

    for subset_name in inventory.ordered_subset_names:
        subset_pending = deque(
            sample
            for sample in inventory.samples_by_subset.get(subset_name, ())
            if not has_participated(progress_payload, sample.sample_uid)
        )
        if not subset_pending:
            continue
        pending_by_subset[subset_name] = subset_pending
        pending_totals[subset_name] = len(subset_pending)
        emitted_counts[subset_name] = 0

    pending: List[MixSample] = []
    ordered_subset_names = [
        subset_name
        for subset_name in inventory.ordered_subset_names
        if subset_name in pending_by_subset
    ]

    while True:
        selected_subset_name = ""
        selected_score = float("inf")
        for subset_name in ordered_subset_names:
            subset_pending = pending_by_subset.get(subset_name)
            if not subset_pending:
                continue
            subset_total = pending_totals[subset_name]
            score = float(emitted_counts[subset_name] + 1) / float(subset_total)
            if score < selected_score:
                selected_subset_name = subset_name
                selected_score = score
        if not selected_subset_name:
            break
        pending.append(pending_by_subset[selected_subset_name].popleft())
        emitted_counts[selected_subset_name] += 1

    return pending


def build_execution_queue(
    *,
    inventories: Sequence[DatasetInventory],
    progress_payload: Mapping[str, Any],
    max_samples: Optional[int],
) -> List[MixSample]:
    pending: List[MixSample] = []
    pending_by_dataset: Dict[str, Deque[MixSample]] = {}
    ordered_dataset_names: List[str] = []

    for inventory in inventories:
        dataset_name = str(inventory.dataset_name).strip()
        dataset_pending = deque(
            _build_inventory_pending_sequence(
                inventory=inventory,
                progress_payload=progress_payload,
            )
        )
        if not dataset_name or not dataset_pending:
            continue
        pending_by_dataset[dataset_name] = dataset_pending
        ordered_dataset_names.append(dataset_name)

    while True:
        advanced = False
        for dataset_name in ordered_dataset_names:
            dataset_pending = pending_by_dataset.get(dataset_name)
            if not dataset_pending:
                continue
            sample = dataset_pending.popleft()
            pending.append(sample)
            advanced = True
            if max_samples is not None and max_samples > 0 and len(pending) >= max_samples:
                return pending
        if not advanced:
            break
    return pending


def build_explicit_sample_queue(
    *,
    inventories: Sequence[DatasetInventory],
    sample_uids: Sequence[str],
) -> List[MixSample]:
    selected: List[MixSample] = []
    seen: Set[str] = set()
    for raw_uid in sample_uids:
        sample_uid = str(raw_uid or "").strip()
        if not sample_uid or sample_uid in seen:
            continue
        selected.append(_find_sample_by_uid(inventories, sample_uid))
        seen.add(sample_uid)
    return selected


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def extract_videomme_option(response: Any) -> str:
    text = str(response or "").strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for prefix in answer_prefixes:
        text = text.replace(prefix, "")
    if len(text.split()) > 10 and not any(option in text for option in "ABCD"):
        return ""
    match = re.search(r"[ABCD]", text)
    return match.group(0) if match else ""


def _is_evidence_unanswerable_answer(value: Any) -> bool:
    return "error: evidence_unanswerable" in str(value or "").strip().lower()


def _build_metrics_from_reward_helper(sample: MixSample, run_result: Mapping[str, Any]) -> Dict[str, Any]:
    trajectory = (
        run_result.get("agent", {}).get("trajectory", {})
        if isinstance(run_result.get("agent"), dict)
        else {}
    )
    final_answer = (
        run_result.get("agent", {}).get("final_answer", "")
        if isinstance(run_result.get("agent"), dict)
        else ""
    )
    features = summarize_mix_reward_features(
        run_result=run_result,
        trajectory=trajectory,
        final_answer=final_answer,
        ground_truth=sample.ground_truth,
        options=sample.options,
        dataset_name=sample.dataset_name,
    )
    tool_use_num = int(features.get("monitor_runtime_tool_count", 0) or 0)
    successful_tool_calls = int(features.get("monitor_successful_tool_calls", 0) or 0)
    planner_iterations = int(features.get("monitor_planner_iterations", 0) or 0)
    subagent_iterations = int(features.get("monitor_subagent_iterations", 0) or 0)
    return {
        "answer_correct": bool(features.get("answer_correct", False)),
        "predicted_label": str(features.get("predicted_label") or ""),
        "ground_truth_labels": list(features.get("ground_truth_labels") or []),
        "tool_use_num": tool_use_num,
        "iter_num": planner_iterations + subagent_iterations,
        "tool_success_rate": round(_safe_divide(successful_tool_calls, tool_use_num), 6),
        "successful_tool_calls": successful_tool_calls,
        "planner_iterations": planner_iterations,
        "subagent_iterations": subagent_iterations,
        "reward_features": features,
    }


def _cleanup_workspace(workspace_root: Path) -> None:
    stale_paths = [
        workspace_root / "queries",
        workspace_root / SAMPLE_RESULT_FILENAME,
        workspace_root / "results" / SAMPLE_RESULT_FILENAME,
        workspace_root / "results" / "mix_sample_result.json",
        workspace_root / "manifest.json",
        workspace_root / "request.json",
        workspace_root / "latest_run.json",
        workspace_root / "query_index.json",
    ]
    for path in stale_paths:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except Exception:
            pass


def _prepared_video_path(sample: MixSample) -> Path:
    return (sample.workspace_root / "inputs" / f"{sample.sample_uid.replace(':', '_')}_clip.mp4").resolve()


def prepare_sample_video(sample: MixSample, ffmpeg_binary: str, launcher_log_path: Path) -> Path:
    bound = sample.official_meta.get("bound")
    if sample.dataset_name != "mvbench" or not bound:
        return sample.video_path.resolve()
    if not isinstance(bound, (list, tuple)) or len(bound) != 2:
        return sample.video_path.resolve()
    start, end = float(bound[0]), float(bound[1])
    duration = max(0.0, end - start)
    prepared_path = _prepared_video_path(sample)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(sample.video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        str(prepared_path),
    ]
    with open(launcher_log_path, "a", encoding="utf-8") as handle:
        handle.write(f"[Launcher] clip_command={' '.join(command)}\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=handle)
    return prepared_path


@contextmanager
def capture_sample_logs(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(open(log_path, "a", encoding="utf-8"))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(handler)
    try:
        with open(log_path, "a", encoding="utf-8") as log_handle:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                yield log_handle
    finally:
        root_logger.removeHandler(handler)
        handler.flush()
        handler.close()


def _write_sample_result(result_path: Path, payload: Mapping[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _extract_worker_result_payload(stdout_text: str) -> Optional[Dict[str, Any]]:
    for raw_line in reversed(str(stdout_text or "").splitlines()):
        marker_index = raw_line.find(RUN_RESULT_LOG_MARKER)
        if marker_index < 0:
            continue
        payload_text = raw_line[marker_index + len(RUN_RESULT_LOG_MARKER):].strip()
        if not payload_text:
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


_WORKER_OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cuda error: out of memory",
    "torch.outofmemoryerror",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "cuda error (2)",
    "memory allocation",
    "cannot allocate memory",
    "not enough memory",
    "no available memory",
)


def _contains_worker_oom_marker(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and any(marker in text for marker in _WORKER_OOM_MARKERS)


def _payload_is_worker_oom(payload: Mapping[str, Any]) -> bool:
    for key in ("error", "raw_final_answer", "worker_error", "exception"):
        if _contains_worker_oom_marker(payload.get(key)):
            return True
    return False


def _contains_online_restart_marker(value: Any) -> bool:
    return "[ONLINE_RUNTIME_RESTART]" in str(value or "")


def _worker_oom_retry_limit() -> int:
    raw_value = str(os.environ.get("TOOLSTAR_WORKER_OOM_RETRIES") or "2").strip()
    try:
        return max(0, min(5, int(raw_value)))
    except (TypeError, ValueError):
        return 2


def _worker_oom_retry_delay_seconds() -> float:
    raw_value = str(
        os.environ.get("TOOLSTAR_WORKER_OOM_RETRY_DELAY_SECONDS") or "30"
    ).strip()
    try:
        return max(0.0, min(600.0, float(raw_value)))
    except (TypeError, ValueError):
        return 30.0


def _visible_cuda_device_ids() -> List[str]:
    raw_visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if raw_visible:
        return [item.strip() for item in raw_visible.split(",") if item.strip()]
    try:
        import torch

        if torch.cuda.is_available():
            return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []
    return []


def _resolve_effective_parallel_samples(
    *,
    configured_parallel_samples: Optional[int],
    cli_parallel_samples: Optional[int],
    selected_count: int,
    visible_device_ids: Sequence[str],
    allow_same_device_parallel: bool = False,
) -> int:
    if selected_count <= 1:
        return 1
    max_parallel = len(list(visible_device_ids)) if visible_device_ids else 1
    requested = cli_parallel_samples if cli_parallel_samples is not None else configured_parallel_samples
    if allow_same_device_parallel and requested is not None:
        # A single visible GPU can host more than one isolated reusable worker
        # when the batch explicitly opts in.  This is deliberately opt-in:
        # the default remains one worker per device for memory safety.
        max_parallel = max(max_parallel, int(requested))
    if requested is None:
        return max(1, min(selected_count, max_parallel))
    if requested <= 0:
        return max(1, min(selected_count, max_parallel))
    return max(1, min(selected_count, requested, max_parallel))


def _resolve_reuse_worker_samples(
    *,
    configured_size: int,
    cli_size: Optional[int],
    selected_count: int,
) -> int:
    requested = cli_size if cli_size is not None else configured_size
    try:
        normalized = int(requested)
    except (TypeError, ValueError):
        normalized = 1
    return max(1, min(max(1, selected_count), normalized))


def _sample_cache_key(video_path: Path) -> str:
    return str(Path(video_path).stem or "").strip().lower()


def _split_pending_samples_by_dataset(
    selected_samples: Sequence[MixSample],
    dataset_order: Sequence[str],
) -> Tuple[List[str], Dict[str, List[MixSample]]]:
    pending_by_dataset: Dict[str, List[MixSample]] = {}
    seen_names: Set[str] = set()
    ordered_names: List[str] = []

    for dataset_name in dataset_order:
        normalized_name = str(dataset_name or "").strip()
        if not normalized_name or normalized_name in seen_names:
            continue
        ordered_names.append(normalized_name)
        seen_names.add(normalized_name)
        pending_by_dataset[normalized_name] = []

    for sample in selected_samples:
        dataset_name = str(sample.dataset_name or "").strip()
        if not dataset_name:
            continue
        if dataset_name not in pending_by_dataset:
            pending_by_dataset[dataset_name] = []
            if dataset_name not in seen_names:
                ordered_names.append(dataset_name)
                seen_names.add(dataset_name)
        pending_by_dataset[dataset_name].append(sample)

    ordered_names = [name for name in ordered_names if pending_by_dataset.get(name)]
    return ordered_names, pending_by_dataset


def _pop_launchable_sample_for_dataset(
    pending_samples: List[MixSample],
    active_cache_keys: Set[str],
) -> Optional[MixSample]:
    for index, sample in enumerate(pending_samples):
        cache_key = _sample_cache_key(sample.video_path)
        if cache_key in active_cache_keys:
            continue
        return pending_samples.pop(index)
    return None


def _find_sample_by_uid(inventories: Sequence[DatasetInventory], sample_uid: str) -> MixSample:
    target_uid = str(sample_uid or "").strip()
    for sample in _iter_all_samples(inventories):
        if sample.sample_uid == target_uid:
            return sample
    raise KeyError(f"Unable to resolve Mix sample_uid={target_uid!r} from current inventories.")


def _json_safe_worker_value(value: Any) -> Any:
    """Convert a MixSample field to a portable worker-spec JSON value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_worker_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_worker_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sample_to_worker_spec(sample: MixSample) -> Dict[str, Any]:
    return {
        "dataset_name": sample.dataset_name,
        "subset_name": sample.subset_name,
        "sample_uid": sample.sample_uid,
        "sample_index": int(sample.sample_index),
        "video_path": str(sample.video_path),
        "workspace_root": str(sample.workspace_root),
        "prompt": sample.prompt,
        "question_block": sample.question_block,
        "options": list(sample.options),
        "ground_truth": sample.ground_truth,
        "ground_truth_option": sample.ground_truth_option,
        "official_meta": _json_safe_worker_value(sample.official_meta),
    }


def _sample_from_worker_spec(payload: Mapping[str, Any]) -> MixSample:
    return MixSample(
        dataset_name=str(payload.get("dataset_name") or ""),
        subset_name=str(payload.get("subset_name") or ""),
        sample_uid=str(payload.get("sample_uid") or ""),
        sample_index=int(payload.get("sample_index") or 0),
        video_path=Path(str(payload.get("video_path") or "")).resolve(),
        workspace_root=Path(str(payload.get("workspace_root") or "")).resolve(),
        prompt=str(payload.get("prompt") or ""),
        question_block=str(payload.get("question_block") or ""),
        options=tuple(str(item) for item in (payload.get("options") or [])),
        ground_truth=str(payload.get("ground_truth") or ""),
        ground_truth_option=str(payload.get("ground_truth_option") or ""),
        official_meta=dict(payload.get("official_meta") or {}),
    )


def _write_worker_sample_spec(path: Path, samples: Sequence[MixSample]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "mix_worker_sample_spec",
        "layout_version": 1,
        "samples": [_sample_to_worker_spec(sample) for sample in samples],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_worker_sample_spec(path: Path) -> List[MixSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("kind") != "mix_worker_sample_spec":
        raise RuntimeError(f"Invalid Mix worker sample spec: {path}")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise RuntimeError(f"Mix worker sample spec has no samples: {path}")
    samples = [_sample_from_worker_spec(item) for item in raw_samples if isinstance(item, Mapping)]
    if len(samples) != len(raw_samples):
        raise RuntimeError(f"Mix worker sample spec contains a malformed sample: {path}")
    return samples


def _bind_worker_to_visible_cuda_device(visible_device_id: str) -> None:
    normalized_device = str(visible_device_id or "").strip()
    if not normalized_device:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized_device
    for key in list(os.environ.keys()):
        if key.startswith("TOOLSTAR_") and key.endswith("_DEVICE_ID"):
            os.environ.pop(key, None)
    os.environ["TOOLSTAR_DEVICE_ID"] = "0"


def _build_worker_env(
    config_path: Path,
    *,
    caption_batch_size: Optional[int] = None,
    caption_max_frames: Optional[int] = None,
    caption_max_new_tokens: Optional[int] = None,
) -> Dict[str, str]:
    """Build the formal worker environment for both serial and grouped paths."""
    worker_env = os.environ.copy()
    # A worker launched directly by the scheduler must resolve the same local
    # checkpoints as a manually prepared evaluation shell. Without this, the
    # settings fallback is relative to the extracted project and points at a
    # nonexistent ``/gemini/code/model`` directory.
    if not str(worker_env.get("MODEL_BASE_PATH") or "").strip():
        default_model_root = Path("/gemini/pretrain/model")
        if default_model_root.is_dir():
            worker_env["MODEL_BASE_PATH"] = str(default_model_root)
    resolved_ffmpeg = _resolve_ffmpeg_binary(worker_env.get("FFMPEG_BINARY"))
    worker_env["FFMPEG_BINARY"] = resolved_ffmpeg
    worker_env["IMAGEIO_FFMPEG_EXE"] = resolved_ffmpeg
    # Formal M3 keeps the original Video_Clip_QA implementation, but serializes
    # its GPU generations inside one reusable worker. This prevents a planner
    # Parallel batch from creating several Qwen video-generation working sets
    # at once; the planner trajectory still records the parallel batch.
    worker_env["TOOLSTAR_SERIALIZE_GPU_VIDEO_CLIP_QA"] = "1"
    worker_env.pop("TOOLSTAR_DISABLE_GPU_VIDEO_CLIP_QA", None)
    # Keep the original Video_Clip_QA tool and provider, but bound the visual
    # payload to its configured sample count before qwen_vl sees long clips.
    worker_env["TOOLSTAR_VIDEO_CLIP_QA_FORCE_FRAME_PAYLOAD"] = "1"
    # Allow a scheduling experiment to release the caption model before the
    # planner model is admitted in the same per-sample worker. Reusable
    # workers keep the historical default; dynamic workers can opt out via
    # the parent environment to avoid two Qwen working sets in one process.
    worker_env["TOOLSTAR_PERSIST_VISUAL_RUNTIME"] = str(
        worker_env.get("TOOLSTAR_PERSIST_VISUAL_RUNTIME") or "1"
    )
    worker_env["TOOLSTAR_PERSIST_EMBEDDING_RUNTIME"] = str(
        worker_env.get("TOOLSTAR_PERSIST_EMBEDDING_RUNTIME") or "1"
    )
    worker_env["TOOLSTAR_PERSIST_ASR_RUNTIME"] = str(
        worker_env.get("TOOLSTAR_PERSIST_ASR_RUNTIME") or "1"
    )
    # The Orion CUDA proxy has a known-risk fused Qwen3.5/cuDNN path. Keep the
    # stable Transformers fallback as the default, but allow an explicit
    # caller override so isolated acceleration probes and formally requested
    # runtime experiments can actually reach every worker.
    worker_env["TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS"] = str(
        worker_env.get("TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS") or "full"
    )
    worker_env["TOOLSTAR_DISABLE_CUDNN"] = str(
        worker_env.get("TOOLSTAR_DISABLE_CUDNN") or "1"
    )
    if "evidence" in config_path.stem.lower():
        # Evidence runs must build a real dense index and retain the text-only
        # optimization; frame paths remain attached auditable evidence.
        worker_env["TOOLSTAR_DISABLE_LOCAL_VECTOR_INDEX"] = "0"
        worker_env.setdefault("TOOLSTAR_SEMANTIC_INDEX_VIDEO", "0")
    # Caption batching is a formal, bounded optimization.  Resolve it from
    # the Mix config before the isolated worker starts; the parent environment
    # must not silently replace the value recorded in the batch result.
    if caption_batch_size is None or caption_max_frames is None or caption_max_new_tokens is None:
        raw_config: Dict[str, Any] = {}
        try:
            raw_config = _load_json_object(Path(config_path).resolve())
            raw_options = raw_config.get("pipeline_options")
            if not isinstance(raw_options, dict):
                raw_options = {}
        except Exception:
            raw_options = {}
        if caption_batch_size is None:
            try:
                caption_batch_size = max(1, min(32, int(
                    raw_options.get("caption_batch_size", raw_config.get("caption_batch_size", 1))
                )))
            except (TypeError, ValueError):
                caption_batch_size = 1
        if caption_max_frames is None:
            try:
                caption_max_frames = max(1, min(8, int(
                    raw_options.get("caption_max_frames", raw_config.get("caption_max_frames", 8))
                )))
            except (TypeError, ValueError):
                caption_max_frames = 8
        if caption_max_new_tokens is None:
            try:
                caption_max_new_tokens = max(32, min(512, int(
                    raw_options.get("caption_max_new_tokens", raw_config.get("caption_max_new_tokens", 256))
                )))
            except (TypeError, ValueError):
                caption_max_new_tokens = 256
    raw_config: Dict[str, Any] = {}
    try:
        raw_config = _load_json_object(Path(config_path).resolve())
        raw_options = raw_config.get("pipeline_options")
        if not isinstance(raw_options, dict):
            raw_options = {}
    except Exception:
        raw_options = {}
    configured_share_visual = raw_options.get(
        "share_planner_visual_runtime",
        raw_config.get("share_planner_visual_runtime", True),
    )
    if isinstance(configured_share_visual, str):
        configured_share_visual = configured_share_visual.strip().lower() in {
            "1", "true", "yes", "on"
        }
    worker_env["TOOLSTAR_SHARE_PLANNER_VISUAL_RUNTIME"] = (
        "1" if bool(configured_share_visual) else "0"
    )
    provider_queue_capacity = _provider_queue_capacity_from_config(raw_config)
    # The outer sample-worker admission queue bounds resident model working
    # sets; the inner provider queue must expose the same explicit capacity so
    # admitted persistent workers can actually overlap GPU turns.
    worker_env["TOOLSTAR_ACCELERATOR_QUEUE_CAPACITY"] = str(provider_queue_capacity)
    worker_env["TOOLSTAR_BATCH_ACCELERATOR_QUEUE_CAPACITY"] = str(provider_queue_capacity)
    worker_env["TOOLSTAR_ACCELERATOR_GENERATION_QUEUE_CAPACITY"] = str(
        _generation_queue_capacity_from_config(raw_config)
    )
    worker_env.setdefault(
        "TOOLSTAR_ACCELERATOR_GENERATION_MEMORY_GUARD",
        "1" if bool(raw_config.get("accelerator_queue_memory_guard", True)) else "0",
    )
    worker_env.setdefault(
        "TOOLSTAR_ACCELERATOR_GENERATION_MIN_FREE_GIB",
        str(raw_config.get("generation_queue_min_free_gib", 1.0)),
    )
    worker_env.setdefault(
        "TOOLSTAR_ACCELERATOR_GENERATION_RESERVE_GIB",
        str(raw_config.get("generation_queue_reserve_gib", 1.0)),
    )
    try:
        asr_beam_size = max(1, min(5, int(
            raw_options.get("asr_beam_size", raw_config.get("asr_beam_size", 5))
        )))
    except (TypeError, ValueError):
        asr_beam_size = 5
    try:
        asr_best_of = max(1, min(5, int(
            raw_options.get("asr_best_of", raw_config.get("asr_best_of", 5))
        )))
    except (TypeError, ValueError):
        asr_best_of = 5
    asr_condition_on_previous_text = raw_options.get(
        "asr_condition_on_previous_text",
        raw_config.get("asr_condition_on_previous_text", True),
    )
    asr_vad_filter = raw_options.get(
        "asr_vad_filter",
        raw_config.get("asr_vad_filter", False),
    )
    if isinstance(asr_condition_on_previous_text, str):
        asr_condition_on_previous_text = asr_condition_on_previous_text.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(asr_vad_filter, str):
        asr_vad_filter = asr_vad_filter.strip().lower() in {"1", "true", "yes", "on"}
    configured_asr_device = str(
        raw_options.get("asr_device", raw_config.get("asr_device", ""))
        or ""
    ).strip()
    if configured_asr_device:
        worker_env.setdefault("TOOLSTAR_ASR_DEVICE", configured_asr_device)
    worker_env.setdefault(
        "TOOLSTAR_SAMPLE_WORKER_ADMISSION_CAPACITY",
        str(raw_config.get("sample_worker_admission_capacity", 2)),
    )
    worker_env.setdefault(
        "TOOLSTAR_ACCELERATOR_QUEUE_MEMORY_GUARD",
        "1" if bool(raw_config.get("accelerator_queue_memory_guard", True)) else "0",
    )
    worker_env.setdefault(
        "TOOLSTAR_ACCELERATOR_QUEUE_MIN_FREE_GIB",
        str(raw_config.get("accelerator_queue_min_free_gib", 4.0)),
    )
    worker_env.setdefault(
        "TOOLSTAR_ACCELERATOR_QUEUE_RESERVE_GIB",
        str(raw_config.get("accelerator_queue_reserve_gib", 6.0)),
    )
    worker_env.setdefault("TOOLSTAR_ASR_BEAM_SIZE", str(int(asr_beam_size)))
    worker_env.setdefault("TOOLSTAR_ASR_BEST_OF", str(int(asr_best_of)))
    worker_env.setdefault("TOOLSTAR_ASR_CONDITION_ON_PREVIOUS_TEXT", "1" if asr_condition_on_previous_text else "0")
    worker_env.setdefault("TOOLSTAR_ASR_VAD_FILTER", "1" if asr_vad_filter else "0")
    worker_env["TOOLSTAR_CAPTION_BATCH_SIZE"] = str(int(caption_batch_size or 1))
    worker_env["TOOLSTAR_CAPTION_MAX_FRAMES"] = str(int(caption_max_frames or 8))
    worker_env["TOOLSTAR_CAPTION_MAX_NEW_TOKENS"] = str(int(caption_max_new_tokens or 256))
    return worker_env


@contextmanager
def _hold_accelerator_queue_slot(
    *,
    config: BatchConfig,
    visible_device_id: str,
) -> Iterable[float]:
    """Bound same-device worker admission without duplicating model loads.

    The scheduler may have several CPU-side futures ready, but only one
    process owns a physical accelerator slot at a time.  This prevents two
    isolated workers from cold-loading Qwen concurrently and records the
    waiting time so a parallel run cannot be mistaken for GPU concurrency.
    """
    device_id = str(visible_device_id or "").strip()
    queue_enabled = bool(config.accelerator_queue_enabled) and bool(device_id)
    if not queue_enabled or SharedAcceleratorQueueLock is None:
        yield 0.0
        return

    queue_root = config.accelerator_queue_root
    if queue_root is None:
        raw_root = str(os.environ.get("TOOL_STAR_SHARED_ACCELERATOR_QUEUE_ROOT") or "").strip()
        queue_root = Path(raw_root).expanduser() if raw_root else None
    slot_key = (
        resolve_local_accelerator_slot_key(device=device_id)
        if resolve_local_accelerator_slot_key is not None
        else f"mix_gpu_{device_id}"
    )
    lock = SharedAcceleratorQueueLock(
        name="mix_sample_worker",
        queue_root=queue_root,
        slot_key=slot_key,
        enabled=True,
        log_wait_seconds=5.0,
    )
    wait_started = time.perf_counter()
    with lock:
        waited_seconds = max(0.0, time.perf_counter() - wait_started)
        if waited_seconds >= 0.25:
            LOGGER.info(
                "Accelerator queue admitted device=%s after %.3fs; pending worker stayed outside GPU.",
                device_id,
                waited_seconds,
            )
        yield waited_seconds


@contextmanager
def _hold_sample_worker_admission_slot(
    *,
    config: BatchConfig,
    visible_device_id: str,
) -> Iterable[float]:
    """Limit resident per-sample GPU workers while keeping a larger task pool.

    The runtime queue protects individual generation calls, but a child can
    keep its loaded model resident after that call. Dynamic sample workers
    therefore need a separate parent-owned admission queue so at most the
    configured number of child processes (and their model working sets) exist
    on one physical GPU at once.
    """
    device_id = str(visible_device_id or "").strip()
    queue_enabled = bool(config.accelerator_queue_enabled) and bool(device_id)
    if not queue_enabled or SharedAcceleratorQueueLock is None:
        yield 0.0
        return

    raw_capacity = os.environ.get("TOOLSTAR_SAMPLE_WORKER_ADMISSION_CAPACITY")
    if raw_capacity in (None, ""):
        capacity = int(config.sample_worker_admission_capacity)
    else:
        try:
            capacity = max(1, min(8, int(raw_capacity)))
        except (TypeError, ValueError):
            capacity = _queue_capacity_from_env()

    queue_root = config.accelerator_queue_root
    if queue_root is None:
        raw_root = str(os.environ.get("TOOL_STAR_SHARED_ACCELERATOR_QUEUE_ROOT") or "").strip()
        queue_root = Path(raw_root).expanduser() if raw_root else None
    base_slot_key = (
        resolve_local_accelerator_slot_key(device=device_id)
        if resolve_local_accelerator_slot_key is not None
        else f"mix_gpu_{device_id}"
    )
    lock = SharedAcceleratorQueueLock(
        name="mix_sample_worker_admission",
        queue_root=queue_root,
        slot_key=f"{base_slot_key}_sample_admission",
        enabled=True,
        log_wait_seconds=5.0,
        capacity=capacity,
        memory_guard_enabled=config.accelerator_queue_memory_guard,
        memory_min_free_mib=float(config.accelerator_queue_min_free_gib) * 1024.0,
        memory_reserve_mib=float(config.accelerator_queue_reserve_gib) * 1024.0,
    )
    wait_started = time.perf_counter()
    with lock:
        waited_seconds = max(0.0, time.perf_counter() - wait_started)
        if waited_seconds >= 0.25:
            LOGGER.info(
                "Sample-worker GPU admission device=%s capacity=%s waited=%.3fs; "
                "extra worker remains queued outside the GPU.",
                device_id,
                capacity,
                waited_seconds,
            )
        yield waited_seconds


def _prepare_tool_library_baseline(config: BatchConfig) -> Tuple[Path, Dict[str, Any]]:
    """Freeze the concrete registry before the first batch worker starts."""

    baseline_path = (config.workspaces_root / "_audit" / "tool_library_baseline.json").resolve()
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_path.exists():
        try:
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Existing tool library baseline is unreadable: {baseline_path}") from exc
        if not isinstance(payload, dict) or payload.get("kind") != "tool_library_baseline":
            raise RuntimeError(f"Existing tool library baseline is invalid: {baseline_path}")
        LOGGER.info(
            "Reusing immutable tool library baseline baseline_id=%s path=%s",
            payload.get("baseline_id"),
            baseline_path,
        )
    else:
        repo_root = Path(__file__).resolve().parents[2]
        baseline_script = repo_root / "RL" / "toolkit" / "baseline.py"
        if not baseline_script.exists():
            raise FileNotFoundError(f"Tool library baseline helper not found: {baseline_script}")
        worker_env = _build_worker_env(config.config_path)
        rl_root = str((repo_root / "RL").resolve())
        existing_pythonpath = str(worker_env.get("PYTHONPATH") or "").strip()
        worker_env["PYTHONPATH"] = (
            rl_root + os.pathsep + existing_pythonpath if existing_pythonpath else rl_root
        )
        command = [
            sys.executable or "python",
            str(baseline_script),
            "--output",
            str(baseline_path),
        ]
        LOGGER.info("Creating immutable tool library baseline: %s", baseline_path)
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=worker_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Tool library baseline creation failed "
                f"return_code={completed.returncode}: {completed.stderr[-4000:]}"
            )
        try:
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Tool library baseline was not written: {baseline_path}") from exc
        LOGGER.info(
            "Tool library baseline frozen baseline_id=%s tool_count=%s schema_digest=%s source_digest=%s",
            payload.get("baseline_id"),
            payload.get("tool_count"),
            payload.get("schema_digest"),
            payload.get("source_digest"),
        )
        if completed.stdout.strip():
            LOGGER.info("Tool library baseline helper: %s", completed.stdout.strip()[-2000:])
    baseline_id = str(payload.get("baseline_id") or "")
    if not baseline_id:
        raise RuntimeError(f"Tool library baseline has no baseline_id: {baseline_path}")
    os.environ["CFV_REGISTRY_BASELINE_SNAPSHOT_PATH"] = str(baseline_path)
    os.environ["CFV_REGISTRY_BASELINE_ID"] = baseline_id
    return baseline_path, payload


def _run_mix_sample_worker(
    *,
    config: BatchConfig,
    config_path: Path,
    sample: MixSample,
    no_cache: bool,
    visible_device_id: str,
    caption_batch_size: Optional[int] = None,
    caption_max_frames: Optional[int] = None,
    caption_max_new_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    sample_uid = str(sample.sample_uid)
    command = [
        sys.executable or "python",
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--worker-sample-uid",
        str(sample_uid),
    ]
    if no_cache:
        command.append("--no-cache")
    if str(visible_device_id or "").strip():
        command.extend(["--worker-device-id", str(visible_device_id)])

    # Do not retain a complete worker transcript in the scheduler's memory.
    # A worker can emit model-loading progress and long planner traces; keeping
    # that stream in PIPE made a long batch unnecessarily fragile and also hid
    # the final return code when a low-level CUDA process abort occurred.
    safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_uid).strip("._") or "sample"
    # A sample workspace is rooted at <workspaces_root>/<dataset>/<sample>.
    worker_log_root = sample.workspace_root.resolve().parents[1]
    worker_log_path = worker_log_root / "_worker_logs" / f"{safe_uid}.log"
    worker_spec_path = worker_log_root / "_worker_logs" / f"{safe_uid}.json"
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_worker_sample_spec(worker_spec_path, [sample])
    command.extend(["--worker-sample-spec-file", str(worker_spec_path)])
    worker_env = _build_worker_env(
        config_path,
        caption_batch_size=caption_batch_size,
        caption_max_frames=caption_max_frames,
        caption_max_new_tokens=caption_max_new_tokens,
    )
    # Queue ownership belongs to the isolated worker.  The worker's runtime
    # components acquire the shared device slot before loading/using CUDA;
    # acquiring it here as well would deadlock the child on the same lock.
    queue_wait_seconds = 0.0
    oom_retry_limit = _worker_oom_retry_limit()
    oom_retry_delay_seconds = _worker_oom_retry_delay_seconds()
    attempt_index = 0
    while True:
        log_mode = "w" if attempt_index == 0 else "a"
        with open(worker_log_path, log_mode, encoding="utf-8") as worker_log:
            worker_log.write(
                f"[Scheduler] worker_attempt={attempt_index + 1}/{oom_retry_limit + 1}\n"
            )
            worker_log.write(f"[Scheduler] worker_command={' '.join(command)}\n")
            worker_log.write("[Scheduler] queue_owner=isolated_worker\n")
            worker_log.write(
                "[Scheduler] caption_env="
                f"batch_size={worker_env.get('TOOLSTAR_CAPTION_BATCH_SIZE')} "
                f"max_frames={worker_env.get('TOOLSTAR_CAPTION_MAX_FRAMES')} "
                f"max_new_tokens={worker_env.get('TOOLSTAR_CAPTION_MAX_NEW_TOKENS')} "
                f"asr_device={worker_env.get('TOOLSTAR_ASR_DEVICE', 'settings-default')} "
                f"share_planner_visual_runtime={worker_env.get('TOOLSTAR_SHARE_PLANNER_VISUAL_RUNTIME')} "
                f"provider_queue_capacity={worker_env.get('TOOLSTAR_ACCELERATOR_QUEUE_CAPACITY')} "
                f"generation_queue_capacity={worker_env.get('TOOLSTAR_ACCELERATOR_GENERATION_QUEUE_CAPACITY')}\n"
            )
            worker_log.flush()
            with _hold_sample_worker_admission_slot(
                config=config,
                visible_device_id=visible_device_id,
            ) as queue_wait_seconds:
                worker_log.write(
                    "[Scheduler] sample_worker_admission_wait_seconds="
                    f"{queue_wait_seconds:.3f}\n"
                )
                worker_log.flush()
                process = subprocess.run(
                    command,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    env=worker_env,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        if (sample.workspace_root / ".paired_pending").exists():
            raise RuntimeError("paired execution incomplete; sample remains pending")
        attempt_payload_is_oom = False
        if int(process.returncode) == 0:
            try:
                result_path = sample.workspace_root.resolve() / SAMPLE_RESULT_FILENAME
                with open(result_path, "r", encoding="utf-8") as result_handle:
                    attempt_payload = json.load(result_handle)
                attempt_payload_is_oom = (
                    isinstance(attempt_payload, dict)
                    and _payload_is_worker_oom(attempt_payload)
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                attempt_payload_is_oom = False
            # A worker can catch CUDA OOM and exit cleanly after writing a
            # payload.  Treat that payload exactly like a process-level OOM so
            # it gets a fresh queued retry instead of being accepted here.
            if not attempt_payload_is_oom:
                break
        try:
            retry_log_text = worker_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            retry_log_text = ""
        if (
            not attempt_payload_is_oom
            and not _contains_worker_oom_marker(retry_log_text[-200000:])
        ) or attempt_index >= oom_retry_limit:
            break
        attempt_index += 1
        LOGGER.warning(
            "Retrying serial Mix worker after CUDA OOM sample_uid=%s attempt=%d/%d delay=%.1fs",
            sample_uid,
            attempt_index,
            oom_retry_limit,
            oom_retry_delay_seconds,
        )
        if oom_retry_delay_seconds > 0:
            time.sleep(oom_retry_delay_seconds)

    payload: Optional[Dict[str, Any]] = None
    try:
        result_path = sample.workspace_root.resolve() / SAMPLE_RESULT_FILENAME
        with open(result_path, "r", encoding="utf-8") as result_handle:
            loaded_payload = json.load(result_handle)
        if isinstance(loaded_payload, dict):
            payload = loaded_payload
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        payload = None

    if payload is not None:
        payload.setdefault(
            "scheduler_accelerator_queue_wait_seconds",
            round(float(queue_wait_seconds), 3),
        )

    worker_output = ""
    try:
        worker_output = worker_log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        worker_output = ""
    payload_is_oom = payload is not None and _payload_is_worker_oom(payload)
    if payload is not None and process.returncode == 0 and not payload_is_oom:
        return payload
    tail_lines = worker_output.splitlines()[-40:]
    tail = "\n".join(tail_lines).strip()
    if payload is not None:
        payload.setdefault("status", "failed")
        payload.setdefault("error", f"Worker process exited with return code {process.returncode}.")
        payload["worker_return_code"] = int(process.returncode)
        payload["worker_log"] = str(worker_log_path)
        if payload_is_oom:
            payload["status"] = "failed"
            payload["acc"] = None
            payload["predicted_label"] = ""
            payload["failure_class"] = "oom"
            payload["accuracy_eligible"] = False
            payload["worker_oom_retry_exhausted"] = True
            payload["worker_oom_retry_attempts"] = int(attempt_index + 1)
        return payload
    raise RuntimeError(
        f"Mix sample worker failed sample_uid={sample_uid} visible_device_id={visible_device_id} "
        f"return_code={process.returncode}. Worker log={worker_log_path}. Worker output tail:\n{tail}"
    )


def _read_sample_result_payload(sample: MixSample) -> Optional[Dict[str, Any]]:
    result_path = sample.workspace_root.resolve() / SAMPLE_RESULT_FILENAME
    if (sample.workspace_root / ".paired_pending").exists():
        return None
    try:
        with open(result_path, "r", encoding="utf-8") as result_handle:
            loaded_payload = json.load(result_handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(loaded_payload, dict):
        return None
    if str(loaded_payload.get("sample_uid") or "") != str(sample.sample_uid):
        return None
    return loaded_payload


def _run_mix_sample_group_worker(
    *,
    config: BatchConfig,
    config_path: Path,
    samples: Sequence[MixSample],
    no_cache: bool,
    visible_device_id: str,
    caption_batch_size: Optional[int] = None,
    caption_max_frames: Optional[int] = None,
    caption_max_new_tokens: Optional[int] = None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], int, Path]:
    """Run an ordered sample group in one isolated process.

    A process boundary still protects the scheduler from SIGABRT/SIGKILL, but
    the child retains the caption runtime cache between samples.  Every sample
    writes its own result file, so the parent can recover partial progress even
    when the child exits mid-group.  A CUDA OOM is the one bounded retryable
    worker failure: completed samples stay committed, while only the affected
    or unexecuted samples are delayed and relaunched in a fresh persistent
    worker.  Normal samples in an attempt still share one loaded model.
    """
    if not samples:
        return [], 0, Path()
    original_samples = list(samples)
    pending_samples = list(original_samples)
    final_results: Dict[str, Dict[str, Any]] = {}
    emitted_uids: Set[str] = set()
    max_oom_retries = _worker_oom_retry_limit()
    retry_delay_seconds = _worker_oom_retry_delay_seconds()
    attempt_index = 0
    online_restart_count = 0
    try:
        online_restart_limit = max(
            1,
            min(32, int(os.environ.get("TOOLSTAR_ONLINE_RUNTIME_RESTARTS", "8") or 8)),
        )
    except (TypeError, ValueError):
        online_restart_limit = 8
    last_return_code = 0
    last_worker_log_path = Path()
    worker_log_root = samples[0].workspace_root.resolve().parents[1]
    (worker_log_root / "_worker_logs").mkdir(parents=True, exist_ok=True)
    while pending_samples:
        attempt_index += 1
        safe_first_uid = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(pending_samples[0].sample_uid)
        ).strip("._") or "sample"
        safe_last_uid = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(pending_samples[-1].sample_uid)
        ).strip("._") or "sample"
        attempt_stem = f"group_{safe_first_uid}__{safe_last_uid}.attempt_{attempt_index}"
        worker_log_path = worker_log_root / "_worker_logs" / f"{attempt_stem}.log"
        worker_uid_path = worker_log_root / "_worker_logs" / f"{attempt_stem}.uids"
        worker_spec_path = worker_log_root / "_worker_logs" / f"{attempt_stem}.json"
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        _write_worker_sample_spec(worker_spec_path, pending_samples)
        with open(worker_uid_path, "w", encoding="utf-8") as worker_uid_file:
            for sample in pending_samples:
                worker_uid_file.write(f"{sample.sample_uid}\n")

        command = [
            sys.executable or "python",
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--worker-sample-uids-file",
            str(worker_uid_path),
        ]
        if no_cache:
            command.append("--no-cache")
        if str(visible_device_id or "").strip():
            command.extend(["--worker-device-id", str(visible_device_id)])
        command.extend(["--worker-sample-spec-file", str(worker_spec_path)])

        worker_env = _build_worker_env(
            config_path,
            caption_batch_size=caption_batch_size,
            caption_max_frames=caption_max_frames,
            caption_max_new_tokens=caption_max_new_tokens,
        )
        group_started_at = time.time()
        streamed_results: Dict[str, Dict[str, Any]] = {}
        queue_wait_seconds = 0.0
        with open(worker_log_path, "w", encoding="utf-8") as worker_log:
            worker_log.write(f"[Scheduler] worker_command={' '.join(command)}\n")
            worker_log.write(f"[Scheduler] worker_group_size={len(pending_samples)}\n")
            worker_log.write(f"[Scheduler] worker_uid_file={worker_uid_path}\n")
            worker_log.write(f"[Scheduler] worker_attempt={attempt_index}\n")
            worker_log.write("[Scheduler] queue_owner=isolated_worker\n")
            worker_log.write(
                "[Scheduler] caption_env="
                f"batch_size={worker_env.get('TOOLSTAR_CAPTION_BATCH_SIZE')} "
                f"max_frames={worker_env.get('TOOLSTAR_CAPTION_MAX_FRAMES')} "
                f"max_new_tokens={worker_env.get('TOOLSTAR_CAPTION_MAX_NEW_TOKENS')} "
                f"asr_device={worker_env.get('TOOLSTAR_ASR_DEVICE', 'settings-default')} "
                f"share_planner_visual_runtime={worker_env.get('TOOLSTAR_SHARE_PLANNER_VISUAL_RUNTIME')} "
                f"provider_queue_capacity={worker_env.get('TOOLSTAR_ACCELERATOR_QUEUE_CAPACITY')} "
                f"generation_queue_capacity={worker_env.get('TOOLSTAR_ACCELERATOR_GENERATION_QUEUE_CAPACITY')}\n"
            )
            worker_log.flush()
            with _hold_sample_worker_admission_slot(
                config=config,
                visible_device_id=visible_device_id,
            ) as queue_wait_seconds:
                worker_log.write(
                    "[Scheduler] persistent_model_worker_admission_wait_seconds="
                    f"{queue_wait_seconds:.3f}\n"
                )
                worker_log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    env=worker_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for raw_line in process.stdout:
                    worker_log.write(raw_line)
                    worker_log.flush()
                    payload = _extract_worker_result_payload(raw_line)
                    if not isinstance(payload, dict):
                        continue
                    sample_uid = str(payload.get("sample_uid") or "").strip()
                    if not sample_uid or sample_uid in streamed_results:
                        continue
                    payload.setdefault(
                        "scheduler_accelerator_queue_wait_seconds",
                        round(float(queue_wait_seconds), 3),
                    )
                    streamed_results[sample_uid] = payload
                    # A reusable worker can own thousands of samples.  Emit
                    # each terminal sample as soon as its marker arrives so
                    # the parent can checkpoint live progress instead of
                    # waiting for the whole group to finish.
                    if (
                        on_result is not None
                        and sample_uid not in emitted_uids
                        and not _payload_is_worker_oom(payload)
                    ):
                        on_result(payload)
                        emitted_uids.add(sample_uid)
                process.stdout.close()
                process.wait()

        last_return_code = int(process.returncode)
        last_worker_log_path = worker_log_path
        try:
            worker_output = worker_log_path.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            worker_output = ""
        worker_log_has_oom = _contains_worker_oom_marker(worker_output[-200000:])
        attempt_payloads: Dict[str, Dict[str, Any]] = dict(streamed_results)
        for sample in pending_samples:
            sample_uid = str(sample.sample_uid)
            if sample_uid in attempt_payloads:
                continue
            result_path = sample.workspace_root.resolve() / SAMPLE_RESULT_FILENAME
            try:
                if float(result_path.stat().st_mtime) + 1.0 < group_started_at:
                    continue
            except OSError:
                continue
            payload = _read_sample_result_payload(sample)
            if payload is not None:
                payload.setdefault(
                    "scheduler_accelerator_queue_wait_seconds",
                    round(float(queue_wait_seconds), 3),
                )
                attempt_payloads[sample_uid] = payload

        retry_samples: List[MixSample] = []
        retry_payloads: Dict[str, Optional[Dict[str, Any]]] = {}
        for sample in pending_samples:
            sample_uid = str(sample.sample_uid)
            payload = attempt_payloads.get(sample_uid)
            if payload is not None and _payload_is_worker_oom(payload):
                retry_samples.append(sample)
                retry_payloads[sample_uid] = payload
                continue
            if payload is None and last_return_code != 0 and worker_log_has_oom:
                retry_samples.append(sample)
                retry_payloads[sample_uid] = None
                continue
            if payload is not None:
                final_results[sample_uid] = payload
                if on_result is not None and sample_uid not in emitted_uids:
                    on_result(payload)
                    emitted_uids.add(sample_uid)

        if retry_samples and attempt_index <= max_oom_retries:
            LOGGER.warning(
                "Retrying %d sample(s) after worker OOM; attempt=%d/%d delay=%.1fs "
                "visible_cuda_device=%s worker_log=%s",
                len(retry_samples),
                attempt_index,
                max_oom_retries,
                retry_delay_seconds,
                visible_device_id,
                worker_log_path,
            )
            if retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
            pending_samples = retry_samples
            continue

        if retry_samples:
            for sample in retry_samples:
                sample_uid = str(sample.sample_uid)
                payload = retry_payloads.get(sample_uid)
                if payload is None:
                    payload = _build_scheduler_failure_result(
                        sample=sample,
                        error_text=(
                            "Persistent Mix worker exhausted CUDA OOM retries. "
                            f"attempts={attempt_index} worker_log={worker_log_path}"
                        ),
                    )
                else:
                    payload = dict(payload)
                    payload["status"] = "failed"
                    payload["acc"] = None
                    payload["predicted_label"] = ""
                    payload["failure_class"] = "oom"
                    payload["accuracy_eligible"] = False
                    payload["worker_oom_retry_exhausted"] = True
                    payload["worker_oom_retry_attempts"] = int(attempt_index)
                    payload["worker_log"] = str(worker_log_path)
                    payload.setdefault(
                        "error",
                        "Persistent Mix worker exhausted CUDA OOM retries.",
                    )
                    _write_sample_result(
                        sample.workspace_root.resolve() / SAMPLE_RESULT_FILENAME,
                        payload,
                    )
                final_results[sample_uid] = payload
                if on_result is not None and sample_uid not in emitted_uids:
                    on_result(payload)
                    emitted_uids.add(sample_uid)
            pending_samples = []
            last_return_code = 0
            continue

        # A source mutation is published atomically while this persistent
        # worker is alive. Python modules cannot be replaced safely in-place;
        # the worker asks for a clean process boundary and the parent retries
        # only UIDs that were not already committed in this attempt.
        if (
            last_return_code == ONLINE_RUNTIME_RESTART_EXIT_CODE
            and _contains_online_restart_marker(worker_output[-200000:])
        ):
            remaining_samples = [
                sample
                for sample in pending_samples
                if str(sample.sample_uid) not in final_results
            ]
            if not remaining_samples:
                pending_samples = []
                last_return_code = 0
                continue
            if online_restart_count < online_restart_limit:
                online_restart_count += 1
                LOGGER.info(
                    "Restarting persistent worker after online source snapshot publication; "
                    "restart=%d/%d remaining=%d",
                    online_restart_count,
                    online_restart_limit,
                    len(remaining_samples),
                )
                pending_samples = remaining_samples
                continue
            LOGGER.error(
                "Online source snapshot restart limit exhausted; remaining samples stay pending."
            )
            break

        if last_return_code != 0:
            # Non-OOM worker aborts remain pending.  The caller records the
            # worker log and can resume them in a later batch invocation.
            break

        # A clean process exit without a per-sample payload is still an
        # execution gap, not benchmark data.  Keep those UIDs pending so the
        # next invocation can retry them instead of manufacturing failures.
        LOGGER.error(
            "Persistent Mix worker exited without payloads for %d sample(s); keeping them pending. worker_log=%s",
            sum(1 for sample in pending_samples if str(sample.sample_uid) not in final_results),
            worker_log_path,
        )
        break

    completed_uids = {str(item) for item in final_results}
    all_completed = all(
        str(sample.sample_uid) in completed_uids for sample in original_samples
    )
    return (
        list(final_results.values()),
        0 if all_completed else int(last_return_code or 1),
        last_worker_log_path,
    )


def _build_scheduler_failure_result(
    *,
    sample: MixSample,
    error_text: str,
) -> Dict[str, Any]:
    workspace_root = sample.workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    launcher_log_path = workspace_root / "launcher.log"
    sample_result_path = workspace_root / SAMPLE_RESULT_FILENAME
    result_item = {
        "sample_uid": sample.sample_uid,
        "dataset_name": sample.dataset_name,
        "subset_name": sample.subset_name,
        "status": "failed",
        "acc": None if _contains_worker_oom_marker(error_text) else 0.0,
        "failure_class": (
            "oom" if _contains_worker_oom_marker(error_text) else "scheduler"
        ),
        "accuracy_eligible": False,
        "tool_use_num": 0,
        "iter_num": 0,
        "tool_success_rate": 0.0,
        "successful_tool_calls": 0,
        "planner_iterations": 0,
        "subagent_iterations": 0,
        "predicted_label": "",
        "ground_truth": sample.ground_truth,
        "ground_truth_option": sample.ground_truth_option,
        "ground_truth_labels": [],
        "planner_success": False,
        "raw_final_answer": "",
        "elapsed_seconds": 0.0,
        "workspace_root": str(workspace_root),
        "launcher_log": str(launcher_log_path),
        "result_path": str(sample_result_path),
        "run_result_path": str((workspace_root / "results" / "run_result.json").resolve()),
        "source_video_path": str(sample.video_path),
        "prepared_video_path": str(sample.video_path),
        "question_block": sample.question_block,
        "options": list(sample.options),
        "official_meta": dict(sample.official_meta),
        "error": str(error_text),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(launcher_log_path, "a", encoding="utf-8") as handle:
        handle.write(f"[Launcher] scheduler_worker_failure={error_text}\n")
    _write_sample_result(sample_result_path, result_item)
    return result_item


def run_single_sample(
    *,
    sample: MixSample,
    config: BatchConfig,
    no_cache: bool,
) -> Dict[str, Any]:
    _sync_online_runtime_snapshot()
    run_e2e_module = _load_run_e2e_module()
    # Reset all sampling RNGs per case, including reusable workers. Candidate
    # and baseline must not inherit different random histories from prior cases.
    import random
    import numpy as np
    import torch
    seed = int(os.environ.get("CFV_REPLAY_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pipeline_options = dict(config.pipeline_options or {})
    env_pipeline_mode = str(os.environ.get("TOOLSTAR_PIPELINE_MODE_OVERRIDE") or "").strip()
    if env_pipeline_mode:
        pipeline_options["mode"] = env_pipeline_mode
    env_single_tool_only = str(os.environ.get("TOOLSTAR_ABLATION_SINGLE_TOOL_ONLY") or "").strip().lower()
    if env_single_tool_only in {"1", "true", "yes", "on"}:
        pipeline_options["planner_toolsearch_direct_single_tool_only"] = True
    env_single_action_per_step = str(os.environ.get("TOOLSTAR_ABLATION_SINGLE_ACTION_PER_STEP") or "").strip().lower()
    if env_single_action_per_step in {"1", "true", "yes", "on"}:
        pipeline_options["planner_single_action_per_step"] = True
    env_disable_meta_tools = str(os.environ.get("TOOLSTAR_ABLATION_DISABLE_META_TOOLS") or "").strip().lower()
    if env_disable_meta_tools in {"1", "true", "yes", "on"}:
        pipeline_options["disable_meta_tools"] = True
    env_disabled_tool_names = str(
        os.environ.get("TOOLSTAR_ABLATION_DISABLED_TOOL_NAMES") or ""
    ).strip()
    if env_disabled_tool_names:
        pipeline_options["disabled_tool_names"] = [
            item.strip()
            for item in env_disabled_tool_names.replace(";", ",").split(",")
            if item.strip()
        ]
    env_disabled_agent_roles = str(
        os.environ.get("TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES") or ""
    ).strip()
    if env_disabled_agent_roles:
        pipeline_options["disabled_agent_roles"] = [
            item.strip().lower()
            for item in env_disabled_agent_roles.replace(";", ",").split(",")
            if item.strip()
        ]
    workspace_root = sample.workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    launcher_log_path = workspace_root / "launcher.log"
    sample_result_path = workspace_root / SAMPLE_RESULT_FILENAME
    _cleanup_workspace(workspace_root)

    started_at = time.perf_counter()
    run_result_payload: Dict[str, Any] = {}
    final_status = "failed"
    error_text = ""
    prepared_video_path = sample.video_path.resolve()

    with open(launcher_log_path, "w", encoding="utf-8") as handle:
        handle.write(f"[Launcher] sample_uid={sample.sample_uid}\n")
        handle.write(f"[Launcher] dataset_name={sample.dataset_name}\n")
        handle.write(f"[Launcher] subset_name={sample.subset_name}\n")
        handle.write(f"[Launcher] source_video={sample.video_path}\n")
        handle.write("[Launcher] user_query_begin\n")
        handle.write(sample.prompt + "\n")
        handle.write("[Launcher] user_query_end\n")

    try:
        prepared_video_path = prepare_sample_video(sample, config.ffmpeg_binary, launcher_log_path)
        execute_pipeline = getattr(run_e2e_module, "execute_pipeline")
        supported_kwargs = set()
        try:
            supported_kwargs = set(inspect.signature(execute_pipeline).parameters.keys())
        except Exception:
            supported_kwargs = set()
        pipeline_call_kwargs: Dict[str, Any] = {
            "video_path": str(prepared_video_path),
            "query": sample.prompt,
            "workspace_root": workspace_root,
            "shared_preprocess_root": config.shared_preprocess_root,
            "no_cache": no_cache,
            "device_type": None,
            "write_log_file": False,
            "persist_query_workspace": False,
            "runtime_profile": "evaluation_local",
        }
        optional_pipeline_kwargs = {
            "input_contract": dict(sample.official_meta.get("input_contract") or {}),
            "pipeline_mode": str(pipeline_options.get("mode") or "react"),
            "planner_direct_answer_prompt_path": (
                str(pipeline_options.get("planner_direct_answer_prompt_path"))
                if pipeline_options.get("planner_direct_answer_prompt_path")
                else None
            ),
            "include_grounding_video_clip_qa": bool(
                pipeline_options.get("include_grounding_video_clip_qa", True)
            ),
            "planner_toolsearch_direct_single_tool_only": bool(
                pipeline_options.get("planner_toolsearch_direct_single_tool_only", False)
            ),
            "planner_single_action_per_step": bool(
                pipeline_options.get("planner_single_action_per_step", False)
            ),
            "runtime_disable_meta_tools": bool(
                pipeline_options.get("disable_meta_tools", False)
            ),
            "runtime_disabled_tool_names": list(
                pipeline_options.get("disabled_tool_names") or []
            ),
            "runtime_disabled_agent_roles": list(
                pipeline_options.get("disabled_agent_roles") or []
            ),
        }
        unsupported_kwargs = [
            key for key in optional_pipeline_kwargs.keys()
            if supported_kwargs and key not in supported_kwargs
        ]
        if unsupported_kwargs:
            with open(launcher_log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    "[Launcher] execute_pipeline_missing_optional_kwargs="
                    + ",".join(sorted(unsupported_kwargs))
                    + "\n"
                )
        for key, value in optional_pipeline_kwargs.items():
            if not supported_kwargs or key in supported_kwargs:
                pipeline_call_kwargs[key] = value
        with capture_sample_logs(launcher_log_path):
            pipeline_payload = asyncio.run(
                execute_pipeline(**pipeline_call_kwargs)
            )
        run_result_payload = dict(pipeline_payload.get("run_result", {})) if isinstance(pipeline_payload, dict) else {}
        metrics_payload = _build_metrics_from_reward_helper(sample, run_result_payload)
        raw_final_answer = (
            run_result_payload.get("agent", {}).get("final_answer", "")
            if isinstance(run_result_payload.get("agent"), dict)
            else ""
        )
        pipeline_error = run_result_payload.get("error", "")
        oom_failure = _contains_worker_oom_marker(raw_final_answer) or _contains_worker_oom_marker(
            pipeline_error
        )
        evidence_unanswerable = _is_evidence_unanswerable_answer(raw_final_answer)
        pipeline_success = bool(run_result_payload.get("success", False))
        planner_terminal_error = str(raw_final_answer or "").strip().lower().startswith(
            ("error:", "failed:")
        )
        final_status = (
            "failed"
            if oom_failure
            else "unanswerable"
            if evidence_unanswerable
            else "failed"
            if not pipeline_success or planner_terminal_error
            else "completed" if metrics_payload["answer_correct"] else "wrong"
        )
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        result_item = {
            "sample_uid": sample.sample_uid,
            "dataset_name": sample.dataset_name,
            "subset_name": sample.subset_name,
            "status": final_status,
            "acc": (
                None
                if evidence_unanswerable or oom_failure
                else 0.0
                if final_status == "failed"
                else 1.0
                if metrics_payload["answer_correct"]
                else 0.0
            ),
            "tool_use_num": metrics_payload["tool_use_num"],
            "iter_num": metrics_payload["iter_num"],
            "tool_success_rate": metrics_payload["tool_success_rate"],
            "successful_tool_calls": metrics_payload["successful_tool_calls"],
            "planner_iterations": metrics_payload["planner_iterations"],
            "subagent_iterations": metrics_payload["subagent_iterations"],
            "predicted_label": (
                ""
                if evidence_unanswerable or oom_failure or final_status == "failed"
                else metrics_payload["predicted_label"]
            ),
            "failure_class": "oom" if oom_failure else (
                "sample_failure" if final_status == "failed" else ""
            ),
            "accuracy_eligible": final_status in {
                "completed", "wrong", "unanswerable"
            } and not oom_failure,
            "ground_truth": sample.ground_truth,
            "ground_truth_option": sample.ground_truth_option,
            "ground_truth_labels": metrics_payload["ground_truth_labels"],
            "planner_success": bool(run_result_payload.get("success", False)),
            "raw_final_answer": raw_final_answer,
            "elapsed_seconds": elapsed_seconds,
            "workspace_root": str(workspace_root),
            "launcher_log": str(launcher_log_path),
            "result_path": str(sample_result_path),
            "run_result_path": str((workspace_root / "results" / "run_result.json").resolve()),
            "source_video_path": str(sample.video_path),
            "prepared_video_path": str(prepared_video_path),
            "question_block": sample.question_block,
            "options": list(sample.options),
            "official_meta": dict(sample.official_meta),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        _write_sample_result(sample_result_path, result_item)
        return result_item
    except Exception as exc:
        error_text = str(exc)
        oom_failure = _contains_worker_oom_marker(error_text)
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        result_item = {
            "sample_uid": sample.sample_uid,
            "dataset_name": sample.dataset_name,
            "subset_name": sample.subset_name,
            "status": "failed",
            "acc": None if oom_failure else 0.0,
            "failure_class": "oom" if oom_failure else "sample_failure",
            "accuracy_eligible": False,
            "tool_use_num": 0,
            "iter_num": 0,
            "tool_success_rate": 0.0,
            "successful_tool_calls": 0,
            "planner_iterations": 0,
            "subagent_iterations": 0,
            "predicted_label": "",
            "ground_truth": sample.ground_truth,
            "ground_truth_option": sample.ground_truth_option,
            "ground_truth_labels": [],
            "planner_success": False,
            "raw_final_answer": "",
            "elapsed_seconds": elapsed_seconds,
            "workspace_root": str(workspace_root),
            "launcher_log": str(launcher_log_path),
            "result_path": str(sample_result_path),
            "run_result_path": str((workspace_root / "results" / "run_result.json").resolve()),
            "source_video_path": str(sample.video_path),
            "prepared_video_path": str(prepared_video_path),
            "question_block": sample.question_block,
            "options": list(sample.options),
            "official_meta": dict(sample.official_meta),
            "error": error_text,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(launcher_log_path, "a", encoding="utf-8") as handle:
            handle.write(f"[Launcher] exception={error_text}\n")
        _write_sample_result(sample_result_path, result_item)
        return result_item


def build_summary_rows(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(float(item.get("acc") or 0.0) for item in results)
    tool_use_sum = sum(float(item.get("tool_use_num") or 0.0) for item in results)
    iter_sum = sum(float(item.get("iter_num") or 0.0) for item in results)
    tool_success_rate_sum = sum(float(item.get("tool_success_rate") or 0.0) for item in results)
    elapsed_seconds_sum = sum(float(item.get("elapsed_seconds") or 0.0) for item in results)
    statuses: Dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "count": total,
        "correct": int(correct),
        "acc": round(_safe_divide(correct, total), 6),
        "tool_use_num_mean": round(_safe_divide(tool_use_sum, total), 6),
        "iter_num_mean": round(_safe_divide(iter_sum, total), 6),
        "tool_success_rate_mean": round(_safe_divide(tool_success_rate_sum, total), 6),
        "elapsed_seconds_mean": round(_safe_divide(elapsed_seconds_sum, total), 6),
        "statuses": dict(sorted(statuses.items())),
    }


def build_summary_by_dataset(
    results: Sequence[Mapping[str, Any]],
    dataset_order: Sequence[str],
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for item in results:
        grouped.setdefault(str(item.get("dataset_name") or "unknown"), []).append(item)
    summary = {
        dataset_name: build_summary_rows(grouped.get(dataset_name, []))
        for dataset_name in _ordered_dataset_names(dataset_order, grouped.keys())
    }
    videomme_items = grouped.get("videomme", [])
    if videomme_items:
        summary["videomme"]["official_breakdown"] = build_videomme_breakdown(videomme_items)
    return summary


def build_subset_summaries(
    results: Sequence[Mapping[str, Any]],
    dataset_order: Sequence[str],
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
    for item in results:
        dataset_name = str(item.get("dataset_name") or "unknown")
        subset_name = str(item.get("subset_name") or "unknown")
        grouped.setdefault(dataset_name, {}).setdefault(subset_name, []).append(item)
    payload: Dict[str, Any] = {}
    for dataset_name in _ordered_dataset_names(dataset_order, grouped.keys()):
        subset_groups = grouped.get(dataset_name, {})
        payload[dataset_name] = {
            subset_name: build_summary_rows(subset_results)
            for subset_name, subset_results in sorted(subset_groups.items())
        }
    return payload


def build_videomme_breakdown(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    axes = {
        "duration": {},
        "domain": {},
        "sub_category": {},
        "task_type": {},
    }
    total_correct = 0.0
    total_answered = 0
    for item in results:
        official_meta = item.get("official_meta") if isinstance(item.get("official_meta"), dict) else {}
        extracted_option = extract_videomme_option(item.get("raw_final_answer"))
        if not extracted_option:
            continue
        is_correct = 1.0 if extracted_option == str(item.get("ground_truth_option") or "") else 0.0
        total_correct += is_correct
        total_answered += 1
        for axis_name, key_name in (
            ("duration", "duration"),
            ("domain", "domain"),
            ("sub_category", "sub_category"),
            ("task_type", "task_type"),
        ):
            axis_value = str(official_meta.get(key_name) or "unknown")
            axis_bucket = axes[axis_name].setdefault(axis_value, {"correct": 0.0, "answered": 0})
            axis_bucket["correct"] += is_correct
            axis_bucket["answered"] += 1
    for axis_name, axis_groups in axes.items():
        for axis_value, stats in axis_groups.items():
            stats["acc"] = round(_safe_divide(float(stats["correct"]), int(stats["answered"])), 6)
    return {
        **axes,
        "overall": {
            "correct": int(total_correct),
            "answered": total_answered,
            "acc": round(_safe_divide(total_correct, total_answered), 6),
        },
    }


def build_acc_line(
    results: Sequence[Mapping[str, Any]],
    dataset_order: Sequence[str],
) -> str:
    ordered_names = _ordered_dataset_names(dataset_order, [])
    grouped: Dict[str, List[Mapping[str, Any]]] = {name: [] for name in ordered_names}
    for item in results:
        dataset_name = str(item.get("dataset_name") or "")
        grouped.setdefault(dataset_name, []).append(item)
    parts: List[str] = []
    for dataset_name in _ordered_dataset_names(dataset_order, grouped.keys()):
        dataset_results = grouped.get(dataset_name, [])
        correct = sum(float(item.get("acc") or 0.0) for item in dataset_results)
        count = len(dataset_results)
        status_counts = {"completed": 0, "wrong": 0, "unanswerable": 0}
        for item in dataset_results:
            status = str(item.get("status") or "").strip().lower()
            if status in status_counts:
                status_counts[status] += 1
        parts.append(
            f"{dataset_name}={correct:.0f}/{count} "
            f"({100.0 * _safe_divide(correct, count):.2f}%) "
            f"[completed={status_counts['completed']},"
            f"wrong={status_counts['wrong']},"
            f"unanswerable={status_counts['unanswerable']}]"
        )
    return "ACC " + " | ".join(parts)


def _dataset_progress_counts(
    progress_payload: Mapping[str, Any],
    inventories: Sequence[DatasetInventory],
    dataset_name: str,
) -> Tuple[int, int]:
    normalized_dataset = str(dataset_name or "").strip()
    if not normalized_dataset:
        return 0, 0
    for inventory in inventories:
        if str(inventory.dataset_name).strip() != normalized_dataset:
            continue
        dataset_samples = [
            sample
            for subset_name in inventory.ordered_subset_names
            for sample in inventory.samples_by_subset.get(subset_name, ())
        ]
        total = len(dataset_samples)
        participated = sum(
            1
            for sample in dataset_samples
            if has_participated(progress_payload, sample.sample_uid)
        )
        return participated, total
    return 0, 0


def build_dataset_progress_line(
    progress_payload: Mapping[str, Any],
    inventories: Sequence[DatasetInventory],
    dataset_name: str,
) -> str:
    participated, total = _dataset_progress_counts(progress_payload, inventories, dataset_name)
    normalized_dataset = str(dataset_name or "").strip()
    if not normalized_dataset or total <= 0:
        return ""
    return f"[{normalized_dataset} {participated}/{total}]"


def build_resume_progress_line(
    progress_payload: Mapping[str, Any],
    inventories: Sequence[DatasetInventory],
    dataset_order: Sequence[str],
) -> str:
    parts: List[str] = []
    for dataset_name in _ordered_dataset_names(dataset_order, [item.dataset_name for item in inventories]):
        marker = build_dataset_progress_line(progress_payload, inventories, dataset_name)
        if marker:
            parts.append(marker)
    if not parts:
        return ""
    return "Resume " + " | ".join(parts)


def build_selection_progress(
    *,
    progress_payload: Mapping[str, Any],
    inventories: Sequence[DatasetInventory],
    selected_samples: Sequence[MixSample],
    imported_skip_true_count: int,
    max_samples: Optional[int],
    parallel_samples: int,
    skip_true: bool,
    dataset_order: Sequence[str],
) -> Dict[str, Any]:
    all_samples = list(_iter_all_samples(inventories))
    summary = summarize_progress(progress_payload, [item.sample_uid for item in all_samples])
    per_dataset: Dict[str, Dict[str, int]] = {}
    for dataset_name in _ordered_dataset_names(dataset_order, [item.dataset_name for item in all_samples]):
        dataset_samples = [item for item in all_samples if item.dataset_name == dataset_name]
        participated = [
            item for item in dataset_samples
            if has_participated(progress_payload, item.sample_uid)
        ]
        per_dataset[dataset_name] = {
            "total": len(dataset_samples),
            "participated": len(participated),
            "remaining": len(dataset_samples) - len(participated),
        }
    summary.update(
        {
            "selected_samples": len(selected_samples),
            "selected_sample_uids": [item.sample_uid for item in selected_samples],
            "imported_skip_true_count": imported_skip_true_count,
            "max_samples": max_samples,
            "parallel_samples": parallel_samples,
            "skip_true": bool(skip_true),
            "dataset_order": list(_ordered_dataset_names(dataset_order, per_dataset.keys())),
            "per_dataset": per_dataset,
        }
    )
    return summary


def build_videomme_official_output(
    inventory: DatasetInventory,
    result_map: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for subset_name in inventory.ordered_subset_names:
        for sample in inventory.samples_by_subset.get(subset_name, ()):
            meta = sample.official_meta
            video_id = str(meta.get("video_id") or "unknown")
            question_id = str(meta.get("question_id") or "")
            video_entry = grouped.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "duration": str(meta.get("duration") or ""),
                    "domain": str(meta.get("domain") or ""),
                    "sub_category": str(meta.get("sub_category") or ""),
                    "missing": True,
                    "questions": [],
                    "_order": int(meta.get("video_item_index") or 0),
                },
            )
            result_item = result_map.get(sample.sample_uid, {})
            response = str(result_item.get("raw_final_answer") or "")
            if response:
                video_entry["missing"] = False
            video_entry["questions"].append(
                {
                    "question_id": question_id,
                    "task_type": str(meta.get("task_type") or ""),
                    "question": sample.question_block.splitlines()[0].replace("Question: ", "", 1),
                    "options": list(sample.options),
                    "answer": sample.ground_truth_option,
                    "response": response,
                    "_order": int(meta.get("question_index") or 0),
                }
            )
    output: List[Dict[str, Any]] = []
    for item in sorted(grouped.values(), key=lambda record: int(record.get("_order", 0))):
        item["questions"] = [
            {key: value for key, value in question.items() if key != "_order"}
            for question in sorted(item["questions"], key=lambda question: int(question.get("_order", 0)))
        ]
        item.pop("_order", None)
        output.append(item)
    return output


def _expanded_parallel_device_ids(
    *,
    visible_device_ids: Sequence[str],
    worker_count: int,
    allow_same_device_parallel: bool,
) -> List[str]:
    devices = [str(item) for item in visible_device_ids if str(item).strip()]
    if not devices:
        devices = [""]
    if len(devices) >= worker_count:
        return devices[:worker_count]
    if not allow_same_device_parallel:
        return devices
    return [devices[index % len(devices)] for index in range(worker_count)]


def _resolve_persistent_model_worker_count(
    *,
    requested_worker_count: int,
    selected_count: int,
    admission_capacity: int = 1,
    resident_model_workers_per_device: int = 1,
    visible_device_ids: Sequence[str] = (),
) -> int:
    """Resolve long-lived model owners within the resident-memory budget.

    ``parallel_samples`` is a logical task-pool setting.  It must not create
    more persistent model owners than the explicit resident-worker cap or the
    outer admission queue can hold per physical device.  The memory guard
    controls admission timing; it cannot estimate the cold-loaded model's
    working set, so it is not a substitute for this topology cap.
    """
    requested = max(1, min(int(requested_worker_count), max(1, int(selected_count))))
    normalized_devices = {
        str(item).strip()
        for item in visible_device_ids
        if str(item).strip()
    }
    physical_device_count = max(1, len(normalized_devices))
    try:
        per_device_capacity = max(1, min(8, int(admission_capacity)))
    except (TypeError, ValueError):
        per_device_capacity = 1
    try:
        resident_per_device = max(1, min(8, int(resident_model_workers_per_device)))
    except (TypeError, ValueError):
        resident_per_device = 1
    resident_worker_cap = max(
        1,
        min(per_device_capacity, resident_per_device) * physical_device_count,
    )
    raw_value = str(os.environ.get("TOOLSTAR_PERSISTENT_MODEL_WORKERS") or "").strip()
    if not raw_value:
        return min(requested, resident_worker_cap)
    try:
        configured = int(raw_value)
    except (TypeError, ValueError):
        configured = requested
    return max(1, min(requested, max(1, configured), resident_worker_cap))


def _partition_reusable_worker_groups(
    selected_samples: Sequence[MixSample],
    worker_count: int,
) -> List[List[MixSample]]:
    """Partition by video so one reusable worker owns a video's cache path."""

    groups: List[List[MixSample]] = [[] for _ in range(max(1, worker_count))]
    assigned_group_by_video: Dict[str, int] = {}
    for sample in selected_samples:
        cache_key = _sample_cache_key(sample.video_path)
        group_index = assigned_group_by_video.get(cache_key)
        if group_index is None:
            group_index = min(range(len(groups)), key=lambda index: len(groups[index]))
            assigned_group_by_video[cache_key] = group_index
        groups[group_index].append(sample)
    return [group for group in groups if group]


def _execute_parallel_reusable_workers(
    *,
    selected_samples: Sequence[MixSample],
    config: BatchConfig,
    no_cache: bool,
    progress_payload: MutableMapping[str, Any],
    inventories: Sequence[DatasetInventory],
    effective_parallel_samples: int,
    visible_device_ids: Sequence[str],
    on_result: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> None:
    """Run persistent worker processes in parallel and stream checkpoints.

    Each worker cold-loads its models once, then processes its assigned samples
    in-process.  This preserves the requested cold-load semantics while using
    the otherwise idle GPU during preprocessing/model waits.  Samples for the
    same video stay on one worker to avoid concurrent writes to shared caches.
    """

    persistent_worker_count = _resolve_persistent_model_worker_count(
        requested_worker_count=effective_parallel_samples,
        selected_count=len(selected_samples),
        admission_capacity=config.sample_worker_admission_capacity,
        resident_model_workers_per_device=config.resident_model_workers_per_device,
        visible_device_ids=visible_device_ids,
    )
    groups = _partition_reusable_worker_groups(selected_samples, persistent_worker_count)
    device_pool = _expanded_parallel_device_ids(
        visible_device_ids=visible_device_ids,
        worker_count=len(groups),
        allow_same_device_parallel=config.allow_same_device_parallel,
    )
    LOGGER.info(
        "Persistent worker admission plan: requested=%d resident_workers=%d "
        "admission_capacity_per_device=%d resident_model_workers_per_device=%d "
        "physical_devices=%d device_pool=%s",
        effective_parallel_samples,
        persistent_worker_count,
        config.sample_worker_admission_capacity,
        config.resident_model_workers_per_device,
        max(1, len({str(item).strip() for item in visible_device_ids if str(item).strip()})),
        device_pool,
    )
    result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    selected_uids = {str(sample.sample_uid) for sample in selected_samples}
    consumed_uids: Set[str] = set()
    worker_failure_messages: List[str] = []
    dataset_order = [
        name
        for name in config.dataset_order
        if bool(config.datasets.get(name, {}).get("enabled", True))
    ]

    def enqueue_result(payload: Dict[str, Any]) -> None:
        result_queue.put(payload)

    def consume_result(payload: Mapping[str, Any]) -> None:
        sample_uid = str(payload.get("sample_uid") or "").strip()
        if not sample_uid or sample_uid not in selected_uids or sample_uid in consumed_uids:
            return
        status = str(payload.get("status") or "").strip().lower()
        if status not in {"completed", "wrong", "unanswerable", "failed"}:
            return
        if _payload_is_worker_oom(payload) and not payload.get("worker_oom_retry_exhausted"):
            return
        # Record final failures immediately; has_participated still leaves
        # them retryable and excludes them from benchmark coverage.
        consumed_uids.add(sample_uid)
        upsert_progress_record(progress_payload, dict(payload))
        save_progress(config.progress_path, progress_payload)
        if on_result is not None:
            on_result(payload)
        progress_line = build_dataset_progress_line(
            progress_payload,
            inventories,
            str(payload.get("dataset_name") or ""),
        )
        cumulative_results = load_result_items_from_progress(progress_payload)
        if progress_line:
            LOGGER.info("%s %s", progress_line, build_acc_line(cumulative_results, dataset_order))
        else:
            LOGGER.info("%s", build_acc_line(cumulative_results, dataset_order))

    last_reconcile_monotonic = 0.0

    def reconcile_persisted_results() -> None:
        """Recover terminal artifacts even before a worker prints its marker.

        A reusable worker writes ``mix_sample_result.json`` before emitting
        ``[RUN_RESULT_JSON]``.  If generation is still in progress, or the
        process later dies, waiting for the whole 2k-sample group to finish
        leaves progress stale and makes a live run look hung.  The artifact is
        the authoritative per-sample checkpoint; only terminal participated
        statuses are imported here.
        """

        for group in groups:
            for sample in group:
                sample_uid = str(sample.sample_uid)
                if sample_uid in consumed_uids:
                    continue
                payload = _read_sample_result_payload(sample)
                if payload is not None:
                    consume_result(payload)

    LOGGER.info(
        "Launching logical_worker_count=%d with persistent_model_worker_count=%d; groups=%s device_pool=%s same_device_parallel=%s admission_capacity=%s resident_model_workers_per_device=%s memory_guard=%s min_free_gib=%.2f reserve_gib=%.2f startup_stagger_seconds=%.1f",
        effective_parallel_samples,
        persistent_worker_count,
        [len(group) for group in groups],
        device_pool,
        bool(config.allow_same_device_parallel),
        config.sample_worker_admission_capacity,
        config.resident_model_workers_per_device,
        bool(config.accelerator_queue_memory_guard),
        float(config.accelerator_queue_min_free_gib),
        float(config.accelerator_queue_reserve_gib),
        float(config.persistent_worker_start_stagger_seconds),
    )
    future_to_group: Dict[
        concurrent.futures.Future[Tuple[List[Dict[str, Any]], int, Path]],
        List[MixSample],
    ] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as executor:
        for index, group in enumerate(groups):
            if index > 0 and config.persistent_worker_start_stagger_seconds > 0:
                LOGGER.info(
                    "Staggering persistent worker startup before group %d/%d by %.1fs to reduce shared-model cold-load I/O contention.",
                    index + 1,
                    len(groups),
                    float(config.persistent_worker_start_stagger_seconds),
                )
                time.sleep(float(config.persistent_worker_start_stagger_seconds))
            LOGGER.info(
                "Reusable parallel group %d/%d size=%d first=%s last=%s visible_cuda_device=%s",
                index + 1,
                len(groups),
                len(group),
                group[0].sample_uid,
                group[-1].sample_uid,
                device_pool[index] if index < len(device_pool) else "",
            )
            future = executor.submit(
                _run_mix_sample_group_worker,
                config=config,
                config_path=config.config_path,
                samples=group,
                no_cache=no_cache,
                visible_device_id=device_pool[index] if index < len(device_pool) else "",
                caption_batch_size=config.caption_batch_size,
                caption_max_frames=config.caption_max_frames,
                caption_max_new_tokens=config.caption_max_new_tokens,
                on_result=enqueue_result,
            )
            future_to_group[future] = group

        pending_futures: Set[concurrent.futures.Future[Tuple[List[Dict[str, Any]], int, Path]]] = set(
            future_to_group
        )
        while pending_futures or not result_queue.empty():
            now_monotonic = time.monotonic()
            if now_monotonic - last_reconcile_monotonic >= 2.0:
                reconcile_persisted_results()
                last_reconcile_monotonic = now_monotonic
            try:
                # Avoid blocking inside ``queue.Queue.get(timeout=...)``.
                # Python signal delivery can interrupt Condition.wait between
                # lock acquire/release and leave the scheduler with the
                # misleading ``release unlocked lock`` traceback during a
                # deliberate Ctrl-C.  A short non-blocking poll preserves
                # streaming checkpoints while keeping shutdown interruptible.
                consume_result(result_queue.get_nowait())
            except queue.Empty:
                time.sleep(0.1)
            done = {future for future in pending_futures if future.done()}
            for future in done:
                pending_futures.discard(future)
                group = future_to_group[future]
                try:
                    result_items, return_code, worker_log_path = future.result()
                except Exception as exc:
                    result_items, return_code, worker_log_path = [], -1, Path()
                    LOGGER.exception("Reusable parallel worker failed: %s", exc)
                for result_item in result_items:
                    consume_result(result_item)
                if return_code != 0:
                    worker_failure_messages.append(
                        "Reusable parallel worker exited before completing its group; "
                        f"return_code={return_code} worker_log={worker_log_path}"
                    )
                    for sample in group:
                        if sample.sample_uid not in consumed_uids:
                            # The worker may have terminated during model
                            # loading, CUDA generation, or a low-level abort.
                            # These samples were not executed and must remain
                            # pending so a later invocation can retry them;
                            # manufacturing per-sample failures would turn a
                            # scheduler/runtime failure into benchmark data.
                            LOGGER.error(
                                "Leaving unexecuted sample pending after parallel worker failure "
                                "sample_uid=%s return_code=%s worker_log=%s",
                                sample.sample_uid,
                                return_code,
                                worker_log_path,
                            )

        reconcile_persisted_results()
        while not result_queue.empty():
            try:
                consume_result(result_queue.get_nowait())
            except queue.Empty:
                break
    if worker_failure_messages:
        preview = " | ".join(worker_failure_messages[:3])
        raise RuntimeError(
            "Reusable parallel worker failure; unexecuted samples remain pending. "
            + preview
        )


def _execute_parallel_samples(
    *,
    selected_samples: Sequence[MixSample],
    config: BatchConfig,
    no_cache: bool,
    progress_payload: MutableMapping[str, Any],
    inventories: Sequence[DatasetInventory],
    effective_parallel_samples: int,
    visible_device_ids: Sequence[str],
    on_result: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> None:
    if config.reuse_worker_samples > 1:
        _execute_parallel_reusable_workers(
            selected_samples=selected_samples,
            config=config,
            no_cache=no_cache,
            progress_payload=progress_payload,
            inventories=inventories,
            effective_parallel_samples=effective_parallel_samples,
            visible_device_ids=visible_device_ids,
            on_result=on_result,
        )
        return

    dataset_order = [
        name
        for name in config.dataset_order
        if bool(config.datasets.get(name, {}).get("enabled", True))
    ]
    available_devices: Deque[str] = deque(list(visible_device_ids)[:effective_parallel_samples])
    ordered_dataset_names, pending_by_dataset = _split_pending_samples_by_dataset(selected_samples, dataset_order)
    active_tasks: Dict[concurrent.futures.Future[Dict[str, Any]], Tuple[MixSample, RunningSampleTask]] = {}
    active_cache_keys: Set[str] = set()
    launch_index = 0
    next_dataset_index = 0
    total_samples = len(selected_samples)

    def _remaining_pending_count() -> int:
        return sum(len(samples) for samples in pending_by_dataset.values())

    def _pop_next_launchable_sample() -> Optional[MixSample]:
        nonlocal next_dataset_index
        if not ordered_dataset_names:
            return None
        dataset_count = len(ordered_dataset_names)
        for offset in range(dataset_count):
            dataset_index = (next_dataset_index + offset) % dataset_count
            dataset_name = ordered_dataset_names[dataset_index]
            dataset_pending = pending_by_dataset.get(dataset_name, [])
            if not dataset_pending:
                continue
            sample = _pop_launchable_sample_for_dataset(dataset_pending, active_cache_keys)
            if sample is None:
                continue
            next_dataset_index = (dataset_index + 1) % dataset_count
            return sample
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_parallel_samples) as executor:
        while _remaining_pending_count() or active_tasks:
            launched_any = False
            while available_devices and _remaining_pending_count():
                sample = _pop_next_launchable_sample()
                if sample is None:
                    break
                cache_key = _sample_cache_key(sample.video_path)
                visible_device_id = available_devices.popleft()
                launch_index += 1
                launcher_log = sample.workspace_root.resolve() / "launcher.log"
                LOGGER.info(
                    "Launching sample %d/%d sample_uid=%s visible_cuda_device=%s launcher_log=%s",
                    launch_index,
                    total_samples,
                    sample.sample_uid,
                    visible_device_id,
                    launcher_log,
                )
                future = executor.submit(
                    _run_mix_sample_worker,
                    config=config,
                    config_path=config.config_path,
                    sample=sample,
                    no_cache=no_cache,
                    visible_device_id=visible_device_id,
                    caption_batch_size=config.caption_batch_size,
                    caption_max_frames=config.caption_max_frames,
                    caption_max_new_tokens=config.caption_max_new_tokens,
                )
                active_tasks[future] = (
                    sample,
                    RunningSampleTask(
                        sample_uid=sample.sample_uid,
                        cache_key=cache_key,
                        visible_device_id=visible_device_id,
                        launch_index=launch_index,
                    ),
                )
                active_cache_keys.add(cache_key)
                launched_any = True

            if not active_tasks:
                remaining_uids = [
                    sample.sample_uid
                    for dataset_samples in pending_by_dataset.values()
                    for sample in dataset_samples
                ]
                raise RuntimeError(
                    "Mix parallel scheduler could not launch any remaining sample. "
                    f"Pending sample_uids={remaining_uids}"
                )

            if launched_any and not _remaining_pending_count():
                continue

            done, _ = concurrent.futures.wait(
                tuple(active_tasks.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                sample, task = active_tasks.pop(future)
                active_cache_keys.discard(task.cache_key)
                available_devices.append(task.visible_device_id)
                try:
                    result_item = future.result()
                except Exception as exc:
                    LOGGER.exception(
                        "Mix worker failed sample_uid=%s visible_cuda_device=%s",
                        sample.sample_uid,
                        task.visible_device_id,
                    )
                    result_item = _build_scheduler_failure_result(sample=sample, error_text=str(exc))
                upsert_progress_record(progress_payload, result_item)
                save_progress(config.progress_path, progress_payload)
                if on_result is not None:
                    on_result(result_item)
                cumulative_results = load_result_items_from_progress(progress_payload)
                progress_line = build_dataset_progress_line(progress_payload, inventories, sample.dataset_name)
                if progress_line:
                    LOGGER.info("%s %s", progress_line, build_acc_line(cumulative_results, dataset_order))
                else:
                    LOGGER.info("%s", build_acc_line(cumulative_results, dataset_order))


def _run_worker_mode(args: argparse.Namespace) -> int:
    if str(args.worker_device_id or "").strip():
        _bind_worker_to_visible_cuda_device(str(args.worker_device_id))
    config = load_config(Path(args.config))
    # Direct/manual workers must receive exactly the same model-root and CUDA
    # safety policy as scheduler-spawned workers.  Previously only the parent
    # scheduler called _build_worker_env(), so manual probes silently inherited
    # fused Qwen kernels and crashed during the first Qwen3.5 generation.
    os.environ.update(
        _build_worker_env(
            Path(args.config).resolve(),
            caption_batch_size=config.caption_batch_size,
            caption_max_frames=config.caption_max_frames,
            caption_max_new_tokens=config.caption_max_new_tokens,
        )
    )
    LOGGER.info(
        "Worker caption runtime resolved from Mix config: batch_size=%s max_frames=%s "
        "max_new_tokens=%s env_batch_size=%s env_max_frames=%s env_max_new_tokens=%s",
        config.caption_batch_size,
        config.caption_max_frames,
        config.caption_max_new_tokens,
        os.environ.get("TOOLSTAR_CAPTION_BATCH_SIZE"),
        os.environ.get("TOOLSTAR_CAPTION_MAX_FRAMES"),
        os.environ.get("TOOLSTAR_CAPTION_MAX_NEW_TOKENS"),
    )
    spec_path = str(args.worker_sample_spec_file or "").strip()
    if spec_path:
        selected_samples = _load_worker_sample_spec(Path(spec_path).resolve())
    else:
        # Backward-compatible manual worker mode.  Scheduler-created workers
        # always use a spec file so they do not import/read every parquet
        # inventory just to resolve a UID.
        inventories = load_inventories(config)
        uid_file_path = str(args.worker_sample_uids_file or "").strip()
        if uid_file_path:
            try:
                sample_uids = [
                    line.strip()
                    for line in Path(uid_file_path).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except OSError as exc:
                raise RuntimeError(f"Could not read worker sample UID file: {uid_file_path}") from exc
        else:
            raw_uids = str(args.worker_sample_uids or "").strip()
            if raw_uids:
                sample_uids = [item.strip() for item in raw_uids.split(",") if item.strip()]
            else:
                sample_uids = [str(args.worker_sample_uid or "").strip()]
        selected_samples = [
            _find_sample_by_uid(inventories, sample_uid)
            for sample_uid in sample_uids
            if sample_uid
        ]
    paired_ablation_role = str(args.worker_paired_ablation_role or "").strip().lower()
    paired_ablation_roles_raw = str(args.worker_paired_ablation_roles or "").strip().lower()
    paired_ablation_roles = [
        item.strip()
        for item in (paired_ablation_roles_raw or paired_ablation_role).replace(";", ",").split(",")
        if item.strip()
    ]
    paired_ablation_root = str(args.worker_paired_ablation_root or "").strip()
    delivery_ablations = (os.environ.get("MIX_PAIRED_ABLATIONS") == "1"
                          and os.environ.get("CFV_ONLINE_EVOLUTION_REPLAY") != "1")
    paired_tools = []
    if delivery_ablations:
        paired_ablation_roles = paired_ablation_roles or ["observer", "reflector"]
        paired_ablation_root = paired_ablation_root or str(config.workspaces_root / "_ablations")
        requested_tools = os.environ.get("MIX_PAIRED_ABLATION_TOOLS", "all")
        baseline_path = config.workspaces_root / "_audit" / "tool_library_baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        active_tools = [tool["name"] for tool in baseline["tools"]]
        paired_tools = active_tools if requested_tools == "all" else [name.strip() for name in requested_tools.split(",") if name.strip()]
        if not paired_tools or set(paired_tools) - set(active_tools):
            raise ValueError("Delivery ablation requires active executable tools from the frozen baseline")
    if bool(paired_ablation_roles) != bool(paired_ablation_root):
        raise RuntimeError(
            "Paired ablation requires an ablation role and "
            "--worker-paired-ablation-root."
        )
    unsupported_ablation_roles = sorted(set(paired_ablation_roles) - {"observer", "reflector"})
    if unsupported_ablation_roles:
        raise RuntimeError(f"Unsupported paired ablation role(s): {unsupported_ablation_roles}")
    if len(set(paired_ablation_roles)) != len(paired_ablation_roles):
        raise RuntimeError(f"Duplicate paired ablation roles are not allowed: {paired_ablation_roles}")

    config_workspace_root = config.workspaces_root.resolve()
    ablation_workspace_root = Path(paired_ablation_root).resolve() if paired_ablation_root else None
    original_disabled_roles = os.environ.pop("TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES", None)
    try:
        for sample_index, sample in enumerate(selected_samples):
            runtime_snapshot = _sync_online_runtime_snapshot()
            if runtime_snapshot.get("restart_required") and sample_index > 0:
                # A source edit cannot be hot-swapped after run_e2e imported
                # its modules. The parent group worker treats this explicit
                # boundary as a clean restart and retries only unexecuted UIDs.
                print(
                    "[ONLINE_RUNTIME_RESTART] "
                    + json.dumps(runtime_snapshot, ensure_ascii=False),
                    flush=True,
                )
                return ONLINE_RUNTIME_RESTART_EXIT_CODE
            bundle_marker = sample.workspace_root / ".paired_pending"
            if paired_ablation_roles or paired_tools:
                bundle_marker.parent.mkdir(parents=True, exist_ok=True)
                bundle_marker.write_text("main result awaits paired execution", encoding="utf-8")
            result_item = run_single_sample(sample=sample, config=config, no_cache=bool(args.no_cache))
            if paired_ablation_roles and ablation_workspace_root is not None:
                try:
                    relative_workspace = sample.workspace_root.resolve().relative_to(config_workspace_root)
                except ValueError:
                    relative_workspace = Path(sample.dataset_name) / sample.workspace_root.name
                for ablation_role in paired_ablation_roles:
                    ablation_sample = replace(
                        sample,
                        workspace_root=ablation_workspace_root / ablation_role / relative_workspace,
                    )
                    os.environ["TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES"] = ablation_role
                    try:
                        ablation_result = run_single_sample(
                            sample=ablation_sample,
                            config=config,
                            no_cache=bool(args.no_cache),
                        )
                    finally:
                        os.environ.pop("TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES", None)
                    # Counterfactuals must never overwrite the main sample's
                    # live checkpoint merely because they share its UID.
                    print("[ABLATION_RESULT_JSON]" + json.dumps(ablation_result, ensure_ascii=False), flush=True)
            if paired_tools and ablation_workspace_root is not None:
                relative_workspace = sample.workspace_root.resolve().relative_to(config_workspace_root)
                for tool in paired_tools:
                    ablation_sample = replace(sample, workspace_root=ablation_workspace_root / ("tool_" + tool) / relative_workspace)
                    ablation_config = replace(config, pipeline_options={**config.pipeline_options,
                        "disabled_tool_names": list(dict.fromkeys([*(config.pipeline_options.get("disabled_tool_names") or []), tool]))})
                    ablation_result = run_single_sample(sample=ablation_sample, config=ablation_config,
                                                       no_cache=bool(args.no_cache))
                    print("[ABLATION_RESULT_JSON]" + json.dumps(ablation_result, ensure_ascii=False), flush=True)
            bundle_marker.unlink(missing_ok=True)
            print(RUN_RESULT_LOG_MARKER + json.dumps(result_item, ensure_ascii=False), flush=True)
    finally:
        os.environ.pop("TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES", None)
        if original_disabled_roles is not None:
            os.environ["TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES"] = original_disabled_roles
    return 0


def main() -> None:
    args = parse_args()
    if (
        str(args.worker_sample_uid or "").strip()
        or str(args.worker_sample_uids or "").strip()
        or str(args.worker_sample_uids_file or "").strip()
        or str(args.worker_sample_spec_file or "").strip()
    ):
        worker_exit_code = _run_worker_mode(args)
        if worker_exit_code:
            raise SystemExit(worker_exit_code)
        return

    configure_logging()
    config = load_config(Path(args.config))
    inventories = load_inventories(config)
    if not inventories:
        raise RuntimeError("No enabled datasets found in mix config.")
    dataset_order = _inventory_dataset_order(inventories)

    progress_payload = load_progress(config.progress_path)
    if args.force_rerun and not args.plan_only:
        # Preserve the old checkpoint for comparison/recovery, then make the
        # formal run's denominator explicit.  Existing per-sample artifacts
        # are intentionally not deleted.
        archive_path = config.progress_path.with_name(
            config.progress_path.stem + "_legacy_before_force_rerun" + config.progress_path.suffix
        )
        if config.progress_path.exists() and not archive_path.exists():
            shutil.copy2(config.progress_path, archive_path)
        progress_payload = {
            "kind": "mix_progress",
            "layout_version": 1,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "records": {},
        }
        save_progress(config.progress_path, progress_payload)
        LOGGER.warning(
            "--force-rerun reset formal progress; archived legacy checkpoint at %s",
            archive_path,
        )
    recovered_result_count = 0
    # A formal force-rerun is a new measurement boundary.  Per-sample
    # checkpoints are useful for ordinary crash recovery, but importing them
    # here would silently turn ``--force-rerun`` into a partial resume and can
    # reintroduce results produced under an older scoring or agent contract.
    if not args.force_rerun:
        for sample in _iter_all_samples(inventories):
            if has_participated(progress_payload, sample.sample_uid):
                continue
            recovered = import_existing_participated_result(
                progress_payload,
                sample_uid=sample.sample_uid,
                workspace_root=sample.workspace_root,
            )
            if recovered is not None:
                recovered_result_count += 1
    if recovered_result_count and not args.plan_only:
        save_progress(config.progress_path, progress_payload)
    if recovered_result_count:
        LOGGER.info(
            "Recovered %s participated sample result(s) from per-sample checkpoints.",
            recovered_result_count,
        )
    imported_skip_true_count = 0
    if args.skip_true:
        imported_skip_true_count = import_skip_true_state(
            progress_payload=progress_payload,
            inventories=inventories,
        )
        save_progress(config.progress_path, progress_payload)

    explicit_sample_uids = [
        item.strip()
        for item in str(args.sample_uids or "").split(",")
        if item.strip()
    ]
    if explicit_sample_uids:
        selected_samples = build_explicit_sample_queue(
            inventories=inventories,
            sample_uids=explicit_sample_uids,
        )
        if args.max_samples is not None and args.max_samples > 0:
            selected_samples = selected_samples[: args.max_samples]
    else:
        selected_samples = build_execution_queue(
            inventories=inventories,
            progress_payload=progress_payload,
            max_samples=args.max_samples,
        )

    if not selected_samples:
        raise RuntimeError("No runnable Mix samples remain after applying progress state and CLI filters.")

    if args.plan_only:
        for index, sample in enumerate(selected_samples, start=1):
            LOGGER.info(
                "Batch plan %d/%d dataset=%s subset=%s sample_index=%s sample_uid=%s",
                index,
                len(selected_samples),
                sample.dataset_name,
                sample.subset_name,
                sample.sample_index,
                sample.sample_uid,
            )
        return

    # The production Mix contract enables online Evolver by default. A
    # missing controller must fail closed instead of silently running a
    # benchmark that appears complete while never adapting its harness.
    configured_online = config.pipeline_options.get("online_evolution", True)
    if isinstance(configured_online, str):
        configured_online = configured_online.strip().lower() in {"1", "true", "yes", "on"}
    online_requested = _env_bool("ONLINE_EVOLUTION", bool(configured_online))
    if online_requested and OnlineEvolutionController is None:
        detail = (
            f" {type(_ONLINE_EVOLUTION_IMPORT_ERROR).__name__}: {_ONLINE_EVOLUTION_IMPORT_ERROR}"
            if _ONLINE_EVOLUTION_IMPORT_ERROR is not None
            else ""
        )
        raise RuntimeError(
            "Online Evolver is enabled but its controller could not be imported."
            " Refusing to continue without the configured self-evolution loop."
            + detail
        )

    baseline_path, baseline_payload = _prepare_tool_library_baseline(config)

    online_controller = None
    if OnlineEvolutionController is not None:
        evolution_config_path = Path(
            os.environ.get("EVOLUTION_CONFIG")
            or (_PROJECT_ROOT / "RL" / "config" / "evolution.yaml")
        ).expanduser().resolve()
        online_controller = OnlineEvolutionController(
            config_path=config.config_path,
            workspaces_root=config.workspaces_root,
            evolution_config=evolution_config_path,
            output_root=config.workspaces_root / "_evolution_audit",
            baseline_path=baseline_path,
            visible_device_id=(
                _visible_cuda_device_ids()[0]
                if _visible_cuda_device_ids()
                else "0"
            ),
            options=config.pipeline_options,
        )
        if online_controller.enabled:
            LOGGER.info(
                "Online Evolver enabled: trigger_every=%d window=%d group_size=%d; "
                "accepted snapshots affect subsequent worker groups only",
                online_controller.trigger_every,
                online_controller.window_size,
                online_controller.group_size,
            )

    visible_device_ids = _visible_cuda_device_ids()
    effective_parallel_samples = _resolve_effective_parallel_samples(
        configured_parallel_samples=config.parallel_samples,
        cli_parallel_samples=args.parallel_samples,
        selected_count=len(selected_samples),
        visible_device_ids=visible_device_ids,
        allow_same_device_parallel=config.allow_same_device_parallel,
    )
    reuse_worker_samples = _resolve_reuse_worker_samples(
        configured_size=config.reuse_worker_samples,
        cli_size=args.reuse_worker_samples,
        selected_count=len(selected_samples),
    )
    if online_controller is not None and online_controller.enabled:
        # Online candidate replay must obtain the accelerator admission slot
        # between worker groups. Keep model reuse, but bound one group to the
        # online checkpoint and use one resident worker per device so a replay
        # cannot be starved behind a 6k-sample persistent process.
        reuse_worker_samples = min(reuse_worker_samples, online_controller.group_size, online_controller.trigger_every)
    execution_config = replace(config, reuse_worker_samples=reuse_worker_samples)
    LOGGER.info(
        "Mix batch execution resolved parallel_samples=%s reuse_worker_samples=%s asr_beam_size=%s asr_best_of=%s asr_condition_on_previous_text=%s asr_vad_filter=%s caption_batch_size=%s caption_max_frames=%s caption_max_new_tokens=%s visible_cuda_devices=%s selected_samples=%s",
        effective_parallel_samples,
        reuse_worker_samples,
        config.asr_beam_size,
        config.asr_best_of,
        config.asr_condition_on_previous_text,
        config.asr_vad_filter,
        config.caption_batch_size,
        config.caption_max_frames,
        config.caption_max_new_tokens,
        list(visible_device_ids) if visible_device_ids else ["cpu"],
        len(selected_samples),
    )
    resume_progress_line = build_resume_progress_line(progress_payload, inventories, dataset_order)
    if resume_progress_line:
        LOGGER.info("%s", resume_progress_line)

    def observe_online_result(payload: Mapping[str, Any]) -> None:
        if online_controller is None or not online_controller.enabled:
            return
        try:
            online_controller.observe_result(payload)
        except Exception as exc:
            # Evolution is advisory to the benchmark scheduler. A malformed
            # trajectory must never discard a committed Mix result or stop the
            # baseline run; the controller records its own error state.
            LOGGER.exception("Online Evolver observation failed: %s", exc)

    incomplete_group_uids: List[str] = []
    incomplete_return_code = 1
    if effective_parallel_samples <= 1 and reuse_worker_samples <= 1:
        cumulative_results = load_result_items_from_progress(progress_payload)
        worker_device_id = visible_device_ids[0] if visible_device_ids else ""
        for index, sample in enumerate(selected_samples, start=1):
            LOGGER.info(
                "Launching sample %d/%d sample_uid=%s launcher_log=%s",
                index,
                len(selected_samples),
                sample.sample_uid,
                sample.workspace_root.resolve() / "launcher.log",
            )
            # Keep even the one-GPU/serial path in a child process.  The local
            # multimodal runtime may terminate with SIGABRT/SIGKILL inside a
            # CUDA proxy; running it in the scheduler process would otherwise
            # lose the batch checkpoint before the sample can be marked failed.
            try:
                result_item = _run_mix_sample_worker(
                    config=config,
                    config_path=config.config_path,
                    sample=sample,
                    no_cache=bool(args.no_cache),
                    visible_device_id=worker_device_id,
                    caption_batch_size=config.caption_batch_size,
                    caption_max_frames=config.caption_max_frames,
                    caption_max_new_tokens=config.caption_max_new_tokens,
                )
            except Exception as exc:
                LOGGER.exception(
                    "Mix worker failed sample_uid=%s visible_cuda_device=%s",
                    sample.sample_uid,
                    worker_device_id,
                )
                # No result payload means the sample was not safely measured.
                # Leave it pending for a later invocation rather than turning
                # a low-level worker/runtime failure into benchmark data.
                incomplete_group_uids.append(str(sample.sample_uid))
                continue
            upsert_progress_record(progress_payload, result_item)
            save_progress(config.progress_path, progress_payload)
            observe_online_result(result_item)
            if online_controller is not None and online_controller.enabled:
                online_controller.on_boundary()
            cumulative_results = load_result_items_from_progress(progress_payload)
            progress_line = build_dataset_progress_line(progress_payload, inventories, sample.dataset_name)
            if progress_line:
                LOGGER.info("%s %s", progress_line, build_acc_line(cumulative_results, dataset_order))
            else:
                LOGGER.info("%s", build_acc_line(cumulative_results, dataset_order))
    elif effective_parallel_samples <= 1:
        cumulative_results = load_result_items_from_progress(progress_payload)
        worker_device_id = visible_device_ids[0] if visible_device_ids else ""
        for group_start in range(0, len(selected_samples), reuse_worker_samples):
            sample_group = selected_samples[group_start : group_start + reuse_worker_samples]
            committed_group_uids: Set[str] = set()

            def checkpoint_reusable_result(payload: Mapping[str, Any]) -> None:
                sample_uid = str(payload.get("sample_uid") or "").strip()
                status = str(payload.get("status") or "").strip().lower()
                if (
                    not sample_uid
                    or sample_uid in committed_group_uids
                    or status not in {"completed", "wrong", "unanswerable", "failed"}
                    or (
                        _payload_is_worker_oom(payload)
                        and not payload.get("worker_oom_retry_exhausted")
                    )
                ):
                    return
                committed_group_uids.add(sample_uid)
                upsert_progress_record(progress_payload, dict(payload))
                save_progress(config.progress_path, progress_payload)
                observe_online_result(payload)
                current_results = load_result_items_from_progress(progress_payload)
                progress_line = build_dataset_progress_line(
                    progress_payload,
                    inventories,
                    str(payload.get("dataset_name") or ""),
                )
                if progress_line:
                    LOGGER.info(
                        "Reusable group live checkpoint %s %s",
                        progress_line,
                        build_acc_line(current_results, dataset_order),
                    )
                else:
                    LOGGER.info(
                        "Reusable group live checkpoint %s",
                        build_acc_line(current_results, dataset_order),
                    )

            LOGGER.info(
                "Launching reusable worker group %d-%d/%d first_sample_uid=%s last_sample_uid=%s",
                group_start + 1,
                group_start + len(sample_group),
                len(selected_samples),
                sample_group[0].sample_uid,
                sample_group[-1].sample_uid,
            )
            result_items, return_code, worker_log_path = _run_mix_sample_group_worker(
                config=config,
                config_path=config.config_path,
                samples=sample_group,
                no_cache=bool(args.no_cache),
                visible_device_id=worker_device_id,
                caption_batch_size=config.caption_batch_size,
                caption_max_frames=config.caption_max_frames,
                caption_max_new_tokens=config.caption_max_new_tokens,
                on_result=checkpoint_reusable_result,
            )
            if online_controller is not None and online_controller.enabled:
                online_controller.on_boundary()
            incomplete_return_code = int(return_code or 1)
            result_by_uid = {
                str(item.get("sample_uid") or ""): item
                for item in result_items
                if isinstance(item, dict)
            }
            group_dirty = False
            group_committed_count = 0
            for sample in sample_group:
                result_item = result_by_uid.get(sample.sample_uid)
                if result_item is None:
                    if int(return_code) != 0:
                        # A reusable worker may exit after a low-level runtime
                        # abort or service failure.  Do not manufacture a
                        # failure result for every later sample in the group:
                        # those samples were never executed and must remain
                        # pending so a subsequent invocation can retry them.
                        incomplete_group_uids.append(str(sample.sample_uid))
                        continue
                    result_item = _build_scheduler_failure_result(
                        sample=sample,
                        error_text=(
                            "Reusable Mix worker did not produce a per-sample result. "
                            f"return_code={return_code} worker_log={worker_log_path}"
                        ),
                    )
                if str(result_item.get("sample_uid") or "") in committed_group_uids:
                    continue
                upsert_progress_record(progress_payload, result_item)
                committed_group_uids.add(str(result_item.get("sample_uid") or ""))
                observe_online_result(result_item)
                group_dirty = True
                group_committed_count += 1
                # A reusable worker already persists each sample artifact.  A
                # bounded parent checkpoint stride keeps crash recovery while
                # avoiding one growing JSON rewrite and full inventory scan
                # for every item in a 6920-sample group.
                if group_committed_count % 32 == 0:
                    save_progress(config.progress_path, progress_payload)
                    cumulative_results = load_result_items_from_progress(progress_payload)
                    LOGGER.info(
                        "Reusable group checkpoint committed=%d/%d %s",
                        group_committed_count,
                        len(sample_group),
                        build_acc_line(cumulative_results, dataset_order),
                    )
            if group_dirty:
                save_progress(config.progress_path, progress_payload)
                cumulative_results = load_result_items_from_progress(progress_payload)
                LOGGER.info(
                    "Reusable group final checkpoint committed=%d/%d %s",
                    group_committed_count,
                    len(sample_group),
                    build_acc_line(cumulative_results, dataset_order),
                )
    else:
        wave_size = (effective_parallel_samples * reuse_worker_samples
                     if online_controller is not None and online_controller.enabled
                     else max(1, len(selected_samples)))
        for wave_start in range(0, len(selected_samples), wave_size):
            _execute_parallel_samples(
                selected_samples=selected_samples[wave_start:wave_start + wave_size],
                config=execution_config,
                no_cache=bool(args.no_cache),
                progress_payload=progress_payload,
                inventories=inventories,
                effective_parallel_samples=effective_parallel_samples,
                visible_device_ids=visible_device_ids,
                on_result=observe_online_result,
            )
            if online_controller is not None and online_controller.enabled:
                online_controller.on_boundary()

    if online_controller is not None and online_controller.enabled:
        # Flush a final partial evolution window after all selected samples;
        # the regular boundaries above remain the path that affects future
        # samples during the run.
        online_controller.close(wait=True)

    cumulative_results = load_result_items_from_progress(progress_payload)
    result_map = {
        str(item.get("sample_uid") or ""): item
        for item in cumulative_results
        if isinstance(item, dict)
    }
    selection_progress = build_selection_progress(
        progress_payload=progress_payload,
        inventories=inventories,
        selected_samples=selected_samples,
        imported_skip_true_count=imported_skip_true_count,
        max_samples=args.max_samples,
        parallel_samples=effective_parallel_samples,
        skip_true=bool(args.skip_true),
        dataset_order=dataset_order,
    )
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config_path": str(config.config_path),
        "results_output_path": str(config.results_path),
        "progress_path": str(config.progress_path),
        "shared_preprocess_root": str(config.shared_preprocess_root),
        "ffmpeg_binary": config.ffmpeg_binary,
        "parallel_samples": effective_parallel_samples,
        "reuse_worker_samples": reuse_worker_samples,
        "caption_runtime": {
            "batch_size": int(config.caption_batch_size),
            "max_frames": int(config.caption_max_frames),
            "max_new_tokens": int(config.caption_max_new_tokens),
            "share_planner_visual_runtime": (
                str(
                    config.pipeline_options.get(
                        "share_planner_visual_runtime",
                        os.environ.get("TOOLSTAR_SHARE_PLANNER_VISUAL_RUNTIME", "1"),
                    )
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "fallback_policy": "batch_failure_or_low_information_output_retries_single_segment",
        },
        "asr_runtime": {
            "beam_size": int(config.asr_beam_size),
            "best_of": int(config.asr_best_of),
            "condition_on_previous_text": bool(config.asr_condition_on_previous_text),
            "vad_filter": bool(config.asr_vad_filter),
        },
        "allow_same_device_parallel": bool(config.allow_same_device_parallel),
        "accelerator_queue": {
            "enabled": bool(config.accelerator_queue_enabled),
            "policy": "persistent_worker_admission_with_memory_wait_and_bounded_oom_retry",
            "root": str(config.accelerator_queue_root or os.environ.get("TOOL_STAR_SHARED_ACCELERATOR_QUEUE_ROOT") or ""),
            "sample_worker_admission_capacity": int(config.sample_worker_admission_capacity),
            "resident_model_workers_per_device": int(config.resident_model_workers_per_device),
            "provider_capacity": _provider_queue_capacity_from_config(
                {
                    "pipeline_options": config.pipeline_options,
                    "sample_worker_admission_capacity": config.sample_worker_admission_capacity,
                }
            ),
            "memory_guard": bool(config.accelerator_queue_memory_guard),
            "min_free_gib": float(config.accelerator_queue_min_free_gib),
            "reserve_gib": float(config.accelerator_queue_reserve_gib),
            "persistent_worker_start_stagger_seconds": float(config.persistent_worker_start_stagger_seconds),
            "wait_seconds_total": round(
                sum(float(item.get("scheduler_accelerator_queue_wait_seconds") or 0.0) for item in cumulative_results),
                3,
            ),
        },
        "visible_cuda_devices": list(visible_device_ids),
        "tool_library_baseline": {
            "path": str(baseline_path),
            "baseline_id": str(baseline_payload.get("baseline_id") or ""),
            "schema_digest": str(baseline_payload.get("schema_digest") or ""),
            "source_digest": str(baseline_payload.get("source_digest") or ""),
            "tool_count": int(baseline_payload.get("tool_count") or 0),
        },
        "selection_progress": selection_progress,
        "summary_by_dataset": build_summary_by_dataset(cumulative_results, dataset_order),
        "subset_summaries_by_dataset": build_subset_summaries(cumulative_results, dataset_order),
        "results": cumulative_results,
        "online_evolution": (
            online_controller.summary() if online_controller is not None else {
                "enabled": False,
                "status": "unavailable",
            }
        ),
    }

    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    videomme_inventory = next((item for item in inventories if item.dataset_name == "videomme"), None)
    if videomme_inventory is not None:
        videomme_output = build_videomme_official_output(videomme_inventory, result_map)
        config.videomme_official_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.videomme_official_output_path, "w", encoding="utf-8") as handle:
            json.dump(videomme_output, handle, ensure_ascii=False, indent=2)

    LOGGER.info("Saved Mix results to %s", config.results_path)
    if incomplete_group_uids:
        if online_controller is not None:
            online_controller.close(wait=True)
        preview = ", ".join(incomplete_group_uids[:5])
        LOGGER.error(
            "Reusable worker exited before producing %d sample result(s); they remain pending for retry. preview=%s",
            len(incomplete_group_uids),
            preview,
        )
        raise RuntimeError(
            "Reusable Mix worker failed before completing the selected group; "
            f"pending_samples={len(incomplete_group_uids)} return_code={incomplete_return_code}."
        )


if __name__ == "__main__":
    main()
