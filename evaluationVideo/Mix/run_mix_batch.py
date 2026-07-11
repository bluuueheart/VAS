import argparse
import asyncio
import concurrent.futures
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

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
SUPPORTED_DATASET_ORDER = ("mvbench", "mlvu", "videomme")
DEFAULT_DATASET_ORDER = ("mvbench", "mlvu", "videomme")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mixed MVBench/MLVU/Video-MME evaluator.")
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
        "--skip_true",
        "--skip-true",
        dest="skip_true",
        action="store_true",
        help="Import already-correct existing per-sample results into progress state before scheduling.",
    )
    parser.add_argument(
        "--worker-sample-uid",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-device-id",
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
        ffmpeg_binary=str(payload.get("ffmpeg_binary") or "ffmpeg"),
        parallel_samples=_parse_parallel_samples_value(payload.get("parallel_samples")),
        datasets=normalized_datasets,
        dataset_order=_parse_dataset_order(payload.get("dataset_order"), normalized_datasets),
        pipeline_options=_load_pipeline_options(config_dir, payload.get("pipeline_options")),
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


def _load_run_e2e_module():
    global _RUN_E2E_MODULE
    if _RUN_E2E_MODULE is not None:
        return _RUN_E2E_MODULE
    project_root = _retool_video_root().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    _RUN_E2E_MODULE = importlib.import_module("run_e2e")
    return _RUN_E2E_MODULE


def load_inventories(config: BatchConfig) -> List[DatasetInventory]:
    mix_dir = config.config_path.parent
    inventories: List[DatasetInventory] = []
    for dataset_name in config.dataset_order:
        dataset_config = config.datasets.get(dataset_name, {})
        if not dataset_config or not bool(dataset_config.get("enabled", True)):
            continue
        if dataset_name == "mvbench":
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
) -> int:
    if selected_count <= 1:
        return 1
    max_parallel = len(list(visible_device_ids)) if visible_device_ids else 1
    requested = cli_parallel_samples if cli_parallel_samples is not None else configured_parallel_samples
    if requested is None:
        return max(1, min(selected_count, max_parallel))
    if requested <= 0:
        return max(1, min(selected_count, max_parallel))
    return max(1, min(selected_count, requested, max_parallel))


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


def _bind_worker_to_visible_cuda_device(visible_device_id: str) -> None:
    normalized_device = str(visible_device_id or "").strip()
    if not normalized_device:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized_device
    for key in list(os.environ.keys()):
        if key.startswith("TOOLSTAR_") and key.endswith("_DEVICE_ID"):
            os.environ.pop(key, None)
    os.environ["TOOLSTAR_DEVICE_ID"] = "0"


def _run_mix_sample_worker(
    *,
    config_path: Path,
    sample_uid: str,
    no_cache: bool,
    visible_device_id: str,
) -> Dict[str, Any]:
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
    process = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    payload = _extract_worker_result_payload(process.stdout or "")
    if payload is not None and process.returncode == 0:
        return payload
    tail_lines = (process.stdout or "").splitlines()[-40:]
    tail = "\n".join(tail_lines).strip()
    if payload is not None:
        payload.setdefault("status", "failed")
        payload.setdefault("error", f"Worker process exited with return code {process.returncode}.")
        return payload
    raise RuntimeError(
        f"Mix sample worker failed sample_uid={sample_uid} visible_device_id={visible_device_id} "
        f"return_code={process.returncode}. Worker output tail:\n{tail}"
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
        "acc": 0.0,
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
    run_e2e_module = _load_run_e2e_module()
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
        final_status = "completed" if metrics_payload["answer_correct"] else "wrong"
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        result_item = {
            "sample_uid": sample.sample_uid,
            "dataset_name": sample.dataset_name,
            "subset_name": sample.subset_name,
            "status": final_status,
            "acc": 1.0 if metrics_payload["answer_correct"] else 0.0,
            "tool_use_num": metrics_payload["tool_use_num"],
            "iter_num": metrics_payload["iter_num"],
            "tool_success_rate": metrics_payload["tool_success_rate"],
            "successful_tool_calls": metrics_payload["successful_tool_calls"],
            "planner_iterations": metrics_payload["planner_iterations"],
            "subagent_iterations": metrics_payload["subagent_iterations"],
            "predicted_label": metrics_payload["predicted_label"],
            "ground_truth": sample.ground_truth,
            "ground_truth_option": sample.ground_truth_option,
            "ground_truth_labels": metrics_payload["ground_truth_labels"],
            "planner_success": bool(run_result_payload.get("success", False)),
            "raw_final_answer": run_result_payload.get("agent", {}).get("final_answer", "")
            if isinstance(run_result_payload.get("agent"), dict)
            else "",
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
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        result_item = {
            "sample_uid": sample.sample_uid,
            "dataset_name": sample.dataset_name,
            "subset_name": sample.subset_name,
            "status": "failed",
            "acc": 0.0,
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
        parts.append(f"{dataset_name}={correct:.0f}/{count} ({100.0 * _safe_divide(correct, count):.2f}%)")
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


def _execute_parallel_samples(
    *,
    selected_samples: Sequence[MixSample],
    config: BatchConfig,
    no_cache: bool,
    progress_payload: MutableMapping[str, Any],
    inventories: Sequence[DatasetInventory],
    effective_parallel_samples: int,
    visible_device_ids: Sequence[str],
) -> None:
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
                    config_path=config.config_path,
                    sample_uid=sample.sample_uid,
                    no_cache=no_cache,
                    visible_device_id=visible_device_id,
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
                cumulative_results = load_result_items_from_progress(progress_payload)
                progress_line = build_dataset_progress_line(progress_payload, inventories, sample.dataset_name)
                if progress_line:
                    LOGGER.info("%s %s", progress_line, build_acc_line(cumulative_results, dataset_order))
                else:
                    LOGGER.info("%s", build_acc_line(cumulative_results, dataset_order))


def _run_worker_mode(args: argparse.Namespace) -> None:
    if str(args.worker_device_id or "").strip():
        _bind_worker_to_visible_cuda_device(str(args.worker_device_id))
    config = load_config(Path(args.config))
    inventories = load_inventories(config)
    sample = _find_sample_by_uid(inventories, str(args.worker_sample_uid))
    result_item = run_single_sample(sample=sample, config=config, no_cache=bool(args.no_cache))
    print(RUN_RESULT_LOG_MARKER + json.dumps(result_item, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    if str(args.worker_sample_uid or "").strip():
        _run_worker_mode(args)
        return

    configure_logging()
    config = load_config(Path(args.config))
    inventories = load_inventories(config)
    if not inventories:
        raise RuntimeError("No enabled datasets found in mix config.")
    dataset_order = _inventory_dataset_order(inventories)

    progress_payload = load_progress(config.progress_path)
    imported_skip_true_count = 0
    if args.skip_true:
        imported_skip_true_count = import_skip_true_state(
            progress_payload=progress_payload,
            inventories=inventories,
        )
        save_progress(config.progress_path, progress_payload)

    selected_samples = build_execution_queue(
        inventories=inventories,
        progress_payload=progress_payload,
        max_samples=args.max_samples,
    )

    if not selected_samples:
        raise RuntimeError("No runnable Mix samples remain after applying progress state and CLI filters.")

    visible_device_ids = _visible_cuda_device_ids()
    effective_parallel_samples = _resolve_effective_parallel_samples(
        configured_parallel_samples=config.parallel_samples,
        cli_parallel_samples=args.parallel_samples,
        selected_count=len(selected_samples),
        visible_device_ids=visible_device_ids,
    )
    LOGGER.info(
        "Mix batch execution resolved parallel_samples=%s visible_cuda_devices=%s selected_samples=%s",
        effective_parallel_samples,
        list(visible_device_ids) if visible_device_ids else ["cpu"],
        len(selected_samples),
    )
    resume_progress_line = build_resume_progress_line(progress_payload, inventories, dataset_order)
    if resume_progress_line:
        LOGGER.info("%s", resume_progress_line)

    if effective_parallel_samples <= 1:
        cumulative_results = load_result_items_from_progress(progress_payload)
        for index, sample in enumerate(selected_samples, start=1):
            LOGGER.info(
                "Launching sample %d/%d sample_uid=%s launcher_log=%s",
                index,
                len(selected_samples),
                sample.sample_uid,
                sample.workspace_root.resolve() / "launcher.log",
            )
            result_item = run_single_sample(sample=sample, config=config, no_cache=bool(args.no_cache))
            upsert_progress_record(progress_payload, result_item)
            save_progress(config.progress_path, progress_payload)
            cumulative_results = load_result_items_from_progress(progress_payload)
            progress_line = build_dataset_progress_line(progress_payload, inventories, sample.dataset_name)
            if progress_line:
                LOGGER.info("%s %s", progress_line, build_acc_line(cumulative_results, dataset_order))
            else:
                LOGGER.info("%s", build_acc_line(cumulative_results, dataset_order))
    else:
        _execute_parallel_samples(
            selected_samples=selected_samples,
            config=config,
            no_cache=bool(args.no_cache),
            progress_payload=progress_payload,
            inventories=inventories,
            effective_parallel_samples=effective_parallel_samples,
            visible_device_ids=visible_device_ids,
        )

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
        "visible_cuda_devices": list(visible_device_ids),
        "selection_progress": selection_progress,
        "summary_by_dataset": build_summary_by_dataset(cumulative_results, dataset_order),
        "subset_summaries_by_dataset": build_subset_summaries(cumulative_results, dataset_order),
        "results": cumulative_results,
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


if __name__ == "__main__":
    main()
