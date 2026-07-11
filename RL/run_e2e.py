"""
Tool Star RL - End-to-end runner.
Video preprocessing + ReAct query pipeline.
"""
from __future__ import annotations
import asyncio
import argparse
import base64
import binascii
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import struct
import sys
import time
import urllib.request
import urllib.error
import zlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from config.device_config import initialize_device, detect_device, get_accelerator_snapshot
from core.sidecar_pool import OpenAIEndpointPool, format_endpoint_list
from runtime.session import runtime_session
from toolkit.base import infer_primary_tool_type
from toolkit.impl.visual_prompts import truncate_think_tail_preserving_outer_text


# Configure base logging (console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


RUN_RESULT_LOG_MARKER = "[RUN_RESULT_JSON]"
_NON_PLANNER_RUNTIME_READY_SIGNATURES: set[str] = set()
_PIPELINE_RUNTIME_PROFILE_DEFAULT = "default"
_PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL = "evaluation_local"


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("run_e2e requires optional dependency: torch") from exc
    return torch


def _parse_export_env_file(env_file: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not env_file.exists():
        return parsed
    for raw_line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        assignment = line[len("export ") :].strip()
        if "=" not in assignment:
            continue
        key, value = assignment.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[key] = value
    return parsed


def _load_sidecar_runtime_env(env_file: Path) -> Dict[str, str]:
    loaded = _parse_export_env_file(env_file)
    for key, value in loaded.items():
        os.environ[key] = value
    return loaded


def _normalize_provider_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "": "auto",
        "sidecar": "service",
        "remote": "service",
        "api": "service",
        "openai": "service",
        "local_process": "process",
        "subprocess": "process",
        "disable": "disabled",
        "off": "disabled",
        "false": "disabled",
        "0": "disabled",
    }
    return aliases.get(normalized, normalized)


def _provider_requests_service(provider_value: Any) -> bool:
    return _normalize_provider_mode(provider_value) in {"auto", "service"}


def _normalize_runtime_profile(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "": _PIPELINE_RUNTIME_PROFILE_DEFAULT,
        "default": _PIPELINE_RUNTIME_PROFILE_DEFAULT,
        "standard": _PIPELINE_RUNTIME_PROFILE_DEFAULT,
        "eval": _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL,
        "evaluation": _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL,
        "evaluation_local": _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL,
        "evaluation-local": _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL,
        "local_evaluation": _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL,
        "local-evaluation": _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL,
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in {
        _PIPELINE_RUNTIME_PROFILE_DEFAULT,
        _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL,
    }:
        logger.warning("Unknown execute_pipeline runtime_profile=%s; using default.", value)
        return _PIPELINE_RUNTIME_PROFILE_DEFAULT
    return resolved


def _evaluation_local_runtime_env_overrides() -> tuple[Dict[str, str], List[str]]:
    overrides = {
        "TOOL_STAR_RUNTIME_PROFILE": _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL,
        "TOOL_STAR_DISABLE_E2E_SIDECAR_AUTOSTART": "1",
        "TOOL_STAR_ENABLE_NON_PLANNER_SIDECARS": "0",
        "TOOL_STAR_NON_PLANNER_MODEL_PROVIDER": "transformers",
        "TOOL_STAR_AUX_MODEL_PROVIDER": "transformers",
        "TOOL_STAR_SUBAGENT_MODEL_PROVIDER": "transformers",
        "TOOL_STAR_SUBAGENT_SERVICE_MODEL_PROVIDER": "transformers",
        "TOOL_STAR_PREPROCESS_MODEL_PROVIDER": "transformers",
        "TOOL_STAR_QWEN_VL_MODEL_PROVIDER": "transformers",
        "TOOL_STAR_OCR_MODEL_PROVIDER": "transformers",
        "TOOL_STAR_NON_PLANNER_EMBEDDING_PROVIDER": "local",
        "TOOL_STAR_TOOL_SEARCH_EMBEDDING_PROVIDER": "local",
        "TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_PROVIDER": "local",
        "TOOL_STAR_PREPROCESS_EMBEDDING_PROVIDER": "local",
    }
    clear_keys: List[str] = []
    for prefix in (
        "NON_PLANNER",
        "AUX",
        "SUBAGENT",
        "SUBAGENT_SERVICE",
        "PREPROCESS",
        "QWEN_VL",
        "OCR",
    ):
        for suffix in ("API_BASES", "API_BASE", "SERVICE_API_BASES", "SERVICE_API_BASE"):
            clear_keys.append(f"TOOL_STAR_{prefix}_{suffix}")
    for prefix in (
        "NON_PLANNER_EMBEDDING",
        "TOOL_SEARCH_EMBEDDING",
        "ROLLOUT_GROUNDING_EMBEDDING",
        "PREPROCESS_EMBEDDING",
    ):
        for suffix in ("API_BASES", "API_BASE"):
            clear_keys.append(f"TOOL_STAR_{prefix}_{suffix}")
    clear_keys.extend(
        [
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ]
    )
    return overrides, clear_keys


@contextmanager
def _temporary_environment_overrides(
    overrides: Dict[str, str],
    *,
    clear_keys: Optional[List[str]] = None,
):
    tracked_keys = set(overrides.keys()) | set(clear_keys or [])
    original_values = {key: os.environ.get(key) for key in tracked_keys}
    try:
        for key in clear_keys or []:
            os.environ.pop(key, None)
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _temporary_settings_overrides(overrides: Dict[str, Any]):
    original_values = {key: getattr(settings, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(settings, key, value)
        yield
    finally:
        for key, value in original_values.items():
            setattr(settings, key, value)


@contextmanager
def _temporary_tool_registry_overrides(
    *,
    disabled_tool_names: Optional[List[str]] = None,
    disable_meta_tools: bool = False,
):
    normalized_disabled_names = [
        str(name or "").strip()
        for name in (disabled_tool_names or [])
        if str(name or "").strip()
    ]
    if not normalized_disabled_names and not disable_meta_tools:
        yield
        return

    merged_disabled_names = list(getattr(settings, "disabled_tool_names_base", ()) or ())
    merged_disabled_names.extend(normalized_disabled_names)
    overrides: Dict[str, Any] = {
        "disabled_tool_names_base": tuple(dict.fromkeys(merged_disabled_names)),
    }
    if disable_meta_tools:
        overrides["disable_meta_tools"] = True

    with _temporary_settings_overrides(overrides):
        yield


@contextmanager
def _pipeline_runtime_profile_context(runtime_profile: Any):
    resolved_profile = _normalize_runtime_profile(runtime_profile)
    if resolved_profile != _PIPELINE_RUNTIME_PROFILE_EVALUATION_LOCAL:
        yield resolved_profile
        return

    env_overrides, clear_keys = _evaluation_local_runtime_env_overrides()
    settings_overrides = {
        "non_planner_model_provider": "transformers",
        "non_planner_embedding_provider": "local",
        "planner_grounding_embedding_provider": "local",
    }
    logger.info(
        "execute_pipeline runtime_profile=%s -> forcing local non-planner/visual/embedding runtime, disabling sidecar autostart, and clearing distributed worker env residue.",
        resolved_profile,
    )
    with _temporary_environment_overrides(env_overrides, clear_keys=clear_keys):
        with _temporary_settings_overrides(settings_overrides):
            yield resolved_profile


def _runtime_profile_engine_kwargs(runtime_profile: Any) -> Dict[str, Any]:
    # runtime_profile is an execution-boundary concern only. It must not
    # silently alter planner/subagent decode policy, whose final values are
    # expected to come from config/settings.
    _normalize_runtime_profile(runtime_profile)
    return {}


def _normalize_pipeline_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "": "react",
        "default": "react",
        "standard": "react",
        "ablation": "planner_toolsearch_direct",
        "toolsearch_direct": "planner_toolsearch_direct",
        "tool_search_direct": "planner_toolsearch_direct",
        "planner_toolsearch_direct": "planner_toolsearch_direct",
        "planner-toolsearch-direct": "planner_toolsearch_direct",
        "planner_direct": "planner_direct_answer",
        "planner_direct_answer": "planner_direct_answer",
        "planner-single-turn": "planner_direct_answer",
        "planner_single_turn": "planner_direct_answer",
        "direct_answer": "planner_direct_answer",
        "pure": "planner_direct_answer",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in {"react", "planner_direct_answer", "planner_toolsearch_direct"}:
        raise ValueError(f"Unsupported pipeline_mode={value!r}.")
    return resolved


def _endpoint_list_is_healthy(raw_endpoints: str, api_key: str) -> bool:
    endpoints = [
        item.strip().rstrip("/")
        for item in str(raw_endpoints or "").replace(";", ",").split(",")
        if item.strip()
    ]
    if not endpoints:
        return False
    for endpoint in endpoints:
        base = endpoint[:-3] if endpoint.endswith("/v1") else endpoint
        request = urllib.request.Request(
            base.rstrip("/") + "/health",
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                status = int(getattr(response, "status", 200))
        except Exception:
            return False
        if not (200 <= status < 300):
            return False
    return True


def _split_endpoint_list(raw_endpoints: str) -> List[str]:
    endpoints: List[str] = []
    for item in str(raw_endpoints or "").replace(";", ",").split(","):
        endpoint = item.strip().rstrip("/")
        if not endpoint or endpoint in endpoints:
            continue
        endpoints.append(endpoint)
    return endpoints


def _sidecar_smoke_png_data_url(max_pixels: int) -> str:
    side = max(28, int(math.ceil(math.sqrt(max(28 * 28, int(max_pixels or 0))))))
    raw = b"".join(b"\0" + (b"\xff\xff\xff" * side) for _ in range(side))

    def _png_chunk(kind: bytes, data: bytes) -> bytes:
        crc = binascii.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _http_json_request(
    url: str,
    *,
    api_key: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key or 'EMPTY'}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_body = response.read()
    if not raw_body:
        return {}
    try:
        return json.loads(raw_body.decode("utf-8"))
    except Exception:
        return {}


def _discover_sidecar_served_model(endpoint: str, api_key: str) -> str:
    try:
        payload = _http_json_request(
            endpoint.rstrip("/") + "/models",
            api_key=api_key,
            method="GET",
            timeout=3.0,
        )
    except Exception:
        return ""
    data = payload.get("data")
    if not isinstance(data, list):
        return ""
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if model_id:
            return model_id
    return ""


def _endpoint_supports_multimodal_chat_smoke(endpoint: str, api_key: str) -> bool:
    model_id = _discover_sidecar_served_model(endpoint, api_key)
    if not model_id:
        return False
    smoke_pixels = min(
        max(28 * 28, int(getattr(settings, "visual_service_max_pixels", 336 * 336) or (336 * 336))),
        max(28 * 28, int(getattr(settings, "qwen_vl_image_max_pixels", 1280 * 28 * 28) or (1280 * 28 * 28))),
    )
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _sidecar_smoke_png_data_url(smoke_pixels)},
                    },
                    {"type": "text", "text": "Reply with ok."},
                ],
            }
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    try:
        _http_json_request(
            endpoint.rstrip("/") + "/chat/completions",
            api_key=api_key,
            method="POST",
            payload=payload,
            timeout=20.0,
        )
        return True
    except urllib.error.HTTPError as exc:
        try:
            response_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            response_text = ""
        logger.warning(
            "Non-planner sidecar multimodal smoke failed endpoint=%s status=%s response=%s",
            endpoint,
            getattr(exc, "code", "<unknown>"),
            response_text[:1000],
        )
        return False
    except Exception as exc:
        logger.warning(
            "Non-planner sidecar multimodal smoke failed endpoint=%s error=%s",
            endpoint,
            exc,
        )
        return False


def _endpoint_list_supports_multimodal_chat_smoke(raw_endpoints: str, api_key: str) -> bool:
    endpoints = _split_endpoint_list(raw_endpoints)
    if not endpoints:
        return False
    return all(_endpoint_supports_multimodal_chat_smoke(endpoint, api_key) for endpoint in endpoints)


def _should_prepare_non_planner_sidecars_for_e2e() -> bool:
    provider = (
        os.environ.get("TOOL_STAR_NON_PLANNER_MODEL_PROVIDER")
        or os.environ.get("TOOL_STAR_AUX_MODEL_PROVIDER")
        or getattr(settings, "non_planner_model_provider", "auto")
    )
    if not _provider_requests_service(provider):
        return False
    if str(os.environ.get("TOOL_STAR_DISABLE_E2E_SIDECAR_AUTOSTART", "")).strip() == "1":
        return False
    if any(os.environ.get(key) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return False
    return True


def _ensure_non_planner_runtime_ready_for_e2e() -> None:
    if not _should_prepare_non_planner_sidecars_for_e2e():
        return

    project_root = Path(__file__).resolve().parent
    env_file = Path(
        os.environ.get(
            "TOOL_STAR_SIDECAR_ENV_FILE",
            str(project_root / "rl_workdir" / "sidecars" / "env.sh"),
        )
    ).resolve()
    api_key = str(os.environ.get("TOOL_STAR_SIDECAR_API_KEY", "EMPTY") or "EMPTY").strip()

    loaded_env = _load_sidecar_runtime_env(env_file) if env_file.exists() else {}
    endpoint_env = (
        loaded_env.get("TOOL_STAR_NON_PLANNER_API_BASES")
        or loaded_env.get("TOOL_STAR_NON_PLANNER_API_BASE")
        or os.environ.get("TOOL_STAR_NON_PLANNER_API_BASES")
        or os.environ.get("TOOL_STAR_NON_PLANNER_API_BASE")
    )
    normalized_endpoints = format_endpoint_list(str(endpoint_env or ""))
    if normalized_endpoints in _NON_PLANNER_RUNTIME_READY_SIGNATURES:
        if _endpoint_list_is_healthy(normalized_endpoints, api_key):
            logger.info("Reusing validated non-planner sidecar runtime env from %s", env_file)
            return
        _NON_PLANNER_RUNTIME_READY_SIGNATURES.discard(normalized_endpoints)
    if (
        _endpoint_list_is_healthy(normalized_endpoints, api_key)
        and _endpoint_list_supports_multimodal_chat_smoke(normalized_endpoints, api_key)
    ):
        logger.info("Loaded healthy non-planner sidecar runtime env from %s", env_file)
        _NON_PLANNER_RUNTIME_READY_SIGNATURES.add(normalized_endpoints)
        return

    start_script = project_root / "scripts" / "start_non_planner_sidecars.sh"
    if not start_script.exists():
        logger.warning(
            "Non-planner sidecar autostart skipped because script is missing: %s",
            start_script,
        )
        return

    logger.info(
        "Non-planner sidecar endpoints are missing or unhealthy for e2e run; starting managed sidecars via %s",
        start_script,
    )
    start_env = os.environ.copy()
    start_env["TOOL_STAR_ENABLE_NON_PLANNER_SIDECARS"] = "1"
    subprocess.run(
        ["sh", str(start_script)],
        cwd=str(project_root),
        env=start_env,
        check=True,
    )
    loaded_env = _load_sidecar_runtime_env(env_file)
    endpoint_env = (
        loaded_env.get("TOOL_STAR_NON_PLANNER_API_BASES")
        or loaded_env.get("TOOL_STAR_NON_PLANNER_API_BASE")
        or os.environ.get("TOOL_STAR_NON_PLANNER_API_BASES")
        or os.environ.get("TOOL_STAR_NON_PLANNER_API_BASE")
    )
    normalized_endpoints = format_endpoint_list(str(endpoint_env or ""))
    if not (
        _endpoint_list_is_healthy(normalized_endpoints, api_key)
        and _endpoint_list_supports_multimodal_chat_smoke(normalized_endpoints, api_key)
    ):
        raise RuntimeError(
            f"Managed non-planner sidecar startup completed but endpoints are still unhealthy: {endpoint_env or '<empty>'}"
        )
    _NON_PLANNER_RUNTIME_READY_SIGNATURES.add(normalized_endpoints)
    logger.info("Managed non-planner sidecars are ready for this e2e run: %s", endpoint_env)


def _get_preprocessing_module_exports():
    from preprocessing import VideoPreprocessingPipeline, PipelineConfig
    from preprocessing.storage.qwen3_vl_embedding_runtime import Qwen3VLEmbeddingRuntimeEncoder
    from preprocessing.video.segmentation import VideoSegmenter

    return VideoPreprocessingPipeline, PipelineConfig, Qwen3VLEmbeddingRuntimeEncoder, VideoSegmenter


def _get_recursive_engine():
    from planner.engine import RecursiveEngine

    return RecursiveEngine


def _get_llm_provider_factory():
    from core.llm.provider import get_llm_provider

    return get_llm_provider


def _get_registry():
    from toolkit.registry import registry

    return registry


def _get_visual_helpers():
    from toolkit.impl.visual import (
        _run_video_clip_qa,
        pack_planner_image_evidence,
        planner_image_attachment_limit,
        prepare_planner_image_attachment,
        resolve_processed_frame_image_path,
        summarize_planner_image_evidence,
    )

    return {
        "run_video_clip_qa": _run_video_clip_qa,
        "pack_planner_image_evidence": pack_planner_image_evidence,
        "planner_image_attachment_limit": planner_image_attachment_limit,
        "prepare_planner_image_attachment": prepare_planner_image_attachment,
        "resolve_processed_frame_image_path": resolve_processed_frame_image_path,
        "summarize_planner_image_evidence": summarize_planner_image_evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tool Star RL end-to-end on one video/query pair.",
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("query", help="Top-level user query to answer against the video.")
    parser.add_argument(
        "workspace_root",
        nargs="?",
        default=None,
        help="Optional shared workspace/cache root. Defaults to settings.data_cache_dir.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable preprocessing cache reuse for this run.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional log file path. When omitted, defaults to the query workspace log file.",
    )
    parser.add_argument(
        "--no-file-log",
        action="store_true",
        help="Disable the internal file logger and rely on stdout/stderr redirection only.",
    )
    parser.add_argument(
        "--flat-workspace",
        action="store_true",
        help="Reuse the provided workspace root directly instead of creating a nested queries/<slug> directory.",
    )
    parser.add_argument(
        "--shared-preprocess-root",
        default=None,
        help="Optional shared video-cache root. When omitted, defaults to workspace_root.",
    )
    parser.add_argument(
        "--runtime-profile",
        default=_PIPELINE_RUNTIME_PROFILE_DEFAULT,
        help="Execution profile. Use 'evaluation_local' to force pure local non-planner evaluation runtime.",
    )
    return parser.parse_args()


def _looks_stale_outputs(segments: Dict) -> bool:
    """Heuristic check for stale/legacy preprocessing artifacts that should be regenerated."""
    if not segments:
        return True

    transcript_zero_time_mismatch = 0

    for seg_id, seg in segments.items():
        caption = str(seg.get("caption", "")).strip().lower()
        transcript = str(seg.get("transcript", ""))
        time_range = str(seg.get("time", "0-0"))
        frame_paths = seg.get("frame_paths") or []
        segment_aliases = seg.get("segment_aliases") or []
        segment_video_path = str(seg.get("segment_video_path", "") or "").strip()
        segment_bounds = _parse_time_range_bounds(time_range)
        transcript_bounds = _extract_transcript_bounds(transcript)

        if "transcript evidence:" in caption:
            return True
        if _is_low_information_caption_text(caption):
            return True
        if "caption" not in seg or "transcript" not in seg:
            return True

        if seg_id != "0" and time_range and "-" in time_range and transcript.startswith("[0.00s ->"):
            transcript_zero_time_mismatch += 1
        if segment_bounds and transcript_bounds:
            seg_start, seg_end = segment_bounds
            transcript_start, transcript_end = transcript_bounds
            if transcript_start < seg_start - 0.05 or transcript_end > seg_end + 0.05:
                return True
        if not frame_paths or not segment_aliases or not segment_video_path:
            return True

    if transcript_zero_time_mismatch > 0:
        return True

    return False


def _is_low_information_caption_text(text: Any) -> bool:
    normalized = " ".join(str(text or "").split()).strip().lower()
    if not normalized:
        return True
    if normalized.startswith("visual caption unavailable"):
        return True
    content_chars = re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)
    if not content_chars:
        return True
    punctuation_count = len(re.findall(r"[^\w\s]", normalized, flags=re.UNICODE))
    if re.search(r"([^\w\s])\1{7,}", normalized, flags=re.UNICODE):
        return True
    if punctuation_count >= 12 and punctuation_count * 2 >= len(normalized):
        return True
    return len(content_chars) <= 1 and len(normalized) <= 3


def _required_preprocessing_artifacts(output_dir: Path) -> List[Path]:
    """Return the minimal artifact set required to safely reuse cached preprocessing outputs."""
    return [
        output_dir / "segments.json",
        output_dir / "segment_times_info.json",
        output_dir / "vectors" / "metadata.json",
    ]


def _parse_time_range_bounds(value: Any) -> Optional[tuple[float, float]]:
    text = str(value or "").strip()
    if "-" not in text:
        return None
    left, right = text.split("-", 1)
    try:
        start = float(left.strip())
        end = float(right.strip())
    except Exception:
        return None
    return (min(start, end), max(start, end))


def _extract_transcript_bounds(transcript: str) -> Optional[tuple[float, float]]:
    matches = re.findall(r"\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]", str(transcript or ""))
    if not matches:
        return None
    try:
        start = float(matches[0][0])
        end = float(matches[-1][1])
    except Exception:
        return None
    return (min(start, end), max(start, end))


def _segment_cache_is_current(working_dir: Path, video_name: str) -> bool:
    _, _, _, VideoSegmenter = _get_preprocessing_module_exports()
    cache_meta_path = working_dir / "_cache" / video_name / "segment_times_info.json"
    if not cache_meta_path.exists():
        return False
    try:
        with open(cache_meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        strategy = str(payload.get("strategy", "")).strip()
        times_map = payload.get("segment_times_info", {})
        if not isinstance(times_map, dict) or not times_map:
            return False
        if len(times_map) == 1:
            only_info = next(iter(times_map.values()), {})
            timestamp = only_info.get("timestamp", (0, 0))
            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                try:
                    duration = float(timestamp[1]) - float(timestamp[0])
                except Exception:
                    duration = 0.0
                if duration > float(VideoSegmenter().hard_max_segment_duration) + 1e-3:
                    return False
        return (
            int(payload.get("cache_version", 0) or 0) == VideoSegmenter.CACHE_VERSION
            and strategy in {"asr_semantic_v4", "scene_v2", "fixed_v1"}
        )
    except Exception:
        return False


def _sanitize_query(query: str) -> str:
    """Convert query text into filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", query).strip("-").lower()
    return slug[:60] or "query"


def _sanitize_workspace_component(
    value: Any,
    *,
    max_length: int = 80,
    fallback: str = "item",
) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "")).strip("-").lower()
    return slug[:max_length] or fallback


def _build_run_id() -> str:
    return f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}_{os.getpid()}"


def _build_video_workspace_slug(video_path: str) -> str:
    video_stem = Path(video_path).stem
    normalized = _sanitize_workspace_component(video_stem, max_length=48, fallback="video")
    digest = hashlib.md5(str(Path(video_path).expanduser()).encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def _build_query_run_artifact_paths(run_workspace: Path) -> Dict[str, Path]:
    resolved_workspace = Path(run_workspace).resolve()
    inputs_dir = resolved_workspace / "inputs"
    logs_dir = resolved_workspace / "logs"
    results_dir = resolved_workspace / "results"
    return {
        "workspace": resolved_workspace,
        "inputs_dir": inputs_dir,
        "logs_dir": logs_dir,
        "results_dir": results_dir,
        "manifest_path": resolved_workspace / "manifest.json",
        "request_path": inputs_dir / "request.json",
        "run_log_path": logs_dir / "run.log",
        "raw_trace_jsonl_path": logs_dir / "raw_llm_trace.jsonl",
        "run_result_path": results_dir / "run_result.json",
        "error_path": results_dir / "error.json",
    }


def _write_json_file(path: Path, payload: Any) -> None:
    resolved_path = Path(path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _create_query_run_workspace(root_dir: Path, video_path: str, query: str) -> Dict[str, Any]:
    video_slug = _build_video_workspace_slug(video_path)
    query_slug = _sanitize_query(query)
    query_group_dir = root_dir / "queries" / video_slug / query_slug
    run_id = _build_run_id()
    run_workspace = query_group_dir / "runs" / run_id
    artifact_paths = _build_query_run_artifact_paths(run_workspace)

    query_group_dir.mkdir(parents=True, exist_ok=True)
    for directory in (
        artifact_paths["workspace"],
        artifact_paths["inputs_dir"],
        artifact_paths["logs_dir"],
        artifact_paths["results_dir"],
    ):
        directory.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Created query run workspace: video_slug=%s query_slug=%s run_id=%s path=%s",
        video_slug,
        query_slug,
        run_id,
        run_workspace,
    )
    return {
        "video_slug": video_slug,
        "query_slug": query_slug,
        "run_id": run_id,
        "query_group_dir": query_group_dir.resolve(),
        "run_workspace": artifact_paths["workspace"],
        "artifact_paths": artifact_paths,
    }


def _attach_file_logger(log_path: Path, rewrite: bool = True) -> logging.FileHandler:
    """Attach file handler so logs are persisted for the current run only."""
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                handler_path = Path(handler.baseFilename).resolve()
                if not rewrite and handler_path == log_path:
                    return handler
                if rewrite or handler_path == log_path:
                    root_logger.removeHandler(handler)
                    handler.close()
            except Exception:
                continue

    if rewrite:
        with open(log_path, "w", encoding="utf-8") as log_seed_handle:
            log_seed_handle.write("\ufeff")
    elif not log_path.exists() or log_path.stat().st_size <= 0:
        with open(log_path, "w", encoding="utf-8") as log_seed_handle:
            log_seed_handle.write("\ufeff")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)
    return file_handler


def _detach_file_logger(handler: Optional[logging.Handler]) -> None:
    if handler is None:
        return
    root_logger = logging.getLogger()
    try:
        root_logger.removeHandler(handler)
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass


def _remove_file_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        logger.warning("Failed to remove stale artifact %s: %s", path, exc)


def _cleanup_runtime_artifacts(workspace: Path) -> None:
    _remove_file_if_exists(workspace / "run_report.json")
    _remove_file_if_exists(workspace / "results" / "run_result.json")
    _remove_file_if_exists(workspace / "results" / "error.json")
    _remove_file_if_exists(workspace / "logs" / "raw_llm_trace.jsonl")


def _truncate_text(text: Any, max_chars: int = 240) -> str:
    """Normalize text for prompt context without truncating content."""
    if text is None:
        return ""
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    return normalized


def _resolve_embedding_device_request(device: Optional[str]) -> tuple[str, Optional[int], str]:
    requested_device = str(device or "").strip()
    if requested_device:
        device_type = requested_device.split(":", 1)[0] if ":" in requested_device else requested_device
        _, device_count = detect_device()
    else:
        device_type, device_count = detect_device()
    initialize_device(device_type)
    stable_device = runtime_session.get_stable_device(
        "embedding",
        device=requested_device or device_type,
    )
    return device_type, device_count, stable_device


def _get_embedder(device: Optional[str] = None):
    """Lazy init Qwen3-VL-Embedding runtime for segment-level grounding."""
    _, _, Qwen3VLEmbeddingRuntimeEncoder, _ = _get_preprocessing_module_exports()
    device_type, device_count, stable_device = _resolve_embedding_device_request(device)
    embed_model_path = Path(settings.qwen3_vl_embedding_model_path).expanduser().resolve()
    cache_key = json.dumps(
        {
            "runtime": "planner_grounding_qwen3_vl_embedding",
            "model_path": str(embed_model_path),
            "device": stable_device,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cached = runtime_session.get_cached_artifact(cache_key)
    cache_hit = cached is not None
    runtime_session.record_cache("grounding_embedder", cache_hit)
    if cached is None:
        with runtime_session.time_block("runtime_load.grounding_embedder_seconds"):
            if not embed_model_path.exists():
                raise FileNotFoundError(f"Local Qwen3-VL-Embedding model path not found: {embed_model_path}")
            runtime = Qwen3VLEmbeddingRuntimeEncoder(
                model_path=str(embed_model_path),
                device=stable_device,
            )
            logger.info(
                "Grounding embedder runtime: device=%s visible_accelerators=%s model=%s",
                stable_device,
                device_count or 0,
                embed_model_path,
            )
            logger.info("Grounding embedder accelerator snapshot: %s", get_accelerator_snapshot(device_type))
            cached = runtime_session.cache_artifact(
                cache_key,
                {
                    "runtime": runtime,
                },
            )
    return cached["runtime"]


def _resolve_grounding_embedding_provider(provider: Optional[str] = None) -> str:
    configured = str(
        provider
        or getattr(settings, "planner_grounding_embedding_provider", "auto")
        or "auto"
    ).strip().lower()
    if configured in {"openai", "api", "remote", "service", "external"}:
        return "service"
    if configured in {"local", "transformers", "qwen3_vl", "qwen3-vl"}:
        return "local"
    if configured != "auto":
        logger.warning("Unknown planner grounding embedding provider=%s; using auto.", configured)

    service_model = _grounding_embedding_service_model()
    service_base = _grounding_embedding_service_base()
    return "service" if service_model and service_base else "local"


def _grounding_embedding_service_base() -> str:
    return format_endpoint_list(
        str(
            os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASES", "")
            or os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_BASE", "")
            or os.environ.get("TOOL_STAR_PREPROCESS_EMBEDDING_API_BASES", "")
            or os.environ.get("TOOL_STAR_PREPROCESS_EMBEDDING_API_BASE", "")
            or getattr(settings, "planner_grounding_embedding_api_base", "")
            or getattr(settings, "non_planner_embedding_api_base", "")
            or ""
        ).strip()
    )


def _grounding_embedding_service_key() -> str:
    return str(
        os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_API_KEY", "")
        or getattr(settings, "planner_grounding_embedding_api_key", "")
        or getattr(settings, "non_planner_embedding_api_key", "")
        or settings.openai_api_key
        or "EMPTY"
    ).strip()


def _grounding_embedding_service_model() -> str:
    return str(
        os.environ.get("TOOL_STAR_ROLLOUT_GROUNDING_EMBEDDING_MODEL", "")
        or getattr(settings, "planner_grounding_embedding_model", "")
        or getattr(settings, "non_planner_embedding_model", "")
        or getattr(settings, "qwen3_vl_embedding_model_path", "")
        or ""
    ).strip()


def _get_grounding_embedding_service_pool() -> OpenAIEndpointPool:
    api_base = _grounding_embedding_service_base()
    api_key = _grounding_embedding_service_key()
    if not api_base:
        raise RuntimeError("planner grounding embedding service requires planner_grounding_embedding_api_base.")
    cache_key = json.dumps(
        {
            "runtime": "planner_grounding_embedding_service_pool",
            "api_base": api_base,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cached = runtime_session.get_cached_artifact(cache_key)
    if cached is not None:
        return cached["pool"]
    pool = OpenAIEndpointPool(
        api_key=api_key,
        base_urls=api_base,
        namespace="planner_grounding_embedding",
    )
    return runtime_session.cache_artifact(cache_key, {"pool": pool})["pool"]


def _format_grounding_embedding_service_input(text: str, kind: str) -> str:
    _, _, Qwen3VLEmbeddingRuntimeEncoder, _ = _get_preprocessing_module_exports()
    instruction = (
        Qwen3VLEmbeddingRuntimeEncoder.QUERY_INSTRUCTION
        if kind == "query"
        else Qwen3VLEmbeddingRuntimeEncoder.DEFAULT_INSTRUCTION
    )
    return f"{instruction}\n{text}"


def _encode_grounding_embedding_via_service(*, text: str, kind: str) -> torch.Tensor:
    torch = _require_torch()
    model = _grounding_embedding_service_model()
    if not model:
        raise RuntimeError("planner grounding embedding service requires planner_grounding_embedding_model.")
    with _get_grounding_embedding_service_pool().borrow() as (client, _endpoint):
        response = client.embeddings.create(
            model=model,
            input=_format_grounding_embedding_service_input(text, kind),
            encoding_format="float",
        )
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError("planner grounding embedding service returned no embedding data.")
    embedding = getattr(data[0], "embedding", None)
    if not embedding:
        raise RuntimeError("planner grounding embedding service returned an empty embedding.")
    vector = torch.tensor(embedding, dtype=torch.float32)
    norm = torch.linalg.vector_norm(vector)
    if torch.isfinite(norm) and float(norm.item()) > 0:
        vector = vector / norm
    return vector


def _cached_grounding_embedding(
    *,
    text: str,
    kind: str,
    embedding_device: Optional[str] = None,
    embedding_provider: Optional[str] = None,
) -> torch.Tensor:
    torch = _require_torch()
    normalized_text = (text or "").strip()
    if not normalized_text:
        return torch.zeros(1, dtype=torch.float32)
    provider = _resolve_grounding_embedding_provider(embedding_provider)
    runtime_device = "service"
    if provider == "local":
        runtime = _get_embedder(device=embedding_device)
        runtime_device = str(getattr(runtime, "runtime_device", embedding_device or "auto"))
    embed_model_path = Path(settings.qwen3_vl_embedding_model_path).expanduser().resolve()
    cache_key = json.dumps(
        {
            "runtime": "planner_grounding_qwen3_vl_embedding_vector",
            "provider": provider,
            "model_path": str(embed_model_path),
            "service_model": _grounding_embedding_service_model(),
            "service_base": _grounding_embedding_service_base(),
            "device": runtime_device,
            "kind": str(kind or "text"),
            "text_sha1": hashlib.sha1(normalized_text.encode("utf-8")).hexdigest(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cached = runtime_session.get_cached_artifact(cache_key)
    runtime_session.record_cache("grounding_embedding_vector", cached is not None)
    if cached is not None:
        return cached

    if provider == "service":
        vector = _encode_grounding_embedding_via_service(text=normalized_text, kind=kind)
    elif kind == "query":
        vector = runtime.encode_query(normalized_text)
    else:
        vector = runtime.encode_text(normalized_text)
    vector = vector.detach().to(device="cpu", dtype=torch.float32)
    return runtime_session.cache_artifact(cache_key, vector)


def _get_cached_llm_provider(
    *,
    model_name: str,
    model_role: str,
    device: Optional[str],
) -> Any:
    return _get_llm_provider_factory()(
        model_name=model_name,
        model_role=model_role,
        device=device,
    )


_REPORT_RETRIEVAL_TOOLS = frozenset({
    "Tool_Search", "Query_Rewrite", "Semantic_Retrieve", "Keyword_Retrieve",
    "Segment_Retrieve", "Graph_Search", "Clean_Results", "Filtering", "Aggregation",
    "evidence_verify",
})
_REPORT_VISUAL_TOOLS = frozenset({
    "Inspect_Frame", "OCR_Extract", "Crop", "Zoom_in", "Entity_Track",
    "Action_Recognition", "Emotion_Analysis", "Audio_Event_Detect",
    "Speaker_Diarization", "Cross_Modal_Verify", "Video_Clip_QA",
})
_REPORT_COMPUTE_TOOLS = frozenset({
    "Python_Executor", "SQL_Executor", "Debt_Interest_Analyzer",
    "Amortization_Schedule", "Total_Interest", "Balance_After_Periods",
    "Simple_Interest", "Compound_Interest", "Percentage_Change",
})


def _report_tool_family(tool_name: str) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return "other"
    if name in _REPORT_RETRIEVAL_TOOLS:
        return "retrieval"
    if name in _REPORT_VISUAL_TOOLS:
        return "visual"
    if name in _REPORT_COMPUTE_TOOLS:
        return "compute"
    if name in {"Finish"}:
        return "finish"
    if name in {"Parallel"}:
        return "parallel"
    return "other"


def _infer_tool_type_from_registry(tool_name: str) -> str:
    tool = _get_registry().get_tool(tool_name, log_missing=False)
    if bool(getattr(tool, "is_virtual", False)):
        return "unknown"
    return infer_primary_tool_type(getattr(tool, "tags", None) or [])


def _infer_tool_capability_tier(tool_name: str) -> str:
    tool = _get_registry().get_tool(tool_name, log_missing=False)
    if bool(getattr(tool, "is_virtual", False)):
        return "unknown"
    tagset = {str(tag).strip().lower() for tag in (getattr(tool, "tags", None) or [])}
    if {"debug_placeholder", "placeholder_debug", "placeholder"} & tagset:
        return "debug"
    if {"weak_proxy", "heuristic_proxy", "weak"} & tagset:
        return "weak"
    return "production"


def _build_tool_entry(tool_name: str) -> Dict[str, Any]:
    normalized_name = str(tool_name or "").strip()
    return {
        "name": normalized_name,
        "tool_type": _infer_tool_type_from_registry(normalized_name),
        "capability_tier": _infer_tool_capability_tier(normalized_name),
        "family": _report_tool_family(normalized_name),
    }


def _iter_trajectory_steps(trajectory: Optional[Dict[str, Any]]):
    if not isinstance(trajectory, dict):
        return
    for step in trajectory.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        yield step
        sub_trajectory = step.get("sub_trajectory")
        if isinstance(sub_trajectory, dict):
            yield from _iter_trajectory_steps(sub_trajectory)
        sub_trajectories = step.get("sub_trajectories")
        if isinstance(sub_trajectories, list):
            for child_trajectory in sub_trajectories:
                if isinstance(child_trajectory, dict):
                    yield from _iter_trajectory_steps(child_trajectory)


def _iter_parallel_child_observation_items(value: Any):
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            nested_items = item.get("items")
            nested_result = item.get("result")
            if isinstance(nested_items, list):
                yield from _iter_parallel_child_observation_items(nested_items)
                continue
            if isinstance(nested_result, dict) and isinstance(nested_result.get("items"), list):
                yield from _iter_parallel_child_observation_items(nested_result.get("items"))
                continue
            yield item
        return

    if isinstance(value, dict):
        nested_items = value.get("items")
        if isinstance(nested_items, list):
            yield from _iter_parallel_child_observation_items(nested_items)


def _extract_parallel_child_tools(step: Dict[str, Any]) -> List[str]:
    child_tools: List[str] = []
    observation_items = step.get("observation_items")
    if isinstance(observation_items, list):
        for item in _iter_parallel_child_observation_items(observation_items):
            tool_name = str(item.get("tool", "")).strip()
            if tool_name and tool_name != "Parallel":
                child_tools.append(tool_name)
    if child_tools:
        return child_tools

    observation_text = step.get("observation")
    if not isinstance(observation_text, str) or not observation_text.strip():
        return child_tools
    try:
        payload = json.loads(observation_text)
    except Exception:
        return child_tools
    for item in _iter_parallel_child_observation_items(payload):
        tool_name = str(item.get("tool", "")).strip()
        if tool_name and tool_name != "Parallel":
            child_tools.append(tool_name)
    return child_tools


def _extract_parallel_batch_tool_names(step: Dict[str, Any]) -> List[str]:
    action_args = step.get("action_args")
    if not isinstance(action_args, dict):
        return []
    batch = action_args.get("batch")
    if not isinstance(batch, list):
        return []

    tool_names: List[str] = []
    for item in batch:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name and name != "Parallel":
            tool_names.append(name)
    return tool_names


def _extract_step_action_tool_names(step: Dict[str, Any]) -> List[str]:
    action = str(step.get("action", "")).strip()
    if not action or action in {"Finish", "System", "Assistant_Note"}:
        return []
    if action == "Parallel":
        return _extract_parallel_batch_tool_names(step)
    return [action]


def _summarize_tool_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    usage: Dict[str, int] = {}
    tool_type_counts: Dict[str, int] = {}
    capability_tier_counts: Dict[str, int] = {}
    tool_family_counts: Dict[str, int] = {}

    for entry in entries:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        usage[name] = usage.get(name, 0) + 1
        tool_type = str(entry.get("tool_type", "")).strip() or "unknown"
        capability_tier = str(entry.get("capability_tier", "")).strip() or "unknown"
        family = str(entry.get("family", "")).strip() or "other"
        tool_type_counts[tool_type] = tool_type_counts.get(tool_type, 0) + 1
        capability_tier_counts[capability_tier] = capability_tier_counts.get(capability_tier, 0) + 1
        tool_family_counts[family] = tool_family_counts.get(family, 0) + 1

    unique_tools = sorted(usage.keys())
    unique_type_counts: Dict[str, int] = {}
    unique_tier_counts: Dict[str, int] = {}
    for name in unique_tools:
        entry = _build_tool_entry(name)
        unique_type = str(entry.get("tool_type", "")).strip() or "unknown"
        unique_tier = str(entry.get("capability_tier", "")).strip() or "unknown"
        unique_type_counts[unique_type] = unique_type_counts.get(unique_type, 0) + 1
        unique_tier_counts[unique_tier] = unique_tier_counts.get(unique_tier, 0) + 1

    return {
        "tool_usage_summary": dict(sorted(usage.items(), key=lambda item: (-item[1], item[0]))),
        "unique_tools": unique_tools,
        "unique_tool_count": len(unique_tools),
        "tool_type_counts": dict(sorted(tool_type_counts.items())),
        "capability_tier_counts": dict(sorted(capability_tier_counts.items())),
        "tool_family_counts": dict(sorted(tool_family_counts.items())),
        "unique_tool_type_counts": dict(sorted(unique_type_counts.items())),
        "unique_capability_tier_counts": dict(sorted(unique_tier_counts.items())),
        "base_meta_coverage": {
            "base_used": bool(unique_type_counts.get("base")),
            "meta_used": bool(unique_type_counts.get("meta")),
            "coverage_ratio": round(
                (
                    int(bool(unique_type_counts.get("base")))
                    + int(bool(unique_type_counts.get("meta")))
                ) / 2.0,
                4,
            ),
        },
    }


def _build_agent_metrics(trajectory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    planner_iterations = 0
    subagent_iterations = 0
    planner_entries: List[Dict[str, Any]] = []
    subagent_entries: List[Dict[str, Any]] = []
    runtime_entries: List[Dict[str, Any]] = []
    delegated_root_steps = 0
    decomposed_root_steps = 0
    parallel_concrete_batches = 0
    runtime_execution_steps = 0

    def _iter_child_trajectories(step: Dict[str, Any]) -> List[Dict[str, Any]]:
        children: List[Dict[str, Any]] = []
        sub_trajectory = step.get("sub_trajectory")
        if isinstance(sub_trajectory, dict):
            children.append(sub_trajectory)
        sub_trajectories = step.get("sub_trajectories")
        if isinstance(sub_trajectories, list):
            children.extend(item for item in sub_trajectories if isinstance(item, dict))
        return children

    def _collect_runtime_entries(current_trajectory: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        if not isinstance(current_trajectory, dict):
            return collected
        for step in current_trajectory.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            action = str(step.get("action", "")).strip()
            if not action or action in {"Finish", "System", "Assistant_Note"}:
                continue

            child_trajectories = _iter_child_trajectories(step)
            if action == "Parallel":
                tool_names = _extract_parallel_child_tools(step) or _extract_parallel_batch_tool_names(step)
                for tool_name in tool_names:
                    collected.append(_build_tool_entry(tool_name))
                continue

            if child_trajectories:
                for child in child_trajectories:
                    collected.extend(_collect_runtime_entries(child))
                continue

            collected.append(_build_tool_entry(action))
        return collected

    def _count_runtime_execution_steps(current_trajectory: Optional[Dict[str, Any]]) -> int:
        if not isinstance(current_trajectory, dict):
            return 0
        total = 0
        for step in current_trajectory.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            action = str(step.get("action", "")).strip()
            if not action or action in {"Finish", "System", "Assistant_Note"}:
                continue
            child_trajectories = _iter_child_trajectories(step)
            if action == "Parallel":
                if _extract_parallel_child_tools(step) or _extract_parallel_batch_tool_names(step):
                    total += 1
                continue
            if child_trajectories:
                for child in child_trajectories:
                    total += _count_runtime_execution_steps(child)
                continue
            total += 1
        return total

    def _count_subagent_action_tools(current_trajectory: Optional[Dict[str, Any]]) -> int:
        if not isinstance(current_trajectory, dict):
            return 0
        total = 0
        for step in _iter_trajectory_steps(current_trajectory):
            depth = int(step.get("depth", 0) or 0)
            if depth <= 0:
                continue
            total += len(_extract_step_action_tool_names(step))
        return total

    def _has_parallel_subagent_decomposition(current_trajectory: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(current_trajectory, dict):
            return False
        for step in _iter_trajectory_steps(current_trajectory):
            if str(step.get("action", "")).strip() != "Parallel":
                continue
            if len(_extract_parallel_batch_tool_names(step)) > 1:
                return True
        return False

    for step in _iter_trajectory_steps(trajectory):
        action = str(step.get("action", "")).strip()
        depth = int(step.get("depth", 0) or 0)
        if depth <= 0:
            planner_iterations += 1
        else:
            subagent_iterations += 1
            for tool_name in _extract_step_action_tool_names(step):
                subagent_entries.append(_build_tool_entry(tool_name))

    root_steps = trajectory.get("steps", []) if isinstance(trajectory, dict) else []
    for step in root_steps or []:
        if not isinstance(step, dict):
            continue
        planned_tools = _extract_step_action_tool_names(step)
        for tool_name in planned_tools:
            planner_entries.append(_build_tool_entry(tool_name))

        child_trajectories = _iter_child_trajectories(step)
        if child_trajectories:
            delegated_root_steps += 1
            child_tool_count = sum(_count_subagent_action_tools(child) for child in child_trajectories)
            if child_tool_count > len(planned_tools) or any(
                _has_parallel_subagent_decomposition(child)
                for child in child_trajectories
            ):
                decomposed_root_steps += 1

    runtime_entries = _collect_runtime_entries(trajectory)
    runtime_execution_steps = _count_runtime_execution_steps(trajectory)
    if isinstance(trajectory, dict):
        for step in _iter_trajectory_steps(trajectory):
            action = str(step.get("action", "")).strip()
            if action != "Parallel":
                continue
            runtime_parallel_names = _extract_parallel_child_tools(step) or _extract_parallel_batch_tool_names(step)
            if len(runtime_parallel_names) > 1:
                parallel_concrete_batches += 1

    planner_summary = _summarize_tool_entries(planner_entries)
    subagent_summary = _summarize_tool_entries(subagent_entries)
    runtime_summary = _summarize_tool_entries(runtime_entries)
    unique_runtime_tools = int(runtime_summary.get("unique_tool_count", 0) or 0)
    runtime_tool_call_count = len(runtime_entries)
    unique_tool_threshold = max(1, int(getattr(settings, "acceptance_min_unique_runtime_tools", 4) or 4))

    return {
        "tool_usage_layers": {
            "planner_proposed": planner_summary,
            "subagent_rewritten": subagent_summary,
            "runtime_executed": runtime_summary,
        },
        "tool_usage_summary": runtime_summary.get("tool_usage_summary", {}),
        "unique_tools": runtime_summary.get("unique_tools", []),
        "tool_family_counts": runtime_summary.get("tool_family_counts", {}),
        "tool_type_counts": runtime_summary.get("tool_type_counts", {}),
        "capability_tier_counts": runtime_summary.get("capability_tier_counts", {}),
        "runtime_tool_call_count": runtime_tool_call_count,
        "runtime_execution_steps": runtime_execution_steps,
        "planner_iterations": planner_iterations,
        "subagent_iterations": subagent_iterations,
        "delegated_root_steps": delegated_root_steps,
        "decomposed_root_steps": decomposed_root_steps,
        "subagent_decomposition_rate": round(
            decomposed_root_steps / delegated_root_steps,
            4,
        ) if delegated_root_steps else 0.0,
        "parallel_concrete_batches": parallel_concrete_batches,
        "parallel_concrete_batch_rate": round(
            parallel_concrete_batches / max(1, runtime_execution_steps),
            4,
        ) if runtime_execution_steps else 0.0,
        "acceptance_metrics": {
            "runtime_tool_call_count": runtime_tool_call_count,
            "runtime_execution_steps": runtime_execution_steps,
            "runtime_tools_executed": runtime_tool_call_count > 0,
            "unique_runtime_tools_count": unique_runtime_tools,
            f"unique_runtime_tools_gte_{unique_tool_threshold}": unique_runtime_tools >= unique_tool_threshold,
            "runtime_base_meta_coverage": runtime_summary.get("base_meta_coverage", {}),
        },
    }


def _encode_grounding_query_embedding(
    text: str,
    *,
    embedding_device: Optional[str] = None,
    embedding_provider: Optional[str] = None,
) -> torch.Tensor:
    """Encode a planner query with the configured Qwen3-VL-Embedding runtime."""
    return _cached_grounding_embedding(
        text=text,
        kind="query",
        embedding_device=embedding_device,
        embedding_provider=embedding_provider,
    )


def _encode_grounding_text_embedding(
    text: str,
    *,
    embedding_device: Optional[str] = None,
    embedding_provider: Optional[str] = None,
) -> torch.Tensor:
    """Encode segment text with the configured Qwen3-VL-Embedding runtime."""
    return _cached_grounding_embedding(
        text=text,
        kind="text",
        embedding_device=embedding_device,
        embedding_provider=embedding_provider,
    )


def _late_interaction_score(query_tokens: torch.Tensor, doc_tokens: torch.Tensor) -> float:
    """Compute a grounding similarity score for normalized embedding tensors."""
    torch = _require_torch()
    if query_tokens.ndim == 1 and doc_tokens.ndim == 1:
        if query_tokens.shape[0] != doc_tokens.shape[0]:
            return 0.0
        return float(torch.dot(query_tokens, doc_tokens).item())

    if query_tokens.ndim != 2 or doc_tokens.ndim != 2:
        return 0.0
    if query_tokens.shape[1] != doc_tokens.shape[1]:
        return 0.0
    sims = torch.matmul(query_tokens, doc_tokens.T)
    max_per_query = sims.max(dim=1).values
    return float(max_per_query.sum().item())


def _keyword_overlap_score(query: str, *texts: str) -> float:
    query_tokens = {
        token for token in re.split(r"[^a-zA-Z0-9_]+", (query or "").lower()) if token
    }
    if not query_tokens:
        return 0.0
    doc_tokens = {
        token
        for text in texts
        for token in re.split(r"[^a-zA-Z0-9_]+", str(text or "").lower())
        if token
    }
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(doc_tokens))
    return overlap / max(1, len(query_tokens))


def _rank_segments_by_fused_similarity(
    query: str,
    segments: Dict[str, Dict[str, Any]],
    video_name: str,
    top_k: int,
    embedding_device: Optional[str] = None,
    embedding_provider: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rank segments for planner grounding using caption + transcript only.

    Graph text is intentionally excluded from the ranking score because it can
    introduce noisy entity/relation matches that hurt first-step segment recall.
    """
    query_tokens = _encode_grounding_query_embedding(
        query,
        embedding_device=embedding_device,
        embedding_provider=embedding_provider,
    )
    ranked: List[Dict[str, Any]] = []

    for seg_id, seg in segments.items():
        if not isinstance(seg, dict):
            continue
        if "caption" not in seg or "transcript" not in seg:
            continue
        caption = str(seg.get("caption", "")).strip()
        transcript = str(seg.get("transcript", "")).strip()
        if not caption or caption.lower().startswith("visual caption unavailable"):
            continue
        modality_scores: Dict[str, float] = {}
        if caption:
            modality_scores["caption"] = _late_interaction_score(
                query_tokens,
                _encode_grounding_text_embedding(
                    caption,
                    embedding_device=embedding_device,
                    embedding_provider=embedding_provider,
                ),
            )
        if transcript:
            modality_scores["transcript"] = _late_interaction_score(
                query_tokens,
                _encode_grounding_text_embedding(
                    transcript,
                    embedding_device=embedding_device,
                    embedding_provider=embedding_provider,
                ),
            )
        primary_scores = [
            modality_scores[key]
            for key in ("caption", "transcript")
            if key in modality_scores
        ]

        if primary_scores:
            semantic_score = float(sum(primary_scores) / len(primary_scores))
        else:
            semantic_score = -1.0
        lexical_score = _keyword_overlap_score(query, caption, transcript)
        fused_score = semantic_score + (0.35 * lexical_score)

        ranked.append(
            {
                "segment_id": f"{video_name}_{seg_id}",
                "segment_ref": f"{video_name}_{seg_id}",
                "raw_segment_id": str(seg_id),
                "time": seg.get("time", ""),
                "time_range": seg.get("time", ""),
                "caption": caption,
                "transcript": transcript,
                "frame_times": seg.get("frame_times", []),
                "frame_paths": seg.get("frame_paths", []),
                "score": fused_score,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "modality_scores": {
                    "caption": modality_scores.get("caption", 0.0),
                    "transcript": modality_scores.get("transcript", 0.0),
                },
            }
        )

    ranked.sort(key=lambda item: float(item.get("score", -1.0)), reverse=True)
    return ranked[:max(1, top_k)]


def _rank_segments_by_cached_vectors(
    *,
    query: str,
    processed_dir: Path,
    top_k: int,
    embedding_device: Optional[str],
    embedding_provider: Optional[str],
) -> List[Dict[str, Any]]:
    vectors_dir = Path(processed_dir) / "vectors"
    if not vectors_dir.exists():
        return []
    try:
        from preprocessing.storage.vector_store import DenseVectorStore

        store_device = "cpu" if str(embedding_device or "").strip().lower() == "cpu" else None
        vector_store = DenseVectorStore(device=store_device)
        vector_store.load(str(vectors_dir))
        query_vector = _encode_grounding_query_embedding(
            query,
            embedding_device=embedding_device,
            embedding_provider=embedding_provider,
        )
        raw_results = vector_store.search(query_vector, k=max(1, top_k))
    except Exception as exc:
        logger.warning(
            "Planner grounding cached vector search failed at %s; falling back to text reranker. error=%s",
            vectors_dir,
            exc,
        )
        return []

    ranked: List[Dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        caption = str(item.get("caption", "")).strip()
        transcript = str(item.get("transcript", "")).strip()
        lexical_score = _keyword_overlap_score(query, caption, transcript)
        semantic_score = float(item.get("similarity_score", 0.0) or 0.0)
        ranked.append(
            {
                "segment_id": str(item.get("segment_id") or item.get("segment_ref") or ""),
                "segment_ref": str(item.get("segment_ref") or item.get("segment_id") or ""),
                "raw_segment_id": str(item.get("raw_segment_id") or ""),
                "time": item.get("time", item.get("time_range", "")),
                "time_range": item.get("time_range", item.get("time", "")),
                "caption": caption,
                "transcript": transcript,
                "frame_times": item.get("frame_times", []),
                "frame_paths": item.get("frame_paths", []),
                "score": semantic_score + (0.35 * lexical_score),
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "modality_scores": dict(item.get("modality_scores") or {}),
            }
        )
    ranked.sort(key=lambda item: float(item.get("score", -1.0)), reverse=True)
    return ranked[:max(1, top_k)]


def _segment_sort_key(item: Dict[str, Any]) -> tuple[float, str]:
    bounds = _parse_time_range_bounds(item.get("time") or item.get("time_range"))
    start = float(bounds[0]) if bounds is not None else float("inf")
    ref = str(item.get("segment_ref") or item.get("segment_id") or "")
    return (start, ref)


def _sample_grounding_frame_indices(
    frame_times: List[Any],
    interval_seconds: float,
) -> List[int]:
    if not isinstance(frame_times, list) or not frame_times:
        return []
    interval = max(0.1, float(interval_seconds))
    selected: List[int] = []
    last_selected_time: Optional[float] = None

    for index, raw_time in enumerate(frame_times):
        try:
            timestamp = float(raw_time)
        except Exception:
            continue
        if last_selected_time is None or (timestamp - last_selected_time) >= (interval - 1e-6):
            selected.append(index)
            last_selected_time = timestamp

    if not selected:
        for index, raw_time in enumerate(frame_times):
            try:
                float(raw_time)
            except Exception:
                continue
            selected.append(index)
            break
    return selected


def _build_grounding_segment_image_evidence(
    item: Dict[str, Any],
    *,
    video_name: str,
) -> List[Dict[str, Any]]:
    visual_helpers = _get_visual_helpers()
    frame_paths = list(item.get("frame_paths") or [])
    frame_times = list(item.get("frame_times") or [])
    if not frame_paths or not frame_times:
        return []

    selected_indices = _sample_grounding_frame_indices(
        frame_times,
        interval_seconds=float(getattr(settings, "planner_grounding_frame_interval_seconds", 5.0) or 5.0),
    )
    raw_evidence: List[Dict[str, Any]] = []
    for index in selected_indices:
        if index >= len(frame_paths) or index >= len(frame_times):
            continue
        resolved_path = visual_helpers["resolve_processed_frame_image_path"](video_name, frame_paths[index])
        if not resolved_path:
            continue
        try:
            timestamp = round(float(frame_times[index]), 3)
        except Exception:
            continue
        raw_evidence.append(
            {
                "image": resolved_path,
                "timestamp": timestamp,
                "sample_times": [timestamp],
                "source_images": [resolved_path],
                "pack_mode": "cached_single",
            }
        )
    if not raw_evidence:
        return []
    return visual_helpers["pack_planner_image_evidence"](
        raw_evidence,
        text_hint=f"grounding {item.get('time', '')} {item.get('caption', '')}",
    )


def _grounding_payload_to_text(payload: Any) -> str:
    if isinstance(payload, list):
        parts: List[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type == "text":
                text_value = " ".join(str(item.get("text", "")).split()).strip()
                if text_value:
                    parts.append(text_value)
            elif item_type == "image":
                parts.append("[image]")
        return "\n".join(parts).strip()
    return str(payload or "").strip()


def _truncate_grounding_think_tail(text: Any) -> str:
    return truncate_think_tail_preserving_outer_text(text)


def _sanitize_grounding_caption_text(text: Any) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return ""

    normalized = re.sub(r"<think>.*?</think>", " ", normalized, flags=re.IGNORECASE | re.DOTALL)
    normalized = re.sub(r"</?think>", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[*_`]+", "", normalized)
    normalized = re.sub(
        r"^The user wants a description of (?:the visual content in )?the provided (?:video segment|frames?)\.?\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"^\d+\.\s*Analyze the (?:frames?|images?):\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(^|\s)(?:frame|frames|image|images)\s*\d+(?:\s*(?:[-,&]|and)\s*\d+)*\s*:\s*",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"(^|\s)(?:the\s+)?(?:image|picture|frame|scene|video)\s+(?:shows|displays|depicts|features|contains)\s+", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(^|\s)(?:it|this)\s+(?:shows|displays|depicts|features|contains)\s+", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(^|\s)(?:these|they)\s+(?:show|display|depict|feature|contain)\s+", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _infer_grounding_video_end_time(
    preprocessing_result: Dict[str, Any],
    candidate_results: Optional[List[Dict[str, Any]]] = None,
) -> Optional[float]:
    def _extract_end(value: Any) -> Optional[float]:
        if isinstance(value, dict):
            end_value = value.get("end")
            try:
                return float(end_value) if end_value not in (None, "") else None
            except Exception:
                return None
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[1])
            except Exception:
                return None
        text = str(value or "").strip()
        if not text or "-" not in text:
            return None
        try:
            return float(text.split("-", 1)[1].strip())
        except Exception:
            return None

    max_end = 0.0
    segments = preprocessing_result.get("segments", {}) if isinstance(preprocessing_result, dict) else {}
    segment_items = segments.values() if isinstance(segments, dict) else []
    for item in segment_items:
        if not isinstance(item, dict):
            continue
        for field in ("timestamp", "time"):
            end_value = _extract_end(item.get(field))
            if end_value is not None:
                max_end = max(max_end, float(end_value))
    for item in list(candidate_results or []):
        if not isinstance(item, dict):
            continue
        for field in ("timestamp", "time_range", "time"):
            end_value = _extract_end(item.get(field))
            if end_value is not None:
                max_end = max(max_end, float(end_value))
    return round(max_end, 3) if max_end > 0 else None


def _resolve_grounding_source_video_path(
    preprocessing_result: Dict[str, Any],
    processed_dir: Path,
) -> Optional[str]:
    direct_video_path = str((preprocessing_result or {}).get("video_path") or "").strip()
    if direct_video_path and Path(direct_video_path).expanduser().exists():
        return str(Path(direct_video_path).expanduser().resolve())

    metadata_path = Path(processed_dir) / "video_metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate = str(payload.get("source_video_path") or "").strip()
            if candidate and Path(candidate).expanduser().exists():
                return str(Path(candidate).expanduser().resolve())
        except Exception as exc:
            logger.warning("Failed to load grounding video metadata from %s: %s", metadata_path, exc)

    query_state = runtime_session.snapshot_query_state()
    active_video_path = str((query_state or {}).get("active_video_path") or "").strip()
    if active_video_path and Path(active_video_path).expanduser().exists():
        return str(Path(active_video_path).expanduser().resolve())
    return None


def _build_grounding_video_clip_qa_baseline(
    *,
    query: str,
    preprocessing_result: Dict[str, Any],
    processed_dir: Path,
    video_end_time: Optional[float],
) -> Optional[Dict[str, str]]:
    if video_end_time is None or float(video_end_time) <= 0:
        return None

    video_path = _resolve_grounding_source_video_path(preprocessing_result, processed_dir)
    if not video_path:
        logger.info("Planner grounding skipped Video_Clip_QA baseline because no source video path was available.")
        return None

    visual_helpers = _get_visual_helpers()
    try:
        qa_result = visual_helpers["run_video_clip_qa"](
            video_path=video_path,
            start=0.0,
            end=float(video_end_time),
            query=query,
            model_path=settings.qwen_vl_model_path,
        )
    except Exception as exc:
        logger.warning("Planner grounding Video_Clip_QA baseline failed for %s: %s", video_path, exc)
        return None

    baseline_answer = _truncate_grounding_think_tail(
        (qa_result or {}).get("answer")
        or (qa_result or {}).get("result")
        or ""
    ).strip()
    if not baseline_answer:
        logger.info(
            "Planner grounding Video_Clip_QA baseline returned no usable answer for %s. payload_keys=%s",
            video_path,
            sorted((qa_result or {}).keys()) if isinstance(qa_result, dict) else [],
        )
        return None

    warning_text = " ".join(str((qa_result or {}).get("warning") or "").split()).strip()
    return {
        "time_range": f"0.0-{float(video_end_time):.3f}",
        "answer": baseline_answer,
        "warning": warning_text,
    }


def _build_planner_grounding_context(
    query: str,
    preprocessing_result: Dict[str, Any],
    processed_dir: Path,
    top_k: int = 1,
    embedding_device: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    include_video_clip_qa: bool = True,
) -> Any:
    """Build planner grounding context from caption evidence plus frame images."""
    visual_helpers = _get_visual_helpers()
    candidate_results: List[Dict[str, Any]] = []
    segments = preprocessing_result.get("segments", {}) if isinstance(preprocessing_result, dict) else {}
    video_name = str(
        (preprocessing_result or {}).get("video_name")
        or processed_dir.name
    ).strip() or processed_dir.name
    sparse_threshold = max(1, int(getattr(settings, "sparse_video_segment_threshold", 3) or 3))

    if isinstance(segments, dict) and segments:
        resolved_embedding_provider = _resolve_grounding_embedding_provider(embedding_provider)
        if resolved_embedding_provider == "local" and str(embedding_device or "").strip().lower() == "cpu":
            candidate_results = _rank_segments_by_cached_vectors(
                query=query,
                processed_dir=Path(processed_dir),
                top_k=max(top_k, 5),
                embedding_device=embedding_device,
                embedding_provider=resolved_embedding_provider,
            )
        if not candidate_results:
            candidate_results = _rank_segments_by_fused_similarity(
                query=query,
                segments=segments,
                video_name=video_name,
                top_k=max(top_k, 5),
                embedding_device=embedding_device,
                embedding_provider=resolved_embedding_provider,
            )

    if not candidate_results:
        raise RuntimeError(
            "Planner grounding requires valid preprocessing captions and transcripts. "
            "No usable caption+transcript evidence blocks were produced."
        )

    use_all_sparse_segments = len(candidate_results) <= sparse_threshold
    if use_all_sparse_segments:
        topk_results = sorted(candidate_results, key=_segment_sort_key)
    else:
        topk_results = candidate_results[:max(1, top_k)]
    logger.info(
        "Planner grounding selected %d top segments using %s.",
        len(topk_results),
        "sparse-video all-segment grounding"
        if use_all_sparse_segments
        else "caption+transcript fused reranker",
    )
    if candidate_results:
        top5_refs = [
            f"{item.get('segment_ref') or item.get('segment_id')}@{item.get('time_range') or item.get('time')}"
            for item in candidate_results[:5]
        ]
        logger.info("Top5 segment refs for traceback: %s", top5_refs)
    if topk_results:
        prompt_ready_results = [
            {
                "time": item.get("time", ""),
                "caption": _sanitize_grounding_caption_text(item.get("caption", "")),
                "transcript": item.get("transcript", ""),
                "frame_times": item.get("frame_times", []),
            }
            for item in topk_results
        ]
        logger.debug(
            "[PLANNER_GROUNDING_SOURCE_BEGIN]\n%s\n[PLANNER_GROUNDING_SOURCE_END]",
            json.dumps(prompt_ready_results, ensure_ascii=False, indent=2),
        )

    display_results = topk_results if use_all_sparse_segments else topk_results[:top_k]
    content: List[Dict[str, Any]] = []
    entire_video_end_time = _infer_grounding_video_end_time(
        preprocessing_result,
        candidate_results=candidate_results,
    )
    intro_lines: List[str] = [
        "[Initial Planner Grounding]",
        f"query: {_truncate_text(query, max_chars=800)}",
        (
            f"entire_video_time: [0.0, {entire_video_end_time:.3f}]"
            if entire_video_end_time is not None
            else "entire_video_time: [0.0, unknown]"
        ),
        (
            "[Grounding Source] Sparse-video all-segment evidence."
            if use_all_sparse_segments
            else "[Grounding Source] Query-retrieved top-k caption evidence."
        ),
        (
            f"Active video contains only {len(candidate_results)} segment(s). All available segments are shown."
            if use_all_sparse_segments
            else f"Using top {len(display_results)} segment(s) from caption-based fused retrieval."
        ),
        f"frame_sampling_rule: approximately {float(getattr(settings, 'planner_grounding_frame_interval_seconds', 5.0) or 5.0):.1f}s per frame.",
        (
            "For both initial grounding collages and later tool-returned collages, "
            "sub-frames are arranged in chronological order clockwise from the top-left. "
            "The accompanying timestamps correspond to those sub-frames in that same order."
        ),
        "Start first-step reasoning from these caption+image grounding blocks and use their time windows for follow-up probing.",
    ]
    content.append({"type": "text", "text": "\n".join(intro_lines)})

    if include_video_clip_qa:
        baseline_video_qa = _build_grounding_video_clip_qa_baseline(
            query=query,
            preprocessing_result=preprocessing_result,
            processed_dir=processed_dir,
            video_end_time=entire_video_end_time,
        )
        if baseline_video_qa is not None:
            baseline_lines = [
                "[Grounding Video_Clip_QA Baseline]",
                f"time: {baseline_video_qa['time_range']}",
                "answer:",
                baseline_video_qa["answer"],
            ]
            if baseline_video_qa.get("warning"):
                baseline_lines.append("warning: " + _truncate_text(baseline_video_qa["warning"], max_chars=600))
            content.append({"type": "text", "text": "\n".join(baseline_lines)})

    total_image_limit = visual_helpers["planner_image_attachment_limit"](text_hint="\n".join(intro_lines))
    remaining_image_budget = max(1, total_image_limit)

    for i, item in enumerate(display_results, 1):
        sanitized_caption = _sanitize_grounding_caption_text(item.get("caption", ""))
        transcript_text = str(item.get("transcript", "") or "").strip()
        segment_images = _build_grounding_segment_image_evidence(item, video_name=video_name)
        if remaining_image_budget <= 0:
            segment_images = []
        elif len(segment_images) > remaining_image_budget:
            segment_images = segment_images[:remaining_image_budget]
        sanitized_segment_images: List[Dict[str, Any]] = []
        for image_item in segment_images:
            prepared_item = visual_helpers["prepare_planner_image_attachment"](image_item)
            if prepared_item is not None:
                sanitized_segment_images.append(prepared_item)
        segment_images = sanitized_segment_images
        remaining_image_budget = max(0, remaining_image_budget - len(segment_images))

        image_stats = visual_helpers["summarize_planner_image_evidence"](segment_images)
        image_timestamp_text = json.dumps(image_stats.get("image_timestamps", []), ensure_ascii=False)
        segment_lines = [
            f"[Grounding Segment #{i}]",
            f"time: {_truncate_text(item.get('time', ''), max_chars=120)}",
            (
                "caption: "
                + _truncate_text(sanitized_caption, max_chars=1200)
                + f" | attached_frame_timestamps: {image_timestamp_text}"
            ),
            "transcript: " + (
                _truncate_text(transcript_text, max_chars=1800)
                if transcript_text
                else "None"
            ),
        ]
        content.append({"type": "text", "text": "\n".join(segment_lines)})
        for image_item in segment_images:
            image_path = str(image_item.get("image", "") or "").strip()
            if image_path:
                content.append(
                    {
                        "type": "image",
                        "image": image_path,
                        "timestamp": image_item.get("timestamp"),
                        "sample_times": list(image_item.get("sample_times") or []),
                        "source_images": list(image_item.get("source_images") or []),
                        "image_order_rule": image_item.get("image_order_rule"),
                    }
                )

    return content


def _load_planner_direct_answer_prompt(prompt_path: Any) -> str:
    resolved_path = Path(str(prompt_path or "")).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Planner direct-answer prompt file not found: {resolved_path}")
    prompt_text = resolved_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Planner direct-answer prompt file is empty: {resolved_path}")
    return prompt_text


def _planner_provider_name(llm: Any) -> str:
    return str(
        getattr(llm, "provider_name", "")
        or getattr(llm, "provider", "")
        or getattr(llm, "llm_provider", "")
        or ""
    ).strip().lower()


def _planner_direct_generation_kwargs(llm: Any) -> Dict[str, Any]:
    generation_kwargs = dict(
        settings.get_generation_config(
            model_role="planner",
            provider=_planner_provider_name(llm),
        )
        or {}
    )
    max_new_tokens = generation_kwargs.pop("max_new_tokens", None)
    if max_new_tokens is not None:
        generation_kwargs["max_tokens"] = max_new_tokens
    generation_kwargs["enable_prefix_caching"] = False
    generation_kwargs["enable_rollout_prefix_cache"] = False
    if _planner_provider_name(llm) in {"transformers", "local"}:
        generation_kwargs.pop("presence_penalty", None)
    return generation_kwargs


def _extract_direct_answer_text(raw_response: Any) -> str:
    cleaned = truncate_think_tail_preserving_outer_text(str(raw_response or "")).strip()
    if not cleaned:
        return ""
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(cleaned)
    except Exception:
        return cleaned
    if isinstance(payload, dict):
        finish_payload = payload.get("Finish")
        if isinstance(finish_payload, dict):
            answer = str(finish_payload.get("answer") or "").strip()
            if answer:
                return answer
        answer = str(payload.get("answer") or "").strip()
        if answer:
            return answer
    return cleaned


def _write_direct_raw_trace(
    *,
    raw_trace_jsonl_path: Optional[Path],
    query: str,
    raw_response: str,
) -> None:
    if raw_trace_jsonl_path is None:
        return
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "depth": 0,
        "iteration": 1,
        "raw_response": raw_response,
    }
    raw_trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_trace_jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_direct_answer_trajectory(
    *,
    query: str,
    final_answer: str,
    raw_response: str,
    started_at: float,
    started_at_iso: str,
    finished_at_iso: str,
) -> Dict[str, Any]:
    finished_at = time.perf_counter()
    finish_args = {
        "answer": final_answer,
        "chain_complete": True,
        "completion_basis": "grounding->answer",
    }
    return {
        "task_id": f"task_direct_{int(started_at * 1000):x}",
        "goal": query,
        "depth": 0,
        "steps": [
            {
                "step_id": 1,
                "depth": 0,
                "thought": "Single-turn planner direct answer from preprocessing grounding.",
                "plan": "Use the provided grounding evidence and answer immediately without tool calls.",
                "action": "Finish",
                "action_args": finish_args,
                "observation": final_answer,
                "observation_items": [
                    {
                        "tool": "Finish",
                        "status": "success",
                        "result": final_answer,
                    }
                ],
                "timestamp": finished_at_iso,
                "is_recursive": False,
                "sub_trajectory": None,
                "sub_trajectories": None,
            }
        ],
        "final_answer": final_answer,
        "success": True,
        "error_message": None,
        "debug_events": [
            {
                "type": "llm_iteration",
                "depth": 0,
                "iteration": 1,
                "response": _extract_direct_answer_text(raw_response),
                "raw_response_chars": len(str(raw_response or "")),
                "response_normalized": _extract_direct_answer_text(raw_response) != str(raw_response or "").strip(),
                "parsed_action": "Finish",
                "parsed_action_args": finish_args,
                "parse_success": True,
            }
        ],
        "start_time": started_at_iso,
        "end_time": finished_at_iso,
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        "duration_seconds": max(0.0, finished_at - started_at),
    }


def run_query_direct(
    query: str,
    *,
    grounding_context: Any,
    planner_prompt_path: Any,
    device_type: Optional[str] = None,
    raw_trace_jsonl_path: Optional[Path] = None,
) -> tuple[str, Dict[str, Any], bool]:
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Running Planner Direct Answer")
    logger.info("=" * 80)
    logger.info("Query: %s", query)
    logger.info("-" * 80)

    planner_llm = _get_cached_llm_provider(
        model_name=settings.planner_model_path,
        model_role="planner",
        device=device_type,
    )
    system_prompt = _load_planner_direct_answer_prompt(planner_prompt_path)
    generation_kwargs = _planner_direct_generation_kwargs(planner_llm)

    if isinstance(grounding_context, list):
        user_content = list(grounding_context)
        user_content.append(
            {
                "type": "text",
                "text": "[User Query]\n" + query,
            }
        )
    elif grounding_context:
        user_content = (
            _grounding_payload_to_text(grounding_context).strip()
            + "\n\n[User Query]\n"
            + query
        )
    else:
        user_content = query

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    grounding_text = _grounding_payload_to_text(grounding_context) if grounding_context else ""
    grounding_image_count = sum(
        1
        for item in grounding_context
        if isinstance(item, dict) and str(item.get("type", "")).strip().lower() == "image"
    ) if isinstance(grounding_context, list) else 0
    logger.info(
        "Planner direct-answer grounding injected (%d chars, %d image item(s))",
        len(grounding_text),
        grounding_image_count,
    )
    logger.info(
        "[PLANNER_CONTEXT_BEGIN]\n%s\n[PLANNER_CONTEXT_END]",
        grounding_text,
    )

    started_at = time.perf_counter()
    started_at_iso = datetime.utcnow().isoformat() + "Z"
    raw_response = planner_llm.generate(messages=messages, **generation_kwargs)
    runtime_session.record_timing("query.solve_seconds", time.perf_counter() - started_at)
    _write_direct_raw_trace(
        raw_trace_jsonl_path=raw_trace_jsonl_path,
        query=query,
        raw_response=str(raw_response or ""),
    )
    logger.info(
        "[PlannerDirect][LLM_FULL_RESPONSE_BEGIN]\n%s\n[PlannerDirect][LLM_FULL_RESPONSE_END]",
        str(raw_response or "").strip(),
    )

    final_answer = _extract_direct_answer_text(raw_response)
    if not final_answer:
        raise RuntimeError("Planner direct-answer mode returned an empty response.")

    trajectory = _build_direct_answer_trajectory(
        query=query,
        final_answer=final_answer,
        raw_response=str(raw_response or ""),
        started_at=started_at,
        started_at_iso=started_at_iso,
        finished_at_iso=datetime.utcnow().isoformat() + "Z",
    )
    logger.info("\n" + "-" * 80)
    logger.info("FINAL RESULT (success):")
    logger.info(final_answer)
    logger.info("=" * 80)
    return final_answer, trajectory, True


def _log_trajectory(trajectory: Optional[Dict]) -> None:
    """Pretty-print trajectory information to the logger."""
    if not trajectory:
        logger.warning("No trajectory information available from engine.")
        return
    
    def _log_step(step: Dict, indent: int = 0):
        pad = "  " * indent
        logger.info(
            "%s[Step %s | Depth %s] Action=%s Args=%s",
            pad,
            step.get("step_id"),
            step.get("depth"),
            step.get("action"),
            step.get("action_args")
        )
        logger.info("%sThought: %s", pad, step.get("thought"))
        logger.info("%sObservation: %s", pad, step.get("observation"))
        
        if step.get("sub_trajectory"):
            logger.info("%s-> Entering sub-trajectory", pad)
            for sub_step in step["sub_trajectory"].get("steps", []):
                _log_step(sub_step, indent + 1)
        for index, sub_trajectory in enumerate(step.get("sub_trajectories") or [], start=1):
            if not isinstance(sub_trajectory, dict):
                continue
            logger.info("%s-> Entering sub-trajectory #%d", pad, index)
            for sub_step in sub_trajectory.get("steps", []):
                _log_step(sub_step, indent + 1)
    
    logger.info("-" * 80)
    logger.info("Detailed Tool Execution Trace")
    logger.info("-" * 80)
    
    for step in trajectory.get("steps", []):
        _log_step(step)
    
    logger.info("-" * 80)
    logger.info("End of trajectory trace")
    logger.info("-" * 80)


def _json_safe(value: Any) -> Any:
    if value is Ellipsis:
        return "..."
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _build_run_result_payload(
    *,
    workspace: Path,
    video_path: str,
    query: str,
    preprocessing_result: Dict,
    processed_dir: Path,
    trajectory: Optional[Dict],
    final_answer: str,
    success: bool,
) -> Dict[str, Any]:
    agent_metrics = _build_agent_metrics(trajectory)
    runtime_metrics = runtime_session.get_query_metrics()
    report = {
        "video_path": video_path,
        "query": query,
        "workspace": str(workspace),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "success": bool(success),
        "preprocessing": {
            "video_name": preprocessing_result.get("video_name"),
            "num_segments": preprocessing_result.get("num_segments"),
            "processed_output_dir": str(processed_dir),
            "segment_index_sample": list(preprocessing_result.get("segments", {}).keys())[:5],
        },
        "agent": {
            "final_answer": final_answer,
            "trajectory": trajectory
        },
        "metrics": {
            **agent_metrics,
            "timing_breakdown_seconds": runtime_metrics.get("timing_breakdown_seconds", {}),
            "cache_stats": runtime_metrics.get("cache_stats", {}),
            "active_video_names": runtime_metrics.get("active_video_names", []),
        },
    }
    return _json_safe(report)


async def preprocess_video(
    video_path: str,
    working_dir: Path,
    device_type: Optional[str] = None,
    embedding_device: Optional[str] = None,
    reuse_cache: bool = True,
    accelerator_lock: Any = None,
) -> Dict:
    """Preprocess the input video."""
    VideoPreprocessingPipeline, PipelineConfig, _, _ = _get_preprocessing_module_exports()
    runtime_session.set_active_video_context(
        video_name=Path(video_path).stem,
        video_path=video_path,
        processed_dir=working_dir / "processed" / Path(video_path).stem,
    )
    logger.info("=" * 80)
    logger.info("Step 1: Video Preprocessing")
    logger.info("=" * 80)
    
    config = PipelineConfig(
        device=device_type,
        working_dir=str(working_dir),
        use_scene_detection=True,
        save_intermediate=True,
        enable_knowledge_graph=False,
        enable_metadata_synthesis=False,
        embedding_device=embedding_device,
        reuse_cache=reuse_cache,
    )
    needs_preprocess_llm = bool(config.enable_knowledge_graph or config.enable_metadata_synthesis)
    if not needs_preprocess_llm:
        logger.info(
            "Skipping preprocess text LLM load because knowledge-graph and metadata synthesis are both disabled."
        )
    
    video_name = Path(video_path).stem
    output_dir = working_dir / "processed" / video_name
    cache_dir = working_dir / "_cache" / video_name
    cache_probe_pipeline = VideoPreprocessingPipeline(
        config=config,
        llm_provider=None,
        accelerator_lock=accelerator_lock,
    )
    preprocess_started = time.perf_counter()

    if not reuse_cache:
        if cache_dir.exists():
            logger.info("Removing cached preprocessing directory before rebuild: %s", cache_dir)
            shutil.rmtree(cache_dir)
        if output_dir.exists():
            logger.info("Removing processed output directory before rebuild: %s", output_dir)
            shutil.rmtree(output_dir)

    if output_dir.exists() and reuse_cache:
        required_artifacts = _required_preprocessing_artifacts(output_dir)
        missing_artifacts = [
            path.relative_to(output_dir)
            for path in required_artifacts
            if not path.exists()
        ]

        cached_payload = None
        segments: Dict[str, Any] = {}
        is_stale = False
        if not missing_artifacts:
            cached_payload = cache_probe_pipeline._try_load_processed(output_dir, video_name)
            runtime_session.record_cache("preprocess.output_dir", cached_payload is not None)
            if cached_payload is None:
                is_stale = True
            else:
                segments = cached_payload.get("segments", {}) or {}
                is_stale = _looks_stale_outputs(segments) or not _segment_cache_is_current(
                    working_dir,
                    Path(video_path).stem,
                )
        else:
            runtime_session.record_cache("preprocess.output_dir", False)

        if missing_artifacts or is_stale:
            config.reuse_cache = False
            if missing_artifacts:
                logger.warning(
                    "Processed output directory %s is incomplete; missing artifacts=%s. Regenerating.",
                    output_dir,
                    [str(path) for path in missing_artifacts],
                )
            else:
                logger.warning(
                    "Detected stale preprocessing artifacts at %s; rebuilding downstream outputs while preserving reusable caches.",
                    output_dir,
                )
            preprocess_llm = (
                _get_cached_llm_provider(
                    model_name=settings.preprocess_model_path,
                    model_role="preprocess",
                    device=device_type,
                )
                if needs_preprocess_llm
                else None
            )
            pipeline = VideoPreprocessingPipeline(
                config=config,
                llm_provider=preprocess_llm,
                accelerator_lock=accelerator_lock,
            )
            logger.info("Re-processing video due to stale cache: %s", video_path)
            result = await pipeline.process_video(video_path)
        else:
            logger.info("Found existing processed outputs at %s, skipping preprocessing.", output_dir)
            result = {
                "video_name": Path(video_path).stem,
                "video_path": video_path,
                "segments": segments,
                "num_segments": len(segments),
            }
    else:
        config.reuse_cache = reuse_cache
        preprocess_llm = (
            _get_cached_llm_provider(
                model_name=settings.preprocess_model_path,
                model_role="preprocess",
                device=device_type,
            )
            if needs_preprocess_llm
            else None
        )
        pipeline = VideoPreprocessingPipeline(
            config=config,
            llm_provider=preprocess_llm,
            accelerator_lock=accelerator_lock,
        )
        if reuse_cache:
            logger.info("Processing video: %s", video_path)
        else:
            logger.info("Processing video with cache reuse disabled: %s", video_path)
        result = await pipeline.process_video(video_path)
        logger.info("\n[OK] Preprocessing complete!")
        logger.info("  Video: %s", result["video_name"])
        logger.info("  Segments: %s", result["num_segments"])
        logger.info("  Output directory: %s", output_dir)
    
    runtime_session.record_timing("preprocess.total_seconds", time.perf_counter() - preprocess_started)
    runtime_session.set_active_video_context(
        video_name=str(result.get("video_name") or video_name),
        video_path=video_path,
        processed_dir=output_dir,
        segment_count=len(result.get("segments", {}) or {}),
    )

    return {
        "result": result,
        "output_dir": output_dir
    }


def run_query(
    query: str,
    grounding_context: Any = "",
    device_type: Optional[str] = None,
    raw_trace_jsonl_path: Optional[Path] = None,
    runtime_profile: str = _PIPELINE_RUNTIME_PROFILE_DEFAULT,
    planner_toolsearch_direct_execution: bool = False,
    planner_toolsearch_direct_single_tool_only: bool = False,
    planner_single_action_per_step: bool = False,
):
    """Run the ReAct query and return result plus trajectory."""
    registry = _get_registry()
    RecursiveEngine = _get_recursive_engine()
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Running ReAct Query")
    logger.info("=" * 80)
    logger.info("Query: %s", query)
    logger.info("-" * 80)
    
    # Initialize tools
    logger.info("Initializing tools...")
    registry_started = time.perf_counter()
    runtime_session.ensure_registry_discovered(registry)
    runtime_session.record_timing("query.registry_init_seconds", time.perf_counter() - registry_started)
    base_tools = registry.get_all_tools(tags=["base"])
    meta_tools = registry.get_all_tools(tags=["meta"])
    logger.info(
        "Registered tools summary: base=%d, meta=%d, total=%d",
        len(base_tools),
        len(meta_tools),
        len(registry.get_all_tools()),
    )
    
    # Create LLM provider (local by default per settings)
    planner_llm = _get_cached_llm_provider(
        model_name=settings.planner_model_path,
        model_role="planner",
        device=device_type,
    )
    planner_multimodal_enabled = bool(
        getattr(planner_llm, "processor", None) is not None
        and getattr(planner_llm, "_supports_multimodal_generation", False)
    )
    logger.info(
        "Planner multimodal input enabled: %s (graph_enable=%s, web_search_enabled=%s)",
        planner_multimodal_enabled,
        bool(settings.graph_enable),
        bool(settings.web_search_enabled),
    )

    # Create engine
    engine_kwargs = _runtime_profile_engine_kwargs(runtime_profile)
    if planner_single_action_per_step:
        engine_kwargs["planner_single_action_per_step"] = True
    if planner_toolsearch_direct_execution:
        engine_kwargs.update({
            "planner_toolsearch_direct_execution": True,
            "planner_toolsearch_direct_single_tool_only": bool(
                planner_toolsearch_direct_single_tool_only
            ),
        })
    engine = RecursiveEngine(
        llm=planner_llm,
        raw_trace_jsonl_path=str(raw_trace_jsonl_path) if raw_trace_jsonl_path else None,
        session_query=query,
        **engine_kwargs,
    )
    
    # Solve the task
    if grounding_context:
        grounding_text = _grounding_payload_to_text(grounding_context)
        grounding_image_count = sum(
            1
            for item in grounding_context
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() == "image"
        ) if isinstance(grounding_context, list) else 0
        logger.info(
            "Planner grounding context injected (%d chars, %d image item(s))",
            len(grounding_text),
            grounding_image_count,
        )
        logger.info(
            "[PLANNER_CONTEXT_BEGIN]\n%s\n[PLANNER_CONTEXT_END]",
            grounding_text,
        )

    query_started = time.perf_counter()
    result = engine.solve(goal=query, extra_context=grounding_context)
    runtime_session.record_timing("query.solve_seconds", time.perf_counter() - query_started)
    trajectory = engine.get_full_trajectory()
    success = bool(trajectory and trajectory.get("success"))
    
    _log_trajectory(trajectory)
    
    logger.info("\n" + "-" * 80)
    logger.info("FINAL RESULT (%s):", "success" if success else "failed")
    logger.info(result)
    logger.info("=" * 80)
    
    return result, trajectory, success


async def _execute_pipeline_impl(
    *,
    video_path: str,
    query: str,
    workspace_root: Path,
    shared_preprocess_root: Optional[Path] = None,
    no_cache: bool = False,
    device_type: Optional[str] = None,
    write_log_file: bool = True,
    log_path: Optional[Path] = None,
    persist_query_workspace: bool = True,
    runtime_profile: str = _PIPELINE_RUNTIME_PROFILE_DEFAULT,
    pipeline_mode: str = "react",
    planner_direct_answer_prompt_path: Optional[str] = None,
    include_grounding_video_clip_qa: bool = True,
    planner_toolsearch_direct_single_tool_only: bool = False,
    planner_single_action_per_step: bool = False,
    runtime_disabled_tool_names: Optional[List[str]] = None,
    runtime_disable_meta_tools: bool = False,
) -> Dict[str, Any]:
    _ensure_non_planner_runtime_ready_for_e2e()
    runtime_session.reset_query_state()
    workspace_root = Path(workspace_root).resolve()
    shared_preprocess_root = Path(shared_preprocess_root).resolve() if shared_preprocess_root else workspace_root
    file_handler: Optional[logging.Handler] = None

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if persist_query_workspace:
        query_run_info = _create_query_run_workspace(workspace_root, video_path, query)
        query_group_dir = Path(query_run_info["query_group_dir"]).resolve()
        query_workspace = Path(query_run_info["run_workspace"]).resolve()
        artifact_paths = query_run_info["artifact_paths"]
        run_id = str(query_run_info["run_id"])
        video_slug = str(query_run_info["video_slug"])
        query_slug = str(query_run_info["query_slug"])
    else:
        query_workspace = workspace_root.resolve()
        query_workspace.mkdir(parents=True, exist_ok=True)
        query_group_dir = query_workspace
        artifact_paths = _build_query_run_artifact_paths(query_workspace)
        for directory in (
            artifact_paths["inputs_dir"],
            artifact_paths["logs_dir"],
            artifact_paths["results_dir"],
        ):
            directory.mkdir(parents=True, exist_ok=True)
        run_id = _build_run_id()
        video_slug = _build_video_workspace_slug(video_path)
        query_slug = _sanitize_query(query)

    _cleanup_runtime_artifacts(query_workspace)
    effective_log_path = Path(log_path).resolve() if log_path else artifact_paths["run_log_path"]
    request_payload = {
        "kind": "query_request",
        "layout_version": 2,
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "video_path": str(Path(video_path).resolve()),
        "video_slug": video_slug,
        "query": query,
        "query_slug": query_slug,
        "workspace_root": str(workspace_root),
        "shared_preprocess_root": str(shared_preprocess_root),
        "query_group_dir": str(query_group_dir),
        "run_workspace": str(query_workspace),
        "persist_query_workspace": bool(persist_query_workspace),
        "write_log_file": bool(write_log_file),
        "log_path": str(effective_log_path),
    }
    manifest_payload = {
        "kind": "query_run",
        "layout_version": 2,
        "status": "running",
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "video_path": request_payload["video_path"],
        "video_slug": video_slug,
        "query": query,
        "query_slug": query_slug,
        "workspace_root": str(workspace_root),
        "query_group_dir": str(query_group_dir),
        "run_workspace": str(query_workspace),
        "shared_preprocess_root": str(shared_preprocess_root),
        "directories": {
            "inputs": str(artifact_paths["inputs_dir"]),
            "logs": str(artifact_paths["logs_dir"]),
            "results": str(artifact_paths["results_dir"]),
            "processed": str((shared_preprocess_root / "processed" / Path(video_path).stem).resolve()),
            "cache": str((shared_preprocess_root / "_cache" / Path(video_path).stem).resolve()),
        },
        "files": {
            "request": str(artifact_paths["request_path"]),
            "run_log": str(effective_log_path),
            "raw_trace_jsonl": str(artifact_paths["raw_trace_jsonl_path"]),
            "run_result": str(artifact_paths["run_result_path"]),
            "error": str(artifact_paths["error_path"]),
        },
    }
    query_index_payload = {
        "kind": "query_index",
        "layout_version": 2,
        "video_slug": video_slug,
        "query_slug": query_slug,
        "video_path": request_payload["video_path"],
        "query": query,
        "query_group_dir": str(query_group_dir),
        "runs_dir": str((query_group_dir / "runs").resolve()) if persist_query_workspace else str(query_workspace),
        "latest_run_id": run_id,
        "latest_run_workspace": str(query_workspace),
        "latest_manifest": str(artifact_paths["manifest_path"]),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "status": "running",
    }
    _write_json_file(artifact_paths["request_path"], request_payload)
    _write_json_file(artifact_paths["manifest_path"], manifest_payload)
    if persist_query_workspace:
        _write_json_file(query_group_dir / "query_index.json", query_index_payload)
        _write_json_file(query_group_dir / "latest_run.json", query_index_payload)

    # Query workspaces store logs/reports only; processed video caches live under
    # the shared workspace root so they can be reused across different queries.
    settings.data_cache_dir = str(shared_preprocess_root)
    runtime_session.set_active_video_context(video_name=Path(video_path).stem, video_path=video_path)

    try:
        if write_log_file:
            file_handler = _attach_file_logger(effective_log_path, rewrite=True)
        with _temporary_tool_registry_overrides(
            disabled_tool_names=runtime_disabled_tool_names,
            disable_meta_tools=bool(runtime_disable_meta_tools),
        ):
            logger.info("=" * 80)
            logger.info("Tool Star RL - End-to-End Pipeline")
            logger.info("=" * 80)
            logger.info("Video: %s", video_path)
            logger.info("Query: %s", query)
            logger.info("Query group: %s", query_group_dir)
            logger.info("Run workspace: %s", query_workspace)
            logger.info("Run manifest: %s", artifact_paths["manifest_path"])
            logger.info("Run log: %s", effective_log_path)
            logger.info("Shared data cache: %s", shared_preprocess_root)
            logger.info("Cache reuse: %s", "disabled" if no_cache else "enabled")
            logger.info("Pipeline mode: %s", _normalize_pipeline_mode(pipeline_mode))
            if runtime_disabled_tool_names or runtime_disable_meta_tools:
                logger.info(
                    "Runtime tool registry overrides: disable_meta_tools=%s disabled_tool_names=%s",
                    bool(runtime_disable_meta_tools),
                    list(runtime_disabled_tool_names or []),
                )
            logger.info("")

            total_started = time.perf_counter()
            preprocessing_payload = await preprocess_video(
                video_path,
                shared_preprocess_root,
                device_type=device_type,
                reuse_cache=not no_cache,
            )
            runtime_session.release_providers_for_roles("preprocess")

            grounding_started = time.perf_counter()
            grounding_context = _build_planner_grounding_context(
                query=query,
                preprocessing_result=preprocessing_payload["result"],
                processed_dir=preprocessing_payload["output_dir"],
                top_k=max(1, int(settings.planner_grounding_top_k)),
                include_video_clip_qa=bool(include_grounding_video_clip_qa),
            )
            runtime_session.record_timing("query.grounding_context_seconds", time.perf_counter() - grounding_started)

            resolved_pipeline_mode = _normalize_pipeline_mode(pipeline_mode)
            if resolved_pipeline_mode == "planner_direct_answer":
                final_answer, trajectory, success = run_query_direct(
                    query,
                    grounding_context=grounding_context,
                    planner_prompt_path=planner_direct_answer_prompt_path,
                    device_type=device_type,
                    raw_trace_jsonl_path=artifact_paths["raw_trace_jsonl_path"],
                )
            elif resolved_pipeline_mode == "planner_toolsearch_direct":
                final_answer, trajectory, success = run_query(
                    query,
                    grounding_context=grounding_context,
                    device_type=device_type,
                    raw_trace_jsonl_path=artifact_paths["raw_trace_jsonl_path"],
                    runtime_profile=runtime_profile,
                    planner_toolsearch_direct_execution=True,
                    planner_toolsearch_direct_single_tool_only=bool(
                        planner_toolsearch_direct_single_tool_only
                    ),
                    planner_single_action_per_step=bool(planner_single_action_per_step),
                )
            else:
                final_answer, trajectory, success = run_query(
                    query,
                    grounding_context=grounding_context,
                    device_type=device_type,
                    raw_trace_jsonl_path=artifact_paths["raw_trace_jsonl_path"],
                    runtime_profile=runtime_profile,
                    planner_single_action_per_step=bool(planner_single_action_per_step),
                )

            runtime_session.record_timing("query.total_seconds", time.perf_counter() - total_started)
            run_payload = _build_run_result_payload(
                workspace=query_workspace,
                video_path=video_path,
                query=query,
                preprocessing_result=preprocessing_payload["result"],
                processed_dir=preprocessing_payload["output_dir"],
                trajectory=trajectory,
                final_answer=final_answer,
                success=success,
            )
            _write_json_file(artifact_paths["run_result_path"], run_payload)
            logger.info("%s %s", RUN_RESULT_LOG_MARKER, json.dumps(run_payload, ensure_ascii=False))

            manifest_payload.update(
                {
                    "status": "completed",
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                    "success": bool(success),
                    "processed_output_dir": str(Path(preprocessing_payload["output_dir"]).resolve()),
                }
            )
            query_index_payload.update(
                {
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                    "status": "completed",
                    "success": bool(success),
                    "processed_output_dir": str(Path(preprocessing_payload["output_dir"]).resolve()),
                }
            )
            _write_json_file(artifact_paths["manifest_path"], manifest_payload)
            if persist_query_workspace:
                _write_json_file(query_group_dir / "query_index.json", query_index_payload)
                _write_json_file(query_group_dir / "latest_run.json", query_index_payload)

            logger.info("\n" + "=" * 80)
            if success:
                logger.info("Pipeline completed successfully!")
            else:
                logger.warning("Pipeline completed with failure (see FINAL RESULT and trajectory).")
            logger.info("Artifacts stored in: %s", query_workspace)
            logger.info("=" * 80)

            return {
                "final_answer": final_answer,
                "trajectory": trajectory,
                "success": success,
                "query_workspace": query_workspace,
                "preprocessing_payload": preprocessing_payload,
                "run_result": run_payload,
            }
    except Exception as exc:
        error_payload = {
            "kind": "query_run_error",
            "layout_version": 2,
            "run_id": run_id,
            "video_path": request_payload["video_path"],
            "query": query,
            "run_workspace": str(query_workspace),
            "error": str(exc),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        _write_json_file(artifact_paths["error_path"], error_payload)
        manifest_payload.update(
            {
                "status": "failed",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "error": str(exc),
            }
        )
        query_index_payload.update(
            {
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "status": "failed",
                "error": str(exc),
            }
        )
        _write_json_file(artifact_paths["manifest_path"], manifest_payload)
        if persist_query_workspace:
            _write_json_file(query_group_dir / "query_index.json", query_index_payload)
            _write_json_file(query_group_dir / "latest_run.json", query_index_payload)
        raise
    finally:
        _detach_file_logger(file_handler)


async def execute_pipeline(
    *,
    video_path: str,
    query: str,
    workspace_root: Path,
    shared_preprocess_root: Optional[Path] = None,
    no_cache: bool = False,
    device_type: Optional[str] = None,
    write_log_file: bool = True,
    log_path: Optional[Path] = None,
    persist_query_workspace: bool = True,
    runtime_profile: str = _PIPELINE_RUNTIME_PROFILE_DEFAULT,
    pipeline_mode: str = "react",
    planner_direct_answer_prompt_path: Optional[str] = None,
    include_grounding_video_clip_qa: bool = True,
    planner_toolsearch_direct_single_tool_only: bool = False,
    planner_single_action_per_step: bool = False,
    runtime_disabled_tool_names: Optional[List[str]] = None,
    runtime_disable_meta_tools: bool = False,
) -> Dict[str, Any]:
    with _pipeline_runtime_profile_context(runtime_profile) as resolved_runtime_profile:
        return await _execute_pipeline_impl(
            video_path=video_path,
            query=query,
            workspace_root=workspace_root,
            shared_preprocess_root=shared_preprocess_root,
            no_cache=no_cache,
            device_type=device_type,
            write_log_file=write_log_file,
            log_path=log_path,
            persist_query_workspace=persist_query_workspace,
            runtime_profile=resolved_runtime_profile,
            pipeline_mode=pipeline_mode,
            planner_direct_answer_prompt_path=planner_direct_answer_prompt_path,
            include_grounding_video_clip_qa=include_grounding_video_clip_qa,
            planner_toolsearch_direct_single_tool_only=planner_toolsearch_direct_single_tool_only,
            planner_single_action_per_step=planner_single_action_per_step,
            runtime_disabled_tool_names=runtime_disabled_tool_names,
            runtime_disable_meta_tools=runtime_disable_meta_tools,
        )


async def main():
    """Main entry point."""
    args = parse_args()

    # Initialize device (CUDA/NPU/CPU auto-detection)
    device_type, device_count = detect_device()
    logger.info(f"Device detected: {device_type}")
    if device_count:
        logger.info(f"Device count: {device_count}")
    visible_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_cuda_devices:
        logger.info("CUDA_VISIBLE_DEVICES=%s", visible_cuda_devices)
    initialize_device(device_type)
    logger.info("Startup accelerator snapshot: %s", get_accelerator_snapshot(device_type))

    payload = await execute_pipeline(
        video_path=args.video_path,
        query=args.query,
        workspace_root=Path(args.workspace_root) if args.workspace_root else Path(settings.data_cache_dir),
        shared_preprocess_root=Path(args.shared_preprocess_root) if args.shared_preprocess_root else None,
        no_cache=bool(args.no_cache),
        device_type=device_type,
        write_log_file=not bool(args.no_file_log),
        log_path=Path(args.log_path) if args.log_path else None,
        persist_query_workspace=not bool(args.flat_workspace),
        runtime_profile=str(args.runtime_profile or _PIPELINE_RUNTIME_PROFILE_DEFAULT),
    )
    return payload["final_answer"]


if __name__ == "__main__":
    asyncio.run(main())
