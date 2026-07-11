# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import asyncio
from dataclasses import fields as dataclass_fields
import inspect
import json
import logging
import os
from pprint import pprint
from typing import Any, Callable, Optional
import warnings

import ray
import vllm.entrypoints.cli.serve
from packaging import version
from ray.actor import ActorHandle
from omegaconf import DictConfig, OmegaConf
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import run_headless
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword, is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler import DistProfiler, build_vllm_profiler_args
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings, qwen2_5_vl_dedup_image_tokens, run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    SuppressSignalInThread,
    build_cli_args_from_config,
    extract_prompt_logprobs,
    get_vllm_max_lora_rank,
)

_VLLM_VERSION = version.parse(vllm.__version__)

if _VLLM_VERSION > version.parse("0.11.0"):
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if _VLLM_VERSION == version.parse("0.12.0"):
        from vllm.entrypoints.harmony_utils import get_encoding

    elif _VLLM_VERSION >= version.parse("0.13.0"):
        from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

    else:
        get_encoding = None

    if get_encoding is not None and os.getenv("VERL_USE_GPT_OSS", "0") == "1":
        get_encoding()
else:
    from vllm.utils import FlexibleArgumentParser


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

_ROPE_WARNING_LOG_FILTER_INSTALLED = False


def _coerce_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except Exception as exc:
        raise TypeError(f"{field_name} must be int-compatible, got {type(value).__name__}: {value!r}") from exc


def _coerce_optional_float(value: Any, field_name: str) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception as exc:
        raise TypeError(f"{field_name} must be float-compatible, got {type(value).__name__}: {value!r}") from exc


def _read_env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_distributed_executor_backend(value: Any) -> Optional[str]:
    if value is None:
        return None

    backend = str(value).strip().lower()
    if not backend:
        return None

    alias_map = {
        "uniproc": "uni",
        "single_process": "uni",
        "single-process": "uni",
        "multiproc": "mp",
        "multi_process": "mp",
        "multi-process": "mp",
    }
    backend = alias_map.get(backend, backend)

    allowed_backends = {"external_launcher", "mp", "ray", "uni"}
    if backend not in allowed_backends:
        logger.warning(
            "Unknown distributed_executor_backend=%r. Ignoring and falling back to auto selection.",
            value,
        )
        return None

    return backend


def _append_pythonwarnings_filter(filter_expr: str) -> None:
    current_filters = os.getenv("PYTHONWARNINGS", "")
    filter_parts = [part.strip() for part in current_filters.split(",") if part.strip()]
    if filter_expr not in filter_parts:
        filter_parts.append(filter_expr)
        os.environ["PYTHONWARNINGS"] = ",".join(filter_parts)


def _configure_known_warning_suppression() -> None:
    """Suppress known noisy third-party warnings in rollout server processes."""
    global _ROPE_WARNING_LOG_FILTER_INSTALLED

    warnings.filterwarnings(
        "ignore",
        message=r"The cuda\.cudart module is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The cuda\.nvrtc module is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"NPU not support router replay for now\.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Unrecognized keys in `rope_.*` for .*rope_type.*default",
        category=UserWarning,
    )

    # Propagate suppression into spawned vLLM worker processes.
    _append_pythonwarnings_filter(
        "ignore:The cuda.cudart module is deprecated:FutureWarning"
    )
    _append_pythonwarnings_filter(
        "ignore:The cuda.nvrtc module is deprecated:FutureWarning"
    )
    _append_pythonwarnings_filter("ignore:NPU not support router replay for now.:UserWarning")
    _append_pythonwarnings_filter(r"ignore:Unrecognized keys in `rope_.*` for .*rope_type.*default:UserWarning")

    if not _ROPE_WARNING_LOG_FILTER_INSTALLED:
        rope_warning_substrings = (
            "Unrecognized keys in `rope_parameters` for `rope_type`='default'",
            "Unrecognized keys in `rope_parameters` for 'rope_type'='default'",
            "Unrecognized keys in `rope_scaling` for `rope_type`='default'",
            "Unrecognized keys in `rope_scaling` for 'rope_type'='default'",
        )

        class _RopeWarningFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                message = record.getMessage()
                return not any(substring in message for substring in rope_warning_substrings)

        rope_filter = _RopeWarningFilter()
        logging.getLogger("transformers").addFilter(rope_filter)
        logging.getLogger("transformers.configuration_utils").addFilter(rope_filter)
        logging.getLogger("transformers.modeling_rope_utils").addFilter(rope_filter)
        _ROPE_WARNING_LOG_FILTER_INSTALLED = True


def _configure_agent_loop_vllm_stability_guards() -> None:
    """Apply conservative vLLM defaults for known unstable kernel paths in AgentLoop rollout."""
    disable_fla_packed_recurrent_decode = _read_env_flag(
        "TOOL_STAR_AGENT_LOOP_DISABLE_FLA_PACKED_RECURRENT_DECODE", True
    )
    if disable_fla_packed_recurrent_decode:
        current_fla_decode_flag = os.getenv("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "")
        if current_fla_decode_flag.strip().lower() not in {"0", "false", "no", "off"}:
            os.environ["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] = "0"
            logger.warning(
                "Stability guard active: set VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 "
                "(override via TOOL_STAR_AGENT_LOOP_DISABLE_FLA_PACKED_RECURRENT_DECODE=0)."
            )

    force_cudagraph_none = _read_env_flag(
        "TOOL_STAR_AGENT_LOOP_FORCE_CUDAGRAPH_NONE", disable_fla_packed_recurrent_decode
    )
    if force_cudagraph_none and "TOOL_STAR_AGENT_LOOP_CUDAGRAPH_MODE" not in os.environ:
        os.environ["TOOL_STAR_AGENT_LOOP_CUDAGRAPH_MODE"] = "NONE"
        logger.warning(
            "Stability guard active: default TOOL_STAR_AGENT_LOOP_CUDAGRAPH_MODE=NONE "
            "for this rollout process. Set TOOL_STAR_AGENT_LOOP_CUDAGRAPH_MODE explicitly to override."
        )

    disable_runtime_lora_updating = _read_env_flag(
        "TOOL_STAR_AGENT_LOOP_DISABLE_RUNTIME_LORA_UPDATING", True
    )
    if disable_runtime_lora_updating:
        current_runtime_lora_flag = os.getenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "")
        if current_runtime_lora_flag.strip().lower() not in {"0", "false", "no", "off"}:
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "0"
            logger.warning(
                "Stability guard active: set VLLM_ALLOW_RUNTIME_LORA_UPDATING=0 "
                "(override via TOOL_STAR_AGENT_LOOP_DISABLE_RUNTIME_LORA_UPDATING=0)."
            )


def _sanitize_default_rope_parameters(rope_parameters: Any) -> tuple[Any, bool]:
    if not isinstance(rope_parameters, dict):
        return rope_parameters, False

    rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
    if rope_type != "default":
        return rope_parameters, False

    sanitized = dict(rope_parameters)
    changed = False
    for key in ("mrope_interleaved", "mrope_section"):
        if key in sanitized:
            sanitized.pop(key, None)
            changed = True

    return sanitized, changed


def _inject_rope_hf_overrides(hf_config: Any, hf_overrides: dict[str, Any]) -> dict[str, Any]:
    """Drop unsupported mrope keys for default rope config to avoid repeated harmless warnings."""
    if not isinstance(hf_overrides, dict):
        return hf_overrides

    try:
        config_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else None
    except Exception:
        return hf_overrides

    if not isinstance(config_dict, dict):
        return hf_overrides

    updated_overrides = dict(hf_overrides)
    updated = False

    if "rope_parameters" not in updated_overrides:
        top_rope, changed = _sanitize_default_rope_parameters(config_dict.get("rope_parameters"))
        if changed:
            updated_overrides["rope_parameters"] = top_rope
            updated = True

    text_config = config_dict.get("text_config")
    if isinstance(text_config, dict):
        text_override = updated_overrides.get("text_config", {})
        if not isinstance(text_override, dict):
            text_override = {}
        if "rope_parameters" not in text_override:
            text_rope, changed = _sanitize_default_rope_parameters(text_config.get("rope_parameters"))
            if changed:
                text_override = dict(text_override)
                text_override["rope_parameters"] = text_rope
                updated_overrides["text_config"] = text_override
                updated = True

    if updated:
        logger.info("Applied hf_overrides sanitization for default rope_parameters (drop mrope_* noise keys).")

    return updated_overrides


def _coerce_optional_bool(value: Any, field_name: str) -> Optional[bool]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be bool-compatible, got {type(value).__name__}: {value!r}")


def _auto_tune_gpu_memory_utilization(requested_utilization: Optional[float]) -> Optional[float]:
    """Clamp utilization to currently free GPU memory ratio to avoid startup OOM.

    This keeps high utilization by default while preventing immediate failures when
    other processes temporarily occupy part of VRAM.
    """
    if requested_utilization is None:
        return None

    if not (0 < requested_utilization <= 1):
        return requested_utilization

    fallback_cap = _coerce_optional_float(
        os.getenv("TOOL_STAR_AGENT_LOOP_GPU_MEMORY_FALLBACK_CAP", "0.68"),
        "TOOL_STAR_AGENT_LOOP_GPU_MEMORY_FALLBACK_CAP",
    )
    if fallback_cap is None:
        fallback_cap = requested_utilization
    fallback_cap = min(max(fallback_cap, 0.1), 0.98)

    try:
        import torch

        if not torch.cuda.is_available():
            return min(requested_utilization, fallback_cap)

        reserve_ratio = float(os.getenv("TOOL_STAR_AGENT_LOOP_GPU_MEMORY_RESERVE_RATIO", "0.05"))
        reserve_ratio = min(max(reserve_ratio, 0.0), 0.2)

        visible_device_count = torch.cuda.device_count()
        if visible_device_count <= 0:
            return min(requested_utilization, fallback_cap)

        min_free_ratio = 1.0
        for device_idx in range(visible_device_count):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            if total_bytes <= 0:
                continue
            min_free_ratio = min(min_free_ratio, free_bytes / total_bytes)

        safe_upper_bound = min(0.98, min_free_ratio - reserve_ratio, fallback_cap)
        safe_upper_bound = max(0.1, safe_upper_bound)
        tuned_utilization = min(requested_utilization, safe_upper_bound)

        if tuned_utilization + 1e-8 < requested_utilization:
            logger.warning(
                "Auto-tune gpu_memory_utilization from %.3f to %.3f based on free memory ratio %.3f (reserve %.3f)",
                requested_utilization,
                tuned_utilization,
                min_free_ratio,
                reserve_ratio,
            )

        return tuned_utilization
    except Exception as exc:
        logger.warning(
            "Auto-tune gpu_memory_utilization skipped due to runtime probe failure: %s. "
            "Fallback to min(requested=%.3f, cap=%.3f)",
            exc,
            requested_utilization,
            fallback_cap,
        )
        return min(requested_utilization, fallback_cap)


def _sanitize_video_agent_rollout_config_for_agent_loop(config: RolloutConfig | DictConfig | dict[str, Any]) -> DictConfig:
    """Project video-agent rollout config to core RolloutConfig fields for AgentLoop replica path."""
    config_dict = OmegaConf.create(config)
    raw_container = OmegaConf.to_container(config_dict, resolve=False)
    if not isinstance(raw_container, dict):
        raise TypeError(f"rollout config must be dict-like, got {type(raw_container).__name__}")

    allowed_fields = {field.name for field in dataclass_fields(RolloutConfig)}
    sanitized: dict[str, Any] = {key: value for key, value in raw_container.items() if key in allowed_fields}

    # AgentLoop rollout replicas currently support built-in async server implementations.
    sanitized["name"] = "vllm"

    # Keep native AgentLoop path stable on limited-memory GPUs.
    if sanitized.get("max_model_len", None) in (None, "", "null"):
        sanitized["max_model_len"] = int(os.getenv("TOOL_STAR_AGENT_LOOP_MAX_MODEL_LEN", "12288"))

    if sanitized.get("gpu_memory_utilization", None) in (None, "", "null"):
        sanitized["gpu_memory_utilization"] = float(
            os.getenv("TOOL_STAR_AGENT_LOOP_GPU_MEMORY_UTILIZATION", "0.90")
        )

    sanitized["max_model_len"] = _coerce_optional_int(sanitized.get("max_model_len"), "rollout.max_model_len")
    sanitized["gpu_memory_utilization"] = _coerce_optional_float(
        sanitized.get("gpu_memory_utilization"), "rollout.gpu_memory_utilization"
    )

    return OmegaConf.create(sanitized)


class vLLMHttpServer:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
            cuda_visible_devices (str): cuda visible devices.
        """
        _configure_known_warning_suppression()
        _configure_agent_loop_vllm_stability_guards()

        os.environ[get_visible_devices_keyword()] = cuda_visible_devices

        self.config = self._init_config(config)
        self.model_config = self._init_model_config(model_config)
        self._validate_configs()
        self._generation_concurrency_limit = self._resolve_generation_concurrency_limit()
        self._generation_concurrency_semaphore: Optional[asyncio.Semaphore] = None

        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling vllm server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            # used for torch.distributed.init_process_group
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)
            # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address, with_alive_sock=True)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address, with_alive_sock=True)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None
            self._dp_master_port = None

        self._post_init(cuda_visible_devices)

    def get_master_address(self):
        """Get master address and port for data parallel.
        Returns:
            tuple: (master_address, master_port, dp_rpc_port)
        """
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get(self._get_engine_kwargs_key(), {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        self._preprocess_engine_kwargs(engine_kwargs)
        self._apply_multimodal_block_table_stability_guard(engine_kwargs)
        self._apply_multimodal_chunked_prefill_guard(engine_kwargs)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = self._get_override_generation_config()
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization, hf_overrides = self._apply_quantization()

        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        compilation_config.setdefault(
            "cudagraph_mode", os.getenv("TOOL_STAR_AGENT_LOOP_CUDAGRAPH_MODE", "PIECEWISE")
        )

        requested_gpu_memory_utilization = _coerce_optional_float(
            self.config.gpu_memory_utilization, "rollout.gpu_memory_utilization"
        )
        effective_gpu_memory_utilization = _auto_tune_gpu_memory_utilization(requested_gpu_memory_utilization)
        effective_max_num_seqs = self._resolve_effective_max_num_seqs()
        effective_max_num_batched_tokens = self._resolve_effective_max_num_batched_tokens(
            engine_kwargs,
            effective_max_num_seqs=effective_max_num_seqs,
        )

        # FULL cuda graph is not yet supported with DCP, downgrade to PIECEWISE
        dcp_size = _coerce_optional_int(
            engine_kwargs.get("decode_context_parallel_size", 1),
            "engine_kwargs.decode_context_parallel_size",
        )
        if dcp_size is None:
            dcp_size = 1

        tp_size = _coerce_optional_int(self.config.tensor_model_parallel_size, "rollout.tensor_model_parallel_size")
        if tp_size is None:
            tp_size = 1

        dp_size = _coerce_optional_int(self.config.data_parallel_size, "rollout.data_parallel_size")
        if dp_size is None:
            dp_size = 1

        if dcp_size > 1 and compilation_config["cudagraph_mode"] == "FULL_AND_PIECEWISE":
            logger.warning(
                "FULL cuda graph is not supported with DCP (decode_context_parallel_size=%d), "
                "downgrading cudagraph_mode to PIECEWISE.",
                dcp_size,
            )
            compilation_config["cudagraph_mode"] = "PIECEWISE"

        requested_backend = engine_kwargs.pop("distributed_executor_backend", None)
        env_backend = os.getenv("TOOL_STAR_AGENT_LOOP_DISTRIBUTED_EXECUTOR_BACKEND", "").strip()
        if env_backend:
            requested_backend = env_backend

        requested_backend = _normalize_distributed_executor_backend(requested_backend)

        single_process_topology = self.nnodes == 1 and tp_size <= 1 and dp_size <= 1 and dcp_size <= 1
        enforce_uni_single_process = _read_env_flag("TOOL_STAR_AGENT_LOOP_ENFORCE_UNI_SINGLE_PROCESS", True)

        if single_process_topology:
            if enforce_uni_single_process and requested_backend not in (None, "uni"):
                logger.warning(
                    "Stability guard active: forcing distributed_executor_backend=uni for single-process "
                    "topology (nnodes=%d,tp=%d,dp=%d,dcp=%d). requested_backend=%s. "
                    "Set TOOL_STAR_AGENT_LOOP_ENFORCE_UNI_SINGLE_PROCESS=0 to opt out.",
                    self.nnodes,
                    tp_size,
                    dp_size,
                    dcp_size,
                    requested_backend,
                )
                distributed_executor_backend = "uni"
            else:
                distributed_executor_backend = requested_backend or "uni"

            disable_v1_multiprocessing = _read_env_flag(
                "TOOL_STAR_AGENT_LOOP_DISABLE_V1_MULTIPROCESSING_SINGLE_PROCESS", True
            )
            if disable_v1_multiprocessing:
                current_v1_multiprocessing = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "")
                if current_v1_multiprocessing.strip().lower() not in {"0", "false", "no", "off"}:
                    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                    logger.warning(
                        "Stability guard active: set VLLM_ENABLE_V1_MULTIPROCESSING=0 for single-process "
                        "topology (override via TOOL_STAR_AGENT_LOOP_DISABLE_V1_MULTIPROCESSING_SINGLE_PROCESS=0)."
                    )
        else:
            if requested_backend == "uni":
                logger.warning(
                    "distributed_executor_backend=uni is incompatible with distributed topology "
                    "(nnodes=%d,tp=%d,dp=%d,dcp=%d); fallback to mp.",
                    self.nnodes,
                    tp_size,
                    dp_size,
                    dcp_size,
                )
                distributed_executor_backend = "mp"
            else:
                distributed_executor_backend = requested_backend or "mp"

        logger.info(
            "Resolved distributed_executor_backend=%s (requested=%s, nnodes=%d, tp=%d, dp=%d, dcp=%d)",
            distributed_executor_backend,
            requested_backend,
            self.nnodes,
            tp_size,
            dp_size,
            dcp_size,
        )

        compilation_config = json.dumps(compilation_config)
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": distributed_executor_backend,
            "worker_extension_cls": self._get_worker_extension_cls(),
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": effective_max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": effective_max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": effective_gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": tp_size,
            "seed": self.replica_rank + self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
            **engine_kwargs,
        }

        # Keep core rollout knobs authoritative even if engine_kwargs contains the same keys.
        args["max_model_len"] = self.config.max_model_len
        args["gpu_memory_utilization"] = effective_gpu_memory_utilization
        args["max_num_seqs"] = effective_max_num_seqs
        args["max_num_batched_tokens"] = effective_max_num_batched_tokens

        # Keep server-wide defaults deterministic and avoid model generation_config side effects.
        generation_config = engine_kwargs.get("generation_config", None)
        if generation_config is None:
            generation_config = os.getenv("TOOL_STAR_AGENT_LOOP_GENERATION_CONFIG", "vllm").strip()
            if generation_config:
                args["generation_config"] = generation_config

        # update profiler args
        profiler_args = build_vllm_profiler_args(
            self.profiler_controller.config, self.profiler_controller.tool_config, self.replica_rank
        )
        if _VLLM_VERSION >= version.parse("0.13.0"):
            # vLLM >= 0.13.0 supports profiler config via CLI args; env vars still work but will be deprecated
            args.update(profiler_args)

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        # mtp (None for diffusion models; only LLM models use speculative decoding)
        if self.config.mtp is not None and self.config.mtp.enable and self.config.mtp.enable_rollout:
            speculative_config = {
                "method": self.config.mtp.method,
                "num_speculative_tokens": self.config.mtp.num_speculative_tokens,
            }
            args["speculative_config"] = speculative_config

        if dp_size > 1:
            assert self.gpus_per_node % tp_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // tp_size
            assert len(self.workers) == data_parallel_size_local * tp_size, (
                f"num workers ({len(self.workers)}) should be equal to "
                f"dp_size_local ({data_parallel_size_local}) * tp_size ({tp_size})"
            )
            dp_args = {
                "data_parallel_size": dp_size,
                "data_parallel_size_local": data_parallel_size_local,
                "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                "data_parallel_address": self._master_address,
                "data_parallel_rpc_port": self._dp_rpc_port,
            }
            args.update(dp_args)

        args.update({"enable_expert_parallel": self.config.expert_parallel_size > 1})

        # used for torch.distributed.init_process_group
        if self.nnodes > 1:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # update lora-related args
        lora_rank = self.model_config.lora.get("rank", 0)
        if lora_rank <= 0:
            lora_rank = (
                self.model_config.lora_rank
            )  # FIXME: fallback to lora_rank for now, we should unify lora settings.

        if self.model_config.lora.get("merge", False):
            lora_rank = 0

        if lora_rank > 0:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
            }
            if self.model_config.lora.get("fully_sharded_loras", False):
                lora_args["fully_sharded_loras"] = True
            args.update(lora_args)

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        explicit_false_args = {key for key, value in args.items() if isinstance(value, bool) and value is False}
        server_args = ["serve", self.model_config.local_path] + build_cli_args_from_config(args)

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = self._get_cli_modules()
        parser = FlexibleArgumentParser(description=self._get_cli_description())
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        for key in explicit_false_args:
            if hasattr(server_args, key):
                setattr(server_args, key, False)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model", kwargs={"vocab_size": len(self.model_config.tokenizer)}
        )

        build_app_sig = inspect.signature(build_app)
        supported_tasks: tuple[Any, ...] = ()
        if "supported_tasks" in build_app_sig.parameters:
            supported_tasks = await engine_client.get_supported_tasks()
            app = build_app(args, supported_tasks)
        else:
            app = build_app(args)

        init_app_sig = inspect.signature(init_app_state)
        if "vllm_config" in init_app_sig.parameters:
            await init_app_state(engine_client, vllm_config, app.state, args)
        elif "supported_tasks" in init_app_sig.parameters:
            await init_app_state(engine_client, app.state, args, supported_tasks)
        else:
            await init_app_state(engine_client, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        args.api_server_count = 0

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    logger.exception(f"run_headless failed with exception: {exc}")
                else:
                    logger.warning("run_headless completed successfully, but it's not expected.")
            except Exception as e:
                logger.exception(f"get result from run_headless failed: {e}")
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)

    def _resolve_generation_concurrency_limit(self) -> Optional[int]:
        if not self._is_video_agent_rollout():
            return None
        if not _read_env_flag("TOOL_STAR_AGENT_LOOP_LIMIT_V1_INFLIGHT_ON_BLOCK_TABLE_RISK", True):
            return None

        limit = _coerce_optional_int(
            os.getenv("TOOL_STAR_AGENT_LOOP_V1_INFLIGHT_LIMIT", "1"),
            "TOOL_STAR_AGENT_LOOP_V1_INFLIGHT_LIMIT",
        )
        if limit is None or limit <= 0:
            return None

        logger.warning(
            "Stability guard active: limiting each video-agent vLLM server to %d in-flight generate "
            "request(s) to avoid V1 block_table slot-mapping instability. "
            "Set TOOL_STAR_AGENT_LOOP_LIMIT_V1_INFLIGHT_ON_BLOCK_TABLE_RISK=0 to opt out, "
            "or TOOL_STAR_AGENT_LOOP_V1_INFLIGHT_LIMIT to tune the limit.",
            limit,
        )
        return int(limit)

    def _get_generation_concurrency_semaphore(self) -> Optional[asyncio.Semaphore]:
        if self._generation_concurrency_limit is None:
            return None
        if self._generation_concurrency_semaphore is None:
            self._generation_concurrency_semaphore = asyncio.Semaphore(self._generation_concurrency_limit)
        return self._generation_concurrency_semaphore

    async def _consume_engine_generate(
        self,
        *,
        prompt: TokensPrompt,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest],
        priority: int,
    ) -> RequestOutput:
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None
        return final_res

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        prompt_ids = normalize_token_ids(prompt_ids)
        sampling_params = dict(sampling_params)

        context_safety_margin = self._get_context_safety_margin()
        max_model_len = _coerce_optional_int(self.config.max_model_len, "rollout.max_model_len")
        if max_model_len is None:
            max_model_len = int(get_max_position_embeddings(self.model_config.hf_config))

        # Calculate the maximum possible new tokens based on available context space
        # This serves as a safety upper bound
        max_possible_tokens = max_model_len - len(prompt_ids) - context_safety_margin
        if max_possible_tokens <= 0:
            logger.warning(
                "[RolloutGenerate][TRUNCATION] request_id=%s prompt/grounding exceeds context budget: "
                "prompt_tokens=%d max_model_len=%d safety_margin=%d. Refusing to truncate prompt.",
                request_id,
                len(prompt_ids),
                int(max_model_len),
                context_safety_margin,
            )
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({max_model_len}) with safety margin {context_safety_margin}."
            )

        # Determine max_tokens from sampling_params or use configured response_length as default
        if "max_tokens" in sampling_params:
            max_tokens = _coerce_optional_int(sampling_params.pop("max_tokens"), "sampling_params.max_tokens")
        elif "max_new_tokens" in sampling_params:
            # support sglang-style 'max_new_tokens' param
            max_tokens = _coerce_optional_int(sampling_params.pop("max_new_tokens"), "sampling_params.max_new_tokens")
        else:
            # Default to a calculation that considers configured lengths
            # Cap max_tokens by response_length to ensure tensor alignment,
            # and by remaining budget to prevent OOM in multi-turn rollouts.
            response_length = _coerce_optional_int(self.config.response_length, "rollout.response_length") or 1
            prompt_length = _coerce_optional_int(self.config.prompt_length, "rollout.prompt_length") or 0
            max_tokens = min(
                response_length, prompt_length + response_length - len(prompt_ids)
            )
        if max_tokens is None:
            max_tokens = self._get_safe_generation_default_max_new_tokens(log_clamp=False)
        if max_tokens <= 0:
            response_length = _coerce_optional_int(self.config.response_length, "rollout.response_length")
            prompt_length = _coerce_optional_int(self.config.prompt_length, "rollout.prompt_length")
            logger.warning(
                "[RolloutGenerate][TRUNCATION] request_id=%s no response budget remains before generation: "
                "requested_max_tokens=%s prompt_tokens=%d prompt_length=%s response_length=%s "
                "max_model_len=%d safety_margin=%d. Refusing to emit a 1-token overflow.",
                request_id,
                max_tokens,
                len(prompt_ids),
                prompt_length,
                response_length,
                int(max_model_len),
                context_safety_margin,
            )
            raise ValueError(
                f"No response budget remains for prompt length {len(prompt_ids)} "
                f"(prompt_length={prompt_length}, response_length={response_length})."
            )

        requested_max_tokens = max_tokens

        max_tokens_cap = self._get_max_tokens_cap()
        if max_tokens_cap is not None and max_tokens > max_tokens_cap:
            logger.warning(
                "[RolloutGenerate][TRUNCATION] request_id=%s max_tokens capped by TOOL_STAR_AGENT_LOOP_MAX_TOKENS_CAP: requested=%s cap=%d",
                request_id,
                requested_max_tokens,
                max_tokens_cap,
            )
            max_tokens = min(max_tokens, max_tokens_cap)

        # Clamp max_tokens to the valid request range before constructing SamplingParams.
        if max_tokens > max_possible_tokens:
            logger.warning(
                "[RolloutGenerate][TRUNCATION] request_id=%s max_tokens clamped by context budget: requested=%s available=%d prompt_tokens=%d max_model_len=%d safety_margin=%d",
                request_id,
                max_tokens,
                max_possible_tokens,
                len(prompt_ids),
                int(max_model_len),
                context_safety_margin,
            )
        max_tokens = max(1, min(max_tokens, max_possible_tokens))

        assert max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        # Add lora request
        lora_request = None
        if self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        semaphore = self._get_generation_concurrency_semaphore()
        if semaphore is None:
            final_res = await self._consume_engine_generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
            )
        else:
            async with semaphore:
                final_res = await self._consume_engine_generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    lora_request=lora_request,
                    priority=priority,
                )

        extra_fields = {"global_steps": self.global_steps}
        extract_prompt_logprobs(
            output=final_res,
            num_prompt_logprobs=sampling_params.prompt_logprobs,
            result_dict=extra_fields,
        )
        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        # Determine stop reason from finish_reason
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "length" or len(token_ids) >= max_tokens:
            logger.warning(
                "[RolloutGenerate][TRUNCATION] request_id=%s output reached generation budget: "
                "finish_reason=%s generated_tokens=%d max_tokens=%d prompt_tokens=%d "
                "max_model_len=%d safety_margin=%d",
                request_id,
                finish_reason,
                len(token_ids),
                max_tokens,
                len(prompt_ids),
                int(max_model_len),
                context_safety_margin,
            )
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None

        if hasattr(final_res.outputs[0], "num_preempted"):
            num_preempted = final_res.outputs[0].num_preempted

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wake_up(self):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            await self.engine.wake_up(tags=self._get_wake_up_tags())
            await self.engine.reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            await self._sleep_hybrid()
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.engine.start_profile(**kwargs)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.engine.stop_profile()

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()

    async def set_global_steps(self, global_steps: int):
        """Set the global steps of the model weights."""
        self.global_steps = global_steps

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all ongoing generation requests.

        On vLLM >= 0.12.0, uses AsyncLLM.pause_generation() to abort in-flight
        requests, drain, and clear caches. The engine remains paused after this
        call — use resume_generation() to accept new requests (e.g. before
        validation).

        On vLLM < 0.12.0, manually aborts each request and resets prefix cache.

        Returns:
            dict[str, Any]: Dictionary containing:
                - aborted_count: Number of requests aborted
                - request_ids: List of aborted request IDs
        """
        try:
            if _VLLM_VERSION >= version.parse("0.12.0"):
                # Snapshot request IDs before pausing for reporting
                request_ids = list(self.engine.output_processor.request_states.keys())

                # pause_generation with wait_for_inflight_requests=False will:
                # 1. Set engine to paused state (blocks new generate calls)
                # 2. Abort all in-flight requests
                # 3. Wait for requests to drain
                # 4. Clear prefix and mm caches if clear_cache=True
                await self.engine.pause_generation(
                    wait_for_inflight_requests=False,
                    clear_cache=reset_prefix_cache,
                )
            else:
                # Take an atomic snapshot to avoid race conditions with the vLLM engine thread
                request_states_snapshot = list(self.engine.output_processor.request_states.items())
                request_ids = [req_id for req_id, _ in request_states_snapshot]

                if not request_ids:
                    return {"aborted_count": 0, "request_ids": []}

                # For each request, create an abort output and put it to its queue
                # This allows the generator to receive the aborted result
                from vllm.v1.engine import FinishReason

                for _, req_state in request_states_snapshot:
                    request_output = req_state.make_request_output(
                        [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                    )
                    req_state.queue.put(request_output)

                # Abort requests in the output processor and engine core
                self.engine.output_processor.abort_requests(request_ids)
                await self.engine.engine_core.abort_requests_async(request_ids)

                # Try to reset prefix cache to ensure clean state
                if reset_prefix_cache:
                    await self.clear_kv_cache()
                    logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def resume_generation(self):
        """Resume generation after abort_all_requests (pause_generation).

        Only effective on vLLM >= 0.12.0 where pause_generation is used.
        No-op on older versions.
        """
        if self.node_rank != 0:
            return
        if _VLLM_VERSION >= version.parse("0.12.0"):
            await self.engine.resume_generation()

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _init_config(self, config):
        """Initialise config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(config)

    def _init_model_config(self, model_config):
        """Initialise model_config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)

    def _validate_configs(self) -> None:
        """Validate config/model_config after initialisation."""
        max_position_embeddings = int(get_max_position_embeddings(self.model_config.hf_config))
        max_model_len = _coerce_optional_int(self.config.max_model_len, "rollout.max_model_len")
        effective_max_model_len = max_position_embeddings if max_model_len is None else max_model_len
        if effective_max_model_len > max_position_embeddings:
            raise ValueError(
                f"max_model_len ({effective_max_model_len}) should be less than or equal to "
                f"max_position_embeddings ({max_position_embeddings})"
            )

        gpu_memory_utilization = _coerce_optional_float(
            self.config.gpu_memory_utilization, "rollout.gpu_memory_utilization"
        )
        if gpu_memory_utilization is not None and not (0 < gpu_memory_utilization <= 1):
            raise ValueError(
                f"gpu_memory_utilization ({gpu_memory_utilization}) should be in (0, 1]"
            )

    def _get_context_safety_margin(self) -> int:
        context_safety_margin = _coerce_optional_int(
            os.getenv("TOOL_STAR_AGENT_LOOP_CONTEXT_SAFETY_MARGIN", "32"),
            "TOOL_STAR_AGENT_LOOP_CONTEXT_SAFETY_MARGIN",
        )
        if context_safety_margin is None:
            return 0
        return max(0, context_safety_margin)

    def _get_max_tokens_cap(self) -> Optional[int]:
        max_tokens_cap = _coerce_optional_int(
            os.getenv("TOOL_STAR_AGENT_LOOP_MAX_TOKENS_CAP", "8192"),
            "TOOL_STAR_AGENT_LOOP_MAX_TOKENS_CAP",
        )
        if max_tokens_cap is not None and max_tokens_cap <= 0:
            return None
        return max_tokens_cap

    def _get_safe_generation_default_max_new_tokens(self, *, log_clamp: bool = False) -> int:
        response_length = _coerce_optional_int(self.config.response_length, "rollout.response_length")
        if response_length is None or response_length <= 0:
            response_length = 1

        max_new_tokens = response_length
        max_tokens_cap = self._get_max_tokens_cap()
        if max_tokens_cap is not None:
            max_new_tokens = min(max_new_tokens, max_tokens_cap)

        max_model_len = _coerce_optional_int(self.config.max_model_len, "rollout.max_model_len")
        prompt_length = _coerce_optional_int(self.config.prompt_length, "rollout.prompt_length")
        if max_model_len is not None and prompt_length is not None:
            default_context_budget = max_model_len - max(0, prompt_length) - self._get_context_safety_margin()
            if default_context_budget > 0:
                max_new_tokens = min(max_new_tokens, default_context_budget)
            else:
                max_new_tokens = 1

        max_new_tokens = max(1, int(max_new_tokens))
        if log_clamp and max_new_tokens < response_length:
            logger.warning(
                "[RolloutGenerate][TRUNCATION] server default max_new_tokens clamped: "
                "response_length=%d effective=%d max_model_len=%s prompt_length=%s cap=%s safety_margin=%d",
                response_length,
                max_new_tokens,
                max_model_len,
                prompt_length,
                max_tokens_cap,
                self._get_context_safety_margin(),
            )
        return max_new_tokens

    def _get_effective_max_tokens_per_mm_item(self, engine_kwargs: dict[str, Any]) -> Optional[int]:
        raw_value = engine_kwargs.get("max_tokens_per_mm_item", None)
        if raw_value is None:
            raw_value = os.getenv("TOOL_STAR_AGENT_LOOP_MAX_TOKENS_PER_MM_ITEM", None)
        if raw_value is None:
            if not self._is_chunked_mm_input_disabled(engine_kwargs):
                return None
            # vLLM V1 defaults this value to 16384 in the multimodal encoder budget path.
            raw_value = os.getenv("TOOL_STAR_AGENT_LOOP_DEFAULT_MAX_TOKENS_PER_MM_ITEM", "16384")
        if isinstance(raw_value, DictConfig):
            raw_value = OmegaConf.to_container(raw_value, resolve=True)
        if isinstance(raw_value, str) and raw_value.strip().startswith(("{", "[")):
            raw_value = json.loads(raw_value)
        if isinstance(raw_value, dict):
            values = [
                _coerce_optional_int(value, f"engine_kwargs.max_tokens_per_mm_item[{key!r}]")
                for key, value in raw_value.items()
            ]
            values = [value for value in values if value is not None]
            return max(values) if values else None
        if isinstance(raw_value, (list, tuple)):
            values = [
                _coerce_optional_int(value, f"engine_kwargs.max_tokens_per_mm_item[{index}]")
                for index, value in enumerate(raw_value)
            ]
            values = [value for value in values if value is not None]
            return max(values) if values else None
        return _coerce_optional_int(raw_value, "engine_kwargs.max_tokens_per_mm_item")

    def _is_chunked_mm_input_disabled(self, engine_kwargs: dict[str, Any]) -> bool:
        disabled = _coerce_optional_bool(
            engine_kwargs.get("disable_chunked_mm_input", False),
            "engine_kwargs.disable_chunked_mm_input",
        )
        return bool(disabled)

    def _is_video_agent_rollout(self) -> bool:
        if _read_env_flag("TOOL_STAR_AGENT_LOOP_IS_VIDEO_AGENT_ROLLOUT", False):
            return True
        rollout_name = str(self.config.get("name", "")).strip()
        return rollout_name == "vllm_video_agent" or bool(
            self.config.get("enable_native_agent_loop_for_video_agent", False)
        )

    def _resolve_effective_max_num_seqs(self) -> int:
        requested = _coerce_optional_int(self.config.max_num_seqs, "rollout.max_num_seqs")
        if requested is None or requested <= 0:
            requested = 1

        effective = int(requested)
        if self._is_video_agent_rollout() and _read_env_flag(
            "TOOL_STAR_AGENT_LOOP_CAP_V1_MAX_NUM_SEQS_ON_BLOCK_TABLE_RISK", True
        ):
            cap = _coerce_optional_int(
                os.getenv("TOOL_STAR_AGENT_LOOP_V1_MAX_NUM_SEQS_CAP", "1"),
                "TOOL_STAR_AGENT_LOOP_V1_MAX_NUM_SEQS_CAP",
            )
            if cap is not None and cap > 0:
                effective = min(effective, int(cap))

        if effective != requested:
            logger.warning(
                "Stability guard adjusted max_num_seqs from %d to %d for video-agent vLLM server "
                "to avoid V1 block_table slot-mapping instability when multiple long planner requests "
                "are scheduled together. Set TOOL_STAR_AGENT_LOOP_CAP_V1_MAX_NUM_SEQS_ON_BLOCK_TABLE_RISK=0 "
                "to opt out, or TOOL_STAR_AGENT_LOOP_V1_MAX_NUM_SEQS_CAP to tune the cap.",
                requested,
                effective,
            )

        return effective

    def _resolve_effective_max_num_batched_tokens(
        self,
        engine_kwargs: dict[str, Any],
        *,
        effective_max_num_seqs: Optional[int] = None,
    ) -> Optional[int]:
        requested = _coerce_optional_int(self.config.max_num_batched_tokens, "rollout.max_num_batched_tokens")
        if requested is None:
            return None

        effective = requested
        max_num_seqs = effective_max_num_seqs
        if max_num_seqs is None:
            max_num_seqs = _coerce_optional_int(self.config.max_num_seqs, "rollout.max_num_seqs")
        if max_num_seqs is not None and effective < max_num_seqs:
            effective = max_num_seqs

        max_model_len = _coerce_optional_int(self.config.max_model_len, "rollout.max_model_len")
        enable_chunked_prefill = self._get_effective_enable_chunked_prefill(engine_kwargs)
        if max_model_len is not None:
            if not enable_chunked_prefill and effective < max_model_len:
                effective = max_model_len
            elif enable_chunked_prefill and _read_env_flag(
                "TOOL_STAR_AGENT_LOOP_STABILIZE_CHUNKED_PREFILL_BUDGET", True
            ):
                floor_override = _coerce_optional_int(
                    os.getenv("TOOL_STAR_AGENT_LOOP_MAX_NUM_BATCHED_TOKENS_FLOOR"),
                    "TOOL_STAR_AGENT_LOOP_MAX_NUM_BATCHED_TOKENS_FLOOR",
                )
                if floor_override is None:
                    throughput_floor = _coerce_optional_int(
                        os.getenv("TOOL_STAR_AGENT_LOOP_MIN_CHUNKED_PREFILL_BATCHED_TOKENS", "8192"),
                        "TOOL_STAR_AGENT_LOOP_MIN_CHUNKED_PREFILL_BATCHED_TOKENS",
                    )
                    prompt_length = _coerce_optional_int(self.config.prompt_length, "rollout.prompt_length") or 0
                    safe_generation_tokens = self._get_safe_generation_default_max_new_tokens(log_clamp=False)
                    floor = max(int(throughput_floor or 0), int(prompt_length) + int(safe_generation_tokens))
                    floor = min(max_model_len, floor)
                else:
                    floor = floor_override

                max_tokens_per_mm_item = self._get_effective_max_tokens_per_mm_item(engine_kwargs)
                if (
                    self.model_config.processor is not None
                    and self._is_chunked_mm_input_disabled(engine_kwargs)
                    and max_tokens_per_mm_item is not None
                    and max_tokens_per_mm_item > 0
                ):
                    floor = max(floor, max_tokens_per_mm_item)

                if floor > 0 and effective < floor:
                    effective = floor

        if effective != requested:
            logger.warning(
                "Stability guard adjusted max_num_batched_tokens from %d to %d "
                "(max_model_len=%s, max_num_seqs=%s, enable_chunked_prefill=%s, "
                "disable_chunked_mm_input=%s, max_tokens_per_mm_item=%s). "
                "Set TOOL_STAR_AGENT_LOOP_STABILIZE_CHUNKED_PREFILL_BUDGET=0 to opt out.",
                requested,
                effective,
                max_model_len,
                max_num_seqs,
                enable_chunked_prefill,
                self._is_chunked_mm_input_disabled(engine_kwargs),
                self._get_effective_max_tokens_per_mm_item(engine_kwargs),
            )

        return effective

    def _get_effective_enable_chunked_prefill(self, engine_kwargs: dict[str, Any]) -> bool:
        raw_value = engine_kwargs.get("enable_chunked_prefill", self.config.enable_chunked_prefill)
        return bool(_coerce_optional_bool(raw_value, "rollout.enable_chunked_prefill"))

    def _apply_multimodal_block_table_stability_guard(self, engine_kwargs: dict[str, Any]) -> None:
        if self.model_config.processor is None:
            return

        if _read_env_flag("TOOL_STAR_AGENT_LOOP_DISABLE_MULTIMODAL_CHUNKED_PREFILL_ON_BLOCK_TABLE_RISK", True):
            enable_chunked_prefill = self._get_effective_enable_chunked_prefill(engine_kwargs)
            if enable_chunked_prefill:
                engine_kwargs["enable_chunked_prefill"] = False
                logger.warning(
                    "Stability guard active: set enable_chunked_prefill=False for multimodal AgentLoop vLLM "
                    "server to avoid V1 block_table slot-mapping instability on long video/image prefills. "
                    "Set TOOL_STAR_AGENT_LOOP_DISABLE_MULTIMODAL_CHUNKED_PREFILL_ON_BLOCK_TABLE_RISK=0 to opt out."
                )

        if _read_env_flag("TOOL_STAR_AGENT_LOOP_ENABLE_HYBRID_KV_CACHE_FOR_MULTIMODAL", True):
            requested_value = engine_kwargs.get("disable_hybrid_kv_cache_manager")
            if requested_value is not False:
                engine_kwargs["disable_hybrid_kv_cache_manager"] = False
                logger.warning(
                    "Stability guard active: set disable_hybrid_kv_cache_manager=False for multimodal AgentLoop "
                    "vLLM server. Current Qwen3.5/FLA hybrid models can fail to unify KV cache specs when "
                    "the hybrid KV cache manager is disabled (requested=%s). "
                    "Set TOOL_STAR_AGENT_LOOP_ENABLE_HYBRID_KV_CACHE_FOR_MULTIMODAL=0 to opt out.",
                    requested_value,
                )

    def _apply_multimodal_chunked_prefill_guard(self, engine_kwargs: dict[str, Any]) -> None:
        enable_chunked_prefill = self._get_effective_enable_chunked_prefill(engine_kwargs)
        if not enable_chunked_prefill or self.model_config.processor is None:
            return

        if not _read_env_flag("TOOL_STAR_AGENT_LOOP_FORCE_DISABLE_CHUNKED_MM_INPUT", True):
            return

        requested_value = engine_kwargs.get("disable_chunked_mm_input")
        if requested_value is not True:
            engine_kwargs["disable_chunked_mm_input"] = True
            logger.warning(
                "Stability guard active: set disable_chunked_mm_input=True for multimodal chunked prefill "
                "(requested=%s). Set TOOL_STAR_AGENT_LOOP_FORCE_DISABLE_CHUNKED_MM_INPUT=0 to opt out.",
                requested_value,
            )

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Called at the end of __init__. Default logs server metadata."""
        logger.info(
            f"{self.__class__.__name__}, replica_rank: {self.replica_rank}, node_rank: {self.node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"master_address: {self._master_address}, master_port: {self._master_port}, "
            f"data_parallel_rpc_port: {self._dp_rpc_port}, data_parallel_master_port: {self._dp_master_port}"
        )

    def _get_engine_kwargs_key(self) -> str:
        """Return the key under config.engine_kwargs for this engine (e.g. 'vllm')."""
        return "vllm"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Mutate engine_kwargs in-place before the CLI args dict is built. No-op by default."""
        pass

    def _get_override_generation_config(self) -> dict:
        """Return the override_generation_config dict."""
        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        return dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self._get_safe_generation_default_max_new_tokens(log_clamp=True),
        )

    def _apply_quantization(self) -> tuple[Optional[str], dict]:
        """Process quantization config. Returns (quantization_str, hf_overrides)."""
        quantization = self.config.quantization
        hf_overrides = {}

        if is_torch_npu_available(check_device=False):
            from verl.utils.vllm.npu_vllm_patch import check_vllm_ascend_before_server_launch

            check_vllm_ascend_before_server_launch()

        # Handle QAT (Quantization-Aware Training) configuration
        qat_config_dict = getattr(self.config, "qat", {}) or {}
        if qat_config_dict.get("enable", False):
            from verl.utils.qat import QATConfig, load_quantization_config

            qat_config = QATConfig(**qat_config_dict)
            quantization_config_dict = load_quantization_config(qat_config)
            quant_method = quantization_config_dict.get("quant_method", None)

            if quant_method == "modelopt":
                from verl.utils.modelopt import apply_modelopt_nvfp4_patches

                apply_modelopt_nvfp4_patches()
                quantization = "modelopt"
            elif quant_method == "compressed-tensors":
                from verl.utils.qat import apply_qat_patches

                apply_qat_patches()
                quantization = "compressed-tensors"
            else:
                raise ValueError(f"Unsupported quant_method: {quant_method}")

            logger.info(f"QAT quantization config injected (quant_method={quant_method})")
            hf_overrides["quantization_config"] = quantization_config_dict
        elif quantization is not None:
            # Handle other quantization methods (fp8, torchao)
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao", "ascend"]
            if quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {quantization}")

            if quantization == "fp8":
                # Ignore MoE router layers for FP8 quantization
                all_mlp_gate_layers = []
                for layer in range(self.model_config.hf_config.num_hidden_layers):
                    all_mlp_gate_layers.append(f"model.layers.{layer}.mlp.gate")

                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                    "ignored_layers": all_mlp_gate_layers,
                }
                hf_overrides["quantization_config"] = dict(FP8_BLOCK_QUANT_KWARGS)
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()
                # for subprocesses patching
                os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"

        if quantization is not None and self.config.quantization_config_file is not None:
            hf_overrides["quantization_config_file"] = self.config.quantization_config_file

        hf_overrides = _inject_rope_hf_overrides(self.model_config.hf_config, hf_overrides)

        return quantization, hf_overrides

    def _get_worker_extension_cls(self) -> str:
        """Return the fully-qualified colocate worker extension class name."""
        return "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        """Return the list of CLI command modules used for argument parsing."""
        return [vllm.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        """Return the description string for the CLI argument parser."""
        return "vLLM CLI"

    def _get_wake_up_tags(self) -> list[str]:
        """Return the tags passed to engine.wake_up(). Default includes kv_cache."""
        return ["kv_cache", "weights"]

    async def _sleep_hybrid(self):
        """HYBRID sleep: lora adapters only need level=1; full weights need level=2."""
        # Don't use engine.sleep(level=2) here
        # lora only update adapter weights, so set sleep level to 1
        if self.lora_as_adapter:
            sleep_level = 1
        else:
            sleep_level = 2
        await self.engine.collective_rpc("sleep", kwargs={"level": sleep_level})
        if _VLLM_VERSION >= version.parse("0.17.0"):
            await self.engine.reset_encoder_cache()


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model)
        self.server_class = ray.remote(vLLMHttpServer)

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        self._validate_launch_requirements()

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            prefix = self._get_server_name_prefix()
            if self.is_reward_model:
                name = f"{prefix}server_reward_{self.replica_rank}_{node_rank}"
            elif self.is_teacher_model:
                name = f"{prefix}server_teacher_{self.replica_rank}_{node_rank}"
            else:
                name = f"{prefix}server_{self.replica_rank}_{node_rank}"
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                        # To prevent hanging or crash during synchronization of weights between actor and rollout
                        # in disaggregated mode. See:
                        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
                        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
                        "NCCL_CUMEM_ENABLE": "0",
                        **self._get_server_runtime_env_vars(),
                    }
                },
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """Abort all ongoing generation requests across all servers.

        Returns:
            dict[str, Any]: Combined abort results from all servers.
        """
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def resume_generation(self):
        """Resume generation on all servers after abort_all_requests."""
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result.
        """
        # TODO(petersh6): we should only abort on the server that has the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _validate_launch_requirements(self) -> None:
        """Validate requirements before launching. Override in subclasses."""
        # NOTE: Executor backend is resolved dynamically in vLLMHttpServer.launch_server.
        # For multi-node MP Executor path without DP (e.g TP=16), vLLM>=0.11.1 is required:
        # https://github.com/vllm-project/vllm/pull/23691
        if self.config.data_parallel_size == 1 and self.nnodes > 1:
            assert _VLLM_VERSION >= version.parse("0.11.1"), (
                "For multi-node MP Executor, either (1) set data_parallel_size > 1 or (2) upgrade vLLM to >= 0.11.1"
            )

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix (e.g. 'vllm_' or 'vllm_omni_')."""
        return "vllm_"

    def _get_server_runtime_env_vars(self) -> dict[str, str]:
        """Return additional environment variables for server actors."""
        return {}


class vLLMVideoAgentReplica(vLLMReplica):
    """Adapter replica to run `vllm_video_agent` with AgentLoop's native rollout replica stack."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig | DictConfig | dict[str, Any],
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
    ):
        sanitized_config = _sanitize_video_agent_rollout_config_for_agent_loop(config)
        super().__init__(
            replica_rank=replica_rank,
            config=sanitized_config,
            model_config=model_config,
            gpus_per_node=gpus_per_node,
            is_reward_model=is_reward_model,
            is_teacher_model=is_teacher_model,
        )

    def _get_server_runtime_env_vars(self) -> dict[str, str]:
        env_vars = super()._get_server_runtime_env_vars()
        env_vars["TOOL_STAR_AGENT_LOOP_IS_VIDEO_AGENT_ROLLOUT"] = "1"
        return env_vars
