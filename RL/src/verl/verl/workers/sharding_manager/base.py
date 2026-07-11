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
"""
Sharding manager to implement HybridEngine
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group


logger = logging.getLogger(__name__)


class BaseShardingManager:
    def __init__(self):
        self.timing = {}

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data


class FSDPVideoAgentShardingManager(BaseShardingManager):
    """
    Materialize full params once so each rank can run its local rollout shard.

    Local `vllm_video_agent` rollout dispatch already shards the prompt batch by
    DP rank. Entering a rollout-lifetime `summon_full_params(...)` context lets
    each rank execute its own episodes without re-entering per-forward FSDP
    collectives during planner generation.
    """

    def __init__(
        self,
        module: Optional[FSDP] = None,
        *,
        materialize_nested_fsdp_modules: bool = False,
    ):
        super().__init__()
        self._module = module if isinstance(module, FSDP) else None
        self._dist_enabled = torch.distributed.is_available() and torch.distributed.is_initialized()
        self._world_size = torch.distributed.get_world_size() if self._dist_enabled else 1
        self._rank = torch.distributed.get_rank() if self._dist_enabled else 0
        self._full_params_context = None
        self._materialize_nested_fsdp_modules = bool(materialize_nested_fsdp_modules)

    def __enter__(self):
        if self._module is not None:
            # Keep nested FSDP wrappers sharded unless explicitly requested.
            # Recursive summon_full_params() previously destabilized nested
            # hybrid actor/rollout runs.
            self._full_params_context = FSDP.summon_full_params(
                self._module,
                writeback=False,
                recurse=self._materialize_nested_fsdp_modules,
            )
            self._full_params_context.__enter__()
        logger.info(
            "[FSDPVideoAgentShardingManager][DIAG] enter world_size=%d rank=%d mode=full_params_local_shard full_params_active=%s recurse=%s",
            self._world_size,
            self._rank,
            self._full_params_context is not None,
            self._materialize_nested_fsdp_modules,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._full_params_context is None:
            return
        self._full_params_context.__exit__(exc_type, exc_value, traceback)
        self._full_params_context = None

    def preprocess_data(self, data: DataProto) -> DataProto:
        local_batch_size = len(data)
        sample_ids = [
            str(item)
            for item in list((data.non_tensor_batch or {}).get("sample_id", np.empty((0,), dtype=object)))
        ]
        padding_count = int(
            sum(
                bool(item)
                for item in list(
                    (data.non_tensor_batch or {}).get("__verl_is_padding", np.empty((0,), dtype=object))
                )
            )
        )
        logger.info(
            "[FSDPVideoAgentShardingManager][DIAG] preprocess world_size=%d rank=%d local_batch_size=%d padding_count=%d sample_ids=%s keys=%s",
            self._world_size,
            self._rank,
            local_batch_size,
            padding_count,
            sample_ids,
            list((data.non_tensor_batch or {}).keys()),
        )
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        local_sample_ids = [
            str(item)
            for item in list((data.non_tensor_batch or {}).get("sample_id", np.empty((0,), dtype=object)))
        ]
        logger.info(
            "[FSDPVideoAgentShardingManager][DIAG] postprocess world_size=%d rank=%d local_batch_size=%d sample_ids=%s",
            self._world_size,
            self._rank,
            len(data),
            local_sample_ids,
        )
        return data


class FSDPVideoAgentUlyssesShardingManager(FSDPVideoAgentShardingManager):
    """
    Local video-agent rollout sharding manager with Ulysses SP runtime context.

    The episode/tool loop should remain leader-driven, but model forward/generate
    calls can still run with an SP group active so one rollout may consume
    multiple GPUs.
    """

    def __init__(
        self,
        module: Optional[FSDP] = None,
        *,
        ulysses_device_mesh=None,
        materialize_nested_fsdp_modules: bool = False,
    ):
        super().__init__(
            module=module,
            materialize_nested_fsdp_modules=materialize_nested_fsdp_modules,
        )
        self._ulysses_device_mesh = ulysses_device_mesh
        self._prev_sp_group = None
        self._sp_enabled = ulysses_device_mesh is not None
        if self._sp_enabled:
            self._sp_rank = int(ulysses_device_mesh["sp"].get_local_rank())
            self._sp_size = int(ulysses_device_mesh["sp"].size())
            self._dp_rank = int(ulysses_device_mesh["dp"].get_local_rank())
        else:
            self._sp_rank = 0
            self._sp_size = 1
            self._dp_rank = self._rank

    def __enter__(self):
        super().__enter__()
        if self._ulysses_device_mesh is not None:
            self._prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self._ulysses_device_mesh["sp"].get_group())
        logger.info(
            "[FSDPVideoAgentUlyssesShardingManager][DIAG] enter world_size=%d rank=%d dp_rank=%d sp_rank=%d sp_size=%d sp_enabled=%s",
            self._world_size,
            self._rank,
            self._dp_rank,
            self._sp_rank,
            self._sp_size,
            self._sp_enabled,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._ulysses_device_mesh is not None:
            set_ulysses_sequence_parallel_group(self._prev_sp_group)
            self._prev_sp_group = None
        super().__exit__(exc_type, exc_value, traceback)

    def preprocess_data(self, data: DataProto) -> DataProto:
        data = super().preprocess_data(data)
        logger.info(
            "[FSDPVideoAgentUlyssesShardingManager][DIAG] preprocess rank=%d dp_rank=%d sp_rank=%d local_batch_size=%d",
            self._rank,
            self._dp_rank,
            self._sp_rank,
            len(data),
        )
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        data = super().postprocess_data(data)
        logger.info(
            "[FSDPVideoAgentUlyssesShardingManager][DIAG] postprocess rank=%d dp_rank=%d sp_rank=%d local_batch_size=%d",
            self._rank,
            self._dp_rank,
            self._sp_rank,
            len(data),
        )
        return data
