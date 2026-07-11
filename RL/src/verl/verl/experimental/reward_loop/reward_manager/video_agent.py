# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import json
from typing import Any, Dict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score.video_agent import (
    build_video_agent_role_training_targets,
    compute_video_agent_reward,
    is_subagent_rl_contribution_enabled,
)


def _as_json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _to_metric_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)):
        return value
    return None


def _append_metric(metrics: Dict[str, Any], key: str, value: Any) -> None:
    scalar = _to_metric_scalar(value)
    if scalar is None:
        return
    metrics[key] = scalar


@register("video_agent")
class VideoAgentRewardManager(RewardManagerBase):
    """Reward manager for multi-turn video-agent rollout."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("video_agent_reward", None)
        if reward_cfg is None:
            reward_cfg = config.reward.get("reward_kwargs", {})
        self.reward_config = dict(reward_cfg or {})
        self.subagent_rl_enabled = is_subagent_rl_contribution_enabled(self.reward_config)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        reward_model = data_item.non_tensor_batch.get("reward_model")
        if not isinstance(reward_model, dict):
            reward_model = {}

        episode_meta = data_item.non_tensor_batch.get("episode_reward_features")
        if not isinstance(episode_meta, dict):
            episode_meta = {}

        role_targets = build_video_agent_role_training_targets(
            trajectory=data_item.non_tensor_batch.get("episode_trajectory"),
            generation_records=data_item.non_tensor_batch.get("episode_generation_records"),
            response_length=int(data_item.batch["responses"].shape[-1]),
            reward_config=self.reward_config,
        )
        role_training_summary = dict(role_targets.get("summary") or {})
        episode_meta["role_training_summary"] = role_training_summary

        breakdown = compute_video_agent_reward(
            run_result=data_item.non_tensor_batch.get("episode_result"),
            trajectory=data_item.non_tensor_batch.get("episode_trajectory"),
            final_answer=data_item.non_tensor_batch.get("episode_final_answer"),
            ground_truth=reward_model.get("ground_truth"),
            options=data_item.non_tensor_batch.get("options"),
            dataset_name=(
                data_item.non_tensor_batch.get("dataset_name")
                or data_item.non_tensor_batch.get("data_source")
                or ""
            ),
            reward_model=reward_model,
            episode_meta=episode_meta,
            reward_config=self.reward_config,
        )

        score = float(breakdown.get("score", 0.0) or 0.0)
        features = dict(breakdown.get("features") or {})
        components = dict(breakdown.get("components") or {})

        reward_extra_info: Dict[str, Any] = {}
        reward_extra_info["acc"] = float(1.0 if bool(features.get("answer_correct", False)) else 0.0)
        reward_extra_info["answer_credit"] = float(features.get("answer_credit", 0.0) or 0.0)

        for feature_key in (
            "acceptance_passed",
            "tool_chain_valid",
            "finish_valid",
            "evidence_verification_score",
            "evidence_logical_rigor_score",
            "planner_evidence_iteration_count",
            "planner_evidence_item_count",
            "planner_verified_evidence_item_count",
            "planner_conflicting_evidence_item_count",
            "planner_unmarked_conflict_evidence_item_count",
            "successful_tool_calls",
            "tool_failure_count",
            "runtime_tool_count",
            "unique_runtime_tool_count",
            "planner_iterations",
            "subagent_iterations",
            "invalid_json_count",
            "invalid_action_count",
            "repeated_action_count",
            "timeout_exceeded",
            "budget_exceeded",
            "wups_0",
            "wups_9",
            "keypoint_recall",
            "rouge_l_f1",
            "elapsed_seconds",
        ):
            _append_metric(reward_extra_info, feature_key, features.get(feature_key))

        for key, value in components.items():
            _append_metric(reward_extra_info, f"component_{key}", value)

        for key, value in role_training_summary.items():
            _append_metric(reward_extra_info, f"role_train_{key}", value)

        # Keep detailed debug payload as plain text so validation metric aggregation
        # can skip it safely.
        reward_extra_info["reward_breakdown"] = _as_json_text(breakdown)

        output = {
            "reward_score": score,
            "reward_extra_info": reward_extra_info,
            # Non-metric training masks are propagated separately and are not
            # included in reward_extra_keys.
            "policy_loss_mask": list(role_targets.get("policy_loss_mask") or []),
            "planner_token_mask": list(role_targets.get("planner_token_mask") or []),
        }

        if self.subagent_rl_enabled:
            output["subagent_token_mask"] = list(role_targets.get("subagent_token_mask") or [])
            output["subagent_local_advantages"] = list(role_targets.get("subagent_local_advantages") or [])

        return output
