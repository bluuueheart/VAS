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

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask


def _load_structured_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                records.append(json.loads(text))
        return pd.DataFrame(records)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("samples") or payload.get("data") or [payload]
        return pd.DataFrame(list(payload or []))
    raise ValueError(f"Unsupported dataset file format: {path}")


def _coerce_options(options: Any) -> List[Dict[str, str]]:
    if isinstance(options, dict):
        normalized: List[Dict[str, str]] = []
        for key, value in options.items():
            normalized.append({"label": str(key), "text": str(value)})
        return normalized
    if isinstance(options, (list, tuple)):
        normalized = []
        for index, item in enumerate(options):
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("option") or chr(ord("A") + index))
                text = str(item.get("text") or item.get("value") or item.get("content") or "")
            else:
                label = chr(ord("A") + index)
                text = str(item)
            normalized.append({"label": label, "text": text})
        return normalized
    if isinstance(options, str) and options.strip():
        parts = [part.strip() for part in options.split("\n") if part.strip()]
        return [{"label": chr(ord("A") + idx), "text": part} for idx, part in enumerate(parts)]
    return []


def _build_prompt_text(question: str, task_type: str, options: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    if task_type:
        lines.append(f"Task Type: {task_type}")
    lines.append(f"Question: {question}")
    if options:
        lines.append("Options:")
        for item in options:
            label = str(item.get("label", "")).strip()
            text = str(item.get("text", "")).strip()
            if label:
                lines.append(f"{label}. {text}")
            else:
                lines.append(text)
        lines.append("Choose exactly one best option and justify it with grounded evidence.")
    return "\n".join(lines).strip()


def _coerce_reward_model(answer: Any, reward_model: Any) -> Dict[str, Any]:
    if isinstance(reward_model, dict):
        normalized = dict(reward_model)
    else:
        normalized = {}
    normalized.setdefault("ground_truth", answer)
    normalized.setdefault("style", "rule")
    return normalized


def _coerce_json_object(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _coerce_string_list(payload: Any) -> List[str]:
    if isinstance(payload, (list, tuple)):
        normalized: List[str] = []
        for item in payload:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    if isinstance(payload, str) and payload.strip():
        return [payload.strip()]
    return []


def _coerce_raw_prompt_messages(payload: Any, fallback_text: str) -> List[Dict[str, str]]:
    if isinstance(payload, list):
        normalized_messages: List[Dict[str, str]] = []
        for item in payload:
            if isinstance(item, dict):
                role = str(item.get("role") or "user").strip() or "user"
                content = str(item.get("content") or "").strip()
                if content:
                    normalized_messages.append({"role": role, "content": content})
            elif item is not None:
                text = str(item).strip()
                if text:
                    normalized_messages.append({"role": "user", "content": text})
        if normalized_messages:
            return normalized_messages

    if isinstance(payload, str) and payload.strip():
        return [{"role": "user", "content": payload.strip()}]

    fallback = str(fallback_text or "").strip()
    return [{"role": "user", "content": fallback}] if fallback else []


class VideoAgentRLDataset(Dataset):
    """Dataset adapter for video-agent RL episodes.

    The dataset keeps rich non-tensor metadata for the rollout environment while
    using a lightweight text placeholder prompt for batching/token budgeting.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/video_agent_rl"))
        self.prompt_key = config.get("prompt_key", "question")
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.filter_prompts = bool(config.get("filter_prompts", True))
        self.truncation = config.get("truncation", "error")
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed", None)

        self.serialize_dataset = False

        self._download()
        self._read_files()

    def _download(self, use_origin_parquet: bool = False) -> None:
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, data_file in enumerate(data_files):
            data_files[i] = copy_local_path_from_hdfs(src=data_file, cache_dir=self.cache_dir)
        if use_origin_parquet:
            self.data_files = data_files

    def _normalize_row(self, row: Dict[str, Any], index: int) -> Dict[str, Any]:
        episode_query = ""
        for candidate in (
            row.get("episode_query"),
            row.get("submitted_user_query"),
            row.get("formatted_query"),
            row.get("query"),
        ):
            text = str(candidate or "").strip()
            if text:
                episode_query = text
                break

        question = str(
            row.get("question")
            or row.get("query")
            or row.get(self.prompt_key)
            or ""
        ).strip()
        if not question:
            raise ValueError(f"Missing question/query field at row index={index}")

        task_type = str(row.get("task_type") or row.get("dataset_task_type") or "").strip()
        options = _coerce_options(row.get("options"))
        answer = row.get("answer")
        if answer in (None, ""):
            answer = row.get("label")
        sample_id = str(
            row.get("sample_id")
            or row.get("id")
            or row.get("uid")
            or f"sample_{index}"
        )
        dataset_name = str(row.get("dataset_name") or row.get("data_source") or "video_agent_rl").strip()
        prompt = _build_prompt_text(question=question, task_type=task_type, options=options)
        if not episode_query:
            episode_query = prompt

        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info = dict(extra_info)
        extra_info.setdefault("index", index)
        if options:
            extra_info.setdefault("options", options)
        if task_type:
            extra_info.setdefault("task_type", task_type)

        agent_name = str(
            row.get("agent_name")
            or extra_info.get("agent_name")
            or "single_turn_agent"
        ).strip()
        if not agent_name:
            agent_name = "single_turn_agent"

        interaction_kwargs = _coerce_json_object(row.get("interaction_kwargs"))
        if not interaction_kwargs:
            interaction_kwargs = _coerce_json_object(extra_info.get("interaction_kwargs"))

        tool_selection = _coerce_string_list(row.get("tool_selection"))
        if not tool_selection:
            tool_selection = _coerce_string_list(extra_info.get("tool_selection"))

        extra_info.setdefault("agent_name", agent_name)
        if interaction_kwargs:
            extra_info.setdefault("interaction_kwargs", dict(interaction_kwargs))
        if tool_selection:
            extra_info.setdefault("tool_selection", list(tool_selection))

        normalized = dict(row)
        raw_prompt_payload = row.get("raw_prompt")
        if raw_prompt_payload is None:
            raw_prompt_payload = row.get("messages")
        if raw_prompt_payload is None:
            raw_prompt_payload = row.get("conversation")
        raw_prompt = _coerce_raw_prompt_messages(raw_prompt_payload, episode_query or prompt)

        normalized.update(
            {
                "uid": str(row.get("uid") or sample_id),
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "data_source": dataset_name,
                "task_type": task_type,
                "question": question,
                "episode_query": episode_query,
                "prompt": prompt,
                "raw_prompt": raw_prompt,
                "options": options,
                "answer": answer,
                "agent_name": agent_name,
                "interaction_kwargs": interaction_kwargs,
                "tool_selection": tool_selection,
                "video_path": row.get("video_path") or row.get("video") or "",
                "video_root": row.get("video_root") or "",
                "dataset_root": row.get("dataset_root") or "",
                "workspace_root": row.get("workspace_root") or row.get("cache_root") or "",
                "processed_root": row.get("processed_root") or "",
                "time_budget_sec": row.get("time_budget_sec"),
                "root_step_budget": row.get("root_step_budget"),
                "subagent_step_budget": row.get("subagent_step_budget"),
                "reward_model": _coerce_reward_model(answer=answer, reward_model=row.get("reward_model")),
                "extra_info": extra_info,
            }
        )
        return normalized

    def _read_files(self) -> None:
        dataframes = []
        for data_file in self.data_files:
            dataframe = _load_structured_table(data_file)
            dataframes.append(dataframe)
        merged = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
        print(f"dataset len: {len(merged)}")

        if self.max_samples > 0 and self.max_samples < len(merged):
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                selected_indices = rng.choice(len(merged), size=self.max_samples, replace=False)
                merged = merged.iloc[selected_indices]
            else:
                merged = merged.iloc[: self.max_samples]
            merged = merged.reset_index(drop=True)

        normalized_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(merged.to_dict(orient="records")):
            normalized_rows.append(self._normalize_row(row, index=index))

        if self.filter_prompts:
            kept_rows: List[Dict[str, Any]] = []
            for row in normalized_rows:
                prompt = str(row.get("prompt", "")).strip()
                token_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                if token_count <= self.max_prompt_length:
                    kept_rows.append(row)
            normalized_rows = kept_rows

        self.records = normalized_rows
        print(f"filter dataset len: {len(self.records)}")

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = False if hasattr(self, "original_data_files") else True
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files()

    @classmethod
    async def process_vision_info(
        cls,
        messages: List[Dict[str, Any]],
        image_patch_size,
        config: DictConfig,
    ):
        """Extract images/videos from message payload for AgentLoop compatibility."""
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        row_dict = copy.deepcopy(self.records[item])
        prompt = str(row_dict.get("prompt", "")).strip()

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["index"] = int(row_dict.get("extra_info", {}).get("index", item))

        if isinstance(row_dict["reward_model"].get("ground_truth"), np.ndarray):
            row_dict["reward_model"]["ground_truth"] = row_dict["reward_model"]["ground_truth"].tolist()

        return row_dict

    def __getstate__(self) -> Dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "records" in state:
                del state["records"]
            return state
        return self.__dict__.copy()
