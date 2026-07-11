from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_SENTENCE_T5_CACHE: Dict[str, Any] = {
    "model_path": "",
    "tokenizer": None,
    "model": None,
    "device": "cpu",
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_open_answer_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fa5\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_open_answer(value: Any) -> List[str]:
    return [token for token in _normalize_open_answer_text(value).split(" ") if token]


def _normalize_dataset_name(dataset_name: Any) -> str:
    return str(dataset_name or "").strip().lower()


def is_subagent_rl_contribution_enabled(reward_config: Optional[Dict[str, Any]] = None) -> bool:
    config = dict(reward_config or {})
    return bool(config.get("enable_subagent_rl_contribution", False))


@lru_cache(maxsize=16384)
def _cached_wordnet_synsets(word: str) -> Tuple[Any, ...]:
    try:
        from nltk.corpus import wordnet
    except Exception:
        return tuple()
    try:
        return tuple(wordnet.synsets(word))
    except Exception:
        return tuple()


def _wup_word_similarity(word1: str, word2: str, alpha: float) -> float:
    if not word1 or not word2:
        return 0.0
    if word1 == word2:
        return 1.0

    synsets1 = _cached_wordnet_synsets(word1)
    synsets2 = _cached_wordnet_synsets(word2)
    if not synsets1 or not synsets2:
        return 0.0

    similarity = synsets1[0].wup_similarity(synsets2[0])
    if similarity is None:
        similarity = 0.0
    if similarity < alpha:
        similarity = 0.1 * float(similarity)
    return float(similarity)


def _wups_directional(words1: Sequence[str], words2: Sequence[str], alpha: float) -> float:
    score = 1.0
    matched = False
    for source_word in words1:
        max_similarity = 0.0
        for target_word in words2:
            word_similarity = _wup_word_similarity(source_word, target_word, alpha)
            if word_similarity > max_similarity:
                max_similarity = word_similarity
        if max_similarity <= 0.0:
            continue
        score *= max_similarity
        matched = True
    if not matched:
        return 0.0
    return float(score)


def _compute_nextqa_wups_scores(prediction: Any, reference: Any) -> Dict[str, float]:
    prediction_tokens = _tokenize_open_answer(prediction)
    reference_tokens = _tokenize_open_answer(reference)
    if not prediction_tokens or not reference_tokens:
        return {"wups_0": 0.0, "wups_9": 0.0}

    directional_0 = _wups_directional(prediction_tokens, reference_tokens, alpha=0.0)
    directional_0_rev = _wups_directional(reference_tokens, prediction_tokens, alpha=0.0)
    directional_9 = _wups_directional(prediction_tokens, reference_tokens, alpha=0.9)
    directional_9_rev = _wups_directional(reference_tokens, prediction_tokens, alpha=0.9)
    return {
        "wups_0": float(min(directional_0, directional_0_rev)),
        "wups_9": float(min(directional_9, directional_9_rev)),
    }


def _resolve_sentence_t5_model_path(reward_config: Optional[Dict[str, Any]] = None) -> str:
    reward_config = dict(reward_config or {})
    configured_candidates = [
        reward_config.get("videobench_sentence_t5_path"),
        reward_config.get("sentence_t5_model_path"),
        os.getenv("VIDEO_AGENT_SENTENCE_T5_PATH", ""),
        "sentence-transformers/sentence-t5-base",
        "./sentence-transformers/sentence-t5-base",
        "./models/sentence-transformers/sentence-t5-base",
    ]
    for candidate in configured_candidates:
        candidate_text = str(candidate or "").strip()
        if candidate_text and os.path.isdir(candidate_text):
            return candidate_text
    return ""


def _load_sentence_t5_encoder(model_path: str):
    if not model_path:
        return None
    cached_path = str(_SENTENCE_T5_CACHE.get("model_path") or "")
    if (
        cached_path == model_path
        and _SENTENCE_T5_CACHE.get("tokenizer") is not None
        and _SENTENCE_T5_CACHE.get("model") is not None
    ):
        return _SENTENCE_T5_CACHE

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
    except Exception:
        return None

    _SENTENCE_T5_CACHE.update(
        {
            "model_path": model_path,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }
    )
    return _SENTENCE_T5_CACHE


def _encode_with_sentence_t5(texts: Sequence[str], encoder_state: Dict[str, Any]):
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return None

    tokenizer = encoder_state.get("tokenizer")
    model = encoder_state.get("model")
    device = encoder_state.get("device", "cpu")
    if tokenizer is None or model is None:
        return None

    try:
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        hidden_states = output.last_hidden_state
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            return None
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled = (hidden_states * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(pooled, p=2, dim=1)
    except Exception:
        return None


def _predict_label_with_sentence_t5(
    final_answer: Any,
    options: Any,
    reward_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, str]:
    normalized_options = _coerce_options(options)
    option_labels = [str(item.get("label", "")).strip() for item in normalized_options]
    option_texts = [str(item.get("text", "")).strip() for item in normalized_options]
    if not normalized_options or not option_texts:
        return "", 0.0, ""

    model_path = _resolve_sentence_t5_model_path(reward_config=reward_config)
    if not model_path:
        return "", 0.0, ""

    encoder_state = _load_sentence_t5_encoder(model_path)
    if encoder_state is None:
        return "", 0.0, model_path

    answer_text = str(final_answer or "").strip()
    if not answer_text:
        return "", 0.0, model_path

    embeddings = _encode_with_sentence_t5([answer_text] + option_texts, encoder_state)
    if embeddings is None:
        return "", 0.0, model_path

    try:
        import torch
    except Exception:
        return "", 0.0, model_path

    answer_embedding = embeddings[0:1]
    option_embeddings = embeddings[1:]
    similarities = torch.matmul(answer_embedding, option_embeddings.T).squeeze(0)
    if similarities.numel() <= 0:
        return "", 0.0, model_path
    best_index = int(torch.argmax(similarities).item())
    if best_index < 0 or best_index >= len(option_labels):
        return "", 0.0, model_path
    return option_labels[best_index], float(similarities[best_index].item()), model_path


def _normalize_keypoints(raw_keypoints: Any, source_hint: str = "") -> Dict[str, List[str]]:
    if isinstance(raw_keypoints, dict):
        video_values = raw_keypoints.get("video", [])
        text_values = raw_keypoints.get("text", [])
        return {
            "video": [str(item).strip() for item in video_values if str(item).strip()],
            "text": [str(item).strip() for item in text_values if str(item).strip()],
        }

    normalized_values = [str(item).strip() for item in (raw_keypoints or []) if str(item).strip()]
    source_name = str(source_hint or "").strip().lower()
    if "vsingleqa" in source_name:
        return {"video": normalized_values, "text": []}
    if "tsingleqa" in source_name:
        return {"video": [], "text": normalized_values}
    return {"video": [], "text": normalized_values}


def _longest_common_subsequence_length(left: List[str], right: List[str]) -> int:
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l_f1(reference: Any, candidate: Any) -> float:
    reference_tokens = _tokenize_open_answer(reference)
    candidate_tokens = _tokenize_open_answer(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    lcs_length = _longest_common_subsequence_length(reference_tokens, candidate_tokens)
    if lcs_length <= 0:
        return 0.0

    precision = float(lcs_length) / float(len(candidate_tokens))
    recall = float(lcs_length) / float(len(reference_tokens))
    if precision + recall <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _keypoint_matches_answer(keypoint: str, answer_tokens: List[str], answer_token_set: set[str], answer_text: str) -> bool:
    normalized_keypoint = _normalize_open_answer_text(keypoint)
    if not normalized_keypoint:
        return False
    keypoint_tokens = [token for token in normalized_keypoint.split(" ") if token]
    if not keypoint_tokens:
        return False

    if normalized_keypoint in answer_text and len(keypoint_tokens) >= 3:
        return True

    keypoint_token_set = set(keypoint_tokens)
    overlap = len(keypoint_token_set & answer_token_set)
    required_overlap = max(1, int(math.ceil(min(len(keypoint_tokens), len(answer_tokens)) * 0.6))) if answer_tokens else 1
    union = keypoint_token_set | answer_token_set if answer_token_set else keypoint_token_set
    jaccard = float(overlap) / float(len(union)) if union else 0.0
    return overlap >= required_overlap or jaccard >= 0.5


def _collect_keypoint_match_metrics(keypoints: Dict[str, List[str]], final_answer: Any) -> Dict[str, Any]:
    answer_text = _normalize_open_answer_text(final_answer)
    answer_tokens = [token for token in answer_text.split(" ") if token]
    answer_token_set = set(answer_tokens)

    matched_video = 0
    matched_text = 0
    total_video = len(keypoints.get("video", []))
    total_text = len(keypoints.get("text", []))

    for keypoint in keypoints.get("video", []):
        if _keypoint_matches_answer(keypoint, answer_tokens, answer_token_set, answer_text):
            matched_video += 1

    for keypoint in keypoints.get("text", []):
        if _keypoint_matches_answer(keypoint, answer_tokens, answer_token_set, answer_text):
            matched_text += 1

    total_keypoints = total_video + total_text
    matched_total = matched_video + matched_text
    return {
        "matched_video_keypoints": matched_video,
        "matched_text_keypoints": matched_text,
        "matched_total_keypoints": matched_total,
        "total_video_keypoints": total_video,
        "total_text_keypoints": total_text,
        "total_keypoints": total_keypoints,
        "video_keypoint_recall": float(matched_video) / float(total_video) if total_video > 0 else 0.0,
        "text_keypoint_recall": float(matched_text) / float(total_text) if total_text > 0 else 0.0,
        "keypoint_recall": float(matched_total) / float(total_keypoints) if total_keypoints > 0 else 0.0,
    }


def _compute_open_answer_alignment(
    *,
    final_answer: Any,
    ground_truth: Any,
    keypoints: Dict[str, List[str]],
    reward_style: str = "",
    dataset_name: str = "",
    reward_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reward_config = dict(reward_config or {})
    normalized_answer = _normalize_open_answer_text(final_answer)
    normalized_ground_truth = _normalize_open_answer_text(ground_truth)
    rouge_l_f1 = _rouge_l_f1(ground_truth, final_answer)
    keypoint_metrics = _collect_keypoint_match_metrics(keypoints, final_answer)

    exact_match = bool(normalized_answer and normalized_answer == normalized_ground_truth)
    contains_ground_truth = bool(
        normalized_ground_truth
        and normalized_answer
        and normalized_ground_truth in normalized_answer
    )
    answer_contained_by_ground_truth = bool(
        normalized_ground_truth
        and normalized_answer
        and normalized_answer in normalized_ground_truth
        and len(_tokenize_open_answer(final_answer)) >= 4
    )

    rouge_weight = float(reward_config.get("open_answer_rouge_weight", 0.35))
    keypoint_weight = float(reward_config.get("open_answer_keypoint_weight", 0.65))
    correct_threshold = float(reward_config.get("open_answer_correct_threshold", 0.6))
    normalized_reward_style = str(reward_style or "").strip().lower()
    normalized_dataset_name = _normalize_dataset_name(dataset_name)
    use_recall_reward = normalized_reward_style == "cfvbench"
    use_nextqa_wups_reward = normalized_dataset_name == "nextqa" and not use_recall_reward

    if keypoint_metrics["total_keypoints"] <= 0:
        blended_credit = rouge_l_f1
    else:
        blended_credit = (
            rouge_weight * float(rouge_l_f1)
            + keypoint_weight * float(keypoint_metrics["keypoint_recall"])
        )

    nextqa_wups_scores = {"wups_0": 0.0, "wups_9": 0.0}

    if use_nextqa_wups_reward:
        nextqa_wups_scores = _compute_nextqa_wups_scores(final_answer, ground_truth)
        answer_credit = float(nextqa_wups_scores["wups_0"])
    elif use_recall_reward:
        if keypoint_metrics["total_keypoints"] > 0:
            answer_credit = float(keypoint_metrics["keypoint_recall"])
        else:
            answer_credit = float(rouge_l_f1)
    else:
        if exact_match:
            answer_credit = 1.0
        else:
            answer_credit = max(0.0, min(1.0, blended_credit))
            if contains_ground_truth:
                answer_credit = max(answer_credit, 0.85)
            elif answer_contained_by_ground_truth:
                answer_credit = max(answer_credit, 0.5)

    if use_nextqa_wups_reward:
        # Official NExTQA-OE reports WUPS scores rather than a binary thresholded accuracy.
        # Keep answer_correct as exact-match diagnostic only; reward itself comes from WUPS@0.
        is_correct = bool(exact_match)
        correct_threshold = 0.0
    else:
        is_correct = bool(exact_match or answer_credit >= correct_threshold)

    answer_reward_basis = "blended_open_answer_alignment"
    if use_recall_reward:
        answer_reward_basis = "keypoint_recall"
    elif use_nextqa_wups_reward:
        answer_reward_basis = "nextqa_wups0"

    return {
        "answer_credit": float(max(0.0, min(1.0, answer_credit))),
        "answer_correct": is_correct,
        "answer_exact_match": exact_match,
        "answer_contains_ground_truth": contains_ground_truth,
        "answer_contained_by_ground_truth": answer_contained_by_ground_truth,
        "rouge_l_f1": float(rouge_l_f1),
        "rouge_weight": float(rouge_weight),
        "keypoint_weight": float(keypoint_weight),
        "correct_threshold": float(correct_threshold),
        "answer_reward_basis": answer_reward_basis,
        "wups_0": float(nextqa_wups_scores["wups_0"]),
        "wups_9": float(nextqa_wups_scores["wups_9"]),
        "wups_9_report_only": bool(use_nextqa_wups_reward),
        **keypoint_metrics,
    }


def _coerce_options(options: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if isinstance(options, dict):
        for label, text in options.items():
            normalized.append({"label": str(label).strip(), "text": str(text).strip()})
        return normalized
    if isinstance(options, (list, tuple)):
        for index, item in enumerate(options):
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("option") or chr(ord("A") + index)).strip()
                text = str(item.get("text") or item.get("value") or item.get("content") or "").strip()
            else:
                label = chr(ord("A") + index)
                text = str(item).strip()
            normalized.append({"label": label, "text": text})
    return normalized


def _dedupe(values: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def resolve_ground_truth_labels(ground_truth: Any, options: Any) -> List[str]:
    normalized_options = _coerce_options(options)
    labels = [str(item.get("label", "")).strip() for item in normalized_options if str(item.get("label", "")).strip()]
    option_text_to_label = {
        _normalize_text(item.get("text")): str(item.get("label", "")).strip()
        for item in normalized_options
        if _normalize_text(item.get("text")) and str(item.get("label", "")).strip()
    }

    values: List[str] = []
    if isinstance(ground_truth, (list, tuple, set)):
        for item in ground_truth:
            values.extend(resolve_ground_truth_labels(item, normalized_options))
        return _dedupe(values)

    text = str(ground_truth or "").strip()
    if not text:
        return []
    normalized = _normalize_text(text)

    for label in labels:
        if normalized == _normalize_text(label):
            values.append(label)

    option_label_match = re.search(r"\b([A-Z])\b", text.upper())
    if option_label_match and option_label_match.group(1) in labels:
        values.append(option_label_match.group(1))

    if normalized in option_text_to_label:
        values.append(option_text_to_label[normalized])

    try:
        numeric_value = int(float(text))
    except Exception:
        numeric_value = None

    if numeric_value is not None and labels:
        if 0 <= numeric_value < len(labels):
            values.append(labels[numeric_value])
        if 1 <= numeric_value <= len(labels):
            values.append(labels[numeric_value - 1])

    return _dedupe(values)


def extract_predicted_label(final_answer: Any, options: Any) -> str:
    answer_text = str(final_answer or "").strip()
    if not answer_text:
        return ""

    normalized_options = _coerce_options(options)
    labels = [str(item.get("label", "")).strip() for item in normalized_options if str(item.get("label", "")).strip()]
    answer_upper = answer_text.upper()

    patterns = (
        r"(?:FINAL ANSWER|ANSWER|OPTION|CHOOSE|PICK)\s*(?:IS|:)?\s*\(?([A-Z])\)?",
        r"\(([A-Z])\)",
        r"\b([A-Z])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, answer_upper)
        for match in reversed(matches):
            label = str(match).strip().upper()
            if label in labels:
                return label

    normalized_answer = _normalize_text(answer_text)
    matched_labels: List[str] = []
    for option in normalized_options:
        label = str(option.get("label", "")).strip()
        option_text = _normalize_text(option.get("text"))
        if label and option_text and option_text in normalized_answer:
            matched_labels.append(label)
    matched_labels = _dedupe(matched_labels)
    if len(matched_labels) == 1:
        return matched_labels[0]

    return ""


def _evaluate_rule_answer(
    final_answer: Any,
    ground_truth: Any,
    options: Any,
    *,
    dataset_name: str = "",
    reward_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_dataset = _normalize_dataset_name(dataset_name)
    predicted_label = extract_predicted_label(final_answer, options)
    option_match_method = "label_or_option_text_match"
    option_match_similarity = 0.0
    sentence_t5_model_path = ""

    if not predicted_label and normalized_dataset == "videobench":
        judged_label, judged_score, judged_model_path = _predict_label_with_sentence_t5(
            final_answer=final_answer,
            options=options,
            reward_config=reward_config,
        )
        if judged_label:
            predicted_label = judged_label
            option_match_method = "videobench_sentence_t5_similarity"
            option_match_similarity = float(judged_score)
            sentence_t5_model_path = str(judged_model_path or "")

    ground_truth_labels = resolve_ground_truth_labels(ground_truth, options)
    label_matched = bool(predicted_label and predicted_label in ground_truth_labels)
    normalized_answer = _normalize_text(final_answer)
    exact_text_match = bool(normalized_answer and normalized_answer == _normalize_text(ground_truth))
    answer_correct = bool(label_matched or exact_text_match)

    if normalized_dataset == "videobench":
        correctness_method = "videobench_option_judge"
        correctness_rule = (
            "official Video-Bench recommendation: map free-form output to option label; "
            "prefer explicit label parse, otherwise use sentence-T5 similarity judge."
        )
        evaluation_mode = "videobench_option_judge"
    else:
        correctness_method = "label_or_exact_text_match"
        correctness_rule = (
            "correct when predicted option label matches any ground-truth label or the normalized final answer exactly "
            "matches the ground truth text"
        )
        evaluation_mode = "accuracy"

    return {
        "answer_correct": bool(answer_correct),
        "predicted_label": str(predicted_label or "").strip(),
        "ground_truth_labels": list(ground_truth_labels),
        "option_match_method": option_match_method,
        "option_match_similarity": float(option_match_similarity),
        "sentence_t5_model_path": sentence_t5_model_path,
        "correctness_method": correctness_method,
        "correctness_rule": correctness_rule,
        "evaluation_mode": evaluation_mode,
    }


def is_answer_correct(
    final_answer: Any,
    ground_truth: Any,
    options: Any,
    *,
    dataset_name: str = "",
    reward_config: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str, List[str]]:
    result = _evaluate_rule_answer(
        final_answer=final_answer,
        ground_truth=ground_truth,
        options=options,
        dataset_name=dataset_name,
        reward_config=reward_config,
    )
    return bool(result.get("answer_correct")), str(result.get("predicted_label") or ""), list(
        result.get("ground_truth_labels") or []
    )


def _iter_child_trajectories(step: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    sub_trajectory = step.get("sub_trajectory")
    if isinstance(sub_trajectory, dict):
        yield sub_trajectory
    sub_trajectories = step.get("sub_trajectories")
    if isinstance(sub_trajectories, list):
        for item in sub_trajectories:
            if isinstance(item, dict):
                yield item


def _iter_nested_observation_items(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_nested_observation_items(item)
        return

    if not isinstance(value, dict):
        return

    nested_items = value.get("items")
    if isinstance(nested_items, list):
        for item in nested_items:
            yield from _iter_nested_observation_items(item)

    tool_name = str(value.get("tool", "")).strip()
    status = str(value.get("status", "")).strip()
    if tool_name:
        yield value
        return

    result_payload = value.get("result")
    if isinstance(result_payload, dict):
        yield from _iter_nested_observation_items(result_payload)


def _infer_step_status(step: Dict[str, Any]) -> str:
    observation_items = step.get("observation_items")
    if isinstance(observation_items, list):
        for item in observation_items:
            if isinstance(item, dict):
                status = str(item.get("status", "")).strip()
                if status:
                    return status
    observation = _normalize_text(step.get("observation"))
    if observation.startswith("error"):
        return "error"
    return "success"


def _collect_tool_records(trajectory: Any) -> List[Dict[str, Any]]:
    if not isinstance(trajectory, dict):
        return []

    depth = int(trajectory.get("depth", 0) or 0)
    records: List[Dict[str, Any]] = []
    for step in trajectory.get("steps", []) or []:
        if not isinstance(step, dict):
            continue

        child_trajectories = list(_iter_child_trajectories(step))
        if child_trajectories:
            for child in child_trajectories:
                records.extend(_collect_tool_records(child))
            continue

        action = str(step.get("action", "")).strip()
        if not action or action in {"Finish", "System", "Assistant_Note", "Protocol_Error"}:
            continue

        if action == "Parallel":
            observation_items = step.get("observation_items")
            nested_records = [
                {**item, "depth": depth}
                for item in _iter_nested_observation_items(observation_items)
                if str(item.get("tool", "")).strip() not in {"", "Parallel", "System", "Assistant_Note"}
            ]
            records.extend(nested_records)
            continue

        status = _infer_step_status(step)
        records.append({"tool": action, "status": status, "depth": depth})
    return records


def collect_debug_event_counts(trajectory: Any, include_recursive: bool = True) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(trajectory, dict):
        return counts

    for event in trajectory.get("debug_events", []) or []:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", "")).strip()
        reason = str(event.get("reason", "")).strip()
        if event_type:
            counts[event_type] = counts.get(event_type, 0) + 1
        if event_type and reason:
            key = f"{event_type}:{reason}"
            counts[key] = counts.get(key, 0) + 1

    for step in trajectory.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        if include_recursive:
            for child in _iter_child_trajectories(step):
                child_counts = collect_debug_event_counts(child, include_recursive=include_recursive)
                for key, value in child_counts.items():
                    counts[key] = counts.get(key, 0) + int(value)

    return counts


def _iter_steps_preorder(trajectory: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(trajectory, dict):
        return
    for step in trajectory.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        yield step
        for child in _iter_child_trajectories(step):
            yield from _iter_steps_preorder(child)


def _normalize_action_args_for_signature(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: Dict[str, Any] = {}
        for key in sorted(str(raw_key) for raw_key in value.keys()):
            if key.startswith("__"):
                continue
            normalized[key] = _normalize_action_args_for_signature(value.get(key))
        return normalized
    if isinstance(value, list):
        return [_normalize_action_args_for_signature(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_action_args_for_signature(item) for item in value)
    return value


def _build_step_signature(step: Dict[str, Any]) -> str:
    action = str(step.get("action", "")).strip()
    if not action:
        return ""
    payload = {
        "action": action,
        "args": _normalize_action_args_for_signature(step.get("action_args") or {}),
    }
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(payload)


def _extract_step_statuses(step: Dict[str, Any]) -> List[str]:
    statuses: List[str] = []
    observation_items = step.get("observation_items")
    if isinstance(observation_items, list):
        for item in observation_items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).strip().lower()
            if status:
                statuses.append(status)
    if statuses:
        return statuses
    inferred = _infer_step_status(step)
    return [inferred] if inferred else []


def _normalize_step_text(step: Dict[str, Any]) -> str:
    fragments = [
        str(step.get("action", "")).strip(),
        str(step.get("observation", "")).strip(),
        _normalize_text(step.get("thought")),
        _normalize_text(step.get("plan")),
    ]
    observation_items = step.get("observation_items")
    if isinstance(observation_items, list):
        try:
            fragments.append(json.dumps(observation_items, ensure_ascii=False, sort_keys=True))
        except Exception:
            fragments.append(str(observation_items))
    return " ".join(fragment for fragment in fragments if fragment).lower()


def _looks_like_invalid_json_step(step: Dict[str, Any]) -> bool:
    action = str(step.get("action", "")).strip().lower()
    if action == "protocol_error":
        return True
    text = _normalize_step_text(step)
    return any(
        marker in text
        for marker in (
            "no valid action parsed",
            "strict json",
            "protocol violation",
            "parse failure",
            "failed to parse",
        )
    )


def _looks_like_invalid_action_step(step: Dict[str, Any]) -> bool:
    text = _normalize_step_text(step)
    return any(
        marker in text
        for marker in (
            "invalid_action_name",
            "invalid action",
            "unknown tool",
            "tool not found",
            "does not exist",
            "no concrete tool",
            "not a concrete tool",
            "not an executable tool",
        )
    )


def _compute_subagent_step_reward(
    step: Optional[Dict[str, Any]],
    *,
    seen_signatures: set[str],
    reward_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reward_config = dict(reward_config or {})
    local_reward_scale = float(reward_config.get("subagent_local_reward_scale", 1.0) or 1.0)
    decomposition_weight = float(reward_config.get("subagent_decomposition_weight", 0.4) or 0.4)
    executable_weight = float(reward_config.get("subagent_executable_weight", 0.4) or 0.4)
    no_duplicate_weight = float(reward_config.get("subagent_no_duplicate_weight", 0.2) or 0.2)
    invalid_json_penalty = float(reward_config.get("subagent_invalid_json_penalty", 0.5) or 0.5)
    invalid_action_penalty = float(reward_config.get("subagent_invalid_action_penalty", 0.5) or 0.5)
    repeated_action_penalty = float(reward_config.get("subagent_repeated_action_penalty", 0.2) or 0.2)
    min_local_reward = float(reward_config.get("subagent_min_local_reward", -1.0) or -1.0)
    max_local_reward = float(reward_config.get("subagent_max_local_reward", 1.0) or 1.0)

    if not isinstance(step, dict):
        return {
            "trainable": False,
            "local_reward": 0.0,
            "decomposition_valid": False,
            "tool_executable": False,
            "no_duplicate_probe": False,
            "invalid_json": False,
            "invalid_action": False,
            "duplicate_probe": False,
        }

    action = str(step.get("action", "")).strip()
    normalized_action = action.lower()
    statuses = set(_extract_step_statuses(step))
    invalid_json = _looks_like_invalid_json_step(step)
    invalid_action = _looks_like_invalid_action_step(step)
    finish_like = normalized_action == "finish"
    has_meaningful_action = bool(action) and normalized_action not in {"protocol_error", "unknown"}
    decomposition_valid = bool(has_meaningful_action and not invalid_json and not finish_like)
    tool_executable = bool(
        decomposition_valid
        and not invalid_action
        and any(status in {"success", "no_valid_result"} for status in statuses)
    )

    signature = _build_step_signature(step) if decomposition_valid else ""
    duplicate_probe = bool(signature and signature in seen_signatures)
    if signature and not duplicate_probe:
        seen_signatures.add(signature)
    no_duplicate_probe = bool(decomposition_valid and not duplicate_probe)

    local_reward = 0.0
    local_reward += decomposition_weight if decomposition_valid else 0.0
    local_reward += executable_weight if tool_executable else 0.0
    local_reward += no_duplicate_weight if no_duplicate_probe else 0.0
    local_reward -= invalid_json_penalty if invalid_json else 0.0
    local_reward -= invalid_action_penalty if invalid_action else 0.0
    local_reward -= repeated_action_penalty if duplicate_probe else 0.0
    local_reward *= local_reward_scale
    local_reward = max(min_local_reward, min(max_local_reward, float(local_reward)))

    return {
        "trainable": bool(decomposition_valid),
        "local_reward": local_reward,
        "decomposition_valid": bool(decomposition_valid),
        "tool_executable": bool(tool_executable),
        "no_duplicate_probe": bool(no_duplicate_probe),
        "invalid_json": bool(invalid_json),
        "invalid_action": bool(invalid_action),
        "duplicate_probe": bool(duplicate_probe),
    }


def build_video_agent_role_training_targets(
    *,
    trajectory: Any,
    generation_records: Any,
    response_length: int,
    reward_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    subagent_rl_enabled = is_subagent_rl_contribution_enabled(reward_config)
    kept_response_length = max(0, int(response_length or 0))
    planner_token_mask = [0.0] * kept_response_length
    subagent_token_mask = [0.0] * kept_response_length
    subagent_local_advantages = [0.0] * kept_response_length
    policy_loss_mask = [0.0] * kept_response_length

    step_buckets: Dict[str, List[Dict[str, Any]]] = {"planner": [], "subagent": []}
    if subagent_rl_enabled:
        for step in _iter_steps_preorder(trajectory):
            role = "planner" if int(step.get("depth", 0) or 0) <= 0 else "subagent"
            step_buckets.setdefault(role, []).append(step)

    role_cursors = {"planner": 0, "subagent": 0}
    seen_subagent_signatures: set[str] = set()
    summary = {
        "planner_generation_count": 0,
        "subagent_generation_count": 0,
        "planner_token_count": 0,
        "subagent_token_count": 0,
        "matched_subagent_steps": 0,
        "unmatched_subagent_steps": 0,
        "trainable_subagent_steps": 0,
        "positive_subagent_steps": 0,
        "negative_subagent_steps": 0,
        "subagent_local_reward_sum": 0.0,
    }

    normalized_records = generation_records if isinstance(generation_records, Sequence) else []
    for raw_record in normalized_records:
        if not isinstance(raw_record, dict):
            continue
        role = str(raw_record.get("role", "")).strip().lower()
        if role not in {"planner", "subagent"}:
            continue

        raw_start = int(raw_record.get("response_start", 0) or 0)
        raw_end = int(raw_record.get("response_end", raw_start) or raw_start)
        start = max(0, min(kept_response_length, raw_start))
        end = max(start, min(kept_response_length, raw_end))
        if end <= start:
            if role == "subagent" and subagent_rl_enabled:
                role_cursors["subagent"] += 1
                summary["unmatched_subagent_steps"] += 1
            continue

        if role == "planner":
            summary["planner_generation_count"] += 1
            summary["planner_token_count"] += end - start
            for index in range(start, end):
                planner_token_mask[index] = 1.0
                policy_loss_mask[index] = 1.0
            role_cursors["planner"] += 1
            continue

        summary["subagent_generation_count"] += 1
        summary["subagent_token_count"] += end - start
        if not subagent_rl_enabled:
            continue

        for index in range(start, end):
            subagent_token_mask[index] = 1.0

        subagent_index = role_cursors["subagent"]
        role_cursors["subagent"] += 1
        matched_step = (
            step_buckets.get("subagent", [])[subagent_index]
            if subagent_index < len(step_buckets.get("subagent", []))
            else None
        )
        reward_info = _compute_subagent_step_reward(
            matched_step,
            seen_signatures=seen_subagent_signatures,
            reward_config=reward_config,
        )

        if matched_step is not None:
            summary["matched_subagent_steps"] += 1
        else:
            summary["unmatched_subagent_steps"] += 1

        if reward_info["trainable"]:
            summary["trainable_subagent_steps"] += 1
            summary["subagent_local_reward_sum"] += float(reward_info["local_reward"])
            if float(reward_info["local_reward"]) > 0.0:
                summary["positive_subagent_steps"] += 1
            elif float(reward_info["local_reward"]) < 0.0:
                summary["negative_subagent_steps"] += 1
            for index in range(start, end):
                subagent_local_advantages[index] = float(reward_info["local_reward"])
                policy_loss_mask[index] = 1.0

    trainable_steps = int(summary["trainable_subagent_steps"])
    summary["subagent_local_reward_mean"] = (
        round(float(summary["subagent_local_reward_sum"]) / float(trainable_steps), 6)
        if trainable_steps > 0
        else 0.0
    )
    summary["planner_step_count_available"] = len(step_buckets.get("planner", []))
    summary["subagent_step_count_available"] = len(step_buckets.get("subagent", []))

    return {
        "planner_token_mask": planner_token_mask,
        "subagent_token_mask": subagent_token_mask,
        "subagent_local_advantages": subagent_local_advantages,
        "policy_loss_mask": policy_loss_mask,
        "summary": summary,
    }


def _extract_finish_step(trajectory: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(trajectory, dict):
        return None
    finish_step: Optional[Dict[str, Any]] = None
    for step in trajectory.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        if str(step.get("action", "")).strip() == "Finish":
            finish_step = step
    return finish_step


def _parse_json_object_maybe(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_evidence_status_for_metrics(value: Any = "") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"verified", "exact"}:
        return "verified"
    if normalized in {"approximate", "missing"}:
        return normalized
    return "missing"


def _coerce_logical_conflict_flag(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _extract_evidence_items_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    candidate_lists: List[Any] = []
    for key in ("Evidence", "evidence"):
        if isinstance(payload.get(key), list):
            candidate_lists.append(payload.get(key))

    finish_payload = payload.get("Finish")
    if isinstance(finish_payload, dict):
        for key in ("Evidence", "evidence"):
            if isinstance(finish_payload.get(key), list):
                candidate_lists.append(finish_payload.get(key))

    if not candidate_lists:
        return []

    raw_items = candidate_lists[0]
    evidence_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        evidence_items.append(
            {
                "label": str(item.get("label", "") or "").strip() or f"slot_{idx}",
                "evidence_status": _normalize_evidence_status_for_metrics(item.get("evidence_status", "")),
                "logical_conflict": _coerce_logical_conflict_flag(
                    item.get("logical conflict", item.get("logical_conflict"))
                ),
            }
        )
    return evidence_items


def _summarize_planner_evidence_quality(trajectory: Any) -> Dict[str, Any]:
    summary = {
        "planner_llm_iteration_count": 0,
        "planner_evidence_iteration_count": 0,
        "planner_evidence_item_count": 0,
        "planner_verified_evidence_item_count": 0,
        "planner_approximate_evidence_item_count": 0,
        "planner_missing_evidence_item_count": 0,
        "planner_non_conflicting_evidence_item_count": 0,
        "planner_conflicting_evidence_item_count": 0,
        "planner_unmarked_conflict_evidence_item_count": 0,
        "evidence_verification_score": 0.0,
        "evidence_logical_rigor_score": 0.0,
    }
    if not isinstance(trajectory, dict):
        return summary

    total_verification_ratio = 0.0
    total_logical_rigor_ratio = 0.0

    for event in trajectory.get("debug_events", []) or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("type", "")).strip() != "llm_iteration":
            continue
        if int(event.get("depth", 0) or 0) != 0:
            continue

        summary["planner_llm_iteration_count"] += 1
        payload = _parse_json_object_maybe(event.get("response"))
        evidence_items = _extract_evidence_items_from_payload(payload)
        if not evidence_items:
            continue

        summary["planner_evidence_iteration_count"] += 1
        item_count = len(evidence_items)
        summary["planner_evidence_item_count"] += item_count

        verified_count = 0
        approximate_count = 0
        missing_count = 0
        non_conflicting_count = 0
        conflicting_count = 0
        unmarked_conflict_count = 0

        for item in evidence_items:
            status = str(item.get("evidence_status", "")).strip().lower()
            if status == "verified":
                verified_count += 1
            elif status == "approximate":
                approximate_count += 1
            else:
                missing_count += 1

            conflict_flag = item.get("logical_conflict")
            if conflict_flag is False:
                non_conflicting_count += 1
            elif conflict_flag is True:
                conflicting_count += 1
            else:
                unmarked_conflict_count += 1

        summary["planner_verified_evidence_item_count"] += verified_count
        summary["planner_approximate_evidence_item_count"] += approximate_count
        summary["planner_missing_evidence_item_count"] += missing_count
        summary["planner_non_conflicting_evidence_item_count"] += non_conflicting_count
        summary["planner_conflicting_evidence_item_count"] += conflicting_count
        summary["planner_unmarked_conflict_evidence_item_count"] += unmarked_conflict_count

        total_verification_ratio += float(verified_count) / float(item_count)
        total_logical_rigor_ratio += float(non_conflicting_count) / float(item_count)

    iteration_denominator = max(1, int(summary["planner_llm_iteration_count"]))
    summary["evidence_verification_score"] = round(
        total_verification_ratio / float(iteration_denominator),
        6,
    )
    summary["evidence_logical_rigor_score"] = round(
        total_logical_rigor_ratio / float(iteration_denominator),
        6,
    )
    return summary


def summarize_video_agent_reward_features(
    *,
    run_result: Any,
    trajectory: Any,
    final_answer: Any,
    ground_truth: Any,
    options: Any,
    dataset_name: str = "",
    reward_model: Optional[Dict[str, Any]] = None,
    episode_meta: Optional[Dict[str, Any]] = None,
    reward_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = {}
    subagent_rl_enabled = is_subagent_rl_contribution_enabled(reward_config)
    if isinstance(run_result, dict):
        metrics = run_result.get("metrics", {}) if isinstance(run_result.get("metrics"), dict) else {}
    acceptance_metrics = metrics.get("acceptance_metrics", {}) if isinstance(metrics.get("acceptance_metrics"), dict) else {}
    runtime_base_meta_coverage = (
        acceptance_metrics.get("runtime_base_meta_coverage", {})
        if isinstance(acceptance_metrics.get("runtime_base_meta_coverage"), dict)
        else (
            acceptance_metrics.get("runtime_biz_meta_coverage", {})
            if isinstance(acceptance_metrics.get("runtime_biz_meta_coverage"), dict)
            else {}
        )
    )
    tool_family_counts = metrics.get("tool_family_counts", {}) if isinstance(metrics.get("tool_family_counts"), dict) else {}

    reward_model = dict(reward_model or {})
    reward_style = str(reward_model.get("style") or "rule").strip().lower()
    normalized_dataset_name = _normalize_dataset_name(
        dataset_name or reward_model.get("dataset_name") or reward_model.get("data_source")
    )
    if reward_style in {"cfvbench", "open_ended", "open_ended_keypoints"}:
        normalized_keypoints = _normalize_keypoints(
            reward_model.get("ground_truth_keypoints"),
            source_hint=str(reward_model.get("question_source") or reward_model.get("source_file") or ""),
        )
        answer_alignment = _compute_open_answer_alignment(
            final_answer=final_answer,
            ground_truth=ground_truth,
            keypoints=normalized_keypoints,
            reward_style=reward_style,
            dataset_name=normalized_dataset_name,
            reward_config=reward_config,
        )
        answer_correct = bool(answer_alignment["answer_correct"])
        predicted_label = ""
        ground_truth_labels: List[str] = []
        if reward_style == "cfvbench":
            correctness_method = "cfvbench_open_answer"
            correctness_rule = (
                "CFVBENCH answer reward uses keypoint_recall as R_ans; "
                "ROUGE-L is retained as a reporting metric only."
            )
            evaluation_mode = "cfvbench_open_answer"
        elif answer_alignment.get("answer_reward_basis") == "nextqa_wups0":
            correctness_method = "nextqa_wups"
            correctness_rule = (
                "official NExTQA-OE recommendation: compute WUPS; use WUPS@0 as answer reward basis, and report WUPS@0.9"
            )
            evaluation_mode = "nextqa_wups"
        else:
            correctness_method = "open_answer_alignment"
            correctness_rule = (
                "correct when exact match or blended answer_credit >= correct_threshold; "
                "blended answer_credit uses rouge_l_f1 and keypoint recall weights"
            )
            evaluation_mode = "open_answer_alignment"
    else:
        rule_evaluation = _evaluate_rule_answer(
            final_answer=final_answer,
            ground_truth=ground_truth,
            options=options,
            dataset_name=normalized_dataset_name,
            reward_config=reward_config,
        )
        answer_correct = bool(rule_evaluation.get("answer_correct"))
        predicted_label = str(rule_evaluation.get("predicted_label") or "")
        ground_truth_labels = list(rule_evaluation.get("ground_truth_labels") or [])
        answer_alignment = {
            "answer_credit": 1.0 if answer_correct else 0.0,
            "answer_exact_match": bool(answer_correct),
            "answer_contains_ground_truth": False,
            "answer_contained_by_ground_truth": False,
            "rouge_l_f1": 0.0,
            "rouge_weight": 0.0,
            "keypoint_weight": 0.0,
            "correct_threshold": 1.0,
            "matched_video_keypoints": 0,
            "matched_text_keypoints": 0,
            "matched_total_keypoints": 0,
            "total_video_keypoints": 0,
            "total_text_keypoints": 0,
            "total_keypoints": 0,
            "video_keypoint_recall": 0.0,
            "text_keypoint_recall": 0.0,
            "keypoint_recall": 0.0,
            "wups_0": 0.0,
            "wups_9": 0.0,
            "answer_reward_basis": "classification_accuracy",
            "option_match_method": str(rule_evaluation.get("option_match_method") or ""),
            "option_match_similarity": float(rule_evaluation.get("option_match_similarity", 0.0) or 0.0),
            "sentence_t5_model_path": str(rule_evaluation.get("sentence_t5_model_path") or ""),
        }
        correctness_method = str(rule_evaluation.get("correctness_method") or "label_or_exact_text_match")
        correctness_rule = str(rule_evaluation.get("correctness_rule") or "")
        evaluation_mode = str(rule_evaluation.get("evaluation_mode") or "accuracy")
    debug_counts = collect_debug_event_counts(
        trajectory,
        include_recursive=subagent_rl_enabled,
    )
    planner_evidence_quality = _summarize_planner_evidence_quality(trajectory)
    monitor_tool_records = _collect_tool_records(trajectory)
    monitor_successful_tool_calls = sum(
        1
        for item in monitor_tool_records
        if str(item.get("status", "")).strip() == "success"
    )
    monitor_tool_failure_count = sum(
        1
        for item in monitor_tool_records
        if str(item.get("status", "")).strip() in {"error", "no_valid_result"}
    )
    monitor_runtime_tool_count = len(monitor_tool_records)
    monitor_unique_runtime_tool_count = len({
        str(item.get("tool", "")).strip()
        for item in monitor_tool_records
        if str(item.get("tool", "")).strip()
    })

    tool_records = list(monitor_tool_records)
    if not subagent_rl_enabled:
        tool_records = [
            item for item in tool_records
            if int(item.get("depth", 0) or 0) <= 0
        ]
    successful_tool_calls = sum(1 for item in tool_records if str(item.get("status", "")).strip() == "success")
    tool_failure_count = sum(
        1
        for item in tool_records
        if str(item.get("status", "")).strip() in {"error", "no_valid_result"}
    )
    runtime_tool_count = len(tool_records)

    finish_step = _extract_finish_step(trajectory)
    finish_emitted = finish_step is not None
    chain_complete = False
    if isinstance(finish_step, dict):
        action_args = finish_step.get("action_args")
        if isinstance(action_args, dict):
            chain_complete = bool(action_args.get("chain_complete"))

    episode_meta = dict(episode_meta or {})
    elapsed_seconds = float(episode_meta.get("elapsed_seconds", 0.0) or 0.0)
    time_budget_sec = float(episode_meta.get("time_budget_sec", 0.0) or 0.0)
    planner_iterations = int(metrics.get("planner_iterations", 0) or 0)
    subagent_iterations = int(metrics.get("subagent_iterations", 0) or 0)
    monitor_planner_iterations = int(planner_iterations)
    monitor_subagent_iterations = int(subagent_iterations)
    root_step_budget = int(episode_meta.get("root_step_budget", 0) or 0)
    subagent_step_budget = int(episode_meta.get("subagent_step_budget", 0) or 0)
    if not subagent_rl_enabled:
        subagent_iterations = 0
        subagent_step_budget = 0

    budget_exceeded = "max_iterations_reached" in str(final_answer or "")
    if root_step_budget > 0 and planner_iterations >= root_step_budget and not finish_emitted:
        budget_exceeded = True
    if subagent_step_budget > 0 and subagent_iterations >= subagent_step_budget and not finish_emitted:
        budget_exceeded = True

    timeout_exceeded = time_budget_sec > 0 and elapsed_seconds > time_budget_sec
    parse_failure_count = int(debug_counts.get("parse_failure", 0))
    invalid_action_count = int(debug_counts.get("parse_failure:invalid_action_name", 0))
    format_violation_count = int(debug_counts.get("format_violation", 0))
    invalid_json_count = max(0, parse_failure_count - invalid_action_count) + int(
        debug_counts.get("protocol_violation", 0)
    )
    invalid_json_count += format_violation_count
    repeated_action_count = int(debug_counts.get("policy_violation:repeated_action_loop", 0)) + int(
        debug_counts.get("policy_violation:blocked_action_signature_reused", 0)
    )

    valid_finish = bool(finish_emitted and chain_complete and str(final_answer or "").strip() and not budget_exceeded)
    unique_runtime_tools = metrics.get("unique_tools", [])
    if not isinstance(unique_runtime_tools, list):
        unique_runtime_tools = []
    if not subagent_rl_enabled:
        unique_runtime_tool_count = len({
            str(item.get("tool", "")).strip()
            for item in tool_records
            if str(item.get("tool", "")).strip()
        })
    else:
        unique_runtime_tool_count = int(
            acceptance_metrics.get("unique_runtime_tools_count", len(unique_runtime_tools)) or len(unique_runtime_tools)
        )
    tool_backed_finish_filter_enabled = bool(reward_config.get("enable_tool_backed_finish_filter", False))
    pseudo_good_no_tool_finish = bool(
        tool_backed_finish_filter_enabled
        and reward_style == "cfvbench"
        and runtime_tool_count <= 0
        and float(answer_alignment.get("keypoint_recall", 0.0) or 0.0) < 0.5
    )
    requires_tool_backed_finish = bool(pseudo_good_no_tool_finish)
    runtime_tools_executed = runtime_tool_count > 0 or unique_runtime_tool_count > 0
    successful_runtime_tool_executed = successful_tool_calls > 0
    tool_backed_finish_satisfied = (not requires_tool_backed_finish) or successful_runtime_tool_executed
    tool_chain_valid = bool(valid_finish and tool_backed_finish_satisfied)
    acceptance_passed = bool(
        tool_chain_valid
        and invalid_json_count == 0
        and invalid_action_count == 0
    )
    base_meta_coverage_ratio = float(runtime_base_meta_coverage.get("coverage_ratio", 0.0) or 0.0)
    subagent_decomposition_rate = float(metrics.get("subagent_decomposition_rate", 0.0) or 0.0)
    parallel_concrete_batch_rate = float(metrics.get("parallel_concrete_batch_rate", 0.0) or 0.0)
    role_training_summary = (
        episode_meta.get("role_training_summary")
        if isinstance(episode_meta.get("role_training_summary"), dict)
        else {}
    )
    if not subagent_rl_enabled:
        subagent_decomposition_rate = 0.0
        parallel_concrete_batch_rate = 0.0
        role_training_summary = {
            "planner_token_count": int(role_training_summary.get("planner_token_count", 0) or 0),
        }

    return {
        "dataset_name": normalized_dataset_name,
        "reward_style": reward_style,
        "answer_correct": bool(answer_correct),
        "answer_credit": round(float(answer_alignment["answer_credit"]), 6),
        "answer_reward_basis": str(answer_alignment.get("answer_reward_basis") or "").strip(),
        "answer_exact_match": bool(answer_alignment["answer_exact_match"]),
        "answer_contains_ground_truth": bool(answer_alignment["answer_contains_ground_truth"]),
        "answer_contained_by_ground_truth": bool(answer_alignment["answer_contained_by_ground_truth"]),
        "evaluation_mode": evaluation_mode,
        "correctness_method": correctness_method,
        "correctness_rule": correctness_rule,
        "predicted_label": predicted_label,
        "ground_truth_labels": ground_truth_labels,
        "rouge_l": round(float(answer_alignment["rouge_l_f1"]), 6),
        "rouge_l_f1": round(float(answer_alignment["rouge_l_f1"]), 6),
        "open_answer_rouge_weight": round(float(answer_alignment["rouge_weight"]), 6),
        "open_answer_keypoint_weight": round(float(answer_alignment["keypoint_weight"]), 6),
        "open_answer_correct_threshold": round(float(answer_alignment["correct_threshold"]), 6),
        "matched_video_keypoints": int(answer_alignment["matched_video_keypoints"]),
        "matched_text_keypoints": int(answer_alignment["matched_text_keypoints"]),
        "matched_total_keypoints": int(answer_alignment["matched_total_keypoints"]),
        "total_video_keypoints": int(answer_alignment["total_video_keypoints"]),
        "total_text_keypoints": int(answer_alignment["total_text_keypoints"]),
        "total_keypoints": int(answer_alignment["total_keypoints"]),
        "video_keypoint_recall": round(float(answer_alignment["video_keypoint_recall"]), 6),
        "text_keypoint_recall": round(float(answer_alignment["text_keypoint_recall"]), 6),
        "keypoint_recall": round(float(answer_alignment["keypoint_recall"]), 6),
        "wups_0": round(float(answer_alignment.get("wups_0", 0.0) or 0.0), 6),
        "wups_9": round(float(answer_alignment.get("wups_9", 0.0) or 0.0), 6),
        "option_match_method": str(answer_alignment.get("option_match_method") or "").strip(),
        "option_match_similarity": round(float(answer_alignment.get("option_match_similarity", 0.0) or 0.0), 6),
        "sentence_t5_model_path": str(answer_alignment.get("sentence_t5_model_path") or "").strip(),
        "tool_backed_finish_filter_enabled": bool(tool_backed_finish_filter_enabled),
        "finish_emitted": bool(finish_emitted),
        "finish_valid": bool(valid_finish),
        "chain_complete": bool(chain_complete),
        "evidence_verification_score": round(float(planner_evidence_quality["evidence_verification_score"]), 6),
        "evidence_logical_rigor_score": round(float(planner_evidence_quality["evidence_logical_rigor_score"]), 6),
        "planner_llm_iteration_count": int(planner_evidence_quality["planner_llm_iteration_count"]),
        "planner_evidence_iteration_count": int(planner_evidence_quality["planner_evidence_iteration_count"]),
        "planner_evidence_item_count": int(planner_evidence_quality["planner_evidence_item_count"]),
        "planner_verified_evidence_item_count": int(planner_evidence_quality["planner_verified_evidence_item_count"]),
        "planner_approximate_evidence_item_count": int(planner_evidence_quality["planner_approximate_evidence_item_count"]),
        "planner_missing_evidence_item_count": int(planner_evidence_quality["planner_missing_evidence_item_count"]),
        "planner_non_conflicting_evidence_item_count": int(planner_evidence_quality["planner_non_conflicting_evidence_item_count"]),
        "planner_conflicting_evidence_item_count": int(planner_evidence_quality["planner_conflicting_evidence_item_count"]),
        "planner_unmarked_conflict_evidence_item_count": int(planner_evidence_quality["planner_unmarked_conflict_evidence_item_count"]),
        "requires_tool_backed_finish": bool(requires_tool_backed_finish),
        "pseudo_good_no_tool_finish": bool(pseudo_good_no_tool_finish),
        "runtime_tools_executed": bool(runtime_tools_executed),
        "successful_runtime_tool_executed": bool(successful_runtime_tool_executed),
        "tool_backed_finish_satisfied": bool(tool_backed_finish_satisfied),
        "tool_chain_valid": bool(tool_chain_valid),
        "acceptance_passed": bool(acceptance_passed),
        "successful_tool_calls": int(successful_tool_calls),
        "tool_failure_count": int(tool_failure_count),
        "invalid_json_count": int(invalid_json_count),
        "format_violation_count": int(format_violation_count),
        "invalid_action_count": int(invalid_action_count),
        "repeated_action_count": int(repeated_action_count),
        "budget_exceeded": bool(budget_exceeded),
        "timeout_exceeded": bool(timeout_exceeded),
        "planner_iterations": planner_iterations,
        "subagent_iterations": subagent_iterations,
        "unique_runtime_tool_count": unique_runtime_tool_count,
        "runtime_tool_count": runtime_tool_count,
        "monitor_successful_tool_calls": int(monitor_successful_tool_calls),
        "monitor_tool_failure_count": int(monitor_tool_failure_count),
        "monitor_runtime_tool_count": int(monitor_runtime_tool_count),
        "monitor_unique_runtime_tool_count": int(monitor_unique_runtime_tool_count),
        "monitor_planner_iterations": int(monitor_planner_iterations),
        "monitor_subagent_iterations": int(monitor_subagent_iterations),
        "base_meta_coverage_ratio": round(base_meta_coverage_ratio, 6),
        "subagent_decomposition_rate": round(subagent_decomposition_rate, 6),
        "parallel_concrete_batch_rate": round(parallel_concrete_batch_rate, 6),
        "planner_training_token_count": int(role_training_summary.get("planner_token_count", 0) or 0),
        "subagent_training_token_count": int(role_training_summary.get("subagent_token_count", 0) or 0),
        "trainable_subagent_steps": int(role_training_summary.get("trainable_subagent_steps", 0) or 0),
        "subagent_local_reward_mean": round(
            float(role_training_summary.get("subagent_local_reward_mean", 0.0) or 0.0),
            6,
        ),
        "retrieval_tool_calls": int(tool_family_counts.get("retrieval", 0) or 0),
        "visual_tool_calls": int(tool_family_counts.get("visual", 0) or 0),
        "compute_tool_calls": int(tool_family_counts.get("compute", 0) or 0),
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def compute_video_agent_reward(
    *,
    run_result: Any,
    trajectory: Any,
    final_answer: Any,
    ground_truth: Any,
    options: Any,
    dataset_name: str = "",
    reward_model: Optional[Dict[str, Any]] = None,
    episode_meta: Optional[Dict[str, Any]] = None,
    reward_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reward_config = dict(reward_config or {})
    features = summarize_video_agent_reward_features(
        run_result=run_result,
        trajectory=trajectory,
        final_answer=final_answer,
        ground_truth=ground_truth,
        options=options,
        dataset_name=dataset_name,
        reward_model=reward_model,
        episode_meta=episode_meta,
        reward_config=reward_config,
    )

    score = 0.0
    components: Dict[str, float] = {}

    answer_reward = float(reward_config.get("answer_reward", 1.0))
    valid_bonus = float(reward_config.get("valid_bonus", 0.2))
    successful_tool_bonus = float(reward_config.get("successful_tool_bonus", 0.1))
    invalid_json_penalty = float(reward_config.get("invalid_json_penalty", 0.5))
    invalid_action_penalty = float(reward_config.get("invalid_action_penalty", 0.5))
    repeated_action_penalty = float(reward_config.get("repeated_action_penalty", 0.2))
    budget_penalty = float(reward_config.get("budget_penalty", 0.2))
    timeout_penalty = float(reward_config.get("timeout_penalty", 0.5))
    diversity_bonus = float(reward_config.get("diversity_bonus", 0.0))
    diversity_target_unique_tools = max(1, int(reward_config.get("diversity_target_unique_tools", 4) or 4))
    base_meta_bonus = float(reward_config.get("base_meta_bonus", reward_config.get("biz_meta_bonus", 0.0)))
    decomposition_bonus = float(reward_config.get("decomposition_bonus", 0.0))
    parallel_bonus = float(reward_config.get("parallel_bonus", 0.0))
    on_budget_bonus = float(reward_config.get("on_budget_bonus", 0.0))
    tool_chain_invalid_penalty = float(reward_config.get("tool_chain_invalid_penalty", 0.0))
    min_reward = float(reward_config.get("min_reward", -1.0))
    max_reward = float(reward_config.get("max_reward", 1.5))

    answer_credit = max(0.0, min(1.0, float(features.get("answer_credit", 1.0 if features["answer_correct"] else 0.0))))
    requires_tool_backed_finish = bool(features.get("requires_tool_backed_finish"))
    has_successful_runtime_tool = bool(features.get("successful_runtime_tool_executed"))
    tool_chain_valid = bool(features.get("tool_chain_valid"))
    answer_reward_total = answer_reward * answer_credit
    if requires_tool_backed_finish and not bool(features.get("tool_backed_finish_satisfied")):
        answer_reward_total = 0.0
    score += answer_reward_total
    components["answer_reward"] = answer_reward_total

    if (
        features["finish_valid"]
        and features["invalid_json_count"] == 0
        and features["invalid_action_count"] == 0
        and (not requires_tool_backed_finish or has_successful_runtime_tool)
    ):
        score += valid_bonus
        components["valid_bonus"] = valid_bonus
    else:
        components["valid_bonus"] = 0.0

    if features["successful_tool_calls"] > 0:
        score += successful_tool_bonus
        components["successful_tool_bonus"] = successful_tool_bonus
    else:
        components["successful_tool_bonus"] = 0.0

    diversity_bonus_total = diversity_bonus * min(
        1.0,
        float(features["unique_runtime_tool_count"]) / float(diversity_target_unique_tools),
    )
    base_meta_bonus_total = base_meta_bonus * float(features.get("base_meta_coverage_ratio", 0.0) or 0.0)
    decomposition_bonus_total = decomposition_bonus * float(features["subagent_decomposition_rate"])
    parallel_bonus_total = parallel_bonus * float(features["parallel_concrete_batch_rate"])
    on_budget_bonus_total = (
        on_budget_bonus
        if (
            not features["timeout_exceeded"]
            and not features["budget_exceeded"]
            and (not requires_tool_backed_finish or has_successful_runtime_tool)
        )
        else 0.0
    )

    score += diversity_bonus_total
    score += base_meta_bonus_total
    score += decomposition_bonus_total
    score += parallel_bonus_total
    score += on_budget_bonus_total

    components["diversity_bonus"] = diversity_bonus_total
    components["base_meta_bonus"] = base_meta_bonus_total
    components["decomposition_bonus"] = decomposition_bonus_total
    components["parallel_bonus"] = parallel_bonus_total
    components["on_budget_bonus"] = on_budget_bonus_total

    json_penalty_total = invalid_json_penalty * max(0, features["invalid_json_count"])
    action_penalty_total = invalid_action_penalty * max(0, features["invalid_action_count"])
    repeated_penalty_total = repeated_action_penalty * max(0, features["repeated_action_count"])
    budget_penalty_total = budget_penalty if features["budget_exceeded"] else 0.0
    timeout_penalty_total = timeout_penalty if features["timeout_exceeded"] else 0.0

    score -= json_penalty_total
    score -= action_penalty_total
    score -= repeated_penalty_total
    score -= budget_penalty_total
    score -= timeout_penalty_total
    tool_chain_invalid_penalty_total = (
        tool_chain_invalid_penalty
        if requires_tool_backed_finish and not tool_chain_valid
        else 0.0
    )
    score -= tool_chain_invalid_penalty_total

    components["invalid_json_penalty"] = -json_penalty_total
    components["invalid_action_penalty"] = -action_penalty_total
    components["repeated_action_penalty"] = -repeated_penalty_total
    components["budget_penalty"] = -budget_penalty_total
    components["timeout_penalty"] = -timeout_penalty_total
    components["tool_chain_invalid_penalty"] = -tool_chain_invalid_penalty_total

    score = max(min_reward, min(max_reward, score))
    return {
        "score": float(score),
        "components": components,
        "features": features,
    }
