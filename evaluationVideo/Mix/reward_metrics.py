from __future__ import annotations

import re
from core.answer_contract import normalize_answer_format
from typing import Any, Dict, Iterable, List, Mapping


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_dataset_name(dataset_name: Any) -> str:
    return str(dataset_name or "").strip().lower()


def _strip_option_label_prefix(text: Any, label: Any) -> str:
    """Remove a duplicated leading option label from option text.

    Dataset loaders expose options in both forms ``"Berries"`` and
    ``"C. Berries."``.  The label is already stored separately, so keeping
    the latter prefix makes a valid answer ``"(C) Berries"`` fail the
    label/text consistency check.
    """
    value = str(text or "").strip()
    normalized_label = str(label or "").strip().upper()
    if not value or not re.fullmatch(r"[A-Z]", normalized_label):
        return value
    prefix = rf"^\s*\(?{re.escape(normalized_label)}\)?\s*[.):\-]\s*"
    return re.sub(prefix, "", value, count=1, flags=re.IGNORECASE).strip()


def _coerce_options(options: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if isinstance(options, dict):
        for label, text in options.items():
            normalized_label = str(label).strip()
            normalized.append({
                "label": normalized_label,
                "text": _strip_option_label_prefix(text, normalized_label),
            })
        return normalized
    if isinstance(options, (list, tuple)):
        for index, item in enumerate(options):
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("option") or chr(ord("A") + index)).strip()
                text = str(item.get("text") or item.get("value") or item.get("content") or "").strip()
            else:
                label = chr(ord("A") + index)
                text = str(item).strip()
            normalized.append({"label": label, "text": _strip_option_label_prefix(text, label)})
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
        # Dataset annotations may use a numeric option *text* (for example
        # "1") as well as numeric zero-/one-based indices.  The normalized
        # option text is the unambiguous interpretation used by the dataset
        # loader, so do not add index-based alternatives here.
        return [option_text_to_label[normalized]]

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
    answer_text = normalize_answer_format(final_answer)
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


def _answer_text_matches_option(final_answer: Any, predicted_label: str, options: Any) -> bool:
    """Require the answer text to agree with its selected label.

    A known label alone identifies its listed option. If text is supplied,
    it must agree: ``(D) 7`` cannot be credited when D actually means ``1``.
    """

    label = str(predicted_label or "").strip().upper()
    if not label:
        return False
    normalized_options = _coerce_options(options)
    selected = next(
        (item for item in normalized_options if str(item.get("label") or "").strip().upper() == label),
        None,
    )
    if selected is None:
        return False
    answer = normalize_answer_format(final_answer, {item['label']: item['text'] for item in normalized_options})
    match = re.search(r"\(([A-Z])\)\s*(.*?)\s*$", answer, re.IGNORECASE | re.DOTALL)
    answer_text = str(match.group(2) if match and match.group(1).upper() == label else "").strip()
    option_text = str(selected.get("text") or "").strip()
    if not answer_text or not option_text:
        return False
    option_tokens = set(re.findall(r"[a-z0-9]+", _normalize_text(option_text)))
    answer_tokens = set(re.findall(r"[a-z0-9]+", _normalize_text(answer_text)))
    return bool(option_tokens) and option_tokens.issubset(answer_tokens)


def _evaluate_rule_answer(
    final_answer: Any,
    ground_truth: Any,
    options: Any,
    *,
    dataset_name: str = "",
) -> Dict[str, Any]:
    del dataset_name
    predicted_label = extract_predicted_label(final_answer, options)
    ground_truth_labels = resolve_ground_truth_labels(ground_truth, options)
    label_matched = bool(
        predicted_label
        and predicted_label in ground_truth_labels
        and _answer_text_matches_option(final_answer, predicted_label, options)
    )
    normalized_answer = _normalize_text(final_answer)
    exact_text_match = bool(normalized_answer and normalized_answer == _normalize_text(ground_truth))
    answer_correct = bool(label_matched or exact_text_match)
    return {
        "answer_correct": bool(answer_correct),
        "predicted_label": str(predicted_label or "").strip(),
        "ground_truth_labels": list(ground_truth_labels),
    }


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


def _infer_observation_item_status(item: Dict[str, Any]) -> str:
    explicit = str(item.get("status", "")).strip().lower()
    if explicit:
        return explicit
    result = _normalize_text(item.get("result"))
    if result.startswith(("error", "failed", "no_valid_result")):
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
                {**item, "status": _infer_observation_item_status(item), "depth": depth}
                for item in _iter_nested_observation_items(observation_items)
                if str(item.get("tool", "")).strip() not in {"", "Parallel", "System", "Assistant_Note"}
            ]
            records.extend(nested_records)
            continue

        status = _infer_step_status(step)
        records.append({"tool": action, "status": status, "depth": depth})
    return records


def summarize_mix_reward_features(
    *,
    run_result: Mapping[str, Any],
    trajectory: Any,
    final_answer: Any,
    ground_truth: Any,
    options: Any,
    dataset_name: str = "",
) -> Dict[str, Any]:
    metrics = run_result.get("metrics", {}) if isinstance(run_result, dict) else {}
    normalized_dataset_name = _normalize_dataset_name(dataset_name)
    rule_evaluation = _evaluate_rule_answer(
        final_answer=final_answer,
        ground_truth=ground_truth,
        options=options,
        dataset_name=normalized_dataset_name,
    )
    monitor_tool_records = _collect_tool_records(trajectory)
    monitor_successful_tool_calls = sum(
        1 for item in monitor_tool_records if str(item.get("status", "")).strip() == "success"
    )
    return {
        "answer_correct": bool(rule_evaluation.get("answer_correct", False)),
        "predicted_label": str(rule_evaluation.get("predicted_label") or ""),
        "ground_truth_labels": list(rule_evaluation.get("ground_truth_labels") or []),
        "monitor_runtime_tool_count": int(len(monitor_tool_records)),
        "monitor_successful_tool_calls": int(monitor_successful_tool_calls),
        "monitor_planner_iterations": int(metrics.get("planner_iterations", 0) or 0),
        "monitor_subagent_iterations": int(metrics.get("subagent_iterations", 0) or 0),
    }
