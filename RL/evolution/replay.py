"""Shared replay-gate calculations for Evolver candidate evaluations.

The functions in this module only compare already materialized reports.  A
caller must separately prove that the candidate was applied to a fresh runtime
and that the candidate runtime produced a new trajectory.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping


_REPLAY_GROUPS = ("trigger", "neighbor", "regression", "held_out")
# A candidate may be compared with the trigger case, but that is not enough
# evidence for a general mutation.  These are structural minimums, not a
# claim of statistical significance.  They prevent an empty group from being
# silently treated as a passing gate.
_DEFAULT_MINIMUM_GROUP_COUNTS = {
    "trigger": 1,
    "neighbor": 1,
    "regression": 1,
    "held_out": 1,
}


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _minimum_group_counts(gates: Mapping[str, Any]) -> Dict[str, int]:
    configured = gates.get("minimum_group_counts")
    configured = configured if isinstance(configured, Mapping) else {}
    result: Dict[str, int] = {}
    for group in _REPLAY_GROUPS:
        raw_value = configured.get(group, _DEFAULT_MINIMUM_GROUP_COUNTS[group])
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = _DEFAULT_MINIMUM_GROUP_COUNTS[group]
        # Every formal replay must contain at least one case per group. A
        # zero/negative configuration is malformed and must not disable a
        # coverage gate at runtime; the audit applies the same normalization.
        result[group] = max(1, value)
    return result


def evaluate_replay(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the configured acceptance gates to baseline/current reports.

    This function deliberately does not label a replay as real.  The caller's
    report must contain evidence of a fresh candidate inference and a changed
    harness policy before using the returned status as an Evolver result.
    """

    gates = dict(config.get("acceptance") or {})
    minimum_group_counts = _minimum_group_counts(gates)
    input_errors: List[str] = []

    def _score_map(payload: Mapping[str, Any], label: str) -> Dict[str, float]:
        if not isinstance(payload, Mapping):
            input_errors.append(f"{label}.not_mapping")
            return {}
        raw_items = payload.get("verifications")
        if not isinstance(raw_items, list):
            input_errors.append(f"{label}.verifications_missing_or_not_list")
            return {}
        scores: Dict[str, float] = {}
        for index, item in enumerate(raw_items):
            if not isinstance(item, Mapping):
                input_errors.append(f"{label}.verifications[{index}]_not_mapping")
                continue
            uid = str(item.get("sample_uid") or "").strip()
            if not uid:
                input_errors.append(f"{label}.verifications[{index}]_missing_sample_uid")
                continue
            if uid in scores:
                input_errors.append(f"{label}.duplicate_sample_uid:{uid}")
                continue
            raw_score = item.get("answer_score")
            if raw_score in (None, ""):
                input_errors.append(f"{label}.missing_answer_score:{uid}")
                continue
            if isinstance(raw_score, bool):
                input_errors.append(f"{label}.invalid_answer_score:{uid}")
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                input_errors.append(f"{label}.invalid_answer_score:{uid}")
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                input_errors.append(f"{label}.answer_score_out_of_range:{uid}")
                continue
            scores[uid] = score
        return scores

    def _metrics_valid(payload: Mapping[str, Any], label: str) -> bool:
        if not isinstance(payload, Mapping):
            input_errors.append(f"{label}.not_mapping")
            return False
        metrics = payload.get("batch_metrics")
        if not isinstance(metrics, Mapping):
            input_errors.append(f"{label}.batch_metrics_missing_or_not_mapping")
            return False
        valid = True
        for key in ("accuracy", "invalid_call_rate", "mean_latency_seconds"):
            raw_value = metrics.get(key)
            if isinstance(raw_value, bool):
                input_errors.append(f"{label}.batch_metrics_missing_or_invalid:{key}")
                valid = False
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                input_errors.append(f"{label}.batch_metrics_missing_or_invalid:{key}")
                valid = False
                continue
            if not math.isfinite(value) or value < 0.0:
                input_errors.append(f"{label}.batch_metrics_out_of_range:{key}")
                valid = False
            if key in {"accuracy", "invalid_call_rate"} and value > 1.0:
                input_errors.append(f"{label}.batch_metrics_out_of_range:{key}")
                valid = False
        return valid

    baseline_scores = _score_map(baseline, "baseline")
    current_scores = _score_map(current, "current")
    baseline_metrics_valid = _metrics_valid(baseline, "baseline")
    current_metrics_valid = _metrics_valid(current, "current")
    trajectory_to_uid: Dict[str, str] = {}
    baseline_verifications = (
        baseline.get("verifications")
        if isinstance(baseline, Mapping)
        else None
    )
    for index, item in enumerate(baseline_verifications or []):
        if not isinstance(item, Mapping):
            continue
        trajectory_id = str(item.get("trajectory_id") or "").strip()
        sample_uid = str(item.get("sample_uid") or "").strip()
        if not trajectory_id or not sample_uid:
            input_errors.append(f"baseline.verifications[{index}]_missing_trajectory_id")
            continue
        if trajectory_id in trajectory_to_uid and trajectory_to_uid[trajectory_id] != sample_uid:
            input_errors.append(f"baseline.duplicate_trajectory_id:{trajectory_id}")
            continue
        trajectory_to_uid[trajectory_id] = sample_uid

    group_deltas: Dict[str, float] = {}
    group_counts: Dict[str, int] = {}
    group_baselines: Dict[str, float] = {}
    replay_groups = baseline.get("replay_groups") if isinstance(baseline, Mapping) else {}
    replay_groups = replay_groups if isinstance(replay_groups, Mapping) else {}
    for group, trajectory_ids in replay_groups.items():
        if not isinstance(trajectory_ids, list):
            input_errors.append(f"baseline.replay_groups_not_list:{group}")
            trajectory_ids = []
        uids = [trajectory_to_uid.get(str(item), "") for item in trajectory_ids]
        uids = [uid for uid in uids if uid in baseline_scores and uid in current_scores]
        group_counts[str(group)] = len(uids)
        group_baselines[str(group)] = _mean([baseline_scores[uid] for uid in uids]) if uids else 0.0
        group_deltas[str(group)] = round(
            _mean([current_scores[uid] for uid in uids])
            - _mean([baseline_scores[uid] for uid in uids]),
            6,
        ) if uids else 0.0

    baseline_uids = set(baseline_scores)
    current_uids = set(current_scores)
    paired_uids = baseline_uids & current_uids
    pairing_valid = (
        bool(baseline_uids)
        and baseline_uids == current_uids
        and not input_errors
    )
    coverage_checks = {
        group: group_counts.get(group, 0) >= minimum_group_counts[group]
        for group in _REPLAY_GROUPS
    }
    coverage_valid = all(coverage_checks.values())

    baseline_metrics = (
        dict(baseline.get("batch_metrics"))
        if isinstance(baseline, Mapping) and isinstance(baseline.get("batch_metrics"), Mapping)
        else {}
    )
    current_metrics = (
        dict(current.get("batch_metrics"))
        if isinstance(current, Mapping) and isinstance(current.get("batch_metrics"), Mapping)
        else {}
    )

    def _metric_number(metrics: Mapping[str, Any], key: str) -> float:
        raw_value = metrics.get(key)
        if isinstance(raw_value, bool):
            return 0.0
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    current_accuracy = _metric_number(current_metrics, "accuracy")
    baseline_accuracy = _metric_number(baseline_metrics, "accuracy")
    current_invalid_rate = _metric_number(current_metrics, "invalid_call_rate")
    baseline_invalid_rate = _metric_number(baseline_metrics, "invalid_call_rate")
    baseline_latency = _metric_number(baseline_metrics, "mean_latency_seconds")
    current_latency = _metric_number(current_metrics, "mean_latency_seconds")
    overall_gain = current_accuracy - baseline_accuracy
    invalid_increase = current_invalid_rate - baseline_invalid_rate
    latency_increase = (
        (current_latency - baseline_latency) / baseline_latency
        if baseline_latency > 0
        else 0.0
    )
    latency_limit = float(
        gates.get("latency_increase_with_answer_gain", 0.20)
        if overall_gain > 0
        else gates.get("latency_increase", 0.10)
    )
    positive_answer_gain = overall_gain > 0.0 or any(
        group_counts.get(group, 0) >= minimum_group_counts[group]
        and group_deltas.get(group, 0.0) > 0.0
        for group in ("trigger", "neighbor")
    )
    latency_improvement = baseline_latency > 0.0 and (
        current_latency <= baseline_latency * (1 - float(gates.get("minimum_latency_gain", 0.05)))
    )
    utility_gain_proven = positive_answer_gain or latency_improvement
    require_positive_utility_gain = bool(
        gates.get("require_positive_utility_gain", True)
    )
    checks = {
        "verification_schema": not input_errors,
        "batch_metrics_complete": baseline_metrics_valid and current_metrics_valid,
        "paired_sample_set": pairing_valid,
        "overall_answer_non_decreasing": overall_gain >= 0.0,
        "trigger_utility_gain": group_counts.get("trigger", 0) == 0
        or group_deltas.get("trigger", 0.0) >= float(gates.get("trigger_utility_gain", 0.0)),
        "neighbor_utility_gain": group_counts.get("neighbor", 0) == 0
        or group_deltas.get("neighbor", 0.0) >= float(gates.get("neighbor_nonregression_floor", 0.0)),
        "regression_answer_drop": group_counts.get("regression", 0) == 0
        or -group_deltas.get("regression", 0.0) <= float(gates.get("regression_answer_drop", 0.01)),
        "held_out_utility_drop": group_counts.get("held_out", 0) == 0
        or -group_deltas.get("held_out", 0.0) <= float(gates.get("held_out_utility_drop", 0.0)),
        "invalid_call_rate_increase": invalid_increase
        <= float(gates.get("invalid_call_rate_increase", 0.005)),
        "latency_increase": latency_increase <= latency_limit,
        # A non-regressing no-op is not Evolver utility.  A candidate must
        # improve answer quality on the batch/trigger/neighbor or reduce
        # latency while preserving the answer gate.
        "positive_utility_gain": (
            not require_positive_utility_gain or utility_gain_proven
        ),
    }
    if input_errors or not pairing_valid:
        metric_gate_status = "invalid_pairing"
    elif not coverage_valid:
        metric_gate_status = "insufficient_coverage"
    else:
        metric_gate_status = "accepted" if all(checks.values()) else "rejected"
    return {
        # ``accepted`` here means only that the numeric replay gates pass.
        # Fresh candidate inference is intentionally proved by mix_replay.py,
        # not by this report-comparison helper.
        "status": metric_gate_status,
        "metric_gate_status": metric_gate_status,
        "acceptance_scope": "metrics_only",
        "fresh_candidate_inference_proven": False,
        "checks": checks,
        "coverage_checks": coverage_checks,
        "coverage_valid": coverage_valid,
        "minimum_group_counts": minimum_group_counts,
        "group_utility_deltas": group_deltas,
        "group_counts": group_counts,
        "paired_sample_count": len(paired_uids),
        "baseline_sample_count": len(baseline_uids),
        "current_sample_count": len(current_uids),
        "overall_answer_gain": round(overall_gain, 6),
        "invalid_call_rate_increase": round(invalid_increase, 6),
        "latency_increase": round(latency_increase, 6),
        "latency_limit": round(latency_limit, 6),
        "require_positive_utility_gain": require_positive_utility_gain,
        "positive_answer_gain": positive_answer_gain,
        "latency_improvement": latency_improvement,
        "utility_gain_proven": utility_gain_proven,
        "input_errors": sorted(set(input_errors)),
    }
