"""Build auditable selective/suffix replay plans for Evolver candidates.

This module deliberately plans a replay; it does not claim that the suffix was
executed.  A plan is executable only when the corresponding runtime hook is
present.  The expensive worker path and the deterministic verifier path can
consume the same plan without silently turning a cached trajectory into a new
inference.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Tuple


_FORBIDDEN_KEYS = {
    "acc",
    "accuracy",
    "annotation",
    "evaluator",
    "ground_truth",
    "groundtruth",
    "ground_truth_labels",
    "ground_truth_option",
    "correct_label",
    "reference_answer",
    "gold_answer",
    "official_meta",
    "predicted_label",
    "raw_final_answer",
    "time_reference",
    "answer",
}


def _key_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _redact(value: Any, path: str = "") -> Tuple[Any, List[str]]:
    """Copy a trajectory state while removing evaluation-only fields."""

    excluded: List[str] = []
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if _key_token(key) in _FORBIDDEN_KEYS:
                excluded.append(child_path)
                continue
            clean, child_excluded = _redact(raw_value, child_path)
            result[key] = clean
            excluded.extend(child_excluded)
        return result, excluded
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            clean, child_excluded = _redact(item, f"{path}[{index}]")
            result.append(clean)
            excluded.extend(child_excluded)
        return result, excluded
    return copy.deepcopy(value), excluded


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _step_records(steps: Iterable[Any]) -> Iterable[Tuple[int, str, Mapping[str, Any]]]:
    for index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, Mapping):
            continue
        action = str(raw_step.get("action") or "").strip()
        yield index, action, raw_step


def _parallel_actions(step: Mapping[str, Any]) -> List[str]:
    actions: List[str] = []
    for item in step.get("observation_items") or []:
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("tool") or item.get("action") or "").strip()
        if action and action not in {"Parallel", "System", "Assistant_Note"}:
            actions.append(action)
    return actions


def _nested_steps(step: Mapping[str, Any]):
    children = list(step.get("sub_trajectories") or [])
    if step.get("sub_trajectory"):
        children.append(step["sub_trajectory"])
    for child in children:
        for item in child.get("steps") or []:
            yield item
            yield from _nested_steps(item)


def _tool_failed(step: Mapping[str, Any]) -> bool:
    status_values = []
    for item in step.get("observation_items") or []:
        if isinstance(item, Mapping):
            status_values.append(str(item.get("status") or "").strip().lower())
    observation = str(step.get("observation") or "").strip().lower()
    return any(status in {"error", "no_valid_result", "failed"} for status in status_values) or observation.startswith((
        "error",
        "failed",
        "no_valid_result",
    ))


def _operations(snapshot: Mapping[str, Any]) -> List[Tuple[str, str, Mapping[str, Any]]]:
    operations: List[Tuple[str, str, Mapping[str, Any]]] = []
    for target, effect in (snapshot.get("source_overrides") or {}).items():
        operations.append(("edit_tool_source", str(target), effect))
    tool_overrides = snapshot.get("tool_overrides")
    if isinstance(tool_overrides, Mapping) and tool_overrides:
        for raw_target, effect in sorted(tool_overrides.items(), key=lambda item: str(item[0])):
            if isinstance(effect, Mapping):
                operations.append(("update_tool_schema", str(raw_target), effect))
    policy = snapshot.get("router_policy")
    policy = policy if isinstance(policy, Mapping) else {}
    if isinstance(policy.get("update_recovery_policy"), Mapping):
        operations.append((
            "update_recovery_policy",
            "low_signal_temporal_grounding",
            policy["update_recovery_policy"],
        ))
    if isinstance(policy.get("add_composition_edge"), Mapping):
        operations.append(("add_composition_edge", "candidate_comparison", policy["add_composition_edge"]))
    return operations


def _find_cut(steps: List[Mapping[str, Any]], action: str, target: str, effect=None) -> Tuple[int, str, bool]:
    """Return one-based cut step, reason and whether the mutation was triggered."""

    finish_step = next((index for index, name, _ in _step_records(steps) if name == "Finish"), None)
    if action == "edit_tool_source":
        return 1, "source/schema discovery may affect routing from the first turn; paired runtime required", True
    if action == "add_composition_edge":
        if finish_step is None:
            return len(steps) + 1, "no Finish step; verifier suffix has no terminal state", False
        return finish_step, "composition edge is consumed by the Finish verifier", True

    if action == "update_tool_schema":
        return 1, "tool schema is visible to routing from the first turn", True

    if action == "update_recovery_policy":
        if (effect or {}).get('enabled') is False:
            # The separate preflight requires a historically executed route.
            # A disabled candidate has no activation point of its own.
            return 1, "paired replay from start to prove recovery suppression", True
        from .intervention import recovery_activation_step
        cut = recovery_activation_step(steps, effect or {})
        if cut is None:
            return len(steps) + 1, "recovery retrieval threshold/dispatch not reached", False
        return cut, "resume before the eligible retrieval dispatch", True

    return finish_step or (len(steps) + 1), "unsupported mutation has no replay cut", False


def _safe_prefix_step(step: Mapping[str, Any]) -> Dict[str, Any]:
    # ``Finish.action_args.answer`` is intentionally not part of a reusable
    # prefix.  The answer may be used by a separate verifier evaluation, never
    # to select a replay window or a tool call.
    allowed = {
        "step_id",
        "depth",
        "thought",
        "plan",
        "action",
        "action_args",
        "observation",
        "observation_items",
        "timestamp",
        "is_recursive",
    }
    projected = {key: step[key] for key in allowed if key in step and step.get("action") != "Finish"}
    clean, _ = _redact(projected)
    return clean if isinstance(clean, dict) else {}


def build_selective_suffix_plan(
    run_result: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create a selective replay plan without executing any model/tool call."""

    agent = run_result.get("agent") if isinstance(run_result.get("agent"), Mapping) else {}
    trajectory = agent.get("trajectory") if isinstance(agent.get("trajectory"), Mapping) else {}
    raw_steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    steps = [step for step in raw_steps if isinstance(step, Mapping)]
    delta = snapshot.get("mutation_delta") or snapshot
    operations = _operations(delta)
    cut_records = [
        (*_find_cut(steps, action, target, effect), action, target, effect)
        for action, target, effect in operations
    ]
    from .intervention import suppressible_recovery_routes
    for index, record in enumerate(cut_records):
        if record[3] == "update_recovery_policy" and suppressible_recovery_routes(trajectory, record[5] or {}):
            cut_records[index] = (1, "paired replay from start to verify reduced recovery", True, *record[3:])
    for index, record in enumerate(cut_records):
        if record[3] == "edit_tool_source" and record[4] in delta.get("source_local_targets", []):
            target = record[4]
            first_call = next((i for i, name, step in _step_records(steps)
                               if target in [name, *_parallel_actions(step)]), None)
            # Delegated calls are also affected: a root step can hold an entire
            # child trajectory. Cut before that root step, never inside it.
            if first_call is None:
                first_call = next((i for i, _, step in _step_records(steps)
                                   if any(n.get("action") == target for n in _nested_steps(step))), None)
            cut_records[index] = (first_call or 1, "run-body-only edit; restore before the first containing root call",
                                  True, *record[3:])
    if cut_records:
        cut_step = min(record[0] for record in cut_records)
        triggered = any(bool(record[2]) for record in cut_records)
        cut_reason = "; ".join(
            f"{record[3]}:{record[4]}: {record[1]}" for record in cut_records
        )
    else:
        cut_step, cut_reason, triggered = len(steps) + 1, "snapshot has no executable operation", False
    cut_offset = max(0, min(len(steps), cut_step - 1))
    prefix = [_safe_prefix_step(step) for step in steps[:cut_offset]]
    suffix = [
        {
            "step_id": step.get("step_id", index),
            "action": str(step.get("action") or ""),
        }
        for index, step in enumerate(steps[cut_offset:], start=cut_step)
        if str(step.get("action") or "") != "Finish"
    ]
    if any(record[3] == "update_recovery_policy" and record[2] for record in cut_records):
        suffix.insert(0, {"action": "Inspect_Frame", "reason": "candidate recovery hook"})

    runtime_required = any(
        record[3] in {"update_tool_schema", "update_recovery_policy", "edit_tool_source"} and record[2]
        for record in cut_records
    )
    replay_kind = (
        "coordinated_suffix"
        if len(operations) > 1
        else "verifier_suffix"
        if operations and operations[0][0] == "add_composition_edge"
        else "runtime_suffix"
    )
    if not triggered:
        causal_status = "not_triggered"
    elif runtime_required:
        causal_status = "runtime_suffix_planned"
    else:
        causal_status = "verifier_suffix_ready"

    state = {
        "trajectory_id": str(trajectory.get("trajectory_id") or ""),
        "task_id": str(trajectory.get("task_id") or ""),
        "goal": str(trajectory.get("goal") or run_result.get("query") or ""),
        "video_path": str(run_result.get("video_path") or ""),
        "processed_output_dir": str(((run_result.get("preprocessing") or {}).get("processed_output_dir") or "")),
        "registry_snapshot_id": str(trajectory.get("registry_snapshot_id") or "snapshot_000"),
        "prefix_steps": prefix,
    }
    safe_state, excluded_fields = _redact(state)
    # Redacted fields are expected audit findings, not leaks.  The invariant
    # is that none of the forbidden keys survives in ``cached_state``.
    def _surviving_forbidden(value: Any) -> List[str]:
        found: List[str] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _key_token(key) in _FORBIDDEN_KEYS:
                    found.append(str(key))
                found.extend(_surviving_forbidden(item))
        elif isinstance(value, list):
            for item in value:
                found.extend(_surviving_forbidden(item))
        return found

    surviving_forbidden = _surviving_forbidden(safe_state)
    leakage_passed = not surviving_forbidden
    return {
        "mode": "selective_suffix",
        "mutation": (
            {
                "action": operations[0][0],
                "target": operations[0][1],
                "effect": copy.deepcopy(dict(operations[0][2])),
            }
            if len(operations) == 1
            else None
        ),
        "mutations": [
            {
                "action": action,
                "target": target,
                "effect": copy.deepcopy(dict(effect)),
            }
            for action, target, effect in operations
        ],
        "cut": {
            "step_index": cut_step,
            "reused_step_count": len(prefix),
            "reexecute_step_count": len(suffix),
            "reason": cut_reason,
            "triggered": triggered,
        },
        "replay_kind": replay_kind,
        "cached_state": safe_state,
        "reexecute_actions": suffix,
        "execution": {
            "runtime_required": runtime_required,
            "status": "not_executed",
            "causal_fix_status": causal_status,
            "no_full_trigger_case_rerun": True,
        },
        "leakage_audit": {
            "passed": leakage_passed,
            "excluded_fields": excluded_fields,
            "surviving_forbidden_fields": surviving_forbidden,
            "ground_truth_used_for_cut": False,
            "annotation_used_for_cut": False,
            "answer_used_for_cut": False,
            "answer_may_be_used_only_by_separate_verifier": True,
        },
        "follow_up": {
            "experimental_snapshot": True,
            "matching_cases_required": True,
            "periodic_regression_required": True,
            "final_acceptance": "causal_fix + matching_cases + periodic_regression + numeric_gates",
        },
        "rollback": {
            "candidate_snapshot_id": str(snapshot.get("snapshot_id") or ""),
            "parent_snapshot_id": str(snapshot.get("parent_snapshot_id") or "snapshot_000"),
            "online_publish": False,
        },
        "state_digest": _canonical_digest(safe_state),
    }
