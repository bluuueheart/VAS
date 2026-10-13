"""Evidence-based value accounting for the four role-specific agents.

This is a diagnostic report, not a claim of causal ablation.  It records what
each role changed in a real trajectory (next action, evidence state, or Finish
gate) and labels cases where a role only echoed an answer or accepted a wrong
one.  The Evolver role is evaluated separately at cycle level because it is an
online controller turn between worker groups rather than a per-tool step.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("response_payload")
    return value if isinstance(value, Mapping) else {}


def _truthful_case_correct(sample: Mapping[str, Any]) -> bool:
    """Recompute option correctness instead of trusting legacy label-only acc."""

    from evaluationVideo.Mix.reward_metrics import _evaluate_rule_answer
    options = sample.get("options")
    truth = sample.get("ground_truth_option") or sample.get("ground_truth")
    answer = sample.get("raw_final_answer")
    if options and truth is not None and answer:
        return _evaluate_rule_answer(answer, truth, options)["answer_correct"]
    return float(sample.get("acc") or 0.0) >= 1.0


def _next_step(steps: List[Mapping[str, Any]], event: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    try:
        iteration = int(event.get("iteration") or 0)
    except (TypeError, ValueError):
        iteration = 0
    if iteration <= 0:
        return None
    index = iteration
    return steps[index] if 0 <= index < len(steps) else None


def _step_action(step: Optional[Mapping[str, Any]]) -> str:
    return str((step or {}).get("action") or "").strip()


def _next_action_matches_recommendation(
    recommendation: Mapping[str, Any],
    next_step: Optional[Mapping[str, Any]],
) -> bool:
    kind = str(recommendation.get("kind") or "").strip().lower()
    action = _step_action(next_step)
    if not kind or not action:
        return False
    if kind == "finish":
        return action == "Finish"
    if kind in {"continue", "reobserve", "cross_verify"}:
        if action == "Finish":
            return False
        tool_hint = str(recommendation.get("tool_hint") or "").strip().lower()
        if not tool_hint or tool_hint in {"none", "null", "finish"}:
            return True
        action_text = str(action).lower()
        raw_action_args = (next_step or {}).get("action_args", {})
        batch = raw_action_args.get("batch", []) if isinstance(raw_action_args, Mapping) else []
        batch_names = {
            str(item.get("name") or "").strip().lower()
            for item in batch
            if isinstance(item, Mapping)
        }
        return tool_hint in action_text or tool_hint in batch_names
    return False


def _role_turns(events: Iterable[Mapping[str, Any]], role: str) -> List[Mapping[str, Any]]:
    return [
        event
        for event in events
        if str(event.get("agent_role") or "").strip().lower() == role
        and str(event.get("type") or "").strip() == "agent_llm_turn"
    ]


def analyze_case(sample: Mapping[str, Any], run_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Analyze observed contributions for one completed or failed case."""
    trajectory = ((run_result.get("agent") or {}).get("trajectory") or {})
    steps = [item for item in (trajectory.get("steps") or []) if isinstance(item, Mapping)]
    events = [item for item in (trajectory.get("debug_events") or []) if isinstance(item, Mapping)]
    correct = _truthful_case_correct(sample)

    planner_turns = [
        event
        for event in events
        if str(event.get("type") or "").strip() == "llm_iteration"
        or str(event.get("agent_role") or "").strip().lower() == "planner"
    ]
    planner_actions = [
        str(event.get("parsed_action") or "").strip()
        for event in planner_turns
        if str(event.get("parsed_action") or "").strip()
    ]
    tool_steps = [
        step for step in steps
        if _step_action(step) not in {"", "Finish", "System", "Assistant_Note"}
    ]
    planner_record = {
        "turns": len(planner_turns),
        "tool_orchestration_steps": len(tool_steps),
        "finish_proposals": sum(1 for action in planner_actions if action == "Finish"),
        "schema_valid_turns": sum(1 for event in planner_turns if bool(event.get("parse_success", True))),
        "value_verdict": "observed" if planner_turns and steps else "not_observed",
        "evidence": ["selected executable actions and/or a terminal answer"],
    }

    observer_turns = _role_turns(events, "observer")
    observer_updates = 0
    observer_verified_updates = 0
    observer_conflicts = 0
    observer_reobservations = 0
    observer_followed = 0
    observer_non_echo = 0
    for event in observer_turns:
        payload = _payload(event)
        updates = payload.get("evidence_updates") or []
        if isinstance(updates, list):
            observer_updates += len(updates)
            observer_verified_updates += sum(
                1 for item in updates
                if isinstance(item, Mapping) and str(item.get("evidence_status") or "").lower() == "verified"
            )
            observer_conflicts += sum(
                1 for item in updates
                if isinstance(item, Mapping) and bool(item.get("logical_conflict"))
            )
        recommendation = payload.get("recommended_action")
        if not isinstance(recommendation, Mapping):
            continue
        kind = str(recommendation.get("kind") or "").lower()
        if kind in {"reobserve", "cross_verify", "continue"}:
            observer_reobservations += 1
        if kind != "finish" or observer_updates or observer_conflicts:
            observer_non_echo += 1
        if _next_action_matches_recommendation(recommendation, _next_step(steps, event)):
            observer_followed += 1
    observer_verdict = "not_observed"
    if observer_turns:
        if observer_conflicts or observer_verified_updates or observer_reobservations and observer_followed:
            observer_verdict = "useful"
        elif observer_non_echo:
            observer_verdict = "mixed"
        else:
            observer_verdict = "redundant_or_unproven"
    observer_record = {
        "turns": len(observer_turns),
        "schema_valid_turns": sum(1 for event in observer_turns if bool(event.get("schema_valid"))),
        "evidence_updates": observer_updates,
        "verified_updates": observer_verified_updates,
        "logical_conflicts_flagged": observer_conflicts,
        "reobserve_or_cross_verify_recommendations": observer_reobservations,
        "recommendations_followed_by_planner": observer_followed,
        "value_verdict": observer_verdict,
        "evidence": (
            ["changed evidence state or directed a subsequent observation action"]
            if observer_verdict == "useful"
            else ["no non-echo contribution was observed"]
        ),
    }

    reflector_turns = _role_turns(events, "reflector")
    reflector_decisions = [str(_payload(event).get("decision") or "").lower() for event in reflector_turns]
    reject_count = sum(decision in {"reject", "request_more_evidence"} for decision in reflector_decisions)
    effective_blocks = 0
    for event, decision in zip(reflector_turns, reflector_decisions):
        if decision not in {"reject", "request_more_evidence"}:
            continue
        next_step = _next_step(steps, event)
        if next_step is not None and _step_action(next_step) != "Finish":
            effective_blocks += 1
    accept_correct = sum(
        decision == "accept" and correct for decision in reflector_decisions
    )
    accept_wrong = sum(
        decision == "accept" and not correct for decision in reflector_decisions
    )
    if not reflector_turns:
        reflector_verdict = "not_observed"
    elif effective_blocks or accept_correct:
        reflector_verdict = "useful"
    elif accept_wrong:
        reflector_verdict = "insufficient_or_harmful"
    else:
        reflector_verdict = "redundant_or_unproven"
    reflector_record = {
        "turns": len(reflector_turns),
        "schema_valid_turns": sum(1 for event in reflector_turns if bool(event.get("schema_valid"))),
        "accept_count": sum(decision == "accept" for decision in reflector_decisions),
        "reject_or_request_more_count": reject_count,
        "effective_finish_blocks": effective_blocks,
        "accept_correct_count": accept_correct,
        "accept_wrong_count": accept_wrong,
        "value_verdict": reflector_verdict,
        "evidence": (
            ["blocked Finish before additional evidence or validated a correct Finish"]
            if reflector_verdict == "useful"
            else ["the gate did not demonstrate a beneficial decision on this case"]
        ),
    }

    return {
        "sample_uid": str(sample.get("sample_uid") or ""),
        "dataset_name": str(sample.get("dataset_name") or ""),
        "answer_correct": correct,
        "predicted_label": str(sample.get("predicted_label") or ""),
        "roles": {
            "planner": planner_record,
            "observer": observer_record,
            "reflector": reflector_record,
            "evolver": {
                "value_verdict": "online_cycle_role_not_case_turn",
                "evidence": ["evaluated by the online cycle replay report"],
            },
        },
    }


def summarize_cases(cases: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    case_list = [item for item in cases if isinstance(item, Mapping)]
    aggregate: Dict[str, Any] = {"case_count": len(case_list), "roles": {}}
    for role in ("planner", "observer", "reflector"):
        verdicts = [
            str(((case.get("roles") or {}).get(role) or {}).get("value_verdict") or "unknown")
            for case in case_list
        ]
        counts = Counter(verdicts)
        aggregate["roles"][role] = {
            "case_verdict_counts": dict(sorted(counts.items())),
            "useful_case_count": int(counts.get("useful", 0) + counts.get("observed", 0)),
        }
    return aggregate


def summarize_evolver(
    evolver_result: Mapping[str, Any],
    *,
    diagnosis_count: int,
    candidate_snapshot: Optional[Mapping[str, Any]],
    replay_status: str,
    trajectory_count: int,
    proposal_validation: Optional[Mapping[str, Any]] = None,
    fresh_candidate_inference_proven: bool = False,
) -> Dict[str, Any]:
    proposal = evolver_result.get("payload") if isinstance(evolver_result.get("payload"), Mapping) else {}
    decision = str(proposal.get("decision") or "").strip()
    schema_valid = bool(evolver_result.get("schema_valid"))
    validation = proposal_validation if isinstance(proposal_validation, Mapping) else {}
    validation_status = str(validation.get("status") or "").strip()
    if not schema_valid:
        verdict = "invalid"
    elif decision == "propose_mutation" and validation_status not in {"candidate_created", ""}:
        verdict = "proposal_rejected_unexecutable"
    elif replay_status == "accepted" and fresh_candidate_inference_proven:
        verdict = "validated_by_replay"
    elif replay_status in {"accepted", "metrics_only_accepted_pending_fresh_inference"}:
        verdict = "metrics_gate_passed_pending_fresh_inference"
    elif candidate_snapshot and replay_status in {"pending", "not_required"}:
        verdict = "candidate_only_replay_pending"
    elif decision == "no_mutation":
        verdict = "no_mutation_requires_more_cases"
    else:
        verdict = "observed_but_unvalidated"
    targets = []
    from .models import normalize_mutation_payload
    proposal = normalize_mutation_payload(proposal)
    mutations = proposal.get("mutations")
    if isinstance(mutations, list):
        mutation_items = mutations
    else:
        mutation_items = [proposal.get("mutation")]
    for mutation in mutation_items:
        if not isinstance(mutation, Mapping):
            continue
        target = mutation.get("target") or mutation.get("target_ids")
        if isinstance(target, list):
            targets.extend(str(item) for item in target)
        elif target:
            targets.append(str(target))
    dependency_analysis = proposal.get("dependency_analysis")
    if not isinstance(dependency_analysis, Mapping):
        dependency_analysis = validation.get("dependency_analysis")
    dependency_analysis = dict(dependency_analysis) if isinstance(dependency_analysis, Mapping) else {}
    return {
        "trajectory_count": int(trajectory_count),
        "diagnosis_count": int(diagnosis_count),
        "schema_valid": schema_valid,
        "decision": decision,
        "mutation_targets": targets,
        "mutation_operation_count": len(mutation_items) if any(isinstance(item, Mapping) for item in mutation_items) else 0,
        "dependency_analysis": dependency_analysis,
        "candidate_snapshot_created": bool(candidate_snapshot),
        "proposal_validation": dict(validation),
        "replay_status": str(replay_status or ""),
        "fresh_candidate_inference_proven": bool(fresh_candidate_inference_proven),
        "value_verdict": verdict,
        "evidence": [
            "a real Evolver kernel turn was recorded",
            "candidate mutation is not accepted until replay/held-out gates pass",
        ],
    }


def analyze_counterfactual_case(
    baseline_sample: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    ablation_sample: Mapping[str, Any],
    ablation_result: Mapping[str, Any],
    *,
    disabled_role: str,
) -> Dict[str, Any]:
    """Compare one real baseline case with one role-disabled rerun.

    This is the causal case-level audit used by the evaluator.  It only emits
    a role-value verdict when both artifacts belong to the same sample UID;
    missing or mismatched pairs remain explicitly unproven.
    """
    baseline_uid = str(baseline_sample.get("sample_uid") or "").strip()
    ablation_uid = str(ablation_sample.get("sample_uid") or "").strip()
    same_case = bool(baseline_uid and baseline_uid == ablation_uid)
    base_trace = (baseline_result.get("agent") or {}).get("trajectory") or {}
    ablated_trace = (ablation_result.get("agent") or {}).get("trajectory") or {}
    contracts = [event for event in ablated_trace.get("debug_events", [])
                 if event.get("type") == "agent_kernel_contract"]
    disabled = any(disabled_role in event.get("disabled_online_roles", []) for event in contracts)
    actual_turns = _role_turns(ablated_trace.get("debug_events", []), disabled_role)
    execution_proven = bool(
        same_case and base_trace.get("goal") and base_trace.get("goal") == ablated_trace.get("goal")
        and base_trace.get("registry_snapshot_id")
        and base_trace.get("registry_snapshot_id") == ablated_trace.get("registry_snapshot_id")
        and base_trace.get("trajectory_id") and ablated_trace.get("trajectory_id")
        and base_trace["trajectory_id"] != ablated_trace["trajectory_id"]
        and base_trace.get("steps") and ablated_trace.get("steps")
        and _role_turns(base_trace.get("debug_events", []), disabled_role)
        and disabled and not actual_turns
    )
    baseline_correct = _truthful_case_correct(baseline_sample)
    ablation_correct = _truthful_case_correct(ablation_sample)
    baseline_success = bool(baseline_result.get("success"))
    ablation_success = bool(ablation_result.get("success"))
    answer_delta = int(baseline_correct) - int(ablation_correct)
    if not execution_proven:
        verdict = "unproven_counterfactual_execution"
    elif answer_delta > 0:
        verdict = "useful"
    elif answer_delta < 0:
        verdict = "harmful"
    else:
        verdict = "no_measurable_answer_effect"
    return {
        "sample_uid": baseline_uid or ablation_uid,
        "disabled_role": str(disabled_role or "").strip().lower(),
        "same_case": same_case,
        "execution_proven": execution_proven,
        "baseline_correct": baseline_correct,
        "ablation_correct": ablation_correct,
        "baseline_success": baseline_success,
        "ablation_success": ablation_success,
        "answer_delta_baseline_minus_ablation": answer_delta,
        "baseline_tool_use_num": int(baseline_sample.get("tool_use_num") or 0),
        "ablation_tool_use_num": int(ablation_sample.get("tool_use_num") or 0),
        "baseline_elapsed_seconds": float(baseline_sample.get("elapsed_seconds") or 0.0),
        "ablation_elapsed_seconds": float(ablation_sample.get("elapsed_seconds") or 0.0),
        "value_verdict": verdict,
    }


def summarize_counterfactual_pairs(pairs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    pair_list = [item for item in pairs if isinstance(item, Mapping)]
    verdicts = Counter(
        str(item.get("value_verdict") or "unknown")
        for item in pair_list
    )
    valid_pairs = [item for item in pair_list if bool(item.get("execution_proven"))]
    answer_gain = sum(
        float(item.get("answer_delta_baseline_minus_ablation") or 0.0)
        for item in valid_pairs
    )
    return {
        "pair_count": len(pair_list),
        "valid_same_case_pair_count": len(valid_pairs),
        "value_verdict_counts": dict(sorted(verdicts.items())),
        "baseline_minus_ablation_correct_count_delta": round(answer_gain, 6),
        "method": "paired same-case rerun with the named online role disabled",
    }
