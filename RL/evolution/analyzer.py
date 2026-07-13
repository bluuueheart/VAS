"""Offline trajectory mining and safe candidate policy generation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .models import (
    EvolutionAction,
    EvolutionIntent,
    FailureDiagnosis,
    FailureType,
    PatchOperation,
    RegistrySnapshot,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:12]}"


def iter_steps(trajectory: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        yield step
        child = step.get("sub_trajectory")
        if isinstance(child, dict):
            yield from iter_steps(child)
        for item in step.get("sub_trajectories") or []:
            if isinstance(item, dict):
                yield from iter_steps(item)


def _iter_parallel_tool_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_parallel_tool_records(item)
        return
    if not isinstance(value, dict):
        return
    nested = value.get("items")
    if isinstance(nested, list):
        for item in nested:
            yield from _iter_parallel_tool_records(item)
    tool = str(value.get("tool") or "").strip()
    if tool and tool not in {"Parallel", "System", "Assistant_Note"}:
        status = str(value.get("status") or "").strip().lower()
        result = str(value.get("result") or "").strip()
        if not status:
            status = "error" if result.lower().startswith(("error", "failed", "no_valid_result")) else "success"
        observation = result if status == "success" else f"{status}: {result}"
        yield {"action": tool, "observation": observation, "status": status}


def _tool_records(trajectory: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    for step in iter_steps(trajectory):
        action = str(step.get("action") or "")
        if action in {"", "Finish", "System", "Assistant_Note"}:
            continue
        if action == "Parallel":
            records.extend(_iter_parallel_tool_records(step.get("observation_items")))
            continue
        records.append(step)
    return records


def _event_ids(trajectory: Mapping[str, Any], event_type: str) -> List[str]:
    result: List[str] = []
    for index, event in enumerate(trajectory.get("debug_events") or []):
        if isinstance(event, dict) and str(event.get("type") or "") == event_type:
            result.append(str(event.get("event_id") or f"debug_{index}"))
    return result


def reconstruct_finish_evidence(trajectory: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], str, str]:
    slots: Dict[str, Dict[str, Any]] = {}
    finish_slots: Optional[Dict[str, Dict[str, Any]]] = None
    for event in trajectory.get("debug_events") or []:
        if not isinstance(event, dict) or event.get("type") != "llm_iteration":
            continue
        response = event.get("response")
        if not isinstance(response, str):
            continue
        try:
            payload = json.loads(response)
        except (TypeError, ValueError):
            continue
        raw_slots = payload.get("Evidence") or payload.get("evidence") or []
        finish_payload = payload.get("Finish") or payload.get("finish") or {}
        if isinstance(finish_payload, dict):
            nested_slots = finish_payload.get("Evidence") or finish_payload.get("evidence") or []
            if isinstance(nested_slots, list):
                raw_slots = list(raw_slots) + nested_slots if isinstance(raw_slots, list) else nested_slots
        current_slots: Dict[str, Dict[str, Any]] = {}
        for item in raw_slots if isinstance(raw_slots, list) else []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "unnamed claim")
            record = {
                "label": label,
                "value": str(item.get("value") or label),
                "citation": str(item.get("citation") or ""),
                "evidence_status": str(item.get("evidence_status") or "missing").lower(),
                "logical_conflict": bool(item.get("logical conflict", item.get("logical_conflict", False))),
            }
            slots[label.lower()] = record
            current_slots[label.lower()] = record
        if isinstance(finish_payload, dict) and current_slots:
            finish_slots = current_slots
    finish = next((item for item in reversed(list(iter_steps(trajectory))) if item.get("action") == "Finish"), {})
    args = finish.get("action_args") if isinstance(finish.get("action_args"), dict) else {}
    answer = str(args.get("answer") or args.get("result") or trajectory.get("final_answer") or "")
    return list((finish_slots or slots).values()), answer, str(args.get("completion_basis") or "")


def has_conflicting_evidence(trajectory: Mapping[str, Any]) -> bool:
    slots, _, _ = reconstruct_finish_evidence(trajectory)
    return any(bool(item.get("logical_conflict")) for item in slots)


def diagnose(
    sample: Mapping[str, Any],
    run_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Optional[FailureDiagnosis]:
    agent = run_result.get("agent") if isinstance(run_result.get("agent"), dict) else {}
    trajectory = agent.get("trajectory") if isinstance(agent.get("trajectory"), dict) else {}
    trajectory_id = str(trajectory.get("trajectory_id") or sample.get("sample_uid") or "unknown")
    tools = _tool_records(trajectory)
    failed_tools = [
        item for item in tools
        if str(item.get("observation") or "").strip().lower().startswith(("error", "failed"))
    ]
    correct = float(sample.get("acc") or 0.0) >= 1.0
    planner_success = bool(sample.get("planner_success", run_result.get("success", False)))
    verification = sample.get("_verification") if isinstance(sample.get("_verification"), dict) else {}
    threshold = int((config.get("diagnosis") or {}).get("excessive_tool_calls", 8))

    if verification and not bool(verification.get("can_finish")):
        has_conflict = float(verification.get("contradiction_score") or 0.0) > 0.0
        failure_type = FailureType.CONTRADICTION_HANDLING if has_conflict else FailureType.PREMATURE_TERMINATION
        component = "evidence_verifier"
        cause = "The historical trajectory reached Finish without satisfying the configured evidence gate."
        fix = str(verification.get("feedback") or "Acquire missing cited evidence before Finish.")
        confidence = 0.95
    elif failed_tools and (not planner_success or not correct):
        failure_type = FailureType.TOOL_EXECUTION
        component = "tool_runtime"
        cause = "The trajectory terminated after one or more tool execution failures."
        fix = "Route to a validated fallback after the first hard tool failure."
        confidence = 0.9
    elif has_conflicting_evidence(trajectory):
        failure_type = FailureType.CONTRADICTION_HANDLING
        component = "evidence_verifier"
        cause = "The trajectory contained conflicting evidence but still produced a terminal answer."
        fix = "Reject Finish until conflicts are explicitly resolved by a new modality or time window."
        confidence = 0.9
    elif not correct and len(tools) > threshold:
        failure_type = FailureType.EVIDENCE_INTEGRATION
        component = "reasoner"
        cause = "Many successful probes did not converge to the correct option."
        fix = "Require claim-level evidence comparison before further retrieval or termination."
        confidence = 0.85
    elif not correct and not tools:
        failure_type = FailureType.PREMATURE_TERMINATION
        component = "evidence_verifier"
        cause = "The planner answered without executing an evidence acquisition action."
        fix = "Require cited structured evidence and a verified cross-modal chain before Finish."
        confidence = 0.95
    elif not correct:
        failure_type = FailureType.ANSWER_SYNTHESIS
        component = "reasoner"
        cause = "The final option was inconsistent with the collected trajectory evidence."
        fix = "Compare every candidate option against cited evidence before selecting the final answer."
        confidence = 0.8
    else:
        return None

    tool_ids = sorted({str(item.get("action") or "") for item in failed_tools or tools if item.get("action")})
    content = [trajectory_id, failure_type.value, cause]
    return FailureDiagnosis(
        diagnosis_id=stable_id("diagnosis", content),
        trajectory_ids=[trajectory_id],
        failure_type=failure_type,
        blamed_component=component,
        blamed_tool_ids=tool_ids,
        supporting_event_ids=_event_ids(trajectory, "verifier_decision"),
        root_cause=cause,
        counterfactual_fix=fix,
        confidence=confidence,
    )


def build_intent(diagnoses: List[FailureDiagnosis], config: Mapping[str, Any]) -> Optional[EvolutionIntent]:
    confidence_floor = float((config.get("diagnosis") or {}).get("confidence_threshold", 0.7))
    eligible = [item for item in diagnoses if item.confidence >= confidence_floor]
    if not eligible:
        return None
    failure_types = {item.failure_type for item in eligible}
    operations: List[PatchOperation] = []
    if FailureType.PREMATURE_TERMINATION in failure_types or FailureType.CONTRADICTION_HANDLING in failure_types:
        operations.append(PatchOperation(
            action=EvolutionAction.UPDATE_ROUTER_RULE,
            target_type="evidence_verifier",
            target_ids=["finish_gate"],
            before={"enforced": False},
            after={"enforced": True, "reject_unverified_finish": True, "resolve_conflicts": True},
            reason="Observed unsupported or conflict-bearing terminal answers.",
        ))
    if FailureType.EVIDENCE_INTEGRATION in failure_types or FailureType.ANSWER_SYNTHESIS in failure_types:
        operations.append(PatchOperation(
            action=EvolutionAction.ADD_COMPOSITION_EDGE,
            target_type="workflow",
            target_ids=["candidate_comparison"],
            before=None,
            after={"after": "evidence_acquisition", "before": "finish", "required": True},
            reason="Collected evidence was not compared consistently across answer candidates.",
        ))
    if FailureType.TOOL_EXECUTION in failure_types:
        operations.append(PatchOperation(
            action=EvolutionAction.ADD_FALLBACK,
            target_type="router",
            target_ids=sorted({tool for item in eligible for tool in item.blamed_tool_ids}),
            before=None,
            after={"on_hard_failure": "next_validated_same_family_tool", "max_retries": 1},
            reason="Hard tool failures should not terminate the evidence chain.",
        ))
    if not operations:
        return None
    diagnosis_ids = [item.diagnosis_id for item in eligible]
    return EvolutionIntent(
        intent_id=stable_id("intent", diagnosis_ids),
        diagnosis_ids=diagnosis_ids,
        scope="global" if len(operations) > 1 else "router",
        operations=operations,
        expected_effects=["fewer unsupported finishes", "higher evidence-grounded answer consistency"],
        risks=["additional tool calls", "higher latency"],
    )


def build_candidate_snapshot(intent: EvolutionIntent, parent_id: str) -> RegistrySnapshot:
    policy: Dict[str, Any] = {}
    argument_policy: Dict[str, Any] = {}
    for operation in intent.operations:
        destination = argument_policy if operation.action == EvolutionAction.UPDATE_ARGUMENT_POLICY else policy
        destination[operation.action.value] = operation.after
    return RegistrySnapshot(
        snapshot_id=stable_id("snapshot", [parent_id, intent.model_dump()]),
        parent_snapshot_id=parent_id,
        created_at=utc_now(),
        router_policy=policy,
        argument_policy=argument_policy,
        patch_ids=[intent.intent_id],
        status="candidate",
    )


def load_sample_pair(result_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with result_path.open("r", encoding="utf-8") as handle:
        run_result = json.load(handle)
    sample_path = result_path.parents[1] / "mix_sample_result.json"
    with sample_path.open("r", encoding="utf-8") as handle:
        sample = json.load(handle)
    return sample, run_result
