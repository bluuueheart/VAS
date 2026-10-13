"""Offline trajectory mining and safe candidate policy generation."""
import hashlib
import json
import os
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import (
    EVOLVER_HARNESS_ACTIONS,
    normalize_mutation_payload,
    valid_recovery_policy,
    SOURCE_BACKED_ACTIONS,
    MAX_EVOLVER_OPERATIONS,
    EvolutionAction,
    EvolutionIntent,
    FailureDiagnosis,
    FailureType,
    PatchOperation,
    RegistrySnapshot,
    analyze_evolver_dependencies,
)
from .source_mutation import validate_source_edit


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
    # Prefer the actual runtime ledger, including Observer updates and
    # supersession. Planner prose alone is not the evidence the verifier saw.
    root_steps = list(trajectory.get("steps") or [])
    finish_index = next((i for i in range(len(root_steps) - 1, -1, -1)
                         if root_steps[i].get("action") == "Finish"), len(root_steps))
    finish = root_steps[finish_index] if finish_index < len(root_steps) else {}
    args = finish.get("action_args") or {}
    answer = str(args.get("answer") or args.get("result") or trajectory.get("final_answer") or "")
    from .runtime_checkpoint import read, decode, fingerprint
    for event in reversed(list(trajectory.get("debug_events") or [])):
        if not isinstance(event, Mapping) or event.get("type") != "runtime_checkpoint" or not event.get("path"):
            continue
        path = Path(event["path"])
        if not path.is_file():
            continue
        body = read(path)
        if (fingerprint(body) != event.get("sha256") or body.get("goal") != trajectory.get("goal")
                or body.get("snapshot_id") != trajectory.get("registry_snapshot_id", "snapshot_000")):
            raise ValueError("runtime evidence checkpoint identity mismatch")
        if body.get("prefix_steps") == finish_index:
            state = decode(body["state"])
            ledger = state.get("evidence_slots")
            if isinstance(ledger, list):
                return ledger, answer, str(args.get("completion_basis") or "")
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
                "claim_id": str(item.get("claim_id") or item.get("id") or "").strip(),
                "scope": str(item.get("scope") or "").strip(),
                "label": label,
                "target": item.get("target", ""),
                "relation": item.get("relation", ""),
                "direction": item.get("direction", ""),
                "qualifiers": item.get("qualifiers", []),
                "observed_value": item.get("observed_value", ""),
                "value": str(item.get("value") or label),
                "citation": str(item.get("citation") or ""),
                "source_tool": str(item.get("source_tool") or "").strip(),
                "uncertainty": str(item.get("uncertainty") or "").strip(),
                "time_window": item.get("time_window", ""),
                "evidence_status": str(item.get("evidence_status") or "missing").lower(),
                "logical_conflict": bool(item.get("logical conflict", item.get("logical_conflict", False))),
            }
            key = str(record["claim_id"] or label).lower() + ":" + str(record["target"]).lower()
            slots[key] = record
            current_slots[key] = record
        if isinstance(finish_payload, dict) and current_slots:
            finish_slots = current_slots
    finish = next((item for item in reversed(list(iter_steps(trajectory))) if item.get("action") == "Finish"), {})
    args = finish.get("action_args") if isinstance(finish.get("action_args"), dict) else {}
    answer = str(args.get("answer") or args.get("result") or trajectory.get("final_answer") or "")
    return list((finish_slots or slots).values()), answer, str(args.get("completion_basis") or "")


def has_conflicting_evidence(trajectory: Mapping[str, Any]) -> bool:
    slots, _, _ = reconstruct_finish_evidence(trajectory)
    return any(bool(item.get("logical_conflict")) for item in slots)


def _layered_failure_attribution(
    *,
    trajectory: Mapping[str, Any],
    verification: Mapping[str, Any],
    correct: bool,
    failed_tools: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Separate agent design, handoff, and framework causes before mutation.

    A wrong answer is not enough to blame Planner, and a verifier rejection is
    not enough to blame the verifier. This compact attribution is evidence
    from the trajectory only; it does not use labels or ground truth as model
    input.
    """
    debug_events = [
        item for item in (trajectory.get("debug_events") or [])
        if isinstance(item, Mapping)
    ]
    role_schema_failures = sorted({
        str(item.get("agent_role") or "").strip()
        for item in debug_events
        if str(item.get("agent_role") or "").strip() in {"planner", "observer", "reflector"}
        and item.get("schema_valid") is False
    })
    repeated_finish_count = sum(
        1 for item in (trajectory.get("steps") or [])
        if isinstance(item, Mapping)
        and str(item.get("action") or "").strip() == "Finish"
    )
    contract = verification.get("claim_contract") if isinstance(verification, Mapping) else {}
    contract = contract if isinstance(contract, Mapping) else {}
    missing_contract = [str(item) for item in (contract.get("missing") or []) if str(item).strip()]
    has_observer_turn = any(str(item.get("agent_role") or "") == "observer" for item in debug_events)
    has_reflector_turn = any(str(item.get("agent_role") or "") == "reflector" for item in debug_events)
    layers: Dict[str, Dict[str, Any]] = {
        "agent_design": {
            "status": "not_indicated",
            "signals": [],
            "candidate_agents": [],
        },
        "agent_interaction": {
            "status": "not_indicated",
            "signals": [],
            "handoff_checks": [],
        },
        "framework_logic": {
            "status": "not_indicated",
            "signals": [],
            "checks": [],
        },
    }
    if not correct and not failed_tools and not bool(verification.get("can_finish")):
        layers["agent_design"]["signals"].append("answer remained unsupported or was synthesized inconsistently")
        layers["agent_design"]["candidate_agents"].append("planner")
    if role_schema_failures:
        layers["agent_design"]["signals"].append(
            "role schema failure: " + ",".join(role_schema_failures)
        )
        layers["agent_design"]["candidate_agents"].extend(role_schema_failures)
    if has_observer_turn and missing_contract:
        layers["agent_interaction"]["signals"].append(
            "observer/planner/verifier claim contract has unresolved dimensions"
        )
        layers["agent_interaction"]["handoff_checks"].extend(missing_contract)
    if has_reflector_turn and not bool(verification.get("can_finish")):
        layers["agent_interaction"]["signals"].append(
            "Reflector judgment and deterministic claim contract must be reconciled"
        )
        layers["agent_interaction"]["handoff_checks"].append("reflector_verifier_boundary")
    if failed_tools:
        layers["framework_logic"]["signals"].append("tool execution/runtime failure")
        layers["framework_logic"]["checks"].append("tool_status_contract")
    if repeated_finish_count > 1:
        layers["framework_logic"]["signals"].append("repeated Finish candidate reached the runtime")
        layers["framework_logic"]["checks"].append("finish_revision_guard")
    if not bool(verification.get("can_finish")) and correct:
        layers["framework_logic"]["signals"].append(
            "correct answer was blocked by the evidence termination boundary"
        )
        layers["framework_logic"]["checks"].append("verifier_reflector_alignment")
    for layer in layers.values():
        layer["status"] = "indicated" if layer["signals"] else "not_indicated"
    if layers["framework_logic"]["signals"] and (failed_tools or repeated_finish_count > 1 or correct):
        primary_layer = "framework_logic"
    elif layers["agent_interaction"]["signals"]:
        primary_layer = "agent_interaction"
    elif layers["agent_design"]["signals"]:
        primary_layer = "agent_design"
    else:
        primary_layer = "unknown"
    return {
        "primary_layer": primary_layer,
        "layers": layers,
        "role_schema_failures": role_schema_failures,
        "claim_contract_missing": missing_contract,
        "finish_candidate_count": repeated_finish_count,
        "contract_pass": bool(primary_layer == "unknown" or not missing_contract),
    }


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
        if (
            str(item.get("observation") or "").strip().lower().startswith(
                ("error", "failed", "no_valid_result")
            )
            or any(
                str(record.get("status") or "").strip().lower()
                in {"error", "failed", "no_valid_result"}
                for record in (item.get("observation_items") or [])
                if isinstance(record, Mapping)
            )
        )
    ]
    correct = float(sample.get("acc") or 0.0) >= 1.0
    planner_success = bool(sample.get("planner_success", run_result.get("success", False)))
    verification = sample.get("_verification") if isinstance(sample.get("_verification"), dict) else {}
    threshold = int((config.get("diagnosis") or {}).get("excessive_tool_calls", 8))
    layered_attribution = _layered_failure_attribution(
        trajectory=trajectory,
        verification=verification,
        correct=correct,
        failed_tools=failed_tools,
    )

    if verification and not bool(verification.get("can_finish")):
        has_conflict = float(verification.get("contradiction_score") or 0.0) > 0.0
        failure_type = FailureType.CONTRADICTION_HANDLING if has_conflict else FailureType.PREMATURE_TERMINATION
        component = "evidence_verifier"
        cause = "The historical trajectory reached Finish without satisfying the configured evidence gate."
        fix = str(verification.get("feedback") or "Acquire missing cited evidence before Finish.")
        confidence = 0.95
    elif failed_tools:
        failure_type = FailureType.TOOL_EXECUTION
        component = "tool_runtime"
        cause = (
            "The trajectory contained one or more tool execution failures; the final answer is not a safe "
            "proxy for tool correctness."
        )
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
    elif len(tools) > threshold:
        failure_type = FailureType.PLANNING
        component = "reasoner"
        cause = "The answer is correct but successful evidence calls exceed the configured cost threshold."
        fix = "Reduce redundant evidence acquisition while preserving answer quality; validate with repeated paired timing."
        confidence = 0.75
    else:
        return None

    tool_ids = sorted({str(item.get("action") or "") for item in failed_tools or tools if item.get("action")})
    content = [trajectory_id, failure_type.value, cause]
    if not correct and failed_tools:
        failure_scope = "global_and_local_failure"
    elif not correct:
        failure_scope = "global_answer_failure"
    elif failed_tools:
        failure_scope = "local_tool_failure"
    elif failure_type == FailureType.PLANNING:
        failure_scope = "efficiency"
    else:
        failure_scope = "local_verifier_failure"
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
        failure_scope=failure_scope,
        attribution={
            "answer_outcome": "correct" if correct else "incorrect",
            "tool_failure_count": len(failed_tools),
            "planner_success": planner_success,
            "verifier_blocked": bool(verification) and not bool(verification.get("can_finish")),
            "blamed_component": component,
            "layered_root_cause": layered_attribution,
        },
    )


def build_intent(diagnoses: List[FailureDiagnosis], config: Mapping[str, Any]) -> Optional[EvolutionIntent]:
    confidence_floor = float((config.get("diagnosis") or {}).get("confidence_threshold", 0.7))
    eligible = [item for item in diagnoses if item.confidence >= confidence_floor]
    if not eligible:
        return None
    failure_types = {item.failure_type for item in eligible}
    operations: List[PatchOperation] = []
    if (
        FailureType.PREMATURE_TERMINATION in failure_types
        or FailureType.CONTRADICTION_HANDLING in failure_types
        or FailureType.EVIDENCE_INTEGRATION in failure_types
        or FailureType.ANSWER_SYNTHESIS in failure_types
    ):
        # The old deterministic intent emitted update_router_rule, but that
        # action only rendered prompt text and had no finish-gate consumer.
        # Keep the diagnostic path executable by using the real verifier edge.
        operations.append(PatchOperation(
            action=EvolutionAction.ADD_COMPOSITION_EDGE,
            target_type="workflow",
            target_ids=["candidate_comparison"],
            before=None,
            after={"after": "evidence_acquisition", "before": "finish", "required": True},
            reason="Collected evidence was not compared consistently across answer candidates.",
        ))
    # Do not emit add_fallback: it used to be a prompt-only policy with no
    # executable router implementation. A future fallback must first acquire
    # an explicit runtime hook and its own replay contract.
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
        dependency_analysis=analyze_evolver_dependencies([
            {
                "action": operation.action.value,
                "target_ids": list(operation.target_ids),
                "after": dict(operation.after),
            }
            for operation in operations
        ]),
    )


def build_candidate_snapshot(intent: EvolutionIntent, parent_id: str) -> RegistrySnapshot:
    dependency_analysis = analyze_evolver_dependencies([
        {
            "action": operation.action.value,
            "target_ids": list(operation.target_ids),
            "after": dict(operation.after),
        }
        for operation in intent.operations
    ])
    if not dependency_analysis.get("dependency_pass"):
        raise ValueError(
            "Evolution intent contains operations that fail dependency analysis: "
            f"{dependency_analysis.get('conflicts') or dependency_analysis.get('missing_dependencies')}"
        )
    policy: Dict[str, Any] = {}
    argument_policy: Dict[str, Any] = {}
    tool_overrides: Dict[str, Dict[str, Any]] = {}
    source_overrides: Dict[str, Dict[str, Any]] = {}
    for operation in intent.operations:
        if operation.action == EvolutionAction.EDIT_TOOL_SOURCE or (
            operation.action.value in SOURCE_BACKED_ACTIONS or operation.action == EvolutionAction.ADD_COMPOSITION_EDGE
        ) and isinstance(operation.after, Mapping) and "edits" in operation.after:
            validate_source_edit(operation.after)
            for target_id in operation.target_ids:
                source_overrides[str(target_id)] = dict(operation.after)
        elif operation.action == EvolutionAction.UPDATE_TOOL_SCHEMA:
            for target_id in operation.target_ids:
                tool_overrides[str(target_id)] = dict(operation.after)
        elif operation.action in {
            EvolutionAction.ADD_COMPOSITION_EDGE,
            EvolutionAction.UPDATE_RECOVERY_POLICY,
        }:
            policy[operation.action.value] = operation.after
        else:
            raise ValueError(
                f"{operation.action.value} has no executable Evolution harness hook"
            )
    return RegistrySnapshot(
        snapshot_id=stable_id("snapshot", [parent_id, intent.model_dump()]),
        parent_snapshot_id=parent_id,
        created_at=utc_now(),
        router_policy=policy,
        argument_policy=argument_policy,
        tool_overrides=tool_overrides,
        source_overrides=source_overrides,
        base_tool_library_baseline_id=(
            str(os.environ.get("CFV_REGISTRY_BASELINE_ID") or "").strip() or None
        ),
        patch_ids=[intent.intent_id],
        status="candidate",
        dependency_analysis=dependency_analysis,
    )


_EVOLVER_HARNESS_ACTIONS = frozenset(EVOLVER_HARNESS_ACTIONS)


def build_candidate_snapshot_from_evolver(
    evolver_result: Mapping[str, Any],
    *,
    parent_id: str,
    diagnosis_ids: Optional[List[str]] = None,
) -> Tuple[Optional[RegistrySnapshot], Dict[str, Any]]:
    """Convert one validated Evolver proposal into an isolated candidate.

    The LLM is allowed to propose only mutations understood by
    ``evolution.harness``. Free-form text policies are diagnostic output, not
    executable changes, and are rejected instead of being silently recorded as
    if they had been applied.
    """
    result = evolver_result if isinstance(evolver_result, Mapping) else {}
    if not bool(result.get("schema_valid")):
        return None, {"status": "invalid_schema", "reason": "Evolver JSON contract failed."}
    payload = normalize_mutation_payload(result.get("payload") if isinstance(result.get("payload"), Mapping) else {})
    decision = str(payload.get("decision") or "").strip()
    if decision != "propose_mutation":
        return None, {"status": "no_candidate", "decision": decision, "reason": "No mutation was proposed."}

    # A coordinated patch is still constrained to the small set of real
    # runtime hooks below. Source edits require exact digest-bound patches;
    # free-text policies are not executable. Keep the singular ``mutation`` shape
    # for backwards compatibility with existing Evolver outputs.
    raw_mutations = payload.get("mutations")
    if raw_mutations is not None:
        if payload.get("mutation") is not None:
            return None, {
                "status": "invalid_mutation",
                "reason": "Use either mutation or mutations, not both.",
            }
        if not isinstance(raw_mutations, list) or not raw_mutations or len(raw_mutations) > MAX_EVOLVER_OPERATIONS:
            return None, {
                "status": "invalid_mutation",
                "reason": f"coordinated mutations require 1..{MAX_EVOLVER_OPERATIONS} executable operations.",
            }
        dependency_analysis = analyze_evolver_dependencies(raw_mutations)
        if not dependency_analysis.get("dependency_pass"):
            return None, {
                "status": "invalid_coordinated_mutation",
                "reason": "coordinated operations failed dependency analysis.",
                "dependency_analysis": dependency_analysis,
            }
        child_snapshots: List[RegistrySnapshot] = []
        child_validations: List[Dict[str, Any]] = []
        for index, mutation in enumerate(raw_mutations):
            child_payload = dict(payload)
            child_payload.pop("mutations", None)
            child_payload["mutation"] = mutation
            child_result = dict(result)
            child_result["payload"] = child_payload
            child_snapshot, child_validation = build_candidate_snapshot_from_evolver(
                child_result,
                parent_id=parent_id,
                diagnosis_ids=diagnosis_ids,
            )
            if child_snapshot is None:
                return None, {
                    "status": "invalid_coordinated_mutation",
                    "index": index,
                    "child": child_validation,
                }
            child_snapshots.append(child_snapshot)
            child_validations.append(child_validation)

        merged_router_policy: Dict[str, Any] = {}
        merged_argument_policy: Dict[str, Any] = {}
        merged_tool_overrides: Dict[str, Dict[str, Any]] = {}
        merged_source_overrides: Dict[str, Dict[str, Any]] = {}
        for child in child_snapshots:
            merged_source_overrides.update(child.source_overrides)
            for key, value in child.router_policy.items():
                if key in merged_router_policy and merged_router_policy[key] != value:
                    return None, {
                        "status": "invalid_coordinated_mutation",
                        "reason": f"conflicting router policies for {key}",
                    }
                merged_router_policy[key] = copy.deepcopy(value)
            for key, value in child.argument_policy.items():
                if key in merged_argument_policy and merged_argument_policy[key] != value:
                    return None, {
                        "status": "invalid_coordinated_mutation",
                        "reason": f"conflicting argument policies for {key}",
                    }
                merged_argument_policy[key] = copy.deepcopy(value)
            for key, value in child.tool_overrides.items():
                if key in merged_tool_overrides and merged_tool_overrides[key] != value:
                    return None, {
                        "status": "invalid_coordinated_mutation",
                        "reason": f"conflicting tool overrides for {key}",
                    }
                merged_tool_overrides[key] = copy.deepcopy(value)

        merged_snapshot = RegistrySnapshot(
            snapshot_id=stable_id(
                "coordinated_snapshot",
                [parent_id, [child.model_dump(mode="json") for child in child_snapshots]],
            ),
            parent_snapshot_id=parent_id,
            created_at=utc_now(),
            router_policy=merged_router_policy,
            argument_policy=merged_argument_policy,
            tool_overrides=merged_tool_overrides,
            source_overrides=merged_source_overrides,
            base_tool_library_baseline_id=(
                str(os.environ.get("CFV_REGISTRY_BASELINE_ID") or "").strip() or None
            ),
            patch_ids=[patch_id for child in child_snapshots for patch_id in child.patch_ids],
            status="candidate",
            dependency_analysis=dependency_analysis,
        )
        return merged_snapshot, {
            "status": "candidate_created",
            "coordinated": True,
            "operation_count": len(child_snapshots),
            "dependency_analysis": dependency_analysis,
            "children": child_validations,
        }

    mutation = payload.get("mutation")
    if not isinstance(mutation, Mapping):
        return None, {"status": "invalid_mutation", "reason": "mutation must be an object."}
    action_text = str(mutation.get("action") or "").strip()
    try:
        action = EvolutionAction(action_text)
    except ValueError:
        return None, {
            "status": "unsupported_action",
            "action": action_text,
            "reason": "Mutation action is not part of the EvolutionAction contract.",
        }
    if action.value not in _EVOLVER_HARNESS_ACTIONS:
        return None, {
            "status": "unsupported_action",
            "action": action.value,
            "reason": "This mutation is not implemented by the isolated harness.",
        }

    target = mutation.get("target")
    target_ids = mutation.get("target_ids")
    if isinstance(target_ids, list):
        normalized_targets = [str(item).strip() for item in target_ids if str(item).strip()]
    else:
        normalized_targets = [str(target or "").strip()]
    if not normalized_targets:
        return None, {"status": "invalid_mutation", "action": action.value, "reason": "target is required."}

    before = mutation.get("before")
    if before is not None and not isinstance(before, Mapping):
        return None, {
            "status": "invalid_mutation",
            "action": action.value,
            "reason": "before must be an object or null for an executable mutation.",
        }
    after = mutation.get("after")
    if not isinstance(after, Mapping):
        return None, {
            "status": "invalid_mutation",
            "action": action.value,
            "reason": "after must be a machine-readable object for an executable mutation.",
        }
    after_dict = {str(key): value for key, value in after.items()}
    source_backed = action.value in SOURCE_BACKED_ACTIONS or (action == EvolutionAction.ADD_COMPOSITION_EDGE and "edits" in after_dict)
    if action == EvolutionAction.EDIT_TOOL_SOURCE or source_backed:
        try:
            expected_kind = SOURCE_BACKED_ACTIONS.get(action.value, "compose" if action == EvolutionAction.ADD_COMPOSITION_EDGE else None)
            if expected_kind and after_dict.get("kind") != expected_kind:
                raise ValueError("this operation requires an executable source patch of kind " + expected_kind)
            validate_source_edit(after_dict)
        except ValueError as exc:
            return None, {"status": "invalid_mutation", "action": action.value, "reason": str(exc)}
    elif action == EvolutionAction.UPDATE_TOOL_SCHEMA:
        allowed_fields = {"description", "parameters", "tags"}
        unknown_fields = sorted(set(after_dict) - allowed_fields)
        if unknown_fields:
            return None, {
                "status": "invalid_mutation",
                "action": action.value,
                "reason": f"tool schema mutation contains unsupported fields: {unknown_fields}",
            }
        if not any(field in after_dict for field in allowed_fields):
            return None, {
                "status": "invalid_mutation",
                "action": action.value,
                "reason": "tool schema mutation must change description, parameters, or tags.",
            }
        if "parameters" in after_dict:
            parameters = after_dict["parameters"]
            if (
                not isinstance(parameters, Mapping)
                or parameters.get("type") != "object"
                or not isinstance(parameters.get("properties"), Mapping)
            ):
                return None, {
                    "status": "invalid_mutation",
                    "action": action.value,
                    "reason": "parameters must be a JSON-schema object with type=object and a properties mapping.",
                }
            if "required" in parameters and not isinstance(parameters.get("required"), list):
                return None, {
                    "status": "invalid_mutation",
                    "action": action.value,
                    "reason": "parameters.required must be an array when present.",
                }
            if any(not isinstance(value, Mapping) for value in parameters.get("properties", {}).values()):
                return None, {
                    "status": "invalid_mutation",
                    "action": action.value,
                    "reason": "parameters.properties values must be JSON-schema objects.",
                }
        if "description" in after_dict and not isinstance(after_dict["description"], str):
            return None, {
                "status": "invalid_mutation",
                "action": action.value,
                "reason": "description must be a string.",
            }
        if "tags" in after_dict and (
            not isinstance(after_dict["tags"], list)
            or any(not isinstance(item, str) for item in after_dict["tags"])
        ):
            return None, {
                "status": "invalid_mutation",
                "action": action.value,
                "reason": "tags must be an array of strings.",
            }
    elif action == EvolutionAction.ADD_COMPOSITION_EDGE:
        if normalized_targets != ["candidate_comparison"] or after_dict != {
            "after": "evidence_acquisition",
            "before": "finish",
            "required": True,
        }:
            return None, {
                "status": "invalid_mutation",
                "action": action.value,
                "reason": "add_composition_edge only supports the required evidence_acquisition -> finish edge.",
            }
    elif action == EvolutionAction.UPDATE_RECOVERY_POLICY:
        if normalized_targets != ["low_signal_temporal_grounding"] or not valid_recovery_policy(after_dict):
            return None, {
                "status": "invalid_mutation",
                "action": action.value,
                "reason": (
                    "update_recovery_policy requires bounded failures 1..8, density 5..120, calls 1..4."
                ),
            }

    dependency_analysis = analyze_evolver_dependencies([mutation])
    if not dependency_analysis.get("dependency_pass"):
        return None, {
            "status": "invalid_mutation",
            "action": action.value,
            "reason": "mutation failed dependency analysis.",
            "dependency_analysis": dependency_analysis,
        }

    scope = {
        **{EvolutionAction(key): "single_tool" for key in SOURCE_BACKED_ACTIONS},
        EvolutionAction.UPDATE_TOOL_SCHEMA: "single_tool",
        EvolutionAction.ADD_COMPOSITION_EDGE: "workflow",
        EvolutionAction.UPDATE_RECOVERY_POLICY: "workflow",
        EvolutionAction.EDIT_TOOL_SOURCE: "single_tool",
    }[action]
    intent = EvolutionIntent(
        intent_id=stable_id("evolver_intent", [payload, diagnosis_ids or []]),
        diagnosis_ids=[str(item) for item in (diagnosis_ids or [])],
        scope=scope,
        operations=[PatchOperation(
            action=action,
            target_type="tool" if action == EvolutionAction.UPDATE_TOOL_SCHEMA else scope,
            target_ids=normalized_targets,
            before={str(key): value for key, value in before.items()} if isinstance(before, Mapping) else None,
            after=after_dict,
            reason=str(mutation.get("reason") or "Evolver replay-gated mutation."),
        )],
        expected_effects=[str(payload.get("expected_effect") or "")],
        risks=[str(payload.get("regression_risk") or "")],
        dependency_analysis=dependency_analysis,
    )
    snapshot = build_candidate_snapshot(intent, parent_id)
    return snapshot, {
        "status": "candidate_created",
        "action": action.value,
        "target_ids": normalized_targets,
        "intent_id": intent.intent_id,
        "dependency_analysis": dependency_analysis,
    }


def load_sample_pair(result_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with result_path.open("r", encoding="utf-8") as handle:
        run_result = json.load(handle)
    sample_path = result_path.parents[1] / "mix_sample_result.json"
    with sample_path.open("r", encoding="utf-8") as handle:
        sample = json.load(handle)
    return sample, run_result
