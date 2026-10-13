"""Audit Mix results, trajectories, evidence gates, and aggregate consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MIX_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MIX_DIR.parents[1]
_TERMINAL_SAMPLE_STATUSES = frozenset({"completed", "wrong", "unanswerable"})
sys.path.insert(0, str(MIX_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "RL"))

from evolution.analyzer import reconstruct_finish_evidence  # noqa: E402
from evolution.models import analyze_evolver_dependencies  # noqa: E402
from evolution.verifier import EvidenceVerifier, load_evolution_config  # noqa: E402
from mix_datasets import DatasetInventory, MixSample  # noqa: E402
from reward_metrics import _evaluate_rule_answer  # noqa: E402
from run_mix_batch import load_config, load_inventories  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_any_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values if float(value) > 0.0)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = max(0.0, min(1.0, float(percentile) / 100.0)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = rank - lower
    value = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
    return round(value, 6)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _query_requires_evidence_first(query: str) -> bool:
    """Lightweight mirror of the planner's evidence-first classifier.

    Keep the audit independent from ``planner.__init__``: importing the
    planner package pulls the full Transformers/scipy stack and makes a
    6920-sample read-only audit unnecessarily expensive and fragile.
    """
    if not query:
        return False
    q = str(query).lower()
    benchmark_markers = ("mlvu task type", "mvbench task type", "videomme", "video-mme", "lvbench", "egoschema")
    if any(marker in q for marker in benchmark_markers):
        return True
    exact_markers = (
        "exact", "exactly", "precise", "specific", "quoted", "total", "amount",
        "value", "values", "number", "numbers", "how much", "how many", "price",
        "prices", "figure", "figures",
    )
    evidence_markers = (
        "video", "clip", "screen", "frame", "shown", "showing", "visible",
        "subtitle", "transcript", "caption", "scene",
    )
    return any(marker in q for marker in exact_markers) and any(marker in q for marker in evidence_markers)


def _iter_steps(trajectory: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for raw_step in trajectory.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue
        yield raw_step
        child = raw_step.get("sub_trajectory")
        if isinstance(child, dict):
            yield from _iter_steps(child)
        for item in raw_step.get("sub_trajectories") or []:
            if isinstance(item, dict):
                yield from _iter_steps(item)


def _iter_trajectories(trajectory: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield the root trajectory and every recursively attached sub-trajectory."""
    yield trajectory
    for raw_step in trajectory.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue
        child = raw_step.get("sub_trajectory")
        if isinstance(child, dict):
            yield from _iter_trajectories(child)
        for item in raw_step.get("sub_trajectories") or []:
            if isinstance(item, dict):
                yield from _iter_trajectories(item)


def _workflow_summary(
    trajectory: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    tool_actions: Sequence[str],
    result_item: Mapping[str, Any],
) -> Dict[str, Any]:
    nodes = list(_iter_trajectories(trajectory))
    nested_nodes = [item for item in nodes[1:] if isinstance(item, Mapping)]
    depths = [_int(item.get("depth")) for item in nodes]
    parallel_steps = [
        step for step in steps
        if str(step.get("action") or "").strip() == "Parallel"
    ]
    parallel_concrete_batches = 0
    for step in parallel_steps:
        child_tools = []
        for item in step.get("observation_items") or []:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if tool and tool not in {"Parallel", "Tool_Search", "System", "Assistant_Note"}:
                child_tools.append(tool)
        if len(child_tools) > 1:
            parallel_concrete_batches += 1
    reward_features = result_item.get("reward_features") if isinstance(result_item.get("reward_features"), dict) else {}
    reported_subagent_iterations = _int(
        result_item.get("subagent_iterations")
        or reward_features.get("monitor_subagent_iterations")
    )
    tool_search_calls = sum(1 for name in tool_actions if name == "Tool_Search")
    subagent_observed = bool(nested_nodes) or reported_subagent_iterations > 0 or max(depths or [0]) > 0
    return {
        "trajectory_node_count": len(nodes),
        "nested_subtrajectory_count": len(nested_nodes),
        "max_depth": max(depths or [0]),
        "recursive_step_count": sum(bool(step.get("is_recursive")) for step in steps),
        "parallel_step_count": len(parallel_steps),
        "parallel_concrete_batch_count": parallel_concrete_batches,
        "tool_search_call_count": tool_search_calls,
        "reported_subagent_iterations": reported_subagent_iterations,
        "subagent_workflow_observed": subagent_observed,
        "subagent_trajectory_ids": [
            str(item.get("trajectory_id") or "")
            for item in nested_nodes
            if str(item.get("trajectory_id") or "").strip()
        ],
        "workflow_contract_pass": bool(
            str(trajectory.get("trajectory_id") or "").strip()
            and str(trajectory.get("registry_snapshot_id") or "").strip()
            and (not subagent_observed or bool(nested_nodes))
        ),
    }


def _tool_actions(trajectory: Mapping[str, Any]) -> List[str]:
    actions: List[str] = []
    for step in _iter_steps(trajectory):
        action = str(step.get("action") or "").strip()
        if action and action not in {"Finish", "System", "Assistant_Note"}:
            actions.append(action)
        if action == "Parallel":
            for item in step.get("observation_items") or []:
                if isinstance(item, dict):
                    tool = str(item.get("tool") or "").strip()
                    if tool and tool not in {"Parallel", "System", "Assistant_Note"}:
                        actions.append(tool)
    return actions


def _finish_steps(trajectory: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [step for step in _iter_steps(trajectory) if str(step.get("action") or "") == "Finish"]


def _tool_error_count(trajectory: Mapping[str, Any]) -> int:
    count = 0
    for step in _iter_steps(trajectory):
        for item in step.get("observation_items") or []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            result = str(item.get("result") or "").strip().lower()
            if status in {"error", "failed", "failure"} or result.startswith(("error", "failed", "no_valid_result")):
                count += 1
    return count


def _concrete_observation_items(steps: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    items: List[Mapping[str, Any]] = []
    for step in steps:
        if str(step.get("action") or "").strip() == "Finish":
            continue
        for item in step.get("observation_items") or []:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if tool and tool not in {"Parallel", "System", "Assistant_Note", "Evidence_Verifier"}:
                items.append(item)
    return items


def _observation_status(item: Mapping[str, Any]) -> str:
    status = str(item.get("status") or "").strip().lower()
    # The runtime deliberately blocks an exact duplicate request before the
    # underlying tool executes.  Older trajectories serialized that guard as
    # status=error; keep it visible as a controlled rejection rather than
    # misreporting a tool implementation failure.
    if str(item.get("reason") or "").strip().lower() == "duplicate_tool_execution_rejected":
        return "controlled_rejection"
    if status in {"error", "failed", "failure"}:
        return "error"
    if status in {"no_valid_result", "rejected", "invalid_request"}:
        # The tool executed and deliberately rejected an invalid/unsupported
        # request.  Keep this distinct from a runtime error so recovery logic
        # is visible in the audit instead of being counted as success.
        return "controlled_rejection"
    if status in {"success", "completed", "ok"}:
        return "success"
    result = item.get("result")
    if isinstance(result, str) and result.strip().lower().startswith(("error", "failed", "no_valid_result")):
        return "error"
    if isinstance(result, dict) and str(result.get("status") or "").strip().lower() in {"error", "failed", "failure"}:
        return "error"
    # Inspect_Frame historically omitted a top-level status even when it
    # returned image evidence. Treat a non-empty observation as executed.
    if result or item.get("image_evidence") or item.get("probes"):
        return "success"
    return "unknown"


def _tool_payload(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract a structured tool payload from a trajectory observation."""
    # Parallel observations are flattened into the tool fields themselves by
    # the trajectory serializer (``status``, ``fallback_used``,
    # ``index_diagnostics``, ...), while ``result`` remains a compact text
    # preview.  Prefer those fields so dense-vs-fallback auditing does not
    # mistake a successful structured result for an opaque string.
    if any(key in item for key in ("status", "fallback_used", "fallback_reason", "index_diagnostics")):
        return item
    candidates = [item.get("parsed_json"), item.get("result")]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, str):
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _semantic_retrieval_audit(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    semantic_items = [item for item in items if str(item.get("tool") or "").strip() == "Semantic_Retrieve"]
    fallback_items = []
    dense_items = []
    empty_index_items = []
    for item in semantic_items:
        payload = _tool_payload(item)
        if bool(payload.get("fallback_used")) or "semantic_fallback_lexical" in str(payload.get("result") or ""):
            fallback_items.append(item)
        elif payload.get("status") == "success":
            dense_items.append(item)
        diagnostics = payload.get("index_diagnostics")
        if diagnostics or "vector_index_empty_or_partial" in str(payload.get("fallback_reason") or ""):
            empty_index_items.append(item)
    return {
        "call_count": len(semantic_items),
        "dense_success_count": len(dense_items),
        "lexical_fallback_count": len(fallback_items),
        "empty_or_partial_index_diagnostic_count": len(empty_index_items),
    }


def _four_agent_audit(
    trajectory: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    tool_actions: Sequence[str],
    result_item: Mapping[str, Any],
    debug_events: Sequence[Mapping[str, Any]],
    query: str,
    launcher_text: str,
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    """Audit four independent LLM/VLM roles without conflating them with subagents."""
    llm_events = [
        event for event in debug_events
        if isinstance(event, dict) and event.get("type") == "llm_iteration"
    ]
    planner_events = [
        event for event in llm_events
        if str(event.get("agent_role") or "planner").strip().lower() == "planner"
    ]
    parse_failures = [event for event in planner_events if event.get("parse_success") is False]
    role_turns = {
        role: [
            event for event in debug_events
            if event.get("type") == "agent_llm_turn"
            and str(event.get("agent_role") or "").strip().lower() == role
        ]
        for role in ("observer", "reflector")
    }
    kernel_contract = next(
        (
            event.get("roles")
            for event in debug_events
            if event.get("type") == "agent_kernel_contract"
            and isinstance(event.get("roles"), dict)
        ),
        {},
    )
    concrete_items = _concrete_observation_items(steps)
    semantic_audit = _semantic_retrieval_audit(concrete_items)
    observation_statuses = [_observation_status(item) for item in concrete_items]
    verifier_events = [
        event for event in debug_events
        if isinstance(event, dict)
        and (
            event.get("event_type") == "VerifierDecisionEvent"
            or event.get("type") == "verifier_decision"
        )
    ]
    evidence_required = bool(_query_requires_evidence_first(query))
    finish_present = bool(steps and str(steps[-1].get("action") or "") == "Finish")
    planner = {
        "implementation": "LLM Planner / ReAct controller",
        "kernel": kernel_contract.get("planner", {}),
        "observed": bool(planner_events),
        "llm_iteration_count": len(planner_events),
        "parse_success_count": len(planner_events) - len(parse_failures),
        "parse_failure_count": len(parse_failures),
        "action_sequence": [str(step.get("action") or "") for step in steps],
        "contract_pass": bool(planner_events) and not parse_failures and finish_present,
    }
    observer = {
        "implementation": "LLM/VLM Observer Agent + constrained tool runtime",
        "kernel": kernel_contract.get("observer", {}),
        "agent_turn_count": len(role_turns["observer"]),
        "agent_schema_pass_count": sum(bool(event.get("schema_valid")) for event in role_turns["observer"]),
        # A valid ReAct trajectory may finish without executing a tool.  In
        # that case the Observer turn is still a real role turn; requiring a
        # concrete observation would misclassify a deliberate zero-tool path.
        "observed": bool(role_turns["observer"]),
        "executed_tool_call_count": len(concrete_items),
        "successful_tool_call_count": sum(status == "success" for status in observation_statuses),
        "error_tool_call_count": sum(status == "error" for status in observation_statuses),
        "controlled_rejection_count": sum(status == "controlled_rejection" for status in observation_statuses),
        "unknown_status_count": sum(status == "unknown" for status in observation_statuses),
        "executed_tools": [str(item.get("tool") or "") for item in concrete_items],
        "actual_video_clip_qa_call": "Video_Clip_QA direct clip attempt" in launcher_text,
        "fallback_observed": _video_clip_qa_fallback_observed(launcher_text),
        "contract_pass": (
            bool(role_turns["observer"])
            and all(bool(event.get("schema_valid")) for event in role_turns["observer"])
            and not any(status == "unknown" for status in observation_statuses)
        ),
    }
    reflector = {
        "implementation": "LLM Reflector Agent + deterministic safety gate",
        "kernel": kernel_contract.get("reflector", {}),
        "agent_turn_count": len(role_turns["reflector"]),
        "agent_schema_pass_count": sum(bool(event.get("schema_valid")) for event in role_turns["reflector"]),
        "required_for_query": evidence_required,
        "observed": bool(role_turns["reflector"]),
        "decision_count": len(verifier_events),
        "blocked_finish_count": sum(not bool(event.get("can_finish")) for event in verifier_events),
        "accepted_finish_count": sum(bool(event.get("can_finish")) for event in verifier_events),
        "last_can_finish": verifier_events[-1].get("can_finish") if verifier_events else None,
        "missing_information": list(verifier_events[-1].get("missing_information") or []) if verifier_events else [],
        "unresolved_claims": list(verifier_events[-1].get("unresolved_claims") or []) if verifier_events else [],
        "evidence_can_finish": evidence.get("can_finish"),
        "contract_pass": (
            bool(role_turns["reflector"])
            and all(bool(event.get("schema_valid")) for event in role_turns["reflector"])
            and (bool(verifier_events) if evidence_required else True)
        ),
        "status": "pass" if (
            bool(role_turns["reflector"])
            and all(bool(event.get("schema_valid")) for event in role_turns["reflector"])
            and (bool(verifier_events) if evidence_required else True)
        ) else "missing",
    }
    evolver = {
        "implementation": "online boundary LLM Evolver Agent + replay gate",
        "kernel": {},
        "observed": False,
        # Evolver is a run-level online role. It is invoked at a completed
        # worker-group boundary, so it is not required to have a turn for
        # every individual sample record.
        "execution_scope": "worker_group_boundary",
        "required_for_current_audit": False,
        "report_count": 0,
        "replay_statuses": [],
        "candidate_replay_statuses": [],
        "contract_pass": False,
        "status": "not_run",
    }
    roles = {
        "Planner": planner,
        "Observer": observer,
        "Reflector": reflector,
        "Evolver": evolver,
    }
    # Planner/Observer/Reflector are per-sample roles. Evolver is a separate
    # online boundary role and is audited from the cycle reports below.
    online_roles = ("Planner", "Observer", "Reflector")
    contract_pass = all(bool(roles[name].get("contract_pass")) for name in online_roles)
    return {
        "roles": roles,
        "observed_role_count": sum(bool(role.get("observed")) for role in roles.values()),
        "four_agent_contract_pass": contract_pass and bool(evolver.get("contract_pass")),
        "online_role_contract_pass": contract_pass,
        "boundary_roles": ["Evolver"],
        "status": "pass" if contract_pass else ("partial" if any(bool(role.get("observed")) for role in roles.values()) else "missing"),
        "subagent_route_observed": bool(_workflow_summary(trajectory, steps, tool_actions, result_item).get("subagent_workflow_observed")),
        "semantic_retrieval": semantic_audit,
        "kernel_contract": kernel_contract,
    }


def _refresh_four_agent_contract(record: Dict[str, Any]) -> None:
    audit = record.get("four_agent_audit")
    if not isinstance(audit, dict):
        return
    roles = audit.get("roles") if isinstance(audit.get("roles"), dict) else {}
    online_roles = ("Planner", "Observer", "Reflector")
    contract_pass = all(bool((roles.get(name) or {}).get("contract_pass")) for name in online_roles)
    audit["four_agent_contract_pass"] = contract_pass and bool((roles.get("Evolver") or {}).get("contract_pass"))
    audit["online_role_contract_pass"] = contract_pass
    audit["boundary_roles"] = ["Evolver"]
    audit["observed_role_count"] = sum(bool((roles.get(name) or {}).get("observed")) for name in ("Planner", "Observer", "Reflector", "Evolver"))
    audit["status"] = "pass" if contract_pass else ("partial" if audit["observed_role_count"] else "missing")


def _log_markers(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    patterns = (
        "Traceback",
        "CUDA out of memory",
        "Memory cgroup out of memory",
        "LLVM",
        "SIGABRT",
        "return_code=-6",
        "returncode=-6",
        "return_code=-9",
        "returncode=-9",
        "Planner Runtime Fallback",
        "Video_Clip_QA video decoding failed",
        "retrying with 32 sampled frames",
        "Serialized GPU Video_Clip_QA",
    )
    return [pattern for pattern in patterns if pattern in text]


def _video_clip_qa_fallback_observed(launcher_text: str) -> bool:
    return any(
        marker in launcher_text
        for marker in (
            "Planner Runtime Fallback",
            "Video_Clip_QA video decoding failed",
            "retrying with 32 sampled frames",
        )
    )


def _failure_class(result_item: Optional[Mapping[str, Any]]) -> str:
    """Separate scheduler placeholders from failures with a sample trajectory.

    A killed reusable worker leaves placeholder ``mix_sample_result.json`` files
    for samples that were queued behind it.  Those records are useful for
    coverage accounting, but they are not independent model failures and have
    no trajectory to audit.
    """
    item = result_item or {}
    status = str(item.get("status") or "missing").strip().lower()
    raw_answer = " ".join(
        str(item.get(key) or "")
        for key in ("raw_final_answer", "final_answer", "error", "message")
    ).lower()
    if str(item.get("failure_class") or "").strip().lower() == "oom" or any(
        marker in raw_answer
        for marker in (
            "out of memory",
            "cuda out of memory",
            "torch.outofmemoryerror",
            "cudnn_status_alloc_failed",
            "cublas_status_alloc_failed",
            "cannot allocate memory",
        )
    ):
        return "oom"
    if "error: evidence_unanswerable" in raw_answer:
        return "evidence_unanswerable"
    if status == "unanswerable":
        return "evidence_unanswerable"
    if status in {"completed", "wrong"}:
        return "participated"
    if status == "failed":
        error = str(item.get("error") or "").strip().lower()
        if "did not produce a per-sample result" in error:
            return "scheduler_worker_placeholder"
        return "actual_sample_failure"
    return "missing"


def _is_evidence_unanswerable(
    result_item: Optional[Mapping[str, Any]],
    run_result: Optional[Mapping[str, Any]] = None,
) -> bool:
    values = []
    for payload in (result_item or {}, run_result or {}):
        if not isinstance(payload, Mapping):
            continue
        values.extend(
            str(payload.get(key) or "")
            for key in ("raw_final_answer", "error", "message", "final_answer")
        )
        agent = payload.get("agent")
        if isinstance(agent, Mapping):
            values.append(str(agent.get("final_answer") or ""))
    return any("error: evidence_unanswerable" in value.lower() for value in values)


def _sample_audit(
    sample: MixSample,
    result_item: Optional[Dict[str, Any]],
    verifier: EvidenceVerifier,
    expected_mode: str,
) -> Dict[str, Any]:
    result_path = sample.workspace_root / "mix_sample_result.json"
    run_path = sample.workspace_root / "results" / "run_result.json"
    record: Dict[str, Any] = {
        "sample_uid": sample.sample_uid,
        "dataset_name": sample.dataset_name,
        "subset_name": sample.subset_name,
        "dataset_scope": (
            {
                "subtitle_mode": str(sample.official_meta.get("subtitle_mode") or ""),
                "external_subtitles_used": bool(sample.official_meta.get("external_subtitles_used", False)),
                "input_contract": dict(sample.official_meta.get("input_contract") or {}),
            }
            if sample.dataset_name == "videomme"
            else {"input_contract": dict(sample.official_meta.get("input_contract") or {})}
        ),
        "result_path": str(result_path),
        "run_result_path": str(run_path),
        "result_file_present": result_path.exists(),
        "run_result_file_present": run_path.exists(),
        "status": str((result_item or {}).get("status") or "missing"),
        "failure_class": _failure_class(result_item),
        "accuracy_eligible": (
            str((result_item or {}).get("status") or "").strip().lower()
            in _TERMINAL_SAMPLE_STATUSES
            and _failure_class(result_item) != "oom"
        ),
        "outcome_class": (
            "evidence_unanswerable"
            if _is_evidence_unanswerable(result_item)
            else "pending"
        ),
        "acc": _float((result_item or {}).get("acc")),
        "elapsed_seconds": _float((result_item or {}).get("elapsed_seconds")),
        "cache_stats": {},
        "trajectory": {},
        "workflow": {},
        "protocol": {},
        "evidence": {},
        "evolution": {"status": "not_run"},
        "four_agent_audit": {"status": "missing", "four_agent_contract_pass": False},
        "inference_input": {},
        "gpu_telemetry": {"status": "not_available"},
        "tool_library": {"status": "not_available", "baseline_contract_pass": False},
        "log_markers": _log_markers(sample.workspace_root / "launcher.log"),
    }
    reported_acc = _float((result_item or {}).get("acc"))
    raw_final_answer = str((result_item or {}).get("raw_final_answer") or "")
    score_check = _evaluate_rule_answer(
        final_answer=raw_final_answer,
        ground_truth=sample.ground_truth,
        options=sample.options,
        dataset_name=sample.dataset_name,
    )
    truthful_acc = 1.0 if bool(score_check.get("answer_correct")) else 0.0
    record["reported_acc"] = reported_acc
    record["acc"] = truthful_acc
    record["score_contract"] = {
        "reported_acc": reported_acc,
        "recomputed_acc": truthful_acc,
        "raw_final_answer": raw_final_answer,
        "predicted_label": str(score_check.get("predicted_label") or ""),
        "ground_truth_labels": list(score_check.get("ground_truth_labels") or []),
        "reported_acc_matches_recomputed": reported_acc == truthful_acc,
    }
    metadata = sample.official_meta if isinstance(sample.official_meta, Mapping) else {}
    annotation_time_reference = str(metadata.get("time_reference") or "").strip()
    input_contract = dict(metadata.get("input_contract") or {})
    query_text = str(sample.prompt or "")
    query_lower = query_text.lower()
    # A normal LVBench question often contains the same interval in its own
    # wording. Only the explicit metadata suffix is an annotation disclosure.
    query_has_annotation_window = "reference time window (dataset metadata" in query_lower
    query_has_answer_field_marker = bool(
        re.search(
            r"(?:[\"`])?(?:ground_truth|ground_truth_option|evaluator_output|correct_answer)(?:[\"`])?\s*[:=]",
            query_lower,
        )
    )
    record["inference_input"] = {
        "query_source": "sample.prompt",
        "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "annotation_time_reference_available": bool(annotation_time_reference),
        "annotation_time_reference_used_for_inference": bool(
            metadata.get("time_reference_used_for_inference", False)
        ),
        "annotation_time_reference_scope": str(
            metadata.get("time_reference_inference_scope") or "annotation_only_not_used"
        ),
        "annotation_time_reference_is_oracle_excluded": not bool(
            metadata.get("time_reference_used_for_inference", False)
        ),
        "input_contract": input_contract,
        "annotation_time_reference_disclosed_in_query": query_has_annotation_window,
        "answer_field_marker_disclosed_in_query": query_has_answer_field_marker,
        "answer_leakage_contract_pass": not (
            query_has_annotation_window
            or query_has_answer_field_marker
            or bool(metadata.get("time_reference_used_for_inference", False))
        ),
    }
    if result_item is None or not run_path.exists():
        record["protocol"]["audit_status"] = "not_available"
        return record

    run_result = _read_json(run_path) or {}
    if _is_evidence_unanswerable(result_item, run_result):
        # Keep the raw scheduler payload available in ``failure_class`` and
        # score_contract, but expose the audited outcome explicitly instead
        # of calling a controlled evidence rejection merely ``wrong``.
        record["status"] = "unanswerable"
        record["outcome_class"] = "evidence_unanswerable"
    inference_query = str(run_result.get("query") or sample.prompt or "")
    record["inference_input"]["runtime_query_matches_sample_prompt"] = (
        inference_query == query_text
    )
    record["inference_input"]["runtime_query_has_answer_field_marker"] = bool(
        re.search(
            r"(?:[\"`])?(?:ground_truth|ground_truth_option|evaluator_output|correct_answer)(?:[\"`])?\s*[:=]",
            inference_query.lower(),
        )
    )
    runtime_contract = dict(
        (run_result.get("metrics") or {}).get("benchmark_input_contract") or {}
    )
    expected_audio = str(
        input_contract.get("audio_track") or input_contract.get("audio") or ""
    ).strip().lower()
    expected_transcript = str(
        input_contract.get("audio_transcript") or ""
    ).strip().lower()
    runtime_audio_policy = str(
        runtime_contract.get("audio_transcript_policy") or ""
    ).strip().lower()
    expected_transcript_allowed = expected_transcript in {
        "allowed",
        "included",
        "provided",
        "derived_allowed",
        "derived-allowed",
        "derived_from_source_audio",
        "derived-from-source-audio",
    }
    expected_track_preserved = expected_audio not in {
        "excluded",
        "none",
        "not_allowed",
        "not-allowed",
        "visual_only",
        "visual-only",
    }
    transcript_contract_pass = (
        runtime_audio_policy == "included"
        if expected_transcript_allowed
        else runtime_audio_policy == "excluded"
    )
    runtime_track_policy = str(
        runtime_contract.get("audio_track_policy") or ""
    ).strip().lower()
    track_contract_pass = (
        not expected_track_preserved
        or runtime_track_policy in {"preserved", "preserved_source", "included"}
    )
    audio_contract_pass = bool(track_contract_pass and transcript_contract_pass)
    record["inference_input"]["runtime_input_contract"] = runtime_contract
    record["inference_input"]["audio_contract_pass"] = bool(audio_contract_pass)
    record["inference_input"]["answer_leakage_contract_pass"] = bool(
        record["inference_input"].get("answer_leakage_contract_pass")
        and record["inference_input"].get("runtime_query_matches_sample_prompt")
        and not record["inference_input"].get("runtime_query_has_answer_field_marker")
        and audio_contract_pass
    )
    agent = run_result.get("agent") if isinstance(run_result.get("agent"), dict) else {}
    trajectory = agent.get("trajectory") if isinstance(agent.get("trajectory"), dict) else {}
    steps = list(_iter_steps(trajectory))
    finish_steps = _finish_steps(trajectory)
    final_finish = finish_steps[-1] if finish_steps else {}
    finish_args = final_finish.get("action_args") if isinstance(final_finish.get("action_args"), dict) else {}
    answer = str(
        finish_args.get("answer")
        or finish_args.get("result")
        or trajectory.get("final_answer")
        or agent.get("final_answer")
        or ""
    )
    debug_events = trajectory.get("debug_events") if isinstance(trajectory.get("debug_events"), list) else []
    llm_events = [item for item in debug_events if isinstance(item, dict) and item.get("type") == "llm_iteration"]
    parse_failures = [item for item in llm_events if item.get("parse_success") is False]
    terminal_evidence_events = [
        item for item in debug_events
        if isinstance(item, dict)
        and item.get("event_type") == "EvidenceContractUnanswerableEvent"
    ]
    actions = [str(step.get("action") or "") for step in steps if str(step.get("action") or "")]
    tool_actions = _tool_actions(trajectory)
    workflow = _workflow_summary(trajectory, steps, tool_actions, result_item)
    launcher_text = ""
    try:
        launcher_text = (sample.workspace_root / "launcher.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    cache_stats = run_result.get("metrics", {}).get("cache_stats", {})
    record["cache_stats"] = dict(cache_stats) if isinstance(cache_stats, dict) else {}
    record["timing_breakdown_seconds"] = dict(run_result.get("metrics", {}).get("timing_breakdown_seconds", {}) or {})
    gpu_telemetry = run_result.get("metrics", {}).get("gpu_telemetry", {})
    record["gpu_telemetry"] = dict(gpu_telemetry) if isinstance(gpu_telemetry, dict) else {"status": "not_available"}
    tool_library = run_result.get("metrics", {}).get("tool_library", {})
    record["tool_library"] = dict(tool_library) if isinstance(tool_library, dict) else {"status": "not_available"}
    harness = run_result.get("metrics", {}).get("evolution_harness", {})
    record["evolution_harness"] = dict(harness) if isinstance(harness, dict) else {}
    tool_library_mutated = bool(
        isinstance(harness, dict)
        and (
            harness.get("mutation_applied")
            or harness.get("registry_mutations_applied")
            or harness.get("tool_overrides")
        )
    )
    record["tool_library"]["baseline_contract_pass"] = bool(
        record["tool_library"].get("baseline_id")
        and record["tool_library"].get("baseline_path")
        and (
            bool(record["tool_library"].get("schema_match"))
            or tool_library_mutated
        )
    )
    record["trajectory"] = {
        "trajectory_id": trajectory.get("trajectory_id"),
        "registry_snapshot_id": trajectory.get("registry_snapshot_id"),
        "step_count": len(steps),
        "actions": actions,
        "tool_actions": tool_actions,
        "finish_attempt_count": len(finish_steps),
        "terminal_action": actions[-1] if actions else "",
        "answer": answer,
        "agent_final_answer": str(agent.get("final_answer") or ""),
        "run_success": bool(run_result.get("success")),
        "trajectory_success": bool(trajectory.get("success")),
        "planner_success": bool(result_item.get("planner_success")),
        "parse_event_count": len(llm_events),
        "parse_failure_count": len(parse_failures),
        "tool_error_count": _tool_error_count(trajectory),
    }
    record["workflow"] = {
        **workflow,
        "semantic_retrieval": _semantic_retrieval_audit(_concrete_observation_items(steps)),
        "actual_video_clip_qa_call": "Video_Clip_QA direct clip attempt" in launcher_text,
        "video_clip_qa_fallback_observed": _video_clip_qa_fallback_observed(launcher_text),
        "gpu_video_clip_qa_serialization_observed": "Serialized GPU Video_Clip_QA" in launcher_text,
    }
    expected_direct = expected_mode == "planner_direct_answer"
    record["protocol"] = {
        "expected_mode": expected_mode,
        "run_success_consistent": bool(run_result.get("success")) == bool(trajectory.get("success")) == bool(result_item.get("planner_success")),
        "terminal_finish": bool(actions) and actions[-1] == "Finish",
        "finish_has_chain_complete": bool(finish_args.get("chain_complete")) if finish_steps else False,
        "finish_has_completion_basis": bool(str(finish_args.get("completion_basis") or "").strip()) if finish_steps else False,
        "answer_consistent": answer == str(agent.get("final_answer") or "") == str(result_item.get("raw_final_answer") or ""),
        "all_parse_events_success": not parse_failures,
        "direct_single_finish": len(finish_steps) == 1 if expected_direct else None,
        "direct_no_tool_calls": not tool_actions if expected_direct else None,
        "registry_snapshot_present": bool(str(trajectory.get("registry_snapshot_id") or "").strip()),
        "tool_library_baseline_present": bool(record.get("tool_library", {}).get("baseline_contract_pass")),
        "workflow_contract_pass": bool(workflow.get("workflow_contract_pass")),
        "unanswerable_terminal": bool(
            record.get("outcome_class") == "evidence_unanswerable"
            and terminal_evidence_events
            and str(agent.get("final_answer") or "").lower().startswith("error: evidence_unanswerable")
        ),
        "protocol_pass": False,
    }
    if record.get("outcome_class") == "evidence_unanswerable":
        # An evidence-contract refusal is a valid terminal outcome. It must
        # preserve the role/runtime contracts and a structured terminal event,
        # but it must not be forced through the normal Finish/evidence gate.
        protocol_fields = [
            "run_success_consistent",
            "unanswerable_terminal",
            "answer_consistent",
            "all_parse_events_success",
            "registry_snapshot_present",
            "tool_library_baseline_present",
            "workflow_contract_pass",
        ]
    else:
        protocol_fields = [
            "run_success_consistent",
            "terminal_finish",
            "finish_has_chain_complete",
            "finish_has_completion_basis",
            "answer_consistent",
            "all_parse_events_success",
            "registry_snapshot_present",
            "tool_library_baseline_present",
            "workflow_contract_pass",
        ]
    if expected_direct:
        protocol_fields.extend(["direct_single_finish", "direct_no_tool_calls"])
    record["protocol"]["protocol_pass"] = all(bool(record["protocol"].get(key)) for key in protocol_fields)

    try:
        evidence_slots, reconstructed_answer, completion_basis = reconstruct_finish_evidence(trajectory)
        verification = verifier.verify_answerability(
            evidence_slots=evidence_slots,
            steps=steps,
            answer=reconstructed_answer or answer,
            completion_basis=completion_basis,
            query=str(run_result.get("query") or sample.prompt),
        )
        record["evidence"] = {
            "slot_count": len(evidence_slots),
            "completion_basis": completion_basis,
            "progress_score": verification.progress_score,
            "claim_coverage": verification.claim_coverage,
            "direct_grounding": verification.direct_grounding,
            "contradiction_score": verification.contradiction_score,
            "cross_modal_support": verification.cross_modal_support,
            "can_finish": verification.can_finish,
            "premise_mismatch": bool(getattr(verification, "premise_mismatch", False)),
            "missing_information": list(verification.missing_information),
            "unresolved_claims": list(verification.unresolved_claims),
        }
    except Exception as exc:
        record["evidence"] = {"audit_error": str(exc), "can_finish": False}

    if record.get("outcome_class") == "pending":
        record["outcome_class"] = (
            "correct" if truthful_acc == 1.0
            else "wrong" if str(record.get("status") or "").lower() in {"completed", "wrong"}
            else "execution_failure"
        )

    record["four_agent_audit"] = _four_agent_audit(
        trajectory=trajectory,
        steps=steps,
        tool_actions=tool_actions,
        result_item=result_item,
        debug_events=debug_events,
        query=inference_query,
        launcher_text=launcher_text,
        evidence=record.get("evidence") or {},
    )

    if str(sample.subset_name).strip().lower() == "needle":
        record["dynamic_task"] = {
            "requires_video_clip_qa": True,
            "video_clip_qa_called": "Video_Clip_QA" in tool_actions,
            "answer_support_protocol_pass": bool(record.get("evidence", {}).get("can_finish")) and "Video_Clip_QA" in tool_actions,
        }
    return record


def _summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = Counter(str(item.get("status") or "missing") for item in records)
    failure_classes = Counter(str(item.get("failure_class") or "missing") for item in records)
    participated = [
        item for item in records
        if str(item.get("status") or "").lower() in _TERMINAL_SAMPLE_STATUSES
    ]
    attempted = [item for item in records if str(item.get("status") or "").lower() in {*_TERMINAL_SAMPLE_STATUSES, "failed"}]
    times = [_float(item.get("elapsed_seconds")) for item in attempted if _float(item.get("elapsed_seconds")) > 0]
    tool_call_counts = [
        len((item.get("trajectory") or {}).get("tool_actions") or [])
        for item in attempted
    ]
    successful_tool_calls = sum(
        int(
            ((item.get("four_agent_audit") or {}).get("roles") or {})
            .get("Observer", {})
            .get("successful_tool_call_count", 0)
            or 0
        )
        for item in attempted
    )
    controlled_tool_calls = sum(
        int(
            ((item.get("four_agent_audit") or {}).get("roles") or {})
            .get("Observer", {})
            .get("controlled_rejection_count", 0)
            or 0
        )
        for item in attempted
    )
    error_tool_calls = sum(
        int(
            ((item.get("four_agent_audit") or {}).get("roles") or {})
            .get("Observer", {})
            .get("error_tool_call_count", 0)
            or 0
        )
        for item in attempted
    )
    gpu_records = [
        item.get("gpu_telemetry") or {}
        for item in attempted
        if str((item.get("gpu_telemetry") or {}).get("status") or "") == "ok"
    ]
    cache = Counter()
    protocol_pass = 0
    evidence_pass = 0
    evidence_contract_pass = 0
    for item in records:
        cache.update(item.get("cache_stats") or {})
        if item.get("protocol", {}).get("protocol_pass"):
            protocol_pass += 1
        if item.get("evidence", {}).get("can_finish"):
            evidence_pass += 1
        if item.get("evidence", {}).get("can_finish") or (
            item.get("outcome_class") == "evidence_unanswerable"
            and item.get("protocol", {}).get("unanswerable_terminal")
        ):
            evidence_contract_pass += 1
    workflow_pass = sum(bool(item.get("workflow", {}).get("workflow_contract_pass")) for item in records)
    subagent_observed = sum(bool(item.get("workflow", {}).get("subagent_workflow_observed")) for item in records)
    leakage_pass = sum(bool(item.get("inference_input", {}).get("answer_leakage_contract_pass")) for item in records)
    return {
        "total_expected": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "scheduler_worker_placeholder_count": failure_classes.get("scheduler_worker_placeholder", 0),
        "actual_sample_failure_count": failure_classes.get("actual_sample_failure", 0),
        "oom_failure_count": failure_classes.get("oom", 0),
        "oom_excluded_from_accuracy_count": failure_classes.get("oom", 0),
        "evidence_unanswerable_count": sum(
            item.get("outcome_class") == "evidence_unanswerable"
            for item in records
        ),
        "participated": len(participated),
        "attempted": len(attempted),
        "attempt_accuracy": round(sum(_float(item.get("acc")) for item in participated) / len(attempted), 6) if attempted else None,
        "attempt_mean_elapsed_seconds": round(statistics.mean(times), 6) if times else None,
        "attempt_samples_per_hour": round(len(times) * 3600 / sum(times), 6) if times else None,
        "runtime_metric_scope": "all_attempted_including_failed",
        "accuracy_denominator": len(participated),
        "accuracy": round(sum(_float(item.get("acc")) for item in participated) / len(participated), 6) if participated else None,
        "mean_elapsed_seconds": round(statistics.mean(times), 6) if times else None,
        "median_elapsed_seconds": round(statistics.median(times), 6) if times else None,
        "p95_elapsed_seconds": _percentile(times, 95.0),
        "total_tool_calls": sum(tool_call_counts),
        "tool_calls_per_sample": round(statistics.mean(tool_call_counts), 6) if tool_call_counts else None,
        "zero_tool_sample_count": sum(count == 0 for count in tool_call_counts),
        "tool_success_rate": round(
            successful_tool_calls / max(1, successful_tool_calls + controlled_tool_calls + error_tool_calls),
            6,
        ) if tool_call_counts else None,
        "invalid_call_rate": round(
            (controlled_tool_calls + error_tool_calls) / max(1, sum(tool_call_counts)),
            6,
        ) if tool_call_counts else None,
        "trajectory_protocol_pass": protocol_pass,
        "evidence_can_finish": evidence_pass,
        "evidence_contract_pass": evidence_contract_pass,
        "premise_mismatch_count": sum(
            bool((item.get("evidence") or {}).get("premise_mismatch"))
            for item in records
        ),
        "workflow_contract_pass": workflow_pass,
        "subagent_workflow_observed": subagent_observed,
        "answer_leakage_contract_pass": leakage_pass,
        "answer_leakage_contract_fail": len(records) - leakage_pass,
        "four_agent_contract_pass": sum(bool(item.get("four_agent_audit", {}).get("four_agent_contract_pass")) for item in records),
        "online_role_contract_pass": sum(bool(item.get("four_agent_audit", {}).get("online_role_contract_pass")) for item in records),
        "planner_observed": sum(bool(item.get("four_agent_audit", {}).get("roles", {}).get("Planner", {}).get("observed")) for item in records),
        "observer_observed": sum(bool(item.get("four_agent_audit", {}).get("roles", {}).get("Observer", {}).get("observed")) for item in records),
        "controlled_tool_rejections": sum(int(item.get("four_agent_audit", {}).get("roles", {}).get("Observer", {}).get("controlled_rejection_count", 0) or 0) for item in records),
        "reflector_observed": sum(bool(item.get("four_agent_audit", {}).get("roles", {}).get("Reflector", {}).get("observed")) for item in records),
        "evolver_observed": sum(bool(item.get("four_agent_audit", {}).get("roles", {}).get("Evolver", {}).get("observed")) for item in records),
        "actual_video_clip_qa_calls": sum(bool(item.get("workflow", {}).get("actual_video_clip_qa_call")) for item in records),
        "semantic_retrieval_calls": sum(int((item.get("workflow", {}).get("semantic_retrieval") or {}).get("call_count", 0) or 0) for item in records),
        "semantic_dense_successes": sum(int((item.get("workflow", {}).get("semantic_retrieval") or {}).get("dense_success_count", 0) or 0) for item in records),
        "semantic_lexical_fallbacks": sum(int((item.get("workflow", {}).get("semantic_retrieval") or {}).get("lexical_fallback_count", 0) or 0) for item in records),
        "semantic_empty_index_diagnostics": sum(int((item.get("workflow", {}).get("semantic_retrieval") or {}).get("empty_or_partial_index_diagnostic_count", 0) or 0) for item in records),
        "video_clip_qa_fallback_samples": sum(bool(item.get("workflow", {}).get("video_clip_qa_fallback_observed")) for item in records),
        "gpu_video_clip_qa_serialization_samples": sum(bool(item.get("workflow", {}).get("gpu_video_clip_qa_serialization_observed")) for item in records),
        "cache_stats": dict(sorted(cache.items())),
        "log_marker_samples": sum(bool(item.get("log_markers")) for item in records),
        "gpu_telemetry_observed_samples": len(gpu_records),
        "gpu_avg_utilization_percent": round(
            statistics.mean(float(item.get("avg_utilization_percent") or 0.0) for item in gpu_records), 6
        ) if gpu_records else None,
        "gpu_peak_utilization_percent": round(
            max(float(item.get("peak_utilization_percent") or 0.0) for item in gpu_records), 6
        ) if gpu_records else None,
        "gpu_peak_memory_used_gib": round(
            max(float(item.get("peak_memory_used_gib") or 0.0) for item in gpu_records), 6
        ) if gpu_records else None,
        "tool_library_baseline_samples": sum(
            bool((item.get("tool_library") or {}).get("baseline_contract_pass"))
            for item in records
        ),
        "tool_library_schema_changed_samples": sum(
            bool((item.get("tool_library") or {}).get("changed_tools"))
            or bool((item.get("tool_library") or {}).get("added_tools"))
            or bool((item.get("tool_library") or {}).get("removed_tools"))
            for item in records
        ),
        "tool_library_scope_hidden_samples": sum(
            bool((item.get("tool_library") or {}).get("scoped_hidden_tools"))
            for item in records
        ),
    }


_UNSUPPORTED_EDGE_TOOL_NAMES = frozenset({
    "Action_Recognition",
    "Audio_Event_Detect",
    "Cross_Modal_Verify",
    "Emotion_Analysis",
    "Entity_Track",
    "OCR_Extract",
    "Speaker_Diarization",
})


def _declared_runtime_tool_names(workspaces_root: Optional[Path] = None) -> List[str]:
    """Read active concrete tool names from the immutable baseline.

    Source-code class discovery is not enough: a class can remain in the
    implementation module while deliberately being removed from the formal
    registry because its backend is unavailable.  The frozen baseline is the
    authoritative active-library contract for this batch.
    """

    if workspaces_root is not None:
        baseline = _read_json(Path(workspaces_root) / "_audit" / "tool_library_baseline.json")
        if isinstance(baseline, dict) and isinstance(baseline.get("tools"), list):
            return sorted(
                str(item.get("name") or "").strip()
                for item in baseline["tools"]
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            )

    names: set[str] = set()
    for path in sorted((PROJECT_ROOT / "RL" / "toolkit" / "impl").glob("*.py")):
        if path.name == "meta_tools.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names.update(re.findall(r'^\s*name:\s*str\s*=\s*"([^"]+)"', text, flags=re.MULTILINE))
    return sorted(names)


def _tool_coverage_audit(records: Sequence[Dict[str, Any]], workspaces_root: Path) -> Dict[str, Any]:
    """Check registry/schema coverage separately from actual execution.

    A registered-but-unobserved tool is never counted as healthy execution.
    This prevents a registry count from being mistaken for a tool-behavior
    audit.
    """

    declared = set(_declared_runtime_tool_names(workspaces_root))
    skipped: set[str] = set()
    registered_counts: List[int] = []
    log_paths = list(workspaces_root.glob("_worker_logs/*.log"))
    log_paths.extend(workspaces_root.glob("*/**/launcher.log"))
    # Archived formal runs are evidence, not the active registry for this
    # batch.  Counting their registration lines would resurrect old 24-tool
    # (pre-removal) state in a new 18-tool preflight.
    log_paths = [path for path in log_paths if "_archive" not in path.parts]
    for path in log_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        skipped.update(re.findall(r"Skipping registration for disabled tool class:\s+(\S+)", text))
        registered_counts.extend(
            int(value)
            for value in re.findall(r"Registered tools summary: base=(\d+)", text)
        )
    active = declared - skipped

    by_tool: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "call_count": 0,
        "success_count": 0,
        "controlled_rejection_count": 0,
        "error_count": 0,
        "unknown_count": 0,
        "timing_seconds_total": 0.0,
        "timing_sample_count": 0,
        "sample_uids": [],
        "evidence_paths": [],
    })
    observed_names: set[str] = set()
    for record in records:
        trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), dict) else {}
        items = _concrete_observation_items(list(_iter_steps(
            ((record.get("_run_result") or {}).get("agent") or {}).get("trajectory") or {}
        )))
        # The audit record normally keeps only compact trajectory metadata.
        # Re-read the full trajectory when available so every call status is
        # checked, including calls inside Parallel.
        run_path = Path(str(record.get("run_result_path") or ""))
        run_result = _read_json(run_path) if run_path.exists() else None
        if isinstance(run_result, dict):
            full_trajectory = ((run_result.get("agent") or {}).get("trajectory") or {})
            items = _concrete_observation_items(list(_iter_steps(full_trajectory)))
        for item in items:
            name = str(item.get("tool") or "").strip()
            if not name:
                continue
            observed_names.add(name)
            summary = by_tool[name]
            summary["call_count"] += 1
            status = _observation_status(item)
            summary[f"{status}_count"] += 1
            uid = str(record.get("sample_uid") or "")
            if uid and uid not in summary["sample_uids"]:
                summary["sample_uids"].append(uid)
            evidence_path = str(record.get("run_result_path") or "")
            if evidence_path and evidence_path not in summary["evidence_paths"]:
                summary["evidence_paths"].append(evidence_path)
        timing = record.get("timing_breakdown_seconds") if isinstance(record.get("timing_breakdown_seconds"), dict) else {}
        item_names = {
            str(item.get("tool") or "").strip()
            for item in items
            if str(item.get("tool") or "").strip()
        }
        for timing_key, timing_value in timing.items():
            key_text = str(timing_key or "")
            if not key_text.startswith("tool.") or not key_text.endswith("_seconds"):
                continue
            timing_slug = re.sub(r"[^a-z0-9]", "", key_text[5:-8].lower())
            for name in item_names:
                normalized_name = re.sub(r"[^a-z0-9]", "", name.lower())
                if normalized_name != timing_slug:
                    continue
                summary = by_tool[name]
                summary["timing_seconds_total"] += _float(timing_value)
                summary["timing_sample_count"] += 1
                break

    all_names = sorted(active | observed_names)
    tool_records: Dict[str, Dict[str, Any]] = {}
    for name in all_names:
        item = dict(by_tool.get(name) or {})
        call_count = int(item.get("call_count") or 0)
        error_count = int(item.get("error_count") or 0)
        unknown_count = int(item.get("unknown_count") or 0)
        item.update({
            "registered_in_project": name in declared,
            "registered_active_in_observed_runtime": name in active,
            "schema_contract_pass": name in declared,
            "execution_observed": call_count > 0,
            "normal_execution_contract_pass": (
                call_count > 0
                and error_count == 0
                and unknown_count == 0
                and int(item.get("success_count") or 0) + int(item.get("controlled_rejection_count") or 0) > 0
            ),
            "coverage_status": (
                "observed_and_checked"
                if call_count > 0
                else "registered_not_exercised"
                if name in active
                else "declared_disabled_or_hidden"
            ),
            "timing_seconds_total": round(float(item.get("timing_seconds_total") or 0.0), 6),
            "timing_sample_count": int(item.get("timing_sample_count") or 0),
            "mean_timing_seconds_per_sample": round(
                float(item.get("timing_seconds_total") or 0.0)
                / max(1, int(item.get("timing_sample_count") or 0)),
                6,
            ) if int(item.get("timing_sample_count") or 0) else None,
            "mean_timing_seconds_per_call": round(
                float(item.get("timing_seconds_total") or 0.0) / max(1, call_count),
                6,
            ) if call_count else None,
        })
        tool_records[name] = item
    unsupported_in_baseline = sorted(_UNSUPPORTED_EDGE_TOOL_NAMES.intersection(declared))
    unsupported_observed = sorted(_UNSUPPORTED_EDGE_TOOL_NAMES.intersection(observed_names))
    return {
        "declared_base_tool_count": len(declared),
        "immutable_baseline_tool_count": len(declared),
        "unsupported_edge_tools_in_baseline": unsupported_in_baseline,
        "unsupported_edge_tools_observed": unsupported_observed,
        "unsupported_edge_tool_registry_contract_pass": not unsupported_in_baseline,
        "observed_runtime_tool_count": len(observed_names),
        "active_registry_count_from_logs": max(registered_counts) if registered_counts else None,
        "active_registry_names_inferred": sorted(active),
        "disabled_or_hidden_names_inferred": sorted(declared - active),
        "unregistered_observed_names": sorted(observed_names - declared),
        "observed_execution_count": sum(int(item.get("call_count") or 0) for item in by_tool.values()),
        "tools": tool_records,
        "coverage_ratio_active_tools": round(len(observed_names & active) / max(1, len(active)), 6),
        "low_usage_threshold_calls": 1,
        "low_usage_observed_tools": sorted(
            name for name in observed_names
            if int((tool_records.get(name) or {}).get("call_count") or 0) <= 1
        ),
        "zero_usage_active_tools": sorted(name for name in active if name not in observed_names),
        "all_observed_execution_contracts_pass": bool(observed_names) and all(
            bool(item.get("normal_execution_contract_pass"))
            for name, item in tool_records.items()
            if name in observed_names
        ) and not unsupported_observed,
        "unexercised_active_tools": sorted(name for name in active if name not in observed_names),
    }


def _tool_ablation_audit(
    records: Sequence[Dict[str, Any]],
    baseline_root: Path,
    ablation_root: Optional[Path],
    disabled_tool_name: str,
) -> Dict[str, Any]:
    """Validate a paired real-tool ablation without treating registry counts as utility.

    A removed tool proves intervention, not utility. Utility additionally
    requires positive net answer gain over complete, identity-matched pairs.
    """

    disabled_name = str(disabled_tool_name or "").strip()
    report: Dict[str, Any] = {
        "status": "not_run",
        "disabled_tool_name": disabled_name,
        "baseline_root": str(baseline_root),
        "ablation_root": str(ablation_root) if ablation_root else "",
        "paired_count": 0,
        "unpaired_count": 0,
        "contract_pass_count": 0,
        "utility_evidence_pass_count": 0,
        "contract_pass": False,
        "utility_evidence_pass": False,
        "pairs": [],
    }
    if not ablation_root or not disabled_name:
        report["reason"] = "tool_ablation_root_and_disabled_tool_are_required"
        return report

    baseline_root = Path(baseline_root).resolve()
    ablation_root = Path(ablation_root).resolve()
    for record in records:
        baseline_result_path = Path(str(record.get("result_path") or ""))
        pair: Dict[str, Any] = {
            "sample_uid": str(record.get("sample_uid") or ""),
            "baseline_result_path": str(baseline_result_path),
            "status": "pending",
            "baseline_tool_call_count": 0,
            "ablation_tool_call_count": 0,
        }
        try:
            relative_workspace = baseline_result_path.parent.relative_to(baseline_root)
        except ValueError:
            pair["status"] = "baseline_path_outside_root"
            report["pairs"].append(pair)
            continue
        ablation_workspace = ablation_root / relative_workspace
        ablation_result_path = ablation_workspace / "mix_sample_result.json"
        ablation_run_path = ablation_workspace / "results" / "run_result.json"
        pair["ablation_result_path"] = str(ablation_result_path)
        pair["ablation_run_result_path"] = str(ablation_run_path)
        baseline_run = _read_json(Path(str(record.get("run_result_path") or ""))) or {}
        ablation_result = _read_json(ablation_result_path) or {}
        ablation_run = _read_json(ablation_run_path) or {}
        if not baseline_run or not ablation_result or not ablation_run:
            pair["status"] = "not_paired"
            report["unpaired_count"] += 1
            report["pairs"].append(pair)
            continue
        baseline_trajectory = ((baseline_run.get("agent") or {}).get("trajectory") or {})
        ablation_trajectory = ((ablation_run.get("agent") or {}).get("trajectory") or {})
        baseline_items = _concrete_observation_items(list(_iter_steps(baseline_trajectory)))
        ablation_items = _concrete_observation_items(list(_iter_steps(ablation_trajectory)))
        # A disabled tool can still appear as a planner attempt serialized
        # with status=no_valid_result.  That is a controlled rejection, not
        # an executed call, so it must not invalidate the zero-execution side
        # of a real tool ablation.
        baseline_names = [
            str(item.get("tool") or "").strip()
            for item in baseline_items
            if str(item.get("tool") or "").strip()
            and _observation_status(item) != "controlled_rejection"
        ]
        ablation_names = [
            str(item.get("tool") or "").strip()
            for item in ablation_items
            if str(item.get("tool") or "").strip()
            and _observation_status(item) != "controlled_rejection"
        ]
        pair["baseline_tool_call_count"] = sum(name == disabled_name for name in baseline_names)
        pair["ablation_tool_call_count"] = sum(name == disabled_name for name in ablation_names)
        pair["baseline_disabled_tool_rejection_count"] = sum(
            str(item.get("tool") or "").strip() == disabled_name
            and _observation_status(item) == "controlled_rejection"
            for item in baseline_items
        )
        pair["ablation_disabled_tool_rejection_count"] = sum(
            str(item.get("tool") or "").strip() == disabled_name
            and _observation_status(item) == "controlled_rejection"
            for item in ablation_items
        )
        from evolution.agent_value import _truthful_case_correct
        baseline_sample = _read_json(baseline_result_path) or {}
        pair["same_execution_input"] = bool(
            pair["sample_uid"] and pair["sample_uid"] == ablation_result.get("sample_uid")
            and pair["sample_uid"] == baseline_sample.get("sample_uid")
            and baseline_trajectory.get("goal")
            and baseline_trajectory.get("goal") == ablation_trajectory.get("goal")
            and baseline_trajectory.get("registry_snapshot_id")
            and baseline_trajectory.get("registry_snapshot_id") == ablation_trajectory.get("registry_snapshot_id")
            and baseline_trajectory.get("trajectory_id") and ablation_trajectory.get("trajectory_id")
            and baseline_trajectory["trajectory_id"] != ablation_trajectory["trajectory_id"]
            and baseline_trajectory.get("steps") and ablation_trajectory.get("steps")
        )
        pair["answer_delta_baseline_minus_ablation"] = (
            int(_truthful_case_correct(baseline_sample)) - int(_truthful_case_correct(ablation_result)))
        pair["baseline_status"] = str(record.get("status") or "missing")
        pair["ablation_status"] = str(ablation_result.get("status") or "missing")
        pair["baseline_terminal"] = pair["baseline_status"].lower() in _TERMINAL_SAMPLE_STATUSES
        pair["ablation_terminal"] = pair["ablation_status"].lower() in _TERMINAL_SAMPLE_STATUSES
        tool_library = ((ablation_run.get("metrics") or {}).get("tool_library") or {})
        hidden_tools = {
            str(name).strip()
            for name in (tool_library.get("scoped_hidden_tools") or [])
            if str(name).strip()
        }
        launcher_text = ""
        try:
            launcher_text = (ablation_workspace / "launcher.log").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            pass
        override_confirmed = disabled_name in hidden_tools or (
            "runtime tool registry overrides" in launcher_text.lower()
            and disabled_name.lower() in launcher_text.lower()
        )
        pair["disabled_tool_registry_override_confirmed"] = bool(override_confirmed)
        pair["baseline_protocol_pass"] = bool((record.get("protocol") or {}).get("protocol_pass"))
        pair["ablation_run_success"] = bool(ablation_run.get("success"))
        runtime_error_markers = {
            "Traceback",
            "CUDA out of memory",
            "Memory cgroup out of memory",
            "LLVM",
            "SIGABRT",
            "return_code=-6",
            "returncode=-6",
            "return_code=-9",
            "returncode=-9",
        }
        ablation_log_text = ""
        for candidate in (
            ablation_workspace / "launcher.log",
            ablation_workspace / "logs" / "run.log",
        ):
            try:
                ablation_log_text += candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        pair["ablation_runtime_error_markers"] = sorted(
            marker for marker in runtime_error_markers if marker in ablation_log_text
        )
        pair["ablation_controlled_terminal"] = bool(
            pair["ablation_terminal"]
            and not pair["ablation_runtime_error_markers"]
        )
        contract_pass = bool(
            baseline_run
            and ablation_result
            and ablation_run
            and pair["baseline_terminal"]
            and pair["ablation_terminal"]
            and pair["disabled_tool_registry_override_confirmed"]
            and pair["same_execution_input"]
            and pair["ablation_tool_call_count"] == 0
            # A tool ablation is allowed to expose a controlled safety
            # degradation (wrong/aborted terminal) as long as it does not
            # crash the runtime.  Keep ablation_run_success visible below;
            # it is not a prerequisite for proving that the disabled tool
            # was actually removed from the execution path.
            and pair["ablation_controlled_terminal"]
        )
        utility_pass = bool(
            contract_pass
            and pair["baseline_protocol_pass"]
            and pair["baseline_tool_call_count"] > 0
            and pair["ablation_tool_call_count"] == 0
            and pair["answer_delta_baseline_minus_ablation"] > 0
        )
        pair["contract_pass"] = contract_pass
        pair["utility_evidence_pass"] = utility_pass
        pair["status"] = "pass" if utility_pass else "contract_only" if contract_pass else "fail"
        report["paired_count"] += 1
        report["contract_pass_count"] += int(contract_pass)
        report["utility_evidence_pass_count"] += int(utility_pass)
        report["pairs"].append(pair)

    report["contract_pass"] = bool(
        report["paired_count"] > 0
        and report["paired_count"] == len(records)
        and report["contract_pass_count"] == report["paired_count"]
    )
    report["net_answer_gain"] = sum(pair.get("answer_delta_baseline_minus_ablation", 0)
                                    for pair in report["pairs"] if pair.get("contract_pass"))
    report["utility_evidence_pass"] = bool(report["contract_pass"]
        and report["utility_evidence_pass_count"] > 0 and report["net_answer_gain"] > 0)
    report["status"] = "pass" if report["utility_evidence_pass"] else "pending"
    return report


_EVOLVER_REPLAY_GROUPS = ("trigger", "neighbor", "regression", "held_out")
_EVOLVER_ALLOWED_ACTIONS = {
    "update_tool_schema",
    "add_composition_edge",
    "update_recovery_policy",
    "edit_tool_source",
}
_EVOLVER_CAUSAL_STATUSES = {
    "causal_effect_verified",
    "causal_fix_verified",
}
_EVOLVER_EXECUTED_SUFFIX_STATUSES = {
    "verifier_suffix_executed",
    "runtime_suffix_executed",
    "coordinated_suffix_executed",
    "paired_runtime_rollout_executed",
}
_EVOLVER_REJECTION_STATUSES = {
    "awaiting_future_validation",
    "local_repair_not_proven",
    "validation_exhausted",
    "invalid_schema",
    "invalid_mutation",
    "invalid_coordinated_mutation",
    "unsupported_action",
    "no_candidate",
    "not_required",
    "candidate_pending_fresh_inference",
    "candidate_inference_complete_pending_causal_replay",
    "candidate_inference_complete_insufficient_coverage",
    "candidate_inference_complete_pending_snapshot_application",
    "insufficient_coverage",
    "invalid_pairing",
    "rejected",
}


def _evolver_artifact_index(audit_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Index Evolver artifacts by their first directory below ``_evolution_audit``.

    An evolution report, candidate replay, suffix proof, and registry snapshot
    are produced by different commands, so their presence must be audited as
    a set rather than inferred from a single report field.  The first relative
    directory is stable across the historical and current layouts and avoids
    treating unrelated agent-value JSON files as Evolver runs.
    """

    index: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    if not audit_root.exists():
        return index

    def add(kind: str, path: Path) -> None:
        try:
            relative = path.relative_to(audit_root)
        except ValueError:
            return
        if not relative.parts:
            return
        index[str(relative.parts[0])][kind].append(path)

    for path in audit_root.rglob("evolution_report.json"):
        if path.is_file() and path.parent.name.startswith("evolution_run_"):
            add("reports", path)
    for filename, kind in (
        ("replay.json", "candidate_replays"),
        ("selective_suffix_trigger.json", "selective_artifacts"),
        ("harness_replay.json", "selective_artifacts"),
        ("replay_preflight.json", "preflight_artifacts"),
        ("snapshot.json", "snapshots"),
    ):
        for path in audit_root.rglob(filename):
            if path.is_file():
                add(kind, path)
    for item in index.values():
        for paths in item.values():
            paths.sort()
    return index


def _evolver_snapshot_operations(snapshot: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Describe operations materialized in a candidate snapshot.

    This is intentionally a structural audit, not a second mutation engine.
    The authoritative runtime preflight remains in ``RL/evolution``; this
    function checks that a report did not claim a candidate while the snapshot
    contains an unsupported or empty operation surface.
    """

    operations: List[Dict[str, Any]] = []
    for target, payload in (snapshot.get("source_overrides") or {}).items():
        operations.append({"action": "edit_tool_source", "target": str(target), "payload_keys": sorted(payload)})
    tool_overrides = snapshot.get("tool_overrides")
    if isinstance(tool_overrides, Mapping):
        for target, payload in sorted(tool_overrides.items(), key=lambda item: str(item[0])):
            operations.append({
                "action": "update_tool_schema",
                "target": str(target),
                "payload_keys": sorted(str(key) for key in payload) if isinstance(payload, Mapping) else [],
            })
    router_policy = snapshot.get("router_policy")
    if isinstance(router_policy, Mapping):
        for action, payload in sorted(router_policy.items(), key=lambda item: str(item[0])):
            action_text = str(action)
            target = {
                "add_composition_edge": "candidate_comparison",
                "update_recovery_policy": "low_signal_temporal_grounding",
            }.get(action_text, action_text)
            operations.append({
                "action": action_text,
                "target": target,
                "payload_keys": sorted(str(key) for key in payload) if isinstance(payload, Mapping) else [],
            })
    argument_policy = snapshot.get("argument_policy")
    if isinstance(argument_policy, Mapping) and argument_policy:
        operations.append({
            "action": "update_recovery_policy",
            "target": "low_signal_temporal_grounding",
            "payload_keys": sorted(str(key) for key in argument_policy),
        })
    return operations


def _evolver_dependency_contract(
    snapshot: Mapping[str, Any],
    validation: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recompute the graph contract instead of trusting Evolver prose."""

    from evolution.models import normalize_mutation_payload
    proposal = normalize_mutation_payload(proposal)
    raw_mutations = proposal.get("mutations")
    if isinstance(raw_mutations, list):
        mutations = [item for item in raw_mutations if isinstance(item, Mapping)]
    elif isinstance(proposal.get("mutation"), Mapping):
        mutations = [proposal.get("mutation")]
    else:
        mutations = []
    computed = analyze_evolver_dependencies(mutations)
    snapshot_analysis = snapshot.get("dependency_analysis")
    if not isinstance(snapshot_analysis, Mapping):
        snapshot_analysis = {}
    validation_analysis = validation.get("dependency_analysis")
    if not isinstance(validation_analysis, Mapping):
        validation_analysis = {}
    metadata_present = bool(snapshot_analysis)
    metadata_pass = bool(snapshot_analysis.get("dependency_pass"))
    operation_count = int(computed.get("operation_count") or 0)
    snapshot_operation_count = snapshot_analysis.get("operation_count")
    if snapshot_operation_count is None:
        count_match = True
    else:
        try:
            count_match = int(snapshot_operation_count) == operation_count
        except (TypeError, ValueError):
            count_match = False
    def _surface_from_nodes(nodes: Any) -> List[Tuple[str, str]]:
        surface: set[Tuple[str, str]] = set()
        if not isinstance(nodes, list):
            return []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            action = str(node.get("action") or "").strip()
            raw_targets = node.get("target_ids")
            if isinstance(raw_targets, list):
                targets = [str(item).strip() for item in raw_targets if str(item).strip()]
            else:
                target = str(node.get("target") or "").strip()
                targets = [target] if target else []
            for target in targets:
                surface.add((action, target))
        return sorted(surface)

    proposal_surface = _surface_from_nodes(computed.get("nodes"))
    snapshot_surface = _surface_from_nodes(_evolver_snapshot_operations(snapshot))
    operation_surface_match = bool(proposal_surface) and proposal_surface == snapshot_surface
    dependency_fields = (
        "graph_version",
        "operation_count",
        "joint_required",
        "nodes",
        "dependency_edges",
        "missing_dependencies",
        "conflicts",
        "dependency_pass",
    )

    def _canonical(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    metadata_structural_match = bool(snapshot_analysis) and all(
        key in snapshot_analysis
        and _canonical(snapshot_analysis.get(key)) == _canonical(computed.get(key))
        for key in dependency_fields
    )
    validation_structural_match = not validation_analysis or all(
        key in validation_analysis
        and _canonical(validation_analysis.get(key)) == _canonical(computed.get(key))
        for key in dependency_fields
    )
    dependency_pass = bool(
        computed.get("dependency_pass")
        and metadata_present
        and metadata_pass
        and count_match
        and operation_surface_match
        and metadata_structural_match
        and validation_structural_match
    )
    effective = dict(snapshot_analysis or validation_analysis or computed)
    return {
        "dependency_analysis": effective,
        "computed_dependency_analysis": computed,
        "validation_dependency_analysis": dict(validation_analysis),
        "snapshot_dependency_analysis": dict(snapshot_analysis),
        "dependency_metadata_present": metadata_present,
        "dependency_pass": dependency_pass,
        "operation_count": operation_count,
        "coordinated": operation_count > 1,
        "joint_replay_required": operation_count > 1,
        "proposal_operation_surface": proposal_surface,
        "snapshot_operation_surface": snapshot_surface,
        "operation_surface_match": operation_surface_match,
        "metadata_structural_match": metadata_structural_match,
        "validation_structural_match": validation_structural_match,
    }


def _evolver_candidate_contract(
    snapshot: Optional[Mapping[str, Any]],
    validation: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> Dict[str, Any]:
    """Audit candidate snapshot identity, executable surface, and boundary use."""

    decision = str(proposal.get("decision") or "").strip()
    validation_status = str(validation.get("status") or "").strip()
    if not snapshot:
        expected_rejection = (
            decision != "propose_mutation"
            or validation_status in _EVOLVER_REJECTION_STATUSES
            or validation_status == ""
        )
        return {
            "required": decision == "propose_mutation",
            "snapshot_present": False,
            "snapshot_contract_pass": not (decision == "propose_mutation" and not expected_rejection),
            "boundary_pass": expected_rejection,
            "validation_status": validation_status,
            "operations": [],
            "unsupported_operations": [],
        }

    operations = _evolver_snapshot_operations(snapshot)
    dependency_contract = _evolver_dependency_contract(snapshot, validation, proposal)
    unsupported = [
        item["action"] for item in operations
        if str(item.get("action") or "") not in _EVOLVER_ALLOWED_ACTIONS
    ]
    identity_pass = bool(
        str(snapshot.get("snapshot_id") or "").strip()
        and str(snapshot.get("parent_snapshot_id") or "").strip()
        and str(snapshot.get("status") or "") in {"candidate", "accepted", "rejected"}
        and str(snapshot.get("candidate_source") or "evolver_llm") in {"", "evolver_llm"}
    )
    executable_surface_pass = bool(operations) and all(
        bool(item.get("payload_keys")) for item in operations
    )
    argument_policy_pass = not bool(snapshot.get("argument_policy"))
    candidate_created = validation_status in {"candidate_created", ""}
    contract_pass = bool(
        decision == "propose_mutation"
        and identity_pass
        and operations
        and executable_surface_pass
        and argument_policy_pass
        and not unsupported
        and candidate_created
        and dependency_contract.get("dependency_pass")
    )
    return {
        "required": True,
        "snapshot_present": True,
        "snapshot_id": str(snapshot.get("snapshot_id") or ""),
        "parent_snapshot_id": str(snapshot.get("parent_snapshot_id") or ""),
        "snapshot_status": str(snapshot.get("status") or ""),
        "snapshot_contract_pass": contract_pass,
        "boundary_pass": contract_pass and decision == "propose_mutation",
        "validation_status": validation_status,
        "operations": operations,
        "executable_surface_pass": executable_surface_pass,
        "argument_policy_pass": argument_policy_pass,
        "unsupported_operations": sorted(set(unsupported)),
        "dependency_contract": dependency_contract,
        "dependency_analysis_pass": bool(dependency_contract.get("dependency_pass")),
        "joint_replay_required": bool(dependency_contract.get("joint_replay_required")),
    }


def _evolver_selective_contract(
    report: Mapping[str, Any],
    selective_payloads: Sequence[Mapping[str, Any]],
    candidate_required: bool,
    expected_snapshot_id: str = "",
) -> Dict[str, Any]:
    """Audit selective/suffix planning, execution, causal effect, and leakage."""

    plans: List[Mapping[str, Any]] = []
    report_plan = report.get("selective_replay")
    if isinstance(report_plan, Mapping):
        plans.append(report_plan)
    for payload in selective_payloads:
        plan = payload.get("selective_suffix_plan")
        if isinstance(plan, Mapping):
            enriched_plan = dict(plan)
            if isinstance(payload.get("harness"), Mapping) and "harness" not in enriched_plan:
                enriched_plan["harness"] = payload.get("harness")
            plans.append(enriched_plan)
        elif isinstance(payload.get("cut"), Mapping):
            plans.append(payload)

    plan_records: List[Dict[str, Any]] = []
    for plan in plans:
        cut = plan.get("cut") if isinstance(plan.get("cut"), Mapping) else {}
        execution = plan.get("execution") if isinstance(plan.get("execution"), Mapping) else {}
        leakage = plan.get("leakage_audit") if isinstance(plan.get("leakage_audit"), Mapping) else {}
        harness = plan.get("harness") if isinstance(plan.get("harness"), Mapping) else {}
        rollback = plan.get("rollback") if isinstance(plan.get("rollback"), Mapping) else {}
        execution_status = str(execution.get("status") or "")
        causal_status = str(execution.get("causal_fix_status") or "")
        plan_snapshot_id = str(
            harness.get("snapshot_id")
            or execution.get("snapshot_id")
            or rollback.get("candidate_snapshot_id")
            or plan.get("candidate_snapshot_id")
            or ""
        ).strip()
        snapshot_match = bool(expected_snapshot_id) and plan_snapshot_id == expected_snapshot_id
        causal_effect_verified = bool(
            causal_status in _EVOLVER_CAUSAL_STATUSES
            and (snapshot_match if candidate_required else True)
        )
        plan_records.append({
            "mode": str(plan.get("mode") or "selective_suffix"),
            "replay_kind": str(plan.get("replay_kind") or ""),
            "triggered": bool(cut.get("triggered")),
            "execution_status": execution_status,
            "executed": execution_status in _EVOLVER_EXECUTED_SUFFIX_STATUSES,
            "causal_fix_status": causal_status,
            "snapshot_id": plan_snapshot_id,
            "snapshot_match": snapshot_match,
            "causal_effect_verified": causal_effect_verified,
            "leakage_pass": bool(leakage.get("passed"))
            and not bool(leakage.get("surviving_forbidden_fields")),
            "surviving_forbidden_fields": list(leakage.get("surviving_forbidden_fields") or []),
            "artifact_path": str(plan.get("artifact_path") or ""),
        })

    leakage_violations = [
        item for item in plan_records if not bool(item.get("leakage_pass"))
    ]
    executed = [item for item in plan_records if item.get("executed")]
    causal = [item for item in plan_records if item.get("causal_effect_verified")]
    return {
        "required": bool(candidate_required),
        "plan_count": len(plan_records),
        "plan_available": bool(plan_records) if candidate_required else True,
        "executed_count": len(executed),
        "causal_effect_verified_count": len(causal),
        "snapshot_match_count": sum(bool(item.get("snapshot_match")) for item in plan_records),
        "leakage_violation_count": len(leakage_violations),
        "causal_effect_verified": bool(causal),
        "leakage_contract_pass": not leakage_violations,
        "records": plan_records,
    }


def _evolver_replay_contract(
    report: Mapping[str, Any],
    candidate_replays: Sequence[Mapping[str, Any]],
    selective_payloads: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Audit fresh inference, snapshot application, pairing, and coverage.

    Snapshot application is a runtime-boundary attestation. It is not a
    second full trigger-case inference; causal trigger evidence comes from a
    selective/suffix artifact, while fresh inference is used for paired
    candidate cohorts.
    """

    sources: List[Mapping[str, Any]] = [report]
    sources.extend(candidate_replays)
    fresh_proofs: List[Dict[str, Any]] = []
    pairing_proofs: List[Dict[str, Any]] = []
    mutation_proofs: List[Dict[str, Any]] = []
    snapshot_application_records: List[Dict[str, Any]] = []
    coherent_records: List[Dict[str, Any]] = []
    replay_statuses: List[str] = []
    final_statuses: List[str] = []
    expected_snapshot_id = str(
        ((report.get("candidate_snapshot") or {}).get("snapshot_id") or "")
        if isinstance(report.get("candidate_snapshot"), Mapping)
        else ""
    ).strip()
    for source_index, payload in enumerate(sources):
        source_kind = "report" if source_index == 0 else "candidate_replay"
        replay_status = str(payload.get("replay_status") or "").strip()
        final_status = str(payload.get("final_acceptance_status") or "").strip()
        if replay_status:
            replay_statuses.append(replay_status)
        if final_status:
            final_statuses.append(final_status)
        evaluation = payload.get("replay_evaluation")
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        checks = evaluation.get("checks")
        checks = checks if isinstance(checks, Mapping) else {}
        group_counts = evaluation.get("group_counts")
        group_counts = group_counts if isinstance(group_counts, Mapping) else {}
        minimums = evaluation.get("minimum_group_counts")
        minimums = minimums if isinstance(minimums, Mapping) else {}
        fresh_ids = payload.get("fresh_trajectory_ids")
        candidate_ids = payload.get("candidate_trajectory_ids")
        candidate_snapshot_id = str(payload.get("candidate_snapshot_id") or "").strip()
        if not candidate_snapshot_id and source_kind == "report":
            report_snapshot = payload.get("candidate_snapshot")
            if isinstance(report_snapshot, Mapping):
                candidate_snapshot_id = str(report_snapshot.get("snapshot_id") or "").strip()
        snapshot_match = bool(expected_snapshot_id) and candidate_snapshot_id == expected_snapshot_id
        fresh_claim = bool(
            payload.get("fresh_inference_proven")
            and isinstance(candidate_ids, list)
            and bool(candidate_ids)
            and all(str(item or "").strip() for item in candidate_ids)
        )
        if not fresh_claim:
            fresh_claim = bool(
                payload.get("candidate_inference_executed")
                and payload.get("fresh_candidate_inference_proven", False)
            )
        if not fresh_claim:
            fresh_claim = bool(
                payload.get("candidate_inference_executed")
                and fresh_ids is True
                and isinstance(candidate_ids, list)
                and bool(candidate_ids)
                and all(str(item or "").strip() for item in candidate_ids)
            )
        # A fresh candidate cohort is a separate replay artifact.  The
        # evolution report may summarize a claim, but it cannot substitute for
        # the candidate worker's snapshot-bound trajectories.
        fresh_claim = bool(fresh_claim and source_kind == "candidate_replay" and snapshot_match)
        fresh_proofs.append({
            "source": str(payload.get("candidate_output_root") or payload.get("run_id") or ""),
            "source_kind": source_kind,
            "snapshot_id": candidate_snapshot_id,
            "snapshot_match": snapshot_match,
            "candidate_inference_executed": bool(payload.get("candidate_inference_executed")),
            "fresh_trajectory_ids": fresh_ids,
            "candidate_trajectory_count": len(candidate_ids) if isinstance(candidate_ids, list) else 0,
            "proven": fresh_claim,
        })
        per_case = payload.get("per_case")
        per_case = per_case if isinstance(per_case, list) else []
        per_case_application_proven = bool(per_case) and all(
            isinstance(item, Mapping)
            and isinstance(item.get("candidate"), Mapping)
            and bool(item["candidate"].get("snapshot_application_proven"))
            and isinstance(item["candidate"].get("snapshot_application_proof"), Mapping)
            and item["candidate"]["snapshot_application_proof"].get("runtime_hook_applied") is True
            and str(item["candidate"]["snapshot_application_proof"].get("proof_scope") or "")
            == "runtime_boundary_attestation"
            and str(item["candidate"].get("harness_snapshot_id") or "").strip()
            == expected_snapshot_id
            for item in per_case
        )
        application_claim = payload.get("snapshot_application_proven") is True
        runtime_proofs = [
            item["candidate"].get("snapshot_application_proof")
            for item in per_case
            if isinstance(item, Mapping) and isinstance(item.get("candidate"), Mapping)
            and isinstance(item["candidate"].get("snapshot_application_proof"), Mapping)
        ]
        application_proven = bool(
            source_kind == "candidate_replay"
            and application_claim
            and snapshot_match
            and per_case_application_proven
        )
        snapshot_application_records.append({
            "source": str(payload.get("candidate_output_root") or payload.get("run_id") or ""),
            "source_kind": source_kind,
            "scope": "candidate_runtime",
            "snapshot_id": candidate_snapshot_id,
            "snapshot_match": snapshot_match,
            "application_claim": application_claim,
            "per_case_count": len(per_case),
            "per_case_application_proven": per_case_application_proven,
            "runtime_proof_count": len(runtime_proofs),
            "proven": application_proven,
        })
        mutation_proofs.append({
            "source": str(payload.get("candidate_output_root") or payload.get("run_id") or ""),
            "source_kind": source_kind,
            "snapshot_id": candidate_snapshot_id,
            "snapshot_match": snapshot_match,
            "mutation_claim": application_claim,
            "per_case_count": len(per_case),
            "per_case_application_proven": per_case_application_proven,
            "runtime_proof_count": len(runtime_proofs),
            "candidate_inference_executed": bool(payload.get("candidate_inference_executed")),
            "proven": application_proven,
        })
        paired = checks.get("paired_sample_set")
        if paired is None:
            paired = evaluation.get("paired_sample_set")
        coverage = evaluation.get("coverage_valid")
        coverage_checks = evaluation.get("coverage_checks")
        if not isinstance(coverage_checks, Mapping):
            coverage_checks = {}
        if coverage is None and coverage_checks:
            coverage = all(bool(coverage_checks.get(group)) for group in _EVOLVER_REPLAY_GROUPS)
        normalized_minimums: Dict[str, int] = {}
        for group in _EVOLVER_REPLAY_GROUPS:
            try:
                configured_minimum = int(minimums.get(group, 1) or 1)
            except (TypeError, ValueError):
                configured_minimum = 1
            # The formal audit never permits a zero/negative minimum to make
            # an absent replay group look covered.
            normalized_minimums[group] = max(1, configured_minimum)
        minimum_checks = {
            group: int(group_counts.get(group, 0) or 0) >= normalized_minimums[group]
            for group in _EVOLVER_REPLAY_GROUPS
        }
        minimum_group_coverage_pass = all(minimum_checks.values())
        coverage = bool(coverage) and minimum_group_coverage_pass
        pairing_proofs.append({
            "source": str(payload.get("candidate_output_root") or payload.get("run_id") or ""),
            "source_kind": source_kind,
            "snapshot_id": candidate_snapshot_id,
            "snapshot_match": snapshot_match,
            "paired_sample_set": bool(paired),
            "coverage_valid": bool(coverage),
            "coverage_checks": {str(key): bool(value) for key, value in coverage_checks.items()},
            "minimum_group_counts": normalized_minimums,
            "group_counts": {str(key): int(value or 0) for key, value in group_counts.items()},
            "minimum_group_coverage_checks": minimum_checks,
            "minimum_group_coverage_pass": minimum_group_coverage_pass,
            "artifact_pairing_errors": list(payload.get("artifact_pairing_errors") or []),
            "metric_gate_status": str(
                evaluation.get("status")
                or evaluation.get("metric_gate_status")
                or payload.get("metric_gate_status")
                or ""
            ),
            "utility_gain_proven": bool(evaluation.get("utility_gain_proven")),
        })
        coherent_records.append({
            "source": str(payload.get("candidate_output_root") or payload.get("run_id") or ""),
            "source_kind": source_kind,
            "snapshot_id": candidate_snapshot_id,
            "snapshot_match": snapshot_match,
            "fresh_proven": fresh_claim,
            "application_proven": application_proven,
            "paired_sample_set": bool(paired),
            "coverage_valid": bool(coverage),
            "minimum_group_coverage_pass": minimum_group_coverage_pass,
            "artifact_pairing_errors": list(payload.get("artifact_pairing_errors") or []),
            "metric_gate_status": str(
                evaluation.get("status")
                or evaluation.get("metric_gate_status")
                or payload.get("metric_gate_status")
                or ""
            ),
            "utility_gain_proven": bool(evaluation.get("utility_gain_proven")),
        })

    for payload in selective_payloads:
        harness = payload.get("harness")
        harness = harness if isinstance(harness, Mapping) else {}
        snapshot_id = str(harness.get("snapshot_id") or "").strip()
        snapshot_match = bool(expected_snapshot_id) and snapshot_id == expected_snapshot_id
        application_claim = harness.get("snapshot_application_proven") is True
        application_proof = harness.get("snapshot_application_proof")
        application_proven = bool(
            application_claim
            and snapshot_match
            and isinstance(application_proof, Mapping)
            and application_proof.get("runtime_hook_applied") is True
            and str(application_proof.get("proof_scope") or "") == "runtime_boundary_attestation"
        )
        snapshot_application_records.append({
            "source": str(payload.get("snapshot_path") or payload.get("run_result_path") or ""),
            "scope": "isolated_harness_preflight",
            "snapshot_id": snapshot_id,
            "snapshot_match": snapshot_match,
            "application_claim": application_claim,
            "application_proof": dict(application_proof) if isinstance(application_proof, Mapping) else {},
            "per_case_count": 0,
            "per_case_application_proven": False,
            "proven": application_proven,
        })

    fresh_proven = any(
        bool(item.get("proven"))
        and item.get("source_kind") == "candidate_replay"
        and bool(item.get("snapshot_match"))
        for item in fresh_proofs
    )
    paired_proven = any(
        item.get("source_kind") == "candidate_replay"
        and bool(item.get("snapshot_match"))
        and bool(item.get("paired_sample_set"))
        and not item.get("artifact_pairing_errors")
        for item in pairing_proofs
    )
    coverage_proven = any(
        item.get("source_kind") == "candidate_replay"
        and bool(item.get("snapshot_match"))
        and bool(item.get("coverage_valid"))
        for item in pairing_proofs
    )
    metric_accepted = any(
        item.get("source_kind") == "candidate_replay"
        and bool(item.get("snapshot_match"))
        and item.get("metric_gate_status") == "accepted"
        for item in pairing_proofs
    )
    utility_gain_proven = any(
        item.get("source_kind") == "candidate_replay"
        and bool(item.get("snapshot_match"))
        and item.get("metric_gate_status") == "accepted"
        and bool(item.get("utility_gain_proven"))
        for item in pairing_proofs
    )
    mutation_application_proven = any(
        item.get("scope") == "candidate_runtime"
        and item.get("source_kind") == "candidate_replay"
        and bool(item.get("proven"))
        for item in snapshot_application_records
    )

    def _application_matches(snapshot_id: str) -> bool:
        return any(
            item.get("scope") == "candidate_runtime"
            and item.get("source_kind") == "candidate_replay"
            and bool(item.get("proven"))
            and str(item.get("snapshot_id") or "").strip() == str(snapshot_id or "").strip()
            for item in snapshot_application_records
        )

    coherent_replay_gate_proven = any(
        item.get("source_kind") == "candidate_replay"
        and bool(item.get("snapshot_match"))
        and bool(item.get("fresh_proven"))
        and bool(item.get("application_proven"))
        and bool(item.get("paired_sample_set"))
        and not item.get("artifact_pairing_errors")
        and bool(item.get("coverage_valid"))
        and item.get("metric_gate_status") == "accepted"
        and bool(item.get("utility_gain_proven"))
        for item in coherent_records
    )
    return {
        "candidate_replay_artifact_count": len(candidate_replays),
        "fresh_candidate_inference_proven": fresh_proven,
        "paired_uid_proven": paired_proven,
        "minimum_replay_group_coverage_proven": coverage_proven,
        "metric_gate_accepted": metric_accepted,
        "utility_gain_proven": utility_gain_proven,
        "mutation_application_proven": mutation_application_proven,
        "snapshot_application_records": snapshot_application_records,
        "coherent_replay_gate_proven": coherent_replay_gate_proven,
        "replay_statuses": sorted(set(replay_statuses)),
        "final_acceptance_statuses": sorted(set(final_statuses)),
        "fresh_inference_records": fresh_proofs,
        "mutation_application_records": mutation_proofs,
        "paired_coverage_records": pairing_proofs,
        "coherent_replay_records": coherent_records,
        "expected_snapshot_id": expected_snapshot_id,
    }


def _evolver_reference_design_contract() -> Dict[str, Any]:
    """Check that the reference design used by Evolver is present and explicit."""

    path = PROJECT_ROOT / "reference.md"
    required_markers = (
        "Evolver设计",
        "failure diagnosis",
        "dependency analysis",
        "jointly modify",
        "second rollout",
        "verify joint modification",
        "harness component dependency graph",
    )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        text = ""
    folded = text.casefold()
    missing = [marker for marker in required_markers if marker.casefold() not in folded]
    return {
        "path": str(path),
        "required_markers": list(required_markers),
        "missing_markers": missing,
        "pass": bool(text.strip()) and not missing,
    }


def _evolver_audit(audit_root: Path) -> Dict[str, Any]:
    """Run the formal Evolver audit item over all materialized Evolver evidence.

    The audit distinguishes four outcomes that were previously conflated:
    an invalid proposal correctly rejected, a candidate waiting for replay,
    a replay with incomplete coverage, and a genuinely accepted candidate.
    It never upgrades a proposal or deterministic verifier replay to fresh
    candidate inference.
    """

    reference_design = _evolver_reference_design_contract()
    artifact_index = _evolver_artifact_index(audit_root)
    report_paths = sorted(
        path
        for group in artifact_index.values()
        for path in group.get("reports", [])
    )
    if not report_paths:
        return {
            "audit_item": "evolver",
            "status": "not_run",
            "audit_scan_pass": False,
            "contract_status": "not_run",
            "acceptance_status": "not_run",
            "utility_status": "not_proven",
            "report_count": 0,
            "candidate_count": 0,
            "blocking_reasons": ["no_evolution_report_found"],
            "reference_design": reference_design,
            "artifact_inventory": {},
            "reports": [],
        }

    report_records: List[Dict[str, Any]] = []
    malformed_reports: List[str] = []
    contract_violations: List[Dict[str, Any]] = []
    for report_path in report_paths:
        payload = _read_json(report_path)
        if payload is None:
            malformed_reports.append(str(report_path))
            continue
        try:
            relative = report_path.relative_to(audit_root)
            root_key = str(relative.parts[0])
        except ValueError:
            root_key = report_path.parent.parent.name
        artifacts = artifact_index.get(root_key, {})
        candidate_replays: List[Mapping[str, Any]] = []
        for path in artifacts.get("candidate_replays", []):
            replay_payload = _read_json(path)
            if replay_payload is not None:
                candidate_replays.append(replay_payload)
        selective_payloads: List[Mapping[str, Any]] = []
        for path in artifacts.get("selective_artifacts", []):
            selective_payload = _read_json(path)
            if selective_payload is not None:
                selective_payloads.append(selective_payload)
        preflight_payloads: List[Mapping[str, Any]] = []
        for path in artifacts.get("preflight_artifacts", []):
            preflight_payload = _read_json(path)
            if preflight_payload is not None:
                preflight_payloads.append(preflight_payload)

        evolver_agent = payload.get("evolver_agent")
        evolver_agent = evolver_agent if isinstance(evolver_agent, Mapping) else {}
        proposal = evolver_agent.get("proposal")
        proposal = proposal if isinstance(proposal, Mapping) else {}
        validation = payload.get("evolver_proposal_validation")
        validation = validation if isinstance(validation, Mapping) else {}
        if not validation:
            validation = evolver_agent.get("proposal_validation")
            validation = validation if isinstance(validation, Mapping) else {}
        snapshot = payload.get("candidate_snapshot")
        snapshot = snapshot if isinstance(snapshot, Mapping) else None
        if snapshot is None and artifacts.get("snapshots"):
            snapshot_payload = _read_json(artifacts["snapshots"][-1])
            snapshot = snapshot_payload if isinstance(snapshot_payload, Mapping) else None

        diagnoses = payload.get("diagnoses")
        diagnoses = diagnoses if isinstance(diagnoses, list) else []
        attribution_records = [
            item for item in diagnoses
            if isinstance(item, Mapping)
        ]
        attribution_missing = [
            str(item.get("diagnosis_id") or index)
            for index, item in enumerate(attribution_records)
            if not str(item.get("failure_scope") or "").strip()
            or not isinstance(item.get("attribution"), Mapping)
        ]
        candidate_contract = _evolver_candidate_contract(snapshot, validation, proposal)
        selective_contract = _evolver_selective_contract(
            payload,
            selective_payloads,
            candidate_required=bool(snapshot),
            expected_snapshot_id=str((snapshot or {}).get("snapshot_id") or "").strip(),
        )
        replay_contract = _evolver_replay_contract(
            payload,
            candidate_replays,
            selective_payloads,
        )
        agent_status = str(evolver_agent.get("status") or "")
        kernel = evolver_agent.get("kernel")
        kernel = kernel if isinstance(kernel, Mapping) else {}
        llm_contract_pass = bool(
            agent_status == "success"
            and evolver_agent.get("schema_valid") is True
            and kernel.get("llm_kernel") is True
        )
        report_structure_pass = all(
            key in payload for key in ("run_id", "evolver_agent", "evolver_input_contract")
        )
        input_contract = payload.get("evolver_input_contract")
        input_contract = input_contract if isinstance(input_contract, Mapping) else {}
        runtime_manifest = input_contract.get("runtime_capability_manifest")
        runtime_manifest = runtime_manifest if isinstance(runtime_manifest, Mapping) else {}
        declared_reference_design = runtime_manifest.get("reference_design")
        declared_reference_design = (
            declared_reference_design
            if isinstance(declared_reference_design, Mapping)
            else {}
        )
        declared_pipeline = {
            str(item).strip().lower()
            for item in (declared_reference_design.get("pipeline") or [])
            if str(item).strip()
        }
        reference_design_contract_pass = bool(
            reference_design.get("pass")
            and declared_reference_design.get("joint_edit_enabled") is True
            and 2 <= _int(declared_reference_design.get("max_operations")) <= 4
            and declared_reference_design.get("same_snapshot_second_rollout_required") is True
            and {"dependency_analysis", "joint_modify", "second_rollout"}.issubset(declared_pipeline)
        )
        decision = str(proposal.get("decision") or "").strip()
        final_status = str(payload.get("final_acceptance_status") or "").strip()
        preflight_statuses = sorted({
            str(item.get("schema_preflight_status") or item.get("status") or "").strip()
            for item in preflight_payloads
            if str(item.get("schema_preflight_status") or item.get("status") or "").strip()
        })
        preflight_rejected = any(
            status in {"invalid_mutation", "unsupported_action", "invalid_schema"}
            for status in preflight_statuses
        )
        report_violations: List[str] = []
        if snapshot and not candidate_contract.get("snapshot_contract_pass"):
            report_violations.append("candidate_snapshot_contract_failed")
        if snapshot and not candidate_contract.get("dependency_analysis_pass"):
            report_violations.append("candidate_dependency_analysis_failed")
        if not reference_design_contract_pass:
            report_violations.append("reference_evolver_design_contract_failed")
        if snapshot and preflight_rejected:
            report_violations.append("preflight_rejected_but_snapshot_materialized")
        if selective_contract.get("leakage_violation_count", 0):
            report_violations.append("selective_replay_leakage_failed")
        if final_status == "accepted":
            if not replay_contract.get("fresh_candidate_inference_proven"):
                report_violations.append("accepted_without_fresh_candidate_inference")
            if not replay_contract.get("mutation_application_proven"):
                report_violations.append("accepted_without_applied_mutation")
            if not replay_contract.get("metric_gate_accepted"):
                report_violations.append("accepted_without_metric_gate")
            if not replay_contract.get("utility_gain_proven"):
                report_violations.append("accepted_without_positive_utility")
            if not replay_contract.get("coherent_replay_gate_proven"):
                report_violations.append("accepted_without_coherent_replay_gates")
            if not replay_contract.get("paired_uid_proven"):
                report_violations.append("accepted_without_paired_replay")
            if not replay_contract.get("minimum_replay_group_coverage_proven"):
                report_violations.append("accepted_without_replay_group_coverage")
            if not selective_contract.get("causal_effect_verified"):
                report_violations.append("accepted_without_causal_replay")
            if not (
                replay_contract.get("fresh_candidate_inference_proven")
                and replay_contract.get("mutation_application_proven")
                and replay_contract.get("metric_gate_accepted")
                and replay_contract.get("utility_gain_proven")
                and replay_contract.get("coherent_replay_gate_proven")
                and replay_contract.get("paired_uid_proven")
                and replay_contract.get("minimum_replay_group_coverage_proven")
                and selective_contract.get("causal_effect_verified")
            ):
                report_violations.append("accepted_without_all_gates")
        contract_violations.extend({
            "report_path": str(report_path),
            "reason": reason,
        } for reason in report_violations)

        if report_violations:
            candidate_status = "contract_violation"
        elif final_status == "accepted":
            candidate_status = "accepted" if (
                replay_contract.get("fresh_candidate_inference_proven")
                and replay_contract.get("mutation_application_proven")
                and replay_contract.get("metric_gate_accepted")
                and replay_contract.get("utility_gain_proven")
                and replay_contract.get("coherent_replay_gate_proven")
                and replay_contract.get("paired_uid_proven")
                and replay_contract.get("minimum_replay_group_coverage_proven")
                and selective_contract.get("causal_effect_verified")
                and candidate_contract.get("dependency_analysis_pass")
            ) else "contract_violation"
        elif snapshot:
            candidate_status = "candidate_pending_replay"
        elif decision == "propose_mutation" and not candidate_contract.get("boundary_pass"):
            candidate_status = "proposal_boundary_violation"
        elif agent_status != "success":
            candidate_status = "evolver_turn_failed"
        elif decision in {"no_mutation", "request_more_cases"}:
            candidate_status = "no_candidate"
        else:
            candidate_status = "proposal_rejected"

        report_records.append({
            "report_path": str(report_path),
            "evolution_root": root_key,
            "run_id": str(payload.get("run_id") or ""),
            "report_structure_pass": report_structure_pass,
            "reference_design_contract_pass": reference_design_contract_pass,
            "reference_design": reference_design,
            "evolver_agent_status": agent_status,
            "evolver_llm_contract_pass": llm_contract_pass,
            "proposal_decision": decision,
            "proposal_validation_status": str(validation.get("status") or ""),
            "candidate_snapshot_present": bool(snapshot),
            "candidate_contract": candidate_contract,
            "failure_attribution": {
                "diagnosis_count": len(attribution_records),
                "complete_count": len(attribution_records) - len(attribution_missing),
                "missing_diagnosis_ids": attribution_missing,
                "contract_pass": not attribution_missing,
            },
            "selective_replay": selective_contract,
            "candidate_replay": replay_contract,
            "replay_artifact_paths": [str(path) for path in artifacts.get("candidate_replays", [])],
            "selective_artifact_paths": [str(path) for path in artifacts.get("selective_artifacts", [])],
            "preflight_artifact_paths": [str(path) for path in artifacts.get("preflight_artifacts", [])],
            "preflight_statuses": preflight_statuses,
            "preflight_contract_pass": not (snapshot and preflight_rejected),
            "final_acceptance_status": final_status,
            "online_publish_status": str(payload.get("online_publish_status") or "not_published"),
            "candidate_status": candidate_status,
            "contract_violations": report_violations,
        })

    candidate_records = [item for item in report_records if item.get("candidate_snapshot_present")]
    accepted_records = [item for item in report_records if item.get("candidate_status") == "accepted"]
    boundary_failures = [
        item for item in report_records
        if item.get("candidate_status") == "proposal_boundary_violation"
    ]
    attribution_failures = [
        item for item in report_records
        if not (item.get("failure_attribution") or {}).get("contract_pass", True)
    ]
    schema_pass_count = sum(bool(item.get("evolver_llm_contract_pass")) for item in report_records)
    fresh_count = sum(
        bool((item.get("candidate_replay") or {}).get("fresh_candidate_inference_proven"))
        for item in candidate_records
    )
    selective_causal_count = sum(
        bool((item.get("selective_replay") or {}).get("causal_effect_verified"))
        for item in candidate_records
    )
    paired_count = sum(
        bool((item.get("candidate_replay") or {}).get("paired_uid_proven"))
        for item in candidate_records
    )
    coverage_count = sum(
        bool((item.get("candidate_replay") or {}).get("minimum_replay_group_coverage_proven"))
        for item in candidate_records
    )
    metric_gate_count = sum(
        bool((item.get("candidate_replay") or {}).get("metric_gate_accepted"))
        for item in candidate_records
    )
    utility_count = sum(
        bool((item.get("candidate_replay") or {}).get("utility_gain_proven"))
        for item in candidate_records
    )
    mutation_count = sum(
        bool((item.get("candidate_replay") or {}).get("mutation_application_proven"))
        for item in candidate_records
    )
    coherent_replay_count = sum(
        bool((item.get("candidate_replay") or {}).get("coherent_replay_gate_proven"))
        for item in candidate_records
    )
    dependency_count = sum(
        bool(item.get("candidate_contract", {}).get("dependency_analysis_pass"))
        for item in candidate_records
    )
    blocking_reasons: List[str] = []
    if not accepted_records:
        blocking_reasons.append("no_final_accepted_candidate")
    if candidate_records and fresh_count < len(candidate_records):
        blocking_reasons.append("fresh_candidate_inference_missing")
    if candidate_records and selective_causal_count < len(candidate_records):
        blocking_reasons.append("selective_suffix_causal_proof_missing")
    if candidate_records and paired_count < len(candidate_records):
        blocking_reasons.append("paired_uid_replay_missing")
    if candidate_records and coverage_count < len(candidate_records):
        blocking_reasons.append("minimum_replay_group_coverage_missing")
    if candidate_records and mutation_count < len(candidate_records):
        blocking_reasons.append("candidate_mutation_application_missing")
    if candidate_records and metric_gate_count < len(candidate_records):
        blocking_reasons.append("metric_replay_gate_missing")
    if candidate_records and utility_count < len(candidate_records):
        blocking_reasons.append("positive_utility_gain_missing")
    if candidate_records and coherent_replay_count < len(candidate_records):
        blocking_reasons.append("coherent_replay_gates_missing")
    if candidate_records and dependency_count < len(candidate_records):
        blocking_reasons.append("dependency_analysis_missing_or_failed")
    reference_design_failures = [
        item for item in report_records
        if not item.get("reference_design_contract_pass")
    ]
    if reference_design_failures:
        blocking_reasons.append("reference_evolver_design_contract_failed")
    if boundary_failures:
        blocking_reasons.append("proposal_boundary_violation")
    if attribution_failures:
        blocking_reasons.append("failure_attribution_incomplete")
    if malformed_reports:
        blocking_reasons.append("malformed_evolution_report")

    contract_status = "fail" if contract_violations or boundary_failures or reference_design_failures else "pass"
    if malformed_reports:
        contract_status = "fail"
    acceptance_status = "accepted" if accepted_records else "pending"
    status = "fail" if contract_status == "fail" else acceptance_status
    return {
        "audit_item": "evolver",
        "status": status,
        "audit_scan_pass": not malformed_reports,
        "contract_status": contract_status,
        "acceptance_status": acceptance_status,
        "utility_status": "proven" if accepted_records else "not_proven",
        "formal_acceptance_pass": bool(accepted_records),
        "report_count": len(report_records),
        "malformed_report_count": len(malformed_reports),
        "candidate_count": len(candidate_records),
        "accepted_candidate_count": len(accepted_records),
        "evolver_llm_contract_pass_count": schema_pass_count,
        "failure_attribution_pass_count": sum(
            bool((item.get("failure_attribution") or {}).get("contract_pass"))
            for item in report_records
        ),
        "selective_causal_proof_count": selective_causal_count,
        "fresh_candidate_inference_count": fresh_count,
        "paired_uid_coverage_count": paired_count,
        "minimum_replay_group_coverage_count": coverage_count,
        "mutation_application_proof_count": mutation_count,
        "metric_gate_accepted_count": metric_gate_count,
        "utility_gain_proof_count": utility_count,
        "coherent_replay_gate_count": coherent_replay_count,
        "contract_violation_count": len(contract_violations),
        "reference_design": reference_design,
        "reference_design_contract_pass_count": sum(
            bool(item.get("reference_design_contract_pass"))
            for item in report_records
        ),
        "blocking_reasons": sorted(set(blocking_reasons)),
        "artifact_inventory": {
            root: {
                kind: len(paths)
                for kind, paths in groups.items()
            }
            for root, groups in sorted(artifact_index.items())
        },
        "contract_checks": {
            "proposal_schema_and_kernel": {
                "pass_count": schema_pass_count,
                "report_count": len(report_records),
                "all_pass": bool(report_records) and schema_pass_count == len(report_records),
            },
            "failure_attribution": {
                "pass_count": sum(
                    bool((item.get("failure_attribution") or {}).get("contract_pass"))
                    for item in report_records
                ),
                "all_pass": not attribution_failures,
            },
            "candidate_boundary": {
                "pass": not boundary_failures,
                "boundary_failure_count": len(boundary_failures),
            },
            "reference_design": {
                "pass_count": sum(
                    bool(item.get("reference_design_contract_pass"))
                    for item in report_records
                ),
                "report_count": len(report_records),
                "all_pass": bool(report_records) and not reference_design_failures,
            },
            "selective_suffix": {
                "candidate_count": len(candidate_records),
                "causal_proof_count": selective_causal_count,
                "all_candidates_proven": bool(candidate_records)
                and selective_causal_count == len(candidate_records),
            },
            "fresh_inference": {
                "candidate_count": len(candidate_records),
                "proven_count": fresh_count,
                "all_candidates_proven": bool(candidate_records)
                and fresh_count == len(candidate_records),
            },
            "paired_replay": {
                "candidate_count": len(candidate_records),
                "paired_count": paired_count,
                "coverage_count": coverage_count,
                "all_candidates_proven": bool(candidate_records)
                and paired_count == len(candidate_records)
                and coverage_count == len(candidate_records),
            },
            "mutation_application": {
                "candidate_count": len(candidate_records),
                "proven_count": mutation_count,
                "all_candidates_proven": bool(candidate_records)
                and mutation_count == len(candidate_records),
            },
            "metric_gate": {
                "candidate_count": len(candidate_records),
                "accepted_count": metric_gate_count,
                "all_candidates_accepted": bool(candidate_records)
                and metric_gate_count == len(candidate_records),
            },
            "utility_gain": {
                "candidate_count": len(candidate_records),
                "proven_count": utility_count,
                "all_candidates_proven": bool(candidate_records)
                and utility_count == len(candidate_records),
            },
            "coherent_replay_gates": {
                "candidate_count": len(candidate_records),
                "proven_count": coherent_replay_count,
                "all_candidates_proven": bool(candidate_records)
                and coherent_replay_count == len(candidate_records),
            },
            "final_acceptance": {
                "accepted_count": len(accepted_records),
                "online_publish_statuses": sorted({
                    str(item.get("online_publish_status") or "not_published")
                    for item in report_records
                }),
                "pass": bool(accepted_records),
            },
        },
        "malformed_reports": malformed_reports,
        "contract_violations": contract_violations,
        "reports": report_records,
    }


def _build_audit_items(
    *,
    records: Sequence[Dict[str, Any]],
    scope_kind: str,
    summary_total: Mapping[str, Any],
    aggregate_consistency: Mapping[str, Any],
    tool_coverage: Mapping[str, Any],
    tool_ablation_audit: Mapping[str, Any],
    videomme_scope: Mapping[str, Any],
    evolver_audit: Mapping[str, Any],
    workspaces_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Build the common audit index for core and auxiliary experiments.

    Every item reports correctness and effectiveness separately.  A clean
    implementation with incomplete experimental evidence is therefore
    ``correctness=pass, effectiveness=pending`` rather than an accidental
    success claim.  The detailed evidence remains in the existing sections of
    the same JSON report.
    """

    participated = int(summary_total.get("participated") or 0)
    expected = int(summary_total.get("total_expected") or 0)
    protocol_pass = int(summary_total.get("trajectory_protocol_pass") or 0)
    evidence_contract_pass = int(summary_total.get("evidence_contract_pass") or 0)
    four_role_pass = int(summary_total.get("online_role_contract_pass") or 0)
    leakage_fail = int(summary_total.get("answer_leakage_contract_fail") or 0)
    actual_failures = int(summary_total.get("actual_sample_failure_count") or 0)
    missing_count = max(0, expected - participated - int(summary_total.get("scheduler_worker_placeholder_count") or 0))
    duplicate_uids = []
    # The caller stores duplicate UIDs in inventory; keeping the check here
    # independent makes this helper safe for unit tests with synthetic input.
    seen: Dict[str, int] = defaultdict(int)
    for record in records:
        seen[str(record.get("sample_uid") or "")] += 1
    duplicate_uids = sorted(uid for uid, count in seen.items() if uid and count > 1)

    consistency_mismatches = sum(
        len(aggregate_consistency.get(key) or [])
        for key in (
            "participated_missing_from_progress",
            "progress_extra_uids",
            "participated_missing_from_aggregate",
            "aggregate_extra_uids",
        )
    )
    tool_contract_correct = bool(
        int(tool_coverage.get("declared_base_tool_count") or 0) > 0
        and not tool_coverage.get("unsupported_edge_tools_in_baseline")
        and not tool_coverage.get("unregistered_observed_names")
        and bool(tool_coverage.get("all_observed_execution_contracts_pass"))
    )
    input_correct = bool(
        not duplicate_uids
        and videomme_scope.get("config_contract_pass")
        and leakage_fail == 0
        and all(
            bool((record.get("inference_input") or {}).get("audio_contract_pass", True))
            for record in records
            if str(record.get("status") or "").lower() in {"completed", "wrong", "unanswerable"}
        )
    )
    trajectory_correct = bool(
        participated > 0
        and actual_failures == 0
        and protocol_pass == participated
        and evidence_contract_pass == participated
    )
    four_role_correct = bool(
        participated > 0
        and four_role_pass == participated
    )
    counterfactual_evidence = sorted(
        str(path) for path in workspaces_root.glob("_evolution_audit/counterfactual_*.json")
        if path.is_file()
    )
    proven_roles = set()
    for evidence_path in counterfactual_evidence:
        try:
            payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
            comparison = payload.get("counterfactual") or {}
            pairs = comparison.get("pairs") or []
            role = comparison.get("disabled_role")
            valid = (
                payload.get("kind") == "four_role_agent_value_audit"
                and Path(str(comparison.get("baseline_root") or "")).resolve() == workspaces_root.resolve()
                and bool(pairs) and not comparison.get("skipped")
                and all(isinstance(pair, dict) and pair.get("same_case") is True
                        and pair.get("sample_uid") and pair.get("execution_proven") is True
                        and pair.get("disabled_role") == role for pair in pairs)
                and len({pair["sample_uid"] for pair in pairs}) == len(pairs)
                and sum(float(pair.get("answer_delta_baseline_minus_ablation") or 0) for pair in pairs) > 0
            )
            if valid and role in {"observer", "reflector"}:
                proven_roles.add(role)
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            continue
    framework_effectiveness = "pass" if proven_roles == {"observer", "reflector"} and four_role_correct else "pending"
    representative_tool_effectiveness = bool(
        tool_contract_correct
        and tool_ablation_audit.get("utility_evidence_pass")
    )
    full_inventory_tool_effectiveness = bool(
        tool_contract_correct
        and tool_ablation_audit.get("utility_evidence_pass")
        and not tool_coverage.get("zero_usage_active_tools")
        and not tool_coverage.get("low_usage_observed_tools")
    )
    tool_effectiveness = (
        "pass"
        if (
            representative_tool_effectiveness
            if scope_kind == "representative"
            else full_inventory_tool_effectiveness
        )
        else "pending"
    )
    tool_blockers = []
    if not tool_contract_correct:
        tool_blockers.append("tool_library_schema_or_execution_contract_failed")
    if not tool_ablation_audit.get("utility_evidence_pass"):
        tool_blockers.append("paired_tool_ablation_utility_evidence_required")
    if scope_kind == "full_inventory":
        if tool_coverage.get("zero_usage_active_tools"):
            tool_blockers.append("active_tools_unexercised_in_full_inventory")
        if tool_coverage.get("low_usage_observed_tools"):
            tool_blockers.append("observed_tools_below_full_inventory_usage_threshold")
    case_records = [item for item in records if item.get("dynamic_task")]
    case_correct = bool(
        case_records
        and all(
            bool((item.get("dynamic_task") or {}).get("answer_support_protocol_pass"))
            for item in case_records
        )
    )
    full_run_complete = bool(expected > 0 and missing_count == 0 and actual_failures == 0)
    runtime_error_markers = {
        "Traceback",
        "CUDA out of memory",
        "Memory cgroup out of memory",
        "LLVM",
        "SIGABRT",
        "return_code=-6",
        "returncode=-6",
        "return_code=-9",
        "returncode=-9",
    }
    observed_runtime_errors = sorted(
        marker
        for record in records
        for marker in (record.get("log_markers") or [])
        if marker in runtime_error_markers
    )
    gpu_observed = int(summary_total.get("gpu_telemetry_observed_samples") or 0)

    def item(
        *,
        scope: str,
        correctness: str,
        effectiveness: str,
        evidence: Sequence[str],
        blockers: Sequence[str],
        design_link: str,
    ) -> Dict[str, Any]:
        return {
            "scope": scope,
            "design_link": design_link,
            "correctness": {
                "status": correctness,
                "definition": "实现/契约/数据边界是否满足设计",
            },
            "effectiveness": {
                "status": effectiveness,
                "definition": "是否通过主指标、消融或案例证据证明预期收益",
            },
            "evidence": list(evidence),
            "blocking_reasons": sorted(set(str(value) for value in blockers if value)),
        }

    return {
        "input_and_anti_leakage": item(
            scope="core",
            correctness="pass" if input_correct else "fail",
            effectiveness="not_applicable",
            evidence=["samples[].inference_input", "videomme_scope", "summary_total.answer_leakage_contract_fail"],
            blockers=[] if input_correct else ["input_contract_or_answer_leakage_check_failed"],
            design_link="official dataset input contract and preprocessing boundary",
        ),
        "execution_completeness": item(
            scope="core",
            correctness="pass" if full_run_complete and not consistency_mismatches else "pending",
            effectiveness="pending" if not full_run_complete else "pass",
            evidence=["aggregate_consistency", "summary_total", "inventory"],
            blockers=[] if full_run_complete and not consistency_mismatches else [
                "full_inventory_not_in_terminal_state" if not full_run_complete else "aggregate_or_progress_mismatch"
            ],
            design_link="atomic per-sample result/progress/aggregate workflow",
        ),
        "trajectory_and_evidence": item(
            scope="core",
            correctness="pass" if trajectory_correct else "pending",
            effectiveness="pass" if trajectory_correct and framework_effectiveness == "pass" else "pending",
            evidence=["samples[].trajectory", "samples[].protocol", "samples[].evidence"],
            blockers=[] if trajectory_correct else [
                "trajectory_or_evidence_gate_not_passed_for_all_participated_cases"
            ],
            design_link="Planner→Observer→Reflector→EvidenceVerifier execution contract",
        ),
        "framework_component_ablation": item(
            scope="auxiliary_experiment",
            correctness="pass" if four_role_correct else "pending",
            effectiveness=framework_effectiveness,
            evidence=["summary_total.online_role_contract_pass", *counterfactual_evidence],
            blockers=[] if framework_effectiveness == "pass" else [
                "paired_counterfactual_role_ablation_required"
            ],
            design_link="M4 agent framework ablation and four-role value audit",
        ),
        "tool_library_ablation": item(
            scope="auxiliary_experiment",
            correctness="pass" if tool_contract_correct else "pending",
            effectiveness=tool_effectiveness,
            evidence=["tool_coverage", "samples[].tool_library", "tool_ablation_audit"],
            blockers=[] if tool_effectiveness == "pass" else tool_blockers,
            design_link="A1/A3 real tool contract, coverage, necessity, and quantity ablation",
        ),
        "case_study_and_bad_case_regression": item(
            scope="auxiliary_experiment",
            correctness="pass" if case_correct else "pending",
            effectiveness="pass" if case_correct and evolver_audit.get("utility_status") == "proven" else "pending",
            evidence=["dynamic_task_audit", "samples[].outcome_class", "samples[].evolution"],
            blockers=[] if case_correct else [
                "representative_positive_negative_case_evidence_incomplete"
            ],
            design_link="A2/A5 evolution cases and bad-case root-cause regression",
        ),
        "runtime_resource_and_queue": item(
            scope="core",
            correctness="pass" if actual_failures == 0 and not observed_runtime_errors else "fail",
            effectiveness=(
                "pass"
                if full_run_complete and gpu_observed == participated
                else "pending"
            ),
            evidence=["summary_total.gpu_telemetry_observed_samples", "samples[].gpu_telemetry", "samples[].log_markers"],
            blockers=[] if not observed_runtime_errors and full_run_complete else [
                "gpu_telemetry_or_full_throughput_evidence_incomplete"
                if not observed_runtime_errors
                else "runtime_error_marker_observed"
            ],
            design_link="single-device queue, bounded concurrency, OOM protection, and GPU utilization experiment",
        ),
        "evolver": {
            **dict(evolver_audit),
            "scope": "core_and_auxiliary_experiment",
            "design_link": "online worker-boundary failure attribution→candidate patch→selective replay→matching/regression→accept/rollback",
            "correctness": {
                "status": str(evolver_audit.get("contract_status") or "pending"),
                "definition": "proposal boundary、candidate hook、轨迹匹配、无泄露和 replay 契约正确",
            },
            "effectiveness": {
                "status": "pass" if evolver_audit.get("utility_status") == "proven" else "pending",
                "definition": "fresh candidate inference、paired replay、分组 coverage 和 utility delta 证明收益",
            },
        },
    }


def _aggregate_consistency(
    config: Any,
    records: Sequence[Dict[str, Any]],
    *,
    scope_kind: str = "full_inventory",
) -> Dict[str, Any]:
    expected = {str(item.get("sample_uid")) for item in records}
    participating = {
        str(item.get("sample_uid"))
        for item in records
        if str(item.get("status") or "").lower() in _TERMINAL_SAMPLE_STATUSES
    }
    progress = _read_json(config.progress_path) or {}
    progress_records = progress.get("records") if isinstance(progress.get("records"), dict) else {}
    progress_participating_all = {
        str(uid) for uid, item in progress_records.items()
        if isinstance(item, dict)
        and str(item.get("status") or "").lower() in _TERMINAL_SAMPLE_STATUSES
    }
    aggregate = _read_json(config.results_path) or {}
    aggregate_items = aggregate.get("results") if isinstance(aggregate.get("results"), list) else []
    aggregate_uids_all = {str(item.get("sample_uid")) for item in aggregate_items if isinstance(item, dict)}
    is_scoped = scope_kind != "full_inventory"
    progress_participating = (
        progress_participating_all & expected
        if is_scoped
        else progress_participating_all
    )
    aggregate_uids = aggregate_uids_all & expected if is_scoped else aggregate_uids_all
    official = _read_any_json(config.videomme_official_output_path)
    return {
        "scope_kind": scope_kind,
        "scope_filtered_out_of_scope_records": is_scoped,
        "progress_file_present": config.progress_path.exists(),
        "progress_record_count": len(progress_records),
        "progress_participated_count": len(progress_participating),
        "progress_participated_count_all": len(progress_participating_all),
        "result_aggregate_present": config.results_path.exists(),
        "result_aggregate_count": len(aggregate_items),
        "result_aggregate_count_in_scope": len(aggregate_uids),
        "result_aggregate_count_all": len(aggregate_uids_all),
        "official_videomme_present": config.videomme_official_output_path.exists(),
        "official_videomme_count": len(official) if isinstance(official, list) else 0,
        "expected_count": len(expected),
        "per_sample_participated_count": len(participating),
        "participated_missing_from_progress": sorted(participating - progress_participating)[:20],
        "progress_extra_uids": [] if is_scoped else sorted(progress_participating - expected)[:20],
        "participated_missing_from_aggregate": sorted(participating - aggregate_uids)[:20],
        "aggregate_extra_uids": [] if is_scoped else sorted(aggregate_uids - expected)[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MIX_DIR / "mix_config_multi.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-delivery", action="store_true", help="Exit nonzero unless all delivery and configured accuracy gates pass")
    parser.add_argument(
        "--sample-uids",
        default="",
        help="Comma/newline-separated representative sample UIDs to audit; omit for the full inventory.",
    )
    parser.add_argument(
        "--tool-ablation-root",
        type=Path,
        default=None,
        help="Separate workspace root containing a paired real-tool ablation.",
    )
    parser.add_argument(
        "--tool-ablation-disabled-tool",
        default="",
        help="Tool name disabled in --tool-ablation-root.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    inventories: List[DatasetInventory] = load_inventories(config)
    evolution_config = load_evolution_config(PROJECT_ROOT / "RL/config/evolution.yaml")
    verifier = EvidenceVerifier(evolution_config)
    expected_mode = str(config.pipeline_options.get("mode") or "react").strip().lower()
    all_records: List[Dict[str, Any]] = []
    seen_uids: Dict[str, int] = defaultdict(int)
    for inventory in inventories:
        for subset_name in inventory.ordered_subset_names:
            for sample in inventory.samples_by_subset.get(subset_name, ()):
                seen_uids[sample.sample_uid] += 1
                all_records.append(_sample_audit(sample, _read_json(sample.workspace_root / "mix_sample_result.json"), verifier, expected_mode))

    requested_uids: List[str] = []
    for raw_uid in re.split(r"[,\n\r]+", str(args.sample_uids or "")):
        uid = raw_uid.strip()
        if uid and uid not in requested_uids:
            requested_uids.append(uid)
    all_inventory_uids = {str(record.get("sample_uid") or "") for record in all_records}
    missing_requested_uids = sorted(set(requested_uids) - all_inventory_uids)
    if requested_uids:
        requested_set = set(requested_uids)
        records = [record for record in all_records if str(record.get("sample_uid") or "") in requested_set]
        scope_kind = "representative"
    else:
        records = all_records
        scope_kind = "full_inventory"
    scope_seen_uids = Counter(str(record.get("sample_uid") or "") for record in records)
    duplicates = sorted(uid for uid, count in scope_seen_uids.items() if uid and count > 1)
    all_duplicates = sorted(uid for uid, count in seen_uids.items() if count > 1)
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[str(record.get("dataset_name"))].append(record)
    evolution_reports: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    evolution_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    audit_root = config.workspaces_root / "_evolution_audit"
    for path in sorted(audit_root.rglob("evolution_report.json")):
        payload = _read_json(path)
        if payload is not None:
            replay_status = str(payload.get("replay_status") or "not_run")
            candidate_replay_status = str(payload.get("candidate_replay_status") or "not_run")
            evolver_agent = payload.get("evolver_agent") if isinstance(payload.get("evolver_agent"), dict) else {}
            report_summary = {
                "path": str(path),
                "trajectory_count": payload.get("trajectory_count"),
                "event_count": payload.get("event_count"),
                "replay_status": replay_status,
                "candidate_replay_status": candidate_replay_status,
                "replay_status_consistent": replay_status == candidate_replay_status,
                "evolver_agent_status": evolver_agent.get("status"),
                "evolver_agent_schema_valid": bool(evolver_agent.get("schema_valid")),
                "evolver_agent_kernel": evolver_agent.get("kernel") or {},
                "can_finish_count": sum(bool(item.get("can_finish")) for item in payload.get("verifications", []) if isinstance(item, dict)),
                "verification_count": len(payload.get("verifications", [])) if isinstance(payload.get("verifications"), list) else 0,
            }
            evolution_reports[path.parent.parent.name].append(report_summary)
            for verification in payload.get("verifications", []) or []:
                if not isinstance(verification, dict):
                    continue
                sample_uid = str(verification.get("sample_uid") or "").strip()
                if sample_uid:
                    evolution_by_sample[sample_uid].append({
                        "report_path": str(path),
                        "run_id": payload.get("run_id"),
                        "trajectory_id": verification.get("trajectory_id"),
                        "can_finish": bool(verification.get("can_finish")),
                        "replay_status": replay_status,
                        "candidate_replay_status": candidate_replay_status,
                        "replay_status_consistent": replay_status == candidate_replay_status,
                        "evolver_agent_status": evolver_agent.get("status"),
                        "evolver_agent_schema_valid": bool(evolver_agent.get("schema_valid")),
                        "evolver_agent_kernel": evolver_agent.get("kernel") or {},
                    })

    stale_evolution_report_count = 0
    current_trajectory_evolution_sample_count = 0
    for record in records:
        sample_uid = str(record.get("sample_uid") or "")
        matches = evolution_by_sample.get(sample_uid, [])
        trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), dict) else {}
        current_trajectory_id = str(trajectory.get("trajectory_id") or "").strip()
        current_matches = [
            item for item in matches
            if current_trajectory_id
            and str(item.get("trajectory_id") or "").strip() == current_trajectory_id
        ]
        stale_matches = [item for item in matches if item not in current_matches]
        stale_evolution_report_count += len(stale_matches)
        if current_matches:
            current_trajectory_evolution_sample_count += 1
            record["evolution"] = {
                "status": "observed",
                "report_count": len(current_matches),
                "reports": current_matches,
                "replay_status_consistent": all(item.get("replay_status_consistent") for item in current_matches),
                "replay_accepted": any(item.get("replay_status") == "accepted" for item in current_matches),
                "trajectory_id": current_trajectory_id,
                "stale_report_count": len(stale_matches),
                "stale_reports": stale_matches,
            }
            four_agent = record.get("four_agent_audit")
            roles = four_agent.get("roles") if isinstance(four_agent, dict) else None
            evolver = roles.get("Evolver") if isinstance(roles, dict) else None
            if isinstance(evolver, dict):
                evolver.update({
                    "implementation": "online boundary LLM Evolver Agent + replay gate",
                    "observed": True,
                    "report_count": len(current_matches),
                    "agent_turn_count": len(current_matches),
                    "agent_schema_pass_count": sum(bool(item.get("evolver_agent_schema_valid")) for item in current_matches),
                    "kernel": (current_matches[-1].get("evolver_agent_kernel") or {}) if current_matches else {},
                    "agent_statuses": [item.get("evolver_agent_status") for item in current_matches],
                    "replay_statuses": [item.get("replay_status") for item in current_matches],
                    "candidate_replay_statuses": [item.get("candidate_replay_status") for item in current_matches],
                    "replay_status_consistent": all(item.get("replay_status_consistent") for item in current_matches),
                    "accepted_replay": any(item.get("replay_status") == "accepted" for item in current_matches),
                    "trajectory_id": current_trajectory_id,
                    "stale_report_count": len(stale_matches),
                    "evolver_llm_contract_pass": bool(current_matches) and all(
                        bool(item.get("evolver_agent_schema_valid"))
                        and str(item.get("evolver_agent_status") or "") == "success"
                        for item in current_matches
                    ),
                    "contract_pass": bool(current_matches)
                    and all(item.get("replay_status_consistent") for item in current_matches)
                    and all(
                        bool(item.get("evolver_agent_schema_valid"))
                        and str(item.get("evolver_agent_status") or "") == "success"
                        for item in current_matches
                    ),
                    "status": "pass" if (
                        bool(current_matches)
                        and all(item.get("replay_status_consistent") for item in current_matches)
                        and all(
                            bool(item.get("evolver_agent_schema_valid"))
                            and str(item.get("evolver_agent_status") or "") == "success"
                            for item in current_matches
                        )
                    ) else "fail",
                })
                _refresh_four_agent_contract(record)
        elif stale_matches:
            record["evolution"] = {
                "status": "stale_only",
                "report_count": 0,
                "reports": [],
                "replay_status_consistent": False,
                "replay_accepted": False,
                "trajectory_id": current_trajectory_id,
                "stale_report_count": len(stale_matches),
                "stale_reports": stale_matches,
            }

    all_evolution_summaries = [item for items in evolution_reports.values() for item in items]
    evolution_summary = {
        "status": "not_run" if not all_evolution_summaries else "observed",
        "report_count": len(all_evolution_summaries),
        "sample_count_with_evolution": current_trajectory_evolution_sample_count,
        "sample_count_with_any_sample_uid_report": len(evolution_by_sample),
        "stale_evolution_report_count": stale_evolution_report_count,
        "replay_status_counts": dict(Counter(item.get("replay_status", "not_run") for item in all_evolution_summaries)),
        "candidate_replay_status_counts": dict(Counter(item.get("candidate_replay_status", "not_run") for item in all_evolution_summaries)),
        "replay_status_consistent_count": sum(bool(item.get("replay_status_consistent")) for item in all_evolution_summaries),
        "accepted_replay_count": sum(item.get("replay_status") == "accepted" for item in all_evolution_summaries),
        "evolver_llm_schema_pass_count": sum(
            bool(item.get("evolver_agent_schema_valid"))
            and str(item.get("evolver_agent_status") or "") == "success"
            for item in all_evolution_summaries
        ),
        "evolver_llm_contract_pass": bool(all_evolution_summaries) and all(
            bool(item.get("evolver_agent_schema_valid"))
            and str(item.get("evolver_agent_status") or "") == "success"
            for item in all_evolution_summaries
        ),
        "evolution_protocol_pass": bool(all_evolution_summaries) and all(
            bool(item.get("replay_status_consistent"))
            and bool(item.get("evolver_agent_schema_valid"))
            and str(item.get("evolver_agent_status") or "") == "success"
            for item in all_evolution_summaries
        ),
    }

    videomme_config = config.datasets.get("videomme", {})
    configured_subtitle_mode = str(videomme_config.get("subtitle_mode") or "").strip().lower()
    videomme_scope = {
        "dataset": "VideoMME",
        "configured_subtitle_mode": configured_subtitle_mode,
        "expected_subtitle_mode": "w/o sub.",
        "external_subtitles_allowed": False,
        "config_contract_pass": configured_subtitle_mode in {"w/o sub.", "w/o subtitles", "without subtitles"},
        "inventory_question_count": len(by_dataset.get("videomme", [])),
        "participated_metadata_mismatches": sorted(
            str(record.get("sample_uid") or "")
            for record in by_dataset.get("videomme", [])
            if str(record.get("status") or "").lower() in _TERMINAL_SAMPLE_STATUSES
            and (
                str((record.get("dataset_scope") or {}).get("subtitle_mode") or "").strip().lower()
                not in {"w/o sub.", "w/o subtitles", "without subtitles"}
                or bool((record.get("dataset_scope") or {}).get("external_subtitles_used"))
            )
        ),
    }

    summary_total = _summary(records)
    aggregate_consistency = _aggregate_consistency(
        config,
        records,
        scope_kind=scope_kind,
    )
    tool_coverage = _tool_coverage_audit(records, config.workspaces_root)
    tool_ablation_audit = _tool_ablation_audit(
        records,
        config.workspaces_root,
        args.tool_ablation_root,
        args.tool_ablation_disabled_tool,
    )
    evolver_audit = _evolver_audit(audit_root)
    audit_items = _build_audit_items(
        records=records,
        scope_kind=scope_kind,
        summary_total=summary_total,
        aggregate_consistency=aggregate_consistency,
        tool_coverage=tool_coverage,
        tool_ablation_audit=tool_ablation_audit,
        videomme_scope=videomme_scope,
        evolver_audit=evolver_audit,
        workspaces_root=config.workspaces_root,
    )
    audit_summary = {
        "item_count": len(audit_items),
        "correctness_status_counts": dict(Counter(
            str((item.get("correctness") or {}).get("status") or "pending")
            for item in audit_items.values()
        )),
        "effectiveness_status_counts": dict(Counter(
            str((item.get("effectiveness") or {}).get("status") or "pending")
            for item in audit_items.values()
        )),
        "all_correctness_pass": all(
            str((item.get("correctness") or {}).get("status") or "") == "pass"
            for item in audit_items.values()
        ),
        "all_effectiveness_proven": all(
            str((item.get("effectiveness") or {}).get("status") or "")
            in {"pass", "not_applicable"}
            for item in audit_items.values()
        ),
    }

    report = {
        "kind": "mix_run_audit",
        "generated_at": _now(),
        "config_path": str(config.config_path),
        "workspaces_root": str(config.workspaces_root),
        "expected_pipeline_mode": expected_mode,
        "audit_scope": {
            "kind": scope_kind,
            "requested_sample_uids": requested_uids,
            "missing_requested_sample_uids": missing_requested_uids,
            "scope_contract_pass": bool(
                (not requested_uids and scope_kind == "full_inventory")
                or (
                    requested_uids
                    and not missing_requested_uids
                    and len(records) == len(requested_uids)
                )
            ),
            "full_inventory_count": len(all_records),
            "scoped_record_count": len(records),
        },
        "videomme_scope": videomme_scope,
        "inventory": {
            "dataset_order": [inventory.dataset_name for inventory in inventories],
            "counts_by_dataset": {name: len(items) for name, items in sorted(by_dataset.items())},
            "total_expected": len(records),
            "duplicate_sample_uids": duplicates,
            "full_inventory_total_expected": len(all_records),
            "full_inventory_duplicate_sample_uids": all_duplicates,
        },
        "summary_by_dataset": {name: _summary(items) for name, items in sorted(by_dataset.items())},
        "summary_total": summary_total,
        "aggregate_consistency": aggregate_consistency,
        "evolution_reports": dict(sorted(evolution_reports.items())),
        "evolution_summary": evolution_summary,
        "tool_coverage": tool_coverage,
        "tool_ablation_audit": tool_ablation_audit,
        "audit_items": audit_items,
        "audit_summary": audit_summary,
        "evolver_audit": evolver_audit,
        "missing_sample_uids": [item["sample_uid"] for item in records if item.get("status") == "missing"][:50],
        "failed_sample_uids": [item["sample_uid"] for item in records if str(item.get("status")).lower() == "failed"][:50],
        "dynamic_task_audit": [item for item in records if item.get("dynamic_task")],
        "samples": records,
    }
    acceptance = json.loads(config.config_path.read_text(encoding="utf-8")).get("delivery_acceptance") or {}
    delivery_failures = []
    if scope_kind != "full_inventory":
        delivery_failures.append("full_inventory_required")
    if not audit_summary["all_correctness_pass"] or not audit_summary["all_effectiveness_proven"]:
        delivery_failures.append("audit_items_not_passed")
    if any(summary_total["status_counts"].get(status, 0) for status in ("failed", "pending", "missing")):
        delivery_failures.append("incomplete_or_failed_samples")
    for dataset, minimum in (acceptance.get("minimum_accuracy") or {}).items():
        measured = (report["summary_by_dataset"].get(dataset) or {}).get("attempt_accuracy")
        if measured is None or measured < float(minimum):
            delivery_failures.append(f"accuracy_below_requirement:{dataset}:{measured}<{minimum}")
    if acceptance.get("require_accepted_evolver", True) and not evolution_summary.get("accepted_replay_count"):
        delivery_failures.append("accepted_evolver_required")
    online_state = _read_json(audit_root / "online_state.json") or {}
    if online_state.get("enabled") and acceptance.get("require_accepted_evolver", True):
        active = _read_json(audit_root / "active_snapshot.json") or {}
        active_id = str(active.get("snapshot_id") or "snapshot_000")
        adoption = [record.get("sample_uid") for record in records
                    if (record.get("evolution_harness") or {}).get("snapshot_id") == active_id
                    and (record.get("evolution_harness") or {}).get("snapshot_application_proven") is True]
        report["online_evolution_adoption"] = {"active_snapshot_id": active_id, "sample_uids": adoption}
        if active.get("status") != "accepted" or active_id == "snapshot_000" or not adoption:
            delivery_failures.append("accepted_online_snapshot_not_applied_to_main_samples")
    report["delivery_gate"] = {"passed": not delivery_failures, "blocking_reasons": delivery_failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "audit_scope": scope_kind,
        "expected": len(records),
        "participated": summary_total["participated"],
        "status_counts": summary_total["status_counts"],
        "protocol_pass": summary_total["trajectory_protocol_pass"],
        "evidence_can_finish": summary_total["evidence_can_finish"],
        "evolver_audit_status": evolver_audit.get("status"),
        "evolver_contract_status": evolver_audit.get("contract_status"),
        "evolver_acceptance_status": evolver_audit.get("acceptance_status"),
    }, ensure_ascii=False))
    return 2 if args.require_delivery and delivery_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
