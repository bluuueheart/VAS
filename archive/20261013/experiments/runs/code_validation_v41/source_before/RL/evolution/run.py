"""Command-line entry point for trajectory verification and evolution mining."""
from __future__ import annotations
import argparse
import json
import math
import os
import copy
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .analyzer import (
    build_candidate_snapshot,
    build_candidate_snapshot_from_evolver,
    build_intent,
    diagnose,
    load_sample_pair,
    reconstruct_finish_evidence,
    stable_id,
    utc_now,
)
from .agent_value import analyze_case, summarize_cases, summarize_evolver
from .models import evolver_capability_manifest
from .replay import evaluate_replay
from .selective_replay import build_selective_suffix_plan
from .verifier import EvidenceVerifier, load_evolution_config
from .source_mutation import digest, edited_sources, source_context, materialize_source_snapshot, inherit_snapshot
from .intervention import mutation_fingerprint, recovery_preflight
from .reflection import review_full_trajectory, execution_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine Mix trajectories and create a safe candidate registry snapshot.")
    parser.add_argument("--workspaces-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset directory to mine, or 'all' for the mixed batch root.",
    )
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--parent-snapshot", type=Path)
    parser.add_argument("--history-root", type=Path)
    parser.add_argument(
        "--parent-snapshot-id",
        default="snapshot_000",
        help="Active snapshot id from which this online evolution cycle branches.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def proposal_summaries(verifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only development observations may influence a proposal."""
    fields = ("sample_uid", "trajectory_id", "answer_score", "sample_status", "can_finish", "missing_information", "unresolved_claims")
    return [{key: item.get(key) for key in fields} for item in verifications if not item.get("held_out")]


def held_out_member(sample: Mapping[str, Any], fraction: float) -> bool:
    media = sample.get("source_video_path") or sample.get("video_path")
    if not media:
        raise ValueError("media identity required for held-out isolation")
    media_key = [str(sample.get("dataset_name") or "").lower(), str(Path(media).expanduser().resolve())]
    return int(stable_id("split", media_key).rsplit("_", 1)[1], 16) / 16**12 < fraction


def _batch_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate an Evolver baseline without masking malformed fields."""
    scores: List[float] = []
    latencies: List[float] = []
    calls_valid = True
    total_calls = 0
    successful_calls = 0
    for item in samples:
        try:
            raw_score = item.get("acc")
            if isinstance(raw_score, bool) or raw_score in (None, ""):
                raise ValueError("missing acc")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError("non-finite acc")
            scores.append(score)
        except (TypeError, ValueError):
            scores = []
            break
    for item in samples:
        try:
            raw_latency = item.get("elapsed_seconds")
            if isinstance(raw_latency, bool) or raw_latency in (None, ""):
                raise ValueError("missing elapsed_seconds")
            latency = float(raw_latency)
            if not math.isfinite(latency):
                raise ValueError("non-finite elapsed_seconds")
            latencies.append(latency)
        except (TypeError, ValueError):
            latencies = []
            break
    for item in samples:
        try:
            raw_total = item.get("tool_use_num")
            raw_successful = item.get("successful_tool_calls")
            if (
                isinstance(raw_total, bool)
                or isinstance(raw_successful, bool)
                or raw_total in (None, "")
                or raw_successful in (None, "")
            ):
                raise ValueError("missing call counters")
            total = int(raw_total)
            successful = int(raw_successful)
            if total < 0 or successful < 0 or successful > total:
                raise ValueError("invalid call counters")
            total_calls += total
            successful_calls += successful
        except (TypeError, ValueError):
            calls_valid = False
            break
    return {
        "accuracy": round(_mean(scores), 6) if scores else None,
        "mean_latency_seconds": round(_mean(latencies), 6) if latencies else None,
        "invalid_call_rate": (
            round((total_calls - successful_calls) / max(1, total_calls), 6)
            if calls_valid
            else None
        ),
    }


def _replay_cohort_key(sample: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the metadata-only cohort used to select neighbor cases.

    Neighbor evidence must be made from distinct cases in the same dataset
    and task family.  It must not be synthesized from the annotation answer,
    timestamp, or any other evaluation-only field.
    """

    official_meta = sample.get("official_meta")
    official_meta = official_meta if isinstance(official_meta, Mapping) else {}
    dataset = str(sample.get("dataset_name") or "").strip().lower()
    subset = str(sample.get("subset_name") or "").strip().lower()
    question_type = str(
        sample.get("question_type") or official_meta.get("question_type") or ""
    ).strip().lower()
    return dataset, subset, question_type


def _load_candidate_history(output_root: Path, limit: int = 8) -> List[Dict[str, Any]]:
    """Load compact prior candidates for an Evolver cycle.

    History is advisory context only.  It contains immutable snapshot/config
    records and gate outcomes, never benchmark metadata used by the Planner.
    Missing or malformed historical reports are skipped rather than turning a
    new evolution run into a false candidate failure.
    """

    reports = sorted(
        output_root.rglob("evolution_report.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )[-max(0, int(limit)):]
    history: List[Dict[str, Any]] = []
    for path in reports:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        snapshot = payload.get("candidate_snapshot") or payload.get("rejected_candidate_snapshot")
        if not isinstance(snapshot, Mapping):
            continue
        history_snapshot = copy.deepcopy(dict(snapshot))
        history_snapshot["source_overrides"] = {
            target: {"kind": spec.get("kind"), "edits": [
                {"path": edit.get("path"), "before_sha256": edit.get("before_sha256"),
                 "after_sha256": digest(str(edit.get("after") or ""))}
                for edit in spec.get("edits") or []]}
            for target, spec in (snapshot.get("source_overrides") or {}).items()
        }
        history.append({
            "run_id": str(payload.get("run_id") or path.parent.name),
            "candidate_snapshot": history_snapshot,
            "proposal_validation": dict(payload.get("evolver_proposal_validation") or {}),
            "local_repair_status": payload.get("local_repair_status", "not_run"),
            "proposal_identity": payload.get("proposal_identity") or {},
            "intervention_preflight": payload.get("intervention_preflight") or {},
            # Acceptance outcomes and trace paths may expose previous held-out
            # evaluations. History is proposal-only; validation stays private.
        })
    return history


def _build_evolver_agent() -> FourRoleAgentSuite:
    """Create the explicit Evolver LLM/VLM role on the configured kernel."""
    from agents import FourRoleAgentSuite
    from config.settings import settings
    from core.llm.provider import get_llm_provider
    provider = get_llm_provider(
        provider="transformers",
        model_name=settings.planner_model_path,
        model_role="evolver",
        device=os.environ.get("FOUR_AGENT_DEVICE") or None,
    )
    return FourRoleAgentSuite(provider)


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    config = load_evolution_config(args.config)
    parent_snapshot_id = str(args.parent_snapshot_id or "snapshot_000").strip() or "snapshot_000"
    parent_snapshot = json.loads(args.parent_snapshot.read_text()) if args.parent_snapshot else {}
    if parent_snapshot and (parent_snapshot.get("snapshot_id") != parent_snapshot_id or parent_snapshot.get("status") != "accepted"):
        raise ValueError("parent snapshot identity/status mismatch")
    source_root = Path(__file__).resolve().parents[1]
    parent_workspace = tempfile.TemporaryDirectory(prefix="evolver_parent_")
    proposal_source_root = source_root
    if parent_snapshot.get("source_overrides"):
        proposal_source_root = materialize_source_snapshot(parent_snapshot, source_root, Path(parent_workspace.name) / "RL")
    baseline_path = args.workspaces_root / "_audit" / "tool_library_baseline.json"
    baseline_library = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {}
    if baseline_library.get("baseline_id"):
        os.environ["CFV_REGISTRY_BASELINE_ID"] = str(baseline_library["baseline_id"])
        os.environ["CFV_REGISTRY_BASELINE_SNAPSHOT_PATH"] = str(baseline_path.resolve())
    verifier = EvidenceVerifier(config)
    dataset_arg = str(args.dataset or "all").strip().lower()
    dataset_names = (
        ["lvbench", "videomme", "egoschema", "mlvu", "mvbench"]
        if dataset_arg in {"", "all", "mixed"}
        else [dataset_arg]
    )
    candidate_paths = [
        path
        for dataset_name in dataset_names
        for path in (args.workspaces_root / dataset_name).glob("*/results/run_result.json")
        if path.is_file()
    ]
    cohorts = defaultdict(deque)
    for path in sorted(candidate_paths, key=lambda path: stable_id("cohort", str(path.relative_to(args.workspaces_root)))):
        sample_path = path.parents[1] / "mix_sample_result.json"
        if sample_path.is_file():
            cohorts[_replay_cohort_key(json.loads(sample_path.read_text(encoding="utf-8")))].append(path)
    paths = []
    while cohorts and len(paths) < args.limit:
        for cohort in sorted(list(cohorts)):
            paths.append(cohorts[cohort].popleft())
            if not cohorts[cohort]:
                del cohorts[cohort]
            if len(paths) >= args.limit:
                break
    if not paths:
        raise RuntimeError("No trajectories available for evolution")
    # A stable UID-derived split survives reruns and file timestamp changes.
    # Membership is independent of answer labels and observed success.
    held_out_fraction = float((config.get("diagnosis") or {}).get("held_out_fraction", 0.2))
    if not 0 < held_out_fraction < 1:
        raise ValueError("held_out_fraction must be between zero and one")
    held_out_paths = set()
    for path in paths:
        sample, _ = load_sample_pair(path)
        if held_out_member(sample, held_out_fraction):
            held_out_paths.add(path)

    run_id = stable_id("evolution_run", [str(path) for path in paths] + [utc_now()])
    run_root = args.output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    events_path = run_root / "trajectory_events.jsonl"
    human_log_path = run_root / "evolution.log"
    diagnoses = []
    verifications = []
    evaluated_samples: List[Dict[str, Any]] = []
    agent_value_cases: List[Dict[str, Any]] = []
    run_results_by_trajectory: Dict[str, Dict[str, Any]] = {}
    event_count = 0
    with events_path.open("w", encoding="utf-8") as events:
        for path in paths:
            sample, result = load_sample_pair(path)
            case_analysis = analyze_case(sample, result)
            agent_value_cases.append(case_analysis)
            evaluated_sample = dict(sample)
            # Reports generated before the option-text scoring gate may carry
            # a label-only ``acc=1`` for answers such as ``(D) 7`` when D is
            # actually ``1``.  Evolution gates must consume the same truthful
            # correctness contract as the Mix evaluator and case audit.
            evaluated_sample["acc"] = 1.0 if bool(case_analysis.get("answer_correct")) else 0.0
            evaluated_samples.append(evaluated_sample)
            trajectory = ((result.get("agent") or {}).get("trajectory") or {})
            trajectory_id = str(trajectory.get("trajectory_id") or sample.get("sample_uid"))
            run_results_by_trajectory[trajectory_id] = result
            slots, answer, completion_basis = reconstruct_finish_evidence(trajectory)
            verification = verifier.verify_answerability(
                evidence_slots=slots,
                steps=trajectory.get("steps") or [],
                answer=answer,
                completion_basis=completion_basis,
                query=str(trajectory.get("goal") or ""),
            )
            verification_record = {
                "trajectory_id": trajectory_id,
                "sample_uid": sample.get("sample_uid"),
                "answer_score": float(evaluated_sample.get("acc") or 0.0),
                "sample_status": str(sample.get("status") or "unknown"),
                "held_out": path in held_out_paths,
                **verification.model_dump(mode="json"),
            }
            verifications.append(verification_record)
            sample["_verification"] = verification_record
            evaluated_sample["_verification"] = verification_record
            events.write(json.dumps({
                "event_id": stable_id("event", [trajectory_id, "verification"]),
                "trajectory_id": trajectory_id,
                "timestamp": utc_now(),
                "step_index": len(trajectory.get("steps") or []) + 1,
                "registry_snapshot_id": trajectory.get("registry_snapshot_id") or parent_snapshot_id,
                "component": "evidence_verifier",
                "event_type": "VerifierDecisionEvent",
                "payload": verification_record,
            }, ensure_ascii=False) + "\n")
            event_count += 1
            for index, step in enumerate(trajectory.get("steps") or [], start=1):
                event = {
                    "event_id": stable_id("event", [trajectory_id, index]),
                    "trajectory_id": trajectory_id,
                    "timestamp": step.get("timestamp") or utc_now(),
                    "step_index": index,
                    "registry_snapshot_id": trajectory.get("registry_snapshot_id") or parent_snapshot_id,
                    "component": "reasoner",
                    "event_type": "AnswerEvent" if step.get("action") == "Finish" else "ToolCallFinishedEvent",
                    "payload": step,
                }
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_count += 1
            diagnosis = None if path in held_out_paths else diagnose(evaluated_sample, result, config)
            if diagnosis is not None:
                diagnoses.append(diagnosis)
                events.write(json.dumps({
                    "event_id": stable_id("event", diagnosis.diagnosis_id),
                    "trajectory_id": trajectory_id,
                    "timestamp": utc_now(),
                    "step_index": len(trajectory.get("steps") or []) + 1,
                    "registry_snapshot_id": trajectory.get("registry_snapshot_id") or parent_snapshot_id,
                    "component": "evolver",
                    "event_type": "FailureDiagnosisEvent",
                    "payload": diagnosis.model_dump(mode="json"),
                }, ensure_ascii=False) + "\n")
                event_count += 1

    intent = build_intent(diagnoses, config)
    diagnostic_snapshot = build_candidate_snapshot(intent, parent_snapshot_id) if intent is not None else None
    sample_by_trajectory = {
        str((item.get("_verification") or {}).get("trajectory_id") or ""): item
        for item in evaluated_samples
    }
    trigger_ids = [item.trajectory_ids[0] for item in diagnoses[:1]]
    trigger_id = trigger_ids[0] if trigger_ids else ""
    trigger_cohort = _replay_cohort_key(sample_by_trajectory.get(trigger_id) or {})
    neighbor_ids = [
        str(item["trajectory_id"])
        for item in verifications
        if not bool(item.get("held_out"))
        and str(item.get("trajectory_id") or "") != trigger_id
        and _replay_cohort_key(
            sample_by_trajectory.get(str(item.get("trajectory_id") or "")) or {}
        ) == trigger_cohort
    ] if trigger_id else []
    regression_ids = [
        item["trajectory_id"] for item in verifications
        if not item["held_out"]
        and item["trajectory_id"] != trigger_id
        and item["can_finish"]
        and item["answer_score"] >= 1.0
    ]
    held_out_ids = [item["trajectory_id"] for item in verifications if item["held_out"]]
    replay_groups = {
        "trigger": trigger_ids,
        "neighbor": neighbor_ids,
        "regression": regression_ids,
        "held_out": held_out_ids,
    }
    report: Dict[str, Any] = {
        "run_id": run_id,
        "trajectory_count": len(paths),
        "event_count": event_count,
        "diagnoses": [item.model_dump(mode="json") for item in diagnoses],
        "verifications": verifications,
        "evolution_intent": intent.model_dump(mode="json") if intent else None,
        "candidate_snapshot": None,
        "diagnostic_candidate_snapshot": diagnostic_snapshot.model_dump(mode="json") if diagnostic_snapshot else None,
        "candidate_source": "none",
        "replay_groups": replay_groups,
        "replay_group_contract": {
            "neighbor_definition": "distinct non-held-out case in the same dataset/subset/question-type cohort as the trigger",
            "answer_or_annotation_fields_used_for_selection": False,
            "trigger_sample_uid": (
                sample_by_trajectory.get(trigger_id, {}).get("sample_uid")
                if trigger_id
                else None
            ),
        },
        "replay_status": "not_required",
        "candidate_replay_status": "not_required",
        "selective_replay": {
            "strategy": "selective_suffix_first",
            "status": "pending_candidate_snapshot",
            "matching_cases_required": True,
            "periodic_regression_required": True,
        },
        "metric_gate_status": "not_run",
        "fresh_candidate_inference_proven": False,
        "snapshot_application_proven": False,
        "snapshot_application_requires_fresh_inference": False,
        "final_acceptance_status": "no_candidate",
        "online_publish_status": "not_published",
        "batch_metrics": _batch_metrics(evaluated_samples),
        "source_paths": [str(path) for path in paths],
        "agent_value_analysis": {
            "method": "trajectory evidence accounting; not causal ablation",
            "cases": agent_value_cases,
            "aggregate": summarize_cases(agent_value_cases),
        },
    }
    evolver_summaries = proposal_summaries(verifications)
    evolver_runtime_capabilities = evolver_capability_manifest(
        (config.get("acceptance") or {}) if isinstance(config, dict) else {}
    )
    active_schemas = copy.deepcopy(baseline_library.get("tools") or [])
    for schema in active_schemas:
        if isinstance(schema, dict):
            schema.update(copy.deepcopy((parent_snapshot.get("tool_overrides") or {}).get(schema.get("name"), {})))
    evolver_runtime_capabilities["active_tool_schemas"] = active_schemas
    evolver_runtime_capabilities["source_context"] = source_context(
        proposal_source_root,
        sorted({tool for diagnosis in diagnoses for tool in diagnosis.blamed_tool_ids}),
    )
    evolver_runtime_capabilities["empty_source_sha256"] = digest("")
    evolver_context = {
        "baseline_metrics": _batch_metrics([
            item for item in evaluated_samples if not (item.get("_verification") or {}).get("held_out")
        ]),
        "replay_groups": {name: ids for name, ids in replay_groups.items() if name != "held_out"},
        "replay_group_contract": report["replay_group_contract"],
        "minimum_group_counts": dict((config.get("acceptance") or {}).get("minimum_group_counts") or {}),
        "current_snapshot_id": parent_snapshot_id,
        "active_runtime_policy": copy.deepcopy(parent_snapshot.get("router_policy") or {}),
        "active_runtime_mutations": [
            {"action": "edit_tool_source", "target": target, "kind": spec.get("kind"),
             "paths": [edit["path"] for edit in spec.get("edits") or []]}
            for target, spec in (parent_snapshot.get("source_overrides") or {}).items()
        ] + [{"action": "update_tool_schema", "target": target, "after": copy.deepcopy(value)}
             for target, value in (parent_snapshot.get("tool_overrides") or {}).items()],
        "dependency_graph": evolver_runtime_capabilities.get("dependency_graph") or {},
        "dependency_analysis_required": True,
        "current_verification_config": dict(config.get("verification") or {})
        if isinstance(config, dict)
        else {},
        "diagnostic_candidate_snapshot": (
            diagnostic_snapshot.model_dump(mode="json")
            if diagnostic_snapshot is not None
            else None
        ),
        "fresh_candidate_inference_required": True,
        "snapshot_application_attestation_required": True,
        "snapshot_application_requires_fresh_inference": False,
        "candidate_behavior_change_required": True,
        "replay_strategy": {
            "first_stage": "selective_suffix_replay",
            "trigger_case_full_rerun": "paired fresh baseline/candidate for runtime or coordinated mutations",
            "matching_cases": "future cases sharing metadata-only dataset/subset/question_type cohort",
            "periodic_regression": "required before final acceptance",
            "not_executed_is_not_success": True,
        },
        "candidate_history": _load_candidate_history(args.history_root or args.output_root),
        "online_publish_status": "not_published",
    }
    report["evolver_input_contract"] = {
        "trajectory_summary_fields": sorted(
            {key for item in evolver_summaries for key in item.keys()}
        ),
        "diagnosis_count": len(diagnoses),
        "runtime_capability_manifest": evolver_runtime_capabilities,
        "evolution_context": evolver_context,
        "output_contract": {
            "decision_values": ["propose_mutation", "no_mutation", "request_more_cases"],
            "mutation_encoding": "mutation or mutations (maximum four non-conflicting executable operations)",
            "dependency_analysis_required": True,
            "coordinated_patch_requires_connected_operations": True,
            "joint_replay_same_snapshot_required": True,
            "one_mutation_maximum": False,
            "candidate_preflight_required": True,
            "snapshot_application_attestation_required": True,
            "snapshot_application_requires_fresh_inference": False,
            "selective_suffix_replay_first": True,
            "fresh_candidate_inference_required": True,
            "matching_cases_and_periodic_regression_required": True,
            "online_publish_status": "not_published",
        },
    }
    evolver_result: Dict[str, Any]
    evolver_agent: Optional[FourRoleAgentSuite] = None
    try:
        evolver_agent = _build_evolver_agent()
        trigger_run = run_results_by_trajectory.get(trigger_id)
        if not trigger_run:
            raise ValueError("no development trigger available for full-trajectory reflection")
        reflection = review_full_trajectory(evolver_agent, trigger_run, output_path=run_root / "reflector_review.json")
        _write_json(run_root / "reflector_review.json", reflection)
        report["reflector_review_path"] = str(run_root / "reflector_review.json")
        evolver_context["reflector_review"] = {k: v for k, v in reflection.items() if k != "calls"}
        report["reflector_full_trajectory_processed"] = reflection["full_trajectory_processed"]
        evolver_result = evolver_agent.evolve(
            trajectory_summaries=evolver_summaries,
            deterministic_diagnoses=[item.model_dump(mode="json") for item in diagnoses],
            runtime_capabilities=evolver_runtime_capabilities,
            evolution_context=evolver_context,
            deadline_monotonic=None,
        )
    except Exception as exc:
        evolver_result = {
            "role": "evolver",
            "status": "error",
            "schema_valid": False,
            "payload": {},
            "raw_response": "",
            "error": f"{type(exc).__name__}: {exc}",
            "event": {
                "type": "agent_llm_turn",
                "event_type": "EvolverAgentTurnEvent",
                "agent_role": "evolver",
                "status": "error",
                "schema_valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
    snapshot, evolver_proposal_validation = build_candidate_snapshot_from_evolver(
        evolver_result,
        parent_id=parent_snapshot_id,
        diagnosis_ids=[item.diagnosis_id for item in diagnoses],
    )
    if snapshot:
        try:
            inherited = inherit_snapshot(parent_snapshot, snapshot.model_dump(mode="json"), source_root, proposal_source_root)
            snapshot = type(snapshot).model_validate(inherited)
            edited_sources(snapshot.model_dump(mode="json"), source_root)
        except (ValueError, SyntaxError, OSError) as exc:
            snapshot = None
            evolver_proposal_validation = {"status": "invalid_mutation", "reason": str(exc)}
    if snapshot:
        proposed = snapshot.model_dump(mode="json")
        trigger_run = run_results_by_trajectory.get(trigger_id) or {}
        preflight = recovery_preflight(proposed, trigger_run)
        identity = {"mutation": mutation_fingerprint(proposed), "parent_snapshot_id": parent_snapshot_id,
                    "trigger_digest": digest(json.dumps(execution_record(trigger_run.get("agent") or {}), sort_keys=True)),
                    "source_digest": digest(json.dumps(evolver_runtime_capabilities.get("source_context"), sort_keys=True))}
        history = evolver_context["candidate_history"]
        duplicate = any(item.get("proposal_identity") == identity
                        and (item.get("local_repair_status") == "not_proven"
                             or (item.get("intervention_preflight") or {}).get("eligible") is False)
                        for item in history)
        report["proposal_identity"] = identity
        report["intervention_preflight"] = preflight
        if not preflight["eligible"] or duplicate:
            report["rejected_candidate_snapshot"] = proposed
            report["local_repair_status"] = "not_proven"
            evolver_proposal_validation = {"status": "candidate_not_applicable" if not preflight["eligible"] else "duplicate_rejected_candidate",
                                          "preflight": preflight, "duplicate": duplicate}
            snapshot = None
    report["candidate_snapshot"] = snapshot.model_dump(mode="json") if snapshot else None
    report["evolver_proposal_validation"] = evolver_proposal_validation
    report["candidate_source"] = "evolver_llm" if snapshot else "none"
    if snapshot and trigger_id:
        trigger_result = run_results_by_trajectory.get(trigger_id)
        if isinstance(trigger_result, Mapping):
            report["selective_replay"] = build_selective_suffix_plan(
                trigger_result,
                snapshot.model_dump(mode="json"),
            )
        else:
            report["selective_replay"] = {
                "strategy": "selective_suffix_first",
                "status": "trigger_artifact_unavailable",
                "matching_cases_required": True,
                "periodic_regression_required": True,
            }
    elif snapshot:
        report["selective_replay"] = {
            "strategy": "selective_suffix_first",
            "status": "no_trigger_diagnosis",
            "matching_cases_required": True,
            "periodic_regression_required": True,
        }
    report["replay_status"] = "candidate_pending_fresh_inference" if snapshot else "not_required"
    report["candidate_replay_status"] = report["replay_status"]
    report["final_acceptance_status"] = (
        "candidate_pending_fresh_inference" if snapshot else "no_candidate"
    )
    baseline_report: Optional[Dict[str, Any]] = None
    replay_evaluation: Optional[Dict[str, Any]] = None
    if args.baseline_report:
        baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        if snapshot:
            replay_evaluation = evaluate_replay(baseline_report, report, config)
            replay_evaluation["baseline_report"] = str(args.baseline_report.resolve())
            replay_evaluation["evaluated_candidate_snapshot_id"] = str(
                ((baseline_report.get("candidate_snapshot") or {}).get("snapshot_id") or "")
            )
        else:
            replay_evaluation = {
                "status": "not_required",
                "reason": "no_executable_candidate",
                "baseline_report": str(args.baseline_report.resolve()),
                "evaluated_candidate_snapshot_id": "",
            }
        report["replay_evaluation"] = replay_evaluation
        report["metric_gate_status"] = replay_evaluation["status"] if snapshot else "not_run"
        if snapshot and replay_evaluation["status"] == "accepted":
            report["replay_status"] = "metrics_only_accepted_pending_fresh_inference"
            report["final_acceptance_status"] = "metrics_gate_passed_pending_fresh_inference"
        else:
            report["replay_status"] = replay_evaluation["status"] if snapshot else "not_required"
            report["final_acceptance_status"] = report["replay_status"]
        report["candidate_replay_status"] = report["replay_status"]
    evolver_manifest = (
        evolver_agent.manifest().get("roles", {}).get("evolver", {})
        if evolver_agent is not None
        else {"agent_role": "evolver", "llm_kernel": False}
    )
    evolver_agent_record = {
        "agent_role": "evolver",
        "implementation": "independent LLM meta-agent",
        "status": evolver_result.get("status"),
        "schema_valid": bool(evolver_result.get("schema_valid")),
        "kernel": evolver_manifest,
        "proposal": evolver_result.get("payload") or {},
        "proposal_validation": evolver_proposal_validation,
        "input_contract": report.get("evolver_input_contract") or {},
        "output_contract": {
            "outer_schema_valid": bool(evolver_result.get("schema_valid")),
            "candidate_preflight_required": True,
            "snapshot_application_attestation_required": True,
            "snapshot_application_requires_fresh_inference": False,
            "selective_suffix_replay_first": True,
            "fresh_candidate_inference_required": True,
            "matching_cases_and_periodic_regression_required": True,
            "online_publish_status": "not_published",
        },
        "error": evolver_result.get("error", ""),
    }
    report["evolver_agent"] = evolver_agent_record
    report["agent_value_analysis"]["evolver"] = summarize_evolver(
        evolver_result,
        diagnosis_count=len(diagnoses),
        candidate_snapshot=snapshot.model_dump(mode="json") if snapshot else None,
        replay_status=str(report.get("replay_status") or ""),
        trajectory_count=len(paths),
        proposal_validation=evolver_proposal_validation,
        fresh_candidate_inference_proven=bool(report.get("fresh_candidate_inference_proven")),
    )
    evolver_event = dict(evolver_result.get("event") or {})
    evolver_event.update({
        "event_id": stable_id("event", [run_id, "evolver_agent"]),
        "trajectory_id": "batch",
        "timestamp": utc_now(),
        "step_index": event_count + 1,
        "registry_snapshot_id": snapshot.snapshot_id if snapshot else parent_snapshot_id,
        "component": "evolver",
        "payload": evolver_agent_record,
    })
    with events_path.open("a", encoding="utf-8") as events:
        if intent:
            events.write(json.dumps({
                "event_id": stable_id("event", intent.intent_id),
                "trajectory_id": "batch",
                "timestamp": utc_now(),
                "step_index": event_count + 1,
                "registry_snapshot_id": parent_snapshot_id,
                "component": "evolver",
                "event_type": "EvolutionIntentEvent",
                "payload": intent.model_dump(mode="json"),
            }, ensure_ascii=False) + "\n")
            event_count += 1
        if snapshot:
            events.write(json.dumps({
                "event_id": stable_id("event", snapshot.snapshot_id),
                "trajectory_id": "batch",
                "timestamp": utc_now(),
                "step_index": event_count + 1,
                "registry_snapshot_id": snapshot.snapshot_id,
                "component": "evolver",
                "event_type": "RegistrySnapshotCreatedEvent",
                "payload": snapshot.model_dump(mode="json"),
            }, ensure_ascii=False) + "\n")
            event_count += 1
        if replay_evaluation:
            events.write(json.dumps({
                "event_id": stable_id("event", [run_id, "replay_evaluation"]),
                "trajectory_id": "batch",
                "timestamp": utc_now(),
                "step_index": event_count + 1,
                "registry_snapshot_id": replay_evaluation.get("evaluated_candidate_snapshot_id") or parent_snapshot_id,
                "component": "replay_evaluator",
                "event_type": "ReplayEvaluationEvent",
                "payload": replay_evaluation,
            }, ensure_ascii=False) + "\n")
            event_count += 1
        events.write(json.dumps({
            "event_id": stable_id("event", [run_id, "replay_plan"]),
            "trajectory_id": "batch",
            "timestamp": utc_now(),
            "step_index": event_count + 1,
        "registry_snapshot_id": snapshot.snapshot_id if snapshot else parent_snapshot_id,
                "component": "replay_evaluator",
                "event_type": "ReplayPlanEvent",
                "payload": {
                    "groups": replay_groups,
                    "status": report["replay_status"],
                    "selective_replay": report.get("selective_replay") or {},
                },
            }, ensure_ascii=False) + "\n")
        event_count += 1
        events.write(json.dumps(evolver_event, ensure_ascii=False) + "\n")
        event_count += 1
    report["event_count"] = event_count
    _write_json(run_root / "evolution_report.json", report)
    if replay_evaluation and baseline_report and snapshot:
        baseline_snapshot = baseline_report.get("candidate_snapshot")
        if isinstance(baseline_snapshot, dict) and baseline_snapshot.get("snapshot_id"):
            finalized_snapshot = dict(baseline_snapshot)
            finalized_snapshot["status"] = replay_evaluation["status"]
            snapshot_id = str(finalized_snapshot["snapshot_id"])
            _write_json(
                args.output_root / "registry" / f"{snapshot_id}_{replay_evaluation['status']}" / "snapshot.json",
                finalized_snapshot,
            )
    if snapshot:
        _write_json(args.output_root / "registry" / f"{snapshot.snapshot_id}_candidate" / "snapshot.json", snapshot.model_dump(mode="json"))
    lines: List[str] = [
        "[Evolver] role=online_trajectory_cycle_miner",
        f"run_id={run_id}",
        f"trajectory_count={len(paths)}",
        f"diagnosis_count={len(diagnoses)}",
        f"verification_accepted={sum(1 for item in verifications if item['can_finish'])}/{len(verifications)}",
        f"accepted_correct={sum(1 for item in verifications if item['can_finish'] and item['answer_score'] >= 1.0)}",
        f"accepted_wrong={sum(1 for item in verifications if item['can_finish'] and item['answer_score'] < 1.0)}",
        "replay_groups=" + json.dumps(replay_groups, ensure_ascii=False, sort_keys=True),
        f"candidate_snapshot={snapshot.snapshot_id if snapshot else 'none'}",
        f"replay_status={report['replay_status']}",
    ]
    for item in diagnoses:
        lines.append(f"diagnosis={item.failure_type.value} confidence={item.confidence:.2f} root_cause={item.root_cause}")
    human_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), **{key: report[key] for key in ("trajectory_count", "event_count", "replay_status")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
