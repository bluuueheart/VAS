"""Command-line entry point for trajectory verification and evolution mining."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .analyzer import (
    build_candidate_snapshot,
    build_intent,
    diagnose,
    load_sample_pair,
    reconstruct_finish_evidence,
    stable_id,
    utc_now,
)
from .verifier import EvidenceVerifier, load_evolution_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine Mix trajectories and create a safe candidate registry snapshot.")
    parser.add_argument("--workspaces-root", type=Path, required=True)
    parser.add_argument("--dataset", default="mlvu")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _batch_metrics(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    total_calls = sum(int(item.get("tool_use_num") or 0) for item in samples)
    successful_calls = sum(int(item.get("successful_tool_calls") or 0) for item in samples)
    return {
        "accuracy": round(_mean([float(item.get("acc") or 0.0) for item in samples]), 6),
        "mean_latency_seconds": round(_mean([float(item.get("elapsed_seconds") or 0.0) for item in samples]), 6),
        "invalid_call_rate": round((total_calls - successful_calls) / max(1, total_calls), 6),
    }


def _evaluate_replay(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    gates = dict(config.get("acceptance") or {})
    baseline_scores = {str(item.get("sample_uid")): float(item.get("answer_score") or 0.0) for item in baseline.get("verifications") or []}
    current_scores = {str(item.get("sample_uid")): float(item.get("answer_score") or 0.0) for item in current.get("verifications") or []}
    trajectory_to_uid = {str(item.get("trajectory_id")): str(item.get("sample_uid")) for item in baseline.get("verifications") or []}

    group_deltas: Dict[str, float] = {}
    group_counts: Dict[str, int] = {}
    for group, trajectory_ids in (baseline.get("replay_groups") or {}).items():
        uids = [trajectory_to_uid.get(str(item), "") for item in trajectory_ids]
        uids = [uid for uid in uids if uid in baseline_scores and uid in current_scores]
        group_counts[str(group)] = len(uids)
        group_deltas[str(group)] = round(
            _mean([current_scores[uid] for uid in uids]) - _mean([baseline_scores[uid] for uid in uids]),
            6,
        ) if uids else 0.0

    baseline_metrics = dict(baseline.get("batch_metrics") or {})
    current_metrics = dict(current.get("batch_metrics") or {})
    overall_gain = float(current_metrics.get("accuracy") or 0.0) - float(baseline_metrics.get("accuracy") or 0.0)
    invalid_increase = float(current_metrics.get("invalid_call_rate") or 0.0) - float(baseline_metrics.get("invalid_call_rate") or 0.0)
    baseline_latency = float(baseline_metrics.get("mean_latency_seconds") or 0.0)
    latency_increase = (
        (float(current_metrics.get("mean_latency_seconds") or 0.0) - baseline_latency) / baseline_latency
        if baseline_latency > 0 else 0.0
    )
    latency_limit = float(
        gates.get("latency_increase_with_answer_gain", 0.20)
        if overall_gain > 0 else gates.get("latency_increase", 0.10)
    )
    checks = {
        "overall_answer_non_decreasing": overall_gain >= 0.0,
        "trigger_utility_gain": group_counts.get("trigger", 0) == 0 or group_deltas.get("trigger", 0.0) >= float(gates.get("trigger_utility_gain", 0.0)),
        "neighbor_utility_gain": group_counts.get("neighbor", 0) == 0 or group_deltas.get("neighbor", 0.0) >= float(gates.get("neighbor_utility_gain", 0.02)),
        "regression_answer_drop": group_counts.get("regression", 0) == 0 or -group_deltas.get("regression", 0.0) <= float(gates.get("regression_answer_drop", 0.01)),
        "held_out_utility_drop": group_counts.get("held_out", 0) == 0 or -group_deltas.get("held_out", 0.0) <= float(gates.get("held_out_utility_drop", 0.0)),
        "invalid_call_rate_increase": invalid_increase <= float(gates.get("invalid_call_rate_increase", 0.005)),
        "latency_increase": latency_increase <= latency_limit,
    }
    return {
        "status": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "group_utility_deltas": group_deltas,
        "group_counts": group_counts,
        "overall_answer_gain": round(overall_gain, 6),
        "invalid_call_rate_increase": round(invalid_increase, 6),
        "latency_increase": round(latency_increase, 6),
        "latency_limit": round(latency_limit, 6),
    }


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    config = load_evolution_config(args.config)
    verifier = EvidenceVerifier(config)
    paths = sorted(
        (args.workspaces_root / args.dataset).glob("*/results/run_result.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )[-args.limit:]
    if len(paths) != args.limit:
        raise RuntimeError(f"Expected {args.limit} trajectories, found {len(paths)}")
    held_out_fraction = float((config.get("diagnosis") or {}).get("held_out_fraction", 0.2))
    held_out_count = max(1, round(len(paths) * held_out_fraction)) if len(paths) > 1 else 0
    held_out_paths = set(paths[-held_out_count:]) if held_out_count else set()

    run_id = stable_id("evolution_run", [str(path) for path in paths] + [utc_now()])
    run_root = args.output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    events_path = run_root / "trajectory_events.jsonl"
    human_log_path = run_root / "evolution.log"
    diagnoses = []
    verifications = []
    evaluated_samples: List[Dict[str, Any]] = []
    event_count = 0
    with events_path.open("w", encoding="utf-8") as events:
        for path in paths:
            sample, result = load_sample_pair(path)
            evaluated_samples.append(sample)
            trajectory = ((result.get("agent") or {}).get("trajectory") or {})
            trajectory_id = str(trajectory.get("trajectory_id") or sample.get("sample_uid"))
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
                "answer_score": float(sample.get("acc") or 0.0),
                "sample_status": str(sample.get("status") or "unknown"),
                "held_out": path in held_out_paths,
                **verification.model_dump(mode="json"),
            }
            verifications.append(verification_record)
            sample["_verification"] = verification_record
            events.write(json.dumps({
                "event_id": stable_id("event", [trajectory_id, "verification"]),
                "trajectory_id": trajectory_id,
                "timestamp": utc_now(),
                "step_index": len(trajectory.get("steps") or []) + 1,
                "registry_snapshot_id": trajectory.get("registry_snapshot_id") or "snapshot_000",
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
                    "registry_snapshot_id": trajectory.get("registry_snapshot_id") or "snapshot_000",
                    "component": "reasoner",
                    "event_type": "AnswerEvent" if step.get("action") == "Finish" else "ToolCallFinishedEvent",
                    "payload": step,
                }
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_count += 1
            diagnosis = None if path in held_out_paths else diagnose(sample, result, config)
            if diagnosis is not None:
                diagnoses.append(diagnosis)
                events.write(json.dumps({
                    "event_id": stable_id("event", diagnosis.diagnosis_id),
                    "trajectory_id": trajectory_id,
                    "timestamp": utc_now(),
                    "step_index": len(trajectory.get("steps") or []) + 1,
                    "registry_snapshot_id": trajectory.get("registry_snapshot_id") or "snapshot_000",
                    "component": "evolver",
                    "event_type": "FailureDiagnosisEvent",
                    "payload": diagnosis.model_dump(mode="json"),
                }, ensure_ascii=False) + "\n")
                event_count += 1

    intent = build_intent(diagnoses, config)
    snapshot = build_candidate_snapshot(intent, "snapshot_000") if intent is not None else None
    trigger_ids = [item.trajectory_ids[0] for item in diagnoses[:1]]
    neighbor_ids = [item.trajectory_ids[0] for item in diagnoses[1:]]
    regression_ids = [
        item["trajectory_id"] for item in verifications
        if not item["held_out"] and item["can_finish"] and item["answer_score"] >= 1.0
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
        "candidate_snapshot": snapshot.model_dump(mode="json") if snapshot else None,
        "replay_groups": replay_groups,
        "replay_status": "pending" if snapshot else "not_required",
        "candidate_replay_status": "pending" if snapshot else "not_required",
        "batch_metrics": _batch_metrics(evaluated_samples),
        "source_paths": [str(path) for path in paths],
    }
    baseline_report: Optional[Dict[str, Any]] = None
    replay_evaluation: Optional[Dict[str, Any]] = None
    if args.baseline_report:
        baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        replay_evaluation = _evaluate_replay(baseline_report, report, config)
        replay_evaluation["baseline_report"] = str(args.baseline_report.resolve())
        replay_evaluation["evaluated_candidate_snapshot_id"] = str(
            ((baseline_report.get("candidate_snapshot") or {}).get("snapshot_id") or "")
        )
        report["replay_evaluation"] = replay_evaluation
        report["replay_status"] = replay_evaluation["status"]
    with events_path.open("a", encoding="utf-8") as events:
        if intent:
            events.write(json.dumps({
                "event_id": stable_id("event", intent.intent_id),
                "trajectory_id": "batch",
                "timestamp": utc_now(),
                "step_index": event_count + 1,
                "registry_snapshot_id": "snapshot_000",
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
                "registry_snapshot_id": replay_evaluation.get("evaluated_candidate_snapshot_id") or "snapshot_000",
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
            "registry_snapshot_id": snapshot.snapshot_id if snapshot else "snapshot_000",
            "component": "replay_evaluator",
            "event_type": "ReplayPlanEvent",
            "payload": {"groups": replay_groups, "status": report["replay_status"]},
        }, ensure_ascii=False) + "\n")
        event_count += 1
    report["event_count"] = event_count
    _write_json(run_root / "evolution_report.json", report)
    if replay_evaluation and baseline_report:
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
