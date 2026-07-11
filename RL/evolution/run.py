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
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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
    event_count = 0
    with events_path.open("w", encoding="utf-8") as events:
        for path in paths:
            sample, result = load_sample_pair(path)
            trajectory = ((result.get("agent") or {}).get("trajectory") or {})
            trajectory_id = str(trajectory.get("trajectory_id") or sample.get("sample_uid"))
            slots, answer, completion_basis = reconstruct_finish_evidence(trajectory)
            verification = verifier.verify_answerability(
                evidence_slots=slots,
                steps=trajectory.get("steps") or [],
                answer=answer,
                completion_basis=completion_basis,
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
        "source_paths": [str(path) for path in paths],
    }
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
