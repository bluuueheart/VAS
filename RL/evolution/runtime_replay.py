"""Paired fresh trigger rollouts for mutations without resumable checkpoints.

This deliberately reports full rollouts, never a fictitious executed suffix.
Both arms use the same seed, input contract and current baseline source.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping


def repair_evidence(baseline: Mapping[str, Any], candidate: Mapping[str, Any], objective: str = "auto") -> dict[str, Any]:
    """Measure a trigger repair; changing an action string is not evidence."""
    def metrics(item):
        score = item.get("acc")
        if score is None and item.get("status") == "unanswerable":
            score = 0.0
        calls, successful = item.get("tool_use_num"), item.get("successful_tool_calls")
        values = (score, calls, successful)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("missing/non-finite trigger metrics")
        if not 0 <= score <= 1 or not 0 <= successful <= calls:
            raise ValueError("invalid trigger metrics")
        return score, calls, calls - successful
    try:
        before, after = metrics(baseline), metrics(candidate)
    except ValueError as exc:
        return {"proven": False, "reason": str(exc)}
    latencies = (baseline.get("elapsed_seconds"), candidate.get("elapsed_seconds"))
    timing_gain = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in latencies) and latencies[1] <= latencies[0] * .95
    answer_gain = after[0] > before[0]
    tool_gain = after[2] < before[2]
    efficiency_gain = after[0] > 0 and (after[1] < before[1] or timing_gain)
    improved = answer_gain if before[0] < 1 else (tool_gain or efficiency_gain)
    if objective in {"global_answer_failure", "global_and_local_failure"}:
        improved = answer_gain
    elif objective == "local_tool_failure":
        improved = tool_gain
    elif objective == "efficiency":
        improved = efficiency_gain
    elif objective == "local_verifier_failure":
        improved = answer_gain or (baseline.get("planner_success") is False and candidate.get("planner_success") is True)
    elif objective != "auto":
        return {"proven": False, "reason": "unknown repair objective", "objective": objective}
    completion_regressed = baseline.get("planner_success") is True and candidate.get("planner_success") is False
    return {"proven": after[0] >= before[0] and after[2] <= before[2] and improved and not completion_regressed,
            "completion_regressed": completion_regressed,
            "objective": objective,
            "repair_scope": "answer" if answer_gain else "tool" if tool_gain else "efficiency" if efficiency_gain else "none",
            "answer_repair_proven": answer_gain, "tool_repair_proven": tool_gain,
            "answer_gain": after[0] - before[0], "failed_call_reduction": before[2] - after[2],
            "tool_call_reduction": before[1] - after[1]}


def monitor_regression(request: Mapping[str, Any], monitor: Mapping[str, Any],
                       records: Mapping[str, Mapping[str, Any]], output_root: Path) -> dict[str, Any]:
    """Periodically compare published code with its rollback parent in shadow."""
    retained = set(monitor.get("record_uids") or [])
    old = [uid for uid in records if uid in retained and records[uid]["sample_result"].get("acc") == 1]
    recent = [uid for uid in records if uid not in retained]
    selected = list(dict.fromkeys(old[:3] + recent[-3:]))
    pairs = []
    for uid in selected:
        try:
            pair = paired_runtime_replay(run_result_path=Path(records[uid]["run_result_path"]),
                snapshot_path=Path(monitor["snapshot_path"]), baseline_snapshot_path=Path(monitor["parent_path"]),
                config_path=Path(request["config_path"]), output_root=output_root / "regression" / str(len(pairs)),
                device_id=str(request.get("visible_device_id") or "0"),
                timeout_seconds=int(request.get("timeout_seconds") or 7200))
        except Exception as exc:
            pair = {"status": "runtime_rollout_failed", "sample_uid": uid,
                    "error": f"{type(exc).__name__}: {exc}"}
        pairs.append(pair)
    valid = bool(pairs) and all(p.get("status") == "paired_runtime_rollout_executed"
                              and p.get("snapshot_application_proven")
                              and "answer_gain" in p.get("repair_evidence", {}) for p in pairs)
    regressed = any(p.get("status") == "paired_runtime_rollout_executed"
                    and p.get("snapshot_application_proven")
                    and (p.get("repair_evidence", {}).get("answer_gain", 0) < 0 or
                         p.get("repair_evidence", {}).get("failed_call_reduction", 0) < 0 or
                         p.get("repair_evidence", {}).get("completion_regressed", False))
                    for p in pairs)
    return {"kind": "online_evolution_monitor_result", "snapshot_id": monitor["snapshot_id"],
            "status": "regression_detected" if regressed else "passed" if valid else "not_proven",
            "pairs": pairs, "selected_uids": selected}


def behavior_signature(result: Mapping[str, Any]) -> str:
    trajectory = (result.get("agent") or {}).get("trajectory") or {}
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: clean(item) for key, item in value.items() if not (
                str(key).startswith("__") or key in {"Evidence", "evidence", "timestamp", "trajectory_id", "step_id"}
                or "path" in str(key) or "latency" in str(key) or "duration" in str(key)
            )}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return json.dumps({
        "answer": trajectory.get("final_answer"),
        "success": trajectory.get("success"),
        "actions": [(step.get("action"), clean(step.get("action_args") or {})) for step in trajectory.get("steps") or []],
    }, sort_keys=True, ensure_ascii=False)


def replay_trigger(**kwargs: Any) -> dict[str, Any]:
    """Execute a verifier-only suffix when possible; otherwise fresh runtime."""
    from .selective_replay import build_selective_suffix_plan
    from .analyzer import reconstruct_finish_evidence
    from .harness import load_harness_snapshot, harness_audit_payload
    from .verifier import EvidenceVerifier
    run = json.loads(kwargs["run_result_path"].read_text())
    snapshot = json.loads(kwargs["snapshot_path"].read_text())
    plan = build_selective_suffix_plan(run, snapshot)
    from .intervention import recovery_preflight
    preflight = recovery_preflight(snapshot, run)
    if not preflight["eligible"]:
        return {"status": "mutation_not_triggered", "causal_fix_status": "not_proven",
                "intervention_preflight": preflight, "snapshot_application_proven": False}
    if plan.get("replay_kind") != "verifier_suffix":
        return paired_runtime_replay(**kwargs, local_replay=True)
    trajectory = (run.get("agent") or {}).get("trajectory") or {}
    slots, answer, completion = reconstruct_finish_evidence(trajectory)
    states = [load_harness_snapshot(kwargs.get("baseline_snapshot_path")), load_harness_snapshot(kwargs["snapshot_path"])]
    outcomes = [EvidenceVerifier(state["verifier_config"]).verify_answerability(evidence_slots=slots,
        steps=trajectory.get("steps") or [], answer=answer, completion_basis=completion,
        query=str(trajectory.get("goal") or "")) for state in states]
    measured = json.loads((kwargs["run_result_path"].parents[1] / "mix_sample_result.json").read_text())
    score = measured.get("acc")
    improved = ((score == 0 and outcomes[0].can_finish and not outcomes[1].can_finish)
                or (score == 1 and not outcomes[0].can_finish and outcomes[1].can_finish))
    harness = harness_audit_payload(states[1])
    proven = improved and harness.get("snapshot_application_proven") and plan.get("cut", {}).get("triggered")
    return {"status": "verifier_suffix_executed", "causal_fix_status": "causal_fix_verified" if proven else "not_proven",
            "harness": harness, "snapshot_application_proven": harness.get("snapshot_application_proven"),
            "prefix_reused": True, "no_full_trigger_case_rerun": True,
            "sample_uid": measured.get("sample_uid"), "leakage_audit": plan["leakage_audit"],
            "baseline": outcomes[0].model_dump(mode="json"), "candidate": outcomes[1].model_dump(mode="json")}


def _replay_source_digest():
    """Invalidate completed arms when the installed runtime or prompts change.

    This is separate from checkpoint compatibility: candidate source edits are
    intentionally allowed by the selective-prefix dependency analysis.
    """
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory in ('agents', 'core', 'evolution', 'planner', 'runtime', 'toolkit', 'config'):
        files.extend(path for path in (root / directory).rglob('*')
                     if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml', '.md'})
    mix = root.parent / 'evaluationVideo' / 'Mix'
    files.extend(mix.glob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root.parent)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def paired_runtime_replay(*, run_result_path: Path, snapshot_path: Path, config_path: Path,
                          output_root: Path, device_id: str, timeout_seconds: int,
                          baseline_snapshot_path: Path | None = None,
                          local_replay: bool = False, repair_objective: str = "auto") -> dict[str, Any]:
    from .mix_replay import _read_json, _run_candidate_worker, _relative_workspace, mix_batch

    config = mix_batch.load_config(config_path.resolve())
    sample_result = _read_json(run_result_path.parents[1] / "mix_sample_result.json")
    uid = str(sample_result["sample_uid"])
    sample = mix_batch._find_sample_by_uid(mix_batch.load_inventories(config), uid)
    snapshot = _read_json(snapshot_path)
    output_root.mkdir(parents=True, exist_ok=True)
    from .runtime_checkpoint import runtime_contract_digest, fingerprint
    from .online import _atomic_write_json
    parent_payload = _read_json(baseline_snapshot_path) if baseline_snapshot_path else {"snapshot_id": "snapshot_000"}
    identity = fingerprint({"snapshot": snapshot, "parent": parent_payload,
                            "run": hashlib.sha256(run_result_path.read_bytes()).hexdigest(),
                            "runtime": runtime_contract_digest(), "installed_sources": _replay_source_digest(),
                            "config": hashlib.sha256(config_path.read_bytes()).hexdigest()})
    run_root = output_root / ("paired_trigger_" + identity)
    run_root.mkdir(parents=True, exist_ok=True)
    baseline_path = run_root / "baseline_snapshot.json"
    requested_baseline = baseline_snapshot_path.resolve() if baseline_snapshot_path else None
    if requested_baseline is not None and requested_baseline.is_file():
        # Online cycles branch from the snapshot that was active when the
        # trigger was observed. Replaying against snapshot_000 would compare
        # a candidate with an obsolete runtime after the first accepted patch.
        baseline_path = requested_baseline
    else:
        baseline_path.write_text(json.dumps({
            "snapshot_id": "snapshot_000", "status": "candidate", "router_policy": {},
            "base_tool_library_baseline_id": snapshot.get("base_tool_library_baseline_id"),
        }), encoding="utf-8")
    arms = {}
    checkpoint = None
    checkpoint_capture = None
    def execute(arm, arm_snapshot, resume=None):
        root = run_root / arm
        result_path = root / "candidate_workspaces" / _relative_workspace(sample, config) / "results" / "run_result.json"
        measured_path = result_path.parents[1] / "mix_sample_result.json"
        commit_path = root / "arm_commit.json"
        resume_digest = hashlib.sha256(resume.read_bytes()).hexdigest() if resume else None
        reused = False
        if commit_path.is_file() and result_path.is_file() and measured_path.is_file():
            commit = _read_json(commit_path)
            reused = (commit.get("identity") == identity and commit.get("resume_digest") == resume_digest
                      and commit.get("run_digest") == hashlib.sha256(result_path.read_bytes()).hexdigest()
                      and commit.get("metrics_digest") == hashlib.sha256(measured_path.read_bytes()).hexdigest())
        if reused:
            code, log = 0, root / "candidate_worker.log"
        else:
            code, _, log = _run_candidate_worker(
                config=config, samples=[sample], snapshot=arm_snapshot, output_root=root,
                device_id=device_id, timeout_seconds=timeout_seconds, no_cache=False,
                **({"resume_checkpoint": resume} if resume is not None else {}),
            )
        if code == 0 and result_path.is_file() and measured_path.is_file() and not reused:
            _atomic_write_json(commit_path, {"identity": identity, "resume_digest": resume_digest,
                "run_digest": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "metrics_digest": hashlib.sha256(measured_path.read_bytes()).hexdigest()})
        result = _read_json(result_path) if result_path.exists() else {}
        measured_path = result_path.parents[1] / "mix_sample_result.json"
        measured = _read_json(measured_path) if measured_path.is_file() else {}
        trajectory = (result.get("agent") or {}).get("trajectory") or {}
        return {
            "return_code": code, "log": str(log), "run_result_path": str(result_path),
            "reused_completed_arm": reused,
            "trajectory_id": trajectory.get("trajectory_id"), "executed": bool(trajectory.get("steps")),
            "behavior_signature": behavior_signature(result),
            "harness": (result.get("metrics") or {}).get("evolution_harness") or {},
            "sample_metrics": measured,
            "trajectory": trajectory,
        }
    if local_replay:
        from .runtime_checkpoint import select_checkpoint
        requested_id = str(_read_json(baseline_path).get("snapshot_id") or "snapshot_000")
        checkpoint = select_checkpoint(_read_json(run_result_path), snapshot, requested_id)
        if checkpoint is None:
            # Legacy traces have no complete checkpoints. Capture one real
            # parent run, then execute both suffix arms from the same state.
            captured = execute("checkpoint_capture", baseline_path)
            checkpoint_capture = captured
            if captured["return_code"] != 0:
                return {"status": "runtime_rollout_failed", "causal_fix_status": "not_proven", "reason": "checkpoint_capture_failed"}
            captured_run = _read_json(Path(captured["run_result_path"]))
            checkpoint = select_checkpoint(captured_run, snapshot, requested_id)
            if checkpoint is None:
                return {"status": "runtime_checkpoint_unavailable", "causal_fix_status": "not_proven", "reason": "no valid root-turn checkpoint"}
    for arm, arm_snapshot in (("baseline", baseline_path), ("candidate", snapshot_path)):
        arms[arm] = execute(arm, arm_snapshot, checkpoint)
    baseline, candidate = arms["baseline"], arms["candidate"]
    executed = all(arm["return_code"] == 0 and arm["executed"] for arm in arms.values())
    executed = executed and baseline["trajectory_id"] != candidate["trajectory_id"]
    suffix_proof = {}
    if checkpoint is not None:
        from .runtime_checkpoint import read, fingerprint
        body = read(checkpoint)
        expected_digest = fingerprint(body)
        for name, arm in arms.items():
            events = arm["trajectory"].get("debug_events") or []
            suffix_proof[name] = any(e.get("type") == "runtime_suffix_restored" and e.get("checkpoint_digest") == expected_digest for e in events)
            suffix_proof[name] = suffix_proof[name] and len(arm["trajectory"].get("steps") or []) > body["prefix_steps"]
        executed = executed and all(suffix_proof.values())
    applied = bool(candidate["harness"].get("snapshot_application_proven")) and candidate["harness"].get("snapshot_id") == snapshot.get("snapshot_id")
    changed = baseline["behavior_signature"] != candidate["behavior_signature"]
    repair = repair_evidence(baseline["sample_metrics"], candidate["sample_metrics"], repair_objective)
    from .intervention import conditional_intervention
    intervention = conditional_intervention(snapshot, candidate["trajectory"],
                                            prefix_steps=body["prefix_steps"] if checkpoint else 0,
                                            baseline_trajectory=baseline["trajectory"])
    from .selective_replay import build_selective_suffix_plan
    leakage = build_selective_suffix_plan(_read_json(run_result_path), snapshot)["leakage_audit"]
    return {
        "status": ("runtime_suffix_executed" if checkpoint is not None else "paired_runtime_rollout_executed") if executed else "runtime_rollout_failed",
        "causal_fix_status": "causal_fix_verified" if executed and applied and intervention["proven"] and repair["proven"] else "not_proven",
        "intervention_proof": intervention,
        "repair_evidence": repair, "leakage_audit": leakage,
        "snapshot_application_proven": applied,
        "harness": candidate["harness"], "arms": arms, "sample_uid": uid,
        "behavior_changed": changed, "prefix_reused": bool(checkpoint is not None and body["prefix_steps"] > 0 and executed),
        "no_full_trigger_case_rerun": bool(checkpoint is not None and body["prefix_steps"] > 0 and executed and checkpoint_capture is None),
        "checkpoint_capture": ({key: checkpoint_capture[key] for key in
            ("return_code", "run_result_path", "reused_completed_arm")} if checkpoint_capture else None),
        "checkpoint_path": str(checkpoint) if checkpoint else None, "suffix_proof": suffix_proof,
        "method": "same-checkpoint paired runtime suffix" if checkpoint is not None else "same-input same-seed fresh active-baseline/candidate rollouts",
        "baseline_snapshot_id": str(_read_json(baseline_path).get("snapshot_id") or "snapshot_000"),
    }
