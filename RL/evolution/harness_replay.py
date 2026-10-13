"""Replay one recorded trajectory through a mutated verifier harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .harness import harness_audit_payload, load_harness_snapshot
from .selective_replay import build_selective_suffix_plan


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Mix config for real runtime mutation replay")
    parser.add_argument("--baseline-snapshot", type=Path, help="Active parent snapshot for paired replay")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument(
        "--mode",
        choices=("auto", "verifier", "selective_plan"),
        default="auto",
        help="auto runs deterministic verifier replay and writes the suffix plan; selective_plan only writes the plan",
    )
    args = parser.parse_args()
    run_result = _read_json(args.run_result)
    snapshot_payload = _read_json(args.snapshot)
    sample_path = args.run_result.parents[1] / "mix_sample_result.json"
    sample_result = _read_json(sample_path) if sample_path.exists() else {}
    trajectory = ((run_result.get("agent") or {}).get("trajectory") or {})
    if not isinstance(trajectory, dict):
        raise ValueError("run_result does not contain agent.trajectory")
    state = load_harness_snapshot(args.snapshot)
    selective_plan = build_selective_suffix_plan(run_result, snapshot_payload)
    harness_payload = harness_audit_payload(state)
    payload = {
        "kind": "evolution_harness_replay",
        "mode": args.mode,
        "run_result_path": str(args.run_result.resolve()),
        "snapshot_path": str(args.snapshot.resolve()),
        "sample_uid": str(sample_result.get("sample_uid") or run_result.get("sample_uid") or ""),
        "trajectory_id": str(trajectory.get("trajectory_id") or ""),
        "harness": harness_payload,
        "snapshot_application_proven": bool(harness_payload.get("snapshot_application_proven")),
        "snapshot_application_scope": "isolated_harness_preflight",
        "selective_suffix_plan": selective_plan,
    }
    runtime_required = selective_plan.get("replay_kind") != "verifier_suffix"
    if args.mode != "selective_plan":
        if runtime_required and (args.config is None or args.mode == "verifier"):
            raise ValueError("runtime mutation requires auto mode and --config")
        from .runtime_replay import replay_trigger
        execution = replay_trigger(
            run_result_path=args.run_result, snapshot_path=args.snapshot, config_path=args.config,
            output_root=args.output.parent, device_id=args.device_id, timeout_seconds=args.timeout_seconds,
            baseline_snapshot_path=args.baseline_snapshot,
        )
        selective_plan["execution"] = execution
        selective_plan["replay_kind"] = "runtime_suffix" if execution.get("status") == "runtime_suffix_executed" else "verifier_suffix" if execution.get("status") == "verifier_suffix_executed" else "paired_full_runtime"
        selective_plan["leakage_audit"] = execution.get("leakage_audit") or {}
        payload["harness"] = execution.get("harness") or {}
        payload["snapshot_application_proven"] = bool(execution.get("snapshot_application_proven"))
        payload["snapshot_application_scope"] = (
            "verifier_suffix" if execution.get("status") == "verifier_suffix_executed"
            else "runtime_suffix" if execution.get("status") == "runtime_suffix_executed"
            else "paired_runtime_worker")
        payload["behavior_changed"] = bool(execution.get("behavior_changed"))
        payload["selective_suffix_plan"] = selective_plan
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "snapshot_id": state["snapshot_id"],
        "mutation_applied": bool(payload["snapshot_application_proven"]),
        "snapshot_application_proven": bool(payload["snapshot_application_proven"]),
        "selective_replay_status": selective_plan["execution"]["causal_fix_status"],
        "behavior_changed": payload.get("behavior_changed"),
        "candidate_finish_blocked": payload.get("candidate_finish_blocked"),
    }, ensure_ascii=False))
    return 2 if args.mode == "auto" and runtime_required and selective_plan["execution"]["status"] == "runtime_rollout_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
