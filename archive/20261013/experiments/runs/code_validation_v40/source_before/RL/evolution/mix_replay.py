"""Execute a real Mix inference with an Evolver candidate snapshot.

``harness_replay`` is intentionally a deterministic verifier replay.  This
module is the separate, expensive path for proving that a candidate also
changes a fresh Planner/Observer/Reflector inference.  It never publishes a
candidate; it only writes an auditable candidate-run report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .replay import evaluate_replay
from .selective_replay import build_selective_suffix_plan
from .verifier import load_evolution_config
from .source_mutation import materialize_source_snapshot


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MIX_ROOT = _PROJECT_ROOT / "evaluationVideo" / "Mix"
if str(_MIX_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIX_ROOT))
import run_mix_batch as mix_batch  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._") or "sample"


def _sample_result_path(result_path: Path) -> Path:
    return result_path.parents[1] / "mix_sample_result.json"


def _baseline_result_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.baseline_report:
        report = _read_json(args.baseline_report)
        for raw_path in report.get("source_paths") or []:
            path = Path(str(raw_path)).expanduser().resolve()
            if path.exists():
                paths.append(path)
    paths.extend(Path(item).expanduser().resolve() for item in (args.baseline_result or []))
    deduplicated: List[Path] = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduplicated.append(path)
    if not deduplicated:
        raise ValueError("No readable baseline run_result.json was supplied")
    return deduplicated


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _batch_metrics(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep missing or malformed metrics explicit for the replay gate."""
    score_values: List[float] = []
    latency_values: List[float] = []
    total_calls = 0
    successful_calls = 0
    calls_valid = True
    for item in samples:
        try:
            raw_score = 0.0 if item.get("status") == "unanswerable" and item.get("acc") is None else item.get("acc")
            if isinstance(raw_score, bool) or raw_score in (None, ""):
                raise ValueError("missing acc")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError("non-finite acc")
            score_values.append(score)
        except (TypeError, ValueError):
            score_values = []
            break
    for item in samples:
        try:
            raw_latency = item.get("elapsed_seconds")
            if isinstance(raw_latency, bool) or raw_latency in (None, ""):
                raise ValueError("missing elapsed_seconds")
            latency = float(raw_latency)
            if not math.isfinite(latency):
                raise ValueError("non-finite elapsed_seconds")
            latency_values.append(latency)
        except (TypeError, ValueError):
            latency_values = []
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
        "accuracy": round(_mean(score_values), 6) if score_values else None,
        "mean_latency_seconds": round(_mean(latency_values), 6) if latency_values else None,
        "invalid_call_rate": (
            round((total_calls - successful_calls) / max(1, total_calls), 6)
            if calls_valid
            else None
        ),
    }


def _relative_workspace(sample: Any, config: Any) -> Path:
    try:
        return sample.workspace_root.resolve().relative_to(config.workspaces_root.resolve())
    except ValueError:
        return Path(_safe_name(sample.dataset_name)) / _safe_name(sample.workspace_root.name)


def _load_baseline_report(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    return _read_json(path) if path else None


def _candidate_verification(sample_result: Mapping[str, Any], run_result: Mapping[str, Any]) -> Dict[str, Any]:
    trajectory = ((run_result.get("agent") or {}).get("trajectory") or {})
    return {
        "trajectory_id": str(trajectory.get("trajectory_id") or sample_result.get("sample_uid") or ""),
        "sample_uid": str(sample_result.get("sample_uid") or ""),
        # Preserve missing/invalid values for replay.evaluate_replay to reject;
        # coercing them to 0.0 would manufacture a numeric gate input.
        "answer_score": 0.0 if sample_result.get("status") == "unanswerable" and sample_result.get("acc") is None else sample_result.get("acc"),
        "sample_status": str(sample_result.get("status") or "unknown"),
    }


def _candidate_run_executed(run_result: Mapping[str, Any]) -> bool:
    trajectory = ((run_result.get("agent") or {}).get("trajectory") or {})
    if not isinstance(trajectory, Mapping) or not str(trajectory.get("trajectory_id") or "").strip():
        return False
    steps = trajectory.get("steps")
    return isinstance(steps, list) and bool(steps) and all(
        isinstance(step, Mapping) and isinstance(step.get("action"), str) and bool(step["action"].strip())
        for step in steps)


def _candidate_trajectory_id(run_result: Mapping[str, Any]) -> str:
    trajectory = ((run_result.get("agent") or {}).get("trajectory") or {})
    return str(trajectory.get("trajectory_id") or "").strip() if isinstance(trajectory, Mapping) else ""


def _evolution_runtime_events(run_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    trajectory = ((run_result.get("agent") or {}).get("trajectory") or {})
    events = trajectory.get("debug_events") if isinstance(trajectory, Mapping) else []
    return [
        dict(event)
        for event in (events or [])
        if isinstance(event, Mapping)
        and str(event.get("type") or "") == "evolution_runtime_route"
    ]


def _selective_replay_proof(
    *,
    artifact_path: Optional[Path],
    snapshot_id: str,
    trigger_trajectory_ids: Sequence[str],
) -> Dict[str, Any]:
    """Validate a separately materialized selective/suffix replay artifact."""

    if artifact_path is None or not artifact_path.exists():
        return {
            "status": "missing",
            "proven": False,
            "reason": "selective suffix artifact is required before final acceptance",
        }
    try:
        payload = _read_json(artifact_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "proven": False, "reason": f"{type(exc).__name__}: {exc}"}
    plan = payload.get("selective_suffix_plan")
    harness = payload.get("harness") if isinstance(payload.get("harness"), Mapping) else {}
    execution = plan.get("execution") if isinstance(plan, Mapping) else {}
    cut = plan.get("cut") if isinstance(plan, Mapping) else {}
    leakage = plan.get("leakage_audit") if isinstance(plan, Mapping) else {}
    checks = {
        "snapshot_match": str(harness.get("snapshot_id") or "") == str(snapshot_id or ""),
        "trajectory_matches_trigger": (
            not trigger_trajectory_ids
            or str(payload.get("trajectory_id") or "") in set(str(item) for item in trigger_trajectory_ids)
        ),
        "triggered": bool(cut.get("triggered")),
        "executed": str(execution.get("status") or "") in {
            "verifier_suffix_executed",
            "runtime_suffix_executed",
            "paired_runtime_rollout_executed",
        },
        "causal_effect": str(execution.get("causal_fix_status") or "") in {
            "causal_effect_verified",
            "causal_fix_verified",
        },
        "leakage_safe": bool(leakage.get("passed")) and not bool(leakage.get("surviving_forbidden_fields")),
    }
    return {
        "status": "verified" if all(checks.values()) else "not_proven",
        "proven": all(checks.values()),
        "artifact_path": str(artifact_path.resolve()),
        "checks": checks,
    }


def _finalize_evolution_report(
    report_path: Path,
    replay_payload: Mapping[str, Any],
    replay_artifact_path: Path,
) -> None:
    """Attach the candidate replay outcome to the proposal report.

    ``run.py`` intentionally stops at proposal/plan.  The replay command is
    the authority for fresh inference and the numeric/causal gates, so the
    report must be finalized from that single artifact before the formal audit
    classifies a candidate as accepted, rejected, or still pending.
    """

    report = _read_json(report_path)
    report_snapshot = report.get("candidate_snapshot")
    report_snapshot_id = (
        str(report_snapshot.get("snapshot_id") or "").strip()
        if isinstance(report_snapshot, Mapping)
        else ""
    )
    replay_snapshot_id = str(replay_payload.get("candidate_snapshot_id") or "").strip()
    if not report_snapshot_id or not replay_snapshot_id or report_snapshot_id != replay_snapshot_id:
        raise ValueError(
            "Candidate replay snapshot does not match the evolution report: "
            f"report={report_snapshot_id!r} replay={replay_snapshot_id!r}"
        )

    replay_status = str(replay_payload.get("replay_status") or "not_proven")
    final_status = str(
        replay_payload.get("final_acceptance_status") or replay_status
    )
    replay_evaluation = replay_payload.get("replay_evaluation")
    replay_evaluation = replay_evaluation if isinstance(replay_evaluation, Mapping) else None
    report.update({
        "replay_status": replay_status,
        "candidate_replay_status": replay_status,
        "final_acceptance_status": final_status,
        "fresh_candidate_inference_proven": bool(
            replay_payload.get("candidate_inference_proven")
        ),
        "snapshot_application_proven": bool(
            replay_payload.get("snapshot_application_proven")
        ),
        "metric_gate_status": str(
            ((replay_evaluation or {}).get("status"))
            or replay_payload.get("metric_gate_status")
            or "not_run"
        ),
        "replay_evaluation": replay_evaluation,
        "candidate_replay_artifact": str(replay_artifact_path.resolve()),
        "candidate_replay_output_root": str(
            replay_payload.get("candidate_output_root") or ""
        ),
        "candidate_batch_metrics": replay_payload.get("candidate_batch_metrics") or {},
        "replay_baseline_metrics": replay_payload.get("baseline_batch_metrics") or {},
        "fresh_baseline_snapshot": replay_payload.get("fresh_baseline_snapshot"),
        "fresh_baseline_result_paths": replay_payload.get("fresh_baseline_result_paths") or [],
        "selective_replay_proof": replay_payload.get("selective_replay_proof") or {},
    })
    report["selective_replay"] = (
        replay_payload.get("selective_replay_preview")
        or report.get("selective_replay")
        or {}
    )
    selective_proof = replay_payload.get("selective_replay_proof")
    selective_proof = selective_proof if isinstance(selective_proof, Mapping) else {}
    selective_path = Path(str(selective_proof.get("artifact_path") or ""))
    if selective_path.is_file():
        try:
            selective_payload = _read_json(selective_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            selective_payload = {}
        selective_plan = selective_payload.get("selective_suffix_plan")
        if isinstance(selective_plan, Mapping):
            merged_plan = dict(report.get("selective_replay") or {})
            merged_plan.update(selective_plan)
            report["selective_replay"] = merged_plan
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_candidate_worker(
    *,
    config: Any,
    samples: Sequence[Any],
    snapshot: Path,
    output_root: Path,
    device_id: str,
    timeout_seconds: int,
    no_cache: bool,
    resume_checkpoint: Optional[Path] = None,
) -> Tuple[int, float, Path]:
    candidate_root = output_root / "candidate_workspaces"
    candidate_root.mkdir(parents=True, exist_ok=True)
    candidate_samples = [
        replace(sample, workspace_root=candidate_root / _relative_workspace(sample, config))
        for sample in samples
    ]
    if resume_checkpoint is not None:
        if len(samples) != 1:
            raise ValueError("a runtime checkpoint belongs to exactly one sample")
        from .runtime_checkpoint import restore_workspace
        for sample in candidate_samples:
            restore_workspace(resume_checkpoint, sample.workspace_root)
    spec_path = output_root / "worker_spec.json"
    mix_batch._write_worker_sample_spec(spec_path, candidate_samples)
    worker_log_path = output_root / "candidate_worker.log"
    command = [
        sys.executable or "python",
        str((_MIX_ROOT / "run_mix_batch.py").resolve()),
        "--config",
        str(config.config_path),
        "--worker-sample-spec-file",
        str(spec_path),
    ]
    if str(device_id or "").strip():
        command.extend(["--worker-device-id", str(device_id)])
    if no_cache:
        command.append("--no-cache")

    worker_env = mix_batch._build_worker_env(config.config_path)
    worker_env.pop("CFV_RUNTIME_RESUME_CHECKPOINT", None)
    if resume_checkpoint is not None:
        if len(samples) != 1:
            raise ValueError("a runtime checkpoint belongs to exactly one sample")
        worker_env["CFV_RUNTIME_RESUME_CHECKPOINT"] = str(resume_checkpoint.resolve())
    worker_env.pop("CFV_RUNTIME_SOURCE_ROOT", None)
    worker_env["CFV_REPLAY_SEED"] = "0"
    # Replay workers must use the explicitly supplied arm snapshot. The live
    # batch worker may export an online active-snapshot pointer; allowing that
    # pointer to win would make candidate/baseline arms non-deterministic.
    worker_env["CFV_ONLINE_EVOLUTION_REPLAY"] = "1"
    worker_env["CFV_REGISTRY_SNAPSHOT_PATH"] = str(snapshot.resolve())
    baseline_snapshot = str(worker_env.get("CFV_REGISTRY_BASELINE_SNAPSHOT_PATH") or "").strip()
    if not baseline_snapshot:
        default_baseline = config.workspaces_root / "_audit" / "tool_library_baseline.json"
        if default_baseline.exists():
            baseline_snapshot = str(default_baseline.resolve())
            worker_env["CFV_REGISTRY_BASELINE_SNAPSHOT_PATH"] = baseline_snapshot
    if baseline_snapshot:
        try:
            baseline_payload = _read_json(Path(baseline_snapshot))
            baseline_id = str(baseline_payload.get("baseline_id") or "").strip()
            if baseline_id:
                worker_env["CFV_REGISTRY_BASELINE_ID"] = baseline_id
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    rl_root = str((_PROJECT_ROOT / "RL").resolve())
    snapshot_payload = _read_json(snapshot)
    if snapshot_payload.get("source_overrides"):
        rl_root = str(materialize_source_snapshot(snapshot_payload, Path(rl_root), output_root / "isolated_runtime"))
        worker_env["CFV_RUNTIME_SOURCE_ROOT"] = rl_root
    current_pythonpath = str(worker_env.get("PYTHONPATH") or "").strip()
    worker_env["PYTHONPATH"] = (
        rl_root + os.pathsep + current_pythonpath if current_pythonpath else rl_root
    )

    output_root.mkdir(parents=True, exist_ok=True)
    with worker_log_path.open("w", encoding="utf-8") as worker_log:
        worker_log.write("[Replay] worker_command=" + " ".join(command) + "\n")
        worker_log.write(f"[Replay] snapshot={snapshot.resolve()}\n")
        worker_log.flush()
        try:
            with mix_batch._hold_sample_worker_admission_slot(
                config=config,
                visible_device_id=device_id,
            ) as queue_wait_seconds:
                worker_log.write(f"[Replay] queue_wait_seconds={queue_wait_seconds:.3f}\n")
                worker_log.flush()
                completed = subprocess.run(
                    command,
                    cwd=str(_PROJECT_ROOT),
                    env=worker_env,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=max(1, int(timeout_seconds)) * max(1, len(samples)),
                )
            return int(completed.returncode), float(queue_wait_seconds), worker_log_path
        except subprocess.TimeoutExpired:
            worker_log.write(f"[Replay] timeout_seconds={timeout_seconds}\n")
            return -124, 0.0, worker_log_path


def _load_candidate_artifacts(
    candidate_samples: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sample_results: List[Dict[str, Any]] = []
    run_results: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for sample in candidate_samples:
        workspace = sample.workspace_root.resolve()
        sample_path = workspace / "mix_sample_result.json"
        run_path = workspace / "results" / "run_result.json"
        try:
            sample_result = _read_json(sample_path)
            run_result = _read_json(run_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            missing.append({
                "sample_uid": str(sample.sample_uid),
                "workspace": str(workspace),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        # Keep the two artifacts paired by UID rather than by list position.
        # If an earlier candidate workspace is missing, compacting both lists
        # and zipping by index can silently attach another sample's trajectory
        # to this result and corrupt the replay audit.
        run_result["_candidate_sample_uid"] = str(sample.sample_uid)
        sample_results.append(sample_result)
        run_results.append(run_result)
    return sample_results, run_results, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--fresh-baseline-snapshot", type=Path)
    parser.add_argument("--baseline-result", action="append", default=[])
    parser.add_argument("--evolution-config", type=Path)
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--evaluate-artifacts-only",
        action="store_true",
        help="re-evaluate an existing candidate workspace without launching a worker",
    )
    parser.add_argument(
        "--selective-replay-artifact",
        type=Path,
        help="verified selective/suffix replay JSON required for final candidate acceptance",
    )
    parser.add_argument(
        "--finalize-evolution-report",
        action="store_true",
        help=(
            "write the replay outcome back to --baseline-report when it is an "
            "evolution_report.json"
        ),
    )
    args = parser.parse_args()

    config = mix_batch.load_config(args.config.resolve())
    snapshot = args.snapshot.resolve()
    if not snapshot.exists():
        raise FileNotFoundError(f"Candidate snapshot does not exist: {snapshot}")
    baseline_paths = _baseline_result_paths(args)
    baseline_report = _load_baseline_report(args.baseline_report)
    inventories = mix_batch.load_inventories(config)
    baseline_records: List[Dict[str, Any]] = []
    samples: List[Any] = []
    for result_path in baseline_paths:
        sample_result = _read_json(_sample_result_path(result_path))
        sample_uid = str(sample_result.get("sample_uid") or "").strip()
        if not sample_uid:
            raise ValueError(f"Baseline sample result has no sample_uid: {result_path}")
        sample = mix_batch._find_sample_by_uid(inventories, sample_uid)
        samples.append(sample)
        baseline_records.append({
            "run_result_path": str(result_path),
            "sample_result_path": str(_sample_result_path(result_path)),
            "sample_uid": sample_uid,
            "sample_result": sample_result,
            "run_result": _read_json(result_path),
        })

    output_root = args.output_root.resolve()
    snapshot_payload = _read_json(snapshot)
    if args.fresh_baseline_snapshot and not args.evaluate_artifacts_only:
        baseline_root = output_root / "fresh_baseline"
        code, _, _ = _run_candidate_worker(config=config, samples=samples,
            snapshot=args.fresh_baseline_snapshot, output_root=baseline_root,
            device_id=args.device_id, timeout_seconds=args.timeout_seconds, no_cache=bool(args.no_cache))
        if code != 0:
            raise RuntimeError("fresh baseline worker failed; historical latency cannot substitute")
        fresh_samples = []
        for record, sample in zip(baseline_records, samples):
            workspace = baseline_root / "candidate_workspaces" / _relative_workspace(sample, config)
            measured = _read_json(workspace / "mix_sample_result.json")
            run = _read_json(workspace / "results" / "run_result.json")
            if measured.get("sample_uid") != record["sample_uid"] or not _candidate_run_executed(run):
                raise ValueError("fresh baseline identity/execution mismatch")
            record["sample_result"], record["run_result"] = measured, run
            record["fresh_run_result_path"] = str(workspace / "results" / "run_result.json")
            fresh_samples.append(measured)
        if baseline_report is not None:
            by_uid = {item["sample_uid"]: item for item in fresh_samples}
            for verification in baseline_report["verifications"]:
                measured = by_uid[verification["sample_uid"]]
                verification["answer_score"] = 0.0 if measured.get("status") == "unanswerable" and measured.get("acc") is None else measured.get("acc")
            baseline_report["batch_metrics"] = _batch_metrics(fresh_samples)
    selective_artifact = args.selective_replay_artifact.resolve() if args.selective_replay_artifact else None
    if selective_artifact is None:
        default_selective_artifact = output_root / "selective_suffix_trigger.json"
        if default_selective_artifact.exists():
            selective_artifact = default_selective_artifact
    if args.evaluate_artifacts_only:
        return_code = 0
        queue_wait_seconds = 0.0
        worker_log_path = output_root / "candidate_worker.log"
    else:
        return_code, queue_wait_seconds, worker_log_path = _run_candidate_worker(
            config=config,
            samples=samples,
            snapshot=snapshot,
            output_root=output_root,
            device_id=str(args.device_id),
            timeout_seconds=int(args.timeout_seconds),
            no_cache=bool(args.no_cache),
        )
    candidate_samples = [
        replace(sample, workspace_root=output_root / "candidate_workspaces" / _relative_workspace(sample, config))
        for sample in samples
    ]
    candidate_workspace_by_uid = {
        str(sample.sample_uid): sample.workspace_root.resolve()
        for sample in candidate_samples
    }
    candidate_sample_results, candidate_run_results, missing = _load_candidate_artifacts(candidate_samples)
    candidate_runs_by_uid = {
        str(run.get("_candidate_sample_uid") or ""): run
        for run in candidate_run_results
        if str(run.get("_candidate_sample_uid") or "").strip()
    }
    candidate_by_uid = {
        str(item.get("sample_uid") or ""): (
            item,
            candidate_runs_by_uid.get(str(item.get("sample_uid") or ""), {}),
        )
        for item in candidate_sample_results
        if str(item.get("sample_uid") or "").strip()
        and str(item.get("sample_uid") or "") in candidate_runs_by_uid
    }
    expected_uids = [str(sample.sample_uid) for sample in samples]
    candidate_uids = [str(item.get("sample_uid") or "") for item in candidate_sample_results]
    candidate_uid_set = set(candidate_uids)
    expected_uid_set = set(expected_uids)
    artifact_pairing_errors: List[str] = []
    if len(candidate_uids) != len(expected_uids):
        artifact_pairing_errors.append(
            f"candidate_count={len(candidate_uids)} expected_count={len(expected_uids)}"
        )
    if len(candidate_uid_set) != len(candidate_uids):
        artifact_pairing_errors.append("candidate_sample_uid_not_unique")
    if candidate_uid_set != expected_uid_set:
        artifact_pairing_errors.append(
            "candidate_sample_uid_set_mismatch"
        )

    per_case: List[Dict[str, Any]] = []
    candidate_verifications: List[Dict[str, Any]] = []
    for baseline_record in baseline_records:
        uid = baseline_record["sample_uid"]
        baseline_sample = baseline_record["sample_result"]
        baseline_run = baseline_record["run_result"]
        candidate_pair = candidate_by_uid.get(uid)
        if candidate_pair is None:
            per_case.append({
                "sample_uid": uid,
                "status": "missing_candidate_artifact",
                "baseline": baseline_sample,
            })
            continue
        candidate_sample, candidate_run = candidate_pair
        baseline_harness = ((baseline_run.get("metrics") or {}).get("evolution_harness") or {})
        candidate_harness = ((candidate_run.get("metrics") or {}).get("evolution_harness") or {})
        candidate_snapshot_id = str(candidate_harness.get("snapshot_id") or "")
        candidate_verification = _candidate_verification(candidate_sample, candidate_run)
        candidate_verifications.append(candidate_verification)
        runtime_events = _evolution_runtime_events(candidate_run)
        per_case.append({
            "sample_uid": uid,
            "status": "completed",
            "baseline": {
                "acc": float(baseline_sample.get("acc") or 0.0),
                "predicted_label": baseline_sample.get("predicted_label"),
                "elapsed_seconds": baseline_sample.get("elapsed_seconds"),
                "harness_snapshot_id": baseline_harness.get("snapshot_id", "snapshot_000"),
                "harness_policy_digest": baseline_harness.get("policy_digest", "baseline"),
            },
            "candidate": {
                "acc": _optional_float(candidate_sample.get("acc")),
                "predicted_label": candidate_sample.get("predicted_label"),
                "elapsed_seconds": candidate_sample.get("elapsed_seconds"),
                "run_result_path": str(
                    candidate_workspace_by_uid[uid] / "results" / "run_result.json"
                ),
                "harness_snapshot_id": candidate_harness.get("snapshot_id", ""),
                "harness_policy_digest": candidate_harness.get("policy_digest", ""),
                "snapshot_id_match": candidate_snapshot_id == str(snapshot_payload.get("snapshot_id") or ""),
                "snapshot_application_proven": bool(
                    candidate_harness.get("snapshot_application_proven")
                ),
                "snapshot_application_proof": dict(
                    candidate_harness.get("snapshot_application_proof")
                    if isinstance(candidate_harness.get("snapshot_application_proof"), Mapping)
                    else {}
                ),
                "mutation_applied": bool(candidate_harness.get("mutation_applied")),
                "candidate_inference_executed": _candidate_run_executed(candidate_run),
                "runtime_policy_events": runtime_events,
            },
            "answer_delta": (
                _optional_float(candidate_sample.get("acc"))
                - _optional_float(baseline_sample.get("acc"))
                if _optional_float(candidate_sample.get("acc")) is not None
                and _optional_float(baseline_sample.get("acc")) is not None
                else None
            ),
        })

    current_report = {
        "verifications": candidate_verifications,
        "batch_metrics": _batch_metrics(candidate_sample_results),
    }
    replay_evaluation = None
    if baseline_report is not None and not missing:
        trigger_trajectory_ids = [
            str(item)
            for item in ((baseline_report.get("replay_groups") or {}).get("trigger") or [])
        ]
        replay_evaluation = evaluate_replay(
            baseline_report,
            current_report,
            load_evolution_config(args.evolution_config.resolve() if args.evolution_config else None),
        )
    else:
        trigger_trajectory_ids = []

    selective_preview: Dict[str, Any] = {
        "status": "not_available",
        "proven": False,
    }
    trigger_record = next(
        (
            record for record in baseline_records
            if str(
                (((record["run_result"].get("agent") or {}).get("trajectory") or {}).get("trajectory_id"))
                or ""
            ) in set(trigger_trajectory_ids)
        ),
        None,
    )
    if trigger_record is not None:
        selective_preview = {
            "status": "planned",
            "plan": build_selective_suffix_plan(trigger_record["run_result"], snapshot_payload),
        }
    selective_proof = _selective_replay_proof(
        artifact_path=selective_artifact,
        snapshot_id=str(snapshot_payload.get("snapshot_id") or ""),
        trigger_trajectory_ids=trigger_trajectory_ids,
    )

    baseline_trajectory_ids = {
        str(
            (((record["run_result"].get("agent") or {}).get("trajectory") or {}).get("trajectory_id"))
            or ""
        )
        for record in baseline_records
    }
    candidate_trajectory_ids = [
        _candidate_trajectory_id(run)
        for run in candidate_run_results
    ]
    fresh_trajectory_ids = all(
        trajectory_id
        and trajectory_id not in baseline_trajectory_ids
        for trajectory_id in candidate_trajectory_ids
    ) and len(set(candidate_trajectory_ids)) == len(candidate_trajectory_ids)
    candidate_inference_proven = (
        return_code == 0
        and not missing
        and not artifact_pairing_errors
        and len(candidate_sample_results) == len(samples)
        and fresh_trajectory_ids
        and all(
            _candidate_run_executed(run)
            for run in candidate_run_results
        )
    )
    snapshot_application_proven = (
        return_code == 0
        and not missing
        and not artifact_pairing_errors
        and len(candidate_sample_results) == len(samples)
        and bool(str(snapshot_payload.get("snapshot_id") or "").strip())
        and all(
            bool(
                ((run.get("metrics") or {}).get("evolution_harness") or {}).get(
                    "snapshot_application_proven"
                )
            )
            and str(
                ((run.get("metrics") or {}).get("evolution_harness") or {}).get("snapshot_id")
                or ""
            )
            == str(snapshot_payload.get("snapshot_id") or "")
            for run in candidate_run_results
        )
    )
    candidate_replay_proven = bool(candidate_inference_proven and snapshot_application_proven)
    timing_validation = {"status": "not_required"}
    if (candidate_replay_proven and replay_evaluation is not None
            and replay_evaluation.get("status") == "accepted"
            and not replay_evaluation.get("positive_answer_gain")):
        if args.fresh_baseline_snapshot and not args.evaluate_artifacts_only:
            from .timing_replay import confirm_latency
            timing_validation = confirm_latency(config=config, samples=samples,
                baseline_snapshot=args.fresh_baseline_snapshot, candidate_snapshot=snapshot,
                output_root=output_root / "timing_validation", device_id=str(args.device_id),
                timeout_seconds=int(args.timeout_seconds), baseline_report=baseline_report,
                candidate_report=current_report,
                evolution_config=load_evolution_config(args.evolution_config.resolve() if args.evolution_config else None))
        else:
            timing_validation = {"proven": False, "reason": "fresh paired timing repetitions required"}
        replay_evaluation["checks"]["stable_latency_gain"] = bool(timing_validation.get("proven"))
        if not timing_validation.get("proven"):
            replay_evaluation["status"] = "rejected"
            replay_evaluation["metric_gate_status"] = "rejected"
    if not candidate_inference_proven:
        replay_status = "not_proven"
    elif not snapshot_application_proven:
        replay_status = "candidate_inference_complete_pending_snapshot_application"
    elif replay_evaluation is None:
        replay_status = "executed_pending_batch_gate"
    else:
        replay_status = str(replay_evaluation.get("status") or "rejected")
        replay_evaluation["fresh_candidate_inference_proven"] = True
        replay_evaluation["acceptance_scope"] = "fresh_candidate_inference_plus_metrics"

    if replay_status == "accepted" and not selective_proof.get("proven"):
        replay_status = "accepted_pending_causal_replay"
    if replay_status == "accepted":
        final_acceptance_status = "accepted"
    elif replay_status in {"rejected", "invalid_pairing"}:
        final_acceptance_status = "rejected"
    elif replay_status == "insufficient_coverage":
        final_acceptance_status = "candidate_inference_complete_insufficient_coverage"
    elif replay_status == "executed_pending_batch_gate":
        final_acceptance_status = "candidate_inference_complete_pending_metrics"
    elif replay_status == "candidate_inference_complete_pending_snapshot_application":
        final_acceptance_status = "candidate_inference_complete_pending_snapshot_application"
    elif replay_status == "accepted_pending_causal_replay":
        final_acceptance_status = "candidate_inference_complete_pending_causal_replay"
    else:
        final_acceptance_status = "candidate_pending_fresh_inference"

    payload = {
        "kind": "evolution_mix_replay",
        "method": (
            "fresh Mix worker inference with explicit candidate harness snapshot; "
            "snapshot application is attested at the runtime boundary, not by a second trigger rerun"
        ),
        "candidate_snapshot": str(snapshot),
        "candidate_snapshot_id": str(_read_json(snapshot).get("snapshot_id") or ""),
        "config": str(args.config.resolve()),
        "baseline_report": str(args.baseline_report.resolve()) if args.baseline_report else "",
        "baseline_result_paths": [str(path) for path in baseline_paths],
        "fresh_baseline_snapshot": str(args.fresh_baseline_snapshot.resolve()) if args.fresh_baseline_snapshot else None,
        "fresh_baseline_result_paths": [record["fresh_run_result_path"] for record in baseline_records if record.get("fresh_run_result_path")],
        "baseline_batch_metrics": (baseline_report or {}).get("batch_metrics") or {},
        "candidate_worker_log": str(worker_log_path.resolve()),
        "candidate_output_root": str(output_root),
        "worker_return_code": return_code,
        "scheduler_queue_wait_seconds": round(queue_wait_seconds, 3),
        "candidate_inference_executed": bool(candidate_inference_proven),
        "fresh_inference_proven": bool(candidate_inference_proven),
        "candidate_inference_proven": bool(candidate_inference_proven),
        "mutation_applied": bool(snapshot_application_proven),
        "snapshot_application_proven": bool(snapshot_application_proven),
        "mutation_application_proven": bool(snapshot_application_proven),
        "candidate_replay_proven": bool(candidate_replay_proven),
        "missing": missing,
        "artifact_pairing_errors": artifact_pairing_errors,
        "baseline_sample_uids": expected_uids,
        "candidate_sample_uids": candidate_uids,
        "candidate_trajectory_ids": candidate_trajectory_ids,
        "fresh_trajectory_ids": fresh_trajectory_ids,
        "per_case": per_case,
        "candidate_batch_metrics": current_report["batch_metrics"],
        "selective_replay_preview": selective_preview,
        "selective_replay_proof": selective_proof,
        "replay_evaluation": replay_evaluation,
        "timing_validation": timing_validation,
        "replay_status": replay_status,
        "final_acceptance_status": final_acceptance_status,
        "online_publish_status": "not_published",
        "runtime_policy_event_count": sum(
            len(item.get("candidate", {}).get("runtime_policy_events") or [])
            for item in per_case
            if isinstance(item, Mapping)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.finalize_evolution_report:
        if args.baseline_report is None:
            raise ValueError("--finalize-evolution-report requires --baseline-report")
        _finalize_evolution_report(args.baseline_report.resolve(), payload, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_inference_executed": payload["candidate_inference_executed"],
        "mutation_applied": payload["mutation_applied"],
        "replay_status": replay_status,
        "case_count": len(per_case),
    }, ensure_ascii=False))
    return 0 if candidate_replay_proven else 2


if __name__ == "__main__":
    raise SystemExit(main())
