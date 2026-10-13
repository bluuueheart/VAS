"""Run one bounded online Evolver proposal and its shadow acceptance gate.

This command is launched by :class:`OnlineEvolutionController` only after a
Mix worker group has released the accelerator. It stages immutable copies to
the observed artifacts, mines one proposal, evaluates it on paired trigger,
metadata-matched neighbor, regression, and held-out cases, then emits a
machine-readable result. Publishing remains the controller's atomic parent
process operation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .runtime_replay import replay_trigger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIX_ROOT = PROJECT_ROOT / "evaluationVideo" / "Mix"
if str(MIX_ROOT) not in sys.path:
    sys.path.insert(0, str(MIX_ROOT))
import run_mix_batch as mix_batch  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    from .online import _atomic_write_json
    committed = dict(payload)
    if path.name == "cycle_result.json" and payload.get("status") not in {"cycle_error", "error"}:
        committed["cycle_committed"] = True
    _atomic_write_json(path, committed)


def _safe_name(value: Any) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._") or "sample"


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _stage_records(request: Mapping[str, Any], cycle_root: Path) -> Tuple[Path, Dict[str, Path], Dict[str, Dict[str, Any]]]:
    stage_root = cycle_root / "evolution_input"
    stage_root.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(str(request.get("baseline_path") or "")).expanduser().resolve()
    if baseline_path.is_file():
        _link_or_copy(baseline_path, stage_root / "_audit" / "tool_library_baseline.json")
    staged_by_uid: Dict[str, Path] = {}
    records_by_uid: Dict[str, Dict[str, Any]] = {}
    for index, raw_record in enumerate(request.get("records") or []):
        if not isinstance(raw_record, Mapping):
            continue
        run_path = Path(str(raw_record.get("run_result_path") or "")).expanduser().resolve()
        if not run_path.is_file():
            continue
        sample_result_path = Path(str(raw_record.get("sample_result_path") or "")).expanduser().resolve()
        if not sample_result_path.is_file():
            sample_result_path = run_path.parents[1] / "mix_sample_result.json"
        if not sample_result_path.is_file():
            continue
        sample_result = _read_json(sample_result_path)
        uid = str(sample_result.get("sample_uid") or raw_record.get("sample_uid") or "").strip()
        if not uid or uid in records_by_uid:
            continue
        dataset = _safe_name(sample_result.get("dataset_name") or "unknown")
        sample_dir = stage_root / dataset / f"{_safe_name(uid)}_{index:04d}"
        _link_or_copy(run_path, sample_dir / "results" / "run_result.json")
        _link_or_copy(sample_result_path, sample_dir / "mix_sample_result.json")
        staged_by_uid[uid] = sample_dir
        record = dict(raw_record)
        record["sample_uid"] = uid
        record["run_result_path"] = str(run_path)
        record["sample_result_path"] = str(sample_result_path)
        record["sample_result"] = sample_result
        record["run_result"] = _read_json(run_path)
        records_by_uid[uid] = record
    if not staged_by_uid:
        raise RuntimeError("online evolution cycle has no readable trajectory artifacts")
    return stage_root, staged_by_uid, records_by_uid


def _run_miner(
    *,
    stage_root: Path,
    cycle_root: Path,
    request: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Dict[str, Any]]:
    evolution_root = cycle_root / "evolution"
    evolution_config = Path(str(request.get("evolution_config") or "")).expanduser().resolve()
    command = [
        sys.executable or "python",
        "-m",
        "evolution.run",
        "--workspaces-root",
        str(stage_root),
        "--dataset",
        "all",
        "--limit",
        str(max(1, len(request.get("records") or []))),
        "--output-root",
        str(evolution_root),
        "--config",
        str(evolution_config),
        "--parent-snapshot-id",
        str(request.get("parent_snapshot_id") or "snapshot_000"),
    ]
    if request.get("active_snapshot_path"):
        command.extend(["--parent-snapshot", str(request["active_snapshot_path"])])
    command.extend(["--history-root", str(cycle_root.parent)])
    environment = os.environ.copy()
    environment.pop("CFV_REGISTRY_SNAPSHOT_PATH", None)
    environment["CFV_ONLINE_EVOLUTION_CYCLE"] = "1"
    rl_root = str(PROJECT_ROOT / "RL")
    existing_pythonpath = str(environment.get("PYTHONPATH") or "").strip()
    environment["PYTHONPATH"] = rl_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    log_path = cycle_root / "evolver.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    reports = sorted(
        path for path in evolution_root.glob("evolution_*/evolution_report.json")
        if path.is_file()
    )
    if completed.returncode != 0 or not reports:
        return None, None, {
            "status": "evolver_failed",
            "return_code": int(completed.returncode),
            "log_path": str(log_path),
        }
    report_path = reports[-1]
    report = _read_json(report_path)
    snapshot_payload = report.get("candidate_snapshot")
    if not isinstance(snapshot_payload, Mapping):
        return report, None, {
            "status": "evolver_failed" if (report.get("evolver_proposal_validation") or {}).get("status") == "invalid_schema" else "no_candidate",
            "report_path": str(report_path),
            "return_code": int(completed.returncode),
        }
    snapshot_id = str(snapshot_payload.get("snapshot_id") or "").strip()
    snapshot_path = evolution_root / "registry" / f"{snapshot_id}_candidate" / "snapshot.json"
    if not snapshot_path.is_file():
        return report, None, {
            "status": "candidate_artifact_missing",
            "report_path": str(report_path),
            "return_code": int(completed.returncode),
        }
    return report, snapshot_path, {
        "status": "candidate_ready",
        "report_path": str(report_path),
        "snapshot_path": str(snapshot_path),
        "return_code": int(completed.returncode),
    }


def _trajectory_uid_map(records_by_uid: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for uid, record in records_by_uid.items():
        run_result = record.get("run_result") if isinstance(record.get("run_result"), Mapping) else {}
        trajectory = (run_result.get("agent") or {}).get("trajectory") or {}
        trajectory_id = str(trajectory.get("trajectory_id") or "").strip()
        if trajectory_id:
            mapping[trajectory_id] = str(uid)
    return mapping


def _group_uids(report: Mapping[str, Any], records_by_uid: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    trajectory_to_uid = _trajectory_uid_map(records_by_uid)
    groups: Dict[str, List[str]] = {}
    for group_name, raw_ids in ((report.get("replay_groups") or {}).items() if isinstance(report.get("replay_groups"), Mapping) else []):
        values: List[str] = []
        for raw_id in raw_ids or []:
            uid = trajectory_to_uid.get(str(raw_id).strip())
            if uid and uid not in values:
                values.append(uid)
        groups[str(group_name)] = values
    return groups


def _extend_validation_report(report: Dict[str, Any], records: Mapping[str, Mapping[str, Any]],
                              proposal_uids: Sequence[str], config_path: Path) -> Dict[str, Any]:
    """Add future observations without invoking or exposing them to proposer."""
    from .run import _batch_metrics, _replay_cohort_key, held_out_member
    from .analyzer import reconstruct_finish_evidence
    from .verifier import EvidenceVerifier, load_evolution_config
    config = load_evolution_config(config_path)
    verifier = EvidenceVerifier(config)
    trigger_uid = (report.get("replay_group_contract") or {}).get("trigger_sample_uid")
    if trigger_uid not in records:
        raise ValueError("pending candidate lost its trigger")
    cohort = _replay_cohort_key(records[trigger_uid]["sample_result"])
    groups = {key: [] for key in ("trigger", "neighbor", "regression", "held_out")}
    verifications, samples, paths = [], [], []
    for uid, record in records.items():
        sample, run = record["sample_result"], record["run_result"]
        trajectory = (run.get("agent") or {}).get("trajectory") or {}
        tid = str(trajectory.get("trajectory_id") or uid)
        held = held_out_member(sample, float(config.get("diagnosis", {}).get("held_out_fraction", .2)))
        slots, answer, completion = reconstruct_finish_evidence(trajectory)
        verification = verifier.verify_answerability(evidence_slots=slots, steps=trajectory.get("steps") or [],
            answer=answer, completion_basis=completion, query=str(trajectory.get("goal") or ""))
        score = 0.0 if sample.get("status") == "unanswerable" and sample.get("acc") is None else sample.get("acc")
        verifications.append({"sample_uid": uid, "trajectory_id": tid, "answer_score": score,
                              "sample_status": sample.get("status"), "held_out": held,
                              **verification.model_dump(mode="json")})
        if uid == trigger_uid:
            groups["trigger"].append(tid)
        elif held:
            # Only unseen media/cases contribute to prospective holdout validation.
            if uid not in proposal_uids:
                groups["held_out"].append(tid)
        elif uid not in proposal_uids and _replay_cohort_key(sample) == cohort:
            groups["neighbor"].append(tid)
        elif verification.can_finish and score == 1.0:
            groups["regression"].append(tid)
        samples.append(dict(sample))
        paths.append(str(record["run_result_path"]))
    return {**report, "source_paths": paths, "trajectory_count": len(paths),
            "verifications": verifications, "replay_groups": groups,
            "batch_metrics": _batch_metrics(samples), "validation_kind": "prospective_frozen_candidate"}


def _gate_candidate(
    *,
    request: Mapping[str, Any],
    cycle_root: Path,
    report: Mapping[str, Any],
    report_path: Path,
    snapshot_path: Path,
    records_by_uid: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    options = request.get("options") if isinstance(request.get("options"), Mapping) else {}
    raw_counts = options.get("online_evolution_min_group_counts") if isinstance(options, Mapping) else None
    from .verifier import load_evolution_config
    configured_counts = load_evolution_config(Path(request["evolution_config"])).get("acceptance", {}).get("minimum_group_counts", {})
    minimum_counts = {key: max(1, int(configured_counts.get(key, default)))
                      for key, default in {"trigger": 1, "neighbor": 3, "regression": 3, "held_out": 3}.items()}
    if isinstance(raw_counts, Mapping):
        for key in minimum_counts:
            try:
                configured_value = raw_counts.get(
                    key,
                    raw_counts.get("heldout", minimum_counts[key])
                    if key == "held_out"
                    else minimum_counts[key],
                )
                minimum_counts[key] = max(minimum_counts[key], int(configured_value))
            except (TypeError, ValueError):
                pass
    groups = _group_uids(report, records_by_uid)
    coverage = {
        key: {
            "available": len(groups.get(key, [])),
            "required": minimum_counts[key],
            "passed": len(groups.get(key, [])) >= minimum_counts[key],
        }
        for key in minimum_counts
    }
    if not all(item["passed"] for item in coverage.values()):
        return {
            "status": "insufficient_coverage",
            "coverage": coverage,
            "groups": groups,
            "reason": "trigger, neighbor, regression and held-out cohorts are required before publication",
        }

    # Freeze a bounded, score-independent selection within each already
    # eligible cohort. Replaying every retained/proposal case makes cycle
    # duration grow without improving the declared coverage contract.
    groups = {key: groups[key][:minimum_counts[key]] for key in minimum_counts}
    selected_uids = {uid for values in groups.values() for uid in values}
    selected_ids = {tid for tid, uid in _trajectory_uid_map(records_by_uid).items() if uid in selected_uids}
    report = dict(report)
    report["source_paths"] = [str(records_by_uid[uid]["run_result_path"])
                              for values in groups.values() for uid in values]
    report["verifications"] = [item for item in report.get("verifications", [])
                               if item.get("sample_uid") in selected_uids]
    report["replay_groups"] = {key: [tid for tid in (report.get("replay_groups") or {}).get(key, [])
                                     if tid in selected_ids] for key in groups}
    report["validation_selection"] = {"method": "first available members of each eligible cohort",
                                      "sample_uids": groups, "counts": minimum_counts}
    _write_json(report_path, report)

    trigger_uid = groups.get("trigger", [""])[0]
    trigger_record = records_by_uid.get(trigger_uid) or {}
    trigger_path = Path(str(trigger_record.get("run_result_path") or "")).expanduser().resolve()
    paired: Dict[str, Any]
    proof_path = str((request.get("pending_candidate") or {}).get("trigger_proof_path") or "")
    if proof_path:
        paired = _read_json(Path(proof_path))
        if (paired.get("harness") or {}).get("snapshot_id") != _read_json(snapshot_path).get("snapshot_id") or paired.get("sample_uid") != trigger_uid:
            raise ValueError("frozen trigger proof does not match candidate and trigger")
    elif not trigger_uid or not trigger_path.is_file():
        paired = {"status": "runtime_rollout_failed", "causal_fix_status": "not_proven", "reason": "trigger artifact missing"}
    else:
        active_path = Path(str(request.get("active_snapshot_path") or "")).expanduser().resolve()
        try:
            paired = replay_trigger(
                run_result_path=trigger_path,
                snapshot_path=snapshot_path,
                config_path=Path(str(request["config_path"])).resolve(),
                output_root=cycle_root / "paired_trigger",
                device_id=str(request.get("visible_device_id") or "0"),
                timeout_seconds=int(request.get("timeout_seconds") or 7200),
                baseline_snapshot_path=active_path if active_path.is_file() else None,
            )
        except Exception as exc:
            paired = {"status": "runtime_rollout_failed", "causal_fix_status": "not_proven", "error": f"{type(exc).__name__}: {exc}"}

    snapshot_id = str(_read_json(snapshot_path).get("snapshot_id") or "")
    trigger_trajectory = ((trigger_record.get("run_result") or {}).get("agent") or {}).get("trajectory") or {}
    selective_artifact = cycle_root / "harness_replay.json"
    selective_payload = {
        "kind": "evolution_harness_replay",
        "mode": "online_paired_trigger",
        "run_result_path": str(trigger_path),
        "snapshot_path": str(snapshot_path),
        "trajectory_id": str(trigger_trajectory.get("trajectory_id") or ""),
        "harness": dict(paired.get("harness") or {}),
        "selective_suffix_plan": {
            "replay_kind": "runtime_suffix" if paired.get("status") == "runtime_suffix_executed" else "verifier_suffix" if paired.get("status") == "verifier_suffix_executed" else "paired_full_runtime",
            "cut": {"triggered": True},
            "execution": {
                "status": str(paired.get("status") or "runtime_rollout_failed"),
                "causal_fix_status": str(paired.get("causal_fix_status") or "not_proven"),
                "snapshot_id": snapshot_id,
            },
            "harness": dict(paired.get("harness") or {}),
            "leakage_audit": dict(paired.get("leakage_audit") or {}),
            "prefix_reused": bool(paired.get("prefix_reused")),
        },
    }
    _write_json(selective_artifact, selective_payload)

    # Reuse the formal Mix replay command so online cycles and the final audit
    # share exactly the same fresh-trajectory, application-proof and metric
    # gate contracts. The command receives the proposal report as its baseline
    # cohort and never writes to the live registry.
    replay_output = cycle_root / "replay.json"
    candidate_output_root = cycle_root / "candidate_replay"
    baseline_snapshot = Path(str(request.get("active_snapshot_path") or cycle_root / "baseline_snapshot.json"))
    if not baseline_snapshot.is_file():
        _write_json(baseline_snapshot, {"snapshot_id": "snapshot_000", "status": "candidate"})
    replay_command = [
        sys.executable or "python",
        "-m",
        "evolution.mix_replay",
        "--config",
        str(Path(str(request["config_path"])).resolve()),
        "--snapshot",
        str(snapshot_path),
        "--output-root",
        str(candidate_output_root),
        "--output",
        str(replay_output),
        "--baseline-report",
        str(report_path),
        "--fresh-baseline-snapshot",
        str(baseline_snapshot),
        "--evolution-config",
        str(Path(str(request["evolution_config"])).resolve()),
        "--device-id",
        str(request.get("visible_device_id") or "0"),
        "--timeout-seconds",
        str(int(request.get("timeout_seconds") or 7200)),
        "--selective-replay-artifact",
        str(selective_artifact),
        "--finalize-evolution-report",
    ]
    replay_log = cycle_root / "candidate_replay.log"
    environment = os.environ.copy()
    environment.pop("CFV_REGISTRY_SNAPSHOT_PATH", None)
    environment["CFV_ONLINE_EVOLUTION_CYCLE"] = "1"
    with replay_log.open("w", encoding="utf-8") as handle:
        replay_process = subprocess.run(
            replay_command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    replay_payload = _read_json(replay_output) if replay_output.is_file() else {
        "replay_status": "runtime_rollout_failed",
        "final_acceptance_status": "candidate_pending_fresh_inference",
    }
    causal = str(paired.get("causal_fix_status") or "") in {"causal_effect_verified", "causal_fix_verified"}
    replay_accepted = str(replay_payload.get("replay_status") or "") == "accepted"
    status = "accepted" if causal and replay_accepted and replay_process.returncode == 0 else (
        "insufficient_coverage" if causal and replay_process.returncode == 0 and replay_payload.get("replay_status") == "insufficient_coverage"
        else "replay_error" if replay_process.returncode != 0 else "rejected")
    return {
        "status": status,
        "coverage": coverage,
        "groups": groups,
        "paired_trigger": paired,
        "causal_effect_verified": causal,
        "replay": replay_payload,
        "replay_return_code": int(replay_process.returncode),
        "replay_log": str(replay_log),
        "no_regressions": bool(
            ((replay_payload.get("replay_evaluation") or {}).get("checks") or {}).get(
                "overall_answer_non_decreasing",
                False,
            )
        ),
        "requirement": "paired trigger effect plus formal fresh replay gates on neighbor/regression/held-out cohorts",
    }


def run(request_path: Path, output_path: Path) -> int:
    request = _read_json(request_path.resolve())
    cycle_root = output_path.resolve().parent
    cycle_index = request.get("cycle_index")
    result: Dict[str, Any] = {
        "kind": "online_evolution_cycle_result",
        "layout_version": 1,
        "cycle_index": cycle_index,
        "started_at": time.time(),
        "status": "error",
        "online_publish_status": "not_published",
    }
    try:
        stage_root, _, records_by_uid = _stage_records(request, cycle_root)
        monitor = request.get("monitor_request") or {}
        if monitor:
            from .runtime_replay import monitor_regression
            result = monitor_regression(request, monitor, records_by_uid, cycle_root)
            _write_json(output_path.resolve(), result)
            return 0
        pending = request.get("pending_candidate") or {}
        proposal_uids = list(pending.get("proposal_record_uids") or records_by_uid)
        if pending:
            snapshot_path = Path(pending["candidate_snapshot_path"])
            report = _extend_validation_report(_read_json(Path(pending["report_path"])), records_by_uid,
                                               proposal_uids, Path(request["evolution_config"]))
            report_path = cycle_root / "evolution" / "evolution_run_validation" / "evolution_report.json"
            frozen_snapshot = _read_json(snapshot_path)
            snapshot_path = cycle_root / "evolution" / "registry" / (frozen_snapshot["snapshot_id"] + "_candidate") / "snapshot.json"
            _write_json(snapshot_path, frozen_snapshot)
            _write_json(report_path, report)
            miner = {"status": "frozen_candidate_validation", "report_path": str(report_path)}
        else:
            report, snapshot_path, miner = _run_miner(
                stage_root=stage_root, cycle_root=cycle_root, request=request,
            )
        result["miner"] = miner
        if report is None or snapshot_path is None:
            result["status"] = str(miner.get("status") or "no_candidate")
            _write_json(output_path.resolve(), result)
            return 0
        result["report_path"] = str(miner.get("report_path") or "")
        result["candidate_snapshot_path"] = str(snapshot_path)
        result["snapshot_id"] = str(_read_json(snapshot_path).get("snapshot_id") or "")
        result["record_uids"] = list(records_by_uid)
        result["proposal_record_uids"] = proposal_uids
        result["trigger_proof_path"] = pending.get("trigger_proof_path", "")
        result["local_replay_root"] = pending.get("local_replay_root") or str(cycle_root / "local_trigger")
        # This journal is durable before any expensive inference. It is not
        # an acceptance result and cannot publish a candidate.
        _write_json(cycle_root / "candidate_pending.json", result)
        if not result["trigger_proof_path"]:
            trigger_uids = _group_uids(report, records_by_uid).get("trigger") or []
            if not trigger_uids:
                raise ValueError("candidate has no diagnostic trigger")
            trigger_tid = ((records_by_uid[trigger_uids[0]]["run_result"].get("agent") or {}).get("trajectory") or {}).get("trajectory_id")
            diagnoses = [d for d in report.get("diagnoses", []) if trigger_tid in d.get("trajectory_ids", [])]
            objective = diagnoses[0].get("failure_scope", "auto") if diagnoses else "auto"
            local = replay_trigger(run_result_path=Path(records_by_uid[trigger_uids[0]]["run_result_path"]),
                snapshot_path=snapshot_path, config_path=Path(request["config_path"]),
                output_root=Path(result["local_replay_root"]), device_id=str(request.get("visible_device_id") or "0"),
                timeout_seconds=int(request.get("timeout_seconds") or 7200),
                baseline_snapshot_path=Path(request["active_snapshot_path"]) if request.get("active_snapshot_path") else None,
                repair_objective=objective)
            proof_path = cycle_root / "trigger_proof.json"
            _write_json(proof_path, local)
            result["trigger_proof_path"] = str(proof_path)
            if local.get("causal_fix_status") != "causal_fix_verified":
                result["status"] = ("local_repair_not_proven" if local.get("status") in {
                    "runtime_suffix_executed", "verifier_suffix_executed", "paired_runtime_rollout_executed"
                } else "replay_error")
                report.update({"replay_status": result["status"], "candidate_replay_status": result["status"],
                               "final_acceptance_status": result["status"], "local_repair_status": local.get("causal_fix_status")})
                _write_json(Path(miner["report_path"]), report)
                _write_json(output_path.resolve(), result)
                return 0
            result["status"] = "awaiting_future_validation"
            report.update({"replay_status": result["status"], "candidate_replay_status": result["status"],
                           "final_acceptance_status": result["status"], "local_repair_status": "causal_fix_verified"})
            _write_json(Path(miner["report_path"]), report)
            _write_json(output_path.resolve(), result)
            return 0
        gate = _gate_candidate(
            request=request,
            cycle_root=cycle_root,
            report=report,
            report_path=Path(str(miner.get("report_path") or "")).resolve(),
            snapshot_path=snapshot_path,
            records_by_uid=records_by_uid,
        )
        result["gate"] = gate
        if str(gate.get("status") or "") == "accepted":
            accepted_snapshot = _read_json(snapshot_path)
            accepted_snapshot["status"] = "accepted"
            accepted_path = cycle_root / "accepted_snapshot.json"
            _write_json(accepted_path, accepted_snapshot)
            result["accepted_snapshot_path"] = str(accepted_path)
            result["status"] = "accepted"
        else:
            result["status"] = str(gate.get("status") or "rejected")
        if result["status"] == "insufficient_coverage":
            maximum_cases = max(32, int((request.get("options") or {}).get("online_evolution_validation_max_cases", 256)))
            if len(records_by_uid) >= maximum_cases:
                result["status"] = "validation_exhausted"
            report.update({"replay_status": result["status"], "candidate_replay_status": result["status"],
                           "final_acceptance_status": result["status"]})
            _write_json(Path(miner["report_path"]), report)
        _write_json(output_path.resolve(), result)
        return 0
    except Exception as exc:
        result.update({
            "status": "cycle_error",
            "error": f"{type(exc).__name__}: {exc}",
        })
        _write_json(output_path.resolve(), result)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.request, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
