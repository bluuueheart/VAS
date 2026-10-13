"""CLI for case-level accounting and paired ablation of the four roles.

The default report is observational and does not infer causality from one
trajectory.  Passing ``--counterfactual-root`` and ``--disabled-role`` adds a
strict same-case comparison against a real rerun with that online role
disabled. The Evolver is attached through its online cycle replay report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .agent_value import (
    analyze_counterfactual_case,
    summarize_cases,
    summarize_counterfactual_pairs,
    summarize_evolver,
)


def _json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _case_paths(root: Path, datasets: List[str]) -> List[Path]:
    paths: List[Path] = []
    for dataset in datasets:
        paths.extend(sorted((root / dataset).glob("*/mix_sample_result.json")))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspaces-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["egoschema", "lvbench", "mlvu", "videomme"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evolution-report", type=Path)
    parser.add_argument(
        "--counterfactual-root",
        type=Path,
        help=(
            "Optional workspace root containing reruns with one online role disabled. "
            "Pairs are matched by sample_uid and are never inferred across different cases."
        ),
    )
    parser.add_argument(
        "--disabled-role",
        choices=["observer", "reflector"],
        help="Role disabled in --counterfactual-root.",
    )
    args = parser.parse_args()

    if bool(args.counterfactual_root) != bool(args.disabled_role):
        parser.error("--counterfactual-root and --disabled-role must be provided together")

    # Import here so the module remains a lightweight read-only audit entry
    # point when callers only need the CLI help or malformed-input checks.
    from .agent_value import analyze_case

    cases: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for sample_path in _case_paths(args.workspaces_root, list(args.datasets)):
        result_path = sample_path.parent / "results" / "run_result.json"
        if not result_path.exists():
            skipped.append({"sample_path": str(sample_path), "reason": "run_result_missing"})
            continue
        try:
            sample = _json(sample_path)
            result = _json(result_path)
            cases.append(analyze_case(sample, result))
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            skipped.append({"sample_path": str(sample_path), "reason": f"{type(exc).__name__}: {exc}"})

    evolver_summary: Dict[str, Any] = {
        "status": "not_run",
        "value_verdict": "cycle_role_requires_evolution_report",
        "evidence": ["Evolver utility is credited from an online cycle replay report."],
    }
    attached_report = ""
    if args.evolution_report:
        report = _json(args.evolution_report)
        evolver_record = report.get("evolver_agent") if isinstance(report.get("evolver_agent"), dict) else {}
        analysis = report.get("agent_value_analysis") if isinstance(report.get("agent_value_analysis"), dict) else {}
        attached = analysis.get("evolver") if isinstance(analysis.get("evolver"), dict) else {}
        if attached:
            evolver_summary = dict(attached)
        elif evolver_record:
            evolver_summary = summarize_evolver(
                evolver_record,
                diagnosis_count=len(report.get("diagnoses") or []),
                candidate_snapshot=report.get("candidate_snapshot"),
                replay_status=str(report.get("replay_status") or ""),
                trajectory_count=int(report.get("trajectory_count") or 0),
                proposal_validation=report.get("evolver_proposal_validation"),
            )
        attached_report = str(args.evolution_report.resolve())

    counterfactual_pairs: List[Dict[str, Any]] = []
    counterfactual_skipped: List[Dict[str, str]] = []
    if args.counterfactual_root:
        baseline_root = args.workspaces_root.resolve()
        ablation_root = args.counterfactual_root.resolve()
        for sample_path in _case_paths(baseline_root, list(args.datasets)):
            relative_path = sample_path.relative_to(baseline_root)
            ablation_sample_path = ablation_root / relative_path
            baseline_result_path = sample_path.parent / "results" / "run_result.json"
            ablation_result_path = ablation_sample_path.parent / "results" / "run_result.json"
            if not baseline_result_path.exists() or not ablation_sample_path.exists() or not ablation_result_path.exists():
                counterfactual_skipped.append({
                    "sample_path": str(sample_path),
                    "reason": "paired_artifact_missing",
                })
                continue
            try:
                baseline_sample = _json(sample_path)
                baseline_result = _json(baseline_result_path)
                ablation_sample = _json(ablation_sample_path)
                ablation_result = _json(ablation_result_path)
                counterfactual_pairs.append(
                    analyze_counterfactual_case(
                        baseline_sample,
                        baseline_result,
                        ablation_sample,
                        ablation_result,
                        disabled_role=args.disabled_role,
                    )
                )
            except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
                counterfactual_skipped.append({
                    "sample_path": str(sample_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                })

    output = {
        "kind": "four_role_agent_value_audit",
        "method": "trajectory evidence accounting; not causal ablation",
        "workspaces_root": str(args.workspaces_root.resolve()),
        "datasets": list(args.datasets),
        "cases": cases,
        "aggregate": summarize_cases(cases),
        "evolver": evolver_summary,
        "attached_evolution_report": attached_report,
        "skipped": skipped,
    }
    if args.counterfactual_root:
        output["counterfactual"] = {
            "baseline_root": str(args.workspaces_root.resolve()),
            "ablation_root": str(args.counterfactual_root.resolve()),
            "disabled_role": args.disabled_role,
            "pairs": counterfactual_pairs,
            "aggregate": summarize_counterfactual_pairs(counterfactual_pairs),
            "skipped": counterfactual_skipped,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "case_count": len(cases), "skipped": len(skipped)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
