"""Full-trajectory Reflector diagnosis before executable mutation proposals."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

FORBIDDEN = {"ground_truth", "groundtruth", "ground_truth_option", "ground_truth_labels",
             "gold_answer", "correct_label", "official_meta", "time_reference", "evaluator_output",
             "acc", "accuracy", "answer_score"}


def execution_record(value):
    if isinstance(value, Mapping):
        return {k: execution_record(v) for k, v in value.items() if str(k).lower() not in FORBIDDEN}
    if isinstance(value, (list, tuple)):
        return [execution_record(v) for v in value]
    return value


def review_full_trajectory(suite, run_result, *, max_fragment_chars=24000, output_path=None):
    """Read every step/event, in order; never silently tail-truncate a trace.

    Large records are streamed as lossless JSON text fragments. Analysis from
    preceding fragments is carried forward, and every input digest is audited.
    Model plans/answers remain explicitly untrusted hypotheses, not ground truth.
    """
    trajectory = execution_record((run_result.get("agent") or {}).get("trajectory") or {})
    steps = trajectory.get("steps") or []
    if not steps:
        raise ValueError("Reflector needs a nonempty complete execution trajectory")
    text = json.dumps(trajectory, ensure_ascii=False, sort_keys=True, default=str)
    fragments = [text[i:i + max_fragment_chars] for i in range(0, len(text), max_fragment_chars)]
    valid_ids = {s.get("step_id", i) for i, s in enumerate(steps, 1)}
    findings, calls = [], []
    def persist(status, processed):
        if output_path is not None:
            output_path.write_text(json.dumps({"status": status, "trajectory_id": trajectory.get("trajectory_id"),
                "full_trajectory_processed": status == "complete", "processed_fragments": processed,
                "fragment_count": len(fragments), "issues": findings, "calls": calls}, ensure_ascii=False, indent=2))
    for index, fragment in enumerate(fragments):
        result = suite._call("reflector", {
            "phase": "trajectory_analysis",
            "goal": trajectory.get("goal", run_result.get("query", "")),
            "trajectory_id": trajectory.get("trajectory_id"),
            "step_ids": sorted(valid_ids),
            "fragment_index": index, "fragment_count": len(fragments),
            "trajectory_fragment": fragment,
            "prior_findings": findings,
            "input_contract": "Lossless ordered fragments of the entire trajectory JSON, including all steps and debug events. No benchmark answers. A fragment may start/end inside JSON; do not infer omitted context. Plans and agent statements are hypotheses, tool observations are execution evidence.",
        }, max_tokens=2048, schema_mode="trajectory_analysis")
        calls.append(result.get("event") or {})
        persist("running", index + 1)
        if not result.get("schema_valid") or result.get("status") != "success":
            persist("invalid_response", index + 1)
            raise ValueError(f"Reflector full-trajectory analysis failed at fragment {index}")
        for issue in result["payload"]["issues"]:
            if not set(issue["step_ids"]).issubset(valid_ids):
                raise ValueError("Reflector cited a nonexistent trajectory step")
            if issue not in findings:
                findings.append(issue)
    persist("complete", len(fragments))
    return {"status": "complete", "trajectory_id": trajectory.get("trajectory_id"),
            "step_count": len(steps), "event_count": len(trajectory.get("debug_events") or []),
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "input_char_count": len(text), "processed_char_count": sum(map(len, fragments)),
            "fragment_count": len(fragments), "full_trajectory_processed": True,
            "issues": findings, "calls": calls}
