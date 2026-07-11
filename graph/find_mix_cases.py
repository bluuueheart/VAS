from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


PARTICIPATED_STATUSES = {"completed", "wrong"}
SAMPLE_RESULT_FILENAME = "mix_sample_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Mix cases by planner logical-conflict trajectory and final answer correctness."
        )
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("mix_plot_config.json")),
        help="Path to the plotting config JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "conflict_recovered_correct", "multiturn_wrong"),
        default="all",
        help="Which case set to print. Default: all.",
    )
    parser.add_argument(
        "--print-header",
        action="store_true",
        help="Print a section header before each result block.",
    )
    return parser.parse_args()


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _strip_json_line_comments(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines)


def _load_json_config(path: Path) -> Dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8-sig")
    normalized = _strip_json_line_comments(_unwrap_markdown_code_fence(raw_text)).strip()
    if not normalized:
        raise ValueError(f"JSON payload is empty: {path}")
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _unwrap_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) < 3:
        return text
    if not lines[0].startswith("```") or lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1])


def _parse_json_object_maybe(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = _unwrap_markdown_code_fence(value).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_logical_conflict(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _extract_evidence_conflicts_from_payload(payload: Any) -> List[Optional[bool]]:
    if not isinstance(payload, dict):
        return []
    raw_items = payload.get("Evidence")
    if not isinstance(raw_items, list):
        raw_items = payload.get("evidence")
    if not isinstance(raw_items, list):
        return []

    flags: List[Optional[bool]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        flags.append(_coerce_logical_conflict(item.get("logical conflict", item.get("logical_conflict"))))
    return flags


def _resolve_run_result_path(result_item: Mapping[str, Any]) -> Optional[Path]:
    explicit_path = str(result_item.get("run_result_path") or "").strip()
    if explicit_path:
        return Path(explicit_path)
    workspace_root = str(result_item.get("workspace_root") or "").strip()
    if workspace_root:
        return Path(workspace_root) / "results" / "run_result.json"
    return None


def _iter_workspace_result_items(workspace_root: Path) -> Iterable[Dict[str, Any]]:
    if not workspace_root.exists():
        return
    for result_path in workspace_root.rglob(SAMPLE_RESULT_FILENAME):
        payload = _load_json_object(result_path)
        if payload is None:
            continue
        if _normalize_status(payload.get("status")) not in PARTICIPATED_STATUSES:
            continue
        yield payload


def _iter_progress_result_items(progress_path: Path) -> Iterable[Dict[str, Any]]:
    if not progress_path.exists():
        return
    progress_payload = _load_json_object(progress_path)
    if progress_payload is None:
        return
    raw_records = progress_payload.get("records")
    if not isinstance(raw_records, dict):
        return
    for item in raw_records.values():
        if not isinstance(item, dict):
            continue
        if _normalize_status(item.get("status")) not in PARTICIPATED_STATUSES:
            continue
        result_path_text = str(item.get("result_path") or "").strip()
        if result_path_text:
            result_payload = _load_json_object(Path(result_path_text))
            if result_payload is not None and _normalize_status(result_payload.get("status")) in PARTICIPATED_STATUSES:
                yield result_payload
                continue
        yield dict(item)


def _load_result_items(config_path: Path) -> List[Dict[str, Any]]:
    config_path = config_path.resolve()
    config_payload = _load_json_config(config_path)
    config_dir = config_path.parent

    mix_config_path = _resolve_path(
        config_dir,
        str(config_payload.get("mix_config_path") or "../evaluationVideo/Mix/mix_config.json"),
    )
    mix_payload = _load_json_config(mix_config_path)
    mix_dir = mix_config_path.parent

    workspace_override = str(config_payload.get("workspace_root") or "").strip()
    progress_override = str(config_payload.get("progress_path") or "").strip()
    workspace_root = _resolve_path(
        config_dir if workspace_override else mix_dir,
        workspace_override or str(mix_payload.get("workspaces_root") or "./workspaces"),
    )
    progress_path = _resolve_path(
        config_dir if progress_override else mix_dir,
        progress_override or str(mix_payload.get("progress_path") or "./workspaces/mix_progress.json"),
    )

    merged: MutableMapping[str, Dict[str, Any]] = {}
    for item in _iter_workspace_result_items(workspace_root):
        sample_uid = str(item.get("sample_uid") or "").strip()
        if sample_uid:
            merged[sample_uid] = item
    for item in _iter_progress_result_items(progress_path):
        sample_uid = str(item.get("sample_uid") or "").strip()
        if not sample_uid:
            continue
        current = merged.get(sample_uid)
        if current is None or str(item.get("updated_at") or "") > str(current.get("updated_at") or ""):
            merged[sample_uid] = item
    if not merged:
        raise FileNotFoundError(
            "No participated Mix sample results were found. "
            f"Checked workspace_root={workspace_root} and progress_path={progress_path}."
        )
    return sorted(merged.values(), key=lambda item: str(item.get("updated_at") or ""))


def _iter_root_llm_evidence_conflicts(trajectory: Any) -> Iterable[List[Optional[bool]]]:
    if not isinstance(trajectory, dict):
        return
    for event in trajectory.get("debug_events", []) or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("type", "")).strip() != "llm_iteration":
            continue
        if _safe_int(event.get("depth"), 0) != 0:
            continue
        payload = _parse_json_object_maybe(event.get("response"))
        flags = _extract_evidence_conflicts_from_payload(payload)
        if flags:
            yield flags


def _is_multiturn(result_item: Mapping[str, Any], run_result_payload: Mapping[str, Any]) -> bool:
    iter_num = _safe_int(result_item.get("iter_num"))
    if iter_num > 1:
        return True
    metrics = run_result_payload.get("metrics")
    if isinstance(metrics, dict) and _safe_int(metrics.get("planner_llm_iteration_count")) > 1:
        return True
    trajectory = run_result_payload.get("agent", {}).get("trajectory") if isinstance(run_result_payload.get("agent"), dict) else None
    root_iters = sum(1 for _ in _iter_root_llm_evidence_conflicts(trajectory))
    return root_iters > 1


def _match_conflict_recovered_correct(
    result_item: Mapping[str, Any],
    run_result_payload: Mapping[str, Any],
) -> bool:
    if _normalize_status(result_item.get("status")) != "completed":
        return False
    if not _is_multiturn(result_item, run_result_payload):
        return False

    agent_payload = run_result_payload.get("agent")
    trajectory = agent_payload.get("trajectory") if isinstance(agent_payload, dict) else None
    evidence_steps = list(_iter_root_llm_evidence_conflicts(trajectory))
    if not evidence_steps:
        return False

    saw_conflict_true = any(flag is True for flags in evidence_steps for flag in flags)
    final_flags = evidence_steps[-1]
    final_has_false = any(flag is False for flag in final_flags)
    final_has_true = any(flag is True for flag in final_flags)
    return saw_conflict_true and final_has_false and not final_has_true


def _match_multiturn_wrong(
    result_item: Mapping[str, Any],
    run_result_payload: Mapping[str, Any],
) -> bool:
    if _normalize_status(result_item.get("status")) != "wrong":
        return False
    return _is_multiturn(result_item, run_result_payload)


def _filter_workspace_roots(
    result_items: Sequence[Mapping[str, Any]],
    *,
    predicate,
) -> List[str]:
    matched: List[str] = []
    for item in result_items:
        if _normalize_status(item.get("status")) not in PARTICIPATED_STATUSES:
            continue
        run_result_path = _resolve_run_result_path(item)
        if run_result_path is None or not run_result_path.exists():
            continue
        run_result_payload = _load_json_object(run_result_path)
        if run_result_payload is None:
            continue
        if not predicate(item, run_result_payload):
            continue
        workspace_root = str(item.get("workspace_root") or "").strip()
        if workspace_root:
            matched.append(workspace_root)
    return sorted(set(matched))


def _print_lines(lines: Sequence[str], header: str = "") -> None:
    if header:
        print(header)
    for line in lines:
        print(line)


def main() -> None:
    args = parse_args()
    result_items = _load_result_items(Path(args.config))

    if args.mode in {"all", "conflict_recovered_correct"}:
        recovered = _filter_workspace_roots(
            result_items,
            predicate=_match_conflict_recovered_correct,
        )
        _print_lines(
            recovered,
            "=== conflict_recovered_correct ===" if args.print_header else "",
        )
        if args.mode == "conflict_recovered_correct":
            return

    if args.mode == "all" and args.print_header:
        print()

    wrong = _filter_workspace_roots(
        result_items,
        predicate=_match_multiturn_wrong,
    )
    _print_lines(
        wrong,
        "=== multiturn_wrong ===" if args.print_header else "",
    )


if __name__ == "__main__":
    main()
