from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_mix_metrics import (
    AXIS_BACKGROUND,
    DATASET_DISPLAY_NAMES,
    DATASET_ORDER,
    PAPER_BACKGROUND,
    PLOT_PALETTES,
    _make_bucket_series,
    _set_paper_style,
    load_plot_config,
    load_records,
    plot_combined_bucket_accuracy_figure,
    tool_use_bucket,
    iter_bucket,
    success_rate_bucket,
)
from plot_mix_runtime_tool_usage import load_result_items
from plot_tool_inventory import (
    build_enabled_records,
    infer_function_category,
)


SUMMARY_DATASET_ORDER = tuple(DATASET_ORDER) + ("overall",)
RESULT_STATUSES = {"completed", "wrong"}


@dataclass(frozen=True)
class EvidenceSample:
    sample_uid: str
    dataset_name: str
    status: str
    acc: float
    evidence_verification_score: float
    evidence_logical_rigor_score: float
    planner_llm_iteration_count: int
    planner_evidence_iteration_count: int
    planner_evidence_item_count: int
    planner_verified_evidence_item_count: int
    planner_non_conflicting_evidence_item_count: int
    format_violation_count: int
    invalid_json_count: int
    evidence_eligible: bool
    format_compliant: bool
    workspace_root: str
    run_result_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Mix evidence metrics tables, plots, and active-tool inventory from cached results."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("mix_plot_config.json")),
        help="Path to the plotting config JSON.",
    )
    return parser.parse_args()


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _normalize_evidence_status(value: Any = "") -> str:
    status = str(value or "").strip().lower()
    if status in {"verified", "exact"}:
        return "verified"
    if status in {"approximate", "partial"}:
        return "approximate"
    return "missing"


def _coerce_logical_conflict(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _extract_evidence_items_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_items = payload.get("Evidence")
    if not isinstance(raw_items, list):
        raw_items = payload.get("evidence")
    if not isinstance(raw_items, list):
        return []

    evidence_items: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        evidence_items.append(
            {
                "evidence_status": _normalize_evidence_status(item.get("evidence_status", "")),
                "logical_conflict": _coerce_logical_conflict(
                    item.get("logical conflict", item.get("logical_conflict"))
                ),
            }
        )
    return evidence_items


def _count_debug_event_types(trajectory: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(trajectory, dict):
        return counts
    for event in trajectory.get("debug_events", []) or []:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", "")).strip()
        if not event_type:
            continue
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _summarize_planner_evidence_quality(trajectory: Any) -> Dict[str, Any]:
    summary = {
        "planner_llm_iteration_count": 0,
        "planner_evidence_iteration_count": 0,
        "planner_evidence_item_count": 0,
        "planner_verified_evidence_item_count": 0,
        "planner_non_conflicting_evidence_item_count": 0,
        "evidence_verification_score": 0.0,
        "evidence_logical_rigor_score": 0.0,
    }
    if not isinstance(trajectory, dict):
        return summary

    total_verification_ratio = 0.0
    total_logical_rigor_ratio = 0.0

    for event in trajectory.get("debug_events", []) or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("type", "")).strip() != "llm_iteration":
            continue
        if _safe_int(event.get("depth"), 0) != 0:
            continue

        summary["planner_llm_iteration_count"] += 1
        payload = _parse_json_object_maybe(event.get("response"))
        evidence_items = _extract_evidence_items_from_payload(payload)
        if not evidence_items:
            continue

        summary["planner_evidence_iteration_count"] += 1
        item_count = len(evidence_items)
        verified_count = 0
        non_conflicting_count = 0
        for item in evidence_items:
            if str(item.get("evidence_status", "")).strip() == "verified":
                verified_count += 1
            if item.get("logical_conflict") is False:
                non_conflicting_count += 1

        summary["planner_evidence_item_count"] += item_count
        summary["planner_verified_evidence_item_count"] += verified_count
        summary["planner_non_conflicting_evidence_item_count"] += non_conflicting_count
        total_verification_ratio += float(verified_count) / float(item_count)
        total_logical_rigor_ratio += float(non_conflicting_count) / float(item_count)

    denominator = max(1, int(summary["planner_llm_iteration_count"]))
    summary["evidence_verification_score"] = round(total_verification_ratio / float(denominator), 6)
    summary["evidence_logical_rigor_score"] = round(total_logical_rigor_ratio / float(denominator), 6)
    return summary


def _build_evidence_sample(
    result_item: Mapping[str, Any],
    run_result_payload: Optional[Mapping[str, Any]],
) -> EvidenceSample:
    metrics = run_result_payload.get("metrics", {}) if isinstance(run_result_payload, Mapping) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    agent_payload = run_result_payload.get("agent", {}) if isinstance(run_result_payload, Mapping) else {}
    if not isinstance(agent_payload, dict):
        agent_payload = {}
    trajectory = agent_payload.get("trajectory")

    evidence_summary = _summarize_planner_evidence_quality(trajectory)
    debug_counts = _count_debug_event_types(trajectory)

    planner_llm_iteration_count = _safe_int(
        metrics.get("planner_llm_iteration_count"),
        evidence_summary["planner_llm_iteration_count"],
    )
    planner_evidence_iteration_count = _safe_int(
        metrics.get("planner_evidence_iteration_count"),
        evidence_summary["planner_evidence_iteration_count"],
    )
    planner_evidence_item_count = _safe_int(
        metrics.get("planner_evidence_item_count"),
        evidence_summary["planner_evidence_item_count"],
    )
    planner_verified_evidence_item_count = _safe_int(
        metrics.get("planner_verified_evidence_item_count"),
        evidence_summary["planner_verified_evidence_item_count"],
    )
    planner_non_conflicting_evidence_item_count = _safe_int(
        metrics.get("planner_non_conflicting_evidence_item_count"),
        evidence_summary["planner_non_conflicting_evidence_item_count"],
    )
    evidence_verification_score = _safe_float(
        metrics.get("evidence_verification_score"),
        evidence_summary["evidence_verification_score"],
    )
    evidence_logical_rigor_score = _safe_float(
        metrics.get("evidence_logical_rigor_score"),
        evidence_summary["evidence_logical_rigor_score"],
    )
    format_violation_count = _safe_int(metrics.get("format_violation_count"), debug_counts.get("format_violation", 0))
    invalid_json_count = _safe_int(metrics.get("invalid_json_count"), 0)

    return EvidenceSample(
        sample_uid=str(result_item.get("sample_uid") or "").strip(),
        dataset_name=str(result_item.get("dataset_name") or "").strip().lower(),
        status=str(result_item.get("status") or "").strip().lower(),
        acc=1.0 if _safe_float(result_item.get("acc"), 0.0) >= 1.0 else 0.0,
        evidence_verification_score=evidence_verification_score,
        evidence_logical_rigor_score=evidence_logical_rigor_score,
        planner_llm_iteration_count=planner_llm_iteration_count,
        planner_evidence_iteration_count=planner_evidence_iteration_count,
        planner_evidence_item_count=planner_evidence_item_count,
        planner_verified_evidence_item_count=planner_verified_evidence_item_count,
        planner_non_conflicting_evidence_item_count=planner_non_conflicting_evidence_item_count,
        format_violation_count=format_violation_count,
        invalid_json_count=invalid_json_count,
        evidence_eligible=planner_evidence_iteration_count > 0,
        format_compliant=format_violation_count == 0,
        workspace_root=str(result_item.get("workspace_root") or ""),
        run_result_path=str(result_item.get("run_result_path") or ""),
    )


def load_evidence_samples(config: Any) -> List[EvidenceSample]:
    result_items = load_result_items(config)
    samples: List[EvidenceSample] = []
    for item in result_items:
        if str(item.get("status") or "").strip().lower() not in RESULT_STATUSES:
            continue
        run_result_payload = None
        run_result_path_text = str(item.get("run_result_path") or "").strip()
        if run_result_path_text:
            run_result_payload = _load_json_object(Path(run_result_path_text))
        samples.append(_build_evidence_sample(item, run_result_payload))
    if not samples:
        raise FileNotFoundError(
            "No participated Mix sample results were found for evidence reporting. "
            f"Checked workspace_root={config.workspace_root} and progress_path={config.progress_path}."
        )
    return samples


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _dataset_or_overall_samples(samples: Sequence[EvidenceSample], dataset_name: str) -> List[EvidenceSample]:
    if dataset_name == "overall":
        return list(samples)
    return [item for item in samples if item.dataset_name == dataset_name]


def build_summary_rows(samples: Sequence[EvidenceSample]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset_name in SUMMARY_DATASET_ORDER:
        scoped = _dataset_or_overall_samples(samples, dataset_name)
        eligible = [item for item in scoped if item.evidence_eligible]
        verification_values = [item.evidence_verification_score for item in eligible]
        rigor_values = [item.evidence_logical_rigor_score for item in eligible]
        row = {
            "dataset_name": dataset_name,
            "dataset_display_name": "Overall" if dataset_name == "overall" else DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name),
            "sample_count": len(scoped),
            "correct_count": int(sum(item.acc for item in scoped)),
            "accuracy": round(_mean([item.acc for item in scoped]) or 0.0, 6),
            "evidence_eligible_sample_count": len(eligible),
            "evidence_coverage_rate": round(len(eligible) / float(len(scoped)), 6) if scoped else 0.0,
            "evidence_verification_score": round(_mean(verification_values) or 0.0, 6),
            "evidence_logical_rigor_score": round(_mean(rigor_values) or 0.0, 6),
            "format_compliance_rate": round(_mean([1.0 if item.format_compliant else 0.0 for item in scoped]) or 0.0, 6),
            "format_violation_rate": round(_mean([1.0 if item.format_violation_count > 0 else 0.0 for item in scoped]) or 0.0, 6),
            "avg_format_violation_count": round(_mean([float(item.format_violation_count) for item in scoped]) or 0.0, 6),
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_stem.with_suffix(".png")), bbox_inches="tight")
    fig.savefig(str(output_stem.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def plot_evidence_metric_triptych(summary_rows: Sequence[Mapping[str, Any]], output_stem: Path) -> None:
    metrics = (
        ("evidence_verification_score", "Evidence Verification Score", 0),
        ("evidence_logical_rigor_score", "Evidence Logical Rigor Score", 1),
        ("format_compliance_rate", "Format Compliance Rate", 2),
    )
    labels = [row["dataset_display_name"] for row in summary_rows]
    x_values = list(range(len(labels)))

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.4), sharey=True)
    fig.patch.set_facecolor(PAPER_BACKGROUND)

    for index, (metric_name, title, palette_index) in enumerate(metrics):
        ax = axes[index]
        palette = PLOT_PALETTES[palette_index % len(PLOT_PALETTES)]
        values = [float(row.get(metric_name, 0.0) or 0.0) for row in summary_rows]
        bars = ax.bar(
            x_values,
            values,
            color=palette["dark"],
            edgecolor=palette["dark"],
            alpha=0.82,
            width=0.62,
        )
        ax.set_facecolor(AXIS_BACKGROUND)
        ax.grid(axis="y")
        ax.set_title(title, pad=14, fontsize=20.2)
        ax.set_xticks(x_values, labels)
        ax.tick_params(axis="x", labelsize=14.0, rotation=0)
        ax.tick_params(axis="y", labelsize=15.2)
        ax.set_ylim(0.0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        if index == 0:
            ax.set_ylabel("Score", fontsize=17.0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for bar, row, value in zip(bars, summary_rows, values):
            count_key = "evidence_eligible_sample_count" if metric_name != "format_compliance_rate" else "sample_count"
            count = int(row.get(count_key, 0) or 0)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(1.0, value + 0.03),
                f"{value * 100:.1f}%\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=11.2,
                color="#25303B",
                linespacing=1.18,
            )

    fig.subplots_adjust(wspace=0.22, bottom=0.2, top=0.84)
    _save_figure(fig, output_stem)


def _split_correct_wrong(samples: Sequence[EvidenceSample], metric_name: str) -> Tuple[List[float], List[float]]:
    eligible = [item for item in samples if item.evidence_eligible]
    correct_values = [float(getattr(item, metric_name)) for item in eligible if item.acc >= 1.0]
    wrong_values = [float(getattr(item, metric_name)) for item in eligible if item.acc < 1.0]
    return correct_values, wrong_values


def _draw_distribution_panel(
    ax: plt.Axes,
    *,
    title: str,
    values_left: Sequence[float],
    values_right: Sequence[float],
    kind: str,
    palette_index: int,
) -> None:
    palette = PLOT_PALETTES[palette_index % len(PLOT_PALETTES)]
    ax.set_facecolor(AXIS_BACKGROUND)
    ax.grid(axis="y")
    ax.set_title(title, pad=12, fontsize=18.0)
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=14.2)
    ax.tick_params(axis="y", labelsize=14.6)
    positions = [0, 1]
    values = [list(values_left), list(values_right)]
    render_values = [bucket if bucket else [math.nan] for bucket in values]

    if kind == "violin":
        violin = ax.violinplot(
            render_values,
            positions=positions,
            widths=0.78,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(palette["light"])
            body.set_edgecolor(palette["dark"])
            body.set_alpha(0.72)
        ax.boxplot(
            render_values,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#FFFFFF", "edgecolor": palette["dark"], "linewidth": 1.2},
            medianprops={"color": palette["dark"], "linewidth": 1.5},
            whiskerprops={"color": palette["dark"], "linewidth": 1.0},
            capprops={"color": palette["dark"], "linewidth": 1.0},
        )
    else:
        ax.boxplot(
            render_values,
            positions=positions,
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": palette["light"], "edgecolor": palette["dark"], "linewidth": 1.2},
            medianprops={"color": palette["dark"], "linewidth": 1.6},
            whiskerprops={"color": palette["dark"], "linewidth": 1.0},
            capprops={"color": palette["dark"], "linewidth": 1.0},
        )

    ax.set_xticks(positions, ["Correct", "Wrong"])
    for xpos, bucket in zip(positions, values):
        ax.text(
            xpos,
            1.01,
            f"n={len(bucket)}",
            ha="center",
            va="bottom",
            fontsize=11.6,
            color="#64748B",
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_evidence_distribution_quartet(samples: Sequence[EvidenceSample], output_stem: Path) -> None:
    verification_correct, verification_wrong = _split_correct_wrong(samples, "evidence_verification_score")
    rigor_correct, rigor_wrong = _split_correct_wrong(samples, "evidence_logical_rigor_score")

    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.3), sharey=True)
    fig.patch.set_facecolor(PAPER_BACKGROUND)

    _draw_distribution_panel(
        axes[0],
        title="Verification Score\nViolin",
        values_left=verification_correct,
        values_right=verification_wrong,
        kind="violin",
        palette_index=0,
    )
    _draw_distribution_panel(
        axes[1],
        title="Logical Rigor Score\nViolin",
        values_left=rigor_correct,
        values_right=rigor_wrong,
        kind="violin",
        palette_index=1,
    )
    _draw_distribution_panel(
        axes[2],
        title="Verification Score\nBox",
        values_left=verification_correct,
        values_right=verification_wrong,
        kind="box",
        palette_index=0,
    )
    _draw_distribution_panel(
        axes[3],
        title="Logical Rigor Score\nBox",
        values_left=rigor_correct,
        values_right=rigor_wrong,
        kind="box",
        palette_index=1,
    )
    axes[0].set_ylabel("Score", fontsize=17.0)

    fig.subplots_adjust(wspace=0.18, bottom=0.17, top=0.83)
    _save_figure(fig, output_stem)


def literal_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def literal_str_sequence(node: ast.AST) -> Optional[List[str]]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    values: List[str] = []
    for item in node.elts:
        value = literal_str(item)
        if value is None:
            return None
        values.append(value)
    return values


def class_inherits_base_tool(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseTool":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseTool":
            return True
    return False


def _parse_tool_metadata(project_root: Path) -> Dict[str, Dict[str, Any]]:
    impl_dir = project_root / "toolkit" / "impl"
    metadata: Dict[str, Dict[str, Any]] = {}
    for module_path in sorted(impl_dir.glob("*.py")):
        if module_path.name.startswith("_"):
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        module_name = f"toolkit.impl.{module_path.stem}"
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not class_inherits_base_tool(node):
                continue
            tool_name = ""
            description = ""
            tags: List[str] = []
            required_params: List[str] = []
            for statement in node.body:
                targets: List[str] = []
                value_node: Optional[ast.AST] = None
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    targets = [statement.target.id]
                    value_node = statement.value
                elif isinstance(statement, ast.Assign):
                    targets = [target.id for target in statement.targets if isinstance(target, ast.Name)]
                    value_node = statement.value
                if value_node is None:
                    continue
                for target_name in targets:
                    if target_name == "name":
                        tool_name = literal_str(value_node) or tool_name
                    elif target_name == "description":
                        description = literal_str(value_node) or description
                    elif target_name == "tags":
                        tags = literal_str_sequence(value_node) or tags
                    elif target_name == "parameters":
                        try:
                            params = ast.literal_eval(value_node)
                        except Exception:
                            params = None
                        if isinstance(params, dict):
                            raw_required = params.get("required", [])
                            if isinstance(raw_required, list):
                                required_params = [str(item).strip() for item in raw_required if str(item).strip()]
            if tool_name:
                metadata[tool_name] = {
                    "description": description,
                    "tags": tags,
                    "required_parameters": required_params,
                    "module": module_name,
                }
    return metadata


def export_active_tool_inventory(output_dir: Path) -> Path:
    project_root = (Path(__file__).resolve().parents[1] / "RL").resolve()
    enabled_records = build_enabled_records(project_root)
    metadata = _parse_tool_metadata(project_root)

    rows: List[Dict[str, Any]] = []
    for tool_name, record in sorted(enabled_records.items()):
        info = metadata.get(tool_name, {})
        rows.append(
            {
                "name": tool_name,
                "description": str(info.get("description") or ""),
                "tool_type": record.tool_type,
                "function_category": infer_function_category(record),
                "module": str(info.get("module") or record.module or ""),
                "tags": "|".join(str(tag) for tag in (info.get("tags") or record.tags or ())),
                "required_parameters": "|".join(str(item) for item in (info.get("required_parameters") or [])),
            }
        )

    output_path = output_dir / "mix_eval_active_tools.csv"
    _write_csv(
        output_path,
        rows,
        fieldnames=("name", "description", "tool_type", "function_category", "module", "tags", "required_parameters"),
    )
    return output_path


def write_summary_outputs(output_dir: Path, prefix: str, samples: Sequence[EvidenceSample]) -> Tuple[Path, Path, Path]:
    summary_rows = build_summary_rows(samples)
    summary_csv_path = output_dir / f"{prefix}_evidence_summary.csv"
    summary_json_path = output_dir / f"{prefix}_evidence_summary.json"
    sample_csv_path = output_dir / f"{prefix}_evidence_sample_metrics.csv"

    _write_csv(
        summary_csv_path,
        summary_rows,
        fieldnames=(
            "dataset_name",
            "dataset_display_name",
            "sample_count",
            "correct_count",
            "accuracy",
            "evidence_eligible_sample_count",
            "evidence_coverage_rate",
            "evidence_verification_score",
            "evidence_logical_rigor_score",
            "format_compliance_rate",
            "format_violation_rate",
            "avg_format_violation_count",
        ),
    )
    _write_json(summary_json_path, {"rows": summary_rows})

    sample_rows = [
        {
            "sample_uid": item.sample_uid,
            "dataset_name": item.dataset_name,
            "status": item.status,
            "acc": item.acc,
            "evidence_eligible": item.evidence_eligible,
            "evidence_verification_score": round(item.evidence_verification_score, 6),
            "evidence_logical_rigor_score": round(item.evidence_logical_rigor_score, 6),
            "planner_llm_iteration_count": item.planner_llm_iteration_count,
            "planner_evidence_iteration_count": item.planner_evidence_iteration_count,
            "planner_evidence_item_count": item.planner_evidence_item_count,
            "planner_verified_evidence_item_count": item.planner_verified_evidence_item_count,
            "planner_non_conflicting_evidence_item_count": item.planner_non_conflicting_evidence_item_count,
            "format_violation_count": item.format_violation_count,
            "invalid_json_count": item.invalid_json_count,
            "format_compliant": item.format_compliant,
            "workspace_root": item.workspace_root,
            "run_result_path": item.run_result_path,
        }
        for item in samples
    ]
    _write_csv(
        sample_csv_path,
        sample_rows,
        fieldnames=(
            "sample_uid",
            "dataset_name",
            "status",
            "acc",
            "evidence_eligible",
            "evidence_verification_score",
            "evidence_logical_rigor_score",
            "planner_llm_iteration_count",
            "planner_evidence_iteration_count",
            "planner_evidence_item_count",
            "planner_verified_evidence_item_count",
            "planner_non_conflicting_evidence_item_count",
            "format_violation_count",
            "invalid_json_count",
            "format_compliant",
            "workspace_root",
            "run_result_path",
        ),
    )
    return summary_csv_path, summary_json_path, sample_csv_path


def main() -> None:
    args = parse_args()
    config = load_plot_config(Path(args.config))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _set_paper_style()

    mix_records = load_records(config)
    evidence_samples = load_evidence_samples(config)
    prefix = config.figure_prefix

    tool_use_series = _make_bucket_series(mix_records, ("0", "1", "2", "3", "4", "5", "6+"), tool_use_bucket)
    iter_series = _make_bucket_series(mix_records, ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"), iter_bucket)
    success_rate_series = _make_bucket_series(
        mix_records,
        ("[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]"),
        success_rate_bucket,
    )
    plot_combined_bucket_accuracy_figure(
        series_items=(
            ("Accuracy vs. Tool Use Number", "Tool Use Number", tool_use_series, 0),
            ("Accuracy vs. Planner Round", "Planner Round", iter_series, 1),
            ("Accuracy vs. Tool Success Rate", "Tool Success Rate", success_rate_series, 2),
        ),
        output_stem=config.output_dir / f"{prefix}_accuracy_triptych",
    )

    summary_rows = build_summary_rows(evidence_samples)
    plot_evidence_metric_triptych(
        summary_rows,
        output_stem=config.output_dir / f"{prefix}_evidence_metric_triptych",
    )
    plot_evidence_distribution_quartet(
        evidence_samples,
        output_stem=config.output_dir / f"{prefix}_evidence_distribution_quartet",
    )

    summary_csv_path, summary_json_path, sample_csv_path = write_summary_outputs(
        config.output_dir,
        prefix,
        evidence_samples,
    )
    tool_csv_path = export_active_tool_inventory(config.output_dir)

    print(f"Loaded {len(mix_records)} participated Mix samples for accuracy plotting.")
    print(f"Loaded {len(evidence_samples)} participated Mix samples for evidence reporting.")
    print(f"Output dir: {config.output_dir}")
    print(f"Evidence summary CSV: {summary_csv_path}")
    print(f"Evidence summary JSON: {summary_json_path}")
    print(f"Sample metrics CSV: {sample_csv_path}")
    print(f"Active tools CSV: {tool_csv_path}")


if __name__ == "__main__":
    main()
