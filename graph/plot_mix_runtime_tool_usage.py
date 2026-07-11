from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_mix_metrics import PlotConfig, load_plot_config
from plot_tool_inventory import (
    AXIS_BACKGROUND,
    FUNCTION_COLORS,
    FUNCTION_ORDER,
    GRAIN_FUNCTION_ORDERS,
    GRAIN_COLORS,
    GRAIN_ORDER,
    PAPER_BACKGROUND,
    TEXT_COLOR,
    ToolRecord,
    discover_tool_records,
    infer_function_category,
    set_plot_style,
    tool_star_root,
)


PARTICIPATED_STATUSES = {"completed", "wrong"}
SAMPLE_RESULT_FILENAME = "mix_sample_result.json"
OTHER_FUNCTION_LABEL = "Other Functions"
FUNCTION_DISPLAY_ORDER = tuple(
    item for item in FUNCTION_ORDER if item != "Other"
) + (OTHER_FUNCTION_LABEL,)
FUNCTION_DISPLAY_COLORS = {
    **FUNCTION_COLORS,
    OTHER_FUNCTION_LABEL: FUNCTION_COLORS["Other"],
}
SEGMENT_TEXT_COLOR = "#FFFFFF"
UNKNOWN_GRAIN_COLOR = "#D8D8D8"
ANNOTATION_TEXT_COLOR = TEXT_COLOR
ANNOTATION_BBOX = {
    "boxstyle": "round,pad=0.18",
    "facecolor": "none",
    "edgecolor": "none",
}
OUTER_LABEL_LINE_COLOR = "#C7CED8"
OUTER_LABEL_TEXT_COLOR = "#2E3A48"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime tool usage frequency from Mix workspaces."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("mix_plot_config.json")),
        help="Path to the plotting config JSON.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help=(
            "Redraw directly from an existing runtime tool usage summary JSON "
            "instead of rescanning Mix workspaces."
        ),
    )
    parser.add_argument(
        "--no-update-summary-json",
        action="store_true",
        help=(
            "When used with --summary-json, redraw from the existing summary "
            "without rewriting the summary JSON file."
        ),
    )
    return parser.parse_args()


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


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


def load_result_items(config: PlotConfig) -> List[Dict[str, Any]]:
    merged: MutableMapping[str, Dict[str, Any]] = {}
    for item in _iter_workspace_result_items(config.workspace_root):
        sample_uid = str(item.get("sample_uid") or "").strip()
        if sample_uid:
            merged[sample_uid] = item
    for item in _iter_progress_result_items(config.progress_path):
        sample_uid = str(item.get("sample_uid") or "").strip()
        if not sample_uid:
            continue
        current = merged.get(sample_uid)
        if current is None or str(item.get("updated_at") or "") > str(current.get("updated_at") or ""):
            merged[sample_uid] = item
    if not merged:
        raise FileNotFoundError(
            "No participated Mix sample results were found. "
            f"Checked workspace_root={config.workspace_root} and progress_path={config.progress_path}."
        )
    return sorted(merged.values(), key=lambda item: str(item.get("updated_at") or ""))


def _resolve_run_result_path(result_item: Mapping[str, Any]) -> Optional[Path]:
    explicit_path = str(result_item.get("run_result_path") or "").strip()
    if explicit_path:
        return Path(explicit_path)
    workspace_root = str(result_item.get("workspace_root") or "").strip()
    if workspace_root:
        return Path(workspace_root) / "results" / "run_result.json"
    return None


def _extract_runtime_tool_usage_summary(run_result_payload: Mapping[str, Any]) -> Dict[str, int]:
    metrics = run_result_payload.get("metrics")
    if isinstance(metrics, dict):
        direct_summary = metrics.get("tool_usage_summary")
        if isinstance(direct_summary, dict):
            return {
                str(name).strip(): int(count)
                for name, count in direct_summary.items()
                if str(name).strip()
            }
        layers = metrics.get("tool_usage_layers")
        if isinstance(layers, dict):
            runtime_executed = layers.get("runtime_executed")
            if isinstance(runtime_executed, dict):
                nested_summary = runtime_executed.get("tool_usage_summary")
                if isinstance(nested_summary, dict):
                    return {
                        str(name).strip(): int(count)
                        for name, count in nested_summary.items()
                        if str(name).strip()
                    }
    return {}


def build_runtime_usage(result_items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tool_usage_counter: Counter[str] = Counter()
    samples_with_runtime_usage = 0
    missing_run_result_paths: List[str] = []
    missing_usage_samples: List[str] = []

    for item in result_items:
        run_result_path = _resolve_run_result_path(item)
        sample_uid = str(item.get("sample_uid") or "").strip()
        if run_result_path is None or not run_result_path.exists():
            missing_run_result_paths.append(sample_uid or str(run_result_path or ""))
            continue
        run_result_payload = _load_json_object(run_result_path)
        if run_result_payload is None:
            missing_run_result_paths.append(sample_uid or str(run_result_path))
            continue
        usage_summary = _extract_runtime_tool_usage_summary(run_result_payload)
        if not usage_summary:
            missing_usage_samples.append(sample_uid or str(run_result_path))
            continue
        tool_usage_counter.update(usage_summary)
        samples_with_runtime_usage += 1

    return {
        "tool_usage_counter": tool_usage_counter,
        "samples_total": len(result_items),
        "samples_with_runtime_usage": samples_with_runtime_usage,
        "samples_missing_run_result": missing_run_result_paths,
        "samples_missing_usage_summary": missing_usage_samples,
    }


def _lookup_tool_record(
    tool_name: str,
    discovered_records: Mapping[str, ToolRecord],
) -> ToolRecord:
    record = discovered_records.get(tool_name)
    if record is not None:
        return record
    return ToolRecord(
        name=tool_name,
        tags=(),
        tool_type="unknown",
        module="",
    )


def build_summary_payload(
    *,
    runtime_usage: Mapping[str, Any],
    discovered_records: Mapping[str, ToolRecord],
) -> Dict[str, Any]:
    tool_usage_counter = Counter(runtime_usage.get("tool_usage_counter") or {})
    granularity_counts: Counter[str] = Counter()
    function_counts: Counter[str] = Counter()
    nested_function_counts: Dict[str, Counter[str]] = {}

    for tool_name, count in tool_usage_counter.items():
        record = _lookup_tool_record(tool_name, discovered_records)
        grain_key = record.tool_type if record.tool_type in GRAIN_ORDER else "unknown"
        granularity_counts[grain_key] += int(count)
        function_name = infer_function_category(record)
        if function_name == "Other":
            function_name = OTHER_FUNCTION_LABEL
        function_counts[function_name] += int(count)
        nested_function_counts.setdefault(grain_key, Counter())
        nested_function_counts[grain_key][function_name] += int(count)

    return {
        "kind": "mix_runtime_tool_usage_summary",
        "layout_version": 1,
        "samples_total": int(runtime_usage.get("samples_total", 0) or 0),
        "samples_with_runtime_usage": int(runtime_usage.get("samples_with_runtime_usage", 0) or 0),
        "total_runtime_tool_calls": int(sum(tool_usage_counter.values())),
        "unique_runtime_tools": int(len(tool_usage_counter)),
        "granularity_counts": {
            key: int(granularity_counts.get(key, 0))
            for key in (*GRAIN_ORDER, "unknown")
            if int(granularity_counts.get(key, 0)) > 0 or key in GRAIN_ORDER
        },
        "function_counts": {
            key: int(function_counts.get(key, 0))
            for key in FUNCTION_DISPLAY_ORDER
            if int(function_counts.get(key, 0)) > 0
        },
        "function_counts_by_granularity": {
            grain_key: _order_nested_function_counts(
                nested_function_counts.get(grain_key, {}),
                grain_key=grain_key,
            )
            for grain_key in (*GRAIN_ORDER, "unknown")
            if nested_function_counts.get(grain_key)
        },
        "tool_usage_summary": dict(sorted(tool_usage_counter.items(), key=lambda item: (-item[1], item[0]))),
        "samples_missing_run_result": list(runtime_usage.get("samples_missing_run_result") or []),
        "samples_missing_usage_summary": list(runtime_usage.get("samples_missing_usage_summary") or []),
    }


def _normalize_function_counts(raw_counts: Mapping[str, Any]) -> Dict[str, int]:
    normalized_function_counts: Dict[str, int] = {}
    for raw_name, raw_count in raw_counts.items():
        name = str(raw_name).strip()
        if not name:
            continue
        if name == "Other":
            name = OTHER_FUNCTION_LABEL
        normalized_function_counts[name] = normalized_function_counts.get(name, 0) + int(raw_count or 0)
    return normalized_function_counts


def _order_nested_function_counts(
    function_counts: Mapping[str, Any],
    *,
    grain_key: str,
) -> Dict[str, int]:
    normalized_counts = _normalize_function_counts(function_counts)
    ordered_names = GRAIN_FUNCTION_ORDERS.get(grain_key, FUNCTION_ORDER)
    ordered: Dict[str, int] = {}
    for function_name in ordered_names:
        count = int(normalized_counts.get(function_name, 0) or 0)
        if count > 0:
            ordered[function_name] = count
    extras = sorted(
        name for name in normalized_counts.keys()
        if name not in ordered
    )
    for function_name in extras:
        ordered[function_name] = int(normalized_counts[function_name])
    return ordered


def _rebuild_nested_function_counts_from_tool_usage(
    *,
    tool_usage_summary: Mapping[str, Any],
    discovered_records: Mapping[str, ToolRecord],
) -> Dict[str, Dict[str, int]]:
    nested_function_counts: Dict[str, Counter[str]] = {}
    for tool_name, raw_count in tool_usage_summary.items():
        normalized_tool_name = str(tool_name).strip()
        count = int(raw_count or 0)
        if not normalized_tool_name or count <= 0:
            continue
        record = _lookup_tool_record(normalized_tool_name, discovered_records)
        grain_key = record.tool_type if record.tool_type in GRAIN_ORDER else "unknown"
        function_name = infer_function_category(record)
        if function_name == "Other":
            function_name = OTHER_FUNCTION_LABEL
        nested_function_counts.setdefault(grain_key, Counter())
        nested_function_counts[grain_key][function_name] += count
    return {
        grain_key: _order_nested_function_counts(counter, grain_key=grain_key)
        for grain_key, counter in nested_function_counts.items()
        if counter
    }


def _build_flat_function_counts(function_counts: Mapping[str, Any]) -> Dict[str, int]:
    return {
        str(name).strip(): int(value)
        for name, value in function_counts.items()
        if str(name).strip() and int(value or 0) > 0
    }


def load_summary_payload(summary_path: Path, discovered_records: Mapping[str, ToolRecord]) -> Dict[str, Any]:
    payload = _load_json_object(summary_path)
    if payload is None:
        raise FileNotFoundError(f"Failed to load runtime tool usage summary JSON: {summary_path}")
    granularity_counts = payload.get("granularity_counts")
    function_counts = payload.get("function_counts")
    if not isinstance(granularity_counts, dict) or not isinstance(function_counts, dict):
        raise ValueError(
            "Runtime tool usage summary JSON must contain object fields "
            "`granularity_counts` and `function_counts`."
        )
    payload["function_counts"] = _normalize_function_counts(function_counts)
    raw_nested_counts = payload.get("function_counts_by_granularity")
    normalized_nested_counts: Dict[str, Dict[str, int]] = {}
    tool_usage_summary = payload.get("tool_usage_summary")
    if isinstance(raw_nested_counts, dict):
        for raw_grain_key, raw_counts in raw_nested_counts.items():
            grain_key = str(raw_grain_key).strip().lower()
            if grain_key not in (*GRAIN_ORDER, "unknown") or not isinstance(raw_counts, dict):
                continue
            normalized_nested_counts[grain_key] = _order_nested_function_counts(raw_counts, grain_key=grain_key)
    elif isinstance(tool_usage_summary, dict):
        normalized_nested_counts = _rebuild_nested_function_counts_from_tool_usage(
            tool_usage_summary=tool_usage_summary,
            discovered_records=discovered_records,
        )
    payload["function_counts_by_granularity"] = normalized_nested_counts
    return payload


def _figure_prefix_from_summary_path(summary_path: Path) -> str:
    suffix = "_runtime_tool_usage_summary"
    stem = summary_path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)] or "mix_eval"
    return stem or "mix_eval"


def build_summary_mode_config(config_path: Path, summary_path: Path) -> PlotConfig:
    try:
        if config_path.exists():
            return load_plot_config(config_path)
    except Exception:
        pass
    resolved_config_path = config_path.resolve()
    return PlotConfig(
        config_path=resolved_config_path,
        mix_config_path=resolved_config_path,
        workspace_root=summary_path.parent,
        progress_path=summary_path,
        output_dir=summary_path.parent,
        figure_prefix=_figure_prefix_from_summary_path(summary_path),
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    return (red, green, blue, alpha)


def _function_color_for_grain(function_name: str, grain_key: str) -> tuple[float, float, float, float]:
    if grain_key == "meta":
        return _hex_to_rgba(FUNCTION_DISPLAY_COLORS.get(function_name, FUNCTION_COLORS["Other"]), 0.92)
    if grain_key == "unknown":
        return _hex_to_rgba(UNKNOWN_GRAIN_COLOR, 0.92)
    if grain_key == "mixed":
        return _hex_to_rgba(FUNCTION_DISPLAY_COLORS.get(function_name, FUNCTION_COLORS["Other"]), 0.84)
    return _hex_to_rgba(FUNCTION_DISPLAY_COLORS.get(function_name, FUNCTION_COLORS["Other"]), 0.74)


def _annotate_pie_wedge(
    ax: plt.Axes,
    wedge: Any,
    text: str,
    *,
    radius_scale: float,
    fontsize: float,
) -> None:
    theta = (float(wedge.theta1) + float(wedge.theta2)) / 2.0
    x = radius_scale * float(wedge.r) * math.cos(math.radians(theta))
    y = radius_scale * float(wedge.r) * math.sin(math.radians(theta))
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=ANNOTATION_TEXT_COLOR,
        linespacing=1.24,
        bbox=ANNOTATION_BBOX,
    )


def _distribute_outer_label_positions(
    label_points: Sequence[Tuple[float, float, float, float, str, str, str]],
    *,
    min_gap: float = 0.12,
    lower_bound: float = -0.98,
    upper_bound: float = 0.98,
) -> List[Tuple[float, float, float, float, str, str, str]]:
    if not label_points:
        return []

    sorted_points = sorted(label_points, key=lambda item: item[1])
    adjusted_points: List[List[Any]] = [list(item) for item in sorted_points]

    for index in range(1, len(adjusted_points)):
        prev_y = float(adjusted_points[index - 1][1])
        current_y = float(adjusted_points[index][1])
        if current_y - prev_y < min_gap:
            adjusted_points[index][1] = prev_y + min_gap

    overflow = float(adjusted_points[-1][1]) - upper_bound
    if overflow > 0:
        for item in adjusted_points:
            item[1] = float(item[1]) - overflow

    for index in range(len(adjusted_points) - 2, -1, -1):
        next_y = float(adjusted_points[index + 1][1])
        current_y = float(adjusted_points[index][1])
        if next_y - current_y < min_gap:
            adjusted_points[index][1] = next_y - min_gap

    underflow = lower_bound - float(adjusted_points[0][1])
    if underflow > 0:
        for item in adjusted_points:
            item[1] = float(item[1]) + underflow

    return [
        (
            float(item[0]),
            float(item[1]),
            float(item[2]),
            float(item[3]),
            str(item[4]),
            str(item[5]),
            str(item[6]),
        )
        for item in adjusted_points
    ]


def _annotate_outer_wedges(
    ax: plt.Axes,
    wedges: Sequence[Any],
    labels: Sequence[str],
    values: Sequence[int],
    colors: Sequence[tuple[float, float, float, float]],
    *,
    total_calls: int,
    min_share: float = 4.5,
) -> List[Tuple[str, float, tuple[float, float, float, float]]]:
    left_points: List[Tuple[float, float, float, float, str, str, str]] = []
    right_points: List[Tuple[float, float, float, float, str, str, str]] = []
    minor_items: List[Tuple[str, float, tuple[float, float, float, float]]] = []

    def _format_share_percent(share: float) -> str:
        return f"{share:.1f}%" if share < 1.0 else f"{share:.0f}%"

    for wedge, label, value, color in zip(wedges, labels, values, colors):
        share = 100.0 * float(value) / float(total_calls)
        if share < min_share:
            minor_items.append((label, share, color))
            continue
        theta = math.radians((float(wedge.theta1) + float(wedge.theta2)) / 2.0)
        anchor_x = 1.005 * float(wedge.r) * math.cos(theta)
        anchor_y = 1.005 * float(wedge.r) * math.sin(theta)
        label_radius = 1.10
        label_x = label_radius * math.cos(theta)
        label_y = label_radius * math.sin(theta)
        text = f"{label}\n{_format_share_percent(share)}"
        if anchor_x >= 0:
            right_points.append((anchor_x, anchor_y, label_x, label_y, text, "left", "right"))
        else:
            left_points.append((anchor_x, anchor_y, label_x, label_y, text, "right", "left"))

    for side_points in (
        _distribute_outer_label_positions(left_points),
        _distribute_outer_label_positions(right_points),
    ):
        for anchor_x, anchor_y, label_x, label_y, text, horizontal_align, connectionstyle in side_points:
            ax.annotate(
                text,
                xy=(anchor_x, anchor_y),
                xytext=(label_x, label_y),
                ha=horizontal_align,
                va="center",
                fontsize=26.0,
                color=OUTER_LABEL_TEXT_COLOR,
                linespacing=1.22,
                arrowprops={
                    "arrowstyle": "-",
                    "color": OUTER_LABEL_LINE_COLOR,
                    "lw": 0.9,
                    "shrinkA": 0,
                    "shrinkB": 3,
                    "connectionstyle": f"arc3,rad={-0.08 if connectionstyle == 'right' else 0.08}",
                },
            )
    return minor_items


def _annotate_minor_slice_legend(
    ax: plt.Axes,
    minor_items: Sequence[Tuple[str, float, tuple[float, float, float, float]]],
) -> None:
    if not minor_items:
        return
    handles = [
        Patch(
            facecolor=color,
            edgecolor=AXIS_BACKGROUND,
            linewidth=0.8,
            label=(f"{label} ({share:.1f}%)" if share < 1.0 else f"{label} ({share:.0f}%)"),
        )
        for label, share, color in minor_items
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=22.0,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0.0,
    )


def _build_outer_segments(payload: Mapping[str, Any]) -> tuple[List[int], List[str], List[str]]:
    total_calls = int(payload.get("total_runtime_tool_calls", 0) or 0)
    nested_counts = payload.get("function_counts_by_granularity", {})
    outer_values: List[int] = []
    outer_labels: List[str] = []
    outer_grains: List[str] = []

    if isinstance(nested_counts, dict):
        for grain_key in [*GRAIN_ORDER, "unknown"]:
            grain_counts = nested_counts.get(grain_key)
            if not isinstance(grain_counts, dict):
                continue
            ordered_names = GRAIN_FUNCTION_ORDERS.get(grain_key, FUNCTION_ORDER)
            function_items: List[Tuple[str, int]] = []
            seen_names = set()
            for function_name in ordered_names:
                count = int(grain_counts.get(function_name, 0) or 0)
                if count <= 0:
                    continue
                function_items.append((function_name, count))
                seen_names.add(function_name)
            extras = sorted(
                (
                    (name, int(count))
                    for name, count in grain_counts.items()
                    if int(count) > 0 and str(name) not in seen_names
                ),
                key=lambda item: (-item[1], item[0]),
            )
            function_items.extend(extras)
            for function_name, count in function_items:
                outer_values.append(count)
                outer_labels.append(function_name)
                outer_grains.append(grain_key)

    if sum(outer_values) == total_calls and outer_values:
        return outer_values, outer_labels, outer_grains

    flat_function_counts = _build_flat_function_counts(payload.get("function_counts", {}))
    flat_items = sorted(flat_function_counts.items(), key=lambda item: (-item[1], item[0]))
    if flat_items:
        return (
            [count for _, count in flat_items],
            [name for name, _ in flat_items],
            ["mixed"] * len(flat_items),
        )

    return [], [], []


def plot_nested_pie_chart(
    ax: plt.Axes,
    *,
    payload: Mapping[str, Any],
) -> None:
    granularity_counts = payload.get("granularity_counts", {})
    grain_order = [key for key in GRAIN_ORDER if int((granularity_counts or {}).get(key, 0) or 0) > 0]
    if int((granularity_counts or {}).get("unknown", 0) or 0) > 0:
        grain_order.append("unknown")
    total_calls = int(payload.get("total_runtime_tool_calls", 0) or 0)

    ax.set_facecolor(AXIS_BACKGROUND)
    if total_calls <= 0 or not grain_order:
        ax.text(
            0.5,
            0.5,
            "No runtime tool usage found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=TEXT_COLOR,
            fontsize=24.0,
        )
        ax.set_axis_off()
        return

    grain_labels = {
        "biz": "base",
        "meta": "meta",
        "unknown": "unknown",
    }
    grain_colors = {
        "biz": _hex_to_rgba(GRAIN_COLORS["biz"], 0.88),
        "meta": _hex_to_rgba(GRAIN_COLORS["meta"], 0.88),
        "unknown": _hex_to_rgba(UNKNOWN_GRAIN_COLOR, 0.88),
    }
    inner_values = [int(granularity_counts.get(key, 0) or 0) for key in grain_order]
    inner_colors = [grain_colors[key] for key in grain_order]
    outer_values, outer_labels, outer_grains = _build_outer_segments(payload)
    outer_colors = [
        _function_color_for_grain(function_name, grain_key)
        for function_name, grain_key in zip(outer_labels, outer_grains)
    ]

    inner_wedges, _ = ax.pie(
        inner_values,
        radius=0.56,
        colors=inner_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.20, "edgecolor": AXIS_BACKGROUND, "linewidth": 1.3},
    )
    outer_wedges, _ = ax.pie(
        outer_values,
        radius=0.97,
        colors=outer_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.31, "edgecolor": AXIS_BACKGROUND, "linewidth": 1.15},
    )
    ax.set_aspect("equal")

    for wedge, grain_key, value in zip(inner_wedges, grain_order, inner_values):
        share = 100.0 * float(value) / float(total_calls)
        if share >= 9.0:
            _annotate_pie_wedge(
                ax,
                wedge,
                f"{grain_labels[grain_key]}\n{share:.0f}%",
                radius_scale=0.66,
                fontsize=26.0,
            )
    minor_items = _annotate_outer_wedges(
        ax,
        outer_wedges,
        outer_labels,
        outer_values,
        outer_colors,
        total_calls=total_calls,
    )
    _annotate_minor_slice_legend(ax, minor_items)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.08, 1.08)
    ax.set_axis_off()


def save_outputs(
    fig: plt.Figure,
    payload: Mapping[str, Any],
    output_dir: Path,
    figure_prefix: str,
    *,
    write_summary_json: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_stem = output_dir / f"{figure_prefix}_runtime_tool_usage"
    fig.savefig(str(figure_stem.with_suffix(".png")), bbox_inches="tight")
    fig.savefig(str(figure_stem.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)

    if write_summary_json:
        summary_path = output_dir / f"{figure_prefix}_runtime_tool_usage_summary.json"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(payload: Mapping[str, Any], config: PlotConfig) -> None:
    print("Mix runtime tool usage chart generated.")
    print(f"- samples_total={payload.get('samples_total', 0)}")
    print(f"- samples_with_runtime_usage={payload.get('samples_with_runtime_usage', 0)}")
    print(f"- total_runtime_tool_calls={payload.get('total_runtime_tool_calls', 0)}")
    print(f"- unique_runtime_tools={payload.get('unique_runtime_tools', 0)}")
    nested_counts = payload.get("function_counts_by_granularity", {})
    if isinstance(nested_counts, dict):
        for grain_key in GRAIN_ORDER:
            grain_counts = nested_counts.get(grain_key, {})
            if not isinstance(grain_counts, dict):
                continue
            print(f"- {grain_key} runtime function counts:")
            for function_name, count in grain_counts.items():
                print(f"  - {function_name}: {int(count)}")
    print(f"Workspace root: {config.workspace_root}")
    print(f"Output dir: {config.output_dir}")


def main() -> None:
    args = parse_args()
    discovered_records = discover_tool_records(tool_star_root())
    summary_json_text = str(args.summary_json or "").strip()
    write_summary_json = True
    if summary_json_text:
        summary_path = Path(summary_json_text).resolve()
        config = build_summary_mode_config(Path(args.config), summary_path)
        payload = load_summary_payload(summary_path, discovered_records)
        write_summary_json = not bool(args.no_update_summary_json)
    else:
        config = load_plot_config(Path(args.config))
        result_items = load_result_items(config)
        runtime_usage = build_runtime_usage(result_items)
        payload = build_summary_payload(
            runtime_usage=runtime_usage,
            discovered_records=discovered_records,
        )

    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(13.2, 8.4))
    fig.patch.set_facecolor(PAPER_BACKGROUND)
    plot_nested_pie_chart(ax, payload=payload)
    fig.subplots_adjust(top=0.96, bottom=0.17, left=0.04, right=0.96)

    save_outputs(
        fig,
        payload,
        config.output_dir,
        config.figure_prefix,
        write_summary_json=write_summary_json,
    )
    print_summary(payload, config)


if __name__ == "__main__":
    main()
