from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter


PARTICIPATED_STATUSES = {"completed", "wrong"}
SAMPLE_RESULT_FILENAME = "mix_sample_result.json"
DATASET_ORDER = ("lvbench", "videomme", "egoschema", "mlvu", "mvbench")
DATASET_DISPLAY_NAMES = {
    "lvbench": "LVBench",
    "egoschema": "EgoSchema",
    "mvbench": "MVBench",
    "mlvu": "MLVU",
    "videomme": "Video-MME",
}

PALETTE_ALPHA = 0.6
PAPER_BACKGROUND = "#FFFFFF"
AXIS_BACKGROUND = "#FFFFFF"
COUNT_TEXT_COLOR = "#64748B"
PLOT_PALETTES = (
    {"dark": "#6C8EBF", "light": "#DAE8FC"},
    {"dark": "#82B366", "light": "#D5E8D4"},
    {"dark": "#9673A6", "light": "#E1D5E7"},
    {"dark": "#B85450", "light": "#F8CECC"},
)

TOOL_USE_BUCKETS = ("0", "1", "2", "3", "4", "5", "6+")
ITER_BUCKETS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10+")
SUCCESS_RATE_BUCKETS = ("[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]")

HEATMAP_TOOL_USE_BUCKETS = ("0", "1", "2", "3-4", "5+")
HEATMAP_ITER_BUCKETS = ("1-2", "3-4", "5-6", "7-8", "9+")


@dataclass(frozen=True)
class PlotConfig:
    config_path: Path
    mix_config_path: Path
    workspace_root: Path
    progress_path: Path
    output_dir: Path
    figure_prefix: str


@dataclass(frozen=True)
class SampleRecord:
    sample_uid: str
    dataset_name: str
    subset_name: str
    status: str
    acc: float
    tool_use_num: int
    iter_num: int
    tool_success_rate: float
    updated_at: str


@dataclass(frozen=True)
class BucketSeries:
    labels: Sequence[str]
    mean_acc: List[float]
    band_low: List[float]
    band_high: List[float]
    counts: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Mix evaluation figures from workspace results.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("mix_plot_config.json")),
        help="Path to the plotting config JSON.",
    )
    return parser.parse_args()


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


def _strip_json_line_comments(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines)


def _load_json_object(path: Path) -> Dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8-sig")
    normalized = _strip_json_line_comments(_unwrap_markdown_code_fence(raw_text)).strip()
    if not normalized:
        raise ValueError(f"JSON payload is empty: {path}")
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def load_plot_config(config_path: Path) -> PlotConfig:
    config_path = config_path.resolve()
    payload = _load_json_object(config_path)
    config_dir = config_path.parent

    mix_config_path = _resolve_path(
        config_dir,
        str(payload.get("mix_config_path") or "../evaluationVideo/Mix/mix_config.json"),
    )
    mix_payload = _load_json_object(mix_config_path)
    mix_dir = mix_config_path.parent

    workspace_override = str(payload.get("workspace_root") or "").strip()
    progress_override = str(payload.get("progress_path") or "").strip()
    output_dir = _resolve_path(config_dir, str(payload.get("output_dir") or "./out"))

    workspace_root = _resolve_path(
        config_dir if workspace_override else mix_dir,
        workspace_override or str(mix_payload.get("workspaces_root") or "./workspaces"),
    )
    progress_path = _resolve_path(
        config_dir if progress_override else mix_dir,
        progress_override or str(mix_payload.get("progress_path") or "./workspaces/mix_progress.json"),
    )
    figure_prefix = str(payload.get("figure_prefix") or "mix_eval").strip() or "mix_eval"

    return PlotConfig(
        config_path=config_path,
        mix_config_path=mix_config_path,
        workspace_root=workspace_root,
        progress_path=progress_path,
        output_dir=output_dir,
        figure_prefix=figure_prefix,
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_record(payload: Mapping[str, Any]) -> Optional[SampleRecord]:
    if not isinstance(payload, Mapping):
        return None
    status = _normalize_status(payload.get("status"))
    if status not in PARTICIPATED_STATUSES:
        return None
    sample_uid = str(payload.get("sample_uid") or "").strip()
    dataset_name = str(payload.get("dataset_name") or "").strip().lower()
    if not sample_uid or not dataset_name:
        return None
    return SampleRecord(
        sample_uid=sample_uid,
        dataset_name=dataset_name,
        subset_name=str(payload.get("subset_name") or "").strip(),
        status=status,
        acc=1.0 if _safe_float(payload.get("acc"), 0.0) >= 1.0 else 0.0,
        tool_use_num=max(0, _safe_int(payload.get("tool_use_num"), 0)),
        iter_num=max(0, _safe_int(payload.get("iter_num"), 0)),
        tool_success_rate=min(1.0, max(0.0, _safe_float(payload.get("tool_success_rate"), 0.0))),
        updated_at=str(payload.get("updated_at") or ""),
    )


def _scan_workspace_results(workspace_root: Path) -> List[SampleRecord]:
    if not workspace_root.exists():
        return []
    records: List[SampleRecord] = []
    for result_path in workspace_root.rglob(SAMPLE_RESULT_FILENAME):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        record = _build_record(payload)
        if record is not None:
            records.append(record)
    return records


def _load_progress_records(progress_path: Path) -> List[SampleRecord]:
    if not progress_path.exists():
        return []
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_records = payload.get("records")
    if not isinstance(raw_records, dict):
        return []
    records: List[SampleRecord] = []
    for item in raw_records.values():
        record = _build_record(item)
        if record is not None:
            records.append(record)
    return records


def load_records(config: PlotConfig) -> List[SampleRecord]:
    merged: MutableMapping[str, SampleRecord] = {}
    for record in _scan_workspace_results(config.workspace_root):
        merged[record.sample_uid] = record
    for record in _load_progress_records(config.progress_path):
        current = merged.get(record.sample_uid)
        if current is None or record.updated_at > current.updated_at:
            merged[record.sample_uid] = record
    records = sorted(
        merged.values(),
        key=lambda item: (DATASET_ORDER.index(item.dataset_name) if item.dataset_name in DATASET_ORDER else 999, item.sample_uid),
    )
    if not records:
        raise FileNotFoundError(
            "No participated Mix sample results were found. "
            f"Checked workspace_root={config.workspace_root} and progress_path={config.progress_path}."
        )
    return records


def _mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / float(len(values))


def _make_bucket_series(
    records: Sequence[SampleRecord],
    labels: Sequence[str],
    bucket_fn: Callable[[SampleRecord], Optional[str]],
) -> BucketSeries:
    per_bucket: Dict[str, List[float]] = {label: [] for label in labels}
    per_dataset_bucket: Dict[str, Dict[str, List[float]]] = {
        dataset_name: {label: [] for label in labels} for dataset_name in DATASET_ORDER
    }

    for record in records:
        label = bucket_fn(record)
        if label not in per_bucket:
            continue
        per_bucket[label].append(record.acc)
        per_dataset_bucket.setdefault(record.dataset_name, {item: [] for item in labels})
        per_dataset_bucket[record.dataset_name][label].append(record.acc)

    mean_acc: List[float] = []
    band_low: List[float] = []
    band_high: List[float] = []
    counts: List[int] = []

    for label in labels:
        overall_values = per_bucket[label]
        dataset_means = [
            _mean(per_dataset_bucket[dataset_name][label])
            for dataset_name in DATASET_ORDER
            if dataset_name in per_dataset_bucket and per_dataset_bucket[dataset_name][label]
        ]
        mean_acc.append(_mean(overall_values))
        band_low.append(min(dataset_means) if dataset_means else math.nan)
        band_high.append(max(dataset_means) if dataset_means else math.nan)
        counts.append(len(overall_values))

    return BucketSeries(
        labels=list(labels),
        mean_acc=mean_acc,
        band_low=band_low,
        band_high=band_high,
        counts=counts,
    )


def tool_use_bucket(record: SampleRecord) -> str:
    value = max(0, record.tool_use_num)
    return str(value) if value <= 5 else "6+"


def iter_bucket(record: SampleRecord) -> Optional[str]:
    value = record.iter_num
    if value <= 0:
        return None
    return str(value) if value <= 9 else "10+"


def success_rate_bucket(record: SampleRecord) -> Optional[str]:
    if record.tool_use_num <= 0:
        return None
    value = record.tool_success_rate
    if value < 0.2:
        return "[0.0, 0.2)"
    if value < 0.4:
        return "[0.2, 0.4)"
    if value < 0.6:
        return "[0.4, 0.6)"
    if value < 0.8:
        return "[0.6, 0.8)"
    return "[0.8, 1.0]"


def heatmap_tool_use_bucket(record: SampleRecord) -> str:
    value = max(0, record.tool_use_num)
    if value <= 2:
        return str(value)
    if value <= 4:
        return "3-4"
    return "5+"


def heatmap_iter_bucket(record: SampleRecord) -> Optional[str]:
    value = record.iter_num
    if value <= 0:
        return None
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    if value <= 6:
        return "5-6"
    if value <= 8:
        return "7-8"
    return "9+"


def _pick_preferred_serif_font() -> tuple[str, List[str]]:
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Nimbus Roman No9 L",
        "Liberation Serif",
        "STIX Two Text",
        "STIXGeneral",
        "DejaVu Serif",
    ]
    primary_font = next((name for name in candidates if name in installed_fonts), "DejaVu Serif")
    serif_fallbacks = [primary_font]
    serif_fallbacks.extend(name for name in candidates if name != primary_font)
    return primary_font, serif_fallbacks


def _set_paper_style() -> None:
    primary_font, serif_fallbacks = _pick_preferred_serif_font()
    plt.rcParams.update(
        {
            "font.family": [primary_font, "serif"],
            "font.serif": serif_fallbacks,
            "font.size": 14.2,
            "axes.titlesize": 18.6,
            "axes.labelsize": 16.4,
            "axes.edgecolor": "#25303B",
            "axes.linewidth": 0.9,
            "xtick.color": "#25303B",
            "ytick.color": "#25303B",
            "grid.color": "#D8DEE6",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "legend.frameon": False,
        }
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> Tuple[float, float, float, float]:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    return (red, green, blue, alpha)


def _annotate_bucket_counts(ax: plt.Axes, labels: Sequence[str], counts: Sequence[int]) -> None:
    for index, (label, count) in enumerate(zip(labels, counts)):
        ax.text(
            index,
            -0.125,
            f"n={count}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10.8,
            color=COUNT_TEXT_COLOR,
            linespacing=1.25,
        )


def _format_bucket_tick_labels(labels: Sequence[str]) -> List[str]:
    formatted: List[str] = []
    for label in labels:
        text = str(label)
        if text.startswith("[") and "," in text:
            left, right = text.split(",", 1)
            formatted.append(f"{left},\n{right.strip()}")
            continue
        formatted.append(text)
    return formatted


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_stem.with_suffix(".png")), bbox_inches="tight")
    fig.savefig(str(output_stem.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def _draw_bucket_accuracy_panel(
    *,
    ax: plt.Axes,
    title: str,
    x_label: str,
    series: BucketSeries,
    palette_index: int,
    show_ylabel: bool,
    legend: bool,
    show_counts: bool,
) -> None:
    palette = PLOT_PALETTES[palette_index % len(PLOT_PALETTES)]
    band_color = _hex_to_rgba(palette["light"], PALETTE_ALPHA)
    ax.set_facecolor(AXIS_BACKGROUND)
    ax.grid(axis="y")
    x_values = list(range(len(series.labels)))

    ax.fill_between(
        x_values,
        series.band_low,
        series.band_high,
        color=band_color,
        linewidth=0,
        label="Dataset spread (min-max)",
    )
    ax.plot(
        x_values,
        series.mean_acc,
        color=palette["dark"],
        marker="o",
        markersize=8.6,
        linewidth=3.2,
        label="Overall accuracy",
    )
    ax.scatter(
        x_values,
        series.mean_acc,
        color=_hex_to_rgba(palette["dark"], PALETTE_ALPHA),
        edgecolors=AXIS_BACKGROUND,
        linewidths=1.0,
        s=72,
        zorder=3,
    )

    ax.set_title(title, pad=14, fontsize=20.8)
    ax.set_xlabel(x_label, fontsize=17.8)
    if show_ylabel:
        ax.set_ylabel("Accuracy", fontsize=17.8)
    ax.set_xticks(x_values, series.labels)
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    formatted_labels = _format_bucket_tick_labels(series.labels)
    ax.tick_params(axis="y", labelsize=16.0)
    ax.tick_params(axis="x", labelsize=14.6, pad=6)
    ax.set_xticklabels(formatted_labels, fontsize=14.6)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(16.0)
    for tick_label in ax.get_xticklabels():
        tick_label.set_linespacing(0.96)
    if legend:
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            fontsize=15.2,
            handlelength=1.8,
            labelspacing=0.55,
            borderaxespad=0.25,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_counts:
        _annotate_bucket_counts(ax, series.labels, series.counts)


def plot_combined_bucket_accuracy_figure(
    *,
    series_items: Sequence[Tuple[str, str, BucketSeries, int]],
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(1, len(series_items), figsize=(18.6, 5.2), sharey=True)
    fig.patch.set_facecolor(PAPER_BACKGROUND)

    if len(series_items) == 1:
        axes = [axes]

    for index, (title, x_label, series, palette_index) in enumerate(series_items):
        _draw_bucket_accuracy_panel(
            ax=axes[index],
            title=title,
            x_label=x_label,
            series=series,
            palette_index=palette_index,
            show_ylabel=True,
            legend=True,
            show_counts=False,
        )

    fig.subplots_adjust(wspace=0.28, bottom=0.29, top=0.86)
    _save_figure(fig, output_stem)


def _build_heatmap_matrix(records: Sequence[SampleRecord]) -> Tuple[List[List[float]], List[List[int]]]:
    value_grid: Dict[str, Dict[str, List[float]]] = {
        iter_label: {tool_label: [] for tool_label in HEATMAP_TOOL_USE_BUCKETS}
        for iter_label in HEATMAP_ITER_BUCKETS
    }
    for record in records:
        tool_label = heatmap_tool_use_bucket(record)
        iter_label = heatmap_iter_bucket(record)
        if iter_label is None:
            continue
        value_grid[iter_label][tool_label].append(record.acc)

    matrix: List[List[float]] = []
    counts: List[List[int]] = []
    for iter_label in HEATMAP_ITER_BUCKETS:
        row_values: List[float] = []
        row_counts: List[int] = []
        for tool_label in HEATMAP_TOOL_USE_BUCKETS:
            cell = value_grid[iter_label][tool_label]
            row_values.append(_mean(cell))
            row_counts.append(len(cell))
        matrix.append(row_values)
        counts.append(row_counts)
    return matrix, counts


def plot_heatmap(records: Sequence[SampleRecord], output_stem: Path) -> None:
    matrix, counts = _build_heatmap_matrix(records)
    fig, ax = plt.subplots(figsize=(7.3, 5.5))
    fig.patch.set_facecolor(PAPER_BACKGROUND)
    ax.set_facecolor(AXIS_BACKGROUND)
    heatmap_palette = PLOT_PALETTES[3]

    cmap = LinearSegmentedColormap.from_list(
        "mix_accuracy_heatmap",
        [
            PAPER_BACKGROUND,
            _hex_to_rgba(heatmap_palette["light"], PALETTE_ALPHA),
            _hex_to_rgba(heatmap_palette["dark"], PALETTE_ALPHA),
            heatmap_palette["dark"],
        ],
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_title("Heatmap of Accuracy over Tool Use Number and Iteration Number", pad=14, fontsize=33.0)
    ax.set_xlabel("Tool Use Number", fontsize=28.0)
    ax.set_ylabel("Iteration Number", fontsize=28.0)
    ax.set_xticks(list(range(len(HEATMAP_TOOL_USE_BUCKETS))), HEATMAP_TOOL_USE_BUCKETS)
    ax.set_yticks(list(range(len(HEATMAP_ITER_BUCKETS))), HEATMAP_ITER_BUCKETS)
    ax.tick_params(axis="both", labelsize=24.0)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            count = counts[row_index][col_index]
            if math.isnan(value):
                label = "NA"
                text_color = "#44515E"
            else:
                label = f"{value * 100:.0f}%\n(n={count})"
                text_color = "#FFFFFF" if value >= 0.58 else "#1F2933"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=18.0,
                color=text_color,
                linespacing=1.35,
            )

    for row_index in range(len(HEATMAP_ITER_BUCKETS) + 1):
        ax.axhline(row_index - 0.5, color="#FFFFFF", linewidth=1.0)
    for col_index in range(len(HEATMAP_TOOL_USE_BUCKETS) + 1):
        ax.axvline(col_index - 0.5, color="#FFFFFF", linewidth=1.0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Accuracy", fontsize=24.0)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    colorbar.ax.tick_params(labelsize=22.0)

    _save_figure(fig, output_stem)


def print_summary(records: Sequence[SampleRecord], config: PlotConfig) -> None:
    dataset_counts: Dict[str, int] = {dataset: 0 for dataset in DATASET_ORDER}
    for record in records:
        dataset_counts[record.dataset_name] = dataset_counts.get(record.dataset_name, 0) + 1
    print(f"Loaded {len(records)} participated Mix samples.")
    for dataset_name in DATASET_ORDER:
        if dataset_counts.get(dataset_name, 0) <= 0:
            continue
        display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
        print(f"  - {display_name}: {dataset_counts[dataset_name]}")
    print(f"Workspace root: {config.workspace_root}")
    print(f"Output dir: {config.output_dir}")


def main() -> None:
    args = parse_args()
    config = load_plot_config(Path(args.config))
    records = load_records(config)
    _set_paper_style()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tool_use_series = _make_bucket_series(records, TOOL_USE_BUCKETS, tool_use_bucket)
    iter_series = _make_bucket_series(records, ITER_BUCKETS, iter_bucket)
    success_rate_series = _make_bucket_series(records, SUCCESS_RATE_BUCKETS, success_rate_bucket)

    prefix = config.figure_prefix
    plot_combined_bucket_accuracy_figure(
        series_items=(
            ("Accuracy vs. Tool Use Number", "Tool Use Number", tool_use_series, 0),
            ("Accuracy vs. Planner Round", "Planner Round", iter_series, 1),
            ("Accuracy vs. Tool Success Rate", "Tool Success Rate", success_rate_series, 2),
        ),
        output_stem=config.output_dir / f"{prefix}_accuracy_triptych",
    )
    plot_heatmap(
        records,
        output_stem=config.output_dir / f"{prefix}_accuracy_heatmap_tool_use_vs_iteration",
    )
    print_summary(records, config)


if __name__ == "__main__":
    main()
