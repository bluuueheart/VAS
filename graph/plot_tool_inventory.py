from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec


PAPER_BACKGROUND = "#FFFFFF"
AXIS_BACKGROUND = "#FFFFFF"
TEXT_COLOR = "#25303B"
GRID_COLOR = "#D8DEE6"
BAR_ALPHA = 0.82

GRAIN_ORDER = ("biz", "meta")
GRAIN_LABELS = {
    "biz": "base",
    "meta": "meta",
}
GRAIN_COLORS = {
    "biz": "#6C8EBF",
    "meta": "#82B366",
}

BASE_FUNCTION_ORDER = (
    "Retrieval/Search",
    "Visual/Video",
    "Audio/Speech",
    "Execution/Coding",
    "Memory/System",
    "Financial Analysis",
    "Other",
)
META_FUNCTION_ORDER = (
    "Summary/Aggregation",
    "Filter/Predicate",
    "Ranking/Selection",
    "Grouping/Frequency",
    "Projection/Reshape",
    "Text/Regex",
    "Math/Statistics",
    "Temporal/Window",
    "Graph Ops",
    "Logic/Boolean",
    "Sampling/Subsetting",
    "Threshold/Binning",
    "Finance Ops",
    "Other",
)
FUNCTION_ORDER = tuple(dict.fromkeys(BASE_FUNCTION_ORDER + META_FUNCTION_ORDER))
GRAIN_FUNCTION_ORDERS = {
    "biz": BASE_FUNCTION_ORDER,
    "meta": META_FUNCTION_ORDER,
}
FUNCTION_COLORS = {
    "Retrieval/Search": "#6E8FB0",
    "Visual/Video": "#B6CEE0",
    "Audio/Speech": "#8DBEBB",
    "Memory/System": "#A8CFA0",
    "Execution/Coding": "#6FAD71",
    "Temporal/Window": "#C3AF67",
    "Graph Ops": "#B8A05A",
    "Financial Analysis": "#D8C76A",
    "Finance Ops": "#D4C168",
    "Filter/Predicate": "#E8C28E",
    "Threshold/Binning": "#DEB17E",
    "Sampling/Subsetting": "#D49C68",
    "Summary/Aggregation": "#CC7B7D",
    "Ranking/Selection": "#C58AA6",
    "Grouping/Frequency": "#B99CBC",
    "Projection/Reshape": "#BFAFCA",
    "Text/Regex": "#C6BEB8",
    "Math/Statistics": "#A48872",
    "Logic/Boolean": "#8B8581",
    "Other": "#D8D8D8",
}


@dataclass(frozen=True)
class ToolRecord:
    name: str
    tags: Tuple[str, ...]
    tool_type: str
    module: str


@dataclass(frozen=True)
class ToolInventoryPlotConfig:
    config_path: Optional[Path]
    output_dir: Path
    figure_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a single enabled-tool inventory chart under two classification schemes."
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional JSON config path. Supports output_dir and figure_prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to write the figure and summary JSON.",
    )
    return parser.parse_args()


def tool_star_root() -> Path:
    return (Path(__file__).resolve().parents[1] / "Tool_Star_RL").resolve()


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


def load_plot_config(args: argparse.Namespace) -> ToolInventoryPlotConfig:
    config_arg = str(getattr(args, "config", "") or "").strip()
    output_dir_arg = str(getattr(args, "output_dir", "") or "").strip()
    default_output_dir = Path(__file__).with_name("out").resolve()
    default_figure_prefix = "tool_inventory_overview"

    if config_arg:
        config_path = Path(config_arg).resolve()
        payload = _load_json_object(config_path)
        output_dir = _resolve_path(
            config_path.parent,
            str(payload.get("output_dir") or output_dir_arg or default_output_dir),
        )
        figure_prefix = str(payload.get("figure_prefix") or default_figure_prefix).strip() or default_figure_prefix
        return ToolInventoryPlotConfig(
            config_path=config_path,
            output_dir=output_dir,
            figure_prefix=figure_prefix,
        )

    output_dir = Path(output_dir_arg).resolve() if output_dir_arg else default_output_dir
    return ToolInventoryPlotConfig(
        config_path=None,
        output_dir=output_dir,
        figure_prefix=default_figure_prefix,
    )


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


def set_plot_style() -> None:
    primary_font, serif_fallbacks = _pick_preferred_serif_font()
    plt.rcParams.update(
        {
            "font.family": [primary_font, "serif"],
            "font.serif": serif_fallbacks,
            "font.size": 22.0,
            "axes.titlesize": 28.0,
            "axes.labelsize": 24.0,
            "axes.edgecolor": TEXT_COLOR,
            "axes.linewidth": 0.9,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


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


def normalize_tags(tags: Iterable[str], *, include_meta: bool = False) -> Tuple[str, ...]:
    values = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
    if include_meta:
        values.append("meta")
    return tuple(dict.fromkeys(values))


def extract_tool_record_from_class(node: ast.ClassDef, module_name: str) -> Optional[ToolRecord]:
    if not class_inherits_base_tool(node):
        return None

    tool_name: Optional[str] = None
    tags: Optional[List[str]] = None
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id == "name":
                tool_name = literal_str(statement.value) or tool_name
            elif statement.target.id == "tags":
                tags = literal_str_sequence(statement.value) or tags
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "name":
                    tool_name = literal_str(statement.value) or tool_name
                elif target.id == "tags":
                    tags = literal_str_sequence(statement.value) or tags

    normalized_name = str(tool_name or "").strip()
    if not normalized_name:
        return None
    normalized_tags = normalize_tags(tags or [])
    tool_type = "meta" if "meta" in normalized_tags else "biz"
    return ToolRecord(
        name=normalized_name,
        tags=normalized_tags,
        tool_type=tool_type,
        module=module_name,
    )


def extract_meta_tool_records(module_path: Path, module_name: str) -> Dict[str, ToolRecord]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    records: Dict[str, ToolRecord] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "META_TOOL_SPECS" for target in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for item in node.value.elts:
            if not isinstance(item, ast.Tuple) or len(item.elts) < 4:
                continue
            tool_name = literal_str(item.elts[0])
            tags = literal_str_sequence(item.elts[3]) or []
            if not tool_name:
                continue
            records[tool_name] = ToolRecord(
                name=tool_name,
                tags=normalize_tags(tags, include_meta=True),
                tool_type="meta",
                module=module_name,
            )
        break
    return records


def discover_tool_records(project_root: Path) -> Dict[str, ToolRecord]:
    impl_dir = project_root / "toolkit" / "impl"
    records: Dict[str, ToolRecord] = {}

    for module_path in sorted(impl_dir.glob("*.py")):
        if module_path.name.startswith("_"):
            continue
        module_name = f"toolkit.impl.{module_path.stem}"
        if module_path.name == "meta_tools.py":
            records.update(extract_meta_tool_records(module_path, module_name))
            continue

        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            record = extract_tool_record_from_class(node, module_name)
            if record is not None:
                records[record.name] = record

    return dict(sorted(records.items()))


def load_settings(project_root: Path) -> Any:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from config.settings import settings

    return settings


def select_records(records: Mapping[str, ToolRecord], *, excluded_names: Iterable[str]) -> Dict[str, ToolRecord]:
    excluded = {str(name).strip() for name in excluded_names if str(name).strip()}
    return {name: record for name, record in records.items() if name not in excluded}


def build_enabled_records(project_root: Path) -> Dict[str, ToolRecord]:
    settings = load_settings(project_root)
    all_records = discover_tool_records(project_root)
    disabled_names = {str(name).strip() for name in (settings.disabled_tool_names or ()) if str(name).strip()}
    return select_records(all_records, excluded_names=disabled_names)


def _order_function_counts(
    function_counts: Mapping[str, Any],
    *,
    grain_key: str,
) -> Dict[str, int]:
    ordered_names = GRAIN_FUNCTION_ORDERS.get(str(grain_key).strip(), FUNCTION_ORDER)
    normalized_counts = {
        str(name).strip(): int(value or 0)
        for name, value in function_counts.items()
        if str(name).strip() and int(value or 0) > 0
    }
    ordered: Dict[str, int] = {}
    for function_name in ordered_names:
        count = int(normalized_counts.get(function_name, 0))
        if count > 0:
            ordered[function_name] = count
    extras = sorted(
        name for name in normalized_counts.keys()
        if name not in ordered
    )
    for function_name in extras:
        ordered[function_name] = int(normalized_counts[function_name])
    return ordered


def infer_function_category(record: ToolRecord) -> str:
    tags = set(record.tags)
    name = record.name

    if record.tool_type == "meta":
        if tags & {"finance", "interest", "loan", "payment", "schedule", "compare"}:
            return "Finance Ops"
        if tags & {"threshold", "bin"}:
            return "Threshold/Binning"
        if tags & {"sample"}:
            return "Sampling/Subsetting"
        if tags & {"logic"}:
            return "Logic/Boolean"
        if tags & {"graph"}:
            return "Graph Ops"
        if tags & {"time", "window", "merge"}:
            return "Temporal/Window"
        if tags & {"math", "normalize", "softmax"}:
            return "Math/Statistics"
        if tags & {"text", "regex", "extract"}:
            return "Text/Regex"
        if tags & {"projection", "flatten", "dedup", "cleaning"}:
            return "Projection/Reshape"
        if tags & {"group", "count"}:
            return "Grouping/Frequency"
        if tags & {"ranking", "sort", "selection"}:
            return "Ranking/Selection"
        if tags & {"filter"}:
            return "Filter/Predicate"
        if tags & {"aggregation", "metric", "extreme", "percentile", "avg", "sum", "median", "variance"}:
            return "Summary/Aggregation"
        return "Other"

    if tags & {"finance", "loan"} or name == "Debt_Interest_Analyzer":
        return "Financial Analysis"
    if tags & {"execution", "python", "sql", "code"} or name in {"Python_Executor", "SQL_Executor"}:
        return "Execution/Coding"
    if tags & {"memory", "compression", "system"} or name in {"Fold_Memory", "Finish"}:
        return "Memory/System"
    if tags & {"audio", "speaker", "speech", "asr"} or name in {"Audio_Event_Detect", "Speaker_Diarization", "asr"}:
        return "Audio/Speech"
    if tags & {"visual", "video", "image", "ocr", "emotion", "action", "crossmodal", "tracking"} or name in {
        "Inspect_Frame",
        "OCR_Extract",
        "Action_Recognition",
        "Emotion_Analysis",
        "Crop",
        "Zoom_in",
        "Video_Clip_QA",
        "Entity_Track",
        "Cross_Modal_Verify",
    }:
        return "Visual/Video"
    if tags & {"retrieval", "search", "keyword", "semantic", "metadata", "rewrite", "verification", "rerank", "graph", "segment", "summary", "web", "aggregation", "composite"} or name in {
        "Tool_Search",
        "Query_Rewrite",
        "Graph_Search",
        "Segment_Retrieve",
        "Keyword_Retrieve",
        "Semantic_Retrieve",
        "Clean_Results",
        "Video_Summary",
        "Aggregation",
        "evidence_verify",
        "Web_Search",
    }:
        return "Retrieval/Search"
    return "Other"


def build_function_counts_by_granularity(
    records: Mapping[str, ToolRecord],
) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {grain_key: Counter() for grain_key in GRAIN_ORDER}
    for record in records.values():
        grain_key = record.tool_type if record.tool_type in GRAIN_ORDER else "biz"
        counters.setdefault(grain_key, Counter())
        counters[grain_key][infer_function_category(record)] += 1
    return {
        grain_key: _order_function_counts(counter, grain_key=grain_key)
        for grain_key, counter in counters.items()
    }


def build_summary_payload(enabled_records: Mapping[str, ToolRecord]) -> Dict[str, Any]:
    return build_summary_payload_from_records(
        all_records=enabled_records,
        enabled_records=enabled_records,
    )


def build_summary_payload_from_records(
    *,
    all_records: Mapping[str, ToolRecord],
    enabled_records: Mapping[str, ToolRecord],
) -> Dict[str, Any]:
    all_grain_counts = Counter(record.tool_type for record in all_records.values())
    grain_counts = Counter(record.tool_type for record in enabled_records.values())
    function_counts_by_granularity = build_function_counts_by_granularity(enabled_records)
    function_counts = Counter()
    for nested_counts in function_counts_by_granularity.values():
        function_counts.update(nested_counts)
    by_grain_counts = {
        key: {
            "all_existing": int(all_grain_counts.get(key, 0)),
            "enabled": int(grain_counts.get(key, 0)),
            "display_name": GRAIN_LABELS.get(key, key),
        }
        for key in GRAIN_ORDER
    }
    return {
        "tool_set": "mvtl_enabled_tools",
        "all_existing_total": len(all_records),
        "enabled_total": len(enabled_records),
        "granularity_counts": {key: int(grain_counts.get(key, 0)) for key in GRAIN_ORDER},
        "granularity_nested_counts": by_grain_counts,
        "function_counts_by_granularity": function_counts_by_granularity,
        "function_counts": {
            key: function_counts.get(key, 0)
            for key in FUNCTION_ORDER
            if function_counts.get(key, 0) > 0
        },
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> Tuple[float, float, float, float]:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    return (red, green, blue, alpha)


def plot_meta_function_panel(ax: plt.Axes, payload: Mapping[str, Any]) -> None:
    nested_counts = payload.get("function_counts_by_granularity", {})
    function_counts = nested_counts.get("meta", {}) if isinstance(nested_counts, dict) else {}
    items = [
        (name, int(count))
        for name, count in function_counts.items()
        if int(count) > 0
    ]
    items.sort(key=lambda item: (-item[1], item[0]))
    labels = [name for name, _ in items]
    values = [count for _, count in items]
    colors = [FUNCTION_COLORS.get(label, FUNCTION_COLORS["Other"]) for label in labels]
    y_positions = list(range(len(labels)))

    ax.set_facecolor(AXIS_BACKGROUND)
    bars = ax.barh(
        y_positions,
        values,
        color=colors,
        alpha=BAR_ALPHA,
        edgecolor=AXIS_BACKGROUND,
        linewidth=0.8,
        height=0.7,
    )
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title("Meta Tool Inventory by Function", pad=14, fontsize=33.0)
    ax.set_xlabel("Number of Tools", fontsize=28.0)
    ax.tick_params(axis="both", labelsize=26.0)
    max_value = max(values) if values else 1
    ax.set_xlim(0, max_value * 1.18 if max_value > 0 else 1)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(0.4, max_value * 0.012),
            bar.get_y() + bar.get_height() / 2.0,
            str(int(value)),
            va="center",
            ha="left",
            fontsize=22.0,
            color=TEXT_COLOR,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_outputs(
    fig: plt.Figure,
    payload: Mapping[str, Any],
    output_dir: Path,
    figure_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_stem = output_dir / figure_prefix
    fig.savefig(str(figure_stem.with_suffix(".png")), bbox_inches="tight")
    fig.savefig(str(figure_stem.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "tool_inventory_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(payload: Mapping[str, Any], output_dir: Path) -> None:
    print("Tool inventory chart generated.")
    print(f"- enabled_total={payload.get('enabled_total', 0)}")
    print(f"- all_existing_total={payload.get('all_existing_total', 0)}")
    nested_counts = payload.get("function_counts_by_granularity", {})
    if isinstance(nested_counts, dict):
        for grain_key in GRAIN_ORDER:
            grain_counts = nested_counts.get(grain_key, {})
            if not isinstance(grain_counts, dict):
                continue
            print(f"- {GRAIN_LABELS.get(grain_key, grain_key)} function counts:")
            for function_name, count in grain_counts.items():
                print(f"  - {function_name}: {int(count)}")
    print(f"Output dir: {output_dir}")


def main() -> None:
    args = parse_args()
    plot_config = load_plot_config(args)
    output_dir = plot_config.output_dir
    project_root = tool_star_root()

    set_plot_style()
    all_records = discover_tool_records(project_root)
    enabled_records = build_enabled_records(project_root)
    payload = build_summary_payload_from_records(
        all_records=all_records,
        enabled_records=enabled_records,
    )

    fig = plt.figure(figsize=(11.8, 6.6))
    fig.patch.set_facecolor(PAPER_BACKGROUND)
    grid = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(grid[0, 0])

    plot_meta_function_panel(ax, payload)

    fig.subplots_adjust(left=0.22, right=0.97, top=0.90, bottom=0.15)

    save_outputs(fig, payload, output_dir, plot_config.figure_prefix)
    print_summary(payload, output_dir)


if __name__ == "__main__":
    main()
