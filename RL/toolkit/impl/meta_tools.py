"""Auto-generated meta tools for fine-grained operations."""
import copy
import logging
import math
import random
import re
from typing import Any, Dict, List, Optional

from pydantic import Field

from toolkit.base import BaseTool

logger = logging.getLogger(__name__)


# --- Helper utilities --------------------------------------------------------
def _get_numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _get_field(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return default


def _parse_time(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        if isinstance(val, str) and ":" in val:
            parts = [float(p) for p in val.split(":")]
            if len(parts) == 3:
                h, m, s = parts
            elif len(parts) == 2:
                h, m = parts
                s = 0
            else:
                return float(val)
            return h * 3600 + m * 60 + s
        return float(val)
    except Exception:
        return 0.0


def _build_params(required: Dict[str, Dict[str, Any]], optional: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    properties = copy.deepcopy(required)
    optional = optional or {}
    properties.update(copy.deepcopy(optional))
    return {
        "type": "object",
        "properties": properties,
        "required": list(required.keys()),
    }


# Common parameter pieces
PROP_DATA = {"type": "array", "description": "List of items (dict or scalar)"}
PROP_FIELD = {"type": "string", "description": "Target field name"}
PROP_VALUE = {"type": "string", "description": "Comparison value"}
PROP_NUM = {"type": "number", "description": "Numeric value"}
PROP_K = {"type": "integer", "description": "Top/Bottom K", "default": 5}
PROP_WINDOW = {"type": "integer", "description": "Window size", "default": 3}
PROP_TIME_RANGE = {"type": "object", "description": "Time range with start/end", "properties": {"start": {"type": "number"}, "end": {"type": "number"}}, "required": ["start", "end"]}


# --- Handlers ----------------------------------------------------------------
def h_sum_field(data: List[dict], field: str, **_) -> float:
    return sum(_get_numeric(_get_field(i, field), 0.0) for i in data)


def h_avg_field(data: List[dict], field: str, **_) -> float:
    nums = [_get_numeric(_get_field(i, field), 0.0) for i in data]
    return sum(nums) / len(nums) if nums else 0.0


def h_median_field(data: List[dict], field: str, **_) -> float:
    nums = sorted(_get_numeric(_get_field(i, field), 0.0) for i in data)
    n = len(nums)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2


def h_variance_field(data: List[dict], field: str, **_) -> float:
    nums = [_get_numeric(_get_field(i, field), 0.0) for i in data]
    if not nums:
        return 0.0
    mean = sum(nums) / len(nums)
    return sum((x - mean) ** 2 for x in nums) / len(nums)


def h_stddev_field(data: List[dict], field: str, **_) -> float:
    return math.sqrt(h_variance_field(data, field))


def _percentile(data: List[dict], field: str, pct: float) -> float:
    nums = sorted(_get_numeric(_get_field(i, field), 0.0) for i in data)
    if not nums:
        return 0.0
    k = (len(nums) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return nums[int(k)]
    return nums[f] * (c - k) + nums[c] * (k - f)


def h_topk_by_field(data: List[dict], field: str, k: int = 5, **_) -> List[dict]:
    return sorted(data, key=lambda x: _get_numeric(_get_field(x, field), 0.0), reverse=True)[:k]


def h_bottomk_by_field(data: List[dict], field: str, k: int = 5, **_) -> List[dict]:
    return sorted(data, key=lambda x: _get_numeric(_get_field(x, field), 0.0))[:k]


def h_sort_by_field(data: List[dict], field: str, asc: bool = True, **_) -> List[dict]:
    return sorted(data, key=lambda x: _get_numeric(_get_field(x, field), 0.0), reverse=not asc)


def h_filter_equals(data: List[dict], field: str, value: Any, **_) -> List[dict]:
    return [i for i in data if _get_field(i, field) == value]


def h_filter_notequals(data: List[dict], field: str, value: Any, **_) -> List[dict]:
    return [i for i in data if _get_field(i, field) != value]


def h_filter_gt(data: List[dict], field: str, value: Any, **_) -> List[dict]:
    v = _get_numeric(value, 0.0)
    return [i for i in data if _get_numeric(_get_field(i, field), 0.0) > v]


def h_filter_lt(data: List[dict], field: str, value: Any, **_) -> List[dict]:
    v = _get_numeric(value, 0.0)
    return [i for i in data if _get_numeric(_get_field(i, field), 0.0) < v]


def h_filter_between(data: List[dict], field: str, low: Any, high: Any, **_) -> List[dict]:
    lo = _get_numeric(low, -math.inf)
    hi = _get_numeric(high, math.inf)
    return [i for i in data if lo <= _get_numeric(_get_field(i, field), 0.0) <= hi]


def h_filter_in(data: List[dict], field: str, values: List[Any], **_) -> List[dict]:
    values = values or []
    return [i for i in data if _get_field(i, field) in values]


def h_filter_not_in(data: List[dict], field: str, values: List[Any], **_) -> List[dict]:
    values = values or []
    return [i for i in data if _get_field(i, field) not in values]


def h_filter_contains(data: List[dict], field: str, substring: str, **_) -> List[dict]:
    return [i for i in data if substring.lower() in str(_get_field(i, field, "")).lower()]


def h_filter_not_contains(data: List[dict], field: str, substring: str, **_) -> List[dict]:
    return [i for i in data if substring.lower() not in str(_get_field(i, field, "")).lower()]


def h_threshold_filter(data: List[dict], field: str, threshold: float, **_) -> List[dict]:
    return [i for i in data if _get_numeric(_get_field(i, field), 0.0) >= threshold]


def h_dedup_by_field(data: List[dict], field: str, **_) -> List[dict]:
    seen = set()
    out = []
    for item in data:
        val = _get_field(item, field)
        if val not in seen:
            seen.add(val)
            out.append(item)
    return out


def h_select_fields(data: List[dict], fields: List[str], **_) -> List[dict]:
    return [{k: v for k, v in item.items() if k in fields} for item in data]


def h_rename_field(data: List[dict], field: str, new_name: str, **_) -> List[dict]:
    out = []
    for item in data:
        new_item = dict(item)
        if field in new_item:
            new_item[new_name] = new_item.pop(field)
        out.append(new_item)
    return out


def h_drop_fields(data: List[dict], fields: List[str], **_) -> List[dict]:
    return [{k: v for k, v in item.items() if k not in fields} for item in data]


def h_fillna_field(data: List[dict], field: str, fill_value: Any, **_) -> List[dict]:
    out = []
    for item in data:
        new_item = dict(item)
        if new_item.get(field) in (None, "", float("nan")):
            new_item[field] = fill_value
        out.append(new_item)
    return out


def h_add_constant(data: List[dict], field: str, value: float, **_) -> List[dict]:
    out = []
    for item in data:
        new_item = dict(item)
        new_item[field] = _get_numeric(_get_field(item, field), 0.0) + _get_numeric(value, 0.0)
        out.append(new_item)
    return out


def h_multiply_field(data: List[dict], field: str, factor: float, **_) -> List[dict]:
    out = []
    for item in data:
        new_item = dict(item)
        new_item[field] = _get_numeric(_get_field(item, field), 0.0) * _get_numeric(factor, 1.0)
        out.append(new_item)
    return out


def h_divide_field(data: List[dict], field: str, divisor: float, **_) -> List[dict]:
    divisor = _get_numeric(divisor, 1.0) or 1.0
    out = []
    for item in data:
        new_item = dict(item)
        new_item[field] = _get_numeric(_get_field(item, field), 0.0) / divisor
        out.append(new_item)
    return out


def h_clip_field(data: List[dict], field: str, low: float, high: float, **_) -> List[dict]:
    lo = _get_numeric(low, -math.inf)
    hi = _get_numeric(high, math.inf)
    out = []
    for item in data:
        val = _get_numeric(_get_field(item, field), 0.0)
        new_item = dict(item)
        new_item[field] = min(max(val, lo), hi)
        out.append(new_item)
    return out


def h_minmax(data: List[dict], field: str, **_) -> List[dict]:
    values = [_get_numeric(_get_field(i, field), 0.0) for i in data]
    if not values:
        return data
    mn, mx = min(values), max(values)
    span = (mx - mn) or 1.0
    out = []
    for item in data:
        new_item = dict(item)
        new_item[field] = (_get_numeric(_get_field(item, field), mn) - mn) / span
        out.append(new_item)
    return out


def h_zscore(data: List[dict], field: str, **_) -> List[dict]:
    values = [_get_numeric(_get_field(i, field), 0.0) for i in data]
    if not values:
        return data
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)) or 1.0
    out = []
    for item in data:
        new_item = dict(item)
        new_item[field] = (_get_numeric(_get_field(item, field), mean) - mean) / std
        out.append(new_item)
    return out


def h_softmax(data: List[dict], field: str, **_) -> List[dict]:
    values = [_get_numeric(_get_field(i, field), 0.0) for i in data]
    if not values:
        return data
    exp_vals = [math.exp(v) for v in values]
    total = sum(exp_vals) or 1.0
    out = []
    for item, ev in zip(data, exp_vals):
        new_item = dict(item)
        new_item[field] = ev / total
        out.append(new_item)
    return out


def h_argmax_field(data: List[dict], field: str, **_) -> Optional[dict]:
    return max(data, key=lambda x: _get_numeric(_get_field(x, field), -math.inf), default=None)


def h_argmin_field(data: List[dict], field: str, **_) -> Optional[dict]:
    return min(data, key=lambda x: _get_numeric(_get_field(x, field), math.inf), default=None)


def h_count_by_field(data: List[dict], field: str, **_) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in data:
        val = str(_get_field(item, field, ""))
        counts[val] = counts.get(val, 0) + 1
    return counts


def h_group_sum(data: List[dict], field: str, target: str, **_) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in data:
        key = str(_get_field(item, field, ""))
        out[key] = out.get(key, 0.0) + _get_numeric(_get_field(item, target), 0.0)
    return out


def h_group_avg(data: List[dict], field: str, target: str, **_) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for item in data:
        key = str(_get_field(item, field, ""))
        sums[key] = sums.get(key, 0.0) + _get_numeric(_get_field(item, target), 0.0)
        counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def h_group_extreme(data: List[dict], field: str, target: str, mode: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in data:
        key = str(_get_field(item, field, ""))
        val = _get_numeric(_get_field(item, target), 0.0)
        if key not in out:
            out[key] = val
        else:
            if mode == "max":
                out[key] = max(out[key], val)
            else:
                out[key] = min(out[key], val)
    return out


def h_merge_by_key(data: List[dict], field: str, **_) -> Dict[str, List[dict]]:
    merged: Dict[str, List[dict]] = {}
    for item in data:
        key = str(_get_field(item, field, ""))
        merged.setdefault(key, []).append(item)
    return merged


def h_flatten_list_field(data: List[dict], field: str, **_) -> List[Any]:
    out: List[Any] = []
    for item in data:
        values = _get_field(item, field, [])
        if isinstance(values, list):
            out.extend(values)
    return out


def h_unique_values(data: List[dict], field: str, **_) -> List[Any]:
    seen = set()
    for item in data:
        seen.add(_get_field(item, field))
    return list(seen)


def h_value_frequencies(data: List[dict], field: str, **_) -> Dict[Any, int]:
    freq: Dict[Any, int] = {}
    for item in data:
        val = _get_field(item, field)
        freq[val] = freq.get(val, 0) + 1
    return freq


def h_regex_filter(data: List[dict], field: str, pattern: str, **_) -> List[dict]:
    regex = re.compile(pattern)
    return [i for i in data if regex.search(str(_get_field(i, field, "")))]


def h_regex_extract(data: List[dict], field: str, pattern: str, **_) -> List[Dict[str, Any]]:
    regex = re.compile(pattern)
    out = []
    for item in data:
        matches = regex.findall(str(_get_field(item, field, "")))
        out.append({"item": item, "matches": matches})
    return out


def h_keyword_filter(data: List[dict], field: str, keyword: str, **_) -> List[dict]:
    return [i for i in data if keyword.lower() in str(_get_field(i, field, "")).lower()]


def h_truncate_text(data: List[dict], field: str, max_len: int, **_) -> List[dict]:
    out = []
    for item in data:
        new_item = dict(item)
        text = str(_get_field(item, field, ""))
        new_item[field] = text[:max_len]
        out.append(new_item)
    return out


def h_lower(data: List[dict], field: str, **_) -> List[dict]:
    return [{**item, field: str(_get_field(item, field, "")).lower()} for item in data]


def h_upper(data: List[dict], field: str, **_) -> List[dict]:
    return [{**item, field: str(_get_field(item, field, "")).upper()} for item in data]


def h_strip(data: List[dict], field: str, **_) -> List[dict]:
    return [{**item, field: str(_get_field(item, field, "")).strip()} for item in data]


def h_concat_fields(data: List[dict], fields: List[str], target: str, sep: str = " ", **_) -> List[dict]:
    out = []
    for item in data:
        parts = [str(item.get(f, "")) for f in fields]
        new_item = dict(item)
        new_item[target] = sep.join(parts)
        out.append(new_item)
    return out


def h_join_with_delimiter(values: List[str], delimiter: str = ",", **_) -> str:
    return delimiter.join(values or [])


def h_time_range_length(time_range: Dict[str, Any], **_) -> float:
    return max(0.0, _parse_time(time_range.get("end")) - _parse_time(time_range.get("start")))


def h_time_overlap(range_a: Dict[str, Any], range_b: Dict[str, Any], **_) -> float:
    start = max(_parse_time(range_a.get("start")), _parse_time(range_b.get("start")))
    end = min(_parse_time(range_a.get("end")), _parse_time(range_b.get("end")))
    return max(0.0, end - start)


def h_merge_touching(ranges: List[Dict[str, Any]], tolerance: float = 0.0, **_) -> List[Dict[str, Any]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: _parse_time(r.get("start")))
    merged = [dict(sorted_ranges[0])]
    for r in sorted_ranges[1:]:
        if _parse_time(r.get("start")) <= _parse_time(merged[-1].get("end")) + tolerance:
            merged[-1]["end"] = max(_parse_time(merged[-1].get("end")), _parse_time(r.get("end")))
        else:
            merged.append(dict(r))
    return merged


def h_align_time_buckets(ranges: List[Dict[str, Any]], bucket: float, **_) -> List[Dict[str, Any]]:
    bucket = bucket or 1.0
    aligned = []
    for r in ranges:
        start = math.floor(_parse_time(r.get("start")) / bucket) * bucket
        end = math.ceil(_parse_time(r.get("end")) / bucket) * bucket
        aligned.append({"start": start, "end": end})
    return aligned


def h_clip_time_range(time_range: Dict[str, Any], clip_start: float, clip_end: float, **_) -> Dict[str, Any]:
    return {"start": max(_parse_time(time_range.get("start")), clip_start), "end": min(_parse_time(time_range.get("end")), clip_end)}


def h_expand_time_range(time_range: Dict[str, Any], delta: float, **_) -> Dict[str, Any]:
    return {"start": _parse_time(time_range.get("start")) - delta, "end": _parse_time(time_range.get("end")) + delta}


def h_shift_time_range(time_range: Dict[str, Any], offset: float, **_) -> Dict[str, Any]:
    return {"start": _parse_time(time_range.get("start")) + offset, "end": _parse_time(time_range.get("end")) + offset}


def h_sort_time_ranges(ranges: List[Dict[str, Any]], **_) -> List[Dict[str, Any]]:
    return sorted(ranges, key=lambda r: _parse_time(r.get("start")))


def h_window_sliding(data: List[Any], window: int = 3, **_) -> List[List[Any]]:
    window = max(1, window)
    return [data[i:i + window] for i in range(max(0, len(data) - window + 1))]


def _normalize_relation_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _coerce_graph_adjacency(graph: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(graph, dict):
        return {}

    relations = graph.get("relations")
    if isinstance(relations, list):
        adjacency: Dict[str, List[Dict[str, Any]]] = {}
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            if not source or not target:
                continue
            payload = {
                "neighbor": target,
                "relation_type": str(
                    rel.get("relation_type") or rel.get("relation") or rel.get("type") or "RELATED_TO"
                ).strip(),
                "description": str(rel.get("description", "") or "").strip(),
                "weight": _get_numeric(rel.get("weight"), 1.0),
                "source_segments": list(rel.get("source_segments") or []),
            }
            adjacency.setdefault(source, []).append(payload)
            adjacency.setdefault(target, []).append(
                {
                    **payload,
                    "neighbor": source,
                }
            )
        return adjacency

    adjacency = {}
    for node, neighbors in graph.items():
        normalized_node = str(node or "").strip()
        if not normalized_node:
            continue
        adjacency[normalized_node] = []
        if isinstance(neighbors, dict):
            neighbor_iterable = neighbors.items()
        else:
            neighbor_iterable = [(neighbor, {}) for neighbor in (neighbors or [])]
        for neighbor, meta in neighbor_iterable:
            normalized_neighbor = str(neighbor or "").strip()
            if not normalized_neighbor:
                continue
            payload = meta if isinstance(meta, dict) else {}
            adjacency[normalized_node].append(
                {
                    "neighbor": normalized_neighbor,
                    "relation_type": str(payload.get("relation_type") or "RELATED_TO").strip(),
                    "description": str(payload.get("description", "") or "").strip(),
                    "weight": _get_numeric(payload.get("weight"), 1.0),
                    "source_segments": list(payload.get("source_segments") or []),
                }
            )
    return adjacency


def h_neighbor_list(
    graph: Dict[str, Any],
    node: str,
    relation: Optional[str] = None,
    include_metadata: bool = False,
    **_,
) -> List[Any]:
    adjacency = _coerce_graph_adjacency(graph)
    relation_filter = _normalize_relation_text(relation)
    neighbors: List[Any] = []
    for edge in adjacency.get(str(node or "").strip(), []):
        relation_text = " ".join(
            [
                _normalize_relation_text(edge.get("relation_type")),
                _normalize_relation_text(edge.get("description")),
            ]
        )
        if relation_filter and relation_filter not in relation_text:
            continue
        if include_metadata:
            neighbors.append(edge)
        else:
            neighbors.append(str(edge.get("neighbor", "")).strip())
    return neighbors


def h_degree_count(graph: Dict[str, Any], node: str, relation: Optional[str] = None, **_) -> int:
    return len(h_neighbor_list(graph, node, relation=relation, include_metadata=False))


def h_edge_filter(
    graph: Dict[str, Any],
    allowed: List[str],
    relation: Optional[str] = None,
    **_,
) -> Dict[str, List[str]]:
    allowed_set = {str(item).strip() for item in (allowed or []) if str(item).strip()}
    relation_filter = _normalize_relation_text(relation)
    adjacency = _coerce_graph_adjacency(graph)
    filtered: Dict[str, List[str]] = {}
    for node, edges in adjacency.items():
        kept_neighbors: List[str] = []
        for edge in edges:
            neighbor = str(edge.get("neighbor", "")).strip()
            relation_text = " ".join(
                [
                    _normalize_relation_text(edge.get("relation_type")),
                    _normalize_relation_text(edge.get("description")),
                ]
            )
            if allowed_set and neighbor not in allowed_set:
                continue
            if relation_filter and relation_filter not in relation_text:
                continue
            kept_neighbors.append(neighbor)
        filtered[node] = kept_neighbors
    return filtered


def h_path_length_filter(
    paths: List[Any],
    max_len: int,
    min_len: int = 1,
    **_,
) -> List[Any]:
    kept = []
    for path in paths or []:
        if isinstance(path, dict):
            path_nodes = path.get("path") or []
        else:
            path_nodes = path or []
        path_len = len(path_nodes)
        if int(min_len) <= path_len <= int(max_len):
            kept.append(path)
    return kept


def h_score_threshold_rank(data: List[dict], field: str, threshold: float, **_) -> List[dict]:
    return [i for i in data if _get_numeric(_get_field(i, field), 0.0) >= threshold]


def h_topk_score(data: List[dict], k: int = 5, **_) -> List[dict]:
    return sorted(data, key=lambda x: _get_numeric(_get_field(x, "score"), 0.0), reverse=True)[:k]


def h_bottomk_score(data: List[dict], k: int = 5, **_) -> List[dict]:
    return sorted(data, key=lambda x: _get_numeric(_get_field(x, "score"), 0.0))[:k]


def h_rerank_by_score(data: List[dict], weight: float = 1.0, **_) -> List[dict]:
    weight = _get_numeric(weight, 1.0)
    return sorted(data, key=lambda x: _get_numeric(_get_field(x, "score"), 0.0) * weight, reverse=True)


def h_weighted_sum_two_fields(data: List[dict], field_a: str, field_b: str, weight_a: float = 0.5, weight_b: float = 0.5, **_) -> List[dict]:
    out = []
    for item in data:
        new_item = dict(item)
        new_item["weighted_sum"] = _get_numeric(_get_field(item, field_a), 0.0) * weight_a + _get_numeric(_get_field(item, field_b), 0.0) * weight_b
        out.append(new_item)
    return out


def h_weighted_avg(data: List[dict], field: str, weight_field: str, **_) -> float:
    total_weight = sum(_get_numeric(_get_field(i, weight_field), 0.0) for i in data)
    if total_weight == 0:
        return 0.0
    return sum(_get_numeric(_get_field(i, field), 0.0) * _get_numeric(_get_field(i, weight_field), 0.0) for i in data) / total_weight


def h_cumsum_field(data: List[dict], field: str, **_) -> List[dict]:
    total = 0.0
    out = []
    for item in data:
        total += _get_numeric(_get_field(item, field), 0.0)
        new_item = dict(item)
        new_item["cumsum"] = total
        out.append(new_item)
    return out


def h_moving_average(data: List[float], window: int = 3, **_) -> List[float]:
    window = max(1, window)
    if len(data) < window:
        return []
    return [sum(data[i:i + window]) / window for i in range(len(data) - window + 1)]


def h_rolling_max(data: List[float], window: int = 3, **_) -> List[float]:
    window = max(1, window)
    if len(data) < window:
        return []
    return [max(data[i:i + window]) for i in range(len(data) - window + 1)]


def h_rolling_min(data: List[float], window: int = 3, **_) -> List[float]:
    window = max(1, window)
    if len(data) < window:
        return []
    return [min(data[i:i + window]) for i in range(len(data) - window + 1)]


def h_rolling_median(data: List[float], window: int = 3, **_) -> List[float]:
    window = max(1, window)
    if len(data) < window:
        return []
    out = []
    for i in range(len(data) - window + 1):
        slice_vals = sorted(data[i:i + window])
        mid = window // 2
        if window % 2:
            out.append(slice_vals[mid])
        else:
            out.append((slice_vals[mid - 1] + slice_vals[mid]) / 2)
    return out


def h_ratio_fields(data: List[dict], numerator: str, denominator: str, **_) -> List[dict]:
    out = []
    for item in data:
        denom = _get_numeric(_get_field(item, denominator), 1.0) or 1.0
        num = _get_numeric(_get_field(item, numerator), 0.0)
        new_item = dict(item)
        new_item["ratio"] = num / denom
        out.append(new_item)
    return out


def h_percentage_change(data: List[float], **_) -> List[float]:
    if len(data) < 2:
        return []
    out = []
    for prev, curr in zip(data, data[1:]):
        prev_val = _get_numeric(prev, 0.0) or 1e-9
        curr_val = _get_numeric(curr, 0.0)
        out.append((curr_val - prev_val) / prev_val)
    return out


def h_boolean_and(flags: List[bool], **_) -> bool:
    return all(bool(f) for f in flags)


def h_boolean_or(flags: List[bool], **_) -> bool:
    return any(bool(f) for f in flags)


def h_boolean_not(flag: bool, **_) -> bool:
    return not bool(flag)


def h_count_true(flags: List[bool], **_) -> int:
    return sum(1 for f in flags if f)


def h_any_true(flags: List[bool], **_) -> bool:
    return any(flags)


def h_all_true(flags: List[bool], **_) -> bool:
    return all(flags)


def h_sample_n(data: List[Any], n: int, **_) -> List[Any]:
    return random.sample(data, min(n, len(data))) if data else []


def h_shuffle_data(data: List[Any], **_) -> List[Any]:
    out = list(data)
    random.shuffle(out)
    return out


def h_take_first_n(data: List[Any], n: int, **_) -> List[Any]:
    return data[:n]


def h_take_last_n(data: List[Any], n: int, **_) -> List[Any]:
    return data[-n:] if n else []


def h_reverse_list(data: List[Any], **_) -> List[Any]:
    return list(reversed(data))


def h_extract_field_list(data: List[dict], field: str, **_) -> List[Any]:
    return [_get_field(item, field) for item in data]


def h_build_index(data: List[dict], field: str, **_) -> Dict[Any, dict]:
    return {_get_field(item, field): item for item in data}


def h_map_field_lookup(data: List[dict], field: str, lookup: Dict[Any, Any], **_) -> List[dict]:
    out = []
    for item in data:
        new_item = dict(item)
        new_item[field] = lookup.get(_get_field(item, field), _get_field(item, field))
        out.append(new_item)
    return out


def h_filter_empty(data: List[Any], **_) -> List[Any]:
    return [d for d in data if d]


def h_ensure_sorted(data: List[Any], **_) -> List[Any]:
    try:
        return sorted(data)
    except Exception:
        return data


# --- Finance / numeric reasoning --------------------------------------------
def h_simple_interest(principal: float, rate: float, periods: float, **_) -> Dict[str, float]:
    p = _get_numeric(principal, 0.0)
    r = _get_numeric(rate, 0.0)
    n = _get_numeric(periods, 0.0)
    interest = p * r * n
    return {"principal": p, "interest": interest, "total": p + interest}


def h_compound_interest(principal: float, rate: float, periods: float, freq: float = 12.0, **_) -> Dict[str, float]:
    p = _get_numeric(principal, 0.0)
    r = _get_numeric(rate, 0.0)
    n = _get_numeric(periods, 0.0)
    f = max(_get_numeric(freq, 12.0), 1.0)
    total = p * (1 + r / f) ** (f * n)
    return {"principal": p, "rate": r, "periods": n, "freq": f, "total": total, "interest": total - p}


def h_annuity_payment(principal: float, rate: float, periods: float, freq: float = 12.0, **_) -> Dict[str, float]:
    p = _get_numeric(principal, 0.0)
    r_annual = _get_numeric(rate, 0.0)
    f = max(_get_numeric(freq, 12.0), 1.0)
    n = max(int(_get_numeric(periods, 0.0) * f), 1)
    r = r_annual / f
    if r == 0:
        payment = p / n
    else:
        payment = p * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return {"payment_per_period": payment, "periods": n, "rate_per_period": r}


def h_amortization_schedule(principal: float, rate: float, periods: float, freq: float = 12.0, payment: Optional[float] = None, **_) -> List[Dict[str, float]]:
    p = _get_numeric(principal, 0.0)
    r_annual = _get_numeric(rate, 0.0)
    f = max(_get_numeric(freq, 12.0), 1.0)
    n = max(int(_get_numeric(periods, 0.0) * f), 1)
    r = r_annual / f
    if payment is None:
        payment = h_annuity_payment(p, r_annual, periods, f)["payment_per_period"]
    bal = p
    schedule = []
    for k in range(1, n + 1):
        interest = bal * r
        principal_paid = payment - interest
        bal = max(0.0, bal - principal_paid)
        schedule.append({"period": k, "payment": payment, "interest": interest, "principal": principal_paid, "balance": bal})
        if bal <= 0:
            break
    return schedule


def h_total_interest(schedule: List[Dict[str, float]], **_) -> float:
    return sum(_get_numeric(s.get("interest"), 0.0) for s in schedule)


def h_balance_after(schedule: List[Dict[str, float]], periods: int, **_) -> float:
    if not schedule:
        return 0.0
    idx = min(max(int(periods), 0), len(schedule)) - 1
    if idx < 0:
        return _get_numeric(schedule[0].get("balance"), 0.0)
    return _get_numeric(schedule[idx].get("balance"), 0.0)


def h_compare_schedules(schedule_a: List[Dict[str, float]], schedule_b: List[Dict[str, float]], **_) -> Dict[str, Any]:
    return {
        "total_interest_a": h_total_interest(schedule_a),
        "total_interest_b": h_total_interest(schedule_b),
        "len_a": len(schedule_a),
        "len_b": len(schedule_b),
    }


def _build_growth_schedule(
    start_value: float,
    rate: float,
    periods: float,
    *,
    freq: float = 12.0,
    step_adjustment: float = 0.0,
) -> List[Dict[str, float]]:
    base_value = _get_numeric(start_value, 0.0)
    total_steps = max(int(round(_get_numeric(periods, 0.0) * max(_get_numeric(freq, 12.0), 1.0))), 0)
    step_rate = _get_numeric(rate, 0.0) / max(_get_numeric(freq, 12.0), 1.0)
    balance = base_value
    schedule: List[Dict[str, float]] = []
    for step in range(1, total_steps + 1):
        growth_amount = balance * step_rate
        balance = balance + growth_amount - _get_numeric(step_adjustment, 0.0)
        schedule.append(
            {
                "step": step,
                "starting_value": balance - growth_amount + _get_numeric(step_adjustment, 0.0),
                "growth_amount": growth_amount,
                "step_adjustment": _get_numeric(step_adjustment, 0.0),
                "ending_value": balance,
            }
        )
    return schedule


def h_compound_growth(
    start_value: float,
    rate: float,
    periods: float,
    freq: float = 12.0,
    step_adjustment: float = 0.0,
    return_schedule: bool = False,
    **_,
) -> Dict[str, Any]:
    growth = h_compound_interest(start_value, rate, periods, freq=freq)
    base_value = _get_numeric(growth.get("principal"), 0.0)
    schedule = _build_growth_schedule(
        start_value,
        rate,
        periods,
        freq=freq,
        step_adjustment=step_adjustment,
    )
    final_value = _get_numeric(schedule[-1]["ending_value"], base_value) if schedule else _get_numeric(growth.get("total"), base_value)
    return {
        "start_value": base_value,
        "rate": _get_numeric(growth.get("rate"), 0.0),
        "periods": _get_numeric(growth.get("periods"), 0.0),
        "freq": _get_numeric(growth.get("freq"), 1.0),
        "total_steps": len(schedule),
        "step_adjustment": _get_numeric(step_adjustment, 0.0),
        "final_value": final_value,
        "delta": final_value - base_value,
        **({"schedule": schedule} if return_schedule else {}),
    }


def h_fixed_step_amount(
    start_value: float,
    rate: float,
    periods: float,
    freq: float = 12.0,
    target_end_value: float = 0.0,
    return_schedule: bool = False,
    **_,
) -> Dict[str, Any]:
    payment = h_annuity_payment(
        max(_get_numeric(start_value, 0.0) - _get_numeric(target_end_value, 0.0), 0.0),
        rate,
        periods,
        freq=freq,
    )
    step_amount = _get_numeric(payment.get("payment_per_period"), 0.0)
    schedule = _build_growth_schedule(
        start_value,
        rate,
        periods,
        freq=freq,
        step_adjustment=step_amount,
    )
    return {
        "start_value": _get_numeric(start_value, 0.0),
        "target_end_value": _get_numeric(target_end_value, 0.0),
        "step_amount": step_amount,
        "total_steps": _get_numeric(payment.get("periods"), 0.0),
        "rate_per_step": _get_numeric(payment.get("rate_per_period"), 0.0),
        "projected_end_value": _get_numeric(schedule[-1]["ending_value"], _get_numeric(start_value, 0.0)) if schedule else _get_numeric(start_value, 0.0),
        **({"schedule": schedule} if return_schedule else {}),
    }


def _sequence_numeric_values(sequence: List[Dict[str, Any]], field: str) -> List[float]:
    return [
        _get_numeric(_get_field(item, field), 0.0)
        for item in (sequence or [])
    ]


def h_sequence_field_total(
    sequence: List[Dict[str, Any]],
    field: str,
    operation: str = "sum",
    **_,
) -> Any:
    values = _sequence_numeric_values(sequence, field)
    if not values:
        return 0.0 if operation != "summary" else {
            "field": field,
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "delta": 0.0,
        }
    normalized_operation = str(operation or "sum").strip().lower()
    if normalized_operation == "count":
        return len(values)
    if normalized_operation == "mean":
        return sum(values) / len(values)
    if normalized_operation == "min":
        return min(values)
    if normalized_operation == "max":
        return max(values)
    if normalized_operation == "delta":
        return values[-1] - values[0]
    if normalized_operation == "summary":
        return {
            "field": field,
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "delta": values[-1] - values[0],
        }
    return sum(values)


def h_field_value_after_steps(
    sequence: List[Dict[str, Any]],
    steps: int,
    field: str = "value",
    index_origin: int = 1,
    return_record: bool = False,
    **_,
) -> Any:
    if not sequence:
        return 0.0
    step_index = int(steps) - int(index_origin)
    idx = min(max(step_index, 0), len(sequence) - 1)
    record = dict(sequence[idx] or {})
    return record if return_record else _get_numeric(_get_field(record, field), 0.0)


def h_compare_record_sequences(
    sequence_a: List[Dict[str, Any]],
    sequence_b: List[Dict[str, Any]],
    field: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    **_,
) -> Dict[str, Any]:
    metrics_set = {
        str(metric).strip().lower()
        for metric in (metrics or ["length", "sum", "delta", "mean"])
        if str(metric).strip()
    }
    values_a = _sequence_numeric_values(sequence_a or [], field) if field else []
    values_b = _sequence_numeric_values(sequence_b or [], field) if field else []
    result: Dict[str, Any] = {
        "len_a": len(sequence_a or []),
        "len_b": len(sequence_b or []),
        "length_delta": len(sequence_a or []) - len(sequence_b or []),
    }
    if field and values_a and values_b:
        total_a = sum(values_a)
        total_b = sum(values_b)
        result.update(
            {
                "field": field,
                **({"field_total_a": total_a, "field_total_b": total_b, "field_total_delta": total_a - total_b} if "sum" in metrics_set else {}),
                **({"field_mean_a": total_a / len(values_a), "field_mean_b": total_b / len(values_b), "field_mean_delta": (total_a / len(values_a)) - (total_b / len(values_b))} if "mean" in metrics_set else {}),
                **({"field_value_delta_a": values_a[-1] - values_a[0], "field_value_delta_b": values_b[-1] - values_b[0], "field_value_delta_gap": (values_a[-1] - values_a[0]) - (values_b[-1] - values_b[0])} if "delta" in metrics_set else {}),
            }
        )
    return result


# --- Spec list ---------------------------------------------------------------
META_TOOL_SPECS = [
    ("Sum_Field", "Sum numeric values of a field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "sum"], h_sum_field),
    ("Average_Field", "Average of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "avg"], h_avg_field),
    ("Median_Field", "Median of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "median"], h_median_field),
    ("Variance_Field", "Variance of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "variance"], h_variance_field),
    ("Stddev_Field", "Standard deviation of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "std"], h_stddev_field),
    ("Percentile_10", "10th percentile of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "percentile"], lambda **k: _percentile(k["data"], k["field"], 0.10)),
    ("Percentile_25", "25th percentile of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "percentile"], lambda **k: _percentile(k["data"], k["field"], 0.25)),
    ("Percentile_50", "50th percentile of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "percentile"], lambda **k: _percentile(k["data"], k["field"], 0.50)),
    ("Percentile_75", "75th percentile of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "percentile"], lambda **k: _percentile(k["data"], k["field"], 0.75)),
    ("Percentile_90", "90th percentile of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "percentile"], lambda **k: _percentile(k["data"], k["field"], 0.90)),
    ("Percentile_95", "95th percentile of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "percentile"], lambda **k: _percentile(k["data"], k["field"], 0.95)),
    ("Percentile_99", "99th percentile of numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["aggregation", "percentile"], lambda **k: _percentile(k["data"], k["field"], 0.99)),
    ("TopK_By_Field", "Top-K items by field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "k": PROP_K}), ["ranking", "topk"], h_topk_by_field),
    ("BottomK_By_Field", "Bottom-K items by field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "k": PROP_K}), ["ranking", "bottomk"], h_bottomk_by_field),
    ("Sort_By_Field_Asc", "Sort ascending by field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["sort"], lambda **k: h_sort_by_field(k["data"], k["field"], True)),
    ("Sort_By_Field_Desc", "Sort descending by field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["sort"], lambda **k: h_sort_by_field(k["data"], k["field"], False)),
    ("Filter_Equals", "Filter where field == value.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "value": PROP_VALUE}), ["filter"], h_filter_equals),
    ("Filter_NotEquals", "Filter where field != value.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "value": PROP_VALUE}), ["filter"], h_filter_notequals),
    ("Filter_GreaterThan", "Filter where field > value.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "value": PROP_NUM}), ["filter"], h_filter_gt),
    ("Filter_LessThan", "Filter where field < value.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "value": PROP_NUM}), ["filter"], h_filter_lt),
    ("Filter_Between", "Filter where low <= field <= high.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "low": PROP_NUM, "high": PROP_NUM}), ["filter"], h_filter_between),
    ("Filter_In", "Filter where field in values.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "values": {"type": "array", "description": "Allowed values"}}), ["filter"], h_filter_in),
    ("Filter_NotIn", "Filter where field not in values.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "values": {"type": "array", "description": "Disallowed values"}}), ["filter"], h_filter_not_in),
    ("Filter_Contains", "Filter where field contains substring.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "substring": PROP_VALUE}), ["filter", "text"], h_filter_contains),
    ("Filter_NotContains", "Filter where field does not contain substring.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "substring": PROP_VALUE}), ["filter", "text"], h_filter_not_contains),
    ("Threshold_Filter", "Keep items where field >= threshold.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "threshold": PROP_NUM}), ["filter", "threshold"], h_threshold_filter),
    ("Deduplicate_By_Field", "Deduplicate by field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["dedup"], h_dedup_by_field),
    ("Select_Fields", "Project selected fields.", _build_params({"data": PROP_DATA, "fields": {"type": "array", "description": "Fields to keep"}}), ["projection"], h_select_fields),
    ("Rename_Field", "Rename a field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "new_name": PROP_FIELD}), ["projection"], h_rename_field),
    ("Drop_Fields", "Drop specified fields.", _build_params({"data": PROP_DATA, "fields": {"type": "array", "description": "Fields to drop"}}), ["projection"], h_drop_fields),
    ("Fillna_Field", "Fill missing field with value.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "fill_value": {"type": "string", "description": "Fill value"}}), ["cleaning"], h_fillna_field),
    ("Add_Constant_Field", "Add constant to numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "value": PROP_NUM}), ["math"], h_add_constant),
    ("Multiply_Field", "Multiply numeric field by factor.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "factor": PROP_NUM}), ["math"], h_multiply_field),
    ("Divide_Field", "Divide numeric field by divisor.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "divisor": PROP_NUM}), ["math"], h_divide_field),
    ("Clip_Field", "Clip numeric field to [low, high].", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "low": PROP_NUM, "high": PROP_NUM}), ["math"], h_clip_field),
    ("Normalize_MinMax", "Min-max normalize numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["math", "normalize"], h_minmax),
    ("Normalize_ZScore", "Z-score normalize numeric field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["math", "normalize"], h_zscore),
    ("Softmax_Scores", "Softmax over field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["math", "softmax"], h_softmax),
    ("Argmax_Field", "Return item with max field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["selection"], h_argmax_field),
    ("Argmin_Field", "Return item with min field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["selection"], h_argmin_field),
    ("Count_By_Field", "Count occurrences by field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["count"], h_count_by_field),
    ("Group_By_Field_Sum", "Group by field and sum target.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "target": PROP_FIELD}), ["group"], h_group_sum),
    ("Group_By_Field_Avg", "Group by field and avg target.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "target": PROP_FIELD}), ["group"], h_group_avg),
    ("Group_By_Field_Max", "Group by field and max target.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "target": PROP_FIELD}), ["group"], lambda **k: h_group_extreme(k["data"], k["field"], k["target"], "max")),
    ("Group_By_Field_Min", "Group by field and min target.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "target": PROP_FIELD}), ["group"], lambda **k: h_group_extreme(k["data"], k["field"], k["target"], "min")),
    ("Merge_By_Key", "Group items into dict keyed by field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["group"], h_merge_by_key),
    ("Flatten_List_Field", "Flatten list values from field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["flatten"], h_flatten_list_field),
    ("Unique_Values", "Unique values of field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["dedup"], h_unique_values),
    ("Value_Frequencies", "Frequency count of field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["count"], h_value_frequencies),
    ("Regex_Filter", "Filter items whose field matches regex.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "pattern": PROP_VALUE}), ["filter", "regex"], h_regex_filter),
    ("Regex_Extract", "Extract regex matches from field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "pattern": PROP_VALUE}), ["extract", "regex"], h_regex_extract),
    ("Keyword_Filter", "Filter items containing keyword.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "keyword": PROP_VALUE}), ["filter", "text"], h_keyword_filter),
    ("Truncate_Text", "Truncate text field to max length.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "max_len": {"type": "integer", "description": "Maximum length"}}), ["text"], h_truncate_text),
    ("To_Lower", "Lowercase text field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["text"], h_lower),
    ("To_Upper", "Uppercase text field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["text"], h_upper),
    ("Strip_Text", "Strip whitespace in text field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["text"], h_strip),
    ("Concat_Fields", "Concatenate multiple fields.", _build_params({"data": PROP_DATA, "fields": {"type": "array", "description": "Fields to concat"}, "target": PROP_FIELD}, {"sep": {"type": "string", "description": "Separator", "default": " "}}), ["text"], h_concat_fields),
    ("Join_With_Delimiter", "Join list of strings with delimiter.", _build_params({"values": {"type": "array", "description": "Values to join"}}, {"delimiter": {"type": "string", "description": "Delimiter", "default": ","}}), ["text"], h_join_with_delimiter),
    ("Time_Range_Length", "Compute duration of a time range.", _build_params({"time_range": PROP_TIME_RANGE}), ["time"], h_time_range_length),
    ("Time_Overlap", "Compute overlap of two ranges.", _build_params({"range_a": PROP_TIME_RANGE, "range_b": PROP_TIME_RANGE}), ["time"], h_time_overlap),
    ("Merge_Touching_Ranges", "Merge overlapping/adjacent ranges.", _build_params({"ranges": {"type": "array", "description": "List of ranges"}}, {"tolerance": PROP_NUM}), ["time", "merge"], h_merge_touching),
    ("Align_Time_Buckets", "Align ranges to bucket size.", _build_params({"ranges": {"type": "array", "description": "List of ranges"}, "bucket": {"type": "number", "description": "Bucket size"}}), ["time"], h_align_time_buckets),
    ("Clip_Time_Range", "Clip time range to bounds.", _build_params({"time_range": PROP_TIME_RANGE, "clip_start": PROP_NUM, "clip_end": PROP_NUM}), ["time"], h_clip_time_range),
    ("Expand_Time_Range", "Expand time range by delta.", _build_params({"time_range": PROP_TIME_RANGE, "delta": PROP_NUM}), ["time"], h_expand_time_range),
    ("Shift_Time_Range", "Shift time range by offset.", _build_params({"time_range": PROP_TIME_RANGE, "offset": PROP_NUM}), ["time"], h_shift_time_range),
    ("Sort_Time_Ranges", "Sort ranges by start time.", _build_params({"ranges": {"type": "array", "description": "List of ranges"}}), ["time", "sort"], h_sort_time_ranges),
    ("Window_Sliding", "Generate sliding windows.", _build_params({"data": PROP_DATA}, {"window": PROP_WINDOW}), ["window"], h_window_sliding),
    ("Neighbor_List_Graph", "Get graph neighbors for a node from either an adjacency dict or a knowledge-graph payload.", _build_params({"graph": {"type": "object", "description": "Adjacency dict or knowledge-graph payload"}, "node": PROP_FIELD}, {"relation": PROP_FIELD, "include_metadata": {"type": "boolean", "description": "Return edge metadata instead of bare neighbor names", "default": False}}), ["graph"], h_neighbor_list),
    ("Degree_Count_Graph", "Count graph neighbors for a node, optionally restricted to one relation family.", _build_params({"graph": {"type": "object", "description": "Adjacency dict or knowledge-graph payload"}, "node": PROP_FIELD}, {"relation": PROP_FIELD}), ["graph"], h_degree_count),
    ("Edge_Filter_Graph", "Filter graph edges by allowed nodes and optional relation name.", _build_params({"graph": {"type": "object", "description": "Adjacency dict or knowledge-graph payload"}, "allowed": {"type": "array", "description": "Allowed nodes"}}, {"relation": PROP_FIELD}), ["graph"], h_edge_filter),
    ("Path_Length_Filter", "Filter graph paths or path records by inclusive length bounds.", _build_params({"paths": {"type": "array", "description": "List of paths or path records"}, "max_len": PROP_K}, {"min_len": {"type": "integer", "description": "Minimum allowed path length", "default": 1}}), ["graph"], h_path_length_filter),
    ("Score_Threshold_Rank", "Filter items by score threshold.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "threshold": PROP_NUM}), ["ranking"], h_score_threshold_rank),
    ("TopK_By_Score", "Top-K by 'score' field.", _build_params({"data": PROP_DATA, "k": PROP_K}), ["ranking"], h_topk_score),
    ("BottomK_By_Score", "Bottom-K by 'score' field.", _build_params({"data": PROP_DATA, "k": PROP_K}), ["ranking"], h_bottomk_score),
    ("Rerank_By_Score", "Rerank by weighted score.", _build_params({"data": PROP_DATA}, {"weight": PROP_NUM}), ["ranking"], h_rerank_by_score),
    ("Weighted_Sum_Two_Fields", "Weighted sum of two fields.", _build_params({"data": PROP_DATA, "field_a": PROP_FIELD, "field_b": PROP_FIELD}, {"weight_a": PROP_NUM, "weight_b": PROP_NUM}), ["math"], h_weighted_sum_two_fields),
    ("Weighted_Average_Field", "Weighted average of field using weight_field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "weight_field": PROP_FIELD}), ["math"], h_weighted_avg),
    ("CumSum_Field", "Cumulative sum over field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["math"], h_cumsum_field),
    ("Moving_Average", "Moving average over numeric list.", _build_params({"data": PROP_DATA}, {"window": PROP_WINDOW}), ["math", "window"], h_moving_average),
    ("Rolling_Max", "Rolling max over numeric list.", _build_params({"data": PROP_DATA}, {"window": PROP_WINDOW}), ["math", "window"], h_rolling_max),
    ("Rolling_Min", "Rolling min over numeric list.", _build_params({"data": PROP_DATA}, {"window": PROP_WINDOW}), ["math", "window"], h_rolling_min),
    ("Rolling_Median", "Rolling median over numeric list.", _build_params({"data": PROP_DATA}, {"window": PROP_WINDOW}), ["math", "window"], h_rolling_median),
    ("Ratio_Fields", "Compute ratio numerator/denominator.", _build_params({"data": PROP_DATA, "numerator": PROP_FIELD, "denominator": PROP_FIELD}), ["math"], h_ratio_fields),
    ("Percentage_Change", "Percent change over numeric series.", _build_params({"data": PROP_DATA}), ["math"], h_percentage_change),
    ("Boolean_And", "Logical AND over booleans.", _build_params({"flags": {"type": "array", "description": "Boolean list"}}), ["logic"], h_boolean_and),
    ("Boolean_Or", "Logical OR over booleans.", _build_params({"flags": {"type": "array", "description": "Boolean list"}}), ["logic"], h_boolean_or),
    ("Boolean_Not", "Logical NOT.", _build_params({"flag": {"type": "boolean", "description": "Flag"}}), ["logic"], h_boolean_not),
    ("Count_True", "Count True values.", _build_params({"flags": {"type": "array", "description": "Boolean list"}}), ["logic", "count"], h_count_true),
    ("Any_True", "Check any True.", _build_params({"flags": {"type": "array", "description": "Boolean list"}}), ["logic"], h_any_true),
    ("All_True", "Check all True.", _build_params({"flags": {"type": "array", "description": "Boolean list"}}), ["logic"], h_all_true),
    ("Sample_N", "Sample N items.", _build_params({"data": PROP_DATA, "n": PROP_K}), ["sample"], h_sample_n),
    ("Shuffle_Data", "Shuffle list.", _build_params({"data": PROP_DATA}), ["sample"], h_shuffle_data),
    ("Take_First_N", "Take first N items.", _build_params({"data": PROP_DATA, "n": PROP_K}), ["selection"], h_take_first_n),
    ("Take_Last_N", "Take last N items.", _build_params({"data": PROP_DATA, "n": PROP_K}), ["selection"], h_take_last_n),
    ("Reverse_List", "Reverse list order.", _build_params({"data": PROP_DATA}), ["selection"], h_reverse_list),
    ("Extract_Field_List", "Extract a field as list.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["projection"], h_extract_field_list),
    ("Build_Index_By_Field", "Build index dict keyed by field.", _build_params({"data": PROP_DATA, "field": PROP_FIELD}), ["projection"], h_build_index),
    ("Map_Field_With_Lookup", "Map field value via lookup dict.", _build_params({"data": PROP_DATA, "field": PROP_FIELD, "lookup": {"type": "object", "description": "Mapping dict"}}), ["projection"], h_map_field_lookup),
    ("Filter_Empty", "Remove empty/falsey items.", _build_params({"data": PROP_DATA}), ["filter"], h_filter_empty),
    ("Ensure_Sorted", "Ensure list is sorted.", _build_params({"data": PROP_DATA}), ["sort"], h_ensure_sorted),
    ("Simple_Interest", "Compute simple interest.", _build_params({"principal": PROP_NUM, "rate": PROP_NUM, "periods": PROP_NUM}), ["finance", "interest"], h_simple_interest),
    ("Compound_Growth", "Compute compound growth for any repeatedly compounding value, optionally with a constant per-step adjustment or returned schedule.", _build_params({"start_value": PROP_NUM, "rate": PROP_NUM, "periods": PROP_NUM}, {"freq": PROP_NUM, "step_adjustment": PROP_NUM, "return_schedule": {"type": "boolean", "description": "Whether to return the per-step schedule", "default": False}}), ["math", "growth", "compound"], h_compound_growth),
    ("Solve_Fixed_Step_Amount", "Solve the constant per-step amount needed to move a compounding value toward a target ending value across fixed steps.", _build_params({"start_value": PROP_NUM, "rate": PROP_NUM, "periods": PROP_NUM}, {"freq": PROP_NUM, "target_end_value": PROP_NUM, "return_schedule": {"type": "boolean", "description": "Whether to return the projected per-step schedule", "default": False}}), ["math", "schedule", "solver"], h_fixed_step_amount),
    ("Amortization_Schedule", "Generate amortization schedule.", _build_params({"principal": PROP_NUM, "rate": PROP_NUM, "periods": PROP_NUM}, {"freq": PROP_NUM, "payment": PROP_NUM}), ["finance", "loan", "schedule"], h_amortization_schedule),
    ("Sequence_Field_Total", "Aggregate one numeric field across any sequence of records using sum/mean/min/max/count/delta/summary.", _build_params({"sequence": {"type": "array", "description": "Sequence of record rows"}, "field": PROP_FIELD}, {"operation": {"type": "string", "description": "sum | mean | min | max | count | delta | summary", "default": "sum"}}), ["aggregation", "sequence"], h_sequence_field_total),
    ("Field_Value_After_Steps", "Read a numeric field or entire record from the row reached after a given number of steps.", _build_params({"sequence": {"type": "array", "description": "Sequence of record rows"}, "steps": PROP_K}, {"field": PROP_FIELD, "index_origin": {"type": "integer", "description": "Whether steps are 0-based or 1-based", "default": 1}, "return_record": {"type": "boolean", "description": "Return the full record instead of one field", "default": False}}), ["selection", "sequence"], h_field_value_after_steps),
    ("Compare_Record_Sequences", "Compare two sequences of records by length and optional numeric-field metrics such as sum/mean/delta.", _build_params({"sequence_a": {"type": "array", "description": "Sequence A"}, "sequence_b": {"type": "array", "description": "Sequence B"}}, {"field": PROP_FIELD, "metrics": {"type": "array", "description": "Requested comparison metrics", "items": {"type": "string"}}}), ["compare", "sequence"], h_compare_record_sequences),
]


def _create_tool_class(name: str, description: str, parameters: Dict[str, Any], tags: List[str], handler):
    def run(self, handler=handler, **kwargs):
        return handler(**kwargs)
    attrs = {
        "name": Field(default=name, description="Unique name of the tool"),
        "description": Field(default=description, description="Human-readable description of what the tool does"),
        "parameters": Field(default=parameters, description="JSON Schema describing the tool's parameters"),
        "tags": Field(
            default_factory=lambda tags=tags: list(set((tags or []) + ["meta"])),
            description="List of tags for categorization/analytics",
        ),
        "run": run,
        "__annotations__": {
            "name": str,
            "description": str,
            "parameters": Dict[str, Any],
            "tags": List[str],
        },
    }
    return type(name, (BaseTool,), attrs)


# Dynamically create tool classes for auto-discovery
for spec in META_TOOL_SPECS:
    cls = _create_tool_class(*spec)
    globals()[cls.__name__] = cls
    __all__ = list(globals().get("__all__", [])) + [cls.__name__]
    logger.info(f"Registered meta tool class: {cls.__name__}")
