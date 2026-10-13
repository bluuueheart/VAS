"""Immutable baseline snapshots for the runtime tool library.

The evaluator may apply an Evolver harness snapshot during a later run.  That
is a policy mutation, not a replacement for the initial tool library.  This
module records the concrete registry, schemas, and toolkit source digests at
the beginning of a batch so every batch has an explicit, comparable starting
point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_TOOLKIT_ROOT = Path(__file__).resolve().parent
_RL_ROOT = _TOOLKIT_ROOT.parent


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tool_records(registry: Any) -> list[Dict[str, Any]]:
    # Use the raw registered map instead of get_all_tools(): the latter hides
    # sparse-scope tools from the planner, while a baseline must also record
    # tools that are available but intentionally hidden by routing policy.
    raw_tools = getattr(registry, "_tools", {})
    records: list[Dict[str, Any]] = []
    for name, tool in sorted(raw_tools.items(), key=lambda item: str(item[0])):
        tool_class = tool.__class__
        records.append(
            {
                "name": str(name),
                "description": str(getattr(tool, "description", "") or ""),
                "parameters": _json_safe(getattr(tool, "parameters", {}) or {}),
                "output_schema": _json_safe(
                    tool.get_output_schema() if hasattr(tool, "get_output_schema") else {}
                ),
                "tags": sorted(str(tag) for tag in (getattr(tool, "tags", None) or [])),
                "is_virtual": bool(getattr(tool, "is_virtual", False)),
                "class_module": str(getattr(tool_class, "__module__", "") or ""),
                "class_qualname": str(getattr(tool_class, "__qualname__", "") or ""),
            }
        )
    return records


def _source_files() -> list[Dict[str, Any]]:
    files: list[Dict[str, Any]] = []
    for path in sorted(_TOOLKIT_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(path.relative_to(_RL_ROOT)),
                "sha256": digest,
                "size_bytes": int(path.stat().st_size),
            }
        )
    return files


def build_registry_snapshot(registry: Any) -> Dict[str, Any]:
    """Build a deterministic description of the currently discovered registry."""

    tools = _tool_records(registry)
    source_files = _source_files()
    try:
        from config.settings import settings

        disabled_tools = sorted(str(name) for name in (settings.disabled_tool_names or ()))
    except Exception:
        disabled_tools = []
    schema_digest = _digest(tools)
    source_digest = _digest(source_files)
    baseline_id = "toolbase_" + _digest(
        {
            "schema_digest": schema_digest,
            "source_digest": source_digest,
            "disabled_tools": disabled_tools,
        }
    )[:20]
    return {
        "kind": "tool_library_baseline",
        "layout_version": 2,
        "result_contract": {
            "status": ["success", "error", "no_valid_result"],
            "required_fields": ["status", "result"],
            "note": "Domain fields are preserved; empty and unavailable results are never success.",
        },
        "baseline_id": baseline_id,
        "schema_digest": schema_digest,
        "source_digest": source_digest,
        "tool_count": len(tools),
        "disabled_tools": disabled_tools,
        "tools": tools,
        "source_files": source_files,
    }


def write_registry_baseline(output_path: Path, registry: Any) -> Dict[str, Any]:
    """Write one baseline atomically, retaining the same content on retries."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    payload = build_registry_snapshot(registry)
    payload["created_at"] = datetime.utcnow().isoformat() + "Z"
    payload["pid"] = os.getpid()
    temporary_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return payload


def load_registry_baseline(path: Optional[Path | str]) -> Optional[Dict[str, Any]]:
    raw_path = str(path or "").strip()
    if not raw_path:
        return None
    resolved_path = Path(raw_path).expanduser().resolve()
    if not resolved_path.exists():
        return None
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "tool_library_baseline":
        raise ValueError(f"Invalid tool library baseline: {resolved_path}")
    payload["path"] = str(resolved_path)
    return payload


def registry_runtime_audit(
    registry: Any,
    baseline: Optional[Mapping[str, Any]],
    baseline_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Compare the active registry with the immutable batch baseline."""

    current = build_registry_snapshot(registry)
    baseline_tools = {
        str(item.get("name")): item
        for item in (baseline or {}).get("tools", [])
        if isinstance(item, Mapping)
    }
    current_tools = {
        str(item.get("name")): item
        for item in current.get("tools", [])
        if isinstance(item, Mapping)
    }
    baseline_schema_digest = str((baseline or {}).get("schema_digest") or "")
    baseline_source_digest = str((baseline or {}).get("source_digest") or "")
    baseline_names = set(baseline_tools)
    current_names = set(current_tools)
    scoped_hidden_tools: List[str] = []
    try:
        from toolkit.impl.retrieval import get_sparse_scope_hidden_tool_payload

        scoped_hidden_tools = sorted(
            name
            for name in (baseline_names - current_names)
            if get_sparse_scope_hidden_tool_payload(name) is not None
        )
    except Exception:
        # The audit must remain usable for non-video/unit-test registries.
        scoped_hidden_tools = []
    removed_tools = sorted((baseline_names - current_names) - set(scoped_hidden_tools))
    added_tools = sorted(current_names - baseline_names)
    changed_tools = sorted(
        name
        for name in baseline_names.intersection(current_names)
        if baseline_tools[name] != current_tools[name]
    )
    scoped_schema_match = not added_tools and not removed_tools and not changed_tools
    return {
        "baseline_id": str((baseline or {}).get("baseline_id") or "snapshot_000"),
        "baseline_path": str(
            Path(baseline_path).expanduser().resolve()
            if baseline_path
            else (baseline or {}).get("path") or ""
        ),
        "baseline_schema_digest": baseline_schema_digest,
        "baseline_source_digest": baseline_source_digest,
        "active_schema_digest": str(current.get("schema_digest") or ""),
        "active_source_digest": str(current.get("source_digest") or ""),
        "active_tool_count": int(current.get("tool_count") or 0),
        "schema_match": (
            bool(baseline_schema_digest)
            and baseline_schema_digest == str(current.get("schema_digest") or "")
        ) or scoped_schema_match,
        "source_match": bool(baseline_source_digest)
        and baseline_source_digest == str(current.get("source_digest") or ""),
        "added_tools": added_tools,
        "removed_tools": removed_tools,
        "scoped_hidden_tools": scoped_hidden_tools,
        "changed_tools": changed_tools,
        "scoped_schema_match": scoped_schema_match,
        "active_tools": sorted(current_tools),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Tool Star registry baseline snapshot.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from toolkit.registry import registry

    registry.auto_discover()
    payload = write_registry_baseline(args.output, registry)
    print(
        json.dumps(
            {
                "baseline_id": payload.get("baseline_id"),
                "schema_digest": payload.get("schema_digest"),
                "source_digest": payload.get("source_digest"),
                "tool_count": payload.get("tool_count"),
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
