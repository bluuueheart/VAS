"""Exact source edits materialized only in isolated candidate workers.

This module is an execution boundary, not a claim that generated code is
correct. Syntax, source identity and registry checks precede paired inference;
only independent replay can establish utility.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


KINDS = {"add", "delete", "update", "compose", "fallback", "prompt"}
PROMPT_PATHS = {"planner/prompts.py", "toolkit/impl/visual_prompts.py", "config/agent_prompts.json"}
FORBIDDEN = {"ground_truth", "groundtruth", "correct_label", "gold_answer", "official_meta", "time_reference", "evaluator_output"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inherit_snapshot(parent: Mapping[str, Any], candidate: Mapping[str, Any],
                     base_root: Path, parent_root: Path) -> dict[str, Any]:
    """Flatten a child into a standalone snapshot without losing parent edits.

    New exact patches address the active parent tree. Persisted patches address
    the immutable base tree, so workers need no mutable parent lookup.
    """
    result = copy.deepcopy(dict(candidate))
    delta = copy.deepcopy(dict(candidate))
    local_targets = []
    for field in ("router_policy", "argument_policy", "tool_overrides"):
        merged = copy.deepcopy(parent.get(field) or {})
        for key, value in (candidate.get(field) or {}).items():
            if field == "tool_overrides" and isinstance(value, dict):
                merged[key] = {**merged.get(key, {}), **copy.deepcopy(value)}
            else:
                merged[key] = copy.deepcopy(value)
        result[field] = merged
    changed = edited_sources(candidate, parent_root)
    for target, spec in (candidate.get("source_overrides") or {}).items():
        safe = spec.get("kind") == "update"
        for edit in spec.get("edits") or []:
            path = parent_root / edit["path"]
            safe = safe and path.suffix == ".py" and path.is_file()
            if not safe:
                break
            before_tree, after_tree = ast.parse(path.read_text()), ast.parse(changed[edit["path"]])
            found = []
            for tree in (before_tree, after_tree):
                methods = []
                for cls in tree.body:
                    if not isinstance(cls, ast.ClassDef):
                        continue
                    tool_names = [n.value.value for n in cls.body if isinstance(n, ast.AnnAssign)
                                  and isinstance(n.target, ast.Name) and n.target.id == "name"
                                  and isinstance(n.value, ast.Constant)]
                    if target in tool_names:
                        for method in cls.body:
                            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name == "run":
                                methods.append(method)
                                method.body = [ast.Pass()]
                found.append(bool(methods))
            safe = safe and all(found) and ast.dump(before_tree) == ast.dump(after_tree)
        if safe:
            local_targets.append(target)
    delta["source_local_targets"] = local_targets
    result["mutation_delta"] = delta
    overrides = copy.deepcopy(parent.get("source_overrides") or {})
    # A file has one owner in a flattened snapshot. Child edits supersede that
    # file, preserving its inherited content through the parent-root patch.
    for target in list(overrides):
        overrides[target]["edits"] = [e for e in overrides[target]["edits"] if e["path"] not in changed]
        if not overrides[target]["edits"]:
            del overrides[target]
    for target, spec in (candidate.get("source_overrides") or {}).items():
        edits = []
        for edit in spec["edits"]:
            path = base_root / edit["path"]
            original = path.read_text(encoding="utf-8") if path.exists() else ""
            edits.append({"path": edit["path"], "before_sha256": digest(original),
                          "before": original, "after": changed[edit["path"]]})
        previous = overrides.get(target, {}).get("edits", [])
        overrides[target] = {"kind": spec["kind"], "edits": previous + edits}
    result["source_overrides"] = overrides
    result["source_materialized"] = True
    result["patch_ids"] = list(dict.fromkeys(list(parent.get("patch_ids") or []) + list(candidate.get("patch_ids") or [])))
    # Identity includes the effective content, not only the newest delta.
    identity = {k: result.get(k) for k in ("parent_snapshot_id", "router_policy", "argument_policy", "tool_overrides", "source_overrides")}
    result["snapshot_id"] = "snapshot_" + digest(json.dumps(identity, sort_keys=True))[:16]
    return result


def validate_source_edit(after: Mapping[str, Any], *, bounded: bool = True) -> None:
    if set(after) != {"kind", "edits"} or after.get("kind") not in KINDS:
        raise ValueError("source mutation requires kind and edits")
    edits = after.get("edits")
    if not isinstance(edits, list) or not 1 <= len(edits) <= (4 if bounded else 256):
        raise ValueError("source mutation requires 1..4 exact edits")
    seen = set()
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"path", "before_sha256", "before", "after"}:
            raise ValueError("each edit requires path, before_sha256, before, after")
        path = Path(str(edit["path"]))
        allowed_tool = len(path.parts) == 3 and path.parts[:2] == ("toolkit", "impl") and path.suffix == ".py" and not path.name.startswith("_")
        if path.is_absolute() or ".." in path.parts or not (allowed_tool or path.as_posix() in PROMPT_PATHS):
            raise ValueError(f"source path is outside the mutation boundary: {path}")
        if after["kind"] == "prompt" and path.as_posix() not in PROMPT_PATHS:
            raise ValueError("prompt mutations may edit only prompt files")
        if path.as_posix() in seen:
            raise ValueError("combine edits to the same file into one exact replacement")
        seen.add(path.as_posix())
        if not all(isinstance(edit[key], str) for key in ("before_sha256", "before", "after")):
            raise ValueError("source edit fields must be strings")
        if edit["before"] == edit["after"] or (bounded and len(edit["after"]) > 48000):
            raise ValueError("source edit is a no-op or exceeds the bounded patch size")
        # Evaluation fields have no place in tool implementations or prompts.
        if bounded and any(token in edit["after"].lower() for token in FORBIDDEN):
            raise ValueError("evaluation-only field in source mutation")


def edited_sources(snapshot: Mapping[str, Any], root: Path) -> dict[str, str]:
    result = {}
    for spec in (snapshot.get("source_overrides") or {}).values():
        validate_source_edit(spec, bounded=not snapshot.get("source_materialized", False))
        for edit in spec["edits"]:
            relative = edit["path"]
            if relative in result:
                raise ValueError(f"joint source edits conflict: {relative}")
            path = root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("source edit escapes its runtime root")
            original = path.read_text(encoding="utf-8") if path.exists() else ""
            if digest(original) != edit["before_sha256"]:
                raise ValueError(f"baseline source digest mismatch: {relative}")
            before = edit["before"]
            if before:
                if original.count(before) != 1:
                    raise ValueError(f"source replacement must match exactly once: {relative}")
                changed = original.replace(before, edit["after"], 1)
            elif original:
                raise ValueError("empty before is allowed only for a new file")
            else:
                changed = edit["after"]
            if path.suffix == ".py":
                compile(changed, relative, "exec")
            else:
                json.loads(changed)
            result[relative] = changed
    return result


def materialize_source_snapshot(snapshot: Mapping[str, Any], source_root: Path, destination: Path) -> Path:
    """Copy code/config only; never patch the baseline or copy model/cache data."""
    changed = edited_sources(snapshot, source_root)
    if destination.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("candidate source tree must be outside the baseline tree")
    if destination.exists():
        raise ValueError(f"candidate source workspace already exists: {destination}")
    destination.mkdir(parents=True)
    excluded = {"src", "runs", "rl_workdir", "videorag_cache", "__pycache__"}
    for directory, dirs, files in os.walk(source_root):
        dirs[:] = [name for name in dirs if name not in excluded and not name.startswith(".")]
        for name in files:
            path = Path(directory) / name
            if path.suffix not in {".py", ".json", ".yaml", ".yml", ".txt", ".sh"} or path.is_symlink():
                continue
            relative = path.relative_to(source_root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    for relative, content in changed.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (destination / "source_snapshot.json").write_text(json.dumps({
        "snapshot_id": snapshot["snapshot_id"],
        "source_root": str(source_root.resolve()),
        "changed_files": {name: digest(content) for name, content in changed.items()},
    }, indent=2), encoding="utf-8")
    return destination


def source_application_proof(snapshot_id: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
    if not overrides:
        return {"applied": False, "targets": []}
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "source_snapshot.json"
    if not manifest_path.is_file():
        return {"applied": False, "targets": [], "reason": "isolated_source_worker_required"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_root = Path(manifest["source_root"])
    expected = edited_sources({"source_overrides": overrides, "source_materialized": True}, original_root)
    valid = manifest.get("snapshot_id") == snapshot_id and all(
        (root / name).is_file() and digest((root / name).read_text(encoding="utf-8")) == digest(content)
        for name, content in expected.items()
    )
    from toolkit.registry import registry
    for target, spec in overrides.items():
        if spec["kind"] == "prompt":
            continue
        present = registry.has_tool(target)
        valid = valid and (not present if spec["kind"] == "delete" else present)
    return {"applied": bool(valid), "targets": sorted(overrides) if valid else [], "changed_files": manifest.get("changed_files", {})}


def source_context(root: Path, targets: list[str]) -> list[dict[str, str]]:
    """Expose bounded exact snippets and digests, never evaluation artifacts."""
    contexts = []
    for path in sorted((root / "toolkit" / "impl").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            names = [child.value.value for child in node.body if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name) and child.target.id == "name" and isinstance(child.value, ast.Constant)]
            if not set(names).intersection(targets):
                continue
            lines = source.splitlines(keepends=True)
            snippet = "".join(lines[node.lineno - 1:min(node.end_lineno, node.lineno + 100)])
            contexts.append({"path": str(path.relative_to(root)), "before_sha256": digest(source), "snippet": snippet})
    prompt_path = root / "config/agent_prompts.json"
    if len(contexts) < 4 and prompt_path.is_file():
        text = prompt_path.read_text(encoding="utf-8")
        start = text.index('"reflector"')
        prompt = text[start:start + 5000]
        contexts.append({"path": "config/agent_prompts.json", "before_sha256": digest(text), "snippet": prompt[:5000]})
    return contexts[:4]
