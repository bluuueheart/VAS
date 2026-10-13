"""Apply Evolver registry snapshots to an isolated runtime harness.

Evolution reports are not evidence of an applied change by themselves. This
module is the boundary between a candidate RegistrySnapshot and the runtime
verifier/planner. A snapshot is opt-in through
CFV_REGISTRY_SNAPSHOT_PATH and never mutates the production registry.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .verifier import load_evolution_config
from .models import valid_recovery_policy
from .source_mutation import source_application_proof, validate_source_edit


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _baseline_state() -> Dict[str, Any]:
    return {
        "enabled": False,
        "status": "baseline",
        "snapshot_id": "snapshot_000",
        "snapshot_path": "",
        "policy_digest": "baseline",
        "operations": [],
        "policy": {},
        "verifier_config": load_evolution_config(),
    }


def load_harness_snapshot(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load an explicit candidate/accepted snapshot for one runtime.

    A malformed explicitly requested snapshot fails fast. With no path the
    baseline state is returned, so ordinary M3 runs remain unchanged.
    """

    raw_path = str(path or os.environ.get("CFV_REGISTRY_SNAPSHOT_PATH") or "").strip()
    if not raw_path:
        return _baseline_state()
    snapshot_path = Path(raw_path).expanduser().resolve()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Evolution harness snapshot not found: {snapshot_path}")
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Evolution harness snapshot: {snapshot_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Evolution harness snapshot must be a JSON object: {snapshot_path}")
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"candidate", "accepted"}:
        raise ValueError(f"Evolution harness snapshot status must be candidate/accepted, got {status!r}")

    expected_baseline_id = str(os.environ.get("CFV_REGISTRY_BASELINE_ID") or "").strip()
    snapshot_baseline_id = str(payload.get("base_tool_library_baseline_id") or "").strip()
    if snapshot_baseline_id and expected_baseline_id and snapshot_baseline_id != expected_baseline_id:
        raise ValueError(
            "Evolution harness snapshot targets a different tool library baseline: "
            f"snapshot={snapshot_baseline_id} runtime={expected_baseline_id}"
        )

    router_policy = payload.get("router_policy") if isinstance(payload.get("router_policy"), dict) else {}
    argument_policy = payload.get("argument_policy") if isinstance(payload.get("argument_policy"), dict) else {}
    raw_tool_overrides = payload.get("tool_overrides") if isinstance(payload.get("tool_overrides"), dict) else {}
    tool_overrides = {
        str(tool_name): copy.deepcopy(override)
        for tool_name, override in raw_tool_overrides.items()
        if str(tool_name).strip() and isinstance(override, dict)
    }
    operations = []
    policy: Dict[str, Any] = {}
    source_overrides = payload.get("source_overrides") or {}
    for target, spec in source_overrides.items():
        validate_source_edit(spec, bounded=not payload.get("source_materialized", False))
        operations.append({"action": "edit_tool_source", "target": target, "effect": {"kind": spec["kind"], "paths": [edit["path"] for edit in spec["edits"]]}})

    for tool_name, override in sorted(tool_overrides.items()):
        operations.append({
            "action": "update_tool_schema",
            "target": tool_name,
            "effect": copy.deepcopy(override),
        })

    composition = router_policy.get("add_composition_edge")
    if isinstance(composition, dict) and bool(composition.get("required")):
        policy["require_option_comparison_all_tasks"] = True
        operations.append({
            "action": "add_composition_edge",
            "target": "candidate_comparison",
            "effect": "require verified comparison evidence before Finish",
        })

    unsupported_policy_keys = sorted(
        set(router_policy).intersection({"update_router_rule", "add_fallback"})
    )
    if unsupported_policy_keys:
        raise ValueError(
            "Evolution snapshot contains policy-only mutations without runtime hooks: "
            f"{unsupported_policy_keys}"
        )

    recovery = router_policy.get("update_recovery_policy")
    if recovery is not None:
        if not valid_recovery_policy(recovery):
            raise ValueError(
                "Evolution update_recovery_policy does not match the implemented runtime contract"
            )
        policy["low_signal_temporal_grounding"] = copy.deepcopy(recovery)
        operations.append({
            "action": "update_recovery_policy",
            "target": "low_signal_temporal_grounding",
            "effect": copy.deepcopy(recovery),
        })

    for action, value in argument_policy.items():
        raise ValueError(
            "Evolution snapshot contains an argument policy without an executable runtime hook: "
            f"{action}"
        )

    state = {
        "enabled": True,
        "status": status,
        "snapshot_id": str(payload.get("snapshot_id") or ""),
        "snapshot_path": str(snapshot_path),
        "policy_digest": _canonical_digest(payload),
        "operations": operations,
        "policy": policy,
        "tool_overrides": tool_overrides,
        "source_overrides": source_overrides,
        "base_tool_library_baseline_id": snapshot_baseline_id or expected_baseline_id,
        "registry_mutations_applied": [],
        "verifier_config": load_evolution_config(),
    }
    verification = state["verifier_config"].setdefault("verification", {})
    if policy.get("require_option_comparison_all_tasks"):
        verification["require_option_comparison_all_tasks"] = True
    state["verifier_config"]["evolution_harness_snapshot_id"] = state["snapshot_id"]
    return state


def apply_harness_registry_overrides(registry: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply candidate tool-schema changes inside the current worker only.

    The registry is process-local.  This function intentionally mutates only
    the isolated worker's singleton and refuses unknown fields so a candidate
    cannot silently alter tool execution semantics outside the declared schema.
    """

    applied = []
    allowed_fields = {"description", "parameters", "tags"}
    for tool_name, override in (state.get("tool_overrides") or {}).items():
        tool = registry.get_tool(str(tool_name), log_missing=False)
        if bool(getattr(tool, "is_virtual", False)) or not hasattr(tool, "name"):
            raise ValueError(f"Evolution tool-schema override targets unavailable tool: {tool_name}")
        unknown_fields = sorted(set(override) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"Evolution tool-schema override for {tool_name} has unsupported fields: {unknown_fields}"
            )
        before = {
            "description": str(getattr(tool, "description", "") or ""),
            "parameters": copy.deepcopy(getattr(tool, "parameters", {}) or {}),
            "tags": list(getattr(tool, "tags", None) or []),
        }
        if "description" in override:
            tool.description = str(override["description"])
        if "parameters" in override:
            if not isinstance(override["parameters"], dict):
                raise ValueError(f"Evolution parameters override for {tool_name} must be an object")
            candidate_parameters = override["parameters"]
            if "properties" not in candidate_parameters:
                raise ValueError(
                    f"Evolution parameters override for {tool_name} must be a JSON-schema object "
                    "with a properties field"
                )
            candidate_properties = candidate_parameters.get("properties", {})
            if not isinstance(candidate_properties, dict):
                raise ValueError(
                    f"Evolution parameters override for {tool_name} must declare properties as an object"
                )
            current_properties = before["parameters"].get("properties", {})
            current_properties = current_properties if isinstance(current_properties, dict) else {}
            added_properties = sorted(set(candidate_properties) - set(current_properties))
            if added_properties:
                run_callable = getattr(tool, "run", None)
                try:
                    signature = inspect.signature(run_callable)
                    explicit_parameters = {
                        name
                        for name, parameter in signature.parameters.items()
                        if name != "self"
                        and parameter.kind
                        in {
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        }
                    }
                except (TypeError, ValueError):
                    explicit_parameters = set()
                unsupported_properties = [
                    name for name in added_properties if name not in explicit_parameters
                ]
                if unsupported_properties:
                    raise ValueError(
                        f"Evolution tool-schema override for {tool_name} adds parameters without a "
                        f"declared runtime implementation: {unsupported_properties}"
                    )
            tool.parameters = copy.deepcopy(candidate_parameters)
        if "tags" in override:
            if not isinstance(override["tags"], list):
                raise ValueError(f"Evolution tags override for {tool_name} must be an array")
            tool.tags = list(override["tags"])
        applied.append({
            "action": "update_tool_schema",
            "target": str(tool_name),
            "before": before,
            "after": copy.deepcopy(override),
        })
    state["registry_mutations_applied"] = applied
    return state


def render_harness_context(state: Mapping[str, Any]) -> str:
    """Render the applied mutation as planner-visible, non-answer context."""

    if not bool(state.get("enabled")):
        return ""
    lines = [
        "[Evolution Harness Policy]",
        f"snapshot_id={state.get('snapshot_id')}",
        f"snapshot_status={state.get('status')}",
        "The following accepted/candidate harness mutations are active for this run:",
    ]
    for operation in state.get("operations") or []:
        if isinstance(operation, Mapping):
            lines.append(
                f"- {operation.get('action')} target={operation.get('target')} effect={operation.get('effect')}"
            )
    for tool_name in sorted((state.get("tool_overrides") or {}).keys()):
        lines.append(
            f"- update_tool_schema target={tool_name}; the declared schema is the only allowed tool mutation"
        )
    lines.append(
        "Do not treat this policy as video evidence; use it only to decide the required workflow and Finish gate."
    )
    return "\n".join(lines)


def _snapshot_application_proof(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Prove runtime-hook application without inferring from a new answer.

    Loading a candidate snapshot and running a candidate trajectory are
    separate facts. Schema mutations additionally need the process-local
    registry override record; policy/composition mutations are materialized
    in the isolated verifier/engine policy before inference starts.
    """

    snapshot_loaded = bool(
        state.get("enabled")
        and str(state.get("snapshot_id") or "").strip()
        and str(state.get("snapshot_path") or "").strip()
    )
    operations = [
        item for item in (state.get("operations") or [])
        if isinstance(item, Mapping)
    ]
    applied_registry_targets = {
        str(item.get("target") or "").strip()
        for item in (state.get("registry_mutations_applied") or [])
        if isinstance(item, Mapping)
        and str(item.get("action") or "") == "update_tool_schema"
    }
    verification = state.get("verifier_config")
    verification = verification.get("verification", {}) if isinstance(verification, Mapping) else {}
    policy = state.get("policy") if isinstance(state.get("policy"), Mapping) else {}
    missing_operations = []
    source_proof = source_application_proof(str(state.get("snapshot_id") or ""), state.get("source_overrides") or {})
    for operation in operations:
        action = str(operation.get("action") or "").strip()
        target = str(operation.get("target") or "").strip()
        applied = False
        if action == "update_tool_schema":
            applied = target in applied_registry_targets
        elif action == "edit_tool_source":
            applied = source_proof.get("applied") and target in source_proof.get("targets", [])
        elif action == "add_composition_edge":
            applied = bool(
                policy.get("require_option_comparison_all_tasks")
                and isinstance(verification, Mapping)
                and verification.get("require_option_comparison_all_tasks")
            )
        elif action == "update_recovery_policy":
            applied = isinstance(policy.get("low_signal_temporal_grounding"), Mapping)
        if not applied:
            missing_operations.append({"action": action, "target": target})
    runtime_hook_applied = bool(snapshot_loaded and operations and not missing_operations)
    return {
        "snapshot_loaded": snapshot_loaded,
        "runtime_hook_applied": runtime_hook_applied,
        "operation_count": len(operations),
        "missing_operations": missing_operations,
        "proof_scope": "runtime_boundary_attestation",
        "requires_fresh_inference": False,
        "source_application": source_proof,
    }


def harness_audit_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return compact proof recorded in run_result."""

    application_proof = _snapshot_application_proof(state)
    return {
        "enabled": bool(state.get("enabled")),
        "status": str(state.get("status") or "baseline"),
        "snapshot_id": str(state.get("snapshot_id") or "snapshot_000"),
        "snapshot_path": str(state.get("snapshot_path") or ""),
        "policy_digest": str(state.get("policy_digest") or ""),
        "base_tool_library_baseline_id": str(state.get("base_tool_library_baseline_id") or ""),
        "operations_applied": copy.deepcopy(list(state.get("operations") or [])),
        "registry_mutations_applied": copy.deepcopy(list(state.get("registry_mutations_applied") or [])),
        "tool_overrides": copy.deepcopy(dict(state.get("tool_overrides") or {})),
        "snapshot_loaded": bool(application_proof["snapshot_loaded"]),
        "snapshot_application_proven": bool(application_proof["runtime_hook_applied"]),
        "snapshot_application_proof": application_proof,
        "mutation_declared": bool(state.get("enabled")) and bool(state.get("operations")),
        # Compatibility alias; unlike the old field, this means the runtime
        # hook was actually materialized, not merely declared in the snapshot.
        "mutation_applied": bool(application_proof["runtime_hook_applied"]),
    }
