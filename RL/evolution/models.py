"""Typed contracts shared by verification and evolution."""
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field


def normalize_mutation_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove empty alternative containers, never resolve conflicting edits."""
    normalized = dict(payload)
    for key in ("mutation", "mutations"):
        if normalized.get(key) in (None, [], {}):
            normalized.pop(key, None)
    return normalized


def valid_recovery_policy(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"enabled", "max_text_failures", "coverage_density_seconds", "max_recovery_calls"}:
        return False
    def bounded(key, low, high, integer=False):
        item = value[key]
        return not isinstance(item, bool) and isinstance(item, int if integer else (int, float)) and low <= item <= high
    return isinstance(value["enabled"], bool) and bounded("max_text_failures", 1, 8, True) and bounded("coverage_density_seconds", 5, 120) and bounded("max_recovery_calls", 1, 4, True)


class VerificationResult(BaseModel):
    progress_score: float = Field(ge=0.0, le=1.0)
    claim_coverage: float = Field(ge=0.0, le=1.0)
    direct_grounding: float = Field(ge=0.0, le=1.0)
    contradiction_score: float = Field(ge=0.0, le=1.0)
    cross_modal_support: float = Field(ge=0.0, le=1.0)
    can_finish: bool
    # True means the available, query-scoped evidence explicitly says the
    # question's premise is absent or the listed choices cannot represent the
    # observed value. This is a data/evidence boundary, not permission to pick
    # a convenient option.
    premise_mismatch: bool = False
    # True means verified structured evidence yields a value/order that no
    # listed option can represent. This is distinct from ordinary missing
    # evidence and must not be resolved by selecting the nearest label.
    option_mismatch: bool = False
    missing_information: List[str] = Field(default_factory=list)
    unresolved_claims: List[str] = Field(default_factory=list)
    # Query-derived claim contract shared with Planner/Observer/Reflector.
    # This is an audit explanation and never upgrades evidence on its own.
    claim_contract: Dict[str, Any] = Field(default_factory=dict)
    feedback: str = ""


class FailureType(str, Enum):
    PLANNING = "planning"
    TOOL_ROUTING = "tool_routing"
    ARGUMENT_GENERATION = "argument_generation"
    TOOL_EXECUTION = "tool_execution"
    TEMPORAL_LOCALIZATION = "temporal_localization"
    SAMPLING_INSUFFICIENT = "sampling_insufficient"
    MODALITY_SELECTION = "modality_selection"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    EVIDENCE_INTEGRATION = "evidence_integration"
    CONTRADICTION_HANDLING = "contradiction_handling"
    PREMATURE_TERMINATION = "premature_termination"
    ANSWER_SYNTHESIS = "answer_synthesis"


class FailureDiagnosis(BaseModel):
    diagnosis_id: str
    trajectory_ids: List[str]
    failure_type: FailureType
    blamed_component: str
    blamed_tool_ids: List[str] = Field(default_factory=list)
    supporting_event_ids: List[str] = Field(default_factory=list)
    root_cause: str
    counterfactual_fix: str
    confidence: float = Field(ge=0.0, le=1.0)
    # The same trajectory can have a wrong final answer and a local tool
    # failure, or a tool failure while the answer happens to be correct.  Keep
    # that distinction explicit so Evolver does blame attribution before it
    # chooses a mutation surface.
    failure_scope: str = "global_answer_failure"
    attribution: Dict[str, Any] = Field(default_factory=dict)


class EvolutionAction(str, Enum):
    UPDATE_ARGUMENT_POLICY = "update_argument_policy"
    UPDATE_TOOL_SCHEMA = "update_tool_schema"
    UPDATE_ROUTER_RULE = "update_router_rule"
    ADD_FALLBACK = "add_fallback"
    ADD_COMPOSITION_EDGE = "add_composition_edge"
    UPDATE_RECOVERY_POLICY = "update_recovery_policy"
    EDIT_TOOL_SOURCE = "edit_tool_source"


# This is the public capability surface given to the online Evolver cycle. Keep
# it in the shared contract module so the prompt, candidate preflight, and
# audit do not silently drift apart.
MAX_EVOLVER_OPERATIONS = 4
SOURCE_BACKED_ACTIONS = {"update_argument_policy": "update", "update_router_rule": "update", "add_fallback": "fallback"}
EVOLVER_HARNESS_ACTIONS = (
    *SOURCE_BACKED_ACTIONS,
    EvolutionAction.UPDATE_TOOL_SCHEMA.value,
    EvolutionAction.ADD_COMPOSITION_EDGE.value,
    EvolutionAction.UPDATE_RECOVERY_POLICY.value,
    EvolutionAction.EDIT_TOOL_SOURCE.value,
)


# The reference design treats coordinated edits as a graph problem: the
# Evolver must identify which runtime surfaces depend on one another before it
# emits a joint patch.  Keep the graph small and explicit because these are
# the only surfaces an isolated RegistrySnapshot can actually change.
EVOLVER_COMPONENT_DEPENDENCY_GRAPH: Dict[str, Dict[str, Any]] = {
    "tool_runtime": {
        "kind": "runtime",
        "requires": [],
        "downstream": ["evidence_acquisition"],
        "actions": [EvolutionAction.UPDATE_TOOL_SCHEMA.value, EvolutionAction.EDIT_TOOL_SOURCE.value, *SOURCE_BACKED_ACTIONS],
    },
    "evidence_acquisition": {
        "kind": "workflow",
        "requires": ["tool_runtime"],
        "downstream": ["workflow_composition"],
        "actions": [EvolutionAction.UPDATE_RECOVERY_POLICY.value],
    },
    "workflow_composition": {
        "kind": "workflow",
        "requires": ["evidence_acquisition"],
        "downstream": ["finish_gate"],
        "actions": [EvolutionAction.ADD_COMPOSITION_EDGE.value],
    },
    "finish_gate": {
        "kind": "verifier",
        "requires": ["workflow_composition"],
        "downstream": [],
        "actions": [],
    },
}


def evolver_dependency_graph() -> Dict[str, Dict[str, Any]]:
    """Return a serializable copy of the executable Evolver dependency graph."""

    return {
        str(component): {
            str(field): list(value) if isinstance(value, list) else value
            for field, value in definition.items()
        }
        for component, definition in EVOLVER_COMPONENT_DEPENDENCY_GRAPH.items()
    }


def analyze_evolver_dependencies(
    mutations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate and explain dependencies for one or two proposed operations.

    The analysis is deterministic and advisory to the LLM: action names,
    targets, duplicate edits, and conflicting edits are checked here before a
    candidate snapshot can be materialized.  Base runtime components are
    already present in the isolated harness, so a single composition-edge
    operation does not need a second operation merely to satisfy its graph
    prerequisites.
    """

    records: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    missing_dependencies: List[Dict[str, Any]] = []
    if not isinstance(mutations, Sequence) or isinstance(mutations, (str, bytes)):
        mutations = []
    for index, mutation in enumerate(mutations):
        if not isinstance(mutation, Mapping):
            records.append({"index": index, "action": "", "target_ids": [], "valid": False})
            conflicts.append({"type": "invalid_operation", "index": index})
            continue
        action = str(getattr(mutation.get("action"), "value", mutation.get("action") or "")).strip()
        raw_targets = mutation.get("target_ids")
        if isinstance(raw_targets, list):
            targets = [str(item).strip() for item in raw_targets if str(item).strip()]
        else:
            target = str(mutation.get("target") or "").strip()
            targets = [target] if target else []
        component = next(
            (
                name
                for name, definition in EVOLVER_COMPONENT_DEPENDENCY_GRAPH.items()
                if action in (definition.get("actions") or [])
            ),
            "",
        )
        records.append({
            "index": index,
            "action": action,
            "target_ids": targets,
            "component": component,
            "valid": bool(action in EVOLVER_HARNESS_ACTIONS and targets and component),
        })

    seen_operation_keys: Dict[str, int] = {}
    target_records: Dict[str, List[int]] = {}
    for record in records:
        operation_key = f"{record.get('action')}::{','.join(record.get('target_ids') or [])}"
        if operation_key in seen_operation_keys:
            conflicts.append({
                "type": "duplicate_operation",
                "index": record.get("index"),
                "other_index": seen_operation_keys[operation_key],
                "operation": operation_key,
            })
        else:
            seen_operation_keys[operation_key] = int(record.get("index", 0))
        for target in record.get("target_ids") or []:
            target_records.setdefault(str(target), []).append(int(record.get("index", 0)))

    # A target may be edited once in a coordinated patch.  Different actions
    # on the same target are also rejected because the snapshot has no ordered
    # transaction semantics with which to resolve them.
    for target, indices in target_records.items():
        if len(indices) < 2:
            continue
        unique_actions = sorted({str(records[index].get("action") or "") for index in indices})
        conflicts.append({
            "type": "conflicting_target",
            "target": target,
            "indices": indices,
            "actions": unique_actions,
        })

    # Build operation-level edges from the component graph.  A pair is
    # connected when one component is upstream of the other or when both
    # converge on the same downstream surface.  This makes the joint edit
    # auditable without pretending that the LLM can patch arbitrary source.
    dependency_edges: List[Dict[str, Any]] = []
    def _reachable(start: str, target: str) -> bool:
        if not start or not target:
            return False
        pending = [start]
        visited = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for downstream in EVOLVER_COMPONENT_DEPENDENCY_GRAPH.get(current, {}).get("downstream") or []:
                if downstream == target:
                    return True
                pending.append(str(downstream))
        return False

    def _all_downstream(start: str) -> set[str]:
        result: set[str] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            for downstream in EVOLVER_COMPONENT_DEPENDENCY_GRAPH.get(current, {}).get("downstream") or []:
                downstream = str(downstream)
                if downstream not in result:
                    result.add(downstream)
                    pending.append(downstream)
        return result

    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            left_component = str(left.get("component") or "")
            right_component = str(right.get("component") or "")
            left_downstream = _all_downstream(left_component)
            right_downstream = _all_downstream(right_component)
            relation = ""
            if _reachable(left_component, right_component):
                relation = "left_upstream"
            elif _reachable(right_component, left_component):
                relation = "right_upstream"
            elif left_downstream.intersection(right_downstream):
                relation = "shared_downstream"
            elif left_component and left_component == right_component:
                relation = "same_component"
            if relation:
                dependency_edges.append({
                    "from_index": left.get("index"),
                    "to_index": right.get("index"),
                    "from_component": left_component,
                    "to_component": right_component,
                    "relation": relation,
                })

    invalid_records = [record for record in records if not record.get("valid")]
    operation_count = len(records)
    joint_required = operation_count > 1
    edge_pairs = {
        tuple(sorted((int(edge["from_index"]), int(edge["to_index"]))))
        for edge in dependency_edges
    }
    all_pairs = {
        tuple(sorted((left_index, right_index)))
        for left_index in range(operation_count)
        for right_index in range(left_index + 1, operation_count)
    }
    if joint_required:
        for pair in sorted(all_pairs - edge_pairs):
            missing_dependencies.append({
                "type": "disconnected_operations",
                "indices": list(pair),
                "reason": "coordinated operations do not share a dependency path in the harness graph",
            })

    dependency_pass = bool(
        1 <= operation_count <= MAX_EVOLVER_OPERATIONS
        and not invalid_records
        and not conflicts
        and not missing_dependencies
    )
    return {
        "graph_version": 1,
        "operation_count": operation_count,
        "joint_required": joint_required,
        "nodes": records,
        "dependency_edges": dependency_edges,
        "missing_dependencies": missing_dependencies,
        "conflicts": conflicts,
        "dependency_pass": dependency_pass,
    }


def evolver_capability_manifest(acceptance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Describe only mutation surfaces understood by the isolated harness.

    The manifest is input to the Evolver, not an authority to bypass replay.
    Runtime behavior still has to be demonstrated by the candidate worker and
    the final metric gates.
    """

    return {
        "manifest_version": 5,
        "reference_design": {
            "source": "reference.md#Evolver设计",
            "pipeline": [
                "trajectory",
                "failure_diagnosis",
                "evolution_intent",
                "dependency_analysis",
                "joint_modify",
                "second_rollout",
                "verify_joint_modification",
            ],
            "joint_edit_enabled": True,
            "max_operations": MAX_EVOLVER_OPERATIONS,
            "same_snapshot_second_rollout_required": True,
            "audit_marker": "harness component dependency graph",
        },
        "dependency_graph": evolver_dependency_graph(),
        "common_tool_output_contract": {
            "required_fields": ["status", "result"],
            "status_values": ["success", "no_valid_result", "error"],
            "output_schema_mutation": "forbidden",
        },
        "actions": {
            **{action: {
                "target": "registered tool whose executable implementation is changed",
                "after_shape": {"kind": kind, "edits": "same exact path/digest/before/after contract as edit_tool_source"},
                "runtime_hook": "isolated source worker; real registry and runtime application proof",
                "limitation": "text-only policies are rejected; the proposal must implement the routing, fallback or argument transformation in tool code",
            } for action, kind in SOURCE_BACKED_ACTIONS.items()},
            "edit_tool_source": {
                "target": "exact tool name, or role name for a prompt edit",
                "after_shape": {"kind": "add|delete|update|compose|fallback|prompt", "edits": [{"path": "toolkit/impl/example.py", "before_sha256": "SHA256 of the complete original UTF-8 file; empty file hash for additions", "before": "exact unique source snippet", "after": "replacement source"}]},
                "allowed_paths": ["toolkit/impl/*.py", "planner/prompts.py", "config/agent_prompts.json"],
                "runtime_hook": "source_mutation.materialize_source_snapshot -> isolated Python worker -> source_application_proof",
                "limitations": "maximum four files; real backend/tool execution required; no evaluator or verification-code edits; no online source writes; compile and paired runtime checks required",
            },
            "update_tool_schema": {
                "target": "one active non-virtual registered tool",
                "after_fields": ["description", "parameters", "tags"],
                "parameters_shape": {
                    "type": "object",
                    "properties": "object",
                    "required": "array (optional)",
                },
                "runtime_hook": "harness.apply_harness_registry_overrides",
                "limitation": "schema cannot create a parameter, output field, or backend behavior; new properties require an explicit run() parameter",
            },
            "add_composition_edge": {
                "target": "candidate_comparison for the Finish gate, or a tool implementing an executable composition",
                "source_alternative": {"kind": "compose", "edits": "exact source edits implementing the dependency chain"},
                "after_shape": {
                    "after": "evidence_acquisition",
                    "before": "finish",
                    "required": True,
                },
                "runtime_hook": "harness.load_harness_snapshot -> EvidenceVerifier",
            },
            "update_recovery_policy": {
                "target": "low_signal_temporal_grounding",
                "after_shape": {
                    "enabled": True,
                    "max_text_failures": 2,
                    "coverage_density_seconds": 30.0,
                    "max_recovery_calls": 1,
                },
                "runtime_hook": (
                    "harness.load_harness_snapshot -> RecursiveEngine._prepare_concrete_tool_call "
                    "-> Inspect_Frame"
                ),
                "limitation": (
                    "bounded visual coverage calls: failures 1..8, density 5..120 seconds, calls 1..4; after text-retrieval "
                    "no_valid_result/error; it cannot use annotation timestamps or answer labels"
                ),
            },
        },
        "acceptance": dict(acceptance or {}),
        "failure_attribution": {
            "scope_values": [
                "global_answer_failure",
                "global_and_local_failure",
                "local_tool_failure",
                "local_verifier_failure",
                "efficiency",
            ],
            "rule": "attribute tool/runtime/verifier blame separately from final answer correctness",
        },
        "replay_contract": {
            "first_stage": "selective_suffix_replay",
            "reuse": "cached trajectory state before the first affected step",
            "runtime_suffix_required_for": ["update_tool_schema", "update_recovery_policy"],
            "runtime_execution": "fresh paired trigger rollouts when no resumable checkpoint is available; recorded as paired_runtime_rollout_executed, never as prefix reuse",
            "verifier_suffix_possible_for": ["add_composition_edge"],
            "not_executed_is_not_success": True,
            "matching_cases_and_periodic_regression_required": True,
        },
        "coordinated_patch": {
            "supported": True,
            "max_operations": MAX_EVOLVER_OPERATIONS,
            "encoding": "payload.mutations array; legacy payload.mutation remains supported",
            "allowed_surfaces": [
                "tool_schema_description_or_parameters",
                "runtime_recovery_policy",
                "workflow_composition",
            ],
            "source_tool_implementation": "exact source edits in isolated worker; real paired runtime and held-out gates required",
            "conflicting_targets": "rejected",
            "dependency_analysis": "required before materialization; operation pairs must be connected in dependency_graph",
        },
        "final_acceptance_requires": [
            "candidate snapshot applied in an isolated runtime",
            "causal selective/suffix replay or an explicitly recorded runtime limitation",
            "fresh candidate Planner/Observer/Reflector trajectory on matching cases",
            "minimum paired trigger/neighbor/regression/held-out groups are present",
            "all trigger/neighbor/regression/held-out/invalid-call/latency gates pass",
            "dependency analysis and joint replay verify every coordinated operation",
        ],
        "online_publish": "disabled until final acceptance",
    }


class PatchOperation(BaseModel):
    action: EvolutionAction
    target_type: str
    target_ids: List[str]
    before: Optional[Dict[str, Any]] = None
    after: Dict[str, Any]
    reason: str


class EvolutionIntent(BaseModel):
    intent_id: str
    diagnosis_ids: List[str]
    scope: Literal["single_tool", "tool_family", "router", "workflow", "global"]
    operations: List[PatchOperation]
    expected_effects: List[str]
    risks: List[str]
    dependency_analysis: Dict[str, Any] = Field(default_factory=dict)


class RegistrySnapshot(BaseModel):
    snapshot_id: str
    parent_snapshot_id: Optional[str] = None
    created_at: str
    router_policy: Dict[str, Any] = Field(default_factory=dict)
    argument_policy: Dict[str, Any] = Field(default_factory=dict)
    tool_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    source_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    source_materialized: bool = False
    mutation_delta: Dict[str, Any] = Field(default_factory=dict)
    base_tool_library_baseline_id: Optional[str] = None
    patch_ids: List[str] = Field(default_factory=list)
    status: Literal["candidate", "accepted", "rejected"]
    dependency_analysis: Dict[str, Any] = Field(default_factory=dict)
