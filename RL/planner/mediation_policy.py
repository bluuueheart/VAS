"""Helpers for root planner -> subagent mediation policy."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


RULE_DIRECT_EXECUTE = "rule_1_direct_execute"
RULE_LOCKED_TARGET = "rule_2_locked_target_rewrite"
RULE_ROUTE_FIRST = "rule_3_or_4_route_first"
LEVEL_DIRECT_EXECUTE = "L1"
LEVEL_LOCKED_TARGET = "L2"
LEVEL_ROUTE_FIRST = "L3 or L4"

_COMPOSITE_INTENT_DIMENSION_PATTERNS = {
    "action": (
        "action",
        "movement",
        "motion",
        "gesture",
        "pose",
        "dance",
        "doing",
    ),
    "style": (
        "style",
        "genre",
        "type",
        "category",
        "kind",
        "classify",
        "classification",
    ),
    "audio_caption": (
        "audio",
        "sound",
        "music",
        "subtitle",
        "caption",
        "transcript",
        "lyrics",
        "spoken",
        "say",
        "text on screen",
        "ocr",
    ),
    "scene_distinction": (
        "scene",
        "background",
        "setting",
        "compare",
        "difference",
        "distinguish",
        "versus",
        "vs",
        "while",
        "transition",
        "before",
        "after",
        "between",
    ),
}


def mediation_level_from_policy(policy: Dict[str, Any]) -> str:
    rule = str((policy or {}).get("rule") or "").strip()
    if rule == RULE_DIRECT_EXECUTE:
        return LEVEL_DIRECT_EXECUTE
    if rule == RULE_LOCKED_TARGET:
        return LEVEL_LOCKED_TARGET
    return LEVEL_ROUTE_FIRST


def _normalized_executable_args(args: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    return {
        key: value
        for key, value in args.items()
        if not str(key).startswith("__")
    }


def supports_direct_execution(tool: Any, args: Dict[str, Any]) -> bool:
    """Return True when the tool exists and normalized args satisfy its schema."""
    if tool is None or bool(getattr(tool, "is_virtual", False)):
        return False
    name = str(getattr(tool, "name", "") or "").strip()
    if name == "Parallel":
        return False

    schema = getattr(tool, "parameters", {}) if isinstance(getattr(tool, "parameters", {}), dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
    executable_args = _normalized_executable_args(args)

    if properties and any(key not in properties for key in executable_args):
        return False
    for field in required:
        if executable_args.get(field) in (None, "", [], {}):
            return False

    validate_args = getattr(tool, "validate_args", None)
    if callable(validate_args):
        try:
            validate_args(**executable_args)
        except Exception:
            return False
    return True


def classify_root_action(
    *,
    tool_name: str,
    tool: Any,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify a root planner action into the fixed mediation priority."""
    normalized_name = str(tool_name or "").strip()
    normalized_args = _normalized_executable_args(args)
    tool_exists = tool is not None and not bool(getattr(tool, "is_virtual", False))

    if tool_exists and supports_direct_execution(tool, normalized_args):
        return {
            "rule": RULE_DIRECT_EXECUTE,
            "tool_exists": True,
            "direct_executable": True,
            "target_tool_name": normalized_name,
            "allowed_resolution_modes": [],
            "locked_target": False,
        }

    if tool_exists:
        return {
            "rule": RULE_LOCKED_TARGET,
            "tool_exists": True,
            "direct_executable": False,
            "target_tool_name": normalized_name,
            "allowed_resolution_modes": [
                "parameter_rewrite_only",
                "single_replacement_if_semantically_incompatible",
                "parallel_decomposition_if_single_tool_is_insufficient",
            ],
            "locked_target": True,
        }

    return {
        "rule": RULE_ROUTE_FIRST,
        "tool_exists": False,
        "direct_executable": False,
        "target_tool_name": "",
        "allowed_resolution_modes": [
            "single_replacement",
            "parallel_decomposition",
        ],
        "locked_target": False,
    }


def schema_lines_from_tools(tool_schemas: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for schema in tool_schemas:
        if not isinstance(schema, dict):
            continue
        name = str(schema.get("name", "")).strip()
        if not name:
            continue
        parameters = schema.get("parameters", {}) if isinstance(schema.get("parameters"), dict) else {}
        properties = parameters.get("properties", {}) if isinstance(parameters.get("properties"), dict) else {}
        required_fields = [
            str(field).strip()
            for field in (schema.get("required") or parameters.get("required") or [])
            if str(field).strip()
        ]
        required_set = set(required_fields)
        param_specs: List[str] = []
        for param_name, property_schema in properties.items():
            property_payload = property_schema if isinstance(property_schema, dict) else {}
            raw_type = property_payload.get("type")
            if isinstance(raw_type, list):
                type_names = [str(item).strip() for item in raw_type if str(item).strip()]
                type_text = "|".join(type_names) if type_names else "any"
            else:
                type_text = str(raw_type).strip() or "any"
            required_suffix = "*" if str(param_name) in required_set else "?"
            param_specs.append(f"{param_name}:{type_text}{required_suffix}")
        required_text = ", ".join(required_fields) if required_fields else "none"
        desc = str(schema.get("description", "")).strip()
        params_text = ", ".join(param_specs) if param_specs else "none"
        line = f"- {name}: required={required_text}; params={params_text}"
        if desc:
            line += f"; {desc}"
        lines.append(line)
    return "\n".join(lines)


def route_plan_summary_lines(candidate_tools: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in candidate_tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        route_family = str(item.get("route_family", "")).strip() or "other"
        route_plan = [
            str(step).strip()
            for step in (item.get("route_plan") or [])
            if str(step).strip()
        ]
        route_reason = str(item.get("route_reason", "")).strip()
        plan_text = " -> ".join(route_plan) if route_plan else route_family
        line = f"- {name}: family={route_family}; route_plan={plan_text}"
        if route_reason:
            line += f"; reason={route_reason}"
        lines.append(line)
    return "\n".join(lines)


def _truncate_prompt_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _collect_root_request_text(action_args: Dict[str, Any], description_text: str) -> str:
    chunks: List[str] = []
    if description_text:
        chunks.append(str(description_text))
    if isinstance(action_args, dict):
        for key in ("query", "question", "prompt", "description", "goal", "task"):
            value = action_args.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
    return " ".join(chunks).strip()


def _detect_composite_intent_guidance(
    *,
    action_args: Dict[str, Any],
    description_text: str,
) -> Optional[Dict[str, Any]]:
    combined_text = _collect_root_request_text(action_args, description_text)
    normalized = re.sub(r"\s+", " ", combined_text.lower()).strip()
    if not normalized:
        return None

    detected_dimensions: List[str] = []
    for dimension, keywords in _COMPOSITE_INTENT_DIMENSION_PATTERNS.items():
        for keyword in keywords:
            if keyword in normalized:
                detected_dimensions.append(dimension)
                break

    if len(detected_dimensions) < 2:
        return None

    return {
        "recommended_resolution": "parallel_decomposition",
        "detected_dimensions": detected_dimensions,
        "reason": (
            "The planner request appears to mix multiple evidence dimensions. "
            "Prefer L4 child actions that probe those dimensions in parallel instead of collapsing "
            "the task into a single holistic tool too early."
        ),
    }


def compact_tool_schema_for_prompt(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}

    compact: Dict[str, Any] = {}
    name = str(schema.get("name", "")).strip()
    if name:
        compact["name"] = name

    description = _truncate_prompt_text(schema.get("description", ""), limit=320)
    if description:
        compact["description"] = description

    route_family = str(schema.get("route_family", "")).strip()
    if route_family:
        compact["route_family"] = route_family

    route_reason = _truncate_prompt_text(schema.get("route_reason", ""), limit=220)
    if route_reason:
        compact["route_reason"] = route_reason

    route_plan = [
        str(step).strip()
        for step in (schema.get("route_plan") or [])
        if str(step).strip()
    ]
    if route_plan:
        compact["route_plan"] = route_plan[:4]

    parameters = schema.get("parameters", {}) if isinstance(schema.get("parameters"), dict) else {}
    required = [
        str(field).strip()
        for field in (parameters.get("required") or [])
        if str(field).strip()
    ]
    if required:
        compact["required"] = required

    properties = parameters.get("properties", {}) if isinstance(parameters.get("properties"), dict) else {}
    compact_properties: Dict[str, Dict[str, Any]] = {}
    for key, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue
        compact_property: Dict[str, Any] = {}
        property_type = str(property_schema.get("type", "")).strip()
        if property_type:
            compact_property["type"] = property_type
        property_description = _truncate_prompt_text(
            property_schema.get("description", ""),
            limit=180,
        )
        if property_description:
            compact_property["description"] = property_description
        enum_values = property_schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            compact_property["enum"] = [
                str(item).strip()
                for item in enum_values[:8]
                if str(item).strip()
            ]
        compact_properties[str(key)] = compact_property or {"type": "any"}

    if compact_properties:
        compact["parameters"] = {"properties": compact_properties}
    return compact


def compact_tool_schemas_for_prompt(tool_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for schema in tool_schemas:
        compact_schema = compact_tool_schema_for_prompt(schema)
        if compact_schema:
            compacted.append(compact_schema)
    return compacted


def build_root_mediation_context(
    *,
    action_name: str,
    action_args: Dict[str, Any],
    description_text: str,
    policy: Dict[str, Any],
    discovered_tools_desc: str = "",
    target_schema: Optional[Dict[str, Any]] = None,
    candidate_tools: Optional[List[Dict[str, Any]]] = None,
    execution_hint: Optional[Dict[str, Any]] = None,
) -> str:
    candidate_tools = list(candidate_tools or [])
    compact_target_schema = compact_tool_schema_for_prompt(target_schema) if target_schema else None
    compact_candidate_tools = compact_tool_schemas_for_prompt(candidate_tools)
    composite_intent_guidance = _detect_composite_intent_guidance(
        action_args=action_args,
        description_text=description_text,
    )
    sections: List[str] = [
        "[Sub-Agent Mode]\nplanner_action_mediation",
        "[Sub-Agent Return Mode]\nresult_only",
        f"[Mediation Rule]\n{policy.get('rule', RULE_ROUTE_FIRST)}",
        f"[Mediation Level]\n{mediation_level_from_policy(policy)}",
        "[Allowed Resolution Modes]\n"
        + json.dumps(policy.get("allowed_resolution_modes", []), ensure_ascii=False, indent=2),
        f"[Planner Action Name]\n{action_name}",
        "[Planner Action Description]\n"
        + (description_text or "No description provided by planner."),
        "[Planner-Supplied Parameters]\n"
        + json.dumps(action_args or {}, ensure_ascii=False, indent=2),
        "[Abstract Tool Description]\n"
        + (description_text or "No description provided by planner."),
        "[Planner-Facing Observation Contract]\n"
        + "Return only executable concrete Actions. "
        + "If execution succeeds, the parent planner receives normalized tool results automatically and the successful result does not stay in the sub-agent loop. "
        + "Only failed executions may be returned to the sub-agent for another repair step within budget. "
        + "Do not emit Finish in this mode. "
        + "A top-level `Result` note is optional, but it never replaces executable Actions.",
    ]

    if policy.get("rule") == RULE_LOCKED_TARGET:
        sections.append(
            "[Execution Priority Contract]\n"
            "Execution priority L2 applies. Keep the target tool fixed and rewrite parameters first. "
            "Only escalate to single replacement or parallel decomposition if you judge the target tool "
            "is semantically incompatible or cannot complete the task alone."
        )
        sections.append(
            "[Locked Target Tool]\n"
            + str(policy.get("target_tool_name") or action_name)
        )
    else:
        sections.append(
            "[Execution Priority Contract]\n"
            "Execution priority L3/L4 applies. Use route metadata and candidate tools to decide whether to replace the abstract tool "
            "with one executable tool or decompose it into multiple executable tools."
        )

    if discovered_tools_desc:
        sections.append(f"[Discovered Tools from parent]\n{discovered_tools_desc}")
    if compact_target_schema:
        sections.append(
            "[Target Tool Schema]\n"
            + json.dumps(compact_target_schema, ensure_ascii=False, indent=2)
        )
    if compact_candidate_tools:
        sections.append(
            "[Candidate Tool Schemas]\n"
            + json.dumps(compact_candidate_tools, ensure_ascii=False, indent=2)
        )
        schema_lines = schema_lines_from_tools(compact_candidate_tools)
        if schema_lines:
            sections.append(f"[Known Tool Schemas]\n{schema_lines}")
        route_lines = route_plan_summary_lines(compact_candidate_tools)
        if route_lines:
            sections.append(f"[Route Plan Candidates]\n{route_lines}")
    if execution_hint:
        sections.append(
            "[Schema-Adapted Execution Hint]\n"
            + json.dumps(execution_hint, ensure_ascii=False, indent=2)
        )
    if composite_intent_guidance:
        sections.append(
            "[Composite Intent Guidance]\n"
            + json.dumps(composite_intent_guidance, ensure_ascii=False, indent=2)
        )

    return "\n\n".join(section for section in sections if section)
