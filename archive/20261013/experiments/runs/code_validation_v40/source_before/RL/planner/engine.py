"""Recursive engine - the brain of the system."""
import ast
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List, Mapping

from config.settings import settings
from config.device_config import is_npu_runtime_error
from core.llm import (
    AcceleratorFatalError,
    AnthropicProvider,
    BaseLLM,
    GenerationTimeoutError,
    OpenAIProvider,
    TransformersProvider,
    build_role_specific_provider_view,
    cache_existing_transformers_provider,
    get_llm_provider,
    is_accelerator_fatal_error,
)
from core.memory import ContextManager, MaxRecursionError, Trajectory
from core.answer_contract import normalize_answer_format
from agents import FourRoleAgentSuite
from evolution import EvidenceVerifier
from evolution.verifier import _slot_has_executed_source
from runtime.session import runtime_session
from toolkit.impl.visual import (
    prepare_planner_image_attachment,
    summarize_planner_image_evidence,
)
from toolkit.observation_utils import (
    build_planner_observation_record,
    build_parallel_observation,
    extract_planner_result_text,
    format_compact_observation,
)
from toolkit.base import normalize_tool_result
from toolkit.registry import registry
from .abstract_tool_resolution import (
    _normalize_field_aliases,
    build_abstract_signature,
    extract_candidate_tools,
    resolve_abstract_tool_call,
)
from .mediation_policy import (
    RULE_LOCKED_TARGET,
    build_root_mediation_context as build_root_mediation_context_payload,
    classify_root_action,
    compact_tool_schemas_for_prompt,
    route_plan_summary_lines,
    schema_lines_from_tools,
)
from .react_agent import ReactAgent
from .prompts import (
    get_recursive_prompt,
    get_runtime_control_tools_description,
    query_requires_evidence_first,
)

logger = logging.getLogger(__name__)

# Video_Clip_QA keeps a large multimodal generation working set.  Planner
# Parallel still records and orchestrates multiple independent calls, but a
# single reusable worker must not materialize several Qwen video generations
# simultaneously on one GPU/host.  The batch launcher enables this guard for
# the formal GPU run; leaving it opt-in keeps other deployments unchanged.
_GPU_VIDEO_CLIP_QA_EXECUTION_LOCK = threading.Lock()


# Action names that must never trigger recursive expansion - these come from
# prompt schema examples or retry-hint templates, not from real LLM decisions.
_PLACEHOLDER_ACTION_NAMES = frozenset({
    "none", "null", "undefined",
    "toolname", "concretetoolname", "concretetool",
    "tool_name", "actionname",
    "actual_tool", "actualexecutabletool",
    "actual_tool_name_here", "actualtoolnamehere",
    "replace_with_real_tool", "replacewithrealtool",
})

_APPROXIMATE_EVIDENCE_RE = re.compile(
    r"\b(about|around|approx(?:\.|imately)?|roughly|nearly|almost|circa)\b|~"
)
_COMPARISON_MARKERS = frozenset({
    "against", "alternative", "although", "but", "compare", "compared", "conflict",
    "comparison", "contradict", "different", "difference", "instead", "not", "other",
    "others", "rather", "versus", "vs", "whereas", "while",
})
_DISCOVERY_ACTIONS = frozenset({"Tool_Search", "Web_Search"})
_RETRIEVAL_ACTIONS = frozenset({
    "Semantic_Retrieve", "Keyword_Retrieve", "Segment_Retrieve",
    "Graph_Search", "Clean_Results", "Filtering", "Aggregation",
})
_VISUAL_ACTIONS = frozenset({
    "Inspect_Frame", "OCR_Extract", "Crop", "Zoom_in",
    "Entity_Track", "Action_Recognition", "Emotion_Analysis",
    "Audio_Event_Detect", "Speaker_Diarization", "Cross_Modal_Verify",
    "Video_Clip_QA",
})
_COMPUTE_ACTIONS = frozenset({
    "Python_Executor", "SQL_Executor", "Debt_Interest_Analyzer",
    "Amortization_Schedule", "Simple_Interest", "Compound_Growth",
    "Solve_Fixed_Step_Amount", "Sequence_Field_Total",
    "Field_Value_After_Steps", "Compare_Record_Sequences",
    "Percentage_Change",
})


def _is_find_needle_goal(goal: Any) -> bool:
    """Return whether the active query declares the MLVU Needle task type."""
    return bool(
        re.search(
            r"\b(?:task\s*type|task_type)\s*:\s*findneedle\b",
            str(goal or ""),
            flags=re.IGNORECASE,
        )
    )


def _normalize_candidate_probe_windows(value: Any) -> List[Dict[str, Any]]:
    """Normalize retrieval-provided candidate windows without answer fields."""
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        segment_ref = str(item.get("segment_ref") or "").strip()
        try:
            start = float(item.get("start_time"))
            end = float(item.get("end_time"))
        except (TypeError, ValueError):
            continue
        if not segment_ref or end <= start:
            continue
        key = f"{segment_ref}:{start:.3f}-{end:.3f}"
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "segment_ref": segment_ref,
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "time_range": str(item.get("time_range") or f"{start:.3f}-{end:.3f}"),
                "verification_required": True,
                "recommended_tools": ["Inspect_Frame", "Video_Clip_QA"],
            }
        )
    return normalized


def _candidate_probe_window_key(window: Mapping[str, Any]) -> str:
    return "{}:{:.3f}-{:.3f}".format(
        str(window.get("segment_ref") or "").strip(),
        float(window.get("start_time") or 0.0),
        float(window.get("end_time") or 0.0),
    )


def _normalize_model_identifier(model_name: Any) -> str:
    text = str(model_name or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def _action_family(action_name: str) -> str:
    name = str(action_name or "").strip()
    if name == "Finish":
        return "finish"
    if name == "Parallel":
        return "parallel"
    if name in _DISCOVERY_ACTIONS:
        return "discovery"
    if name in _RETRIEVAL_ACTIONS:
        return "retrieval"
    if name in _VISUAL_ACTIONS:
        return "visual"
    if name in _COMPUTE_ACTIONS:
        return "compute"
    return "other"


def _infer_evidence_status(
    value: str = "",
    citation: str = "",
    evidence_status: str = "",
) -> str:
    normalized_status = str(evidence_status or "").strip().lower()
    if normalized_status in {"verified", "exact", "approximate", "approx", "partial", "missing"}:
        if normalized_status in {"verified", "exact"} and (not value or not citation):
            return "approximate"
        if normalized_status in {"verified", "exact"}:
            return "verified"
        if normalized_status in {"approximate", "approx", "partial"}:
            return "approximate"
        return "missing"

    if normalized_status:
        missing_markers = ("missing", "unverified", "unsupported", "unresolved", "unknown", "not_found")
        if any(marker in normalized_status for marker in missing_markers):
            return "missing"
        # Never promote an unrecognized explicit status to verified.
        return "approximate" if value or citation else "missing"

    if not value and not citation:
        return "missing"
    if not value or not citation:
        return "approximate"
    text = f"{value} {citation}".lower()
    if _APPROXIMATE_EVIDENCE_RE.search(text):
        return "approximate"
    return "verified"


_ACTION_COUNT_LABEL_MARKERS = frozenset({
    "action", "added", "add", "put", "placed", "place", "removed", "remove",
    "inserted", "insert", "poured", "pour", "taken", "take", "picked", "pick",
})
_TOTAL_COUNT_LABEL_MARKERS = frozenset({
    "total", "existing", "already", "visible", "present", "remaining", "scene",
})
_COUNT_LABEL_MARKERS = frozenset({"count", "number", "quantity", "amount", "num"})
_TOTAL_EVIDENCE_MARKERS = frozenset({
    "already", "existing", "exists", "contain", "contains", "contained", "total",
    "altogether", "resulting", "afterward", "afterwards", "visible", "present",
})
_ACTION_QUERY_MARKERS = frozenset({
    "put", "puts", "place", "placed", "add", "adds", "added", "remove", "removes",
    "removed", "insert", "inserts", "inserted", "pour", "pours", "poured", "take",
    "takes", "took", "pick", "picks", "picked",
})
_EVIDENCE_SCOPES = frozenset({
    "action_count", "total_count", "state_count", "scene_count", "generic_count",
    "event", "dynamic_relation", "comparison", "qualifier", "generic", "claim",
})


def _evidence_slot_scope(slot: Dict[str, Any]) -> str:
    """Classify a count claim without deciding whether it is correct.

    The ledger previously keyed only on the literal label.  That made
    ``stick_count``, ``stick_count_total`` and ``action_count`` look like
    independent contradictions even when the latter explicitly disambiguated
    the former.  Scope is metadata for reconciliation; it never promotes a
    missing/approximate claim to verified.
    """
    label = str(slot.get("label", "")).strip().lower()
    claim = " ".join(
        str(slot.get(key, "") or "").strip().lower()
        for key in ("label", "value", "citation")
    )
    label_tokens = set(re.findall(r"[a-z0-9]+", label))
    has_count = bool(label_tokens.intersection(_COUNT_LABEL_MARKERS)) or "count" in claim
    has_action_label = bool(label_tokens.intersection(_ACTION_COUNT_LABEL_MARKERS))
    has_total_label = bool(label_tokens.intersection(_TOTAL_COUNT_LABEL_MARKERS))
    direct_action = bool(re.search(
        r"\b(?:put|puts|place|places|placed|add|adds|added|remove|removes|removed|insert|inserts|inserted|"
        r"pour|pours|poured|take|takes|took|pick|picks|picked)\b\s+"
        r"(?:exactly\s+)?(?:(?:a|an)\s+)?(?:single|one|two|three|four|five|six|seven|eight|nine|ten|\d+)",
        claim,
        re.IGNORECASE,
    ))
    if (has_action_label and has_count) or direct_action:
        return "action_count"
    if has_total_label or bool(label_tokens.intersection({"count", "number"})) and any(
        marker in claim for marker in _TOTAL_EVIDENCE_MARKERS
    ):
        return "total_count"
    if has_count:
        return "generic_count"
    return "claim"


def _query_asks_action_count(query: str) -> bool:
    normalized = str(query or "").lower()
    if not any(marker in normalized for marker in ("how many", "number of", "count")):
        return False
    tokens = set(re.findall(r"[a-z]+", normalized))
    if tokens.intersection(_ACTION_QUERY_MARKERS):
        return True
    return bool(re.search(
        r"\b(?:instance|instances|occurrence|occurrences)\b[^.?!]{0,60}\b"
        r"(?:action|actions|event|events|scene|scenes)\b",
        normalized,
    ))


def _slot_status(slot: Dict[str, Any]) -> str:
    return str(slot.get("evidence_status", "missing") or "missing").strip().lower()


def _slot_is_active(slot: Dict[str, Any]) -> bool:
    return not bool(slot.get("superseded", False))


def _extract_evidence_slots(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_slots: List[Any] = []
    top_level = payload.get("Evidence", payload.get("evidence"))
    if isinstance(top_level, list):
        raw_slots.extend(top_level)
    finish_payload = payload.get("Finish", payload.get("finish"))
    if isinstance(finish_payload, dict):
        nested = finish_payload.get("Evidence", finish_payload.get("evidence"))
        if isinstance(nested, list):
            raw_slots.extend(nested)
    if not raw_slots:
        return []

    extracted: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_slots, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip() or f"slot_{idx}"
        value = str(item.get("value", "")).strip()
        citation = str(item.get("citation", "")).strip()
        status = _infer_evidence_status(
            value=value or label,
            citation=citation,
            evidence_status=str(item.get("evidence_status", "")).strip(),
        )
        uncertainty = str(item.get("uncertainty") or "").strip().lower()
        if status == "verified" and any(
            marker in uncertainty
            for marker in (
                "approx", "uncertain", "unclear", "unknown", "insufficient",
                "ambiguous", "maybe", "possibly", "not sure", "low confidence",
                "medium confidence",
            )
        ):
            # An explicit uncertainty field is stronger than the presence of
            # a value/citation.  Do not let an Observer mark an approximate
            # visual count as verified merely because it copied a timestamp.
            status = "approximate"
        explicit_scope = str(item.get("scope") or "").strip().lower()
        scope = explicit_scope if explicit_scope in _EVIDENCE_SCOPES else _evidence_slot_scope({
            "label": label,
            "value": value,
            "citation": citation,
        })
        extracted.append({
            "label": label,
            "value": value,
            "citation": citation,
            "evidence_status": status,
            "logical_conflict": bool(
                item.get("logical conflict", item.get("logical_conflict", False))
            ),
            # Preserve an agent-declared scope as provenance metadata, but
            # let the deterministic verifier decide whether that scope is
            # actually supported by the tool evidence. Dropping it here
            # turned explicit action_count claims into generic_count claims
            # and made later audits unable to distinguish the two.
            "scope": scope,
        })
        # Preserve the shared claim contract across Planner -> Observer ->
        # Reflector -> Verifier. These fields are provenance, not evidence
        # by themselves; the deterministic verifier still requires a
        # successful cited observation before a claim can satisfy a gate.
        for key in (
            "claim_id", "target", "relation", "direction", "qualifiers",
            "uncertainty", "observed_value", "source_tool", "time_window",
        ):
            if key in item and item.get(key) not in (None, "", [], {}):
                # `extracted` is the outer list.  Write provenance fields to
                # the slot just appended above; indexing the list by a
                # string here caused every claim-bearing Planner/Observer
                # response to fail with ``list indices must be integers``.
                extracted[-1][key] = item.get(key)
    return extracted


def _native_fallback_evidence_slots(
    *,
    goal: str,
    observation_items: Any,
) -> List[Dict[str, Any]]:
    """Preserve structured native answers when the Observer JSON is invalid.

    Observer is an auxiliary LLM role.  A malformed Observer response must not
    erase a successful executable visual answer from the trajectory.  This
    fallback copies only fields emitted by Video_Clip_QA/Action_Recognition;
    it never infers facts from option labels, Python narration, or frame paths.
    Specialized verifier gates still decide whether the copied claim is enough
    to finish.
    """
    semantic = str(goal or "").lower()
    question_match = re.search(r"(?is)\bquestion\s*:\s*(.*)", semantic)
    if question_match:
        semantic = question_match.group(1)
    semantic = re.split(r"(?im)^\s*(?:options?|choices?)\s*:\s*", semantic, maxsplit=1)[0]
    semantic = re.split(r"(?im)^\s*reply\s+with\b", semantic, maxsplit=1)[0]
    count_query = bool(re.search(r"\b(?:how\s+many|number\s+of|count|total)\b", semantic))
    action_count_query = count_query and bool(
        re.search(
            r"\b(?:put|place|add|remove|insert|pour|take|pick)\w*\b",
            semantic,
        )
        or re.search(
            r"\b(?:instance|instances|occurrence|occurrences)\b"
            r"[^.!?]{0,60}\b(?:action|event|scene)\b",
            semantic,
        )
    )
    comparison_query = bool(re.search(
        r"\b(?:different|difference|compare|comparison|versus|vs|other|deviation|impact)\b",
        semantic,
    ))
    negative_query = bool(re.search(
        r"\b(?:not|never|without|does\s+not|doesn't|did\s+not|didn't|absent)\b",
        semantic,
    ))
    rationale_query = bool(re.search(
        r"\b(?:purpose|function|why|importance|important|significance|"
        r"reason|inferred|contribute|demonstrate|impact)\b",
        semantic,
    ))

    def _flatten(value: Any) -> List[Mapping[str, Any]]:
        if isinstance(value, Mapping):
            records: List[Mapping[str, Any]] = []
            if str(value.get("tool") or "").strip():
                records.append(value)
            for key in ("items", "result", "payload"):
                records.extend(_flatten(value.get(key)))
            return records
        if isinstance(value, list):
            records = []
            for item in value:
                records.extend(_flatten(item))
            return records
        return []

    def _mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except Exception:
                return {}
            return parsed if isinstance(parsed, Mapping) else {}
        return {}

    def _scalar_answer(value: Any) -> str:
        current = value
        seen: set[str] = set()
        for _ in range(4):
            if isinstance(current, Mapping):
                current = (
                    current.get("answer")
                    or current.get("answer_value")
                    or current.get("value")
                    or current.get("text")
                )
                continue
            text = str(current or "").strip()
            if not text:
                return ""
            if text.startswith(("{", "[")):
                if text in seen:
                    return ""
                seen.add(text)
                current = _mapping(text)
                if current:
                    continue
            return text
        return ""

    def _list_text(value: Any) -> List[str]:
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        if value not in (None, ""):
            return [str(value).strip()]
        return []

    def _range_text(item: Mapping[str, Any]) -> str:
        for key in (
            "effective_time_range", "requested_time_range", "search_window",
            "time_range", "timestamp_range",
        ):
            value = item.get(key)
            if isinstance(value, Mapping):
                start = value.get("start", value.get("start_time", value.get("t_start")))
                end = value.get("end", value.get("end_time", value.get("t_end")))
                if start not in (None, "") or end not in (None, ""):
                    return f"{start}-{end}"
            elif value not in (None, ""):
                return str(value)
        if item.get("t_start") is not None or item.get("t_end") is not None:
            return f"{item.get('t_start')}-{item.get('t_end')}"
        return ""

    native_tool_names = {
        "video_clip_qa": "Video_Clip_QA",
        "action_recognition": "Action_Recognition",
    }
    slots: List[Dict[str, Any]] = []
    for index, item in enumerate(_flatten(observation_items), start=1):
        raw_tool = str(item.get("tool") or "").strip()
        tool = native_tool_names.get(raw_tool.casefold(), raw_tool)
        if tool not in set(native_tool_names.values()):
            continue
        status = str(item.get("status") or "success").strip().lower()
        if status != "success":
            continue
        result_payload = _mapping(item.get("result"))
        parsed_payload = _mapping(item.get("parsed_json"))
        if not parsed_payload:
            parsed_payload = _mapping(result_payload.get("parsed_json"))
        answer = _scalar_answer(
            parsed_payload.get("answer")
            if parsed_payload
            else result_payload.get("answer")
            if result_payload
            else item.get("answer_value")
        )
        if not answer:
            answer = _scalar_answer(
                item.get("answer_value")
                or result_payload.get("answer_value")
                or result_payload.get("answer")
            )
        if not answer:
            continue
        payload = parsed_payload or result_payload
        evidence = _list_text(
            payload.get("evidence")
            or payload.get("evidence_items")
            or item.get("evidence_items")
        )
        summary = _scalar_answer(
            payload.get("video_summary")
            or payload.get("video_summary_value")
            or item.get("video_summary_value")
        )
        uncertainty = _scalar_answer(
            payload.get("uncertainty")
            or payload.get("uncertainty_value")
            or item.get("uncertainty_value")
        )
        uncertainty_text = uncertainty.lower()
        evidence_status = (
            "approximate"
            if not uncertainty
            or uncertainty_text in {"none", "null", "n/a"}
            or any(marker in uncertainty_text for marker in (
                "uncertain", "unknown", "unclear", "ambiguous", "maybe",
                "possibly", "insufficient", "not sure",
            ))
            else "verified"
        )
        if uncertainty_text in {"none", "null", "n/a"}:
            evidence_status = "verified"
        if count_query:
            scope = "action_count" if action_count_query else "scene_count"
        elif comparison_query or negative_query:
            scope = "comparison"
        elif rationale_query:
            scope = "generic"
        else:
            scope = "event"
        range_text = _range_text(item)
        citation = f"{tool} native answer"
        if range_text:
            citation += f" ({range_text})"
        claim_text = answer
        if evidence:
            claim_text += "; " + "; ".join(evidence[:2])
        if summary and summary not in claim_text:
            claim_text += f"; {summary}"
        digest = hashlib.sha1(
            f"{tool}|{range_text}|{answer}|{index}".encode("utf-8")
        ).hexdigest()[:12]
        slots.append({
            "claim_id": f"native:{tool.lower()}:{digest}",
            "scope": scope,
            "label": "native answer",
            "target": semantic[:160],
            "relation": "answers",
            "direction": "positive",
            "qualifiers": [],
            "observed_value": answer,
            "value": claim_text,
            "citation": citation,
            "source_tool": tool,
            "time_window": range_text,
            "uncertainty": uncertainty or "None",
            "evidence_status": evidence_status,
            "logical_conflict": False,
        })
    return slots


def _bind_observer_visual_sources(slots: List[Dict[str, Any]], observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bind relative frame citations to the actual image-producing tool.

    Retrieved frames can be visually read by Observer without an Inspect_Frame
    call. Preserve that real retrieval provenance instead of inventing a call.
    """
    sources: set[str] = set()
    def contains_images(value: Any) -> bool:
        if isinstance(value, list):
            return any(contains_images(item) for item in value)
        if not isinstance(value, Mapping) or str(value.get("status") or "success").strip().lower() != "success":
            return False
        return bool(value.get("image_evidence")) or any(contains_images(value.get(key)) for key in ("items", "probes", "matched_segments", "candidate_segments"))
    def collect_sources(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                collect_sources(child)
            return
        if not isinstance(value, Mapping):
            return
        status = str(value.get("status") or "").strip().lower()
        if status and status != "success":
            return
        tool = str(value.get("tool") or "").strip()
        if tool and tool not in {"Parallel", "System", "Assistant_Note"} and status == "success" and contains_images(value):
            sources.add(tool)
        for key in ("items", "probes", "result", "payload"):
            collect_sources(value.get(key))

    collect_sources(observations)
    if len(sources) != 1:
        return slots
    source = next(iter(sources))
    bound = []
    for slot in slots:
        item = dict(slot)
        if (str(item.get("source_tool") or "") not in sources
                and re.search(r"\bframe|\bpack[_ ]|top.left|top.right|bottom.left|bottom.right", str(item.get("citation") or ""), re.I)):
            item["declared_source_tool"] = item.get("source_tool")
            item["source_tool"] = source
            item["source_binding"] = "observer_attached_visual_observation"
        bound.append(item)
    return bound


def _canonical_evidence_slot_key(slot: Mapping[str, Any], fallback: str) -> str:
    """Return a stable semantic identity for one active evidence slot.

    Planner/Observer wording is intentionally free-form.  Treating labels such
    as ``social activity evidence`` and ``social_interaction`` as different
    slots lets stale claims survive every visual update.  Canonicalisation is
    deliberately narrow: only well-known scopes and event/count aliases are
    collapsed; otherwise the normalized label remains the identity.
    """
    label = str(slot.get("label", "") or "").strip().lower()
    citation = str(slot.get("citation", "") or "").strip().lower()
    scope = str(slot.get("scope", "") or "").strip().lower()
    claim_id = re.sub(
        r"[^a-z0-9:_\-]+",
        "_",
        str(slot.get("claim_id") or slot.get("id") or "").strip().lower(),
    ).strip("_")
    generic_claim_ids = {
        "claim", "evidence", "fact", "answer", "answer_fact", "selected_answer",
        "action_count", "total_count", "state_count", "scene_count", "generic_count",
        "count", "comparison", "dynamic_relation", "qualifier",
    }
    is_generic_claim_id = claim_id in generic_claim_ids or bool(
        re.fullmatch(r"(?:action|total|state|scene|generic)_count", claim_id)
    )
    semantic_parts = [
        claim_id,
        scope,
        label,
        str(slot.get("target", "") or "").strip().lower(),
        str(slot.get("relation", "") or "").strip().lower(),
        str(slot.get("direction", "") or "").strip().lower(),
    ]
    if claim_id and not is_generic_claim_id:
        target = re.sub(r"[^a-z0-9:_\-]+", "_", str(slot.get("target") or "").strip().lower()).strip("_")
        return f"claim_id:{claim_id[:160]}" + (f":target:{target[:120]}" if target else "")
    if claim_id and is_generic_claim_id:
        semantic_identity = re.sub(
            r"[^a-z0-9:_\-]+",
            "_",
            ":".join(part for part in semantic_parts if part),
        ).strip("_:")
        if semantic_identity:
            return f"claim_id:{semantic_identity[:240]}"
    label_text = re.sub(r"[^a-z0-9]+", " ", label).strip()
    compact = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    if scope in {"action_count", "total_count", "state_count", "scene_count", "generic_count"}:
        semantic_identity = re.sub(
            r"[^a-z0-9:_\-]+",
            "_",
            ":".join(
                part
                for part in (
                    scope,
                    label,
                    str(slot.get("target", "") or "").strip().lower(),
                    str(slot.get("relation", "") or "").strip().lower(),
                )
                if part
            ),
        ).strip("_:")
        return f"scope:{semantic_identity[:240] or scope}"
    event_match = re.search(r"\bevent\s*[_\- ]?\s*(\d+)\b", label_text)
    if event_match:
        return f"event:{event_match.group(1)}"
    if "social" in label_text and any(token in label_text for token in ("activity", "interaction")):
        return "social_activity"
    if "individual" in label_text and "activity" in label_text:
        return "individual_activity"
    if compact in {"answer", "answer_fact", "selected_answer", "fact", "claim", "evidence"}:
        value = re.sub(r"[^a-z0-9]+", "_", str(slot.get("value", "") or "").lower()).strip("_")
        if value:
            return f"claim:{value[:120]}"
    return compact or re.sub(r"[^a-z0-9]+", "_", citation).strip("_") or fallback


def _merge_evidence_slots(
    existing_slots: List[Dict[str, Any]],
    new_slots: List[Dict[str, Any]],
    query: str = "",
    *,
    source: str = "planner",
) -> List[Dict[str, Any]]:
    def _slot_key(slot: Mapping[str, Any], fallback: str) -> str:
        return _canonical_evidence_slot_key(slot, fallback)

    score = {"missing": 0, "approximate": 1, "verified": 2}
    merged: Dict[str, Dict[str, str]] = {}
    ordered_keys: List[str] = []

    def _support_values(slot: Mapping[str, Any], primary_key: str, history_key: str) -> List[str]:
        values: List[str] = []
        raw_primary = slot.get(primary_key)
        primary_values = raw_primary if isinstance(raw_primary, (list, tuple, set)) else [raw_primary]
        raw_history = slot.get(history_key)
        history_values = raw_history if isinstance(raw_history, (list, tuple, set)) else [raw_history]
        for value in [*primary_values, *history_values]:
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)
        return values

    def _preserve_support_history(
        current: Mapping[str, Any],
        prepared: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        histories: Dict[str, List[str]] = {}
        for primary_key, history_key in (
            ("citation", "supporting_citations"),
            ("source_tool", "supporting_source_tools"),
            ("time_window", "supporting_time_windows"),
        ):
            values = _support_values(current, primary_key, history_key)
            for value in _support_values(prepared, primary_key, history_key):
                if value not in values:
                    values.append(value)
            if values:
                histories[history_key] = values
        return histories

    for idx, slot in enumerate(existing_slots):
        key = _slot_key(slot, f"existing_{idx}")
        prepared = dict(slot)
        prepared.setdefault("scope", _evidence_slot_scope(prepared))
        prepared.setdefault("superseded", False)
        prepared.setdefault("context_only", False)
        prepared.setdefault("provenance", "unknown")
        merged[key] = prepared
        ordered_keys.append(key)

    for idx, slot in enumerate(new_slots):
        key = _slot_key(slot, f"new_{idx}")
        prepared = dict(slot)
        prepared.setdefault("scope", _evidence_slot_scope(prepared))
        prepared["superseded"] = False
        prepared["context_only"] = False
        prepared["provenance"] = str(source or "planner").strip() or "planner"
        current = merged.get(key)
        if current is None:
            merged[key] = prepared
            ordered_keys.append(key)
            continue
        support_history = _preserve_support_history(current, prepared)
        for history_key, values in support_history.items():
            prepared[history_key] = values
        current_score = score.get(str(current.get("evidence_status", "missing")), 0)
        next_score = score.get(str(prepared.get("evidence_status", "missing")), 0)
        # A later observation is the authoritative update for the same
        # semantic slot when it is at least as informative.  Keeping the old
        # value merely because its citation is longer was the source of stale
        # action-count claims surviving a direct Inspect_Frame update.
        if next_score > current_score or (
            next_score == current_score
            and (
                len(str(prepared.get("citation", ""))) >= len(str(current.get("citation", "")))
                or str(prepared.get("value", "")).strip() != str(current.get("value", "")).strip()
            )
        ):
            merged[key] = prepared
        else:
            # Keep an independent citation/tool/time-window history even when
            # the newer claim is not selected as the active value.  Replacing
            # a claim must not erase evidence that a later verifier gate needs
            # to establish redundancy.
            for history_key, values in support_history.items():
                current[history_key] = values

    merged_values = [merged[key] for key in ordered_keys if key in merged]
    query_lower = str(query or "").strip().lower()
    comparison_query = any(
        marker in query_lower
        for marker in (
            "different from", "different than", "difference between", "compared to",
            "compared with", "other people", "other person", "what part is different",
        )
    )
    if comparison_query:
        # Video_Clip_QA is useful for discovery but can return an approximate
        # holistic statement about several body parts.  Once a later
        # Inspect_Frame/packed-frame observation gives a verified comparison,
        # retain that old statement in the raw trajectory but mark it as
        # superseded for the active safety ledger.  Otherwise one low-quality
        # "everything is different" sentence can block a correct answer even
        # after direct visual comparison has ruled the alternatives out.
        direct_comparison_slots = [
            slot
            for slot in merged_values
            if _slot_is_active(slot)
            and _slot_status(slot) == "verified"
            and any(
                marker in " ".join(
                    str(slot.get(key, "") or "").lower()
                    for key in ("label", "value", "citation")
                )
                for marker in ("frame", "pack_", "inspect_frame", "visual evidence")
            )
            and bool(
                _COMPARISON_MARKERS.intersection(
                    set(
                        re.findall(
                            r"[a-z0-9]+",
                            " ".join(
                                str(slot.get(key, "") or "").lower()
                                for key in ("label", "value", "citation")
                            ),
                        )
                    )
                )
            )
        ]
        if direct_comparison_slots:
            target_label = str(direct_comparison_slots[-1].get("label") or "direct_visual_comparison")
            for slot in merged_values:
                if not _slot_is_active(slot) or _slot_status(slot) != "approximate":
                    continue
                source_text = " ".join(
                    str(slot.get(key, "") or "").lower()
                    for key in ("label", "value", "citation")
                )
                if "video_clip_qa" not in source_text and not bool(
                    slot.get("logical conflict", slot.get("logical_conflict", False))
                ):
                    continue
                slot["superseded"] = True
                slot["superseded_by"] = target_label
                slot["context_only"] = True
                slot["logical_conflict"] = False
    action_query = _query_asks_action_count(query)
    if action_query:
        verified_action_slots = [
            slot for slot in merged_values
            if _slot_is_active(slot)
            and _slot_status(slot) == "verified"
            and str(slot.get("scope") or _evidence_slot_scope(slot)) == "action_count"
        ]
        if verified_action_slots:
            # Prefer the newest/most specific action claim as the target slot.
            # This is a relevance update, not an answer decision: the raw
            # trajectory still preserves every prior LLM claim.
            target = verified_action_slots[-1]
            target_label = str(target.get("label") or "action_count")
            for slot in merged_values:
                if slot is target or not _slot_is_active(slot):
                    continue
                scope = str(slot.get("scope") or _evidence_slot_scope(slot))
                status = _slot_status(slot)
                if scope in {"total_count", "generic_count"}:
                    # A total/pre-existing or otherwise unscoped count is
                    # useful context for the action claim, but it is not a
                    # competing answer slot.  The old rule only demoted
                    # ``generic_count`` when it was missing/approximate.
                    # That left a verified generic slot such as "several
                    # sticks are present" active after Video_Clip_QA had
                    # supplied a direct action_count, so the deterministic
                    # verifier kept rejecting a correct Finish forever.
                    # Once a verified action slot exists, all non-action
                    # count scopes are explicitly state/context evidence.
                    slot["context_only"] = True
                    slot["superseded_by"] = target_label
                    slot["logical_conflict"] = False
                elif scope == "action_count" and status in {"missing", "approximate"}:
                    slot["superseded"] = True
                    slot["superseded_by"] = target_label

    return merged_values


class RecursiveEngine:
    """Recursive engine that orchestrates the entire agent system.
    
    This is the core component that implements recursive expansion:
    - Manages the call stack and context
    - Spawns sub-agents for high-level abstractions
    - Coordinates execution between parent and child agents
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        subagent_llm: Optional[BaseLLM] = None,
        context_manager: Optional[ContextManager] = None,
        raw_trace_jsonl_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize recursive engine.
        
        Args:
            llm: Language model instance (created if not provided)
            context_manager: Context manager (created if not provided)
            raw_trace_jsonl_path: Optional path to a JSONL file for persisting
                raw LLM request/response traces (used for offline analysis).
            **kwargs: Additional keyword arguments (stored but not used directly).
        """
        if kwargs:
            logger.info("RecursiveEngine received extra kwargs (stored): %s", list(kwargs.keys()))
        # Planner默认使用独立的思考模型
        planner_llm = llm or get_llm_provider(
            model_name=settings.planner_model_path,
            model_role="planner",
        )
        self.planner_llm = cache_existing_transformers_provider(
            build_role_specific_provider_view(planner_llm, "planner")
        )
        self.role_agents = FourRoleAgentSuite(self.planner_llm)
        raw_disabled_roles = kwargs.get("disabled_agent_roles")
        if raw_disabled_roles is None:
            raw_disabled_roles = os.environ.get("TOOLSTAR_DISABLED_AGENT_ROLES", "")
        if isinstance(raw_disabled_roles, str):
            disabled_roles = {
                item.strip().lower()
                for item in raw_disabled_roles.replace(";", ",").split(",")
                if item.strip()
            }
        elif isinstance(raw_disabled_roles, (list, tuple, set, frozenset)):
            disabled_roles = {
                str(item).strip().lower()
                for item in raw_disabled_roles
                if str(item).strip()
            }
        else:
            disabled_roles = set()
        unsupported_disabled_roles = disabled_roles.difference({"observer", "reflector"})
        if unsupported_disabled_roles:
            raise ValueError(
                "Only observer and reflector support online role ablation; "
                f"unsupported disabled_agent_roles={sorted(unsupported_disabled_roles)}"
            )
        self.disabled_agent_roles = frozenset(disabled_roles)
        self._subagent_llm = (
            cache_existing_transformers_provider(
                build_role_specific_provider_view(subagent_llm, "subagent")
            )
            if subagent_llm is not None
            else None
        )
        self._last_completed_trajectory: Optional[Trajectory] = None
        self._last_completed_trajectory_depth: Optional[int] = None
        # Preserve the legacy attribute for planner-root call sites.
        self.llm = self.planner_llm
        self.context = context_manager or ContextManager()
        self.root_trajectory: Optional[Trajectory] = None
        self._active_executed_tool_signatures: Dict[str, str] = {}
        self.raw_trace_jsonl_path = raw_trace_jsonl_path
        self.extra_params = kwargs
        wall_clock_deadline = kwargs.get("wall_clock_deadline_monotonic", kwargs.get("wall_clock_deadline"))
        try:
            self.wall_clock_deadline_monotonic = float(wall_clock_deadline) if wall_clock_deadline is not None else None
        except Exception:
            self.wall_clock_deadline_monotonic = None
        try:
            self.wall_clock_budget_seconds = float(kwargs.get("wall_clock_budget_seconds", 0.0) or 0.0)
        except Exception:
            self.wall_clock_budget_seconds = 0.0
        self.progress_callback = kwargs.get("progress_callback") if callable(kwargs.get("progress_callback")) else None
        try:
            self.planner_generation_timeout_seconds = float(
                kwargs.get("planner_generation_timeout_seconds", 0.0) or 0.0
            )
        except Exception:
            self.planner_generation_timeout_seconds = 0.0
        try:
            self.subagent_generation_timeout_seconds = float(
                kwargs.get("subagent_generation_timeout_seconds", 0.0) or 0.0
            )
        except Exception:
            self.subagent_generation_timeout_seconds = 0.0
        try:
            self.generation_heartbeat_interval_seconds = max(
                5.0,
                float(kwargs.get("generation_heartbeat_interval_seconds", 30.0) or 30.0),
            )
        except Exception:
            self.generation_heartbeat_interval_seconds = 30.0
        self.turn_yield_callback = kwargs.get("turn_yield_callback") if callable(kwargs.get("turn_yield_callback")) else None

        logger.info(
            "RecursiveEngine initialized with planner_llm role=%s thinking=%s shared_source_role=%s subagent_llm_role=%s subagent_thinking=%s",
            getattr(self.planner_llm, "model_role", None),
            getattr(self.planner_llm, "enable_thinking", None),
            getattr(self.planner_llm, "_shared_provider_source_role", None),
            getattr(self._subagent_llm, "model_role", None) if self._subagent_llm is not None else None,
            getattr(self._subagent_llm, "enable_thinking", None) if self._subagent_llm is not None else None,
        )
        logger.info("Four-agent LLM/VLM kernel contract: %s", self.role_agents.manifest())
        if self.disabled_agent_roles:
            logger.info(
                "Four-agent online role ablation enabled: disabled_roles=%s",
                sorted(self.disabled_agent_roles),
            )

    def _emit_progress(self, **payload: Any) -> None:
        callback = self.progress_callback
        if not callable(callback):
            return
        event = {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }
        if not event:
            return
        try:
            callback(event)
        except Exception as exc:
            logger.debug("Progress callback failed: %s", exc)

    def _yield_turn(self, **payload: Any) -> float:
        callback = self.turn_yield_callback
        if not callable(callback):
            return 0.0
        event = {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }
        paused_seconds = 0.0
        self._emit_progress(**dict(event, phase="turn_yield"))
        try:
            result = callback(event)
        except Exception as exc:
            logger.warning("Turn-yield callback failed; continuing without pause: %s", exc)
            self._emit_progress(
                **dict(
                    event,
                    phase="turn_resume",
                    status="callback_error",
                    error=str(exc),
                )
            )
            return 0.0
        if isinstance(result, dict):
            try:
                paused_seconds = float(result.get("paused_seconds", 0.0) or 0.0)
            except Exception:
                paused_seconds = 0.0
        elif result is not None:
            try:
                paused_seconds = float(result)
            except Exception:
                paused_seconds = 0.0
        if paused_seconds > 0:
            self._emit_progress(
                **dict(
                    event,
                    phase="turn_resume",
                    status="running",
                    elapsed_seconds=round(paused_seconds, 6),
                )
            )
        return max(0.0, paused_seconds)

    def _get_llm_for_depth(self, depth: int) -> BaseLLM:
        if depth <= 0:
            return self.planner_llm

        if self._subagent_llm is not None:
            return self._subagent_llm

        subagent_model_name = _normalize_model_identifier(settings.subagent_model_path)
        planner_model_name = _normalize_model_identifier(getattr(self.planner_llm, "model_name", ""))
        if planner_model_name and planner_model_name == subagent_model_name:
            self._subagent_llm = build_role_specific_provider_view(
                self.planner_llm,
                "subagent",
            )
            logger.info(
                "Subagent reusing planner LLM weights via a role-specific shared provider view: %s",
                subagent_model_name,
            )
            return self._subagent_llm

        self._subagent_llm = get_llm_provider(
            model_name=settings.subagent_model_path,
            model_role="subagent",
        )
        logger.info("Subagent initialized independent LLM provider: %s", subagent_model_name)
        return self._subagent_llm

    @staticmethod
    def _infer_provider_name(llm: BaseLLM) -> Optional[str]:
        if isinstance(llm, TransformersProvider):
            return "transformers"
        if isinstance(llm, OpenAIProvider):
            return "openai"
        if isinstance(llm, AnthropicProvider):
            return "anthropic"
        return None

    def _get_generation_kwargs_for_depth(self, depth: int, llm: BaseLLM) -> Dict[str, Any]:
        role = "planner" if depth == 0 else "subagent"
        config = dict(
            settings.get_generation_config(
                role,
                provider=self._infer_provider_name(llm),
            )
            or {}
        )
        max_new_tokens = config.pop("max_new_tokens", None)
        if max_new_tokens is not None:
            config["max_tokens"] = max_new_tokens
        override_key = "planner_generation_overrides" if depth == 0 else "subagent_generation_overrides"
        overrides = self.extra_params.get(override_key)
        if isinstance(overrides, dict) and overrides:
            logger.warning(
                "[GenerationConfig][DIAG] role=%s ignored %s because planner/subagent main generation parameters must come from config/settings.",
                role,
                override_key,
            )
        # Role-specific defaults come from settings.get_generation_config(...).
        # Keep only runtime-execution flags here.
        config["enable_prefix_caching"] = False
        config["enable_rollout_prefix_cache"] = False
        if "enable_thinking" not in config:
            logger.warning(
                "[GenerationConfig][DIAG] role=%s missing enable_thinking in config/settings; defaulting to False.",
                role,
            )
        effective_enable_thinking = bool(config.get("enable_thinking"))
        config["enable_thinking"] = effective_enable_thinking
        if role == "planner" and effective_enable_thinking:
            try:
                min_thinking_tokens = int(getattr(settings, "planner_min_thinking_generation_tokens", 8192) or 8192)
            except Exception:
                min_thinking_tokens = 8192
            if min_thinking_tokens > 0:
                try:
                    current_max_tokens = int(config.get("max_tokens") or 0)
                except Exception:
                    current_max_tokens = 0
                if current_max_tokens < min_thinking_tokens:
                    logger.warning(
                        "[GenerationConfig][DIAG] role=%s max_tokens=%s raised to planner_min_thinking_generation_tokens=%d because planner thinking is enabled.",
                        role,
                        config.get("max_tokens"),
                        min_thinking_tokens,
                    )
                    config["max_tokens"] = min_thinking_tokens
        provider_name = self._infer_provider_name(llm)
        if provider_name in {"transformers", "local"}:
            config.pop("presence_penalty", None)
        logger.info(
            "[GenerationConfig][DIAG] role=%s max_tokens=%s enable_thinking=%s temperature=%s top_p=%s top_k=%s",
            role,
            config.get("max_tokens"),
            bool(config.get("enable_thinking")),
            config.get("temperature"),
            config.get("top_p"),
            config.get("top_k"),
        )
        return config

    # Track 1: persist raw LLM responses to JSONL for SFT/RL

    def _write_raw_trace(
        self, *, depth: int, iteration: int, goal: str, raw_response: str,
    ) -> None:
        """Append one JSONL line with the complete raw LLM response.

        This preserves <think>...</think> blocks for offline SFT/RL data
        collection.  No-op when *raw_trace_jsonl_path* is not configured.
        """
        if not self.raw_trace_jsonl_path:
            return
        from datetime import datetime, timezone
        from pathlib import Path
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": self.extra_params.get("session_query") or goal,
            "depth": depth,
            "agent_role": "planner" if depth == 0 else "subagent",
            "iteration": iteration,
            "raw_response": raw_response,
        }
        try:
            path = Path(self.raw_trace_jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write raw trace JSONL: %s", exc)

    @staticmethod
    def _extract_context_section(context: str, marker: str) -> str:
        text = str(context or "")
        if not text or marker not in text:
            return ""
        start = text.index(marker) + len(marker)
        next_section = text.find("\n[", start)
        if next_section > start:
            return text[start:next_section].strip()
        return text[start:].strip()

    @classmethod
    def _context_has_mode(cls, context: str, mode_name: str) -> bool:
        current = cls._extract_context_section(context, "[Sub-Agent Mode]\n")
        return current == str(mode_name or "").strip()

    @staticmethod
    def _build_mediation_return(
        final_answer: str,
        success: bool,
        execution_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta = dict(execution_meta or {})
        payload = {
            "mode": "planner_action_mediation",
            "status": str(meta.get("status") or ("success" if success else "error")),
            "executed_tool": str(meta.get("tool", "")).strip(),
            "executed_params": meta.get("args", {}) if isinstance(meta.get("args"), dict) else {},
            "tool_status": str(meta.get("status") or ("success" if success else "error")),
            "observation_text": str(meta.get("observation", "")).strip(),
            "observation_payload": meta.get("payload"),
            "source": str(meta.get("source", "subagent")).strip() or "subagent",
            "abstract_tool": str(meta.get("abstract_tool", "")).strip(),
            "final_result": final_answer,
            "success": bool(success),
            "is_recursive": bool(meta.get("is_recursive", False)),
            "executed": bool(meta.get("executed", False)),
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _parse_mediation_return(raw_result: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw_result, dict):
            payload = raw_result
        elif isinstance(raw_result, str):
            text = raw_result.strip()
            if not text:
                return None
            try:
                payload = json.loads(text)
            except Exception:
                return None
        else:
            return None
        if not isinstance(payload, dict):
            return None
        if str(payload.get("mode", "")).strip() != "planner_action_mediation":
            return None
        return payload

    def _remember_completed_trajectory(self, trajectory: Trajectory) -> None:
        self._last_completed_trajectory = trajectory
        self._last_completed_trajectory_depth = trajectory.depth

    def _get_last_completed_trajectory(self, expected_depth: Optional[int] = None) -> Optional[Trajectory]:
        trajectory = self._last_completed_trajectory
        if trajectory is None:
            return None
        if expected_depth is not None and self._last_completed_trajectory_depth != expected_depth:
            return None
        return trajectory
    
    def solve(self, goal: str, depth: int = 0, extra_context: str = "") -> str:
        """Solve a goal recursively.
        
        This is the main entry point that implements the recursive logic:
        1. Check recursion depth
        2. Push new scope frame
        3. Create agent and trajectory
        4. Run ReAct loop
        5. Handle actions (primitive vs recursive)
        6. Pop scope frame
        
        Args:
            goal: Goal to achieve
            depth: Current recursion depth (internal use)
            extra_context: Additional context to prepend (e.g. discovered tools from parent)
            
        Returns:
            Final answer
            
        Raises:
            MaxRecursionError: If max depth exceeded
        """
        role_label = "Planner" if depth == 0 else "Subagent"
        logger.info("%s[%s][Depth %d] Solving goal: %s", '  ' * depth, role_label, depth, goal)
        
        # Step 1: Check depth
        try:
            self.context.push_frame(goal)
        except MaxRecursionError as e:
            logger.error(f"Max recursion depth exceeded: {e}")
            return f"Error: {str(e)}"
        
        # Step 2: Create trajectory for this level
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        trajectory = Trajectory(
            task_id=task_id,
            goal=goal,
            depth=depth
        )

        if depth == 0:
            trajectory.add_debug_event({
                "type": "agent_kernel_contract",
                "event_type": "AgentKernelContractEvent",
                "agent_role": "workflow",
                "roles": self.role_agents.manifest(),
                "contract": "four_independent_llm_vlm_roles",
                "disabled_online_roles": sorted(self.disabled_agent_roles),
                "active_online_roles": [
                    role for role in ("planner", "observer", "reflector")
                    if role not in self.disabled_agent_roles
                ],
            })
            logger.info(
                "[Planner] Four-agent workflow started roles=%s kernel_id=%s",
                list(self.role_agents.manifest().get("roles", {}).keys()),
                self.role_agents.manifest().get("kernel_id"),
            )
        
        if depth == 0:
            self.root_trajectory = trajectory
        
        # Step 3: Get context information
        # Context isolation rule:
        # - Root depth: first-round grounding evidence may be multimodal (text + images),
        #   injected once and then kept in history.
        # - Sub-agent depth: do NOT inherit raw grounding evidence.
        context_info = self._build_context_info(depth)
        first_round_context: Any = ""
        if depth == 0 and extra_context:
            first_round_context = extra_context
        elif extra_context:
            context_info = f"{context_info}\n{extra_context}".strip()
        harness_context = str(self.extra_params.get("evolution_harness_context") or "").strip()
        if depth == 0 and harness_context:
            first_round_context = (
                f"{first_round_context}\n\n{harness_context}".strip()
                if first_round_context
                else harness_context
            )
        if depth > 0 and not self._context_has_mode(context_info, "planner_action_mediation"):
            context_info = (
                "[Sub-Agent Mode]\nplanner_action_mediation\n\n"
                "[Sub-Agent Return Mode]\nresult_only\n\n"
                f"{context_info}"
            ).strip()
        
        # Step 4: Create and run agent
        agent_max_iterations = settings.max_iterations if depth == 0 else settings.subagent_max_iterations
        current_llm = self._get_llm_for_depth(depth)
        agent = ReactAgent(
            llm=current_llm,
            trajectory=trajectory,
            max_iterations=agent_max_iterations,
        )
        
        success = False
        try:
            # Run the agent with custom action handling
            mediation_mode = depth > 0 and self._context_has_mode(context_info, "planner_action_mediation")
            final_answer, success, execution_meta = self._run_agent_with_recursion(
                agent=agent,
                goal=goal,
                context_info=context_info,
                depth=depth,
                first_round_context=first_round_context,
            )

            if mediation_mode:
                final_answer = self._build_mediation_return(
                    final_answer=final_answer,
                    success=success,
                    execution_meta=execution_meta,
                )
            
            trajectory.finish(final_answer, success=success)
            self._remember_completed_trajectory(trajectory)
            status_text = "Completed" if success else "Failed"
            logger.info("%s[%s][Depth %d] %s: %s", '  ' * depth, role_label, depth, status_text, final_answer)
            
        except Exception as e:
            error_msg = f"Error at depth {depth}: {str(e)}"
            logger.error(error_msg)
            trajectory.finish("", success=False, error_message=error_msg)
            self._remember_completed_trajectory(trajectory)
            final_answer = error_msg
        
        finally:
            # Step 5: Pop scope frame
            self.context.pop_frame()
        
        return final_answer
    
    def _run_agent_with_recursion(
        self,
        agent: ReactAgent,
        goal: str,
        context_info: str,
        depth: int,
        first_round_context: Any = "",
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """Run agent with custom handling for recursive actions.
        
        This method intercepts the agent's execution to handle recursive calls.
        
        Args:
            agent: ReactAgent instance
            goal: Current goal
            context_info: Context information
            depth: Current depth
            
        Returns:
            Final answer, success flag, and the last concrete execution metadata.
        """
        from .prompts import get_system_prompt

        current_llm = self._get_llm_for_depth(depth)
        generation_kwargs = self._get_generation_kwargs_for_depth(depth, current_llm)
        
        # Supply the live executable tool catalog to the root planner.  This
        # keeps tool selection schema-first and avoids asking the model to
        # invent abstract names that later require fuzzy recovery.
        try:
            # Root prompt keeps grounding out of system prompt to avoid repeated injection
            # across every round. Grounding is injected once as first-round user context.
            if depth > 0:
                is_mediation_subagent = self._context_has_mode(context_info, "planner_action_mediation")
                matched = re.match(r"^\s*Implement function\s+([^\(\s]+)\((.*)\)\s*$", goal)
                function_name = matched.group(1) if matched else "Sub_Task"
                function_args = matched.group(2).strip() if matched else ""
                abstract_tools = get_runtime_control_tools_description(
                    mediation_mode=is_mediation_subagent
                )
                # Extract abstract tool description from extra_context marker
                abstract_tool_desc = self._extract_context_section(
                    context_info,
                    "[Planner Action Description]\n",
                ) or self._extract_context_section(
                    context_info,
                    "[Abstract Tool Description]\n",
                )
                system_prompt = get_recursive_prompt(
                    function_name=function_name,
                    function_args=function_args,
                    goal=goal,
                    parent_context=context_info or "No parent context provided.",
                    tools_description=abstract_tools,
                    abstract_tool_description=abstract_tool_desc,
                )
            else:
                system_prompt = get_system_prompt(
                    tools_description=registry.get_tools_description(),
                    query=goal,
                    grounding_context="",
                    routing_context=first_round_context or context_info,
                )
        except Exception as e:
            logger.error(f"Failed to build system prompt: {e}")
            system_prompt = "You are an assistant. Respond in Thought/Action format."

        # Keep the static role contract immutable and append only a bounded,
        # runtime-generated recovery directive when the evidence boundary
        # repeatedly rejects Finish. Planner and Reflector receive the same
        # escalation state without rewriting the base prompt.
        base_system_prompt = system_prompt
        active_recovery_system_directive = ""

        def _effective_system_prompt() -> str:
            if not active_recovery_system_directive:
                return base_system_prompt
            return f"{base_system_prompt}\n\n{active_recovery_system_directive}"
        
        # Initialize conversation with always-on Fold_Memory compression
        history: List[Dict[str, str]] = []
        memory_state: Dict[str, Any] = {}
        tool_error_streaks: Dict[str, int] = {}
        consecutive_finish_without_answer = 0  # Track Finish() calls without answer
        # Discovered tools: populated when Tool_Search returns results.
        # Injected into context so the planner can call them in subsequent steps.
        discovered_tools_desc: str = ""
        root_effective_tool_executed = False
        last_observation_signature = ""
        consecutive_low_progress = 0
        low_progress_hint_sent = False
        blocked_action_signatures: set[str] = set()
        if depth == 0 or not isinstance(getattr(self, "_active_executed_tool_signatures", None), dict):
            self._active_executed_tool_signatures = {}
        executed_tool_signatures = self._active_executed_tool_signatures
        repeated_loop_violations = 0
        subagent_toolsearch_failures = 0
        MAX_SUBAGENT_TOOLSEARCH_FAILURES = 3
        abstract_tool_cache: Dict[str, Dict[str, Any]] = {}
        evidence_first_mode = depth == 0 and query_requires_evidence_first(goal)
        mediation_mode = depth > 0 and self._context_has_mode(context_info, "planner_action_mediation")
        role_label = "Planner" if depth == 0 else "Subagent"
        ablation_direct_toolsearch_mode = (
            depth == 0 and bool(self.extra_params.get("planner_toolsearch_direct_execution", False))
        )
        ablation_single_tool_only = (
            ablation_direct_toolsearch_mode
            and bool(self.extra_params.get("planner_toolsearch_direct_single_tool_only", False))
        )
        single_action_per_step_mode = bool(self.extra_params.get("planner_single_action_per_step", False))
        mediation_execution_hint = (
            self._extract_context_section(context_info, "[Schema-Adapted Execution Hint]\n")
            if mediation_mode
            else ""
        )
        evidence_slots: List[Dict[str, Any]] = []
        # Evidence updates are transactional.  Planner-declared slots are
        # pending claims until a concrete successful observation (or an
        # explicit evidence-only checkpoint) commits them.  Finish's self
        # report never crosses this boundary.
        evidence_revision = 0
        pending_planner_evidence_slots: List[Dict[str, Any]] = []
        verifier_config = self.extra_params.get("evolution_harness_verifier_config")
        raw_evolution_policy = self.extra_params.get("evolution_harness_policy")
        evolution_policy = raw_evolution_policy if isinstance(raw_evolution_policy, dict) else {}
        low_signal_recovery_policy = evolution_policy.get("low_signal_temporal_grounding")
        if not isinstance(low_signal_recovery_policy, dict):
            low_signal_recovery_policy = {}
        evidence_verifier = (
            EvidenceVerifier(config=verifier_config)
            if evidence_first_mode and isinstance(verifier_config, dict)
            else EvidenceVerifier()
            if evidence_first_mode
            else None
        )
        visual_probe_count = 0
        retrieval_probe_count = 0
        retrieval_failure_count = 0
        low_signal_recovery_call_count = 0
        try:
            max_low_signal_visual_probes = max(
                2,
                int(self.extra_params.get("max_low_signal_visual_probes", 4) or 4),
            )
        except (TypeError, ValueError):
            max_low_signal_visual_probes = 4
        low_signal_visual_probe_streak = 0
        evolution_policy_events: List[Dict[str, Any]] = []
        direct_visual_probe_count = 0
        cross_modal_verify_count = 0
        modality_attempts: Dict[str, int] = {"text": 0, "visual": 0, "cross_modal": 0}
        modality_successes: Dict[str, int] = {"text": 0, "visual": 0, "cross_modal": 0}
        needle_candidate_windows: List[Dict[str, Any]] = []
        needle_candidate_completed: set[str] = set()
        needle_candidate_reserved: set[str] = set()
        identical_action_failures: Dict[str, int] = {}
        last_execution_meta: Dict[str, Any] = {}
        no_progress_guard_signature = ""
        no_progress_guard_family = ""
        no_progress_abort_reason = ""
        verifier_issue_counts: Dict[str, int] = {}
        verifier_issue_revisions: Dict[str, set[int]] = {}
        # Count Finish candidates rejected by either the deterministic verifier
        # or Reflector.  A candidate can re-enter only after the evidence
        # revision advances; failed/policy actions do not advance it.
        finish_rejection_counts: Dict[str, int] = {}
        finish_rejection_revisions: Dict[str, int] = {}
        finish_rejection_reasons: Dict[str, str] = {}
        finish_recovery_violations = 0
        finish_recovery_mode = False
        finish_recovery_turns_used = 0
        finish_recovery_finish_pending = False
        finish_rejection_streak = 0
        finish_recovery_last_reason = ""
        finish_recovery_last_targets: List[str] = []
        forced_finish_recovery_probe_count = 0
        try:
            max_finish_recovery_rounds = max(
                2,
                min(
                    6,
                    int(self.extra_params.get("max_finish_recovery_rounds", 3) or 3),
                ),
            )
        except (TypeError, ValueError):
            max_finish_recovery_rounds = 3
        try:
            max_finish_recovery_retries = max(
                2,
                min(
                    4,
                    int(self.extra_params.get("max_finish_recovery_retries", 3) or 3),
                ),
            )
        except (TypeError, ValueError):
            max_finish_recovery_retries = 3
        low_signal_guard_recovery_count = 0
        low_signal_recovery_triggered_this_observation = False
        try:
            max_low_signal_guard_recoveries = max(
                1,
                min(
                    3,
                    int(self.extra_params.get("max_low_signal_guard_recoveries", 2) or 2),
                ),
            )
        except (TypeError, ValueError):
            max_low_signal_guard_recoveries = 2

        def _finish_rejected_without_new_evidence(signature: str) -> bool:
            rejected_revision = finish_rejection_revisions.get(str(signature or ""))
            return (
                depth == 0
                and evidence_first_mode
                and rejected_revision is not None
                and evidence_revision <= int(rejected_revision)
            )

        final_turn_warning_sent = False
        forced_final_turn_retry_used = False
        route_refresh_signatures: set[str] = set()
        route_refresh_count = 0
        adjacent_probe_repeat_count = 0
        last_probe_window_state: Dict[str, Any] = {
            "family": "",
            "window": None,
            "signature": "",
        }
        if ablation_direct_toolsearch_mode:
            execution_mode_note = (
                "[Execution Mode]\n"
                "Root planner actions run in ablation direct Tool_Search execution mode. "
                "For each planner action, the runtime will retrieve the top-1 most relevant executable tool "
                "with Tool_Search and execute it directly without spawning sub-agents."
            )
            if ablation_single_tool_only or single_action_per_step_mode:
                execution_mode_note += (
                    " If you emit multiple actions in one turn, only the first action will be executed and "
                    "the rest will be skipped."
                )
            history.append({"role": "user", "content": execution_mode_note})
            history.append({
                "role": "assistant",
                "content": "Understood. I will plan with direct Tool_Search execution in mind.",
            })
        elif single_action_per_step_mode:
            history.append({
                "role": "user",
                "content": (
                    "[Execution Mode]\n"
                    "Single-action-per-step ablation mode is enabled. "
                    "If you emit multiple actions or tools in one step, only the first one will be executed "
                    "and the rest will be skipped."
                ),
            })
            history.append({
                "role": "assistant",
                "content": "Understood. I will keep each step to one action or tool.",
            })

        max_ctx = current_llm.get_max_context_length()
        # Compute threshold for DYNAMIC content only (history + folded memory state).
        # The static system_prompt is always present and cannot be compressed, so
        # including it in the estimate was causing threshold to be exceeded on every
        # iteration even with an empty history, leading to compression stagnation.
        _sp_tokens = current_llm.count_message_tokens(
            [{"role": "system", "content": _effective_system_prompt()}],
            enable_thinking=generation_kwargs.get("enable_thinking"),
        )
        _enforce_cap = bool(getattr(settings, "enforce_max_output_tokens", False))
        _configured_cap = int(getattr(settings, "max_output_tokens", 0) or 0)
        _gen_headroom = _configured_cap if (_enforce_cap and _configured_cap > 0) else 0
        ctx_threshold = max(512, max_ctx - _sp_tokens - _gen_headroom)
        logger.info(
            "[%s][Depth %d] Context window: max=%d, sp_tokens=%d, output_headroom=%d, dynamic_budget=%d tokens",
            role_label,
            depth,
            max_ctx,
            _sp_tokens,
            _gen_headroom,
            ctx_threshold,
        )

        def _memory_state_payload() -> Dict[str, Any]:
            tracked_keys = (
                "found_facts",
                "discovered_tools",
                "failed_actions",
                "do_not_repeat",
                "missing_evidence",
                "next_allowed_tools",
                "recent_relevant_observations",
                "evidence_slots",
                "modality_progress",
            )
            payload: Dict[str, Any] = {}
            for key in tracked_keys:
                value = memory_state.get(key)
                if value:
                    payload[key] = value
            return payload

        def _memory_state_text() -> str:
            payload = _memory_state_payload()
            if not payload:
                return ""
            return json.dumps({"MemoryState": payload}, ensure_ascii=False)

        def _current_discovered_tool_lines() -> List[str]:
            if not discovered_tools_desc:
                return []
            return [line.strip() for line in discovered_tools_desc.splitlines() if line.strip()]

        def _compact_discovered_tools_for_subagent(max_lines: int = 12) -> str:
            lines = _current_discovered_tool_lines()
            if not lines:
                return ""
            return "\n".join(lines[-max(1, int(max_lines)):])

        def _format_candidate_tools_for_subagent(candidate_tools: Any) -> str:
            if isinstance(candidate_tools, list):
                candidate_tools = [
                    item for item in candidate_tools
                    if isinstance(item, dict) and str(item.get("name", "")).strip() != "Tool_Search"
                ]
            if not isinstance(candidate_tools, list) or not candidate_tools:
                return ""
            compact_candidate_tools = compact_tool_schemas_for_prompt(candidate_tools)
            if not compact_candidate_tools:
                return ""
            sections = [
                json.dumps(compact_candidate_tools, ensure_ascii=False, indent=2),
            ]
            schema_lines = schema_lines_from_tools(compact_candidate_tools)
            if schema_lines:
                sections.append("[Known Tool Parameter Requirements]\n" + schema_lines)
            return "\n\n".join(section for section in sections if section)

        def _refresh_modality_progress() -> None:
            if not evidence_first_mode:
                return
            memory_state["modality_progress"] = {
                "text_attempts": modality_attempts["text"],
                "visual_attempts": modality_attempts["visual"],
                "cross_modal_checks": modality_attempts["cross_modal"],
                "text_successes": modality_successes["text"],
                "visual_successes": modality_successes["visual"],
                "direct_visual_probes": direct_visual_probe_count,
                "cross_modal_successes": modality_successes["cross_modal"],
            }

        def _dynamic_token_estimate() -> int:
            dyn_messages: List[Dict[str, Any]] = []
            memory_text = _memory_state_text()
            if memory_text:
                dyn_messages.append({"role": "user", "content": memory_text})
            dyn_messages.extend(history)
            return current_llm.count_message_tokens(
                dyn_messages,
                enable_thinking=generation_kwargs.get("enable_thinking"),
            )

        def _deterministic_compact_history(keep_recent_count: int = 2) -> bool:
            """Bound history when the model/provider rejects the full prompt.

            Fold_Memory is optional and can itself fail under pressure. Keep
            the runtime-owned evidence ledger in memory_state, retain the
            latest assistant/user turns, and replace older multimodal payloads
            with short textual checkpoints so one bounded retry can proceed.
            """
            nonlocal history
            keep = max(1, int(keep_recent_count or 1))
            if len(history) <= keep:
                return False
            recent = list(history[-keep:])
            older = list(history[:-keep])
            state_payload = _memory_state_payload()
            state_text = ""
            if state_payload:
                try:
                    state_text = json.dumps(
                        state_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                except Exception:
                    state_text = str(state_payload)
                if len(state_text) > 5000:
                    state_text = state_text[:4997] + "..."
            summary_lines: List[str] = [
                "[Deterministic compacted planner history]",
                "Older turns were compacted after a context-pressure event. "
                "Use the runtime evidence ledger and the recent turns as authority.",
            ]
            if state_text:
                summary_lines.append("MemoryState=" + state_text)
            for item in older[-8:]:
                if not isinstance(item, Mapping):
                    continue
                role = str(item.get("role") or "turn").strip()
                text_value = " ".join(_history_content_to_text(item.get("content")).split())
                if text_value:
                    if len(text_value) > 420:
                        text_value = text_value[:417] + "..."
                    summary_lines.append(f"{role}: {text_value}")
            history = [{"role": "user", "content": "\n".join(summary_lines)}] + recent
            return True

        def _maybe_fold_memory() -> None:
            nonlocal history, memory_state

            if not bool(getattr(settings, "enable_auto_memory_folding", True)):
                return

            est_tokens = _dynamic_token_estimate()
            keep_recent_count = 2
            if len(history) > keep_recent_count:
                recent_start = max(0, len(history) - keep_recent_count)
                while recent_start > 0:
                    current_role = str(history[recent_start].get("role", ""))
                    prev_role = str(history[recent_start - 1].get("role", ""))
                    if current_role == "user" and prev_role == "assistant":
                        recent_start -= 1
                        continue
                    break
                keep_recent_count = len(history) - recent_start

            if est_tokens < ctx_threshold or len(history) <= keep_recent_count:
                return

            fold_tool = registry.get_tool("Fold_Memory")
            if fold_tool.is_virtual:
                logger.warning("Fold_Memory tool is unavailable; skipping automatic memory folding.")
                _deterministic_compact_history(keep_recent_count=keep_recent_count)
                return

            logger.info(
                "Tokens ~%d exceed threshold %d after history update. Auto-running Fold_Memory.",
                est_tokens,
                ctx_threshold,
            )
            try:
                fold_result = fold_tool.execute(
                    context_window=list(history),
                    keep_recent=keep_recent_count,
                    previous_state=memory_state,
                    discovered_tools=_current_discovered_tool_lines(),
                    blocked_action_signatures=sorted(blocked_action_signatures),
                    llm=current_llm,
                )
            except Exception as exc:
                logger.warning("Fold_Memory auto-execution failed: %s", exc)
                _deterministic_compact_history(keep_recent_count=keep_recent_count)
                return

            if not isinstance(fold_result, dict) or not fold_result.get("compressed"):
                _deterministic_compact_history(keep_recent_count=keep_recent_count)
                return

            next_state = fold_result.get("memory_state", {})
            next_recent = fold_result.get("recent_steps", history[-keep_recent_count:])
            if isinstance(next_state, dict):
                memory_state = next_state
            if isinstance(next_recent, list):
                filtered_recent = [
                    item for item in next_recent
                    if isinstance(item, dict) and "role" in item and "content" in item
                ]
                if filtered_recent:
                    history = filtered_recent

            logger.info(
                "Fold_Memory produced state: facts=%d tools=%d failures=%d gaps=%d recent=%d",
                len(memory_state.get("found_facts", []) or []),
                len(memory_state.get("discovered_tools", []) or []),
                len(memory_state.get("failed_actions", []) or []),
                len(memory_state.get("missing_evidence", []) or []),
                len(history),
            )
            if _dynamic_token_estimate() >= ctx_threshold:
                _deterministic_compact_history(keep_recent_count=keep_recent_count)

        def _history_content_to_text(content: Any) -> str:
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type", "")).strip().lower()
                    if item_type == "text":
                        text_value = " ".join(str(item.get("text", "")).split()).strip()
                        if text_value:
                            parts.append(text_value)
                    elif item_type == "image":
                        parts.append("[image]")
                    elif item_type == "video":
                        parts.append("[video]")
                return " ".join(part for part in parts if part).strip()
            return str(content or "").strip()

        def _append_history(role: str, content: Any) -> None:
            history.append({"role": role, "content": content})
            if role == "user":
                _maybe_fold_memory()

        def _truncate_history_text(text: Any, max_chars: int = 1200) -> str:
            normalized = " ".join(_history_content_to_text(text).split()).strip()
            if max_chars <= 0:
                return ""
            if len(normalized) <= max_chars:
                return normalized
            if max_chars <= 3:
                return normalized[:max_chars]
            return normalized[: max_chars - 3] + "..."

        def _strip_think_markup(text: Any, *, drop_block_content: bool = False) -> str:
            normalized = agent.sanitize_response(text)
            if drop_block_content:
                normalized = re.sub(r"<think>.*?</think>", "", normalized, flags=re.IGNORECASE | re.DOTALL)
            normalized = re.sub(r"</?think>", "", normalized, flags=re.IGNORECASE)
            return normalized.strip()

        def _strip_empty_think_blocks(text: Any) -> str:
            normalized = agent.sanitize_response(text)
            normalized = re.sub(r"<think>\s*</think>", "", normalized, flags=re.IGNORECASE | re.DOTALL)
            normalized = re.sub(r"\n{3,}", "\n\n", normalized)
            return normalized.strip()

        def _contains_nonempty_think_block(text: Any) -> bool:
            normalized = agent.sanitize_response(text)
            for match in re.finditer(r"<think>(.*?)</think>", normalized, flags=re.IGNORECASE | re.DOTALL):
                if str(match.group(1) or "").strip():
                    return True
            return False

        def _strip_leading_generated_assistant_role(text: str) -> str:
            return re.sub(
                r"^\s*(?:<\|im_start\|>\s*)?assistant(?:\s*<\|im_end\|>)?\s*\n+",
                "",
                str(text or ""),
                count=1,
                flags=re.IGNORECASE,
            ).strip()

        def _truncate_after_first_synthetic_role_turn(text: str) -> str:
            normalized_text = _strip_leading_generated_assistant_role(text)
            role_line_pattern = re.compile(
                r"(?im)^\s*(?:<\|im_start\|>\s*)?(system|user|tool)(?:\s*<\|im_end\|>)?\s*$"
            )
            for match in role_line_pattern.finditer(normalized_text):
                prefix = normalized_text[: match.start()].strip()
                if not prefix:
                    continue
                return prefix
            return normalized_text

        def _extract_effective_json_payload(text: str) -> Optional[str]:
            payload_block = agent._extract_json_payload_block(text)
            if payload_block is None:
                return None
            payload_text, _, _ = payload_block
            payload_text = str(payload_text or "").strip()
            return payload_text or None

        def _normalize_llm_response_for_role(raw_text: Any) -> str:
            normalized = agent.sanitize_response(raw_text)
            candidates = [normalized]
            truncated_transcript = _truncate_after_first_synthetic_role_turn(normalized)
            if truncated_transcript and truncated_transcript != normalized:
                candidates.append(truncated_transcript)
            cleaned_without_think = _strip_think_markup(normalized, drop_block_content=True)
            if cleaned_without_think != normalized:
                candidates.append(cleaned_without_think)
            cleaned_truncated_without_think = _strip_think_markup(truncated_transcript, drop_block_content=True)
            if cleaned_truncated_without_think and cleaned_truncated_without_think not in candidates:
                candidates.append(cleaned_truncated_without_think)

            for candidate in candidates:
                payload_text = _extract_effective_json_payload(candidate)
                if not payload_text:
                    continue
                return payload_text

            return truncated_transcript or normalized

        def _normalize_llm_response_for_log(
            raw_text: Any,
            *,
            enable_thinking: Optional[bool],
        ) -> str:
            normalized = agent.sanitize_response(raw_text)
            candidates: List[str] = []
            truncated_transcript = _truncate_after_first_synthetic_role_turn(normalized)
            if truncated_transcript:
                candidates.append(truncated_transcript)
            if normalized and normalized not in candidates:
                candidates.append(normalized)

            normalized_candidates: List[str] = []
            for candidate in candidates:
                cleaned_candidate = candidate
                if enable_thinking is False:
                    cleaned_candidate = _strip_empty_think_blocks(cleaned_candidate)
                if cleaned_candidate and cleaned_candidate not in normalized_candidates:
                    normalized_candidates.append(cleaned_candidate)

            for candidate in normalized_candidates:
                if _extract_effective_json_payload(candidate):
                    return candidate
            return normalized_candidates[0] if normalized_candidates else (truncated_transcript or normalized)

        def _log_llm_response_sanitization(
            *,
            iteration_idx: int,
            raw_response_text: str,
            log_response_text: str,
            effective_response_text: str,
            phase_tag: str = "",
        ) -> None:
            if (
                raw_response_text == log_response_text
                and log_response_text == effective_response_text
            ):
                return

            notes: List[str] = []
            if raw_response_text != log_response_text:
                if _strip_empty_think_blocks(raw_response_text) == log_response_text:
                    notes.append("suppressed empty no-think <think> scaffold in log display")
                else:
                    notes.append("trimmed echoed prompt / generated role transcript for log display")
            if log_response_text != effective_response_text:
                notes.append("preserved <think> blocks in logs while extracting effective JSON for execution")
            if not notes:
                notes.append("normalized response for execution")

            logger.info(
                "%s[%s][LLM_RESPONSE_SANITIZED][Depth %d][Iter %d]%s %s (%d raw -> %d log -> %d effective chars). "
                "Full raw response is preserved in raw trace JSONL.",
                '  ' * depth,
                role_label,
                depth,
                iteration_idx,
                phase_tag,
                "; ".join(notes),
                len(raw_response_text or ""),
                len(log_response_text or ""),
                len(effective_response_text or ""),
            )

        def _log_unexpected_think_when_disabled(
            *,
            iteration_idx: int,
            log_response_text: str,
            enable_thinking: Optional[bool],
            phase_tag: str = "",
        ) -> None:
            if enable_thinking is not False:
                return
            if not _contains_nonempty_think_block(log_response_text):
                return
            logger.warning(
                "%s[%s][LLM_THINK_UNEXPECTED][Depth %d][Iter %d]%s response contains non-empty <think>...</think> even though enable_thinking=False.",
                '  ' * depth,
                role_label,
                depth,
                iteration_idx,
                phase_tag,
            )

        def _truncate_text(text: Any, max_chars: int = 1200) -> str:
            return _truncate_history_text(text, max_chars=max_chars)

        def _activate_finish_recovery(
            reason: str,
            targets: Optional[List[Any]] = None,
        ) -> None:
            """Escalate a rejected Finish into a shared bounded recovery mode."""
            nonlocal active_recovery_system_directive
            nonlocal finish_recovery_mode, finish_recovery_finish_pending
            nonlocal finish_rejection_streak
            nonlocal finish_recovery_last_reason, finish_recovery_last_targets
            if depth != 0 or not evidence_first_mode:
                return
            finish_recovery_mode = True
            finish_recovery_finish_pending = False
            finish_rejection_streak += 1
            finish_recovery_last_reason = _truncate_text(reason, max_chars=280)
            normalized_targets: List[str] = []
            for target in list(targets or []):
                target_text = _truncate_text(target, max_chars=180)
                if target_text and target_text not in normalized_targets:
                    normalized_targets.append(target_text)
                if len(normalized_targets) >= 3:
                    break
            finish_recovery_last_targets = normalized_targets
            target_text = "; ".join(normalized_targets) or "the latest evidence gap"
            active_recovery_system_directive = (
                "[ACTIVE FINISH RECOVERY DIRECTIVE]\n"
                f"Recovery escalation {finish_rejection_streak}; bounded recovery budget="
                f"{max_finish_recovery_rounds} planner turn(s). Trigger: "
                f"{finish_recovery_last_reason or 'Finish was rejected'}.\n"
                "Planner and Reflector share this state. The next Planner turn must "
                "execute one concrete non-Finish evidence action with materially "
                "changed tool, arguments, modality, or time window; do not resubmit "
                "the same Finish candidate or merely rewrite self-reported Evidence. "
                "Prefer a probe that directly resolves these targets: "
                f"{target_text}. A failed, blocked, policy-only, or runtime-error "
                "action is not new evidence. Do not call a case unanswerable merely "
                "because the Planner budget or a tool failed. After a successful "
                "grounded observation, reassess once and use the forced Finish-only "
                "boundary when requested. Only a verified premise/option mismatch "
                "or exhausted genuine evidence recovery may end in evidence_unanswerable."
            )
            logger.warning(
                "%s[Planner][Depth %d] Finish recovery escalation=%d/%d reason=%s targets=%s",
                "  " * depth,
                depth,
                finish_rejection_streak,
                max_finish_recovery_rounds,
                finish_recovery_last_reason,
                normalized_targets or ["latest evidence gap"],
            )
            agent.trajectory.add_debug_event({
                "type": "finish_recovery_escalation",
                "event_type": "FinishRecoveryEscalationEvent",
                "depth": depth,
                "iteration": iteration,
                "reason": finish_recovery_last_reason,
                "targets": list(normalized_targets),
                "rejection_streak": finish_rejection_streak,
                "max_recovery_rounds": max_finish_recovery_rounds,
                "system_directive_applied": True,
            })

        def _json_safe_for_log(value: Any, max_text_chars: int = 4000) -> Any:
            if isinstance(value, dict):
                serialized: Dict[str, Any] = {}
                for key, item in value.items():
                    normalized_key = str(key or "")
                    if normalized_key.lower() in {"data", "bytes", "buffer"}:
                        continue
                    serialized[normalized_key] = _json_safe_for_log(item, max_text_chars=max_text_chars)
                return serialized
            if isinstance(value, (list, tuple)):
                return [
                    _json_safe_for_log(item, max_text_chars=max_text_chars)
                    for item in value
                ]
            if isinstance(value, str):
                if value.startswith("data:"):
                    return _truncate_text(value, max_chars=min(200, max_text_chars)) + "...<truncated_data_url>"
                return _truncate_text(value, max_chars=max_text_chars)
            if value is None or isinstance(value, (int, float, bool)):
                return value
            return _truncate_text(str(value), max_chars=max_text_chars)

        def _log_first_input_payload(
            messages_payload: List[Dict[str, Any]],
            generation_payload: Dict[str, Any],
        ) -> None:
            try:
                serialized_messages = json.dumps(
                    _json_safe_for_log(messages_payload),
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception as exc:
                serialized_messages = f"<failed to serialize first input messages: {exc}>"
            logger.debug(
                "%s[%s][FIRST_INPUT_MESSAGES_BEGIN][Depth %d]\n%s\n%s[%s][FIRST_INPUT_MESSAGES_END][Depth %d]",
                '  ' * depth,
                role_label,
                depth,
                serialized_messages,
                '  ' * depth,
                role_label,
                depth,
            )

            prompt_builder = getattr(current_llm, "_build_prompt", None)
            if callable(prompt_builder):
                try:
                    rendered_prompt = prompt_builder(
                        messages_payload,
                        enable_thinking=generation_payload.get("enable_thinking"),
                    )
                except TypeError:
                    try:
                        rendered_prompt = prompt_builder(messages_payload)
                    except Exception as exc:
                        rendered_prompt = f"<failed to render first input prompt: {exc}>"
                except Exception as exc:
                    rendered_prompt = f"<failed to render first input prompt: {exc}>"
                if str(rendered_prompt or "").strip():
                    logger.debug(
                        "%s[%s][FIRST_INPUT_RENDERED_PROMPT_BEGIN][Depth %d]\n%s\n%s[%s][FIRST_INPUT_RENDERED_PROMPT_END][Depth %d]",
                        '  ' * depth,
                        role_label,
                        depth,
                        rendered_prompt,
                        '  ' * depth,
                        role_label,
                        depth,
                    )

            try:
                serialized_generation = json.dumps(
                    _json_safe_for_log(generation_payload, max_text_chars=800),
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception as exc:
                serialized_generation = f"<failed to serialize generation kwargs: {exc}>"
            logger.debug(
                "%s[%s][FIRST_INPUT_GENERATION_KWARGS][Depth %d]\n%s",
                '  ' * depth,
                role_label,
                depth,
                serialized_generation,
            )

        def _log_observation_preview(label: str, observation_text: str, max_chars: int = 320) -> None:
            preview = _truncate_history_text(observation_text, max_chars=max_chars)
            if preview:
                logger.info(
                    "%s[%s][Depth %d] %s: %s",
                    '  ' * depth,
                    role_label,
                    depth,
                    label,
                    preview,
                )

        def _extract_parse_failure_note(raw_response: str, payload: Optional[Any]) -> str:
            note_chunks: List[str] = []
            if isinstance(payload, dict):
                evidence = payload.get("Evidence") or payload.get("evidence")
                if isinstance(evidence, list) and evidence:
                    try:
                        note_chunks.append(json.dumps({"Evidence": evidence}, ensure_ascii=False))
                    except Exception:
                        pass
                for key in ("summary", "Summary", "result", "Result", "answer", "Answer", "notes", "Notes"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        note_chunks.append(value.strip())

            cleaned = re.sub(r"<think>.*?</think>", "", raw_response or "", flags=re.DOTALL).strip()
            cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()
            if cleaned:
                note_chunks.append(cleaned)

            if not note_chunks:
                return ""
            merged = " ".join(_truncate_history_text(chunk, 900) for chunk in note_chunks if chunk)
            merged = re.sub(r"\s+", " ", merged).strip()
            return _truncate_history_text(merged, 1200)

        def _stable_signature(payload: Any) -> str:
            try:
                if isinstance(payload, str):
                    canonical = re.sub(r"\s+", " ", payload.strip().lower())
                else:
                    canonical = json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    canonical = re.sub(r"\s+", " ", canonical.lower()).strip()
            except Exception:
                canonical = re.sub(r"\s+", " ", str(payload or "").strip().lower())

            if not canonical:
                return ""
            return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

        def _extract_primary_probe_target(payload: Any, observation_text: str = "") -> str:
            if isinstance(payload, str):
                try:
                    parsed_payload = json.loads(payload)
                except Exception:
                    parsed_payload = None
                if parsed_payload is not None:
                    return _extract_primary_probe_target(parsed_payload, observation_text=observation_text)

            if isinstance(payload, list):
                for item in payload:
                    target = _extract_primary_probe_target(item, observation_text=observation_text)
                    if target:
                        return target

            if isinstance(payload, dict):
                for key in ("search_window", "time_range", "time", "timestamp", "segment_ref", "segment_id", "raw_segment_id", "source_segment"):
                    value = payload.get(key)
                    if value not in (None, "", [], {}):
                        return str(value).strip()
                t_start = payload.get("t_start")
                t_end = payload.get("t_end")
                if t_start not in (None, "") or t_end not in (None, ""):
                    return f"{t_start}-{t_end}".strip("-")
                matched_segments = payload.get("matched_segments")
                if isinstance(matched_segments, list) and matched_segments:
                    target = _extract_primary_probe_target(matched_segments[0], observation_text=observation_text)
                    if target:
                        return target
                items = payload.get("items")
                if isinstance(items, list) and items:
                    target = _extract_primary_probe_target(items[0], observation_text=observation_text)
                    if target:
                        return target

            text = str(observation_text or "").strip()
            if not text:
                return ""
            match = re.search(r"\b\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\b", text)
            if match:
                return match.group(0)
            return ""

        def _extract_probe_window(payload: Any, observation_text: str = "") -> Optional[Tuple[float, float]]:
            def _from_value(value: Any) -> Optional[Tuple[float, float]]:
                if value in (None, "", [], {}):
                    return None
                if isinstance(value, (int, float)):
                    numeric = float(value)
                    return (numeric, numeric)
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    try:
                        start = float(value[0])
                        end = float(value[1])
                    except Exception:
                        return None
                    return (min(start, end), max(start, end))
                text = str(value).strip()
                if not text:
                    return None
                match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", text)
                if match:
                    start = float(match.group(1))
                    end = float(match.group(2))
                    return (min(start, end), max(start, end))
                try:
                    numeric = float(text)
                except Exception:
                    return None
                return (numeric, numeric)

            if isinstance(payload, str):
                try:
                    parsed_payload = json.loads(payload)
                except Exception:
                    parsed_payload = None
                if parsed_payload is not None:
                    return _extract_probe_window(parsed_payload, observation_text=observation_text)

            if isinstance(payload, list):
                for item in payload:
                    window = _extract_probe_window(item, observation_text=observation_text)
                    if window is not None:
                        return window

            if isinstance(payload, dict):
                for key in ("search_window", "time_range", "timestamp_range", "time", "timestamp"):
                    window = _from_value(payload.get(key))
                    if window is not None:
                        return window
                t_start = payload.get("t_start")
                if t_start in (None, ""):
                    t_start = payload.get("time_range_start", payload.get("start_time"))
                t_end = payload.get("t_end")
                if t_end in (None, ""):
                    t_end = payload.get("time_range_end", payload.get("end_time"))
                if t_start not in (None, "") or t_end not in (None, ""):
                    try:
                        start = float(t_start if t_start not in (None, "") else t_end)
                        end = float(t_end if t_end not in (None, "") else t_start)
                    except Exception:
                        start = end = None
                    if start is not None and end is not None:
                        return (min(start, end), max(start, end))
                matched_segments = payload.get("matched_segments")
                if isinstance(matched_segments, list) and matched_segments:
                    window = _extract_probe_window(matched_segments[0], observation_text=observation_text)
                    if window is not None:
                        return window
                items = payload.get("items")
                if isinstance(items, list) and items:
                    window = _extract_probe_window(items[0], observation_text=observation_text)
                    if window is not None:
                        return window

            # Free-form observations can contain sample IDs or image sizes.
            # Only a labelled temporal range is usable as a recovery window.
            labelled_range = re.search(
                r"(?:search_window|time_range|time window|window)\s*[:=]\s*[\"']?(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?)",
                str(observation_text or ""), re.IGNORECASE,
            )
            return _from_value(labelled_range.group(1)) if labelled_range else None

        def _windows_overlap_or_adjacent(
            left: Optional[Tuple[float, float]],
            right: Optional[Tuple[float, float]],
        ) -> bool:
            if left is None or right is None:
                return False
            l0, l1 = left
            r0, r1 = right
            if min(l1, r1) >= max(l0, r0):
                return True
            left_duration = max(0.0, l1 - l0)
            right_duration = max(0.0, r1 - r0)
            tolerance = max(1.0, min(5.0, 0.25 * max(left_duration, right_duration, 1.0)))
            gap = max(r0 - l1, l0 - r1)
            return gap <= tolerance

        def _normalized_probe_window_key(window: Optional[Tuple[float, float]]) -> str:
            if window is None:
                return ""
            start, end = window
            return f"{start:.3f}-{end:.3f}"

        def _payload_has_image_evidence(payload: Any) -> bool:
            if isinstance(payload, dict):
                images = payload.get("image_evidence")
                if isinstance(images, list) and images:
                    return True
                for key in ("items", "matched_segments"):
                    value = payload.get(key)
                    if isinstance(value, list) and any(_payload_has_image_evidence(item) for item in value):
                        return True
                return False
            if isinstance(payload, list):
                return any(_payload_has_image_evidence(item) for item in payload)
            return False

        def _materially_different_probe_intent(
            current_args: Mapping[str, Any],
            previous_args: Mapping[str, Any],
        ) -> bool:
            """Allow a new question over an already returned frame window.

            The image-level duplicate guard is useful for accidental blind
            re-inspection, but the old window-only rule also blocked a second
            question about the same frame.  A changed query, crop/region, or
            density is a real evidence request even when its time bounds are
            contained in the previous call.
            """
            if not isinstance(current_args, Mapping) or not isinstance(previous_args, Mapping):
                return False
            for key in ("image_region", "region", "crop", "zoom", "density", "frame_indices"):
                if current_args.get(key) not in (None, "", [], {}) and current_args.get(key) != previous_args.get(key):
                    return True
            def _intent_tokens(args: Mapping[str, Any]) -> set[str]:
                raw = args.get("query") or args.get("question") or args.get("description") or ""
                return {
                    token
                    for token in re.findall(r"[a-z0-9]+", str(raw).lower())
                    if len(token) > 2
                }
            current_tokens = _intent_tokens(current_args)
            previous_tokens = _intent_tokens(previous_args)
            if not current_tokens or not previous_tokens:
                return False
            overlap = len(current_tokens.intersection(previous_tokens))
            union = len(current_tokens.union(previous_tokens))
            return union > 0 and (union - overlap >= 2 or overlap / union < 0.70)

        def _detect_redundant_visual_reprobe(action_name: str, action_args: Dict[str, Any]) -> Optional[str]:
            if depth != 0 or str(action_name or "").strip() not in {"Inspect_Frame", "OCR_Extract"}:
                return None
            if not isinstance(last_execution_meta, dict):
                return None

            previous_tool = str(last_execution_meta.get("tool", "")).strip()
            if previous_tool not in {"Inspect_Frame", "OCR_Extract", "Action_Recognition", "Emotion_Analysis", "Entity_Track"}:
                return None

            previous_payload = last_execution_meta.get("payload")
            if not _payload_has_image_evidence(previous_payload):
                return None

            if _materially_different_probe_intent(
                action_args,
                last_execution_meta.get("args", {})
                if isinstance(last_execution_meta.get("args", {}), Mapping)
                else {},
            ):
                return None

            current_window = _extract_probe_window(action_args)
            previous_window = _extract_probe_window(
                previous_payload,
                observation_text=str(last_execution_meta.get("observation", "")),
            )
            if current_window is None or previous_window is None:
                return None
            if not _windows_overlap_or_adjacent(previous_window, current_window):
                return None

            current_within_previous = (
                current_window[0] >= previous_window[0] - 1e-6
                and current_window[1] <= previous_window[1] + 1e-6
            )
            if not current_within_previous:
                return None

            return (
                f"Error: {previous_tool} already returned attached image evidence for this same time window. "
                "Those images are directly visible to the planner, so do not call another raw image-returning tool "
                f"({action_name}) on the same timestamps just to re-inspect them. Either reason from the attached "
                "images, apply Crop/Zoom_in to a returned image, or move to a materially different window."
            )

        def _build_progress_signature(
            action_name: str,
            observation_text: str,
            observation_payload: Any = None,
        ) -> str:
            family = _infer_progress_family(action_name, observation_payload)
            payload = observation_payload if observation_payload not in (None, "") else observation_text
            result_text = extract_planner_result_text(payload, tool_name=str(action_name or ""))
            target = _extract_primary_probe_target(payload, observation_text=observation_text)
            result_sig = _stable_signature(result_text or observation_text)
            return "|".join([family or "other", target or "no_target", result_sig or "no_result"])

        def _infer_progress_family(action_name: str, observation_payload: Any = None) -> str:
            family = _action_family(action_name)
            if family != "parallel":
                return family

            items: List[Any] = []
            if isinstance(observation_payload, dict) and isinstance(observation_payload.get("items"), list):
                items = observation_payload.get("items", [])
            elif isinstance(observation_payload, list):
                items = observation_payload

            inferred_families: List[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_tool = str(item.get("tool", "")).strip()
                item_family = _action_family(item_tool)
                if item_family not in {"parallel", "other", "finish"}:
                    inferred_families.append(item_family)

            if inferred_families:
                return inferred_families[0]
            return family

        def _planner_visual_attachment_limit(text_hint: str = "") -> int:
            token_budget = max(
                1024,
                int(getattr(settings, "planner_visual_observation_token_budget", 12288) or 12288),
            )
            image_token_estimate = max(
                256,
                int(getattr(settings, "planner_visual_observation_image_token_estimate", 1536) or 1536),
            )
            text_tokens = max(256, len(" ".join(str(text_hint or "").split())) // 4)
            remaining_budget = max(image_token_estimate, token_budget - text_tokens)
            return max(1, remaining_budget // image_token_estimate)

        def _limit_visual_records(records: List[Dict[str, Any]], text_hint: str) -> List[Dict[str, Any]]:
            if not records:
                return records
            if bool(getattr(settings, "preserve_full_observations", False)):
                return records
            max_images = _planner_visual_attachment_limit(text_hint=text_hint)
            remaining_budget = max_images
            limited: List[Dict[str, Any]] = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                cloned = dict(record)
                images = [
                    item for item in (record.get("image_evidence", []) or [])
                    if isinstance(item, dict)
                ]
                if remaining_budget <= 0:
                    cloned["image_evidence"] = []
                    if "image_count" in cloned:
                        cloned["image_count"] = 0
                elif images:
                    kept = images[:remaining_budget]
                    cloned["image_evidence"] = kept
                    cloned["image_count"] = len(kept)
                    remaining_budget -= len(kept)
                limited.append(cloned)
            return limited

        def _collect_record_image_evidence(payload: Any) -> List[Dict[str, Any]]:
            collected: List[Dict[str, Any]] = []

            def _walk(value: Any) -> None:
                if isinstance(value, dict):
                    images = value.get("image_evidence")
                    if isinstance(images, list):
                        for image_item in images:
                            if isinstance(image_item, dict):
                                collected.append(image_item)
                    for key, nested in value.items():
                        if key == "image_evidence":
                            continue
                        _walk(nested)
                    return
                if isinstance(value, list):
                    for nested in value:
                        _walk(nested)

            _walk(payload)
            return collected

        def _append_observation_json(
            items: List[Dict[str, Any]],
            *,
            commit_planner_evidence: bool = False,
        ) -> None:
            # Observer feedback updates the solve-scope evidence ledger.  This
            # nested helper must explicitly write to the outer ledger; without
            # `nonlocal`, Python treats the merge assignment below as a local
            # binding and raises UnboundLocalError on the first Observer turn.
            nonlocal evidence_slots, evidence_revision, pending_planner_evidence_slots
            nonlocal repeated_loop_violations
            nonlocal finish_recovery_violations
            nonlocal finish_recovery_mode, finish_recovery_turns_used
            nonlocal finish_recovery_finish_pending
            nonlocal finish_rejection_streak, active_recovery_system_directive
            nonlocal final_answer, success, finished, no_progress_abort_reason
            nonlocal low_signal_visual_probe_streak
            nonlocal low_signal_guard_recovery_count
            nonlocal low_signal_recovery_triggered_this_observation
            normalized: List[Dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                record = build_planner_observation_record(
                    tool_name=str(item.get("tool", "Unknown")),
                    result=item.get("result", ""),
                    status=str(item.get("status") or "success"),
                )
                sanitized_image_evidence: List[Dict[str, Any]] = []
                for image_item in record.get("image_evidence", []) or []:
                    prepared_item = prepare_planner_image_attachment(image_item)
                    if prepared_item is not None:
                        sanitized_image_evidence.append(prepared_item)
                if sanitized_image_evidence or record.get("image_evidence"):
                    image_stats = summarize_planner_image_evidence(sanitized_image_evidence)
                    record["image_evidence"] = sanitized_image_evidence
                    record["image_count"] = image_stats.get("image_count", 0)
                    record["source_image_count"] = image_stats.get("source_image_count", 0)
                    record["image_pack_modes"] = image_stats.get("image_pack_modes", [])
                    record["contains_packed_collage"] = image_stats.get("contains_packed_collage", False)
                    record["image_pack_summary"] = image_stats.get("image_pack_summary", "")
                    record["image_timestamps"] = image_stats.get("image_timestamps", [])
                    record["image_order_rule"] = image_stats.get("image_order_rule")
                normalized.append(record)

            if not normalized:
                normalized = [{"tool": "Unknown", "status": "error", "result": "Empty observation"}]

            payload_preview = json.dumps({"Observation": normalized}, ensure_ascii=False)
            normalized = _limit_visual_records(normalized, text_hint=payload_preview)

            def _flatten_tool_records(value: Any) -> List[Mapping[str, Any]]:
                records: List[Mapping[str, Any]] = []
                if isinstance(value, Mapping):
                    if str(value.get("tool") or "").strip():
                        records.append(value)
                    for nested_key in ("items", "result", "payload"):
                        records.extend(_flatten_tool_records(value.get(nested_key)))
                elif isinstance(value, list):
                    for nested in value:
                        records.extend(_flatten_tool_records(nested))
                return records

            meaningful_success = any(
                str(item.get("tool") or "").strip()
                not in {"System", "Evidence_Verifier", "Assistant_Note", "Unknown", "Finish"}
                and str(item.get("status") or "success").strip().lower() == "success"
                for item in normalized
                if isinstance(item, dict)
            )
            concrete_evidence_tools = frozenset({
                "Semantic_Retrieve", "Keyword_Retrieve", "Segment_Retrieve", "Graph_Search",
                "Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in", "OCR_Extract",
                "Video_Clip_QA", "Video_Summary", "Action_Recognition", "Emotion_Analysis",
                "Entity_Track", "Cross_Modal_Verify", "asr",
            })

            canonical_tool_names = {
                name.casefold(): name
                for name in concrete_evidence_tools.union({
                    "Python_Executor", "SQL_Executor", "Video_Summary",
                })
            }

            def _canonical_tool_name(value: Any) -> str:
                text = str(value or "").strip()
                return canonical_tool_names.get(text.casefold(), text)

            def _contains_concrete_evidence_success(value: Any) -> bool:
                if isinstance(value, Mapping):
                    tool_name = _canonical_tool_name(value.get("tool"))
                    status = str(value.get("status") or "success").strip().lower()
                    if tool_name in concrete_evidence_tools and status == "success":
                        return True
                    return any(
                        _contains_concrete_evidence_success(value.get(nested_key))
                        for nested_key in ("items", "result", "payload")
                    )
                if isinstance(value, list):
                    return any(_contains_concrete_evidence_success(item) for item in value)
                return False

            concrete_evidence_success = any(
                _contains_concrete_evidence_success(item)
                for item in normalized
                if isinstance(item, dict)
            )
            low_signal_recovery_was_triggered = low_signal_recovery_triggered_this_observation
            low_signal_recovery_triggered_this_observation = False
            evidence_changed = False
            grounded_observation_added = False

            # A successful tool call is only a progress boundary when it
            # contributes a claim that can be tied back to an executed
            # observation. Retrieval hits and empty visual packs must not
            # advance evidence_revision or authorize a Finish-only retry.
            evidence_source_steps = [
                *agent.trajectory.steps,
                {"observation_items": normalized},
            ]

            def _slot_has_grounded_observation(slot: Mapping[str, Any]) -> bool:
                status = str(slot.get("evidence_status") or "").strip().lower()
                if status not in {"verified", "exact", "approximate", "approx"}:
                    return False
                uncertainty = str(slot.get("uncertainty") or "").strip().lower()
                if any(
                    marker in uncertainty
                    for marker in (
                        "uncertain", "unclear", "unknown", "insufficient",
                        "ambiguous", "maybe", "possibly", "not sure",
                        "low confidence", "medium confidence",
                    )
                ):
                    return False
                return _slot_has_executed_source(slot, evidence_source_steps)

            if pending_planner_evidence_slots and (concrete_evidence_success or commit_planner_evidence):
                previous_slots = list(evidence_slots)
                evidence_slots = _merge_evidence_slots(
                    evidence_slots,
                    pending_planner_evidence_slots,
                    query=goal,
                    source="planner_after_observation",
                )
                memory_state["evidence_slots"] = list(evidence_slots)
                evidence_changed = _stable_signature(previous_slots) != _stable_signature(evidence_slots)
                grounded_observation_added = grounded_observation_added or any(
                    _slot_has_grounded_observation(slot)
                    for slot in pending_planner_evidence_slots
                )
            # A failed/blocked action is not a recovery boundary.  Clear its
            # uncommitted self-report so it cannot authorize a Finish retry.
            pending_planner_evidence_slots = []
            if meaningful_success:
                finish_recovery_violations = 0
                # A real successful observation is also the loop-progress
                # boundary.  Repeated policy counters must not accumulate
                # across unrelated successful probes.
                repeated_loop_violations = 0
            if grounded_observation_added:
                finish_recovery_violations = 0
                # If this evidence arrived during an already-authorized
                # post-limit recovery turn, keep one bounded Finish-only turn
                # active so the Planner can submit Finish against the new
                # ledger. Do not force another probe after real evidence.
                if low_signal_recovery_was_triggered:
                    # A grounded claim arrived after the low-signal guard.
                    # Preserve one bounded Finish-only turn so the Planner
                    # can submit the answer against this newly committed
                    # ledger instead of launching another probe.
                    finish_recovery_mode = True
                    finish_recovery_finish_pending = True
                elif finish_recovery_turns_used <= 0:
                    finish_recovery_mode = False
                    finish_recovery_finish_pending = False
                    active_recovery_system_directive = ""
                else:
                    finish_recovery_mode = True
                    finish_recovery_finish_pending = True
                    active_recovery_system_directive = (
                        "[ACTIVE FINISH RECOVERY DIRECTIVE]\n"
                        "A grounded observation was added during bounded recovery. "
                        "The next Planner turn is Finish-only: use `Actions: []` and "
                        "submit the best evidence-backed top-level Finish. Do not call "
                        "another tool unless the verifier identifies a concrete new "
                        "claim gap; a prior runtime/tool error is not evidence_unanswerable."
                    )
                finish_rejection_streak = 0
            if evidence_changed and grounded_observation_added:
                evidence_revision += 1
            payload = {
                "Observation": normalized
            }
            history_content: Any = json.dumps(payload, ensure_ascii=False)
            visual_items: List[Dict[str, Any]] = []
            seen_visual_refs: set[str] = set()
            for record in normalized:
                if not isinstance(record, dict):
                    continue
                for image_item in _collect_record_image_evidence(record):
                    prepared_item = prepare_planner_image_attachment(image_item)
                    if prepared_item is None:
                        continue
                    image_ref = str(prepared_item.get("image", "") or prepared_item.get("path", "") or "").strip()
                    if not image_ref or image_ref in seen_visual_refs:
                        continue
                    seen_visual_refs.add(image_ref)
                    visual_item = {"type": "image", "image": image_ref}
                    source_images = list(prepared_item.get("source_images") or [])
                    if source_images:
                        visual_item["source_images"] = source_images
                    if prepared_item.get("timestamp") is not None:
                        visual_item["timestamp"] = prepared_item.get("timestamp")
                    if isinstance(prepared_item.get("sample_times"), list) and prepared_item.get("sample_times"):
                        visual_item["sample_times"] = list(prepared_item.get("sample_times") or [])
                    if prepared_item.get("max_pixels") not in (None, "", []):
                        visual_item["max_pixels"] = prepared_item.get("max_pixels")
                    visual_items.append(visual_item)
            if visual_items:
                history_content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}] + visual_items
            _append_history("user", history_content)

            # The Observer is a real VLM role.  It sees the normalized result
            # after the runtime has preserved tool semantics and feeds one
            # structured next-action recommendation back to the Planner.
            observer_items = [
                item
                for item in normalized
                if str(item.get("tool", "")).strip()
                not in {"System", "Evidence_Verifier", "Assistant_Note", "Unknown"}
            ]
            observer_slots: List[Dict[str, Any]] = []
            native_fallback_slots: List[Dict[str, Any]] = []
            if depth == 0 and observer_items and "observer" in self.disabled_agent_roles:
                agent.trajectory.add_debug_event({
                    "type": "agent_role_skipped",
                    "event_type": "ObserverAgentTurnEvent",
                    "agent_role": "observer",
                    "status": "disabled",
                    "schema_valid": False,
                    "depth": depth,
                    "iteration": iteration,
                    "observed_tools": [str(item.get("tool", "")) for item in observer_items],
                    "reason": "online_role_ablation",
                })
                logger.info(
                    "[Observer] role disabled by online ablation iteration=%s",
                    iteration,
                )
            if depth == 0 and observer_items and "observer" not in self.disabled_agent_roles:
                observer_deadline = time.perf_counter() + 90.0
                if self.wall_clock_deadline_monotonic is not None:
                    observer_deadline = min(observer_deadline, self.wall_clock_deadline_monotonic)
                observer_result = self.role_agents.observe(
                    goal=goal,
                    observation_items=observer_items,
                    trajectory_tail=[step.to_dict() for step in agent.trajectory.steps[-3:]],
                    deadline_monotonic=observer_deadline,
                )
                observer_event = dict(observer_result.get("event") or {})
                observer_event.update({
                    "depth": depth,
                    "iteration": iteration,
                    "observed_tools": [str(item.get("tool", "")) for item in observer_items],
                })
                runtime_session.record_timing(
                    "observer.llm_generate_seconds",
                    float(observer_event.get("elapsed_seconds") or 0.0),
                )
                agent.trajectory.add_debug_event(observer_event)
                observer_payload = observer_result.get("payload") or {}
                if not isinstance(observer_payload, Mapping):
                    logger.warning(
                        "[Observer] normalized non-mapping payload to empty object type=%s",
                        type(observer_payload).__name__,
                    )
                    observer_payload = {}
                if observer_result.get("schema_valid"):
                    memory_state["observer_feedback"] = observer_payload
                    # Observer evidence is a first-class evidence update, not
                    # merely a routing hint.  Without merging these slots,
                    # Reflector only sees the Planner's last proposed claim
                    # and can report that Inspect_Frame returned timestamps
                    # even when the Observer supplied grounded visual facts.
                    observer_slots = _extract_evidence_slots(
                        {"Evidence": observer_payload.get("evidence_updates", [])}
                    )
                    observer_slots = _bind_observer_visual_sources(observer_slots, observer_items)
                    if observer_slots:
                        previous_slots = list(evidence_slots)
                        evidence_slots = _merge_evidence_slots(
                            evidence_slots,
                            observer_slots,
                            query=goal,
                            source="observer",
                        )
                        memory_state["evidence_slots"] = list(evidence_slots)
                        if _stable_signature(previous_slots) != _stable_signature(evidence_slots):
                            evidence_changed = True
                        grounded_observation_added = grounded_observation_added or any(
                            _slot_has_grounded_observation(slot)
                            for slot in observer_slots
                        )
                    _append_history(
                        "user",
                        "[Observer Agent Feedback]\n" + json.dumps(
                            observer_payload,
                            ensure_ascii=False,
                        ),
                    )
                elif observer_items:
                    # Preserve executable native answers even when the
                    # auxiliary Observer response is malformed/truncated.
                    # The fallback copies structured Video_Clip_QA fields only;
                    # it does not turn a failed Observer or a Python narration
                    # into visual evidence.
                    native_fallback_slots = _native_fallback_evidence_slots(
                        goal=goal,
                        observation_items=observer_items,
                    )
                    if native_fallback_slots:
                        previous_slots = list(evidence_slots)
                        evidence_slots = _merge_evidence_slots(
                            evidence_slots,
                            _extract_evidence_slots({"Evidence": native_fallback_slots}),
                            query=goal,
                            source="native_tool_fallback",
                        )
                        memory_state["evidence_slots"] = list(evidence_slots)
                        if _stable_signature(previous_slots) != _stable_signature(evidence_slots):
                            evidence_changed = True
                        grounded_observation_added = grounded_observation_added or any(
                            _slot_has_grounded_observation(slot)
                            for slot in native_fallback_slots
                        )
                        agent.trajectory.add_debug_event({
                            "type": "native_evidence_fallback",
                            "event_type": "NativeEvidenceFallbackEvent",
                            "depth": depth,
                            "iteration": iteration,
                            "source_tools": sorted({
                                str(slot.get("source_tool") or "")
                                for slot in native_fallback_slots
                                if str(slot.get("source_tool") or "")
                            }),
                            "evidence_count": len(native_fallback_slots),
                            "reason": "observer_schema_invalid",
                        })
                logger.info(
                    "[Observer] feedback status=%s schema_valid=%s iteration=%s",
                    observer_result.get("status"),
                    observer_result.get("schema_valid"),
                    iteration,
                )

            # A changing timestamp is not the same as a changing evidence
            # state.  Bound the common failure mode where the Planner scans
            # unrelated windows, receives only empty visual/QA results, and
            # eventually guesses an option.  Retrieval alone is not semantic
            # progress: it must not reset the guard unless the Observer
            # contributes a grounded evidence update.
            if (
                depth == 0
                and evidence_first_mode
                and observer_items
                and "observer" not in self.disabled_agent_roles
            ):
                tool_records = _flatten_tool_records(observer_items)
                successful_visual = any(
                    _action_family(str(item.get("tool") or "").strip()) == "visual"
                    and str(item.get("status") or "success").strip().lower() == "success"
                    for item in tool_records
                )
                grounded_observer_slots = [
                    slot
                    for slot in observer_slots
                    if _slot_has_grounded_observation(slot)
                ]
                grounded_observer_slots.extend(
                    slot
                    for slot in native_fallback_slots
                    if _slot_has_grounded_observation(slot)
                )
                if grounded_observer_slots:
                    low_signal_visual_probe_streak = 0
                elif successful_visual:
                    low_signal_visual_probe_streak += 1
                memory_state["low_signal_visual_probe_streak"] = low_signal_visual_probe_streak
                if low_signal_visual_probe_streak >= max_low_signal_visual_probes:
                    if low_signal_guard_recovery_count < max_low_signal_guard_recoveries:
                        low_signal_guard_recovery_count += 1
                        low_signal_recovery_triggered_this_observation = True
                        _activate_finish_recovery(
                            "repeated low-signal Observer visual probes",
                            targets=[
                                "switch to a different evidence modality or tool family",
                                "inspect a materially different temporal window",
                                "obtain one grounded Observer claim for the unresolved question",
                            ],
                        )
                        low_signal_visual_probe_streak = 0
                        memory_state["low_signal_visual_probe_streak"] = 0
                        agent.trajectory.add_debug_event({
                            "type": "evidence_recovery_escalation",
                            "event_type": "EvidenceRecoveryEscalationEvent",
                            "depth": depth,
                            "iteration": iteration,
                            "reason": "observer_low_signal_probe_budget_exhausted",
                            "recovery_index": low_signal_guard_recovery_count,
                            "max_recoveries": max_low_signal_guard_recoveries,
                            "terminal": False,
                        })
                        logger.warning(
                            "%s[Depth %d] Observer low-signal guard escalated to alternate evidence recovery %d/%d; continuing",
                            "  " * depth,
                            depth,
                            low_signal_guard_recovery_count,
                            max_low_signal_guard_recoveries,
                        )
                        return
                    no_progress_abort_reason = (
                        "Failed: evidence_recovery_exhausted_without_grounded_observation "
                        "(observer_low_signal_probe_budget_exhausted)"
                    )
                    agent.trajectory.add_debug_event({
                        "type": "evidence_convergence_guard",
                        "event_type": "EvidenceConvergenceGuardEvent",
                        "depth": depth,
                        "iteration": iteration,
                        "issue": "low_signal_visual_probes",
                        "probe_streak": low_signal_visual_probe_streak,
                        "probe_budget": max_low_signal_visual_probes,
                        "observed_tools": [
                            str(item.get("tool") or "").strip()
                            for item in observer_items
                            if isinstance(item, Mapping)
                        ],
                        "observer_evidence_update_count": len(grounded_observer_slots),
                        "evidence_revision": evidence_revision,
                    })
                    logger.error(
                        "%s[Depth %d] Evidence convergence guard stopped %d low-signal visual probes",
                        "  " * depth,
                        depth,
                        low_signal_visual_probe_streak,
                    )
                    final_answer = no_progress_abort_reason
                    success = False
                    finished = True

        def _extract_finish_payload(payload: Optional[Dict[str, Any]]) -> Any:
            if not isinstance(payload, dict):
                return None
            finish_payload = payload.get("Finish")
            if finish_payload is None:
                finish_payload = payload.get("finish")
            return finish_payload

        def _extract_top_level_result(payload: Optional[Dict[str, Any]]) -> Any:
            if not isinstance(payload, dict):
                return None
            result_payload = payload.get("Result")
            if result_payload in (None, ""):
                result_payload = payload.get("result")
            return result_payload

        def _has_effective_finish_payload(finish_payload: Any) -> bool:
            if isinstance(finish_payload, str):
                return bool(finish_payload.strip())
            if isinstance(finish_payload, dict):
                for key in ("answer", "result", "chain_complete", "completion_basis", "reason"):
                    value = finish_payload.get(key)
                    if value not in (None, "", [], {}):
                        return True
            return False

        def _extract_structured_actions(payload: Optional[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
            if not isinstance(payload, dict):
                return []
            json_actions = (
                payload.get("Actions")
                or payload.get("actions")
                or payload.get("动作列表")
            )
            parsed_batch = agent._parse_actions_payload(json_actions)
            if parsed_batch:
                return parsed_batch
            json_action = (
                payload.get("Action")
                or payload.get("action")
                or payload.get("动作")
            )
            if json_action is None:
                return []
            parsed_name, parsed_args = agent._parse_action_text(str(json_action).strip())
            if not parsed_name:
                return []
            return [(parsed_name, parsed_args if isinstance(parsed_args, dict) else {})]

        def _is_recoverable_empty_evidence_turn(payload: Optional[Dict[str, Any]]) -> bool:
            """Recognize one bounded evidence checkpoint emitted by the root planner.

            In evidence-first mode the model can validly finish its evidence
            consolidation before it emits the terminal answer.  A strict
            root response normally includes ``Finish`` on that turn, but a
            model occasionally emits a structurally valid object with a
            non-empty Evidence array and ``Actions: []`` first.  Treating
            that exact shape as an explicit, non-terminal checkpoint lets the
            runtime request the missing Finish field without masking malformed
            JSON, empty evidence, or repeated no-op turns.
            """
            if not evidence_first_mode or depth != 0 or not isinstance(payload, dict):
                return False
            if _has_effective_finish_payload(_extract_finish_payload(payload)):
                return False
            raw_actions = None
            for key in ("Actions", "actions", "动作列表"):
                if key in payload:
                    raw_actions = payload.get(key)
                    break
            if not isinstance(raw_actions, list) or raw_actions:
                return False
            if _extract_structured_actions(payload):
                return False
            evidence = payload.get("Evidence")
            if evidence is None:
                evidence = payload.get("evidence")
            return isinstance(evidence, list) and bool(evidence)

        def _validate_structured_protocol(payload: Optional[Dict[str, Any]]) -> Optional[str]:
            if depth > 0 and not isinstance(payload, dict):
                return (
                    "Error: sub-agent responses must be exactly one strict JSON object. "
                    "Do not output `<think>` tags, `Thought:`/`Plan:` prose, markdown fences, or any text outside JSON."
                )
            if not isinstance(payload, dict):
                return None
            finish_payload = _extract_finish_payload(payload)
            top_level_result = _extract_top_level_result(payload)
            has_finish = _has_effective_finish_payload(finish_payload)
            actions = _extract_structured_actions(payload)
            has_finish_action = any(
                str(name or "").strip().lower() == "finish"
                for name, _ in actions
                if str(name or "").strip()
            )
            has_non_finish_actions = any(
                str(name or "").strip().lower() != "finish"
                for name, _ in actions
                if str(name or "").strip()
            )
            if has_finish_action:
                return (
                    "Error: `Finish` must appear only as the top-level JSON `Finish` field, "
                    "never as an item inside `Actions`."
                )
            if depth == 0 and has_finish and has_non_finish_actions:
                return (
                    "Error: root planner output mixed non-empty `Actions` with a top-level `Finish`. "
                    "Choose exactly one root template per turn: either continue with executable `Actions`, "
                    "or finish with `Actions: []` and a top-level `Finish`."
                )
            if depth > 0 and has_finish:
                return (
                    "Error: only the root planner may emit top-level `Finish`. "
                    "Sub-agents must return executable `Actions` only, and the runtime will return the tool result to the parent planner. "
                    "A top-level `Result` note is optional, but it never replaces executable `Actions`."
                )
            if mediation_mode and not actions and top_level_result not in (None, "", [], {}):
                return (
                    "Error: planner_action_mediation requires at least one executable `Action`. "
                    "A top-level `Result` note is optional, but `Result` alone cannot replace a tool call."
                )
            return None

        def _format_action_for_history(name: str, args: Dict[str, Any]) -> str:
            if not args:
                return f"{name}()"
            items = []
            for key, value in args.items():
                val_repr = json.dumps(value, ensure_ascii=False)
                items.append(f"{key}={val_repr}")
            return f"{name}({', '.join(items)})"

        def _build_actions_payload(act_name: str, act_args: Dict[str, Any]) -> List[Dict[str, Any]]:
            if act_name == "Parallel":
                batch = act_args.get("batch", []) if isinstance(act_args, dict) else []
                actions: List[Dict[str, Any]] = []
                for call in batch:
                    if not isinstance(call, dict):
                        continue
                    name = call.get("name")
                    args = call.get("args", {})
                    if isinstance(name, str):
                        actions.append({"tool": name, "params": args if isinstance(args, dict) else {}})
                return actions
            return [{"tool": act_name, "params": act_args if isinstance(act_args, dict) else {}}]

        def _append_assistant_turn(thought_text: str, plan_text: str, act_name: str, act_args: Dict[str, Any]) -> None:
            payload = {
                "Thought": (thought_text or "").strip() or "[empty]",
                "Plan": (plan_text or "").strip(),
            }
            if act_name == "Finish":
                payload["Actions"] = []
                payload["Finish"] = act_args if isinstance(act_args, dict) else {}
            else:
                payload["Actions"] = _build_actions_payload(act_name, act_args)
            if evidence_first_mode and evidence_slots:
                payload["Evidence"] = list(evidence_slots)
            _append_history("assistant", json.dumps(payload, ensure_ascii=False))

        def _is_placeholder_finish_answer(text: str) -> bool:
            t = (text or "").strip().lower()
            return any(
                marker in t for marker in (
                    "use the computed values from the observation above",
                    "use the retrieved fact from the observation above",
                    "answer this query only",
                    "your complete answer here",
                )
            )

        def _coerce_explicit_bool(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized == "true":
                    return True
                if normalized == "false":
                    return False
            return None

        def _is_placeholder_python_code(code: str) -> bool:
            c = (code or "").strip().lower()
            if not c:
                return False
            placeholder_patterns = (
                "result = 42",
                "print(result)",
                "print('hello')",
                'print("hello")',
            )
            return any(p in c for p in placeholder_patterns)

        goal_numbers = [
            n.replace(",", "")
            for n in re.findall(r"\d[\d,]*(?:\.\d+)?", goal)
        ]

        def _code_mentions_goal_numbers(code: str) -> bool:
            if not goal_numbers:
                return True
            normalized_code = (code or "").replace(",", "")
            return any(n in normalized_code for n in goal_numbers)

        def _python_script_has_meaningful_compute(code: str) -> bool:
            try:
                tree = ast.parse(str(code or ""))
            except SyntaxError:
                return False

            builtin_compute_calls = {
                "sum", "len", "max", "min", "sorted", "round",
                "abs", "all", "any", "enumerate", "zip", "range",
            }
            module_prefixes = {"json", "statistics", "math", "numpy", "np", "pandas", "pd"}
            meaningful_node_types = (
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.For,
                ast.While,
                ast.ListComp,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
                ast.Subscript,
            )

            for node in ast.walk(tree):
                if isinstance(node, meaningful_node_types):
                    return True
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id in builtin_compute_calls:
                        return True
                    if isinstance(func, ast.Attribute):
                        owner = func.value
                        if isinstance(owner, ast.Name) and owner.id in module_prefixes:
                            return True
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        root_name = str(alias.name or "").split(".", 1)[0]
                        if root_name in module_prefixes:
                            return True
            return False

        def _planned_tool_names(action_name: str, action_args: Dict[str, Any]) -> List[str]:
            if action_name == "Parallel" and isinstance(action_args, dict):
                planned_tools: List[str] = []
                for item in action_args.get("batch", []) or []:
                    if not isinstance(item, dict):
                        continue
                    planned_tool = str(item.get("name") or "").strip()
                    if planned_tool:
                        planned_tools.append(planned_tool)
                return planned_tools
            normalized_name = str(action_name or "").strip()
            return [normalized_name] if normalized_name else []

        def _build_action_recovery_hint(action_name: str, action_args: Dict[str, Any]) -> str:
            action_family = _action_family(action_name)
            planned_tools = _planned_tool_names(action_name, action_args)
            mentions_python = "Python_Executor" in planned_tools or action_family == "compute"
            hints: List[str] = []
            goal_semantic = str(goal or "").lower()
            goal_semantic = re.split(
                r"(?im)^\s*(?:options?|choices?)\s*:\s*",
                goal_semantic,
                maxsplit=1,
            )[0]
            negative_query = bool(re.search(
                r"\b(?:not|never|without|does\s+not|doesn't|did\s+not|didn't|absent)\b",
                goal_semantic,
            ))
            comparison_query = bool(re.search(
                r"\b(?:different|difference|compare|comparison|versus|vs|other)\b",
                goal_semantic,
            ))
            count_query = bool(re.search(
                r"\b(?:how\s+many|number\s+of|count|total)\b",
                goal_semantic,
            ))
            if negative_query:
                hints.append(
                    "Negative-choice contract: enumerate the listed actions/attributes, mark each as present or absent "
                    "in the queried window, and select an option only when its absence is directly supported."
                )
            if comparison_query:
                hints.append(
                    "Comparison contract: compare every listed alternative against the same target and window; "
                    "a generic 'they are different' statement does not identify the selected option."
                )
            if count_query:
                hints.append(
                    "Count contract: cover the full requested interval with non-overlapping visual samples and "
                    "report the exact total before mapping it to an option."
                )
            if depth == 0 and _is_find_needle_goal(goal):
                pending_candidates = _uncompleted_needle_candidates()
                if pending_candidates:
                    next_candidate = pending_candidates[0]
                    hints.append(
                        "Needle candidate follow-up: directly inspect the next unverified candidate window "
                        f"{next_candidate.get('segment_ref')} "
                        f"({next_candidate.get('start_time')}-{next_candidate.get('end_time')} seconds) "
                        "with Inspect_Frame at density <= 1 second/frame before Finish."
                    )
            observer_feedback = memory_state.get("observer_feedback")
            if isinstance(observer_feedback, Mapping):
                recommendation = observer_feedback.get("recommended_action")
                if isinstance(recommendation, Mapping):
                    kind = str(recommendation.get("kind") or "").strip().lower()
                    tool_hint = str(recommendation.get("tool_hint") or "").strip()
                    # A rejected Finish needs fresh evidence, even when the
                    # Observer's preceding recommendation was to finish.
                    if tool_hint.casefold() in {"finish", "verify", "parallel"}:
                        tool_hint = ""
                    if kind in {"reobserve", "cross_verify", "continue", "finish"} or tool_hint:
                        gaps = observer_feedback.get("claim_gaps") or []
                        gap_ids = [
                            str(item.get("claim_id") or "").strip()
                            for item in gaps
                            if isinstance(item, Mapping) and str(item.get("claim_id") or "").strip()
                        ]
                        hints.append(
                            "Observer recovery recommendation: emit a concrete non-Finish "
                            f"evidence action ({tool_hint or 'a registered visual/cross-modal tool'})"
                            + (
                                f" addressing {', '.join(gap_ids)}."
                                if gap_ids
                                else "."
                            )
                        )
                        hints.append(
                            "Use materially changed parameters or a focused window/crop; "
                            "do not answer from the same attached view without a new evidence action."
                        )
            if mentions_python and evidence_first_mode:
                hints.append(
                    "Python_Executor is blocked here because it cannot search raw video or create evidence from literals."
                )
            if evidence_first_mode:
                if action_family == "discovery":
                    hints.append(
                        "Do not repeat registry discovery when the missing information must come from local video evidence."
                    )
                hints.append(
                    "Your next action MUST switch to local evidence retrieval, direct visual probing, or cross-modal verification."
                )
                hints.append(
                    "Materially change the time window / segment based on the latest observation instead of repeating the same template."
                )
            else:
                hints.append(
                    "Your next action MUST materially change tool family or arguments based on the latest observation."
                )
            hints.append(
                "If you still lack a concrete tool, describe the missing capability abstractly so the sub-agent can route it."
            )
            return " ".join(hints).strip()

        def _build_forced_finish_recovery_probe(attempt: int) -> Optional[Tuple[str, Dict[str, Any]]]:
            """Create one deterministic evidence probe after repeated Finish loops.

            A Planner that keeps emitting the same rejected Finish can otherwise
            exhaust the recovery budget without acquiring new evidence.  The
            probe is derived only from the active query and the last executed
            observation window; it never uses labels, ground truth, or
            evaluator output.
            """
            if attempt <= 0 or attempt > 2:
                return None
            previous_payload = (
                last_execution_meta.get("payload")
                if isinstance(last_execution_meta, Mapping)
                else None
            )
            previous_observation = (
                str(last_execution_meta.get("observation") or "")
                if isinstance(last_execution_meta, Mapping)
                else ""
            )
            window = _extract_probe_window(
                previous_payload,
                observation_text=previous_observation,
            )
            if window is None:
                # Dispatch/policy failures overwrite ``last_execution_meta``
                # without carrying the preceding time range.  Recover the
                # most recent successful visual/native window from the
                # trajectory so a forced probe stays near the unresolved
                # event instead of silently falling back to 0--30 seconds.
                def _prior_observation_records(value: Any) -> List[Mapping[str, Any]]:
                    records: List[Mapping[str, Any]] = []
                    if isinstance(value, Mapping):
                        if str(value.get("tool") or "").strip():
                            records.append(value)
                        for nested_key in ("items", "result", "payload"):
                            records.extend(_prior_observation_records(value.get(nested_key)))
                    elif isinstance(value, list):
                        for nested in value:
                            records.extend(_prior_observation_records(nested))
                    return records

                for prior_step in reversed(agent.trajectory.steps):
                    prior_items = (
                        prior_step.get("observation_items") or []
                        if isinstance(prior_step, Mapping)
                        else getattr(prior_step, "observation_items", None) or []
                    )
                    for prior_observation in reversed(_prior_observation_records(prior_items)):
                        prior_tool = str(prior_observation.get("tool") or "").strip()
                        prior_status = str(prior_observation.get("status") or "success").strip().lower()
                        if prior_tool not in {
                            "Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in",
                            "Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify",
                        } or prior_status in {"error", "failed", "no_valid_result"}:
                            continue
                        window = _extract_probe_window(prior_observation)
                        if window is not None:
                            break
                    if window is not None:
                        break
            if window is None:
                start, end = 0.0, 30.0
            else:
                start, end = window
                margin = 1.5 + 0.5 * max(0, attempt - 1)
                start = max(0.0, start - margin)
                end = end + margin
            # Recovery requests obey the same native-tool contract as Planner
            # calls. Split a broad range across bounded attempts instead of
            # spending both attempts on deterministically invalid requests.
            max_window = float(settings.video_clip_qa_max_window_seconds)
            if end - start > max_window:
                start = min(start + (attempt - 1) * max_window, end - max_window)
                end = start + max_window
            question_match = re.search(r"(?is)\bquestion\s*:\s*(.*)", str(goal or ""))
            probe_question = question_match.group(1) if question_match else str(goal or "")
            # Alternatives are inference inputs, not ground truth. Preserve
            # them whenever the probe asks the native tool to compare them.
            probe_question = re.split(r"(?im)^\s*Reply with\b", probe_question, maxsplit=1)[0].strip()
            probe_query = (
                f"Recovery probe {attempt}: answer the question from this exact video window. "
                "Focus on the unresolved relation and report concrete positive/negative evidence "
                f"for the listed alternatives. Question: {probe_question}"
            )
            semantic_lower = re.split(r"(?im)^\s*(?:options?|choices?)\s*:", probe_question, maxsplit=1)[0].lower()
            recovery_context = " ".join(
                [
                    str(finish_recovery_last_reason or ""),
                    " ".join(str(item or "") for item in finish_recovery_last_targets),
                    " ".join(
                        str(item or "")
                        for item in (memory_state.get("missing_evidence") or [])[-4:]
                    ),
                ]
            ).lower()
            comparison_needed = bool(
                re.search(
                    r"\b(?:compare|comparison|different|alternative|versus|vs|"
                    r"plausible\s+alternatives|selected\s+option\s+lacks)\b",
                    recovery_context,
                )
                or re.search(
                    r"\b(?:compare|comparison|different|alternative|versus|vs)\b",
                    semantic_lower,
                )
            )
            rationale_needed = bool(
                re.search(
                    r"\b(?:purpose|rationale|outcome|importance|function|goal|"
                    r"multiple\s+option\s+outcomes)\b",
                    recovery_context,
                )
                or re.search(
                    r"\b(?:purpose|rationale|importance|function|goal|why)\b",
                    semantic_lower,
                )
            )
            if re.search(
                r"\b(?:not|never|without|does\s+not|doesn't|did\s+not|didn't|absent)\b",
                semantic_lower,
            ):
                probe_query += (
                    " Enumerate every listed alternative, state PRESENT or ABSENT for each one, "
                    "and identify the answer only from an explicitly absent alternative; do not infer absence "
                    "from an incomplete action list."
                )
            elif re.search(r"\b(?:different|difference|compare|comparison|versus|vs|other)\b", semantic_lower):
                probe_query += (
                    " Compare each listed alternative separately with the reference subject; report the "
                    "specific differing attribute and the evidence for the selected option."
                )
            if comparison_needed and not re.search(
                r"\b(?:compare|comparison|different|versus|vs)\b", probe_query.lower()
            ):
                probe_query += (
                    " Perform an explicit option comparison: name the selected alternative and state why "
                    "each other listed alternative is present, absent, or incompatible in this same window; "
                    "do not return a generic 'different' statement."
                )
            if rationale_needed:
                probe_query += (
                    " Separate the directly visible action from its purpose or outcome, and state the "
                    "observed result that distinguishes the listed rationale options."
                )
            if any(marker in semantic_lower for marker in ("count", "how many", "number of", "total")):
                return (
                    "Inspect_Frame",
                    {
                        "t_start": round(start, 3),
                        "t_end": round(max(start + 1.0, end), 3),
                        "density": 2.0,
                        "query": probe_query,
                    },
                )
            return (
                "Video_Clip_QA",
                {
                    "start_time": round(start, 3),
                    "end_time": round(max(start + 1.0, end), 3),
                    "query": probe_query,
                },
            )

        def _build_strict_json_retry_message(protocol_violation: str = "") -> str:
            hinted_tool_name = ""
            hinted_tool_params: Dict[str, Any] = {}
            parsed_execution_hint: Dict[str, Any] = {}
            if isinstance(mediation_execution_hint, dict):
                parsed_execution_hint = mediation_execution_hint
            elif isinstance(mediation_execution_hint, str):
                parsed_execution_hint = agent._parse_dict_like(mediation_execution_hint) or {}
            if isinstance(parsed_execution_hint, dict):
                hinted_tool_name = str(parsed_execution_hint.get("tool_name") or "").strip()
                hinted_tool_params = (
                    parsed_execution_hint.get("params", {})
                    if isinstance(parsed_execution_hint.get("params"), dict)
                    else {}
                )
            if mediation_mode:
                tool_name = hinted_tool_name or "Tool_Search"
                tool_params = hinted_tool_params or {"intent_description": "find the next executable evidence tool"}
                required_json = (
                    "{\"Thought\":\"one short sentence\",\"Plan\":\"one short sentence\","
                    f"\"Actions\":[{{\"tool\":{json.dumps(tool_name, ensure_ascii=False)},"
                    f"\"params\":{json.dumps(tool_params, ensure_ascii=False)}}}],"
                    "\"Result\":\"optional short execution note\"}"
                )
            elif evidence_first_mode:
                required_json = (
                    "{\"Thought\":\"one short sentence\",\"Plan\":\"one short sentence\","
                    "\"Evidence\":[{\"label\":\"...\",\"value\":\"\",\"citation\":\"\",\"evidence_status\":\"missing\"}],"
                    "\"Actions\":[{\"tool\":\"ACTUAL_TOOL_NAME_HERE\",\"description\":\"find the local time window or modality probe most likely to resolve the missing evidence\",\"params\":{\"arg\":\"value\"}}]}"
                )
            else:
                required_json = (
                    "{\"Thought\":\"one short sentence\",\"Plan\":\"one short sentence\","
                    "\"Actions\":[{\"tool\":\"ACTUAL_TOOL_NAME_HERE\",\"description\":\"describe the missing capability so the runtime can route it\",\"params\":{\"arg\":\"value\"}}]}"
                )
            lines: List[str] = []
            if protocol_violation:
                lines.append(protocol_violation)
                lines.append("")
            allow_think_wrapper = depth == 0 and not mediation_mode
            if allow_think_wrapper:
                lines.append(
                    "You may include `<think>...</think>` before JSON, but your reply MUST end with exactly one complete JSON object."
                )
                lines.append("The final non-whitespace character of your reply must be `}`.")
                lines.append("Do NOT output markdown fences or any prose after the final JSON object.")
            else:
                lines.append("Return ONLY one strict JSON object.")
                lines.append("The first character of your reply must be `{` and the last character must be `}`.")
                lines.append("Do NOT output `<think>`, markdown fences, `Thought:` labels, or any prose outside JSON.")
            if mediation_mode and mediation_execution_hint:
                lines.append("Prefer the [Schema-Adapted Execution Hint] directly if it is executable.")
                lines.append("If you include `Result`, keep it short. It is optional and never replaces executable `Actions`.")
            lines.append("")
            lines.append(required_json)
            return "\n".join(lines)

        def _validate_python_executor_action(code: str) -> Optional[str]:
            normalized = str(code or "").strip()
            if not normalized:
                return "Error: Python_Executor requires a non-empty code_script."

            try:
                compile(normalized, "<python_executor>", "exec")
            except SyntaxError as exc:
                detail = str(exc.msg or exc).strip()
                if "invalid decimal literal" in detail.lower():
                    detail = (
                        "percent-style text such as `12%` is not valid Python syntax; "
                        "convert it to a decimal number or quoted string first"
                    )
                return (
                    "Error: Python_Executor requires runnable Python syntax only. "
                    f"Rewrite the code_script as valid Python ({detail})."
                )

            if not _active_local_video_scope_present() or _goal_requests_external_web():
                return None

            lowered = normalized.lower()
            visual_path_mentions = bool(
                re.search(r"\.(?:jpg|jpeg|png|bmp|webp|gif|mp4|mov|avi|mkv)\b", lowered)
            )
            visual_read_markers = (
                "image.open(",
                "pil.image.open(",
                "cv2.imread(",
                "cv2.videocapture(",
                "imageio.imread(",
                "imageio.v2.imread(",
                "imageio.get_reader(",
                "videofileclip(",
                "decord.videoreader(",
                "av.open(",
            )
            reads_visual_pixels = any(marker in lowered for marker in visual_read_markers)
            structured_compute_markers = (
                "json.loads(",
                "json.load(",
                "sum(",
                "len(",
                "max(",
                "min(",
                "sorted(",
                "statistics.",
                "pandas",
                "pd.",
                "numpy",
                "np.",
                "counter(",
                "for ",
                "while ",
            )
            has_structured_compute = any(marker in lowered for marker in structured_compute_markers)
            has_meaningful_compute = _python_script_has_meaningful_compute(normalized)
            narrative_print = bool(
                re.search(r"print\(\s*f?[\"']\s*(action|answer|result)\s*:", normalized, flags=re.IGNORECASE)
            )

            if visual_path_mentions and not reads_visual_pixels:
                return (
                    "Error: Python_Executor cannot infer what local video images show from file paths alone. "
                    "This script referenced image/video files but never loaded pixel data. Use the attached visual "
                    "evidence directly, or switch to a materially different visual probe / Crop / Zoom_in instead."
                )
            if narrative_print and not reads_visual_pixels and not has_structured_compute:
                return (
                    "Error: Python_Executor is not a free-form narration tool for local video tasks. "
                    "This script prints a visual conclusion without real computation or pixel analysis. "
                    "Use direct visual evidence or return to the planner without Python."
                )
            if not reads_visual_pixels and not has_structured_compute and not has_meaningful_compute:
                return (
                    "Error: Python_Executor must compute over already extracted evidence, not wrap literals or "
                    "simulate a search step. This script does not perform meaningful computation or pixel analysis. "
                    "Switch to retrieval / visual tools first, then use Python only for post-processing exact evidence."
                )
            return None

        def _action_signature(name: str, args: Dict[str, Any]) -> str:
            normalized_args = args
            normalized_name = str(name or "").strip()
            if isinstance(args, dict) and normalized_name == "Video_Clip_QA":
                start_value = args.get("start_time")
                end_value = args.get("end_time")
                try:
                    start = round(float(start_value), 3) if start_value is not None else None
                except Exception:
                    start = str(start_value).strip() if start_value not in (None, "") else None
                try:
                    end = round(float(end_value), 3) if end_value is not None else None
                except Exception:
                    end = str(end_value).strip() if end_value not in (None, "") else None
                normalized_args = {
                    "video": args.get("video") or args.get("video_path") or args.get("segment_id"),
                    "start_time": start,
                    "end_time": end,
                    # The question is part of the concrete Video_Clip_QA
                    # request.  Two questions over the same window can be
                    # legitimate independent evidence probes; omitting it
                    # incorrectly rejects a refined follow-up query.
                    "query": " ".join(str(args.get("query") or "").lower().split()),
                }
            elif isinstance(args, dict) and normalized_name in {"Inspect_Frame", "OCR_Extract"}:
                raw_timestamp = args.get("timestamp")
                if isinstance(raw_timestamp, list):
                    normalized_timestamp: Any = []
                    for item in raw_timestamp:
                        try:
                            normalized_timestamp.append(round(float(item), 3))
                        except Exception:
                            normalized_timestamp.append(str(item).strip())
                elif raw_timestamp not in (None, ""):
                    try:
                        normalized_timestamp = round(float(raw_timestamp), 3)
                    except Exception:
                        normalized_timestamp = str(raw_timestamp).strip()
                else:
                    normalized_timestamp = None
                normalized_args = {
                    "probe_window": _normalized_probe_window_key(_extract_probe_window(args)),
                    "timestamp": normalized_timestamp,
                    "density": args.get("density"),
                    "language": args.get("language") if normalized_name == "OCR_Extract" else None,
                    "image_region": args.get("image_region") if normalized_name == "OCR_Extract" else None,
                    "video": args.get("video") or args.get("video_path") or args.get("segment_id"),
                    # The textual probe is part of the concrete visual
                    # request.  Keeping it in the runtime signature allows a
                    # targeted follow-up question over the same frame/window
                    # while still rejecting an exact duplicate call.
                    "query": " ".join(str(args.get("query") or "").lower().split()),
                }
            try:
                return f"{name}:{json.dumps(normalized_args, ensure_ascii=False, sort_keys=True)}"
            except Exception:
                return f"{name}:{str(normalized_args)}"

        def _observation_signature(text: str) -> str:
            return _stable_signature(text)

        def _update_progress_state(
            action_name: str,
            observation_text: str,
            observation_payload: Any = None,
        ) -> None:
            nonlocal last_observation_signature, consecutive_low_progress, low_progress_hint_sent
            nonlocal no_progress_guard_signature, no_progress_guard_family, no_progress_abort_reason
            nonlocal adjacent_probe_repeat_count, last_probe_window_state
            nonlocal final_answer, success, finished
            nonlocal low_signal_guard_recovery_count, low_signal_visual_probe_streak
            nonlocal low_signal_recovery_triggered_this_observation

            if (action_name or "").lower() == "finish":
                return

            current_sig = _build_progress_signature(
                action_name=action_name,
                observation_text=observation_text,
                observation_payload=observation_payload,
            )
            if not current_sig:
                return

            current_family = _infer_progress_family(action_name, observation_payload)
            current_window = _extract_probe_window(
                observation_payload,
                observation_text=observation_text,
            )
            current_window_key = _normalized_probe_window_key(current_window)
            previous_family = str(last_probe_window_state.get("family", "")).strip()
            previous_window = last_probe_window_state.get("window")

            if (
                depth == 0
                and current_family in {"visual", "retrieval"}
                and previous_family == current_family
                and _windows_overlap_or_adjacent(previous_window, current_window)
            ):
                adjacent_probe_repeat_count += 1
            else:
                adjacent_probe_repeat_count = 0

            last_probe_window_state = {
                "family": current_family,
                "window": current_window,
                "signature": current_window_key,
            }

            if current_sig == last_observation_signature:
                consecutive_low_progress += 1
            else:
                consecutive_low_progress = 0
                low_progress_hint_sent = False
                no_progress_guard_signature = ""
                no_progress_guard_family = ""
            last_observation_signature = current_sig

            if depth == 0 and consecutive_low_progress >= 2 and not low_progress_hint_sent:
                hint = (
                    "Progress warning: recent observations are repetitive or low-information. "
                    "Next action MUST change tool family, materially change the probe target, "
                    "or express the missing capability as an abstract tool so the sub-agent can route/compose executable tools."
                )
                if evidence_first_mode:
                    hint += (
                        " Continue from nearby cited windows or adjacent segments before broadening search. "
                        "Prefer abstract capability goals plus local evidence routing over repeating the same concrete probes."
                    )
                else:
                    hint += " If uncertain, describe the missing capability instead of repeating the same concrete tool."
                _append_history("user", hint)
                low_progress_hint_sent = True
            if depth == 0 and consecutive_low_progress >= 1:
                _maybe_refresh_routes_for_low_progress(action_name, observation_text)
            if depth == 0 and adjacent_probe_repeat_count >= 1 and current_family in {"visual", "retrieval"}:
                _maybe_refresh_routes_for_low_progress(action_name, observation_text)

            if depth == 0 and evidence_first_mode and current_family in {"visual", "retrieval"}:
                family_threshold = 3 if current_family == "visual" else 2
                if consecutive_low_progress >= family_threshold and not no_progress_guard_signature:
                    no_progress_guard_signature = current_sig
                    no_progress_guard_family = current_family
                    _append_history(
                        "user",
                        "No new evidence was produced after repeated probing of the same tool family and target. "
                        "The next action MUST switch tool family, materially change the time window / segment, "
                        "or delegate an abstract capability for sub-agent routing/decomposition. "
                        "Do not repeat the same probe pattern again.",
                    )
                elif (
                    adjacent_probe_repeat_count >= max(1, family_threshold - 1)
                    and current_window_key
                    and not no_progress_guard_signature
                ):
                    no_progress_guard_signature = f"{current_family}:{current_window_key}"
                    no_progress_guard_family = current_family
                    _append_history(
                        "user",
                        "No-progress guard: you are repeatedly probing the same tool family over the same or adjacent windows. "
                        "The next action MUST switch tool family, jump to a materially different time window, "
                        "or delegate an abstract capability for sub-agent decomposition.",
                    )
                elif (
                    no_progress_guard_signature
                    and (
                        current_sig == no_progress_guard_signature
                        or (
                            current_window_key
                            and no_progress_guard_signature == f"{current_family}:{current_window_key}"
                        )
                    )
                    and current_family == no_progress_guard_family
                    and (
                        consecutive_low_progress >= family_threshold + 1
                        or adjacent_probe_repeat_count >= family_threshold
                    )
                ):
                    if low_signal_guard_recovery_count < max_low_signal_guard_recoveries:
                        low_signal_guard_recovery_count += 1
                        low_signal_recovery_triggered_this_observation = True
                        _activate_finish_recovery(
                            "repeated low-signal evidence probes",
                            targets=[
                                "switch from the current tool family or modality",
                                "inspect a materially different temporal window",
                                "obtain one grounded observation for the unresolved claim",
                            ],
                        )
                        low_signal_visual_probe_streak = 0
                        consecutive_low_progress = 0
                        no_progress_guard_signature = ""
                        no_progress_guard_family = ""
                        low_progress_hint_sent = True
                        agent.trajectory.add_debug_event({
                            "type": "evidence_recovery_escalation",
                            "event_type": "EvidenceRecoveryEscalationEvent",
                            "depth": depth,
                            "iteration": iteration,
                            "reason": "low_signal_visual_probe_budget_exhausted",
                            "recovery_index": low_signal_guard_recovery_count,
                            "max_recoveries": max_low_signal_guard_recoveries,
                            "terminal": False,
                        })
                        logger.warning(
                            "%s[Depth %d] Low-signal guard escalated to alternate evidence recovery %d/%d; continuing",
                            "  " * depth,
                            depth,
                            low_signal_guard_recovery_count,
                            max_low_signal_guard_recoveries,
                        )
                        return
                    no_progress_abort_reason = (
                        "Failed: evidence_recovery_exhausted_without_grounded_observation "
                        "(no_new_evidence_after_probe_budget)"
                    )
                    logger.error(
                        "%s[Depth %d] Aborting due to repeated %s probes without new evidence: %s",
                        '  ' * depth,
                        depth,
                        current_family,
                        current_window_key or current_sig,
                    )
                    final_answer = no_progress_abort_reason
                    success = False
                    finished = True
                    return

        def _record_root_planner_tool_invocation(tool_name: str, observation_status: str, *, executed: bool) -> None:
            if depth != 0 or not executed:
                return
            normalized_tool = str(tool_name or "").strip()
            if not normalized_tool or normalized_tool in {"Parallel", "Finish", "System"}:
                return

            stats = planner_tool_invocation_stats.setdefault(
                normalized_tool,
                {"calls": 0, "failures": 0},
            )
            stats["calls"] += 1
            if str(observation_status or "success").strip().lower() != "success":
                stats["failures"] += 1

        def _build_root_tool_retry_advisory() -> str:
            if depth != 0 or not isinstance(last_execution_meta, dict):
                return ""
            normalized_tool = str(last_execution_meta.get("tool", "")).strip()
            if not normalized_tool:
                return ""

            stats = planner_tool_invocation_stats.get(normalized_tool) or {}
            call_count = int(stats.get("calls", 0) or 0)
            failure_count = int(stats.get("failures", 0) or 0)
            if call_count < 3:
                return ""
            if failure_count == call_count and failure_count >= 3:
                return (
                    "[System Advisory]\n"
                    f"`{normalized_tool}` has already been called {call_count} times in this sample, "
                    f"and all {failure_count} executions failed. Prefer switching to another tool unless you have a strong reason to retry it; "
                    "this tool may be unavailable."
                )
            return (
                "[System Advisory]\n"
                f"`{normalized_tool}` has already been called {call_count} times in this sample. "
                "Try a different tool unless this tool is clearly necessary."
            )

        def _remember_execution(
            *,
            tool_name: str,
            tool_args: Dict[str, Any],
            raw_result: Any,
            observation_text: str,
            observation_status: str,
            source: str = "tool",
            abstract_tool: str = "",
            is_recursive: bool = False,
            executed: bool = True,
        ) -> None:
            nonlocal last_execution_meta
            _, _, compact_payload = format_compact_observation(
                raw_result if raw_result not in (None, "") else {"result": observation_text, "status": observation_status},
                tool_name=tool_name,
            )
            last_execution_meta = {
                "tool": str(tool_name or "").strip(),
                "args": dict(tool_args or {}),
                "status": str(observation_status or "success").strip() or "success",
                "payload": compact_payload,
                "observation": observation_text,
                "source": str(source or "tool").strip() or "tool",
                "abstract_tool": str(abstract_tool or "").strip(),
                "is_recursive": bool(is_recursive),
                "executed": bool(executed),
                "route_refresh_count": route_refresh_count,
                "forced_final_turn_finish": forced_final_turn_retry_used,
            }
            _record_root_planner_tool_invocation(
                str(tool_name or "").strip(),
                str(observation_status or "success").strip() or "success",
                executed=bool(executed),
            )
            if executed:
                runtime_session.note_tool_usage(str(tool_name or "").strip())
            self._emit_progress(
                role=role_label.lower(),
                depth=depth,
                iteration=iteration,
                phase="tool_result",
                status=str(observation_status or "success").strip() or "success",
                current_tool=str(tool_name or "").strip(),
                source=str(source or "tool").strip() or "tool",
                result_preview=_truncate_text(str(observation_text or "").strip(), 180),
            )

        def _record_progress_for_execution(tool_name: str, observation_status: str) -> None:
            nonlocal visual_probe_count, retrieval_probe_count, direct_visual_probe_count, cross_modal_verify_count
            normalized_tool = str(tool_name or "").strip()
            if not normalized_tool:
                return
            family = _action_family(normalized_tool)
            if family == "visual":
                visual_probe_count += 1
                if normalized_tool == "Cross_Modal_Verify":
                    cross_modal_verify_count += 1
                    modality_attempts["cross_modal"] += 1
                    if observation_status == "success":
                        modality_successes["cross_modal"] += 1
                else:
                    direct_visual_probe_count += 1
                    modality_attempts["visual"] += 1
                    if observation_status == "success":
                        modality_successes["visual"] += 1
            elif family == "retrieval":
                retrieval_probe_count += 1
                modality_attempts["text"] += 1
                if observation_status == "success":
                    modality_successes["text"] += 1
            _refresh_modality_progress()

        def _normalize_subagent_result(
            *,
            abstract_tool: str,
            planner_args: Dict[str, Any],
            raw_sub_result: Any,
        ) -> Tuple[str, str, Any, str, Dict[str, Any]]:
            envelope = self._parse_mediation_return(raw_sub_result)
            if envelope:
                executed = bool(envelope.get("executed", False))
                executed_tool = str(envelope.get("executed_tool", "")).strip() or str(abstract_tool or "").strip()
                executed_args = envelope.get("executed_params", {})
                if not isinstance(executed_args, dict):
                    executed_args = dict(planner_args or {})
                raw_payload = envelope.get("observation_payload")
                if raw_payload in (None, "", {}):
                    raw_payload = {
                        "result": str(
                            envelope.get("observation_text")
                            or envelope.get("final_result")
                            or raw_sub_result
                        ).strip(),
                        "status": str(
                            envelope.get("tool_status")
                            or envelope.get("status")
                            or "success"
                        ),
                    }
                if isinstance(raw_payload, dict):
                    payload = dict(raw_payload)
                    payload.setdefault("source", "subagent")
                    payload.setdefault("abstract_tool", str(abstract_tool or "").strip())
                    if executed_tool:
                        payload.setdefault("resolved_tool", executed_tool)
                else:
                    payload = raw_payload
                internal_observation_text, inferred_status, compact_payload = format_compact_observation(
                    payload,
                    tool_name=executed_tool,
                )
                final_status = str(
                    envelope.get("tool_status")
                    or envelope.get("status")
                    or inferred_status
                    or "success"
                ).strip() or "success"
                if not executed:
                    payload = {
                        "status": "error",
                        "result": (
                            "Sub-agent protocol violation: planner-action mediation returned without "
                            "executing a concrete tool."
                        ),
                        "source": str(envelope.get("source", "subagent")).strip() or "subagent",
                        "abstract_tool": str(envelope.get("abstract_tool") or abstract_tool or "").strip(),
                        "raw_sub_result": str(envelope.get("final_result") or raw_sub_result or "").strip(),
                    }
                    internal_observation_text, _, compact_payload = format_compact_observation(
                        payload,
                        tool_name=str(abstract_tool or "").strip(),
                    )
                    final_status = "error"
                planner_payload = payload if payload not in (None, "", {}) else compact_payload
                planner_result_text = extract_planner_result_text(
                    planner_payload,
                    tool_name=executed_tool,
                ).strip()
                if not planner_result_text:
                    planner_result_text = str(internal_observation_text or "").strip()
                meta = {
                    "tool": executed_tool,
                    "args": executed_args,
                    "status": final_status,
                    "payload": planner_payload,
                    "observation": internal_observation_text,
                    "planner_result": planner_result_text,
                    "source": str(envelope.get("source", "subagent")).strip() or "subagent",
                    "abstract_tool": str(envelope.get("abstract_tool") or abstract_tool or "").strip(),
                    "is_recursive": bool(envelope.get("is_recursive", True)),
                    "executed": executed,
                }
                return executed_tool, final_status, planner_payload, planner_result_text, meta

            fallback_payload = {
                "status": "error",
                "result": (
                    "Sub-agent did not return a valid planner-action mediation payload. "
                    f"Raw return: {str(raw_sub_result or '').strip()}"
                ),
                "source": "subagent",
                "abstract_tool": str(abstract_tool or "").strip(),
            }
            internal_observation_text, fallback_status, compact_payload = format_compact_observation(
                fallback_payload,
                tool_name=str(abstract_tool or "").strip(),
            )
            planner_result_text = extract_planner_result_text(
                fallback_payload,
                tool_name=str(abstract_tool or "").strip(),
            ).strip()
            if not planner_result_text:
                planner_result_text = str(internal_observation_text or "").strip()
            meta = {
                "tool": str(abstract_tool or "").strip(),
                "args": dict(planner_args or {}),
                "status": fallback_status,
                "payload": fallback_payload,
                "observation": internal_observation_text,
                "planner_result": planner_result_text,
                "source": "subagent",
                "abstract_tool": str(abstract_tool or "").strip(),
                "is_recursive": True,
                "executed": False,
            }
            return str(abstract_tool or "").strip(), fallback_status, fallback_payload, planner_result_text, meta

        def _build_root_mediation_context(
            *,
            action_name: str,
            action_args: Dict[str, Any],
            description_text: str,
            policy: Dict[str, Any],
        ) -> str:
            tool = registry.get_tool(action_name, log_missing=False)
            target_schema: Optional[Dict[str, Any]] = None
            candidate_tools: List[Dict[str, Any]] = []
            execution_hint: Dict[str, Any] = {}

            if policy.get("rule") == RULE_LOCKED_TARGET and not tool.is_virtual:
                target_schema = tool.get_schema()
                candidate_tools = [target_schema]
                resolution = resolve_abstract_tool_call(
                    abstract_tool=action_name,
                    abstract_args=action_args,
                    description=description_text,
                    candidates=candidate_tools,
                    target_tool_name=action_name,
                    target_tool_schema=target_schema,
                )
                if resolution.get("resolved"):
                    execution_hint = {
                        "tool_name": resolution.get("tool_name"),
                        "params": resolution.get("params", {}),
                    }
            else:
                resolution = _resolve_virtual_subagent_action(
                    abstract_tool=action_name,
                    abstract_args=action_args,
                    description_text=description_text,
                )
                candidate_tools = list(
                    (resolution.get("observation", {}) or {}).get("candidate_tools", [])
                    if isinstance(resolution.get("observation", {}), dict)
                    else []
                )
                if resolution.get("resolved"):
                    execution_hint = {
                        "tool_name": resolution.get("tool_name"),
                        "params": resolution.get("params", {}),
                    }

            return build_root_mediation_context_payload(
                action_name=action_name,
                action_args=action_args,
                description_text=description_text,
                policy=policy,
                discovered_tools_desc=_compact_discovered_tools_for_subagent(),
                target_schema=target_schema,
                candidate_tools=candidate_tools,
                execution_hint=execution_hint,
            )

        def _delegate_root_action_via_subagent(
            *,
            action_name: str,
            action_args: Dict[str, Any],
            description_text: str,
        ) -> Dict[str, Any]:
            nonlocal success, root_effective_tool_executed
            logger.info(
                "%s[Planner Route][Depth %d] action=%s args=%s description=%s",
                '  ' * depth,
                depth,
                action_name,
                action_args,
                _truncate_text(description_text or "", 220),
            )
            routed = _auto_route_root_action(
                action_name=action_name,
                action_args=action_args,
                description_text=description_text,
                defer_execution=False,
            )
            routed_tool = str(routed.get("tool", action_name)).strip() or str(action_name or "").strip()
            routed_args = (
                routed.get("delegate_action_args", action_args)
                if isinstance(routed.get("delegate_action_args", action_args), dict)
                else (action_args if isinstance(action_args, dict) else {})
            )
            routed_payload = routed.get("payload")
            routed_observation = str(routed.get("observation", "") or "").strip()
            routed_status = str(routed.get("status", "") or "").strip()
            executed = bool(routed.get("handled", False))
            if not isinstance(routed_payload, dict):
                routed_payload = {
                    "status": routed_status or ("success" if executed else "error"),
                    "result": routed_observation or (
                        f"Root action '{action_name}' could not be resolved to an executable tool."
                    ),
                    "source": "root_route",
                    "abstract_tool": str(action_name or "").strip(),
                    "resolved_tool": routed_tool,
                }
            else:
                routed_payload = dict(routed_payload)
                routed_payload.setdefault("source", "root_route")
                routed_payload.setdefault("abstract_tool", str(action_name or "").strip())
                routed_payload.setdefault("resolved_tool", routed_tool)

            observation_text, obs_status, compact_payload = format_compact_observation(
                routed_payload,
                tool_name=routed_tool,
            )
            planner_result_text = extract_planner_result_text(
                routed_payload,
                tool_name=routed_tool,
            ).strip()
            if not planner_result_text:
                planner_result_text = str(observation_text or "").strip()
            meta = {
                "tool": routed_tool,
                "args": routed_args,
                "status": obs_status,
                "payload": routed_payload if routed_payload not in (None, "", {}) else compact_payload,
                "observation": observation_text,
                "planner_result": planner_result_text,
                "source": "root_route",
                "abstract_tool": str(action_name or "").strip(),
                "is_recursive": False,
                "executed": executed,
            }
            _remember_execution(
                tool_name=meta.get("tool", routed_tool),
                tool_args=meta.get("args", action_args),
                raw_result=meta.get("payload", routed_payload),
                observation_text=str(meta.get("observation", observation_text)),
                observation_status=obs_status,
                source=str(meta.get("source", "root_route")),
                abstract_tool=str(meta.get("abstract_tool", action_name)),
                is_recursive=False,
                executed=bool(meta.get("executed", False)),
            )
            executed_tool = routed_tool
            if executed_tool and bool(meta.get("executed", False)):
                _record_progress_for_execution(executed_tool, obs_status)
                root_effective_tool_executed = True
            if obs_status == "no_valid_result" and executed_tool:
                _record_no_valid_result(executed_tool, observation_text)
            if obs_status == "success":
                success = True
            if executed_tool == "Tool_Search":
                try:
                    lines = _render_tool_lines(
                        (routed_payload.get("candidate_tools") if isinstance(routed_payload, dict) else None)
                        or (routed_payload.get("items") if isinstance(routed_payload, dict) else None)
                        or []
                    )
                    if lines:
                        _merge_discovered_tool_lines(lines)
                except Exception as exc:
                    logger.warning("Root-routed Tool_Search result parse failed: %s", exc)
            _store_result_aliases(routed_tool or str(action_name or "").strip(), meta.get("payload", routed_payload))
            logger.info(
                "%s[Root Route][Depth %d] resolved_tool=%s status=%s result=%s",
                '  ' * depth,
                depth,
                routed_tool or action_name,
                obs_status,
                _truncate_text(str(observation_text or ""), 220),
            )
            return {
                "tool": routed_tool or action_name,
                "status": obs_status,
                "payload": meta.get("payload", routed_payload),
                "observation": observation_text,
                "executed": bool(meta.get("executed", False)),
                "meta": meta,
                "sub_trajectory": None,
            }

        def _dispatch_root_action_via_observer(
            *,
            planner_action_name: str,
            planner_action_args: Dict[str, Any],
            description_text: str,
            routed_name: str,
            routed_args: Dict[str, Any],
            resolution: Mapping[str, Any],
        ) -> Dict[str, Any]:
            """Make the live Observer the Planner-to-runtime dispatch boundary."""
            if (
                depth != 0
                or "observer" in self.disabled_agent_roles
                or ablation_direct_toolsearch_mode
            ):
                return {
                    "status": "skipped",
                    "tool": routed_name,
                    "params": dict(routed_args or {}),
                    "reason": "observer_dispatch_disabled_or_non_root",
                }

            candidate_names = [str(routed_name or "").strip()]
            raw_candidates = resolution.get("candidate_tools", []) if isinstance(resolution, Mapping) else []
            for candidate in raw_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                candidate_name = str(candidate.get("name") or "").strip()
                if candidate_name and candidate_name not in candidate_names:
                    candidate_names.append(candidate_name)
            route_payload = {
                "resolved_tool": str(routed_name or "").strip(),
                "resolved_params": dict(routed_args or {}),
                "candidate_tools": candidate_names,
                "resolution_status": str(resolution.get("resolved") or "").strip(),
            }
            observer_deadline = time.perf_counter() + 90.0
            if self.wall_clock_deadline_monotonic is not None:
                observer_deadline = min(observer_deadline, self.wall_clock_deadline_monotonic)
            try:
                dispatch_result = self.role_agents.dispatch(
                    goal=goal,
                    planner_action={
                        "tool": str(planner_action_name or "").strip(),
                        "params": dict(planner_action_args or {}),
                        "description": str(description_text or "").strip(),
                    },
                    resolved_route=route_payload,
                    trajectory_tail=[step.to_dict() for step in agent.trajectory.steps[-3:]],
                    deadline_monotonic=observer_deadline,
                )
            except Exception as exc:
                logger.exception("[Observer] dispatch turn failed; preserving deterministic route")
                return {
                    "status": "fallback",
                    "tool": routed_name,
                    "params": dict(routed_args or {}),
                    "reason": f"dispatch_exception:{type(exc).__name__}: {exc}",
                }

            dispatch_event = dict(dispatch_result.get("event") or {})
            dispatch_payload = dispatch_result.get("payload")
            dispatch_payload = dispatch_payload if isinstance(dispatch_payload, Mapping) else {}
            dispatch_decision = str(dispatch_payload.get("dispatch_decision") or "").strip().lower()
            executable_action = dispatch_payload.get("executable_action")
            executable_action = executable_action if isinstance(executable_action, Mapping) else {}
            dispatch_event.update({
                "depth": depth,
                "iteration": iteration,
                "phase": "dispatch",
                "planner_action_name": str(planner_action_name or "").strip(),
                "planner_action_args": dict(planner_action_args or {}),
                "resolved_tool": str(routed_name or "").strip(),
                "dispatch_decision": dispatch_decision,
                "dispatch_selected_tool": str(executable_action.get("tool") or "").strip(),
                "dispatch_applied": False,
            })
            runtime_session.record_timing(
                "observer.dispatch_llm_generate_seconds",
                float(dispatch_event.get("elapsed_seconds") or 0.0),
            )

            if not dispatch_result.get("schema_valid"):
                reason = str(dispatch_result.get("error") or "invalid observer dispatch contract").strip()
                logger.warning(
                    "[Observer] dispatch schema failed for %s; deterministic route fallback: %s",
                    routed_name,
                    reason,
                )
                agent.trajectory.add_debug_event(dispatch_event)
                return {
                    "status": "fallback",
                    "tool": routed_name,
                    "params": dict(routed_args or {}),
                    "reason": reason,
                }
            if dispatch_decision != "execute":
                reason = str(dispatch_payload.get("reason") or "Observer rejected executable dispatch").strip()
                # The Observer is a routing validator, not the duplicate-call
                # authority.  Its VLM can over-generalize a changed query on
                # the same short window as a duplicate (the historical
                # acrobatics recovery loop).  Let the deterministic runtime
                # apply the exact-call signature instead of discarding a
                # materially reformulated evidence probe.  Invalid routes
                # still fail through the normal registry/argument checks.
                duplicate_like = bool(re.search(
                    r"\b(?:duplicate|previous|same\s+(?:segment|window|parameters|call)|"
                    r"already\s+(?:executed|attempted)|rejected\s+as)\b",
                    reason,
                    re.IGNORECASE,
                ))
                if duplicate_like:
                    dispatch_event["dispatch_duplicate_bypass"] = True
                    dispatch_event["dispatch_bypass_reason"] = reason
                    dispatch_event["dispatch_applied"] = True
                    logger.info(
                        "[Observer] bypassing VLM duplicate-like rejection for deterministic exact-call guard tool=%s",
                        routed_name,
                    )
                    agent.trajectory.add_debug_event(dispatch_event)
                    return {
                        "status": "accepted",
                        "tool": routed_name,
                        "params": dict(routed_args or {}),
                        "reason": f"observer_duplicate_like_rejection_bypassed:{reason}",
                    }
                dispatch_event["dispatch_rejected"] = True
                logger.warning("[Observer] dispatch rejected tool=%s reason=%s", routed_name, reason)
                agent.trajectory.add_debug_event(dispatch_event)
                return {
                    "status": "rejected",
                    "tool": routed_name,
                    "params": dict(routed_args or {}),
                    "reason": reason,
                }

            selected_name = str(executable_action.get("tool") or "").strip()
            selected_args = executable_action.get("params", {})
            selected_args = {
                str(key): value
                for key, value in selected_args.items()
                if not str(key).startswith("__")
            } if isinstance(selected_args, Mapping) else {}
            selected_tool = registry.get_tool(selected_name, log_missing=False)
            if (
                not selected_name
                or selected_name not in candidate_names
                or selected_tool.is_virtual
            ):
                reason = (
                    f"Observer selected unsupported route '{selected_name}'. "
                    f"Allowed executable candidates: {candidate_names}."
                )
                dispatch_event["dispatch_validation_error"] = reason
                logger.warning("[Observer] %s", reason)
                agent.trajectory.add_debug_event(dispatch_event)
                return {
                    "status": "rejected",
                    "tool": routed_name,
                    "params": dict(routed_args or {}),
                    "reason": reason,
                }
            dispatch_event["dispatch_applied"] = True
            agent.trajectory.add_debug_event(dispatch_event)
            return {
                "status": "accepted",
                "tool": selected_name,
                "params": selected_args,
                "reason": str(dispatch_payload.get("reason") or "").strip(),
            }

        def _auto_route_root_action(
            *,
            action_name: str,
            action_args: Dict[str, Any],
            description_text: str,
            defer_execution: bool = False,
        ) -> Dict[str, Any]:
            nonlocal success, root_effective_tool_executed

            tool = registry.get_tool(action_name, log_missing=False)
            policy = _classify_root_action(action_name, action_args)
            resolution: Dict[str, Any] = {}

            if policy.get("tool_exists") and not tool.is_virtual:
                target_schema = tool.get_schema()
                resolution = resolve_abstract_tool_call(
                    abstract_tool=action_name,
                    abstract_args=action_args,
                    description=description_text,
                    candidates=[target_schema],
                    target_tool_name=action_name,
                    target_tool_schema=target_schema,
                )
            else:
                resolution = _resolve_virtual_subagent_action(
                    abstract_tool=action_name,
                    abstract_args=action_args,
                    description_text=description_text,
                )

            routed_name = str(resolution.get("tool_name", "") or "").strip()
            routed_args = (
                resolution.get("params", {})
                if isinstance(resolution.get("params"), dict)
                else {}
            )
            if resolution.get("resolved") and routed_name:
                routed_policy = _classify_root_action(routed_name, routed_args)
                if routed_policy.get("direct_executable"):
                    observer_dispatch = _dispatch_root_action_via_observer(
                        planner_action_name=action_name,
                        planner_action_args=action_args if isinstance(action_args, dict) else {},
                        description_text=description_text,
                        routed_name=routed_name,
                        routed_args=routed_args,
                        resolution=resolution,
                    )
                    if observer_dispatch.get("status") == "rejected":
                        rejection = str(observer_dispatch.get("reason") or "Observer rejected dispatch").strip()
                        return {
                            "handled": False,
                            "tool": routed_name,
                            "status": "error",
                            "payload": {
                                "status": "error",
                                "result": f"Observer dispatch rejected {routed_name}: {rejection}",
                                "source": "observer_dispatch",
                                "resolved_tool": routed_name,
                                "observer_dispatch": observer_dispatch,
                            },
                            "observation": rejection,
                            "delegate_action_name": routed_name,
                            "delegate_action_args": routed_args,
                            "observer_dispatch": observer_dispatch,
                        }
                    if observer_dispatch.get("status") == "accepted":
                        routed_name = str(observer_dispatch.get("tool") or routed_name).strip() or routed_name
                        routed_args = (
                            observer_dispatch.get("params")
                            if isinstance(observer_dispatch.get("params"), dict)
                            else routed_args
                        )
                        routed_policy = _classify_root_action(routed_name, routed_args)
                    if defer_execution:
                        return {
                            "handled": False,
                            "requires_execution": True,
                            "tool": routed_name,
                            "status": "",
                            "payload": None,
                            "observation": "",
                            "delegate_action_name": routed_name,
                            "delegate_action_args": routed_args,
                            "observer_dispatch": observer_dispatch,
                        }
                    logger.info(
                        "%s[Root Auto-Route][Depth %d] planner_action=%s routed_tool=%s",
                        '  ' * depth,
                        depth,
                        action_name,
                        routed_name,
                    )
                    try:
                        result, observation_text, obs_status, _ = _execute_concrete_tool_call(routed_name, routed_args)
                    except Exception as exc:
                        # Parameter-shape or schema-related failures must be delegated to sub-agent rewrite.
                        error_text = str(exc).strip() or "invalid parameters"
                        if not _looks_like_parameter_error(error_text):
                            raise
                        logger.info(
                            "%s[Root Auto-Route][Depth %d] direct execution rejected for %s: %s; delegating to subagent",
                            '  ' * depth,
                            depth,
                            routed_name,
                            error_text,
                        )
                        return {
                            "handled": False,
                            "tool": routed_name,
                            "status": "error",
                            "payload": {
                                "status": "error",
                                "result": error_text,
                                "source": "root_auto_route",
                                "tool": routed_name,
                            },
                            "observation": error_text,
                            "delegate_action_name": routed_name,
                            "delegate_action_args": routed_args,
                            "observer_dispatch": observer_dispatch,
                        }
                    if obs_status == "error" and _looks_like_parameter_error(result, observation_text):
                        error_text = _extract_error_text_from_payload(result) or str(observation_text or "").strip() or "invalid parameters"
                        logger.info(
                            "%s[Root Auto-Route][Depth %d] direct execution returned parameter-like error for %s: %s; delegating to subagent",
                            '  ' * depth,
                            depth,
                            routed_name,
                            _truncate_text(error_text, 220),
                        )
                        return {
                            "handled": False,
                            "tool": routed_name,
                            "status": "error",
                            "payload": {
                                "status": "error",
                                "result": error_text,
                                "source": "root_auto_route",
                                "tool": routed_name,
                            },
                            "observation": error_text,
                            "delegate_action_name": routed_name,
                            "delegate_action_args": routed_args,
                            "observer_dispatch": observer_dispatch,
                        }
                    if obs_status == "success":
                        success = True
                        root_effective_tool_executed = True
                    if routed_name == "Tool_Search":
                        try:
                            lines = _render_tool_lines(result)
                            if lines:
                                _merge_discovered_tool_lines(lines)
                        except Exception as exc:
                            logger.warning("Root auto-routed Tool_Search result parse failed: %s", exc)
                    return {
                        "handled": True,
                        "tool": routed_name,
                        "status": obs_status,
                        "payload": result,
                        "observation": observation_text,
                        "delegate_action_name": routed_name,
                        "delegate_action_args": routed_args,
                        "observer_dispatch": observer_dispatch,
                    }
                if routed_policy.get("tool_exists"):
                    return {
                        "handled": False,
                        "tool": routed_name or action_name,
                        "status": "",
                        "payload": resolution.get("observation"),
                        "observation": str((resolution.get("observation", {}) or {}).get("result", "")),
                        "delegate_action_name": routed_name,
                        "delegate_action_args": routed_args,
                        "observer_dispatch": {"status": "not_required"},
                    }

            return {
                "handled": False,
                "tool": action_name,
                "status": "",
                "payload": resolution.get("observation"),
                "observation": str((resolution.get("observation", {}) or {}).get("result", "")),
                "delegate_action_name": action_name,
                "delegate_action_args": action_args if isinstance(action_args, dict) else {},
                "observer_dispatch": {"status": "not_required"},
            }

        def _record_tool_error(tool_name: str) -> int:
            current = tool_error_streaks.get(tool_name, 0) + 1
            tool_error_streaks[tool_name] = current
            return current

        def _clear_tool_error(tool_name: str) -> None:
            if tool_name in tool_error_streaks:
                del tool_error_streaks[tool_name]

        def _record_identical_action_failure(signature: str) -> int:
            current = identical_action_failures.get(signature, 0) + 1
            identical_action_failures[signature] = current
            return current

        def _clear_identical_action_failure(signature: str) -> None:
            if signature in identical_action_failures:
                del identical_action_failures[signature]

        def _concrete_tool_execution_signature(tool_name: str, resolved_args: Dict[str, Any]) -> str:
            return _action_signature(
                str(tool_name or "").strip(),
                resolved_args if isinstance(resolved_args, dict) else {},
            )

        def _duplicate_tool_execution_message(tool_name: str, resolved_args: Dict[str, Any]) -> str:
            normalized_tool = str(tool_name or "").strip()
            call_preview = _format_action_for_history(normalized_tool, resolved_args)
            execution_signature = _concrete_tool_execution_signature(normalized_tool, resolved_args)
            previous_preview = executed_tool_signatures.get(execution_signature, call_preview)
            blocked_message = (
                "Error: this exact tool call was already executed earlier in this run: "
                f"{previous_preview}. You cannot execute the same tool with the same parameters twice. "
                "Choose a different tool or materially different arguments."
            )
            if evidence_first_mode:
                blocked_message += (
                    " Continue from nearby evidence windows or switch to another relevant retrieval, "
                    "visual, or verification tool."
                )
            return blocked_message.strip()

        def _append_memory_state_message(key: str, message: str, *, limit: int = 8) -> None:
            normalized_message = str(message or "").strip()
            if not normalized_message:
                return
            current_values = memory_state.get(key)
            values = [
                str(item).strip()
                for item in (current_values or [])
                if str(item).strip()
            ] if isinstance(current_values, list) else []
            if normalized_message in values:
                return
            values.append(normalized_message)
            if limit > 0 and len(values) > limit:
                values = values[-limit:]
            memory_state[key] = values

        def _record_no_valid_result(tool_name: str, reason: str) -> None:
            normalized_tool = str(tool_name or "").strip()
            if not normalized_tool or normalized_tool in {"Tool_Search", "Finish", "Parallel"}:
                return
            normalized_reason = _truncate_text(str(reason or "").strip(), 220)
            failure_summary = f"{normalized_tool}: status=no_valid_result"
            if normalized_reason:
                failure_summary = f"{failure_summary}; {normalized_reason}"
            _append_memory_state_message("failed_actions", failure_summary)
            _append_memory_state_message(
                "do_not_repeat",
                f"{normalized_tool}: do not repeat the exact failed call; correct invalid parameters or switch tools.",
            )
            if evidence_first_mode:
                _append_memory_state_message(
                    "next_allowed_tools",
                    "Correct parameter/window validation failures; otherwise switch to a different evidence tool.",
                )
            logger.info(
                "%s[%s][Depth %d] Recorded no_valid_result: %s reason=%s",
                '  ' * depth,
                role_label,
                depth,
                normalized_tool,
                normalized_reason or "[empty]",
            )

        _unresolved_pointer = object()

        def _normalize_variable_key(value: Any) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")

        def _dedupe_preserve_order(values: List[str]) -> List[str]:
            ordered: List[str] = []
            seen: set[str] = set()
            for value in values:
                if not value or value in seen:
                    continue
                seen.add(value)
                ordered.append(value)
            return ordered

        def _resolve_context_variable(name: str) -> Any:
            key = str(name or "").strip()
            if not key:
                return _unresolved_pointer

            direct_value = self.context.get_variable(key)
            if direct_value is not None:
                return direct_value

            normalized_target = _normalize_variable_key(key)
            available = self.context.get_all_variables()
            for candidate_key, candidate_value in available.items():
                raw_candidate = str(candidate_key or "").strip()
                stripped_candidate = re.sub(r"^depth\d+_", "", raw_candidate)
                normalized_candidate = _normalize_variable_key(stripped_candidate)
                if key in {raw_candidate, stripped_candidate}:
                    return candidate_value
                if normalized_target and normalized_candidate == normalized_target:
                    return candidate_value
            return _unresolved_pointer

        def _resolve_pointer_expression(value: str) -> Any:
            text = str(value or "").strip()
            if not text.startswith("$") or len(text) <= 1:
                return _unresolved_pointer

            expression = text[1:]
            parts = [part for part in expression.split(".") if part]
            if not parts:
                return _unresolved_pointer

            resolved = _resolve_context_variable(parts[0])
            if resolved is _unresolved_pointer:
                return _unresolved_pointer

            for part in parts[1:]:
                if isinstance(resolved, dict):
                    if part in resolved:
                        resolved = resolved[part]
                        continue
                    normalized_part = _normalize_variable_key(part)
                    matched_key = next(
                        (
                            key
                            for key in resolved.keys()
                            if _normalize_variable_key(key) == normalized_part
                        ),
                        None,
                    )
                    if matched_key is None:
                        return _unresolved_pointer
                    resolved = resolved[matched_key]
                    continue

                if isinstance(resolved, list) and part.isdigit():
                    index = int(part)
                    if 0 <= index < len(resolved):
                        resolved = resolved[index]
                        continue
                return _unresolved_pointer

            return resolved

        def _resolve_tool_args(value: Any) -> Any:
            if isinstance(value, str):
                resolved = _resolve_pointer_expression(value)
                return value if resolved is _unresolved_pointer else resolved
            if isinstance(value, list):
                return [_resolve_tool_args(item) for item in value]
            if isinstance(value, dict):
                return {key: _resolve_tool_args(item) for key, item in value.items()}
            return value

        def _store_result_aliases(tool_name: str, result: Any) -> None:
            label = str(tool_name or "").strip()
            normalized_label = _normalize_variable_key(label)
            alias_pairs: List[Tuple[str, Any]] = [("last_result", result)]

            if label:
                alias_pairs.extend(
                    [
                        (f"{label}_result", result),
                        (f"last_{label}_result", result),
                    ]
                )
            if normalized_label:
                alias_pairs.extend(
                    [
                        (normalized_label, result),
                        (f"{normalized_label}_result", result),
                        (f"last_{normalized_label}_result", result),
                    ]
                )

            if isinstance(result, dict):
                image_payload = result.get("image_evidence")
                if image_payload:
                    if label:
                        alias_pairs.extend(
                            [
                                (f"image_evidence_from_{label}", image_payload),
                                (f"frame_evidence_from_{label}", image_payload),
                            ]
                        )
                    if normalized_label:
                        alias_pairs.extend(
                            [
                                (f"{normalized_label}_image_evidence", image_payload),
                                (f"{normalized_label}_frame_evidence", image_payload),
                            ]
                        )

                for field_name in ("matched_segments", "items", "occurrences"):
                    field_value = result.get(field_name)
                    if not field_value:
                        continue
                    if label:
                        alias_pairs.append((f"{field_name}_from_{label}", field_value))
                    if normalized_label:
                        alias_pairs.append((f"{normalized_label}_{field_name}", field_value))

            for alias, payload in alias_pairs:
                for candidate in _dedupe_preserve_order(
                    [
                        alias,
                        re.sub(r"^depth\d+_", "", alias),
                        _normalize_variable_key(alias),
                    ]
                ):
                    self.context.set_variable(candidate, payload)

        def _uncompleted_needle_candidates() -> List[Dict[str, Any]]:
            return [
                window
                for window in needle_candidate_windows
                if (
                    _candidate_probe_window_key(window) not in needle_candidate_completed
                    and _candidate_probe_window_key(window) not in needle_candidate_reserved
                )
            ]

        def _select_needle_candidate(
            requested_range: Optional[Tuple[float, float]],
        ) -> Optional[Dict[str, Any]]:
            available = _uncompleted_needle_candidates()
            if not available:
                return None
            if requested_range is not None:
                matching = [
                    window
                    for window in available
                    if (
                        abs(float(window["start_time"]) - float(requested_range[0])) <= 0.75
                        and abs(float(window["end_time"]) - float(requested_range[1])) <= 0.75
                    )
                ]
                if matching:
                    return matching[0]
            return available[0]

        def _prepare_concrete_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal low_signal_recovery_call_count
            requested_tool_name = str(name or "").strip()
            evolution_policy_route: Optional[Dict[str, Any]] = None
            needle_candidate_route: Optional[Dict[str, Any]] = None
            if (
                depth == 0
                and _is_find_needle_goal(goal)
                and requested_tool_name in {"Inspect_Frame", "Video_Clip_QA"}
                and needle_candidate_windows
            ):
                requested_range = _extract_probe_window(args if isinstance(args, dict) else {})
                candidate = _select_needle_candidate(requested_range)
                if candidate is not None:
                    candidate_key = _candidate_probe_window_key(candidate)
                    needle_candidate_reserved.add(candidate_key)
                    needle_candidate_route = {
                        "policy": "needle_candidate_followup",
                        "requested_tool": requested_tool_name,
                        "replacement_tool": "Inspect_Frame",
                        "candidate_window": dict(candidate),
                        "candidate_key": candidate_key,
                        "requested_range": list(requested_range) if requested_range is not None else None,
                        "uses_annotation_time": False,
                        "uses_answer_label": False,
                    }
                    original_args = args if isinstance(args, dict) else {}
                    name = "Inspect_Frame"
                    args = {
                        "t_start": float(candidate["start_time"]),
                        "t_end": float(candidate["end_time"]),
                        "density": 1.0,
                        "query": str(
                            original_args.get("query")
                            or "visually verify the selected Needle candidate window"
                        ),
                    }
                    logger.info(
                        "[Needle Candidate Follow-up] routed %s to Inspect_Frame window=%s density=1.0",
                        requested_tool_name,
                        candidate.get("time_range"),
                    )
            if (
                depth == 0
                and bool(low_signal_recovery_policy.get("enabled"))
                and _action_family(requested_tool_name) == "retrieval"
            ):
                try:
                    max_text_failures = int(low_signal_recovery_policy.get("max_text_failures", 0))
                    max_recovery_calls = int(low_signal_recovery_policy.get("max_recovery_calls", 0))
                    coverage_density_seconds = float(
                        low_signal_recovery_policy.get("coverage_density_seconds", 0.0)
                    )
                except (TypeError, ValueError):
                    max_text_failures = 0
                    max_recovery_calls = 0
                    coverage_density_seconds = 0.0
                if (
                    max_text_failures > 0
                    and max_recovery_calls > 0
                    and coverage_density_seconds > 0.0
                    and retrieval_failure_count >= max_text_failures
                    and low_signal_recovery_call_count < max_recovery_calls
                ):
                    requested_args = args if isinstance(args, dict) else {}
                    recovery_query = str(
                        requested_args.get("query")
                        or requested_args.get("claim")
                        or "visual evidence after low-signal retrieval"
                    ).strip()
                    low_signal_recovery_call_count += 1
                    evolution_policy_route = {
                        "policy": "low_signal_temporal_grounding",
                        "requested_tool": requested_tool_name,
                        "replacement_tool": "Inspect_Frame",
                        "trigger": "text_retrieval_failures",
                        "text_retrieval_failures": retrieval_failure_count,
                        "coverage_density_seconds": coverage_density_seconds,
                        "recovery_call_index": low_signal_recovery_call_count,
                        "uses_annotation_time": False,
                        "uses_answer_label": False,
                    }
                    logger.info(
                        "[Evolution Runtime] low_signal_temporal_grounding routed %s -> Inspect_Frame "
                        "after %d text-retrieval failures",
                        requested_tool_name,
                        retrieval_failure_count,
                    )
                    name = "Inspect_Frame"
                    args = {
                        "t_start": 0.0,
                        "t_end": "entire_video_time",
                        "density": coverage_density_seconds,
                        "query": recovery_query,
                    }
            if (
                os.environ.get("TOOLSTAR_DISABLE_GPU_VIDEO_CLIP_QA", "").strip().lower()
                in {"1", "true", "yes"}
                and name == "Video_Clip_QA"
            ):
                video_args = args if isinstance(args, dict) else {}
                try:
                    clip_start = float(video_args.get("start_time", 0.0))
                    clip_end = float(video_args.get("end_time", clip_start + 30.0))
                except (TypeError, ValueError):
                    clip_start, clip_end = 0.0, 30.0
                if clip_end <= clip_start:
                    clip_end = clip_start + 30.0
                logger.info("[Planner Runtime Fallback] Replaced GPU video clip QA with Inspect_Frame.")
                name = "Inspect_Frame"
                args = {
                    "t_start": clip_start,
                    "t_end": clip_end,
                    "density": 3.0,
                    "query": f"Visual fallback for video QA: {str(video_args.get('query', 'video evidence') or 'video evidence')}",
                }
            if (
                os.environ.get("TOOLSTAR_DISABLE_GPU_SEMANTIC_RETRIEVAL", "").strip().lower()
                in {"1", "true", "yes"}
                and name == "Semantic_Retrieve"
            ):
                semantic_args = args if isinstance(args, dict) else {}
                semantic_query = str(semantic_args.get("query", "video evidence") or "video evidence")
                t_start, t_end = 0.0, None
                requested_range = semantic_args.get("time_range")
                if isinstance(requested_range, (list, tuple)) and len(requested_range) >= 2:
                    try:
                        t_start, t_end = float(requested_range[0]), float(requested_range[1])
                    except (TypeError, ValueError):
                        pass
                elif isinstance(requested_range, str):
                    # MLVU requests use forms such as "200-408".  Time is
                    # non-negative here, so treat the hyphen as a separator,
                    # not as the sign of the second endpoint.
                    range_values = re.findall(r"\\d+(?:\\.\\d+)?", requested_range)
                    if len(range_values) >= 2:
                        t_start, t_end = float(range_values[0]), float(range_values[1])
                has_requested_range = t_end is not None and t_end > t_start
                if not has_requested_range:
                    # Concrete tool normalization can remove an unsupported
                    # time_range from Semantic_Retrieve.  In that case, use a
                    # coarse full-video overview rather than silently probing
                    # only the first 90 seconds.
                    t_start, t_end = 0.0, 600.0
                # The concrete-call de-duplication key for Inspect_Frame does
                # not retain the free-text query.  Give semantically distinct
                # fallback requests stable sampling densities so they remain
                # separate evidence probes rather than being rejected as
                # repeats of the first fallback call.
                density_base = 3.0 if has_requested_range else 20.0
                query_fingerprint = int(hashlib.sha1(semantic_query.encode("utf-8")).hexdigest()[:8], 16)
                # Keep the sampling cadence close to the intended rate while
                # making each query's concrete-call signature effectively
                # unique; a four-bucket hash caused unrelated requests to be
                # rejected as duplicate evidence probes.
                fallback_density = density_base + (query_fingerprint % 1000000) / 1000000.0
                logger.info("[Planner Runtime Fallback] Replaced GPU semantic retrieval with Inspect_Frame.")
                name = "Inspect_Frame"
                args = {
                    "t_start": t_start,
                    "t_end": t_end,
                    "density": fallback_density,
                    "query": f"Visual fallback for semantic request: {semantic_query}",
                }
            resolved_args = _resolve_tool_args(args if isinstance(args, dict) else {})
            logger.info(
                "%s[%s][Depth %d] Executing tool: %s",
                '  ' * depth,
                role_label,
                depth,
                _format_action_for_history(name, resolved_args),
            )
            execution_signature = _concrete_tool_execution_signature(str(name or "").strip(), resolved_args)
            duplicate_message = ""
            if execution_signature in executed_tool_signatures:
                duplicate_message = _duplicate_tool_execution_message(str(name or "").strip(), resolved_args)
            if duplicate_message:
                result = {
                    "status": "error",
                    "result": duplicate_message,
                    "blocked_tool": str(name or "").strip(),
                    "reason": "duplicate_tool_execution_rejected",
                    "source": "system",
                }
                logger.info(
                    "%s[Observer][Depth %d] Tool %s controlled_rejection reason=duplicate_tool_execution_rejected",
                    '  ' * depth,
                    depth,
                    str(name or "").strip(),
                )
                _append_memory_state_message(
                    "do_not_repeat",
                    f"{_format_action_for_history(str(name or '').strip(), resolved_args)}: already executed once in this run; do not call it again.",
                )
                return {
                    "tool_name": str(name or "").strip(),
                    "requested_tool_name": requested_tool_name,
                    "resolved_args": resolved_args,
                    "mode": "duplicate_rejected",
                    "result": result,
                    "source": "system",
                    "executed": False,
                    "execution_signature": execution_signature,
                    "evolution_policy_route": evolution_policy_route,
                    "needle_candidate_route": needle_candidate_route,
                }
            tool = registry.get_tool(name, log_missing=False)
            if tool.is_virtual:
                raise ValueError(f"Tool '{name}' is not executable.")
            from toolkit.impl.retrieval import get_sparse_scope_hidden_tool_payload

            executed_tool_signatures[execution_signature] = _format_action_for_history(
                str(name or "").strip(),
                resolved_args,
            )

            video_hint = None
            if isinstance(resolved_args, dict):
                for key in ("video", "video_path", "segment_id"):
                    value = resolved_args.get(key)
                    if value not in (None, ""):
                        video_hint = str(value)
                        break
            hidden_payload = get_sparse_scope_hidden_tool_payload(name, video_hint=video_hint)
            if hidden_payload is not None:
                return {
                    "tool_name": str(name or "").strip(),
                    "requested_tool_name": requested_tool_name,
                    "resolved_args": resolved_args,
                    "mode": "synthetic_result",
                    "result": hidden_payload,
                    "source": "tool",
                    "executed": True,
                    "execution_signature": execution_signature,
                    "evolution_policy_route": evolution_policy_route,
                    "needle_candidate_route": needle_candidate_route,
                }
            return {
                "tool_name": str(name or "").strip(),
                "requested_tool_name": requested_tool_name,
                "resolved_args": resolved_args,
                "mode": "execute",
                "tool": tool,
                "source": "tool",
                "executed": True,
                "execution_signature": execution_signature,
                "evolution_policy_route": evolution_policy_route,
                "needle_candidate_route": needle_candidate_route,
            }

        def _run_prepared_concrete_tool_call(prepared_call: Dict[str, Any]) -> Any:
            mode = str(prepared_call.get("mode", "")).strip()
            if mode == "execute":
                tool = prepared_call.get("tool")
                if tool is None:
                    raise ValueError("Prepared tool execution is missing tool instance.")
                tool_name = str(prepared_call.get("tool_name", "")).strip()
                logger.info(
                    "%s[Observer][Depth %d] Executing concrete tool=%s mode=execute",
                    '  ' * depth,
                    depth,
                    tool_name,
                )
                serialize_video_clip_qa = (
                    os.environ.get("TOOLSTAR_SERIALIZE_GPU_VIDEO_CLIP_QA", "")
                    .strip()
                    .lower()
                    in {"1", "true", "yes", "on"}
                )
                if serialize_video_clip_qa and tool_name == "Video_Clip_QA":
                    wait_started = time.perf_counter()
                    with _GPU_VIDEO_CLIP_QA_EXECUTION_LOCK:
                        waited_seconds = time.perf_counter() - wait_started
                        if waited_seconds >= 0.25:
                            logger.info(
                                "[%s] Serialized GPU Video_Clip_QA execution after waiting %.2fs.",
                                threading.current_thread().name,
                                waited_seconds,
                            )
                        return tool.execute(**prepared_call.get("resolved_args", {}))
                return tool.execute(**prepared_call.get("resolved_args", {}))
            return prepared_call.get("result")

        def _finalize_prepared_concrete_tool_call(
            prepared_call: Dict[str, Any],
            result: Any,
            *,
            abstract_tool: Optional[str] = None,
        ) -> Tuple[Any, str, str]:
            nonlocal retrieval_failure_count
            name = str(prepared_call.get("tool_name", "")).strip()
            requested_tool_name = str(
                prepared_call.get("requested_tool_name") or name
            ).strip()
            evolution_policy_route = prepared_call.get("evolution_policy_route")
            resolved_args = prepared_call.get("resolved_args", {})
            mode = str(prepared_call.get("mode", "")).strip()
            executed = bool(prepared_call.get("executed", True))
            source = str(prepared_call.get("source", "tool")).strip() or "tool"
            abstract_label = str(abstract_tool or name).strip() or name

            _clear_tool_error(name)

            # Concrete implementations may return domain-specific dictionaries,
            # lists, or scalars.  Normalize at the single planner boundary so
            # synthetic rejections and real tool executions share one truthful
            # contract without changing internal retrieval helpers.
            result = normalize_tool_result(result, tool_name=name)

            if depth == 0 and _is_find_needle_goal(goal) and name == "Semantic_Retrieve":
                needle_candidate_windows.clear()
                needle_candidate_completed.clear()
                needle_candidate_reserved.clear()
                if isinstance(result, dict):
                    needle_candidate_windows.extend(
                        _normalize_candidate_probe_windows(result.get("candidate_probe_windows"))
                    )

            if isinstance(evolution_policy_route, dict):
                if isinstance(result, dict):
                    result = dict(result)
                    result["evolution_policy_route"] = dict(evolution_policy_route)
                evolution_policy_events.append(dict(evolution_policy_route))
                agent.trajectory.add_debug_event({
                    "type": "evolution_runtime_route",
                    "event_type": "EvolutionRuntimeRouteEvent",
                    "policy": str(evolution_policy_route.get("policy") or ""),
                    "requested_tool": requested_tool_name,
                    "replacement_tool": name,
                    "trigger": str(evolution_policy_route.get("trigger") or ""),
                    "text_retrieval_failures": int(
                        evolution_policy_route.get("text_retrieval_failures", 0) or 0
                    ),
                    "uses_annotation_time": False,
                    "uses_answer_label": False,
                    "executed": bool(prepared_call.get("executed", True)),
                })

            _store_result_aliases(name, result)
            observation_text, observation_status, _ = format_compact_observation(result, tool_name=name)
            needle_route = prepared_call.get("needle_candidate_route")
            if isinstance(needle_route, Mapping):
                candidate_key = str(needle_route.get("candidate_key") or "").strip()
                if candidate_key:
                    needle_candidate_reserved.discard(candidate_key)
                    if observation_status in {"success", "no_valid_result"}:
                        needle_candidate_completed.add(candidate_key)
                agent.trajectory.add_debug_event({
                    "type": "needle_candidate_followup",
                    "event_type": "NeedleCandidateFollowupEvent",
                    "requested_tool": str(needle_route.get("requested_tool") or ""),
                    "executed_tool": name,
                    "candidate_window": needle_route.get("candidate_window"),
                    "candidate_key": candidate_key,
                    "requested_range": needle_route.get("requested_range"),
                    "status": observation_status,
                    "uses_annotation_time": False,
                    "uses_answer_label": False,
                })
            if observation_status == "no_valid_result":
                _record_no_valid_result(name, observation_text)
            if (
                _action_family(requested_tool_name) == "retrieval"
                and observation_status in {"no_valid_result", "error"}
            ):
                retrieval_failure_count += 1
            _remember_execution(
                tool_name=name,
                tool_args=resolved_args if isinstance(resolved_args, dict) else {},
                raw_result=result,
                observation_text=observation_text,
                observation_status=observation_status,
                source=source,
                abstract_tool=abstract_label,
                is_recursive=False,
                executed=executed,
            )
            if isinstance(evolution_policy_route, dict):
                last_execution_meta["evolution_policy_route"] = dict(evolution_policy_route)
            _record_progress_for_execution(name, observation_status)

            logger.info(
                "%s[%s][Depth %d] Tool %s completed with status=%s",
                '  ' * depth,
                role_label,
                depth,
                name,
                observation_status,
            )
            logger.info(
                "%s[Observer][Depth %d] Tool %s finalized mode=%s executed=%s status=%s source=%s",
                '  ' * depth,
                depth,
                name,
                mode or "unknown",
                executed,
                observation_status,
                source,
            )
            _log_observation_preview(f"Tool {name} result preview", observation_text)
            return result, observation_text, observation_status

        def _execute_parallel_concrete_tool_calls(
            requests: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            prepared_requests: List[Dict[str, Any]] = []
            executable_requests: List[Dict[str, Any]] = []

            for request in requests:
                prepared_call = _prepare_concrete_tool_call(
                    str(request.get("tool_name", "")).strip(),
                    request.get("tool_args", {}) if isinstance(request.get("tool_args", {}), dict) else {},
                )
                prepared_request = dict(request)
                prepared_request["prepared_call"] = prepared_call
                prepared_requests.append(prepared_request)
                if str(prepared_call.get("mode", "")).strip() == "execute":
                    executable_requests.append(prepared_request)
                else:
                    prepared_request["result"] = _run_prepared_concrete_tool_call(prepared_call)

            if executable_requests:
                worker_count = max(1, len(executable_requests))
                logger.info(
                    "%s[%s][Depth %d] Executing %d tools concurrently within one Parallel step.",
                    '  ' * depth,
                    role_label,
                    depth,
                    len(executable_requests),
                )
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="tool_parallel",
                ) as executor:
                    for request in executable_requests:
                        request["future"] = executor.submit(
                            _run_prepared_concrete_tool_call,
                            request["prepared_call"],
                        )
                    for request in executable_requests:
                        future = request.get("future")
                        try:
                            request["result"] = future.result()
                        except Exception as exc:
                            request["error"] = exc
                        finally:
                            request.pop("future", None)

            return prepared_requests

        def _execute_concrete_tool_call(name: str, args: Dict[str, Any]) -> Tuple[Any, str, str, bool]:
            prepared_call = _prepare_concrete_tool_call(name, args)
            result = _run_prepared_concrete_tool_call(prepared_call)
            finalized_result, observation_text, observation_status = _finalize_prepared_concrete_tool_call(
                prepared_call,
                result,
                abstract_tool=name,
            )
            return (
                finalized_result,
                observation_text,
                observation_status,
                bool(prepared_call.get("executed", True)),
            )

        def _extract_error_text_from_payload(payload: Any) -> str:
            chunks: List[str] = []

            def _append_text(value: Any) -> None:
                text = " ".join(str(value or "").split()).strip()
                if text and text not in chunks:
                    chunks.append(text)

            if isinstance(payload, dict):
                for key in ("error", "result", "message", "warning", "detail", "details"):
                    value = payload.get(key)
                    if value not in (None, "", [], {}):
                        _append_text(value)
            elif payload not in (None, "", [], {}):
                _append_text(payload)

            return " | ".join(chunks)

        def _looks_like_parameter_error(*candidates: Any) -> bool:
            fragments: List[str] = []

            for value in candidates:
                if isinstance(value, dict):
                    extracted = _extract_error_text_from_payload(value)
                    if extracted:
                        fragments.append(extracted)
                    continue
                text = " ".join(str(value or "").split()).strip()
                if text:
                    fragments.append(text)

            normalized = " ".join(fragments).lower().strip()
            if not normalized:
                return False

            strong_markers = (
                "missing required parameter",
                "missing required argument",
                "missing positional argument",
                "unexpected keyword argument",
                "unexpected argument",
                "invalid parameter",
                "invalid parameters",
                "invalid argument",
                "invalid args",
                "parameter payload",
                "must be an integer",
                "must be numeric",
                "requires image_region",
                "requires timestamp",
                "requires t_start",
                "requires t_end",
                "t_start and t_end must",
                "parsed parameter payload lost all executable arguments",
            )
            if any(marker in normalized for marker in strong_markers):
                return True

            soft_tokens = (
                "parameter",
                "argument",
                "schema",
                "field",
                "payload",
                "timestamp",
                "t_start",
                "t_end",
                "density",
                "query",
            )
            soft_verbs = (
                "missing",
                "invalid",
                "must",
                "require",
                "required",
                "expected",
            )
            return any(token in normalized for token in soft_tokens) and any(verb in normalized for verb in soft_verbs)

        def _coerce_schema_value(name: str, schema: Dict[str, Any], value: Any) -> Any:
            field_type = schema.get("type")
            if isinstance(field_type, list):
                types = {str(item).strip().lower() for item in field_type}
            else:
                types = {str(field_type).strip().lower()} if field_type else set()
            allows_string = "string" in types

            if value is None:
                return None

            if "array" in types:
                if isinstance(value, str) and (schema.get("items") or "," in value or name == "fields"):
                    items = [part.strip() for part in re.split(r"[\s,]+", value) if part.strip()]
                    item_schema = schema.get("items", {}) if isinstance(schema.get("items"), dict) else {}
                    return [_coerce_schema_value(f"{name}[]", item_schema, item) for item in items]
                if isinstance(value, (list, tuple)):
                    item_schema = schema.get("items", {}) if isinstance(schema.get("items"), dict) else {}
                    return [_coerce_schema_value(f"{name}[]", item_schema, item) for item in value]

            if "integer" in types and isinstance(value, str) and not allows_string:
                text = value.strip()
                if re.fullmatch(r"[+-]?\d+", text):
                    return int(text)
                raise ValueError(f"{name} must be an integer, but received {value!r}.")

            if "number" in types and isinstance(value, str) and not allows_string:
                text = value.strip()
                try:
                    return float(text)
                except Exception as exc:
                    raise ValueError(f"{name} must be numeric, but received {value!r}.") from exc

            return value

        def _normalize_visual_probe_param_aliases(args: Dict[str, Any]) -> Dict[str, Any]:
            normalized = dict(args or {})

            for source_key, target_key in (
                ("time_range_start", "t_start"),
                ("start_time", "t_start"),
                ("time_range_end", "t_end"),
                ("end_time", "t_end"),
                ("timestamp_range", "time_range"),
                ("search_window", "time_range"),
            ):
                if normalized.get(target_key) in (None, "") and normalized.get(source_key) not in (None, ""):
                    normalized[target_key] = normalized.get(source_key)

            raw_video = (
                normalized.get("video")
                or normalized.get("video_path")
                or normalized.get("video_id")
            )
            raw_video_text = str(raw_video or "").strip().lower()
            if (
                normalized.get("video") in (None, "")
                and normalized.get("video_path") in (None, "")
                and raw_video not in (None, "")
                and raw_video_text not in {"local_video", "current_video", "current_clip", "video", "clip"}
            ):
                normalized["video"] = raw_video

            if normalized.get("density") in (None, ""):
                frame_count = normalized.get("frame_count", normalized.get("num_frames"))
                t_start = normalized.get("t_start")
                t_end = normalized.get("t_end")
                if frame_count not in (None, "") and t_start not in (None, "") and t_end not in (None, ""):
                    try:
                        sample_count = max(1, int(float(frame_count)))
                        start = float(t_start)
                        end = float(t_end)
                        window = max(0.0, end - start)
                        normalized["density"] = 0.5 if sample_count <= 1 or window <= 0 else max(window / float(sample_count), 1e-3)
                    except Exception:
                        pass

            for stale_key in (
                "time_range_start",
                "time_range_end",
                "start_time",
                "end_time",
                "frame_count",
                "num_frames",
                "video_id",
            ):
                normalized.pop(stale_key, None)

            return normalized

        def _normalize_action_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(args, dict):
                return {}
            if name == "Parallel":
                batch = []
                for call in args.get("batch", []):
                    if not isinstance(call, dict):
                        continue
                    batch_name = str(call.get("name", "")).strip()
                    batch_args = call.get("args", {})
                    batch.append({
                        "name": batch_name,
                        "args": _normalize_action_args(batch_name, batch_args if isinstance(batch_args, dict) else {}),
                    })
                return {"batch": batch}

            normalized = dict(args)
            raw_declared = any(
                bool(normalized.get(meta_key))
                for meta_key in ("__raw_params_declared__", "__raw_params_declared")
            )
            raw_param_keys = normalized.get("__raw_param_keys__")
            if raw_param_keys in (None, ""):
                raw_param_keys = normalized.get("__raw_param_keys")
            for meta_key in (
                "__raw_params_declared__",
                "__raw_params_declared",
                "__raw_param_keys__",
                "__raw_param_keys",
            ):
                normalized.pop(meta_key, None)
            if raw_declared:
                normalized["__raw_params_declared__"] = True
                if isinstance(raw_param_keys, (list, tuple)):
                    normalized["__raw_param_keys__"] = [
                        str(key).strip()
                        for key in raw_param_keys
                        if str(key).strip()
                    ]

            if name in {"Inspect_Frame", "OCR_Extract"}:
                normalized = _normalize_visual_probe_param_aliases(normalized)

            tool = registry.get_tool(name, log_missing=False)
            if tool.is_virtual:
                return normalized

            properties = tool.parameters.get("properties", {}) if isinstance(tool.parameters, dict) else {}
            for key, value in list(normalized.items()):
                if key.startswith("__"):
                    continue
                schema = properties.get(key)
                if not isinstance(schema, dict):
                    continue
                normalized[key] = _coerce_schema_value(key, schema, value)
            if name == "Segment_Retrieve" and "fields" in normalized:
                raw_fields = normalized.get("fields")
                if isinstance(raw_fields, list):
                    normalized_fields = _normalize_field_aliases([str(item) for item in raw_fields])
                elif raw_fields in (None, ""):
                    normalized_fields = []
                else:
                    normalized_fields = _normalize_field_aliases([str(raw_fields)])
                if raw_fields not in (None, "", []) and not normalized_fields:
                    raise ValueError(
                        "fields must use Segment_Retrieve schema names "
                        "(caption, transcript, graph, frame_times, frame_paths)."
                    )
                normalized["fields"] = normalized_fields
            return normalized

        def _classify_root_action(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            tool = registry.get_tool(name, log_missing=False)
            from toolkit.impl.retrieval import get_sparse_scope_hidden_tool_payload

            normalized_name = str(name or "").strip()
            executable_args = {
                key: value
                for key, value in (args.items() if isinstance(args, dict) else [])
                if not str(key).startswith("__")
            }

            video_hint = None
            if isinstance(args, dict):
                for key in ("video", "video_path", "segment_id"):
                    value = args.get(key)
                    if value not in (None, ""):
                        video_hint = str(value)
                        break
            if get_sparse_scope_hidden_tool_payload(name, video_hint=video_hint) is not None:
                return {
                    "rule": "rule_3_or_4_route_first",
                    "tool_exists": False,
                    "direct_executable": False,
                    "target_tool_name": "",
                    "allowed_resolution_modes": [
                        "single_replacement",
                        "parallel_decomposition",
                    ],
                    "locked_target": False,
                    "hidden_by_sparse_scope": True,
                }
            if (
                normalized_name == "Python_Executor"
                and _active_local_video_scope_present()
                and not _goal_requests_external_web()
                and executable_args.get("code_script") in (None, "")
            ):
                return {
                    "rule": "rule_3_or_4_route_first",
                    "tool_exists": False,
                    "direct_executable": False,
                    "target_tool_name": "",
                    "allowed_resolution_modes": [
                        "single_replacement",
                        "parallel_decomposition",
                    ],
                    "locked_target": False,
                    "python_executor_requires_evidence_first": True,
                }
            return classify_root_action(
                tool_name=name,
                tool=tool,
                args=args if isinstance(args, dict) else {},
            )

        def _probe_budget_exhausted() -> bool:
            return (
                iteration >= max(3, agent.max_iterations - 1)
                or repeated_loop_violations >= 2
                or (visual_probe_count > 0 and retrieval_probe_count > 1 and consecutive_low_progress >= 1)
            )

        def _active_local_video_scope_present() -> bool:
            return bool(runtime_session.get_active_video_names())

        def _remaining_wall_clock_seconds() -> Optional[float]:
            deadline = self.wall_clock_deadline_monotonic
            if deadline is None:
                return None
            return deadline - time.perf_counter()

        def _per_turn_generation_timeout_seconds() -> Optional[float]:
            raw_value = (
                self.planner_generation_timeout_seconds
                if depth == 0
                else self.subagent_generation_timeout_seconds
            )
            if raw_value is None or float(raw_value) <= 0:
                return None
            return float(raw_value)

        def _generation_kwargs_with_deadlines(base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
            updated = dict(base_kwargs or {})
            if self.wall_clock_deadline_monotonic is not None:
                updated["wall_clock_deadline_monotonic"] = self.wall_clock_deadline_monotonic
            per_turn_timeout_seconds = _per_turn_generation_timeout_seconds()
            if per_turn_timeout_seconds is not None:
                updated["generation_timeout_seconds"] = per_turn_timeout_seconds
                updated["generation_deadline_monotonic"] = time.perf_counter() + per_turn_timeout_seconds
            return updated

        def _generate_with_heartbeat(
            *,
            messages_payload: List[Dict[str, Any]],
            generation_payload: Dict[str, Any],
            iteration_idx: int,
            phase: str,
        ) -> Tuple[str, float]:
            heartbeat_interval_seconds = self.generation_heartbeat_interval_seconds
            generation_state: Dict[str, Any] = {"response": None, "error": None}
            generation_done = threading.Event()
            wall_clock_warning_emitted = False
            generation_deadline = generation_payload.get("generation_deadline_monotonic")
            started = time.perf_counter()

            def _should_use_inline_rollout_generation(llm: BaseLLM) -> bool:
                if bool(getattr(llm, "supports_internal_generation_heartbeat", False)):
                    return True
                # Older / deserialized rollout adapters may miss the instance
                # flag even though they still route generation into the local
                # in-process actor. Detect that shape structurally so planner
                # generation never falls back to the helper-thread path.
                return (
                    type(llm).__name__ == "RolloutChatAdapter"
                    and callable(getattr(llm, "response_generator", None))
                )

            def _deadline_remaining_text(raw_deadline: Any) -> str:
                if raw_deadline in (None, ""):
                    return "unbounded"
                try:
                    return f"{float(raw_deadline) - time.perf_counter():.3f}s"
                except Exception:
                    return "invalid"

            def _emit_generation_progress(update: Optional[Dict[str, Any]] = None) -> None:
                payload = dict(update or {})
                elapsed_value = payload.get("elapsed_seconds")
                try:
                    elapsed_seconds = (
                        float(elapsed_value)
                        if elapsed_value not in (None, "")
                        else (time.perf_counter() - started)
                    )
                except Exception:
                    elapsed_seconds = time.perf_counter() - started
                remaining_seconds = _remaining_wall_clock_seconds()
                remaining_generation_seconds = payload.get("generation_remaining_seconds")
                if remaining_generation_seconds in (None, "") and generation_deadline not in (None, ""):
                    try:
                        remaining_generation_seconds = float(generation_deadline) - time.perf_counter()
                    except Exception:
                        remaining_generation_seconds = None
                self._emit_progress(
                    role=role_label.lower(),
                    depth=depth,
                    iteration=iteration_idx,
                    phase=f"generation_{phase}",
                    status=str(payload.get("status", "running")).strip() or "running",
                    elapsed_seconds=round(float(elapsed_seconds), 3),
                    generation_remaining_seconds=(
                        round(float(remaining_generation_seconds), 3)
                        if remaining_generation_seconds not in (None, "")
                        else None
                    ),
                    wall_clock_remaining_seconds=(
                        round(float(remaining_seconds), 3)
                        if remaining_seconds is not None
                        else None
                    ),
                    current_tool=str(last_execution_meta.get("tool", "")).strip(),
                    source=str(payload.get("source", "llm")).strip() or "llm",
                    result_preview=_truncate_text(str(payload.get("result_preview", "")).strip(), 220),
                )

            llm_type_name = type(current_llm).__name__
            llm_supports_flag = bool(getattr(current_llm, "supports_internal_generation_heartbeat", False))
            llm_has_response_generator = callable(getattr(current_llm, "response_generator", None))
            use_inline_rollout_generation = _should_use_inline_rollout_generation(current_llm)
            logger.info(
                "%s[%s][Depth %d][Iter %d][Generation:%s][DIAG] llm_type=%s llm_role=%s llm_enable_thinking=%s inline_selected=%s supports_flag=%s has_response_generator=%s message_count=%d generation_keys=%s generation_deadline_remaining=%s wall_clock_remaining=%s",
                '  ' * depth,
                role_label,
                depth,
                iteration_idx,
                phase,
                llm_type_name,
                getattr(current_llm, "model_role", None),
                getattr(current_llm, "enable_thinking", None),
                use_inline_rollout_generation,
                llm_supports_flag,
                llm_has_response_generator,
                len(messages_payload or []),
                sorted(str(key) for key in dict(generation_payload or {}).keys()),
                _deadline_remaining_text(generation_deadline),
                _deadline_remaining_text(self.wall_clock_deadline_monotonic),
            )

            if use_inline_rollout_generation:
                inline_generation_payload = dict(generation_payload or {})
                inline_generation_payload["generation_heartbeat_interval_seconds"] = heartbeat_interval_seconds
                inline_generation_payload["progress_callback"] = _emit_generation_progress
                logger.info(
                    "%s[%s][Depth %d][Iter %d][Generation:%s] using inline rollout heartbeat path (llm_type=%s).",
                    '  ' * depth,
                    role_label,
                    depth,
                    iteration_idx,
                    phase,
                    type(current_llm).__name__,
                )
                response = current_llm.generate(
                    messages_payload,
                    **inline_generation_payload,
                )
                return str(response or ""), time.perf_counter() - started

            delegate_llm = getattr(current_llm, "delegate_llm", None)
            if delegate_llm is not None:
                logger.info(
                    "%s[%s][Depth %d][Iter %d][Generation:%s][DIAG] using helper-thread generation path for delegated provider llm_type=%s delegate_type=%s.",
                    '  ' * depth,
                    role_label,
                    depth,
                    iteration_idx,
                    phase,
                    llm_type_name,
                    type(delegate_llm).__name__,
                )
            else:
                log_method = logger.info if llm_type_name == "TransformersProvider" else logger.debug
                log_method(
                    "%s[%s][Depth %d][Iter %d][Generation:%s][DIAG] using helper-thread generation path for llm_type=%s because inline heartbeat is unavailable.",
                    '  ' * depth,
                    role_label,
                    depth,
                    iteration_idx,
                    phase,
                    llm_type_name,
                )

            def _generation_worker() -> None:
                try:
                    generation_state["response"] = current_llm.generate(
                        messages_payload,
                        **generation_payload,
                    )
                except Exception as exc:
                    generation_state["error"] = exc
                finally:
                    generation_done.set()

            worker = threading.Thread(
                target=_generation_worker,
                name=f"video-agent-{role_label}-{depth}-{iteration_idx}-{phase}",
                daemon=True,
            )
            worker.start()

            while True:
                wait_timeout_seconds = heartbeat_interval_seconds
                if generation_deadline not in (None, ""):
                    try:
                        remaining_before_wait = float(generation_deadline) - time.perf_counter()
                    except Exception:
                        remaining_before_wait = None
                    if remaining_before_wait is not None:
                        if remaining_before_wait <= 0 and not generation_done.is_set():
                            elapsed = time.perf_counter() - started
                            logger.error(
                                "%s[%s][Depth %d][Iter %d][Generation:%s] per-turn deadline reached while waiting for provider response.",
                                '  ' * depth,
                                role_label,
                                depth,
                                iteration_idx,
                                phase,
                            )
                            raise GenerationTimeoutError(
                                "Per-turn generation deadline reached before LLM provider returned.",
                                partial_response=str(generation_state.get("response") or ""),
                                elapsed_seconds=elapsed,
                                role=role_label.lower(),
                                phase=phase,
                            )
                        wait_timeout_seconds = min(
                            wait_timeout_seconds,
                            max(0.1, float(remaining_before_wait)),
                        )

                if generation_done.wait(timeout=wait_timeout_seconds):
                    break

                elapsed = time.perf_counter() - started
                remaining_seconds = _remaining_wall_clock_seconds()
                remaining_generation_seconds = None
                if generation_deadline not in (None, ""):
                    try:
                        remaining_generation_seconds = float(generation_deadline) - time.perf_counter()
                    except Exception:
                        remaining_generation_seconds = None
                logger.info(
                    "%s[%s][Depth %d][Iter %d][Generation:%s] still running elapsed=%.1fs remaining_generation_seconds=%s remaining_wall_clock_seconds=%s",
                    '  ' * depth,
                    role_label,
                    depth,
                    iteration_idx,
                    phase,
                    elapsed,
                    (
                        f"{float(remaining_generation_seconds):.3f}"
                        if remaining_generation_seconds is not None
                        else "unbounded"
                    ),
                    (
                        f"{float(remaining_seconds):.3f}"
                        if remaining_seconds is not None
                        else "unbounded"
                    ),
                )
                _emit_generation_progress(
                    {
                        "elapsed_seconds": elapsed,
                        "generation_remaining_seconds": remaining_generation_seconds,
                        "source": "heartbeat_wait",
                        "result_preview": str(last_execution_meta.get("observation", "")).strip(),
                    }
                )
                if remaining_generation_seconds is not None and remaining_generation_seconds <= 0:
                    logger.error(
                        "%s[%s][Depth %d][Iter %d][Generation:%s] per-turn deadline exceeded (remaining_generation_seconds=%.3f); aborting current turn.",
                        '  ' * depth,
                        role_label,
                        depth,
                        iteration_idx,
                        phase,
                        float(remaining_generation_seconds),
                    )
                    raise GenerationTimeoutError(
                        "Per-turn generation timeout exceeded while waiting for provider response.",
                        partial_response=str(generation_state.get("response") or ""),
                        elapsed_seconds=elapsed,
                        role=role_label.lower(),
                        phase=phase,
                    )
                if (
                    not wall_clock_warning_emitted
                    and remaining_seconds is not None
                    and remaining_seconds <= 0
                ):
                    logger.warning(
                        "%s[%s][Depth %d][Iter %d][Generation:%s] wall-clock budget expired while waiting for generation to return.",
                        '  ' * depth,
                        role_label,
                        depth,
                        iteration_idx,
                        phase,
                    )
                    wall_clock_warning_emitted = True

            worker.join(timeout=0)
            error = generation_state.get("error")
            if error is not None:
                raise error
            return str(generation_state.get("response") or ""), time.perf_counter() - started

        def _build_wall_clock_timeout_message() -> str:
            configured_budget = float(self.wall_clock_budget_seconds or 0.0)
            budget_text = f"{configured_budget:.0f}s" if configured_budget > 0 else "configured sample budget"
            role = "planner" if depth <= 0 else "subagent"
            return (
                f"Failed: {role} exceeded the wall-clock time budget ({budget_text}) before completing the task. "
                "Stop reasoning and return control to avoid burning additional rollout time."
            )

        def _abort_due_to_wall_clock_budget(reason: str) -> Tuple[str, bool, bool]:
            message = _build_wall_clock_timeout_message()
            logger.warning(
                "%s[%s][Depth %d] %s remaining_wall_clock_seconds=%.3f",
                '  ' * depth,
                role_label,
                depth,
                reason,
                float(_remaining_wall_clock_seconds() or 0.0),
            )
            agent.trajectory.add_step(
                thought="Wall-clock budget exhausted",
                plan="Abort the current trajectory to honor the sample time budget.",
                action="System",
                action_args={},
                observation=message,
                is_recursive=False,
            )
            return message, False, True

        def _abort_due_to_generation_timeout(
            *,
            iteration_idx: int,
            phase: str,
            exc: GenerationTimeoutError,
        ) -> Tuple[str, bool, bool]:
            timeout_seconds = _per_turn_generation_timeout_seconds()
            timeout_text = (
                f"{float(timeout_seconds):.0f}s"
                if timeout_seconds is not None and timeout_seconds > 0
                else "configured per-turn budget"
            )
            partial_response = str(getattr(exc, "partial_response", "") or "").strip()
            self._write_raw_trace(
                depth=depth,
                iteration=iteration_idx,
                goal=goal,
                raw_response=partial_response or f"[GENERATION_TIMEOUT] {exc}",
            )
            if partial_response:
                logger.info(
                    "%s[%s][LLM_FULL_RESPONSE_BEGIN][Depth %d][Iter %d][%s][TIMEOUT_PARTIAL]\n%s\n%s[%s][LLM_FULL_RESPONSE_END][Depth %d][Iter %d][%s][TIMEOUT_PARTIAL]",
                    '  ' * depth,
                    role_label,
                    depth,
                    iteration_idx,
                    phase.upper(),
                    partial_response,
                    '  ' * depth,
                    role_label,
                    depth,
                    iteration_idx,
                    phase.upper(),
                )
            message = (
                f"Failed: {('planner' if depth == 0 else 'subagent')} generation exceeded the per-turn timeout "
                f"({timeout_text}) during {phase}. Stop the current trajectory instead of waiting indefinitely."
            )
            logger.error(
                "%s[%s][Depth %d][Iter %d][Generation:%s] per-turn timeout triggered after %.3fs",
                '  ' * depth,
                role_label,
                depth,
                iteration_idx,
                phase,
                float(getattr(exc, "elapsed_seconds", 0.0) or 0.0),
            )
            if settings.enable_llm_output_debug:
                agent.trajectory.add_debug_event({
                    "type": "generation_timeout",
                    "depth": depth,
                    "iteration": iteration_idx,
                    "phase": phase,
                    "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": float(getattr(exc, "elapsed_seconds", 0.0) or 0.0),
                    "partial_response_chars": len(partial_response),
                })
            agent.trajectory.add_step(
                thought="Generation timeout",
                plan="Abort this trajectory because the current LLM turn exceeded the per-turn budget.",
                action="System",
                action_args={},
                observation=message,
                is_recursive=False,
            )
            self._emit_progress(
                role=role_label.lower(),
                depth=depth,
                iteration=iteration_idx,
                phase=f"generation_{phase}_timeout",
                status="error",
                elapsed_seconds=round(float(getattr(exc, "elapsed_seconds", 0.0) or 0.0), 3),
                current_tool=str(last_execution_meta.get("tool", "")).strip(),
                result_preview=_truncate_text(partial_response or message, 180),
            )
            return message, False, True

        def _goal_requests_external_web() -> bool:
            goal_text = " ".join(str(goal or "").strip().lower().split())
            return any(
                marker in goal_text
                for marker in ("web", "internet", "latest", "current", "news", "online", "external")
            )

        def _render_tool_lines(raw_result: Any) -> List[str]:
            raw = raw_result if isinstance(raw_result, list) else []
            lines: List[str] = []
            for t in raw:
                if (
                    isinstance(t, dict)
                    and "name" in t
                    and str(t.get("name", "")).strip() != "Tool_Search"
                ):
                    desc = t.get("description", "")
                    params = ", ".join(
                        t.get("parameters", {}).get("properties", {}).keys()
                    )
                    lines.append(f"- {t['name']}({params}): {desc}")
            return lines

        def _filter_toolsearch_candidates(raw_candidates: Any) -> List[Dict[str, Any]]:
            if isinstance(raw_candidates, dict) and isinstance(raw_candidates.get("items"), list):
                raw_candidates = raw_candidates.get("items")
            if not isinstance(raw_candidates, list):
                return []
            return [
                item for item in raw_candidates
                if isinstance(item, dict) and str(item.get("name", "")).strip() != "Tool_Search"
            ]

        def _merge_discovered_tool_lines(lines: List[str]) -> int:
            nonlocal discovered_tools_desc
            if not lines:
                return 0
            if discovered_tools_desc:
                existing_names = {
                    ln.split("(")[0].lstrip("- ")
                    for ln in discovered_tools_desc.splitlines()
                    if ln.strip()
                }
                new_lines = [
                    ln for ln in lines
                    if ln.split("(")[0].lstrip("- ") not in existing_names
                ]
                if new_lines:
                    discovered_tools_desc += "\n" + "\n".join(new_lines)
                return len(new_lines)
            discovered_tools_desc = "\n".join(lines)
            return len(lines)

        def _maybe_refresh_routes_for_low_progress(
            action_name: str,
            observation_text: str,
        ) -> None:
            nonlocal route_refresh_count
            if depth != 0 or not _probe_budget_exhausted():
                return

            current_signature = str(last_observation_signature or "").strip()
            if not current_signature:
                return
            refresh_key = f"{_action_family(action_name)}:{current_signature}"
            if refresh_key in route_refresh_signatures:
                return

            tool_search = registry.get_tool("Tool_Search", log_missing=False)
            if tool_search.is_virtual:
                return

            intent_chunks = [
                f"goal: {goal}",
                f"last_action: {action_name}",
                f"recent_low_information_observation: {_truncate_text(observation_text, 320)}",
            ]
            missing_labels = [
                str(slot.get("label", "")).strip()
                for slot in evidence_slots
                if str(slot.get("evidence_status", "")).strip().lower() != "verified"
            ]
            if missing_labels:
                intent_chunks.append(f"missing_evidence: {', '.join(label for label in missing_labels if label)}")
            recent_tools = runtime_session.get_recent_tools(limit=6)
            if recent_tools:
                intent_chunks.append(f"recent_tools: {', '.join(recent_tools)}")
            intent = " | ".join(chunk for chunk in intent_chunks if chunk)

            try:
                refresh_started = time.perf_counter()
                raw_results = tool_search.execute(
                    intent_description=intent,
                    top_k=5,
                    tool_type="all",
                )
                runtime_session.record_timing(
                    "planner.route_refresh_seconds",
                    time.perf_counter() - refresh_started,
                )
            except Exception as exc:
                logger.warning("Automatic route refresh failed after low progress: %s", exc)
                return

            candidate_tools = extract_candidate_tools(_filter_toolsearch_candidates(raw_results))
            if not candidate_tools:
                return

            route_refresh_signatures.add(refresh_key)
            route_refresh_count += 1
            _merge_discovered_tool_lines(_render_tool_lines(candidate_tools))
            logger.info(
                "%s[%s][Depth %d] Internal route refresh discovered %d candidate tools.",
                '  ' * depth,
                role_label,
                depth,
                len(candidate_tools),
            )

        def _resolve_virtual_subagent_action(
            *,
            abstract_tool: str,
            abstract_args: Dict[str, Any],
            description_text: str,
        ) -> Dict[str, Any]:
            signature = build_abstract_signature(
                tool_name=abstract_tool,
                args=abstract_args,
                description=description_text,
            )
            cached = abstract_tool_cache.get(signature)
            if cached is not None:
                return cached

            intent_chunks = [
                f"goal: {goal}",
                f"abstract_tool: {abstract_tool}",
                f"abstract_args: {json.dumps(abstract_args or {}, ensure_ascii=False)}",
            ]
            if description_text:
                intent_chunks.append(f"description: {description_text}")
            intent = " | ".join(intent_chunks)
            tool_search = registry.get_tool("Tool_Search")
            if tool_search.is_virtual:
                resolution = {
                    "resolved": False,
                    "observation": {
                        "status": "error",
                        "result": "Tool_Search is unavailable in registry.",
                        "candidate_tools": [],
                    },
                }
                abstract_tool_cache[signature] = resolution
                return resolution
            results = tool_search.execute(intent_description=intent, top_k=5, tool_type="all")
            filtered_results = _filter_toolsearch_candidates(results)
            lines = _render_tool_lines(filtered_results)
            _merge_discovered_tool_lines(lines)
            resolution = resolve_abstract_tool_call(
                abstract_tool=abstract_tool,
                abstract_args=abstract_args,
                description=description_text,
                candidates=extract_candidate_tools(filtered_results),
            )
            abstract_tool_cache[signature] = resolution
            return resolution

        def _resolve_root_action_via_toolsearch_top1(
            *,
            abstract_tool: str,
            abstract_args: Dict[str, Any],
            description_text: str,
        ) -> Dict[str, Any]:
            signature = "ablation_top1|" + build_abstract_signature(
                tool_name=abstract_tool,
                args=abstract_args,
                description=description_text,
            )
            cached = abstract_tool_cache.get(signature)
            if cached is not None:
                return cached

            tool_search = registry.get_tool("Tool_Search")
            if tool_search.is_virtual:
                resolution = {
                    "resolved": False,
                    "observation": {
                        "status": "error",
                        "result": "Tool_Search is unavailable in registry.",
                        "candidate_tools": [],
                        "source": "ablation_toolsearch_direct",
                    },
                }
                abstract_tool_cache[signature] = resolution
                return resolution

            intent_chunks = [
                f"goal: {goal}",
                f"abstract_tool: {abstract_tool}",
                f"abstract_args: {json.dumps(abstract_args or {}, ensure_ascii=False)}",
            ]
            if description_text:
                intent_chunks.append(f"description: {description_text}")
            intent = " | ".join(intent_chunks)
            results = tool_search.execute(intent_description=intent, top_k=5, tool_type="all")
            filtered_results = _filter_toolsearch_candidates(results)
            candidate_tools = extract_candidate_tools(filtered_results)
            lines = _render_tool_lines(filtered_results)
            _merge_discovered_tool_lines(lines)

            if not candidate_tools:
                resolution = {
                    "resolved": False,
                    "observation": {
                        "status": "error",
                        "result": (
                            f"Ablation mode could not find any executable Tool_Search candidate for "
                            f"planner action '{abstract_tool}'."
                        ),
                        "candidate_tools": [],
                        "source": "ablation_toolsearch_direct",
                    },
                }
                abstract_tool_cache[signature] = resolution
                return resolution

            top_candidate = dict(candidate_tools[0] or {})
            top_candidate_name = str(top_candidate.get("name", "")).strip()
            if not top_candidate_name:
                resolution = {
                    "resolved": False,
                    "observation": {
                        "status": "error",
                        "result": (
                            f"Ablation mode found malformed Tool_Search candidates for planner action "
                            f"'{abstract_tool}'."
                        ),
                        "candidate_tools": candidate_tools[:5],
                        "source": "ablation_toolsearch_direct",
                    },
                }
                abstract_tool_cache[signature] = resolution
                return resolution

            resolution = resolve_abstract_tool_call(
                abstract_tool=abstract_tool,
                abstract_args=abstract_args,
                description=description_text,
                candidates=[top_candidate],
                target_tool_name=top_candidate_name,
                target_tool_schema=top_candidate,
            )
            observation = dict(resolution.get("observation", {}) or {})
            observation["candidate_tools"] = candidate_tools[:5]
            observation["source"] = "ablation_toolsearch_direct"
            if resolution.get("resolved"):
                observation["result"] = (
                    f"Ablation mode routed planner action '{abstract_tool}' to top-1 Tool_Search candidate "
                    f"'{top_candidate_name}'."
                )
                observation["resolved_tool"] = top_candidate_name
            resolution["observation"] = observation
            abstract_tool_cache[signature] = resolution
            return resolution

        def _is_nonrecoverable_resolution_failure(resolution: Dict[str, Any]) -> bool:
            observation = resolution.get("observation", {}) if isinstance(resolution, dict) else {}
            candidate_tools = observation.get("candidate_tools") if isinstance(observation, dict) else []
            status = str(observation.get("status", "")).strip().lower() if isinstance(observation, dict) else ""
            result_text = str(observation.get("result", "")).strip().lower() if isinstance(observation, dict) else ""
            return (
                not candidate_tools
                and status in {"error", "no_valid_result"}
            ) or "no executable tools were discovered" in result_text or "tool_search is unavailable" in result_text

        if evidence_first_mode:
            _refresh_modality_progress()
        
        if depth == 0 and first_round_context:
            if isinstance(first_round_context, list):
                grounding_label = list(first_round_context)
            else:
                grounding_label = (
                    "[Initial Planner Grounding Evidence - inject once and keep in history]\n"
                    f"{first_round_context}"
                )
            _append_history("user", grounding_label)
            initial_msg = f"Goal: {goal}\n\nUse the initial grounding evidence above for first-step reasoning. Begin reasoning:"
        elif context_info:
            initial_msg = f"Goal: {goal}\n\nContext:\n{context_info}\n\nBegin reasoning:"
        else:
            initial_msg = f"Goal: {goal}\n\nBegin reasoning:"

        if evidence_first_mode:
            initial_msg += (
                "\n\nThis is an evidence-first exact-value task. "
                "Keep the Evidence array updated, prefer retrieval/visual tools, "
                "and continue from nearby windows before broadening the search."
            )
        initial_prompt_tokens = current_llm.count_message_tokens(
            current_llm.format_messages(
                system_prompt=_effective_system_prompt(),
                user_message=initial_msg,
                history=list(history),
            ),
            enable_thinking=generation_kwargs.get("enable_thinking"),
        )
        logger.info(
            "%s[%s][Depth %d] Initial prompt token estimate: %d/%d",
            '  ' * depth,
            role_label,
            depth,
            initial_prompt_tokens,
            max_ctx,
        )
        
        iteration = 0
        finished = False
        final_answer = None
        success = False
        consecutive_parse_failures = 0
        recoverable_empty_evidence_turns = 0
        last_action_signature = ""
        repeated_action_count = 0
        context_overflow_recovery_attempts = 0
        planner_tool_invocation_stats: Dict[str, Dict[str, int]] = {}
        MAX_PARSE_FAILURES = 3  # Abort after N consecutive parse failures
        MAX_REPEATED_ACTIONS = 3

        # Checkpoints live at root-turn boundaries, after delegated calls have
        # joined. Restore explicit Python locals so all closure guards observe
        # the same state; models and the candidate policy remain freshly loaded.
        from evolution.runtime_checkpoint import STATE_NAMES, restore as restore_checkpoint, save as save_checkpoint
        checkpoint_root = (Path(self.raw_trace_jsonl_path).parent / "runtime_checkpoints"
                           if self.raw_trace_jsonl_path else None)
        resume_path = os.environ.get("CFV_RUNTIME_RESUME_CHECKPOINT", "") if depth == 0 else ""
        if resume_path:
            restored_state = restore_checkpoint(self, agent, Path(resume_path), STATE_NAMES)
            history = restored_state['history']
            memory_state = restored_state['memory_state']
            tool_error_streaks = restored_state['tool_error_streaks']
            consecutive_finish_without_answer = restored_state['consecutive_finish_without_answer']
            discovered_tools_desc = restored_state['discovered_tools_desc']
            root_effective_tool_executed = restored_state['root_effective_tool_executed']
            last_observation_signature = restored_state['last_observation_signature']
            consecutive_low_progress = restored_state['consecutive_low_progress']
            low_progress_hint_sent = restored_state['low_progress_hint_sent']
            blocked_action_signatures = restored_state['blocked_action_signatures']
            executed_tool_signatures = restored_state['executed_tool_signatures']
            repeated_loop_violations = restored_state['repeated_loop_violations']
            subagent_toolsearch_failures = restored_state['subagent_toolsearch_failures']
            abstract_tool_cache = restored_state['abstract_tool_cache']
            evidence_slots = restored_state['evidence_slots']
            evidence_revision = restored_state['evidence_revision']
            pending_planner_evidence_slots = restored_state['pending_planner_evidence_slots']
            visual_probe_count = restored_state['visual_probe_count']
            retrieval_probe_count = restored_state['retrieval_probe_count']
            retrieval_failure_count = restored_state['retrieval_failure_count']
            low_signal_recovery_call_count = restored_state['low_signal_recovery_call_count']
            low_signal_visual_probe_streak = restored_state['low_signal_visual_probe_streak']
            evolution_policy_events = restored_state['evolution_policy_events']
            direct_visual_probe_count = restored_state['direct_visual_probe_count']
            cross_modal_verify_count = restored_state['cross_modal_verify_count']
            modality_attempts = restored_state['modality_attempts']
            modality_successes = restored_state['modality_successes']
            needle_candidate_windows = restored_state['needle_candidate_windows']
            needle_candidate_completed = restored_state['needle_candidate_completed']
            needle_candidate_reserved = restored_state['needle_candidate_reserved']
            identical_action_failures = restored_state['identical_action_failures']
            last_execution_meta = restored_state['last_execution_meta']
            no_progress_guard_signature = restored_state['no_progress_guard_signature']
            no_progress_guard_family = restored_state['no_progress_guard_family']
            no_progress_abort_reason = restored_state['no_progress_abort_reason']
            verifier_issue_counts = restored_state['verifier_issue_counts']
            verifier_issue_revisions = restored_state['verifier_issue_revisions']
            finish_rejection_counts = restored_state['finish_rejection_counts']
            finish_rejection_revisions = restored_state['finish_rejection_revisions']
            finish_rejection_reasons = restored_state['finish_rejection_reasons']
            finish_recovery_violations = restored_state['finish_recovery_violations']
            finish_recovery_mode = restored_state['finish_recovery_mode']
            finish_recovery_turns_used = restored_state['finish_recovery_turns_used']
            finish_recovery_finish_pending = restored_state['finish_recovery_finish_pending']
            finish_rejection_streak = restored_state['finish_rejection_streak']
            finish_recovery_last_reason = restored_state['finish_recovery_last_reason']
            finish_recovery_last_targets = restored_state['finish_recovery_last_targets']
            forced_finish_recovery_probe_count = restored_state['forced_finish_recovery_probe_count']
            low_signal_guard_recovery_count = restored_state['low_signal_guard_recovery_count']
            low_signal_recovery_triggered_this_observation = restored_state['low_signal_recovery_triggered_this_observation']
            final_turn_warning_sent = restored_state['final_turn_warning_sent']
            forced_final_turn_retry_used = restored_state['forced_final_turn_retry_used']
            route_refresh_signatures = restored_state['route_refresh_signatures']
            route_refresh_count = restored_state['route_refresh_count']
            adjacent_probe_repeat_count = restored_state['adjacent_probe_repeat_count']
            last_probe_window_state = restored_state['last_probe_window_state']
            active_recovery_system_directive = restored_state['active_recovery_system_directive']
            iteration = restored_state['iteration']
            finished = restored_state['finished']
            final_answer = restored_state['final_answer']
            success = restored_state['success']
            consecutive_parse_failures = restored_state['consecutive_parse_failures']
            recoverable_empty_evidence_turns = restored_state['recoverable_empty_evidence_turns']
            last_action_signature = restored_state['last_action_signature']
            repeated_action_count = restored_state['repeated_action_count']
            context_overflow_recovery_attempts = restored_state['context_overflow_recovery_attempts']
            planner_tool_invocation_stats = restored_state['planner_tool_invocation_stats']

        # Recovery consumes the final reserved turns of the original budget.
        # It must not silently extend the designed 15-turn Planner limit.
        while not finished and iteration < agent.max_iterations:
            if depth == 0 and checkpoint_root is not None:
                try:
                    save_checkpoint(self, agent, locals(), STATE_NAMES, checkpoint_root)
                except (ValueError, TypeError, OSError) as checkpoint_error:
                    agent.trajectory.add_debug_event({"type": "runtime_checkpoint_unavailable", "reason": str(checkpoint_error), "iteration": iteration})
            iteration += 1
            if resume_path:
                agent.trajectory.add_debug_event({"type": "runtime_suffix_iteration", "iteration": iteration})
            recovery_turn = (
                depth == 0
                and evidence_first_mode
                and finish_recovery_mode
                and iteration > max(0, agent.max_iterations - max_finish_recovery_rounds)
            )
            finish_recovery_finish_turn = recovery_turn and finish_recovery_finish_pending
            if recovery_turn and not finish_recovery_finish_turn:
                finish_recovery_turns_used += 1
            iteration_display = f"{iteration}/{agent.max_iterations}"
            if recovery_turn:
                if finish_recovery_finish_turn:
                    iteration_display += " (recovery Finish-only)"
                else:
                    iteration_display += (
                        f" (recovery {finish_recovery_turns_used}/{max_finish_recovery_rounds})"
                    )
            logger.info(
                "%s[%s][Depth %d] Iteration %s",
                '  ' * depth,
                role_label,
                depth,
                iteration_display,
            )
            self._emit_progress(
                role=role_label.lower(),
                depth=depth,
                iteration=iteration,
                phase="iteration_start",
                status="running",
                current_tool=str(last_execution_meta.get("tool", "")).strip(),
                result_preview=_truncate_text(str(last_execution_meta.get("observation", "")).strip(), 180),
            )
            thought = ""
            plan = ""
            action_name = "System"
            action_args: Dict[str, Any] = {}
            iteration_phase = "iteration_start"
            iteration_generated_response = False
            trajectory_step_count_before_iteration = len(agent.trajectory.steps)
            remaining_wall_clock_seconds = _remaining_wall_clock_seconds()
            if remaining_wall_clock_seconds is not None and remaining_wall_clock_seconds <= 0:
                final_answer, success, finished = _abort_due_to_wall_clock_budget(
                    "Wall-clock deadline reached before the next generation turn."
                )
                break

            try:
                # Track dynamic context pressure; folding is triggered automatically
                # after complete user-side history updates, never mid-generation.
                current_dynamic_tokens = _dynamic_token_estimate()
                logger.debug(
                    "%s[%s][Depth %d] Dynamic context tokens before generation: ~%d / %d",
                    '  ' * depth,
                    role_label,
                    depth,
                    current_dynamic_tokens,
                    ctx_threshold,
                )

                if (
                    depth == 0
                    and iteration == agent.max_iterations - 1
                    and not final_turn_warning_sent
                ):
                    _append_history(
                        "user",
                        "Warning: the next planner turn is the final allowed turn. "
                        "If you still have enough evidence, round 15 MUST return strict JSON with "
                        "`Actions: []` and a top-level `Finish` containing `chain_complete`, "
                        "`completion_basis`, and the final `answer`.",
                    )
                    final_turn_warning_sent = True

                # Build messages for the planner LLM.
                if iteration == 1:
                    current_message = initial_msg
                elif finish_recovery_finish_turn:
                    current_message = (
                        "Bounded recovery produced grounded evidence. Finish-only turn: output strict JSON "
                        "with `Actions: []` and a top-level `Finish` using the best supported listed option. "
                        "Do not call tools or convert a prior runtime/tool error into unanswerable."
                    )
                elif recovery_turn:
                    current_message = (
                        f"Finish recovery turn {finish_recovery_turns_used}/{max_finish_recovery_rounds}. "
                        "A previous Finish was rejected. Execute one concrete evidence action with a materially "
                        "different tool, modality, argument, or time window; do not submit Finish again without "
                        "new grounded evidence. If the latest evidence now supports one listed option, submit "
                        "the top-level Finish on the next turn."
                    )
                elif depth == 0 and iteration == agent.max_iterations:
                    current_message = (
                        "Final turn. You MUST output strict JSON with `Actions: []` and a top-level `Finish` only. "
                        "Do not call any more tools on this turn."
                    )
                else:
                    current_message = "Continue reasoning:"
                repeated_tool_advisory = _build_root_tool_retry_advisory()
                if repeated_tool_advisory:
                    current_message = f"{current_message}\n\n{repeated_tool_advisory}"

                # Prepend folded memory state and discovered tools into the history window.
                full_history: List[Dict[str, str]] = []
                folded_state = _memory_state_text()
                if folded_state:
                    full_history.append({
                        "role": "user",
                        "content": f"[Fold_Memory State]\n{folded_state}",
                    })
                if depth > 0 and discovered_tools_desc:
                    full_history.append({
                        "role": "user",
                        "content": f"[Discovered Tools - available for this session]\n{discovered_tools_desc}",
                    })
                    full_history.append({
                        "role": "assistant",
                        "content": "Understood. I can now call these tools.",
                    })
                full_history.extend(history)

                messages = current_llm.format_messages(
                    system_prompt=_effective_system_prompt(),
                    user_message=current_message,
                    history=full_history
                )
                 
                effective_generation_kwargs = _generation_kwargs_with_deadlines(generation_kwargs)
                if iteration == 1:
                    _log_first_input_payload(messages, effective_generation_kwargs)
                try:
                    raw_response, llm_elapsed_seconds = _generate_with_heartbeat(
                        messages_payload=messages,
                        generation_payload=effective_generation_kwargs,
                        iteration_idx=iteration,
                        phase="main",
                    )
                    iteration_generated_response = True
                except GenerationTimeoutError as exc:
                    final_answer, success, finished = _abort_due_to_generation_timeout(
                        iteration_idx=iteration,
                        phase="main",
                        exc=exc,
                    )
                    break
                runtime_session.record_timing(
                    "planner.llm_generate_seconds" if depth == 0 else "subagent.llm_generate_seconds",
                    llm_elapsed_seconds,
                )

                # Track 1: persist full raw response (incl. <think>) to JSONL
                self._write_raw_trace(
                    depth=depth, iteration=iteration,
                    goal=goal, raw_response=raw_response,
                )
                response = _normalize_llm_response_for_role(raw_response)
                response_for_log = _normalize_llm_response_for_log(
                    raw_response,
                    enable_thinking=effective_generation_kwargs.get("enable_thinking"),
                )
                raw_response_text = agent.sanitize_response(raw_response)
                _log_llm_response_sanitization(
                    iteration_idx=iteration,
                    raw_response_text=raw_response_text,
                    log_response_text=response_for_log,
                    effective_response_text=response,
                )
                _log_unexpected_think_when_disabled(
                    iteration_idx=iteration,
                    log_response_text=response_for_log,
                    enable_thinking=effective_generation_kwargs.get("enable_thinking"),
                )

                logger.debug(f"{'  ' * depth}LLM Response:\n{response}")
                logger.info(
                    "%s[%s][LLM_FULL_RESPONSE_BEGIN][Depth %d][Iter %d]\n%s\n%s[%s][LLM_FULL_RESPONSE_END][Depth %d][Iter %d]",
                    '  ' * depth,
                    role_label,
                    depth,
                    iteration,
                    response_for_log,
                    '  ' * depth,
                    role_label,
                    depth,
                    iteration,
                )
                debug_event: Optional[Dict[str, Any]] = None
                if settings.enable_llm_output_debug:
                    debug_event = {
                        "type": "llm_iteration",
                        "agent_role": "planner" if depth == 0 else "subagent",
                        "kernel": (
                            self.role_agents.manifest().get("roles", {}).get("planner", {})
                            if depth == 0
                            else {"agent_role": "subagent", "provider_class": type(current_llm).__name__}
                        ),
                        "depth": depth,
                        "iteration": iteration,
                        "response": response,
                        "raw_response_chars": len(raw_response or ""),
                        "response_normalized": response != str(raw_response or "").strip(),
                    }
                    logger.info(
                        "%s[%s][LLM_DEBUG][Depth %d][Iter %d] Effective response captured (%d chars, raw=%d chars).",
                        '  ' * depth,
                        role_label,
                        depth,
                        iteration,
                        len(response or ""),
                        len(raw_response or ""),
                    )

                remaining_wall_clock_seconds = _remaining_wall_clock_seconds()
                deadline_reached_after_generation = (
                    remaining_wall_clock_seconds is not None and remaining_wall_clock_seconds <= 0
                )
                if deadline_reached_after_generation:
                    logger.warning(
                        "%s[%s][Depth %d] Wall-clock deadline reached after a completed LLM generation turn; "
                        "preserving the response for finish parsing. remaining_wall_clock_seconds=%.3f",
                        '  ' * depth,
                        role_label,
                        depth,
                        float(remaining_wall_clock_seconds or 0.0),
                    )
                 
                iteration_phase = "response_parse"
                structured_payload = agent._extract_json_payload(response)
                parsed_evidence_slots: List[Dict[str, Any]] = []
                if evidence_first_mode:
                    parsed_evidence_slots = _extract_evidence_slots(structured_payload)

                # Parse response
                thought, plan, action_name, action_args = agent._parse_response(response)
                # Evidence embedded in a proposed Finish is a synthesis, not a
                # new observation. It must remain in the raw trajectory for
                # auditability but cannot mutate the active safety ledger; the
                # old ordering let repeated Finish calls manufacture changing
                # evidence and defeat both verifier stability and loop
                # detection. Evidence-only checkpoints and concrete tool
                # turns may still advance the ledger.
                pending_planner_evidence_slots = (
                    list(parsed_evidence_slots)
                    if evidence_first_mode
                    and parsed_evidence_slots
                    and str(action_name or "").lower() != "finish"
                    else []
                )
                recoverable_empty_evidence_turn = bool(
                    not action_name
                    and recoverable_empty_evidence_turns == 0
                    and _is_recoverable_empty_evidence_turn(structured_payload)
                )
                forced_finish_boundary = depth == 0 and (
                    iteration == agent.max_iterations
                    or (
                        recovery_turn
                        and (
                            finish_recovery_turns_used >= max_finish_recovery_rounds
                            or finish_recovery_finish_turn
                        )
                    )
                )
                if forced_finish_boundary and str(action_name or "").lower() != "finish":
                    forced_final_turn_retry_used = True
                    logger.warning(
                        "%s[%s][Depth %d] Finish boundary produced non-finish action '%s'. Triggering one finish-only retry.",
                        '  ' * depth,
                        role_label,
                        depth,
                        action_name or "[none]",
                    )
                    forced_history = list(full_history) + [{"role": "assistant", "content": response}]
                    forced_messages = current_llm.format_messages(
                        system_prompt=_effective_system_prompt(),
                        user_message=(
                            "Finish boundary retry. Output strict JSON with `Actions: []` and a top-level `Finish` only. "
                            "Do not call tools on this turn. Provide the best answer supported by the current "
                            "evidence; do not use a runtime/tool error as proof that the video is unanswerable."
                        ),
                        history=forced_history,
                    )
                    try:
                        forced_raw_response, forced_elapsed_seconds = _generate_with_heartbeat(
                            messages_payload=forced_messages,
                            generation_payload=_generation_kwargs_with_deadlines(generation_kwargs),
                            iteration_idx=iteration,
                            phase=("forced_recovery_finish" if recovery_turn else "forced_final_retry"),
                        )
                    except GenerationTimeoutError as exc:
                        final_answer, success, finished = _abort_due_to_generation_timeout(
                            iteration_idx=iteration,
                            phase="forced_final_retry",
                            exc=exc,
                        )
                        break
                    runtime_session.record_timing(
                        "planner.llm_generate_seconds",
                        forced_elapsed_seconds,
                    )
                    remaining_wall_clock_seconds = _remaining_wall_clock_seconds()
                    if remaining_wall_clock_seconds is not None and remaining_wall_clock_seconds <= 0:
                        final_answer, success, finished = _abort_due_to_wall_clock_budget(
                            "Wall-clock deadline reached during the forced final-turn retry."
                        )
                        break
                    self._write_raw_trace(
                        depth=depth,
                        iteration=iteration,
                        goal=goal,
                        raw_response=f"[FORCED_FINAL_RETRY]\n{forced_raw_response}",
                    )
                    response = _normalize_llm_response_for_role(forced_raw_response)
                    response_for_log = _normalize_llm_response_for_log(
                        forced_raw_response,
                        enable_thinking=generation_kwargs.get("enable_thinking"),
                    )
                    forced_raw_response_text = agent.sanitize_response(forced_raw_response)
                    _log_llm_response_sanitization(
                        iteration_idx=iteration,
                        raw_response_text=forced_raw_response_text,
                        log_response_text=response_for_log,
                        effective_response_text=response,
                        phase_tag="[FORCED_FINAL_RETRY]",
                    )
                    _log_unexpected_think_when_disabled(
                        iteration_idx=iteration,
                        log_response_text=response_for_log,
                        enable_thinking=generation_kwargs.get("enable_thinking"),
                        phase_tag="[FORCED_FINAL_RETRY]",
                    )
                    logger.info(
                        "%s[%s][LLM_FULL_RESPONSE_BEGIN][Depth %d][Iter %d][FORCED_FINAL_RETRY]\n%s\n%s[%s][LLM_FULL_RESPONSE_END][Depth %d][Iter %d][FORCED_FINAL_RETRY]",
                        '  ' * depth,
                        role_label,
                        depth,
                        iteration,
                        response_for_log,
                        '  ' * depth,
                        role_label,
                        depth,
                        iteration,
                    )
                    structured_payload = agent._extract_json_payload(response)
                    parsed_evidence_slots = []
                    if evidence_first_mode:
                        parsed_evidence_slots = _extract_evidence_slots(structured_payload)
                    thought, plan, action_name, action_args = agent._parse_response(response)
                    pending_planner_evidence_slots = (
                        list(parsed_evidence_slots)
                        if evidence_first_mode
                        and parsed_evidence_slots
                        and str(action_name or "").lower() != "finish"
                        else []
                    )
                    recoverable_empty_evidence_turn = False
                    if str(action_name or "").lower() != "finish":
                        final_answer = "Error: finish_boundary_retry_returned_non_finish."
                        success = False
                        if depth == 0 and evidence_first_mode:
                            _activate_finish_recovery(
                                "forced Finish-only retry returned a non-Finish response",
                                targets=[
                                    "emit a top-level Finish with one selected listed option",
                                    "use the latest grounded evidence rather than repeating the prior action",
                                ],
                            )
                            finished = False
                        else:
                            finished = True
                        agent.trajectory.add_step(
                            thought=thought or "Final turn retry failed",
                            plan=plan,
                            action="Finish",
                            action_args={},
                            observation=final_answer,
                            is_recursive=False,
                        )
                        _append_observation_json([
                            {"tool": "System", "status": "error", "result": final_answer}
                        ])
                        if finished:
                            break
                        continue
                if deadline_reached_after_generation and str(action_name or "").lower() != "finish":
                    final_answer, success, finished = _abort_due_to_wall_clock_budget(
                        "Wall-clock deadline reached after a completed LLM generation turn, "
                        "and the preserved response did not finish the task."
                    )
                    break
                if settings.enable_llm_output_debug and debug_event is not None:
                    debug_event.update({
                        "parsed_plan": plan,
                        "parsed_action": action_name,
                        "parsed_action_args": action_args,
                        "thought_preview": thought or "",
                        "parse_success": bool(action_name) or recoverable_empty_evidence_turn,
                        "parse_mode": "evidence_checkpoint" if recoverable_empty_evidence_turn else "action",
                    })
                    agent.trajectory.add_debug_event(debug_event)

                iteration_phase = "protocol_validate"
                protocol_violation = _validate_structured_protocol(structured_payload)
                if protocol_violation:
                    consecutive_parse_failures += 1
                    if settings.enable_llm_output_debug:
                        agent.trajectory.add_debug_event({
                            "type": "protocol_violation",
                            "depth": depth,
                            "iteration": iteration,
                            "consecutive_failures": consecutive_parse_failures,
                            "message": protocol_violation,
                        })
                    agent.trajectory.add_step(
                        thought=thought or "Protocol violation",
                        plan=plan,
                        action=action_name or "Protocol_Error",
                        action_args=action_args if isinstance(action_args, dict) else {},
                        observation=protocol_violation,
                        is_recursive=False,
                    )
                    if action_name:
                        _append_assistant_turn(
                            thought,
                            plan,
                            action_name,
                            action_args if isinstance(action_args, dict) else {},
                        )
                    _append_observation_json([
                        {"tool": "System", "status": "error", "result": protocol_violation}
                    ])
                    if consecutive_parse_failures >= MAX_PARSE_FAILURES:
                        logger.error(
                            "Max consecutive protocol violations reached. "
                            "Sub-agent is not complying with the strict JSON contract. Aborting."
                        )
                        final_answer = (
                            "Failed: sub-agent could not follow the strict JSON protocol after "
                            f"{MAX_PARSE_FAILURES} attempts."
                        )
                        finished = True
                        break
                    retry_msg = _build_strict_json_retry_message(protocol_violation)
                    _append_history("user", retry_msg)
                    continue

                if not action_name and recoverable_empty_evidence_turn:
                    recoverable_empty_evidence_turns += 1
                    consecutive_parse_failures = 0
                    agent.trajectory.add_debug_event({
                        "type": "evidence_checkpoint",
                        "event_type": "EvidenceCheckpointEvent",
                        "depth": depth,
                        "iteration": iteration,
                        "checkpoint_count": recoverable_empty_evidence_turns,
                        "evidence_count": len(parsed_evidence_slots),
                        "message": (
                            "Accepted one non-terminal evidence checkpoint with empty Actions; "
                            "the planner must emit a top-level Finish on the next turn."
                        ),
                    })
                    checkpoint_payload = dict(structured_payload or {})
                    checkpoint_payload["Actions"] = []
                    if parsed_evidence_slots and not checkpoint_payload.get("Evidence"):
                        checkpoint_payload["Evidence"] = list(parsed_evidence_slots)
                    _append_history(
                        "assistant",
                        json.dumps(checkpoint_payload, ensure_ascii=False),
                    )
                    _append_observation_json([
                        {
                            "tool": "System",
                            "status": "success",
                            "result": (
                                "Evidence checkpoint recorded. This turn was non-terminal; "
                                "return strict JSON with Actions: [] and a top-level Finish now."
                            ),
                        }
                    ], commit_planner_evidence=True)
                    _append_history(
                        "user",
                        "Evidence checkpoint accepted. Do not call another tool unless evidence changed; "
                        "emit the required top-level Finish with the selected answer on this turn.",
                    )
                    continue
                
                if not action_name:
                    salvage_note = _extract_parse_failure_note(response, structured_payload)
                    consecutive_parse_failures += 1
                    looks_like_think_cutoff = (
                        "<think>" in (response or "")
                        and "</think>" not in (response or "")
                    )
                    lowered_response = str(response or "").lower()
                    looks_like_reasoning_only_cutoff = (
                        depth == 0
                        and not structured_payload
                        and (
                            "thinking process" in lowered_response
                            or "analysis:" in lowered_response
                            or re.search(r"(?m)^\s*\d+\.\s+\*\*", str(response or "")) is not None
                        )
                    )
                    if salvage_note:
                        missing_action_items = [
                            {
                                "tool": "Assistant_Note",
                                "status": "success",
                                "result": {
                                    "summary": salvage_note,
                                    "source": "assistant_non_action_output",
                                },
                            }
                        ]
                    else:
                        missing_action_items = []
                    if depth == 0:
                        missing_action_or_finish_msg = (
                            "Error: response did not include executable `Actions` or a top-level `Finish`."
                        )
                    else:
                        missing_action_or_finish_msg = (
                            "Error: sub-agent response did not include valid executable `Actions`."
                        )
                    missing_action_items.append({
                        "tool": "System",
                        "status": "error",
                        "result": missing_action_or_finish_msg,
                    })
                    _append_observation_json(missing_action_items)
                    if settings.enable_llm_output_debug:
                        agent.trajectory.add_debug_event({
                            "type": "parse_failure",
                            "depth": depth,
                            "iteration": iteration,
                            "consecutive_failures": consecutive_parse_failures,
                            "reason": "no_valid_action",
                            "think_cutoff_suspected": looks_like_think_cutoff,
                            "reasoning_only_cutoff_suspected": looks_like_reasoning_only_cutoff,
                        })
                    logger.warning(
                        f"No valid action parsed (failure {consecutive_parse_failures}/{MAX_PARSE_FAILURES})"
                    )
                    if consecutive_parse_failures >= MAX_PARSE_FAILURES:
                        logger.error(
                            "Max consecutive parse failures reached. "
                            "LLM is not producing valid Action format. Aborting."
                        )
                        final_answer = (
                            "Failed: LLM could not produce a valid action after "
                            f"{MAX_PARSE_FAILURES} attempts. Check model compatibility "
                            "and prompt format."
                        )
                        finished = True
                        break
                    # Concise retry hint without appending the full LLM response
                    if looks_like_think_cutoff or looks_like_reasoning_only_cutoff:
                        if mediation_mode:
                            retry_msg = _build_strict_json_retry_message(
                                "Your last output spent the turn on reasoning before the executable JSON."
                            )
                        elif evidence_first_mode:
                            retry_msg = _build_strict_json_retry_message(
                                "Your last output spent the turn on reasoning before producing executable JSON."
                            )
                        else:
                            retry_msg = (
                                "Your last output appears to have spent the turn on reasoning before producing Action. "
                                "Keep Thought to at most 2 short sentences and output Actions immediately. "
                                "Do NOT output long thinking prose before the JSON action.\n\n"
                                "Required JSON format:\n"
                                "{\"Thought\":\"...\",\"Plan\":\"...\",\"Actions\":[{\"tool\":\"Inspect_Frame\",\"params\":{\"timestamp\":\"12.5\",\"query\":\"inspect the delegated evidence target\"}}]}"
                            )
                        _append_history("user", retry_msg)
                    else:
                        if mediation_mode:
                            retry_msg = _build_strict_json_retry_message(
                                "Your response did not contain valid executable `Actions`."
                            )
                        elif evidence_first_mode:
                            retry_msg = (
                                "Your response did not contain valid Actions or a top-level Finish. "
                                "For this evidence-first task, either continue with an Evidence array plus retrieval/visual Actions, "
                                "or finish only with `Actions: []` and a top-level `Finish` after the chain is complete.\n\n"
                                "Continue template:\n"
                                "{\"Thought\":\"...\",\"Plan\":\"...\",\"Evidence\":[{\"label\":\"...\",\"value\":\"\",\"citation\":\"\",\"evidence_status\":\"missing\"}],"
                                "\"Actions\":[{\"tool\":\"ACTUAL_TOOL_NAME_HERE\",\"description\":\"find the next retrieval or visual capability needed for the missing evidence\",\"params\":{\"arg\":\"value\"}}]}\n\n"
                                "Finish template:\n"
                                "{\"Thought\":\"...\",\"Plan\":\"...\",\"Actions\":[],\"Finish\":{\"chain_complete\":true,\"completion_basis\":\"text->visual->answer\",\"answer\":\"...\"}}"
                            )
                        elif depth == 0:
                            retry_msg = (
                                "Your response did not match one of the two valid root templates because it was missing executable `Actions` or a top-level `Finish`. "
                                "You may include `<think>...</think>` before JSON, but the reply must end with one complete JSON object. "
                                "Use exactly one of these:\n\n"
                                "1. Continue with tools:\n"
                                "{\"Thought\":\"...\",\"Plan\":\"...\",\"Actions\":[{\"tool\":\"ACTUAL_TOOL_NAME_HERE\",\"description\":\"describe the next tool capability precisely\",\"params\":{\"arg\":\"value\"}}]}\n\n"
                                "2. Finish the task:\n"
                                "{\"Thought\":\"...\",\"Plan\":\"...\",\"Actions\":[],\"Finish\":{\"chain_complete\":true,\"completion_basis\":\"text->visual->answer\",\"answer\":\"...\"}}"
                            )
                        else:
                            retry_msg = (
                                "Your sub-agent response did not contain valid executable `Actions`. "
                                "Return exactly one strict JSON object with one tool call or multiple parallel tool calls.\n\n"
                                "{\"Thought\":\"...\",\"Plan\":\"...\",\"Actions\":[{\"tool\":\"Inspect_Frame\",\"params\":{\"timestamp\":\"12.5\",\"query\":\"inspect the delegated evidence target\"}}]}"
                            )
                        _append_history("user", retry_msg)
                    continue

                # Guard invalid placeholder action names BEFORE repetition logic.
                if action_name.lower() in _PLACEHOLDER_ACTION_NAMES:
                    logger.warning(f"Invalid action name '{action_name}', treating as parse failure")
                    consecutive_parse_failures += 1
                    if settings.enable_llm_output_debug:
                        agent.trajectory.add_debug_event({
                            "type": "parse_failure",
                            "depth": depth,
                            "iteration": iteration,
                            "consecutive_failures": consecutive_parse_failures,
                            "reason": "invalid_action_name",
                            "action_name": action_name,
                        })
                    if consecutive_parse_failures >= MAX_PARSE_FAILURES:
                        final_answer = "Failed: LLM produced invalid action names repeatedly."
                        finished = True
                        break
                    _append_history("user", (
                        f"'{action_name}' is not a valid action. Never output placeholder names. "
                        "Call a real executable tool name from the runtime instead of a template placeholder. "
                        "If you do not know the exact executable tool yet, describe the missing capability abstractly so the runtime/sub-agent can route it."
                    ))
                    continue

                # Reset parse failure counter only after action name is validated.
                consecutive_parse_failures = 0

                try:
                    action_args = _normalize_action_args(action_name, action_args)
                except ValueError as exc:
                    normalization_msg = (
                        f"Error: invalid parameters for {action_name}: {exc} "
                        "Use the tool schema exactly and retry with corrected arguments."
                    )
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=normalization_msg,
                        is_recursive=False,
                    )
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {"tool": "System", "status": "error", "result": normalization_msg}
                    ])
                    continue

                raw_params_declared = bool(action_args.get("__raw_params_declared__")) if isinstance(action_args, dict) else False
                raw_param_keys = action_args.get("__raw_param_keys__", []) if isinstance(action_args, dict) else []
                normalized_param_keys = [
                    key for key, value in (action_args.items() if isinstance(action_args, dict) else [])
                    if not str(key).startswith("__") and value not in (None, "", [], {})
                ]
                if (
                    action_name not in {"Finish", "Parallel"}
                    and raw_params_declared
                    and raw_param_keys
                    and not normalized_param_keys
                ):
                    normalization_msg = (
                        f"Error: invalid parameters for {action_name}: parsed parameter payload lost all executable "
                        f"arguments after normalization. Raw parameter keys were: {', '.join(map(str, raw_param_keys))}. "
                        "Retry with schema-compatible JSON keys and values."
                    )
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=normalization_msg,
                        is_recursive=False,
                    )
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {"tool": "System", "status": "error", "result": normalization_msg}
                    ])
                    continue
                if isinstance(action_args, dict):
                    action_args.pop("__raw_params_declared__", None)
                    action_args.pop("__raw_param_keys__", None)
                    format_violation_reason = str(action_args.pop("__format_violation__", "") or "").strip()
                else:
                    format_violation_reason = ""

                if format_violation_reason:
                    agent.trajectory.add_debug_event({
                        "type": "format_violation",
                        "depth": depth,
                        "iteration": iteration,
                        "reason": format_violation_reason,
                        "action": action_name,
                    })

                # Repetition circuit breaker: block identical action+args loops.
                # Finish is special: its generated Evidence array is a report,
                # not part of the answer candidate.  Including that array in
                # the signature made the same answer look new whenever the
                # verifier/Observer rewrote a ledger slot, so rejected Finish
                # calls escaped this guard until max_iterations.
                signature_args = action_args
                if action_name.lower() == "finish" and isinstance(action_args, dict):
                    raw_finish_answer = action_args.get("answer")
                    if raw_finish_answer in (None, ""):
                        raw_finish_answer = action_args.get("result", "")
                    if isinstance(raw_finish_answer, (dict, list, tuple)):
                        raw_finish_answer = json.dumps(
                            raw_finish_answer,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    finish_answer_text = " ".join(normalize_answer_format(raw_finish_answer).split()).strip()
                    finish_label_match = re.search(
                        r"\(([A-Z])\)", finish_answer_text.upper()
                    )
                    # For multiple-choice tasks, explanations may be rewritten
                    # after every rejection while the selected option remains
                    # the same.  Identify the candidate by option label so
                    # prose variation cannot evade the loop guard.  Free-form
                    # answers retain a whitespace-normalized identity.
                    finish_answer_identity = (
                        f"option:{finish_label_match.group(1)}"
                        if finish_label_match
                        else f"text:{finish_answer_text.lower()}"
                    )
                    signature_args = {
                        "answer_identity": finish_answer_identity,
                        # A rejected candidate may be reconsidered after the
                        # evidence ledger changes. Rewording Finish alone
                        # does not advance this runtime-owned revision.
                        "evidence_revision": evidence_revision,
                    }
                current_signature = _action_signature(action_name, signature_args)
                action_family = _action_family(action_name)
                # A rejected Finish can be rewritten with a different option
                # label on the next turn, which otherwise evades the exact
                # signature guard and burns the recovery budget.  Once a
                # Finish has been rejected at the current evidence revision,
                # force the next Finish-shaped response through the same
                # bounded evidence probe used for an exact duplicate.  A
                # grounded observation resets ``finish_rejection_streak``;
                # the pending Finish-only boundary is intentionally exempt.
                finish_recovery_probe_due = bool(
                    depth == 0
                    and evidence_first_mode
                    and str(action_name or "").strip().lower() == "finish"
                    and _finish_rejected_without_new_evidence(current_signature)
                    and not finish_recovery_finish_pending
                )
                finish_signature_was_blocked = current_signature in blocked_action_signatures
                if (
                    (finish_signature_was_blocked or finish_recovery_probe_due)
                    and depth == 0
                    and evidence_first_mode
                    and str(action_name or "").strip().lower() == "finish"
                ):
                    forced_probe = _build_forced_finish_recovery_probe(
                        forced_finish_recovery_probe_count + 1
                    )
                    if forced_probe is not None:
                        forced_finish_recovery_probe_count += 1
                        action_name, action_args = forced_probe
                        signature_args = action_args
                        current_signature = _action_signature(action_name, signature_args)
                        action_family = _action_family(action_name)
                        agent.trajectory.add_debug_event({
                            "type": "forced_finish_recovery_probe",
                            "event_type": "ForcedFinishRecoveryProbeEvent",
                            "depth": depth,
                            "iteration": iteration,
                            "attempt": forced_finish_recovery_probe_count,
                            "tool": action_name,
                            "params": dict(action_args),
                            "reason": (
                                "repeated Finish signature replaced by a deterministic evidence probe"
                                if finish_signature_was_blocked
                                else "rejected Finish candidate replaced before evidence revision"
                            ),
                        })
                if current_signature in blocked_action_signatures:
                    repeated_loop_violations += 1
                    signature_failures = _record_identical_action_failure(current_signature)
                    recovery_hint = _build_action_recovery_hint(action_name, action_args)
                    blocked_msg = (
                        "Error: this exact action signature is blocked due to prior repetition. "
                        "You MUST choose a different tool or materially different arguments. "
                        "If stuck, reformulate the capability and choose a different executable action."
                    )
                    if evidence_first_mode and action_family in {"discovery", "compute"}:
                        blocked_msg = (
                            "Error: this exact action signature is blocked due to prior repetition. "
                            "Switch to retrieval or visual probing on nearby evidence windows instead of "
                            "repeating registry search or fallback computation."
                        )
                    if recovery_hint:
                        blocked_msg = f"{blocked_msg} {recovery_hint}".strip()
                    if settings.enable_llm_output_debug:
                        agent.trajectory.add_debug_event({
                            "type": "policy_violation",
                            "depth": depth,
                            "iteration": iteration,
                            "reason": "blocked_action_signature_reused",
                            "action": action_name,
                            "count": repeated_loop_violations,
                        })
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=blocked_msg,
                        is_recursive=False,
                    )
                    if depth > 0 and signature_failures >= MAX_SUBAGENT_TOOLSEARCH_FAILURES:
                        final_answer = (
                            "Sub-agent fallback to planner: repeated a blocked tool call without changing "
                            "tool name or parameters within the allowed retry budget."
                        )
                        success = False
                        finished = True
                        break
                    if (
                        depth == 0
                        and action_name.lower() == "finish"
                        and _finish_rejected_without_new_evidence(current_signature)
                    ):
                        _activate_finish_recovery(
                            "the same Finish candidate was rejected without an evidence revision",
                            targets=[recovery_hint],
                        )
                        if finish_recovery_violations < max_finish_recovery_retries:
                            finish_recovery_violations += 1
                            recovery_hint = _build_action_recovery_hint(action_name, action_args)
                            recovery_msg = (
                                "Finish was rejected without new evidence. "
                                "This is a recovery retry, not an unanswerable video. "
                                "The next Planner response MUST contain a concrete non-Finish "
                                "evidence action. "
                                f"{recovery_hint}"
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "Evidence_Verifier", "status": "rejected", "result": recovery_msg}
                            ])
                            _append_history("user", recovery_msg)
                            continue
                        if finish_recovery_turns_used < max_finish_recovery_rounds:
                            finish_recovery_violations = 0
                            recovery_msg = (
                                "The same Finish candidate was rejected repeatedly without a new evidence "
                                "revision. Entering bounded alternate-direction recovery; execute a concrete "
                                "non-Finish evidence action before trying Finish again."
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "Evidence_Verifier", "status": "rejected", "result": recovery_msg}
                            ])
                            _append_history("user", recovery_msg)
                            continue
                        terminal_reason = (
                            "Failed: finish_recovery_budget_exhausted_after_repeated_finish_rejection."
                        )
                        agent.trajectory.add_debug_event({
                            "type": "finish_recovery_budget_exhausted",
                            "event_type": "FinishRecoveryBudgetExhaustedEvent",
                            "depth": depth,
                            "iteration": iteration,
                            "issue": "repeated_finish_without_new_evidence",
                            "reason": terminal_reason,
                            "finish_signature": current_signature,
                            "finish_rejection_count": finish_rejection_counts.get(current_signature, 0),
                        })
                        final_answer = terminal_reason
                        success = False
                        finished = True
                        break
                    if depth == 0 and repeated_loop_violations >= 2:
                        if (
                            evidence_first_mode
                            and finish_recovery_turns_used < max_finish_recovery_rounds
                        ):
                            recovery_msg = (
                                "Planner action loop detected. Do not repeat the blocked action. "
                                "Use the active recovery directive and switch tool family, modality, "
                                "or temporal window before attempting Finish."
                            )
                            _activate_finish_recovery(
                                "Planner repeated blocked actions",
                                targets=[recovery_hint, recovery_msg],
                            )
                            repeated_loop_violations = 0
                            last_action_signature = ""
                            repeated_action_count = 0
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "System", "status": "error", "result": recovery_msg}
                            ])
                            _append_history("user", recovery_msg)
                            continue
                        final_answer = (
                            "Failed: planner_repeated_blocked_action_loop."
                            if evidence_first_mode
                            else "Failed: planner kept reusing a blocked action signature. "
                            "Early stop triggered to avoid wasting iterations in a policy dead loop."
                        )
                        success = False
                        finished = True
                        break
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {"tool": "System", "status": "error", "result": blocked_msg}
                    ])
                    if recovery_hint:
                        _append_history("user", recovery_hint)
                    continue

                if current_signature == last_action_signature:
                    repeated_action_count += 1
                else:
                    repeated_action_count = 1
                    last_action_signature = current_signature

                if repeated_action_count >= MAX_REPEATED_ACTIONS:
                    blocked_action_signatures.add(current_signature)
                    repeated_loop_violations += 1
                    signature_failures = _record_identical_action_failure(current_signature)
                    recovery_hint = _build_action_recovery_hint(action_name, action_args)
                    repetition_msg = (
                        f"Error: repeated same action {repeated_action_count} times without progress: "
                        f"{_format_action_for_history(action_name, action_args)}. "
                        "You MUST choose a different action or different arguments based on the latest observation. "
                        "Do not repeat template code or placeholder answers."
                    )
                    if evidence_first_mode:
                        if action_family == "discovery":
                            repetition_msg += (
                                " Stop using registry/web discovery as evidence retrieval. "
                                "Switch to local retrieval or direct probing instead of repeating discovery."
                            )
                        elif action_family == "compute":
                            repetition_msg += (
                                " If you keep using computation, materially change the inputs and ground them in retrieved evidence."
                            )
                        else:
                            repetition_msg += (
                                " Continue from adjacent segments or nearby time windows rather than restarting broad search."
                            )
                    if goal_numbers:
                        repetition_msg += (
                            f" For Python code, ground computation in query numbers: {', '.join(goal_numbers)}."
                        )
                    if recovery_hint:
                        repetition_msg = f"{repetition_msg} {recovery_hint}".strip()
                    if settings.enable_llm_output_debug:
                        agent.trajectory.add_debug_event({
                            "type": "policy_violation",
                            "depth": depth,
                            "iteration": iteration,
                            "reason": "repeated_action_loop",
                            "action": action_name,
                            "count": repeated_action_count,
                        })
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=repetition_msg,
                        is_recursive=False,
                    )
                    if (
                        depth == 0
                        and action_name.lower() == "finish"
                        and _finish_rejected_without_new_evidence(current_signature)
                    ):
                        _activate_finish_recovery(
                            "the same Finish candidate was repeated after rejection",
                            targets=[recovery_hint],
                        )
                        if finish_recovery_violations < max_finish_recovery_retries:
                            finish_recovery_violations += 1
                            recovery_hint = _build_action_recovery_hint(action_name, action_args)
                            recovery_msg = (
                                "Finish was rejected without new evidence. "
                                "This is a recovery retry, not an unanswerable video. "
                                "The next Planner response MUST contain a concrete non-Finish "
                                "evidence action. "
                                f"{recovery_hint}"
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "Evidence_Verifier", "status": "rejected", "result": recovery_msg}
                            ])
                            _append_history("user", recovery_msg)
                            continue
                        if finish_recovery_turns_used < max_finish_recovery_rounds:
                            finish_recovery_violations = 0
                            recovery_msg = (
                                "The same Finish candidate was repeated after rejection. Use the active "
                                "recovery directive: execute one materially different concrete evidence "
                                "action before submitting Finish again."
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "Evidence_Verifier", "status": "rejected", "result": recovery_msg}
                            ])
                            _append_history("user", recovery_msg)
                            continue
                        terminal_reason = (
                            "Failed: finish_recovery_budget_exhausted_after_repeated_finish_rejection."
                        )
                        agent.trajectory.add_debug_event({
                            "type": "finish_recovery_budget_exhausted",
                            "event_type": "FinishRecoveryBudgetExhaustedEvent",
                            "depth": depth,
                            "iteration": iteration,
                            "issue": "repeated_finish_without_new_evidence",
                            "reason": terminal_reason,
                            "finish_signature": current_signature,
                            "finish_rejection_count": finish_rejection_counts.get(current_signature, 0),
                        })
                        final_answer = terminal_reason
                        success = False
                        finished = True
                        break
                    if depth > 0 and signature_failures >= MAX_SUBAGENT_TOOLSEARCH_FAILURES:
                        final_answer = (
                            "Sub-agent fallback to planner: repeated the same tool call without a material "
                            "parameter rewrite three times."
                        )
                        success = False
                        finished = True
                        break
                    if depth == 0 and repeated_loop_violations >= 2:
                        if (
                            evidence_first_mode
                            and finish_recovery_turns_used < max_finish_recovery_rounds
                        ):
                            recovery_msg = (
                                "Planner repeated a policy-blocked action. Do not retry that signature. "
                                "Switch direction using the active recovery directive and acquire a new "
                                "concrete evidence observation before Finish."
                            )
                            _activate_finish_recovery(
                                "Planner repeated an action-policy loop",
                                targets=[recovery_hint, recovery_msg],
                            )
                            repeated_loop_violations = 0
                            last_action_signature = ""
                            repeated_action_count = 0
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "System", "status": "error", "result": recovery_msg}
                            ])
                            _append_history("user", recovery_msg)
                            continue
                        final_answer = (
                            "Failed: planner_repeated_action_policy_loop."
                            if evidence_first_mode
                            else "Failed: planner stuck in repeated-action policy loop. "
                            "Model did not produce a materially different action after multiple enforced retries."
                        )
                        success = False
                        finished = True
                        break
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {"tool": "System", "status": "error", "result": repetition_msg}
                    ])
                    if recovery_hint:
                        _append_history("user", recovery_hint)
                    continue
                
                if action_name == "Web_Search" and _active_local_video_scope_present() and not _goal_requests_external_web():
                    policy_msg = (
                        "Error: Web_Search is blocked for local-video reasoning while an active video scope is loaded. "
                        "Use local retrieval, direct evidence, or verification tools instead."
                    )
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=policy_msg,
                        is_recursive=False,
                    )
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {"tool": "System", "status": "error", "result": policy_msg}
                    ])
                    continue

                redundant_visual_msg = _detect_redundant_visual_reprobe(
                    action_name,
                    action_args if isinstance(action_args, dict) else {},
                )
                if redundant_visual_msg:
                    blocked_action_signatures.add(current_signature)
                    repeated_loop_violations += 1
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=redundant_visual_msg,
                        is_recursive=False,
                    )
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {"tool": "System", "status": "error", "result": redundant_visual_msg}
                    ])
                    continue

                # Handle finish - require explicit answer parameter
                if action_name.lower() == "finish":
                    iteration_phase = "finish_boundary"
                    if depth > 0:
                        policy_msg = (
                            "Error: only the root planner may emit Finish. "
                            "Sub-agents must execute a concrete tool instead; successful tool executions return to the parent planner automatically. "
                            "A top-level `Result` note is optional, but it never replaces executable `Actions`."
                        )
                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=policy_msg,
                            is_recursive=False,
                        )
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {"tool": "System", "status": "error", "result": policy_msg}
                        ])
                        continue

                    answer = action_args.get("answer", "")
                    if not answer:
                        answer = action_args.get("result", "")
                    if answer is None:
                        answer = ""
                    if not isinstance(answer, str):
                        answer = json.dumps(answer, ensure_ascii=False)
                    completion_basis = str(action_args.get("completion_basis", "") or "").strip()

                    if _is_placeholder_finish_answer(answer):
                        placeholder_msg = (
                            "Error: Finish answer looks like a template/example placeholder. "
                            "Use concrete values computed/retrieved in this run."
                        )
                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=placeholder_msg,
                            is_recursive=False
                        )
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {"tool": "System", "status": "error", "result": placeholder_msg}
                        ])
                        continue

                    if depth == 0:
                        chain_complete = _coerce_explicit_bool(action_args.get("chain_complete"))
                        if chain_complete is None:
                            policy_msg = (
                                "Error: finish_payload_invalid. Root Finish must include "
                                "`chain_complete: true|false` and a non-empty `answer` or `result`."
                            )
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=policy_msg,
                                is_recursive=False,
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "System", "status": "error", "result": policy_msg}
                            ])
                            continue
                        if not chain_complete:
                            policy_msg = (
                                "Error: planner_declared_incomplete. Root planner emitted Finish with "
                                "`chain_complete=false`; continue reasoning instead of finishing."
                            )
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=policy_msg,
                                is_recursive=False,
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "System", "status": "error", "result": policy_msg}
                            ])
                            continue
                    if not answer or not answer.strip():
                        consecutive_finish_without_answer += 1
                        logger.warning(
                            f"{'  ' * depth}[Depth {depth}] Finish called without answer/result parameter "
                            f"(attempt {consecutive_finish_without_answer}/3)"
                        )
                        
                        # Abort if LLM repeatedly fails to provide answer
                        if consecutive_finish_without_answer >= 3:
                            logger.error(
                                f"{'  ' * depth}[Depth {depth}] LLM failed to provide answer parameter "
                                "after 3 attempts. Aborting with partial result."
                            )
                            final_answer = (
                                "Error: finish_payload_invalid. The agent repeatedly attempted to finish "
                                "without providing a valid non-empty `answer`/`result`. "
                                f"Last thought: {thought}"
                            )
                            observation = final_answer
                            finished = True
                            success = False
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=observation,
                                is_recursive=False
                            )
                            break

                        _append_history("user", (
                            "Error: finish_payload_invalid. Root Finish must be called with a non-empty "
                            "`answer` or `result` parameter. Provide it in the top-level JSON Finish field.\n"
                            "Please provide the complete answer to the user's question.\n\n"
                            "IMPORTANT: Use this exact root format:\n"
                            "{\"Thought\":\"...\",\"Plan\":\"...\",\"Actions\":[],\"Finish\":{\"chain_complete\":true,\"completion_basis\":\"text->visual->answer\",\"answer\":\"Your complete answer here\"}}"
                        ))
                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation="Error: Missing required top-level Finish.answer/result parameter",
                            is_recursive=False
                        )
                        continue

                    if (
                        depth == 0
                        and evidence_first_mode
                        and _is_find_needle_goal(goal)
                        and needle_candidate_windows
                        and not needle_candidate_completed
                    ):
                        pending_candidates = _uncompleted_needle_candidates()
                        if pending_candidates:
                            next_candidate = pending_candidates[0]
                            candidate_window_text = (
                                f"{next_candidate.get('segment_ref')} "
                                f"({next_candidate.get('start_time')}-{next_candidate.get('end_time')} seconds)"
                            )
                            recovery_msg = (
                                "Finish was held because the latest Needle retrieval returned an unverified "
                                f"candidate window {candidate_window_text}. Directly execute Inspect_Frame on "
                                "that candidate at density <= 1 second/frame, then reassess the answer. "
                                "The candidate is a temporal locator only, not answer evidence."
                            )
                            finish_rejection_counts[current_signature] = (
                                int(finish_rejection_counts.get(current_signature, 0)) + 1
                            )
                            finish_rejection_revisions[current_signature] = evidence_revision
                            finish_rejection_reasons[current_signature] = "needle_candidate_probe_required"
                            _activate_finish_recovery(
                                "Needle Finish was held until the retrieved candidate window is directly inspected",
                                targets=[candidate_window_text],
                            )
                            agent.trajectory.add_debug_event({
                                "type": "needle_candidate_probe_required",
                                "event_type": "NeedleCandidateProbeRequiredEvent",
                                "candidate_window": dict(next_candidate),
                                "uses_annotation_time": False,
                                "uses_answer_label": False,
                            })
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=recovery_msg,
                                is_recursive=False,
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "Evidence_Verifier", "status": "rejected", "result": recovery_msg}
                            ])
                            _append_history("user", recovery_msg)
                            continue

                    if (
                        depth == 0
                        and _finish_rejected_without_new_evidence(current_signature)
                    ):
                        _activate_finish_recovery(
                            "Finish was rejected without a new evidence revision",
                            targets=[_build_action_recovery_hint(action_name, action_args)],
                        )
                        if finish_recovery_violations < max_finish_recovery_retries:
                            finish_recovery_violations += 1
                            recovery_hint = _build_action_recovery_hint(action_name, action_args)
                            recovery_retry_msg = (
                                "Finish was rejected without new evidence. "
                                "This is a recovery retry, not an unanswerable video. "
                                "On the next turn emit a concrete non-Finish evidence action. "
                                f"{recovery_hint}"
                            )
                            agent.trajectory.add_debug_event({
                                "type": "finish_recovery_retry",
                                "event_type": "FinishRecoveryRetryEvent",
                                "depth": depth,
                                "iteration": iteration,
                                "finish_signature": current_signature,
                                "retry_index": finish_recovery_violations,
                                "reason": recovery_retry_msg,
                            })
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=recovery_retry_msg,
                                is_recursive=False,
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "Evidence_Verifier", "status": "rejected", "result": recovery_retry_msg}
                            ])
                            _append_history("user", recovery_retry_msg)
                            continue
                        if finish_recovery_turns_used < max_finish_recovery_rounds:
                            finish_recovery_violations = 0
                            recovery_msg = (
                                "Finish recovery is still available. Execute one concrete evidence action "
                                "with a materially changed route/window before submitting any Finish candidate."
                            )
                            agent.trajectory.add_debug_event({
                                "type": "finish_recovery_retry",
                                "event_type": "FinishRecoveryRetryEvent",
                                "depth": depth,
                                "iteration": iteration,
                                "finish_signature": current_signature,
                                "retry_index": finish_recovery_violations,
                                "reason": recovery_msg,
                            })
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=recovery_msg,
                                is_recursive=False,
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "Evidence_Verifier", "status": "rejected", "result": recovery_msg}
                            ])
                            _append_history("user", recovery_msg)
                            continue
                        recovery_msg = (
                            "Failed: finish_recovery_budget_exhausted_after_repeated_finish_rejection."
                        )
                        finish_recovery_violations += 1
                        agent.trajectory.add_debug_event({
                            "type": "finish_recovery_budget_exhausted",
                            "event_type": "FinishRecoveryBudgetExhaustedEvent",
                            "depth": depth,
                            "iteration": iteration,
                            "issue": "finish_without_new_evidence",
                            "reason": recovery_msg,
                            "finish_signature": current_signature,
                            "finish_rejection_count": finish_rejection_counts.get(current_signature, 0),
                            "rejected_evidence_revision": finish_rejection_revisions.get(current_signature),
                            "current_evidence_revision": evidence_revision,
                            "rejection_reason": finish_rejection_reasons.get(current_signature, "verifier"),
                        })
                        final_answer = recovery_msg
                        success = False
                        finished = True
                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=recovery_msg,
                            is_recursive=False,
                        )
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {"tool": "Evidence_Verifier", "status": "rejected", "result": recovery_msg}
                        ])
                        break

                    if depth == 0:
                        verification = None
                        deterministic_payload: Dict[str, Any] = {
                            "can_finish": True,
                            "claim_coverage": None,
                            "direct_grounding": None,
                            "contradiction_score": None,
                            "cross_modal_support": None,
                            "missing_information": [],
                            "unresolved_claims": [],
                            "claim_contract": {},
                            "feedback": "",
                            "verification_source": "llm_reflector_only",
                        }
                        if evidence_verifier is not None:
                            verification = evidence_verifier.verify_answerability(
                                evidence_slots=evidence_slots,
                                steps=agent.trajectory.steps,
                                answer=answer,
                                completion_basis=completion_basis,
                                query=goal,
                            )
                            deterministic_payload = verification.model_dump()
                            deterministic_payload["verification_source"] = "deterministic_safety_gate"

                        reflector_evidence_slots = [
                            slot for slot in evidence_slots
                            if _slot_is_active(slot)
                        ]

                        reflector_deadline = time.perf_counter() + 90.0
                        if self.wall_clock_deadline_monotonic is not None:
                            reflector_deadline = min(reflector_deadline, self.wall_clock_deadline_monotonic)
                        if "reflector" in self.disabled_agent_roles:
                            reflector_result = {
                                "role": "reflector",
                                "status": "disabled",
                                "schema_valid": False,
                                "payload": {},
                                "raw_response": "",
                                "error": "",
                                "event": {
                                    "type": "agent_role_skipped",
                                    "event_type": "ReflectorAgentTurnEvent",
                                    "agent_role": "reflector",
                                    "status": "disabled",
                                    "schema_valid": False,
                                    "reason": "online_role_ablation",
                                },
                            }
                        else:
                            reflector_result = self.role_agents.reflect(
                                goal=goal,
                                answer=answer,
                                completion_basis=completion_basis,
                                evidence_slots=reflector_evidence_slots,
                                trajectory_steps=[step.to_dict() for step in agent.trajectory.steps],
                                deterministic_verification=deterministic_payload,
                                deadline_monotonic=reflector_deadline,
                                recovery_directive=active_recovery_system_directive,
                            )
                        reflector_event = dict(reflector_result.get("event") or {})
                        reflector_event.update({
                            "depth": depth,
                            "iteration": iteration,
                            "proposed_answer": answer,
                        })
                        runtime_session.record_timing(
                            "reflector.llm_generate_seconds",
                            float(reflector_event.get("elapsed_seconds") or 0.0),
                        )
                        agent.trajectory.add_debug_event(reflector_event)
                        reflector_payload = reflector_result.get("payload") or {}
                        reflector_schema_valid = bool(reflector_result.get("schema_valid"))
                        reflector_can_finish = (
                            bool(reflector_payload.get("can_finish"))
                            if reflector_schema_valid
                            else False
                        )
                        reflector_disabled = str(
                            reflector_result.get("status") or ""
                        ).strip().lower() == "disabled"
                        verification_payload = dict(deterministic_payload)
                        verification_payload.update({
                            "type": "verifier_decision",
                            "event_type": "VerifierDecisionEvent",
                            "depth": depth,
                            "iteration": iteration,
                            "reflector_agent_status": reflector_result.get("status"),
                            "reflector_agent_schema_valid": reflector_schema_valid,
                            "reflector_agent_decision": reflector_payload.get("decision"),
                            "reflector_agent_can_finish": reflector_payload.get("can_finish"),
                            "reflector_agent_payload": reflector_payload,
                        })
                        logger.info(
                            "%s[Reflector][Depth %d][Iter %d] LLM decision=%s can_finish=%s schema_valid=%s safety_can_finish=%s",
                            '  ' * depth,
                            depth,
                            iteration,
                            reflector_payload.get("decision"),
                            reflector_payload.get("can_finish"),
                            reflector_schema_valid,
                            deterministic_payload.get("can_finish"),
                        )
                        agent.trajectory.add_debug_event(verification_payload)
                        deterministic_block = verification is not None and not verification.can_finish
                        # A live Reflector turn is part of the termination
                        # contract. Invalid JSON/provider failure cannot
                        # silently authorize Finish; only an explicit online
                        # ablation (status=disabled) removes this role.
                        reflector_block = (
                            not reflector_disabled
                            and (not reflector_schema_valid or not reflector_can_finish)
                        )
                        if deterministic_block or reflector_block:
                            terminal_issue_keys: List[str] = []
                            if verification is not None and bool(
                                getattr(verification, "premise_mismatch", False)
                            ):
                                terminal_issue_keys.append("premise_mismatch")
                            if verification is not None and bool(
                                getattr(verification, "option_mismatch", False)
                            ):
                                terminal_issue_keys.append("option_mismatch")
                            if verification is not None and any(
                                "outside the listed options" in str(item)
                                for item in (verification.unresolved_claims or [])
                            ):
                                terminal_issue_keys.append("unlisted_action_value")
                            if verification is not None and float(
                                getattr(verification, "contradiction_score", 0.0) or 0.0
                            ) > 0.2:
                                terminal_issue_keys.append("contradictory_evidence")
                            for issue_key in terminal_issue_keys:
                                verifier_issue_counts[issue_key] = (
                                    int(verifier_issue_counts.get(issue_key, 0)) + 1
                                )
                                verifier_issue_revisions.setdefault(issue_key, set()).add(
                                    evidence_revision
                                )
                            exhausted_issue = next(
                                (
                                    issue_key
                                    for issue_key in terminal_issue_keys
                                    if (
                                        verifier_issue_counts.get(issue_key, 0) >= 2
                                        and len(verifier_issue_revisions.get(issue_key, set())) >= 2
                                    )
                                ),
                                None,
                            )
                            if exhausted_issue is not None:
                                terminal_reason = (
                                    "Error: evidence_unanswerable ("
                                    f"{exhausted_issue}; repeated direct evidence failure)"
                                )
                                logger.error(
                                    "%s[Depth %d] Terminating unresolved evidence contract: %s",
                                    "  " * depth,
                                    depth,
                                    terminal_reason,
                                )
                                agent.trajectory.add_debug_event({
                                    "type": "terminal_evidence_issue",
                                    "event_type": "EvidenceContractUnanswerableEvent",
                                    "depth": depth,
                                    "iteration": iteration,
                                    "issue": exhausted_issue,
                                    "reason": terminal_reason,
                                    "verification": verification_payload,
                                })
                                agent.trajectory.add_step(
                                    thought=thought,
                                    plan=plan,
                                    action=action_name,
                                    action_args=action_args,
                                    observation=terminal_reason,
                                    is_recursive=False,
                                )
                                final_answer = terminal_reason
                                success = False
                                finished = True
                                break
                            # Any role rejection is a real Finish rejection.
                            # Previously Reflector-only rejection was not
                            # recorded, so the same candidate could reach the
                            # global max-iteration fallback forever.
                            finish_rejection_counts[current_signature] = (
                                int(finish_rejection_counts.get(current_signature, 0)) + 1
                            )
                            finish_rejection_revisions[current_signature] = evidence_revision
                            finish_rejection_reasons[current_signature] = (
                                "deterministic_verifier"
                                if deterministic_block
                                else (
                                    "reflector_schema"
                                    if not reflector_schema_valid
                                    else "reflector"
                                )
                            )
                            if deterministic_block and verification is not None:
                                finish_recovery_violations = 0
                            if deterministic_block and verification is not None:
                                feedback = verification.feedback
                            elif reflector_schema_valid:
                                feedback = str(
                                    reflector_payload.get("evidence_assessment")
                                    or "Reflector requested more evidence before Finish."
                                )
                            else:
                                feedback = (
                                    "Reflector did not return a valid structured judgment; "
                                    "acquire one concrete cited observation and resubmit Finish only after the role recovers."
                                )
                            missing = list(reflector_payload.get("missing_information") or [])
                            unresolved = list(reflector_payload.get("unresolved_claims") or [])
                            if missing or unresolved:
                                feedback = (
                                    f"{feedback} Missing information: {missing}. "
                                    f"Unresolved claims: {unresolved}."
                                )
                            if deterministic_block or reflector_block:
                                recovery_targets = list(dict.fromkeys(
                                    [str(item).strip() for item in (missing + unresolved) if str(item).strip()]
                                ))
                                _activate_finish_recovery(
                                    (
                                        "Verifier/Reflector rejected Finish; repeated judgments require "
                                        "an alternate evidence direction"
                                    ),
                                    targets=(recovery_targets or terminal_issue_keys or [feedback]),
                                )
                                recovery_contract = (
                                    " [Finish Recovery Contract] Finish is blocked by the evidence judgment "
                                    "(deterministic verifier or Reflector). On the next planner turn, emit a concrete non-Finish executable action "
                                    "that directly addresses one of these targets: "
                                    f"{recovery_targets or ['the verifier feedback above']}. "
                                    "Do not resubmit the same answer candidate without new tool evidence. "
                                    "If a targeted probe confirms that the premise or all listed options do not "
                                    "match the video, do not map to the nearest option; let the runtime issue a "
                                    "controlled evidence_unanswerable outcome."
                                )
                                feedback = f"{feedback}{recovery_contract}"
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=feedback,
                                is_recursive=False,
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([{
                                "tool": "Evidence_Verifier",
                                "status": "rejected",
                                "result": verification_payload,
                            }])
                            continue

                    # Successfully got answer - reset counter
                    consecutive_finish_without_answer = 0
                    _clear_identical_action_failure(current_signature)
                    final_answer = answer
                    observation = final_answer
                    finished = True
                    success = True
                    if depth == 0:
                        logger.info(
                            "%s[Depth %d] Planner finish accepted: chain_complete=%s completion_basis=%s answer_preview=%s",
                            '  ' * depth,
                            depth,
                            True,
                            completion_basis or "[empty]",
                            _truncate_text(answer, 240),
                        )
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=observation,
                        is_recursive=False
                    )
                    _update_progress_state(action_name, observation)
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {
                            "tool": action_name,
                            "status": "success",
                            "result": {
                                "result": observation,
                                "chain_complete": action_args.get("chain_complete"),
                                "completion_basis": completion_basis,
                                "evidence_slots": list(evidence_slots),
                            },
                        }
                    ])
                    continue
                
                iteration_phase = "tool_routing"
                if action_name == "Tool_Search":
                    consecutive_parse_failures += 1
                    logger.warning("%s emitted Tool_Search directly; Tool_Search is automatic.", role_label)
                    _append_history("user", (
                        "Do not output Tool_Search. Tool_Search is automatic engine behavior. "
                        "Choose the target tool name(s) and fill parameters; the engine will search/validate automatically."
                    ))
                    continue

                # Handle implicit parallel batches
                if action_name == "Parallel":
                    batch = action_args.get("batch", [])
                    parallel_items: List[Dict[str, Any]] = []
                    delegated_sub_trajectories: List[Trajectory] = []
                    any_recursive = depth == 0
                    # Filter out placeholder action names from Parallel batch.
                    batch = [
                        c for c in batch
                        if (c.get("name") or "").lower() not in _PLACEHOLDER_ACTION_NAMES
                    ]
                    if not batch:
                        consecutive_parse_failures += 1
                        logger.warning("All Parallel batch actions were placeholder names.")
                        _append_history("user", (
                            "All actions in your batch were placeholder names. "
                            "Call concrete executable tools, or rewrite the capability into valid executable actions. "
                            "If you still need routing help, describe the missing capability abstractly so the runtime/sub-agent can route it."
                        ))
                        continue
                    if any(str((c.get("name") or "")).strip() == "Tool_Search" for c in batch):
                        consecutive_parse_failures += 1
                        logger.warning("%s emitted Tool_Search inside a Parallel batch; Tool_Search is automatic.", role_label)
                        _append_history("user", (
                            "Do not output Tool_Search. Tool_Search is automatic engine behavior. "
                            "Output the candidate tool name(s) you want and fill schema-compatible parameters."
                        ))
                        continue

                    if depth == 0:
                        if ablation_direct_toolsearch_mode:
                            parallel_slots: List[Optional[Dict[str, Any]]] = [None] * len(batch)
                            concrete_parallel_requests: List[Dict[str, Any]] = []
                            selected_calls = list(enumerate(batch))
                            if (ablation_single_tool_only or single_action_per_step_mode) and len(selected_calls) > 1:
                                selected_calls = selected_calls[:1]
                                for skipped_idx, skipped_call in enumerate(batch[1:], start=1):
                                    skipped_name = str(skipped_call.get("name", "")).strip() or f"planner_action_{skipped_idx + 1}"
                                    parallel_slots[skipped_idx] = {
                                        "tool": skipped_name,
                                        "status": "skipped",
                                        "payload": {
                                            "status": "skipped",
                                            "result": (
                                                "Ablation mode executed only the first planner action in this step."
                                            ),
                                            "planner_action_index": skipped_idx + 1,
                                            "planner_action_name": skipped_name,
                                            "source": "ablation_toolsearch_direct",
                                        },
                                    }

                            for idx, call in selected_calls:
                                name = call.get("name")
                                args = call.get("args", {})
                                batch_desc = args.pop("__description__", "") if isinstance(args, dict) else ""
                                normalized_name = str(name or "").strip()
                                normalized_args = args if isinstance(args, dict) else {}
                                resolution = _resolve_root_action_via_toolsearch_top1(
                                    abstract_tool=normalized_name,
                                    abstract_args=normalized_args,
                                    description_text=batch_desc or thought or plan,
                                )
                                if resolution.get("resolved"):
                                    concrete_parallel_requests.append({
                                        "index": idx,
                                        "tool_name": str(resolution.get("tool_name", normalized_name)).strip(),
                                        "tool_args": (
                                            resolution.get("params", {})
                                            if isinstance(resolution.get("params"), dict)
                                            else {}
                                        ),
                                        "planner_action_name": normalized_name,
                                    })
                                    continue
                                failure_payload = dict(resolution.get("observation", {}) or {})
                                failure_payload["abstract_tool"] = normalized_name
                                parallel_slots[idx] = {
                                    "tool": "Tool_Search",
                                    "status": str(failure_payload.get("status", "no_valid_result")),
                                    "payload": failure_payload,
                                }

                            for request in _execute_parallel_concrete_tool_calls(concrete_parallel_requests):
                                idx = int(request.get("index", 0))
                                tool_name = str(request.get("tool_name", "")).strip()
                                planner_action_name = str(
                                    request.get("planner_action_name", tool_name)
                                ).strip() or tool_name
                                prepared_call = request.get("prepared_call", {})
                                if request.get("error") is not None:
                                    error_text = str(request.get("error")).strip() or "tool execution failed"
                                    error_observation = (
                                        f"Error executing {tool_name}: {error_text}. "
                                        "Planner should try another relevant tool or adjust the probe window."
                                    )
                                    error_payload = {
                                        "status": "error",
                                        "result": error_observation,
                                        "resolved_tool": tool_name,
                                        "abstract_tool": planner_action_name,
                                        "source": "tool",
                                    }
                                    _remember_execution(
                                        tool_name=tool_name,
                                        tool_args=(
                                            request.get("tool_args", {})
                                            if isinstance(request.get("tool_args", {}), dict)
                                            else {}
                                        ),
                                        raw_result=error_payload,
                                        observation_text=error_observation,
                                        observation_status="error",
                                        source="tool",
                                        abstract_tool=planner_action_name,
                                        is_recursive=False,
                                    )
                                    parallel_slots[idx] = {
                                        "tool": tool_name,
                                        "status": "error",
                                        "payload": error_payload,
                                    }
                                    continue

                                result, _, obs_status = _finalize_prepared_concrete_tool_call(
                                    prepared_call,
                                    request.get("result"),
                                    abstract_tool=planner_action_name,
                                )
                                if obs_status == "success":
                                    success = True
                                    root_effective_tool_executed = True
                                parallel_slots[idx] = {
                                    "tool": tool_name,
                                    "status": obs_status,
                                    "payload": result,
                                }

                            parallel_items = [item for item in parallel_slots if isinstance(item, dict)]
                            observation, parallel_status, parallel_payload = build_parallel_observation(parallel_items)
                            if any(str(item.get("status", "")).lower() == "success" for item in parallel_items):
                                _clear_identical_action_failure(current_signature)
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=observation,
                                observation_items=parallel_payload.get("items") if isinstance(parallel_payload, dict) else None,
                                is_recursive=False,
                            )
                            _update_progress_state(
                                action_name,
                                observation,
                                observation_payload=parallel_payload,
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": "Parallel", "status": parallel_status, "result": parallel_payload}
                            ])
                            continue

                        parallel_slots: List[Optional[Dict[str, Any]]] = [None] * len(batch)
                        deferred_root_calls: List[Dict[str, Any]] = []
                        delegated_root_calls: List[Dict[str, Any]] = []
                        selected_calls = list(enumerate(batch))
                        if single_action_per_step_mode and len(selected_calls) > 1:
                            selected_calls = selected_calls[:1]
                            for skipped_idx, skipped_call in enumerate(batch[1:], start=1):
                                skipped_name = str(skipped_call.get("name", "")).strip() or f"planner_action_{skipped_idx + 1}"
                                parallel_slots[skipped_idx] = {
                                    "tool": skipped_name,
                                    "status": "skipped",
                                    "payload": {
                                        "status": "skipped",
                                        "result": (
                                            "Single-action-per-step ablation mode executed only the first planner "
                                            "action in this step."
                                        ),
                                        "planner_action_index": skipped_idx + 1,
                                        "planner_action_name": skipped_name,
                                        "source": "ablation_single_action_per_step",
                                    },
                                }

                        for idx, call in selected_calls:
                            name = call.get("name")
                            args = call.get("args", {})
                            _batch_desc = args.pop("__description__", "") if isinstance(args, dict) else ""
                            normalized_name = str(name or "").strip()
                            normalized_args = args if isinstance(args, dict) else {}
                            routed = _auto_route_root_action(
                                action_name=normalized_name,
                                action_args=normalized_args,
                                description_text=_batch_desc or thought or plan,
                                defer_execution=True,
                            )
                            if routed.get("handled"):
                                parallel_slots[idx] = {
                                    "tool": str(routed.get("tool", normalized_name)),
                                    "status": str(routed.get("status") or "success"),
                                    "payload": routed.get("payload"),
                                }
                                continue
                            if routed.get("requires_execution"):
                                deferred_root_calls.append({
                                    "index": idx,
                                    "tool_name": str(routed.get("tool", normalized_name)),
                                    "tool_args": (
                                        routed.get("delegate_action_args", normalized_args)
                                        if isinstance(routed.get("delegate_action_args", normalized_args), dict)
                                        else normalized_args
                                    ),
                                    "planner_action_name": normalized_name,
                                    "planner_action_args": normalized_args,
                                    "description_text": _batch_desc or thought or plan,
                                })
                                continue
                            delegated_root_calls.append({
                                "index": idx,
                                "action_name": str(routed.get("delegate_action_name", normalized_name)),
                                "action_args": (
                                    routed.get("delegate_action_args", normalized_args)
                                    if isinstance(routed.get("delegate_action_args", normalized_args), dict)
                                    else normalized_args
                                ),
                                "fallback_tool": normalized_name,
                                "description_text": _batch_desc or thought or plan,
                            })

                        for request in _execute_parallel_concrete_tool_calls(deferred_root_calls):
                            idx = int(request.get("index", 0))
                            tool_name = str(request.get("tool_name", "")).strip()
                            tool_args = request.get("tool_args", {}) if isinstance(request.get("tool_args", {}), dict) else {}
                            planner_action_name = str(request.get("planner_action_name", tool_name)).strip() or tool_name
                            prepared_call = request.get("prepared_call", {})
                            if request.get("error") is not None:
                                exc = request.get("error")
                                error_text = str(exc).strip() or "invalid parameters"
                                if _looks_like_parameter_error(error_text):
                                    logger.info(
                                        "%s[Root Parallel][Depth %d] concurrent execution rejected for %s: %s; delegating to subagent",
                                        '  ' * depth,
                                        depth,
                                        tool_name,
                                        error_text,
                                    )
                                    delegated_root_calls.append({
                                        "index": idx,
                                        "action_name": tool_name,
                                        "action_args": tool_args,
                                        "fallback_tool": planner_action_name,
                                        "description_text": request.get("description_text", "") or thought or plan,
                                    })
                                    continue
                                raise exc

                            result, observation_text, obs_status = _finalize_prepared_concrete_tool_call(
                                prepared_call,
                                request.get("result"),
                                abstract_tool=planner_action_name,
                            )
                            if tool_name == "Tool_Search":
                                try:
                                    lines = _render_tool_lines(result)
                                    _merge_discovered_tool_lines(lines)
                                except Exception as _te:
                                    logger.warning("Tool_Search parallel result parse failed: %s", _te)
                            if obs_status == "error" and _looks_like_parameter_error(result, observation_text):
                                error_text = (
                                    _extract_error_text_from_payload(result)
                                    or str(observation_text or "").strip()
                                    or "invalid parameters"
                                )
                                logger.info(
                                    "%s[Root Parallel][Depth %d] concurrent execution returned parameter-like error for %s: %s; delegating to subagent",
                                    '  ' * depth,
                                    depth,
                                    tool_name,
                                    _truncate_text(error_text, 220),
                                )
                                delegated_root_calls.append({
                                    "index": idx,
                                    "action_name": tool_name,
                                    "action_args": tool_args,
                                    "fallback_tool": planner_action_name,
                                    "description_text": request.get("description_text", "") or thought or plan,
                                })
                                continue
                            if obs_status == "success":
                                success = True
                                root_effective_tool_executed = True
                            parallel_slots[idx] = {
                                "tool": tool_name,
                                "status": obs_status,
                                "payload": result,
                            }

                        for delegated in sorted(delegated_root_calls, key=lambda item: int(item.get("index", 0))):
                            idx = int(delegated.get("index", 0))
                            fallback_tool = str(delegated.get("fallback_tool", "")).strip()
                            mediated = _delegate_root_action_via_subagent(
                                action_name=str(delegated.get("action_name", fallback_tool)),
                                action_args=(
                                    delegated.get("action_args", {})
                                    if isinstance(delegated.get("action_args", {}), dict)
                                    else {}
                                ),
                                description_text=str(delegated.get("description_text", "") or thought or plan),
                            )
                            delegated_trajectory = mediated.get("sub_trajectory")
                            if delegated_trajectory is not None:
                                delegated_sub_trajectories.append(delegated_trajectory)
                            parallel_slots[idx] = {
                                "tool": mediated.get("tool", fallback_tool),
                                "status": mediated.get("status") or "success",
                                "payload": mediated.get("payload"),
                            }

                        parallel_items = [item for item in parallel_slots if isinstance(item, dict)]
                        observation, parallel_status, parallel_payload = build_parallel_observation(parallel_items)
                        if any(str(item.get("status", "")).lower() == "success" for item in parallel_items):
                            _clear_identical_action_failure(current_signature)
                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=observation,
                            observation_items=parallel_payload.get("items") if isinstance(parallel_payload, dict) else None,
                            is_recursive=True,
                            sub_trajectories=delegated_sub_trajectories or None,
                        )
                        _update_progress_state(
                            action_name,
                            observation,
                            observation_payload=parallel_payload,
                        )
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {"tool": "Parallel", "status": parallel_status, "result": parallel_payload}
                        ])
                        continue

                    parallel_slots = [None] * len(batch)
                    concrete_parallel_requests: List[Dict[str, Any]] = []
                    selected_calls = list(enumerate(batch))
                    if single_action_per_step_mode and len(selected_calls) > 1:
                        selected_calls = selected_calls[:1]
                        for skipped_idx, skipped_call in enumerate(batch[1:], start=1):
                            skipped_name = str(skipped_call.get("name", "")).strip() or f"subagent_action_{skipped_idx + 1}"
                            parallel_slots[skipped_idx] = {
                                "tool": skipped_name,
                                "status": "skipped",
                                "payload": {
                                    "status": "skipped",
                                    "result": (
                                        "Single-action-per-step ablation mode executed only the first subagent "
                                        "tool in this step."
                                    ),
                                    "planner_action_index": skipped_idx + 1,
                                    "planner_action_name": skipped_name,
                                    "source": "ablation_single_action_per_step",
                                },
                            }
                    for idx, call in selected_calls:
                        name = call.get("name")
                        args = call.get("args", {})
                        normalized_name = str(name or "").strip()
                        normalized_args = args if isinstance(args, dict) else {}
                        tool = registry.get_tool(name, log_missing=False)
                        if tool.is_virtual:
                            _batch_desc = normalized_args.pop("__description__", "") if isinstance(normalized_args, dict) else ""
                            resolution = _resolve_virtual_subagent_action(
                                abstract_tool=name,
                                abstract_args=normalized_args,
                                description_text=_batch_desc or thought or plan,
                            )
                            if resolution.get("resolved"):
                                concrete_name = str(resolution.get("tool_name", "")).strip()
                                concrete_args = resolution.get("params", {}) if isinstance(resolution.get("params"), dict) else {}
                                logger.info(
                                    "%s[Depth %d] Abstract action %s auto-resolved to executable tool %s.",
                                    '  ' * depth,
                                    depth,
                                    name,
                                    concrete_name,
                                )
                                concrete_parallel_requests.append({
                                    "index": idx,
                                    "tool_name": concrete_name,
                                    "tool_args": concrete_args,
                                    "abstract_tool": normalized_name,
                                })
                            else:
                                failure_payload = dict(resolution.get("observation", {}) or {})
                                failure_payload["abstract_tool"] = name
                                if _is_nonrecoverable_resolution_failure(resolution):
                                    if depth > 0:
                                        subagent_toolsearch_failures += 1
                                        failure_payload["attempt"] = subagent_toolsearch_failures
                                        _remember_execution(
                                            tool_name="Tool_Search",
                                            tool_args={"intent_description": name},
                                            raw_result=failure_payload,
                                            observation_text=str(failure_payload.get("result", "")),
                                            observation_status=str(failure_payload.get("status", "no_valid_result")),
                                            source="system",
                                            abstract_tool=name,
                                            is_recursive=False,
                                            executed=False,
                                        )
                                        parallel_slots[idx] = {
                                            "tool": name,
                                            "status": str(failure_payload.get("status", "no_valid_result")),
                                            "payload": failure_payload,
                                        }
                                        if subagent_toolsearch_failures >= MAX_SUBAGENT_TOOLSEARCH_FAILURES:
                                            final_answer = (
                                                "Sub-agent fallback to planner: auto Tool_Search repeatedly failed "
                                                "to resolve abstract action into executable tools."
                                            )
                                            success = False
                                            finished = True
                                            break
                                    else:
                                        parallel_slots[idx] = {
                                            "tool": "Tool_Search",
                                            "status": str(failure_payload.get("status", "no_valid_result")),
                                            "payload": failure_payload,
                                        }
                                elif depth == 0:
                                    parallel_slots[idx] = {
                                        "tool": "Tool_Search",
                                        "status": str(failure_payload.get("status", "no_valid_result")),
                                        "payload": failure_payload,
                                    }
                                else:
                                    any_recursive = True
                                    logger.info(
                                        "%s[Depth %d] Abstract action %s requires recursive delegation; spawning sub-agent.",
                                        '  ' * depth,
                                        depth,
                                        name,
                                    )
                                    args_str = ", ".join([f"{k}={v}" for k, v in normalized_args.items()])
                                    sub_goal = f"Implement function {name}({args_str})"
                                    sub_ctx = ""
                                    compact_discovered_tools = _compact_discovered_tools_for_subagent()
                                    if compact_discovered_tools:
                                        sub_ctx = f"[Discovered Tools from parent]\n{compact_discovered_tools}"
                                    if _batch_desc:
                                        sub_ctx += f"\n\n[Abstract Tool Description]\n{_batch_desc}"
                                    candidate_tool_text = _format_candidate_tools_for_subagent(
                                        failure_payload.get("candidate_tools")
                                    )
                                    if candidate_tool_text:
                                        sub_ctx += (
                                            "\n\n[Auto Tool_Search Candidate Tools]\n"
                                            + candidate_tool_text
                                        )
                                    if mediation_mode:
                                        prefix = "[Sub-Agent Mode]\nplanner_action_mediation\n\n[Sub-Agent Return Mode]\nresult_only"
                                        sub_ctx = f"{prefix}\n\n{sub_ctx}".strip()
                                    sub_result = self.solve(goal=sub_goal, depth=depth + 1, extra_context=sub_ctx)
                                    delegated_trajectory = self._get_last_completed_trajectory(expected_depth=depth + 1)
                                    if delegated_trajectory is not None:
                                        delegated_sub_trajectories.append(delegated_trajectory)
                                    resolved_tool, sub_status, sub_payload, sub_observation, sub_meta = _normalize_subagent_result(
                                        abstract_tool=name,
                                        planner_args=normalized_args if isinstance(normalized_args, dict) else {},
                                        raw_sub_result=sub_result,
                                    )
                                    _remember_execution(
                                        tool_name=sub_meta.get("tool", resolved_tool),
                                        tool_args=sub_meta.get("args", {}),
                                        raw_result=sub_meta.get("payload", sub_payload),
                                        observation_text=sub_meta.get("observation", str(sub_result)),
                                        observation_status=sub_status,
                                        source=str(sub_meta.get("source", "subagent")),
                                        abstract_tool=str(sub_meta.get("abstract_tool", name)),
                                        is_recursive=True,
                                    )
                                    _log_observation_preview(f"Sub-agent {name} returned", str(sub_result))
                                    parallel_slots[idx] = {
                                        "tool": resolved_tool or name,
                                        "status": sub_status,
                                        "payload": sub_payload,
                                    }
                                    if sub_status == "success":
                                        success = True
                                    result_var = f"result_{name}_{iteration}"
                                    self.context.set_variable(result_var, sub_result)
                        else:
                            concrete_parallel_requests.append({
                                "index": idx,
                                "tool_name": normalized_name,
                                "tool_args": normalized_args,
                                "abstract_tool": normalized_name,
                            })

                    if finished:
                        break

                    for request in _execute_parallel_concrete_tool_calls(concrete_parallel_requests):
                        idx = int(request.get("index", 0))
                        tool_name = str(request.get("tool_name", "")).strip()
                        tool_args = request.get("tool_args", {}) if isinstance(request.get("tool_args", {}), dict) else {}
                        abstract_tool = str(request.get("abstract_tool", tool_name)).strip() or tool_name
                        prepared_call = request.get("prepared_call", {})
                        if request.get("error") is not None:
                            exc = request.get("error")
                            streak = _record_tool_error(tool_name)
                            err_text = (
                                f"Error executing {tool_name}: {str(exc)}. "
                                "Planner should try another relevant tool or adjust the probe window."
                            )
                            error_payload = {
                                "error": err_text,
                                "status": "error",
                            }
                            if abstract_tool != tool_name:
                                error_payload["resolved_tool"] = tool_name
                                error_payload["abstract_tool"] = abstract_tool
                            _remember_execution(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                raw_result=error_payload,
                                observation_text=err_text,
                                observation_status="error",
                                source="tool",
                                abstract_tool=abstract_tool,
                                is_recursive=False,
                            )
                            parallel_slots[idx] = {
                                "tool": tool_name,
                                "status": "error",
                                "payload": error_payload,
                            }
                            if streak >= 3:
                                logger.error(
                                    "%s[Depth %s] Tool %s failed %s times consecutively. Aborting.",
                                    '  ' * depth,
                                    depth,
                                    tool_name,
                                    streak,
                                )
                                observation = f"{err_text} (consecutive_failures={streak})"
                                agent.trajectory.add_step(
                                    thought=thought,
                                    plan=plan,
                                    action=action_name,
                                    action_args=action_args,
                                    observation=observation,
                                    is_recursive=any_recursive,
                                )
                                final_answer = observation
                                success = False
                                finished = True
                                break
                            continue

                        result, _, obs_status = _finalize_prepared_concrete_tool_call(
                            prepared_call,
                            request.get("result"),
                            abstract_tool=abstract_tool,
                        )
                        if tool_name == "Tool_Search":
                            try:
                                lines = _render_tool_lines(result)
                                _merge_discovered_tool_lines(lines)
                            except Exception as _te:
                                logger.warning("Tool_Search parallel result parse failed: %s", _te)
                        parallel_slots[idx] = {
                            "tool": tool_name,
                            "status": obs_status,
                            "payload": result,
                        }
                        if obs_status == "success":
                            success = True
                    if finished:
                        break
                    parallel_items = [item for item in parallel_slots if isinstance(item, dict)]
                    observation, parallel_status, parallel_payload = build_parallel_observation(parallel_items)
                    any_parallel_success = any(
                        str(item.get("status", "")).lower() == "success"
                        for item in parallel_items
                    )
                    if mediation_mode and any_parallel_success:
                        _remember_execution(
                            tool_name="Parallel",
                            tool_args=action_args,
                            raw_result=parallel_payload,
                            observation_text=observation,
                            observation_status=parallel_status,
                            source="tool",
                            abstract_tool=action_name,
                            is_recursive=any_recursive,
                            executed=True,
                        )
                    
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=observation,
                        observation_items=parallel_payload.get("items") if isinstance(parallel_payload, dict) else None,
                        is_recursive=any_recursive,
                        sub_trajectories=delegated_sub_trajectories or None,
                    )
                    if depth == 0 and any(str(item.get("status", "")).lower() == "success" for item in parallel_items):
                        root_effective_tool_executed = True
                    _update_progress_state(
                        action_name,
                        observation,
                        observation_payload=parallel_payload,
                    )
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {"tool": "Parallel", "status": parallel_status, "result": parallel_payload}
                    ])
                    if mediation_mode and any_parallel_success:
                        logger.info(
                            "%s[Subagent->Planner][Depth %d] Returning immediately after successful parallel execution.",
                            '  ' * depth,
                            depth,
                        )
                        final_answer = extract_planner_result_text(
                            parallel_payload,
                            tool_name="Parallel",
                        ).strip() or observation
                        success = True
                        finished = True
                        break
                    continue
                
                # Standard branch
                if depth == 0:
                    _sub_desc = action_args.pop("__description__", "") if isinstance(action_args, dict) else ""
                    if ablation_direct_toolsearch_mode:
                        resolution = _resolve_root_action_via_toolsearch_top1(
                            abstract_tool=action_name,
                            abstract_args=action_args if isinstance(action_args, dict) else {},
                            description_text=_sub_desc or thought or plan,
                        )
                        if resolution.get("resolved"):
                            concrete_name = str(resolution.get("tool_name", action_name)).strip() or action_name
                            concrete_args = (
                                resolution.get("params", {})
                                if isinstance(resolution.get("params"), dict)
                                else {}
                            )
                            try:
                                result, observation, obs_status, tool_executed = _execute_concrete_tool_call(
                                    concrete_name,
                                    concrete_args,
                                )
                                if tool_executed and obs_status == "success":
                                    success = True
                                    root_effective_tool_executed = True
                                    _clear_identical_action_failure(current_signature)
                            except Exception as e:
                                tool_executed = False
                                observation = (
                                    f"Error executing {concrete_name}: {str(e)}. "
                                    "Planner should try another relevant tool or adjust the probe window."
                                )
                                obs_status = "error"
                                result = {
                                    "error": observation,
                                    "status": "error",
                                    "resolved_tool": concrete_name,
                                    "abstract_tool": action_name,
                                }
                                _remember_execution(
                                    tool_name=concrete_name,
                                    tool_args=concrete_args,
                                    raw_result=result,
                                    observation_text=observation,
                                    observation_status=obs_status,
                                    source="tool",
                                    abstract_tool=action_name,
                                    is_recursive=False,
                                )

                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=observation,
                                is_recursive=False,
                            )
                            _update_progress_state(
                                concrete_name,
                                observation,
                                observation_payload=result,
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {
                                    "tool": concrete_name,
                                    "status": obs_status,
                                    "result": result,
                                }
                            ])
                            continue

                        failure_payload = dict(resolution.get("observation", {}) or {})
                        failure_payload["abstract_tool"] = action_name
                        observation, obs_status, _ = format_compact_observation(
                            failure_payload,
                            tool_name="Tool_Search",
                        )
                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=observation,
                            is_recursive=False,
                        )
                        _update_progress_state(
                            "Tool_Search",
                            observation,
                            observation_payload=failure_payload,
                        )
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {
                                "tool": "Tool_Search",
                                "status": str(failure_payload.get("status", "no_valid_result")),
                                "result": failure_payload,
                            }
                        ])
                        continue

                    routed = _auto_route_root_action(
                        action_name=action_name,
                        action_args=action_args if isinstance(action_args, dict) else {},
                        description_text=_sub_desc or thought or plan,
                    )
                    if routed.get("handled"):
                        observation = str(routed.get("observation", ""))
                        obs_status = str(routed.get("status") or "success")
                        if obs_status == "success":
                            _clear_identical_action_failure(current_signature)
                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=observation,
                            is_recursive=False,
                        )
                        _update_progress_state(
                            str(routed.get("tool", action_name)),
                            observation,
                            observation_payload=routed.get("payload"),
                        )
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {
                                "tool": str(routed.get("tool", action_name)),
                                "status": obs_status,
                                "result": routed.get("payload", observation),
                            }
                        ])
                        continue

                    mediated = _delegate_root_action_via_subagent(
                        action_name=str(routed.get("delegate_action_name", action_name)),
                        action_args=(
                            routed.get("delegate_action_args", action_args if isinstance(action_args, dict) else {})
                            if isinstance(routed.get("delegate_action_args", action_args if isinstance(action_args, dict) else {}), dict)
                            else (action_args if isinstance(action_args, dict) else {})
                        ),
                        description_text=_sub_desc or thought or plan,
                    )
                    observation = str(mediated.get("observation", ""))
                    obs_status = str(mediated.get("status") or "success")
                    if obs_status == "success":
                        _clear_identical_action_failure(current_signature)
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=observation,
                        is_recursive=True,
                        sub_trajectory=mediated.get("sub_trajectory"),
                    )
                    _update_progress_state(
                        str(mediated.get("tool", action_name)),
                        observation,
                        observation_payload=mediated.get("payload"),
                    )
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {
                            "tool": str(mediated.get("tool", action_name)),
                            "status": obs_status,
                            "result": mediated.get("payload", observation),
                        }
                    ])
                    continue

                tool = registry.get_tool(action_name, log_missing=False)
                
                if not tool.is_virtual:
                    logger.info(
                        "%s[%s][Depth %d] Selected executable tool directly: %s",
                        '  ' * depth,
                        role_label,
                        depth,
                        action_name,
                    )
                    if action_name == "Python_Executor":
                        code_script = str(action_args.get("code_script", ""))
                        python_policy_msg = _validate_python_executor_action(code_script)
                        if python_policy_msg:
                            observation = python_policy_msg
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=observation,
                                is_recursive=False
                            )
                            _update_progress_state(
                                action_name,
                                observation,
                                observation_payload={"error": observation},
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": action_name, "status": "error", "result": observation}
                            ])
                            continue
                        if _is_placeholder_python_code(code_script):
                            observation = (
                                "Error: Placeholder Python code detected (e.g., result=42/hello). "
                                "Generate task-specific code for the current goal and compute real values. "
                                "Use query numbers/constraints directly; never use hardcoded demo constants."
                            )
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=observation,
                                is_recursive=False
                            )
                            _update_progress_state(
                                action_name,
                                observation,
                                observation_payload={"error": observation},
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": action_name, "status": "error", "result": observation}
                            ])
                            continue
                        if not _code_mentions_goal_numbers(code_script):
                            observation = (
                                "Error: Python code is not grounded in query-specific numbers/constraints. "
                                "Rewrite code to use actual values from the user query before execution."
                            )
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=observation,
                                is_recursive=False
                            )
                            _update_progress_state(
                                action_name,
                                observation,
                                observation_payload={"error": observation},
                            )
                            _append_assistant_turn(thought, plan, action_name, action_args)
                            _append_observation_json([
                                {"tool": action_name, "status": "error", "result": observation}
                            ])
                            continue

                    tool_executed = False
                    try:
                        result, observation, obs_status, tool_executed = _execute_concrete_tool_call(action_name, action_args)
                        if tool_executed:
                            _clear_identical_action_failure(current_signature)

                        # Tool_Search post-processing: inject discovered tools
                        if action_name == "Tool_Search":
                            try:
                                lines = _render_tool_lines(result)
                                if lines:
                                    _merge_discovered_tool_lines(lines)
                                    logger.info(
                                        "%s[Depth %d] Discovered %d tools via Tool_Search.",
                                        '  ' * depth, depth, len(lines),
                                    )
                            except Exception as _te:
                                logger.warning("Tool_Search result parse failed: %s", _te)

                        if tool_executed:
                            success = True
                        if depth == 0 and tool_executed:
                            root_effective_tool_executed = True
                    except Exception as e:
                        tool_executed = False
                        streak = _record_tool_error(action_name)
                        observation = (
                            f"Error executing {action_name}: {str(e)}. "
                            "Planner should try another relevant tool or adjust the probe window."
                        )
                        obs_status = "error"
                        _remember_execution(
                            tool_name=action_name,
                            tool_args=action_args if isinstance(action_args, dict) else {},
                            raw_result={"error": observation, "status": "error"},
                            observation_text=observation,
                            observation_status=obs_status,
                            source="tool",
                            abstract_tool=action_name,
                            is_recursive=False,
                        )
                        if streak >= 3:
                            logger.error(
                                "%s[Depth %s] Tool %s failed %s times consecutively. Aborting.",
                                '  ' * depth,
                                depth,
                                action_name,
                                streak,
                            )
                            observation = f"{observation} (consecutive_failures={streak})"
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=observation,
                                is_recursive=False
                            )
                            final_answer = observation
                            success = False
                            finished = True
                            break
                    
                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=observation,
                        is_recursive=False
                    )
                    _update_progress_state(
                        action_name,
                        observation,
                        observation_payload=result if obs_status == "success" else {"error": observation},
                    )
                    observation_tool_name = action_name if tool_executed else "System"
                    if mediation_mode and obs_status == "success":
                        logger.info(
                            "%s[Subagent->Planner][Depth %d] Returning immediately after successful tool execution: %s",
                            '  ' * depth,
                            depth,
                            action_name,
                        )
                        final_answer = extract_planner_result_text(
                            result,
                            tool_name=action_name,
                        ).strip() or observation
                        success = True
                        finished = True
                        break
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {
                            "tool": observation_tool_name,
                            "status": obs_status,
                            "result": result if obs_status == "success" else {"error": observation},
                        }
                    ])
                
                else:
                    _sub_desc = action_args.pop("__description__", "") if isinstance(action_args, dict) else ""
                    resolution = _resolve_virtual_subagent_action(
                        abstract_tool=action_name,
                        abstract_args=action_args,
                        description_text=_sub_desc or thought or plan,
                    )

                    if resolution.get("resolved"):
                        concrete_name = str(resolution.get("tool_name", "")).strip()
                        concrete_args = resolution.get("params", {}) if isinstance(resolution.get("params"), dict) else {}
                        logger.info(
                            "%s[Depth %d] Abstract action %s auto-resolved to executable tool %s.",
                            '  ' * depth,
                            depth,
                            action_name,
                            concrete_name,
                        )
                        tool_executed = False
                        try:
                            result, observation, obs_status, tool_executed = _execute_concrete_tool_call(concrete_name, concrete_args)
                            if tool_executed:
                                _clear_identical_action_failure(current_signature)
                            if concrete_name == "Tool_Search":
                                try:
                                    lines = _render_tool_lines(result)
                                    if lines:
                                        _merge_discovered_tool_lines(lines)
                                except Exception as _te:
                                    logger.warning("Tool_Search result parse failed: %s", _te)
                            if tool_executed:
                                success = True
                            if depth == 0 and tool_executed:
                                root_effective_tool_executed = True
                        except Exception as e:
                            tool_executed = False
                            streak = _record_tool_error(concrete_name)
                            observation = (
                                f"Error executing {concrete_name}: {str(e)}. "
                                "Planner should try another relevant tool or adjust the probe window."
                            )
                            obs_status = "error"
                            _remember_execution(
                                tool_name=concrete_name,
                                tool_args=concrete_args,
                                raw_result={
                                    "error": observation,
                                    "status": "error",
                                    "resolved_tool": concrete_name,
                                    "abstract_tool": action_name,
                                },
                                observation_text=observation,
                                observation_status=obs_status,
                                source="tool",
                                abstract_tool=action_name,
                                is_recursive=False,
                            )
                            if streak >= 3:
                                logger.error(
                                    "%s[Depth %s] Tool %s failed %s times consecutively. Aborting.",
                                    '  ' * depth,
                                    depth,
                                    concrete_name,
                                    streak,
                                )
                                final_answer = observation
                                success = False
                                finished = True
                                agent.trajectory.add_step(
                                    thought=thought,
                                    plan=plan,
                                    action=action_name,
                                    action_args=action_args,
                                    observation=observation,
                                    is_recursive=False
                                )
                                break

                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=observation,
                            is_recursive=False
                        )
                        _update_progress_state(
                            concrete_name,
                            observation,
                            observation_payload=result if obs_status == "success" else {
                                "error": observation,
                                "resolved_tool": concrete_name,
                                "abstract_tool": action_name,
                            },
                        )
                        observation_tool_name = concrete_name if tool_executed else "System"
                        if mediation_mode and obs_status == "success":
                            logger.info(
                                "%s[Subagent->Planner][Depth %d] Returning immediately after successful tool execution: %s",
                                '  ' * depth,
                                depth,
                                concrete_name,
                            )
                            final_answer = extract_planner_result_text(
                                result,
                                tool_name=concrete_name,
                            ).strip() or observation
                            success = True
                            finished = True
                            break
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {
                                "tool": observation_tool_name,
                                "status": obs_status,
                                "result": result if obs_status == "success" else {"error": observation},
                            }
                        ])
                        continue

                    failure_payload = dict(resolution.get("observation", {}) or {})
                    failure_payload["abstract_tool"] = action_name

                    if depth > 0:
                        subagent_toolsearch_failures += 1
                        failure_payload["attempt"] = subagent_toolsearch_failures
                        observation, obs_status, _ = format_compact_observation(
                            failure_payload,
                            tool_name="Tool_Search",
                        )
                        _remember_execution(
                            tool_name="Tool_Search",
                            tool_args={"intent_description": action_name},
                            raw_result=failure_payload,
                            observation_text=observation,
                            observation_status=obs_status,
                            source="system",
                            abstract_tool=action_name,
                            is_recursive=False,
                            executed=False,
                        )
                        if (
                            _is_nonrecoverable_resolution_failure(resolution)
                            or subagent_toolsearch_failures >= MAX_SUBAGENT_TOOLSEARCH_FAILURES
                        ):
                            final_answer = (
                                "Sub-agent fallback to planner: auto Tool_Search repeatedly failed "
                                "to map abstract action to executable tools."
                            )
                            success = False
                            finished = True
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=observation,
                                is_recursive=False
                            )
                            break

                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=observation,
                            is_recursive=False
                        )
                        _update_progress_state(
                            "Tool_Search",
                            observation,
                            observation_payload=failure_payload,
                        )
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {
                                "tool": "Tool_Search",
                                "status": obs_status,
                                "result": failure_payload,
                            }
                        ])
                        continue

                    if _is_nonrecoverable_resolution_failure(resolution):
                        observation, obs_status, _ = format_compact_observation(
                            failure_payload,
                            tool_name="Tool_Search",
                        )
                        agent.trajectory.add_step(
                            thought=thought,
                            plan=plan,
                            action=action_name,
                            action_args=action_args,
                            observation=observation,
                            is_recursive=False
                        )
                        _update_progress_state(
                            "Tool_Search",
                            observation,
                            observation_payload=failure_payload,
                        )
                        _append_assistant_turn(thought, plan, action_name, action_args)
                        _append_observation_json([
                            {
                                "tool": "Tool_Search",
                                "status": str(failure_payload.get("status", "no_valid_result")),
                                "result": failure_payload,
                            }
                        ])
                        continue

                    if depth == 0:
                        logger.info(
                            "%s[Depth %d] Identified root-level abstract action %s; resolving without sub-agent recursion.",
                            '  ' * depth,
                            depth,
                            action_name,
                        )
                        routed = _delegate_root_action_via_subagent(
                            action_name=action_name,
                            action_args=action_args if isinstance(action_args, dict) else {},
                            description_text=_sub_desc or thought or plan,
                        )
                        observation = str(routed.get("observation", ""))
                        sub_status = str(routed.get("status") or "success")
                        sub_payload = routed.get("payload")
                        resolved_tool = str(routed.get("tool", action_name)).strip() or action_name
                        delegated_trajectory = routed.get("sub_trajectory")
                    else:
                        logger.info(
                            f"{'  ' * depth}[Depth {depth}] Identified high-level abstraction: {action_name}. "
                            f"Spawning sub-agent..."
                        )
                        args_str = ", ".join([f"{k}={v}" for k, v in action_args.items()])
                        sub_goal = f"Implement function {action_name}({args_str})"
                        # Parent passes abstract tool hints + planner description (no raw grounding evidence).
                        sub_context = ""
                        compact_discovered_tools = _compact_discovered_tools_for_subagent()
                        if compact_discovered_tools:
                            sub_context = f"[Discovered Tools from parent]\n{compact_discovered_tools}"
                        if _sub_desc:
                            sub_context += f"\n\n[Abstract Tool Description]\n{_sub_desc}"
                        candidate_tool_text = _format_candidate_tools_for_subagent(
                            failure_payload.get("candidate_tools")
                        )
                        if candidate_tool_text:
                            sub_context += (
                                "\n\n[Auto Tool_Search Candidate Tools]\n"
                                + candidate_tool_text
                            )
                        if mediation_mode:
                            prefix = "[Sub-Agent Mode]\nplanner_action_mediation\n\n[Sub-Agent Return Mode]\nresult_only"
                            sub_context = f"{prefix}\n\n{sub_context}".strip()
                        sub_result = self.solve(goal=sub_goal, depth=depth + 1, extra_context=sub_context)
                        delegated_trajectory = self._get_last_completed_trajectory(expected_depth=depth + 1)
                        resolved_tool, sub_status, sub_payload, observation, sub_meta = _normalize_subagent_result(
                            abstract_tool=action_name,
                            planner_args=action_args if isinstance(action_args, dict) else {},
                            raw_sub_result=sub_result,
                        )
                        _remember_execution(
                            tool_name=sub_meta.get("tool", resolved_tool),
                            tool_args=sub_meta.get("args", {}),
                            raw_result=sub_meta.get("payload", sub_payload),
                            observation_text=sub_meta.get("observation", observation),
                            observation_status=sub_status,
                            source=str(sub_meta.get("source", "subagent")),
                            abstract_tool=str(sub_meta.get("abstract_tool", action_name)),
                            is_recursive=True,
                        )
                        if sub_status == "no_valid_result":
                            _record_no_valid_result(
                                sub_meta.get("tool", resolved_tool),
                                sub_meta.get("observation", observation),
                            )
                    if sub_status == "success":
                        _clear_identical_action_failure(current_signature)
                    _log_observation_preview(
                        (
                            f"Sub-agent {action_name} returned"
                            if depth > 0
                            else f"Root route {action_name} returned"
                        ),
                        str(observation),
                    )

                    agent.trajectory.add_step(
                        thought=thought,
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=observation,
                        is_recursive=depth > 0,
                        sub_trajectory=delegated_trajectory,
                    )
                    if depth == 0:
                        root_effective_tool_executed = True
                    _update_progress_state(
                        resolved_tool or action_name,
                        observation,
                        observation_payload=sub_payload,
                    )
                    if mediation_mode and sub_status == "success":
                        logger.info(
                            "%s[Subagent->Planner][Depth %d] Returning immediately after recursive sub-agent success: %s",
                            '  ' * depth,
                            depth,
                            resolved_tool or action_name,
                        )
                        final_answer = extract_planner_result_text(
                            sub_payload,
                            tool_name=str(resolved_tool or action_name),
                        ).strip() or observation
                        success = True
                        finished = True
                        break
                    _append_assistant_turn(thought, plan, action_name, action_args)
                    _append_observation_json([
                        {
                            "tool": resolved_tool or action_name,
                            "status": sub_status,
                            "result": sub_payload,
                        }
                    ])

                    result_var = f"result_{action_name}_{iteration}"
                    self.context.set_variable(result_var, sub_result)
                
            except Exception as e:
                error_text = str(e)
                if isinstance(e, AcceleratorFatalError) or is_accelerator_fatal_error(e):
                    logger.exception(
                        "%s[Depth %d] Fatal accelerator error at iteration %d; aborting rollout immediately.",
                        '  ' * depth,
                        depth,
                        iteration,
                    )
                    raise AcceleratorFatalError(error_text) from e

                is_prompt_overflow = (
                    "prompt requires" in error_text.lower()
                    and "max_input_tokens" in error_text.lower()
                    and "exceeding configured" in error_text.lower()
                )
                if is_prompt_overflow:
                    if context_overflow_recovery_attempts < 1:
                        context_overflow_recovery_attempts += 1
                        compacted = _deterministic_compact_history(keep_recent_count=2)
                        if not compacted:
                            # A two-turn history can still contain several
                            # large image payloads. Preserve the text parts
                            # and drop only those attachments for one retry.
                            compacted_messages: List[Dict[str, Any]] = []
                            for entry in history:
                                if not isinstance(entry, Mapping):
                                    continue
                                content = entry.get("content")
                                if isinstance(content, list):
                                    text_parts = [
                                        item.get("text")
                                        for item in content
                                        if isinstance(item, Mapping)
                                        and str(item.get("type") or "").lower() == "text"
                                        and str(item.get("text") or "").strip()
                                    ]
                                    if text_parts:
                                        compacted_messages.append({
                                            "role": entry.get("role", "user"),
                                            "content": "\n".join(str(item) for item in text_parts),
                                        })
                                    else:
                                        compacted_messages.append({
                                            "role": entry.get("role", "user"),
                                            "content": "[multimodal history omitted after context pressure]",
                                        })
                                else:
                                    compacted_messages.append(dict(entry))
                            if compacted_messages and compacted_messages != history:
                                history = compacted_messages
                                compacted = True
                        if compacted:
                            agent.trajectory.add_debug_event({
                                "type": "context_compaction_retry",
                                "event_type": "ContextCompactionRetryEvent",
                                "depth": depth,
                                "iteration": iteration,
                                "attempt": context_overflow_recovery_attempts,
                                "history_messages": len(history),
                                "reason": "provider_prompt_overflow",
                            })
                            _append_history(
                                "user",
                                "Context was compacted after a provider prompt-overflow error. "
                                "Continue with the current evidence ledger and a targeted action.",
                            )
                            # Reuse the current iteration budget for the
                            # bounded retry instead of consuming the final
                            # planner turn solely on a transport failure.
                            iteration = max(0, iteration - 1)
                            continue
                    if last_action_signature:
                        blocked_action_signatures.add(last_action_signature)
                        _record_identical_action_failure(last_action_signature)
                    overflow_msg = (
                        "Error: planner context overflowed after the latest history update. "
                        "Do not retry the same tool call. Narrow the requested fields/time window or switch to a "
                        "smaller, more targeted probe before continuing."
                    )
                    logger.error(
                        "%s[Depth %d] %s Raw provider error: %s",
                        '  ' * depth,
                        depth,
                        overflow_msg,
                        error_text,
                    )
                    agent.trajectory.add_step(
                        thought=thought or "Context overflow occurred",
                        plan=plan,
                        action=action_name,
                        action_args=action_args,
                        observation=f"{overflow_msg} Provider detail: {error_text}",
                        is_recursive=False
                    )
                    final_answer = (
                        "Failed: planner context overflow prevented further reasoning. "
                        "The previous tool output or accumulated history was too large, so the same action was blocked "
                        "instead of being retried meaninglessly."
                    )
                    success = False
                    finished = True
                    break

                # Check for OOM errors (both NPU and CUDA)
                is_oom = (
                    "out of memory" in error_text.lower()
                    or is_npu_runtime_error(e)
                )
                
                if is_oom:
                    # Try to free memory immediately
                    try:
                        import torch
                        if hasattr(torch, 'npu') and torch.npu.is_available():
                            torch.npu.empty_cache()
                        elif torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    except Exception:
                        pass
                    
                    error_msg = f"OOM error: {error_text}"
                    logger.error(f"{'  ' * depth}[Depth {depth}] OOM at iteration {iteration}. Aborting.")
                    agent.trajectory.add_step(
                        thought=thought if 'thought' in locals() else "OOM occurred",
                        plan=plan if 'plan' in locals() else "",
                        action=action_name if 'action_name' in locals() else "Unknown",
                        action_args=action_args if 'action_args' in locals() else {},
                        observation=error_msg
                    )
                    final_answer = error_msg
                    success = False
                    finished = True
                    break

                logger.exception(
                    "%s[Depth %d] Error in iteration %d phase=%s action=%s: %s",
                    '  ' * depth,
                    depth,
                    iteration,
                    iteration_phase,
                    action_name,
                    e,
                )
                agent.trajectory.add_debug_event({
                    "type": "planner_iteration_error",
                    "agent_role": "planner" if depth == 0 else "subagent",
                    "depth": depth,
                    "iteration": iteration,
                    "phase": iteration_phase,
                    "action": action_name,
                    "exception_type": type(e).__name__,
                    "error": error_text[:500],
                })
                observation = f"Error: {error_text}"
                
                agent.trajectory.add_step(
                    thought=thought if 'thought' in locals() else "Error occurred",
                    plan=plan if 'plan' in locals() else "",
                    action=action_name if 'action_name' in locals() else "Unknown",
                    action_args=action_args if 'action_args' in locals() else {},
                    observation=observation
                )
                 
                _append_history("user", f"An error occurred: {observation}\n\nTry a different approach.")
            finally:
                trajectory_step_count_after_iteration = len(agent.trajectory.steps)
                should_yield_turn = (
                    not finished
                    and (
                        iteration_generated_response
                        or trajectory_step_count_after_iteration > trajectory_step_count_before_iteration
                    )
                )
                if should_yield_turn:
                    last_observation = ""
                    if trajectory_step_count_after_iteration > trajectory_step_count_before_iteration:
                        try:
                            last_observation = str(agent.trajectory.steps[-1].observation or "")
                        except Exception:
                            last_observation = ""
                    paused_seconds = self._yield_turn(
                        role=role_label.lower(),
                        depth=depth,
                        iteration=iteration,
                        status="paused",
                        current_tool=str(action_name or "").strip() or str(last_execution_meta.get("tool", "")).strip(),
                        result_preview=_truncate_text(last_observation, 180),
                    )
                    if paused_seconds > 0 and self.wall_clock_deadline_monotonic is not None:
                        self.wall_clock_deadline_monotonic += paused_seconds
        
        if isinstance(last_execution_meta, dict):
            last_execution_meta.setdefault("route_refresh_count", route_refresh_count)
            last_execution_meta.setdefault("forced_final_turn_finish", forced_final_turn_retry_used)
            last_execution_meta.setdefault("evidence_revision", evidence_revision)
            last_execution_meta.setdefault(
                "finish_rejection_counts",
                dict(finish_rejection_counts),
            )
            last_execution_meta.setdefault(
                "finish_rejection_revisions",
                dict(finish_rejection_revisions),
            )
            last_execution_meta.setdefault("recovery_turns_used", finish_recovery_turns_used)
            last_execution_meta.setdefault("max_recovery_rounds", max_finish_recovery_rounds)
            last_execution_meta.setdefault("finish_recovery_finish_pending", finish_recovery_finish_pending)
            last_execution_meta.setdefault("finish_recovery_directive_applied", bool(active_recovery_system_directive))
            last_execution_meta.setdefault("low_signal_guard_recoveries", low_signal_guard_recovery_count)
            last_execution_meta.setdefault(
                "evolution_policy_events",
                list(evolution_policy_events),
            )
        if finished and final_answer:
            normalized_final_answer = str(final_answer or "").strip().lower()
            if normalized_final_answer.startswith(("failed:", "error:")):
                success = False
            return final_answer, success, last_execution_meta
        else:
            if depth == 0 and evidence_first_mode:
                # Exhausting Planner/recovery turns is a framework execution
                # failure, not evidence that the video has no answer. Keep
                # genuine evidence-boundary outcomes (premise/option mismatch
                # or the low-signal guard) on their explicit error path above.
                agent.trajectory.add_debug_event({
                    "type": "finish_recovery_budget_exhausted",
                    "event_type": "FinishRecoveryBudgetExhaustedEvent",
                    "depth": depth,
                    "iteration": iteration,
                    "recovery_turns_used": finish_recovery_turns_used,
                    "max_recovery_rounds": max_finish_recovery_rounds,
                    "finish_rejection_streak": finish_rejection_streak,
                })
            terminal_fallback = (
                "Failed: planner_recovery_exhausted_without_verifiable_finish."
                if depth == 0 and evidence_first_mode
                else "Failed: planner_recovery_exhausted_without_finish."
            )
            return terminal_fallback, False, last_execution_meta
    
    # Observation formatting helpers

    @staticmethod
    def _format_tool_result(result: Any, tool_name: str = "") -> str:
        """Format a tool result into a filtered, size-bounded observation string.

        Applies PRD TODO-3 structural filtering then truncates if still too long.
        """
        observation, _, _ = format_compact_observation(result, tool_name=tool_name)
        return observation

    def _summarize_dict(self, result: Dict[str, Any], max_depth: int = 2) -> str:
        """Summarize a dict result preserving key information.
        
        Args:
            result: Dictionary to summarize
            max_depth: Maximum nesting depth to preserve
            
        Returns:
            Summarized string
        """
        import json
        
        def summarize_value(value, depth=0):
            if depth >= max_depth:
                return "..."
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            elif isinstance(value, dict):
                return {k: summarize_value(v, depth+1) for k, v in list(value.items())[:5]}
            elif isinstance(value, list):
                if len(value) <= 3:
                    return value
                return f"[{len(value)} items]"
            return str(value)[:50]
        
        summary = {k: summarize_value(v) for k, v in result.items()}
        return json.dumps(summary, indent=2, ensure_ascii=False)
    
    def _build_context_info(self, depth: int) -> str:
        """Build context information for the current depth.
        
        Args:
            depth: Current recursion depth
            
        Returns:
            Formatted context string
        """
        if depth == 0:
            # Root level - minimal context
            return ""
        
        # Sub-agent isolation: do not inject parent raw evidence (caption/transcript/graph)
        # or full variable snapshots. Sub-agent receives only abstract execution constraints.
        context_lines = [
            "Sub-agent mode: decompose abstract function into executable video-task tools.",
            "Do NOT request direct raw caption/transcript/graph context.",
            "Use abstract tool semantics and prefer runtime-routed candidate resolution over hardcoded favorite tools.",
            f"Current Depth: {depth}",
        ]
        return "\n".join(context_lines)
    
    def get_full_trajectory(self) -> Optional[Dict]:
        """Get the complete trajectory including all sub-trajectories.
        
        Returns:
            Trajectory dictionary or None
        """
        if self.root_trajectory:
            return self.root_trajectory.to_dict()
        return None
    
    def reset(self):
        """Reset the engine to initial state."""
        self.context.reset()
        self.root_trajectory = None
        self._active_executed_tool_signatures = {}
        logger.info("RecursiveEngine reset")
