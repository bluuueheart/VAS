"""Four-role LLM/VLM agent suite.

The roles intentionally share one loaded local kernel when possible, but each
role has its own provider view, prompt, structured contract, and event stream.
Sharing weights is a memory optimization; it does not turn the roles into one
deterministic service or let one role silently make another role's decision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from config.settings import settings
from core.llm.base import BaseLLM
from core.answer_contract import normalize_answer_format
from core.llm.provider import (
    AnthropicProvider,
    OpenAIProvider,
    TransformersProvider,
    build_role_specific_provider_view,
    cache_existing_transformers_provider,
)
from evolution.models import EVOLVER_HARNESS_ACTIONS as _EVOLVER_HARNESS_ACTIONS
from evolution.models import MAX_EVOLVER_OPERATIONS, SOURCE_BACKED_ACTIONS, valid_recovery_policy, normalize_mutation_payload

logger = logging.getLogger(__name__)


ROLE_ORDER = ("planner", "observer", "reflector", "evolver")
EVOLVER_HARNESS_ACTIONS = tuple(_EVOLVER_HARNESS_ACTIONS)

ROLE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "planner": {
        "agent_type": "tool_orchestration_agent",
        "kernel_modality": "vlm",
        "input_contract": "goal + trajectory + registered tool observations + optional frames",
        "decision_authority": "select executable tool calls and propose Finish",
        "delegates_to": ["observer", "reflector"],
    },
    "observer": {
        "agent_type": "execution_and_evidence_agent",
        "kernel_modality": "vlm",
        "input_contract": "Planner delegation + resolved route candidates + completed tool observations + attached image evidence",
        "decision_authority": "validate/repair one executable dispatch before execution and extract grounded evidence after execution",
        "delegates_to": ["planner"],
    },
   "reflector": {
       "agent_type": "evidence_judgment_agent",
       "kernel_modality": "llm",
        "input_contract": "proposed answer + evidence slots + trajectory + Reflector verify safety check",
        "decision_authority": "combine LLM judgment with verify result to accept or block Finish; never execute tools",
       "delegates_to": ["planner"],
   },
    "evolver": {
        "agent_type": "meta_evolution_agent",
        "kernel_modality": "llm",
        "input_contract": "completed trajectory summaries + deterministic diagnoses + runtime capability manifest + replay context",
        "decision_authority": "attribute failures, analyze dependencies, and propose bounded coordinated replay-gated harness mutations",
        "delegates_to": ["replay_gate"],
    },
}

ROLE_PROMPTS: Dict[str, str] = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "agent_prompts.json").read_text(encoding="utf-8")
)


def _provider_name(provider: Any) -> str:
    if isinstance(provider, TransformersProvider):
        return "transformers"
    if isinstance(provider, OpenAIProvider):
        return "openai"
    if isinstance(provider, AnthropicProvider):
        return "anthropic"
    return type(provider).__name__.lower()


def _json_value(value: Any, *, max_chars: int = 12000) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, max_chars=max_chars) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, max_chars=max_chars) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and len(value) > max_chars:
        return value[: max(0, max_chars - 40)] + "...<truncated>"
    return value


def _extract_json_object(
    text: Any,
    *,
    schema_validator: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    decoder = json.JSONDecoder()
    fallback: Optional[Dict[str, Any]] = None
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[index:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            if fallback is None:
                fallback = payload
            if schema_validator is None:
                return payload
            try:
                if bool(schema_validator(payload)):
                    return payload
            except Exception:
                continue
    return fallback


def _image_items(observation_items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build a bounded image attachment list with recency and probe locality.

    Evenly sampling every image in a growing trajectory can discard the most
    recent ``Inspect_Frame`` result and leave Reflector looking at an old
    retrieval collage.  That is especially harmful for short inserted events.
    Walk the newest observation records first, prefer their packed collages,
    and only then fall back to older records.  Sampling is still even within
    the selected record so parallel probes retain temporal coverage.
    """

    records = list(observation_items or [])
    max_visual_items = 4

    def collect_images(value: Any) -> List[Mapping[str, Any]]:
        collected: List[Mapping[str, Any]] = []
        if isinstance(value, Mapping):
            evidence = value.get("image_evidence")
            if isinstance(evidence, list):
                collected.extend(
                    image for image in evidence if isinstance(image, Mapping)
                )
            focus_evidence = value.get("focus_image_evidence")
            if isinstance(focus_evidence, list):
                for image in focus_evidence:
                    if not isinstance(image, Mapping):
                        continue
                    marked = dict(image)
                    marked["_focus_image"] = True
                    collected.append(marked)
            for key, nested in value.items():
                if key not in {"image_evidence", "focus_image_evidence"}:
                    collected.extend(collect_images(nested))
        elif isinstance(value, (list, tuple)):
            for nested in value:
                collected.extend(collect_images(nested))
        return collected

    def image_ref(image: Mapping[str, Any]) -> str:
        return str(image.get("image") or image.get("path") or "").strip()

    def is_packed(image: Mapping[str, Any]) -> bool:
        ref = image_ref(image).lower()
        mode = str(image.get("pack_mode") or "").lower()
        return mode not in {"", "single", "cached_single"} or "/pack_" in ref or ref.endswith("pack.jpg")

    def is_focus(image: Mapping[str, Any]) -> bool:
        return bool(image.get("_focus_image"))

    def evenly_select(images: List[Mapping[str, Any]], limit: int) -> List[Mapping[str, Any]]:
        if len(images) <= limit:
            return images
        if limit <= 1:
            return images[-1:]
        indices = [
            round(index * (len(images) - 1) / (limit - 1))
            for index in range(limit)
        ]
        return [images[index] for index in indices]

    selected: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in reversed(records):
        candidates = collect_images(record)
        if not candidates:
            continue
        focus = [image for image in candidates if is_focus(image)]
        packed = [image for image in candidates if is_packed(image) and not is_focus(image)]
        ordered = focus + packed + [
            image for image in candidates if not is_focus(image) and not is_packed(image)
        ]
        remaining = max_visual_items - len(selected)
        for image in evenly_select(ordered, remaining):
            ref = image_ref(image)
            if not ref or ref in seen:
                continue
            seen.add(ref)
            selected.append(image)
            if len(selected) >= max_visual_items:
                break
        if len(selected) >= max_visual_items:
            break

    visuals: List[Dict[str, Any]] = []
    for image in selected:
        ref = image_ref(image)
        visual: Dict[str, Any] = {"type": "image", "image": ref}
        for key in ("timestamp", "sample_times", "max_pixels"):
            if image.get(key) not in (None, "", []):
                visual[key] = image.get(key)
        visuals.append(visual)
    return visuals


def _reflector_trajectory_tail(
    trajectory_steps: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep evidence-bearing fields while removing stale private reasoning.

    Reflector already receives the proposed answer, evidence ledger, and the
    deterministic safety result.  Replaying every historical Planner thought
    lets an early visual guess remain an apparently active competing claim even
    after later direct frames resolve it.  Tool observations and structured
    evidence remain available for audit and judgment.
    """
    sanitized: List[Dict[str, Any]] = []
    for step in list(trajectory_steps or [])[-6:]:
        if not isinstance(step, Mapping):
            continue
        item = dict(step)
        item.pop("thought", None)
        item.pop("plan", None)
        item.pop("raw_response", None)
        item.pop("raw_llm_response", None)
        sanitized.append(item)
    return sanitized


def _reflector_evidence_slots(
    evidence_slots: Iterable[Mapping[str, Any]],
    deterministic_verification: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Present the reconciled ledger once the deterministic gate accepts it."""
    slots = [dict(slot) for slot in (evidence_slots or []) if isinstance(slot, Mapping)]
    if not bool((deterministic_verification or {}).get("can_finish")):
        return slots
    resolved = [
        slot
        for slot in slots
        if str(slot.get("evidence_status") or "").lower() == "verified"
        and not bool(slot.get("logical conflict", slot.get("logical_conflict", False)))
        and not bool(slot.get("superseded", False))
    ]
    return resolved or slots


class FourRoleAgentSuite:
    """Own four role-specific LLM/VLM provider views and contracts."""

    def __init__(self, kernel_provider: BaseLLM) -> None:
        self._kernel_provider = kernel_provider
        # Every role receives a role-specific provider view, including the
        # planner.  The view is shallow and therefore shares the loaded model
        # object, but its role defaults/contract identity are independent.  A
        # caller that constructs the suite for Evolver must not leave the
        # manifest's Planner mislabeled as ``model_role=evolver``.
        self.providers: Dict[str, BaseLLM] = {}
        for role in ROLE_ORDER:
            role_view = build_role_specific_provider_view(kernel_provider, role)
            self.providers[role] = cache_existing_transformers_provider(role_view)
        self._manifest = self._build_manifest()

    def _build_manifest(self) -> Dict[str, Any]:
        kernel_name = str(getattr(self._kernel_provider, "model_name", "") or "")
        kernel_id = "{}:{}".format(_provider_name(self._kernel_provider), kernel_name)
        return {
            "kernel_id": hashlib.sha256(kernel_id.encode("utf-8")).hexdigest()[:16],
            "kernel_provider": _provider_name(self._kernel_provider),
            "kernel_model": kernel_name,
            "roles": {
                role: {
                    "agent_role": role,
                    "provider_class": type(provider).__name__,
                    "model_role": str(getattr(provider, "model_role", role) or role),
                    "model_name": str(getattr(provider, "model_name", kernel_name) or kernel_name),
                    "modality": "vlm" if role in {"planner", "observer"} else "llm",
                    "independent_provider_view": True,
                    "independent_prompt": True,
                    "prompt_source": "planner_prompts" if role == "planner" else "four_role_prompts",
                    "structured_contract": (
                        "observer_dispatch_or_evidence_json"
                        if role == "observer"
                        else "planner_react_json"
                        if role == "planner"
                        else f"{role}_json"
                    ),
                    "llm_kernel": not type(provider).__name__.lower().startswith("local"),
                    "agent_definition": ROLE_DEFINITIONS[role],
                }
                for role, provider in self.providers.items()
            },
            "weights_shared_for_memory": True,
            "role_order": list(ROLE_ORDER),
        }

    def manifest(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._manifest, ensure_ascii=False))

    def _generation_config(self, role: str, provider: BaseLLM, max_tokens: int) -> Dict[str, Any]:
        provider_config = _provider_name(provider)
        config = dict(settings.get_generation_config(role, provider=provider_config) or {})
        configured_max = int(config.pop("max_new_tokens", max_tokens) or max_tokens)
        config["max_tokens"] = max(32, min(configured_max, int(max_tokens)))
        config.setdefault("temperature", 0.0)
        config.setdefault("enable_thinking", False)
        config["do_sample"] = False
        return config

    def _call(
        self,
        role: str,
        user_payload: Mapping[str, Any],
        *,
        visual_items: Optional[Sequence[Mapping[str, Any]]] = None,
        max_tokens: int,
        deadline_monotonic: Optional[float] = None,
        schema_mode: Optional[str] = None,
        system_directive: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_role = str(role or "").strip().lower()
        if normalized_role not in self.providers or normalized_role == "planner":
            raise ValueError(f"Unsupported auxiliary agent role: {role!r}")
        provider = self.providers[normalized_role]
        contract_mode = str(schema_mode or "").strip().lower()
        payload_text = json.dumps(_json_value(dict(user_payload)), ensure_ascii=False, sort_keys=True)
        user_content: Any = payload_text
        visual_payload = [dict(item) for item in (visual_items or []) if isinstance(item, Mapping)]
        if visual_payload:
            user_content = [{"type": "text", "text": payload_text}] + visual_payload
        system_content = ROLE_PROMPTS[normalized_role]
        if str(system_directive or "").strip():
            system_content += (
                "\n\n# Active Runtime Recovery Directive\n"
                + str(system_directive).strip()
            )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        generation_kwargs = self._generation_config(normalized_role, provider, max_tokens)
        if deadline_monotonic is not None:
            generation_kwargs["generation_deadline_monotonic"] = deadline_monotonic
        started = time.perf_counter()
        raw_response = ""
        error = ""
        initial_raw_response = ""
        schema_retry_count = 0
        try:
            logger.info(
                "[%s] LLM/VLM kernel call role=%s provider=%s model=%s max_tokens=%s",
                normalized_role.capitalize(),
                normalized_role,
                _provider_name(provider),
                getattr(provider, "model_name", ""),
                generation_kwargs.get("max_tokens"),
            )
            raw_response = str(provider.generate(messages, **generation_kwargs) or "")
            initial_raw_response = raw_response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s] LLM/VLM kernel call failed", normalized_role.capitalize())
        payload = _extract_json_object(
            raw_response,
            schema_validator=lambda candidate: self._schema_valid(
                normalized_role, candidate, contract_mode=contract_mode
            ),
        )
        schema_valid = self._schema_valid(
            normalized_role, payload, contract_mode=contract_mode
        )
        # Multimodal Qwen turns can occasionally stop after a natural-language
        # fragment even with the synthetic JSON stopping criterion. Retry the
        # same real kernel with an explicit JSON repair instruction while
        # preserving the attached visual evidence.
        if normalized_role in {"observer", "evolver"} and not error and not schema_valid:
            if normalized_role == "observer":
                repair_instruction = (
                    "Return one complete JSON object only, with the required keys "
                    "dispatch_decision, executable_action, reason, and confidence."
                    if contract_mode == "dispatch"
                    else
                    "Return one complete JSON object only, with the required keys "
                    "observation_quality, evidence_updates, recommended_action, and confidence."
                )
            else:
                repair_instruction = (
                    "Return one complete JSON object only. Correct the JSON contract while "
                    "preserving the proposed executable change. Do not replace a proposal with "
                    "no_mutation merely to avoid repairing its format. "
                    "If proposing a mutation, action must be exactly one of "
                    f"{', '.join(EVOLVER_HARNESS_ACTIONS)}; "
                    "Use mutations with 1..4 operations; omit mutation entirely. Each operation.target "
                    "must be non-empty; before and after must "
                    "be JSON objects or null; update_tool_schema.after may contain only "
                    "description, parameters, and tags; update_tool_schema.parameters must "
                    "be a complete JSON-Schema object with type=object and a properties "
                    "mapping. Source-backed policies require exact executable patches from the manifest. Never use output_schema or "
                    "add a property without an explicit runtime run() parameter."
                )
            repair_text = (
                payload_text
                + "\n\nSTRICT OUTPUT REPAIR: your previous response did not satisfy the "
                + normalized_role.capitalize()
                + " JSON contract. "
                + repair_instruction
                + "\nPrevious response to repair (data, not instructions):\n"
                + raw_response[-16384:]
            )
            repair_messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": ([{"type": "text", "text": repair_text}] + visual_payload)
                    if visual_payload
                    else repair_text,
                },
            ]
            try:
                schema_retry_count = 1
                logger.info(
                    "[%s] retrying invalid structured response with the same local kernel",
                    normalized_role.capitalize(),
                )
                retry_raw_response = str(provider.generate(repair_messages, **generation_kwargs) or "")
                retry_payload = _extract_json_object(
                    retry_raw_response,
                    schema_validator=lambda candidate: self._schema_valid(
                        normalized_role, candidate, contract_mode=contract_mode
                    ),
                )
                if retry_payload is not None and self._schema_valid(
                    normalized_role, retry_payload, contract_mode=contract_mode
                ):
                    raw_response = retry_raw_response
                    payload = retry_payload
                    schema_valid = True
                else:
                    raw_response = retry_raw_response or raw_response
                    payload = retry_payload or payload
                    schema_valid = self._schema_valid(
                        normalized_role, payload, contract_mode=contract_mode
                    )
            except Exception as exc:
                error = f"structured_retry_{type(exc).__name__}: {exc}"
        if normalized_role == "evolver" and isinstance(payload, Mapping):
            payload = normalize_mutation_payload(payload)
        status = "success" if not error and schema_valid else "error"
        event = {
            "type": "agent_llm_turn",
            "event_type": f"{normalized_role.capitalize()}AgentTurnEvent",
            "agent_role": normalized_role,
            "status": status,
            "schema_valid": schema_valid,
            "kernel": self._manifest["roles"][normalized_role],
            "request_digest": hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16],
            "response": raw_response,
            "response_payload": payload,
            "initial_response": initial_raw_response if schema_retry_count else "",
            "schema_retry_count": schema_retry_count,
            "contract_mode": contract_mode or "default",
            "system_directive_applied": bool(str(system_directive or "").strip()),
            "error": error,
            "input_modality": "multimodal" if visual_payload else "text",
            "visual_item_count": len(visual_payload),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
        logger.info(
            "[%s] Agent turn status=%s schema_valid=%s elapsed=%.3fs",
            normalized_role.capitalize(),
            status,
            schema_valid,
            event["elapsed_seconds"],
        )
        if raw_response:
            logger.info(
                "[%s][RAW_RESPONSE_BEGIN]\n%s\n[%s][RAW_RESPONSE_END]",
                normalized_role.capitalize(),
                raw_response,
                normalized_role.capitalize(),
            )
        return {
            "role": normalized_role,
            "status": status,
            "schema_valid": schema_valid,
            "payload": payload or {},
            "raw_response": raw_response,
            "error": error,
            "event": event,
        }

    @staticmethod
    def _schema_valid(
        role: str,
        payload: Optional[Mapping[str, Any]],
        *,
        contract_mode: str = "",
    ) -> bool:
        if not isinstance(payload, Mapping):
            return False
        if role == "observer":
            if str(contract_mode or "").strip().lower() == "dispatch":
                decision = str(payload.get("dispatch_decision") or "").strip().lower()
                action = payload.get("executable_action")
                if decision not in {"execute", "reject"} or not isinstance(action, Mapping):
                    return False
                if not isinstance(action.get("params", {}), Mapping):
                    return False
                return decision == "reject" or bool(str(action.get("tool") or "").strip())
            quality = str(payload.get("observation_quality") or "").strip().lower()
            if quality not in {"sufficient", "insufficient", "contradictory"}:
                return False
            if not isinstance(payload.get("evidence_updates"), list):
                return False
            if not isinstance(payload.get("claim_gaps"), list):
                return False
            recommended_action = payload.get("recommended_action")
            if not isinstance(recommended_action, Mapping):
                return False
            kind = str(recommended_action.get("kind") or "").strip().lower()
            if kind not in {"continue", "reobserve", "cross_verify", "finish"}:
                return False
            if "params" in recommended_action and not isinstance(
                recommended_action.get("params"), Mapping
            ):
                return False
            confidence = payload.get("confidence")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                return False
            return 0.0 <= confidence_value <= 1.0
        if role == "reflector":
            return str(payload.get("decision") or "") in {"accept", "reject", "request_more_evidence"} and isinstance(payload.get("can_finish"), bool)
        if role == "evolver":
            payload = normalize_mutation_payload(payload)
            decision = str(payload.get("decision") or "")
            if decision not in {"propose_mutation", "no_mutation", "request_more_cases"}:
                return False
            if decision != "propose_mutation":
                return True
            raw_mutations = payload.get("mutations")
            if raw_mutations is not None:
                if payload.get("mutation") is not None:
                    return False
                if not isinstance(raw_mutations, list) or not raw_mutations or len(raw_mutations) > MAX_EVOLVER_OPERATIONS:
                    return False
                mutations = raw_mutations
            else:
                mutations = [payload.get("mutation")]
            declared_dependencies = payload.get("dependency_analysis")
            if declared_dependencies is not None and not isinstance(declared_dependencies, Mapping):
                return False
            seen_targets = set()
            for mutation in mutations:
                if not isinstance(mutation, Mapping):
                    return False
                if str(mutation.get("action") or "") not in EVOLVER_HARNESS_ACTIONS:
                    return False
                action = str(mutation.get("action") or "")
                target = str(mutation.get("target") or "").strip()
                if not target:
                    return False
                target_key = (action, target)
                if target_key in seen_targets:
                    return False
                seen_targets.add(target_key)
                after = mutation.get("after")
                if not isinstance(after, Mapping):
                    return False
                before = mutation.get("before")
                if before is not None and not isinstance(before, Mapping):
                    return False
                if action == "edit_tool_source" or action in SOURCE_BACKED_ACTIONS or (action == "add_composition_edge" and "edits" in after):
                    from evolution.source_mutation import validate_source_edit
                    try:
                        expected_kind = SOURCE_BACKED_ACTIONS.get(action, "compose" if action == "add_composition_edge" else None)
                        if expected_kind and after.get("kind") != expected_kind:
                            return False
                        validate_source_edit(after)
                    except ValueError:
                        return False
                elif action == "update_tool_schema":
                    if set(after) - {"description", "parameters", "tags"}:
                        return False
                    parameters = after.get("parameters")
                    if parameters is not None and (
                        not isinstance(parameters, Mapping)
                        or parameters.get("type") != "object"
                        or not isinstance(parameters.get("properties"), Mapping)
                    ):
                        return False
                elif action == "add_composition_edge":
                    if target != "candidate_comparison" or after != {
                        "after": "evidence_acquisition",
                        "before": "finish",
                        "required": True,
                    }:
                        return False
                elif action == "update_recovery_policy":
                    if target != "low_signal_temporal_grounding" or not valid_recovery_policy(after):
                        return False
            return True
        return False

    def dispatch(
        self,
        *,
        goal: str,
        planner_action: Mapping[str, Any],
        resolved_route: Mapping[str, Any],
        trajectory_tail: Sequence[Mapping[str, Any]],
        deadline_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Let the live Observer own the Planner-to-runtime dispatch boundary."""
        return self._call(
            "observer",
            {
                "phase": "dispatch",
                "goal": goal,
                "planner_delegation": dict(planner_action or {}),
                "resolved_route": dict(resolved_route or {}),
                "recent_trajectory": list(trajectory_tail)[-3:],
                "action_boundary": (
                    "Return one executable action for the runtime. Preserve the intended capability; "
                    "do not execute a different tool, invent parameters, or claim that execution already happened."
                ),
            },
            max_tokens=min(int(settings.observer_max_generation_tokens), 512),
            deadline_monotonic=deadline_monotonic,
            schema_mode="dispatch",
        )

    def observe(
        self,
        *,
        goal: str,
        observation_items: Sequence[Mapping[str, Any]],
        trajectory_tail: Sequence[Mapping[str, Any]],
        deadline_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._call(
            "observer",
            {
                "goal": goal,
                "observation_items": list(observation_items),
                "recent_trajectory": list(trajectory_tail)[-3:],
                "claim_contract_instruction": (
                "Extract stable claim_id/scope/target/relation/direction/qualifiers/uncertainty "
                "from the native result; preserve decisive answer, count, order, negation and time window. "
                "Treat attached images as native evidence and inspect them, even when the text result "
                "contains only timestamps. Report missing fields in claim_gaps rather than inferring them. "
                "For reobserve/cross_verify, include the concrete tool name and materially changed params "
                "when they are inferable from the observation."
                ),
                "action_boundary": "Return one recommended next evidence action; the Planner executes it.",
            },
            visual_items=_image_items(observation_items),
            max_tokens=int(settings.observer_max_generation_tokens),
            deadline_monotonic=deadline_monotonic,
        )

    def reflect(
        self,
        *,
        goal: str,
        answer: str,
        completion_basis: str,
        evidence_slots: Sequence[Mapping[str, Any]],
        trajectory_steps: Sequence[Mapping[str, Any]],
        deterministic_verification: Optional[Mapping[str, Any]] = None,
        deadline_monotonic: Optional[float] = None,
        recovery_directive: str = "",
    ) -> Dict[str, Any]:
        """Run Reflector LLM judgment with its deterministic verify result."""
        answer = normalize_answer_format(answer)
        verification = dict(deterministic_verification or {})
        evidence_for_reflector = _reflector_evidence_slots(
            evidence_slots,
            verification,
        )
        trajectory_for_reflector = _reflector_trajectory_tail(trajectory_steps)
        options = dict(re.findall(r"(?m)^\s*\(([A-Z])\)\s*(.+)$", goal))
        answer = normalize_answer_format(answer, options)
        answer_match = re.match(r"\s*\(([A-Z])\)", answer)
        answer_label = answer_match.group(1) if answer_match else ""
        return self._call(
            "reflector",
            {
                "goal": goal,
                "proposed_answer": answer,
                "answer_options": options,
                "proposed_option_label": answer_label,
                "proposed_option_text": options.get(answer_label, ""),
                "completion_basis": completion_basis,
                "evidence_slots": evidence_for_reflector,
                "recent_trajectory": trajectory_for_reflector,
                "deterministic_safety_check": verification,
                "claim_contract": dict(verification.get("claim_contract") or {}),
                "evidence_precedence": (
                    "The deterministic safety check has reconciled earlier approximate or conflicting slots "
                    "with later direct visual evidence; judge the remaining verified slots as authoritative."
                    if bool(verification.get("can_finish"))
                    else "Later direct observations may supersede earlier approximate or conflicting claims."
                ),
                "action_boundary": "Your action is accept or block Finish; do not call tools.",
                "runtime_recovery_directive": str(recovery_directive or ""),
            },
            visual_items=_image_items(trajectory_steps),
            max_tokens=int(settings.reflector_max_generation_tokens),
            deadline_monotonic=deadline_monotonic,
            system_directive=recovery_directive,
        )

    def evolve(
        self,
        *,
        trajectory_summaries: Sequence[Mapping[str, Any]],
        deterministic_diagnoses: Sequence[Mapping[str, Any]],
        runtime_capabilities: Optional[Mapping[str, Any]] = None,
        evolution_context: Optional[Mapping[str, Any]] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._call(
            "evolver",
            {
                "trajectory_summaries": list(trajectory_summaries),
                "deterministic_diagnoses": list(deterministic_diagnoses),
                "runtime_capabilities": dict(runtime_capabilities or {}),
                "evolution_context": dict(evolution_context or {}),
                "allowed_mutation_actions": [
                    *EVOLVER_HARNESS_ACTIONS,
                ],
                "action_boundary": (
                    "Return an executable candidate for the online evolution controller. "
                    "Never claim publication: the controller publishes only after paired "
                    "trigger, neighbor, regression and held-out replay gates pass."
                ),
            },
            max_tokens=int(settings.evolver_max_generation_tokens),
            deadline_monotonic=deadline_monotonic,
        )
