"""System and memory management tools."""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from config.settings import settings
from toolkit.base import BaseTool, enrich_tool_result

logger = logging.getLogger(__name__)


def _dedupe_texts(values: List[str], limit: Optional[int] = None) -> List[str]:
    seen = set()
    results: List[str] = []
    for raw in values:
        text = " ".join(str(raw or "").split()).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
        if limit is not None and len(results) >= limit:
            break
    return results


def _truncate_text(text: Any, max_chars: int = 220) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def _safe_json_loads(text: Any) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if not ((stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))):
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _extract_grounding_clues(text: str) -> List[str]:
    if "[Initial Top-1 Grounding Evidence" not in text and "[Grounding:" not in text:
        return []

    clues: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        if (
            line.startswith("[Grounding:")
            or line.startswith("- #")
            or line.startswith("caption=")
            or line.startswith("transcript=")
            or line.startswith("[Grounding Rule]")
        ):
            clues.append(_truncate_text(line, 360))
    if clues:
        return clues[:12]
    return [_truncate_text(text, 1200)]


def _extract_segment_summaries(payload: Any, limit: int = 2, max_chars: int = 160) -> List[str]:
    segments: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        matched_segments = payload.get("matched_segments")
        if isinstance(matched_segments, list):
            segments.extend(item for item in matched_segments if isinstance(item, dict))
        items = payload.get("items")
        if isinstance(items, list):
            segments.extend(item for item in items if isinstance(item, dict))
    elif isinstance(payload, list):
        segments.extend(item for item in payload if isinstance(item, dict))

    summaries: List[str] = []
    for item in segments:
        ref = item.get("segment_ref") or item.get("segment_id") or item.get("id") or item.get("source_segment")
        time_range = item.get("time") or item.get("time_range") or item.get("timestamp")
        detail = (
            item.get("caption")
            or item.get("transcript")
            or item.get("extracted_text")
            or item.get("description")
            or item.get("result")
        )
        pieces: List[str] = []
        if ref:
            pieces.append(_truncate_text(ref, 80))
        if time_range:
            pieces.append(_truncate_text(time_range, 60))
        if detail:
            pieces.append(_truncate_text(detail, max_chars))
        if pieces:
            summaries.append(" | ".join(pieces))
        if len(summaries) >= limit:
            break
    return summaries


class FoldMemory(BaseTool):
    """Compress and summarize long reasoning history."""
    
    name: str = "Fold_Memory"
    description: str = (
        "Summarize and compress overly long reasoning history. "
        "In multi-turn dialogue or long-chain reasoning, release context window space, "
        "retain key conclusions, forget non-essential details and known errors."
    )
    tags: list = ["base", "memory", "compression"]
    
    @property
    def model_path(self) -> str:
        return settings.fold_memory_model_path
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "context_window": {
                "type": "array",
                "description": "Historical steps or information to compress"
            },
            "keep_recent": {
                "type": "integer",
                "description": "Number of recent steps to keep uncompressed",
                "default": 2
            },
            "previous_state": {
                "type": "object",
                "description": "Previous structured folded memory state to merge forward",
                "default": None,
            },
            "discovered_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Current discovered tool summaries available to the planner",
                "default": [],
            },
            "blocked_action_signatures": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Action signatures that runtime has already blocked from repeating",
                "default": [],
            },
            "next_allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Runtime-suggested next tools that remain available",
                "default": [],
            },
            "importance_field": {
                "type": "string",
                "description": "Field name indicating importance score/flag in each step",
                "default": "importance"
            },
            "error_field": {
                "type": "string",
                "description": "Field name indicating error info in each step",
                "default": "error"
            }
        },
        "required": ["context_window"]
    }
    
    def run(
        self,
        context_window: List[Any],
        keep_recent: int = 2,
        previous_state: Optional[Dict[str, Any]] = None,
        discovered_tools: Optional[List[str]] = None,
        blocked_action_signatures: Optional[List[str]] = None,
        next_allowed_tools: Optional[List[str]] = None,
        importance_field: str = "importance",
        error_field: str = "error",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute structured memory folding for long planner histories."""
        llm = kwargs.get("llm")
        if keep_recent < 0:
            keep_recent = 0

        parsed_messages = self._normalize_context_window(context_window)
        prior_state = self._normalize_state(previous_state)
        blocked_signatures = _dedupe_texts(list(blocked_action_signatures or []), limit=32)
        live_discovered_tools = _dedupe_texts(list(discovered_tools or []), limit=24)
        live_next_allowed = _dedupe_texts(list(next_allowed_tools or []), limit=16)

        if len(parsed_messages) <= keep_recent:
            state = self._merge_runtime_state(
                prior_state,
                discovered_tools=live_discovered_tools,
                blocked_action_signatures=blocked_signatures,
                next_allowed_tools=live_next_allowed,
            )
            return enrich_tool_result({
                "compressed": False,
                "recent_steps": parsed_messages,
                "memory_state": state,
                "summary": "No compression needed",
            }, tool_name=self.name)

        recent = parsed_messages[-keep_recent:] if keep_recent > 0 else []
        to_fold = parsed_messages[:-keep_recent] if keep_recent > 0 else parsed_messages

        extracted_state, evidence_candidates = self._extract_state(
            to_fold,
            importance_field=importance_field,
            error_field=error_field,
        )
        memory_state = self._merge_states(prior_state, extracted_state)
        memory_state = self._merge_runtime_state(
            memory_state,
            discovered_tools=live_discovered_tools,
            blocked_action_signatures=blocked_signatures,
            next_allowed_tools=live_next_allowed,
        )
        memory_state = self._compress_long_sections(memory_state, evidence_candidates, llm=llm)

        compressed_summary = (
            f"Folded {len(to_fold)} messages into structured memory state. "
            f"facts={len(memory_state['found_facts'])}, tools={len(memory_state['discovered_tools'])}, "
            f"failed_actions={len(memory_state['failed_actions'])}, gaps={len(memory_state['missing_evidence'])}."
        )
        result = {
            "compressed": True,
            "original_length": len(parsed_messages),
            "compression_ratio": len(to_fold) / len(parsed_messages) if parsed_messages else 0.0,
            "compressed_summary": compressed_summary,
            "memory_state": memory_state,
            "recent_steps": recent,
        }

        logger.info(
            "Fold_Memory: folded=%d kept_recent=%d facts=%d failures=%d tools=%d",
            len(to_fold),
            len(recent),
            len(memory_state["found_facts"]),
            len(memory_state["failed_actions"]),
            len(memory_state["discovered_tools"]),
        )
        return enrich_tool_result(result, tool_name=self.name)

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "found_facts": [],
            "discovered_tools": [],
            "failed_actions": [],
            "do_not_repeat": [],
            "missing_evidence": [],
            "next_allowed_tools": [],
            "recent_relevant_observations": [],
        }

    def _normalize_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = self._empty_state()
        if not isinstance(state, dict):
            return normalized
        for key in normalized:
            value = state.get(key, [])
            if key == "failed_actions":
                items = value if isinstance(value, list) else []
                cleaned: List[Dict[str, Any]] = []
                seen = set()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    signature = " ".join(str(item.get("action_signature", "")).split()).strip()
                    if not signature or signature in seen:
                        continue
                    seen.add(signature)
                    cleaned.append({
                        "action_signature": signature,
                        "tool": _truncate_text(item.get("tool", ""), 80),
                        "reason": _truncate_text(item.get("reason", ""), 180),
                        "failure_count": max(1, int(item.get("failure_count", 1) or 1)),
                        "recent_observation": _truncate_text(item.get("recent_observation", ""), 220),
                    })
                normalized[key] = cleaned[:16]
                continue
            normalized[key] = _dedupe_texts(value if isinstance(value, list) else [], limit=24)
        return normalized

    def _normalize_context_window(self, context_window: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in context_window:
            if isinstance(item, dict) and "role" in item and "content" in item:
                content = item.get("content", "")
                if isinstance(content, list):
                    text_parts: List[str] = []
                    for entry in content:
                        if not isinstance(entry, dict):
                            continue
                        if str(entry.get("type", "")).strip().lower() == "text":
                            text_value = str(entry.get("text", "")).strip()
                            if text_value:
                                text_parts.append(text_value)
                    content = "\n".join(text_parts)
                normalized.append({
                    "role": str(item.get("role", "user")),
                    "content": str(content),
                })
            elif isinstance(item, dict):
                normalized.append({
                    "role": str(item.get("role", "user")),
                    "content": json.dumps(item, ensure_ascii=False),
                })
            else:
                normalized.append({"role": "user", "content": str(item)})
        return normalized

    def _extract_state(
        self,
        messages: List[Dict[str, Any]],
        importance_field: str,
        error_field: str,
    ) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
        state = self._empty_state()
        evidence_candidates: List[Dict[str, str]] = []
        last_actions: List[Dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            payload = _safe_json_loads(content)

            if role == "assistant":
                last_actions = self._extract_actions(payload)
                continue

            observations = self._extract_observations(payload)
            if observations:
                self._consume_observations(
                    observations=observations,
                    last_actions=last_actions,
                    state=state,
                    evidence_candidates=evidence_candidates,
                )
                continue

            grounding_clues = _extract_grounding_clues(content)
            if grounding_clues:
                state["found_facts"].extend(grounding_clues[:6])
                state["recent_relevant_observations"].extend(grounding_clues[:8])

            if isinstance(message, dict):
                importance = message.get(importance_field)
                has_error = bool(message.get(error_field))
                if has_error:
                    state["missing_evidence"].append(_truncate_text(content, 180))
                elif importance not in (None, 0, 0.0):
                    state["found_facts"].append(_truncate_text(content, 180))

            lowered = content.lower()
            if "must choose a different tool" in lowered or "blocked due to prior repetition" in lowered:
                state["do_not_repeat"].append(_truncate_text(content, 200))
            elif any(token in lowered for token in ("missing", "not found", "insufficient", "no valid result")):
                state["missing_evidence"].append(_truncate_text(content, 180))

        state["found_facts"] = _dedupe_texts(state["found_facts"], limit=16)
        state["discovered_tools"] = _dedupe_texts(state["discovered_tools"], limit=24)
        state["do_not_repeat"] = _dedupe_texts(state["do_not_repeat"], limit=24)
        state["missing_evidence"] = _dedupe_texts(state["missing_evidence"], limit=16)
        state["recent_relevant_observations"] = _dedupe_texts(state["recent_relevant_observations"], limit=12)
        state["next_allowed_tools"] = _dedupe_texts(state["next_allowed_tools"], limit=12)
        state["failed_actions"] = self._dedupe_failed_actions(state["failed_actions"])
        return state, evidence_candidates

    def _extract_actions(self, payload: Optional[Any]) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        actions = payload.get("Actions")
        if not isinstance(actions, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            tool = str(action.get("tool", "")).strip()
            params = action.get("params", {})
            if not tool:
                continue
            normalized.append({
                "tool": tool,
                "params": params if isinstance(params, dict) else {},
                "action_signature": self._action_signature(tool, params if isinstance(params, dict) else {}),
            })
        return normalized

    def _extract_observations(self, payload: Optional[Any]) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        observations = payload.get("Observation")
        if not isinstance(observations, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["tool"] = str(item.get("tool", "")).strip()
            entry["status"] = str(item.get("status") or "success").strip().lower()
            entry["result"] = str(item.get("result", ""))
            normalized.append(entry)
        return normalized

    def _consume_observations(
        self,
        observations: List[Dict[str, Any]],
        last_actions: List[Dict[str, Any]],
        state: Dict[str, Any],
        evidence_candidates: List[Dict[str, str]],
    ) -> None:
        action_by_tool = {str(item.get("tool")): item for item in last_actions}
        for observation in observations:
            tool = observation.get("tool", "") or "Unknown"
            status = observation.get("status") or "success"
            result_text = observation.get("result", "")
            result_payload = _safe_json_loads(result_text)
            if result_payload is None and isinstance(observation, dict):
                extra_payload = {
                    key: value
                    for key, value in observation.items()
                    if key not in {"tool", "status", "result"} and value not in (None, "", [], {})
                }
                if extra_payload:
                    result_payload = dict(extra_payload)
                    if result_text.strip():
                        result_payload.setdefault("result", result_text)
            action_info = action_by_tool.get(tool) or (last_actions[0] if len(last_actions) == 1 else None)
            action_signature = ""
            if isinstance(action_info, dict):
                action_signature = str(action_info.get("action_signature", ""))

            is_failure = status != "success"
            failure_reason = ""
            if isinstance(result_payload, dict):
                payload_status = str(result_payload.get("status", "")).lower()
                if payload_status == "no_valid_result":
                    is_failure = True
                    failure_reason = str(result_payload.get("result") or result_payload.get("error") or "No valid result")
                elif result_payload.get("error"):
                    is_failure = True
                    failure_reason = str(result_payload.get("error"))
            lowered_result = result_text.lower()
            if any(token in lowered_result for token in ("no valid result", "no matching", "not found", "did not provide usable evidence")):
                is_failure = True
                if not failure_reason:
                    failure_reason = result_text

            if tool == "Tool_Search":
                state["discovered_tools"].extend(self._extract_tool_summaries(result_payload, result_text))

            if tool == "Parallel" and isinstance(result_payload, dict):
                nested_items = result_payload.get("items")
                if isinstance(nested_items, list):
                    nested_observations: List[Dict[str, Any]] = []
                    for item in nested_items:
                        if not isinstance(item, dict):
                            continue
                        nested_observations.append({
                            "tool": str(item.get("tool", "")).strip() or "Unknown",
                            "status": str(item.get("status") or "success").strip().lower(),
                            "result": json.dumps(item, ensure_ascii=False),
                        })
                    if nested_observations:
                        self._consume_observations(
                            nested_observations,
                            last_actions,
                            state,
                            evidence_candidates,
                        )

            observation_summary = self._summarize_observation(tool, result_payload, result_text)
            if observation_summary:
                state["recent_relevant_observations"].append(observation_summary)
                if len(result_text) > 180:
                    evidence_candidates.append({"tool": tool, "text": result_text})

            if is_failure:
                normalized_reason = _truncate_text(failure_reason or result_text or "Tool failed", 180)
                state["missing_evidence"].append(normalized_reason)
                if action_signature:
                    state["failed_actions"].append({
                        "action_signature": action_signature,
                        "tool": tool,
                        "reason": normalized_reason,
                        "failure_count": 1,
                        "recent_observation": _truncate_text(result_text, 220),
                    })
                    state["do_not_repeat"].append(action_signature)
                else:
                    state["do_not_repeat"].append(_truncate_text(f"{tool}: {normalized_reason}", 180))
                continue

            inferred_facts = self._extract_fact_candidates(tool, result_payload, result_text)
            state["found_facts"].extend(inferred_facts)

    def _extract_tool_summaries(self, result_payload: Optional[Any], result_text: str) -> List[str]:
        tools: List[str] = []
        candidates: List[Dict[str, Any]] = []
        if isinstance(result_payload, dict):
            items = result_payload.get("items")
            if isinstance(items, list):
                candidates.extend(item for item in items if isinstance(item, dict))
        elif isinstance(result_payload, list):
            candidates.extend(item for item in result_payload if isinstance(item, dict))

        for item in candidates[:8]:
            name = str(item.get("name", "")).strip()
            description = _truncate_text(item.get("description", ""), 110)
            required = item.get("parameters", {}).get("required", []) if isinstance(item.get("parameters"), dict) else []
            required_text = ",".join(str(x) for x in required[:4]) if isinstance(required, list) else ""
            if name:
                summary = f"{name}: {description}"
                if required_text:
                    summary += f" | required={required_text}"
                tools.append(summary)

        if tools:
            return tools

        pattern = re.compile(r"-\s*([A-Za-z_]\w*)\((.*?)\):\s*(.+)")
        for line in result_text.splitlines():
            match = pattern.search(line.strip())
            if not match:
                continue
            name = match.group(1)
            params = _truncate_text(match.group(2), 60)
            description = _truncate_text(match.group(3), 100)
            tools.append(f"{name}: {description} | params={params}")
        return tools

    def _summarize_observation(self, tool: str, result_payload: Optional[Any], result_text: str) -> str:
        segment_summaries = _extract_segment_summaries(result_payload, limit=2, max_chars=140)
        if segment_summaries:
            return f"{tool}: {' || '.join(segment_summaries)}"
        if isinstance(result_payload, dict):
            for key in ("result", "summary", "answer", "description", "context"):
                value = result_payload.get(key)
                if isinstance(value, str) and value.strip():
                    return f"{tool}: {_truncate_text(value, 180)}"
            if isinstance(result_payload.get("items"), list):
                pieces = []
                for item in result_payload["items"][:3]:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("tool", "")).strip() or tool
                    value = str(item.get("result", "")).strip()
                    if value:
                        pieces.append(f"{label}: {_truncate_text(value, 90)}")
                if pieces:
                    return " | ".join(pieces)
            if isinstance(result_payload.get("items_preview"), list):
                preview = " | ".join(_truncate_text(x, 80) for x in result_payload["items_preview"][:3])
                if preview:
                    return f"{tool}: {preview}"
        return f"{tool}: {_truncate_text(result_text, 180)}" if result_text.strip() else ""

    def _extract_fact_candidates(self, tool: str, result_payload: Optional[Any], result_text: str) -> List[str]:
        facts: List[str] = []
        segment_summaries = _extract_segment_summaries(result_payload, limit=3, max_chars=180)
        for summary in segment_summaries:
            facts.append(f"{tool}: {summary}")
        if isinstance(result_payload, dict):
            for key in ("result", "summary", "answer", "description"):
                value = result_payload.get(key)
                if isinstance(value, str) and value.strip():
                    facts.append(f"{tool}: {_truncate_text(value, 180)}")
                    break
            for key in ("segment_id", "time", "time_range", "video"):
                value = result_payload.get(key)
                if value:
                    facts.append(f"{tool}: {key}={_truncate_text(value, 80)}")
            if isinstance(result_payload.get("items"), list):
                for item in result_payload["items"][:3]:
                    if not isinstance(item, dict):
                        continue
                    item_tool = str(item.get("tool", tool)).strip() or tool
                    item_result = str(item.get("result", "")).strip()
                    if item_result:
                        facts.append(f"{item_tool}: {_truncate_text(item_result, 160)}")
        elif isinstance(result_payload, list):
            for item in result_payload[:2]:
                if isinstance(item, dict):
                    segment = item.get("segment_id") or item.get("id")
                    time_range = item.get("time") or item.get("time_range")
                    if segment or time_range:
                        facts.append(
                            f"{tool}: segment={_truncate_text(segment, 80)} time={_truncate_text(time_range, 80)}"
                        )
        elif result_text.strip():
            facts.append(f"{tool}: {_truncate_text(result_text, 160)}")
        return facts

    def _dedupe_failed_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for action in actions:
            if not isinstance(action, dict):
                continue
            signature = " ".join(str(action.get("action_signature", "")).split()).strip()
            if not signature:
                continue
            if signature not in merged:
                merged[signature] = {
                    "action_signature": signature,
                    "tool": _truncate_text(action.get("tool", ""), 80),
                    "reason": _truncate_text(action.get("reason", ""), 180),
                    "failure_count": max(1, int(action.get("failure_count", 1) or 1)),
                    "recent_observation": _truncate_text(action.get("recent_observation", ""), 220),
                }
                continue
            existing = merged[signature]
            existing["failure_count"] = max(
                int(existing.get("failure_count", 1) or 1),
                int(existing.get("failure_count", 1) or 1) + int(action.get("failure_count", 1) or 1),
            )
            if len(str(action.get("reason", ""))) > len(str(existing.get("reason", ""))):
                existing["reason"] = _truncate_text(action.get("reason", ""), 180)
            if len(str(action.get("recent_observation", ""))) > len(str(existing.get("recent_observation", ""))):
                existing["recent_observation"] = _truncate_text(action.get("recent_observation", ""), 220)
        return list(merged.values())[:16]

    def _merge_states(self, previous_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._normalize_state(previous_state)
        normalized_new = self._normalize_state(new_state)
        for key in ("found_facts", "discovered_tools", "do_not_repeat", "missing_evidence", "next_allowed_tools", "recent_relevant_observations"):
            merged[key] = _dedupe_texts(list(merged.get(key, [])) + list(normalized_new.get(key, [])), limit=24)
        merged["failed_actions"] = self._dedupe_failed_actions(
            list(merged.get("failed_actions", [])) + list(normalized_new.get("failed_actions", []))
        )
        return merged

    def _merge_runtime_state(
        self,
        state: Dict[str, Any],
        *,
        discovered_tools: List[str],
        blocked_action_signatures: List[str],
        next_allowed_tools: List[str],
    ) -> Dict[str, Any]:
        merged = self._normalize_state(state)
        merged["discovered_tools"] = _dedupe_texts(list(merged["discovered_tools"]) + discovered_tools, limit=24)
        merged["do_not_repeat"] = _dedupe_texts(list(merged["do_not_repeat"]) + blocked_action_signatures, limit=24)
        failed_tool_names = [item.get("tool", "") for item in merged["failed_actions"] if isinstance(item, dict)]
        merged["next_allowed_tools"] = _dedupe_texts(
            list(merged["next_allowed_tools"]) + next_allowed_tools + [
                tool_summary for tool_summary in merged["discovered_tools"]
                if not any(tool_summary.lower().startswith(f"{str(name).strip().lower()}:") for name in failed_tool_names if name)
            ],
            limit=16,
        )
        return merged

    def _compress_long_sections(
        self,
        memory_state: Dict[str, Any],
        evidence_candidates: List[Dict[str, str]],
        *,
        llm: Any = None,
    ) -> Dict[str, Any]:
        state = self._normalize_state(memory_state)
        candidates = [item for item in evidence_candidates if isinstance(item, dict) and str(item.get("text", "")).strip()]
        if not candidates or llm is None:
            return state

        prompt = (
            "Compress planner trace evidence into concise structured memory. "
            "Return strict JSON with keys found_facts, missing_evidence, recent_relevant_observations. "
            "Each value must be a list of short factual strings. Keep only evidence-grounded content. "
            "Do not invent tools, numbers, or conclusions."
        )
        payload = {
            "current_state": {
                "found_facts": state["found_facts"][:8],
                "missing_evidence": state["missing_evidence"][:8],
                "recent_relevant_observations": state["recent_relevant_observations"][:6],
            },
            "candidate_evidence": [
                {
                    "tool": _truncate_text(item.get("tool", ""), 60),
                    "text": _truncate_text(item.get("text", ""), 500),
                }
                for item in candidates[:8]
            ],
        }
        try:
            messages = llm.format_messages(
                system_prompt=prompt,
                user_message=json.dumps(payload, ensure_ascii=False, indent=2),
            )
            max_fold_tokens = int(
                settings.fold_memory_generation_config.get("max_new_tokens", 768) or 768
            )
            response = llm.generate(
                messages,
                max_tokens=max_fold_tokens,
                temperature=0.0,
            )
            parsed = _safe_json_loads(response)
            if not isinstance(parsed, dict):
                return state
            for key in ("found_facts", "missing_evidence", "recent_relevant_observations"):
                values = parsed.get(key)
                if isinstance(values, list):
                    state[key] = _dedupe_texts(list(state.get(key, [])) + list(values), limit=24)
        except Exception as exc:
            logger.warning("Fold_Memory LLM compression skipped due to error: %s", exc)
        return state

    @staticmethod
    def _action_signature(tool: str, params: Dict[str, Any]) -> str:
        try:
            serialized = json.dumps(params or {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            serialized = str(params)
        return f"{tool}:{serialized}"


