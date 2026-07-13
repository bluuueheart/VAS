"""Deterministic evidence verifier used at the planner termination boundary."""
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from .models import VerificationResult


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "evolution.yaml"

_ANSWER_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "to", "of", "and", "or", "option", "answer",
    "at", "for", "from", "her", "in", "it", "on", "that", "this", "where", "while", "with",
})
_DYNAMIC_GENERIC_TOKENS = frozenset({
    "action", "appear", "caption", "computer", "frame", "hand", "laptop",
    "screen", "segment", "transcript", "video", "visual", "woman",
})
_OPTION_LINE_RE = re.compile(r"^\s*\(([A-Z])\)\s+(.+?)\s*$", re.MULTILINE)
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def load_evolution_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = (path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Evolution config must be a mapping: {config_path}")
    return payload


def _successful_families(steps: Iterable[Any]) -> set[str]:
    families: set[str] = set()
    for step in steps:
        if isinstance(step, dict):
            action = str(step.get("action") or "")
            observation = str(step.get("observation") or "").strip().lower()
        else:
            action = str(getattr(step, "action", "") or "")
            observation = str(getattr(step, "observation", "") or "").strip().lower()
        if not action or action in {"Finish", "System", "Parallel"} or observation.startswith("error"):
            continue
        if action in {"Semantic_Retrieve", "Keyword_Retrieve", "Segment_Retrieve", "Video_Summary", "asr"}:
            families.add("text")
        elif action in {"Inspect_Frame", "Inspect_Clip", "Video_Clip_QA", "OCR_Extract", "Action_Recognition"}:
            families.add("visual")
        elif action in {"Cross_Modal_Verify", "evidence_verify"}:
            families.add("cross_modal")
    return families


def _stem_token(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[:-len(suffix)]
    return token


def _content_tokens(value: Any) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return {
        _NUMBER_WORDS.get(_stem_token(token), _stem_token(token))
        for token in tokens
        if (len(token) > 1 or token.isdigit()) and token not in _ANSWER_STOPWORDS
    }


def _has_direct_answer_support(answer: str, verified_slots: Iterable[Mapping[str, Any]]) -> bool:
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return False
    for item in verified_slots:
        evidence_tokens = _content_tokens(f"{item.get('label', '')} {item.get('value', '')}")
        if answer_tokens.intersection(evidence_tokens):
            return True
    return False


def _iter_observation_items(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_observation_items(item)
        return
    if not isinstance(value, Mapping):
        return
    nested = value.get("items")
    if isinstance(nested, list):
        yield from _iter_observation_items(nested)
    if str(value.get("tool") or "").strip():
        yield value


def _has_dynamic_answer_support(
    answer: str,
    steps: Iterable[Any],
    verified_slots: Iterable[Mapping[str, Any]],
) -> bool:
    answer_tokens = _content_tokens(answer)
    required_overlap = 1 if len(answer_tokens) <= 2 else 2
    dynamic_result_tokens: List[set[str]] = []
    for step in steps:
        if isinstance(step, Mapping):
            items = step.get("observation_items") or []
        else:
            items = getattr(step, "observation_items", None) or []
        for item in _iter_observation_items(items):
            if str(item.get("tool") or "") != "Video_Clip_QA":
                continue
            status = str(item.get("status") or "success").lower()
            if status != "success":
                continue
            result_tokens = _content_tokens(item.get("result"))
            dynamic_result_tokens.append(result_tokens)
            if len(answer_tokens.intersection(result_tokens)) >= required_overlap:
                return True
    if not dynamic_result_tokens:
        return False
    for item in verified_slots:
        if "video_clip_qa" not in str(item.get("citation") or "").lower():
            continue
        claim_tokens = _content_tokens(f"{item.get('label', '')} {item.get('value', '')}")
        if answer_tokens.intersection(claim_tokens) and any(
            len({
                token for token in claim_tokens.intersection(result_tokens)
                if token not in _DYNAMIC_GENERIC_TOKENS and not token.isdigit()
            }) >= 2
            for result_tokens in dynamic_result_tokens
        ):
            return True
    return False


class EvidenceVerifier:
    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        payload = dict(config or load_evolution_config())
        self.thresholds = dict(payload.get("verification") or {})

    def verify_answerability(
        self,
        *,
        evidence_slots: List[Dict[str, Any]],
        steps: Iterable[Any],
        answer: str,
        completion_basis: str,
        query: str = "",
    ) -> VerificationResult:
        slots = [item for item in evidence_slots if isinstance(item, dict)]
        options = {label: text for label, text in _OPTION_LINE_RE.findall(str(query or ""))}
        answer_label_match = re.search(r"\(([A-Z])\)", str(answer or "").upper())
        answer_label = answer_label_match.group(1) if answer_label_match else ""
        answer_tokens = _content_tokens(answer)
        if options and answer_label in options:
            has_verified_answer_slot = any(
                str(item.get("evidence_status") or "").lower() == "verified"
                and bool(answer_tokens.intersection(_content_tokens(f"{item.get('label', '')} {item.get('value', '')}")))
                for item in slots
            )
            if has_verified_answer_slot:
                relevant_slots = [
                    item for item in slots
                    if bool(item.get("logical conflict", item.get("logical_conflict", False)))
                    or bool(answer_tokens.intersection(_content_tokens(f"{item.get('label', '')} {item.get('value', '')}")))
                ]
                if relevant_slots:
                    slots = relevant_slots
        verified = [item for item in slots if str(item.get("evidence_status", "")).lower() == "verified"]
        contradicted = [item for item in slots if bool(item.get("logical conflict", item.get("logical_conflict", False)))]
        grounded = [item for item in verified if str(item.get("citation", "")).strip()]
        denominator = max(1, len(slots))
        coverage = len(verified) / denominator if slots else 0.0
        grounding = len(grounded) / denominator if slots else 0.0
        contradiction = len(contradicted) / denominator if slots else 0.0
        families = _successful_families(steps)
        normalized_basis = completion_basis.lower().replace(" ", "")
        basis_is_cross_modal = "text->visual" in normalized_basis or "visual->text" in normalized_basis
        cross_modal = 1.0 if (
            "cross_modal" in families or {"text", "visual"}.issubset(families) or basis_is_cross_modal
        ) else 0.0
        progress = min(1.0, 0.45 * coverage + 0.35 * grounding + 0.20 * cross_modal)

        missing: List[str] = []
        unresolved: List[str] = []
        if not answer.strip():
            missing.append("non-empty answer candidate")
        if options and answer_label not in options:
            missing.append("final answer must select one listed option label")
        if not slots:
            missing.append("structured evidence claims")
        if coverage < float(self.thresholds.get("claim_coverage", 0.8)):
            unresolved.extend(str(item.get("label") or "unnamed claim") for item in slots if item not in verified)
            if not unresolved:
                unresolved.append("answer claim has no verified evidence")
        if grounding < float(self.thresholds.get("direct_grounding", 0.65)):
            missing.append("direct evidence citation")
        if bool(self.thresholds.get("require_answer_support", True)) and answer.strip():
            if not _has_direct_answer_support(answer, verified):
                unresolved.append("answer candidate lacks directly matched verified evidence")
        dynamic_tasks = [
            str(item).strip().lower()
            for item in self.thresholds.get("dynamic_answer_support_tasks", [])
            if str(item).strip()
        ]
        normalized_query = str(query or "").lower()
        if any(f"task type: {task}" in normalized_query for task in dynamic_tasks):
            if not _has_dynamic_answer_support(answer, steps, verified):
                missing.append("direct dynamic Video_Clip_QA result supporting the selected action")
        if contradiction > float(self.thresholds.get("contradiction_score", 0.2)):
            missing.append("resolve contradictory evidence")
        require_cross_modal = bool(self.thresholds.get("require_cross_modal", True))
        if require_cross_modal and cross_modal == 0.0:
            missing.append("text and visual evidence chain")
        if not completion_basis.strip():
            missing.append("completion basis")

        can_finish = not missing and not unresolved
        feedback = (
            "Evidence verifier accepted termination."
            if can_finish
            else "Evidence verifier rejected termination; acquire or reconcile: " + ", ".join(missing + unresolved)
        )
        return VerificationResult(
            progress_score=round(progress, 6),
            claim_coverage=round(coverage, 6),
            direct_grounding=round(grounding, 6),
            contradiction_score=round(contradiction, 6),
            cross_modal_support=round(cross_modal, 6),
            can_finish=can_finish,
            missing_information=missing,
            unresolved_claims=unresolved,
            feedback=feedback,
        )
