"""Deterministic evidence verifier used at the planner termination boundary."""
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from .models import VerificationResult


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "evolution.yaml"


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
    ) -> VerificationResult:
        slots = [item for item in evidence_slots if isinstance(item, dict)]
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
        if not slots:
            missing.append("structured evidence claims")
        if coverage < float(self.thresholds.get("claim_coverage", 0.8)):
            unresolved.extend(str(item.get("label") or "unnamed claim") for item in slots if item not in verified)
            if not unresolved:
                unresolved.append("answer claim has no verified evidence")
        if grounding < float(self.thresholds.get("direct_grounding", 0.65)):
            missing.append("direct evidence citation")
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
