"""Typed contracts shared by verification and evolution."""
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    progress_score: float = Field(ge=0.0, le=1.0)
    claim_coverage: float = Field(ge=0.0, le=1.0)
    direct_grounding: float = Field(ge=0.0, le=1.0)
    contradiction_score: float = Field(ge=0.0, le=1.0)
    cross_modal_support: float = Field(ge=0.0, le=1.0)
    can_finish: bool
    missing_information: List[str] = Field(default_factory=list)
    unresolved_claims: List[str] = Field(default_factory=list)
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


class EvolutionAction(str, Enum):
    UPDATE_ARGUMENT_POLICY = "update_argument_policy"
    UPDATE_ROUTER_RULE = "update_router_rule"
    ADD_FALLBACK = "add_fallback"
    ADD_COMPOSITION_EDGE = "add_composition_edge"


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


class RegistrySnapshot(BaseModel):
    snapshot_id: str
    parent_snapshot_id: Optional[str] = None
    created_at: str
    router_policy: Dict[str, Any] = Field(default_factory=dict)
    argument_policy: Dict[str, Any] = Field(default_factory=dict)
    patch_ids: List[str] = Field(default_factory=list)
    status: Literal["candidate", "accepted", "rejected"]
