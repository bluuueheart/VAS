"""Recursive engine - the brain of the system."""
import ast
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List

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
from evolution import EvidenceVerifier
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
_DISCOVERY_ACTIONS = frozenset({"Tool_Search", "Web_Search"})
_RETRIEVAL_ACTIONS = frozenset({
    "Query_Rewrite", "Semantic_Retrieve", "Keyword_Retrieve", "Segment_Retrieve",
    "Graph_Search", "Clean_Results", "Filtering", "Aggregation", "evidence_verify",
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
    if normalized_status in {"verified", "exact", "approximate", "missing"}:
        if normalized_status in {"verified", "exact"} and (not value or not citation):
            return "approximate"
        return "verified" if normalized_status in {"verified", "exact"} else normalized_status

    if not value and not citation:
        return "missing"
    if not value or not citation:
        return "approximate"
    text = f"{value} {citation}".lower()
    if _APPROXIMATE_EVIDENCE_RE.search(text):
        return "approximate"
    return "verified"


def _extract_evidence_slots(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_slots = payload.get("Evidence")
    if raw_slots is None:
        raw_slots = payload.get("evidence")
    if not isinstance(raw_slots, list):
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
        extracted.append({
            "label": label,
            "value": value,
            "citation": citation,
            "evidence_status": status,
            "logical_conflict": bool(
                item.get("logical conflict", item.get("logical_conflict", False))
            ),
        })
    return extracted


def _merge_evidence_slots(
    existing_slots: List[Dict[str, str]],
    new_slots: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    def _slot_key(slot: Dict[str, str], fallback: str) -> str:
        label = str(slot.get("label", "")).strip().lower()
        citation = str(slot.get("citation", "")).strip().lower()
        return label or citation or fallback

    score = {"missing": 0, "approximate": 1, "verified": 2}
    merged: Dict[str, Dict[str, str]] = {}
    ordered_keys: List[str] = []

    for idx, slot in enumerate(existing_slots):
        key = _slot_key(slot, f"existing_{idx}")
        merged[key] = dict(slot)
        ordered_keys.append(key)

    for idx, slot in enumerate(new_slots):
        key = _slot_key(slot, f"new_{idx}")
        current = merged.get(key)
        if current is None:
            merged[key] = dict(slot)
            ordered_keys.append(key)
            continue
        current_score = score.get(str(current.get("evidence_status", "missing")), 0)
        next_score = score.get(str(slot.get("evidence_status", "missing")), 0)
        if next_score > current_score or (
            next_score == current_score
            and len(str(slot.get("citation", ""))) > len(str(current.get("citation", "")))
        ):
            merged[key] = dict(slot)

    return [merged[key] for key in ordered_keys if key in merged]


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
        
        # Slim system prompt: root planner emits capability requests instead of concrete tool calls.
        # Domain tools are not pre-loaded into the planner prompt; routing happens at runtime.
        # This reduces sp_tokens from ~2800 to ~400, freeing the dynamic context window.
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
                system_prompt = get_system_prompt(query=goal, grounding_context="")
        except Exception as e:
            logger.error(f"Failed to build system prompt: {e}")
            system_prompt = "You are an assistant. Respond in Thought/Action format."
        
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
        evidence_verifier = EvidenceVerifier() if evidence_first_mode else None
        visual_probe_count = 0
        retrieval_probe_count = 0
        direct_visual_probe_count = 0
        cross_modal_verify_count = 0
        modality_attempts: Dict[str, int] = {"text": 0, "visual": 0, "cross_modal": 0}
        modality_successes: Dict[str, int] = {"text": 0, "visual": 0, "cross_modal": 0}
        identical_action_failures: Dict[str, int] = {}
        invalidated_no_result_tools: Dict[str, str] = {}
        last_execution_meta: Dict[str, Any] = {}
        no_progress_guard_signature = ""
        no_progress_guard_family = ""
        no_progress_abort_reason = ""
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
            [{"role": "system", "content": system_prompt}],
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
                return

            logger.info(
                "Tokens ~%d exceed threshold %d after history update. Auto-running Fold_Memory.",
                est_tokens,
                ctx_threshold,
            )
            try:
                fold_result = fold_tool.run(
                    context_window=list(history),
                    keep_recent=keep_recent_count,
                    previous_state=memory_state,
                    discovered_tools=_current_discovered_tool_lines(),
                    blocked_action_signatures=sorted(blocked_action_signatures),
                    llm=current_llm,
                )
            except Exception as exc:
                logger.warning("Fold_Memory auto-execution failed: %s", exc)
                return

            if not isinstance(fold_result, dict) or not fold_result.get("compressed"):
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

            return _from_value(observation_text)

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

        def _append_observation_json(items: List[Dict[str, Any]]) -> None:
            normalized: List[Dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                record = build_planner_observation_record(
                    tool_name=str(item.get("tool", "Unknown")),
                    result=item.get("result", ""),
                    status=str(item.get("status", "success")),
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
                    no_progress_abort_reason = "Error: no_new_evidence_after_probe_budget."
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
                _record_no_valid_result_invalidation(executed_tool, observation_text)
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
                    }

            return {
                "handled": False,
                "tool": action_name,
                "status": "",
                "payload": resolution.get("observation"),
                "observation": str((resolution.get("observation", {}) or {}).get("result", "")),
                "delegate_action_name": action_name,
                "delegate_action_args": action_args if isinstance(action_args, dict) else {},
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

        def _record_no_valid_result_invalidation(tool_name: str, reason: str) -> None:
            normalized_tool = str(tool_name or "").strip()
            if not normalized_tool or normalized_tool in {"Tool_Search", "Finish", "Parallel"}:
                return
            normalized_reason = _truncate_text(str(reason or "").strip(), 220)
            invalidated_no_result_tools[normalized_tool] = normalized_reason
            failure_summary = f"{normalized_tool}: status=no_valid_result"
            if normalized_reason:
                failure_summary = f"{failure_summary}; {normalized_reason}"
            _append_memory_state_message("failed_actions", failure_summary)
            _append_memory_state_message(
                "do_not_repeat",
                f"{normalized_tool}: returned no_valid_result in this branch; switch tools instead of retrying it with new parameters.",
            )
            if evidence_first_mode:
                _append_memory_state_message(
                    "next_allowed_tools",
                    "Use a different retrieval, visual, or verification tool instead of retrying a no_valid_result tool.",
                )
            logger.info(
                "%s[%s][Depth %d] Invalidated tool after no_valid_result: %s reason=%s",
                '  ' * depth,
                role_label,
                depth,
                normalized_tool,
                normalized_reason or "[empty]",
            )

        def _blocked_no_valid_result_message(tool_name: str) -> str:
            normalized_tool = str(tool_name or "").strip()
            if not normalized_tool:
                return ""
            if normalized_tool not in invalidated_no_result_tools:
                return ""
            normalized_reason = str(invalidated_no_result_tools.get(normalized_tool, "") or "").strip()
            blocked_message = (
                f"Error: {normalized_tool} already returned status=no_valid_result in this branch, "
                "so this tool is invalid for the current task. Switch to a different tool instead of retrying it "
                "with adjusted parameters."
            )
            if normalized_reason:
                blocked_message = f"{blocked_message} Last observation: {normalized_reason}"
            return blocked_message.strip()

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

        def _prepare_concrete_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            resolved_args = _resolve_tool_args(args if isinstance(args, dict) else {})
            logger.info(
                "%s[%s][Depth %d] Executing tool: %s",
                '  ' * depth,
                role_label,
                depth,
                _format_action_for_history(name, resolved_args),
            )
            blocked_message = _blocked_no_valid_result_message(name)
            if blocked_message:
                result = {
                    "status": "error",
                    "result": blocked_message,
                    "blocked_tool": str(name or "").strip(),
                    "previous_status": "no_valid_result",
                    "source": "system",
                }
                return {
                    "tool_name": str(name or "").strip(),
                    "resolved_args": resolved_args,
                    "mode": "blocked",
                    "result": result,
                    "source": "system",
                    "executed": False,
                }
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
                _append_memory_state_message(
                    "do_not_repeat",
                    f"{_format_action_for_history(str(name or '').strip(), resolved_args)}: already executed once in this run; do not call it again.",
                )
                return {
                    "tool_name": str(name or "").strip(),
                    "resolved_args": resolved_args,
                    "mode": "duplicate_rejected",
                    "result": result,
                    "source": "system",
                    "executed": False,
                    "execution_signature": execution_signature,
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
                    "resolved_args": resolved_args,
                    "mode": "synthetic_result",
                    "result": hidden_payload,
                    "source": "tool",
                    "executed": True,
                    "execution_signature": execution_signature,
                }
            return {
                "tool_name": str(name or "").strip(),
                "resolved_args": resolved_args,
                "mode": "execute",
                "tool": tool,
                "source": "tool",
                "executed": True,
                "execution_signature": execution_signature,
            }

        def _run_prepared_concrete_tool_call(prepared_call: Dict[str, Any]) -> Any:
            mode = str(prepared_call.get("mode", "")).strip()
            if mode == "execute":
                tool = prepared_call.get("tool")
                if tool is None:
                    raise ValueError("Prepared tool execution is missing tool instance.")
                return tool.run(**prepared_call.get("resolved_args", {}))
            return prepared_call.get("result")

        def _finalize_prepared_concrete_tool_call(
            prepared_call: Dict[str, Any],
            result: Any,
            *,
            abstract_tool: Optional[str] = None,
        ) -> Tuple[Any, str, str]:
            name = str(prepared_call.get("tool_name", "")).strip()
            resolved_args = prepared_call.get("resolved_args", {})
            mode = str(prepared_call.get("mode", "")).strip()
            executed = bool(prepared_call.get("executed", True))
            source = str(prepared_call.get("source", "tool")).strip() or "tool"
            abstract_label = str(abstract_tool or name).strip() or name

            if mode != "blocked":
                _clear_tool_error(name)

            _store_result_aliases(name, result)
            observation_text, observation_status, _ = format_compact_observation(result, tool_name=name)
            if observation_status == "no_valid_result":
                _record_no_valid_result_invalidation(name, observation_text)
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
            _record_progress_for_execution(name, observation_status)

            if mode == "blocked":
                logger.warning(
                    "%s[%s][Depth %d] Blocked retry of invalidated tool after no_valid_result: %s",
                    '  ' * depth,
                    role_label,
                    depth,
                    name,
                )
                _log_observation_preview(f"Tool {name} blocked retry preview", observation_text)
                return result, observation_text, observation_status

            logger.info(
                "%s[%s][Depth %d] Tool %s completed with status=%s",
                '  ' * depth,
                role_label,
                depth,
                name,
                observation_status,
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
                raw_results = tool_search.run(
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
            results = tool_search.run(intent_description=intent, top_k=5, tool_type="all")
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
            results = tool_search.run(intent_description=intent, top_k=5, tool_type="all")
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
                system_prompt=system_prompt,
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
        last_action_signature = ""
        repeated_action_count = 0
        planner_tool_invocation_stats: Dict[str, Dict[str, int]] = {}
        MAX_PARSE_FAILURES = 3  # Abort after N consecutive parse failures
        MAX_REPEATED_ACTIONS = 3

        while iteration < agent.max_iterations and not finished:
            iteration += 1
            logger.info(
                "%s[%s][Depth %d] Iteration %d/%d",
                '  ' * depth,
                role_label,
                depth,
                iteration,
                agent.max_iterations,
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
                    system_prompt=system_prompt,
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
                 
                structured_payload = agent._extract_json_payload(response)
                if evidence_first_mode:
                    parsed_evidence_slots = _extract_evidence_slots(structured_payload)
                    if parsed_evidence_slots:
                        evidence_slots = _merge_evidence_slots(evidence_slots, parsed_evidence_slots)
                        memory_state["evidence_slots"] = list(evidence_slots)

                # Parse response
                thought, plan, action_name, action_args = agent._parse_response(response)
                if depth == 0 and iteration == agent.max_iterations and str(action_name or "").lower() != "finish":
                    forced_final_turn_retry_used = True
                    logger.warning(
                        "%s[%s][Depth %d] Final turn produced non-finish action '%s'. Triggering one finish-only retry.",
                        '  ' * depth,
                        role_label,
                        depth,
                        action_name or "[none]",
                    )
                    forced_history = list(full_history) + [{"role": "assistant", "content": response}]
                    forced_messages = current_llm.format_messages(
                        system_prompt=system_prompt,
                        user_message=(
                            "Final turn retry. Output strict JSON with `Actions: []` and a top-level `Finish` only. "
                            "Do not call tools. Provide the best supported final answer now."
                        ),
                        history=forced_history,
                    )
                    try:
                        forced_raw_response, forced_elapsed_seconds = _generate_with_heartbeat(
                            messages_payload=forced_messages,
                            generation_payload=_generation_kwargs_with_deadlines(generation_kwargs),
                            iteration_idx=iteration,
                            phase="forced_final_retry",
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
                    if evidence_first_mode:
                        parsed_evidence_slots = _extract_evidence_slots(structured_payload)
                        if parsed_evidence_slots:
                            evidence_slots = _merge_evidence_slots(evidence_slots, parsed_evidence_slots)
                            memory_state["evidence_slots"] = list(evidence_slots)
                    thought, plan, action_name, action_args = agent._parse_response(response)
                    if str(action_name or "").lower() != "finish":
                        final_answer = "Error: final_turn_finish_required."
                        success = False
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
                        break
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
                        "parse_success": bool(action_name),
                    })
                    agent.trajectory.add_debug_event(debug_event)

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
                current_signature = _action_signature(action_name, action_args)
                action_family = _action_family(action_name)
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
                    if depth == 0 and repeated_loop_violations >= 2:
                        final_answer = (
                            "Failed: planner kept reusing a blocked action signature. "
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
                    if depth > 0 and signature_failures >= MAX_SUBAGENT_TOOLSEARCH_FAILURES:
                        final_answer = (
                            "Sub-agent fallback to planner: repeated the same tool call without a material "
                            "parameter rewrite three times."
                        )
                        success = False
                        finished = True
                        break
                    if depth == 0 and repeated_loop_violations >= 2:
                        final_answer = (
                            "Failed: planner stuck in repeated-action policy loop. "
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

                    if evidence_verifier is not None:
                        verification = evidence_verifier.verify_answerability(
                            evidence_slots=evidence_slots,
                            steps=agent.trajectory.steps,
                            answer=answer,
                            completion_basis=completion_basis,
                        )
                        verification_payload = verification.model_dump()
                        verification_payload.update({
                            "type": "verifier_decision",
                            "event_type": "VerifierDecisionEvent",
                            "depth": depth,
                            "iteration": iteration,
                        })
                        agent.trajectory.add_debug_event(verification_payload)
                        if not verification.can_finish:
                            agent.trajectory.add_step(
                                thought=thought,
                                plan=plan,
                                action=action_name,
                                action_args=action_args,
                                observation=verification.feedback,
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
                                    "status": str(routed.get("status", "success")),
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
                                "status": mediated.get("status", "success"),
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
                        obs_status = str(routed.get("status", "success"))
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
                    obs_status = str(mediated.get("status", "success"))
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
                        sub_status = str(routed.get("status", "success"))
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
                            _record_no_valid_result_invalidation(
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

                logger.error(f"{'  ' * depth}Error in iteration {iteration}: {e}")
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
        if finished and final_answer:
            normalized_final_answer = str(final_answer or "").strip().lower()
            if normalized_final_answer.startswith(("failed:", "error:")):
                success = False
            return final_answer, success, last_execution_meta
        else:
            return "Error: max_iterations_reached (planner_never_finished).", False, last_execution_meta
    
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
