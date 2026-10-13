"""ReAct agent for single-layer reasoning loop."""
import ast
import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple

from core.llm import BaseLLM
from core.memory import Trajectory
from toolkit.observation_utils import build_parallel_observation, format_compact_observation
from toolkit.registry import registry
from config.settings import settings
from .prompts import get_system_prompt

logger = logging.getLogger(__name__)

_RESPONSE_SCHEMA_KEYS = frozenset((
    "Actions", "actions", "Action", "action",
    "Thought", "thought", "Plan", "plan",
    "Finish", "finish",
    "Result", "result",
))


class ReactAgent:
    """Single-layer ReAct reasoning loop.
    
    Implements the Thought-Plan-Actions-Observation cycle for one recursion level.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        trajectory: Trajectory,
        max_iterations: Optional[int] = None
    ):
        """Initialize ReAct agent.
        
        Args:
            llm: Language model instance
            trajectory: Trajectory to record steps
            max_iterations: Maximum reasoning iterations
        """
        self.llm = llm
        self.trajectory = trajectory
        self.max_iterations = max_iterations or settings.max_iterations
        self.iteration = 0
        self.finished = False
        self.final_answer = None
    
    def run(self, goal: str, context_info: str = "") -> str:
        """Execute the ReAct loop.
        
        Args:
            goal: Goal to achieve
            context_info: Additional context information
            
        Returns:
            Final answer
        """
        # Build slim system prompt: only always-available tools.
        # Domain tools are discovered on demand via Tool_Search.
        system_prompt = get_system_prompt(query=goal, grounding_context=context_info)
        
        # Initialize conversation history
        history: List[Dict[str, str]] = []
        
        # Add context if provided
        if context_info:
            initial_msg = f"Goal: {goal}\n\nContext:\n{context_info}\n\nBegin reasoning:"
        else:
            initial_msg = f"Goal: {goal}\n\nBegin reasoning:"
        
        consecutive_parse_failures = 0
        MAX_PARSE_FAILURES = 3  # Abort after N consecutive parse failures
        
        while self.iteration < self.max_iterations and not self.finished:
            self.iteration += 1
            logger.info(f"ReAct iteration {self.iteration}/{self.max_iterations}")
            
            # Get LLM response
            try:
                # Format messages for this iteration
                if self.iteration == 1:
                    current_message = initial_msg
                else:
                    # Include recent trajectory context
                    recent_context = self.trajectory.get_context_summary(max_steps=3)
                    current_message = f"{recent_context}\n\nContinue reasoning:"
                
                messages = self.llm.format_messages(
                    system_prompt=system_prompt,
                    user_message=current_message,
                    history=history
                )
                
                response = self.sanitize_response(
                    self.llm.generate(
                        messages,
                        max_tokens=int(settings.planner_max_generation_tokens),
                        enable_thinking=bool(settings.planner_enable_thinking),
                    )
                )
                logger.debug(f"LLM Response:\n{response}")
                
                # Parse thought/plan and action
                thought, plan, action, action_args = self._parse_response(response)
                
                if not action:
                    consecutive_parse_failures += 1
                    logger.warning(
                        f"No valid action parsed (failure {consecutive_parse_failures}/{MAX_PARSE_FAILURES})"
                    )
                    if consecutive_parse_failures >= MAX_PARSE_FAILURES:
                        logger.error(
                            "Max consecutive parse failures reached. "
                            "LLM is not producing valid Action format. Aborting."
                        )
                        self.finished = True
                        self.final_answer = (
                            "Failed: LLM could not produce a valid action after "
                            f"{MAX_PARSE_FAILURES} attempts. Check model compatibility "
                            "and prompt format."
                        )
                        break
                    # Provide a concise retry hint without appending the full LLM response
                    history.append({
                        "role": "user",
                        "content": (
                            "Your response did not contain valid Actions. "
                            "Use exactly one strict JSON template:\n\n"
                            "1. Continue with tools:\n"
                            "{\"Thought\":\"...\",\"Plan\":\"...\",\"Actions\":[{\"tool\":\"Python_Executor\",\"params\":{\"code_script\":\"...\"}}]}\n\n"
                            "2. Finish the task:\n"
                            "{\"Thought\":\"...\",\"Plan\":\"...\",\"Actions\":[],\"Finish\":{\"chain_complete\":true,\"completion_basis\":\"text->visual->answer\",\"answer\":\"...\"}}\n\n"
                            "Do not mix non-empty Actions with a valid top-level Finish."
                        )
                    })
                    continue
                
                # Reset parse failure counter on successful parse
                consecutive_parse_failures = 0
                
                # Execute action and get observation
                observation, is_recursive = self._execute_action(action, action_args)
                
                # Record step in trajectory
                self.trajectory.add_step(
                    thought=thought,
                    plan=plan,
                    action=action,
                    action_args=action_args,
                    observation=observation,
                    is_recursive=is_recursive
                )
                
                # Check if this was the Finish action
                if action.lower() == "finish":
                    self.finished = True
                    self.final_answer = observation
                    break
                
                # Update conversation history
                history.append({"role": "assistant", "content": response})
                history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
                
            except Exception as e:
                logger.error(f"Error in ReAct iteration {self.iteration}: {e}")
                observation = f"Error: {str(e)}"
                
                self.trajectory.add_step(
                    thought=thought if 'thought' in locals() else "Error occurred",
                    plan=plan if 'plan' in locals() else "",
                    action=action if 'action' in locals() else "Unknown",
                    action_args=action_args if 'action_args' in locals() else {},
                    observation=observation
                )
                
                # Give agent a chance to recover
                history.append({
                    "role": "user",
                    "content": f"An error occurred: {observation}\n\nPlease try a different approach."
                })
        
        # Return final answer or indicate failure
        if self.finished and self.final_answer:
            return self.final_answer
        else:
            return f"Maximum iterations ({self.max_iterations}) reached without completing the task."

    def sanitize_response(self, response: str) -> str:
        """Normalize line endings without rewriting model semantics."""
        normalized = str(response or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()
    
    def _parse_response(self, response: str) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
        """Parse LLM response to extract thought, plan, action, and arguments.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (thought, plan, action_name, action_args)
        """
        thought = ""
        plan = ""
        action_name = None
        action_args = {}
        
        # Strip <think>...</think> tags (Qwen3-Thinking model output format)
        normalized_response = self.sanitize_response(response)
        cleaned_response = re.sub(
            r'<think>.*?</think>',
            '',
            normalized_response,
            flags=re.DOTALL
        ).strip()
        # Fallback: if stripping removed everything, use original
        if not cleaned_response:
            cleaned_response = normalized_response or response
        
        # Extract thought from plain text format
        thought_match = re.search(
            r'(?:Thought|思考)\s*[:：]\s*(.+?)(?=(?:Action|动作)\s*[:：]|$)',
            cleaned_response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        plan_match = re.search(
            r'(?:Plan|计划)\s*[:：]\s*(.+?)(?=(?:Action|动作)\s*[:：]|$)',
            cleaned_response,
            re.DOTALL | re.IGNORECASE,
        )
        if plan_match:
            plan = plan_match.group(1).strip()

        # Extract thought/action from JSON-like response if present
        json_payload = self._extract_json_payload(cleaned_response)
        if json_payload:
            if self._is_direct_action_payload(json_payload):
                direct_actions = self._parse_actions_payload(json_payload)
                if direct_actions:
                    parsed_name, parsed_args = direct_actions[0]
                    logger.debug(
                        "Parsed direct action payload - Thought: %s | Plan: %s | Action: %s | Args: %s",
                        thought,
                        plan,
                        parsed_name,
                        parsed_args,
                    )
                    return thought, plan, parsed_name, parsed_args
            if not thought:
                thought = str(
                    json_payload.get("Thought")
                    or json_payload.get("thought")
                    or json_payload.get("思考")
                    or ""
                ).strip()
            if not plan:
                plan = str(
                    json_payload.get("Plan")
                    or json_payload.get("plan")
                    or json_payload.get("计划")
                    or ""
                ).strip()
            json_actions = (
                json_payload.get("Actions")
                or json_payload.get("actions")
                or json_payload.get("动作列表")
            )
            parsed_batch = self._parse_actions_payload(json_actions)
            json_finish = json_payload.get("Finish")
            if json_finish is None:
                json_finish = json_payload.get("finish")
            json_result = json_payload.get("Result")
            if json_result in (None, ""):
                json_result = json_payload.get("result")
            has_effective_finish = False
            if isinstance(json_finish, str):
                has_effective_finish = bool(json_finish.strip())
            elif isinstance(json_finish, dict):
                has_effective_finish = any(
                    json_finish.get(key) not in (None, "", [], {})
                    for key in ("answer", "result", "chain_complete", "completion_basis", "reason")
                )
            tool_field_mentions = (
                cleaned_response.count('"tool"')
                + cleaned_response.count('"tool_name"')
                + cleaned_response.count('"name"')
            )
            if (not parsed_batch or len(parsed_batch) == 1) and tool_field_mentions > 1:
                repaired_batch = self._extract_actions_from_json_text(cleaned_response)
                if len(repaired_batch) > len(parsed_batch):
                    parsed_batch = repaired_batch
            if parsed_batch:
                if len(parsed_batch) == 1 and parsed_batch[0][0].lower() == "finish":
                    parsed_name, parsed_args = parsed_batch[0]
                else:
                    parsed_name = "Parallel"
                    parsed_args = {
                        "batch": [{"name": n, "args": a} for n, a in parsed_batch]
                    }
                if has_effective_finish and any(
                    str(name or "").strip().lower() != "finish"
                    for name, _ in parsed_batch
                    if str(name or "").strip()
                ):
                    logger.debug(
                        "Detected mixed top-level Finish plus executable Actions; keeping Actions for parsing and leaving rejection to the runtime protocol validator."
                    )
                logger.debug(
                    "Parsed JSON Actions - Thought: %s | Plan: %s | Action: %s | Args: %s",
                    thought,
                    plan,
                    parsed_name,
                    parsed_args,
                )
                return thought, plan, parsed_name, parsed_args
            finish_args = self._normalize_finish_payload(json_finish)
            if not finish_args and json_result not in (None, "", [], {}):
                finish_args = self._normalize_finish_payload({"result": json_result})
            if finish_args:
                logger.debug(
                    "Parsed JSON Finish/Result field - Thought: %s | Plan: %s | Args: %s",
                    thought,
                    plan,
                    finish_args,
                )
                return thought, plan, "Finish", finish_args
            json_action = (
                json_payload.get("Action")
                or json_payload.get("action")
                or json_payload.get("动作")
            )
            if json_action is not None:
                parsed_name, parsed_args = self._parse_action_text(str(json_action).strip())
                if parsed_name:
                    logger.debug(
                        "Parsed JSON - Thought: %s | Plan: %s | Action: %s | Args: %s",
                        thought,
                        plan,
                        parsed_name,
                        parsed_args,
                    )
                    return thought, plan, parsed_name, parsed_args

        repaired_batch = self._extract_actions_from_json_text(cleaned_response)
        if repaired_batch:
            if len(repaired_batch) == 1 and repaired_batch[0][0].lower() == "finish":
                parsed_name, parsed_args = repaired_batch[0]
            else:
                parsed_name = "Parallel"
                parsed_args = {
                    "batch": [{"name": n, "args": a} for n, a in repaired_batch]
                }
            logger.debug(
                "Recovered Actions from mixed JSON text - Thought: %s | Plan: %s | Action: %s | Args: %s",
                thought,
                plan,
                parsed_name,
                parsed_args,
            )
            return thought, plan, parsed_name, parsed_args

        repaired_finish = self._extract_finish_from_json_text(cleaned_response)
        if repaired_finish:
            logger.debug(
                "Recovered Finish from mixed JSON text - Thought: %s | Plan: %s | Args: %s",
                thought,
                plan,
                repaired_finish,
            )
            return thought, plan, "Finish", repaired_finish

        # Parse only explicit action lines to avoid picking tool-like words from free-form text.
        action_line_pattern = re.compile(
            r'^\s*(?:Action|动作)\s*[:：]\s*(.+)$',
            re.IGNORECASE | re.MULTILINE,
        )
        raw_action_matches = list(action_line_pattern.finditer(cleaned_response))
        if not raw_action_matches:
            logger.debug(f"Parsed - Thought: {thought} | Plan: {plan} | Action: {action_name} | Args: {action_args}")
            return thought, plan, action_name, action_args

        # Heuristic: filter out action lines from prompt-echoed few-shot/template sections.
        # This prevents selecting an Action from copied examples instead of the actual answer.
        example_markers = re.compile(
            r'(few-shot|example\s*\d|user\s*query\s*:|syntax illustration|note\s*:\s*the action line above)',
            re.IGNORECASE,
        )
        filtered_action_lines: List[str] = []
        for match in raw_action_matches:
            context_window = cleaned_response[max(0, match.start() - 320): match.start()]
            if example_markers.search(context_window):
                continue
            filtered_action_lines.append(match.group(1).strip())

        action_lines: List[str]
        if filtered_action_lines:
            action_lines = filtered_action_lines
        else:
            # If multiple Action lines exist but all are in example-like context,
            # treat as parse failure so the engine can request a strict retry.
            if len(raw_action_matches) > 1:
                logger.warning("Detected prompt-echo style response; ignoring example Action lines.")
                return thought, plan, None, {}
            action_lines = [raw_action_matches[0].group(1).strip()]

        # Use the last declared Action line as the authoritative action.
        action_text = action_lines[-1].strip()
        action_name, action_args = self._parse_action_text(action_text)

        logger.debug(f"Parsed - Thought: {thought} | Plan: {plan} | Action: {action_name} | Args: {action_args}")
        return thought, plan, action_name, action_args

    @staticmethod
    def _extract_tool_name_from_action_payload(payload: Dict[str, Any]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        for key in ("tool", "tool_name", "name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if str(payload.get("type", "")).strip().lower() == "tool_call":
            for key in ("tool", "tool_name", "name"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @classmethod
    def _is_direct_action_payload(cls, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        tool_name = cls._extract_tool_name_from_action_payload(payload)
        if not isinstance(tool_name, str) or not tool_name.strip():
            return False
        payload_keys = {str(key) for key in payload.keys()}
        allowed_keys = {
            "tool",
            "tool_name",
            "name",
            "type",
            "description",
            "params",
            "parameters",
            "arguments",
            "args",
        }
        if payload_keys <= allowed_keys:
            return True
        return bool(payload_keys & {"params", "parameters", "arguments", "args"})

    @staticmethod
    def _extract_action_params_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], bool]:
        if not isinstance(payload, dict):
            return {}, [], False
        raw_declared = False
        raw_payload: Any = None
        for key in ("params", "parameters", "arguments", "args"):
            if key in payload:
                raw_declared = True
                raw_payload = payload.get(key)
                break
        if raw_payload is None:
            flat_params = {
                str(key): value
                for key, value in payload.items()
                if str(key) not in {"tool", "tool_name", "name", "type", "description"}
                and str(key) not in {"__raw_params_declared__", "__raw_params_declared", "__raw_param_keys__", "__raw_param_keys"}
                and value not in (None, "")
            }
            if flat_params:
                return flat_params, [str(key) for key in flat_params.keys()], True
            return {}, [], raw_declared
        if not isinstance(raw_payload, dict):
            return {}, [], raw_declared
        normalized = {
            str(key): value
            for key, value in raw_payload.items()
            if str(key) not in {"__raw_params_declared__", "__raw_params_declared", "__raw_param_keys__", "__raw_param_keys"}
        }
        raw_keys = [str(key) for key in normalized.keys() if not str(key).startswith("__")]
        return normalized, raw_keys, raw_declared

    @staticmethod
    def _normalize_raw_param_keys(raw_param_keys: Any) -> List[str]:
        if isinstance(raw_param_keys, str):
            candidates = [part.strip() for part in raw_param_keys.split(",")]
        elif isinstance(raw_param_keys, (list, tuple)):
            candidates = [str(item).strip() for item in raw_param_keys]
        else:
            candidates = []
        deduped: List[str] = []
        for key in candidates:
            if not key or key.startswith("__") or key in deduped:
                continue
            deduped.append(key)
        return deduped

    @classmethod
    def _attach_runtime_param_metadata(
        cls,
        params: Dict[str, Any],
        raw_param_keys: Any,
        raw_declared: bool,
    ) -> Dict[str, Any]:
        normalized = {
            str(key): value
            for key, value in dict(params or {}).items()
            if str(key) not in {"__raw_params_declared__", "__raw_params_declared", "__raw_param_keys__", "__raw_param_keys"}
        }
        if raw_declared:
            normalized["__raw_params_declared__"] = True
            canonical_keys = cls._normalize_raw_param_keys(raw_param_keys or list(normalized.keys()))
            if canonical_keys:
                normalized["__raw_param_keys__"] = canonical_keys
        return normalized

    def _parse_actions_payload(self, payload: Any) -> List[Tuple[str, Dict[str, Any]]]:
        """Parse Actions array payload from JSON schema.

        Expected format:
            [{"tool": "ToolName", "params": {...}}, ...]
        """
        if payload is None:
            return []
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return []
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed_payload = loader(stripped)
                except Exception:
                    continue
                recovered = self._parse_actions_payload(parsed_payload)
                if recovered:
                    return recovered
            parsed_name, parsed_args = self._parse_action_text(stripped)
            if parsed_name:
                return [(parsed_name, parsed_args)]
            return []
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return []
        parsed: List[Tuple[str, Dict[str, Any]]] = []
        for item in payload:
            if isinstance(item, str):
                parsed_name, parsed_args = self._parse_action_text(item.strip())
                if parsed_name:
                    parsed.append((parsed_name, parsed_args))
                continue
            if isinstance(item, list):
                parsed.extend(self._parse_actions_payload(item))
                continue
            if not isinstance(item, dict):
                continue
            tool_name = self._extract_tool_name_from_action_payload(item)
            params, raw_param_keys, raw_declared = self._extract_action_params_payload(item)
            if not isinstance(tool_name, str):
                continue
            # Some planner responses put routing metadata beside ``params``
            # (for example ``{"segment_id": "video_5"}``).  Preserve it so
            # segment-relative timestamps reach the concrete visual tool.
            for routing_key in ("segment_id", "segment_ref", "raw_segment_id", "video", "video_path"):
                if routing_key in item and routing_key not in params and item.get(routing_key) not in (None, ""):
                    params[routing_key] = item.get(routing_key)
                    raw_param_keys.append(routing_key)
                    raw_declared = True
            params = self._attach_runtime_param_metadata(params, raw_param_keys, raw_declared)
            # Preserve planner's description for subagent decomposition
            desc = item.get("description", "")
            if desc and isinstance(desc, str):
                params["__description__"] = desc.strip()
            parsed.append((tool_name.strip(), params))
        return [(n, a) for n, a in parsed if n]

    @staticmethod
    def _parse_dict_like(text: str) -> Optional[Dict[str, Any]]:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_balanced_block(
        text: str,
        start_idx: int,
        open_char: str,
        close_char: str,
    ) -> Tuple[Optional[str], int]:
        if start_idx < 0 or start_idx >= len(text) or text[start_idx] != open_char:
            return None, start_idx
        depth = 0
        in_string = False
        escape = False
        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : idx + 1], idx + 1
        return None, start_idx

    @classmethod
    def _extract_json_payload_block(cls, text: str) -> Optional[Tuple[str, int, int]]:
        """Return the last top-level JSON block relevant to planner parsing."""
        candidates: List[Tuple[str, int, int]] = []
        schema_marked_candidates: List[Tuple[str, int, int]] = []
        depth = 0
        start = -1
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start : idx + 1]
                    parsed = cls._parse_dict_like(candidate)
                    if isinstance(parsed, dict) and (
                        _RESPONSE_SCHEMA_KEYS & parsed.keys() or cls._is_direct_action_payload(parsed)
                    ):
                        candidates.append((candidate, start, idx + 1))
                    elif any(
                        marker in candidate
                        for marker in (
                            '"Actions"',
                            '"actions"',
                            '"Action"',
                            '"action"',
                            '"Finish"',
                            '"finish"',
                            '"Result"',
                            '"result"',
                        )
                    ):
                        schema_marked_candidates.append((candidate, start, idx + 1))
                if depth < 0:
                    depth = 0

        if candidates:
            return candidates[-1]
        if schema_marked_candidates:
            return schema_marked_candidates[-1]
        return None

    @staticmethod
    def _normalize_finish_payload(json_finish: Any) -> Dict[str, Any]:
        finish_args: Dict[str, Any] = {}
        if isinstance(json_finish, str) and json_finish.strip():
            finish_args["answer"] = json_finish.strip()
            return finish_args
        if not isinstance(json_finish, dict):
            return finish_args
        if json_finish.get("answer") not in (None, ""):
            finish_args["answer"] = json_finish.get("answer")
        if json_finish.get("result") not in (None, ""):
            finish_args["result"] = json_finish.get("result")
        if json_finish.get("chain_complete") not in (None, ""):
            finish_args["chain_complete"] = json_finish.get("chain_complete")
        completion_basis = (
            json_finish.get("completion_basis")
            if json_finish.get("completion_basis") not in (None, "")
            else json_finish.get("reason")
        )
        if completion_basis not in (None, ""):
            finish_args["completion_basis"] = completion_basis
        return finish_args

    @staticmethod
    def _mark_non_strict_finish_payload(finish_args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(finish_args, dict) or not finish_args:
            return {}
        marked = dict(finish_args)
        marked["__format_violation__"] = "non_strict_finish_format"
        return marked

    def _extract_actions_from_json_text(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        matches = list(re.finditer(r'"(?:Actions|actions)"\s*:\s*', text))
        if not matches:
            return []

        anchor = matches[-1]
        pos = anchor.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return []

        parsed_payload: Any = None
        actions_block: Optional[str] = None
        if text[pos] == "[":
            actions_block, _ = self._extract_balanced_block(text, pos, "[", "]")
            if actions_block:
                for loader in (json.loads, ast.literal_eval):
                    try:
                        parsed_payload = loader(actions_block)
                        break
                    except Exception:
                        continue
        elif text[pos] == "{":
            actions_block, _ = self._extract_balanced_block(text, pos, "{", "}")
            if actions_block:
                parsed_payload = self._parse_dict_like(actions_block)
        elif text[pos] == '"':
            end_pos = pos + 1
            escape = False
            while end_pos < len(text):
                ch = text[end_pos]
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    break
                end_pos += 1
            if end_pos < len(text) and text[end_pos] == '"':
                try:
                    parsed_payload = json.loads(text[pos : end_pos + 1])
                except Exception:
                    parsed_payload = None

        recovered = self._parse_actions_payload(parsed_payload)
        if recovered:
            return recovered
        actions_source = actions_block if actions_block else text[pos:]
        if not actions_source:
            return []

        tool_re = re.compile(r'"(?:tool|tool_name|name)"\s*:\s*"([^"]+)"')
        params_re = re.compile(r'"(?:params|parameters|arguments|args)"\s*:\s*')
        desc_re = re.compile(r'"description"\s*:\s*"([^"]*)"')

        parsed: List[Tuple[str, Dict[str, Any]]] = []
        pos = 0
        while True:
            tool_match = tool_re.search(actions_source, pos)
            if tool_match is None:
                break

            tool_name = tool_match.group(1).strip()
            search_from = tool_match.end()
            next_tool_match = tool_re.search(actions_source, search_from)
            params_match = params_re.search(actions_source, search_from)
            params: Dict[str, Any] = {}
            next_pos = next_tool_match.start() if next_tool_match is not None else search_from

            if params_match is not None and (next_tool_match is None or params_match.start() < next_tool_match.start()):
                brace_start = actions_source.find("{", params_match.end())
                params_payload, balanced_next_pos = (
                    self._extract_balanced_block(actions_source, brace_start, "{", "}")
                    if brace_start >= 0
                    else (None, params_match.end())
                )
                if params_payload:
                    parsed_params = self._parse_dict_like(params_payload)
                    if isinstance(parsed_params, dict):
                        params = parsed_params
                    raw_param_keys = [str(key) for key in params.keys() if not str(key).startswith("__")]
                    params = self._attach_runtime_param_metadata(params, raw_param_keys, True)
                    next_pos = max(balanced_next_pos, search_from)
                else:
                    partial_end = next_tool_match.start() if next_tool_match is not None else len(actions_source)
                    partial_prefix = actions_source[max(0, brace_start + 1) : partial_end] if brace_start >= 0 else ""
                    partial_params = self._extract_scalar_json_fields_from_prefix(partial_prefix)
                    raw_param_keys = [str(key) for key in partial_params.keys() if not str(key).startswith("__")]
                    params = self._attach_runtime_param_metadata(partial_params, raw_param_keys, True)
                    next_pos = partial_end

            desc_search_end = (
                params_match.start()
                if params_match is not None
                else (next_tool_match.start() if next_tool_match is not None else len(actions_source))
            )
            desc_match = desc_re.search(actions_source, search_from, desc_search_end)
            if desc_match and desc_match.group(1).strip():
                params["__description__"] = desc_match.group(1).strip()

            if tool_name:
                parsed.append((tool_name, params))
            pos = max(next_pos, search_from)

        return [(name, args) for name, args in parsed if name]

    def _extract_finish_from_json_text(self, text: str) -> Dict[str, Any]:
        matches = list(re.finditer(r'"(?:Finish|finish|Result|result)"\s*:\s*', text))
        matches.extend(
            re.finditer(
                r'(?im)(?:^|[\r\n])\s*(?:\*\*\s*)?(?:Finish|finish|Result|result)(?:\s*\*\*)?\s*[:：]\s*',
                text,
            )
        )
        if not matches:
            return {}

        anchor = max(matches, key=lambda match: match.start())
        pos = anchor.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return {}

        if text[pos] == "{":
            finish_payload, _ = self._extract_balanced_block(text, pos, "{", "}")
            parsed = self._parse_dict_like(finish_payload) if finish_payload else None
            return self._mark_non_strict_finish_payload(self._normalize_finish_payload(parsed))

        if text[pos] == '"':
            end_pos = pos + 1
            escape = False
            while end_pos < len(text):
                ch = text[end_pos]
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    break
                end_pos += 1
            if end_pos < len(text) and text[end_pos] == '"':
                try:
                    return self._mark_non_strict_finish_payload(
                        self._normalize_finish_payload(json.loads(text[pos : end_pos + 1]))
                    )
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _extract_scalar_json_fields_from_prefix(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        field_re = re.compile(
            r'"([^"\\]+)"\s*:\s*("(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
            re.DOTALL,
        )
        parsed: Dict[str, Any] = {}
        for match in field_re.finditer(text):
            key = str(match.group(1) or "").strip()
            if not key or key.startswith("__"):
                continue
            raw_value = str(match.group(2) or "").strip()
            try:
                parsed[key] = json.loads(raw_value)
            except Exception:
                continue
        return parsed

    @staticmethod
    def _safe_literal_eval(node: ast.AST) -> Any:
        """Safely convert AST node into Python value when possible."""
        try:
            value = ast.literal_eval(node)
            if value is Ellipsis:
                return "..."
            return value
        except Exception:
            if hasattr(ast, "unparse"):
                return ast.unparse(node)
            return str(node)

    @staticmethod
    def _split_top_level(text: str, sep: str) -> List[str]:
        """Split string by separator while respecting nested brackets and quoted strings."""
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        in_quote: Optional[str] = None
        escape = False

        for ch in text:
            if in_quote:
                current.append(ch)
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == in_quote:
                    in_quote = None
                continue

            if ch in ('"', "'"):
                in_quote = ch
                current.append(ch)
                continue

            if ch in "([{":
                depth += 1
            elif ch in ")]}" and depth > 0:
                depth -= 1

            if ch == sep and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
            else:
                current.append(ch)

        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def _parse_function_call(self, call_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Parse function call text like ToolName(a=1, b='x')."""
        text = call_text.strip()
        if not text:
            return None, {}

        # Allow bare tool name: Finish
        bare = re.fullmatch(r"[A-Za-z_]\w*", text)
        if bare:
            return bare.group(0), {}

        try:
            expr = ast.parse(text, mode="eval")
        except SyntaxError:
            # Fallback regex for partially malformed outputs
            m = re.match(r"(\w+)\s*\((.*)\)\s*$", text, re.DOTALL)
            if not m:
                return None, {}
            name = m.group(1)
            args_str = m.group(2).strip()
            if not args_str:
                return name, {}
            parsed_args: Dict[str, Any] = {}
            for piece in self._split_top_level(args_str, ","):
                kv = piece.split("=", 1)
                if len(kv) != 2:
                    continue
                key = kv[0].strip()
                val = kv[1].strip()
                if key:
                    parsed_args[key] = val.strip('"\'')
            return name, parsed_args

        node = expr.body
        if not isinstance(node, ast.Call):
            return None, {}
        if not isinstance(node.func, ast.Name):
            return None, {}

        action_name = node.func.id
        action_args: Dict[str, Any] = {}

        for idx, arg in enumerate(node.args):
            action_args[f"arg{idx}"] = self._safe_literal_eval(arg)
        for kw in node.keywords:
            if kw.arg:
                action_args[kw.arg] = self._safe_literal_eval(kw.value)

        return action_name, action_args

    def _parse_action_text(self, action_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Parse action text supporting single call and Parallel batch."""
        text = action_text.strip()
        if not text:
            return None, {}

        # Action payload itself may be JSON object (tool/name + args)
        if text.startswith("{") and text.endswith("}"):
            payload = self._parse_dict_like(text)
            if isinstance(payload, dict):
                action_name = self._extract_tool_name_from_action_payload(payload)
                action_args, raw_param_keys, raw_declared = self._extract_action_params_payload(payload)
                if isinstance(action_name, str) and isinstance(action_args, dict):
                    action_args = self._attach_runtime_param_metadata(action_args, raw_param_keys, raw_declared)
                    desc = payload.get("description", "")
                    if isinstance(desc, str) and desc.strip():
                        action_args["__description__"] = desc.strip()
                    return action_name, action_args

        parallel_match = re.match(r"Parallel\s*\[(.*)\]\s*$", text, re.IGNORECASE | re.DOTALL)
        if parallel_match:
            calls = self._split_top_level(parallel_match.group(1), ";")
            batch = []
            for call in calls:
                name, args = self._parse_function_call(call)
                if name:
                    batch.append({"name": name, "args": args})
            if batch:
                return "Parallel", {"batch": batch}
            return None, {}

        return self._parse_function_call(text)

    @classmethod
    def _extract_json_payload(cls, text: str) -> Optional[Dict[str, Any]]:
        """Extract the last valid JSON/dict payload from text.

        Strategy: collect ALL brace-balanced candidate substrings by scanning
        for top-level ``{...}`` blocks with depth tracking, then check
        candidates from **last to first** - the model's final output is
        always at the end, so preferring the last match avoids picking up
        echoed prompt fragments or ``<think>`` remnants.
        """
        ACTION_KEYS = _RESPONSE_SCHEMA_KEYS

        # Phase 1: extract from markdown code blocks (last preference)
        code_blocks = re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)

        # Phase 2: brace-depth-aware extraction of all top-level {}
        brace_candidates: list[str] = []
        depth = 0
        start = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    brace_candidates.append(text[start : i + 1])
                if depth < 0:
                    depth = 0

        # Build ordered candidates: code-block extracts first, then brace
        # matches.  We will iterate in **reverse** so the last candidate wins.
        all_candidates = [c.strip() for c in code_blocks if c.strip()]
        all_candidates.extend(brace_candidates)

        def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
            """Try json.loads then ast.literal_eval; return dict or None."""
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(candidate)
                    if isinstance(parsed, dict) and (
                        ACTION_KEYS & parsed.keys() or cls._is_direct_action_payload(parsed)
                    ):
                        return parsed
                except Exception:
                    continue
            return None

        # Iterate last-to-first so the model's actual answer takes priority
        for candidate in reversed(all_candidates):
            result = _try_parse(candidate)
            if result is not None:
                return result

        return None
    
    def _summarize_result(self, result: Dict[str, Any], max_length: int = 4000) -> str:
        """Summarize a dict result to fit within max_length while preserving key information.
        
        Args:
            result: Result dictionary to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized string representation
        """
        import json
        
        summary_parts = []
        
        # Extract scalar values (numbers, strings, booleans)
        for key, value in result.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                summary_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                # Recursively summarize nested dicts
                nested_summary = self._summarize_result(value, max_length // 2)
                summary_parts.append(f"{key}: {{{nested_summary}}}")
            elif isinstance(value, list):
                # Show list length and first few items
                summary_parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        
        return ", ".join(summary_parts)
    
    def _execute_action(self, action_name: str, action_args: Dict[str, Any]) -> Tuple[str, bool]:
        """Execute an action using the tool registry.
        
        Args:
            action_name: Name of the tool/action
            action_args: Arguments for the action
            
        Returns:
            Tuple of (observation_string, is_recursive)
        """
        if action_name == "Finish":
            answer = action_args.get("answer")
            if answer in (None, ""):
                answer = action_args.get("result", "")
            return str(answer or ""), False

        # Parallel batch execution
        if action_name == "Parallel":
            batch = action_args.get("batch", [])
            observations = []
            any_recursive = False
            for call in batch:
                name = call.get("name")
                args = call.get("args", {})
                tool = registry.get_tool(name)
                is_recursive = tool.is_virtual
                if is_recursive:
                    any_recursive = True
                    logger.info(f"[Parallel] '{name}' is high-level (sub-agent will handle).")
                    observations.append({
                        "tool": name,
                        "status": "error",
                        "payload": {
                            "status": "error",
                            "result": f"Tool '{name}' is abstract and requires concrete tool resolution.",
                            "reason": "abstract_tool_requires_resolution",
                        },
                    })
                else:
                    try:
                        result = tool.execute(**args)
                        obs, _, _ = format_compact_observation(result, tool_name=name)
                        observations.append({
                            "tool": name,
                            "status": str(result.get("status", "no_valid_result")),
                            "payload": result,
                        })
                    except Exception as e:
                        error_msg = (
                            f"Error executing {name}: {str(e)}. "
                            "Try another relevant tool or adjust the time range / query."
                        )
                        logger.error(error_msg)
                        observations.append({"tool": name, "status": "error", "payload": {"error": error_msg}})
            observation, _, _ = build_parallel_observation(observations)
            return observation, any_recursive
        
        # Get tool from registry (returns PlaceholderTool if not found)
        tool = registry.get_tool(action_name)
        
        is_recursive = tool.is_virtual
        
        if is_recursive:
            # This will be handled by the engine
            logger.info(f"Action '{action_name}' is a high-level abstraction (will trigger recursion)")
            return json.dumps({
                "status": "error",
                "result": f"Tool '{action_name}' is abstract and requires concrete tool resolution.",
                "reason": "abstract_tool_requires_resolution",
            }, ensure_ascii=False), True
        
        # Execute the tool
        try:
            result = tool.execute(**action_args)
            
            # Convert result to string for observation
            observation, _, _ = format_compact_observation(result, tool_name=action_name)
            
            logger.info("Executed %s: %s", action_name, observation)
            return observation, False
            
        except Exception as e:
            error_msg = (
                f"Error executing {action_name}: {str(e)}. "
                "Try another relevant tool or adjust the time range / query."
            )
            logger.error(error_msg)
            return error_msg, False
