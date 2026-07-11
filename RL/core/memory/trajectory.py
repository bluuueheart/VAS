"""Trajectory recording for agent execution steps."""
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from toolkit.base import summarize_tool_result


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc_timestamp(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _compact_observation_for_export(
    observation: Any,
    observation_items: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if isinstance(observation_items, list) and observation_items:
        if len(observation_items) == 1 and isinstance(observation_items[0], dict):
            direct_result = observation_items[0].get("result")
            if isinstance(direct_result, str) and direct_result.strip():
                return direct_result.strip()
        summary = summarize_tool_result(observation_items)
        if summary:
            return summary

    text = str(observation or "").strip()
    if not text:
        return text
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    summary = summarize_tool_result(parsed)
    return summary or text


@dataclass
class Step:
    """Single step in the agent execution trajectory."""
    
    step_id: int
    depth: int
    thought: str
    action: str
    action_args: Dict[str, Any]
    observation: str
    plan: str = ""
    observation_items: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=_utc_now)
    is_recursive: bool = False
    sub_trajectory: Optional['Trajectory'] = None
    sub_trajectories: Optional[List['Trajectory']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "step_id": self.step_id,
            "depth": self.depth,
            "thought": self.thought,
            "plan": self.plan,
            "action": self.action,
            "action_args": self.action_args,
            "observation": _compact_observation_for_export(self.observation, self.observation_items),
            "observation_items": self.observation_items,
            "timestamp": _format_utc_timestamp(self.timestamp),
            "is_recursive": self.is_recursive,
            "sub_trajectory": self.sub_trajectory.to_dict() if self.sub_trajectory else None,
            "sub_trajectories": [
                trajectory.to_dict()
                for trajectory in (self.sub_trajectories or [])
                if trajectory is not None
            ] or None,
        }


@dataclass
class Trajectory:
    """Complete execution trajectory for a task."""
    
    task_id: str
    goal: str
    depth: int
    trajectory_id: str = field(default_factory=lambda: f"trajectory_{uuid.uuid4().hex}")
    registry_snapshot_id: str = field(
        default_factory=lambda: str(os.environ.get("CFV_REGISTRY_SNAPSHOT_ID") or "snapshot_000")
    )
    steps: List[Step] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    debug_events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=_utc_now)
    end_time: Optional[datetime] = None

    def add_debug_event(self, event: Dict[str, Any]) -> None:
        """Append a debugging event for offline troubleshooting."""
        self.debug_events.append(event)
    
    def add_step(
        self,
        thought: str,
        action: str,
        action_args: Dict[str, Any],
        observation: str,
        plan: str = "",
        observation_items: Optional[List[Dict[str, Any]]] = None,
        is_recursive: bool = False,
        sub_trajectory: Optional['Trajectory'] = None,
        sub_trajectories: Optional[List['Trajectory']] = None,
    ) -> Step:
        """Add a new step to the trajectory.
        
        Args:
            thought: Agent's reasoning
            plan: Agent's short plan for current step
            action: Action taken
            action_args: Action arguments
            observation: Result of action
            observation_items: Structured observation list in JSON schema format
            is_recursive: Whether this triggered recursion
            sub_trajectory: Sub-agent trajectory if recursive
            sub_trajectories: Multiple delegated sub-trajectories when one parent
                step fans out into parallel recursive delegations
            
        Returns:
            Created step
        """
        if observation_items is None:
            status = "error" if str(observation).strip().lower().startswith("error") else "success"
            observation_items = [{
                "tool": action,
                "status": status,
                "result": observation,
            }]

        step = Step(
            step_id=len(self.steps) + 1,
            depth=self.depth,
            thought=thought,
            plan=plan,
            action=action,
            action_args=action_args,
            observation=observation,
            observation_items=observation_items,
            is_recursive=is_recursive,
            sub_trajectory=sub_trajectory,
            sub_trajectories=sub_trajectories,
        )
        self.steps.append(step)
        return step
    
    def finish(self, final_answer: str, success: bool = True, error_message: Optional[str] = None):
        """Mark trajectory as finished.
        
        Args:
            final_answer: Final output
            success: Whether task succeeded
            error_message: Error message if failed
        """
        self.final_answer = final_answer
        self.success = success
        self.error_message = error_message
        self.end_time = _utc_now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "registry_snapshot_id": self.registry_snapshot_id,
            "task_id": self.task_id,
            "goal": self.goal,
            "depth": self.depth,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "success": self.success,
            "error_message": self.error_message,
            "debug_events": self.debug_events,
            "start_time": _format_utc_timestamp(self.start_time),
            "end_time": _format_utc_timestamp(self.end_time),
            "started_at": _format_utc_timestamp(self.start_time),
            "finished_at": _format_utc_timestamp(self.end_time),
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }
    
    def get_context_summary(self, max_steps: int = 5) -> str:
        """Get summary of recent steps for context.
        
        Args:
            max_steps: Maximum number of recent steps to include
            
        Returns:
            Formatted summary string
        """
        if not self.steps:
            return "No steps executed yet."
        
        recent_steps = self.steps[-max_steps:]
        summary_lines = [f"Goal: {self.goal}\n"]
        
        for step in recent_steps:
            summary_lines.append(
                f"Step {step.step_id}:\n"
                f"  Thought: {step.thought}\n"
                f"  Plan: {step.plan}\n"
                f"  Action: {step.action}({step.action_args})\n"
                f"  Observation: {step.observation}\n"
            )
        
        return "\n".join(summary_lines)
