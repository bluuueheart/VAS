"""Context manager for variable namespace and recursion stack."""
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from config.settings import settings

logger = logging.getLogger(__name__)


# Observation filtering utilities


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into dot-separated key-value pairs."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            flat[key] = f"[{len(v)} items]" if len(v) > 3 else v
        else:
            flat[key] = v
    return flat


def _filter_value(value: Any, depth: int = 0) -> Any:
    """Recursively filter a parsed JSON value.

    - Lists longer than 10 -> sample [0, 1, mid, -2, -1].
    - Dicts at depth >= 3 -> flatten to key-value pairs.
    """
    if isinstance(value, list):
        if len(value) > 10:
            n = len(value)
            indices = sorted({0, 1, n // 2, n - 2, n - 1} & set(range(n)))
            sampled = [_filter_value(value[i], depth + 1) for i in indices]
            sampled.append(f"... ({n} items total, sampled {len(indices)})")
            return sampled
        return [_filter_value(v, depth + 1) for v in value]
    if isinstance(value, dict):
        if depth >= 3:
            return _flatten_dict(value)
        return {k: _filter_value(v, depth + 1) for k, v in value.items()}
    return value


def filter_observation(observation: str) -> str:
    """Filter an observation string to reduce token footprint.

    Rules (from PRD TODO-3):
      - list length > 10 -> keep indices [0, 1, len//2, -2, -1]
      - JSON nesting > 3 levels -> flatten to key-value pairs
    Non-JSON text is returned unchanged.
    """
    try:
        data = json.loads(observation)
    except (json.JSONDecodeError, TypeError, ValueError):
        return observation
    filtered = _filter_value(data, depth=0)
    return json.dumps(filtered, ensure_ascii=False, indent=2)


class MaxRecursionError(Exception):
    """Raised when maximum recursion depth is exceeded."""
    pass


@dataclass
class ScopeFrame:
    """Represents a single scope in the call stack."""
    
    depth: int
    goal: str
    local_vars: Dict[str, Any] = field(default_factory=dict)
    parent_frame: Optional['ScopeFrame'] = None
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Get variable from current or parent scope.
        
        Args:
            name: Variable name
            
        Returns:
            Variable value or None if not found
        """
        # Check local scope first
        if name in self.local_vars:
            return self.local_vars[name]

        if self.depth > 0:
            prefixed_name = f"depth{self.depth}_{name}"
            if prefixed_name in self.local_vars:
                return self.local_vars[prefixed_name]
        
        # Check parent scopes
        if self.parent_frame:
            return self.parent_frame.get_variable(name)
        
        return None
    
    def set_variable(self, name: str, value: Any, scope_prefix: bool = True):
        """Set variable in current scope.
        
        Args:
            name: Variable name
            value: Variable value
            scope_prefix: Whether to add depth prefix to avoid pollution
        """
        if scope_prefix and self.depth > 0:
            prefixed_name = f"depth{self.depth}_{name}"
            self.local_vars[prefixed_name] = value
            logger.debug(f"Set scoped variable: {prefixed_name} = {value}")
        else:
            self.local_vars[name] = value
            logger.debug(f"Set variable: {name} = {value}")


class ContextManager:
    """Manages variable namespace and recursion stack for the agent system."""
    
    def __init__(self, max_depth: Optional[int] = None):
        """Initialize context manager.
        
        Args:
            max_depth: Maximum recursion depth (defaults to settings)
        """
        self.max_depth = max_depth or settings.max_recursion_depth
        self.global_vars: Dict[str, Any] = {}
        self.call_stack: List[ScopeFrame] = []
        self.current_frame: Optional[ScopeFrame] = None
    
    @property
    def current_depth(self) -> int:
        """Get current recursion depth."""
        return len(self.call_stack)
    
    def push_frame(self, goal: str) -> ScopeFrame:
        """Push a new scope frame onto the call stack.
        
        Args:
            goal: Goal for this scope
            
        Returns:
            New scope frame
            
        Raises:
            MaxRecursionError: If max depth exceeded
        """
        if self.current_depth >= self.max_depth:
            raise MaxRecursionError(
                f"Maximum recursion depth {self.max_depth} exceeded. "
                f"Current depth: {self.current_depth}"
            )
        
        new_frame = ScopeFrame(
            depth=self.current_depth,
            goal=goal,
            parent_frame=self.current_frame
        )
        
        self.call_stack.append(new_frame)
        self.current_frame = new_frame
        
        logger.info(f"Pushed frame at depth {new_frame.depth}: {goal}")
        return new_frame
    
    def pop_frame(self) -> Optional[ScopeFrame]:
        """Pop the current scope frame from the call stack.
        
        Returns:
            Popped frame or None if stack is empty
        """
        if not self.call_stack:
            logger.warning("Attempted to pop from empty call stack")
            return None
        
        popped_frame = self.call_stack.pop()
        self.current_frame = self.call_stack[-1] if self.call_stack else None
        
        logger.info(f"Popped frame at depth {popped_frame.depth}")
        return popped_frame
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Get variable value from current scope or global.
        
        Args:
            name: Variable name
            
        Returns:
            Variable value or None if not found
        """
        # Check current frame and parents
        if self.current_frame:
            value = self.current_frame.get_variable(name)
            if value is not None:
                return value
        
        # Check global variables
        return self.global_vars.get(name)
    
    def set_variable(self, name: str, value: Any, global_scope: bool = False):
        """Set variable in current or global scope.
        
        Args:
            name: Variable name
            value: Variable value
            global_scope: Whether to set in global scope
        """
        if global_scope:
            self.global_vars[name] = value
            logger.debug(f"Set global variable: {name} = {value}")
        elif self.current_frame:
            self.current_frame.set_variable(name, value)
        else:
            # No frame, default to global
            self.global_vars[name] = value
            logger.debug(f"Set global variable (no frame): {name} = {value}")
    
    def get_all_variables(self) -> Dict[str, Any]:
        """Get all accessible variables in current scope.
        
        Returns:
            Dictionary of all accessible variables
        """
        all_vars = dict(self.global_vars)
        
        if self.current_frame:
            frames: List[ScopeFrame] = []
            frame = self.current_frame
            while frame:
                frames.append(frame)
                frame = frame.parent_frame

            for frame in reversed(frames):
                for key, value in frame.local_vars.items():
                    all_vars[key] = value
                    prefix = f"depth{frame.depth}_"
                    if key.startswith(prefix):
                        all_vars[key[len(prefix):]] = value
        
        return all_vars
    
    def get_context_info(self) -> str:
        """Get human-readable context information.
        
        Returns:
            Formatted context string
        """
        lines = [
            f"Current Depth: {self.current_depth}/{self.max_depth}",
            f"Global Variables: {len(self.global_vars)}",
        ]
        
        if self.current_frame:
            lines.append(f"Current Goal: {self.current_frame.goal}")
            lines.append(f"Local Variables: {len(self.current_frame.local_vars)}")
        
        if len(self.call_stack) > 1:
            lines.append("\nCall Stack:")
            for i, frame in enumerate(self.call_stack):
                lines.append(f"  [{i}] Depth {frame.depth}: {frame.goal}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset context manager to initial state."""
        self.global_vars.clear()
        self.call_stack.clear()
        self.current_frame = None
        logger.info("Context manager reset")
