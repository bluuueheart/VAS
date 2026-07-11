"""Base class for all tools in the toolkit."""
from abc import ABC, abstractmethod
import json
from typing import Any, Dict, Iterable, Optional
from pydantic import BaseModel, ConfigDict, Field


_PREFERRED_TEXT_KEYS = (
    "result",
    "answer",
    "summary",
    "compressed_summary",
    "explanation",
    "findings",
    "caption",
    "extracted_text",
    "description",
    "context",
    "snippet",
    "normalized",
    "expanded",
    "enhanced_details",
    "error",
)


def _tool_retry_guidance(tool_name: str = "") -> str:
    label = tool_name or "This tool"
    return (
        f"{label} did not provide usable evidence. "
        "Planner should try another relevant tool, broaden/narrow the time range, or reformulate the probe."
    )


def _is_effectively_empty_result(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        return not result.strip()
    if isinstance(result, list):
        if not result:
            return True
        meaningful_items = 0
        for item in result[:5]:
            if isinstance(item, dict) and item.get("error"):
                continue
            if summarize_tool_result(item):
                meaningful_items += 1
        return meaningful_items == 0
    if isinstance(result, dict):
        if result.get("error"):
            return False
        for key in ("result", "answer", "summary", "compressed_summary", "explanation", "findings", "caption", "extracted_text", "description"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return False
        if isinstance(result.get("items"), list) and result["items"]:
            return False
        if isinstance(result.get("matched_segments"), list) and result["matched_segments"]:
            return False
        if isinstance(result.get("paths"), list) and result["paths"]:
            return False
        if isinstance(result.get("sub_queries"), list) and result["sub_queries"]:
            return False
        return True
    return False


def summarize_tool_result(result: Any, tool_name: str = "") -> str:
    """Extract a concise human-readable summary from an arbitrary tool result."""
    if result is None:
        return _tool_retry_guidance(tool_name)
    if isinstance(result, str):
        text = result.strip()
        return text or _tool_retry_guidance(tool_name)
    if isinstance(result, (int, float, bool)):
        return str(result)
    if isinstance(result, dict):
        if result.get("error"):
            return (
                f"{tool_name or 'Tool'} failed: {str(result['error']).strip()}. "
                "Planner should try another relevant tool or adjust the probe."
            )
        for key in _PREFERRED_TEXT_KEYS:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        primitives = []
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                text_value = str(value).strip()
                if len(text_value) > 160:
                    text_value = text_value[:160] + "..."
                primitives.append(f"{key}={text_value}")
            if len(primitives) >= 4:
                break
        if primitives:
            return "; ".join(primitives)
        return _tool_retry_guidance(tool_name)

    if isinstance(result, list):
        if not result:
            return _tool_retry_guidance(tool_name)

        item_summaries = []
        for item in result[:3]:
            item_summary = summarize_tool_result(item, tool_name=tool_name)
            if item_summary and item_summary not in item_summaries:
                item_summaries.append(item_summary)
        if item_summaries:
            prefix = f"{len(result)} results"
            if tool_name:
                prefix = f"{tool_name}: {prefix}"
            return f"{prefix}. " + " | ".join(item_summaries)
        return _tool_retry_guidance(tool_name)

    return str(result).strip()


def enrich_tool_result(result: Any, tool_name: str = "") -> Any:
    """Inject a textual `result` field into dict results when missing."""
    if isinstance(result, dict):
        if isinstance(result.get("result"), str) and result["result"].strip():
            return result
        summary = summarize_tool_result(result, tool_name=tool_name)
        if not summary:
            return result
        enriched = dict(result)
        enriched["result"] = summary
        return enriched
    return result


def format_tool_result_observation(result: Any, tool_name: str = "") -> str:
    """Render a tool result as observation text with a guaranteed readable summary."""
    enriched = enrich_tool_result(result, tool_name=tool_name)
    if _is_effectively_empty_result(enriched):
        payload = {
            "result": _tool_retry_guidance(tool_name),
            "status": "no_valid_result",
            "raw_result": enriched,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if isinstance(enriched, str):
        return enriched
    if isinstance(enriched, list):
        summary = summarize_tool_result(enriched, tool_name=tool_name)
        if summary:
            payload = {"result": summary}
            serialized_length = len(json.dumps(enriched, ensure_ascii=False))
            if len(enriched) <= 3 and serialized_length <= 2000:
                payload["items"] = enriched
            else:
                preview_items = []
                for item in enriched[:3]:
                    preview_items.append(summarize_tool_result(item, tool_name=tool_name))
                payload["item_count"] = len(enriched)
                payload["items_preview"] = preview_items
            return json.dumps(payload, indent=2, ensure_ascii=False)
        return json.dumps(enriched, indent=2, ensure_ascii=False)
    return json.dumps(enriched, indent=2, ensure_ascii=False)


def normalize_tool_tags(tags: Optional[Iterable[Any]]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags or []:
        text = str(tag or "").strip().lower()
        if not text:
            continue
        if text == "biz":
            text = "base"
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def infer_primary_tool_type(tags: Optional[Iterable[Any]]) -> str:
    normalized = set(normalize_tool_tags(tags))
    if "meta" in normalized:
        return "meta"
    return "base"


class BaseTool(BaseModel, ABC):
    """Abstract base class for all tools.
    
    All tools must inherit from this class and implement the run method.
    """
    
    name: str = Field(..., description="Unique name of the tool")
    description: str = Field(..., description="Human-readable description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the tool's parameters"
    )
    tags: Optional[list] = Field(
        default_factory=list,
        description="List of tags for categorization/analytics (e.g., ['aggregation', 'count'])"
    )
    is_virtual: bool = Field(
        default=False,
        description="Whether this is a placeholder for a high-level abstraction"
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.tags = normalize_tool_tags(self.tags)
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM function calling.
        
        Returns:
            Dictionary with tool name, description, and parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def validate_args(self, **kwargs) -> bool:
        """Validate arguments against parameter schema.
        
        Args:
            **kwargs: Arguments to validate
            
        Returns:
            True if valid
        """
        # Basic validation - can be extended with jsonschema
        required_params = self.parameters.get("required", [])
        
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        
        return True


class PlaceholderTool(BaseTool):
    """Placeholder tool for high-level abstractions that trigger recursion."""
    
    name: str
    description: str = "High-level abstraction to be implemented by sub-agent"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    is_virtual: bool = True
    
    def run(self, **kwargs) -> str:
        """Placeholder execution - should trigger recursion.
        
        Args:
            **kwargs: Arguments passed to the virtual tool
            
        Returns:
            Placeholder message
        """
        return f"PlaceholderTool '{self.name}' called with args: {kwargs}. This should trigger recursion."


class FallbackTool(BaseTool):
    """Concrete fallback tool to avoid virtual recursion when a tool is missing."""

    name: str
    description: str = Field(
        default="Fallback tool returned when a requested tool is not implemented.",
        description="Fallback tool description"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: Optional[list] = Field(default_factory=lambda: ["fallback"])
    is_virtual: bool = False

    def run(self, **kwargs) -> str:
        """Return a deterministic message indicating the tool is unavailable."""
        return (
            f"Tool '{self.name}' is not registered. "
            f"Received args: {kwargs}. Please use available tools or implement this tool."
        )
