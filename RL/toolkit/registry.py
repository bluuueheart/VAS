"""Tool registry for managing and discovering tools."""
import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type

from config.settings import settings
from .base import BaseTool, PlaceholderTool, normalize_tool_tags

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Singleton registry for managing all available tools.
    
    Implements auto-discovery of tools and smart dispatch logic.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize registry."""
        if self._initialized:
            return
        
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._missing_tool_warnings: set[str] = set()
        self._initialized = True
        logger.info("ToolRegistry initialized")

    def _resolve_tool_tags(
        self,
        name: str,
        *,
        tool: Optional[BaseTool] = None,
        tool_tags: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        if tool_tags is not None:
            return normalize_tool_tags(tool_tags)
        if tool is not None:
            return normalize_tool_tags(getattr(tool, "tags", None) or [])
        cached_tool = self._tools.get(name)
        if cached_tool is not None:
            return normalize_tool_tags(getattr(cached_tool, "tags", None) or [])
        return None

    def _is_tool_enabled(
        self,
        name: str,
        *,
        tool: Optional[BaseTool] = None,
        tool_tags: Optional[List[str]] = None,
    ) -> bool:
        resolved_tags = self._resolve_tool_tags(name, tool=tool, tool_tags=tool_tags)
        return settings.is_tool_enabled(name, tool_tags=resolved_tags)
    
    def register(self, tool: BaseTool):
        """Register a tool instance.
        
        Args:
            tool: Tool instance to register
        """
        if not self._is_tool_enabled(tool.name, tool=tool):
            self._tools.pop(tool.name, None)
            self._tool_classes.pop(tool.name, None)
            self._missing_tool_warnings.discard(tool.name)
            logger.info(f"Skipping registration for disabled tool: {tool.name}")
            return

        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        
        self._tools[tool.name] = tool
        self._missing_tool_warnings.discard(tool.name)
        logger.debug(f"Registered tool: {tool.name}")
    
    def register_class(self, tool_class: Type[BaseTool]):
        """Register a tool class (will be instantiated on demand).
        
        Args:
            tool_class: Tool class to register
        """
        tool_name = str(getattr(tool_class, "name", "") or "").strip()
        if not tool_name:
            # Fallback for tool classes that only define the name at instance level.
            tool_name = tool_class().name

        instance = tool_class()
        if not self._is_tool_enabled(tool_name, tool=instance):
            self._tools.pop(tool_name, None)
            self._tool_classes.pop(tool_name, None)
            self._missing_tool_warnings.discard(tool_name)
            logger.info(f"Skipping registration for disabled tool class: {tool_name}")
            return

        self._tool_classes[instance.name] = tool_class
        self._tools[instance.name] = instance
        self._missing_tool_warnings.discard(instance.name)
        logger.debug(f"Registered tool class: {instance.name}")
    
    def get_tool(self, name: str, *, log_missing: bool = True) -> BaseTool:
        """Get tool by name with smart dispatch logic.
        
        This is the key method that implements the recursive expansion:
        - If tool exists: return the actual tool instance
        - If tool doesn't exist: return PlaceholderTool to trigger recursion
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or PlaceholderTool
        """
        if not self._is_tool_enabled(name):
            if log_missing and name not in self._missing_tool_warnings:
                logger.info(f"Tool '{name}' is disabled by runtime settings. Returning PlaceholderTool.")
                self._missing_tool_warnings.add(name)
            return PlaceholderTool(name=name)

        if name in self._tools:
            return self._tools[name]
        
        if log_missing and name not in self._missing_tool_warnings:
            logger.warning(f"Tool '{name}' not found. Returning PlaceholderTool for recursive expansion.")
            self._missing_tool_warnings.add(name)
        return PlaceholderTool(name=name)
    
    def has_tool(self, name: str) -> bool:
        """Check if a real (non-placeholder) tool exists.
        
        Args:
            name: Tool name
            
        Returns:
            True if real tool exists
        """
        return (
            self._is_tool_enabled(name)
            and name in self._tools
            and not self._tools[name].is_virtual
            and not self._is_runtime_hidden_tool(name)
        )

    @staticmethod
    def _is_runtime_hidden_tool(name: str) -> bool:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            return False
        try:
            from toolkit.impl.retrieval import get_sparse_scope_hidden_tool_payload

            return get_sparse_scope_hidden_tool_payload(normalized_name) is not None
        except Exception:
            return False

    def get_all_tools(self, tags: Optional[List[str]] = None, include_untagged: bool = False) -> List[BaseTool]:
        """Get list of all registered tools, optionally filtered by tags.
        
        Args:
            tags: optional list of tags; tool is kept if it has ANY of the tags
            include_untagged: when True, also keep tools with empty/None tags
        
        Returns:
            List of tool instances
        """
        tools = [
            tool for tool in self._tools.values()
            if not tool.is_virtual
            and self._is_tool_enabled(tool.name, tool=tool)
            and not self._is_runtime_hidden_tool(tool.name)
        ]
        if tags:
            tagset = set(normalize_tool_tags(tags))
            filtered = []
            for t in tools:
                t_tags = set(normalize_tool_tags(t.tags or []))
                if tagset.intersection(t_tags):
                    filtered.append(t)
                elif include_untagged and not t_tags:
                    filtered.append(t)
            tools = filtered
        return tools
    
    def get_tool_schemas(self, tags: Optional[List[str]] = None) -> List[Dict]:
        """Get schemas for tools (for LLM context), optionally filtered by tags."""
        return [tool.get_schema() for tool in self.get_all_tools(tags=tags)]
    
    def auto_discover(self, impl_dir: Optional[Path] = None):
        """Auto-discover and load tools from impl directory.
        
        Args:
            impl_dir: Directory containing tool implementations
        """
        if impl_dir is None:
            # Default to toolkit/impl/
            impl_dir = Path(__file__).parent / "impl"
        
        if not impl_dir.exists():
            logger.warning(f"Tool implementation directory not found: {impl_dir}")
            return
        
        logger.info(f"Auto-discovering tools from: {impl_dir}")
        
        # Find all Python files in impl directory
        for py_file in impl_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            module_name = f"toolkit.impl.{py_file.stem}"
            
            try:
                module = importlib.import_module(module_name)
                
                # Find all BaseTool subclasses in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseTool) and 
                        obj is not BaseTool and 
                        obj is not PlaceholderTool):
                        self.register_class(obj)
                        
            except Exception as e:
                logger.error(f"Failed to load tools from {module_name}: {e}")
    
    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        self._tool_classes.clear()
        self._missing_tool_warnings.clear()
        logger.info("Tool registry cleared")
    
    def get_tools_description(self, tags: Optional[List[str]] = None) -> str:
        """Get formatted description of all available tools.
        
        Returns:
            Formatted string with tool descriptions
        """
        tools = self.get_all_tools(tags=tags)
        if not tools:
            return "No tools available."
        
        lines = ["Available Tools:"]
        for tool in tools:
            lines.append(f"\n{tool.name}:")
            lines.append(f"  Description: {tool.description}")
            if tool.parameters.get("properties"):
                lines.append("  Parameters:")
                for param_name, param_info in tool.parameters["properties"].items():
                    required = param_name in tool.parameters.get("required", [])
                    req_str = " (required)" if required else ""
                    lines.append(f"    - {param_name}{req_str}: {param_info.get('description', '')}")
        
        return "\n".join(lines)


# Global registry instance
registry = ToolRegistry()
