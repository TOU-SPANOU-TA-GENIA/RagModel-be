from typing import Dict, List, Type
from app.core.interfaces import Tool
from app.config import get_config
from app.tools.library import (
    ReadFileTool, WriteFileTool, ListFilesTool, 
    SearchKnowledgeBaseTool, SystemDiagnosticsTool
)
from app.tools.analysis_tools import AnalyzeDocumentTool
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ToolRegistry:
    """
    Manages the lifecycle of tools.
    Loads enabled tools from config.
    """
    
    _AVAILABLE_TOOLS = {
        "read_file": ReadFileTool,
        "write_file": WriteFileTool,
        "list_files": ListFilesTool,
        "search_knowledge_base": SearchKnowledgeBaseTool,
        "system_diagnostics": SystemDiagnosticsTool,
        "analyze_document": AnalyzeDocumentTool
    }

    @classmethod
    def get_configured_tools(cls) -> Dict[str, Tool]:
        """
        Instantiate tools based on 'tools' section in config.json.
        """
        config = get_config()
        tool_config = config.get("tools", {})
        
        # Check specific 'enabled' flags if they exist, or default to all
        # Example config: {"tools": {"read_file": {"enabled": true}, "write_file": {"enabled": false}}}
        
        active_tools = {}
        
        for name, tool_cls in cls._AVAILABLE_TOOLS.items():
            # Default to enabled if not specified in config
            settings = tool_config.get(name, {"enabled": True})
            
            if settings.get("enabled", True):
                try:
                    # Instantiate tool
                    tool_instance = tool_cls()
                    active_tools[name] = tool_instance
                    logger.debug(f"Tool loaded: {name}")
                except Exception as e:
                    logger.error(f"Failed to load tool {name}: {e}")
        
        return active_tools

    @classmethod
    def get_tool(cls, name: str) -> Tool:
        tools = cls.get_configured_tools()
        return tools.get(name)

# Convenience accessor
def get_tool_registry(config: dict = None) -> Dict[str, Tool]:
    return ToolRegistry.get_configured_tools()