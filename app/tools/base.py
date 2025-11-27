# app/tools/base.py
"""
Tool system - re-exports from split modules.
"""

from pathlib import Path

from app.tools.models import (
    ToolResult,
    BaseTool,
    SimpleToolRegistry,
    ToolChain
)
from app.tools.file_tools import (
    ReadFileTool,
    WriteFileTool,
    ListFilesTool
)
from app.tools.system_tools import (
    ExecuteCommandTool,
    DatabaseQueryTool
)
from app.config import KNOWLEDGE_DIR, DATA_DIR, BASE_DIR
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "ToolResult",
    "BaseTool",
    "SimpleToolRegistry",
    "ToolChain",
    "ReadFileTool",
    "WriteFileTool",
    "ListFilesTool",
    "ExecuteCommandTool",
    "DatabaseQueryTool",
    "create_default_tools",
    "create_tool_registry_for_military"
]


def create_default_tools() -> SimpleToolRegistry:
    """Create registry with default tools."""
    registry = SimpleToolRegistry()
    
    try:
        from app.tools.enhanced_file_tools import create_enhanced_file_tools
        
        search_dirs = [KNOWLEDGE_DIR, DATA_DIR, BASE_DIR / "logs"]
        enhanced_tools = create_enhanced_file_tools(search_dirs)
        
        for tool in enhanced_tools.values():
            registry.register(tool)
        
        logger.info("Registered enhanced file tools")
    except ImportError as e:
        logger.warning(f"Enhanced tools unavailable: {e}, using basic")
        registry.register(ReadFileTool(allowed_dirs=[KNOWLEDGE_DIR, DATA_DIR]))
    
    registry.register(WriteFileTool())
    registry.register(ListFilesTool())
    registry.register(ExecuteCommandTool())
    
    try:
        from app.tools.document_generator import DocumentGeneratorTool
        registry.register(DocumentGeneratorTool())
        logger.info("Registered document generator")
    except ImportError as e:
        logger.warning(f"Document generator unavailable: {e}")
    
    logger.info(f"Created tool registry with {len(registry.tools)} tools")
    return registry


def create_tool_registry_for_military() -> SimpleToolRegistry:
    """Create registry for military environment."""
    registry = SimpleToolRegistry()
    
    military_dirs = [KNOWLEDGE_DIR, DATA_DIR, BASE_DIR / "logs"]
    
    try:
        from app.tools.enhanced_file_tools import create_enhanced_file_tools
        enhanced_tools = create_enhanced_file_tools(military_dirs)
        for tool in enhanced_tools.values():
            registry.register(tool)
    except ImportError:
        registry.register(ReadFileTool(allowed_dirs=military_dirs))
    
    registry.register(WriteFileTool(allowed_dirs=[Path("/data/outputs")]))
    
    safe_commands = ["ls", "pwd", "date", "whoami", "df", "free"]
    registry.register(ExecuteCommandTool(allowed_commands=safe_commands))
    
    logger.info("Created military tool registry")
    return registry