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
from app.config import DATA_DIR, BASE_DIR
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
    "create_restricted_tool_registry"
]

def create_default_tools() -> "SimpleToolRegistry":
    """Create registry with default tools."""
    from app.tools.models import SimpleToolRegistry
    from app.utils.logger import setup_logger
    from pathlib import Path
    
    logger = setup_logger(__name__)
    
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    registry = SimpleToolRegistry()
    
    # =========================================================================
    # ENHANCED FILE TOOLS
    # =========================================================================
    try:
        from app.tools.enhanced_file_tools import create_enhanced_file_tools
        
        search_dirs = [DATA_DIR, BASE_DIR / "logs"]
        enhanced_tools = create_enhanced_file_tools(search_dirs)
        
        for tool in enhanced_tools.values():
            registry.register(tool)
        
        logger.info("Registered enhanced file tools")
    except ImportError as e:
        logger.warning(f"Enhanced tools unavailable: {e}")
    
    # =========================================================================
    # BASIC FILE TOOLS (with safe imports)
    # =========================================================================
    try:
        from app.tools.file_tools import WriteFileTool
        registry.register(WriteFileTool())
        logger.info("Registered WriteFileTool")
    except (ImportError, AttributeError) as e:
        logger.warning(f"WriteFileTool unavailable: {e}")
    
    try:
        from app.tools.file_tools import ListFilesTool
        registry.register(ListFilesTool())
        logger.info("Registered ListFilesTool")
    except (ImportError, AttributeError) as e:
        logger.warning(f"ListFilesTool unavailable: {e}")
    
    try:
        from app.tools.file_tools import ExecuteCommandTool
        registry.register(ExecuteCommandTool())
        logger.info("Registered ExecuteCommandTool")
    except (ImportError, AttributeError) as e:
        logger.warning(f"ExecuteCommandTool unavailable: {e}")
    
    # =========================================================================
    # DOCUMENT GENERATOR
    # =========================================================================
    try:
        from app.tools.document_generator import DocumentGeneratorTool
        registry.register(DocumentGeneratorTool())
        logger.info("Registered document generator")
    except ImportError as e:
        logger.warning(f"Document generator unavailable: {e}")
    
    # =========================================================================
    # INTELLIGENCE ANALYSIS TOOLS
    # =========================================================================
    try:
        from app.tools.intelligence_report_tool import IntelligenceReportTool, BatchDocumentAnalysisTool
        registry.register(IntelligenceReportTool())
        registry.register(BatchDocumentAnalysisTool())
        logger.info("Registered intelligence analysis tools")
    except ImportError as e:
        logger.warning(f"Intelligence tools unavailable: {e}")
    
    # =========================================================================
    # LOGISTICS ANALYSIS TOOLS
    # =========================================================================
    try:
        from app.tools.logistics_anomaly_tool import register_logistics_tools
        register_logistics_tools(registry)
        logger.info("Registered logistics analysis tools")
    except ImportError as e:
        logger.warning(f"Logistics tools unavailable: {e}")
    
    # =========================================================================
    # FILE SERVER TOOL
    # =========================================================================
    try:
        from app.tools.file_server_tool import FileServerTool
        from app.services.file_server import create_file_server
        from app.config.file_server_config import get_settings
        
        settings = get_settings()
        file_server = create_file_server(
            mount_point=settings.mount_point,
            base_path=settings.smb_path,
            aliases=settings.folder_aliases
        )
        registry.register(FileServerTool(file_server))
        logger.info(f"Registered file server tool (mount: {settings.mount_point})")
    except ImportError as e:
        logger.warning(f"File server tool unavailable: {e}")
    except Exception as e:
        logger.warning(f"File server connection failed: {e}")
    
    logger.info(f"Created tool registry with {len(registry.tools)} tools")
    return registry

def create_restricted_tool_registry() -> SimpleToolRegistry:
    """Create registry for military environment."""
    registry = SimpleToolRegistry()
    
    restricted_dirs = [DATA_DIR, BASE_DIR / "logs"]
    
    try:
        from app.tools.enhanced_file_tools import create_enhanced_file_tools
        enhanced_tools = create_enhanced_file_tools(restricted_dirs)
        for tool in enhanced_tools.values():
            registry.register(tool)
    except ImportError:
        registry.register(ReadFileTool(allowed_dirs=restricted_dirs))
    
    registry.register(WriteFileTool(allowed_dirs=[Path("/data/outputs")]))
    
    safe_commands = ["ls", "pwd", "date", "whoami", "df", "free"]
    registry.register(ExecuteCommandTool(allowed_commands=safe_commands))
    
    # Intelligence tools are essential for military use
    try:
        from app.tools.intelligence_report_tool import IntelligenceReportTool, BatchDocumentAnalysisTool
        registry.register(IntelligenceReportTool())
        registry.register(BatchDocumentAnalysisTool())
        logger.info("Registered intelligence analysis tools (military)")
    except ImportError as e:
        logger.warning(f"Intelligence tools unavailable: {e}")
    
    logger.info("Created military tool registry")
    return registry