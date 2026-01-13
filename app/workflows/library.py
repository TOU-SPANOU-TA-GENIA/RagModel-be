from typing import Dict, Any, Optional
import re
from app.tools import get_tool_registry
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class StepExecutor:
    """
    Executes a single step by looking up the tool and resolving parameters.
    """
    
    def __init__(self):
        self.tools = get_tool_registry()

    def execute(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Any:
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry.")

        # Resolve templated parameters (e.g., "{{ filename }}")
        resolved_params = self._resolve_params(params, context)
        
        logger.debug(f"Executing tool {tool_name} with params: {resolved_params}")
        result = tool.execute(**resolved_params)
        
        if not result.success:
            raise RuntimeError(f"Tool execution failed: {result.error}")
            
        return result.output

    def _resolve_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replaces {{ key }} in string values with values from context.
        """
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and "{{" in v:
                # Simple regex replacer for {{ key }}
                # In production, use Jinja2 for robustness
                def replace_match(match):
                    key = match.group(1).strip()
                    return str(context.get(key, match.group(0)))
                
                resolved[k] = re.sub(r'\{\{(.*?)\}\}', replace_match, v)
            else:
                resolved[k] = v
        return resolved