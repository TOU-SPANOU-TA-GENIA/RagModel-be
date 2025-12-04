# app/llm/prompt_templates/builder.py
"""
Config-driven prompt builder - builds prompts using templates from config.
"""

from typing import Dict, List, Any, Optional

from app.core.interfaces import PromptBuilder, Context
from app.llm.prompt_templates.registry import (
    PromptTemplate,
    template_registry,
    get_template_for_model,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfigDrivenPromptBuilder(PromptBuilder):
    """
    Prompt builder that uses templates from config.
    
    Automatically selects template based on active model.
    """
    
    def __init__(
        self,
        system_instruction: str,
        template_name: Optional[str] = None,
        tools: Optional[Dict[str, Any]] = None
    ):
        self.system_instruction = system_instruction
        self.template_name = template_name
        self.tools = tools or {}
        self._template: Optional[PromptTemplate] = None
    
    @property
    def template(self) -> PromptTemplate:
        """Get the prompt template (lazy loaded)."""
        if self._template is None:
            if self.template_name:
                self._template = template_registry.get_or_default(self.template_name)
            else:
                self._template = get_template_for_model()
        return self._template
    
    def build(self, context: Context, **kwargs) -> str:
        """
        Build prompt from context using config template.
        
        Args:
            context: Execution context with query and history
            **kwargs: Additional arguments
                - rag_context: RAG-retrieved context
                - tool_result: Tool execution result
                - custom_system: Override system instruction
        """
        # Build system message
        system = self._build_system_message(context, kwargs)
        
        # Build conversation messages
        messages = self._build_messages(context, kwargs)
        
        # Use template to build final prompt
        prompt = self.template.build_prompt(
            system=system,
            messages=messages,
            include_assistant_start=True
        )
        
        logger.debug(f"Built prompt using template '{self.template.name}', length: {len(prompt)}")
        return prompt
    
    def _build_system_message(self, context: Context, kwargs: Dict) -> str:
        """Build the system message."""
        parts = []
        
        # Base instruction
        system = kwargs.get("custom_system", self.system_instruction)
        parts.append(system)
        
        # Add RAG context if present
        rag_context = kwargs.get("rag_context", "")
        if rag_context:
            parts.append(f"\n\nRelevant context:\n{rag_context}")
        
        # Add tool descriptions if present
        if self.tools:
            tool_desc = self._format_tools()
            if tool_desc:
                parts.append(f"\n\nAvailable tools:\n{tool_desc}")
        
        return "\n".join(parts)
    
    def _build_messages(self, context: Context, kwargs: Dict) -> List[Dict[str, str]]:
        """Build conversation messages."""
        messages = []
        
        # Add chat history
        for msg in context.chat_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Add current query
        user_content = context.query
        
        # Add tool result if present
        tool_result = kwargs.get("tool_result")
        if tool_result:
            user_content = f"{context.query}\n\nTool result:\n{self._format_tool_result(tool_result)}"
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def _format_tools(self) -> str:
        """Format tool descriptions."""
        if not self.tools:
            return ""
        
        lines = []
        for name, tool in self.tools.items():
            desc = getattr(tool, 'description', str(tool))
            lines.append(f"- {name}: {desc}")
        
        return "\n".join(lines)
    
    def _format_tool_result(self, result: Any) -> str:
        """Format tool execution result."""
        if hasattr(result, 'data'):
            return str(result.data)
        return str(result)


def create_prompt_builder_for_model(
    system_instruction: str,
    model_id: Optional[str] = None,
    tools: Optional[Dict[str, Any]] = None
) -> ConfigDrivenPromptBuilder:
    """
    Create a prompt builder configured for a specific model.
    
    Args:
        system_instruction: Base system instruction
        model_id: Model ID (uses active model if None)
        tools: Available tools
    """
    template_name = None
    
    try:
        from app.llm.model_registry import model_registry
        
        if model_id:
            model = model_registry.get_model(model_id)
        else:
            model = model_registry.get_active_model()
        
        if model:
            template_name = model.prompt_template
    except Exception as e:
        logger.debug(f"Could not get model template: {e}")
    
    return ConfigDrivenPromptBuilder(
        system_instruction=system_instruction,
        template_name=template_name,
        tools=tools
    )