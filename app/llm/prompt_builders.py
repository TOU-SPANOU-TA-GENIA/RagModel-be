# app/llm/prompt_builders.py
"""
Prompt building implementations.
"""

from typing import Dict, Any, List, Optional

from app.core.interfaces import PromptBuilder, Context
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SimplePromptBuilder(PromptBuilder):
    """Simple prompt builder creating structured prompts."""
    
    def __init__(self, system_instruction: str = "You are a helpful AI assistant."):
        self.system_instruction = system_instruction
    
    def build(self, context: Context, **kwargs) -> str:
        sections = []
        
        sections.append(f"<system>\n{self.system_instruction}\n</system>")
        
        if context.chat_history:
            history_text = self._format_history(context.chat_history)
            sections.append(f"<history>\n{history_text}\n</history>")
        
        rag_context = kwargs.get("rag_context", "")
        if rag_context:
            sections.append(f"<context>\n{rag_context}\n</context>")
        
        tool_result = kwargs.get("tool_result")
        if tool_result:
            tool_text = self._format_tool_result(tool_result)
            sections.append(f"<tool_result>\n{tool_text}\n</tool_result>")
        
        sections.append(f"<query>\n{context.query}\n</query>")
        sections.append("\nAssistant:")
        
        return "\n\n".join(sections)
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        recent = history[-10:] if len(history) > 10 else history
        lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        if tool_result.get("success"):
            data = tool_result.get("data", {})
            if isinstance(data, dict) and "content" in data:
                return f"Tool execution successful:\n{data['content']}"
            return f"Tool execution successful:\n{data}"
        return f"Tool execution failed: {tool_result.get('error', 'Unknown error')}"


class ToolAwarePromptBuilder(SimplePromptBuilder):
    """Prompt builder that includes tool descriptions."""
    
    def __init__(self, system_instruction: str, tools: Dict[str, Any]):
        super().__init__(system_instruction)
        self.tools = tools
    
    def build(self, context: Context, **kwargs) -> str:
        base_prompt = super().build(context, **kwargs)
        
        intent = context.metadata.get("intent")
        if intent and intent.value == "action" and self.tools:
            tool_section = self._build_tool_section()
            base_prompt = base_prompt.replace(
                "<system>",
                f"<system>\n{self.system_instruction}\n\n{tool_section}"
            )
        
        return base_prompt
    
    def _build_tool_section(self) -> str:
        lines = ["Available tools:"]
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)