# app/llm/greek_prompt_builder.py
"""
Greek Prompt Builder - Integrates localization with thinking management.

Builds prompts that:
1. Use Greek system instructions
2. Add thinking tag rules
3. Handle RAG context
4. Support tool results
"""

from typing import Dict, Any, Optional

from app.core.interfaces import PromptBuilder, Context
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GreekPromptBuilder(PromptBuilder):
    """
    Prompt builder optimized for Greek language and thinking management.
    """
    
    def __init__(
        self, 
        system_instruction: Optional[str] = None,
        tools: Optional[Dict[str, Any]] = None,
        enable_thinking_tags: bool = True
    ):
        """
        Initialize builder.
        
        Args:
            system_instruction: Custom system instruction (uses Greek default if None)
            tools: Available tools dict
            enable_thinking_tags: Whether to add thinking tag rules
        """
        from app.localization.greek import get_greek_system_prompt
        
        self.system_instruction = system_instruction or get_greek_system_prompt()
        self.tools = tools or {}
        self.enable_thinking_tags = enable_thinking_tags
    
    def build(self, context: Context, **kwargs) -> str:
        """
        Build complete prompt with Greek instructions and thinking rules.
        """
        sections = []
        
        # System instruction with thinking rules
        sections.append(self._build_system_section())
        
        # Conversation history
        history = context.chat_history
        if history:
            sections.append(self._build_history_section(history))
        
        # RAG context
        rag_context = kwargs.get("rag_context", "")
        if rag_context:
            sections.append(self._build_rag_section(rag_context))
        
        # Tool results
        tool_result = kwargs.get("tool_result")
        if tool_result:
            sections.append(self._build_tool_section(tool_result))
        
        # Current query
        sections.append(self._build_query_section(context.query))
        
        # Response instruction
        sections.append(self._build_response_instruction())
        
        return "\n\n".join(sections)
    
    def _build_system_section(self) -> str:
        """Build system section with optional thinking rules."""
        system = f"<|system|>\n{self.system_instruction}"
        
        if self.enable_thinking_tags:
            thinking_rules = """
<σκέψη>
Αν χρειαστεί να σκεφτείς πριν απαντήσεις, χρησιμοποίησε <think> tags:

<think>
[Η εσωτερική σου ανάλυση εδώ - δεν θα εμφανιστεί στον χρήστη]
</think>

[Η πραγματική απάντησή σου]

ΚΑΝΟΝΕΣ:
- Η σκέψη στο <think> block είναι για εσωτερική χρήση
- ΠΟΤΕ μην συμπεριλαμβάνεις τη σκέψη στην απάντηση
- Απάντα φυσικά και άμεσα
- Αν η απάντηση είναι απλή, μην χρειάζεται thinking
</σκέψη>
"""
            system += f"\n{thinking_rules}"
        
        system += "\n<|end|>"
        return system
    
    def _build_history_section(self, history: list) -> str:
        """Build conversation history section."""
        formatted = []
        
        for msg in history[-10:]:  # Keep last 10 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                formatted.append(f"<|user|>\n{content}\n<|end|>")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}\n<|end|>")
        
        return "\n".join(formatted)
    
    def _build_rag_section(self, rag_context: str) -> str:
        """Build RAG context section."""
        return f"""<context>
{rag_context}
</context>

Χρησιμοποίησε τις παραπάνω πληροφορίες για να απαντήσεις."""
    
    def _build_tool_section(self, tool_result: Dict[str, Any]) -> str:
        """Build tool result section."""
        if tool_result.get("success"):
            data = tool_result.get("data", {})
            
            # Handle file content
            if "content" in data:
                return f"""<αποτέλεσμα_εργαλείου>
Αρχείο: {data.get('file_name', 'unknown')}
Τοποθεσία: {data.get('path', 'unknown')}
Μέγεθος: {data.get('size_bytes', 0)} bytes

Περιεχόμενο:
{data['content']}
</αποτέλεσμα_εργαλείου>

Χρησιμοποίησε αυτές τις πληροφορίες για να απαντήσεις στον χρήστη."""
            
            return f"<αποτέλεσμα_εργαλείου>\n{data}\n</αποτέλεσμα_εργαλείου>"
        else:
            error = tool_result.get("error", "Άγνωστο σφάλμα")
            return f"""<αποτέλεσμα_εργαλείου>
Αποτυχία: {error}
</αποτέλεσμα_εργαλείου>

Ενημέρωσε τον χρήστη για το σφάλμα."""
    
    def _build_query_section(self, query: str) -> str:
        """Build user query section."""
        return f"<|user|>\n{query}\n<|end|>"
    
    def _build_response_instruction(self) -> str:
        """Build response instruction."""
        return "<|assistant|>"


class GreekStreamingPromptBuilder(GreekPromptBuilder):
    """
    Extended builder for streaming with explicit thinking separation.
    """
    
    def build(self, context: Context, **kwargs) -> str:
        """Build prompt optimized for streaming."""
        # Use parent build
        base_prompt = super().build(context, **kwargs)
        
        # Add streaming-specific instruction
        streaming_note = """
Ξεκίνα να γράφεις την απάντησή σου αμέσως. Μην περιμένεις."""
        
        return f"{base_prompt}\n{streaming_note}"


def create_greek_prompt_builder(
    custom_instruction: Optional[str] = None,
    tools: Optional[Dict] = None,
    streaming: bool = False,
    enable_thinking: bool = True
) -> PromptBuilder:
    """
    Factory function for Greek prompt builders.
    
    Args:
        custom_instruction: Custom system instruction
        tools: Available tools
        streaming: Whether to use streaming-optimized builder
        enable_thinking: Whether to enable thinking tags
    """
    if streaming:
        return GreekStreamingPromptBuilder(
            system_instruction=custom_instruction,
            tools=tools,
            enable_thinking_tags=enable_thinking
        )
    
    return GreekPromptBuilder(
        system_instruction=custom_instruction,
        tools=tools,
        enable_thinking_tags=enable_thinking
    )