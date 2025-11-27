# app/llm/thinking_step.py
"""
Thinking Step - Generates internal reasoning before final response.
The thinking is stored separately and used to guide response generation,
but never exposed to the user.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from app.core.interfaces import PipelineStep, Context, LLMProvider
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ThinkingResult:
    """Container for thinking output."""
    reasoning: str
    response_strategy: str
    tone: str
    key_points: list


class ThinkingStep(PipelineStep):
    """
    Pipeline step that generates internal reasoning before final response.
    
    This step:
    1. Analyzes the user's query
    2. Generates structured thinking about how to respond
    3. Stores thinking in context.metadata for use by LLM generation
    4. Never exposes thinking to the user
    """
    
    def __init__(self, llm_provider: LLMProvider, enabled: bool = True):
        self.llm = llm_provider
        self.enabled = enabled
        self._thinking_prompt_template = self._build_thinking_template()
    
    @property
    def name(self) -> str:
        return "Thinking Generation"
    
    def _build_thinking_template(self) -> str:
        """Build the prompt template for generating thinking."""
        return """<thinking_task>
Analyze the user's message and generate internal reasoning about how to respond.
This thinking will guide your response but will NOT be shown to the user.

User message: {query}

Context available:
- Has RAG context: {has_rag}
- Has tool result: {has_tool}
- Intent classified as: {intent}
- Conversation history length: {history_length}

Generate your thinking in this format:

ANALYSIS:
- What is the user asking for?
- What type of response do they expect?
- Any implicit needs or context?

STRATEGY:
- How should I structure my response?
- What information is most relevant?
- What tone is appropriate?

KEY_POINTS:
- List 2-3 main points to address

TONE:
- One word describing appropriate tone (e.g., professional, friendly, technical)

Output ONLY the thinking, no final response.
</thinking_task>"""
    
    def process(self, context: Context) -> Context:
        """Generate thinking and store in context metadata."""
        if not self.enabled:
            logger.debug("Thinking step disabled, skipping")
            return context
        
        try:
            thinking = self._generate_thinking(context)
            context.metadata["thinking"] = thinking
            context.metadata["thinking_enabled"] = True
            
            logger.info(f"Generated thinking: {len(thinking)} chars")
            logger.debug(f"Thinking preview: {thinking[:200]}...")
            
        except Exception as e:
            logger.warning(f"Thinking generation failed: {e}, continuing without")
            context.metadata["thinking"] = None
            context.metadata["thinking_enabled"] = False
        
        return context
    
    def _generate_thinking(self, context: Context) -> str:
        """Generate internal reasoning for the query."""
        # Gather context info
        has_rag = bool(context.metadata.get("rag_context"))
        has_tool = bool(context.metadata.get("tool_result"))
        intent = context.metadata.get("intent", "unknown")
        if hasattr(intent, "value"):
            intent = intent.value
        
        history = context.chat_history or []
        
        # Build thinking prompt
        thinking_prompt = self._thinking_prompt_template.format(
            query=context.query,
            has_rag=has_rag,
            has_tool=has_tool,
            intent=intent,
            history_length=len(history)
        )
        
        # Generate thinking (with lower token limit for efficiency)
        thinking = self.llm.generate(
            thinking_prompt,
            max_tokens=150,  # Keep thinking concise
            temperature=0.3  # More focused/deterministic
        )
        
        return thinking.strip()
    
    def parse_thinking(self, thinking_text: str) -> ThinkingResult:
        """Parse thinking text into structured result."""
        # Default values
        result = ThinkingResult(
            reasoning=thinking_text,
            response_strategy="direct",
            tone="professional",
            key_points=[]
        )
        
        # Try to extract structured sections
        lines = thinking_text.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("ANALYSIS:"):
                current_section = "analysis"
            elif line.startswith("STRATEGY:"):
                current_section = "strategy"
            elif line.startswith("KEY_POINTS:"):
                current_section = "key_points"
            elif line.startswith("TONE:"):
                current_section = "tone"
            elif line.startswith("- ") and current_section == "key_points":
                result.key_points.append(line[2:])
            elif current_section == "tone" and line:
                result.tone = line.lower()
            elif current_section == "strategy" and line.startswith("- "):
                result.response_strategy = line[2:]
        
        return result


class ThinkingAwarePromptBuilder:
    """
    Wrapper that enhances prompts with thinking context.
    Use this to wrap your existing prompt builder.
    """
    
    def __init__(self, base_builder):
        self.base_builder = base_builder
    
    def build(self, context: Context, **kwargs) -> str:
        """Build prompt, incorporating thinking if available."""
        # Get base prompt
        base_prompt = self.base_builder.build(context, **kwargs)
        
        # Check for thinking
        thinking = context.metadata.get("thinking")
        if not thinking:
            return base_prompt
        
        # Inject thinking guidance into prompt
        thinking_guidance = self._format_thinking_guidance(thinking)
        
        # Insert thinking before the final "Assistant:" marker
        if "\nAssistant:" in base_prompt:
            parts = base_prompt.rsplit("\nAssistant:", 1)
            enhanced_prompt = f"{parts[0]}\n\n{thinking_guidance}\nAssistant:"
            if len(parts) > 1:
                enhanced_prompt += parts[1]
        else:
            enhanced_prompt = f"{base_prompt}\n\n{thinking_guidance}"
        
        return enhanced_prompt
    
    def _format_thinking_guidance(self, thinking: str) -> str:
        """Format thinking as internal guidance."""
        return f"""<internal_reasoning>
{thinking}

Use this analysis to guide your response. Do NOT include this thinking in your answer.
Respond naturally and directly to the user's message.
</internal_reasoning>"""


# Factory function for easy integration
def create_thinking_pipeline_step(
    llm_provider: LLMProvider,
    enabled: bool = True
) -> ThinkingStep:
    """Create a thinking step for the pipeline."""
    return ThinkingStep(llm_provider, enabled)


def wrap_with_thinking(base_builder) -> ThinkingAwarePromptBuilder:
    """Wrap an existing prompt builder with thinking awareness."""
    return ThinkingAwarePromptBuilder(base_builder)